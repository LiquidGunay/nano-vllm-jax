"""Scheduler for continuous batching."""

from collections import deque
from typing import Deque, List, Tuple

import jax.numpy as jnp

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus, SamplingParams
from nanovllm_jax.engine.block_manager import BlockManager


class Scheduler:
    """Scheduler for continuous batching.
    
    Manages:
    - Waiting queue (sequences waiting to start)
    - Running queue (sequences being generated)
    - Block allocation via BlockManager
    - Preemption (swapping out sequences)
    """

    def __init__(self, config: Qwen3_5Config):
        self.max_num_seqs = getattr(config, 'max_num_seqs', 16)
        self.max_num_batched_tokens = getattr(config, 'max_num_batched_tokens', 2048)
        self.eos = getattr(config, 'eos', None)
        self.enable_prefix_cache_execution = not getattr(config, "linear_attn_layers", ())
        self.prefill_buckets = tuple(getattr(config, "prefill_buckets", ()))
        self.batch_size_buckets = tuple(getattr(config, "batch_size_buckets", ()))
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, 
            config.block_size
        )
        # Override sequence block size
        Sequence.block_size = config.block_size
        
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """Check if all sequences are done."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Add a sequence to the waiting queue."""
        if self.max_blocks_per_seq is not None:
            max_tokens_per_seq = self.max_blocks_per_seq * seq.block_size
            requested_tokens = seq.num_tokens + seq.max_tokens
            if seq.num_blocks > self.max_blocks_per_seq:
                raise ValueError(
                    f"prompt needs {seq.num_blocks} blocks but max_blocks_per_seq is {self.max_blocks_per_seq}"
                )
            if requested_tokens > max_tokens_per_seq:
                raise ValueError(
                    f"request needs {requested_tokens} total tokens but per-sequence capacity is {max_tokens_per_seq}"
                )
        if len(seq) > self.max_num_batched_tokens:
            raise ValueError(
                f"prompt has {len(seq)} tokens but max_num_batched_tokens is {self.max_num_batched_tokens}"
            )
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], ScheduledBatch]:
        """Schedule sequences for execution.
        
        Returns:
            Tuple of (scheduled sequences, scheduled batch)
        """
        scheduled_seqs: List[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # Phase 1: Prefill - schedule waiting sequences
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # Check constraints
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(seq):
                break
            
            # Allocate and schedule
            self.block_manager.allocate(seq)
            if not self.enable_prefix_cache_execution:
                seq.num_cached_tokens = 0
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            uncached_tokens = len(seq) - seq.num_cached_tokens
            if uncached_tokens == 0:
                continue

            num_seqs += 1
            num_batched_tokens += uncached_tokens
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, self.build_scheduled_batch(scheduled_seqs, is_prefill=True)
        
        # Phase 2: Decode - schedule running sequences
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Ensure we can append
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt a running sequence
                    self.preempt(self.running.pop())
                else:
                    # Must preempt current sequence
                    self.preempt(seq)
                    break
            else:
                # Can append - schedule for decode
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        if not scheduled_seqs:
            raise RuntimeError("No sequence can be scheduled; KV cache capacity is exhausted")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, self.build_scheduled_batch(scheduled_seqs, is_prefill=False)

    def build_scheduled_batch(
        self,
        seqs: List[Sequence],
        *,
        is_prefill: bool,
        query_len_bucket: int | None = None,
        batch_size_bucket: int | None = None,
        max_blocks_per_seq: int | None = None,
    ) -> ScheduledBatch:
        """Build the canonical engine batch contract for one step."""
        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        if max_blocks_per_seq is None:
            max_blocks_per_seq = self.max_blocks_per_seq
        if max_blocks_per_seq is not None:
            if max_blocks > max_blocks_per_seq:
                raise ValueError(f"scheduled block table needs {max_blocks} blocks but bucket has {max_blocks_per_seq}")
            max_blocks = max_blocks_per_seq
        for seq in seqs:
            if is_prefill:
                start = seq.num_cached_tokens
                tokens = seq.token_ids[start:]
                positions = list(range(start, seq.num_tokens))
            else:
                tokens = [seq.last_token]
                positions = [seq.num_tokens - 1]

            if not tokens:
                raise ValueError(f"Scheduled sequence {seq.seq_id} has no executable tokens")

            query_tokens.append(tokens)
            query_positions.append(positions)
            block_tables.append(seq.block_table + [0] * (max_blocks - len(seq.block_table)))
            seq_lens.append(seq.num_tokens)
            query_lens.append(len(tokens))

        max_query_len = max(query_lens)
        if query_len_bucket is None and is_prefill and self.prefill_buckets:
            query_len_bucket = self._select_bucket(max_query_len, self.prefill_buckets, "prefill")
        if query_len_bucket is None:
            query_len_bucket = max_query_len
        if max_query_len > query_len_bucket:
            raise ValueError(f"scheduled query needs {max_query_len} tokens but bucket has {query_len_bucket}")

        if batch_size_bucket is None and self.batch_size_buckets:
            batch_size_bucket = self._select_bucket(len(seqs), self.batch_size_buckets, "batch")
        if batch_size_bucket is None:
            batch_size_bucket = len(seqs)
        if len(seqs) > batch_size_bucket:
            raise ValueError(f"scheduled batch has {len(seqs)} seqs but bucket has {batch_size_bucket}")

        padded_tokens = [tokens + [0] * (query_len_bucket - len(tokens)) for tokens in query_tokens]
        padded_positions = [positions + [0] * (query_len_bucket - len(positions)) for positions in query_positions]
        query_start_loc = [0]
        for qlen in query_lens:
            query_start_loc.append(query_start_loc[-1] + qlen)
        for _ in range(batch_size_bucket - len(seqs)):
            padded_tokens.append([0] * query_len_bucket)
            padded_positions.append([0] * query_len_bucket)
            block_tables.append([0] * max_blocks)
            seq_lens.append(0)
            query_lens.append(0)
            query_start_loc.append(query_start_loc[-1])

        return ScheduledBatch(
            tokens=jnp.array(padded_tokens, dtype=jnp.int32),
            positions=jnp.array(padded_positions, dtype=jnp.int32),
            seq_ids=jnp.array([seq.seq_id for seq in seqs] + [-1] * (batch_size_bucket - len(seqs)), dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else len(seqs),
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
        )

    @staticmethod
    def _select_bucket(size: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in sorted(buckets):
            if size <= bucket:
                return bucket
        raise ValueError(f"{name} size {size} exceeds configured buckets {buckets}")

    def preempt(self, seq: Sequence):
        """Preempt a sequence (move back to waiting)."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, 
        seqs: List[Sequence], 
        token_ids: List[int | List[int]]
    ) -> List[bool]:
        """Post-process after generation step.
        
        Args:
            seqs: Sequences that were scheduled
            token_ids: Generated token IDs
            
        Returns:
            List of is_finished flags for each sequence
        """
        finished_flags = []
        self.last_num_generated_tokens = 0
        
        for seq, generated in zip(seqs, token_ids):
            generated_tokens = generated if isinstance(generated, list) else [generated]
            finished = False

            for idx, token_id in enumerate(generated_tokens):
                seq.append_token(int(token_id))
                self.last_num_generated_tokens += 1

                if idx < len(generated_tokens) - 1:
                    self.block_manager.commit_processed_token(seq)

                # Check termination conditions
                is_eos = (token_id == self.eos)
                is_max_tokens = (seq.num_completion_tokens == seq.max_tokens)
                
                if (not seq.ignore_eos and is_eos) or is_max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    finished = True
                    break

            finished_flags.append(finished)
        
        return finished_flags
