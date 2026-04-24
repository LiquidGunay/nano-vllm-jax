"""Scheduler for continuous batching."""

from collections import deque
from typing import List, Tuple, Deque

from nanovllm_jax.config import Qwen3_5Config
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
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """Schedule sequences for execution.
        
        Returns:
            Tuple of (scheduled sequences, is_prefill)
            - is_prefill=True if we're scheduling new sequences
            - is_prefill=False if we're continuing running sequences
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
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # is_prefill=True
        
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
        
        assert scheduled_seqs, "Should have at least one scheduled sequence"
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # is_prefill=False

    def preempt(self, seq: Sequence):
        """Preempt a sequence (move back to waiting)."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, 
        seqs: List[Sequence], 
        token_ids: List[int]
    ) -> List[bool]:
        """Post-process after generation step.
        
        Args:
            seqs: Sequences that were scheduled
            token_ids: Generated token IDs
            
        Returns:
            List of is_finished flags for each sequence
        """
        finished_flags = []
        
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # Check termination conditions
            is_eos = (token_id == self.eos)
            is_max_tokens = (seq.num_completion_tokens == seq.max_tokens)
            
            if (not seq.ignore_eos and is_eos) or is_max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_flags.append(True)
            else:
                finished_flags.append(False)
        
        return finished_flags
