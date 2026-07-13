"""Dynamic Python scheduler for continuous batching.

Owns:
    Waiting/running queues, prefix-cache decisions, and capacity reservations.
Receives:
    Admitted ``Sequence`` objects and static serving capacity config.
Returns:
    Scheduled sequences plus an immutable host-only ``SchedulePlan``.
Invariant:
    Scheduling never creates or reads accelerator arrays.
"""

from collections import deque
from dataclasses import replace
from typing import Deque, List, Tuple

from nanovllm_jax.config import RuntimeConfig
from nanovllm_jax.batch import BucketShape, ScheduledRow, SchedulePlan
from nanovllm_jax.output import is_device_token
from nanovllm_jax.sequence import Sequence, SequenceStatus, SamplingParams
from nanovllm_jax.block_manager import BlockManager, PrefixCacheEntry

def _config_flag(config: RuntimeConfig | None, attr: str, *, default: bool = False) -> bool:
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return bool(default)


def _config_int(config: RuntimeConfig | None, attr: str, *, default: int = 0) -> int:
    if config is not None and hasattr(config, attr):
        return int(getattr(config, attr) or default)
    return int(default)


class Scheduler:
    """Scheduler for continuous batching.
    
    Manages:
    - Waiting queue (sequences waiting to start)
    - Running queue (sequences being generated)
    - Block allocation via BlockManager
    - Whole-request KV capacity reservation
    """

    def __init__(self, config: RuntimeConfig):
        self.max_num_seqs = int(getattr(config, 'max_num_seqs', 16) or 16)
        self.max_num_resident_seqs = int(
            getattr(config, "max_num_resident_seqs", None) or self.max_num_seqs
        )
        if self.max_num_resident_seqs < self.max_num_seqs:
            raise ValueError("max_num_resident_seqs must be >= max_num_seqs")
        self.max_num_batched_tokens = getattr(config, 'max_num_batched_tokens', 2048)
        self.eos_token_ids = frozenset(
            int(token_id)
            for token_id in getattr(config, "eos_token_ids", ())
        )
        self.block_size = config.block_size
        self.prefix_cache_enabled = bool(config.prefix_cache)
        self.prefill_buckets = tuple(getattr(config, "prefill_buckets", ()))
        self.prefill_token_buckets = tuple(getattr(config, "prefill_token_buckets", ()))
        self.prefill_layout = str(getattr(config, "prefill_layout", "packed") or "packed").lower()
        if self.prefill_layout not in {"packed", "dense"}:
            raise ValueError("prefill_layout must be 'packed' or 'dense'")
        self.batch_size_buckets = tuple(getattr(config, "batch_size_buckets", ()))
        self.decode_block_table_buckets = tuple(getattr(config, "decode_block_table_buckets", ()) or ())
        self.device_token_carry = _config_flag(config, "device_token_carry")
        if self.prefill_layout == "packed" and self.prefill_buckets:
            self.prefill_chunk_budget = max(self.prefill_buckets)
        else:
            self.prefill_chunk_budget = (
                max(self.prefill_token_buckets or self.prefill_buckets)
                if (self.prefill_token_buckets or self.prefill_buckets)
                else max(
                    64,
                    self.max_num_batched_tokens
                    if self.max_num_batched_tokens > 0
                    else 64,
                )
            )
        self.decode_lookahead_tokens = 1
        self.greedy_decode_burst_steps = max(
            1,
            _config_int(
                config,
                "greedy_decode_burst_steps",
                default=1,
            ),
        )
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.block_size,
            prefix_state_capacity=(
                self.max_num_resident_seqs
                if self.prefix_cache_enabled and config.linear_attn_layers
                else 0
            ),
        )
        
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

    def _can_reserve_waiting(self, seq: Sequence) -> bool:
        return self.block_manager.can_reserve(
            seq,
            total_blocks=self._required_blocks(seq),
            use_prefix_cache=self.prefix_cache_enabled,
        )

    def _reserve_waiting(self, seq: Sequence) -> None:
        self.block_manager.reserve(
            seq,
            total_blocks=self._required_blocks(seq),
            use_prefix_cache=self.prefix_cache_enabled,
        )

    def _required_blocks(self, seq: Sequence) -> int:
        total_tokens = seq.num_prompt_tokens + seq.max_tokens
        return (total_tokens + self.block_size - 1) // self.block_size

    def _admit_first_fitting_waiter(self) -> Sequence | None:
        """Return the first request that fits without stalling later waiters."""
        for _ in range(len(self.waiting)):
            candidate = self.waiting.popleft()
            if self._can_reserve_waiting(candidate):
                self._reserve_waiting(candidate)
                return candidate
            self.waiting.append(candidate)
        return None

    def cached_prefix_entries(
        self,
        seqs: List[Sequence],
    ) -> dict[int, PrefixCacheEntry]:
        """Return exact host metadata for prefixes selected during admission."""
        if not self.block_manager.prefix_cache.requires_state:
            return {}
        entries: dict[int, PrefixCacheEntry] = {}
        for seq in seqs:
            if seq.cached_prefix_hybrid_seeded:
                continue
            prefix_hash = seq.cached_prefix_hash
            if prefix_hash is None or seq.num_cached_tokens <= 0:
                continue
            entry = self.block_manager.prefix_cache.get(prefix_hash)
            if entry is None or entry.hybrid_state_handle is None:
                raise RuntimeError(f"missing hybrid state for prefix {prefix_hash}")
            if entry.token_count != seq.num_cached_tokens:
                raise RuntimeError(
                    "prefix KV and hybrid-state token counts differ: "
                    f"entry={entry.token_count}, request={seq.num_cached_tokens}"
                )
            entries[int(seq.seq_id)] = entry
        return entries

    def record_computed_prefixes(
        self,
        seqs: List[Sequence],
        prefill_chunk_lengths: List[int],
    ) -> dict[int, PrefixCacheEntry]:
        """Publish KV metadata and return prefixes that still need GDN state."""
        if not self.prefix_cache_enabled:
            return {}
        if len(prefill_chunk_lengths) != len(seqs):
            return {}
        pending: dict[int, PrefixCacheEntry] = {}
        for seq, chunk_len in zip(seqs, prefill_chunk_lengths):
            chunk_len = int(chunk_len)
            if chunk_len <= 0:
                continue
            computed_tokens = min(seq.num_prompt_tokens, seq.num_cached_tokens + chunk_len)
            if computed_tokens <= 0:
                continue
            entry = self.block_manager.publish_computed_prefix(seq, computed_tokens)
            if (
                self.block_manager.prefix_cache.requires_state
                and entry is not None
                and entry.hybrid_state_handle is None
            ):
                pending[int(seq.seq_id)] = entry
        self.block_manager.prefix_cache.make_state_room(list(pending.values()))
        return pending

    def publish_prefix_states(
        self,
        pending: dict[int, PrefixCacheEntry],
        handles_by_hash: dict[int, int],
    ) -> None:
        """Attach runner-owned state handles to already materialized KV entries."""
        published: set[int] = set()
        for entry in pending.values():
            if entry.prefix_hash in published:
                continue
            handle = handles_by_hash.get(entry.prefix_hash)
            if handle is None:
                raise RuntimeError(f"runner did not cache prefix {entry.prefix_hash}")
            self.block_manager.prefix_cache.publish(
                replace(entry, hybrid_state_handle=handle)
            )
            published.add(entry.prefix_hash)

    def take_released_prefix_state_handles(self) -> tuple[int, ...]:
        """Drain state handles invalidated by KV reuse or state-budget pressure."""
        return self.block_manager.prefix_cache.take_released_state_handles()

    def is_finished(self) -> bool:
        """Check if all sequences are done."""
        return not self.waiting and not self.running

    def is_pristine(self) -> bool:
        """Whether runner warmup can reset state without dangling host metadata."""
        return self.is_finished() and not self.block_manager.prefix_cache.entries

    def add(self, seq: Sequence):
        """Add a sequence to the waiting queue."""
        if seq.block_size != self.block_size:
            raise ValueError("sequence and scheduler use different block sizes")
        required_blocks = self._required_blocks(seq)
        if required_blocks > len(self.block_manager.blocks):
            raise ValueError(
                f"request needs {required_blocks} blocks but the engine has "
                f"{len(self.block_manager.blocks)}"
            )
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
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], SchedulePlan]:
        """Schedule sequences for execution.
        
        Returns:
            Tuple of scheduled sequences and their host execution plan.
        """
        scheduled_seqs: List[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        prefill_chunk_lens: List[int] = []
        scheduled_running: List[Sequence] = []
        prefill_token_budget = self._max_prefill_token_budget()
        ready_decode_rows = sum(
            1
            for seq in self.running
            if seq.num_cached_tokens >= seq.num_prompt_tokens
        )
        defer_waiting_prefill_for_decode = (
            self.max_num_resident_seqs > self.max_num_seqs
            and ready_decode_rows >= self.max_num_seqs
        )
        # Phase 1: Prefill - schedule new/waiting sequences and unfinished
        # prompt tails from already-allocated running sequences.
        while num_seqs < self.max_num_seqs:
            if num_batched_tokens >= prefill_token_budget:
                break
            seq = None
            from_waiting = False
            if (
                self.waiting
                and not defer_waiting_prefill_for_decode
                and len(self.running) + len(scheduled_running) < self.max_num_resident_seqs
            ):
                seq = self._admit_first_fitting_waiter()
                if seq is not None:
                    from_waiting = True

            if seq is None:
                next_seq = None
                for candidate in self.running:
                    if candidate.num_cached_tokens < candidate.num_prompt_tokens:
                        next_seq = candidate
                        break
                if next_seq is None:
                    break
                seq = next_seq
                from_waiting = False
                self.running.remove(seq)

            remaining_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
            if remaining_tokens <= 0:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                continue
            chunk_len = min(
                remaining_tokens,
                self.prefill_chunk_budget,
                prefill_token_budget - num_batched_tokens,
            )

            if self.prefill_layout != "packed":
                prospective_query_bucket = self._select_prefill_query_bucket(
                    max(prefill_chunk_lens + [chunk_len])
                )
                prospective_batch_bucket = self._select_batch_size_bucket(num_seqs + 1)
                prospective_padded_tokens = prospective_query_bucket * prospective_batch_bucket
                if (
                    scheduled_seqs
                    and self.max_num_batched_tokens > 0
                    and prospective_padded_tokens > self.max_num_batched_tokens
                ):
                    if from_waiting:
                        seq.status = SequenceStatus.WAITING
                        self.block_manager.deallocate(seq)
                        self.waiting.appendleft(seq)
                    else:
                        self.running.appendleft(seq)
                    break

            seq.status = SequenceStatus.RUNNING
            num_seqs += 1
            num_batched_tokens += chunk_len
            prefill_chunk_lens.append(chunk_len)
            scheduled_seqs.append(seq)
            scheduled_running.append(seq)
            # A non-final prompt chunk should not stop filling the current
            # prefill batch. Keep packing one chunk per active sequence until
            # the sequence or token budget is exhausted; unfinished prompts
            # remain in ``running`` and are picked up by a later prefill wave.
            if chunk_len < remaining_tokens:
                continue

        if scheduled_seqs:
            self.running.extend(scheduled_running)
            return scheduled_seqs, self.build_schedule_plan(
                scheduled_seqs,
                is_prefill=True,
                prefill_chunk_lens=prefill_chunk_lens,
            )
        
        # Phase 2: Decode - schedule running sequences
        running_candidates = 0
        running_budget = len(self.running)
        while self.running and num_seqs < self.max_num_seqs and running_candidates < running_budget:
            running_candidates += 1
            seq = self.running.popleft()
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                self.running.append(seq)
                continue
            
            # Complete capacity credits were reserved before prefill, so decode
            # can allocate physical pages without eviction or recomputation.
            remaining_tokens = max(1, seq.max_tokens - seq.num_completion_tokens)
            lookahead_tokens = 1
            if self.greedy_decode_burst_steps > 1 and seq.temperature == 0 and seq.ignore_eos:
                lookahead_tokens = min(self.greedy_decode_burst_steps, remaining_tokens)
            if not self.block_manager.can_append_slots(seq, lookahead_tokens):
                raise AssertionError("reserved request ran out of KV blocks")
            num_seqs += 1
            self.block_manager.may_append_slots(seq, lookahead_tokens)
            scheduled_seqs.append(seq)
        
        if not scheduled_seqs:
            raise RuntimeError(self._capacity_exhausted_message())
        self.running.extendleft(reversed(scheduled_seqs))
        decode_step_count = self._decode_step_count(scheduled_seqs)
        return scheduled_seqs, self.build_schedule_plan(
            scheduled_seqs,
            is_prefill=False,
            decode_step_count=decode_step_count,
        )

    def _decode_step_count(self, seqs: List[Sequence]) -> int:
        step_counts: List[int] = []
        for seq in seqs:
            remaining_tokens = max(1, seq.max_tokens - seq.num_completion_tokens)
            if self.greedy_decode_burst_steps > 1 and seq.temperature == 0 and seq.ignore_eos:
                step_count = min(self.greedy_decode_burst_steps, remaining_tokens)
            else:
                step_count = 1
            step_counts.append(int(step_count))
        return min(step_counts) if step_counts else 1

    def _max_prefill_token_budget(self) -> int:
        """Largest prefill token count that the configured buckets cover."""
        max_token_budget = max(
            1,
            self.max_num_batched_tokens
            if self.max_num_batched_tokens > 0
            else self.prefill_chunk_budget,
        )
        if self.prefill_token_buckets:
            max_token_budget = min(max_token_budget, max(self.prefill_token_buckets))
        elif self.prefill_layout == "packed" and self.prefill_buckets:
            max_token_budget = min(max_token_budget, max(self.prefill_buckets))
        elif self.prefill_buckets:
            max_dense_batch = (
                max(self.batch_size_buckets)
                if self.batch_size_buckets
                else self.max_num_seqs
            )
            max_token_budget = min(
                max_token_budget,
                max(self.prefill_buckets) * max(1, int(max_dense_batch)),
            )
        return int(max_token_budget)

    def build_schedule_plan(
        self,
        seqs: List[Sequence],
        *,
        is_prefill: bool,
        query_len_bucket: int | None = None,
        batch_size_bucket: int | None = None,
        max_blocks_per_seq: int | None = None,
        prefill_chunk_lens: List[int] | None = None,
        decode_step_count: int = 1,
    ) -> SchedulePlan:
        """Describe one step without allocating accelerator arrays."""
        rows: List[ScheduledRow] = []

        actual_max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        block_table_width = actual_max_blocks
        if max_blocks_per_seq is None:
            max_blocks_per_seq = self.max_blocks_per_seq
        if max_blocks_per_seq is not None:
            if actual_max_blocks > max_blocks_per_seq:
                raise ValueError(
                    f"scheduled block table needs {actual_max_blocks} blocks "
                    f"but bucket has {max_blocks_per_seq}"
                )
            block_table_width = max_blocks_per_seq
        if not is_prefill and self.decode_block_table_buckets:
            block_table_width = self._select_bucket(
                actual_max_blocks,
                self.decode_block_table_buckets,
                "decode block table",
            )
            if max_blocks_per_seq is not None and block_table_width > max_blocks_per_seq:
                raise ValueError(
                    f"decode block table bucket {block_table_width} exceeds "
                    f"max_blocks_per_seq {max_blocks_per_seq}"
                )

        for index, seq in enumerate(seqs):
            if is_prefill:
                start = seq.num_cached_tokens
                chunk_len = seq.num_tokens - start
                if prefill_chunk_lens is not None:
                    if index >= len(prefill_chunk_lens):
                        raise ValueError(
                            "prefill_chunk_lens length must match scheduled sequences"
                        )
                    chunk_len = prefill_chunk_lens[index]
                if chunk_len <= 0:
                    raise ValueError(
                        f"Scheduled sequence {seq.seq_id} has no executable tokens"
                    )
                end = start + chunk_len
                tokens = seq.token_ids[start:end]
                positions = range(start, end)
                final_chunk = end >= seq.num_tokens
                seq_len = end
            else:
                tokens = [seq.last_token]
                positions = [seq.num_tokens - 1]
                final_chunk = True
                seq_len = seq.num_tokens

            rows.append(
                ScheduledRow(
                    seq_id=int(seq.seq_id),
                    token_ids=tuple(int(token) for token in tokens),
                    positions=tuple(positions),
                    block_table=tuple(int(block) for block in seq.block_table),
                    seq_len=int(seq_len),
                    prefill_is_final=bool(final_chunk),
                    carries_device_token=(
                        not is_prefill
                        and seq.temperature == 0
                        and seq.ignore_eos
                        and is_device_token(getattr(seq, "last_token_device", None))
                    ),
                )
            )

        if batch_size_bucket is None:
            batch_size_bucket = self._select_batch_size_bucket(len(rows))
        if len(rows) > batch_size_bucket:
            raise ValueError(
                f"scheduled batch has {len(rows)} rows but bucket has {batch_size_bucket}"
            )

        max_query_len = max(row.query_len for row in rows)
        packed_prefill = is_prefill and self.prefill_layout == "packed"
        if packed_prefill:
            query_tokens = self._select_prefill_token_bucket(
                sum(row.query_len for row in rows)
            )
        else:
            if query_len_bucket is None and is_prefill:
                query_len_bucket = self._select_prefill_query_bucket(max_query_len)
            if query_len_bucket is None:
                query_len_bucket = max_query_len
            if (
                is_prefill
                and prefill_chunk_lens is not None
                and any(seq.num_cached_tokens > 0 for seq in seqs)
            ):
                query_len_bucket = max(query_len_bucket, 2)
            if max_query_len > query_len_bucket:
                raise ValueError(
                    f"scheduled query needs {max_query_len} tokens "
                    f"but bucket has {query_len_bucket}"
                )
            query_tokens = query_len_bucket

        return SchedulePlan(
            phase="prefill" if is_prefill else "decode",
            rows=tuple(rows),
            bucket=BucketShape(
                batch_size=batch_size_bucket,
                query_tokens=query_tokens,
                block_table_width=block_table_width,
                packed_prefill=packed_prefill,
            ),
            decode_steps=1 if is_prefill else max(1, int(decode_step_count)),
        )
    @staticmethod
    def _select_bucket(size: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in sorted(buckets):
            if size <= bucket:
                return bucket
        raise ValueError(f"{name} size {size} exceeds configured buckets {buckets}")

    def _select_batch_size_bucket(self, size: int) -> int:
        if self.batch_size_buckets:
            return self._select_bucket(size, self.batch_size_buckets, "batch")
        return size

    def _select_prefill_query_bucket(self, size: int) -> int:
        if self.prefill_buckets:
            return self._select_bucket(size, self.prefill_buckets, "prefill")
        return size

    def _select_prefill_token_bucket(self, size: int) -> int:
        buckets = self.prefill_token_buckets or self.prefill_buckets
        if buckets:
            return self._select_bucket(size, buckets, "prefill token")
        return size
    def _capacity_exhausted_message(self) -> str:
        stats = self.block_manager.stats()

        def seq_snapshot(seq: Sequence) -> dict[str, int]:
            return {
                "seq_id": int(seq.seq_id),
                "tokens": int(seq.num_tokens),
                "prompt_tokens": int(seq.num_prompt_tokens),
                "completion_tokens": int(seq.num_completion_tokens),
                "max_tokens": int(seq.max_tokens),
                "cached_tokens": int(seq.num_cached_tokens),
                "blocks": int(len(seq.block_table)),
                "required_blocks": self._required_blocks(seq),
            }

        running = [seq_snapshot(seq) for seq in list(self.running)[:8]]
        waiting = [seq_snapshot(seq) for seq in list(self.waiting)[:8]]
        return (
            "No sequence can be scheduled; KV cache capacity is exhausted "
            f"stats={stats} max_num_batched_tokens={self.max_num_batched_tokens} "
            f"max_num_seqs={self.max_num_seqs} max_num_resident_seqs={self.max_num_resident_seqs} "
            f"max_blocks_per_seq={self.max_blocks_per_seq} running={running} waiting={waiting}"
        )

    def release(self, seq: Sequence) -> None:
        """Release queue and block resources after a terminal transition."""
        self.block_manager.deallocate(seq)
        for requests in (self.waiting, self.running):
            if seq in requests:
                requests.remove(seq)
