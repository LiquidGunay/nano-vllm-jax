"""Scheduler for continuous batching."""

import os
from collections import deque
from dataclasses import replace
from typing import Deque, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.engine.sequence import DeviceTokenRef, Sequence, SequenceStatus, SamplingParams
from nanovllm_jax.engine.block_manager import BlockManager


def _device_int32_arrays(*values):
    return jax.device_put(tuple(np.asarray(value, dtype=np.int32) for value in values))


_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}


def _config_or_env_flag(config: Qwen3_5Config | None, attr: str, env_name: str, *, default: bool = False) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value in _TRUE_ENV_VALUES
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return bool(default)


def _config_or_env_int(config: Qwen3_5Config | None, attr: str, env_name: str, *, default: int = 0) -> int:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value or default)
    if config is not None and hasattr(config, attr):
        return int(getattr(config, attr) or default)
    return int(default)


def _is_device_token(value) -> bool:
    return isinstance(value, DeviceTokenRef) or (hasattr(value, "dtype") and hasattr(value, "shape"))


class Scheduler:
    """Scheduler for continuous batching.
    
    Manages:
    - Waiting queue (sequences waiting to start)
    - Running queue (sequences being generated)
    - Block allocation via BlockManager
    - Preemption (swapping out sequences)
    """

    def __init__(self, config: Qwen3_5Config):
        self.max_num_seqs = int(getattr(config, 'max_num_seqs', 16) or 16)
        self.max_num_resident_seqs = int(
            getattr(config, "max_num_resident_seqs", None) or self.max_num_seqs
        )
        if self.max_num_resident_seqs < self.max_num_seqs:
            raise ValueError("max_num_resident_seqs must be >= max_num_seqs")
        self.max_num_batched_tokens = getattr(config, 'max_num_batched_tokens', 2048)
        self.eos = getattr(config, 'eos', None)
        self.block_size = config.block_size
        self.enable_prefix_cache_execution = bool(getattr(config, "prefix_cache", True))
        self.prefix_cache_requires_hybrid_state = bool(getattr(config, "linear_attn_layers", ()))
        self.prefix_cache_hybrid_states: dict[int, object] = {}
        self.prefill_buckets = tuple(getattr(config, "prefill_buckets", ()))
        self.prefill_token_buckets = tuple(getattr(config, "prefill_token_buckets", ()))
        self.prefill_layout = str(getattr(config, "prefill_layout", "packed") or "packed").lower()
        if self.prefill_layout not in {"packed", "dense"}:
            raise ValueError("prefill_layout must be 'packed' or 'dense'")
        self.batch_size_buckets = tuple(getattr(config, "batch_size_buckets", ()))
        self.decode_block_table_buckets = tuple(getattr(config, "decode_block_table_buckets", ()) or ())
        self.jax_execution = getattr(config, "jax_execution", "eager")
        self.device_token_carry = _config_or_env_flag(
            config,
            "device_token_carry",
            "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        )
        self.static_decode_metadata = _config_or_env_flag(
            config,
            "static_decode_metadata",
            "NANO_VLLM_JAX_STATIC_DECODE_METADATA",
        )
        self.resident_decode_metadata = _config_or_env_flag(
            config,
            "resident_decode_metadata",
            "NANO_VLLM_JAX_RESIDENT_DECODE_METADATA",
        )
        self.static_decode_seq_lens_carry = _config_or_env_flag(
            config,
            "static_decode_seq_lens_carry",
            "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
        )
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
        self.speculative_method = str(getattr(config, "speculative_method", "none") or "none").lower()
        raw_num_speculative_tokens = max(0, int(getattr(config, "num_speculative_tokens", 0) or 0))
        if self.speculative_method == "none" and raw_num_speculative_tokens > 0:
            self.speculative_method = "mtp"
        self.num_speculative_tokens = (
            raw_num_speculative_tokens
            if self.speculative_method == "mtp"
            else 0
        )
        self.mtp_burst_groups = max(
            1,
            _config_or_env_int(
                config,
                "mtp_burst_groups",
                "NANO_VLLM_JAX_MTP_BURST_GROUPS",
                default=1,
            ),
        )
        self.mtp_max_active_rows = max(
            0,
            _config_or_env_int(
                config,
                "mtp_max_active_rows",
                "NANO_VLLM_JAX_MTP_MAX_ACTIVE_ROWS",
                default=0,
            ),
        )
        self.decode_lookahead_tokens = max(1, 1 + self.num_speculative_tokens)
        self.greedy_decode_burst_steps = max(
            1,
            _config_or_env_int(
                config,
                "greedy_decode_burst_steps",
                "NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS",
                default=1,
            ),
        )
        self.mtp_min_accept_rate = float(
            os.environ.get(
                "NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE",
                os.environ.get("NANO_VLLM_JAX_MTP_CONFIDENCE_MIN_ACCEPT_RATE", "0.75"),
            )
            or "0.75"
        )
        self.mtp_min_accept_samples = int(os.environ.get("NANO_VLLM_JAX_MTP_MIN_ACCEPT_SAMPLES", "8") or "8")
        self.mtp_min_speedup = float(os.environ.get("NANO_VLLM_JAX_MTP_MIN_SPEEDUP", "1.0") or "1.0")
        self.mtp_probe_steps = int(
            os.environ.get(
                "NANO_VLLM_JAX_MTP_PROBE_STEPS",
                "1",
            )
            or "1"
        )
        self.mtp_scheduler_gate_enabled = (
            self.speculative_method == "mtp"
            and self.num_speculative_tokens > 0
            and (self.mtp_min_accept_rate > 0 or self.mtp_min_speedup > 0)
        )
        self.mtp_dtype = str(config.get_dtype())
        self.mtp_backend = str(getattr(config, "backend", "unknown"))
        self.mtp_bucket_stats: dict[tuple[int, int, str, str, int | None, int], dict[str, object]] = {}
        self.mtp_active_bucket_key: tuple[int, int, str, str, int | None, int] | None = None
        self.mtp_observed_accepted = 0
        self.mtp_observed_rejected = 0
        self.mtp_stats_seen_accepted = 0
        self.mtp_stats_seen_rejected = 0
        self.mtp_stats_seen_proposed = 0
        self.mtp_stats_seen_seeded_main_steps = 0
        self.mtp_stats_seen_partial_rows = 0
        self.mtp_admission_enabled = True
        self.mtp_admission_reason = "enabled"
        self.mtp_latency_alpha = float(os.environ.get("NANO_VLLM_JAX_MTP_LATENCY_ALPHA", "0.2") or "0.2")
        self.mtp_latency_min_steps = int(os.environ.get("NANO_VLLM_JAX_MTP_LATENCY_MIN_STEPS", "1") or "1")
        self.mtp_baseline_ms_per_token: float | None = None
        self.mtp_spec_ms_per_token: float | None = None
        self.mtp_baseline_latency_steps = 0
        self.mtp_spec_latency_steps = 0
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        self._static_decode_metadata_cache: dict[str, object] | None = None
        self._static_decode_metadata_cache_by_key: dict[tuple[object, ...], dict[str, object]] = {}
        self._static_decode_constant_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, 
            config.block_size
        )
        # Override sequence block size
        Sequence.block_size = config.block_size
        
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

    def _prefix_cacheable_hashes(self) -> set[int] | None:
        if not self.enable_prefix_cache_execution:
            return None
        if not self.prefix_cache_requires_hybrid_state:
            return None
        return set(self.prefix_cache_hybrid_states)

    def _can_allocate_waiting(self, seq: Sequence) -> bool:
        return self.block_manager.can_allocate(
            seq,
            use_prefix_cache=self.enable_prefix_cache_execution,
            cacheable_hashes=self._prefix_cacheable_hashes(),
        )

    def _allocate_waiting(self, seq: Sequence) -> None:
        self.block_manager.allocate(
            seq,
            use_prefix_cache=self.enable_prefix_cache_execution,
            cacheable_hashes=self._prefix_cacheable_hashes(),
        )

    def record_computed_prefix_states(
        self,
        seqs: List[Sequence],
        prefill_chunk_lengths: List[int],
        prefix_states_by_seq: dict[int, object] | None = None,
    ) -> None:
        """Publish prefix-cache entries after prefill materializes them.

        For hybrid/GDN models, a prefix is reusable only when the matching
        hybrid state for that exact full-block prefix is also available.
        """
        if not self.enable_prefix_cache_execution:
            return
        if len(prefill_chunk_lengths) != len(seqs):
            return
        for seq, chunk_len in zip(seqs, prefill_chunk_lengths):
            chunk_len = int(chunk_len)
            if chunk_len <= 0:
                continue
            computed_tokens = min(seq.num_prompt_tokens, seq.num_cached_tokens + chunk_len)
            if computed_tokens <= 0:
                continue
            if self.prefix_cache_requires_hybrid_state:
                block_hash = self.block_manager.record_computed_prefix(
                    seq,
                    computed_tokens,
                    publish=False,
                )
                if block_hash is None:
                    continue
                state = (prefix_states_by_seq or {}).get(int(seq.seq_id))
                if state is None:
                    continue
                self.prefix_cache_hybrid_states[int(block_hash)] = state
                self.block_manager.publish_computed_prefix(seq, computed_tokens)
            else:
                self.block_manager.record_computed_prefix(
                    seq,
                    computed_tokens,
                    publish=True,
                )

    def set_mtp_backend(self, backend: object) -> None:
        """Set backend identity used in scheduler-side MTP bucket keys."""
        self.mtp_backend = type(backend).__name__ if not isinstance(backend, str) else backend

    def reset_mtp_admission(self) -> None:
        """Reset adaptive speculative admission counters."""
        self.mtp_bucket_stats = {}
        self.mtp_active_bucket_key = None
        self.mtp_observed_accepted = 0
        self.mtp_observed_rejected = 0
        self.mtp_stats_seen_accepted = 0
        self.mtp_stats_seen_rejected = 0
        self.mtp_stats_seen_proposed = 0
        self.mtp_stats_seen_seeded_main_steps = 0
        self.mtp_stats_seen_partial_rows = 0
        self.mtp_admission_enabled = True
        self.mtp_admission_reason = "enabled"
        self.mtp_baseline_ms_per_token = None
        self.mtp_spec_ms_per_token = None
        self.mtp_baseline_latency_steps = 0
        self.mtp_spec_latency_steps = 0

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
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], ScheduledBatch]:
        """Schedule sequences for execution.
        
        Returns:
            Tuple of (scheduled sequences, scheduled batch)
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
        waiting_blocked_by_kv = False
        while num_seqs < self.max_num_seqs:
            seq = None
            from_waiting = False
            if (
                self.waiting
                and not waiting_blocked_by_kv
                and not defer_waiting_prefill_for_decode
                and len(self.running) + len(scheduled_running) < self.max_num_resident_seqs
            ):
                candidate = self.waiting[0]
                if self._can_allocate_waiting(candidate):
                    seq = self.waiting.popleft()
                    from_waiting = True
                    self._allocate_waiting(seq)
                else:
                    waiting_blocked_by_kv = True

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
            chunk_len = min(remaining_tokens, self.prefill_chunk_budget)
            if (
                prefill_token_budget > 0
                and num_batched_tokens + chunk_len > prefill_token_budget
            ):
                available = prefill_token_budget - num_batched_tokens
                if available <= 0:
                    if from_waiting:
                        seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)
                    else:
                        scheduled_running.append(seq)
                    break
                chunk_len = min(chunk_len, available)

            chunk_len = min(chunk_len, remaining_tokens)
            if self.prefill_layout != "packed":
                prospective_query_bucket = self._select_prefill_query_bucket(
                    max(prefill_chunk_lens + [chunk_len])
                )
                prospective_batch_bucket = self._select_mtp_static_batch_size_bucket(num_seqs + 1)
                prospective_padded_tokens = prospective_query_bucket * prospective_batch_bucket
                if (
                    scheduled_seqs
                    and self.max_num_batched_tokens > 0
                    and prospective_padded_tokens > self.max_num_batched_tokens
                ):
                    if from_waiting:
                        seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)
                    else:
                        self.running.appendleft(seq)
                    break

            seq.status = SequenceStatus.RUNNING
            mtp_admitted = self.should_admit_mtp(
                seq,
                batch_size_bucket=self._select_mtp_static_batch_size_bucket(num_seqs + 1),
                active_decode_rows=0,
            )
            seq.mtp_admitted = mtp_admitted
            seq.mtp_admission_reason = self.mtp_admission_reason if mtp_admitted else "scheduler_gate"

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
            return scheduled_seqs, self.build_scheduled_batch(
                scheduled_seqs,
                is_prefill=True,
                prefill_chunk_lens=prefill_chunk_lens,
            )
        
        # Phase 2: Decode - schedule running sequences
        running_candidates = 0
        running_budget = len(self.running)
        ready_decode_rows = sum(
            1
            for seq in self.running
            if seq.num_cached_tokens >= seq.num_prompt_tokens
        )
        mtp_active_rows_cap_blocks_decode = (
            self.num_speculative_tokens > 0
            and self.mtp_max_active_rows > 0
            and min(ready_decode_rows, self.max_num_seqs) > self.mtp_max_active_rows
        )
        while self.running and num_seqs < self.max_num_seqs and running_candidates < running_budget:
            running_candidates += 1
            seq = self.running.popleft()
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                self.running.append(seq)
                continue
            
            # Ensure we can append
            if mtp_active_rows_cap_blocks_decode:
                mtp_admitted = False
                self.mtp_admission_reason = "active_rows_cap"
            else:
                mtp_admitted = self.should_admit_mtp(
                    seq,
                    for_decode=True,
                    batch_size_bucket=self._select_mtp_static_batch_size_bucket(num_seqs + 1),
                    active_decode_rows=num_seqs + 1,
                )
            seq.mtp_admitted = mtp_admitted
            seq.mtp_admission_reason = self.mtp_admission_reason if mtp_admitted else "scheduler_gate"
            remaining_tokens = max(1, seq.max_tokens - seq.num_completion_tokens)
            mtp_burst_groups = 1
            if mtp_admitted and self.num_speculative_tokens > 0:
                mtp_burst_groups = self.mtp_burst_groups
            lookahead_tokens = min(
                (
                    mtp_burst_groups * (1 + self.num_speculative_tokens)
                    if mtp_admitted
                    else 1
                ),
                remaining_tokens,
            )
            if (
                not mtp_admitted
                and self.greedy_decode_burst_steps > 1
                and seq.temperature == 0
                and seq.ignore_eos
            ):
                lookahead_tokens = min(self.greedy_decode_burst_steps, remaining_tokens)
            while not self.block_manager.can_append_slots(seq, lookahead_tokens):
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
                self.block_manager.may_append_slots(seq, lookahead_tokens)
                scheduled_seqs.append(seq)
        
        if not scheduled_seqs:
            raise RuntimeError(self._capacity_exhausted_message())
        if (
            self.num_speculative_tokens > 0
            and self.mtp_max_active_rows > 0
            and len(scheduled_seqs) > self.mtp_max_active_rows
        ):
            for seq in scheduled_seqs:
                seq.mtp_admitted = False
                seq.mtp_admission_reason = "active_rows_cap"
        elif self.num_speculative_tokens > 0:
            final_active_rows = len(scheduled_seqs)
            final_batch_size_bucket = self._select_mtp_static_batch_size_bucket(final_active_rows)
            for seq in scheduled_seqs:
                if not bool(getattr(seq, "mtp_admitted", False)):
                    continue
                final_admitted = self.should_admit_mtp(
                    seq,
                    for_decode=True,
                    batch_size_bucket=final_batch_size_bucket,
                    active_decode_rows=final_active_rows,
                )
                seq.mtp_admitted = final_admitted
                seq.mtp_admission_reason = self.mtp_admission_reason if final_admitted else "scheduler_gate"
        self.running.extendleft(reversed(scheduled_seqs))
        decode_step_count = self._decode_step_count_for_scheduled_batch(scheduled_seqs)
        return scheduled_seqs, self.build_scheduled_batch(
            scheduled_seqs,
            is_prefill=False,
            decode_step_count=decode_step_count,
        )

    def _decode_step_count_for_scheduled_batch(self, seqs: List[Sequence]) -> int:
        step_counts: List[int] = []
        for seq in seqs:
            remaining_tokens = max(1, seq.max_tokens - seq.num_completion_tokens)
            if bool(getattr(seq, "mtp_admitted", False)) and self.num_speculative_tokens > 0:
                step_count = min(
                    self.mtp_burst_groups * (1 + self.num_speculative_tokens),
                    remaining_tokens,
                )
            elif (
                self.greedy_decode_burst_steps > 1
                and seq.temperature == 0
                and seq.ignore_eos
            ):
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

    def _schedule_mixed_prefill_decode(self) -> Tuple[List[Sequence], ScheduledBatch] | None:
        """Backfill underfull decode batches with bounded packed prefill chunks."""
        if self.prefill_layout != "packed":
            return None
        if self.num_speculative_tokens != 0:
            return None

        ready_decode_rows = [
            seq
            for seq in self.running
            if seq.num_cached_tokens >= seq.num_prompt_tokens
        ]
        if not ready_decode_rows or len(ready_decode_rows) >= self.max_num_seqs:
            return None

        has_running_prefill_tail = any(
            seq.num_cached_tokens < seq.num_prompt_tokens for seq in self.running
        )
        if not self.waiting and not has_running_prefill_tail:
            return None

        max_token_budget = self._max_prefill_token_budget()

        decode_seqs: List[Sequence] = []
        for seq in ready_decode_rows:
            if len(decode_seqs) >= self.max_num_seqs - 1:
                break
            if not self.block_manager.can_append_slots(seq, 1):
                continue
            decode_seqs.append(seq)

        if not decode_seqs:
            return None

        row_budget = self.max_num_seqs - len(decode_seqs)
        mixed_chunk_budget = self._mixed_prefill_chunk_budget()
        mixed_prefill_budget = row_budget * mixed_chunk_budget
        remaining_token_budget = min(max_token_budget - len(decode_seqs), mixed_prefill_budget)
        if row_budget <= 0 or remaining_token_budget <= 0:
            return None

        prefill_seqs: List[Sequence] = []
        prefill_chunk_lens: List[int] = []
        waiting_to_allocate: List[Sequence] = []

        for seq in list(self.running):
            if len(prefill_seqs) >= row_budget or remaining_token_budget <= 0:
                break
            if seq.num_cached_tokens >= seq.num_prompt_tokens:
                continue
            remaining_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
            chunk_len = min(remaining_tokens, mixed_chunk_budget, remaining_token_budget)
            if chunk_len <= 0:
                continue
            prefill_seqs.append(seq)
            prefill_chunk_lens.append(chunk_len)
            remaining_token_budget -= chunk_len

        while (
            self.waiting
            and len(prefill_seqs) < row_budget
            and remaining_token_budget > 0
            and len(self.running) + len(waiting_to_allocate) < self.max_num_resident_seqs
        ):
            candidate = self.waiting[0]
            if not self._can_allocate_waiting(candidate):
                break
            seq = self.waiting.popleft()
            self._allocate_waiting(seq)
            remaining_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
            if remaining_tokens <= 0:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                continue
            chunk_len = min(remaining_tokens, mixed_chunk_budget, remaining_token_budget)
            if chunk_len <= 0:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                break
            prefill_seqs.append(seq)
            prefill_chunk_lens.append(chunk_len)
            waiting_to_allocate.append(seq)
            remaining_token_budget -= chunk_len

        if not prefill_seqs:
            return None

        for seq in decode_seqs:
            seq.mtp_admitted = False
            seq.mtp_admission_reason = "mixed_prefill_decode"
            self.block_manager.may_append_slots(seq, 1)

        for seq in waiting_to_allocate:
            seq.status = SequenceStatus.RUNNING
            seq.mtp_admitted = False
            seq.mtp_admission_reason = "mixed_prefill_decode"
            self.running.append(seq)

        for seq in prefill_seqs:
            if seq not in waiting_to_allocate:
                seq.status = SequenceStatus.RUNNING
                seq.mtp_admitted = False
                seq.mtp_admission_reason = "mixed_prefill_decode"

        scheduled_seqs = decode_seqs + prefill_seqs
        return scheduled_seqs, self.build_mixed_prefill_decode_batch(
            decode_seqs=decode_seqs,
            prefill_seqs=prefill_seqs,
            prefill_chunk_lens=prefill_chunk_lens,
        )

    def _mixed_prefill_chunk_budget(self) -> int:
        """Smallest warmed prefill chunk used when decode rows are live."""
        buckets = self.prefill_token_buckets or self.prefill_buckets
        if buckets:
            positive_buckets = [int(bucket) for bucket in buckets if int(bucket) > 0]
            if positive_buckets:
                return max(1, min(positive_buckets))
        return max(1, min(self.prefill_chunk_budget, self.max_num_batched_tokens or self.prefill_chunk_budget))

    def build_scheduled_batch(
        self,
        seqs: List[Sequence],
        *,
        is_prefill: bool,
        query_len_bucket: int | None = None,
        batch_size_bucket: int | None = None,
        max_blocks_per_seq: int | None = None,
        prefill_chunk_lens: List[int] | None = None,
        decode_step_count: int = 1,
    ) -> ScheduledBatch:
        """Build the canonical engine batch contract for one step."""
        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []
        prefill_is_final: List[bool] = []

        actual_max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        max_blocks = actual_max_blocks
        if max_blocks_per_seq is None:
            max_blocks_per_seq = self.max_blocks_per_seq
        if max_blocks_per_seq is not None:
            if actual_max_blocks > max_blocks_per_seq:
                raise ValueError(
                    f"scheduled block table needs {actual_max_blocks} blocks but bucket has {max_blocks_per_seq}"
                )
            max_blocks = max_blocks_per_seq
        if not is_prefill and self.decode_block_table_buckets:
            max_blocks = self._select_bucket(
                actual_max_blocks,
                self.decode_block_table_buckets,
                "decode block table",
            )
            if max_blocks_per_seq is not None and max_blocks > max_blocks_per_seq:
                raise ValueError(
                    f"decode block table bucket {max_blocks} exceeds max_blocks_per_seq {max_blocks_per_seq}"
                )
        for seq in seqs:
            if is_prefill:
                start = seq.num_cached_tokens
                chunk_len = seq.num_tokens - start
                if prefill_chunk_lens is not None:
                    seq_idx = len(query_lens)
                    if seq_idx >= len(prefill_chunk_lens):
                        raise ValueError(
                            "prefill_chunk_lens length must match number of scheduled prefill sequences"
                        )
                    chunk_len = prefill_chunk_lens[seq_idx]
                if chunk_len <= 0:
                    raise ValueError(f"Scheduled sequence {seq.seq_id} has no executable tokens")
                end = start + chunk_len
                tokens = seq.token_ids[start:end]
                positions = list(range(start, end))
                final_chunk = end >= seq.num_tokens
            else:
                tokens = [seq.last_token]
                positions = [seq.num_tokens - 1]
                final_chunk = True

            if not tokens:
                raise ValueError(f"Scheduled sequence {seq.seq_id} has no executable tokens")

            query_tokens.append(tokens)
            query_positions.append(positions)
            block_tables.append(seq.block_table + [0] * (max_blocks - len(seq.block_table)))
            seq_lens.append(end if is_prefill else seq.num_tokens)
            query_lens.append(len(tokens))
            prefill_is_final.append(final_chunk)

        if batch_size_bucket is None and self.batch_size_buckets:
            if is_prefill and self.prefill_layout == "packed":
                batch_size_bucket = self._select_mtp_static_batch_size_bucket(len(seqs))
            elif (
                not is_prefill
                and self.speculative_method == "mtp"
                and self.num_speculative_tokens > 0
                and any(bool(getattr(seq, "mtp_admitted", False)) for seq in seqs)
            ):
                batch_size_bucket = self._select_mtp_static_batch_size_bucket(len(seqs))
            else:
                batch_size_bucket = self._select_bucket(len(seqs), self.batch_size_buckets, "batch")
        if batch_size_bucket is None:
            batch_size_bucket = len(seqs)
        if len(seqs) > batch_size_bucket:
            raise ValueError(f"scheduled batch has {len(seqs)} seqs but bucket has {batch_size_bucket}")
        seq_ids_host = tuple([seq.seq_id for seq in seqs] + [-1] * (batch_size_bucket - len(seqs)))
        speculative_admitted_host = tuple(
            (
                not is_prefill
                and self.speculative_method == "mtp"
                and self.num_speculative_tokens > 0
                and row < len(seqs)
                and bool(getattr(seqs[row], "mtp_admitted", False))
            )
            for row in range(batch_size_bucket)
        )
        speculative_draft_tokens_host = tuple(-1 for _ in range(batch_size_bucket))

        if is_prefill and self.prefill_layout == "packed":
            return self._build_packed_prefill_batch(
                query_tokens=query_tokens,
                query_positions=query_positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_lens=query_lens,
                prefill_is_final=prefill_is_final,
                batch_size_bucket=batch_size_bucket,
                max_blocks=max_blocks,
                seq_ids_host=seq_ids_host,
            )

        max_query_len = max(query_lens)
        if query_len_bucket is None and is_prefill and self.prefill_buckets:
            query_len_bucket = self._select_bucket(max_query_len, self.prefill_buckets, "prefill")
        if query_len_bucket is None:
            query_len_bucket = max_query_len
            if is_prefill and prefill_chunk_lens is not None:
                has_cached_prefix = any(seq.num_cached_tokens > 0 for seq in seqs)
                if has_cached_prefix:
                    query_len_bucket = max(query_len_bucket, 2)
        if max_query_len > query_len_bucket:
            raise ValueError(f"scheduled query needs {max_query_len} tokens but bucket has {query_len_bucket}")

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

        query_lens_host = tuple(query_lens)
        seq_lens_host = tuple(seq_lens)
        block_tables_host = tuple(tuple(int(block) for block in row) for row in block_tables)
        uses_static_decode_metadata = self._can_use_static_decode_metadata(
            seqs,
            is_prefill=is_prefill,
            query_len_bucket=query_len_bucket,
            padded_tokens=padded_tokens,
            query_lens_host=query_lens_host,
        )
        if uses_static_decode_metadata:
            (
                tokens_array,
                positions_array,
                seq_ids_array,
                query_start_loc_array,
                block_tables_array,
                seq_lens_array,
            ) = self._static_decode_device_arrays(
                padded_tokens=padded_tokens,
                padded_positions=padded_positions,
                seq_ids_host=seq_ids_host,
                query_start_loc=query_start_loc,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_lens_host=query_lens_host,
                resident_decode_metadata=self.resident_decode_metadata,
            )
        else:
            if not is_prefill:
                self._static_decode_metadata_cache = None
            (
                tokens_array,
                positions_array,
                seq_ids_array,
                query_start_loc_array,
                block_tables_array,
                seq_lens_array,
            ) = _device_int32_arrays(
                padded_tokens,
                padded_positions,
                seq_ids_host,
                query_start_loc,
                block_tables,
                seq_lens,
            )
        return ScheduledBatch(
            tokens=tokens_array,
            positions=positions_array,
            seq_ids=seq_ids_array,
            query_start_loc=query_start_loc_array,
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else sum(query_lens),
            block_tables=block_tables_array,
            seq_lens=seq_lens_array,
            prefill_is_final=prefill_is_final if is_prefill else None,
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            decode_step_count_host=1 if is_prefill else max(1, int(decode_step_count)),
            uses_static_decode_metadata=uses_static_decode_metadata,
            speculative_method=(
                self.speculative_method
                if not is_prefill and any(speculative_admitted_host)
                else "none"
            ),
            speculative_num_tokens=(
                self.num_speculative_tokens
                if not is_prefill and any(speculative_admitted_host)
                else 0
            ),
            speculative_draft_tokens=jnp.array(speculative_draft_tokens_host, dtype=jnp.int32)
            if not is_prefill and self.speculative_method == "mtp"
            else None,
            speculative_draft_tokens_host=speculative_draft_tokens_host
            if not is_prefill and self.speculative_method == "mtp"
            else None,
            speculative_admitted_host=speculative_admitted_host
            if not is_prefill and self.speculative_method == "mtp"
            else None,
        )

    def build_mixed_prefill_decode_batch(
        self,
        *,
        decode_seqs: List[Sequence],
        prefill_seqs: List[Sequence],
        prefill_chunk_lens: List[int],
    ) -> ScheduledBatch:
        """Build a packed cached step containing decode rows and prefill chunks."""
        if not decode_seqs:
            raise ValueError("mixed prefill/decode batch requires at least one decode row")
        if not prefill_seqs:
            raise ValueError("mixed prefill/decode batch requires at least one prefill row")
        if len(prefill_seqs) != len(prefill_chunk_lens):
            raise ValueError("prefill_chunk_lens length must match prefill_seqs")

        seqs = decode_seqs + prefill_seqs
        actual_max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        max_blocks = actual_max_blocks
        max_blocks_per_seq = self.max_blocks_per_seq
        if max_blocks_per_seq is not None:
            if actual_max_blocks > max_blocks_per_seq:
                raise ValueError(
                    f"scheduled block table needs {actual_max_blocks} blocks but bucket has {max_blocks_per_seq}"
                )
            max_blocks = max_blocks_per_seq

        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []
        prefill_is_final: List[bool] = []

        for seq in decode_seqs:
            query_tokens.append([seq.last_token])
            query_positions.append([seq.num_tokens - 1])
            block_tables.append(seq.block_table + [0] * (max_blocks - len(seq.block_table)))
            seq_lens.append(seq.num_tokens)
            query_lens.append(1)
            prefill_is_final.append(True)

        for seq, chunk_len in zip(prefill_seqs, prefill_chunk_lens):
            start = seq.num_cached_tokens
            remaining_tokens = seq.num_prompt_tokens - start
            chunk_len = min(int(chunk_len), remaining_tokens)
            if chunk_len <= 0:
                raise ValueError(f"Scheduled sequence {seq.seq_id} has no executable prefill tokens")
            end = start + chunk_len
            tokens = seq.token_ids[start:end]
            positions = list(range(start, end))
            query_tokens.append(tokens)
            query_positions.append(positions)
            block_tables.append(seq.block_table + [0] * (max_blocks - len(seq.block_table)))
            seq_lens.append(end)
            query_lens.append(len(tokens))
            prefill_is_final.append(end >= seq.num_tokens)

        if self.batch_size_buckets:
            batch_size_bucket = self._select_bucket(len(seqs), self.batch_size_buckets, "batch")
        else:
            batch_size_bucket = len(seqs)
        if len(seqs) > batch_size_bucket:
            raise ValueError(f"scheduled batch has {len(seqs)} seqs but bucket has {batch_size_bucket}")
        seq_ids_host = tuple([seq.seq_id for seq in seqs] + [-1] * (batch_size_bucket - len(seqs)))

        batch = self._build_packed_prefill_batch(
            query_tokens=query_tokens,
            query_positions=query_positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_lens=query_lens,
            prefill_is_final=prefill_is_final,
            batch_size_bucket=batch_size_bucket,
            max_blocks=max_blocks,
            seq_ids_host=seq_ids_host,
        )
        return replace(batch, mixed_prefill_decode=True)

    def _build_packed_prefill_batch(
        self,
        *,
        query_tokens: List[List[int]],
        query_positions: List[List[int]],
        block_tables: List[List[int]],
        seq_lens: List[int],
        query_lens: List[int],
        prefill_is_final: List[bool],
        batch_size_bucket: int,
        max_blocks: int,
        seq_ids_host: tuple[int, ...],
    ) -> ScheduledBatch:
        actual_tokens = sum(query_lens)
        if actual_tokens <= 0:
            raise ValueError("packed prefill requires at least one executable token")

        token_bucket = self._select_prefill_token_bucket(actual_tokens)
        if actual_tokens > token_bucket:
            raise ValueError(f"scheduled prefill needs {actual_tokens} tokens but bucket has {token_bucket}")

        packed_tokens: List[int] = []
        packed_positions: List[int] = []
        token_row_ids: List[int] = []
        query_start_loc = [0]
        for row, (tokens, positions, qlen) in enumerate(zip(query_tokens, query_positions, query_lens)):
            if qlen != len(tokens):
                raise ValueError("query_lens must match packed query token lengths")
            packed_tokens.extend(tokens)
            packed_positions.extend(positions)
            token_row_ids.extend([row] * qlen)
            query_start_loc.append(query_start_loc[-1] + qlen)

        for _ in range(batch_size_bucket - len(query_lens)):
            block_tables.append([0] * max_blocks)
            seq_lens.append(0)
            query_lens.append(0)
            query_start_loc.append(query_start_loc[-1])

        pad = token_bucket - actual_tokens
        packed_tokens.extend([0] * pad)
        packed_positions.extend([0] * pad)
        token_row_ids.extend([0] * pad)

        query_lens_host = tuple(query_lens)
        seq_lens_host = tuple(seq_lens)
        block_tables_host = tuple(tuple(int(block) for block in row) for row in block_tables)
        return ScheduledBatch(
            tokens=jnp.array([packed_tokens], dtype=jnp.int32),
            positions=jnp.array([packed_positions], dtype=jnp.int32),
            seq_ids=jnp.array(seq_ids_host, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=True,
            num_prefill_tokens=actual_tokens,
            num_decode_tokens=0,
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
            prefill_is_final=prefill_is_final,
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            packed_prefill=True,
            token_row_ids=jnp.array([token_row_ids], dtype=jnp.int32),
        )

    def _can_use_static_decode_metadata(
        self,
        seqs: List[Sequence],
        *,
        is_prefill: bool,
        query_len_bucket: int,
        padded_tokens: List[List[int]],
        query_lens_host: tuple[int, ...],
    ) -> bool:
        if (
            is_prefill
            or not self.static_decode_metadata
            or self.jax_execution not in {"decode-jit", "jit"}
            or not self.device_token_carry
            or (
                self.num_speculative_tokens != 0
                and any(bool(getattr(seq, "mtp_admitted", False)) for seq in seqs)
            )
            or query_len_bucket != 1
            or not seqs
        ):
            return False
        for row, seq in enumerate(seqs):
            if int(query_lens_host[row]) != 1:
                return False
            if seq.temperature != 0 or not seq.ignore_eos:
                return False
            if not _is_device_token(getattr(seq, "last_token_device", None)):
                return False
            # The runner must replace this placeholder from its device-token
            # carry map before executing the JIT. If it cannot, it raises.
            if int(padded_tokens[row][0]) != 0:
                return False
        return True

    def _static_decode_device_arrays(
        self,
        *,
        padded_tokens: List[List[int]],
        padded_positions: List[List[int]],
        seq_ids_host: tuple[int, ...],
        query_start_loc: List[int],
        block_tables: List[List[int]],
        seq_lens: List[int],
        query_lens_host: tuple[int, ...],
        resident_decode_metadata: bool = False,
    ):
        token_shape = tuple((len(padded_tokens), len(padded_tokens[0]) if padded_tokens else 0))
        block_table_shape = tuple((len(block_tables), len(block_tables[0]) if block_tables else 0))
        if resident_decode_metadata:
            device_seq_ids_host = tuple(
                row if int(query_len) > 0 else -1
                for row, query_len in enumerate(query_lens_host)
            )
            constant_key = (token_shape, query_lens_host, "resident")
        else:
            device_seq_ids_host = seq_ids_host
            constant_key = (token_shape, seq_ids_host, query_lens_host)
        constant_cache = self._static_decode_constant_cache.get(constant_key)
        if constant_cache is None:
            query_start_loc_array = jax.device_put(np.asarray(query_start_loc, dtype=np.int32))
            # Decode positions are recomputed from seq_lens inside the compiled
            # executor for width-1 decode. Keep a stable placeholder here so
            # block-table cache misses do not rebuild an unused host array.
            positions_array = jax.device_put(np.zeros_like(np.asarray(padded_positions, dtype=np.int32)))
            tokens_array = jax.device_put(np.asarray(padded_tokens, dtype=np.int32))
            seq_ids_array = jax.device_put(np.asarray(device_seq_ids_host, dtype=np.int32))
            constant_cache = {
                "tokens": tokens_array,
                "positions": positions_array,
                "seq_ids": seq_ids_array,
                "query_start_loc": query_start_loc_array,
            }
            self._static_decode_constant_cache[constant_key] = constant_cache
        if resident_decode_metadata:
            key = (token_shape, block_table_shape, query_lens_host, "resident")
        else:
            key = (
                token_shape,
                block_table_shape,
                seq_ids_host,
                query_lens_host,
                tuple(tuple(row) for row in block_tables),
            )
        cache = self._static_decode_metadata_cache
        if resident_decode_metadata:
            cache = self._static_decode_metadata_cache_by_key.get(key)
        if cache is None or cache.get("key") != key:
            if resident_decode_metadata:
                block_tables_array = jax.device_put(np.zeros(block_table_shape, dtype=np.int32))
                seq_lens_array = jax.device_put(np.zeros((len(seq_lens),), dtype=np.int32))
            else:
                block_tables_array = jax.device_put(np.asarray(block_tables, dtype=np.int32))
                seq_lens_array = jax.device_put(np.asarray(seq_lens, dtype=np.int32))
            cache = {
                "key": key,
                "tokens": constant_cache["tokens"],
                "positions": constant_cache["positions"],
                "seq_ids": constant_cache["seq_ids"],
                "query_start_loc": constant_cache["query_start_loc"],
                "block_tables": block_tables_array,
                "seq_lens": seq_lens_array,
            }
            if resident_decode_metadata:
                self._static_decode_metadata_cache_by_key[key] = cache
            else:
                self._static_decode_metadata_cache = cache
        else:
            if self.static_decode_seq_lens_carry or resident_decode_metadata:
                seq_lens_array = cache["seq_lens"]
            else:
                seq_lens_array = jax.device_put(np.asarray(seq_lens, dtype=np.int32))
        return (
            cache["tokens"],
            cache["positions"],
            cache["seq_ids"],
            cache["query_start_loc"],
            cache["block_tables"],
            seq_lens_array,
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

    def _select_mtp_static_batch_size_bucket(self, size: int) -> int:
        """Return a reusable physical row bucket for active MTP serving.

        ``mtp_max_active_rows`` is the scheduler's cap for rows that may enter
        speculative decode.  When it is configured, prefill metadata can use
        that same row bucket for all smaller MTP-serving batches so generic
        warmup/runtime do not compile separate packed-prefill executables for
        row counts 1, 2, ..., cap.
        """

        if (
            self.speculative_method == "mtp"
            and self.num_speculative_tokens > 0
            and self.mtp_max_active_rows > 0
            and size <= self.mtp_max_active_rows
        ):
            return self._select_batch_size_bucket(self.mtp_max_active_rows)
        return self._select_batch_size_bucket(size)

    def _select_prefill_query_bucket(self, size: int) -> int:
        if self.prefill_buckets:
            return self._select_bucket(size, self.prefill_buckets, "prefill")
        return size

    def _select_prefill_token_bucket(self, size: int) -> int:
        buckets = self.prefill_token_buckets or self.prefill_buckets
        if buckets:
            return self._select_bucket(size, buckets, "prefill token")
        return size

    def _mtp_bucket_key(
        self,
        *,
        batch_size_bucket: int,
        active_decode_rows: int,
    ) -> tuple[int, int, str, str, int | None, int]:
        return (
            int(batch_size_bucket),
            int(active_decode_rows),
            self.mtp_dtype,
            self.mtp_backend,
            self.max_blocks_per_seq,
            self.num_speculative_tokens,
        )

    @staticmethod
    def _new_mtp_bucket_stats() -> dict[str, object]:
        return {
            "observed_accepted": 0,
            "observed_rejected": 0,
            "observed_proposed": 0,
            "observed_seeded_main_steps": 0,
            "observed_partial_rows": 0,
            "admission_enabled": True,
            "admission_reason": "enabled",
            "baseline_ms_per_token": None,
            "spec_ms_per_token": None,
            "baseline_latency_steps": 0,
            "spec_latency_steps": 0,
        }

    def _get_mtp_bucket_stats(
        self,
        key: tuple[int, int, str, str, int | None, int],
    ) -> dict[str, object]:
        stats = self.mtp_bucket_stats.get(key)
        if stats is None:
            stats = self._new_mtp_bucket_stats()
            self.mtp_bucket_stats[key] = stats
        return stats

    def _sync_legacy_mtp_fields(
        self,
        key: tuple[int, int, str, str, int | None, int],
        stats: dict[str, object],
    ) -> None:
        self.mtp_active_bucket_key = key
        self.mtp_observed_accepted = int(stats["observed_accepted"])
        self.mtp_observed_rejected = int(stats["observed_rejected"])
        self.mtp_admission_enabled = bool(stats["admission_enabled"])
        self.mtp_admission_reason = str(stats["admission_reason"])
        self.mtp_baseline_ms_per_token = stats["baseline_ms_per_token"]  # type: ignore[assignment]
        self.mtp_spec_ms_per_token = stats["spec_ms_per_token"]  # type: ignore[assignment]
        self.mtp_baseline_latency_steps = int(stats["baseline_latency_steps"])
        self.mtp_spec_latency_steps = int(stats["spec_latency_steps"])

    def _mtp_physical_bucket_block_reason(
        self,
        key: tuple[int, int, str, str, int | None, int],
    ) -> str | None:
        """Return why this K=1 active-row bucket is blocked by its physical bucket.

        Admission stats are tracked per active row count so mixed/underfilled
        serving can be diagnosed. For serving, however, a full physical bucket
        that is already measured as slower with K=1 MTP should also disable
        speculative decode for smaller active-row siblings in the same compiled
        shape. Otherwise tail batches can stay in ``warming`` and keep taking
        the slow repair path after the main B bucket has proven unprofitable.
        """
        batch_size, active_rows, dtype, backend, max_blocks_per_seq, num_speculative_tokens = key
        if num_speculative_tokens != 1:
            return None
        if os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNDERFILLED_AFTER_PHYSICAL_LOW_THROUGHPUT", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            return None

        for sibling_key, sibling_stats in self.mtp_bucket_stats.items():
            (
                sibling_batch_size,
                sibling_active_rows,
                sibling_dtype,
                sibling_backend,
                sibling_max_blocks,
                sibling_num_speculative,
            ) = sibling_key
            same_physical_bucket = (
                sibling_batch_size == batch_size
                and sibling_dtype == dtype
                and sibling_backend == backend
                and sibling_max_blocks == max_blocks_per_seq
                and sibling_num_speculative == num_speculative_tokens
            )
            if not same_physical_bucket:
                continue
            if sibling_active_rows < active_rows:
                continue
            if str(sibling_stats.get("admission_reason")) == "low_throughput":
                return "physical_low_throughput"
        return None

    def _apply_mtp_physical_bucket_gate(
        self,
        key: tuple[int, int, str, str, int | None, int],
        stats: dict[str, object],
    ) -> None:
        reason = self._mtp_physical_bucket_block_reason(key)
        if reason is None:
            return
        if str(stats.get("admission_reason")) == "low_throughput":
            return
        stats["admission_enabled"] = False
        stats["admission_reason"] = reason

    def should_admit_mtp(
        self,
        seq: Sequence,
        *,
        for_decode: bool = False,
        batch_size_bucket: int | None = None,
        active_decode_rows: int | None = None,
    ) -> bool:
        """Return whether this decode row may attempt speculative decoding."""
        if self.num_speculative_tokens <= 0:
            self.mtp_admission_reason = "disabled"
            return False
        if seq.temperature != 0:
            self.mtp_admission_reason = "temperature"
            return False
        if for_decode and self.device_token_carry and not seq.ignore_eos:
            self.mtp_admission_reason = "eos_host_check"
            return False
        remaining = seq.max_tokens - seq.num_completion_tokens
        if remaining <= 1:
            self.mtp_admission_reason = "remaining_tokens"
            return False
        relax_start_boundary = (
            os.environ.get("NANO_VLLM_JAX_MTP_RELAX_START_BOUNDARY", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        if (
            for_decode
            and self.num_speculative_tokens == 1
            and not relax_start_boundary
            and seq.num_tokens % self.block_size == 0
        ):
            self.mtp_admission_reason = "start_boundary"
            return False
        if (
            for_decode
            and self.num_speculative_tokens == 1
            and not relax_start_boundary
            and (seq.num_tokens + 2) % self.block_size == 0
        ):
            self.mtp_admission_reason = "bonus_boundary"
            return False
        if for_decode and self.mtp_max_active_rows > 0:
            if active_decode_rows is None:
                active_decode_rows = batch_size_bucket if batch_size_bucket is not None else 1
            if int(active_decode_rows) > self.mtp_max_active_rows:
                self.mtp_admission_reason = "active_rows_cap"
                return False
        if not self.mtp_scheduler_gate_enabled:
            self.mtp_admission_reason = "enabled"
            return True
        if batch_size_bucket is None:
            batch_size_bucket = 1
        if active_decode_rows is None:
            active_decode_rows = batch_size_bucket if for_decode else 0
        key = self._mtp_bucket_key(
            batch_size_bucket=batch_size_bucket,
            active_decode_rows=active_decode_rows,
        )
        bucket_stats = self._get_mtp_bucket_stats(key)
        self._update_mtp_admission_decision(bucket_stats)
        self._apply_mtp_physical_bucket_gate(key, bucket_stats)
        self._sync_legacy_mtp_fields(key, bucket_stats)
        return bool(bucket_stats["admission_enabled"])

    def update_mtp_admission(
        self,
        speculative_stats: dict,
        *,
        is_decode: bool = False,
        elapsed_seconds: float | None = None,
        emitted_tokens: int = 0,
        batch: ScheduledBatch | None = None,
    ) -> None:
        """Update scheduler-level speculative admission from cumulative stats.

        The runner owns correctness and exact accept/reject accounting. The
        scheduler consumes only cumulative counters and decides whether future
        decode rows should reserve lookahead slots and enter the verifier.
        """
        if not self.mtp_scheduler_gate_enabled:
            return
        if batch is None:
            batch_size_bucket = 1
            active_decode_rows = 1 if is_decode else 0
        else:
            batch_size_bucket = int(batch.tokens.shape[0])
            active_decode_rows = int(batch.num_decode_tokens) if is_decode else 0
        key = self._mtp_bucket_key(
            batch_size_bucket=batch_size_bucket,
            active_decode_rows=active_decode_rows,
        )
        bucket_stats = self._get_mtp_bucket_stats(key)
        accepted = int(speculative_stats.get("drafts_accepted", 0) or 0)
        rejected = int(speculative_stats.get("drafts_rejected", 0) or 0)
        proposed = int(speculative_stats.get("drafts_proposed", 0) or 0)
        seeded_main_steps = int(speculative_stats.get("fallback_seeded_main_steps", 0) or 0)
        partial_rows = int(speculative_stats.get("fallback_partial_rows", 0) or 0)
        delta_accepted = max(0, accepted - self.mtp_stats_seen_accepted)
        delta_rejected = max(0, rejected - self.mtp_stats_seen_rejected)
        delta_proposed = max(0, proposed - self.mtp_stats_seen_proposed)
        delta_seeded_main_steps = max(0, seeded_main_steps - self.mtp_stats_seen_seeded_main_steps)
        delta_partial_rows = max(0, partial_rows - self.mtp_stats_seen_partial_rows)
        self.mtp_stats_seen_accepted = accepted
        self.mtp_stats_seen_rejected = rejected
        self.mtp_stats_seen_proposed = proposed
        self.mtp_stats_seen_seeded_main_steps = seeded_main_steps
        self.mtp_stats_seen_partial_rows = partial_rows
        attempted_spec = (delta_accepted + delta_rejected) > 0
        speculative_overhead = attempted_spec or delta_proposed > 0 or delta_seeded_main_steps > 0 or delta_partial_rows > 0

        if is_decode and elapsed_seconds is not None and emitted_tokens > 0:
            ms_per_token = (elapsed_seconds * 1000.0) / emitted_tokens
            if speculative_overhead:
                bucket_stats["spec_ms_per_token"] = self._ewma(
                    bucket_stats["spec_ms_per_token"],  # type: ignore[arg-type]
                    ms_per_token,
                    self.mtp_latency_alpha,
                )
                bucket_stats["spec_latency_steps"] = int(bucket_stats["spec_latency_steps"]) + 1
            else:
                bucket_stats["baseline_ms_per_token"] = self._ewma(
                    bucket_stats["baseline_ms_per_token"],  # type: ignore[arg-type]
                    ms_per_token,
                    self.mtp_latency_alpha,
                )
                bucket_stats["baseline_latency_steps"] = int(bucket_stats["baseline_latency_steps"]) + 1
        if delta_accepted == 0 and delta_rejected == 0 and delta_proposed == 0 and delta_seeded_main_steps == 0 and delta_partial_rows == 0:
            self._update_mtp_admission_decision(bucket_stats)
            self._apply_mtp_physical_bucket_gate(key, bucket_stats)
            self._sync_legacy_mtp_fields(key, bucket_stats)
            return

        bucket_stats["observed_accepted"] = int(bucket_stats["observed_accepted"]) + delta_accepted
        bucket_stats["observed_rejected"] = int(bucket_stats["observed_rejected"]) + delta_rejected
        bucket_stats["observed_proposed"] = int(bucket_stats["observed_proposed"]) + delta_proposed
        bucket_stats["observed_seeded_main_steps"] = int(bucket_stats["observed_seeded_main_steps"]) + delta_seeded_main_steps
        bucket_stats["observed_partial_rows"] = int(bucket_stats["observed_partial_rows"]) + delta_partial_rows
        self._update_mtp_admission_decision(bucket_stats)
        self._apply_mtp_physical_bucket_gate(key, bucket_stats)
        self._sync_legacy_mtp_fields(key, bucket_stats)

    @staticmethod
    def _ewma(current: float | None, value: float, alpha: float) -> float:
        if current is None:
            return value
        return (1.0 - alpha) * current + alpha * value

    def _update_mtp_admission_decision(self, stats: dict[str, object]) -> None:
        verified = int(stats["observed_accepted"]) + int(stats["observed_rejected"])
        spec_steps = int(stats["spec_latency_steps"])
        baseline_steps = int(stats["baseline_latency_steps"])
        accept_rate = int(stats["observed_accepted"]) / max(1, verified) if verified else 0.0
        probing = spec_steps < self.mtp_probe_steps
        acceptance_ready = verified >= self.mtp_min_accept_samples
        acceptance_ok = (not acceptance_ready) or accept_rate >= self.mtp_min_accept_rate
        latency_ok = True
        latency_ready = (
            baseline_steps >= self.mtp_latency_min_steps
            and spec_steps >= max(self.mtp_latency_min_steps, self.mtp_probe_steps)
            and stats["baseline_ms_per_token"] is not None
            and stats["spec_ms_per_token"] is not None
        )
        if latency_ready:
            measured_speedup = float(stats["baseline_ms_per_token"]) / max(
                1e-9,
                float(stats["spec_ms_per_token"]),
            )
            latency_ok = measured_speedup >= self.mtp_min_speedup

        stats["probe_complete"] = not probing
        stats["acceptance_ready"] = acceptance_ready
        stats["latency_ready"] = latency_ready
        if probing:
            stats["admission_enabled"] = True
            stats["admission_reason"] = "probing_mtp"
        elif (
            not latency_ready
            and self.mtp_min_speedup > 0
            and stats["spec_ms_per_token"] is not None
        ):
            # Once the speculative probe has run, collect baseline latency right
            # away. Waiting for the acceptance confidence window first can keep
            # small tail buckets on an obviously expensive verifier for many
            # steps without ever measuring the ordinary decode alternative.
            stats["admission_enabled"] = False
            stats["admission_reason"] = "waiting_baseline_probe"
        elif not latency_ok:
            stats["admission_enabled"] = False
            stats["admission_reason"] = "low_throughput"
        elif not acceptance_ready:
            stats["admission_enabled"] = True
            stats["admission_reason"] = "warming_acceptance"
        elif not acceptance_ok:
            stats["admission_enabled"] = False
            stats["admission_reason"] = "low_acceptance"
        else:
            stats["admission_enabled"] = True
            stats["admission_reason"] = "enabled"

    def get_mtp_admission_report(self) -> dict[str, object]:
        """Return JSON-friendly per-bucket MTP admission stats."""
        buckets = []
        for key, stats in self.mtp_bucket_stats.items():
            batch_size, active_decode_rows, dtype, backend, max_blocks_per_seq, num_speculative_tokens = key
            verified = int(stats["observed_accepted"]) + int(stats["observed_rejected"])
            accept_rate = int(stats["observed_accepted"]) / verified if verified else 0.0
            baseline_ms = stats["baseline_ms_per_token"]
            spec_ms = stats["spec_ms_per_token"]
            measured_speedup = None
            if baseline_ms is not None and spec_ms is not None:
                measured_speedup = float(baseline_ms) / max(1e-9, float(spec_ms))
            physical_bucket_reason = self._mtp_physical_bucket_block_reason(key)
            buckets.append(
                {
                    "key": {
                        "physical_batch_size": batch_size,
                        "active_decode_rows": active_decode_rows,
                        "dtype": dtype,
                        "backend": backend,
                        "max_blocks_per_seq": max_blocks_per_seq,
                        "num_speculative_tokens": num_speculative_tokens,
                    },
                    "admission_enabled": bool(stats["admission_enabled"]),
                    "admission_reason": str(stats["admission_reason"]),
                    "physical_bucket_admission_enabled": physical_bucket_reason is None,
                    "physical_bucket_reason": physical_bucket_reason or "enabled",
                    "observed_accepted": int(stats["observed_accepted"]),
                    "observed_rejected": int(stats["observed_rejected"]),
                    "observed_proposed": int(stats.get("observed_proposed", 0)),
                    "observed_seeded_main_steps": int(stats.get("observed_seeded_main_steps", 0)),
                    "observed_partial_rows": int(stats.get("observed_partial_rows", 0)),
                    "acceptance_rate": accept_rate,
                    "confidence_min_accept_rate": self.mtp_min_accept_rate,
                    "acceptance_ready": bool(stats.get("acceptance_ready", verified >= self.mtp_min_accept_samples)),
                    "probe_steps_required": self.mtp_probe_steps,
                    "probe_complete": bool(stats.get("probe_complete", int(stats["spec_latency_steps"]) >= self.mtp_probe_steps)),
                    "baseline_ms_per_token": baseline_ms,
                    "spec_ms_per_token": spec_ms,
                    "baseline_latency_steps": int(stats["baseline_latency_steps"]),
                    "spec_latency_steps": int(stats["spec_latency_steps"]),
                    "latency_ready": bool(stats.get("latency_ready", False)),
                    "measured_speedup": measured_speedup,
                    "active": key == self.mtp_active_bucket_key,
                }
            )
        return {
            "enabled": bool(self.mtp_scheduler_gate_enabled),
            "active_bucket": self._bucket_key_to_report(self.mtp_active_bucket_key),
            "confidence_min_accept_rate": self.mtp_min_accept_rate,
            "probe_steps": self.mtp_probe_steps,
            "buckets": buckets,
        }

    @staticmethod
    def _bucket_key_to_report(
        key: tuple[int, int, str, str, int | None, int] | None,
    ) -> dict[str, object] | None:
        if key is None:
            return None
        batch_size, active_decode_rows, dtype, backend, max_blocks_per_seq, num_speculative_tokens = key
        return {
            "physical_batch_size": batch_size,
            "active_decode_rows": active_decode_rows,
            "dtype": dtype,
            "backend": backend,
            "max_blocks_per_seq": max_blocks_per_seq,
            "num_speculative_tokens": num_speculative_tokens,
        }

    def preempt(self, seq: Sequence):
        """Preempt a sequence (move back to waiting)."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

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
                "required_blocks": int((len(seq) + self.block_size - 1) // self.block_size),
            }

        running = [seq_snapshot(seq) for seq in list(self.running)[:8]]
        waiting = [seq_snapshot(seq) for seq in list(self.waiting)[:8]]
        return (
            "No sequence can be scheduled; KV cache capacity is exhausted "
            f"stats={stats} max_num_batched_tokens={self.max_num_batched_tokens} "
            f"max_num_seqs={self.max_num_seqs} max_num_resident_seqs={self.max_num_resident_seqs} "
            f"max_blocks_per_seq={self.max_blocks_per_seq} running={running} waiting={waiting}"
        )

    def postprocess(
        self, 
        seqs: List[Sequence], 
        token_ids: List[int | List[int]],
        prefill_chunk_lengths: List[int] | None = None,
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

        if prefill_chunk_lengths is None:
            prefill_chunk_lengths = [0] * len(seqs)
        if len(prefill_chunk_lengths) != len(seqs):
            raise ValueError("prefill_chunk_lengths must align with scheduled sequences")
        use_device_carry = self.device_token_carry
        
        for seq, generated, prefill_chunk_len in zip(seqs, token_ids, prefill_chunk_lengths):
            is_prefill_chunk = prefill_chunk_len > 0
            if prefill_chunk_len > 0:
                seq.num_cached_tokens = min(seq.num_prompt_tokens, seq.num_cached_tokens + prefill_chunk_len)

            generated_tokens = generated if isinstance(generated, list) else [generated]
            finished = False

            for idx, token_id in enumerate(generated_tokens):
                device_token = (
                    use_device_carry
                    and _is_device_token(token_id)
                    and seq.ignore_eos
                )
                if device_token:
                    seq.append_token_device(token_id)
                    is_eos = False
                else:
                    token_id = int(token_id)
                    seq.append_token(token_id)
                    is_eos = (token_id == self.eos)
                self.last_num_generated_tokens += 1

                if idx < len(generated_tokens) - 1:
                    self.block_manager.commit_processed_token(seq)

                # Check termination conditions. The final prefill chunk can
                # emit the first completion token, so it must participate in
                # max-token/EOS termination. Use >= to avoid leaking requests
                # if a path emits more than one token in a step.
                is_max_tokens = (seq.num_completion_tokens >= seq.max_tokens)

                if (not seq.ignore_eos and is_eos) or is_max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    finished = True
                    break

            finished_flags.append(finished)
        
        return finished_flags
