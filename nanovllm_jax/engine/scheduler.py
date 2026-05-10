"""Scheduler for continuous batching."""

import os
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
        self.block_size = config.block_size
        self.enable_prefix_cache_execution = not getattr(config, "linear_attn_layers", ())
        self.prefill_buckets = tuple(getattr(config, "prefill_buckets", ()))
        self.batch_size_buckets = tuple(getattr(config, "batch_size_buckets", ()))
        self.prefill_chunk_budget = (
            max(self.prefill_buckets)
            if self.prefill_buckets
            else max(64, self.max_num_batched_tokens if self.max_num_batched_tokens > 0 else 64)
        )
        self.decode_lookahead_tokens = max(1, 1 + int(getattr(config, "num_speculative_tokens", 0) or 0))
        self.num_speculative_tokens = max(0, int(getattr(config, "num_speculative_tokens", 0) or 0))
        self.mtp_min_accept_rate = float(os.environ.get("NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE", "0") or "0")
        self.mtp_min_accept_samples = int(os.environ.get("NANO_VLLM_JAX_MTP_MIN_ACCEPT_SAMPLES", "8") or "8")
        self.mtp_min_speedup = float(os.environ.get("NANO_VLLM_JAX_MTP_MIN_SPEEDUP", "1.0") or "1.0")
        self.mtp_scheduler_gate_enabled = (
            self.num_speculative_tokens > 0
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
        self.mtp_admission_enabled = True
        self.mtp_admission_reason = "enabled"
        self.mtp_latency_alpha = float(os.environ.get("NANO_VLLM_JAX_MTP_LATENCY_ALPHA", "0.2") or "0.2")
        self.mtp_latency_min_steps = int(os.environ.get("NANO_VLLM_JAX_MTP_LATENCY_MIN_STEPS", "2") or "2")
        self.mtp_baseline_ms_per_token: float | None = None
        self.mtp_spec_ms_per_token: float | None = None
        self.mtp_baseline_latency_steps = 0
        self.mtp_spec_latency_steps = 0
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, 
            config.block_size
        )
        # Override sequence block size
        Sequence.block_size = config.block_size
        
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

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
        
        # Phase 1: Prefill - schedule new/waiting sequences and unfinished
        # prompt tails from already-allocated running sequences.
        while num_seqs < self.max_num_seqs:
            if self.waiting:
                seq = self.waiting.popleft()
                from_waiting = True
            else:
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
                if not from_waiting:
                    self.running.append(seq)
                continue
            chunk_len = min(remaining_tokens, self.prefill_chunk_budget)
            if num_batched_tokens + chunk_len > self.max_num_batched_tokens:
                available = self.max_num_batched_tokens - num_batched_tokens
                if available <= 0:
                    if from_waiting:
                        self.waiting.appendleft(seq)
                    else:
                        scheduled_running.append(seq)
                    break
                chunk_len = min(chunk_len, available)
            
            # Check constraints
            if from_waiting:
                if not self.block_manager.can_allocate(seq):
                    self.waiting.appendleft(seq)
                    break
            
            # Allocate and schedule
            if from_waiting:
                self.block_manager.allocate(seq)
                if not self.enable_prefix_cache_execution:
                    seq.num_cached_tokens = 0
            seq.status = SequenceStatus.RUNNING
            mtp_admitted = self.should_admit_mtp(
                seq,
                batch_size_bucket=self._select_batch_size_bucket(num_seqs + 1),
                active_decode_rows=0,
            )
            seq.mtp_admitted = mtp_admitted
            seq.mtp_admission_reason = self.mtp_admission_reason if mtp_admitted else "scheduler_gate"

            num_seqs += 1
            chunk_len = min(chunk_len, remaining_tokens)
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
        while self.running and num_seqs < self.max_num_seqs and running_candidates < running_budget:
            running_candidates += 1
            seq = self.running.popleft()
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                self.running.append(seq)
                continue
            
            # Ensure we can append
            mtp_admitted = self.should_admit_mtp(
                seq,
                for_decode=True,
                batch_size_bucket=self._select_batch_size_bucket(num_seqs + 1),
                active_decode_rows=num_seqs + 1,
            )
            seq.mtp_admitted = mtp_admitted
            seq.mtp_admission_reason = self.mtp_admission_reason if mtp_admitted else "scheduler_gate"
            lookahead_tokens = min(
                1 + (self.num_speculative_tokens if mtp_admitted else 0),
                max(1, seq.max_tokens - seq.num_completion_tokens),
            )
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
        prefill_chunk_lens: List[int] | None = None,
    ) -> ScheduledBatch:
        """Build the canonical engine batch contract for one step."""
        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []
        prefill_is_final: List[bool] = []

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
            num_decode_tokens=0 if is_prefill else sum(query_lens),
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
            prefill_is_final=prefill_is_final if is_prefill else None,
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
        delta_accepted = max(0, accepted - self.mtp_stats_seen_accepted)
        delta_rejected = max(0, rejected - self.mtp_stats_seen_rejected)
        self.mtp_stats_seen_accepted = accepted
        self.mtp_stats_seen_rejected = rejected
        attempted_spec = (delta_accepted + delta_rejected) > 0

        if is_decode and elapsed_seconds is not None and emitted_tokens > 0:
            ms_per_token = (elapsed_seconds * 1000.0) / emitted_tokens
            if attempted_spec:
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
        if delta_accepted == 0 and delta_rejected == 0:
            self._update_mtp_admission_decision(bucket_stats)
            self._sync_legacy_mtp_fields(key, bucket_stats)
            return

        bucket_stats["observed_accepted"] = int(bucket_stats["observed_accepted"]) + delta_accepted
        bucket_stats["observed_rejected"] = int(bucket_stats["observed_rejected"]) + delta_rejected
        self._update_mtp_admission_decision(bucket_stats)
        self._sync_legacy_mtp_fields(key, bucket_stats)

    @staticmethod
    def _ewma(current: float | None, value: float, alpha: float) -> float:
        if current is None:
            return value
        return (1.0 - alpha) * current + alpha * value

    def _update_mtp_admission_decision(self, stats: dict[str, object]) -> None:
        verified = int(stats["observed_accepted"]) + int(stats["observed_rejected"])
        acceptance_ok = True
        if verified < self.mtp_min_accept_samples:
            acceptance_ok = True
        else:
            accept_rate = int(stats["observed_accepted"]) / max(1, verified)
            acceptance_ok = accept_rate >= self.mtp_min_accept_rate

        latency_ok = True
        if (
            int(stats["baseline_latency_steps"]) >= self.mtp_latency_min_steps
            and int(stats["spec_latency_steps"]) >= self.mtp_latency_min_steps
            and stats["baseline_ms_per_token"] is not None
            and stats["spec_ms_per_token"] is not None
        ):
            measured_speedup = float(stats["baseline_ms_per_token"]) / max(
                1e-9,
                float(stats["spec_ms_per_token"]),
            )
            latency_ok = measured_speedup >= self.mtp_min_speedup

        stats["admission_enabled"] = acceptance_ok and latency_ok
        if not acceptance_ok:
            stats["admission_reason"] = "low_acceptance"
        elif not latency_ok:
            stats["admission_reason"] = "low_throughput"
        elif verified < self.mtp_min_accept_samples:
            stats["admission_enabled"] = True
            stats["admission_reason"] = "warming"
        else:
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
                    "observed_accepted": int(stats["observed_accepted"]),
                    "observed_rejected": int(stats["observed_rejected"]),
                    "acceptance_rate": accept_rate,
                    "baseline_ms_per_token": baseline_ms,
                    "spec_ms_per_token": spec_ms,
                    "baseline_latency_steps": int(stats["baseline_latency_steps"]),
                    "spec_latency_steps": int(stats["spec_latency_steps"]),
                    "measured_speedup": measured_speedup,
                    "active": key == self.mtp_active_bucket_key,
                }
            )
        return {
            "enabled": bool(self.mtp_scheduler_gate_enabled),
            "active_bucket": self._bucket_key_to_report(self.mtp_active_bucket_key),
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
        
        for seq, generated, prefill_chunk_len in zip(seqs, token_ids, prefill_chunk_lengths):
            is_prefill_chunk = prefill_chunk_len > 0
            if prefill_chunk_len > 0:
                seq.num_cached_tokens = min(seq.num_prompt_tokens, seq.num_cached_tokens + prefill_chunk_len)

            generated_tokens = generated if isinstance(generated, list) else [generated]
            finished = False

            for idx, token_id in enumerate(generated_tokens):
                seq.append_token(int(token_id))
                self.last_num_generated_tokens += 1

                if idx < len(generated_tokens) - 1:
                    self.block_manager.commit_processed_token(seq)

                # Check termination conditions. The final prefill chunk can
                # emit the first completion token, so it must participate in
                # max-token/EOS termination. Use >= to avoid leaking requests
                # if a path emits more than one token in a step.
                is_eos = (token_id == self.eos)
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
