"""Persistent accelerator-state runner.

Owns:
    KV cache arrays, GDN hybrid-state slots, resident decode metadata, device
    token carry, and compile-bucket lookup.
Receives:
    Host-only ``SchedulePlan`` objects from the scheduler.
Returns:
    Generated token ids or device token references for scheduled sequences.
Invariant:
    The runner does not decide which requests should run; it only executes the
    already-built batch and advances resident device state.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
from functools import partial
from dataclasses import dataclass, replace

from nanovllm_jax.ops import ServingOps, ServingOpsProtocol, resolve_kv_cache_spec
from nanovllm_jax.batch import SchedulePlan
from nanovllm_jax.config import RuntimeSpec
from nanovllm_jax.device_batch import BatchMaterializer, DeviceBatch, HostBatch
from nanovllm_jax.executor import ModelExecutor
from nanovllm_jax.model import ModelParams
from nanovllm_jax.mtp import MTPParams, MTPState
from nanovllm_jax.output import DeviceTokenRef
from nanovllm_jax.speculation import VerificationResult
from nanovllm_jax.routes import (
    BatchPhase,
    ExecutionPlan,
    RouteCapability,
    RouteRequest,
    TokenMode,
    WarmupScenario,
    decode_warmup_scenarios,
    select_route,
    validate_executor,
)
from nanovllm_jax.step import RunResult
from nanovllm_jax.sequence import SamplingParams, Sequence
from nanovllm_jax.block_manager import PrefixCacheEntry
from nanovllm_jax.cache import (
    HybridLayerState,
    KVCacheStorage,
    KVCacheSpec,
    init_hybrid_state,
)


def _block_until_ready_tree(value: object) -> None:
    ready = getattr(value, "block_until_ready", None)
    if callable(ready):
        ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready_tree(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready_tree(item)
        return
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if dataclass_fields is not None:
        for name in dataclass_fields:
            _block_until_ready_tree(getattr(value, name))
        return
    for leaf in jax.tree_util.tree_leaves(value):
        leaf_ready = getattr(leaf, "block_until_ready", None)
        if callable(leaf_ready):
            leaf_ready()


def _nbytes(value: object) -> int:
    if hasattr(value, "size") and hasattr(value, "dtype"):
        return int(value.size) * int(value.dtype.itemsize)
    if isinstance(value, dict):
        return sum(_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(item) for item in value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return sum(_nbytes(getattr(value, name)) for name in fields)
    return 0


def _int32_device_vector(value) -> jnp.ndarray:
    """Return a 1D int32 device vector without re-wrapping existing int32 arrays."""

    if hasattr(value, "dtype") and getattr(value, "dtype", None) == jnp.dtype(jnp.int32):
        if getattr(value, "ndim", None) == 1:
            return value
        return value.reshape(-1)
    return jnp.asarray(value, dtype=jnp.int32).reshape(-1)


@dataclass(frozen=True)
class PrefixStateSnapshot:
    """Runner-owned state tied to one exact content-addressed prefix."""

    prefix_hash: int
    token_count: int
    state: HybridLayerState


class ModelRunner:
    """Canonical engine runner built around ModelExecutor.forward_step()."""

    def __init__(
        self,
        config: RuntimeSpec,
        params: ModelParams,
        ops: ServingOpsProtocol | None = None,
        *,
        mtp_params: MTPParams | None = None,
    ):
        self.config = config
        self.params = params
        if (config.drafter is None) != (mtp_params is None):
            raise ValueError("MTP configuration and predictor weights must be supplied together")
        self.backend = ops if ops is not None else ServingOps(config.kernels)
        self.executor = ModelExecutor(
            config,
            params,
            mtp_params=mtp_params,
            backend=self.backend,
        )
        validate_executor(self.executor)
        self.block_size = config.capacity.block_size

        max_seqs = config.capacity.max_num_resident_seqs
        kv_spec = resolve_kv_cache_spec(
            KVCacheSpec(
                num_layers=config.model.num_hidden_layers,
                num_blocks=config.capacity.num_kvcache_blocks,
                block_size=config.capacity.block_size,
                num_kv_heads=config.model.num_key_value_heads,
                head_dim=config.model.head_dim,
                dtype=config.compile.jax_dtype(),
                max_kv_cache_bytes=config.capacity.max_kv_cache_bytes,
            ),
            config.kernels,
        )
        if kv_spec.num_blocks != config.capacity.num_kvcache_blocks:
            raise ValueError(
                "RuntimeSpec cache capacity was not finalized before runner construction"
            )
        self.max_blocks_per_seq = config.capacity.max_blocks_per_seq
        self.execution = config.compile.execution
        self.greedy_token_fastpath = config.kernels.greedy_token_fastpath
        self.sampled_token_fastpath = config.kernels.sampled_token_fastpath
        self.device_token_carry = config.kernels.device_token_carry
        self.static_decode_metadata = config.kernels.static_decode_metadata
        self.resident_decode_metadata = config.kernels.resident_decode_metadata
        self.static_decode_seq_lens_carry = config.kernels.static_decode_seq_lens_carry
        self.batch_materializer = BatchMaterializer(
            execution=self.execution,
            device_token_carry=self.device_token_carry,
            static_decode_metadata=self.static_decode_metadata,
            resident_decode_metadata=self.resident_decode_metadata,
            static_decode_seq_lens_carry=self.static_decode_seq_lens_carry,
        )

        self.cache_storage = self.backend.allocate_kv_cache(
            kv_spec,
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
        )
        self.mtp_state: MTPState | None = None
        if config.drafter is not None:
            mtp_kv_spec = resolve_kv_cache_spec(
                KVCacheSpec(
                    num_layers=1,
                    num_blocks=kv_spec.num_blocks,
                    block_size=kv_spec.block_size,
                    num_kv_heads=kv_spec.num_kv_heads,
                    head_dim=kv_spec.head_dim,
                    dtype=kv_spec.dtype,
                ),
                config.kernels,
            )
            if mtp_kv_spec.num_blocks != kv_spec.num_blocks:
                raise ValueError("MTP and target KV caches must share block capacity")
            self.mtp_state = MTPState(
                cache_storage=self.backend.allocate_kv_cache(
                    mtp_kv_spec,
                    max_seqs=max_seqs,
                    max_blocks_per_seq=self.max_blocks_per_seq,
                ),
                draft_token_ids=jnp.zeros(
                    (max_seqs, config.drafter.width),
                    dtype=jnp.int32,
                ),
            )
        kv_bytes = sum(self.persistent_kv_bytes().values())
        if kv_bytes > config.capacity.max_kv_cache_bytes:
            raise RuntimeError(
                f"persistent KV allocations need {kv_bytes} bytes but the "
                f"configured cap is {config.capacity.max_kv_cache_bytes}"
            )
        self._max_hybrid_slots = max_seqs
        self._hybrid_slots: dict[int, int] = {}
        self._free_hybrid_slots: list[int] = list(range(max_seqs))
        self._zeroed_hybrid_slots: set[int] = set(range(max_seqs))
        self._prefix_hybrid_states: dict[int, PrefixStateSnapshot] = {}
        self._prefix_hybrid_state_capacity = (
            max_seqs if config.capacity.prefix_cache and config.model.linear_attn_layers else 0
        )
        self._next_prefix_hybrid_state_handle = 0

        empty_hybrid_state = init_hybrid_state(
            config=config.model,
            batch_size=1,
            dtype=config.compile.jax_dtype(),
        )
        self._empty_hybrid_state = empty_hybrid_state
        self._hybrid_state_table = init_hybrid_state(
            self.config.model,
            batch_size=max_seqs,
            dtype=self.config.compile.jax_dtype(),
        )
        self._resident_block_tables = jnp.zeros(
            (max_seqs, self.max_blocks_per_seq),
            dtype=jnp.int32,
        )
        self._resident_seq_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_block_tables_host: list[tuple[int, ...]] = [
            tuple(0 for _ in range(self.max_blocks_per_seq)) for _ in range(max_seqs)
        ]
        self._resident_block_counts_host: list[int] = [0 for _ in range(max_seqs)]
        self._resident_seq_lens_host: list[int] = [0 for _ in range(max_seqs)]
        self._resident_last_tokens = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_rng_counters = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_rng_counter_reset_slots: set[int] = set()
        self._resident_last_tokens_stale_seq_ids: set[int] = set()
        self._device_token_carry_seq_ids: tuple[int, ...] | None = None
        self._device_token_carry_tokens = None
        self._device_token_carry_by_seq_id: dict[int, DeviceTokenRef] = {}
        self._device_seq_lens_carry_seq_ids: tuple[int, ...] | None = None
        self._device_seq_lens_carry = None
        self._hybrid_slot_ids_device_cache: dict[tuple[int, ...], jax.Array] = {}
        self._prefill_final_flags_device_cache: dict[tuple[bool, ...], jax.Array] = {}
        self._resident_metadata_scatter_cache: dict[tuple[Any, ...], Any] = {}
        self._resident_update_slots_device_cache: dict[tuple[int, ...], jax.Array] = {}
        self._sample_fn = jax.jit(self._sample_logits)
        self._warmup_compiled = False
        self._mtp_ready_seq_ids: set[int] = set()
        self.speculation_stats = {
            "drafted": 0,
            "accepted": 0,
            "rejected": 0,
            "bonus": 0,
            "target_tokens": 0,
        }

    def reset_speculation_stats(self) -> None:
        for name in self.speculation_stats:
            self.speculation_stats[name] = 0

    def memory_bytes(self) -> dict[str, int]:
        """Return persistent allocations and bounded dynamic-state capacity."""

        return {
            **self.persistent_kv_bytes(),
            "hybrid_state": _nbytes((self._empty_hybrid_state, self._hybrid_state_table)),
            "prefix_hybrid_state_current": _nbytes(
                tuple(snapshot.state for snapshot in self._prefix_hybrid_states.values())
            ),
            "prefix_hybrid_state_capacity": (
                self._prefix_hybrid_state_capacity * _nbytes(self._empty_hybrid_state)
            ),
            "resident_metadata": _nbytes(
                (
                    self._resident_block_tables,
                    self._resident_seq_lens,
                    self._resident_last_tokens,
                    self._resident_rng_counters,
                )
            ),
            "mtp_draft_tokens": (
                _nbytes(self.mtp_state.draft_token_ids) if self.mtp_state is not None else 0
            ),
        }

    def persistent_kv_bytes(self) -> dict[str, int]:
        """Enumerate every persistent paged KV owner."""

        allocations = {"target_kv": _nbytes(self.cache_storage)}
        if self.mtp_state is not None:
            allocations["predictor_kv"] = _nbytes(self.mtp_state.cache_storage)
        return allocations

    def _warmup_sequences(
        self,
        batch_size: int,
        *,
        temperature: float,
        ignore_eos: bool = True,
    ) -> list[Sequence]:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max(2, self.config.kernels.greedy_decode_burst_steps + 1),
            ignore_eos=ignore_eos,
        )
        return [
            Sequence([0], params, seq_id=row, block_size=self.block_size)
            for row in range(batch_size)
        ]

    def _prime_warmup_decode_batch(self, batch: DeviceBatch) -> None:
        if not (self.device_token_carry and self.static_decode_metadata and batch.host.seq_ids):
            return
        self._batch_hybrid_slot_ids(batch)
        tokens = jnp.zeros((len(batch.host.seq_ids), 1), dtype=jnp.int32)
        self._device_token_carry_seq_ids = tuple(batch.host.seq_ids)
        self._device_token_carry_tokens = tokens
        self._device_token_carry_by_seq_id = {
            int(seq_id): DeviceTokenRef(tokens=tokens, row=row)
            for row, seq_id in enumerate(batch.host.seq_ids)
        }

    def _warmup_decode_inputs(
        self,
        batch_size: int,
        block_table_width: int,
        scenario: WarmupScenario,
    ) -> tuple[list[Sequence], DeviceBatch]:
        active_rows = batch_size if scenario.full_bucket else 1
        batch = self._dummy_batch(
            batch_size=active_rows,
            token_bucket=1,
            is_prefill=False,
            max_blocks_per_seq=block_table_width,
        )
        batch = self._pad_decode_batch_to_rows(batch, batch_size)
        batch = replace(
            batch,
            host=replace(
                batch.host,
                decode_steps=scenario.decode_steps,
                uses_static_decode_metadata=scenario.static_token_carry,
            ),
        )
        self._clear_device_token_carry()
        if scenario.static_token_carry:
            self._prime_warmup_decode_batch(batch)
        temperature = 1.0 if scenario.tokens is TokenMode.SAMPLED else 0.0
        seqs = self._warmup_sequences(
            active_rows,
            temperature=temperature,
            ignore_eos=scenario.ignore_eos,
        )
        return seqs, batch

    def _warm_route(
        self,
        seqs: list[Sequence],
        batch: DeviceBatch,
    ) -> ExecutionPlan:
        route = self._select_route(seqs, batch)
        outputs = self._run_execution_plan(route, seqs, batch)
        _block_until_ready_tree(
            (
                outputs,
                self.cache_storage,
                self._hybrid_state_table,
                self._resident_block_tables,
                self._resident_seq_lens,
                self._resident_last_tokens,
                self._resident_rng_counters,
                self.mtp_state,
            )
        )
        return route

    def warmup_compilation(
        self,
        max_prefill_len: int = 64,
        max_batch: int = 1,
        *,
        include_sampled_routes: bool = True,
        prefill_token_buckets: tuple[int, ...] | None = None,
        batch_size_buckets: tuple[int, ...] | None = None,
        decode_block_table_buckets: tuple[int, ...] | None = None,
    ):
        """Compile the configured prefill/decode buckets for the promoted path."""
        summary: dict[str, Any] = {
            "mode": "generic_bucket_startup",
            "execution": self.execution,
            "prefill_buckets": [],
            "batch_size_buckets": [],
            "prefill_runs": [],
            "prefill_skipped": [],
            "decode_runs": [],
            "decode_skipped": [],
            "decode_block_table_buckets": [],
            "resident_metadata_scatter_runs": [],
            "sampled_token_fastpath_runs": [],
            "warmed_routes": [],
            "include_sampled_routes": bool(include_sampled_routes),
            "already_warmed": bool(self._warmup_compiled),
        }
        if self._warmup_compiled:
            return summary
        if self.execution not in {"decode-jit", "jit"}:
            self._warmup_compiled = True
            return summary

        prefill_buckets = tuple(int(bucket) for bucket in (prefill_token_buckets or ())) or (
            self.config.compile.prefill_token_buckets or (max_prefill_len,)
        )
        batch_buckets = tuple(int(bucket) for bucket in (batch_size_buckets or ())) or (
            self.config.compile.batch_size_buckets or (max_batch,)
        )
        decode_block_table_buckets = tuple(
            int(bucket) for bucket in (decode_block_table_buckets or ())
        ) or (self.config.compile.decode_block_table_buckets or (int(self.max_blocks_per_seq),))
        summary["prefill_buckets"] = list(prefill_buckets)
        summary["batch_size_buckets"] = list(batch_buckets)
        summary["decode_block_table_buckets"] = [int(width) for width in decode_block_table_buckets]

        warm_sampled = bool(include_sampled_routes) and self.sampled_token_fastpath
        greedy_decode_burst_steps = self.config.kernels.greedy_decode_burst_steps

        for token_bucket in prefill_buckets:
            if self.execution != "jit":
                break
            for batch_size in batch_buckets:
                dense_prefill_tokens = int(batch_size) * int(token_bucket)
                max_batched_tokens = self.config.capacity.max_num_batched_tokens
                packed_prefill_layout = self.config.compile.prefill_layout == "packed"
                if (
                    not packed_prefill_layout
                    and max_batched_tokens > 0
                    and dense_prefill_tokens > max_batched_tokens
                ):
                    summary["prefill_skipped"].append(
                        {
                            "batch_size": int(batch_size),
                            "token_bucket": int(token_bucket),
                            "dense_prefill_tokens": dense_prefill_tokens,
                            "max_num_batched_tokens": max_batched_tokens,
                            "reason": "dense_prefill_tokens_exceed_budget",
                        }
                    )
                    continue
                if packed_prefill_layout and int(token_bucket) < int(batch_size):
                    summary["prefill_skipped"].append(
                        {
                            "batch_size": int(batch_size),
                            "token_bucket": int(token_bucket),
                            "reason": "token_bucket_smaller_than_active_batch",
                        }
                    )
                    continue
                capacity_reason = self._warmup_capacity_reason(
                    int(batch_size),
                    int(self.max_blocks_per_seq),
                    is_prefill=True,
                    token_bucket=int(token_bucket),
                )
                if capacity_reason is not None:
                    summary["prefill_skipped"].append(
                        {
                            "batch_size": int(batch_size),
                            "token_bucket": int(token_bucket),
                            "reason": capacity_reason,
                        }
                    )
                    continue
                batch = self._dummy_batch(
                    batch_size=batch_size,
                    token_bucket=token_bucket,
                    is_prefill=True,
                )
                route = self._warm_route(
                    self._warmup_sequences(batch_size, temperature=0.0),
                    batch,
                )
                summary["warmed_routes"].append(route.kind.value)
                summary["prefill_runs"].append(
                    {
                        "batch_size": int(batch_size),
                        "token_bucket": int(token_bucket),
                        "tokens_shape": list(batch.tokens.shape),
                        "block_tables_shape": list(batch.block_tables.shape),
                        "num_prefill_tokens": int(batch.num_prefill_tokens),
                        "route": route.kind.value,
                    }
                )
                if warm_sampled:
                    sampled_route = self._warm_route(
                        self._warmup_sequences(batch_size, temperature=1.0),
                        batch,
                    )
                    summary["warmed_routes"].append(sampled_route.kind.value)
                    summary["sampled_token_fastpath_runs"].append(
                        {
                            "kind": "prefill",
                            "batch_size": int(batch_size),
                            "token_bucket": int(token_bucket),
                            "route": sampled_route.kind.value,
                        }
                    )

        def record_decode(
            batch_size: int,
            route: ExecutionPlan,
            batch: DeviceBatch,
            scenario: WarmupScenario,
        ) -> None:
            self._sample_fn(
                jnp.zeros((batch_size, self.config.model.vocab_size), dtype=jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            ).block_until_ready()
            summary["warmed_routes"].append(route.kind.value)
            summary["decode_runs"].append(
                {
                    "batch_size": int(batch_size),
                    "tokens_shape": list(batch.tokens.shape),
                    "block_tables_shape": list(batch.block_tables.shape),
                    "num_decode_tokens": int(batch.num_decode_tokens),
                    "route": route.kind.value,
                    "scenario": scenario.name,
                    "decode_steps": int(route.decode_steps),
                }
            )

        # Warm speculative shapes while the proposals seeded by prefill are
        # still live. Ordinary decode deliberately invalidates them below.
        if self.config.drafter is not None:
            for batch_size in batch_buckets:
                for block_table_width in decode_block_table_buckets:
                    capacity_reason = self._warmup_capacity_reason(
                        int(batch_size),
                        int(block_table_width),
                        is_prefill=False,
                        token_bucket=1,
                    )
                    if capacity_reason is not None:
                        summary["decode_skipped"].append(
                            {
                                "batch_size": int(batch_size),
                                "block_table_width": int(block_table_width),
                                "scenario": "speculative",
                                "reason": capacity_reason,
                            }
                        )
                        continue
                    scenario = WarmupScenario(
                        "speculative",
                        TokenMode.SPECULATIVE,
                        True,
                        True,
                        True,
                    )
                    seqs, batch = self._warmup_decode_inputs(
                        batch_size,
                        int(block_table_width),
                        scenario,
                    )
                    for seq in seqs:
                        seq.max_tokens = int(self.config.drafter.width) + 2
                    route = self._warm_route(seqs, batch)
                    record_decode(batch_size, route, batch, scenario)

        for batch_size in batch_buckets:
            for block_table_width in decode_block_table_buckets:
                capacity_reason = self._warmup_capacity_reason(
                    int(batch_size),
                    int(block_table_width),
                    is_prefill=False,
                    token_bucket=1,
                )
                if capacity_reason is not None:
                    summary["decode_skipped"].append(
                        {
                            "batch_size": int(batch_size),
                            "block_table_width": int(block_table_width),
                            "scenario": "all_non_speculative",
                            "reason": capacity_reason,
                        }
                    )
                    continue
                scenarios = decode_warmup_scenarios(
                    static_token_carry=bool(
                        self.greedy_token_fastpath
                        and self.device_token_carry
                        and self.static_decode_metadata
                    ),
                    sparse_bucket=batch_size > 1,
                    include_sampled=warm_sampled,
                    burst_steps=(greedy_decode_burst_steps if self.greedy_token_fastpath else 1),
                )

                for scenario in scenarios:
                    seqs, batch = self._warmup_decode_inputs(
                        batch_size,
                        int(block_table_width),
                        scenario,
                    )
                    route = self._warm_route(seqs, batch)
                    record_decode(batch_size, route, batch, scenario)
                    if scenario.tokens is not TokenMode.SAMPLED:
                        continue
                    summary["sampled_token_fastpath_runs"].append(
                        {
                            "kind": "decode",
                            "batch_size": int(batch_size),
                            "block_tables_shape": list(batch.block_tables.shape),
                            "route": route.kind.value,
                        }
                    )
        if self.resident_decode_metadata:
            for row_count in range(1, int(max(batch_buckets)) + 1):
                if row_count > int(self._resident_block_tables.shape[0]):
                    break
                slots = jnp.arange(row_count, dtype=jnp.int32)
                block_rows = jnp.zeros(
                    (row_count, int(self._resident_block_tables.shape[1])), dtype=jnp.int32
                )
                seq_lens = jnp.zeros((row_count,), dtype=jnp.int32)
                token_rows = jnp.arange(row_count, dtype=jnp.int32)
                last_tokens = jnp.zeros((row_count, 1), dtype=jnp.int32)
                self._resident_block_tables = self._scatter_resident_block_table_rows(
                    self._resident_block_tables,
                    slots,
                    block_rows,
                )
                self._resident_seq_lens = self._scatter_resident_seq_lens(
                    self._resident_seq_lens,
                    slots,
                    seq_lens,
                )
                self._resident_last_tokens = self._scatter_resident_last_tokens(
                    self._resident_last_tokens,
                    slots,
                    last_tokens,
                    token_rows,
                )
                _block_until_ready_tree(self._resident_block_tables)
                _block_until_ready_tree(self._resident_seq_lens)
                _block_until_ready_tree(self._resident_last_tokens)
                summary["resident_metadata_scatter_runs"].append(
                    {
                        "row_count": int(row_count),
                        "block_rows_shape": list(block_rows.shape),
                        "seq_lens_shape": list(seq_lens.shape),
                        "last_tokens_shape": list(last_tokens.shape),
                    }
                )
        summary["warmed_routes"] = sorted(set(summary["warmed_routes"]))
        self._reset_runtime_state_after_warmup()
        _block_until_ready_tree(
            (
                self._hybrid_state_table,
                self._resident_block_tables,
                self._resident_seq_lens,
                self._resident_last_tokens,
                self._resident_rng_counters,
                self.mtp_state,
            )
        )
        summary["state_reset_after_warmup"] = True
        self._warmup_compiled = True
        return summary

    def _dummy_batch(
        self,
        *,
        batch_size: int,
        token_bucket: int,
        is_prefill: bool,
        max_blocks_per_seq: int | None = None,
    ) -> DeviceBatch:
        block_table_width = int(max_blocks_per_seq or self.max_blocks_per_seq)
        capacity_reason = self._warmup_capacity_reason(
            batch_size,
            block_table_width,
            is_prefill=is_prefill,
            token_bucket=token_bucket,
        )
        if capacity_reason is not None:
            raise ValueError(capacity_reason)
        seq_lens, live_block_counts = self._warmup_live_layout(
            batch_size,
            block_table_width,
            is_prefill=is_prefill,
            token_bucket=token_bucket,
        )
        block_tables = []
        next_block = 0
        for live_blocks in live_block_counts:
            live = list(range(next_block, next_block + live_blocks))
            next_block += live_blocks
            block_tables.append(live + [0] * (block_table_width - live_blocks))
        if is_prefill and self.config.compile.prefill_layout == "packed":
            token_bucket = int(token_bucket)
            query_lens = seq_lens
            query_start_loc = [0]
            packed_positions = []
            token_row_ids = []
            for row, qlen in enumerate(query_lens):
                query_start_loc.append(query_start_loc[-1] + qlen)
                packed_positions.extend(range(qlen))
                token_row_ids.extend([row] * qlen)
            return DeviceBatch(
                tokens=jnp.zeros((1, token_bucket), dtype=jnp.int32),
                positions=jnp.array([packed_positions], dtype=jnp.int32),
                seq_ids=jnp.arange(batch_size, dtype=jnp.int32),
                query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
                is_prefill=True,
                num_prefill_tokens=token_bucket,
                num_decode_tokens=0,
                block_tables=jnp.array(block_tables, dtype=jnp.int32),
                seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
                host=HostBatch(
                    seq_ids=tuple(range(batch_size)),
                    query_lens=tuple(query_lens),
                    seq_lens=tuple(seq_lens),
                    block_tables=tuple(tuple(int(block) for block in row) for row in block_tables),
                ),
                packed_prefill=True,
                token_row_ids=jnp.array([token_row_ids], dtype=jnp.int32),
            )

        query_len = int(token_bucket)
        query_lens = [query_len if is_prefill else 1] * batch_size
        query_start_loc = [0]
        for qlen in query_lens:
            query_start_loc.append(query_start_loc[-1] + qlen)
        positions = (
            [list(range(query_len)) for _ in range(batch_size)]
            if is_prefill
            else [[seq_len - 1] for seq_len in seq_lens]
        )
        return DeviceBatch(
            tokens=jnp.zeros((batch_size, query_len), dtype=jnp.int32),
            positions=jnp.array(positions, dtype=jnp.int32),
            seq_ids=jnp.arange(batch_size, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else batch_size,
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
            host=HostBatch(
                seq_ids=tuple(range(batch_size)),
                query_lens=tuple(query_lens),
                seq_lens=tuple(seq_lens),
                block_tables=tuple(tuple(int(block) for block in row) for row in block_tables),
            ),
        )

    def _warmup_live_layout(
        self,
        batch_size: int,
        block_table_width: int,
        *,
        is_prefill: bool,
        token_bucket: int,
    ) -> tuple[list[int], list[int]]:
        if is_prefill:
            if self.config.compile.prefill_layout == "packed":
                base, remainder = divmod(int(token_bucket), int(batch_size))
                seq_lens = [base + (1 if row < remainder else 0) for row in range(batch_size)]
            else:
                seq_lens = [int(token_bucket)] * int(batch_size)
            live_blocks = [
                max(1, (seq_len + self.block_size - 1) // self.block_size) for seq_len in seq_lens
            ]
            return seq_lens, live_blocks

        buckets = tuple(self.config.compile.decode_block_table_buckets)
        if block_table_width in buckets:
            previous = max(
                (bucket for bucket in buckets if bucket < block_table_width),
                default=0,
            )
            first_live_blocks = previous + 1
        else:
            first_live_blocks = block_table_width
        live_blocks = [first_live_blocks] + [1] * (int(batch_size) - 1)
        seq_lens = [(blocks - 1) * self.block_size + 1 for blocks in live_blocks]
        return seq_lens, live_blocks

    def _warmup_capacity_reason(
        self,
        batch_size: int,
        block_table_width: int,
        *,
        is_prefill: bool,
        token_bucket: int,
    ) -> str | None:
        _, live_blocks = self._warmup_live_layout(
            batch_size,
            block_table_width,
            is_prefill=is_prefill,
            token_bucket=token_bucket,
        )
        if any(blocks > int(block_table_width) for blocks in live_blocks):
            return (
                "warmup_sequence_exceeds_block_table: "
                f"block_table_width={block_table_width} live_blocks={live_blocks}"
            )
        required = sum(live_blocks)
        available = int(self.config.capacity.num_kvcache_blocks)
        if required <= available:
            return None
        return (
            "warmup_requires_disjoint_blocks: "
            f"batch_size={batch_size} block_table_width={block_table_width} "
            f"live_blocks={live_blocks} requires={required} available={available}"
        )

    def release(self, seq_ids: list[int]):
        """Release per-sequence hybrid state once a request is finished."""
        for seq_id in seq_ids:
            self._mtp_ready_seq_ids.discard(int(seq_id))
            slot = self._hybrid_slots.pop(seq_id, None)
            if slot is not None:
                self._free_hybrid_slots.append(slot)
                self._resident_block_tables_host[slot] = tuple(
                    0 for _ in range(self.max_blocks_per_seq)
                )
                self._resident_block_counts_host[slot] = 0
                self._resident_seq_lens_host[slot] = 0
                self._resident_rng_counter_reset_slots.add(int(slot))
            self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        carry_by_seq_id = self._device_token_carry_by_seq_id
        if carry_by_seq_id and any(seq_id in carry_by_seq_id for seq_id in seq_ids):
            finished_seq_ids = {int(seq_id) for seq_id in seq_ids}
            remaining_carry = {
                int(seq_id): token_ref
                for seq_id, token_ref in carry_by_seq_id.items()
                if int(seq_id) not in finished_seq_ids
            }
            if remaining_carry:
                self._device_token_carry_seq_ids = None
                self._device_token_carry_tokens = None
                self._device_token_carry_by_seq_id = remaining_carry
            else:
                self._clear_device_token_carry()

    def _reset_runtime_state_after_warmup(self) -> None:
        """Drop dummy warmup sequence state while keeping compiled executables."""
        self.reset_speculation_stats()
        self._mtp_ready_seq_ids.clear()
        if self.mtp_state is not None:
            self.mtp_state = MTPState(
                cache_storage=KVCacheStorage(
                    jnp.zeros_like(self.mtp_state.cache_storage.k_cache),
                    jnp.zeros_like(self.mtp_state.cache_storage.v_cache),
                ),
                draft_token_ids=jnp.zeros_like(self.mtp_state.draft_token_ids),
            )
        self._prefix_hybrid_states.clear()
        self._next_prefix_hybrid_state_handle = 0
        self._hybrid_slots.clear()
        self._free_hybrid_slots = list(range(self._max_hybrid_slots))
        self._zeroed_hybrid_slots = set()
        self._zero_hybrid_slots(tuple(range(self._max_hybrid_slots)))
        self._zeroed_hybrid_slots = set(range(self._max_hybrid_slots))
        self._clear_device_token_carry()
        self._resident_block_tables = jnp.zeros_like(self._resident_block_tables)
        self._resident_seq_lens = jnp.zeros_like(self._resident_seq_lens)
        self._resident_last_tokens = jnp.zeros_like(self._resident_last_tokens)
        self._resident_rng_counters = jnp.zeros_like(self._resident_rng_counters)
        self._resident_rng_counter_reset_slots.clear()
        self._resident_last_tokens_stale_seq_ids.clear()
        self._resident_block_tables_host = [
            tuple(0 for _ in range(self.max_blocks_per_seq)) for _ in range(self._max_hybrid_slots)
        ]
        self._resident_block_counts_host = [0 for _ in range(self._max_hybrid_slots)]
        self._resident_seq_lens_host = [0 for _ in range(self._max_hybrid_slots)]

    def _clear_device_token_carry(self) -> None:
        self._device_token_carry_seq_ids = None
        self._device_token_carry_tokens = None
        self._device_token_carry_by_seq_id = {}
        self._device_seq_lens_carry_seq_ids = None
        self._device_seq_lens_carry = None

    @staticmethod
    def _active_decode_rows_host(batch: DeviceBatch) -> list[int]:
        if not batch.host.seq_ids or not batch.host.query_lens:
            return []
        return [
            row
            for row, (seq_id, query_len) in enumerate(
                zip(batch.host.seq_ids, batch.host.query_lens)
            )
            if int(seq_id) >= 0 and int(query_len) > 0
        ]

    def _maybe_apply_device_token_carry(self, batch: DeviceBatch) -> DeviceBatch:
        static_decode_metadata = bool(batch.host.uses_static_decode_metadata)
        active_rows = self._active_decode_rows_host(batch)
        carry_enabled = self.device_token_carry
        if (
            not carry_enabled
            or batch.is_prefill
            or not self._device_token_carry_by_seq_id
            or not batch.host.seq_ids
            or batch.tokens.shape[1] != 1
        ):
            if static_decode_metadata:
                raise RuntimeError(
                    "static decode metadata requires a device-token carry for every active row"
                )
            return batch

        carried_seq_ids = self._device_token_carry_seq_ids
        carried_tokens = self._device_token_carry_tokens
        carried_seq_lens_ids = self._device_seq_lens_carry_seq_ids
        carried_seq_lens = self._device_seq_lens_carry
        use_seq_lens_carry = self.static_decode_seq_lens_carry
        tokens = batch.tokens
        seq_lens = batch.seq_lens

        if (
            carried_seq_ids is not None
            and tuple(batch.host.seq_ids) == carried_seq_ids
            and carried_tokens is not None
        ):
            token_array = jnp.asarray(carried_tokens, dtype=jnp.int32)
            if tuple(token_array.shape) == tuple(tokens.shape):
                tokens = token_array
                applied = True
            else:
                token_vector = _int32_device_vector(token_array)
                if token_vector.shape[0] == int(tokens.shape[0]):
                    tokens = jnp.reshape(token_vector, tokens.shape)
                    applied = True
                else:
                    applied = False
        else:
            applied = False

        missing_static_rows: list[int] = []
        if not applied:
            for row, seq_id in enumerate(batch.host.seq_ids):
                token_ref = self._device_token_carry_by_seq_id.get(int(seq_id))
                if token_ref is None:
                    if static_decode_metadata and row in active_rows:
                        missing_static_rows.append(row)
                    continue
                token_array = jnp.asarray(token_ref.tokens, dtype=jnp.int32)
                if token_array.ndim == 2 and token_array.shape[1] == 1:
                    tokens = tokens.at[row, 0].set(token_array[int(token_ref.row), 0])
                else:
                    token_vector = _int32_device_vector(token_array)
                    tokens = tokens.at[row, 0].set(token_vector[int(token_ref.row)])
                applied = True
        if missing_static_rows:
            raise RuntimeError(
                "static decode metadata is missing device-token carry rows "
                f"{tuple(missing_static_rows)}"
            )
        if not applied:
            if static_decode_metadata:
                raise RuntimeError("static decode metadata did not apply any device-token carry")
            return batch

        seq_lens_applied = False
        if use_seq_lens_carry:
            if (
                carried_seq_lens_ids is not None
                and tuple(batch.host.seq_ids) == carried_seq_lens_ids
                and carried_seq_lens is not None
            ):
                seq_lens_vector = _int32_device_vector(carried_seq_lens)
                if seq_lens_vector.shape[0] == int(seq_lens.shape[0]):
                    seq_lens = seq_lens_vector
                    seq_lens_applied = True
            if (
                not seq_lens_applied
                and carried_seq_lens is not None
                and carried_seq_lens_ids is not None
            ):
                seq_lens_vector = _int32_device_vector(carried_seq_lens)
                seq_id_to_row = {
                    int(seq_id): row for row, seq_id in enumerate(carried_seq_lens_ids)
                }
                for row, seq_id in enumerate(batch.host.seq_ids):
                    source_row = seq_id_to_row.get(int(seq_id))
                    if source_row is None:
                        continue
                    seq_lens = seq_lens.at[row].set(seq_lens_vector[source_row])
                    seq_lens_applied = True
        if (
            static_decode_metadata
            and use_seq_lens_carry
            and not seq_lens_applied
            and carried_seq_lens is not None
        ):
            raise RuntimeError("static decode metadata requires carried device seq_lens")
        return replace(batch, tokens=tokens, seq_lens=seq_lens)

    def _resident_slot_token_decode_ready(
        self,
        batch: DeviceBatch,
        *,
        active_rows: list[int],
    ) -> bool:
        if batch.is_prefill or not batch.host.seq_ids or not active_rows:
            return False
        carry_by_seq_id = self._device_token_carry_by_seq_id
        if not carry_by_seq_id:
            return False
        hybrid_slots = self._hybrid_slots
        stale_seq_ids = self._resident_last_tokens_stale_seq_ids
        for row in active_rows:
            seq_id = int(batch.host.seq_ids[row])
            if seq_id in stale_seq_ids:
                return False
            if seq_id < 0 or seq_id not in carry_by_seq_id or seq_id not in hybrid_slots:
                return False
        return True

    def _resident_slot_token_dense_decode_ready(
        self,
        batch: DeviceBatch,
        *,
        active_rows: list[int],
    ) -> bool:
        if not self._resident_slot_token_decode_ready(batch, active_rows=active_rows):
            return False
        batch_size = int(batch.tokens.shape[0])
        if active_rows != list(range(batch_size)):
            return False
        query_lens = (
            list(batch.host.query_lens)
            if batch.host.query_lens
            else [int(x) for x in batch.query_lens[:batch_size].tolist()]
        )
        if len(query_lens) < batch_size or any(
            int(query_lens[row]) != 1 for row in range(batch_size)
        ):
            return False
        seq_ids = list(batch.host.seq_ids or ())
        if len(seq_ids) != batch_size:
            return False
        hybrid_slots = self._hybrid_slots
        slot_values = [int(hybrid_slots.get(int(seq_id), -1)) for seq_id in seq_ids]
        return all(slot >= 0 for slot in slot_values) and len(set(slot_values)) == len(slot_values)

    def _record_resident_last_tokens(
        self,
        batch: DeviceBatch,
        token_ids: jnp.ndarray,
        *,
        eligible_rows: list[int],
        active_row_to_token_row: dict[int, int],
        full_batch_tokens: bool,
    ) -> None:
        if not eligible_rows or not batch.host.seq_ids:
            return
        slot_values = [self._hybrid_slots.get(int(seq_id), -1) for seq_id in batch.host.seq_ids]
        slots: list[int] = []
        token_rows: list[int] = []
        for row in eligible_rows:
            if row >= len(slot_values):
                continue
            slot = int(slot_values[row])
            if slot < 0:
                continue
            token_row = row if full_batch_tokens else active_row_to_token_row[row]
            slots.append(slot)
            token_rows.append(int(token_row))
        if not slots:
            return
        self._resident_last_tokens = self._scatter_resident_last_tokens(
            self._resident_last_tokens,
            self._resident_update_slots_device(slots),
            token_ids,
            jax.device_put(np.asarray(token_rows, dtype=np.int32)),
        )

    def _record_device_token_carry(
        self,
        batch: DeviceBatch,
        token_ids: jnp.ndarray,
        *,
        active_rows: list[int],
        prefill_final_flags: list[bool],
        seqs: list[Sequence],
        update_resident_tokens: bool = True,
        resident_tokens_already_current: bool = False,
    ) -> None:
        if (
            not self.device_token_carry
            or not batch.host.seq_ids
            or not active_rows
            or any(row >= len(seqs) or not seqs[row].ignore_eos for row in active_rows)
        ):
            self._clear_device_token_carry()
            return
        eligible_rows = active_rows
        if batch.is_prefill:
            eligible_rows = [
                row
                for row in active_rows
                if row < len(prefill_final_flags) and prefill_final_flags[row]
            ]
            if not eligible_rows:
                return
        if getattr(token_ids, "dtype", None) != jnp.dtype(jnp.int32):
            token_ids = jnp.asarray(token_ids, dtype=jnp.int32)
        full_batch_tokens = int(token_ids.shape[0]) == int(batch.tokens.shape[0])
        active_row_to_token_row = {row: index for index, row in enumerate(active_rows)}
        carry_by_seq_id = dict(self._device_token_carry_by_seq_id)
        new_carry_by_seq_id: dict[int, DeviceTokenRef] = {}
        for row in eligible_rows:
            seq_id = int(batch.host.seq_ids[row])
            if seq_id < 0:
                continue
            token_row = row if full_batch_tokens else active_row_to_token_row[row]
            token_ref = DeviceTokenRef(tokens=token_ids, row=token_row)
            carry_by_seq_id[seq_id] = token_ref
            new_carry_by_seq_id[seq_id] = token_ref
        if not new_carry_by_seq_id:
            return
        if update_resident_tokens:
            self._record_resident_last_tokens(
                batch,
                token_ids,
                eligible_rows=eligible_rows,
                active_row_to_token_row=active_row_to_token_row,
                full_batch_tokens=full_batch_tokens,
            )
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        elif resident_tokens_already_current:
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        else:
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.add(int(seq_id))
        self._device_token_carry_seq_ids = (
            tuple(int(seq_id) for seq_id in batch.host.seq_ids)
            if full_batch_tokens
            else tuple(new_carry_by_seq_id)
        )
        self._device_token_carry_tokens = token_ids
        self._device_token_carry_by_seq_id = carry_by_seq_id
        use_seq_lens_carry = self.static_decode_seq_lens_carry
        if batch.is_prefill:
            self._device_seq_lens_carry_seq_ids = None
            self._device_seq_lens_carry = None
        elif use_seq_lens_carry:
            self._device_seq_lens_carry_seq_ids = tuple(
                int(seq_id) for seq_id in batch.host.seq_ids
            )
            if active_rows == list(range(int(batch.tokens.shape[0]))):
                self._device_seq_lens_carry = batch.seq_lens.astype(jnp.int32) + jnp.asarray(
                    1, dtype=jnp.int32
                )
            else:
                active_mask = jnp.zeros((int(batch.tokens.shape[0]),), dtype=bool)
                active_mask = active_mask.at[jnp.asarray(active_rows, dtype=jnp.int32)].set(True)
                self._device_seq_lens_carry = jnp.where(
                    active_mask,
                    batch.seq_lens.astype(jnp.int32) + jnp.asarray(1, dtype=jnp.int32),
                    batch.seq_lens.astype(jnp.int32),
                )
        else:
            self._device_seq_lens_carry_seq_ids = None
            self._device_seq_lens_carry = None

    def _materialize_static_decode_metadata_batch(
        self,
        batch: DeviceBatch,
    ) -> DeviceBatch:
        """Build a concrete decode batch from static/resident scheduler metadata.

        Static resident decode batches intentionally carry placeholder token
        metadata because the resident fast path gathers block tables, sequence
        lengths, positions, and last tokens inside the compiled boundary.
        """
        if (
            batch.is_prefill
            or not bool(batch.host.uses_static_decode_metadata)
            or not batch.host.block_tables
            or not batch.host.seq_lens
        ):
            return batch

        batch_size = int(batch.tokens.shape[0])
        query_width = int(batch.tokens.shape[1])
        seq_lens_host = tuple(int(x) for x in batch.host.seq_lens)
        block_tables_host = tuple(
            tuple(int(block) for block in row) for row in batch.host.block_tables
        )
        seq_ids_host = tuple(
            int(seq_id)
            for seq_id in (
                batch.host.seq_ids
                if batch.host.seq_ids
                else tuple(int(x) for x in jax.device_get(batch.seq_ids).tolist())
            )
        )
        query_lens_host = tuple(
            int(query_len)
            for query_len in (
                batch.host.query_lens
                if batch.host.query_lens
                else tuple(int(x) for x in jax.device_get(batch.query_lens).tolist())
            )
        )

        positions = np.zeros((batch_size, query_width), dtype=np.int32)
        for row in range(min(batch_size, len(seq_lens_host), len(query_lens_host))):
            if query_lens_host[row] > 0:
                positions[row, 0] = max(seq_lens_host[row] - 1, 0)

        return replace(
            batch,
            positions=jax.device_put(positions),
            seq_ids=jax.device_put(np.asarray(seq_ids_host, dtype=np.int32)),
            block_tables=jax.device_put(np.asarray(block_tables_host, dtype=np.int32)),
            seq_lens=jax.device_put(np.asarray(seq_lens_host, dtype=np.int32)),
            host=replace(batch.host, uses_static_decode_metadata=False),
        )

    def _device_token_carry_enabled(self) -> bool:
        return self.device_token_carry

    @staticmethod
    def _materialize_device_token_outputs(
        outputs: dict[int, list[object] | object],
    ) -> dict[int, list[int] | int]:
        """Resolve deferred token refs for non-device-carry execution paths."""
        resolved_arrays: dict[int, np.ndarray] = {}

        def resolve_token(token: object) -> int:
            if isinstance(token, DeviceTokenRef):
                key = id(token.tokens)
                if key not in resolved_arrays:
                    resolved_arrays[key] = np.asarray(jax.device_get(token.tokens)).reshape(-1)
                return int(resolved_arrays[key][int(token.row)])
            if hasattr(token, "dtype") and hasattr(token, "shape"):
                return int(np.asarray(jax.device_get(token)).reshape(-1)[0])
            return int(token)  # type: ignore[arg-type]

        materialized: dict[int, list[int] | int] = {}
        for row, value in outputs.items():
            if isinstance(value, list):
                materialized[int(row)] = [resolve_token(token) for token in value]
            else:
                materialized[int(row)] = resolve_token(value)
        return materialized

    def _zero_hybrid_slot(self, slot: int):
        self._zero_hybrid_slots([slot])

    def _zero_hybrid_slots(self, slots: list[int] | tuple[int, ...]):
        slots = tuple(int(slot) for slot in slots if int(slot) >= 0)
        if not slots:
            return
        slots_to_zero = tuple(slot for slot in slots if slot not in self._zeroed_hybrid_slots)
        if not slots_to_zero:
            return
        conv_state = self._hybrid_state_table.conv_state
        recurrent_state = self._hybrid_state_table.recurrent_state
        if (
            conv_state is not None
            and recurrent_state is not None
            and len(slots_to_zero) == int(conv_state.shape[0])
            and slots_to_zero == tuple(range(int(conv_state.shape[0])))
        ):
            next_conv_state = jnp.zeros_like(conv_state)
            next_recurrent_state = jnp.zeros_like(recurrent_state)
        else:
            slot_ids = jnp.asarray(slots_to_zero, dtype=jnp.int32)
            next_conv_state = (
                conv_state.at[slot_ids].set(
                    jnp.zeros(
                        (len(slots_to_zero),) + conv_state.shape[1:],
                        dtype=conv_state.dtype,
                    )
                )
                if conv_state is not None
                else None
            )
            next_recurrent_state = (
                recurrent_state.at[slot_ids].set(
                    jnp.zeros(
                        (len(slots_to_zero),) + recurrent_state.shape[1:],
                        dtype=recurrent_state.dtype,
                    )
                )
                if recurrent_state is not None
                else None
            )
        self._hybrid_state_table = HybridLayerState(
            conv_state=next_conv_state,
            recurrent_state=next_recurrent_state,
        )
        self._zeroed_hybrid_slots.update(slots_to_zero)

    def _mark_hybrid_slots_written(self, slots: list[int] | tuple[int, ...]):
        for slot in slots:
            if int(slot) >= 0:
                self._zeroed_hybrid_slots.discard(int(slot))

    def _assign_hybrid_slot(
        self, seq_id: int, preferred_slot: int | None = None
    ) -> tuple[int, bool]:
        if seq_id < 0:
            return -1, False
        slot = self._hybrid_slots.get(seq_id)
        if slot is not None:
            return slot, False
        if not self._free_hybrid_slots:
            raise RuntimeError("No free hybrid-state slots; max_num_resident_seqs is exhausted")
        if (
            preferred_slot is not None
            and 0 <= preferred_slot < self._max_hybrid_slots
            and preferred_slot in self._free_hybrid_slots
        ):
            slot = preferred_slot
            self._free_hybrid_slots.remove(slot)
        elif seq_id < self._max_hybrid_slots and seq_id in self._free_hybrid_slots:
            slot = seq_id
            self._free_hybrid_slots.remove(slot)
        else:
            slot = self._free_hybrid_slots.pop()
        self._hybrid_slots[seq_id] = slot
        return slot, True

    def _ensure_hybrid_slot(self, seq_id: int, preferred_slot: int | None = None) -> int:
        slot, allocated = self._assign_hybrid_slot(seq_id, preferred_slot=preferred_slot)
        if allocated:
            self._zero_hybrid_slots([slot])
        return slot

    def _get_hybrid_state(self, seq_id: int) -> HybridLayerState:
        if seq_id < 0:
            return self._empty_hybrid_state
        slot = self._ensure_hybrid_slot(seq_id)
        return HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state[slot : slot + 1]
            if self._hybrid_state_table.conv_state is not None
            else None,
            recurrent_state=self._hybrid_state_table.recurrent_state[slot : slot + 1]
            if self._hybrid_state_table.recurrent_state is not None
            else None,
        )

    def _set_hybrid_state(self, seq_id: int, state: HybridLayerState | None):
        if state is None or seq_id < 0:
            return
        slot = self._ensure_hybrid_slot(seq_id)
        self._hybrid_state_table = HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state.at[slot].set(state.conv_state[0])
            if self._hybrid_state_table.conv_state is not None and state.conv_state is not None
            else self._hybrid_state_table.conv_state,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot].set(
                state.recurrent_state[0]
            )
            if self._hybrid_state_table.recurrent_state is not None
            and state.recurrent_state is not None
            else self._hybrid_state_table.recurrent_state,
        )

    def hybrid_state_for_sequence(self, seq_id: int) -> HybridLayerState | None:
        if seq_id < 0:
            return None
        if (
            self._hybrid_state_table.conv_state is None
            and self._hybrid_state_table.recurrent_state is None
        ):
            return None
        if seq_id not in self._hybrid_slots:
            return None
        return self._get_hybrid_state(seq_id)

    @staticmethod
    def _snapshot_hybrid_state(state: HybridLayerState) -> HybridLayerState:
        """Detach a cache snapshot from the donated resident-state table."""
        return HybridLayerState(
            conv_state=(
                jax.device_put(state.conv_state, may_alias=False)
                if state.conv_state is not None
                else None
            ),
            recurrent_state=(
                jax.device_put(state.recurrent_state, may_alias=False)
                if state.recurrent_state is not None
                else None
            ),
        )

    def cache_prefix_hybrid_states(
        self,
        entries_by_seq: dict[int, PrefixCacheEntry],
    ) -> dict[int, int]:
        """Snapshot exact prefix states and return opaque handles by hash."""
        representative: dict[int, tuple[int, int]] = {}
        for seq_id, entry in entries_by_seq.items():
            previous = representative.setdefault(
                entry.prefix_hash,
                (int(seq_id), entry.token_count),
            )
            if previous[1] != entry.token_count:
                raise RuntimeError("one prefix hash has inconsistent token counts")
        if (
            len(self._prefix_hybrid_states) + len(representative)
            > self._prefix_hybrid_state_capacity
        ):
            raise RuntimeError("runner prefix-state capacity is exhausted")

        states: dict[int, PrefixStateSnapshot] = {}
        for prefix_hash, (seq_id, token_count) in representative.items():
            state = self.hybrid_state_for_sequence(seq_id)
            if state is None:
                raise RuntimeError(f"sequence {seq_id} has no hybrid state to cache")
            states[prefix_hash] = PrefixStateSnapshot(
                prefix_hash=prefix_hash,
                token_count=token_count,
                state=self._snapshot_hybrid_state(state),
            )

        handles: dict[int, int] = {}
        for prefix_hash, snapshot in states.items():
            handle = self._next_prefix_hybrid_state_handle
            self._next_prefix_hybrid_state_handle += 1
            self._prefix_hybrid_states[handle] = snapshot
            handles[prefix_hash] = handle
        return handles

    def release_prefix_hybrid_states(self, handles: tuple[int, ...]) -> None:
        """Release cache snapshots after their host metadata is invalidated."""
        for handle in handles:
            if self._prefix_hybrid_states.pop(int(handle), None) is None:
                raise AssertionError(f"unknown prefix-state handle {handle}")

    def prefix_hybrid_state_stats(self) -> dict[str, int]:
        return {
            "handles": len(self._prefix_hybrid_states),
            "capacity": self._prefix_hybrid_state_capacity,
        }

    def install_cached_prefix_hybrid_states(
        self,
        seqs: list[Sequence],
        entries_by_seq: dict[int, PrefixCacheEntry],
    ) -> None:
        if not entries_by_seq:
            return
        for seq in seqs:
            seq_id = int(seq.seq_id)
            entry = entries_by_seq.get(seq_id)
            if entry is None or entry.hybrid_state_handle is None:
                continue
            if seq.cached_prefix_hybrid_seeded:
                continue
            snapshot = self._prefix_hybrid_states.get(entry.hybrid_state_handle)
            if snapshot is None:
                raise RuntimeError(
                    f"missing runner-owned hybrid prefix state handle {entry.hybrid_state_handle}"
                )
            if snapshot.prefix_hash != entry.prefix_hash:
                raise RuntimeError("prefix metadata and runner state hashes differ")
            if (
                snapshot.token_count != entry.token_count
                or snapshot.token_count != seq.num_cached_tokens
            ):
                raise RuntimeError("prefix KV and runner state token counts differ")
            self._set_hybrid_state(seq_id, snapshot.state)
            seq.cached_prefix_hybrid_seeded = True

    def _slice_batch(self, batch: DeviceBatch, idx: int) -> DeviceBatch:
        query_len = int(batch.query_lens[idx])
        return DeviceBatch(
            tokens=batch.tokens[idx : idx + 1, :query_len],
            positions=batch.positions[idx : idx + 1, :query_len],
            seq_ids=batch.seq_ids[idx : idx + 1],
            query_start_loc=jnp.array([0, query_len], dtype=jnp.int32),
            is_prefill=batch.is_prefill,
            num_prefill_tokens=query_len if batch.is_prefill else 0,
            num_decode_tokens=0 if batch.is_prefill else 1,
            block_tables=batch.block_tables[idx : idx + 1],
            seq_lens=batch.seq_lens[idx : idx + 1],
            host=HostBatch(
                seq_ids=batch.host.seq_ids[idx : idx + 1],
                query_lens=(query_len,),
                seq_lens=batch.host.seq_lens[idx : idx + 1],
                block_tables=batch.host.block_tables[idx : idx + 1],
                prefill_is_final=batch.host.prefill_is_final[idx : idx + 1],
                decode_steps=batch.host.decode_steps,
            ),
        )

    def _masked_decode_batch(
        self,
        batch: DeviceBatch,
        rows: list[int],
        *,
        token_values: list[int] | None = None,
        position_values: list[int] | None = None,
        seq_len_values: list[int] | None = None,
    ) -> DeviceBatch:
        if not rows:
            raise ValueError("rows must not be empty")
        if batch.is_prefill:
            raise ValueError("masked decode batches require a decode batch")
        batch_size = int(batch.tokens.shape[0])
        row_ids = jnp.array(rows, dtype=jnp.int32)
        active = jnp.zeros((batch_size,), dtype=bool).at[row_ids].set(True)
        tokens = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        seq_lens = jnp.zeros((batch_size,), dtype=jnp.int32)
        if token_values is None:
            tokens = tokens.at[row_ids, 0].set(batch.tokens[row_ids, 0])
        else:
            tokens = tokens.at[row_ids, 0].set(jnp.array(token_values, dtype=jnp.int32))
        if position_values is None:
            positions = positions.at[row_ids, 0].set(batch.positions[row_ids, 0])
        else:
            positions = positions.at[row_ids, 0].set(jnp.array(position_values, dtype=jnp.int32))
        if seq_len_values is None:
            seq_lens = seq_lens.at[row_ids].set(batch.seq_lens[row_ids])
        else:
            seq_lens = seq_lens.at[row_ids].set(jnp.array(seq_len_values, dtype=jnp.int32))
        query_lens = active.astype(jnp.int32)
        block_tables_host: tuple[tuple[int, ...], ...] = ()
        if batch.host.block_tables:
            zero_row = tuple(0 for _ in batch.host.block_tables[0])
            block_tables_host = tuple(
                tuple(batch.host.block_tables[row]) if row in rows else zero_row
                for row in range(batch_size)
            )
        row_set = set(int(row) for row in rows)
        seq_ids_host: tuple[int, ...] = ()
        if batch.host.seq_ids:
            seq_ids_host = tuple(
                int(batch.host.seq_ids[row]) if row in row_set else -1 for row in range(batch_size)
            )
        query_lens_host = tuple(1 if row in row_set else 0 for row in range(batch_size))
        seq_lens_host: tuple[int, ...] = ()
        if seq_len_values is not None:
            row_to_seq_len = {int(row): int(value) for row, value in zip(rows, seq_len_values)}
            seq_lens_host = tuple(
                row_to_seq_len[row] if row in row_to_seq_len else 0 for row in range(batch_size)
            )
        elif batch.host.seq_lens:
            seq_lens_host = tuple(
                int(batch.host.seq_lens[row]) if row in row_set else 0 for row in range(batch_size)
            )
        return DeviceBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=jnp.where(active, batch.seq_ids, jnp.full_like(batch.seq_ids, -1)),
            query_start_loc=jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens),
                ]
            ),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=len(rows),
            block_tables=jnp.where(
                active[:, None], batch.block_tables, jnp.zeros_like(batch.block_tables)
            ),
            seq_lens=seq_lens,
            host=HostBatch(
                seq_ids=seq_ids_host,
                query_lens=query_lens_host,
                seq_lens=seq_lens_host,
                block_tables=block_tables_host,
            ),
        )

    def _pad_decode_batch_to_rows(self, batch: DeviceBatch, target_rows: int) -> DeviceBatch:
        """Pad a decode batch with inactive rows to stabilize compiled shapes."""

        if batch.is_prefill:
            raise ValueError("decode batch padding requires a decode batch")
        target_rows = int(target_rows)
        current_rows = int(batch.tokens.shape[0])
        if target_rows <= current_rows:
            return batch
        query_width = int(batch.tokens.shape[1])
        block_width = int(batch.block_tables.shape[1])
        pad_rows = target_rows - current_rows

        query_lens = jnp.concatenate(
            [
                batch.query_lens.astype(jnp.int32),
                jnp.zeros((pad_rows,), dtype=jnp.int32),
            ]
        )
        zero_tokens = jnp.zeros((pad_rows, query_width), dtype=batch.tokens.dtype)
        zero_block_tables = jnp.zeros((pad_rows, block_width), dtype=batch.block_tables.dtype)

        seq_ids_host: tuple[int, ...] = ()
        if batch.host.seq_ids:
            seq_ids_host = tuple(int(x) for x in batch.host.seq_ids) + tuple(
                -1 for _ in range(pad_rows)
            )
        query_lens_host: tuple[int, ...] = ()
        if batch.host.query_lens:
            query_lens_host = tuple(int(x) for x in batch.host.query_lens) + tuple(
                0 for _ in range(pad_rows)
            )
        seq_lens_host: tuple[int, ...] = ()
        if batch.host.seq_lens:
            seq_lens_host = tuple(int(x) for x in batch.host.seq_lens) + tuple(
                0 for _ in range(pad_rows)
            )
        block_tables_host: tuple[tuple[int, ...], ...] = ()
        if batch.host.block_tables:
            zero_row = tuple(0 for _ in range(block_width))
            block_tables_host = tuple(
                tuple(int(block) for block in row) for row in batch.host.block_tables
            ) + tuple(zero_row for _ in range(pad_rows))
        return replace(
            batch,
            tokens=jnp.concatenate([batch.tokens, zero_tokens], axis=0),
            positions=jnp.concatenate([batch.positions, jnp.zeros_like(zero_tokens)], axis=0),
            seq_ids=jnp.concatenate(
                [
                    batch.seq_ids.astype(jnp.int32),
                    jnp.full((pad_rows,), -1, dtype=jnp.int32),
                ]
            ),
            query_start_loc=jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens),
                ]
            ),
            block_tables=jnp.concatenate([batch.block_tables, zero_block_tables], axis=0),
            seq_lens=jnp.concatenate(
                [
                    batch.seq_lens.astype(jnp.int32),
                    jnp.zeros((pad_rows,), dtype=jnp.int32),
                ]
            ),
            host=HostBatch(
                seq_ids=seq_ids_host,
                query_lens=query_lens_host,
                seq_lens=seq_lens_host,
                block_tables=block_tables_host,
                prefill_is_final=batch.host.prefill_is_final,
                decode_steps=batch.host.decode_steps,
            ),
        )

    def _with_committed_seq_lens(
        self,
        batch: DeviceBatch,
        committed_seq_lens: jnp.ndarray | None,
    ) -> DeviceBatch:
        if committed_seq_lens is None:
            return batch
        active = (batch.seq_ids >= 0) & (batch.query_lens > 0)
        seq_lens = jnp.where(
            active,
            jnp.asarray(committed_seq_lens, dtype=jnp.int32),
            batch.seq_lens.astype(jnp.int32),
        )
        seq_lens_host = batch.host.seq_lens
        if seq_lens_host:
            committed_host = np.asarray(jax.device_get(committed_seq_lens), dtype=np.int32).reshape(
                -1
            )
            seq_lens_values = [int(value) for value in seq_lens_host]
            seq_ids_host = batch.host.seq_ids
            query_lens_host = batch.host.query_lens
            for row in range(min(len(seq_lens_values), int(committed_host.shape[0]))):
                active_host = True
                if row < len(seq_ids_host):
                    active_host = active_host and int(seq_ids_host[row]) >= 0
                if row < len(query_lens_host):
                    active_host = active_host and int(query_lens_host[row]) > 0
                if active_host:
                    seq_lens_values[row] = int(committed_host[row])
            seq_lens_host = tuple(seq_lens_values)
        return replace(
            batch,
            seq_lens=seq_lens,
            host=replace(batch.host, seq_lens=seq_lens_host),
        )

    def _compact_decode_batch(
        self,
        batch: DeviceBatch,
        rows: list[int],
        *,
        token_values: list[int] | None = None,
        position_values: list[int] | None = None,
        seq_len_values: list[int] | None = None,
    ) -> DeviceBatch:
        if not rows:
            raise ValueError("rows must not be empty")
        if batch.is_prefill:
            raise ValueError("compact decode batches require a decode batch")

        row_ids = jnp.array(rows, dtype=jnp.int32)
        if token_values is None:
            tokens = batch.tokens[row_ids, :1]
        else:
            tokens = jnp.array(token_values, dtype=jnp.int32)[:, None]
        if position_values is None:
            positions = batch.positions[row_ids, :1]
        else:
            positions = jnp.array(position_values, dtype=jnp.int32)[:, None]
        if seq_len_values is None:
            seq_lens = batch.seq_lens[row_ids]
        else:
            seq_lens = jnp.array(seq_len_values, dtype=jnp.int32)
        block_tables_host: tuple[tuple[int, ...], ...] = ()
        if batch.host.block_tables:
            block_tables_host = tuple(tuple(batch.host.block_tables[row]) for row in rows)
        seq_ids_host: tuple[int, ...] = ()
        if batch.host.seq_ids:
            seq_ids_host = tuple(int(batch.host.seq_ids[row]) for row in rows)
        query_lens_host: tuple[int, ...] = ()
        if batch.host.query_lens:
            query_lens_host = tuple(int(batch.host.query_lens[row]) for row in rows)
        seq_lens_host: tuple[int, ...] = ()
        if seq_len_values is not None:
            seq_lens_host = tuple(int(value) for value in seq_len_values)
        elif batch.host.seq_lens:
            seq_lens_host = tuple(int(batch.host.seq_lens[row]) for row in rows)
        compact_size = len(rows)
        return DeviceBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=batch.seq_ids[row_ids],
            query_start_loc=jnp.arange(compact_size + 1, dtype=jnp.int32),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=compact_size,
            block_tables=batch.block_tables[row_ids],
            seq_lens=seq_lens,
            host=HostBatch(
                seq_ids=seq_ids_host,
                query_lens=query_lens_host,
                seq_lens=seq_lens_host,
                block_tables=block_tables_host,
                decode_steps=batch.host.decode_steps,
            ),
        )

    def _batch_hybrid_state(
        self,
        batch: DeviceBatch,
    ) -> tuple[HybridLayerState, list[int]]:
        seq_ids = (
            list(batch.host.seq_ids)
            if batch.host.seq_ids
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(seq_ids) == self._hybrid_state_table.conv_state.shape[0]
        ):
            direct_slots = True
            for row, seq_id in enumerate(seq_ids):
                if seq_id < 0 or self._hybrid_slots.get(int(seq_id)) != row:
                    direct_slots = False
                    break
            if direct_slots:
                slot_values = list(range(len(seq_ids)))
                return self._hybrid_state_table, slot_values
        slot_allocations = [
            self._assign_hybrid_slot(int(seq_id), preferred_slot=row)
            for row, seq_id in enumerate(seq_ids)
        ]
        slot_values = [slot for slot, _ in slot_allocations]
        newly_allocated = [allocated for _, allocated in slot_allocations]
        self._zero_hybrid_slots([slot for slot, allocated in slot_allocations if allocated])
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(slot_values) == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
            and all(newly_allocated)
        ):
            return self._hybrid_state_table, slot_values
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(slot_values) == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
            and not any(newly_allocated)
        ):
            return self._hybrid_state_table, slot_values
        slot_ids = jnp.array(slot_values, dtype=jnp.int32)
        safe_slot_ids = jnp.maximum(slot_ids, 0)
        valid = (slot_ids >= 0) & jnp.logical_not(jnp.array(newly_allocated, dtype=bool))
        conv_state = None
        recurrent_state = None
        if self._hybrid_state_table.conv_state is not None:
            conv_state = self._hybrid_state_table.conv_state[safe_slot_ids]
            conv_state = jnp.where(
                valid.reshape((valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                conv_state,
                jnp.zeros_like(conv_state),
            )
        if self._hybrid_state_table.recurrent_state is not None:
            recurrent_state = self._hybrid_state_table.recurrent_state[safe_slot_ids]
            recurrent_state = jnp.where(
                valid.reshape((valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                recurrent_state,
                jnp.zeros_like(recurrent_state),
            )
        return HybridLayerState(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
        ), slot_values

    def _store_batch_hybrid_state(
        self,
        batch: DeviceBatch,
        state: HybridLayerState | None,
        slot_values_all: list[int],
    ) -> None:
        if state is None:
            return
        valid_rows: list[int] = []
        query_lens = (
            list(batch.host.query_lens)
            if batch.host.query_lens
            else [int(x) for x in batch.query_lens.tolist()]
        )
        seq_ids = (
            list(batch.host.seq_ids)
            if batch.host.seq_ids
            else [int(x) for x in batch.seq_ids.tolist()]
        )
        slot_values: list[int] = []
        for row, seq_id in enumerate(seq_ids):
            if seq_id < 0 or (not batch.is_prefill and query_lens[row] <= 0):
                continue
            valid_rows.append(row)
            slot_values.append(slot_values_all[row])
        if not valid_rows:
            return
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and state.conv_state is not None
            and state.recurrent_state is not None
            and len(valid_rows) == len(slot_values) == state.conv_state.shape[0]
            and state.conv_state.shape[0] == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
        ):
            self._hybrid_state_table = state
            self._mark_hybrid_slots_written(slot_values)
            return
        row_ids = jnp.array(valid_rows, dtype=jnp.int32)
        slot_ids = jnp.array(slot_values, dtype=jnp.int32)
        self._hybrid_state_table = HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state.at[slot_ids].set(
                state.conv_state[row_ids]
            )
            if self._hybrid_state_table.conv_state is not None and state.conv_state is not None
            else self._hybrid_state_table.conv_state,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot_ids].set(
                state.recurrent_state[row_ids]
            )
            if self._hybrid_state_table.recurrent_state is not None
            and state.recurrent_state is not None
            else self._hybrid_state_table.recurrent_state,
        )
        self._mark_hybrid_slots_written(slot_values)

    def _batch_hybrid_slot_ids(
        self,
        batch: DeviceBatch,
    ) -> tuple[jnp.ndarray, list[int]]:
        """Assign hybrid slots for a batch without gathering the state table."""

        seq_ids = (
            list(batch.host.seq_ids)
            if batch.host.seq_ids
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        slot_values: list[int] = []
        for row, seq_id in enumerate(seq_ids):
            slot, allocated = self._assign_hybrid_slot(int(seq_id), preferred_slot=row)
            if allocated:
                self._zero_hybrid_slots([slot])
            slot_values.append(slot)
        slot_key = tuple(slot_values)
        cache = self._hybrid_slot_ids_device_cache
        cached = cache.get(slot_key)
        if cached is None:
            cached = jax.device_put(np.asarray(slot_key, dtype=np.int32))
            cache[slot_key] = cached
        return cached, slot_values

    def _prefill_final_flags_device(self, batch: DeviceBatch) -> jnp.ndarray:
        rows = max(0, int(batch.query_start_loc.shape[0]) - 1)
        flags = [bool(flag) for flag in list(batch.prefill_final_flags)[:rows]]
        if len(flags) < rows:
            flags.extend([False] * (rows - len(flags)))
        key = tuple(flags)
        cache = self._prefill_final_flags_device_cache
        cached = cache.get(key)
        if cached is None:
            cached = jax.device_put(np.asarray(key, dtype=bool))
            cache[key] = cached
        return cached

    def _resident_metadata_scatter_fn(
        self, kind: str, table_shape: tuple[int, ...], update_shape: tuple[int, ...]
    ):
        cache = self._resident_metadata_scatter_cache
        key = (kind, tuple(int(x) for x in table_shape), tuple(int(x) for x in update_shape))
        fn = cache.get(key)
        if fn is None:

            def scatter_rows(table, slots, rows):
                return table.at[slots].set(rows)

            fn = jax.jit(scatter_rows, donate_argnums=(0,))
            cache[key] = fn
        return fn

    def _scatter_resident_block_table_rows(
        self,
        table: jnp.ndarray,
        slots: jnp.ndarray,
        rows: jnp.ndarray,
    ) -> jnp.ndarray:
        rows = jnp.asarray(rows, dtype=jnp.int32)
        slots = _int32_device_vector(slots)
        if rows.ndim != 2:
            raise ValueError("resident block-table row updates must be rank-2")
        if int(rows.shape[1]) != int(table.shape[1]):
            raise ValueError(
                "resident block-table update width must match the resident table width"
            )
        fn = self._resident_metadata_scatter_fn(
            "block_tables",
            tuple(int(x) for x in table.shape),
            tuple(int(x) for x in rows.shape),
        )
        return fn(table, slots, rows)

    def _scatter_resident_seq_lens(
        self,
        table: jnp.ndarray,
        slots: jnp.ndarray,
        seq_lens: jnp.ndarray,
    ) -> jnp.ndarray:
        seq_lens = _int32_device_vector(seq_lens)
        slots = _int32_device_vector(slots)
        fn = self._resident_metadata_scatter_fn(
            "seq_lens",
            tuple(int(x) for x in table.shape),
            tuple(int(x) for x in seq_lens.shape),
        )
        return fn(table, slots, seq_lens)

    def _resident_last_tokens_scatter_fn(
        self,
        table_shape: tuple[int, ...],
        slots_shape: tuple[int, ...],
        token_shape: tuple[int, ...],
        token_rows_shape: tuple[int, ...],
    ):
        cache = self._resident_metadata_scatter_cache
        key = (
            "last_tokens_from_rows",
            tuple(int(x) for x in table_shape),
            tuple(int(x) for x in slots_shape),
            tuple(int(x) for x in token_shape),
            tuple(int(x) for x in token_rows_shape),
        )
        fn = cache.get(key)
        if fn is None:

            def scatter_tokens(table, slots, token_ids, token_rows):
                token_vector = jnp.asarray(token_ids, dtype=jnp.int32).reshape(-1)
                token_rows = jnp.asarray(token_rows, dtype=jnp.int32).reshape(-1)
                slots = jnp.asarray(slots, dtype=jnp.int32).reshape(-1)
                return table.at[slots].set(token_vector[token_rows])

            fn = jax.jit(scatter_tokens, donate_argnums=(0,))
            cache[key] = fn
        return fn

    def _scatter_resident_last_tokens(
        self,
        table: jnp.ndarray,
        slots: jnp.ndarray,
        token_ids: jnp.ndarray,
        token_rows: jnp.ndarray,
    ) -> jnp.ndarray:
        slots = _int32_device_vector(slots)
        token_ids = jnp.asarray(token_ids, dtype=jnp.int32)
        token_rows = _int32_device_vector(token_rows)
        fn = self._resident_last_tokens_scatter_fn(
            tuple(int(x) for x in table.shape),
            tuple(int(x) for x in slots.shape),
            tuple(int(x) for x in token_ids.shape),
            tuple(int(x) for x in token_rows.shape),
        )
        return fn(table, slots, token_ids, token_rows)

    def _resident_update_slots_device(self, slots: list[int] | tuple[int, ...]) -> jnp.ndarray:
        key = tuple(int(slot) for slot in slots)
        cache = self._resident_update_slots_device_cache
        cached = cache.get(key)
        if cached is None:
            cached = jax.device_put(np.asarray(key, dtype=np.int32))
            cache[key] = cached
        return cached

    def _sync_resident_decode_metadata(
        self,
        batch: DeviceBatch,
        slot_values: list[int] | tuple[int, ...],
        *,
        sync_seq_lens: bool,
        force_block_tables: bool = False,
    ) -> None:
        """Refresh resident per-slot paging metadata from scheduler-owned rows.

        Block allocations are still owned by the Python scheduler/block manager.
        This method mirrors only the changed rows into device-resident tables so
        decode can gather compact metadata by slot id inside the JIT boundary.
        """

        if not batch.host.block_tables:
            return
        seq_ids = (
            list(batch.host.seq_ids)
            if batch.host.seq_ids
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        query_lens = (
            list(batch.host.query_lens)
            if batch.host.query_lens
            else [int(query_len) for query_len in batch.query_lens.tolist()]
        )
        seq_lens = (
            list(batch.host.seq_lens)
            if batch.host.seq_lens
            else [int(seq_len) for seq_len in batch.seq_lens.tolist()]
        )
        block_size = self.block_size
        changed_block_slots: list[int] = []
        changed_block_rows: list[tuple[int, ...]] = []
        changed_seq_lens_slots: list[int] = []
        changed_seq_lens: list[int] = []
        for row, slot in enumerate(slot_values):
            slot = int(slot)
            if slot < 0 or row >= len(seq_ids) or int(seq_ids[row]) < 0:
                continue
            if (not batch.is_prefill) and row < len(query_lens) and int(query_lens[row]) <= 0:
                continue

            next_block_count = None
            if row < len(seq_lens):
                seq_len_for_blocks = max(0, int(seq_lens[row]))
                next_block_count = (seq_len_for_blocks + block_size - 1) // block_size
            skip_block_row_check = (
                not force_block_tables
                and not batch.is_prefill
                and next_block_count is not None
                and self._resident_block_counts_host[slot] == next_block_count
            )
            if not skip_block_row_check:
                source_row = tuple(int(block) for block in batch.host.block_tables[row])
                if len(source_row) < self.max_blocks_per_seq:
                    source_row = source_row + tuple(
                        0 for _ in range(self.max_blocks_per_seq - len(source_row))
                    )
                elif len(source_row) > self.max_blocks_per_seq:
                    source_row = source_row[: self.max_blocks_per_seq]
                if self._resident_block_tables_host[slot] != source_row:
                    self._resident_block_tables_host[slot] = source_row
                    changed_block_slots.append(slot)
                    changed_block_rows.append(source_row)
                if next_block_count is not None:
                    self._resident_block_counts_host[slot] = next_block_count

            if sync_seq_lens and row < len(seq_lens):
                seq_len = int(seq_lens[row])
                if self._resident_seq_lens_host[slot] != seq_len:
                    self._resident_seq_lens_host[slot] = seq_len
                    changed_seq_lens_slots.append(slot)
                    changed_seq_lens.append(seq_len)

        if changed_block_slots:
            self._resident_block_tables = self._scatter_resident_block_table_rows(
                self._resident_block_tables,
                self._resident_update_slots_device(changed_block_slots),
                jax.device_put(np.asarray(changed_block_rows, dtype=np.int32)),
            )
        if changed_seq_lens_slots:
            self._resident_seq_lens = self._scatter_resident_seq_lens(
                self._resident_seq_lens,
                self._resident_update_slots_device(changed_seq_lens_slots),
                jax.device_put(np.asarray(changed_seq_lens, dtype=np.int32)),
            )

    def _advance_resident_seq_lens_host(
        self,
        slot_values: list[int] | tuple[int, ...],
        *,
        active_rows: list[int],
        steps: int,
    ) -> None:
        if steps <= 0:
            return
        active = set(int(row) for row in active_rows)
        for row, slot in enumerate(slot_values):
            slot = int(slot)
            if slot >= 0 and row in active:
                self._resident_seq_lens_host[slot] += int(steps)

    def _step_fn(self, batch: DeviceBatch):
        execution = self.execution
        if execution == "jit" or (execution == "decode-jit" and not batch.is_prefill):
            return self.executor.forward_step_jit
        return self.executor.forward_step

    def _can_use_greedy_token_fastpath(self, seqs: list[Sequence], batch: DeviceBatch) -> bool:
        if not self.greedy_token_fastpath:
            return False
        execution = self.execution
        if execution != "jit" and not (execution == "decode-jit" and not batch.is_prefill):
            return False
        for seq in seqs:
            if seq.temperature != 0.0:
                return False
        return True

    def _can_use_sampled_token_fastpath(self, seqs: list[Sequence], batch: DeviceBatch) -> bool:
        if not self.sampled_token_fastpath:
            return False
        execution = self.execution
        if execution != "jit" and not (execution == "decode-jit" and not batch.is_prefill):
            return False
        has_sampling = False
        for seq in seqs:
            temperature = seq.temperature
            if temperature < 0.0:
                return False
            if temperature > 0.0:
                has_sampling = True
        return has_sampling

    def _can_speculate(self, seqs: list[Sequence], batch: DeviceBatch) -> bool:
        drafter = self.config.drafter
        if drafter is None or self.mtp_state is None or batch.is_prefill:
            return False
        if not (self.device_token_carry and self.resident_decode_metadata):
            return False
        if int(batch.tokens.shape[0]) != len(seqs):
            return False
        if any(int(seq.seq_id) not in self._mtp_ready_seq_ids for seq in seqs):
            return False
        needed = drafter.verification_width
        return all(
            seq.temperature == 0
            and seq.ignore_eos
            and seq.max_tokens - seq.num_completion_tokens >= needed
            for seq in seqs
        )

    def _sample_temperatures_device(self, seqs: list[Sequence], batch: DeviceBatch) -> jnp.ndarray:
        row_count = (
            len(seqs) if batch.is_prefill and batch.packed_prefill else int(batch.tokens.shape[0])
        )
        values = [0.0 for _ in range(row_count)]
        active_limit = min(len(seqs), row_count)
        for row in range(active_limit):
            if batch.host.query_lens and int(batch.host.query_lens[row]) <= 0:
                continue
            values[row] = float(getattr(seqs[row], "temperature", 0.0))
        return jnp.asarray(values, dtype=jnp.float32)

    @staticmethod
    def _next_prompt_tokens_device(
        seqs: list[Sequence],
        batch: DeviceBatch,
    ) -> jnp.ndarray:
        row_count = int(batch.block_tables.shape[0])
        values = [0] * row_count
        final_flags = batch.prefill_final_flags
        seq_lens = list(batch.host.seq_lens)
        for row, seq in enumerate(seqs[:row_count]):
            if row >= len(final_flags) or final_flags[row]:
                continue
            end = int(seq_lens[row])
            if not 0 <= end < seq.num_prompt_tokens:
                raise ValueError("chunked prefill is missing its next prompt token")
            values[row] = int(seq.prompt_token_ids[end])
        return jnp.asarray(values, dtype=jnp.int32)

    def _sample_rng_slots_and_counters_device(
        self,
        batch: DeviceBatch,
        slot_values: list[int],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self._flush_resident_rng_counter_resets()
        row_count = (
            len(batch.host.query_lens)
            if batch.is_prefill and batch.packed_prefill and batch.host.query_lens
            else int(batch.tokens.shape[0])
        )
        slot_values = list(slot_values)
        if len(slot_values) < row_count:
            slot_values.extend([-1] * (row_count - len(slot_values)))
        safe_slots = [max(0, int(slot)) for slot in slot_values[:row_count]]
        slot_ids = jnp.asarray(safe_slots, dtype=jnp.int32)
        counters = self._resident_rng_counters[slot_ids]
        return slot_ids, counters.astype(jnp.int32)

    def _flush_resident_rng_counter_resets(self) -> None:
        """Apply deferred sampled-RNG counter resets before sampled paths read them."""

        reset_slots = self._resident_rng_counter_reset_slots
        if not reset_slots:
            return
        slots = tuple(sorted(int(slot) for slot in reset_slots if int(slot) >= 0))
        reset_slots.clear()
        if not slots:
            return
        slot_ids = jnp.asarray(slots, dtype=jnp.int32)
        self._resident_rng_counters = self._resident_rng_counters.at[slot_ids].set(
            jnp.zeros((len(slots),), dtype=jnp.int32)
        )

    def _record_resident_rng_counters(
        self,
        batch: DeviceBatch,
        updated_counters: jnp.ndarray | None,
        *,
        slot_values: list[int],
        active_rows: list[int],
        prefill_final_flags: list[bool],
    ) -> None:
        if updated_counters is None:
            return
        self._flush_resident_rng_counter_resets()
        if not slot_values:
            return
        slots: list[int] = []
        rows: list[int] = []
        for row in active_rows:
            if row >= len(slot_values):
                continue
            if batch.is_prefill and (
                row >= len(prefill_final_flags) or not prefill_final_flags[row]
            ):
                continue
            slot = int(slot_values[row])
            if slot < 0:
                continue
            slots.append(slot)
            rows.append(row)
        if not slots:
            return
        self._resident_rng_counters = self._resident_rng_counters.at[
            jnp.asarray(slots, dtype=jnp.int32)
        ].set(updated_counters[jnp.asarray(rows, dtype=jnp.int32)].astype(jnp.int32))

    def _greedy_decode_burst_steps(self, seqs: list[Sequence], batch: DeviceBatch) -> int:
        if batch.is_prefill:
            return 1
        configured_steps = self.config.kernels.greedy_decode_burst_steps
        if configured_steps <= 1:
            return 1
        if batch.host.decode_steps <= 1:
            return 1
        if batch.host.query_lens:
            active_query_lens = list(batch.host.query_lens[: len(seqs)])
            if any(int(length) != 1 for length in active_query_lens):
                return 1
        for seq in seqs:
            if seq.temperature != 0 or not seq.ignore_eos:
                return 1
        remaining = [seq.max_tokens - seq.num_completion_tokens for seq in seqs]
        if not remaining or min(remaining) <= 1:
            return 1
        return max(
            1,
            min(
                configured_steps,
                int(batch.host.decode_steps),
                min(remaining),
            ),
        )

    @staticmethod
    def _prefill_final_flags_for_batch(
        seqs: list[Sequence], batch: DeviceBatch
    ) -> tuple[bool, ...]:
        if batch.is_prefill:
            prefill_final_flags = list(batch.prefill_final_flags)[: len(seqs)]
            if len(prefill_final_flags) < len(seqs):
                prefill_final_flags.extend([True] * (len(seqs) - len(prefill_final_flags)))
            return tuple(bool(flag) for flag in prefill_final_flags)
        return tuple(True for _ in seqs)

    @staticmethod
    def _host_query_lens_and_seq_ids(
        batch: DeviceBatch, row_count: int
    ) -> tuple[list[int], list[int]]:
        if batch.host.query_lens:
            query_lens = [int(x) for x in batch.host.query_lens[:row_count]]
        else:
            query_lens = [int(x) for x in batch.query_lens[:row_count].tolist()]
        if batch.host.seq_ids:
            seq_ids_host = [int(x) for x in batch.host.seq_ids[:row_count]]
        else:
            seq_ids_host = [int(batch.seq_ids[row]) for row in range(row_count)]
        return query_lens, seq_ids_host

    def _select_route(self, seqs: list[Sequence], batch: DeviceBatch) -> ExecutionPlan:
        prefill_final_flags = self._prefill_final_flags_for_batch(seqs, batch)
        greedy = self._can_use_greedy_token_fastpath(seqs, batch)
        sampled = not greedy and self._can_use_sampled_token_fastpath(seqs, batch)
        speculation_candidate = greedy and self._can_speculate(seqs, batch)
        decode_steps = (
            1
            if speculation_candidate
            else self._greedy_decode_burst_steps(seqs, batch)
            if greedy
            else 1
        )
        phase = BatchPhase.PREFILL if batch.is_prefill else BatchPhase.DECODE
        tokens = (
            TokenMode.BURST
            if decode_steps > 1
            else TokenMode.GREEDY
            if greedy
            else TokenMode.SAMPLED
            if sampled
            else TokenMode.LOGITS
        )

        query_lens, seq_ids_host = self._host_query_lens_and_seq_ids(batch, len(seqs))
        active_rows = tuple(
            row
            for row, query_len in enumerate(query_lens)
            if query_len > 0 and seq_ids_host[row] >= 0
        )
        has_hybrid_table = (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
        )
        capabilities: set[RouteCapability] = set()
        if has_hybrid_table:
            capabilities.add(RouteCapability.TABLE_STATE)
        if self.mtp_state is not None:
            capabilities.add(RouteCapability.DRAFT_STATE)
        if batch.is_prefill and self.device_token_carry:
            capabilities.add(RouteCapability.PREFILL_TOKEN_SEED)
        if not batch.is_prefill and self.resident_decode_metadata:
            capabilities.add(RouteCapability.RESIDENT_METADATA)
        active_decode_rows = self._active_decode_rows_host(batch)
        slot_tokens_ready = self._resident_slot_token_decode_ready(
            batch,
            active_rows=active_decode_rows,
        )
        if (
            decode_steps <= 1
            and slot_tokens_ready
            and (bool(batch.host.uses_static_decode_metadata) or speculation_candidate)
        ):
            capabilities.add(RouteCapability.SLOT_TOKENS)
            if self._resident_slot_token_dense_decode_ready(
                batch,
                active_rows=active_decode_rows,
            ):
                capabilities.add(RouteCapability.DENSE_ROWS)
        if speculation_candidate and RouteCapability.DENSE_ROWS in capabilities:
            tokens = TokenMode.SPECULATIVE
            decode_steps = 1
        kind = select_route(
            RouteRequest(
                phase=phase,
                tokens=tokens,
                capabilities=frozenset(capabilities),
            )
        )
        return ExecutionPlan(
            kind=kind,
            active_rows=active_rows,
            decode_steps=decode_steps,
            prefill_final_flags=prefill_final_flags,
        )

    def _prepare_batch_for_route(self, route: ExecutionPlan, batch: DeviceBatch) -> DeviceBatch:
        if not route.spec.uses_slot_tokens:
            return self._maybe_apply_device_token_carry(batch)
        return batch

    def _hybrid_inputs_for_route(
        self,
        route: ExecutionPlan,
        batch: DeviceBatch,
    ) -> tuple[jnp.ndarray | None, list[int], HybridLayerState]:
        if route.spec.uses_table_state:
            hybrid_slot_ids, hybrid_slot_values = self._batch_hybrid_slot_ids(batch)
            hybrid_state = self._hybrid_state_table
        else:
            hybrid_slot_ids = None
            hybrid_state, hybrid_slot_values = self._batch_hybrid_state(batch)
        return hybrid_slot_ids, hybrid_slot_values, hybrid_state

    def _sync_route_resident_metadata(
        self,
        route: ExecutionPlan,
        batch: DeviceBatch,
        hybrid_slot_values: list[int],
    ) -> None:
        if route.spec.uses_resident_metadata:
            self._sync_resident_decode_metadata(
                batch,
                hybrid_slot_values,
                sync_seq_lens=True,
                force_block_tables=route.spec.tokens is TokenMode.SPECULATIVE,
            )

    def _commit_route_output(
        self,
        route: ExecutionPlan,
        batch: DeviceBatch,
        output: Any,
        *,
        hybrid_slot_values: list[int],
        emitted_counts: tuple[int, ...] | None = None,
    ) -> None:
        self.cache_storage = output.cache_storage
        if route.spec.uses_draft_state:
            if output.mtp_state is None:
                raise RuntimeError("MTP route did not return persistent predictor state")
            self.mtp_state = output.mtp_state
            if batch.is_prefill:
                final_flags = batch.prefill_final_flags
                for row, seq_id in enumerate(batch.host.seq_ids):
                    if seq_id < 0 or row >= len(batch.host.query_lens):
                        continue
                    if int(batch.host.query_lens[row]) <= 0:
                        continue
                    if row < len(final_flags) and final_flags[row]:
                        self._mtp_ready_seq_ids.add(int(seq_id))
                    else:
                        self._mtp_ready_seq_ids.discard(int(seq_id))
        elif self.config.drafter is not None and not batch.is_prefill:
            for row in route.active_rows:
                if row < len(batch.host.seq_ids):
                    self._mtp_ready_seq_ids.discard(int(batch.host.seq_ids[row]))
        if route.spec.uses_table_state:
            self._hybrid_state_table = output.hybrid_state
            self._mark_hybrid_slots_written(hybrid_slot_values)
            if route.spec.uses_resident_metadata and output.resident_seq_lens is not None:
                self._resident_seq_lens = output.resident_seq_lens
                if emitted_counts is None:
                    self._advance_resident_seq_lens_host(
                        hybrid_slot_values,
                        active_rows=list(route.active_rows),
                        steps=1,
                    )
                else:
                    active = set(route.active_rows)
                    for row, slot in enumerate(hybrid_slot_values):
                        if row in active and slot >= 0:
                            self._resident_seq_lens_host[slot] += emitted_counts[row]
        else:
            self._store_batch_hybrid_state(
                batch,
                output.hybrid_state,
                hybrid_slot_values,
            )
            if route.spec.tokens is TokenMode.SAMPLED:
                self._record_resident_rng_counters(
                    batch,
                    output.resident_rng_counters,
                    slot_values=hybrid_slot_values,
                    active_rows=list(route.active_rows),
                    prefill_final_flags=list(route.prefill_final_flags),
                )
        if batch.is_prefill and self.resident_decode_metadata:
            self._sync_resident_decode_metadata(
                batch,
                hybrid_slot_values,
                sync_seq_lens=True,
            )

    def _invoke_route(
        self,
        route: ExecutionPlan,
        seqs: list[Sequence],
        batch: DeviceBatch,
        *,
        hybrid_slot_ids: jnp.ndarray | None,
        hybrid_slot_values: list[int],
        hybrid_state: HybridLayerState,
    ) -> Any:
        spec = route.spec
        if spec.executor is None:
            return self._step_fn(batch)(
                batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                return_hidden=False,
                return_hidden_with_logits=False,
                last_logits_only=True,
            )

        kwargs: dict[str, Any] = {"cache_storage": self.cache_storage}
        if spec.uses_table_state:
            kwargs.update(
                hybrid_state_table=hybrid_state,
                hybrid_slot_ids=hybrid_slot_ids,
            )
        else:
            kwargs["hybrid_state"] = hybrid_state
        if spec.uses_resident_metadata:
            kwargs.update(
                resident_block_tables=self._resident_block_tables,
                resident_seq_lens=self._resident_seq_lens,
            )
        if spec.uses_slot_tokens or spec.seeds_slot_tokens:
            kwargs["resident_last_tokens"] = self._resident_last_tokens
        if spec.seeds_slot_tokens:
            kwargs["prefill_final_flags"] = self._prefill_final_flags_device(batch)
        if spec.uses_draft_state:
            if self.mtp_state is None:
                raise RuntimeError("MTP route requires persistent predictor state")
            kwargs["mtp_state"] = self.mtp_state
            if batch.is_prefill:
                kwargs["next_prompt_tokens"] = self._next_prompt_tokens_device(
                    seqs,
                    batch,
                )
        if spec.tokens is TokenMode.BURST:
            kwargs["decode_steps"] = route.decode_steps
        if spec.tokens is TokenMode.SPECULATIVE:
            if batch.is_prefill:
                raise AssertionError("speculative decode route received prefill")
        if spec.tokens is TokenMode.SAMPLED:
            kwargs["temperatures"] = self._sample_temperatures_device(seqs, batch)
            rng_slots, rng_counters = self._sample_rng_slots_and_counters_device(
                batch,
                hybrid_slot_values,
            )
            kwargs.update(rng_slots=rng_slots, rng_counters=rng_counters)
        return getattr(self.executor, spec.executor)(batch, **kwargs)

    @staticmethod
    def _verification_output(
        route: ExecutionPlan,
        output: Any,
    ) -> tuple[
        VerificationResult | None,
        tuple[int, ...] | None,
        tuple[int, ...] | None,
    ]:
        if route.spec.tokens is not TokenMode.SPECULATIVE:
            return None, None, None
        verification = output.activations
        if not isinstance(verification, VerificationResult):
            raise TypeError("speculative executor must return VerificationResult")
        emitted, accepted = jax.device_get(
            (verification.emitted_counts, verification.accepted_counts)
        )
        return (
            verification,
            tuple(int(value) for value in np.asarray(emitted)),
            tuple(int(value) for value in np.asarray(accepted)),
        )

    def _speculative_token_rows(
        self,
        seqs: list[Sequence],
        batch: DeviceBatch,
        verification: VerificationResult,
        emitted_counts: tuple[int, ...],
        accepted_counts: tuple[int, ...],
        *,
        active_rows: list[int],
        prefill_final_flags: list[bool],
    ) -> list[list[DeviceTokenRef]]:
        self._record_device_token_carry(
            batch,
            verification.next_token_ids,
            active_rows=active_rows,
            prefill_final_flags=prefill_final_flags,
            seqs=seqs,
            update_resident_tokens=False,
            resident_tokens_already_current=True,
        )
        draft_width = int(verification.emitted_token_ids.shape[1]) - 1
        drafted = len(active_rows) * draft_width
        accepted = sum(accepted_counts[row] for row in active_rows)
        self.speculation_stats["drafted"] += drafted
        self.speculation_stats["accepted"] += accepted
        self.speculation_stats["rejected"] += drafted - accepted
        self.speculation_stats["bonus"] += sum(
            accepted_counts[row] == draft_width for row in active_rows
        )
        self.speculation_stats["target_tokens"] += len(active_rows) * (draft_width + 1)
        width = draft_width + 1
        return [
            [
                DeviceTokenRef(
                    tokens=verification.emitted_token_ids,
                    row=row * width + column,
                )
                for column in range(emitted_counts[row])
            ]
            if row in active_rows
            else []
            for row in range(len(seqs))
        ]

    def _run_main_and_sample(
        self,
        seqs: list[Sequence],
        batch: DeviceBatch,
    ) -> list[Any | list[Any]]:
        route = self._select_route(seqs, batch)
        return self._run_execution_plan(route, seqs, batch)

    def _run_execution_plan(
        self,
        route: ExecutionPlan,
        seqs: list[Sequence],
        batch: DeviceBatch,
    ) -> list[Any | list[Any]]:
        batch = self._prepare_batch_for_route(route, batch)
        prefill_final_flags = list(route.prefill_final_flags)
        active_rows = list(route.active_rows)
        decode_burst_steps = route.decode_steps

        hybrid_slot_ids, hybrid_slot_values, hybrid_state = self._hybrid_inputs_for_route(
            route, batch
        )
        self._sync_route_resident_metadata(route, batch, hybrid_slot_values)
        output = self._invoke_route(
            route,
            seqs,
            batch,
            hybrid_slot_ids=hybrid_slot_ids,
            hybrid_slot_values=hybrid_slot_values,
            hybrid_state=hybrid_state,
        )
        verification, emitted_counts, accepted_counts = self._verification_output(route, output)
        prefill_resident_tokens_seeded = (
            route.spec.seeds_slot_tokens and output.resident_last_tokens is not None
        )
        if output.resident_last_tokens is not None:
            self._resident_last_tokens = output.resident_last_tokens
        self._commit_route_output(
            route,
            batch,
            output,
            hybrid_slot_values=hybrid_slot_values,
            emitted_counts=emitted_counts,
        )

        if verification is not None:
            if emitted_counts is None or accepted_counts is None:
                raise AssertionError("verification counts were not materialized")
            return self._speculative_token_rows(
                seqs,
                batch,
                verification,
                emitted_counts,
                accepted_counts,
                active_rows=active_rows,
                prefill_final_flags=prefill_final_flags,
            )

        emits_token_ids = route.spec.tokens is not TokenMode.LOGITS
        token_ids_all = None
        if decode_burst_steps > 1:
            token_ids_all = output.activations[: len(seqs), :decode_burst_steps]
            last_logits = None
        elif emits_token_ids:
            token_ids_all = (
                output.activations[0]
                if isinstance(output.activations, tuple)
                else output.activations
            )
            if int(token_ids_all.shape[0]) != len(seqs):
                token_ids_all = token_ids_all[: len(seqs)]
            last_logits = None
        else:
            last_logits = output.activations[: len(seqs), 0]

        carry_device_tokens = (
            emits_token_ids
            and self.device_token_carry
            and all(seqs[row].ignore_eos for row in active_rows)
        )
        if emits_token_ids and decode_burst_steps <= 1:
            carry_tokens = token_ids_all if token_ids_all is not None else output.activations
            resident_tokens_already_current = (
                route.spec.uses_slot_tokens or prefill_resident_tokens_seeded
            )
            self._record_device_token_carry(
                batch,
                carry_tokens,
                active_rows=active_rows,
                prefill_final_flags=prefill_final_flags,
                seqs=seqs,
                update_resident_tokens=not resident_tokens_already_current,
                resident_tokens_already_current=resident_tokens_already_current,
            )
        elif route.spec.tokens is TokenMode.BURST and carry_device_tokens:
            self._record_device_token_carry(
                batch,
                output.activations[:, -1:],
                active_rows=active_rows,
                prefill_final_flags=prefill_final_flags,
                seqs=seqs,
            )
        else:
            self._clear_device_token_carry()

        token_by_row: dict[int, Any] = {}
        token_list_by_row: dict[int, list[int]] = {}
        if active_rows:
            if decode_burst_steps > 1:
                token_rows = token_ids_all
                if active_rows != list(range(len(seqs))):
                    token_rows = token_rows[jnp.array(active_rows, dtype=jnp.int32)]
                if carry_device_tokens:
                    burst_width = int(token_rows.shape[1])
                    token_list_by_row = {
                        row: [
                            DeviceTokenRef(tokens=token_rows, row=index * burst_width + step)
                            for step in range(burst_width)
                        ]
                        for index, row in enumerate(active_rows)
                    }
                else:
                    token_list_by_row = {
                        row: [int(token_id) for token_id in token_row]
                        for row, token_row in zip(active_rows, token_rows.tolist())
                    }
            elif emits_token_ids:
                token_ids = (
                    token_ids_all
                    if active_rows == list(range(len(seqs)))
                    else token_ids_all[jnp.array(active_rows, dtype=jnp.int32)]
                )
            else:
                active_idx = jnp.array(active_rows, dtype=jnp.int32)
                temperatures = jnp.array(
                    [seqs[row].temperature for row in active_rows], dtype=jnp.float32
                )
                token_ids = self._sample_fn(last_logits[active_idx], temperatures)
            if decode_burst_steps <= 1:
                if carry_device_tokens:
                    token_by_row = {
                        row: DeviceTokenRef(tokens=token_ids, row=index)
                        for index, row in enumerate(active_rows)
                    }
                else:
                    host_token_ids = (
                        token_ids[:, 0]
                        if getattr(token_ids, "ndim", 0) == 2 and int(token_ids.shape[1]) == 1
                        else token_ids
                    )
                    token_by_row = {
                        row: int(token_id)
                        for row, token_id in zip(active_rows, host_token_ids.tolist())
                    }

        outputs: list[int | list[int]] = []
        for row, _seq in enumerate(seqs):
            if row not in token_by_row and row not in token_list_by_row:
                outputs.append([])
                continue
            if batch.is_prefill and not prefill_final_flags[row]:
                outputs.append([])
                continue
            outputs.append(
                token_list_by_row[row] if row in token_list_by_row else token_by_row[row]
            )
        return outputs

    def materialize(self, plan: SchedulePlan) -> DeviceBatch:
        """Turn a host schedule into fixed-shape accelerator inputs."""
        return self.batch_materializer.materialize(plan)

    def execute(
        self,
        seqs: list[Sequence],
        batch: DeviceBatch,
    ) -> RunResult:
        """Execute one materialized engine step."""
        before = dict(self.speculation_stats)
        rows = self._run_main_and_sample(seqs, batch)
        return RunResult.from_rows(
            rows,
            verified_target_tokens=(
                self.speculation_stats["target_tokens"] - before["target_tokens"]
            ),
            draft_tokens=self.speculation_stats["drafted"] - before["drafted"],
            accepted_draft_tokens=(self.speculation_stats["accepted"] - before["accepted"]),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _sample_logits(
        self,
        logits: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        import jax.lax as lax

        def sample_single(logit, temp):
            def greedy(_):
                return jnp.argmax(logit)

            def sample(_):
                scaled = logit / temp
                return jax.random.categorical(jax.random.PRNGKey(0), scaled)

            return lax.cond(temp == 0.0, greedy, sample, None)

        return jax.vmap(sample_single)(logits, temperatures)
