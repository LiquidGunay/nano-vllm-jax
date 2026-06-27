"""Persistent accelerator-state runner.

Owns:
    KV cache arrays, GDN hybrid-state slots, resident decode metadata, device
    token carry, and compile-bucket lookup.
Receives:
    Completed ``ScheduledBatch`` objects from the scheduler.
Returns:
    Generated token ids or device token references for scheduled sequences.
Invariant:
    The runner does not decide which requests should run; it only executes the
    already-built batch and advances resident device state.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from functools import partial
from dataclasses import replace

from nanovllm_jax.ops import ServingOps, ServingOpsProtocol
from nanovllm_jax.batch import ScheduledBatch
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import ModelParams
from nanovllm_jax.sequence import DeviceTokenRef, Sequence
from nanovllm_jax.cache import (
    KVCacheState,
    KVCacheSpec,
    cap_num_kv_cache_blocks,
    init_kv_cache,
    init_hybrid_state,
    compute_slot_mapping,
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


def _config_flag(config: Qwen3_5Config | None, attr: str, *, default: bool = False) -> bool:
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return bool(default)


def _config_int(config: Qwen3_5Config | None, attr: str, *, default: int = 0) -> int:
    if config is not None and hasattr(config, attr):
        return int(getattr(config, attr) or default)
    return int(default)


def _int32_device_vector(value) -> jnp.ndarray:
    """Return a 1D int32 device vector without re-wrapping existing int32 arrays."""

    if hasattr(value, "dtype") and getattr(value, "dtype", None) == jnp.dtype(jnp.int32):
        if getattr(value, "ndim", None) == 1:
            return value
        return value.reshape(-1)
    return jnp.asarray(value, dtype=jnp.int32).reshape(-1)




from nanovllm_jax.executor import ModelExecutor
from nanovllm_jax.cache import HybridLayerState, KVCacheStorage


class ModelRunner:
    """Canonical engine runner built around ModelExecutor.forward_step()."""

    def __init__(self, config: Qwen3_5Config, params: ModelParams, ops: ServingOpsProtocol | None = None):
        self.config = config
        self.params = params
        self.backend = ops if ops is not None else ServingOps(config=config)
        self.executor = ModelExecutor(config, params, self.backend)
        self.block_size = config.block_size

        max_seqs = int(
            getattr(config, "max_num_resident_seqs", None)
            or getattr(config, "max_num_seqs", 16)
        )
        kv_spec = KVCacheSpec(
            num_layers=config.num_hidden_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.get_dtype(),
            max_kv_cache_bytes=config.max_kv_cache_bytes,
        )
        effective_num_blocks = cap_num_kv_cache_blocks(kv_spec)
        if effective_num_blocks != config.num_kvcache_blocks:
            print(
                "KV cache capped: "
                f"{config.num_kvcache_blocks} -> {effective_num_blocks} blocks "
                f"({config.max_kv_cache_bytes} byte cap)"
            )
            config.num_kvcache_blocks = effective_num_blocks
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        if self.max_blocks_per_seq is None:
            self.max_blocks_per_seq = max(1, effective_num_blocks // max_seqs)
            config.max_blocks_per_seq = self.max_blocks_per_seq
        self.execution = getattr(config, "jax_execution", "eager")
        self.greedy_token_fastpath = _config_flag(
            config,
            "greedy_token_fastpath",
            default=True,
        )
        self.device_token_carry = _config_flag(
            config,
            "device_token_carry",
        )
        self.static_decode_metadata = _config_flag(
            config,
            "static_decode_metadata",
        )
        self.resident_decode_metadata = bool(
            getattr(config, "resident_decode_metadata", False)
        )
        self.static_decode_seq_lens_carry = _config_flag(
            config,
            "static_decode_seq_lens_carry",
        )

        self.cache_storage = self.backend.allocate_kv_cache(
            replace(kv_spec, num_blocks=effective_num_blocks),
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
        )
        self.full_attention_nhd_cache = self.backend.allocate_full_attention_nhd_kv_cache(
            replace(kv_spec, num_blocks=effective_num_blocks),
            full_attention_layers=tuple(
                layer_id
                for layer_id, layer_type in enumerate(config.layer_types)
                if layer_type == "full_attention"
            ),
        )
        self.hybrid_states: Dict[int, HybridLayerState] = {}
        self._max_hybrid_slots = max_seqs
        self._hybrid_slots: Dict[int, int] = {}
        self._free_hybrid_slots: List[int] = list(range(max_seqs))
        self._zeroed_hybrid_slots: set[int] = set(range(max_seqs))

        empty_hybrid_state = init_hybrid_state(
            config=config,
            batch_size=1,
            dtype=config.get_dtype(),
        )
        self._empty_hybrid_state = empty_hybrid_state
        self._hybrid_state_table = init_hybrid_state(
            self.config,
            batch_size=max_seqs,
            dtype=self.config.get_dtype(),
        )
        self._resident_block_tables = jnp.zeros(
            (max_seqs, self.max_blocks_per_seq),
            dtype=jnp.int32,
        )
        self._resident_seq_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_block_tables_host: list[tuple[int, ...]] = [
            tuple(0 for _ in range(self.max_blocks_per_seq))
            for _ in range(max_seqs)
        ]
        self._resident_block_counts_host: list[int] = [0 for _ in range(max_seqs)]
        self._resident_seq_lens_host: list[int] = [0 for _ in range(max_seqs)]
        self._resident_last_tokens = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_rng_counters = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_rng_counter_reset_slots: set[int] = set()
        self._sample_fn = jax.jit(self._sample_logits)
        self._warmup_compiled = False


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
        def _block_until_ready(value: object) -> None:
            _block_until_ready_tree(value)

        summary: dict[str, Any] = {
            "mode": "generic_bucket_startup",
            "execution": getattr(self, "execution", "eager"),
            "prefill_buckets": [],
            "batch_size_buckets": [],
            "prefill_runs": [],
            "prefill_skipped": [],
            "decode_runs": [],
            "decode_block_table_buckets": [],
            "resident_metadata_scatter_runs": [],
            "sampled_token_fastpath_runs": [],
            "include_sampled_routes": bool(include_sampled_routes),
            "already_warmed": bool(self._warmup_compiled),
        }
        if self._warmup_compiled:
            return summary
        if self.execution not in {"decode-jit", "jit"}:
            self._warmup_compiled = True
            return summary

        prefill_buckets = tuple(int(bucket) for bucket in (prefill_token_buckets or ())) or (
            tuple(getattr(self.config, "prefill_token_buckets", ()))
            or tuple(getattr(self.config, "prefill_buckets", ()))
            or (max_prefill_len,)
        )
        batch_buckets = tuple(int(bucket) for bucket in (batch_size_buckets or ())) or (
            tuple(getattr(self.config, "batch_size_buckets", ())) or (max_batch,)
        )
        decode_block_table_buckets = tuple(
            int(bucket) for bucket in (decode_block_table_buckets or ())
        ) or (
            tuple(getattr(self.config, "decode_block_table_buckets", ()) or ())
            or (int(self.max_blocks_per_seq),)
        )
        summary["prefill_buckets"] = list(prefill_buckets)
        summary["batch_size_buckets"] = list(batch_buckets)
        summary["decode_block_table_buckets"] = [int(width) for width in decode_block_table_buckets]

        use_greedy_token_fastpath = bool(
            getattr(
                self,
                "greedy_token_fastpath",
                _config_flag(
                    getattr(self, "config", None),
                    "greedy_token_fastpath",
                    default=True,
                ),
            )
        )
        use_sampled_token_fastpath = (
            bool(include_sampled_routes)
            and bool(
                getattr(
                    self,
                    "sampled_token_fastpath",
                    _config_flag(
                        getattr(self, "config", None),
                        "sampled_token_fastpath",
                        default=True,
                    ),
                )
            )
        )
        hybrid_state_table = getattr(self, "_hybrid_state_table", None)
        table_state_ready = (
            hybrid_state_table is not None
            and hybrid_state_table.conv_state is not None
            and hybrid_state_table.recurrent_state is not None
        )
        use_hybrid_table_decode = (
            use_greedy_token_fastpath
            and table_state_ready
        )
        use_hybrid_table_prefill = (
            use_greedy_token_fastpath
            and table_state_ready
        )
        use_prefill_slot_carry_table = (
            use_hybrid_table_prefill
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
        )
        use_resident_decode = (
            use_hybrid_table_decode
            and bool(getattr(self, "resident_decode_metadata", False))
        )
        use_resident_slot_decode = (
            use_hybrid_table_decode
            and bool(getattr(self, "resident_decode_metadata", False))
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
        )
        use_resident_dense_decode = use_resident_slot_decode
        use_sampled_resident_dense_decode = (
            use_sampled_token_fastpath
            and table_state_ready
            and bool(getattr(self, "resident_decode_metadata", False))
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
            and hasattr(self, "_resident_rng_counters")
        )
        greedy_decode_burst_steps = max(
            1,
            _config_int(
                getattr(self, "config", None),
                "greedy_decode_burst_steps",
                default=1,
            ),
        )

        for prefill_len in prefill_buckets:
            if self.execution != "jit":
                break
            for batch_size in batch_buckets:
                dense_prefill_tokens = int(batch_size) * int(prefill_len)
                max_batched_tokens = int(getattr(self.config, "max_num_batched_tokens", 0) or 0)
                packed_prefill_layout = str(getattr(self.config, "prefill_layout", "packed")).lower() == "packed"
                if (
                    not packed_prefill_layout
                    and max_batched_tokens > 0
                    and dense_prefill_tokens > max_batched_tokens
                ):
                    summary["prefill_skipped"].append(
                        {
                            "batch_size": int(batch_size),
                            "query_len": int(prefill_len),
                            "dense_prefill_tokens": dense_prefill_tokens,
                            "max_num_batched_tokens": max_batched_tokens,
                            "reason": "dense_prefill_tokens_exceed_budget",
                        }
                    )
                    continue
                batch = self._dummy_batch(batch_size=batch_size, query_len=prefill_len, is_prefill=True)
                hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                if use_prefill_slot_carry_table:
                    hybrid_slot_ids = jnp.arange(int(batch_size), dtype=jnp.int32)
                    batch.hybrid_slot_ids_host = tuple(range(int(batch_size)))
                    output = self.executor.forward_prefill_token_ids_slot_carry_table_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state_table=self._hybrid_state_table,
                        hybrid_slot_ids=hybrid_slot_ids,
                        prefill_final_flags=self._prefill_final_flags_device(batch),
                        resident_last_tokens=self._resident_last_tokens,
                    )
                    self._hybrid_state_table = output.hybrid_state
                    if output.resident_last_tokens is not None:
                        self._resident_last_tokens = output.resident_last_tokens
                    route = "forward_prefill_token_ids_slot_carry_table_jit:prefill"
                elif use_hybrid_table_prefill:
                    hybrid_slot_ids = jnp.arange(int(batch_size), dtype=jnp.int32)
                    batch.hybrid_slot_ids_host = tuple(range(int(batch_size)))
                    output = self.executor.forward_prefill_token_ids_table_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state_table=self._hybrid_state_table,
                        hybrid_slot_ids=hybrid_slot_ids,
                    )
                    self._hybrid_state_table = output.hybrid_state
                    route = "forward_prefill_token_ids_table_jit:prefill"
                elif use_greedy_token_fastpath:
                    output = self.executor.forward_step_token_ids_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=hybrid_state,
                    )
                    route = "forward_step_token_ids_jit:prefill"
                else:
                    output = self.executor.forward_step_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=hybrid_state,
                        last_logits_only=True,
                    )
                    route = "forward_step_jit:prefill"
                _block_until_ready(output)
                self.cache_storage = output.cache_storage
                summary["prefill_runs"].append(
                    {
                        "batch_size": int(batch_size),
                        "query_len": int(prefill_len),
                        "tokens_shape": list(batch.tokens.shape),
                        "block_tables_shape": list(batch.block_tables.shape),
                        "num_prefill_tokens": int(batch.num_prefill_tokens),
                        "route": route,
                    }
                )
                if use_sampled_token_fastpath:
                    sampled_hybrid_state = init_hybrid_state(
                        self.config,
                        batch_size=batch_size,
                        dtype=self.config.get_dtype(),
                    )
                    temperatures = jnp.ones((int(batch_size),), dtype=jnp.float32)
                    rng_slots = jnp.arange(int(batch_size), dtype=jnp.int32)
                    rng_counters = jnp.zeros((int(batch_size),), dtype=jnp.int32)
                    sampled_output = self.executor.forward_step_sampled_token_ids_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=sampled_hybrid_state,
                        temperatures=temperatures,
                        rng_counters=rng_counters,
                        rng_slots=rng_slots,
                    )
                    _block_until_ready(sampled_output)
                    self.cache_storage = sampled_output.cache_storage
                    summary["sampled_token_fastpath_runs"].append(
                        {
                            "kind": "prefill",
                            "batch_size": int(batch_size),
                            "query_len": int(prefill_len),
                            "route": "forward_step_sampled_token_ids_jit:prefill",
                        }
                    )

        for batch_size in batch_buckets:
            for block_table_width in decode_block_table_buckets:
                batch = self._dummy_batch(
                    batch_size=batch_size,
                    query_len=1,
                    is_prefill=False,
                    max_blocks_per_seq=int(block_table_width),
                )

                def _record_decode_warmup(output, route: str, decode_steps: int = 1) -> None:
                    _block_until_ready(output)
                    self.cache_storage = output.cache_storage
                    self._sample_fn(
                        jnp.zeros((batch_size, self.config.vocab_size), dtype=jnp.float32),
                        jnp.zeros((batch_size,), dtype=jnp.float32),
                    ).block_until_ready()
                    summary["decode_runs"].append(
                        {
                            "batch_size": int(batch_size),
                            "tokens_shape": list(batch.tokens.shape),
                            "block_tables_shape": list(batch.block_tables.shape),
                            "num_decode_tokens": int(batch.num_decode_tokens),
                            "route": route,
                            "decode_steps": int(decode_steps),
                        }
                    )

                if use_greedy_token_fastpath and greedy_decode_burst_steps > 1:
                    for burst_steps in range(2, greedy_decode_burst_steps + 1):
                        if use_hybrid_table_decode:
                            hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                            batch.hybrid_slot_ids_host = tuple(range(batch_size))
                            output = self.executor.forward_greedy_decode_burst_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                decode_steps=burst_steps,
                            )
                            self._hybrid_state_table = output.hybrid_state
                            _record_decode_warmup(output, "forward_greedy_decode_burst_table_jit:decode", burst_steps)
                        else:
                            hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                            output = self.executor.forward_greedy_decode_burst_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=hybrid_state,
                                decode_steps=burst_steps,
                            )
                            _record_decode_warmup(output, "forward_greedy_decode_burst_jit:decode", burst_steps)
                if use_greedy_token_fastpath:
                    if use_hybrid_table_decode:
                        hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                        batch.hybrid_slot_ids_host = tuple(range(batch_size))
                        if use_resident_dense_decode:
                            self._sync_resident_decode_metadata(batch, list(batch.hybrid_slot_ids_host), sync_seq_lens=True)
                            output = self.executor.forward_step_token_ids_resident_dense_slot_carry_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_block_tables=self._resident_block_tables,
                                resident_seq_lens=self._resident_seq_lens,
                                resident_last_tokens=self._resident_last_tokens,
                            )
                            route = "forward_step_token_ids_resident_dense_slot_carry_jit:decode"
                        elif use_resident_slot_decode:
                            self._sync_resident_decode_metadata(batch, list(batch.hybrid_slot_ids_host), sync_seq_lens=True)
                            output = self.executor.forward_step_token_ids_resident_slot_carry_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_block_tables=self._resident_block_tables,
                                resident_seq_lens=self._resident_seq_lens,
                                resident_last_tokens=self._resident_last_tokens,
                            )
                            route = "forward_step_token_ids_resident_slot_carry_jit:decode"
                        elif use_resident_decode:
                            self._sync_resident_decode_metadata(batch, list(batch.hybrid_slot_ids_host), sync_seq_lens=True)
                            output = self.executor.forward_step_token_ids_resident_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_block_tables=self._resident_block_tables,
                                resident_seq_lens=self._resident_seq_lens,
                            )
                            route = "forward_step_token_ids_resident_jit:decode"
                        else:
                            output = self.executor.forward_step_token_ids_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                            )
                            route = "forward_step_token_ids_table_jit:decode"
                        self._hybrid_state_table = output.hybrid_state
                        if output.resident_seq_lens is not None:
                            self._resident_seq_lens = output.resident_seq_lens
                        if output.resident_last_tokens is not None:
                            self._resident_last_tokens = output.resident_last_tokens
                    else:
                        hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                        output = self.executor.forward_step_token_ids_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=hybrid_state,
                        )
                        route = "forward_step_token_ids_jit:decode"
                    _record_decode_warmup(output, route)
                if use_sampled_token_fastpath:
                    temperatures = jnp.ones((int(batch_size),), dtype=jnp.float32)
                    if use_sampled_resident_dense_decode:
                        hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                        batch.hybrid_slot_ids_host = tuple(range(batch_size))
                        self._sync_resident_decode_metadata(batch, list(batch.hybrid_slot_ids_host), sync_seq_lens=True)
                        sampled_output = self.executor.forward_step_sampled_token_ids_resident_dense_slot_carry_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                            resident_last_tokens=self._resident_last_tokens,
                            resident_rng_counters=self._resident_rng_counters,
                            temperatures=temperatures,
                        )
                        self._hybrid_state_table = sampled_output.hybrid_state
                        if sampled_output.resident_seq_lens is not None:
                            self._resident_seq_lens = sampled_output.resident_seq_lens
                        if sampled_output.resident_last_tokens is not None:
                            self._resident_last_tokens = sampled_output.resident_last_tokens
                        if sampled_output.resident_rng_counters is not None:
                            self._resident_rng_counters = sampled_output.resident_rng_counters
                        sampled_route = "forward_step_sampled_token_ids_resident_dense_slot_carry_jit:decode"
                    else:
                        sampled_hybrid_state = init_hybrid_state(
                            self.config,
                            batch_size=batch_size,
                            dtype=self.config.get_dtype(),
                        )
                        rng_slots = jnp.arange(int(batch_size), dtype=jnp.int32)
                        rng_counters = jnp.zeros((int(batch_size),), dtype=jnp.int32)
                        sampled_output = self.executor.forward_step_sampled_token_ids_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=sampled_hybrid_state,
                            temperatures=temperatures,
                            rng_counters=rng_counters,
                            rng_slots=rng_slots,
                        )
                        sampled_route = "forward_step_sampled_token_ids_jit:decode"
                    _record_decode_warmup(sampled_output, sampled_route)
                    summary["sampled_token_fastpath_runs"].append(
                        {
                            "kind": "decode",
                            "batch_size": int(batch_size),
                            "block_tables_shape": list(batch.block_tables.shape),
                            "route": sampled_route,
                        }
                    )
        if bool(getattr(self, "resident_decode_metadata", False)):
            for row_count in range(1, int(max(batch_buckets)) + 1):
                if row_count > int(self._resident_block_tables.shape[0]):
                    break
                slots = jnp.arange(row_count, dtype=jnp.int32)
                block_rows = jnp.zeros((row_count, int(self._resident_block_tables.shape[1])), dtype=jnp.int32)
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
                _block_until_ready(self._resident_block_tables)
                _block_until_ready(self._resident_seq_lens)
                _block_until_ready(self._resident_last_tokens)
                summary["resident_metadata_scatter_runs"].append(
                    {
                        "row_count": int(row_count),
                        "block_rows_shape": list(block_rows.shape),
                        "seq_lens_shape": list(seq_lens.shape),
                        "last_tokens_shape": list(last_tokens.shape),
                    }
                )
        self._reset_runtime_state_after_warmup()
        if hasattr(self, "_hybrid_state_table"):
            _block_until_ready(self._hybrid_state_table)
        if hasattr(self, "_resident_block_tables"):
            _block_until_ready(self._resident_block_tables)
        if hasattr(self, "_resident_seq_lens"):
            _block_until_ready(self._resident_seq_lens)
        if hasattr(self, "_resident_last_tokens"):
            _block_until_ready(self._resident_last_tokens)
        if hasattr(self, "_resident_rng_counters"):
            _block_until_ready(self._resident_rng_counters)
        summary["state_reset_after_warmup"] = True
        self._warmup_compiled = True
        return summary

    def _dummy_batch(
        self,
        *,
        batch_size: int,
        query_len: int,
        is_prefill: bool,
        max_blocks_per_seq: int | None = None,
    ) -> ScheduledBatch:
        block_tables = []
        num_blocks = max(1, int(getattr(self.config, "num_kvcache_blocks", 1) or 1))
        block_table_width = int(max_blocks_per_seq or self.max_blocks_per_seq)
        for row in range(batch_size):
            start = row * block_table_width
            block_tables.append(
                [
                    (start + offset) % num_blocks
                    for offset in range(block_table_width)
                ]
            )
        if is_prefill and str(getattr(self.config, "prefill_layout", "packed")).lower() == "packed":
            token_bucket = int(query_len)
            base = token_bucket // batch_size
            rem = token_bucket % batch_size
            query_lens = [base + (1 if row < rem else 0) for row in range(batch_size)]
            query_start_loc = [0]
            packed_positions = []
            token_row_ids = []
            for row, qlen in enumerate(query_lens):
                query_start_loc.append(query_start_loc[-1] + qlen)
                packed_positions.extend(range(qlen))
                token_row_ids.extend([row] * qlen)
            return ScheduledBatch(
                tokens=jnp.zeros((1, token_bucket), dtype=jnp.int32),
                positions=jnp.array([packed_positions], dtype=jnp.int32),
                seq_ids=jnp.arange(batch_size, dtype=jnp.int32),
                query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
                is_prefill=True,
                num_prefill_tokens=token_bucket,
                num_decode_tokens=0,
                block_tables=jnp.array(block_tables, dtype=jnp.int32),
                seq_lens=jnp.array(query_lens, dtype=jnp.int32),
                seq_ids_host=tuple(range(batch_size)),
                query_lens_host=tuple(query_lens),
                seq_lens_host=tuple(query_lens),
                block_tables_host=tuple(tuple(int(block) for block in row) for row in block_tables),
                packed_prefill=True,
                token_row_ids=jnp.array([token_row_ids], dtype=jnp.int32),
            )

        query_lens = [query_len if is_prefill else 1] * batch_size
        query_start_loc = [0]
        for qlen in query_lens:
            query_start_loc.append(query_start_loc[-1] + qlen)
        positions = [list(range(query_len)) for _ in range(batch_size)]
        return ScheduledBatch(
            tokens=jnp.zeros((batch_size, query_len), dtype=jnp.int32),
            positions=jnp.array(positions, dtype=jnp.int32),
            seq_ids=jnp.arange(batch_size, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else batch_size,
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.full((batch_size,), query_len if is_prefill else 1, dtype=jnp.int32),
            seq_ids_host=tuple(range(batch_size)),
            query_lens_host=tuple(query_lens),
            seq_lens_host=tuple([query_len if is_prefill else 1] * batch_size),
            block_tables_host=tuple(tuple(int(block) for block in row) for row in block_tables),
        )

    def release(self, seq_ids: List[int]):
        """Release per-sequence hybrid state once a request is finished."""
        released_slots: list[int] = []
        for seq_id in seq_ids:
            self.hybrid_states.pop(seq_id, None)
            slot = self._hybrid_slots.pop(seq_id, None)
            if slot is not None:
                self._free_hybrid_slots.append(slot)
                released_slots.append(int(slot))
                if hasattr(self, "_resident_block_tables_host"):
                    self._resident_block_tables_host[slot] = tuple(
                        0 for _ in range(self.max_blocks_per_seq)
                    )
                if hasattr(self, "_resident_block_counts_host"):
                    self._resident_block_counts_host[slot] = 0
                if hasattr(self, "_resident_seq_lens_host"):
                    self._resident_seq_lens_host[slot] = 0
                if hasattr(self, "_resident_rng_counters"):
                    reset_slots = getattr(
                        self,
                        "_resident_rng_counter_reset_slots",
                        None,
                    )
                    if reset_slots is None:
                        reset_slots = set()
                        self._resident_rng_counter_reset_slots = reset_slots
                    reset_slots.add(int(slot))
            if hasattr(self, "_resident_last_tokens_stale_seq_ids"):
                self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        carry_by_seq_id = getattr(self, "_device_token_carry_by_seq_id", {})
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
        if hasattr(self, "hybrid_states"):
            self.hybrid_states.clear()
        if hasattr(self, "_hybrid_slots"):
            self._hybrid_slots.clear()
        if hasattr(self, "_max_hybrid_slots"):
            self._free_hybrid_slots = list(range(self._max_hybrid_slots))
            self._zeroed_hybrid_slots = set()
            self._zero_hybrid_slots(tuple(range(self._max_hybrid_slots)))
            self._zeroed_hybrid_slots = set(range(self._max_hybrid_slots))
        if hasattr(self, "_clear_device_token_carry"):
            self._clear_device_token_carry()
        if hasattr(self, "_resident_block_tables"):
            self._resident_block_tables = jnp.zeros_like(self._resident_block_tables)
        if hasattr(self, "_resident_seq_lens"):
            self._resident_seq_lens = jnp.zeros_like(self._resident_seq_lens)
        if hasattr(self, "_resident_last_tokens"):
            self._resident_last_tokens = jnp.zeros_like(self._resident_last_tokens)
        if hasattr(self, "_resident_rng_counters"):
            self._resident_rng_counters = jnp.zeros_like(self._resident_rng_counters)
        if hasattr(self, "_resident_rng_counter_reset_slots"):
            self._resident_rng_counter_reset_slots.clear()
        if hasattr(self, "_resident_last_tokens_stale_seq_ids"):
            self._resident_last_tokens_stale_seq_ids.clear()
        if hasattr(self, "_resident_block_tables_host") and hasattr(self, "_max_hybrid_slots"):
            self._resident_block_tables_host = [
                tuple(0 for _ in range(self.max_blocks_per_seq))
                for _ in range(self._max_hybrid_slots)
            ]
        if hasattr(self, "_resident_block_counts_host") and hasattr(self, "_max_hybrid_slots"):
            self._resident_block_counts_host = [0 for _ in range(self._max_hybrid_slots)]
        if hasattr(self, "_resident_seq_lens_host") and hasattr(self, "_max_hybrid_slots"):
            self._resident_seq_lens_host = [0 for _ in range(self._max_hybrid_slots)]

    def _clear_device_token_carry(self) -> None:
        self._device_token_carry_seq_ids = None
        self._device_token_carry_tokens = None
        self._device_token_carry_by_seq_id = {}
        self._device_seq_lens_carry_seq_ids = None
        self._device_seq_lens_carry = None

    @staticmethod
    def _active_decode_rows_host(batch: ScheduledBatch) -> List[int]:
        if batch.seq_ids_host is None or batch.query_lens_host is None:
            return []
        return [
            row
            for row, (seq_id, query_len) in enumerate(zip(batch.seq_ids_host, batch.query_lens_host))
            if int(seq_id) >= 0 and int(query_len) > 0
        ]

    def _maybe_apply_device_token_carry(self, batch: ScheduledBatch) -> ScheduledBatch:
        static_decode_metadata = bool(getattr(batch, "uses_static_decode_metadata", False))
        active_rows = self._active_decode_rows_host(batch)
        carry_enabled = bool(
            getattr(
                self,
                "device_token_carry",
                _config_flag(
                    getattr(self, "config", None),
                    "device_token_carry",
                ),
            )
        )
        if (
            carry_enabled
            and batch.is_prefill
            and batch.packed_prefill
            and getattr(batch, "mixed_prefill_decode", False)
            and getattr(self, "_device_token_carry_by_seq_id", {})
            and batch.seq_ids_host is not None
            and batch.query_lens_host is not None
            and batch.tokens.shape[0] == 1
        ):
            tokens = batch.tokens
            offset = 0
            applied = False
            for row, (seq_id, query_len) in enumerate(zip(batch.seq_ids_host, batch.query_lens_host)):
                query_len = int(query_len)
                if int(seq_id) >= 0 and query_len == 1:
                    token_ref = self._device_token_carry_by_seq_id.get(int(seq_id))
                    if token_ref is not None:
                        token_array = jnp.asarray(token_ref.tokens, dtype=jnp.int32)
                        if token_array.ndim == 2 and token_array.shape[1] == 1:
                            token_value = token_array[int(token_ref.row), 0]
                        else:
                            token_value = _int32_device_vector(token_array)[int(token_ref.row)]
                        tokens = tokens.at[0, offset].set(token_value)
                        applied = True
                offset += query_len
            if applied:
                return replace(batch, tokens=tokens)

        if (
            not carry_enabled
            or batch.is_prefill
            or not getattr(self, "_device_token_carry_by_seq_id", {})
            or batch.seq_ids_host is None
            or batch.tokens.shape[1] != 1
        ):
            if static_decode_metadata:
                raise RuntimeError("static decode metadata requires a device-token carry for every active row")
            return batch

        carried_seq_ids = getattr(self, "_device_token_carry_seq_ids", None)
        carried_tokens = getattr(self, "_device_token_carry_tokens", None)
        carried_seq_lens_ids = getattr(self, "_device_seq_lens_carry_seq_ids", None)
        carried_seq_lens = getattr(self, "_device_seq_lens_carry", None)
        use_seq_lens_carry = bool(
            getattr(
                self,
                "static_decode_seq_lens_carry",
                _config_flag(
                    getattr(self, "config", None),
                    "static_decode_seq_lens_carry",
                ),
            )
        )
        tokens = batch.tokens
        seq_lens = batch.seq_lens

        if (
            carried_seq_ids is not None
            and tuple(batch.seq_ids_host) == carried_seq_ids
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

        missing_static_rows: List[int] = []
        if not applied:
            for row, seq_id in enumerate(batch.seq_ids_host):
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
                and tuple(batch.seq_ids_host) == carried_seq_lens_ids
                and carried_seq_lens is not None
            ):
                seq_lens_vector = _int32_device_vector(carried_seq_lens)
                if seq_lens_vector.shape[0] == int(seq_lens.shape[0]):
                    seq_lens = seq_lens_vector
                    seq_lens_applied = True
            if not seq_lens_applied and carried_seq_lens is not None and carried_seq_lens_ids is not None:
                seq_lens_vector = _int32_device_vector(carried_seq_lens)
                seq_id_to_row = {int(seq_id): row for row, seq_id in enumerate(carried_seq_lens_ids)}
                for row, seq_id in enumerate(batch.seq_ids_host):
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
        batch: ScheduledBatch,
        *,
        active_rows: list[int],
    ) -> bool:
        if (
            batch.is_prefill
            or not bool(getattr(batch, "uses_static_decode_metadata", False))
            or batch.seq_ids_host is None
            or not active_rows
            or not hasattr(self, "_resident_last_tokens")
        ):
            return False
        carry_by_seq_id = getattr(self, "_device_token_carry_by_seq_id", {})
        if not carry_by_seq_id:
            return False
        hybrid_slots = getattr(self, "_hybrid_slots", {})
        stale_seq_ids = getattr(self, "_resident_last_tokens_stale_seq_ids", set())
        for row in active_rows:
            seq_id = int(batch.seq_ids_host[row])
            if seq_id in stale_seq_ids:
                return False
            if seq_id < 0 or seq_id not in carry_by_seq_id or seq_id not in hybrid_slots:
                return False
        return True

    def _resident_slot_token_dense_decode_ready(
        self,
        batch: ScheduledBatch,
        *,
        active_rows: list[int],
    ) -> bool:
        if not self._resident_slot_token_decode_ready(batch, active_rows=active_rows):
            return False
        batch_size = int(batch.tokens.shape[0])
        if active_rows != list(range(batch_size)):
            return False
        query_lens = (
            list(batch.query_lens_host)
            if batch.query_lens_host is not None
            else [int(x) for x in batch.query_lens[:batch_size].tolist()]
        )
        if len(query_lens) < batch_size or any(int(query_lens[row]) != 1 for row in range(batch_size)):
            return False
        seq_ids = list(batch.seq_ids_host or ())
        if len(seq_ids) != batch_size:
            return False
        hybrid_slots = getattr(self, "_hybrid_slots", {})
        slot_values = [int(hybrid_slots.get(int(seq_id), -1)) for seq_id in seq_ids]
        return all(slot >= 0 for slot in slot_values) and len(set(slot_values)) == len(slot_values)

    def _record_resident_last_tokens(
        self,
        batch: ScheduledBatch,
        token_ids: jnp.ndarray,
        *,
        eligible_rows: list[int],
        active_row_to_token_row: dict[int, int],
        full_batch_tokens: bool,
    ) -> None:
        if (
            not eligible_rows
            or not hasattr(self, "_resident_last_tokens")
            or batch.seq_ids_host is None
        ):
            return
        slot_values = list(batch.hybrid_slot_ids_host or ())
        if not slot_values:
            slot_values = [
                self._hybrid_slots.get(int(seq_id), -1)
                for seq_id in batch.seq_ids_host
            ]
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
        batch: ScheduledBatch,
        token_ids: jnp.ndarray,
        *,
        active_rows: list[int],
        prefill_final_flags: list[bool],
        seqs: List[Sequence],
        update_resident_tokens: bool = True,
        resident_tokens_already_current: bool = False,
    ) -> None:
        if (
            not bool(
                getattr(
                    self,
                    "device_token_carry",
                    _config_flag(
                        getattr(self, "config", None),
                        "device_token_carry",
                    ),
                )
            )
            or batch.seq_ids_host is None
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
        carry_by_seq_id: dict[int, DeviceTokenRef] = dict(
            getattr(self, "_device_token_carry_by_seq_id", {})
        )
        new_carry_by_seq_id: dict[int, DeviceTokenRef] = {}
        for row in eligible_rows:
            seq_id = int(batch.seq_ids_host[row])
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
            if hasattr(self, "_resident_last_tokens_stale_seq_ids"):
                for seq_id in new_carry_by_seq_id:
                    self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        elif resident_tokens_already_current and hasattr(
            self,
            "_resident_last_tokens_stale_seq_ids",
        ):
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        elif hasattr(self, "_resident_last_tokens_stale_seq_ids"):
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.add(int(seq_id))
        self._device_token_carry_seq_ids = (
            tuple(int(seq_id) for seq_id in batch.seq_ids_host)
            if full_batch_tokens
            else tuple(new_carry_by_seq_id)
        )
        self._device_token_carry_tokens = token_ids
        self._device_token_carry_by_seq_id = carry_by_seq_id
        use_seq_lens_carry = bool(
            getattr(
                self,
                "static_decode_seq_lens_carry",
                _config_flag(
                    getattr(self, "config", None),
                    "static_decode_seq_lens_carry",
                ),
            )
        )
        if batch.is_prefill:
            self._device_seq_lens_carry_seq_ids = None
            self._device_seq_lens_carry = None
        elif use_seq_lens_carry:
            self._device_seq_lens_carry_seq_ids = tuple(int(seq_id) for seq_id in batch.seq_ids_host)
            if active_rows == list(range(int(batch.tokens.shape[0]))):
                self._device_seq_lens_carry = batch.seq_lens.astype(jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
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
        batch: ScheduledBatch,
    ) -> ScheduledBatch:
        """Build a concrete decode batch from static/resident scheduler metadata.

        Static resident decode batches intentionally carry placeholder token
        metadata because the resident fast path gathers block tables, sequence
        lengths, positions, and last tokens inside the compiled boundary.
        """
        if (
            batch.is_prefill
            or not bool(getattr(batch, "uses_static_decode_metadata", False))
            or batch.block_tables_host is None
            or batch.seq_lens_host is None
        ):
            return batch

        batch_size = int(batch.tokens.shape[0])
        query_width = int(batch.tokens.shape[1])
        seq_lens_host = tuple(int(x) for x in batch.seq_lens_host)
        block_tables_host = tuple(
            tuple(int(block) for block in row)
            for row in batch.block_tables_host
        )
        seq_ids_host = tuple(
            int(seq_id)
            for seq_id in (
                batch.seq_ids_host
                if batch.seq_ids_host is not None
                else tuple(int(x) for x in jax.device_get(batch.seq_ids).tolist())
            )
        )
        query_lens_host = tuple(
            int(query_len)
            for query_len in (
                batch.query_lens_host
                if batch.query_lens_host is not None
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
            uses_static_decode_metadata=False,
        )


    def _device_token_carry_enabled(self) -> bool:
        return bool(
            getattr(
                self,
                "device_token_carry",
                _config_flag(
                    getattr(self, "config", None),
                    "device_token_carry",
                ),
            )
        )

    @staticmethod
    def _materialize_device_token_outputs(
        outputs: dict[int, List[object] | object],
    ) -> dict[int, List[int] | int]:
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

        materialized: dict[int, List[int] | int] = {}
        for row, value in outputs.items():
            if isinstance(value, list):
                materialized[int(row)] = [resolve_token(token) for token in value]
            else:
                materialized[int(row)] = resolve_token(value)
        return materialized

    def _build_scheduled_batch(self, seqs: List[Sequence], is_prefill: bool) -> ScheduledBatch:
        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        actual_max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        max_blocks = actual_max_blocks
        if self.max_blocks_per_seq is not None:
            if actual_max_blocks > self.max_blocks_per_seq:
                raise ValueError(
                    f"scheduled block table needs {actual_max_blocks} blocks but bucket has {self.max_blocks_per_seq}"
                )
            max_blocks = self.max_blocks_per_seq
        if not is_prefill:
            decode_block_table_buckets = tuple(getattr(self.config, "decode_block_table_buckets", ()) or ())
            if decode_block_table_buckets:
                max_blocks = self._select_bucket(actual_max_blocks, decode_block_table_buckets, "decode block table")
                if self.max_blocks_per_seq is not None and max_blocks > self.max_blocks_per_seq:
                    raise ValueError(
                        f"decode block table bucket {max_blocks} exceeds max_blocks_per_seq {self.max_blocks_per_seq}"
                    )
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
        query_len_bucket = max_query_len
        prefill_buckets = tuple(getattr(self.config, "prefill_buckets", ()))
        if is_prefill and prefill_buckets:
            query_len_bucket = self._select_bucket(max_query_len, prefill_buckets, "prefill")

        batch_size_bucket = len(seqs)
        batch_size_buckets = tuple(getattr(self.config, "batch_size_buckets", ()))
        if batch_size_buckets:
            batch_size_bucket = self._select_bucket(len(seqs), batch_size_buckets, "batch")

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

        seq_ids_host = tuple([seq.seq_id for seq in seqs] + [-1] * (batch_size_bucket - len(seqs)))
        query_lens_host = tuple(query_lens)
        return ScheduledBatch(
            tokens=jnp.array(padded_tokens, dtype=jnp.int32),
            positions=jnp.array(padded_positions, dtype=jnp.int32),
            seq_ids=jnp.array(seq_ids_host, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else sum(query_lens),
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=tuple(seq_lens),
        )

    @staticmethod
    def _select_bucket(size: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in sorted(buckets):
            if size <= bucket:
                return bucket
        raise ValueError(f"{name} size {size} exceeds configured buckets {buckets}")


    def _zero_hybrid_slot(self, slot: int):
        self._zero_hybrid_slots([slot])

    def _zero_hybrid_slots(self, slots: List[int] | Tuple[int, ...]):
        slots = tuple(int(slot) for slot in slots if int(slot) >= 0)
        if not slots:
            return
        if not hasattr(self, "_zeroed_hybrid_slots"):
            self._zeroed_hybrid_slots = set()
        slots_to_zero = tuple(
            slot for slot in slots if slot not in self._zeroed_hybrid_slots
        )
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

    def _mark_hybrid_slots_written(self, slots: List[int] | Tuple[int, ...]):
        if not hasattr(self, "_zeroed_hybrid_slots"):
            self._zeroed_hybrid_slots = set()
        for slot in slots:
            if int(slot) >= 0:
                self._zeroed_hybrid_slots.discard(int(slot))

    def _assign_hybrid_slot(self, seq_id: int, preferred_slot: int | None = None) -> tuple[int, bool]:
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
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot].set(state.recurrent_state[0])
            if self._hybrid_state_table.recurrent_state is not None and state.recurrent_state is not None
            else self._hybrid_state_table.recurrent_state,
        )

    def hybrid_state_for_sequence(self, seq_id: int) -> HybridLayerState | None:
        if seq_id < 0 or getattr(self, "_hybrid_state_table", None) is None:
            return None
        if self._hybrid_state_table.conv_state is None and self._hybrid_state_table.recurrent_state is None:
            return None
        if seq_id not in self._hybrid_slots:
            return None
        return self._get_hybrid_state(seq_id)

    def hybrid_states_for_sequences(self, seqs: List[Sequence]) -> dict[int, HybridLayerState]:
        states: dict[int, HybridLayerState] = {}
        for seq in seqs:
            state = self.hybrid_state_for_sequence(int(seq.seq_id))
            if state is not None:
                states[int(seq.seq_id)] = state
        return states

    def install_cached_prefix_hybrid_states(
        self,
        seqs: List[Sequence],
        prefix_states: dict[int, HybridLayerState] | None,
    ) -> None:
        if not prefix_states:
            return
        for seq in seqs:
            prefix_hash = getattr(seq, "cached_prefix_hash", None)
            if prefix_hash is None or int(getattr(seq, "num_cached_tokens", 0)) <= 0:
                continue
            if getattr(seq, "cached_prefix_hybrid_seeded", False):
                continue
            state = prefix_states.get(int(prefix_hash))
            if state is None:
                raise RuntimeError(
                    f"missing hybrid prefix state for cached prefix hash {int(prefix_hash)}"
                )
            self._set_hybrid_state(int(seq.seq_id), state)
            seq.cached_prefix_hybrid_seeded = True

    def _slice_batch(self, batch: ScheduledBatch, idx: int) -> ScheduledBatch:
        query_len = int(batch.query_lens[idx])
        block_tables_host = None
        if batch.block_tables_host is not None:
            block_tables_host = (tuple(batch.block_tables_host[idx]),)
        return ScheduledBatch(
            tokens=batch.tokens[idx : idx + 1, :query_len],
            positions=batch.positions[idx : idx + 1, :query_len],
            seq_ids=batch.seq_ids[idx : idx + 1],
            query_start_loc=jnp.array([0, query_len], dtype=jnp.int32),
            is_prefill=batch.is_prefill,
            num_prefill_tokens=query_len if batch.is_prefill else 0,
            num_decode_tokens=0 if batch.is_prefill else 1,
            block_tables=batch.block_tables[idx : idx + 1],
            seq_lens=batch.seq_lens[idx : idx + 1],
            block_tables_host=block_tables_host,
        )

    def _masked_decode_batch(
        self,
        batch: ScheduledBatch,
        rows: List[int],
        *,
        token_values: List[int] | None = None,
        position_values: List[int] | None = None,
        seq_len_values: List[int] | None = None,
    ) -> ScheduledBatch:
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
        block_tables_host = None
        if batch.block_tables_host is not None:
            zero_row = tuple(0 for _ in batch.block_tables_host[0])
            block_tables_host = tuple(
                tuple(batch.block_tables_host[row]) if row in rows else zero_row
                for row in range(batch_size)
            )
        row_set = set(int(row) for row in rows)
        seq_ids_host = None
        if batch.seq_ids_host is not None:
            seq_ids_host = tuple(
                int(batch.seq_ids_host[row]) if row in row_set else -1
                for row in range(batch_size)
            )
        query_lens_host = tuple(1 if row in row_set else 0 for row in range(batch_size))
        seq_lens_host = None
        if seq_len_values is not None:
            row_to_seq_len = {
                int(row): int(value)
                for row, value in zip(rows, seq_len_values)
            }
            seq_lens_host = tuple(
                row_to_seq_len[row] if row in row_to_seq_len else 0
                for row in range(batch_size)
            )
        elif batch.seq_lens_host is not None:
            seq_lens_host = tuple(
                int(batch.seq_lens_host[row]) if row in row_set else 0
                for row in range(batch_size)
            )
        return ScheduledBatch(
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
            block_tables=jnp.where(active[:, None], batch.block_tables, jnp.zeros_like(batch.block_tables)),
            seq_lens=seq_lens,
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            uses_static_decode_metadata=False,
        )

    def _pad_decode_batch_to_rows(self, batch: ScheduledBatch, target_rows: int) -> ScheduledBatch:
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

        seq_ids_host = None
        if batch.seq_ids_host is not None:
            seq_ids_host = tuple(int(x) for x in batch.seq_ids_host) + tuple(-1 for _ in range(pad_rows))
        query_lens_host = None
        if batch.query_lens_host is not None:
            query_lens_host = tuple(int(x) for x in batch.query_lens_host) + tuple(0 for _ in range(pad_rows))
        seq_lens_host = None
        if batch.seq_lens_host is not None:
            seq_lens_host = tuple(int(x) for x in batch.seq_lens_host) + tuple(0 for _ in range(pad_rows))
        block_tables_host = None
        if batch.block_tables_host is not None:
            zero_row = tuple(0 for _ in range(block_width))
            block_tables_host = tuple(tuple(int(block) for block in row) for row in batch.block_tables_host) + tuple(
                zero_row for _ in range(pad_rows)
            )
        hybrid_slot_ids_host = None
        if batch.hybrid_slot_ids_host is not None:
            hybrid_slot_ids_host = tuple(int(x) for x in batch.hybrid_slot_ids_host) + tuple(-1 for _ in range(pad_rows))

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
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            hybrid_slot_ids_host=hybrid_slot_ids_host,
            uses_static_decode_metadata=False,
        )

    def _with_committed_seq_lens(
        self,
        batch: ScheduledBatch,
        committed_seq_lens: jnp.ndarray | None,
    ) -> ScheduledBatch:
        if committed_seq_lens is None:
            return batch
        active = (batch.seq_ids >= 0) & (batch.query_lens > 0)
        seq_lens = jnp.where(
            active,
            jnp.asarray(committed_seq_lens, dtype=jnp.int32),
            batch.seq_lens.astype(jnp.int32),
        )
        seq_lens_host = batch.seq_lens_host
        if seq_lens_host is not None:
            committed_host = np.asarray(jax.device_get(committed_seq_lens), dtype=np.int32).reshape(-1)
            seq_lens_values = [int(value) for value in seq_lens_host]
            seq_ids_host = batch.seq_ids_host
            query_lens_host = batch.query_lens_host
            for row in range(min(len(seq_lens_values), int(committed_host.shape[0]))):
                active_host = True
                if seq_ids_host is not None and row < len(seq_ids_host):
                    active_host = active_host and int(seq_ids_host[row]) >= 0
                if query_lens_host is not None and row < len(query_lens_host):
                    active_host = active_host and int(query_lens_host[row]) > 0
                if active_host:
                    seq_lens_values[row] = int(committed_host[row])
            seq_lens_host = tuple(seq_lens_values)
        return replace(batch, seq_lens=seq_lens, seq_lens_host=seq_lens_host)

    def _compact_decode_batch(
        self,
        batch: ScheduledBatch,
        rows: List[int],
        *,
        token_values: List[int] | None = None,
        position_values: List[int] | None = None,
        seq_len_values: List[int] | None = None,
    ) -> ScheduledBatch:
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
        block_tables_host = None
        if batch.block_tables_host is not None:
            block_tables_host = tuple(tuple(batch.block_tables_host[row]) for row in rows)
        seq_ids_host = None
        if batch.seq_ids_host is not None:
            seq_ids_host = tuple(int(batch.seq_ids_host[row]) for row in rows)
        query_lens_host = None
        if batch.query_lens_host is not None:
            query_lens_host = tuple(int(batch.query_lens_host[row]) for row in rows)
        seq_lens_host = None
        if seq_len_values is not None:
            seq_lens_host = tuple(int(value) for value in seq_len_values)
        elif batch.seq_lens_host is not None:
            seq_lens_host = tuple(int(batch.seq_lens_host[row]) for row in rows)
        hybrid_slot_ids_host = None
        if batch.hybrid_slot_ids_host is not None:
            hybrid_slot_ids_host = tuple(int(batch.hybrid_slot_ids_host[row]) for row in rows)

        compact_size = len(rows)
        return ScheduledBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=batch.seq_ids[row_ids],
            query_start_loc=jnp.arange(compact_size + 1, dtype=jnp.int32),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=compact_size,
            block_tables=batch.block_tables[row_ids],
            seq_lens=seq_lens,
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            hybrid_slot_ids_host=hybrid_slot_ids_host,
        )

    def _batch_hybrid_state(self, batch: ScheduledBatch) -> HybridLayerState:
        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
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
                batch.hybrid_slot_ids_host = tuple(range(len(seq_ids)))
                return self._hybrid_state_table
        slot_allocations = [
            self._assign_hybrid_slot(int(seq_id), preferred_slot=row)
            for row, seq_id in enumerate(seq_ids)
        ]
        slot_values = [slot for slot, _ in slot_allocations]
        newly_allocated = [allocated for _, allocated in slot_allocations]
        self._zero_hybrid_slots(
            [slot for slot, allocated in slot_allocations if allocated]
        )
        batch.hybrid_slot_ids_host = tuple(slot_values)
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(slot_values) == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
            and all(newly_allocated)
        ):
            return self._hybrid_state_table
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(slot_values) == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
            and not any(newly_allocated)
        ):
            return self._hybrid_state_table
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
        return HybridLayerState(conv_state=conv_state, recurrent_state=recurrent_state)

    def _store_batch_hybrid_state(self, batch: ScheduledBatch, state: HybridLayerState | None):
        if state is None:
            return
        valid_rows: List[int] = []
        query_lens = (
            list(batch.query_lens_host)
            if batch.query_lens_host is not None
            else [int(x) for x in batch.query_lens.tolist()]
        )
        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(x) for x in batch.seq_ids.tolist()]
        )
        slot_values_all = (
            list(batch.hybrid_slot_ids_host)
            if batch.hybrid_slot_ids_host is not None
            else [self._ensure_hybrid_slot(seq_id) for seq_id in seq_ids]
        )
        slot_values: List[int] = []
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
            conv_state=self._hybrid_state_table.conv_state.at[slot_ids].set(state.conv_state[row_ids])
            if self._hybrid_state_table.conv_state is not None and state.conv_state is not None
            else self._hybrid_state_table.conv_state,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot_ids].set(state.recurrent_state[row_ids])
            if self._hybrid_state_table.recurrent_state is not None and state.recurrent_state is not None
                else self._hybrid_state_table.recurrent_state,
        )
        self._mark_hybrid_slots_written(slot_values)

    def _batch_hybrid_slot_ids(self, batch: ScheduledBatch) -> jnp.ndarray:
        """Assign hybrid slots for a batch without gathering the state table."""

        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        slot_values: List[int] = []
        for row, seq_id in enumerate(seq_ids):
            slot, allocated = self._assign_hybrid_slot(int(seq_id), preferred_slot=row)
            if allocated:
                self._zero_hybrid_slots([slot])
            slot_values.append(slot)
        batch.hybrid_slot_ids_host = tuple(slot_values)
        slot_key = tuple(slot_values)
        cache = getattr(self, "_hybrid_slot_ids_device_cache", None)
        if cache is None:
            cache = {}
            self._hybrid_slot_ids_device_cache = cache
        cached = cache.get(slot_key)
        if cached is None:
            cached = jax.device_put(np.asarray(slot_key, dtype=np.int32))
            cache[slot_key] = cached
        return cached

    def _prefill_final_flags_device(self, batch: ScheduledBatch) -> jnp.ndarray:
        rows = max(0, int(batch.query_start_loc.shape[0]) - 1)
        flags = [bool(flag) for flag in list(batch.prefill_final_flags)[:rows]]
        if len(flags) < rows:
            flags.extend([False] * (rows - len(flags)))
        key = tuple(flags)
        cache = getattr(self, "_prefill_final_flags_device_cache", None)
        if cache is None:
            cache = {}
            self._prefill_final_flags_device_cache = cache
        cached = cache.get(key)
        if cached is None:
            cached = jax.device_put(np.asarray(key, dtype=bool))
            cache[key] = cached
        return cached

    def _resident_metadata_scatter_fn(self, kind: str, table_shape: tuple[int, ...], update_shape: tuple[int, ...]):
        cache = getattr(self, "_resident_metadata_scatter_cache", None)
        if cache is None:
            cache = {}
            self._resident_metadata_scatter_cache = cache
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
        cache = getattr(self, "_resident_metadata_scatter_cache", None)
        if cache is None:
            cache = {}
            self._resident_metadata_scatter_cache = cache
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

    def _resident_update_slots_device(self, slots: List[int] | Tuple[int, ...]) -> jnp.ndarray:
        key = tuple(int(slot) for slot in slots)
        cache = getattr(self, "_resident_update_slots_device_cache", None)
        if cache is None:
            cache = {}
            self._resident_update_slots_device_cache = cache
        cached = cache.get(key)
        if cached is None:
            cached = jax.device_put(np.asarray(key, dtype=np.int32))
            cache[key] = cached
        return cached

    def _sync_resident_decode_metadata(
        self,
        batch: ScheduledBatch,
        slot_values: List[int] | Tuple[int, ...],
        *,
        sync_seq_lens: bool,
    ) -> None:
        """Refresh resident per-slot paging metadata from scheduler-owned rows.

        Block allocations are still owned by the Python scheduler/block manager.
        This method mirrors only the changed rows into device-resident tables so
        decode can gather compact metadata by slot id inside the JIT boundary.
        """

        if batch.block_tables_host is None:
            return
        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        query_lens = (
            list(batch.query_lens_host)
            if batch.query_lens_host is not None
            else [int(query_len) for query_len in batch.query_lens.tolist()]
        )
        seq_lens = (
            list(batch.seq_lens_host)
            if batch.seq_lens_host is not None
            else [int(seq_len) for seq_len in batch.seq_lens.tolist()]
        )
        if not hasattr(self, "_resident_block_counts_host"):
            self._resident_block_counts_host = [
                0 for _ in range(len(self._resident_block_tables_host))
            ]
        block_size = int(
            getattr(
                self,
                "block_size",
                getattr(getattr(self, "config", None), "block_size", 16),
            )
        )
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
                not batch.is_prefill
                and next_block_count is not None
                and self._resident_block_counts_host[slot] == next_block_count
            )
            if not skip_block_row_check:
                source_row = tuple(int(block) for block in batch.block_tables_host[row])
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
        slot_values: List[int] | Tuple[int, ...],
        *,
        active_rows: List[int],
        steps: int,
    ) -> None:
        if steps <= 0:
            return
        active = set(int(row) for row in active_rows)
        for row, slot in enumerate(slot_values):
            slot = int(slot)
            if slot >= 0 and row in active:
                self._resident_seq_lens_host[slot] += int(steps)

    def _record_resident_committed_seq_lens(self, batch: ScheduledBatch) -> None:
        """Mirror committed per-row decode lengths into resident metadata."""
        if (
            not bool(getattr(self, "resident_decode_metadata", False))
            or not hasattr(self, "_resident_seq_lens")
            or batch.hybrid_slot_ids_host is None
            or batch.seq_ids_host is None
            or batch.query_lens_host is None
        ):
            return
        slots: list[int] = []
        rows: list[int] = []
        for row, (slot, seq_id, query_len) in enumerate(
            zip(batch.hybrid_slot_ids_host, batch.seq_ids_host, batch.query_lens_host)
        ):
            if int(slot) < 0 or int(seq_id) < 0 or int(query_len) <= 0:
                continue
            slots.append(int(slot))
            rows.append(row)
        if not slots:
            return
        row_ids = jnp.asarray(rows, dtype=jnp.int32)
        committed_lens = batch.seq_lens.astype(jnp.int32)[row_ids]
        self._resident_seq_lens = self._scatter_resident_seq_lens(
            self._resident_seq_lens,
            self._resident_update_slots_device(slots),
            committed_lens,
        )
        if hasattr(self, "_resident_seq_lens_host") and batch.seq_lens_host is not None:
            for slot, row in zip(slots, rows):
                if row < len(batch.seq_lens_host):
                    self._resident_seq_lens_host[int(slot)] = int(batch.seq_lens_host[row])

    def _record_resident_committed_seq_lens_host(
        self,
        batch: ScheduledBatch,
        row_to_committed_len: dict[int, int],
    ) -> None:
        """Mirror committed per-row decode lengths into the resident host cache."""
        if (
            not row_to_committed_len
            or not bool(getattr(self, "resident_decode_metadata", False))
            or not hasattr(self, "_resident_seq_lens_host")
            or batch.hybrid_slot_ids_host is None
            or batch.seq_ids_host is None
            or batch.query_lens_host is None
        ):
            return
        for row, committed_len in row_to_committed_len.items():
            row = int(row)
            if (
                row < 0
                or row >= len(batch.hybrid_slot_ids_host)
                or row >= len(batch.seq_ids_host)
                or row >= len(batch.query_lens_host)
            ):
                continue
            slot = int(batch.hybrid_slot_ids_host[row])
            if (
                slot < 0
                or int(batch.seq_ids_host[row]) < 0
                or int(batch.query_lens_host[row]) <= 0
            ):
                continue
            self._resident_seq_lens_host[slot] = int(committed_len)

    def _step_fn(self, batch: ScheduledBatch):
        execution = getattr(self, "execution", "eager")
        if execution == "jit" or (execution == "decode-jit" and not batch.is_prefill):
            return self.executor.forward_step_jit
        return self.executor.forward_step

    def _can_use_greedy_token_fastpath(self, seqs: List[Sequence], batch: ScheduledBatch) -> bool:
        if not bool(
            getattr(
                self,
                "greedy_token_fastpath",
                _config_flag(
                    getattr(self, "config", None),
                    "greedy_token_fastpath",
                    default=True,
                ),
            )
        ):
            return False
        execution = getattr(self, "execution", "eager")
        if execution != "jit" and not (execution == "decode-jit" and not batch.is_prefill):
            return False
        if batch.is_prefill and bool(getattr(self, "_capture_prefill_logits", False)):
            return False
        for seq in seqs:
            if float(getattr(seq, "temperature", 0.0)) != 0.0:
                return False
        return True

    def _can_use_sampled_token_fastpath(self, seqs: List[Sequence], batch: ScheduledBatch) -> bool:
        if not bool(
            getattr(
                self,
                "sampled_token_fastpath",
                _config_flag(
                    getattr(self, "config", None),
                    "sampled_token_fastpath",
                    default=True,
                ),
            )
        ):
            return False
        execution = getattr(self, "execution", "eager")
        if execution != "jit" and not (execution == "decode-jit" and not batch.is_prefill):
            return False
        if batch.is_prefill and bool(getattr(self, "_capture_prefill_logits", False)):
            return False
        has_sampling = False
        for seq in seqs:
            temperature = float(getattr(seq, "temperature", 0.0))
            if temperature < 0.0:
                return False
            if temperature > 0.0:
                has_sampling = True
        return has_sampling

    def _sample_temperatures_device(self, seqs: List[Sequence], batch: ScheduledBatch) -> jnp.ndarray:
        row_count = len(seqs) if batch.is_prefill and batch.packed_prefill else int(batch.tokens.shape[0])
        values = [0.0 for _ in range(row_count)]
        active_limit = min(len(seqs), row_count)
        for row in range(active_limit):
            if batch.query_lens_host is not None and int(batch.query_lens_host[row]) <= 0:
                continue
            values[row] = float(getattr(seqs[row], "temperature", 0.0))
        return jnp.asarray(values, dtype=jnp.float32)

    def _sample_rng_slots_and_counters_device(
        self,
        batch: ScheduledBatch,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self._flush_resident_rng_counter_resets()
        row_count = (
            len(batch.query_lens_host)
            if batch.is_prefill and batch.packed_prefill and batch.query_lens_host is not None
            else int(batch.tokens.shape[0])
        )
        slot_values = list(batch.hybrid_slot_ids_host or ())
        if len(slot_values) < row_count:
            slot_values.extend([-1] * (row_count - len(slot_values)))
        safe_slots = [max(0, int(slot)) for slot in slot_values[:row_count]]
        slot_ids = jnp.asarray(safe_slots, dtype=jnp.int32)
        if hasattr(self, "_resident_rng_counters"):
            counters = self._resident_rng_counters[slot_ids]
        else:
            counters = jnp.zeros((row_count,), dtype=jnp.int32)
        return slot_ids, counters.astype(jnp.int32)

    def _flush_resident_rng_counter_resets(self) -> None:
        """Apply deferred sampled-RNG counter resets before sampled paths read them."""

        reset_slots = getattr(self, "_resident_rng_counter_reset_slots", set())
        if not reset_slots or not hasattr(self, "_resident_rng_counters"):
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
        batch: ScheduledBatch,
        updated_counters: jnp.ndarray | None,
        *,
        active_rows: list[int],
        prefill_final_flags: list[bool],
    ) -> None:
        if updated_counters is None or not hasattr(self, "_resident_rng_counters"):
            return
        self._flush_resident_rng_counter_resets()
        slot_values = list(batch.hybrid_slot_ids_host or ())
        if not slot_values:
            return
        slots: list[int] = []
        rows: list[int] = []
        for row in active_rows:
            if row >= len(slot_values):
                continue
            if batch.is_prefill and (row >= len(prefill_final_flags) or not prefill_final_flags[row]):
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

    def _greedy_decode_burst_steps(self, seqs: List[Sequence], batch: ScheduledBatch) -> int:
        if batch.is_prefill:
            return 1
        configured_steps = max(
            1,
            _config_int(
                getattr(self, "config", None),
                "greedy_decode_burst_steps",
                default=1,
            ),
        )
        if configured_steps <= 1:
            return 1
        if getattr(batch, "decode_step_count_host", 1) <= 1:
            return 1
        if batch.query_lens_host is not None:
            active_query_lens = list(batch.query_lens_host[: len(seqs)])
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
                int(getattr(batch, "decode_step_count_host", 1)),
                min(remaining),
            ),
        )


    def _run_main_and_sample(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
    ) -> List[int | List[int]]:
        if batch.is_prefill:
            prefill_final_flags = list(batch.prefill_final_flags)[: len(seqs)]
            if len(prefill_final_flags) < len(seqs):
                prefill_final_flags.extend([True] * (len(seqs) - len(prefill_final_flags)))
        else:
            prefill_final_flags = [True] * len(seqs)

        use_greedy_token_fastpath = self._can_use_greedy_token_fastpath(seqs, batch)
        use_sampled_token_fastpath = (
            not use_greedy_token_fastpath
            and self._can_use_sampled_token_fastpath(seqs, batch)
        )
        decode_burst_steps = self._greedy_decode_burst_steps(seqs, batch) if use_greedy_token_fastpath else 1
        use_hybrid_table_decode = (
            use_greedy_token_fastpath
            and not batch.is_prefill
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
        )
        use_hybrid_table_prefill = (
            use_greedy_token_fastpath
            and batch.is_prefill
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
        )
        use_sampled_hybrid_table_decode = (
            use_sampled_token_fastpath
            and not batch.is_prefill
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and bool(getattr(self, "resident_decode_metadata", False))
        )
        use_prefill_slot_carry_table = (
            use_hybrid_table_prefill
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
        )
        active_rows_for_carry = self._active_decode_rows_host(batch)
        resident_slot_token_decode = (
            use_hybrid_table_decode
            and decode_burst_steps <= 1
            and not bool(getattr(self, "resident_decode_metadata", False))
            and self._resident_slot_token_decode_ready(batch, active_rows=active_rows_for_carry)
        )
        resident_slot_token_metadata_decode = (
            use_hybrid_table_decode
            and decode_burst_steps <= 1
            and bool(getattr(self, "resident_decode_metadata", False))
            and self._resident_slot_token_decode_ready(batch, active_rows=active_rows_for_carry)
        )
        resident_dense_slot_token_metadata_decode = (
            resident_slot_token_metadata_decode
            and self._resident_slot_token_dense_decode_ready(batch, active_rows=active_rows_for_carry)
        )
        sampled_resident_dense_slot_token_metadata_decode = (
            use_sampled_hybrid_table_decode
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
            and hasattr(self, "_resident_rng_counters")
            and self._resident_slot_token_dense_decode_ready(batch, active_rows=active_rows_for_carry)
        )
        if not (
            resident_slot_token_decode
            or resident_slot_token_metadata_decode
            or sampled_resident_dense_slot_token_metadata_decode
        ):
            batch = self._maybe_apply_device_token_carry(batch)

        if batch.query_lens_host is not None:
            query_lens = [int(x) for x in batch.query_lens_host[: len(seqs)]]
        else:
            query_lens = [int(x) for x in batch.query_lens[: len(seqs)].tolist()]
        if batch.seq_ids_host is not None:
            seq_ids_host = [int(x) for x in batch.seq_ids_host[: len(seqs)]]
        else:
            seq_ids_host = [int(batch.seq_ids[row]) for row in range(len(seqs))]
        active_rows = [row for row, query_len in enumerate(query_lens) if query_len > 0 and seq_ids_host[row] >= 0]

        if use_hybrid_table_decode or use_hybrid_table_prefill or sampled_resident_dense_slot_token_metadata_decode:
            hybrid_slot_ids = self._batch_hybrid_slot_ids(batch)
            hybrid_slot_values = list(batch.hybrid_slot_ids_host or ())
            hybrid_state = self._hybrid_state_table
        else:
            hybrid_slot_ids = None
            hybrid_slot_values = list(batch.hybrid_slot_ids_host or ())
            hybrid_state = self._batch_hybrid_state(batch)
            hybrid_slot_values = list(batch.hybrid_slot_ids_host or hybrid_slot_values)

        use_resident_slot_decode = (
            use_hybrid_table_decode
            and decode_burst_steps <= 1
            and bool(getattr(self, "resident_decode_metadata", False))
            and not resident_slot_token_metadata_decode
        )
        if use_resident_slot_decode or resident_slot_token_metadata_decode or sampled_resident_dense_slot_token_metadata_decode:
            self._sync_resident_decode_metadata(batch, hybrid_slot_values, sync_seq_lens=True)

        prefill_resident_tokens_seeded = False
        if decode_burst_steps > 1:
            if use_hybrid_table_decode:
                output = self.executor.forward_greedy_decode_burst_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    decode_steps=decode_burst_steps,
                )
            else:
                output = self.executor.forward_greedy_decode_burst_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    decode_steps=decode_burst_steps,
                )
        elif use_greedy_token_fastpath:
            if use_prefill_slot_carry_table:
                output = self.executor.forward_prefill_token_ids_slot_carry_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    prefill_final_flags=self._prefill_final_flags_device(batch),
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
                    prefill_resident_tokens_seeded = True
            elif use_hybrid_table_prefill:
                output = self.executor.forward_prefill_token_ids_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                )
            elif resident_dense_slot_token_metadata_decode:
                output = self.executor.forward_step_token_ids_resident_dense_slot_carry_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
            elif resident_slot_token_metadata_decode:
                output = self.executor.forward_step_token_ids_resident_slot_carry_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
            elif use_resident_slot_decode:
                output = self.executor.forward_step_token_ids_resident_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                )
            elif resident_slot_token_decode:
                output = self.executor.forward_step_token_ids_slot_carry_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
            elif use_hybrid_table_decode:
                output = self.executor.forward_step_token_ids_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                )
            else:
                output = self.executor.forward_step_token_ids_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                )
        elif use_sampled_token_fastpath:
            temperatures = self._sample_temperatures_device(seqs, batch)
            if sampled_resident_dense_slot_token_metadata_decode:
                self._flush_resident_rng_counter_resets()
                output = self.executor.forward_step_sampled_token_ids_resident_dense_slot_carry_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                    resident_last_tokens=self._resident_last_tokens,
                    resident_rng_counters=self._resident_rng_counters,
                    temperatures=temperatures,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
                if output.resident_rng_counters is not None:
                    self._resident_rng_counters = output.resident_rng_counters
            else:
                rng_slots, rng_counters = self._sample_rng_slots_and_counters_device(batch)
                output = self.executor.forward_step_sampled_token_ids_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    temperatures=temperatures,
                    rng_counters=rng_counters,
                    rng_slots=rng_slots,
                )
        else:
            output = self._step_fn(batch)(
                batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                return_hidden=False,
                return_hidden_with_logits=False,
                last_logits_only=True,
            )

        self.cache_storage = output.cache_storage
        if use_hybrid_table_decode or use_hybrid_table_prefill or sampled_resident_dense_slot_token_metadata_decode:
            self._hybrid_state_table = output.hybrid_state
            self._mark_hybrid_slots_written(list(batch.hybrid_slot_ids_host or ()))
            if (
                (use_resident_slot_decode or resident_slot_token_metadata_decode or sampled_resident_dense_slot_token_metadata_decode)
                and output.resident_seq_lens is not None
            ):
                self._resident_seq_lens = output.resident_seq_lens
                self._advance_resident_seq_lens_host(hybrid_slot_values, active_rows=active_rows, steps=1)
        else:
            self._store_batch_hybrid_state(batch, output.hybrid_state)
            if use_sampled_token_fastpath:
                self._record_resident_rng_counters(
                    batch,
                    output.resident_rng_counters,
                    active_rows=active_rows,
                    prefill_final_flags=prefill_final_flags,
                )
        if batch.is_prefill and bool(getattr(self, "resident_decode_metadata", False)):
            self._sync_resident_decode_metadata(batch, list(batch.hybrid_slot_ids_host or ()), sync_seq_lens=True)

        token_ids_all = None
        if decode_burst_steps > 1:
            token_ids_all = output.activations[: len(seqs), :decode_burst_steps]
            last_logits = None
        elif use_greedy_token_fastpath or use_sampled_token_fastpath:
            token_ids_all = output.activations[0] if isinstance(output.activations, tuple) else output.activations
            if int(token_ids_all.shape[0]) != len(seqs):
                token_ids_all = token_ids_all[: len(seqs)]
            last_logits = None
        else:
            last_logits = output.activations[: len(seqs), 0]

        carry_device_tokens = (
            (use_greedy_token_fastpath or use_sampled_token_fastpath)
            and bool(
                getattr(
                    self,
                    "device_token_carry",
                    _config_flag(
                        getattr(self, "config", None),
                        "device_token_carry",
                    ),
                )
            )
            and all(seqs[row].ignore_eos for row in active_rows)
        )
        if (use_greedy_token_fastpath or use_sampled_token_fastpath) and decode_burst_steps <= 1:
            carry_tokens = token_ids_all if token_ids_all is not None else output.activations
            resident_tokens_already_current = (
                resident_slot_token_decode
                or resident_slot_token_metadata_decode
                or sampled_resident_dense_slot_token_metadata_decode
                or prefill_resident_tokens_seeded
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
        elif use_greedy_token_fastpath and decode_burst_steps > 1 and carry_device_tokens:
            self._record_device_token_carry(
                batch,
                output.activations[:, -1:],
                active_rows=active_rows,
                prefill_final_flags=prefill_final_flags,
                seqs=seqs,
            )
        else:
            self._clear_device_token_carry()

        if batch.is_prefill and last_logits is not None:
            prefill_logits_by_seq = getattr(self, "_last_prefill_logits_by_seq", None)
            if prefill_logits_by_seq is None:
                prefill_logits_by_seq = {}
                self._last_prefill_logits_by_seq = prefill_logits_by_seq
            for row, seq in enumerate(seqs):
                if row in active_rows and row < len(prefill_final_flags) and prefill_final_flags[row]:
                    prefill_logits_by_seq[int(seq.seq_id)] = last_logits[row]

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
                        row: [DeviceTokenRef(tokens=token_rows, row=index * burst_width + step) for step in range(burst_width)]
                        for index, row in enumerate(active_rows)
                    }
                else:
                    token_list_by_row = {
                        row: [int(token_id) for token_id in token_row]
                        for row, token_row in zip(active_rows, token_rows.tolist())
                    }
            elif use_greedy_token_fastpath or use_sampled_token_fastpath:
                token_ids = token_ids_all if active_rows == list(range(len(seqs))) else token_ids_all[jnp.array(active_rows, dtype=jnp.int32)]
            else:
                active_idx = jnp.array(active_rows, dtype=jnp.int32)
                temperatures = jnp.array([seqs[row].temperature for row in active_rows], dtype=jnp.float32)
                token_ids = self._sample_fn(last_logits[active_idx], temperatures)
            if decode_burst_steps <= 1:
                if carry_device_tokens:
                    token_by_row = {row: DeviceTokenRef(tokens=token_ids, row=index) for index, row in enumerate(active_rows)}
                else:
                    host_token_ids = token_ids[:, 0] if getattr(token_ids, "ndim", 0) == 2 and int(token_ids.shape[1]) == 1 else token_ids
                    token_by_row = {row: int(token_id) for row, token_id in zip(active_rows, host_token_ids.tolist())}

        outputs: List[int | List[int]] = []
        for row, _seq in enumerate(seqs):
            if row not in token_by_row and row not in token_list_by_row:
                outputs.append([])
                continue
            if batch.is_prefill and not prefill_final_flags[row]:
                outputs.append([])
                continue
            outputs.append(token_list_by_row[row] if row in token_list_by_row else token_by_row[row])
        return outputs


    def run(
        self,
        seqs: List[Sequence],
        is_prefill: bool | None = None,
        *,
        batch: ScheduledBatch | None = None,
    ) -> List[int | List[int]]:
        """Run one engine step through the promoted executor path."""
        if batch is None:
            if is_prefill is None:
                raise ValueError("Either is_prefill or batch must be provided")
            batch = self._build_scheduled_batch(seqs, is_prefill=is_prefill)
        return self._run_main_and_sample(seqs, batch)

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
