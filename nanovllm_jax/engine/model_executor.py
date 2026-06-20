"""Canonical model execution path shared by the engine and tests."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from nanovllm_jax.backends import InferenceBackend, select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import AttentionMetadata, HybridLayerState, KVCacheState, KVCacheStorage
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import (
    ModelParams,
    _lm_head_greedy_top1_token_ids,
    forward_step as model_forward_step,
    lm_head_sample_token_ids,
    lm_head_token_ids_and_topk,
)
from nanovllm_jax.mtp.mtp_layer import mtp_forward, mtp_forward_token_ids


_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}


def _config_or_env_flag(config: Qwen3_5Config | None, attr: str, env_name: str) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value in _TRUE_ENV_VALUES
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return False


def _needs_static_prefill_token_count(config: Qwen3_5Config | None = None) -> bool:
    return (
        _config_or_env_flag(config, "compact_prefill_in_proj_qkv", "NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV")
        or _config_or_env_flag(config, "compact_prefill_gdn_z", "NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z")
        or _config_or_env_flag(config, "compact_prefill_full_attn_proj", "NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ")
        or _config_or_env_flag(config, "compact_prefill_mlp", "NANO_VLLM_JAX_COMPACT_PREFILL_MLP")
    )


def _compact_prefill_token_count(
    batch: ScheduledBatch,
    *,
    config: Qwen3_5Config | None = None,
    max_num_batched_tokens: int | None = None,
) -> int:
    env_mode = os.environ.get("NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE")
    mode = (
        env_mode
        if env_mode is not None
        else getattr(config, "compact_prefill_token_count_mode", "exact")
    )
    mode = str(mode or "exact").strip().lower()
    if mode in {"exact", "true", "true_tokens"}:
        return int(batch.num_prefill_tokens)
    if mode in {"bucket", "padded", "padded_bucket"}:
        padded_tokens = int(batch.tokens.shape[0] * batch.tokens.shape[1])
        if max_num_batched_tokens is None or max_num_batched_tokens <= 0:
            return padded_tokens
        return min(padded_tokens, int(max_num_batched_tokens))
    raise ValueError(
        "NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE must be "
        f"'exact' or 'bucket', got {mode!r}"
    )


def _static_prefill_token_count_for_batch(
    batch: ScheduledBatch,
    *,
    config: Qwen3_5Config | None = None,
    max_num_batched_tokens: int | None = None,
) -> int:
    if batch.is_prefill and _needs_static_prefill_token_count(config):
        return _compact_prefill_token_count(
            batch,
            config=config,
            max_num_batched_tokens=max_num_batched_tokens,
        )
    return 0


@dataclass
class ExecutorOutput:
    activations: object
    cache_storage: Optional[KVCacheStorage]
    attention_metadata: Optional[AttentionMetadata]
    hybrid_state: Optional[HybridLayerState]
    resident_seq_lens: Optional[jnp.ndarray] = None
    resident_last_tokens: Optional[jnp.ndarray] = None
    resident_rng_counters: Optional[jnp.ndarray] = None
    debug_payload: object | None = None


@dataclass
class MTP1GreedyOutput:
    target_token: object
    bonus_token: object
    next_draft_token: object
    accepted: object
    cache_storage: KVCacheStorage
    hybrid_state: HybridLayerState
    committed_seq_lens: object | None = None
    host_payload: object | None = None
    emitted_tokens: object | None = None
    emitted_counts: object | None = None
    accepted_counts: object | None = None
    emitted_totals: object | None = None
    accepted_totals: object | None = None
    rejected_totals: object | None = None
    bonus_totals: object | None = None
    accepted_bitmask: object | None = None
    compact_summary: object | None = None
    burst_groups: int | None = None
    hybrid_state_is_table: bool = False
    debug_payload: object | None = None


@dataclass
class MTP1ParityDebugOutput:
    slot0_logit_max_abs: object
    slot1_logit_max_abs: object
    slot0_hidden_max_abs: object
    slot1_hidden_max_abs: object
    current_k_slot_max_abs: object
    draft_k_slot_max_abs: object
    current_v_slot_max_abs: object
    draft_v_slot_max_abs: object
    conv_state_max_abs: object
    recurrent_state_max_abs: object
    fused_top5_slot0: object
    seq_top5_slot0: object
    fused_top5_slot1: object
    seq_top5_slot1: object
    fused_target_token: object
    seq_target_token: object
    fused_bonus_token: object
    seq_bonus_token: object


@dataclass
class MTP1LayerwiseDriftDebugOutput:
    hidden_max_abs: object
    k_slot_max_abs: object
    v_slot_max_abs: object
    conv_state_max_abs: object
    recurrent_state_max_abs: object
    k_prewrite_max_abs: object
    v_prewrite_max_abs: object
    block_stage_max_abs: object


class ModelExecutor:
    """Single canonical inference path for model execution."""

    def __init__(
        self,
        config: Qwen3_5Config,
        params: ModelParams,
        backend: str | InferenceBackend = "auto",
    ):
        self.config = config
        self.params = params
        params_leaves, self._params_treedef = jax.tree_util.tree_flatten(self.params)
        self._params_leaves = tuple(params_leaves)
        self.backend = select_backend(backend, config=config) if isinstance(backend, str) else backend
        self._jit_cache = {}
        self._jit_metrics: dict[tuple, dict] = {}
        self._configure_persistent_cache()

    def _configure_persistent_cache(self):
        """Enable a deterministic JAX persistent compilation cache."""
        mount_root = Path(os.getenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp"))
        default_cache_dir = (
            mount_root / ".cache" / "jax"
            if mount_root.exists()
            else Path.cwd() / ".cache" / "jax"
        )
        cache_dir = os.getenv(
            "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
            os.getenv("JAX_COMPILATION_CACHE_DIR", str(default_cache_dir)),
        )
        if cache_dir:
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("NANO_VLLM_JAX_COMPILE_CACHE_DIR", cache_dir)
                os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
                jax.config.update("jax_enable_compilation_cache", True)
                jax.config.update("jax_compilation_cache_dir", cache_dir)
            except AttributeError:
                # Older JAX releases do not expose this flag.
                print(
                    "[Executor] jax_compilation_cache_dir is not supported by this "
                    "JAX build; skipping cache-directory configuration."
                )
            except Exception as exc:
                print(
                    "[Executor] failed to configure jax_compilation_cache_dir: "
                    f"{type(exc).__name__}: {exc}"
                )

        try:
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        except Exception:
            # Older JAX versions may not expose this flag in older releases.
            pass

    @staticmethod
    def _batch_shapes(batch: ScheduledBatch) -> tuple:
        return (
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            bool(batch.is_prefill),
            int(batch.num_prefill_tokens),
            int(batch.num_decode_tokens),
            bool(batch.seq_ids.shape[0] == batch.tokens.shape[0]),
            bool(batch.seq_lens.shape[0] == batch.tokens.shape[0]),
            bool(batch.positions.shape == batch.tokens.shape),
        )

    def _log_step(self, mode: str, batch: ScheduledBatch, *, return_hidden: bool, last_logits_only: bool):
        if os.environ.get("NANO_VLLM_JAX_EXEC_LOG_STEPS", "0") not in {"1", "true", "yes", "on", "True"}:
            return
        print(
            f"[Executor] {mode}: tokens={tuple(batch.tokens.shape)} positions={tuple(batch.positions.shape)} "
            f"block_tables={tuple(batch.block_tables.shape)} is_prefill={bool(batch.is_prefill)} "
            f"return_hidden={bool(return_hidden)} last_logits_only={bool(last_logits_only)}"
            f" query_lens={tuple(batch.query_lens.tolist())}"
        )

    def _validate_batch_contract(self, batch: ScheduledBatch):
        if batch.tokens.ndim != 2:
            raise ValueError(f"Scheduled batch tokens must be 2D, got {batch.tokens.ndim}D")
        if batch.positions.ndim != 2:
            raise ValueError(f"Scheduled batch positions must be 2D, got {batch.positions.ndim}D")
        if batch.positions.shape != batch.tokens.shape:
            raise ValueError("Scheduled batch tokens and positions must have matching shape")
        if batch.block_tables.ndim != 2:
            raise ValueError("Scheduled batch block_tables must be 2D")
        metadata_rows = int(batch.block_tables.shape[0]) if batch.packed_prefill else int(batch.tokens.shape[0])
        if batch.packed_prefill:
            if not batch.is_prefill:
                raise ValueError("packed ScheduledBatch is supported only for prefill")
            if batch.token_row_ids is None:
                raise ValueError("packed prefill requires token_row_ids")
            if batch.token_row_ids.shape != batch.tokens.shape:
                raise ValueError("packed prefill token_row_ids must match tokens shape")
            if batch.tokens.shape[0] != 1:
                raise ValueError("packed prefill tokens must have shape [1, token_bucket]")
        if batch.seq_ids.shape[0] != metadata_rows:
            raise ValueError("Scheduled batch seq_ids size must match batch size")
        if batch.seq_lens.shape[0] != metadata_rows:
            raise ValueError("Scheduled batch seq_lens size must match batch size")
        if batch.block_tables.shape[0] != metadata_rows:
            raise ValueError("Scheduled batch block_tables size must match batch size")
        if batch.query_start_loc.shape[0] != metadata_rows + 1:
            raise ValueError("Scheduled batch query_start_loc size must be batch_size + 1")

        if (
            batch.query_lens_host is not None
            and batch.seq_ids_host is not None
            and batch.seq_lens_host is not None
        ):
            self._validate_batch_contract_host(batch)
            return

        query_lens = jnp.diff(batch.query_start_loc).astype(jnp.int32)
        if bool(jnp.any(query_lens < 0)):
            raise ValueError("Scheduled batch query lengths must be non-negative")
        if batch.is_prefill:
            if int(jnp.sum(query_lens)) != int(batch.num_prefill_tokens):
                raise ValueError("Prefill num_prefill_tokens must match query lens sum")
        else:
            if batch.positions.shape[1] < 1:
                raise ValueError("Decode batches must have at least one query token")
            if int(jnp.max(query_lens)) > batch.positions.shape[1]:
                raise ValueError("Decode query_lens must fit within the token width")
            expected_decode = int(jnp.sum(query_lens))
            if int(batch.num_decode_tokens) != expected_decode:
                raise ValueError("Decode num_decode_tokens must match query lens sum")
            active_rows = query_lens > 0
            real_rows = batch.seq_ids >= 0
            if bool(jnp.any(active_rows != real_rows)):
                raise ValueError("Decode rows must use seq_id=-1 exactly when query_len is 0")
            if bool(jnp.any(jnp.where(active_rows, batch.seq_lens < 1, batch.seq_lens != 0))):
                raise ValueError("Decode inactive rows must have seq_len=0 and active rows must have seq_len>=1")

    @staticmethod
    def _validate_batch_contract_host(batch: ScheduledBatch):
        query_lens = tuple(int(x) for x in batch.query_lens_host or ())
        seq_ids = tuple(int(x) for x in batch.seq_ids_host or ())
        seq_lens = tuple(int(x) for x in batch.seq_lens_host or ())
        batch_size = int(batch.batch_size)
        if len(query_lens) != batch_size:
            raise ValueError("Scheduled batch host query_lens size must match batch size")
        if len(seq_ids) != batch_size:
            raise ValueError("Scheduled batch host seq_ids size must match batch size")
        if len(seq_lens) != batch_size:
            raise ValueError("Scheduled batch host seq_lens size must match batch size")
        if any(query_len < 0 for query_len in query_lens):
            raise ValueError("Scheduled batch query lengths must be non-negative")
        if batch.is_prefill:
            if sum(query_lens) != int(batch.num_prefill_tokens):
                raise ValueError("Prefill num_prefill_tokens must match query lens sum")
            return

        if batch.positions.shape[1] < 1:
            raise ValueError("Decode batches must have at least one query token")
        if max(query_lens, default=0) > batch.positions.shape[1]:
            raise ValueError("Decode query_lens must fit within the token width")
        expected_decode = sum(query_lens)
        if int(batch.num_decode_tokens) != expected_decode:
            raise ValueError("Decode num_decode_tokens must match query lens sum")
        for seq_id, query_len, seq_len in zip(seq_ids, query_lens, seq_lens):
            active = query_len > 0
            real = seq_id >= 0
            if active != real:
                raise ValueError("Decode rows must use seq_id=-1 exactly when query_len is 0")
            if (active and seq_len < 1) or ((not active) and seq_len != 0):
                raise ValueError("Decode inactive rows must have seq_len=0 and active rows must have seq_len>=1")

    def _packed_prefill_max_query_len(self, batch: ScheduledBatch) -> int | None:
        if not batch.packed_prefill:
            return None
        prefill_buckets = tuple(getattr(self.config, "prefill_buckets", ()) or ())
        if prefill_buckets:
            return int(max(prefill_buckets))
        if batch.query_lens_host:
            return max(int(length) for length in batch.query_lens_host)
        return int(batch.tokens.shape[1])

    def _jit_metric(self, key: tuple) -> dict:
        return self._jit_metrics.setdefault(
            key,
            {
                "compile_ms": None,
                "first_execution_ms": None,
                "warmed_execution_ms": None,
                "num_calls": 0,
            },
        )

    def _profile_jit_call(
        self,
        cache_key: tuple,
        fn,
        args: tuple,
        call_stage: str,
    ):
        if os.environ.get("NANO_VLLM_JAX_PROFILE_JIT", "0") not in _TRUE_ENV_VALUES:
            return fn(*args)

        metrics = self._jit_metric(cache_key)
        if metrics["compile_ms"] is None:
            compile_start = time.perf_counter()
            try:
                fn.lower(*args).compile()
            finally:
                metrics["compile_ms"] = (time.perf_counter() - compile_start) * 1000
                print(
                    f"[Executor] compiled key={cache_key} "
                    f"in {metrics['compile_ms']:.3f}ms for {call_stage}"
                )

        stage_start = time.perf_counter()
        outputs = fn(*args)
        self._block_until_ready(outputs)
        elapsed_ms = (time.perf_counter() - stage_start) * 1000

        metrics["num_calls"] += 1
        if metrics["first_execution_ms"] is None:
            metrics["first_execution_ms"] = elapsed_ms
            print(f"[Executor] first_execution key={cache_key} stage={call_stage} took {elapsed_ms:.3f}ms")
        else:
            prev = metrics["warmed_execution_ms"]
            metrics["warmed_execution_ms"] = elapsed_ms if prev is None else prev
            print(
                f"[Executor] warmed_execution key={cache_key} stage={call_stage} "
                f"took {elapsed_ms:.3f}ms"
            )

        return outputs

    @staticmethod
    def _block_until_ready(outputs):
        for leaf in jax.tree_util.tree_leaves(outputs):
            ready = getattr(leaf, "block_until_ready", None)
            if ready is not None:
                ready()

    def forward_step(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: Optional[KVCacheStorage] = None,
        hybrid_state: Optional[HybridLayerState] = None,
        return_hidden: bool = False,
        return_hidden_with_logits: bool = False,
        last_logits_only: bool = False,
    ) -> ExecutorOutput:
        self._log_step("forward_step", batch, return_hidden=return_hidden, last_logits_only=last_logits_only)
        self._validate_batch_contract(batch)
        attention_metadata = None
        kv_state = None
        if cache_storage is not None:
            attention_metadata = self.backend.build_attention_metadata(
                positions=batch.positions,
                block_tables=batch.block_tables,
                seq_lens=batch.seq_lens,
                block_size=self.config.block_size,
                is_prefill=batch.is_prefill,
                query_start_loc=batch.query_start_loc,
                num_prefill_tokens=batch.num_prefill_tokens,
                num_decode_tokens=batch.num_decode_tokens,
                token_row_ids=batch.token_row_ids if batch.packed_prefill else None,
                max_query_len=self._packed_prefill_max_query_len(batch),
            )
            kv_state = KVCacheState(
                k_cache=cache_storage.k_cache,
                v_cache=cache_storage.v_cache,
                block_table=batch.block_tables,
                kv_lens=batch.seq_lens,
                slot_mapping=attention_metadata.slot_mapping,
            )

        model_return_hidden = return_hidden or (last_logits_only and batch.packed_prefill)
        model_return_hidden_with_logits = return_hidden_with_logits and not (
            last_logits_only and batch.packed_prefill
        )
        activations, updated_kv_state, updated_hybrid_state = model_forward_step(
            batch.tokens,
            self.params,
            self.config,
            positions=batch.positions,
            kv_cache_state=kv_state,
            attention_metadata=attention_metadata,
            hybrid_state=hybrid_state,
            is_prefill=batch.is_prefill,
            return_hidden=model_return_hidden,
            return_hidden_with_logits=model_return_hidden_with_logits,
            last_logits_only=last_logits_only and not batch.packed_prefill,
            logit_positions=(
                self._logit_positions(batch)
                if last_logits_only and not batch.packed_prefill
                else None
            ),
            backend=self.backend,
        )
        if last_logits_only and batch.packed_prefill:
            hidden = activations
            gather_positions = self._logit_positions(batch)
            gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
            gathered_hidden = hidden[0, gather_idx, :][:, None, :]
            if return_hidden_with_logits or not return_hidden:
                normed = rms_norm(gathered_hidden, self.params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                output_weight = self.params.lm_head if self.params.lm_head is not None else self.params.embed_tokens.T
                logits = jnp.dot(normed, output_weight)
                activations = (gathered_hidden, logits) if return_hidden_with_logits else logits
            else:
                activations = gathered_hidden

        updated_storage = cache_storage
        if updated_kv_state is not None:
            updated_storage = updated_kv_state.storage

        return ExecutorOutput(
            activations=activations,
            cache_storage=updated_storage,
            attention_metadata=attention_metadata,
            hybrid_state=updated_hybrid_state,
        )

    def forward_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        return_hidden: bool = False,
        return_hidden_with_logits: bool = False,
        last_logits_only: bool = False,
    ) -> ExecutorOutput:
        """JIT-compiled variant for fixed-shape benchmark/server steps."""
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_jit requires initialized hybrid_state")
        self._log_step("forward_step_jit", batch, return_hidden=return_hidden, last_logits_only=last_logits_only)
        self._validate_batch_contract(batch)

        key = (
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            bool(batch.is_prefill),
            bool(return_hidden),
            bool(return_hidden_with_logits),
            bool(last_logits_only),
            _static_prefill_token_count_for_batch(
                batch,
                config=self.config,
                max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
            ),
        )
        if key not in self._jit_cache:
            is_prefill = bool(batch.is_prefill)
            static_num_prefill_tokens = (
                _compact_prefill_token_count(
                    batch,
                    config=self.config,
                    max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
                )
                if is_prefill and _needs_static_prefill_token_count(self.config)
                else None
            )
            static_packed_max_query_len = (
                self._packed_prefill_max_query_len(batch) if is_prefill else None
            )

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                token_row_ids,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                num_prefill_tokens = static_num_prefill_tokens if static_num_prefill_tokens is not None else num_query_tokens
                step_positions = positions
                if not is_prefill and positions.shape[1] == 1:
                    step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.zeros((tokens.shape[0],), dtype=jnp.int32),
                    query_start_loc=query_start_loc,
                    is_prefill=is_prefill,
                    num_prefill_tokens=num_prefill_tokens if is_prefill else 0,
                    num_decode_tokens=0 if is_prefill else num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    packed_prefill=is_prefill and token_row_ids is not None,
                    token_row_ids=token_row_ids,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=step_batch.is_prefill,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=step_batch.num_prefill_tokens,
                    num_decode_tokens=step_batch.num_decode_tokens,
                    token_row_ids=step_batch.token_row_ids if step_batch.packed_prefill else None,
                    max_query_len=static_packed_max_query_len,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                model_return_hidden = return_hidden or (last_logits_only and step_batch.packed_prefill)
                model_return_hidden_with_logits = (
                    return_hidden_with_logits and not (last_logits_only and step_batch.packed_prefill)
                )
                activations, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=step_batch.is_prefill,
                    return_hidden=model_return_hidden,
                    return_hidden_with_logits=model_return_hidden_with_logits,
                    last_logits_only=last_logits_only and not step_batch.packed_prefill,
                    logit_positions=(
                        self._logit_positions(step_batch)
                        if last_logits_only and not step_batch.packed_prefill
                        else None
                    ),
                    backend=self.backend,
                )
                if last_logits_only and step_batch.packed_prefill:
                    hidden = activations
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gathered_hidden = hidden[0, gather_idx, :][:, None, :]
                    if return_hidden_with_logits or not return_hidden:
                        normed = rms_norm(gathered_hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                        output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
                        logits = jnp.dot(normed, output_weight)
                        activations = (gathered_hidden, logits) if return_hidden_with_logits else logits
                    else:
                        activations = gathered_hidden
                return (
                    activations,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(7, 8),
            )

        activations, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.token_row_ids,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
            ),
            f"forward_step_jit:{'prefill' if batch.is_prefill else 'decode'}",
        )
        return ExecutorOutput(
            activations=activations,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def forward_step_token_ids_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
    ) -> ExecutorOutput:
        """JIT path that returns greedy token ids instead of full logits.

        This is a serving-oriented specialization for temperature-0 generation.
        The dense LM head still runs inside the compiled graph, but only the
        small argmax result crosses the Python/JAX boundary. Correctness and
        diagnostics that need logits should continue using ``forward_step_jit``.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_token_ids_jit requires initialized hybrid_state")
        self._log_step("forward_step_token_ids_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "token-ids",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            bool(batch.is_prefill),
            _static_prefill_token_count_for_batch(
                batch,
                config=self.config,
                max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
            ),
        )
        if key not in self._jit_cache:
            is_prefill = bool(batch.is_prefill)
            static_num_prefill_tokens = (
                _compact_prefill_token_count(
                    batch,
                    config=self.config,
                    max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
                )
                if is_prefill and _needs_static_prefill_token_count(self.config)
                else None
            )
            static_packed_max_query_len = (
                self._packed_prefill_max_query_len(batch) if is_prefill else None
            )

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                token_row_ids,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                num_prefill_tokens = static_num_prefill_tokens if static_num_prefill_tokens is not None else num_query_tokens
                step_positions = positions
                if not is_prefill and positions.shape[1] == 1:
                    step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.zeros((tokens.shape[0],), dtype=jnp.int32),
                    query_start_loc=query_start_loc,
                    is_prefill=is_prefill,
                    num_prefill_tokens=num_prefill_tokens if is_prefill else 0,
                    num_decode_tokens=0 if is_prefill else num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    packed_prefill=is_prefill and token_row_ids is not None,
                    token_row_ids=token_row_ids,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=step_batch.is_prefill,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=step_batch.num_prefill_tokens,
                    num_decode_tokens=step_batch.num_decode_tokens,
                    token_row_ids=step_batch.token_row_ids if step_batch.packed_prefill else None,
                    max_query_len=static_packed_max_query_len,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=step_batch.is_prefill,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                if step_batch.packed_prefill:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gathered = hidden[0, gather_idx, :]
                    last_hidden = gathered[:, None, :]
                elif is_prefill:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gather_idx = gather_idx[:, None, None]
                    gather_idx = jnp.broadcast_to(gather_idx, (hidden.shape[0], 1, hidden.shape[-1]))
                    last_hidden = jnp.take_along_axis(hidden, gather_idx, axis=1)
                else:
                    last_hidden = hidden[:, :1, :]
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    last_hidden,
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=step_batch.is_prefill,
                    top_k=0,
                )
                emitted_token_ids = (
                    token_ids[:, 0].astype(jnp.int32)
                    if is_prefill
                    else token_ids[:, 0].astype(jnp.int32)
                )
                return (
                    emitted_token_ids,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(7, 8),
            )

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.token_row_ids,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
            ),
            f"forward_step_token_ids_jit:{'prefill' if batch.is_prefill else 'decode'}",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def forward_prefill_token_ids_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        return_last_hidden: bool = False,
        return_mtp_draft: bool = False,
    ) -> ExecutorOutput:
        """Prefill greedy-token path that owns hybrid table gather/scatter in JIT."""
        if not batch.is_prefill:
            raise ValueError("forward_prefill_token_ids_table_jit is prefill-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_prefill_token_ids_table_jit requires initialized hybrid state tables")
        self._log_step("forward_prefill_token_ids_table_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "token-ids-prefill-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            bool(return_last_hidden),
            bool(return_mtp_draft),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            _static_prefill_token_count_for_batch(
                batch,
                config=self.config,
                max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
            ),
        )
        if key not in self._jit_cache:
            static_num_prefill_tokens = (
                _compact_prefill_token_count(
                    batch,
                    config=self.config,
                    max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
                )
                if _needs_static_prefill_token_count(self.config)
                else None
            )
            static_packed_max_query_len = self._packed_prefill_max_query_len(batch)

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                token_row_ids,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                num_prefill_tokens = (
                    static_num_prefill_tokens
                    if static_num_prefill_tokens is not None
                    else num_query_tokens
                )
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=True,
                    num_prefill_tokens=num_prefill_tokens,
                    num_decode_tokens=0,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    packed_prefill=token_row_ids is not None,
                    token_row_ids=token_row_ids,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=True,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=step_batch.num_prefill_tokens,
                    num_decode_tokens=0,
                    token_row_ids=step_batch.token_row_ids if step_batch.packed_prefill else None,
                    max_query_len=static_packed_max_query_len,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=True,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                if step_batch.packed_prefill:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gathered = hidden[0, gather_idx, :]
                    last_hidden = gathered[:, None, :]
                else:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gather_idx = gather_idx[:, None, None]
                    gather_idx = jnp.broadcast_to(gather_idx, (hidden.shape[0], 1, hidden.shape[-1]))
                    last_hidden = jnp.take_along_axis(hidden, gather_idx, axis=1)
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    last_hidden,
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=True,
                    top_k=0,
                )
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                emitted_token_ids = token_ids[:, 0].astype(jnp.int32)
                if return_mtp_draft:
                    mtp_hidden = (
                        rms_norm(last_hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                        if str(getattr(self.config, "mtp_hidden_source", "pre_norm") or "pre_norm").lower()
                        == "final_normed"
                        else last_hidden
                    )
                    mtp_positions = (gather_positions + 1).astype(jnp.int32)
                    draft_len = max(1, int(getattr(self.config, "num_speculative_tokens", 1) or 1))
                    current_hidden = mtp_hidden
                    current_token = emitted_token_ids[:, None]
                    current_position = mtp_positions[:, None]
                    mtp_drafts = []
                    for _ in range(draft_len):
                        current_token, current_hidden = mtp_forward_token_ids(
                            hidden_state=current_hidden,
                            next_token_ids=current_token,
                            embed_tokens=params.embed_tokens,
                            params=params.mtp_params,
                            config=self.config,
                            positions=current_position,
                        )
                        mtp_drafts.append(current_token[:, 0].astype(jnp.int32))
                        current_position = current_position + 1
                    mtp_draft_tokens = jnp.stack(mtp_drafts, axis=1)
                    activations = (
                        emitted_token_ids,
                        mtp_draft_tokens[:, 0] if draft_len == 1 else mtp_draft_tokens,
                    )
                elif return_last_hidden:
                    activations = (emitted_token_ids, last_hidden[:, 0, :])
                else:
                    activations = emitted_token_ids
                return (
                    activations,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(7, 8, 9, 10),
            )

        activations, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.token_row_ids,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
            ),
            "forward_prefill_token_ids_table_jit:prefill",
        )
        return ExecutorOutput(
            activations=activations,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def forward_step_token_ids_mtp_draft_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        mtp_hidden_final_normed: bool,
    ) -> ExecutorOutput:
        """Decode one target token and one unverified MTP draft in one JIT."""
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_mtp_draft_jit is decode-only")
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_token_ids_mtp_draft_jit requires initialized hybrid_state")
        self._log_step(
            "forward_step_token_ids_mtp_draft_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        key = (
            "token-ids-mtp-draft",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state.conv_state.shape),
            tuple(hybrid_state.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
            ):
                del positions
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=query_start_loc[-1].astype(jnp.int32),
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                target_token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                target_token_ids = target_token_ids[:, 0].astype(jnp.int32)
                mtp_hidden = (
                    rms_norm(hidden[:, :1, :], params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    if mtp_hidden_final_normed
                    else hidden[:, :1, :]
                )
                mtp_positions = (
                    seq_lens.astype(jnp.int32)
                    + jnp.asarray(int(getattr(self.config, "mtp_position_offset", 0)), dtype=jnp.int32)
                )[:, None]
                mtp_token_ids, _ = mtp_forward_token_ids(
                    hidden_state=mtp_hidden,
                    next_token_ids=target_token_ids[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=mtp_positions,
                )
                draft_token_ids = mtp_token_ids[:, 0].astype(jnp.int32)
                active = (query_lens > 0) & (seq_ids >= 0)
                token_rows = jnp.stack([target_token_ids, draft_token_ids], axis=1)
                token_rows = jnp.where(active[:, None], token_rows, jnp.zeros_like(token_rows))
                return (
                    token_rows,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(7, 8))

        token_rows, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
            ),
            "forward_step_token_ids_mtp_draft_jit:decode",
        )
        return ExecutorOutput(
            activations=token_rows,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def forward_step_token_ids_mtp_draft_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> ExecutorOutput:
        """Decode one target token and one unverified MTP draft with table-owned hybrid state."""
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_mtp_draft_table_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError(
                "forward_step_token_ids_mtp_draft_table_jit requires initialized hybrid state tables"
            )
        self._log_step(
            "forward_step_token_ids_mtp_draft_table_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        key = (
            "token-ids-mtp-draft-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_last_tokens.shape),
            bool(mtp_hidden_final_normed),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                last_tokens_table,
            ):
                del positions
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)

                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                target_token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                target_token_ids = target_token_ids[:, 0].astype(jnp.int32)
                mtp_hidden = (
                    rms_norm(hidden[:, :1, :], params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    if mtp_hidden_final_normed
                    else hidden[:, :1, :]
                )
                mtp_positions = (
                    seq_lens.astype(jnp.int32)
                    + jnp.asarray(int(getattr(self.config, "mtp_position_offset", 0)), dtype=jnp.int32)
                )[:, None]
                mtp_token_ids, _ = mtp_forward_token_ids(
                    hidden_state=mtp_hidden,
                    next_token_ids=target_token_ids[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=mtp_positions,
                )
                draft_token_ids = mtp_token_ids[:, 0].astype(jnp.int32)
                active = (query_lens > 0) & (slot_ids >= 0)
                token_rows = jnp.stack([target_token_ids, draft_token_ids], axis=1)
                token_rows = jnp.where(active[:, None], token_rows, jnp.zeros_like(token_rows))

                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                updated_last_tokens = last_tokens_table.at[scatter_slot_ids].set(
                    draft_token_ids,
                    mode="drop",
                )
                return (
                    token_rows,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_last_tokens,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(6, 7, 8, 9, 11),
            )

        token_rows, k_cache, v_cache, conv_state, recurrent_state, last_tokens_table = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_last_tokens,
            ),
            "forward_step_token_ids_mtp_draft_table_jit:decode",
        )
        return ExecutorOutput(
            activations=token_rows,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_last_tokens=last_tokens_table,
        )

    def forward_step_token_ids_mtp_draft_resident_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> ExecutorOutput:
        """MTP draft decode that gathers input tokens from resident slot state."""
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_mtp_draft_resident_table_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError(
                "forward_step_token_ids_mtp_draft_resident_table_jit requires initialized hybrid state tables"
            )
        self._log_step(
            "forward_step_token_ids_mtp_draft_resident_table_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        key = (
            "token-ids-mtp-draft-resident-table",
            tuple(batch.tokens.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_last_tokens.shape),
            max(1, int(getattr(self.config, "num_speculative_tokens", 1) or 1)),
            bool(mtp_hidden_final_normed),
        )
        if key not in self._jit_cache:
            draft_len = max(1, int(getattr(self.config, "num_speculative_tokens", 1) or 1))

            def compiled(
                params_leaves,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                last_tokens_table,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)

                tokens = last_tokens_table[safe_slot_ids].astype(jnp.int32)[:, None]
                tokens = jnp.where(row_valid[:, None], tokens, jnp.zeros_like(tokens))
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                target_token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                target_token_ids = target_token_ids[:, 0].astype(jnp.int32)
                mtp_hidden = (
                    rms_norm(hidden[:, :1, :], params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    if mtp_hidden_final_normed
                    else hidden[:, :1, :]
                )
                mtp_positions = (
                    seq_lens.astype(jnp.int32)
                    + jnp.asarray(int(getattr(self.config, "mtp_position_offset", 0)), dtype=jnp.int32)
                )[:, None]
                current_hidden = mtp_hidden
                current_token_ids = target_token_ids[:, None]
                draft_columns = []
                for _ in range(draft_len):
                    current_token_ids, current_hidden = mtp_forward_token_ids(
                        hidden_state=current_hidden,
                        next_token_ids=current_token_ids,
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=mtp_positions,
                    )
                    draft_columns.append(current_token_ids[:, 0].astype(jnp.int32))
                    mtp_positions = mtp_positions + jnp.asarray(1, dtype=jnp.int32)
                draft_token_ids = draft_columns[-1]
                token_rows = jnp.stack([target_token_ids, *draft_columns], axis=1)
                token_rows = jnp.where(row_valid[:, None], token_rows, jnp.zeros_like(token_rows))

                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                updated_last_tokens = last_tokens_table.at[scatter_slot_ids].set(
                    draft_token_ids,
                    mode="drop",
                )
                return (
                    token_rows,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_last_tokens,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(4, 5, 6, 7, 9),
            )

        token_rows, k_cache, v_cache, conv_state, recurrent_state, last_tokens_table = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_last_tokens,
            ),
            "forward_step_token_ids_mtp_draft_resident_table_jit:decode",
        )
        return ExecutorOutput(
            activations=token_rows,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_last_tokens=last_tokens_table,
        )

    def forward_step_token_ids_mtp_draft_chain_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        mtp_hidden_final_normed: bool,
        draft_len: int,
    ) -> ExecutorOutput:
        """Decode one target token and seed a greedy MTP draft chain in one JIT."""
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_mtp_draft_chain_jit is decode-only")
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_token_ids_mtp_draft_chain_jit requires initialized hybrid_state")
        draft_len = int(draft_len)
        if draft_len < 1:
            raise ValueError("draft_len must be >= 1")
        self._log_step(
            "forward_step_token_ids_mtp_draft_chain_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)
        chain_logit_debug = os.environ.get(
            "NANO_VLLM_JAX_MTP_CHAIN_LOGIT_DEBUG",
            "0",
        ) in {"1", "true", "yes", "on", "True"}

        key = (
            "token-ids-mtp-draft-chain",
            "target-carry-v2",
            draft_len,
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state.conv_state.shape),
            tuple(hybrid_state.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            bool(chain_logit_debug),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
            ):
                del positions
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=query_start_loc[-1].astype(jnp.int32),
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                target_token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                target_token_ids = target_token_ids[:, 0].astype(jnp.int32)
                current_hidden = (
                    rms_norm(hidden[:, :1, :], params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    if mtp_hidden_final_normed
                    else hidden[:, :1, :]
                )
                current_token = target_token_ids[:, None]
                current_position = (
                    seq_lens.astype(jnp.int32)
                    + jnp.asarray(int(getattr(self.config, "mtp_position_offset", 0)), dtype=jnp.int32)
                )[:, None]
                draft_tokens = []
                draft_top_ids = []
                draft_top_values = []
                for _ in range(draft_len):
                    if chain_logit_debug:
                        draft_logits, current_hidden = mtp_forward(
                            hidden_state=current_hidden,
                            next_token_ids=current_token,
                            embed_tokens=params.embed_tokens,
                            params=params.mtp_params,
                            config=self.config,
                            positions=current_position,
                        )
                        values, ids = jax.lax.top_k(draft_logits[:, 0].astype(jnp.float32), 5)
                        mtp_token_ids = jnp.argmax(draft_logits[:, 0], axis=-1).astype(jnp.int32)[:, None]
                        draft_top_ids.append(ids.astype(jnp.int32))
                        draft_top_values.append(values.astype(jnp.float32))
                    else:
                        mtp_token_ids, current_hidden = mtp_forward_token_ids(
                            hidden_state=current_hidden,
                            next_token_ids=current_token,
                            embed_tokens=params.embed_tokens,
                            params=params.mtp_params,
                            config=self.config,
                            positions=current_position,
                        )
                    current_token = mtp_token_ids[:, 0].astype(jnp.int32)[:, None]
                    draft_tokens.append(current_token[:, 0])
                    current_position = current_position + 1
                token_rows = jnp.concatenate(
                    [target_token_ids[:, None], jnp.stack(draft_tokens, axis=1)],
                    axis=1,
                ).astype(jnp.int32)
                active = (query_lens > 0) & (seq_ids >= 0)
                token_rows = jnp.where(active[:, None], token_rows, jnp.zeros_like(token_rows))
                if chain_logit_debug:
                    debug_payload = (
                        jnp.stack(draft_top_ids, axis=1),
                        jnp.stack(draft_top_values, axis=1),
                    )
                else:
                    debug_payload = None
                return (
                    token_rows,
                    target_token_ids[:, None],
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                    debug_payload,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(7, 8))

        token_rows, target_token_carry, k_cache, v_cache, conv_state, recurrent_state, debug_payload = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
            ),
            "forward_step_token_ids_mtp_draft_chain_jit:decode",
        )
        return ExecutorOutput(
            activations=token_rows,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_last_tokens=target_token_carry,
            debug_payload=debug_payload,
        )

    def forward_step_sampled_token_ids_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        temperatures: jnp.ndarray,
        rng_counters: jnp.ndarray,
        rng_slots: jnp.ndarray,
    ) -> ExecutorOutput:
        """JIT path that samples token ids without returning full logits."""
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_sampled_token_ids_jit requires initialized hybrid_state")
        self._log_step("forward_step_sampled_token_ids_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "sampled-token-ids",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            bool(batch.is_prefill),
            tuple(temperatures.shape),
            tuple(rng_counters.shape),
            tuple(rng_slots.shape),
            _static_prefill_token_count_for_batch(
                batch,
                config=self.config,
                max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
            ),
        )
        if key not in self._jit_cache:
            is_prefill = bool(batch.is_prefill)
            static_num_prefill_tokens = (
                _compact_prefill_token_count(
                    batch,
                    config=self.config,
                    max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
                )
                if is_prefill and _needs_static_prefill_token_count(self.config)
                else None
            )
            static_packed_max_query_len = (
                self._packed_prefill_max_query_len(batch) if is_prefill else None
            )
            base_key = jax.random.PRNGKey(0)

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                token_row_ids,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                temp_values,
                counter_values,
                slot_values,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                num_prefill_tokens = static_num_prefill_tokens if static_num_prefill_tokens is not None else num_query_tokens
                step_positions = positions
                if not is_prefill and positions.shape[1] == 1:
                    step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.zeros((tokens.shape[0],), dtype=jnp.int32),
                    query_start_loc=query_start_loc,
                    is_prefill=is_prefill,
                    num_prefill_tokens=num_prefill_tokens if is_prefill else 0,
                    num_decode_tokens=0 if is_prefill else num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    packed_prefill=is_prefill and token_row_ids is not None,
                    token_row_ids=token_row_ids,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=step_batch.is_prefill,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=step_batch.num_prefill_tokens,
                    num_decode_tokens=step_batch.num_decode_tokens,
                    token_row_ids=step_batch.token_row_ids if step_batch.packed_prefill else None,
                    max_query_len=static_packed_max_query_len,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=step_batch.is_prefill,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                if step_batch.packed_prefill:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gathered = hidden[0, gather_idx, :]
                    last_hidden = gathered[:, None, :]
                elif is_prefill:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gather_idx = gather_idx[:, None, None]
                    gather_idx = jnp.broadcast_to(gather_idx, (hidden.shape[0], 1, hidden.shape[-1]))
                    last_hidden = jnp.take_along_axis(hidden, gather_idx, axis=1)
                else:
                    last_hidden = hidden[:, :1, :]
                rng_keys = jax.vmap(
                    lambda slot, counter: jax.random.fold_in(
                        jax.random.fold_in(base_key, slot.astype(jnp.uint32)),
                        counter.astype(jnp.uint32),
                    )
                )(slot_values.astype(jnp.uint32), counter_values.astype(jnp.uint32))
                token_ids = lm_head_sample_token_ids(
                    last_hidden,
                    params,
                    self.config,
                    temperatures=temp_values,
                    rng_keys=rng_keys,
                    hidden_is_normed=False,
                    is_prefill=step_batch.is_prefill,
                )
                return (
                    token_ids.astype(jnp.int32),
                    counter_values.astype(jnp.int32) + jnp.asarray(1, dtype=jnp.int32),
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(7, 8),
            )

        token_ids, updated_rng_counters, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.token_row_ids,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                temperatures,
                rng_counters,
                rng_slots,
            ),
            f"forward_step_sampled_token_ids_jit:{'prefill' if batch.is_prefill else 'decode'}",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_rng_counters=updated_rng_counters,
        )

    def forward_step_token_ids_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
    ) -> ExecutorOutput:
        """Decode greedy-token path that owns hybrid table gather/scatter in JIT.

        This is the ABI bridge toward vLLM-style state-indexed GDN kernels. The
        public JAX contract is still functional: the compiled function returns
        updated state tables. Internally the decode step gathers active rows by
        ``hybrid_slot_ids`` before the model call and scatters the updated rows
        back into the table before returning.
        """
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_table_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_step_token_ids_table_jit requires initialized hybrid state tables")
        self._log_step("forward_step_token_ids_table_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "token-ids-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
        )
        if key not in self._jit_cache:
            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                step_positions = positions
                if positions.shape[1] == 1:
                    step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                        is_prefill=False,
                        num_prefill_tokens=0,
                        num_decode_tokens=num_query_tokens,
                        block_tables=block_tables,
                        seq_lens=seq_lens,
                    )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                return (
                    token_ids[:, 0].astype(jnp.int32),
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(6, 7, 8, 9),
            )

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
            ),
            "forward_step_token_ids_table_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def forward_prefill_token_ids_slot_carry_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        prefill_final_flags: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
    ) -> ExecutorOutput:
        """Prefill greedy-token path that also seeds resident slot tokens in JIT."""
        if not batch.is_prefill:
            raise ValueError("forward_prefill_token_ids_slot_carry_table_jit is prefill-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_prefill_token_ids_slot_carry_table_jit requires initialized hybrid state tables")
        self._log_step(
            "forward_prefill_token_ids_slot_carry_table_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        key = (
            "token-ids-prefill-slot-carry-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(prefill_final_flags.shape),
            tuple(resident_last_tokens.shape),
            _static_prefill_token_count_for_batch(
                batch,
                config=self.config,
                max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
            ),
        )
        if key not in self._jit_cache:
            static_num_prefill_tokens = (
                _compact_prefill_token_count(
                    batch,
                    config=self.config,
                    max_num_batched_tokens=getattr(self.config, "max_num_batched_tokens", None),
                )
                if _needs_static_prefill_token_count(self.config)
                else None
            )
            static_packed_max_query_len = self._packed_prefill_max_query_len(batch)

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                token_row_ids,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                final_flags,
                last_tokens_table,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                num_prefill_tokens = (
                    static_num_prefill_tokens
                    if static_num_prefill_tokens is not None
                    else num_query_tokens
                )
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=True,
                    num_prefill_tokens=num_prefill_tokens,
                    num_decode_tokens=0,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    packed_prefill=token_row_ids is not None,
                    token_row_ids=token_row_ids,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=True,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=step_batch.num_prefill_tokens,
                    num_decode_tokens=0,
                    token_row_ids=step_batch.token_row_ids if step_batch.packed_prefill else None,
                    max_query_len=static_packed_max_query_len,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=True,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                if step_batch.packed_prefill:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gathered = hidden[0, gather_idx, :]
                    last_hidden = gathered[:, None, :]
                else:
                    gather_positions = self._logit_positions(step_batch)
                    gather_idx = jnp.clip(gather_positions, 0, hidden.shape[1] - 1).astype(jnp.int32)
                    gather_idx = gather_idx[:, None, None]
                    gather_idx = jnp.broadcast_to(gather_idx, (hidden.shape[0], 1, hidden.shape[-1]))
                    last_hidden = jnp.take_along_axis(hidden, gather_idx, axis=1)
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    last_hidden,
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=True,
                    top_k=0,
                )
                token_ids = token_ids[:, 0].astype(jnp.int32)
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                final_flags = final_flags.astype(bool)
                token_scatter_slot_ids = jnp.where(
                    row_valid & final_flags,
                    slot_ids,
                    jnp.full_like(slot_ids, last_tokens_table.shape[0]),
                )
                updated_last_tokens = last_tokens_table.at[token_scatter_slot_ids].set(
                    token_ids,
                    mode="drop",
                )
                return (
                    token_ids,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_last_tokens,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(7, 8, 9, 10, 13),
            )

        (
            token_ids,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            last_tokens_table,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.token_row_ids,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                prefill_final_flags,
                resident_last_tokens,
            ),
            "forward_prefill_token_ids_slot_carry_table_jit:prefill",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_last_tokens=last_tokens_table,
        )

    def forward_step_token_ids_slot_carry_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
    ) -> ExecutorOutput:
        """Decode greedy-token path that gathers input tokens by resident slot.

        This keeps the generated-token history immutable in the runner while
        removing the per-step Python/JAX token-carry rewrite from the decode
        hot path. The compiled boundary gathers the previous token from
        ``resident_last_tokens[hybrid_slot_ids]`` and scatters the newly sampled
        token back to the same per-slot table before returning.
        """
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_slot_carry_table_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_step_token_ids_slot_carry_table_jit requires initialized hybrid state tables")
        self._log_step(
            "forward_step_token_ids_slot_carry_table_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        key = (
            "token-ids-slot-carry-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_last_tokens.shape),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                last_tokens_table,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                tokens = last_tokens_table[safe_slot_ids].astype(jnp.int32)[:, None]
                tokens = jnp.where(row_valid[:, None], tokens, jnp.zeros_like(tokens))

                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                step_positions = positions
                if positions.shape[1] == 1:
                    step_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=step_positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                token_ids = token_ids[:, 0].astype(jnp.int32)
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                updated_last_tokens = last_tokens_table.at[scatter_slot_ids].set(
                    token_ids,
                    mode="drop",
                )
                return (
                    token_ids,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_last_tokens,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(5, 6, 7, 8, 10),
            )

        (
            token_ids,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            last_tokens_table,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_last_tokens,
            ),
            "forward_step_token_ids_slot_carry_table_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_last_tokens=last_tokens_table,
        )

    def forward_step_token_ids_resident_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_block_tables: jnp.ndarray,
        resident_seq_lens: jnp.ndarray,
    ) -> ExecutorOutput:
        """Decode greedy-token path from resident slot metadata.

        The scheduler still chooses compact rows and reserves paged KV blocks.
        The runner keeps the current per-slot block-table and sequence-length
        tables on device. This compiled boundary gathers those tables by
        ``hybrid_slot_ids`` so the decode call no longer needs full per-step
        block tables or sequence lengths as dynamic inputs.
        """
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_resident_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_step_token_ids_resident_jit requires initialized hybrid state tables")
        self._log_step("forward_step_token_ids_resident_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        static_block_table_width = int(batch.block_tables.shape[1])
        key = (
            "token-ids-resident",
            tuple(batch.tokens.shape),
            static_block_table_width,
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_block_tables.shape),
            tuple(resident_seq_lens.shape),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                tokens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                block_table_table,
                seq_lens_table,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                row_valid = slot_ids >= 0
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                query_lens = row_valid.astype(jnp.int32)
                query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(query_lens),
                    ],
                    axis=0,
                )
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)

                block_tables = block_table_table[safe_slot_ids, :static_block_table_width]
                block_tables = jnp.where(
                    row_valid[:, None],
                    block_tables,
                    jnp.zeros_like(block_tables),
                )
                seq_lens = seq_lens_table[safe_slot_ids]
                seq_lens = jnp.where(row_valid, seq_lens, jnp.zeros_like(seq_lens))
                positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]

                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                updated_seq_lens_table = seq_lens_table.at[scatter_slot_ids].set(
                    seq_lens + row_valid.astype(seq_lens.dtype),
                    mode="drop",
                )
                return (
                    token_ids[:, 0].astype(jnp.int32),
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_seq_lens_table,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(2, 3, 4, 5),
            )

        token_ids, k_cache, v_cache, conv_state, recurrent_state, seq_lens_table = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
            ),
            "forward_step_token_ids_resident_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_seq_lens=seq_lens_table,
        )

    def forward_step_token_ids_resident_slot_carry_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_block_tables: jnp.ndarray,
        resident_seq_lens: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
    ) -> ExecutorOutput:
        """Decode greedy-token path from resident slot tokens and metadata."""
        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_resident_slot_carry_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_step_token_ids_resident_slot_carry_jit requires initialized hybrid state tables")
        self._log_step(
            "forward_step_token_ids_resident_slot_carry_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        static_block_table_width = int(batch.block_tables.shape[1])
        key = (
            "token-ids-resident-slot-carry",
            tuple(batch.tokens.shape),
            static_block_table_width,
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_block_tables.shape),
            tuple(resident_seq_lens.shape),
            tuple(resident_last_tokens.shape),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                block_table_table,
                seq_lens_table,
                last_tokens_table,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                row_valid = slot_ids >= 0
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                query_lens = row_valid.astype(jnp.int32)
                query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(query_lens),
                    ],
                    axis=0,
                )
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)

                tokens = last_tokens_table[safe_slot_ids].astype(jnp.int32)[:, None]
                tokens = jnp.where(row_valid[:, None], tokens, jnp.zeros_like(tokens))
                block_tables = block_table_table[safe_slot_ids, :static_block_table_width]
                block_tables = jnp.where(
                    row_valid[:, None],
                    block_tables,
                    jnp.zeros_like(block_tables),
                )
                seq_lens = seq_lens_table[safe_slot_ids]
                seq_lens = jnp.where(row_valid, seq_lens, jnp.zeros_like(seq_lens))
                positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]

                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=jnp.where(
                        row_valid,
                        safe_slot_ids.astype(jnp.int32),
                        jnp.full_like(safe_slot_ids, -1),
                    ),
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_query_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                token_ids = token_ids[:, 0].astype(jnp.int32)
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    updated_conv,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    updated_recurrent,
                    mode="drop",
                )
                updated_seq_lens_table = seq_lens_table.at[scatter_slot_ids].set(
                    seq_lens + row_valid.astype(seq_lens.dtype),
                    mode="drop",
                )
                updated_last_tokens = last_tokens_table.at[scatter_slot_ids].set(
                    token_ids,
                    mode="drop",
                )
                return (
                    token_ids,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_seq_lens_table,
                    updated_last_tokens,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(1, 2, 3, 4, 7, 8),
            )

        (
            token_ids,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            seq_lens_table,
            last_tokens_table,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
                resident_last_tokens,
            ),
            "forward_step_token_ids_resident_slot_carry_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_seq_lens=seq_lens_table,
            resident_last_tokens=last_tokens_table,
        )

    def forward_step_token_ids_resident_dense_slot_carry_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_block_tables: jnp.ndarray,
        resident_seq_lens: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
    ) -> ExecutorOutput:
        """Dense decode path for compact active rows with arbitrary slots.

        Random serving emits compact decode batches: every row in the shaped
        decode batch is active, but the resident hybrid-state slot can still be
        an arbitrary scheduler slot. This boundary keeps the resident metadata
        table contract while avoiding the invalid-row masking and cumsum needed
        by the more general resident slot-carry entry point.
        """

        if batch.is_prefill:
            raise ValueError("forward_step_token_ids_resident_dense_slot_carry_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError(
                "forward_step_token_ids_resident_dense_slot_carry_jit requires initialized "
                "hybrid state tables"
            )
        self._log_step(
            "forward_step_token_ids_resident_dense_slot_carry_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        batch_size = int(batch.tokens.shape[0])
        static_block_table_width = int(batch.block_tables.shape[1])
        key = (
            "token-ids-resident-dense-slot-carry",
            tuple(batch.tokens.shape),
            static_block_table_width,
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_block_tables.shape),
            tuple(resident_seq_lens.shape),
            tuple(resident_last_tokens.shape),
        )
        if key not in self._jit_cache:

            def compiled(
                params_leaves,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                block_table_table,
                seq_lens_table,
                last_tokens_table,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                slot_ids = slot_ids.astype(jnp.int32)
                seq_lens = seq_lens_table[slot_ids]
                tokens = last_tokens_table[slot_ids].astype(jnp.int32)[:, None]
                block_tables = block_table_table[slot_ids, :static_block_table_width]
                positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                query_start_loc = jnp.arange(batch_size + 1, dtype=jnp.int32)

                conv_state = conv_state_table[slot_ids]
                recurrent_state = recurrent_state_table[slot_ids]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=slot_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.asarray(batch_size, dtype=jnp.int32),
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                token_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                token_ids = token_ids[:, 0].astype(jnp.int32)
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[slot_ids].set(updated_conv)
                updated_recurrent_table = recurrent_state_table.at[slot_ids].set(
                    updated_recurrent
                )
                updated_seq_lens_table = seq_lens_table.at[slot_ids].set(seq_lens + 1)
                updated_last_tokens = last_tokens_table.at[slot_ids].set(token_ids)
                return (
                    token_ids,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_seq_lens_table,
                    updated_last_tokens,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(1, 2, 3, 4, 7, 8),
            )

        (
            token_ids,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            seq_lens_table,
            last_tokens_table,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
                resident_last_tokens,
            ),
            "forward_step_token_ids_resident_dense_slot_carry_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_seq_lens=seq_lens_table,
            resident_last_tokens=last_tokens_table,
        )

    def forward_step_sampled_token_ids_resident_dense_slot_carry_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        resident_block_tables: jnp.ndarray,
        resident_seq_lens: jnp.ndarray,
        resident_last_tokens: jnp.ndarray,
        resident_rng_counters: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> ExecutorOutput:
        """Dense resident decode path that samples token ids in the JIT."""

        if batch.is_prefill:
            raise ValueError("forward_step_sampled_token_ids_resident_dense_slot_carry_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError(
                "forward_step_sampled_token_ids_resident_dense_slot_carry_jit requires initialized "
                "hybrid state tables"
            )
        self._log_step(
            "forward_step_sampled_token_ids_resident_dense_slot_carry_jit",
            batch,
            return_hidden=True,
            last_logits_only=False,
        )
        self._validate_batch_contract(batch)

        batch_size = int(batch.tokens.shape[0])
        static_block_table_width = int(batch.block_tables.shape[1])
        key = (
            "sampled-token-ids-resident-dense-slot-carry",
            tuple(batch.tokens.shape),
            static_block_table_width,
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            tuple(resident_block_tables.shape),
            tuple(resident_seq_lens.shape),
            tuple(resident_last_tokens.shape),
            tuple(resident_rng_counters.shape),
            tuple(temperatures.shape),
        )
        if key not in self._jit_cache:
            base_key = jax.random.PRNGKey(0)

            def compiled(
                params_leaves,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                block_table_table,
                seq_lens_table,
                last_tokens_table,
                rng_counter_table,
                temp_values,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                slot_ids = slot_ids.astype(jnp.int32)
                seq_lens = seq_lens_table[slot_ids]
                tokens = last_tokens_table[slot_ids].astype(jnp.int32)[:, None]
                block_tables = block_table_table[slot_ids, :static_block_table_width]
                positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                query_start_loc = jnp.arange(batch_size + 1, dtype=jnp.int32)

                conv_state = conv_state_table[slot_ids]
                recurrent_state = recurrent_state_table[slot_ids]
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=slot_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.asarray(batch_size, dtype=jnp.int32),
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )
                counters = rng_counter_table[slot_ids].astype(jnp.int32)
                rng_keys = jax.vmap(
                    lambda slot, counter: jax.random.fold_in(
                        jax.random.fold_in(base_key, slot.astype(jnp.uint32)),
                        counter.astype(jnp.uint32),
                    )
                )(slot_ids.astype(jnp.uint32), counters.astype(jnp.uint32))
                token_ids = lm_head_sample_token_ids(
                    hidden[:, :1, :],
                    params,
                    self.config,
                    temperatures=temp_values,
                    rng_keys=rng_keys,
                    hidden_is_normed=False,
                    is_prefill=False,
                ).astype(jnp.int32)
                updated_conv = updated_hybrid_state.conv_state.astype(conv_state_table.dtype)
                updated_recurrent = updated_hybrid_state.recurrent_state.astype(
                    recurrent_state_table.dtype
                )
                updated_conv_table = conv_state_table.at[slot_ids].set(updated_conv)
                updated_recurrent_table = recurrent_state_table.at[slot_ids].set(
                    updated_recurrent
                )
                updated_seq_lens_table = seq_lens_table.at[slot_ids].set(seq_lens + 1)
                updated_last_tokens = last_tokens_table.at[slot_ids].set(token_ids)
                updated_rng_counters = rng_counter_table.at[slot_ids].set(counters + 1)
                return (
                    token_ids,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    updated_seq_lens_table,
                    updated_last_tokens,
                    updated_rng_counters,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(1, 2, 3, 4, 7, 8, 9),
            )

        (
            token_ids,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            seq_lens_table,
            last_tokens_table,
            rng_counter_table,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
                resident_last_tokens,
                resident_rng_counters,
                temperatures,
            ),
            "forward_step_sampled_token_ids_resident_dense_slot_carry_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            resident_seq_lens=seq_lens_table,
            resident_last_tokens=last_tokens_table,
            resident_rng_counters=rng_counter_table,
        )

    def forward_greedy_decode_burst_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        decode_steps: int,
    ) -> ExecutorOutput:
        """Run greedy burst decode while owning hybrid table gather/scatter in JIT."""
        if batch.is_prefill:
            raise ValueError("forward_greedy_decode_burst_table_jit is decode-only")
        if decode_steps < 1:
            raise ValueError("decode_steps must be positive")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_greedy_decode_burst_table_jit requires initialized hybrid state tables")
        self._log_step("forward_greedy_decode_burst_table_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "token-ids-burst-table",
            int(decode_steps),
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
        )
        if key not in self._jit_cache:
            static_decode_steps = int(decode_steps)

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                active = query_lens > 0
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                initial_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]

                def step(carry, _):
                    (
                        step_tokens,
                        step_positions,
                        step_seq_lens,
                        step_k_cache,
                        step_v_cache,
                        step_conv_state,
                        step_recurrent_state,
                    ) = carry
                    step_batch = ScheduledBatch(
                        tokens=step_tokens,
                        positions=step_positions,
                        seq_ids=jnp.where(
                            active,
                            safe_slot_ids.astype(jnp.int32),
                            jnp.full_like(safe_slot_ids, -1),
                        ),
                        query_start_loc=query_start_loc,
                        is_prefill=False,
                        num_prefill_tokens=0,
                        num_decode_tokens=num_query_tokens,
                        block_tables=block_tables,
                        seq_lens=step_seq_lens,
                    )
                    attention_metadata = self.backend.build_attention_metadata(
                        positions=step_batch.positions,
                        block_tables=step_batch.block_tables,
                        seq_lens=step_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=False,
                        query_start_loc=step_batch.query_start_loc,
                        num_prefill_tokens=0,
                        num_decode_tokens=step_batch.num_decode_tokens,
                    )
                    kv_state = KVCacheState(
                        k_cache=step_k_cache,
                        v_cache=step_v_cache,
                        block_table=step_batch.block_tables,
                        kv_lens=step_batch.seq_lens,
                        slot_mapping=attention_metadata.slot_mapping,
                    )
                    hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                        step_batch.tokens,
                        params,
                        self.config,
                        positions=step_batch.positions,
                        kv_cache_state=kv_state,
                        attention_metadata=attention_metadata,
                        hybrid_state=HybridLayerState(step_conv_state, step_recurrent_state),
                        is_prefill=False,
                        return_hidden=True,
                        return_hidden_with_logits=False,
                        last_logits_only=False,
                        backend=self.backend,
                        hybrid_state_layerwise=True,
                    )
                    token_ids, _, _ = lm_head_token_ids_and_topk(
                        hidden[:, :1, :],
                        params,
                        self.config,
                        hidden_is_normed=False,
                        is_prefill=False,
                        top_k=0,
                    )
                    token_ids = token_ids[:, 0].astype(jnp.int32)
                    next_tokens = token_ids[:, None]
                    next_positions = step_positions + active[:, None].astype(step_positions.dtype)
                    next_seq_lens = jnp.where(active, step_seq_lens + 1, step_seq_lens)
                    next_k_cache = updated_kv_state.k_cache.astype(step_k_cache.dtype)
                    next_v_cache = updated_kv_state.v_cache.astype(step_v_cache.dtype)
                    next_conv_state = updated_hybrid_state.conv_state.astype(step_conv_state.dtype)
                    next_recurrent_state = updated_hybrid_state.recurrent_state.astype(step_recurrent_state.dtype)
                    return (
                        next_tokens,
                        next_positions,
                        next_seq_lens,
                        next_k_cache,
                        next_v_cache,
                        next_conv_state,
                        next_recurrent_state,
                    ), token_ids

                initial = (
                    tokens,
                    initial_positions,
                    seq_lens,
                    k_cache,
                    v_cache,
                    conv_state,
                    recurrent_state,
                )
                final, token_ids_by_step = jax.lax.scan(
                    step,
                    initial,
                    jnp.arange(static_decode_steps, dtype=jnp.int32),
                )
                (
                    _tokens,
                    _positions,
                    _seq_lens,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                ) = final
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    final_conv_state,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    final_recurrent_state,
                    mode="drop",
                )
                return (
                    token_ids_by_step.transpose(1, 0),
                    final_k_cache,
                    final_v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(6, 7, 8, 9),
            )

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
            ),
            "forward_greedy_decode_burst_table_jit:decode",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def forward_greedy_decode_burst_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        decode_steps: int,
    ) -> ExecutorOutput:
        """Run several greedy decode steps inside one compiled graph.

        The final emitted token is intentionally not written to the KV/GDN
        state, matching the normal server contract: a generated token is cached
        only when it becomes the next scheduled decode input.
        """
        if batch.is_prefill:
            raise ValueError("forward_greedy_decode_burst_jit requires a decode batch")
        if decode_steps < 1:
            raise ValueError("decode_steps must be positive")
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_greedy_decode_burst_jit requires initialized hybrid_state")
        self._log_step("forward_greedy_decode_burst_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "token-ids-burst",
            int(decode_steps),
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
        )
        if key not in self._jit_cache:
            static_decode_steps = int(decode_steps)

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                active = query_lens > 0
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                initial_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]

                def step(carry, _):
                    (
                        step_tokens,
                        step_positions,
                        step_seq_lens,
                        step_k_cache,
                        step_v_cache,
                        step_conv_state,
                        step_recurrent_state,
                    ) = carry
                    step_batch = ScheduledBatch(
                        tokens=step_tokens,
                        positions=step_positions,
                        seq_ids=jnp.where(
                            active,
                            jnp.arange(step_tokens.shape[0], dtype=jnp.int32),
                            jnp.full((step_tokens.shape[0],), -1, dtype=jnp.int32),
                        ),
                        query_start_loc=query_start_loc,
                        is_prefill=False,
                        num_prefill_tokens=0,
                        num_decode_tokens=num_query_tokens,
                        block_tables=block_tables,
                        seq_lens=step_seq_lens,
                    )
                    attention_metadata = self.backend.build_attention_metadata(
                        positions=step_batch.positions,
                        block_tables=step_batch.block_tables,
                        seq_lens=step_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=False,
                        query_start_loc=step_batch.query_start_loc,
                        num_prefill_tokens=0,
                        num_decode_tokens=step_batch.num_decode_tokens,
                    )
                    kv_state = KVCacheState(
                        k_cache=step_k_cache,
                        v_cache=step_v_cache,
                        block_table=step_batch.block_tables,
                        kv_lens=step_batch.seq_lens,
                        slot_mapping=attention_metadata.slot_mapping,
                    )
                    hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                        step_batch.tokens,
                        params,
                        self.config,
                        positions=step_batch.positions,
                        kv_cache_state=kv_state,
                        attention_metadata=attention_metadata,
                        hybrid_state=HybridLayerState(step_conv_state, step_recurrent_state),
                        is_prefill=False,
                        return_hidden=True,
                        return_hidden_with_logits=False,
                        last_logits_only=False,
                        backend=self.backend,
                    )
                    token_ids, _, _ = lm_head_token_ids_and_topk(
                        hidden[:, :1, :],
                        params,
                        self.config,
                        hidden_is_normed=False,
                        is_prefill=False,
                        top_k=0,
                    )
                    token_ids = token_ids[:, 0].astype(jnp.int32)
                    next_tokens = token_ids[:, None]
                    next_positions = step_positions + active[:, None].astype(step_positions.dtype)
                    next_seq_lens = jnp.where(active, step_seq_lens + 1, step_seq_lens)
                    next_k_cache = updated_kv_state.k_cache.astype(step_k_cache.dtype)
                    next_v_cache = updated_kv_state.v_cache.astype(step_v_cache.dtype)
                    next_conv_state = updated_hybrid_state.conv_state.astype(step_conv_state.dtype)
                    next_recurrent_state = updated_hybrid_state.recurrent_state.astype(step_recurrent_state.dtype)
                    return (
                        next_tokens,
                        next_positions,
                        next_seq_lens,
                        next_k_cache,
                        next_v_cache,
                        next_conv_state,
                        next_recurrent_state,
                    ), token_ids

                initial = (
                    tokens,
                    initial_positions,
                    seq_lens,
                    k_cache,
                    v_cache,
                    conv_state,
                    recurrent_state,
                )
                final, token_ids_by_step = jax.lax.scan(
                    step,
                    initial,
                    jnp.arange(static_decode_steps, dtype=jnp.int32),
                )
                (
                    _tokens,
                    _positions,
                    _seq_lens,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                ) = final
                return (
                    token_ids_by_step.transpose(1, 0),
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                )

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(6, 7),
            )

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
            ),
            "forward_step_token_ids_jit:decode_burst",
        )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def mtp1_greedy_burst_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        draft_token: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """K=1 verifier that reuses the optimized greedy burst table graph.

        For greedy decoding, an accepted draft is exactly the target model's
        first emitted token. Running the normal two-step greedy burst therefore
        gives the same accepted-path state and bonus token without feeding the
        dynamic draft token into the second target-model step. Rejected outputs
        are only safe to discard; callers must not commit this path unless all
        active rows accept.
        """
        if batch.is_prefill:
            raise ValueError("mtp1_greedy_burst_table_jit is decode-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("mtp1_greedy_burst_table_jit requires initialized hybrid state tables")
        self._log_step("mtp1_greedy_burst_table_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-greedy-burst-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            os.environ.get("NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0"),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            assume_all_accept = os.environ.get(
                "NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0"
            ) in {"1", "true", "yes", "on", "True"}
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                draft_token_arg,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0) & (draft_token_arg >= 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                active = row_valid
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)
                initial_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]

                def build_step_outputs(include_hidden: bool):
                    def step(carry, _):
                        (
                            step_tokens,
                            step_positions,
                            step_seq_lens,
                            step_k_cache,
                            step_v_cache,
                            step_conv_state,
                            step_recurrent_state,
                        ) = carry
                        step_batch = ScheduledBatch(
                            tokens=step_tokens,
                            positions=step_positions,
                            seq_ids=jnp.where(
                                active,
                                safe_slot_ids.astype(jnp.int32),
                                jnp.full_like(safe_slot_ids, -1),
                            ),
                            query_start_loc=query_start_loc,
                            is_prefill=False,
                            num_prefill_tokens=0,
                            num_decode_tokens=num_query_tokens,
                            block_tables=block_tables,
                            seq_lens=step_seq_lens,
                        )
                        attention_metadata = self.backend.build_attention_metadata(
                            positions=step_batch.positions,
                            block_tables=step_batch.block_tables,
                            seq_lens=step_batch.seq_lens,
                            block_size=self.config.block_size,
                            is_prefill=False,
                            query_start_loc=step_batch.query_start_loc,
                            num_prefill_tokens=0,
                            num_decode_tokens=step_batch.num_decode_tokens,
                        )
                        kv_state = KVCacheState(
                            k_cache=step_k_cache,
                            v_cache=step_v_cache,
                            block_table=step_batch.block_tables,
                            kv_lens=step_batch.seq_lens,
                            slot_mapping=attention_metadata.slot_mapping,
                        )
                        hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                            step_batch.tokens,
                            params,
                            self.config,
                            positions=step_batch.positions,
                            kv_cache_state=kv_state,
                            attention_metadata=attention_metadata,
                            hybrid_state=HybridLayerState(step_conv_state, step_recurrent_state),
                            is_prefill=False,
                            return_hidden=True,
                            return_hidden_with_logits=False,
                            last_logits_only=False,
                            backend=self.backend,
                            hybrid_state_layerwise=True,
                        )
                        token_ids, _, _ = lm_head_token_ids_and_topk(
                            hidden[:, :1, :],
                            params,
                            self.config,
                            hidden_is_normed=False,
                            is_prefill=False,
                            top_k=0,
                        )
                        token_ids = token_ids[:, 0].astype(jnp.int32)
                        next_tokens = token_ids[:, None]
                        next_positions = step_positions + active[:, None].astype(step_positions.dtype)
                        next_seq_lens = jnp.where(active, step_seq_lens + 1, step_seq_lens)
                        next_carry = (
                            next_tokens,
                            next_positions,
                            next_seq_lens,
                            updated_kv_state.k_cache.astype(step_k_cache.dtype),
                            updated_kv_state.v_cache.astype(step_v_cache.dtype),
                            updated_hybrid_state.conv_state.astype(step_conv_state.dtype),
                            updated_hybrid_state.recurrent_state.astype(step_recurrent_state.dtype),
                        )
                        if include_hidden:
                            return next_carry, (token_ids, hidden)
                        return next_carry, token_ids

                    initial = (
                        tokens,
                        initial_positions,
                        seq_lens,
                        k_cache,
                        v_cache,
                        conv_state,
                        recurrent_state,
                    )
                    return jax.lax.scan(
                        step,
                        initial,
                        jnp.arange(2, dtype=jnp.int32),
                    )

                if compute_next_draft:
                    final, scan_outputs = build_step_outputs(include_hidden=True)
                    token_ids_by_step, hidden_by_step = scan_outputs
                else:
                    final, token_ids_by_step = build_step_outputs(include_hidden=False)
                    hidden_by_step = None
                (
                    _tokens,
                    _positions,
                    _seq_lens,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                ) = final
                target_token = token_ids_by_step[0].astype(jnp.int32)
                bonus_token = token_ids_by_step[1].astype(jnp.int32)
                accepted = (target_token == draft_token_arg) & row_valid
                if compute_next_draft:
                    hidden1 = hidden_by_step[1]
                    mtp_hidden = (
                        rms_norm(hidden1, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                        if mtp_hidden_final_normed
                        else hidden1
                    )
                    mtp_positions = initial_positions[:, 0] + jnp.asarray(2, dtype=positions.dtype)
                    mtp_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=mtp_hidden,
                        next_token_ids=bonus_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=mtp_positions[:, None],
                    )
                    next_draft_token = mtp_token_ids[:, 0].astype(jnp.int32)
                else:
                    next_draft_token = jnp.full_like(target_token, -1)
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    final_conv_state,
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    final_recurrent_state,
                    mode="drop",
                )
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    accepted,
                    jnp.stack(
                        [
                            target_token,
                            bonus_token,
                            next_draft_token,
                            accepted.astype(jnp.int32),
                        ],
                        axis=1,
                    ),
                    final_k_cache,
                    final_v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    seq_lens + row_valid.astype(jnp.int32),
                )

            donate_argnums = (6, 7, 8, 9) if assume_all_accept else ()
            self._jit_cache[key] = jax.jit(compiled, donate_argnums=donate_argnums)

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            host_payload,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_greedy_burst_table_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            host_payload=host_payload,
        )

    def mtp1_greedy_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_token: int | jnp.ndarray,
        next_mtp_position: int | jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """Fused JIT path for K=1 greedy MTP verification.

        This keeps accept/reject policy in the engine, but avoids separate
        dispatches for verifier logits, bonus-token selection, and accepted-path
        MTP draft seeding.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp1_greedy_step_jit requires initialized hybrid_state")
        if not batch.is_prefill or batch.tokens.shape[1] != 2:
            raise ValueError("mtp1_greedy_step_jit expects a two-token verifier prefill batch")
        self._log_step("mtp1_greedy_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
        )
        if key not in self._jit_cache:

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_prefill_tokens,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_token_arg,
                next_mtp_position_arg,
            ):
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=True,
                    num_prefill_tokens=num_prefill_tokens,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                attention_metadata = self.backend.build_attention_metadata(
                    positions=step_batch.positions,
                    block_tables=step_batch.block_tables,
                    seq_lens=step_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=True,
                    query_start_loc=step_batch.query_start_loc,
                    num_prefill_tokens=step_batch.num_prefill_tokens,
                    num_decode_tokens=step_batch.num_decode_tokens,
                )
                kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=step_batch.block_tables,
                    kv_lens=step_batch.seq_lens,
                    slot_mapping=attention_metadata.slot_mapping,
                )
                (hidden, verify_logits), updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=True,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    last_logits_only=False,
                    backend=self.backend,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids = jnp.argmax(verify_logits, axis=-1).astype(jnp.int32)
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = target_token == draft_token_arg

                mtp_hidden = hidden_norm[:, 1:2, :] if mtp_hidden_final_normed else hidden[:, 1:2, :]
                next_mtp_positions = jnp.broadcast_to(
                    jnp.asarray(next_mtp_position_arg, dtype=jnp.int32),
                    (batch.tokens.shape[0],),
                )
                mtp_token_ids, _ = mtp_forward_token_ids(
                    hidden_state=mtp_hidden,
                    next_token_ids=bonus_token[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=next_mtp_positions[:, None],
                )
                next_draft_token = mtp_token_ids[:, 0].astype(jnp.int32)
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    accepted,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                )

            # Do not donate here: the runner only commits this speculative
            # verifier output if every row accepts. On rejection, it discards
            # these tentative cache/hybrid updates and falls back to the
            # canonical target-decode repair path.
            donate_argnums = (8, 9) if os.environ.get(
                "NANO_VLLM_JAX_MTP_DONATE_CACHE", "0"
            ) in {"1", "true", "yes", "on", "True"} else ()
            self._jit_cache[key] = jax.jit(compiled, donate_argnums=donate_argnums)

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_prefill_tokens, dtype=jnp.int32),
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_token, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp1_greedy_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    def mtp1_two_decode_greedy_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """K=1 greedy MTP verifier using one two-token target decode.

        1. Hidden at position 0 verifies the stored draft.
        2. Hidden at position 1 produces the target bonus token.
        3. Per-token hybrid prefix states select the committed state:
           rejected rows commit after position 0, accepted rows after position 1.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp1_two_decode_greedy_step_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_two_decode_greedy_step_jit expects a decode batch")
        self._log_step("mtp1_two_decode_greedy_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-one-pass-prefix-prefill-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1"),
            os.environ.get("NANO_VLLM_JAX_MTP_ASSUME_ALL_ACCEPT", "0"),
            os.environ.get("NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0"),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            one_pass_decode_mode = os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            assume_all_accept = os.environ.get("NANO_VLLM_JAX_MTP_ASSUME_ALL_ACCEPT", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            } or os.environ.get("NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_token_arg,
                next_mtp_position_arg,
            ):
                verify_tokens = jnp.concatenate([tokens, draft_token_arg[:, None]], axis=1)
                verify_positions = jnp.concatenate([positions, positions + 1], axis=1)
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (row_query_lens > 0) & (seq_ids >= 0)
                row_has_draft = row_valid & (draft_token_arg >= 0)
                verify_query_lens = (
                    jnp.where(row_valid, jnp.full_like(row_query_lens, 2), jnp.zeros_like(row_query_lens))
                    if not one_pass_decode_mode
                    else row_query_lens + row_has_draft.astype(jnp.int32)
                )
                verify_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(verify_query_lens),
                    ]
                )
                if not one_pass_decode_mode:
                    packed_width = int(tokens.shape[0]) * int(verify_tokens.shape[1])
                    packed_prefill_token_row_ids = jnp.broadcast_to(
                        jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None],
                        verify_tokens.shape,
                    ).reshape((1, packed_width))
                    verify_model_tokens = verify_tokens.reshape((1, packed_width))
                    verify_model_positions = verify_positions.reshape((1, packed_width))
                else:
                    packed_prefill_token_row_ids = None
                    verify_model_tokens = verify_tokens
                    verify_model_positions = verify_positions
                verify_batch = ScheduledBatch(
                    tokens=verify_model_tokens,
                    positions=verify_model_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=not one_pass_decode_mode,
                    num_prefill_tokens=0 if one_pass_decode_mode else jnp.sum(verify_query_lens),
                    num_decode_tokens=jnp.sum(verify_query_lens) if one_pass_decode_mode else 0,
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_has_draft.astype(jnp.int32),
                    packed_prefill=packed_prefill_token_row_ids is not None,
                    token_row_ids=packed_prefill_token_row_ids,
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=not one_pass_decode_mode,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=verify_batch.num_prefill_tokens,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                    token_row_ids=(
                        verify_batch.token_row_ids
                        if verify_batch.packed_prefill
                        else None
                    ),
                    max_query_len=2 if verify_batch.packed_prefill else None,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                forward_result = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=not one_pass_decode_mode,
                    return_hidden=True,
                    return_first_prefix_hybrid=not assume_all_accept,
                    backend=self.backend,
                )
                if assume_all_accept:
                    hidden, updated_kv_state, updated_hybrid_state = forward_result
                    first_prefix_hybrid_state = updated_hybrid_state
                else:
                    (
                        hidden,
                        updated_kv_state,
                        updated_hybrid_state,
                        first_prefix_hybrid_state,
                    ) = forward_result

                if not one_pass_decode_mode:
                    hidden = hidden.reshape((tokens.shape[0], verify_tokens.shape[1], hidden.shape[-1]))
                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                    hidden_norm,
                    params,
                    self.config,
                    hidden_is_normed=True,
                    is_prefill=not one_pass_decode_mode,
                    top_k=2 if bonus_margin_threshold > 0 else 0,
                )
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = (target_token == draft_token_arg) & row_has_draft
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))
                if bonus_margin_threshold > 0:
                    bonus_top2 = bonus_topk_values[:, 1]
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))

                pos_current = verify_positions[:, 0]
                pos_next_after_reject = pos_current + jnp.asarray(1, dtype=pos_current.dtype)
                pos_next_after_accept = pos_current + jnp.asarray(2, dtype=pos_current.dtype)

                # Select the next MTP seed per row before running the MTP head.
                # The previous rowwise implementation ran one full MTP forward
                # for accepted rows and another for rejected rows, then selected
                # between their logits. Mixed B>1 batches paid both costs.
                hidden_for_mtp = hidden_norm if mtp_hidden_final_normed else hidden
                selected_mtp_hidden = jnp.where(
                    accepted[:, None, None],
                    hidden_for_mtp[:, 1:2, :],
                    hidden_for_mtp[:, 0:1, :],
                )
                selected_mtp_token = jnp.where(accepted, bonus_token, target_token)
                selected_mtp_position = jnp.where(
                    accepted,
                    pos_next_after_accept,
                    pos_next_after_reject,
                )
                def run_next_mtp(_):
                    next_mtp_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=selected_mtp_hidden,
                        next_token_ids=selected_mtp_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=selected_mtp_position[:, None],
                    )
                    return next_mtp_token_ids[:, 0].astype(jnp.int32)

                def skip_next_mtp(_):
                    return jnp.full_like(target_token, -1)

                if compute_next_draft:
                    next_draft_token = jax.lax.cond(
                        jnp.any(row_valid),
                        run_next_mtp,
                        skip_next_mtp,
                        operand=None,
                    )
                    next_draft_token = jnp.where(
                        row_valid,
                        next_draft_token,
                        jnp.full_like(next_draft_token, -1),
                    )
                else:
                    next_draft_token = skip_next_mtp(None)

                if assume_all_accept:
                    hybrid_after_current = updated_hybrid_state
                else:
                    hybrid_after_current = HybridLayerState(
                        conv_state=first_prefix_hybrid_state.conv_state
                        if first_prefix_hybrid_state.conv_state is not None
                        else None,
                        recurrent_state=first_prefix_hybrid_state.recurrent_state
                        if first_prefix_hybrid_state.recurrent_state is not None
                        else None,
                    )
                hybrid_after_draft = HybridLayerState(
                    conv_state=updated_hybrid_state.conv_state,
                    recurrent_state=updated_hybrid_state.recurrent_state,
                )

                def restore_slots(new_cache, old_cache, slots, keep_new):
                    leading_shape = new_cache.shape[:-4] if new_cache.ndim == 5 else new_cache.shape[:-3]
                    flat_new = new_cache.reshape(leading_shape + (-1,) + new_cache.shape[-2:])
                    flat_old = old_cache.reshape(leading_shape + (-1,) + old_cache.shape[-2:])
                    new_values = flat_new[..., slots, :, :]
                    old_values = flat_old[..., slots, :, :]
                    slot_mask = keep_new.reshape((1,) * len(leading_shape) + (keep_new.shape[0], 1, 1))
                    selected_values = jnp.where(slot_mask, new_values, old_values)
                    flat_new = flat_new.at[..., slots, :, :].set(selected_values)
                    return flat_new.reshape(new_cache.shape)

                def select_row_state(old_state, after_current, after_draft):
                    valid_mask = row_valid.reshape((row_valid.shape[0],) + (1,) * (old_state.ndim - 1))
                    accept_mask = accepted.reshape((accepted.shape[0],) + (1,) * (old_state.ndim - 1))
                    valid_selected = jnp.where(accept_mask, after_draft, after_current)
                    return jnp.where(valid_mask, valid_selected, old_state)

                selected_conv = select_row_state(
                    conv_state,
                    hybrid_after_current.conv_state,
                    hybrid_after_draft.conv_state,
                )
                selected_recurrent = select_row_state(
                    recurrent_state,
                    hybrid_after_current.recurrent_state,
                    hybrid_after_draft.recurrent_state,
                )

                # Slot 0 is the canonical current-token write for every active
                # row. Rejected draft slots are allowed to remain dirty: only
                # the current token is logically committed on reject, and the
                # next decode overwrites the draft slot.
                selected_k_cache = updated_kv_state.k_cache
                selected_v_cache = updated_kv_state.v_cache
                emitted_tokens = jnp.stack(
                    [
                        jnp.where(row_valid, target_token, jnp.zeros_like(target_token)),
                        jnp.where(accepted, bonus_token, jnp.zeros_like(bonus_token)),
                    ],
                    axis=1,
                ).astype(jnp.int32)
                emitted_counts = jnp.where(
                    row_valid,
                    jnp.asarray(1, dtype=jnp.int32) + accepted.astype(jnp.int32),
                    jnp.zeros_like(accepted.astype(jnp.int32)),
                )[:, None]
                committed_seq_lens = seq_lens + emitted_counts[:, 0]
                accepted_counts = accepted.astype(jnp.int32)[:, None]
                emitted_totals = emitted_counts[:, 0].astype(jnp.int32)
                accepted_totals = accepted.astype(jnp.int32)
                rejected_totals = (row_valid & ~accepted).astype(jnp.int32)
                bonus_totals = accepted.astype(jnp.int32)
                accepted_bitmask = accepted.astype(jnp.int32)
                compact_summary = jnp.stack(
                    [
                        emitted_totals,
                        accepted_totals,
                        rejected_totals,
                        bonus_totals,
                        accepted_bitmask,
                    ],
                    axis=1,
                ).astype(jnp.int32)
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    accepted,
                    selected_k_cache,
                    selected_v_cache,
                    selected_conv,
                    selected_recurrent,
                    committed_seq_lens,
                    emitted_tokens,
                    emitted_counts,
                    accepted_counts,
                    emitted_totals,
                    accepted_totals,
                    rejected_totals,
                    bonus_totals,
                    accepted_bitmask,
                    compact_summary,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(8, 9))

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            emitted_tokens,
            emitted_counts,
            accepted_counts,
            emitted_totals,
            accepted_totals,
            rejected_totals,
            bonus_totals,
            accepted_bitmask,
            compact_summary,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_token, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp1_two_decode_greedy_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            emitted_tokens=emitted_tokens,
            emitted_counts=emitted_counts,
            accepted_counts=accepted_counts,
            emitted_totals=emitted_totals,
            accepted_totals=accepted_totals,
            rejected_totals=rejected_totals,
            bonus_totals=bonus_totals,
            accepted_bitmask=accepted_bitmask,
            compact_summary=compact_summary,
            burst_groups=1,
        )

    def mtp1_two_decode_greedy_table_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """Exact K=1 greedy verifier that owns resident hybrid table updates.

        This is the rowwise-correct form of the two-token verifier. It gathers
        the active rows from the resident hybrid table, runs one width-2 target
        decode, selects either the after-current-token or after-draft state per
        row, and scatters the selected state back into the resident table.
        """
        del next_mtp_position
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("mtp1_two_decode_greedy_table_step_jit requires initialized hybrid_state_table")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_two_decode_greedy_table_step_jit expects a decode batch")
        self._log_step("mtp1_two_decode_greedy_table_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-one-pass-prefix-table-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1"),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            one_pass_decode_mode = os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params_leaves,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                draft_token_arg,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (row_query_lens > 0) & (seq_ids >= 0) & (slot_ids >= 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                state_mask = row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1))
                conv_state = jnp.where(state_mask, conv_state, jnp.zeros_like(conv_state))
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                verify_tokens = jnp.concatenate([tokens, draft_token_arg[:, None]], axis=1)
                verify_positions = jnp.concatenate([positions, positions + 1], axis=1)
                row_has_draft = row_valid & (draft_token_arg >= 0)
                verify_query_lens = (
                    jnp.where(row_valid, jnp.full_like(row_query_lens, 2), jnp.zeros_like(row_query_lens))
                    if not one_pass_decode_mode
                    else row_query_lens + row_has_draft.astype(jnp.int32)
                )
                verify_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(verify_query_lens),
                    ]
                )
                if not one_pass_decode_mode:
                    packed_width = int(tokens.shape[0]) * int(verify_tokens.shape[1])
                    packed_prefill_token_row_ids = jnp.broadcast_to(
                        jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None],
                        verify_tokens.shape,
                    ).reshape((1, packed_width))
                    verify_model_tokens = verify_tokens.reshape((1, packed_width))
                    verify_model_positions = verify_positions.reshape((1, packed_width))
                else:
                    packed_prefill_token_row_ids = None
                    verify_model_tokens = verify_tokens
                    verify_model_positions = verify_positions
                verify_batch = ScheduledBatch(
                    tokens=verify_model_tokens,
                    positions=verify_model_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=not one_pass_decode_mode,
                    num_prefill_tokens=0 if one_pass_decode_mode else jnp.sum(verify_query_lens),
                    num_decode_tokens=jnp.sum(verify_query_lens) if one_pass_decode_mode else 0,
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_has_draft.astype(jnp.int32),
                    packed_prefill=packed_prefill_token_row_ids is not None,
                    token_row_ids=packed_prefill_token_row_ids,
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=not one_pass_decode_mode,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=verify_batch.num_prefill_tokens,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                    token_row_ids=(
                        verify_batch.token_row_ids
                        if verify_batch.packed_prefill
                        else None
                    ),
                    max_query_len=2 if verify_batch.packed_prefill else None,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                (
                    hidden,
                    updated_kv_state,
                    updated_hybrid_state,
                    first_prefix_hybrid_state,
                ) = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=not one_pass_decode_mode,
                    return_hidden=True,
                    return_first_prefix_hybrid=True,
                    backend=self.backend,
                )
                if not one_pass_decode_mode:
                    hidden = hidden.reshape((tokens.shape[0], verify_tokens.shape[1], hidden.shape[-1]))

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                    hidden_norm,
                    params,
                    self.config,
                    hidden_is_normed=True,
                    is_prefill=not one_pass_decode_mode,
                    top_k=2 if bonus_margin_threshold > 0 else 0,
                )
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = (target_token == draft_token_arg) & row_has_draft
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))
                if bonus_margin_threshold > 0:
                    bonus_top2 = bonus_topk_values[:, 1]
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))

                pos_current = verify_positions[:, 0]
                pos_next_after_reject = pos_current + jnp.asarray(1, dtype=pos_current.dtype)
                pos_next_after_accept = pos_current + jnp.asarray(2, dtype=pos_current.dtype)
                hidden_for_mtp = hidden_norm if mtp_hidden_final_normed else hidden
                selected_mtp_hidden = jnp.where(
                    accepted[:, None, None],
                    hidden_for_mtp[:, 1:2, :],
                    hidden_for_mtp[:, 0:1, :],
                )
                selected_mtp_token = jnp.where(accepted, bonus_token, target_token)
                selected_mtp_position = jnp.where(
                    accepted,
                    pos_next_after_accept,
                    pos_next_after_reject,
                )

                def run_next_mtp(_):
                    next_mtp_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=selected_mtp_hidden,
                        next_token_ids=selected_mtp_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=selected_mtp_position[:, None],
                    )
                    return next_mtp_token_ids[:, 0].astype(jnp.int32)

                def skip_next_mtp(_):
                    return jnp.full_like(target_token, -1)

                if compute_next_draft:
                    next_draft_token = jax.lax.cond(
                        jnp.any(row_valid),
                        run_next_mtp,
                        skip_next_mtp,
                        operand=None,
                    )
                    next_draft_token = jnp.where(
                        row_valid,
                        next_draft_token,
                        jnp.full_like(next_draft_token, -1),
                    )
                else:
                    next_draft_token = skip_next_mtp(None)

                valid_mask = row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1))
                accept_conv_mask = accepted.reshape((accepted.shape[0],) + (1,) * (conv_state.ndim - 1))
                selected_conv = jnp.where(
                    valid_mask,
                    jnp.where(
                        accept_conv_mask,
                        updated_hybrid_state.conv_state,
                        first_prefix_hybrid_state.conv_state,
                    ),
                    conv_state,
                )
                recurrent_valid_mask = row_valid.reshape(
                    (row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)
                )
                accept_recurrent_mask = accepted.reshape(
                    (accepted.shape[0],) + (1,) * (recurrent_state.ndim - 1)
                )
                selected_recurrent = jnp.where(
                    recurrent_valid_mask,
                    jnp.where(
                        accept_recurrent_mask,
                        updated_hybrid_state.recurrent_state,
                        first_prefix_hybrid_state.recurrent_state,
                    ),
                    recurrent_state,
                )
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    selected_conv.astype(conv_state_table.dtype),
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    selected_recurrent.astype(recurrent_state_table.dtype),
                    mode="drop",
                )

                emitted_tokens = jnp.stack(
                    [
                        jnp.where(row_valid, target_token, jnp.zeros_like(target_token)),
                        jnp.where(accepted, bonus_token, jnp.zeros_like(bonus_token)),
                    ],
                    axis=1,
                ).astype(jnp.int32)
                emitted_counts = jnp.where(
                    row_valid,
                    jnp.asarray(1, dtype=jnp.int32) + accepted.astype(jnp.int32),
                    jnp.zeros_like(accepted.astype(jnp.int32)),
                )[:, None]
                committed_seq_lens = seq_lens + emitted_counts[:, 0]
                accepted_counts = accepted.astype(jnp.int32)[:, None]
                emitted_totals = emitted_counts[:, 0].astype(jnp.int32)
                accepted_totals = accepted.astype(jnp.int32)
                rejected_totals = (row_valid & ~accepted).astype(jnp.int32)
                bonus_totals = accepted.astype(jnp.int32)
                accepted_bitmask = accepted.astype(jnp.int32)
                compact_summary = jnp.stack(
                    [
                        emitted_totals,
                        accepted_totals,
                        rejected_totals,
                        bonus_totals,
                        accepted_bitmask,
                    ],
                    axis=1,
                ).astype(jnp.int32)
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    accepted,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    committed_seq_lens,
                    emitted_tokens,
                    emitted_counts,
                    accepted_counts,
                    emitted_totals,
                    accepted_totals,
                    rejected_totals,
                    bonus_totals,
                    accepted_bitmask,
                    compact_summary,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(7, 8, 9, 10))

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            emitted_tokens,
            emitted_counts,
            accepted_counts,
            emitted_totals,
            accepted_totals,
            rejected_totals,
            bonus_totals,
            accepted_bitmask,
            compact_summary,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_two_decode_greedy_table_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            emitted_tokens=emitted_tokens,
            emitted_counts=emitted_counts,
            accepted_counts=accepted_counts,
            emitted_totals=emitted_totals,
            accepted_totals=accepted_totals,
            rejected_totals=rejected_totals,
            bonus_totals=bonus_totals,
            accepted_bitmask=accepted_bitmask,
            compact_summary=compact_summary,
            burst_groups=1,
            hybrid_state_is_table=True,
        )

    def mtp1_two_decode_greedy_table_burst_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
        burst_groups: int,
    ) -> MTP1GreedyOutput:
        """Exact K=1 greedy verifier for several resident-table groups.

        Each group verifies one draft with the target model, selects the
        accepted or rejected prefix state on device, regenerates the next draft,
        and feeds the committed token into the next group. The host only sees a
        fixed per-group emitted-token/count matrix after the burst completes.
        """
        del next_mtp_position
        burst_groups = int(burst_groups)
        if burst_groups <= 1:
            return self.mtp1_two_decode_greedy_table_step_jit(
                batch,
                cache_storage=cache_storage,
                hybrid_state_table=hybrid_state_table,
                hybrid_slot_ids=hybrid_slot_ids,
                draft_token=draft_token,
                next_mtp_position=jnp.zeros((batch.tokens.shape[0],), dtype=jnp.int32),
                mtp_hidden_final_normed=mtp_hidden_final_normed,
            )
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("mtp1_two_decode_greedy_table_burst_step_jit requires initialized hybrid_state_table")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_two_decode_greedy_table_burst_step_jit expects a decode batch")
        self._log_step("mtp1_two_decode_greedy_table_burst_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-one-pass-prefix-table-greedy-burst",
            burst_groups,
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1"),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            static_burst_groups = burst_groups
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            one_pass_decode_mode = os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params_leaves,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                draft_token_arg,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (row_query_lens > 0) & (seq_ids >= 0) & (slot_ids >= 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                initial_conv_state = conv_state_table[safe_slot_ids]
                initial_recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_mask = row_valid.reshape((row_valid.shape[0],) + (1,) * (initial_conv_state.ndim - 1))
                recurrent_mask = row_valid.reshape(
                    (row_valid.shape[0],) + (1,) * (initial_recurrent_state.ndim - 1)
                )
                initial_conv_state = jnp.where(conv_mask, initial_conv_state, jnp.zeros_like(initial_conv_state))
                initial_recurrent_state = jnp.where(
                    recurrent_mask,
                    initial_recurrent_state,
                    jnp.zeros_like(initial_recurrent_state),
                )

                def run_group(carry, _group_idx):
                    (
                        current_tokens,
                        current_positions,
                        current_seq_lens,
                        current_k_cache,
                        current_v_cache,
                        current_conv_state,
                        current_recurrent_state,
                        current_draft_token,
                        current_active,
                    ) = carry
                    row_has_draft = current_active & (current_draft_token >= 0)
                    verify_tokens = jnp.concatenate([current_tokens, current_draft_token[:, None]], axis=1)
                    verify_positions = jnp.concatenate([current_positions, current_positions + 1], axis=1)
                    verify_query_lens = (
                        jnp.where(
                            current_active,
                            jnp.full_like(current_active.astype(jnp.int32), 2),
                            jnp.zeros_like(current_active.astype(jnp.int32)),
                        )
                        if not one_pass_decode_mode
                        else current_active.astype(jnp.int32) + row_has_draft.astype(jnp.int32)
                    )
                    verify_query_start_loc = jnp.concatenate(
                        [
                            jnp.zeros((1,), dtype=jnp.int32),
                            jnp.cumsum(verify_query_lens),
                        ]
                    )
                    if not one_pass_decode_mode:
                        packed_width = int(tokens.shape[0]) * int(verify_tokens.shape[1])
                        packed_prefill_token_row_ids = jnp.broadcast_to(
                            jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None],
                            verify_tokens.shape,
                        ).reshape((1, packed_width))
                        verify_model_tokens = verify_tokens.reshape((1, packed_width))
                        verify_model_positions = verify_positions.reshape((1, packed_width))
                    else:
                        packed_prefill_token_row_ids = None
                        verify_model_tokens = verify_tokens
                        verify_model_positions = verify_positions
                    verify_batch = ScheduledBatch(
                        tokens=verify_model_tokens,
                        positions=verify_model_positions,
                        seq_ids=jnp.where(
                            current_active,
                            seq_ids,
                            jnp.full_like(seq_ids, -1),
                        ),
                        query_start_loc=verify_query_start_loc,
                        is_prefill=not one_pass_decode_mode,
                        num_prefill_tokens=0 if one_pass_decode_mode else jnp.sum(verify_query_lens),
                        num_decode_tokens=jnp.sum(verify_query_lens) if one_pass_decode_mode else 0,
                        block_tables=block_tables,
                        seq_lens=current_seq_lens + row_has_draft.astype(jnp.int32),
                        packed_prefill=packed_prefill_token_row_ids is not None,
                        token_row_ids=packed_prefill_token_row_ids,
                    )
                    verify_metadata = self.backend.build_attention_metadata(
                        positions=verify_batch.positions,
                        block_tables=verify_batch.block_tables,
                        seq_lens=verify_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=not one_pass_decode_mode,
                        query_start_loc=verify_batch.query_start_loc,
                        num_prefill_tokens=verify_batch.num_prefill_tokens,
                        num_decode_tokens=verify_batch.num_decode_tokens,
                        token_row_ids=(
                            verify_batch.token_row_ids
                            if verify_batch.packed_prefill
                            else None
                        ),
                        max_query_len=2 if verify_batch.packed_prefill else None,
                    )
                    verify_kv_state = KVCacheState(
                        k_cache=current_k_cache,
                        v_cache=current_v_cache,
                        block_table=verify_batch.block_tables,
                        kv_lens=verify_batch.seq_lens,
                        slot_mapping=verify_metadata.slot_mapping,
                    )
                    (
                        hidden,
                        updated_kv_state,
                        updated_hybrid_state,
                        first_prefix_hybrid_state,
                    ) = model_forward_step(
                        verify_batch.tokens,
                        params,
                        self.config,
                        positions=verify_batch.positions,
                        kv_cache_state=verify_kv_state,
                        attention_metadata=verify_metadata,
                        hybrid_state=HybridLayerState(current_conv_state, current_recurrent_state),
                        is_prefill=not one_pass_decode_mode,
                        return_hidden=True,
                        return_first_prefix_hybrid=True,
                        backend=self.backend,
                    )
                    if not one_pass_decode_mode:
                        hidden = hidden.reshape((tokens.shape[0], verify_tokens.shape[1], hidden.shape[-1]))

                    hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    token_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                        hidden_norm,
                        params,
                        self.config,
                        hidden_is_normed=True,
                        is_prefill=not one_pass_decode_mode,
                        top_k=2 if bonus_margin_threshold > 0 else 0,
                    )
                    target_token = token_ids[:, 0]
                    bonus_token = token_ids[:, 1]
                    accepted = (target_token == current_draft_token) & row_has_draft
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))
                    if bonus_margin_threshold > 0:
                        bonus_top2 = bonus_topk_values[:, 1]
                        bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                        accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                        if batch_accept_policy == "all_or_none":
                            accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))

                    pos_current = verify_positions[:, 0]
                    pos_next_after_reject = pos_current + jnp.asarray(1, dtype=pos_current.dtype)
                    pos_next_after_accept = pos_current + jnp.asarray(2, dtype=pos_current.dtype)
                    hidden_for_mtp = hidden_norm if mtp_hidden_final_normed else hidden
                    selected_mtp_hidden = jnp.where(
                        accepted[:, None, None],
                        hidden_for_mtp[:, 1:2, :],
                        hidden_for_mtp[:, 0:1, :],
                    )
                    selected_mtp_token = jnp.where(accepted, bonus_token, target_token)
                    selected_mtp_position = jnp.where(
                        accepted,
                        pos_next_after_accept,
                        pos_next_after_reject,
                    )

                    def run_next_mtp(_):
                        next_mtp_token_ids, _ = mtp_forward_token_ids(
                            hidden_state=selected_mtp_hidden,
                            next_token_ids=selected_mtp_token[:, None],
                            embed_tokens=params.embed_tokens,
                            params=params.mtp_params,
                            config=self.config,
                            positions=selected_mtp_position[:, None],
                        )
                        return next_mtp_token_ids[:, 0].astype(jnp.int32)

                    def skip_next_mtp(_):
                        return jnp.full_like(target_token, -1)

                    if compute_next_draft:
                        next_draft_token = jax.lax.cond(
                            jnp.any(current_active),
                            run_next_mtp,
                            skip_next_mtp,
                            operand=None,
                        )
                        next_draft_token = jnp.where(
                            current_active,
                            next_draft_token,
                            jnp.full_like(next_draft_token, -1),
                        )
                    else:
                        next_draft_token = skip_next_mtp(None)

                    valid_mask = current_active.reshape(
                        (current_active.shape[0],) + (1,) * (current_conv_state.ndim - 1)
                    )
                    accept_conv_mask = accepted.reshape((accepted.shape[0],) + (1,) * (current_conv_state.ndim - 1))
                    selected_conv = jnp.where(
                        valid_mask,
                        jnp.where(
                            accept_conv_mask,
                            updated_hybrid_state.conv_state,
                            first_prefix_hybrid_state.conv_state,
                        ),
                        current_conv_state,
                    )
                    recurrent_valid_mask = current_active.reshape(
                        (current_active.shape[0],) + (1,) * (current_recurrent_state.ndim - 1)
                    )
                    accept_recurrent_mask = accepted.reshape(
                        (accepted.shape[0],) + (1,) * (current_recurrent_state.ndim - 1)
                    )
                    selected_recurrent = jnp.where(
                        recurrent_valid_mask,
                        jnp.where(
                            accept_recurrent_mask,
                            updated_hybrid_state.recurrent_state,
                            first_prefix_hybrid_state.recurrent_state,
                        ),
                        current_recurrent_state,
                    )

                    emitted_counts = jnp.where(
                        current_active,
                        jnp.asarray(1, dtype=jnp.int32) + accepted.astype(jnp.int32),
                        jnp.zeros_like(accepted.astype(jnp.int32)),
                    )
                    next_seq_lens = current_seq_lens + emitted_counts
                    # A rejected row has only committed the target token. Stop
                    # it inside this burst and resume from the next scheduler
                    # step so rowwise mixed batches keep the same boundary as
                    # canonical one-token decode.
                    next_active = current_active & accepted & (next_draft_token >= 0)
                    emitted_tokens = jnp.stack(
                        [
                            jnp.where(current_active, target_token, jnp.zeros_like(target_token)),
                            jnp.where(accepted, bonus_token, jnp.zeros_like(bonus_token)),
                        ],
                        axis=1,
                    ).astype(jnp.int32)
                    accepted_counts = accepted.astype(jnp.int32)
                    next_carry = (
                        selected_mtp_token[:, None].astype(current_tokens.dtype),
                        selected_mtp_position[:, None].astype(current_positions.dtype),
                        next_seq_lens,
                        updated_kv_state.k_cache,
                        updated_kv_state.v_cache,
                        selected_conv.astype(current_conv_state.dtype),
                        selected_recurrent.astype(current_recurrent_state.dtype),
                        next_draft_token,
                        next_active,
                    )
                    group_output = (
                        emitted_tokens,
                        emitted_counts,
                        accepted_counts,
                    )
                    return next_carry, group_output

                initial_carry = (
                    tokens,
                    positions,
                    seq_lens,
                    k_cache,
                    v_cache,
                    initial_conv_state,
                    initial_recurrent_state,
                    draft_token_arg.astype(jnp.int32),
                    row_valid,
                )
                final_carry, scan_outputs = jax.lax.scan(
                    run_group,
                    initial_carry,
                    jnp.arange(static_burst_groups, dtype=jnp.int32),
                )
                (
                    _final_tokens,
                    _final_positions,
                    final_seq_lens,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                    final_draft_token,
                    _final_active,
                ) = final_carry
                emitted_tokens_by_group, emitted_counts_by_group, accepted_counts_by_group = scan_outputs
                emitted_tokens = jnp.transpose(emitted_tokens_by_group, (1, 0, 2)).reshape(
                    (tokens.shape[0], static_burst_groups * 2)
                )
                emitted_counts = jnp.transpose(emitted_counts_by_group, (1, 0)).astype(jnp.int32)
                accepted_counts = jnp.transpose(accepted_counts_by_group, (1, 0)).astype(jnp.int32)
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    final_conv_state.astype(conv_state_table.dtype),
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    final_recurrent_state.astype(recurrent_state_table.dtype),
                    mode="drop",
                )
                target_token = emitted_tokens[:, 0]
                bonus_token = emitted_tokens[:, 1]
                accepted = accepted_counts[:, 0].astype(bool)
                return (
                    target_token,
                    bonus_token,
                    final_draft_token,
                    accepted,
                    final_k_cache,
                    final_v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    final_seq_lens,
                    emitted_tokens,
                    emitted_counts,
                    accepted_counts,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(7, 8, 9, 10))

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            emitted_tokens,
            emitted_counts,
            accepted_counts,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_two_decode_greedy_table_burst_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            emitted_tokens=emitted_tokens,
            emitted_counts=emitted_counts,
            accepted_counts=accepted_counts,
            burst_groups=burst_groups,
            hybrid_state_is_table=True,
        )

    def mtp1_seed_then_table_burst_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        mtp_hidden_final_normed: bool,
        burst_groups: int,
    ) -> MTP1GreedyOutput:
        """Seed the first K=1 draft and verify table-resident groups in one JIT."""
        burst_groups = int(burst_groups)
        if burst_groups < 1:
            raise ValueError("burst_groups must be >= 1")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("mtp1_seed_then_table_burst_step_jit requires initialized hybrid_state_table")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_seed_then_table_burst_step_jit expects a decode batch")
        self._log_step("mtp1_seed_then_table_burst_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-seed-then-table-burst-greedy",
            burst_groups,
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1"),
        )
        if key not in self._jit_cache:
            static_burst_groups = burst_groups
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            one_pass_decode_mode = os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params_leaves,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
            ):
                del positions
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (row_query_lens > 0) & (seq_ids >= 0) & (slot_ids >= 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                initial_conv_state = conv_state_table[safe_slot_ids]
                initial_recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_mask = row_valid.reshape((row_valid.shape[0],) + (1,) * (initial_conv_state.ndim - 1))
                recurrent_mask = row_valid.reshape(
                    (row_valid.shape[0],) + (1,) * (initial_recurrent_state.ndim - 1)
                )
                initial_conv_state = jnp.where(conv_mask, initial_conv_state, jnp.zeros_like(initial_conv_state))
                initial_recurrent_state = jnp.where(
                    recurrent_mask,
                    initial_recurrent_state,
                    jnp.zeros_like(initial_recurrent_state),
                )

                seed_positions = jnp.maximum(seq_lens - 1, 0).astype(jnp.int32)[:, None]
                seed_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=seed_positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=query_start_loc[-1].astype(jnp.int32),
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                seed_metadata = self.backend.build_attention_metadata(
                    positions=seed_batch.positions,
                    block_tables=seed_batch.block_tables,
                    seq_lens=seed_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=seed_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=seed_batch.num_decode_tokens,
                )
                seed_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=seed_batch.block_tables,
                    kv_lens=seed_batch.seq_lens,
                    slot_mapping=seed_metadata.slot_mapping,
                )
                seed_hidden, seed_updated_kv, seed_updated_hybrid = model_forward_step(
                    seed_batch.tokens,
                    params,
                    self.config,
                    positions=seed_batch.positions,
                    kv_cache_state=seed_kv_state,
                    attention_metadata=seed_metadata,
                    hybrid_state=HybridLayerState(initial_conv_state, initial_recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                seed_target_token, _, _ = lm_head_token_ids_and_topk(
                    seed_hidden[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                seed_target_token = seed_target_token[:, 0].astype(jnp.int32)
                seed_mtp_hidden = (
                    rms_norm(seed_hidden[:, :1, :], params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    if mtp_hidden_final_normed
                    else seed_hidden[:, :1, :]
                )
                seed_mtp_position = (
                    seq_lens.astype(jnp.int32)
                    + jnp.asarray(int(getattr(self.config, "mtp_position_offset", 0)), dtype=jnp.int32)
                )[:, None]
                initial_draft_token, _ = mtp_forward_token_ids(
                    hidden_state=seed_mtp_hidden,
                    next_token_ids=seed_target_token[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=seed_mtp_position,
                )
                initial_draft_token = jnp.where(
                    row_valid,
                    initial_draft_token[:, 0].astype(jnp.int32),
                    jnp.full_like(seed_target_token, -1),
                )

                def run_group(carry, _group_idx):
                    (
                        current_tokens,
                        current_positions,
                        current_seq_lens,
                        current_k_cache,
                        current_v_cache,
                        current_conv_state,
                        current_recurrent_state,
                        current_draft_token,
                        current_active,
                    ) = carry
                    row_has_draft = current_active & (current_draft_token >= 0)
                    verify_tokens = jnp.concatenate([current_tokens, current_draft_token[:, None]], axis=1)
                    verify_positions = jnp.concatenate([current_positions, current_positions + 1], axis=1)
                    verify_query_lens = (
                        jnp.where(
                            current_active,
                            jnp.full_like(current_active.astype(jnp.int32), 2),
                            jnp.zeros_like(current_active.astype(jnp.int32)),
                        )
                        if not one_pass_decode_mode
                        else current_active.astype(jnp.int32) + row_has_draft.astype(jnp.int32)
                    )
                    verify_query_start_loc = jnp.concatenate(
                        [
                            jnp.zeros((1,), dtype=jnp.int32),
                            jnp.cumsum(verify_query_lens),
                        ]
                    )
                    if not one_pass_decode_mode:
                        packed_width = int(tokens.shape[0]) * int(verify_tokens.shape[1])
                        packed_prefill_token_row_ids = jnp.broadcast_to(
                            jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None],
                            verify_tokens.shape,
                        ).reshape((1, packed_width))
                        verify_model_tokens = verify_tokens.reshape((1, packed_width))
                        verify_model_positions = verify_positions.reshape((1, packed_width))
                    else:
                        packed_prefill_token_row_ids = None
                        verify_model_tokens = verify_tokens
                        verify_model_positions = verify_positions
                    verify_batch = ScheduledBatch(
                        tokens=verify_model_tokens,
                        positions=verify_model_positions,
                        seq_ids=jnp.where(
                            current_active,
                            seq_ids,
                            jnp.full_like(seq_ids, -1),
                        ),
                        query_start_loc=verify_query_start_loc,
                        is_prefill=not one_pass_decode_mode,
                        num_prefill_tokens=0 if one_pass_decode_mode else jnp.sum(verify_query_lens),
                        num_decode_tokens=jnp.sum(verify_query_lens) if one_pass_decode_mode else 0,
                        block_tables=block_tables,
                        seq_lens=current_seq_lens + row_has_draft.astype(jnp.int32),
                        packed_prefill=packed_prefill_token_row_ids is not None,
                        token_row_ids=packed_prefill_token_row_ids,
                    )
                    verify_metadata = self.backend.build_attention_metadata(
                        positions=verify_batch.positions,
                        block_tables=verify_batch.block_tables,
                        seq_lens=verify_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=not one_pass_decode_mode,
                        query_start_loc=verify_batch.query_start_loc,
                        num_prefill_tokens=verify_batch.num_prefill_tokens,
                        num_decode_tokens=verify_batch.num_decode_tokens,
                        token_row_ids=(
                            verify_batch.token_row_ids
                            if verify_batch.packed_prefill
                            else None
                        ),
                        max_query_len=2 if verify_batch.packed_prefill else None,
                    )
                    verify_kv_state = KVCacheState(
                        k_cache=current_k_cache,
                        v_cache=current_v_cache,
                        block_table=verify_batch.block_tables,
                        kv_lens=verify_batch.seq_lens,
                        slot_mapping=verify_metadata.slot_mapping,
                    )
                    (
                        hidden,
                        updated_kv_state,
                        updated_hybrid_state,
                        first_prefix_hybrid_state,
                    ) = model_forward_step(
                        verify_batch.tokens,
                        params,
                        self.config,
                        positions=verify_batch.positions,
                        kv_cache_state=verify_kv_state,
                        attention_metadata=verify_metadata,
                        hybrid_state=HybridLayerState(current_conv_state, current_recurrent_state),
                        is_prefill=not one_pass_decode_mode,
                        return_hidden=True,
                        return_first_prefix_hybrid=True,
                        backend=self.backend,
                    )
                    if not one_pass_decode_mode:
                        hidden = hidden.reshape((tokens.shape[0], verify_tokens.shape[1], hidden.shape[-1]))

                    hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    token_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                        hidden_norm,
                        params,
                        self.config,
                        hidden_is_normed=True,
                        is_prefill=not one_pass_decode_mode,
                        top_k=2 if bonus_margin_threshold > 0 else 0,
                    )
                    target_token = token_ids[:, 0]
                    bonus_token = token_ids[:, 1]
                    accepted = (target_token == current_draft_token) & row_has_draft
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))
                    if bonus_margin_threshold > 0:
                        bonus_top2 = bonus_topk_values[:, 1]
                        bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                        accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                        if batch_accept_policy == "all_or_none":
                            accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))

                    pos_current = verify_positions[:, 0]
                    pos_next_after_reject = pos_current + jnp.asarray(1, dtype=pos_current.dtype)
                    pos_next_after_accept = pos_current + jnp.asarray(2, dtype=pos_current.dtype)
                    hidden_for_mtp = hidden_norm if mtp_hidden_final_normed else hidden
                    selected_mtp_hidden = jnp.where(
                        accepted[:, None, None],
                        hidden_for_mtp[:, 1:2, :],
                        hidden_for_mtp[:, 0:1, :],
                    )
                    selected_mtp_token = jnp.where(accepted, bonus_token, target_token)
                    selected_mtp_position = jnp.where(
                        accepted,
                        pos_next_after_accept,
                        pos_next_after_reject,
                    )

                    def run_next_mtp(_):
                        next_mtp_token_ids, _ = mtp_forward_token_ids(
                            hidden_state=selected_mtp_hidden,
                            next_token_ids=selected_mtp_token[:, None],
                            embed_tokens=params.embed_tokens,
                            params=params.mtp_params,
                            config=self.config,
                            positions=selected_mtp_position[:, None],
                        )
                        return next_mtp_token_ids[:, 0].astype(jnp.int32)

                    def skip_next_mtp(_):
                        return jnp.full_like(target_token, -1)

                    next_draft_token = jax.lax.cond(
                        jnp.any(current_active),
                        run_next_mtp,
                        skip_next_mtp,
                        operand=None,
                    )
                    next_draft_token = jnp.where(
                        current_active,
                        next_draft_token,
                        jnp.full_like(next_draft_token, -1),
                    )

                    valid_mask = current_active.reshape(
                        (current_active.shape[0],) + (1,) * (current_conv_state.ndim - 1)
                    )
                    accept_conv_mask = accepted.reshape((accepted.shape[0],) + (1,) * (current_conv_state.ndim - 1))
                    selected_conv = jnp.where(
                        valid_mask,
                        jnp.where(
                            accept_conv_mask,
                            updated_hybrid_state.conv_state,
                            first_prefix_hybrid_state.conv_state,
                        ),
                        current_conv_state,
                    )
                    recurrent_valid_mask = current_active.reshape(
                        (current_active.shape[0],) + (1,) * (current_recurrent_state.ndim - 1)
                    )
                    accept_recurrent_mask = accepted.reshape(
                        (accepted.shape[0],) + (1,) * (current_recurrent_state.ndim - 1)
                    )
                    selected_recurrent = jnp.where(
                        recurrent_valid_mask,
                        jnp.where(
                            accept_recurrent_mask,
                            updated_hybrid_state.recurrent_state,
                            first_prefix_hybrid_state.recurrent_state,
                        ),
                        current_recurrent_state,
                    )

                    emitted_counts = jnp.where(
                        current_active,
                        jnp.asarray(1, dtype=jnp.int32) + accepted.astype(jnp.int32),
                        jnp.zeros_like(accepted.astype(jnp.int32)),
                    )
                    next_seq_lens = current_seq_lens + emitted_counts
                    # A rejected row has only committed the target token. Stop
                    # it inside this burst and resume from the next scheduler
                    # step so rowwise mixed batches keep the same boundary as
                    # canonical one-token decode.
                    next_active = current_active & accepted & (next_draft_token >= 0)
                    emitted_tokens = jnp.stack(
                        [
                            jnp.where(current_active, target_token, jnp.zeros_like(target_token)),
                            jnp.where(accepted, bonus_token, jnp.zeros_like(bonus_token)),
                        ],
                        axis=1,
                    ).astype(jnp.int32)
                    accepted_counts = accepted.astype(jnp.int32)
                    next_carry = (
                        selected_mtp_token[:, None].astype(current_tokens.dtype),
                        selected_mtp_position[:, None].astype(current_positions.dtype),
                        next_seq_lens,
                        updated_kv_state.k_cache,
                        updated_kv_state.v_cache,
                        selected_conv.astype(current_conv_state.dtype),
                        selected_recurrent.astype(current_recurrent_state.dtype),
                        next_draft_token,
                        next_active,
                    )
                    return next_carry, (emitted_tokens, emitted_counts, accepted_counts)

                initial_carry = (
                    seed_target_token[:, None].astype(tokens.dtype),
                    seq_lens.astype(jnp.int32)[:, None],
                    seq_lens.astype(jnp.int32) + row_valid.astype(jnp.int32),
                    seed_updated_kv.k_cache,
                    seed_updated_kv.v_cache,
                    seed_updated_hybrid.conv_state,
                    seed_updated_hybrid.recurrent_state,
                    initial_draft_token,
                    row_valid,
                )
                final_carry, scan_outputs = jax.lax.scan(
                    run_group,
                    initial_carry,
                    jnp.arange(static_burst_groups, dtype=jnp.int32),
                )
                (
                    _final_tokens,
                    _final_positions,
                    final_seq_lens,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                    final_draft_token,
                    _final_active,
                ) = final_carry
                emitted_tokens_by_group, emitted_counts_by_group, accepted_counts_by_group = scan_outputs
                verifier_emitted_tokens = jnp.transpose(emitted_tokens_by_group, (1, 0, 2)).reshape(
                    (tokens.shape[0], static_burst_groups * 2)
                )
                emitted_counts = jnp.transpose(emitted_counts_by_group, (1, 0)).astype(jnp.int32)
                accepted_counts = jnp.transpose(accepted_counts_by_group, (1, 0)).astype(jnp.int32)
                group_width = 2
                compact_width = 1 + static_burst_groups * group_width
                starts = (
                    jnp.asarray(1, dtype=jnp.int32)
                    + jnp.cumsum(emitted_counts, axis=1)
                    - emitted_counts
                )
                offsets = jnp.arange(group_width, dtype=jnp.int32)
                valid_offsets = offsets[None, None, :] < emitted_counts[:, :, None]
                dest = starts[:, :, None] + offsets[None, None, :]
                dest = jnp.where(valid_offsets, dest, compact_width)
                batch_idx = jnp.broadcast_to(
                    jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None, None],
                    (tokens.shape[0], static_burst_groups, group_width),
                )
                compact_tokens = jnp.zeros(
                    (tokens.shape[0], compact_width),
                    dtype=jnp.int32,
                )
                compact_tokens = compact_tokens.at[
                    batch_idx.reshape(-1),
                    dest.reshape(-1),
                ].set(
                    verifier_emitted_tokens.reshape(-1),
                    mode="drop",
                )
                compact_tokens = compact_tokens.at[:, 0].set(
                    jnp.where(row_valid, seed_target_token, jnp.zeros_like(seed_target_token))
                )
                emitted_tokens = compact_tokens
                accepted_bitmask = jnp.sum(
                    accepted_counts.astype(jnp.int32)
                    * (
                        jnp.asarray(1, dtype=jnp.int32)
                        << jnp.arange(static_burst_groups, dtype=jnp.int32)
                    )[None, :],
                    axis=1,
                ).astype(jnp.int32)
                accepted_totals = jnp.sum(accepted_counts, axis=1).astype(jnp.int32)
                emitted_totals = (
                    row_valid.astype(jnp.int32) + jnp.sum(emitted_counts, axis=1).astype(jnp.int32)
                )
                rejected_totals = jnp.sum(
                    jnp.where(row_valid[:, None], 1 - accepted_counts, 0),
                    axis=1,
                ).astype(jnp.int32)
                bonus_totals = accepted_totals
                compact_summary = jnp.stack(
                    [
                        emitted_totals,
                        accepted_totals,
                        rejected_totals,
                        bonus_totals,
                        accepted_bitmask,
                    ],
                    axis=1,
                ).astype(jnp.int32)
                scatter_slot_ids = jnp.where(
                    row_valid,
                    slot_ids,
                    jnp.full_like(slot_ids, conv_state_table.shape[0]),
                )
                updated_conv_table = conv_state_table.at[scatter_slot_ids].set(
                    final_conv_state.astype(conv_state_table.dtype),
                    mode="drop",
                )
                updated_recurrent_table = recurrent_state_table.at[scatter_slot_ids].set(
                    final_recurrent_state.astype(recurrent_state_table.dtype),
                    mode="drop",
                )
                return (
                    seed_target_token,
                    emitted_tokens[:, 1],
                    final_draft_token,
                    accepted_counts[:, 0].astype(bool),
                    final_k_cache,
                    final_v_cache,
                    updated_conv_table,
                    updated_recurrent_table,
                    final_seq_lens,
                    emitted_tokens,
                    emitted_counts,
                    accepted_counts,
                    emitted_totals,
                    accepted_totals,
                    rejected_totals,
                    bonus_totals,
                    accepted_bitmask,
                    compact_summary,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(7, 8, 9, 10))

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            emitted_tokens,
            emitted_counts,
            accepted_counts,
            emitted_totals,
            accepted_totals,
            rejected_totals,
            bonus_totals,
            accepted_bitmask,
            compact_summary,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
            ),
            "mtp1_seed_then_table_burst_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            emitted_tokens=emitted_tokens,
            emitted_counts=emitted_counts,
            accepted_counts=accepted_counts,
            emitted_totals=emitted_totals,
            accepted_totals=accepted_totals,
            rejected_totals=rejected_totals,
            bonus_totals=bonus_totals,
            accepted_bitmask=accepted_bitmask,
            compact_summary=compact_summary,
            burst_groups=burst_groups,
            hybrid_state_is_table=True,
        )

    def mtp1_layer_parity_debug_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1ParityDebugOutput:
        """Debug-only K=1 parity probe for fused one-pass vs sequential decode.

        This intentionally returns small metrics only. It does not donate input
        cache buffers because it must run from the same pre-state that the
        production one-pass call will use.
        """
        del next_mtp_position, mtp_hidden_final_normed
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp1_layer_parity_debug_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_layer_parity_debug_jit expects a decode batch")
        self._validate_batch_contract(batch)

        key = (
            "mtp1-layer-parity-debug",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1"),
        )
        if key not in self._jit_cache:
            one_pass_decode_mode = os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def _max_abs(left, right, row_mask):
                diff = jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
                mask = row_mask.reshape((row_mask.shape[0],) + (1,) * (diff.ndim - 1))
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)))

            def _slot_max_abs(left, right, slots, row_mask):
                leading_shape = left.shape[:-4] if left.ndim == 5 else left.shape[:-3]
                flat_left = left.reshape(leading_shape + (-1,) + left.shape[-2:])
                flat_right = right.reshape(leading_shape + (-1,) + right.shape[-2:])
                left_values = flat_left[..., slots, :, :].astype(jnp.float32)
                right_values = flat_right[..., slots, :, :].astype(jnp.float32)
                diff = jnp.abs(left_values - right_values)
                mask = row_mask.reshape((1,) * len(leading_shape) + (row_mask.shape[0], 1, 1))
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)))

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_token_arg,
            ):
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_active = (row_query_lens > 0) & (seq_ids >= 0)
                row_has_draft = row_active & (draft_token_arg >= 0)

                verify_tokens = jnp.concatenate([tokens, draft_token_arg[:, None]], axis=1)
                verify_positions = jnp.concatenate([positions, positions + 1], axis=1)
                verify_query_lens = row_query_lens + row_has_draft.astype(jnp.int32)
                verify_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(verify_query_lens),
                    ]
                )
                verify_batch = ScheduledBatch(
                    tokens=verify_tokens,
                    positions=verify_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=not one_pass_decode_mode,
                    num_prefill_tokens=0 if one_pass_decode_mode else jnp.sum(verify_query_lens),
                    num_decode_tokens=jnp.sum(verify_query_lens) if one_pass_decode_mode else 0,
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_has_draft.astype(jnp.int32),
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=not one_pass_decode_mode,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=verify_batch.num_prefill_tokens,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                (fused_hidden, fused_logits), fused_kv, fused_hybrid = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=not one_pass_decode_mode,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    backend=self.backend,
                )

                first_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                first_metadata = self.backend.build_attention_metadata(
                    positions=first_batch.positions,
                    block_tables=first_batch.block_tables,
                    seq_lens=first_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=first_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=first_batch.num_decode_tokens,
                )
                first_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=first_batch.block_tables,
                    kv_lens=first_batch.seq_lens,
                    slot_mapping=first_metadata.slot_mapping,
                )
                seq_hidden0, seq_kv0, seq_hybrid0 = model_forward_step(
                    first_batch.tokens,
                    params,
                    self.config,
                    positions=first_batch.positions,
                    kv_cache_state=first_kv_state,
                    attention_metadata=first_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    backend=self.backend,
                )
                output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
                seq_hidden0_norm = rms_norm(seq_hidden0, params.norm_weight, self.config.rms_norm_eps).astype(
                    jnp.float32
                )
                seq_logits0 = jnp.dot(seq_hidden0_norm[:, 0], output_weight)

                second_query_lens = row_has_draft.astype(jnp.int32)
                second_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(second_query_lens),
                    ]
                )
                second_batch = ScheduledBatch(
                    tokens=jnp.where(row_has_draft[:, None], draft_token_arg[:, None], jnp.zeros_like(tokens)),
                    positions=jnp.where(row_has_draft[:, None], positions + 1, jnp.zeros_like(positions)),
                    seq_ids=jnp.where(row_has_draft, seq_ids, jnp.full_like(seq_ids, -1)),
                    query_start_loc=second_query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.sum(second_query_lens),
                    block_tables=jnp.where(row_has_draft[:, None], block_tables, jnp.zeros_like(block_tables)),
                    seq_lens=jnp.where(row_has_draft, seq_lens + 1, jnp.zeros_like(seq_lens)),
                )
                second_metadata = self.backend.build_attention_metadata(
                    positions=second_batch.positions,
                    block_tables=second_batch.block_tables,
                    seq_lens=second_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=second_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=second_batch.num_decode_tokens,
                )
                second_kv_state = KVCacheState(
                    k_cache=seq_kv0.k_cache,
                    v_cache=seq_kv0.v_cache,
                    block_table=second_batch.block_tables,
                    kv_lens=second_batch.seq_lens,
                    slot_mapping=second_metadata.slot_mapping,
                )
                (seq_hidden1, seq_logits1), seq_kv1, seq_hybrid1 = model_forward_step(
                    second_batch.tokens,
                    params,
                    self.config,
                    positions=second_batch.positions,
                    kv_cache_state=second_kv_state,
                    attention_metadata=second_metadata,
                    hybrid_state=seq_hybrid0,
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    backend=self.backend,
                )

                fused_top5_slot0 = jax.lax.top_k(fused_logits[:, 0].astype(jnp.float32), 5)[1].astype(jnp.int32)
                seq_top5_slot0 = jax.lax.top_k(seq_logits0.astype(jnp.float32), 5)[1].astype(jnp.int32)
                fused_top5_slot1 = jax.lax.top_k(fused_logits[:, 1].astype(jnp.float32), 5)[1].astype(jnp.int32)
                seq_top5_slot1 = jax.lax.top_k(seq_logits1[:, 0].astype(jnp.float32), 5)[1].astype(jnp.int32)

                current_slots = verify_metadata.slot_mapping[:, 0]
                draft_slots = verify_metadata.slot_mapping[:, 1]
                return (
                    _max_abs(fused_logits[:, 0], seq_logits0, row_active),
                    _max_abs(fused_logits[:, 1], seq_logits1[:, 0], row_has_draft),
                    _max_abs(fused_hidden[:, 0], seq_hidden0[:, 0], row_active),
                    _max_abs(fused_hidden[:, 1], seq_hidden1[:, 0], row_has_draft),
                    _slot_max_abs(fused_kv.k_cache, seq_kv1.k_cache, current_slots, row_active),
                    _slot_max_abs(fused_kv.k_cache, seq_kv1.k_cache, draft_slots, row_has_draft),
                    _slot_max_abs(fused_kv.v_cache, seq_kv1.v_cache, current_slots, row_active),
                    _slot_max_abs(fused_kv.v_cache, seq_kv1.v_cache, draft_slots, row_has_draft),
                    _max_abs(fused_hybrid.conv_state, seq_hybrid1.conv_state, row_active),
                    _max_abs(fused_hybrid.recurrent_state, seq_hybrid1.recurrent_state, row_active),
                    fused_top5_slot0,
                    seq_top5_slot0,
                    fused_top5_slot1,
                    seq_top5_slot1,
                    jnp.argmax(fused_logits[:, 0], axis=-1).astype(jnp.int32),
                    jnp.argmax(seq_logits0, axis=-1).astype(jnp.int32),
                    jnp.argmax(fused_logits[:, 1], axis=-1).astype(jnp.int32),
                    jnp.argmax(seq_logits1[:, 0], axis=-1).astype(jnp.int32),
                )

            self._jit_cache[key] = jax.jit(compiled)

        outputs = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_layer_parity_debug_jit",
        )
        return MTP1ParityDebugOutput(*outputs)

    def mtp1_layerwise_drift_debug_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_token: jnp.ndarray,
    ) -> MTP1LayerwiseDriftDebugOutput:
        """Debug-only per-layer drift probe for fused width-2 vs sequential width-1 decode."""
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp1_layerwise_drift_debug_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_layerwise_drift_debug_jit expects a decode batch")
        self._validate_batch_contract(batch)

        key = (
            "mtp1-layerwise-drift-debug",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1"),
        )
        if key not in self._jit_cache:
            one_pass_decode_mode = os.environ.get("NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            layer_types = tuple(self.config.layer_types)

            def _layer_hidden_max_abs(fused_layers, seq_layers, row_mask):
                diff = jnp.abs(fused_layers[:, :, 0, :].astype(jnp.float32) - seq_layers[:, :, 0, :].astype(jnp.float32))
                mask = row_mask[None, :, None]
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)), axis=(1, 2))

            def _kv_slot_layer_max_abs(left, right, slots, row_mask):
                flat_left = left.reshape((left.shape[0], -1) + left.shape[-2:])
                flat_right = right.reshape((right.shape[0], -1) + right.shape[-2:])
                diff = jnp.abs(
                    flat_left[:, slots, :, :].astype(jnp.float32)
                    - flat_right[:, slots, :, :].astype(jnp.float32)
                )
                mask = row_mask[None, :, None, None]
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)), axis=(1, 2, 3))

            def _hybrid_layer_max_abs(left, right, linear_idx, row_mask):
                diff = jnp.abs(
                    left[:, linear_idx].astype(jnp.float32)
                    - right[:, linear_idx].astype(jnp.float32)
                )
                mask = row_mask.reshape((row_mask.shape[0],) + (1,) * (diff.ndim - 1))
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)))

            def _prewrite_layer_max_abs(left, right, row_mask):
                diff = jnp.abs(
                    left[:, :, 0, :, :].astype(jnp.float32)
                    - right[:, :, 0, :, :].astype(jnp.float32)
                )
                mask = row_mask[None, :, None, None]
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)), axis=(1, 2, 3))

            def _block_stage_max_abs(fused_stages, seq_stages, row_mask):
                # [L, S, B, T, D] -> [L, S] for slot0/current token.
                diff = jnp.abs(
                    fused_stages[:, :, :, 0, :].astype(jnp.float32)
                    - seq_stages[:, :, :, 0, :].astype(jnp.float32)
                )
                mask = row_mask[None, None, :, None]
                return jnp.max(jnp.where(mask, diff, jnp.zeros_like(diff)), axis=(2, 3))

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_token_arg,
            ):
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_active = (row_query_lens > 0) & (seq_ids >= 0)
                row_has_draft = row_active & (draft_token_arg >= 0)

                verify_tokens = jnp.concatenate([tokens, draft_token_arg[:, None]], axis=1)
                verify_positions = jnp.concatenate([positions, positions + 1], axis=1)
                verify_query_lens = row_query_lens + row_has_draft.astype(jnp.int32)
                verify_query_start_loc = jnp.concatenate(
                    [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(verify_query_lens)]
                )
                verify_batch = ScheduledBatch(
                    tokens=verify_tokens,
                    positions=verify_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=not one_pass_decode_mode,
                    num_prefill_tokens=0 if one_pass_decode_mode else jnp.sum(verify_query_lens),
                    num_decode_tokens=jnp.sum(verify_query_lens) if one_pass_decode_mode else 0,
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_has_draft.astype(jnp.int32),
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=not one_pass_decode_mode,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=verify_batch.num_prefill_tokens,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                (
                    (_fused_hidden, _fused_logits),
                    fused_kv,
                    fused_hybrid,
                    fused_layers,
                    fused_prewrite_k,
                    fused_prewrite_v,
                    fused_stages,
                ) = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=not one_pass_decode_mode,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    return_layer_hidden=True,
                    return_kv_prewrite=True,
                    return_layer_stages=True,
                    backend=self.backend,
                )

                first_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                first_metadata = self.backend.build_attention_metadata(
                    positions=first_batch.positions,
                    block_tables=first_batch.block_tables,
                    seq_lens=first_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=first_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=first_batch.num_decode_tokens,
                )
                first_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=first_batch.block_tables,
                    kv_lens=first_batch.seq_lens,
                    slot_mapping=first_metadata.slot_mapping,
                )
                (
                    _seq_hidden0,
                    seq_kv0,
                    seq_hybrid0,
                    seq_layers0,
                    seq_prewrite_k0,
                    seq_prewrite_v0,
                    seq_stages0,
                ) = model_forward_step(
                    first_batch.tokens,
                    params,
                    self.config,
                    positions=first_batch.positions,
                    kv_cache_state=first_kv_state,
                    attention_metadata=first_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_layer_hidden=True,
                    return_kv_prewrite=True,
                    return_layer_stages=True,
                    backend=self.backend,
                )

                second_query_lens = row_has_draft.astype(jnp.int32)
                second_query_start_loc = jnp.concatenate(
                    [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(second_query_lens)]
                )
                second_batch = ScheduledBatch(
                    tokens=jnp.where(row_has_draft[:, None], draft_token_arg[:, None], jnp.zeros_like(tokens)),
                    positions=jnp.where(row_has_draft[:, None], positions + 1, jnp.zeros_like(positions)),
                    seq_ids=jnp.where(row_has_draft, seq_ids, jnp.full_like(seq_ids, -1)),
                    query_start_loc=second_query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.sum(second_query_lens),
                    block_tables=jnp.where(row_has_draft[:, None], block_tables, jnp.zeros_like(block_tables)),
                    seq_lens=jnp.where(row_has_draft, seq_lens + 1, jnp.zeros_like(seq_lens)),
                )
                second_metadata = self.backend.build_attention_metadata(
                    positions=second_batch.positions,
                    block_tables=second_batch.block_tables,
                    seq_lens=second_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=second_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=second_batch.num_decode_tokens,
                )
                second_kv_state = KVCacheState(
                    k_cache=seq_kv0.k_cache,
                    v_cache=seq_kv0.v_cache,
                    block_table=second_batch.block_tables,
                    kv_lens=second_batch.seq_lens,
                    slot_mapping=second_metadata.slot_mapping,
                )
                _seq_hidden1, seq_kv1, seq_hybrid1, _seq_layers1 = model_forward_step(
                    second_batch.tokens,
                    params,
                    self.config,
                    positions=second_batch.positions,
                    kv_cache_state=second_kv_state,
                    attention_metadata=second_metadata,
                    hybrid_state=seq_hybrid0,
                    is_prefill=False,
                    return_hidden=True,
                    return_layer_hidden=True,
                    backend=self.backend,
                )

                hidden_max_abs = _layer_hidden_max_abs(fused_layers, seq_layers0, row_active)
                current_slots = verify_metadata.slot_mapping[:, 0]
                k_slot_all = _kv_slot_layer_max_abs(fused_kv.k_cache, seq_kv1.k_cache, current_slots, row_active)
                v_slot_all = _kv_slot_layer_max_abs(fused_kv.v_cache, seq_kv1.v_cache, current_slots, row_active)

                prewrite_k_all = _prewrite_layer_max_abs(fused_prewrite_k, seq_prewrite_k0, row_active)
                prewrite_v_all = _prewrite_layer_max_abs(fused_prewrite_v, seq_prewrite_v0, row_active)
                block_stage_all = _block_stage_max_abs(fused_stages, seq_stages0, row_active)

                k_values = []
                v_values = []
                conv_values = []
                recurrent_values = []
                prewrite_k_values = []
                prewrite_v_values = []
                linear_idx = 0
                for layer_idx, layer_type in enumerate(layer_types):
                    if layer_type == "full_attention":
                        k_values.append(k_slot_all[layer_idx])
                        v_values.append(v_slot_all[layer_idx])
                        conv_values.append(jnp.asarray(0.0, dtype=jnp.float32))
                        recurrent_values.append(jnp.asarray(0.0, dtype=jnp.float32))
                        prewrite_k_values.append(prewrite_k_all[layer_idx])
                        prewrite_v_values.append(prewrite_v_all[layer_idx])
                    else:
                        k_values.append(jnp.asarray(0.0, dtype=jnp.float32))
                        v_values.append(jnp.asarray(0.0, dtype=jnp.float32))
                        prewrite_k_values.append(jnp.asarray(0.0, dtype=jnp.float32))
                        prewrite_v_values.append(jnp.asarray(0.0, dtype=jnp.float32))
                        conv_values.append(
                            _hybrid_layer_max_abs(
                                fused_hybrid.conv_state,
                                seq_hybrid1.conv_state,
                                linear_idx,
                                row_active,
                            )
                        )
                        recurrent_values.append(
                            _hybrid_layer_max_abs(
                                fused_hybrid.recurrent_state,
                                seq_hybrid1.recurrent_state,
                                linear_idx,
                                row_active,
                            )
                        )
                        linear_idx += 1

                return (
                    hidden_max_abs,
                    jnp.stack(k_values),
                    jnp.stack(v_values),
                    jnp.stack(conv_values),
                    jnp.stack(recurrent_values),
                    jnp.stack(prewrite_k_values),
                    jnp.stack(prewrite_v_values),
                    block_stage_all,
                )

            self._jit_cache[key] = jax.jit(compiled)

        outputs = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_layerwise_drift_debug_jit",
        )
        return MTP1LayerwiseDriftDebugOutput(*outputs)

    def mtp1_two_decode_greedy_fast_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """Fast K=1 greedy verifier for the common all-accepted case.

        This avoids extracting token-0 prefix hybrid state. The runner may only
        commit this output when every row accepts; otherwise it must discard it
        and use the safe repair path.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp1_two_decode_greedy_fast_step_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_two_decode_greedy_fast_step_jit expects a decode batch")
        self._log_step("mtp1_two_decode_greedy_fast_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-two-token-fast-all-accept-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_token_arg,
                next_mtp_position_arg,
            ):
                verify_tokens = jnp.concatenate([tokens, draft_token_arg[:, None]], axis=1)
                verify_positions = jnp.concatenate([positions, positions + 1], axis=1)
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = row_query_lens > 0
                row_has_draft = row_valid & (draft_token_arg >= 0)
                verify_query_lens = row_query_lens + row_has_draft.astype(jnp.int32)
                verify_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(verify_query_lens),
                    ]
                )
                verify_batch = ScheduledBatch(
                    tokens=verify_tokens,
                    positions=verify_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.sum(verify_query_lens),
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_has_draft.astype(jnp.int32),
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    backend=self.backend,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                    hidden_norm,
                    params,
                    self.config,
                    hidden_is_normed=True,
                    is_prefill=False,
                    top_k=2 if bonus_margin_threshold > 0 else 0,
                )
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = (target_token == draft_token_arg) & row_has_draft
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))
                if bonus_margin_threshold > 0:
                    bonus_top2 = bonus_topk_values[:, 1]
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))
                pos_current = verify_positions[:, 0]
                pos_next_after_reject = pos_current + jnp.asarray(1, dtype=pos_current.dtype)
                pos_next_after_accept = pos_current + jnp.asarray(2, dtype=pos_current.dtype)
                hidden_for_mtp = hidden_norm if mtp_hidden_final_normed else hidden
                selected_mtp_hidden = jnp.where(
                    accepted[:, None, None],
                    hidden_for_mtp[:, 1:2, :],
                    hidden_for_mtp[:, 0:1, :],
                )
                selected_mtp_token = jnp.where(accepted, bonus_token, target_token)
                selected_mtp_position = jnp.where(
                    accepted,
                    pos_next_after_accept,
                    pos_next_after_reject,
                )
                def run_next_mtp(_):
                    mtp_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=selected_mtp_hidden,
                        next_token_ids=selected_mtp_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=selected_mtp_position[:, None],
                    )
                    return mtp_token_ids[:, 0].astype(jnp.int32)

                def skip_next_mtp(_):
                    return jnp.full_like(target_token, -1)

                next_draft_token = jax.lax.cond(
                    jnp.any(accepted),
                    run_next_mtp,
                    skip_next_mtp,
                    operand=None,
                )
                output_accepted = jnp.where(row_valid, accepted, True)
                verify_slots = verify_metadata.slot_mapping[:, :2].reshape((-1,))
                slot_accept = jnp.repeat(accepted, 2)

                def restore_rejected_slots(new_cache, old_cache):
                    leading_shape = new_cache.shape[:-4] if new_cache.ndim == 5 else new_cache.shape[:-3]
                    flat_new = new_cache.reshape(leading_shape + (-1,) + new_cache.shape[-2:])
                    flat_old = old_cache.reshape(leading_shape + (-1,) + old_cache.shape[-2:])
                    new_values = flat_new[..., verify_slots, :, :]
                    old_values = flat_old[..., verify_slots, :, :]
                    slot_mask = slot_accept.reshape((1,) * len(leading_shape) + (slot_accept.shape[0], 1, 1))
                    selected_values = jnp.where(slot_mask, new_values, old_values)
                    flat_new = flat_new.at[..., verify_slots, :, :].set(selected_values)
                    return flat_new.reshape(new_cache.shape)

                selected_k_cache = restore_rejected_slots(updated_kv_state.k_cache, k_cache)
                selected_v_cache = restore_rejected_slots(updated_kv_state.v_cache, v_cache)
                conv_mask = accepted.reshape((accepted.shape[0],) + (1,) * (updated_hybrid_state.conv_state.ndim - 1))
                recurrent_mask = accepted.reshape(
                    (accepted.shape[0],) + (1,) * (updated_hybrid_state.recurrent_state.ndim - 1)
                )
                selected_conv = jnp.where(conv_mask, updated_hybrid_state.conv_state, conv_state)
                selected_recurrent = jnp.where(recurrent_mask, updated_hybrid_state.recurrent_state, recurrent_state)
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    output_accepted,
                    selected_k_cache,
                    selected_v_cache,
                    selected_conv,
                    selected_recurrent,
                    seq_lens + accepted.astype(jnp.int32),
                )

            self._jit_cache[key] = jax.jit(compiled)

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_token, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp1_two_decode_greedy_fast_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
        )

    def mtp1_two_decode_greedy_fast_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """Fast K=1 greedy verifier that gathers hybrid state inside JIT.

        This keeps the hot speculative path aligned with the normal table-based
        decode route. The runner may commit this output only when every row
        accepts; otherwise the tentative cache and compact hybrid state are
        discarded.
        """
        del next_mtp_position
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("mtp1_two_decode_greedy_fast_table_jit requires initialized hybrid_state_table")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_two_decode_greedy_fast_table_jit expects a decode batch")
        self._log_step("mtp1_two_decode_greedy_fast_table_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-two-token-fast-table-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                draft_token_arg,
            ):
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (row_query_lens > 0) & (seq_ids >= 0) & (slot_ids >= 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )

                verify_tokens = jnp.concatenate([tokens, draft_token_arg[:, None]], axis=1)
                verify_positions = jnp.concatenate([positions, positions + 1], axis=1)
                row_has_draft = row_valid & (draft_token_arg >= 0)
                verify_query_lens = row_query_lens + row_has_draft.astype(jnp.int32)
                verify_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(verify_query_lens),
                    ]
                )
                verify_batch = ScheduledBatch(
                    tokens=verify_tokens,
                    positions=verify_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.sum(verify_query_lens),
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_has_draft.astype(jnp.int32),
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    backend=self.backend,
                    hybrid_state_layerwise=True,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                    hidden_norm,
                    params,
                    self.config,
                    hidden_is_normed=True,
                    is_prefill=False,
                    top_k=2 if bonus_margin_threshold > 0 else 0,
                )
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = (target_token == draft_token_arg) & row_has_draft
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))
                if bonus_margin_threshold > 0:
                    bonus_top2 = bonus_topk_values[:, 1]
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))

                pos_current = verify_positions[:, 0]
                pos_next_after_reject = pos_current + jnp.asarray(1, dtype=pos_current.dtype)
                pos_next_after_accept = pos_current + jnp.asarray(2, dtype=pos_current.dtype)
                hidden_for_mtp = hidden_norm if mtp_hidden_final_normed else hidden
                selected_mtp_hidden = jnp.where(
                    accepted[:, None, None],
                    hidden_for_mtp[:, 1:2, :],
                    hidden_for_mtp[:, 0:1, :],
                )
                selected_mtp_token = jnp.where(accepted, bonus_token, target_token)
                selected_mtp_position = jnp.where(
                    accepted,
                    pos_next_after_accept,
                    pos_next_after_reject,
                )

                def run_next_mtp(_):
                    mtp_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=selected_mtp_hidden,
                        next_token_ids=selected_mtp_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=selected_mtp_position[:, None],
                    )
                    return mtp_token_ids[:, 0].astype(jnp.int32)

                def skip_next_mtp(_):
                    return jnp.full_like(target_token, -1)

                next_draft_token = jax.lax.cond(
                    compute_next_draft & jnp.any(accepted),
                    run_next_mtp,
                    skip_next_mtp,
                    operand=None,
                )
                output_accepted = jnp.where(row_valid, accepted, True)
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    output_accepted,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                    seq_lens + accepted.astype(jnp.int32),
                )

            self._jit_cache[key] = jax.jit(compiled)

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_two_decode_greedy_fast_table_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
        )

    def mtp1_burst_verify_table_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state_table: HybridLayerState,
        hybrid_slot_ids: jnp.ndarray,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """K=1 verifier shaped like the greedy burst table path.

        The graph always runs two width-1 target-model decode steps:
        current-token verification, then draft-token bonus generation.  On
        rejection the second step's KV slot is left dirty but uncommitted and
        will be overwritten by the next real decode; recurrent state is
        selected back to the first-step prefix.
        """
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("mtp1_burst_verify_table_jit requires initialized hybrid_state_table")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_burst_verify_table_jit expects a decode batch")
        self._log_step("mtp1_burst_verify_table_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-burst-verify-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(hybrid_state_table.conv_state.shape),
            tuple(hybrid_state_table.recurrent_state.shape),
            bool(mtp_hidden_final_normed),
            os.environ.get("NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0"),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            assume_all_accept = os.environ.get(
                "NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0"
            ) in {"1", "true", "yes", "on", "True"}
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def compiled(
                params_leaves,
                tokens,
                positions,
                query_start_loc,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state_table,
                recurrent_state_table,
                slot_ids,
                draft_token_arg,
            ):
                params = jax.tree_util.tree_unflatten(self._params_treedef, params_leaves)
                query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_valid = (slot_ids >= 0) & (query_lens > 0) & (draft_token_arg >= 0)
                safe_slot_ids = jnp.maximum(slot_ids, 0)
                conv_state = conv_state_table[safe_slot_ids]
                recurrent_state = recurrent_state_table[safe_slot_ids]
                conv_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                    conv_state,
                    jnp.zeros_like(conv_state),
                )
                recurrent_state = jnp.where(
                    row_valid.reshape((row_valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                    recurrent_state,
                    jnp.zeros_like(recurrent_state),
                )
                seq_ids = jnp.where(
                    row_valid,
                    safe_slot_ids.astype(jnp.int32),
                    jnp.full_like(safe_slot_ids, -1),
                )
                num_query_tokens = query_start_loc[-1].astype(jnp.int32)

                def build_step_outputs(include_hidden: bool):
                    def step(carry, step_i):
                        (
                            step_tokens,
                            step_positions,
                            step_seq_lens,
                            step_k_cache,
                            step_v_cache,
                            step_conv_state,
                            step_recurrent_state,
                        ) = carry
                        step_batch = ScheduledBatch(
                            tokens=step_tokens,
                            positions=step_positions,
                            seq_ids=seq_ids,
                            query_start_loc=query_start_loc,
                            is_prefill=False,
                            num_prefill_tokens=0,
                            num_decode_tokens=num_query_tokens,
                            block_tables=block_tables,
                            seq_lens=step_seq_lens,
                        )
                        metadata = self.backend.build_attention_metadata(
                            positions=step_batch.positions,
                            block_tables=step_batch.block_tables,
                            seq_lens=step_batch.seq_lens,
                            block_size=self.config.block_size,
                            is_prefill=False,
                            query_start_loc=step_batch.query_start_loc,
                            num_prefill_tokens=0,
                            num_decode_tokens=step_batch.num_decode_tokens,
                        )
                        kv_state = KVCacheState(
                            k_cache=step_k_cache,
                            v_cache=step_v_cache,
                            block_table=step_batch.block_tables,
                            kv_lens=step_batch.seq_lens,
                            slot_mapping=metadata.slot_mapping,
                        )
                        hidden, updated_kv_state, updated_hybrid_state = model_forward_step(
                            step_batch.tokens,
                            params,
                            self.config,
                            positions=step_batch.positions,
                            kv_cache_state=kv_state,
                            attention_metadata=metadata,
                            hybrid_state=HybridLayerState(step_conv_state, step_recurrent_state),
                            is_prefill=False,
                            return_hidden=True,
                            return_hidden_with_logits=False,
                            last_logits_only=False,
                            backend=self.backend,
                            hybrid_state_layerwise=True,
                        )
                        token_ids, _, _ = lm_head_token_ids_and_topk(
                            hidden[:, :1, :],
                            params,
                            self.config,
                            hidden_is_normed=False,
                            is_prefill=False,
                            top_k=0,
                        )
                        token_ids = token_ids[:, 0].astype(jnp.int32)
                        draft_tokens = jnp.where(
                            row_valid[:, None],
                            draft_token_arg[:, None],
                            jnp.zeros_like(tokens),
                        )
                        next_tokens = jnp.where(step_i == 0, draft_tokens, token_ids[:, None])
                        next_positions = step_positions + row_valid[:, None].astype(step_positions.dtype)
                        next_seq_lens = jnp.where(row_valid, step_seq_lens + 1, step_seq_lens)
                        next_carry = (
                            next_tokens,
                            next_positions,
                            next_seq_lens,
                            updated_kv_state.k_cache.astype(step_k_cache.dtype),
                            updated_kv_state.v_cache.astype(step_v_cache.dtype),
                            updated_hybrid_state.conv_state.astype(step_conv_state.dtype),
                            updated_hybrid_state.recurrent_state.astype(step_recurrent_state.dtype),
                        )
                        if include_hidden:
                            return next_carry, (token_ids, hidden)
                        return next_carry, token_ids

                    initial = (
                        tokens,
                        positions,
                        seq_lens,
                        k_cache,
                        v_cache,
                        conv_state,
                        recurrent_state,
                    )
                    return jax.lax.scan(
                        step,
                        initial,
                        jnp.arange(2, dtype=jnp.int32),
                    )

                if compute_next_draft:
                    final, scan_outputs = build_step_outputs(include_hidden=True)
                    token_ids_by_step, hidden_by_step = scan_outputs
                else:
                    final, token_ids_by_step = build_step_outputs(include_hidden=False)
                    hidden_by_step = None

                (
                    _tokens,
                    _positions,
                    _seq_lens,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state,
                    final_recurrent_state,
                ) = final
                target_token = token_ids_by_step[0].astype(jnp.int32)
                bonus_token = token_ids_by_step[1].astype(jnp.int32)
                accepted = (target_token == draft_token_arg) & row_valid
                committed_seq_lens = seq_lens + accepted.astype(jnp.int32)

                if compute_next_draft:
                    hidden1 = hidden_by_step[1]
                    mtp_hidden = (
                        rms_norm(hidden1, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                        if mtp_hidden_final_normed
                        else hidden1
                    )
                    mtp_positions = positions[:, 0] + jnp.asarray(2, dtype=positions.dtype)
                    mtp_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=mtp_hidden,
                        next_token_ids=bonus_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=mtp_positions[:, None],
                    )
                    next_draft_token = jnp.where(
                        accepted,
                        mtp_token_ids[:, 0].astype(jnp.int32),
                        jnp.full_like(target_token, -1),
                    )
                else:
                    next_draft_token = jnp.full_like(target_token, -1)
                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    accepted,
                    final_k_cache,
                    final_v_cache,
                    final_conv_state.astype(conv_state.dtype),
                    final_recurrent_state.astype(recurrent_state.dtype),
                    committed_seq_lens,
                )

            donate_argnums = (6, 7) if assume_all_accept else ()
            self._jit_cache[key] = jax.jit(compiled, donate_argnums=donate_argnums)

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self._params_leaves,
                batch.tokens,
                batch.positions,
                batch.query_start_loc,
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                jnp.asarray(draft_token, dtype=jnp.int32),
            ),
            "mtp1_burst_verify_table_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
        )

    def mtp1_commit_select_greedy_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_token: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """K=1 greedy MTP verifier with device-side commit selection.

        This path avoids host-side accept gating. It runs the current decode
        token and the verifier-confirmed target token as two sequential one-token target decodes,
        then selects the committed hybrid state on device:

        - accepted rows commit state after the draft token and emit draft+bonus
        - rejected rows commit state after the current token and emit target

        KV writes for rejected draft slots are harmless because those slots are
        not logically committed and will be overwritten by the next decode.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp1_commit_select_greedy_step_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp1_commit_select_greedy_step_jit expects a decode batch")
        self._log_step("mtp1_commit_select_greedy_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp1-commit-select-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", getattr(self.config, "mtp_bonus_margin", 0.0))
            )
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            compute_next_draft = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self.config, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }

            def _expand_row_mask(mask: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
                return mask.reshape((mask.shape[0],) + (1,) * (target.ndim - 1))

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_token_arg,
                next_mtp_position_arg,
            ):
                first_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                first_metadata = self.backend.build_attention_metadata(
                    positions=first_batch.positions,
                    block_tables=first_batch.block_tables,
                    seq_lens=first_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=first_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=first_batch.num_decode_tokens,
                )
                first_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=first_batch.block_tables,
                    kv_lens=first_batch.seq_lens,
                    slot_mapping=first_metadata.slot_mapping,
                )
                hidden0, kv_after_current, hybrid_after_current = model_forward_step(
                    first_batch.tokens,
                    params,
                    self.config,
                    positions=first_batch.positions,
                    kv_cache_state=first_kv_state,
                    attention_metadata=first_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    backend=self.backend,
                )

                hidden0_norm = rms_norm(hidden0, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                target_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden0_norm,
                    params,
                    self.config,
                    hidden_is_normed=True,
                    is_prefill=False,
                )
                target_token = target_ids[:, 0]
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_active = (row_query_lens > 0) & (seq_ids >= 0)
                row_has_draft = row_active & (draft_token_arg >= 0)
                accepted = (target_token == draft_token_arg) & row_has_draft
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_has_draft, accepted, True))

                def run_second_decode(_):
                    second_query_lens = accepted.astype(jnp.int32)
                    second_query_start_loc = jnp.concatenate(
                        [
                            jnp.zeros((1,), dtype=jnp.int32),
                            jnp.cumsum(second_query_lens),
                        ]
                    )
                    second_seq_lens = jnp.where(accepted, seq_lens + 1, jnp.zeros_like(seq_lens))
                    second_tokens = jnp.where(accepted[:, None], target_token[:, None], jnp.zeros_like(tokens))
                    second_positions = jnp.where(accepted[:, None], positions + 1, jnp.zeros_like(positions))
                    second_seq_ids = jnp.where(accepted, seq_ids, jnp.full_like(seq_ids, -1))
                    second_block_tables = jnp.where(
                        accepted[:, None],
                        block_tables,
                        jnp.zeros_like(block_tables),
                    )
                    second_batch = ScheduledBatch(
                        tokens=second_tokens,
                        positions=second_positions,
                        seq_ids=second_seq_ids,
                        query_start_loc=second_query_start_loc,
                        is_prefill=False,
                        num_prefill_tokens=0,
                        num_decode_tokens=jnp.sum(second_query_lens),
                        block_tables=second_block_tables,
                        seq_lens=second_seq_lens,
                    )
                    second_metadata = self.backend.build_attention_metadata(
                        positions=second_batch.positions,
                        block_tables=second_batch.block_tables,
                        seq_lens=second_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=False,
                        query_start_loc=second_batch.query_start_loc,
                        num_prefill_tokens=0,
                        num_decode_tokens=second_batch.num_decode_tokens,
                    )
                    second_kv_state = KVCacheState(
                        k_cache=kv_after_current.k_cache,
                        v_cache=kv_after_current.v_cache,
                        block_table=second_batch.block_tables,
                        kv_lens=second_batch.seq_lens,
                        slot_mapping=second_metadata.slot_mapping,
                    )
                    next_hidden, next_kv, next_hybrid = model_forward_step(
                        second_batch.tokens,
                        params,
                        self.config,
                        positions=second_batch.positions,
                        kv_cache_state=second_kv_state,
                        attention_metadata=second_metadata,
                        hybrid_state=hybrid_after_current,
                        is_prefill=False,
                        return_hidden=True,
                        backend=self.backend,
                    )
                    return (
                        next_hidden,
                        next_kv.k_cache,
                        next_kv.v_cache,
                        next_hybrid.conv_state,
                        next_hybrid.recurrent_state,
                    )

                def skip_second_decode(_):
                    return (
                        jnp.zeros_like(hidden0),
                        kv_after_current.k_cache,
                        kv_after_current.v_cache,
                        hybrid_after_current.conv_state,
                        hybrid_after_current.recurrent_state,
                    )

                second_decode_needed = jnp.any(accepted)
                (
                    hidden1,
                    draft_k_cache,
                    draft_v_cache,
                    draft_conv_state,
                    draft_recurrent_state,
                ) = jax.lax.cond(
                    second_decode_needed,
                    run_second_decode,
                    skip_second_decode,
                    operand=None,
                )
                hybrid_after_draft = HybridLayerState(
                    conv_state=draft_conv_state,
                    recurrent_state=draft_recurrent_state,
                )

                hidden1_norm = rms_norm(hidden1, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                bonus_ids, bonus_topk_values, _ = lm_head_token_ids_and_topk(
                    hidden1_norm,
                    params,
                    self.config,
                    hidden_is_normed=True,
                    is_prefill=False,
                    top_k=2 if bonus_margin_threshold > 0 else 0,
                )
                bonus_token = bonus_ids[:, 0]
                if bonus_margin_threshold > 0:
                    bonus_top2 = bonus_topk_values[:, 0]
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_active, accepted, True))

                accept_mask = accepted[:, None, None]
                selected_hidden = jnp.where(
                    accept_mask,
                    hidden1_norm if mtp_hidden_final_normed else hidden1,
                    hidden0_norm if mtp_hidden_final_normed else hidden0,
                )
                selected_next_token = jnp.where(accepted, bonus_token, target_token)
                selected_mtp_position = jnp.where(
                    accepted,
                    next_mtp_position_arg,
                    next_mtp_position_arg - 1,
                )
                if compute_next_draft:
                    next_draft_token_ids, _ = mtp_forward_token_ids(
                        hidden_state=selected_hidden,
                        next_token_ids=selected_next_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=selected_mtp_position[:, None],
                    )
                    next_draft_token = next_draft_token_ids[:, 0].astype(jnp.int32)
                else:
                    next_draft_token = jnp.full_like(target_token, -1)

                final_any_accepted = jnp.any(accepted)

                def select_hybrid_after_draft(_):
                    return (
                        jnp.where(
                            _expand_row_mask(accepted, hybrid_after_draft.conv_state),
                            hybrid_after_draft.conv_state,
                            hybrid_after_current.conv_state,
                        ),
                        jnp.where(
                            _expand_row_mask(accepted, hybrid_after_draft.recurrent_state),
                            hybrid_after_draft.recurrent_state,
                            hybrid_after_current.recurrent_state,
                        ),
                    )

                selected_conv, selected_recurrent = jax.lax.cond(
                    final_any_accepted,
                    select_hybrid_after_draft,
                    lambda _: (hybrid_after_current.conv_state, hybrid_after_current.recurrent_state),
                    operand=None,
                )
                # Rejected draft slots are dirty but uncommitted because the
                # committed length stays at the current-token prefix.
                selected_k_cache = draft_k_cache
                selected_v_cache = draft_v_cache

                return (
                    target_token,
                    bonus_token,
                    next_draft_token,
                    accepted,
                    selected_k_cache,
                    selected_v_cache,
                    selected_conv,
                    selected_recurrent,
                    seq_lens + accepted.astype(jnp.int32),
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(8, 9))

        (
            target_token,
            bonus_token,
            next_draft_token,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_token, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp1_commit_select_greedy_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
        )

    def mtp2_commit_select_greedy_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_tokens: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """K=2 greedy MTP verifier with sequential target decodes."""
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp2_commit_select_greedy_step_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp2_commit_select_greedy_step_jit expects a decode batch")
        if draft_tokens.ndim != 2 or draft_tokens.shape[0] != batch.tokens.shape[0] or draft_tokens.shape[1] != 2:
            raise ValueError("draft_tokens must have shape [batch, 2]")
        self._log_step("mtp2_commit_select_greedy_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)

        key = (
            "mtp2-commit-select-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
            os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            ),
        )
        if key not in self._jit_cache:
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )

            def _expand_row_mask(mask: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
                return mask.reshape((mask.shape[0],) + (1,) * (target.ndim - 1))

            def _select3(prefix_len: jnp.ndarray, value0: jnp.ndarray, value1: jnp.ndarray, value2: jnp.ndarray):
                ge1 = _expand_row_mask(prefix_len >= 1, value0)
                ge2 = _expand_row_mask(prefix_len >= 2, value0)
                return jnp.where(ge2, value2, jnp.where(ge1, value1, value0))

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                query_start_loc,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_tokens_arg,
                next_mtp_position_arg,
            ):
                first_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )
                first_metadata = self.backend.build_attention_metadata(
                    positions=first_batch.positions,
                    block_tables=first_batch.block_tables,
                    seq_lens=first_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=False,
                    query_start_loc=first_batch.query_start_loc,
                    num_prefill_tokens=0,
                    num_decode_tokens=first_batch.num_decode_tokens,
                )
                first_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=first_batch.block_tables,
                    kv_lens=first_batch.seq_lens,
                    slot_mapping=first_metadata.slot_mapping,
                )
                hidden0, kv_after_current, hybrid_after_current = model_forward_step(
                    first_batch.tokens,
                    params,
                    self.config,
                    positions=first_batch.positions,
                    kv_cache_state=first_kv_state,
                    attention_metadata=first_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    last_logits_only=False,
                    backend=self.backend,
                )
                target0_ids, _, _ = lm_head_token_ids_and_topk(
                    hidden0[:, :1, :],
                    params,
                    self.config,
                    hidden_is_normed=False,
                    is_prefill=False,
                    top_k=0,
                )
                target0 = target0_ids[:, 0].astype(jnp.int32)
                row_query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
                row_active = (row_query_lens > 0) & (seq_ids >= 0)
                accepted0 = (target0 == draft_tokens_arg[:, 0]) & row_active
                if batch_accept_policy == "all_or_none":
                    accepted0 = accepted0 & jnp.all(jnp.where(row_active, accepted0, True))

                def run_second_decode(_):
                    second_query_lens = accepted0.astype(jnp.int32)
                    second_query_start_loc = jnp.concatenate(
                        [
                            jnp.zeros((1,), dtype=jnp.int32),
                            jnp.cumsum(second_query_lens),
                        ]
                    )
                    second_seq_lens = jnp.where(accepted0, seq_lens + 1, jnp.zeros_like(seq_lens))
                    second_batch = ScheduledBatch(
                        tokens=jnp.where(accepted0[:, None], target0[:, None], jnp.zeros_like(tokens)),
                        positions=jnp.where(accepted0[:, None], positions + 1, jnp.zeros_like(positions)),
                        seq_ids=jnp.where(accepted0, seq_ids, jnp.full_like(seq_ids, -1)),
                        query_start_loc=second_query_start_loc,
                        is_prefill=False,
                        num_prefill_tokens=0,
                        num_decode_tokens=jnp.sum(second_query_lens),
                        block_tables=jnp.where(
                            accepted0[:, None],
                            block_tables,
                            jnp.zeros_like(block_tables),
                        ),
                        seq_lens=second_seq_lens,
                    )
                    second_metadata = self.backend.build_attention_metadata(
                        positions=second_batch.positions,
                        block_tables=second_batch.block_tables,
                        seq_lens=second_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=False,
                        query_start_loc=second_batch.query_start_loc,
                        num_prefill_tokens=0,
                        num_decode_tokens=second_batch.num_decode_tokens,
                    )
                    second_kv_state = KVCacheState(
                        k_cache=kv_after_current.k_cache,
                        v_cache=kv_after_current.v_cache,
                        block_table=second_batch.block_tables,
                        kv_lens=second_batch.seq_lens,
                        slot_mapping=second_metadata.slot_mapping,
                    )
                    next_hidden, next_kv, next_hybrid = model_forward_step(
                        second_batch.tokens,
                        params,
                        self.config,
                        positions=second_batch.positions,
                        kv_cache_state=second_kv_state,
                        attention_metadata=second_metadata,
                        hybrid_state=hybrid_after_current,
                        is_prefill=False,
                        return_hidden=True,
                        return_hidden_with_logits=False,
                        last_logits_only=False,
                        backend=self.backend,
                    )
                    next_target_ids, _, _ = lm_head_token_ids_and_topk(
                        next_hidden[:, :1, :],
                        params,
                        self.config,
                        hidden_is_normed=False,
                        is_prefill=False,
                        top_k=0,
                    )
                    return (
                        next_hidden,
                        next_target_ids[:, 0].astype(jnp.int32),
                        next_kv.k_cache,
                        next_kv.v_cache,
                        next_hybrid.conv_state,
                        next_hybrid.recurrent_state,
                        second_metadata.slot_mapping[:, 0],
                    )

                def skip_second_decode(_):
                    return (
                        jnp.zeros_like(hidden0),
                        jnp.zeros_like(target0),
                        kv_after_current.k_cache,
                        kv_after_current.v_cache,
                        hybrid_after_current.conv_state,
                        hybrid_after_current.recurrent_state,
                        jnp.zeros((tokens.shape[0],), dtype=jnp.int32),
                    )

                (
                    hidden1,
                    target1,
                    token1_k_cache,
                    token1_v_cache,
                    token1_conv_state,
                    token1_recurrent_state,
                    token1_slots,
                ) = jax.lax.cond(
                    jnp.any(accepted0),
                    run_second_decode,
                    skip_second_decode,
                    operand=None,
                )
                kv_after_token1 = KVCacheStorage(token1_k_cache, token1_v_cache)
                hybrid_after_token1 = HybridLayerState(
                    conv_state=token1_conv_state,
                    recurrent_state=token1_recurrent_state,
                )
                accepted1 = accepted0 & (target1 == draft_tokens_arg[:, 1])
                if batch_accept_policy == "all_or_none":
                    accepted1 = accepted1 & jnp.all(jnp.where(row_active, accepted1, True))

                def run_third_decode(_):
                    third_query_lens = accepted1.astype(jnp.int32)
                    third_query_start_loc = jnp.concatenate(
                        [
                            jnp.zeros((1,), dtype=jnp.int32),
                            jnp.cumsum(third_query_lens),
                        ]
                    )
                    third_seq_lens = jnp.where(accepted1, seq_lens + 2, jnp.zeros_like(seq_lens))
                    third_batch = ScheduledBatch(
                        tokens=jnp.where(accepted1[:, None], target1[:, None], jnp.zeros_like(tokens)),
                        positions=jnp.where(accepted1[:, None], positions + 2, jnp.zeros_like(positions)),
                        seq_ids=jnp.where(accepted1, seq_ids, jnp.full_like(seq_ids, -1)),
                        query_start_loc=third_query_start_loc,
                        is_prefill=False,
                        num_prefill_tokens=0,
                        num_decode_tokens=jnp.sum(third_query_lens),
                        block_tables=jnp.where(
                            accepted1[:, None],
                            block_tables,
                            jnp.zeros_like(block_tables),
                        ),
                        seq_lens=third_seq_lens,
                    )
                    third_metadata = self.backend.build_attention_metadata(
                        positions=third_batch.positions,
                        block_tables=third_batch.block_tables,
                        seq_lens=third_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=False,
                        query_start_loc=third_batch.query_start_loc,
                        num_prefill_tokens=0,
                        num_decode_tokens=third_batch.num_decode_tokens,
                    )
                    third_kv_state = KVCacheState(
                        k_cache=kv_after_token1.k_cache,
                        v_cache=kv_after_token1.v_cache,
                        block_table=third_batch.block_tables,
                        kv_lens=third_batch.seq_lens,
                        slot_mapping=third_metadata.slot_mapping,
                    )
                    next_hidden, next_kv, next_hybrid = model_forward_step(
                        third_batch.tokens,
                        params,
                        self.config,
                        positions=third_batch.positions,
                        kv_cache_state=third_kv_state,
                        attention_metadata=third_metadata,
                        hybrid_state=hybrid_after_token1,
                        is_prefill=False,
                        return_hidden=True,
                        return_hidden_with_logits=False,
                        last_logits_only=False,
                        backend=self.backend,
                    )
                    next_target_ids, _, _ = lm_head_token_ids_and_topk(
                        next_hidden[:, :1, :],
                        params,
                        self.config,
                        hidden_is_normed=False,
                        is_prefill=False,
                        top_k=0,
                    )
                    return (
                        next_hidden,
                        next_target_ids[:, 0].astype(jnp.int32),
                        next_kv.k_cache,
                        next_kv.v_cache,
                        next_hybrid.conv_state,
                        next_hybrid.recurrent_state,
                        third_metadata.slot_mapping[:, 0],
                    )

                def skip_third_decode(_):
                    return (
                        jnp.zeros_like(hidden0),
                        jnp.zeros_like(target0),
                        kv_after_token1.k_cache,
                        kv_after_token1.v_cache,
                        hybrid_after_token1.conv_state,
                        hybrid_after_token1.recurrent_state,
                        jnp.zeros((tokens.shape[0],), dtype=jnp.int32),
                    )

                (
                    hidden2,
                    target2,
                    token2_k_cache,
                    token2_v_cache,
                    token2_conv_state,
                    token2_recurrent_state,
                    token2_slots,
                ) = jax.lax.cond(
                    jnp.any(accepted1),
                    run_third_decode,
                    skip_third_decode,
                    operand=None,
                )
                kv_after_token2 = KVCacheStorage(token2_k_cache, token2_v_cache)
                hybrid_after_token2 = HybridLayerState(
                    conv_state=token2_conv_state,
                    recurrent_state=token2_recurrent_state,
                )

                prefix_len = accepted0.astype(jnp.int32) + accepted1.astype(jnp.int32)
                hidden0_norm = rms_norm(hidden0, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                hidden1_norm = rms_norm(hidden1, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                hidden2_norm = rms_norm(hidden2, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                selected_hidden = _select3(
                    prefix_len,
                    hidden0_norm if mtp_hidden_final_normed else hidden0,
                    hidden1_norm if mtp_hidden_final_normed else hidden1,
                    hidden2_norm if mtp_hidden_final_normed else hidden2,
                )
                selected_next_token = jnp.where(prefix_len >= 2, target2, jnp.where(prefix_len >= 1, target1, target0))
                selected_mtp_position = next_mtp_position_arg - (2 - prefix_len)

                current_hidden = selected_hidden
                current_token = selected_next_token[:, None]
                current_position = selected_mtp_position[:, None]
                next_drafts = []
                for _ in range(2):
                    mtp_token_ids, current_hidden = mtp_forward_token_ids(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=current_position,
                    )
                    current_token = mtp_token_ids[:, 0].astype(jnp.int32)[:, None]
                    next_drafts.append(current_token[:, 0])
                    current_position = current_position + 1
                next_draft_tokens = jnp.stack(next_drafts, axis=1)

                selected_conv = _select3(
                    prefix_len,
                    hybrid_after_current.conv_state,
                    hybrid_after_token1.conv_state,
                    hybrid_after_token2.conv_state,
                )
                selected_recurrent = _select3(
                    prefix_len,
                    hybrid_after_current.recurrent_state,
                    hybrid_after_token1.recurrent_state,
                    hybrid_after_token2.recurrent_state,
                )

                selected_k_cache = kv_after_token2.k_cache
                selected_v_cache = kv_after_token2.v_cache

                host_payload = jnp.concatenate(
                    [
                        jnp.stack([target0, target1], axis=1).astype(jnp.int32),
                        target2.astype(jnp.int32)[:, None],
                        next_draft_tokens.astype(jnp.int32),
                        jnp.stack([accepted0, accepted1], axis=1).astype(jnp.int32),
                    ],
                    axis=1,
                )
                return (
                    jnp.stack([target0, target1], axis=1),
                    target2,
                    next_draft_tokens,
                    jnp.stack([accepted0, accepted1], axis=1),
                    selected_k_cache,
                    selected_v_cache,
                    selected_conv,
                    selected_recurrent,
                    seq_lens + prefix_len,
                    host_payload,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(8, 9))

        (
            target_tokens,
            bonus_token,
            next_draft_tokens,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            host_payload,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                batch.query_start_loc,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_tokens, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp2_commit_select_greedy_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_tokens,
            bonus_token=bonus_token,
            next_draft_token=next_draft_tokens,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            host_payload=host_payload,
        )

    def mtp_k_decode_greedy_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_tokens: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
    ) -> MTP1GreedyOutput:
        """Greedy speculative verifier for a fixed-length MTP draft chain.

        The target model decodes ``[last_token, draft_1, ..., draft_k]`` once.
        Logits from positions ``0..k-1`` verify the draft prefix. Logits from
        position ``k`` produce the target bonus token. The compiled body
        selects the committed hybrid state at each row's accepted-prefix length,
        so mixed accept/reject batches do not need a serial K=1 repair pass.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp_k_decode_greedy_step_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp_k_decode_greedy_step_jit expects a decode batch")
        if draft_tokens.ndim != 2 or draft_tokens.shape[0] != batch.tokens.shape[0]:
            raise ValueError("draft_tokens must have shape [batch, draft_len]")
        draft_len = int(draft_tokens.shape[1])
        if draft_len < 1:
            raise ValueError("draft_tokens must contain at least one draft token")
        self._log_step("mtp_k_decode_greedy_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)
        verify_mode = os.environ.get(
            "NANO_VLLM_JAX_MTP_K_VERIFY_MODE",
            "decode",
        ).strip().lower()
        if verify_mode in {"packed_prefill", "prefill_packed"}:
            verify_mode = "prefill"
        if verify_mode not in {"decode", "prefill"}:
            raise ValueError(
                "NANO_VLLM_JAX_MTP_K_VERIFY_MODE must be 'decode' or 'prefill'"
            )
        batch_accept_policy = os.environ.get(
            "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
            str(getattr(self.config, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
        )
        carry_bonus_as_draft = os.environ.get(
            "NANO_VLLM_JAX_MTP_CARRY_BONUS_AS_DRAFT",
            "0",
        ) in {"1", "true", "yes", "on", "True"}
        logit_debug_enabled = os.environ.get(
            "NANO_VLLM_JAX_MTP_K_LOGIT_DEBUG",
            "0",
        ) in {"1", "true", "yes", "on", "True"}

        key = (
            "mtp-k-decode-greedy-prefix-select",
            verify_mode,
            draft_len,
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
            bool(carry_bonus_as_draft),
            bool(logit_debug_enabled),
            batch_accept_policy,
        )
        if key not in self._jit_cache:

            def _gather_prefix(value: jnp.ndarray, prefix_len: jnp.ndarray) -> jnp.ndarray:
                gather_idx = prefix_len.astype(jnp.int32)
                gather_idx = gather_idx.reshape((gather_idx.shape[0],) + (1,) * (value.ndim - 1))
                gather_idx = jnp.broadcast_to(
                    gather_idx,
                    (value.shape[0], 1) + value.shape[2:],
                )
                return jnp.take_along_axis(value, gather_idx, axis=1)[:, 0, ...]

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_tokens_arg,
                next_mtp_position_arg,
            ):
                row_count = int(tokens.shape[0])
                verify_tokens_rows = jnp.concatenate([tokens, draft_tokens_arg], axis=1)
                verify_positions_rows = (
                    positions
                    + jnp.arange(draft_len + 1, dtype=jnp.int32)[None, :]
                )
                verify_query_start_loc = (
                    jnp.arange(row_count + 1, dtype=jnp.int32) * (draft_len + 1)
                )
                verify_as_prefill = verify_mode == "prefill"
                if verify_as_prefill:
                    verify_tokens = verify_tokens_rows.reshape(1, row_count * (draft_len + 1))
                    verify_positions = verify_positions_rows.reshape(
                        1,
                        row_count * (draft_len + 1),
                    )
                    token_row_ids = jnp.broadcast_to(
                        jnp.arange(row_count, dtype=jnp.int32)[:, None],
                        (row_count, draft_len + 1),
                    ).reshape(1, row_count * (draft_len + 1))
                else:
                    verify_tokens = verify_tokens_rows
                    verify_positions = verify_positions_rows
                    token_row_ids = None
                # Packed prefill writes all verifier tokens before attention, so
                # kv_lens must include the current token and every draft token.
                verify_seq_lens = seq_lens + (
                    draft_len + 1 if verify_as_prefill else draft_len
                )
                verify_batch = ScheduledBatch(
                    tokens=verify_tokens,
                    positions=verify_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=verify_as_prefill,
                    num_prefill_tokens=num_decode_tokens * (draft_len + 1) if verify_as_prefill else 0,
                    num_decode_tokens=0 if verify_as_prefill else num_decode_tokens * (draft_len + 1),
                    block_tables=block_tables,
                    seq_lens=verify_seq_lens,
                    packed_prefill=verify_as_prefill,
                    token_row_ids=token_row_ids,
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=verify_as_prefill,
                    query_start_loc=verify_batch.query_start_loc,
                    num_prefill_tokens=verify_batch.num_prefill_tokens,
                    num_decode_tokens=verify_batch.num_decode_tokens,
                    token_row_ids=token_row_ids,
                    max_query_len=draft_len + 1 if verify_as_prefill else None,
                )
                verify_kv_state = KVCacheState(
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=verify_batch.block_tables,
                    kv_lens=verify_batch.seq_lens,
                    slot_mapping=verify_metadata.slot_mapping,
                )
                hidden, updated_kv_state, updated_hybrid_state, prefix_hybrid_state = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=verify_as_prefill,
                    return_hidden=True,
                    return_hidden_with_logits=False,
                    return_prefix_hybrid=True,
                    backend=self.backend,
                )
                if verify_as_prefill:
                    hidden = hidden.reshape(row_count, draft_len + 1, hidden.shape[-1])
                    prefix_hybrid_state = HybridLayerState(
                        conv_state=prefix_hybrid_state.conv_state.reshape(
                            (row_count, draft_len + 1)
                            + prefix_hybrid_state.conv_state.shape[2:]
                        )
                        if prefix_hybrid_state.conv_state is not None
                        else None,
                        recurrent_state=prefix_hybrid_state.recurrent_state.reshape(
                            (row_count, draft_len + 1)
                            + prefix_hybrid_state.recurrent_state.shape[2:]
                        )
                        if prefix_hybrid_state.recurrent_state is not None
                        else None,
                    )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
                hidden_norm_for_top1 = hidden_norm.astype(output_weight.dtype)
                token_ids = _lm_head_greedy_top1_token_ids(
                    hidden_norm_for_top1.reshape(hidden_norm.shape[0] * hidden_norm.shape[1], 1, hidden_norm.shape[-1]),
                    output_weight,
                    self.config,
                ).reshape(hidden_norm.shape[0], hidden_norm.shape[1]).astype(jnp.int32)
                target_tokens = token_ids[:, :draft_len]
                bonus_token = token_ids[:, draft_len]
                if logit_debug_enabled:
                    verifier_logits = jnp.dot(
                        hidden_norm[:, :draft_len, :],
                        output_weight,
                    ).astype(jnp.float32)
                    verifier_top_values, verifier_top_ids = jax.lax.top_k(
                        verifier_logits,
                        5,
                    )
                row_query_lens = jnp.diff(verify_query_start_loc).astype(jnp.int32)
                row_active = (row_query_lens > 0) & (seq_ids >= 0)
                raw_accepted = (target_tokens == draft_tokens_arg) & row_active[:, None]
                if batch_accept_policy == "all_or_none":
                    active_or_accepted = jnp.where(row_active[:, None], raw_accepted, True)
                    accepted_by_position = jnp.all(active_or_accepted, axis=0)
                    global_prefix = jnp.cumprod(
                        accepted_by_position.astype(jnp.int32),
                        axis=0,
                    ).astype(jnp.bool_)
                    accepted = raw_accepted & global_prefix[None, :]
                else:
                    accepted = jnp.cumprod(
                        raw_accepted.astype(jnp.int32),
                        axis=1,
                    ).astype(jnp.bool_)
                prefix_len = jnp.sum(accepted.astype(jnp.int32), axis=1)

                selected_hidden_for_mtp = _gather_prefix(
                    hidden_norm if mtp_hidden_final_normed else hidden,
                    prefix_len,
                )[:, None, :]
                selected_next_token = jnp.take_along_axis(
                    token_ids,
                    prefix_len[:, None],
                    axis=1,
                )[:, 0].astype(jnp.int32)
                selected_mtp_position = next_mtp_position_arg - (draft_len - prefix_len)
                current_hidden = selected_hidden_for_mtp
                current_token = selected_next_token[:, None]
                current_position = selected_mtp_position[:, None]
                next_drafts = [selected_next_token] if carry_bonus_as_draft else []
                mtp_drafts_to_generate = draft_len - 1 if carry_bonus_as_draft else draft_len
                for _ in range(mtp_drafts_to_generate):
                    mtp_token_ids, current_hidden = mtp_forward_token_ids(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=current_position,
                    )
                    current_token = mtp_token_ids[:, 0].astype(jnp.int32)[:, None]
                    next_drafts.append(current_token[:, 0])
                    current_position = current_position + 1
                next_draft_tokens = jnp.stack(next_drafts, axis=1)
                selected_conv = (
                    _gather_prefix(prefix_hybrid_state.conv_state, prefix_len)
                    if prefix_hybrid_state.conv_state is not None
                    else updated_hybrid_state.conv_state
                )
                selected_recurrent = (
                    _gather_prefix(prefix_hybrid_state.recurrent_state, prefix_len)
                    if prefix_hybrid_state.recurrent_state is not None
                    else updated_hybrid_state.recurrent_state
                )
                emit_columns = jnp.arange(draft_len + 1, dtype=jnp.int32)[None, :]
                draft_columns = jnp.broadcast_to(
                    jnp.clip(emit_columns, 0, draft_len - 1),
                    (draft_tokens_arg.shape[0], draft_len + 1),
                )
                draft_values = jnp.take_along_axis(
                    draft_tokens_arg,
                    draft_columns,
                    axis=1,
                )
                emitted_counts = jnp.where(
                    row_active,
                    prefix_len + jnp.asarray(1, dtype=jnp.int32),
                    jnp.zeros_like(prefix_len),
                )[:, None]
                group_emitted = jnp.where(
                    emit_columns < prefix_len[:, None],
                    draft_values,
                    jnp.where(
                        emit_columns == prefix_len[:, None],
                        selected_next_token[:, None],
                        jnp.zeros_like(draft_values),
                    ),
                ).astype(jnp.int32)
                emitted_tokens = group_emitted[:, None, :]
                accepted_counts = prefix_len[:, None].astype(jnp.int32)
                emitted_totals = None
                accepted_totals = None
                rejected_totals = None
                bonus_totals = None
                accepted_bitmask = None
                compact_summary = None
                if draft_len == 1:
                    emitted_tokens = group_emitted
                    emitted_totals = emitted_counts[:, 0].astype(jnp.int32)
                    accepted_totals = prefix_len.astype(jnp.int32)
                    rejected_totals = (
                        row_active & (prefix_len < draft_len)
                    ).astype(jnp.int32)
                    bonus_totals = (
                        row_active & (prefix_len == draft_len)
                    ).astype(jnp.int32)
                    accepted_bitmask = (prefix_len > 0).astype(jnp.int32)
                    compact_summary = jnp.stack(
                        [
                            emitted_totals,
                            accepted_totals,
                            rejected_totals,
                            bonus_totals,
                            accepted_bitmask,
                        ],
                        axis=1,
                    ).astype(jnp.int32)
                if logit_debug_enabled:
                    debug_payload = (
                        jnp.zeros(
                            (row_count, 1, draft_len, 5),
                            dtype=jnp.int32,
                        ),
                        jnp.zeros(
                            (row_count, 1, draft_len, 5),
                            dtype=jnp.float32,
                        ),
                        verifier_top_ids[:, None, :, :].astype(jnp.int32),
                        verifier_top_values[:, None, :, :].astype(jnp.float32),
                        draft_tokens_arg[:, None, :].astype(jnp.int32),
                        target_tokens[:, None, :].astype(jnp.int32),
                    )
                else:
                    debug_payload = None
                return (
                    target_tokens,
                    bonus_token,
                    next_draft_tokens,
                    accepted,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    selected_conv,
                    selected_recurrent,
                    seq_lens + prefix_len,
                    emitted_tokens,
                    emitted_counts,
                    accepted_counts,
                    emitted_totals,
                    accepted_totals,
                    rejected_totals,
                    bonus_totals,
                    accepted_bitmask,
                    compact_summary,
                    debug_payload,
                )

            self._jit_cache[key] = jax.jit(compiled, donate_argnums=(8, 9))

        (
            target_tokens,
            bonus_token,
            next_draft_tokens,
            accepted,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            emitted_tokens,
            emitted_counts,
            accepted_counts,
            emitted_totals,
            accepted_totals,
            rejected_totals,
            bonus_totals,
            accepted_bitmask,
            compact_summary,
            debug_payload,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_tokens, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp_k_decode_greedy_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_tokens,
            bonus_token=bonus_token,
            next_draft_token=next_draft_tokens,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            host_payload=None,
            emitted_tokens=emitted_tokens,
            emitted_counts=emitted_counts,
            accepted_counts=accepted_counts,
            emitted_totals=emitted_totals,
            accepted_totals=accepted_totals,
            rejected_totals=rejected_totals,
            bonus_totals=bonus_totals,
            accepted_bitmask=accepted_bitmask,
            compact_summary=compact_summary,
            debug_payload=debug_payload,
            burst_groups=1,
        )

    def mtp_k_burst_greedy_step_jit(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: KVCacheStorage,
        hybrid_state: HybridLayerState,
        draft_tokens: jnp.ndarray,
        next_mtp_position: jnp.ndarray,
        mtp_hidden_final_normed: bool,
        burst_groups: int,
    ) -> MTP1GreedyOutput:
        """Run several greedy MTP verifier groups before returning to Python.

        Each group commits the accepted prefix entirely inside the compiled
        boundary, then continues the next group from that committed token/state.
        The runner drains the fixed emitted-token/count tensors after the burst;
        it does not repair rejected rows between groups.
        """
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("mtp_k_burst_greedy_step_jit requires initialized hybrid_state")
        if batch.is_prefill or batch.tokens.shape[1] != 1:
            raise ValueError("mtp_k_burst_greedy_step_jit expects a decode batch")
        if draft_tokens.ndim != 2 or draft_tokens.shape[0] != batch.tokens.shape[0]:
            raise ValueError("draft_tokens must have shape [batch, draft_len]")
        draft_len = int(draft_tokens.shape[1])
        if draft_len < 1:
            raise ValueError("draft_tokens must contain at least one draft token")
        burst_groups = int(burst_groups)
        if burst_groups <= 1:
            return self.mtp_k_decode_greedy_step_jit(
                batch,
                cache_storage=cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=draft_tokens,
                next_mtp_position=next_mtp_position,
                mtp_hidden_final_normed=mtp_hidden_final_normed,
            )
        self._log_step("mtp_k_burst_greedy_step_jit", batch, return_hidden=True, last_logits_only=False)
        self._validate_batch_contract(batch)
        verify_mode = os.environ.get(
            "NANO_VLLM_JAX_MTP_K_VERIFY_MODE",
            "decode",
        ).strip().lower()
        if verify_mode in {"packed_prefill", "prefill_packed"}:
            verify_mode = "prefill"
        if verify_mode not in {"decode", "prefill"}:
            raise ValueError(
                "NANO_VLLM_JAX_MTP_K_VERIFY_MODE must be 'decode' or 'prefill'"
            )
        logit_debug_enabled = os.environ.get(
            "NANO_VLLM_JAX_MTP_K_LOGIT_DEBUG",
            "0",
        ) in {"1", "true", "yes", "on", "True"}

        key = (
            "mtp-k-burst-greedy-compact-lm-head",
            verify_mode,
            draft_len,
            burst_groups,
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
            bool(logit_debug_enabled),
        )
        if key not in self._jit_cache:

            def compiled(
                params,
                tokens,
                positions,
                seq_ids,
                num_decode_tokens,
                block_tables,
                seq_lens,
                k_cache,
                v_cache,
                conv_state,
                recurrent_state,
                draft_tokens_arg,
                next_mtp_position_arg,
            ):
                current_tokens = tokens
                current_positions = positions
                current_seq_lens = seq_lens
                current_drafts = draft_tokens_arg
                current_next_mtp_position = next_mtp_position_arg
                current_k_cache = k_cache
                current_v_cache = v_cache
                current_conv_state = conv_state
                current_recurrent_state = recurrent_state

                emitted_groups = []
                emitted_count_groups = []
                target_groups = []
                bonus_groups = []
                accepted_groups = []
                accepted_count_groups = []
                debug_draft_top_ids_groups = []
                debug_draft_top_values_groups = []
                debug_verifier_top_ids_groups = []
                debug_verifier_top_values_groups = []
                debug_draft_token_groups = []

                def _gather_prefix(value: jnp.ndarray, prefix_len: jnp.ndarray) -> jnp.ndarray:
                    gather_idx = prefix_len.astype(jnp.int32)
                    gather_idx = gather_idx.reshape((gather_idx.shape[0],) + (1,) * (value.ndim - 1))
                    gather_idx = jnp.broadcast_to(
                        gather_idx,
                        (value.shape[0], 1) + value.shape[2:],
                    )
                    return jnp.take_along_axis(value, gather_idx, axis=1)[:, 0, ...]

                for _ in range(burst_groups):
                    verify_tokens_rows = jnp.concatenate([current_tokens, current_drafts], axis=1)
                    verify_positions_rows = (
                        current_positions
                        + jnp.arange(draft_len + 1, dtype=jnp.int32)[None, :]
                    )
                    verify_query_start_loc = (
                        jnp.arange(tokens.shape[0] + 1, dtype=jnp.int32) * (draft_len + 1)
                    )
                    verify_as_prefill = verify_mode == "prefill"
                    if verify_as_prefill:
                        verify_tokens = verify_tokens_rows.reshape(
                            1,
                            tokens.shape[0] * (draft_len + 1),
                        )
                        verify_positions = verify_positions_rows.reshape(
                            1,
                            tokens.shape[0] * (draft_len + 1),
                        )
                        token_row_ids = jnp.broadcast_to(
                            jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None],
                            (tokens.shape[0], draft_len + 1),
                        ).reshape(1, tokens.shape[0] * (draft_len + 1))
                    else:
                        verify_tokens = verify_tokens_rows
                        verify_positions = verify_positions_rows
                        token_row_ids = None
                    # Packed prefill writes all verifier tokens before attention,
                    # so kv_lens must include the current token and every draft token.
                    verify_seq_lens = current_seq_lens + (
                        draft_len + 1 if verify_as_prefill else draft_len
                    )
                    verify_batch = ScheduledBatch(
                        tokens=verify_tokens,
                        positions=verify_positions,
                        seq_ids=seq_ids,
                        query_start_loc=verify_query_start_loc,
                        is_prefill=verify_as_prefill,
                        num_prefill_tokens=num_decode_tokens * (draft_len + 1) if verify_as_prefill else 0,
                        num_decode_tokens=0 if verify_as_prefill else num_decode_tokens * (draft_len + 1),
                        block_tables=block_tables,
                        seq_lens=verify_seq_lens,
                        packed_prefill=verify_as_prefill,
                        token_row_ids=token_row_ids,
                    )
                    verify_metadata = self.backend.build_attention_metadata(
                        positions=verify_batch.positions,
                        block_tables=verify_batch.block_tables,
                        seq_lens=verify_batch.seq_lens,
                        block_size=self.config.block_size,
                        is_prefill=verify_as_prefill,
                        query_start_loc=verify_batch.query_start_loc,
                        num_prefill_tokens=verify_batch.num_prefill_tokens,
                        num_decode_tokens=verify_batch.num_decode_tokens,
                        token_row_ids=token_row_ids,
                        max_query_len=draft_len + 1 if verify_as_prefill else None,
                    )
                    verify_kv_state = KVCacheState(
                        k_cache=current_k_cache,
                        v_cache=current_v_cache,
                        block_table=verify_batch.block_tables,
                        kv_lens=verify_batch.seq_lens,
                        slot_mapping=verify_metadata.slot_mapping,
                    )
                    hidden, updated_kv_state, updated_hybrid_state, prefix_hybrid_state = model_forward_step(
                        verify_batch.tokens,
                        params,
                        self.config,
                        positions=verify_batch.positions,
                        kv_cache_state=verify_kv_state,
                        attention_metadata=verify_metadata,
                        hybrid_state=HybridLayerState(current_conv_state, current_recurrent_state),
                        is_prefill=verify_as_prefill,
                        return_hidden=True,
                        return_hidden_with_logits=False,
                        return_prefix_hybrid=True,
                        backend=self.backend,
                    )
                    if verify_as_prefill:
                        hidden = hidden.reshape(tokens.shape[0], draft_len + 1, hidden.shape[-1])
                        prefix_hybrid_state = HybridLayerState(
                            conv_state=prefix_hybrid_state.conv_state.reshape(
                                (tokens.shape[0], draft_len + 1)
                                + prefix_hybrid_state.conv_state.shape[2:]
                            )
                            if prefix_hybrid_state.conv_state is not None
                            else None,
                            recurrent_state=prefix_hybrid_state.recurrent_state.reshape(
                                (tokens.shape[0], draft_len + 1)
                                + prefix_hybrid_state.recurrent_state.shape[2:]
                            )
                            if prefix_hybrid_state.recurrent_state is not None
                            else None,
                        )

                    hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                    output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
                    hidden_norm_for_top1 = hidden_norm.astype(output_weight.dtype)
                    token_ids = _lm_head_greedy_top1_token_ids(
                        hidden_norm_for_top1.reshape(
                            hidden_norm.shape[0] * hidden_norm.shape[1],
                            1,
                            hidden_norm.shape[-1],
                        ),
                        output_weight,
                        self.config,
                    ).reshape(hidden_norm.shape[0], hidden_norm.shape[1]).astype(jnp.int32)
                    target_tokens = token_ids[:, :draft_len]
                    bonus_token = token_ids[:, draft_len]
                    if logit_debug_enabled:
                        verifier_logits = jnp.dot(
                            hidden_norm[:, :draft_len, :],
                            output_weight,
                        ).astype(jnp.float32)
                        verifier_top_values, verifier_top_ids = jax.lax.top_k(verifier_logits, 5)
                        debug_hidden_current = (
                            hidden_norm[:, 0:1, :]
                            if mtp_hidden_final_normed
                            else hidden[:, 0:1, :]
                        )
                        debug_token_current = current_tokens
                        debug_position_current = current_positions
                        draft_top_ids = []
                        draft_top_values = []
                        draft_debug_tokens = []
                        for _debug_idx in range(draft_len):
                            draft_logits, debug_hidden_current = mtp_forward(
                                hidden_state=debug_hidden_current,
                                next_token_ids=debug_token_current,
                                embed_tokens=params.embed_tokens,
                                params=params.mtp_params,
                                config=self.config,
                                positions=debug_position_current,
                            )
                            top_values, top_ids = jax.lax.top_k(
                                draft_logits[:, 0].astype(jnp.float32),
                                5,
                            )
                            debug_token_current = jnp.argmax(
                                draft_logits[:, 0],
                                axis=-1,
                            ).astype(jnp.int32)[:, None]
                            draft_top_ids.append(top_ids.astype(jnp.int32))
                            draft_top_values.append(top_values.astype(jnp.float32))
                            draft_debug_tokens.append(debug_token_current[:, 0])
                            debug_position_current = debug_position_current + 1
                        debug_draft_top_ids_groups.append(
                            jnp.stack(draft_top_ids, axis=1)
                        )
                        debug_draft_top_values_groups.append(
                            jnp.stack(draft_top_values, axis=1)
                        )
                        debug_verifier_top_ids_groups.append(verifier_top_ids.astype(jnp.int32))
                        debug_verifier_top_values_groups.append(verifier_top_values.astype(jnp.float32))
                        debug_draft_token_groups.append(
                            jnp.stack(draft_debug_tokens, axis=1)
                        )
                    row_query_lens = jnp.diff(verify_query_start_loc).astype(jnp.int32)
                    row_active = (row_query_lens > 0) & (seq_ids >= 0)
                    raw_accepted = (target_tokens == current_drafts) & row_active[:, None]
                    accepted = jnp.cumprod(
                        raw_accepted.astype(jnp.int32),
                        axis=1,
                    ).astype(jnp.bool_)
                    prefix_len = jnp.sum(accepted.astype(jnp.int32), axis=1)
                    emitted_count = jnp.where(
                        row_active,
                        prefix_len + jnp.asarray(1, dtype=jnp.int32),
                        jnp.zeros_like(prefix_len),
                    )
                    selected_next_token = jnp.take_along_axis(
                        token_ids,
                        prefix_len[:, None],
                        axis=1,
                    )[:, 0].astype(jnp.int32)
                    selected_next_token = jnp.where(
                        row_active,
                        selected_next_token,
                        jnp.zeros_like(selected_next_token),
                    )

                    mtp_hidden = (
                        _gather_prefix(hidden_norm, prefix_len)[:, None, :]
                        if mtp_hidden_final_normed
                        else _gather_prefix(hidden, prefix_len)[:, None, :]
                    )
                    mtp_drafts = []
                    mtp_hidden_current = mtp_hidden
                    mtp_token_current = selected_next_token[:, None]
                    mtp_position_current = (
                        current_next_mtp_position - (draft_len - prefix_len)
                    )[:, None]
                    for _draft_idx in range(draft_len):
                        mtp_token_ids, mtp_hidden_current = mtp_forward_token_ids(
                            hidden_state=mtp_hidden_current,
                            next_token_ids=mtp_token_current,
                            embed_tokens=params.embed_tokens,
                            params=params.mtp_params,
                            config=self.config,
                            positions=mtp_position_current,
                        )
                        mtp_token_current = mtp_token_ids[:, 0].astype(jnp.int32)[:, None]
                        mtp_drafts.append(mtp_token_current[:, 0])
                        mtp_position_current = mtp_position_current + 1
                    next_draft_tokens = jnp.stack(mtp_drafts, axis=1)

                    emit_columns = jnp.arange(draft_len + 1, dtype=jnp.int32)[None, :]
                    draft_columns = jnp.broadcast_to(
                        jnp.clip(emit_columns, 0, draft_len - 1),
                        (current_drafts.shape[0], draft_len + 1),
                    )
                    draft_values = jnp.take_along_axis(
                        current_drafts,
                        draft_columns,
                        axis=1,
                    )
                    group_emitted = jnp.where(
                        emit_columns < prefix_len[:, None],
                        draft_values,
                        jnp.where(
                            emit_columns == prefix_len[:, None],
                            selected_next_token[:, None],
                            jnp.zeros_like(draft_values),
                        ),
                    ).astype(jnp.int32)
                    selected_conv = (
                        _gather_prefix(prefix_hybrid_state.conv_state, prefix_len)
                        if prefix_hybrid_state.conv_state is not None
                        else updated_hybrid_state.conv_state
                    )
                    selected_recurrent = (
                        _gather_prefix(prefix_hybrid_state.recurrent_state, prefix_len)
                        if prefix_hybrid_state.recurrent_state is not None
                        else updated_hybrid_state.recurrent_state
                    )

                    emitted_groups.append(group_emitted)
                    emitted_count_groups.append(emitted_count)
                    target_groups.append(target_tokens)
                    bonus_groups.append(bonus_token)
                    accepted_groups.append(accepted)
                    accepted_count_groups.append(prefix_len)

                    current_tokens = selected_next_token[:, None]
                    current_positions = current_positions + emitted_count[:, None]
                    current_seq_lens = current_seq_lens + emitted_count
                    current_drafts = next_draft_tokens
                    current_next_mtp_position = current_next_mtp_position + emitted_count
                    current_k_cache = updated_kv_state.k_cache
                    current_v_cache = updated_kv_state.v_cache
                    current_conv_state = selected_conv
                    current_recurrent_state = selected_recurrent

                emitted_tokens = jnp.stack(emitted_groups, axis=1)
                emitted_counts = jnp.stack(emitted_count_groups, axis=1)
                target_tokens = jnp.stack(target_groups, axis=1)
                bonus_tokens = jnp.stack(bonus_groups, axis=1)
                accepted = jnp.stack(accepted_groups, axis=1)
                accepted_counts = jnp.stack(accepted_count_groups, axis=1)
                emitted_totals = None
                accepted_totals = None
                rejected_totals = None
                bonus_totals = None
                accepted_bitmask = None
                compact_summary = None
                if draft_len == 1:
                    group_width = draft_len + 1
                    compact_width = burst_groups * group_width
                    starts = jnp.cumsum(emitted_counts, axis=1) - emitted_counts
                    offsets = jnp.arange(group_width, dtype=jnp.int32)
                    valid_offsets = offsets[None, None, :] < emitted_counts[:, :, None]
                    dest = starts[:, :, None] + offsets[None, None, :]
                    dest = jnp.where(valid_offsets, dest, compact_width)
                    batch_idx = jnp.broadcast_to(
                        jnp.arange(tokens.shape[0], dtype=jnp.int32)[:, None, None],
                        (tokens.shape[0], burst_groups, group_width),
                    )
                    compact_tokens = jnp.zeros(
                        (tokens.shape[0], compact_width),
                        dtype=jnp.int32,
                    )
                    compact_tokens = compact_tokens.at[
                        batch_idx.reshape(-1),
                        dest.reshape(-1),
                    ].set(
                        emitted_tokens.reshape(-1),
                        mode="drop",
                    )
                    emitted_tokens = compact_tokens
                    emitted_totals = jnp.sum(emitted_counts, axis=1).astype(jnp.int32)
                    accepted_totals = jnp.sum(accepted_counts, axis=1).astype(jnp.int32)
                    rejected_totals = jnp.sum(
                        (accepted_counts < draft_len).astype(jnp.int32),
                        axis=1,
                    )
                    bonus_totals = jnp.sum(
                        (accepted_counts == draft_len).astype(jnp.int32),
                        axis=1,
                    )
                    bit_values = jnp.left_shift(
                        jnp.ones((burst_groups,), dtype=jnp.int32),
                        jnp.arange(burst_groups, dtype=jnp.int32),
                    )
                    accepted_bitmask = jnp.sum(
                        (accepted_counts > 0).astype(jnp.int32) * bit_values[None, :],
                        axis=1,
                    )
                    compact_summary = jnp.stack(
                        [
                            emitted_totals,
                            accepted_totals,
                            rejected_totals,
                            bonus_totals,
                            accepted_bitmask,
                        ],
                        axis=1,
                    ).astype(jnp.int32)
                if logit_debug_enabled:
                    debug_payload = (
                        jnp.stack(debug_draft_top_ids_groups, axis=1),
                        jnp.stack(debug_draft_top_values_groups, axis=1),
                        jnp.stack(debug_verifier_top_ids_groups, axis=1),
                        jnp.stack(debug_verifier_top_values_groups, axis=1),
                        jnp.stack(debug_draft_token_groups, axis=1),
                        target_tokens,
                    )
                else:
                    debug_payload = None
                return (
                    emitted_tokens,
                    emitted_counts,
                    target_tokens,
                    bonus_tokens,
                    current_drafts,
                    accepted,
                    accepted_counts,
                    current_k_cache,
                    current_v_cache,
                    current_conv_state,
                    current_recurrent_state,
                    current_seq_lens,
                    emitted_totals,
                    accepted_totals,
                    rejected_totals,
                    bonus_totals,
                    accepted_bitmask,
                    compact_summary,
                    debug_payload,
                )

            self._jit_cache[key] = jax.jit(compiled)

        (
            emitted_tokens,
            emitted_counts,
            target_tokens,
            bonus_tokens,
            next_draft_tokens,
            accepted,
            accepted_counts,
            k_cache,
            v_cache,
            conv_state,
            recurrent_state,
            committed_seq_lens,
            emitted_totals,
            accepted_totals,
            rejected_totals,
            bonus_totals,
            accepted_bitmask,
            compact_summary,
            debug_payload,
        ) = self._profile_jit_call(
            key,
            self._jit_cache[key],
            (
                self.params,
                batch.tokens,
                batch.positions,
                batch.seq_ids,
                jnp.asarray(batch.num_decode_tokens, dtype=jnp.int32),
                batch.block_tables,
                batch.seq_lens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state.conv_state,
                hybrid_state.recurrent_state,
                jnp.asarray(draft_tokens, dtype=jnp.int32),
                jnp.asarray(next_mtp_position, dtype=jnp.int32),
            ),
            "mtp_k_burst_greedy_step_jit",
        )
        return MTP1GreedyOutput(
            target_token=target_tokens,
            bonus_token=bonus_tokens,
            next_draft_token=next_draft_tokens,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
            committed_seq_lens=committed_seq_lens,
            emitted_tokens=emitted_tokens,
            emitted_counts=emitted_counts,
            accepted_counts=accepted_counts,
            emitted_totals=emitted_totals,
            accepted_totals=accepted_totals,
            rejected_totals=rejected_totals,
            bonus_totals=bonus_totals,
            accepted_bitmask=accepted_bitmask,
            compact_summary=compact_summary,
            burst_groups=burst_groups,
            debug_payload=debug_payload,
        )

    @staticmethod
    def _logit_positions(batch: ScheduledBatch):
        query_lens = jnp.diff(batch.query_start_loc).astype(jnp.int32)
        if batch.packed_prefill:
            return jnp.where(
                query_lens > 0,
                batch.query_start_loc[1:] - 1,
                jnp.zeros_like(query_lens),
            )
        return jnp.where(batch.is_prefill, query_lens - 1, jnp.zeros_like(query_lens))
