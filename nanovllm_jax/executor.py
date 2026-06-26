"""Compiled model execution boundary.

Owns:
    JIT cache keys, compiled prefill/decode functions, and executor outputs.
Receives:
    Batch arrays plus persistent full-attention KV and GDN hybrid state.
Returns:
    Activations, token ids, and updated cache/state objects.
Invariant:
    Executor functions do not own scheduler queues or request lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from nanovllm_jax.ops import ServingOps, ServingOpsProtocol
from nanovllm_jax.batch import ScheduledBatch
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.cache import AttentionMetadata, HybridLayerState, KVCacheState, KVCacheStorage
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import (
    ModelParams,
    _lm_head_greedy_top1_token_ids,
    forward_step as model_forward_step,
    lm_head_sample_token_ids,
    lm_head_token_ids_and_topk,
)

def _config_flag(config: Qwen3_5Config | None, attr: str) -> bool:
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return False


def _needs_static_prefill_token_count(config: Qwen3_5Config | None = None) -> bool:
    return (
        _config_flag(config, "compact_prefill_in_proj_qkv")
        or _config_flag(config, "compact_prefill_gdn_z")
        or _config_flag(config, "compact_prefill_full_attn_proj")
        or _config_flag(config, "compact_prefill_mlp")
    )


def _compact_prefill_token_count(
    batch: ScheduledBatch,
    *,
    config: Qwen3_5Config | None = None,
    max_num_batched_tokens: int | None = None,
) -> int:
    mode = getattr(config, "compact_prefill_token_count_mode", "exact")
    mode = str(mode or "exact").strip().lower()
    if mode in {"exact", "true", "true_tokens"}:
        return int(batch.num_prefill_tokens)
    if mode in {"bucket", "padded", "padded_bucket"}:
        padded_tokens = int(batch.tokens.shape[0] * batch.tokens.shape[1])
        if max_num_batched_tokens is None or max_num_batched_tokens <= 0:
            return padded_tokens
        return min(padded_tokens, int(max_num_batched_tokens))
    raise ValueError(
        f"compact_prefill_token_count_mode must be 'exact' or 'bucket', got {mode!r}"
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


class ModelExecutor:
    """Single canonical inference path for model execution."""

    def __init__(
        self,
        config: Qwen3_5Config,
        params: ModelParams,
        backend: ServingOpsProtocol | None = None,
    ):
        self.config = config
        self.params = params
        params_leaves, self._params_treedef = jax.tree_util.tree_flatten(self.params)
        self._params_leaves = tuple(params_leaves)
        self.backend = backend if backend is not None else ServingOps(config=config)
        self._jit_cache = {}

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
        """JIT-compiled variant for fixed-shape serving steps."""
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_jit requires initialized hybrid_state")
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

        activations, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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
    ) -> ExecutorOutput:
        """Prefill greedy-token path that owns hybrid table gather/scatter in JIT."""
        if not batch.is_prefill:
            raise ValueError("forward_prefill_token_ids_table_jit is prefill-only")
        if hybrid_state_table.conv_state is None or hybrid_state_table.recurrent_state is None:
            raise ValueError("forward_prefill_token_ids_table_jit requires initialized hybrid state tables")
        self._validate_batch_contract(batch)

        key = (
            "token-ids-prefill-table",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            tuple(batch.token_row_ids.shape) if batch.token_row_ids is not None else None,
            bool(batch.packed_prefill),
            bool(return_last_hidden),
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
                if return_last_hidden:
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

        activations, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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
            )
        return ExecutorOutput(
            activations=activations,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
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

        token_ids, updated_rng_counters, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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
        ) = self._jit_cache[key](
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
        ) = self._jit_cache[key](
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

        token_ids, k_cache, v_cache, conv_state, recurrent_state, seq_lens_table = self._jit_cache[key](
                self._params_leaves,
                batch.tokens,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
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
        ) = self._jit_cache[key](
                self._params_leaves,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
                resident_last_tokens,
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
        ) = self._jit_cache[key](
                self._params_leaves,
                cache_storage.k_cache,
                cache_storage.v_cache,
                hybrid_state_table.conv_state,
                hybrid_state_table.recurrent_state,
                hybrid_slot_ids,
                resident_block_tables,
                resident_seq_lens,
                resident_last_tokens,
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
        ) = self._jit_cache[key](
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

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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

        token_ids, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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
            )
        return ExecutorOutput(
            activations=token_ids,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
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
