"""Canonical model execution path shared by the engine and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from nanovllm_jax.backends import InferenceBackend, select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import AttentionMetadata, HybridLayerState, KVCacheState, KVCacheStorage
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import ModelParams, forward_step as model_forward_step
from nanovllm_jax.mtp.mtp_layer import mtp_forward


@dataclass
class ExecutorOutput:
    activations: object
    cache_storage: Optional[KVCacheStorage]
    attention_metadata: Optional[AttentionMetadata]
    hybrid_state: Optional[HybridLayerState]


@dataclass
class MTP1GreedyOutput:
    target_token: object
    bonus_token: object
    next_draft_token: object
    accepted: object
    cache_storage: KVCacheStorage
    hybrid_state: HybridLayerState


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
        self.backend = select_backend(backend) if isinstance(backend, str) else backend
        self._jit_cache = {}

    def forward_step(
        self,
        batch: ScheduledBatch,
        *,
        cache_storage: Optional[KVCacheStorage] = None,
        hybrid_state: Optional[HybridLayerState] = None,
        return_hidden: bool = False,
        last_logits_only: bool = False,
    ) -> ExecutorOutput:
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
            )
            kv_state = KVCacheState(
                k_cache=cache_storage.k_cache,
                v_cache=cache_storage.v_cache,
                block_table=batch.block_tables,
                kv_lens=batch.seq_lens,
                slot_mapping=attention_metadata.slot_mapping,
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
            return_hidden=return_hidden,
            last_logits_only=last_logits_only,
            logit_positions=self._logit_positions(batch) if last_logits_only else None,
            backend=self.backend,
        )

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
        last_logits_only: bool = False,
    ) -> ExecutorOutput:
        """JIT-compiled variant for fixed-shape benchmark/server steps."""
        if hybrid_state.conv_state is None or hybrid_state.recurrent_state is None:
            raise ValueError("forward_step_jit requires initialized hybrid_state")

        key = (
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(batch.is_prefill),
            bool(return_hidden),
            bool(last_logits_only),
        )
        if key not in self._jit_cache:
            is_prefill = bool(batch.is_prefill)

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
            ):
                step_batch = ScheduledBatch(
                    tokens=tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=is_prefill,
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
                    is_prefill=step_batch.is_prefill,
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
                activations, updated_kv_state, updated_hybrid_state = model_forward_step(
                    step_batch.tokens,
                    params,
                    self.config,
                    positions=step_batch.positions,
                    kv_cache_state=kv_state,
                    attention_metadata=attention_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=step_batch.is_prefill,
                    return_hidden=return_hidden,
                    last_logits_only=last_logits_only,
                    logit_positions=self._logit_positions(step_batch) if last_logits_only else None,
                    backend=self.backend,
                )
                return (
                    activations,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                )

            self._jit_cache[key] = jax.jit(compiled)

        activations, k_cache, v_cache, conv_state, recurrent_state = self._jit_cache[key](
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
        )
        return ExecutorOutput(
            activations=activations,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            attention_metadata=None,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
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
        if not batch.is_prefill or batch.tokens.shape != (1, 2):
            raise ValueError("mtp1_greedy_step_jit expects a single two-token verifier prefill batch")

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
                    last_logits_only=False,
                    backend=self.backend,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
                token_ids = jnp.argmax(jnp.dot(hidden_norm, output_weight), axis=-1).astype(jnp.int32)
                target_token = token_ids[0, 0]
                bonus_token = token_ids[0, 1]
                accepted = target_token == draft_token_arg

                mtp_hidden = hidden_norm[:, 1:2, :] if mtp_hidden_final_normed else hidden[:, 1:2, :]
                mtp_logits, _ = mtp_forward(
                    hidden_state=mtp_hidden,
                    next_token_ids=bonus_token.reshape(1, 1),
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=next_mtp_position_arg.reshape(1, 1),
                )
                next_draft_token = jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)[0]
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
        ) = self._jit_cache[key](
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
        )
        return MTP1GreedyOutput(
            target_token=target_token,
            bonus_token=bonus_token,
            next_draft_token=next_draft_token,
            accepted=accepted,
            cache_storage=KVCacheStorage(k_cache, v_cache),
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )

    @staticmethod
    def _logit_positions(batch: ScheduledBatch):
        query_lens = jnp.diff(batch.query_start_loc).astype(jnp.int32)
        return jnp.where(batch.is_prefill, query_lens - 1, jnp.zeros_like(query_lens))
