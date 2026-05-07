"""Canonical model execution path shared by the engine and tests."""

from __future__ import annotations

import os
import time
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
    committed_seq_lens: object | None = None


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
        self._jit_metrics: dict[tuple, dict] = {}
        self._configure_persistent_cache()

    def _configure_persistent_cache(self):
        """Enable a deterministic JAX persistent compilation cache."""
        cache_dir = os.getenv(
            "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
            os.path.expanduser("~/.cache/nano_vllm_jax/jax_compile_cache"),
        )
        if cache_dir:
            try:
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
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
        except Exception:
            # Older JAX versions may not expose this flag in older releases.
            pass

    @staticmethod
    def _batch_shapes(batch: ScheduledBatch) -> tuple:
        return (
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
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
        if batch.seq_ids.shape[0] != batch.tokens.shape[0]:
            raise ValueError("Scheduled batch seq_ids size must match batch size")
        if batch.seq_lens.shape[0] != batch.tokens.shape[0]:
            raise ValueError("Scheduled batch seq_lens size must match batch size")
        if batch.block_tables.shape[0] != batch.tokens.shape[0]:
            raise ValueError("Scheduled batch block_tables size must match batch size")
        if batch.query_start_loc.shape[0] != batch.tokens.shape[0] + 1:
            raise ValueError("Scheduled batch query_start_loc size must be batch_size + 1")

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
        if os.environ.get("NANO_VLLM_JAX_PROFILE_JIT", "0") not in {"1", "true", "yes", "on", "True"}:
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
            return_hidden_with_logits=return_hidden_with_logits,
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
            bool(batch.is_prefill),
            bool(return_hidden),
            bool(return_hidden_with_logits),
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
                    return_hidden_with_logits=return_hidden_with_logits,
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

            self._jit_cache[key] = jax.jit(
                compiled,
                donate_argnums=(9, 10),
            )

        activations, k_cache, v_cache, conv_state, recurrent_state = self._profile_jit_call(
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
            ),
            f"forward_step_jit:{'prefill' if batch.is_prefill else 'decode'}",
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
                mtp_logits, _ = mtp_forward(
                    hidden_state=mtp_hidden,
                    next_token_ids=bonus_token[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=next_mtp_positions[:, None],
                )
                next_draft_token = jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)
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
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", "0")),
            os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none"),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", "0"))
            batch_accept_policy = os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none")

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
                verify_query_lens = row_query_lens * 2
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
                    is_prefill=True,
                    num_prefill_tokens=jnp.sum(verify_query_lens),
                    num_decode_tokens=0,
                    block_tables=block_tables,
                    seq_lens=seq_lens + row_valid.astype(jnp.int32),
                )
                verify_metadata = self.backend.build_attention_metadata(
                    positions=verify_batch.positions,
                    block_tables=verify_batch.block_tables,
                    seq_lens=verify_batch.seq_lens,
                    block_size=self.config.block_size,
                    is_prefill=True,
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
                (hidden, verify_logits), updated_kv_state, updated_hybrid_state, prefix_hybrid_state = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=True,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    return_prefix_hybrid=True,
                    backend=self.backend,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids = jnp.argmax(verify_logits[:, :2], axis=-1).astype(jnp.int32)
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = (target_token == draft_token_arg) & row_valid
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))
                if bonus_margin_threshold > 0:
                    bonus_top2, _ = jax.lax.top_k(verify_logits[:, 1].astype(jnp.float32), 2)
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))

                def accepted_next_draft():
                    mtp_hidden = hidden_norm[:, 1:2, :] if mtp_hidden_final_normed else hidden[:, 1:2, :]
                    mtp_logits, _ = mtp_forward(
                        hidden_state=mtp_hidden,
                        next_token_ids=bonus_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=next_mtp_position_arg[:, None],
                    )
                    return jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)

                def rejected_next_draft():
                    rejected_hidden = hidden_norm[:, 0:1, :] if mtp_hidden_final_normed else hidden[:, 0:1, :]
                    rejected_logits, _ = mtp_forward(
                        hidden_state=rejected_hidden,
                        next_token_ids=target_token[:, None],
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=(next_mtp_position_arg - 1)[:, None],
                    )
                    return jnp.argmax(rejected_logits[:, 0], axis=-1).astype(jnp.int32)

                def mixed_next_draft():
                    accepted_draft = accepted_next_draft()
                    rejected_draft = rejected_next_draft()
                    return jnp.where(accepted, accepted_draft, rejected_draft)

                accepted_all_valid = jnp.all(jnp.where(row_valid, accepted, True))
                if batch_accept_policy == "all_or_none":
                    next_draft_token = jax.lax.cond(
                        accepted_all_valid,
                        lambda _: accepted_next_draft(),
                        lambda _: rejected_next_draft(),
                        operand=None,
                    )
                else:
                    next_draft_token = jax.lax.cond(
                        accepted_all_valid,
                        lambda _: accepted_next_draft(),
                        lambda _: mixed_next_draft(),
                        operand=None,
                    )

                hybrid_after_draft = HybridLayerState(
                    conv_state=prefix_hybrid_state.conv_state[:, 1]
                    if prefix_hybrid_state.conv_state is not None
                    else updated_hybrid_state.conv_state,
                    recurrent_state=prefix_hybrid_state.recurrent_state[:, 1]
                    if prefix_hybrid_state.recurrent_state is not None
                    else updated_hybrid_state.recurrent_state,
                )

                def restore_rejected_slots(new_cache, old_cache, slots, slot_accept):
                    leading_shape = new_cache.shape[:-4] if new_cache.ndim == 5 else new_cache.shape[:-3]
                    flat_new = new_cache.reshape(leading_shape + (-1,) + new_cache.shape[-2:])
                    flat_old = old_cache.reshape(leading_shape + (-1,) + old_cache.shape[-2:])
                    new_values = flat_new[..., slots, :, :]
                    old_values = flat_old[..., slots, :, :]
                    slot_mask = slot_accept.reshape((1,) * len(leading_shape) + (slot_accept.shape[0], 1, 1))
                    selected_values = jnp.where(slot_mask, new_values, old_values)
                    flat_new = flat_new.at[..., slots, :, :].set(selected_values)
                    return flat_new.reshape(new_cache.shape)

                conv_mask = accepted.reshape((accepted.shape[0],) + (1,) * (hybrid_after_draft.conv_state.ndim - 1))
                recurrent_mask = accepted.reshape(
                    (accepted.shape[0],) + (1,) * (hybrid_after_draft.recurrent_state.ndim - 1)
                )
                if batch_accept_policy == "rowwise":
                    restore_slots = verify_metadata.slot_mapping[:, :2].reshape((-1,))
                    restore_accept = jnp.repeat(accepted, 2)
                    selected_conv = jnp.where(conv_mask, hybrid_after_draft.conv_state, conv_state)
                    selected_recurrent = jnp.where(recurrent_mask, hybrid_after_draft.recurrent_state, recurrent_state)
                else:
                    hybrid_after_current = HybridLayerState(
                        conv_state=prefix_hybrid_state.conv_state[:, 0]
                        if prefix_hybrid_state.conv_state is not None
                        else None,
                        recurrent_state=prefix_hybrid_state.recurrent_state[:, 0]
                        if prefix_hybrid_state.recurrent_state is not None
                        else None,
                    )
                    restore_slots = verify_metadata.slot_mapping[:, 1]
                    restore_accept = accepted
                    selected_conv = jnp.where(
                        conv_mask,
                        hybrid_after_draft.conv_state,
                        hybrid_after_current.conv_state,
                    )
                    selected_recurrent = jnp.where(
                        recurrent_mask,
                        hybrid_after_draft.recurrent_state,
                        hybrid_after_current.recurrent_state,
                    )
                selected_k_cache = restore_rejected_slots(updated_kv_state.k_cache, k_cache, restore_slots, restore_accept)
                selected_v_cache = restore_rejected_slots(updated_kv_state.v_cache, v_cache, restore_slots, restore_accept)
                committed_seq_lens = seq_lens + accepted.astype(jnp.int32)
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
        )

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
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", "0")),
            os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none"),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", "0"))
            batch_accept_policy = os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none")

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
                verify_query_lens = row_query_lens * 2
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
                    seq_lens=seq_lens + row_valid.astype(jnp.int32),
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
                (hidden, verify_logits), updated_kv_state, updated_hybrid_state = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    backend=self.backend,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids = jnp.argmax(verify_logits[:, :2], axis=-1).astype(jnp.int32)
                target_token = token_ids[:, 0]
                bonus_token = token_ids[:, 1]
                accepted = (target_token == draft_token_arg) & row_valid
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))
                if bonus_margin_threshold > 0:
                    bonus_top2, _ = jax.lax.top_k(verify_logits[:, 1].astype(jnp.float32), 2)
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(jnp.where(row_valid, accepted, True))
                mtp_hidden = hidden_norm[:, 1:2, :] if mtp_hidden_final_normed else hidden[:, 1:2, :]
                mtp_logits, _ = mtp_forward(
                    hidden_state=mtp_hidden,
                    next_token_ids=bonus_token[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=next_mtp_position_arg[:, None],
                )
                next_draft_token = jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)
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
            float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", "0")),
            os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none"),
        )
        if key not in self._jit_cache:
            bonus_margin_threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_BONUS_MARGIN", "0"))
            batch_accept_policy = os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none")

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

                output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
                hidden0_norm = rms_norm(hidden0, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                target_token = jnp.argmax(jnp.dot(hidden0_norm[:, 0], output_weight), axis=-1).astype(jnp.int32)
                accepted = target_token == draft_token_arg
                if batch_accept_policy == "all_or_none":
                    accepted = accepted & jnp.all(accepted)

                second_tokens = target_token[:, None]
                second_positions = positions + 1
                second_query_lens = accepted.astype(jnp.int32)
                second_query_start_loc = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(second_query_lens),
                    ]
                )
                second_seq_lens = jnp.where(accepted, seq_lens + 1, seq_lens)
                second_batch = ScheduledBatch(
                    tokens=second_tokens,
                    positions=second_positions,
                    seq_ids=seq_ids,
                    query_start_loc=second_query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=jnp.sum(second_query_lens),
                    block_tables=block_tables,
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
                (hidden1, bonus_logits), kv_after_draft, hybrid_after_draft = model_forward_step(
                    second_batch.tokens,
                    params,
                    self.config,
                    positions=second_batch.positions,
                    kv_cache_state=second_kv_state,
                    attention_metadata=second_metadata,
                    hybrid_state=hybrid_after_current,
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    last_logits_only=True,
                    backend=self.backend,
                )

                bonus_token = jnp.argmax(bonus_logits[:, 0], axis=-1).astype(jnp.int32)
                if bonus_margin_threshold > 0:
                    bonus_top2, _ = jax.lax.top_k(bonus_logits[:, 0].astype(jnp.float32), 2)
                    bonus_margin = bonus_top2[:, 0] - bonus_top2[:, 1]
                    accepted = accepted & (bonus_margin >= bonus_margin_threshold)
                    if batch_accept_policy == "all_or_none":
                        accepted = accepted & jnp.all(accepted)

                hidden1_norm = rms_norm(hidden1, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
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
                next_draft_logits, _ = mtp_forward(
                    hidden_state=selected_hidden,
                    next_token_ids=selected_next_token[:, None],
                    embed_tokens=params.embed_tokens,
                    params=params.mtp_params,
                    config=self.config,
                    positions=selected_mtp_position[:, None],
                )
                next_draft_token = jnp.argmax(next_draft_logits[:, 0], axis=-1).astype(jnp.int32)

                selected_conv = jnp.where(
                    _expand_row_mask(accepted, hybrid_after_draft.conv_state),
                    hybrid_after_draft.conv_state,
                    hybrid_after_current.conv_state,
                )
                selected_recurrent = jnp.where(
                    _expand_row_mask(accepted, hybrid_after_draft.recurrent_state),
                    hybrid_after_draft.recurrent_state,
                    hybrid_after_current.recurrent_state,
                )
                draft_slots = second_metadata.slot_mapping[:, 0]

                def restore_rejected_draft_slots(new_cache, current_cache):
                    leading_shape = new_cache.shape[:-4] if new_cache.ndim == 5 else new_cache.shape[:-3]
                    flat_new = new_cache.reshape(leading_shape + (-1,) + new_cache.shape[-2:])
                    flat_current = current_cache.reshape(leading_shape + (-1,) + current_cache.shape[-2:])
                    new_values = flat_new[..., draft_slots, :, :]
                    current_values = flat_current[..., draft_slots, :, :]
                    slot_mask = accepted.reshape((1,) * len(leading_shape) + (accepted.shape[0], 1, 1))
                    selected_values = jnp.where(slot_mask, new_values, current_values)
                    flat_new = flat_new.at[..., draft_slots, :, :].set(selected_values)
                    return flat_new.reshape(new_cache.shape)

                selected_k_cache = restore_rejected_draft_slots(kv_after_draft.k_cache, kv_after_current.k_cache)
                selected_v_cache = restore_rejected_draft_slots(kv_after_draft.v_cache, kv_after_current.v_cache)

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
        self._log_step("mtp2_commit_select_greedy_step_jit", batch, return_hidden=True, last_logits_only=True)
        self._validate_batch_contract(batch)

        key = (
            "mtp2-commit-select-greedy",
            tuple(batch.tokens.shape),
            tuple(batch.positions.shape),
            tuple(batch.block_tables.shape),
            bool(mtp_hidden_final_normed),
        )
        if key not in self._jit_cache:

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
                (hidden0, logits0), kv_after_current, hybrid_after_current = model_forward_step(
                    first_batch.tokens,
                    params,
                    self.config,
                    positions=first_batch.positions,
                    kv_cache_state=first_kv_state,
                    attention_metadata=first_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    last_logits_only=True,
                    backend=self.backend,
                )
                target0 = jnp.argmax(logits0[:, 0], axis=-1).astype(jnp.int32)
                accepted0 = target0 == draft_tokens_arg[:, 0]

                second_batch = ScheduledBatch(
                    tokens=target0[:, None],
                    positions=positions + 1,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens + 1,
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
                (hidden1, logits1), kv_after_token1, hybrid_after_token1 = model_forward_step(
                    second_batch.tokens,
                    params,
                    self.config,
                    positions=second_batch.positions,
                    kv_cache_state=second_kv_state,
                    attention_metadata=second_metadata,
                    hybrid_state=hybrid_after_current,
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    last_logits_only=True,
                    backend=self.backend,
                )
                target1 = jnp.argmax(logits1[:, 0], axis=-1).astype(jnp.int32)
                accepted1 = accepted0 & (target1 == draft_tokens_arg[:, 1])

                third_batch = ScheduledBatch(
                    tokens=target1[:, None],
                    positions=positions + 2,
                    seq_ids=seq_ids,
                    query_start_loc=query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens + 2,
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
                (hidden2, logits2), kv_after_token2, hybrid_after_token2 = model_forward_step(
                    third_batch.tokens,
                    params,
                    self.config,
                    positions=third_batch.positions,
                    kv_cache_state=third_kv_state,
                    attention_metadata=third_metadata,
                    hybrid_state=hybrid_after_token1,
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    last_logits_only=True,
                    backend=self.backend,
                )
                target2 = jnp.argmax(logits2[:, 0], axis=-1).astype(jnp.int32)

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
                    mtp_logits, current_hidden = mtp_forward(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=current_position,
                    )
                    current_token = jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)[:, None]
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

                def restore_slot(new_cache, reference_cache, slots, keep_mask):
                    leading_shape = new_cache.shape[:-4] if new_cache.ndim == 5 else new_cache.shape[:-3]
                    flat_new = new_cache.reshape(leading_shape + (-1,) + new_cache.shape[-2:])
                    flat_ref = reference_cache.reshape(leading_shape + (-1,) + reference_cache.shape[-2:])
                    new_values = flat_new[..., slots, :, :]
                    ref_values = flat_ref[..., slots, :, :]
                    slot_mask = keep_mask.reshape((1,) * len(leading_shape) + (keep_mask.shape[0], 1, 1))
                    selected_values = jnp.where(slot_mask, new_values, ref_values)
                    flat_new = flat_new.at[..., slots, :, :].set(selected_values)
                    return flat_new.reshape(new_cache.shape)

                token1_slots = second_metadata.slot_mapping[:, 0]
                token2_slots = third_metadata.slot_mapping[:, 0]
                selected_k_cache = restore_slot(
                    kv_after_token2.k_cache,
                    kv_after_token1.k_cache,
                    token2_slots,
                    prefix_len >= 2,
                )
                selected_v_cache = restore_slot(
                    kv_after_token2.v_cache,
                    kv_after_token1.v_cache,
                    token2_slots,
                    prefix_len >= 2,
                )
                selected_k_cache = restore_slot(
                    selected_k_cache,
                    kv_after_current.k_cache,
                    token1_slots,
                    prefix_len >= 1,
                )
                selected_v_cache = restore_slot(
                    selected_v_cache,
                    kv_after_current.v_cache,
                    token1_slots,
                    prefix_len >= 1,
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
        position ``k`` produce the target bonus token. On all-row acceptance,
        the runner commits the cache/hybrid state after ``draft_k`` and stores
        a fresh MTP draft chain seeded from ``hidden(draft_k), bonus``.
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

        key = (
            "mtp-k-decode-greedy",
            draft_len,
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
                verify_tokens = jnp.concatenate([tokens, draft_tokens_arg], axis=1)
                verify_positions = positions + jnp.arange(draft_len + 1, dtype=jnp.int32)[None, :]
                verify_query_start_loc = jnp.arange(tokens.shape[0] + 1, dtype=jnp.int32) * (draft_len + 1)
                verify_batch = ScheduledBatch(
                    tokens=verify_tokens,
                    positions=verify_positions,
                    seq_ids=seq_ids,
                    query_start_loc=verify_query_start_loc,
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=num_decode_tokens * (draft_len + 1),
                    block_tables=block_tables,
                    seq_lens=seq_lens + draft_len,
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
                (hidden, verify_logits), updated_kv_state, updated_hybrid_state = model_forward_step(
                    verify_batch.tokens,
                    params,
                    self.config,
                    positions=verify_batch.positions,
                    kv_cache_state=verify_kv_state,
                    attention_metadata=verify_metadata,
                    hybrid_state=HybridLayerState(conv_state, recurrent_state),
                    is_prefill=False,
                    return_hidden=True,
                    return_hidden_with_logits=True,
                    backend=self.backend,
                )

                hidden_norm = rms_norm(hidden, params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)
                token_ids = jnp.argmax(verify_logits, axis=-1).astype(jnp.int32)
                target_tokens = token_ids[:, :draft_len]
                bonus_token = token_ids[:, draft_len]
                accepted = target_tokens == draft_tokens_arg

                mtp_hidden = (
                    hidden_norm[:, draft_len : draft_len + 1, :]
                    if mtp_hidden_final_normed
                    else hidden[:, draft_len : draft_len + 1, :]
                )
                current_hidden = mtp_hidden
                current_token = bonus_token[:, None]
                current_position = next_mtp_position_arg[:, None]
                next_drafts = []
                for _ in range(draft_len):
                    mtp_logits, current_hidden = mtp_forward(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=params.embed_tokens,
                        params=params.mtp_params,
                        config=self.config,
                        positions=current_position,
                    )
                    current_token = jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)[:, None]
                    next_drafts.append(current_token[:, 0])
                    current_position = current_position + 1
                next_draft_tokens = jnp.stack(next_drafts, axis=1)
                return (
                    target_tokens,
                    bonus_token,
                    next_draft_tokens,
                    accepted,
                    updated_kv_state.k_cache,
                    updated_kv_state.v_cache,
                    updated_hybrid_state.conv_state,
                    updated_hybrid_state.recurrent_state,
                    seq_lens + draft_len,
                )

            self._jit_cache[key] = jax.jit(compiled)

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
        )

    @staticmethod
    def _logit_positions(batch: ScheduledBatch):
        query_lens = jnp.diff(batch.query_start_loc).astype(jnp.int32)
        return jnp.where(batch.is_prefill, query_lens - 1, jnp.zeros_like(query_lens))
