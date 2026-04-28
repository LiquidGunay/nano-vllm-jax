"""Inference backend boundaries.

The engine owns scheduling and logical cache metadata. Backends own physical
cache arrays and replaceable kernels. The pure JAX backend is the correctness
path; GPU/TPU backends should override only the operations that need kernels.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import jax
import jax.numpy as jnp

from nanovllm_jax.kv_cache import (
    AttentionMetadata,
    KVCacheSpec,
    KVCacheStorage,
    cap_num_kv_cache_blocks,
    compute_slot_mapping,
    init_kv_cache,
    paged_attention,
    paged_attention_prefill,
    paged_attention_decode,
    update_kv_cache,
)


class InferenceBackend(Protocol):
    """Small backend API shared by pure JAX, GPU, and TPU paths."""

    name: str

    def allocate_kv_cache(
        self,
        spec: KVCacheSpec,
        max_seqs: int,
        max_blocks_per_seq: int,
    ) -> KVCacheStorage:
        ...

    def build_attention_metadata(
        self,
        positions: jnp.ndarray,
        block_tables: jnp.ndarray,
        seq_lens: jnp.ndarray,
        block_size: int,
        is_prefill: bool,
    ) -> AttentionMetadata:
        ...

    def write_kv(
        self,
        layer_id: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
    ) -> KVCacheStorage:
        ...

    def attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> jnp.ndarray:
        ...

    def gated_delta_prefill(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_decode(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...


class PureJAXBackend:
    """Reference backend implemented with ordinary JAX operations."""

    name = "pure_jax"

    def allocate_kv_cache(
        self,
        spec: KVCacheSpec,
        max_seqs: int,
        max_blocks_per_seq: int,
    ) -> KVCacheStorage:
        capped_spec = replace(spec, num_blocks=cap_num_kv_cache_blocks(spec))
        state = init_kv_cache(
            num_blocks=capped_spec.num_blocks,
            block_size=capped_spec.block_size,
            num_kv_heads=capped_spec.num_kv_heads,
            head_dim=capped_spec.head_dim,
            max_seqs=max_seqs,
            max_blocks_per_seq=max_blocks_per_seq,
            num_layers=capped_spec.num_layers,
            dtype=capped_spec.dtype,
            max_kv_cache_bytes=capped_spec.max_kv_cache_bytes,
        )
        return state.storage

    def build_attention_metadata(
        self,
        positions: jnp.ndarray,
        block_tables: jnp.ndarray,
        seq_lens: jnp.ndarray,
        block_size: int,
        is_prefill: bool,
        query_start_loc: jnp.ndarray | None = None,
        num_prefill_tokens: int | None = None,
        num_decode_tokens: int | None = None,
    ) -> AttentionMetadata:
        slot_mapping = compute_slot_mapping(
            positions=positions,
            block_table=block_tables,
            block_size=block_size,
            is_prefill=is_prefill,
        )
        batch, query_len = positions.shape
        if query_start_loc is None:
            query_lens = jnp.where(is_prefill, seq_lens, jnp.ones_like(seq_lens))
            query_start_loc = jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens.astype(jnp.int32)),
                ]
            )
        if num_prefill_tokens is None:
            num_prefill_tokens = batch * query_len if is_prefill else 0
        if num_decode_tokens is None:
            num_decode_tokens = 0 if is_prefill else batch
        return AttentionMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            positions=positions,
        )

    def write_kv(
        self,
        layer_id: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
    ) -> KVCacheStorage:
        query_lens = jnp.diff(metadata.query_start_loc).astype(jnp.int32)
        valid_mask = jnp.arange(metadata.slot_mapping.shape[1])[None, :] < query_lens[:, None]
        k_cache, v_cache = update_kv_cache(
            cache.k_cache,
            cache.v_cache,
            metadata.slot_mapping,
            k,
            v,
            layer_idx=layer_id,
            valid_mask=valid_mask,
        )
        return KVCacheStorage(k_cache, v_cache)

    def attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> jnp.ndarray:
        if is_prefill:
            if metadata.positions is None:
                return paged_attention(
                    query=query,
                    k_cache=cache.k_cache,
                    v_cache=cache.v_cache,
                    slot_mapping=metadata.slot_mapping,
                    kv_lens=metadata.seq_lens,
                    scale=scale,
                    num_key_value_groups=num_key_value_groups,
                    layer_idx=layer_id,
                )
            return paged_attention_prefill(
                query=query,
                k_cache=cache.k_cache,
                v_cache=cache.v_cache,
                block_table=metadata.block_tables,
                kv_lens=metadata.seq_lens,
                positions=metadata.positions,
                block_size=block_size,
                scale=scale,
                num_key_value_groups=num_key_value_groups,
                layer_idx=layer_id,
            )

        return paged_attention_decode(
            query=query,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            block_table=metadata.block_tables,
            kv_lens=metadata.seq_lens,
            block_size=block_size,
            scale=scale,
            num_key_value_groups=num_key_value_groups,
            layer_idx=layer_id,
        )

    def gated_delta_prefill(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        from nanovllm_jax.model import jax_chunk_gated_delta_rule

        return jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output_final_state=True,
        )

    def gated_delta_decode(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        from nanovllm_jax.model import jax_recurrent_gated_delta_rule

        return jax_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


class KernelBackendPlaceholder(PureJAXBackend):
    """Explicit placeholder until GPU/TPU kernels are implemented."""

    def __init__(self, name: str):
        self.name = name


def select_backend(name: str = "auto") -> InferenceBackend:
    """Select an inference backend.

    `auto` currently returns the pure JAX correctness backend even on GPU/TPU.
    That keeps semantics stable until kernel backends pass parity gates.
    """

    normalized = name.lower()
    if normalized in {"auto", "pure_jax", "jax"}:
        return PureJAXBackend()
    if normalized in {"gpu", "cuda", "tpu"}:
        platform = jax.default_backend()
        if normalized == "gpu" and platform != "gpu":
            raise RuntimeError(f"Requested GPU backend, but JAX default backend is {platform!r}")
        if normalized == "tpu" and platform != "tpu":
            raise RuntimeError(f"Requested TPU backend, but JAX default backend is {platform!r}")
        return KernelBackendPlaceholder(normalized)
    raise ValueError(f"Unknown backend {name!r}; expected auto, pure_jax, gpu, or tpu")
