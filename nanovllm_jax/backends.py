"""Inference backend boundaries.

The engine owns scheduling and logical cache metadata. Backends own physical
cache arrays and replaceable kernels. The pure JAX backend is the correctness
path; GPU/TPU backends should override only the operations that need kernels.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import core

from nanovllm_jax.kernels.registry import (
    KernelBackendStatus,
    KernelBackendUnavailable,
    select_kernel_backend,
)
from nanovllm_jax.kv_cache import (
    AttentionMetadata,
    FullAttentionNHDKVCacheStorage,
    KVCacheSpec,
    KVCacheStorage,
    cap_num_kv_cache_blocks,
    compute_slot_mapping,
    init_kv_cache,
    init_full_attention_nhd_kv_cache,
    paged_attention,
    paged_attention_prefill,
    paged_attention_decode,
    update_kv_cache,
)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}
_NHD_FULL_ATTN_CACHE_ENV = "NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE"
_FLASHINFER_KV_APPEND_ENV = "NANO_VLLM_JAX_FLASHINFER_KV_APPEND"
_CUDA_FP32_KV_APPEND_ENV = "NANO_VLLM_JAX_CUDA_FP32_KV_APPEND"
_CUDA_FP32_DECODE_ATTN_ENV = "NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN"
_CUDA_FP32_GDN_DECODE_ENV = "NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE"
_GDN_PACKED_DECODE_IMPL_ENV = "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL"
_GDN_PACKED_DECODE_PRENORMALIZE_QK_ENV = (
    "NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK"
)
_GDN_PREFILL_POST_CONV_IMPL_ENV = "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL"
_GDN_PREFILL_ACT_DTYPE_ENV = "NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE"
_GDN_PREFILL_QKV_DTYPE_ENV = "NANO_VLLM_JAX_GDN_PREFILL_QKV_DTYPE"
_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE_ENV = (
    "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE"
)
_GDN_PACKED_DECODE_QKV_DTYPE_ENV = "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE"
_OFF_ENV_VALUES = {"", "0", "false", "no", "off", "none", "False"}


def _gdn_packed_decode_impl() -> str:
    value = os.environ.get(_GDN_PACKED_DECODE_IMPL_ENV, "off").strip()
    if value in _TRUE_ENV_VALUES:
        return "cuda_fp32"
    normalized = value.lower()
    if normalized in _OFF_ENV_VALUES:
        return "off"
    if normalized in {"reference", "jax", "pure_jax"}:
        return "reference"
    if normalized in {"triton_fla", "fla_triton", "jax_triton", "triton"}:
        return "triton_fla"
    if normalized in {"cuda", "cuda_fp32", "fast"}:
        return "cuda_fp32"
    raise ValueError(
        f"Unknown {_GDN_PACKED_DECODE_IMPL_ENV}={value!r}; "
        "expected off, reference, triton_fla, or cuda_fp32"
    )


def gdn_packed_decode_enabled() -> bool:
    return _gdn_packed_decode_impl() != "off"


def _gdn_prefill_post_conv_impl() -> str:
    value = os.environ.get(_GDN_PREFILL_POST_CONV_IMPL_ENV, "off").strip()
    if value in _TRUE_ENV_VALUES:
        return "reference"
    normalized = value.lower()
    if normalized in _OFF_ENV_VALUES:
        return "off"
    if normalized in {"reference", "jax", "pure_jax"}:
        return "reference"
    if normalized in {"reference_fla", "reference_fla_chunk32", "fla_reference"}:
        return "reference_fla_chunk32"
    if normalized in {
        "reference_fla_packed",
        "reference_fla_composed",
        "fla_packed_reference",
        "fla_composed_reference",
    }:
        return "reference_fla_packed"
    if normalized in {
        "triton_fla_wrapper",
        "fla_wrapper",
    }:
        return "triton_fla_wrapper"
    if normalized in {
        "triton_fla_packed",
        "fla_triton_packed",
        "triton_fla_packed_reference",
        "packed_triton_fla",
    }:
        return "triton_fla_packed"
    if normalized in {
        "triton_fla_padded",
        "triton_fla_grid",
        "fla_triton_padded",
    }:
        return "triton_fla_padded"
    if normalized in {
        "triton_fla_prep_bf16",
        "fla_triton_prep_bf16",
        "triton_prep_bf16",
        "jax_triton_prep_bf16",
    }:
        return "triton_fla_prep_bf16"
    if normalized in {
        "cuda_fla",
        "cuda_fla_chunk32",
        "cuda_fla_chunk32_fp32",
        "cuda_prepared_fp32",
    }:
        return "cuda_fla_chunk32_fp32"
    if normalized in {"cuda_prep", "cuda_prep_fp32", "fused_prep_fp32"}:
        return "cuda_prep_fp32"
    if normalized in {
        "cuda_prep_prefill",
        "cuda_prep_prefill_fp32",
        "cuda_prefill_fp32",
        "fast",
    }:
        return "cuda_prep_prefill_fp32"
    raise ValueError(
        f"Unknown {_GDN_PREFILL_POST_CONV_IMPL_ENV}={value!r}; "
        "expected off, reference, reference_fla_chunk32, reference_fla_packed, "
        "triton_fla_padded, triton_fla_packed, triton_fla_prep_bf16, "
        "cuda_fla_chunk32_fp32, "
        "cuda_prep_fp32, "
        "or cuda_prep_prefill_fp32"
    )


def gdn_prefill_post_conv_enabled() -> bool:
    return _gdn_prefill_post_conv_impl() != "off"


def _normalize_gdn_prefill_dtype(value: str, env_name: str) -> str:
    normalized = value.lower()
    if normalized in _OFF_ENV_VALUES or normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(f"Unknown {env_name}={value!r}; expected fp32 or bf16")


def _gdn_prefill_qkv_activation_dtype() -> str:
    value = os.environ.get(_GDN_PREFILL_QKV_DTYPE_ENV)
    env_name = _GDN_PREFILL_QKV_DTYPE_ENV
    if value is None:
        value = os.environ.get(_GDN_PREFILL_ACT_DTYPE_ENV, "fp32")
        env_name = _GDN_PREFILL_ACT_DTYPE_ENV
    return _normalize_gdn_prefill_dtype(value.strip(), env_name)


def _gdn_packed_decode_qkv_activation_dtype() -> str:
    value = os.environ.get(_GDN_PACKED_DECODE_QKV_DTYPE_ENV, "fp32").strip()
    return _normalize_gdn_prefill_dtype(value, _GDN_PACKED_DECODE_QKV_DTYPE_ENV)


def _gdn_prefill_activation_dtype() -> str:
    return _gdn_prefill_qkv_activation_dtype()


def _gdn_prefill_qkv_activation_jnp_dtype() -> jnp.dtype:
    if _gdn_prefill_qkv_activation_dtype() == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _gdn_packed_decode_qkv_activation_jnp_dtype() -> jnp.dtype:
    if _gdn_packed_decode_qkv_activation_dtype() == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _gdn_packed_decode_pre_normalize_qk() -> bool:
    return (
        os.environ.get(_GDN_PACKED_DECODE_PRENORMALIZE_QK_ENV, "0").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _gdn_prefill_post_conv_output_dtype() -> str:
    value = os.environ.get(_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE_ENV, "fp32").strip()
    return _normalize_gdn_prefill_dtype(
        value,
        _GDN_PREFILL_POST_CONV_OUTPUT_DTYPE_ENV,
    )


def _cast_gdn_prefill_activations(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    act_dtype = _gdn_prefill_qkv_activation_jnp_dtype()
    if act_dtype == jnp.bfloat16:
        return (
            query.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
        )
    return query, key, value


def _cast_gdn_prefill_post_conv_output(output: jnp.ndarray) -> jnp.ndarray:
    if _gdn_prefill_post_conv_output_dtype() == "bf16":
        return output.astype(jnp.bfloat16)
    return output


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

    def allocate_full_attention_nhd_kv_cache(
        self,
        spec: KVCacheSpec,
        full_attention_layers: tuple[int, ...],
    ) -> FullAttentionNHDKVCacheStorage | None:
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

    def gated_delta_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        valid_token_mask: jnp.ndarray | None,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
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

    def gated_delta_packed_decode(
        self,
        mixed_qkv: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        initial_state: jnp.ndarray,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...


class PureJAXBackend:
    """Reference backend implemented with ordinary JAX operations."""

    name = "pure_jax"

    def __init__(self, kernel_backend: KernelBackendStatus | None = None):
        self.kernel_backend = kernel_backend or select_kernel_backend("pure_jax")

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

    def allocate_full_attention_nhd_kv_cache(
        self,
        spec: KVCacheSpec,
        full_attention_layers: tuple[int, ...],
    ) -> FullAttentionNHDKVCacheStorage | None:
        if os.environ.get(_NHD_FULL_ATTN_CACHE_ENV, "0") not in _TRUE_ENV_VALUES:
            return None
        return init_full_attention_nhd_kv_cache(
            spec=replace(spec, num_blocks=cap_num_kv_cache_blocks(spec)),
            full_attention_layers=full_attention_layers,
        )

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
        if positions.ndim != 2:
            raise ValueError("positions must be a 2D tensor [batch, query_len]")
        if block_tables.ndim != 2:
            raise ValueError("block_tables must be a 2D tensor [batch, max_blocks_per_seq]")
        if positions.shape[0] != block_tables.shape[0]:
            raise ValueError(
                "positions and block_tables batch dimensions must match"
            )
        if positions.shape[0] != seq_lens.shape[0]:
            raise ValueError("positions and seq_lens batch dimensions must match")

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
            max_kv_len=block_tables.shape[1] * block_size if not is_prefill else None,
        )

    def write_kv(
        self,
        layer_id: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
    ) -> KVCacheStorage:
        if os.environ.get(_CUDA_FP32_KV_APPEND_ENV, "0") in _TRUE_ENV_VALUES:
            if os.environ.get(_FLASHINFER_KV_APPEND_ENV, "0") in _TRUE_ENV_VALUES:
                raise ValueError(
                    "Set only one KV append backend: CUDA FP32 or FlashInfer"
                )
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "CUDA FP32 KV append requires cache shape "
                    "[num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            from nanovllm_jax.kernels.cuda_fp32_ffi import (
                kv_append_paged_nhd_fp32_from_metadata,
            )

            k_cache_layer, v_cache_layer = kv_append_paged_nhd_fp32_from_metadata(
                k,
                v,
                cache.k_cache[layer_id],
                cache.v_cache[layer_id],
                metadata,
                page_size=int(cache.k_cache.shape[2]),
            )
            return KVCacheStorage(
                cache.k_cache.at[layer_id].set(k_cache_layer),
                cache.v_cache.at[layer_id].set(v_cache_layer),
            )

        if os.environ.get(_FLASHINFER_KV_APPEND_ENV, "0") in _TRUE_ENV_VALUES:
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "FlashInfer KV append requires cache shape "
                    "[num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            from nanovllm_jax.kernels.flashinfer_ffi import (
                kv_append_paged_nhd_from_metadata,
            )

            k_cache_layer, v_cache_layer = kv_append_paged_nhd_from_metadata(
                k,
                v,
                cache.k_cache[layer_id],
                cache.v_cache[layer_id],
                metadata,
                page_size=int(cache.k_cache.shape[2]),
            )
            return KVCacheStorage(
                cache.k_cache.at[layer_id].set(k_cache_layer),
                cache.v_cache.at[layer_id].set(v_cache_layer),
            )

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

        if metadata.positions is None:
            raise ValueError("metadata.positions is required for decode attention")
        if metadata.positions.shape[0] != metadata.block_tables.shape[0]:
            raise ValueError("positions and block_tables batch dimensions must match")
        if (
            os.environ.get(_CUDA_FP32_DECODE_ATTN_ENV, "0") in _TRUE_ENV_VALUES
            and query.shape[1] == 1
            and query.dtype == jnp.float32
            and cache.k_cache.dtype == jnp.float32
            and cache.v_cache.dtype == jnp.float32
            and cache.k_cache.ndim == 5
            and cache.v_cache.ndim == 5
        ):
            from nanovllm_jax.kernels.cuda_fp32_ffi import (
                paged_decode_attention_gqa_nhd_fp32,
            )
            from nanovllm_jax.kernels.paged_attention import (
                dense_block_tables_to_kv_indptr,
                kv_last_page_len_from_seq_lens,
            )

            kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(
                metadata.block_tables,
            )
            out = paged_decode_attention_gqa_nhd_fp32(
                query[:, 0],
                cache.k_cache[layer_id],
                cache.v_cache[layer_id],
                kv_indptr,
                kv_indices,
                kv_last_page_len_from_seq_lens(metadata.seq_lens, block_size),
                metadata.seq_lens.astype(jnp.int32),
                scale,
            )
            return out.reshape(
                query.shape[0],
                1,
                query.shape[2] * query.shape[3],
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
            max_kv_len=metadata.max_kv_len,
            positions=metadata.positions,
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

    def gated_delta_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        valid_token_mask: jnp.ndarray | None,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        impl = _gdn_prefill_post_conv_impl()
        if impl == "off":
            raise RuntimeError(
                f"{_GDN_PREFILL_POST_CONV_IMPL_ENV} is off; use gated_delta_prefill"
            )
        if impl == "reference":
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_post_conv_prefill_reference_from_decay,
            )

            return gdn_post_conv_prefill_reference_from_decay(
                conv_out,
                a,
                b,
                decay,
                dt_bias,
                valid_token_mask,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                chunk_size=chunk_size,
                initial_state=initial_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if impl == "reference_fla_chunk32":
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_fla_prefill_chunk32_fp32_reference,
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
            )

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            prepared = prepare_gdn_fla_prefill_kernel_inputs(
                query,
                key,
                value,
                gate.astype(jnp.float32),
                beta.astype(jnp.float32),
                seq_lens,
                initial_state.astype(jnp.float32),
                qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(),
            )
            output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                prepared.query,
                prepared.key,
                prepared.value,
                prepared.gate,
                prepared.beta,
                prepared.seq_lens,
                prepared.initial_state,
                chunk_size=chunk_size,
            )
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl == "reference_fla_packed":
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_fla_prefill_varlen_composed_reference,
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
            )

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            prepared = prepare_gdn_fla_prefill_kernel_inputs(
                query,
                key,
                value,
                gate.astype(jnp.float32),
                beta.astype(jnp.float32),
                seq_lens,
                initial_state.astype(jnp.float32),
                qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(),
            )
            output, final_state = gdn_fla_prefill_varlen_composed_reference(
                prepared.query,
                prepared.key,
                prepared.value,
                prepared.gate,
                prepared.beta,
                prepared.seq_lens,
                prepared.initial_state,
                chunk_size=chunk_size,
            )
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl == "triton_fla_packed":
            from nanovllm_jax.kernels.gdn_fla import (
                pack_prepared_gdn_prefill_inputs,
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
                gdn_fla_prefill_chunk32_fp32_reference,
                unpack_prepared_gdn_prefill_output,
            )
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_fla_chunk_gated_delta_rule_packed_triton,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                gdn_fla_chunk_gated_delta_rule_packed_triton = None

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            prepared = prepare_gdn_fla_prefill_kernel_inputs(
                query,
                key,
                value,
                gate.astype(jnp.float32),
                beta.astype(jnp.float32),
                seq_lens,
                initial_state.astype(jnp.float32),
                qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(),
            )
            if isinstance(prepared.seq_lens, core.Tracer):
                output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                    prepared.initial_state,
                    chunk_size=chunk_size,
                )
                output = _cast_gdn_prefill_post_conv_output(output)
                return output.transpose(0, 2, 1, 3), final_state

            if gdn_fla_chunk_gated_delta_rule_packed_triton is None:
                output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                    prepared.initial_state,
                    chunk_size=chunk_size,
                )
                output = _cast_gdn_prefill_post_conv_output(output)
                return output.transpose(0, 2, 1, 3), final_state

            packed_query, packed_key, packed_value, packed_gate, packed_beta, cu_seqlens = (
                pack_prepared_gdn_prefill_inputs(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                )
            )
            output, final_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
                packed_query,
                packed_key,
                packed_value,
                packed_gate,
                packed_beta,
                cu_seqlens,
                prepared.initial_state,
                chunk_size=chunk_size,
                # Q/K normalization was already applied during post-conv prep
                # when requested; do not normalize the packed tensors again.
                use_qk_l2norm_in_kernel=False,
            )
            output = unpack_prepared_gdn_prefill_output(
                output,
                cu_seqlens,
                prepared.query.shape[1],
            )
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl == "triton_fla_wrapper":
            from nanovllm_jax.kernels.gdn_fla import (
                pack_prepared_gdn_prefill_inputs,
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
                unpack_prepared_gdn_prefill_output,
                gdn_segmented_prefill_chunk32,
            )

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            prepared = prepare_gdn_fla_prefill_kernel_inputs(
                query,
                key,
                value,
                gate.astype(jnp.float32),
                beta.astype(jnp.float32),
                seq_lens,
                initial_state.astype(jnp.float32),
                qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(),
            )
            if isinstance(prepared.seq_lens, core.Tracer):
                from nanovllm_jax.kernels.gdn_fla import (
                    gdn_fla_prefill_chunk32_fp32_reference,
                )

                output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                    prepared.initial_state,
                    chunk_size=chunk_size,
                )
                output = _cast_gdn_prefill_post_conv_output(output)
                return output.transpose(0, 2, 1, 3), final_state

            packed_query, packed_key, packed_value, packed_gate, packed_beta, cu_seqlens = (
                pack_prepared_gdn_prefill_inputs(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                )
            )
            output, final_state = gdn_segmented_prefill_chunk32(
                packed_query,
                packed_key,
                packed_value,
                packed_beta,
                packed_gate,
                cu_seqlens,
                prepared.initial_state,
                chunk_size=chunk_size,
                # Q/K normalization was already applied during post-conv prep
                # when requested; do not normalize the packed tensors again.
                use_qk_l2norm_in_kernel=False,
            )
            output = unpack_prepared_gdn_prefill_output(
                output,
                cu_seqlens,
                prepared.query.shape[1],
            )
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl == "triton_fla_padded":
            from nanovllm_jax.kernels.gdn_fla import (
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
                gdn_fla_prefill_chunk32_fp32_reference,
            )
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_fla_chunk_gated_delta_rule_packed_triton,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                gdn_fla_chunk_gated_delta_rule_packed_triton = None

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            prepared = prepare_gdn_fla_prefill_kernel_inputs(
                query,
                key,
                value,
                gate.astype(jnp.float32),
                beta.astype(jnp.float32),
                seq_lens,
                initial_state.astype(jnp.float32),
                qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(),
            )
            packed_query = prepared.query.reshape(
                -1,
                prepared.query.shape[2],
                prepared.query.shape[3],
            )
            packed_key = prepared.key.reshape(
                -1,
                prepared.key.shape[2],
                prepared.key.shape[3],
            )
            packed_value = prepared.value.reshape(
                -1,
                prepared.value.shape[2],
                prepared.value.shape[3],
            )
            packed_gate = prepared.gate.reshape(
                -1,
                prepared.gate.shape[2],
            )
            packed_beta = prepared.beta.reshape(
                -1,
                prepared.beta.shape[2],
            )
            batch, seq_len, num_key_heads_out, key_head_dim_prepared = (
                prepared.query.shape
            )
            if key_head_dim_prepared != key_head_dim:
                raise ValueError(
                    "prepared query head dim must match configured key_head_dim"
                )
            value_dim = prepared.value.shape[-1]
            packed_cu_seqlens = (
                jnp.arange(batch + 1, dtype=jnp.int32) * jnp.int32(seq_len)
            )
            max_chunks_per_row = (seq_len + chunk_size - 1) // chunk_size
            row_ids = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), max_chunks_per_row)
            chunk_ids = jnp.tile(
                jnp.arange(max_chunks_per_row, dtype=jnp.int32), batch
            )
            packed_chunk_indices = jnp.stack((row_ids, chunk_ids), axis=1)
            packed_chunk_offsets = (
                jnp.arange(batch + 1, dtype=jnp.int32) * max_chunks_per_row
            )

            if gdn_fla_chunk_gated_delta_rule_packed_triton is None:
                output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                    prepared.initial_state,
                    chunk_size=chunk_size,
                )
                output = output.reshape(batch, seq_len, num_key_heads_out, value_dim)
                output = _cast_gdn_prefill_post_conv_output(output)
                return output.transpose(0, 2, 1, 3), final_state

            output, final_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
                packed_query,
                packed_key,
                packed_value,
                packed_gate,
                packed_beta,
                packed_cu_seqlens,
                prepared.initial_state,
                chunk_size=chunk_size,
                # Q/K normalization was already applied during post-conv prep
                # when requested; do not normalize the packed tensors again.
                use_qk_l2norm_in_kernel=False,
                max_row_chunks=max_chunks_per_row,
                chunk_indices=packed_chunk_indices,
                chunk_offsets=packed_chunk_offsets,
            )
            output = output.reshape(batch, seq_len, num_key_heads_out, value_dim)
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl == "triton_fla_prep_bf16":
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_fla_prefill_chunk32_fp32_reference,
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
            )
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_post_conv_prep_bf16,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                gdn_post_conv_prep_bf16 = None

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            if valid_token_mask is None:
                valid_token_mask = jnp.ones(conv_out.shape[:2], dtype=jnp.int32)
            else:
                valid_token_mask = valid_token_mask.astype(jnp.int32)
            if gdn_post_conv_prep_bf16 is None:
                query, key, value, gate, beta, seq_lens = (
                    prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                        conv_out,
                        a,
                        b,
                        decay,
                        dt_bias,
                        valid_token_mask,
                        num_key_heads=num_key_heads,
                        num_value_heads=num_value_heads,
                        key_head_dim=key_head_dim,
                        value_head_dim=value_head_dim,
                        normalize_qk=use_qk_l2norm_in_kernel,
                    )
                )
            else:
                query, key, value, gate, beta = gdn_post_conv_prep_bf16(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
                seq_lens = valid_token_mask.sum(axis=1).astype(jnp.int32)
            beta = jax.nn.sigmoid(b.astype(jnp.float32))
            gate = -decay.astype(jnp.float32) * jax.nn.softplus(
                a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
            )
            beta = jnp.where(valid_token_mask[:, :, None] > 0, beta, 0.0)
            gate = jnp.where(valid_token_mask[:, :, None] > 0, gate, 0.0)
            output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                query,
                key,
                value,
                gate,
                beta,
                seq_lens,
                initial_state.astype(jnp.float32),
                chunk_size=chunk_size,
            )
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl == "cuda_fla_chunk32_fp32":
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_fla_prefill_chunk32_fp32_reference,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
            )

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            if (
                chunk_size == 32
                and query.shape[1] % 32 == 0
                and value.shape[3] % 32 == 0
            ):
                from nanovllm_jax.kernels.cuda_fp32_ffi import (
                    gdn_prefill_chunk32_prepared_fp32,
                )

                output, final_state = gdn_prefill_chunk32_prepared_fp32(
                    query.astype(jnp.float32),
                    key.astype(jnp.float32),
                    value.astype(jnp.float32),
                    gate.astype(jnp.float32),
                    beta.astype(jnp.float32),
                    seq_lens.astype(jnp.int32),
                    initial_state.astype(jnp.float32),
                )
            else:
                output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    seq_lens,
                    initial_state.astype(jnp.float32),
                    chunk_size=chunk_size,
                )
            output = _cast_gdn_prefill_post_conv_output(output)
            return output.transpose(0, 2, 1, 3), final_state

        if impl in {"cuda_prep_fp32", "cuda_prep_prefill_fp32"}:
            if not use_qk_l2norm_in_kernel:
                raise ValueError("CUDA FP32 post-conv prep requires q/k l2norm")
            from nanovllm_jax.kernels.cuda_fp32_ffi import (
                gdn_post_conv_prep_fp32,
            )
            from nanovllm_jax.model import jax_chunk_gated_delta_rule

            if valid_token_mask is None:
                valid_mask_i32 = jnp.ones(conv_out.shape[:2], dtype=jnp.int32)
            else:
                valid_mask_i32 = valid_token_mask.astype(jnp.int32)
            query, key, value, gate, beta = gdn_post_conv_prep_fp32(
                conv_out.astype(jnp.float32),
                a.astype(jnp.float32),
                b.astype(jnp.float32),
                decay.astype(jnp.float32),
                dt_bias.astype(jnp.float32),
                valid_mask_i32,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
            )
            if impl == "cuda_prep_prefill_fp32":
                if initial_state is None:
                    initial_state = jnp.zeros(
                        (
                            conv_out.shape[0],
                            num_value_heads,
                            value_head_dim,
                            key_head_dim,
                        ),
                        dtype=jnp.float32,
                    )
                seq_lens = valid_mask_i32.sum(axis=1).astype(jnp.int32)
                if query.shape[2] % 32 == 0 and value_head_dim % 32 == 0:
                    from nanovllm_jax.kernels.cuda_fp32_ffi import (
                        gdn_prefill_chunk32_normalized_fp32,
                        gdn_prefill_chunk32_v64_normalized_fp32,
                    )

                    query_scaled = query * (
                        1.0 / jnp.sqrt(jnp.asarray(key_head_dim, dtype=jnp.float32))
                    )
                    if value_head_dim % 64 == 0:
                        return gdn_prefill_chunk32_v64_normalized_fp32(
                            query_scaled,
                            key,
                            value,
                            gate,
                            beta,
                            seq_lens,
                            initial_state.astype(jnp.float32),
                        )
                    return gdn_prefill_chunk32_normalized_fp32(
                        query_scaled,
                        key,
                        value,
                        gate,
                        beta,
                        seq_lens,
                        initial_state.astype(jnp.float32),
                    )
            return jax_chunk_gated_delta_rule(
                query,
                key,
                value,
                gate,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )

        raise AssertionError(f"Unhandled GDN post-conv prefill implementation {impl!r}")

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
        if (
            os.environ.get(_CUDA_FP32_GDN_DECODE_ENV, "0") in _TRUE_ENV_VALUES
            and use_qk_l2norm_in_kernel
            and initial_state is not None
            and query.shape[2] == 1
            and query.dtype == jnp.float32
            and key.dtype == jnp.float32
            and value.dtype == jnp.float32
            and g.dtype == jnp.float32
            and beta.dtype == jnp.float32
            and initial_state.dtype == jnp.float32
        ):
            from nanovllm_jax.kernels.cuda_fp32_ffi import (
                gdn_recurrent_decode_step_fp32,
            )

            return gdn_recurrent_decode_step_fp32(
                query,
                key,
                value,
                g,
                beta,
                initial_state,
            )

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

    def gated_delta_packed_decode(
        self,
        mixed_qkv: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        initial_state: jnp.ndarray,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        impl = _gdn_packed_decode_impl()
        packed_qkv_dtype = _gdn_packed_decode_qkv_activation_jnp_dtype()
        if impl == "off":
            raise RuntimeError(
                f"{_GDN_PACKED_DECODE_IMPL_ENV} is off; use gated_delta_decode"
            )

        if impl == "reference":
            from nanovllm_jax.kernels.gdn_fla import (
                prepare_gdn_packed_decode_reference_from_decay_inputs,
                gdn_packed_decode_reference_from_decay,
            )

            mixed_qkv, a, b, decay, dt_bias, initial_state = (
                prepare_gdn_packed_decode_reference_from_decay_inputs(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    decay=decay,
                    dt_bias=dt_bias,
                    state=initial_state,
                    qkv_dtype=packed_qkv_dtype,
                )
            )
            return gdn_packed_decode_reference_from_decay(
                mixed_qkv,
                a,
                b,
                decay,
                dt_bias,
                initial_state,
                qkv_dtype=packed_qkv_dtype,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if impl == "cuda_fp32":
            if not use_qk_l2norm_in_kernel:
                raise ValueError("CUDA FP32 packed GDN decode requires q/k l2norm")
            if packed_qkv_dtype == jnp.bfloat16:
                from nanovllm_jax.kernels.gdn_fla import (
                    prepare_gdn_packed_decode_reference_from_decay_inputs,
                )

                mixed_qkv, a, b, decay, dt_bias, initial_state = (
                    prepare_gdn_packed_decode_reference_from_decay_inputs(
                        mixed_qkv=mixed_qkv,
                        a=a,
                        b=b,
                        decay=decay,
                        dt_bias=dt_bias,
                        state=initial_state,
                        qkv_dtype=packed_qkv_dtype,
                    )
                )
                from nanovllm_jax.kernels.cuda_fp32_ffi import (
                    gdn_packed_decode_step_fp32_bf16qkv,
                )
                return gdn_packed_decode_step_fp32_bf16qkv(
                    mixed_qkv,
                    a,
                    b,
                    decay,
                    dt_bias,
                    initial_state,
                )

            for name, value_array in (
                ("mixed_qkv", mixed_qkv),
                ("a", a),
                ("b", b),
                ("decay", decay),
                ("dt_bias", dt_bias),
                ("initial_state", initial_state),
            ):
                if value_array.dtype != jnp.float32:
                    raise ValueError(f"{name} must be float32 for CUDA packed GDN decode")
            from nanovllm_jax.kernels.cuda_fp32_ffi import (
                gdn_packed_decode_step_fp32,
            )

            return gdn_packed_decode_step_fp32(
                mixed_qkv,
                a,
                b,
                jnp.log(decay),
                dt_bias,
                initial_state,
            )

        if impl == "triton_fla":
            pre_normalize_qk = _gdn_packed_decode_pre_normalize_qk()
            if pre_normalize_qk and use_qk_l2norm_in_kernel:
                # q/k are already normalized before entering the kernel. Leave the
                # argument False so triton avoids redundant reductions.
                use_qk_l2norm_in_kernel = False
            elif not use_qk_l2norm_in_kernel:
                raise ValueError(
                    "Triton FLA packed GDN decode requires q/k l2norm; "
                    "set NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK=1 to "
                    "pre-normalize outside the kernel"
                )
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_packed_decode_pre_normalize_qk,
                prepare_gdn_packed_decode_reference_from_decay_inputs,
                gdn_packed_decode_reference_from_decay,
            )

            if pre_normalize_qk:
                mixed_qkv = gdn_packed_decode_pre_normalize_qk(
                    mixed_qkv,
                    initial_state,
                )
            mixed_qkv, a, b, decay, dt_bias, initial_state = (
                prepare_gdn_packed_decode_reference_from_decay_inputs(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    decay=decay,
                    dt_bias=dt_bias,
                    state=initial_state,
                    qkv_dtype=jnp.bfloat16,
                )
            )
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_packed_decode_step_bf16,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                return gdn_packed_decode_reference_from_decay(
                    mixed_qkv,
                    a,
                    b,
                    decay,
                    dt_bias,
                    initial_state,
                    qkv_dtype=jnp.bfloat16,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )

            return gdn_packed_decode_step_bf16(
                mixed_qkv,
                a,
                b,
                decay,
                dt_bias,
                initial_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        raise AssertionError(f"Unhandled packed GDN decode implementation {impl!r}")


class KernelBackendPlaceholder(PureJAXBackend):
    """Explicit placeholder until GPU/TPU kernels are implemented."""

    def __init__(self, name: str, kernel_backend: KernelBackendStatus):
        super().__init__(kernel_backend=kernel_backend)
        self.name = name


def select_backend(name: str = "auto") -> InferenceBackend:
    """Select an inference backend.

    `auto` currently returns the pure JAX correctness backend even on GPU/TPU.
    That keeps semantics stable until kernel backends pass parity gates.
    """

    normalized = name.lower()
    if normalized == "auto":
        kernel_backend = select_kernel_backend()
        if (
            kernel_backend.requested not in {"auto", "pure_jax"}
            and not kernel_backend.external_kernels_enabled
        ):
            raise KernelBackendUnavailable(kernel_backend.reason)
        return PureJAXBackend(kernel_backend=kernel_backend)
    if normalized in {"pure_jax", "jax"}:
        return PureJAXBackend()
    if normalized in {"gpu", "cuda", "tpu"}:
        platform = jax.default_backend()
        if normalized == "gpu" and platform != "gpu":
            raise RuntimeError(f"Requested GPU backend, but JAX default backend is {platform!r}")
        if normalized == "tpu" and platform != "tpu":
            raise RuntimeError(f"Requested TPU backend, but JAX default backend is {platform!r}")
        kernel_backend = select_kernel_backend()
        if (
            kernel_backend.requested not in {"auto", "pure_jax"}
            and not kernel_backend.external_kernels_enabled
        ):
            raise KernelBackendUnavailable(kernel_backend.reason)
        return KernelBackendPlaceholder(normalized, kernel_backend)
    raise ValueError(f"Unknown backend {name!r}; expected auto, pure_jax, gpu, or tpu")
