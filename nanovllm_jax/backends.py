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
    if normalized in {"cuda", "cuda_fp32", "fast"}:
        return "cuda_fp32"
    raise ValueError(
        f"Unknown {_GDN_PACKED_DECODE_IMPL_ENV}={value!r}; "
        "expected off, reference, or cuda_fp32"
    )


def gdn_packed_decode_enabled() -> bool:
    return _gdn_packed_decode_impl() != "off"


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
        if impl == "off":
            raise RuntimeError(
                f"{_GDN_PACKED_DECODE_IMPL_ENV} is off; use gated_delta_decode"
            )

        if impl == "reference":
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_packed_decode_reference_from_decay,
            )

            return gdn_packed_decode_reference_from_decay(
                mixed_qkv,
                a,
                b,
                decay,
                dt_bias,
                initial_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if impl == "cuda_fp32":
            if not use_qk_l2norm_in_kernel:
                raise ValueError("CUDA FP32 packed GDN decode requires q/k l2norm")
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
