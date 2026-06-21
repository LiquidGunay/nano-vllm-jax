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
    paged_attention_prefill_packed,
    paged_attention_decode,
    compute_packed_slot_mapping,
    update_kv_cache,
)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}
_NHD_FULL_ATTN_CACHE_ENV = "NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE"
_FLASHINFER_KV_APPEND_ENV = "NANO_VLLM_JAX_FLASHINFER_KV_APPEND"
_ALLOW_LOCAL_CUDA_PROBES_ENV = "NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES"
_CUDA_FP32_KV_APPEND_ENV = "NANO_VLLM_JAX_CUDA_FP32_KV_APPEND"
_CUDA_FP32_DECODE_ATTN_ENV = "NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN"
_CUDA_FP32_GDN_DECODE_ENV = "NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE"
_FULL_ATTN_KV_CACHE_DTYPE_ENV = "NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE"
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
_GDN_PREFILL_FLA_VLLM_LIKE_ENV = "NANO_VLLM_JAX_GDN_PREFILL_FLA_VLLM_LIKE"
_GDN_PACKED_DECODE_QKV_DTYPE_ENV = "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE"
_GDN_PACKED_DECODE_MAX_BATCH_ENV = "NANO_VLLM_JAX_GDN_PACKED_DECODE_MAX_BATCH"
_GDN_DISABLE_FALLBACKS_ENV = "NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS"
_OFF_ENV_VALUES = {"", "0", "false", "no", "off", "none", "False"}


def _config_or_env_bool(config, attr: str, env_name: str, *, default: bool = False) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value.strip().lower() in _TRUE_ENV_VALUES
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return default


def _config_or_env_str(config, attr: str, env_name: str, *, default: str) -> str:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value.strip().lower()
    if config is not None and hasattr(config, attr):
        return str(getattr(config, attr) or default).strip().lower()
    return default


def _config_or_env_int_or_none(config, attr: str, env_name: str) -> int | None:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        value = env_value.strip()
        if not value:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    if config is not None and hasattr(config, attr):
        value = getattr(config, attr)
        if value is None:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _full_attention_kv_cache_dtype(default_dtype, config=None):
    value = _config_or_env_str(
        config,
        "full_attention_kv_cache_dtype",
        _FULL_ATTN_KV_CACHE_DTYPE_ENV,
        default="default",
    )
    if value == "default" or value in _OFF_ENV_VALUES:
        return default_dtype
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if value in {"fp16", "float16"}:
        return jnp.float16
    if value in {"fp32", "float32"}:
        return jnp.float32
    raise ValueError(
        f"Unknown {_FULL_ATTN_KV_CACHE_DTYPE_ENV}={value!r}; "
        "expected fp32, bf16, fp16, default, or off"
    )


def _full_attention_kv_append_impl(config=None) -> str:
    value = str(
        getattr(config, "full_attention_kv_append_impl", "reference")
        if config is not None
        else "reference"
    ).strip().lower()
    if value in _OFF_ENV_VALUES or value in {"reference", "jax", "pure_jax"}:
        return "reference"
    if value in {"flashinfer", "flashinfer_paged", "paged_flashinfer"}:
        return "flashinfer"
    raise ValueError(
        "Unknown full_attention_kv_append_impl="
        f"{value!r}; expected reference or flashinfer"
    )


def _full_attention_decode_impl(config=None) -> str:
    value = str(
        getattr(config, "full_attention_decode_impl", "reference")
        if config is not None
        else "reference"
    ).strip().lower()
    if value in _OFF_ENV_VALUES or value in {"reference", "jax", "pure_jax"}:
        return "reference"
    if value in {
        "triton",
        "triton_paged",
        "paged_triton",
        "triton_paged_decode",
    }:
        return "triton_paged"
    if value in {
        "triton_paged_fused_append",
        "triton_fused_append",
        "paged_triton_fused_append",
        "triton_paged_append",
    }:
        return "triton_paged_fused_append"
    if value in {
        "flashinfer",
        "flashinfer_paged",
        "paged_flashinfer",
        "flashinfer_paged_decode",
    }:
        return "flashinfer_paged"
    raise ValueError(
        "Unknown full_attention_decode_impl="
        f"{value!r}; expected reference, triton_paged, triton_paged_fused_append, "
        "or flashinfer_paged"
    )


def _full_attention_prefill_impl(config=None) -> str:
    value = str(
        getattr(config, "full_attention_prefill_impl", "reference")
        if config is not None
        else "reference"
    ).strip().lower()
    if value in _OFF_ENV_VALUES or value in {"reference", "jax", "pure_jax"}:
        return "reference"
    if value in {
        "triton",
        "triton_packed",
        "packed_triton",
        "triton_paged",
        "packed_paged_triton",
    }:
        return "triton_packed"
    raise ValueError(
        "Unknown full_attention_prefill_impl="
        f"{value!r}; expected reference or triton_packed"
    )


def _gdn_disable_fallbacks(config=None) -> bool:
    return _config_or_env_bool(
        config,
        "gdn_disable_fallbacks",
        _GDN_DISABLE_FALLBACKS_ENV,
    )


def gdn_disable_fallbacks_enabled(config=None) -> bool:
    """Return whether GDN kernel requests must fail instead of falling back."""

    return _gdn_disable_fallbacks(config)


def _raise_if_gdn_fallback_disabled(reason: str, config=None) -> None:
    if _gdn_disable_fallbacks(config):
        raise RuntimeError(
            f"{reason}; implicit GDN kernel fallbacks are disabled by "
            f"{_GDN_DISABLE_FALLBACKS_ENV}=1"
        )


def _allow_local_cuda_probes() -> bool:
    return (
        os.environ.get(_ALLOW_LOCAL_CUDA_PROBES_ENV, "0").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _require_local_cuda_probe_opt_in(reason: str) -> None:
    if not _allow_local_cuda_probes():
        raise RuntimeError(
            f"{reason} uses local CUDA/JAX FFI probe code. These probes are not "
            "a serving kernel direction; use Pallas/CuteDSL or borrowed/adapted "
            f"Triton kernels instead. Set {_ALLOW_LOCAL_CUDA_PROBES_ENV}=1 only "
            "for explicit diagnostics."
        )


def _gdn_packed_decode_impl(config=None) -> str:
    value = _config_or_env_str(
        config,
        "gdn_packed_decode_impl",
        _GDN_PACKED_DECODE_IMPL_ENV,
        default="off",
    )
    if value in _TRUE_ENV_VALUES:
        return "cuda_fp32"
    normalized = value.lower()
    if normalized in _OFF_ENV_VALUES:
        return "off"
    if normalized in {"reference", "jax", "pure_jax"}:
        return "reference"
    if normalized in {"triton_fla", "fla_triton", "jax_triton", "triton"}:
        return "triton_fla"
    if normalized in {
        "triton_fla_raw_gates",
        "triton_raw_gates",
        "fla_triton_raw_gates",
        "triton_fla_raw",
    }:
        return "triton_fla_raw_gates"
    if normalized in {
        "triton_fla_raw_gates_tail",
        "triton_raw_gates_tail",
        "fla_triton_raw_gates_tail",
        "triton_fla_raw_tail",
        "triton_fla_tail_fused",
    }:
        return "triton_fla_raw_gates_tail"
    if normalized in {
        "triton_fla_raw_gates_split_tail",
        "triton_raw_gates_split_tail",
        "fla_triton_raw_gates_split_tail",
        "triton_fla_split_tail",
    }:
        return "triton_fla_raw_gates_split_tail"
    if normalized in {
        "triton_fla_conv_raw_gates",
        "triton_conv_raw_gates",
        "fla_triton_conv_raw_gates",
        "triton_fla_conv",
    }:
        return "triton_fla_conv_raw_gates"
    if normalized in {
        "triton_fla_conv_raw_gates_tail",
        "triton_conv_raw_gates_tail",
        "fla_triton_conv_raw_gates_tail",
        "triton_fla_conv_tail",
        "triton_fla_conv_tail_fused",
    }:
        return "triton_fla_conv_raw_gates_tail"
    if normalized in {"cuda", "cuda_fp32", "fast"}:
        return "cuda_fp32"
    raise ValueError(
        f"Unknown {_GDN_PACKED_DECODE_IMPL_ENV}={value!r}; "
        "expected off, reference, triton_fla, triton_fla_raw_gates, "
        "triton_fla_raw_gates_tail, "
        "triton_fla_raw_gates_split_tail, "
        "triton_fla_conv_raw_gates, "
        "triton_fla_conv_raw_gates_tail, or cuda_fp32"
    )


def gdn_packed_decode_impl(config=None) -> str:
    """Return the normalized packed GDN decode implementation name."""

    return _gdn_packed_decode_impl(config)


def gdn_packed_decode_enabled(config=None) -> bool:
    return _gdn_packed_decode_impl(config) != "off"


def gdn_packed_decode_conv_enabled(config=None) -> bool:
    return _gdn_packed_decode_impl(config) in {
        "triton_fla_conv_raw_gates",
        "triton_fla_conv_raw_gates_tail",
    }


def gdn_packed_decode_tail_fused_enabled(config=None) -> bool:
    return _gdn_packed_decode_impl(config) in {
        "triton_fla_raw_gates_tail",
        "triton_fla_raw_gates_split_tail",
        "triton_fla_conv_raw_gates_tail",
    }


def gdn_packed_decode_max_batch(config=None) -> int | None:
    return _config_or_env_int_or_none(
        config,
        "gdn_packed_decode_max_batch",
        _GDN_PACKED_DECODE_MAX_BATCH_ENV,
    )


def _gdn_prefill_post_conv_impl(config=None) -> str:
    value = _config_or_env_str(
        config,
        "gdn_prefill_post_conv_impl",
        _GDN_PREFILL_POST_CONV_IMPL_ENV,
        default="off",
    )
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


def gdn_prefill_post_conv_enabled(config=None) -> bool:
    return _gdn_prefill_post_conv_impl(config) != "off"


def _normalize_gdn_prefill_dtype(value: str, env_name: str) -> str:
    normalized = value.lower()
    if normalized in _OFF_ENV_VALUES or normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(f"Unknown {env_name}={value!r}; expected fp32 or bf16")


def _gdn_prefill_qkv_activation_dtype(config=None) -> str:
    value = os.environ.get(_GDN_PREFILL_QKV_DTYPE_ENV)
    env_name = _GDN_PREFILL_QKV_DTYPE_ENV
    if value is None:
        value = os.environ.get(_GDN_PREFILL_ACT_DTYPE_ENV)
        env_name = _GDN_PREFILL_ACT_DTYPE_ENV
    if value is None and config is not None and hasattr(config, "gdn_prefill_qkv_dtype"):
        value = getattr(config, "gdn_prefill_qkv_dtype")
        env_name = "gdn_prefill_qkv_dtype"
    if value is None:
        value = "fp32"
    return _normalize_gdn_prefill_dtype(str(value).strip(), env_name)


def _gdn_packed_decode_qkv_activation_dtype(config=None) -> str:
    value = _config_or_env_str(
        config,
        "gdn_packed_decode_qkv_dtype",
        _GDN_PACKED_DECODE_QKV_DTYPE_ENV,
        default="fp32",
    )
    return _normalize_gdn_prefill_dtype(value, _GDN_PACKED_DECODE_QKV_DTYPE_ENV)


def _gdn_prefill_activation_dtype(config=None) -> str:
    return _gdn_prefill_qkv_activation_dtype(config)


def _gdn_prefill_qkv_activation_jnp_dtype(config=None) -> jnp.dtype:
    if _gdn_prefill_qkv_activation_dtype(config) == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _gdn_packed_decode_qkv_activation_jnp_dtype(config=None) -> jnp.dtype:
    if _gdn_packed_decode_qkv_activation_dtype(config) == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _gdn_packed_decode_pre_normalize_qk(config=None) -> bool:
    return _config_or_env_bool(
        config,
        "gdn_packed_decode_pre_normalize_qk",
        _GDN_PACKED_DECODE_PRENORMALIZE_QK_ENV,
    )


def _gdn_prefill_fla_vllm_like_enabled() -> bool:
    return (
        os.environ.get(_GDN_PREFILL_FLA_VLLM_LIKE_ENV, "0").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _gdn_prefill_post_conv_output_dtype(config=None) -> str:
    value = _config_or_env_str(
        config,
        "gdn_prefill_post_conv_output_dtype",
        _GDN_PREFILL_POST_CONV_OUTPUT_DTYPE_ENV,
        default="fp32",
    )
    return _normalize_gdn_prefill_dtype(
        value,
        _GDN_PREFILL_POST_CONV_OUTPUT_DTYPE_ENV,
    )


def _cast_gdn_prefill_activations(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    config=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    act_dtype = _gdn_prefill_qkv_activation_jnp_dtype(config)
    if act_dtype == jnp.bfloat16:
        return (
            query.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
        )
    return query, key, value


def _cast_gdn_prefill_post_conv_output(output: jnp.ndarray, config=None) -> jnp.ndarray:
    if _gdn_prefill_post_conv_output_dtype(config) == "bf16":
        return output.astype(jnp.bfloat16)
    return output


def _static_packed_gdn_chunk_metadata(
    *,
    row_count: int,
    token_bucket: int,
    chunk_size: int,
    max_row_tokens: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    if row_count <= 0:
        raise ValueError("packed GDN prefill requires at least one row")
    if token_bucket <= 0:
        raise ValueError("packed GDN prefill requires a positive token bucket")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    row_token_bucket = token_bucket
    if max_row_tokens is not None and max_row_tokens > 0:
        row_token_bucket = min(token_bucket, int(max_row_tokens))
    max_row_chunks = (row_token_bucket + chunk_size - 1) // chunk_size
    rows = jnp.repeat(
        jnp.arange(row_count, dtype=jnp.int32),
        max_row_chunks,
    )
    chunks = jnp.tile(
        jnp.arange(max_row_chunks, dtype=jnp.int32),
        row_count,
    )
    chunk_indices = jnp.stack((rows, chunks), axis=1)
    chunk_offsets = (
        jnp.arange(row_count + 1, dtype=jnp.int32) * jnp.int32(max_row_chunks)
    )
    return chunk_indices, chunk_offsets, max_row_chunks


def _prepare_packed_gdn_post_conv_inputs_from_decay(
    conv_out: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    decay: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    use_qk_l2norm_in_kernel: bool,
    config=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if conv_out.ndim != 3 or conv_out.shape[0] != 1:
        raise ValueError("packed conv_out must have shape [1, token_bucket, conv_dim]")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != 1 or b.shape[0] != 1:
        raise ValueError("packed a/b must have shape [1, token_bucket, value_heads]")
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must have shape [row_count + 1]")
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")

    _, token_bucket, conv_dim = conv_out.shape
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    expected_conv_dim = 2 * key_dim + value_dim
    if conv_dim != expected_conv_dim:
        raise ValueError(
            f"conv_out last dimension must be {expected_conv_dim}, got {conv_dim}"
        )
    if a.shape != (1, token_bucket, num_value_heads):
        raise ValueError("a must have shape [1, token_bucket, value_heads]")
    if b.shape != (1, token_bucket, num_value_heads):
        raise ValueError("b must have shape [1, token_bucket, value_heads]")
    if decay.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("decay and dt_bias must have shape [value_heads]")

    query = conv_out[:, :, :key_dim].reshape(
        1,
        token_bucket,
        num_key_heads,
        key_head_dim,
    )
    key = conv_out[:, :, key_dim : key_dim * 2].reshape(
        1,
        token_bucket,
        num_key_heads,
        key_head_dim,
    )
    value = conv_out[:, :, key_dim * 2 :].reshape(
        1,
        token_bucket,
        num_value_heads,
        value_head_dim,
    )
    beta = jax.nn.sigmoid(b)
    gate = -decay * jax.nn.softplus(a + dt_bias)

    heads_per_key = num_value_heads // num_key_heads
    if heads_per_key > 1:
        query = jnp.repeat(query, heads_per_key, axis=2)
        key = jnp.repeat(key, heads_per_key, axis=2)

    if use_qk_l2norm_in_kernel:
        from nanovllm_jax.layers import l2norm

        query = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6)
        key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)

    valid = (
        jnp.arange(token_bucket, dtype=jnp.int32)
        < query_start_loc[-1].astype(jnp.int32)
    )
    query = jnp.where(valid[None, :, None, None], query, 0.0)
    key = jnp.where(valid[None, :, None, None], key, 0.0)
    value = jnp.where(valid[None, :, None, None], value, 0.0)
    gate = jnp.where(valid[None, :, None], gate, 0.0)
    beta = jnp.where(valid[None, :, None], beta, 0.0)

    qkv_dtype = _gdn_prefill_qkv_activation_jnp_dtype(config)
    packed_query = query.reshape(token_bucket, num_value_heads, key_head_dim)
    packed_key = key.reshape(token_bucket, num_value_heads, key_head_dim)
    packed_value = value.reshape(token_bucket, num_value_heads, value_head_dim)
    packed_gate = gate.reshape(token_bucket, num_value_heads)
    packed_beta = beta.reshape(token_bucket, num_value_heads)
    return (
        packed_query.astype(qkv_dtype),
        packed_key.astype(qkv_dtype),
        packed_value.astype(qkv_dtype),
        packed_gate.astype(jnp.float32),
        packed_beta.astype(jnp.float32),
        valid,
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
        query_start_loc: jnp.ndarray | None = None,
        num_prefill_tokens: int | None = None,
        num_decode_tokens: int | None = None,
        token_row_ids: jnp.ndarray | None = None,
        max_query_len: int | None = None,
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

    def write_kv_and_attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> tuple[KVCacheStorage, jnp.ndarray]:
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

    def gated_delta_packed_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
        max_row_tokens: int | None = None,
        return_prefix_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    def gated_delta_conv_packed_decode(
        self,
        mixed_qkv: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        conv_state: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_bias: jnp.ndarray | None,
        recurrent_state: jnp.ndarray,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_conv_packed_projection_decode(
        self,
        packed_proj: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        conv_state: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_bias: jnp.ndarray | None,
        recurrent_state: jnp.ndarray,
        *,
        qkv_dim: int,
        use_qk_l2norm_in_kernel: bool,
        norm_weight: jnp.ndarray | None = None,
        rms_norm_eps: float = 1.0e-6,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_conv_packed_projection_decode_state_pool(
        self,
        packed_proj: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        conv_state_pool: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_bias: jnp.ndarray | None,
        recurrent_state_pool: jnp.ndarray,
        *,
        qkv_dim: int,
        linear_layer_idx: int,
        use_qk_l2norm_in_kernel: bool,
        norm_weight: jnp.ndarray,
        valid_rows: jnp.ndarray | None = None,
        rms_norm_eps: float = 1.0e-6,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...


class PureJAXBackend:
    """Reference backend implemented with ordinary JAX operations."""

    name = "pure_jax"

    def __init__(self, kernel_backend: KernelBackendStatus | None = None, config=None):
        self.kernel_backend = kernel_backend or select_kernel_backend("pure_jax")
        self.config = config

    def allocate_kv_cache(
        self,
        spec: KVCacheSpec,
        max_seqs: int,
        max_blocks_per_seq: int,
    ) -> KVCacheStorage:
        cache_dtype = _full_attention_kv_cache_dtype(spec.dtype, self.config)
        capped_spec = replace(
            spec,
            dtype=cache_dtype,
            num_blocks=cap_num_kv_cache_blocks(replace(spec, dtype=cache_dtype)),
        )
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
        cache_dtype = _full_attention_kv_cache_dtype(spec.dtype, self.config)
        return init_full_attention_nhd_kv_cache(
            spec=replace(
                spec,
                dtype=cache_dtype,
                num_blocks=cap_num_kv_cache_blocks(replace(spec, dtype=cache_dtype)),
            ),
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
        token_row_ids: jnp.ndarray | None = None,
        max_query_len: int | None = None,
    ) -> AttentionMetadata:
        if positions.ndim != 2:
            raise ValueError("positions must be a 2D tensor [batch, query_len]")
        if block_tables.ndim != 2:
            raise ValueError("block_tables must be a 2D tensor [batch, max_blocks_per_seq]")
        packed_prefill = token_row_ids is not None
        if packed_prefill:
            if not is_prefill:
                raise ValueError("token_row_ids are only supported for packed prefill")
            if token_row_ids.shape != positions.shape:
                raise ValueError("token_row_ids and positions must have matching shapes")
            if seq_lens.shape[0] != block_tables.shape[0]:
                raise ValueError("seq_lens shape must align with packed prefill rows")
            slot_mapping = compute_packed_slot_mapping(
                positions=positions,
                block_table=block_tables,
                token_row_ids=token_row_ids,
                block_size=block_size,
            )
        else:
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
            token_row_ids=token_row_ids,
            max_query_len=max_query_len if max_query_len is not None else query_len,
        )

    def write_kv(
        self,
        layer_id: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
    ) -> KVCacheStorage:
        kv_append_impl = _full_attention_kv_append_impl(self.config)
        flashinfer_kv_append_requested = (
            kv_append_impl == "flashinfer"
            or os.environ.get(_FLASHINFER_KV_APPEND_ENV, "0") in _TRUE_ENV_VALUES
        )
        if os.environ.get(_CUDA_FP32_KV_APPEND_ENV, "0") in _TRUE_ENV_VALUES:
            _require_local_cuda_probe_opt_in("CUDA FP32 KV append")
            if flashinfer_kv_append_requested:
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

        if flashinfer_kv_append_requested:
            if metadata.token_row_ids is not None:
                raise ValueError(
                    "FlashInfer KV append currently supports rectangular scheduled "
                    "K/V tensors only; packed prefill must use the reference append "
                    "or a decode path that owns its append boundary"
                )
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

        if metadata.token_row_ids is not None:
            actual_tokens = metadata.query_start_loc[-1].astype(jnp.int32)
            valid_mask = jnp.arange(metadata.slot_mapping.size, dtype=jnp.int32).reshape(
                metadata.slot_mapping.shape
            ) < actual_tokens
        else:
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
            prefill_impl = _full_attention_prefill_impl(self.config)
            if metadata.token_row_ids is not None:
                if metadata.positions is None:
                    raise ValueError("metadata.positions is required for packed prefill attention")
                return paged_attention_prefill_packed(
                    query=query,
                    k_cache=cache.k_cache,
                    v_cache=cache.v_cache,
                    block_table=metadata.block_tables,
                    kv_lens=metadata.seq_lens,
                    positions=metadata.positions,
                    token_row_ids=metadata.token_row_ids,
                    query_start_loc=metadata.query_start_loc,
                    block_size=block_size,
                    scale=scale,
                    num_key_value_groups=num_key_value_groups,
                    layer_idx=layer_id,
                    max_query_len=metadata.max_query_len,
                    use_triton=prefill_impl == "triton_packed",
                )
            if prefill_impl != "reference":
                raise ValueError(
                    "full_attention.prefill_impl=triton_packed requires packed prefill "
                    "metadata with token_row_ids"
                )
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
            _require_local_cuda_probe_opt_in("CUDA FP32 decode attention")
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
        decode_impl = _full_attention_decode_impl(self.config)
        if decode_impl == "triton_paged":
            if query.shape[1] != 1:
                raise ValueError(
                    "full_attention.decode_impl=triton_paged supports only "
                    "width-1 decode attention"
                )
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "full_attention.decode_impl=triton_paged requires cache "
                    "shape [num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            from nanovllm_jax.kernels.full_attention_triton import (
                paged_decode_attention_triton,
            )

            return paged_decode_attention_triton(
                query=query,
                k_cache_layer=cache.k_cache[layer_id],
                v_cache_layer=cache.v_cache[layer_id],
                block_table=metadata.block_tables,
                seq_lens=metadata.seq_lens,
                block_size=block_size,
                scale=scale,
                num_key_value_groups=num_key_value_groups,
            )
        if decode_impl == "flashinfer_paged" and query.shape[1] == 1:
            if query.shape[1] != 1:
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged supports only "
                    "width-1 decode attention"
                )
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged requires cache "
                    "shape [num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            if cache.k_cache.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged requires BF16/FP16 "
                    "NHD KV cache; set full_attention_kv_cache_dtype to bf16 or fp16"
                )
            from nanovllm_jax.kernels.flashinfer_ffi import (
                paged_decode_attention_gqa_nhd,
            )
            from nanovllm_jax.kernels.paged_attention import (
                dense_block_tables_to_kv_indptr,
                kv_last_page_len_from_seq_lens,
            )

            kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(
                metadata.block_tables,
            )
            query_for_kernel = query[:, 0].astype(cache.k_cache.dtype)
            out = paged_decode_attention_gqa_nhd(
                query_for_kernel,
                cache.k_cache[layer_id],
                cache.v_cache[layer_id],
                kv_indptr,
                kv_indices,
                kv_last_page_len_from_seq_lens(metadata.seq_lens, block_size),
                scale=scale,
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

    def write_kv_and_attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> tuple[KVCacheStorage, jnp.ndarray]:
        decode_impl = _full_attention_decode_impl(self.config)
        if not is_prefill and decode_impl == "flashinfer_paged" and query.shape[1] > 1:
            if metadata.positions is None:
                raise ValueError("metadata.positions is required for FlashInfer decode attention")
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged requires cache shape "
                    "[num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            if cache.k_cache.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged requires BF16/FP16 "
                    "NHD KV cache; set full_attention_kv_cache_dtype to bf16 or fp16"
                )
            from nanovllm_jax.kernels.flashinfer_ffi import (
                paged_decode_attention_with_kv_append_gqa_nhd,
            )
            from nanovllm_jax.kernels.paged_attention import (
                dense_block_tables_to_kv_indptr,
                kv_last_page_len_from_seq_lens,
            )

            kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(
                metadata.block_tables,
            )
            width = int(query.shape[1])
            k_cache = cache.k_cache
            v_cache = cache.v_cache
            outputs = []
            for token_idx in range(width):
                seq_lens_step = metadata.seq_lens - jnp.asarray(
                    width - 1 - token_idx,
                    dtype=metadata.seq_lens.dtype,
                )
                out, k_cache, v_cache = paged_decode_attention_with_kv_append_gqa_nhd(
                    query[:, token_idx].astype(cache.k_cache.dtype),
                    k[:, token_idx].astype(cache.k_cache.dtype),
                    v[:, token_idx].astype(cache.v_cache.dtype),
                    k_cache,
                    v_cache,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len_from_seq_lens(seq_lens_step, block_size),
                    metadata.positions[:, token_idx].astype(jnp.int32),
                    layer_id=layer_id,
                    scale=scale,
                )
                outputs.append(
                    out.reshape(query.shape[0], 1, query.shape[2] * query.shape[3])
                )
            return KVCacheStorage(k_cache, v_cache), jnp.concatenate(outputs, axis=1)
        if (
            not is_prefill
            and decode_impl == "flashinfer_paged"
            and query.shape[1] == 1
            and k.shape[1] == 1
            and v.shape[1] == 1
        ):
            if metadata.positions is None:
                raise ValueError("metadata.positions is required for FlashInfer decode attention")
            if query.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged supports only width-1 decode"
                )
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged requires cache shape "
                    "[num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            if cache.k_cache.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
                raise ValueError(
                    "full_attention.decode_impl=flashinfer_paged requires BF16/FP16 "
                    "NHD KV cache; set full_attention_kv_cache_dtype to bf16 or fp16"
                )
            from nanovllm_jax.kernels.flashinfer_ffi import (
                paged_decode_attention_with_kv_append_gqa_nhd,
            )
            from nanovllm_jax.kernels.paged_attention import (
                dense_block_tables_to_kv_indptr,
                kv_last_page_len_from_seq_lens,
            )

            kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(
                metadata.block_tables,
            )
            out, k_cache, v_cache = paged_decode_attention_with_kv_append_gqa_nhd(
                query[:, 0].astype(cache.k_cache.dtype),
                k[:, 0].astype(cache.k_cache.dtype),
                v[:, 0].astype(cache.v_cache.dtype),
                cache.k_cache,
                cache.v_cache,
                kv_indptr,
                kv_indices,
                kv_last_page_len_from_seq_lens(metadata.seq_lens, block_size),
                metadata.positions.reshape(query.shape[0]).astype(jnp.int32),
                layer_id=layer_id,
                scale=scale,
            )
            return (
                KVCacheStorage(k_cache, v_cache),
                out.reshape(query.shape[0], 1, query.shape[2] * query.shape[3]),
            )
        if not is_prefill and decode_impl == "triton_paged_fused_append":
            if metadata.positions is None:
                raise ValueError("metadata.positions is required for fused decode attention")
            if query.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
                raise ValueError(
                    "full_attention.decode_impl=triton_paged_fused_append "
                    "supports only width-1 decode"
                )
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError(
                    "full_attention.decode_impl=triton_paged_fused_append requires "
                    "cache shape [num_layers, num_pages, page_size, num_kv_heads, head_dim]"
                )
            from nanovllm_jax.kernels.full_attention_triton import (
                paged_decode_attention_with_kv_append_triton,
            )

            out, k_cache, v_cache = paged_decode_attention_with_kv_append_triton(
                query=query,
                new_k=k,
                new_v=v,
                k_cache=cache.k_cache,
                v_cache=cache.v_cache,
                block_table=metadata.block_tables,
                seq_lens=metadata.seq_lens,
                positions=metadata.positions,
                layer_id=layer_id,
                block_size=block_size,
                scale=scale,
                num_key_value_groups=num_key_value_groups,
            )
            return KVCacheStorage(k_cache, v_cache), out

        cache_storage = self.write_kv(
            layer_id=layer_id,
            k=k,
            v=v,
            cache=cache,
            metadata=metadata,
        )
        out = self.attention(
            layer_id=layer_id,
            query=query,
            cache=cache_storage,
            metadata=metadata,
            block_size=block_size,
            scale=scale,
            num_key_value_groups=num_key_value_groups,
            is_prefill=is_prefill,
        )
        return cache_storage, out

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
        impl = _gdn_prefill_post_conv_impl(self.config)
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
                vllm_like=_gdn_prefill_fla_vllm_like_enabled(),
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
                _raise_if_gdn_fallback_disabled(
                    "triton_fla_packed requires concrete seq_lens for packed prefill"
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

            if gdn_fla_chunk_gated_delta_rule_packed_triton is None:
                _raise_if_gdn_fallback_disabled(
                    "Triton FLA packed prefill kernel is unavailable"
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
                _raise_if_gdn_fallback_disabled(
                    "triton_fla_wrapper requires concrete seq_lens for packed prefill"
                )
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
                _raise_if_gdn_fallback_disabled(
                    "Triton FLA padded prefill kernel is unavailable"
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
                _raise_if_gdn_fallback_disabled(
                    "Triton BF16 GDN post-conv prep kernel is unavailable"
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
            _require_local_cuda_probe_opt_in("CUDA FP32 GDN chunk32 prefill")
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
                _raise_if_gdn_fallback_disabled(
                    "CUDA FP32 chunk32 prefill kernel cannot handle this shape"
                )
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
            _require_local_cuda_probe_opt_in("CUDA FP32 GDN prep/prefill")
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
                _raise_if_gdn_fallback_disabled(
                    "CUDA FP32 prep+prefill kernel cannot handle this shape"
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

    def gated_delta_packed_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
        max_row_tokens: int | None = None,
        return_prefix_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        impl = _gdn_prefill_post_conv_impl(self.config)
        if impl == "off":
            raise RuntimeError(
                f"{_GDN_PREFILL_POST_CONV_IMPL_ENV} is off; use reference packed prefill"
            )

        token_bucket = int(conv_out.shape[1])
        row_count = int(query_start_loc.shape[0]) - 1
        if initial_state is None:
            initial_state = jnp.zeros(
                (
                    row_count,
                    num_value_heads,
                    value_head_dim,
                    key_head_dim,
                ),
                dtype=jnp.float32,
            )

        (
            packed_query,
            packed_key,
            packed_value,
            packed_gate,
            packed_beta,
            valid_tokens,
        ) = _prepare_packed_gdn_post_conv_inputs_from_decay(
            conv_out,
            a,
            b,
            decay,
            dt_bias,
            query_start_loc,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            config=self.config,
        )

        def reference_scan() -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            token_positions = jnp.arange(token_bucket, dtype=jnp.int32)
            cu = query_start_loc.astype(jnp.int32)
            row_ids = jnp.sum(
                token_positions[:, None] >= cu[1:][None, :],
                axis=1,
            ).astype(jnp.int32)
            safe_rows = jnp.clip(row_ids, 0, row_count - 1)
            query_tokens = packed_query.astype(jnp.float32) * (
                1.0 / jnp.sqrt(key_head_dim)
            )
            key_tokens = packed_key.astype(jnp.float32)
            value_tokens = packed_value.astype(jnp.float32)
            gate_tokens = packed_gate.astype(jnp.float32)
            beta_tokens = packed_beta.astype(jnp.float32)

            def recurrent_scan(state, inputs):
                row, valid, q_t, k_t, v_t, gate_t, beta_t = inputs
                previous_row_state = state[row]
                decay_t = jnp.exp(gate_t)[:, None, None]
                decayed_state = previous_row_state * decay_t
                kv_mem = jnp.einsum("hvk,hk->hv", decayed_state, k_t)
                delta = (v_t - kv_mem) * beta_t[:, None]
                next_row_state = decayed_state + delta[:, :, None] * k_t[:, None, :]
                out_t = jnp.einsum("hvk,hk->hv", next_row_state, q_t)
                next_row_state = jnp.where(valid, next_row_state, previous_row_state)
                state = state.at[row].set(next_row_state)
                out_t = jnp.where(valid, out_t, jnp.zeros_like(out_t))
                return state, (out_t, next_row_state)

            final_state, (output_tokens, prefix_states) = jax.lax.scan(
                recurrent_scan,
                initial_state.astype(jnp.float32),
                (
                    safe_rows,
                    valid_tokens,
                    query_tokens,
                    key_tokens,
                    value_tokens,
                    gate_tokens,
                    beta_tokens,
                ),
            )
            output = _cast_gdn_prefill_post_conv_output(output_tokens, self.config)
            if return_prefix_state:
                return output[None, :, :, :], final_state, prefix_states
            return output[None, :, :, :], final_state

        if impl in {
            "reference",
            "reference_fla_chunk32",
            "reference_fla_packed",
        }:
            return reference_scan()

        if impl not in {
            "triton_fla_packed",
            "triton_fla_wrapper",
            "triton_fla_padded",
            "triton_fla_prep_bf16",
        }:
            _raise_if_gdn_fallback_disabled(
                f"{impl!r} does not support packed GDN prefill ABI",
                self.config,
            )
            return reference_scan()

        try:
            from nanovllm_jax.kernels.gdn_fla_triton import (
                gdn_fla_chunk_gated_delta_rule_packed_triton,
            )
        except (ImportError, ModuleNotFoundError, AttributeError):
            gdn_fla_chunk_gated_delta_rule_packed_triton = None

        if gdn_fla_chunk_gated_delta_rule_packed_triton is None:
            _raise_if_gdn_fallback_disabled(
                "Triton FLA packed prefill kernel is unavailable",
                self.config,
            )
            return reference_scan()

        if return_prefix_state:
            if max_row_tokens is None:
                _raise_if_gdn_fallback_disabled(
                    "Packed-prefix GDN verifier requires static max_row_tokens",
                    self.config,
                )
                return reference_scan()
            if int(max_row_tokens) > 16:
                _raise_if_gdn_fallback_disabled(
                    "Packed-prefix GDN verifier requires max_row_tokens <= 16",
                    self.config,
                )
                return reference_scan()
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_packed_prefix_state_triton,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                gdn_packed_prefix_state_triton = None
            if gdn_packed_prefix_state_triton is None:
                _raise_if_gdn_fallback_disabled(
                    "Tiny packed-prefix GDN kernel is unavailable",
                    self.config,
                )
                return reference_scan()
            output, final_state, prefix_states = gdn_packed_prefix_state_triton(
                packed_query,
                packed_key,
                packed_value,
                packed_gate,
                packed_beta,
                query_start_loc.astype(jnp.int32),
                initial_state.astype(jnp.float32),
                max_row_tokens=int(max_row_tokens),
            )
            output = jnp.where(valid_tokens[:, None, None], output, 0.0)
            output = _cast_gdn_prefill_post_conv_output(output, self.config)
            prefix_states = jnp.where(
                valid_tokens[:, None, None, None],
                prefix_states,
                0.0,
            )
            return output[None, :, :, :], final_state, prefix_states

        chunk_indices, chunk_offsets, max_row_chunks = (
            _static_packed_gdn_chunk_metadata(
                row_count=row_count,
                token_bucket=token_bucket,
                chunk_size=chunk_size,
                max_row_tokens=max_row_tokens,
            )
        )
        output, final_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
            packed_query,
            packed_key,
            packed_value,
            packed_gate,
            packed_beta,
            query_start_loc.astype(jnp.int32),
            initial_state.astype(jnp.float32),
            chunk_size=chunk_size,
            # Q/K normalization is applied above when requested.
            use_qk_l2norm_in_kernel=False,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            max_row_chunks=max_row_chunks,
        )
        output = jnp.where(valid_tokens[:, None, None], output, 0.0)
        output = _cast_gdn_prefill_post_conv_output(output, self.config)
        return output[None, :, :, :], final_state

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
        cuda_decode_requested = (
            os.environ.get(_CUDA_FP32_GDN_DECODE_ENV, "0") in _TRUE_ENV_VALUES
        )
        cuda_decode_compatible = (
            use_qk_l2norm_in_kernel
            and initial_state is not None
            and query.shape[2] == 1
            and query.dtype == jnp.float32
            and key.dtype == jnp.float32
            and value.dtype == jnp.float32
            and g.dtype == jnp.float32
            and beta.dtype == jnp.float32
            and initial_state.dtype == jnp.float32
        )
        if cuda_decode_requested:
            _require_local_cuda_probe_opt_in("CUDA FP32 recurrent GDN decode")
            if not cuda_decode_compatible:
                _raise_if_gdn_fallback_disabled(
                    "CUDA FP32 recurrent GDN decode kernel cannot handle this shape or dtype"
                )
            else:
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
        *,
        z: jnp.ndarray | None = None,
        norm_weight: jnp.ndarray | None = None,
        rms_norm_eps: float = 1.0e-6,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        impl = _gdn_packed_decode_impl(self.config)
        packed_qkv_dtype = _gdn_packed_decode_qkv_activation_jnp_dtype(self.config)
        if impl == "off":
            raise RuntimeError(
                f"{_GDN_PACKED_DECODE_IMPL_ENV} is off; use gated_delta_decode"
            )
        tail_fused = impl in {
            "triton_fla_raw_gates_tail",
            "triton_fla_raw_gates_split_tail",
        }
        if tail_fused and (z is None or norm_weight is None):
            raise ValueError(
                "tail-fused raw-gate packed GDN decode requires z and norm_weight"
            )

        def tail_from_core(core_out: jnp.ndarray) -> jnp.ndarray:
            assert z is not None
            assert norm_weight is not None
            batch = core_out.shape[0]
            value_heads = initial_state.shape[1]
            value_dim = initial_state.shape[2]
            tail = core_out.transpose(0, 2, 1, 3).reshape(
                batch,
                value_heads,
                value_dim,
            )
            tail = tail.astype(jnp.float32)
            variance = jnp.mean(jnp.square(tail), axis=-1, keepdims=True)
            tail = tail * jax.lax.rsqrt(variance + float(rms_norm_eps))
            tail = tail * norm_weight.astype(jnp.float32)
            tail = tail * jax.nn.silu(
                z.astype(jnp.bfloat16)
                .astype(jnp.float32)
                .reshape(batch, value_heads, value_dim)
            )
            return tail.reshape(batch, 1, value_heads * value_dim)

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
            _require_local_cuda_probe_opt_in("CUDA FP32 packed GDN decode")
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

        if impl in {
            "triton_fla",
            "triton_fla_raw_gates",
            "triton_fla_raw_gates_tail",
            "triton_fla_raw_gates_split_tail",
        }:
            pre_normalize_qk = _gdn_packed_decode_pre_normalize_qk(self.config)
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
                if impl == "triton_fla_raw_gates_tail":
                    from nanovllm_jax.kernels.gdn_fla_triton import (
                        gdn_packed_decode_step_bf16_raw_gates_tail,
                    )
                elif impl == "triton_fla_raw_gates_split_tail":
                    from nanovllm_jax.kernels.gdn_fla_triton import (
                        gdn_decode_tail_rms_silu_bf16,
                        gdn_packed_decode_step_bf16_raw_gates,
                    )
                elif impl == "triton_fla_raw_gates":
                    from nanovllm_jax.kernels.gdn_fla_triton import (
                        gdn_packed_decode_step_bf16_raw_gates,
                    )
                else:
                    from nanovllm_jax.kernels.gdn_fla_triton import (
                        gdn_packed_decode_step_bf16,
                    )
            except (ImportError, ModuleNotFoundError, AttributeError):
                _raise_if_gdn_fallback_disabled(
                    "Triton FLA packed decode kernel is unavailable",
                    self.config,
                )
                core_out, new_state = gdn_packed_decode_reference_from_decay(
                    mixed_qkv,
                    a,
                    b,
                    decay,
                    dt_bias,
                    initial_state,
                    qkv_dtype=jnp.bfloat16,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                if tail_fused:
                    return tail_from_core(core_out), new_state
                return core_out, new_state

            if impl == "triton_fla_raw_gates_tail":
                assert z is not None
                assert norm_weight is not None
                return gdn_packed_decode_step_bf16_raw_gates_tail(
                    mixed_qkv,
                    a,
                    b,
                    decay,
                    dt_bias,
                    initial_state,
                    z.astype(jnp.bfloat16),
                    norm_weight.astype(jnp.float32),
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    rms_norm_eps=float(rms_norm_eps),
                )

            if impl == "triton_fla_raw_gates_split_tail":
                assert z is not None
                assert norm_weight is not None
                core_out, new_state = gdn_packed_decode_step_bf16_raw_gates(
                    mixed_qkv,
                    a,
                    b,
                    decay,
                    dt_bias,
                    initial_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                return (
                    gdn_decode_tail_rms_silu_bf16(
                        core_out,
                        z.astype(jnp.bfloat16),
                        norm_weight.astype(jnp.float32),
                        rms_norm_eps=float(rms_norm_eps),
                    ),
                    new_state,
                )

            if impl == "triton_fla_raw_gates":
                return gdn_packed_decode_step_bf16_raw_gates(
                    mixed_qkv,
                    a,
                    b,
                    decay,
                    dt_bias,
                    initial_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )

            gate = -decay * jax.nn.softplus(a + dt_bias)
            beta = jax.nn.sigmoid(b)
            return gdn_packed_decode_step_bf16(
                mixed_qkv,
                gate,
                beta,
                initial_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        raise AssertionError(f"Unhandled packed GDN decode implementation {impl!r}")

    def gated_delta_conv_packed_decode(
        self,
        mixed_qkv: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        conv_state: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_bias: jnp.ndarray | None,
        recurrent_state: jnp.ndarray,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        impl = _gdn_packed_decode_impl(self.config)
        if impl not in {"triton_fla_conv_raw_gates", "triton_fla_conv_raw_gates_tail"}:
            raise RuntimeError(
                "gated_delta_conv_packed_decode is only enabled by "
                f"{_GDN_PACKED_DECODE_IMPL_ENV}=triton_fla_conv_raw_gates or "
                "triton_fla_conv_raw_gates_tail"
            )
        if not use_qk_l2norm_in_kernel:
            raise ValueError("conv packed GDN decode requires q/k l2norm")

        def reference() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            from nanovllm_jax.layers import causal_conv1d_update
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_packed_decode_pre_normalize_qk,
                gdn_packed_decode_reference_from_decay,
            )

            conv_out_t, new_conv_state = causal_conv1d_update(
                mixed_qkv[:, :, None],
                conv_state,
                conv_weight,
                conv_bias,
                "silu",
            )
            conv_out = gdn_packed_decode_pre_normalize_qk(
                conv_out_t[:, :, 0].astype(jnp.float32),
                recurrent_state,
            )
            out, new_recurrent_state = gdn_packed_decode_reference_from_decay(
                conv_out,
                a,
                b,
                decay,
                dt_bias,
                recurrent_state,
                qkv_dtype=jnp.bfloat16,
                use_qk_l2norm_in_kernel=False,
            )
            return out, new_conv_state, new_recurrent_state

        try:
            from nanovllm_jax.kernels.gdn_fla_triton import (
                gdn_conv_packed_decode_step_bf16_raw_gates,
            )
        except (ImportError, ModuleNotFoundError, AttributeError):
            _raise_if_gdn_fallback_disabled(
                "Triton FLA conv+packed decode kernel is unavailable",
                self.config,
            )
            return reference()

        if conv_bias is None:
            conv_bias = jnp.zeros((conv_weight.shape[0],), dtype=conv_weight.dtype)
        try:
            return gdn_conv_packed_decode_step_bf16_raw_gates(
                mixed_qkv.astype(jnp.bfloat16),
                a.astype(jnp.float32),
                b.astype(jnp.float32),
                decay.astype(jnp.float32),
                dt_bias.astype(jnp.float32),
                conv_state.astype(jnp.float32),
                conv_weight.astype(jnp.float32),
                conv_bias.astype(jnp.float32),
                recurrent_state.astype(jnp.float32),
            )
        except Exception:
            _raise_if_gdn_fallback_disabled(
                "Triton FLA conv+packed decode kernel cannot handle this shape",
                self.config,
            )
            return reference()

    def gated_delta_conv_packed_projection_decode(
        self,
        packed_proj: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        conv_state: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_bias: jnp.ndarray | None,
        recurrent_state: jnp.ndarray,
        *,
        qkv_dim: int,
        use_qk_l2norm_in_kernel: bool,
        norm_weight: jnp.ndarray | None = None,
        rms_norm_eps: float = 1.0e-6,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        impl = _gdn_packed_decode_impl(self.config)
        if impl not in {"triton_fla_conv_raw_gates", "triton_fla_conv_raw_gates_tail"}:
            raise RuntimeError(
                "gated_delta_conv_packed_projection_decode is only enabled by "
                f"{_GDN_PACKED_DECODE_IMPL_ENV}=triton_fla_conv_raw_gates or "
                "triton_fla_conv_raw_gates_tail"
            )
        if not use_qk_l2norm_in_kernel:
            raise ValueError("conv packed GDN decode requires q/k l2norm")
        tail_fused = impl == "triton_fla_conv_raw_gates_tail"
        if tail_fused and norm_weight is None:
            raise ValueError("tail-fused packed-projection GDN decode requires norm_weight")

        def reference_core() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            from nanovllm_jax.layers import causal_conv1d_update
            from nanovllm_jax.kernels.gdn_fla import (
                gdn_packed_decode_pre_normalize_qk,
                gdn_packed_decode_reference_from_decay,
            )

            mixed_qkv = packed_proj[:, :qkv_dim]
            value_heads = recurrent_state.shape[1]
            a = packed_proj[:, qkv_dim : qkv_dim + value_heads].astype(jnp.float32)
            b = packed_proj[:, qkv_dim + value_heads : qkv_dim + 2 * value_heads].astype(jnp.float32)
            conv_out_t, new_conv_state = causal_conv1d_update(
                mixed_qkv[:, :, None],
                conv_state,
                conv_weight,
                conv_bias,
                "silu",
            )
            conv_out = gdn_packed_decode_pre_normalize_qk(
                conv_out_t[:, :, 0].astype(jnp.float32),
                recurrent_state,
            )
            out, new_recurrent_state = gdn_packed_decode_reference_from_decay(
                conv_out,
                a,
                b,
                decay,
                dt_bias,
                recurrent_state,
                qkv_dtype=jnp.bfloat16,
                use_qk_l2norm_in_kernel=False,
            )
            return out, new_conv_state, new_recurrent_state

        def reference() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            core_out, new_conv_state, new_recurrent_state = reference_core()
            if not tail_fused:
                return core_out, new_conv_state, new_recurrent_state
            assert norm_weight is not None
            batch = packed_proj.shape[0]
            value_heads = recurrent_state.shape[1]
            value_dim = recurrent_state.shape[2]
            z_offset = qkv_dim + 2 * value_heads
            if packed_proj.shape[1] < z_offset + value_heads * value_dim:
                raise ValueError("tail-fused packed-projection GDN decode requires full z region")
            z = packed_proj[:, z_offset : z_offset + value_heads * value_dim].reshape(
                batch,
                value_heads,
                value_dim,
            )
            tail = core_out.transpose(0, 2, 1, 3).reshape(batch, value_heads, value_dim)
            tail = tail.astype(jnp.float32)
            variance = jnp.mean(jnp.square(tail), axis=-1, keepdims=True)
            tail = tail * jax.lax.rsqrt(variance + float(rms_norm_eps))
            tail = tail * norm_weight.astype(jnp.float32)
            tail = tail * jax.nn.silu(z.astype(jnp.float32))
            return tail.reshape(batch, 1, value_heads * value_dim), new_conv_state, new_recurrent_state

        try:
            if tail_fused:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_conv_packed_projection_decode_step_bf16_raw_gates_tail,
                )
            else:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_conv_packed_projection_decode_step_bf16_raw_gates,
                )
        except (ImportError, ModuleNotFoundError, AttributeError):
            _raise_if_gdn_fallback_disabled(
                "Triton FLA packed-projection conv+decode kernel is unavailable",
                self.config,
            )
            return reference()

        if conv_bias is None:
            conv_bias = jnp.zeros((conv_weight.shape[0],), dtype=conv_weight.dtype)
        try:
            if tail_fused:
                assert norm_weight is not None
                return gdn_conv_packed_projection_decode_step_bf16_raw_gates_tail(
                    packed_proj.astype(jnp.bfloat16),
                    decay.astype(jnp.float32),
                    dt_bias.astype(jnp.float32),
                    conv_state.astype(jnp.float32),
                    conv_weight.astype(jnp.float32),
                    conv_bias.astype(jnp.float32),
                    recurrent_state.astype(jnp.float32),
                    norm_weight.astype(jnp.float32),
                    qkv_dim=int(qkv_dim),
                    rms_norm_eps=float(rms_norm_eps),
                )
            return gdn_conv_packed_projection_decode_step_bf16_raw_gates(
                packed_proj.astype(jnp.bfloat16),
                decay.astype(jnp.float32),
                dt_bias.astype(jnp.float32),
                conv_state.astype(jnp.float32),
                conv_weight.astype(jnp.float32),
                conv_bias.astype(jnp.float32),
                recurrent_state.astype(jnp.float32),
                qkv_dim=int(qkv_dim),
            )
        except Exception:
            _raise_if_gdn_fallback_disabled(
                "Triton FLA packed-projection conv+decode kernel cannot handle this shape",
                self.config,
            )
            return reference()

    def gated_delta_conv_packed_projection_decode_state_pool(
        self,
        packed_proj: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        conv_state_pool: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_bias: jnp.ndarray | None,
        recurrent_state_pool: jnp.ndarray,
        *,
        qkv_dim: int,
        linear_layer_idx: int,
        use_qk_l2norm_in_kernel: bool,
        norm_weight: jnp.ndarray,
        valid_rows: jnp.ndarray | None = None,
        rms_norm_eps: float = 1.0e-6,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        impl = _gdn_packed_decode_impl(self.config)
        if impl != "triton_fla_conv_raw_gates_tail":
            raise RuntimeError(
                "state-pool packed-projection GDN decode is only enabled by "
                f"{_GDN_PACKED_DECODE_IMPL_ENV}=triton_fla_conv_raw_gates_tail"
            )
        if not use_qk_l2norm_in_kernel:
            raise ValueError("state-pool conv packed GDN decode requires q/k l2norm")
        if conv_bias is None:
            conv_bias = jnp.zeros((conv_weight.shape[0],), dtype=conv_weight.dtype)

        try:
            from nanovllm_jax.kernels.gdn_fla_triton import (
                gdn_conv_packed_projection_decode_step_bf16_raw_gates_tail_state_pool,
            )
        except (ImportError, ModuleNotFoundError, AttributeError):
            _raise_if_gdn_fallback_disabled(
                "Triton FLA state-pool packed-projection conv+decode kernel is unavailable",
                self.config,
            )
            raise

        try:
            return gdn_conv_packed_projection_decode_step_bf16_raw_gates_tail_state_pool(
                packed_proj.astype(jnp.bfloat16),
                decay.astype(jnp.float32),
                dt_bias.astype(jnp.float32),
                conv_state_pool.astype(jnp.float32),
                conv_weight.astype(jnp.float32),
                conv_bias.astype(jnp.float32),
                recurrent_state_pool.astype(jnp.float32),
                norm_weight.astype(jnp.float32),
                valid_rows,
                qkv_dim=int(qkv_dim),
                linear_layer_idx=int(linear_layer_idx),
                rms_norm_eps=float(rms_norm_eps),
            )
        except Exception:
            _raise_if_gdn_fallback_disabled(
                "Triton FLA state-pool packed-projection conv+decode kernel cannot handle this shape",
                self.config,
            )
            raise


class KernelBackendPlaceholder(PureJAXBackend):
    """Explicit placeholder until GPU/TPU kernels are implemented."""

    def __init__(self, name: str, kernel_backend: KernelBackendStatus, config=None):
        super().__init__(kernel_backend=kernel_backend, config=config)
        self.name = name


def select_backend(name: str = "auto", *, config=None) -> InferenceBackend:
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
        return PureJAXBackend(kernel_backend=kernel_backend, config=config)
    if normalized in {"pure_jax", "jax"}:
        return PureJAXBackend(config=config)
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
        return KernelBackendPlaceholder(normalized, kernel_backend, config=config)
    raise ValueError(f"Unknown backend {name!r}; expected auto, pure_jax, gpu, or tpu")
