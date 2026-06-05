"""Qwen 3.5 model implementation in pure JAX - matching HF exactly."""

import os
import jax
import jax.numpy as jnp
from jax import nn, lax
from typing import Optional, List, Dict
from dataclasses import dataclass, replace
from nanovllm_jax.backends import (
    InferenceBackend,
    gdn_packed_decode_conv_enabled,
    gdn_packed_decode_enabled,
    gdn_packed_decode_max_batch,
    gdn_prefill_post_conv_enabled,
    select_backend,
)
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import rms_norm, apply_rope, repeat_kv, causal_mask, get_activation, l2norm, causal_conv1d_update
from nanovllm_jax.kv_cache import AttentionMetadata, HybridLayerState, KVCacheState, init_linear_attention_states
from nanovllm_jax.mtp.mtp_layer import MTPParams
from nanovllm_jax.conv1d_metal import causal_conv1d_metal


@dataclass
class ModelParams:
    embed_tokens: jnp.ndarray
    layers: List[Dict[str, jnp.ndarray]]
    norm_weight: jnp.ndarray
    lm_head: Optional[jnp.ndarray] = None
    mtp_params: Optional[MTPParams] = None  # MTP head parameters for speculative decoding


_GDN_DECODE_IN_PROJ_PACKED_KEY = "in_proj_qkv_abz"
_FULL_ATTN_DECODE_QKV_PACKED_KEY = "qkv_proj_decode"
_MLP_GATE_UP_PACKED_KEY = "gate_up_proj"
_TRUE_CONFIG_VALUES = {"1", "true", "yes", "on", "True"}


def _config_or_env_bool(
    config: Optional[Qwen3_5Config],
    attr: str,
    env_name: str,
    *,
    default: bool = False,
) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value in _TRUE_CONFIG_VALUES
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return default


def _config_or_env_str(
    config: Optional[Qwen3_5Config],
    attr: str,
    env_name: str,
    *,
    default: str,
) -> str:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return str(env_value).strip().lower()
    if config is not None and hasattr(config, attr):
        return str(getattr(config, attr) or default).strip().lower()
    return default


def _config_or_env_int(
    config: Optional[Qwen3_5Config],
    attr: str,
    env_name: str,
    *,
    default: int,
) -> int:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value or default)
    if config is not None and hasattr(config, attr):
        return int(getattr(config, attr) or default)
    return default


def _tokenwise_decode_dot(x: jnp.ndarray, weight: jnp.ndarray, *, force_width1: bool = False) -> jnp.ndarray:
    """Apply a tokenwise linear with width-1 matmul shapes for multi-token decode.

    TPU bf16 matmuls can be shape dependent. For cached speculative decode, the
    first token in a width-2 verifier must match a standalone width-1 decode
    before any commit decision. Slicing the sequence dimension keeps the matmul
    shape aligned with sequential decode while preserving one compiled graph.
    """
    if not force_width1 or x.ndim != 3 or x.shape[1] <= 1:
        return jnp.dot(x, weight)
    batch, seq_len, hidden = x.shape
    outputs = []
    for t in range(seq_len):
        out_t = jnp.dot(x[:, t : t + 1].reshape(batch, hidden), weight)
        outputs.append(out_t[:, None, :])
    return jnp.concatenate(outputs, axis=1)


def _stable_rmsnorm_fp32(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Apply RMSNorm with an fp32 reduction and return the original dtype."""
    x_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    y = x32 * lax.rsqrt(variance + eps)
    y = y * weight.astype(jnp.float32)
    return y.astype(x_dtype)


def _packed_causal_conv1d_prefill(
    mixed_qkv: jnp.ndarray,
    initial_conv_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    token_row_ids: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    *,
    max_row_tokens: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized packed GDN prefill convolution.

    Computes the same result as applying `causal_conv1d_update` token by token
    per packed row, but avoids a token-length scan and per-token state scatter.
    """
    if mixed_qkv.ndim != 3 or mixed_qkv.shape[0] != 1:
        raise ValueError("packed mixed_qkv must have shape [1, token_bucket, conv_dim]")
    if token_row_ids.shape != mixed_qkv.shape[:2]:
        raise ValueError("token_row_ids must have shape [1, token_bucket]")
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must have shape [row_count + 1]")

    _, token_bucket, conv_dim = mixed_qkv.shape
    row_count = int(query_start_loc.shape[0]) - 1
    if row_count <= 0:
        raise ValueError("packed prefill requires at least one row")

    kernel_size = int(initial_conv_state.shape[-1])
    row_query_len = token_bucket if max_row_tokens is None else min(token_bucket, int(max_row_tokens))
    row_query_len = max(1, row_query_len)
    row_offsets = jnp.arange(row_query_len, dtype=jnp.int32)
    row_starts = query_start_loc[:-1].astype(jnp.int32)
    row_lens = (query_start_loc[1:] - query_start_loc[:-1]).astype(jnp.int32)
    token_indices = row_starts[:, None] + row_offsets[None, :]
    valid_queries = row_offsets[None, :] < row_lens[:, None]
    safe_token_indices = jnp.clip(token_indices, 0, token_bucket - 1)

    packed_mixed = mixed_qkv[0]
    mixed_rows = packed_mixed[safe_token_indices]
    mixed_rows = jnp.where(valid_queries[:, :, None], mixed_rows, 0.0)
    conv_input = jnp.concatenate(
        [initial_conv_state, mixed_rows.transpose(0, 2, 1)],
        axis=-1,
    )
    conv_all = causal_conv1d_metal(
        conv_input,
        conv_weight,
        conv_bias,
        activation="silu",
    )
    conv_rows = conv_all[:, :, kernel_size : kernel_size + row_query_len].transpose(
        0,
        2,
        1,
    )
    conv_rows = jnp.where(valid_queries[:, :, None], conv_rows, 0.0)
    conv_out_flat = jnp.zeros((token_bucket, conv_dim), dtype=conv_rows.dtype)
    conv_out_flat = conv_out_flat.at[safe_token_indices.reshape(-1)].add(
        conv_rows.reshape(-1, conv_dim)
    )

    state_positions = row_lens[:, None] + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
    state_positions = jnp.clip(state_positions, 0, conv_input.shape[-1] - 1)
    final_conv_state = jnp.take_along_axis(
        conv_input,
        state_positions[:, None, :],
        axis=-1,
    )
    return conv_out_flat[None, :, :], final_conv_state


def _decode_width1_rms_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float,
    *,
    force_width1: bool = False,
) -> jnp.ndarray:
    """Apply RMSNorm with width-1 sequence shapes for multi-token decode.

    The K=1 one-pass verifier evaluates a two-token decode block, but its first
    token must match the canonical single-token decode path exactly enough for
    greedy parity. TPU BF16 reductions can be shape-dependent even when each
    token is independent along the normalized dimension, so run the same
    `[B, 1, ...]` RMSNorm shape used by baseline decode and concatenate the
    token results inside the compiled graph.
    """
    stable_decode_norm = os.environ.get("NANO_VLLM_JAX_STABLE_DECODE_RMSNORM", "0") in {
        "1",
        "true",
        "yes",
        "on",
        "True",
    }
    if force_width1:
        from nanovllm_jax.kernels.decode_reductions import (
            decode_rms_norm,
            lowered_decode_rms_norm_enabled,
        )

        if lowered_decode_rms_norm_enabled():
            return decode_rms_norm(x, weight, eps)
    if not force_width1 or x.ndim < 3 or x.shape[1] <= 1:
        return _stable_rmsnorm_fp32(x, weight, eps) if stable_decode_norm and force_width1 else rms_norm(x, weight, eps)
    norm_fn = _stable_rmsnorm_fp32 if stable_decode_norm else rms_norm
    if os.environ.get("NANO_VLLM_JAX_SCAN_WIDTH1_RMSNORM", "0") in {
        "1",
        "true",
        "yes",
        "on",
        "True",
    }:
        x_time_major = jnp.swapaxes(x, 0, 1)

        def scan_norm(_, x_t):
            y_t = norm_fn(x_t[:, None, ...], weight, eps)
            return None, y_t[:, 0, ...]

        _, y_time_major = lax.scan(scan_norm, None, x_time_major)
        return jnp.swapaxes(y_time_major, 0, 1)
    parts = [norm_fn(x[:, t : t + 1, ...], weight, eps) for t in range(x.shape[1])]
    return jnp.concatenate(parts, axis=1)


def _force_width1_decode_math() -> bool:
    """Use width-1-shaped matmuls in multi-token decode by default.

    K=1 MTP verifies a width-2 decode block but must match the canonical
    width-1 baseline token for every physical batch shape. TPU BF16 matmuls are
    shape-sensitive enough that the width-2 verifier can diverge at larger
    batches unless tokenwise projections use the width-1 decode shape.
    """
    return os.environ.get("NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_MATH", "1") in {
        "1",
        "true",
        "yes",
        "on",
        "True",
    }


def _lm_head_decode_activation_dtype(config: Optional[Qwen3_5Config] = None) -> jnp.dtype:
    value = _config_or_env_str(
        config,
        "lm_head_decode_act_dtype",
        "NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE",
        default="fp32",
    )
    if value in {"", "0", "false", "no", "off", "none", "fp32", "float32"}:
        return jnp.float32
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    raise ValueError(
        "NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE must be fp32 or bf16, "
        f"got {value!r}"
    )


def _decode_padded_gemm_enabled(config: Optional[Qwen3_5Config] = None) -> bool:
    return _config_or_env_bool(
        config,
        "decode_padded_gemm",
        "NANO_VLLM_JAX_DECODE_PADDED_GEMM",
    )


def _decode_padded_gemm_gate_up_enabled(config: Optional[Qwen3_5Config] = None) -> bool:
    return _config_or_env_bool(
        config,
        "decode_padded_gemm_gate_up",
        "NANO_VLLM_JAX_DECODE_PADDED_GEMM_GATE_UP",
    )


def _decode_padded_gemm_rows(config: Optional[Qwen3_5Config] = None) -> int:
    value = _config_or_env_int(
        config,
        "decode_padded_gemm_rows",
        "NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS",
        default=8,
    )
    try:
        rows = int(value)
    except ValueError as exc:
        raise ValueError(
            "NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS must be an integer, "
            f"got {value!r}"
        ) from exc
    if rows < 1:
        raise ValueError("NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS must be positive")
    return rows


def _decode_padded_gemm_max_out_dim(config: Optional[Qwen3_5Config] = None) -> int:
    value = _config_or_env_int(
        config,
        "decode_padded_gemm_max_out_dim",
        "NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM",
        default=8192,
    )
    try:
        out_dim = int(value)
    except ValueError as exc:
        raise ValueError(
            "NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM must be an integer, "
            f"got {value!r}"
        ) from exc
    if out_dim < 1:
        raise ValueError("NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM must be positive")
    return out_dim


def _can_use_decode_padded_gemm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    config: Optional[Qwen3_5Config] = None,
) -> bool:
    rows = _decode_padded_gemm_rows(config)
    return (
        _decode_padded_gemm_enabled(config)
        and x.ndim == 3
        and weight.ndim == 2
        and int(x.shape[0]) <= rows
        and int(x.shape[1]) == 1
        and int(x.shape[-1]) == int(weight.shape[0])
        and int(weight.shape[1]) <= _decode_padded_gemm_max_out_dim(config)
    )


def _decode_padded_gemm_dot(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    config: Optional[Qwen3_5Config] = None,
) -> jnp.ndarray:
    """Run a small-B decode projection through a row-padded GEMM."""
    batch = int(x.shape[0])
    hidden = int(x.shape[-1])
    out_dim = int(weight.shape[1])
    rows = _decode_padded_gemm_rows(config)
    x_rows = jnp.reshape(x, (batch, hidden))
    if batch < rows:
        x_padded = jnp.pad(x_rows, ((0, rows - batch), (0, 0)))
    else:
        x_padded = x_rows
    out = jnp.dot(x_padded, weight)
    return out[:batch, :].reshape(batch, 1, out_dim)


def _decode_projection_activation_dtype(
    batch_size: int | None = None,
    config: Optional[Qwen3_5Config] = None,
) -> jnp.dtype:
    value = _config_or_env_str(
        config,
        "decode_proj_act_dtype",
        "NANO_VLLM_JAX_DECODE_PROJ_ACT_DTYPE",
        default="fp32",
    )
    if value in {"", "0", "false", "no", "off", "none", "fp32", "float32"}:
        return jnp.float32
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if value in {"bf16_single_seq", "bfloat16_single_seq", "bf16_single_sequence"}:
        return jnp.bfloat16 if batch_size == 1 else jnp.float32
    raise ValueError(
        "NANO_VLLM_JAX_DECODE_PROJ_ACT_DTYPE must be fp32, bf16, "
        "or bf16_single_seq, "
        f"got {value!r}"
    )


def _use_gdn_decode_packed_in_proj(
    params: Dict[str, jnp.ndarray],
    *,
    is_prefill: bool,
    batch: int,
    seq_len: int,
) -> bool:
    return (
        not is_prefill
        and batch > 1
        and seq_len == 1
        and _GDN_DECODE_IN_PROJ_PACKED_KEY in params
    )


def _use_gdn_prefill_packed_in_proj(
    params: Dict[str, jnp.ndarray],
    *,
    is_prefill: bool,
    config: Optional[Qwen3_5Config] = None,
) -> bool:
    return (
        is_prefill
        and _GDN_DECODE_IN_PROJ_PACKED_KEY in params
        and _enable_compact_prefill_in_proj_qkv(config)
        and _enable_compact_prefill_gdn_z(config)
    )


def _use_full_attention_decode_packed_qkv(
    params: Dict[str, jnp.ndarray],
    *,
    is_prefill: bool,
    batch: int,
    seq_len: int,
) -> bool:
    return (
        not is_prefill
        and batch > 1
        and seq_len == 1
        and _FULL_ATTN_DECODE_QKV_PACKED_KEY in params
    )


def _use_full_attention_prefill_packed_qkv(
    params: Dict[str, jnp.ndarray],
    *,
    is_prefill: bool,
    config: Optional[Qwen3_5Config] = None,
) -> bool:
    return (
        is_prefill
        and _FULL_ATTN_DECODE_QKV_PACKED_KEY in params
        and _enable_compact_prefill_full_attn_proj(config)
    )


def _enable_chunked_gdn_prefill() -> bool:
    """Use the chunked cached-prefill GDN path; set env to 0 for fallback."""
    return os.environ.get("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1") in {
        "1",
        "true",
        "yes",
        "on",
        "True",
    }


def _enable_compact_prefill_in_proj_qkv(config: Optional[Qwen3_5Config] = None) -> bool:
    """Compact true prefill tokens for the GDN QKV input projection."""
    return _config_or_env_bool(
        config,
        "compact_prefill_in_proj_qkv",
        "NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV",
    )


def _enable_compact_prefill_mlp(config: Optional[Qwen3_5Config] = None) -> bool:
    """Compact true prefill tokens for tokenwise MLP projections."""
    return _config_or_env_bool(
        config,
        "compact_prefill_mlp",
        "NANO_VLLM_JAX_COMPACT_PREFILL_MLP",
    )


def _enable_compact_prefill_gdn_z(config: Optional[Qwen3_5Config] = None) -> bool:
    """Compact true prefill tokens for the GDN Z input projection."""
    return _config_or_env_bool(
        config,
        "compact_prefill_gdn_z",
        "NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z",
    )


def _enable_compact_prefill_full_attn_proj(config: Optional[Qwen3_5Config] = None) -> bool:
    """Compact true prefill tokens for full-attention Q/K/V projections."""
    return _config_or_env_bool(
        config,
        "compact_prefill_full_attn_proj",
        "NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ",
    )


def _compact_prefill_dot_if_enabled(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    valid_token_mask: Optional[jnp.ndarray],
    compact_num_tokens: Optional[int],
    *,
    enabled: bool,
) -> jnp.ndarray:
    """Run a tokenwise projection only on true ragged prefill tokens."""
    if (
        not enabled
        or valid_token_mask is None
        or compact_num_tokens is None
        or x.ndim != 3
    ):
        return jnp.dot(x, weight)
    batch, seq_len, _ = x.shape
    output_features = weight.shape[-1]
    row_idx, col_idx = jnp.nonzero(valid_token_mask, size=int(compact_num_tokens))
    compact_x = x[row_idx, col_idx, :]
    compact_out = jnp.dot(compact_x, weight)
    out = jnp.zeros((batch, seq_len, output_features), dtype=compact_out.dtype)
    return out.at[row_idx, col_idx, :].set(compact_out)


def _compact_prefill_tokenwise_dot(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    valid_token_mask: Optional[jnp.ndarray],
    compact_num_tokens: Optional[int],
    config: Optional[Qwen3_5Config] = None,
) -> jnp.ndarray:
    """Run a tokenwise projection only on true ragged prefill tokens."""
    return _compact_prefill_dot_if_enabled(
        x,
        weight,
        valid_token_mask,
        compact_num_tokens,
        enabled=_enable_compact_prefill_in_proj_qkv(config),
    )


def _compact_prefill_mlp(
    x: jnp.ndarray,
    gate_weight: jnp.ndarray,
    up_weight: jnp.ndarray,
    down_weight: jnp.ndarray,
    activation_fn,
    valid_token_mask: Optional[jnp.ndarray],
    compact_num_tokens: Optional[int],
    config: Optional[Qwen3_5Config] = None,
) -> jnp.ndarray:
    """Run tokenwise prefill MLP only on true ragged tokens."""
    if (
        not _enable_compact_prefill_mlp(config)
        or valid_token_mask is None
        or compact_num_tokens is None
        or x.ndim != 3
    ):
        gate = _tokenwise_decode_dot(x, gate_weight, force_width1=False)
        up = _tokenwise_decode_dot(x, up_weight, force_width1=False)
        return _tokenwise_decode_dot(activation_fn(gate) * up, down_weight, force_width1=False)
    batch, seq_len, _ = x.shape
    output_features = down_weight.shape[-1]
    row_idx, col_idx = jnp.nonzero(valid_token_mask, size=int(compact_num_tokens))
    compact_x = x[row_idx, col_idx, :]
    gate = jnp.dot(compact_x, gate_weight)
    up = jnp.dot(compact_x, up_weight)
    compact_out = jnp.dot(activation_fn(gate) * up, down_weight)
    out = jnp.zeros((batch, seq_len, output_features), dtype=compact_out.dtype)
    return out.at[row_idx, col_idx, :].set(compact_out)


def _compact_prefill_mlp_packed(
    x: jnp.ndarray,
    gate_up_weight: jnp.ndarray,
    down_weight: jnp.ndarray,
    activation_fn,
    valid_token_mask: Optional[jnp.ndarray],
    compact_num_tokens: Optional[int],
    config: Optional[Qwen3_5Config] = None,
) -> jnp.ndarray:
    """Run tokenwise prefill MLP only on true ragged tokens with packed gate/up."""
    if (
        not _enable_compact_prefill_mlp(config)
        or valid_token_mask is None
        or compact_num_tokens is None
        or x.ndim != 3
    ):
        gate_up = _tokenwise_decode_dot(x, gate_up_weight, force_width1=False)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return _tokenwise_decode_dot(activation_fn(gate) * up, down_weight, force_width1=False)
    batch, seq_len, _ = x.shape
    output_features = down_weight.shape[-1]
    row_idx, col_idx = jnp.nonzero(valid_token_mask, size=int(compact_num_tokens))
    compact_x = x[row_idx, col_idx, :]
    gate_up = jnp.dot(compact_x, gate_up_weight)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    compact_out = jnp.dot(activation_fn(gate) * up, down_weight)
    out = jnp.zeros((batch, seq_len, output_features), dtype=compact_out.dtype)
    return out.at[row_idx, col_idx, :].set(compact_out)


def lm_head_token_ids_and_topk(
    hidden: jnp.ndarray,
    params: ModelParams,
    config,
    *,
    hidden_is_normed: bool = False,
    is_prefill: bool = True,
    top_k: int = 0,
):
    """Return greedy LM-head token ids and optional top-k values on device.

    Speculative verification needs exact target token ids and sometimes a
    top-2 margin, but returning full `[B, width, vocab]` logits from the JIT
    bloats the verifier path. Keep the dense LM-head computation inside the
    compiled graph and return only small verifier products.
    """
    if hidden_is_normed:
        hidden_norm = hidden
    else:
        if not is_prefill:
            from nanovllm_jax.kernels.decode_reductions import (
                decode_rms_norm,
                lowered_decode_rms_norm_enabled,
            )

            hidden_norm = (
                decode_rms_norm(hidden, params.norm_weight, config.rms_norm_eps)
                if lowered_decode_rms_norm_enabled()
                else rms_norm(hidden, params.norm_weight, config.rms_norm_eps)
            )
        else:
            hidden_norm = rms_norm(hidden, params.norm_weight, config.rms_norm_eps)
    hidden_norm = hidden_norm.astype(
        _lm_head_decode_activation_dtype(config) if not is_prefill else jnp.float32
    )
    output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
    logits = _tokenwise_decode_dot(
        hidden_norm,
        output_weight,
        force_width1=(not is_prefill) and hidden_norm.ndim == 3 and hidden_norm.shape[1] > 1 and _force_width1_decode_math(),
    )
    token_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if top_k > 0:
        top_values, top_indices = jax.lax.top_k(logits.astype(jnp.float32), top_k)
        return token_ids, top_values, top_indices.astype(jnp.int32)
    return token_ids, None, None


# Register ModelParams as a JAX pytree node for JIT compatibility
def _model_params_flatten(params: ModelParams):
    """Flatten ModelParams into children and auxiliary data."""
    # Flatten all layer dicts into a tuple of arrays
    layer_children = []
    layer_aux = []
    for layer in params.layers:
        # Sort keys for consistent ordering
        keys = sorted(layer.keys())
        if _MLP_GATE_UP_PACKED_KEY in layer:
            keys = [k for k in keys if k not in {"gate_proj", "up_proj"}]
        layer_aux.append(keys)
        for k in keys:
            layer_children.append(layer[k])
    
    children = (
        params.embed_tokens,
        *layer_children,
        params.norm_weight,
        params.lm_head if params.lm_head is not None else jnp.zeros((1,), dtype=jnp.float16),
        params.mtp_params if params.mtp_params is not None else jnp.zeros((1,), dtype=jnp.float16),
    )
    aux_data = (
        len(params.layers),
        layer_aux,
        params.lm_head is not None,
        params.mtp_params is not None,
    )
    return children, aux_data


def _model_params_unflatten(aux_data, children):
    """Unflatten children and auxiliary data into ModelParams."""
    num_layers, layer_aux, has_lm_head, has_mtp = aux_data
    
    # Reconstruct layers
    layers = []
    child_idx = 1  # Skip embed_tokens
    for layer_keys in layer_aux:
        layer = {}
        for k in layer_keys:
            layer[k] = children[child_idx]
            child_idx += 1
        layers.append(layer)
    
    # Get remaining fields
    norm_weight = children[child_idx]
    child_idx += 1
    lm_head = children[child_idx] if has_lm_head else None
    child_idx += 1
    mtp_params = children[child_idx] if has_mtp else None
    
    return ModelParams(
        embed_tokens=children[0],
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
        mtp_params=mtp_params,
    )


jax.tree_util.register_pytree_node(
    ModelParams,
    _model_params_flatten,
    _model_params_unflatten
)


def init_params(key: jax.Array, config: Qwen3_5Config) -> ModelParams:
    keys = jax.random.split(key, config.num_hidden_layers + 3)
    embed_tokens = jax.random.normal(keys[0], (config.vocab_size, config.hidden_size)) * (config.hidden_size ** -0.5)
    layers = [init_transformer_block(keys[i + 1], config, i) for i in range(config.num_hidden_layers)]
    norm_weight = jnp.ones(config.hidden_size)
    lm_head = None if config.tie_word_embeddings else jax.random.normal(keys[-2], (config.hidden_size, config.vocab_size)) * (config.hidden_size ** -0.5)
    return ModelParams(embed_tokens=embed_tokens, layers=layers, norm_weight=norm_weight, lm_head=lm_head)


def init_transformer_block(key: jax.Array, config: Qwen3_5Config, layer_idx: int) -> Dict[str, jnp.ndarray]:
    keys = jax.random.split(key, 10)
    if config.layer_types[layer_idx] == "full_attention":
        # Qwen3.5 full attention: q_proj outputs [query, gate] each of size num_attention_heads * head_dim
        attn_out_dim = config.num_attention_heads * config.head_dim
        q_proj = jax.random.normal(keys[0], (config.hidden_size, attn_out_dim * 2)) * (config.hidden_size ** -0.5)
        k_proj = jax.random.normal(keys[1], (config.hidden_size, config.num_key_value_heads * config.head_dim)) * (config.hidden_size ** -0.5)
        v_proj = jax.random.normal(keys[2], (config.hidden_size, config.num_key_value_heads * config.head_dim)) * (config.hidden_size ** -0.5)
        gate_proj = jax.random.normal(keys[5], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        up_proj = jax.random.normal(keys[6], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        return {
            "q_proj": q_proj,
            "k_proj": k_proj,
            "v_proj": v_proj,
            _FULL_ATTN_DECODE_QKV_PACKED_KEY: jnp.concatenate(
                [q_proj, k_proj, v_proj],
                axis=1,
            ),
            "o_proj": jax.random.normal(keys[3], (attn_out_dim, config.hidden_size)) * (config.hidden_size ** -0.5),
            "q_norm": jnp.ones((config.num_attention_heads, config.head_dim)),
            "k_norm": jnp.ones((config.num_key_value_heads, config.head_dim)),
            "input_norm": jnp.ones(config.hidden_size),
            "post_attn_norm": jnp.ones(config.hidden_size),
            "gate_proj": gate_proj,
            "up_proj": up_proj,
            _MLP_GATE_UP_PACKED_KEY: jnp.concatenate([gate_proj, up_proj], axis=1),
            "down_proj": jax.random.normal(keys[7], (config.intermediate_size, config.hidden_size)) * (config.hidden_size ** -0.5),
            "ffn_norm": jnp.ones(config.hidden_size),
        }
    else:
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        in_proj_qkv = jax.random.normal(keys[0], (config.hidden_size, conv_dim)) * (config.hidden_size ** -0.5)
        in_proj_z = jax.random.normal(keys[1], (config.hidden_size, value_dim)) * (config.hidden_size ** -0.5)
        in_proj_a = jax.random.normal(keys[2], (config.hidden_size, config.linear_num_value_heads)) * (config.hidden_size ** -0.5)
        in_proj_b = jax.random.normal(keys[3], (config.hidden_size, config.linear_num_value_heads)) * (config.hidden_size ** -0.5)
        gate_proj = jax.random.normal(keys[5], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        up_proj = jax.random.normal(keys[6], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        return {
            "input_norm": jnp.ones(config.hidden_size),
            "in_proj_qkv": in_proj_qkv,
            "in_proj_z": in_proj_z,
            "in_proj_a": in_proj_a,
            "in_proj_b": in_proj_b,
            _GDN_DECODE_IN_PROJ_PACKED_KEY: jnp.concatenate(
                [in_proj_qkv, in_proj_a, in_proj_b, in_proj_z],
                axis=1,
            ),
            "conv1d_weight": jax.random.normal(keys[4], (conv_dim, config.linear_conv_kernel_size)) * 0.02,
            "dt_bias": jnp.ones(config.linear_num_value_heads),
            "A": jnp.exp(jnp.full(config.linear_num_value_heads, 0.0)),
            "norm_weight": jnp.ones(config.linear_value_head_dim),
            "out_proj": jax.random.normal(keys[6], (value_dim, config.hidden_size)) * (config.hidden_size ** -0.5),
            "gate_proj": gate_proj,
            "up_proj": up_proj,
            _MLP_GATE_UP_PACKED_KEY: jnp.concatenate([gate_proj, up_proj], axis=1),
            "down_proj": jax.random.normal(keys[7], (config.intermediate_size, config.hidden_size)) * (config.hidden_size ** -0.5),
            "ffn_norm": jnp.ones(config.hidden_size),
        }


def jax_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, 
                                output_final_state=False, use_qk_l2norm_in_kernel=False):
    """
    JAX implementation of chunk gated delta rule (matching HF torch_chunk_gated_delta_rule).
    Input shapes: [B, H, T, D] for query/key/value, [B, H, T] for g/beta
    Output shape: [B, H, T, D]
    """
    if query.shape[2] > chunk_size and not _enable_chunked_gdn_prefill():
        # The multi-chunk JAX chunk kernel still has measurable drift from the
        # HF/PyTorch chunked reference. Use the recurrent reference path for
        # correctness; the chunk kernel can be restored behind parity tests.
        output, final_state = jax_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return output, final_state if output_final_state else None

    import jax
    
    initial_dtype = query.dtype
    
    # Apply L2 norm if requested
    if use_qk_l2norm_in_kernel:
        query = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6)
        key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)
    
    # Convert to float32 for computation
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    g = g.astype(jnp.float32)
    
    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    # Pad to chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))
    
    total_seq_len = seq_len + pad_size
    scale = 1.0 / jnp.sqrt(k_head_dim)
    query = query * scale
    
    # v_beta = value * beta[..., None]
    v_beta = value * beta[..., None]
    # k_beta = key * beta[..., None]
    k_beta = key * beta[..., None]
    
    # Reshape to chunks: [B, H, n_chunks, chunk_size, D]
    def reshape_to_chunks(x):
        return x.reshape(batch_size, num_heads, -1, chunk_size, x.shape[-1])
    
    query_chunks = reshape_to_chunks(query)
    key_chunks = reshape_to_chunks(key)
    value_chunks = reshape_to_chunks(value)
    k_beta_chunks = reshape_to_chunks(k_beta)
    v_beta_chunks = reshape_to_chunks(v_beta)
    
    # g reshaped: [B, H, n_chunks, chunk_size]
    g_chunks = g.reshape(batch_size, num_heads, -1, chunk_size)
    n_chunks = g_chunks.shape[2]
    
    # Create mask for upper triangle (within chunk)
    mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    mask_strict_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)
    
    # Compute decay: cumulative sum of g within each chunk
    # Use Metal-compatible cumsum if on Metal backend
    if jax.default_backend() == 'METAL':
        from nanovllm_jax.metal_ops import cumsum_metal
        g_cumsum = cumsum_metal(g_chunks, axis=-1)
    else:
        g_cumsum = jnp.cumsum(g_chunks, axis=-1)
    
    # decay_mask[b,h,n,i,j] = exp(g_cumsum[b,h,n,i] - g_cumsum[b,h,n,j]) for i >= j, else 0
    decay_mask = jnp.tril(jnp.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
    
    # Compute k_beta @ key.transpose(-1, -2) for each chunk
    # k_beta_chunks: [B, H, n, cs, K], key_chunks: [B, H, n, cs, K]
    # Result: [B, H, n, cs, cs] where result[b,h,n,i,j] = sum_k(k_beta[b,h,n,i,k] * key[b,h,n,j,k])
    kkt = jnp.einsum('bhnck,bhnjk->bhncj', k_beta_chunks, key_chunks)
    
    # attn = -((k_beta @ k.T) * decay_mask), masked to lower triangle
    attn = -(kkt * decay_mask)
    attn = jnp.where(mask_upper, 0.0, attn)  # Zero out diagonal and upper triangle
    
    # Recursive computation within chunk (matching HF exactly)
    # for i in range(1, chunk_size):
    #     row = attn[..., i, :i].clone()  # [B, H, n, i]
    #     sub = attn[..., :i, :i].clone()  # [B, H, n, i, i]
    #     attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    def update_row(carry, i):
        attn_carry = carry
        
        # Create a mask that's True for indices < i
        mask = jnp.arange(chunk_size) < i  # [cs]
        
        # Extract row i, cols 0:i -> shape [B, H, n, cs] but only first i entries are valid
        row_i = attn_carry[..., i, :] * mask[None, None, None, :]  # [B, H, n, cs]
        
        # Extract submatrix [0:i, 0:i] -> shape [B, H, n, cs, cs] but only [0:i, 0:i] is valid
        # Multiply by mask on both dimensions
        sub_i = attn_carry * mask[None, None, None, :, None] * mask[None, None, None, None, :]  # [B, H, n, cs, cs]
        
        # HF: (row.unsqueeze(-1) * sub).sum(-2)
        # This computes: contribution[k] = sum_j(row[j] * sub[j, k]) for j,k in 0:i
        # Using einsum: 'bhnj,bhnjk->bhnk'
        contribution = jnp.einsum('bhnj,bhnjk->bhnk', row_i, sub_i)  # [B, H, n, cs]
        
        # new_row for cols 0:i
        new_row = row_i + contribution
        # Mask to keep only cols 0:i
        new_row = new_row * mask[None, None, None, :]
        
        # Update attn at row i
        attn_carry = attn_carry.at[..., i, :].set(new_row)
        return attn_carry, i
    
    attn, _ = lax.scan(update_row, attn, jnp.arange(1, chunk_size))
    
    # Add identity matrix
    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)[None, None, None, :, :]
    
    # value_transformed = attn @ v_beta
    # attn: [B, H, n, cs, cs], v_beta: [B, H, n, cs, V]
    # result: [B, H, n, cs, V]
    value_transformed = jnp.einsum('bhnct,bhntv->bhncv', attn, v_beta_chunks)
    
    # Initial-state correction uses decay from the start of each chunk.
    k_cumdecay = jnp.einsum('bhnct,bhntv->bhncv', attn, k_beta_chunks * jnp.exp(g_cumsum)[..., None])
    
    # Initialize state [B, H, V, K].
    if initial_state is None:
        state = jnp.zeros((batch_size, num_heads, v_head_dim, k_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)
    
    # Process each chunk sequentially
    def process_chunk(carry, i):
        state = carry
        q_i = query_chunks[:, :, i]      # [B, H, cs, K]
        k_i = key_chunks[:, :, i]        # [B, H, cs, K]
        v_i = value_transformed[:, :, i] # [B, H, cs, V]
        decay_mask_i = decay_mask[:, :, i]  # [B, H, cs, cs]
        k_cumdecay_i = k_cumdecay[:, :, i]  # [B, H, cs, K]
        g_cumsum_i = g_cumsum[:, :, i]    # [B, H, cs] - use cumsum version!
        
        # Within-chunk attention: attn = (q @ k.T * decay_mask), masked to strict upper triangle
        attn_i = jnp.einsum('bhck,bhdk->bhcd', q_i, k_i) * decay_mask_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)
        
        # v_prime = k_cumdecay @ state
        # k_cumdecay_i: [B, H, cs, K], state: [B, H, V, K]
        # v_prime[b,h,c,v] = sum_k(k_cumdecay_i[b,h,c,k] * state[b,h,v,k])
        v_prime = jnp.einsum('bhck,bhvk->bhcv', k_cumdecay_i, state)
        
        # v_new = v_i - v_prime
        v_new = v_i - v_prime
        
        # attn_inter = (q * exp(g)) @ state
        # q_i * exp(g_cumsum_i): [B, H, cs, K]
        # result: [B, H, cs, V] = sum_K(q[b,h,c,k] * exp(g_cumsum[b,h,c]) * state[b,h,v,k])
        attn_inter = jnp.einsum('bhck,bhvk->bhcv', q_i * jnp.exp(g_cumsum_i)[..., None], state)
        
        # core_attn_out = attn_inter + attn @ v_new
        # attn_i: [B, H, cs, cs], v_new: [B, H, cs, V]
        # result: [B, H, cs, V] = sum_d(attn_i[b,h,c,d] * v_new[b,h,d,v])
        attn_v_new = jnp.einsum('bhcd,bhdv->bhcv', attn_i, v_new)
        core_attn_out_i = attn_inter + attn_v_new
        
        # Update state
        # state = state * exp(g_cumsum[b,h,cs-1]) + (k * exp(g_cumsum[cs-1] - g_cumsum)).T @ v_new
        # g_last_minus_g[b,h,c] = g_cumsum[b,h,-1] - g_cumsum[b,h,c]
        g_last_minus_g = g_cumsum_i[..., -1, None] - g_cumsum_i  # [B, H, cs]
        # k_weighted[b,h,c,k] = k_i[b,h,c,k] * exp(g_last_minus_g[b,h,c])
        k_weighted = k_i * jnp.exp(g_last_minus_g)[..., None]  # [B, H, cs, K]
        # state_update[b,h,v,k] = sum_c(v_new[b,h,c,v] * k_weighted[b,h,c,k])
        state_update = jnp.einsum('bhcv,bhck->bhvk', v_new, k_weighted)
        state = state * jnp.exp(g_cumsum_i[..., -1, None, None]) + state_update
        
        return state, core_attn_out_i
    
    final_state, core_attn_out_chunks = lax.scan(process_chunk, state, jnp.arange(n_chunks))
    
    # lax.scan returns [n_chunks, B, H, chunk, V]; HF keeps chunk as the
    # third dimension [B, H, n_chunks, chunk, V] before flattening time.
    core_attn_out = core_attn_out_chunks.transpose(1, 2, 0, 3, 4).reshape(
        batch_size,
        num_heads,
        -1,
        v_head_dim,
    )
    core_attn_out = core_attn_out[:, :, :seq_len]  # Remove padding
    
    if not output_final_state:
        final_state = None
    
    core_attn_out = core_attn_out.astype(initial_dtype)
    return core_attn_out, final_state


def jax_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state=None,
    use_qk_l2norm_in_kernel=False,
    return_state_sequence: bool = False,
    return_first_state: bool = False,
):
    """
    JAX implementation of recurrent gated delta rule (matching HF torch_recurrent_gated_delta_rule).
    Input shapes: [B, H, T, D] for query/key/value, [B, H, T] for g/beta
    Output shape: [B, H, T, D]
    State shape: [B, H, v_head_dim, k_head_dim] (kernel-native V,K layout)
    """
    initial_dtype = query.dtype

    # Keep as [B, H, T, D]. For cached decode, normalize q/k in
    # float32 before l2norm so width-2 recurrent scans match the width-1
    # sequential decode path as closely as possible on TPU bf16.
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    value = value.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    batch, num_heads, time_dim, k_head_dim = query.shape
    v_head_dim = value.shape[-1]

    # Scale query
    query = query * (1.0 / jnp.sqrt(k_head_dim))

    # Initialize state: [B, H, V, K] so decode uses the same V,K layout as
    # the planned external GDN kernels.
    if initial_state is None:
        state = jnp.zeros((batch, num_heads, v_head_dim, k_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    def step_one(state, q_t, k_t, v_t, g_t_raw, beta_t):
        g_t = jnp.exp(g_t_raw)  # [B, H]

        # Reshape for broadcasting
        g_t_exp = g_t[:, :, None, None]    # [B, H, 1, 1]
        beta_t_exp = beta_t[:, :, None]    # [B, H, 1]
        k_t_exp = k_t[:, :, None, :]       # [B, H, 1, K]

        # Decay state: state * exp(g_t)
        state = state * g_t_exp

        # kv_mem = (state * k_t[..., None]).sum(-2)
        kv_mem = jnp.einsum('bhvk,bhk->bhv', state, k_t)

        # delta = (v_t - kv_mem) * beta_t
        delta = (v_t - kv_mem) * beta_t_exp  # [B, H, V]

        # state = state + delta[..., None] * k_t[:, :, None, :]
        state = state + delta[:, :, :, None] * k_t_exp

        # output_t = (state * q_t[:, :, None, :]).sum(-1)
        out_t = jnp.einsum('bhvk,bhk->bhv', state, q_t)

        return state, out_t

    def step_index(state, t):
        return step_one(
            state,
            query[:, :, t, :],
            key[:, :, t, :],
            value[:, :, t, :],
            g[:, :, t],
            beta[:, :, t],
        )

    if time_dim == 2:
        # Cached K=1 verifier uses width-2 decode. Avoid a two-step scan so the
        # first token is computed as the same explicit width-1 recurrence used by
        # the sequential commit-select reference, then apply the second token.
        state_0, out_0 = step_one(
            state,
            query[:, :, 0, :],
            key[:, :, 0, :],
            value[:, :, 0, :],
            g[:, :, 0],
            beta[:, :, 0],
        )
        state_1, out_1 = step_one(
            state_0,
            query[:, :, 1, :],
            key[:, :, 1, :],
            value[:, :, 1, :],
            g[:, :, 1],
            beta[:, :, 1],
        )
        output = jnp.stack([out_0, out_1], axis=2).astype(initial_dtype)
        if return_state_sequence:
            state_sequence = jnp.stack([state_0, state_1], axis=1)
            return output, state_1, state_sequence
        if return_first_state:
            return output, state_1, state_0
        return output, state_1

    if return_state_sequence:
        def step_fn(carry, t):
            next_state, out_t = step_index(carry, t)
            return next_state, (out_t, next_state)

        final_state, (all_outputs, all_states) = lax.scan(step_fn, state, jnp.arange(time_dim))
        output = all_outputs.transpose(1, 2, 0, 3)
        state_sequence = all_states.transpose(1, 0, 2, 3, 4)
        output = output.astype(initial_dtype)
        return output, final_state, state_sequence

    if return_first_state:
        def step_fn(carry, t):
            next_state, out_t = step_index(carry, t)
            return next_state, (out_t, next_state)

        final_state, (all_outputs, all_states) = lax.scan(step_fn, state, jnp.arange(time_dim))
        output = all_outputs.transpose(1, 2, 0, 3)
        first_state = all_states[0]
        output = output.astype(initial_dtype)
        return output, final_state, first_state

    def step_fn(carry, t):
        return step_index(carry, t)

    final_state, all_outputs = lax.scan(step_fn, state, jnp.arange(time_dim))

    # all_outputs: [T, B, H, V] -> transpose to [B, H, T, V]
    output = all_outputs.transpose(1, 2, 0, 3)

    output = output.astype(initial_dtype)
    return output, final_state


def gated_deltanet_block(
    x,
    params,
    positions,
    config,
    layer_idx: int,
    is_prefill: bool = True,
    hybrid_state: Optional[HybridLayerState] = None,
    valid_token_mask: Optional[jnp.ndarray] = None,
    compact_prefill_tokens: Optional[int] = None,
    backend: Optional[InferenceBackend] = None,
    return_prefix_state: bool = False,
    return_first_prefix_state: bool = False,
    hybrid_state_is_layer: bool = False,
    packed_token_row_ids: Optional[jnp.ndarray] = None,
    packed_query_start_loc: Optional[jnp.ndarray] = None,
):
    """Gated DeltaNet block with decode mode support.
    
    Args:
        x: Input [batch, seq_len, hidden]
        params: Layer parameters
        positions: Position IDs
        config: Model config
        layer_idx: Layer index (0-based)
        is_prefill: Whether this is prefill (True) or decode (False)
        hybrid_state: Optional linear-attention state for this batch
        
    Returns:
        tuple: (output, updated_hybrid_state) or just output for prefill
    """
    batch, seq_len, _ = x.shape
    if backend is None:
        backend = select_backend("pure_jax", config=config)
    prefix_layer_state = None
    
    # Cast to target dtype (bfloat16 for CPU/CUDA, float16 for Metal)
    dtype = config.get_dtype()
    x_cast = x.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    v_heads_per_k = config.linear_num_value_heads // config.linear_num_key_heads
    conv_dim = key_dim * 2 + value_dim
    
    # Check if we can use cached states
    use_cached = (
        not is_prefill and 
        hybrid_state is not None and 
        hybrid_state.conv_state is not None and
        hybrid_state.recurrent_state is not None and
        seq_len <= max(2, 1 + int(getattr(config, "num_speculative_tokens", 0) or 0))
    )
    use_cached_prefill = (
        is_prefill
        and hybrid_state is not None
        and hybrid_state.conv_state is not None
        and hybrid_state.recurrent_state is not None
    )
    use_recurrent_prefill = (
        use_cached_prefill
        and (
            seq_len <= int(getattr(config, "linear_recurrent_prefill_threshold", 8))
            or not _enable_chunked_gdn_prefill()
        )
    )
    linear_layer_idx = len([l for l in config.linear_attn_layers if l < layer_idx])
    row_valid = None
    
    # === PROJECTIONS (same for both modes) ===
    force_width1_dot = (not is_prefill) and seq_len > 1 and _force_width1_decode_math()
    use_packed_decode_in_proj = _use_gdn_decode_packed_in_proj(
        params,
        is_prefill=is_prefill,
        batch=batch,
        seq_len=seq_len,
    )
    use_packed_prefill_in_proj = _use_gdn_prefill_packed_in_proj(
        params,
        is_prefill=is_prefill,
        config=config,
    )
    packed_decode_projection = None
    if use_packed_decode_in_proj or use_packed_prefill_in_proj:
        if is_prefill:
            packed_proj = _compact_prefill_dot_if_enabled(
                x_cast,
                params[_GDN_DECODE_IN_PROJ_PACKED_KEY],
                valid_token_mask,
                compact_prefill_tokens,
                enabled=True,
            )
        else:
            packed_proj = _tokenwise_decode_dot(
                x_cast,
                params[_GDN_DECODE_IN_PROJ_PACKED_KEY],
                force_width1=force_width1_dot,
            )
        qkv_end = conv_dim
        a_end = qkv_end + config.linear_num_value_heads
        b_end = a_end + config.linear_num_value_heads
        if use_packed_decode_in_proj and not is_prefill:
            packed_decode_projection = packed_proj
            mixed_qkv = packed_proj[:, :, :qkv_end]
            a = None
            b = None
            z = packed_proj[:, :, b_end:].reshape(batch, seq_len, -1)
        else:
            mixed_qkv, a, b, z = jnp.split(packed_proj, [qkv_end, a_end, b_end], axis=-1)
            z = z.reshape(batch, seq_len, -1)
            a = a.reshape(batch, seq_len, config.linear_num_value_heads)
            b = b.reshape(batch, seq_len, config.linear_num_value_heads)
    else:
        if is_prefill:
            mixed_qkv = _compact_prefill_tokenwise_dot(
                x_cast,
                params["in_proj_qkv"],
                valid_token_mask,
                compact_prefill_tokens,
                config,
            )
        else:
            if _can_use_decode_padded_gemm(x_cast, params["in_proj_qkv"], config):
                mixed_qkv = _decode_padded_gemm_dot(x_cast, params["in_proj_qkv"], config)
            else:
                mixed_qkv = _tokenwise_decode_dot(
                    x_cast,
                    params["in_proj_qkv"],
                    force_width1=force_width1_dot,
                )
        if is_prefill:
            z = _compact_prefill_dot_if_enabled(
                x_cast,
                params["in_proj_z"],
                valid_token_mask,
                compact_prefill_tokens,
                enabled=_enable_compact_prefill_gdn_z(config),
            ).reshape(batch, seq_len, -1)
        else:
            z = _tokenwise_decode_dot(x_cast, params["in_proj_z"], force_width1=force_width1_dot).reshape(batch, seq_len, -1)
        a = _tokenwise_decode_dot(
            x_cast,
            params["in_proj_a"],
            force_width1=force_width1_dot,
        ).reshape(batch, seq_len, config.linear_num_value_heads)
        b = _tokenwise_decode_dot(
            x_cast,
            params["in_proj_b"],
            force_width1=force_width1_dot,
        ).reshape(batch, seq_len, config.linear_num_value_heads)
    
    if use_cached:
        # === DECODE MODE ===
        # 1. Convolution update - use per-layer conv_state
        # Table mode stores [batch, num_linear_layers, conv_dim, kernel_size].
        # Layerwise mode stores only [batch, conv_dim, kernel_size].
        layer_conv_state = (
            hybrid_state.conv_state
            if hybrid_state_is_layer
            else hybrid_state.conv_state[:, linear_layer_idx]
        )
        conv_weight = params["conv1d_weight"].reshape(conv_dim, config.linear_conv_kernel_size)
        conv_bias = params.get("conv1d_bias")

        # recurrent_state shape: [batch, num_layers, num_heads, v_dim, k_dim]
        # Extract recurrent state for this layer: [batch, num_heads, v_dim, k_dim]
        # linear_layer_idx computed above
        if hybrid_state.recurrent_state is not None:
            initial_recurrent = (
                hybrid_state.recurrent_state
                if hybrid_state_is_layer
                else hybrid_state.recurrent_state[:, linear_layer_idx]
            )
        else:
            initial_recurrent = None

        packed_decode_max_batch = gdn_packed_decode_max_batch(config)
        use_packed_decode = (
            gdn_packed_decode_enabled(config)
            and (packed_decode_max_batch is None or batch <= packed_decode_max_batch)
            and seq_len == 1
            and not return_prefix_state
            and not return_first_prefix_state
            and initial_recurrent is not None
        )
        use_conv_packed_decode = use_packed_decode and gdn_packed_decode_conv_enabled(config)
        if use_conv_packed_decode:
            if packed_decode_projection is not None:
                core_attn_out, new_layer_conv_state, new_recurrent_state_single = (
                    backend.gated_delta_conv_packed_projection_decode(
                        packed_decode_projection[:, 0, :],
                        params["A"].astype(jnp.float32),
                        params["dt_bias"].astype(jnp.float32),
                        layer_conv_state,
                        conv_weight,
                        conv_bias,
                        initial_recurrent.astype(jnp.float32),
                        qkv_dim=conv_dim,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            else:
                core_attn_out, new_layer_conv_state, new_recurrent_state_single = (
                    backend.gated_delta_conv_packed_decode(
                        mixed_qkv[:, 0, :],
                        a[:, 0, :].astype(jnp.float32),
                        b[:, 0, :].astype(jnp.float32),
                        params["A"].astype(jnp.float32),
                        params["dt_bias"].astype(jnp.float32),
                        layer_conv_state,
                        conv_weight,
                        conv_bias,
                        initial_recurrent.astype(jnp.float32),
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            prefix_layer_conv_state = None
            prefix_recurrent_state_single = None
        else:
            if a is None or b is None:
                a = packed_decode_projection[:, :, qkv_end:a_end].reshape(batch, seq_len, config.linear_num_value_heads)
                b = packed_decode_projection[:, :, a_end:b_end].reshape(batch, seq_len, config.linear_num_value_heads)
            mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # [B, D, T]
            def conv_step(state, mixed_qkv_t_step):
                conv_out_t, next_state = causal_conv1d_update(
                    mixed_qkv_t_step,
                    state,
                    conv_weight,
                    conv_bias,
                    "silu",
                )
                return next_state, conv_out_t

            if seq_len > 1:
                # Match sequential decode exactly: unroll static width-1 conv updates
                # instead of scanning over a width-2 tensor. Each call sees [B, D, 1],
                # the same shape used by the normal single-token decode path.
                state = layer_conv_state
                conv_out_parts = []
                conv_state_parts = []
                for t in range(seq_len):
                    state, conv_out_t = conv_step(state, mixed_qkv_t[:, :, t : t + 1])
                    conv_out_parts.append(conv_out_t)
                    if return_prefix_state or (return_first_prefix_state and t == 0):
                        conv_state_parts.append(state)
                new_layer_conv_state = state
                conv_out_steps = jnp.stack(conv_out_parts, axis=0)
                if return_prefix_state:
                    prefix_layer_conv_state = jnp.stack(conv_state_parts, axis=0).transpose(1, 0, 2, 3)
                elif return_first_prefix_state:
                    prefix_layer_conv_state = conv_state_parts[0] if conv_state_parts else state
                else:
                    prefix_layer_conv_state = None
            elif return_prefix_state or return_first_prefix_state:
                new_layer_conv_state, conv_out_t = conv_step(layer_conv_state, mixed_qkv_t[:, :, :1])
                conv_out_steps = conv_out_t[None, ...]
                prefix_layer_conv_state = (
                    new_layer_conv_state[:, None, :, :]
                    if return_prefix_state
                    else new_layer_conv_state
                )
            else:
                new_layer_conv_state, conv_out_t = conv_step(layer_conv_state, mixed_qkv_t[:, :, :1])
                conv_out_steps = conv_out_t[None, ...]
                prefix_layer_conv_state = None
            conv_out = conv_out_steps.transpose(1, 0, 2, 3).reshape(batch, seq_len, conv_dim)

        if use_packed_decode and not use_conv_packed_decode:
            core_attn_out, new_recurrent_state_single = backend.gated_delta_packed_decode(
                conv_out_t[:, :, 0].astype(jnp.float32),
                a[:, 0, :].astype(jnp.float32),
                b[:, 0, :].astype(jnp.float32),
                params["A"].astype(jnp.float32),
                params["dt_bias"].astype(jnp.float32),
                initial_recurrent.astype(jnp.float32),
                use_qk_l2norm_in_kernel=True,
            )
            prefix_recurrent_state_single = None
        elif not use_conv_packed_decode:
            # 2. Split q, k, v
            query = conv_out[:, :, :key_dim].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            key = conv_out[:, :, key_dim:key_dim*2].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            value = conv_out[:, :, key_dim*2:].reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)

            # 3. Compute gates
            beta = nn.sigmoid(b)  # [B, 1, H_v]
            g = -params["A"] * nn.softplus(a + params["dt_bias"])  # [B, 1, H_v]

            # 4. Repeat for GQA
            if v_heads_per_k > 1:
                query = jnp.repeat(query, v_heads_per_k, axis=2)
                key = jnp.repeat(key, v_heads_per_k, axis=2)

            # 5. Transpose to [B, H, T, D] format.
            query = query.transpose(0, 2, 1, 3)  # [B, H, T, D_k]
            key = key.transpose(0, 2, 1, 3)  # [B, H, T, D_k]
            value = value.transpose(0, 2, 1, 3)  # [B, H, T, D_v]
            g = g.transpose(0, 2, 1)  # [B, H, T]
            beta = beta.transpose(0, 2, 1)  # [B, H, T]

            # 6. Recurrent update
            use_recurrent_decode_scan = return_prefix_state or seq_len > 1
            if return_prefix_state:
                core_attn_out, new_recurrent_state_single, recurrent_state_steps = jax_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                    return_state_sequence=return_prefix_state,
                )
                prefix_recurrent_state_single = recurrent_state_steps
            elif return_first_prefix_state:
                core_attn_out, new_recurrent_state_single, prefix_recurrent_state_single = jax_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                    return_first_state=True,
                )
            elif use_recurrent_decode_scan:
                core_attn_out, new_recurrent_state_single = jax_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                    return_state_sequence=False,
                )
                prefix_recurrent_state_single = None
            else:
                core_attn_out, new_recurrent_state_single = backend.gated_delta_decode(
                    query, key, value, g, beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                )
                prefix_recurrent_state_single = None
        # new_recurrent_state_single has shape [batch, num_heads, v_dim, k_dim]
        if valid_token_mask is not None:
            if row_valid is None:
                row_valid = (valid_token_mask.astype(jnp.int32).sum(axis=1) > 0)
            conv_keep = row_valid[:, None, None]
            recurrent_keep = row_valid[:, None, None, None]
            new_layer_conv_state = jnp.where(
                conv_keep,
                new_layer_conv_state,
                layer_conv_state,
            )
            if initial_recurrent is not None:
                new_recurrent_state_single = jnp.where(
                    recurrent_keep,
                    new_recurrent_state_single,
                    initial_recurrent,
                )
                if return_first_prefix_state and prefix_recurrent_state_single is not None:
                    prefix_recurrent_state_single = jnp.where(
                        recurrent_keep,
                        prefix_recurrent_state_single,
                        initial_recurrent,
                    )
            if return_first_prefix_state and prefix_layer_conv_state is not None:
                prefix_layer_conv_state = jnp.where(
                    conv_keep,
                    prefix_layer_conv_state,
                    layer_conv_state,
                )

        # Update cache with new recurrent state and conv state for this layer
        if hybrid_state.recurrent_state is not None:
            if hybrid_state_is_layer:
                new_recurrent_state = new_recurrent_state_single
                new_conv_state = new_layer_conv_state
            else:
                new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(new_recurrent_state_single)
                new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(new_layer_conv_state)
        else:
            new_recurrent_state = new_recurrent_state_single[jnp.newaxis, :, :, :, :]  # Add layer dim
            new_conv_state = new_layer_conv_state[jnp.newaxis, :, :, :]  # Add layer dim
        
        hybrid_state = replace(
            hybrid_state,
            conv_state=new_conv_state,
            recurrent_state=new_recurrent_state,
        )
        prefix_layer_state = (
            HybridLayerState(
                conv_state=prefix_layer_conv_state,
                recurrent_state=prefix_recurrent_state_single,
            )
            if return_prefix_state or return_first_prefix_state
            else None
        )
        
        # core_attn_out is [B, H, T, D_v] - transpose to [B, T, H, D_v] for reshaping
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)  # [B, T, H, D_v]
        
    else:
        # === PREFILL MODE (Metal-compatible implementation) ===
        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # [B, D, T]
        if packed_token_row_ids is not None:
            if return_prefix_state or return_first_prefix_state:
                raise ValueError("packed prefill does not support prefix hybrid-state returns")
            if packed_query_start_loc is None:
                raise ValueError("packed_query_start_loc is required for packed GDN prefill")
            if batch != 1:
                raise ValueError("packed GDN prefill expects token tensors shaped [1, token_bucket, ...]")

            row_count = int(packed_query_start_loc.shape[0]) - 1
            token_rows = packed_token_row_ids.reshape(-1).astype(jnp.int32)
            valid_tokens = jnp.arange(seq_len, dtype=jnp.int32) < packed_query_start_loc[-1].astype(jnp.int32)
            safe_rows = jnp.clip(token_rows, 0, row_count - 1)
            conv_weight = params["conv1d_weight"].reshape(conv_dim, config.linear_conv_kernel_size)
            conv_bias = params.get("conv1d_bias")
            if use_cached_prefill:
                initial_conv_state = (
                    hybrid_state.conv_state
                    if hybrid_state_is_layer
                    else hybrid_state.conv_state[:, linear_layer_idx]
                )
            else:
                initial_conv_state = jnp.zeros(
                    (row_count, conv_dim, config.linear_conv_kernel_size),
                    dtype=mixed_qkv_t.dtype,
                )

            max_row_tokens = (
                max(tuple(getattr(config, "prefill_buckets", ()) or ()))
                if tuple(getattr(config, "prefill_buckets", ()) or ())
                else seq_len
            )
            conv_out, final_conv_state = _packed_causal_conv1d_prefill(
                mixed_qkv,
                initial_conv_state,
                conv_weight,
                conv_bias,
                packed_token_row_ids,
                packed_query_start_loc,
                max_row_tokens=max_row_tokens,
            )

            initial_recurrent = (
                (
                    hybrid_state.recurrent_state
                    if hybrid_state_is_layer
                    else hybrid_state.recurrent_state[:, linear_layer_idx]
                )
                if use_cached_prefill
                else jnp.zeros(
                    (
                        row_count,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                        config.linear_key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            )

            use_packed_post_conv_prefill = (
                gdn_prefill_post_conv_enabled(config)
                and not use_recurrent_prefill
            )
            if use_packed_post_conv_prefill:
                core_attn_out, final_state = backend.gated_delta_packed_prefill_post_conv(
                    conv_out,
                    a,
                    b,
                    params["A"],
                    params["dt_bias"],
                    packed_query_start_loc,
                    num_key_heads=config.linear_num_key_heads,
                    num_value_heads=config.linear_num_value_heads,
                    key_head_dim=config.linear_key_head_dim,
                    value_head_dim=config.linear_value_head_dim,
                    chunk_size=config.linear_chunk_size,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=config.use_qk_norm_in_gdn,
                    max_row_tokens=max_row_tokens,
                )
                core_attn_out = core_attn_out.astype(dtype)
            else:
                query = conv_out[:, :, :key_dim].reshape(
                    1,
                    seq_len,
                    config.linear_num_key_heads,
                    config.linear_key_head_dim,
                )
                key = conv_out[:, :, key_dim:key_dim * 2].reshape(
                    1,
                    seq_len,
                    config.linear_num_key_heads,
                    config.linear_key_head_dim,
                )
                value = conv_out[:, :, key_dim * 2:].reshape(
                    1,
                    seq_len,
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                )
                beta = nn.sigmoid(b)
                g = -params["A"] * nn.softplus(a + params["dt_bias"])

                if v_heads_per_k > 1:
                    query = jnp.repeat(query, v_heads_per_k, axis=2)
                    key = jnp.repeat(key, v_heads_per_k, axis=2)

                query_tokens = query[0].astype(jnp.float32)
                key_tokens = key[0].astype(jnp.float32)
                if config.use_qk_norm_in_gdn:
                    query_tokens = l2norm(query_tokens, axis=-1, eps=1e-6)
                    key_tokens = l2norm(key_tokens, axis=-1, eps=1e-6)
                query_tokens = query_tokens * (1.0 / jnp.sqrt(config.linear_key_head_dim))
                value_tokens = value[0].astype(jnp.float32)
                g_tokens = g[0].astype(jnp.float32)
                beta_tokens = beta[0].astype(jnp.float32)

                def recurrent_scan(state, inputs):
                    row, valid, q_t, k_t, v_t, g_t, beta_t = inputs
                    previous_row_state = state[row]
                    decay = jnp.exp(g_t)[:, None, None]
                    decayed_state = previous_row_state * decay
                    kv_mem = jnp.einsum("hvk,hk->hv", decayed_state, k_t)
                    delta = (v_t - kv_mem) * beta_t[:, None]
                    next_row_state = decayed_state + delta[:, :, None] * k_t[:, None, :]
                    out_t = jnp.einsum("hvk,hk->hv", next_row_state, q_t)
                    next_row_state = jnp.where(valid, next_row_state, previous_row_state)
                    state = state.at[row].set(next_row_state)
                    out_t = jnp.where(valid, out_t, jnp.zeros_like(out_t))
                    return state, out_t

                final_state, recurrent_out_tokens = lax.scan(
                    recurrent_scan,
                    initial_recurrent.astype(jnp.float32),
                    (
                        safe_rows,
                        valid_tokens,
                        query_tokens,
                        key_tokens,
                        value_tokens,
                        g_tokens,
                        beta_tokens,
                    ),
                )
                core_attn_out = recurrent_out_tokens[None, :, :, :].astype(dtype)

            if (
                hybrid_state is not None
                and hybrid_state.conv_state is not None
                and hybrid_state.recurrent_state is not None
            ):
                if hybrid_state_is_layer:
                    new_conv_state = final_conv_state.astype(dtype)
                    new_recurrent_state = final_state
                else:
                    new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(
                        final_conv_state.astype(dtype)
                    )
                    new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(final_state)
                hybrid_state = replace(
                    hybrid_state,
                    conv_state=new_conv_state,
                    recurrent_state=new_recurrent_state,
                )

            core_attn_out = core_attn_out.reshape(
                batch * seq_len,
                -1,
                config.linear_value_head_dim,
            )
            z_packed = z.reshape(batch * seq_len, -1, config.linear_value_head_dim)
            core_attn_out = _stable_rmsnorm_fp32(
                core_attn_out,
                params["norm_weight"],
                config.rms_norm_eps,
            )
            core_attn_out = core_attn_out * nn.silu(z_packed)
            core_attn_out = core_attn_out.reshape(batch, seq_len, -1)
            attn_out = _tokenwise_decode_dot(
                core_attn_out.astype(dtype),
                params["out_proj"],
                force_width1=False,
            )
            if hybrid_state is not None:
                return attn_out, hybrid_state
            return attn_out
        elif use_cached_prefill:
            layer_conv_state = (
                hybrid_state.conv_state
                if hybrid_state_is_layer
                else hybrid_state.conv_state[:, linear_layer_idx]
            )
            conv_input = jnp.concatenate([layer_conv_state, mixed_qkv_t], axis=-1)
            conv_out = causal_conv1d_metal(
                conv_input,
                params["conv1d_weight"],
                params.get("conv1d_bias"),
                activation="silu",
            )[:, :, -seq_len:]
            prefix_layer_conv_state = None
            if return_prefix_state:
                kernel_size = config.linear_conv_kernel_size
                step_starts = jnp.arange(seq_len, dtype=jnp.int32)[:, None] + 1
                gather_idx = step_starts + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
                gather_idx = jnp.broadcast_to(
                    gather_idx[None, :, None, :],
                    (batch, seq_len, conv_dim, kernel_size),
                )
                conv_input_expanded = jnp.broadcast_to(
                    conv_input[:, None, :, :],
                    (batch, seq_len, conv_dim, conv_input.shape[-1]),
                )
                prefix_layer_conv_state = jnp.take_along_axis(
                    conv_input_expanded,
                    gather_idx,
                    axis=3,
                )
        else:
            # Use Metal-compatible conv1d (no lax.conv_general_dilated)
            conv_out = causal_conv1d_metal(
                mixed_qkv_t,
                params["conv1d_weight"],
                params.get("conv1d_bias"),
                activation="silu",
            )
        conv_out = conv_out.transpose(0, 2, 1)  # [B, T, D]
        
        initial_recurrent = (
            (
                hybrid_state.recurrent_state
                if hybrid_state_is_layer
                else hybrid_state.recurrent_state[:, linear_layer_idx]
            )
            if use_cached_prefill
            else None
        )
        prefix_recurrent_state_single = None
        use_post_conv_prefill = (
            gdn_prefill_post_conv_enabled(config)
            and not use_recurrent_prefill
            and not return_prefix_state
            and not return_first_prefix_state
        )
        if use_post_conv_prefill:
            core_attn_out, final_state = backend.gated_delta_prefill_post_conv(
                conv_out,
                a,
                b,
                params["A"],
                params["dt_bias"],
                valid_token_mask,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                chunk_size=config.linear_chunk_size,
                initial_state=initial_recurrent,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            query = conv_out[:, :, :key_dim].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            key = conv_out[:, :, key_dim:key_dim*2].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            value = conv_out[:, :, key_dim*2:].reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)

            beta = nn.sigmoid(b)
            g = -params["A"] * nn.softplus(a + params["dt_bias"])

            if valid_token_mask is not None:
                valid = valid_token_mask.astype(jnp.bool_)
                query = jnp.where(valid[:, :, None, None], query, 0.0)
                key = jnp.where(valid[:, :, None, None], key, 0.0)
                value = jnp.where(valid[:, :, None, None], value, 0.0)
                beta = jnp.where(valid[:, :, None], beta, 0.0)
                g = jnp.where(valid[:, :, None], g, 0.0)

            if v_heads_per_k > 1:
                query = jnp.repeat(query, v_heads_per_k, axis=2)
                key = jnp.repeat(key, v_heads_per_k, axis=2)

            # Transpose to [B, H, T, D] format for chunk_gated_delta_rule
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            g = g.transpose(0, 2, 1)
            beta = beta.transpose(0, 2, 1)

            if use_recurrent_prefill:
                # Small cached suffixes can use the recurrent path directly and remain
                # aligned with iterative decode.
                if return_prefix_state and use_cached_prefill:
                    core_attn_out, final_state, recurrent_state_steps = jax_recurrent_gated_delta_rule(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        initial_state=initial_recurrent,
                        use_qk_l2norm_in_kernel=True,
                        return_state_sequence=True,
                    )
                    prefix_recurrent_state_single = recurrent_state_steps
                else:
                    core_attn_out, final_state = backend.gated_delta_decode(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        initial_state=initial_recurrent,
                        use_qk_l2norm_in_kernel=True,
                    )
            else:
                # Longer prefill chunks use chunked prefill to amortize work.
                core_attn_out, final_state = backend.gated_delta_prefill(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    chunk_size=config.linear_chunk_size,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                )
        
        # Save final state to cache for decode mode
        if (
            hybrid_state is not None
            and hybrid_state.conv_state is not None
            and hybrid_state.recurrent_state is not None
        ):
            # Extract the last real kernel_size inputs. Bucket padding is not
            # part of the convolution history used by recurrent decode.
            kernel_size = config.linear_conv_kernel_size
            prev_conv_state = (
                hybrid_state.conv_state
                if hybrid_state_is_layer
                else hybrid_state.conv_state[:, linear_layer_idx]
            )

            if valid_token_mask is not None:
                valid = valid_token_mask.astype(jnp.bool_)
                # Keep prior cache context and only write new valid positions.
                masked_mixed_qkv_t = jnp.where(
                    valid[:, None, :],
                    mixed_qkv_t,
                    jnp.zeros_like(mixed_qkv_t),
                )
            else:
                masked_mixed_qkv_t = mixed_qkv_t

            valid_lens = (
                valid_token_mask.astype(jnp.int32).sum(axis=1)
                if valid_token_mask is not None
                else jnp.full((batch,), seq_len, dtype=jnp.int32)
            )
            if use_cached_prefill:
                conv_input = jnp.concatenate([prev_conv_state, masked_mixed_qkv_t], axis=-1)
                gather_start = valid_lens
            else:
                conv_input = jnp.concatenate(
                    [
                        jnp.zeros((batch, conv_dim, kernel_size), dtype=masked_mixed_qkv_t.dtype),
                        masked_mixed_qkv_t,
                    ],
                    axis=-1,
                )
                gather_start = valid_lens
            gather_idx = gather_start[:, None] + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
            gather_idx = jnp.broadcast_to(gather_idx[:, None, :], (batch, conv_dim, kernel_size))
            layer_conv_state = jnp.take_along_axis(conv_input, gather_idx, axis=2)

            if hybrid_state_is_layer:
                new_recurrent_state = final_state
                new_conv_state = layer_conv_state.astype(dtype)
            else:
                new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(final_state)
                new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(layer_conv_state.astype(dtype))
            hybrid_state = replace(
                hybrid_state,
                conv_state=new_conv_state,
                recurrent_state=new_recurrent_state,
            )
            if return_prefix_state and use_cached_prefill:
                prefix_layer_state = HybridLayerState(
                    conv_state=prefix_layer_conv_state.astype(dtype)
                    if prefix_layer_conv_state is not None
                    else None,
                    recurrent_state=prefix_recurrent_state_single,
                )
        
        # Output is [B, H, T, D] - transpose to [B, T, H, D] for reshaping
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)  # [B, H, T, D] -> [B, T, H, D]
    
    # === OUTPUT PROCESSING (same for both modes) ===
    # Reshape to apply per-head gated norm
    core_attn_out = core_attn_out.reshape(batch * seq_len, -1, config.linear_value_head_dim)  # [B*T, H, D]
    z = z.reshape(batch * seq_len, -1, config.linear_value_head_dim)  # [B*T, H, D]
    
    # Apply gated RMSNorm per head. HF-style RMSNorm computes the reduction in
    # fp32, which also removes one source of width-dependent TPU bf16 drift.
    core_attn_out = _stable_rmsnorm_fp32(core_attn_out, params["norm_weight"], config.rms_norm_eps)
    core_attn_out = core_attn_out * nn.silu(z)
    
    # Reshape back and project
    core_attn_out = core_attn_out.reshape(batch, seq_len, -1)
    core_attn_out_proj = core_attn_out.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    attn_out = _tokenwise_decode_dot(
        core_attn_out_proj,
        params["out_proj"],
        force_width1=(not is_prefill) and seq_len > 1 and _force_width1_decode_math(),
    )
    
    if use_cached:
        if return_prefix_state:
            return attn_out, hybrid_state, prefix_layer_state
        return attn_out, hybrid_state
    elif hybrid_state is not None:
        # Prefill mode with cache - return state for decode
        if return_prefix_state and prefix_layer_state is not None:
            return attn_out, hybrid_state, prefix_layer_state
        return attn_out, hybrid_state
    else:
        # No cache - just return output
        return attn_out


def full_attention_block(
    x,
    params,
    positions,
    mask,
    config,
    kv_cache_state: Optional[KVCacheState] = None,
    is_prefill: bool = True,
    layer_idx: int = 0,
    attention_metadata: Optional[AttentionMetadata] = None,
    backend: Optional[InferenceBackend] = None,
    return_kv_prewrite: bool = False,
):
    """Full attention block with optional KV cache support.
    
    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        params: Layer parameters
        positions: Position IDs [batch, seq_len]
        mask: Causal mask [seq_len, seq_len]
        config: Model config
        kv_cache_state: Optional KV cache state (None for no cache)
        is_prefill: Whether this is prefill (vs decode)
        layer_idx: Layer index for per-layer KV cache
    
    Returns:
        tuple: (output, updated_kv_cache_state)
    """
    batch, seq_len, _ = x.shape
    
    # Cast to target dtype (bfloat16 for CPU/CUDA, float16 for Metal)
    dtype = config.get_dtype()
    x_cast = x.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    valid_token_mask = None
    compact_prefill_tokens = None
    if is_prefill and attention_metadata is not None:
        if attention_metadata.token_row_ids is not None:
            valid_token_mask = (
                jnp.arange(seq_len, dtype=jnp.int32)[None, :]
                < attention_metadata.query_start_loc[-1].astype(jnp.int32)
            )
        else:
            query_lens = jnp.diff(attention_metadata.query_start_loc).astype(jnp.int32)
            valid_token_mask = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < query_lens[:, None]
        compact_prefill_tokens = (
            int(attention_metadata.num_prefill_tokens)
            if isinstance(attention_metadata.num_prefill_tokens, int)
            else None
        )

    def _proj(inp: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        """Project a [B, T, D] tensor with a linear matrix [D, O].

        Using an explicit flatten+dot keeps row-wise projection numerically aligned
        between full-sequence and single-token calls across backends.
        """
        if not is_prefill and seq_len > 1 and _force_width1_decode_math():
            return _tokenwise_decode_dot(inp, weight, force_width1=True)
        if is_prefill:
            return _compact_prefill_dot_if_enabled(
                inp,
                weight,
                valid_token_mask,
                compact_prefill_tokens,
                enabled=_enable_compact_prefill_full_attn_proj(config),
            )
        out = jnp.dot(inp.reshape(-1, inp.shape[-1]), weight)
        return out.reshape(batch, seq_len, -1)
    
    # Qwen3.5 full attention uses fused Q + gate projection
    # q_proj: [hidden_size, hidden_size] -> output split into query and gate
    # query: [batch, seq_len, num_attention_heads * head_dim]
    # gate: [batch, seq_len, num_attention_heads * head_dim]
    attn_out_dim = config.num_attention_heads * config.head_dim
    force_width1_full_attn = (
        (not is_prefill)
        and seq_len > 1
        and os.environ.get("NANO_VLLM_JAX_FORCE_WIDTH1_FULL_ATTN", "0")
        in {"1", "true", "yes", "on", "True"}
    )

    if force_width1_full_attn:
        query_parts = []
        k_parts = []
        v_parts = []
        gate_parts = []
        for t in range(seq_len):
            x_t = x_cast[:, t : t + 1, :]
            hidden_t = x_t.shape[-1]
            q_gate_t = jnp.dot(x_t.reshape(batch, hidden_t), params["q_proj"])[:, None, :]
            q_gate_t = q_gate_t.reshape(batch, 1, config.num_attention_heads, 2 * config.head_dim)
            query_t, gate_t = jnp.split(q_gate_t, 2, axis=-1)
            k_t = jnp.dot(x_t.reshape(batch, hidden_t), params["k_proj"])[:, None, :].reshape(
                batch, 1, config.num_key_value_heads, config.head_dim
            )
            v_t = jnp.dot(x_t.reshape(batch, hidden_t), params["v_proj"])[:, None, :].reshape(
                batch, 1, config.num_key_value_heads, config.head_dim
            )

            query_t = rms_norm(query_t, params["q_norm"], config.rms_norm_eps).transpose(0, 2, 1, 3)
            k_t = rms_norm(k_t, params["k_norm"], config.rms_norm_eps).transpose(0, 2, 1, 3)
            v_t = v_t.transpose(0, 2, 1, 3)
            pos_t = positions[:, :, t : t + 1] if positions.ndim == 3 else positions[:, t : t + 1]
            query_t = apply_rope(
                query_t,
                pos_t,
                config.head_dim,
                config.rope_theta,
                config.partial_rotary_factor,
                layout="BHTD",
                mrope_section=config.mrope_section,
            )
            k_t = apply_rope(
                k_t,
                pos_t,
                config.head_dim,
                config.rope_theta,
                config.partial_rotary_factor,
                layout="BHTD",
                mrope_section=config.mrope_section,
            )
            query_parts.append(query_t)
            k_parts.append(k_t)
            v_parts.append(v_t)
            gate_parts.append(gate_t.reshape(batch, 1, -1))
        query = jnp.concatenate(query_parts, axis=2)
        k = jnp.concatenate(k_parts, axis=2)
        v = jnp.concatenate(v_parts, axis=2)
        gate = jnp.concatenate(gate_parts, axis=1)
    else:
        use_packed_decode_qkv = _use_full_attention_decode_packed_qkv(
            params,
            is_prefill=is_prefill,
            batch=batch,
            seq_len=seq_len,
        )
        use_packed_prefill_qkv = _use_full_attention_prefill_packed_qkv(
            params,
            is_prefill=is_prefill,
            config=config,
        )
        if use_packed_decode_qkv or use_packed_prefill_qkv:
            packed_qkv = _proj(x_cast, params[_FULL_ATTN_DECODE_QKV_PACKED_KEY])
            q_gate_end = attn_out_dim * 2
            k_end = q_gate_end + config.num_key_value_heads * config.head_dim
            q_gate, k_raw, v_raw = jnp.split(packed_qkv, [q_gate_end, k_end], axis=-1)
        else:
            q_gate = _proj(x_cast, params["q_proj"])
            k_raw = _proj(x_cast, params["k_proj"])
            v_raw = _proj(x_cast, params["v_proj"])

        q_gate_reshaped = q_gate.reshape(batch, seq_len, config.num_attention_heads, 2 * config.head_dim)
        query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
        gate = gate.reshape(batch, seq_len, -1)

        k = k_raw.reshape(
            batch, seq_len, config.num_key_value_heads, config.head_dim
        )
        v = v_raw.reshape(
            batch, seq_len, config.num_key_value_heads, config.head_dim
        )

        # Apply RMSNorm BEFORE transpose (on head dimension, in [B, T, H, D] layout)
        force_width1_norm = (not is_prefill) and _force_width1_decode_math()
        query = _decode_width1_rms_norm(
            query,
            params["q_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norm,
        )
        k = _decode_width1_rms_norm(
            k,
            params["k_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norm,
        )

        # Transpose to [B, H, T, D]
        query = query.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE (now in [B, H, T, D] layout)
        query = apply_rope(query, positions, config.head_dim, config.rope_theta, config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
        k = apply_rope(k, positions, config.head_dim, config.rope_theta, config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
    
    prewrite_k_cache_input = jnp.zeros((batch, seq_len, config.num_key_value_heads, config.head_dim), dtype=dtype)
    prewrite_v_cache_input = jnp.zeros((batch, seq_len, config.num_key_value_heads, config.head_dim), dtype=dtype)

    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    
    if kv_cache_state is not None:
        if backend is None:
            backend = select_backend("pure_jax", config=config)

        # Transpose K, V back to [B, T, K, H] for cache storage
        k_cache_input = k.transpose(0, 2, 1, 3)  # [B, T, K, H]
        v_cache_input = v.transpose(0, 2, 1, 3)  # [B, T, K, H]
        prewrite_k_cache_input = k_cache_input
        prewrite_v_cache_input = v_cache_input

        metadata = attention_metadata
        if metadata is None:
            metadata_positions = positions[0] if positions.ndim == 3 else positions
            metadata = backend.build_attention_metadata(
                positions=metadata_positions,
                block_tables=kv_cache_state.block_table,
                seq_lens=kv_cache_state.kv_lens,
                block_size=config.block_size,
                is_prefill=is_prefill,
            )
        # query is currently [batch, num_heads, seq_len, head_dim] (BHTD)
        # Backend attention expects [batch, seq_len, num_heads, head_dim] (BTNH)
        query_btnh = query.transpose(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]

        cache_storage, out = backend.write_kv_and_attention(
            layer_id=layer_idx,
            query=query_btnh,
            k=k_cache_input,
            v=v_cache_input,
            cache=kv_cache_state.storage,
            metadata=metadata,
            block_size=config.block_size,
            scale=1.0 / jnp.sqrt(config.head_dim),
            num_key_value_groups=num_key_value_groups,
            is_prefill=is_prefill,
        )
        
        # Reshape out to [batch, seq_len, hidden_dim]
        # For prefill: out is [batch, seq_len, hidden_dim]
        # For decode: out is [batch, 1, hidden_dim]
        # Both are already in the correct format
        
        # Update KV cache state (preserve linear attention states)
        kv_cache_state = replace(
            kv_cache_state,
            k_cache=cache_storage.k_cache,
            v_cache=cache_storage.v_cache,
            slot_mapping=metadata.slot_mapping,
        )
    else:
        # No cache - standard attention (for prefill without caching)
        k = jnp.repeat(k, num_key_value_groups, axis=1)
        v = jnp.repeat(v, num_key_value_groups, axis=1)
        
        attn = nn.softmax(jnp.einsum("bhtd,bhsd->bhts", query, k) / jnp.sqrt(config.head_dim) + mask[None, None, :, :].astype(query.dtype), -1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    out = out * nn.sigmoid(gate)
    out = out.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    out = _tokenwise_decode_dot(
        out,
        params["o_proj"],
        force_width1=(not is_prefill) and seq_len > 1 and _force_width1_decode_math(),
    )
    
    if return_kv_prewrite:
        return out, kv_cache_state, prewrite_k_cache_input, prewrite_v_cache_input
    return out, kv_cache_state


def transformer_block(
    x,
    params,
    positions,
    mask=None,
    layer_idx=0,
    config=None,
    kv_cache_state=None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    prefix_hybrid_state: Optional[HybridLayerState] = None,
    is_prefill=True,
    backend: Optional[InferenceBackend] = None,
    return_prefix_hybrid: bool = False,
    return_first_prefix_hybrid: bool = False,
    return_layer_hidden: bool = False,
    return_kv_prewrite: bool = False,
    return_layer_stages: bool = False,
    hybrid_state_is_layer: bool = False,
):
    """Matches HF Qwen3_5DecoderLayer - applies norms and residuals."""
    block_input = x
    residual = x

    # Apply input_layernorm (both full attention and linear attention)
    # HF applies input_layernorm before both layer types
    force_width1_norm = (not is_prefill) and x.ndim == 3 and _force_width1_decode_math()
    x = _decode_width1_rms_norm(
        x,
        params["input_norm"],
        config.rms_norm_eps,
        force_width1=force_width1_norm,
    )
    input_norm_out = x

    valid_token_mask = None
    if attention_metadata is not None:
        if attention_metadata.token_row_ids is not None:
            valid_token_mask = (
                jnp.arange(x.shape[1], dtype=jnp.int32)[None, :]
                < attention_metadata.query_start_loc[-1].astype(jnp.int32)
            )
        else:
            query_lens = jnp.diff(attention_metadata.query_start_loc).astype(jnp.int32)
            valid_token_mask = jnp.arange(x.shape[1], dtype=jnp.int32)[None, :] < query_lens[:, None]
    compact_prefill_tokens = (
        int(attention_metadata.num_prefill_tokens)
        if (
            is_prefill
            and attention_metadata is not None
            and isinstance(attention_metadata.num_prefill_tokens, int)
        )
        else None
    )

    # Apply attention/linear_attn
    layer_prewrite_k = jnp.zeros(
        (x.shape[0], x.shape[1], config.num_key_value_heads, config.head_dim),
        dtype=config.get_dtype(),
    )
    layer_prewrite_v = jnp.zeros(
        (x.shape[0], x.shape[1], config.num_key_value_heads, config.head_dim),
        dtype=config.get_dtype(),
    )
    if config.layer_types[layer_idx] == "full_attention":
        if return_kv_prewrite:
            x, kv_cache_state, layer_prewrite_k, layer_prewrite_v = full_attention_block(
                x,
                params,
                positions,
                mask,
                config,
                kv_cache_state,
                is_prefill,
                layer_idx=layer_idx,
                attention_metadata=attention_metadata,
                backend=backend,
                return_kv_prewrite=True,
            )
        else:
            x, kv_cache_state = full_attention_block(
                x,
                params,
                positions,
                mask,
                config,
                kv_cache_state,
                is_prefill,
                layer_idx=layer_idx,
                attention_metadata=attention_metadata,
                backend=backend,
            )
    else:
        result = gated_deltanet_block(
            x,
            params,
            positions,
            config,
            layer_idx,
            is_prefill=is_prefill,
            hybrid_state=hybrid_state,
            valid_token_mask=valid_token_mask,
            compact_prefill_tokens=compact_prefill_tokens,
            backend=backend,
            return_prefix_state=return_prefix_hybrid,
            return_first_prefix_state=return_first_prefix_hybrid,
            hybrid_state_is_layer=hybrid_state_is_layer,
            packed_token_row_ids=(
                attention_metadata.token_row_ids
                if is_prefill and attention_metadata is not None
                else None
            ),
            packed_query_start_loc=(
                attention_metadata.query_start_loc
                if is_prefill and attention_metadata is not None and attention_metadata.token_row_ids is not None
                else None
            ),
        )
        if isinstance(result, tuple):
            if (return_prefix_hybrid or return_first_prefix_hybrid) and len(result) == 3:
                x, hybrid_state, prefix_layer_state = result
                if prefix_hybrid_state is not None and prefix_layer_state is not None:
                    linear_layer_idx = len([l for l in config.linear_attn_layers if l < layer_idx])
                    if return_prefix_hybrid:
                        prefix_hybrid_state = replace(
                            prefix_hybrid_state,
                            conv_state=prefix_hybrid_state.conv_state.at[:, :, linear_layer_idx].set(
                                prefix_layer_state.conv_state
                            )
                            if prefix_hybrid_state.conv_state is not None
                            and prefix_layer_state.conv_state is not None
                            else prefix_hybrid_state.conv_state,
                            recurrent_state=prefix_hybrid_state.recurrent_state.at[:, :, linear_layer_idx].set(
                                prefix_layer_state.recurrent_state
                            )
                            if prefix_hybrid_state.recurrent_state is not None
                            and prefix_layer_state.recurrent_state is not None
                            else prefix_hybrid_state.recurrent_state,
                        )
                    else:
                        prefix_hybrid_state = replace(
                            prefix_hybrid_state,
                            conv_state=prefix_hybrid_state.conv_state.at[:, linear_layer_idx].set(
                                prefix_layer_state.conv_state
                            )
                            if prefix_hybrid_state.conv_state is not None
                            and prefix_layer_state.conv_state is not None
                            else prefix_hybrid_state.conv_state,
                            recurrent_state=prefix_hybrid_state.recurrent_state.at[:, linear_layer_idx].set(
                                prefix_layer_state.recurrent_state
                            )
                            if prefix_hybrid_state.recurrent_state is not None
                            and prefix_layer_state.recurrent_state is not None
                            else prefix_hybrid_state.recurrent_state,
                        )
            else:
                x, hybrid_state = result
        else:
            x = result

    attn_out = x

    # Add residual
    x = residual + x
    attn_residual_out = x

    # MLP path
    residual = x
    x = _decode_width1_rms_norm(
        x,
        params["ffn_norm"],
        config.rms_norm_eps,
        force_width1=force_width1_norm,
    )
    ffn_norm_out = x

    # MLP computation (stays in bfloat16)
    force_width1_dot = (not is_prefill) and x.ndim == 3 and x.shape[1] > 1 and _force_width1_decode_math()
    activation_fn = get_activation(config.hidden_act)
    if is_prefill:
        if _MLP_GATE_UP_PACKED_KEY in params:
            x = _compact_prefill_mlp_packed(
                x,
                params[_MLP_GATE_UP_PACKED_KEY],
                params["down_proj"],
                activation_fn,
                valid_token_mask,
                compact_prefill_tokens,
                config,
            )
        else:
            x = _compact_prefill_mlp(
                x,
                params["gate_proj"],
                params["up_proj"],
                params["down_proj"],
                activation_fn,
                valid_token_mask,
                compact_prefill_tokens,
                config,
            )
    else:
        x_proj = x.astype(_decode_projection_activation_dtype(x.shape[0], config))
        if _MLP_GATE_UP_PACKED_KEY in params:
            if (
                _decode_padded_gemm_gate_up_enabled(config)
                and _can_use_decode_padded_gemm(x_proj, params[_MLP_GATE_UP_PACKED_KEY], config)
            ):
                gate_up = _decode_padded_gemm_dot(x_proj, params[_MLP_GATE_UP_PACKED_KEY], config)
            else:
                gate_up = _tokenwise_decode_dot(
                    x_proj,
                    params[_MLP_GATE_UP_PACKED_KEY],
                    force_width1=force_width1_dot,
                )
            gate, up = jnp.split(gate_up, 2, axis=-1)
        else:
            if (
                _decode_padded_gemm_gate_up_enabled(config)
                and _can_use_decode_padded_gemm(x_proj, params["gate_proj"], config)
                and params["up_proj"].shape == params["gate_proj"].shape
            ):
                gate = _decode_padded_gemm_dot(x_proj, params["gate_proj"], config)
                up = _decode_padded_gemm_dot(x_proj, params["up_proj"], config)
            else:
                gate = _tokenwise_decode_dot(x_proj, params["gate_proj"], force_width1=force_width1_dot)
                up = _tokenwise_decode_dot(x_proj, params["up_proj"], force_width1=force_width1_dot)
        x = activation_fn(gate) * up
        if _can_use_decode_padded_gemm(x, params["down_proj"], config):
            x = _decode_padded_gemm_dot(x, params["down_proj"], config)
        else:
            x = _tokenwise_decode_dot(x, params["down_proj"], force_width1=force_width1_dot)
    mlp_out = x

    x = residual + x
    block_output = x

    outputs = [x, kv_cache_state, hybrid_state]
    if return_prefix_hybrid or return_first_prefix_hybrid:
        outputs.append(prefix_hybrid_state)
    if return_kv_prewrite:
        outputs.extend([layer_prewrite_k, layer_prewrite_v])
    if return_layer_stages:
        outputs.append(
            jnp.stack(
                [
                    block_input,
                    input_norm_out,
                    attn_out,
                    attn_residual_out,
                    ffn_norm_out,
                    mlp_out,
                    block_output,
                ],
                axis=0,
            )
        )
    return tuple(outputs)


def forward_step(
    tokens,
    params,
    config,
    *,
    positions=None,
    kv_cache_state: Optional[KVCacheState] = None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    is_prefill: bool = True,
    return_hidden: bool = False,
    return_hidden_with_logits: bool = False,
    last_logits_only: bool = False,
    logit_positions: Optional[jnp.ndarray] = None,
    backend: Optional[InferenceBackend] = None,
    return_prefix_hybrid: bool = False,
    return_first_prefix_hybrid: bool = False,
    return_layer_hidden: bool = False,
    return_kv_prewrite: bool = False,
    return_layer_stages: bool = False,
    hybrid_state_layerwise: bool = False,
):
    """Canonical forward step shared by cached and non-cached inference paths."""
    batch, seq_len = tokens.shape
    dtype = config.get_dtype()
    x = params.embed_tokens[tokens].astype(dtype)

    if positions is None:
        positions_2d = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
    elif positions.ndim == 3:
        positions_2d = positions[0]
    else:
        positions_2d = positions
    positions_mrope = jnp.stack([positions_2d, positions_2d, positions_2d], axis=0)
    mask = causal_mask(seq_len, seq_len)
    prefix_hybrid_state = None
    if return_prefix_hybrid and hybrid_state is not None:
        prefix_hybrid_state = HybridLayerState(
            conv_state=jnp.broadcast_to(
                hybrid_state.conv_state[:, None, ...],
                (batch, seq_len) + hybrid_state.conv_state.shape[1:],
            )
            if hybrid_state.conv_state is not None
            else None,
            recurrent_state=jnp.broadcast_to(
                hybrid_state.recurrent_state[:, None, ...],
                (batch, seq_len) + hybrid_state.recurrent_state.shape[1:],
            )
            if hybrid_state.recurrent_state is not None
            else None,
        )
    if return_first_prefix_hybrid and hybrid_state is not None:
        prefix_hybrid_state = HybridLayerState(
            conv_state=hybrid_state.conv_state,
            recurrent_state=hybrid_state.recurrent_state,
        )

    layer_hidden_states = [] if return_layer_hidden else None
    kv_prewrite_k_states = [] if return_kv_prewrite else None
    kv_prewrite_v_states = [] if return_kv_prewrite else None
    layer_stage_states = [] if return_layer_stages else None
    num_linear_layers = len(config.linear_attn_layers)
    use_layerwise_hybrid = (
        hybrid_state_layerwise
        and hybrid_state is not None
        and hybrid_state.conv_state is not None
        and hybrid_state.recurrent_state is not None
        and num_linear_layers > 0
        and not return_prefix_hybrid
        and not return_first_prefix_hybrid
        and not return_kv_prewrite
        and not return_layer_stages
    )
    if use_layerwise_hybrid:
        hybrid_conv_layers = [
            hybrid_state.conv_state[:, linear_idx]
            for linear_idx in range(num_linear_layers)
        ]
        hybrid_recurrent_layers = [
            hybrid_state.recurrent_state[:, linear_idx]
            for linear_idx in range(num_linear_layers)
        ]
        linear_layer_cursor = 0

    for i, lp in enumerate(params.layers):
        block_hybrid_state = hybrid_state
        block_hybrid_state_is_layer = False
        if use_layerwise_hybrid and config.layer_types[i] != "full_attention":
            block_hybrid_state = HybridLayerState(
                conv_state=hybrid_conv_layers[linear_layer_cursor],
                recurrent_state=hybrid_recurrent_layers[linear_layer_cursor],
            )
            block_hybrid_state_is_layer = True
        block_result = transformer_block(
            x,
            lp,
            positions_mrope,
            mask,
            i,
            config,
            kv_cache_state,
            attention_metadata=attention_metadata,
            hybrid_state=block_hybrid_state,
            prefix_hybrid_state=prefix_hybrid_state,
            is_prefill=is_prefill,
            backend=backend,
            return_prefix_hybrid=return_prefix_hybrid,
            return_first_prefix_hybrid=return_first_prefix_hybrid,
            return_kv_prewrite=return_kv_prewrite,
            return_layer_stages=return_layer_stages,
            hybrid_state_is_layer=block_hybrid_state_is_layer,
        )
        x, kv_cache_state, block_updated_hybrid_state = block_result[:3]
        if block_hybrid_state_is_layer:
            hybrid_conv_layers[linear_layer_cursor] = block_updated_hybrid_state.conv_state
            hybrid_recurrent_layers[linear_layer_cursor] = block_updated_hybrid_state.recurrent_state
            linear_layer_cursor += 1
        else:
            hybrid_state = block_updated_hybrid_state
        offset = 3
        if return_prefix_hybrid or return_first_prefix_hybrid:
            prefix_hybrid_state = block_result[offset]
            offset += 1
        if return_kv_prewrite:
            layer_prewrite_k = block_result[offset]
            layer_prewrite_v = block_result[offset + 1]
            offset += 2
        if return_layer_stages:
            layer_stage = block_result[offset]
        if layer_hidden_states is not None:
            layer_hidden_states.append(x)
        if kv_prewrite_k_states is not None:
            kv_prewrite_k_states.append(layer_prewrite_k)
            kv_prewrite_v_states.append(layer_prewrite_v)
        if layer_stage_states is not None:
            layer_stage_states.append(layer_stage)

    if use_layerwise_hybrid:
        hybrid_state = HybridLayerState(
            conv_state=jnp.stack(hybrid_conv_layers, axis=1),
            recurrent_state=jnp.stack(hybrid_recurrent_layers, axis=1),
        )

    hidden_pre = x
    layer_hidden_result = (
        jnp.stack(layer_hidden_states, axis=0)
        if layer_hidden_states is not None
        else None
    )
    kv_prewrite_k_result = (
        jnp.stack(kv_prewrite_k_states, axis=0)
        if kv_prewrite_k_states is not None
        else None
    )
    kv_prewrite_v_result = (
        jnp.stack(kv_prewrite_v_states, axis=0)
        if kv_prewrite_v_states is not None
        else None
    )
    layer_stage_result = (
        jnp.stack(layer_stage_states, axis=0)
        if layer_stage_states is not None
        else None
    )

    if return_hidden and not return_hidden_with_logits:
        if return_layer_hidden:
            if return_prefix_hybrid or return_first_prefix_hybrid:
                return hidden_pre, kv_cache_state, hybrid_state, prefix_hybrid_state, layer_hidden_result
            if return_kv_prewrite:
                return hidden_pre, kv_cache_state, hybrid_state, layer_hidden_result, kv_prewrite_k_result, kv_prewrite_v_result, layer_stage_result
            return hidden_pre, kv_cache_state, hybrid_state, layer_hidden_result
        if return_prefix_hybrid or return_first_prefix_hybrid:
            return hidden_pre, kv_cache_state, hybrid_state, prefix_hybrid_state
        return hidden_pre, kv_cache_state, hybrid_state

    x = rms_norm(x, params.norm_weight, config.rms_norm_eps)
    x = x.astype(jnp.float32)
    if last_logits_only:
        if logit_positions is None:
            x = x[:, -1:, :]
        else:
            gather_idx = jnp.clip(logit_positions, 0, seq_len - 1).astype(jnp.int32)
            gather_idx = gather_idx[:, None, None]
            gather_idx = jnp.broadcast_to(gather_idx, (batch, 1, x.shape[-1]))
            x = jnp.take_along_axis(x, gather_idx, axis=1)
    output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
    logits = _tokenwise_decode_dot(
        x,
        output_weight,
        force_width1=(not is_prefill) and seq_len > 1 and _force_width1_decode_math(),
    )
    if return_hidden:
        hidden_result = (hidden_pre, logits) if return_hidden_with_logits else hidden_pre
        if return_layer_hidden:
            if return_prefix_hybrid or return_first_prefix_hybrid:
                return hidden_result, kv_cache_state, hybrid_state, prefix_hybrid_state, layer_hidden_result
            if return_kv_prewrite:
                return hidden_result, kv_cache_state, hybrid_state, layer_hidden_result, kv_prewrite_k_result, kv_prewrite_v_result, layer_stage_result
            return hidden_result, kv_cache_state, hybrid_state, layer_hidden_result
        if return_prefix_hybrid or return_first_prefix_hybrid:
            return hidden_result, kv_cache_state, hybrid_state, prefix_hybrid_state
        return hidden_result, kv_cache_state, hybrid_state
    if return_layer_hidden:
        if return_prefix_hybrid or return_first_prefix_hybrid:
            return logits, kv_cache_state, hybrid_state, prefix_hybrid_state, layer_hidden_result
        if return_kv_prewrite:
            return logits, kv_cache_state, hybrid_state, layer_hidden_result, kv_prewrite_k_result, kv_prewrite_v_result, layer_stage_result
        return logits, kv_cache_state, hybrid_state, layer_hidden_result
    if return_prefix_hybrid or return_first_prefix_hybrid:
        return logits, kv_cache_state, hybrid_state, prefix_hybrid_state
    return logits, kv_cache_state, hybrid_state


def forward(
    tokens,
    params,
    config,
    kv_cache_state=None,
    is_prefill=True,
    return_hidden=False,
    return_hidden_with_logits: bool = False,
    last_logits_only: bool = False,
    logit_positions: Optional[jnp.ndarray] = None,
    positions=None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    backend: Optional[InferenceBackend] = None,
):
    """Compatibility wrapper over the canonical forward step."""
    if is_prefill and kv_cache_state is not None and hybrid_state is None:
        kv_cache_state = init_linear_attention_states(
            kv_cache_state,
            config,
            batch_size=tokens.shape[0],
        )
        hybrid_state = kv_cache_state.hybrid_state
    elif hybrid_state is None and kv_cache_state is not None:
        hybrid_state = kv_cache_state.hybrid_state

    result, updated_kv_state, updated_hybrid_state = forward_step(
        tokens,
        params,
        config,
        positions=positions,
        kv_cache_state=kv_cache_state,
        attention_metadata=attention_metadata,
        hybrid_state=hybrid_state,
        is_prefill=is_prefill,
        return_hidden=return_hidden,
        return_hidden_with_logits=return_hidden_with_logits,
        last_logits_only=last_logits_only,
        logit_positions=logit_positions,
        backend=backend,
    )

    if updated_kv_state is not None and updated_hybrid_state is not None:
        updated_kv_state = replace(
            updated_kv_state,
            conv_state=updated_hybrid_state.conv_state,
            recurrent_state=updated_hybrid_state.recurrent_state,
        )

    return result, updated_kv_state


class Qwen3_5:
    def __init__(self, config, key):
        self.config, self.params = config, init_params(key, config)
    
    def forward(self, tokens, kv_cache_state=None, is_prefill=True, backend: Optional[InferenceBackend] = None):
        """Forward pass with optional KV cache."""
        return forward(
            tokens,
            self.params,
            self.config,
            kv_cache_state=kv_cache_state,
            is_prefill=is_prefill,
            backend=backend,
        )
