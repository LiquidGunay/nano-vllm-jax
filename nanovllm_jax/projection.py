"""Shared projection helpers for the Qwen3.5 serving path."""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import lax

from nanovllm_jax.fastpath import KernelPlan
from nanovllm_jax.layers import rms_norm

_GDN_DECODE_IN_PROJ_PACKED_KEY = "in_proj_qkv_abz"
_FULL_ATTN_DECODE_QKV_PACKED_KEY = "qkv_proj_decode"
_MLP_GATE_UP_PACKED_KEY = "gate_up_proj"
def _causal_conv1d(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    activation: str = "silu",
) -> jnp.ndarray:
    """Reference grouped causal 1D convolution for GDN projections."""

    batch, dim, seq_len = x.shape
    kernel_size = int(weight.shape[-1])
    padded = jnp.pad(x, ((0, 0), (0, 0), (kernel_size - 1, 0)))
    out = jnp.zeros((batch, dim, seq_len), dtype=x.dtype)
    for k in range(kernel_size):
        out = out + padded[:, :, k : k + seq_len] * weight[:, k : k + 1]
    if bias is not None:
        out = out + bias[:, None]
    if activation == "silu":
        return jax.nn.silu(out)
    if activation == "relu":
        return jax.nn.relu(out)
    return out


def _tokenwise_decode_dot(x: jnp.ndarray, weight: jnp.ndarray, *, force_width1: bool = False) -> jnp.ndarray:
    """Apply a tokenwise linear with width-1 matmul shapes for multi-token decode.

    BF16 matmuls can be shape dependent. For cached decode, the first token in
    a width-2 route must match a standalone width-1 decode before any commit
    decision. Slicing the sequence dimension keeps the matmul shape aligned
    with sequential decode while preserving one compiled graph.
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
    conv_all = _causal_conv1d(
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

    The compact decode route evaluates a two-token block, but its first
    token must match the canonical single-token decode path exactly enough for
    greedy parity. BF16 reductions can be shape-dependent even when each token
    is independent along the normalized dimension, so run the same `[B, 1, ...]`
    RMSNorm shape used by baseline decode and concatenate the token results
    inside the compiled graph.
    """
    stable_decode_norm = False
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
    parts = [norm_fn(x[:, t : t + 1, ...], weight, eps) for t in range(x.shape[1])]
    return jnp.concatenate(parts, axis=1)


def _force_width1_decode_math() -> bool:
    """Use width-1-shaped matmuls in multi-token decode by default.

    Multi-token decode must match the canonical width-1 baseline token for
    every physical batch shape. BF16 matmuls are shape-sensitive enough that
    wider decode can diverge at larger batches unless tokenwise projections use
    the width-1 decode shape.
    """
    return True


def _lm_head_decode_activation_dtype(plan: KernelPlan) -> jnp.dtype:
    value = plan.lm_head_decode_act_dtype
    if value in {"", "0", "false", "no", "off", "none", "fp32", "float32"}:
        return jnp.float32
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    raise ValueError(
        f"lm_head_decode_act_dtype must be fp32 or bf16, got {value!r}"
    )


def _decode_padded_gemm_enabled(plan: KernelPlan) -> bool:
    return plan.decode_padded_gemm


def _decode_padded_gemm_gate_up_enabled(plan: KernelPlan) -> bool:
    return plan.decode_padded_gemm_gate_up


def _decode_rms_padded_gemm_enabled(plan: KernelPlan) -> bool:
    return plan.decode_rms_padded_gemm


def _decode_padded_gemm_rows(plan: KernelPlan) -> int:
    rows = plan.decode_padded_gemm_rows
    if rows < 1:
        raise ValueError("decode_padded_gemm_rows must be positive")
    return rows


def _decode_padded_gemm_max_out_dim(plan: KernelPlan) -> int:
    out_dim = plan.decode_padded_gemm_max_out_dim
    if out_dim < 1:
        raise ValueError("decode_padded_gemm_max_out_dim must be positive")
    return out_dim


def _lm_head_topk_impl(plan: KernelPlan) -> str:
    value = plan.lm_head_sampled
    if value in {"", "0", "false", "no", "off", "none", "jax", "reference"}:
        return "jax"
    raise ValueError(f"lm_head_topk_impl must be jax, got {value!r}")


def _lm_head_greedy_top1_impl(plan: KernelPlan) -> str:
    value = plan.lm_head_greedy
    if value in {"", "0", "false", "no", "off", "none", "jax", "reference"}:
        return "jax"
    if value in {"triton", "triton_tensorcore", "triton_epilogue"}:
        return "triton"
    raise ValueError(f"lm_head_greedy_top1_impl must be jax or triton, got {value!r}")


def _can_use_decode_padded_gemm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    plan: KernelPlan,
) -> bool:
    rows = _decode_padded_gemm_rows(plan)
    return (
        _decode_padded_gemm_enabled(plan)
        and x.ndim == 3
        and weight.ndim == 2
        and int(x.shape[0]) <= rows
        and int(x.shape[1]) == 1
        and int(x.shape[-1]) == int(weight.shape[0])
        and int(weight.shape[1]) <= _decode_padded_gemm_max_out_dim(plan)
    )


def _decode_padded_gemm_dot(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    plan: KernelPlan,
) -> jnp.ndarray:
    """Run a small-B decode projection through a row-padded GEMM."""
    batch = int(x.shape[0])
    hidden = int(x.shape[-1])
    out_dim = int(weight.shape[1])
    rows = _decode_padded_gemm_rows(plan)
    x_rows = jnp.reshape(x, (batch, hidden))
    if batch < rows:
        x_padded = jnp.pad(x_rows, ((0, rows - batch), (0, 0)))
    else:
        x_padded = x_rows
    out = jnp.dot(x_padded, weight)
    return out[:batch, :].reshape(batch, 1, out_dim)


def _can_use_decode_rms_padded_gemm(
    x: jnp.ndarray,
    norm_weight: jnp.ndarray,
    weight: jnp.ndarray,
    plan: KernelPlan,
) -> bool:
    rows = _decode_padded_gemm_rows(plan)
    return (
        _decode_rms_padded_gemm_enabled(plan)
        and _decode_padded_gemm_gate_up_enabled(plan)
        and _decode_projection_activation_dtype(int(x.shape[0]), plan) == jnp.bfloat16
        and x.ndim == 3
        and norm_weight.ndim == 1
        and weight.ndim == 2
        and int(x.shape[0]) <= rows
        and int(x.shape[1]) == 1
        and int(x.shape[-1]) == int(norm_weight.shape[0])
        and int(x.shape[-1]) == int(weight.shape[0])
        and weight.dtype == jnp.bfloat16
        and int(weight.shape[1]) <= _decode_padded_gemm_max_out_dim(plan)
    )


def _decode_rms_padded_gemm_dot(
    x: jnp.ndarray,
    norm_weight: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float,
    plan: KernelPlan,
) -> jnp.ndarray:
    from nanovllm_jax.kernels.decode_reductions import triton_decode_rms_padded_gemm

    return triton_decode_rms_padded_gemm(
        x,
        norm_weight,
        weight,
        eps=eps,
        rows=_decode_padded_gemm_rows(plan),
    )


def _decode_projection_activation_dtype(
    batch_size: int | None,
    plan: KernelPlan,
) -> jnp.dtype:
    value = plan.decode_proj_act_dtype
    if value in {"", "0", "false", "no", "off", "none", "fp32", "float32"}:
        return jnp.float32
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if value in {"bf16_single_seq", "bfloat16_single_seq", "bf16_single_sequence"}:
        return jnp.bfloat16 if batch_size == 1 else jnp.float32
    raise ValueError(
        f"decode_proj_act_dtype must be fp32, bf16, or bf16_single_seq, got {value!r}"
    )


def _use_gdn_decode_packed_in_proj(
    params: Dict[str, jnp.ndarray],
    *,
    is_prefill: bool,
    batch: int,
    seq_len: int,
    plan: KernelPlan,
) -> bool:
    width_one = plan.gdn_width1_packed_input_projection
    return (
        not is_prefill
        and _GDN_DECODE_IN_PROJ_PACKED_KEY in params
        and (seq_len == 1 or plan.gdn_decode != "off")
        and (batch > 1 or width_one)
    )


def _use_gdn_prefill_packed_in_proj(
    params: Dict[str, jnp.ndarray],
    *,
    is_prefill: bool,
    plan: KernelPlan,
) -> bool:
    return (
        is_prefill
        and _GDN_DECODE_IN_PROJ_PACKED_KEY in params
        and _enable_compact_prefill_in_proj_qkv(plan)
        and _enable_compact_prefill_gdn_z(plan)
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
    plan: KernelPlan,
) -> bool:
    return (
        is_prefill
        and _FULL_ATTN_DECODE_QKV_PACKED_KEY in params
        and _enable_compact_prefill_full_attn_proj(plan)
    )


def _enable_compact_prefill_in_proj_qkv(plan: KernelPlan) -> bool:
    """Compact true prefill tokens for the GDN QKV input projection."""
    return plan.compact_prefill_in_proj_qkv


def _enable_compact_prefill_mlp(plan: KernelPlan) -> bool:
    """Compact true prefill tokens for tokenwise MLP projections."""
    return plan.compact_prefill_mlp


def _enable_compact_prefill_gdn_z(plan: KernelPlan) -> bool:
    """Compact true prefill tokens for the GDN Z input projection."""
    return plan.compact_prefill_gdn_z


def _enable_compact_prefill_full_attn_proj(plan: KernelPlan) -> bool:
    """Compact true prefill tokens for full-attention Q/K/V projections."""
    return plan.compact_prefill_full_attn_proj


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
    plan: KernelPlan,
) -> jnp.ndarray:
    """Run a tokenwise projection only on true ragged prefill tokens."""
    return _compact_prefill_dot_if_enabled(
        x,
        weight,
        valid_token_mask,
        compact_num_tokens,
        enabled=_enable_compact_prefill_in_proj_qkv(plan),
    )


def _compact_prefill_mlp(
    x: jnp.ndarray,
    gate_weight: jnp.ndarray,
    up_weight: jnp.ndarray,
    down_weight: jnp.ndarray,
    activation_fn,
    valid_token_mask: Optional[jnp.ndarray],
    compact_num_tokens: Optional[int],
    plan: KernelPlan,
) -> jnp.ndarray:
    """Run tokenwise prefill MLP only on true ragged tokens."""
    if (
        not _enable_compact_prefill_mlp(plan)
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
    plan: KernelPlan,
) -> jnp.ndarray:
    """Run tokenwise prefill MLP only on true ragged tokens with packed gate/up."""
    if (
        not _enable_compact_prefill_mlp(plan)
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
