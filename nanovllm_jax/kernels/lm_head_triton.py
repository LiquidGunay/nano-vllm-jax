"""Triton LM-head projection-plus-top1 kernels.

This module is intentionally narrow: greedy decode only, width 1 only.  The
stage-1 kernel keeps the projection in a tensor-core `tl.dot` tile and writes
one top-1 candidate per row/vocab tile instead of full logits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

from nanovllm_jax.kernels.gdn_fla_triton import _configure_triton_runtime


_configure_triton_runtime()


@triton.jit
def _lm_head_top1_stage1_kernel(
    hidden,
    weight,
    partial_values,
    partial_indices,
    batch_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    REDUCE_CAST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    vocab_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, hidden_dim, BLOCK_K):
        k = k_start + k_offsets
        a = tl.load(
            hidden + row_offsets[:, None] * hidden_dim + k[None, :],
            mask=(row_offsets[:, None] < batch_size) & (k[None, :] < hidden_dim),
            other=0.0,
        )
        b = tl.load(
            weight + k[:, None] * vocab_size + vocab_offsets[None, :],
            mask=(k[:, None] < hidden_dim) & (vocab_offsets[None, :] < vocab_size),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    if REDUCE_CAST == 1:
        reduce_acc = acc.to(tl.bfloat16).to(tl.float32)
    elif REDUCE_CAST == 2:
        reduce_acc = acc.to(tl.float16).to(tl.float32)
    else:
        reduce_acc = acc

    valid_vocab = vocab_offsets < vocab_size
    logits = tl.where(valid_vocab[None, :], reduce_acc, -float("inf"))
    block_max = tl.max(logits, axis=1)
    vocab_ids = tl.broadcast_to(vocab_offsets[None, :], (BLOCK_M, BLOCK_N))
    tie_ids = tl.where(logits == block_max[:, None], vocab_ids, vocab_size)
    block_arg = tl.min(tie_ids, axis=1)

    valid_rows = row_offsets < batch_size
    out_offsets = row_offsets * num_vocab_blocks + pid_n
    tl.store(partial_values + out_offsets, block_max, mask=valid_rows)
    tl.store(partial_indices + out_offsets, block_arg, mask=valid_rows)


@triton.jit
def _lm_head_top1_stage2_kernel(
    partial_values,
    partial_indices,
    token_ids,
    num_vocab_blocks: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_B)
    mask = offsets < num_vocab_blocks
    values = tl.load(
        partial_values + row * num_vocab_blocks + offsets,
        mask=mask,
        other=-float("inf"),
    )
    indices = tl.load(
        partial_indices + row * num_vocab_blocks + offsets,
        mask=mask,
        other=2147483647,
    )
    best = tl.max(values, axis=0)
    best_indices = tl.where(values == best, indices, 2147483647)
    token = tl.min(best_indices, axis=0)
    tl.store(token_ids + row, token)


def _next_power_of_2_bounded(value: int, *, minimum: int = 1, maximum: int = 8192) -> int:
    rounded = int(jt.next_power_of_2(max(minimum, int(value))))
    if rounded > maximum:
        raise ValueError(
            f"Triton LM-head top1 reduction block {rounded} exceeds supported maximum {maximum}"
        )
    return rounded


def lm_head_greedy_top1_triton(
    hidden_norm: jax.Array,
    output_weight: jax.Array,
    *,
    block_m: int = 8,
    block_n: int = 256,
    block_k: int = 64,
) -> jax.Array:
    """Return greedy token ids for `[B, 1, H] x [H, V]`.

    The implementation computes full-precision tile accumulators and never
    materializes `[B, V]` logits.  It returns `[B, 1]` int32 token ids.
    """

    if hidden_norm.ndim != 3 or int(hidden_norm.shape[1]) != 1:
        raise ValueError("Triton LM-head top1 requires hidden shape [B, 1, H]")
    if output_weight.ndim != 2:
        raise ValueError("Triton LM-head top1 requires output weight shape [H, V]")
    batch = int(hidden_norm.shape[0])
    hidden_dim = int(hidden_norm.shape[-1])
    weight_hidden = int(output_weight.shape[0])
    vocab_size = int(output_weight.shape[1])
    if batch <= 0 or hidden_dim <= 0 or vocab_size <= 0:
        raise ValueError("Triton LM-head top1 requires non-empty dimensions")
    if hidden_dim != weight_hidden:
        raise ValueError("hidden dimension must match output weight rows")
    if hidden_norm.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise ValueError(f"unsupported hidden dtype for Triton LM-head top1: {hidden_norm.dtype}")
    if output_weight.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise ValueError(f"unsupported weight dtype for Triton LM-head top1: {output_weight.dtype}")

    x = jnp.reshape(hidden_norm, (batch, hidden_dim))
    reduce_cast = 0
    if hidden_norm.dtype == jnp.bfloat16 and output_weight.dtype == jnp.bfloat16:
        reduce_cast = 1
    elif hidden_norm.dtype == jnp.float16 and output_weight.dtype == jnp.float16:
        reduce_cast = 2
    num_vocab_blocks = int(jt.cdiv(vocab_size, block_n))
    partial_shape = (batch, num_vocab_blocks)
    partial_values, partial_indices = jt.triton_call(
        x,
        output_weight,
        kernel=_lm_head_top1_stage1_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(partial_shape, jnp.float32),
            jax.ShapeDtypeStruct(partial_shape, jnp.int32),
        ),
        grid=(jt.cdiv(batch, block_m), num_vocab_blocks),
        name="lm_head_top1_tensorcore_stage1",
        batch_size=batch,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_vocab_blocks=num_vocab_blocks,
        REDUCE_CAST=reduce_cast,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        num_warps=4,
        num_stages=3,
    )
    token_ids = jt.triton_call(
        partial_values,
        partial_indices,
        kernel=_lm_head_top1_stage2_kernel,
        out_shape=jax.ShapeDtypeStruct((batch,), jnp.int32),
        grid=(batch,),
        name="lm_head_top1_tensorcore_stage2",
        num_vocab_blocks=num_vocab_blocks,
        BLOCK_B=_next_power_of_2_bounded(num_vocab_blocks),
        num_warps=8 if num_vocab_blocks >= 1024 else 4,
        num_stages=3,
    )
    return token_ids[:, None]


__all__ = ["lm_head_greedy_top1_triton"]
