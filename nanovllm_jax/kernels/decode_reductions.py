"""Strict opt-in Pallas kernels for decode-time reductions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _pallas_modules():
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "Pallas decode-reduction fastpath was requested, but Pallas/Triton "
            "is unavailable. Disable the fastpath or install the Pallas GPU stack."
        ) from exc
    return pl, plt


def _compiler_params(hidden_dim: int):
    _pl, plt = _pallas_modules()
    if hidden_dim >= 1024:
        num_warps = 8
    elif hidden_dim >= 256:
        num_warps = 4
    else:
        num_warps = 1
    return plt.CompilerParams(num_warps=num_warps)


def _triton_modules():
    try:
        import jax_triton as jt
        import triton
        import triton.language as tl
        from nanovllm_jax.kernels.gdn_fla_triton import _configure_triton_runtime
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "Triton decode-reduction fastpath was requested, but jax-triton/Triton "
            "is unavailable. Disable the fastpath or install the Triton GPU stack."
        ) from exc
    _configure_triton_runtime()
    return jt, triton, tl


def _triton_rms_kernel():
    _jt, triton, tl = _triton_modules()

    @triton.jit
    def _kernel(
        x,
        weight,
        out,
        total_rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        weight_rows: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_H)
        mask = offsets < hidden_dim
        values = tl.load(x + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(values * values, axis=0) / hidden_dim
        scale = tl.rsqrt(mean_sq + 1.0e-6)
        weight_row = row % weight_rows
        weights = tl.load(
            weight + weight_row * hidden_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output = values * scale * (1.0 + weights)
        tl.store(out + row * hidden_dim + offsets, output, mask=mask)

    return _kernel


def _triton_rms_num_warps(hidden_dim: int) -> int:
    if hidden_dim >= 1024:
        return 8
    if hidden_dim >= 256:
        return 4
    return 1


def _triton_rms_padded_gemm_kernel():
    _jt, triton, tl = _triton_modules()

    @triton.jit
    def _kernel(
        x,
        norm_weight,
        mat_weight,
        out,
        batch_size: tl.constexpr,
        hidden_dim: tl.constexpr,
        out_dim: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        row_mask = rows < batch_size

        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, hidden_dim, BLOCK_K):
            k = k_start + k_offsets
            values = tl.load(
                x + rows[:, None] * hidden_dim + k[None, :],
                mask=row_mask[:, None] & (k[None, :] < hidden_dim),
                other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(values * values, axis=1)
        scale = tl.rsqrt(sum_sq / hidden_dim + EPS)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, hidden_dim, BLOCK_K):
            k = k_start + k_offsets
            values = tl.load(
                x + rows[:, None] * hidden_dim + k[None, :],
                mask=row_mask[:, None] & (k[None, :] < hidden_dim),
                other=0.0,
            ).to(tl.float32)
            weights = tl.load(norm_weight + k, mask=k < hidden_dim, other=0.0).to(tl.float32)
            a = values * scale[:, None] * (1.0 + weights[None, :])
            a = a.to(tl.bfloat16)
            b = tl.load(
                mat_weight + k[:, None] * out_dim + cols[None, :],
                mask=(k[:, None] < hidden_dim) & (cols[None, :] < out_dim),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=tl.float32)

        tl.store(
            out + rows[:, None] * out_dim + cols[None, :],
            acc,
            mask=row_mask[:, None] & (cols[None, :] < out_dim),
        )

    return _kernel


def triton_decode_rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Apply Qwen RMSNorm over the last dimension using a Triton custom call."""
    if eps != 1e-6:
        raise ValueError("Triton decode RMSNorm currently supports eps=1e-6")
    if x.ndim < 2:
        raise ValueError("Triton RMSNorm requires at least two dimensions")
    hidden_dim = int(x.shape[-1])
    if hidden_dim <= 0:
        raise ValueError("Triton RMSNorm requires a non-empty hidden dimension")
    if int(weight.shape[-1]) != hidden_dim:
        raise ValueError("Triton RMSNorm weight dimension must match input hidden dimension")
    if weight.ndim not in {1, 2}:
        raise ValueError("Triton RMSNorm supports 1D weights or per-head 2D weights")
    weight_rows = int(weight.shape[0]) if weight.ndim == 2 else 1
    if weight.ndim == 2 and (x.ndim < 3 or int(x.shape[-2]) != weight_rows):
        raise ValueError("Triton RMSNorm per-head weight rows must match input head dimension")

    jt, _triton, _tl = _triton_modules()
    leading = 1
    for dim in x.shape[:-1]:
        leading *= int(dim)
    x_2d = jnp.reshape(x, (leading, hidden_dim))
    weight_2d = jnp.reshape(weight, (weight_rows, hidden_dim))
    block_h = jt.next_power_of_2(hidden_dim)
    out = jt.triton_call(
        x_2d,
        weight_2d,
        kernel=_triton_rms_kernel(),
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, x.dtype),
        grid=(leading,),
        name="decode_rms_norm_triton",
        total_rows=leading,
        hidden_dim=hidden_dim,
        weight_rows=weight_rows,
        BLOCK_H=block_h,
        num_warps=_triton_rms_num_warps(hidden_dim),
        num_stages=3,
    )
    return jnp.reshape(out, x.shape)


def triton_decode_rms_padded_gemm(
    x: jnp.ndarray,
    norm_weight: jnp.ndarray,
    mat_weight: jnp.ndarray,
    *,
    eps: float = 1e-6,
    rows: int = 8,
    block_n: int = 128,
    block_k: int = 64,
) -> jnp.ndarray:
    """Fuse decode RMSNorm and a BF16 row-padded projection.

    This implements ``rms_norm(x, norm_weight).astype(bfloat16) @ mat_weight``
    for width-1 decode tensors without materializing the normalized activation.
    """

    if eps != 1e-6:
        raise ValueError("Triton decode RMS+padded GEMM currently supports eps=1e-6")
    if x.ndim != 3 or int(x.shape[1]) != 1:
        raise ValueError("Triton decode RMS+padded GEMM requires x shape [B, 1, H]")
    if norm_weight.ndim != 1:
        raise ValueError("Triton decode RMS+padded GEMM requires 1D norm weights")
    if mat_weight.ndim != 2:
        raise ValueError("Triton decode RMS+padded GEMM requires mat_weight shape [H, O]")
    batch = int(x.shape[0])
    hidden_dim = int(x.shape[-1])
    out_dim = int(mat_weight.shape[1])
    if batch <= 0 or hidden_dim <= 0 or out_dim <= 0:
        raise ValueError("Triton decode RMS+padded GEMM requires non-empty dimensions")
    if batch > rows:
        raise ValueError("batch exceeds configured padded GEMM rows")
    if int(norm_weight.shape[0]) != hidden_dim or int(mat_weight.shape[0]) != hidden_dim:
        raise ValueError("hidden dimensions must match for Triton decode RMS+padded GEMM")
    if mat_weight.dtype != jnp.bfloat16:
        raise ValueError("Triton decode RMS+padded GEMM currently requires BF16 mat weights")
    if x.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError(f"unsupported x dtype for Triton decode RMS+padded GEMM: {x.dtype}")
    if norm_weight.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError(
            f"unsupported norm weight dtype for Triton decode RMS+padded GEMM: {norm_weight.dtype}"
        )

    jt, _triton, _tl = _triton_modules()
    x_2d = jnp.reshape(x, (batch, hidden_dim))
    out = jt.triton_call(
        x_2d,
        norm_weight,
        mat_weight,
        kernel=_triton_rms_padded_gemm_kernel(),
        out_shape=jax.ShapeDtypeStruct((batch, out_dim), jnp.bfloat16),
        grid=(jt.cdiv(out_dim, block_n),),
        name="decode_rms_padded_gemm_triton",
        batch_size=batch,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        EPS=float(eps),
        BLOCK_M=int(rows),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        num_warps=4,
        num_stages=3,
    )
    return jnp.reshape(out, (batch, 1, out_dim))


def decode_rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Deprecated dispatcher retained for compatibility with older call sites."""
    raise RuntimeError("decode_rms_norm has no implicit lowered implementation")


def lowered_decode_rms_norm_enabled() -> bool:
    return False


def pallas_decode_rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Apply Qwen RMSNorm over the last dimension using one Pallas program per row."""
    if x.ndim < 2:
        raise ValueError("Pallas RMSNorm requires at least two dimensions")
    hidden_dim = int(x.shape[-1])
    if hidden_dim <= 0:
        raise ValueError("Pallas RMSNorm requires a non-empty hidden dimension")
    if int(weight.shape[-1]) != hidden_dim:
        raise ValueError("Pallas RMSNorm weight dimension must match input hidden dimension")
    weight_ndim = int(weight.ndim)
    if weight_ndim not in {1, 2}:
        raise ValueError("Pallas RMSNorm supports 1D weights or per-head 2D weights")
    weight_rows = int(weight.shape[0]) if weight_ndim == 2 else 1
    if weight_ndim == 2 and (x.ndim < 3 or int(x.shape[-2]) != weight_rows):
        raise ValueError("Pallas RMSNorm per-head weight rows must match input head dimension")

    pl, _plt = _pallas_modules()
    leading = 1
    for dim in x.shape[:-1]:
        leading *= int(dim)
    x_2d = jnp.reshape(x, (leading, hidden_dim))
    out_dtype = x.dtype

    def kernel(x_ref, weight_ref, out_ref):
        row = pl.program_id(0)
        offsets = jnp.arange(hidden_dim)
        values = x_ref[row, offsets].astype(jnp.float32)
        mean_sq = jnp.sum(values * values, axis=0) / hidden_dim
        scale = jax.lax.rsqrt(mean_sq + eps)
        if weight_ndim == 1:
            weights = weight_ref[offsets].astype(jnp.float32)
        else:
            head = row % weight_rows
            weights = weight_ref[head, offsets].astype(jnp.float32)
        out_ref[row, offsets] = (values * scale * (1.0 + weights)).astype(out_dtype)

    out_2d = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, out_dtype),
        grid=(leading,),
        name="decode_rms_norm_pallas",
        compiler_params=_compiler_params(hidden_dim),
    )(x_2d, weight)
    return jnp.reshape(out_2d, x.shape)


def pallas_l2_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Apply L2 normalization over the last dimension using Pallas."""
    if x.ndim < 2:
        raise ValueError("Pallas L2 norm requires at least two dimensions")
    hidden_dim = int(x.shape[-1])
    if hidden_dim <= 0:
        raise ValueError("Pallas L2 norm requires a non-empty hidden dimension")

    pl, _plt = _pallas_modules()
    leading = 1
    for dim in x.shape[:-1]:
        leading *= int(dim)
    x_2d = jnp.reshape(x, (leading, hidden_dim))
    out_dtype = x.dtype

    def kernel(x_ref, out_ref):
        row = pl.program_id(0)
        offsets = jnp.arange(hidden_dim)
        values = x_ref[row, offsets].astype(jnp.float32)
        scale = jax.lax.rsqrt(jnp.sum(values * values, axis=0) + eps)
        out_ref[row, offsets] = (values * scale).astype(out_dtype)

    out_2d = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, out_dtype),
        grid=(leading,),
        name="decode_l2_norm_pallas",
        compiler_params=_compiler_params(hidden_dim),
    )(x_2d)
    return jnp.reshape(out_2d, x.shape)


def gdn_packed_decode_pre_normalize_qk_pallas(
    mixed_qkv: jnp.ndarray,
    state: jnp.ndarray,
    *,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Fuse packed GDN decode Q/K L2-normalization and value copy in Pallas."""
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [batch, packed_dim]")
    if state.ndim != 4:
        raise ValueError("state must have shape [batch, value_heads, value_dim, key_dim]")
    if mixed_qkv.shape[0] != state.shape[0]:
        raise ValueError("mixed_qkv batch must match state batch")

    batch = int(mixed_qkv.shape[0])
    num_value_heads = int(state.shape[1])
    value_dim = int(state.shape[2])
    key_dim = int(state.shape[3])
    packed_dim = int(mixed_qkv.shape[1])
    qk_dim = packed_dim - num_value_heads * value_dim
    if qk_dim <= 0 or qk_dim % (2 * key_dim) != 0:
        raise ValueError("mixed_qkv has an invalid packed Q/K dimension")
    num_q_heads = qk_dim // (2 * key_dim)
    q_dim = num_q_heads * key_dim
    output_dtype = jnp.dtype(jnp.float32)

    pl, _plt = _pallas_modules()

    def qk_kernel(mixed_ref, qk_out_ref):
        row = pl.program_id(0)
        head = pl.program_id(1)
        offsets = jnp.arange(key_dim)
        q_base = head * key_dim
        k_base = q_dim + head * key_dim
        query = mixed_ref[row, q_base + offsets].astype(jnp.float32)
        key = mixed_ref[row, k_base + offsets].astype(jnp.float32)
        query = query * jax.lax.rsqrt(jnp.sum(query * query, axis=0) + eps)
        key = key * jax.lax.rsqrt(jnp.sum(key * key, axis=0) + eps)
        qk_out_ref[row, q_base + offsets] = query.astype(output_dtype)
        qk_out_ref[row, k_base + offsets] = key.astype(output_dtype)

    if value_dim != key_dim:
        qk = pl.pallas_call(
            qk_kernel,
            out_shape=jax.ShapeDtypeStruct((batch, 2 * q_dim), output_dtype),
            grid=(batch, num_q_heads),
            name="gdn_packed_decode_qk_prenorm_pallas",
            compiler_params=_compiler_params(key_dim),
        )(mixed_qkv)
        value = mixed_qkv[:, 2 * q_dim :].astype(output_dtype)
        return jnp.concatenate([qk, value], axis=-1)

    def kernel(mixed_ref, out_ref):
        row = pl.program_id(0)
        block = pl.program_id(1)
        offsets = jnp.arange(key_dim)

        def normalize_qk(_):
            q_base = block * key_dim
            k_base = q_dim + block * key_dim
            query = mixed_ref[row, q_base + offsets].astype(jnp.float32)
            key = mixed_ref[row, k_base + offsets].astype(jnp.float32)
            query = query * jax.lax.rsqrt(jnp.sum(query * query, axis=0) + eps)
            key = key * jax.lax.rsqrt(jnp.sum(key * key, axis=0) + eps)
            out_ref[row, q_base + offsets] = query.astype(output_dtype)
            out_ref[row, k_base + offsets] = key.astype(output_dtype)

        def copy_value(_):
            value_head = block - num_q_heads
            value_base = 2 * q_dim + value_head * value_dim
            out_ref[row, value_base + offsets] = mixed_ref[
                row,
                value_base + offsets,
            ].astype(output_dtype)

        jax.lax.cond(block < num_q_heads, normalize_qk, copy_value, operand=None)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(mixed_qkv.shape, output_dtype),
        grid=(batch, num_q_heads + num_value_heads),
        name="gdn_packed_decode_qk_prenorm_pallas",
        compiler_params=_compiler_params(key_dim),
    )(mixed_qkv)
