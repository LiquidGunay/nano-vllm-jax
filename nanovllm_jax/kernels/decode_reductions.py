"""Strict opt-in Pallas kernels for decode-time reductions."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}


def pallas_decode_rms_norm_enabled() -> bool:
    return os.environ.get("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", "0") in _TRUE_ENV_VALUES


def triton_decode_rms_norm_enabled() -> bool:
    return os.environ.get("NANO_VLLM_JAX_TRITON_DECODE_RMSNORM", "0") in _TRUE_ENV_VALUES


def pallas_gdn_qk_prenorm_enabled() -> bool:
    return os.environ.get("NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM", "0") in _TRUE_ENV_VALUES


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


def decode_rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Dispatch decode RMSNorm through the selected strict opt-in implementation."""
    if triton_decode_rms_norm_enabled():
        return triton_decode_rms_norm(x, weight, eps)
    if pallas_decode_rms_norm_enabled():
        return pallas_decode_rms_norm(x, weight, eps)
    raise RuntimeError("decode_rms_norm called without an enabled lowered implementation")


def lowered_decode_rms_norm_enabled() -> bool:
    return triton_decode_rms_norm_enabled() or pallas_decode_rms_norm_enabled()


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
