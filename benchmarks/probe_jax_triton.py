"""Smoke-test the optional jax-triton bridge on the active GPU.

This is a dependency/runtime probe, not a serving benchmark. It deliberately
keeps caches under ``NANO_VLLM_JAX_CACHE_ROOT`` or ``/mountpoint/.exp`` via the
shared runtime path helpers.
"""

from __future__ import annotations

import os

from runtime_paths import (
    configure_compilation_cache,
    configure_flashinfer_cache,
    configure_xla_flags,
    default_runtime_root,
)

root = default_runtime_root()
os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / ".cache" / "triton"))
os.environ.setdefault("JAX_PLATFORMS", "cuda")
configure_compilation_cache()
configure_flashinfer_cache()
configure_xla_flags()

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _add(x: jax.Array, y: jax.Array) -> jax.Array:
    block_size = 128
    return jt.triton_call(
        x,
        y,
        kernel=_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(jt.cdiv(x.size, block_size),),
        n=x.size,
        block_size=block_size,
        num_warps=4,
    )


def main() -> None:
    x = jnp.arange(1024, dtype=jnp.float32)
    y = jnp.arange(1024, dtype=jnp.float32) * 2
    z = jax.jit(_add)(x, y)
    expected = x + y
    ok = bool(jnp.allclose(z, expected))
    print(
        {
            "devices": [str(device) for device in jax.devices()],
            "jax": jax.__version__,
            "jax_triton": getattr(jt, "__version__", None),
            "triton": getattr(triton, "__version__", None),
            "z17": float(z[17]),
            "z_last": float(z[-1]),
            "ok": ok,
        }
    )
    if not ok:
        raise SystemExit("jax-triton add smoke failed")


if __name__ == "__main__":
    main()
