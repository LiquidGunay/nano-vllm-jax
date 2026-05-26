"""Focused CUDA tests for local FP32 JAX FFI kernels."""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.kernels.cuda_fp32_ffi import (
    build_cuda_fp32_kernels,
    kv_append_paged_nhd_fp32,
)
from nanovllm_jax.kernels.flashinfer_ffi import kv_append_paged_nhd_reference


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _has_nvcc() -> bool:
    configured = os.getenv("NANO_VLLM_JAX_NVCC")
    if configured and os.path.exists(configured):
        return True
    purelib = importlib.util.find_spec("jax")
    if purelib is not None:
        site_packages = Path(purelib.origin).resolve().parents[1]
        nvcc = site_packages / "nvidia" / "cu13" / "bin" / "nvcc"
        if nvcc.exists():
            return True
    return shutil.which("nvcc") is not None


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
def test_build_cuda_fp32_kernels_under_mountpoint(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")

    shared_object = build_cuda_fp32_kernels()

    assert str(shared_object).startswith("/mountpoint/.exp/")
    assert shared_object.exists()


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_kv_append_paged_nhd_fp32_cuda_matches_reference(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    head_dim = 256
    append_key = jnp.arange(4 * 2 * head_dim, dtype=jnp.float32).reshape(
        4, 2, head_dim
    )
    append_value = append_key + 1000.0
    batch_indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    positions = jnp.array([0, 0, 3, 2], dtype=jnp.int32)
    k_cache = jnp.full((4, 4, 2, head_dim), -1.0, dtype=jnp.float32)
    v_cache = jnp.full((4, 4, 2, head_dim), -2.0, dtype=jnp.float32)
    kv_indices = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    kv_indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    kv_last_page_len = jnp.array([4, 3], dtype=jnp.int32)

    expected_k, expected_v = kv_append_paged_nhd_reference(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )
    actual_k, actual_v = jax.jit(kv_append_paged_nhd_fp32)(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )

    np.testing.assert_array_equal(np.asarray(actual_k), np.asarray(expected_k))
    np.testing.assert_array_equal(np.asarray(actual_v), np.asarray(expected_v))
