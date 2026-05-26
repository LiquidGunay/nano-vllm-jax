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

jax.config.update("jax_default_matmul_precision", "highest")

from nanovllm_jax.backends import PureJAXBackend
from nanovllm_jax.kernels.cuda_fp32_ffi import (
    build_cuda_fp32_kernels,
    kv_append_paged_nhd_fp32,
    paged_decode_attention_gqa_nhd_fp32,
)
from nanovllm_jax.kernels.flashinfer_ffi import kv_append_paged_nhd_reference
from nanovllm_jax.kernels.paged_attention import (
    dense_block_tables_to_kv_indptr,
    kv_last_page_len_from_seq_lens,
    paged_decode_attention_gqa_nhd_reference,
)
from nanovllm_jax.kv_cache import (
    AttentionMetadata,
    KVCacheStorage,
    compute_slot_mapping,
    update_kv_cache,
)


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


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_cuda_fp32_kv_append_opt_in_matches_canonical_update(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    monkeypatch.setenv("NANO_VLLM_JAX_CUDA_FP32_KV_APPEND", "1")

    page_size = 4
    head_dim = 256
    layer_id = 1
    k = jnp.arange(2 * 4 * 2 * head_dim, dtype=jnp.float32).reshape(
        2, 4, 2, head_dim
    )
    v = k + 1000.0
    k_cache = jnp.full((3, 6, page_size, 2, head_dim), -1.0, dtype=jnp.float32)
    v_cache = jnp.full((3, 6, page_size, 2, head_dim), -2.0, dtype=jnp.float32)
    positions = jnp.array(
        [
            [0, 1, 2, 0],
            [4, 5, 0, 0],
        ],
        dtype=jnp.int32,
    )
    block_tables = jnp.array(
        [
            [2, 0, 1],
            [3, 5, 4],
        ],
        dtype=jnp.int32,
    )
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    seq_lens = jnp.array([3, 6], dtype=jnp.int32)
    metadata = AttentionMetadata(
        slot_mapping=compute_slot_mapping(
            positions=positions,
            block_table=block_tables,
            block_size=page_size,
            is_prefill=True,
        ),
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        positions=positions,
    )
    query_lens = jnp.diff(query_start_loc).astype(jnp.int32)
    valid_mask = jnp.arange(positions.shape[1])[None, :] < query_lens[:, None]
    expected_k, expected_v = update_kv_cache(
        k_cache,
        v_cache,
        metadata.slot_mapping,
        k,
        v,
        layer_idx=layer_id,
        valid_mask=valid_mask,
    )

    actual = PureJAXBackend().write_kv(
        layer_id=layer_id,
        k=k,
        v=v,
        cache=KVCacheStorage(k_cache, v_cache),
        metadata=metadata,
    )

    np.testing.assert_array_equal(np.asarray(actual.k_cache), np.asarray(expected_k))
    np.testing.assert_array_equal(np.asarray(actual.v_cache), np.asarray(expected_v))


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.parametrize(
    ("page_size", "max_pages_per_sequence", "num_q_heads", "num_kv_heads", "head_dim"),
    [
        (4, 3, 4, 2, 8),
        (16, 2, 8, 2, 256),
    ],
)
def test_paged_decode_attention_gqa_nhd_fp32_cuda_matches_reference(
    monkeypatch,
    page_size,
    max_pages_per_sequence,
    num_q_heads,
    num_kv_heads,
    head_dim,
):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    batch = 2
    num_pages = max_pages_per_sequence * batch + 1
    seq_lens = jnp.array(
        [page_size + 1, max_pages_per_sequence * page_size - 1],
        dtype=jnp.int32,
    )
    block_tables = jnp.array(
        [
            [num_pages - 1, 1, 2][:max_pages_per_sequence],
            [0, 3, 4][:max_pages_per_sequence],
        ],
        dtype=jnp.int32,
    )
    k_cache = jnp.linspace(
        -0.3,
        0.4,
        num_pages * page_size * num_kv_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(num_pages, page_size, num_kv_heads, head_dim)
    v_cache = jnp.linspace(
        0.2,
        0.9,
        num_pages * page_size * num_kv_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(num_pages, page_size, num_kv_heads, head_dim)
    q = jnp.linspace(
        -0.5,
        0.5,
        batch * num_q_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_q_heads, head_dim)
    kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(block_tables)
    kv_last_page_len = kv_last_page_len_from_seq_lens(seq_lens, page_size)
    scale = float(1.0 / np.sqrt(head_dim))

    expected = paged_decode_attention_gqa_nhd_reference(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
        scale,
        max_pages_per_sequence=max_pages_per_sequence,
    )
    actual = jax.jit(
        lambda query, key_cache, value_cache, indptr, indices, last_page_len, lens: (
            paged_decode_attention_gqa_nhd_fp32(
                query,
                key_cache,
                value_cache,
                indptr,
                indices,
                last_page_len,
                lens,
                scale,
            )
        )
    )(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_cuda_fp32_decode_attention_opt_in_matches_pure_jax(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    monkeypatch.delenv("NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN", raising=False)

    page_size = 16
    max_pages_per_sequence = 2
    batch = 2
    num_pages = max_pages_per_sequence * batch + 1
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 256
    layer_id = 0
    seq_lens = jnp.array([page_size + 1, max_pages_per_sequence * page_size - 1], dtype=jnp.int32)
    positions = (seq_lens - 1)[:, None]
    block_tables = jnp.array(
        [
            [num_pages - 1, 1],
            [0, 3],
        ],
        dtype=jnp.int32,
    )
    k_cache = jnp.linspace(
        -0.3,
        0.4,
        num_pages * page_size * num_kv_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(1, num_pages, page_size, num_kv_heads, head_dim)
    v_cache = jnp.linspace(
        0.2,
        0.9,
        num_pages * page_size * num_kv_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(1, num_pages, page_size, num_kv_heads, head_dim)
    query = jnp.linspace(
        -0.5,
        0.5,
        batch * num_q_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(batch, 1, num_q_heads, head_dim)
    metadata = AttentionMetadata(
        slot_mapping=compute_slot_mapping(
            positions=positions,
            block_table=block_tables,
            block_size=page_size,
            is_prefill=False,
        ),
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=jnp.arange(batch + 1, dtype=jnp.int32),
        num_prefill_tokens=0,
        num_decode_tokens=batch,
        positions=positions,
        max_kv_len=max_pages_per_sequence * page_size,
    )

    expected = PureJAXBackend().attention(
        layer_id=layer_id,
        query=query,
        cache=KVCacheStorage(k_cache, v_cache),
        metadata=metadata,
        block_size=page_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_q_heads // num_kv_heads,
        is_prefill=False,
    )
    monkeypatch.setenv("NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN", "1")
    actual = PureJAXBackend().attention(
        layer_id=layer_id,
        query=query,
        cache=KVCacheStorage(k_cache, v_cache),
        metadata=metadata,
        block_size=page_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_q_heads // num_kv_heads,
        is_prefill=False,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=2e-5,
        atol=2e-5,
    )
