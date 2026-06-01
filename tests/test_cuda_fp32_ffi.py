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
    gdn_packed_decode_step_fp32,
    gdn_packed_decode_step_fp32_reference,
    gdn_prefill_chunk32_prepared_fp32,
    gdn_prefill_chunk32_normalized_fp32,
    gdn_prefill_chunk32_v64_normalized_fp32,
    gdn_recurrent_decode_step_fp32,
    gdn_recurrent_decode_step_fp32_reference,
    kv_append_paged_nhd_fp32,
    paged_decode_attention_gqa_nhd_fp32,
)
from nanovllm_jax.kernels.gdn_fla import gdn_fla_prefill_chunk32_fp32_reference
from nanovllm_jax.layers import l2norm
from nanovllm_jax.model import jax_chunk_gated_delta_rule
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


@pytest.fixture(autouse=True)
def _allow_local_cuda_probe_backend_routes(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES", "1")


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


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.parametrize(
    ("batch", "num_heads", "key_dim", "value_dim"),
    [
        (2, 2, 8, 12),
        (2, 16, 128, 128),
    ],
)
def test_gdn_recurrent_decode_step_fp32_cuda_matches_reference(
    monkeypatch,
    batch,
    num_heads,
    key_dim,
    value_dim,
):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    query = jnp.linspace(
        -0.5,
        0.5,
        batch * num_heads * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, key_dim)
    key = jnp.linspace(
        0.4,
        -0.4,
        batch * num_heads * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, key_dim)
    value = jnp.linspace(
        -0.2,
        0.3,
        batch * num_heads * value_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, value_dim)
    g = jnp.linspace(-0.08, -0.02, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
        1,
    )
    beta = jnp.linspace(0.2, 0.8, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
        1,
    )
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, value_dim, key_dim)

    expected_out, expected_state = gdn_recurrent_decode_step_fp32_reference(
        query,
        key,
        value,
        g,
        beta,
        state,
    )
    actual_out, actual_state = jax.jit(gdn_recurrent_decode_step_fp32)(
        query,
        key,
        value,
        g,
        beta,
        state,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.parametrize(
    ("prefill_fn", "value_dim"),
    [
        (gdn_prefill_chunk32_normalized_fp32, 32),
        (gdn_prefill_chunk32_v64_normalized_fp32, 64),
    ],
)
def test_gdn_prefill_chunk32_normalized_fp32_cuda_matches_chunk_reference(
    monkeypatch,
    prefill_fn,
    value_dim,
):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    batch = 2
    num_heads = 2
    seq_len = 64
    key_dim = 32
    lengths = jnp.array([37, 64], dtype=jnp.int32)
    valid = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]

    query = jnp.linspace(
        -0.5,
        0.5,
        batch * num_heads * seq_len * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, seq_len, key_dim)
    key = jnp.linspace(
        0.4,
        -0.4,
        batch * num_heads * seq_len * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, seq_len, key_dim)
    value = jnp.linspace(
        -0.2,
        0.3,
        batch * num_heads * seq_len * value_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, seq_len, value_dim)
    g = jnp.linspace(-0.08, -0.02, batch * num_heads * seq_len, dtype=jnp.float32).reshape(
        batch,
        num_heads,
        seq_len,
    )
    beta = jnp.linspace(0.2, 0.8, batch * num_heads * seq_len, dtype=jnp.float32).reshape(
        batch,
        num_heads,
        seq_len,
    )
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_heads * key_dim * value_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, value_dim, key_dim)

    query = jnp.where(valid[:, None, :, None], query, 0.0)
    key = jnp.where(valid[:, None, :, None], key, 0.0)
    value = jnp.where(valid[:, None, :, None], value, 0.0)
    g = jnp.where(valid[:, None, :], g, 0.0)
    beta = jnp.where(valid[:, None, :], beta, 0.0)

    expected_out, expected_state = jax_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=32,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    query_norm_scaled = l2norm(query, axis=-1, eps=1e-6) * (1.0 / jnp.sqrt(key_dim))
    key_norm = l2norm(key, axis=-1, eps=1e-6)
    actual_out, actual_state = jax.jit(prefill_fn)(
        query_norm_scaled,
        key_norm,
        value,
        g,
        beta,
        lengths,
        state,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_gdn_prefill_chunk32_prepared_fp32_cuda_matches_fla_reference(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    batch = 2
    seq_len = 64
    num_heads = 2
    key_dim = 32
    value_dim = 32
    lengths = jnp.array([37, 64], dtype=jnp.int32)

    query = jnp.linspace(
        -0.5,
        0.5,
        batch * seq_len * num_heads * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_heads, key_dim)
    key = jnp.linspace(
        0.4,
        -0.4,
        batch * seq_len * num_heads * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_heads, key_dim)
    value = jnp.linspace(
        -0.2,
        0.3,
        batch * seq_len * num_heads * value_dim,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_heads, value_dim)
    gate = jnp.linspace(
        -0.08,
        -0.02,
        batch * seq_len * num_heads,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_heads)
    beta = jnp.linspace(
        0.2,
        0.8,
        batch * seq_len * num_heads,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_heads)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, value_dim, key_dim)

    query_norm = l2norm(query, axis=-1, eps=1e-6)
    key_norm = l2norm(key, axis=-1, eps=1e-6)
    expected_out, expected_state = gdn_fla_prefill_chunk32_fp32_reference(
        query_norm,
        key_norm,
        value,
        gate,
        beta,
        lengths,
        state,
    )
    actual_out, actual_state = jax.jit(gdn_prefill_chunk32_prepared_fp32)(
        query_norm,
        key_norm,
        value,
        gate,
        beta,
        lengths,
        state,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_cuda_fp32_gdn_decode_opt_in_matches_pure_jax(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    monkeypatch.delenv("NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE", raising=False)

    batch = 2
    num_heads = 16
    head_dim = 128
    query = jnp.linspace(
        -0.5,
        0.5,
        batch * num_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, head_dim)
    key = jnp.linspace(
        0.4,
        -0.4,
        batch * num_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, head_dim)
    value = jnp.linspace(
        -0.2,
        0.3,
        batch * num_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, head_dim)
    g = jnp.linspace(-0.08, -0.02, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
        1,
    )
    beta = jnp.linspace(0.2, 0.8, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
        1,
    )
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_heads * head_dim * head_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, head_dim, head_dim)

    expected_out, expected_state = PureJAXBackend().gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )
    monkeypatch.setenv("NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE", "1")
    actual_out, actual_state = PureJAXBackend().gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.parametrize(
    ("num_q_heads", "num_value_heads", "key_dim", "value_dim"),
    [
        (4, 4, 16, 12),
        (2, 4, 16, 16),
    ],
)
def test_gdn_packed_decode_step_fp32_matches_reference(
    monkeypatch,
    num_q_heads,
    num_value_heads,
    key_dim,
    value_dim,
):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")

    batch = 2
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = jnp.linspace(
        -0.5,
        0.5,
        batch * packed_dim,
        dtype=jnp.float32,
    ).reshape(batch, packed_dim)
    a = jnp.linspace(-0.3, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    a_log = jnp.linspace(-0.2, 0.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(0.05, 0.2, num_value_heads, dtype=jnp.float32)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_value_heads * key_dim * value_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads, value_dim, key_dim)

    expected_out, expected_state = gdn_packed_decode_step_fp32_reference(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
    )
    actual_out, actual_state = jax.jit(gdn_packed_decode_step_fp32)(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=3e-5,
        atol=3e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_gdn_packed_decode_reference_and_cuda_match(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")

    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = jnp.linspace(
        -0.5,
        0.5,
        batch * packed_dim,
        dtype=jnp.float32,
    ).reshape(batch, packed_dim)
    a = jnp.linspace(-0.3, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    a_log = jnp.linspace(-0.2, 0.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(0.05, 0.2, num_value_heads, dtype=jnp.float32)
    decay = jnp.exp(a_log)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_value_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads, value_dim, key_dim)

    expected_out, expected_state = gdn_packed_decode_step_fp32_reference(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "reference")
    ref_out, ref_state = PureJAXBackend().gated_delta_packed_decode(
        mixed_qkv,
        a,
        b,
        decay,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "cuda_fp32")
    actual_out, actual_state = PureJAXBackend().gated_delta_packed_decode(
        mixed_qkv,
        a,
        b,
        decay,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )

    for out, recurrent_state in ((ref_out, ref_state), (actual_out, actual_state)):
        np.testing.assert_allclose(
            np.asarray(out),
            np.asarray(expected_out),
            rtol=3e-5,
            atol=3e-5,
        )
        np.testing.assert_allclose(
            np.asarray(recurrent_state),
            np.asarray(expected_state),
            rtol=3e-5,
            atol=3e-5,
        )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_gdn_packed_decode_bf16_cuda_matches_reference(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")

    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = jnp.linspace(
        -0.5,
        0.5,
        batch * packed_dim,
        dtype=jnp.float32,
    ).reshape(batch, packed_dim)
    a = jnp.linspace(-0.3, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    a_log = jnp.linspace(-0.2, 0.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(0.05, 0.2, num_value_heads, dtype=jnp.float32)
    decay = jnp.exp(a_log)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_value_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads, value_dim, key_dim)

    expected_out, expected_state = gdn_packed_decode_step_fp32_reference(
        mixed_qkv.astype(jnp.bfloat16),
        a,
        b,
        a_log,
        dt_bias,
        state,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "reference")
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE", "bf16")
    ref_out, ref_state = PureJAXBackend().gated_delta_packed_decode(
        mixed_qkv,
        a,
        b,
        decay,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )
    assert ref_out.dtype == jnp.float32
    assert ref_state.dtype == jnp.float32

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "cuda_fp32")
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE", "bf16")
    actual_out, actual_state = PureJAXBackend().gated_delta_packed_decode(
        mixed_qkv,
        a,
        b,
        decay,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )

    np.testing.assert_allclose(
        np.asarray(ref_out),
        np.asarray(expected_out),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(ref_state),
        np.asarray(expected_state),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=3e-5,
        atol=3e-5,
    )


@pytest.mark.skipif(not _has_nvcc(), reason="nvcc is required for local CUDA FFI")
@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_gdn_packed_decode_bf16_cuda_stays_exact_over_recurrent_steps(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")

    # Model-shaped packed decode geometry similar to real decode widths.
    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 32
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    num_steps = 500

    state = jnp.zeros(
        (batch, num_value_heads, value_dim, key_dim),
        dtype=jnp.float32,
    )
    decay = jnp.logspace(-0.35, -0.10, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.12, 0.12, num_value_heads, dtype=jnp.float32)

    mixed_qkv_rng = np.random.default_rng(12345)
    step_control_rng = np.random.default_rng(54321)
    mixed_qkv_steps = []
    a_steps = []
    b_steps = []
    for step in range(num_steps):
        mixed_qkv_steps.append(
            jnp.asarray(
                mixed_qkv_rng.standard_normal((batch, packed_dim), dtype=np.float32),
                dtype=jnp.float32,
            )
        )
        a_steps.append(
            jnp.asarray(
                0.6 * np.tanh(
                    np.sin(step_control_rng.standard_normal((batch, num_value_heads), dtype=np.float32))
                ),
                dtype=jnp.float32,
            )
        )
        b_steps.append(
            jnp.asarray(
                np.tanh(
                    step_control_rng.standard_normal((batch, num_value_heads), dtype=np.float32)
                ),
                dtype=jnp.float32,
            )
        )

    ref_state = state
    cuda_state = state
    ref_env = {
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL": "reference",
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE": "bf16",
    }
    cuda_env = {
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL": "cuda_fp32",
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE": "bf16",
    }

    def _run_with_env(env: dict[str, str], mixed_qkv_step, a_step, b_step, current_state):
        with monkeypatch.context() as _patch:
            for key, value in env.items():
                _patch.setenv(key, value)
            return PureJAXBackend().gated_delta_packed_decode(
                mixed_qkv_step,
                a_step,
                b_step,
                decay,
                dt_bias,
                current_state,
                use_qk_l2norm_in_kernel=True,
            )

    for step, (mixed_qkv, a_step, b_step) in enumerate(
        zip(mixed_qkv_steps, a_steps, b_steps)
    ):
        ref_out, ref_state = _run_with_env(
            ref_env,
            mixed_qkv,
            jnp.asarray(a_step),
            jnp.asarray(b_step),
            ref_state,
        )
        cuda_out, cuda_state = _run_with_env(
            cuda_env,
            mixed_qkv,
            jnp.asarray(a_step),
            jnp.asarray(b_step),
            cuda_state,
        )
        np.testing.assert_allclose(
            np.asarray(cuda_out),
            np.asarray(ref_out),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"out mismatch at step={step}",
        )
        np.testing.assert_allclose(
            np.asarray(cuda_state),
            np.asarray(ref_state),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"state mismatch at step={step}",
        )
