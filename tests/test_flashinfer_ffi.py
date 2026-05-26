"""Focused CUDA tests for optional FlashInfer/JAX FFI kernels."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.backends import PureJAXBackend
from nanovllm_jax.kernels.flashinfer_ffi import (
    kv_append_paged_nhd,
    kv_append_paged_nhd_reference,
)
from nanovllm_jax.kv_cache import (
    AttentionMetadata,
    KVCacheStorage,
    compute_slot_mapping,
    update_kv_cache,
)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


@pytest.mark.skipif(
    not (_has_module("flashinfer") and _has_module("jax_tvm_ffi")),
    reason="FlashInfer/JAX FFI optional dependencies are not installed",
)
@pytest.mark.skipif(
    not _has_cuda_backend(),
    reason="FlashInfer FFI test requires a CUDA JAX backend",
)
@pytest.mark.parametrize("head_dim", [128, 256])
def test_kv_append_paged_nhd_flashinfer_matches_reference(head_dim):
    append_key = jnp.arange(4 * 2 * head_dim, dtype=jnp.float32).reshape(
        4, 2, head_dim
    )
    append_key = append_key.astype(jnp.bfloat16)
    append_value = (append_key + 1000).astype(jnp.bfloat16)
    batch_indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    positions = jnp.array([0, 0, 3, 2], dtype=jnp.int32)
    k_cache = jnp.full((4, 4, 2, head_dim), -1, dtype=jnp.bfloat16)
    v_cache = jnp.full((4, 4, 2, head_dim), -2, dtype=jnp.bfloat16)
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
    actual_k, actual_v = jax.jit(kv_append_paged_nhd)(
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


@pytest.mark.skipif(
    not (_has_module("flashinfer") and _has_module("jax_tvm_ffi")),
    reason="FlashInfer/JAX FFI optional dependencies are not installed",
)
@pytest.mark.skipif(
    not _has_cuda_backend(),
    reason="FlashInfer FFI test requires a CUDA JAX backend",
)
def test_backend_flashinfer_kv_append_opt_in_matches_canonical_update(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_FLASHINFER_KV_APPEND", "1")

    page_size = 4
    head_dim = 256
    layer_id = 1
    k = jnp.arange(2 * 4 * 2 * head_dim, dtype=jnp.float32).reshape(
        2, 4, 2, head_dim
    )
    k = k.astype(jnp.bfloat16)
    v = (k + 1000).astype(jnp.bfloat16)
    k_cache = jnp.full((3, 6, page_size, 2, head_dim), -1, dtype=jnp.bfloat16)
    v_cache = jnp.full((3, 6, page_size, 2, head_dim), -2, dtype=jnp.bfloat16)
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
