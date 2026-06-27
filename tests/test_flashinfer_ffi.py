"""Focused CUDA tests for optional FlashInfer/JAX FFI kernels."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.ops import ServingOps
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kernels.flashinfer_ffi import (
    kv_append_paged_nhd,
    kv_append_paged_nhd_reference,
    paged_decode_attention_gqa_nhd,
    paged_decode_attention_with_kv_append_gqa_nhd,
    radix_topk,
)
from nanovllm_jax.kernels.paged_attention import (
    dense_block_tables_to_kv_indptr,
    kv_last_page_len_from_seq_lens,
    paged_decode_attention_gqa_nhd_reference,
)
from nanovllm_jax.cache import (
    AttentionMetadata,
    KVCacheStorage,
    compute_slot_mapping,
    update_kv_cache,
)


def test_kv_append_paged_nhd_rejects_fp32_cache_before_ffi_registration():
    append_key = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    append_value = jnp.zeros_like(append_key)
    k_cache = jnp.zeros((1, 1, 2, 4), dtype=jnp.float32)
    v_cache = jnp.zeros_like(k_cache)

    with pytest.raises(ValueError, match="supports only FP16/BF16"):
        kv_append_paged_nhd(
            append_key,
            append_value,
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            k_cache,
            v_cache,
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
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
def test_radix_topk_flashinfer_matches_jax_top1():
    logits = jnp.array(
        [
            [0.0, 1.5, -1.0, 0.5],
            [3.0, -2.0, 8.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    values, indices = jax.jit(lambda x: radix_topk(x, top_k=1))(logits)

    np.testing.assert_allclose(
        np.asarray(values),
        np.asarray(jnp.max(logits, axis=-1, keepdims=True)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        np.asarray(indices),
        np.asarray(jnp.argmax(logits, axis=-1, keepdims=True)),
    )


@pytest.mark.skipif(
    not (_has_module("flashinfer") and _has_module("jax_tvm_ffi")),
    reason="FlashInfer/JAX FFI optional dependencies are not installed",
)
@pytest.mark.skipif(
    not _has_cuda_backend(),
    reason="FlashInfer FFI test requires a CUDA JAX backend",
)
def test_paged_decode_attention_flashinfer_matches_reference():
    key = jax.random.PRNGKey(0)
    batch = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128
    page_size = 16
    max_pages_per_sequence = 2
    num_pages = batch * max_pages_per_sequence
    scale = 1.0 / np.sqrt(head_dim)
    block_tables = jnp.array([[2, 0], [3, 1]], dtype=jnp.int32)
    seq_lens = jnp.array([17, 30], dtype=jnp.int32)
    query = jax.random.normal(
        key,
        (batch, num_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    k_cache = jax.random.normal(
        jax.random.fold_in(key, 1),
        (num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.fold_in(key, 2),
        (num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(block_tables)
    kv_last_page_len = kv_last_page_len_from_seq_lens(seq_lens, page_size)

    actual = jax.jit(
        lambda q, k, v: paged_decode_attention_gqa_nhd(
            q,
            k,
            v,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            scale=scale,
        )
    )(query, k_cache, v_cache)
    expected = paged_decode_attention_gqa_nhd_reference(
        query,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
        scale,
        max_pages_per_sequence=max_pages_per_sequence,
    )

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.skipif(
    not (_has_module("flashinfer") and _has_module("jax_tvm_ffi")),
    reason="FlashInfer/JAX FFI optional dependencies are not installed",
)
@pytest.mark.skipif(
    not _has_cuda_backend(),
    reason="FlashInfer FFI test requires a CUDA JAX backend",
)
def test_paged_decode_fused_append_flashinfer_matches_reference():
    key = jax.random.PRNGKey(1)
    batch = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128
    page_size = 16
    max_pages_per_sequence = 2
    num_pages = batch * max_pages_per_sequence
    layer_id = 1
    scale = 1.0 / np.sqrt(head_dim)
    block_tables = jnp.array([[2, 0], [3, 1]], dtype=jnp.int32)
    seq_lens = jnp.array([17, 30], dtype=jnp.int32)
    positions = seq_lens - 1
    query = jax.random.normal(
        key,
        (batch, num_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    new_k = jax.random.normal(
        jax.random.fold_in(key, 1),
        (batch, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    new_v = jax.random.normal(
        jax.random.fold_in(key, 2),
        (batch, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    k_cache = jax.random.normal(
        jax.random.fold_in(key, 3),
        (2, num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.fold_in(key, 4),
        (2, num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(block_tables)
    kv_last_page_len = kv_last_page_len_from_seq_lens(seq_lens, page_size)

    actual, actual_k, actual_v = jax.jit(
        lambda q, nk, nv, kc, vc: paged_decode_attention_with_kv_append_gqa_nhd(
            q,
            nk,
            nv,
            kc,
            vc,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            positions,
            layer_id=layer_id,
            scale=scale,
        )
    )(query, new_k, new_v, k_cache, v_cache)

    expected_k_layer, expected_v_layer = kv_append_paged_nhd_reference(
        new_k,
        new_v,
        jnp.arange(batch, dtype=jnp.int32),
        positions.astype(jnp.int32),
        k_cache[layer_id],
        v_cache[layer_id],
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )
    expected_k = k_cache.at[layer_id].set(expected_k_layer)
    expected_v = v_cache.at[layer_id].set(expected_v_layer)
    expected = paged_decode_attention_gqa_nhd_reference(
        query,
        expected_k[layer_id],
        expected_v[layer_id],
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
        scale,
        max_pages_per_sequence=max_pages_per_sequence,
    )

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
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
def test_backend_rejects_removed_flashinfer_kv_append_opt_in():
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
    with pytest.raises(ValueError, match="full_attention_kv_append_impl"):
        ServingOps(Qwen3_5Config(full_attention_kv_append_impl="flashinfer")).write_kv(
            layer_id=layer_id,
            k=k,
            v=v,
            cache=KVCacheStorage(k_cache, v_cache),
            metadata=metadata,
        )
