"""Focused tests for paged-attention kernel ABIs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.kernels.paged_attention import (
    dense_block_tables_to_kv_indptr,
    kv_last_page_len_from_seq_lens,
    paged_decode_attention_gqa_nhd_reference,
)
from nanovllm_jax.kv_cache import paged_attention_decode


def test_paged_decode_attention_gqa_nhd_reference_matches_current_decode_path():
    batch = 2
    page_size = 4
    max_pages_per_sequence = 3
    num_pages = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    seq_lens = jnp.array([7, 10], dtype=jnp.int32)
    block_tables = jnp.array(
        [
            [3, 1, 6],
            [0, 5, 2],
        ],
        dtype=jnp.int32,
    )
    k_cache = (
        jnp.arange(num_pages * page_size * num_kv_heads * head_dim, dtype=jnp.float32)
        .reshape(num_pages, page_size, num_kv_heads, head_dim)
        / 100.0
    )
    v_cache = (
        jnp.arange(num_pages * page_size * num_kv_heads * head_dim, dtype=jnp.float32)
        .reshape(num_pages, page_size, num_kv_heads, head_dim)
        / 70.0
    )
    q = (
        jnp.arange(batch * num_q_heads * head_dim, dtype=jnp.float32)
        .reshape(batch, num_q_heads, head_dim)
        / 50.0
    )
    kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(block_tables)
    kv_last_page_len = kv_last_page_len_from_seq_lens(seq_lens, page_size)
    scale = 1.0 / np.sqrt(head_dim)

    decode_reference = jax.jit(
        lambda query, key_cache, value_cache, indptr, indices, last_page_len, lens: (
            paged_decode_attention_gqa_nhd_reference(
                query,
                key_cache,
                value_cache,
                indptr,
                indices,
                last_page_len,
                lens,
                scale,
                max_pages_per_sequence=max_pages_per_sequence,
            )
        )
    )
    actual = decode_reference(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
    )
    expected = paged_attention_decode(
        query=q[:, None, :, :],
        k_cache=k_cache[None, ...],
        v_cache=v_cache[None, ...],
        block_table=block_tables,
        kv_lens=seq_lens,
        block_size=page_size,
        scale=scale,
        num_key_value_groups=num_q_heads // num_kv_heads,
        layer_idx=0,
        max_kv_len=max_pages_per_sequence * page_size,
    ).reshape(batch, num_q_heads, head_dim)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_paged_decode_attention_gqa_nhd_reference_model_head_shape_fp32():
    batch = 2
    page_size = 16
    max_pages_per_sequence = 2
    num_pages = 5
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 256
    seq_lens = jnp.array([17, 23], dtype=jnp.int32)
    block_tables = jnp.array(
        [
            [4, 1],
            [0, 3],
        ],
        dtype=jnp.int32,
    )
    k_cache = jnp.linspace(
        -0.25,
        0.25,
        num_pages * page_size * num_kv_heads * head_dim,
        dtype=jnp.float32,
    ).reshape(num_pages, page_size, num_kv_heads, head_dim)
    v_cache = jnp.linspace(
        0.1,
        0.6,
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
    scale = 1.0 / np.sqrt(head_dim)

    actual = paged_decode_attention_gqa_nhd_reference(
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

    assert actual.shape == (batch, num_q_heads, head_dim)
    assert actual.dtype == jnp.float32
