"""Focused tests for the planned segmented GDN prefill ABI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from nanovllm_jax.kernels.gdn_fla import (
    gdn_fla_chunk_local_cumsum_packed_reference,
    gdn_segmented_prefill_chunk32_reference,
    pack_padded_gdn_inputs,
    prepare_gdn_fla_chunk_metadata,
    unpack_segmented_gdn_output,
)
from nanovllm_jax.model import jax_chunk_gated_delta_rule


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def test_prepare_gdn_fla_chunk_metadata_preserves_zero_rows():
    cu_seqlens = jnp.array([0, 0, 5, 37, 74, 138], dtype=jnp.int32)

    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=32,
    )

    np.testing.assert_array_equal(
        np.asarray(chunk_indices),
        np.asarray(
            [
                [1, 0],
                [2, 0],
                [3, 0],
                [3, 1],
                [4, 0],
                [4, 1],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(chunk_offsets),
        np.asarray([0, 0, 1, 2, 4, 6], dtype=np.int32),
    )


def test_prepare_gdn_fla_chunk_metadata_empty_batch_rows():
    cu_seqlens = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=64,
    )

    assert np.asarray(chunk_indices).shape == (0, 2)
    np.testing.assert_array_equal(
        np.asarray(chunk_offsets),
        np.asarray([0, 0, 0, 0], dtype=np.int32),
    )


def test_gdn_fla_chunk_local_cumsum_packed_reference_resets_per_chunk():
    cu_seqlens = jnp.array([0, 0, 5, 22], dtype=jnp.int32)
    gate = jnp.arange(22 * 2, dtype=jnp.float32).reshape(22, 2) * 0.01
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=8)

    actual = gdn_fla_chunk_local_cumsum_packed_reference(
        gate,
        cu_seqlens,
        chunk_size=8,
        chunk_indices=chunk_indices,
    )
    actual_reverse = gdn_fla_chunk_local_cumsum_packed_reference(
        gate,
        cu_seqlens,
        chunk_size=8,
        chunk_indices=chunk_indices,
        reverse=True,
    )

    expected = np.zeros((22, 2), dtype=np.float32)
    expected_reverse = np.zeros((22, 2), dtype=np.float32)
    offsets = [0, 0, 5, 22]
    for row in range(len(offsets) - 1):
        for start in range(offsets[row], offsets[row + 1], 8):
            end = min(offsets[row + 1], start + 8)
            chunk = np.asarray(gate[start:end], dtype=np.float32)
            expected[start:end] = np.cumsum(chunk, axis=0)
            expected_reverse[start:end] = np.flip(
                np.cumsum(np.flip(chunk, axis=0), axis=0),
                axis=0,
            )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(actual_reverse),
        expected_reverse,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_segmented_gdn_prefill_reference_matches_padded_chunk32(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    batch = 5
    num_heads = 3
    seq_len = 64
    key_dim = 16
    value_dim = 16
    lengths = jnp.array([0, 5, 32, 37, 64], dtype=jnp.int32)
    valid = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]

    keys = jax.random.split(jax.random.PRNGKey(20260526), 6)
    query = jax.random.normal(
        keys[0],
        (batch, num_heads, seq_len, key_dim),
        dtype=jnp.float32,
    )
    key = jax.random.normal(
        keys[1],
        (batch, num_heads, seq_len, key_dim),
        dtype=jnp.float32,
    )
    value = jax.random.normal(
        keys[2],
        (batch, num_heads, seq_len, value_dim),
        dtype=jnp.float32,
    )
    gate = jax.random.normal(
        keys[3],
        (batch, num_heads, seq_len),
        dtype=jnp.float32,
    ) * 0.1
    beta = jax.random.uniform(
        keys[4],
        (batch, num_heads, seq_len),
        dtype=jnp.float32,
    )
    initial_state = jax.random.normal(
        keys[5],
        (batch, num_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01

    query = jnp.where(valid[:, None, :, None], query, 0.0)
    key = jnp.where(valid[:, None, :, None], key, 0.0)
    value = jnp.where(valid[:, None, :, None], value, 0.0)
    gate = jnp.where(valid[:, None, :], gate, 0.0)
    beta = jnp.where(valid[:, None, :], beta, 0.0)

    padded_out, padded_state = jax_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        chunk_size=32,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    (
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
    ) = pack_padded_gdn_inputs(query, key, value, gate, beta, lengths)
    packed_out, segmented_state = gdn_segmented_prefill_chunk32_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_beta,
        packed_gate,
        cu_seqlens,
        initial_state,
        chunk_size=32,
        use_qk_l2norm_in_kernel=True,
    )
    padded_packed_out, padded_segmented_state = gdn_segmented_prefill_chunk32_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_beta,
        packed_gate,
        cu_seqlens,
        initial_state,
        chunk_size=32,
        use_qk_l2norm_in_kernel=True,
        reference_seq_len=seq_len,
    )
    segmented_out = unpack_segmented_gdn_output(packed_out, cu_seqlens, seq_len)
    padded_segmented_out = unpack_segmented_gdn_output(
        padded_packed_out,
        cu_seqlens,
        seq_len,
    )

    valid_output_diff = jnp.where(
        valid[:, None, :, None],
        segmented_out.astype(jnp.float32) - padded_out.astype(jnp.float32),
        0.0,
    )
    state_diff = segmented_state.astype(jnp.float32) - padded_state.astype(jnp.float32)
    padded_valid_output_diff = jnp.where(
        valid[:, None, :, None],
        padded_segmented_out.astype(jnp.float32) - padded_out.astype(jnp.float32),
        0.0,
    )
    padded_state_diff = (
        padded_segmented_state.astype(jnp.float32) - padded_state.astype(jnp.float32)
    )

    assert np.asarray(packed_query).shape[0] == int(np.asarray(lengths).sum())
    assert np.asarray(cu_seqlens).tolist() == [0, 0, 5, 37, 74, 138]
    assert float(jnp.max(jnp.abs(valid_output_diff))) <= 1e-5
    assert float(jnp.max(jnp.abs(state_diff))) <= 1e-5
    assert float(jnp.max(jnp.abs(padded_valid_output_diff))) <= 1e-5
    assert float(jnp.max(jnp.abs(padded_state_diff))) <= 1e-5
