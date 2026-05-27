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
    gdn_fla_chunk_delta_h_packed_reference,
    gdn_fla_chunk_fwd_o_packed_reference,
    gdn_fla_chunk_gated_delta_rule_packed_reference,
    gdn_fla_chunk_local_cumsum_packed_reference,
    gdn_fla_chunk_scaled_dot_kkt_packed_reference,
    gdn_fla_recompute_w_u_packed_reference,
    gdn_fla_solve_tril_packed_reference,
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


def test_gdn_fla_chunk_scaled_dot_kkt_packed_reference_matches_formula():
    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.02
    beta = (
        jnp.arange(13 * 4, dtype=jnp.float32).reshape(13, 4) * 0.01 + 0.25
    )
    gate = jnp.linspace(-0.5, 0.5, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)

    actual = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    expected = np.zeros((13, 4, 4), dtype=np.float32)
    key_np = np.asarray(key)
    beta_np = np.asarray(beta)
    gate_np = np.asarray(gate)
    offsets = [0, 0, 5, 13]
    for row in range(len(offsets) - 1):
        for start in range(offsets[row], offsets[row + 1], 4):
            end = min(offsets[row + 1], start + 4)
            length = end - start
            lower = np.tril(np.ones((length, length), dtype=np.float32), k=-1)
            for head in range(4):
                key_head = head // 2
                chunk_key = key_np[start:end, key_head, :].astype(np.float32)
                chunk_beta = beta_np[start:end, head].astype(np.float32)
                matrix = (chunk_key * chunk_beta[:, None]) @ chunk_key.T
                chunk_gate = gate_np[start:end, head].astype(np.float32)
                matrix *= np.exp(chunk_gate[:, None] - chunk_gate[None, :])
                expected[start:end, head, :length] = matrix * lower

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_gdn_fla_solve_tril_packed_reference_inverts_i_plus_a_per_chunk():
    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    beta = jnp.ones((13, 4), dtype=jnp.float32) * 0.2
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    actual = gdn_fla_solve_tril_packed_reference(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    expected = np.zeros((13, 4, 4), dtype=np.float32)
    attention_np = np.asarray(attention_matrix)
    offsets = [0, 0, 5, 13]
    for row in range(len(offsets) - 1):
        for start in range(offsets[row], offsets[row + 1], 4):
            end = min(offsets[row + 1], start + 4)
            length = end - start
            identity = np.eye(length, dtype=np.float32)
            for head in range(4):
                matrix = attention_np[start:end, head, :length].astype(np.float32)
                inverse = np.linalg.inv(identity + matrix)
                expected[start:end, head, :length] = inverse

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


def test_gdn_fla_recompute_w_u_packed_reference_matches_formula():
    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    value = jnp.arange(13 * 4 * 5, dtype=jnp.float32).reshape(13, 4, 5) * 0.02
    beta = jnp.linspace(0.2, 0.7, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    gate = jnp.linspace(-0.25, 0.25, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    attention_inverse = gdn_fla_solve_tril_packed_reference(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    actual_w, actual_u = gdn_fla_recompute_w_u_packed_reference(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    expected_w = np.zeros((13, 4, 3), dtype=np.float32)
    expected_u = np.zeros((13, 4, 5), dtype=np.float32)
    key_np = np.asarray(key)
    value_np = np.asarray(value)
    beta_np = np.asarray(beta)
    gate_np = np.asarray(gate)
    inverse_np = np.asarray(attention_inverse)
    offsets = [0, 0, 5, 13]
    for row in range(len(offsets) - 1):
        for start in range(offsets[row], offsets[row + 1], 4):
            end = min(offsets[row + 1], start + 4)
            length = end - start
            for head in range(4):
                key_head = head // 2
                matrix = inverse_np[start:end, head, :length].astype(np.float32)
                chunk_beta = beta_np[start:end, head].astype(np.float32)
                chunk_gate = gate_np[start:end, head].astype(np.float32)
                chunk_value = value_np[start:end, head, :].astype(np.float32)
                chunk_key = key_np[start:end, key_head, :].astype(np.float32)
                expected_u[start:end, head, :] = matrix @ (
                    chunk_value * chunk_beta[:, None]
                )
                expected_w[start:end, head, :] = matrix @ (
                    chunk_key
                    * chunk_beta[:, None]
                    * np.exp(chunk_gate)[:, None]
                )

    np.testing.assert_allclose(np.asarray(actual_w), expected_w, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_u), expected_u, rtol=1e-5, atol=1e-6)


def test_gdn_fla_chunk_delta_h_packed_reference_matches_formula():
    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=4,
    )
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    value = jnp.arange(13 * 4 * 5, dtype=jnp.float32).reshape(13, 4, 5) * 0.02
    beta = jnp.linspace(0.2, 0.7, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    gate = jnp.linspace(-0.25, 0.25, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    initial_state = (
        jnp.arange(3 * 4 * 5 * 3, dtype=jnp.float32).reshape(3, 4, 5, 3) * 0.001
    )
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    attention_inverse = gdn_fla_solve_tril_packed_reference(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    w, u = gdn_fla_recompute_w_u_packed_reference(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    actual_h, actual_v_new, actual_final = gdn_fla_chunk_delta_h_packed_reference(
        key,
        w,
        u,
        gate,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    expected_h = np.zeros((4, 4, 5, 3), dtype=np.float32)
    expected_v_new = np.zeros((13, 4, 5), dtype=np.float32)
    expected_final = np.asarray(initial_state).astype(np.float32).copy()
    key_np = np.asarray(key)
    w_np = np.asarray(w)
    u_np = np.asarray(u)
    gate_np = np.asarray(gate)
    offsets = [0, 0, 5, 13]
    chunk_offset_values = [0, 0, 2, 4]
    for row in range(len(offsets) - 1):
        row_state = expected_final[row].copy()
        for chunk in range(chunk_offset_values[row + 1] - chunk_offset_values[row]):
            flat_chunk = chunk_offset_values[row] + chunk
            start = offsets[row] + chunk * 4
            end = min(offsets[row + 1], start + 4)
            for head in range(4):
                key_head = head // 2
                head_state = row_state[head]
                expected_h[flat_chunk, head] = head_state
                delta = (
                    u_np[start:end, head, :]
                    - w_np[start:end, head, :] @ head_state.T
                )
                expected_v_new[start:end, head, :] = delta
                chunk_gate = gate_np[start:end, head].astype(np.float32)
                last_gate = chunk_gate[-1]
                update_delta = delta * np.exp(last_gate - chunk_gate)[:, None]
                row_state[head] = (
                    head_state * np.exp(last_gate)
                    + update_delta.T @ key_np[start:end, key_head, :]
                )
        expected_final[row] = row_state

    np.testing.assert_allclose(np.asarray(actual_h), expected_h, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(actual_v_new),
        expected_v_new,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_final),
        expected_final,
        rtol=1e-5,
        atol=1e-6,
    )


def test_gdn_fla_chunk_fwd_o_packed_reference_matches_formula():
    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=4,
    )
    query = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.015
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    value = jnp.arange(13 * 4 * 5, dtype=jnp.float32).reshape(13, 4, 5) * 0.02
    beta = jnp.linspace(0.2, 0.7, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    gate = jnp.linspace(-0.25, 0.25, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    initial_state = (
        jnp.arange(3 * 4 * 5 * 3, dtype=jnp.float32).reshape(3, 4, 5, 3) * 0.001
    )
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    attention_inverse = gdn_fla_solve_tril_packed_reference(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    w, u = gdn_fla_recompute_w_u_packed_reference(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    h, v_new, _ = gdn_fla_chunk_delta_h_packed_reference(
        key,
        w,
        u,
        gate,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    actual = gdn_fla_chunk_fwd_o_packed_reference(
        query,
        key,
        v_new,
        h,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
        scale=0.7,
    )

    expected = np.zeros((13, 4, 5), dtype=np.float32)
    query_np = np.asarray(query)
    key_np = np.asarray(key)
    v_new_np = np.asarray(v_new)
    h_np = np.asarray(h)
    gate_np = np.asarray(gate)
    chunk_index_values = np.asarray(chunk_indices)
    offsets = [0, 0, 5, 13]
    for flat_chunk, (row, chunk) in enumerate(chunk_index_values):
        start = offsets[row] + chunk * 4
        end = min(offsets[row + 1], start + 4)
        length = end - start
        causal = np.tril(np.ones((length, length), dtype=np.float32), k=0)
        for head in range(4):
            key_head = head // 2
            chunk_query = query_np[start:end, key_head, :].astype(np.float32)
            chunk_key = key_np[start:end, key_head, :].astype(np.float32)
            state = h_np[flat_chunk, head].astype(np.float32)
            state_out = chunk_query @ state.T
            attention = chunk_query @ chunk_key.T
            chunk_gate = gate_np[start:end, head].astype(np.float32)
            state_out *= np.exp(chunk_gate)[:, None]
            attention *= np.exp(chunk_gate[:, None] - chunk_gate[None, :])
            attention *= causal
            expected[start:end, head, :] = (
                state_out + attention @ v_new_np[start:end, head, :]
            ) * 0.7

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


def test_gdn_fla_chunk_gated_delta_rule_packed_reference_matches_segmented(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    batch = 4
    num_heads = 3
    seq_len = 9
    key_dim = 5
    value_dim = 4
    lengths = jnp.array([0, 3, 7, 9], dtype=jnp.int32)
    valid = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]
    keys = jax.random.split(jax.random.PRNGKey(20260527), 6)
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
    ) * 0.05
    beta = jax.random.uniform(
        keys[4],
        (batch, num_heads, seq_len),
        dtype=jnp.float32,
        minval=0.05,
        maxval=0.95,
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
    (
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
    ) = pack_padded_gdn_inputs(query, key, value, gate, beta, lengths)

    actual_out, actual_state = gdn_fla_chunk_gated_delta_rule_packed_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        use_qk_l2norm_in_kernel=True,
    )
    expected_out, expected_state = gdn_segmented_prefill_chunk32_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_beta,
        packed_gate,
        cu_seqlens,
        initial_state,
        chunk_size=4,
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
