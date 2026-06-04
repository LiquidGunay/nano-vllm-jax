"""Focused tests for the planned segmented GDN prefill ABI."""

import os
import importlib.util
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
from nanovllm_jax.kernels.gdn_fla import gdn_segmented_prefill_chunk32
from nanovllm_jax.model import jax_chunk_gated_delta_rule


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _has_jax_triton() -> bool:
    return importlib.util.find_spec("jax_triton") is not None


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
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_local_cumsum_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_local_cumsum_packed_triton,
    )

    cu_seqlens = jnp.array([0, 0, 5, 22], dtype=jnp.int32)
    gate = jnp.arange(22 * 3, dtype=jnp.float32).reshape(22, 3) * 0.01
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=8)

    actual = gdn_fla_chunk_local_cumsum_packed_triton(
        gate,
        cu_seqlens,
        chunk_size=8,
        chunk_indices=chunk_indices,
    )
    expected = gdn_fla_chunk_local_cumsum_packed_reference(
        gate,
        cu_seqlens,
        chunk_size=8,
        chunk_indices=chunk_indices,
        reverse=False,
    )
    expected_reverse = gdn_fla_chunk_local_cumsum_packed_reference(
        gate,
        cu_seqlens,
        chunk_size=8,
        chunk_indices=chunk_indices,
        reverse=True,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(gdn_fla_chunk_local_cumsum_packed_triton(
            gate,
            cu_seqlens,
            chunk_size=8,
            reverse=True,
        )),
        np.asarray(expected_reverse),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(gdn_fla_chunk_local_cumsum_packed_triton(
            gate,
            cu_seqlens,
            chunk_size=8,
        )),
        np.asarray(expected),
        rtol=0,
        atol=0,
    )


def test_gdn_fla_chunk_local_cumsum_packed_reference_handles_zero_length_rows():
    cu_seqlens = jnp.array([0, 0, 0, 5, 9], dtype=jnp.int32)
    gate = jnp.arange(9 * 2, dtype=jnp.float32).reshape(9, 2) * 0.03
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
    expected = gdn_fla_chunk_local_cumsum_packed_reference(
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    expected_manual = np.zeros((9, 2), dtype=np.float32)
    offsets = [0, 0, 0, 5, 9]
    for row in range(len(offsets) - 1):
        for start in range(offsets[row], offsets[row + 1], 4):
            end = min(offsets[row + 1], start + 4)
            expected_manual[start:end] = np.cumsum(np.asarray(gate[start:end]), axis=0)

    np.testing.assert_allclose(np.asarray(expected), expected_manual, rtol=0, atol=0)


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


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_scaled_dot_kkt_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_scaled_dot_kkt_packed_triton,
    )

    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.02
    beta = (
        jnp.arange(13 * 4, dtype=jnp.float32).reshape(13, 4) * 0.01 + 0.25
    )
    gate = jnp.linspace(-0.5, 0.5, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)

    actual = gdn_fla_chunk_scaled_dot_kkt_packed_triton(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    expected = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    # Triton and JAX reference can diverge at the last bit here due FP32
    # reduction-association differences in stage-local matmul-like accumulation.
    # This is a local arithmetic-drift gate for testing parity of the stage
    # output itself, not an end-to-end serving relaxation.
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-6
    )

    actual_none_gate = gdn_fla_chunk_scaled_dot_kkt_packed_triton(
        key,
        beta,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    expected_none_gate = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    np.testing.assert_allclose(
        np.asarray(actual_none_gate),
        np.asarray(expected_none_gate),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_solve_tril_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_solve_tril_packed_triton,
    )

    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    beta = jnp.ones((13, 4), dtype=jnp.float32) * 0.2
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    actual = gdn_fla_solve_tril_packed_triton(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    expected = gdn_fla_solve_tril_packed_reference(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    # This stage performs triangular solve accumulation in Triton; keep a tight
    # stage-local tolerance in case of minor FP32 accumulation-order drift.
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_solve_tril_packed_triton_matches_reference_num_heads3():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_solve_tril_packed_triton,
    )

    # Covers num_heads=3 with a short final chunk and key_dim=5 for the
    # composition edge case that previously diverged.
    cu_seqlens = jnp.array([0, 0, 3, 7, 9], dtype=jnp.int32)
    key = jnp.arange(9 * 3 * 5, dtype=jnp.float32).reshape(9, 3, 5) * 0.01
    beta = jnp.ones((9, 3), dtype=jnp.float32) * 0.2
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    actual = gdn_fla_solve_tril_packed_triton(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    expected = gdn_fla_solve_tril_packed_reference(
        attention_matrix,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_scaled_dot_kkt_packed_triton_block_dot_matches_reference_shape():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_scaled_dot_kkt_packed_triton_block,
    )

    rng = np.random.default_rng(7)
    cu_seqlens = jnp.array([0, 64], dtype=jnp.int32)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=64)
    key = jnp.asarray(
        rng.normal(0.0, 0.2, size=(64, 2, 64)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    beta = jnp.asarray(
        rng.uniform(0.1, 0.8, size=(64, 4)).astype(np.float32),
    )
    gate = jnp.asarray(
        rng.normal(0.0, 0.02, size=(64, 4)).astype(np.float32),
    )

    actual = gdn_fla_chunk_scaled_dot_kkt_packed_triton_block(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=64,
        chunk_indices=chunk_indices,
    )
    expected = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        key,
        beta,
        gate,
        cu_seqlens,
        chunk_size=64,
        chunk_indices=chunk_indices,
    )

    actual_np = np.asarray(actual)
    np.testing.assert_equal(actual_np.shape, (64, 4, 64))
    assert np.isfinite(actual_np).all()
    np.testing.assert_allclose(
        actual_np,
        np.asarray(expected),
        rtol=0,
        atol=2e-3,
    )
    for head in range(actual_np.shape[1]):
        np.testing.assert_allclose(
            np.triu(actual_np[:, head, :], k=0),
            np.zeros((64, 64), dtype=np.float32),
            rtol=0,
            atol=0,
        )


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


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 17, 20, 32, 48, 64])
def test_vllm_like_inverse_matches_full_precision_formula(n: int):
    import nanovllm_jax.kernels.gdn_fla as gdn_fla_mod

    rng = np.random.default_rng(123)
    strict_lower = np.tril(rng.standard_normal((n, n), dtype=np.float32) * 0.01, k=-1)
    actual = gdn_fla_mod._vllm_like_inverse_i_plus_strict_lower(
        strict_lower,
        output_dtype="float32",
    )
    expected = np.linalg.inv(np.eye(n, dtype=np.float32) + strict_lower.astype(np.float32))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-7)


def test_vllm_like_inverse_quantizes_like_vllm_output_dtype():
    import nanovllm_jax.kernels.gdn_fla as gdn_fla_mod
    import jax.numpy as jnp

    strict_lower = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.03, 0.0]],
        dtype=np.float32,
    )
    actual = gdn_fla_mod._vllm_like_inverse_i_plus_strict_lower(
        strict_lower,
        output_dtype="bfloat16",
    )
    expected = np.asarray(
        jnp.asarray(np.linalg.inv(np.eye(3, dtype=np.float32) + strict_lower), dtype=jnp.bfloat16),
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_vllm_like_recompute_w_u_quantizes_rhs_before_dot():
    def bf16(x):
        return np.asarray(jnp.asarray(x, dtype=jnp.bfloat16), dtype=np.float32)

    cu_seqlens = jnp.array([0, 4], dtype=jnp.int32)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
    key = jnp.array(
        np.linspace(-0.7, 0.9, 4 * 1 * 3, dtype=np.float32).reshape(4, 1, 3)
    )
    value = jnp.array(
        np.linspace(0.15, 1.25, 4 * 1 * 2, dtype=np.float32).reshape(4, 1, 2)
    )
    beta = jnp.array([[0.17], [0.31], [0.43], [0.59]], dtype=jnp.float32)
    gate = jnp.array([[-0.2], [0.05], [0.17], [0.33]], dtype=jnp.float32)
    attention_inverse = jnp.array(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[-0.11, 1.0, 0.0, 0.0]],
            [[0.07, -0.19, 1.0, 0.0]],
            [[-0.03, 0.13, -0.23, 1.0]],
        ],
        dtype=jnp.float32,
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
        vllm_like=True,
        stage_output_dtype="bfloat16",
    )

    matrix = bf16(np.asarray(attention_inverse)[:, 0, :])
    weighted_value = bf16(np.asarray(value)[:, 0, :] * np.asarray(beta)[:, 0, None])
    weighted_key = bf16(
        np.asarray(key)[:, 0, :]
        * np.asarray(beta)[:, 0, None]
        * np.exp(np.asarray(gate)[:, 0, None])
    )
    expected_u = bf16(matrix @ weighted_value)[:, None, :]
    expected_w = bf16(matrix @ weighted_key)[:, None, :]

    np.testing.assert_allclose(np.asarray(actual_u), expected_u, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(actual_w), expected_w, rtol=0, atol=0)


def test_vllm_like_chunk_delta_h_quantizes_delta_after_gate_for_recurrence():
    def bf16(x):
        return np.asarray(jnp.asarray(x, dtype=jnp.bfloat16), dtype=np.float32)

    cu_seqlens = jnp.array([0, 4], dtype=jnp.int32)
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=4,
    )
    key = jnp.array(
        np.linspace(-0.35, 0.55, 4 * 1 * 3, dtype=np.float32).reshape(4, 1, 3)
    )
    w = jnp.array(
        np.linspace(0.12, 0.82, 4 * 1 * 3, dtype=np.float32).reshape(4, 1, 3)
    )
    u = jnp.array(
        np.linspace(-0.44, 0.63, 4 * 1 * 2, dtype=np.float32).reshape(4, 1, 2)
    )
    gate = jnp.array([[-0.3], [-0.05], [0.1], [0.27]], dtype=jnp.float32)
    initial_state = jnp.array(
        np.linspace(-0.2, 0.4, 1 * 1 * 2 * 3, dtype=np.float32).reshape(1, 1, 2, 3)
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
        vllm_like=True,
        stage_output_dtype="bfloat16",
    )

    state_for_dot = bf16(np.asarray(initial_state)[0, 0])
    delta = bf16(np.asarray(u)[:, 0]) - bf16(np.asarray(w)[:, 0]) @ state_for_dot.T
    last_gate = np.asarray(gate)[-1, 0]
    update_delta = bf16(delta * np.exp(last_gate - np.asarray(gate)[:, 0])[:, None])
    expected_final = (
        state_for_dot * np.exp(last_gate)
        + update_delta.T @ bf16(np.asarray(key)[:, 0])
    )[None, None, :, :]

    np.testing.assert_allclose(np.asarray(actual_h)[0, 0], state_for_dot, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(actual_v_new)[:, 0], bf16(delta), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(actual_final), expected_final, rtol=0, atol=2e-7)


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax_triton is required")
def test_gdn_fla_recompute_w_u_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_recompute_w_u_packed_triton,
    )

    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    value = jnp.arange(13 * 4 * 5, dtype=jnp.float32).reshape(13, 4, 5) * 0.02
    beta = (
        jnp.arange(13 * 4, dtype=jnp.float32).reshape(13, 4) * 0.01 + 0.25
    )
    gate = jnp.linspace(-0.25, 0.25, 13 * 4, dtype=jnp.float32).reshape(13, 4)
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)
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

    actual_w, actual_u = gdn_fla_recompute_w_u_packed_triton(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    expected_w, expected_u = gdn_fla_recompute_w_u_packed_reference(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    assert actual_w.shape == expected_w.shape
    assert actual_u.shape == expected_u.shape

    # Keep stage-local FP32 strict tolerance only for deterministic
    # accumulation-order drift, not for end-to-end checks.
    np.testing.assert_allclose(
        np.asarray(actual_w),
        np.asarray(expected_w),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_u),
        np.asarray(expected_u),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax_triton is required")
def test_gdn_fla_recompute_w_u_packed_triton_block_dot_matches_reference(monkeypatch):
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_recompute_w_u_packed_triton,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT", "1")
    cu_seqlens = jnp.array([0, 96], dtype=jnp.int32)
    chunk_size = 64
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=chunk_size)
    keys = jax.random.split(jax.random.PRNGKey(20260603), 5)
    key = jax.random.normal(keys[0], (96, 2, 64), dtype=jnp.float32) * 0.02
    value = jax.random.normal(keys[1], (96, 4, 64), dtype=jnp.float32) * 0.03
    beta = jax.random.uniform(keys[2], (96, 4), dtype=jnp.float32, minval=0.2, maxval=0.7)
    gate = jax.random.normal(keys[3], (96, 4), dtype=jnp.float32) * 0.01
    attention_inverse = jax.random.normal(
        keys[4], (96, 4, chunk_size), dtype=jnp.float32
    ) * 0.02

    actual_w, actual_u = gdn_fla_recompute_w_u_packed_triton(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    expected_w, expected_u = gdn_fla_recompute_w_u_packed_reference(
        key,
        value,
        beta,
        gate,
        attention_inverse,
        cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    np.testing.assert_allclose(
        np.asarray(actual_w),
        np.asarray(expected_w),
        rtol=0,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(actual_u),
        np.asarray(expected_u),
        rtol=0,
        atol=2e-4,
    )


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


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_delta_h_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_delta_h_packed_triton,
    )

    cu_seqlens = jnp.array([0, 0, 5, 13], dtype=jnp.int32)
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=4,
    )
    key = jnp.arange(13 * 2 * 3, dtype=jnp.float32).reshape(13, 2, 3) * 0.01
    value = jnp.arange(13 * 4 * 5, dtype=jnp.float32).reshape(13, 4, 5) * 0.02
    beta = (
        jnp.arange(13 * 4, dtype=jnp.float32).reshape(13, 4) * 0.01 + 0.25
    )
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

    actual_h, actual_v_new, actual_final = gdn_fla_chunk_delta_h_packed_triton(
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
    expected_h, expected_v_new, expected_final = gdn_fla_chunk_delta_h_packed_reference(
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

    np.testing.assert_allclose(np.asarray(actual_h), np.asarray(expected_h), rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(actual_v_new),
        np.asarray(expected_v_new),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_final),
        np.asarray(expected_final),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_delta_h_packed_triton_block_dot_matches_reference(monkeypatch):
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_delta_h_packed_triton,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT", "1")
    cu_seqlens = jnp.array([0, 96], dtype=jnp.int32)
    chunk_size = 64
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=chunk_size,
    )
    keys = jax.random.split(jax.random.PRNGKey(20260602), 5)
    key = jax.random.normal(keys[0], (96, 2, 64), dtype=jnp.float32) * 0.02
    w = jax.random.normal(keys[1], (96, 4, 64), dtype=jnp.float32) * 0.02
    u = jax.random.normal(keys[2], (96, 4, 64), dtype=jnp.float32) * 0.03
    gate = jax.random.normal(keys[3], (96, 4), dtype=jnp.float32) * 0.01
    initial_state = jax.random.normal(keys[4], (1, 4, 64, 64), dtype=jnp.float32) * 0.01

    actual_h, actual_v_new, actual_final = gdn_fla_chunk_delta_h_packed_triton(
        key,
        w,
        u,
        gate,
        cu_seqlens,
        initial_state,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    expected_h, expected_v_new, expected_final = gdn_fla_chunk_delta_h_packed_reference(
        key,
        w,
        u,
        gate,
        cu_seqlens,
        initial_state,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    np.testing.assert_allclose(
        np.asarray(actual_h),
        np.asarray(expected_h),
        rtol=0,
        atol=8e-4,
    )
    np.testing.assert_allclose(
        np.asarray(actual_v_new),
        np.asarray(expected_v_new),
        rtol=0,
        atol=8e-4,
    )
    np.testing.assert_allclose(
        np.asarray(actual_final),
        np.asarray(expected_final),
        rtol=0,
        atol=8e-4,
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


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_fwd_o_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_fwd_o_packed_triton,
    )

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

    actual = gdn_fla_chunk_fwd_o_packed_triton(
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
    expected = gdn_fla_chunk_fwd_o_packed_reference(
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

    # Use strict stage-local FP32 tolerance for deterministic local arithmetic
    # accumulation ordering differences; this does not alter end-to-end checks.
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=0,
        atol=1e-6,
    )

    actual_no_gate = gdn_fla_chunk_fwd_o_packed_triton(
        query,
        key,
        v_new,
        h,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
        scale=0.7,
    )
    expected_no_gate = gdn_fla_chunk_fwd_o_packed_reference(
        query,
        key,
        v_new,
        h,
        None,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
        scale=0.7,
    )
    np.testing.assert_allclose(
        np.asarray(actual_no_gate),
        np.asarray(expected_no_gate),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_fwd_o_packed_triton_block_dot_matches_reference_shape():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_fwd_o_packed_triton_block,
    )

    cu_seqlens = jnp.array([0, 96], dtype=jnp.int32)
    chunk_size = 64
    chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=chunk_size)
    keys = jax.random.split(jax.random.PRNGKey(20260601), 5)
    query = jax.random.normal(keys[0], (96, 2, 64), dtype=jnp.float32) * 0.02
    key = jax.random.normal(keys[1], (96, 2, 64), dtype=jnp.float32) * 0.02
    v_new = jax.random.normal(keys[2], (96, 4, 64), dtype=jnp.float32) * 0.03
    h = jax.random.normal(keys[3], (2, 4, 64, 64), dtype=jnp.float32) * 0.01
    gate = jax.random.normal(keys[4], (96, 4), dtype=jnp.float32) * 0.01

    actual = gdn_fla_chunk_fwd_o_packed_triton_block(
        query,
        key,
        v_new,
        h,
        gate,
        cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        scale=0.125,
    )
    expected = gdn_fla_chunk_fwd_o_packed_reference(
        query,
        key,
        v_new,
        h,
        gate,
        cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        scale=0.125,
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=0,
        atol=2e-4,
    )


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


def _assert_stage_close(name: str, actual: jax.Array, expected: jax.Array) -> None:
    """Assert exact-match stage parity and raise with a focused stage label."""

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=0,
        atol=1e-6,
        err_msg=name,
    )


def _assert_allclose_with_nan_counts(
    name: str,
    actual: jax.Array,
    expected: jax.Array,
    *,
    atol: float = 1e-6,
) -> None:
    """Assert stage parity with a first-failed diagnostic including NaN counts."""

    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    actual_nan_count = int(np.isnan(actual_np).sum())
    expected_nan_count = int(np.isnan(expected_np).sum())
    if actual_nan_count or expected_nan_count:
        raise AssertionError(
            f"{name}: unexpected NaNs (actual={actual_nan_count}, expected={expected_nan_count})"
        )
    np.testing.assert_allclose(
        np.asarray(actual_np),
        np.asarray(expected_np),
        rtol=0,
        atol=atol,
        err_msg=name,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_gated_delta_rule_packed_triton_matches_reference():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_gated_delta_rule_packed_triton,
    )

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

    actual_out, actual_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        use_qk_l2norm_in_kernel=False,
    )
    expected_out, expected_state = gdn_fla_chunk_gated_delta_rule_packed_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        use_qk_l2norm_in_kernel=False,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=0,
        # Local FP32 stage arithmetic can drift a tiny amount between JAX and
        # Triton orderings. Keep this strict local-stage tolerance; it is not a
        # relaxation of final generation/token parity checks.
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=0,
        # Keep this tolerance scoped to isolated stage arithmetic; downstream
        # full-stack token-level correctness remains exact in separate tests.
        atol=4e-6,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_gated_delta_rule_packed_triton_block_dot_mixed_lengths(
    monkeypatch,
):
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_gated_delta_rule_packed_triton,
    )

    for env_name in (
        "NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS",
        "NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT",
        "NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT",
        "NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT",
        "NANO_VLLM_JAX_GDN_FWD_O_BLOCK_DOT",
    ):
        monkeypatch.setenv(env_name, "1")

    chunk_size = 64
    cu_seqlens = jnp.array([0, 0, 17, 81, 160, 290], dtype=jnp.int32)
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(
        cu_seqlens,
        chunk_size=chunk_size,
    )
    total_tokens = int(cu_seqlens[-1])
    batch = int(cu_seqlens.shape[0] - 1)
    num_heads = 4
    key_dim = 64
    value_dim = 64
    keys = jax.random.split(jax.random.PRNGKey(20260604), 6)
    query = jax.random.normal(
        keys[0],
        (total_tokens, num_heads, key_dim),
        dtype=jnp.float32,
    ) * 0.02
    key = jax.random.normal(
        keys[1],
        (total_tokens, num_heads, key_dim),
        dtype=jnp.float32,
    ) * 0.02
    value = jax.random.normal(
        keys[2],
        (total_tokens, num_heads, value_dim),
        dtype=jnp.float32,
    ) * 0.03
    gate = jax.random.normal(
        keys[3],
        (total_tokens, num_heads),
        dtype=jnp.float32,
    ) * 0.01
    beta = jax.random.uniform(
        keys[4],
        (total_tokens, num_heads),
        dtype=jnp.float32,
        minval=0.2,
        maxval=0.7,
    )
    initial_state = jax.random.normal(
        keys[5],
        (batch, num_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01

    actual_out, actual_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
        query,
        key,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=False,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        max_row_chunks=3,
    )
    expected_out, expected_state = gdn_fla_chunk_gated_delta_rule_packed_reference(
        query,
        key,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=False,
    )

    _assert_allclose_with_nan_counts(
        "block_dot_mixed_lengths_output",
        actual_out,
        expected_out,
        atol=2e-3,
    )
    _assert_allclose_with_nan_counts(
        "block_dot_mixed_lengths_state",
        actual_state,
        expected_state,
        atol=5e-3,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_gdn_fla_chunk_gated_delta_rule_triton_stage_diagnostics():
    from nanovllm_jax.kernels.gdn_fla_triton import (
        gdn_fla_chunk_delta_h_packed_triton,
        gdn_fla_chunk_fwd_o_packed_triton,
        gdn_fla_chunk_gated_delta_rule_packed_triton,
        gdn_fla_chunk_local_cumsum_packed_triton,
        gdn_fla_chunk_scaled_dot_kkt_packed_triton,
        gdn_fla_recompute_w_u_packed_triton,
        gdn_fla_solve_tril_packed_triton,
    )

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

    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size=4)

    gate_cumsum_t = gdn_fla_chunk_local_cumsum_packed_triton(
        packed_gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    gate_cumsum_ref = gdn_fla_chunk_local_cumsum_packed_reference(
        packed_gate,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    _assert_allclose_with_nan_counts("gate_cumsum", gate_cumsum_t, gate_cumsum_ref)

    attention_matrix_t = gdn_fla_chunk_scaled_dot_kkt_packed_triton(
        packed_key,
        packed_beta,
        gate_cumsum_t,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    attention_matrix_ref = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        packed_key,
        packed_beta,
        gate_cumsum_ref,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    _assert_allclose_with_nan_counts("attention_matrix", attention_matrix_t, attention_matrix_ref)

    attention_inverse_t = gdn_fla_solve_tril_packed_triton(
        attention_matrix_t,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    attention_inverse_ref = gdn_fla_solve_tril_packed_reference(
        attention_matrix_ref,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    _assert_allclose_with_nan_counts(
        "attention_inverse",
        attention_inverse_t,
        attention_inverse_ref,
    )

    w_t, u_t = gdn_fla_recompute_w_u_packed_triton(
        packed_key,
        packed_value,
        packed_beta,
        gate_cumsum_t,
        attention_inverse_t,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    w_ref, u_ref = gdn_fla_recompute_w_u_packed_reference(
        packed_key,
        packed_value,
        packed_beta,
        gate_cumsum_ref,
        attention_inverse_ref,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    _assert_allclose_with_nan_counts("recompute_w", w_t, w_ref)
    _assert_allclose_with_nan_counts("recompute_u", u_t, u_ref)

    h_t, v_new_t, final_state_t = gdn_fla_chunk_delta_h_packed_triton(
        packed_key,
        w_t,
        u_t,
        gate_cumsum_t,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    h_ref, v_new_ref, final_state_ref = gdn_fla_chunk_delta_h_packed_reference(
        packed_key,
        w_ref,
        u_ref,
        gate_cumsum_ref,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    # Allow strict local FP32 drift for this stage; this does not relax
    # downstream generated-token/logit parity checks.
    _assert_allclose_with_nan_counts("chunk_delta_h", h_t, h_ref, atol=2e-6)
    _assert_allclose_with_nan_counts("v_new", v_new_t, v_new_ref, atol=2e-6)
    # final_state carries long accumulation chains in this composition path; allow
    # a slightly wider local Triton-vs-JAX FP32 accumulation tolerance only for this
    # diagnostic, not for end-to-end token/logit acceptance.
    _assert_allclose_with_nan_counts("final_state", final_state_t, final_state_ref, atol=4e-6)

    output_t = gdn_fla_chunk_fwd_o_packed_triton(
        packed_query,
        packed_key,
        v_new_t,
        h_t,
        gate_cumsum_t,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    output_ref = gdn_fla_chunk_fwd_o_packed_reference(
        packed_query,
        packed_key,
        v_new_ref,
        h_ref,
        gate_cumsum_ref,
        cu_seqlens,
        chunk_size=4,
        chunk_indices=chunk_indices,
    )
    _assert_allclose_with_nan_counts("chunk_fwd_o", output_t, output_ref, atol=2e-6)

    output_cmp_t = gdn_fla_chunk_gated_delta_rule_packed_triton(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        use_qk_l2norm_in_kernel=False,
    )[0]
    output_cmp_ref = gdn_fla_chunk_gated_delta_rule_packed_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
        initial_state,
        chunk_size=4,
        use_qk_l2norm_in_kernel=False,
    )[0]
    _assert_allclose_with_nan_counts(
        "full_chunk_gated_delta_rule_output",
        output_cmp_t,
        output_cmp_ref,
        atol=2e-6,
    )

    if _has_cuda_backend():
        # Keep this helper side-effect free in CI with a clear failure surface.
        # This test is only for focused GPU diagnostics and does not change semantics.
        # The explicit calls above ensure the first failed stage is surfaced by name.
        pass


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
    assert float(jnp.max(jnp.abs(padded_state_diff))) <= 1e-5


def test_gdn_segmented_prefill_chunk32_matches_reference(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "gdn_fla")

    batch = 2
    num_heads = 2
    key_dim = 4
    value_dim = 4
    lengths = jnp.array([3, 5], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(42), 6)
    query = jax.random.normal(keys[0], (batch, num_heads, 5, key_dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (batch, num_heads, 5, key_dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (batch, num_heads, 5, value_dim), dtype=jnp.float32)
    gate = jax.random.normal(keys[3], (batch, num_heads, 5), dtype=jnp.float32) * 0.05
    beta = jax.random.uniform(keys[4], (batch, num_heads, 5), dtype=jnp.float32, minval=0.05, maxval=0.95)
    initial_state = jax.random.normal(keys[5], (batch, num_heads, value_dim, key_dim), dtype=jnp.float32) * 0.01

    valid = jnp.arange(5, dtype=jnp.int32)[None, :] < lengths[:, None]
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

    ref_out, ref_state = gdn_segmented_prefill_chunk32_reference(
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

    # Monkeypatch require_available so the public wrapper does not raise
    import nanovllm_jax.kernels.gdn_fla as _gdn_fla_mod
    monkeypatch.setattr(_gdn_fla_mod, "require_available", lambda: None)

    wrapper_out, wrapper_state = gdn_segmented_prefill_chunk32(
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

    np.testing.assert_allclose(
        np.asarray(wrapper_out),
        np.asarray(ref_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(wrapper_state),
        np.asarray(ref_state),
        rtol=2e-5,
        atol=2e-5,
    )
