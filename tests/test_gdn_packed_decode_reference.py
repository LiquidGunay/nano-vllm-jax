"""Focused tests for the vLLM-style packed GDN decode reference ABI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    jax = None
    jnp = None
else:
    from nanovllm_jax.kernels.cuda_gdn import (
        gdn_packed_decode_reference_local_state,
        k_last_gdn_state_to_local,
        local_gdn_state_to_k_last,
        split_packed_gdn_decode_mixed_qkv,
    )
    from nanovllm_jax.model import jax_recurrent_gated_delta_rule

pytestmark = pytest.mark.skipif(
    jax is None,
    reason="JAX is required for GDN reference tests",
)


def _has_cuda_backend() -> bool:
    if jax is None:
        return False
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_packed_gdn_decode_reference_matches_current_recurrent_path():
    batch = 2
    num_heads = 4
    key_dim = 6
    value_dim = 8
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
    a = jnp.linspace(-0.3, 0.2, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
    )
    a_log = jnp.linspace(-0.2, 0.1, num_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(0.05, 0.2, num_heads, dtype=jnp.float32)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, value_dim, key_dim)

    mixed_qkv = jnp.concatenate(
        [
            query.reshape(batch, -1),
            key.reshape(batch, -1),
            value.reshape(batch, -1),
        ],
        axis=-1,
    )
    gate = -jnp.exp(a_log[None, :]) * jax.nn.softplus(a + dt_bias[None, :])
    beta = jax.nn.sigmoid(b)

    expected_out, expected_state = jax_recurrent_gated_delta_rule(
        query,
        key,
        value,
        gate[:, :, None],
        beta[:, :, None],
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = gdn_packed_decode_reference_local_state(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
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
def test_packed_gdn_decode_reference_repeats_qk_for_gva():
    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 6
    value_dim = 10
    mixed_qkv = jnp.linspace(
        -0.4,
        0.4,
        batch * (2 * num_q_heads * key_dim + num_value_heads * value_dim),
        dtype=jnp.float32,
    ).reshape(batch, -1)
    a = jnp.linspace(-0.2, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    a_log = jnp.zeros((num_value_heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((num_value_heads,), dtype=jnp.float32)
    state = jnp.zeros((batch, num_value_heads, value_dim, key_dim), dtype=jnp.float32)

    query, key, value = split_packed_gdn_decode_mixed_qkv(
        mixed_qkv,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    expected_query = jnp.repeat(query, num_value_heads // num_q_heads, axis=1)
    expected_key = jnp.repeat(key, num_value_heads // num_q_heads, axis=1)
    gate = -jnp.exp(a_log[None, :]) * jax.nn.softplus(a + dt_bias[None, :])
    beta = jax.nn.sigmoid(b)
    expected_out, expected_state = jax_recurrent_gated_delta_rule(
        expected_query,
        expected_key,
        value,
        gate[:, :, None],
        beta[:, :, None],
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = gdn_packed_decode_reference_local_state(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
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
def test_gdn_state_k_last_roundtrip_preserves_local_layout():
    state = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5)

    k_last = local_gdn_state_to_k_last(state)
    local = k_last_gdn_state_to_local(k_last)

    assert k_last.shape == state.shape
    np.testing.assert_array_equal(np.asarray(local), np.asarray(state))
