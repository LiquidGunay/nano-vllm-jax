"""Test causal_conv1d_update roll→concat equivalence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.layers import causal_conv1d_update


def _roll_scatter_reference(x, conv_state, weight, bias, activation):
    """Old roll+scatter implementation for parity check."""
    new_state = jnp.roll(conv_state, shift=-1, axis=-1)
    new_state = new_state.at[:, :, -1].set(x.squeeze(-1))
    out = jnp.einsum('bdk,dk->bd', new_state, weight)
    if bias is not None:
        out = out + bias
    if activation == "silu":
        out = jax.nn.silu(out)
    return out[:, :, None], new_state


@pytest.mark.parametrize("batch,dim,kernel", [(1, 8, 4), (4, 32, 4), (2, 16, 3)])
def test_causal_conv1d_update_parity(batch, dim, kernel):
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    conv_state = jax.random.normal(k1, (batch, dim, kernel))
    x = jax.random.normal(k2, (batch, dim, 1))
    weight = jax.random.normal(k3, (dim, kernel))
    bias = jax.random.normal(k4, (dim,))

    out_new, state_new = causal_conv1d_update(x, conv_state, weight, bias, "silu")
    out_ref, state_ref = _roll_scatter_reference(x, conv_state, weight, bias, "silu")

    np.testing.assert_allclose(np.asarray(state_new), np.asarray(state_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_new), np.asarray(out_ref), atol=1e-6)


def test_causal_conv1d_update_no_bias():
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    conv_state = jax.random.normal(k1, (2, 16, 4))
    x = jax.random.normal(k2, (2, 16, 1))
    weight = jax.random.normal(k3, (16, 4))

    out_new, state_new = causal_conv1d_update(x, conv_state, weight, None, "silu")
    out_ref, state_ref = _roll_scatter_reference(x, conv_state, weight, None, "silu")

    np.testing.assert_allclose(np.asarray(state_new), np.asarray(state_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_new), np.asarray(out_ref), atol=1e-6)
