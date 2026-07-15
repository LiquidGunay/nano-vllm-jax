import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.ops import ServingOps
from nanovllm_jax.fastpath import KernelPlan


def _has_cuda():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def test_strict_policy_allows_only_packed_prefix_state_kernel():
    backend = ServingOps(
        KernelPlan(
            gdn_prefill="triton_fla_padded",
            gdn_disable_fallbacks=True,
        )
    )

    assert backend.use_gdn_post_conv_prefill(
        recurrent=True,
        return_prefix_state=True,
        return_first_prefix_state=False,
        packed=True,
    )
    with pytest.raises(RuntimeError, match="return_prefix_state"):
        backend.use_gdn_post_conv_prefill(
            recurrent=True,
            return_prefix_state=True,
            return_first_prefix_state=False,
            packed=False,
        )


def test_packed_post_conv_prefill_returns_prefix_states():
    num_key_heads = 1
    num_value_heads = 2
    key_dim = 3
    value_dim = 2
    token_count = 5
    conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    conv_out = jnp.linspace(
        -0.4,
        0.5,
        token_count * conv_dim,
        dtype=jnp.float32,
    ).reshape(1, token_count, conv_dim)
    a = jnp.zeros((1, token_count, num_value_heads), dtype=jnp.float32)
    b = jnp.linspace(
        -0.2,
        0.2,
        token_count * num_value_heads,
        dtype=jnp.float32,
    ).reshape(1, token_count, num_value_heads)
    decay = jnp.linspace(0.5, 0.8, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.1, 0.1, num_value_heads, dtype=jnp.float32)
    query_start_loc = jnp.array([0, 2, 5], dtype=jnp.int32)
    initial_state = jnp.zeros(
        (2, num_value_heads, value_dim, key_dim),
        dtype=jnp.float32,
    )
    backend = ServingOps(KernelPlan(gdn_prefill="reference"))

    without_prefix = backend.gated_delta_packed_prefill_post_conv(
        conv_out,
        a,
        b,
        decay,
        dt_bias,
        query_start_loc,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        chunk_size=8,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
        max_row_tokens=3,
    )
    with_prefix = backend.gated_delta_packed_prefill_post_conv(
        conv_out,
        a,
        b,
        decay,
        dt_bias,
        query_start_loc,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        chunk_size=8,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
        max_row_tokens=3,
        return_prefix_state=True,
    )

    output, final_state = without_prefix
    prefix_output, prefix_final_state, prefix_states = with_prefix

    assert prefix_output.shape == output.shape
    assert prefix_final_state.shape == final_state.shape
    assert prefix_states.shape == (token_count, num_value_heads, value_dim, key_dim)
    np.testing.assert_allclose(np.asarray(prefix_output), np.asarray(output))
    np.testing.assert_allclose(np.asarray(prefix_final_state), np.asarray(final_state))
    np.testing.assert_allclose(np.asarray(prefix_states[1]), np.asarray(final_state[0]))
    np.testing.assert_allclose(np.asarray(prefix_states[4]), np.asarray(final_state[1]))


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for Triton prefix state")
def test_triton_packed_prefix_state_matches_reference():
    token_count = 6
    heads = 2
    key_dim = 8
    value_dim = 8
    conv_dim = 2 * heads * key_dim + heads * value_dim
    conv_out = jnp.linspace(
        -0.4,
        0.5,
        token_count * conv_dim,
        dtype=jnp.float32,
    ).reshape(1, token_count, conv_dim)
    a = jnp.zeros((1, token_count, heads), dtype=jnp.float32)
    b = jnp.linspace(-0.2, 0.2, token_count * heads, dtype=jnp.float32).reshape(
        1, token_count, heads
    )
    decay = jnp.linspace(0.5, 0.8, heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.1, 0.1, heads, dtype=jnp.float32)
    starts = jnp.array([0, 3, 6], dtype=jnp.int32)
    initial = jnp.linspace(
        -0.05,
        0.05,
        2 * heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(2, heads, value_dim, key_dim)
    kwargs = dict(
        num_key_heads=heads,
        num_value_heads=heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        chunk_size=8,
        initial_state=initial,
        use_qk_l2norm_in_kernel=True,
        max_row_tokens=3,
        return_prefix_state=True,
    )
    reference = ServingOps(KernelPlan(gdn_prefill="reference"))
    triton = ServingOps(
        KernelPlan(
            gdn_prefill="triton_fla_padded",
            gdn_disable_fallbacks=True,
        )
    )

    expected = reference.gated_delta_packed_prefill_post_conv(
        conv_out, a, b, decay, dt_bias, starts, **kwargs
    )
    actual = triton.gated_delta_packed_prefill_post_conv(
        conv_out, a, b, decay, dt_bias, starts, **kwargs
    )

    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=2e-4, atol=2e-4)
