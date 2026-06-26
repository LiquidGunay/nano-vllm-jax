import jax.numpy as jnp
import numpy as np

from nanovllm_jax.ops import ServingOps
from nanovllm_jax.config import Qwen3_5Config


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
    backend = ServingOps(Qwen3_5Config(gdn_prefill_post_conv_impl="reference"))

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
