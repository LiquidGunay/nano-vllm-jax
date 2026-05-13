from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import ModelParams, lm_head_token_ids_and_topk


def test_lm_head_token_ids_and_topk_matches_full_logits():
    hidden = jnp.array(
        [
            [[0.2, -0.4, 0.7, 1.0], [1.1, 0.3, -0.2, 0.5]],
            [[-0.5, 0.8, 0.4, -0.1], [0.9, -0.6, 0.2, 0.7]],
        ],
        dtype=jnp.float32,
    )
    embed_tokens = jnp.linspace(-0.7, 0.9, 28, dtype=jnp.float32).reshape(7, 4)
    params = ModelParams(
        embed_tokens=embed_tokens,
        layers=[],
        norm_weight=jnp.array([1.0, 0.8, 1.2, 0.6], dtype=jnp.float32),
        lm_head=None,
    )
    config = SimpleNamespace(rms_norm_eps=1e-6)

    token_ids, top_values, top_indices = lm_head_token_ids_and_topk(
        hidden,
        params,
        config,
        is_prefill=False,
        top_k=2,
    )

    hidden_norm = rms_norm(hidden, params.norm_weight, config.rms_norm_eps).astype(jnp.float32)
    logits = jnp.dot(hidden_norm, embed_tokens.T)
    expected_top_values, expected_top_indices = jnp.flip(
        jnp.sort(logits, axis=-1)[..., -2:],
        axis=-1,
    ), jnp.argsort(logits, axis=-1)[..., -2:][..., ::-1]

    np.testing.assert_array_equal(np.array(token_ids), np.array(jnp.argmax(logits, axis=-1)))
    np.testing.assert_allclose(np.array(top_values), np.array(expected_top_values), rtol=0, atol=0)
    np.testing.assert_array_equal(np.array(top_indices), np.array(expected_top_indices))

    normed_token_ids, _, _ = lm_head_token_ids_and_topk(
        hidden_norm,
        params,
        config,
        hidden_is_normed=True,
        is_prefill=False,
    )
    np.testing.assert_array_equal(np.array(normed_token_ids), np.array(token_ids))
