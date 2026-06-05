from types import SimpleNamespace

import numpy as np
from jax import nn
import jax.numpy as jnp

from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import (
    ModelParams,
    _compact_prefill_dot_if_enabled,
    _compact_prefill_mlp,
    lm_head_token_ids_and_topk,
)


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


def test_lm_head_can_use_decode_padded_gemm_when_vocab_allowed():
    hidden = jnp.array(
        [
            [[0.2, -0.4, 0.7, 1.0]],
            [[-0.5, 0.8, 0.4, -0.1]],
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
    reference_config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=False,
        lm_head_decode_act_dtype="fp32",
    )
    padded_config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=True,
        decode_padded_gemm_rows=4,
        decode_padded_gemm_max_out_dim=7,
        lm_head_decode_act_dtype="fp32",
    )

    expected = lm_head_token_ids_and_topk(
        hidden,
        params,
        reference_config,
        is_prefill=False,
        top_k=2,
    )
    actual = lm_head_token_ids_and_topk(
        hidden,
        params,
        padded_config,
        is_prefill=False,
        top_k=2,
    )

    np.testing.assert_array_equal(np.array(actual[0]), np.array(expected[0]))
    np.testing.assert_allclose(np.array(actual[1]), np.array(expected[1]), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.array(actual[2]), np.array(expected[2]))


def test_compact_prefill_mlp_matches_dense_on_valid_tokens(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_COMPACT_PREFILL_MLP", "1")
    x = jnp.array(
        [
            [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2], [0.5, 0.6, -0.8], [0.9, -0.1, 0.4]],
            [[-0.5, 0.8, 0.4], [0.9, -0.6, 0.2], [0.3, 0.1, -0.7], [0.8, 0.2, 0.5]],
        ],
        dtype=jnp.float32,
    )
    gate_weight = jnp.linspace(-0.6, 0.7, 15, dtype=jnp.float32).reshape(3, 5)
    up_weight = jnp.linspace(0.4, -0.5, 15, dtype=jnp.float32).reshape(3, 5)
    down_weight = jnp.linspace(-0.3, 0.8, 20, dtype=jnp.float32).reshape(5, 4)
    valid_mask = jnp.array(
        [
            [True, True, True, False],
            [True, False, False, False],
        ]
    )

    compact = _compact_prefill_mlp(
        x,
        gate_weight,
        up_weight,
        down_weight,
        nn.silu,
        valid_mask,
        compact_num_tokens=4,
    )
    dense = jnp.dot(nn.silu(jnp.dot(x, gate_weight)) * jnp.dot(x, up_weight), down_weight)

    compact_np = np.array(compact)
    dense_np = np.array(dense)
    valid_np = np.array(valid_mask)
    np.testing.assert_allclose(compact_np[valid_np], dense_np[valid_np], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(compact_np[~valid_np], np.zeros_like(compact_np[~valid_np]))


def test_compact_prefill_dot_matches_dense_on_valid_tokens():
    x = jnp.array(
        [
            [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2], [0.5, 0.6, -0.8]],
            [[-0.5, 0.8, 0.4], [0.9, -0.6, 0.2], [0.3, 0.1, -0.7]],
        ],
        dtype=jnp.float32,
    )
    weight = jnp.linspace(-0.6, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
    valid_mask = jnp.array([[True, True, False], [True, False, False]])

    compact = _compact_prefill_dot_if_enabled(
        x,
        weight,
        valid_mask,
        compact_num_tokens=3,
        enabled=True,
    )
    dense = jnp.dot(x, weight)

    compact_np = np.array(compact)
    dense_np = np.array(dense)
    valid_np = np.array(valid_mask)
    np.testing.assert_allclose(compact_np[valid_np], dense_np[valid_np], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(compact_np[~valid_np], np.zeros_like(compact_np[~valid_np]))
