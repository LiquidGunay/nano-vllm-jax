from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax import nn
import jax.numpy as jnp

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import (
    ModelParams,
    _can_use_decode_padded_gemm,
    _compact_prefill_dot_if_enabled,
    _compact_prefill_mlp,
    _lm_head_greedy_top1_impl,
    lm_head_sample_token_ids,
    lm_head_token_ids_and_topk,
)


def test_lm_head_token_ids_and_topk_matches_full_logits(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_TRITON_DECODE_RMSNORM", raising=False)
    monkeypatch.setenv("NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_MATH", "0")
    monkeypatch.setenv("NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE", "fp32")
    monkeypatch.setenv("NANO_VLLM_JAX_DECODE_PADDED_GEMM", "0")
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
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=False,
        lm_head_decode_act_dtype="fp32",
    )

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


def test_lm_head_greedy_top1_impl_rejects_unimplemented_cutlass_backend(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_TRITON_DECODE_RMSNORM", raising=False)
    hidden = jnp.array([[[0.2, -0.4, 0.7, 1.0]]], dtype=jnp.float32)
    embed_tokens = jnp.linspace(-0.7, 0.9, 28, dtype=jnp.float32).reshape(7, 4)
    params = ModelParams(
        embed_tokens=embed_tokens,
        layers=[],
        norm_weight=jnp.array([1.0, 0.8, 1.2, 0.6], dtype=jnp.float32),
        lm_head=None,
    )
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=False,
        lm_head_decode_act_dtype="fp32",
        lm_head_greedy_top1_impl="cutlass",
    )

    assert _lm_head_greedy_top1_impl(config) == "cutlass"
    with np.testing.assert_raises(NotImplementedError):
        lm_head_token_ids_and_topk(hidden, params, config, is_prefill=False, top_k=0)

    token_ids, top_values, top_indices = lm_head_token_ids_and_topk(
        hidden,
        params,
        config,
        is_prefill=False,
        top_k=2,
    )
    assert token_ids.shape == (1, 1)
    assert top_values.shape == (1, 1, 2)
    assert top_indices.shape == (1, 1, 2)


def test_lm_head_greedy_top1_triton_matches_jax_on_cuda(monkeypatch):
    pytest.importorskip("jax_triton")
    if jax.default_backend() != "gpu":
        pytest.skip("Triton LM-head top1 requires the CUDA backend")
    monkeypatch.delenv("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_TRITON_DECODE_RMSNORM", raising=False)
    key_hidden, key_embed = jax.random.split(jax.random.PRNGKey(19))
    hidden = jax.random.normal(key_hidden, (3, 1, 32), dtype=jnp.float32).astype(jnp.bfloat16)
    embed_tokens = jax.random.normal(key_embed, (257, 32), dtype=jnp.float32).astype(jnp.bfloat16)
    params = ModelParams(
        embed_tokens=embed_tokens,
        layers=[],
        norm_weight=jnp.zeros((32,), dtype=jnp.float32),
        lm_head=None,
    )
    reference_config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=True,
        decode_padded_gemm_rows=8,
        decode_padded_gemm_max_out_dim=1024,
        lm_head_decode_act_dtype="bf16",
        lm_head_greedy_top1_impl="jax",
    )
    triton_config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=True,
        decode_padded_gemm_rows=8,
        decode_padded_gemm_max_out_dim=1024,
        lm_head_decode_act_dtype="bf16",
        lm_head_greedy_top1_impl="triton",
    )

    expected, _, _ = lm_head_token_ids_and_topk(
        hidden,
        params,
        reference_config,
        is_prefill=False,
        top_k=0,
    )
    actual, _, _ = lm_head_token_ids_and_topk(
        hidden,
        params,
        triton_config,
        is_prefill=False,
        top_k=0,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_decode_padded_gemm_default_cap_admits_qwen_vocab_projection():
    config = Qwen3_5Config(decode_padded_gemm=True)
    x = jnp.zeros((8, 1, 1), dtype=jnp.bfloat16)
    qwen_vocab_projection = jnp.zeros((1, 248064), dtype=jnp.bfloat16)

    assert config.decode_padded_gemm_max_out_dim >= qwen_vocab_projection.shape[1]
    assert _can_use_decode_padded_gemm(x, qwen_vocab_projection, config)


def test_lm_head_sample_token_ids_matches_greedy_and_categorical(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_MATH", "0")
    monkeypatch.setenv("NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE", "fp32")
    monkeypatch.setenv("NANO_VLLM_JAX_DECODE_PADDED_GEMM", "0")
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
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        decode_padded_gemm=False,
        lm_head_decode_act_dtype="fp32",
    )
    rng_keys = jax.random.split(jax.random.PRNGKey(7), 2)
    temperatures = jnp.array([0.0, 0.7], dtype=jnp.float32)

    actual = lm_head_sample_token_ids(
        hidden,
        params,
        config,
        temperatures=temperatures,
        rng_keys=rng_keys,
        is_prefill=False,
    )

    hidden_norm = rms_norm(hidden, params.norm_weight, config.rms_norm_eps).astype(jnp.float32)
    logits = jnp.dot(hidden_norm, embed_tokens.T)[:, 0]
    expected = jnp.array(
        [
            jnp.argmax(logits[0]),
            jax.random.categorical(rng_keys[1], logits[1] / temperatures[1], axis=-1),
        ],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


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
