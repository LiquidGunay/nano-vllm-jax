"""Focused tests for the GDN post-convolution prefill boundary."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kernels.cuda_fp32_ffi import gdn_post_conv_prep_fp32
from nanovllm_jax.kv_cache import HybridLayerState
from nanovllm_jax.layers import l2norm
from nanovllm_jax.model import gated_deltanet_block, init_transformer_block

jax.config.update("jax_default_matmul_precision", "highest")


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _small_gdn_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_size=4,
        linear_chunk_size=8,
        linear_recurrent_prefill_threshold=4,
        layer_types=("linear_attention",),
        linear_attn_layers=(0,),
        dtype="float32",
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_model_post_conv_prefill_reference_matches_default_with_mask(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", raising=False)
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    config = _small_gdn_config()
    params = init_transformer_block(jax.random.PRNGKey(0), config, layer_idx=0)
    batch = 3
    seq_len = 16
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim

    x = jnp.linspace(
        -0.35,
        0.45,
        batch * seq_len * config.hidden_size,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, config.hidden_size)
    hybrid_state = HybridLayerState(
        conv_state=jnp.linspace(
            -0.2,
            0.2,
            batch * 1 * conv_dim * config.linear_conv_kernel_size,
            dtype=jnp.float32,
        ).reshape(batch, 1, conv_dim, config.linear_conv_kernel_size),
        recurrent_state=jnp.linspace(
            -0.03,
            0.04,
            batch
            * 1
            * config.linear_num_value_heads
            * config.linear_value_head_dim
            * config.linear_key_head_dim,
            dtype=jnp.float32,
        ).reshape(
            batch,
            1,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        ),
    )
    lengths = jnp.array([16, 9, 0], dtype=jnp.int32)
    valid_token_mask = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :],
        (batch, seq_len),
    )

    expected_out, expected_state = gated_deltanet_block(
        x,
        params,
        positions,
        config,
        layer_idx=0,
        is_prefill=True,
        hybrid_state=hybrid_state,
        valid_token_mask=valid_token_mask,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", "reference")
    actual_out, actual_state = gated_deltanet_block(
        x,
        params,
        positions,
        config,
        layer_idx=0,
        is_prefill=True,
        hybrid_state=hybrid_state,
        valid_token_mask=valid_token_mask,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state.conv_state),
        np.asarray(expected_state.conv_state),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state.recurrent_state),
        np.asarray(expected_state.recurrent_state),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_cuda_post_conv_prep_matches_jax_prep_with_mask():
    batch = 2
    seq_len = 7
    num_key_heads = 2
    num_value_heads = 4
    key_dim = 4
    value_dim = 6
    conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    conv_out = jnp.linspace(
        -0.5,
        0.7,
        batch * seq_len * conv_dim,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, conv_dim)
    a = jnp.linspace(
        -0.3,
        0.4,
        batch * seq_len * num_value_heads,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_value_heads)
    b = jnp.linspace(
        -0.8,
        0.9,
        batch * seq_len * num_value_heads,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_value_heads)
    decay = jnp.linspace(0.8, 1.2, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.1, 0.2, num_value_heads, dtype=jnp.float32)
    valid_mask = jnp.array(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=jnp.int32,
    )

    key_total = num_key_heads * key_dim
    query = conv_out[:, :, :key_total].reshape(
        batch,
        seq_len,
        num_key_heads,
        key_dim,
    )
    key = conv_out[:, :, key_total : key_total * 2].reshape(
        batch,
        seq_len,
        num_key_heads,
        key_dim,
    )
    value = conv_out[:, :, key_total * 2 :].reshape(
        batch,
        seq_len,
        num_value_heads,
        value_dim,
    )
    gate = -decay * jax.nn.softplus(a + dt_bias)
    beta = jax.nn.sigmoid(b)
    valid = valid_mask.astype(jnp.bool_)
    query = jnp.where(valid[:, :, None, None], query, 0.0)
    key = jnp.where(valid[:, :, None, None], key, 0.0)
    value = jnp.where(valid[:, :, None, None], value, 0.0)
    gate = jnp.where(valid[:, :, None], gate, 0.0)
    beta = jnp.where(valid[:, :, None], beta, 0.0)
    repeat = num_value_heads // num_key_heads
    query = jnp.repeat(query, repeat, axis=2)
    key = jnp.repeat(key, repeat, axis=2)
    expected_query = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6).transpose(0, 2, 1, 3)
    expected_key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6).transpose(0, 2, 1, 3)
    expected_value = value.transpose(0, 2, 1, 3)
    expected_gate = gate.transpose(0, 2, 1)
    expected_beta = beta.transpose(0, 2, 1)

    actual_query, actual_key, actual_value, actual_gate, actual_beta = (
        gdn_post_conv_prep_fp32(
            conv_out,
            a,
            b,
            decay,
            dt_bias,
            valid_mask,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
        )
    )

    np.testing.assert_allclose(
        np.asarray(actual_query),
        np.asarray(expected_query),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_key),
        np.asarray(expected_key),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_value),
        np.asarray(expected_value),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gate),
        np.asarray(expected_gate),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_beta),
        np.asarray(expected_beta),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_model_cuda_post_conv_prep_matches_default_with_mask(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", raising=False)
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    config = _small_gdn_config()
    params = init_transformer_block(jax.random.PRNGKey(1), config, layer_idx=0)
    batch = 2
    seq_len = 16
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    x = jnp.linspace(
        -0.25,
        0.35,
        batch * seq_len * config.hidden_size,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, config.hidden_size)
    hybrid_state = HybridLayerState(
        conv_state=jnp.linspace(
            -0.1,
            0.2,
            batch * 1 * conv_dim * config.linear_conv_kernel_size,
            dtype=jnp.float32,
        ).reshape(batch, 1, conv_dim, config.linear_conv_kernel_size),
        recurrent_state=jnp.linspace(
            -0.02,
            0.03,
            batch
            * 1
            * config.linear_num_value_heads
            * config.linear_value_head_dim
            * config.linear_key_head_dim,
            dtype=jnp.float32,
        ).reshape(
            batch,
            1,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        ),
    )
    lengths = jnp.array([16, 11], dtype=jnp.int32)
    valid_token_mask = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :],
        (batch, seq_len),
    )

    expected_out, expected_state = gated_deltanet_block(
        x,
        params,
        positions,
        config,
        layer_idx=0,
        is_prefill=True,
        hybrid_state=hybrid_state,
        valid_token_mask=valid_token_mask,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", "cuda_prep_fp32")
    actual_out, actual_state = gated_deltanet_block(
        x,
        params,
        positions,
        config,
        layer_idx=0,
        is_prefill=True,
        hybrid_state=hybrid_state,
        valid_token_mask=valid_token_mask,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state.conv_state),
        np.asarray(expected_state.conv_state),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state.recurrent_state),
        np.asarray(expected_state.recurrent_state),
        rtol=2e-5,
        atol=2e-5,
    )
