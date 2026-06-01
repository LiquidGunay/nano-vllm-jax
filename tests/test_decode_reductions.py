import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.kernels.decode_reductions import (
    gdn_packed_decode_pre_normalize_qk_pallas,
    pallas_decode_rms_norm,
    triton_decode_rms_norm,
)
from nanovllm_jax.kernels.gdn_fla import gdn_packed_decode_pre_normalize_qk
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import ModelParams, lm_head_token_ids_and_topk


def _has_gpu() -> bool:
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="Pallas decode reductions require a GPU backend")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_decode_rms_norm_matches_jax(dtype):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 128), dtype=jnp.float32).astype(dtype)
    weight = jax.random.normal(jax.random.PRNGKey(1), (128,), dtype=jnp.float32)

    actual = pallas_decode_rms_norm(x, weight, eps=1e-6)
    expected = rms_norm(x, weight, eps=1e-6)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_triton_decode_rms_norm_matches_jax(dtype):
    x = jax.random.normal(jax.random.PRNGKey(6), (2, 3, 128), dtype=jnp.float32).astype(dtype)
    weight = jax.random.normal(jax.random.PRNGKey(7), (128,), dtype=jnp.float32)

    actual = triton_decode_rms_norm(x, weight, eps=1e-6)
    expected = rms_norm(x, weight, eps=1e-6)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_triton_decode_rms_norm_matches_per_head_weight():
    x = jax.random.normal(jax.random.PRNGKey(8), (2, 3, 4, 64), dtype=jnp.float32)
    weight = jax.random.normal(jax.random.PRNGKey(9), (4, 64), dtype=jnp.float32)

    actual = triton_decode_rms_norm(x, weight, eps=1e-6)
    expected = rms_norm(x, weight, eps=1e-6)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_lm_head_pallas_decode_rms_norm_matches_default(monkeypatch):
    hidden = jax.random.normal(jax.random.PRNGKey(2), (2, 1, 128), dtype=jnp.float32)
    embed_tokens = jax.random.normal(jax.random.PRNGKey(3), (64, 128), dtype=jnp.float32)
    params = ModelParams(
        embed_tokens=embed_tokens,
        layers=[],
        norm_weight=jax.random.normal(jax.random.PRNGKey(4), (128,), dtype=jnp.float32),
    )
    config = type("Config", (), {"rms_norm_eps": 1e-6})()

    monkeypatch.delenv("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", raising=False)
    expected = lm_head_token_ids_and_topk(hidden, params, config, is_prefill=False, top_k=0)[0]

    monkeypatch.setenv("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", "1")
    actual = lm_head_token_ids_and_topk(hidden, params, config, is_prefill=False, top_k=0)[0]

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_lm_head_triton_decode_rms_norm_matches_default(monkeypatch):
    hidden = jax.random.normal(jax.random.PRNGKey(10), (2, 1, 128), dtype=jnp.float32)
    embed_tokens = jax.random.normal(jax.random.PRNGKey(11), (64, 128), dtype=jnp.float32)
    params = ModelParams(
        embed_tokens=embed_tokens,
        layers=[],
        norm_weight=jax.random.normal(jax.random.PRNGKey(12), (128,), dtype=jnp.float32),
    )
    config = type("Config", (), {"rms_norm_eps": 1e-6})()

    monkeypatch.delenv("NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_TRITON_DECODE_RMSNORM", raising=False)
    expected = lm_head_token_ids_and_topk(hidden, params, config, is_prefill=False, top_k=0)[0]

    monkeypatch.setenv("NANO_VLLM_JAX_TRITON_DECODE_RMSNORM", "1")
    actual = lm_head_token_ids_and_topk(hidden, params, config, is_prefill=False, top_k=0)[0]

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_pallas_gdn_qk_prenorm_matches_reference(monkeypatch):
    batch = 2
    num_heads = 4
    value_heads = 4
    dim = 128
    mixed_qkv = jax.random.normal(
        jax.random.PRNGKey(5),
        (batch, 2 * num_heads * dim + value_heads * dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    state = jnp.zeros((batch, value_heads, dim, dim), dtype=jnp.float32)

    monkeypatch.delenv("NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM", raising=False)
    expected = gdn_packed_decode_pre_normalize_qk(mixed_qkv, state)
    actual = gdn_packed_decode_pre_normalize_qk_pallas(mixed_qkv, state)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)
