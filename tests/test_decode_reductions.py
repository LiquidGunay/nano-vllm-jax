import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.kernels.decode_reductions import (
    gdn_packed_decode_pre_normalize_qk_pallas,
    pallas_decode_rms_norm,
    triton_decode_rms_padded_gemm,
    triton_decode_rms_norm,
)
from nanovllm_jax.kernels.gdn_fla import gdn_packed_decode_pre_normalize_qk
from nanovllm_jax.layers import rms_norm


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


def test_triton_decode_rms_padded_gemm_matches_jax():
    batch = 3
    rows = 8
    hidden = 128
    out_dim = 192
    x = jax.random.normal(jax.random.PRNGKey(13), (batch, 1, hidden), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    norm_weight = jax.random.normal(jax.random.PRNGKey(14), (hidden,), dtype=jnp.float32)
    mat_weight = jax.random.normal(
        jax.random.PRNGKey(15), (hidden, out_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    actual = triton_decode_rms_padded_gemm(
        x,
        norm_weight,
        mat_weight,
        eps=1e-6,
        rows=rows,
        block_n=64,
        block_k=64,
    )
    hidden_norm = rms_norm(x, norm_weight, eps=1e-6).astype(jnp.bfloat16)
    hidden_rows = hidden_norm.reshape(batch, hidden)
    expected = jnp.dot(
        jnp.pad(hidden_rows, ((0, rows - batch), (0, 0))),
        mat_weight,
    )[:batch, :].reshape(batch, 1, out_dim)

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_pallas_gdn_qk_prenorm_matches_reference():
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

    expected = gdn_packed_decode_pre_normalize_qk(mixed_qkv, state)
    actual = gdn_packed_decode_pre_normalize_qk_pallas(mixed_qkv, state)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)
