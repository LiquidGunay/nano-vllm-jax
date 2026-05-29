"""Focused tests for the GDN post-convolution prefill boundary."""

import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.backends import PureJAXBackend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kernels.cuda_fp32_ffi import gdn_post_conv_prep_fp32
from nanovllm_jax.kernels.gdn_fla import (
    gdn_fla_prefill_chunk32_fp32_reference,
    gdn_fla_prefill_varlen_composed_reference,
    gdn_fla_prefill_varlen_reference,
    gdn_post_conv_prefill_reference_from_decay,
    pack_prepared_gdn_prefill_inputs,
    prepare_gdn_fla_prefill_kernel_inputs,
    prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
    unpack_prepared_gdn_prefill_output,
    validate_gdn_fla_prefill_kernel_output,
)
from nanovllm_jax.kv_cache import HybridLayerState
from nanovllm_jax.layers import l2norm
from nanovllm_jax.model import gated_deltanet_block, init_transformer_block

jax.config.update("jax_default_matmul_precision", "highest")


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _has_jax_triton() -> bool:
    return importlib.util.find_spec("jax_triton") is not None


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
def test_post_conv_fla_input_prep_matches_manual_reference():
    batch = 2
    seq_len = 5
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
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
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
    expected_query = jnp.repeat(query, repeat, axis=2)
    expected_key = jnp.repeat(key, repeat, axis=2)
    expected_seq_lens = valid_mask.sum(axis=1).astype(jnp.int32)

    actual = prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
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
        normalize_qk=False,
    )
    (
        actual_query,
        actual_key,
        actual_value,
        actual_gate,
        actual_beta,
        actual_seq_lens,
    ) = actual

    assert actual_query.shape == (batch, seq_len, num_value_heads, key_dim)
    assert actual_key.shape == (batch, seq_len, num_value_heads, key_dim)
    assert actual_value.shape == (batch, seq_len, num_value_heads, value_dim)
    np.testing.assert_allclose(np.asarray(actual_query), np.asarray(expected_query))
    np.testing.assert_allclose(np.asarray(actual_key), np.asarray(expected_key))
    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(value))
    np.testing.assert_allclose(np.asarray(actual_gate), np.asarray(gate))
    np.testing.assert_allclose(np.asarray(actual_beta), np.asarray(beta))
    np.testing.assert_array_equal(np.asarray(actual_seq_lens), np.asarray(expected_seq_lens))

    normalized_query, normalized_key, *_ = (
        prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
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
            normalize_qk=True,
        )
    )
    np.testing.assert_allclose(
        np.asarray(normalized_query),
        np.asarray(l2norm(expected_query.astype(jnp.float32), axis=-1, eps=1e-6)),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(normalized_key),
        np.asarray(l2norm(expected_key.astype(jnp.float32), axis=-1, eps=1e-6)),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_prepared_fla_chunk32_reference_matches_post_conv_reference():
    batch = 2
    seq_len = 8
    num_key_heads = 1
    num_value_heads = 2
    key_dim = 4
    value_dim = 4
    conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    keys = jax.random.split(jax.random.PRNGKey(20260527), 6)
    conv_out = jax.random.normal(keys[0], (batch, seq_len, conv_dim), dtype=jnp.float32)
    a = jax.random.normal(keys[1], (batch, seq_len, num_value_heads), dtype=jnp.float32)
    b = jax.random.normal(keys[2], (batch, seq_len, num_value_heads), dtype=jnp.float32)
    decay = jnp.linspace(0.8, 1.2, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.1, 0.2, num_value_heads, dtype=jnp.float32)
    valid_mask = jnp.array(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=jnp.int32,
    )
    initial_state = jax.random.normal(
        keys[3],
        (batch, num_value_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01

    expected_out, expected_state = gdn_post_conv_prefill_reference_from_decay(
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
        chunk_size=8,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
    )
    query, key, value, gate, beta, seq_lens = (
        prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
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
            normalize_qk=True,
        )
    )
    actual_out, actual_state = gdn_fla_prefill_chunk32_fp32_reference(
        query,
        key,
        value,
        gate,
        beta,
        seq_lens,
        initial_state,
        chunk_size=8,
    )

    np.testing.assert_allclose(
        np.asarray(actual_out.transpose(0, 2, 1, 3)),
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


@pytest.mark.parametrize(
    ("impl", "reference_fn"),
    [
        ("reference_fla_chunk32", gdn_fla_prefill_chunk32_fp32_reference),
        ("reference_fla_packed", gdn_fla_prefill_varlen_composed_reference),
    ],
)
@pytest.mark.parametrize(
    ("output_dtype_env", "expected_output_dtype"),
    [
        (None, jnp.float32),
        ("bf16", jnp.bfloat16),
    ],
)
def test_backend_prepared_fla_bf16_qkv_keeps_fp32_state_and_default_output(
    monkeypatch,
    impl,
    reference_fn,
    output_dtype_env,
    expected_output_dtype,
):
    batch = 2
    seq_len = 8
    num_key_heads = 1
    num_value_heads = 2
    key_dim = 4
    value_dim = 4
    conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    keys = jax.random.split(jax.random.PRNGKey(20260530), 6)
    conv_out = jax.random.normal(keys[0], (batch, seq_len, conv_dim), dtype=jnp.float32)
    a = jax.random.normal(keys[1], (batch, seq_len, num_value_heads), dtype=jnp.float32)
    b = jax.random.normal(keys[2], (batch, seq_len, num_value_heads), dtype=jnp.float32)
    decay = jnp.linspace(0.8, 1.2, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.1, 0.2, num_value_heads, dtype=jnp.float32)
    valid_mask = jnp.array(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=jnp.int32,
    )
    initial_state = jax.random.normal(
        keys[3],
        (batch, num_value_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01

    query, key, value, gate, beta, seq_lens = (
        prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
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
            normalize_qk=True,
        )
    )
    expected_out, expected_state = reference_fn(
        query.astype(jnp.bfloat16),
        key.astype(jnp.bfloat16),
        value.astype(jnp.bfloat16),
        gate.astype(jnp.float32),
        beta.astype(jnp.float32),
        seq_lens,
        initial_state.astype(jnp.float32),
        chunk_size=8,
    )
    if output_dtype_env == "bf16":
        expected_out = expected_out.astype(jnp.bfloat16)

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", impl)
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PREFILL_QKV_DTYPE", "bf16")
    if output_dtype_env is None:
        monkeypatch.delenv(
            "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE",
            output_dtype_env,
        )
    actual_out, actual_state = PureJAXBackend().gated_delta_prefill_post_conv(
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
        chunk_size=8,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
    )

    assert gate.dtype == jnp.float32
    assert beta.dtype == jnp.float32
    assert initial_state.dtype == jnp.float32
    assert actual_out.dtype == expected_output_dtype
    assert actual_state.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(actual_out.transpose(0, 2, 1, 3), dtype=np.float32),
        np.asarray(expected_out, dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=0,
        atol=0,
    )


def test_prepared_fla_prefill_rejects_bf16_gate_beta_or_state():
    batch = 1
    seq_len = 4
    num_heads = 2
    key_dim = 4
    value_dim = 4
    query = jnp.ones((batch, seq_len, num_heads, key_dim), dtype=jnp.bfloat16)
    key = jnp.ones_like(query)
    value = jnp.ones((batch, seq_len, num_heads, value_dim), dtype=jnp.bfloat16)
    gate = jnp.zeros((batch, seq_len, num_heads), dtype=jnp.float32)
    beta = jnp.ones((batch, seq_len, num_heads), dtype=jnp.float32)
    seq_lens = jnp.array([seq_len], dtype=jnp.int32)
    initial_state = jnp.zeros((batch, num_heads, value_dim, key_dim), dtype=jnp.float32)

    with pytest.raises(ValueError, match="gate must be float32"):
        gdn_fla_prefill_chunk32_fp32_reference(
            query,
            key,
            value,
            gate.astype(jnp.bfloat16),
            beta,
            seq_lens,
            initial_state,
            chunk_size=4,
        )
    with pytest.raises(ValueError, match="beta must be float32"):
        gdn_fla_prefill_chunk32_fp32_reference(
            query,
            key,
            value,
            gate,
            beta.astype(jnp.bfloat16),
            seq_lens,
            initial_state,
            chunk_size=4,
        )
    with pytest.raises(ValueError, match="initial_state must be float32"):
        gdn_fla_prefill_chunk32_fp32_reference(
            query,
            key,
            value,
            gate,
            beta,
            seq_lens,
            initial_state.astype(jnp.bfloat16),
            chunk_size=4,
        )


def test_gdn_fla_prefill_kernel_boundary_bf16_qkv_fp32_accumulators():
    batch = 2
    seq_len = 4
    num_heads = 2
    key_dim = 4
    value_dim = 3
    query = jnp.ones((batch, seq_len, num_heads, key_dim), dtype=jnp.float32)
    key = jnp.ones_like(query) * 2.0
    value = jnp.ones((batch, seq_len, num_heads, value_dim), dtype=jnp.float32) * 3.0
    gate = jnp.zeros((batch, seq_len, num_heads), dtype=jnp.float32)
    beta = jnp.ones((batch, seq_len, num_heads), dtype=jnp.float32)
    seq_lens = jnp.array([seq_len, 2], dtype=jnp.int16)
    initial_state = jnp.zeros((batch, num_heads, value_dim, key_dim), dtype=jnp.float32)

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        query,
        key,
        value,
        gate,
        beta,
        seq_lens,
        initial_state,
        qkv_dtype=jnp.bfloat16,
    )

    assert prepared.query.dtype == jnp.bfloat16
    assert prepared.key.dtype == jnp.bfloat16
    assert prepared.value.dtype == jnp.bfloat16
    assert prepared.gate.dtype == jnp.float32
    assert prepared.beta.dtype == jnp.float32
    assert prepared.initial_state.dtype == jnp.float32
    assert prepared.seq_lens.dtype == jnp.int32

    output = jnp.zeros(value.shape, dtype=jnp.float32)
    final_state = jnp.zeros_like(initial_state)
    validate_gdn_fla_prefill_kernel_output(output, final_state, prepared)

    with pytest.raises(ValueError, match="output must be float32"):
        validate_gdn_fla_prefill_kernel_output(
            output.astype(jnp.bfloat16),
            final_state,
            prepared,
        )
    with pytest.raises(ValueError, match="gate must be float32"):
        prepare_gdn_fla_prefill_kernel_inputs(
            query,
            key,
            value,
            gate.astype(jnp.bfloat16),
            beta,
            seq_lens,
            initial_state,
            qkv_dtype=jnp.bfloat16,
        )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_prepared_fla_chunk32_reference_masks_padded_rows():
    batch = 2
    seq_len = 8
    num_heads = 2
    key_dim = 4
    value_dim = 4
    keys = jax.random.split(jax.random.PRNGKey(20260528), 6)
    query = jax.random.normal(keys[0], (batch, seq_len, num_heads, key_dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (batch, seq_len, num_heads, key_dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (batch, seq_len, num_heads, value_dim), dtype=jnp.float32)
    gate = jax.random.normal(keys[3], (batch, seq_len, num_heads), dtype=jnp.float32) * 0.1
    beta = jax.random.uniform(keys[4], (batch, seq_len, num_heads), dtype=jnp.float32)
    initial_state = jax.random.normal(
        keys[5],
        (batch, num_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01
    seq_lens = jnp.array([5, 8], dtype=jnp.int32)
    valid = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < seq_lens[:, None]

    clean_query = jnp.where(valid[:, :, None, None], query, 0.0)
    clean_key = jnp.where(valid[:, :, None, None], key, 0.0)
    clean_value = jnp.where(valid[:, :, None, None], value, 0.0)
    clean_gate = jnp.where(valid[:, :, None], gate, 0.0)
    clean_beta = jnp.where(valid[:, :, None], beta, 0.0)
    dirty_query = jnp.where(valid[:, :, None, None], query, 17.0)
    dirty_key = jnp.where(valid[:, :, None, None], key, -11.0)
    dirty_value = jnp.where(valid[:, :, None, None], value, 23.0)
    dirty_gate = jnp.where(valid[:, :, None], gate, 3.0)
    dirty_beta = jnp.where(valid[:, :, None], beta, 0.9)

    clean_out, clean_state = gdn_fla_prefill_chunk32_fp32_reference(
        clean_query,
        clean_key,
        clean_value,
        clean_gate,
        clean_beta,
        seq_lens,
        initial_state,
        chunk_size=8,
    )
    dirty_out, dirty_state = gdn_fla_prefill_chunk32_fp32_reference(
        dirty_query,
        dirty_key,
        dirty_value,
        dirty_gate,
        dirty_beta,
        seq_lens,
        initial_state,
        chunk_size=8,
    )

    np.testing.assert_allclose(
        np.asarray(dirty_out),
        np.asarray(clean_out),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(dirty_state),
        np.asarray(clean_state),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_prepared_fla_varlen_reference_matches_rectangular_reference(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    batch = 4
    seq_len = 32
    num_heads = 3
    key_dim = 8
    value_dim = 8
    lengths = jnp.array([0, 5, 17, 32], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(20260531), 6)
    query = jax.random.normal(keys[0], (batch, seq_len, num_heads, key_dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (batch, seq_len, num_heads, key_dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (batch, seq_len, num_heads, value_dim), dtype=jnp.float32)
    gate = jax.random.normal(keys[3], (batch, seq_len, num_heads), dtype=jnp.float32) * 0.1
    beta = jax.random.uniform(keys[4], (batch, seq_len, num_heads), dtype=jnp.float32)
    initial_state = jax.random.normal(
        keys[5],
        (batch, num_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01

    valid = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]
    query = jnp.where(valid[:, :, None, None], query, 0.0)
    key = jnp.where(valid[:, :, None, None], key, 0.0)
    value = jnp.where(valid[:, :, None, None], value, 0.0)
    gate = jnp.where(valid[:, :, None], gate, 0.0)
    beta = jnp.where(valid[:, :, None], beta, 0.0)

    (
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens,
    ) = pack_prepared_gdn_prefill_inputs(query, key, value, gate, beta, lengths)
    assert np.asarray(cu_seqlens).tolist() == [0, 0, 5, 22, 54]
    assert np.asarray(packed_query).shape == (54, num_heads, key_dim)
    assert np.asarray(packed_key).shape == (54, num_heads, key_dim)
    assert np.asarray(packed_value).shape == (54, num_heads, value_dim)
    assert np.asarray(packed_gate).shape == (54, num_heads)
    assert np.asarray(packed_beta).shape == (54, num_heads)

    expected_out, expected_state = gdn_fla_prefill_chunk32_fp32_reference(
        query,
        key,
        value,
        gate,
        beta,
        lengths,
        initial_state,
        chunk_size=8,
    )
    actual_out, actual_state = gdn_fla_prefill_varlen_reference(
        query,
        key,
        value,
        gate,
        beta,
        lengths,
        initial_state,
        chunk_size=8,
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

    unpacked = unpack_prepared_gdn_prefill_output(
        packed_value,
        cu_seqlens,
        seq_len,
    )
    np.testing.assert_allclose(
        np.asarray(unpacked),
        np.asarray(jnp.where(valid[:, :, None, None], value, 0.0)),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_backend_cuda_prepared_fla_chunk32_matches_reference(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")

    batch = 2
    seq_len = 32
    num_key_heads = 2
    num_value_heads = 2
    key_dim = 32
    value_dim = 32
    conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    keys = jax.random.split(jax.random.PRNGKey(20260529), 5)
    conv_out = jax.random.normal(keys[0], (batch, seq_len, conv_dim), dtype=jnp.float32)
    a = jax.random.normal(keys[1], (batch, seq_len, num_value_heads), dtype=jnp.float32)
    b = jax.random.normal(keys[2], (batch, seq_len, num_value_heads), dtype=jnp.float32)
    decay = jnp.linspace(0.8, 1.2, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.1, 0.2, num_value_heads, dtype=jnp.float32)
    valid_mask = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < jnp.array(
        [19, 32],
        dtype=jnp.int32,
    )[:, None]
    initial_state = jax.random.normal(
        keys[3],
        (batch, num_value_heads, value_dim, key_dim),
        dtype=jnp.float32,
    ) * 0.01
    backend = PureJAXBackend()

    monkeypatch.setenv(
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        "reference_fla_chunk32",
    )
    expected_out, expected_state = backend.gated_delta_prefill_post_conv(
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
        chunk_size=32,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
    )
    monkeypatch.setenv(
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        "cuda_fla_chunk32_fp32",
    )
    actual_out, actual_state = backend.gated_delta_prefill_post_conv(
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
        chunk_size=32,
        initial_state=initial_state,
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
def test_model_post_conv_prepared_fla_reference_matches_default_with_mask(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", raising=False)
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    config = _small_gdn_config()
    params = init_transformer_block(jax.random.PRNGKey(2), config, layer_idx=0)
    batch = 2
    seq_len = 16
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    x = jnp.linspace(
        -0.4,
        0.3,
        batch * seq_len * config.hidden_size,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, config.hidden_size)
    hybrid_state = HybridLayerState(
        conv_state=jnp.linspace(
            -0.15,
            0.25,
            batch * 1 * conv_dim * config.linear_conv_kernel_size,
            dtype=jnp.float32,
        ).reshape(batch, 1, conv_dim, config.linear_conv_kernel_size),
        recurrent_state=jnp.linspace(
            -0.025,
            0.035,
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
    lengths = jnp.array([16, 7], dtype=jnp.int32)
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

    monkeypatch.setenv(
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        "reference_fla_chunk32",
    )
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
def test_model_post_conv_packed_fla_reference_matches_default_with_mask(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", raising=False)
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    config = _small_gdn_config()
    params = init_transformer_block(jax.random.PRNGKey(4), config, layer_idx=0)
    batch = 2
    seq_len = 16
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    x = jnp.linspace(
        -0.3,
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
            0.025,
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
    lengths = jnp.array([16, 6], dtype=jnp.int32)
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

    monkeypatch.setenv(
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        "reference_fla_packed",
    )
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
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
@pytest.mark.parametrize("impl", ["triton_fla_packed", "triton_fla_padded"])
def test_model_post_conv_prepared_fla_triton_packed_matches_reference(monkeypatch, impl):
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", raising=False)
    monkeypatch.setenv("NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL", "1")

    config = _small_gdn_config()
    params = init_transformer_block(jax.random.PRNGKey(7), config, layer_idx=0)
    batch = 2
    seq_len = 12
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    x = jnp.linspace(
        -0.2,
        0.3,
        batch * seq_len * config.hidden_size,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, config.hidden_size)
    hybrid_state = HybridLayerState(
        conv_state=jnp.linspace(
            -0.05,
            0.15,
            batch * 1 * conv_dim * config.linear_conv_kernel_size,
            dtype=jnp.float32,
        ).reshape(batch, 1, conv_dim, config.linear_conv_kernel_size),
        recurrent_state=jnp.linspace(
            -0.01,
            0.02,
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
    lengths = jnp.array([12, 10], dtype=jnp.int32)
    valid_token_mask = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < lengths[:, None]
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :],
        (batch, seq_len),
    )

    monkeypatch.setenv(
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        "reference_fla_packed",
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
    monkeypatch.setenv(
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        impl,
    )
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
        rtol=0,
        atol=4e-6,
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
        rtol=0,
        atol=4e-6,
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


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_post_conv_triton_bf16_prep_matches_reference_boundary():
    from nanovllm_jax.kernels.gdn_fla_triton import gdn_post_conv_prep_bf16

    batch = 2
    seq_len = 7
    num_key_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16
    conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    conv_out = jnp.linspace(
        -0.4,
        0.5,
        batch * seq_len * conv_dim,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, conv_dim)
    a = jnp.linspace(
        -0.35,
        0.45,
        batch * seq_len * num_value_heads,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_value_heads)
    b = jnp.linspace(
        -0.5,
        0.25,
        batch * seq_len * num_value_heads,
        dtype=jnp.float32,
    ).reshape(batch, seq_len, num_value_heads)
    decay = jnp.linspace(0.8, 1.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.15, 0.2, num_value_heads, dtype=jnp.float32)
    valid_token_mask = jnp.array(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]],
        dtype=jnp.int32,
    )

    ref_query, ref_key, ref_value, ref_gate, ref_beta, _ = (
        prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
            conv_out,
            a,
            b,
            decay,
            dt_bias,
            valid_token_mask,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
            normalize_qk=True,
        )
    )
    actual_query, actual_key, actual_value, actual_gate, actual_beta = jax.jit(
        lambda conv, a_in, b_in, valid: gdn_post_conv_prep_bf16(
            conv,
            a_in,
            b_in,
            decay,
            dt_bias,
            valid,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
            normalize_qk=True,
        )
    )(conv_out, a, b, valid_token_mask)

    np.testing.assert_allclose(
        np.asarray(actual_query.astype(jnp.float32)),
        np.asarray(ref_query.astype(jnp.bfloat16).astype(jnp.float32)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_key.astype(jnp.float32)),
        np.asarray(ref_key.astype(jnp.bfloat16).astype(jnp.float32)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_value.astype(jnp.float32)),
        np.asarray(ref_value.astype(jnp.bfloat16).astype(jnp.float32)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gate),
        np.asarray(ref_gate),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_beta),
        np.asarray(ref_beta),
        rtol=2e-6,
        atol=2e-6,
    )
