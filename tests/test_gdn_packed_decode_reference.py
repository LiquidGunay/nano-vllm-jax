"""Focused tests for the vLLM-style packed GDN decode reference ABI."""

import os
import sys
import shutil
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    jax = None
    jnp = None
else:
    from nanovllm_jax.backends import PureJAXBackend
    from nanovllm_jax.kernels.gdn_fla import (
        availability,
        gdn_packed_decode_reference_from_decay,
        gdn_packed_decode_reference_local_state,
        gdn_recurrent_decode_step,
        k_last_gdn_state_to_local,
        local_gdn_state_to_k_last,
        split_packed_gdn_decode_mixed_qkv,
    )
    from nanovllm_jax.kernels.registry import KernelBackendUnavailable
    from nanovllm_jax.config import Qwen3_5Config
    from nanovllm_jax.kv_cache import HybridLayerState
    from nanovllm_jax.model import (
        gated_deltanet_block,
        init_transformer_block,
        jax_recurrent_gated_delta_rule,
    )

pytestmark = pytest.mark.skipif(
    jax is None,
    reason="JAX is required for GDN reference tests",
)


def _has_cuda_backend() -> bool:
    if jax is None:
        return False
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _has_nvcc() -> bool:
    if os.getenv("NANO_VLLM_JAX_NVCC"):
        return True
    return shutil.which("nvcc") is not None


def _has_jax_triton() -> bool:
    return importlib.util.find_spec("jax_triton") is not None


def test_gdn_fla_reference_module_is_fallback_only():
    status = availability()

    assert status.requested == "gdn_fla"
    assert status.selected == "pure_jax"
    assert not status.external_kernels_enabled
    with pytest.raises(KernelBackendUnavailable):
        gdn_recurrent_decode_step(None)


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_packed_gdn_decode_reference_matches_current_recurrent_path():
    batch = 2
    num_heads = 4
    key_dim = 6
    value_dim = 8
    query = jnp.linspace(
        -0.5,
        0.5,
        batch * num_heads * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, key_dim)
    key = jnp.linspace(
        0.4,
        -0.4,
        batch * num_heads * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, key_dim)
    value = jnp.linspace(
        -0.2,
        0.3,
        batch * num_heads * value_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, 1, value_dim)
    a = jnp.linspace(-0.3, 0.2, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_heads, dtype=jnp.float32).reshape(
        batch,
        num_heads,
    )
    a_log = jnp.linspace(-0.2, 0.1, num_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(0.05, 0.2, num_heads, dtype=jnp.float32)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_heads, value_dim, key_dim)

    mixed_qkv = jnp.concatenate(
        [
            query.reshape(batch, -1),
            key.reshape(batch, -1),
            value.reshape(batch, -1),
        ],
        axis=-1,
    )
    gate = -jnp.exp(a_log[None, :]) * jax.nn.softplus(a + dt_bias[None, :])
    beta = jax.nn.sigmoid(b)

    expected_out, expected_state = jax_recurrent_gated_delta_rule(
        query,
        key,
        value,
        gate[:, :, None],
        beta[:, :, None],
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = gdn_packed_decode_reference_local_state(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
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
def test_packed_gdn_decode_reference_repeats_qk_for_gva():
    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 6
    value_dim = 10
    mixed_qkv = jnp.linspace(
        -0.4,
        0.4,
        batch * (2 * num_q_heads * key_dim + num_value_heads * value_dim),
        dtype=jnp.float32,
    ).reshape(batch, -1)
    a = jnp.linspace(-0.2, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    a_log = jnp.zeros((num_value_heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((num_value_heads,), dtype=jnp.float32)
    state = jnp.zeros((batch, num_value_heads, value_dim, key_dim), dtype=jnp.float32)

    query, key, value = split_packed_gdn_decode_mixed_qkv(
        mixed_qkv,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    expected_query = jnp.repeat(query, num_value_heads // num_q_heads, axis=1)
    expected_key = jnp.repeat(key, num_value_heads // num_q_heads, axis=1)
    gate = -jnp.exp(a_log[None, :]) * jax.nn.softplus(a + dt_bias[None, :])
    beta = jax.nn.sigmoid(b)
    expected_out, expected_state = jax_recurrent_gated_delta_rule(
        expected_query,
        expected_key,
        value,
        gate[:, :, None],
        beta[:, :, None],
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = gdn_packed_decode_reference_local_state(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
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
def test_packed_gdn_decode_from_decay_matches_a_log_reference():
    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 6
    value_dim = 10
    mixed_qkv = jnp.linspace(
        -0.4,
        0.4,
        batch * (2 * num_q_heads * key_dim + num_value_heads * value_dim),
        dtype=jnp.float32,
    ).reshape(batch, -1)
    a = jnp.linspace(-0.2, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-1.0, 1.0, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    a_log = jnp.linspace(-0.2, 0.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(0.05, 0.2, num_value_heads, dtype=jnp.float32)
    state = jnp.linspace(
        -0.03,
        0.04,
        batch * num_value_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads, value_dim, key_dim)

    expected_out, expected_state = gdn_packed_decode_reference_local_state(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = gdn_packed_decode_reference_from_decay(
        mixed_qkv,
        a,
        b,
        jnp.exp(a_log),
        dt_bias,
        state,
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
def test_model_packed_gdn_decode_reference_matches_default(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", raising=False)

    config = Qwen3_5Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_size=4,
        layer_types=("linear_attention",),
        linear_attn_layers=(0,),
        dtype="float32",
    )
    params = init_transformer_block(jax.random.PRNGKey(0), config, layer_idx=0)
    batch = 2
    seq_len = 1
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
    positions = jnp.zeros((batch, seq_len), dtype=jnp.int32)

    expected_out, expected_state = gated_deltanet_block(
        x,
        params,
        positions,
        config,
        layer_idx=0,
        is_prefill=False,
        hybrid_state=hybrid_state,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "reference")
    actual_out, actual_state = gated_deltanet_block(
        x,
        params,
        positions,
        config,
        layer_idx=0,
        is_prefill=False,
        hybrid_state=hybrid_state,
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


def _packed_decode_expected(
    mixed_qkv: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    decay: jnp.ndarray,
    dt_bias: jnp.ndarray,
    state: jnp.ndarray,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch = state.shape[0]
    num_value_heads = state.shape[1]
    key_dim = state.shape[3]
    value_dim = state.shape[2]
    qk_dim = mixed_qkv.shape[1] - num_value_heads * value_dim
    if qk_dim <= 0 or qk_dim % (2 * key_dim) != 0:
        raise ValueError("mixed_qkv has an invalid packed Q/K dimension")
    num_q_heads = qk_dim // (2 * key_dim)
    query, key, value = split_packed_gdn_decode_mixed_qkv(
        mixed_qkv,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    if num_value_heads != num_q_heads:
        repeat = num_value_heads // num_q_heads
        query = jnp.repeat(query, repeat, axis=1)
        key = jnp.repeat(key, repeat, axis=1)
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    gate = -decay.astype(jnp.float32)[None, :] * jax.nn.softplus(a + dt_bias[None, :])
    beta = jax.nn.sigmoid(b).astype(jnp.float32)
    return (
        jax_recurrent_gated_delta_rule(
            query,
            key,
            value,
            gate[:, :, None],
            beta[:, :, None],
            initial_state=state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    )


def test_packed_gdn_decode_reference_bf16_mixed_qkv_keeps_fp32_output_state(monkeypatch):
    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 4
    value_dim = 5
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = jnp.linspace(
        -0.45,
        0.55,
        batch * packed_dim,
        dtype=jnp.float32,
    ).reshape(batch, packed_dim)
    a = jnp.linspace(
        -0.3,
        0.3,
        batch * num_value_heads,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads)
    b = jnp.linspace(-0.8, 0.6, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    decay = jnp.linspace(0.85, 1.15, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.2, 0.2, num_value_heads, dtype=jnp.float32)
    state = jnp.linspace(
        -0.01,
        0.02,
        batch * num_value_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads, value_dim, key_dim)

    expected_out, expected_state = _packed_decode_expected(
        mixed_qkv.astype(jnp.bfloat16),
        a,
        b,
        decay,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "reference")
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE", "bf16")
    actual_out, actual_state = PureJAXBackend().gated_delta_packed_decode(
        mixed_qkv,
        a,
        b,
        decay,
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=True,
    )

    assert actual_out.dtype == jnp.float32
    assert actual_state.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out.astype(jnp.float32)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("invalid_name", "mutate", "match"),
    [
        ("a", lambda a: a.astype(jnp.bfloat16), "a must be float32"),
        ("b", lambda b: b.astype(jnp.bfloat16), "b must be float32"),
        ("decay", lambda d: d.astype(jnp.bfloat16), "decay must be float32"),
        ("dt_bias", lambda d: d.astype(jnp.bfloat16), "dt_bias must be float32"),
        ("state", lambda s: s.astype(jnp.bfloat16), "state must be float32"),
    ],
)
def test_packed_gdn_decode_reference_bf16_path_rejects_gate_like_bf16_inputs(
    monkeypatch,
    invalid_name: str,
    mutate,
    match: str,
):
    batch = 1
    num_q_heads = 1
    num_value_heads = 2
    key_dim = 4
    value_dim = 3
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    base = jnp.linspace(-0.4, 0.4, batch * packed_dim, dtype=jnp.float32).reshape(
        batch,
        packed_dim,
    )
    a = jnp.linspace(-0.2, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-0.1, 0.1, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    decay = jnp.linspace(0.9, 1.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.2, 0.2, num_value_heads, dtype=jnp.float32)
    state = jnp.zeros((batch, num_value_heads, value_dim, key_dim), dtype=jnp.float32)

    invalid_a = mutate(a) if invalid_name == "a" else a
    invalid_b = mutate(b) if invalid_name == "b" else b
    invalid_decay = mutate(decay) if invalid_name == "decay" else decay
    invalid_dt_bias = mutate(dt_bias) if invalid_name == "dt_bias" else dt_bias
    invalid_state = mutate(state) if invalid_name == "state" else state

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "reference")
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE", "bf16")
    with pytest.raises(ValueError, match=match):
        PureJAXBackend().gated_delta_packed_decode(
            base,
            invalid_a,
            invalid_b,
            invalid_decay,
            invalid_dt_bias,
            invalid_state,
            use_qk_l2norm_in_kernel=True,
        )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.parametrize(
    ("invalid_name", "mutate", "match"),
    [
        ("a", lambda a: a.astype(jnp.bfloat16), "a must be float32"),
        ("b", lambda b: b.astype(jnp.bfloat16), "b must be float32"),
        ("decay", lambda d: d.astype(jnp.bfloat16), "decay must be float32"),
        ("dt_bias", lambda d: d.astype(jnp.bfloat16), "dt_bias must be float32"),
        ("state", lambda s: s.astype(jnp.bfloat16), "state must be float32"),
    ],
)
def test_packed_gdn_decode_cuda_bf16_path_rejects_gate_like_bf16_inputs(
    monkeypatch,
    invalid_name: str,
    mutate,
    match: str,
):
    if not _has_nvcc():
        pytest.skip("nvcc is required for CUDA packed GDN decode BF16 path")

    batch = 1
    num_q_heads = 1
    num_value_heads = 2
    key_dim = 4
    value_dim = 3
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    base = jnp.linspace(-0.4, 0.4, batch * packed_dim, dtype=jnp.float32).reshape(
        batch,
        packed_dim,
    )
    a = jnp.linspace(-0.2, 0.2, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    b = jnp.linspace(-0.1, 0.1, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    decay = jnp.linspace(0.9, 1.1, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.2, 0.2, num_value_heads, dtype=jnp.float32)
    state = jnp.zeros((batch, num_value_heads, value_dim, key_dim), dtype=jnp.float32)

    invalid_a = mutate(a) if invalid_name == "a" else a
    invalid_b = mutate(b) if invalid_name == "b" else b
    invalid_decay = mutate(decay) if invalid_name == "decay" else decay
    invalid_dt_bias = mutate(dt_bias) if invalid_name == "dt_bias" else dt_bias
    invalid_state = mutate(state) if invalid_name == "state" else state

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "cuda_fp32")
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE", "bf16")
    with pytest.raises(ValueError, match=match):
        PureJAXBackend().gated_delta_packed_decode(
            base,
            invalid_a,
            invalid_b,
            invalid_decay,
            invalid_dt_bias,
            invalid_state,
            use_qk_l2norm_in_kernel=True,
        )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_packed_gdn_decode_triton_bf16_matches_reference(monkeypatch):
    batch = 2
    num_q_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = jnp.linspace(
        -0.35,
        0.4,
        batch * packed_dim,
        dtype=jnp.float32,
    ).reshape(batch, packed_dim)
    a = jnp.linspace(
        -0.25,
        0.35,
        batch * num_value_heads,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads)
    b = jnp.linspace(-0.5, 0.4, batch * num_value_heads, dtype=jnp.float32).reshape(
        batch,
        num_value_heads,
    )
    decay = jnp.linspace(0.75, 1.2, num_value_heads, dtype=jnp.float32)
    dt_bias = jnp.linspace(-0.15, 0.2, num_value_heads, dtype=jnp.float32)
    state = jnp.linspace(
        -0.02,
        0.025,
        batch * num_value_heads * value_dim * key_dim,
        dtype=jnp.float32,
    ).reshape(batch, num_value_heads, value_dim, key_dim)

    expected_out, expected_state = gdn_packed_decode_reference_from_decay(
        mixed_qkv.astype(jnp.bfloat16),
        a,
        b,
        decay,
        dt_bias,
        state,
        qkv_dtype=jnp.bfloat16,
        use_qk_l2norm_in_kernel=True,
    )

    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "triton_fla")
    backend = PureJAXBackend()
    actual_out, actual_state = jax.jit(
        lambda mixed, a_in, b_in, decay_in, dt_bias_in, state_in: (
            backend.gated_delta_packed_decode(
                mixed,
                a_in,
                b_in,
                decay_in,
                dt_bias_in,
                state_in,
                use_qk_l2norm_in_kernel=True,
            )
        )
    )(
        mixed_qkv,
        a,
        b,
        decay,
        dt_bias,
        state,
    )

    assert actual_out.dtype == jnp.float32
    assert actual_state.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual_out), np.asarray(expected_out), atol=5e-7)
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        atol=5e-7,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
def test_gdn_state_k_last_roundtrip_preserves_local_layout():
    state = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5)

    k_last = local_gdn_state_to_k_last(state)
    local = k_last_gdn_state_to_local(k_last)

    assert k_last.shape == state.shape
    np.testing.assert_array_equal(np.asarray(local), np.asarray(state))
