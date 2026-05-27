"""vLLM/FLA-shaped Gated DeltaNet ABI references.

This module defines the planned GDN external-kernel boundary without enabling
an external kernel. The helpers are pure JAX correctness references for FP32
activation math and native V,K recurrent state.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nanovllm_jax.kernels.registry import KernelBackendUnavailable, backend_status


def availability():
    return backend_status("gdn_fla")


def require_available() -> None:
    status = availability()
    if not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)


def gdn_recurrent_decode_step(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("gdn_recurrent_decode_step FLA wrapper is not implemented yet")


def gdn_segmented_prefill_chunk32(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("gdn_segmented_prefill_chunk32 FLA wrapper is not implemented yet")


def local_gdn_state_to_k_last(state: jnp.ndarray) -> jnp.ndarray:
    """Return local recurrent state in k-last `[B,H,V,K]` layout."""

    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, value_dim, key_dim]")
    return state


def k_last_gdn_state_to_local(state: jnp.ndarray) -> jnp.ndarray:
    """Return k-last `[B,H,V,K]` recurrent state as local serving state."""

    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, value_dim, key_dim]")
    return state


def split_packed_gdn_decode_mixed_qkv(
    mixed_qkv: jnp.ndarray,
    *,
    num_q_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split vLLM-style packed decode QKV into local `[B,H,1,D]` tensors."""

    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [batch, packed_dim]")
    if num_q_heads <= 0 or num_value_heads <= 0:
        raise ValueError("head counts must be positive")
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("head dimensions must be positive")
    if num_value_heads % num_q_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_q_heads")
    query_size = num_q_heads * key_dim
    key_size = num_q_heads * key_dim
    value_size = num_value_heads * value_dim
    expected = query_size + key_size + value_size
    if mixed_qkv.shape[1] != expected:
        raise ValueError(
            "mixed_qkv last dimension must equal "
            f"2*num_q_heads*key_dim + num_value_heads*value_dim ({expected})"
        )
    query, key, value = jnp.split(mixed_qkv, (query_size, query_size + key_size), axis=-1)
    batch = mixed_qkv.shape[0]
    query = query.reshape(batch, num_q_heads, 1, key_dim)
    key = key.reshape(batch, num_q_heads, 1, key_dim)
    value = value.reshape(batch, num_value_heads, 1, value_dim)
    return query, key, value


def gdn_packed_decode_reference_local_state(
    mixed_qkv: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    state: jnp.ndarray,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX reference for vLLM-style packed GDN decode input.

    This mirrors the upstream packed decode boundary while using the local
    serving recurrent-state contract `[B,H,V,K]`.
    """

    from nanovllm_jax.model import jax_recurrent_gated_delta_rule

    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, value_dim, key_dim]")
    batch, num_value_heads, value_dim, key_dim = state.shape
    if mixed_qkv.shape[0] != batch:
        raise ValueError("mixed_qkv batch must match state batch")
    if a.shape != (batch, num_value_heads) or b.shape != (batch, num_value_heads):
        raise ValueError("a and b must have shape [batch, value_heads]")
    if a_log.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("a_log and dt_bias must have shape [value_heads]")

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

    gate = -jnp.exp(a_log[None, :]) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias[None, :])
    beta = jax.nn.sigmoid(b).astype(jnp.float32)
    output, new_state = jax_recurrent_gated_delta_rule(
        query,
        key,
        value,
        gate[:, :, None],
        beta[:, :, None],
        initial_state=state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return output, new_state
