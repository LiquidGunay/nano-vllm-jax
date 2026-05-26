"""CUDA Gated DeltaNet kernel placeholders and segmented reference helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.kernels.registry import KernelBackendUnavailable, backend_status
from nanovllm_jax.model import jax_chunk_gated_delta_rule


def availability():
    return backend_status("gdn_cuda")


def require_available() -> None:
    status = availability()
    if not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)


def gdn_recurrent_decode_step(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("gdn_recurrent_decode_step CUDA wrapper is not implemented yet")


def gdn_segmented_prefill_chunk32(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("gdn_segmented_prefill_chunk32 CUDA wrapper is not implemented yet")


def _seq_lens_to_tuple(seq_lens: Any) -> tuple[int, ...]:
    values = np.asarray(jax.device_get(seq_lens), dtype=np.int64).reshape(-1)
    if np.any(values < 0):
        raise ValueError("seq_lens entries must be non-negative")
    return tuple(int(value) for value in values)


def cu_seqlens_from_seq_lens(seq_lens: Any) -> jnp.ndarray:
    """Build FlashAttention-style cumulative sequence lengths."""

    lengths = _seq_lens_to_tuple(seq_lens)
    offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int32), np.cumsum(lengths, dtype=np.int32)]
    )
    return jnp.asarray(offsets, dtype=jnp.int32)


def pack_padded_gdn_inputs(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    seq_lens: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pack current `[B, H, T, D]` GDN tensors into `[nnz, H, D]` ABI tensors."""

    lengths = _seq_lens_to_tuple(seq_lens)
    batch, num_heads, seq_len, key_dim = query.shape
    value_dim = value.shape[-1]
    if len(lengths) != batch:
        raise ValueError("seq_lens must have one entry per batch row")
    if key.shape != query.shape:
        raise ValueError("query and key shapes must match")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value must match query [batch, heads, time]")
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("g and beta must have shape [batch, heads, time]")
    if any(length > seq_len for length in lengths):
        raise ValueError("seq_lens entries must be <= padded sequence length")

    q_parts = []
    k_parts = []
    v_parts = []
    g_parts = []
    beta_parts = []
    for row, length in enumerate(lengths):
        if length == 0:
            continue
        q_parts.append(jnp.transpose(query[row, :, :length, :], (1, 0, 2)))
        k_parts.append(jnp.transpose(key[row, :, :length, :], (1, 0, 2)))
        v_parts.append(jnp.transpose(value[row, :, :length, :], (1, 0, 2)))
        g_parts.append(jnp.transpose(g[row, :, :length], (1, 0)))
        beta_parts.append(jnp.transpose(beta[row, :, :length], (1, 0)))

    if q_parts:
        packed_query = jnp.concatenate(q_parts, axis=0)
        packed_key = jnp.concatenate(k_parts, axis=0)
        packed_value = jnp.concatenate(v_parts, axis=0)
        packed_g = jnp.concatenate(g_parts, axis=0)
        packed_beta = jnp.concatenate(beta_parts, axis=0)
    else:
        packed_query = jnp.zeros((0, num_heads, key_dim), dtype=query.dtype)
        packed_key = jnp.zeros((0, num_heads, key_dim), dtype=key.dtype)
        packed_value = jnp.zeros((0, num_heads, value_dim), dtype=value.dtype)
        packed_g = jnp.zeros((0, num_heads), dtype=g.dtype)
        packed_beta = jnp.zeros((0, num_heads), dtype=beta.dtype)
    return (
        packed_query,
        packed_key,
        packed_value,
        packed_g,
        packed_beta,
        cu_seqlens_from_seq_lens(lengths),
    )


def unpack_segmented_gdn_output(
    packed_output: jnp.ndarray,
    cu_seqlens: Any,
    max_seq_len: int,
) -> jnp.ndarray:
    """Unpack `[nnz, H, V]` segmented output into `[B, H, T, V]` layout."""

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0:
        raise ValueError("cu_seqlens must contain at least one offset")
    batch = len(offsets) - 1
    num_heads = packed_output.shape[1]
    value_dim = packed_output.shape[2]
    output = jnp.zeros(
        (batch, num_heads, max_seq_len, value_dim),
        dtype=packed_output.dtype,
    )
    for row in range(batch):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        length = end - start
        if length == 0:
            continue
        output = output.at[row, :, :length, :].set(
            jnp.transpose(packed_output[start:end], (1, 0, 2))
        )
    return output


def gdn_segmented_prefill_chunk32_reference(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    beta: jnp.ndarray,
    gate: jnp.ndarray,
    cu_seqlens: Any,
    initial_state: jnp.ndarray,
    *,
    chunk_size: int = 32,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX reference for the planned packed segmented GDN prefill ABI.

    Inputs use the planned compact layout: `query/key/value` are `[nnz, H, D]`,
    `beta/gate` are `[nnz, H]`, and `cu_seqlens` maps packed tokens to rows.
    This helper is intentionally not a speed path; it is the correctness oracle
    for future CUDA/ported kernels.
    """

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != query.shape[0]:
        raise ValueError("last cu_seqlens entry must equal nnz token count")
    if key.shape != query.shape:
        raise ValueError("query and key shapes must match")
    if value.shape[:2] != query.shape[:2]:
        raise ValueError("value must match query [nnz, heads]")
    if beta.shape != query.shape[:2] or gate.shape != query.shape[:2]:
        raise ValueError("beta and gate must have shape [nnz, heads]")

    batch = len(offsets) - 1
    if initial_state.shape[:2] != (batch, query.shape[1]):
        raise ValueError("initial_state batch/head dimensions must match cu_seqlens/query")

    output_parts = []
    final_states = []
    value_dim = value.shape[-1]
    for row in range(batch):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        if end == start:
            final_states.append(initial_state[row])
            continue
        q_row = jnp.transpose(query[start:end], (1, 0, 2))[None, ...]
        k_row = jnp.transpose(key[start:end], (1, 0, 2))[None, ...]
        v_row = jnp.transpose(value[start:end], (1, 0, 2))[None, ...]
        g_row = jnp.transpose(gate[start:end], (1, 0))[None, ...]
        beta_row = jnp.transpose(beta[start:end], (1, 0))[None, ...]
        out_row, state_row = jax_chunk_gated_delta_rule(
            q_row,
            k_row,
            v_row,
            g_row,
            beta_row,
            chunk_size=chunk_size,
            initial_state=initial_state[row : row + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        output_parts.append(jnp.transpose(out_row[0], (1, 0, 2)))
        final_states.append(state_row[0])

    if output_parts:
        packed_output = jnp.concatenate(output_parts, axis=0)
    else:
        packed_output = jnp.zeros((0, query.shape[1], value_dim), dtype=value.dtype)
    return packed_output, jnp.stack(final_states, axis=0)
