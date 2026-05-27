"""vLLM/FLA-shaped Gated DeltaNet ABI references.

This module defines the planned GDN external-kernel boundary without enabling
an external kernel. The helpers are pure JAX correctness references for FP32
activation math and native V,K recurrent state.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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

    if a_log.ndim != 1:
        raise ValueError("a_log must have shape [value_heads]")
    return gdn_packed_decode_reference_from_decay(
        mixed_qkv,
        a,
        b,
        jnp.exp(a_log.astype(jnp.float32)),
        dt_bias,
        state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def gdn_packed_decode_reference_from_decay(
    mixed_qkv: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    decay: jnp.ndarray,
    dt_bias: jnp.ndarray,
    state: jnp.ndarray,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX packed GDN decode reference using local positive decay `A`.

    Local loaded weights store `A = exp(A_log)` to preserve the established
    FP32 activation contract. Production-shaped decode call sites can use this
    helper without adding a hot-path parameter-layout dependency on checkpoint
    naming.
    """

    from nanovllm_jax.model import jax_recurrent_gated_delta_rule

    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, value_dim, key_dim]")
    batch, num_value_heads, value_dim, key_dim = state.shape
    if mixed_qkv.shape[0] != batch:
        raise ValueError("mixed_qkv batch must match state batch")
    if a.shape != (batch, num_value_heads) or b.shape != (batch, num_value_heads):
        raise ValueError("a and b must have shape [batch, value_heads]")
    if decay.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("decay and dt_bias must have shape [value_heads]")

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

    gate = -decay.astype(jnp.float32)[None, :] * jax.nn.softplus(a.astype(jnp.float32) + dt_bias[None, :])
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


def prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
    conv_out: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    decay: jnp.ndarray,
    dt_bias: jnp.ndarray,
    valid_token_mask: jnp.ndarray | None,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    normalize_qk: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare post-conv GDN prefill tensors in FLA `[B,T,H,D]` layout.

    The returned query/key tensors have value-head count after GQA repeat. This
    is the stable ABI for a future vLLM/FLA-derived FP32 fast body; callers can
    transpose to the legacy local `[B,H,T,D]` chunk reference as a fallback.
    """

    if conv_out.ndim != 3:
        raise ValueError("conv_out must have shape [batch, time, conv_dim]")
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must have shape [batch, time, value_heads]")
    if num_key_heads <= 0 or num_value_heads <= 0:
        raise ValueError("head counts must be positive")
    if key_head_dim <= 0 or value_head_dim <= 0:
        raise ValueError("head dimensions must be positive")
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")

    batch, seq_len, conv_dim = conv_out.shape
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    expected_conv_dim = 2 * key_dim + value_dim
    if conv_dim != expected_conv_dim:
        raise ValueError(
            f"conv_out last dimension must be {expected_conv_dim}, got {conv_dim}"
        )
    if a.shape != (batch, seq_len, num_value_heads):
        raise ValueError("a must have shape [batch, time, value_heads]")
    if b.shape != (batch, seq_len, num_value_heads):
        raise ValueError("b must have shape [batch, time, value_heads]")
    if decay.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("decay and dt_bias must have shape [value_heads]")
    if valid_token_mask is not None and valid_token_mask.shape != (batch, seq_len):
        raise ValueError("valid_token_mask must have shape [batch, time]")

    query = conv_out[:, :, :key_dim].reshape(
        batch,
        seq_len,
        num_key_heads,
        key_head_dim,
    )
    key = conv_out[:, :, key_dim : key_dim * 2].reshape(
        batch,
        seq_len,
        num_key_heads,
        key_head_dim,
    )
    value = conv_out[:, :, key_dim * 2 :].reshape(
        batch,
        seq_len,
        num_value_heads,
        value_head_dim,
    )
    beta = jax.nn.sigmoid(b)
    gate = -decay * jax.nn.softplus(a + dt_bias)

    if valid_token_mask is None:
        seq_lens = jnp.full((batch,), seq_len, dtype=jnp.int32)
    else:
        valid = valid_token_mask.astype(jnp.bool_)
        query = jnp.where(valid[:, :, None, None], query, 0.0)
        key = jnp.where(valid[:, :, None, None], key, 0.0)
        value = jnp.where(valid[:, :, None, None], value, 0.0)
        beta = jnp.where(valid[:, :, None], beta, 0.0)
        gate = jnp.where(valid[:, :, None], gate, 0.0)
        seq_lens = valid.astype(jnp.int32).sum(axis=1)

    heads_per_key = num_value_heads // num_key_heads
    if heads_per_key > 1:
        query = jnp.repeat(query, heads_per_key, axis=2)
        key = jnp.repeat(key, heads_per_key, axis=2)

    if normalize_qk:
        from nanovllm_jax.layers import l2norm

        query = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6)
        key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)

    return query, key, value, gate, beta, seq_lens


def gdn_fla_prefill_chunk32_fp32_reference(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    gate: jnp.ndarray,
    beta: jnp.ndarray,
    seq_lens: jnp.ndarray,
    initial_state: jnp.ndarray,
    *,
    chunk_size: int = 32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX reference for the prepared FLA-layout FP32 chunk body.

    Inputs are rectangular and already in the future-kernel layout:
    q/k/v `[B,T,H,D]`, gate/beta `[B,T,H]`, and state `[B,H,V,K]`. Query/key
    normalization, if desired, is owned by the prep helper; this body only
    applies the chunk rule's query scale once.
    """

    from nanovllm_jax.model import jax_chunk_gated_delta_rule

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, time, heads, dim]")
    if gate.ndim != 3 or beta.ndim != 3:
        raise ValueError("gate and beta must have shape [batch, time, heads]")
    if key.shape != query.shape:
        raise ValueError("query and key shapes must match")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value must match query [batch, time, heads]")

    batch, seq_len, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if gate.shape != (batch, seq_len, num_heads):
        raise ValueError("gate must have shape [batch, time, heads]")
    if beta.shape != (batch, seq_len, num_heads):
        raise ValueError("beta must have shape [batch, time, heads]")
    if seq_lens.shape != (batch,):
        raise ValueError("seq_lens must have shape [batch]")
    if initial_state.shape != (batch, num_heads, value_dim, key_dim):
        raise ValueError("initial_state must have shape [batch, heads, value_dim, key_dim]")

    valid = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < seq_lens.astype(jnp.int32)[:, None]
    query = jnp.where(valid[:, :, None, None], query.astype(jnp.float32), 0.0)
    key = jnp.where(valid[:, :, None, None], key.astype(jnp.float32), 0.0)
    value = jnp.where(valid[:, :, None, None], value.astype(jnp.float32), 0.0)
    gate = jnp.where(valid[:, :, None], gate.astype(jnp.float32), 0.0)
    beta = jnp.where(valid[:, :, None], beta.astype(jnp.float32), 0.0)

    output, final_state = jax_chunk_gated_delta_rule(
        query.transpose(0, 2, 1, 3),
        key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3),
        gate.transpose(0, 2, 1),
        beta.transpose(0, 2, 1),
        chunk_size=chunk_size,
        initial_state=initial_state.astype(jnp.float32),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return output.transpose(0, 2, 1, 3), final_state


def gdn_post_conv_prefill_reference_from_decay(
    conv_out: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    decay: jnp.ndarray,
    dt_bias: jnp.ndarray,
    valid_token_mask: jnp.ndarray | None,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    chunk_size: int,
    initial_state: jnp.ndarray | None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX reference for the GDN post-convolution prefill boundary.

    This mirrors vLLM/FLA's useful `fused_post_conv_prep` boundary while keeping
    the current FP32 math contract and native local state layout `[B,H,V,K]`.
    It owns split, gate construction, valid-token masking, GQA repeat, layout
    packing, and the final call into the existing chunked reference.
    """

    from nanovllm_jax.model import jax_chunk_gated_delta_rule

    query, key, value, gate, beta, _ = prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
        conv_out,
        a,
        b,
        decay,
        dt_bias,
        valid_token_mask,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        normalize_qk=False,
    )
    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    gate = gate.transpose(0, 2, 1)
    beta = beta.transpose(0, 2, 1)
    return jax_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


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


def prepare_gdn_fla_chunk_metadata(
    cu_seqlens: Any,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build FLA-style varlen chunk indices and per-row chunk offsets.

    `chunk_indices` has rows `[sequence_index, chunk_index_in_sequence]`, one
    per active chunk. `chunk_offsets` has shape `[batch + 1]` and maps original
    sequence rows to their first active chunk. Unlike the upstream helper, this
    preserves original row ids when bucket padding introduces zero-length rows.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")

    lengths = offsets[1:] - offsets[:-1]
    chunks_per_row = (lengths + int(chunk_size) - 1) // int(chunk_size)
    chunk_offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int32), np.cumsum(chunks_per_row, dtype=np.int32)]
    )
    if int(chunk_offsets[-1]) == 0:
        chunk_indices = np.zeros((0, 2), dtype=np.int32)
    else:
        rows = np.repeat(np.arange(len(lengths), dtype=np.int32), chunks_per_row)
        chunks = np.concatenate(
            [
                np.arange(int(num_chunks), dtype=np.int32)
                for num_chunks in chunks_per_row
                if int(num_chunks) > 0
            ]
        )
        chunk_indices = np.stack([rows, chunks], axis=1).astype(np.int32)
    return (
        jnp.asarray(chunk_indices, dtype=jnp.int32),
        jnp.asarray(chunk_offsets, dtype=jnp.int32),
    )


def gdn_fla_chunk_local_cumsum_packed_reference(
    gate: jnp.ndarray,
    cu_seqlens: Any,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
    reverse: bool = False,
) -> jnp.ndarray:
    """Reference for FLA scalar `chunk_local_cumsum` over packed `[nnz,H]` gate."""

    if gate.ndim != 2:
        raise ValueError("gate must have shape [nnz_tokens, heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != gate.shape[0]:
        raise ValueError("last cu_seqlens entry must equal gate token count")
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size)
    chunk_index_values = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    if chunk_index_values.ndim != 2 or chunk_index_values.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    output = jnp.zeros(gate.shape, dtype=jnp.float32)
    for row, chunk in chunk_index_values:
        if row < 0 or row + 1 >= len(offsets):
            raise ValueError("chunk_indices row is out of range")
        chunk_start = int(offsets[row]) + int(chunk) * int(chunk_size)
        chunk_end = min(int(offsets[row + 1]), chunk_start + int(chunk_size))
        if chunk_start < int(offsets[row]) or chunk_start >= int(offsets[row + 1]):
            raise ValueError("chunk_indices chunk is out of range for row")
        chunk_gate = gate[chunk_start:chunk_end].astype(jnp.float32)
        if reverse:
            chunk_cumsum = jnp.flip(
                jnp.cumsum(jnp.flip(chunk_gate, axis=0), axis=0),
                axis=0,
            )
        else:
            chunk_cumsum = jnp.cumsum(chunk_gate, axis=0)
        output = output.at[chunk_start:chunk_end].set(chunk_cumsum)
    return output


def gdn_fla_chunk_scaled_dot_kkt_packed_reference(
    key: jnp.ndarray,
    beta: jnp.ndarray,
    gate_cumsum: jnp.ndarray | None,
    cu_seqlens: Any,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
) -> jnp.ndarray:
    """Reference for FLA `chunk_scaled_dot_kkt` over packed varlen tensors."""

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if beta.ndim != 2:
        raise ValueError("beta must have shape [nnz_tokens, output_heads]")
    if key.shape[0] != beta.shape[0]:
        raise ValueError("key and beta token counts must match")
    if gate_cumsum is not None and gate_cumsum.shape != beta.shape:
        raise ValueError("gate_cumsum must have the same shape as beta")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    key_heads = key.shape[1]
    output_heads = beta.shape[1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != key.shape[0]:
        raise ValueError("last cu_seqlens entry must equal key token count")
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size)
    chunk_index_values = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    if chunk_index_values.ndim != 2 or chunk_index_values.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    head_group = output_heads // key_heads
    output = jnp.zeros((key.shape[0], output_heads, chunk_size), dtype=jnp.float32)
    for row, chunk in chunk_index_values:
        if row < 0 or row + 1 >= len(offsets):
            raise ValueError("chunk_indices row is out of range")
        chunk_start = int(offsets[row]) + int(chunk) * int(chunk_size)
        chunk_end = min(int(offsets[row + 1]), chunk_start + int(chunk_size))
        if chunk_start < int(offsets[row]) or chunk_start >= int(offsets[row + 1]):
            raise ValueError("chunk_indices chunk is out of range for row")
        length = chunk_end - chunk_start
        local_mask = jnp.tril(jnp.ones((length, length), dtype=jnp.float32), k=-1)
        for head in range(output_heads):
            key_head = head // head_group
            chunk_key = key[chunk_start:chunk_end, key_head, :].astype(jnp.float32)
            chunk_beta = beta[chunk_start:chunk_end, head].astype(jnp.float32)
            chunk_matrix = (chunk_key * chunk_beta[:, None]) @ chunk_key.T
            if gate_cumsum is not None:
                chunk_gate = gate_cumsum[chunk_start:chunk_end, head].astype(jnp.float32)
                chunk_matrix = chunk_matrix * jnp.exp(
                    chunk_gate[:, None] - chunk_gate[None, :]
                )
            chunk_matrix = chunk_matrix * local_mask
            output = output.at[chunk_start:chunk_end, head, :length].set(chunk_matrix)
    return output


def gdn_fla_solve_tril_packed_reference(
    attention_matrix: jnp.ndarray,
    cu_seqlens: Any,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
) -> jnp.ndarray:
    """Reference for FLA `solve_tril` over packed varlen chunk matrices."""

    if attention_matrix.ndim != 3:
        raise ValueError("attention_matrix must have shape [nnz_tokens, heads, chunk_size]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if attention_matrix.shape[-1] != chunk_size:
        raise ValueError("attention_matrix last dimension must equal chunk_size")
    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != attention_matrix.shape[0]:
        raise ValueError("last cu_seqlens entry must equal token count")
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size)
    chunk_index_values = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    if chunk_index_values.ndim != 2 or chunk_index_values.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    output = jnp.zeros(attention_matrix.shape, dtype=jnp.float32)
    for row, chunk in chunk_index_values:
        if row < 0 or row + 1 >= len(offsets):
            raise ValueError("chunk_indices row is out of range")
        chunk_start = int(offsets[row]) + int(chunk) * int(chunk_size)
        chunk_end = min(int(offsets[row + 1]), chunk_start + int(chunk_size))
        if chunk_start < int(offsets[row]) or chunk_start >= int(offsets[row + 1]):
            raise ValueError("chunk_indices chunk is out of range for row")
        length = chunk_end - chunk_start
        identity = jnp.eye(length, dtype=jnp.float32)
        for head in range(attention_matrix.shape[1]):
            matrix = attention_matrix[chunk_start:chunk_end, head, :length].astype(
                jnp.float32
            )
            inverse = jnp.linalg.inv(identity + matrix)
            output = output.at[chunk_start:chunk_end, head, :length].set(inverse)
    return output


def gdn_fla_recompute_w_u_packed_reference(
    key: jnp.ndarray,
    value: jnp.ndarray,
    beta: jnp.ndarray,
    gate_cumsum: jnp.ndarray,
    attention_inverse: jnp.ndarray,
    cu_seqlens: Any,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reference for FLA `recompute_w_u_fwd` over packed varlen tensors."""

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if value.ndim != 3:
        raise ValueError("value must have shape [nnz_tokens, output_heads, value_dim]")
    if beta.ndim != 2:
        raise ValueError("beta must have shape [nnz_tokens, output_heads]")
    if gate_cumsum.shape != beta.shape:
        raise ValueError("gate_cumsum must have the same shape as beta")
    if attention_inverse.ndim != 3:
        raise ValueError(
            "attention_inverse must have shape [nnz_tokens, output_heads, chunk_size]"
        )
    if key.shape[0] != value.shape[0] or key.shape[0] != beta.shape[0]:
        raise ValueError("key, value, and beta token counts must match")
    if value.shape[:2] != beta.shape:
        raise ValueError("value and beta must agree on token and output heads")
    if attention_inverse.shape != (key.shape[0], beta.shape[1], chunk_size):
        raise ValueError("attention_inverse shape must match [tokens, output_heads, chunk_size]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    key_heads = key.shape[1]
    output_heads = beta.shape[1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != key.shape[0]:
        raise ValueError("last cu_seqlens entry must equal token count")
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size)
    chunk_index_values = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    if chunk_index_values.ndim != 2 or chunk_index_values.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    head_group = output_heads // key_heads
    w = jnp.zeros((key.shape[0], output_heads, key.shape[-1]), dtype=jnp.float32)
    u = jnp.zeros(value.shape, dtype=jnp.float32)
    for row, chunk in chunk_index_values:
        if row < 0 or row + 1 >= len(offsets):
            raise ValueError("chunk_indices row is out of range")
        chunk_start = int(offsets[row]) + int(chunk) * int(chunk_size)
        chunk_end = min(int(offsets[row + 1]), chunk_start + int(chunk_size))
        if chunk_start < int(offsets[row]) or chunk_start >= int(offsets[row + 1]):
            raise ValueError("chunk_indices chunk is out of range for row")
        length = chunk_end - chunk_start
        for head in range(output_heads):
            key_head = head // head_group
            matrix = attention_inverse[chunk_start:chunk_end, head, :length].astype(
                jnp.float32
            )
            chunk_beta = beta[chunk_start:chunk_end, head].astype(jnp.float32)
            chunk_gate = gate_cumsum[chunk_start:chunk_end, head].astype(jnp.float32)
            chunk_value = value[chunk_start:chunk_end, head, :].astype(jnp.float32)
            chunk_key = key[chunk_start:chunk_end, key_head, :].astype(jnp.float32)
            u_chunk = matrix @ (chunk_value * chunk_beta[:, None])
            w_chunk = matrix @ (
                chunk_key * chunk_beta[:, None] * jnp.exp(chunk_gate)[:, None]
            )
            u = u.at[chunk_start:chunk_end, head, :].set(u_chunk)
            w = w.at[chunk_start:chunk_end, head, :].set(w_chunk)
    return w, u


def gdn_fla_chunk_delta_h_packed_reference(
    key: jnp.ndarray,
    w: jnp.ndarray,
    u: jnp.ndarray,
    gate_cumsum: jnp.ndarray | None,
    cu_seqlens: Any,
    initial_state: jnp.ndarray | None,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
    chunk_offsets: Any | None = None,
    output_final_state: bool = True,
    save_new_value: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Reference for FLA `chunk_gated_delta_rule_fwd_h` over packed varlen tensors."""

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if w.ndim != 3:
        raise ValueError("w must have shape [nnz_tokens, output_heads, key_dim]")
    if u.ndim != 3:
        raise ValueError("u must have shape [nnz_tokens, output_heads, value_dim]")
    if key.shape[0] != w.shape[0] or key.shape[0] != u.shape[0]:
        raise ValueError("key, w, and u token counts must match")
    if w.shape[:2] != u.shape[:2]:
        raise ValueError("w and u must agree on token and output heads")
    if w.shape[-1] != key.shape[-1]:
        raise ValueError("w key dimension must match key")
    if gate_cumsum is not None and gate_cumsum.shape != w.shape[:2]:
        raise ValueError("gate_cumsum must have shape [nnz_tokens, output_heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    key_heads = key.shape[1]
    output_heads = w.shape[1]
    key_dim = key.shape[-1]
    value_dim = u.shape[-1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != key.shape[0]:
        raise ValueError("last cu_seqlens entry must equal token count")
    if chunk_indices is None or chunk_offsets is None:
        prepared_indices, prepared_offsets = prepare_gdn_fla_chunk_metadata(
            cu_seqlens,
            chunk_size,
        )
        if chunk_indices is None:
            chunk_indices = prepared_indices
        if chunk_offsets is None:
            chunk_offsets = prepared_offsets
    chunk_index_values = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    chunk_offset_values = np.asarray(jax.device_get(chunk_offsets), dtype=np.int64).reshape(-1)
    if chunk_index_values.ndim != 2 or chunk_index_values.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")
    if chunk_offset_values.shape != (len(offsets),):
        raise ValueError("chunk_offsets must have shape [batch + 1]")
    if int(chunk_offset_values[-1]) != len(chunk_index_values):
        raise ValueError("last chunk_offsets entry must equal number of chunks")

    batch = len(offsets) - 1
    if initial_state is None:
        state = jnp.zeros((batch, output_heads, value_dim, key_dim), dtype=jnp.float32)
    else:
        if initial_state.shape != (batch, output_heads, value_dim, key_dim):
            raise ValueError("initial_state must have shape [batch, output_heads, value_dim, key_dim]")
        state = initial_state.astype(jnp.float32)

    head_group = output_heads // key_heads
    h = jnp.zeros((len(chunk_index_values), output_heads, value_dim, key_dim), dtype=jnp.float32)
    v_new = (
        jnp.zeros((key.shape[0], output_heads, value_dim), dtype=jnp.float32)
        if save_new_value
        else None
    )
    for row in range(batch):
        row_state = state[row]
        row_start = int(offsets[row])
        row_end = int(offsets[row + 1])
        for chunk in range(int(chunk_offset_values[row + 1] - chunk_offset_values[row])):
            flat_chunk = int(chunk_offset_values[row]) + chunk
            if tuple(chunk_index_values[flat_chunk]) != (row, chunk):
                raise ValueError("chunk_indices and chunk_offsets disagree")
            chunk_start = row_start + chunk * int(chunk_size)
            chunk_end = min(row_end, chunk_start + int(chunk_size))
            if chunk_start < row_start or chunk_start >= row_end:
                raise ValueError("chunk_offsets imply an out-of-range chunk")
            length = chunk_end - chunk_start
            for head in range(output_heads):
                key_head = head // head_group
                head_state = row_state[head]
                h = h.at[flat_chunk, head].set(head_state)
                chunk_w = w[chunk_start:chunk_end, head, :].astype(jnp.float32)
                chunk_u = u[chunk_start:chunk_end, head, :].astype(jnp.float32)
                delta = chunk_u - chunk_w @ head_state.T
                if v_new is not None:
                    v_new = v_new.at[chunk_start:chunk_end, head, :].set(delta)
                update_delta = delta
                if gate_cumsum is not None:
                    chunk_gate = gate_cumsum[chunk_start:chunk_end, head].astype(
                        jnp.float32
                    )
                    last_gate = chunk_gate[length - 1]
                    update_delta = delta * jnp.exp(last_gate - chunk_gate)[:, None]
                    head_state = head_state * jnp.exp(last_gate)
                chunk_key = key[chunk_start:chunk_end, key_head, :].astype(jnp.float32)
                head_state = head_state + update_delta.T @ chunk_key
                row_state = row_state.at[head].set(head_state)
        state = state.at[row].set(row_state)

    final_state = state if output_final_state else None
    return h, v_new, final_state


def gdn_fla_chunk_fwd_o_packed_reference(
    query: jnp.ndarray,
    key: jnp.ndarray,
    v_new: jnp.ndarray,
    h: jnp.ndarray,
    gate_cumsum: jnp.ndarray | None,
    cu_seqlens: Any,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
    scale: float | None = None,
) -> jnp.ndarray:
    """Reference for FLA `chunk_fwd_o` over packed varlen tensors."""

    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [nnz_tokens, key_heads, key_dim]")
    if v_new.ndim != 3:
        raise ValueError("v_new must have shape [nnz_tokens, output_heads, value_dim]")
    if h.ndim != 4:
        raise ValueError("h must have shape [num_chunks, output_heads, value_dim, key_dim]")
    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")
    if query.shape[0] != v_new.shape[0]:
        raise ValueError("query and v_new token counts must match")
    if h.shape[1] != v_new.shape[1]:
        raise ValueError("h and v_new output heads must match")
    if h.shape[2] != v_new.shape[2] or h.shape[3] != query.shape[2]:
        raise ValueError("h value/key dimensions must match v_new/query")
    if gate_cumsum is not None and gate_cumsum.shape != v_new.shape[:2]:
        raise ValueError("gate_cumsum must have shape [nnz_tokens, output_heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    key_heads = query.shape[1]
    output_heads = v_new.shape[1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")
    if scale is None:
        scale = float(query.shape[-1] ** -0.5)

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != query.shape[0]:
        raise ValueError("last cu_seqlens entry must equal token count")
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu_seqlens, chunk_size)
    chunk_index_values = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    if chunk_index_values.ndim != 2 or chunk_index_values.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")
    if h.shape[0] != len(chunk_index_values):
        raise ValueError("h first dimension must equal number of chunks")

    head_group = output_heads // key_heads
    output = jnp.zeros(v_new.shape, dtype=jnp.float32)
    for flat_chunk, (row, chunk) in enumerate(chunk_index_values):
        if row < 0 or row + 1 >= len(offsets):
            raise ValueError("chunk_indices row is out of range")
        chunk_start = int(offsets[row]) + int(chunk) * int(chunk_size)
        chunk_end = min(int(offsets[row + 1]), chunk_start + int(chunk_size))
        if chunk_start < int(offsets[row]) or chunk_start >= int(offsets[row + 1]):
            raise ValueError("chunk_indices chunk is out of range for row")
        length = chunk_end - chunk_start
        causal = jnp.tril(jnp.ones((length, length), dtype=jnp.float32), k=0)
        for head in range(output_heads):
            key_head = head // head_group
            chunk_query = query[chunk_start:chunk_end, key_head, :].astype(jnp.float32)
            chunk_key = key[chunk_start:chunk_end, key_head, :].astype(jnp.float32)
            chunk_v_new = v_new[chunk_start:chunk_end, head, :].astype(jnp.float32)
            state = h[flat_chunk, head].astype(jnp.float32)
            state_out = chunk_query @ state.T
            attention = chunk_query @ chunk_key.T
            if gate_cumsum is not None:
                chunk_gate = gate_cumsum[chunk_start:chunk_end, head].astype(
                    jnp.float32
                )
                state_out = state_out * jnp.exp(chunk_gate)[:, None]
                attention = attention * jnp.exp(
                    chunk_gate[:, None] - chunk_gate[None, :]
                )
            attention = attention * causal
            chunk_output = (state_out + attention @ chunk_v_new) * float(scale)
            output = output.at[chunk_start:chunk_end, head, :].set(chunk_output)
    return output


def pack_padded_gdn_inputs(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    seq_lens: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pack current `[B,H,T,D]` GDN tensors into `[nnz,H,D]` ABI tensors."""

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


def pack_prepared_gdn_prefill_inputs(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    gate: jnp.ndarray,
    beta: jnp.ndarray,
    seq_lens: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pack FLA-layout `[B,T,H,D]` prefill tensors into `[nnz,H,D]` ABI tensors."""

    lengths = _seq_lens_to_tuple(seq_lens)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, time, heads, dim]")
    if gate.ndim != 3 or beta.ndim != 3:
        raise ValueError("gate and beta must have shape [batch, time, heads]")
    batch, seq_len, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if len(lengths) != batch:
        raise ValueError("seq_lens must have one entry per batch row")
    if key.shape != query.shape:
        raise ValueError("query and key shapes must match")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value must match query [batch, time, heads]")
    if gate.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("gate and beta must have shape [batch, time, heads]")
    if any(length > seq_len for length in lengths):
        raise ValueError("seq_lens entries must be <= padded sequence length")

    q_parts = []
    k_parts = []
    v_parts = []
    gate_parts = []
    beta_parts = []
    for row, length in enumerate(lengths):
        if length == 0:
            continue
        q_parts.append(query[row, :length])
        k_parts.append(key[row, :length])
        v_parts.append(value[row, :length])
        gate_parts.append(gate[row, :length])
        beta_parts.append(beta[row, :length])

    if q_parts:
        packed_query = jnp.concatenate(q_parts, axis=0)
        packed_key = jnp.concatenate(k_parts, axis=0)
        packed_value = jnp.concatenate(v_parts, axis=0)
        packed_gate = jnp.concatenate(gate_parts, axis=0)
        packed_beta = jnp.concatenate(beta_parts, axis=0)
    else:
        packed_query = jnp.zeros((0, num_heads, key_dim), dtype=query.dtype)
        packed_key = jnp.zeros((0, num_heads, key_dim), dtype=key.dtype)
        packed_value = jnp.zeros((0, num_heads, value_dim), dtype=value.dtype)
        packed_gate = jnp.zeros((0, num_heads), dtype=gate.dtype)
        packed_beta = jnp.zeros((0, num_heads), dtype=beta.dtype)
    return (
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        cu_seqlens_from_seq_lens(lengths),
    )


def unpack_prepared_gdn_prefill_output(
    packed_output: jnp.ndarray,
    cu_seqlens: Any,
    max_seq_len: int,
) -> jnp.ndarray:
    """Unpack `[nnz,H,V]` segmented output into FLA layout `[B,T,H,V]`."""

    legacy = unpack_segmented_gdn_output(packed_output, cu_seqlens, max_seq_len)
    return jnp.transpose(legacy, (0, 2, 1, 3))


def gdn_fla_prefill_varlen_reference(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    gate: jnp.ndarray,
    beta: jnp.ndarray,
    seq_lens: jnp.ndarray,
    initial_state: jnp.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX reference for vLLM/FLA varlen prefill tensors.

    Inputs stay in the upstream-facing FLA layout: q/k/v `[B,T,H,D]`,
    gate/beta `[B,T,H]`, `seq_lens [B]`, and V,K state `[B,H,V,K]`. The
    reference packs valid tokens into `[nnz,H,D]`, calls the segmented ABI
    reference, then unpacks the valid outputs back to `[B,T,H,V]`.
    """

    packed_query, packed_key, packed_value, packed_gate, packed_beta, cu_seqlens = (
        pack_prepared_gdn_prefill_inputs(query, key, value, gate, beta, seq_lens)
    )
    packed_output, final_state = gdn_segmented_prefill_chunk32_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_beta,
        packed_gate,
        cu_seqlens,
        initial_state,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=False,
    )
    return (
        unpack_prepared_gdn_prefill_output(packed_output, cu_seqlens, query.shape[1]),
        final_state,
    )


def unpack_segmented_gdn_output(
    packed_output: jnp.ndarray,
    cu_seqlens: Any,
    max_seq_len: int,
) -> jnp.ndarray:
    """Unpack `[nnz,H,V]` segmented output into `[B,H,T,V]` layout."""

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
    reference_seq_len: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX reference for the planned packed segmented GDN prefill ABI."""

    from nanovllm_jax.model import jax_chunk_gated_delta_rule

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
        row_len = end - start
        if reference_seq_len is not None and reference_seq_len < row_len:
            raise ValueError("reference_seq_len must be >= every packed row length")
        run_len = row_len if reference_seq_len is None else int(reference_seq_len)
        if run_len == row_len:
            q_segment = query[start:end]
            k_segment = key[start:end]
            v_segment = value[start:end]
            g_segment = gate[start:end]
            beta_segment = beta[start:end]
        else:
            q_segment = jnp.zeros((run_len, query.shape[1], query.shape[2]), dtype=query.dtype)
            k_segment = jnp.zeros((run_len, key.shape[1], key.shape[2]), dtype=key.dtype)
            v_segment = jnp.zeros((run_len, value.shape[1], value.shape[2]), dtype=value.dtype)
            g_segment = jnp.zeros((run_len, gate.shape[1]), dtype=gate.dtype)
            beta_segment = jnp.zeros((run_len, beta.shape[1]), dtype=beta.dtype)
            q_segment = q_segment.at[:row_len].set(query[start:end])
            k_segment = k_segment.at[:row_len].set(key[start:end])
            v_segment = v_segment.at[:row_len].set(value[start:end])
            g_segment = g_segment.at[:row_len].set(gate[start:end])
            beta_segment = beta_segment.at[:row_len].set(beta[start:end])

        q_row = jnp.transpose(q_segment, (1, 0, 2))[None, ...]
        k_row = jnp.transpose(k_segment, (1, 0, 2))[None, ...]
        v_row = jnp.transpose(v_segment, (1, 0, 2))[None, ...]
        g_row = jnp.transpose(g_segment, (1, 0))[None, ...]
        beta_row = jnp.transpose(beta_segment, (1, 0))[None, ...]
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
        output_parts.append(jnp.transpose(out_row[0, :, :row_len, :], (1, 0, 2)))
        final_states.append(state_row[0])

    if output_parts:
        packed_output = jnp.concatenate(output_parts, axis=0)
    else:
        packed_output = jnp.zeros((0, query.shape[1], value_dim), dtype=value.dtype)
    return packed_output, jnp.stack(final_states, axis=0)
