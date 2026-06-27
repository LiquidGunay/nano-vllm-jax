"""Gated DeltaNet reference and promoted serving paths."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax, nn

from nanovllm_jax.cache import HybridLayerState
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import causal_conv1d_update, l2norm
from nanovllm_jax.ops import (
    ServingOps,
    ServingOpsProtocol,
    gdn_disable_fallbacks_enabled,
    gdn_packed_decode_enabled,
    gdn_packed_decode_impl,
    gdn_packed_decode_max_batch,
    gdn_prefill_post_conv_enabled,
)
from nanovllm_jax.projection import (
    _GDN_DECODE_IN_PROJ_PACKED_KEY,
    _causal_conv1d,
    _compact_prefill_dot_if_enabled,
    _compact_prefill_tokenwise_dot,
    _decode_padded_gemm_dot,
    _decode_projection_activation_dtype,
    _enable_chunked_gdn_prefill,
    _enable_compact_prefill_gdn_z,
    _force_width1_decode_math,
    _packed_causal_conv1d_prefill,
    _stable_rmsnorm_fp32,
    _tokenwise_decode_dot,
    _use_gdn_decode_packed_in_proj,
    _use_gdn_prefill_packed_in_proj,
    _can_use_decode_padded_gemm,
)

def jax_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                output_final_state=False, use_qk_l2norm_in_kernel=False):
    """
    JAX implementation of chunk gated delta rule (matching HF torch_chunk_gated_delta_rule).
    Input shapes: [B, H, T, D] for query/key/value, [B, H, T] for g/beta
    Output shape: [B, H, T, D]
    """
    if query.shape[2] > chunk_size and not _enable_chunked_gdn_prefill():
        # The multi-chunk JAX chunk kernel still has measurable drift from the
        # HF/PyTorch chunked reference. Use the recurrent reference path for
        # correctness; the chunk kernel can be restored behind parity tests.
        output, final_state = jax_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return output, final_state if output_final_state else None

    import jax

    initial_dtype = query.dtype

    # Apply L2 norm if requested
    if use_qk_l2norm_in_kernel:
        query = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6)
        key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)

    # Convert to float32 for computation
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    g = g.astype(jnp.float32)

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Pad to chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

    total_seq_len = seq_len + pad_size
    scale = 1.0 / jnp.sqrt(k_head_dim)
    query = query * scale

    # v_beta = value * beta[..., None]
    v_beta = value * beta[..., None]
    # k_beta = key * beta[..., None]
    k_beta = key * beta[..., None]

    # Reshape to chunks: [B, H, n_chunks, chunk_size, D]
    def reshape_to_chunks(x):
        return x.reshape(batch_size, num_heads, -1, chunk_size, x.shape[-1])

    query_chunks = reshape_to_chunks(query)
    key_chunks = reshape_to_chunks(key)
    value_chunks = reshape_to_chunks(value)
    k_beta_chunks = reshape_to_chunks(k_beta)
    v_beta_chunks = reshape_to_chunks(v_beta)

    # g reshaped: [B, H, n_chunks, chunk_size]
    g_chunks = g.reshape(batch_size, num_heads, -1, chunk_size)
    n_chunks = g_chunks.shape[2]

    # Create mask for upper triangle (within chunk)
    mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    mask_strict_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)

    # Compute decay: cumulative sum of g within each chunk.
    g_cumsum = jnp.cumsum(g_chunks, axis=-1)

    # decay_mask[b,h,n,i,j] = exp(g_cumsum[b,h,n,i] - g_cumsum[b,h,n,j]) for i >= j, else 0
    decay_mask = jnp.tril(jnp.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :]))

    # Compute k_beta @ key.transpose(-1, -2) for each chunk
    # k_beta_chunks: [B, H, n, cs, K], key_chunks: [B, H, n, cs, K]
    # Result: [B, H, n, cs, cs] where result[b,h,n,i,j] = sum_k(k_beta[b,h,n,i,k] * key[b,h,n,j,k])
    kkt = jnp.einsum('bhnck,bhnjk->bhncj', k_beta_chunks, key_chunks)

    # attn = -((k_beta @ k.T) * decay_mask), masked to lower triangle
    attn = -(kkt * decay_mask)
    attn = jnp.where(mask_upper, 0.0, attn)  # Zero out diagonal and upper triangle

    # Recursive computation within chunk (matching HF exactly)
    # for i in range(1, chunk_size):
    #     row = attn[..., i, :i].clone()  # [B, H, n, i]
    #     sub = attn[..., :i, :i].clone()  # [B, H, n, i, i]
    #     attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    def update_row(carry, i):
        attn_carry = carry

        # Create a mask that's True for indices < i
        mask = jnp.arange(chunk_size) < i  # [cs]

        # Extract row i, cols 0:i -> shape [B, H, n, cs] but only first i entries are valid
        row_i = attn_carry[..., i, :] * mask[None, None, None, :]  # [B, H, n, cs]

        # Extract submatrix [0:i, 0:i] -> shape [B, H, n, cs, cs] but only [0:i, 0:i] is valid
        # Multiply by mask on both dimensions
        sub_i = attn_carry * mask[None, None, None, :, None] * mask[None, None, None, None, :]  # [B, H, n, cs, cs]

        # HF: (row.unsqueeze(-1) * sub).sum(-2)
        # This computes: contribution[k] = sum_j(row[j] * sub[j, k]) for j,k in 0:i
        # Using einsum: 'bhnj,bhnjk->bhnk'
        contribution = jnp.einsum('bhnj,bhnjk->bhnk', row_i, sub_i)  # [B, H, n, cs]

        # new_row for cols 0:i
        new_row = row_i + contribution
        # Mask to keep only cols 0:i
        new_row = new_row * mask[None, None, None, :]

        # Update attn at row i
        attn_carry = attn_carry.at[..., i, :].set(new_row)
        return attn_carry, i

    attn, _ = lax.scan(update_row, attn, jnp.arange(1, chunk_size))

    # Add identity matrix
    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)[None, None, None, :, :]

    # value_transformed = attn @ v_beta
    # attn: [B, H, n, cs, cs], v_beta: [B, H, n, cs, V]
    # result: [B, H, n, cs, V]
    value_transformed = jnp.einsum('bhnct,bhntv->bhncv', attn, v_beta_chunks)

    # Initial-state correction uses decay from the start of each chunk.
    k_cumdecay = jnp.einsum('bhnct,bhntv->bhncv', attn, k_beta_chunks * jnp.exp(g_cumsum)[..., None])

    # Initialize state [B, H, V, K].
    if initial_state is None:
        state = jnp.zeros((batch_size, num_heads, v_head_dim, k_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    # Process each chunk sequentially
    def process_chunk(carry, i):
        state = carry
        q_i = query_chunks[:, :, i]      # [B, H, cs, K]
        k_i = key_chunks[:, :, i]        # [B, H, cs, K]
        v_i = value_transformed[:, :, i] # [B, H, cs, V]
        decay_mask_i = decay_mask[:, :, i]  # [B, H, cs, cs]
        k_cumdecay_i = k_cumdecay[:, :, i]  # [B, H, cs, K]
        g_cumsum_i = g_cumsum[:, :, i]    # [B, H, cs] - use cumsum version!

        # Within-chunk attention: attn = (q @ k.T * decay_mask), masked to strict upper triangle
        attn_i = jnp.einsum('bhck,bhdk->bhcd', q_i, k_i) * decay_mask_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)

        # v_prime = k_cumdecay @ state
        # k_cumdecay_i: [B, H, cs, K], state: [B, H, V, K]
        # v_prime[b,h,c,v] = sum_k(k_cumdecay_i[b,h,c,k] * state[b,h,v,k])
        v_prime = jnp.einsum('bhck,bhvk->bhcv', k_cumdecay_i, state)

        # v_new = v_i - v_prime
        v_new = v_i - v_prime

        # attn_inter = (q * exp(g)) @ state
        # q_i * exp(g_cumsum_i): [B, H, cs, K]
        # result: [B, H, cs, V] = sum_K(q[b,h,c,k] * exp(g_cumsum[b,h,c]) * state[b,h,v,k])
        attn_inter = jnp.einsum('bhck,bhvk->bhcv', q_i * jnp.exp(g_cumsum_i)[..., None], state)

        # core_attn_out = attn_inter + attn @ v_new
        # attn_i: [B, H, cs, cs], v_new: [B, H, cs, V]
        # result: [B, H, cs, V] = sum_d(attn_i[b,h,c,d] * v_new[b,h,d,v])
        attn_v_new = jnp.einsum('bhcd,bhdv->bhcv', attn_i, v_new)
        core_attn_out_i = attn_inter + attn_v_new

        # Update state
        # state = state * exp(g_cumsum[b,h,cs-1]) + (k * exp(g_cumsum[cs-1] - g_cumsum)).T @ v_new
        # g_last_minus_g[b,h,c] = g_cumsum[b,h,-1] - g_cumsum[b,h,c]
        g_last_minus_g = g_cumsum_i[..., -1, None] - g_cumsum_i  # [B, H, cs]
        # k_weighted[b,h,c,k] = k_i[b,h,c,k] * exp(g_last_minus_g[b,h,c])
        k_weighted = k_i * jnp.exp(g_last_minus_g)[..., None]  # [B, H, cs, K]
        # state_update[b,h,v,k] = sum_c(v_new[b,h,c,v] * k_weighted[b,h,c,k])
        state_update = jnp.einsum('bhcv,bhck->bhvk', v_new, k_weighted)
        state = state * jnp.exp(g_cumsum_i[..., -1, None, None]) + state_update

        return state, core_attn_out_i

    final_state, core_attn_out_chunks = lax.scan(process_chunk, state, jnp.arange(n_chunks))

    # lax.scan returns [n_chunks, B, H, chunk, V]; HF keeps chunk as the
    # third dimension [B, H, n_chunks, chunk, V] before flattening time.
    core_attn_out = core_attn_out_chunks.transpose(1, 2, 0, 3, 4).reshape(
        batch_size,
        num_heads,
        -1,
        v_head_dim,
    )
    core_attn_out = core_attn_out[:, :, :seq_len]  # Remove padding

    if not output_final_state:
        final_state = None

    core_attn_out = core_attn_out.astype(initial_dtype)
    return core_attn_out, final_state


def jax_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state=None,
    use_qk_l2norm_in_kernel=False,
    return_state_sequence: bool = False,
    return_first_state: bool = False,
):
    """
    JAX implementation of recurrent gated delta rule (matching HF torch_recurrent_gated_delta_rule).
    Input shapes: [B, H, T, D] for query/key/value, [B, H, T] for g/beta
    Output shape: [B, H, T, D]
    State shape: [B, H, v_head_dim, k_head_dim] (kernel-native V,K layout)
    """
    initial_dtype = query.dtype

    # Keep as [B, H, T, D]. For cached decode, normalize q/k in
    # float32 before l2norm so width-2 recurrent scans match the width-1
    # sequential decode path as closely as possible in bf16.
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    value = value.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    batch, num_heads, time_dim, k_head_dim = query.shape
    v_head_dim = value.shape[-1]

    # Scale query
    query = query * (1.0 / jnp.sqrt(k_head_dim))

    # Initialize state: [B, H, V, K] so decode uses the same V,K layout as
    # the planned external GDN kernels.
    if initial_state is None:
        state = jnp.zeros((batch, num_heads, v_head_dim, k_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    def step_one(state, q_t, k_t, v_t, g_t_raw, beta_t):
        g_t = jnp.exp(g_t_raw)  # [B, H]

        # Reshape for broadcasting
        g_t_exp = g_t[:, :, None, None]    # [B, H, 1, 1]
        beta_t_exp = beta_t[:, :, None]    # [B, H, 1]
        k_t_exp = k_t[:, :, None, :]       # [B, H, 1, K]

        # Decay state: state * exp(g_t)
        state = state * g_t_exp

        # kv_mem = (state * k_t[..., None]).sum(-2)
        kv_mem = jnp.einsum('bhvk,bhk->bhv', state, k_t)

        # delta = (v_t - kv_mem) * beta_t
        delta = (v_t - kv_mem) * beta_t_exp  # [B, H, V]

        # state = state + delta[..., None] * k_t[:, :, None, :]
        state = state + delta[:, :, :, None] * k_t_exp

        # output_t = (state * q_t[:, :, None, :]).sum(-1)
        out_t = jnp.einsum('bhvk,bhk->bhv', state, q_t)

        return state, out_t

    def step_index(state, t):
        return step_one(
            state,
            query[:, :, t, :],
            key[:, :, t, :],
            value[:, :, t, :],
            g[:, :, t],
            beta[:, :, t],
        )

    if time_dim == 2:
        # Cached K=1 route uses width-2 decode. Avoid a two-step scan so the
        # first token is computed as the same explicit width-1 recurrence used by
        # the sequential commit-select reference, then apply the second token.
        state_0, out_0 = step_one(
            state,
            query[:, :, 0, :],
            key[:, :, 0, :],
            value[:, :, 0, :],
            g[:, :, 0],
            beta[:, :, 0],
        )
        state_1, out_1 = step_one(
            state_0,
            query[:, :, 1, :],
            key[:, :, 1, :],
            value[:, :, 1, :],
            g[:, :, 1],
            beta[:, :, 1],
        )
        output = jnp.stack([out_0, out_1], axis=2).astype(initial_dtype)
        if return_state_sequence:
            state_sequence = jnp.stack([state_0, state_1], axis=1)
            return output, state_1, state_sequence
        if return_first_state:
            return output, state_1, state_0
        return output, state_1

    if return_state_sequence:
        def step_fn(carry, t):
            next_state, out_t = step_index(carry, t)
            return next_state, (out_t, next_state)

        final_state, (all_outputs, all_states) = lax.scan(step_fn, state, jnp.arange(time_dim))
        output = all_outputs.transpose(1, 2, 0, 3)
        state_sequence = all_states.transpose(1, 0, 2, 3, 4)
        output = output.astype(initial_dtype)
        return output, final_state, state_sequence

    if return_first_state:
        def step_fn(carry, t):
            next_state, out_t = step_index(carry, t)
            return next_state, (out_t, next_state)

        final_state, (all_outputs, all_states) = lax.scan(step_fn, state, jnp.arange(time_dim))
        output = all_outputs.transpose(1, 2, 0, 3)
        first_state = all_states[0]
        output = output.astype(initial_dtype)
        return output, final_state, first_state

    def step_fn(carry, t):
        return step_index(carry, t)

    final_state, all_outputs = lax.scan(step_fn, state, jnp.arange(time_dim))

    # all_outputs: [T, B, H, V] -> transpose to [B, H, T, V]
    output = all_outputs.transpose(1, 2, 0, 3)

    output = output.astype(initial_dtype)
    return output, final_state


def gated_deltanet_block(
    x,
    params,
    positions,
    config,
    layer_idx: int,
    is_prefill: bool = True,
    hybrid_state: Optional[HybridLayerState] = None,
    valid_token_mask: Optional[jnp.ndarray] = None,
    compact_prefill_tokens: Optional[int] = None,
    backend: Optional[ServingOpsProtocol] = None,
    return_prefix_state: bool = False,
    return_first_prefix_state: bool = False,
    hybrid_state_is_layer: bool = False,
    packed_token_row_ids: Optional[jnp.ndarray] = None,
    packed_query_start_loc: Optional[jnp.ndarray] = None,
):
    """Gated DeltaNet block with decode mode support.

    Args:
        x: Input [batch, seq_len, hidden]
        params: Layer parameters
        positions: Position IDs
        config: Model config
        layer_idx: Layer index (0-based)
        is_prefill: Whether this is prefill (True) or decode (False)
        hybrid_state: Optional linear-attention state for this batch

    Returns:
        tuple: (output, updated_hybrid_state) or just output for prefill
    """
    batch, seq_len, _ = x.shape
    if backend is None:
        backend = ServingOps(config=config)
    prefix_layer_state = None

    # Cast to target dtype for the promoted CUDA/JAX path.
    dtype = config.get_dtype()
    x_cast = x.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )

    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    v_heads_per_k = config.linear_num_value_heads // config.linear_num_key_heads
    conv_dim = key_dim * 2 + value_dim

    # Check if we can use cached states
    use_cached = (
        not is_prefill and
        hybrid_state is not None and
        hybrid_state.conv_state is not None and
        hybrid_state.recurrent_state is not None and
        seq_len <= 2
    )
    use_cached_prefill = (
        is_prefill
        and hybrid_state is not None
        and hybrid_state.conv_state is not None
        and hybrid_state.recurrent_state is not None
    )
    use_recurrent_prefill = (
        use_cached_prefill
        and (
            seq_len <= int(getattr(config, "linear_recurrent_prefill_threshold", 8))
            or not _enable_chunked_gdn_prefill()
        )
    )
    linear_layer_idx = len([l for l in config.linear_attn_layers if l < layer_idx])
    row_valid = None

    # === PROJECTIONS (same for both modes) ===
    force_width1_dot = (not is_prefill) and seq_len > 1 and _force_width1_decode_math()
    use_packed_decode_in_proj = _use_gdn_decode_packed_in_proj(
        params,
        is_prefill=is_prefill,
        batch=batch,
        seq_len=seq_len,
        config=config,
    )
    use_packed_prefill_in_proj = _use_gdn_prefill_packed_in_proj(
        params,
        is_prefill=is_prefill,
        config=config,
    )
    packed_decode_projection = None
    if use_packed_decode_in_proj or use_packed_prefill_in_proj:
        if is_prefill:
            packed_proj = _compact_prefill_dot_if_enabled(
                x_cast,
                params[_GDN_DECODE_IN_PROJ_PACKED_KEY],
                valid_token_mask,
                compact_prefill_tokens,
                enabled=True,
            )
        else:
            packed_proj = _tokenwise_decode_dot(
                x_cast,
                params[_GDN_DECODE_IN_PROJ_PACKED_KEY],
                force_width1=force_width1_dot,
            )
        qkv_end = conv_dim
        a_end = qkv_end + config.linear_num_value_heads
        b_end = a_end + config.linear_num_value_heads
        if use_packed_decode_in_proj and not is_prefill:
            packed_decode_projection = packed_proj
            mixed_qkv = packed_proj[:, :, :qkv_end]
            a = None
            b = None
            z = packed_proj[:, :, b_end:].reshape(batch, seq_len, -1)
        else:
            mixed_qkv, a, b, z = jnp.split(packed_proj, [qkv_end, a_end, b_end], axis=-1)
            z = z.reshape(batch, seq_len, -1)
            a = a.reshape(batch, seq_len, config.linear_num_value_heads)
            b = b.reshape(batch, seq_len, config.linear_num_value_heads)
    else:
        if is_prefill:
            mixed_qkv = _compact_prefill_tokenwise_dot(
                x_cast,
                params["in_proj_qkv"],
                valid_token_mask,
                compact_prefill_tokens,
                config,
            )
        else:
            if _can_use_decode_padded_gemm(x_cast, params["in_proj_qkv"], config):
                mixed_qkv = _decode_padded_gemm_dot(x_cast, params["in_proj_qkv"], config)
            else:
                mixed_qkv = _tokenwise_decode_dot(
                    x_cast,
                    params["in_proj_qkv"],
                    force_width1=force_width1_dot,
                )
        if is_prefill:
            z = _compact_prefill_dot_if_enabled(
                x_cast,
                params["in_proj_z"],
                valid_token_mask,
                compact_prefill_tokens,
                enabled=_enable_compact_prefill_gdn_z(config),
            ).reshape(batch, seq_len, -1)
        else:
            z = _tokenwise_decode_dot(x_cast, params["in_proj_z"], force_width1=force_width1_dot).reshape(batch, seq_len, -1)
        a = _tokenwise_decode_dot(
            x_cast,
            params["in_proj_a"],
            force_width1=force_width1_dot,
        ).reshape(batch, seq_len, config.linear_num_value_heads)
        b = _tokenwise_decode_dot(
            x_cast,
            params["in_proj_b"],
            force_width1=force_width1_dot,
        ).reshape(batch, seq_len, config.linear_num_value_heads)
    if use_cached:
        layer_conv_state = (
            hybrid_state.conv_state
            if hybrid_state_is_layer
            else hybrid_state.conv_state[:, linear_layer_idx]
        )
        conv_weight = params["conv1d_weight"].reshape(conv_dim, config.linear_conv_kernel_size)
        conv_bias = params.get("conv1d_bias")
        initial_recurrent = (
            hybrid_state.recurrent_state
            if hybrid_state_is_layer
            else hybrid_state.recurrent_state[:, linear_layer_idx]
        )

        packed_decode_max_batch = gdn_packed_decode_max_batch(config)
        packed_decode_requested = gdn_packed_decode_enabled(config)
        use_packed_decode = (
            packed_decode_requested
            and (packed_decode_max_batch is None or batch <= packed_decode_max_batch)
            and seq_len == 1
            and not return_prefix_state
            and not return_first_prefix_state
            and initial_recurrent is not None
        )
        use_packed_multitoken_decode = (
            packed_decode_requested
            and seq_len > 1
            and initial_recurrent is not None
        )
        if (
            packed_decode_requested
            and not use_packed_decode
            and not use_packed_multitoken_decode
            and gdn_disable_fallbacks_enabled(config)
        ):
            reasons = []
            if packed_decode_max_batch is not None and batch > packed_decode_max_batch:
                reasons.append(f"batch {batch} exceeds packed_decode.max_batch {packed_decode_max_batch}")
            if seq_len != 1:
                reasons.append(f"sequence width {seq_len} is not width-1 decode")
            if return_prefix_state:
                reasons.append("return_prefix_state needs state-sequence output")
            if return_first_prefix_state:
                reasons.append("return_first_prefix_state needs prefix-state output")
            if initial_recurrent is None:
                reasons.append("initial recurrent state is missing")
            raise RuntimeError(
                "GDN packed decode fallback is disabled, but "
                f"{gdn_packed_decode_impl(config)!r} cannot run for this decode batch: "
                + "; ".join(reasons or ["the packed decode predicate was false"])
            )

        if a is None or b is None:
            a = packed_decode_projection[:, :, qkv_end:a_end].reshape(batch, seq_len, config.linear_num_value_heads)
            b = packed_decode_projection[:, :, a_end:b_end].reshape(batch, seq_len, config.linear_num_value_heads)

        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)

        def conv_step(state, mixed_qkv_t_step):
            conv_out_t, next_state = causal_conv1d_update(
                mixed_qkv_t_step,
                state,
                conv_weight,
                conv_bias,
                "silu",
            )
            return next_state, conv_out_t

        if seq_len > 1:
            state = layer_conv_state
            conv_out_parts = []
            conv_state_parts = []
            for token_idx in range(seq_len):
                state, conv_out_t = conv_step(state, mixed_qkv_t[:, :, token_idx : token_idx + 1])
                conv_out_parts.append(conv_out_t)
                if return_prefix_state or (return_first_prefix_state and token_idx == 0):
                    conv_state_parts.append(state)
            new_layer_conv_state = state
            conv_out_steps = jnp.stack(conv_out_parts, axis=0)
            if return_prefix_state:
                prefix_layer_conv_state = jnp.stack(conv_state_parts, axis=0).transpose(1, 0, 2, 3)
            elif return_first_prefix_state:
                prefix_layer_conv_state = conv_state_parts[0] if conv_state_parts else state
            else:
                prefix_layer_conv_state = None
        else:
            new_layer_conv_state, conv_out_t = conv_step(layer_conv_state, mixed_qkv_t[:, :, :1])
            conv_out_steps = conv_out_t[None, ...]
            prefix_layer_conv_state = (
                new_layer_conv_state[:, None, :, :]
                if return_prefix_state
                else new_layer_conv_state if return_first_prefix_state else None
            )
        conv_out = conv_out_steps.transpose(1, 0, 2, 3).reshape(batch, seq_len, conv_dim)

        if use_packed_multitoken_decode:
            recurrent_state = initial_recurrent.astype(jnp.float32)
            output_parts = []
            recurrent_state_parts = []
            for token_idx in range(seq_len):
                out_t, recurrent_state = backend.gated_delta_packed_decode(
                    conv_out[:, token_idx, :].astype(jnp.float32),
                    a[:, token_idx, :].astype(jnp.float32),
                    b[:, token_idx, :].astype(jnp.float32),
                    params["A"].astype(jnp.float32),
                    params["dt_bias"].astype(jnp.float32),
                    recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                )
                output_parts.append(out_t)
                if return_prefix_state or (return_first_prefix_state and token_idx == 0):
                    recurrent_state_parts.append(recurrent_state)
            core_attn_out = jnp.concatenate(output_parts, axis=2)
            new_recurrent_state_single = recurrent_state
            if return_prefix_state:
                prefix_recurrent_state_single = jnp.stack(recurrent_state_parts, axis=1)
            elif return_first_prefix_state:
                prefix_recurrent_state_single = recurrent_state_parts[0]
            else:
                prefix_recurrent_state_single = None
        elif use_packed_decode:
            core_attn_out, new_recurrent_state_single = backend.gated_delta_packed_decode(
                conv_out[:, 0, :].astype(jnp.float32),
                a[:, 0, :].astype(jnp.float32),
                b[:, 0, :].astype(jnp.float32),
                params["A"].astype(jnp.float32),
                params["dt_bias"].astype(jnp.float32),
                initial_recurrent.astype(jnp.float32),
                use_qk_l2norm_in_kernel=True,
            )
            prefix_recurrent_state_single = None
        else:
            query = conv_out[:, :, :key_dim].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            key = conv_out[:, :, key_dim:key_dim * 2].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            value = conv_out[:, :, key_dim * 2:].reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)
            beta = nn.sigmoid(b)
            g = -params["A"] * nn.softplus(a + params["dt_bias"])
            if v_heads_per_k > 1:
                query = jnp.repeat(query, v_heads_per_k, axis=2)
                key = jnp.repeat(key, v_heads_per_k, axis=2)
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            g = g.transpose(0, 2, 1)
            beta = beta.transpose(0, 2, 1)
            if return_prefix_state:
                core_attn_out, new_recurrent_state_single, prefix_recurrent_state_single = jax_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                    return_state_sequence=True,
                )
            elif return_first_prefix_state:
                core_attn_out, new_recurrent_state_single, prefix_recurrent_state_single = jax_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                    return_first_state=True,
                )
            elif seq_len > 1:
                core_attn_out, new_recurrent_state_single = jax_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                )
                prefix_recurrent_state_single = None
            else:
                core_attn_out, new_recurrent_state_single = backend.gated_delta_decode(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                )
                prefix_recurrent_state_single = None

        if valid_token_mask is not None:
            if row_valid is None:
                row_valid = valid_token_mask.astype(jnp.int32).sum(axis=1) > 0
            conv_keep = row_valid[:, None, None]
            recurrent_keep = row_valid[:, None, None, None]
            new_layer_conv_state = jnp.where(conv_keep, new_layer_conv_state, layer_conv_state)
            new_recurrent_state_single = jnp.where(
                recurrent_keep,
                new_recurrent_state_single,
                initial_recurrent,
            )
            if return_first_prefix_state and prefix_recurrent_state_single is not None:
                prefix_recurrent_state_single = jnp.where(
                    recurrent_keep,
                    prefix_recurrent_state_single,
                    initial_recurrent,
                )
            if return_first_prefix_state and prefix_layer_conv_state is not None:
                prefix_layer_conv_state = jnp.where(conv_keep, prefix_layer_conv_state, layer_conv_state)

        if hybrid_state_is_layer:
            new_recurrent_state = new_recurrent_state_single
            new_conv_state = new_layer_conv_state
        else:
            new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(new_recurrent_state_single)
            new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(new_layer_conv_state)
        hybrid_state = replace(
            hybrid_state,
            conv_state=new_conv_state,
            recurrent_state=new_recurrent_state,
        )
        prefix_layer_state = (
            HybridLayerState(
                conv_state=prefix_layer_conv_state,
                recurrent_state=prefix_recurrent_state_single,
            )
            if return_prefix_state or return_first_prefix_state
            else None
        )
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)

    else:
        # === PREFILL MODE ===
        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # [B, D, T]
        if packed_token_row_ids is not None:
            if packed_query_start_loc is None:
                raise ValueError("packed_query_start_loc is required for packed GDN prefill")
            if batch != 1:
                raise ValueError("packed GDN prefill expects token tensors shaped [1, token_bucket, ...]")

            row_count = int(packed_query_start_loc.shape[0]) - 1
            token_rows = packed_token_row_ids.reshape(-1).astype(jnp.int32)
            valid_tokens = jnp.arange(seq_len, dtype=jnp.int32) < packed_query_start_loc[-1].astype(jnp.int32)
            safe_rows = jnp.clip(token_rows, 0, row_count - 1)
            max_row_tokens = (
                max(tuple(getattr(config, "prefill_buckets", ()) or ()))
                if tuple(getattr(config, "prefill_buckets", ()) or ())
                else seq_len
            )
            row_query_len = (
                max(1, seq_len // row_count)
                if return_prefix_state or return_first_prefix_state
                else (
                    seq_len
                    if max_row_tokens is None
                    else min(seq_len, int(max_row_tokens))
                )
            )
            row_query_len = max(1, row_query_len)
            row_offsets = jnp.arange(row_query_len, dtype=jnp.int32)
            row_starts = packed_query_start_loc[:-1].astype(jnp.int32)
            row_lens = (
                packed_query_start_loc[1:] - packed_query_start_loc[:-1]
            ).astype(jnp.int32)
            token_indices_by_row = row_starts[:, None] + row_offsets[None, :]
            valid_queries_by_row = row_offsets[None, :] < row_lens[:, None]
            safe_token_indices_by_row = jnp.clip(
                token_indices_by_row,
                0,
                seq_len - 1,
            )
            packed_prefix_state_post_conv = False
            use_packed_post_conv_prefill = (
                gdn_prefill_post_conv_enabled(config)
                and (not use_recurrent_prefill or packed_prefix_state_post_conv)
                and (not return_prefix_state or packed_prefix_state_post_conv)
                and not return_first_prefix_state
            )
            if (
                gdn_disable_fallbacks_enabled(config)
                and not use_packed_post_conv_prefill
            ):
                reasons = []
                if not gdn_prefill_post_conv_enabled(config):
                    reasons.append("packed post-conv prefill kernel is disabled")
                if use_recurrent_prefill and not packed_prefix_state_post_conv:
                    reasons.append("recurrent prefill is requested")
                if return_prefix_state and not packed_prefix_state_post_conv:
                    reasons.append("return_prefix_state needs a kernel-backed tiny prefix output")
                if return_first_prefix_state:
                    reasons.append("return_first_prefix_state needs prefix-state output")
                if not reasons:
                    reasons.append("the packed post-conv prefill predicate was false")
                raise RuntimeError(
                    "GDN packed prefill post-conv fallback is disabled, but the "
                    "requested packed prefill route would use the slow JAX "
                    "recurrent scan: "
                    + "; ".join(reasons)
                )
            conv_weight = params["conv1d_weight"].reshape(conv_dim, config.linear_conv_kernel_size)
            conv_bias = params.get("conv1d_bias")
            if use_cached_prefill:
                initial_conv_state = (
                    hybrid_state.conv_state
                    if hybrid_state_is_layer
                    else hybrid_state.conv_state[:, linear_layer_idx]
                )
            else:
                initial_conv_state = jnp.zeros(
                    (row_count, conv_dim, config.linear_conv_kernel_size),
                    dtype=mixed_qkv_t.dtype,
                )

            conv_out, final_conv_state = _packed_causal_conv1d_prefill(
                mixed_qkv,
                initial_conv_state,
                conv_weight,
                conv_bias,
                packed_token_row_ids,
                packed_query_start_loc,
                max_row_tokens=max_row_tokens,
            )

            initial_recurrent = (
                (
                    hybrid_state.recurrent_state
                    if hybrid_state_is_layer
                    else hybrid_state.recurrent_state[:, linear_layer_idx]
                )
                if use_cached_prefill
                else jnp.zeros(
                    (
                        row_count,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                        config.linear_key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            )
            flat_prefix_recurrent_state = None
            if use_packed_post_conv_prefill:
                post_conv_result = backend.gated_delta_packed_prefill_post_conv(
                    conv_out,
                    a,
                    b,
                    params["A"],
                    params["dt_bias"],
                    packed_query_start_loc,
                    num_key_heads=config.linear_num_key_heads,
                    num_value_heads=config.linear_num_value_heads,
                    key_head_dim=config.linear_key_head_dim,
                    value_head_dim=config.linear_value_head_dim,
                    chunk_size=config.linear_chunk_size,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=config.use_qk_norm_in_gdn,
                    max_row_tokens=row_query_len if return_prefix_state else max_row_tokens,
                    return_prefix_state=return_prefix_state,
                )
                if return_prefix_state:
                    core_attn_out, final_state, flat_prefix_recurrent_state = post_conv_result
                else:
                    core_attn_out, final_state = post_conv_result
                core_attn_out = core_attn_out.astype(dtype)
            else:
                query = conv_out[:, :, :key_dim].reshape(
                    1,
                    seq_len,
                    config.linear_num_key_heads,
                    config.linear_key_head_dim,
                )
                key = conv_out[:, :, key_dim:key_dim * 2].reshape(
                    1,
                    seq_len,
                    config.linear_num_key_heads,
                    config.linear_key_head_dim,
                )
                value = conv_out[:, :, key_dim * 2:].reshape(
                    1,
                    seq_len,
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                )
                beta = nn.sigmoid(b)
                g = -params["A"] * nn.softplus(a + params["dt_bias"])

                if v_heads_per_k > 1:
                    query = jnp.repeat(query, v_heads_per_k, axis=2)
                    key = jnp.repeat(key, v_heads_per_k, axis=2)

                query_tokens = query[0].astype(jnp.float32)
                key_tokens = key[0].astype(jnp.float32)
                if config.use_qk_norm_in_gdn:
                    query_tokens = l2norm(query_tokens, axis=-1, eps=1e-6)
                    key_tokens = l2norm(key_tokens, axis=-1, eps=1e-6)
                query_tokens = query_tokens * (1.0 / jnp.sqrt(config.linear_key_head_dim))
                value_tokens = value[0].astype(jnp.float32)
                g_tokens = g[0].astype(jnp.float32)
                beta_tokens = beta[0].astype(jnp.float32)

                def recurrent_scan(state, inputs):
                    row, valid, q_t, k_t, v_t, g_t, beta_t = inputs
                    previous_row_state = state[row]
                    decay = jnp.exp(g_t)[:, None, None]
                    decayed_state = previous_row_state * decay
                    kv_mem = jnp.einsum("hvk,hk->hv", decayed_state, k_t)
                    delta = (v_t - kv_mem) * beta_t[:, None]
                    next_row_state = decayed_state + delta[:, :, None] * k_t[:, None, :]
                    out_t = jnp.einsum("hvk,hk->hv", next_row_state, q_t)
                    next_row_state = jnp.where(valid, next_row_state, previous_row_state)
                    state = state.at[row].set(next_row_state)
                    out_t = jnp.where(valid, out_t, jnp.zeros_like(out_t))
                    return state, (out_t, next_row_state)

                final_state, (recurrent_out_tokens, recurrent_state_tokens) = lax.scan(
                    recurrent_scan,
                    initial_recurrent.astype(jnp.float32),
                    (
                        safe_rows,
                        valid_tokens,
                        query_tokens,
                        key_tokens,
                        value_tokens,
                        g_tokens,
                        beta_tokens,
                    ),
                )
                core_attn_out = recurrent_out_tokens[None, :, :, :].astype(dtype)
                if return_prefix_state or return_first_prefix_state:
                    flat_prefix_recurrent_state = recurrent_state_tokens

            if (return_prefix_state or return_first_prefix_state) and flat_prefix_recurrent_state is not None:
                prefix_recurrent_rows = flat_prefix_recurrent_state[safe_token_indices_by_row]
                prefix_recurrent_rows = jnp.where(
                    valid_queries_by_row[
                        :,
                        :,
                        None,
                        None,
                        None,
                    ],
                    prefix_recurrent_rows,
                    initial_recurrent[:, None, :, :, :],
                )
                prefix_recurrent_state_single = (
                    prefix_recurrent_rows
                    if return_prefix_state
                    else prefix_recurrent_rows[:, 0]
                )
                mixed_rows = mixed_qkv[0][safe_token_indices_by_row]
                mixed_rows = jnp.where(
                    valid_queries_by_row[:, :, None],
                    mixed_rows,
                    0.0,
                )
                conv_input_by_row = jnp.concatenate(
                    [
                        initial_conv_state.astype(mixed_qkv.dtype),
                        mixed_rows.transpose(0, 2, 1),
                    ],
                    axis=-1,
                )
                kernel_size = config.linear_conv_kernel_size
                gather_starts = row_offsets + 1
                gather_idx = gather_starts[:, None] + jnp.arange(
                    kernel_size,
                    dtype=jnp.int32,
                )[None, :]
                gather_idx = jnp.broadcast_to(
                    gather_idx[None, :, None, :],
                    (
                        row_count,
                        row_query_len,
                        conv_dim,
                        kernel_size,
                    ),
                )
                conv_input_expanded = jnp.broadcast_to(
                    conv_input_by_row[:, None, :, :],
                    (
                        row_count,
                        row_query_len,
                        conv_dim,
                        conv_input_by_row.shape[-1],
                    ),
                )
                prefix_conv_rows = jnp.take_along_axis(
                    conv_input_expanded,
                    gather_idx,
                    axis=3,
                )
                prefix_conv_rows = jnp.where(
                    valid_queries_by_row[:, :, None, None],
                    prefix_conv_rows,
                    initial_conv_state[:, None, :, :],
                )
                prefix_layer_conv_state = (
                    prefix_conv_rows.astype(dtype)
                    if return_prefix_state
                    else prefix_conv_rows[:, 0].astype(dtype)
                )

            if (
                hybrid_state is not None
                and hybrid_state.conv_state is not None
                and hybrid_state.recurrent_state is not None
            ):
                if hybrid_state_is_layer:
                    new_conv_state = final_conv_state.astype(dtype)
                    new_recurrent_state = final_state
                else:
                    new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(
                        final_conv_state.astype(dtype)
                    )
                    new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(final_state)
                hybrid_state = replace(
                    hybrid_state,
                    conv_state=new_conv_state,
                    recurrent_state=new_recurrent_state,
                )
                if return_prefix_state or return_first_prefix_state:
                    prefix_layer_state = HybridLayerState(
                        conv_state=prefix_layer_conv_state,
                        recurrent_state=prefix_recurrent_state_single,
                    )

            core_attn_out = core_attn_out.reshape(
                batch * seq_len,
                -1,
                config.linear_value_head_dim,
            )
            z_packed = z.reshape(batch * seq_len, -1, config.linear_value_head_dim)
            core_attn_out = _stable_rmsnorm_fp32(
                core_attn_out,
                params["norm_weight"],
                config.rms_norm_eps,
            )
            core_attn_out = core_attn_out * nn.silu(z_packed)
            core_attn_out = core_attn_out.reshape(batch, seq_len, -1)
            attn_out = _tokenwise_decode_dot(
                core_attn_out.astype(dtype),
                params["out_proj"],
                force_width1=False,
            )
            if hybrid_state is not None:
                if return_prefix_state or return_first_prefix_state:
                    return attn_out, hybrid_state, prefix_layer_state
                return attn_out, hybrid_state
            return attn_out
        elif use_cached_prefill:
            layer_conv_state = (
                hybrid_state.conv_state
                if hybrid_state_is_layer
                else hybrid_state.conv_state[:, linear_layer_idx]
            )
            conv_input = jnp.concatenate([layer_conv_state, mixed_qkv_t], axis=-1)
            conv_out = _causal_conv1d(
                conv_input,
                params["conv1d_weight"],
                params.get("conv1d_bias"),
                activation="silu",
            )[:, :, -seq_len:]
            prefix_layer_conv_state = None
            if return_prefix_state:
                kernel_size = config.linear_conv_kernel_size
                step_starts = jnp.arange(seq_len, dtype=jnp.int32)[:, None] + 1
                gather_idx = step_starts + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
                gather_idx = jnp.broadcast_to(
                    gather_idx[None, :, None, :],
                    (batch, seq_len, conv_dim, kernel_size),
                )
                conv_input_expanded = jnp.broadcast_to(
                    conv_input[:, None, :, :],
                    (batch, seq_len, conv_dim, conv_input.shape[-1]),
                )
                prefix_layer_conv_state = jnp.take_along_axis(
                    conv_input_expanded,
                    gather_idx,
                    axis=3,
                )
        else:
            conv_out = _causal_conv1d(
                mixed_qkv_t,
                params["conv1d_weight"],
                params.get("conv1d_bias"),
                activation="silu",
            )
        conv_out = conv_out.transpose(0, 2, 1)  # [B, T, D]

        initial_recurrent = (
            (
                hybrid_state.recurrent_state
                if hybrid_state_is_layer
                else hybrid_state.recurrent_state[:, linear_layer_idx]
            )
            if use_cached_prefill
            else None
        )
        prefix_recurrent_state_single = None
        use_post_conv_prefill = (
            gdn_prefill_post_conv_enabled(config)
            and not use_recurrent_prefill
            and not return_prefix_state
            and not return_first_prefix_state
        )
        if gdn_disable_fallbacks_enabled(config) and not use_post_conv_prefill:
            reasons = []
            if not gdn_prefill_post_conv_enabled(config):
                reasons.append("post-conv prefill kernel is disabled")
            if use_recurrent_prefill:
                reasons.append("recurrent prefill is requested")
            if return_prefix_state:
                reasons.append("return_prefix_state needs state-sequence output")
            if return_first_prefix_state:
                reasons.append("return_first_prefix_state needs prefix-state output")
            if not reasons:
                reasons.append("the post-conv prefill predicate was false")
            raise RuntimeError(
                "GDN prefill post-conv fallback is disabled, but the requested "
                "prefill route would use the slow JAX recurrent/chunked path: "
                + "; ".join(reasons)
            )
        if use_post_conv_prefill:
            core_attn_out, final_state = backend.gated_delta_prefill_post_conv(
                conv_out,
                a,
                b,
                params["A"],
                params["dt_bias"],
                valid_token_mask,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                chunk_size=config.linear_chunk_size,
                initial_state=initial_recurrent,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            query = conv_out[:, :, :key_dim].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            key = conv_out[:, :, key_dim:key_dim*2].reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            value = conv_out[:, :, key_dim*2:].reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)

            beta = nn.sigmoid(b)
            g = -params["A"] * nn.softplus(a + params["dt_bias"])

            if valid_token_mask is not None:
                valid = valid_token_mask.astype(jnp.bool_)
                query = jnp.where(valid[:, :, None, None], query, 0.0)
                key = jnp.where(valid[:, :, None, None], key, 0.0)
                value = jnp.where(valid[:, :, None, None], value, 0.0)
                beta = jnp.where(valid[:, :, None], beta, 0.0)
                g = jnp.where(valid[:, :, None], g, 0.0)

            if v_heads_per_k > 1:
                query = jnp.repeat(query, v_heads_per_k, axis=2)
                key = jnp.repeat(key, v_heads_per_k, axis=2)

            # Transpose to [B, H, T, D] format for chunk_gated_delta_rule
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            g = g.transpose(0, 2, 1)
            beta = beta.transpose(0, 2, 1)

            if use_recurrent_prefill:
                # Small cached suffixes can use the recurrent path directly and remain
                # aligned with iterative decode.
                if return_prefix_state and use_cached_prefill:
                    core_attn_out, final_state, recurrent_state_steps = jax_recurrent_gated_delta_rule(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        initial_state=initial_recurrent,
                        use_qk_l2norm_in_kernel=True,
                        return_state_sequence=True,
                    )
                    prefix_recurrent_state_single = recurrent_state_steps
                else:
                    core_attn_out, final_state = backend.gated_delta_decode(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        initial_state=initial_recurrent,
                        use_qk_l2norm_in_kernel=True,
                    )
            else:
                # Longer prefill chunks use chunked prefill to amortize work.
                core_attn_out, final_state = backend.gated_delta_prefill(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    chunk_size=config.linear_chunk_size,
                    initial_state=initial_recurrent,
                    use_qk_l2norm_in_kernel=True,
                )

        # Save final state to cache for decode mode
        if (
            hybrid_state is not None
            and hybrid_state.conv_state is not None
            and hybrid_state.recurrent_state is not None
        ):
            # Extract the last real kernel_size inputs. Bucket padding is not
            # part of the convolution history used by recurrent decode.
            kernel_size = config.linear_conv_kernel_size
            prev_conv_state = (
                hybrid_state.conv_state
                if hybrid_state_is_layer
                else hybrid_state.conv_state[:, linear_layer_idx]
            )

            if valid_token_mask is not None:
                valid = valid_token_mask.astype(jnp.bool_)
                # Keep prior cache context and only write new valid positions.
                masked_mixed_qkv_t = jnp.where(
                    valid[:, None, :],
                    mixed_qkv_t,
                    jnp.zeros_like(mixed_qkv_t),
                )
            else:
                masked_mixed_qkv_t = mixed_qkv_t

            valid_lens = (
                valid_token_mask.astype(jnp.int32).sum(axis=1)
                if valid_token_mask is not None
                else jnp.full((batch,), seq_len, dtype=jnp.int32)
            )
            if use_cached_prefill:
                conv_input = jnp.concatenate([prev_conv_state, masked_mixed_qkv_t], axis=-1)
                gather_start = valid_lens
            else:
                conv_input = jnp.concatenate(
                    [
                        jnp.zeros((batch, conv_dim, kernel_size), dtype=masked_mixed_qkv_t.dtype),
                        masked_mixed_qkv_t,
                    ],
                    axis=-1,
                )
                gather_start = valid_lens
            gather_idx = gather_start[:, None] + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
            gather_idx = jnp.broadcast_to(gather_idx[:, None, :], (batch, conv_dim, kernel_size))
            layer_conv_state = jnp.take_along_axis(conv_input, gather_idx, axis=2)

            if hybrid_state_is_layer:
                new_recurrent_state = final_state
                new_conv_state = layer_conv_state.astype(dtype)
            else:
                new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(final_state)
                new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(layer_conv_state.astype(dtype))
            hybrid_state = replace(
                hybrid_state,
                conv_state=new_conv_state,
                recurrent_state=new_recurrent_state,
            )
            if return_prefix_state and use_cached_prefill:
                prefix_layer_state = HybridLayerState(
                    conv_state=prefix_layer_conv_state.astype(dtype)
                    if prefix_layer_conv_state is not None
                    else None,
                    recurrent_state=prefix_recurrent_state_single,
                )

        # Output is [B, H, T, D] - transpose to [B, T, H, D] for reshaping
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)  # [B, H, T, D] -> [B, T, H, D]

    # === OUTPUT PROCESSING (same for both modes) ===
    core_attn_out = core_attn_out.reshape(batch * seq_len, -1, config.linear_value_head_dim)
    z = z.reshape(batch * seq_len, -1, config.linear_value_head_dim)
    core_attn_out = _stable_rmsnorm_fp32(core_attn_out, params["norm_weight"], config.rms_norm_eps)
    core_attn_out = core_attn_out * nn.silu(z)
    core_attn_out = core_attn_out.reshape(batch, seq_len, -1)
    core_attn_out_proj = core_attn_out.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    attn_out = _tokenwise_decode_dot(
        core_attn_out_proj,
        params["out_proj"],
        force_width1=(not is_prefill) and seq_len > 1 and _force_width1_decode_math(),
    )

    if use_cached:
        if return_prefix_state or return_first_prefix_state:
            return attn_out, hybrid_state, prefix_layer_state
        return attn_out, hybrid_state
    elif hybrid_state is not None:
        # Prefill mode with cache - return state for decode
        if (return_prefix_state or return_first_prefix_state) and prefix_layer_state is not None:
            return attn_out, hybrid_state, prefix_layer_state
        return attn_out, hybrid_state
    else:
        # No cache - just return output
        return attn_out
