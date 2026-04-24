"""Metal-compatible operations for linear attention."""

import jax
import jax.numpy as jnp
from jax import lax


def cumsum_metal(x, axis=-1):
    """Cumulative sum that works on Metal.
    
    Standard jnp.cumsum triggers LLVM error on Metal due to mhlo.pad.
    This implementation uses a lax.scan loop instead.
    
    Args:
        x: Input array
        axis: Axis along which to compute cumsum
    
    Returns:
        Cumulative sum along axis
    """
    # Move target axis to last position for easier iteration
    if axis != -1 and axis != x.ndim - 1:
        x = jnp.moveaxis(x, axis, -1)
    
    shape_prefix = x.shape[:-1]
    seq_len = x.shape[-1]
    
    # Reshape to [prefix_dims, seq_len]
    x_flat = x.reshape(-1, seq_len)
    batch_size = x_flat.shape[0]
    
    def scan_fn(carry, i):
        """Accumulate along sequence dimension."""
        # carry: [batch, 1]
        # x_flat: [batch, seq_len]
        current = x_flat[:, i]  # [batch]
        new_carry = carry + current  # [batch]
        return new_carry, new_carry
    
    # Initialize with zeros
    init = jnp.zeros(batch_size, dtype=x.dtype)
    
    # Scan over sequence dimension
    _, cumsum_flat = lax.scan(scan_fn, init, jnp.arange(seq_len))
    
    # cumsum_flat: [seq_len, batch] -> [batch, seq_len]
    cumsum_flat = cumsum_flat.transpose(1, 0)
    
    # Reshape back
    cumsum_out = cumsum_flat.reshape(*shape_prefix, seq_len)
    
    # Move axis back if needed
    if axis != -1 and axis != x.ndim - 1:
        cumsum_out = jnp.moveaxis(cumsum_out, -1, axis)
    
    return cumsum_out


def jax_chunk_gated_delta_rule_metal(
    query, key, value, g, beta, 
    chunk_size=64, initial_state=None, 
    output_final_state=False, use_qk_l2norm_in_kernel=False,
    eps=1e-6
):
    """
    Metal-compatible chunk gated delta rule.
    
    Uses loop-based cumsum instead of jnp.cumsum to avoid LLVM errors.
    """
    from nanovllm_jax.layers import l2norm
    
    initial_dtype = query.dtype
    
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=eps)
        key = l2norm(key, axis=-1, eps=eps)
    
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
    
    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]
    
    def reshape_to_chunks(x):
        return x.reshape(batch_size, num_heads, -1, chunk_size, x.shape[-1])
    
    query_chunks = reshape_to_chunks(query)
    key_chunks = reshape_to_chunks(key)
    value_chunks = reshape_to_chunks(value)
    k_beta_chunks = reshape_to_chunks(k_beta)
    v_beta_chunks = reshape_to_chunks(v_beta)
    
    g_chunks = g.reshape(batch_size, num_heads, -1, chunk_size)
    n_chunks = g_chunks.shape[2]
    
    mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    mask_strict_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)
    
    # Use Metal-compatible cumsum
    g_cumsum = cumsum_metal(g_chunks, axis=-1)
    
    decay_mask = jnp.tril(jnp.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
    
    kkt = jnp.einsum('bhnck,bhnjk->bhncj', k_beta_chunks, key_chunks)
    
    attn = -(kkt * decay_mask)
    attn = jnp.where(mask_upper, 0.0, attn)
    
    def update_row(carry, i):
        attn_carry = carry
        
        mask = jnp.arange(chunk_size) < i
        
        row_i = attn_carry[..., i, :] * mask[None, None, None, :]
        
        sub_i = attn_carry * mask[None, None, None, :, None] * mask[None, None, None, None, :]
        
        contribution = jnp.einsum('bhnj,bhnjk->bhnk', row_i, sub_i)
        
        new_row = row_i + contribution
        new_row = new_row * mask[None, None, None, :]
        
        attn_carry = attn_carry.at[..., i, :].set(new_row)
        return attn_carry, i
    
    attn, _ = lax.scan(update_row, attn, jnp.arange(1, chunk_size))
    
    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)[None, None, None, :, :]
    
    value_transformed = jnp.einsum('bhnct,bhntv->bhncv', attn, v_beta_chunks)
    
    k_cumdecay = jnp.einsum('bhnct,bhntv->bhncv', attn, k_beta_chunks * jnp.exp(g_chunks)[..., None])
    
    if initial_state is None:
        state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)
    
    def process_chunk(carry, i):
        state = carry
        q_i = query_chunks[:, :, i]
        k_i = key_chunks[:, :, i]
        v_i = value_transformed[:, :, i]
        decay_mask_i = decay_mask[:, :, i]
        k_cumdecay_i = k_cumdecay[:, :, i]
        g_cumsum_i = g_cumsum[:, :, i]
        
        attn_i = jnp.einsum('bhck,bhdk->bhcd', q_i, k_i) * decay_mask_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)
        
        v_prime = jnp.einsum('bhck,bhkv->bhcv', k_cumdecay_i, state)
        
        v_new = v_i - v_prime
        
        attn_inter = jnp.einsum('bhck,bhkv->bhcv', q_i * jnp.exp(g_cumsum_i)[..., None], state)
        
        attn_v_new = jnp.einsum('bhcd,bhdv->bhcv', attn_i, v_new)
        core_attn_out_i = attn_inter + attn_v_new
        
        g_last_minus_g = g_cumsum_i[..., -1, None] - g_cumsum_i
        k_weighted = k_i * jnp.exp(g_last_minus_g)[..., None]
        state_update = jnp.einsum('bhck,bhcv->bhkv', k_weighted, v_new)
        state = state * jnp.exp(g_cumsum_i[..., -1, None, None]) + state_update
        
        return state, core_attn_out_i
    
    final_state, core_attn_out_chunks = lax.scan(process_chunk, state, jnp.arange(n_chunks))
    
    core_attn_out = core_attn_out_chunks.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :seq_len]
    
    if not output_final_state:
        final_state = None
    
    core_attn_out = core_attn_out.astype(initial_dtype)
    return core_attn_out, final_state
