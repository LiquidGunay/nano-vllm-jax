"""Chunked prefill implementation for fixed-shape compilation.

This module implements chunked prefill that compiles ONCE for a fixed chunk size,
then can handle any sequence length by processing in chunks.
"""

import jax
import jax.numpy as jnp
from typing import Optional
from dataclasses import replace

def create_chunked_attention_mask(
    chunk_size: int,
    actual_len: int,
    is_causal: bool = True,
) -> jnp.ndarray:
    """Create attention mask for a chunk.
    
    Args:
        chunk_size: Fixed chunk size (compiled shape)
        actual_len: Actual length within chunk (<= chunk_size)
        is_causal: Whether to apply causal masking
    
    Returns:
        Mask of shape (chunk_size, chunk_size) where:
        - 0.0 = attend
        - -inf = mask out
    """
    # Create causal mask
    if is_causal:
        mask = jnp.triu(jnp.full((chunk_size, chunk_size), -1e10, dtype=jnp.float32), k=1)
    else:
        mask = jnp.zeros((chunk_size, chunk_size), dtype=jnp.float32)
    
    # Mask out padding (positions >= actual_len)
    padding_mask = jnp.arange(chunk_size) >= actual_len  # [chunk_size]
    
    # Apply padding mask: mask out rows and columns for padding positions
    # For query positions (rows): don't attend FROM padding
    # For key positions (columns): don't attend TO padding
    mask = jnp.where(padding_mask[None, :], -1e10, mask)  # Mask columns (keys)
    mask = jnp.where(padding_mask[:, None], -1e10, mask)  # Mask rows (queries)
    
    return mask


def chunked_paged_attention(
    query: jnp.ndarray,  # [batch, chunk_size, num_heads, head_dim]
    k_cache: jnp.ndarray,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jnp.ndarray,
    slot_mapping: jnp.ndarray,  # [batch, chunk_size]
    kv_lens: jnp.ndarray,  # [batch]
    actual_lens: jnp.ndarray,  # [batch] - actual length within this chunk
    scale: float,
    num_key_value_groups: int,
    chunk_size: int,
) -> jnp.ndarray:
    """Chunked paged attention with fixed compilation shape.
    
    This function ALWAYS processes chunk_size tokens, padding if necessary.
    The actual_lens parameter tells us the real length within each batch item.
    
    Args:
        query: Query tensor (always chunk_size, padded)
        k_cache: Key cache (paged)
        v_cache: Value cache (paged)
        slot_mapping: Maps positions to physical cache locations
        kv_lens: Total sequence lengths so far
        actual_lens: Actual length within this chunk (<= chunk_size)
        scale: Attention scale
        num_key_value_groups: GQA groups
        chunk_size: Fixed chunk size
    
    Returns:
        Attention output [batch, chunk_size, num_heads * head_dim]
    """
    batch, seq_len, num_heads, head_dim = query.shape
    assert seq_len == chunk_size, f"Query seq_len {seq_len} != chunk_size {chunk_size}"
    
    # Gather KV pairs for all positions using slot_mapping
    k_gathered = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])[slot_mapping]
    v_gathered = v_cache.reshape(-1, v_cache.shape[2], v_cache.shape[3])[slot_mapping]
    
    # Expand KV for GQA
    if num_key_value_groups > 1:
        k_gathered = jnp.repeat(k_gathered, num_key_value_groups, axis=2)
        v_gathered = jnp.repeat(v_gathered, num_key_value_groups, axis=2)
    
    # Compute attention scores: [B, N, chunk_size, chunk_size]
    attn_scores = jnp.einsum("btnh,bsnh->bnts", query, k_gathered) * scale
    
    # Apply per-batch masking (each batch item can have different actual_len)
    # We need to broadcast the mask across batch
    def apply_mask_for_batch(scores, actual_len):
        mask = create_chunked_attention_mask(chunk_size, actual_len, is_causal=True)
        return scores + mask[None, :, :]  # Broadcast across heads
    
    # Apply mask for each batch item
    # actual_lens shape: [batch], we need to apply different masks
    # Use jax.vmap to apply per-batch
    attn_scores = jax.vmap(apply_mask_for_batch, in_axes=(0, 0))(attn_scores, actual_lens)
    
    # Softmax and output
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum("bnts,bsnh->btnh", attn_weights, v_gathered)
    out = out.reshape(batch, chunk_size, num_heads * head_dim)
    
    return out


def pad_to_chunk_size(
    tokens: jnp.ndarray,  # [batch, seq_len]
    chunk_size: int,
    pad_value: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad tokens to chunk_size.
    
    Args:
        tokens: Input tokens
        chunk_size: Target chunk size
        pad_value: Padding token ID
    
    Returns:
        (padded_tokens, actual_lens)
    """
    batch, seq_len = tokens.shape
    actual_lens = jnp.full((batch,), seq_len, dtype=jnp.int32)
    
    if seq_len >= chunk_size:
        # Truncate if too long (shouldn't happen in proper chunking)
        return tokens[:, :chunk_size], actual_lens
    
    # Pad to chunk_size
    pad_len = chunk_size - seq_len
    padded = jnp.pad(tokens, ((0, 0), (0, pad_len)), constant_values=pad_value)
    return padded, actual_lens


def chunked_full_attention_block(
    x: jnp.ndarray,  # [batch, seq_len, hidden_dim] - will be padded to chunk_size
    params: dict,
    positions: jnp.ndarray,  # [batch, seq_len]
    config,
    kv_cache_state: Optional['KVCacheState'],
    actual_lens: jnp.ndarray,  # [batch] - actual length within chunk
    chunk_size: int,
):
    """Full attention block with chunked prefill support.
    
    Args:
        x: Input tensor (will be padded to chunk_size)
        params: Layer parameters
        positions: Position IDs
        config: Model config
        kv_cache_state: KV cache state
        actual_lens: Actual length within chunk for each batch item
        chunk_size: Fixed chunk size for compilation
    
    Returns:
        (output, updated_kv_cache_state)
    """
    from nanovllm_jax.layers import rms_norm, apply_rope
    from nanovllm_jax.kv_cache import update_kv_cache
    import jax.nn as nn
    
    batch, seq_len, hidden = x.shape
    
    # Pad to chunk_size if necessary
    if seq_len < chunk_size:
        pad_len = chunk_size - seq_len
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
        positions = jnp.pad(positions, ((0, 0), (0, pad_len)), constant_values=0)
    elif seq_len > chunk_size:
        raise ValueError(f"Input seq_len {seq_len} > chunk_size {chunk_size}. Use chunking.")
    
    # Cast to bfloat16
    x_bf16 = x.astype(jnp.bfloat16)
    
    # Q, K, V projections
    q_gate = jnp.dot(x_bf16, params["q_proj"])
    q_gate_reshaped = q_gate.reshape(batch, chunk_size, config.num_attention_heads, 2 * config.head_dim)
    query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
    gate = gate.reshape(batch, chunk_size, -1)
    
    k = jnp.dot(x_bf16, params["k_proj"]).reshape(batch, chunk_size, config.num_key_value_heads, config.head_dim)
    v = jnp.dot(x_bf16, params["v_proj"]).reshape(batch, chunk_size, config.num_key_value_heads, config.head_dim)
    
    # Apply RMSNorm
    query = rms_norm(query, params["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, params["k_norm"], config.rms_norm_eps)
    
    # Transpose to [B, H, T, D]
    query = query.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    # Apply RoPE
    query = apply_rope(query, positions, config.head_dim, config.rope_theta, 
                      config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
    k = apply_rope(k, positions, config.head_dim, config.rope_theta,
                  config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
    
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    
    if kv_cache_state is not None:
        # Transpose K, V back to [B, T, K, H] for cache storage
        k_cache_input = k.transpose(0, 2, 1, 3)
        v_cache_input = v.transpose(0, 2, 1, 3)
        
        # Update KV cache
        k_cache, v_cache = update_kv_cache(
            kv_cache_state.k_cache,
            kv_cache_state.v_cache,
            kv_cache_state.slot_mapping,
            k_cache_input,
            v_cache_input,
        )
        
        # Use chunked paged attention
        query_btnh = query.transpose(0, 2, 1, 3)
        
        out = chunked_paged_attention(
            query=query_btnh,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=kv_cache_state.slot_mapping,
            kv_lens=kv_cache_state.kv_lens,
            actual_lens=actual_lens,
            scale=1.0 / jnp.sqrt(config.head_dim),
            num_key_value_groups=num_key_value_groups,
            chunk_size=chunk_size,
        )
        
        # Update cache state
        kv_cache_state = replace(
            kv_cache_state,
            k_cache=k_cache,
            v_cache=v_cache,
        )
    else:
        # No cache - standard attention with chunked masking
        k = jnp.repeat(k, num_key_value_groups, axis=1)
        v = jnp.repeat(v, num_key_value_groups, axis=1)
        
        # Create chunked mask (same for all batch items without cache)
        # For simplicity, use max actual_len
        # In practice, should use per-batch masking
        actual_len = jnp.max(actual_lens)
        mask = create_chunked_attention_mask(chunk_size, actual_len, is_causal=True)
        
        attn = nn.softmax(
            jnp.einsum("bhtd,bhsd->bhts", query, k) / jnp.sqrt(config.head_dim) + mask[None, None, :, :].astype(query.dtype),
            axis=-1
        )
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(batch, chunk_size, -1)
    
    # Apply gate and output projection
    out = out * nn.sigmoid(gate)
    out = jnp.dot(out, params["o_proj"])
    
    return out, kv_cache_state
