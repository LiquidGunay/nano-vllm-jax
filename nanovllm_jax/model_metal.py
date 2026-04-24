"""Metal hybrid forward pass for Qwen3.5.

This module provides JIT-compiled functions for Metal backend while maintaining
exact parity with the CPU implementation.

Strategy:
- JIT compile on Metal: RMS Norm, MLP, Attention projections
- Run on CPU: Linear attention, RoPE, KV cache operations
- Maintain exact numerical parity with HF implementation
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kv_cache import KVCacheState, update_kv_cache, paged_attention, paged_attention_decode, init_linear_attention_states
from nanovllm_jax.layers import rms_norm, apply_rope, causal_mask
import jax.nn as nn


def get_jit_backend():
    """Get the best available JIT backend."""
    import os
    backend = os.environ.get('JAX_PLATFORMS', '').upper()
    if backend == 'METAL':
        return 'METAL'
    return None


# ============================================================================
# Metal JIT Functions
# ============================================================================

def make_rms_norm_jit(backend=None):
    """Create JIT-compiled RMS norm function."""
    def rms_norm_impl(x, weight, eps):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jnp.reciprocal(jnp.sqrt(variance + eps))
        return x_norm * (1.0 + weight)
    
    return jax.jit(rms_norm_impl, static_argnums=(2,), backend=backend)


def make_mlp_jit(backend=None):
    """Create JIT-compiled MLP function."""
    def mlp_impl(x, gate_proj, up_proj, down_proj):
        gate = jnp.dot(x, gate_proj)
        up = jnp.dot(x, up_proj)
        hidden = jax.nn.silu(gate) * up
        return jnp.dot(hidden, down_proj)
    
    return jax.jit(mlp_impl, backend=backend)


def make_attention_projections_jit(backend=None):
    """Create JIT-compiled attention projection function."""
    def attention_proj_impl(x, q_proj, k_proj, v_proj, num_heads, num_kv_heads, head_dim):
        batch, seq_len, _ = x.shape
        
        # Q + gate projection (Qwen3.5 specific)
        q_gate = jnp.dot(x, q_proj)
        q_gate_reshaped = q_gate.reshape(batch, seq_len, num_heads, 2 * head_dim)
        query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
        gate = gate.reshape(batch, seq_len, -1)
        
        # K and V projections
        k = jnp.dot(x, k_proj).reshape(batch, seq_len, num_kv_heads, head_dim)
        v = jnp.dot(x, v_proj).reshape(batch, seq_len, num_kv_heads, head_dim)
        
        return query, gate, k, v
    
    return jax.jit(attention_proj_impl, backend=backend)


# ============================================================================
# Helper Functions (CPU)
# ============================================================================

def _apply_attention_with_cache(
    query, gate, k, v, params, config, kv_cache_state, is_prefill, mask, positions
):
    """Apply attention with KV cache (runs on CPU to maintain parity)."""
    batch = query.shape[0]
    seq_len = query.shape[1]
    
    # Apply RMSNorm to Q and K BEFORE transpose
    query = rms_norm(query, params["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, params["k_norm"], config.rms_norm_eps)
    
    # Transpose to [B, H, T, D]
    query = query.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    # Apply RoPE (CPU)
    query = apply_rope(
        query, positions, config.head_dim, config.rope_theta,
        config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section
    )
    k = apply_rope(
        k, positions, config.head_dim, config.rope_theta,
        config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section
    )
    
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    
    if kv_cache_state is not None:
        # Transpose K, V back to [B, T, K, H] for cache storage
        k_cache_input = k.transpose(0, 2, 1, 3)
        v_cache_input = v.transpose(0, 2, 1, 3)
        
        # Update KV cache (CPU)
        k_cache, v_cache = update_kv_cache(
            kv_cache_state.k_cache,
            kv_cache_state.v_cache,
            kv_cache_state.slot_mapping,
            k_cache_input,
            v_cache_input,
        )
        
        # Transpose query to [B, T, N, H]
        query_btnh = query.transpose(0, 2, 1, 3)
        
        if is_prefill:
            # Prefill attention (CPU)
            attn_out_btnh = paged_attention(
                query=query_btnh,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=kv_cache_state.slot_mapping,
                kv_lens=kv_cache_state.kv_lens,
                scale=1.0 / jnp.sqrt(config.head_dim),
                num_key_value_groups=num_key_value_groups,
            )
            out = attn_out_btnh
        else:
            # Decode attention (CPU)
            attn_out_btnh = paged_attention_decode(
                query=query_btnh,
                k_cache=k_cache,
                v_cache=v_cache,
                block_table=kv_cache_state.block_table,
                kv_lens=kv_cache_state.kv_lens,
                block_size=config.block_size,
                scale=1.0 / jnp.sqrt(config.head_dim),
                num_key_value_groups=num_key_value_groups,
            )
            out = attn_out_btnh
        
        # Update cache state
        kv_cache_state = replace(
            kv_cache_state,
            k_cache=k_cache,
            v_cache=v_cache,
        )
    else:
        # No cache - standard attention (CPU)
        k = jnp.repeat(k, num_key_value_groups, axis=1)
        v = jnp.repeat(v, num_key_value_groups, axis=1)
        
        attn = nn.softmax(
            jnp.einsum("bhtd,bhsd->bhts", query, k) / jnp.sqrt(config.head_dim) + mask[None, None, :, :].astype(query.dtype),
            -1
        )
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    # Apply gate and output projection
    out = out * nn.sigmoid(gate)
    out = jnp.dot(out, params["o_proj"])
    
    return out, kv_cache_state


# ============================================================================
# Metal Hybrid Forward Pass
# ============================================================================

def forward_metal_hybrid(
    tokens,
    params,
    config,
    kv_cache_state=None,
    is_prefill=True,
    return_hidden=False,
    use_jit=True,
):
    """Forward pass with Metal JIT optimization.
    
    This maintains exact parity with the CPU implementation while JIT compiling
    compatible operations on Metal.
    
    Args:
        tokens: Input token IDs [batch, seq_len]
        params: Model parameters
        config: Model config
        kv_cache_state: Optional KV cache state
        is_prefill: Whether this is prefill vs decode
        return_hidden: If True, return pre-norm hidden states (for MTP)
        use_jit: If True, use Metal JIT; if False, run everything on CPU
    
    Returns:
        tuple: (logits, updated_kv_cache_state) or (pre_norm_hidden, kv_cache_state)
    """
    batch, seq_len = tokens.shape
    dtype = config.get_dtype()
    
    # Get JIT backend
    backend = get_jit_backend() if use_jit else None
    
    # Create JIT functions if using Metal
    if backend == 'METAL':
        rms_norm_fn = make_rms_norm_jit(backend)
        mlp_fn = make_mlp_jit(backend)
        attn_proj_fn = make_attention_projections_jit(backend)
    else:
        # Fall back to CPU functions
        rms_norm_fn = rms_norm
        mlp_fn = None
        attn_proj_fn = None
    
    # Embedding lookup (CPU for now, could JIT later)
    x = params.embed_tokens[tokens].astype(dtype)
    
    # Create position IDs
    positions_1d = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
    positions = jnp.stack([positions_1d, positions_1d, positions_1d], axis=0)
    
    # Causal mask (CPU, needs static args)
    mask = causal_mask(seq_len, seq_len)
    
    # Initialize linear attention states during prefill
    if is_prefill and kv_cache_state is not None:
        kv_cache_state = init_linear_attention_states(kv_cache_state, config, batch_size=batch)
    
    # Import linear attention block (CPU only)
    from nanovllm_jax.model import gated_deltanet_block, get_activation
    
    # Process each layer
    for i, lp in enumerate(params.layers):
        residual = x
        
        # Input layer norm (Metal JIT if available)
        if backend == 'METAL':
            x = rms_norm_fn(x, lp["input_norm"], config.rms_norm_eps)
        else:
            x = rms_norm(x, lp["input_norm"], config.rms_norm_eps)
        
        # Attention/Linear attention
        if config.layer_types[i] == "full_attention":
            # Full attention block
            if backend == 'METAL' and attn_proj_fn is not None:
                # Use Metal JIT for projections
                query, gate, k, v = attn_proj_fn(
                    x, lp["q_proj"], lp["k_proj"], lp["v_proj"],
                    config.num_attention_heads, config.num_key_value_heads, config.head_dim
                )
                # Apply attention with cache (CPU)
                x, kv_cache_state = _apply_attention_with_cache(
                    query, gate, k, v, lp, config, kv_cache_state, is_prefill, mask, positions
                )
            else:
                # CPU fallback - use original implementation
                from nanovllm_jax.model import full_attention_block
                x, kv_cache_state = full_attention_block(
                    x, lp, positions, mask, config, kv_cache_state, is_prefill
                )
        else:
            # Linear attention (CPU only)
            result = gated_deltanet_block(
                x, lp, positions, config, i,
                is_prefill=is_prefill,
                kv_cache_state=kv_cache_state
            )
            if isinstance(result, tuple):
                x, kv_cache_state = result
            else:
                x = result
        
        # Add residual
        x = residual + x
        
        # MLP path
        residual = x
        
        # FFN norm (Metal JIT if available)
        if backend == 'METAL':
            x = rms_norm_fn(x, lp["ffn_norm"], config.rms_norm_eps)
        else:
            x = rms_norm(x, lp["ffn_norm"], config.rms_norm_eps)
        
        # MLP computation (Metal JIT if available)
        if backend == 'METAL' and mlp_fn is not None:
            x = mlp_fn(
                x, lp["gate_proj"], lp["up_proj"], lp["down_proj"]
            )
        else:
            # CPU fallback
            gate = jnp.dot(x, lp["gate_proj"])
            up = jnp.dot(x, lp["up_proj"])
            x = get_activation(config.hidden_act)(gate) * up
            x = jnp.dot(x, lp["down_proj"])
        
        # Add residual
        x = residual + x
    
    # Save pre-norm hidden for MTP
    hidden_pre = x
    
    # Final norm (Metal JIT if available)
    if backend == 'METAL':
        x = rms_norm_fn(x, params.norm_weight, config.rms_norm_eps)
    else:
        x = rms_norm(x, params.norm_weight, config.rms_norm_eps)
    
    if return_hidden:
        return hidden_pre, kv_cache_state
    
    # Logits (CPU for numerical stability)
    x = x.astype(jnp.float32)
    logits = jnp.dot(x, params.lm_head) if params.lm_head is not None else jnp.dot(x, params.embed_tokens.T)
    
    return logits, kv_cache_state
