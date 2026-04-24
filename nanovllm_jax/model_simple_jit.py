"""Simple JIT-compiled forward pass using jax.jit on the full forward function."""

import jax
import jax.numpy as jnp
from jax import nn, lax
from typing import Tuple, Optional
from functools import partial
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import rms_norm, apply_rope, causal_mask, get_activation, l2norm, causal_conv1d_update
from nanovllm_jax.kv_cache import KVCacheState, update_kv_cache, paged_attention, paged_attention_decode, init_linear_attention_states
from nanovllm_jax.conv1d_metal import causal_conv1d_metal
from nanovllm_jax.model import (
    jax_recurrent_gated_delta_rule,
    jax_chunk_gated_delta_rule,
    gated_deltanet_block,
    full_attention_block,
    transformer_block,
)


def forward_simple_jit(
    tokens,
    params,
    config,
    kv_cache_state=None,
    is_prefill=True,
    return_hidden=False,
):
    """Simple JIT-friendly forward pass.
    
    This is a wrapper that JIT compiles the core operations.
    """
    batch, seq_len = tokens.shape
    dtype = config.get_dtype()
    
    # Embedding lookup
    x = params.embed_tokens[tokens].astype(dtype)
    
    # Position IDs - for decode, start from current sequence position
    if is_prefill:
        positions_1d = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
    else:
        # Decode: position is (kv_lens) before increment = position of NEW token
        # kv_lens will be incremented before attention, so position is current kv_lens
        positions_1d = kv_cache_state.kv_lens[:, None]
    positions = jnp.stack([positions_1d, positions_1d, positions_1d], axis=0)
    
    # Causal mask
    mask = causal_mask(seq_len, seq_len)
    
    # Initialize linear attention states during prefill
    if is_prefill and kv_cache_state is not None:
        kv_cache_state = init_linear_attention_states(kv_cache_state, config, batch_size=batch, dtype=dtype)
        # Set kv_lens to prefill sequence length
        kv_cache_state = replace(kv_cache_state, kv_lens=jnp.full((batch,), seq_len, dtype=jnp.int32))
    
    # For decode: increment kv_lens BEFORE attention so new K/V is visible
    # kv_lens represents the TOTAL sequence length including the new token
    if not is_prefill and kv_cache_state is not None:
        kv_cache_state = replace(kv_cache_state, kv_lens=kv_cache_state.kv_lens + 1)
    
    # Process layers
    for i, lp in enumerate(params.layers):
        x, kv_cache_state = transformer_block(
            x, lp, positions, mask, i, config, kv_cache_state, is_prefill
        )
    
    # Final norm
    hidden_pre = x
    x = rms_norm(x, params.norm_weight, config.rms_norm_eps)
    
    if return_hidden:
        # Return pre-norm hidden state for MTP (mlx-lm convention)
        return hidden_pre, kv_cache_state
    
    # LM head
    x = x.astype(jnp.float32)
    if params.lm_head is not None:
        logits = jnp.dot(x, params.lm_head)
    else:
        logits = jnp.dot(x, params.embed_tokens.T)
    
    return logits, kv_cache_state


# Create JIT-compiled version with config, is_prefill, and return_hidden as static
forward_jit = jax.jit(forward_simple_jit, static_argnames=['config', 'is_prefill', 'return_hidden'])
