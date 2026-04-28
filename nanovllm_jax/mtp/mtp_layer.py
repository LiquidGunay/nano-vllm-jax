"""MTP layer implementation for Qwen3.5."""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import rms_norm, apply_rope, causal_mask, get_activation, silu


@dataclass
class MTPParams:
    """Parameters for MTP head.
    
    Attributes:
        eh_proj: Input fusion projection [hidden_size*2, hidden_size]
        layers: List of MTP layer parameters (usually 1 layer)
        pre_fc_norm_hidden: Pre-norm for hidden state [hidden_size]
        pre_fc_norm_embedding: Pre-norm for embedding [hidden_size]
        final_norm: Final norm after MTP layers [hidden_size]
        lm_head: Output projection to vocab [hidden_size, vocab_size] (shared with main model)
    """
    eh_proj: jnp.ndarray
    layers: list
    pre_fc_norm_hidden: jnp.ndarray
    pre_fc_norm_embedding: jnp.ndarray
    final_norm: Optional[jnp.ndarray] = None
    lm_head: Optional[jnp.ndarray] = None
    
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        next_token_ids: jnp.ndarray,
        embed_tokens: jnp.ndarray,
        config: Qwen3_5Config,
        positions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Forward pass through MTP head.
        
        Args:
            hidden_state: Pre-norm hidden state [batch, seq_len, hidden_size]
            next_token_ids: Token IDs of confirmed token [batch, seq_len]
            embed_tokens: Embedding table [vocab_size, hidden_size]
            config: Model config
            positions: Position IDs [batch, seq_len]
            
        Returns:
            Draft logits [batch, seq_len, vocab_size]
        """
        return mtp_forward(
            hidden_state=hidden_state,
            next_token_ids=next_token_ids,
            embed_tokens=embed_tokens,
            params=self,
            config=config,
            positions=positions,
        )


@dataclass
class MTPConfig:
    """Configuration for MTP head.
    
    Mirrors the main model's attention config.
    """
    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 3584
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000
    partial_rotary_factor: float = 0.25
    hidden_act: str = "silu"
    num_layers: int = 1


def init_mtp_params(key: jax.Array, config: Qwen3_5Config) -> MTPParams:
    """Initialize random MTP parameters.
    
    Args:
        key: JAX PRNG key
        config: Main model config (MTP uses same architecture)
        
    Returns:
        MTPParams with initialized weights
    """
    keys = jax.random.split(key, 4)
    
    # Input fusion projection: [hidden_size*2, hidden_size]
    eh_proj = jax.random.normal(
        keys[0], 
        (config.hidden_size * 2, config.hidden_size)
    ) * (config.hidden_size ** -0.5)
    
    # MTP layers (usually just 1)
    layers = []
    for i in range(config.mtp_num_hidden_layers):
        layer_key = keys[1] if i == 0 else jax.random.split(keys[1])[i]
        layer = init_mtp_layer(layer_key, config)
        layers.append(layer)
    
    # Pre-norms
    pre_fc_norm_hidden = jnp.ones(config.hidden_size)
    pre_fc_norm_embedding = jnp.ones(config.hidden_size)
    
    # Final norm (ALWAYS present, as in mlx-lm MTPModule.norm)
    final_norm = jnp.ones(config.hidden_size)
    
    # LM head (shared with main model, but we initialize here for standalone use)
    lm_head = jax.random.normal(
        keys[2],
        (config.hidden_size, config.vocab_size)
    ) * (config.hidden_size ** -0.5)
    
    return MTPParams(
        eh_proj=eh_proj,
        layers=layers,
        pre_fc_norm_hidden=pre_fc_norm_hidden,
        pre_fc_norm_embedding=pre_fc_norm_embedding,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def init_mtp_layer(key: jax.Array, config: Qwen3_5Config) -> Dict[str, jnp.ndarray]:
    """Initialize a single MTP layer (full attention).
    
    Args:
        key: JAX PRNG key
        config: Model config
        
    Returns:
        Dictionary of layer parameters
    """
    keys = jax.random.split(key, 8)
    attn_out_dim = config.num_attention_heads * config.head_dim
    
    return {
        # Self-attention
        "q_proj": jax.random.normal(
            keys[0], 
            (config.hidden_size, attn_out_dim * 2)
        ) * (config.hidden_size ** -0.5),
        "k_proj": jax.random.normal(
            keys[1],
            (config.hidden_size, config.num_key_value_heads * config.head_dim)
        ) * (config.hidden_size ** -0.5),
        "v_proj": jax.random.normal(
            keys[2],
            (config.hidden_size, config.num_key_value_heads * config.head_dim)
        ) * (config.hidden_size ** -0.5),
        "o_proj": jax.random.normal(
            keys[3],
            (attn_out_dim, config.hidden_size)
        ) * (config.hidden_size ** -0.5),
        
        # Norms for attention (shared across heads, as in mlx-lm and checkpoint)
        "q_norm": jnp.ones(config.head_dim),
        "k_norm": jnp.ones(config.head_dim),
        "input_norm": jnp.ones(config.hidden_size),
        "post_attn_norm": jnp.ones(config.hidden_size),

        # Feed-forward block
        "gate_proj": jax.random.normal(
            keys[4],
            (config.hidden_size, config.intermediate_size),
        ) * (config.hidden_size ** -0.5),
        "up_proj": jax.random.normal(
            keys[5],
            (config.hidden_size, config.intermediate_size),
        ) * (config.hidden_size ** -0.5),
        "down_proj": jax.random.normal(
            keys[6],
            (config.intermediate_size, config.hidden_size),
        ) * (config.intermediate_size ** -0.5),
    }


def mtp_forward(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """MTP forward pass to generate draft logits.
    
    Args:
        hidden_state: Pre-norm hidden state from main model [batch, seq_len, hidden_size]
        next_token_ids: Token IDs of confirmed token t+1 [batch, seq_len]
        embed_tokens: Embedding table [vocab_size, hidden_size]
        params: MTP parameters
        config: Main model config
        positions: Position IDs [batch, seq_len] (optional, will compute if not provided)
        
    Returns:
        Tuple of:
        - Draft logits [batch, seq_len, vocab_size]
        - Output hidden state [batch, seq_len, hidden_size] (for chaining MTP predictions)
    """
    batch, seq_len, _ = hidden_state.shape
    
    # Compute positions if not provided: [0, 1, ..., seq_len-1] for each batch item
    if positions is None:
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
    
    # Lookup embeddings for confirmed tokens (like mlx-lm does)
    next_token_embed = embed_tokens[next_token_ids]
    
    # Apply pre-norms
    hidden_norm = rms_norm(hidden_state, params.pre_fc_norm_hidden, config.rms_norm_eps)
    embed_norm = rms_norm(next_token_embed, params.pre_fc_norm_embedding, config.rms_norm_eps)
    
    # Fuse inputs: [embed, hidden] -> [batch, seq_len, hidden_size*2]
    # Note: Order matters! mlx-lm and HF use [embedding, hidden], not [hidden, embedding]
    fused = jnp.concatenate([embed_norm, hidden_norm], axis=-1)
    
    # Input projection: [batch, seq_len, hidden_size*2] @ [hidden_size*2, hidden_size] -> [batch, seq_len, hidden_size]
    x = jnp.dot(fused, params.eh_proj)
    
    # Run MTP layers
    for i, layer_params in enumerate(params.layers):
        x = mtp_layer_forward(x, layer_params, config, positions)
    
    # Apply final norm after MTP layers (ALWAYS, as in mlx-lm)
    x_normed = rms_norm(x, params.final_norm, config.rms_norm_eps)
    
    # Output projection to vocab
    logits = jnp.dot(x_normed, params.lm_head)
    
    return logits, x


def mtp_layer_forward(
    x: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Forward pass for a single MTP layer.
    
    Args:
        x: Input [batch, seq_len, hidden_size]
        params: Layer parameters
        config: Model config
        positions: Position IDs (optional)
        
    Returns:
        Output [batch, seq_len, hidden_size]
    """
    batch, seq_len, _ = x.shape
    
    # Input norm
    x_norm = rms_norm(x, params["input_norm"], config.rms_norm_eps)
    
    # QKV projection
    # Qwen3.5: q_proj outputs [query, gate]
    q_gate = jnp.dot(x_norm, params["q_proj"])
    attn_out_dim = config.num_attention_heads * config.head_dim
    q_gate_reshaped = q_gate.reshape(batch, seq_len, config.num_attention_heads, 2 * config.head_dim)
    query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
    gate = gate.reshape(batch, seq_len, -1)
    
    k = jnp.dot(x_norm, params["k_proj"]).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    v = jnp.dot(x_norm, params["v_proj"]).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    
    # Transpose to [B, H, T, D] first
    query = query.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    # Apply RMSNorm on last dimension (head_dim) - matches mlx-lm
    # q_norm and k_norm are 1D arrays of shape (head_dim,)
    query = rms_norm(query, params["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, params["k_norm"], config.rms_norm_eps)
    
    # Apply RoPE if positions provided
    # MTP uses standard 1D RoPE (not MROPE) - pass None for mrope_section
    if positions is not None:
        query = apply_rope(
            query, 
            positions, 
            config.head_dim, 
            config.rope_theta, 
            config.partial_rotary_factor,
            layout="BHTD",
            mrope_section=None  # MTP uses standard RoPE, not MROPE
        )
        k = apply_rope(
            k, 
            positions, 
            config.head_dim, 
            config.rope_theta, 
            config.partial_rotary_factor,
            layout="BHTD",
            mrope_section=None  # MTP uses standard RoPE, not MROPE
        )
    
    # Causal mask
    mask = causal_mask(seq_len, seq_len)
    
    # GQA: repeat KV to match Q heads
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    k = jnp.repeat(k, num_key_value_groups, axis=1)
    v = jnp.repeat(v, num_key_value_groups, axis=1)
    
    # Attention: [B, H, T, D]
    attn_weights = jnp.einsum("bhtd,bhsd->bhts", query, k) / jnp.sqrt(config.head_dim)
    attn_weights = attn_weights + mask
    attn_probs = jax.nn.softmax(attn_weights, axis=-1)
    attn_out = jnp.einsum("bhts,bhsd->bhtd", attn_probs, v)
    
    # Transpose back to [B, T, H, D]
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    # Apply gate (SwiGLU-style)
    attn_out = attn_out * jax.nn.sigmoid(gate)
    
    # Output projection
    attn_out = jnp.dot(attn_out, params["o_proj"])
    
    # Residual first
    x = x + attn_out
    
    # Post-attention norm (applied to residual, before MLP) - matches mlx-lm
    x_norm = rms_norm(x, params["post_attn_norm"], config.rms_norm_eps)
    
    # MLP (SwiGLU)
    gate_proj = jnp.dot(x_norm, params["gate_proj"])
    up_proj = jnp.dot(x_norm, params["up_proj"])
    activation_fn = get_activation(config.hidden_act)
    # SwiGLU: silu(gate) * up
    mlp_out = jnp.dot(activation_fn(gate_proj) * up_proj, params["down_proj"])
    
    # Residual
    x = x + mlp_out
    
    return x


def _mtp_params_flatten(params):
    """Flatten MTPParams for JAX pytree."""
    children = (
        params.eh_proj,
        params.layers,
        params.pre_fc_norm_hidden,
        params.pre_fc_norm_embedding,
        params.final_norm,
        params.lm_head,
    )
    aux_data = None
    return children, aux_data


def _mtp_params_unflatten(aux_data, children):
    """Unflatten MTPParams for JAX pytree."""
    return MTPParams(
        eh_proj=children[0],
        layers=children[1],
        pre_fc_norm_hidden=children[2],
        pre_fc_norm_embedding=children[3],
        final_norm=children[4],
        lm_head=children[5],
    )


jax.tree_util.register_pytree_node(MTPParams, _mtp_params_flatten, _mtp_params_unflatten)


# JIT-compiled MTP forward
mtp_forward_jit = jax.jit(mtp_forward, static_argnames=['config'])
