"""MTP layer implementation for Qwen3.5."""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, replace
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.backends import InferenceBackend, select_backend
from nanovllm_jax.kv_cache import AttentionMetadata, KVCacheState
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


def _mtp_forward_hidden(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run the MTP transformer block and return normed/output hidden states."""
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
    return x_normed, x


def _mtp_greedy_top1_token_ids(
    x_normed: jnp.ndarray,
    output_weight: jnp.ndarray,
    config: Qwen3_5Config,
) -> jnp.ndarray:
    impl = str(getattr(config, "mtp_lm_head_greedy_top1_impl", "jax") or "jax").strip().lower()
    if impl in {"triton", "triton_tensorcore", "triton_top1"}:
        from nanovllm_jax.kernels.lm_head_triton import lm_head_greedy_top1_triton

        if x_normed.ndim != 3:
            raise ValueError("Triton MTP greedy top1 expects hidden shape [B, T, H]")
        if int(x_normed.shape[1]) == 1:
            return lm_head_greedy_top1_triton(
                x_normed.astype(output_weight.dtype),
                output_weight,
            ).astype(jnp.int32)
        batch, seq_len, hidden_dim = x_normed.shape
        flat_hidden = x_normed.reshape((batch * seq_len, 1, hidden_dim))
        flat_token_ids = lm_head_greedy_top1_triton(
            flat_hidden.astype(output_weight.dtype),
            output_weight,
        ).astype(jnp.int32)
        return flat_token_ids.reshape((batch, seq_len))
    if impl in {"cutlass", "cutlass_top1", "cutlass_fused_gemm", "fused_gemm"}:
        raise NotImplementedError(
            "lm_head_greedy_top1_impl='cutlass' is not implemented for MTP draft seeding"
        )
    if impl not in {"jax", "", "none"}:
        raise ValueError(f"unsupported MTP greedy top1 implementation: {impl!r}")
    logits = jnp.dot(x_normed, output_weight)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _mtp_decode_activation_dtype(config: Qwen3_5Config) -> jnp.dtype:
    value = str(getattr(config, "lm_head_decode_act_dtype", "fp32") or "fp32").strip().lower()
    if value in {"", "0", "false", "no", "off", "none", "fp32", "float32"}:
        return jnp.float32
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    raise ValueError(f"unsupported MTP decode activation dtype: {value!r}")


def mtp_forward(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
    return_normed_hidden: bool = False,
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
    x_normed, x = _mtp_forward_hidden(
        hidden_state=hidden_state,
        next_token_ids=next_token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )

    # Output projection to vocab. MTP checkpoints may omit lm_head when
    # embeddings are shared with the main model, so fallback to tied
    # embedding weights in that case.
    output_weight = params.lm_head if params.lm_head is not None else embed_tokens.T
    logits = jnp.dot(x_normed, output_weight)

    return logits, x_normed if return_normed_hidden else x


def mtp_forward_token_ids(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
    return_normed_hidden: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """MTP forward pass for greedy draft token ids without materializing logits."""
    x_normed, x = _mtp_forward_hidden(
        hidden_state=hidden_state,
        next_token_ids=next_token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )
    output_weight = params.lm_head if params.lm_head is not None else embed_tokens.T
    x_normed = x_normed.astype(_mtp_decode_activation_dtype(config))
    token_ids = _mtp_greedy_top1_token_ids(x_normed, output_weight, config)
    return token_ids, x_normed if return_normed_hidden else x


def _mtp_forward_hidden_cached(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: jnp.ndarray,
    kv_cache_state: KVCacheState,
    attention_metadata: AttentionMetadata,
    backend: Optional[InferenceBackend] = None,
    is_prefill: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, KVCacheState]:
    """Run the MTP block with a persistent draft-model KV cache.

    vLLM's Qwen3.5 MTP predictor is an autoregressive full-attention decoder
    layer. The draft input ids are shifted, but the MTP positions stay aligned
    with the target hidden positions. This cached path keeps that draft decoder
    state in the same paged block layout as the target model.
    """
    if backend is None:
        backend = select_backend("pure_jax", config=config)

    next_token_embed = embed_tokens[next_token_ids]
    hidden_norm = rms_norm(hidden_state, params.pre_fc_norm_hidden, config.rms_norm_eps)
    embed_norm = rms_norm(next_token_embed, params.pre_fc_norm_embedding, config.rms_norm_eps)
    x = jnp.dot(jnp.concatenate([embed_norm, hidden_norm], axis=-1), params.eh_proj)

    for layer_idx, layer_params in enumerate(params.layers):
        x, kv_cache_state = mtp_layer_forward_cached(
            x,
            layer_params,
            config,
            positions=positions,
            kv_cache_state=kv_cache_state,
            attention_metadata=attention_metadata,
            backend=backend,
            layer_idx=layer_idx,
            is_prefill=is_prefill,
        )

    x_normed = rms_norm(x, params.final_norm, config.rms_norm_eps)
    return x_normed, x, kv_cache_state


def mtp_forward_token_ids_cached(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: jnp.ndarray,
    kv_cache_state: KVCacheState,
    attention_metadata: AttentionMetadata,
    backend: Optional[InferenceBackend] = None,
    return_normed_hidden: bool = False,
    is_prefill: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, KVCacheState]:
    """Greedy MTP token ids while updating the persistent draft KV cache."""
    x_normed, x, kv_cache_state = _mtp_forward_hidden_cached(
        hidden_state=hidden_state,
        next_token_ids=next_token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
        kv_cache_state=kv_cache_state,
        attention_metadata=attention_metadata,
        backend=backend,
        is_prefill=is_prefill,
    )
    output_weight = params.lm_head if params.lm_head is not None else embed_tokens.T
    token_ids = _mtp_greedy_top1_token_ids(
        x_normed.astype(_mtp_decode_activation_dtype(config)),
        output_weight,
        config,
    )
    return token_ids, x_normed if return_normed_hidden else x, kv_cache_state


def mtp_forward_selected_token_ids_cached(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: jnp.ndarray,
    kv_cache_state: KVCacheState,
    attention_metadata: AttentionMetadata,
    selected_indices: jnp.ndarray,
    backend: Optional[InferenceBackend] = None,
    return_normed_hidden: bool = False,
    is_prefill: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, KVCacheState]:
    """Update cached MTP state for a span and score only selected rows.

    The persistent MTP KV cache still needs every committed verifier position,
    but recursive draft generation only consumes one selected hidden state per
    request. Avoid running the vocabulary top-1 GEMM for unused positions.
    """
    x_normed, x, kv_cache_state = _mtp_forward_hidden_cached(
        hidden_state=hidden_state,
        next_token_ids=next_token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
        kv_cache_state=kv_cache_state,
        attention_metadata=attention_metadata,
        backend=backend,
        is_prefill=is_prefill,
    )
    batch, _seq_len, hidden_dim = x_normed.shape
    selected_indices = selected_indices.astype(jnp.int32)
    if selected_indices.shape[0] != batch and batch == 1:
        selected_normed = x_normed[0, selected_indices, :][:, None, :]
    else:
        gather_idx = selected_indices.reshape(batch, 1, 1)
        gather_idx = jnp.broadcast_to(gather_idx, (batch, 1, hidden_dim))
        selected_normed = jnp.take_along_axis(x_normed, gather_idx, axis=1)
    output_weight = params.lm_head if params.lm_head is not None else embed_tokens.T
    token_ids = _mtp_greedy_top1_token_ids(
        selected_normed.astype(_mtp_decode_activation_dtype(config)),
        output_weight,
        config,
    )
    return token_ids, x_normed if return_normed_hidden else x, kv_cache_state


def mtp_forward_last(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
    return_normed_hidden: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """MTP logits and chain hidden for the final position only."""
    x_normed, x = _mtp_forward_hidden(
        hidden_state=hidden_state,
        next_token_ids=next_token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )
    output_weight = params.lm_head if params.lm_head is not None else embed_tokens.T
    last_normed = x_normed[:, -1:, :]
    logits = jnp.dot(last_normed, output_weight)
    chain_hidden = x_normed if return_normed_hidden else x
    return logits, chain_hidden[:, -1:, :]


def mtp_forward_last_token_ids(
    hidden_state: jnp.ndarray,
    next_token_ids: jnp.ndarray,
    embed_tokens: jnp.ndarray,
    params: MTPParams,
    config: Qwen3_5Config,
    positions: Optional[jnp.ndarray] = None,
    return_normed_hidden: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Greedy MTP token IDs and chain hidden for the final position only."""
    x_normed, x = _mtp_forward_hidden(
        hidden_state=hidden_state,
        next_token_ids=next_token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )
    output_weight = params.lm_head if params.lm_head is not None else embed_tokens.T
    last_normed = x_normed[:, -1:, :].astype(_mtp_decode_activation_dtype(config))
    token_ids = _mtp_greedy_top1_token_ids(last_normed, output_weight, config)
    chain_hidden = x_normed if return_normed_hidden else x
    return token_ids, chain_hidden[:, -1:, :]


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
    
    # GQA: group query heads by KV head without materializing repeated K/V.
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    query = query.reshape(
        batch,
        config.num_key_value_heads,
        num_key_value_groups,
        seq_len,
        config.head_dim,
    )
    
    # Attention: query [B, KVH, G, T, D], KV [B, KVH, S, D].
    attn_weights = jnp.einsum("bkgtd,bksd->bkgts", query, k) / jnp.sqrt(config.head_dim)
    attn_weights = attn_weights + mask
    attn_probs = jax.nn.softmax(attn_weights, axis=-1)
    attn_out = jnp.einsum("bkgts,bksd->bkgtd", attn_probs, v)
    attn_out = attn_out.reshape(batch, config.num_attention_heads, seq_len, config.head_dim)
    
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


def mtp_layer_forward_cached(
    x: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    config: Qwen3_5Config,
    positions: jnp.ndarray,
    kv_cache_state: KVCacheState,
    attention_metadata: AttentionMetadata,
    backend: InferenceBackend,
    layer_idx: int = 0,
    is_prefill: bool = False,
) -> Tuple[jnp.ndarray, KVCacheState]:
    """Single cached full-attention MTP decoder layer."""
    batch, seq_len, _ = x.shape
    dtype = config.get_dtype()
    x_cast = x.astype(dtype)

    x_norm = rms_norm(x_cast, params["input_norm"], config.rms_norm_eps)
    attn_out_dim = config.num_attention_heads * config.head_dim
    q_gate = jnp.dot(x_norm, params["q_proj"])
    key_raw = jnp.dot(x_norm, params["k_proj"])
    value_raw = jnp.dot(x_norm, params["v_proj"])
    q_gate = q_gate.reshape(
        batch,
        seq_len,
        config.num_attention_heads,
        2 * config.head_dim,
    )
    query, gate = jnp.split(q_gate, 2, axis=-1)
    gate = gate.reshape(batch, seq_len, -1)
    key = key_raw.reshape(
        batch,
        seq_len,
        config.num_key_value_heads,
        config.head_dim,
    )
    value = value_raw.reshape(
        batch,
        seq_len,
        config.num_key_value_heads,
        config.head_dim,
    )

    query = rms_norm(query, params["q_norm"], config.rms_norm_eps).transpose(0, 2, 1, 3)
    key = rms_norm(key, params["k_norm"], config.rms_norm_eps).transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    query = apply_rope(
        query,
        positions,
        config.head_dim,
        config.rope_theta,
        config.partial_rotary_factor,
        layout="BHTD",
        mrope_section=None,
    )
    key = apply_rope(
        key,
        positions,
        config.head_dim,
        config.rope_theta,
        config.partial_rotary_factor,
        layout="BHTD",
        mrope_section=None,
    )

    query_btnh = query.transpose(0, 2, 1, 3)
    key_btnh = key.transpose(0, 2, 1, 3)
    value_btnh = value.transpose(0, 2, 1, 3)
    cache_storage, attn_out = backend.write_kv_and_attention(
        layer_id=layer_idx,
        query=query_btnh,
        k=key_btnh,
        v=value_btnh,
        cache=kv_cache_state.storage,
        metadata=attention_metadata,
        block_size=config.block_size,
        scale=1.0 / jnp.sqrt(config.head_dim),
        num_key_value_groups=config.num_attention_heads // config.num_key_value_heads,
        is_prefill=is_prefill,
    )
    kv_cache_state = replace(
        kv_cache_state,
        k_cache=cache_storage.k_cache,
        v_cache=cache_storage.v_cache,
        slot_mapping=attention_metadata.slot_mapping,
    )

    attn_out = attn_out * jax.nn.sigmoid(gate)
    attn_out = jnp.dot(attn_out.astype(dtype), params["o_proj"])
    x = x + attn_out

    x_norm = rms_norm(x, params["post_attn_norm"], config.rms_norm_eps)
    gate_proj = jnp.dot(x_norm.astype(dtype), params["gate_proj"])
    up_proj = jnp.dot(x_norm.astype(dtype), params["up_proj"])
    mlp_out = jnp.dot(get_activation(config.hidden_act)(gate_proj) * up_proj, params["down_proj"])
    x = x + mlp_out
    return x, kv_cache_state


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
