"""Qwen 3.5 model implementation in pure JAX - matching HF exactly."""

import jax
import jax.numpy as jnp
from jax import nn, lax
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, replace
from nanovllm_jax.backends import InferenceBackend, select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import rms_norm, apply_rope, repeat_kv, causal_mask, get_activation, l2norm, causal_conv1d_update
from nanovllm_jax.kv_cache import AttentionMetadata, HybridLayerState, KVCacheState, init_linear_attention_states
from nanovllm_jax.conv1d_metal import causal_conv1d_metal


@dataclass
class ModelParams:
    embed_tokens: jnp.ndarray
    layers: List[Dict[str, jnp.ndarray]]
    norm_weight: jnp.ndarray
    lm_head: Optional[jnp.ndarray] = None
    mtp_params: Optional['MTPParams'] = None  # MTP head parameters for speculative decoding


# Register ModelParams as a JAX pytree node for JIT compatibility
def _model_params_flatten(params: ModelParams):
    """Flatten ModelParams into children and auxiliary data."""
    # Flatten all layer dicts into a tuple of arrays
    layer_children = []
    layer_aux = []
    for layer in params.layers:
        # Sort keys for consistent ordering
        keys = sorted(layer.keys())
        layer_aux.append(keys)
        for k in keys:
            layer_children.append(layer[k])
    
    children = (
        params.embed_tokens,
        *layer_children,
        params.norm_weight,
        params.lm_head if params.lm_head is not None else jnp.zeros((1,), dtype=jnp.float16),
        params.mtp_params if params.mtp_params is not None else jnp.zeros((1,), dtype=jnp.float16),
    )
    aux_data = (
        len(params.layers),
        layer_aux,
        params.lm_head is not None,
        params.mtp_params is not None,
    )
    return children, aux_data


def _model_params_unflatten(aux_data, children):
    """Unflatten children and auxiliary data into ModelParams."""
    num_layers, layer_aux, has_lm_head, has_mtp = aux_data
    
    # Reconstruct layers
    layers = []
    child_idx = 1  # Skip embed_tokens
    for layer_keys in layer_aux:
        layer = {}
        for k in layer_keys:
            layer[k] = children[child_idx]
            child_idx += 1
        layers.append(layer)
    
    # Get remaining fields
    norm_weight = children[child_idx]
    child_idx += 1
    lm_head = children[child_idx] if has_lm_head else None
    child_idx += 1
    mtp_params = children[child_idx] if has_mtp else None
    
    return ModelParams(
        embed_tokens=children[0],
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
        mtp_params=mtp_params,
    )


jax.tree_util.register_pytree_node(
    ModelParams,
    _model_params_flatten,
    _model_params_unflatten
)


def init_params(key: jax.Array, config: Qwen3_5Config) -> ModelParams:
    keys = jax.random.split(key, config.num_hidden_layers + 3)
    embed_tokens = jax.random.normal(keys[0], (config.vocab_size, config.hidden_size)) * (config.hidden_size ** -0.5)
    layers = [init_transformer_block(keys[i + 1], config, i) for i in range(config.num_hidden_layers)]
    norm_weight = jnp.ones(config.hidden_size)
    lm_head = None if config.tie_word_embeddings else jax.random.normal(keys[-2], (config.hidden_size, config.vocab_size)) * (config.hidden_size ** -0.5)
    return ModelParams(embed_tokens=embed_tokens, layers=layers, norm_weight=norm_weight, lm_head=lm_head)


def init_transformer_block(key: jax.Array, config: Qwen3_5Config, layer_idx: int) -> Dict[str, jnp.ndarray]:
    keys = jax.random.split(key, 10)
    if config.layer_types[layer_idx] == "full_attention":
        # Qwen3.5 full attention: q_proj outputs [query, gate] each of size num_attention_heads * head_dim
        attn_out_dim = config.num_attention_heads * config.head_dim
        return {
            "q_proj": jax.random.normal(keys[0], (config.hidden_size, attn_out_dim * 2)) * (config.hidden_size ** -0.5),
            "k_proj": jax.random.normal(keys[1], (config.hidden_size, config.num_key_value_heads * config.head_dim)) * (config.hidden_size ** -0.5),
            "v_proj": jax.random.normal(keys[2], (config.hidden_size, config.num_key_value_heads * config.head_dim)) * (config.hidden_size ** -0.5),
            "o_proj": jax.random.normal(keys[3], (attn_out_dim, config.hidden_size)) * (config.hidden_size ** -0.5),
            "q_norm": jnp.ones((config.num_attention_heads, config.head_dim)),
            "k_norm": jnp.ones((config.num_key_value_heads, config.head_dim)),
            "input_norm": jnp.ones(config.hidden_size),
            "post_attn_norm": jnp.ones(config.hidden_size),
            "gate_proj": jax.random.normal(keys[5], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5),
            "up_proj": jax.random.normal(keys[6], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5),
            "down_proj": jax.random.normal(keys[7], (config.intermediate_size, config.hidden_size)) * (config.hidden_size ** -0.5),
            "ffn_norm": jnp.ones(config.hidden_size),
        }
    else:
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        return {
            "input_norm": jnp.ones(config.hidden_size),
            "in_proj_qkv": jax.random.normal(keys[0], (config.hidden_size, conv_dim)) * (config.hidden_size ** -0.5),
            "in_proj_z": jax.random.normal(keys[1], (config.hidden_size, value_dim)) * (config.hidden_size ** -0.5),
            "in_proj_a": jax.random.normal(keys[2], (config.hidden_size, config.linear_num_value_heads)) * (config.hidden_size ** -0.5),
            "in_proj_b": jax.random.normal(keys[3], (config.hidden_size, config.linear_num_value_heads)) * (config.hidden_size ** -0.5),
            "conv1d_weight": jax.random.normal(keys[4], (conv_dim, config.linear_conv_kernel_size)) * 0.02,
            "dt_bias": jnp.ones(config.linear_num_value_heads),
            "A": jnp.exp(jnp.full(config.linear_num_value_heads, 0.0)),
            "norm_weight": jnp.ones(config.linear_value_head_dim),
            "out_proj": jax.random.normal(keys[6], (value_dim, config.hidden_size)) * (config.hidden_size ** -0.5),
            "gate_proj": jax.random.normal(keys[5], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5),
            "up_proj": jax.random.normal(keys[6], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5),
            "down_proj": jax.random.normal(keys[7], (config.intermediate_size, config.hidden_size)) * (config.hidden_size ** -0.5),
            "ffn_norm": jnp.ones(config.hidden_size),
        }


def jax_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, 
                                output_final_state=False, use_qk_l2norm_in_kernel=False):
    """
    JAX implementation of chunk gated delta rule (matching HF torch_chunk_gated_delta_rule).
    Input shapes: [B, H, T, D] for query/key/value, [B, H, T] for g/beta
    Output shape: [B, H, T, D]
    """
    import jax
    
    initial_dtype = query.dtype
    
    # Apply L2 norm if requested
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    
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
    
    # Compute decay: cumulative sum of g within each chunk
    # Use Metal-compatible cumsum if on Metal backend
    if jax.default_backend() == 'METAL':
        from nanovllm_jax.metal_ops import cumsum_metal
        g_cumsum = cumsum_metal(g_chunks, axis=-1)
    else:
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
    
    # k_cumdecay = attn @ (k_beta * exp(g))
    k_cumdecay = jnp.einsum('bhnct,bhntv->bhncv', attn, k_beta_chunks * jnp.exp(g_chunks)[..., None])
    
    # Initialize state [B, H, K, V]
    if initial_state is None:
        state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)
    
    # Process each chunk sequentially
    def process_chunk(carry, i):
        state = carry
        q_i = query_chunks[:, :, i]      # [B, H, cs, K]
        k_i = key_chunks[:, :, i]        # [B, H, cs, K]
        v_i = value_transformed[:, :, i] # [B, H, cs, V]
        decay_mask_i = decay_mask[:, :, i]  # [B, H, cs, cs]
        k_cumdecay_i = k_cumdecay[:, :, i]  # [B, H, cs, V]
        g_cumsum_i = g_cumsum[:, :, i]    # [B, H, cs] - use cumsum version!
        
        # Within-chunk attention: attn = (q @ k.T * decay_mask), masked to strict upper triangle
        attn_i = jnp.einsum('bhck,bhdk->bhcd', q_i, k_i) * decay_mask_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)
        
        # v_prime = k_cumdecay @ state
        # k_cumdecay_i: [B, H, cs, V], state: [B, H, K, V]
        # v_prime[b,h,c,v] = sum_k(k_cumdecay_i[b,h,c,k] * state[b,h,k,v])
        v_prime = jnp.einsum('bhck,bhkv->bhcv', k_cumdecay_i, state)
        
        # v_new = v_i - v_prime
        v_new = v_i - v_prime
        
        # attn_inter = (q * exp(g)) @ state
        # q_i * exp(g_cumsum_i): [B, H, cs, K]
        # result: [B, H, cs, V] = sum_K(q[b,h,c,k] * exp(g_cumsum[b,h,c]) * state[b,h,k,v])
        attn_inter = jnp.einsum('bhck,bhkv->bhcv', q_i * jnp.exp(g_cumsum_i)[..., None], state)
        
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
        # state_update[b,h,k,v] = sum_c(k_weighted[b,h,c,k] * v_new[b,h,c,v])
        state_update = jnp.einsum('bhck,bhcv->bhkv', k_weighted, v_new)
        state = state * jnp.exp(g_cumsum_i[..., -1, None, None]) + state_update
        
        return state, core_attn_out_i
    
    final_state, core_attn_out_chunks = lax.scan(process_chunk, state, jnp.arange(n_chunks))
    
    # Reshape output back to [B, H, T, V]
    core_attn_out = core_attn_out_chunks.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :seq_len]  # Remove padding
    
    if not output_final_state:
        final_state = None
    
    core_attn_out = core_attn_out.astype(initial_dtype)
    return core_attn_out, final_state


def jax_recurrent_gated_delta_rule(
    query, key, value, g, beta, 
    initial_state=None,
    use_qk_l2norm_in_kernel=False
):
    """
    JAX implementation of recurrent gated delta rule (matching HF torch_recurrent_gated_delta_rule).
    Input shapes: [B, H, T, D] for query/key/value, [B, H, T] for g/beta
    Output shape: [B, H, T, D]
    State shape: [B, H, k_head_dim, v_head_dim] (per-head state, matching HF)
    """
    initial_dtype = query.dtype
    
    # Apply L2 norm if requested
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    
    # Keep as [B, H, T, D], convert to float32
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    
    batch, num_heads, time_dim, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    
    # Scale query
    query = query * (1.0 / jnp.sqrt(k_head_dim))
    
    # Initialize state: [B, H, K, V] (per-head state, matching HF)
    if initial_state is None:
        state = jnp.zeros((batch, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)
    
    # Process each time step
    def step_fn(carry, t):
        state = carry  # [B, H, K, V]
        
        # Get data for this time step: indexing dim 2 (T dimension)
        q_t = query[:, :, t, :]      # [B, H, D]
        k_t = key[:, :, t, :]        # [B, H, D]
        v_t = value[:, :, t, :]      # [B, H, V]
        g_t = jnp.exp(g[:, :, t])    # [B, H]
        beta_t = beta[:, :, t]       # [B, H]
        
        # Reshape for broadcasting
        g_t_exp = g_t[:, :, None, None]    # [B, H, 1, 1]
        beta_t_exp = beta_t[:, :, None]    # [B, H, 1]
        k_t_exp = k_t[:, :, :, None]       # [B, H, D, 1]
        q_t_exp = q_t[:, :, :, None]       # [B, H, D, 1]
        
        # Decay state: state * exp(g_t)
        state = state * g_t_exp
        
        # kv_mem = (state * k_t[..., None]).sum(-2)
        # state: [B, H, K, V], k_t: [B, H, K]
        # kv_mem: [B, H, V]
        kv_mem = jnp.einsum('bhkv,bhk->bhv', state, k_t)
        
        # delta = (v_t - kv_mem) * beta_t
        delta = (v_t - kv_mem) * beta_t_exp  # [B, H, V]
        
        # state = state + k_t[..., None] * delta[..., None, :]
        state = state + k_t_exp * delta[:, :, None, :]
        
        # output_t = (state * q_t[..., None]).sum(-2)
        out_t = jnp.einsum('bhkv,bhk->bhv', state, q_t)
        
        return state, out_t
    
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
    backend: Optional[InferenceBackend] = None,
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
        backend = select_backend("pure_jax")
    
    # Cast to target dtype (bfloat16 for CPU/CUDA, float16 for Metal)
    dtype = config.get_dtype()
    x_cast = x.astype(dtype)
    
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    v_heads_per_k = config.linear_num_value_heads // config.linear_num_key_heads
    conv_dim = key_dim * 2 + value_dim
    
    # Check if we can use cached states
    use_cached = (
        not is_prefill and 
        hybrid_state is not None and 
        hybrid_state.conv_state is not None and
        seq_len == 1
    )
    use_cached_prefill = (
        is_prefill
        and hybrid_state is not None
        and hybrid_state.conv_state is not None
        and hybrid_state.recurrent_state is not None
    )
    recurrent_prefill_threshold = int(getattr(config, "linear_recurrent_prefill_threshold", config.linear_chunk_size))
    use_recurrent_prefill = use_cached_prefill and seq_len <= recurrent_prefill_threshold
    linear_layer_idx = len([l for l in config.linear_attn_layers if l < layer_idx])
    
    # === PROJECTIONS (same for both modes) ===
    mixed_qkv = jnp.dot(x_cast, params["in_proj_qkv"])
    z = jnp.dot(x_cast, params["in_proj_z"]).reshape(batch, seq_len, -1)
    a = jnp.dot(x_cast, params["in_proj_a"]).reshape(batch, seq_len, config.linear_num_value_heads)
    b = jnp.dot(x_cast, params["in_proj_b"]).reshape(batch, seq_len, config.linear_num_value_heads)
    
    if use_cached:
        # === DECODE MODE ===
        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # [B, D, 1]
        
        # 1. Convolution update - use per-layer conv_state
        # conv_state shape: [batch, num_linear_layers, conv_dim, kernel_size]
        layer_conv_state = hybrid_state.conv_state[:, linear_layer_idx]  # [batch, conv_dim, kernel_size]
        conv_out, new_layer_conv_state = causal_conv1d_update(
            mixed_qkv_t,
            layer_conv_state,
            params["conv1d_weight"].reshape(conv_dim, config.linear_conv_kernel_size),
            params.get("conv1d_bias"),
            "silu"
        )
        conv_out = conv_out.transpose(0, 2, 1)  # [B, 1, D]
        
        # 2. Split q, k, v
        query = conv_out[:, :, :key_dim].reshape(batch, 1, config.linear_num_key_heads, config.linear_key_head_dim)
        key = conv_out[:, :, key_dim:key_dim*2].reshape(batch, 1, config.linear_num_key_heads, config.linear_key_head_dim)
        value = conv_out[:, :, key_dim*2:].reshape(batch, 1, config.linear_num_value_heads, config.linear_value_head_dim)
        
        # 3. Compute gates
        beta = nn.sigmoid(b)  # [B, 1, H_v]
        g = -params["A"] * nn.softplus(a + params["dt_bias"])  # [B, 1, H_v]
        
        # 4. Repeat for GQA
        if v_heads_per_k > 1:
            query = jnp.repeat(query, v_heads_per_k, axis=2)
            key = jnp.repeat(key, v_heads_per_k, axis=2)
        
        # 5. Transpose to [B, H, T, D] format (keep T=1 dimension)
        query = query.transpose(0, 2, 1, 3)  # [B, H, 1, D_k]
        key = key.transpose(0, 2, 1, 3)  # [B, H, 1, D_k]
        value = value.transpose(0, 2, 1, 3)  # [B, H, 1, D_v]
        g = g.transpose(0, 2, 1)  # [B, H, 1]
        beta = beta.transpose(0, 2, 1)  # [B, H, 1]
        
        # 6. Recurrent update
        # recurrent_state shape: [batch, num_layers, num_heads, k_dim, v_dim]
        # Extract recurrent state for this layer: [batch, num_heads, k_dim, v_dim]
        # linear_layer_idx computed above
        initial_recurrent = hybrid_state.recurrent_state[:, linear_layer_idx] if hybrid_state.recurrent_state is not None else None

        core_attn_out, new_recurrent_state_single = backend.gated_delta_decode(
            query, key, value, g, beta,
            initial_state=initial_recurrent,
            use_qk_l2norm_in_kernel=True,
        )
        # new_recurrent_state_single has shape [batch, num_heads, k_dim, v_dim]
        
        # Update cache with new recurrent state and conv state for this layer
        if hybrid_state.recurrent_state is not None:
            new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(new_recurrent_state_single)
            new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(new_layer_conv_state)
        else:
            new_recurrent_state = new_recurrent_state_single[jnp.newaxis, :, :, :, :]  # Add layer dim
            new_conv_state = new_layer_conv_state[jnp.newaxis, :, :, :]  # Add layer dim
        
        hybrid_state = replace(
            hybrid_state,
            conv_state=new_conv_state,
            recurrent_state=new_recurrent_state,
        )
        
        # core_attn_out is [B, H, T=1, D_v] - transpose to [B, T, H, D_v] for reshaping
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)  # [B, 1, H, D_v]
        
    else:
        # === PREFILL MODE (Metal-compatible implementation) ===
        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # [B, D, T]
        if use_cached_prefill:
            layer_conv_state = hybrid_state.conv_state[:, linear_layer_idx]
            conv_input = jnp.concatenate([layer_conv_state, mixed_qkv_t], axis=-1)
            conv_out = causal_conv1d_metal(
                conv_input,
                params["conv1d_weight"],
                params.get("conv1d_bias"),
                activation="silu",
            )[:, :, -seq_len:]
        else:
            # Use Metal-compatible conv1d (no lax.conv_general_dilated)
            conv_out = causal_conv1d_metal(
                mixed_qkv_t,
                params["conv1d_weight"],
                params.get("conv1d_bias"),
                activation="silu",
            )
        conv_out = conv_out.transpose(0, 2, 1)  # [B, T, D]
        
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
        
        initial_recurrent = (
            hybrid_state.recurrent_state[:, linear_layer_idx]
            if use_cached_prefill
            else None
        )
        if use_recurrent_prefill:
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
            # Chunked rule is the right path for prompt/chunk prefill. Short
            # cached suffixes use the recurrent path above to avoid padding a
            # tiny verifier suffix up to the chunk size.
            core_attn_out, final_state = backend.gated_delta_prefill(
                query, key, value, g, beta,
                chunk_size=config.linear_chunk_size,
                initial_state=initial_recurrent,
                use_qk_l2norm_in_kernel=True,
            )
        
        # Save final state to cache for decode mode
        if hybrid_state is not None:
            # Extract the last real kernel_size inputs. Bucket padding is not
            # part of the convolution history used by recurrent decode.
            kernel_size = config.linear_conv_kernel_size
            if valid_token_mask is not None:
                valid_lens = valid_token_mask.astype(jnp.int32).sum(axis=1)
                gather_idx = valid_lens[:, None] - kernel_size + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
                gather_valid = gather_idx >= 0
                gather_idx = jnp.clip(gather_idx, 0, max(seq_len - 1, 0))
                gather_idx = jnp.broadcast_to(gather_idx[:, None, :], (batch, conv_dim, kernel_size))
                layer_conv_state = jnp.take_along_axis(mixed_qkv_t, gather_idx, axis=2)
                layer_conv_state = jnp.where(gather_valid[:, None, :], layer_conv_state, 0.0)
            elif seq_len >= kernel_size:
                layer_conv_state = mixed_qkv_t[:, :, -kernel_size:]  # [B, D, K]
            else:
                # If sequence is shorter than kernel_size, pad with zeros
                pad_width = kernel_size - seq_len
                layer_conv_state = jnp.pad(mixed_qkv_t, ((0, 0), (0, 0), (pad_width, 0)))
            
            # Compute layer index for this linear attention layer
            # Update recurrent state and conv_state at this layer index
            # recurrent_state shape: [batch, num_linear_layers, num_heads, k_dim, v_dim]
            # final_state shape: [batch, num_heads, k_dim, v_dim]
            # conv_state shape: [batch, num_linear_layers, conv_dim, kernel_size]
            if hybrid_state.recurrent_state is not None:
                if use_cached_prefill:
                    kernel_size = config.linear_conv_kernel_size
                    conv_input = jnp.concatenate([hybrid_state.conv_state[:, linear_layer_idx], mixed_qkv_t], axis=-1)
                    if valid_token_mask is not None:
                        valid_lens = valid_token_mask.astype(jnp.int32).sum(axis=1)
                    else:
                        valid_lens = jnp.full((batch,), seq_len, dtype=jnp.int32)
                    gather_idx = valid_lens[:, None] + jnp.arange(kernel_size, dtype=jnp.int32)[None, :]
                    gather_idx = jnp.broadcast_to(gather_idx[:, None, :], (batch, conv_dim, kernel_size))
                    layer_conv_state = jnp.take_along_axis(conv_input, gather_idx, axis=2)
                new_recurrent_state = hybrid_state.recurrent_state.at[:, linear_layer_idx].set(final_state)
                new_conv_state = hybrid_state.conv_state.at[:, linear_layer_idx].set(layer_conv_state.astype(dtype))
            else:
                # Should not happen if init_linear_attention_states was called
                new_recurrent_state = final_state[jnp.newaxis, :, :, :, :]
                new_conv_state = layer_conv_state.astype(dtype)[jnp.newaxis, :, :, :]
            
            hybrid_state = replace(
                hybrid_state,
                conv_state=new_conv_state,
                recurrent_state=new_recurrent_state,
            )
        
        # Output is [B, H, T, D] - transpose to [B, T, H, D] for reshaping
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)  # [B, H, T, D] -> [B, T, H, D]
    
    # === OUTPUT PROCESSING (same for both modes) ===
    # Reshape to apply per-head gated norm
    core_attn_out = core_attn_out.reshape(batch * seq_len, -1, config.linear_value_head_dim)  # [B*T, H, D]
    z = z.reshape(batch * seq_len, -1, config.linear_value_head_dim)  # [B*T, H, D]
    
    # Apply gated RMSNorm per head (matching HF: compute in input dtype)
    input_dtype = core_attn_out.dtype
    variance = (core_attn_out ** 2).mean(-1, keepdims=True)
    core_attn_out = core_attn_out * jax.lax.rsqrt(variance + config.rms_norm_eps)
    core_attn_out = params["norm_weight"] * core_attn_out
    core_attn_out = core_attn_out * nn.silu(z)
    
    # Reshape back and project
    core_attn_out = core_attn_out.reshape(batch, seq_len, -1)
    attn_out = jnp.dot(core_attn_out, params["out_proj"])
    
    if use_cached:
        return attn_out, hybrid_state
    elif hybrid_state is not None:
        # Prefill mode with cache - return state for decode
        return attn_out, hybrid_state
    else:
        # No cache - just return output
        return attn_out


def full_attention_block(
    x,
    params,
    positions,
    mask,
    config,
    kv_cache_state: Optional[KVCacheState] = None,
    is_prefill: bool = True,
    layer_idx: int = 0,
    attention_metadata: Optional[AttentionMetadata] = None,
    backend: Optional[InferenceBackend] = None,
):
    """Full attention block with optional KV cache support.
    
    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        params: Layer parameters
        positions: Position IDs [batch, seq_len]
        mask: Causal mask [seq_len, seq_len]
        config: Model config
        kv_cache_state: Optional KV cache state (None for no cache)
        is_prefill: Whether this is prefill (vs decode)
        layer_idx: Layer index for per-layer KV cache
    
    Returns:
        tuple: (output, updated_kv_cache_state)
    """
    batch, seq_len, _ = x.shape
    
    # Cast to target dtype (bfloat16 for CPU/CUDA, float16 for Metal)
    dtype = config.get_dtype()
    x_cast = x.astype(dtype)
    
    # Qwen3.5 full attention uses fused Q + gate projection
    # q_proj: [hidden_size, hidden_size] -> output split into query and gate
    # query: [batch, seq_len, num_attention_heads * head_dim]
    # gate: [batch, seq_len, num_attention_heads * head_dim]
    q_gate = jnp.dot(x_cast, params["q_proj"])
    attn_out_dim = config.num_attention_heads * config.head_dim
    
    q_gate_reshaped = q_gate.reshape(batch, seq_len, config.num_attention_heads, 2 * config.head_dim)
    query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
    gate = gate.reshape(batch, seq_len, -1)
    
    k = jnp.dot(x_cast, params["k_proj"]).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    v = jnp.dot(x_cast, params["v_proj"]).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    
    # Apply RMSNorm BEFORE transpose (on head dimension, in [B, T, H, D] layout)
    query = rms_norm(query, params["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, params["k_norm"], config.rms_norm_eps)
    
    # Transpose to [B, H, T, D]
    query = query.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    # Apply RoPE (now in [B, H, T, D] layout)
    query = apply_rope(query, positions, config.head_dim, config.rope_theta, config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
    k = apply_rope(k, positions, config.head_dim, config.rope_theta, config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
    
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    
    if kv_cache_state is not None:
        if backend is None:
            backend = select_backend("pure_jax")

        # Transpose K, V back to [B, T, K, H] for cache storage
        k_cache_input = k.transpose(0, 2, 1, 3)  # [B, T, K, H]
        v_cache_input = v.transpose(0, 2, 1, 3)  # [B, T, K, H]

        metadata = attention_metadata
        if metadata is None:
            metadata_positions = positions[0] if positions.ndim == 3 else positions
            metadata = backend.build_attention_metadata(
                positions=metadata_positions,
                block_tables=kv_cache_state.block_table,
                seq_lens=kv_cache_state.kv_lens,
                block_size=config.block_size,
                is_prefill=is_prefill,
            )
        cache_storage = backend.write_kv(
            layer_id=layer_idx,
            k=k_cache_input,
            v=v_cache_input,
            cache=kv_cache_state.storage,
            metadata=metadata,
        )
        
        # query is currently [batch, num_heads, seq_len, head_dim] (BHTD)
        # Backend attention expects [batch, seq_len, num_heads, head_dim] (BTNH)
        query_btnh = query.transpose(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]

        out = backend.attention(
            layer_id=layer_idx,
            query=query_btnh,
            cache=cache_storage,
            metadata=metadata,
            block_size=config.block_size,
            scale=1.0 / jnp.sqrt(config.head_dim),
            num_key_value_groups=num_key_value_groups,
            is_prefill=is_prefill,
        )
        
        # Reshape out to [batch, seq_len, hidden_dim]
        # For prefill: out is [batch, seq_len, hidden_dim]
        # For decode: out is [batch, 1, hidden_dim]
        # Both are already in the correct format
        
        # Update KV cache state (preserve linear attention states)
        kv_cache_state = replace(
            kv_cache_state,
            k_cache=cache_storage.k_cache,
            v_cache=cache_storage.v_cache,
            slot_mapping=metadata.slot_mapping,
        )
    else:
        # No cache - standard attention (for prefill without caching)
        k = jnp.repeat(k, num_key_value_groups, axis=1)
        v = jnp.repeat(v, num_key_value_groups, axis=1)
        
        attn = nn.softmax(jnp.einsum("bhtd,bhsd->bhts", query, k) / jnp.sqrt(config.head_dim) + mask[None, None, :, :].astype(query.dtype), -1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    out = out * nn.sigmoid(gate)
    out = jnp.dot(out, params["o_proj"])
    
    return out, kv_cache_state


def transformer_block(
    x,
    params,
    positions,
    mask=None,
    layer_idx=0,
    config=None,
    kv_cache_state=None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    is_prefill=True,
    backend: Optional[InferenceBackend] = None,
):
    """Matches HF Qwen3_5DecoderLayer - applies norms and residuals.
    
    Args:
        x: Input tensor
        params: Layer parameters
        positions: Position IDs
        mask: Causal mask
        layer_idx: Layer index
        config: Model config
        kv_cache_state: Optional KV cache state
        is_prefill: Whether this is prefill
    
    Returns:
        tuple: (output, updated_kv_cache_state, updated_hybrid_state)
    """
    residual = x
    
    # Apply input_layernorm (both full attention and linear attention)
    # HF applies input_layernorm before both layer types
    x = rms_norm(x, params["input_norm"], config.rms_norm_eps)
    
    valid_token_mask = None
    if attention_metadata is not None and is_prefill:
        query_lens = jnp.diff(attention_metadata.query_start_loc).astype(jnp.int32)
        valid_token_mask = jnp.arange(x.shape[1], dtype=jnp.int32)[None, :] < query_lens[:, None]

    # Apply attention/linear_attn
    if config.layer_types[layer_idx] == "full_attention":
        x, kv_cache_state = full_attention_block(
            x,
            params,
            positions,
            mask,
            config,
            kv_cache_state,
            is_prefill,
            layer_idx=layer_idx,
            attention_metadata=attention_metadata,
            backend=backend,
        )
    else:
        # Linear attention with decode mode support
        result = gated_deltanet_block(
            x, params, positions, config, layer_idx,
            is_prefill=is_prefill,
            hybrid_state=hybrid_state,
            valid_token_mask=valid_token_mask,
            backend=backend,
        )
        if isinstance(result, tuple):
            x, hybrid_state = result
        else:
            x = result
    
    # Add residual
    x = residual + x
    
    # MLP path
    residual = x
    x = rms_norm(x, params["ffn_norm"], config.rms_norm_eps)
    
    # MLP computation (stays in bfloat16)
    gate = jnp.dot(x, params["gate_proj"])
    up = jnp.dot(x, params["up_proj"])
    x = get_activation(config.hidden_act)(gate) * up
    x = jnp.dot(x, params["down_proj"])
    
    x = residual + x
    
    return x, kv_cache_state, hybrid_state


def forward_step(
    tokens,
    params,
    config,
    *,
    positions=None,
    kv_cache_state: Optional[KVCacheState] = None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    is_prefill: bool = True,
    return_hidden: bool = False,
    last_logits_only: bool = False,
    logit_positions: Optional[jnp.ndarray] = None,
    backend: Optional[InferenceBackend] = None,
):
    """Canonical forward step shared by cached and non-cached inference paths."""
    batch, seq_len = tokens.shape
    dtype = config.get_dtype()
    x = params.embed_tokens[tokens].astype(dtype)

    if positions is None:
        positions_2d = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
    elif positions.ndim == 3:
        positions_2d = positions[0]
    else:
        positions_2d = positions
    positions_mrope = jnp.stack([positions_2d, positions_2d, positions_2d], axis=0)
    mask = causal_mask(seq_len, seq_len)

    for i, lp in enumerate(params.layers):
        x, kv_cache_state, hybrid_state = transformer_block(
            x,
            lp,
            positions_mrope,
            mask,
            i,
            config,
            kv_cache_state,
            attention_metadata=attention_metadata,
            hybrid_state=hybrid_state,
            is_prefill=is_prefill,
            backend=backend,
        )

    hidden_pre = x
    x = rms_norm(x, params.norm_weight, config.rms_norm_eps)

    if return_hidden:
        return hidden_pre, kv_cache_state, hybrid_state

    x = x.astype(jnp.float32)
    if last_logits_only:
        if logit_positions is None:
            x = x[:, -1:, :]
        else:
            gather_idx = jnp.clip(logit_positions, 0, seq_len - 1).astype(jnp.int32)
            gather_idx = gather_idx[:, None, None]
            gather_idx = jnp.broadcast_to(gather_idx, (batch, 1, x.shape[-1]))
            x = jnp.take_along_axis(x, gather_idx, axis=1)
    logits = jnp.dot(x, params.lm_head) if params.lm_head is not None else jnp.dot(x, params.embed_tokens.T)
    return logits, kv_cache_state, hybrid_state


def forward(
    tokens,
    params,
    config,
    kv_cache_state=None,
    is_prefill=True,
    return_hidden=False,
    last_logits_only: bool = False,
    logit_positions: Optional[jnp.ndarray] = None,
    positions=None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    backend: Optional[InferenceBackend] = None,
):
    """Compatibility wrapper over the canonical forward step."""
    if is_prefill and kv_cache_state is not None and hybrid_state is None:
        kv_cache_state = init_linear_attention_states(
            kv_cache_state,
            config,
            batch_size=tokens.shape[0],
        )
        hybrid_state = kv_cache_state.hybrid_state
    elif hybrid_state is None and kv_cache_state is not None:
        hybrid_state = kv_cache_state.hybrid_state

    result, updated_kv_state, updated_hybrid_state = forward_step(
        tokens,
        params,
        config,
        positions=positions,
        kv_cache_state=kv_cache_state,
        attention_metadata=attention_metadata,
        hybrid_state=hybrid_state,
        is_prefill=is_prefill,
        return_hidden=return_hidden,
        last_logits_only=last_logits_only,
        logit_positions=logit_positions,
        backend=backend,
    )

    if updated_kv_state is not None and updated_hybrid_state is not None:
        updated_kv_state = replace(
            updated_kv_state,
            conv_state=updated_hybrid_state.conv_state,
            recurrent_state=updated_hybrid_state.recurrent_state,
        )

    return result, updated_kv_state


class Qwen3_5:
    def __init__(self, config, key):
        self.config, self.params = config, init_params(key, config)
    
    def forward(self, tokens, kv_cache_state=None, is_prefill=True, backend: Optional[InferenceBackend] = None):
        """Forward pass with optional KV cache."""
        return forward(
            tokens,
            self.params,
            self.config,
            kv_cache_state=kv_cache_state,
            is_prefill=is_prefill,
            backend=backend,
        )
