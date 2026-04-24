"""JIT-compiled forward pass for maximum performance."""

import jax
import jax.numpy as jnp
from jax import nn, lax
from typing import Tuple, Optional
from functools import partial

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.layers import rms_norm, apply_rope, causal_mask, get_activation, l2norm, causal_conv1d_update
from nanovllm_jax.kv_cache import KVCacheState, update_kv_cache, paged_attention, paged_attention_decode, init_linear_attention_states
from nanovllm_jax.conv1d_metal import causal_conv1d_metal


def _jax_recurrent_gated_delta_rule_jit(
    query, key, value, g, beta, 
    initial_state=None,
    use_qk_l2norm_in_kernel=False
):
    """JIT-optimized recurrent gated delta rule."""
    initial_dtype = query.dtype
    
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    
    batch, num_heads, time_dim, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    
    query = query * (1.0 / jnp.sqrt(k_head_dim))
    
    if initial_state is None:
        state = jnp.zeros((batch, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)
    
    def step_fn(carry, t):
        state = carry
        
        q_t = query[:, :, t, :]
        k_t = key[:, :, t, :]
        v_t = value[:, :, t, :]
        g_t = jnp.exp(g[:, :, t])
        beta_t = beta[:, :, t]
        
        g_t_exp = g_t[:, :, None, None]
        beta_t_exp = beta_t[:, :, None]
        k_t_exp = k_t[:, :, :, None]
        
        state = state * g_t_exp
        
        kv_mem = jnp.einsum('bhkv,bhk->bhv', state, k_t)
        
        delta = (v_t - kv_mem) * beta_t_exp
        
        state = state + k_t_exp * delta[:, :, None, :]
        
        out_t = jnp.einsum('bhkv,bhk->bhv', state, q_t)
        
        return state, out_t
    
    final_state, all_outputs = lax.scan(step_fn, state, jnp.arange(time_dim))
    
    output = all_outputs.transpose(1, 2, 0, 3)
    
    output = output.astype(initial_dtype)
    return output, final_state


def _jax_chunk_gated_delta_rule_jit(query, key, value, g, beta, chunk_size=64, initial_state=None, 
                                output_final_state=False, use_qk_l2norm_in_kernel=False):
    """JIT-optimized chunk gated delta rule."""
    initial_dtype = query.dtype
    
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    g = g.astype(jnp.float32)
    
    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
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
    
    g_cumsum = jnp.cumsum(g_chunks, axis=-1)
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


@partial(jax.jit, static_argnums=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
def _gated_deltanet_block_jit(
    x,
    params_tuple,
    linear_num_key_heads,
    linear_key_head_dim,
    linear_num_value_heads,
    linear_value_head_dim,
    linear_conv_kernel_size,
    linear_chunk_size,
    rms_norm_eps,
    dtype_code,
    is_prefill,
    layer_idx,
    conv_state=None,
    recurrent_state=None,
    slot_mapping=None,
):
    """JIT-compiled gated deltanet block.
    
    Args:
        dtype_code: 0=float16, 1=bfloat16, 2=float32 (static)
    """
    dtype = [jnp.float16, jnp.bfloat16, jnp.float32][dtype_code]
    
    in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d_weight, conv1d_bias, dt_bias, A, norm_weight, out_proj = params_tuple
    
    batch, seq_len, _ = x.shape
    
    x_cast = x.astype(dtype)
    
    key_dim = linear_num_key_heads * linear_key_head_dim
    value_dim = linear_num_value_heads * linear_value_head_dim
    v_heads_per_k = linear_num_value_heads // linear_num_key_heads
    conv_dim = key_dim * 2 + value_dim
    
    mixed_qkv = jnp.dot(x_cast, in_proj_qkv)
    z = jnp.dot(x_cast, in_proj_z).reshape(batch, seq_len, -1)
    a = jnp.dot(x_cast, in_proj_a).reshape(batch, seq_len, linear_num_value_heads)
    b = jnp.dot(x_cast, in_proj_b).reshape(batch, seq_len, linear_num_value_heads)
    
    use_cached = (not is_prefill and conv_state is not None and seq_len == 1)
    
    if use_cached:
        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)
        
        conv_out, new_conv_state = causal_conv1d_update(
            mixed_qkv_t,
            conv_state,
            conv1d_weight.reshape(conv_dim, linear_conv_kernel_size),
            conv1d_bias,
            "silu"
        )
        conv_out = conv_out.transpose(0, 2, 1)
        
        query = conv_out[:, :, :key_dim].reshape(batch, 1, linear_num_key_heads, linear_key_head_dim)
        key = conv_out[:, :, key_dim:key_dim*2].reshape(batch, 1, linear_num_key_heads, linear_key_head_dim)
        value = conv_out[:, :, key_dim*2:].reshape(batch, 1, linear_num_value_heads, linear_value_head_dim)
        
        beta = nn.sigmoid(b)
        g = -A * nn.softplus(a + dt_bias)
        
        if v_heads_per_k > 1:
            query = jnp.repeat(query, v_heads_per_k, axis=2)
            key = jnp.repeat(key, v_heads_per_k, axis=2)
        
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        g = g.transpose(0, 2, 1)
        beta = beta.transpose(0, 2, 1)
        
        core_attn_out, new_recurrent_state = _jax_recurrent_gated_delta_rule_jit(
            query, key, value, g, beta,
            initial_state=recurrent_state,
            use_qk_l2norm_in_kernel=True
        )
        
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)
    else:
        mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)
        
        conv_out = causal_conv1d_metal(
            mixed_qkv_t,
            conv1d_weight,
            activation="silu"
        )
        conv_out = conv_out.transpose(0, 2, 1)
        
        query = conv_out[:, :, :key_dim].reshape(batch, seq_len, linear_num_key_heads, linear_key_head_dim)
        key = conv_out[:, :, key_dim:key_dim*2].reshape(batch, seq_len, linear_num_key_heads, linear_key_head_dim)
        value = conv_out[:, :, key_dim*2:].reshape(batch, seq_len, linear_num_value_heads, linear_value_head_dim)
        
        beta = nn.sigmoid(b)
        g = -A * nn.softplus(a + dt_bias)
        
        if v_heads_per_k > 1:
            query = jnp.repeat(query, v_heads_per_k, axis=2)
            key = jnp.repeat(key, v_heads_per_k, axis=2)
        
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        g = g.transpose(0, 2, 1)
        beta = beta.transpose(0, 2, 1)
        
        core_attn_out, final_state = _jax_chunk_gated_delta_rule_jit(
            query, key, value, g, beta, 
            chunk_size=linear_chunk_size,
            use_qk_l2norm_in_kernel=True,
            output_final_state=True
        )
        
        new_conv_state = None
        new_recurrent_state = None
        
        core_attn_out = core_attn_out.transpose(0, 2, 1, 3)
    
    core_attn_out = core_attn_out.reshape(batch * seq_len, -1, linear_value_head_dim)
    z = z.reshape(batch * seq_len, -1, linear_value_head_dim)
    
    variance = (core_attn_out ** 2).mean(-1, keepdims=True)
    core_attn_out = core_attn_out * jax.lax.rsqrt(variance + rms_norm_eps)
    core_attn_out = norm_weight * core_attn_out
    core_attn_out = core_attn_out * nn.silu(z)
    
    core_attn_out = core_attn_out.reshape(batch, seq_len, -1)
    attn_out = jnp.dot(core_attn_out, out_proj)
    
    return attn_out, new_conv_state, new_recurrent_state


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _full_attention_block_jit(
    x,
    params_tuple,
    num_attention_heads,
    head_dim,
    num_key_value_heads,
    rms_norm_eps,
    dtype_code,
    positions,
    mask,
    k_cache=None,
    v_cache=None,
    slot_mapping=None,
    block_table=None,
    kv_lens=None,
    block_size=None,
    is_prefill=True,
):
    """JIT-compiled full attention block."""
    dtype = [jnp.float16, jnp.bfloat16, jnp.float32][dtype_code]
    
    q_proj, k_proj, v_proj, o_proj, q_norm, k_norm = params_tuple
    
    batch, seq_len, _ = x.shape
    
    x_cast = x.astype(dtype)
    
    q_gate = jnp.dot(x_cast, q_proj)
    attn_out_dim = num_attention_heads * head_dim
    
    q_gate_reshaped = q_gate.reshape(batch, seq_len, num_attention_heads, 2 * head_dim)
    query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
    gate = gate.reshape(batch, seq_len, -1)
    
    k = jnp.dot(x_cast, k_proj).reshape(batch, seq_len, num_key_value_heads, head_dim)
    v = jnp.dot(x_cast, v_proj).reshape(batch, seq_len, num_key_value_heads, head_dim)
    
    query = rms_norm(query, q_norm, rms_norm_eps)
    k = rms_norm(k, k_norm, rms_norm_eps)
    
    query = query.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    query = apply_rope(query, positions, head_dim, config_rope_theta=10000.0, partial_rotary_factor=1.0, layout="BHTD")
    k = apply_rope(k, positions, head_dim, config_rope_theta=10000.0, partial_rotary_factor=1.0, layout="BHTD")
    
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    if k_cache is not None:
        k_cache_input = k.transpose(0, 2, 1, 3)
        v_cache_input = v.transpose(0, 2, 1, 3)
        
        k_cache, v_cache = update_kv_cache(
            k_cache,
            v_cache,
            slot_mapping,
            k_cache_input,
            v_cache_input,
        )
        
        query_btnh = query.transpose(0, 2, 1, 3)
        
        if is_prefill:
            attn_out_btnh = paged_attention(
                query=query_btnh,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mapping,
                kv_lens=kv_lens,
                scale=1.0 / jnp.sqrt(head_dim),
                num_key_value_groups=num_key_value_groups,
            )
            out = attn_out_btnh
        else:
            attn_out_btnh = paged_attention_decode(
                query=query_btnh,
                k_cache=k_cache,
                v_cache=v_cache,
                block_table=block_table,
                kv_lens=kv_lens,
                block_size=block_size,
                scale=1.0 / jnp.sqrt(head_dim),
                num_key_value_groups=num_key_value_groups,
            )
            out = attn_out_btnh
    else:
        k = jnp.repeat(k, num_key_value_groups, axis=1)
        v = jnp.repeat(v, num_key_value_groups, axis=1)
        
        attn = nn.softmax(jnp.einsum("bhtd,bhsd->bhts", query, k) / jnp.sqrt(head_dim) + mask[None, None, :, :].astype(query.dtype), -1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    out = out * nn.sigmoid(gate)
    out = jnp.dot(out, o_proj)
    
    return out, k_cache, v_cache


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def _mlp_block_jit(
    x,
    params_tuple,
    hidden_act,
    rms_norm_eps,
    dtype_code,
):
    """JIT-compiled MLP block."""
    dtype = [jnp.float16, jnp.bfloat16, jnp.float32][dtype_code]
    
    ffn_norm, gate_proj, up_proj, down_proj = params_tuple
    
    x = rms_norm(x, ffn_norm, rms_norm_eps)
    x = x.astype(dtype)
    
    gate = jnp.dot(x, gate_proj)
    up = jnp.dot(x, up_proj)
    x = get_activation(hidden_act)(gate) * up
    x = jnp.dot(x, down_proj)
    
    return x


@partial(jax.jit, static_argnums=(3,))
def _embedding_lookup_jit(tokens, embed_tokens, dtype_code):
    """JIT-compiled embedding lookup."""
    dtype = [jnp.float16, jnp.bfloat16, jnp.float32][dtype_code]
    return embed_tokens[tokens].astype(dtype)


@partial(jax.jit, static_argnums=(2,))
def _lm_head_jit(x, lm_head):
    """JIT-compiled LM head projection."""
    x = x.astype(jnp.float32)
    return jnp.dot(x, lm_head)


def forward_jit(
    tokens,
    params,
    config,
    kv_cache_state=None,
    is_prefill=True,
):
    """JIT-compiled forward pass (wrapper).
    
    This function orchestrates JIT-compiled blocks for maximum performance.
    """
    batch, seq_len = tokens.shape
    dtype = config.get_dtype()
    dtype_code = 0 if dtype == jnp.float16 else (1 if dtype == jnp.bfloat16 else 2)
    
    x = _embedding_lookup_jit(tokens, params.embed_tokens, dtype_code)
    
    positions_1d = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
    positions = jnp.stack([positions_1d, positions_1d, positions_1d], axis=0)
    
    mask = causal_mask(seq_len, seq_len)
    
    if is_prefill and kv_cache_state is not None:
        kv_cache_state = init_linear_attention_states(kv_cache_state, config, batch_size=batch)
    
    linear_layer_idx = 0
    
    for i, lp in enumerate(params.layers):
        residual = x
        
        input_norm = lp["input_norm"]
        x = rms_norm(x, input_norm, config.rms_norm_eps)
        
        if config.layer_types[i] == "full_attention":
            params_tuple = (lp["q_proj"], lp["k_proj"], lp["v_proj"], lp["o_proj"], lp["q_norm"], lp["k_norm"])
            
            k_cache = kv_cache_state.k_cache if kv_cache_state else None
            v_cache = kv_cache_state.v_cache if kv_cache_state else None
            slot_mapping = kv_cache_state.slot_mapping if kv_cache_state else None
            block_table = kv_cache_state.block_table if kv_cache_state else None
            kv_lens = kv_cache_state.kv_lens if kv_cache_state else None
            
            attn_out, k_cache_new, v_cache_new = _full_attention_block_jit(
                x,
                params_tuple,
                config.num_attention_heads,
                config.head_dim,
                config.num_key_value_heads,
                config.rms_norm_eps,
                dtype_code,
                positions,
                mask,
                k_cache,
                v_cache,
                slot_mapping,
                block_table,
                kv_lens,
                config.block_size,
                is_prefill,
            )
            
            if kv_cache_state:
                from dataclasses import replace
                kv_cache_state = replace(kv_cache_state, k_cache=k_cache_new, v_cache=v_cache_new)
        else:
            params_tuple = (
                lp["in_proj_qkv"],
                lp["in_proj_z"],
                lp["in_proj_a"],
                lp["in_proj_b"],
                lp["conv1d_weight"],
                lp.get("conv1d_bias"),
                lp["dt_bias"],
                lp["A"],
                lp["norm_weight"],
                lp["out_proj"],
            )
            
            conv_state = None
            recurrent_state = None
            if kv_cache_state and kv_cache_state.conv_state is not None:
                conv_state = kv_cache_state.conv_state[:, linear_layer_idx]
                recurrent_state = kv_cache_state.recurrent_state[:, linear_layer_idx]
            
            attn_out, new_conv_state, new_recurrent_state = _gated_deltanet_block_jit(
                x,
                params_tuple,
                config.linear_num_key_heads,
                config.linear_key_head_dim,
                config.linear_num_value_heads,
                config.linear_value_head_dim,
                config.linear_conv_kernel_size,
                config.linear_chunk_size,
                config.rms_norm_eps,
                dtype_code,
                is_prefill,
                i,
                conv_state,
                recurrent_state,
                kv_cache_state.slot_mapping if kv_cache_state else None,
            )
            
            if kv_cache_state and new_conv_state is not None:
                from dataclasses import replace
                kv_cache_state = replace(
                    kv_cache_state,
                    conv_state=kv_cache_state.conv_state.at[:, linear_layer_idx].set(new_conv_state),
                    recurrent_state=kv_cache_state.recurrent_state.at[:, linear_layer_idx].set(new_recurrent_state),
                )
            
            linear_layer_idx += 1
        
        x = residual + attn_out
        
        residual = x
        
        mlp_params = (lp["ffn_norm"], lp["gate_proj"], lp["up_proj"], lp["down_proj"])
        mlp_out = _mlp_block_jit(x, mlp_params, config.hidden_act, config.rms_norm_eps, dtype_code)
        
        x = residual + mlp_out
    
    hidden_pre = x
    x = rms_norm(x, params.norm_weight, config.rms_norm_eps)
    
    logits = _lm_head_jit(x, params.lm_head if params.lm_head is not None else params.embed_tokens.T)
    
    return logits, kv_cache_state
