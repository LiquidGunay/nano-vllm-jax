#!/usr/bin/env python3
"""
Benchmark Qwen3.5 MTP (Multi-Token Prediction).

IMPORTANT: Qwen3.5 MTP predicts token t+2, not t+1!

How MTP works:
1. Main model at position T produces hidden state h_T
2. Main model predicts token T+1 from h_T
3. MTP receives h_T + embedding(T+1) and predicts token T+2
4. We get 2 tokens from 1 main model forward pass!

This is NOT traditional speculative decoding. Instead, it's "lookahead decoding"
where we use MTP to predict the next-next token while we're generating the next token.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
sys.path.insert(0, '.')

import time
import jax.numpy as jnp
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights_float16 import load_weights_from_hf_float16
from nanovllm_jax.model import ModelParams
from nanovllm_jax.model_simple_jit import forward_simple_jit
from nanovllm_jax.kv_cache import init_kv_cache, KVCacheState
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.mtp.mtp_layer import MTPParams, mtp_forward

print("=" * 70)
print("BENCHMARK: Qwen3.5 MTP (Multi-Token Prediction)")
print("=" * 70)

# Load config and weights
print("\nLoading model...")
config = Qwen3_5Config(dtype='float16')
jax_weights, _ = load_weights_from_hf_float16('Qwen/Qwen3.5-0.8B', verbose=False)

def get_layer_params(layer_idx):
    prefix = f'model.language_model.layers.{layer_idx}'
    if layer_idx in config.linear_attn_layers:
        return {
            'type': 'linear',
            'in_proj_qkv': jax_weights[f'{prefix}.linear_attn.in_proj_qkv.weight'].T,
            'in_proj_z': jax_weights[f'{prefix}.linear_attn.in_proj_z.weight'].T,
            'in_proj_a': jax_weights[f'{prefix}.linear_attn.in_proj_a.weight'].T,
            'in_proj_b': jax_weights[f'{prefix}.linear_attn.in_proj_b.weight'].T,
            'out_proj': jax_weights[f'{prefix}.linear_attn.out_proj.weight'].T,
            'conv1d_weight': jax_weights[f'{prefix}.linear_attn.conv1d.weight'].squeeze(1),
            'conv1d_bias': jax_weights.get(f'{prefix}.linear_attn.conv1d.bias'),
            'norm_weight': jax_weights[f'{prefix}.linear_attn.norm.weight'],
            'A': jnp.exp(jax_weights[f'{prefix}.linear_attn.A_log']),
            'dt_bias': jax_weights[f'{prefix}.linear_attn.dt_bias'],
            'input_norm': jax_weights[f'{prefix}.input_layernorm.weight'],
            'ffn_norm': jax_weights[f'{prefix}.post_attention_layernorm.weight'],
            'gate_proj': jax_weights[f'{prefix}.mlp.gate_proj.weight'].T,
            'up_proj': jax_weights[f'{prefix}.mlp.up_proj.weight'].T,
            'down_proj': jax_weights[f'{prefix}.mlp.down_proj.weight'].T,
        }
    else:
        return {
            'type': 'full',
            'q_proj': jax_weights[f'{prefix}.self_attn.q_proj.weight'].T,
            'k_proj': jax_weights[f'{prefix}.self_attn.k_proj.weight'].T,
            'v_proj': jax_weights[f'{prefix}.self_attn.v_proj.weight'].T,
            'o_proj': jax_weights[f'{prefix}.self_attn.o_proj.weight'].T,
            'q_norm': jax_weights[f'{prefix}.self_attn.q_norm.weight'],
            'k_norm': jax_weights[f'{prefix}.self_attn.k_norm.weight'],
            'input_norm': jax_weights[f'{prefix}.input_layernorm.weight'],
            'ffn_norm': jax_weights[f'{prefix}.post_attention_layernorm.weight'],
            'gate_proj': jax_weights[f'{prefix}.mlp.gate_proj.weight'].T,
            'up_proj': jax_weights[f'{prefix}.mlp.up_proj.weight'].T,
            'down_proj': jax_weights[f'{prefix}.mlp.down_proj.weight'].T,
        }

params = ModelParams(
    embed_tokens=jax_weights['model.language_model.embed_tokens.weight'],
    layers=[get_layer_params(i) for i in range(config.num_hidden_layers)],
    norm_weight=jax_weights['model.language_model.norm.weight'],
    lm_head=None,
    mtp_params=None,
)

mtp_params = MTPParams(
    eh_proj=jax_weights['mtp.fc.weight'].T,
    layers=[{
        'q_proj': jax_weights['mtp.layers.0.self_attn.q_proj.weight'].T,
        'k_proj': jax_weights['mtp.layers.0.self_attn.k_proj.weight'].T,
        'v_proj': jax_weights['mtp.layers.0.self_attn.v_proj.weight'].T,
        'o_proj': jax_weights['mtp.layers.0.self_attn.o_proj.weight'].T,
        'q_norm': jax_weights['mtp.layers.0.self_attn.q_norm.weight'],
        'k_norm': jax_weights['mtp.layers.0.self_attn.k_norm.weight'],
        'input_norm': jax_weights['mtp.layers.0.input_layernorm.weight'],
        'post_attn_norm': jax_weights['mtp.layers.0.post_attention_layernorm.weight'],
        'gate_proj': jax_weights['mtp.layers.0.mlp.gate_proj.weight'].T,
        'up_proj': jax_weights['mtp.layers.0.mlp.up_proj.weight'].T,
        'down_proj': jax_weights['mtp.layers.0.mlp.down_proj.weight'].T,
    }],
    pre_fc_norm_hidden=jax_weights['mtp.pre_fc_norm_hidden.weight'],
    pre_fc_norm_embedding=jax_weights['mtp.pre_fc_norm_embedding.weight'],
    final_norm=jax_weights.get('mtp.norm.weight'),
    lm_head=jax_weights['model.language_model.embed_tokens.weight'].T,
)

embed_tokens = jax_weights['model.language_model.embed_tokens.weight']

# ============================================================
# Warmup
# ============================================================
print("\nWarming up JIT...")
t0 = time.time()

num_blocks = 100
kv = init_kv_cache(
    num_blocks, config.block_size, config.num_key_value_heads, config.head_dim,
    1, 100, num_layers=config.num_hidden_layers, dtype=config.get_dtype()
)
kv = KVCacheState(
    k_cache=kv.k_cache, v_cache=kv.v_cache,
    block_table=jnp.arange(num_blocks)[jnp.newaxis, :],
    kv_lens=jnp.array([0]),
    slot_mapping=jnp.arange(5, dtype=jnp.int32)[jnp.newaxis, :],
    conv_state=kv.conv_state, recurrent_state=kv.recurrent_state,
)

# Warmup prefill
hidden_pre, kv = forward_simple_jit(jnp.array([[1, 2, 3, 4, 5]]), params, config, kv_cache_state=kv, is_prefill=True, return_hidden=True)

# Warmup MTP
hidden_last = hidden_pre[:, -1:, :]
_ = mtp_forward(
    hidden_state=hidden_last,
    next_token_ids=jnp.array([[1]]),
    embed_tokens=embed_tokens,
    params=mtp_params,
    config=config,
    positions=jnp.array([[4]]),  # Position 4 for the last token
)

# Warmup decode
kv_decode = replace(kv, slot_mapping=jnp.array([[5]], dtype=jnp.int32))
_ = forward_simple_jit(jnp.array([[1]]), params, config, kv_cache_state=kv_decode, is_prefill=False)

print(f"Warmup complete in {time.time()-t0:.1f}s")

# ============================================================
# Test 1: Standard Generation
# ============================================================
print("\n" + "=" * 70)
print("1. Standard Generation (1 token per forward pass)")
print("=" * 70)

input_ids = [760, 6511, 314, 9338, 369, 11751, 13, 198]  # "Tell me a joke about"
max_new_tokens = 30

print(f"\nPrompt: {len(input_ids)} tokens, generating {max_new_tokens}...")

t0 = time.time()

kv = init_kv_cache(
    num_blocks, config.block_size, config.num_key_value_heads, config.head_dim,
    1, 100, num_layers=config.num_hidden_layers, dtype=config.get_dtype()
)
kv = KVCacheState(
    k_cache=kv.k_cache, v_cache=kv.v_cache,
    block_table=jnp.arange(num_blocks)[jnp.newaxis, :],
    kv_lens=jnp.array([0]),
    slot_mapping=jnp.arange(len(input_ids), dtype=jnp.int32)[jnp.newaxis, :],
    conv_state=kv.conv_state, recurrent_state=kv.recurrent_state,
)

tokens = jnp.array([input_ids])
logits, kv = forward_simple_jit(tokens, params, config, kv_cache_state=kv, is_prefill=True)
next_token = int(jnp.argmax(logits[0, -1, :]))
all_ids = list(input_ids) + [next_token]

for j in range(max_new_tokens - 1):
    decode_pos = len(all_ids) - 1
    kv_decode = replace(kv, slot_mapping=jnp.array([[decode_pos]], dtype=jnp.int32))
    tokens = jnp.array([[next_token]])
    logits, kv = forward_simple_jit(tokens, params, config, kv_cache_state=kv_decode, is_prefill=False)
    next_token = int(jnp.argmax(logits[0, -1, :]))
    all_ids.append(next_token)

standard_time = time.time() - t0
standard_tps = max_new_tokens / standard_time
print(f"Generated {max_new_tokens} tokens in {standard_time:.2f}s ({standard_tps:.2f} tps)")
print(f"Forward passes: {max_new_tokens}")

# ============================================================
# Test 2: MTP Generation (up to 2 tokens per forward pass)
# ============================================================
print("\n" + "=" * 70)
print("2. MTP Generation (2 tokens per forward pass)")
print("=" * 70)

print(f"\nPrompt: {len(input_ids)} tokens, generating {max_new_tokens}...")

t0 = time.time()

kv = init_kv_cache(
    num_blocks, config.block_size, config.num_key_value_heads, config.head_dim,
    1, 100, num_layers=config.num_hidden_layers, dtype=config.get_dtype()
)
kv = KVCacheState(
    k_cache=kv.k_cache, v_cache=kv.v_cache,
    block_table=jnp.arange(num_blocks)[jnp.newaxis, :],
    kv_lens=jnp.array([0]),
    slot_mapping=jnp.arange(len(input_ids), dtype=jnp.int32)[jnp.newaxis, :],
    conv_state=kv.conv_state, recurrent_state=kv.recurrent_state,
)

# Prefill
tokens = jnp.array([input_ids])
hidden_pre, kv = forward_simple_jit(tokens, params, config, kv_cache_state=kv, is_prefill=True, return_hidden=True)
hidden_normed = rms_norm(hidden_pre, params.norm_weight, config.rms_norm_eps)
logits = jnp.dot(hidden_normed.astype(jnp.float32), params.embed_tokens.T)
token_t1 = int(jnp.argmax(logits[0, -1, :]))

# Use MTP to draft token T+2
hidden_last = hidden_pre[:, -1:, :]
mtp_logits, _ = mtp_forward(
    hidden_state=hidden_last,
    next_token_ids=jnp.array([[token_t1]]),
    embed_tokens=embed_tokens,
    params=mtp_params,
    config=config,
    positions=jnp.array([[len(input_ids) - 1]]),
)
token_t2_draft = int(jnp.argmax(mtp_logits[0, 0, :]))

all_ids = list(input_ids) + [token_t1, token_t2_draft]
num_forward_passes = 1  # Only the prefill forward pass so far

# Now we need to verify token_t2_draft
# And continue generating with MTP
tokens_generated = 2
mtp_matches = 0

while tokens_generated < max_new_tokens:
    # Main model verifies T+2 by running forward on T+1
    decode_pos = len(all_ids) - 2  # Position of T+1
    kv_verify = replace(kv, slot_mapping=jnp.array([[decode_pos]], dtype=jnp.int32))
    hidden_pre, kv = forward_simple_jit(
        jnp.array([[token_t1]]),
        params,
        config,
        kv_cache_state=kv_verify,
        is_prefill=False,
        return_hidden=True,
    )
    num_forward_passes += 1
    
    # Get main model's T+2 prediction
    hidden_normed = rms_norm(hidden_pre, params.norm_weight, config.rms_norm_eps)
    logits = jnp.dot(hidden_normed.astype(jnp.float32), params.embed_tokens.T)
    token_t2_main = int(jnp.argmax(logits[0, -1, :]))
    
    # Check if MTP matched
    if token_t2_draft == token_t2_main:
        mtp_matches += 1
        # MTP was correct, keep token_t2_draft
        # Now predict T+3 using main model's hidden state
        hidden_last = hidden_pre
        token_t3 = int(jnp.argmax(logits[0, -1, :]))
        
        if tokens_generated + 1 >= max_new_tokens:
            break
        
        # Use MTP to draft T+4
        mtp_logits, _ = mtp_forward(
            hidden_state=hidden_last,
            next_token_ids=jnp.array([[token_t2_draft]]),
            embed_tokens=embed_tokens,
            params=mtp_params,
            config=config,
            positions=jnp.array([[decode_pos + 1]]),
        )
        token_t4_draft = int(jnp.argmax(mtp_logits[0, 0, :]))
        
        all_ids.append(token_t3)
        all_ids.append(token_t4_draft)
        tokens_generated += 2
        token_t1 = token_t2_draft
        token_t2_draft = token_t4_draft
    else:
        # MTP was wrong, use main model's T+2
        all_ids[-1] = token_t2_main  # Replace the draft
        token_t1 = token_t2_main
        
        if tokens_generated + 1 >= max_new_tokens:
            break
        
        # Use MTP to draft T+3
        hidden_last = hidden_pre
        mtp_logits, _ = mtp_forward(
            hidden_state=hidden_last,
            next_token_ids=jnp.array([[token_t2_main]]),
            embed_tokens=embed_tokens,
            params=mtp_params,
            config=config,
            positions=jnp.array([[decode_pos + 1]]),
        )
        token_t3_draft = int(jnp.argmax(mtp_logits[0, 0, :]))
        
        all_ids.append(token_t3_draft)
        tokens_generated += 1
        token_t2_draft = token_t3_draft

mtp_time = time.time() - t0
mtp_tps = max_new_tokens / mtp_time
mtp_match_rate = mtp_matches / (num_forward_passes - 1) if num_forward_passes > 1 else 0

print(f"Generated {len(all_ids) - len(input_ids)} tokens in {mtp_time:.2f}s ({mtp_tps:.2f} tps)")
print(f"Forward passes: {num_forward_passes}")
print(f"MTP match rate: {mtp_match_rate:.1%}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

speedup = standard_time / mtp_time if mtp_time > 0 else 0

print(f"\n{'Method':<30} {'Tokens':>8} {'Time':>10} {'TPS':>10} {'Passes':>8}")
print("-" * 66)
print(f"{'Standard (1 token/pass)':<30} {max_new_tokens:>8} {standard_time:>10.2f}s {standard_tps:>10.2f} {max_new_tokens:>8}")
print(f"{'MTP (2 tokens/pass)':<30} {len(all_ids) - len(input_ids):>8} {mtp_time:>10.2f}s {mtp_tps:>10.2f} {num_forward_passes:>8}")

print(f"\nMTP match rate: {mtp_match_rate:.1%}")
print(f"Speedup: {speedup:.2f}x")

theoretical_speedup = max_new_tokens / num_forward_passes
print(f"Theoretical speedup (if MTP 100%): {theoretical_speedup:.2f}x")
