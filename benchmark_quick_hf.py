#!/usr/bin/env python3
"""Quick JAX vs HuggingFace comparison."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
sys.path.insert(0, '.')

import time
import jax.numpy as jnp
import torch
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights_float16 import load_weights_from_hf_float16
from nanovllm_jax.model import ModelParams
from nanovllm_jax.model_simple_jit import forward_simple_jit
from nanovllm_jax.kv_cache import init_kv_cache, KVCacheState
from transformers import AutoModelForCausalLM

print("=" * 70)
print("JAX vs HuggingFace Throughput Comparison")
print("=" * 70)

# Load models
print("\nLoading models...")
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

jax_params = ModelParams(
    embed_tokens=jax_weights['model.language_model.embed_tokens.weight'],
    layers=[get_layer_params(i) for i in range(config.num_hidden_layers)],
    norm_weight=jax_weights['model.language_model.norm.weight'],
    lm_head=None,
    mtp_params=None,
)

hf_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-0.8B',
    torch_dtype=torch.float16,
    device_map='cpu',
    trust_remote_code=True,
)
hf_model.eval()

# Warmup
print("Warming up...")
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
_ = forward_simple_jit(jnp.array([[1, 2, 3, 4, 5]]), jax_params, config, kv_cache_state=kv, is_prefill=True)
kv_decode = replace(kv, slot_mapping=jnp.array([[5]], dtype=jnp.int32))
_ = forward_simple_jit(jnp.array([[1]]), jax_params, config, kv_cache_state=kv_decode, is_prefill=False)
with torch.no_grad():
    _ = hf_model(torch.tensor([[1, 2, 3, 4, 5]]))

# Test
input_ids = [760, 6511, 314, 9338, 369, 11751, 13, 198]  # "Tell me a joke about"
max_new_tokens = 10

print(f"\nPrompt: {len(input_ids)} tokens, generating {max_new_tokens} tokens")
print("=" * 70)

# JAX with KV cache
print("\n1. JAX (with KV cache)...")
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
logits, kv = forward_simple_jit(tokens, jax_params, config, kv_cache_state=kv, is_prefill=True)
next_token = int(jnp.argmax(logits[0, -1, :]))
jax_ids = list(input_ids) + [next_token]

for _ in range(max_new_tokens - 1):
    decode_pos = len(jax_ids) - 1
    kv_decode = replace(kv, slot_mapping=jnp.array([[decode_pos]], dtype=jnp.int32))
    tokens = jnp.array([[next_token]])
    logits, kv = forward_simple_jit(tokens, jax_params, config, kv_cache_state=kv_decode, is_prefill=False)
    next_token = int(jnp.argmax(logits[0, -1, :]))
    jax_ids.append(next_token)

jax_time = time.time() - t0
jax_tps = max_new_tokens / jax_time
print(f"  {max_new_tokens} tokens in {jax_time:.2f}s ({jax_tps:.2f} tps)")
print(f"  Output: {jax_ids[:15]}")

# HuggingFace with KV cache
print("\n2. HuggingFace (with KV cache)...")
t0 = time.time()

with torch.no_grad():
    input_tensor = torch.tensor([input_ids])
    outputs = hf_model(input_tensor, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = int(torch.argmax(outputs.logits[0, -1, :]))
    hf_ids = list(input_ids) + [next_token]
    
    for _ in range(max_new_tokens - 1):
        input_tensor = torch.tensor([[next_token]])
        outputs = hf_model(input_tensor, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = int(torch.argmax(outputs.logits[0, -1, :]))
        hf_ids.append(next_token)

hf_time = time.time() - t0
hf_tps = max_new_tokens / hf_time
print(f"  {max_new_tokens} tokens in {hf_time:.2f}s ({hf_tps:.2f} tps)")
print(f"  Output: {hf_ids[:15]}")

# HuggingFace without KV cache
print("\n3. HuggingFace (no KV cache)...")
t0 = time.time()

with torch.no_grad():
    hf_nocache_ids = list(input_ids)
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([hf_nocache_ids])
        logits = hf_model(input_tensor).logits
        next_token = int(torch.argmax(logits[0, -1, :]))
        hf_nocache_ids.append(next_token)

hf_nocache_time = time.time() - t0
hf_nocache_tps = max_new_tokens / hf_nocache_time
print(f"  {max_new_tokens} tokens in {hf_nocache_time:.2f}s ({hf_nocache_tps:.2f} tps)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Implementation':<35} {'Time':>10} {'TPS':>10}")
print("-" * 55)
print(f"{'JAX (KV cache)':<35} {jax_time:>10.2f}s {jax_tps:>10.2f}")
print(f"{'HuggingFace (KV cache)':<35} {hf_time:>10.2f}s {hf_tps:>10.2f}")
print(f"{'HuggingFace (no cache)':<35} {hf_nocache_time:>10.2f}s {hf_nocache_tps:>10.2f}")

print(f"\nJAX speedup vs HF (with cache):   {jax_tps / hf_tps:.2f}x")
print(f"JAX speedup vs HF (no cache):     {jax_tps / hf_nocache_tps:.2f}x")

# Check parity
if jax_ids == hf_ids:
    print(f"\n✓ JAX and HF outputs MATCH exactly!")
else:
    print(f"\n✗ MISMATCH detected")
    for i, (j, h) in enumerate(zip(jax_ids, hf_ids)):
        if j != h:
            print(f"  First diff at pos {i}: JAX={j}, HF={h}")
            break
