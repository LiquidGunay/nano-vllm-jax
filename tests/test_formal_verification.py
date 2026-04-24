"""
Formal verification test suite for nanovllm_jax.

Tests against PLAN.md success criteria:
- KV cache parity: decode with cache matches no-cache
- End-to-end: generation produces consistent results
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights_float16 import load_weights_from_hf_float16
from nanovllm_jax.model import ModelParams
from nanovllm_jax.model_simple_jit import forward_simple_jit
from nanovllm_jax.kv_cache import init_kv_cache, KVCacheState

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

print("=" * 70)
print("FORMAL VERIFICATION TEST SUITE")
print("=" * 70)
print(f"Model: Qwen3.5-0.8B")
print(f"Config: {config.num_hidden_layers} layers, dtype={config.dtype}")
print()

# Test prompts
test_cases = [
    {"name": "short", "tokens": [760, 6511, 314, 9338, 369]},
    {"name": "medium", "tokens": [760, 6511, 314, 9338, 369, 11751, 13, 198]},
]

results = {}

# ============================================================
# TEST: KV Cache Parity (Decode with cache = No cache)
# ============================================================
print("=" * 70)
print("TEST: KV Cache Parity")
print("=" * 70)

all_pass = True

for test_case in test_cases:
    input_ids = test_case["tokens"]
    num_decode = 10
    print(f"\nTest case: {test_case['name']} ({len(input_ids)} tokens, {num_decode} decode)")
    
    # Method 1: No cache (reference)
    all_ids_nocache = list(input_ids)
    for i in range(num_decode):
        tokens = jnp.array([all_ids_nocache])
        logits, _ = forward_simple_jit(tokens, params, config, kv_cache_state=None, is_prefill=True)
        next_token = int(jnp.argmax(logits[0, -1, :]))
        all_ids_nocache.append(next_token)
    
    # Method 2: With cache
    num_blocks = 100
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
    all_ids_cache = list(input_ids) + [next_token]
    
    for i in range(num_decode - 1):
        decode_pos = len(all_ids_cache) - 1
        kv_decode = replace(kv, slot_mapping=jnp.array([[decode_pos]], dtype=jnp.int32))
        tokens = jnp.array([[next_token]])
        logits, kv = forward_simple_jit(tokens, params, config, kv_cache_state=kv_decode, is_prefill=False)
        next_token = int(jnp.argmax(logits[0, -1, :]))
        all_ids_cache.append(next_token)
    
    match = all_ids_nocache == all_ids_cache
    status = "✓ PASS" if match else "❌ FAIL"
    print(f"  No cache: {all_ids_nocache}")
    print(f"  With cache: {all_ids_cache}")
    print(f"  {status}")
    
    results[test_case['name']] = match
    all_pass = all_pass and match

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

for name, passed in results.items():
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"  {name}: {status}")

print("\n" + "=" * 70)
if all_pass:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ❌")
print("=" * 70)
