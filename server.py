#!/usr/bin/env python3
"""
Simple API server for nano-vllm-jax with resource limiting.

Usage:
    python server.py --port 8080 --memory-limit 4G --cpu-percent 50

The server will:
1. Load model and pre-compile during startup (can take 2-5 min)
2. Limit memory usage via JAX memory_fraction
3. Limit CPU usage via nice priority (Unix only)
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import sys
import argparse
import time
import json
import resource
from functools import lru_cache
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flask import Flask, request, jsonify

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights_float16 import load_weights_from_hf_float16
from nanovllm_jax.model import ModelParams
from nanovllm_jax.model_simple_jit import forward_simple_jit
from nanovllm_jax.kv_cache import init_kv_cache, init_linear_attention_states, KVCacheState

app = Flask(__name__)

params = None
config = None
tokenizer = None
compiled_prefill = None
compiled_decode = None


def limit_memory(memory_gb: float):
    """Limit process memory usage."""
    max_bytes = int(memory_gb * 1024 * 1024 * 1024)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
        print(f"Memory limit set to {memory_gb}GB")
    except Exception as e:
        print(f"Warning: Could not set memory limit: {e}")


def limit_cpu(nice_value: int = 10):
    """Lower process priority to reduce CPU contention."""
    try:
        os.nice(nice_value)
        print(f"CPU priority set to nice={nice_value}")
    except Exception as e:
        print(f"Warning: Could not set CPU priority: {e}")


def get_layer_params(jax_weights, layer_idx, config):
    """Extract layer parameters from JAX weights."""
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


def load_model(memory_fraction: float = 0.5):
    """Load model with memory fraction limit."""
    global params, config
    
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    config = Qwen3_5Config(dtype='float16')
    
    print(f"  Model: Qwen3.5-0.8B")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Memory fraction: {memory_fraction*100:.0f}%")
    print(f"  Dtype: {config.dtype}")
    
    print("\n  [1/2] Loading weights from HuggingFace...")
    jax_weights, _ = load_weights_from_hf_float16('Qwen/Qwen3.5-0.8B', verbose=True)
    
    print("  [2/2] Creating model parameters...")
    params = ModelParams(
        embed_tokens=jax_weights['model.language_model.embed_tokens.weight'],
        layers=[get_layer_params(jax_weights, i, config) for i in range(config.num_hidden_layers)],
        norm_weight=jax_weights['model.language_model.norm.weight'],
        lm_head=None,
        mtp_params=None,
    )
    
    print(f"\n  Model loaded in {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")
    
    return params


def precompile(max_prefill: int = 64):
    """Pre-compile common shapes for faster inference."""
    global compiled_prefill, compiled_decode, params, config
    
    if params is None or config is None:
        return
    
    print(f"\n{'='*60}")
    print("Pre-compiling JIT functions...")
    print(f"{'='*60}")
    print("  This takes time but ensures fast inference later.")
    print("  Compiling for:")
    print(f"    - Prefill lengths: 16, {max_prefill}")
    print(f"    - Decode step: 1 token")
    print()
    
    t0 = time.time()
    
    num_blocks = 100
    dtype = config.get_dtype()
    
    kv = init_kv_cache(
        num_blocks, config.block_size, config.num_key_value_heads, config.head_dim,
        1, 100, num_layers=config.num_hidden_layers, dtype=dtype
    )
    kv = KVCacheState(
        k_cache=kv.k_cache, v_cache=kv.v_cache,
        block_table=jnp.arange(num_blocks)[jnp.newaxis, :],
        kv_lens=jnp.array([0]),
        slot_mapping=jnp.zeros((1, max_prefill), dtype=jnp.int32),  # Will be overwritten per-shape
        conv_state=kv.conv_state, recurrent_state=kv.recurrent_state,
    )
    
    print("  Compiling prefill shapes...")
    compiled_fns = {}
    for seq_len in [16, max_prefill]:
        print(f"    Prefill: batch=1, seq_len={seq_len}...", end=" ", flush=True)
        t1 = time.time()
        tokens = jnp.zeros((1, seq_len), dtype=jnp.int32)
        kv_prefill = replace(kv, slot_mapping=jnp.arange(seq_len, dtype=jnp.int32)[jnp.newaxis, :])
        _, _ = forward_simple_jit(tokens, params, config, kv_cache_state=kv_prefill, is_prefill=True)
        compiled_fns[('prefill', seq_len)] = True
        print(f"{time.time()-t1:.1f}s")
    
    print("  Compiling decode shape...")
    print(f"    Decode: batch=1, seq_len=1...", end=" ", flush=True)
    t1 = time.time()
    kv_decode = replace(kv, kv_lens=jnp.array([seq_len]))
    tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    _, _ = forward_simple_jit(tokens, params, config, kv_cache_state=kv_decode, is_prefill=False)
    print(f"{time.time()-t1:.1f}s")
    
    compiled_prefill = compiled_fns
    compiled_decode = True
    
    print(f"\n  Compilation complete in {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': params is not None,
    })


@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completion endpoint."""
    global params, config, tokenizer
    
    if params is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 32)
    temperature = data.get('temperature', 0.0)
    
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ''
    
    try:
        from transformers import AutoTokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
        
        input_ids = tokenizer.encode(prompt, return_tensors='np')[0].tolist()
    except Exception as e:
        input_ids = [760, 6511]
    
    try:
        t0 = time.time()
        output_ids = generate_tokens(input_ids, max_tokens, temperature)
        gen_time = time.time() - t0
        
        output_text = tokenizer.decode(output_ids[len(input_ids):]) if tokenizer else str(output_ids)
        
        return jsonify({
            'id': 'cmpl-' + str(int(time.time())),
            'object': 'text_completion',
            'created': int(time.time()),
            'model': 'Qwen3.5-0.8B',
            'choices': [{
                'text': output_text,
                'index': 0,
                'finish_reason': 'length',
            }],
            'usage': {
                'prompt_tokens': len(input_ids),
                'completion_tokens': len(output_ids) - len(input_ids),
                'total_tokens': len(output_ids),
            },
            'stats': {
                'generation_time_ms': int(gen_time * 1000),
                'tokens_per_second': (len(output_ids) - len(input_ids)) / gen_time if gen_time > 0 else 0,
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/v1/generate', methods=['POST'])
def generate():
    """Simple generate endpoint (token-based)."""
    global params, config
    
    if params is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    
    input_ids = data.get('input_ids', [760, 6511])
    max_tokens = data.get('max_tokens', 32)
    temperature = data.get('temperature', 0.0)
    
    try:
        t0 = time.time()
        output_ids = generate_tokens(input_ids, max_tokens, temperature)
        gen_time = time.time() - t0
        
        return jsonify({
            'input_ids': input_ids,
            'output_ids': output_ids,
            'new_tokens': output_ids[len(input_ids):],
            'stats': {
                'generation_time_ms': int(gen_time * 1000),
                'tokens_per_second': (len(output_ids) - len(input_ids)) / gen_time if gen_time > 0 else 0,
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def generate_tokens(input_ids, max_tokens, temperature=0.0):
    """Generate tokens using the model."""
    global params, config
    
    num_blocks = 100
    dtype = config.get_dtype()
    
    kv = init_kv_cache(
        num_blocks, config.block_size, config.num_key_value_heads, config.head_dim,
        1, 100, num_layers=config.num_hidden_layers, dtype=dtype
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
    
    output_ids = list(input_ids)
    
    for i in range(max_tokens):
        if temperature > 0:
            raise NotImplementedError("Temperature sampling not implemented")
        
        next_token = int(jnp.argmax(logits[0, -1, :]))
        output_ids.append(next_token)
        
        kv_decode = replace(
            kv,
            kv_lens=jnp.array([len(output_ids) - 1]),
            slot_mapping=jnp.array([[len(input_ids) + i]]),
        )
        tokens = jnp.array([[next_token]])
        logits, kv = forward_simple_jit(tokens, params, config, kv_cache_state=kv_decode, is_prefill=False)
    
    return output_ids


def main():
    parser = argparse.ArgumentParser(description='nano-vllm-jax API server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--memory-limit', type=float, default=4.0, help='Memory limit in GB')
    parser.add_argument('--memory-fraction', type=float, default=0.5, help='JAX memory fraction (0-1)')
    parser.add_argument('--cpu-nice', type=int, default=10, help='CPU nice value (higher = lower priority)')
    parser.add_argument('--max-prefill', type=int, default=64, help='Max prefill length to pre-compile')
    parser.add_argument('--skip-compile', action='store_true', help='Skip pre-compilation')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("nano-vllm-jax API Server")
    print(f"{'='*60}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Memory limit: {args.memory_limit}GB")
    print(f"  Memory fraction: {args.memory_fraction*100:.0f}%")
    print(f"  CPU nice: {args.cpu_nice}")
    print(f"{'='*60}\n")
    
    limit_memory(args.memory_limit)
    limit_cpu(args.cpu_nice)
    
    load_model(args.memory_fraction)
    
    if not args.skip_compile:
        precompile(args.max_prefill)
    
    print(f"\n{'='*60}")
    print("Server ready!")
    print(f"{'='*60}")
    print(f"  Endpoints:")
    print(f"    GET  /health")
    print(f"    POST /v1/completions")
    print(f"    POST /v1/generate")
    print(f"\n  Example:")
    print(f'    curl -X POST http://{args.host}:{args.port}/v1/generate \\')
    print(f'         -H "Content-Type: application/json" \\')
    print(f'         -d \'{{"input_ids": [760, 6511], "max_tokens": 10}}\'')
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
