"""
Formal verification test suite for nanovllm_jax.

Tests against PLAN.md success criteria:
- KV cache parity: decode with cache matches no-cache
- End-to-end: generation produces consistent results
"""

import os
import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kv_cache import init_kv_cache, KVCacheState
from nanovllm_jax.load_weights_float16 import load_weights_from_hf_float16
from nanovllm_jax.model import ModelParams
from nanovllm_jax.model_simple_jit import forward_simple_jit
from pathlib import Path


def _build_formal_params():
    # Keep cache on available backend (GPU if present, CPU otherwise).
    config = Qwen3_5Config(dtype="float16")
    jax_weights, _ = load_weights_from_hf_float16("Qwen/Qwen3.5-0.8B", verbose=False)

    def get_layer_params(layer_idx):
        prefix = f"model.language_model.layers.{layer_idx}"
        if layer_idx in config.linear_attn_layers:
            return {
                "type": "linear",
                "in_proj_qkv": jax_weights[f"{prefix}.linear_attn.in_proj_qkv.weight"].T,
                "in_proj_z": jax_weights[f"{prefix}.linear_attn.in_proj_z.weight"].T,
                "in_proj_a": jax_weights[f"{prefix}.linear_attn.in_proj_a.weight"].T,
                "in_proj_b": jax_weights[f"{prefix}.linear_attn.in_proj_b.weight"].T,
                "out_proj": jax_weights[f"{prefix}.linear_attn.out_proj.weight"].T,
                "conv1d_weight": jax_weights[f"{prefix}.linear_attn.conv1d.weight"].squeeze(1),
                "conv1d_bias": jax_weights.get(f"{prefix}.linear_attn.conv1d.bias"),
                "norm_weight": jax_weights[f"{prefix}.linear_attn.norm.weight"],
                "A": jnp.exp(jax_weights[f"{prefix}.linear_attn.A_log"]),
                "dt_bias": jax_weights[f"{prefix}.linear_attn.dt_bias"],
                "input_norm": jax_weights[f"{prefix}.input_layernorm.weight"],
                "ffn_norm": jax_weights[f"{prefix}.post_attention_layernorm.weight"],
                "gate_proj": jax_weights[f"{prefix}.mlp.gate_proj.weight"].T,
                "up_proj": jax_weights[f"{prefix}.mlp.up_proj.weight"].T,
                "down_proj": jax_weights[f"{prefix}.mlp.down_proj.weight"].T,
            }
        return {
            "type": "full",
            "q_proj": jax_weights[f"{prefix}.self_attn.q_proj.weight"].T,
            "k_proj": jax_weights[f"{prefix}.self_attn.k_proj.weight"].T,
            "v_proj": jax_weights[f"{prefix}.self_attn.v_proj.weight"].T,
            "o_proj": jax_weights[f"{prefix}.self_attn.o_proj.weight"].T,
            "q_norm": jax_weights[f"{prefix}.self_attn.q_norm.weight"],
            "k_norm": jax_weights[f"{prefix}.self_attn.k_norm.weight"],
            "input_norm": jax_weights[f"{prefix}.input_layernorm.weight"],
            "ffn_norm": jax_weights[f"{prefix}.post_attention_layernorm.weight"],
            "gate_proj": jax_weights[f"{prefix}.mlp.gate_proj.weight"].T,
            "up_proj": jax_weights[f"{prefix}.mlp.up_proj.weight"].T,
            "down_proj": jax_weights[f"{prefix}.mlp.down_proj.weight"].T,
        }

    params = ModelParams(
        embed_tokens=jax_weights["model.language_model.embed_tokens.weight"],
        layers=[get_layer_params(i) for i in range(config.num_hidden_layers)],
        norm_weight=jax_weights["model.language_model.norm.weight"],
        lm_head=None,
        mtp_params=None,
    )
    return config, params


@pytest.fixture(scope="session")
def formal_params():
    model_name = "Qwen/Qwen3.5-0.8B"
    cache_root = Path(os.path.expanduser(os.getenv("HF_HUB_CACHE", "~/.cache/huggingface/hub")))
    marker = f"models--{model_name.replace('/', '--')}"
    has_cache = (
        cache_root.exists()
        and any(
            entry.is_dir() and entry.name.startswith(marker)
            for entry in cache_root.iterdir()
        )
    )
    if not has_cache:
        pytest.skip(
            f"Model {model_name} not cached locally. Set HF_HUB_CACHE and predownload weights before running this test."
        )
    return _build_formal_params()


@pytest.mark.parametrize(
    "test_case",
    (
        [760, 6511, 314, 9338, 369],
        [760, 6511, 314, 9338, 369, 11751, 13, 198],
    ),
)
def test_kv_cache_parity(formal_params, test_case):
    config, params = formal_params

    num_decode = 4
    all_ids_nocache = list(test_case)
    for _ in range(num_decode):
        logits, _ = forward_simple_jit(
            jnp.array([all_ids_nocache]),
            params,
            config,
            kv_cache_state=None,
            is_prefill=True,
        )
        all_ids_nocache.append(int(jnp.argmax(logits[0, -1, :])))

    kv = init_kv_cache(
        num_blocks=100,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seqs=1,
        max_blocks_per_seq=100,
        num_layers=config.num_hidden_layers,
        dtype=config.get_dtype(),
    )
    kv_state = KVCacheState(
        k_cache=kv.k_cache,
        v_cache=kv.v_cache,
        block_table=jnp.arange(100)[jnp.newaxis, :],
        kv_lens=jnp.array([0]),
        slot_mapping=jnp.arange(len(test_case), dtype=jnp.int32)[jnp.newaxis, :],
        conv_state=kv.conv_state,
        recurrent_state=kv.recurrent_state,
    )

    logits, kv_state = forward_simple_jit(
        jnp.array([test_case]),
        params,
        config,
        kv_cache_state=kv_state,
        is_prefill=True,
    )
    next_token = int(jnp.argmax(logits[0, -1, :]))
    all_ids_cache = list(test_case) + [next_token]

    for _ in range(num_decode - 1):
        decode_pos = len(all_ids_cache) - 1
        kv_decode = kv_state.replace(slot_mapping=jnp.array([[decode_pos]], dtype=jnp.int32))
        logits, kv_state = forward_simple_jit(
            jnp.array([[next_token]]),
            params,
            config,
            kv_cache_state=kv_decode,
            is_prefill=False,
        )
        next_token = int(jnp.argmax(logits[0, -1, :]))
        all_ids_cache.append(next_token)

    assert np.array_equal(np.array(all_ids_nocache), np.array(all_ids_cache))
