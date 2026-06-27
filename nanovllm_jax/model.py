"""Qwen3.5 model: parameters, transformer loop, and forward wrappers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp

from nanovllm_jax.attention import full_attention_block
from nanovllm_jax.cache import AttentionMetadata, HybridLayerState, KVCacheState, init_linear_attention_states
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.gdn import (
    gated_deltanet_block,
    jax_chunk_gated_delta_rule,
    jax_recurrent_gated_delta_rule,
)
from nanovllm_jax.layers import causal_mask, get_activation, rms_norm
from nanovllm_jax.lm_head import (
    _lm_head_greedy_top1_impl,
    _lm_head_greedy_top1_token_ids,
    lm_head_sample_token_ids,
    lm_head_token_ids_and_topk,
)
from nanovllm_jax.ops import ServingOpsProtocol
from nanovllm_jax.projection import (
    _FULL_ATTN_DECODE_QKV_PACKED_KEY,
    _GDN_DECODE_IN_PROJ_PACKED_KEY,
    _MLP_GATE_UP_PACKED_KEY,
    _can_use_decode_padded_gemm,
    _can_use_decode_rms_padded_gemm,
    _compact_prefill_dot_if_enabled,
    _compact_prefill_mlp,
    _compact_prefill_mlp_packed,
    _decode_padded_gemm_dot,
    _decode_padded_gemm_gate_up_enabled,
    _decode_projection_activation_dtype,
    _decode_rms_padded_gemm_dot,
    _decode_width1_rms_norm,
    _force_width1_decode_math,
    _stable_rmsnorm_fp32,
    _tokenwise_decode_dot,
)

@dataclass
class ModelParams:
    embed_tokens: jnp.ndarray
    layers: List[Dict[str, jnp.ndarray]]
    norm_weight: jnp.ndarray
    lm_head: Optional[jnp.ndarray] = None




def _model_params_flatten(params: ModelParams):
    """Flatten ModelParams into children and auxiliary data."""
    # Flatten all layer dicts into a tuple of arrays
    layer_children = []
    layer_aux = []
    for layer in params.layers:
        # Sort keys for consistent ordering
        keys = sorted(layer.keys())
        if _MLP_GATE_UP_PACKED_KEY in layer:
            keys = [k for k in keys if k not in {"gate_proj", "up_proj"}]
        layer_aux.append(keys)
        for k in keys:
            layer_children.append(layer[k])

    children = (
        params.embed_tokens,
        *layer_children,
        params.norm_weight,
        params.lm_head if params.lm_head is not None else jnp.zeros((1,), dtype=jnp.float16),
    )
    aux_data = (
        len(params.layers),
        layer_aux,
        params.lm_head is not None,
    )
    return children, aux_data


def _model_params_unflatten(aux_data, children):
    """Unflatten children and auxiliary data into ModelParams."""
    num_layers, layer_aux, has_lm_head = aux_data

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

    return ModelParams(
        embed_tokens=children[0],
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
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
        q_proj = jax.random.normal(keys[0], (config.hidden_size, attn_out_dim * 2)) * (config.hidden_size ** -0.5)
        k_proj = jax.random.normal(keys[1], (config.hidden_size, config.num_key_value_heads * config.head_dim)) * (config.hidden_size ** -0.5)
        v_proj = jax.random.normal(keys[2], (config.hidden_size, config.num_key_value_heads * config.head_dim)) * (config.hidden_size ** -0.5)
        gate_proj = jax.random.normal(keys[5], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        up_proj = jax.random.normal(keys[6], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        return {
            "q_proj": q_proj,
            "k_proj": k_proj,
            "v_proj": v_proj,
            _FULL_ATTN_DECODE_QKV_PACKED_KEY: jnp.concatenate(
                [q_proj, k_proj, v_proj],
                axis=1,
            ),
            "o_proj": jax.random.normal(keys[3], (attn_out_dim, config.hidden_size)) * (config.hidden_size ** -0.5),
            "q_norm": jnp.ones((config.num_attention_heads, config.head_dim)),
            "k_norm": jnp.ones((config.num_key_value_heads, config.head_dim)),
            "input_norm": jnp.ones(config.hidden_size),
            "post_attn_norm": jnp.ones(config.hidden_size),
            "gate_proj": gate_proj,
            "up_proj": up_proj,
            _MLP_GATE_UP_PACKED_KEY: jnp.concatenate([gate_proj, up_proj], axis=1),
            "down_proj": jax.random.normal(keys[7], (config.intermediate_size, config.hidden_size)) * (config.hidden_size ** -0.5),
            "ffn_norm": jnp.ones(config.hidden_size),
        }
    else:
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        in_proj_qkv = jax.random.normal(keys[0], (config.hidden_size, conv_dim)) * (config.hidden_size ** -0.5)
        in_proj_z = jax.random.normal(keys[1], (config.hidden_size, value_dim)) * (config.hidden_size ** -0.5)
        in_proj_a = jax.random.normal(keys[2], (config.hidden_size, config.linear_num_value_heads)) * (config.hidden_size ** -0.5)
        in_proj_b = jax.random.normal(keys[3], (config.hidden_size, config.linear_num_value_heads)) * (config.hidden_size ** -0.5)
        gate_proj = jax.random.normal(keys[5], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        up_proj = jax.random.normal(keys[6], (config.hidden_size, config.intermediate_size)) * (config.hidden_size ** -0.5)
        return {
            "input_norm": jnp.ones(config.hidden_size),
            "in_proj_qkv": in_proj_qkv,
            "in_proj_z": in_proj_z,
            "in_proj_a": in_proj_a,
            "in_proj_b": in_proj_b,
            _GDN_DECODE_IN_PROJ_PACKED_KEY: jnp.concatenate(
                [in_proj_qkv, in_proj_a, in_proj_b, in_proj_z],
                axis=1,
            ),
            "conv1d_weight": jax.random.normal(keys[4], (conv_dim, config.linear_conv_kernel_size)) * 0.02,
            "dt_bias": jnp.ones(config.linear_num_value_heads),
            "A": jnp.exp(jnp.full(config.linear_num_value_heads, 0.0)),
            "norm_weight": jnp.ones(config.linear_value_head_dim),
            "out_proj": jax.random.normal(keys[6], (value_dim, config.hidden_size)) * (config.hidden_size ** -0.5),
            "gate_proj": gate_proj,
            "up_proj": up_proj,
            _MLP_GATE_UP_PACKED_KEY: jnp.concatenate([gate_proj, up_proj], axis=1),
            "down_proj": jax.random.normal(keys[7], (config.intermediate_size, config.hidden_size)) * (config.hidden_size ** -0.5),
            "ffn_norm": jnp.ones(config.hidden_size),
        }




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
    prefix_hybrid_state: Optional[HybridLayerState] = None,
    is_prefill=True,
    backend: Optional[ServingOpsProtocol] = None,
    return_prefix_hybrid: bool = False,
    return_first_prefix_hybrid: bool = False,
    return_layer_hidden: bool = False,
    return_kv_prewrite: bool = False,
    return_layer_stages: bool = False,
    hybrid_state_is_layer: bool = False,
):
    """Matches HF Qwen3_5DecoderLayer - applies norms and residuals."""
    block_input = x
    residual = x

    # Apply input_layernorm (both full attention and linear attention)
    # HF applies input_layernorm before both layer types
    force_width1_norm = (not is_prefill) and x.ndim == 3 and _force_width1_decode_math()
    x = _decode_width1_rms_norm(
        x,
        params["input_norm"],
        config.rms_norm_eps,
        force_width1=force_width1_norm,
    )
    input_norm_out = x

    valid_token_mask = None
    if attention_metadata is not None:
        if attention_metadata.token_row_ids is not None:
            valid_token_mask = (
                jnp.arange(x.shape[1], dtype=jnp.int32)[None, :]
                < attention_metadata.query_start_loc[-1].astype(jnp.int32)
            )
        else:
            query_lens = jnp.diff(attention_metadata.query_start_loc).astype(jnp.int32)
            valid_token_mask = jnp.arange(x.shape[1], dtype=jnp.int32)[None, :] < query_lens[:, None]
    compact_prefill_tokens = (
        int(attention_metadata.num_prefill_tokens)
        if (
            is_prefill
            and attention_metadata is not None
            and isinstance(attention_metadata.num_prefill_tokens, int)
        )
        else None
    )

    # Apply attention/linear_attn
    layer_prewrite_k = jnp.zeros(
        (x.shape[0], x.shape[1], config.num_key_value_heads, config.head_dim),
        dtype=config.get_dtype(),
    )
    layer_prewrite_v = jnp.zeros(
        (x.shape[0], x.shape[1], config.num_key_value_heads, config.head_dim),
        dtype=config.get_dtype(),
    )
    if config.layer_types[layer_idx] == "full_attention":
        if return_kv_prewrite:
            x, kv_cache_state, layer_prewrite_k, layer_prewrite_v = full_attention_block(
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
                return_kv_prewrite=True,
            )
        else:
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
        result = gated_deltanet_block(
            x,
            params,
            positions,
            config,
            layer_idx,
            is_prefill=is_prefill,
            hybrid_state=hybrid_state,
            valid_token_mask=valid_token_mask,
            compact_prefill_tokens=compact_prefill_tokens,
            backend=backend,
            return_prefix_state=return_prefix_hybrid,
            return_first_prefix_state=return_first_prefix_hybrid,
            hybrid_state_is_layer=hybrid_state_is_layer,
            packed_token_row_ids=(
                attention_metadata.token_row_ids
                if is_prefill and attention_metadata is not None
                else None
            ),
            packed_query_start_loc=(
                attention_metadata.query_start_loc
                if is_prefill and attention_metadata is not None and attention_metadata.token_row_ids is not None
                else None
            ),
        )
        if isinstance(result, tuple):
            if (return_prefix_hybrid or return_first_prefix_hybrid) and len(result) == 3:
                x, hybrid_state, prefix_layer_state = result
                if prefix_hybrid_state is not None and prefix_layer_state is not None:
                    linear_layer_idx = len([l for l in config.linear_attn_layers if l < layer_idx])
                    if return_prefix_hybrid:
                        prefix_hybrid_state = replace(
                            prefix_hybrid_state,
                            conv_state=prefix_hybrid_state.conv_state.at[:, :, linear_layer_idx].set(
                                prefix_layer_state.conv_state
                            )
                            if prefix_hybrid_state.conv_state is not None
                            and prefix_layer_state.conv_state is not None
                            else prefix_hybrid_state.conv_state,
                            recurrent_state=prefix_hybrid_state.recurrent_state.at[:, :, linear_layer_idx].set(
                                prefix_layer_state.recurrent_state
                            )
                            if prefix_hybrid_state.recurrent_state is not None
                            and prefix_layer_state.recurrent_state is not None
                            else prefix_hybrid_state.recurrent_state,
                        )
                    else:
                        prefix_hybrid_state = replace(
                            prefix_hybrid_state,
                            conv_state=prefix_hybrid_state.conv_state.at[:, linear_layer_idx].set(
                                prefix_layer_state.conv_state
                            )
                            if prefix_hybrid_state.conv_state is not None
                            and prefix_layer_state.conv_state is not None
                            else prefix_hybrid_state.conv_state,
                            recurrent_state=prefix_hybrid_state.recurrent_state.at[:, linear_layer_idx].set(
                                prefix_layer_state.recurrent_state
                            )
                            if prefix_hybrid_state.recurrent_state is not None
                            and prefix_layer_state.recurrent_state is not None
                            else prefix_hybrid_state.recurrent_state,
                        )
            else:
                x, hybrid_state = result
        else:
            x = result

    attn_out = x

    # Add residual
    x = residual + x
    attn_residual_out = x

    # MLP path
    residual = x
    fused_mlp_gate_up = None
    if (
        not is_prefill
        and not return_layer_stages
        and _MLP_GATE_UP_PACKED_KEY in params
        and _can_use_decode_rms_padded_gemm(
            x,
            params["ffn_norm"],
            params[_MLP_GATE_UP_PACKED_KEY],
            config,
        )
    ):
        fused_mlp_gate_up = _decode_rms_padded_gemm_dot(
            x,
            params["ffn_norm"],
            params[_MLP_GATE_UP_PACKED_KEY],
            config,
        )
        ffn_norm_out = x
    else:
        x = _decode_width1_rms_norm(
            x,
            params["ffn_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norm,
        )
        ffn_norm_out = x

    # MLP computation (stays in bfloat16)
    force_width1_dot = (not is_prefill) and x.ndim == 3 and x.shape[1] > 1 and _force_width1_decode_math()
    activation_fn = get_activation(config.hidden_act)
    if is_prefill:
        if _MLP_GATE_UP_PACKED_KEY in params:
            x = _compact_prefill_mlp_packed(
                x,
                params[_MLP_GATE_UP_PACKED_KEY],
                params["down_proj"],
                activation_fn,
                valid_token_mask,
                compact_prefill_tokens,
                config,
            )
        else:
            x = _compact_prefill_mlp(
                x,
                params["gate_proj"],
                params["up_proj"],
                params["down_proj"],
                activation_fn,
                valid_token_mask,
                compact_prefill_tokens,
                config,
            )
    else:
        if fused_mlp_gate_up is not None:
            gate_up = fused_mlp_gate_up
            gate, up = jnp.split(gate_up, 2, axis=-1)
        else:
            x_proj = x.astype(_decode_projection_activation_dtype(x.shape[0], config))
        if fused_mlp_gate_up is None and _MLP_GATE_UP_PACKED_KEY in params:
            if (
                _decode_padded_gemm_gate_up_enabled(config)
                and _can_use_decode_padded_gemm(x_proj, params[_MLP_GATE_UP_PACKED_KEY], config)
            ):
                gate_up = _decode_padded_gemm_dot(x_proj, params[_MLP_GATE_UP_PACKED_KEY], config)
            else:
                gate_up = _tokenwise_decode_dot(
                    x_proj,
                    params[_MLP_GATE_UP_PACKED_KEY],
                    force_width1=force_width1_dot,
                )
            gate, up = jnp.split(gate_up, 2, axis=-1)
        elif fused_mlp_gate_up is None:
            if (
                _decode_padded_gemm_gate_up_enabled(config)
                and _can_use_decode_padded_gemm(x_proj, params["gate_proj"], config)
                and params["up_proj"].shape == params["gate_proj"].shape
            ):
                gate = _decode_padded_gemm_dot(x_proj, params["gate_proj"], config)
                up = _decode_padded_gemm_dot(x_proj, params["up_proj"], config)
            else:
                gate = _tokenwise_decode_dot(x_proj, params["gate_proj"], force_width1=force_width1_dot)
                up = _tokenwise_decode_dot(x_proj, params["up_proj"], force_width1=force_width1_dot)
        x = activation_fn(gate) * up
        if _can_use_decode_padded_gemm(x, params["down_proj"], config):
            x = _decode_padded_gemm_dot(x, params["down_proj"], config)
        else:
            x = _tokenwise_decode_dot(x, params["down_proj"], force_width1=force_width1_dot)
    mlp_out = x

    x = residual + x
    block_output = x

    outputs = [x, kv_cache_state, hybrid_state]
    if return_prefix_hybrid or return_first_prefix_hybrid:
        outputs.append(prefix_hybrid_state)
    if return_kv_prewrite:
        outputs.extend([layer_prewrite_k, layer_prewrite_v])
    if return_layer_stages:
        outputs.append(
            jnp.stack(
                [
                    block_input,
                    input_norm_out,
                    attn_out,
                    attn_residual_out,
                    ffn_norm_out,
                    mlp_out,
                    block_output,
                ],
                axis=0,
            )
        )
    return tuple(outputs)


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
    return_hidden_with_logits: bool = False,
    last_logits_only: bool = False,
    logit_positions: Optional[jnp.ndarray] = None,
    backend: Optional[ServingOpsProtocol] = None,
    return_prefix_hybrid: bool = False,
    return_first_prefix_hybrid: bool = False,
    return_layer_hidden: bool = False,
    return_kv_prewrite: bool = False,
    return_layer_stages: bool = False,
    hybrid_state_layerwise: bool = False,
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
    prefix_hybrid_state = None
    if return_prefix_hybrid and hybrid_state is not None:
        packed_token_rows = (
            attention_metadata.token_row_ids
            if attention_metadata is not None
            and attention_metadata.token_row_ids is not None
            else None
        )
        if packed_token_rows is not None:
            row_count = (
                int(hybrid_state.conv_state.shape[0])
                if hybrid_state.conv_state is not None
                else int(hybrid_state.recurrent_state.shape[0])
            )
            row_query_len = int(
                attention_metadata.max_query_len
                if attention_metadata.max_query_len is not None
                else seq_len
            )
            prefix_hybrid_state = HybridLayerState(
                conv_state=jnp.broadcast_to(
                    hybrid_state.conv_state[:, None, ...],
                    (row_count, row_query_len) + hybrid_state.conv_state.shape[1:],
                )
                if hybrid_state.conv_state is not None
                else None,
                recurrent_state=jnp.broadcast_to(
                    hybrid_state.recurrent_state[:, None, ...],
                    (row_count, row_query_len)
                    + hybrid_state.recurrent_state.shape[1:],
                )
                if hybrid_state.recurrent_state is not None
                else None,
            )
        else:
            prefix_hybrid_state = HybridLayerState(
                conv_state=jnp.broadcast_to(
                    hybrid_state.conv_state[:, None, ...],
                    (batch, seq_len) + hybrid_state.conv_state.shape[1:],
                )
                if hybrid_state.conv_state is not None
                else None,
                recurrent_state=jnp.broadcast_to(
                    hybrid_state.recurrent_state[:, None, ...],
                    (batch, seq_len) + hybrid_state.recurrent_state.shape[1:],
                )
                if hybrid_state.recurrent_state is not None
                else None,
            )
    if return_first_prefix_hybrid and hybrid_state is not None:
        prefix_hybrid_state = HybridLayerState(
            conv_state=hybrid_state.conv_state,
            recurrent_state=hybrid_state.recurrent_state,
        )

    layer_hidden_states = [] if return_layer_hidden else None
    kv_prewrite_k_states = [] if return_kv_prewrite else None
    kv_prewrite_v_states = [] if return_kv_prewrite else None
    layer_stage_states = [] if return_layer_stages else None
    num_linear_layers = len(config.linear_attn_layers)
    use_layerwise_hybrid = (
        hybrid_state_layerwise
        and hybrid_state is not None
        and hybrid_state.conv_state is not None
        and hybrid_state.recurrent_state is not None
        and num_linear_layers > 0
        and not return_prefix_hybrid
        and not return_first_prefix_hybrid
        and not return_kv_prewrite
        and not return_layer_stages
    )
    if use_layerwise_hybrid:
        hybrid_conv_layers = [
            hybrid_state.conv_state[:, linear_idx]
            for linear_idx in range(num_linear_layers)
        ]
        hybrid_recurrent_layers = [
            hybrid_state.recurrent_state[:, linear_idx]
            for linear_idx in range(num_linear_layers)
        ]
        linear_layer_cursor = 0

    for i, lp in enumerate(params.layers):
        block_hybrid_state = hybrid_state
        block_hybrid_state_is_layer = False
        if use_layerwise_hybrid and config.layer_types[i] != "full_attention":
            block_hybrid_state = HybridLayerState(
                conv_state=hybrid_conv_layers[linear_layer_cursor],
                recurrent_state=hybrid_recurrent_layers[linear_layer_cursor],
            )
            block_hybrid_state_is_layer = True
        block_result = transformer_block(
            x,
            lp,
            positions_mrope,
            mask,
            i,
            config,
            kv_cache_state,
            attention_metadata=attention_metadata,
            hybrid_state=block_hybrid_state,
            prefix_hybrid_state=prefix_hybrid_state,
            is_prefill=is_prefill,
            backend=backend,
            return_prefix_hybrid=return_prefix_hybrid,
            return_first_prefix_hybrid=return_first_prefix_hybrid,
            return_kv_prewrite=return_kv_prewrite,
            return_layer_stages=return_layer_stages,
            hybrid_state_is_layer=block_hybrid_state_is_layer,
        )
        x, kv_cache_state, block_updated_hybrid_state = block_result[:3]
        if block_hybrid_state_is_layer:
            hybrid_conv_layers[linear_layer_cursor] = block_updated_hybrid_state.conv_state
            hybrid_recurrent_layers[linear_layer_cursor] = block_updated_hybrid_state.recurrent_state
            linear_layer_cursor += 1
        else:
            hybrid_state = block_updated_hybrid_state
        offset = 3
        if return_prefix_hybrid or return_first_prefix_hybrid:
            prefix_hybrid_state = block_result[offset]
            offset += 1
        if return_kv_prewrite:
            layer_prewrite_k = block_result[offset]
            layer_prewrite_v = block_result[offset + 1]
            offset += 2
        if return_layer_stages:
            layer_stage = block_result[offset]
        if layer_hidden_states is not None:
            layer_hidden_states.append(x)
        if kv_prewrite_k_states is not None:
            kv_prewrite_k_states.append(layer_prewrite_k)
            kv_prewrite_v_states.append(layer_prewrite_v)
        if layer_stage_states is not None:
            layer_stage_states.append(layer_stage)

    if use_layerwise_hybrid:
        hybrid_state = HybridLayerState(
            conv_state=jnp.stack(hybrid_conv_layers, axis=1),
            recurrent_state=jnp.stack(hybrid_recurrent_layers, axis=1),
        )

    hidden_pre = x
    layer_hidden_result = (
        jnp.stack(layer_hidden_states, axis=0)
        if layer_hidden_states is not None
        else None
    )
    kv_prewrite_k_result = (
        jnp.stack(kv_prewrite_k_states, axis=0)
        if kv_prewrite_k_states is not None
        else None
    )
    kv_prewrite_v_result = (
        jnp.stack(kv_prewrite_v_states, axis=0)
        if kv_prewrite_v_states is not None
        else None
    )
    layer_stage_result = (
        jnp.stack(layer_stage_states, axis=0)
        if layer_stage_states is not None
        else None
    )

    if return_hidden and not return_hidden_with_logits:
        if return_layer_hidden:
            if return_prefix_hybrid or return_first_prefix_hybrid:
                return hidden_pre, kv_cache_state, hybrid_state, prefix_hybrid_state, layer_hidden_result
            if return_kv_prewrite:
                return hidden_pre, kv_cache_state, hybrid_state, layer_hidden_result, kv_prewrite_k_result, kv_prewrite_v_result, layer_stage_result
            return hidden_pre, kv_cache_state, hybrid_state, layer_hidden_result
        if return_prefix_hybrid or return_first_prefix_hybrid:
            return hidden_pre, kv_cache_state, hybrid_state, prefix_hybrid_state
        return hidden_pre, kv_cache_state, hybrid_state

    x = rms_norm(x, params.norm_weight, config.rms_norm_eps)
    x = x.astype(jnp.float32)
    if last_logits_only:
        if logit_positions is None:
            x = x[:, -1:, :]
        else:
            gather_idx = jnp.clip(logit_positions, 0, seq_len - 1).astype(jnp.int32)
            gather_idx = gather_idx[:, None, None]
            gather_idx = jnp.broadcast_to(gather_idx, (batch, 1, x.shape[-1]))
            x = jnp.take_along_axis(x, gather_idx, axis=1)
    output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
    logits = _tokenwise_decode_dot(
        x,
        output_weight,
        force_width1=(not is_prefill) and seq_len > 1 and _force_width1_decode_math(),
    )
    if return_hidden:
        hidden_result = (hidden_pre, logits) if return_hidden_with_logits else hidden_pre
        if return_layer_hidden:
            if return_prefix_hybrid or return_first_prefix_hybrid:
                return hidden_result, kv_cache_state, hybrid_state, prefix_hybrid_state, layer_hidden_result
            if return_kv_prewrite:
                return hidden_result, kv_cache_state, hybrid_state, layer_hidden_result, kv_prewrite_k_result, kv_prewrite_v_result, layer_stage_result
            return hidden_result, kv_cache_state, hybrid_state, layer_hidden_result
        if return_prefix_hybrid or return_first_prefix_hybrid:
            return hidden_result, kv_cache_state, hybrid_state, prefix_hybrid_state
        return hidden_result, kv_cache_state, hybrid_state
    if return_layer_hidden:
        if return_prefix_hybrid or return_first_prefix_hybrid:
            return logits, kv_cache_state, hybrid_state, prefix_hybrid_state, layer_hidden_result
        if return_kv_prewrite:
            return logits, kv_cache_state, hybrid_state, layer_hidden_result, kv_prewrite_k_result, kv_prewrite_v_result, layer_stage_result
        return logits, kv_cache_state, hybrid_state, layer_hidden_result
    if return_prefix_hybrid or return_first_prefix_hybrid:
        return logits, kv_cache_state, hybrid_state, prefix_hybrid_state
    return logits, kv_cache_state, hybrid_state


def forward(
    tokens,
    params,
    config,
    kv_cache_state=None,
    is_prefill=True,
    return_hidden=False,
    return_hidden_with_logits: bool = False,
    last_logits_only: bool = False,
    logit_positions: Optional[jnp.ndarray] = None,
    positions=None,
    attention_metadata: Optional[AttentionMetadata] = None,
    hybrid_state: Optional[HybridLayerState] = None,
    backend: Optional[ServingOpsProtocol] = None,
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
        return_hidden_with_logits=return_hidden_with_logits,
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
