"""Full-attention block for the promoted Qwen3.5 serving path."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import jax.numpy as jnp
from jax import nn

from nanovllm_jax.cache import AttentionMetadata, KVCacheState
from nanovllm_jax.layers import apply_rope, rms_norm
from nanovllm_jax.ops import ServingOps, ServingOpsProtocol
from nanovllm_jax.projection import (
    _FULL_ATTN_DECODE_QKV_PACKED_KEY,
    _compact_prefill_dot_if_enabled,
    _decode_projection_activation_dtype,
    _decode_width1_rms_norm,
    _enable_compact_prefill_full_attn_proj,
    _force_width1_decode_math,
    _tokenwise_decode_dot,
    _use_full_attention_decode_packed_qkv,
    _use_full_attention_prefill_packed_qkv,
)

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
    backend: Optional[ServingOpsProtocol] = None,
    return_kv_prewrite: bool = False,
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

    # Cast to target dtype for the promoted CUDA/JAX path.
    dtype = config.get_dtype()
    x_cast = x.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    valid_token_mask = None
    compact_prefill_tokens = None
    if is_prefill and attention_metadata is not None:
        if attention_metadata.token_row_ids is not None:
            valid_token_mask = (
                jnp.arange(seq_len, dtype=jnp.int32)[None, :]
                < attention_metadata.query_start_loc[-1].astype(jnp.int32)
            )
        else:
            query_lens = jnp.diff(attention_metadata.query_start_loc).astype(jnp.int32)
            valid_token_mask = jnp.arange(seq_len, dtype=jnp.int32)[None, :] < query_lens[:, None]
        compact_prefill_tokens = (
            int(attention_metadata.num_prefill_tokens)
            if isinstance(attention_metadata.num_prefill_tokens, int)
            else None
        )

    def _proj(inp: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        """Project a [B, T, D] tensor with a linear matrix [D, O].

        Using an explicit flatten+dot keeps row-wise projection numerically aligned
        between full-sequence and single-token calls across backends.
        """
        if not is_prefill and seq_len > 1 and _force_width1_decode_math():
            return _tokenwise_decode_dot(inp, weight, force_width1=True)
        if is_prefill:
            return _compact_prefill_dot_if_enabled(
                inp,
                weight,
                valid_token_mask,
                compact_prefill_tokens,
                enabled=_enable_compact_prefill_full_attn_proj(config),
            )
        out = jnp.dot(inp.reshape(-1, inp.shape[-1]), weight)
        return out.reshape(batch, seq_len, -1)

    # Qwen3.5 full attention uses fused Q + gate projection
    # q_proj: [hidden_size, hidden_size] -> output split into query and gate
    # query: [batch, seq_len, num_attention_heads * head_dim]
    # gate: [batch, seq_len, num_attention_heads * head_dim]
    attn_out_dim = config.num_attention_heads * config.head_dim
    force_width1_full_attn = False

    if force_width1_full_attn:
        query_parts = []
        k_parts = []
        v_parts = []
        gate_parts = []
        for t in range(seq_len):
            x_t = x_cast[:, t : t + 1, :]
            hidden_t = x_t.shape[-1]
            q_gate_t = jnp.dot(x_t.reshape(batch, hidden_t), params["q_proj"])[:, None, :]
            q_gate_t = q_gate_t.reshape(batch, 1, config.num_attention_heads, 2 * config.head_dim)
            query_t, gate_t = jnp.split(q_gate_t, 2, axis=-1)
            k_t = jnp.dot(x_t.reshape(batch, hidden_t), params["k_proj"])[:, None, :].reshape(
                batch, 1, config.num_key_value_heads, config.head_dim
            )
            v_t = jnp.dot(x_t.reshape(batch, hidden_t), params["v_proj"])[:, None, :].reshape(
                batch, 1, config.num_key_value_heads, config.head_dim
            )

            query_t = rms_norm(query_t, params["q_norm"], config.rms_norm_eps).transpose(0, 2, 1, 3)
            k_t = rms_norm(k_t, params["k_norm"], config.rms_norm_eps).transpose(0, 2, 1, 3)
            v_t = v_t.transpose(0, 2, 1, 3)
            pos_t = positions[:, :, t : t + 1] if positions.ndim == 3 else positions[:, t : t + 1]
            query_t = apply_rope(
                query_t,
                pos_t,
                config.head_dim,
                config.rope_theta,
                config.partial_rotary_factor,
                layout="BHTD",
                mrope_section=config.mrope_section,
            )
            k_t = apply_rope(
                k_t,
                pos_t,
                config.head_dim,
                config.rope_theta,
                config.partial_rotary_factor,
                layout="BHTD",
                mrope_section=config.mrope_section,
            )
            query_parts.append(query_t)
            k_parts.append(k_t)
            v_parts.append(v_t)
            gate_parts.append(gate_t.reshape(batch, 1, -1))
        query = jnp.concatenate(query_parts, axis=2)
        k = jnp.concatenate(k_parts, axis=2)
        v = jnp.concatenate(v_parts, axis=2)
        gate = jnp.concatenate(gate_parts, axis=1)
    else:
        use_packed_decode_qkv = _use_full_attention_decode_packed_qkv(
            params,
            is_prefill=is_prefill,
            batch=batch,
            seq_len=seq_len,
        )
        use_packed_prefill_qkv = _use_full_attention_prefill_packed_qkv(
            params,
            is_prefill=is_prefill,
            config=config,
        )
        if use_packed_decode_qkv or use_packed_prefill_qkv:
            packed_qkv = _proj(x_cast, params[_FULL_ATTN_DECODE_QKV_PACKED_KEY])
            q_gate_end = attn_out_dim * 2
            k_end = q_gate_end + config.num_key_value_heads * config.head_dim
            q_gate, k_raw, v_raw = jnp.split(packed_qkv, [q_gate_end, k_end], axis=-1)
        else:
            q_gate = _proj(x_cast, params["q_proj"])
            k_raw = _proj(x_cast, params["k_proj"])
            v_raw = _proj(x_cast, params["v_proj"])

        q_gate_reshaped = q_gate.reshape(batch, seq_len, config.num_attention_heads, 2 * config.head_dim)
        query, gate = jnp.split(q_gate_reshaped, 2, axis=-1)
        gate = gate.reshape(batch, seq_len, -1)

        k = k_raw.reshape(
            batch, seq_len, config.num_key_value_heads, config.head_dim
        )
        v = v_raw.reshape(
            batch, seq_len, config.num_key_value_heads, config.head_dim
        )

        # Apply RMSNorm BEFORE transpose (on head dimension, in [B, T, H, D] layout)
        force_width1_norm = (not is_prefill) and _force_width1_decode_math()
        query = _decode_width1_rms_norm(
            query,
            params["q_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norm,
        )
        k = _decode_width1_rms_norm(
            k,
            params["k_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norm,
        )

        # Transpose to [B, H, T, D]
        query = query.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE (now in [B, H, T, D] layout)
        query = apply_rope(query, positions, config.head_dim, config.rope_theta, config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)
        k = apply_rope(k, positions, config.head_dim, config.rope_theta, config.partial_rotary_factor, layout="BHTD", mrope_section=config.mrope_section)

    prewrite_k_cache_input = jnp.zeros((batch, seq_len, config.num_key_value_heads, config.head_dim), dtype=dtype)
    prewrite_v_cache_input = jnp.zeros((batch, seq_len, config.num_key_value_heads, config.head_dim), dtype=dtype)

    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    if kv_cache_state is not None:
        if backend is None:
            backend = ServingOps(config=config)

        # Transpose K, V back to [B, T, K, H] for cache storage
        k_cache_input = k.transpose(0, 2, 1, 3)  # [B, T, K, H]
        v_cache_input = v.transpose(0, 2, 1, 3)  # [B, T, K, H]
        prewrite_k_cache_input = k_cache_input
        prewrite_v_cache_input = v_cache_input

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
        # query is currently [batch, num_heads, seq_len, head_dim] (BHTD)
        # Backend attention expects [batch, seq_len, num_heads, head_dim] (BTNH)
        query_btnh = query.transpose(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]

        cache_storage, out = backend.write_kv_and_attention(
            layer_id=layer_idx,
            query=query_btnh,
            k=k_cache_input,
            v=v_cache_input,
            cache=kv_cache_state.storage,
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
    out = out.astype(
        _decode_projection_activation_dtype(batch, config) if not is_prefill else dtype
    )
    out = _tokenwise_decode_dot(
        out,
        params["o_proj"],
        force_width1=(not is_prefill) and seq_len > 1 and _force_width1_decode_math(),
    )

    if return_kv_prewrite:
        return out, kv_cache_state, prewrite_k_cache_input, prewrite_v_cache_input
    return out, kv_cache_state
