"""Persistent Qwen3.5 multi-token predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nanovllm_jax.cache import (
    AttentionMetadata,
    KVCacheState,
    KVCacheStorage,
)
from nanovllm_jax.config import RuntimeSpec
from nanovllm_jax.layers import apply_rope, get_activation, rms_norm
from nanovllm_jax.lm_head import lm_head_greedy_token_ids_from_normed
from nanovllm_jax.ops import ServingOpsProtocol
from nanovllm_jax.projection import (
    _FULL_ATTN_DECODE_QKV_PACKED_KEY,
    _MLP_GATE_UP_PACKED_KEY,
)


class MTPParams(NamedTuple):
    """The predictor's weights; embeddings and vocabulary head stay shared."""

    input_projection: jax.Array
    layer: dict[str, jax.Array]
    hidden_norm: jax.Array
    embedding_norm: jax.Array
    output_norm: jax.Array


@dataclass(frozen=True)
class MTPState:
    """Runner-owned predictor KV and one proposal row per resident slot."""

    cache_storage: KVCacheStorage
    draft_token_ids: jax.Array


def init_mtp_params(key: jax.Array, config: RuntimeSpec) -> MTPParams:
    """Create small random predictor weights for focused tests."""

    model = config.model
    keys = jax.random.split(key, 7)
    query_dim = model.num_attention_heads * model.head_dim
    kv_dim = model.num_key_value_heads * model.head_dim
    packed_qkv = jax.random.normal(
        keys[1],
        (model.hidden_size, 2 * query_dim + 2 * kv_dim),
    ) * model.hidden_size**-0.5
    gate_up = jax.random.normal(
        keys[4],
        (model.hidden_size, 2 * model.intermediate_size),
    ) * model.hidden_size**-0.5
    return MTPParams(
        input_projection=jax.random.normal(
            keys[0],
            (2 * model.hidden_size, model.hidden_size),
        )
        * model.hidden_size**-0.5,
        layer={
            _FULL_ATTN_DECODE_QKV_PACKED_KEY: packed_qkv,
            "o_proj": jax.random.normal(
                keys[2],
                (query_dim, model.hidden_size),
            )
            * query_dim**-0.5,
            "q_norm": jnp.ones((model.head_dim,)),
            "k_norm": jnp.ones((model.head_dim,)),
            "input_norm": jnp.ones((model.hidden_size,)),
            "post_attn_norm": jnp.ones((model.hidden_size,)),
            _MLP_GATE_UP_PACKED_KEY: gate_up,
            "down_proj": jax.random.normal(
                keys[5],
                (model.intermediate_size, model.hidden_size),
            )
            * model.intermediate_size**-0.5,
        },
        hidden_norm=jnp.ones((model.hidden_size,)),
        embedding_norm=jnp.ones((model.hidden_size,)),
        output_norm=jnp.ones((model.hidden_size,)),
    )


def mtp_forward_cached(
    hidden_states: jax.Array,
    next_token_ids: jax.Array,
    *,
    embed_tokens: jax.Array,
    params: MTPParams,
    config: RuntimeSpec,
    positions: jax.Array,
    cache_storage: KVCacheStorage,
    metadata: AttentionMetadata,
    backend: ServingOpsProtocol,
) -> tuple[jax.Array, KVCacheStorage]:
    """Advance the full-attention predictor and return final-normed hidden."""

    model = config.model
    dtype = config.compile.jax_dtype()
    hidden = rms_norm(
        hidden_states.astype(dtype),
        params.hidden_norm,
        model.rms_norm_eps,
    )
    embeddings = rms_norm(
        embed_tokens[next_token_ids].astype(dtype),
        params.embedding_norm,
        model.rms_norm_eps,
    )
    x = jnp.dot(
        jnp.concatenate((embeddings, hidden), axis=-1),
        params.input_projection,
    )

    residual = x
    x = rms_norm(x, params.layer["input_norm"], model.rms_norm_eps)
    packed_qkv = jnp.dot(
        x.astype(dtype),
        params.layer[_FULL_ATTN_DECODE_QKV_PACKED_KEY],
    )
    query_dim = model.num_attention_heads * model.head_dim
    kv_dim = model.num_key_value_heads * model.head_dim
    q_gate, key, value = jnp.split(
        packed_qkv,
        (2 * query_dim, 2 * query_dim + kv_dim),
        axis=-1,
    )
    batch, width, _ = q_gate.shape
    query, gate = jnp.split(
        q_gate.reshape(batch, width, model.num_attention_heads, 2 * model.head_dim),
        2,
        axis=-1,
    )
    key = key.reshape(batch, width, model.num_key_value_heads, model.head_dim)
    value = value.reshape(batch, width, model.num_key_value_heads, model.head_dim)
    query = rms_norm(query, params.layer["q_norm"], model.rms_norm_eps)
    key = rms_norm(key, params.layer["k_norm"], model.rms_norm_eps)
    query = apply_rope(
        query.transpose(0, 2, 1, 3),
        positions,
        model.head_dim,
        model.rope_theta,
        model.partial_rotary_factor,
        layout="BHTD",
        mrope_section=None,
    ).transpose(0, 2, 1, 3)
    key = apply_rope(
        key.transpose(0, 2, 1, 3),
        positions,
        model.head_dim,
        model.rope_theta,
        model.partial_rotary_factor,
        layout="BHTD",
        mrope_section=None,
    ).transpose(0, 2, 1, 3)
    cache_storage, attention = backend.write_kv_and_attention(
        layer_id=0,
        query=query,
        k=key,
        v=value,
        cache=cache_storage,
        metadata=metadata,
        block_size=config.capacity.block_size,
        scale=1.0 / jnp.sqrt(model.head_dim),
        num_key_value_groups=model.num_attention_heads // model.num_key_value_heads,
        is_prefill=True,
    )
    attention = attention * jax.nn.sigmoid(gate.reshape(batch, width, -1))
    x = residual + jnp.dot(attention.astype(dtype), params.layer["o_proj"])

    residual = x
    x = rms_norm(x, params.layer["post_attn_norm"], model.rms_norm_eps)
    gate, up = jnp.split(
        jnp.dot(x.astype(dtype), params.layer[_MLP_GATE_UP_PACKED_KEY]),
        2,
        axis=-1,
    )
    x = residual + jnp.dot(
        get_activation(model.hidden_act)(gate) * up,
        params.layer["down_proj"],
    )
    return rms_norm(x, params.output_norm, model.rms_norm_eps), cache_storage


def mtp_token_ids(
    hidden_states: jax.Array,
    *,
    embed_tokens: jax.Array,
    config: RuntimeSpec,
) -> jax.Array:
    """Project final-normed predictor hidden through the tied vocabulary head."""

    return lm_head_greedy_token_ids_from_normed(
        hidden_states,
        embed_tokens,
        config,
    )


def mtp_draft_chain(
    first_hidden: jax.Array,
    *,
    width: int,
    start_positions: jax.Array,
    row_valid: jax.Array,
    block_tables: jax.Array,
    embed_tokens: jax.Array,
    params: MTPParams,
    config: RuntimeSpec,
    cache_storage: KVCacheStorage,
    backend: ServingOpsProtocol,
) -> tuple[jax.Array, KVCacheStorage]:
    """Recursively produce a fixed-width proposal from one predictor state."""

    row_count = int(first_hidden.shape[0])
    hidden = first_hidden
    token = mtp_token_ids(
        hidden,
        embed_tokens=embed_tokens,
        config=config,
    )[:, 0]
    drafts = [token]
    query_start_loc = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(row_valid.astype(jnp.int32)),
        )
    )
    token_row_ids = jnp.arange(row_count, dtype=jnp.int32)[None, :]

    for offset in range(1, int(width)):
        positions = (start_positions + offset - 1)[None, :].astype(jnp.int32)
        seq_lens = jnp.where(
            row_valid,
            positions[0] + 1,
            jnp.zeros_like(positions[0]),
        )
        metadata = backend.build_attention_metadata(
            positions=positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=config.capacity.block_size,
            is_prefill=True,
            query_start_loc=query_start_loc,
            num_prefill_tokens=row_count,
            num_decode_tokens=0,
            token_row_ids=token_row_ids,
            max_query_len=1,
        )
        hidden, cache_storage = mtp_forward_cached(
            jnp.swapaxes(hidden, 0, 1),
            token[None, :],
            embed_tokens=embed_tokens,
            params=params,
            config=config,
            positions=positions,
            cache_storage=cache_storage,
            metadata=metadata,
            backend=backend,
        )
        hidden = jnp.swapaxes(hidden, 0, 1)
        token = mtp_token_ids(
            hidden,
            embed_tokens=embed_tokens,
            config=config,
        )[:, 0]
        drafts.append(token)
    return jnp.stack(drafts, axis=1).astype(jnp.int32), cache_storage
