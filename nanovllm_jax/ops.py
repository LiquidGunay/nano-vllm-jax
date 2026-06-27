"""Serving operation implementations.

The engine owns scheduling and logical cache metadata. This module owns the
operation implementations selected by ``fastpath.py``. It is intentionally a
single serving operation surface, not a backend-selection hierarchy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import jax
import jax.numpy as jnp
from nanovllm_jax.cache import (
    AttentionMetadata,
    FullAttentionNHDKVCacheStorage,
    KVCacheSpec,
    KVCacheStorage,
    cap_num_kv_cache_blocks,
    compute_slot_mapping,
    init_kv_cache,
    init_full_attention_nhd_kv_cache,
    paged_attention,
    paged_attention_prefill,
    paged_attention_prefill_packed,
    paged_attention_decode,
    compute_packed_slot_mapping,
    update_kv_cache,
)

_TRUE_CONFIG_VALUES = {"1", "true", "yes", "on"}
_OFF_CONFIG_VALUES = {"", "0", "false", "no", "off", "none"}


def _config_bool(config, attr: str, *, default: bool = False) -> bool:
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return default


def _config_str(config, attr: str, *, default: str) -> str:
    if config is not None and hasattr(config, attr):
        return str(getattr(config, attr) or default).strip().lower()
    return default


def _config_int_or_none(config, attr: str) -> int | None:
    if config is not None and hasattr(config, attr):
        value = getattr(config, attr)
        if value is None:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _full_attention_kv_cache_dtype(default_dtype, config=None):
    value = _config_str(
        config,
        "full_attention_kv_cache_dtype",
        default="default",
    )
    if value == "default" or value in _OFF_CONFIG_VALUES:
        return default_dtype
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if value in {"fp16", "float16"}:
        return jnp.float16
    if value in {"fp32", "float32"}:
        return jnp.float32
    raise ValueError(
        f"Unknown full_attention_kv_cache_dtype={value!r}; "
        "expected fp32, bf16, fp16, default, or off"
    )


def _full_attention_kv_append_impl(config=None) -> str:
    value = _config_str(config, "full_attention_kv_append_impl", default="reference")
    if value == "reference":
        return "reference"
    raise ValueError("full_attention_kv_append_impl must be exactly 'reference'")


def _full_attention_decode_impl(config=None) -> str:
    value = _config_str(config, "full_attention_decode_impl", default="reference")
    if value in {"reference", "flashinfer_paged"}:
        return value
    raise ValueError("full_attention_decode_impl must be 'reference' or 'flashinfer_paged'")


def _full_attention_prefill_impl(config=None) -> str:
    value = _config_str(config, "full_attention_prefill_impl", default="reference")
    if value in {"reference", "triton_packed"}:
        return value
    raise ValueError("full_attention_prefill_impl must be 'reference' or 'triton_packed'")


def _gdn_disable_fallbacks(config=None) -> bool:
    return _config_bool(
        config,
        "gdn_disable_fallbacks",
    )


def gdn_disable_fallbacks_enabled(config=None) -> bool:
    """Return whether GDN kernel requests must fail instead of falling back."""

    return _gdn_disable_fallbacks(config)


def _raise_if_gdn_fallback_disabled(reason: str, config=None) -> None:
    if _gdn_disable_fallbacks(config):
        raise RuntimeError(
            f"{reason}; implicit GDN kernel fallbacks are disabled by "
            "gdn_disable_fallbacks=True"
        )


def _gdn_packed_decode_impl(config=None) -> str:
    value = _config_str(config, "gdn_packed_decode_impl", default="off")
    if value in {"off", "reference"}:
        return value
    raise ValueError("gdn_packed_decode_impl must be 'off' or 'reference'")


def gdn_packed_decode_impl(config=None) -> str:
    """Return the normalized packed GDN decode implementation name."""

    return _gdn_packed_decode_impl(config)


def gdn_packed_decode_enabled(config=None) -> bool:
    return _gdn_packed_decode_impl(config) != "off"


def gdn_packed_decode_max_batch(config=None) -> int | None:
    return _config_int_or_none(
        config,
        "gdn_packed_decode_max_batch",
    )


def _gdn_prefill_post_conv_impl(config=None) -> str:
    value = _config_str(config, "gdn_prefill_post_conv_impl", default="off")
    if value in {"off", "reference", "triton_fla_padded"}:
        return value
    raise ValueError("gdn_prefill_post_conv_impl must be 'off', 'reference', or 'triton_fla_padded'")


def gdn_prefill_post_conv_enabled(config=None) -> bool:
    return _gdn_prefill_post_conv_impl(config) != "off"


def _normalize_gdn_prefill_dtype(value: str, field_name: str) -> str:
    normalized = value.lower()
    if normalized in _OFF_CONFIG_VALUES or normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(f"Unknown {field_name}={value!r}; expected fp32 or bf16")


def _gdn_prefill_qkv_activation_dtype(config=None) -> str:
    if config is not None and hasattr(config, "gdn_prefill_qkv_dtype"):
        value = getattr(config, "gdn_prefill_qkv_dtype")
    else:
        value = "fp32"
    return _normalize_gdn_prefill_dtype(str(value).strip(), "gdn_prefill_qkv_dtype")


def _gdn_packed_decode_qkv_activation_dtype(config=None) -> str:
    value = _config_str(
        config,
        "gdn_packed_decode_qkv_dtype",
        default="fp32",
    )
    return _normalize_gdn_prefill_dtype(value, "gdn_packed_decode_qkv_dtype")


def _gdn_prefill_activation_dtype(config=None) -> str:
    return _gdn_prefill_qkv_activation_dtype(config)


def _gdn_prefill_qkv_activation_jnp_dtype(config=None) -> jnp.dtype:
    if _gdn_prefill_qkv_activation_dtype(config) == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _gdn_packed_decode_qkv_activation_jnp_dtype(config=None) -> jnp.dtype:
    if _gdn_packed_decode_qkv_activation_dtype(config) == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _gdn_packed_decode_pre_normalize_qk(config=None) -> bool:
    return _config_bool(
        config,
        "gdn_packed_decode_pre_normalize_qk",
    )


def _gdn_prefill_fla_vllm_like_enabled() -> bool:
    return False


def _gdn_prefill_post_conv_output_dtype(config=None) -> str:
    value = _config_str(
        config,
        "gdn_prefill_post_conv_output_dtype",
        default="fp32",
    )
    return _normalize_gdn_prefill_dtype(
        value,
        "gdn_prefill_post_conv_output_dtype",
    )


def _cast_gdn_prefill_activations(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    config=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    act_dtype = _gdn_prefill_qkv_activation_jnp_dtype(config)
    if act_dtype == jnp.bfloat16:
        return (
            query.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
        )
    return query, key, value


def _cast_gdn_prefill_post_conv_output(output: jnp.ndarray, config=None) -> jnp.ndarray:
    if _gdn_prefill_post_conv_output_dtype(config) == "bf16":
        return output.astype(jnp.bfloat16)
    return output


def _static_packed_gdn_chunk_metadata(
    *,
    row_count: int,
    token_bucket: int,
    chunk_size: int,
    max_row_tokens: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    if row_count <= 0:
        raise ValueError("packed GDN prefill requires at least one row")
    if token_bucket <= 0:
        raise ValueError("packed GDN prefill requires a positive token bucket")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    row_token_bucket = token_bucket
    if max_row_tokens is not None and max_row_tokens > 0:
        row_token_bucket = min(token_bucket, int(max_row_tokens))
    max_row_chunks = (row_token_bucket + chunk_size - 1) // chunk_size
    rows = jnp.repeat(
        jnp.arange(row_count, dtype=jnp.int32),
        max_row_chunks,
    )
    chunks = jnp.tile(
        jnp.arange(max_row_chunks, dtype=jnp.int32),
        row_count,
    )
    chunk_indices = jnp.stack((rows, chunks), axis=1)
    chunk_offsets = (
        jnp.arange(row_count + 1, dtype=jnp.int32) * jnp.int32(max_row_chunks)
    )
    return chunk_indices, chunk_offsets, max_row_chunks


def _prepare_packed_gdn_post_conv_inputs_from_decay(
    conv_out: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    decay: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    use_qk_l2norm_in_kernel: bool,
    config=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if conv_out.ndim != 3 or conv_out.shape[0] != 1:
        raise ValueError("packed conv_out must have shape [1, token_bucket, conv_dim]")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != 1 or b.shape[0] != 1:
        raise ValueError("packed a/b must have shape [1, token_bucket, value_heads]")
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must have shape [row_count + 1]")
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")

    _, token_bucket, conv_dim = conv_out.shape
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    expected_conv_dim = 2 * key_dim + value_dim
    if conv_dim != expected_conv_dim:
        raise ValueError(
            f"conv_out last dimension must be {expected_conv_dim}, got {conv_dim}"
        )
    if a.shape != (1, token_bucket, num_value_heads):
        raise ValueError("a must have shape [1, token_bucket, value_heads]")
    if b.shape != (1, token_bucket, num_value_heads):
        raise ValueError("b must have shape [1, token_bucket, value_heads]")
    if decay.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("decay and dt_bias must have shape [value_heads]")

    query = conv_out[:, :, :key_dim].reshape(
        1,
        token_bucket,
        num_key_heads,
        key_head_dim,
    )
    key = conv_out[:, :, key_dim : key_dim * 2].reshape(
        1,
        token_bucket,
        num_key_heads,
        key_head_dim,
    )
    value = conv_out[:, :, key_dim * 2 :].reshape(
        1,
        token_bucket,
        num_value_heads,
        value_head_dim,
    )
    beta = jax.nn.sigmoid(b)
    gate = -decay * jax.nn.softplus(a + dt_bias)

    heads_per_key = num_value_heads // num_key_heads
    if heads_per_key > 1:
        query = jnp.repeat(query, heads_per_key, axis=2)
        key = jnp.repeat(key, heads_per_key, axis=2)

    if use_qk_l2norm_in_kernel:
        from nanovllm_jax.layers import l2norm

        query = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6)
        key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)

    valid = (
        jnp.arange(token_bucket, dtype=jnp.int32)
        < query_start_loc[-1].astype(jnp.int32)
    )
    query = jnp.where(valid[None, :, None, None], query, 0.0)
    key = jnp.where(valid[None, :, None, None], key, 0.0)
    value = jnp.where(valid[None, :, None, None], value, 0.0)
    gate = jnp.where(valid[None, :, None], gate, 0.0)
    beta = jnp.where(valid[None, :, None], beta, 0.0)

    qkv_dtype = _gdn_prefill_qkv_activation_jnp_dtype(config)
    packed_query = query.reshape(token_bucket, num_value_heads, key_head_dim)
    packed_key = key.reshape(token_bucket, num_value_heads, key_head_dim)
    packed_value = value.reshape(token_bucket, num_value_heads, value_head_dim)
    packed_gate = gate.reshape(token_bucket, num_value_heads)
    packed_beta = beta.reshape(token_bucket, num_value_heads)
    return (
        packed_query.astype(qkv_dtype),
        packed_key.astype(qkv_dtype),
        packed_value.astype(qkv_dtype),
        packed_gate.astype(jnp.float32),
        packed_beta.astype(jnp.float32),
        valid,
    )


class ServingOpsProtocol(Protocol):
    """Operation API used by the runner and model."""

    name: str

    def allocate_kv_cache(
        self,
        spec: KVCacheSpec,
        max_seqs: int,
        max_blocks_per_seq: int,
    ) -> KVCacheStorage:
        ...

    def allocate_full_attention_nhd_kv_cache(
        self,
        spec: KVCacheSpec,
        full_attention_layers: tuple[int, ...],
    ) -> FullAttentionNHDKVCacheStorage | None:
        ...

    def build_attention_metadata(
        self,
        positions: jnp.ndarray,
        block_tables: jnp.ndarray,
        seq_lens: jnp.ndarray,
        block_size: int,
        is_prefill: bool,
        query_start_loc: jnp.ndarray | None = None,
        num_prefill_tokens: int | None = None,
        num_decode_tokens: int | None = None,
        token_row_ids: jnp.ndarray | None = None,
        max_query_len: int | None = None,
    ) -> AttentionMetadata:
        ...

    def write_kv(
        self,
        layer_id: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
    ) -> KVCacheStorage:
        ...

    def attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> jnp.ndarray:
        ...

    def write_kv_and_attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> tuple[KVCacheStorage, jnp.ndarray]:
        ...

    def gated_delta_prefill(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        valid_token_mask: jnp.ndarray | None,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_packed_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
        max_row_tokens: int | None = None,
        return_prefix_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_decode(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...

    def gated_delta_packed_decode(
        self,
        mixed_qkv: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        initial_state: jnp.ndarray,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...



class ServingOps:
    """Operation dispatcher for the promoted serving path."""

    name = "serving_manifest"

    def __init__(self, config=None):
        self.config = config

    def allocate_kv_cache(
        self,
        spec: KVCacheSpec,
        max_seqs: int,
        max_blocks_per_seq: int,
    ) -> KVCacheStorage:
        cache_dtype = _full_attention_kv_cache_dtype(spec.dtype, self.config)
        capped_spec = replace(
            spec,
            dtype=cache_dtype,
            num_blocks=cap_num_kv_cache_blocks(replace(spec, dtype=cache_dtype)),
        )
        state = init_kv_cache(
            num_blocks=capped_spec.num_blocks,
            block_size=capped_spec.block_size,
            num_kv_heads=capped_spec.num_kv_heads,
            head_dim=capped_spec.head_dim,
            max_seqs=max_seqs,
            max_blocks_per_seq=max_blocks_per_seq,
            num_layers=capped_spec.num_layers,
            dtype=capped_spec.dtype,
            max_kv_cache_bytes=capped_spec.max_kv_cache_bytes,
        )
        return state.storage

    def allocate_full_attention_nhd_kv_cache(
        self,
        spec: KVCacheSpec,
        full_attention_layers: tuple[int, ...],
    ) -> FullAttentionNHDKVCacheStorage | None:
        if _full_attention_decode_impl(self.config) != "flashinfer_paged":
            return None
        cache_dtype = _full_attention_kv_cache_dtype(spec.dtype, self.config)
        return init_full_attention_nhd_kv_cache(
            spec=replace(
                spec,
                dtype=cache_dtype,
                num_blocks=cap_num_kv_cache_blocks(replace(spec, dtype=cache_dtype)),
            ),
            full_attention_layers=full_attention_layers,
        )

    def build_attention_metadata(
        self,
        positions: jnp.ndarray,
        block_tables: jnp.ndarray,
        seq_lens: jnp.ndarray,
        block_size: int,
        is_prefill: bool,
        query_start_loc: jnp.ndarray | None = None,
        num_prefill_tokens: int | None = None,
        num_decode_tokens: int | None = None,
        token_row_ids: jnp.ndarray | None = None,
        max_query_len: int | None = None,
    ) -> AttentionMetadata:
        if positions.ndim != 2:
            raise ValueError("positions must be a 2D tensor [batch, query_len]")
        if block_tables.ndim != 2:
            raise ValueError("block_tables must be a 2D tensor [batch, max_blocks_per_seq]")
        packed_prefill = token_row_ids is not None
        if packed_prefill:
            if not is_prefill:
                raise ValueError("token_row_ids are only supported for packed prefill")
            if token_row_ids.shape != positions.shape:
                raise ValueError("token_row_ids and positions must have matching shapes")
            if seq_lens.shape[0] != block_tables.shape[0]:
                raise ValueError("seq_lens shape must align with packed prefill rows")
            slot_mapping = compute_packed_slot_mapping(
                positions=positions,
                block_table=block_tables,
                token_row_ids=token_row_ids,
                block_size=block_size,
            )
        else:
            if positions.shape[0] != block_tables.shape[0]:
                raise ValueError(
                    "positions and block_tables batch dimensions must match"
                )
            if positions.shape[0] != seq_lens.shape[0]:
                raise ValueError("positions and seq_lens batch dimensions must match")
            slot_mapping = compute_slot_mapping(
                positions=positions,
                block_table=block_tables,
                block_size=block_size,
                is_prefill=is_prefill,
            )
        batch, query_len = positions.shape
        if query_start_loc is None:
            query_lens = jnp.where(is_prefill, seq_lens, jnp.ones_like(seq_lens))
            query_start_loc = jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens.astype(jnp.int32)),
                ]
            )
        if num_prefill_tokens is None:
            num_prefill_tokens = batch * query_len if is_prefill else 0
        if num_decode_tokens is None:
            num_decode_tokens = 0 if is_prefill else batch
        return AttentionMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            positions=positions,
            max_kv_len=block_tables.shape[1] * block_size if not is_prefill else None,
            token_row_ids=token_row_ids,
            max_query_len=max_query_len if max_query_len is not None else query_len,
        )

    def write_kv(
        self,
        layer_id: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
    ) -> KVCacheStorage:
        if _full_attention_kv_append_impl(self.config) != "reference":
            raise AssertionError("KV append policy was validated incorrectly")
        if metadata.token_row_ids is not None:
            actual_tokens = metadata.query_start_loc[-1].astype(jnp.int32)
            valid_mask = jnp.arange(metadata.slot_mapping.size, dtype=jnp.int32).reshape(
                metadata.slot_mapping.shape
            ) < actual_tokens
        else:
            query_lens = jnp.diff(metadata.query_start_loc).astype(jnp.int32)
            valid_mask = jnp.arange(metadata.slot_mapping.shape[1])[None, :] < query_lens[:, None]
        k_cache, v_cache = update_kv_cache(
            cache.k_cache,
            cache.v_cache,
            metadata.slot_mapping,
            k,
            v,
            layer_idx=layer_id,
            valid_mask=valid_mask,
        )
        return KVCacheStorage(k_cache, v_cache)

    def attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> jnp.ndarray:
        if is_prefill:
            prefill_impl = _full_attention_prefill_impl(self.config)
            if metadata.token_row_ids is not None:
                if metadata.positions is None:
                    raise ValueError("metadata.positions is required for packed prefill attention")
                return paged_attention_prefill_packed(
                    query=query,
                    k_cache=cache.k_cache,
                    v_cache=cache.v_cache,
                    block_table=metadata.block_tables,
                    kv_lens=metadata.seq_lens,
                    positions=metadata.positions,
                    token_row_ids=metadata.token_row_ids,
                    query_start_loc=metadata.query_start_loc,
                    block_size=block_size,
                    scale=scale,
                    num_key_value_groups=num_key_value_groups,
                    layer_idx=layer_id,
                    max_query_len=metadata.max_query_len,
                    use_triton=prefill_impl == "triton_packed",
                )
            if prefill_impl != "reference":
                raise ValueError("triton_packed full-attention prefill requires packed metadata")
            if metadata.positions is None:
                return paged_attention(
                    query=query,
                    k_cache=cache.k_cache,
                    v_cache=cache.v_cache,
                    slot_mapping=metadata.slot_mapping,
                    kv_lens=metadata.seq_lens,
                    scale=scale,
                    num_key_value_groups=num_key_value_groups,
                    layer_idx=layer_id,
                )
            return paged_attention_prefill(
                query=query,
                k_cache=cache.k_cache,
                v_cache=cache.v_cache,
                block_table=metadata.block_tables,
                kv_lens=metadata.seq_lens,
                positions=metadata.positions,
                block_size=block_size,
                scale=scale,
                num_key_value_groups=num_key_value_groups,
                layer_idx=layer_id,
            )

        if metadata.positions is None:
            raise ValueError("metadata.positions is required for decode attention")
        if metadata.positions.shape[0] != metadata.block_tables.shape[0]:
            raise ValueError("positions and block_tables batch dimensions must match")
        if _full_attention_decode_impl(self.config) == "flashinfer_paged" and query.shape[1] == 1:
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError("FlashInfer decode requires NHD cache storage")
            if cache.k_cache.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
                raise ValueError("FlashInfer decode requires BF16/FP16 NHD KV cache")
            from nanovllm_jax.kernels.flashinfer_ffi import paged_decode_attention_gqa_nhd
            from nanovllm_jax.kernels.paged_attention import dense_block_tables_to_kv_indptr, kv_last_page_len_from_seq_lens

            kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(metadata.block_tables)
            out = paged_decode_attention_gqa_nhd(
                query[:, 0].astype(cache.k_cache.dtype),
                cache.k_cache[layer_id],
                cache.v_cache[layer_id],
                kv_indptr,
                kv_indices,
                kv_last_page_len_from_seq_lens(metadata.seq_lens, block_size),
                scale=scale,
            )
            return out.reshape(query.shape[0], 1, query.shape[2] * query.shape[3])
        return paged_attention_decode(
            query=query,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            block_table=metadata.block_tables,
            kv_lens=metadata.seq_lens,
            block_size=block_size,
            scale=scale,
            num_key_value_groups=num_key_value_groups,
            layer_idx=layer_id,
            max_kv_len=metadata.max_kv_len,
            positions=metadata.positions,
        )

    def write_kv_and_attention(
        self,
        layer_id: int,
        query: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cache: KVCacheStorage,
        metadata: AttentionMetadata,
        block_size: int,
        scale: float,
        num_key_value_groups: int,
        is_prefill: bool,
    ) -> tuple[KVCacheStorage, jnp.ndarray]:
        if not is_prefill and _full_attention_decode_impl(self.config) == "flashinfer_paged":
            if metadata.positions is None:
                raise ValueError("metadata.positions is required for FlashInfer decode attention")
            if cache.k_cache.ndim != 5 or cache.v_cache.ndim != 5:
                raise ValueError("FlashInfer decode requires NHD cache storage")
            if cache.k_cache.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
                raise ValueError("FlashInfer decode requires BF16/FP16 NHD KV cache")
            from nanovllm_jax.kernels.flashinfer_ffi import paged_decode_attention_with_kv_append_gqa_nhd
            from nanovllm_jax.kernels.paged_attention import dense_block_tables_to_kv_indptr, kv_last_page_len_from_seq_lens

            kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(metadata.block_tables)
            k_cache, v_cache, outputs = cache.k_cache, cache.v_cache, []
            width = int(query.shape[1])
            for token_idx in range(width):
                seq_lens_step = metadata.seq_lens - jnp.asarray(width - 1 - token_idx, dtype=metadata.seq_lens.dtype)
                out, k_cache, v_cache = paged_decode_attention_with_kv_append_gqa_nhd(
                    query[:, token_idx].astype(cache.k_cache.dtype),
                    k[:, token_idx].astype(cache.k_cache.dtype),
                    v[:, token_idx].astype(cache.v_cache.dtype),
                    k_cache,
                    v_cache,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len_from_seq_lens(seq_lens_step, block_size),
                    metadata.positions[:, token_idx].astype(jnp.int32),
                    layer_id=layer_id,
                    scale=scale,
                )
                outputs.append(out.reshape(query.shape[0], 1, query.shape[2] * query.shape[3]))
            return KVCacheStorage(k_cache, v_cache), jnp.concatenate(outputs, axis=1)

        cache_storage = self.write_kv(layer_id, k, v, cache, metadata)
        out = self.attention(
            layer_id=layer_id,
            query=query,
            cache=cache_storage,
            metadata=metadata,
            block_size=block_size,
            scale=scale,
            num_key_value_groups=num_key_value_groups,
            is_prefill=is_prefill,
        )
        return cache_storage, out

    def gated_delta_prefill(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        from nanovllm_jax.model import jax_chunk_gated_delta_rule

        return jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output_final_state=True,
        )

    def gated_delta_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        valid_token_mask: jnp.ndarray | None,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        impl = _gdn_prefill_post_conv_impl(self.config)
        if impl == "off":
            raise RuntimeError("gdn_prefill_post_conv_impl is off; use gated_delta_prefill")
        if impl == "reference":
            from nanovllm_jax.kernels.gdn_fla import gdn_post_conv_prefill_reference_from_decay

            return gdn_post_conv_prefill_reference_from_decay(
                conv_out,
                a,
                b,
                decay,
                dt_bias,
                valid_token_mask,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                chunk_size=chunk_size,
                initial_state=initial_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if impl == "triton_fla_padded":
            from nanovllm_jax.kernels.gdn_fla import (
                prepare_gdn_fla_prefill_kernel_inputs,
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay,
                gdn_fla_prefill_chunk32_fp32_reference,
            )
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_fla_chunk_gated_delta_rule_packed_triton,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                gdn_fla_chunk_gated_delta_rule_packed_triton = None

            if initial_state is None:
                initial_state = jnp.zeros(
                    (
                        conv_out.shape[0],
                        num_value_heads,
                        value_head_dim,
                        key_head_dim,
                    ),
                    dtype=jnp.float32,
                )
            query, key, value, gate, beta, seq_lens = (
                prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
                    conv_out,
                    a,
                    b,
                    decay,
                    dt_bias,
                    valid_token_mask,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    normalize_qk=use_qk_l2norm_in_kernel,
                )
            )
            prepared = prepare_gdn_fla_prefill_kernel_inputs(
                query,
                key,
                value,
                gate.astype(jnp.float32),
                beta.astype(jnp.float32),
                seq_lens,
                initial_state.astype(jnp.float32),
                qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(self.config),
            )
            packed_query = prepared.query.reshape(
                -1,
                prepared.query.shape[2],
                prepared.query.shape[3],
            )
            packed_key = prepared.key.reshape(
                -1,
                prepared.key.shape[2],
                prepared.key.shape[3],
            )
            packed_value = prepared.value.reshape(
                -1,
                prepared.value.shape[2],
                prepared.value.shape[3],
            )
            packed_gate = prepared.gate.reshape(
                -1,
                prepared.gate.shape[2],
            )
            packed_beta = prepared.beta.reshape(
                -1,
                prepared.beta.shape[2],
            )
            batch, seq_len, num_key_heads_out, key_head_dim_prepared = (
                prepared.query.shape
            )
            if key_head_dim_prepared != key_head_dim:
                raise ValueError(
                    "prepared query head dim must match configured key_head_dim"
                )
            value_dim = prepared.value.shape[-1]
            packed_cu_seqlens = (
                jnp.arange(batch + 1, dtype=jnp.int32) * jnp.int32(seq_len)
            )
            max_chunks_per_row = (seq_len + chunk_size - 1) // chunk_size
            row_ids = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), max_chunks_per_row)
            chunk_ids = jnp.tile(
                jnp.arange(max_chunks_per_row, dtype=jnp.int32), batch
            )
            packed_chunk_indices = jnp.stack((row_ids, chunk_ids), axis=1)
            packed_chunk_offsets = (
                jnp.arange(batch + 1, dtype=jnp.int32) * max_chunks_per_row
            )

            if gdn_fla_chunk_gated_delta_rule_packed_triton is None:
                _raise_if_gdn_fallback_disabled(
                    "Triton FLA padded prefill kernel is unavailable",
                    self.config,
                )
                output, final_state = gdn_fla_prefill_chunk32_fp32_reference(
                    prepared.query,
                    prepared.key,
                    prepared.value,
                    prepared.gate,
                    prepared.beta,
                    prepared.seq_lens,
                    prepared.initial_state,
                    chunk_size=chunk_size,
                )
                output = output.reshape(batch, seq_len, num_key_heads_out, value_dim)
                output = _cast_gdn_prefill_post_conv_output(output, self.config)
                return output.transpose(0, 2, 1, 3), final_state

            output, final_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
                packed_query,
                packed_key,
                packed_value,
                packed_gate,
                packed_beta,
                packed_cu_seqlens,
                prepared.initial_state,
                chunk_size=chunk_size,
                # Q/K normalization was already applied during post-conv prep
                # when requested; do not normalize the packed tensors again.
                use_qk_l2norm_in_kernel=False,
                max_row_chunks=max_chunks_per_row,
                chunk_indices=packed_chunk_indices,
                chunk_offsets=packed_chunk_offsets,
            )
            output = output.reshape(batch, seq_len, num_key_heads_out, value_dim)
            output = _cast_gdn_prefill_post_conv_output(output, self.config)
            return output.transpose(0, 2, 1, 3), final_state


        raise AssertionError(f"Unhandled GDN post-conv prefill implementation {impl!r}")

    def gated_delta_packed_prefill_post_conv(
        self,
        conv_out: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
        max_row_tokens: int | None = None,
        return_prefix_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        impl = _gdn_prefill_post_conv_impl(self.config)
        if impl == "off":
            raise RuntimeError(
                "gdn_prefill_post_conv_impl is off; use reference packed prefill"
            )

        token_bucket = int(conv_out.shape[1])
        row_count = int(query_start_loc.shape[0]) - 1
        if initial_state is None:
            initial_state = jnp.zeros(
                (
                    row_count,
                    num_value_heads,
                    value_head_dim,
                    key_head_dim,
                ),
                dtype=jnp.float32,
            )

        (
            packed_query,
            packed_key,
            packed_value,
            packed_gate,
            packed_beta,
            valid_tokens,
        ) = _prepare_packed_gdn_post_conv_inputs_from_decay(
            conv_out,
            a,
            b,
            decay,
            dt_bias,
            query_start_loc,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            config=self.config,
        )

        def reference_scan() -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            token_positions = jnp.arange(token_bucket, dtype=jnp.int32)
            cu = query_start_loc.astype(jnp.int32)
            row_ids = jnp.sum(
                token_positions[:, None] >= cu[1:][None, :],
                axis=1,
            ).astype(jnp.int32)
            safe_rows = jnp.clip(row_ids, 0, row_count - 1)
            query_tokens = packed_query.astype(jnp.float32) * (
                1.0 / jnp.sqrt(key_head_dim)
            )
            key_tokens = packed_key.astype(jnp.float32)
            value_tokens = packed_value.astype(jnp.float32)
            gate_tokens = packed_gate.astype(jnp.float32)
            beta_tokens = packed_beta.astype(jnp.float32)

            def recurrent_scan(state, inputs):
                row, valid, q_t, k_t, v_t, gate_t, beta_t = inputs
                previous_row_state = state[row]
                decay_t = jnp.exp(gate_t)[:, None, None]
                decayed_state = previous_row_state * decay_t
                kv_mem = jnp.einsum("hvk,hk->hv", decayed_state, k_t)
                delta = (v_t - kv_mem) * beta_t[:, None]
                next_row_state = decayed_state + delta[:, :, None] * k_t[:, None, :]
                out_t = jnp.einsum("hvk,hk->hv", next_row_state, q_t)
                next_row_state = jnp.where(valid, next_row_state, previous_row_state)
                state = state.at[row].set(next_row_state)
                out_t = jnp.where(valid, out_t, jnp.zeros_like(out_t))
                return state, (out_t, next_row_state)

            final_state, (output_tokens, prefix_states) = jax.lax.scan(
                recurrent_scan,
                initial_state.astype(jnp.float32),
                (
                    safe_rows,
                    valid_tokens,
                    query_tokens,
                    key_tokens,
                    value_tokens,
                    gate_tokens,
                    beta_tokens,
                ),
            )
            output = _cast_gdn_prefill_post_conv_output(output_tokens, self.config)
            if return_prefix_state:
                return output[None, :, :, :], final_state, prefix_states
            return output[None, :, :, :], final_state

        if impl == "reference":
            return reference_scan()
        if impl != "triton_fla_padded":
            _raise_if_gdn_fallback_disabled(
                f"{impl!r} does not support packed GDN prefill ABI",
                self.config,
            )
            return reference_scan()

        try:
            from nanovllm_jax.kernels.gdn_fla_triton import (
                gdn_fla_chunk_gated_delta_rule_packed_triton,
            )
        except (ImportError, ModuleNotFoundError, AttributeError):
            gdn_fla_chunk_gated_delta_rule_packed_triton = None

        if gdn_fla_chunk_gated_delta_rule_packed_triton is None:
            _raise_if_gdn_fallback_disabled(
                "Triton FLA packed prefill kernel is unavailable",
                self.config,
            )
            return reference_scan()

        if return_prefix_state:
            if max_row_tokens is None:
                _raise_if_gdn_fallback_disabled(
                    "Packed-prefix GDN route requires static max_row_tokens",
                    self.config,
                )
                return reference_scan()
            if int(max_row_tokens) > 16:
                _raise_if_gdn_fallback_disabled(
                    "Packed-prefix GDN route requires max_row_tokens <= 16",
                    self.config,
                )
                return reference_scan()
            try:
                from nanovllm_jax.kernels.gdn_fla_triton import (
                    gdn_packed_prefix_state_triton,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                gdn_packed_prefix_state_triton = None
            if gdn_packed_prefix_state_triton is None:
                _raise_if_gdn_fallback_disabled(
                    "Tiny packed-prefix GDN kernel is unavailable",
                    self.config,
                )
                return reference_scan()
            output, final_state, prefix_states = gdn_packed_prefix_state_triton(
                packed_query,
                packed_key,
                packed_value,
                packed_gate,
                packed_beta,
                query_start_loc.astype(jnp.int32),
                initial_state.astype(jnp.float32),
                max_row_tokens=int(max_row_tokens),
            )
            output = jnp.where(valid_tokens[:, None, None], output, 0.0)
            output = _cast_gdn_prefill_post_conv_output(output, self.config)
            prefix_states = jnp.where(
                valid_tokens[:, None, None, None],
                prefix_states,
                0.0,
            )
            return output[None, :, :, :], final_state, prefix_states

        chunk_indices, chunk_offsets, max_row_chunks = (
            _static_packed_gdn_chunk_metadata(
                row_count=row_count,
                token_bucket=token_bucket,
                chunk_size=chunk_size,
                max_row_tokens=max_row_tokens,
            )
        )
        output, final_state = gdn_fla_chunk_gated_delta_rule_packed_triton(
            packed_query,
            packed_key,
            packed_value,
            packed_gate,
            packed_beta,
            query_start_loc.astype(jnp.int32),
            initial_state.astype(jnp.float32),
            chunk_size=chunk_size,
            # Q/K normalization is applied above when requested.
            use_qk_l2norm_in_kernel=False,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            max_row_chunks=max_row_chunks,
        )
        output = jnp.where(valid_tokens[:, None, None], output, 0.0)
        output = _cast_gdn_prefill_post_conv_output(output, self.config)
        return output[None, :, :, :], final_state

    def gated_delta_decode(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        g: jnp.ndarray,
        beta: jnp.ndarray,
        initial_state: jnp.ndarray | None,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        from nanovllm_jax.model import jax_recurrent_gated_delta_rule

        return jax_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def gated_delta_packed_decode(
        self,
        mixed_qkv: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        decay: jnp.ndarray,
        dt_bias: jnp.ndarray,
        initial_state: jnp.ndarray,
        use_qk_l2norm_in_kernel: bool,
        *,
        z: jnp.ndarray | None = None,
        norm_weight: jnp.ndarray | None = None,
        rms_norm_eps: float = 1.0e-6,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if _gdn_packed_decode_impl(self.config) != "reference":
            raise RuntimeError("gdn_packed_decode_impl is off; use gated_delta_decode")
        from nanovllm_jax.kernels.gdn_fla import (
            gdn_packed_decode_reference_from_decay,
            prepare_gdn_packed_decode_reference_from_decay_inputs,
        )

        mixed_qkv, a, b, decay, dt_bias, initial_state = (
            prepare_gdn_packed_decode_reference_from_decay_inputs(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                decay=decay,
                dt_bias=dt_bias,
                state=initial_state,
                qkv_dtype=_gdn_packed_decode_qkv_activation_jnp_dtype(self.config),
            )
        )
        return gdn_packed_decode_reference_from_decay(
            mixed_qkv,
            a,
            b,
            decay,
            dt_bias,
            initial_state,
            qkv_dtype=_gdn_packed_decode_qkv_activation_jnp_dtype(self.config),
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
