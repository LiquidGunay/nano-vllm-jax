"""KV Cache implementation for Qwen 3.5 JAX with vLLM paging compatibility."""

import jax
import jax.numpy as jnp
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, replace


@dataclass
class KVCacheSpec:
    """Logical KV cache shape requested by the engine."""

    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: Any = jnp.float32
    max_kv_cache_bytes: Optional[int] = None


@dataclass
class KVCacheStorage:
    """Backend-owned physical KV arrays.

    The pure JAX backend uses the canonical teaching layout:
    [num_layers, num_blocks, block_size, num_kv_heads, head_dim].
    GPU/TPU backends can use different physical layouts behind this boundary.
    """

    k_cache: jnp.ndarray
    v_cache: jnp.ndarray


@dataclass(frozen=True)
class FullAttentionNHDKVCacheStorage:
    """Optional NHD cache for full-attention external kernels.

    Shape: [num_full_attention_layers, num_pages, page_size, num_kv_heads, head_dim].
    Individual kernel calls receive one layer slice with FlashInfer's NHD layout:
    [num_pages, page_size, num_kv_heads, head_dim].
    """

    k_cache: jnp.ndarray
    v_cache: jnp.ndarray
    layer_indices: tuple[int, ...]
    page_size: int
    layout: str = "NHD"


@dataclass
class AttentionMetadata:
    """Per-step paged/ragged attention metadata."""

    slot_mapping: jnp.ndarray
    block_tables: jnp.ndarray
    seq_lens: jnp.ndarray
    query_start_loc: jnp.ndarray
    num_prefill_tokens: int
    num_decode_tokens: int
    positions: Optional[jnp.ndarray] = None
    max_kv_len: Optional[int] = None


@dataclass
class BlockTables:
    """Python/runtime-owned logical allocation state."""

    tables: List[List[int]]
    ref_counts: Optional[Any] = None
    hashes: Optional[Any] = None


@dataclass
class HybridLayerState:
    """Qwen3.5 linear/GDN state kept separate from full-attention KV storage."""

    conv_state: Optional[jnp.ndarray] = None
    recurrent_state: Optional[jnp.ndarray] = None


@dataclass
class KVCacheState:
    """KV cache state for batched inference with vLLM-style paging.
    
    This is a JAX-compatible dataclass that tracks:
    - k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
    - v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
    - block_table: Maps sequence -> blocks [max_seqs, max_blocks_per_seq]
    - kv_lens: Current length of each sequence [max_seqs]
    - slot_mapping: Maps token position -> (block_id, slot) [batch, seq_len]
    - conv_state: Linear attention conv state [batch, conv_dim, kernel_size]
    - recurrent_state: Linear attention recurrent state [batch, num_heads, v_dim, k_dim]
    
    For vLLM compatibility, we use:
    - Block-based KV cache (not contiguous)
    - Page table for logical->physical block mapping
    
    This dataclass is registered as a JAX pytree node for JIT compatibility.
    """
    # For num_layers == 1 we keep flat slots:
    # [1, num_blocks * block_size, num_kv_heads, head_dim]
    # For num_layers > 1 we keep block-grouped layout:
    # [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    k_cache: jnp.ndarray
    v_cache: jnp.ndarray
    block_table: jnp.ndarray  # [max_seqs, max_blocks_per_seq]
    kv_lens: jnp.ndarray  # [max_seqs]
    slot_mapping: jnp.ndarray  # [batch, seq_len]
    conv_state: Optional[jnp.ndarray] = None  # [batch, conv_dim, kernel_size]
    recurrent_state: Optional[jnp.ndarray] = None  # [batch, num_heads, v_dim, k_dim]

    @property
    def storage(self) -> KVCacheStorage:
        return KVCacheStorage(self.k_cache, self.v_cache)

    @property
    def attention_metadata(self) -> AttentionMetadata:
        query_len = self.slot_mapping.shape[1]
        batch = self.slot_mapping.shape[0]
        query_start_loc = jnp.arange(batch + 1, dtype=jnp.int32) * query_len
        return AttentionMetadata(
            slot_mapping=self.slot_mapping,
            block_tables=self.block_table,
            seq_lens=self.kv_lens,
            query_start_loc=query_start_loc,
            num_prefill_tokens=int(batch * query_len),
            num_decode_tokens=0,
            positions=None,
        )

    @property
    def hybrid_state(self) -> HybridLayerState:
        return HybridLayerState(self.conv_state, self.recurrent_state)

    def replace(self, **kwargs) -> "KVCacheState":
        return replace(self, **kwargs)


def estimate_kv_cache_bytes(spec: KVCacheSpec) -> int:
    """Return total bytes for K and V arrays for a cache spec."""
    dtype = jnp.dtype(spec.dtype)
    elements_per_cache = (
        spec.num_layers
        * spec.num_blocks
        * spec.block_size
        * spec.num_kv_heads
        * spec.head_dim
    )
    return int(elements_per_cache * dtype.itemsize * 2)


def cap_num_kv_cache_blocks(spec: KVCacheSpec) -> int:
    """Return the largest block count that fits the configured byte cap."""
    if spec.max_kv_cache_bytes is None:
        return spec.num_blocks

    bytes_per_block = estimate_kv_cache_bytes(replace(spec, num_blocks=1))
    if bytes_per_block <= 0:
        raise ValueError("Invalid KV cache spec: bytes_per_block must be positive")

    capped_blocks = spec.max_kv_cache_bytes // bytes_per_block
    if capped_blocks < 1:
        raise ValueError(
            "max_kv_cache_bytes is too small for one KV block "
            f"({spec.max_kv_cache_bytes} < {bytes_per_block})"
        )
    return min(spec.num_blocks, int(capped_blocks))


# Register KVCacheState as a JAX pytree node for JIT compatibility
def _kv_cache_state_flatten(state: KVCacheState):
    """Flatten KVCacheState into children and auxiliary data."""
    children = (
        state.k_cache,
        state.v_cache,
        state.block_table,
        state.kv_lens,
        state.slot_mapping,
        state.conv_state if state.conv_state is not None else jnp.zeros((1,), dtype=jnp.float32),
        state.recurrent_state if state.recurrent_state is not None else jnp.zeros((1,), dtype=jnp.float32),
    )
    aux_data = (
        state.conv_state is not None,
        state.recurrent_state is not None,
    )
    return children, aux_data


def _kv_cache_state_unflatten(aux_data, children):
    """Unflatten children and auxiliary data into KVCacheState."""
    (k_cache, v_cache, block_table, kv_lens, slot_mapping, conv_state, recurrent_state) = children
    (has_conv_state, has_recurrent_state) = aux_data
    
    return KVCacheState(
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        kv_lens=kv_lens,
        slot_mapping=slot_mapping,
        conv_state=conv_state if has_conv_state else None,
        recurrent_state=recurrent_state if has_recurrent_state else None,
    )


# Register the pytree node
jax.tree_util.register_pytree_node(
    KVCacheState,
    _kv_cache_state_flatten,
    _kv_cache_state_unflatten
)


def init_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqs: int,
    max_blocks_per_seq: int,
    num_layers: int = 24,
    dtype=jnp.float32,
    max_kv_cache_bytes: Optional[int] = None,
) -> KVCacheState:
    """Initialize empty KV cache.
    
    Args:
        num_blocks: Total number of blocks available
        block_size: Tokens per block (e.g., 16)
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        max_seqs: Maximum concurrent sequences
        max_blocks_per_seq: Max blocks per sequence
        num_layers: Number of transformer layers (for per-layer cache)
        dtype: Data type for KV cache
    
    Returns:
        Initialized KVCacheState with zeros
    """
    spec = KVCacheSpec(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        max_kv_cache_bytes=max_kv_cache_bytes,
    )
    capped_num_blocks = cap_num_kv_cache_blocks(spec)

    k_cache = jnp.zeros(
        (num_layers, capped_num_blocks, block_size, num_kv_heads, head_dim),
        dtype=dtype,
    )
    v_cache = jnp.zeros(
        (num_layers, capped_num_blocks, block_size, num_kv_heads, head_dim),
        dtype=dtype,
    )
    block_table = jnp.zeros((max_seqs, max_blocks_per_seq), dtype=jnp.int32)
    kv_lens = jnp.zeros(max_seqs, dtype=jnp.int32)
    slot_mapping = jnp.zeros((max_seqs, 1), dtype=jnp.int32)  # Will be expanded
    
    return KVCacheState(
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        kv_lens=kv_lens,
        slot_mapping=slot_mapping,
    )


def init_full_attention_nhd_kv_cache(
    spec: KVCacheSpec,
    full_attention_layers: tuple[int, ...],
) -> FullAttentionNHDKVCacheStorage:
    """Allocate an optional full-attention KV cache in NHD page layout.

    The caller is expected to pass an already-capped `spec.num_blocks`, so this
    helper does not change scheduler/block-manager capacity.
    """

    shape = full_attention_nhd_kv_cache_shape(spec, full_attention_layers)
    k_cache = jnp.zeros(
        shape,
        dtype=spec.dtype,
    )
    v_cache = jnp.zeros_like(k_cache)
    return FullAttentionNHDKVCacheStorage(
        k_cache=k_cache,
        v_cache=v_cache,
        layer_indices=tuple(int(layer) for layer in full_attention_layers),
        page_size=int(spec.block_size),
    )


def full_attention_nhd_kv_cache_shape(
    spec: KVCacheSpec,
    full_attention_layers: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    """Return the NHD full-attention cache shape without allocating arrays."""

    if not full_attention_layers:
        raise ValueError("full_attention_layers must contain at least one layer")
    return (
        len(full_attention_layers),
        int(spec.num_blocks),
        int(spec.block_size),
        int(spec.num_kv_heads),
        int(spec.head_dim),
    )


def init_linear_attention_states(
    cache: KVCacheState,
    config,
    batch_size: int = 1,
    dtype=None,
) -> KVCacheState:
    """Initialize linear attention cache states.
    
    Args:
        cache: Existing KVCacheState
        config: Model config with linear attention parameters
        batch_size: Batch size
        dtype: Data type for conv_state (defaults to config.get_dtype() or float32)
        
    Returns:
        Updated KVCacheState with initialized linear states
    """
    hybrid_state = init_hybrid_state(config, batch_size=batch_size, dtype=dtype)

    return replace(
        cache,
        conv_state=hybrid_state.conv_state,
        recurrent_state=hybrid_state.recurrent_state,
    )


def _resolve_kv_layer(
    cache: jnp.ndarray,
    layer_idx: int,
) -> Tuple[jnp.ndarray, bool]:
    """Resolve per-layer cache tensor.

    Returns:
        layer_cache: Cache for a single layer.
            - Flat layout: [num_slots, num_kv_heads, head_dim]
            - Block layout: [num_blocks, block_size, num_kv_heads, head_dim]
        flat: True when layer_cache is already flat [num_slots, ...]
    """
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")

    if cache.ndim == 5:
        if layer_idx >= cache.shape[0]:
            raise ValueError(
                f"layer_idx {layer_idx} is out of bounds for {cache.shape[0]} layers"
            )
        return cache[layer_idx], False
    if cache.ndim == 4:
        if layer_idx >= cache.shape[0]:
            raise ValueError(
                f"layer_idx {layer_idx} is out of bounds for {cache.shape[0]} layers"
            )
        return cache[layer_idx], True

    raise ValueError(f"Unsupported KV cache rank: {cache.ndim}")


def init_hybrid_state(
    config,
    batch_size: int = 1,
    dtype=None,
) -> HybridLayerState:
    """Initialize GDN conv/recurrent state separately from the KV cache."""
    if dtype is None:
        dtype = getattr(config, "get_dtype", lambda: jnp.float32)()

    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = key_dim * 2 + value_dim
    num_linear_layers = sum(1 for lt in config.layer_types if lt == "linear_attention")

    conv_state = jnp.zeros(
        (batch_size, num_linear_layers, conv_dim, config.linear_conv_kernel_size),
        dtype=dtype,
    )
    recurrent_state = jnp.zeros(
        (
            batch_size,
            num_linear_layers,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        ),
        dtype=jnp.float32,
    )
    return HybridLayerState(conv_state=conv_state, recurrent_state=recurrent_state)


def compute_slot_mapping(
    positions: jnp.ndarray,  # [batch, seq_len] - absolute positions
    block_table: jnp.ndarray,  # [max_seqs, max_blocks_per_seq]
    block_size: int,
    is_prefill: bool = True,
) -> jnp.ndarray:
    """Compute slot mapping for paged attention.
    
    Maps each token position to its physical (block_id, slot) location.
    
    Args:
        positions: Absolute token positions [batch, seq_len]
        block_table: Logical block assignments
        block_size: Tokens per block
        is_prefill: Whether this is prefill (all tokens) or decode (one token)
    
    Returns:
        slot_mapping: Physical indices [batch, seq_len]
    """
    batch, seq_len = positions.shape
    
    # Compute block indices: position // block_size
    block_indices = positions // block_size
    
    # Compute slot indices within each block: position % block_size
    slot_indices = positions % block_size
    
    # Gather physical block IDs from block_table
    # block_indices are per-sequence, so we need to index correctly
    # block_table[seq_id, block_idx]
    seq_ids = jnp.arange(batch)[:, None]  # [batch, 1]
    physical_blocks = block_table[seq_ids, block_indices]  # [batch, seq_len]
    
    # Compute flat slot mapping: physical_block * block_size + slot
    slot_mapping = physical_blocks * block_size + slot_indices
    
    return slot_mapping


def update_kv_cache(
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    slot_mapping: jnp.ndarray,
    new_k: jnp.ndarray,  # [batch, seq_len, num_kv_heads, head_dim]
    new_v: jnp.ndarray,
    layer_idx: int = 0,
    valid_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update KV cache with new key/value pairs.
    
    Uses scatter operations to write new KV pairs to their physical locations.
    
    Args:
        k_cache: Current key cache [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Current value cache
        slot_mapping: Physical indices [batch, seq_len]
        new_k: New keys to write
        new_v: New values to write
        layer_idx: Layer index for per-layer cache
    
    Returns:
        Updated k_cache, v_cache
    """
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")

    k_cache_layer, cache_is_flat = _resolve_kv_layer(k_cache, layer_idx)
    v_cache_layer, _ = _resolve_kv_layer(v_cache, layer_idx)

    num_kv_heads = k_cache_layer.shape[-2]
    head_dim = k_cache_layer.shape[-1]

    k_cache_flat = (
        k_cache_layer
        if cache_is_flat
        else k_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )
    v_cache_flat = (
        v_cache_layer
        if cache_is_flat
        else v_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )
    
    # Flatten slot_mapping and new_k/v
    slot_flat = slot_mapping.reshape(-1)
    k_flat = new_k.reshape(-1, num_kv_heads, head_dim)
    v_flat = new_v.reshape(-1, num_kv_heads, head_dim)
    if valid_mask is not None:
        # Preserve invalid padded entries by redirecting them to a sentinel
        # slot that is removed before returning to cache shape.
        valid_flat = valid_mask.reshape(-1)
        sentinel_idx = k_cache_flat.shape[0]
        safe_slots = jnp.where(valid_flat, slot_flat, sentinel_idx)
        pad_row = jnp.zeros((1, num_kv_heads, head_dim), dtype=k_cache_flat.dtype)
        k_cache_flat_ext = jnp.concatenate([k_cache_flat, pad_row], axis=0)
        v_cache_flat_ext = jnp.concatenate([v_cache_flat, pad_row], axis=0)
        k_cache_flat_ext = k_cache_flat_ext.at[safe_slots].set(k_flat)
        v_cache_flat_ext = v_cache_flat_ext.at[safe_slots].set(v_flat)
        k_cache_flat = k_cache_flat_ext[:-1]
        v_cache_flat = v_cache_flat_ext[:-1]
    else:
        # Scatter update
        k_cache_flat = k_cache_flat.at[slot_flat].set(k_flat)
        v_cache_flat = v_cache_flat.at[slot_flat].set(v_flat)
    
    if not cache_is_flat:
        _, num_blocks, block_size, _, _ = k_cache.shape
        k_cache_layer = k_cache_flat.reshape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
        v_cache_layer = v_cache_flat.reshape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
    else:
        k_cache_layer = k_cache_flat
        v_cache_layer = v_cache_flat
    
    # Update the layer cache in the full cache
    k_cache = k_cache.at[layer_idx].set(k_cache_layer)
    v_cache = v_cache.at[layer_idx].set(v_cache_layer)
    
    return k_cache, v_cache


def paged_attention(
    query: jnp.ndarray,  # [batch, seq_len, num_heads, head_dim]
    k_cache: jnp.ndarray,  # [num_layers, num_blocks, block_size, num_kv_heads, head_dim] or [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jnp.ndarray,
    slot_mapping: jnp.ndarray,  # [batch, seq_len]
    kv_lens: jnp.ndarray,  # [batch]
    scale: float,
    num_key_value_groups: int,
    layer_idx: int = 0,
) -> jnp.ndarray:
    """Paged attention computation.
    
    This is a pedagogical implementation that shows how paged attention works.
    For production, use Pallas/Triton kernels.
    
    Args:
        query: Query tensor
        k_cache: Key cache (paged, with optional layer dimension)
        v_cache: Value cache
        slot_mapping: Maps positions to physical cache locations
        kv_lens: Actual sequence lengths
        scale: Attention scale (1/sqrt(head_dim))
        num_key_value_groups: GQA groups
        layer_idx: Layer index for per-layer cache
    
    Returns:
        Attention output [batch, seq_len, num_heads * head_dim]
    """
    batch, seq_len, num_heads, head_dim = query.shape
    
    k_cache_layer, cache_is_flat = _resolve_kv_layer(k_cache, layer_idx)
    v_cache_layer, _ = _resolve_kv_layer(v_cache, layer_idx)

    num_kv_heads = k_cache_layer.shape[-2]
    head_dim = k_cache_layer.shape[-1]

    k_flat = (
        k_cache_layer
        if cache_is_flat
        else k_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )
    v_flat = (
        v_cache_layer
        if cache_is_flat
        else v_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )

    # Gather KV pairs for all positions using slot_mapping
    # slot_mapping maps each (batch, seq_pos) to a flat index in the KV cache.
    k_gathered = k_flat[slot_mapping]
    v_gathered = v_flat[slot_mapping]
    
    # Expand KV for grouped GQA without materializing repeated values.
    query_grouped = query.reshape(batch, seq_len, num_kv_heads, num_key_value_groups, head_dim)
    k_gathered_grouped = k_gathered[:, :, :, None, :]
    v_gathered_grouped = v_gathered[:, :, :, None, :]

    # Shape: [B, T, KV_heads, G, T]
    attn_scores = jnp.einsum(
        "btkgd,bskgd->btkgs",
        query_grouped,
        k_gathered_grouped,
    ) * scale
    
    # Apply causal mask
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    attn_scores = jnp.where(mask[None, :, None, None, :], -1e10, attn_scores)
    
    # Softmax and output
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    # attn_weights: [B, N, T, T], v_gathered: [B, T, N, H]
    # Output: [B, N, T, H]
    out = jnp.einsum("btkgs,bskgd->btkgd", attn_weights, v_gathered_grouped)
    # Reshape to [B, T, N*H]
    out = out.reshape(batch, seq_len, num_heads, head_dim).reshape(
        batch,
        seq_len,
        num_heads * head_dim,
    )
    
    return out


def paged_attention_prefill(
    query: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    block_table: jnp.ndarray,
    kv_lens: jnp.ndarray,
    positions: jnp.ndarray,
    block_size: int,
    scale: float,
    num_key_value_groups: int,
    layer_idx: int = 0,
) -> jnp.ndarray:
    """Reference paged prefill attention that supports cached prefixes.

    Query positions may be an uncached suffix of a larger logical sequence.
    """

    batch, query_len, num_heads, head_dim = query.shape

    k_cache_layer, cache_is_flat = _resolve_kv_layer(k_cache, layer_idx)
    v_cache_layer, _ = _resolve_kv_layer(v_cache, layer_idx)
    num_kv_heads = k_cache_layer.shape[-2]
    head_dim = k_cache_layer.shape[-1]

    k_flat = (
        k_cache_layer
        if cache_is_flat
        else k_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )
    v_flat = (
        v_cache_layer
        if cache_is_flat
        else v_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )

    max_kv_len = block_table.shape[1] * block_size
    key_positions = jnp.arange(max_kv_len, dtype=jnp.int32)[None, :]
    key_positions = jnp.broadcast_to(key_positions, (batch, max_kv_len))

    block_indices = key_positions // block_size
    slot_indices = key_positions % block_size
    batch_indices = jnp.arange(batch, dtype=jnp.int32)[:, None]
    flat_indices = batch_indices * block_table.shape[1] + block_indices
    physical_blocks = block_table.reshape(-1)[flat_indices]
    slot_mapping = physical_blocks * block_size + slot_indices

    k_gathered = k_flat[slot_mapping.reshape(-1)].reshape(
        batch, max_kv_len, num_kv_heads, head_dim
    )
    v_gathered = v_flat[slot_mapping.reshape(-1)].reshape(
        batch, max_kv_len, num_kv_heads, head_dim
    )

    query_grouped = query.reshape(batch, query_len, num_kv_heads, num_key_value_groups, head_dim)
    k_gathered_grouped = k_gathered[:, :, :, None, :]
    v_gathered_grouped = v_gathered[:, :, :, None, :]

    attn_scores = jnp.einsum(
        "btkgd,bskgd->btkgs",
        query_grouped,
        k_gathered_grouped,
    ) * scale
    valid_keys = key_positions < kv_lens[:, None]
    causal_keys = key_positions[:, None, :] <= positions[:, :, None]
    attn_mask = valid_keys[:, None, :] & causal_keys
    attn_scores = jnp.where(attn_mask[:, :, None, None, :], attn_scores, -1e10)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum("btkgs,bskgd->btkgd", attn_weights, v_gathered_grouped)
    return out.reshape(batch, query_len, num_heads * head_dim)


def paged_attention_decode(
    query: jnp.ndarray,  # [batch, query_len, num_heads, head_dim]
    k_cache: jnp.ndarray,  # [num_layers, num_blocks, block_size, num_kv_heads, head_dim] or [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jnp.ndarray,
    block_table: jnp.ndarray,  # [batch, max_blocks_per_seq]
    kv_lens: jnp.ndarray,  # [batch]
    block_size: int,
    scale: float,
    num_key_value_groups: int,
    max_kv_len: int = None,  # Optional: static max length for JIT compatibility
    layer_idx: int = 0,  # Layer index for per-layer cache
    positions: Optional[jnp.ndarray] = None,  # [batch, query_len]
) -> jnp.ndarray:
    """Paged attention for decode steps with a bounded paged window.
    
    This path is optimized for fixed-width decode buckets. It supports
    query_len > 1 for speculative verification by applying an explicit causal
    mask from the runtime query positions while keeping the materialized KV
    window static from the block-table width.
    
    Args:
        query: Query tensor [batch, query_len, num_heads, head_dim]
        k_cache: Key cache (with optional layer dimension)
        v_cache: Value cache
        block_table: Block assignments per sequence
        kv_lens: Current sequence lengths
        block_size: Tokens per block
        scale: Attention scale
        num_key_value_groups: GQA groups
        max_kv_len: Static maximum KV length (if None, derived from block_table width and block_size)
        layer_idx: Layer index for per-layer cache
        positions: Runtime query positions used for multi-token causal masking
    
    Returns:
        Attention output [batch, query_len, hidden_dim]
    """
    batch, query_len, num_heads, head_dim = query.shape

    if query_len < 1:
        raise ValueError("decode attention requires query_len >= 1")
    if block_table.shape[0] != batch:
        raise ValueError(
            f"query batch {batch} must match block_tables batch {block_table.shape[0]}"
        )
    if block_table.ndim != 2:
        raise ValueError("block_tables must be a 2D tensor [batch, max_blocks_per_seq]")
    if kv_lens.shape[0] != batch:
        raise ValueError("seq_lens shape must align with query batch")
    if positions is not None and positions.shape != (batch, query_len):
        raise ValueError("positions must have shape [batch, query_len]")
    if positions is None and query_len != 1:
        raise ValueError("positions are required for multi-token decode attention")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if max_kv_len is None:
        max_kv_len = block_table.shape[1] * block_size
    
    # Check if cache has layer dimension
    k_cache_layer, cache_is_flat = _resolve_kv_layer(k_cache, layer_idx)
    v_cache_layer, _ = _resolve_kv_layer(v_cache, layer_idx)
    num_kv_heads = k_cache_layer.shape[-2]
    head_dim = k_cache_layer.shape[-1]

    k_flat = (
        k_cache_layer
        if cache_is_flat
        else k_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )
    v_flat = (
        v_cache_layer
        if cache_is_flat
        else v_cache_layer.reshape(-1, num_kv_heads, head_dim)
    )
    num_slots = k_flat.shape[0]
    
    if max_kv_len <= 0:
        raise ValueError("max_kv_len must be positive")
    if max_kv_len > block_table.shape[1] * block_size:
        raise ValueError("max_kv_len exceeds the physical cache span covered by block_tables")
    if max_kv_len > num_slots:
        raise ValueError("max_kv_len exceeds the physical cache slot capacity")
    
    # Positions: [batch, max_seq_len] - all positions 0 to max_seq_len-1
    positions_2d = jnp.arange(max_kv_len)[None, :]  # [1, max_kv_len]
    positions_2d = jnp.broadcast_to(positions_2d, (batch, max_kv_len))
    
    # Compute block and slot indices
    block_indices = positions_2d // block_size  # [batch, max_kv_len]
    slot_indices = positions_2d % block_size  # [batch, max_kv_len]
    
    # Gather physical block IDs for each sequence
    # block_table: [batch, max_blocks_per_seq]
    # For each position, we need block_table[batch_idx, block_idx]
    # Work around Metal bug: use gather instead of take_along_axis
    batch_indices = jnp.arange(batch)[:, None]  # [batch, 1]
    batch_indices = jnp.broadcast_to(batch_indices, (batch, max_kv_len))  # [batch, max_kv_len]
    
    # Manual gather: block_table[batch_indices, block_indices]
    # Reshape to 2D for indexing, then reshape back
    block_table_flat = block_table.reshape(-1)  # [batch * max_blocks_per_seq]
    flat_indices = batch_indices * block_table.shape[1] + block_indices  # [batch, max_kv_len]
    physical_blocks = block_table_flat[flat_indices]  # [batch, max_kv_len]
    
    # Compute slot mapping: [batch, max_kv_len]
    slot_mapping = physical_blocks * block_size + slot_indices
    
    # Gather K and V for all positions
    # Need to flatten slot_mapping for indexing, then reshape back
    # slot_mapping_flat: [batch * max_kv_len]
    slot_mapping_flat = slot_mapping.reshape(-1)
    
    # k_flat[slot_mapping_flat] -> [batch * max_kv_len, num_kv_heads, head_dim]
    k_gathered_flat = k_flat[slot_mapping_flat]
    v_gathered_flat = v_flat[slot_mapping_flat]
    
    # Reshape back to [batch, max_kv_len, num_kv_heads, head_dim]
    k_gathered = k_gathered_flat.reshape(batch, max_kv_len, num_kv_heads, head_dim)
    v_gathered = v_gathered_flat.reshape(batch, max_kv_len, num_kv_heads, head_dim)
    
    valid_mask = positions_2d < kv_lens[:, None]
    if positions is not None:
        causal_mask = positions_2d[:, None, :] <= positions[:, :, None]
        attn_mask = valid_mask[:, None, :] & causal_mask
    else:
        attn_mask = valid_mask[:, None, :]

    query_grouped = query.reshape(batch, query_len, num_kv_heads, num_key_value_groups, head_dim)
    k_gathered_grouped = k_gathered[:, :, :, None, :]
    v_gathered_grouped = v_gathered[:, :, :, None, :]

    attn_scores = jnp.einsum(
        "btkgd,bskgd->btkgs",
        query_grouped,
        k_gathered_grouped,
    ) * scale
    attn_scores = jnp.where(
        attn_mask[:, :, None, None, :],
        attn_scores,
        -1e10,
    )
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    out = jnp.einsum("btkgs,bskgd->btkgd", attn_weights, v_gathered_grouped)
    out = out.reshape(batch, query_len, num_heads, head_dim).reshape(batch, query_len, num_heads * head_dim)
    
    return out
