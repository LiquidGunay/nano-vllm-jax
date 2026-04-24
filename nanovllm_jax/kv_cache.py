"""KV Cache implementation for Qwen 3.5 JAX with vLLM paging compatibility."""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, replace


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
    - recurrent_state: Linear attention recurrent state [batch, num_heads, k_dim, v_dim]
    
    For vLLM compatibility, we use:
    - Block-based KV cache (not contiguous)
    - Page table for logical->physical block mapping
    
    This dataclass is registered as a JAX pytree node for JIT compatibility.
    """
    k_cache: jnp.ndarray  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jnp.ndarray  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: jnp.ndarray  # [max_seqs, max_blocks_per_seq]
    kv_lens: jnp.ndarray  # [max_seqs]
    slot_mapping: jnp.ndarray  # [batch, seq_len]
    conv_state: Optional[jnp.ndarray] = None  # [batch, conv_dim, kernel_size]
    recurrent_state: Optional[jnp.ndarray] = None  # [batch, num_heads, k_dim, v_dim]


# Register KVCacheState as a JAX pytree node for JIT compatibility
def _kv_cache_state_flatten(state: KVCacheState):
    """Flatten KVCacheState into children and auxiliary data."""
    children = (
        state.k_cache,
        state.v_cache,
        state.block_table,
        state.kv_lens,
        state.slot_mapping,
        state.conv_state if state.conv_state is not None else jnp.zeros((1,), dtype=jnp.bfloat16),
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
    dtype=jnp.bfloat16,
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
    k_cache = jnp.zeros((num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    v_cache = jnp.zeros((num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
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
        dtype: Data type for conv_state (defaults to config.get_dtype() or bfloat16)
        
    Returns:
        Updated KVCacheState with initialized linear states
    """
    if dtype is None:
        dtype = getattr(config, 'get_dtype', lambda: jnp.bfloat16)()
    
    # Compute conv_dim = key_dim * 2 + value_dim
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = key_dim * 2 + value_dim
    
    # Count number of linear attention layers
    num_linear_layers = sum(1 for lt in config.layer_types if lt == "linear_attention")
    
    # Initialize conv state: [batch, num_linear_layers, conv_dim, kernel_size]
    # Each linear layer has its own conv state
    conv_state = jnp.zeros(
        (batch_size, num_linear_layers, conv_dim, config.linear_conv_kernel_size),
        dtype=dtype
    )
    
    # Initialize recurrent state: [batch, num_linear_layers, num_heads, k_dim, v_dim]
    # This stores per-layer state for all linear attention layers
    recurrent_state = jnp.zeros(
        (batch_size, num_linear_layers,
         config.linear_num_key_heads,
         config.linear_key_head_dim, config.linear_value_head_dim),
        dtype=jnp.float32  # HF uses float32 for recurrent state
    )
    
    return replace(
        cache,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
    )


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
    num_layers, num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    
    # Get layer-specific cache
    k_cache_layer = k_cache[layer_idx]  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_layer = v_cache[layer_idx]
    
    # Reshape cache to flat form for easier indexing
    k_cache_flat = k_cache_layer.reshape(-1, num_kv_heads, head_dim)
    v_cache_flat = v_cache_layer.reshape(-1, num_kv_heads, head_dim)
    
    # Flatten slot_mapping and new_k/v
    slot_flat = slot_mapping.reshape(-1)
    k_flat = new_k.reshape(-1, num_kv_heads, head_dim)
    v_flat = new_v.reshape(-1, num_kv_heads, head_dim)
    
    # Scatter update
    k_cache_flat = k_cache_flat.at[slot_flat].set(k_flat)
    v_cache_flat = v_cache_flat.at[slot_flat].set(v_flat)
    
    # Reshape back
    k_cache_layer = k_cache_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_layer = v_cache_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)
    
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
    
    # Check if cache has layer dimension
    if k_cache.ndim == 5:
        # Per-layer cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        k_cache_layer = k_cache[layer_idx]
        v_cache_layer = v_cache[layer_idx]
    else:
        # Legacy cache: [num_blocks, block_size, num_kv_heads, head_dim]
        k_cache_layer = k_cache
        v_cache_layer = v_cache
    
    num_blocks, block_size, num_kv_heads, _ = k_cache_layer.shape
    
    # Gather KV pairs for all positions using slot_mapping
    # slot_mapping maps each (batch, seq_pos) to a flat index in the KV cache
    # k_cache_layer.reshape(-1, num_kv_heads, head_dim) gives us [num_blocks * block_size, K, H]
    k_gathered = k_cache_layer.reshape(-1, num_kv_heads, head_dim)[slot_mapping]
    v_gathered = v_cache_layer.reshape(-1, num_kv_heads, head_dim)[slot_mapping]
    
    # Expand KV for GQA
    if num_key_value_groups > 1:
        # Repeat KV for GQA: [B, T, K, H] -> [B, T, K*n_rep, H] = [B, T, N, H]
        k_gathered = jnp.repeat(k_gathered, num_key_value_groups, axis=2)
        v_gathered = jnp.repeat(v_gathered, num_key_value_groups, axis=2)
    
    # Compute attention scores
    # query: [B, T, N, H], k_gathered: [B, T, N, H]
    # Need: [B, N, T, T] scores
    attn_scores = jnp.einsum("btnh,bsnh->bnts", query, k_gathered) * scale
    
    # Apply causal mask
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    attn_scores = jnp.where(mask == 1, -1e10, attn_scores)
    
    # Softmax and output
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    # attn_weights: [B, N, T, T], v_gathered: [B, T, N, H]
    # Output: [B, N, T, H]
    out = jnp.einsum("bnts,bsnh->btnh", attn_weights, v_gathered)
    # Reshape to [B, T, N*H]
    out = out.transpose(0, 1, 2, 3).reshape(batch, seq_len, num_heads * head_dim)
    
    return out


def paged_attention_decode(
    query: jnp.ndarray,  # [batch, 1, num_heads, head_dim]
    k_cache: jnp.ndarray,  # [num_layers, num_blocks, block_size, num_kv_heads, head_dim] or [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: jnp.ndarray,
    block_table: jnp.ndarray,  # [batch, max_blocks_per_seq]
    kv_lens: jnp.ndarray,  # [batch]
    block_size: int,
    scale: float,
    num_key_value_groups: int,
    max_kv_len: int = None,  # Optional: static max length for JIT compatibility
    layer_idx: int = 0,  # Layer index for per-layer cache
) -> jnp.ndarray:
    """Paged attention for decode step (single token).
    
    Simple and fast version using direct gather with limited window.
    For long sequences, uses a sliding window of most recent tokens.
    
    Args:
        query: Single token query [batch, 1, num_heads, head_dim]
        k_cache: Key cache (with optional layer dimension)
        v_cache: Value cache
        block_table: Block assignments per sequence
        kv_lens: Current sequence lengths
        block_size: Tokens per block
        scale: Attention scale
        num_key_value_groups: GQA groups
        max_kv_len: Static maximum KV length (if None, uses num_blocks * block_size)
        layer_idx: Layer index for per-layer cache
    
    Returns:
        Attention output [batch, 1, hidden_dim]
    """
    batch, _, num_heads, head_dim = query.shape
    
    # Check if cache has layer dimension
    if k_cache.ndim == 5:
        # Per-layer cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        k_cache_layer = k_cache[layer_idx]
        v_cache_layer = v_cache[layer_idx]
    else:
        # Legacy cache: [num_blocks, block_size, num_kv_heads, head_dim]
        k_cache_layer = k_cache
        v_cache_layer = v_cache
    
    num_blocks, block_size_cache, num_kv_heads, _ = k_cache_layer.shape
    
    # Use static max_kv_len for JIT compatibility
    if max_kv_len is None:
        max_kv_len = num_blocks * block_size_cache
    
    # For faster compilation, use a fixed window size
    # This is a trade-off: smaller window = faster compilation but may miss some context
    # For typical decode scenarios, 128-256 tokens is usually sufficient
    WINDOW_SIZE = 128
    
    # Flatten cache: [num_blocks * block_size, num_kv_heads, head_dim]
    k_flat = k_cache_layer.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache_layer.reshape(-1, num_kv_heads, head_dim)
    
    # Create window of positions to attend to
    # For each sequence, we attend to the last WINDOW_SIZE tokens (or all if shorter)
    # Start position: max(0, kv_lens - WINDOW_SIZE)
    # But for JIT, we use fixed-size window
    
    # Positions to attend to [WINDOW_SIZE]
    window_positions = jnp.arange(WINDOW_SIZE)
    
    # For each batch, compute the actual positions (relative to kv_lens)
    # This creates [batch, WINDOW_SIZE] of absolute positions
    # For seq with kv_len=10, we want positions 0-9 (padded with 0)
    # For seq with kv_len=100, we want positions 0-127 (most recent)
    
    # Simple approach: attend to positions 0 to WINDOW_SIZE-1
    # Mask out positions >= kv_lens
    
    # Compute block and slot indices for each position
    block_indices = window_positions // block_size  # [WINDOW_SIZE]
    slot_indices = window_positions % block_size  # [WINDOW_SIZE]
    
    # Gather physical block IDs for each sequence
    # block_table: [batch, max_blocks_per_seq]
    # block_indices: [WINDOW_SIZE]
    physical_blocks = block_table[:, block_indices]  # [batch, WINDOW_SIZE]
    
    # Compute slot mapping: [batch, WINDOW_SIZE]
    slot_mapping = physical_blocks * block_size + slot_indices
    
    # Gather K and V for all positions in window
    # k_flat[slot_mapping] -> [batch, WINDOW_SIZE, num_kv_heads, head_dim]
    k_gathered = k_flat[slot_mapping]
    v_gathered = v_flat[slot_mapping]
    
    # Create validity mask: position is valid if < kv_lens
    # [batch, WINDOW_SIZE]
    valid_mask = window_positions[None, :] < kv_lens[:, None]
    
    # Expand KV for GQA if needed
    if num_key_value_groups > 1:
        k_gathered = jnp.repeat(k_gathered, num_key_value_groups, axis=2)
        v_gathered = jnp.repeat(v_gathered, num_key_value_groups, axis=2)
    
    # k_gathered, v_gathered: [batch, WINDOW_SIZE, num_heads, head_dim]
    # Transpose to [batch, num_heads, WINDOW_SIZE, head_dim]
    k_for_attn = k_gathered.transpose(0, 2, 1, 3)
    v_for_attn = v_gathered.transpose(0, 2, 1, 3)
    
    # query: [batch, 1, num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
    query_t = query.transpose(0, 2, 1, 3)
    
    # Compute attention scores: [batch, num_heads, 1, WINDOW_SIZE]
    attn_scores = jnp.einsum("bh1d,bhkd->bh1k", query_t, k_for_attn) * scale
    
    # Apply validity mask
    # valid_mask: [batch, WINDOW_SIZE] -> [batch, 1, 1, WINDOW_SIZE]
    valid_mask_expanded = valid_mask[:, None, None, :]
    attn_scores = jnp.where(valid_mask_expanded, attn_scores, -1e10)
    
    # Softmax and output
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    
    # out: [batch, num_heads, 1, head_dim]
    out = jnp.einsum("bh1k,bhkd->bh1d", attn_weights, v_for_attn)
    
    # Transpose back: [batch, num_heads, 1, head_dim] -> [batch, 1, hidden_dim]
    out = out.transpose(0, 2, 1, 3).reshape(batch, 1, -1)
    
    return out
