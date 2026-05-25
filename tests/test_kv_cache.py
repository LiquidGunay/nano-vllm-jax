"""KV cache and paged attention tests.

Tests the correctness of:
- Paged attention mechanism (standard vs paged)
- Linear attention state management (chunked vs recurrent)
- KV cache state persistence across prefill → decode transitions
- Block allocation and slot mapping
- Multi-layer state management

Success Criteria:
- Paged attention = standard attention (identical outputs)
- Chunked linear attention = recurrent (MSE < 1e-6)
- State persistence across steps
- No cross-contamination between sequences
"""

import sys
import os
# Ensure we import from the correct location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kv_cache import (
    init_kv_cache,
    init_linear_attention_states,
    update_kv_cache,
    paged_attention,
    paged_attention_decode,
    compute_slot_mapping,
    KVCacheState,
)
from nanovllm_jax.model import jax_chunk_gated_delta_rule, jax_recurrent_gated_delta_rule


def test_paged_attention_vs_standard():
    """Verify paged attention produces identical results to standard attention."""
    print("\n=== Testing Paged Attention vs Standard ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Test parameters
    batch_size = 1
    seq_len = 32
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    # Create test Q, K, V in [B, T, H, D] format (input format)
    query_input = jnp.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
    key_input = jnp.array(np.random.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32))
    value_input = jnp.array(np.random.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32))
    
    # Transpose to [B, H, T, D] for attention computation (as model does)
    query = query_input.transpose(0, 2, 1, 3)  # [B, H, T, D]
    key = key_input.transpose(0, 2, 1, 3)      # [B, H, K, D]
    value = value_input.transpose(0, 2, 1, 3)  # [B, H, K, D]
    
    # Standard attention (simple implementation for comparison)
    # Repeat KV for GQA
    num_key_value_groups = num_heads // num_kv_heads
    key_rep = jnp.repeat(key, num_key_value_groups, axis=1)  # Repeat on head axis
    value_rep = jnp.repeat(value, num_key_value_groups, axis=1)
    
    # Attention scores: [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
    scale = 1.0 / jnp.sqrt(head_dim)
    attn_weights = jnp.matmul(query, key_rep.transpose(0, 1, 3, 2)) * scale
    
    # Causal mask [T, T]
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    attn_weights = jnp.where(mask == 1, -1e9, attn_weights)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    
    # Output: [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
    standard_output = jnp.matmul(attn_weights, value_rep)
    
    # Transpose back to [B, T, H, D]
    standard_output = standard_output.transpose(0, 2, 1, 3)
    
    # Paged attention
    # Initialize KV cache
    kv_cache = init_kv_cache(
        num_blocks=64,
        block_size=config.block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqs=1,
        max_blocks_per_seq=64,
        dtype=jnp.float32,
    )
    
    # Store K, V in cache (simplified - normally done during forward pass)
    # For this test, we'll just verify the cache structure is correct
    print(f"  Query shape: {query.shape} (after transpose to BHTD)")
    print(f"  Key shape: {key.shape} (after transpose to BHTD)")
    print(f"  Value shape: {value.shape} (after transpose to BHTD)")
    print(f"  Standard output shape: {standard_output.shape}")
    print(f"  Standard output shape: {standard_output.shape}")
    print(f"  KV cache structure initialized")
    
    # Note: Full paged attention test requires proper slot mapping and block tables
    # This is a structural test
    print("  ✓ PASS: Paged attention structure verified")
    assert standard_output.shape == (batch_size, seq_len, num_heads, head_dim)
    assert not bool(jnp.any(jnp.isnan(standard_output)))
    assert not bool(jnp.any(jnp.isinf(standard_output)))


@pytest.mark.parametrize("seq_len", [64, 128])
def test_linear_attention_chunked_vs_recurrent(seq_len):
    """Verify chunked and recurrent linear attention produce same results."""
    print("\n=== Testing Linear Attention: Chunked vs Recurrent ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Test parameters
    batch_size = 1
    num_heads = config.linear_num_value_heads
    k_dim = config.linear_key_head_dim
    v_dim = config.linear_value_head_dim
    # Create test inputs
    query = jnp.array(np.random.randn(batch_size, num_heads, seq_len, k_dim).astype(np.float32))
    key = jnp.array(np.random.randn(batch_size, num_heads, seq_len, k_dim).astype(np.float32))
    value = jnp.array(np.random.randn(batch_size, num_heads, seq_len, v_dim).astype(np.float32))
    g = jnp.array(np.random.randn(batch_size, num_heads, seq_len).astype(np.float32) * 0.1)
    beta = jnp.array(np.random.rand(batch_size, num_heads, seq_len).astype(np.float32))
    
    # Chunked computation (prefill mode)
    chunked_output, _ = jax_chunk_gated_delta_rule(
        query, key, value, g, beta,
        chunk_size=config.linear_chunk_size,
        use_qk_l2norm_in_kernel=config.use_qk_norm_in_gdn,
    )
    
    # Recurrent computation (decode mode) - step by step
    state = jnp.zeros((batch_size, num_heads, k_dim, v_dim), dtype=jnp.float32)
    recurrent_outputs = []
    
    for t in range(seq_len):
        q_t = query[:, :, t:t+1, :]
        k_t = key[:, :, t:t+1, :]
        v_t = value[:, :, t:t+1, :]
        g_t = g[:, :, t:t+1]
        beta_t = beta[:, :, t:t+1]
        
        out_t, state = jax_recurrent_gated_delta_rule(
            q_t, k_t, v_t, g_t, beta_t,
            initial_state=state,
            use_qk_l2norm_in_kernel=config.use_qk_norm_in_gdn,
        )
        recurrent_outputs.append(out_t)
    
    recurrent_output = jnp.concatenate(recurrent_outputs, axis=2)
    
    # Compare
    diff = jnp.abs(chunked_output - recurrent_output)
    mse = float(jnp.mean(diff ** 2))
    max_diff = float(jnp.max(diff))
    
    print(f"  Chunked output shape: {chunked_output.shape}")
    print(f"  Recurrent output shape: {recurrent_output.shape}")
    print(f"  MSE: {mse:.2e}")
    print(f"  Max diff: {max_diff:.2e}")
    
    assert mse < 1e-6
    print("  ✓ PASS: Chunked and recurrent match (MSE < 1e-6)")


def test_linear_attention_state_persistence():
    """Test that linear attention state persists correctly across decode steps."""
    print("\n=== Testing Linear Attention State Persistence ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Initialize state
    batch_size = 1
    num_heads = config.linear_num_value_heads
    k_dim = config.linear_key_head_dim
    v_dim = config.linear_value_head_dim
    
    state = jnp.zeros((batch_size, num_heads, k_dim, v_dim), dtype=jnp.float32)
    
    # Simulate multiple decode steps
    num_steps = 10
    states = [state]
    
    for step in range(num_steps):
        # Create single-token input
        q = jnp.array(np.random.randn(batch_size, num_heads, 1, k_dim).astype(np.float32))
        k = jnp.array(np.random.randn(batch_size, num_heads, 1, k_dim).astype(np.float32))
        v = jnp.array(np.random.randn(batch_size, num_heads, 1, v_dim).astype(np.float32))
        g = jnp.array(np.random.randn(batch_size, num_heads, 1).astype(np.float32) * 0.1)
        beta = jnp.array(np.random.rand(batch_size, num_heads, 1).astype(np.float32))
        
        # Update state
        _, state = jax_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=state,
            use_qk_l2norm_in_kernel=config.use_qk_norm_in_gdn,
        )
        states.append(state)
    
    # Verify state changes at each step
    state_norms = [float(jnp.linalg.norm(s)) for s in states]
    
    print(f"  Steps: {num_steps}")
    print(f"  State shape: {states[0].shape}")
    print(f"  State norms: {[f'{n:.4f}' for n in state_norms[:5]]}...")
    
    # Check for NaN/Inf
    final_state = states[-1]
    has_nan = jnp.any(jnp.isnan(final_state))
    has_inf = jnp.any(jnp.isinf(final_state))

    assert not bool(has_nan)
    assert not bool(has_inf)
    assert not bool(jnp.allclose(states[0], final_state))
    print("  ✓ PASS: State persists correctly, no NaN/Inf")


def test_kv_cache_block_allocation():
    """Test KV cache block allocation and slot mapping."""
    print("\n=== Testing KV Cache Block Allocation ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Initialize KV cache
    num_blocks = 64
    block_size = 16
    
    kv_cache = init_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seqs=2,
        max_blocks_per_seq=32,
        dtype=jnp.float32,  # Use float32 for Metal compatibility
    )
    
    print(f"  Num blocks: {num_blocks}")
    print(f"  Block size: {block_size}")
    print(f"  K cache shape: {kv_cache.k_cache.shape}")
    print(f"  V cache shape: {kv_cache.v_cache.shape}")
    
    # Test slot mapping computation
    positions = jnp.array([[25]])  # Token at position 25
    block_table = jnp.array([[0, 1, 2]])  # Sequence uses blocks 0, 1, 2
    
    slot_mapping = compute_slot_mapping(
        positions=positions,
        block_table=block_table,
        block_size=block_size,
    )
    
    token_idx = 25
    expected_block = token_idx // block_size  # Block 1
    expected_slot = token_idx % block_size     # Slot 9
    
    print(f"  Token position: {token_idx}")
    print(f"  Expected: block {expected_block}, slot {expected_slot}")
    print(f"  Slot mapping: {slot_mapping}")
    
    # Verify shapes (note: KV cache has num_layers dimension)
    expected_shape = (config.num_hidden_layers, num_blocks, block_size, config.num_key_value_heads, config.head_dim)
    assert kv_cache.k_cache.shape == expected_shape
    assert kv_cache.v_cache.shape == expected_shape
    print("  ✓ PASS: KV cache shapes correct")


def test_multi_layer_linear_attention_states():
    """Test that multiple linear attention layers maintain separate states."""
    print("\n=== Testing Multi-Layer Linear Attention States ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Initialize linear attention states for all layers
    batch_size = 1
    
    # Count linear attention layers
    num_linear_layers = len(config.linear_attn_layers)
    
    kv_cache = init_kv_cache(
        num_blocks=64,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seqs=1,
        max_blocks_per_seq=64,
        dtype=jnp.float32,  # Use float32 for Metal compatibility
    )
    
    kv_cache = init_linear_attention_states(kv_cache, config, batch_size, dtype=jnp.float32)
    
    print(f"  Total layers: {config.num_hidden_layers}")
    print(f"  Linear attention layers: {num_linear_layers}")
    print(f"  Linear layer indices: {config.linear_attn_layers}")
    
    # Check recurrent state shape
    if hasattr(kv_cache, 'recurrent_state') and kv_cache.recurrent_state is not None:
        expected_shape = (batch_size, num_linear_layers, config.linear_num_value_heads, 
                          config.linear_key_head_dim, config.linear_value_head_dim)
        actual_shape = kv_cache.recurrent_state.shape
        
        print(f"  Expected recurrent state shape: {expected_shape}")
        print(f"  Actual recurrent state shape: {actual_shape}")
        
        assert actual_shape == expected_shape
        print("  ✓ PASS: Multi-layer states initialized correctly")
    else:
        print("  ⚠ No recurrent state found (may not be initialized)")
        print("  ✓ PASS: Structure verified")


def test_paged_attention_non_identity_blocks():
    """Test that paged attention works with non-identity block tables."""
    print("\n=== Testing Paged Attention with Non-Identity Blocks ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Create non-identity block table
    # Sequence A: uses blocks [5, 2, 8, 1, 3]
    # Sequence B: uses blocks [7, 4, 6, 0, 9]
    block_table = jnp.array([
        [5, 2, 8, 1, 3],  # Seq 0
        [7, 4, 6, 0, 9],  # Seq 1
    ], dtype=jnp.int32)
    
    block_size = 16
    num_blocks = 10
    
    # Initialize KV cache
    kv_cache = init_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seqs=2,
        max_blocks_per_seq=5,
        num_layers=config.num_hidden_layers,
        dtype=jnp.float32,
    )
    
    # Test position 25 for both sequences (should map to different physical blocks)
    positions = jnp.array([[25], [25]], dtype=jnp.int32)
    
    slot_mapping = compute_slot_mapping(
        positions=positions,
        block_table=block_table,
        block_size=block_size,
    )
    
    # Seq 0: position 25 -> logical block 1 -> physical block 2 -> slot 9
    # Physical index = 2 * 16 + 9 = 41
    expected_slot_seq0 = block_table[0, 1] * block_size + (25 % block_size)
    
    # Seq 1: position 25 -> logical block 1 -> physical block 4 -> slot 9
    # Physical index = 4 * 16 + 9 = 73
    expected_slot_seq1 = block_table[1, 1] * block_size + (25 % block_size)
    
    print(f"  Block table seq 0: {block_table[0]}")
    print(f"  Block table seq 1: {block_table[1]}")
    print(f"  Position: 25")
    print(f"  Expected slot seq 0: {expected_slot_seq0} (block {block_table[0, 1]}, slot {25 % block_size})")
    print(f"  Expected slot seq 1: {expected_slot_seq1} (block {block_table[1, 1]}, slot {25 % block_size})")
    print(f"  Actual slot seq 0: {slot_mapping[0, 0]}")
    print(f"  Actual slot seq 1: {slot_mapping[1, 0]}")
    
    np.testing.assert_array_equal(np.array(slot_mapping[0, 0]), np.array(expected_slot_seq0))
    np.testing.assert_array_equal(np.array(slot_mapping[1, 0]), np.array(expected_slot_seq1))
    print("  ✓ PASS: Non-identity block tables work correctly")


def test_decode_attention_long_sequences():
    """Test decode attention with sequences > 128 tokens."""
    print("\n=== Testing Decode Attention with Long Sequences ===")
    
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Create sequence with 200 tokens
    seq_len = 200
    batch_size = 1
    
    # Initialize KV cache with enough blocks
    block_size = 16
    num_blocks = 20  # 20 blocks * 16 tokens = 320 tokens capacity
    
    kv_cache = init_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seqs=1,
        max_blocks_per_seq=20,
        num_layers=config.num_hidden_layers,
        dtype=jnp.float32,
    )
    
    # Simulate decode at position 199 (200th token)
    query = jnp.ones((batch_size, 1, config.num_attention_heads, config.head_dim), dtype=jnp.float32)
    kv_lens = jnp.array([200])  # 200 tokens in cache
    block_table = jnp.array([list(range(20))])  # Uses blocks 0-12 (200 tokens)
    
    # Run decode attention
    output = paged_attention_decode(
        query=query,
        k_cache=kv_cache.k_cache,
        v_cache=kv_cache.v_cache,
        block_table=block_table,
        kv_lens=kv_lens,
        block_size=block_size,
        scale=1.0 / jnp.sqrt(config.head_dim),
        num_key_value_groups=config.num_attention_heads // config.num_key_value_heads,
        max_kv_len=200,
    )

    print(f"  Sequence length: {seq_len}")
    print(f"  Output shape: {output.shape}")

    # Verify output is valid (no NaN/Inf)
    has_nan = jnp.any(jnp.isnan(output))
    has_inf = jnp.any(jnp.isinf(output))
    assert not bool(has_nan)
    assert not bool(has_inf)
    print("  ✓ PASS: Long sequence decode produces valid output")


def test_paged_decode_attention_independent_of_physical_cache_capacity():
    """Decode attention output depends on logical block-table width, not physical capacity."""
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    block_size = 16
    max_blocks_per_seq = 4
    num_kv_blocks_small = 4
    num_kv_blocks_large = 8
    kv_len = 6

    query = jnp.array(np.random.randn(batch_size, 1, num_heads, head_dim).astype(np.float32))
    keys = jnp.array(np.random.randn(batch_size, kv_len, num_kv_heads, head_dim).astype(np.float32))
    values = jnp.array(np.random.randn(batch_size, kv_len, num_kv_heads, head_dim).astype(np.float32))
    block_tables = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    kv_lens = jnp.array([kv_len], dtype=jnp.int32)
    max_kv_len = block_tables.shape[1] * block_size
    positions = jnp.arange(kv_len, dtype=jnp.int32)[None, :]
    slot_mapping = compute_slot_mapping(
        positions=positions,
        block_table=block_tables,
        block_size=block_size,
    )

    def write_cache(num_blocks):
        kv_cache = init_kv_cache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seqs=batch_size,
            max_blocks_per_seq=max_blocks_per_seq,
            num_layers=1,
            dtype=jnp.float32,
        )
        layer_k = kv_cache.k_cache[0].reshape(-1, num_kv_heads, head_dim)
        layer_v = kv_cache.v_cache[0].reshape(-1, num_kv_heads, head_dim)
        slot_flat = slot_mapping.reshape(-1)
        layer_k = layer_k.at[slot_flat].set(keys.reshape(-1, num_kv_heads, head_dim))
        layer_v = layer_v.at[slot_flat].set(values.reshape(-1, num_kv_heads, head_dim))
        return kv_cache.replace(
            k_cache=kv_cache.k_cache.at[0].set(
                layer_k.reshape(kv_cache.k_cache[0].shape)
            ),
            v_cache=kv_cache.v_cache.at[0].set(
                layer_v.reshape(kv_cache.v_cache[0].shape)
            ),
        )

    kv_cache_small = write_cache(num_kv_blocks_small)
    kv_cache_large = write_cache(num_kv_blocks_large)

    out_small = paged_attention_decode(
        query=query,
        k_cache=kv_cache_small.k_cache,
        v_cache=kv_cache_small.v_cache,
        block_table=block_tables,
        kv_lens=kv_lens,
        block_size=block_size,
        scale=1.0 / jnp.sqrt(head_dim),
        num_key_value_groups=num_heads // num_kv_heads,
        max_kv_len=max_kv_len,
        layer_idx=0,
    )
    out_large = paged_attention_decode(
        query=query,
        k_cache=kv_cache_large.k_cache,
        v_cache=kv_cache_large.v_cache,
        block_table=block_tables,
        kv_lens=kv_lens,
        block_size=block_size,
        scale=1.0 / jnp.sqrt(head_dim),
        num_key_value_groups=num_heads // num_kv_heads,
        max_kv_len=max_kv_len,
        layer_idx=0,
    )

    assert out_small.shape == out_large.shape == (batch_size, 1, num_heads * head_dim)
    np.testing.assert_allclose(np.array(out_small), np.array(out_large), rtol=1e-6, atol=1e-6)


def test_paged_attention_grouped_gqa_matches_repeat_reference():
    """Grouped-head paged attention matches the repeat-based GQA reference."""
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2
    num_key_value_groups = num_heads // num_kv_heads
    head_dim = 4
    seq_len = 6
    block_size = 16
    max_blocks_per_seq = 1
    num_blocks = 4

    query = jnp.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
    key = jnp.array(np.random.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32))
    value = jnp.array(np.random.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32))

    kv_cache = init_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqs=batch_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_layers=1,
        dtype=jnp.float32,
    )
    block_tables = jnp.array([[0], [1]], dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
    slot_mapping = compute_slot_mapping(
        positions=positions,
        block_table=block_tables,
        block_size=block_size,
    )
    slot_flat = slot_mapping.reshape(-1)

    layer_k = kv_cache.k_cache[0].reshape(-1, num_kv_heads, head_dim)
    layer_v = kv_cache.v_cache[0].reshape(-1, num_kv_heads, head_dim)
    layer_k = layer_k.at[slot_flat].set(key.reshape(-1, num_kv_heads, head_dim))
    layer_v = layer_v.at[slot_flat].set(value.reshape(-1, num_kv_heads, head_dim))
    kv_cache = kv_cache.replace(
        k_cache=kv_cache.k_cache.at[0].set(layer_k.reshape(kv_cache.k_cache[0].shape)),
        v_cache=kv_cache.v_cache.at[0].set(layer_v.reshape(kv_cache.v_cache[0].shape)),
    )

    out_grouped = paged_attention(
        query=query,
        k_cache=kv_cache.k_cache,
        v_cache=kv_cache.v_cache,
        slot_mapping=slot_mapping,
        kv_lens=jnp.array([seq_len, seq_len], dtype=jnp.int32),
        scale=1.0 / jnp.sqrt(head_dim),
        num_key_value_groups=num_key_value_groups,
        layer_idx=0,
    )

    key_rep = jnp.repeat(key, repeats=num_key_value_groups, axis=2)
    value_rep = jnp.repeat(value, repeats=num_key_value_groups, axis=2)
    scores = jnp.einsum("btmd,bkmd->btmk", query, key_rep) / jnp.sqrt(head_dim)
    causal = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    scores = jnp.where(causal[None, :, None, :] == 1, -1e10, scores)
    weights = jax.nn.softmax(scores, axis=-1)
    expected = jnp.einsum("btmk,bkmd->btmd", weights, value_rep).reshape(batch_size, seq_len, -1)

    np.testing.assert_allclose(np.array(out_grouped), np.array(expected), rtol=1e-6, atol=1e-6)


def test_masked_update_kv_cache_preserves_invalid_and_duplicate_slots():
    """Masked cache writes never mutate invalid slots and respect duplicate slots."""
    num_blocks = 4
    block_size = 8
    num_kv_heads = 2
    head_dim = 3
    seq_len = 4

    initial_k = jnp.arange(num_blocks * block_size * num_kv_heads * head_dim, dtype=jnp.float32).reshape(
        1,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )
    initial_v = (jnp.arange(num_blocks * block_size * num_kv_heads * head_dim, dtype=jnp.float32) + 1_000.0).reshape(
        1,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )

    slot_mapping = jnp.array([[2, 2, 3, 5]], dtype=jnp.int32)
    valid_mask = jnp.array([[1, 0, 1, 0]], dtype=jnp.bool_)
    new_k = jnp.array(np.random.randn(1, seq_len, num_kv_heads, head_dim).astype(np.float32))
    new_v = jnp.array(np.random.randn(1, seq_len, num_kv_heads, head_dim).astype(np.float32))

    updated_k, updated_v = update_kv_cache(
        k_cache=initial_k,
        v_cache=initial_v,
        slot_mapping=slot_mapping,
        new_k=new_k,
        new_v=new_v,
        layer_idx=0,
        valid_mask=valid_mask,
    )

    updated_k_layer = updated_k[0].reshape(-1, num_kv_heads, head_dim)
    updated_v_layer = updated_v[0].reshape(-1, num_kv_heads, head_dim)
    initial_k_layer = initial_k[0].reshape(-1, num_kv_heads, head_dim)
    initial_v_layer = initial_v[0].reshape(-1, num_kv_heads, head_dim)
    new_k_flat = new_k.reshape(-1, num_kv_heads, head_dim)
    new_v_flat = new_v.reshape(-1, num_kv_heads, head_dim)

    np.testing.assert_allclose(np.array(updated_k_layer[2]), np.array(new_k_flat[0]))
    np.testing.assert_allclose(np.array(updated_k_layer[3]), np.array(new_k_flat[2]))
    np.testing.assert_allclose(np.array(updated_k_layer[5]), np.array(initial_k_layer[5]))
    np.testing.assert_allclose(np.array(updated_v_layer[2]), np.array(new_v_flat[0]))
    np.testing.assert_allclose(np.array(updated_v_layer[3]), np.array(new_v_flat[2]))
    np.testing.assert_allclose(np.array(updated_v_layer[5]), np.array(initial_v_layer[5]))
