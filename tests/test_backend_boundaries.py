"""Backend/cache boundary tests that do not require model weights."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.backends import PureJAXBackend, select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.model_executor import ModelExecutor
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.sequence import SamplingParams, Sequence, SequenceStatus
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import HybridLayerState, KVCacheSpec, cap_num_kv_cache_blocks
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.model import forward, init_params
from nanovllm_jax.mtp.mtp_layer import init_mtp_params, mtp_forward


def _dense_decode_attention(query, keys, values, seq_lens, scale):
    """Reference decode attention for [B, 1, H, D] query and dense KV."""
    query_t = query.transpose(0, 2, 1, 3)
    keys_t = keys.transpose(0, 2, 1, 3)
    values_t = values.transpose(0, 2, 1, 3)
    scores = jnp.einsum("bh1d,bhkd->bh1k", query_t, keys_t) * scale
    positions = jnp.arange(keys.shape[1])[None, :]
    valid = positions < seq_lens[:, None]
    scores = jnp.where(valid[:, None, None, :], scores, -1e10)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bh1k,bhkd->bh1d", weights, values_t)
    return out.transpose(0, 2, 1, 3).reshape(query.shape[0], 1, -1)


def _tiny_full_attention_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        block_size=2,
        num_kvcache_blocks=4,
        dtype="float32",
        tie_word_embeddings=True,
        layer_types=("full_attention",),
        linear_attn_layers=(),
        max_kv_cache_bytes=4 * 2 * 2 * 1 * 8 * 4 * 2,
    )


def _tiny_linear_attention_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_size=4,
        linear_chunk_size=4,
        block_size=2,
        num_kvcache_blocks=4,
        dtype="float32",
        tie_word_embeddings=True,
        layer_types=("linear_attention",),
        linear_attn_layers=(0,),
        max_kv_cache_bytes=4 * 2 * 2 * 1 * 8 * 4 * 2,
    )


def _scheduled_batch(
    *,
    tokens,
    positions,
    block_tables,
    seq_lens,
    is_prefill,
) -> ScheduledBatch:
    query_len = len(tokens)
    return ScheduledBatch(
        tokens=jnp.array([tokens], dtype=jnp.int32),
        positions=jnp.array([positions], dtype=jnp.int32),
        seq_ids=jnp.array([0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, query_len], dtype=jnp.int32),
        is_prefill=is_prefill,
        num_prefill_tokens=query_len if is_prefill else 0,
        num_decode_tokens=0 if is_prefill else 1,
        block_tables=jnp.array([block_tables], dtype=jnp.int32),
        seq_lens=jnp.array([seq_lens], dtype=jnp.int32),
    )


def _bucketed_scheduled_batch(
    *,
    tokens,
    bucket_len,
    block_tables,
    seq_lens,
    is_prefill,
) -> ScheduledBatch:
    padded_tokens = tokens + [0] * (bucket_len - len(tokens))
    positions = list(range(len(tokens))) + [0] * (bucket_len - len(tokens))
    return ScheduledBatch(
        tokens=jnp.array([padded_tokens], dtype=jnp.int32),
        positions=jnp.array([positions], dtype=jnp.int32),
        seq_ids=jnp.array([0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, len(tokens)], dtype=jnp.int32),
        is_prefill=is_prefill,
        num_prefill_tokens=len(tokens) if is_prefill else 0,
        num_decode_tokens=0 if is_prefill else 1,
        block_tables=jnp.array([block_tables], dtype=jnp.int32),
        seq_lens=jnp.array([seq_lens], dtype=jnp.int32),
    )


def test_kv_cache_block_count_is_capped_by_bytes():
    spec = KVCacheSpec(
        num_layers=2,
        num_blocks=100,
        block_size=4,
        num_kv_heads=1,
        head_dim=8,
        dtype=jnp.float32,
        max_kv_cache_bytes=2 * 2 * 4 * 1 * 8 * 4 * 2,
    )

    assert cap_num_kv_cache_blocks(spec) == 2


def test_pure_jax_metadata_uses_non_identity_block_tables():
    backend = PureJAXBackend()
    block_tables = jnp.array([[5, 2, 8], [7, 4, 6]], dtype=jnp.int32)
    positions = jnp.array([[25], [25]], dtype=jnp.int32)
    seq_lens = jnp.array([26, 26], dtype=jnp.int32)

    metadata = backend.build_attention_metadata(
        positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=16,
        is_prefill=False,
    )

    np.testing.assert_array_equal(np.array(metadata.slot_mapping), np.array([[41], [73]]))
    np.testing.assert_array_equal(np.array(metadata.query_start_loc), np.array([0, 1, 2]))
    assert metadata.num_prefill_tokens == 0
    assert metadata.num_decode_tokens == 2


def test_pure_jax_decode_attention_matches_dense_reference_non_contiguous_blocks():
    backend = PureJAXBackend()
    block_size = 2
    spec = KVCacheSpec(
        num_layers=1,
        num_blocks=6,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=2,
        dtype=jnp.float32,
        max_kv_cache_bytes=4096,
    )
    cache = backend.allocate_kv_cache(spec, max_seqs=2, max_blocks_per_seq=3)
    block_tables = jnp.array([[3, 1, 5], [2, 4, 0]], dtype=jnp.int32)
    seq_lens = jnp.array([4, 3], dtype=jnp.int32)

    dense_k = jnp.array(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]]],
            [[[0.5, 0.0]], [[0.0, 0.5]], [[1.0, -1.0]], [[9.0, 9.0]]],
        ],
        dtype=jnp.float32,
    )
    dense_v = jnp.array(
        [
            [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]],
            [[[2.0, 1.0]], [[4.0, 3.0]], [[6.0, 5.0]], [[8.0, 7.0]]],
        ],
        dtype=jnp.float32,
    )

    k_cache = cache.k_cache
    v_cache = cache.v_cache
    for batch_idx in range(2):
        for pos in range(int(seq_lens[batch_idx])):
            block = int(block_tables[batch_idx, pos // block_size])
            slot = pos % block_size
            k_cache = k_cache.at[0, block, slot].set(dense_k[batch_idx, pos])
            v_cache = v_cache.at[0, block, slot].set(dense_v[batch_idx, pos])
    cache = type(cache)(k_cache, v_cache)

    query = jnp.array([[[[1.0, 0.5]]], [[[0.25, 1.0]]]], dtype=jnp.float32)
    metadata = backend.build_attention_metadata(
        positions=jnp.array([[4], [3]], dtype=jnp.int32),
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        is_prefill=False,
    )

    actual = backend.attention(
        layer_id=0,
        query=query,
        cache=cache,
        metadata=metadata,
        block_size=block_size,
        scale=1.0,
        num_key_value_groups=1,
        is_prefill=False,
    )
    expected = _dense_decode_attention(query, dense_k, dense_v, seq_lens, scale=1.0)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6)


def test_auto_backend_selects_pure_jax_reference_path():
    assert select_backend("auto").name == "pure_jax"


def test_scheduler_builds_scheduled_batch_for_uncached_suffix():
    config = _tiny_full_attention_config()
    scheduler = Scheduler(config)
    seq = Sequence([10, 11, 12, 13], SamplingParams(temperature=0.0), seq_id=7)
    seq.block_table = [5, 6]
    seq.num_cached_tokens = 2

    batch = scheduler.build_scheduled_batch([seq], is_prefill=True)

    np.testing.assert_array_equal(np.array(batch.tokens), np.array([[12, 13]]))
    np.testing.assert_array_equal(np.array(batch.positions), np.array([[2, 3]]))
    np.testing.assert_array_equal(np.array(batch.query_start_loc), np.array([0, 2]))
    np.testing.assert_array_equal(np.array(batch.block_tables), np.array([[5, 6]]))
    np.testing.assert_array_equal(np.array(batch.seq_lens), np.array([4]))
    assert batch.num_prefill_tokens == 2
    assert batch.num_decode_tokens == 0


def test_scheduler_pads_scheduled_batch_to_static_buckets():
    config = _tiny_full_attention_config()
    config.prefill_buckets = (4, 8)
    config.batch_size_buckets = (4,)
    config.max_blocks_per_seq = 4
    scheduler = Scheduler(config)
    seq_a = Sequence([10, 11, 12], SamplingParams(temperature=0.0), seq_id=7)
    seq_b = Sequence([20], SamplingParams(temperature=0.0), seq_id=8)
    seq_a.block_table = [5, 6]
    seq_b.block_table = [9]

    batch = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=True)

    assert batch.tokens.shape == (4, 4)
    assert batch.positions.shape == (4, 4)
    assert batch.block_tables.shape == (4, 4)
    np.testing.assert_array_equal(np.array(batch.query_start_loc), np.array([0, 3, 4, 4, 4]))
    np.testing.assert_array_equal(np.array(batch.seq_ids), np.array([7, 8, -1, -1]))
    assert batch.num_prefill_tokens == 4


def test_scheduler_rejects_requests_exceeding_static_capacity():
    config = _tiny_full_attention_config()
    config.max_blocks_per_seq = 2
    config.max_num_batched_tokens = 8
    scheduler = Scheduler(config)

    too_many_blocks = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0.0, max_tokens=1), seq_id=1)
    with pytest.raises(ValueError, match="prompt needs 3 blocks"):
        scheduler.add(too_many_blocks)

    too_many_total_tokens = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=2), seq_id=2)
    with pytest.raises(ValueError, match="per-sequence capacity"):
        scheduler.add(too_many_total_tokens)


def test_scheduler_rejects_single_prompt_larger_than_prefill_budget():
    config = _tiny_full_attention_config()
    config.max_blocks_per_seq = 8
    config.max_num_batched_tokens = 3
    scheduler = Scheduler(config)
    seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=0.0, max_tokens=1), seq_id=3)

    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        scheduler.add(seq)


def test_llm_engine_generate_validates_inputs_before_mutating_scheduler():
    engine = LLMEngine.__new__(LLMEngine)

    def tokenize(text):
        return [1] if text else []

    engine._tokenize = tokenize

    with pytest.raises(ValueError, match="sampling_params length"):
        LLMEngine.generate(
            engine,
            [[1], [2]],
            sampling_params=[SamplingParams(temperature=0.0, max_tokens=1)],
            use_tqdm=False,
        )

    with pytest.raises(ValueError, match="at least one token"):
        LLMEngine.generate(
            engine,
            [[]],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        LLMEngine.generate(
            engine,
            [[1]],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=0),
            use_tqdm=False,
        )


def test_executor_cached_prefill_matches_no_cache_prefill_logits():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(0), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)

    full_tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dense_logits, _ = forward(
        full_tokens,
        params,
        config,
        kv_cache_state=None,
        is_prefill=True,
        backend=executor.backend,
    )

    prefix_batch = _scheduled_batch(
        tokens=[1, 2],
        positions=[0, 1],
        block_tables=[0, 1],
        seq_lens=2,
        is_prefill=True,
    )
    prefix_out = executor.forward_step(prefix_batch, cache_storage=cache)

    suffix_batch = _scheduled_batch(
        tokens=[3, 4],
        positions=[2, 3],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=True,
    )
    suffix_out = executor.forward_step(suffix_batch, cache_storage=prefix_out.cache_storage)

    np.testing.assert_allclose(
        np.array(suffix_out.activations),
        np.array(dense_logits[:, 2:]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_executor_cached_decode_matches_recompute_logits():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(1), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)

    full_tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dense_logits, _ = forward(
        full_tokens,
        params,
        config,
        kv_cache_state=None,
        is_prefill=True,
        backend=executor.backend,
    )

    prompt_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1],
        seq_lens=3,
        is_prefill=True,
    )
    prompt_out = executor.forward_step(prompt_batch, cache_storage=cache)
    decode_batch = _scheduled_batch(
        tokens=[4],
        positions=[3],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=False,
    )
    decode_out = executor.forward_step(decode_batch, cache_storage=prompt_out.cache_storage)

    np.testing.assert_allclose(
        np.array(decode_out.activations[:, 0]),
        np.array(dense_logits[:, -1]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_bucketed_prefill_last_logits_match_exact_prefill():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(3), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3)

    exact_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )
    bucketed_batch = _bucketed_scheduled_batch(
        tokens=[1, 2, 3],
        bucket_len=5,
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )

    exact = executor.forward_step(exact_batch, cache_storage=cache, last_logits_only=True)
    bucketed = executor.forward_step(bucketed_batch, cache_storage=cache, last_logits_only=True)

    np.testing.assert_allclose(
        np.array(bucketed.activations),
        np.array(exact.activations),
        rtol=1e-5,
        atol=1e-5,
    )


def test_bucketed_prefill_jit_reuses_shape_for_different_prompt_lengths():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(5), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache_a = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3)
    cache_b = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3)
    empty_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 0, 1, 1)),
        recurrent_state=jnp.zeros((1, 0, 1, 1, 1)),
    )

    batch_a = _bucketed_scheduled_batch(
        tokens=[1, 2, 3],
        bucket_len=5,
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )
    batch_b = _bucketed_scheduled_batch(
        tokens=[1, 2],
        bucket_len=5,
        block_tables=[0, 1, 2],
        seq_lens=2,
        is_prefill=True,
    )

    out_a = executor.forward_step_jit(
        batch_a,
        cache_storage=cache_a,
        hybrid_state=empty_hybrid,
        last_logits_only=True,
    )
    compiled_entries = len(executor._jit_cache)
    out_b = executor.forward_step_jit(
        batch_b,
        cache_storage=cache_b,
        hybrid_state=empty_hybrid,
        last_logits_only=True,
    )

    assert len(executor._jit_cache) == compiled_entries
    assert out_a.activations.shape == out_b.activations.shape == (1, 1, config.vocab_size)


def test_bucketed_linear_prefill_preserves_hybrid_state_for_decode():
    config = _tiny_linear_attention_config()
    params = init_params(jax.random.PRNGKey(4), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    exact_cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3)
    bucketed_cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3)
    exact_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 1, 12, config.linear_conv_kernel_size), dtype=config.get_dtype()),
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_key_head_dim, config.linear_value_head_dim), dtype=jnp.float32),
    )
    bucketed_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 1, 12, config.linear_conv_kernel_size), dtype=config.get_dtype()),
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_key_head_dim, config.linear_value_head_dim), dtype=jnp.float32),
    )

    exact_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )
    bucketed_batch = _bucketed_scheduled_batch(
        tokens=[1, 2, 3],
        bucket_len=5,
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )

    exact = executor.forward_step(
        exact_batch,
        cache_storage=exact_cache,
        hybrid_state=exact_hybrid,
        last_logits_only=True,
    )
    bucketed = executor.forward_step(
        bucketed_batch,
        cache_storage=bucketed_cache,
        hybrid_state=bucketed_hybrid,
        last_logits_only=True,
    )

    np.testing.assert_allclose(np.array(bucketed.activations), np.array(exact.activations), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.array(bucketed.hybrid_state.conv_state),
        np.array(exact.hybrid_state.conv_state),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(bucketed.hybrid_state.recurrent_state),
        np.array(exact.hybrid_state.recurrent_state),
        rtol=1e-5,
        atol=1e-5,
    )

    decode_batch = _scheduled_batch(
        tokens=[4],
        positions=[3],
        block_tables=[0, 1, 2],
        seq_lens=4,
        is_prefill=False,
    )
    exact_decode = executor.forward_step(
        decode_batch,
        cache_storage=exact.cache_storage,
        hybrid_state=exact.hybrid_state,
        last_logits_only=True,
    )
    bucketed_decode = executor.forward_step(
        decode_batch,
        cache_storage=bucketed.cache_storage,
        hybrid_state=bucketed.hybrid_state,
        last_logits_only=True,
    )

    np.testing.assert_allclose(
        np.array(bucketed_decode.activations),
        np.array(exact_decode.activations),
        rtol=1e-5,
        atol=1e-5,
    )


def test_linear_suffix_prefill_matches_sequential_decode_state():
    config = _tiny_linear_attention_config()
    params = init_params(jax.random.PRNGKey(5), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3)
    hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 1, 12, config.linear_conv_kernel_size), dtype=config.get_dtype()),
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_key_head_dim, config.linear_value_head_dim), dtype=jnp.float32),
    )

    prefix_batch = _scheduled_batch(
        tokens=[1, 2],
        positions=[0, 1],
        block_tables=[0, 1, 2],
        seq_lens=2,
        is_prefill=True,
    )
    prefix = executor.forward_step(
        prefix_batch,
        cache_storage=cache,
        hybrid_state=hybrid,
        return_hidden=True,
    )

    suffix_batch = _scheduled_batch(
        tokens=[3, 4],
        positions=[2, 3],
        block_tables=[0, 1, 2],
        seq_lens=4,
        is_prefill=True,
    )
    suffix = executor.forward_step(
        suffix_batch,
        cache_storage=prefix.cache_storage,
        hybrid_state=prefix.hybrid_state,
        return_hidden=True,
    )

    decode_one_batch = _scheduled_batch(
        tokens=[3],
        positions=[2],
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=False,
    )
    decode_one = executor.forward_step(
        decode_one_batch,
        cache_storage=prefix.cache_storage,
        hybrid_state=prefix.hybrid_state,
        return_hidden=True,
    )
    decode_two_batch = _scheduled_batch(
        tokens=[4],
        positions=[3],
        block_tables=[0, 1, 2],
        seq_lens=4,
        is_prefill=False,
    )
    decode_two = executor.forward_step(
        decode_two_batch,
        cache_storage=decode_one.cache_storage,
        hybrid_state=decode_one.hybrid_state,
        return_hidden=True,
    )

    sequential_hidden = jnp.concatenate([decode_one.activations, decode_two.activations], axis=1)
    np.testing.assert_allclose(
        np.array(suffix.activations),
        np.array(sequential_hidden),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.array(suffix.hybrid_state.conv_state),
        np.array(decode_two.hybrid_state.conv_state),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(suffix.hybrid_state.recurrent_state),
        np.array(decode_two.hybrid_state.recurrent_state),
        rtol=1e-4,
        atol=1e-4,
    )


def test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(6), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache = executor.backend.allocate_kv_cache(spec, max_seqs=3, max_blocks_per_seq=2)

    seq_a_dense, _ = forward(
        jnp.array([[1, 2, 3, 6]], dtype=jnp.int32),
        params,
        config,
        kv_cache_state=None,
        is_prefill=True,
        backend=executor.backend,
    )
    seq_b_dense, _ = forward(
        jnp.array([[4, 5, 7]], dtype=jnp.int32),
        params,
        config,
        kv_cache_state=None,
        is_prefill=True,
        backend=executor.backend,
    )

    prefill_batch = ScheduledBatch(
        tokens=jnp.array([[1, 2, 3], [4, 5, 0]], dtype=jnp.int32),
        positions=jnp.array([[0, 1, 2], [0, 1, 0]], dtype=jnp.int32),
        seq_ids=jnp.array([0, 1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32),
        is_prefill=True,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        block_tables=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
        seq_lens=jnp.array([3, 2], dtype=jnp.int32),
    )
    prefill = executor.forward_step(prefill_batch, cache_storage=cache)

    decode_batch = ScheduledBatch(
        tokens=jnp.array([[6], [7], [0]], dtype=jnp.int32),
        positions=jnp.array([[3], [2], [0]], dtype=jnp.int32),
        seq_ids=jnp.array([0, 1, -1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2, 2], dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=2,
        block_tables=jnp.array([[0, 1], [2, 3], [0, 0]], dtype=jnp.int32),
        seq_lens=jnp.array([4, 3, 0], dtype=jnp.int32),
    )
    decode = executor.forward_step(
        decode_batch,
        cache_storage=prefill.cache_storage,
        last_logits_only=True,
    )

    np.testing.assert_allclose(
        np.array(decode.activations[0, 0]),
        np.array(seq_a_dense[0, -1]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(decode.activations[1, 0]),
        np.array(seq_b_dense[0, -1]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_model_runner_uses_bucketed_batched_jit_path():
    config = _tiny_full_attention_config()
    config.max_num_seqs = 2
    config.prefill_buckets = (4,)
    config.batch_size_buckets = (2,)
    config.max_blocks_per_seq = 2
    config.jax_execution = "jit"
    params = init_params(jax.random.PRNGKey(7), config)
    scheduler = Scheduler(config)
    runner = ModelRunner(config, params, backend="pure_jax")

    seq_a = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=1), seq_id=10)
    seq_b = Sequence([4, 5], SamplingParams(temperature=0.0, max_tokens=1), seq_id=11)
    scheduler.add(seq_a)
    scheduler.add(seq_b)
    seqs, batch = scheduler.schedule()

    token_ids = runner.run(seqs, batch=batch)

    assert batch.tokens.shape == (2, 4)
    assert len(token_ids) == 2
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert len(runner.executor._jit_cache) == 1


def test_model_runner_warmup_uses_requested_prefill_len_without_buckets():
    config = _tiny_full_attention_config()
    config.jax_execution = "jit"
    config.prefill_buckets = ()
    config.batch_size_buckets = ()
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.block_size = config.block_size
    runner.max_blocks_per_seq = 2
    runner.execution = "jit"
    runner.cache_storage = object()
    runner._warmup_compiled = False

    class Ready:
        def block_until_ready(self):
            return self

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def forward_step_jit(self, batch, **kwargs):
            self.calls.append((tuple(batch.tokens.shape), batch.is_prefill))
            return type("Output", (), {"activations": Ready()})()

    runner.executor = FakeExecutor()
    runner._sample_fn = lambda logits, temperatures: Ready()

    runner.warmup_compilation(max_prefill_len=3, max_batch=1)

    assert runner.executor.calls == [((1, 3), True), ((1, 1), False)]


def test_model_runner_direct_batch_builder_uses_static_buckets():
    config = _tiny_full_attention_config()
    config.prefill_buckets = (4,)
    config.batch_size_buckets = (2,)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.max_blocks_per_seq = 2

    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=1), seq_id=21)
    seq.block_table = [0, 1]

    batch = runner._build_scheduled_batch([seq], is_prefill=True)

    assert batch.tokens.shape == (2, 4)
    assert batch.block_tables.shape == (2, 2)
    np.testing.assert_array_equal(np.array(batch.seq_ids), np.array([21, -1]))
    np.testing.assert_array_equal(np.array(batch.query_start_loc), np.array([0, 3, 3]))


def test_scheduler_postprocess_accepts_mtp1_multi_token_output():
    config = _tiny_full_attention_config()
    scheduler = Scheduler(config)
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4), seq_id=31)
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [0, 1]
    scheduler.running.append(seq)

    finished = scheduler.postprocess([seq], [[4, 5]])

    assert finished == [False]
    assert scheduler.last_num_generated_tokens == 2
    assert seq.completion_token_ids == [4, 5]
    assert seq in scheduler.running


def test_model_runner_mtp1_accepts_draft_and_returns_bonus_token():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = _tiny_full_attention_config()
    runner.block_size = runner.config.block_size
    runner.mtp_enabled = True
    runner.mtp1_enabled = True
    runner.mtp_hidden_source = "pre_norm"
    runner._mtp1_drafts = {41: 4}
    runner.cache_storage = "base-cache"
    runner._sample_fn = lambda logits, temperatures: jnp.argmax(logits, axis=-1)
    runner._logits_from_hidden = lambda hidden: hidden
    runner._batch_hybrid_state = lambda batch: "hybrid"
    runner._store_batch_hybrid_state = lambda batch, state: None
    runner._refresh_kv_snapshot = lambda batch, state: None
    seeded = []
    runner._seed_mtp1_draft = lambda seq, hidden, confirmed_token_id, position: seeded.append(
        (seq.seq_id, int(confirmed_token_id), int(position))
    )

    class FakeExecutor:
        def forward_step(self, batch, **kwargs):
            np.testing.assert_array_equal(np.array(batch.tokens), np.array([[3, 4]]))
            np.testing.assert_array_equal(np.array(batch.positions), np.array([[2, 3]]))
            np.testing.assert_array_equal(np.array(batch.seq_lens), np.array([4]))
            assert batch.is_prefill is True
            logits = jnp.zeros((1, 2, 8), dtype=jnp.float32)
            logits = logits.at[0, 0, 4].set(10.0)
            logits = logits.at[0, 1, 5].set(10.0)
            return type("Output", (), {
                "activations": logits,
                "cache_storage": "accepted-cache",
                "hybrid_state": "accepted-hybrid",
            })()

    runner.executor = FakeExecutor()
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=5), seq_id=41)
    seq.block_table = [0, 1]
    batch = _scheduled_batch(tokens=[3], positions=[2], block_tables=[0, 1], seq_lens=3, is_prefill=False)
    batch.seq_ids = jnp.array([41], dtype=jnp.int32)

    token_ids = runner._run_mtp1([seq], batch)

    assert token_ids == [[4, 5]]
    assert runner.cache_storage == "accepted-cache"
    assert seeded == [(41, 5, 4)]


def test_model_runner_mtp1_rejection_falls_back_without_committing_verifier_cache():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = _tiny_full_attention_config()
    runner.block_size = runner.config.block_size
    runner.mtp_enabled = True
    runner.mtp1_enabled = True
    runner.mtp_hidden_source = "pre_norm"
    runner._mtp1_drafts = {42: 4}
    runner.cache_storage = "base-cache"
    runner._logits_from_hidden = lambda hidden: hidden
    runner._batch_hybrid_state = lambda batch: "hybrid"
    runner._run_main_and_sample = lambda seqs, batch, seed_mtp1: [6]

    class FakeExecutor:
        def forward_step(self, batch, **kwargs):
            np.testing.assert_array_equal(np.array(batch.tokens), np.array([[3, 4]]))
            assert batch.is_prefill is True
            logits = jnp.zeros((1, 2, 8), dtype=jnp.float32)
            logits = logits.at[0, 0, 7].set(10.0)
            return type("Output", (), {
                "activations": logits,
                "cache_storage": "rejected-cache",
                "hybrid_state": "rejected-hybrid",
            })()

    runner.executor = FakeExecutor()
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=5), seq_id=42)
    seq.block_table = [0, 1]
    batch = _scheduled_batch(tokens=[3], positions=[2], block_tables=[0, 1], seq_lens=3, is_prefill=False)
    batch.seq_ids = jnp.array([42], dtype=jnp.int32)

    token_ids = runner._run_mtp1([seq], batch)

    assert token_ids == [6]
    assert runner.cache_storage == "base-cache"


def test_executor_jit_matches_eager_cached_decode():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(2), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    eager_cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)
    jit_cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)

    prompt_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1],
        seq_lens=3,
        is_prefill=True,
    )
    eager_prompt = executor.forward_step(prompt_batch, cache_storage=eager_cache)
    empty_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 0, 1, 1)),
        recurrent_state=jnp.zeros((1, 0, 1, 1, 1)),
    )
    jit_prompt = executor.forward_step_jit(
        prompt_batch,
        cache_storage=jit_cache,
        hybrid_state=empty_hybrid,
    )

    decode_batch = _scheduled_batch(
        tokens=[4],
        positions=[3],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=False,
    )
    eager_decode = executor.forward_step(decode_batch, cache_storage=eager_prompt.cache_storage)
    jit_decode = executor.forward_step_jit(
        decode_batch,
        cache_storage=jit_prompt.cache_storage,
        hybrid_state=jit_prompt.hybrid_state,
    )
    compiled_entries_after_decode = len(executor._jit_cache)

    next_decode_batch = _scheduled_batch(
        tokens=[5],
        positions=[4],
        block_tables=[0, 1],
        seq_lens=5,
        is_prefill=False,
    )
    _ = executor.forward_step_jit(
        next_decode_batch,
        cache_storage=jit_decode.cache_storage,
        hybrid_state=jit_decode.hybrid_state,
    )
    assert len(executor._jit_cache) == compiled_entries_after_decode

    np.testing.assert_allclose(
        np.array(jit_decode.activations),
        np.array(eager_decode.activations),
        rtol=1e-5,
        atol=1e-5,
    )


def test_executor_mtp1_greedy_step_jit_matches_separate_path():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(12), config)
    params.mtp_params = init_mtp_params(jax.random.PRNGKey(13), config)
    params.mtp_params.lm_head = params.lm_head if params.lm_head is not None else params.embed_tokens.T
    executor = ModelExecutor(config, params, backend="pure_jax")
    spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)
    empty_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 0, 1, 1)),
        recurrent_state=jnp.zeros((1, 0, 1, 1, 1)),
    )

    prompt_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1],
        seq_lens=3,
        is_prefill=True,
    )
    prompt = executor.forward_step_jit(
        prompt_batch,
        cache_storage=cache,
        hybrid_state=empty_hybrid,
    )
    verifier_batch = _scheduled_batch(
        tokens=[3, 4],
        positions=[2, 3],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=True,
    )
    separate = executor.forward_step(
        verifier_batch,
        cache_storage=prompt.cache_storage,
        hybrid_state=prompt.hybrid_state,
        return_hidden=True,
        last_logits_only=False,
    )
    hidden_norm = rms_norm(separate.activations, params.norm_weight, config.rms_norm_eps).astype(jnp.float32)
    output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
    expected_tokens = jnp.argmax(jnp.dot(hidden_norm, output_weight), axis=-1).astype(jnp.int32)
    mtp_logits, _ = mtp_forward(
        hidden_state=hidden_norm[:, 1:2, :],
        next_token_ids=expected_tokens[:, 1:2],
        embed_tokens=params.embed_tokens,
        params=params.mtp_params,
        config=config,
        positions=jnp.array([[4]], dtype=jnp.int32),
    )
    expected_next_draft = jnp.argmax(mtp_logits[:, 0], axis=-1).astype(jnp.int32)[0]

    fused = executor.mtp1_greedy_step_jit(
        verifier_batch,
        cache_storage=prompt.cache_storage,
        hybrid_state=prompt.hybrid_state,
        draft_token=expected_tokens[0, 0],
        next_mtp_position=4,
        mtp_hidden_final_normed=True,
    )

    assert int(fused.target_token) == int(expected_tokens[0, 0])
    assert int(fused.bonus_token) == int(expected_tokens[0, 1])
    assert int(fused.next_draft_token) == int(expected_next_draft)
    assert bool(fused.accepted)
    np.testing.assert_allclose(
        np.array(fused.cache_storage.k_cache),
        np.array(separate.cache_storage.k_cache),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(fused.cache_storage.v_cache),
        np.array(separate.cache_storage.v_cache),
        rtol=1e-5,
        atol=1e-5,
    )


def test_prefix_cache_shares_blocks_and_refcounts():
    block_manager = BlockManager(num_blocks=6, block_size=2)
    seq_a = Sequence([1, 2, 9], SamplingParams(temperature=0.0), seq_id=1)
    seq_b = Sequence([1, 2, 7], SamplingParams(temperature=0.0), seq_id=2)

    block_manager.allocate(seq_a)
    block_manager.allocate(seq_b)

    shared_block = seq_a.block_table[0]
    assert shared_block == seq_b.block_table[0]
    assert block_manager.blocks[shared_block].ref_count == 2
    block_manager.deallocate(seq_a)
    assert block_manager.blocks[shared_block].ref_count == 1


def test_prefix_cache_allocation_allows_used_full_block_hit_without_free_blocks():
    block_manager = BlockManager(num_blocks=1, block_size=2)
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0), seq_id=1)
    seq_b = Sequence([1, 2], SamplingParams(temperature=0.0), seq_id=2)

    block_manager.allocate(seq_a)

    assert len(block_manager.free_block_ids) == 0
    assert block_manager.can_allocate(seq_b)
    block_manager.allocate(seq_b)
    assert seq_a.block_table == seq_b.block_table == [0]
    assert block_manager.blocks[0].ref_count == 2


def test_server_generate_accepts_batched_prompts(monkeypatch):
    import server

    class FakeEngine:
        config = type("Config", (), {"jax_execution": "jit"})()
        model_runner = type("Runner", (), {"executor": type("Executor", (), {"_jit_cache": {"prefill": True}})()})()

        def _tokenize(self, text):
            return list(range(len(text)))

        def generate(self, inputs, sampling_params, use_tqdm):
            assert len(inputs) == 2
            assert sampling_params.max_tokens == 2
            assert use_tqdm is False
            return [
                {"text": f"out-{idx}", "token_ids": [idx + 10, idx + 20]}
                for idx, _ in enumerate(inputs)
            ]

    monkeypatch.setattr(server, "engine", FakeEngine())

    response = server.app.test_client().post(
        "/v1/generate",
        json={"prompt": ["a", "bc"], "max_tokens": 2, "temperature": 0.0},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["results"]) == 2
    assert payload["usage"]["prompt_tokens"] == 3
    assert payload["usage"]["completion_tokens"] == 4
    assert payload["stats"]["jit_cache_entries"] == 1


def test_server_generate_keeps_single_input_ids_response_shape(monkeypatch):
    import server

    class FakeEngine:
        config = type("Config", (), {"jax_execution": "jit", "prefill_buckets": (), "batch_size_buckets": ()})()
        model_runner = type("Runner", (), {"executor": type("Executor", (), {"_jit_cache": {}})()})()

        def _tokenize(self, text):
            raise AssertionError("tokenizer should not be used for input_ids")

        def generate(self, inputs, sampling_params, use_tqdm):
            assert inputs == [[1, 2, 3]]
            return [{"text": "decoded", "token_ids": [4]}]

    monkeypatch.setattr(server, "engine", FakeEngine())

    response = server.app.test_client().post(
        "/v1/generate",
        json={"input_ids": [1, 2, 3], "max_tokens": 1},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "results" not in payload
    assert payload["token_ids"] == [4]
    assert payload["usage"]["prompt_tokens"] == 3


def test_server_generate_rejects_prompt_that_exceeds_prefill_bucket(monkeypatch):
    import server

    class FakeEngine:
        config = type(
            "Config",
            (),
            {"prefill_buckets": (1,), "batch_size_buckets": (), "max_num_seqs": 1, "max_blocks_per_seq": None},
        )()
        model_runner = type("Runner", (), {"executor": type("Executor", (), {"_jit_cache": {}})()})()

        def _tokenize(self, text):
            return list(text)

        def generate(self, inputs, sampling_params, use_tqdm):
            raise AssertionError("generation should not run for an oversized prompt")

    monkeypatch.setattr(server, "engine", FakeEngine())

    response = server.app.test_client().post(
        "/v1/generate",
        json={"prompt": "too long", "max_tokens": 1},
    )

    assert response.status_code == 400
    assert "largest prefill bucket" in response.get_json()["error"]


def test_server_generate_rejects_request_that_exceeds_kv_capacity(monkeypatch):
    import server

    class FakeEngine:
        config = type(
            "Config",
            (),
            {
                "prefill_buckets": (),
                "batch_size_buckets": (),
                "max_num_seqs": 1,
                "max_blocks_per_seq": 2,
                "block_size": 2,
            },
        )()
        model_runner = type("Runner", (), {"executor": type("Executor", (), {"_jit_cache": {}})()})()

        def _tokenize(self, text):
            return [1, 2, 3]

        def generate(self, inputs, sampling_params, use_tqdm):
            raise AssertionError("generation should not run for an over-capacity request")

    monkeypatch.setattr(server, "engine", FakeEngine())

    response = server.app.test_client().post(
        "/v1/generate",
        json={"prompt": "abc", "max_tokens": 2},
    )

    assert response.status_code == 400
    assert "per-sequence KV capacity" in response.get_json()["error"]


def test_server_generate_rejects_bad_sampling_values(monkeypatch):
    import server

    class FakeEngine:
        config = type("Config", (), {"prefill_buckets": (), "batch_size_buckets": (), "max_blocks_per_seq": None})()
        model_runner = type("Runner", (), {"executor": type("Executor", (), {"_jit_cache": {}})()})()

        def _tokenize(self, text):
            return [1]

        def generate(self, inputs, sampling_params, use_tqdm):
            raise AssertionError("generation should not run for bad sampling values")

    monkeypatch.setattr(server, "engine", FakeEngine())

    response = server.app.test_client().post(
        "/v1/generate",
        json={"prompt": "x", "max_tokens": None},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "max_tokens must be an integer"
