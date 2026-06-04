"""Backend/cache boundary tests that do not require model weights."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from nanovllm_jax.backends import PureJAXBackend, select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.model_executor import ModelExecutor
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.sequence import SamplingParams, Sequence, SequenceStatus
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import (
    HybridLayerState,
    KVCacheSpec,
    cap_num_kv_cache_blocks,
    init_hybrid_state,
    init_kv_cache,
    paged_attention_decode,
    paged_attention_prefill,
    paged_attention_prefill_packed,
)
from nanovllm_jax.kernels.paged_attention import (
    dense_block_tables_to_kv_indptr,
    kv_last_page_len_from_seq_lens,
    paged_decode_attention_gqa_nhd_reference,
)
from nanovllm_jax.layers import causal_conv1d_update, rms_norm
from nanovllm_jax.model import (
    _can_use_decode_padded_gemm,
    _decode_padded_gemm_dot,
    _packed_causal_conv1d_prefill,
    forward,
    init_params,
)
from nanovllm_jax.mtp.mtp_layer import init_mtp_params, mtp_forward


def _has_cuda_backend() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _has_jax_triton() -> bool:
    return importlib.util.find_spec("jax_triton") is not None


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
        prefill_layout="dense",
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
        prefill_layout="dense",
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


def test_full_attention_kv_cache_dtype_override(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE", "bf16")
    backend = PureJAXBackend()
    spec = KVCacheSpec(
        num_layers=2,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=8,
        dtype=jnp.float32,
        max_kv_cache_bytes=4096,
    )

    cache = backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=4)

    assert cache.k_cache.dtype == jnp.bfloat16
    assert cache.v_cache.dtype == jnp.bfloat16


def test_full_attention_kv_cache_dtype_default_config_uses_spec_dtype(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE", raising=False)
    config = Qwen3_5Config(full_attention_kv_cache_dtype="default")
    backend = PureJAXBackend(config)
    spec = KVCacheSpec(
        num_layers=2,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=8,
        dtype=jnp.float32,
        max_kv_cache_bytes=4096,
    )

    cache = backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=4)

    assert cache.k_cache.dtype == jnp.float32
    assert cache.v_cache.dtype == jnp.float32


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
    assert batch.seq_ids_host == (7, 8, -1, -1)
    assert batch.query_lens_host == (3, 1, 0, 0)
    assert batch.seq_lens_host == (3, 1, 0, 0)
    assert batch.num_prefill_tokens == 4


def _hybrid_state_runner_with_two_slots() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner._max_hybrid_slots = 2
    runner._hybrid_slots = {0: 0, 1: 1}
    runner._free_hybrid_slots = []
    runner._hybrid_state_table = HybridLayerState(
        conv_state=jnp.arange(24, dtype=jnp.float32).reshape(2, 1, 3, 4),
        recurrent_state=jnp.arange(12, dtype=jnp.float32).reshape(2, 1, 1, 2, 3),
    )
    return runner


def _hybrid_state_batch(seq_ids, query_lens) -> ScheduledBatch:
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    return ScheduledBatch(
        tokens=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        positions=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        seq_ids=jnp.array(seq_ids, dtype=jnp.int32),
        query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=sum(1 for seq_id, query_len in zip(seq_ids, query_lens) if seq_id >= 0 and query_len > 0),
        block_tables=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        seq_lens=jnp.array([1 if seq_id >= 0 else 0 for seq_id in seq_ids], dtype=jnp.int32),
        seq_ids_host=tuple(seq_ids),
        query_lens_host=tuple(query_lens),
        seq_lens_host=tuple(1 if seq_id >= 0 else 0 for seq_id in seq_ids),
    )


def test_model_runner_hybrid_state_uses_full_table_fast_path():
    runner = _hybrid_state_runner_with_two_slots()
    batch = _hybrid_state_batch([8, 1], [1, 1])

    batched_state = runner._batch_hybrid_state(batch)

    assert batched_state is runner._hybrid_state_table
    assert batch.hybrid_slot_ids_host == (0, 1)

    new_state = HybridLayerState(
        conv_state=jnp.full_like(runner._hybrid_state_table.conv_state, 10.0),
        recurrent_state=jnp.full_like(runner._hybrid_state_table.recurrent_state, 20.0),
    )
    runner._store_batch_hybrid_state(batch, new_state)

    assert runner._hybrid_state_table is new_state


def test_model_runner_hybrid_state_prefers_physical_row_slots_for_new_sequences():
    runner = _hybrid_state_runner_with_two_slots()
    runner._hybrid_slots = {}
    runner._free_hybrid_slots = [0, 1]
    batch = _hybrid_state_batch([8, 9], [1, 1])

    batched_state = runner._batch_hybrid_state(batch)

    assert batched_state is runner._hybrid_state_table
    assert runner._hybrid_slots == {8: 0, 9: 1}
    assert batch.hybrid_slot_ids_host == (0, 1)
    np.testing.assert_array_equal(np.array(batched_state.conv_state), np.zeros((2, 1, 3, 4), dtype=np.float32))
    np.testing.assert_array_equal(np.array(batched_state.recurrent_state), np.zeros((2, 1, 1, 2, 3), dtype=np.float32))


def test_model_runner_hybrid_slot_ids_zero_new_allocations():
    runner = _hybrid_state_runner_with_two_slots()
    runner._hybrid_slots = {1: 1}
    runner._free_hybrid_slots = [0]
    runner._hybrid_state_table = HybridLayerState(
        conv_state=jnp.full_like(runner._hybrid_state_table.conv_state, 7.0),
        recurrent_state=jnp.full_like(runner._hybrid_state_table.recurrent_state, 9.0),
    )
    batch = _hybrid_state_batch([8, 1], [1, 1])

    slot_ids = runner._batch_hybrid_slot_ids(batch)

    np.testing.assert_array_equal(np.asarray(slot_ids), np.array([0, 1], dtype=np.int32))
    assert batch.hybrid_slot_ids_host == (0, 1)
    assert runner._hybrid_slots == {1: 1, 8: 0}
    np.testing.assert_array_equal(
        np.array(runner._hybrid_state_table.conv_state[0]),
        np.zeros((1, 3, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(runner._hybrid_state_table.recurrent_state[0]),
        np.zeros((1, 1, 2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(runner._hybrid_state_table.conv_state[1]),
        np.full((1, 3, 4), 7.0, dtype=np.float32),
    )


def test_model_runner_hybrid_state_zeroes_reused_slot_on_allocation_not_release():
    runner = _hybrid_state_runner_with_two_slots()
    runner.hybrid_states = {0: object()}
    runner._mtp1_drafts = {0: 123}
    zeroed_slots: list[int] = []
    runner._zero_hybrid_slots = lambda slots: zeroed_slots.extend(slots)

    runner.release([0])

    assert zeroed_slots == []
    assert runner._hybrid_slots == {1: 1}
    assert runner._free_hybrid_slots == [0]
    assert runner.hybrid_states == {}
    assert runner._mtp1_drafts == {}

    slot = runner._ensure_hybrid_slot(8, preferred_slot=0)

    assert slot == 0
    assert zeroed_slots == [0]
    assert runner._hybrid_slots == {1: 1, 8: 0}


def test_model_runner_hybrid_state_does_not_replace_full_table_with_inactive_rows():
    runner = _hybrid_state_runner_with_two_slots()
    original_state = runner._hybrid_state_table
    batch = _hybrid_state_batch([0, -1], [1, 0])

    batched_state = runner._batch_hybrid_state(batch)

    assert batched_state is not runner._hybrid_state_table
    assert batch.hybrid_slot_ids_host == (0, -1)
    np.testing.assert_array_equal(np.array(batched_state.conv_state[1]), np.zeros((1, 3, 4), dtype=np.float32))
    np.testing.assert_array_equal(np.array(batched_state.recurrent_state[1]), np.zeros((1, 1, 2, 3), dtype=np.float32))

    new_state = HybridLayerState(
        conv_state=jnp.full_like(original_state.conv_state, 10.0),
        recurrent_state=jnp.full_like(original_state.recurrent_state, 20.0),
    )
    runner._store_batch_hybrid_state(batch, new_state)

    assert runner._hybrid_state_table is not new_state
    np.testing.assert_array_equal(np.array(runner._hybrid_state_table.conv_state[0]), np.array(new_state.conv_state[0]))
    np.testing.assert_array_equal(np.array(runner._hybrid_state_table.recurrent_state[0]), np.array(new_state.recurrent_state[0]))
    np.testing.assert_array_equal(np.array(runner._hybrid_state_table.conv_state[1]), np.array(original_state.conv_state[1]))
    np.testing.assert_array_equal(
        np.array(runner._hybrid_state_table.recurrent_state[1]),
        np.array(original_state.recurrent_state[1]),
    )


def test_mtp_admission_gate_tracks_logical_decode_rows(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE", "0")
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_MIN_ACCEPT_SAMPLES", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_MIN_SPEEDUP", "1.0")
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_LATENCY_MIN_STEPS", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_LATENCY_ALPHA", "1.0")
    config = _tiny_full_attention_config()
    config.num_speculative_tokens = 1
    config.batch_size_buckets = (4,)
    scheduler = Scheduler(config)

    assert scheduler.mtp_scheduler_gate_enabled

    batch_active_2 = ScheduledBatch(
        tokens=jnp.zeros((4, 1), dtype=jnp.int32),
        positions=jnp.zeros((4, 1), dtype=jnp.int32),
        seq_ids=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2, 2, 2], dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=2,
        block_tables=jnp.zeros((4, 2), dtype=jnp.int32),
        seq_lens=jnp.array([3, 3, 0, 0], dtype=jnp.int32),
    )

    scheduler.update_mtp_admission(
        {},
        is_decode=True,
        elapsed_seconds=0.02,
        emitted_tokens=2,
        batch=batch_active_2,
    )
    scheduler.update_mtp_admission(
        {"drafts_accepted": 1, "drafts_rejected": 0},
        is_decode=True,
        elapsed_seconds=0.06,
        emitted_tokens=2,
        batch=batch_active_2,
    )

    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4), seq_id=99)
    assert not scheduler.should_admit_mtp(
        seq,
        for_decode=True,
        batch_size_bucket=4,
        active_decode_rows=2,
    )
    assert scheduler.mtp_admission_reason == "low_throughput"

    assert scheduler.should_admit_mtp(
        seq,
        for_decode=True,
        batch_size_bucket=4,
        active_decode_rows=4,
    )
    assert scheduler.mtp_admission_reason == "probing_mtp"

    report = scheduler.get_mtp_admission_report()
    buckets = {bucket["key"]["active_decode_rows"]: bucket for bucket in report["buckets"]}
    assert buckets[2]["key"]["physical_batch_size"] == 4
    assert buckets[2]["admission_reason"] == "low_throughput"
    assert buckets[2]["measured_speedup"] == pytest.approx(1 / 3)
    assert buckets[4]["admission_reason"] == "probing_mtp"


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


def test_scheduler_chunks_prefill_by_max_batched_tokens_budget():
    config = _tiny_full_attention_config()
    config.max_blocks_per_seq = 8
    config.max_num_batched_tokens = 3
    scheduler = Scheduler(config)
    seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=0.0, max_tokens=1), seq_id=3)

    scheduler.add(seq)

    first_batch_seqs, first_batch = scheduler.schedule()
    assert first_batch.is_prefill
    assert first_batch.prefill_final_flags == [False]
    assert int(first_batch.query_lens[0]) == 3
    np.testing.assert_array_equal(
        np.array(first_batch.positions),
        np.array([[0, 1, 2]]),
    )

    finished = scheduler.postprocess(first_batch_seqs, [[]], prefill_chunk_lengths=[3])
    assert finished == [False]
    assert first_batch_seqs[0].num_cached_tokens == 3

    second_batch_seqs, second_batch = scheduler.schedule()
    assert second_batch.is_prefill
    assert second_batch.prefill_final_flags == [True]
    assert int(second_batch.query_lens[0]) == 1
    np.testing.assert_array_equal(
        np.array(second_batch.positions),
        np.array([[3, 0]]),
    )

    finished = scheduler.postprocess(second_batch_seqs, [99], prefill_chunk_lengths=[1])
    assert finished == [True]
    assert second_batch_seqs[0].completion_token_ids == [99]
    assert second_batch_seqs[0].status == SequenceStatus.FINISHED
    assert second_batch_seqs[0].num_cached_tokens == 0
    assert second_batch_seqs[0].block_table == []


def test_scheduler_continues_running_prefill_when_waiting_head_needs_kv():
    config = _tiny_full_attention_config()
    config.max_num_seqs = 2
    config.num_kvcache_blocks = 4
    config.max_blocks_per_seq = 4
    config.max_num_batched_tokens = 3
    scheduler = Scheduler(config)

    running = Sequence([1, 2, 3, 4, 5, 6], SamplingParams(temperature=0.0, max_tokens=1), seq_id=0)
    waiting = Sequence([11, 12, 13, 14], SamplingParams(temperature=0.0, max_tokens=1), seq_id=1)
    scheduler.add(running)
    scheduler.add(waiting)

    first_batch_seqs, first_batch = scheduler.schedule()
    assert first_batch.is_prefill
    assert [seq.seq_id for seq in first_batch_seqs] == [0]
    assert int(first_batch.query_lens[0]) == 3
    assert len(scheduler.block_manager.free_block_ids) == 1

    finished = scheduler.postprocess(first_batch_seqs, [[]], prefill_chunk_lengths=[3])
    assert finished == [False]
    assert [seq.seq_id for seq in scheduler.running] == [0]
    assert [seq.seq_id for seq in scheduler.waiting] == [1]

    second_batch_seqs, second_batch = scheduler.schedule()
    assert second_batch.is_prefill
    assert [seq.seq_id for seq in second_batch_seqs] == [0]
    assert int(second_batch.query_lens[0]) == 3
    assert second_batch.prefill_final_flags == [True]
    assert [seq.seq_id for seq in scheduler.waiting] == [1]
    assert waiting.block_table == []


def test_scheduler_prefill_packing_respects_padded_bucket_budget():
    config = _tiny_full_attention_config()
    config.max_num_seqs = 2
    config.num_kvcache_blocks = 20
    config.max_blocks_per_seq = 8
    config.max_num_batched_tokens = 8
    config.prefill_buckets = (4, 8)
    config.batch_size_buckets = (1, 2)
    scheduler = Scheduler(config)

    long = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0.0, max_tokens=1), seq_id=0)
    short = Sequence([9], SamplingParams(temperature=0.0, max_tokens=1), seq_id=1)
    scheduler.add(long)
    scheduler.add(short)

    seqs, batch = scheduler.schedule()

    assert [seq.seq_id for seq in seqs] == [0]
    assert batch.tokens.shape == (1, 8)
    assert int(batch.num_prefill_tokens) == 5
    assert [seq.seq_id for seq in scheduler.waiting] == [1]
    assert short.block_table == []


def test_scheduler_builds_packed_prefill_token_bucket():
    config = _tiny_full_attention_config()
    config.prefill_layout = "packed"
    config.max_num_seqs = 4
    config.num_kvcache_blocks = 20
    config.max_blocks_per_seq = 8
    config.max_num_batched_tokens = 8
    config.prefill_token_buckets = (8,)
    config.batch_size_buckets = (4,)
    scheduler = Scheduler(config)

    seq_a = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0.0, max_tokens=1), seq_id=10)
    seq_b = Sequence([6, 7], SamplingParams(temperature=0.0, max_tokens=1), seq_id=11)
    scheduler.add(seq_a)
    scheduler.add(seq_b)

    seqs, batch = scheduler.schedule()

    assert [seq.seq_id for seq in seqs] == [10, 11]
    assert batch.packed_prefill
    assert batch.tokens.shape == (1, 8)
    assert batch.positions.shape == (1, 8)
    assert batch.token_row_ids.shape == (1, 8)
    assert batch.block_tables.shape == (4, 8)
    assert batch.seq_ids_host == (10, 11, -1, -1)
    assert batch.query_lens_host == (5, 2, 0, 0)
    assert int(batch.num_prefill_tokens) == 7
    np.testing.assert_array_equal(np.array(batch.query_start_loc), np.array([0, 5, 7, 7, 7]))
    np.testing.assert_array_equal(np.array(batch.token_row_ids), np.array([[0, 0, 0, 0, 0, 1, 1, 0]]))


def test_scheduler_packed_prefill_uses_prefill_bucket_as_row_chunk_budget():
    config = _tiny_full_attention_config()
    config.prefill_layout = "packed"
    config.max_num_seqs = 4
    config.num_kvcache_blocks = 20
    config.max_blocks_per_seq = 8
    config.max_num_batched_tokens = 8
    config.prefill_buckets = (4,)
    config.prefill_token_buckets = (8,)
    config.batch_size_buckets = (2,)
    scheduler = Scheduler(config)

    seq_a = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0.0, max_tokens=1), seq_id=10)
    seq_b = Sequence([6, 7, 8, 9, 10], SamplingParams(temperature=0.0, max_tokens=1), seq_id=11)
    scheduler.add(seq_a)
    scheduler.add(seq_b)

    seqs, batch = scheduler.schedule()

    assert [seq.seq_id for seq in seqs] == [10, 11]
    assert batch.packed_prefill
    assert batch.query_lens_host == (4, 4)
    assert int(batch.num_prefill_tokens) == 8
    np.testing.assert_array_equal(np.array(batch.query_start_loc), np.array([0, 4, 8]))


def test_packed_prefill_metadata_uses_paged_slot_mapping():
    backend = PureJAXBackend()
    positions = jnp.array([[0, 1, 2, 0]], dtype=jnp.int32)
    token_row_ids = jnp.array([[0, 0, 1, 0]], dtype=jnp.int32)
    block_tables = jnp.array([[3, 4], [5, 6]], dtype=jnp.int32)
    metadata = backend.build_attention_metadata(
        positions=positions,
        block_tables=block_tables,
        seq_lens=jnp.array([2, 3], dtype=jnp.int32),
        block_size=2,
        is_prefill=True,
        query_start_loc=jnp.array([0, 2, 3], dtype=jnp.int32),
        num_prefill_tokens=3,
        num_decode_tokens=0,
        token_row_ids=token_row_ids,
    )

    np.testing.assert_array_equal(np.array(metadata.slot_mapping), np.array([[6, 7, 12, 6]]))
    assert metadata.token_row_ids is token_row_ids


def test_packed_full_attention_prefill_matches_dense_rows():
    key = jax.random.PRNGKey(29)
    block_size = 2
    num_kv_heads = 1
    num_heads = 2
    head_dim = 4
    num_groups = num_heads // num_kv_heads
    max_blocks = 3
    max_kv_len = max_blocks * block_size
    block_table = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    k_cache = jax.random.normal(
        key,
        (1, 6 * block_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    )
    v_cache = jax.random.normal(
        jax.random.fold_in(key, 1),
        (1, 6 * block_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    )
    query_packed = jax.random.normal(
        jax.random.fold_in(key, 2),
        (1, 6, num_heads, head_dim),
        dtype=jnp.float32,
    )
    positions_packed = jnp.array([[0, 1, 2, 0, 1, 0]], dtype=jnp.int32)
    token_row_ids = jnp.array([[0, 0, 0, 1, 1, 0]], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    seq_lens = jnp.array([3, 2], dtype=jnp.int32)

    actual = paged_attention_prefill_packed(
        query=query_packed,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        kv_lens=seq_lens,
        positions=positions_packed,
        token_row_ids=token_row_ids,
        query_start_loc=query_start_loc,
        block_size=block_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_groups,
        layer_idx=0,
        max_query_len=3,
    )

    query_dense = jnp.zeros((2, 3, num_heads, head_dim), dtype=jnp.float32)
    query_dense = query_dense.at[0, :3].set(query_packed[0, :3])
    query_dense = query_dense.at[1, :2].set(query_packed[0, 3:5])
    positions_dense = jnp.array([[0, 1, 2], [0, 1, 0]], dtype=jnp.int32)
    dense = paged_attention_prefill(
        query=query_dense,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        kv_lens=seq_lens,
        positions=positions_dense,
        block_size=block_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_groups,
        layer_idx=0,
    )
    expected = jnp.zeros((1, 6, num_heads * head_dim), dtype=jnp.float32)
    expected = expected.at[0, :3].set(dense[0, :3])
    expected = expected.at[0, 3:5].set(dense[1, :2])

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-6)


def test_packed_causal_conv1d_prefill_matches_token_scan():
    key = jax.random.PRNGKey(37)
    token_bucket = 8
    conv_dim = 5
    kernel_size = 3
    row_count = 2
    mixed_qkv = jax.random.normal(
        key,
        (1, token_bucket, conv_dim),
        dtype=jnp.float32,
    )
    initial_state = jax.random.normal(
        jax.random.fold_in(key, 1),
        (row_count, conv_dim, kernel_size),
        dtype=jnp.float32,
    )
    weight = jax.random.normal(
        jax.random.fold_in(key, 2),
        (conv_dim, kernel_size),
        dtype=jnp.float32,
    )
    bias = jax.random.normal(jax.random.fold_in(key, 3), (conv_dim,), dtype=jnp.float32)
    token_row_ids = jnp.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)

    actual_out, actual_state = _packed_causal_conv1d_prefill(
        mixed_qkv,
        initial_state,
        weight,
        bias,
        token_row_ids,
        query_start_loc,
        max_row_tokens=4,
    )

    state = initial_state
    expected_tokens = []
    safe_rows = token_row_ids.reshape(-1)
    valid = jnp.arange(token_bucket, dtype=jnp.int32) < query_start_loc[-1]
    for token_idx in range(token_bucket):
        row = int(safe_rows[token_idx])
        previous = state[row]
        conv_t, next_state = causal_conv1d_update(
            mixed_qkv[:, token_idx, :].reshape(1, conv_dim, 1),
            previous[None, :, :],
            weight,
            bias,
            "silu",
        )
        if bool(valid[token_idx]):
            state = state.at[row].set(next_state[0])
            expected_tokens.append(conv_t[0, :, 0])
        else:
            expected_tokens.append(jnp.zeros((conv_dim,), dtype=jnp.float32))
    expected_out = jnp.stack(expected_tokens, axis=0)[None, :, :]

    np.testing.assert_allclose(np.asarray(actual_out), np.asarray(expected_out), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_state), np.asarray(state), rtol=0, atol=1e-6)


def test_decode_padded_gemm_supports_small_decode_batches(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DECODE_PADDED_GEMM", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS", "8")
    monkeypatch.setenv("NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM", "16")
    x = (jnp.arange(4 * 1 * 3, dtype=jnp.float32).reshape(4, 1, 3) / 10.0).astype(jnp.bfloat16)
    weight = (jnp.arange(3 * 5, dtype=jnp.float32).reshape(3, 5) / 7.0).astype(jnp.bfloat16)

    assert _can_use_decode_padded_gemm(x, weight)
    actual = _decode_padded_gemm_dot(x, weight)
    expected = jnp.dot(x.reshape(4, 3), weight).reshape(4, 1, 5)

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_packed_full_attention_prefill_triton_matches_dense_rows_bf16_cache():
    key = jax.random.PRNGKey(31)
    block_size = 2
    num_kv_heads = 1
    num_heads = 2
    head_dim = 8
    num_groups = num_heads // num_kv_heads
    block_table = jnp.array([[2, 0, 4], [3, 1, 5]], dtype=jnp.int32)
    k_cache = jax.random.normal(
        key,
        (1, 6, block_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.fold_in(key, 1),
        (1, 6, block_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    query_packed = jax.random.normal(
        jax.random.fold_in(key, 2),
        (1, 8, num_heads, head_dim),
        dtype=jnp.float32,
    )
    positions_packed = jnp.array([[0, 1, 2, 0, 1, 0, 0, 0]], dtype=jnp.int32)
    token_row_ids = jnp.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    seq_lens = jnp.array([3, 2], dtype=jnp.int32)

    actual = paged_attention_prefill_packed(
        query=query_packed,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        kv_lens=seq_lens,
        positions=positions_packed,
        token_row_ids=token_row_ids,
        query_start_loc=query_start_loc,
        block_size=block_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_groups,
        layer_idx=0,
        max_query_len=3,
    )

    query_dense = jnp.zeros((2, 3, num_heads, head_dim), dtype=jnp.float32)
    query_dense = query_dense.at[0, :3].set(query_packed[0, :3])
    query_dense = query_dense.at[1, :2].set(query_packed[0, 3:5])
    positions_dense = jnp.array([[0, 1, 2], [0, 1, 0]], dtype=jnp.int32)
    dense = paged_attention_prefill(
        query=query_dense.astype(jnp.bfloat16),
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        kv_lens=seq_lens,
        positions=positions_dense,
        block_size=block_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_groups,
        layer_idx=0,
    )
    expected = jnp.zeros((1, 8, num_heads * head_dim), dtype=dense.dtype)
    expected = expected.at[0, :3].set(dense[0, :3])
    expected = expected.at[0, 3:5].set(dense[1, :2])

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not _has_cuda_backend(), reason="CUDA JAX backend is required")
@pytest.mark.skipif(not _has_jax_triton(), reason="jax-triton is required")
def test_paged_decode_attention_triton_matches_reference_bf16_cache():
    from nanovllm_jax.kernels.full_attention_triton import paged_decode_attention_triton

    key = jax.random.PRNGKey(41)
    block_size = 2
    num_kv_heads = 1
    num_heads = 2
    head_dim = 16
    num_groups = num_heads // num_kv_heads
    block_table = jnp.array([[2, 0, 4], [3, 1, 5], [0, 0, 0]], dtype=jnp.int32)
    seq_lens = jnp.array([5, 3, 0], dtype=jnp.int32)
    k_cache = jax.random.normal(
        key,
        (1, 6, block_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.fold_in(key, 1),
        (1, 6, block_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    query = jax.random.normal(
        jax.random.fold_in(key, 2),
        (3, 1, num_heads, head_dim),
        dtype=jnp.float32,
    )

    actual = paged_decode_attention_triton(
        query=query,
        k_cache_layer=k_cache[0],
        v_cache_layer=v_cache[0],
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        scale=1.0 / np.sqrt(head_dim),
        num_key_value_groups=num_groups,
    )
    kv_indices, kv_indptr = dense_block_tables_to_kv_indptr(block_table)
    expected = paged_decode_attention_gqa_nhd_reference(
        query[:, 0],
        k_cache[0],
        v_cache[0],
        kv_indptr,
        kv_indices,
        kv_last_page_len_from_seq_lens(seq_lens, block_size),
        seq_lens,
        1.0 / np.sqrt(head_dim),
        max_pages_per_sequence=block_table.shape[1],
    ).reshape(3, 1, num_heads * head_dim)

    np.testing.assert_allclose(
        np.asarray(actual[:2], dtype=np.float32),
        np.asarray(expected[:2], dtype=np.float32),
        rtol=3e-2,
        atol=3e-2,
    )


def test_packed_prefill_gdn_reference_updates_rows_independently():
    config = _tiny_linear_attention_config()
    config.prefill_layout = "packed"
    config.num_kvcache_blocks = 8
    params = init_params(jax.random.PRNGKey(17), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    cache = init_kv_cache(
        config.num_kvcache_blocks,
        config.block_size,
        config.num_key_value_heads,
        config.head_dim,
        max_seqs=2,
        max_blocks_per_seq=4,
        dtype=jnp.float32,
    ).storage
    hybrid = init_hybrid_state(config, batch_size=2, dtype=jnp.float32)
    batch = ScheduledBatch(
        tokens=jnp.array([[1, 2, 3, 4, 5, 0]], dtype=jnp.int32),
        positions=jnp.array([[0, 1, 2, 0, 1, 0]], dtype=jnp.int32),
        seq_ids=jnp.array([10, 11], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32),
        is_prefill=True,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        block_tables=jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32),
        seq_lens=jnp.array([3, 2], dtype=jnp.int32),
        seq_ids_host=(10, 11),
        query_lens_host=(3, 2),
        seq_lens_host=(3, 2),
        packed_prefill=True,
        token_row_ids=jnp.array([[0, 0, 0, 1, 1, 0]], dtype=jnp.int32),
    )

    out = executor.forward_step(
        batch,
        cache_storage=cache,
        hybrid_state=hybrid,
        last_logits_only=True,
    )

    assert out.activations.shape == (2, 1, config.vocab_size)
    assert out.hybrid_state.conv_state.shape[0] == 2
    assert out.hybrid_state.recurrent_state.shape[0] == 2
    assert float(jnp.max(jnp.abs(out.hybrid_state.conv_state[0] - out.hybrid_state.conv_state[1]))) > 0.0


def test_packed_prefill_gdn_post_conv_reference_matches_scan(monkeypatch):
    config = _tiny_linear_attention_config()
    config.prefill_layout = "packed"
    config.num_kvcache_blocks = 8
    params = init_params(jax.random.PRNGKey(23), config)

    def make_batch():
        return ScheduledBatch(
            tokens=jnp.array([[1, 2, 3, 4, 5, 0]], dtype=jnp.int32),
            positions=jnp.array([[0, 1, 2, 0, 1, 0]], dtype=jnp.int32),
            seq_ids=jnp.array([10, 11], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32),
            is_prefill=True,
            num_prefill_tokens=5,
            num_decode_tokens=0,
            block_tables=jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32),
            seq_lens=jnp.array([3, 2], dtype=jnp.int32),
            seq_ids_host=(10, 11),
            query_lens_host=(3, 2),
            seq_lens_host=(3, 2),
            packed_prefill=True,
            token_row_ids=jnp.array([[0, 0, 0, 1, 1, 0]], dtype=jnp.int32),
        )

    def run_once():
        executor = ModelExecutor(config, params, backend="pure_jax")
        cache = init_kv_cache(
            config.num_kvcache_blocks,
            config.block_size,
            config.num_key_value_heads,
            config.head_dim,
            max_seqs=2,
            max_blocks_per_seq=4,
            dtype=jnp.float32,
        ).storage
        hybrid = init_hybrid_state(config, batch_size=2, dtype=jnp.float32)
        return executor.forward_step(
            make_batch(),
            cache_storage=cache,
            hybrid_state=hybrid,
            last_logits_only=True,
        )

    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", raising=False)
    scan_out = run_once()
    monkeypatch.setenv("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", "reference")
    backend_out = run_once()

    np.testing.assert_allclose(
        np.asarray(backend_out.activations),
        np.asarray(scan_out.activations),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(backend_out.hybrid_state.conv_state),
        np.asarray(scan_out.hybrid_state.conv_state),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(backend_out.hybrid_state.recurrent_state),
        np.asarray(scan_out.hybrid_state.recurrent_state),
        rtol=0,
        atol=1e-5,
    )


def test_packed_prefill_greedy_token_jit_returns_row_tokens():
    config = _tiny_linear_attention_config()
    config.prefill_layout = "packed"
    config.num_kvcache_blocks = 8
    params = init_params(jax.random.PRNGKey(19), config)
    executor = ModelExecutor(config, params, backend="pure_jax")
    cache = init_kv_cache(
        config.num_kvcache_blocks,
        config.block_size,
        config.num_key_value_heads,
        config.head_dim,
        max_seqs=2,
        max_blocks_per_seq=4,
        dtype=jnp.float32,
    ).storage
    hybrid = init_hybrid_state(config, batch_size=2, dtype=jnp.float32)
    batch = ScheduledBatch(
        tokens=jnp.array([[1, 2, 3, 4, 5, 0]], dtype=jnp.int32),
        positions=jnp.array([[0, 1, 2, 0, 1, 0]], dtype=jnp.int32),
        seq_ids=jnp.array([10, 11], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32),
        is_prefill=True,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        block_tables=jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32),
        seq_lens=jnp.array([3, 2], dtype=jnp.int32),
        seq_ids_host=(10, 11),
        query_lens_host=(3, 2),
        seq_lens_host=(3, 2),
        packed_prefill=True,
        token_row_ids=jnp.array([[0, 0, 0, 1, 1, 0]], dtype=jnp.int32),
    )

    out = executor.forward_step_token_ids_jit(
        batch,
        cache_storage=cache,
        hybrid_state=hybrid,
    )

    assert out.activations.shape == (2,)
    assert out.activations.dtype == jnp.int32


def test_scheduler_does_not_admit_waiting_prompts_when_active_slots_are_full():
    config = _tiny_full_attention_config()
    config.max_num_seqs = 2
    config.num_kvcache_blocks = 8
    config.max_blocks_per_seq = 4
    config.max_num_batched_tokens = 8
    scheduler = Scheduler(config)

    seqs = [
        Sequence([1 + idx * 2, 2 + idx * 2], SamplingParams(temperature=0.0, max_tokens=2), seq_id=idx)
        for idx in range(4)
    ]
    for seq in seqs:
        scheduler.add(seq)

    prefill_seqs, prefill_batch = scheduler.schedule()
    assert prefill_batch.is_prefill
    assert [seq.seq_id for seq in prefill_seqs] == [0, 1]

    finished = scheduler.postprocess(prefill_seqs, [[], []], prefill_chunk_lengths=[2, 2])
    assert finished == [False, False]
    assert [seq.seq_id for seq in scheduler.running] == [0, 1]
    assert [seq.seq_id for seq in scheduler.waiting] == [2, 3]

    decode_seqs, decode_batch = scheduler.schedule()
    assert not decode_batch.is_prefill
    assert [seq.seq_id for seq in decode_seqs] == [0, 1]
    assert [seq.seq_id for seq in scheduler.waiting] == [2, 3]


def test_scheduler_rejects_single_prompt_larger_than_prefill_budget():
    config = _tiny_full_attention_config()
    config.max_blocks_per_seq = 1
    scheduler = Scheduler(config)

    with pytest.raises(ValueError, match="prompt needs 2 blocks"):
        scheduler.add(Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=1), seq_id=4))

    with pytest.raises(ValueError, match="per-sequence capacity"):
        scheduler.add(Sequence([1], SamplingParams(temperature=0.0, max_tokens=2), seq_id=5))


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
    cache_for_reference = executor.backend.allocate_kv_cache(
        spec, max_seqs=1, max_blocks_per_seq=2
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
    exact_batch = _scheduled_batch(
        tokens=[1, 2, 3, 4],
        positions=[0, 1, 2, 3],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=True,
    )
    exact_out = executor.forward_step(exact_batch, cache_storage=cache_for_reference)

    exact_k_slot3 = exact_out.cache_storage.k_cache[0, 1, 1, 0, :]
    decode_k_slot3 = decode_out.cache_storage.k_cache[0, 1, 1, 0, :]
    exact_v_slot3 = exact_out.cache_storage.v_cache[0, 1, 1, 0, :]
    decode_v_slot3 = decode_out.cache_storage.v_cache[0, 1, 1, 0, :]

    # TPU exhibits small compile/runtime drift for staged decode-cache writes and output logits;
    # keep this as a bounded parity check rather than strict exactness.
    np.testing.assert_allclose(
        np.array(exact_k_slot3),
        np.array(decode_k_slot3),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.array(exact_v_slot3),
        np.array(decode_v_slot3),
        rtol=1e-2,
        atol=1e-2,
    )

    np.testing.assert_allclose(
        np.array(decode_out.activations[:, 0]),
        np.array(exact_out.activations[:, -1]),
        rtol=1e-2,
        atol=1e-2,
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
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_value_head_dim, config.linear_key_head_dim), dtype=jnp.float32),
    )
    bucketed_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 1, 12, config.linear_conv_kernel_size), dtype=config.get_dtype()),
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_value_head_dim, config.linear_key_head_dim), dtype=jnp.float32),
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


def test_executor_table_hybrid_decode_matches_sliced_decode():
    config = _tiny_linear_attention_config()
    params = init_params(jax.random.PRNGKey(44), config)
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
    base_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 1, 12, config.linear_conv_kernel_size), dtype=config.get_dtype()),
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_value_head_dim, config.linear_key_head_dim), dtype=jnp.float32),
    )
    prompt_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )
    ref_prompt = executor.forward_step_token_ids_jit(
        prompt_batch,
        cache_storage=executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3),
        hybrid_state=base_hybrid,
    )
    table_prompt = executor.forward_step_token_ids_jit(
        prompt_batch,
        cache_storage=executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=3),
        hybrid_state=base_hybrid,
    )
    ref_prompt_tokens = (
        ref_prompt.activations[:, None]
        if ref_prompt.activations.ndim == 1
        else ref_prompt.activations
    )
    decode_batch = ScheduledBatch(
        tokens=ref_prompt_tokens,
        positions=jnp.array([[3]], dtype=jnp.int32),
        seq_ids=jnp.array([0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1], dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        block_tables=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        seq_lens=jnp.array([4], dtype=jnp.int32),
        seq_ids_host=(0,),
        query_lens_host=(1,),
        seq_lens_host=(4,),
    )

    ref = executor.forward_step_token_ids_jit(
        decode_batch,
        cache_storage=ref_prompt.cache_storage,
        hybrid_state=ref_prompt.hybrid_state,
    )
    conv_table = jnp.zeros(
        (2,) + table_prompt.hybrid_state.conv_state.shape[1:],
        dtype=table_prompt.hybrid_state.conv_state.dtype,
    ).at[1].set(table_prompt.hybrid_state.conv_state[0])
    recurrent_table = jnp.zeros(
        (2,) + table_prompt.hybrid_state.recurrent_state.shape[1:],
        dtype=table_prompt.hybrid_state.recurrent_state.dtype,
    ).at[1].set(table_prompt.hybrid_state.recurrent_state[0])
    actual = executor.forward_step_token_ids_table_jit(
        decode_batch,
        cache_storage=table_prompt.cache_storage,
        hybrid_state_table=HybridLayerState(conv_table, recurrent_table),
        hybrid_slot_ids=jnp.array([1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.array(actual.activations), np.array(ref.activations))
    np.testing.assert_allclose(
        np.array(actual.hybrid_state.conv_state[1:2]),
        np.array(ref.hybrid_state.conv_state),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(actual.hybrid_state.recurrent_state[1:2]),
        np.array(ref.hybrid_state.recurrent_state),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.array(actual.hybrid_state.conv_state[0]),
        np.zeros_like(np.array(actual.hybrid_state.conv_state[0])),
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
        recurrent_state=jnp.zeros((1, 1, 1, config.linear_value_head_dim, config.linear_key_head_dim), dtype=jnp.float32),
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
            return type(
                "Output",
                (),
                {
                    "activations": Ready(),
                    "cache_storage": kwargs["cache_storage"],
                },
            )()

    runner.executor = FakeExecutor()
    runner._sample_fn = lambda logits, temperatures: Ready()

    runner.warmup_compilation(max_prefill_len=3, max_batch=1)

    assert runner.executor.calls == [((1, 3), True), ((1, 1), False)]


def test_model_runner_warmup_uses_greedy_token_fastpath_without_mtp(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH", "1")
    config = _tiny_full_attention_config()
    config.jax_execution = "jit"
    config.prefill_buckets = (4,)
    config.batch_size_buckets = (2,)
    config.max_blocks_per_seq = 2
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.block_size = config.block_size
    runner.max_blocks_per_seq = 2
    runner.execution = "jit"
    runner.cache_storage = object()
    runner._warmup_compiled = False
    runner.mtp1_enabled = False
    runner._hybrid_state_table = None

    class Ready:
        def block_until_ready(self):
            return self

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def forward_step_jit(self, batch, **kwargs):
            self.calls.append(("logits", tuple(batch.tokens.shape), batch.is_prefill))
            return type(
                "Output",
                (),
                {
                    "activations": Ready(),
                    "cache_storage": kwargs["cache_storage"],
                },
            )()

        def forward_step_token_ids_jit(self, batch, **kwargs):
            self.calls.append(("token_ids", tuple(batch.tokens.shape), batch.is_prefill))
            return type(
                "Output",
                (),
                {
                    "activations": Ready(),
                    "cache_storage": kwargs["cache_storage"],
                },
            )()

    runner.executor = FakeExecutor()
    runner._sample_fn = lambda logits, temperatures: Ready()

    summary = runner.warmup_compilation(max_prefill_len=4, max_batch=2)

    assert runner.executor.calls == [
        ("token_ids", (2, 4), True),
        ("token_ids", (2, 1), False),
    ]
    assert summary["prefill_runs"][0]["route"] == "forward_step_token_ids_jit:prefill"
    assert summary["decode_runs"][0]["route"] == "forward_step_token_ids_jit:decode"


def test_model_runner_warmup_compiles_decode_block_table_buckets(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH", "1")
    config = _tiny_full_attention_config()
    config.jax_execution = "jit"
    config.prefill_buckets = (4,)
    config.batch_size_buckets = (2,)
    config.max_blocks_per_seq = 4
    config.decode_block_table_buckets = (2, 4)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.block_size = config.block_size
    runner.max_blocks_per_seq = 4
    runner.execution = "jit"
    runner.cache_storage = object()
    runner._warmup_compiled = False
    runner.mtp1_enabled = False
    runner._hybrid_state_table = None

    class Ready:
        def block_until_ready(self):
            return self

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def forward_step_token_ids_jit(self, batch, **kwargs):
            self.calls.append(
                (
                    tuple(batch.tokens.shape),
                    tuple(batch.block_tables.shape),
                    batch.is_prefill,
                )
            )
            return type(
                "Output",
                (),
                {
                    "activations": Ready(),
                    "cache_storage": kwargs["cache_storage"],
                },
            )()

    runner.executor = FakeExecutor()
    runner._sample_fn = lambda logits, temperatures: Ready()

    summary = runner.warmup_compilation(max_prefill_len=4, max_batch=2)

    assert runner.executor.calls == [
        ((2, 4), (2, 4), True),
        ((2, 1), (2, 2), False),
        ((2, 1), (2, 4), False),
    ]
    assert summary["decode_block_table_buckets"] == [2, 4]
    assert [run["block_tables_shape"] for run in summary["decode_runs"]] == [[2, 2], [2, 4]]


def test_model_runner_warmup_compiles_greedy_decode_burst(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS", raising=False)
    config = _tiny_full_attention_config()
    config.jax_execution = "jit"
    config.prefill_buckets = (4,)
    config.batch_size_buckets = (2,)
    config.max_blocks_per_seq = 2
    config.greedy_token_fastpath = True
    config.greedy_decode_burst_steps = 2
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.block_size = config.block_size
    runner.max_blocks_per_seq = 2
    runner.execution = "jit"
    runner.cache_storage = object()
    runner._warmup_compiled = False
    runner.mtp1_enabled = False
    runner._hybrid_state_table = None

    class Ready:
        def block_until_ready(self):
            return self

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def forward_step_token_ids_jit(self, batch, **kwargs):
            self.calls.append(("token_ids", tuple(batch.tokens.shape), batch.is_prefill))
            return type(
                "Output",
                (),
                {
                    "activations": Ready(),
                    "cache_storage": kwargs["cache_storage"],
                },
            )()

        def forward_greedy_decode_burst_jit(self, batch, **kwargs):
            self.calls.append(
                (
                    "burst",
                    tuple(batch.tokens.shape),
                    batch.is_prefill,
                    int(kwargs["decode_steps"]),
                )
            )
            return type(
                "Output",
                (),
                {
                    "activations": Ready(),
                    "cache_storage": kwargs["cache_storage"],
                    "hybrid_state": kwargs["hybrid_state"],
                },
            )()

    runner.executor = FakeExecutor()
    runner._sample_fn = lambda logits, temperatures: Ready()

    summary = runner.warmup_compilation(max_prefill_len=4, max_batch=2)

    assert runner.executor.calls == [
        ("token_ids", (2, 4), True),
        ("burst", (2, 1), False, 2),
        ("token_ids", (2, 1), False),
    ]
    assert summary["decode_runs"][0]["route"] == "forward_greedy_decode_burst_jit:decode"
    assert summary["decode_runs"][0]["decode_steps"] == 2
    assert summary["decode_runs"][1]["route"] == "forward_step_token_ids_jit:decode"
    assert summary["decode_runs"][1]["decode_steps"] == 1


def test_model_runner_batch_hybrid_state_reuses_fresh_full_slot_table():
    config = _tiny_linear_attention_config()
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner._hybrid_state_table = init_hybrid_state(config, batch_size=2, dtype=config.get_dtype())
    runner._max_hybrid_slots = 2
    runner._hybrid_slots = {}
    runner._free_hybrid_slots = [0, 1]
    batch = ScheduledBatch(
        tokens=jnp.zeros((2, 1), dtype=jnp.int32),
        positions=jnp.zeros((2, 1), dtype=jnp.int32),
        seq_ids=jnp.array([7, 8], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=2,
        block_tables=jnp.zeros((2, 2), dtype=jnp.int32),
        seq_lens=jnp.ones((2,), dtype=jnp.int32),
        seq_ids_host=(7, 8),
        query_lens_host=(1, 1),
    )

    state = runner._batch_hybrid_state(batch)

    assert state is runner._hybrid_state_table
    assert batch.hybrid_slot_ids_host == (0, 1)
    assert runner._hybrid_slots == {7: 0, 8: 1}
    assert runner._free_hybrid_slots == []


def test_compact_prefill_token_count_can_use_bucket_mode(monkeypatch):
    from nanovllm_jax.engine.model_executor import _static_prefill_token_count_for_batch

    batch = ScheduledBatch(
        tokens=jnp.zeros((2, 4), dtype=jnp.int32),
        positions=jnp.zeros((2, 4), dtype=jnp.int32),
        seq_ids=jnp.array([0, 1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32),
        is_prefill=True,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        block_tables=jnp.zeros((2, 2), dtype=jnp.int32),
        seq_lens=jnp.array([3, 2], dtype=jnp.int32),
        query_lens_host=(3, 2),
        seq_ids_host=(0, 1),
        seq_lens_host=(3, 2),
    )

    monkeypatch.setenv("NANO_VLLM_JAX_COMPACT_PREFILL_MLP", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE", "exact")
    assert _static_prefill_token_count_for_batch(batch) == 5

    monkeypatch.setenv("NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE", "bucket")
    assert _static_prefill_token_count_for_batch(batch) == 8
    assert _static_prefill_token_count_for_batch(batch, max_num_batched_tokens=6) == 6


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


def test_scheduler_postprocess_commits_boundary_between_multi_token_mtp_output():
    config = _tiny_full_attention_config()
    scheduler = Scheduler(config)
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4), seq_id=32)
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [0, 1]
    scheduler.running.append(seq)

    commit_calls: list[tuple[int, int]] = []
    original_commit = scheduler.block_manager.commit_processed_token

    try:
        def probe_commit(processed_seq: Sequence):
            commit_calls.append((int(processed_seq.num_tokens), int(processed_seq.last_token)))
            original_commit(processed_seq)

        scheduler.block_manager.commit_processed_token = probe_commit
        finished = scheduler.postprocess([seq], [[4, 5]])
    finally:
        scheduler.block_manager.commit_processed_token = original_commit

    assert finished == [False]
    assert commit_calls == [(4, 4)]
    assert seq.completion_token_ids == [4, 5]
    assert seq in scheduler.running


def test_scheduler_postprocess_no_boundary_commit_for_single_token_mtp_output():
    config = _tiny_full_attention_config()
    scheduler = Scheduler(config)
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4), seq_id=33)
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [0, 1]
    scheduler.running.append(seq)

    commit_calls: list[tuple[int, int]] = []
    original_commit = scheduler.block_manager.commit_processed_token

    try:
        def probe_commit(processed_seq: Sequence):
            commit_calls.append((int(processed_seq.num_tokens), int(processed_seq.last_token)))
            original_commit(processed_seq)

        scheduler.block_manager.commit_processed_token = probe_commit
        finished = scheduler.postprocess([seq], [4])
    finally:
        scheduler.block_manager.commit_processed_token = original_commit

    assert finished == [False]
    assert commit_calls == []
    assert seq.completion_token_ids == [4]
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


def test_model_runner_mtp1_acceptance_matches_two_step_main_decode_reference():
    config = _tiny_full_attention_config()

    class TwoStepExecutor:
        def __init__(self):
            self.calls: list[tuple[int, int]] = []

        def forward_step(self, batch, **kwargs):
            q_len = int(batch.query_lens[0])
            if q_len == 1:
                token = 4 if len(self.calls) == 0 else 5
                logits = jnp.full((1, 1, 8), -1e9, dtype=jnp.float32)
                logits = logits.at[0, 0, token].set(10.0)
                cache = f"decode-cache-{len(self.calls) + 1}"
                hybrid = f"decode-hybrid-{len(self.calls) + 1}"
                activations = logits
            else:
                cache = "decode-cache-2"
                hybrid = "decode-hybrid-2"
                activations = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
                token = 4
            self.calls.append((q_len, token))
            return type(
                "Output",
                (),
                {
                    "activations": activations,
                    "cache_storage": cache,
                    "hybrid_state": hybrid,
                },
            )()

    baseline_runner = ModelRunner.__new__(ModelRunner)
    baseline_runner.config = config
    baseline_runner.block_size = config.block_size
    baseline_runner.execution = "eager"
    baseline_runner.cache_storage = "base-cache"
    baseline_runner._sample_fn = lambda logits, temperatures: jnp.argmax(logits, axis=-1)
    baseline_hybrid_states: list[str] = []
    baseline_runner._batch_hybrid_state = lambda batch: "baseline"
    baseline_runner._store_batch_hybrid_state = lambda batch, state: baseline_hybrid_states.append(state)
    baseline_runner._refresh_kv_snapshot = lambda batch, state: None

    baseline_executor = TwoStepExecutor()
    baseline_runner.executor = baseline_executor
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=5), seq_id=91)
    seq.block_table = [0, 1]
    main_batch_1 = _scheduled_batch(tokens=[3], positions=[2], block_tables=[0, 1], seq_lens=3, is_prefill=False)
    main_batch_2 = _scheduled_batch(tokens=[4], positions=[3], block_tables=[0, 1], seq_lens=4, is_prefill=False)

    first = baseline_runner._run_main_and_sample([seq], main_batch_1, seed_mtp1=False)
    assert first == [4]
    seq.append_token(4)
    second = baseline_runner._run_main_and_sample([seq], main_batch_2, seed_mtp1=False)
    assert second == [5]

    assert baseline_runner.cache_storage == "decode-cache-2"
    assert baseline_hybrid_states == ["decode-hybrid-1", "decode-hybrid-2"]
    assert baseline_runner.executor.calls == [(1, 4), (1, 5)]

    accepted_runner = ModelRunner.__new__(ModelRunner)
    accepted_runner.config = config
    accepted_runner.block_size = config.block_size
    accepted_runner.execution = "eager"
    accepted_runner.params = object()
    accepted_runner.cache_storage = "base-cache"
    accepted_runner.mtp_enabled = True
    accepted_runner.mtp1_enabled = True
    accepted_runner.mtp_hidden_source = "pre_norm"
    accepted_runner._mtp1_drafts = {91: 4}
    accepted_runner.reset_speculative_stats()
    accepted_runner._sample_fn = lambda logits, temperatures: jnp.argmax(logits, axis=-1)
    accepted_runner._seed_mtp1_draft = lambda seq, hidden, confirmed_token_id, position: None
    accepted_runner._greedy_tokens_from_hidden = lambda hidden: jnp.array([[4, 5]], dtype=jnp.int32)
    accepted_runner._logits_from_hidden = lambda hidden: hidden
    accepted_hybrid_states: list[str] = []
    accepted_runner._batch_hybrid_state = lambda batch: "baseline"
    accepted_runner._store_batch_hybrid_state = lambda batch, state: accepted_hybrid_states.append(state)
    accepted_runner._refresh_kv_snapshot = lambda batch, state: None

    accepted_executor = TwoStepExecutor()
    accepted_runner.executor = accepted_executor
    accepted_seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=5), seq_id=91)
    accepted_seq.block_table = [0, 1]
    mtp_batch = _scheduled_batch(tokens=[3], positions=[2], block_tables=[0, 1], seq_lens=3, is_prefill=False)
    accepted_tokens = accepted_runner._run_mtp1([accepted_seq], mtp_batch)

    assert accepted_tokens == [[4, 5]]
    assert baseline_runner.cache_storage == accepted_runner.cache_storage
    assert baseline_runner.executor.calls == [(1, 4), (1, 5)]
    assert accepted_hybrid_states == ["decode-hybrid-2"]
    assert accepted_executor.calls[0][0] == 2


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


def test_model_runner_mtp1_rejection_matches_single_step_main_decode_reference():
    config = _tiny_full_attention_config()

    class RejectingExecutor:
        def __init__(self):
            self.calls: list[int] = []

        def forward_step(self, batch, **kwargs):
            q_len = int(batch.query_lens[0])
            if q_len == 1:
                token = 6
                logits = jnp.full((1, 1, 8), -1e9, dtype=jnp.float32)
                logits = logits.at[0, 0, token].set(10.0)
                cache = "decode-cache"
                hybrid = "decode-hybrid"
                if kwargs.get("return_hidden_with_logits"):
                    hidden = jnp.zeros((1, 1, config.hidden_size), dtype=jnp.float32)
                    activations = (hidden, logits)
                else:
                    activations = logits
            else:
                cache = "verify-cache"
                hybrid = "verify-hybrid"
                activations = jnp.zeros((1, 2, config.hidden_size), dtype=jnp.float32)
                token = 4
            self.calls.append(q_len)
            return type(
                "Output",
                (),
                {
                    "activations": activations,
                    "cache_storage": cache,
                    "hybrid_state": hybrid,
                },
            )()

    baseline_runner = ModelRunner.__new__(ModelRunner)
    baseline_runner.config = config
    baseline_runner.block_size = config.block_size
    baseline_runner.execution = "eager"
    baseline_runner.cache_storage = "base-cache"
    baseline_runner._sample_fn = lambda logits, temperatures: jnp.argmax(logits, axis=-1)
    baseline_hybrid_states: list[str] = []
    baseline_runner._batch_hybrid_state = lambda batch: "baseline"
    baseline_runner._store_batch_hybrid_state = lambda batch, state: baseline_hybrid_states.append(state)
    baseline_runner._refresh_kv_snapshot = lambda batch, state: None
    baseline_runner.executor = RejectingExecutor()

    baseline_seq = Sequence([1, 2, 3], SamplingParams(temperature=1.0, max_tokens=5), seq_id=92)
    baseline_seq.block_table = [0, 1]
    main_batch = _scheduled_batch(tokens=[3], positions=[2], block_tables=[0, 1], seq_lens=3, is_prefill=False)
    baseline_step = baseline_runner._run_main_and_sample([baseline_seq], main_batch, seed_mtp1=False)
    assert baseline_step == [6]
    assert baseline_runner.cache_storage == "decode-cache"
    assert baseline_hybrid_states == ["decode-hybrid"]

    rejecting_runner = ModelRunner.__new__(ModelRunner)
    rejecting_runner.config = config
    rejecting_runner.block_size = config.block_size
    rejecting_runner.execution = "eager"
    rejecting_runner.cache_storage = "base-cache"
    rejecting_runner.mtp_enabled = True
    rejecting_runner.mtp1_enabled = True
    rejecting_runner.mtp_hidden_source = "pre_norm"
    rejecting_runner._mtp1_drafts = {92: 4}
    rejecting_runner._sample_fn = lambda logits, temperatures: jnp.argmax(logits, axis=-1)
    rejecting_runner._logits_from_hidden = lambda hidden: hidden
    rejecting_runner._greedy_tokens_from_hidden = lambda hidden: jnp.array([[7, 0]], dtype=jnp.int32)
    rejecting_runner._seed_mtp1_draft = lambda seq, hidden, confirmed_token_id, position: None
    rejecting_runner._batch_hybrid_state = lambda batch: "baseline"
    rejecting_hybrid_states: list[str] = []
    rejecting_runner._store_batch_hybrid_state = lambda batch, state: rejecting_hybrid_states.append(state)
    rejecting_runner._refresh_kv_snapshot = lambda batch, state: None
    rejecting_runner.reset_speculative_stats()

    rejecting_executor = RejectingExecutor()
    rejecting_runner.executor = rejecting_executor
    rejecting_seq = Sequence([1, 2, 3], SamplingParams(temperature=1.0, max_tokens=5), seq_id=92)
    rejecting_seq.block_table = [0, 1]
    rejecting_batch = _scheduled_batch(tokens=[3], positions=[2], block_tables=[0, 1], seq_lens=3, is_prefill=False)
    rejecting_tokens = rejecting_runner._run_mtp1([rejecting_seq], rejecting_batch)

    assert rejecting_tokens == [6]
    assert rejecting_runner.cache_storage == baseline_runner.cache_storage
    assert rejecting_hybrid_states == ["decode-hybrid"]
    assert rejecting_runner.executor.calls == [2, 1]


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


def test_executor_jit_decode_derives_position_from_seq_len_for_static_metadata():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(22), config)
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
    correct_cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)
    stale_cache = executor.backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)
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
    correct_prompt = executor.forward_step_jit(
        prompt_batch,
        cache_storage=correct_cache,
        hybrid_state=empty_hybrid,
    )
    stale_prompt = executor.forward_step_jit(
        prompt_batch,
        cache_storage=stale_cache,
        hybrid_state=empty_hybrid,
    )
    correct_position_batch = _scheduled_batch(
        tokens=[4],
        positions=[3],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=False,
    )
    stale_position_batch = _scheduled_batch(
        tokens=[4],
        positions=[0],
        block_tables=[0, 1],
        seq_lens=4,
        is_prefill=False,
    )

    correct = executor.forward_step_jit(
        correct_position_batch,
        cache_storage=correct_prompt.cache_storage,
        hybrid_state=correct_prompt.hybrid_state,
    )
    stale = executor.forward_step_jit(
        stale_position_batch,
        cache_storage=stale_prompt.cache_storage,
        hybrid_state=stale_prompt.hybrid_state,
    )

    np.testing.assert_allclose(
        np.array(stale.activations),
        np.array(correct.activations),
        rtol=1e-5,
        atol=1e-5,
    )


def test_executor_greedy_decode_burst_matches_iterative_token_path():
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(22), config)
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
    empty_hybrid = HybridLayerState(
        conv_state=jnp.zeros((1, 0, 1, 1)),
        recurrent_state=jnp.zeros((1, 0, 1, 1, 1)),
    )
    prompt_batch = _scheduled_batch(
        tokens=[1, 2, 3],
        positions=[0, 1, 2],
        block_tables=[0, 1, 2],
        seq_lens=3,
        is_prefill=True,
    )

    ref_prompt = executor.forward_step_token_ids_jit(
        prompt_batch,
        cache_storage=executor.backend.allocate_kv_cache(
            spec,
            max_seqs=1,
            max_blocks_per_seq=3,
        ),
        hybrid_state=empty_hybrid,
    )
    burst_prompt = executor.forward_step_token_ids_jit(
        prompt_batch,
        cache_storage=executor.backend.allocate_kv_cache(
            spec,
            max_seqs=1,
            max_blocks_per_seq=3,
        ),
        hybrid_state=empty_hybrid,
    )
    table_burst_prompt = executor.forward_step_token_ids_jit(
        prompt_batch,
        cache_storage=executor.backend.allocate_kv_cache(
            spec,
            max_seqs=1,
            max_blocks_per_seq=3,
        ),
        hybrid_state=empty_hybrid,
    )
    def decode_batch(tokens, positions, seq_lens, *, decode_steps=1):
        return ScheduledBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=jnp.array([0], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 1], dtype=jnp.int32),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=1,
            block_tables=jnp.array([[0, 1, 2]], dtype=jnp.int32),
            seq_lens=seq_lens,
            seq_ids_host=(0,),
            query_lens_host=(1,),
            seq_lens_host=(int(np.asarray(seq_lens)[0]),),
            decode_step_count_host=decode_steps,
        )

    ref_cache = ref_prompt.cache_storage
    ref_hybrid = ref_prompt.hybrid_state
    ref_tokens_in = (
        ref_prompt.activations[:, None]
        if ref_prompt.activations.ndim == 1
        else ref_prompt.activations
    )
    ref_positions = jnp.array([[3]], dtype=jnp.int32)
    ref_seq_lens = jnp.array([4], dtype=jnp.int32)
    ref_generated = []
    for _ in range(3):
        ref_out = executor.forward_step_token_ids_jit(
            decode_batch(ref_tokens_in, ref_positions, ref_seq_lens),
            cache_storage=ref_cache,
            hybrid_state=ref_hybrid,
        )
        ref_generated.append(
            ref_out.activations[:, 0]
            if ref_out.activations.ndim == 2
            else ref_out.activations
        )
        ref_tokens_in = (
            ref_out.activations[:, None]
            if ref_out.activations.ndim == 1
            else ref_out.activations
        )
        ref_positions = ref_positions + 1
        ref_seq_lens = ref_seq_lens + 1
        ref_cache = ref_out.cache_storage
        ref_hybrid = ref_out.hybrid_state
    ref_generated = jnp.stack(ref_generated, axis=1)

    burst_out = executor.forward_greedy_decode_burst_jit(
        decode_batch(
            burst_prompt.activations[:, None]
            if burst_prompt.activations.ndim == 1
            else burst_prompt.activations,
            jnp.array([[3]], dtype=jnp.int32),
            jnp.array([4], dtype=jnp.int32),
            decode_steps=3,
        ),
        cache_storage=burst_prompt.cache_storage,
        hybrid_state=burst_prompt.hybrid_state,
        decode_steps=3,
    )
    table_burst_out = executor.forward_greedy_decode_burst_table_jit(
        decode_batch(
            table_burst_prompt.activations[:, None]
            if table_burst_prompt.activations.ndim == 1
            else table_burst_prompt.activations,
            jnp.array([[3]], dtype=jnp.int32),
            jnp.array([4], dtype=jnp.int32),
            decode_steps=3,
        ),
        cache_storage=table_burst_prompt.cache_storage,
        hybrid_state_table=table_burst_prompt.hybrid_state,
        hybrid_slot_ids=jnp.array([0], dtype=jnp.int32),
        decode_steps=3,
    )

    np.testing.assert_array_equal(
        np.asarray(burst_out.activations),
        np.asarray(ref_generated),
    )
    np.testing.assert_array_equal(
        np.asarray(table_burst_out.activations),
        np.asarray(ref_generated),
    )
    np.testing.assert_allclose(
        np.asarray(burst_out.cache_storage.k_cache),
        np.asarray(ref_cache.k_cache),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(burst_out.cache_storage.v_cache),
        np.asarray(ref_cache.v_cache),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(table_burst_out.cache_storage.k_cache),
        np.asarray(ref_cache.k_cache),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(table_burst_out.cache_storage.v_cache),
        np.asarray(ref_cache.v_cache),
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

    assert int(fused.target_token[0]) == int(expected_tokens[0, 0])
    assert int(fused.bonus_token[0]) == int(expected_tokens[0, 1])
    assert int(fused.next_draft_token[0]) == int(expected_next_draft)
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


def test_no_prefix_cache_allocation_keeps_repeated_prompts_on_unique_blocks():
    block_manager = BlockManager(num_blocks=4, block_size=2)
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0), seq_id=1)
    seq_b = Sequence([1, 2], SamplingParams(temperature=0.0), seq_id=2)

    block_manager.allocate(seq_a, use_prefix_cache=False)
    block_manager.allocate(seq_b, use_prefix_cache=False)

    assert seq_a.block_table != seq_b.block_table
    assert block_manager.blocks[seq_a.block_table[0]].ref_count == 1
    assert block_manager.blocks[seq_b.block_table[0]].ref_count == 1
    assert block_manager.blocks[seq_a.block_table[0]].hash != -1
    assert block_manager.blocks[seq_b.block_table[0]].hash != -1
    assert block_manager.hash_to_block_id == {}


def test_no_prefix_cache_allocation_requires_free_blocks_for_repeated_prompts():
    block_manager = BlockManager(num_blocks=1, block_size=2)
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0), seq_id=1)
    seq_b = Sequence([1, 2], SamplingParams(temperature=0.0), seq_id=2)

    block_manager.allocate(seq_a, use_prefix_cache=False)

    assert len(block_manager.free_block_ids) == 0
    assert not block_manager.can_allocate(seq_b, use_prefix_cache=False)


def test_block_manager_hashes_completed_prompt_tail_before_next_block():
    block_manager = BlockManager(num_blocks=3, block_size=2)
    seq = Sequence([1], SamplingParams(temperature=0.0), seq_id=1)
    block_manager.allocate(seq, use_prefix_cache=False)
    assert block_manager.blocks[seq.block_table[-1]].hash == -1

    seq.append_token(2)
    block_manager.may_append_slots(seq, 1)
    completed_block = seq.block_table[-1]
    assert block_manager.blocks[completed_block].hash != -1

    seq.append_token(3)
    block_manager.may_append_slots(seq, 1)
    assert len(seq.block_table) == 2
    assert seq.block_table[0] == completed_block


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


def test_server_generate_allows_prompt_longer_than_single_prefill_bucket(monkeypatch):
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
            assert inputs == ["too long"]
            assert sampling_params.max_tokens == 1
            assert use_tqdm is False
            return [{"text": "ok", "token_ids": [10, 11]}]

    monkeypatch.setattr(server, "engine", FakeEngine())

    response = server.app.test_client().post(
        "/v1/generate",
        json={"prompt": "too long", "max_tokens": 1},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["text"] == "ok"
    assert payload["usage"]["prompt_tokens"] == 8
    assert payload["usage"]["completion_tokens"] == 2


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
