"""Focused tests for deferred greedy token materialization."""

import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.llm_engine import (
    LLMEngine,
    _trace_token_prefetch_enabled,
)
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.sequence import SequenceStatus
from nanovllm_jax.engine.sequence import DeviceTokenRef, SamplingParams, Sequence


class _AsyncScalar:
    def __init__(self, value: int, events: list[str] | None = None):
        self.value = int(value)
        self.events = events if events is not None else []
        self.prefetch_count = 0
        self.materialize_count = 0

    def copy_to_host_async(self):
        self.prefetch_count += 1
        self.events.append(f"prefetch-{self.value}")

    def __array__(self, dtype=None, copy=None):
        self.materialize_count += 1
        self.events.append(f"materialize-{self.value}")
        return np.asarray(self.value, dtype=dtype)


def test_trace_token_prefetch_env_can_disable_config_default(monkeypatch):
    config = Qwen3_5Config(trace_token_prefetch=True)

    monkeypatch.setenv("NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH", "0")

    assert _trace_token_prefetch_enabled(config) is False


def test_sequence_materializes_deferred_device_tokens():
    Sequence.block_size = 16
    seq = Sequence([11, 22], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))

    seq.append_token_device(jnp.asarray(33, dtype=jnp.int32))

    assert seq.num_completion_tokens == 1
    assert seq.has_unmaterialized_device_tokens
    assert seq.block_has_unmaterialized_device_tokens(0)
    assert seq.completion_token_ids == [33]
    assert not seq.has_unmaterialized_device_tokens
    assert seq.last_token == 33


def test_sequence_materializes_deferred_device_tokens_for_multiple_sequences():
    seq_a = Sequence([11], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    seq_b = Sequence([22], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))

    seq_a.append_token_device(jnp.asarray(33, dtype=jnp.int32))
    seq_b.append_token_device(jnp.asarray(44, dtype=jnp.int32))

    Sequence.materialize_device_tokens_for_sequences([seq_a, seq_b])

    assert seq_a.completion_token_ids == [33]
    assert seq_b.completion_token_ids == [44]
    assert not seq_a.has_unmaterialized_device_tokens
    assert not seq_b.has_unmaterialized_device_tokens
    assert seq_a.last_token == 33
    assert seq_b.last_token == 44


def test_sequence_materializes_deferred_device_token_refs_for_multiple_sequences():
    seq_a = Sequence([11], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    seq_b = Sequence([22], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    token_vector = jnp.asarray([33, 44], dtype=jnp.int32)

    seq_a.append_token_device(DeviceTokenRef(tokens=token_vector, row=0))
    seq_b.append_token_device(DeviceTokenRef(tokens=token_vector, row=1))

    Sequence.materialize_device_tokens_for_sequences([seq_a, seq_b])

    assert seq_a.completion_token_ids == [33]
    assert seq_b.completion_token_ids == [44]
    assert not seq_a.has_unmaterialized_device_tokens
    assert not seq_b.has_unmaterialized_device_tokens


def test_sequence_materializes_device_token_snapshot_without_clearing_newer_tokens():
    Sequence.block_size = 16
    seq = Sequence([11], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True))

    seq.append_token_device(jnp.asarray(33, dtype=jnp.int32))
    snapshot = Sequence.snapshot_device_token_slots_for_sequences([seq])
    Sequence.prefetch_device_token_slots(snapshot)
    seq.append_token_device(jnp.asarray(44, dtype=jnp.int32))

    Sequence.materialize_device_token_slots(snapshot)

    assert seq.token_ids == [11, 33, 0]
    assert seq.has_unmaterialized_device_tokens
    assert seq.block_has_unmaterialized_device_tokens(0)
    assert seq.last_token == 0
    assert seq.last_token_device is not None

    Sequence.materialize_device_tokens_for_sequences([seq])

    assert seq.completion_token_ids == [33, 44]
    assert not seq.has_unmaterialized_device_tokens
    assert seq.last_token == 44


def test_sequence_prefetches_snapshot_before_later_materialization_without_clearing_newer_tokens():
    events: list[str] = []
    first_token = _AsyncScalar(33, events)
    second_token = _AsyncScalar(44, events)
    seq = Sequence([11], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True))

    seq.append_token_device(first_token)
    snapshot = Sequence.snapshot_device_token_slots_for_sequences([seq])
    prefetched = Sequence.prefetch_device_token_slots(snapshot)
    seq.append_token_device(second_token)

    assert prefetched == snapshot
    assert first_token.prefetch_count == 1
    assert second_token.prefetch_count == 0
    assert seq.materialized_completion_token_ids() == []

    Sequence.materialize_device_token_slots(snapshot)

    assert seq.materialized_completion_token_ids() == [33]
    assert seq.token_ids == [11, 33, 0]
    assert seq.has_unmaterialized_device_tokens
    assert seq.last_token_device is second_token
    assert events == ["prefetch-33", "materialize-33"]


def _decode_batch(
    seq_ids: tuple[int, ...],
    tokens: list[int],
    *,
    seq_lens: list[int] | None = None,
) -> ScheduledBatch:
    query_lens = [1 if seq_id >= 0 else 0 for seq_id in seq_ids]
    if seq_lens is None:
        seq_lens = query_lens
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    return ScheduledBatch(
        tokens=jnp.asarray(tokens, dtype=jnp.int32)[:, None],
        positions=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        seq_ids=jnp.asarray(seq_ids, dtype=jnp.int32),
        query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=sum(query_lens),
        block_tables=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
        seq_ids_host=seq_ids,
        query_lens_host=tuple(query_lens),
        seq_lens_host=tuple(seq_lens),
    )


def _prefill_batch(
    seq_ids: tuple[int, ...],
    tokens: list[int],
    *,
    final_flags: tuple[bool, ...],
) -> ScheduledBatch:
    query_lens = [1 if seq_id >= 0 else 0 for seq_id in seq_ids]
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    return ScheduledBatch(
        tokens=jnp.asarray(tokens, dtype=jnp.int32)[:, None],
        positions=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        seq_ids=jnp.asarray(seq_ids, dtype=jnp.int32),
        query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
        is_prefill=True,
        num_prefill_tokens=sum(query_lens),
        num_decode_tokens=0,
        block_tables=jnp.zeros((len(seq_ids), 1), dtype=jnp.int32),
        seq_lens=jnp.asarray(query_lens, dtype=jnp.int32),
        prefill_is_final=final_flags,
        seq_ids_host=seq_ids,
        query_lens_host=tuple(query_lens),
        seq_lens_host=tuple(query_lens),
    )


def test_model_runner_device_token_carry_uses_whole_vector_when_seq_ids_match(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    previous_batch = _decode_batch((7, 8), [10, 20])

    runner._record_device_token_carry(
        previous_batch,
        jnp.asarray([70, 80], dtype=jnp.int32),
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
    )
    next_batch = _decode_batch((7, 8), [200, 100])

    carried_batch = runner._maybe_apply_device_token_carry(next_batch)

    np.testing.assert_array_equal(np.asarray(carried_batch.tokens[:, 0]), np.asarray([70, 80]))


def test_model_runner_device_token_carry_records_full_static_decode_rows(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    batch = _decode_batch((7, 8, -1, -1), [0, 0, 0, 0], seq_lens=[4, 5, 0, 0])
    token_matrix = jnp.asarray([[70], [80], [0], [0]], dtype=jnp.int32)

    runner._record_device_token_carry(
        batch,
        token_matrix,
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
    )

    assert runner._device_token_carry_seq_ids == (7, 8, -1, -1)
    assert runner._device_token_carry_tokens is token_matrix
    assert set(runner._device_token_carry_by_seq_id) == {7, 8}

    carried_batch = runner._maybe_apply_device_token_carry(batch)

    assert carried_batch.tokens is token_matrix


def test_model_runner_device_token_carry_updates_resident_last_tokens(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    runner._resident_last_tokens = jnp.zeros((4,), dtype=jnp.int32)
    runner._hybrid_slots = {7: 2, 8: 1}
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    batch = _decode_batch((7, 8, -1, -1), [0, 0, 0, 0], seq_lens=[4, 5, 0, 0])
    batch.hybrid_slot_ids_host = (2, 1, -1, -1)

    runner._record_device_token_carry(
        batch,
        jnp.asarray([[70], [80], [0], [0]], dtype=jnp.int32),
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
    )

    np.testing.assert_array_equal(
        np.asarray(runner._resident_last_tokens),
        np.asarray([0, 80, 70, 0], dtype=np.int32),
    )


def test_model_runner_device_token_carry_clears_resident_stale_when_tokens_already_current(
    monkeypatch,
):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    runner._resident_last_tokens_stale_seq_ids = {7, 8, 99}
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    batch = _decode_batch((7, 8), [0, 0], seq_lens=[4, 5])

    runner._record_device_token_carry(
        batch,
        jnp.asarray([70, 80], dtype=jnp.int32),
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
        update_resident_tokens=False,
        resident_tokens_already_current=True,
    )

    assert runner._resident_last_tokens_stale_seq_ids == {99}
    assert set(runner._device_token_carry_by_seq_id) == {7, 8}


def test_model_runner_device_token_carry_marks_resident_stale_when_not_updated(
    monkeypatch,
):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    runner._resident_last_tokens_stale_seq_ids = {99}
    seq = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    batch = _decode_batch((7,), [0], seq_lens=[4])

    runner._record_device_token_carry(
        batch,
        jnp.asarray([70], dtype=jnp.int32),
        active_rows=[0],
        prefill_final_flags=[True],
        seqs=[seq],
        update_resident_tokens=False,
    )

    assert runner._resident_last_tokens_stale_seq_ids == {7, 99}


def test_model_runner_device_token_carry_follows_seq_ids_after_row_order_change(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    previous_batch = _decode_batch((7, 8, -1), [10, 20, 0])

    runner._record_device_token_carry(
        previous_batch,
        jnp.asarray([70, 80, 0], dtype=jnp.int32),
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
    )
    next_batch = _decode_batch((8, 7, -1), [200, 100, 0])

    carried_batch = runner._maybe_apply_device_token_carry(next_batch)

    np.testing.assert_array_equal(np.asarray(carried_batch.tokens[:, 0]), np.asarray([80, 70, 0]))


def test_model_runner_device_token_carry_survives_nonfinal_prefill_chunk(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)

    runner._record_device_token_carry(
        _prefill_batch((7,), [10], final_flags=(True,)),
        jnp.asarray([70], dtype=jnp.int32),
        active_rows=[0],
        prefill_final_flags=[True],
        seqs=[seq_a],
    )
    runner._record_device_token_carry(
        _prefill_batch((8,), [20], final_flags=(False,)),
        jnp.asarray([0], dtype=jnp.int32),
        active_rows=[0],
        prefill_final_flags=[False],
        seqs=[seq_b],
    )
    runner._record_device_token_carry(
        _prefill_batch((8,), [21], final_flags=(True,)),
        jnp.asarray([80], dtype=jnp.int32),
        active_rows=[0],
        prefill_final_flags=[True],
        seqs=[seq_b],
    )
    next_batch = _decode_batch((7, 8), [0, 0], seq_lens=[1, 1])
    next_batch.uses_static_decode_metadata = True

    carried_batch = runner._maybe_apply_device_token_carry(next_batch)

    np.testing.assert_array_equal(np.asarray(carried_batch.tokens[:, 0]), np.asarray([70, 80]))


def test_model_runner_release_preserves_carry_for_still_running_rows():
    runner = ModelRunner.__new__(ModelRunner)
    runner.hybrid_states = {}
    runner._hybrid_slots = {}
    runner._free_hybrid_slots = []
    runner._mtp1_drafts = {}
    token_vector = jnp.asarray([70, 80], dtype=jnp.int32)
    runner._device_token_carry_seq_ids = (7, 8)
    runner._device_token_carry_tokens = token_vector
    runner._device_token_carry_by_seq_id = {
        7: DeviceTokenRef(tokens=token_vector, row=0),
        8: DeviceTokenRef(tokens=token_vector, row=1),
    }

    runner.release([7])

    assert runner._device_token_carry_seq_ids is None
    assert runner._device_token_carry_tokens is None
    assert set(runner._device_token_carry_by_seq_id) == {8}
    assert runner._device_token_carry_by_seq_id[8].row == 1


def test_scheduler_static_decode_metadata_reuses_fixed_device_arrays(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_METADATA", "1")
    token_vector = jnp.asarray([70, 80], dtype=jnp.int32)
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            batch_size_buckets=(2,),
            max_blocks_per_seq=2,
            num_kvcache_blocks=4,
            jax_execution="jit",
            device_token_carry=True,
            static_decode_metadata=True,
        )
    )
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=7)
    seq_b = Sequence([3, 4], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=8)
    seq_a.block_table = [0]
    seq_b.block_table = [1]
    seq_a.last_token = 0
    seq_b.last_token = 0
    seq_a.last_token_device = DeviceTokenRef(tokens=token_vector, row=0)
    seq_b.last_token_device = DeviceTokenRef(tokens=token_vector, row=1)

    first = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)
    seq_a.append_token_device(DeviceTokenRef(tokens=token_vector, row=0))
    seq_b.append_token_device(DeviceTokenRef(tokens=token_vector, row=1))
    second = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)

    assert first.uses_static_decode_metadata
    assert second.uses_static_decode_metadata
    assert second.tokens is first.tokens
    assert second.positions is first.positions
    assert second.seq_ids is first.seq_ids
    assert second.query_start_loc is first.query_start_loc
    assert second.block_tables is first.block_tables
    np.testing.assert_array_equal(np.asarray(first.seq_lens), np.asarray([2, 2]))
    np.testing.assert_array_equal(np.asarray(second.seq_lens), np.asarray([3, 3]))
    assert second.seq_lens_host == (3, 3)


def test_scheduler_decode_block_table_buckets_select_smallest_fit():
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            batch_size_buckets=(2,),
            max_blocks_per_seq=8,
            decode_block_table_buckets=(2, 4, 8),
            num_kvcache_blocks=16,
            jax_execution="jit",
        )
    )
    seq_a = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=2), seq_id=7)
    seq_b = Sequence([4, 5, 6], SamplingParams(temperature=0.0, max_tokens=2), seq_id=8)
    seq_a.block_table = [0, 2, 4]
    seq_b.block_table = [1, 3]

    batch = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)

    assert batch.block_tables.shape == (2, 4)
    np.testing.assert_array_equal(np.asarray(batch.block_tables[0]), np.asarray([0, 2, 4, 0]))
    np.testing.assert_array_equal(np.asarray(batch.block_tables[1]), np.asarray([1, 3, 0, 0]))


def test_scheduler_mixed_prefill_decode_backfills_underfull_decode_batch():
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=4,
            max_num_resident_seqs=6,
            max_num_batched_tokens=16,
            prefill_buckets=(16,),
            prefill_token_buckets=(16,),
            batch_size_buckets=(4,),
            max_blocks_per_seq=4,
            num_kvcache_blocks=16,
            jax_execution="jit",
        )
    )
    decode_a = Sequence(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        seq_id=7,
    )
    decode_b = Sequence(
        [3, 4],
        SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        seq_id=8,
    )
    for seq, token in ((decode_a, 97), (decode_b, 98)):
        scheduler.block_manager.allocate(seq, use_prefix_cache=False)
        seq.num_cached_tokens = seq.num_prompt_tokens
        seq.append_token(token)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

    waiting_a = Sequence(
        [10, 11, 12],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        seq_id=9,
    )
    waiting_b = Sequence(
        [20, 21, 22],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        seq_id=10,
    )
    scheduler.add(waiting_a)
    scheduler.add(waiting_b)

    scheduled = scheduler._schedule_mixed_prefill_decode()
    assert scheduled is not None
    seqs, batch = scheduled

    assert [seq.seq_id for seq in seqs] == [7, 8, 9, 10]
    assert batch.is_prefill
    assert batch.packed_prefill
    assert batch.mixed_prefill_decode
    assert batch.query_lens_host == (1, 1, 3, 3)
    assert batch.seq_lens_host == (3, 3, 3, 3)
    assert batch.prefill_final_flags[:4] == [True, True, True, True]
    assert not scheduler.waiting
    assert len(scheduler.running) == 4
    np.testing.assert_array_equal(
        np.asarray(batch.tokens)[0, :8],
        np.asarray([97, 98, 10, 11, 12, 20, 21, 22]),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.token_row_ids)[0, :8],
        np.asarray([0, 1, 2, 2, 2, 3, 3, 3]),
    )


def test_model_runner_applies_device_token_carry_to_mixed_packed_decode_rows(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner.device_token_carry = True
    runner.config = Qwen3_5Config(device_token_carry=True)
    token_vector = jnp.asarray([70], dtype=jnp.int32)
    runner._device_token_carry_by_seq_id = {7: DeviceTokenRef(tokens=token_vector, row=0)}

    batch = ScheduledBatch(
        tokens=jnp.asarray([[0, 11, 12]], dtype=jnp.int32),
        positions=jnp.asarray([[2, 0, 1]], dtype=jnp.int32),
        seq_ids=jnp.asarray([7, 8], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 1, 3], dtype=jnp.int32),
        is_prefill=True,
        num_prefill_tokens=3,
        num_decode_tokens=0,
        block_tables=jnp.zeros((2, 2), dtype=jnp.int32),
        seq_lens=jnp.asarray([3, 2], dtype=jnp.int32),
        prefill_is_final=(True, False),
        seq_ids_host=(7, 8),
        query_lens_host=(1, 2),
        seq_lens_host=(3, 2),
        packed_prefill=True,
        token_row_ids=jnp.asarray([[0, 1, 1]], dtype=jnp.int32),
        mixed_prefill_decode=True,
    )

    carried = runner._maybe_apply_device_token_carry(batch)

    np.testing.assert_array_equal(np.asarray(carried.tokens), np.asarray([[70, 11, 12]]))


def test_scheduler_resident_capacity_can_exceed_execution_batch():
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            max_num_resident_seqs=4,
            max_num_batched_tokens=4,
            batch_size_buckets=(2,),
            max_blocks_per_seq=2,
            num_kvcache_blocks=8,
            jax_execution="jit",
        )
    )
    seqs = [
        Sequence(
            [1 + idx, 11 + idx],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            seq_id=idx,
        )
        for idx in range(4)
    ]
    for seq in seqs:
        scheduler.add(seq)

    first_prefill_seqs, first_prefill = scheduler.schedule()
    assert [seq.seq_id for seq in first_prefill_seqs] == [0, 1]
    assert first_prefill.seq_ids_host == (0, 1)
    for seq in first_prefill_seqs:
        seq.num_cached_tokens = seq.num_prompt_tokens

    first_decode_seqs, first_decode = scheduler.schedule()
    assert [seq.seq_id for seq in first_decode_seqs] == [0, 1]
    assert first_decode.tokens.shape[0] == 2
    assert len(scheduler.waiting) == 2

    scheduler.postprocess(first_decode_seqs, [101, 102])
    assert len(scheduler.running) == 0

    second_prefill_seqs, second_prefill = scheduler.schedule()
    assert [seq.seq_id for seq in second_prefill_seqs] == [2, 3]
    assert second_prefill.seq_ids_host == (2, 3)
    for seq in second_prefill_seqs:
        seq.num_cached_tokens = seq.num_prompt_tokens

    assert not scheduler.waiting
    assert len(scheduler.running) == 2

    decode_seqs, decode_batch = scheduler.schedule()
    assert len(decode_seqs) == 2
    assert decode_batch.tokens.shape[0] == 2
    assert len(scheduler.running) == 2


def test_scheduler_static_decode_metadata_can_reuse_seq_lens_placeholder(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_METADATA", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY", "1")
    token_vector = jnp.asarray([70, 80], dtype=jnp.int32)
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            batch_size_buckets=(2,),
            max_blocks_per_seq=2,
            num_kvcache_blocks=4,
            jax_execution="jit",
            device_token_carry=True,
            static_decode_metadata=True,
            static_decode_seq_lens_carry=True,
        )
    )
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=7)
    seq_b = Sequence([3, 4], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=8)
    seq_a.block_table = [0]
    seq_b.block_table = [1]
    seq_a.last_token = 0
    seq_b.last_token = 0
    seq_a.last_token_device = DeviceTokenRef(tokens=token_vector, row=0)
    seq_b.last_token_device = DeviceTokenRef(tokens=token_vector, row=1)

    first = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)
    seq_a.append_token_device(DeviceTokenRef(tokens=token_vector, row=0))
    seq_b.append_token_device(DeviceTokenRef(tokens=token_vector, row=1))
    second = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)

    assert first.uses_static_decode_metadata
    assert second.uses_static_decode_metadata
    assert second.seq_lens is first.seq_lens
    assert second.seq_lens_host == (3, 3)


def test_scheduler_resident_decode_metadata_uses_device_placeholders(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_METADATA", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_RESIDENT_DECODE_METADATA", "1")
    token_vector = jnp.asarray([70, 80], dtype=jnp.int32)
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            batch_size_buckets=(2,),
            max_blocks_per_seq=3,
            num_kvcache_blocks=6,
            jax_execution="jit",
            device_token_carry=True,
            static_decode_metadata=True,
            resident_decode_metadata=True,
        )
    )
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=7)
    seq_b = Sequence([3, 4], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=8)
    seq_a.block_table = [1]
    seq_b.block_table = [2]
    seq_a.last_token = 0
    seq_b.last_token = 0
    seq_a.last_token_device = DeviceTokenRef(tokens=token_vector, row=0)
    seq_b.last_token_device = DeviceTokenRef(tokens=token_vector, row=1)

    first = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)
    seq_a.append_token_device(DeviceTokenRef(tokens=token_vector, row=0))
    seq_b.append_token_device(DeviceTokenRef(tokens=token_vector, row=1))
    seq_a.block_table = [1, 4]
    second = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)

    assert first.uses_static_decode_metadata
    assert second.uses_static_decode_metadata
    assert second.block_tables is first.block_tables
    assert second.seq_lens is first.seq_lens
    np.testing.assert_array_equal(np.asarray(first.block_tables), np.zeros((2, 3), dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(second.seq_lens), np.zeros((2,), dtype=np.int32))
    assert first.block_tables_host == ((1, 0, 0), (2, 0, 0))
    assert second.block_tables_host == ((1, 4, 0), (2, 0, 0))
    assert second.seq_lens_host == (3, 3)


def test_scheduler_resident_decode_metadata_reuses_placeholders_across_seq_ids(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_METADATA", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_RESIDENT_DECODE_METADATA", "1")
    token_vector = jnp.asarray([70, 80], dtype=jnp.int32)
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            batch_size_buckets=(2,),
            max_blocks_per_seq=3,
            num_kvcache_blocks=6,
            jax_execution="jit",
            device_token_carry=True,
            static_decode_metadata=True,
            resident_decode_metadata=True,
        )
    )
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=7)
    seq_b = Sequence([3, 4], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=8)
    seq_a.block_table = [1]
    seq_b.block_table = [2]
    seq_a.last_token = 0
    seq_b.last_token = 0
    seq_a.last_token_device = DeviceTokenRef(tokens=token_vector, row=0)
    seq_b.last_token_device = DeviceTokenRef(tokens=token_vector, row=1)

    first = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False)

    seq_c = Sequence([5, 6], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=17)
    seq_d = Sequence([7, 8], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), seq_id=18)
    seq_c.block_table = [3]
    seq_d.block_table = [4]
    seq_c.last_token = 0
    seq_d.last_token = 0
    seq_c.last_token_device = DeviceTokenRef(tokens=token_vector, row=0)
    seq_d.last_token_device = DeviceTokenRef(tokens=token_vector, row=1)

    second = scheduler.build_scheduled_batch([seq_c, seq_d], is_prefill=False)

    assert first.uses_static_decode_metadata
    assert second.uses_static_decode_metadata
    assert first.seq_ids_host == (7, 8)
    assert second.seq_ids_host == (17, 18)
    assert second.tokens is first.tokens
    assert second.positions is first.positions
    assert second.seq_ids is first.seq_ids
    assert second.query_start_loc is first.query_start_loc
    assert second.block_tables is first.block_tables
    assert second.seq_lens is first.seq_lens
    np.testing.assert_array_equal(np.asarray(first.seq_ids), np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(second.block_tables), np.zeros((2, 3), dtype=np.int32))


def test_scheduler_static_decode_metadata_allows_greedy_burst(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_METADATA", "1")
    token_vector = jnp.asarray([70, 80], dtype=jnp.int32)
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=2,
            batch_size_buckets=(2,),
            max_blocks_per_seq=2,
            num_kvcache_blocks=4,
            jax_execution="jit",
            device_token_carry=True,
            static_decode_metadata=True,
            greedy_decode_burst_steps=2,
        )
    )
    seq_a = Sequence([1, 2], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True), seq_id=7)
    seq_b = Sequence([3, 4], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True), seq_id=8)
    seq_a.block_table = [0]
    seq_b.block_table = [1]
    seq_a.last_token = 0
    seq_b.last_token = 0
    seq_a.last_token_device = DeviceTokenRef(tokens=token_vector, row=0)
    seq_b.last_token_device = DeviceTokenRef(tokens=token_vector, row=1)

    batch = scheduler.build_scheduled_batch([seq_a, seq_b], is_prefill=False, decode_step_count=2)

    assert batch.uses_static_decode_metadata
    assert batch.decode_step_count_host == 2


def test_model_runner_static_decode_metadata_requires_token_carry(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    batch = _decode_batch((7,), [0])
    batch.uses_static_decode_metadata = True

    with pytest.raises(RuntimeError, match="requires a device-token carry"):
        runner._maybe_apply_device_token_carry(batch)


def test_model_runner_static_decode_metadata_applies_carried_seq_lens(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    runner._device_seq_lens_carry_seq_ids = None
    runner._device_seq_lens_carry = None
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    previous_batch = _decode_batch((7, 8), [10, 20], seq_lens=[5, 9])

    runner._record_device_token_carry(
        previous_batch,
        jnp.asarray([70, 80], dtype=jnp.int32),
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
    )
    next_batch = _decode_batch((7, 8), [0, 0], seq_lens=[1, 1])
    next_batch.uses_static_decode_metadata = True

    carried_batch = runner._maybe_apply_device_token_carry(next_batch)

    np.testing.assert_array_equal(np.asarray(carried_batch.tokens[:, 0]), np.asarray([70, 80]))
    np.testing.assert_array_equal(np.asarray(carried_batch.seq_lens), np.asarray([6, 10]))


def test_model_runner_first_static_decode_can_use_scheduler_seq_lens(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY", "1")
    runner = ModelRunner.__new__(ModelRunner)
    runner._device_token_carry_seq_ids = None
    runner._device_token_carry_tokens = None
    runner._device_token_carry_by_seq_id = {}
    runner._device_seq_lens_carry_seq_ids = None
    runner._device_seq_lens_carry = None
    seq_a = Sequence([1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=7)
    seq_b = Sequence([2], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), seq_id=8)
    previous_prefill = _prefill_batch((7, 8), [10, 20], final_flags=(True, True))

    runner._record_device_token_carry(
        previous_prefill,
        jnp.asarray([70, 80], dtype=jnp.int32),
        active_rows=[0, 1],
        prefill_final_flags=[True, True],
        seqs=[seq_a, seq_b],
    )
    next_batch = _decode_batch((7, 8), [0, 0], seq_lens=[5, 9])
    next_batch.uses_static_decode_metadata = True

    carried_batch = runner._maybe_apply_device_token_carry(next_batch)

    np.testing.assert_array_equal(np.asarray(carried_batch.tokens[:, 0]), np.asarray([70, 80]))
    np.testing.assert_array_equal(np.asarray(carried_batch.seq_lens), np.asarray([5, 9]))


def test_scheduler_postprocess_can_defer_greedy_device_token(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=1,
            num_kvcache_blocks=4,
            max_blocks_per_seq=4,
            device_token_carry=True,
        )
    )
    seq = Sequence(
        [101],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        seq_id=7,
    )

    finished = scheduler.postprocess([seq], [jnp.asarray(202, dtype=jnp.int32)])

    assert finished == [False]
    assert seq.num_completion_tokens == 1
    assert seq.has_unmaterialized_device_tokens
    assert seq.completion_token_ids == [202]


def test_scheduler_postprocess_can_defer_sampled_device_token_ref(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=1,
            num_kvcache_blocks=4,
            max_blocks_per_seq=4,
            device_token_carry=True,
        )
    )
    seq = Sequence(
        [101],
        SamplingParams(temperature=0.8, max_tokens=2, ignore_eos=True),
        seq_id=7,
    )
    token_vector = jnp.asarray([202], dtype=jnp.int32)

    finished = scheduler.postprocess([seq], [DeviceTokenRef(tokens=token_vector, row=0)])

    assert finished == [False]
    assert seq.num_completion_tokens == 1
    assert seq.has_unmaterialized_device_tokens
    assert seq.completion_token_ids == [202]


def test_iter_generate_overlapped_prefetch_emits_after_next_step(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_OVERLAPPED_STREAMING_TOKEN_PREFETCH", "1")
    events: list[str] = []
    deferred_tokens: list[_AsyncScalar] = []
    engine = LLMEngine.__new__(LLMEngine)
    seq_holder: dict[str, Sequence] = {}
    remaining_steps = {"value": 2}

    def add_request(prompt, sampling_params):
        seq = Sequence(prompt, sampling_params, seq_id=7)
        seq.status = SequenceStatus.RUNNING
        seq_holder["seq"] = seq
        return seq

    def is_finished():
        return remaining_steps["value"] <= 0

    def step():
        step_index = 2 - remaining_steps["value"]
        events.append(f"step-{step_index}")
        token = _AsyncScalar(202 + step_index, events)
        deferred_tokens.append(token)
        seq = seq_holder["seq"]
        seq.append_token_device(token)
        remaining_steps["value"] -= 1
        if remaining_steps["value"] <= 0:
            seq.status = SequenceStatus.FINISHED
        return [], -1

    engine.add_request = add_request
    engine.is_finished = is_finished
    engine.step = step
    engine._detokenize = lambda token_ids: ""

    stream_events = list(
        engine.iter_generate(
            [[101]],
            SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            include_text=False,
        )
    )

    token_events = [event for event in stream_events if event["event"] == "token"]
    assert [event["token_id"] for event in token_events] == [202, 203]
    assert [event["completion_index"] for event in token_events] == [0, 1]
    assert stream_events[-1]["event"] == "done"
    assert stream_events[-1]["results"][0]["token_ids"] == [202, 203]
    assert all(token.prefetch_count == 1 for token in deferred_tokens)
    assert events.index("step-1") < events.index("materialize-202")
    assert events == [
        "step-0",
        "prefetch-202",
        "step-1",
        "prefetch-203",
        "materialize-202",
        "materialize-203",
    ]


def test_iter_generate_offline_events_preserve_final_tokens(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_OFFLINE_STREAMING_TOKEN_EVENTS", "1")
    events: list[str] = []
    deferred_tokens: list[_AsyncScalar] = []
    engine = LLMEngine.__new__(LLMEngine)
    seq_holder: dict[str, Sequence] = {}
    remaining_steps = {"value": 2}

    def add_request(prompt, sampling_params):
        seq = Sequence(prompt, sampling_params, seq_id=7)
        seq.status = SequenceStatus.RUNNING
        seq_holder["seq"] = seq
        return seq

    def is_finished():
        return remaining_steps["value"] <= 0

    def step():
        step_index = 2 - remaining_steps["value"]
        token = _AsyncScalar(202 + step_index, events)
        seq = seq_holder["seq"]
        seq.append_token_device(token)
        deferred_tokens.append(token)
        remaining_steps["value"] -= 1
        if remaining_steps["value"] <= 0:
            seq.status = SequenceStatus.FINISHED
        return [], -1

    engine.add_request = add_request
    engine.is_finished = is_finished
    engine.step = step
    engine._detokenize = lambda token_ids: ""

    stream_events = list(
        engine.iter_generate(
            [[101]],
            SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            include_text=False,
        )
    )

    token_events = [event for event in stream_events if event["event"] == "token"]
    assert [event["token_id"] for event in token_events] == [None, None]
    assert len(token_events) == 2
    assert token_events[0]["completion_index"] == 0
    assert token_events[1]["completion_index"] == 1
    assert stream_events[-1]["event"] == "done"
    assert stream_events[-1]["results"][0]["token_ids"] == [202, 203]
    assert [event for event in events if event.startswith("materialize-")] == [
        "materialize-202",
        "materialize-203",
    ]


def test_generate_with_trace_disables_overlapped_prefetch(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_OVERLAPPED_STREAMING_TOKEN_PREFETCH", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH", "0")
    prefetch_calls: list[tuple] = []
    events: list[str] = []
    remaining_steps = {"value": 2}
    seq_holder: dict[str, Sequence] = {}

    original_prefetch = Sequence.prefetch_device_token_slots

    def _prefetch_device_token_slots(slots):
        prefetch_calls.append(tuple(slots))
        return original_prefetch(slots)

    def add_request(prompt, sampling_params):
        seq = Sequence(prompt, sampling_params, seq_id=7)
        seq.status = SequenceStatus.RUNNING
        seq_holder["seq"] = seq
        return seq

    def is_finished():
        return remaining_steps["value"] <= 0

    def step():
        step_index = 2 - remaining_steps["value"]
        token = _AsyncScalar(202 + step_index, events)
        seq = seq_holder["seq"]
        seq.append_token_device(token)
        remaining_steps["value"] -= 1
        if remaining_steps["value"] <= 0:
            seq.status = SequenceStatus.FINISHED
        return [], -1

    monkeypatch.setattr(
        Sequence,
        "prefetch_device_token_slots",
        staticmethod(_prefetch_device_token_slots),
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.add_request = add_request
    engine.is_finished = is_finished
    engine.step = step
    engine._detokenize = lambda token_ids: ""

    trace = engine.generate_with_trace(
        [[101]],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        include_text=False,
    )

    assert prefetch_calls == []
    token_events = [event for event in trace["events"] if event["event"] == "token"]
    assert [event["token_id"] for event in token_events] == [202, 203]
    assert trace["events"][-1]["event"] == "done"
    assert trace["events"][-1]["results"] == trace["results"]
    assert isinstance(trace["events"][-1]["elapsed_seconds"], float)
    assert trace["results"][0]["token_ids"] == [202, 203]


def test_generate_with_trace_prefetches_when_trace_prefetch_enabled(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH", "1")
    prefetch_calls: list[tuple] = []
    events: list[str] = []
    remaining_steps = {"value": 2}
    seq_holder: dict[str, Sequence] = {}

    original_prefetch = Sequence.prefetch_device_token_slots

    def _prefetch_device_token_slots(slots):
        prefetch_calls.append(tuple(slots))
        return original_prefetch(slots)

    def add_request(prompt, sampling_params):
        seq = Sequence(prompt, sampling_params, seq_id=7)
        seq.status = SequenceStatus.RUNNING
        seq_holder["seq"] = seq
        return seq

    def is_finished():
        return remaining_steps["value"] <= 0

    def step():
        step_index = 2 - remaining_steps["value"]
        events.append(f"step-{step_index}")
        token = _AsyncScalar(202 + step_index, events)
        seq = seq_holder["seq"]
        seq.append_token_device(token)
        remaining_steps["value"] -= 1
        if remaining_steps["value"] <= 0:
            seq.status = SequenceStatus.FINISHED
        return [], -1

    monkeypatch.setattr(
        Sequence,
        "prefetch_device_token_slots",
        staticmethod(_prefetch_device_token_slots),
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.add_request = add_request
    engine.is_finished = is_finished
    engine.step = step
    engine._detokenize = lambda token_ids: ""

    trace = engine.generate_with_trace(
        [[101]],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        include_text=False,
    )

    assert len(prefetch_calls) == 2
    token_events = [event for event in trace["events"] if event["event"] == "token"]
    assert [event["token_id"] for event in token_events] == [202, 203]
    assert trace["events"][-1]["event"] == "done"
    assert trace["results"][0]["token_ids"] == [202, 203]
    assert events == [
        "step-0",
        "step-1",
        "prefetch-202",
        "prefetch-203",
        "materialize-202",
        "materialize-203",
    ]


def test_generate_with_trace_keeps_eos_compatible_path_when_ignore_eos_false(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    calls = []

    def iter_generate(prompts, sampling_params=None, *, include_text=True):
        calls.append((prompts, sampling_params, include_text))
        yield {"event": "token", "token_id": 123}
        yield {"event": "done", "results": [{"text": "ok", "token_ids": [123]}]}

    engine = LLMEngine.__new__(LLMEngine)
    engine.iter_generate = iter_generate

    trace = engine.generate_with_trace(
        [[101]],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=False),
        include_text=False,
    )

    assert calls
    assert trace["events"][0]["event"] == "token"
    assert trace["results"] == [{"text": "ok", "token_ids": [123]}]
