from types import SimpleNamespace
import subprocess
import sys

import pytest

from nanovllm_jax.block_manager import BlockManager
from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams, Sequence
from nanovllm_jax.step import RunResult
from tests.runtime_specs import runtime_spec


def test_scheduler_and_sequence_import_without_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import nanovllm_jax.scheduler, nanovllm_jax.sequence; "
                "assert 'jax' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _scheduler(*, block_size: int = 2, num_blocks: int = 3) -> Scheduler:
    return Scheduler(
        runtime_spec(
            capacity={
                "block_size": block_size,
                "num_kvcache_blocks": num_blocks,
                "max_kv_cache_bytes": 1 << 30,
                "max_num_seqs": 1,
                "max_num_resident_seqs": 1,
                "max_num_batched_tokens": 2,
                "max_blocks_per_seq": num_blocks,
                "prefix_cache": False,
            },
            compile={
                "prefill_token_buckets": (2,),
                "batch_size_buckets": (1,),
                "decode_block_table_buckets": (num_blocks,),
            },
        )
    )


def _commit(scheduler: Scheduler, seqs, plan, rows):
    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler
    return engine.commit(seqs, plan, RunResult.from_rows(rows))


def test_capacity_is_reserved_before_generation_and_never_preempted():
    scheduler = _scheduler()
    first = Sequence(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        seq_id=0,
        block_size=2,
    )
    second = Sequence(
        [3, 4],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        seq_id=1,
        block_size=2,
    )
    scheduler.add(first)
    scheduler.add(second)

    seqs, plan = scheduler.schedule()
    assert seqs == [first]
    assert len(first.block_table) == 1
    assert scheduler.block_manager.stats()["reserved_blocks"] == 2
    _commit(scheduler, seqs, plan, [10])

    for token in (11, 12, 13):
        seqs, plan = scheduler.schedule()
        assert seqs == [first]
        _commit(scheduler, seqs, plan, [token])

    assert first.output.token_ids() == [10, 11, 12, 13]
    assert first.is_finished
    assert len(scheduler.block_manager.free_block_ids) == 3
    assert scheduler.block_manager.stats()["reserved_blocks"] == 0

    seqs, _ = scheduler.schedule()
    assert seqs == [second]
    assert len(second.block_table) == 1
    assert scheduler.block_manager.stats()["reserved_blocks"] == 1


def test_request_larger_than_engine_is_rejected_before_prefill():
    scheduler = _scheduler()
    seq = Sequence(
        [1, 2],
        SamplingParams(max_tokens=6),
        block_size=2,
    )

    with pytest.raises(ValueError, match="request needs 4 blocks"):
        scheduler.add(seq)


def test_sequence_ids_and_block_sizes_are_engine_local():
    class Queue:
        def __init__(self):
            self.seqs = []

        def add(self, seq):
            self.seqs.append(seq)

    def bare_engine(block_size: int) -> LLMEngine:
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(capacity=SimpleNamespace(block_size=block_size))
        engine.scheduler = Queue()
        engine._next_seq_id = 0
        return engine

    first = bare_engine(8)
    second = bare_engine(32)
    params = SamplingParams(max_tokens=1)

    first_a = first.add_request([1], params)
    first_b = first.add_request([2], params)
    second_a = second.add_request([3], params)

    assert (first_a.seq_id, first_b.seq_id, second_a.seq_id) == (0, 1, 0)
    assert (first_a.block_size, first_b.block_size, second_a.block_size) == (8, 8, 32)


def test_future_capacity_reservation_does_not_evict_cached_prefix():
    manager = BlockManager(num_blocks=4, block_size=2)
    cached = Sequence(
        [1, 2],
        SamplingParams(max_tokens=1),
        seq_id=0,
        block_size=2,
    )
    manager.reserve(cached, total_blocks=1)
    cached_block = cached.block_table[0]
    cached_hash = manager.record_computed_prefix(cached, 2, publish=True)
    manager.deallocate(cached)

    assert manager.prefix_cache.get(cached_hash).block_ids[-1] == cached_block

    long_request = Sequence(
        [9],
        SamplingParams(max_tokens=5),
        seq_id=1,
        block_size=2,
    )
    manager.reserve(long_request, total_blocks=3)

    assert len(long_request.block_table) == 1
    assert manager.stats()["reserved_blocks"] == 2
    assert manager.prefix_cache.get(cached_hash).block_ids[-1] == cached_block
    assert manager.blocks[cached_block].token_ids == [1, 2]


def test_first_fitting_waiter_bypasses_blocked_large_request():
    scheduler = Scheduler(
        runtime_spec(
            capacity={
                "block_size": 2,
                "num_kvcache_blocks": 5,
                "max_kv_cache_bytes": 1 << 30,
                "max_num_seqs": 2,
                "max_num_resident_seqs": 2,
                "max_num_batched_tokens": 2,
                "max_blocks_per_seq": 5,
                "prefix_cache": False,
            },
            compile={
                "prefill_token_buckets": (2,),
                "batch_size_buckets": (1, 2),
                "decode_block_table_buckets": (1, 2, 3, 4, 5),
            },
        )
    )
    active = Sequence([1, 2], SamplingParams(max_tokens=2), seq_id=0, block_size=2)
    scheduler.add(active)
    seqs, plan = scheduler.schedule()
    _commit(scheduler, seqs, plan, [10])

    large = Sequence([3, 4], SamplingParams(max_tokens=6), seq_id=1, block_size=2)
    small = Sequence([5], SamplingParams(max_tokens=1), seq_id=2, block_size=2)
    scheduler.add(large)
    scheduler.add(small)

    seqs, _ = scheduler.schedule()

    assert seqs == [small]
    assert list(scheduler.waiting) == [large]
