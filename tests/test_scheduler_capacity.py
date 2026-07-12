from types import SimpleNamespace

import pytest

from nanovllm_jax.config import RuntimeConfig
from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams, Sequence


def _scheduler(*, block_size: int = 2, num_blocks: int = 3) -> Scheduler:
    return Scheduler(
        RuntimeConfig(
            block_size=block_size,
            num_kvcache_blocks=num_blocks,
            max_kv_cache_bytes=1 << 30,
            max_num_seqs=1,
            max_num_resident_seqs=1,
            max_num_batched_tokens=2,
            max_blocks_per_seq=num_blocks,
            prefill_buckets=(2,),
            prefill_token_buckets=(2,),
            batch_size_buckets=(1,),
            decode_block_table_buckets=(num_blocks,),
            prefix_cache=False,
            device_token_carry=False,
            static_decode_metadata=False,
            resident_decode_metadata=False,
        )
    )


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

    seqs, _ = scheduler.schedule()
    assert seqs == [first]
    assert len(first.block_table) == 3
    scheduler.postprocess(seqs, [10], prefill_chunk_lengths=[2])

    for token in (11, 12, 13):
        seqs, _ = scheduler.schedule()
        assert seqs == [first]
        scheduler.postprocess(seqs, [token])

    assert first.completion_token_ids == [10, 11, 12, 13]
    assert first.is_finished
    assert len(scheduler.block_manager.free_block_ids) == 3

    seqs, _ = scheduler.schedule()
    assert seqs == [second]
    assert len(second.block_table) == 2


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
        engine.config = SimpleNamespace(block_size=block_size)
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
