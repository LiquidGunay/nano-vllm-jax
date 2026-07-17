import subprocess
import sys
from threading import Lock
from types import SimpleNamespace

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


def _bare_engine(*, block_size: int = 2, num_blocks: int = 3) -> LLMEngine:
    engine = object.__new__(LLMEngine)
    engine.config = runtime_spec(
        model={"vocab_size": 16},
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
    engine.scheduler = Scheduler(engine.config)
    engine.model_runner = SimpleNamespace(release=lambda seq_ids: None)
    engine._next_seq_id = 0
    engine._closed = False
    engine._control_owner = None
    engine._control_lock = Lock()
    return engine


def _commit(scheduler: Scheduler, seqs, plan, rows):
    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler
    return engine._commit(seqs, plan, RunResult.from_rows(rows))


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
    first = _bare_engine(block_size=8)
    second = _bare_engine(block_size=32)
    params = SamplingParams(max_tokens=1)

    first_a = first.add_request([1], params)
    first_b = first.add_request([2], params)
    second_a = second.add_request([3], params)

    assert (first_a.seq_id, first_b.seq_id, second_a.seq_id) == (0, 1, 0)
    assert (first_a.block_size, first_b.block_size, second_a.block_size) == (8, 8, 32)


def test_control_lease_guards_every_mutating_entrypoint():
    engine = _bare_engine()
    owner = object()
    engine.claim_control(owner)

    with pytest.raises(RuntimeError, match="active control owner"):
        engine.add_request([1], SamplingParams(max_tokens=1))
    with pytest.raises(RuntimeError, match="active control owner"):
        engine.step()
    with pytest.raises(RuntimeError, match="active control owner"):
        engine.commit([], None, RunResult.from_rows([]))
    with pytest.raises(RuntimeError, match="active control owner"):
        engine.warmup_compilation()

    admitted = engine.add_request([1], SamplingParams(max_tokens=1), owner=owner)
    assert admitted.seq_id == 0
    engine.cancel_request(admitted, owner=owner)
    engine.release_control(owner)


def test_control_lease_rejects_none_owner():
    engine = _bare_engine()

    with pytest.raises(ValueError, match="cannot be None"):
        engine.claim_control(None)

    assert engine._control_owner is None


def test_control_lease_rejects_preexisting_manual_requests():
    engine = _bare_engine()
    engine.add_request([1], SamplingParams(max_tokens=1))

    with pytest.raises(RuntimeError, match="requires an idle engine"):
        engine.claim_control(object())


@pytest.mark.parametrize("entrypoint", ("generate", "iter_generate"))
def test_offline_batch_admission_is_atomic(entrypoint):
    engine = _bare_engine()
    prompts = [[1], [2]]
    sampling = [SamplingParams(max_tokens=1), SamplingParams(max_tokens=6)]

    with pytest.raises(ValueError, match=r"request\[1\] needs 4 blocks"):
        if entrypoint == "generate":
            engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
        else:
            next(engine.iter_generate(prompts, sampling_params=sampling))

    assert not engine.scheduler.waiting
    assert not engine.scheduler.running
    assert engine.scheduler.block_manager.stats()["reserved_blocks"] == 0
    assert engine._next_seq_id == 0

    admitted = engine.add_request([3], SamplingParams(max_tokens=1))
    fresh = _bare_engine().add_request([3], SamplingParams(max_tokens=1))
    assert (admitted.seq_id, admitted.prompt_token_ids) == (
        fresh.seq_id,
        fresh.prompt_token_ids,
    )


@pytest.mark.parametrize(
    ("token_ids", "error"),
    (
        ([True], TypeError),
        ([1.0], TypeError),
        ([-1], ValueError),
        ([16], ValueError),
    ),
)
def test_engine_rejects_invalid_prompt_token_ids_before_admission(token_ids, error):
    engine = _bare_engine()

    with pytest.raises(error):
        engine.add_request(token_ids, SamplingParams(max_tokens=1))

    assert not engine.scheduler.waiting
    assert engine._next_seq_id == 0


@pytest.mark.parametrize(
    "factory",
    (
        lambda: SamplingParams(temperature=float("nan")),
        lambda: SamplingParams(temperature=float("inf")),
        lambda: SamplingParams(max_tokens=True),
        lambda: SamplingParams(max_tokens=1.5),
        lambda: SamplingParams(ignore_eos="false"),
    ),
)
def test_sampling_params_reject_ambiguous_values(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_sampling_params_default_to_greedy():
    assert SamplingParams().temperature == 0.0


def test_close_drops_heavy_state_even_when_request_release_fails():
    def fail_release(_seq_ids):
        raise RuntimeError("release failed")

    engine = _bare_engine()
    engine.params = object()
    engine.mtp_params = object()
    engine.tokenizer = object()
    engine.model_runner = SimpleNamespace(
        release=fail_release,
        release_prefix_hybrid_states=lambda _handles: None,
    )
    seq = engine.add_request([1], SamplingParams(max_tokens=1))

    with pytest.raises(RuntimeError, match="release failed"):
        engine.close()

    assert engine._closed
    assert seq.is_finished
    assert engine.scheduler.is_finished()
    for name in ("model_runner", "params", "mtp_params", "tokenizer"):
        assert not hasattr(engine, name)
    engine.close()


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
