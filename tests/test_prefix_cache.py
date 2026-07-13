from dataclasses import replace

import jax
import pytest

from nanovllm_jax.block_manager import BlockManager, PrefixCacheEntry
from nanovllm_jax.config import RuntimeConfig
from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.model import init_params
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams, Sequence


def _sequence(tokens, *, seq_id=0, max_tokens=1):
    return Sequence(
        list(tokens),
        SamplingParams(max_tokens=max_tokens),
        seq_id=seq_id,
        block_size=2,
    )


def _tiny_hybrid_config(*, prefix_cache, max_num_resident_seqs=1):
    return RuntimeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_size=4,
        layer_types=("linear_attention",),
        linear_attn_layers=(0,),
        dtype="float32",
        block_size=2,
        num_kvcache_blocks=8,
        max_kv_cache_bytes=1 << 20,
        max_num_seqs=1,
        max_num_resident_seqs=max_num_resident_seqs,
        max_num_batched_tokens=4,
        max_blocks_per_seq=4,
        prefill_buckets=(2, 4),
        prefill_token_buckets=(2, 4),
        batch_size_buckets=(1,),
        decode_block_table_buckets=(4,),
        prefix_cache=prefix_cache,
        device_token_carry=False,
    )


class _TinyEngine(LLMEngine):
    """Explicit lightweight engine fixture that skips checkpoint loading."""

    def __init__(self, config, params):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config, params)
        self._next_seq_id = 0


def _prefix_scheduler(*, max_num_seqs=1, max_num_resident_seqs=1):
    return Scheduler(
        RuntimeConfig(
            block_size=2,
            num_kvcache_blocks=16,
            max_kv_cache_bytes=1 << 20,
            max_num_seqs=max_num_seqs,
            max_num_resident_seqs=max_num_resident_seqs,
            max_num_batched_tokens=2,
            max_blocks_per_seq=4,
            prefill_buckets=(2,),
            prefill_token_buckets=(2,),
            batch_size_buckets=tuple(range(1, max_num_seqs + 1)),
            decode_block_table_buckets=(4,),
            prefix_cache=True,
            linear_attn_layers=(0,),
        )
    )


def _prime_prefix(scheduler, tokens, *, seq_id, handle):
    seq = _sequence(tokens, seq_id=seq_id)
    scheduler.add(seq)
    seqs, plan = scheduler.schedule()
    pending = scheduler.record_computed_prefixes(
        seqs,
        list(plan.prefill_chunk_lengths),
    )
    entry = pending[seq_id]
    scheduler.publish_prefix_states(pending, {entry.prefix_hash: handle})
    scheduler.release(seq)
    return entry


def _generate(engine, prompt):
    seq = engine.add_request(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
    )
    while not seq.is_finished:
        engine.step()
    return seq.output.token_ids()


def _has_cuda():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def test_physical_reuse_releases_the_whole_prefix_entry():
    manager = BlockManager(4, 2, prefix_state_capacity=1)
    cached = _sequence([1, 2, 3])
    manager.reserve(cached, total_blocks=2)
    entry = manager.publish_computed_prefix(cached, 2)
    assert entry is not None
    entry = manager.prefix_cache.publish(replace(entry, hybrid_state_handle=7))
    manager.deallocate(cached)

    assert entry.token_count == 2
    assert entry.block_ids == (0,)
    assert manager.take_released_prefix_state_handles() == ()

    churn = _sequence([9, 8, 7, 6, 5, 4, 3], seq_id=1)
    manager.reserve(churn, total_blocks=4, use_prefix_cache=False)

    assert manager.prefix_cache.get(entry.prefix_hash) is None
    assert manager.take_released_prefix_state_handles() == (7,)
    assert manager.stats()["prefix_evictions"] == 1


def test_prefix_state_budget_is_lru_and_keeps_kv_metadata():
    manager = BlockManager(6, 2, prefix_state_capacity=1)
    first = _sequence([1, 2, 3])
    manager.reserve(first, total_blocks=2)
    first_entry = manager.publish_computed_prefix(first, 2)
    assert first_entry is not None
    first_entry = manager.prefix_cache.publish(
        replace(first_entry, hybrid_state_handle=10)
    )
    manager.deallocate(first)

    second = _sequence([4, 5, 6], seq_id=1)
    manager.reserve(second, total_blocks=2)
    second_entry = manager.publish_computed_prefix(second, 2)
    assert second_entry is not None

    manager.prefix_cache.make_state_room([second_entry])
    manager.prefix_cache.publish(replace(second_entry, hybrid_state_handle=11))

    assert manager.take_released_prefix_state_handles() == (10,)
    assert manager.prefix_cache.get(first_entry.prefix_hash) is not None
    assert manager.prefix_cache.get(first_entry.prefix_hash).hybrid_state_handle is None
    assert manager.prefix_cache.num_state_entries == 1
    assert manager.stats()["prefix_state_evictions"] == 1

    extended = _sequence([1, 2, 9], seq_id=2)
    assert manager.cached_prefix_info(extended) == (0, None)


def test_cache_hit_leaves_one_prompt_token_for_logits():
    manager = BlockManager(4, 2)
    cached = _sequence([1, 2, 3])
    manager.reserve(cached, total_blocks=2)
    entry = manager.publish_computed_prefix(cached, 2)
    manager.deallocate(cached)
    assert entry is not None

    assert manager.cached_prefix_info(_sequence([1, 2], seq_id=1)) == (0, None)
    assert manager.cached_prefix_info(_sequence([1, 2, 3], seq_id=2)) == (
        2,
        entry.prefix_hash,
    )


def test_prefix_churn_keeps_metadata_and_handles_bounded():
    manager = BlockManager(4, 2, prefix_state_capacity=1)
    released = []

    for index in range(64):
        seq = _sequence([index, index + 1, index + 2], seq_id=index)
        manager.reserve(seq, total_blocks=2)
        entry = manager.publish_computed_prefix(seq, 2)
        assert entry is not None
        manager.prefix_cache.make_state_room([entry])
        released.extend(manager.take_released_prefix_state_handles())
        manager.prefix_cache.publish(
            replace(entry, hybrid_state_handle=index)
        )
        manager.deallocate(seq)

        stats = manager.stats()
        assert stats["prefix_entries"] <= stats["total_blocks"]
        assert stats["prefix_state_entries"] <= stats["prefix_state_capacity"]

    assert sorted(released) == list(range(63))


def test_scheduler_passes_only_exact_host_state_handles():
    scheduler = _prefix_scheduler()
    original = _sequence([1, 2, 3])
    scheduler.add(original)
    seqs, plan = scheduler.schedule()
    pending = scheduler.record_computed_prefixes(seqs, list(plan.prefill_chunk_lengths))
    entry = pending[original.seq_id]
    scheduler.publish_prefix_states(pending, {entry.prefix_hash: 17})
    scheduler.release(original)

    reused = _sequence([1, 2, 9], seq_id=1)
    scheduler.add(reused)
    seqs, _ = scheduler.schedule()

    assert seqs == [reused]
    assert reused.num_cached_tokens == 2
    entries = scheduler.cached_prefix_entries(seqs)
    assert entries[reused.seq_id].hybrid_state_handle == 17
    assert entries[reused.seq_id].token_count == reused.num_cached_tokens
    assert scheduler.take_released_prefix_state_handles() == ()


def test_prefill_budget_does_not_admit_an_unseeded_prefix_hit():
    scheduler = _prefix_scheduler(max_num_seqs=2, max_num_resident_seqs=3)
    old = _prime_prefix(scheduler, [3, 4, 30], seq_id=0, handle=10)
    _prime_prefix(scheduler, [5, 6, 30], seq_id=1, handle=11)
    kept = _prime_prefix(scheduler, [1, 2, 30], seq_id=2, handle=12)

    filler = _sequence([7, 8, 30], seq_id=3)
    kept_waiter = _sequence([1, 2, 31], seq_id=4)
    old_waiter = _sequence([3, 4, 31], seq_id=5)
    for seq in (filler, kept_waiter, old_waiter):
        scheduler.add(seq)

    seqs, plan = scheduler.schedule()

    assert seqs == [filler]
    assert list(scheduler.waiting) == [kept_waiter, old_waiter]
    assert kept_waiter.block_table == old_waiter.block_table == []
    assert kept_waiter.cached_prefix_hash is None

    pending = scheduler.record_computed_prefixes(
        seqs,
        list(plan.prefill_chunk_lengths),
    )
    new_entry = pending[filler.seq_id]
    assert scheduler.take_released_prefix_state_handles() == (10,)
    scheduler.publish_prefix_states(pending, {new_entry.prefix_hash: 13})
    assert scheduler.block_manager.prefix_cache.get(old.prefix_hash) is not None
    assert (
        scheduler.block_manager.prefix_cache.get(old.prefix_hash).hybrid_state_handle
        is None
    )

    scheduler.release(filler)
    seqs, _ = scheduler.schedule()

    assert seqs[0] is kept_waiter
    assert kept_waiter.num_cached_tokens == 2
    entries = scheduler.cached_prefix_entries(seqs)
    entry = entries[kept_waiter.seq_id]
    assert entry.prefix_hash == kept.prefix_hash
    assert entry.hybrid_state_handle == 12
    assert old_waiter.num_cached_tokens == 0
    assert old_waiter.seq_id not in entries


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for engine parity")
def test_hybrid_prefix_hit_matches_no_cache_execution():
    params = init_params(jax.random.PRNGKey(0), _tiny_hybrid_config(prefix_cache=False))
    reference = _TinyEngine(_tiny_hybrid_config(prefix_cache=False), params)
    cached = _TinyEngine(_tiny_hybrid_config(prefix_cache=True), params)

    expected = _generate(reference, [1, 2, 3, 4])
    _generate(cached, [1, 2])
    compiled_routes = set(cached.model_runner.executor._jit_cache)
    actual = _generate(cached, [1, 2, 3, 4])

    assert actual == expected
    assert set(cached.model_runner.executor._jit_cache) == compiled_routes
    cache_stats = cached.scheduler.block_manager.stats()
    assert cache_stats["prefix_hits"] == 1
    assert cache_stats["prefix_state_evictions"] == 1
    assert cache_stats["prefix_entries"] <= cache_stats["total_blocks"]
    assert cached.model_runner.prefix_hybrid_state_stats() == {
        "handles": 1,
        "capacity": 1,
    }
    memory = cached.model_runner.memory_bytes()
    assert (
        0
        < memory["prefix_hybrid_state_current"]
        <= memory["prefix_hybrid_state_capacity"]
    )
    with pytest.raises(AssertionError, match="unknown prefix-state handle"):
        cached.model_runner.release_prefix_hybrid_states((999,))


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for engine parity")
def test_late_warmup_cannot_invalidate_or_crosswire_prefix_state():
    config = _tiny_hybrid_config(prefix_cache=True, max_num_resident_seqs=2)
    reference_config = _tiny_hybrid_config(
        prefix_cache=False,
        max_num_resident_seqs=2,
    )
    params = init_params(jax.random.PRNGKey(1), reference_config)
    reference = _TinyEngine(reference_config, params)
    cached = _TinyEngine(config, params)

    expected = _generate(reference, [1, 2, 3, 4])
    _generate(cached, [1, 2])
    prefix_hash = cached.scheduler.block_manager.compute_hash([1, 2])
    entry = cached.scheduler.block_manager.prefix_cache.get(prefix_hash)
    assert entry is not None and entry.hybrid_state_handle is not None
    handle = entry.hybrid_state_handle
    snapshot = cached.model_runner._prefix_hybrid_states[handle]

    crosswired = _sequence([1, 2, 9], seq_id=99)
    crosswired.num_cached_tokens = entry.token_count
    with pytest.raises(RuntimeError, match="state hashes differ"):
        cached.model_runner.install_cached_prefix_hybrid_states(
            [crosswired],
            {99: replace(entry, prefix_hash=prefix_hash + 1)},
        )
    with pytest.raises(RuntimeError, match="warmup_compilation must run before"):
        cached.warmup_compilation()
    assert cached.scheduler.block_manager.prefix_cache.get(prefix_hash) == entry
    assert cached.model_runner._prefix_hybrid_states[handle] is snapshot

    _generate(cached, [5, 6])
    actual = _generate(cached, [1, 2, 3, 4])

    assert actual == expected
