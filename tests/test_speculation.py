import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import FrozenInstanceError, replace

from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.model import init_params
from nanovllm_jax.mtp import MTPState, init_mtp_params
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.routes import RouteKind
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.sequence import Sequence
from nanovllm_jax.speculation import (
    DrafterConfig,
    DraftProposal,
    VerificationResult,
    verify_greedy_drafts,
)
from tests.runtime_specs import runtime_spec


def test_greedy_verification_handles_reject_partial_and_full_accept():
    proposal = DraftProposal(
        jnp.array(
            [
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32],
            ],
            dtype=jnp.int32,
        )
    )
    target = jnp.array(
        [
            [9, 11, 12, 13],
            [20, 8, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=jnp.int32,
    )

    result = verify_greedy_drafts(proposal, target)

    np.testing.assert_array_equal(result.accepted_counts, [0, 1, 3])
    np.testing.assert_array_equal(result.emitted_counts, [1, 2, 4])
    np.testing.assert_array_equal(result.next_token_ids, [9, 8, 33])
    np.testing.assert_array_equal(
        result.emitted_token_ids,
        [
            [9, 0, 0, 0],
            [20, 8, 0, 0],
            [30, 31, 32, 33],
        ],
    )


def test_speculation_values_are_jax_pytrees():
    proposal = DraftProposal(jnp.zeros((2, 3), dtype=jnp.int32))
    result = verify_greedy_drafts(
        proposal,
        jnp.zeros((2, 4), dtype=jnp.int32),
    )

    assert len(jax.tree_util.tree_leaves(proposal)) == 1
    assert isinstance(jax.tree_util.tree_map(lambda value: value + 1, result), VerificationResult)


def test_drafter_config_owns_width_and_reservations():
    drafter = DrafterConfig.mtp(3)

    assert drafter.verification_width == 4
    assert drafter.prefill_lookahead_tokens == 2
    assert drafter.decode_lookahead_slots == 6
    assert drafter.capacity_padding_tokens == 1
    with pytest.raises(FrozenInstanceError):
        drafter.width = 2
    with pytest.raises(TypeError, match="integer"):
        DrafterConfig.mtp(2.5)


def test_runtime_rejects_incompatible_mtp_at_construction():
    with pytest.raises(ValueError, match="prefix_cache=False"):
        runtime_spec(
            compile={"execution": "jit", "prefill_layout": "packed"},
            kernels={
                "device_token_carry": True,
                "static_decode_metadata": True,
                "resident_decode_metadata": True,
            },
            drafter=DrafterConfig.mtp(2),
        )
    with pytest.raises(ValueError, match="resident_decode_metadata"):
        runtime_spec(
            capacity={"prefix_cache": False},
            compile={"execution": "jit", "prefill_layout": "packed"},
            kernels={
                "device_token_carry": True,
                "static_decode_metadata": True,
            },
            drafter=DrafterConfig.mtp(2),
        )


def test_scheduler_derives_mtp_capacity_from_frozen_width():
    config = replace(_runtime(mtp=True), drafter=DrafterConfig.mtp(3))
    scheduler = Scheduler(config)
    seq = Sequence(
        [1, 2, 3],
        SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True),
        block_size=config.capacity.block_size,
    )

    assert scheduler._required_blocks(seq) == 5
    assert scheduler._speculative_lookahead(seq, remaining_tokens=4) == 6


def test_verifier_requires_one_bonus_target():
    proposal = DraftProposal(jnp.zeros((1, 2), dtype=jnp.int32))
    with pytest.raises(ValueError, match="one bonus token"):
        verify_greedy_drafts(proposal, jnp.zeros((1, 2), dtype=jnp.int32))


def _has_cuda():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def _runtime(*, mtp: bool = False, max_seqs: int = 1):
    return runtime_spec(
        model={
            "vocab_size": 64,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_size": 4,
            "layer_types": ("linear_attention", "full_attention"),
        },
        capacity={
            "block_size": 2,
            "num_kvcache_blocks": 64,
            "max_kv_cache_bytes": 1 << 20,
            "max_num_seqs": max_seqs,
            "max_num_resident_seqs": max_seqs,
            "max_num_batched_tokens": 3 * max_seqs,
            "max_blocks_per_seq": 16,
            "prefix_cache": False,
        },
        compile={
            "dtype": "float32",
            "execution": "jit",
            "prefill_token_buckets": (3 * max_seqs,),
            "batch_size_buckets": (max_seqs,),
            "decode_block_table_buckets": (16,),
        },
        kernels={
            "device_token_carry": True,
            "static_decode_metadata": True,
            "resident_decode_metadata": True,
        },
        drafter=DrafterConfig.mtp(2) if mtp else None,
    )


class _Engine(LLMEngine):
    def __init__(self, config, params, mtp_params=None):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config, params, mtp_params=mtp_params)
        self._next_seq_id = 0


def _generate(engine, *, max_tokens, prompt=(1, 2, 3), materialize_each_step=False):
    seq = engine.add_request(
        list(prompt),
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
    )
    while not engine.is_finished():
        engine.step()
        if materialize_each_step:
            seq.output.materialize()
    return seq.output.materialize()


def _replace_slot_drafts(engine, seq, token_ids):
    runner = engine.model_runner
    state = runner.mtp_state
    slot = runner._hybrid_slots[seq.seq_id]
    runner.mtp_state = MTPState(
        state.cache_storage,
        state.draft_token_ids.at[slot].set(
            jnp.asarray(token_ids, dtype=jnp.int32)
        ),
    )


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for verifier parity")
def test_packed_verifier_matches_decode_across_accept_and_reject_paths():
    base_config = _runtime()
    mtp_config = _runtime(mtp=True)
    params = init_params(jax.random.PRNGKey(0), base_config.model)
    expected = _generate(_Engine(base_config, params), max_tokens=10)
    engine = _Engine(
        mtp_config,
        params,
        init_mtp_params(jax.random.PRNGKey(1), mtp_config),
    )
    routes = []
    select_route = engine.model_runner._select_route

    def record_route(seqs, batch):
        route = select_route(seqs, batch)
        routes.append(route.kind)
        return route

    engine.model_runner._select_route = record_route
    actual = _generate(engine, max_tokens=10)

    assert actual == expected
    assert routes[:2] == [RouteKind.PREFILL_MTP, RouteKind.DECODE_SPECULATIVE]
    assert engine.model_runner.speculation_stats["rejected"] > 0
    assert engine.model_runner.speculation_stats["target_tokens"] > 0

    expected_stream = _generate(
        _Engine(base_config, params),
        max_tokens=8,
        prompt=(4, 5, 6),
    )
    route_start = len(routes)
    actual_stream = _generate(
        engine,
        max_tokens=8,
        prompt=(4, 5, 6),
        materialize_each_step=True,
    )

    assert actual_stream == expected_stream
    assert routes[route_start:].count(RouteKind.DECODE_SPECULATIVE) >= 2


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for verifier parity")
def test_packed_target_state_continues_after_reject_partial_and_full_accept():
    base_config = _runtime()
    mtp_config = _runtime(mtp=True)
    params = init_params(jax.random.PRNGKey(5), base_config.model)
    control = _Engine(base_config, params)
    engine = _Engine(
        mtp_config,
        params,
        init_mtp_params(jax.random.PRNGKey(6), mtp_config),
    )

    for accepted, prompt in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9))):
        expected = _generate(control, max_tokens=10, prompt=prompt)
        seq = engine.add_request(
            list(prompt),
            SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True),
        )
        assert engine.step().phase == "prefill"
        proposal = expected[1:3]
        if accepted < 2:
            proposal[accepted] = (proposal[accepted] + 1) % mtp_config.model.vocab_size
        _replace_slot_drafts(engine, seq, proposal)

        result = engine.step()

        assert result.num_emitted_tokens == accepted + 1
        assert result.draft_tokens == 2
        assert result.accepted_draft_tokens == accepted
        while not engine.is_finished():
            engine.step()
        assert seq.output.materialize() == expected


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for verifier parity")
def test_rejected_speculative_kv_is_safe_after_cancellation_and_slot_reuse():
    base_config = _runtime()
    config = _runtime(mtp=True)
    params = init_params(jax.random.PRNGKey(1), config.model)
    control = _Engine(base_config, params)
    first_expected = _generate(control, max_tokens=10)
    second_expected = _generate(control, max_tokens=8, prompt=(4, 5, 6))
    engine = _Engine(
        config,
        params,
        init_mtp_params(jax.random.PRNGKey(2), config),
    )
    first = engine.add_request(
        [1, 2, 3],
        SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True),
    )
    for _ in range(6):
        engine.step()
        if engine.model_runner.speculation_stats["rejected"]:
            break

    assert engine.model_runner.speculation_stats["rejected"] > 0
    assert engine.cancel_request(first)
    actual = _generate(engine, max_tokens=8, prompt=(4, 5, 6))

    assert actual == second_expected


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for MTP lifecycle")
def test_persistent_mtp_lifecycle_supports_two_rows():
    base_config = _runtime(max_seqs=2)
    mtp_config = _runtime(mtp=True, max_seqs=2)
    params = init_params(jax.random.PRNGKey(3), base_config.model)
    control = _Engine(base_config, params)
    first_expected = _generate(control, max_tokens=10, prompt=(1, 2, 3))
    second_expected = _generate(control, max_tokens=10, prompt=(4, 5, 6))
    third_expected = _generate(control, max_tokens=10, prompt=(7, 8, 9))
    engine = _Engine(
        mtp_config,
        params,
        init_mtp_params(jax.random.PRNGKey(4), mtp_config),
    )
    routes = []
    select_route = engine.model_runner._select_route

    def record_route(seqs, batch):
        route = select_route(seqs, batch)
        routes.append(route.kind)
        return route

    engine.model_runner._select_route = record_route
    sampling = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)
    first = engine.add_request([1, 2, 3], sampling)
    second = engine.add_request([4, 5, 6], sampling)
    assert engine.step().phase == "prefill"
    first_slot = engine.model_runner._hybrid_slots[first.seq_id]
    _replace_slot_drafts(engine, first, first_expected[1:3])
    rejected = second_expected[1:3]
    rejected[0] = (rejected[0] + 1) % mtp_config.model.vocab_size
    _replace_slot_drafts(engine, second, rejected)

    mixed = engine.step()
    emitted_by_seq = {
        seq_id: sum(event.seq_id == seq_id for event in mixed.emitted_tokens)
        for seq_id in (first.seq_id, second.seq_id)
    }

    assert emitted_by_seq == {first.seq_id: 3, second.seq_id: 1}
    assert mixed.draft_tokens == 4
    assert mixed.accepted_draft_tokens == 2
    assert engine.cancel_request(first)
    third = engine.add_request([7, 8, 9], sampling)
    assert engine.step().phase == "prefill"
    assert engine.model_runner._hybrid_slots[third.seq_id] == first_slot

    route_count = len(routes)
    engine.step()

    assert routes[route_count] is RouteKind.DECODE_SPECULATIVE
    while not engine.is_finished():
        engine.step()

    assert second.output.materialize() == second_expected
    assert third.output.materialize() == third_expected
    assert not engine.model_runner._mtp_ready_seq_ids
