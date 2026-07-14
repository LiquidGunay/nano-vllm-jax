import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.model import init_params
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.sequence import Sequence
from nanovllm_jax.speculation import (
    DraftProposal,
    SuppliedDrafter,
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


def test_supplied_drafter_uses_each_sequence_completion_cursor():
    first = Sequence([1], seq_id=4)
    second = Sequence([2], seq_id=7)
    first.output.append(100)
    first.output.append(101)
    drafter = SuppliedDrafter(
        {
            4: [100, 101, 102, 103, 104],
            7: [200, 201, 202],
        },
        width=2,
    )

    proposal = drafter.propose([first, second])

    np.testing.assert_array_equal(
        proposal.token_ids,
        [[102, 103], [200, 201]],
    )


def test_verifier_requires_one_bonus_target():
    proposal = DraftProposal(jnp.zeros((1, 2), dtype=jnp.int32))
    with pytest.raises(ValueError, match="one bonus token"):
        verify_greedy_drafts(proposal, jnp.zeros((1, 2), dtype=jnp.int32))


def _has_cuda():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def _runtime():
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
            "max_num_seqs": 1,
            "max_num_resident_seqs": 1,
            "max_num_batched_tokens": 3,
            "max_blocks_per_seq": 16,
            "prefix_cache": False,
        },
        compile={
            "dtype": "float32",
            "execution": "jit",
            "prefill_token_buckets": (3,),
            "batch_size_buckets": (1,),
            "decode_block_table_buckets": (16,),
        },
        kernels={
            "device_token_carry": True,
            "static_decode_metadata": True,
            "resident_decode_metadata": True,
        },
    )


class _Engine(LLMEngine):
    def __init__(self, config, params):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config, params)
        self._next_seq_id = 0


def _generate(engine, *, max_tokens, prompt=(1, 2, 3)):
    seq = engine.add_request(
        list(prompt),
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
    )
    while not engine.is_finished():
        engine.step()
    return seq.output.materialize()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for verifier parity")
def test_packed_verifier_matches_decode_across_accept_and_reject_paths():
    config = _runtime()
    params = init_params(jax.random.PRNGKey(0), config.model)
    expected = _generate(_Engine(config, params), max_tokens=10)
    supplied = list(expected)
    supplied[4] = (supplied[4] + 1) % config.model.vocab_size

    engine = _Engine(config, params)
    engine.install_drafter(SuppliedDrafter({0: supplied}, width=2))
    actual = _generate(engine, max_tokens=10)

    assert actual == expected
    assert engine.model_runner.speculation_stats["accepted"] > 0
    assert engine.model_runner.speculation_stats["rejected"] > 0
    assert engine.model_runner.speculation_stats["bonus"] > 0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for verifier parity")
def test_rejected_speculative_kv_is_safe_after_cancellation_and_slot_reuse():
    config = _runtime()
    params = init_params(jax.random.PRNGKey(1), config.model)
    control = _Engine(config, params)
    first_expected = _generate(control, max_tokens=10)
    second_expected = _generate(control, max_tokens=8, prompt=(4, 5, 6))
    first_drafts = list(first_expected)
    first_drafts[4] = (first_drafts[4] + 1) % config.model.vocab_size

    engine = _Engine(config, params)
    engine.install_drafter(
        SuppliedDrafter({0: first_drafts, 1: second_expected}, width=2)
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
