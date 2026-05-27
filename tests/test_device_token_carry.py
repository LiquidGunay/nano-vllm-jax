"""Focused tests for deferred greedy token materialization."""

import jax.numpy as jnp

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.sequence import DeviceTokenRef, SamplingParams, Sequence


def test_sequence_materializes_deferred_device_tokens():
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


def test_scheduler_postprocess_can_defer_greedy_device_token(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", "1")
    scheduler = Scheduler(
        Qwen3_5Config(
            max_num_seqs=1,
            num_kvcache_blocks=4,
            max_blocks_per_seq=4,
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
