from nanovllm_jax.config import RuntimeConfig
from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams, Sequence
from nanovllm_jax.step import FinishReason, RunResult


class _Runner:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.released = []

    def install_cached_prefix_hybrid_states(self, seqs, states):
        pass

    def materialize(self, plan):
        return plan

    def execute(self, seqs, batch):
        return RunResult.from_rows(next(self.rows))

    def release(self, seq_ids):
        self.released.extend(seq_ids)


def test_step_executes_then_commits_one_typed_transition():
    config = RuntimeConfig(
        block_size=2,
        num_kvcache_blocks=2,
        max_kv_cache_bytes=1 << 20,
        max_num_seqs=1,
        max_num_resident_seqs=1,
        max_num_batched_tokens=2,
        max_blocks_per_seq=2,
        prefill_buckets=(2,),
        prefill_token_buckets=(2,),
        batch_size_buckets=(1,),
        decode_block_table_buckets=(2,),
        prefix_cache=False,
        device_token_carry=False,
    )
    engine = object.__new__(LLMEngine)
    engine.scheduler = Scheduler(config)
    engine.model_runner = _Runner([[101], [102]])
    seq = Sequence(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        block_size=2,
    )
    engine.scheduler.add(seq)

    prefill = engine.step()
    decode = engine.step()

    assert prefill.phase == "prefill"
    assert prefill.scheduled_tokens == 2
    assert prefill.emitted_tokens[0].completion_index == 0
    assert not prefill.finished
    assert decode.phase == "decode"
    assert decode.emitted_tokens[0].completion_index == 1
    assert decode.finished[0].reason is FinishReason.LENGTH
    assert seq.output.token_ids() == [101, 102]
    assert engine.model_runner.released == [seq.seq_id]
    assert engine.is_finished()


def test_cancel_request_releases_scheduler_and_runner_state():
    config = RuntimeConfig(
        block_size=2,
        num_kvcache_blocks=2,
        max_kv_cache_bytes=1 << 20,
        max_num_seqs=1,
        max_num_resident_seqs=1,
        max_num_batched_tokens=2,
        max_blocks_per_seq=2,
        prefix_cache=False,
    )
    engine = object.__new__(LLMEngine)
    engine.scheduler = Scheduler(config)
    engine.model_runner = _Runner([])
    seq = Sequence(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        block_size=2,
    )
    engine.scheduler.add(seq)

    assert engine.cancel_request(seq) is True
    assert engine.cancel_request(seq) is False
    assert seq.is_finished
    assert engine.scheduler.is_finished()
    assert engine.model_runner.released == [seq.seq_id]
