from dataclasses import dataclass, field

import pytest

from nanovllm_jax.output import OutputBuffer
from nanovllm_jax.service import EngineService
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.step import FinishedRequest, FinishReason, StepResult, TokenEvent


@dataclass
class _FakeSeq:
    seq_id: int
    sampling_params: SamplingParams
    output: OutputBuffer = field(default_factory=OutputBuffer)
    is_finished: bool = False


class _FakeEngine:
    def __init__(self):
        self._next_seq_id = 0
        self.seqs: list[_FakeSeq] = []
        self.step_batches: list[tuple[int, ...]] = []

    def add_request(self, prompt, sampling_params):
        seq = _FakeSeq(seq_id=self._next_seq_id, sampling_params=sampling_params)
        self._next_seq_id += 1
        self.seqs.append(seq)
        return seq

    def step(self):
        active = [seq for seq in self.seqs if not seq.is_finished]
        self.step_batches.append(tuple(seq.seq_id for seq in active))
        emitted = []
        finished = []
        for seq in active:
            token_id = 100 + seq.seq_id + len(seq.output)
            index = seq.output.append(token_id)
            emitted.append(TokenEvent(seq.seq_id, index, token_id))
            if len(seq.output) >= seq.sampling_params.max_tokens:
                seq.is_finished = True
                finished.append(FinishedRequest(seq.seq_id, FinishReason.LENGTH))
        return StepResult("decode", len(active), tuple(emitted), tuple(finished))

    def is_finished(self):
        return all(seq.is_finished for seq in self.seqs)

    def _detokenize(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class _FailingEngine(_FakeEngine):
    def step(self):
        raise RuntimeError("accelerator failed")


def test_service_admits_independent_requests_into_same_engine_step():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.02)
    service.start()
    try:
        sampling = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
        first = service.submit([11], sampling)
        second = service.submit([22], sampling)

        first_result = first.wait(timeout=1.0)
        second_result = second.wait(timeout=1.0)

        assert first_result.token_ids == [100, 101]
        assert second_result.token_ids == [101, 102]
        assert first_result.finish_reason is FinishReason.LENGTH
        assert second_result.finish_reason is FinishReason.LENGTH
        assert (0, 1) in engine.step_batches
    finally:
        service.stop()


def test_service_engine_failure_fails_active_pending_and_future_requests():
    engine = _FailingEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    try:
        handle = service.submit(
            [11],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )

        with pytest.raises(RuntimeError, match="accelerator failed"):
            handle.wait(timeout=1.0)
        with pytest.raises(RuntimeError, match="engine service failed"):
            service.submit(
                [22],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            )
    finally:
        service.stop()


def test_service_stop_fails_pending_requests_before_start():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    handle = service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )

    service.stop()

    with pytest.raises(RuntimeError, match="stopped"):
        handle.wait(timeout=0.1)


def test_service_rejects_when_queue_is_full():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0, max_queue_size=1)
    service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )

    with pytest.raises(RuntimeError, match="queue is full"):
        service.submit(
            [22],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )


def test_service_streams_token_events_before_done():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    try:
        handle = service.submit(
            [11],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            stream=True,
        )
        events = list(handle.events())

        assert events[0]["event"] == "token"
        assert events[0]["token_id"] == 100
        assert events[-1]["event"] == "done"
        assert events[-1]["result"]["finish_reason"] == "length"
        assert handle.wait(timeout=1.0).token_ids == [100]
    finally:
        service.stop()
