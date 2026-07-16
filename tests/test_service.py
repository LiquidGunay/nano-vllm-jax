from dataclasses import dataclass, field
import threading

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
        self.cancelled: list[int] = []
        self.control_owner = None

    def claim_control(self, owner):
        if self.control_owner is not None:
            raise RuntimeError("engine already has an active control owner")
        self.control_owner = owner

    def release_control(self, owner):
        assert self.control_owner is owner
        self.control_owner = None

    def add_request(self, prompt, sampling_params, *, owner=None):
        assert owner is self.control_owner
        seq = _FakeSeq(seq_id=self._next_seq_id, sampling_params=sampling_params)
        self._next_seq_id += 1
        self.seqs.append(seq)
        return seq

    def step(self, *, owner=None):
        assert owner is self.control_owner
        active = [seq for seq in self.seqs if not seq.is_finished]
        self.step_batches.append(tuple(seq.seq_id for seq in active))
        emitted = []
        finished = []
        for seq in active:
            token_id = self.next_token(seq)
            index = seq.output.append(token_id)
            emitted.append(TokenEvent(seq.seq_id, index, token_id))
            if len(seq.output) >= seq.sampling_params.max_tokens:
                seq.is_finished = True
                finished.append(FinishedRequest(seq.seq_id, FinishReason.LENGTH))
        return StepResult("decode", len(active), tuple(emitted), tuple(finished))

    def next_token(self, seq):
        return 100 + seq.seq_id + len(seq.output)

    def cancel_request(self, seq, *, owner=None):
        assert owner is self.control_owner
        if seq.is_finished:
            return False
        seq.is_finished = True
        self.cancelled.append(seq.seq_id)
        return True

    def is_finished(self):
        return all(seq.is_finished for seq in self.seqs)

    def detokenize(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class _FailingEngine(_FakeEngine):
    def step(self, *, owner=None):
        assert owner is self.control_owner
        raise RuntimeError("accelerator failed")


class _BlockingEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def step(self, *, owner=None):
        assert owner is self.control_owner
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test engine remained blocked")
        return super().step(owner=owner)


class _PacedUnicodeEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.permits = threading.Semaphore(0)

    def step(self, *, owner=None):
        assert owner is self.control_owner
        if not self.permits.acquire(timeout=2.0):
            raise TimeoutError("test engine was not released")
        return super().step(owner=owner)

    def next_token(self, seq):
        return (1, 2, 3)[len(seq.output)]

    def detokenize(self, token_ids):
        return {
            (1,): "\ufffd",
            (1, 2): "é",
            (1, 2, 3): "é!",
        }[tuple(token_ids)]


class _BadDetokenizer:
    def detokenize(self, token_ids):
        raise ValueError("decode failed")


class _BadFinalEngine(_BadDetokenizer, _FakeEngine):
    pass


class _BlockingBadFinalEngine(_BadDetokenizer, _BlockingEngine):
    pass


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
        health = service.health()
        assert health["state"] == "failed"
        assert health["error"] == "accelerator failed"
    finally:
        service.stop()


def test_natural_finalization_failure_terminates_handle():
    service = EngineService(_BadFinalEngine(), batch_window_seconds=0.0)
    service.start()
    try:
        handle = service.submit(
            [11],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )
        with pytest.raises(ValueError, match="decode failed"):
            handle.wait(timeout=1.0)
        assert service.health()["in_flight"] == 0
    finally:
        service.stop()


def test_cancel_finalization_failure_terminates_handle():
    engine = _BlockingBadFinalEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    handle = service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
    )
    try:
        assert engine.entered.wait(timeout=1.0)
        handle.cancel()
        engine.release.set()
        with pytest.raises(ValueError, match="decode failed"):
            handle.wait(timeout=1.0)
        assert service.health()["in_flight"] == 0
    finally:
        engine.release.set()
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


def test_service_owns_engine_control_until_stopped():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)

    service.start()
    assert engine.control_owner is service
    with pytest.raises(RuntimeError, match="active control owner"):
        engine.claim_control(object())

    service.stop()
    assert engine.control_owner is None


def test_service_rejects_when_queue_is_full():
    engine = _BlockingEngine()
    service = EngineService(engine, batch_window_seconds=0.0, max_queue_size=1)
    service.start()
    try:
        first = service.submit(
            [11],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )
        assert engine.entered.wait(timeout=1.0)

        with pytest.raises(RuntimeError, match="queue is full"):
            service.submit(
                [22],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            )
        engine.release.set()
        assert first.wait(timeout=1.0).finish_reason is FinishReason.LENGTH
    finally:
        engine.release.set()
        service.stop()


def test_generate_many_reserves_its_whole_batch_atomically():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0, max_queue_size=1)
    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    with pytest.raises(RuntimeError, match="queue is full"):
        service.generate_many([[11], [22]], sampling)
    assert service.health()["in_flight"] == 0

    service.start()
    try:
        handle = service.submit([33], sampling)
        assert handle.request_id == 0
        assert handle.wait(timeout=1.0).finish_reason is FinishReason.LENGTH
        assert len(engine.seqs) == 1
    finally:
        service.stop()


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

        assert events[0]["event"] == "tokens"
        assert events[0]["completion_start"] == 0
        assert events[0]["token_ids"] == [100]
        assert events[-1]["event"] == "done"
        assert events[-1]["result"]["finish_reason"] == "length"
        assert handle.wait(timeout=1.0).token_ids == [100]
    finally:
        service.stop()


def test_slow_stream_consumer_observes_one_coalesced_notification():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    try:
        handle = service.submit(
            [11],
            SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
            stream=True,
        )
        result = handle.wait(timeout=1.0)

        assert handle._wake.maxsize == handle._wake.qsize() == 1
        events = list(handle.events())
        chunks = [event for event in events if event["event"] == "tokens"]
        assert len(chunks) == 1
        assert chunks[0]["token_ids"] == result.token_ids
    finally:
        service.stop()


def test_stream_close_cancels_active_request():
    engine = _PacedUnicodeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    handle = service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        stream=True,
    )
    try:
        events = handle.events()
        engine.permits.release()
        assert next(events)["event"] == "tokens"
        events.close()
        engine.permits.release()

        result = handle.wait(timeout=1.0)
        assert result.finish_reason is FinishReason.CANCELLED
        assert engine.cancelled == [handle.seq_id]
    finally:
        engine.permits.release()
        service.stop()


def test_cancelled_pending_request_is_never_admitted():
    engine = _FakeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    handle = service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )
    handle.cancel()
    service.start()
    try:
        result = handle.wait(timeout=1.0)
        assert result.finish_reason is FinishReason.CANCELLED
        assert result.token_ids == []
        assert engine.seqs == []
    finally:
        service.stop()


def test_stream_text_waits_for_complete_unicode():
    engine = _PacedUnicodeEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    handle = service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        stream=True,
    )
    events = handle.events()
    try:
        engine.permits.release()
        first = next(events)
        engine.permits.release()
        second = next(events)
        engine.permits.release()
        third = next(events)
        done = next(events)

        assert first["text"] == ""
        assert second["text"] == "é"
        assert third["text"] == "!"
        assert done["result"]["text"] == "é!"
    finally:
        engine.permits.release()
        events.close()
        service.stop()


def test_stop_reports_a_worker_that_is_still_running():
    engine = _BlockingEngine()
    service = EngineService(engine, batch_window_seconds=0.0)
    service.start()
    handle = service.submit(
        [11],
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
    )
    try:
        assert engine.entered.wait(timeout=1.0)
        with pytest.raises(TimeoutError, match="did not stop"):
            service.stop(timeout=0.01)
        assert service.health()["state"] == "stopping"
    finally:
        engine.release.set()

    with pytest.raises(RuntimeError, match="stopped"):
        handle.wait(timeout=1.0)
    assert service.health()["in_flight"] == 0
    service.stop(timeout=1.0)
    assert service.health()["state"] == "stopped"
    assert service.health()["worker_alive"] is False
