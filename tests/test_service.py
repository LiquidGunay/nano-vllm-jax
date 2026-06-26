from dataclasses import dataclass, field

from nanovllm_jax.service import EngineService
from nanovllm_jax.sequence import SamplingParams


@dataclass
class _FakeSeq:
    seq_id: int
    sampling_params: SamplingParams
    completion: list[int] = field(default_factory=list)
    is_finished: bool = False

    @property
    def completion_token_ids(self):
        return list(self.completion)


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
        outputs = []
        for seq in active:
            seq.completion.append(100 + seq.seq_id + len(seq.completion))
            if len(seq.completion) >= seq.sampling_params.max_tokens:
                seq.is_finished = True
                outputs.append((seq.seq_id, list(seq.completion)))
        return outputs, -len(active)

    def is_finished(self):
        return all(seq.is_finished for seq in self.seqs)

    def _detokenize(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


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
        assert (0, 1) in engine.step_batches
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

        assert events[0]["event"] == "token"
        assert events[0]["token_id"] == 100
        assert events[-1]["event"] == "done"
        assert handle.wait(timeout=1.0).token_ids == [100]
    finally:
        service.stop()
