"""Bounded online admission for the single-writer engine."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Iterable

from nanovllm_jax.output import OutputBuffer
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.step import FinishReason, StepResult


_STOP = object()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: FinishReason


@dataclass(frozen=True)
class _PendingRequest:
    prompt: str | list[int]
    sampling_params: SamplingParams
    handle: "RequestHandle"


@dataclass
class _ActiveRequest:
    seq: Any
    handle: "RequestHandle"


def _text_delta(decoded: str, previous: str, terminal: bool) -> tuple[str, str]:
    stable = decoded if terminal else decoded.rstrip("\ufffd")
    if not stable.startswith(previous):
        return "", previous
    return stable[len(previous):], stable


class RequestHandle:
    """One result and a one-slot notification channel for streaming."""

    def __init__(self, request_id: int, *, stream: bool = False):
        self.request_id = int(request_id)
        self.stream = bool(stream)
        self.seq_id: int | None = None
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._wake: queue.Queue[object] = queue.Queue(maxsize=1)
        self._output: OutputBuffer | None = None
        self._detokenize: Callable[[list[int]], str] | None = None
        self._stream_through = 0
        self._events_started = False
        self._result: GenerationResult | None = None
        self._exception: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> bool:
        """Ask the engine worker to cancel this request."""
        if self.done:
            return False
        self._cancel.set()
        return True

    def wait(self, timeout: float | None = None) -> GenerationResult:
        if not self._done.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish before timeout")
        if self._exception is not None:
            raise self._exception
        if self._result is None:
            raise RuntimeError(f"request {self.request_id} finished without a result")
        return self._result

    def events(self) -> Iterable[dict[str, Any]]:
        if not self.stream:
            raise RuntimeError("request was not submitted for streaming")
        if self._events_started:
            raise RuntimeError("request events may be consumed only once")
        self._events_started = True
        return self._iter_events()

    def _iter_events(self) -> Iterable[dict[str, Any]]:
        cursor = 0
        streamed_text = ""
        try:
            while True:
                self._wake.get()
                available = self._stream_through
                terminal = self.done

                if cursor < available:
                    if self._output is None:
                        raise RuntimeError("streaming request has no output buffer")
                    token_ids = [self._output.token_id(index) for index in range(available)]
                    event = {
                        "event": "tokens",
                        "request_id": self.request_id,
                        "seq_id": self.seq_id,
                        "completion_start": cursor,
                        "token_ids": token_ids[cursor:],
                    }
                    if self._detokenize is not None:
                        delta, streamed_text = _text_delta(
                            self._detokenize(token_ids), streamed_text, terminal
                        )
                        event["text"] = delta
                    yield event
                    cursor = available

                if not terminal:
                    continue
                if self._exception is not None:
                    yield {
                        "event": "error",
                        "request_id": self.request_id,
                        "seq_id": self.seq_id,
                        "error": str(self._exception),
                    }
                elif self._result is not None:
                    yield {
                        "event": "done",
                        "request_id": self.request_id,
                        "seq_id": self.seq_id,
                        "result": {
                            "text": self._result.text,
                            "token_ids": self._result.token_ids,
                            "finish_reason": self._result.finish_reason.value,
                        },
                    }
                return
        finally:
            if not self.done:
                self.cancel()

    def _bind(
        self,
        seq_id: int,
        output: OutputBuffer,
        detokenize: Callable[[list[int]], str] | None,
    ) -> None:
        self.seq_id = int(seq_id)
        self._output = output
        self._detokenize = detokenize

    def _notify(self) -> None:
        try:
            self._wake.put_nowait(None)
        except queue.Full:
            pass

    def _publish_through(self, completion_count: int) -> None:
        self._stream_through = max(self._stream_through, int(completion_count))
        self._notify()

    def _finish(self, result: GenerationResult) -> bool:
        if self.done:
            return False
        self._result = result
        self._done.set()
        self._notify()
        return True

    def _fail(self, exc: BaseException) -> bool:
        if self.done:
            return False
        self._exception = exc
        self._done.set()
        self._notify()
        return True


class EngineService:
    """Bounded request set advanced by one engine-worker thread."""

    def __init__(
        self,
        engine: Any,
        *,
        engine_lock: threading.Lock | threading.RLock | None = None,
        batch_window_seconds: float = 0.002,
        max_queue_size: int | None = 256,
    ):
        self.engine = engine
        self.engine_lock = engine_lock or threading.RLock()
        self.batch_window_seconds = max(0.0, float(batch_window_seconds))
        if max_queue_size is not None and int(max_queue_size) <= 0:
            raise ValueError("max_queue_size must be positive when provided")
        self._max_requests = None if max_queue_size is None else int(max_queue_size)
        self._incoming: queue.Queue[_PendingRequest | object] = queue.Queue()
        self._active: dict[int, _ActiveRequest] = {}
        self._next_request_id = 0
        self._state_lock = threading.Lock()
        self._in_flight = 0
        self._stop = threading.Event()
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._failed is not None:
                raise RuntimeError("engine service is failed and must be recreated")
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="nanovllm-engine-service", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        with self._state_lock:
            self._stop.set()
        self._incoming.put(_STOP)
        thread = self._thread
        if thread is threading.current_thread():
            raise RuntimeError("engine service worker cannot join itself")
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("engine service worker did not stop before timeout")

        stopped = RuntimeError("engine service stopped before request completed")
        self._abort_all_active(stopped)
        self._fail_all_pending(stopped)
        with self._state_lock:
            if self._thread is thread:
                self._thread = None

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            thread = self._thread
            alive = thread is not None and thread.is_alive()
            failed = self._failed
            stopping = self._stop.is_set()
            if failed is not None:
                state = "failed"
            elif alive:
                state = "stopping" if stopping else "running"
            else:
                state = "stopped"
            return {
                "state": state,
                "worker_alive": alive,
                "in_flight": self._in_flight,
                "error": str(failed) if failed is not None else None,
            }

    def is_idle(self) -> bool:
        with self._state_lock:
            return self._in_flight == 0

    def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        stream: bool = False,
    ) -> RequestHandle:
        return self._submit_many([prompt], [sampling_params], stream=stream)[0]

    def _submit_many(
        self,
        prompts: list[str | list[int]],
        params: list[SamplingParams],
        *,
        stream: bool = False,
    ) -> list[RequestHandle]:
        with self._state_lock:
            if self._failed is not None:
                raise RuntimeError("engine service failed; restart the server") from self._failed
            if self._stop.is_set():
                raise RuntimeError("engine service is stopping")
            count = len(prompts)
            if self._max_requests is not None and self._in_flight + count > self._max_requests:
                raise RuntimeError("engine service queue is full")
            handles = [
                RequestHandle(self._next_request_id + index, stream=stream)
                for index in range(count)
            ]
            self._next_request_id += count
            self._in_flight += count
            for prompt, param, handle in zip(prompts, params, handles):
                self._incoming.put(_PendingRequest(prompt, param, handle))
            return handles

    def generate(self, prompt: str | list[int], sampling_params: SamplingParams) -> GenerationResult:
        return self.submit(prompt, sampling_params).wait()

    def generate_many(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[GenerationResult]:
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError("sampling_params length must match prompts length")
            params = sampling_params
        else:
            params = [sampling_params for _ in prompts]
        handles = self._submit_many(prompts, params)
        return [handle.wait() for handle in handles]

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                pending = self._take_pending(block=not self._active)
                if pending:
                    self._admit_pending(pending)
                self._cancel_requested()
                if self._stop.is_set() or not self._active:
                    continue
                with self.engine_lock:
                    step_result = self.engine.step()
                self._publish_progress(step_result)
                self._publish_finished(step_result)
        except BaseException as exc:
            self._mark_failed(exc)
        finally:
            stopped = RuntimeError("engine service stopped before request completed")
            self._abort_all_active(stopped)
            self._fail_all_pending(stopped)

    def _take_pending(self, *, block: bool) -> list[_PendingRequest]:
        pending: list[_PendingRequest] = []
        deadline = None
        if block:
            try:
                item = self._incoming.get(timeout=0.05)
            except queue.Empty:
                return pending
            if item is _STOP:
                self._stop.set()
                return pending
            pending.append(item)
            deadline = time.perf_counter() + self.batch_window_seconds

        while True:
            try:
                if deadline is not None and deadline > time.perf_counter():
                    item = self._incoming.get(timeout=deadline - time.perf_counter())
                else:
                    item = self._incoming.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                self._stop.set()
            else:
                pending.append(item)
        return pending

    def _admit_pending(self, pending: list[_PendingRequest]) -> None:
        detokenize = getattr(self.engine, "_detokenize", None)
        with self.engine_lock:
            for request in pending:
                if request.handle.cancel_requested:
                    self._finish_handle(
                        request.handle, GenerationResult("", [], FinishReason.CANCELLED)
                    )
                    continue
                try:
                    seq = self.engine.add_request(request.prompt, request.sampling_params)
                except BaseException as exc:
                    self._fail_handle(request.handle, exc)
                    continue
                request.handle._bind(seq.seq_id, seq.output, detokenize)
                self._active[int(seq.seq_id)] = _ActiveRequest(seq, request.handle)

    def _cancel_requested(self) -> None:
        cancelled = [
            (seq_id, active)
            for seq_id, active in self._active.items()
            if active.handle.cancel_requested
        ]
        if not cancelled:
            return
        OutputBuffer.materialize_many(active.seq.output for _, active in cancelled)
        with self.engine_lock:
            for _, active in cancelled:
                self.engine.cancel_request(active.seq)
        for seq_id, active in cancelled:
            self._finish_active(active, FinishReason.CANCELLED)
            self._active.pop(seq_id, None)

    def _publish_progress(self, step: StepResult) -> None:
        through: dict[int, int] = {}
        for token in step.emitted_tokens:
            active = self._active.get(token.seq_id)
            if active is not None and active.handle.stream:
                through[token.seq_id] = max(
                    through.get(token.seq_id, 0), token.completion_index + 1
                )
        OutputBuffer.materialize_many(
            self._active[seq_id].seq.output for seq_id in through
        )
        for seq_id, count in through.items():
            self._active[seq_id].handle._publish_through(count)

    def _publish_finished(self, step: StepResult) -> None:
        finished = [
            (request, self._active.get(request.seq_id)) for request in step.finished
        ]
        OutputBuffer.materialize_many(
            active.seq.output for _, active in finished if active is not None
        )
        for request, _ in finished:
            active = self._active.get(request.seq_id)
            if active is not None:
                self._finish_active(active, request.reason)
                self._active.pop(request.seq_id, None)

    def _finish_active(self, active: _ActiveRequest, reason: FinishReason) -> None:
        token_ids = active.seq.output.token_ids()
        detokenize = getattr(self.engine, "_detokenize", None)
        text = detokenize(token_ids) if detokenize is not None else ""
        self._finish_handle(active.handle, GenerationResult(text, token_ids, reason))

    def _finish_handle(self, handle: RequestHandle, result: GenerationResult) -> None:
        if handle._finish(result):
            self._release_request()

    def _fail_handle(self, handle: RequestHandle, exc: BaseException) -> None:
        if handle._fail(exc):
            self._release_request()

    def _release_request(self) -> None:
        with self._state_lock:
            if self._in_flight <= 0:
                raise AssertionError("service request accounting underflow")
            self._in_flight -= 1

    def _abort_all_active(self, exc: BaseException) -> None:
        active = list(self._active.values())
        self._active.clear()
        with self.engine_lock:
            for request in active:
                try:
                    self.engine.cancel_request(request.seq)
                except BaseException:
                    pass
        for request in active:
            self._fail_handle(request.handle, exc)

    def _fail_all_pending(self, exc: BaseException) -> None:
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _PendingRequest):
                self._fail_handle(item.handle, exc)

    def _mark_failed(self, exc: BaseException) -> None:
        with self._state_lock:
            self._failed = exc
            self._stop.set()
        self._abort_all_active(exc)
        self._fail_all_pending(exc)
