"""Online request queue for continuous batching.

Owns:
    HTTP-facing request admission and per-request result channels.
Receives:
    An engine with ``add_request()``, ``step()``, ``is_finished()``, and
    ``_detokenize()``.
Returns:
    Completed generation results or streaming token events.
Invariant:
    Handler threads enqueue work; only the service worker advances the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Iterable

from nanovllm_jax.sequence import SamplingParams


_STOP = object()
_DONE = object()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]


@dataclass(frozen=True)
class _PendingRequest:
    prompt: str | list[int]
    sampling_params: SamplingParams
    handle: "RequestHandle"


@dataclass
class _ActiveRequest:
    seq: Any
    handle: "RequestHandle"
    seen_completion_tokens: int = 0


class RequestHandle:
    """Result and event channel for one submitted request."""

    def __init__(self, request_id: int, *, stream: bool = False):
        self.request_id = int(request_id)
        self.stream = bool(stream)
        self.seq_id: int | None = None
        self._done = threading.Event()
        self._events: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._result: GenerationResult | None = None
        self._exception: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> GenerationResult:
        if not self._done.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish before timeout")
        if self._exception is not None:
            raise self._exception
        if self._result is None:
            raise RuntimeError(f"request {self.request_id} finished without a result")
        return self._result

    def events(self) -> Iterable[dict[str, Any]]:
        while True:
            item = self._events.get()
            if item is _DONE:
                if self._exception is not None:
                    raise self._exception
                return
            yield item

    def _set_seq_id(self, seq_id: int) -> None:
        self.seq_id = int(seq_id)

    def _publish(self, event: dict[str, Any]) -> None:
        self._events.put(event)

    def _finish(self, result: GenerationResult) -> None:
        self._result = result
        self._events.put(
            {
                "event": "done",
                "request_id": self.request_id,
                "seq_id": self.seq_id,
                "result": {"text": result.text, "token_ids": result.token_ids},
            }
        )
        self._events.put(_DONE)
        self._done.set()

    def _fail(self, exc: BaseException) -> None:
        self._exception = exc
        self._events.put(
            {
                "event": "error",
                "request_id": self.request_id,
                "seq_id": self.seq_id,
                "error": str(exc),
            }
        )
        self._events.put(_DONE)
        self._done.set()


class EngineService:
    """Single-worker service loop that batches requests across callers."""

    def __init__(
        self,
        engine: Any,
        *,
        engine_lock: threading.Lock | threading.RLock | None = None,
        batch_window_seconds: float = 0.002,
    ):
        self.engine = engine
        self.engine_lock = engine_lock or threading.RLock()
        self.batch_window_seconds = max(0.0, float(batch_window_seconds))
        self._incoming: queue.Queue[_PendingRequest | object] = queue.Queue()
        self._active: dict[int, _ActiveRequest] = {}
        self._next_request_id = 0
        self._request_id_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="nanovllm-engine-service", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        self._incoming.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_idle(self) -> bool:
        return not self._active and self._incoming.empty()

    def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        stream: bool = False,
    ) -> RequestHandle:
        with self._request_id_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
        handle = RequestHandle(request_id, stream=stream)
        self._incoming.put(_PendingRequest(prompt=prompt, sampling_params=sampling_params, handle=handle))
        return handle

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
        handles = [self.submit(prompt, param) for prompt, param in zip(prompts, params)]
        return [handle.wait() for handle in handles]

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._active:
                item = self._blocking_get()
                if item is _STOP:
                    break
                if item is not None:
                    self._admit_pending([item])
                    self._collect_batch_window()
            else:
                self._drain_available()

            if not self._active:
                continue

            try:
                with self.engine_lock:
                    outputs, _ = self.engine.step()
            except BaseException as exc:
                self._fail_all_active(exc)
                continue

            self._publish_progress()
            self._publish_finished(outputs)

    def _blocking_get(self) -> _PendingRequest | object | None:
        try:
            return self._incoming.get(timeout=0.05)
        except queue.Empty:
            return None

    def _collect_batch_window(self) -> None:
        if self.batch_window_seconds <= 0.0:
            self._drain_available()
            return
        deadline = time.perf_counter() + self.batch_window_seconds
        while True:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                item = self._incoming.get(timeout=timeout)
            except queue.Empty:
                break
            if item is _STOP:
                self._stop.set()
                break
            self._admit_pending([item])

    def _drain_available(self) -> None:
        pending: list[_PendingRequest] = []
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                self._stop.set()
                continue
            pending.append(item)
        if pending:
            self._admit_pending(pending)

    def _admit_pending(self, pending: list[_PendingRequest]) -> None:
        with self.engine_lock:
            for request in pending:
                try:
                    seq = self.engine.add_request(request.prompt, request.sampling_params)
                except BaseException as exc:
                    request.handle._fail(exc)
                    continue
                request.handle._set_seq_id(int(seq.seq_id))
                self._active[int(seq.seq_id)] = _ActiveRequest(seq=seq, handle=request.handle)

    def _publish_progress(self) -> None:
        for seq_id, active in list(self._active.items()):
            if not active.handle.stream:
                continue
            seq = active.seq
            if not hasattr(seq, "completion_token_ids"):
                continue
            completion = list(seq.completion_token_ids)
            if len(completion) <= active.seen_completion_tokens:
                continue
            for completion_index, token_id in enumerate(
                completion[active.seen_completion_tokens:],
                start=active.seen_completion_tokens,
            ):
                event = {
                    "event": "token",
                    "request_id": active.handle.request_id,
                    "seq_id": seq_id,
                    "completion_index": completion_index,
                    "token_id": int(token_id),
                }
                detokenize = getattr(self.engine, "_detokenize", None)
                if detokenize is not None:
                    event["text"] = detokenize([int(token_id)])
                active.handle._publish(event)
            active.seen_completion_tokens = len(completion)

    def _publish_finished(self, outputs: list[tuple[int, list[int]]]) -> None:
        for seq_id, token_ids in outputs:
            active = self._active.pop(int(seq_id), None)
            if active is None:
                continue
            token_ids = [int(token) for token in token_ids]
            detokenize = getattr(self.engine, "_detokenize", None)
            text = detokenize(token_ids) if detokenize is not None else ""
            active.handle._finish(GenerationResult(text=text, token_ids=token_ids))

    def _fail_all_active(self, exc: BaseException) -> None:
        for active in self._active.values():
            active.handle._fail(exc)
        self._active.clear()
