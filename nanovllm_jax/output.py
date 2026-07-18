"""Generated-token storage and explicit host materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DeviceTokenRef:
    """One row in a device-resident token vector."""

    tokens: Any
    row: int


def is_device_token(value: Any) -> bool:
    return isinstance(value, DeviceTokenRef)


@dataclass(frozen=True)
class OutputSlot:
    """A stable snapshot of one deferred output token."""

    buffer: "OutputBuffer"
    index: int
    token: DeviceTokenRef


class OutputBuffer:
    """Own generated tokens without hiding accelerator synchronization."""

    def __init__(self) -> None:
        self._token_ids: list[int] = []
        self._deferred: dict[int, DeviceTokenRef] = {}

    def __len__(self) -> int:
        return len(self._token_ids)

    def append(self, token_id: int) -> int:
        self._token_ids.append(int(token_id))
        return len(self._token_ids) - 1

    def append_device(self, token: DeviceTokenRef) -> int:
        if not isinstance(token, DeviceTokenRef):
            raise TypeError("deferred output tokens must use DeviceTokenRef")
        index = len(self._token_ids)
        self._token_ids.append(0)
        self._deferred[index] = token
        return index

    @property
    def last_token(self) -> int:
        return self._token_ids[-1]

    @property
    def last_device_token(self) -> DeviceTokenRef | None:
        return self._deferred.get(len(self._token_ids) - 1)

    @property
    def has_deferred_tokens(self) -> bool:
        return bool(self._deferred)

    def has_deferred_between(self, start: int, end: int) -> bool:
        return any(start <= index < end for index in self._deferred)

    def logical_token_ids(self) -> list[int]:
        """Return host values plus zero placeholders without synchronizing."""
        return list(self._token_ids)

    def materialized_prefix(self) -> list[int]:
        end = min(self._deferred, default=len(self._token_ids))
        return list(self._token_ids[:end])

    def token_id(self, index: int) -> int:
        if index in self._deferred:
            raise RuntimeError("output token is on device; materialize it first")
        return int(self._token_ids[index])

    def token_ids(self) -> list[int]:
        if self._deferred:
            raise RuntimeError("output tokens are on device; materialize them first")
        return list(self._token_ids)

    def snapshot(self, start: int = 0) -> "OutputSnapshot":
        return OutputSnapshot(
            tuple(
                OutputSlot(self, index, token)
                for index, token in self._deferred.items()
                if index >= start
            )
        )

    def prefetch(self) -> "OutputSnapshot":
        return self.snapshot().prefetch()

    def materialize(self) -> list[int]:
        """Synchronize all current output tokens and return host ids."""
        self.snapshot().prefetch().materialize()
        return self.token_ids()

    @staticmethod
    def snapshot_many(buffers: Iterable["OutputBuffer"]) -> "OutputSnapshot":
        return OutputSnapshot(tuple(slot for buffer in buffers for slot in buffer.snapshot().slots))

    @staticmethod
    def materialize_many(buffers: Iterable["OutputBuffer"]) -> list[list[int]]:
        buffers = list(buffers)
        OutputBuffer.snapshot_many(buffers).prefetch().materialize()
        return [buffer.token_ids() for buffer in buffers]


@dataclass(frozen=True)
class OutputSnapshot:
    """Deferred slots captured at one point in time."""

    slots: tuple[OutputSlot, ...]

    def prefetch(self) -> "OutputSnapshot":
        seen: set[int] = set()
        for slot in self.slots:
            array = slot.token.tokens
            if id(array) in seen:
                continue
            seen.add(id(array))
            copy_to_host_async = getattr(array, "copy_to_host_async", None)
            if copy_to_host_async is not None:
                copy_to_host_async()
        return self

    def materialize(self) -> None:
        slots = [slot for slot in self.slots if slot.buffer._deferred.get(slot.index) is slot.token]
        if not slots:
            return

        import jax
        import jax.numpy as jnp

        vectors: list[Any] = []
        vector_indices: dict[int, int] = {}
        locations: list[tuple[int, int]] = []
        for slot in slots:
            vector = slot.token.tokens
            vector_index = vector_indices.setdefault(id(vector), len(vectors))
            if vector_index == len(vectors):
                vectors.append(jnp.asarray(vector, dtype=jnp.int32).reshape(-1))
            locations.append((vector_index, int(slot.token.row)))

        vector_values = jax.device_get(vectors)
        for slot, (array_index, row) in zip(slots, locations):
            value = vector_values[array_index][row]
            if slot.buffer._deferred.get(slot.index) is slot.token:
                slot.buffer._token_ids[slot.index] = int(value)
                del slot.buffer._deferred[slot.index]
