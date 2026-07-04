"""Deferred output-token host transfer.

Owns:
    Device token references, slot snapshots, prefetch, and materialization.
Receives:
    Sequence-like objects that store logical token ids plus private deferred
    output slots.
Returns:
    Host-resident token ids written back into the owning sequences.
Invariant:
    Materializing a snapshot must not clear newer deferred tokens appended after
    that snapshot was taken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceTokenRef:
    """Deferred reference to one row in a device-resident token vector."""

    tokens: Any
    row: int


@dataclass(frozen=True)
class DeviceTokenSlot:
    """Snapshot of one deferred token slot in a sequence."""

    seq: Any
    index: int
    token: Any


def snapshot_device_token_slots_for_sequences(seqs: list[Any]) -> tuple[DeviceTokenSlot, ...]:
    """Capture currently deferred token slots without resolving them."""
    slots: list[DeviceTokenSlot] = []
    for seq in seqs:
        for index, token in seq._device_token_slots:
            slots.append(DeviceTokenSlot(seq=seq, index=index, token=token))
    return tuple(slots)


def snapshot_new_device_token_slots_for_sequences(
    seqs: list[Any],
    min_completion_lengths: dict[int, int],
) -> tuple[DeviceTokenSlot, ...]:
    """Capture deferred slots added after each sequence's known prefix."""
    slots: list[DeviceTokenSlot] = []
    for seq in seqs:
        min_index = seq.num_prompt_tokens + int(min_completion_lengths.get(int(seq.seq_id), 0))
        for index, token in seq._device_token_slots:
            if index >= min_index:
                slots.append(DeviceTokenSlot(seq=seq, index=index, token=token))
    return tuple(slots)


def prefetch_device_token_slots(slots: tuple[DeviceTokenSlot, ...]) -> tuple[DeviceTokenSlot, ...]:
    """Ask JAX to start host transfer for the arrays referenced by a slot snapshot."""
    seen_arrays: set[int] = set()
    for slot in slots:
        token = slot.token.tokens if isinstance(slot.token, DeviceTokenRef) else slot.token
        token_id = id(token)
        if token_id in seen_arrays:
            continue
        seen_arrays.add(token_id)
        copy_to_host_async = getattr(token, "copy_to_host_async", None)
        if copy_to_host_async is not None:
            copy_to_host_async()
    return slots


def materialize_device_tokens_for_sequences(seqs: list[Any]) -> None:
    """Resolve deferred device token IDs for multiple sequences in one sync."""
    materialize_device_token_slots(snapshot_device_token_slots_for_sequences(seqs))


def materialize_device_token_slots(slots: tuple[DeviceTokenSlot, ...]) -> None:
    """Resolve only the deferred token slots captured by ``slots``."""
    entries: list[DeviceTokenSlot] = []
    scalar_entries: list[int] = []
    scalar_arrays = []
    vector_entries: list[tuple[int, int, int]] = []
    vector_arrays = []
    vector_slots: dict[int, int] = {}
    current_slot_keys_by_seq: dict[int, set[tuple[int, int]]] = {}
    for slot in slots:
        seq = slot.seq
        seq_key = id(seq)
        current_slot_keys = current_slot_keys_by_seq.get(seq_key)
        if current_slot_keys is None:
            current_slot_keys = {
                (int(index), id(token))
                for index, token in seq._device_token_slots
            }
            current_slot_keys_by_seq[seq_key] = current_slot_keys
        if (int(slot.index), id(slot.token)) not in current_slot_keys:
            continue
        entries.append(slot)
        entry_index = len(entries) - 1
        if isinstance(slot.token, DeviceTokenRef):
            vector_id = id(slot.token.tokens)
            vector_slot = vector_slots.get(vector_id)
            if vector_slot is None:
                vector_slot = len(vector_arrays)
                vector_slots[vector_id] = vector_slot
                vector_arrays.append(slot.token.tokens)
            vector_entries.append((entry_index, vector_slot, int(slot.token.row)))
        else:
            scalar_entries.append(entry_index)
            scalar_arrays.append(slot.token)
    if not scalar_arrays and not vector_arrays:
        return

    import jax
    import jax.numpy as jnp

    values_by_entry: dict[int, int] = {}
    if scalar_arrays:
        scalar_arrays = [jnp.asarray(token, dtype=jnp.int32).reshape(()) for token in scalar_arrays]
        scalar_values = jax.device_get(jnp.stack(scalar_arrays)).tolist()
        for entry_index, value in zip(scalar_entries, scalar_values):
            values_by_entry[entry_index] = int(value)
    if vector_arrays:
        vector_arrays = [
            jnp.asarray(tokens, dtype=jnp.int32).reshape(-1)
            for tokens in vector_arrays
        ]
        host_vectors = jax.device_get(vector_arrays)
        for entry_index, vector_slot, row in vector_entries:
            values_by_entry[entry_index] = int(host_vectors[vector_slot][row])

    touched: list[Any] = []
    seen: set[int] = set()
    materialized_by_seq: dict[int, set[tuple[int, int]]] = {}
    for entry_index, slot in enumerate(entries):
        value = values_by_entry[entry_index]
        seq = slot.seq
        seq.token_ids[slot.index] = value
        materialized_by_seq.setdefault(id(seq), set()).add((slot.index, id(slot.token)))
        seq_id = id(seq)
        if seq_id not in seen:
            touched.append(seq)
            seen.add(seq_id)
    for seq in touched:
        materialized = materialized_by_seq[id(seq)]
        seq._device_token_slots = [
            (index, token)
            for index, token in seq._device_token_slots
            if (index, id(token)) not in materialized
        ]
        seq._device_token_indices = {index for index, _ in seq._device_token_slots}
        seq.last_token = seq.token_ids[-1]
        if seq._device_token_slots and seq._device_token_slots[-1][0] == len(seq.token_ids) - 1:
            seq.last_token_device = seq._device_token_slots[-1][1]
        else:
            seq.last_token_device = None


@dataclass(frozen=True)
class OutputBuffer:
    """Snapshot of deferred output tokens.

    ``Sequence`` still stores logical slots, but host transfer lives in this
    module so request state does not own JAX synchronization policy.
    """

    slots: tuple[DeviceTokenSlot, ...]

    @classmethod
    def capture(cls, seqs: list[Any]) -> "OutputBuffer":
        return cls(snapshot_device_token_slots_for_sequences(seqs))

    @classmethod
    def capture_new(
        cls,
        seqs: list[Any],
        min_completion_lengths: dict[int, int],
    ) -> "OutputBuffer":
        return cls(
            snapshot_new_device_token_slots_for_sequences(
                seqs,
                min_completion_lengths,
            )
        )

    def prefetch(self) -> "OutputBuffer":
        return type(self)(prefetch_device_token_slots(self.slots))

    def materialize(self) -> None:
        materialize_device_token_slots(self.slots)

    @staticmethod
    def materialize_sequences(seqs: list[Any]) -> None:
        materialize_device_tokens_for_sequences(seqs)
