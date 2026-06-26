"""Deferred output-token materialization.

Owns:
    Snapshots of device-resident generated token ids and their host transfer.
Receives:
    Sequences with deferred output slots created by the runner.
Returns:
    Host token ids for streaming events or final completions.
Invariant:
    Materializing a snapshot must not clear newer deferred tokens appended after
    that snapshot was taken.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanovllm_jax.sequence import DeviceTokenSlot, Sequence


@dataclass(frozen=True)
class OutputBuffer:
    """Snapshot wrapper for deferred token slots owned by sequences.

    The sequence object still stores the physical slots during this extraction
    phase.  Routing all call sites through ``OutputBuffer`` makes the intended
    ownership boundary explicit and keeps the future move away from ``Sequence``
    localized.
    """

    slots: tuple[DeviceTokenSlot, ...]

    @classmethod
    def capture(cls, seqs: list[Sequence]) -> "OutputBuffer":
        return cls(Sequence.snapshot_device_token_slots_for_sequences(seqs))

    @classmethod
    def capture_new(
        cls,
        seqs: list[Sequence],
        min_completion_lengths: dict[int, int],
    ) -> "OutputBuffer":
        return cls(
            Sequence.snapshot_new_device_token_slots_for_sequences(
                seqs,
                min_completion_lengths,
            )
        )

    def prefetch(self) -> "OutputBuffer":
        return type(self)(Sequence.prefetch_device_token_slots(self.slots))

    def materialize(self) -> None:
        Sequence.materialize_device_token_slots(self.slots)

    @staticmethod
    def materialize_sequences(seqs: list[Sequence]) -> None:
        Sequence.materialize_device_tokens_for_sequences(seqs)
