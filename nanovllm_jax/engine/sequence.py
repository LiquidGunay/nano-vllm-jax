"""Sequence management for JAX engine."""

from copy import copy
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass, field
from typing import Any, List, Optional


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SamplingParams:
    """Sampling parameters for generation."""
    temperature: float = 1.0
    max_tokens: int = 256
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = -1


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


class Sequence:
    """Represents a sequence being generated."""
    
    block_size: int = 16  # Will be overridden by config
    counter = count()

    def __init__(
        self, 
        token_ids: List[int], 
        sampling_params: Optional[SamplingParams] = None,
        seq_id: Optional[int] = None,
    ):
        if seq_id is None:
            self.seq_id = next(Sequence.counter)
        else:
            self.seq_id = seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.last_token_device: Any | None = None
        self._device_token_slots: List[tuple[int, Any]] = []
        self._device_token_indices: set[int] = set()
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.cached_prefix_hash: int | None = None
        self.cached_prefix_hybrid_seeded = False
        self.prefix_cache_enabled = False
        self.block_table: List[int] = []
        # Scheduler-owned speculative decoding admission. The model runner
        # treats this as a per-step permission bit and must fall back to
        # baseline decode when it is false.
        self.mtp_admitted = True
        self.mtp_admission_reason = "default"
        
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        self.materialize_device_tokens()
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_materialized_completion_tokens(self) -> int:
        """Return the contiguous completion prefix that is already on host."""
        completion_start = self.num_prompt_tokens
        completion_end = self.num_tokens
        for index in sorted(self._device_token_indices):
            if index >= completion_start:
                completion_end = min(completion_end, index)
                break
        return max(0, completion_end - completion_start)

    def materialized_completion_token_ids(self) -> List[int]:
        """Return only the contiguous completion prefix that does not sync."""
        completion_end = self.num_prompt_tokens + self.num_materialized_completion_tokens
        return self.token_ids[self.num_prompt_tokens:completion_end]

    @property
    def has_unmaterialized_device_tokens(self) -> bool:
        return bool(self._device_token_slots)

    def block_has_unmaterialized_device_tokens(self, block_idx: int) -> bool:
        start = block_idx * self.block_size
        end = (block_idx + 1) * self.block_size
        return any(start <= index < end for index in self._device_token_indices)

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """Get token IDs for block i."""
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = (i + 1) * self.block_size
        return self.token_ids[start:end]

    def append_token(self, token_id: int):
        """Append a generated token."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.last_token_device = None
        self.num_tokens += 1

    def append_token_device(self, token_id: Any):
        """Append a generated token that is still resident on device."""
        index = len(self.token_ids)
        self.token_ids.append(0)
        self.last_token = 0
        self.last_token_device = token_id
        self._device_token_slots.append((index, token_id))
        self._device_token_indices.add(index)
        self.num_tokens += 1

    def materialize_device_tokens(self):
        """Resolve deferred device token IDs into the Python token list."""
        self.materialize_device_tokens_for_sequences([self])

    @staticmethod
    def snapshot_device_token_slots_for_sequences(seqs: List["Sequence"]) -> tuple[DeviceTokenSlot, ...]:
        """Capture currently deferred token slots without resolving them."""
        slots: List[DeviceTokenSlot] = []
        for seq in seqs:
            for index, token in seq._device_token_slots:
                slots.append(DeviceTokenSlot(seq=seq, index=index, token=token))
        return tuple(slots)

    @staticmethod
    def snapshot_new_device_token_slots_for_sequences(
        seqs: List["Sequence"],
        min_completion_lengths: dict[int, int],
    ) -> tuple[DeviceTokenSlot, ...]:
        """Capture deferred slots added after each sequence's known prefix."""
        slots: List[DeviceTokenSlot] = []
        for seq in seqs:
            min_index = seq.num_prompt_tokens + int(min_completion_lengths.get(int(seq.seq_id), 0))
            for index, token in seq._device_token_slots:
                if index >= min_index:
                    slots.append(DeviceTokenSlot(seq=seq, index=index, token=token))
        return tuple(slots)

    @staticmethod
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

    @staticmethod
    def materialize_device_tokens_for_sequences(seqs: List["Sequence"]):
        """Resolve deferred device token IDs for multiple sequences in one sync."""
        Sequence.materialize_device_token_slots(
            Sequence.snapshot_device_token_slots_for_sequences(seqs)
        )

    @staticmethod
    def materialize_device_token_slots(slots: tuple[DeviceTokenSlot, ...]):
        """Resolve only the deferred token slots captured by ``slots``."""
        entries: List[DeviceTokenSlot] = []
        scalar_entries: List[int] = []
        scalar_arrays = []
        vector_entries: List[tuple[int, int, int]] = []
        vector_arrays = []
        vector_slots: dict[int, int] = {}
        for slot in slots:
            seq = slot.seq
            if not any(index == slot.index and token is slot.token for index, token in seq._device_token_slots):
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

        touched: List["Sequence"] = []
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

    def get_absolute_positions(self) -> List[int]:
        """Get absolute positions for all tokens in sequence."""
        return list(range(self.num_tokens))

    def get_new_positions(self) -> List[int]:
        """Get positions for new tokens (not yet cached)."""
        return list(range(self.num_cached_tokens, self.num_tokens))
