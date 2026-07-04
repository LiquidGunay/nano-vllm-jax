"""Logical request sequence state.

Owns:
    Request token ids, status, block table, prefix-cache markers, and sampling
    parameters.
Receives:
    Prompt token ids plus sampling parameters at request admission.
Returns:
    Logical sequence lengths and token views consumed by scheduler/runner.
Invariant:
    ``num_tokens`` is the committed logical length for the request.
"""

from copy import copy
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass
from typing import Any, List, Optional

from nanovllm_jax.output import (
    DeviceTokenRef,
    DeviceTokenSlot,
    materialize_device_token_slots,
    materialize_device_tokens_for_sequences,
    prefetch_device_token_slots,
    snapshot_device_token_slots_for_sequences,
    snapshot_new_device_token_slots_for_sequences,
)


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
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

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
        materialize_device_tokens_for_sequences([self])

    @staticmethod
    def snapshot_device_token_slots_for_sequences(seqs: List["Sequence"]) -> tuple[DeviceTokenSlot, ...]:
        """Capture currently deferred token slots without resolving them."""
        return snapshot_device_token_slots_for_sequences(seqs)

    @staticmethod
    def snapshot_new_device_token_slots_for_sequences(
        seqs: List["Sequence"],
        min_completion_lengths: dict[int, int],
    ) -> tuple[DeviceTokenSlot, ...]:
        """Capture deferred slots added after each sequence's known prefix."""
        return snapshot_new_device_token_slots_for_sequences(seqs, min_completion_lengths)

    @staticmethod
    def prefetch_device_token_slots(slots: tuple[DeviceTokenSlot, ...]) -> tuple[DeviceTokenSlot, ...]:
        """Ask JAX to start host transfer for the arrays referenced by a slot snapshot."""
        return prefetch_device_token_slots(slots)

    @staticmethod
    def materialize_device_tokens_for_sequences(seqs: List["Sequence"]):
        """Resolve deferred device token IDs for multiple sequences in one sync."""
        materialize_device_tokens_for_sequences(seqs)

    @staticmethod
    def materialize_device_token_slots(slots: tuple[DeviceTokenSlot, ...]):
        """Resolve only the deferred token slots captured by ``slots``."""
        materialize_device_token_slots(slots)

    def get_absolute_positions(self) -> List[int]:
        """Get absolute positions for all tokens in sequence."""
        return list(range(self.num_tokens))

    def get_new_positions(self) -> List[int]:
        """Get positions for new tokens (not yet cached)."""
        return list(range(self.num_cached_tokens, self.num_tokens))
