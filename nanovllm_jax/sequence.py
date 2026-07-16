"""Logical state for one generation request."""

from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
import math
from numbers import Integral, Real
from typing import List, Optional

from nanovllm_jax.output import OutputBuffer


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class SamplingParams:
    """Sampling parameters for generation."""

    temperature: float = 0.0
    max_tokens: int = 256
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise TypeError("temperature must be a real number")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, Integral):
            raise TypeError("max_tokens must be an integer")
        max_tokens = int(self.max_tokens)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not isinstance(self.ignore_eos, bool):
            raise TypeError("ignore_eos must be a boolean")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_tokens", max_tokens)


class Sequence:
    """Logical positions, cache metadata, and sampling policy for one request."""

    def __init__(
        self,
        token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        seq_id: int = 0,
        block_size: int = 16,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.seq_id = int(seq_id)
        self.block_size = int(block_size)
        self.status = SequenceStatus.WAITING
        self.prompt_token_ids = copy(token_ids)
        self.output = OutputBuffer()
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.cached_prefix_hash: int | None = None
        self.cached_prefix_hybrid_seeded = False
        self.prefix_cache_enabled = False
        self.block_table: List[int] = []
        sampling_params = sampling_params or SamplingParams()
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def token_ids(self) -> list[int]:
        """Logical ids with zero placeholders for deferred output tokens."""
        return self.prompt_token_ids + self.output.logical_token_ids()

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output)

    @property
    def last_token(self) -> int:
        return self.output.last_token if len(self.output) else self.prompt_token_ids[-1]

    @property
    def last_token_device(self):
        return self.output.last_device_token

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return len(self.output)

    def block_has_unmaterialized_device_tokens(self, block_idx: int) -> bool:
        start = max(0, block_idx * self.block_size - self.num_prompt_tokens)
        end = max(0, (block_idx + 1) * self.block_size - self.num_prompt_tokens)
        return self.output.has_deferred_between(start, end)

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, index: int) -> list[int]:
        assert 0 <= index < self.num_blocks
        start = index * self.block_size
        return self.token_ids[start:start + self.block_size]

    def get_absolute_positions(self) -> List[int]:
        return list(range(self.num_tokens))

    def get_new_positions(self) -> List[int]:
        return list(range(self.num_cached_tokens, self.num_tokens))
