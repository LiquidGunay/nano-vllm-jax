"""Sequence management for JAX engine."""

from copy import copy
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass, field
from typing import List, Optional


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
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
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
        return self.token_ids[self.num_prompt_tokens:]

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
        self.num_tokens += 1

    def get_absolute_positions(self) -> List[int]:
        """Get absolute positions for all tokens in sequence."""
        return list(range(self.num_tokens))

    def get_new_positions(self) -> List[int]:
        """Get positions for new tokens (not yet cached)."""
        return list(range(self.num_cached_tokens, self.num_tokens))
