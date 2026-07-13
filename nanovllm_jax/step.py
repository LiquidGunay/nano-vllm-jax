"""Typed values crossing the execute and commit boundaries."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from nanovllm_jax.batch import Phase


class FinishReason(str, Enum):
    EOS = "eos"
    LENGTH = "length"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunResult:
    """Tokens emitted by each physical execution row."""

    rows: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Any]) -> "RunResult":
        return cls(
            tuple(
                tuple(row) if isinstance(row, (list, tuple)) else (row,)
                for row in rows
            )
        )


@dataclass(frozen=True)
class TokenEvent:
    seq_id: int
    completion_index: int
    token: Any


@dataclass(frozen=True)
class FinishedRequest:
    seq_id: int
    reason: FinishReason


@dataclass(frozen=True)
class StepResult:
    """One committed engine transition."""

    phase: Phase
    scheduled_tokens: int
    emitted_tokens: tuple[TokenEvent, ...]
    finished: tuple[FinishedRequest, ...]

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def num_emitted_tokens(self) -> int:
        return len(self.emitted_tokens)
