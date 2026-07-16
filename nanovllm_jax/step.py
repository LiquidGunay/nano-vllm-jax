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
    verified_target_tokens: int = 0
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Any],
        *,
        verified_target_tokens: int = 0,
        draft_tokens: int = 0,
        accepted_draft_tokens: int = 0,
    ) -> "RunResult":
        return cls(
            tuple(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in rows),
            int(verified_target_tokens),
            int(draft_tokens),
            int(accepted_draft_tokens),
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
    verified_target_tokens: int = 0
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def num_emitted_tokens(self) -> int:
        return len(self.emitted_tokens)
