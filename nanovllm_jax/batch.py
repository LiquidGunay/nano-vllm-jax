"""Host-only scheduling contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Phase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class ScheduledRow:
    """Logical work for one request in an engine step."""

    seq_id: int
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    block_table: tuple[int, ...]
    seq_len: int
    prefill_is_final: bool = True
    carries_device_token: bool = False

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("scheduled row must contain at least one token")
        if len(self.token_ids) != len(self.positions):
            raise ValueError("scheduled row token and position counts must match")

    @property
    def query_len(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class BucketShape:
    """Static accelerator shape selected by the scheduler."""

    batch_size: int
    query_tokens: int
    block_table_width: int
    packed_prefill: bool = False

    def __post_init__(self) -> None:
        if min(self.batch_size, self.query_tokens, self.block_table_width) <= 0:
            raise ValueError("bucket dimensions must be positive")


@dataclass(frozen=True)
class SchedulePlan:
    """Immutable host description of one scheduled step."""

    phase: Phase
    rows: tuple[ScheduledRow, ...]
    bucket: BucketShape
    decode_steps: int = 1

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported phase: {self.phase}")
        if not self.rows:
            raise ValueError("schedule plan must contain at least one row")
        if len(self.rows) > self.bucket.batch_size:
            raise ValueError("scheduled rows exceed the batch bucket")
        if self.decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        if self.phase == "decode" and any(row.query_len != 1 for row in self.rows):
            raise ValueError("decode rows must contain exactly one token")

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(row.query_len for row in self.rows)

    @property
    def prefill_chunk_lengths(self) -> tuple[int, ...]:
        if not self.is_prefill:
            return ()
        return tuple(row.query_len for row in self.rows)
