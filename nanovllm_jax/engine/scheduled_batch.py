"""Engine-level batch contract shared by the scheduler and executor."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class ScheduledBatch:
    """A flat scheduled batch for one engine step.

    `tokens` and `positions` are padded to rectangular arrays, while
    `query_start_loc` preserves the true ragged query lengths.

    Dense prefill/decode uses `tokens.shape[0]` rows.  Packed prefill keeps
    query tokens in a single row (`tokens.shape == [1, token_bucket]`) and uses
    `token_row_ids` plus the paged row metadata (`block_tables`, `seq_lens`,
    `query_start_loc`) to map each token back to its request.
    """

    tokens: jnp.ndarray
    positions: jnp.ndarray
    seq_ids: jnp.ndarray
    query_start_loc: jnp.ndarray
    is_prefill: bool
    num_prefill_tokens: int
    num_decode_tokens: int
    block_tables: jnp.ndarray
    seq_lens: jnp.ndarray
    prefill_is_final: list[bool] | tuple[bool, ...] | None = None
    seq_ids_host: tuple[int, ...] | None = None
    query_lens_host: tuple[int, ...] | None = None
    seq_lens_host: tuple[int, ...] | None = None
    block_tables_host: tuple[tuple[int, ...], ...] | None = None
    hybrid_slot_ids_host: tuple[int, ...] | None = None
    decode_step_count_host: int = 1
    uses_static_decode_metadata: bool = False
    packed_prefill: bool = False
    token_row_ids: jnp.ndarray | None = None
    mixed_prefill_decode: bool = False

    @property
    def batch_size(self) -> int:
        if self.packed_prefill:
            return int(self.block_tables.shape[0])
        return int(self.tokens.shape[0])

    @property
    def query_lens(self) -> jnp.ndarray:
        return jnp.diff(self.query_start_loc).astype(jnp.int32)

    @property
    def active_decode_rows(self) -> jnp.ndarray:
        return (self.seq_ids >= 0) & (self.query_lens > 0)

    @property
    def prefill_final_flags(self) -> list[bool]:
        if self.prefill_is_final is None:
            return [True] * self.batch_size
        if isinstance(self.prefill_is_final, np.ndarray):
            return [bool(x) for x in np.array(self.prefill_is_final, dtype=bool).tolist()]
        if isinstance(self.prefill_is_final, tuple):
            return list(self.prefill_is_final)
        if isinstance(self.prefill_is_final, jnp.ndarray):
            return [bool(x) for x in list(np.array(self.prefill_is_final, dtype=bool).tolist())]
        return list(self.prefill_is_final)
