"""Engine-level batch contract shared by the scheduler and executor."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class ScheduledBatch:
    """A flat scheduled batch for one engine step.

    `tokens` and `positions` are padded to rectangular arrays, while
    `query_start_loc` preserves the true ragged query lengths.
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

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def query_lens(self) -> jnp.ndarray:
        return jnp.diff(self.query_start_loc).astype(jnp.int32)
