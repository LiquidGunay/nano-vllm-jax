"""Typed boundary between a token drafter and the target verifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, NamedTuple, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from nanovllm_jax.sequence import Sequence as RequestSequence


class DraftProposal(NamedTuple):
    """Fixed-width draft token ids, one row per physical decode row."""

    token_ids: jax.Array

    @property
    def width(self) -> int:
        return int(self.token_ids.shape[1])


class VerificationResult(NamedTuple):
    """Compact device result from one target-model verification pass."""

    emitted_token_ids: jax.Array
    emitted_counts: jax.Array
    accepted_counts: jax.Array
    next_token_ids: jax.Array

def verify_greedy_drafts(
    proposal: DraftProposal,
    target_token_ids: jax.Array,
) -> VerificationResult:
    """Accept the longest matching draft prefix and append one target token."""

    draft_width = proposal.width
    if target_token_ids.shape != (proposal.token_ids.shape[0], draft_width + 1):
        raise ValueError("target tokens must contain one bonus token per draft row")
    accepted = jnp.cumprod(
        (target_token_ids[:, :draft_width] == proposal.token_ids).astype(jnp.int32),
        axis=1,
    )
    accepted_counts = jnp.sum(accepted, axis=1).astype(jnp.int32)
    next_tokens = jnp.take_along_axis(
        target_token_ids,
        accepted_counts[:, None],
        axis=1,
    )[:, 0]
    columns = jnp.arange(draft_width + 1, dtype=jnp.int32)[None, :]
    drafts = jnp.pad(proposal.token_ids, ((0, 0), (0, 1)))
    emitted = jnp.where(
        columns < accepted_counts[:, None],
        drafts,
        jnp.where(
            columns == accepted_counts[:, None],
            next_tokens[:, None],
            jnp.zeros_like(drafts),
        ),
    ).astype(jnp.int32)
    return VerificationResult(
        emitted,
        accepted_counts + 1,
        accepted_counts,
        next_tokens.astype(jnp.int32),
    )


class Drafter(Protocol):
    """Produces device token proposals without defining verification policy."""

    @property
    def width(self) -> int: ...

    def propose(
        self,
        seqs: Sequence[RequestSequence],
    ) -> DraftProposal: ...


class SuppliedDrafter:
    """Diagnostic drafter backed by known completion token streams."""

    def __init__(
        self,
        token_ids_by_seq: Mapping[int, Sequence[int]],
        *,
        width: int,
    ) -> None:
        if width < 1:
            raise ValueError("draft width must be positive")
        self._width = int(width)
        self._tokens = {
            int(seq_id): tuple(int(token) for token in tokens)
            for seq_id, tokens in token_ids_by_seq.items()
        }

    @property
    def width(self) -> int:
        return self._width

    def propose(
        self,
        seqs: Sequence[RequestSequence],
    ) -> DraftProposal:
        rows = np.zeros((len(seqs), self.width), dtype=np.int32)
        for row, seq in enumerate(seqs):
            source = self._tokens.get(int(seq.seq_id))
            if source is None:
                raise KeyError(f"no supplied drafts for sequence {seq.seq_id}")
            start = int(seq.num_completion_tokens)
            drafts = source[start : start + self.width]
            if len(drafts) != self.width:
                raise ValueError(
                    f"sequence {seq.seq_id} needs {self.width} drafts at token {start}"
                )
            rows[row] = drafts
        return DraftProposal(jnp.asarray(rows))
