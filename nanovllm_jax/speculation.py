"""Small values shared by draft models and target verification."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import jax


MAX_DRAFT_WIDTH = 15


@dataclass(frozen=True)
class DrafterConfig:
    """Constructor-time selection of the experimental persistent drafter."""

    kind: str
    width: int

    def __post_init__(self) -> None:
        if self.kind != "mtp":
            raise ValueError("the only supported drafter kind is 'mtp'")
        if isinstance(self.width, bool):
            raise TypeError("draft width must be an integer")
        try:
            width = index(self.width)
        except TypeError as exc:
            raise TypeError("draft width must be an integer") from exc
        if not 1 <= width <= MAX_DRAFT_WIDTH:
            raise ValueError(
                f"draft width must be between 1 and {MAX_DRAFT_WIDTH}"
            )
        object.__setattr__(self, "width", width)

    @classmethod
    def mtp(cls, width: int = 3) -> "DrafterConfig":
        return cls("mtp", width)

    @property
    def verification_width(self) -> int:
        return self.width + 1

    @property
    def prefill_lookahead_tokens(self) -> int:
        return max(0, self.width - 1)

    @property
    def decode_lookahead_slots(self) -> int:
        return 2 * self.width

    @property
    def capacity_padding_tokens(self) -> int:
        return max(0, self.width - 2)


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

    import jax.numpy as jnp

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
