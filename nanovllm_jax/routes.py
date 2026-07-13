"""Typed execution routes shared by selection, dispatch, and warmup."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BatchPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class TokenMode(Enum):
    LOGITS = "logits"
    GREEDY = "greedy"
    SAMPLED = "sampled"
    BURST = "burst"


class RouteCapability(Enum):
    TABLE_STATE = "table_state"
    PREFILL_TOKEN_SEED = "prefill_token_seed"
    RESIDENT_METADATA = "resident_metadata"
    SLOT_TOKENS = "slot_tokens"
    DENSE_ROWS = "dense_rows"


class RouteKind(Enum):
    PREFILL_LOGITS = "prefill_logits"
    PREFILL_GREEDY = "prefill_greedy"
    PREFILL_SAMPLED = "prefill_sampled"
    PREFILL_TABLE = "prefill_table"
    PREFILL_RESIDENT = "prefill_resident"
    DECODE_LOGITS = "decode_logits"
    DECODE_GREEDY = "decode_greedy"
    DECODE_SAMPLED = "decode_sampled"
    DECODE_TABLE = "decode_table"
    DECODE_SLOT_TOKENS = "decode_slot_tokens"
    DECODE_RESIDENT_METADATA = "decode_resident_metadata"
    DECODE_RESIDENT = "decode_resident"
    DECODE_RESIDENT_DENSE = "decode_resident_dense"
    DECODE_BURST = "decode_burst"
    DECODE_TABLE_BURST = "decode_table_burst"


@dataclass(frozen=True)
class RouteSpec:
    kind: RouteKind
    phase: BatchPhase
    tokens: TokenMode
    executor: str | None
    requires: frozenset[RouteCapability] = frozenset()

    @property
    def uses_table_state(self) -> bool:
        return RouteCapability.TABLE_STATE in self.requires

    @property
    def uses_resident_metadata(self) -> bool:
        return RouteCapability.RESIDENT_METADATA in self.requires

    @property
    def uses_slot_tokens(self) -> bool:
        return RouteCapability.SLOT_TOKENS in self.requires

    @property
    def seeds_slot_tokens(self) -> bool:
        return RouteCapability.PREFILL_TOKEN_SEED in self.requires


def _caps(*values: RouteCapability) -> frozenset[RouteCapability]:
    return frozenset(values)


ROUTE_SPECS = (
    RouteSpec(
        RouteKind.PREFILL_RESIDENT,
        BatchPhase.PREFILL,
        TokenMode.GREEDY,
        "forward_prefill_token_ids_slot_carry_table_jit",
        _caps(RouteCapability.TABLE_STATE, RouteCapability.PREFILL_TOKEN_SEED),
    ),
    RouteSpec(
        RouteKind.PREFILL_TABLE,
        BatchPhase.PREFILL,
        TokenMode.GREEDY,
        "forward_prefill_token_ids_table_jit",
        _caps(RouteCapability.TABLE_STATE),
    ),
    RouteSpec(
        RouteKind.PREFILL_GREEDY,
        BatchPhase.PREFILL,
        TokenMode.GREEDY,
        "forward_step_token_ids_jit",
    ),
    RouteSpec(
        RouteKind.PREFILL_SAMPLED,
        BatchPhase.PREFILL,
        TokenMode.SAMPLED,
        "forward_step_sampled_token_ids_jit",
    ),
    RouteSpec(RouteKind.PREFILL_LOGITS, BatchPhase.PREFILL, TokenMode.LOGITS, None),
    RouteSpec(
        RouteKind.DECODE_RESIDENT_DENSE,
        BatchPhase.DECODE,
        TokenMode.GREEDY,
        "forward_step_token_ids_resident_dense_slot_carry_jit",
        _caps(
            RouteCapability.TABLE_STATE,
            RouteCapability.RESIDENT_METADATA,
            RouteCapability.SLOT_TOKENS,
            RouteCapability.DENSE_ROWS,
        ),
    ),
    RouteSpec(
        RouteKind.DECODE_RESIDENT,
        BatchPhase.DECODE,
        TokenMode.GREEDY,
        "forward_step_token_ids_resident_slot_carry_jit",
        _caps(
            RouteCapability.TABLE_STATE,
            RouteCapability.RESIDENT_METADATA,
            RouteCapability.SLOT_TOKENS,
        ),
    ),
    RouteSpec(
        RouteKind.DECODE_RESIDENT_METADATA,
        BatchPhase.DECODE,
        TokenMode.GREEDY,
        "forward_step_token_ids_resident_jit",
        _caps(RouteCapability.TABLE_STATE, RouteCapability.RESIDENT_METADATA),
    ),
    RouteSpec(
        RouteKind.DECODE_SLOT_TOKENS,
        BatchPhase.DECODE,
        TokenMode.GREEDY,
        "forward_step_token_ids_slot_carry_table_jit",
        _caps(RouteCapability.TABLE_STATE, RouteCapability.SLOT_TOKENS),
    ),
    RouteSpec(
        RouteKind.DECODE_TABLE,
        BatchPhase.DECODE,
        TokenMode.GREEDY,
        "forward_step_token_ids_table_jit",
        _caps(RouteCapability.TABLE_STATE),
    ),
    RouteSpec(
        RouteKind.DECODE_GREEDY,
        BatchPhase.DECODE,
        TokenMode.GREEDY,
        "forward_step_token_ids_jit",
    ),
    RouteSpec(
        RouteKind.DECODE_SAMPLED,
        BatchPhase.DECODE,
        TokenMode.SAMPLED,
        "forward_step_sampled_token_ids_jit",
    ),
    RouteSpec(RouteKind.DECODE_LOGITS, BatchPhase.DECODE, TokenMode.LOGITS, None),
    RouteSpec(
        RouteKind.DECODE_TABLE_BURST,
        BatchPhase.DECODE,
        TokenMode.BURST,
        "forward_greedy_decode_burst_table_jit",
        _caps(RouteCapability.TABLE_STATE),
    ),
    RouteSpec(
        RouteKind.DECODE_BURST,
        BatchPhase.DECODE,
        TokenMode.BURST,
        "forward_greedy_decode_burst_jit",
    ),
)

_ROUTES = {spec.kind: spec for spec in ROUTE_SPECS}


@dataclass(frozen=True)
class RouteRequest:
    phase: BatchPhase
    tokens: TokenMode
    capabilities: frozenset[RouteCapability]


@dataclass(frozen=True)
class ExecutionPlan:
    kind: RouteKind
    active_rows: tuple[int, ...]
    decode_steps: int
    prefill_final_flags: tuple[bool, ...]

    @property
    def spec(self) -> RouteSpec:
        return _ROUTES[self.kind]


def select_route(request: RouteRequest) -> RouteKind:
    """Choose the most capable route whose requirements are satisfied."""

    candidates = (
        spec
        for spec in ROUTE_SPECS
        if spec.phase is request.phase
        and spec.tokens is request.tokens
        and spec.requires <= request.capabilities
    )
    try:
        return max(candidates, key=lambda spec: len(spec.requires)).kind
    except ValueError as exc:
        raise RuntimeError(
            f"no {request.phase.value}/{request.tokens.value} execution route"
        ) from exc


def validate_executor(executor: object) -> None:
    missing = sorted(
        {
            spec.executor
            for spec in ROUTE_SPECS
            if spec.executor is not None and not hasattr(executor, spec.executor)
        }
    )
    if missing:
        raise TypeError("executor is missing route methods: " + ", ".join(missing))
