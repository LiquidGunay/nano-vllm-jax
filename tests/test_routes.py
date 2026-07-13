from dataclasses import fields

from nanovllm_jax.routes import (
    ExecutionPlan,
    ROUTE_SPECS,
    RouteCapability,
    RouteKind,
    RouteRequest,
    select_route,
    validate_executor,
)


def test_every_route_is_selected_from_its_requirements():
    assert {spec.kind for spec in ROUTE_SPECS} == set(RouteKind)
    for spec in ROUTE_SPECS:
        request = RouteRequest(spec.phase, spec.tokens, spec.requires)
        assert select_route(request) is spec.kind


def test_more_capable_decode_routes_replace_their_fallbacks():
    by_kind = {spec.kind: spec for spec in ROUTE_SPECS}
    full = by_kind[RouteKind.DECODE_RESIDENT_DENSE]
    request = RouteRequest(full.phase, full.tokens, full.requires)
    assert select_route(request) is RouteKind.DECODE_RESIDENT_DENSE

    without_dense = full.requires - {RouteCapability.DENSE_ROWS}
    request = RouteRequest(full.phase, full.tokens, without_dense)
    assert select_route(request) is RouteKind.DECODE_RESIDENT

    without_tokens = without_dense - {RouteCapability.SLOT_TOKENS}
    request = RouteRequest(full.phase, full.tokens, without_tokens)
    assert select_route(request) is RouteKind.DECODE_RESIDENT_METADATA


def test_execution_plan_contains_one_route_choice():
    assert [field.name for field in fields(ExecutionPlan)] == [
        "kind",
        "active_rows",
        "decode_steps",
        "prefill_final_flags",
    ]
    plan = ExecutionPlan(RouteKind.PREFILL_RESIDENT, (0,), 1, (True,))
    assert plan.spec.kind is plan.kind


def test_executor_validation_uses_the_registry():
    class MissingRoutes:
        pass

    try:
        validate_executor(MissingRoutes())
    except TypeError as exc:
        assert "forward_step_token_ids_jit" in str(exc)
    else:
        raise AssertionError("missing route methods must fail validation")
