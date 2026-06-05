import pytest

from benchmarks.benchmark_jax_server_trace import (
    _performance_with_token_scopes,
    _timing_metrics,
)


def test_timing_metrics_reports_final_materialization_gap():
    events = [
        {"event": "token", "request_index": 0, "elapsed_seconds": 0.10},
        {"event": "token", "request_index": 0, "elapsed_seconds": 0.20},
        {"event": "token", "request_index": 1, "elapsed_seconds": 0.15},
    ]

    metrics = _timing_metrics(events, elapsed=0.25, total_tokens=3)

    assert metrics["seconds"] == 0.25
    assert metrics["last_token_elapsed_seconds"] == 0.20
    assert metrics["post_last_token_drain_seconds"] == pytest.approx(0.05)
    assert metrics["tokens_per_second"] == 12.0
    assert metrics["token_event_tokens_per_second"] == 15.0


def test_performance_with_token_scopes_keeps_end_to_end_and_token_event_rates():
    performance = {
        "token_event_tokens_per_second": 15.0,
    }
    rows = [
        {"prompt_length": 4, "generated_tokens": 2},
        {"prompt_length": 6, "generated_tokens": 1},
    ]

    enriched = _performance_with_token_scopes(rows, performance, elapsed=0.25)

    assert enriched["output_token_throughput"] == 12.0
    assert enriched["token_event_output_token_throughput"] == 15.0
    assert enriched["total_token_throughput"] == 52.0
