#!/usr/bin/env python3
"""Render a concise Markdown report from a GPU matrix summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", help="Path to results/gpu_matrix_<timestamp>.json")
    parser.add_argument("--output-md", default="", help="Optional path to write the Markdown report")
    parser.add_argument("--top-profile-deltas", type=int, default=8)
    parser.add_argument("--top-scoped-events", type=int, default=5)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, *, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value}{suffix}"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def _fmt_ratio(value: Any) -> str:
    return _fmt(value, suffix="x", digits=3)


def _goal_target_row(summary: dict[str, Any]) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    target = summary.get("goal_target") or {}
    workload = str(target.get("workload") or "")
    config = str(target.get("config") or "")
    comparison = ((summary.get("comparisons") or {}).get(workload) or {}).get(config) or {}
    acceptance = ((summary.get("acceptance") or {}).get(workload) or {}).get(config) or {}
    return workload, config, comparison, acceptance


def _acceptance_failure_text(workload: str, config: str, acceptance: dict[str, Any]) -> str | None:
    checks = acceptance.get("checks") or {}
    failed_checks = sorted(key for key, value in checks.items() if not value)
    parts: list[str] = []
    if failed_checks:
        parts.append("failed checks: " + ", ".join(failed_checks))
    if not acceptance.get("speed_claim_ready"):
        parts.append("speed_claim_ready=false")
    if not acceptance.get("target_vllm_ratio_met"):
        parts.append(f"target_vllm_ratio_met=false target={acceptance.get('target_vllm_ratio')}")
    missing_profile = acceptance.get("missing_profile_counters") or []
    if missing_profile:
        parts.append(f"missing_profile_counters={len(missing_profile)}")
    if not parts:
        return None
    return f"{workload}/{config}: " + "; ".join(parts)


def acceptance_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for workload in summary.get("workloads") or []:
        for config in summary.get("configs") or []:
            acceptance = ((summary.get("acceptance") or {}).get(workload) or {}).get(config) or {}
            failure = _acceptance_failure_text(str(workload), str(config), acceptance)
            if failure:
                failures.append(failure)
    return failures


def _matrix_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for workload in summary.get("workloads") or []:
        for config in summary.get("configs") or []:
            comparison = ((summary.get("comparisons") or {}).get(workload) or {}).get(config) or {}
            acceptance = ((summary.get("acceptance") or {}).get(workload) or {}).get(config) or {}
            rows.append(
                [
                    str(workload),
                    str(config),
                    _fmt(acceptance.get("speed_claim_ready")),
                    _fmt(acceptance.get("target_vllm_ratio_met")),
                    _fmt(comparison.get("jax_tokens_per_second_median")),
                    _fmt(comparison.get("vllm_tokens_per_second")),
                    _fmt_ratio(comparison.get("jax_over_vllm_throughput")),
                    _fmt(comparison.get("target_tokens_per_second")),
                    _fmt(comparison.get("tokens_per_second_gap_to_target")),
                    _fmt(comparison.get("jax_reference_tokens_per_second")),
                    _fmt_ratio(comparison.get("jax_over_jax_reference_throughput")),
                ]
            )
    return rows


def _profile_delta_rows(comparison: dict[str, Any], *, limit: int) -> list[list[str]]:
    deltas = comparison.get("profile_delta_vs_jax_reference") or {}
    rows = []
    for name, bucket in deltas.items():
        delta = bucket.get("total_ms_delta")
        if delta is None:
            continue
        rows.append(
            (
                abs(float(delta)),
                [
                    str(name),
                    _fmt(bucket.get("current_total_ms_median"), suffix=" ms"),
                    _fmt(bucket.get("reference_total_ms"), suffix=" ms"),
                    _fmt(delta, suffix=" ms"),
                    _fmt_ratio(bucket.get("total_ms_ratio")),
                    _fmt(bucket.get("current_count_median"), digits=1),
                    _fmt(bucket.get("reference_count"), digits=1),
                    _fmt(bucket.get("count_delta"), digits=1),
                ],
            )
        )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[: max(0, int(limit))]]


def _profile_delta_bullets(comparison: dict[str, Any], *, limit: int) -> list[str]:
    rows = _profile_delta_rows(comparison, limit=limit)
    if not rows:
        return ["- No profile delta rows are available."]
    return [
        (
            f"- `{row[0]}`: current {row[1]}, reference {row[2]}, "
            f"delta {row[3]}, ratio {row[4]}, count delta {row[7]}"
        )
        for row in rows
    ]


def _scheduler_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for workload in summary.get("workloads") or []:
        for config in summary.get("configs") or []:
            matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
            scheduler = (matrix_row.get("aggregate") or {}).get("scheduler_diagnostics_median") or {}
            if not scheduler.get("available"):
                continue
            rows.append(
                [
                    str(workload),
                    str(config),
                    _fmt(scheduler.get("prefill_step_count"), digits=1),
                    _fmt(scheduler.get("decode_step_count"), digits=1),
                    _fmt(scheduler.get("max_prefill_step_sequences"), digits=1),
                    _fmt(scheduler.get("max_step_tokens"), digits=1),
                    _fmt(scheduler.get("prefill_step_seconds_total"), suffix=" s"),
                    _fmt(scheduler.get("decode_step_seconds_total"), suffix=" s"),
                ]
            )
    return rows


def _scoped_profile_event_rows(
    summary: dict[str, Any],
    *,
    scopes: tuple[str, ...] = ("gpu", "cpu"),
    limit: int = 5,
) -> list[list[str]]:
    rows: list[list[str]] = []
    max_events = max(0, int(limit))
    if max_events == 0:
        return rows
    for workload in summary.get("workloads") or []:
        for config in summary.get("configs") or []:
            matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
            for repeat in matrix_row.get("repeats") or []:
                metrics = repeat.get("metrics") or {}
                scoped_events = metrics.get("profile_scoped_top_events_by_total_ms") or {}
                for scope in scopes:
                    for event in (scoped_events.get(scope) or [])[:max_events]:
                        rows.append(
                            [
                                str(workload),
                                str(config),
                                _fmt(repeat.get("repeat")),
                                str(scope),
                                str(event.get("name")),
                                _fmt(event.get("total_ms"), suffix=" ms"),
                                _fmt(event.get("count")),
                            ]
                        )
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _logbook_template(summary: dict[str, Any], *, top_profile_deltas: int) -> list[str]:
    workload, config, goal_comparison, goal_acceptance = _goal_target_row(summary)
    lines = [
        "## Logbook Entry Template",
        "",
        "Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.",
        "",
        f"- artifact: `{summary.get('output_json', '-')}`",
        f"- report: `{summary.get('report_md', '-')}`",
        f"- target: `{workload}/{config}`",
        f"- speed_claim_ready: {_fmt(goal_acceptance.get('speed_claim_ready'))}",
        f"- target_vllm_ratio_met: {_fmt(goal_acceptance.get('target_vllm_ratio_met'))}",
        f"- JAX/vLLM: {_fmt_ratio(goal_comparison.get('jax_over_vllm_throughput'))}",
        f"- JAX/reference: {_fmt_ratio(goal_comparison.get('jax_over_jax_reference_throughput'))}",
        f"- TTFT delta vs reference: {_fmt(goal_comparison.get('ttft_ms_p50_delta_vs_jax_reference'), suffix=' ms')}",
        f"- ITL delta vs reference: {_fmt(goal_comparison.get('itl_ms_p50_delta_vs_jax_reference'), suffix=' ms')}",
        "- profile movement to explain:",
    ]
    lines.extend(_profile_delta_bullets(goal_comparison, limit=top_profile_deltas))
    lines.extend(
        [
            "- interpretation: <explain whether the profile movement supports the claimed change>",
            "- decision: <keep/reject/follow up, with reason>",
        ]
    )
    return lines


def render_markdown(
    summary: dict[str, Any],
    *,
    top_profile_deltas: int = 8,
    top_scoped_events: int = 5,
) -> str:
    workload, config, goal_comparison, goal_acceptance = _goal_target_row(summary)
    target_ratio = (summary.get("goal_target") or {}).get("target_vllm_ratio")
    lines = [
        "# GPU Matrix Report",
        "",
        f"- created_at_utc: `{summary.get('created_at_utc', '-')}`",
        f"- dry_run: {_fmt(summary.get('dry_run'))}",
        f"- repeats: {_fmt(summary.get('repeats'))}",
        f"- run_dir: `{summary.get('run_dir', '-')}`",
        f"- output_json: `{summary.get('output_json', '-')}`",
    ]
    jax_python = summary.get("jax_python") or {}
    if jax_python:
        lines.append(
            f"- jax_python: `{jax_python.get('path', '-')}` "
            f"(available: {_fmt(jax_python.get('available'))})"
        )

    lines.extend(
        [
            "",
            "## Goal Target",
            "",
            f"- target: `{workload}/{config}`",
            f"- speed_claim_ready: {_fmt(goal_acceptance.get('speed_claim_ready'))}",
            f"- target_vllm_ratio_met: {_fmt(goal_acceptance.get('target_vllm_ratio_met'))}",
            f"- JAX/vLLM: {_fmt_ratio(goal_comparison.get('jax_over_vllm_throughput'))} "
            f"(target {_fmt_ratio(target_ratio)})",
            f"- JAX tok/s: {_fmt(goal_comparison.get('jax_tokens_per_second_median'))}",
            f"- vLLM tok/s: {_fmt(goal_comparison.get('vllm_tokens_per_second'))}",
            f"- target tok/s: {_fmt(goal_comparison.get('target_tokens_per_second'))}",
            f"- gap to target tok/s: {_fmt(goal_comparison.get('tokens_per_second_gap_to_target'))}",
            "",
            "## Matrix",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "workload",
                "config",
                "ready",
                "target met",
                "JAX tok/s",
                "vLLM tok/s",
                "JAX/vLLM",
                "target tok/s",
                "gap tok/s",
                "JAX ref tok/s",
                "JAX/ref",
            ],
            _matrix_rows(summary),
        )
    )

    scheduler_rows = _scheduler_rows(summary)
    if scheduler_rows:
        lines.extend(["", "## Scheduler Diagnostics", ""])
        lines.extend(
            _markdown_table(
                [
                    "workload",
                    "config",
                    "prefill steps",
                    "decode steps",
                    "max prefill seqs",
                    "max step tokens",
                    "prefill step s",
                    "decode step s",
                ],
                scheduler_rows,
            )
        )

    scoped_event_rows = _scoped_profile_event_rows(summary, limit=top_scoped_events)
    if scoped_event_rows:
        lines.extend(["", "## Top Scoped Profile Events", ""])
        lines.extend(
            _markdown_table(
                [
                    "workload",
                    "config",
                    "repeat",
                    "scope",
                    "event",
                    "total",
                    "count",
                ],
                scoped_event_rows,
            )
        )

    failures = acceptance_failures(summary)
    lines.extend(["", "## Acceptance Failures", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("None.")

    lines.extend(["", "## Top Profile Deltas Vs JAX Reference", ""])
    any_profile_rows = False
    for workload_name in summary.get("workloads") or []:
        for config_name in summary.get("configs") or []:
            comparison = ((summary.get("comparisons") or {}).get(workload_name) or {}).get(config_name) or {}
            rows = _profile_delta_rows(comparison, limit=top_profile_deltas)
            if not rows:
                continue
            any_profile_rows = True
            lines.extend([f"### `{workload_name}/{config_name}`", ""])
            lines.extend(
                _markdown_table(
                    [
                        "bucket",
                        "current",
                        "reference",
                        "delta",
                        "ratio",
                        "current count",
                        "reference count",
                        "count delta",
                    ],
                    rows,
                )
            )
            lines.append("")
    if not any_profile_rows:
        lines.append("No profile deltas available.")
    lines.extend(["", *_logbook_template(summary, top_profile_deltas=top_profile_deltas)])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    summary = _load_json(Path(args.summary_json))
    report = render_markdown(
        summary,
        top_profile_deltas=args.top_profile_deltas,
        top_scoped_events=args.top_scoped_events,
    )
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(output_path)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
