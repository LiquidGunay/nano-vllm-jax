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


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(summary: dict[str, Any], *, top_profile_deltas: int = 8) -> str:
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
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    summary = _load_json(Path(args.summary_json))
    report = render_markdown(summary, top_profile_deltas=args.top_profile_deltas)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(output_path)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
