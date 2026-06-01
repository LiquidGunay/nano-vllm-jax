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


def _fmt_hash(value: Any) -> str:
    if not value:
        return "-"
    text = str(value)
    return text[:12] if len(text) > 12 else text


def _fmt_prompt_value(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _fmt(value)


def _median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


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


def _profile_scoped_delta_rows(comparison: dict[str, Any], *, limit: int) -> list[list[str]]:
    deltas = comparison.get("profile_scoped_delta_vs_jax_reference") or {}
    rows = []
    for scope, scope_deltas in deltas.items():
        for name, bucket in (scope_deltas or {}).items():
            delta = bucket.get("total_ms_delta")
            if delta is None:
                continue
            rows.append(
                (
                    abs(float(delta)),
                    [
                        str(scope),
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


def _first_repeat_prompt(matrix_row: dict[str, Any]) -> dict[str, Any]:
    for repeat in matrix_row.get("repeats") or []:
        metrics = repeat.get("metrics") or {}
        prompt = metrics.get("prompt") or {}
        if prompt:
            return prompt
    return {}


def _prompt_provenance_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for workload in summary.get("workloads") or []:
        vllm_metrics = ((summary.get("vllm_references") or {}).get(workload) or {}).get("metrics") or {}
        vllm_prompt = vllm_metrics.get("prompt") or {}
        for config in summary.get("configs") or []:
            matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
            configured = matrix_row.get("workload") or {}
            current_prompt = _first_repeat_prompt(matrix_row)
            source = current_prompt.get("prompt_source") or configured.get("prompt_source")
            dataset = current_prompt.get("dataset_name") or configured.get("dataset_name")
            num_prompts = current_prompt.get("num_prompts") or configured.get("num_prompts")
            seed = current_prompt.get("seed")
            if seed is None:
                seed = configured.get("seed")
            current_manifest = current_prompt.get("prompt_manifest_sha256")
            vllm_manifest = vllm_prompt.get("prompt_manifest_sha256")
            manifest_match = (
                current_manifest == vllm_manifest
                if current_manifest and vllm_manifest
                else None
            )
            rows.append(
                [
                    str(workload),
                    str(config),
                    _fmt(source),
                    _fmt(dataset),
                    _fmt(num_prompts),
                    _fmt(seed),
                    _fmt(current_prompt.get("random_input_len") or configured.get("random_input_len")),
                    _fmt(current_prompt.get("random_output_len") or configured.get("random_output_len")),
                    _fmt_prompt_value(current_prompt.get("random_range_ratio") or configured.get("random_range_ratio")),
                    _fmt_hash(current_manifest),
                    _fmt_hash(vllm_manifest),
                    _fmt(manifest_match),
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


def _scoped_profile_range_rows(
    summary: dict[str, Any],
    *,
    scopes: tuple[str, ...] = ("gpu", "cpu"),
    limit: int = 8,
) -> list[list[str]]:
    rows: list[tuple[float, list[str]]] = []
    for workload in summary.get("workloads") or []:
        for config in summary.get("configs") or []:
            matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
            scoped_medians = (matrix_row.get("aggregate") or {}).get("profile_scoped_range_medians") or {}
            if not scoped_medians:
                scoped_medians = {}
                for scope in scopes:
                    buckets: dict[str, dict[str, list[float]]] = {}
                    for repeat in matrix_row.get("repeats") or []:
                        metrics = repeat.get("metrics") or {}
                        scoped_ranges = metrics.get("profile_scoped_ranges") or {}
                        for bucket, values in (scoped_ranges.get(scope) or {}).items():
                            bucket_values = buckets.setdefault(
                                str(bucket),
                                {"total_ms": [], "count": []},
                            )
                            if values.get("total_ms") is not None:
                                bucket_values["total_ms"].append(float(values["total_ms"]))
                            if values.get("count") is not None:
                                bucket_values["count"].append(float(values["count"]))
                    scoped_medians[scope] = {
                        bucket: {
                            "total_ms_median": _median(values["total_ms"]),
                            "count_median": _median(values["count"]),
                        }
                        for bucket, values in buckets.items()
                    }
            for scope in scopes:
                for bucket, values in (scoped_medians.get(scope) or {}).items():
                    total = values.get("total_ms_median")
                    if total is None:
                        continue
                    rows.append(
                        (
                            float(total),
                            [
                                str(workload),
                                str(config),
                                str(scope),
                                str(bucket),
                                _fmt(total, suffix=" ms"),
                                _fmt(values.get("count_median"), digits=1),
                            ],
                        )
                    )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[: max(0, int(limit))]]


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _host_replay_rows(summary: dict[str, Any]) -> list[list[str]]:
    """Expose launch/update density as a proxy for static replay movement."""

    buckets = (
        "forward_step_token_ids_jit",
        "PjRtCApiLoadedExecutable::Execute",
        "command_buffer::execute",
        "command_buffer::update",
        "np.asarray(jax.Array)",
    )
    rows: list[list[str]] = []
    for workload in summary.get("workloads") or []:
        for config in summary.get("configs") or []:
            matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
            aggregate = matrix_row.get("aggregate") or {}
            scheduler = aggregate.get("scheduler_diagnostics_median") or {}
            step_count = scheduler.get("step_count")
            decode_step_count = scheduler.get("decode_step_count")
            if not scheduler.get("available") or not step_count:
                continue
            cpu_ranges = (aggregate.get("profile_scoped_range_medians") or {}).get("cpu") or {}
            comparison = ((summary.get("comparisons") or {}).get(workload) or {}).get(config) or {}
            cpu_delta = (comparison.get("profile_scoped_delta_vs_jax_reference") or {}).get("cpu") or {}
            for bucket in buckets:
                current = cpu_ranges.get(bucket) or {}
                current_count = current.get("count_median")
                current_total = current.get("total_ms_median")
                if current_count is None and current_total is None:
                    continue
                reference = cpu_delta.get(bucket) or {}
                reference_count = reference.get("reference_count")
                reference_total = reference.get("reference_total_ms")
                rows.append(
                    [
                        str(workload),
                        str(config),
                        bucket,
                        _fmt(step_count, digits=1),
                        _fmt(decode_step_count, digits=1),
                        _fmt(current_count, digits=1),
                        _fmt(_safe_divide(current_count, step_count), digits=2),
                        _fmt(_safe_divide(current_total, step_count), suffix=" ms"),
                        _fmt(reference_count, digits=1),
                        _fmt(_safe_divide(reference_count, step_count), digits=2),
                        _fmt(_safe_divide(reference_total, step_count), suffix=" ms"),
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

    host_replay_rows = _host_replay_rows(summary)
    if host_replay_rows:
        lines.extend(["", "## Host Replay Diagnostics", ""])
        lines.extend(
            _markdown_table(
                [
                    "workload",
                    "config",
                    "bucket",
                    "steps",
                    "decode steps",
                    "count",
                    "count/step",
                    "ms/step",
                    "ref count",
                    "ref count/step",
                    "ref ms/step",
                ],
                host_replay_rows,
            )
        )

    prompt_rows = _prompt_provenance_rows(summary)
    if prompt_rows:
        lines.extend(["", "## Prompt Provenance", ""])
        lines.extend(
            _markdown_table(
                [
                    "workload",
                    "config",
                    "source",
                    "dataset",
                    "prompts",
                    "seed",
                    "random input",
                    "random output",
                    "range ratio",
                    "current manifest",
                    "vLLM manifest",
                    "manifest match",
                ],
                prompt_rows,
            )
        )

    scoped_range_rows = _scoped_profile_range_rows(summary, limit=top_profile_deltas)
    if scoped_range_rows:
        lines.extend(["", "## Scoped Profile Range Medians", ""])
        lines.extend(
            _markdown_table(
                [
                    "workload",
                    "config",
                    "scope",
                    "bucket",
                    "median total",
                    "median count",
                ],
                scoped_range_rows,
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

    lines.extend(["", "## Scoped Profile Deltas Vs JAX Reference", ""])
    any_scoped_profile_rows = False
    for workload_name in summary.get("workloads") or []:
        for config_name in summary.get("configs") or []:
            comparison = ((summary.get("comparisons") or {}).get(workload_name) or {}).get(config_name) or {}
            rows = _profile_scoped_delta_rows(comparison, limit=top_profile_deltas)
            if not rows:
                continue
            any_scoped_profile_rows = True
            lines.extend([f"### `{workload_name}/{config_name}`", ""])
            lines.extend(
                _markdown_table(
                    [
                        "scope",
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
    if not any_scoped_profile_rows:
        lines.append("No scoped profile deltas available.")
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
