#!/usr/bin/env python3
"""Summarize Chrome/Perfetto trace event durations from JAX profiles."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = (
    "PjRtCApiLoadedExecutable::Execute",
    "command_buffer::execute",
    "command_buffer::update",
    "forward_step_token_ids_jit",
    "cutlass",
    "gemm_fusion",
    "input_reduce_fusion",
    "loop_dynamic_slice",
    "loop_dynamic_update_slice",
    "wrapped_concatenate",
    "while",
    "MemcpyD2D",
    "gather",
    "transpose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json_gz", nargs="+", help="Path(s) to *.trace.json.gz")
    parser.add_argument("--scope", choices=("all", "gpu", "cpu"), default="all")
    parser.add_argument("--top-events", type=int, default=30)
    parser.add_argument("--top-hlo-ops", type=int, default=20)
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Substring to aggregate. Can be passed more than once.",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON output path")
    parser.add_argument("--output-md", default="", help="Optional Markdown output path")
    return parser.parse_args()


def _read_trace(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _process_names(events: Iterable[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, str] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            names[int(event.get("pid", -1))] = str((event.get("args") or {}).get("name") or "")
    return names


def _event_in_scope(event: dict[str, Any], process_names: dict[int, str], scope: str) -> bool:
    if scope == "all":
        return True
    process = process_names.get(int(event.get("pid", -1)), "")
    if scope == "gpu":
        return process.startswith("/device:GPU")
    return process.startswith("/host:CPU")


def _event_rows(totals: dict[str, list[float | int]], limit: int) -> list[dict[str, Any]]:
    rows = [
        {
            "name": name,
            "total_ms": float(total),
            "count": int(count),
        }
        for name, (total, count) in totals.items()
        if float(total) > 0.0
    ]
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows[: max(0, int(limit))]


def _hlo_rows(totals: dict[tuple[str, str, str], list[Any]], limit: int) -> list[dict[str, Any]]:
    rows = [
        {
            "hlo_module": module,
            "hlo_op": hlo_op,
            "event": event_name,
            "total_ms": float(total),
            "count": int(count),
            "kernel_details": str(kernel_details or ""),
        }
        for (module, hlo_op, event_name), (total, count, kernel_details) in totals.items()
        if float(total) > 0.0
    ]
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows[: max(0, int(limit))]


def summarize_trace(
    path: Path,
    *,
    scope: str = "all",
    top_events: int = 30,
    top_hlo_ops: int = 20,
    patterns: Iterable[str] = DEFAULT_PATTERNS,
) -> dict[str, Any]:
    trace = _read_trace(path)
    events = list(trace.get("traceEvents") or [])
    process_names = _process_names(events)
    event_totals: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    pattern_totals: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    hlo_totals: dict[tuple[str, str, str], list[Any]] = defaultdict(lambda: [0.0, 0, ""])
    pattern_list = tuple(dict.fromkeys(str(pattern) for pattern in patterns if str(pattern)))

    for event in events:
        duration = event.get("dur")
        if duration is None or not _event_in_scope(event, process_names, scope):
            continue
        name = str(event.get("name") or "")
        duration_ms = float(duration) / 1000.0
        event_totals[name][0] += duration_ms
        event_totals[name][1] += 1
        args = event.get("args") or {}
        hlo_op = str(args.get("hlo_op") or "")
        if hlo_op:
            hlo_module = str(args.get("hlo_module") or "")
            hlo_key = (hlo_module, hlo_op, name)
            hlo_totals[hlo_key][0] += duration_ms
            hlo_totals[hlo_key][1] += 1
            if not hlo_totals[hlo_key][2]:
                hlo_totals[hlo_key][2] = str(args.get("kernel_details") or "")
        for pattern in pattern_list:
            if pattern in name:
                pattern_totals[pattern][0] += duration_ms
                pattern_totals[pattern][1] += 1

    return {
        "trace_json_gz": str(path),
        "scope": scope,
        "top_events_by_total_ms": _event_rows(event_totals, top_events),
        "top_hlo_ops_by_total_ms": _hlo_rows(hlo_totals, top_hlo_ops),
        "patterns": {
            pattern: {
                "total_ms": float(total),
                "count": int(count),
            }
            for pattern, (total, count) in sorted(pattern_totals.items())
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Profile Trace Summary", ""]
    for trace in summary["traces"]:
        lines.extend(
            [
                f"## `{trace['trace_json_gz']}`",
                "",
                f"- scope: `{trace['scope']}`",
                "",
                "### Pattern Totals",
                "",
            ]
        )
        pattern_rows = trace.get("patterns") or {}
        if pattern_rows:
            lines.extend(["| pattern | total ms | count |", "| --- | ---: | ---: |"])
            for pattern, row in pattern_rows.items():
                lines.append(f"| `{pattern}` | {row['total_ms']:.2f} | {row['count']} |")
        else:
            lines.append("No pattern rows.")
        lines.extend(["", "### Top Events", ""])
        top_rows = trace.get("top_events_by_total_ms") or []
        if top_rows:
            lines.extend(["| event | total ms | count |", "| --- | ---: | ---: |"])
            for row in top_rows:
                lines.append(f"| `{row['name']}` | {row['total_ms']:.2f} | {row['count']} |")
        else:
            lines.append("No event rows.")
        lines.extend(["", "### Top HLO Ops", ""])
        hlo_rows = trace.get("top_hlo_ops_by_total_ms") or []
        if hlo_rows:
            lines.extend(
                [
                    "| HLO module | HLO op | event | total ms | count | kernel details |",
                    "| --- | --- | --- | ---: | ---: | --- |",
                ]
            )
            for row in hlo_rows:
                details = str(row.get("kernel_details") or "")
                lines.append(
                    f"| `{row['hlo_module']}` | `{row['hlo_op']}` | `{row['event']}` | "
                    f"{row['total_ms']:.2f} | {row['count']} | `{details}` |"
                )
        else:
            lines.append("No HLO rows.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    summary = {
        "traces": [
            summarize_trace(
                Path(path),
                scope=args.scope,
                top_events=args.top_events,
                top_hlo_ops=args.top_hlo_ops,
                patterns=patterns,
            )
            for path in args.trace_json_gz
        ],
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(render_markdown(summary), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(render_markdown(summary), end="")


if __name__ == "__main__":
    main()
