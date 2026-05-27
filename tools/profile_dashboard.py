#!/usr/bin/env python3
"""Serve a lightweight dashboard for GPU matrix results and JAX profiles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PROFILE_ROOT = Path("/mountpoint/.exp/profiles")
DEFAULT_BEST = "results/gpu_matrix_20260527_device_token_carry_vector_target.json"
CACHE_DIR = Path("/mountpoint/.exp/profile_dashboard_cache")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rel(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def _as_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _safe_file(path_text: str) -> Path | None:
    candidate = _as_path(unquote(path_text))
    if candidate is None:
        return None
    try:
        resolved = candidate.resolve()
        allowed = (REPO_ROOT.resolve(), PROFILE_ROOT.resolve())
        if any(resolved == root or root in resolved.parents for root in allowed):
            return resolved
    except OSError:
        return None
    return None


def _read_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return _load_json(path)


def _trace_cache_path(path: Path) -> Path:
    stat = path.stat()
    key = hashlib.sha256(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.timeline.json"


def _process_names(events: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, str] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            try:
                names[int(event.get("pid", -1))] = str((event.get("args") or {}).get("name") or "")
            except (TypeError, ValueError):
                continue
    return names


def _scope_for_event(event: dict[str, Any], process_names: dict[int, str]) -> str:
    try:
        process = process_names.get(int(event.get("pid", -1)), "")
    except (TypeError, ValueError):
        process = ""
    if process.startswith("/device:GPU"):
        return "gpu"
    if process.startswith("/host:CPU"):
        return "cpu"
    return "other"


def summarize_trace_timeline(path: Path, *, limit_per_scope: int = 90) -> dict[str, Any]:
    cache_path = _trace_cache_path(path)
    if cache_path.exists():
        try:
            return _load_json(cache_path)
        except (OSError, json.JSONDecodeError):
            pass

    trace = _read_trace(path)
    trace_events = [event for event in trace.get("traceEvents") or [] if isinstance(event, dict)]
    process_names = _process_names(trace_events)
    duration_events: list[dict[str, Any]] = []
    min_ts: float | None = None
    max_ts: float | None = None
    totals: dict[str, dict[str, float]] = {
        "cpu": {"dur_us": 0.0, "count": 0.0},
        "gpu": {"dur_us": 0.0, "count": 0.0},
        "other": {"dur_us": 0.0, "count": 0.0},
    }

    for event in trace_events:
        dur = event.get("dur")
        ts = event.get("ts")
        if dur is None or ts is None:
            continue
        try:
            dur_us = float(dur)
            ts_us = float(ts)
        except (TypeError, ValueError):
            continue
        if dur_us <= 0.0:
            continue
        scope = _scope_for_event(event, process_names)
        totals.setdefault(scope, {"dur_us": 0.0, "count": 0.0})
        totals[scope]["dur_us"] += dur_us
        totals[scope]["count"] += 1
        min_ts = ts_us if min_ts is None else min(min_ts, ts_us)
        max_ts = ts_us + dur_us if max_ts is None else max(max_ts, ts_us + dur_us)
        duration_events.append(
            {
                "name": str(event.get("name") or ""),
                "scope": scope,
                "ts_us": ts_us,
                "dur_us": dur_us,
                "pid": event.get("pid"),
                "tid": event.get("tid"),
            }
        )

    if min_ts is None or max_ts is None:
        summary = {"trace": str(path), "duration_ms": 0.0, "events": [], "totals": totals}
    else:
        selected: list[dict[str, Any]] = []
        for scope in ("gpu", "cpu", "other"):
            scope_events = [event for event in duration_events if event["scope"] == scope]
            scope_events.sort(key=lambda event: float(event["dur_us"]), reverse=True)
            selected.extend(scope_events[:limit_per_scope])
        selected.sort(key=lambda event: (float(event["ts_us"]), str(event["scope"])))
        span_us = max(max_ts - min_ts, 1.0)
        summary = {
            "trace": str(path),
            "file_name": path.name,
            "event_count": len(duration_events),
            "duration_ms": span_us / 1000.0,
            "totals": {
                scope: {
                    "total_ms": values["dur_us"] / 1000.0,
                    "count": int(values["count"]),
                }
                for scope, values in totals.items()
            },
            "events": [
                {
                    "name": event["name"],
                    "scope": event["scope"],
                    "start_ms": (float(event["ts_us"]) - min_ts) / 1000.0,
                    "duration_ms": float(event["dur_us"]) / 1000.0,
                    "start_pct": (float(event["ts_us"]) - min_ts) / span_us * 100.0,
                    "width_pct": max(float(event["dur_us"]) / span_us * 100.0, 0.08),
                    "pid": event["pid"],
                    "tid": event["tid"],
                }
                for event in selected
            ],
        }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _short_name(path: str | None) -> str:
    if not path:
        return "-"
    return Path(path).name


def _flags_from_metrics(metrics: dict[str, Any]) -> str:
    run_config = metrics.get("run_config") or {}
    names: list[str] = []
    for group_name in (
        "serving_fastpath_flags",
        "gdn_kernel_flags",
        "full_attention_kernel_flags",
    ):
        flags = run_config.get(group_name) or {}
        for key, value in sorted(flags.items()):
            if value is True:
                names.append(key)
            elif isinstance(value, str) and value not in ("", "off", "false", "0"):
                names.append(f"{key}={value}")
    kernel = run_config.get("kernel_backend_resolved")
    if kernel:
        names.append(f"kernel={kernel}")
    return ", ".join(dict.fromkeys(names)) or "-"


def _artifact_for_repeat(run_dir: str | None, workload: str, config: str, repeat: Any) -> str | None:
    if not run_dir:
        return None
    repeat_id = repeat.get("repeat")
    artifact = Path(run_dir) / f"{workload}_{config}_repeat{repeat_id}.json"
    return str(artifact) if artifact.exists() else None


def _profile_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, bucket in (comparison.get("profile_delta_vs_jax_reference") or {}).items():
        rows.append(
            {
                "bucket": name,
                "current_ms": bucket.get("current_total_ms_median"),
                "reference_ms": bucket.get("reference_total_ms"),
                "delta_ms": bucket.get("total_ms_delta"),
                "ratio": bucket.get("total_ms_ratio"),
                "current_count": bucket.get("current_count_median"),
                "reference_count": bucket.get("reference_count"),
            }
        )
    rows.sort(key=lambda row: abs(float(row["delta_ms"] or 0.0)), reverse=True)
    return rows


def _scoped_range_rows(matrix_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    scoped = (matrix_row.get("aggregate") or {}).get("profile_scoped_range_medians") or {}
    for scope, buckets in scoped.items():
        for bucket, values in (buckets or {}).items():
            total = values.get("total_ms_median")
            if total is None:
                continue
            rows.append(
                {
                    "scope": scope,
                    "bucket": bucket,
                    "total_ms": total,
                    "count": values.get("count_median"),
                }
            )
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def _repeat_rows(summary: dict[str, Any], workload: str, config: str) -> list[dict[str, Any]]:
    matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
    rows = []
    for repeat in matrix_row.get("repeats") or []:
        metrics = repeat.get("metrics") or {}
        performance = metrics.get("performance") or {}
        trace = metrics.get("profile_trace_json_gz")
        artifact = _artifact_for_repeat(summary.get("run_dir"), workload, config, repeat)
        artifact_metrics = metrics
        artifact_path = _as_path(artifact)
        if artifact_path and artifact_path.exists():
            try:
                artifact_metrics = _load_json(artifact_path)
            except (OSError, json.JSONDecodeError):
                artifact_metrics = metrics
        scoped_events = metrics.get("profile_scoped_top_events_by_total_ms") or {}
        if not scoped_events:
            scoped_events = artifact_metrics.get("profile_scoped_top_events_by_total_ms") or {}
        if not trace:
            trace = artifact_metrics.get("profile_trace_json_gz")
        performance = performance or artifact_metrics.get("performance") or {}
        rows.append(
            {
                "repeat": repeat.get("repeat"),
                "status": (repeat.get("run") or {}).get("status"),
                "returncode": (repeat.get("run") or {}).get("returncode"),
                "tokens_per_second": performance.get("tokens_per_second"),
                "ttft_ms_p50": performance.get("ttft_ms_p50"),
                "itl_ms_p50": performance.get("itl_ms_p50"),
                "flags": _flags_from_metrics(artifact_metrics),
                "artifact": _rel(artifact),
                "profile_path": metrics.get("run", {}).get("profile_path"),
                "trace": trace,
                "top_gpu_events": (scoped_events.get("gpu") or [])[:12],
                "top_cpu_events": (scoped_events.get("cpu") or [])[:12],
            }
        )
    return rows


def build_data() -> dict[str, Any]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("gpu_matrix*.json")):
        try:
            summary = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            runs.append({"id": str(path.relative_to(REPO_ROOT)), "error": str(exc)})
            continue
        run_id = str(path.relative_to(REPO_ROOT))
        run_rows = []
        for workload in summary.get("workloads") or []:
            for config in summary.get("configs") or []:
                comparison = ((summary.get("comparisons") or {}).get(workload) or {}).get(config) or {}
                acceptance = ((summary.get("acceptance") or {}).get(workload) or {}).get(config) or {}
                matrix_row = ((summary.get("matrix") or {}).get(workload) or {}).get(config) or {}
                repeat_rows = _repeat_rows(summary, str(workload), str(config))
                run_rows.append(
                    {
                        "workload": workload,
                        "config": config,
                        "ready": acceptance.get("speed_claim_ready"),
                        "target_met": acceptance.get("target_vllm_ratio_met"),
                        "checks": acceptance.get("checks") or {},
                        "missing_profile_counters": acceptance.get("missing_profile_counters") or [],
                        "jax_tokens_per_second": comparison.get("jax_tokens_per_second_median"),
                        "vllm_tokens_per_second": comparison.get("vllm_tokens_per_second"),
                        "jax_over_vllm": comparison.get("jax_over_vllm_throughput"),
                        "target_tokens_per_second": comparison.get("target_tokens_per_second"),
                        "gap_to_target": comparison.get("tokens_per_second_gap_to_target"),
                        "jax_reference_tokens_per_second": comparison.get("jax_reference_tokens_per_second"),
                        "jax_over_reference": comparison.get("jax_over_jax_reference_throughput"),
                        "flags": (repeat_rows[0].get("flags") if repeat_rows else "-"),
                        "profile_deltas": _profile_rows(comparison),
                        "scoped_ranges": _scoped_range_rows(matrix_row),
                        "repeats": repeat_rows,
                    }
                )
        runs.append(
            {
                "id": run_id,
                "title": path.stem,
                "created_at_utc": summary.get("created_at_utc"),
                "run_dir": summary.get("run_dir"),
                "output_json": _rel(summary.get("output_json") or path),
                "report_md": _rel(summary.get("report_md")),
                "goal_target": summary.get("goal_target") or {},
                "rows": run_rows,
                "is_best": run_id == DEFAULT_BEST,
            }
        )
    runs.sort(key=lambda row: (not row.get("is_best"), row.get("created_at_utc") or "", row.get("id") or ""), reverse=False)
    return {"repo_root": str(REPO_ROOT), "profile_root": str(PROFILE_ROOT), "default_best": DEFAULT_BEST, "runs": runs}


def _trace_instruction(trace: str | None) -> str:
    if not trace:
        return "No trace path recorded for this repeat."
    return (
        "Open https://ui.perfetto.dev, choose Open trace file, then select "
        f"{trace}. The raw trace can also be downloaded from this dashboard."
    )


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nano-vLLM JAX Profile Dashboard</title>
<style>
:root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --line:#d9dee7; --text:#171f2a; --muted:#647184; --accent:#0f766e; --blue:#2563eb; --amber:#b45309; --bad:#b42318; --ok:#027a48; --gpu:#7c3aed; --cpu:#0f766e; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
header { padding:18px 28px 14px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:2; }
h1 { margin:0 0 10px; font-size:23px; font-weight:680; letter-spacing:0; }
h2 { margin:0 0 12px; font-size:16px; }
h3 { margin:12px 0 8px; font-size:14px; }
main { padding:18px 28px 40px; max-width:1720px; }
.toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
label { color:var(--muted); font-size:12px; }
select, input { display:block; margin-top:3px; font:inherit; padding:7px 9px; border:1px solid var(--line); border-radius:6px; background:#fff; min-width:250px; }
input { min-width:320px; }
.layout { display:grid; grid-template-columns:370px minmax(0,1fr); gap:16px; align-items:start; }
.stack { display:grid; gap:16px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.metric { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; min-height:78px; }
.metric .label { color:var(--muted); font-size:12px; }
.metric .value { margin-top:4px; font-size:22px; font-weight:680; font-variant-numeric:tabular-nums; }
.metric .sub { color:var(--muted); font-size:12px; margin-top:2px; }
.run-list { display:grid; gap:8px; max-height:470px; overflow:auto; padding-right:2px; }
.run-card { border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; cursor:pointer; }
.run-card.active { border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.13); }
.run-card .title { font-weight:650; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.run-card .bars { display:grid; gap:4px; margin-top:8px; }
.mini-bar { height:8px; background:#e8edf4; border-radius:999px; overflow:hidden; }
.mini-bar span { display:block; height:100%; background:var(--blue); }
.mini-bar .target { background:var(--accent); }
.pill { display:inline-block; padding:2px 7px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:#f6f8fb; white-space:nowrap; }
.pill.ok { color:var(--ok); border-color:#84c7a2; background:#ecfdf3; }
.pill.bad { color:var(--bad); border-color:#f1a29b; background:#fff1f0; }
.pill.warn { color:var(--amber); border-color:#e7bd7e; background:#fff8eb; }
.muted { color:var(--muted); }
.small { font-size:12px; }
a { color:#075985; text-decoration:none; }
a:hover { text-decoration:underline; }
code { background:#eef2f6; padding:1px 4px; border-radius:4px; word-break:break-all; }
.links { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.button { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:6px; padding:6px 9px; background:#fff; color:#17324d; }
.chart { width:100%; min-height:260px; border:1px solid var(--line); border-radius:8px; background:#fff; overflow:hidden; }
.chart svg { width:100%; height:auto; display:block; }
.bar-row { display:grid; grid-template-columns:minmax(140px,270px) minmax(0,1fr) 82px; gap:10px; align-items:center; padding:7px 0; border-bottom:1px solid #edf0f5; }
.bar-row:last-child { border-bottom:0; }
.bar-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar-track { height:12px; background:#e8edf4; border-radius:999px; overflow:hidden; }
.bar-fill { height:100%; background:var(--blue); border-radius:999px; }
.bar-fill.gpu { background:var(--gpu); }
.bar-fill.cpu { background:var(--cpu); }
.bar-fill.neg { background:var(--ok); }
.bar-fill.pos { background:var(--bad); }
.bar-value { text-align:right; font-variant-numeric:tabular-nums; color:#263241; }
.split { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }
.trace-box { background:#fbfcfe; border:1px solid var(--line); border-radius:8px; padding:10px; }
.timeline { border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; overflow:hidden; }
.timeline svg { width:100%; height:auto; display:block; }
.timeline-help { margin-top:8px; color:var(--muted); font-size:12px; }
table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }
th, td { padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#eef2f6; color:#263241; font-weight:650; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.scroll { overflow:auto; max-height:290px; border:1px solid var(--line); border-radius:8px; }
@media (max-width: 1100px) { .layout, .split { grid-template-columns:1fr; } .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 700px) { main, header { padding-left:14px; padding-right:14px; } .metric-grid { grid-template-columns:1fr; } select, input { min-width:100%; } .bar-row { grid-template-columns:1fr; gap:4px; } .bar-value { text-align:left; } }
</style>
</head>
<body>
<header>
  <h1>nano-vLLM JAX Profile Dashboard</h1>
  <div class="toolbar">
    <label>Run <select id="runSelect"></select></label>
    <label>Row <select id="rowSelect"></select></label>
    <label>Repeat <select id="repeatSelect"></select></label>
    <input id="filter" placeholder="Filter leaderboard">
  </div>
</header>
<main>
  <div class="layout">
    <aside class="panel">
      <h2>Run Comparison</h2>
      <div class="small muted">Sorted with the current best run first. Bars show JAX throughput and target gap.</div>
      <div id="runCards" class="run-list"></div>
    </aside>
    <div class="stack">
      <section class="metric-grid" id="metrics"></section>
      <section class="panel">
        <h2>Throughput Against vLLM Target</h2>
        <div id="throughputChart" class="chart"></div>
      </section>
      <section class="split">
        <div class="panel">
          <h2>Profile Movement Vs JAX Reference</h2>
          <div id="deltaBars"></div>
        </div>
        <div class="panel">
          <h2>Scoped Time Buckets</h2>
          <div id="rangeBars"></div>
        </div>
      </section>
      <section class="split">
        <div class="panel">
          <h2>Repeat Trace Navigation</h2>
          <div id="repeatDetail"></div>
        </div>
        <div class="panel">
          <h2>Top Events In Selected Trace</h2>
          <div id="eventBars"></div>
        </div>
      </section>
      <section class="panel">
        <h2>Compact Trace Timeline</h2>
        <div id="traceTimeline" class="timeline"></div>
      </section>
      <section class="panel">
        <h2>Artifacts</h2>
        <div id="artifactDetail"></div>
      </section>
      <section class="panel">
        <h2>Compact Leaderboard</h2>
        <div class="scroll"><table id="leaderboard"></table></div>
      </section>
    </div>
  </div>
</main>
<script>
const fmt = (v, digits=2) => v === null || v === undefined ? '-' : (typeof v === 'number' ? v.toFixed(digits) : String(v));
const boolPill = (v) => `<span class="pill ${v ? 'ok' : 'bad'}">${v ? 'yes' : 'no'}</span>`;
const fileLink = (p, text) => p ? `<a href="/file?path=${encodeURIComponent(p)}">${text || p.split('/').pop()}</a>` : '-';
const traceLink = (p) => p ? `<a class="button" href="/file?path=${encodeURIComponent(p)}">Download trace</a>` : '-';
let DATA = null;
let activeTraceRequest = 0;

function table(el, headers, rows) {
  const head = `<thead><tr>${headers.map(h => `<th class="${h.num ? 'num' : ''}">${h.label}</th>`).join('')}</tr></thead>`;
  const body = rows.length ? rows.map(r => `<tr>${r.map((c, i) => `<td class="${headers[i].num ? 'num' : ''}">${c}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${headers.length}" class="muted">No rows.</td></tr>`;
  el.innerHTML = head + '<tbody>' + body + '</tbody>';
}

function allRows() {
  const rows = [];
  for (const run of DATA.runs) for (const row of run.rows || []) rows.push({run, row});
  return rows;
}

function firstRow(run) {
  return (run.rows || [])[0] || {};
}

function selectedRun() {
  return DATA.runs.find(r => r.id === document.getElementById('runSelect').value) || DATA.runs[0];
}

function selectedRow() {
  const run = selectedRun();
  return run.rows[Number(document.getElementById('rowSelect').value || 0)] || run.rows[0] || {};
}

function selectedRepeat() {
  const row = selectedRow();
  return (row.repeats || [])[Number(document.getElementById('repeatSelect').value || 0)] || {};
}

function barRows(el, rows, valueKey, options={}) {
  const limit = options.limit || rows.length;
  const clean = rows.filter(r => r[valueKey] !== null && r[valueKey] !== undefined).slice(0, limit);
  if (!clean.length) {
    el.innerHTML = '<p class="muted">No rows.</p>';
    return;
  }
  const max = Math.max(...clean.map(r => Math.abs(Number(r[valueKey]))), 1);
  el.innerHTML = clean.map(r => {
    const value = Number(r[valueKey]);
    const width = Math.max(1, Math.abs(value) / max * 100);
    const cls = options.classFor ? options.classFor(r, value) : '';
    const label = options.label ? options.label(r) : r.name || r.bucket;
    const valueText = options.value ? options.value(r, value) : fmt(value);
    return `<div class="bar-row" title="${label}">
      <div class="bar-label">${label}</div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${width}%"></div></div>
      <div class="bar-value">${valueText}</div>
    </div>`;
  }).join('');
}

function renderRunCards() {
  const filter = document.getElementById('filter').value.toLowerCase();
  const maxTok = Math.max(...allRows().map(x => Number(x.row.jax_tokens_per_second || 0)), 1);
  const cards = [];
  for (const run of DATA.runs) {
    const row = firstRow(run);
    const text = `${run.id} ${row.workload || ''} ${row.config || ''} ${row.flags || ''}`.toLowerCase();
    if (filter && !text.includes(filter)) continue;
    const active = run.id === selectedRun().id ? ' active' : '';
    const tokWidth = Math.max(0, Number(row.jax_tokens_per_second || 0)) / maxTok * 100;
    const ratio = row.jax_over_vllm === null || row.jax_over_vllm === undefined ? '-' : `${fmt(row.jax_over_vllm, 3)}x`;
    cards.push(`<div class="run-card${active}" data-run-id="${run.id}">
      <div class="title">${run.is_best ? '<span class="pill ok">best</span> ' : ''}${run.title}</div>
      <div class="small muted">${run.created_at_utc || '-'} · ${row.workload || '-'} / ${row.config || '-'}</div>
      <div class="bars"><div class="mini-bar"><span style="width:${tokWidth}%"></span></div></div>
      <div class="small">JAX ${fmt(row.jax_tokens_per_second)} tok/s · vLLM ratio ${ratio} · gap ${fmt(row.gap_to_target)}</div>
    </div>`);
  }
  const container = document.getElementById('runCards');
  container.innerHTML = cards.join('') || '<p class="muted">No matching runs.</p>';
  container.querySelectorAll('.run-card').forEach(card => card.addEventListener('click', () => {
    document.getElementById('runSelect').value = card.dataset.runId;
    renderRowSelector();
  }));
}

function renderSelectors() {
  const runSelect = document.getElementById('runSelect');
  runSelect.innerHTML = DATA.runs.map(r => `<option value="${r.id}">${r.is_best ? 'BEST - ' : ''}${r.created_at_utc || ''} ${r.title}</option>`).join('');
  runSelect.value = DATA.default_best;
  renderRowSelector();
}

function renderRowSelector() {
  const run = selectedRun();
  const rowSelect = document.getElementById('rowSelect');
  rowSelect.innerHTML = (run.rows || []).map((r, i) => `<option value="${i}">${r.workload}/${r.config}</option>`).join('');
  renderRepeatSelector();
}

function renderRepeatSelector() {
  const row = selectedRow();
  const repeatSelect = document.getElementById('repeatSelect');
  repeatSelect.innerHTML = (row.repeats || []).map((r, i) => `<option value="${i}">repeat ${r.repeat || i + 1}</option>`).join('');
  renderAll();
}

function renderMetrics() {
  const row = selectedRow();
  const run = selectedRun();
  const readyClass = row.ready ? 'ok' : 'bad';
  const targetClass = row.target_met ? 'ok' : 'warn';
  document.getElementById('metrics').innerHTML = `
    <div class="metric"><div class="label">Selected run</div><div class="value" style="font-size:16px">${run.is_best ? 'Best run' : run.created_at_utc || '-'}</div><div class="sub">${run.title}</div></div>
    <div class="metric"><div class="label">JAX throughput</div><div class="value">${fmt(row.jax_tokens_per_second)}</div><div class="sub">tok/s, target ${fmt(row.target_tokens_per_second)}</div></div>
    <div class="metric"><div class="label">vLLM ratio</div><div class="value">${fmt(row.jax_over_vllm, 3)}x</div><div class="sub">vLLM ${fmt(row.vllm_tokens_per_second)} tok/s</div></div>
    <div class="metric"><div class="label">Readiness</div><div class="value"><span class="pill ${readyClass}">ready ${row.ready ? 'yes' : 'no'}</span> <span class="pill ${targetClass}">target ${row.target_met ? 'met' : 'gap'}</span></div><div class="sub">gap ${fmt(row.gap_to_target)} tok/s</div></div>
  `;
}

function renderThroughputChart() {
  const selected = selectedRun().id;
  const rows = allRows().filter(x => x.row.jax_tokens_per_second !== null && x.row.jax_tokens_per_second !== undefined).slice(0, 18);
  const width = 980, left = 210, right = 28, top = 20, rowH = 34;
  const height = Math.max(120, top * 2 + rows.length * rowH);
  const max = Math.max(...rows.flatMap(x => [Number(x.row.jax_tokens_per_second || 0), Number(x.row.vllm_tokens_per_second || 0), Number(x.row.target_tokens_per_second || 0)]), 1);
  const scale = v => left + Number(v || 0) / max * (width - left - right);
  const lines = rows.map((x, i) => {
    const y = top + i * rowH + 8;
    const active = x.run.id === selected;
    const label = `${x.run.is_best ? 'BEST ' : ''}${x.run.created_at_utc || ''} ${x.run.title.replace('gpu_matrix_', '')}`.slice(0, 48);
    const targetX = scale(x.row.target_tokens_per_second);
    return `<g>
      <text x="8" y="${y + 10}" font-size="11" fill="${active ? '#0f766e' : '#344054'}">${label}</text>
      <rect x="${left}" y="${y}" width="${Math.max(1, scale(x.row.vllm_tokens_per_second)-left)}" height="8" fill="#c7d2fe"></rect>
      <rect x="${left}" y="${y + 11}" width="${Math.max(1, scale(x.row.jax_tokens_per_second)-left)}" height="10" fill="${active ? '#0f766e' : '#2563eb'}"></rect>
      ${x.row.target_tokens_per_second ? `<line x1="${targetX}" x2="${targetX}" y1="${y - 2}" y2="${y + 24}" stroke="#b45309" stroke-width="2"></line>` : ''}
      <text x="${width - 8}" y="${y + 14}" text-anchor="end" font-size="11" fill="#344054">${fmt(x.row.jax_tokens_per_second)} tok/s · ${fmt(x.row.jax_over_vllm, 3)}x</text>
    </g>`;
  }).join('');
  document.getElementById('throughputChart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Throughput chart">
    <rect width="${width}" height="${height}" fill="#fff"></rect>
    ${lines}
    <g transform="translate(${left},${height - 16})"><rect width="12" height="8" fill="#2563eb"></rect><text x="18" y="8" font-size="11">JAX</text><rect x="64" width="12" height="8" fill="#c7d2fe"></rect><text x="82" y="8" font-size="11">vLLM</text><line x1="136" x2="136" y1="-2" y2="10" stroke="#b45309" stroke-width="2"></line><text x="146" y="8" font-size="11">target</text></g>
  </svg>`;
}

function renderLeaderboard() {
  const filter = document.getElementById('filter').value.toLowerCase();
  const rows = [];
  for (const run of DATA.runs) {
    for (const row of run.rows || []) {
      const text = `${run.id} ${row.workload} ${row.config} ${row.flags}`.toLowerCase();
      if (filter && !text.includes(filter)) continue;
      rows.push([
        run.is_best ? '<span class="pill ok">best</span>' : '',
        fileLink(run.output_json, run.title),
        row.workload,
        row.config,
        boolPill(row.ready),
        boolPill(row.target_met),
        fmt(row.jax_tokens_per_second),
        fmt(row.vllm_tokens_per_second),
        fmt(row.jax_over_vllm, 3) + 'x',
        fmt(row.gap_to_target),
        `<span class="small">${row.flags}</span>`,
        fileLink(run.report_md, 'report')
      ]);
    }
  }
  table(document.getElementById('leaderboard'), [
    {label:''}, {label:'artifact'}, {label:'workload'}, {label:'config'}, {label:'ready'}, {label:'target'}, {label:'JAX tok/s', num:true}, {label:'vLLM tok/s', num:true}, {label:'JAX/vLLM', num:true}, {label:'gap', num:true}, {label:'flags'}, {label:'links'}
  ], rows);
}

function renderArtifacts() {
  const run = selectedRun();
  const row = selectedRow();
  const missing = (row.missing_profile_counters || []).map(x => `<code>${x}</code>`).join(' ');
  document.getElementById('artifactDetail').innerHTML = `
    <div class="links">${fileLink(run.output_json, 'Matrix JSON')} ${fileLink(run.report_md, 'Markdown report')}</div>
    <p class="muted small">run_dir: <code>${run.run_dir || '-'}</code></p>
    <p><b>Target:</b> <code>${row.workload || '-'}/${row.config || '-'}</code> · <b>JAX/reference:</b> ${fmt(row.jax_over_reference, 3)}x</p>
    <p><b>Flags:</b> <span class="small">${row.flags || '-'}</span></p>
    <p><b>Missing profile counters:</b> ${missing || '-'}</p>
  `;
}

function renderRepeatDetail() {
  const repeat = selectedRepeat();
  const traceInstruction = repeat.trace ? `Open <a href="https://ui.perfetto.dev" target="_blank" rel="noreferrer">Perfetto</a>, choose Open trace file, then select <code>${repeat.trace}</code>.` : 'No trace path recorded for this repeat.';
  document.getElementById('repeatDetail').innerHTML = `
    <div class="metric-grid" style="grid-template-columns:repeat(3,minmax(0,1fr)); margin-bottom:10px">
      <div class="metric"><div class="label">Repeat</div><div class="value">${repeat.repeat || '-'}</div><div class="sub">${repeat.status || '-'}</div></div>
      <div class="metric"><div class="label">TTFT p50</div><div class="value">${fmt(repeat.ttft_ms_p50)}</div><div class="sub">ms</div></div>
      <div class="metric"><div class="label">ITL p50</div><div class="value">${fmt(repeat.itl_ms_p50)}</div><div class="sub">ms</div></div>
    </div>
    <div class="trace-box">
      <div class="links">${fileLink(repeat.artifact, 'Repeat JSON')} ${traceLink(repeat.trace)} ${repeat.trace ? '<button class="button" type="button" id="openPerfetto">Load in Perfetto</button>' : ''} <a class="button" href="https://ui.perfetto.dev" target="_blank" rel="noreferrer">Open empty Perfetto UI</a></div>
      <p>${traceInstruction}</p>
      <p class="muted small">The Load in Perfetto button opens ui.perfetto.dev and sends the selected trace with Perfetto's postMessage integration. Browser popup settings can block it; the download/open-file path is the fallback.</p>
      <p class="muted small">profile_path: <code>${repeat.profile_path || '-'}</code></p>
    </div>
  `;
  const button = document.getElementById('openPerfetto');
  if (button) button.addEventListener('click', () => openPerfettoWithTrace(repeat.trace));
}

function renderProfileVisuals() {
  const row = selectedRow();
  const repeat = selectedRepeat();
  barRows(document.getElementById('deltaBars'), row.profile_deltas || [], 'delta_ms', {
    limit: 10,
    label: r => r.bucket,
    value: (r, v) => `${fmt(v)} ms`,
    classFor: (r, v) => v < 0 ? 'neg' : 'pos'
  });
  barRows(document.getElementById('rangeBars'), row.scoped_ranges || [], 'total_ms', {
    limit: 12,
    label: r => `${r.scope}: ${r.bucket}`,
    value: (r, v) => `${fmt(v)} ms`,
    classFor: r => r.scope === 'gpu' ? 'gpu' : 'cpu'
  });
  const events = [
    ...(repeat.top_gpu_events || []).slice(0, 8).map(r => ({...r, scope:'gpu'})),
    ...(repeat.top_cpu_events || []).slice(0, 8).map(r => ({...r, scope:'cpu'}))
  ].sort((a, b) => Number(b.total_ms || 0) - Number(a.total_ms || 0)).slice(0, 14);
  barRows(document.getElementById('eventBars'), events, 'total_ms', {
    label: r => `${r.scope}: ${r.name}`,
    value: (r, v) => `${fmt(v)} ms`,
    classFor: r => r.scope
  });
}

async function openPerfettoWithTrace(tracePath) {
  const button = document.getElementById('openPerfetto');
  if (button) button.textContent = 'Fetching trace...';
  const perfettoWindow = window.open('https://ui.perfetto.dev', '_blank');
  if (!perfettoWindow) {
    if (button) button.textContent = 'Popup blocked';
    return;
  }
  try {
    const response = await fetch(`/file?path=${encodeURIComponent(tracePath)}`);
    const buffer = await response.arrayBuffer();
    const title = `${selectedRun().title} repeat ${selectedRepeat().repeat || ''}`;
    if (button) button.textContent = 'Waiting for Perfetto...';
    let sent = false;
    const sendTrace = () => {
      if (sent) return;
      sent = true;
      perfettoWindow.postMessage({
        perfetto: {
          buffer,
          title,
          fileName: tracePath.split('/').pop() || 'trace.json.gz'
        }
      }, 'https://ui.perfetto.dev');
      if (button) button.textContent = 'Loaded in Perfetto';
    };
    const listener = (event) => {
      if (event.origin !== 'https://ui.perfetto.dev') return;
      if (event.data === 'PONG' || (event.data && event.data.perfetto === 'PONG')) {
        window.removeEventListener('message', listener);
        clearInterval(ping);
        sendTrace();
      }
    };
    window.addEventListener('message', listener);
    const ping = setInterval(() => perfettoWindow.postMessage('PING', 'https://ui.perfetto.dev'), 250);
    setTimeout(() => {
      if (!sent) {
        window.removeEventListener('message', listener);
        clearInterval(ping);
        sendTrace();
      }
    }, 5000);
  } catch (error) {
    if (button) button.textContent = 'Perfetto load failed';
    console.error(error);
  }
}

function renderTraceTimeline() {
  const trace = selectedRepeat().trace;
  const container = document.getElementById('traceTimeline');
  const requestId = ++activeTraceRequest;
  if (!trace) {
    container.innerHTML = '<p class="muted">No trace path recorded for this repeat.</p>';
    return;
  }
  container.innerHTML = '<p class="muted">Parsing local trace timeline...</p>';
  fetch(`/trace_summary?path=${encodeURIComponent(trace)}`).then(r => r.json()).then(summary => {
    if (requestId !== activeTraceRequest) return;
    if (summary.error) {
      container.innerHTML = `<p class="muted">${summary.error}</p>`;
      return;
    }
    const events = summary.events || [];
    if (!events.length) {
      container.innerHTML = '<p class="muted">No duration events found in this trace.</p>';
      return;
    }
    const width = 1100, left = 110, right = 20, laneH = 20, top = 34;
    const lanes = [
      {scope:'gpu', label:'GPU', color:'#7c3aed'},
      {scope:'cpu', label:'CPU', color:'#0f766e'},
      {scope:'other', label:'Other', color:'#647184'}
    ];
    const height = top + lanes.length * laneH + 44;
    const x = pct => left + pct / 100 * (width - left - right);
    const rects = events.map(event => {
      const laneIndex = Math.max(0, lanes.findIndex(l => l.scope === event.scope));
      const y = top + laneIndex * laneH + 3;
      const rectX = x(event.start_pct);
      const rectW = Math.max(1, event.width_pct / 100 * (width - left - right));
      const lane = lanes[laneIndex];
      return `<rect x="${rectX}" y="${y}" width="${rectW}" height="13" rx="2" fill="${lane.color}" opacity=".76"><title>${event.scope}: ${event.name}\nstart ${fmt(event.start_ms)} ms, dur ${fmt(event.duration_ms)} ms</title></rect>`;
    }).join('');
    const laneLabels = lanes.map((lane, i) => `<text x="12" y="${top + i * laneH + 14}" font-size="12" fill="#344054">${lane.label}</text><line x1="${left}" x2="${width - right}" y1="${top + i * laneH + 10}" y2="${top + i * laneH + 10}" stroke="#eef2f6"></line>`).join('');
    const totals = summary.totals || {};
    container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Compact trace timeline">
      <rect width="${width}" height="${height}" fill="#fff"></rect>
      <text x="12" y="18" font-size="12" fill="#344054">${summary.file_name || 'trace'} · ${fmt(summary.duration_ms)} ms span · ${summary.event_count || 0} duration events</text>
      ${laneLabels}
      ${rects}
      <line x1="${left}" x2="${width - right}" y1="${height - 25}" y2="${height - 25}" stroke="#98a2b3"></line>
      <text x="${left}" y="${height - 8}" font-size="11" fill="#667085">0 ms</text>
      <text x="${width - right}" y="${height - 8}" text-anchor="end" font-size="11" fill="#667085">${fmt(summary.duration_ms)} ms</text>
    </svg>
    <div class="timeline-help">Showing the longest duration slices per scope, positioned on the trace time axis. GPU total ${fmt(totals.gpu && totals.gpu.total_ms)} ms across ${fmt(totals.gpu && totals.gpu.count, 0)} events; CPU total ${fmt(totals.cpu && totals.cpu.total_ms)} ms across ${fmt(totals.cpu && totals.cpu.count, 0)} events.</div>`;
  }).catch(error => {
    if (requestId !== activeTraceRequest) return;
    container.innerHTML = `<p class="muted">Could not parse trace summary: ${error}</p>`;
  });
}

function renderAll() {
  renderRunCards();
  renderMetrics();
  renderThroughputChart();
  renderLeaderboard();
  renderArtifacts();
  renderRepeatDetail();
  renderProfileVisuals();
  renderTraceTimeline();
}

fetch('/data').then(r => r.json()).then(data => {
  DATA = data;
  renderSelectors();
  document.getElementById('runSelect').addEventListener('change', renderRowSelector);
  document.getElementById('rowSelect').addEventListener('change', renderRepeatSelector);
  document.getElementById('repeatSelect').addEventListener('change', renderAll);
  document.getElementById('filter').addEventListener('input', renderLeaderboard);
});
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ProfileDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(HTTPStatus.OK, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/data":
            body = json.dumps(build_data(), indent=2).encode("utf-8")
            self._send_bytes(HTTPStatus.OK, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/trace_summary":
            query = parse_qs(parsed.query)
            path = _safe_file((query.get("path") or [""])[0])
            if path is None or not path.exists() or not path.is_file():
                self._send_bytes(HTTPStatus.NOT_FOUND, b'{"error":"trace file not found"}\n', "application/json; charset=utf-8")
                return
            try:
                body = json.dumps(summarize_trace_timeline(path), indent=2).encode("utf-8")
            except Exception as exc:
                body = json.dumps({"error": str(exc), "trace": str(path)}, indent=2).encode("utf-8")
            self._send_bytes(HTTPStatus.OK, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/file":
            query = parse_qs(parsed.query)
            path = _safe_file((query.get("path") or [""])[0])
            if path is None or not path.exists() or not path.is_file():
                self._send_bytes(HTTPStatus.NOT_FOUND, b"file not found\n", "text/plain; charset=utf-8")
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{quote(path.name)}")
            self.send_header("Access-Control-Allow-Origin", "https://ui.perfetto.dev")
            self.end_headers()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return
        self._send_bytes(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6789)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving profile dashboard at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
