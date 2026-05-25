#!/usr/bin/env python3
"""Serving-workload benchmark suite wrapper for ``benchmark_mtp1_engine.py``.

This script keeps correctness checks in the existing engine benchmark and adds
fixed, rerunnable workload shapes that better resemble serving traffic. It
emits a combined JSON report and an optional Markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    batch_prompts: int
    prompt_lengths: tuple[int, ...]
    max_tokens: int
    max_num_seqs: int
    batch_size_buckets: tuple[int, ...]
    prefill_buckets: tuple[int, ...]
    num_kvcache_blocks: int
    output_lengths: tuple[int, ...] = ()
    arrival_steps: tuple[int, ...] = ()
    max_blocks_per_seq: int | None = None
    prompt_suite: str = "real"


@dataclass(frozen=True)
class Mode:
    name: str
    description: str
    num_speculative_tokens: int
    compile_mtp_draft: bool
    env: dict[str, str] = field(default_factory=dict)
    unsafe_one_pass: bool = False


WORKLOADS: dict[str, Workload] = {
    "decode_steady_b1": Workload(
        name="decode_steady_b1",
        description="Single active row with warmup; emphasizes steady decode after one normal prefill.",
        batch_prompts=1,
        prompt_lengths=(64,),
        max_tokens=96,
        max_num_seqs=1,
        batch_size_buckets=(1,),
        prefill_buckets=(128,),
        num_kvcache_blocks=128,
        max_blocks_per_seq=24,
    ),
    "long_output_b1": Workload(
        name="long_output_b1",
        description="Single active row with longer prompt and output to expose long decode behavior.",
        batch_prompts=1,
        prompt_lengths=(256,),
        max_tokens=256,
        max_num_seqs=1,
        batch_size_buckets=(1,),
        prefill_buckets=(256,),
        num_kvcache_blocks=384,
        max_blocks_per_seq=48,
    ),
    "heterogeneous_b4": Workload(
        name="heterogeneous_b4",
        description="Four active rows with short, medium, and long prompt lengths in one batch.",
        batch_prompts=4,
        prompt_lengths=(16, 64, 160, 320),
        max_tokens=96,
        max_num_seqs=4,
        batch_size_buckets=(4,),
        prefill_buckets=(64, 256, 512),
        num_kvcache_blocks=768,
        max_blocks_per_seq=64,
        prompt_suite="expanded",
    ),
    "interleaved_prefill_decode_b4": Workload(
        name="interleaved_prefill_decode_b4",
        description=(
            "Four rows with staggered arrivals, small prefill buckets, and mixed long/short prompts so "
            "later prefills are scheduled while earlier rows are decoding."
        ),
        batch_prompts=4,
        prompt_lengths=(32, 448, 48, 512),
        max_tokens=64,
        output_lengths=(48, 16, 64, 24),
        arrival_steps=(0, 0, 3, 6),
        max_num_seqs=4,
        batch_size_buckets=(4,),
        prefill_buckets=(64, 128),
        num_kvcache_blocks=1024,
        max_blocks_per_seq=80,
        prompt_suite="expanded",
    ),
    "mixed_active_inactive_b4": Workload(
        name="mixed_active_inactive_b4",
        description="Three active rows in fixed physical B4 buckets to cover inactive row handling.",
        batch_prompts=3,
        prompt_lengths=(32, 96, 192),
        max_tokens=96,
        max_num_seqs=4,
        batch_size_buckets=(4,),
        prefill_buckets=(128, 256),
        num_kvcache_blocks=512,
        max_blocks_per_seq=48,
        prompt_suite="expanded",
    ),
    "heterogeneous_lengths_b16_partial": Workload(
        name="heterogeneous_lengths_b16_partial",
        description=(
            "Nine active rows in a physical B16 bucket with boundary prompt lengths "
            "1,15,16,17,31,32,127,128,129 and heterogeneous output lengths."
        ),
        batch_prompts=9,
        prompt_lengths=(1, 15, 16, 17, 31, 32, 127, 128, 129),
        max_tokens=33,
        output_lengths=(1, 2, 4, 8, 16, 32, 7, 15, 31),
        arrival_steps=(0, 0, 0, 2, 2, 4, 4, 6, 8),
        max_num_seqs=16,
        batch_size_buckets=(1, 4, 8, 16),
        prefill_buckets=(16, 32, 128, 256),
        num_kvcache_blocks=1024,
        max_blocks_per_seq=32,
        prompt_suite="expanded",
    ),
}


COMMON_MTP_ENV = {
    "NANO_VLLM_JAX_MTP_FUSED_VERIFY": "1",
    "NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED": "1",
    "NANO_VLLM_JAX_MTP_PREFIX_SAFE": "1",
    "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY": "rowwise",
}


MODES: dict[str, Mode] = {
    "baseline": Mode(
        name="baseline",
        description="Non-speculative target baseline; included to preserve a no-MTP timed reference.",
        num_speculative_tokens=0,
        compile_mtp_draft=False,
        env={},
    ),
    "unsafe_one_pass_no_seed": Mode(
        name="unsafe_one_pass_no_seed",
        description=(
            "Unsafe one-pass K=1 verifier without continuous seeded reuse. "
            "Speed fields are invalid unless exact-token and next-step-logit sanity pass."
        ),
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ENABLE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE": "1",
            "NANO_VLLM_JAX_MTP_DISABLE_PREFILL_SEED": "1",
            "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS": "0",
            "NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1": "0",
        },
        unsafe_one_pass=True,
    ),
    "unsafe_one_pass_seeded_cap2": Mode(
        name="unsafe_one_pass_seeded_cap2",
        description=(
            "Unsafe seeded one-pass K=1 verifier with seeded-chain cap 2. "
            "Speed fields are invalid unless exact-token and next-step-logit sanity pass."
        ),
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ENABLE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE": "1",
            "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS": "1",
            "NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN": "2",
        },
        unsafe_one_pass=True,
    ),
    "unsafe_one_pass_seeded_cap4": Mode(
        name="unsafe_one_pass_seeded_cap4",
        description=(
            "Unsafe seeded one-pass K=1 verifier with seeded-chain cap 4. "
            "Speed fields are invalid unless exact-token and next-step-logit sanity pass."
        ),
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ENABLE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE": "1",
            "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS": "1",
            "NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN": "4",
        },
        unsafe_one_pass=True,
    ),
    "always_on_mtp": Mode(
        name="always_on_mtp",
        description="Safe K=1 commit-select MTP with scheduler adaptive admission disabled, so MTP is always admitted.",
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_COMMIT_SELECT": "1",
            "NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE": "0",
            "NANO_VLLM_JAX_MTP_MIN_SPEEDUP": "0",
        },
    ),
    "adaptive_mtp": Mode(
        name="adaptive_mtp",
        description=(
            "Safe K=1 commit-select MTP with scheduler adaptive admission enabled from measured "
            "acceptance and EWMA latency stats."
        ),
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_COMMIT_SELECT": "1",
            "NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE": "0.01",
            "NANO_VLLM_JAX_MTP_MIN_ACCEPT_SAMPLES": "8",
            "NANO_VLLM_JAX_MTP_MIN_SPEEDUP": "1.0",
            "NANO_VLLM_JAX_MTP_LATENCY_MIN_STEPS": "2",
            "NANO_VLLM_JAX_MTP_LATENCY_ALPHA": "0.2",
        },
    ),
    "commit_select": Mode(
        name="commit_select",
        description="Safe sequential commit-select K=1 verifier reference; no unsafe one-pass env.",
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_COMMIT_SELECT": "1",
            "NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1": "1",
        },
    ),
    "compact_commit_select": Mode(
        name="compact_commit_select",
        description=(
            "Env-gated compact commit-select K=1 comparison path. Treat as separate from "
            "the safe full-physical-bucket commit_select reference."
        ),
        num_speculative_tokens=1,
        compile_mtp_draft=True,
        env={
            **COMMON_MTP_ENV,
            "NANO_VLLM_JAX_MTP_COMMIT_SELECT": "1",
            "NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1": "1",
            "NANO_VLLM_JAX_MTP_ALLOW_PARTIAL_COMMIT_SELECT": "1",
            "NANO_VLLM_JAX_MTP_ENABLE_COMPACT_COMMIT_SELECT": "1",
        },
    ),
}


DEFAULT_WORKLOADS = (
    "decode_steady_b1",
    "heterogeneous_b4",
    "heterogeneous_lengths_b16_partial",
    "long_output_b1",
    "interleaved_prefill_decode_b4",
)


DEFAULT_MODES = (
    "baseline",
    "always_on_mtp",
    "adaptive_mtp",
)


METRIC_KEYS = [
    "prefill_tok_s",
    "decode_tok_s",
    "end_to_end_tok_s",
    "decode_speedup",
    "end_to_end_speedup",
    "acceptance_rate",
    "fallback_count",
    "accepted_step_count",
    "rejected_step_count",
    "fallback_step_count",
    "scheduler_admission_reason",
    "scheduler_baseline_ewma_ms_per_token",
    "scheduler_spec_ewma_ms_per_token",
    "measured_scheduler_speedup",
    "accepted_itl_p50_ms",
    "accepted_itl_p95_ms",
    "rejected_itl_p50_ms",
    "rejected_itl_p95_ms",
    "fallback_itl_p50_ms",
    "fallback_itl_p95_ms",
    "host_time_ms",
    "runner_device_time_ms",
    "postprocess_time_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-script", default="benchmark_mtp1_engine.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--config-preset", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--weight-dtype", choices=["float16", "bfloat16", "float32", "auto"], default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--jax-execution", default="jit")
    parser.add_argument("--max-kv-cache-mb", type=int, default=2048)
    parser.add_argument("--workload", action="append", choices=sorted(WORKLOADS), default=None)
    parser.add_argument("--mode", action="append", choices=sorted(MODES), default=None)
    parser.add_argument("--max-tokens-override", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-json", default="serving_workloads_report.json")
    parser.add_argument("--output-md", default="serving_workloads_report.md")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-tpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-hf-logits", action="store_true")
    parser.add_argument("--check-next-step-sanity", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")
    parser.add_argument("--hf-device", default="cpu")
    parser.add_argument("--hf-max-prompts", type=int, default=1)
    parser.add_argument("--hf-logits-cache", default="")
    parser.add_argument("--hf-logits-cache-mode", choices=["auto", "read", "refresh"], default="auto")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--step-profile", action="store_true")
    parser.add_argument("--show-outputs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build commands and reports without invoking benchmark_mtp1_engine.py.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one bounded decode_steady_b1/no_seed_one_pass case unless workload/mode are supplied.",
    )
    return parser.parse_args()


def csv_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def nested_get(obj: Any, path: tuple[Any, ...], default: Any = None) -> Any:
    current = obj
    for part in path:
        if isinstance(current, dict):
            current = current.get(part, default)
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            return default
    return current


def first_number(obj: dict[str, Any], paths: list[tuple[Any, ...]]) -> float | int | None:
    for path in paths:
        value = nested_get(obj, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def first_present(obj: dict[str, Any], paths: list[tuple[Any, ...]]) -> Any:
    for path in paths:
        value = nested_get(obj, path)
        if value is not None:
            return value
    return None


def primary_result(data: dict[str, Any]) -> dict[str, Any]:
    variants = data.get("variants")
    if isinstance(variants, list) and variants:
        for variant in variants:
            if isinstance(variant, dict) and "timed_results" in variant:
                return variant
        if isinstance(variants[0], dict):
            return variants[0]
    rows = data.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_variants = row.get("variants")
            if not isinstance(row_variants, list) or not row_variants:
                continue
            for variant in row_variants:
                if isinstance(variant, dict) and "timed_results" in variant:
                    return variant
            if isinstance(row_variants[0], dict):
                return row_variants[0]
    required = data.get("required_metrics")
    if isinstance(required, dict) and required:
        return required
    return data


def selected_timing(result: dict[str, Any]) -> dict[str, Any]:
    timing = result.get("timed_results")
    raw = result.get("raw_timed_results")
    if isinstance(timing, dict) and timing.get("valid") is False and isinstance(raw, dict) and raw:
        return raw
    if isinstance(timing, dict) and timing:
        return timing
    if isinstance(raw, dict) and raw:
        return raw
    return result


def normalize_metrics(data: dict[str, Any]) -> dict[str, Any]:
    result = primary_result(data)
    timing = selected_timing(result)
    merged = {**data, **result, "timed_results": timing}
    valid = bool(
        first_present(
            merged,
            [
                ("timed_results_valid",),
                ("throughput_valid",),
                ("timed_results", "valid"),
            ],
        )
    )
    first_diff = first_present(merged, [("first_diff",), ("timed_results", "first_diff")])
    correct = first_present(merged, [("correct",), ("all_correct",), ("exact_token_match",)])
    hf_logits = first_present(merged, [("hf_logits_check",), ("hf_logit_check",), ("hf_logits",)])
    hf_ok = True
    if isinstance(hf_logits, dict):
        hf_ok = bool(hf_logits.get("ok", hf_logits.get("passed", True)))
    prefill_tok_s = first_number(
        merged,
        [
            ("timed_results", "prefill_tok_s"),
            ("mtp_prefill_tps",),
            ("prefill_tokens_per_second",),
            ("timed_results", "prefill_tokens_per_second"),
        ],
    )
    decode_tok_s = first_number(
        merged,
        [
            ("timed_results", "decode_tok_s"),
            ("mtp_decode_tps",),
            ("decode_tokens_per_second",),
            ("timed_results", "decode_tokens_per_second"),
        ],
    )
    end_to_end_tok_s = first_number(
        merged,
        [
            ("timed_results", "end_to_end_tok_s"),
            ("end_to_end_tok_s",),
        ],
    )
    if end_to_end_tok_s is None:
        total_tokens = first_number(merged, [("decode_tokens",), ("timed_results", "decode_tokens")])
        seconds = first_number(merged, [("seconds",), ("timed_results", "seconds")])
        if total_tokens is not None and seconds:
            end_to_end_tok_s = float(total_tokens) / max(1e-9, float(seconds))

    metrics = {
        "valid": valid,
        "correct": correct,
        "exact_token_match": correct,
        "next_step_sanity_ok": first_present(
            merged,
            [
                ("next_step_sanity_ok",),
                ("next_step_logit_sanity", "ok"),
                ("next_step_logits_check", "ok"),
            ],
        ),
        "hf_logits_ok": hf_ok,
        "first_diff": first_diff,
        "invalid_reason": first_present(
            merged,
            [
                ("timed_results_invalid_reason",),
                ("timed_results", "invalid_reason"),
            ],
        ),
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": end_to_end_tok_s,
        "decode_speedup": first_number(merged, [("timed_results", "decode_speedup"), ("decode_speedup",)]),
        "end_to_end_speedup": first_number(
            merged,
            [
                ("timed_results", "end_to_end_speedup"),
                ("end_to_end_speedup",),
                ("speedup",),
            ],
        ),
        "adaptive_predicted_speedup_diagnostic": first_number(
            merged,
            [
                ("adaptive_mtp_gating", "predicted_speedup"),
                ("timed_results", "adaptive_mtp_gating", "predicted_speedup"),
                ("raw_timed_results", "adaptive_mtp_gating", "predicted_speedup"),
            ],
        ),
        "adaptive_predicted_speedup_authoritative": False,
        "serving_enable_criterion": "measured decode_speedup",
        "acceptance_rate": first_number(
            merged,
            [
                ("timed_results", "acceptance_rate"),
                ("acceptance_rate",),
            ],
        ),
        "drafts_proposed": first_number(
            merged,
            [
                ("speculative", "drafts_proposed"),
                ("speculative_counts", "drafts_proposed"),
                ("timed_results", "drafts_proposed"),
            ],
        ),
        "drafts_accepted": first_number(
            merged,
            [
                ("speculative", "drafts_accepted"),
                ("speculative_counts", "drafts_accepted"),
                ("timed_results", "drafts_accepted"),
            ],
        ),
        "drafts_rejected": first_number(
            merged,
            [
                ("speculative", "drafts_rejected"),
                ("speculative_counts", "drafts_rejected"),
                ("timed_results", "drafts_rejected"),
            ],
        ),
        "fallback_count": first_number(
            merged,
            [
                ("timed_results", "fallback_count"),
                ("fallback_count",),
                ("speculative_counts", "fallback_decode_steps"),
            ],
        ),
        "accepted_step_count": first_number(
            merged,
            [
                ("timed_results", "accepted_step_count"),
                ("accepted_step_count",),
                ("speculative_counts", "accepted_decode_steps"),
                ("step_mode_counts", "accepted_decode_steps"),
            ],
        ),
        "rejected_step_count": first_number(
            merged,
            [
                ("timed_results", "rejected_step_count"),
                ("rejected_step_count",),
                ("speculative_counts", "rejected_decode_steps"),
                ("step_mode_counts", "rejected_decode_steps"),
            ],
        ),
        "fallback_step_count": first_number(
            merged,
            [
                ("timed_results", "fallback_step_count"),
                ("fallback_step_count",),
                ("speculative_counts", "fallback_decode_steps"),
                ("step_mode_counts", "fallback_decode_steps"),
            ],
        ),
        "scheduler_admission_reason": first_present(
            merged,
            [
                ("timed_results", "scheduler_admission_reason"),
                ("scheduler_mtp_admission", "reason"),
            ],
        ),
        "scheduler_baseline_ewma_ms_per_token": first_number(
            merged,
            [
                ("timed_results", "scheduler_baseline_ewma_ms_per_token"),
                ("scheduler_mtp_admission", "baseline_ewma_ms_per_token"),
            ],
        ),
        "scheduler_spec_ewma_ms_per_token": first_number(
            merged,
            [
                ("timed_results", "scheduler_spec_ewma_ms_per_token"),
                ("scheduler_mtp_admission", "spec_ewma_ms_per_token"),
            ],
        ),
        "measured_scheduler_speedup": first_number(
            merged,
            [
                ("timed_results", "measured_scheduler_speedup"),
                ("scheduler_mtp_admission", "measured_scheduler_speedup"),
            ],
        ),
        "accepted_itl_p50_ms": first_number(
            merged,
            [
                ("inter_token_latency_ms", "accepted", "p50"),
                ("accepted_inter_token_latency_ms", "p50"),
            ],
        ),
        "accepted_itl_p95_ms": first_number(
            merged,
            [
                ("inter_token_latency_ms", "accepted", "p95"),
                ("accepted_inter_token_latency_ms", "p95"),
            ],
        ),
        "rejected_itl_p50_ms": first_number(
            merged,
            [
                ("inter_token_latency_ms", "rejected", "p50"),
                ("rejected_inter_token_latency_ms", "p50"),
            ],
        ),
        "rejected_itl_p95_ms": first_number(
            merged,
            [
                ("inter_token_latency_ms", "rejected", "p95"),
                ("rejected_inter_token_latency_ms", "p95"),
            ],
        ),
        "fallback_itl_p50_ms": first_number(
            merged,
            [
                ("inter_token_latency_ms", "fallback", "p50"),
                ("fallback_inter_token_latency_ms", "p50"),
            ],
        ),
        "fallback_itl_p95_ms": first_number(
            merged,
            [
                ("inter_token_latency_ms", "fallback", "p95"),
                ("fallback_inter_token_latency_ms", "p95"),
            ],
        ),
        "host_time_ms": first_number(merged, [("timed_results", "host_time_ms"), ("host_time_ms",)]),
        "runner_device_time_ms": first_number(
            merged,
            [
                ("timed_results", "runner_device_time_ms"),
                ("runner_device_time_ms",),
            ],
        ),
        "postprocess_time_ms": first_number(
            merged,
            [
                ("timed_results", "postprocess_time_ms"),
                ("postprocess_time_ms",),
            ],
        ),
    }
    predicted = metrics.get("adaptive_predicted_speedup_diagnostic")
    measured = metrics.get("decode_speedup")
    if isinstance(predicted, (int, float)) and isinstance(measured, (int, float)):
        metrics["adaptive_predicted_vs_measured_delta"] = float(predicted) - float(measured)
        metrics["adaptive_predicted_disagrees_with_measured"] = (
            (float(predicted) >= 1.0) != (float(measured) >= 1.0)
        )
        if metrics["adaptive_predicted_disagrees_with_measured"]:
            metrics["adaptive_predicted_warning"] = (
                "diagnostic predicted_speedup disagrees with measured decode_speedup; "
                "use measured decode_speedup for serving enablement"
            )
    return metrics


def hf_logits_status(data: dict[str, Any]) -> tuple[bool, bool]:
    result = primary_result(data)
    merged = {**data, **result}
    hf_logits = first_present(merged, [("hf_logits_check",), ("hf_logit_check",), ("hf_logits",)])
    if not isinstance(hf_logits, dict):
        return False, False
    checked = bool(hf_logits.get("checked", hf_logits.get("enabled", False)))
    ok = bool(hf_logits.get("ok", hf_logits.get("passed", False)))
    return checked, ok


def format_value(value: Any, *, na_label: str = "") -> str:
    if value is None:
        return na_label
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_command(
    *,
    args: argparse.Namespace,
    workload: Workload,
    mode: Mode,
    run_json: Path,
) -> list[str]:
    max_tokens = args.max_tokens_override if args.max_tokens_override is not None else workload.max_tokens
    command = [
        args.python,
        args.engine_script,
        "--model",
        args.model,
        "--config-preset",
        args.config_preset,
        "--dtype",
        args.dtype,
        "--weight-dtype",
        args.weight_dtype,
        "--backend",
        args.backend,
        "--jax-execution",
        args.jax_execution,
        "--max-tokens",
        str(max_tokens),
        "--num-speculative-tokens",
        str(mode.num_speculative_tokens),
        "--max-num-seqs",
        str(workload.max_num_seqs),
        "--batch-size-buckets",
        csv_ints(workload.batch_size_buckets),
        "--batch-prompts",
        str(workload.batch_prompts),
        "--prompt-lengths",
        csv_ints(workload.prompt_lengths),
        "--prompt-suite",
        workload.prompt_suite,
        "--output-lengths",
        csv_ints(workload.output_lengths),
        "--arrival-steps",
        csv_ints(workload.arrival_steps),
        "--prefill-buckets",
        csv_ints(workload.prefill_buckets),
        "--num-kvcache-blocks",
        str(workload.num_kvcache_blocks),
        "--max-kv-cache-mb",
        str(args.max_kv_cache_mb),
        "--repeats",
        str(args.repeats),
        "--output-json",
        str(run_json),
    ]
    if workload.max_blocks_per_seq is not None:
        command.extend(["--max-blocks-per-seq", str(workload.max_blocks_per_seq)])
    if args.warmup:
        command.append("--warmup")
    if args.require_tpu:
        command.append("--require-tpu")
    if args.check_hf_logits:
        command.extend(
            [
                "--check-hf-logits",
                "--hf-device",
                args.hf_device,
                "--hf-max-prompts",
                str(args.hf_max_prompts),
            ]
        )
        if args.hf_logits_cache:
            command.extend(["--hf-logits-cache", args.hf_logits_cache])
        if args.hf_logits_cache_mode != "auto":
            command.extend(["--hf-logits-cache-mode", args.hf_logits_cache_mode])
    if args.check_next_step_sanity:
        command.append("--check-next-step-sanity")
    if args.hf_offline:
        command.append("--hf-offline")
    if args.correctness_only:
        command.append("--correctness-only")
    if args.step_profile:
        command.append("--step-profile")
    if args.show_outputs:
        command.append("--show-outputs")
    if mode.compile_mtp_draft:
        command.append("--compile-mtp-draft")
    return command


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    na_label = "n/a (smoke)" if report.get("smoke") or report.get("correctness_only") else ""

    def show(value: Any) -> str:
        return format_value(value, na_label=na_label)

    lines = [
        "# Serving workload benchmark report",
        "",
        f"Generated at: `{report['generated_at_unix']}`",
        f"Model: `{report['model']}`",
        "",
        "## Workloads",
        "",
    ]
    for workload in report["workloads"]:
        lines.append(f"- `{workload['name']}`: {workload['description']}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| workload | mode | TPU | valid | exact | next-step | HF | prefill tok/s | decode tok/s | e2e tok/s | decode speedup | measured enable | scheduler reason | scheduler speedup | scheduler EWMA base/spec ms | predicted diag | predicted disagrees | e2e speedup | acceptance | drafts p/a/r | step a/r/f | fallback | acc p50/p95 ms | rej p50/p95 ms | fb p50/p95 ms | host ms | device ms | post ms | first_diff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in report["runs"]:
        metrics = run.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    run["workload"],
                    run["mode"],
                    format_value(metrics.get("tpu_backend_used")),
                    format_value(metrics.get("valid")),
                    format_value(metrics.get("exact_token_match")),
                    format_value(metrics.get("next_step_sanity_ok")),
                    format_value(metrics.get("hf_logits_ok")),
                    show(metrics.get("prefill_tok_s")),
                    show(metrics.get("decode_tok_s")),
                    show(metrics.get("end_to_end_tok_s")),
                    show(metrics.get("decode_speedup")),
                    format_value(metrics.get("serving_enable_by_measured_decode")),
                    format_value(metrics.get("scheduler_admission_reason")),
                    show(metrics.get("measured_scheduler_speedup")),
                    f"{show(metrics.get('scheduler_baseline_ewma_ms_per_token'))}/{show(metrics.get('scheduler_spec_ewma_ms_per_token'))}",
                    show(metrics.get("adaptive_predicted_speedup_diagnostic")),
                    format_value(metrics.get("adaptive_predicted_disagrees_with_measured")),
                    show(metrics.get("end_to_end_speedup")),
                    show(metrics.get("acceptance_rate")),
                    f"{show(metrics.get('drafts_proposed'))}/{show(metrics.get('drafts_accepted'))}/{show(metrics.get('drafts_rejected'))}",
                    f"{show(metrics.get('accepted_step_count'))}/{show(metrics.get('rejected_step_count'))}/{show(metrics.get('fallback_step_count'))}",
                    show(metrics.get("fallback_count")),
                    f"{show(metrics.get('accepted_itl_p50_ms'))}/{show(metrics.get('accepted_itl_p95_ms'))}",
                    f"{show(metrics.get('rejected_itl_p50_ms'))}/{show(metrics.get('rejected_itl_p95_ms'))}",
                    f"{show(metrics.get('fallback_itl_p50_ms'))}/{show(metrics.get('fallback_itl_p95_ms'))}",
                    show(metrics.get("host_time_ms")),
                    show(metrics.get("runner_device_time_ms")),
                    show(metrics.get("postprocess_time_ms")),
                    format_value(metrics.get("first_diff")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(args: argparse.Namespace, workload: Workload, mode: Mode, run_dir: Path) -> dict[str, Any]:
    run_id = f"{workload.name}__{mode.name}"
    run_json = run_dir / f"{run_id}.json"
    command = build_command(args=args, workload=workload, mode=mode, run_json=run_json)
    env = os.environ.copy()
    env.update(mode.env)
    if args.dry_run:
        metrics = {
            "valid": False,
            "correct": None,
            "exact_token_match": None,
            "next_step_sanity_ok": None,
            "hf_logits_ok": None,
            "first_diff": None,
            "invalid_reason": "dry-run only",
            "unsafe_one_pass": mode.unsafe_one_pass,
            "requires_next_step_logit_sanity": mode.unsafe_one_pass,
            "adaptive_predicted_speedup_diagnostic": None,
            "adaptive_predicted_speedup_authoritative": False,
            "adaptive_predicted_disagrees_with_measured": None,
            "serving_enable_criterion": "measured decode_speedup" if mode.name == "commit_select" else None,
            "serving_enable_by_measured_decode": None,
            "tpu_backend_used": bool(args.require_tpu),
        }
        return {
            "workload": workload.name,
            "mode": mode.name,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "command": command,
            "env_overrides": mode.env,
            "output_json": str(run_json),
            "dry_run": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "metrics": metrics,
        }
    started = time.time()
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    elapsed = time.time() - started
    data: dict[str, Any] = {}
    if run_json.exists():
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            data = {"json_error": str(exc)}
    metrics = normalize_metrics(data) if data else {}
    if mode.unsafe_one_pass:
        hf_checked, hf_ok = hf_logits_status(data)
        metrics["unsafe_one_pass"] = True
        metrics["requires_next_step_logit_sanity"] = True
        if not hf_checked or not hf_ok:
            metrics["valid"] = False
            metrics["hf_logits_ok"] = False
            metrics["invalid_reason"] = (
                "unsafe one-pass mode requires exact-token match and next-step-logit sanity; "
                "rerun with --check-hf-logits"
                if not hf_checked
                else "unsafe one-pass mode failed next-step-logit sanity"
            )
    else:
        metrics["unsafe_one_pass"] = False
    if mode.name in {"commit_select", "always_on_mtp", "adaptive_mtp"}:
        measured_decode_speedup = metrics.get("decode_speedup")
        metrics["serving_enable_by_measured_decode"] = (
            bool(metrics.get("valid"))
            and isinstance(measured_decode_speedup, (int, float))
            and float(measured_decode_speedup) >= 1.0
        )
    else:
        metrics["serving_enable_by_measured_decode"] = None
    metrics["tpu_backend_used"] = bool(args.require_tpu and completed.returncode == 0)
    record = {
        "workload": workload.name,
        "mode": mode.name,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "env_overrides": mode.env,
        "output_json": str(run_json),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "metrics": metrics,
    }
    if completed.returncode != 0 and not metrics.get("invalid_reason"):
        record["metrics"]["valid"] = False
        record["metrics"]["invalid_reason"] = f"subprocess exited {completed.returncode}"
    return record


def main() -> int:
    args = parse_args()
    workload_names = args.workload or list(DEFAULT_WORKLOADS)
    mode_names = args.mode or list(DEFAULT_MODES)
    if args.smoke:
        workload_names = args.workload or ["decode_steady_b1"]
        mode_names = args.mode or ["unsafe_one_pass_no_seed"]
        if args.max_tokens_override is None:
            args.max_tokens_override = 8
        if not args.check_hf_logits:
            args.correctness_only = True

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.output_json).with_suffix("").parent / "serving_workload_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_workloads = [WORKLOADS[name] for name in workload_names]
    selected_modes = [MODES[name] for name in mode_names]
    report = {
        "generated_at_unix": time.time(),
        "model": args.model,
        "config_preset": args.config_preset,
        "dtype": args.dtype,
        "weight_dtype": args.weight_dtype,
        "backend": args.backend,
        "jax_execution": args.jax_execution,
        "warmup": args.warmup,
        "require_tpu": args.require_tpu,
        "check_hf_logits": args.check_hf_logits,
        "check_next_step_sanity": args.check_next_step_sanity,
        "hf_logits_cache": args.hf_logits_cache,
        "hf_logits_cache_mode": args.hf_logits_cache_mode,
        "correctness_only": args.correctness_only,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "workloads": [
            {
                "name": workload.name,
                "description": workload.description,
                "batch_prompts": workload.batch_prompts,
                "prompt_lengths": workload.prompt_lengths,
                "max_tokens": args.max_tokens_override if args.max_tokens_override is not None else workload.max_tokens,
                "output_lengths": workload.output_lengths,
                "arrival_steps": workload.arrival_steps,
                "max_num_seqs": workload.max_num_seqs,
                "batch_size_buckets": workload.batch_size_buckets,
                "prefill_buckets": workload.prefill_buckets,
                "num_kvcache_blocks": workload.num_kvcache_blocks,
                "max_blocks_per_seq": workload.max_blocks_per_seq,
            }
            for workload in selected_workloads
        ],
        "modes": [
            {
                "name": mode.name,
                "description": mode.description,
                "num_speculative_tokens": mode.num_speculative_tokens,
                "unsafe_one_pass": mode.unsafe_one_pass,
                "validity_policy": (
                    "invalid unless exact-token and next-step-logit sanity pass"
                    if mode.unsafe_one_pass
                    else "valid when engine correctness gates pass"
                ),
                "env": mode.env,
            }
            for mode in selected_modes
        ],
        "runs": [],
    }

    for workload in selected_workloads:
        for mode in selected_modes:
            record = run_one(args, workload, mode, run_dir)
            report["runs"].append(record)
            if args.fail_fast and record["returncode"] != 0:
                Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if args.output_md:
                    write_markdown(Path(args.output_md), report)
                return record["returncode"]

    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    return 0 if all(run["returncode"] == 0 for run in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
