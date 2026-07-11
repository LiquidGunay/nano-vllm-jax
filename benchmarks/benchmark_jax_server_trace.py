#!/usr/bin/env python3
"""Server-path JAX benchmark with per-token timing traces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_vllm_qwen35 import compare_reference, prepare_prompt_rows
from benchmarks.summarize_profile_trace import summarize_trace
from run_tracking import RunRecorder
from runtime_paths import configure_compilation_cache, configure_flashinfer_cache, configure_xla_flags

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
configure_xla_flags()
configure_compilation_cache()
configure_flashinfer_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--weight-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="jit")
    parser.add_argument("--input-lens", default="16,32,64,128")
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--output-lengths", default="")
    parser.add_argument("--prompt-suite", choices=["synthetic", "real", "mixed", "server_shapes"], default="server_shapes")
    parser.add_argument(
        "--prompt-source",
        choices=["tokenized_seed_repeat", "manifest", "vllm_random"],
        default="tokenized_seed_repeat",
    )
    parser.add_argument("--prompt-manifest-jsonl", default="")
    parser.add_argument("--prompt-manifest-output-jsonl", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--num-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-input-len", type=int, default=1280)
    parser.add_argument("--random-output-len", type=int, default=16)
    parser.add_argument("--random-range-ratio", default='{"input":0.0,"output":0.0}')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-top-k",
        type=int,
        default=-1,
        help="JAX SamplingParams top_k. Keep -1 for the compiled temperature-sampling fast path.",
    )
    parser.add_argument("--speculative-method", choices=["none", "mtp"], default="none")
    parser.add_argument("--draft-sample-method", choices=["greedy", "probabilistic"], default="greedy")
    parser.add_argument(
        "--mtp-verifier-impl",
        choices=["two_decode", "commit_select", "k_decode", "packed_prefix"],
        default="packed_prefix",
    )
    parser.add_argument("--mtp-batch-accept-policy", choices=["rowwise", "all_or_none"], default="rowwise")
    parser.add_argument("--mtp-seed-after-bonus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mtp-bonus-margin", type=float, default=0.0)
    parser.add_argument("--mtp-draft-margin", type=float, default=0.0)
    parser.add_argument("--mtp-hidden-source", choices=["pre_norm", "final_normed"], default="pre_norm")
    parser.add_argument("--mtp-chain-hidden-source", choices=["raw", "final_normed"], default="raw")
    parser.add_argument("--mtp-chain-mode", choices=["recursive", "sequence"], default="recursive")
    parser.add_argument("--mtp-token-source", choices=["generated", "current"], default="generated")
    parser.add_argument("--mtp-position-offset", type=int, default=0)
    parser.add_argument("--mtp-lm-head-greedy-top1-impl", default="jax")
    parser.add_argument("--mtp-draft-vocab-size", type=int, default=0)
    parser.add_argument("--num-speculative-tokens", type=int, choices=list(range(0, 9)), default=0)
    parser.add_argument("--mtp-burst-groups", type=int, default=1)
    parser.add_argument("--mtp-max-active-rows", type=int, default=0)
    parser.add_argument("--mtp-prefill-seed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-kv-cache-mb", type=int, default=1024)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--max-num-resident-seqs",
        type=int,
        default=0,
        help="Resident request capacity; 0 keeps it equal to --max-num-seqs.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-buckets", default="16,32,64,128")
    parser.add_argument("--prefill-token-buckets", default="")
    parser.add_argument("--prefill-layout", choices=["packed", "dense"], default="packed")
    parser.add_argument("--batch-size-buckets", default="1,2,4")
    parser.add_argument("--max-blocks-per-seq", type=int, default=16)
    parser.add_argument("--decode-block-table-buckets", default="")
    parser.add_argument("--startup-warmup-prefill-token-buckets", default="")
    parser.add_argument("--startup-warmup-batch-size-buckets", default="")
    parser.add_argument("--startup-warmup-decode-block-table-buckets", default="")
    parser.add_argument(
        "--startup-warmup-include-sampled-routes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--greedy-token-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampled-token-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-decode-burst-steps", type=int, default=1)
    parser.add_argument("--device-token-carry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-decode-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-decode-seq-lens-carry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resident-decode-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-token-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-host-token-sink-min-completion-tokens", type=int, default=1024)
    parser.add_argument("--summary-host-token-sink-min-avg-completion-tokens", type=int, default=0)
    parser.add_argument("--materialize-tied-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-prefill-in-proj-qkv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-prefill-gdn-z", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-prefill-full-attn-proj", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-prefill-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-prefill-token-count-mode", default="exact")
    parser.add_argument("--lm-head-decode-act-dtype", default="fp32")
    parser.add_argument("--lm-head-topk-impl", default="jax")
    parser.add_argument("--lm-head-greedy-top1-impl", default="jax")
    parser.add_argument("--decode-proj-act-dtype", default="fp32")
    parser.add_argument("--decode-padded-gemm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-padded-gemm-gate-up", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-rms-padded-gemm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-padded-gemm-rows", type=int, default=8)
    parser.add_argument("--decode-padded-gemm-max-out-dim", type=int, default=300000)
    parser.add_argument(
        "--gdn-width1-packed-input-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--full-attention-kv-cache-dtype", default="default")
    parser.add_argument("--full-attention-kv-append-impl", default="reference")
    parser.add_argument("--full-attention-decode-impl", default="reference")
    parser.add_argument("--full-attention-prefill-impl", default="reference")
    parser.add_argument("--gdn-disable-fallbacks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gdn-prefill-post-conv-impl", default="off")
    parser.add_argument("--gdn-prefill-qkv-dtype", default="fp32")
    parser.add_argument("--gdn-prefill-post-conv-output-dtype", default="fp32")
    parser.add_argument("--gdn-packed-decode-impl", default="off")
    parser.add_argument("--gdn-packed-decode-qkv-dtype", default="fp32")
    parser.add_argument("--gdn-packed-decode-pre-normalize-qk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gdn-packed-decode-max-batch", type=int, default=0)
    parser.add_argument(
        "--linear-chunk-size",
        type=int,
        default=0,
        help="Override Qwen3.5 Gated DeltaNet prefill chunk size; 0 keeps the model default.",
    )
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument(
        "--warmup-mode",
        choices=["generic", "request"],
        default="generic",
        help="generic compiles configured server buckets; request replays the measured prompt list for diagnostics only.",
    )
    parser.add_argument(
        "--fail-on-jit-cache-growth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if new executor JIT keys are created during the measured generation phase.",
    )
    parser.add_argument(
        "--trace-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store detailed per-token events. The default summary mode keeps "
            "TTFT/ITL metrics without per-token Python event overhead."
        ),
    )
    parser.add_argument("--reference-json", default="")
    parser.add_argument("--output-json", default="results/qwen08_jax_server_trace.json")
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=os.environ.get("NANO_VLLM_JAX_PROFILE", "1") not in {"0", "false", "False", "no", "off"},
    )
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") in {"1", "true", "yes", "on", "True"}


def _gpu_memory_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    return max(values) if values else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _block_until_ready(value: Any) -> None:
    ready = getattr(value, "block_until_ready", None)
    if callable(ready):
        ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item)


def _jit_cache_key_snapshot(cache: Any) -> set[str] | None:
    if cache is None:
        return None
    return {repr(key) for key in cache.keys()}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _timing_metrics(events: list[dict[str, Any]], elapsed: float, total_tokens: int) -> dict[str, Any]:
    by_request: dict[int, list[float]] = {}
    token_elapsed_seconds: list[float] = []
    for event in events:
        if event.get("event") != "token":
            continue
        event_elapsed = float(event["elapsed_seconds"])
        token_elapsed_seconds.append(event_elapsed)
        by_request.setdefault(int(event["request_index"]), []).append(event_elapsed)
    ttfts = []
    itls = []
    for timestamps in by_request.values():
        if timestamps:
            ttfts.append(1000.0 * timestamps[0])
            itls.extend(1000.0 * (right - left) for left, right in zip(timestamps, timestamps[1:]))
    last_token_elapsed = max(token_elapsed_seconds) if token_elapsed_seconds else None
    post_last_token_drain_seconds = (
        max(0.0, elapsed - last_token_elapsed)
        if last_token_elapsed is not None
        else None
    )
    return {
        "seconds": elapsed,
        "last_token_elapsed_seconds": last_token_elapsed,
        "post_last_token_drain_seconds": post_last_token_drain_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "token_event_tokens_per_second": (
            total_tokens / max(last_token_elapsed, 1e-9)
            if last_token_elapsed is not None
            else None
        ),
        "token_event_scope": "host events; may exclude deferred device completion",
        "ttft_ms_mean": float(sum(ttfts) / len(ttfts)) if ttfts else None,
        "ttft_ms_p50": _percentile(ttfts, 50),
        "ttft_ms_p95": _percentile(ttfts, 95),
        "itl_ms_mean": float(sum(itls) / len(itls)) if itls else None,
        "itl_ms_p50": _percentile(itls, 50),
        "itl_ms_p95": _percentile(itls, 95),
        "itl_source": "jax_server_step_trace",
    }


def _timing_metrics_from_trace(trace: dict[str, Any], elapsed: float, total_tokens: int) -> dict[str, Any]:
    summary = trace.get("timing_summary")
    if not summary:
        return _timing_metrics(trace.get("events") or [], elapsed, total_tokens)

    last_token_elapsed = summary.get("last_token_elapsed_seconds")
    post_last_token_drain_seconds = (
        max(0.0, elapsed - float(last_token_elapsed))
        if last_token_elapsed is not None
        else None
    )
    return {
        "seconds": elapsed,
        "last_token_elapsed_seconds": last_token_elapsed,
        "post_last_token_drain_seconds": post_last_token_drain_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "token_event_tokens_per_second": (
            total_tokens / max(float(last_token_elapsed), 1e-9)
            if last_token_elapsed is not None
            else None
        ),
        "token_event_scope": "host events; may exclude deferred device completion",
        "ttft_ms_mean": summary.get("ttft_ms_mean"),
        "ttft_ms_p50": summary.get("ttft_ms_p50"),
        "ttft_ms_p95": summary.get("ttft_ms_p95"),
        "itl_ms_mean": summary.get("itl_ms_mean"),
        "itl_ms_p50": summary.get("itl_ms_p50"),
        "itl_ms_p95": summary.get("itl_ms_p95"),
        "itl_source": summary.get("source") or "jax_server_step_summary",
    }


def _performance_with_token_scopes(rows: list[dict[str, Any]], performance: dict[str, Any], elapsed: float) -> dict[str, Any]:
    total_input_tokens = sum(int(row["prompt_length"]) for row in rows)
    total_output_tokens = sum(int(row["generated_tokens"]) for row in rows)
    request_count = len(rows)
    performance.update(
        {
            "request_count": request_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput": request_count / max(elapsed, 1e-9),
            "output_token_throughput": total_output_tokens / max(elapsed, 1e-9),
            "token_event_output_token_throughput": performance.get("token_event_tokens_per_second"),
            "total_token_throughput": (total_input_tokens + total_output_tokens) / max(elapsed, 1e-9),
        }
    )
    return performance


def _profile_counters(profile_path: Path) -> dict[str, Any]:
    traces = sorted(profile_path.glob("plugins/profile/*/*.trace.json.gz"))
    if not traces:
        return {
            "trace_json_gz": None,
            "ranges": {},
            "top_events_by_total_ms": [],
            "scoped_ranges": {},
            "scoped_top_events_by_total_ms": {},
        }
    trace_path = traces[-1]
    needles = [
        "generate_with_trace",
        "_run_main_and_sample",
        "forward_step_token_ids_jit",
        "forward_step_jit",
        "PjRtCApiLoadedExecutable::Execute",
        "jit_compiled:XLA GPU module",
        "command_buffer::execute",
        "command_buffer::update",
        "input_reduce_fusion",
        "loop_dynamic_update_slice_fusion",
        "loop_multiply_fusion",
        "wrapped_concatenate",
        "MemcpyD2D",
        "Thunks::Initialize",
        "_batch_hybrid_state",
        "_store_batch_hybrid_state",
        "_record_kv_snapshot",
        "_refresh_kv_snapshot",
        "array.py:325 tolist",
        "np.asarray(jax.Array)",
        "gemm_fusion",
        "cutlass",
        "gather",
        "transpose",
        "fusion",
        "while",
    ]
    try:
        all_summary = summarize_trace(
            trace_path,
            scope="all",
            top_events=40,
            patterns=needles,
        )
        gpu_summary = summarize_trace(
            trace_path,
            scope="gpu",
            top_events=40,
            patterns=needles,
        )
        cpu_summary = summarize_trace(
            trace_path,
            scope="cpu",
            top_events=40,
            patterns=needles,
        )
    except Exception as exc:
        return {
            "trace_json_gz": str(trace_path),
            "error": f"{type(exc).__name__}: {exc}",
            "ranges": {},
            "top_events_by_total_ms": [],
            "scoped_ranges": {},
            "scoped_top_events_by_total_ms": {},
        }
    return {
        "trace_json_gz": str(trace_path),
        "ranges": all_summary["patterns"],
        "scoped_ranges": {
            "gpu": gpu_summary["patterns"],
            "cpu": cpu_summary["patterns"],
        },
        "top_events_by_total_ms": all_summary["top_events_by_total_ms"],
        "scoped_top_events_by_total_ms": {
            "gpu": gpu_summary["top_events_by_total_ms"],
            "cpu": cpu_summary["top_events_by_total_ms"],
        },
    }


def _build_sampling_params(
    output_lengths: list[int],
    default_output_len: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
):
    from nanovllm_jax.engine.sequence import SamplingParams

    if output_lengths:
        return [
            SamplingParams(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                max_tokens=int(length),
                ignore_eos=True,
            )
            for length in output_lengths
        ]
    return SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_tokens=int(default_output_len),
        ignore_eos=True,
    )


def _shape_env_value(shapes: list[tuple[int, ...]]) -> str:
    return ";".join(":".join(str(int(part)) for part in shape) for shape in shapes)


def _parse_shape_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace("x", ":").replace(",", ":").split(":") if part.strip())


def _mtp_static_warmup_row_count(
    rows: int,
    args: argparse.Namespace,
    config: Any,
) -> int:
    """Mirror the runner's fixed-row MTP verifier bucket selection."""

    rows = int(rows)
    max_active_rows = max(
        0,
        int(
            getattr(
                config,
                "mtp_max_active_rows",
                getattr(args, "mtp_max_active_rows", 0),
            )
            or 0
        ),
    )
    if max_active_rows <= 0 or rows > max_active_rows:
        return rows
    batch_buckets = tuple(
        sorted(int(bucket) for bucket in (getattr(config, "batch_size_buckets", ()) or ()))
    )
    if not batch_buckets:
        return max_active_rows
    for bucket in batch_buckets:
        if max_active_rows <= bucket:
            return bucket
    raise ValueError(
        "fixed-row MTP warmup target "
        f"{max_active_rows} exceeds configured batch buckets {batch_buckets}"
    )


def _manifest_mtp_table_warmup_specs(
    prompt_rows: list[dict[str, Any]],
    output_lengths: list[int],
    args: argparse.Namespace,
    config: Any,
) -> dict[str, Any]:
    """Record table-verifier shapes from an optimistic MTP dry schedule."""
    from nanovllm_jax.engine.scheduler import Scheduler
    from nanovllm_jax.engine.sequence import SamplingParams, Sequence

    draft_budget = max(0, int(getattr(args, "num_speculative_tokens", 0) or 0))
    burst_budget = max(1, int(getattr(args, "mtp_burst_groups", 1) or 1))
    if draft_budget <= 0:
        return {"enabled": False, "specs": [], "env": ""}
    fixed_width_packed_verifier = str(
        getattr(args, "mtp_verifier_impl", "none") or "none"
    ).lower() in {"packed_prefix", "packed_prefill", "prefill_packed"}

    scheduler = Scheduler(config)
    for index, (row, output_len) in enumerate(zip(prompt_rows, output_lengths)):
        scheduler.add(
            Sequence(
                [int(token) for token in row["input_ids"]],
                SamplingParams(
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.sampling_top_k),
                    max_tokens=int(output_len),
                    ignore_eos=True,
                ),
                seq_id=index,
            )
        )

    block_size = int(getattr(config, "block_size", 16) or 16)
    relax_bonus_boundary = os.environ.get(
        "NANO_VLLM_JAX_MTP_RELAX_BONUS_BOUNDARY",
        "0",
    ) in {"1", "true", "yes", "on", "True"}
    draft_by_seq_id: dict[int, int] = {}
    specs: set[tuple[int, int, int, int, int]] = set()
    events: list[dict[str, Any]] = []
    max_steps = (
        sum(int(row["prompt_length"]) + int(length) for row, length in zip(prompt_rows, output_lengths))
        + 16
    )
    steps = 0
    while not scheduler.is_finished():
        steps += 1
        if steps > max_steps:
            raise RuntimeError("dry MTP warmup-spec scheduler exceeded safety step limit")
        seqs, batch = scheduler.schedule()
        if batch.is_prefill:
            query_lens = [int(length) for length in (batch.query_lens_host or ())[: len(seqs)]]
            final_flags = batch.prefill_final_flags[: len(seqs)]
            generated: list[int | list[int]] = [
                0 if bool(final_flags[row]) else []
                for row in range(len(seqs))
            ]
            scheduler.postprocess(seqs, generated, prefill_chunk_lengths=query_lens)
            for row, seq in enumerate(seqs):
                if bool(final_flags[row]) and seq.temperature == 0 and seq.num_completion_tokens + 1 < seq.max_tokens:
                    draft_by_seq_id[int(seq.seq_id)] = draft_budget
                else:
                    draft_by_seq_id.pop(int(seq.seq_id), None)
            continue

        rows = list(range(len(seqs)))
        admitted_rows = [
            row
            for row, seq in enumerate(seqs)
            if bool(getattr(seq, "mtp_admitted", False))
        ]
        fused_rows = [
            row
            for row in admitted_rows
            if draft_by_seq_id.get(int(seqs[row].seq_id), 0) > 0
        ]
        generated = [0 for _ in seqs]
        if fused_rows == rows and rows:
            draft_len = min(
                draft_budget,
                min(draft_by_seq_id.get(int(seq.seq_id), draft_budget) for seq in seqs),
            )
            remaining_tokens = [
                max(0, int(seq.max_tokens - seq.num_completion_tokens))
                for seq in seqs
            ]
            min_remaining = min(remaining_tokens) if remaining_tokens else 0
            emit_bonus = True
            if min_remaining <= 0:
                scheduler.postprocess(seqs, generated)
                continue
            bonus_boundary_no_bonus = (
                not relax_bonus_boundary
                and any((seq.num_tokens + draft_len + 1) % block_size == 0 for seq in seqs)
            )
            if fixed_width_packed_verifier and (
                min_remaining < draft_len + 1 or bonus_boundary_no_bonus
            ):
                draft_len = 0
            elif min_remaining < draft_len + 1 or bonus_boundary_no_bonus:
                draft_len = min(draft_len, min_remaining)
                emit_bonus = False
            if draft_len > 0:
                burst_emit_tokens = burst_budget * (draft_len + 1)
                burst_fits_completion_budget = all(
                    seq.num_completion_tokens + burst_emit_tokens <= seq.max_tokens
                    for seq in seqs
                )
                burst_fits_with_final_bonus_clamp = all(
                    seq.num_completion_tokens + burst_emit_tokens - 1 <= seq.max_tokens
                    for seq in seqs
                )
                burst_final_bonus_boundary = (
                    not relax_bonus_boundary
                    and any((seq.num_tokens + burst_emit_tokens) % block_size == 0 for seq in seqs)
                )
                burst_groups = 1
                if (
                    emit_bonus
                    and burst_budget > 1
                    and (burst_fits_completion_budget or burst_fits_with_final_bonus_clamp)
                    and not burst_final_bonus_boundary
                ):
                    burst_groups = burst_budget
                scheduled_physical_rows = int(batch.tokens.shape[0])
                physical_rows = _mtp_static_warmup_row_count(
                    scheduled_physical_rows,
                    args,
                    config,
                )
                block_table_width = int(batch.block_tables.shape[1])
                spec = (
                    physical_rows,
                    block_table_width,
                    int(draft_len),
                    int(burst_groups),
                    1 if emit_bonus else 0,
                )
                specs.add(spec)
                if burst_groups > 1:
                    # Real rejections can leave a one-group tail.
                    specs.add((*spec[:3], 1, spec[4]))
                emitted_per_row = min(
                    burst_groups * (draft_len + (1 if emit_bonus else 0)),
                    min_remaining,
                )
                generated = [[0] * emitted_per_row for _ in seqs]
                events.append(
                    {
                        "rows": physical_rows,
                        "scheduled_rows": scheduled_physical_rows,
                        "block_table_width": block_table_width,
                        "draft_len": int(draft_len),
                        "burst_groups": int(burst_groups),
                        "emit_bonus": bool(emit_bonus),
                        "emitted_per_row": int(emitted_per_row),
                    }
                )
        scheduler.postprocess(seqs, generated)
        for seq, row_generated in zip(seqs, generated):
            seq_id = int(seq.seq_id)
            emitted = len(row_generated) if isinstance(row_generated, list) else 1
            if (
                seq.status.name != "FINISHED"
                and bool(getattr(seq, "mtp_admitted", False))
                and seq.temperature == 0
                and emitted > 0
                and seq.num_completion_tokens < seq.max_tokens
            ):
                draft_by_seq_id[seq_id] = draft_budget
            else:
                draft_by_seq_id.pop(seq_id, None)

    sorted_specs = sorted(specs)
    return {
        "enabled": True,
        "source": "optimistic_mtp_dry_scheduler",
        "dry_steps": steps,
        "specs": sorted_specs,
        "events": events[:64],
        "event_count": len(events),
        "fixed_width_packed_verifier": fixed_width_packed_verifier,
        "env": _shape_env_value(sorted_specs),
    }


def _manifest_warmup_shapes(
    prompt_rows: list[dict[str, Any]],
    output_lengths: list[int],
    args: argparse.Namespace,
    config: Any,
) -> dict[str, Any]:
    from nanovllm_jax.engine.scheduler import Scheduler
    from nanovllm_jax.engine.sequence import SamplingParams, Sequence

    scheduler = Scheduler(config)
    for index, (row, output_len) in enumerate(zip(prompt_rows, output_lengths)):
        sampling = SamplingParams(
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.sampling_top_k),
            max_tokens=int(output_len),
            ignore_eos=True,
        )
        scheduler.add(
            Sequence(
                [int(token) for token in row["input_ids"]],
                sampling,
                seq_id=index,
            )
        )

    prefill_shapes: set[tuple[int, int, int]] = set()
    prefill_slot_carry_shapes: set[tuple[int, int, int]] = set()
    prefill_step_shapes: list[tuple[int, int, int]] = []
    decode_shapes: set[tuple[int, int]] = set()
    max_steps = sum(int(row["prompt_length"]) + int(length) for row, length in zip(prompt_rows, output_lengths)) + 16
    steps = 0
    while not scheduler.is_finished():
        steps += 1
        if steps > max_steps:
            raise RuntimeError("dry warmup-shape scheduler exceeded safety step limit")
        seqs, batch = scheduler.schedule()
        if batch.is_prefill:
            prefill_shape = (
                int(batch.tokens.shape[1]),
                int(batch.block_tables.shape[0]),
                int(batch.block_tables.shape[1]),
            )
            prefill_shapes.add(prefill_shape)
            prefill_step_shapes.append(prefill_shape)
            slot_carry_mode = os.environ.get(
                "NANO_VLLM_JAX_MANIFEST_WARMUP_PREFILL_SLOT_CARRY_MODE",
                "edges",
            ).strip().lower()
            if slot_carry_mode == "all" or (
                slot_carry_mode == "final"
                and any(bool(flag) for flag in batch.prefill_final_flags[: len(seqs)])
            ):
                prefill_slot_carry_shapes.add(prefill_shape)
            query_lens = [int(length) for length in (batch.query_lens_host or ())[: len(seqs)]]
            generated = [
                0 if (batch.prefill_is_final is not None and bool(batch.prefill_is_final[row])) else []
                for row in range(len(seqs))
            ]
            scheduler.postprocess(seqs, generated, prefill_chunk_lengths=query_lens)
        else:
            decode_shapes.add(
                (
                    int(batch.tokens.shape[0]),
                    int(batch.block_tables.shape[1]),
                )
            )
            scheduler.postprocess(seqs, [0 for _ in seqs])

    if (
        str(getattr(args, "speculative_method", "none")).lower() == "mtp"
        and int(getattr(args, "num_speculative_tokens", 0) or 0) > 0
    ):
        if prefill_step_shapes and not prefill_slot_carry_shapes:
            prefill_slot_carry_shapes.add(prefill_step_shapes[0])
            prefill_slot_carry_shapes.add(prefill_step_shapes[-1])
        widths = {int(width) for _, width in decode_shapes}
        if os.environ.get("NANO_VLLM_JAX_MANIFEST_WARMUP_INCLUDE_CONFIG_WIDTHS", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            widths.update(
                int(width)
                for width in getattr(config, "decode_block_table_buckets", ()) or ()
                if int(width) > 0
            )
        extra_widths = _parse_shape_ints(
            os.environ.get("NANO_VLLM_JAX_MANIFEST_WARMUP_EXTRA_DECODE_WIDTHS", "")
        )
        widths.update(width for width in extra_widths if width > 0)
        if not widths:
            widths.add(int(getattr(config, "max_blocks_per_seq", 1) or 1))
        expand_decode_rows = os.environ.get(
            "NANO_VLLM_JAX_MANIFEST_WARMUP_EXPAND_DECODE_ROWS",
            "1",
        ) in {"1", "true", "yes", "on", "True"}
        if expand_decode_rows:
            max_rows = int(
                getattr(config, "max_num_seqs", args.max_num_seqs)
                or args.max_num_seqs
            )
            for batch_size in range(1, max_rows + 1):
                for width in widths:
                    decode_shapes.add((batch_size, width))

    mtp_table_specs = {"enabled": False, "specs": [], "env": ""}
    if (
        str(getattr(args, "speculative_method", "none")).lower() == "mtp"
        and int(getattr(args, "num_speculative_tokens", 0) or 0) > 0
        and os.environ.get(
            "NANO_VLLM_JAX_MANIFEST_WARMUP_MTP_TABLE_SPECS",
            "1",
        ) in {"1", "true", "yes", "on", "True"}
    ):
        mtp_table_specs = _manifest_mtp_table_warmup_specs(
            prompt_rows,
            output_lengths,
            args,
            config,
        )
        if (
            mtp_table_specs.get("enabled")
            and os.environ.get(
                "NANO_VLLM_JAX_MANIFEST_WARMUP_MTP_TABLE_SAFETY_NO_BONUS",
                "1",
            ) in {"1", "true", "yes", "on", "True"}
        ):
            draft_budget = max(1, int(getattr(args, "num_speculative_tokens", 1) or 1))
            spec_set = {
                tuple(int(part) for part in spec)
                for spec in mtp_table_specs.get("specs", [])
            }
            warm_all_tail_widths = os.environ.get(
                "NANO_VLLM_JAX_MANIFEST_WARMUP_MTP_TABLE_ALL_TAIL_WIDTHS",
                "0",
            ) in {"1", "true", "yes", "on", "True"}
            safety_tail_widths = (
                range(1, draft_budget + 1)
                if warm_all_tail_widths
                else (draft_budget,)
            )
            fixed_width_packed_verifier = str(
                getattr(args, "mtp_verifier_impl", "none") or "none"
            ).lower() in {"packed_prefix", "packed_prefill", "prefill_packed"}
            if fixed_width_packed_verifier:
                safety_tail_widths = (draft_budget,)
            # The fixed-K packed verifier keeps K constant at short request
            # tails and block boundaries, but suppresses the bonus and caps
            # emitted tokens per row.  Warm that exact no-bonus executable;
            # the optimistic dry schedule already contributes the normal
            # bonus-bearing route.
            safety_bonus_modes = (0,) if fixed_width_packed_verifier else (0, 1)
            mtp_decode_shapes = {
                (
                    _mtp_static_warmup_row_count(batch_size, args, config),
                    int(block_table_width),
                )
                for batch_size, block_table_width in decode_shapes
            }
            for batch_size, block_table_width in mtp_decode_shapes:
                for draft_width in safety_tail_widths:
                    for emit_bonus in safety_bonus_modes:
                        spec_set.add(
                            (
                                int(batch_size),
                                int(block_table_width),
                                int(draft_width),
                                1,
                                int(emit_bonus),
                            )
                        )
            sorted_specs = sorted(spec_set)
            mtp_table_specs["specs"] = sorted_specs
            mtp_table_specs["env"] = _shape_env_value(sorted_specs)
            mtp_table_specs["safety_tail_specs"] = True
            mtp_table_specs["safety_all_tail_widths"] = warm_all_tail_widths
            mtp_table_specs["fixed_width_packed_verifier"] = (
                fixed_width_packed_verifier
            )

    return {
        "enabled": True,
        "source": "dry_scheduler",
        "dry_steps": steps,
        "prefill_shapes": sorted(prefill_shapes),
        "prefill_slot_carry_shapes": sorted(prefill_slot_carry_shapes),
        "decode_shapes": sorted(decode_shapes),
        "decode_widths": sorted({int(width) for _, width in decode_shapes}),
        "prefill_env": _shape_env_value(sorted(prefill_shapes)),
        "prefill_slot_carry_env": _shape_env_value(sorted(prefill_slot_carry_shapes)),
        "decode_env": _shape_env_value(sorted(decode_shapes)),
        "mtp_table_warmup_specs": mtp_table_specs,
    }


def run_benchmark(args: argparse.Namespace, recorder: RunRecorder) -> dict:
    from transformers import AutoTokenizer

    from nanovllm_jax.engine.llm_engine import LLMEngine

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_rows, prompt_info = prepare_prompt_rows(tokenizer, args)
    input_lens = [int(row["prompt_length"]) for row in prompt_rows]
    output_lengths = [int(row["output_len"]) for row in prompt_rows]
    prompts = [row["input_ids"] for row in prompt_rows]

    engine_kwargs = {
        "backend": args.backend,
        "dtype": args.dtype,
        "weight_dtype": args.weight_dtype,
        "max_kv_cache_bytes": int(args.max_kv_cache_mb * 1024 * 1024),
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "max_num_seqs": args.max_num_seqs,
        "max_num_resident_seqs": (
            args.max_num_resident_seqs if args.max_num_resident_seqs > 0 else None
        ),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prefix_cache": args.prefix_cache,
        "prefill_buckets": tuple(_parse_ints(args.prefill_buckets)),
        "prefill_token_buckets": (
            tuple(_parse_ints(args.prefill_token_buckets))
            if args.prefill_token_buckets
            else tuple(_parse_ints(args.prefill_buckets))
        ),
        "prefill_layout": args.prefill_layout,
        "batch_size_buckets": tuple(_parse_ints(args.batch_size_buckets)),
        "max_blocks_per_seq": args.max_blocks_per_seq,
        "decode_block_table_buckets": tuple(_parse_ints(args.decode_block_table_buckets)),
        "jax_execution": args.jax_execution,
        "speculative_method": args.speculative_method,
        "draft_sample_method": args.draft_sample_method,
        "mtp_verifier_impl": args.mtp_verifier_impl,
        "mtp_batch_accept_policy": args.mtp_batch_accept_policy,
        "mtp_seed_after_bonus": args.mtp_seed_after_bonus,
        "mtp_bonus_margin": args.mtp_bonus_margin,
        "mtp_draft_margin": args.mtp_draft_margin,
        "mtp_hidden_source": args.mtp_hidden_source,
        "mtp_chain_hidden_source": args.mtp_chain_hidden_source,
        "mtp_chain_mode": args.mtp_chain_mode,
        "mtp_token_source": args.mtp_token_source,
        "mtp_position_offset": args.mtp_position_offset,
        "mtp_lm_head_greedy_top1_impl": args.mtp_lm_head_greedy_top1_impl,
        "mtp_draft_vocab_size": args.mtp_draft_vocab_size,
        "num_speculative_tokens": args.num_speculative_tokens,
        "mtp_burst_groups": args.mtp_burst_groups,
        "mtp_max_active_rows": args.mtp_max_active_rows,
        "mtp_prefill_seed": args.mtp_prefill_seed,
        "greedy_token_fastpath": args.greedy_token_fastpath,
        "sampled_token_fastpath": args.sampled_token_fastpath,
        "greedy_decode_burst_steps": max(1, int(args.greedy_decode_burst_steps or 1)),
        "device_token_carry": args.device_token_carry,
        "static_decode_metadata": args.static_decode_metadata,
        "static_decode_seq_lens_carry": args.static_decode_seq_lens_carry,
        "resident_decode_metadata": args.resident_decode_metadata,
        "trace_token_prefetch": args.trace_token_prefetch,
        "summary_host_token_sink_min_completion_tokens": (
            args.summary_host_token_sink_min_completion_tokens
        ),
        "summary_host_token_sink_min_avg_completion_tokens": (
            args.summary_host_token_sink_min_avg_completion_tokens
            if args.summary_host_token_sink_min_avg_completion_tokens > 0
            else None
        ),
        "materialize_tied_lm_head": args.materialize_tied_lm_head,
        "compact_prefill_in_proj_qkv": args.compact_prefill_in_proj_qkv,
        "compact_prefill_gdn_z": args.compact_prefill_gdn_z,
        "compact_prefill_full_attn_proj": args.compact_prefill_full_attn_proj,
        "compact_prefill_mlp": args.compact_prefill_mlp,
        "compact_prefill_token_count_mode": args.compact_prefill_token_count_mode,
        "lm_head_decode_act_dtype": args.lm_head_decode_act_dtype,
        "lm_head_topk_impl": args.lm_head_topk_impl,
        "lm_head_greedy_top1_impl": args.lm_head_greedy_top1_impl,
        "decode_proj_act_dtype": args.decode_proj_act_dtype,
        "decode_padded_gemm": args.decode_padded_gemm,
        "decode_padded_gemm_gate_up": args.decode_padded_gemm_gate_up,
        "decode_rms_padded_gemm": args.decode_rms_padded_gemm,
        "decode_padded_gemm_rows": args.decode_padded_gemm_rows,
        "decode_padded_gemm_max_out_dim": args.decode_padded_gemm_max_out_dim,
        "gdn_width1_packed_input_projection": (
            args.gdn_width1_packed_input_projection
        ),
        "full_attention_kv_cache_dtype": args.full_attention_kv_cache_dtype,
        "full_attention_kv_append_impl": args.full_attention_kv_append_impl,
        "full_attention_decode_impl": args.full_attention_decode_impl,
        "full_attention_prefill_impl": args.full_attention_prefill_impl,
        "gdn_disable_fallbacks": args.gdn_disable_fallbacks,
        "gdn_prefill_post_conv_impl": args.gdn_prefill_post_conv_impl,
        "gdn_prefill_qkv_dtype": args.gdn_prefill_qkv_dtype,
        "gdn_prefill_post_conv_output_dtype": args.gdn_prefill_post_conv_output_dtype,
        "gdn_packed_decode_impl": args.gdn_packed_decode_impl,
        "gdn_packed_decode_qkv_dtype": args.gdn_packed_decode_qkv_dtype,
        "gdn_packed_decode_pre_normalize_qk": args.gdn_packed_decode_pre_normalize_qk,
        "gdn_packed_decode_max_batch": (
            args.gdn_packed_decode_max_batch
            if args.gdn_packed_decode_max_batch > 0
            else None
        ),
    }
    if args.linear_chunk_size:
        engine_kwargs["linear_chunk_size"] = args.linear_chunk_size

    gpu_memory_before_engine = _gpu_memory_used_mb()
    engine = LLMEngine(
        args.model,
        **engine_kwargs,
    )
    gpu_memory_after_engine = _gpu_memory_used_mb()
    kernel_backend = getattr(engine.model_runner.backend, "kernel_backend", None)
    kernel_backend_dict = kernel_backend.as_dict() if kernel_backend is not None else None
    nhd_cache = getattr(engine.model_runner, "full_attention_nhd_cache", None)
    executor = getattr(engine.model_runner, "executor", None)
    jit_cache = getattr(executor, "_jit_cache", None)
    warmup_summary: dict[str, Any] = {
        "enabled": bool(args.warmup),
        "mode": args.warmup_mode if args.warmup else "disabled",
        "seconds": 0.0,
        "jit_cache_entries_before": len(jit_cache) if jit_cache is not None else None,
        "jit_cache_entries_after": len(jit_cache) if jit_cache is not None else None,
        "request_specific": False,
    }
    if args.warmup:
        warmup_started = time.perf_counter()
        if args.warmup_mode == "generic":
            manifest_shape_warmup = None
            use_manifest_shape_warmup = (
                args.prompt_source in {"manifest", "vllm_random"}
                and os.environ.get(
                    "NANO_VLLM_JAX_MANIFEST_WARMUP_SHAPES",
                    "1",
                )
                in {"1", "true", "yes", "on", "True"}
            )
            if use_manifest_shape_warmup:
                manifest_shape_warmup = _manifest_warmup_shapes(
                    prompt_rows,
                    output_lengths,
                    args,
                    engine.config,
                )
                os.environ["NANO_VLLM_JAX_PREFILL_WARMUP_SHAPES"] = str(
                    manifest_shape_warmup["prefill_env"]
                )
                os.environ["NANO_VLLM_JAX_PREFILL_SLOT_CARRY_WARMUP_SHAPES"] = str(
                    manifest_shape_warmup["prefill_slot_carry_env"]
                )
                os.environ["NANO_VLLM_JAX_DECODE_WARMUP_SHAPES"] = str(
                    manifest_shape_warmup["decode_env"]
                )
                mtp_table_specs = manifest_shape_warmup.get(
                    "mtp_table_warmup_specs",
                    {},
                )
                if mtp_table_specs.get("env"):
                    os.environ["NANO_VLLM_JAX_MTP_TABLE_WARMUP_SPECS"] = str(
                        mtp_table_specs["env"]
                    )
                os.environ.setdefault("NANO_VLLM_JAX_MTP_WARMUP_MINIMAL", "1")
            include_sampled_routes = not (
                float(args.temperature) == 0.0
                and float(args.top_p) == 1.0
                and int(args.sampling_top_k) == -1
            )
            if args.startup_warmup_include_sampled_routes is not None:
                include_sampled_routes = bool(args.startup_warmup_include_sampled_routes)
            warmup_summary = engine.warmup_compilation(
                max_prefill_len=max(
                    tuple(getattr(engine.config, "prefill_token_buckets", ()) or ())
                    or
                    tuple(getattr(engine.config, "prefill_buckets", ()) or ())
                    or (int(args.max_num_batched_tokens),)
                ),
                max_batch=max(
                    tuple(getattr(engine.config, "batch_size_buckets", ()) or ())
                    or (int(args.max_num_seqs),)
                ),
                include_sampled_routes=include_sampled_routes,
                prefill_token_buckets=tuple(
                    _parse_ints(args.startup_warmup_prefill_token_buckets)
                ) or None,
                batch_size_buckets=tuple(
                    _parse_ints(args.startup_warmup_batch_size_buckets)
                ) or None,
                decode_block_table_buckets=tuple(
                    _parse_ints(args.startup_warmup_decode_block_table_buckets)
                ) or None,
            )
            if manifest_shape_warmup is not None:
                warmup_summary["manifest_shape_warmup"] = manifest_shape_warmup
        else:
            warmup_params = _build_sampling_params(
                output_lengths,
                args.output_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.sampling_top_k,
            )
            warmup_trace = engine.generate_with_trace(
                prompts,
                sampling_params=warmup_params,
                include_text=False,
                trace_events=bool(args.trace_events),
            )
            _block_until_ready(warmup_trace)
            warmup_summary.update(
                {
                    "mode": "request_specific",
                    "seconds": time.perf_counter() - warmup_started,
                    "jit_cache_entries_after": len(jit_cache) if jit_cache is not None else None,
                    "request_specific": True,
                    "request_output_lengths": output_lengths,
                }
            )
    gpu_memory_after_warmup = _gpu_memory_used_mb()
    jit_cache_entries_before_measurement = len(jit_cache) if jit_cache is not None else None
    jit_cache_keys_before_measurement = _jit_cache_key_snapshot(jit_cache)

    sampling_params = _build_sampling_params(
        output_lengths,
        args.output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.sampling_top_k,
    )
    recorder.start_jax_profile(enabled=args.profile)
    started = time.perf_counter()
    trace = engine.generate_with_trace(
        prompts,
        sampling_params=sampling_params,
        include_text=False,
        trace_events=bool(args.trace_events),
    )
    _block_until_ready(trace)
    elapsed = time.perf_counter() - started
    recorder.stop_jax_profile()
    gpu_memory_after_measurement = _gpu_memory_used_mb()
    profile_counters = _profile_counters(recorder.profile_path) if args.profile else None
    jit_cache_entries_after_measurement = len(jit_cache) if jit_cache is not None else None
    jit_cache_keys_after_measurement = _jit_cache_key_snapshot(jit_cache)
    jit_cache_growth = (
        jit_cache_entries_after_measurement - jit_cache_entries_before_measurement
        if jit_cache_entries_before_measurement is not None
        and jit_cache_entries_after_measurement is not None
        else None
    )
    jit_cache_new_keys = None
    if jit_cache_keys_before_measurement is not None and jit_cache_keys_after_measurement is not None:
        jit_cache_new_keys = sorted(jit_cache_keys_after_measurement - jit_cache_keys_before_measurement)
    if args.fail_on_jit_cache_growth and jit_cache_growth and jit_cache_growth > 0:
        detail = f" new_keys={jit_cache_new_keys!r}" if jit_cache_new_keys else ""
        raise RuntimeError(
            "Executor JIT cache grew during measured generation: "
            f"{jit_cache_entries_before_measurement} -> {jit_cache_entries_after_measurement}.{detail}"
        )

    rows = []
    total_tokens = 0
    for prompt, result in zip(prompt_rows, trace["results"]):
        token_ids = [int(token) for token in result["token_ids"]]
        total_tokens += len(token_ids)
        rows.append(
            {
                "name": prompt["name"],
                "prompt_length": prompt["prompt_length"],
                "output_len": int(prompt["output_len"]),
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "topk_logprobs_by_step": [],
            }
        )

    return {
        "run_config": {
            "model": args.model,
            "dtype": args.dtype,
            "weight_dtype": args.weight_dtype,
            "backend": args.backend,
            "kernel_backend": kernel_backend_dict,
            "kernel_backend_requested": (kernel_backend_dict or {}).get("requested"),
            "kernel_backend_resolved": (kernel_backend_dict or {}).get("selected"),
            "kernel_backend_external_enabled": (kernel_backend_dict or {}).get("external_kernels_enabled"),
            "kernel_backend_unavailable_reason": (kernel_backend_dict or {}).get("reason"),
            "kernel_backend_external_call_counts": {},
            "kernel_backend_fallback_counts": {},
            "nhd_full_attention_kv_cache_enabled": nhd_cache is not None,
            "nhd_full_attention_kv_cache_shape": list(nhd_cache.k_cache.shape) if nhd_cache is not None else None,
            "nhd_full_attention_layers": list(nhd_cache.layer_indices) if nhd_cache is not None else None,
            "full_attention_kernel_flags": {
                "nhd_full_attention_kv_cache": _env_flag("NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE"),
                "flashinfer_kv_append": _env_flag("NANO_VLLM_JAX_FLASHINFER_KV_APPEND"),
                "cuda_fp32_kv_append": _env_flag("NANO_VLLM_JAX_CUDA_FP32_KV_APPEND"),
                "cuda_fp32_decode_attention": _env_flag("NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN"),
                "kv_cache_dtype": str(engine.config.full_attention_kv_cache_dtype),
                "kv_append_impl": str(engine.config.full_attention_kv_append_impl),
                "decode_impl": str(engine.config.full_attention_decode_impl),
                "prefill_impl": str(engine.config.full_attention_prefill_impl),
            },
            "gdn_kernel_flags": {
                "allow_local_cuda_probes": _env_flag(
                    "NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES"
                ),
                "cuda_fp32_gdn_decode": _env_flag("NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE"),
                "disable_fallbacks": bool(engine.config.gdn_disable_fallbacks),
                "packed_decode_impl": str(engine.config.gdn_packed_decode_impl),
                "packed_decode_pre_normalize_qk": bool(engine.config.gdn_packed_decode_pre_normalize_qk),
                "packed_decode_qkv_dtype": str(engine.config.gdn_packed_decode_qkv_dtype),
                "packed_decode_max_batch": engine.config.gdn_packed_decode_max_batch,
                "prefill_post_conv_impl": str(engine.config.gdn_prefill_post_conv_impl),
                "prefill_kkt_block_dot": _env_flag(
                    "NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT"
                ),
                "prefill_fwd_o_block_dot": _env_flag(
                    "NANO_VLLM_JAX_GDN_FWD_O_BLOCK_DOT"
                ),
                "prefill_delta_h_block_dot": _env_flag(
                    "NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT"
                ),
                "prefill_recompute_block_dot": _env_flag(
                    "NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT"
                ),
                "prefill_qkv_dtype": str(engine.config.gdn_prefill_qkv_dtype),
                "prefill_act_dtype": os.environ.get(
                    "NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE",
                    "fp32",
                ),
                "prefill_post_conv_output_dtype": str(engine.config.gdn_prefill_post_conv_output_dtype),
            },
            "jax_execution": args.jax_execution,
            "prefill_layout": str(engine.config.prefill_layout),
            "prefill_buckets": list(engine.config.prefill_buckets),
            "prefill_token_buckets": list(engine.config.prefill_token_buckets),
            "batch_size_buckets": list(engine.config.batch_size_buckets),
            "num_kvcache_blocks": int(engine.config.num_kvcache_blocks),
            "requested_num_kvcache_blocks": int(args.num_kvcache_blocks),
            "max_kv_cache_bytes": (
                int(engine.config.max_kv_cache_bytes)
                if engine.config.max_kv_cache_bytes is not None
                else None
            ),
            "requested_max_kv_cache_mb": float(args.max_kv_cache_mb),
            "max_num_seqs": int(engine.config.max_num_seqs),
            "max_num_resident_seqs": int(engine.config.max_num_resident_seqs),
            "max_num_batched_tokens": int(engine.config.max_num_batched_tokens),
            "prefix_cache": bool(engine.config.prefix_cache),
            "max_blocks_per_seq": int(engine.config.max_blocks_per_seq),
            "decode_block_table_buckets": list(engine.config.decode_block_table_buckets),
            "linear_chunk_size": int(engine.config.linear_chunk_size),
            "speculative_method": str(engine.config.speculative_method),
            "draft_sample_method": str(engine.config.draft_sample_method),
            "mtp_verifier_impl": str(engine.config.mtp_verifier_impl),
            "mtp_batch_accept_policy": str(engine.config.mtp_batch_accept_policy),
            "mtp_seed_after_bonus": bool(engine.config.mtp_seed_after_bonus),
            "mtp_prefill_seed": bool(engine.config.mtp_prefill_seed),
            "mtp_bonus_margin": float(engine.config.mtp_bonus_margin),
            "mtp_draft_margin": float(engine.config.mtp_draft_margin),
            "mtp_hidden_source": str(engine.config.mtp_hidden_source),
            "mtp_chain_hidden_source": str(engine.config.mtp_chain_hidden_source),
            "mtp_chain_mode": str(engine.config.mtp_chain_mode),
            "mtp_token_source": str(engine.config.mtp_token_source),
            "mtp_position_offset": int(engine.config.mtp_position_offset),
            "mtp_lm_head_greedy_top1_impl": str(engine.config.mtp_lm_head_greedy_top1_impl),
            "num_speculative_tokens": int(engine.config.num_speculative_tokens),
            "mtp_burst_groups": int(engine.config.mtp_burst_groups),
            "mtp_max_active_rows": int(engine.config.mtp_max_active_rows),
            "mtp_prefill_seed": bool(engine.config.mtp_prefill_seed),
            "sampling": {
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.sampling_top_k),
            },
            **prompt_info,
            "greedy_token_fastpath": bool(engine.config.greedy_token_fastpath),
            "sampled_token_fastpath": bool(engine.config.sampled_token_fastpath),
            "warmup": warmup_summary,
            "jit_cache_audit": {
                "entries_before_measurement": jit_cache_entries_before_measurement,
                "entries_after_measurement": jit_cache_entries_after_measurement,
                "growth_during_measurement": jit_cache_growth,
                "new_keys": jit_cache_new_keys,
                "fail_on_growth": bool(args.fail_on_jit_cache_growth),
            },
            "serving_fastpath_flags": {
                "greedy_token_fastpath": bool(engine.config.greedy_token_fastpath),
                "sampled_token_fastpath": bool(engine.config.sampled_token_fastpath),
                "materialize_tied_lm_head": bool(engine.config.materialize_tied_lm_head),
                "compact_prefill_in_proj_qkv": bool(engine.config.compact_prefill_in_proj_qkv),
                "compact_prefill_gdn_z": bool(engine.config.compact_prefill_gdn_z),
                "compact_prefill_full_attn_proj": bool(engine.config.compact_prefill_full_attn_proj),
                "compact_prefill_mlp": bool(engine.config.compact_prefill_mlp),
                "compact_prefill_token_count_mode": str(engine.config.compact_prefill_token_count_mode),
                "lm_head_decode_act_dtype": str(engine.config.lm_head_decode_act_dtype),
                "lm_head_topk_impl": str(engine.config.lm_head_topk_impl),
                "lm_head_greedy_top1_impl": str(engine.config.lm_head_greedy_top1_impl),
                "decode_proj_act_dtype": str(engine.config.decode_proj_act_dtype),
                "decode_padded_gemm": bool(engine.config.decode_padded_gemm),
                "decode_padded_gemm_gate_up": bool(engine.config.decode_padded_gemm_gate_up),
                "decode_rms_padded_gemm": bool(engine.config.decode_rms_padded_gemm),
                "decode_padded_gemm_rows": int(engine.config.decode_padded_gemm_rows),
                "decode_padded_gemm_max_out_dim": int(engine.config.decode_padded_gemm_max_out_dim),
                "device_token_carry": bool(engine.config.device_token_carry),
                "static_decode_metadata": bool(engine.config.static_decode_metadata),
                "static_decode_seq_lens_carry": bool(engine.config.static_decode_seq_lens_carry),
                "resident_decode_metadata": bool(engine.config.resident_decode_metadata),
                "greedy_decode_burst_steps": int(engine.config.greedy_decode_burst_steps),
                "trace_token_prefetch": bool(engine.config.trace_token_prefetch),
                "summary_host_token_sink_min_completion_tokens": int(
                    engine.config.summary_host_token_sink_min_completion_tokens
                ),
                "summary_host_token_sink_min_avg_completion_tokens": int(
                    engine.config.summary_host_token_sink_min_avg_completion_tokens
                )
                if engine.config.summary_host_token_sink_min_avg_completion_tokens is not None
                else None,
            },
            "trace_mode": "events" if args.trace_events else "summary",
        },
        "performance": _performance_with_token_scopes(
            rows,
            _timing_metrics_from_trace(trace, elapsed, total_tokens),
            elapsed,
        ),
        "memory": {
            "gpu_memory_mb_before_engine": gpu_memory_before_engine,
            "gpu_memory_mb_after_engine": gpu_memory_after_engine,
            "gpu_memory_mb_after_warmup": gpu_memory_after_warmup,
            "gpu_memory_mb_after_measurement": gpu_memory_after_measurement,
        },
        "rows": rows,
        "events": trace["events"] if args.trace_events else [],
        "timing_summary": trace.get("timing_summary"),
        "profile_counters": profile_counters,
        "speculative": engine.model_runner.get_speculative_stats(),
        "mtp_admission": engine.get_mtp_admission_report(),
    }


def main() -> None:
    args = parse_args()
    recorder = RunRecorder.create(
        script=Path(__file__).name,
        args=vars(args),
        run_label=args.run_label or "jax_server_trace",
        profile_dir=args.profile_dir or None,
        run_log=args.run_log or None,
    )
    try:
        summary = run_benchmark(args, recorder)
        correctness = compare_reference(summary, args.reference_json)
        summary["correctness"] = correctness
        if correctness.get("checked") and not correctness.get("ok"):
            recorder.record_issue(
                summary="JAX server trace generated tokens diverged from the reference JSON",
                severity="error",
                status="open",
                details=correctness,
                learnings=["Streaming/trace throughput is only useful when generated tokens match the reference."],
                resolution="pending",
            )
        summary["run"] = recorder.metadata()
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
        recorder.finish(
            status="ok" if correctness.get("ok", True) else "failed_correctness",
            summary={
                "performance": summary["performance"],
                "correctness": correctness,
                "speculative_method": (
                    (summary.get("run_config") or summary.get("config") or {}).get(
                        "speculative_method",
                        args.speculative_method,
                    )
                ),
                "num_speculative_tokens": args.num_speculative_tokens,
            },
            learnings=["JAX server-path timing now records per-token step timestamps."],
            resolution="Use matching vLLM async artifacts for ITL comparison.",
        )
    except Exception as exc:
        recorder.stop_jax_profile()
        recorder.finish_exception(exc)
        raise


if __name__ == "__main__":
    main()
