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
    parser.add_argument("--mtp-verifier-impl", choices=["two_decode", "commit_select", "k_decode"], default="two_decode")
    parser.add_argument("--mtp-batch-accept-policy", choices=["rowwise", "all_or_none"], default="rowwise")
    parser.add_argument("--mtp-seed-after-bonus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mtp-bonus-margin", type=float, default=0.0)
    parser.add_argument("--mtp-draft-margin", type=float, default=0.0)
    parser.add_argument("--mtp-hidden-source", choices=["pre_norm", "final_normed"], default="pre_norm")
    parser.add_argument("--mtp-token-source", choices=["generated", "current"], default="generated")
    parser.add_argument("--mtp-position-offset", type=int, default=0)
    parser.add_argument("--mtp-lm-head-greedy-top1-impl", default="jax")
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
    parser.add_argument("--prefill-buckets", default="16,32,64,128")
    parser.add_argument("--prefill-token-buckets", default="")
    parser.add_argument("--prefill-layout", choices=["packed", "dense"], default="packed")
    parser.add_argument("--batch-size-buckets", default="1,2,4")
    parser.add_argument("--max-blocks-per-seq", type=int, default=16)
    parser.add_argument("--decode-block-table-buckets", default="")
    parser.add_argument("--greedy-token-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampled-token-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-decode-burst-steps", type=int, default=1)
    parser.add_argument("--device-token-carry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-decode-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-decode-seq-lens-carry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resident-decode-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-token-prefetch", action=argparse.BooleanOptionalAction, default=True)
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
        "ttft_ms_mean": float(sum(ttfts) / len(ttfts)) if ttfts else None,
        "ttft_ms_p50": _percentile(ttfts, 50),
        "ttft_ms_p95": _percentile(ttfts, 95),
        "itl_ms_mean": float(sum(itls) / len(itls)) if itls else None,
        "itl_ms_p50": _percentile(itls, 50),
        "itl_ms_p95": _percentile(itls, 95),
        "itl_source": "jax_server_step_trace",
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
        "mtp_token_source": args.mtp_token_source,
        "mtp_position_offset": args.mtp_position_offset,
        "mtp_lm_head_greedy_top1_impl": args.mtp_lm_head_greedy_top1_impl,
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
            include_sampled_routes = not (
                float(args.temperature) == 0.0
                and float(args.top_p) == 1.0
                and int(args.sampling_top_k) == -1
            )
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
            )
        else:
            warmup_params = _build_sampling_params(
                output_lengths,
                args.output_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.sampling_top_k,
            )
            warmup_trace = engine.generate_with_trace(prompts, sampling_params=warmup_params, include_text=False)
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
    trace = engine.generate_with_trace(prompts, sampling_params=sampling_params, include_text=False)
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
            },
        },
        "performance": _performance_with_token_scopes(
            rows,
            _timing_metrics(trace["events"], elapsed, total_tokens),
            elapsed,
        ),
        "memory": {
            "gpu_memory_mb_before_engine": gpu_memory_before_engine,
            "gpu_memory_mb_after_engine": gpu_memory_after_engine,
            "gpu_memory_mb_after_warmup": gpu_memory_after_warmup,
            "gpu_memory_mb_after_measurement": gpu_memory_after_measurement,
        },
        "rows": rows,
        "events": trace["events"],
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
