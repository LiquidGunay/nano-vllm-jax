#!/usr/bin/env python3
"""Server-path JAX benchmark with per-token timing traces."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--num-speculative-tokens", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-kv-cache-mb", type=int, default=1024)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--prefill-buckets", default="16,32,64,128")
    parser.add_argument("--batch-size-buckets", default="1,2,4")
    parser.add_argument("--max-blocks-per-seq", type=int, default=16)
    parser.add_argument(
        "--linear-chunk-size",
        type=int,
        default=0,
        help="Override Qwen3.5 Gated DeltaNet prefill chunk size; 0 keeps the model default.",
    )
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
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
    for event in events:
        if event.get("event") != "token":
            continue
        by_request.setdefault(int(event["request_index"]), []).append(float(event["elapsed_seconds"]))
    ttfts = []
    itls = []
    for timestamps in by_request.values():
        if timestamps:
            ttfts.append(1000.0 * timestamps[0])
            itls.extend(1000.0 * (right - left) for left, right in zip(timestamps, timestamps[1:]))
    return {
        "seconds": elapsed,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
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


def _build_sampling_params(output_lengths: list[int], default_output_len: int):
    from nanovllm_jax.engine.sequence import SamplingParams

    if output_lengths:
        return [
            SamplingParams(temperature=0.0, max_tokens=int(length), ignore_eos=True)
            for length in output_lengths
        ]
    return SamplingParams(temperature=0.0, max_tokens=int(default_output_len), ignore_eos=True)


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
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prefill_buckets": tuple(_parse_ints(args.prefill_buckets)),
        "batch_size_buckets": tuple(_parse_ints(args.batch_size_buckets)),
        "max_blocks_per_seq": args.max_blocks_per_seq,
        "jax_execution": args.jax_execution,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    if args.linear_chunk_size:
        engine_kwargs["linear_chunk_size"] = args.linear_chunk_size

    engine = LLMEngine(
        args.model,
        **engine_kwargs,
    )
    kernel_backend = getattr(engine.model_runner.backend, "kernel_backend", None)
    kernel_backend_dict = kernel_backend.as_dict() if kernel_backend is not None else None
    nhd_cache = getattr(engine.model_runner, "full_attention_nhd_cache", None)
    if args.warmup:
        warmup_params = _build_sampling_params([], min(2, args.output_len))
        engine.generate_with_trace(prompts, sampling_params=warmup_params, include_text=False)

    sampling_params = _build_sampling_params(output_lengths, args.output_len)
    recorder.start_jax_profile(enabled=args.profile)
    started = time.perf_counter()
    trace = engine.generate_with_trace(prompts, sampling_params=sampling_params, include_text=False)
    elapsed = time.perf_counter() - started
    recorder.stop_jax_profile()
    profile_counters = _profile_counters(recorder.profile_path) if args.profile else None

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
            },
            "gdn_kernel_flags": {
                "cuda_fp32_gdn_decode": _env_flag("NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE"),
                "packed_decode_impl": os.environ.get(
                    "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL",
                    "off",
                ),
            },
            "jax_execution": args.jax_execution,
            "linear_chunk_size": int(engine.config.linear_chunk_size),
            "num_speculative_tokens": args.num_speculative_tokens,
            **prompt_info,
            "greedy_token_fastpath": _env_flag("NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH"),
            "serving_fastpath_flags": {
                "greedy_token_fastpath": _env_flag("NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH"),
                "materialize_tied_lm_head": _env_flag("NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD"),
                "compact_prefill_in_proj_qkv": _env_flag("NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV"),
                "compact_prefill_gdn_z": _env_flag("NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z"),
                "compact_prefill_full_attn_proj": _env_flag("NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ"),
                "compact_prefill_mlp": _env_flag("NANO_VLLM_JAX_COMPACT_PREFILL_MLP"),
            },
        },
        "performance": _performance_with_token_scopes(
            rows,
            _timing_metrics(trace["events"], elapsed, total_tokens),
            elapsed,
        ),
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
