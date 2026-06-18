#!/usr/bin/env python3
"""Run multiple prompt suites against one warmed JAX server process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_gpu_matrix import CONFIG_DIR, WORKLOADS
from nanovllm_jax.server_config import engine_overrides_from_config, runtime_env_from_config


RANDOM_LARGE_ENVELOPE = {
    "max_kv_cache_mb": 2048,
    "num_kvcache_blocks": 2048,
    "max_num_seqs": 8,
    "max_num_resident_seqs": 8,
    "max_num_batched_tokens": 1024,
    "prefill_buckets": "512,1024",
    "prefill_token_buckets": "512,1024",
    "prefill_layout": "packed",
    "batch_size_buckets": "1,2,3,4,5,6,7,8",
    "max_blocks_per_seq": 128,
}

PSEUDO_WORKLOADS = {"random_large"}
RANDOM_LARGE_WORKLOAD = {
    "seed": 1234,
    "num_requests": 8,
    "min_input_tokens": 512,
    "max_input_tokens": 1024,
    "min_output_tokens": 128,
    "max_output_tokens": 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-config", default=str(CONFIG_DIR / "gpu_paged_gdn_fla_decode_static_metadata.json"))
    parser.add_argument("--serving-envelope", choices=["random_large"], default="random_large")
    parser.add_argument("--workloads", default="hetero8")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--weight-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="jit")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sampling-top-k", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-jit-cache-growth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--trace-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store detailed per-token events. The default summary mode keeps "
            "TTFT/ITL metrics without per-token Python event overhead."
        ),
    )
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json", default="results/jax_server_multisuite.json")
    parser.add_argument("--run-label", default="jax_server_multisuite")
    parser.add_argument("--reference-dir", default="")
    return parser.parse_args()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


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


def _jit_cache_snapshot(cache: Any) -> set[str] | None:
    if cache is None:
        return None
    return {repr(key) for key in cache.keys()}


def _reset_mtp_measurement_counters(engine: Any) -> None:
    model_runner = getattr(engine, "model_runner", None)
    if model_runner is not None and hasattr(model_runner, "reset_speculative_stats"):
        model_runner.reset_speculative_stats()
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "reset_mtp_admission"):
        scheduler.reset_mtp_admission()


def _workload_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    missing = [name for name in names if name not in WORKLOADS and name not in PSEUDO_WORKLOADS]
    if missing:
        raise ValueError(f"unknown workload(s): {', '.join(missing)}")
    return names


def _prompt_args(base_args: argparse.Namespace, workload_name: str, output_json: Path) -> argparse.Namespace:
    if workload_name in PSEUDO_WORKLOADS:
        raise ValueError(f"{workload_name} is generated directly, not through prepare_prompt_rows")
    workload = WORKLOADS[workload_name]
    return SimpleNamespace(
        input_lens=workload.input_lens,
        output_len=workload.output_len,
        output_lengths="",
        prompt_suite=workload.prompt_suite,
        prompt_source=workload.prompt_source,
        prompt_manifest_jsonl="",
        prompt_manifest_output_jsonl="",
        dataset_name=workload.dataset_name or "",
        num_prompts=workload.num_prompts or 0,
        seed=workload.seed,
        random_input_len=workload.random_input_len or 0,
        random_output_len=workload.random_output_len or 0,
        random_range_ratio=workload.random_range_ratio or '{"input":0.0,"output":0.0}',
        output_json=str(output_json),
    )


def _reference_for(config: dict[str, Any], workload_name: str, reference_dir: str) -> str:
    if reference_dir:
        candidate = Path(reference_dir) / f"{workload_name}.json"
        if candidate.exists():
            return str(candidate)
    if workload_name in PSEUDO_WORKLOADS:
        return ""
    mapping = config.get("workload_reference_jsons") or {}
    value = mapping.get(workload_name) or config.get("reference_json") or ""
    return str(REPO_ROOT / value) if value and not Path(value).is_absolute() else str(value)


def _random_large_prompt_rows(tokenizer: Any, output_json: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from benchmarks.benchmark_random_request_sidecar import (
        generate_random_request_rows,
        write_prompt_manifest,
    )

    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    rows = generate_random_request_rows(
        seed=int(RANDOM_LARGE_WORKLOAD["seed"]),
        num_requests=int(RANDOM_LARGE_WORKLOAD["num_requests"]),
        min_input_tokens=int(RANDOM_LARGE_WORKLOAD["min_input_tokens"]),
        max_input_tokens=int(RANDOM_LARGE_WORKLOAD["max_input_tokens"]),
        min_output_tokens=int(RANDOM_LARGE_WORKLOAD["min_output_tokens"]),
        max_output_tokens=int(RANDOM_LARGE_WORKLOAD["max_output_tokens"]),
        token_vocab_size=vocab_size,
        eos_token_id=eos_id,
    )
    manifest_path = output_json.with_suffix(".prompts.jsonl")
    manifest_sha = write_prompt_manifest(rows, manifest_path)
    prompt_rows = [
        {
            "name": row["name"],
            "request_id": row["request_id"],
            "prompt_length": int(row["prompt_len"]),
            "input_ids": [int(token) for token in row["prompt_token_ids"]],
            "output_len": int(row["output_len"]),
        }
        for row in rows
    ]
    input_lens = [int(row["prompt_length"]) for row in prompt_rows]
    output_lens = [int(row["output_len"]) for row in prompt_rows]
    return prompt_rows, {
        "prompt_source": "manifest",
        "prompt_suite": "mixed",
        "dataset_name": "random",
        "num_prompts": len(prompt_rows),
        "seed": int(RANDOM_LARGE_WORKLOAD["seed"]),
        "prompt_manifest_jsonl": str(manifest_path),
        "prompt_manifest_sha256": manifest_sha,
        "input_lens": input_lens,
        "output_len": None,
        "output_lengths": output_lens,
        "random_input_len": 0,
        "random_output_len": 0,
        "random_range_ratio": {"input": 0.0, "output": 0.0},
        "input_range": {
            "min": int(RANDOM_LARGE_WORKLOAD["min_input_tokens"]),
            "max": int(RANDOM_LARGE_WORKLOAD["max_input_tokens"]),
        },
        "output_range": {
            "min": int(RANDOM_LARGE_WORKLOAD["min_output_tokens"]),
            "max": int(RANDOM_LARGE_WORKLOAD["max_output_tokens"]),
        },
    }


def _engine_kwargs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    config_args = dict(engine_overrides_from_config(config))
    config_args.update(config.get("args", {}))
    if args.serving_envelope == "random_large":
        config_args.update(RANDOM_LARGE_ENVELOPE)
    config_args.update(
        {
            "model": args.model,
            "backend": args.backend,
            "dtype": args.dtype,
            "weight_dtype": args.weight_dtype,
            "jax_execution": args.jax_execution,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_top_k": args.sampling_top_k,
            "top_k": args.top_k,
            "prefix_cache": args.prefix_cache,
        }
    )
    return {
        "backend": config_args["backend"],
        "dtype": config_args["dtype"],
        "weight_dtype": config_args["weight_dtype"],
        "max_kv_cache_bytes": int(config_args["max_kv_cache_mb"] * 1024 * 1024),
        "num_kvcache_blocks": int(config_args["num_kvcache_blocks"]),
        "max_num_seqs": int(config_args["max_num_seqs"]),
        "max_num_resident_seqs": int(config_args["max_num_resident_seqs"]),
        "max_num_batched_tokens": int(config_args["max_num_batched_tokens"]),
        "prefix_cache": bool(config_args.get("prefix_cache", True)),
        "prefill_buckets": _parse_ints(config_args["prefill_buckets"]),
        "prefill_token_buckets": _parse_ints(config_args["prefill_token_buckets"]),
        "prefill_layout": str(config_args["prefill_layout"]),
        "batch_size_buckets": _parse_ints(config_args["batch_size_buckets"]),
        "max_blocks_per_seq": int(config_args["max_blocks_per_seq"]),
        "decode_block_table_buckets": _parse_ints(config_args.get("decode_block_table_buckets", "")),
        "jax_execution": config_args["jax_execution"],
        "speculative_method": str(config_args.get("speculative_method", "none")),
        "draft_sample_method": str(config_args.get("draft_sample_method", "greedy")),
        "mtp_verifier_impl": str(config_args.get("mtp_verifier_impl", "two_decode")),
        "mtp_batch_accept_policy": str(config_args.get("mtp_batch_accept_policy", "rowwise")),
        "mtp_seed_after_bonus": bool(config_args.get("mtp_seed_after_bonus", False)),
        "mtp_bonus_margin": float(config_args.get("mtp_bonus_margin", 0.0)),
        "mtp_draft_margin": float(config_args.get("mtp_draft_margin", 0.0)),
        "mtp_hidden_source": str(config_args.get("mtp_hidden_source", "pre_norm")),
        "mtp_token_source": str(config_args.get("mtp_token_source", "generated")),
        "mtp_position_offset": int(config_args.get("mtp_position_offset", 0)),
        "mtp_lm_head_greedy_top1_impl": str(
            config_args.get("mtp_lm_head_greedy_top1_impl", "jax")
        ),
        "num_speculative_tokens": int(config_args.get("num_speculative_tokens", 0)),
        "mtp_burst_groups": int(config_args.get("mtp_burst_groups", 1)),
        "mtp_max_active_rows": int(config_args.get("mtp_max_active_rows", 0)),
        "mtp_prefill_seed": bool(config_args.get("mtp_prefill_seed", True)),
        "mtp_unverified_draft_append": bool(config_args.get("mtp_unverified_draft_append", False)),
        "mtp_unverified_fused_append": bool(config_args.get("mtp_unverified_fused_append", False)),
        "greedy_token_fastpath": bool(config_args.get("greedy_token_fastpath", True)),
        "sampled_token_fastpath": bool(config_args.get("sampled_token_fastpath", True)),
        "greedy_decode_burst_steps": max(1, int(config_args.get("greedy_decode_burst_steps", 1) or 1)),
        "device_token_carry": bool(config_args.get("device_token_carry", False)),
        "static_decode_metadata": bool(config_args.get("static_decode_metadata", False)),
        "static_decode_seq_lens_carry": bool(config_args.get("static_decode_seq_lens_carry", False)),
        "resident_decode_metadata": bool(config_args.get("resident_decode_metadata", False)),
        "trace_token_prefetch": bool(config_args.get("trace_token_prefetch", True)),
        "materialize_tied_lm_head": bool(config_args.get("materialize_tied_lm_head", False)),
        "compact_prefill_in_proj_qkv": bool(config_args.get("compact_prefill_in_proj_qkv", False)),
        "compact_prefill_gdn_z": bool(config_args.get("compact_prefill_gdn_z", False)),
        "compact_prefill_full_attn_proj": bool(config_args.get("compact_prefill_full_attn_proj", False)),
        "compact_prefill_mlp": bool(config_args.get("compact_prefill_mlp", False)),
        "compact_prefill_token_count_mode": str(config_args.get("compact_prefill_token_count_mode", "exact")),
        "lm_head_decode_act_dtype": str(config_args.get("lm_head_decode_act_dtype", "fp32")),
        "lm_head_topk_impl": str(config_args.get("lm_head_topk_impl", "jax")),
        "lm_head_greedy_top1_impl": str(config_args.get("lm_head_greedy_top1_impl", "jax")),
        "decode_proj_act_dtype": str(config_args.get("decode_proj_act_dtype", "fp32")),
        "decode_padded_gemm": bool(config_args.get("decode_padded_gemm", False)),
        "decode_padded_gemm_gate_up": bool(config_args.get("decode_padded_gemm_gate_up", False)),
        "decode_rms_padded_gemm": bool(config_args.get("decode_rms_padded_gemm", False)),
        "decode_padded_gemm_rows": int(config_args.get("decode_padded_gemm_rows", 8)),
        "decode_padded_gemm_max_out_dim": int(config_args.get("decode_padded_gemm_max_out_dim", 300000)),
        "full_attention_kv_cache_dtype": str(config_args.get("full_attention_kv_cache_dtype", "default")),
        "full_attention_kv_append_impl": str(config_args.get("full_attention_kv_append_impl", "reference")),
        "full_attention_decode_impl": str(config_args.get("full_attention_decode_impl", "reference")),
        "full_attention_prefill_impl": str(config_args.get("full_attention_prefill_impl", "reference")),
        "gdn_disable_fallbacks": bool(config_args.get("gdn_disable_fallbacks", False)),
        "gdn_prefill_post_conv_impl": str(config_args.get("gdn_prefill_post_conv_impl", "off")),
        "gdn_prefill_qkv_dtype": str(config_args.get("gdn_prefill_qkv_dtype", "fp32")),
        "gdn_prefill_post_conv_output_dtype": str(config_args.get("gdn_prefill_post_conv_output_dtype", "fp32")),
        "gdn_packed_decode_impl": str(config_args.get("gdn_packed_decode_impl", "off")),
        "gdn_packed_decode_qkv_dtype": str(config_args.get("gdn_packed_decode_qkv_dtype", "fp32")),
        "gdn_packed_decode_pre_normalize_qk": bool(config_args.get("gdn_packed_decode_pre_normalize_qk", False)),
        "gdn_packed_decode_max_batch": (
            int(config_args["gdn_packed_decode_max_batch"])
            if int(config_args.get("gdn_packed_decode_max_batch", 0) or 0) > 0
            else None
        ),
        "linear_chunk_size": int(config_args.get("linear_chunk_size", 0) or 0),
    }


def _rows_from_trace(prompt_rows: list[dict[str, Any]], trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for prompt, result in zip(prompt_rows, trace["results"]):
        token_ids = [int(token) for token in result["token_ids"]]
        rows.append(
            {
                "name": prompt["name"],
                "prompt_length": int(prompt["prompt_length"]),
                "output_len": int(prompt["output_len"]),
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "topk_logprobs_by_step": [],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = _load_config(args.jax_config)
    os.environ.update(runtime_env_from_config(config))

    # Import after applying config-derived env so XLA/cache setup sees it.
    from benchmarks import benchmark_jax_server_trace as trace_mod
    from benchmarks.benchmark_vllm_qwen35 import compare_reference, prepare_prompt_rows
    from run_tracking import RunRecorder
    from transformers import AutoTokenizer
    from nanovllm_jax.engine.llm_engine import LLMEngine

    workload_names = _workload_names(args.workloads)
    output_path = Path(args.output_json)
    recorder = RunRecorder.create(
        script=Path(__file__).name,
        args=vars(args),
        run_label=args.run_label,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine_kwargs = _engine_kwargs(args, config)
    if not engine_kwargs["linear_chunk_size"]:
        engine_kwargs.pop("linear_chunk_size", None)
    gpu_before_engine = trace_mod._gpu_memory_used_mb()
    engine = LLMEngine(args.model, **engine_kwargs)
    gpu_after_engine = trace_mod._gpu_memory_used_mb()
    executor = getattr(engine.model_runner, "executor", None)
    jit_cache = getattr(executor, "_jit_cache", None)

    warmup_summary: dict[str, Any] = {"enabled": bool(args.warmup)}
    if args.warmup:
        include_sampled_routes = not (
            float(args.temperature) == 0.0
            and float(args.top_p) == 1.0
            and int(args.sampling_top_k) == -1
        )
        warmup_summary = engine.warmup_compilation(
            max_prefill_len=max(tuple(getattr(engine.config, "prefill_token_buckets", ()) or (1024,))),
            max_batch=max(tuple(getattr(engine.config, "batch_size_buckets", ()) or (8,))),
            include_sampled_routes=include_sampled_routes,
        )
    gpu_after_warmup = trace_mod._gpu_memory_used_mb()

    workload_results: dict[str, Any] = {}
    for workload_name in workload_names:
        workload_output = output_path.with_name(f"{output_path.stem}_{workload_name}.json")
        if workload_name == "random_large":
            prompt_rows, prompt_info = _random_large_prompt_rows(tokenizer, workload_output)
            default_output_len = max(int(row["output_len"]) for row in prompt_rows)
        else:
            prompt_args = _prompt_args(args, workload_name, workload_output)
            prompt_rows, prompt_info = prepare_prompt_rows(tokenizer, prompt_args)
            default_output_len = WORKLOADS[workload_name].output_len
        output_lengths = [int(row["output_len"]) for row in prompt_rows]
        prompts = [row["input_ids"] for row in prompt_rows]
        sampling_params = trace_mod._build_sampling_params(
            output_lengths,
            default_output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.sampling_top_k,
        )

        _reset_mtp_measurement_counters(engine)
        before_count = len(jit_cache) if jit_cache is not None else None
        before_keys = _jit_cache_snapshot(jit_cache)
        recorder.start_jax_profile(enabled=args.profile)
        started = time.perf_counter()
        trace = engine.generate_with_trace(
            prompts,
            sampling_params=sampling_params,
            include_text=False,
            trace_events=bool(args.trace_events),
        )
        trace_mod._block_until_ready(trace)
        elapsed = time.perf_counter() - started
        recorder.stop_jax_profile()
        after_count = len(jit_cache) if jit_cache is not None else None
        after_keys = _jit_cache_snapshot(jit_cache)
        growth = after_count - before_count if before_count is not None and after_count is not None else None
        new_keys = sorted(after_keys - before_keys) if before_keys is not None and after_keys is not None else None
        if args.fail_on_jit_cache_growth and growth and growth > 0:
            raise RuntimeError(
                f"{workload_name}: executor JIT cache grew during measurement "
                f"{before_count} -> {after_count}; new_keys={new_keys!r}"
            )

        rows = _rows_from_trace(prompt_rows, trace)
        total_tokens = sum(row["generated_tokens"] for row in rows)
        performance = trace_mod._performance_with_token_scopes(
            rows,
            trace_mod._timing_metrics_from_trace(trace, elapsed, total_tokens),
            elapsed,
        )
        summary = {
            "run_config": {
                "workload": workload_name,
                "serving_envelope": args.serving_envelope,
                "model": args.model,
                "dtype": args.dtype,
                "weight_dtype": args.weight_dtype,
                "backend": args.backend,
                "jax_execution": args.jax_execution,
                "max_kv_cache_mb": engine_kwargs["max_kv_cache_bytes"] // (1024 * 1024),
                "num_kvcache_blocks": engine_kwargs["num_kvcache_blocks"],
                "max_num_seqs": engine_kwargs["max_num_seqs"],
                "max_num_resident_seqs": engine_kwargs["max_num_resident_seqs"],
                "max_num_batched_tokens": engine_kwargs["max_num_batched_tokens"],
                "prefix_cache": bool(engine_kwargs["prefix_cache"]),
                "prefill_buckets": list(engine_kwargs["prefill_buckets"]),
                "prefill_token_buckets": list(engine_kwargs["prefill_token_buckets"]),
                "batch_size_buckets": list(engine_kwargs["batch_size_buckets"]),
                "max_blocks_per_seq": engine_kwargs["max_blocks_per_seq"],
                "speculative_method": str(engine.config.speculative_method),
                "draft_sample_method": str(engine.config.draft_sample_method),
                "num_speculative_tokens": int(engine.config.num_speculative_tokens),
                "mtp_verifier_impl": str(engine.config.mtp_verifier_impl),
                "mtp_batch_accept_policy": str(engine.config.mtp_batch_accept_policy),
                "mtp_seed_after_bonus": bool(engine.config.mtp_seed_after_bonus),
                "mtp_burst_groups": int(engine.config.mtp_burst_groups),
                "mtp_max_active_rows": int(engine.config.mtp_max_active_rows),
                "mtp_prefill_seed": bool(engine.config.mtp_prefill_seed),
                "mtp_unverified_draft_append": bool(engine.config.mtp_unverified_draft_append),
                "mtp_unverified_fused_append": bool(engine.config.mtp_unverified_fused_append),
                "mtp_hidden_source": str(engine.config.mtp_hidden_source),
                "mtp_token_source": str(engine.config.mtp_token_source),
                "mtp_position_offset": int(engine.config.mtp_position_offset),
                "mtp_lm_head_greedy_top1_impl": str(engine.config.mtp_lm_head_greedy_top1_impl),
                "greedy_decode_burst_steps": int(engine.config.greedy_decode_burst_steps),
                "warmup": warmup_summary,
                "trace_mode": "events" if args.trace_events else "summary",
                "jit_cache_audit": {
                    "entries_before_measurement": before_count,
                    "entries_after_measurement": after_count,
                    "growth_during_measurement": growth,
                    "new_keys": new_keys,
                    "fail_on_growth": bool(args.fail_on_jit_cache_growth),
                },
                **prompt_info,
            },
            "performance": performance,
            "memory": {
                "gpu_memory_mb_before_engine": gpu_before_engine,
                "gpu_memory_mb_after_engine": gpu_after_engine,
                "gpu_memory_mb_after_warmup": gpu_after_warmup,
                "gpu_memory_mb_after_measurement": trace_mod._gpu_memory_used_mb(),
            },
            "rows": rows,
            "events": trace["events"] if args.trace_events else [],
            "timing_summary": trace.get("timing_summary"),
            "speculative": engine.model_runner.get_speculative_stats(),
            "mtp_admission": engine.get_mtp_admission_report(),
            "correctness": compare_reference(
                {"rows": rows},
                _reference_for(config, workload_name, args.reference_dir),
            ),
        }
        workload_results[workload_name] = summary

    output = {
        "schema_version": 1,
        "run_config": {
            "jax_config": args.jax_config,
            "serving_envelope": args.serving_envelope,
            "workloads": workload_names,
            "runtime_env": runtime_env_from_config(config),
        },
        "warmup": warmup_summary,
        "workloads": workload_results,
        "run": recorder.metadata(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recorder.finish(
        status="ok",
        summary={
            "workloads": {
                name: result["performance"]
                for name, result in workload_results.items()
            }
        },
        learnings=["Multi-suite benchmark keeps one warmed JAX server process across workloads."],
        resolution="Use for shared-envelope validation before broad matrix claims.",
    )


if __name__ == "__main__":
    main()
