#!/usr/bin/env python3
"""Standalone Gated DeltaNet prefill kernel microbenchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

from run_tracking import RunRecorder
from runtime_paths import configure_compilation_cache, configure_xla_flags

configure_xla_flags()
configure_compilation_cache()

import jax
import jax.numpy as jnp

from nanovllm_jax.model import jax_chunk_gated_delta_rule

jax.config.update("jax_default_matmul_precision", "highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--lengths", default="64,128,192,256,320,384,448,512")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variants", default="current_jax_chunk32_padded")
    parser.add_argument("--output-json", default="results/gdn_prefill_kernel_baseline_hetero8_64_512x32.json")
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=os.environ.get("NANO_VLLM_JAX_PROFILE", "1") not in {"0", "false", "False", "no", "off"},
    )
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--run-label", default="gdn_prefill_kernel_hetero8_64_512x32")
    return parser.parse_args()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


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


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()
    return value


def _trace_annotation(name: str):
    annotation = getattr(jax.profiler, "TraceAnnotation", None)
    if annotation is None:
        return nullcontext()
    return annotation(name)


def _make_inputs(args: argparse.Namespace, lengths: tuple[int, ...]):
    if len(lengths) != args.batch_size:
        raise ValueError("--lengths must contain exactly --batch-size entries")
    if any(length < 0 or length > args.seq_len for length in lengths):
        raise ValueError("all --lengths entries must be in [0, seq_len]")
    if args.seq_len % args.chunk_size != 0:
        raise ValueError("this benchmark keeps seq_len divisible by chunk_size for stable shape comparisons")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 6)
    shape_k = (args.batch_size, args.num_heads, args.seq_len, args.key_dim)
    shape_v = (args.batch_size, args.num_heads, args.seq_len, args.value_dim)
    state_shape = (args.batch_size, args.num_heads, args.key_dim, args.value_dim)

    query = jax.random.normal(keys[0], shape_k, dtype=jnp.float32)
    key = jax.random.normal(keys[1], shape_k, dtype=jnp.float32)
    value = jax.random.normal(keys[2], shape_v, dtype=jnp.float32)
    g = jax.random.normal(keys[3], (args.batch_size, args.num_heads, args.seq_len), dtype=jnp.float32) * 0.1
    beta = jax.random.uniform(keys[4], (args.batch_size, args.num_heads, args.seq_len), dtype=jnp.float32)
    initial_state = jax.random.normal(keys[5], state_shape, dtype=jnp.float32) * 0.01

    valid = jnp.arange(args.seq_len, dtype=jnp.int32)[None, :] < jnp.asarray(lengths, dtype=jnp.int32)[:, None]
    query = jnp.where(valid[:, None, :, None], query, 0.0)
    key = jnp.where(valid[:, None, :, None], key, 0.0)
    value = jnp.where(valid[:, None, :, None], value, 0.0)
    g = jnp.where(valid[:, None, :], g, 0.0)
    beta = jnp.where(valid[:, None, :], beta, 0.0)
    return query, key, value, g, beta, initial_state, valid


def _padded_chunked_fn(chunk_size: int) -> Callable:
    def run(query, key, value, g, beta, initial_state):
        return jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    return run


def _variant_fns(args: argparse.Namespace, lengths: tuple[int, ...]) -> dict[str, Callable]:
    available = {
        "current_jax_chunk32_padded": _padded_chunked_fn(args.chunk_size),
    }
    requested = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(available)}")
    return {name: available[name] for name in requested}


def _compare_outputs(reference: tuple[jnp.ndarray, jnp.ndarray], candidate: tuple[jnp.ndarray, jnp.ndarray], valid: jnp.ndarray) -> dict[str, float]:
    ref_out, ref_state = reference
    out, state = candidate
    output_diff = (out.astype(jnp.float32) - ref_out.astype(jnp.float32))
    state_diff = state.astype(jnp.float32) - ref_state.astype(jnp.float32)
    valid_output_diff = jnp.where(valid[:, None, :, None], output_diff, 0.0)
    valid_count = max(int(valid.sum()) * int(out.shape[1]) * int(out.shape[-1]), 1)
    return {
        "output_max_abs": float(jnp.max(jnp.abs(output_diff))),
        "output_mse": float(jnp.mean(jnp.square(output_diff))),
        "valid_output_max_abs": float(jnp.max(jnp.abs(valid_output_diff))),
        "valid_output_mse": float(jnp.sum(jnp.square(valid_output_diff)) / valid_count),
        "state_max_abs": float(jnp.max(jnp.abs(state_diff))),
        "state_mse": float(jnp.mean(jnp.square(state_diff))),
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _device_summary() -> dict[str, Any]:
    devices = jax.devices()
    first = devices[0] if devices else None
    return {
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "device": str(first) if first is not None else None,
        "platform": getattr(first, "platform", None) if first is not None else None,
        "device_kind": getattr(first, "device_kind", None) if first is not None else None,
    }


def _profile_counters(profile_path: Path, profiled_iterations: int) -> dict[str, Any]:
    traces = sorted(profile_path.glob("plugins/profile/*/*.trace.json.gz"))
    if not traces:
        return {
            "trace_json_gz": None,
            "profiled_iterations": profiled_iterations,
            "ranges": {},
        }
    trace_path = traces[-1]
    needles = [
        "gdn_prefill/current_jax_chunk32_padded",
        "PjRtCApiLoadedExecutable::Execute",
        "jit_compiled:XLA GPU module",
        "command_buffer::execute",
        "command_buffer::update",
        "input_reduce_fusion",
        "loop_dynamic_update_slice_fusion",
        "loop_multiply_fusion",
        "MemcpyD2D",
        "Thunks::Initialize",
        "while",
        "fusion",
        "transpose",
        "gather",
    ]
    aggregate: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    try:
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            events = json.load(handle).get("traceEvents", [])
    except Exception as exc:
        return {
            "trace_json_gz": str(trace_path),
            "profiled_iterations": profiled_iterations,
            "error": f"{type(exc).__name__}: {exc}",
            "ranges": {},
        }
    for event in events:
        duration = event.get("dur")
        name = event.get("name", "")
        if duration is None:
            continue
        duration_ms = float(duration) / 1000.0
        for needle in needles:
            if needle in name:
                aggregate[needle][0] += duration_ms
                aggregate[needle][1] += 1
    denominator = max(int(profiled_iterations), 1)
    return {
        "trace_json_gz": str(trace_path),
        "profiled_iterations": profiled_iterations,
        "ranges": {
            needle: {
                "total_ms": total,
                "count": count,
                "ms_per_iter": total / denominator,
                "count_per_iter": count / denominator,
            }
            for needle, (total, count) in sorted(aggregate.items())
        },
    }


def run_benchmark(args: argparse.Namespace, recorder: RunRecorder) -> dict[str, Any]:
    lengths = _parse_ints(args.lengths)
    query, key, value, g, beta, initial_state, valid = _make_inputs(args, lengths)
    inputs = (query, key, value, g, beta, initial_state)
    variants = _variant_fns(args, lengths)
    true_tokens = int(sum(lengths))
    rectangular_tokens = int(args.batch_size * args.seq_len)
    active_chunks = int(sum((length + args.chunk_size - 1) // args.chunk_size for length in lengths))
    total_chunks = int(args.batch_size * (args.seq_len // args.chunk_size))

    variant_results: dict[str, dict[str, Any]] = {}
    outputs: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}

    # Compile before profiling. The trace should focus on warmed kernel behavior.
    compiled_variants: list[tuple[str, Callable, Any, float]] = []
    for name, fn in variants.items():
        started = time.perf_counter()
        compiled = jax.jit(fn).lower(*inputs).compile()
        compiled_variants.append((name, fn, compiled, time.perf_counter() - started))

    recorder.start_jax_profile(enabled=args.profile)
    try:
        for name, _fn, compiled, compile_seconds in compiled_variants:
            warmup_ms = []
            last_output = None
            for _ in range(args.warmups):
                with _trace_annotation(f"gdn_prefill/{name}/warmup"):
                    started = time.perf_counter()
                    last_output = _block_until_ready(compiled(*inputs))
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation(f"gdn_prefill/{name}/repeat"):
                    started = time.perf_counter()
                    last_output = _block_until_ready(compiled(*inputs))
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            outputs[name] = last_output
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            variant_results[name] = {
                "compile_seconds": compile_seconds,
                "warmup_ms": warmup_ms,
                "repeat_ms": repeat_ms,
                "mean_ms": mean_ms,
                "p50_ms": _percentile(repeat_ms, 50),
                "p95_ms": _percentile(repeat_ms, 95),
                "min_ms": min(repeat_ms) if repeat_ms else None,
                "max_ms": max(repeat_ms) if repeat_ms else None,
                "true_tokens_per_second": (1000.0 * true_tokens / mean_ms) if mean_ms else None,
                "rectangular_tokens_per_second": (1000.0 * rectangular_tokens / mean_ms) if mean_ms else None,
            }
    finally:
        recorder.stop_jax_profile()

    reference_name = "current_jax_chunk32_padded"
    if reference_name not in outputs:
        reference_name = next(iter(outputs))
    comparisons = {}
    for name, output in outputs.items():
        comparisons[f"{name}_vs_{reference_name}"] = _compare_outputs(outputs[reference_name], output, valid)
    profiled_iterations = len(variants) * (int(args.warmups) + int(args.repeats))

    return {
        "run_config": {
            "git_head": _git_head(),
            **_device_summary(),
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "seq_len": args.seq_len,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "chunk_size": args.chunk_size,
            "lengths": lengths,
            "true_tokens": true_tokens,
            "rectangular_tokens": rectangular_tokens,
            "active_chunks": active_chunks,
            "total_chunks": total_chunks,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "variants": list(variants),
            "dtype_contract": "fp32 activations/state, output cast follows current jax_chunk_gated_delta_rule",
        },
        "variants": variant_results,
        "comparisons": comparisons,
        "profile_counters": _profile_counters(recorder.profile_path, profiled_iterations) if args.profile else None,
        "pallas_feasibility": {
            "attempted": False,
            "lowering_ok": None,
            "error": None,
            "custom_call_count": None,
        },
        "historical_negative_controls": [
            "Entry053 static row-chunk ragged GDN prefill",
            "Entry056 static chunk-major GDN prefill",
        ],
        "decision": {
            "promote_to_server_routing": False,
            "reason": "baseline scaffold only; no backend-owned candidate was routed",
        },
        "run": recorder.metadata(),
    }


def main() -> None:
    args = parse_args()
    recorder = RunRecorder.create(
        script=Path(__file__).name,
        args=vars(args),
        run_label=args.run_label,
        profile_dir=args.profile_dir or None,
        run_log=args.run_log or None,
    )
    try:
        summary = run_benchmark(args, recorder)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
        recorder.finish(
            status="ok",
            summary={
                "variants": {
                    name: {
                        "mean_ms": result["mean_ms"],
                        "true_tokens_per_second": result["true_tokens_per_second"],
                    }
                    for name, result in summary["variants"].items()
                },
                "output_json": str(output_path),
            },
            learnings=[
                "Standalone GDN microbenchmarks are gates for backend kernels only; server routing still needs full hetero8 proof.",
            ],
            resolution="wrote benchmark summary",
        )
    except BaseException as exc:
        recorder.finish_exception(exc)
        raise


if __name__ == "__main__":
    main()
