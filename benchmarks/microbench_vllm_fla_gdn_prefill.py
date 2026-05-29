#!/usr/bin/env python3
"""Microbenchmark vLLM/FLA GDN prefill vs JAX reference and Triton paths."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    order = sorted(values)
    index = (len(order) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(order) - 1)
    if lo == hi:
        return order[lo]
    frac = index - lo
    return order[lo] * (1.0 - frac) + order[hi] * frac


def _stats_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def _collect_env() -> dict[str, Any]:
    out: dict[str, Any] = {
        "NANO_VLLM_JAX_CACHE_ROOT": os.getenv("NANO_VLLM_JAX_CACHE_ROOT"),
        "HF_HOME": os.getenv("HF_HOME"),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME"),
        "JAX_PLATFORMS": os.getenv("JAX_PLATFORMS"),
        "vllm_fla_path": "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages",
    }
    try:
        import torch  # type: ignore[import-not-found]

        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["torch_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        out["torch_cuda_available"] = False
        out["torch_error"] = repr(exc)

    try:
        import jax  # type: ignore[import-not-found]

        out["jax_platforms"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # pragma: no cover
        out["jax_platforms"] = []
        out["jax_error"] = repr(exc)
    return out


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.max(diff[finite]))


def _build_case(
    batch: int,
    seq_len: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    query_np = rng.normal(size=(batch * seq_len, num_heads, key_dim)).astype(np.float32)
    key_np = rng.normal(size=(batch * seq_len, num_heads, key_dim)).astype(np.float32)
    value_np = rng.normal(size=(batch * seq_len, num_heads, value_dim)).astype(np.float32)
    gate_np = rng.normal(size=(batch * seq_len, num_heads), scale=0.05).astype(np.float32)
    beta_np = rng.normal(size=(batch * seq_len, num_heads), scale=0.05).astype(np.float32)
    initial_state_np = np.zeros((batch, num_heads, value_dim, key_dim), dtype=np.float32)
    cu_np = (np.arange(batch + 1, dtype=np.int32) * seq_len).reshape(-1)
    return {
        "batch": batch,
        "seq_len": seq_len,
        "heads": num_heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunk_size": chunk_size,
        "tokens": batch * seq_len,
        "query_np": query_np,
        "key_np": key_np,
        "value_np": value_np,
        "gate_np": gate_np,
        "beta_np": beta_np,
        "initial_state_np": initial_state_np,
        "cu_np": cu_np,
    }


def _run_vllm(
    torch: Any,
    chunk_kernel: Any,
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    query_t = torch.tensor(case["query_np"], device="cuda", dtype=torch.bfloat16)
    key_t = torch.tensor(case["key_np"], device="cuda", dtype=torch.bfloat16)
    value_t = torch.tensor(case["value_np"], device="cuda", dtype=torch.bfloat16)
    gate_t = torch.tensor(case["gate_np"], device="cuda", dtype=torch.float32)
    beta_t = torch.tensor(case["beta_np"], device="cuda", dtype=torch.float32)
    init_t = torch.tensor(case["initial_state_np"], device="cuda", dtype=torch.float32)
    cu_t = torch.tensor(case["cu_np"], device="cuda", dtype=torch.int64)

    out_t, state_t = chunk_kernel(
        q=query_t[None],
        k=key_t[None],
        v=value_t[None],
        g=gate_t[None],
        beta=beta_t[None],
        initial_state=init_t,
        output_final_state=True,
        cu_seqlens=cu_t,
        scale=float(case["key_dim"] ** -0.5),
    )
    torch.cuda.synchronize()
    return (
        np.asarray(out_t[0].to(torch.float32).cpu().numpy()),
        np.asarray(state_t.to(torch.float32).cpu().numpy()),
    )


def _run_reference(jnp: Any, kernel: Any, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    out, state = kernel(
        jnp.asarray(case["query_np"]),
        jnp.asarray(case["key_np"]),
        jnp.asarray(case["value_np"]),
        jnp.asarray(case["gate_np"]),
        jnp.asarray(case["beta_np"]),
        jnp.asarray(case["cu_np"], dtype=jnp.int32),
        jnp.asarray(case["initial_state_np"]),
        chunk_size=case["chunk_size"],
    )
    return np.asarray(out), np.asarray(state)


def _run_timed_torch(fn, warmups: int, repeats: int, torch: Any) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return _stats_ms(timings)


def _run_timed_jax(fn, warmups: int, repeats: int) -> tuple[dict[str, float], Any]:
    last = None
    for _ in range(warmups):
        last = fn()
    _block_until_ready(last)

    timings: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        _block_until_ready(last)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return _stats_ms(timings), last


def _block_until_ready(value: Any) -> None:
    if value is None:
        return
    leaves = value if isinstance(value, tuple) else (value,)
    for leaf in leaves:
        ready = getattr(leaf, "block_until_ready", None)
        if callable(ready):
            ready()


def _run_case(
    case: dict[str, Any],
    *,
    warmups: int,
    repeats: int,
    include_triton: bool,
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]

    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_gated_delta_rule_packed_reference,
    )

    vllm_path = "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages"
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    vllm_chunk = importlib.import_module("vllm.model_executor.layers.fla.ops.chunk")
    chunk_kernel = getattr(vllm_chunk, "chunk_gated_delta_rule")

    # vLLM/FLA kernel timing + output
    vllm_stats = _run_timed_torch(
        lambda: _run_vllm(torch, chunk_kernel, case),
        warmups,
        repeats,
        torch,
    )
    vllm_out, vllm_state = _run_vllm(torch, chunk_kernel, case)

    # JAX packed reference timing + output
    jax_ref_fn = lambda: _run_reference(
        jnp,
        gdn_fla_chunk_gated_delta_rule_packed_reference,
        case,
    )
    ref_stats, ref_last = _run_timed_jax(jax_ref_fn, warmups, repeats)
    ref_out, ref_state = ref_last
    ref_out = np.asarray(ref_out)
    ref_state = np.asarray(ref_state)

    result: dict[str, Any] = {
        "shape": {
            "batch": case["batch"],
            "seq_len": case["seq_len"],
            "heads": case["heads"],
            "key_dim": case["key_dim"],
            "value_dim": case["value_dim"],
            "chunk_size": case["chunk_size"],
            "tokens": case["tokens"],
        },
        "vllm": {
            "timing_ms": vllm_stats,
            "tokens_per_s": case["tokens"] * 1000.0 / max(vllm_stats["mean_ms"], 1e-9),
        },
        "reference": {
            "timing_ms": ref_stats,
            "tokens_per_s": case["tokens"] * 1000.0 / max(ref_stats["mean_ms"], 1e-9),
            "max_abs_diff_vs_vllm_out": _max_abs_diff(ref_out, vllm_out),
            "max_abs_diff_vs_vllm_state": _max_abs_diff(ref_state, vllm_state),
        },
    }

    if include_triton:
        try:
            from nanovllm_jax.kernels.gdn_fla_triton import (
                gdn_fla_chunk_gated_delta_rule_packed_triton,
            )

            triton_fn = lambda: _run_reference(
                jnp,
                gdn_fla_chunk_gated_delta_rule_packed_triton,
                case,
            )
            triton_stats, triton_last = _run_timed_jax(triton_fn, warmups, repeats)
            triton_out = np.asarray(triton_last[0])
            triton_state = np.asarray(triton_last[1])
            result["triton"] = {
                "timing_ms": triton_stats,
                "tokens_per_s": case["tokens"] * 1000.0 / max(triton_stats["mean_ms"], 1e-9),
                "max_abs_diff_vs_vllm_out": _max_abs_diff(triton_out, vllm_out),
                "max_abs_diff_vs_vllm_state": _max_abs_diff(triton_state, vllm_state),
            }
        except Exception as exc:  # pragma: no cover
            result["triton"] = {"status": "unavailable", "error": repr(exc)}

    return result


def _parse_shapes(value: str) -> list[tuple[int, int, int, int, int]]:
    shapes = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 5:
            raise argparse.ArgumentTypeError("each shape must be B,T,H,D,V")
        shapes.append(tuple(int(part) for part in parts))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--shapes",
        default="1,512,16,128,128;4,512,16,128,128;4,2048,16,128,128",
        type=_parse_shapes,
    )
    parser.add_argument("--include-triton", action="store_true")
    parser.add_argument(
        "--output-json",
        default="results/vllm_fla_gdn_microbench_20260528.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload: dict[str, Any] = {
        "env": _collect_env(),
        "shapes": [
            dict(B=b, T=t, H=h, D=d, V=v, chunk_size=args.chunk_size) for b, t, h, d, v in args.shapes
        ],
        "cases": [],
        "command": "microbench_vllm_fla_gdn_prefill.py",
    }

    payload_cases = []
    for idx, (batch, seq_len, num_heads, key_dim, value_dim) in enumerate(args.shapes):
        case = _build_case(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            chunk_size=args.chunk_size,
            seed=args.seed + idx,
        )
        payload_cases.append(
            _run_case(
                case,
                warmups=args.warmups,
                repeats=args.repeats,
                include_triton=args.include_triton,
            )
        )
    payload["cases"] = payload_cases

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["output_json"] = str(out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
