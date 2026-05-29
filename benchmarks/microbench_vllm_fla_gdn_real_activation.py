#!/usr/bin/env python3
"""Real-activation vLLM/FLA GDN prefill probe against JAX references."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _collect_environment() -> dict[str, Any]:
    env = {
        "NANO_VLLM_JAX_CACHE_ROOT": os.getenv("NANO_VLLM_JAX_CACHE_ROOT"),
        "HF_HOME": os.getenv("HF_HOME"),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME"),
        "JAX_PLATFORMS": os.getenv("JAX_PLATFORMS"),
    }
    try:
        import jax  # type: ignore[import-not-found]

        env["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover
        env["jax_devices_error"] = repr(exc)

    try:
        import torch  # type: ignore[import-not-found]

        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["torch_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        env["torch_error"] = repr(exc)

    return env


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    frac = index - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _stats_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def _run_blocked(value: Any) -> None:
    ready = getattr(value, "block_until_ready", None)
    if callable(ready):
        ready()


def _run_timed_jax(fn, warmups: int, repeats: int) -> tuple[dict[str, float], Any]:
    last = None
    for _ in range(warmups):
        last = fn()
    if last is not None:
        _run_blocked(last[0] if isinstance(last, tuple) else last)
    timings: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        _run_blocked(last[0] if isinstance(last, tuple) else last)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return _stats_ms(timings), last


def _run_timed_torch(fn, warmups: int, repeats: int) -> tuple[dict[str, float], Any]:
    import torch

    last = None
    for _ in range(warmups):
        last = fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return _stats_ms(timings), last


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.max(diff[finite]))


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.mean(diff[finite]))


def _rms_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(diff[finite]))))


def _to_str_lengths(length: int) -> str:
    return f"{length}tok"


def _ensure_prompt_tokens(tokenizer, text: str, target_tokens: int) -> np.ndarray:
    token_ids: list[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or 1]
    if len(token_ids) < target_tokens:
        repeat = (target_tokens + len(token_ids) - 1) // len(token_ids)
        token_ids = token_ids * repeat
    return np.array(token_ids[:target_tokens], dtype=np.int32)[None, :]


@dataclass
class CaseData:
    token_len: int
    tokens: np.ndarray
    layer_idx: int
    query: Any
    key: Any
    value: Any
    gate: Any
    beta: Any
    initial_state: Any
    cu_seqlens: Any
    seq_lens: Any


def _build_real_case(
    params: Any,
    config: Any,
    tokenizer,
    layer_idx: int,
    prompt: str,
    token_len: int,
    *,
    qkv_dtype: str,
) -> CaseData:
    import jax.numpy as jnp
    import nanovllm_jax.model as model
    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_post_conv_prefill_fla_inputs_from_decay

    tokens = _ensure_prompt_tokens(tokenizer, prompt, token_len)
    layer_params = params.layers[layer_idx]

    x = params.embed_tokens[tokens]
    if x.dtype != config.get_dtype():
        x = x.astype(config.get_dtype())

    normed = model._decode_width1_rms_norm(
        x,
        layer_params["input_norm"],
        config.rms_norm_eps,
        force_width1=False,
    )
    mixed_qkv = model._compact_prefill_tokenwise_dot(
        normed,
        layer_params["in_proj_qkv"],
        valid_token_mask=None,
        compact_num_tokens=None,
    )
    a = model._tokenwise_decode_dot(
        normed, layer_params["in_proj_a"], force_width1=False
    ).reshape(*tokens.shape, -1)
    b = model._tokenwise_decode_dot(
        normed, layer_params["in_proj_b"], force_width1=False
    ).reshape(*tokens.shape, -1)

    mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)
    conv_out = model.causal_conv1d_metal(
        mixed_qkv_t,
        layer_params["conv1d_weight"],
        layer_params.get("conv1d_bias"),
        "silu",
    ).transpose(0, 2, 1)

    query, key, value, gate, beta, seq_lens = prepare_gdn_post_conv_prefill_fla_inputs_from_decay(
        conv_out,
        a,
        b,
        layer_params["A"],
        layer_params["dt_bias"],
        valid_token_mask=None,
        num_key_heads=config.linear_num_key_heads,
        num_value_heads=config.linear_num_value_heads,
        key_head_dim=config.linear_key_head_dim,
        value_head_dim=config.linear_value_head_dim,
        normalize_qk=False,
    )
    batch, seq_len = tokens.shape
    initial_state = jnp.zeros(
        (
            batch,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        ),
        dtype=jnp.float32,
    )
    cu_seqlens = np.array([0, seq_len], dtype=np.int32)
    return CaseData(
        token_len=int(seq_len),
        tokens=tokens,
        layer_idx=layer_idx,
        query=query.astype(np.dtype(qkv_dtype)),
        key=key.astype(np.dtype(qkv_dtype)),
        value=value.astype(np.dtype(qkv_dtype)),
        gate=np.asarray(gate.astype(jnp.float32)),
        beta=np.asarray(beta.astype(jnp.float32)),
        initial_state=np.asarray(initial_state),
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
    )


def _run_vllm_case(case: CaseData) -> tuple[np.ndarray, np.ndarray]:
    import torch

    sys.path.insert(0, "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages")
    vllm_chunk = __import__("vllm.model_executor.layers.fla.ops.chunk", fromlist=["chunk_gated_delta_rule"])
    chunk_kernel = getattr(vllm_chunk, "chunk_gated_delta_rule")

    q_t = torch.tensor(case.query, device="cuda", dtype=torch.bfloat16)
    k_t = torch.tensor(case.key, device="cuda", dtype=torch.bfloat16)
    v_t = torch.tensor(case.value, device="cuda", dtype=torch.bfloat16)
    g_t = torch.tensor(case.gate, device="cuda", dtype=torch.float32)
    beta_t = torch.tensor(case.beta, device="cuda", dtype=torch.float32)
    init_t = torch.tensor(case.initial_state, device="cuda", dtype=torch.float32)
    cu_t = torch.tensor(case.cu_seqlens, device="cuda", dtype=torch.int64)

    out, state = chunk_kernel(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        initial_state=init_t,
        output_final_state=True,
        cu_seqlens=cu_t,
        scale=case.query.shape[-1] ** -0.5,
    )
    torch.cuda.synchronize()
    return (
        np.asarray(out[0].to(torch.float32).cpu().numpy()),
        np.asarray(state.to(torch.float32).cpu().numpy()),
    )


def _run_jax_reference(case: CaseData, *, use_triton: bool) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_gated_delta_rule_packed_reference,
        pack_prepared_gdn_prefill_inputs,
        prepare_gdn_fla_prefill_kernel_inputs,
        unpack_prepared_gdn_prefill_output,
    )
    from nanovllm_jax.backends import _gdn_prefill_qkv_activation_jnp_dtype

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        jnp.asarray(case.query),
        jnp.asarray(case.key),
        jnp.asarray(case.value),
        jnp.asarray(case.gate),
        jnp.asarray(case.beta),
        jnp.asarray(case.seq_lens, dtype=jnp.int32),
        jnp.asarray(case.initial_state, dtype=jnp.float32),
        qkv_dtype=_gdn_prefill_qkv_activation_jnp_dtype(),
    )
    packed_query, packed_key, packed_value, packed_gate, packed_beta, cu_seqlens = (
        pack_prepared_gdn_prefill_inputs(
            prepared.query,
            prepared.key,
            prepared.value,
            prepared.gate,
            prepared.beta,
            prepared.seq_lens,
        )
    )
    output_shape = tuple(prepared.query.shape)

    if use_triton:
        from nanovllm_jax.kernels.gdn_fla_triton import (
            gdn_fla_chunk_gated_delta_rule_packed_triton,
        )
        out, state = gdn_fla_chunk_gated_delta_rule_packed_triton(
            packed_query,
            packed_key,
            packed_value,
            packed_gate,
            packed_beta,
            cu_seqlens,
            prepared.initial_state,
            chunk_size=64,
            use_qk_l2norm_in_kernel=False,
        )
        out = unpack_prepared_gdn_prefill_output(
            out,
            cu_seqlens,
            output_shape[1],
        )
        return np.asarray(out), np.asarray(state)

    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_gated_delta_rule_packed_reference,
    )

    out, state = gdn_fla_chunk_gated_delta_rule_packed_reference(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        packed_beta,
        jnp.asarray(cu_seqlens, dtype=jnp.int32),
        prepared.initial_state,
        chunk_size=64,
    )
    out = unpack_prepared_gdn_prefill_output(
        out,
        cu_seqlens,
        output_shape[1],
    )
    return np.asarray(out), np.asarray(state)


def _run_case(name: str, case: CaseData, *, repeats: int, warmups: int, run_triton: bool) -> dict[str, Any]:
    vllm_stats, (vllm_out, vllm_state) = _run_timed_torch(
        lambda: _run_vllm_case(case),
        warmups=warmups,
        repeats=repeats,
    )
    ref_stats, (ref_out, ref_state) = _run_timed_jax(
        lambda: _run_jax_reference(case, use_triton=False),
        warmups=warmups,
        repeats=repeats,
    )
    result: dict[str, Any] = {
        "name": name,
        "token_len": case.token_len,
        "layer_idx": case.layer_idx,
        "vllm": {
            "timing_ms": vllm_stats,
            "tokens_per_s": case.token_len * 1000.0 / max(vllm_stats["mean_ms"], 1e-9),
        },
        "jax_reference": {
            "timing_ms": ref_stats,
            "tokens_per_s": case.token_len * 1000.0 / max(ref_stats["mean_ms"], 1e-9),
            "max_abs_diff_vs_vllm_out": _max_abs_diff(ref_out, vllm_out),
            "mean_abs_diff_vs_vllm_out": _mean_abs_diff(ref_out, vllm_out),
            "rms_diff_vs_vllm_out": _rms_abs_diff(ref_out, vllm_out),
            "max_abs_diff_vs_vllm_state": _max_abs_diff(ref_state, vllm_state),
            "mean_abs_diff_vs_vllm_state": _mean_abs_diff(ref_state, vllm_state),
            "rms_diff_vs_vllm_state": _rms_abs_diff(ref_state, vllm_state),
        },
    }

    if run_triton:
        try:
            triton_stats, (triton_out, triton_state) = _run_timed_jax(
                lambda: _run_jax_reference(case, use_triton=True),
                warmups=warmups,
                repeats=repeats,
            )
            result["jax_triton"] = {
                "status": "ok",
                "timing_ms": triton_stats,
                "tokens_per_s": case.token_len * 1000.0 / max(
                    triton_stats["mean_ms"], 1e-9
                ),
                "max_abs_diff_vs_vllm_out": _max_abs_diff(triton_out, vllm_out),
                "mean_abs_diff_vs_vllm_out": _mean_abs_diff(triton_out, vllm_out),
                "rms_diff_vs_vllm_out": _rms_abs_diff(triton_out, vllm_out),
                "max_abs_diff_vs_vllm_state": _max_abs_diff(triton_state, vllm_state),
                "mean_abs_diff_vs_vllm_state": _mean_abs_diff(triton_state, vllm_state),
                "rms_diff_vs_vllm_state": _rms_abs_diff(triton_state, vllm_state),
            }
        except Exception as exc:
            result["jax_triton"] = {"status": "unavailable", "error": repr(exc)}
    return result


def _parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected one or more positive ints")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="HF model id for real weights.",
    )
    parser.add_argument(
        "--model-dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Weight/dtype for model parameter loading and activation math.",
    )
    parser.add_argument("--layer-idx", type=int, default=0, help="Target GDN layer index.")
    parser.add_argument(
        "--token-lengths",
        type=_parse_int_list,
        default="64,512",
        help="Comma-separated token lengths to probe.",
    )
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-triton", action="store_true")
    parser.add_argument(
        "--output-json",
        default="results/vllm_fla_real_activation_probe_20260528.json",
    )
    return parser.parse_args()


def _configure_runtime_env() -> None:
    os.environ.setdefault("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    os.environ.setdefault("HF_HOME", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/mountpoint/.exp/.cache")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/mountpoint/.exp/.cache/flashinfer")


def main() -> int:
    _configure_runtime_env()
    args = parse_args()

    import jax
    from nanovllm_jax.config import Qwen3_5Config
    from nanovllm_jax.load_weights import load_weights_from_hf
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    if args.layer_idx < 0:
        raise ValueError("layer-idx must be >= 0")
    config = Qwen3_5Config.qwen3_5_0_8b()
    config.dtype = args.model_dtype

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    params = load_weights_from_hf(args.model, config, verbose=False)
    if args.layer_idx >= len(params.layers):
        raise ValueError(f"layer_idx {args.layer_idx} is out of range for {len(params.layers)} layers")
    qkv_dtype = args.model_dtype if args.model_dtype != "float32" else "float32"

    cases: list[dict[str, Any]] = []
    for length in args.token_lengths:
        case = _build_real_case(
            params=params,
            config=config,
            tokenizer=tokenizer,
            layer_idx=args.layer_idx,
            prompt=args.prompt,
            token_len=length,
            qkv_dtype=qkv_dtype,
        )
        cases.append(_run_case(
            _to_str_lengths(length),
            case,
            warmups=args.warmups,
            repeats=args.repeats,
            run_triton=args.run_triton,
        ))

    payload: dict[str, Any] = {
        "environment": _collect_environment(),
        "model": args.model,
        "config_dtype": args.model_dtype,
        "layer_idx": args.layer_idx,
        "token_lengths": args.token_lengths,
        "run_triton": args.run_triton,
        "command": "microbench_vllm_fla_gdn_real_activation.py",
        "cases": cases,
        "result": {
            "jnp_backend": str(jax.devices()[0]) if jax.devices() else "unavailable",
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
