#!/usr/bin/env python3
"""Measure vLLM's vendored FLA GDN kernels on this host.

This is a porting probe, not a serving benchmark. It runs the Torch/Triton
kernels that vLLM uses for Qwen3.5/Qwen3-Next GDN prefill and packed decode so
we can decide whether their schedule is worth porting behind the repo's JAX
GDN ABI.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_paths import configure_flashinfer_cache, default_runtime_root


def _configure_runtime_paths() -> dict[str, str]:
    root = default_runtime_root()
    env_defaults = {
        "TMPDIR": root / "tmp",
        "XDG_CACHE_HOME": root / ".cache",
        "TRITON_CACHE_DIR": root / ".cache" / "triton",
        "TORCHINDUCTOR_CACHE_DIR": root / ".cache" / "torchinductor",
        "VLLM_CACHE_ROOT": root / ".cache" / "vllm",
        "VLLM_RPC_BASE_PATH": root / "tmp" / "vllm-rpc",
    }
    for key, path in env_defaults.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    configure_flashinfer_cache()
    return {
        key: os.environ[key]
        for key in (
            "TMPDIR",
            "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "VLLM_CACHE_ROOT",
            "VLLM_RPC_BASE_PATH",
            "FLASHINFER_WORKSPACE_BASE",
            "FLASHINFER_CUBIN_DIR",
        )
        if key in os.environ
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _parse_int_list(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def _time_cuda_ms(
    torch: Any,
    fn: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float | int], Any]:
    last = None
    with torch.no_grad():
        for _ in range(warmups):
            last = fn()
        torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            last = fn()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return _stats_ms(timings), last


def _max_diff(actual: Any, expected: Any) -> dict[str, float]:
    import torch

    diff = actual.to(torch.float32) - expected.to(torch.float32)
    return {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0,
    }


def _torch_recurrent_gdn_reference(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    initial_state: Any,
    *,
    cu_seqlens: Any | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    output_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Independent recurrent reference for vLLM's FLA GDN contract."""

    if q.ndim != 4:
        raise ValueError("q/k/v must have shape [B,T,H,D]")
    if scale is None:
        scale = q.shape[-1] ** -0.5

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    g_f = g.to(torch.float32)
    beta_f = beta.to(torch.float32)
    if use_qk_l2norm_in_kernel:
        q_f = q_f / torch.sqrt(torch.sum(q_f * q_f, dim=-1, keepdim=True) + 1e-6)
        k_f = k_f / torch.sqrt(torch.sum(k_f * k_f, dim=-1, keepdim=True) + 1e-6)
    q_f = q_f * scale

    if cu_seqlens is None:
        lengths = [int(q.shape[1])] * int(q.shape[0])
        starts = [row * int(q.shape[1]) for row in range(int(q.shape[0]))]
    else:
        offsets = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
        lengths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        starts = offsets[:-1]
        if q.shape[0] != 1:
            raise ValueError("varlen reference expects q batch dimension 1")

    output = torch.zeros_like(v_f)
    final_state = initial_state.to(torch.float32).clone()
    for row, row_len in enumerate(lengths):
        state = final_state[row]
        for local_t in range(row_len):
            token = starts[row] + local_t if cu_seqlens is not None else local_t
            batch_index = 0 if cu_seqlens is not None else row
            q_t = q_f[batch_index, token]
            k_t = k_f[batch_index, token]
            v_t = v_f[batch_index, token]
            g_t = torch.exp(g_f[batch_index, token])[:, None, None]
            beta_t = beta_f[batch_index, token][:, None]

            state = state * g_t
            kv_mem = torch.einsum("hvk,hk->hv", state, k_t)
            delta = (v_t - kv_mem) * beta_t
            state = state + delta[:, :, None] * k_t[:, None, :]
            out_t = torch.einsum("hvk,hk->hv", state, q_t)
            output[batch_index, token] = out_t
        final_state[row] = state

    if output_dtype is not None:
        output = output.to(output_dtype)
    return output, final_state


def _torch_device_report(torch: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        props = torch.cuda.get_device_properties(index)
        report.update(
            {
                "current_device": int(index),
                "name": torch.cuda.get_device_name(index),
                "compute_capability": [int(major), int(minor)],
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return report


def _run_reference_checks(
    torch: Any,
    fused_post_conv_prep: Callable[..., Any],
    chunk_gated_delta_rule: Callable[..., Any],
    fused_recurrent_gated_delta_rule_packed_decode: Callable[..., Any],
) -> dict[str, Any]:
    h = 16
    hv = 16
    k_dim = 128
    v_dim = 128
    qkv_dim = 2 * h * k_dim + hv * v_dim
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device)
    generator.manual_seed(9876)

    # Use uneven rows and a >64-token row to exercise varlen chunking without
    # making the Python reference dominate the probe runtime.
    lengths = [17, 64, 65]
    total_tokens = sum(lengths)
    conv_output = torch.randn(
        total_tokens,
        qkv_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    a = torch.randn(total_tokens, hv, device=device, dtype=dtype, generator=generator)
    b = torch.randn(total_tokens, hv, device=device, dtype=dtype, generator=generator)
    a_log = torch.full((hv,), -3.0, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(hv, device=device, dtype=torch.float32)
    initial_state = torch.randn(
        len(lengths),
        hv,
        v_dim,
        k_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 0.01
    cu_seqlens = torch.tensor(
        [0, *[sum(lengths[: i + 1]) for i in range(len(lengths))]],
        device=device,
        dtype=torch.int32,
    )

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_output,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        num_k_heads=h,
        head_k_dim=k_dim,
        head_v_dim=v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )
    q_4d = q.unsqueeze(0)
    k_4d = k.unsqueeze(0)
    v_4d = v.unsqueeze(0)
    g_3d = g.unsqueeze(0)
    beta_3d = beta.unsqueeze(0)
    fla_output, fla_state = chunk_gated_delta_rule(
        q=q_4d,
        k=k_4d,
        v=v_4d,
        g=g_3d,
        beta=beta_3d,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    ref_output, ref_state = _torch_recurrent_gdn_reference(
        torch,
        q_4d,
        k_4d,
        v_4d,
        g_3d,
        beta_3d,
        initial_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output_dtype=fla_output.dtype,
    )

    batch = 4
    mixed_qkv = torch.randn(batch, qkv_dim, device=device, dtype=dtype, generator=generator)
    a_decode = torch.randn(batch, hv, device=device, dtype=dtype, generator=generator)
    b_decode = torch.randn(batch, hv, device=device, dtype=dtype, generator=generator)
    decode_state = torch.randn(
        batch + 1,
        hv,
        v_dim,
        k_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 0.01
    decode_out = torch.empty(batch, 1, hv, v_dim, device=device, dtype=dtype)
    state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    packed_out, packed_state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a_decode,
        b=b_decode,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=k_dim**-0.5,
        initial_state=decode_state.clone(),
        out=decode_out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    q_flat, k_flat, v_flat = torch.split(
        mixed_qkv,
        [h * k_dim, h * k_dim, hv * v_dim],
        dim=-1,
    )
    q_decode = q_flat.reshape(batch, 1, h, k_dim)
    k_decode = k_flat.reshape(batch, 1, h, k_dim)
    v_decode = v_flat.reshape(batch, 1, hv, v_dim)
    x = a_decode.to(torch.float32) + dt_bias[None, :]
    softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g_decode = (-torch.exp(a_log)[None, :] * softplus_x).reshape(batch, 1, hv)
    beta_decode = torch.sigmoid(b_decode.to(torch.float32)).to(dtype).to(torch.float32)
    ref_decode_out, ref_decode_state = _torch_recurrent_gdn_reference(
        torch,
        q_decode,
        k_decode,
        v_decode,
        g_decode,
        beta_decode.reshape(batch, 1, hv),
        decode_state[1:].clone(),
        use_qk_l2norm_in_kernel=True,
        output_dtype=packed_out.dtype,
    )

    return {
        "prefill": {
            "lengths": lengths,
            "output": _max_diff(fla_output, ref_output),
            "final_state": _max_diff(fla_state, ref_state),
            "output_shape": list(fla_output.shape),
            "state_shape": list(fla_state.shape),
        },
        "packed_decode": {
            "batch": batch,
            "output": _max_diff(packed_out, ref_decode_out),
            "state": _max_diff(packed_state[1:], ref_decode_state),
            "output_shape": list(packed_out.shape),
            "state_shape": list(packed_state.shape),
        },
    }


def _run_prefill_probe(
    torch: Any,
    fused_post_conv_prep: Callable[..., Any],
    chunk_gated_delta_rule: Callable[..., Any],
    *,
    lengths: list[int],
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    h = 16
    hv = 16
    k_dim = 128
    v_dim = 128
    qkv_dim = 2 * h * k_dim + hv * v_dim
    total_tokens = sum(lengths)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    conv_output = torch.randn(
        total_tokens,
        qkv_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    a = torch.randn(total_tokens, hv, device=device, dtype=dtype, generator=generator)
    b = torch.randn(total_tokens, hv, device=device, dtype=dtype, generator=generator)
    # Keep decay small enough that exp(g) does not aggressively underflow over long rows.
    a_log = torch.full((hv,), -3.0, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(hv, device=device, dtype=torch.float32)
    initial_state = torch.zeros(len(lengths), hv, v_dim, k_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor(
        [0, *[sum(lengths[: i + 1]) for i in range(len(lengths))]],
        device=device,
        dtype=torch.int32,
    )

    def prep_only():
        return fused_post_conv_prep(
            conv_output=conv_output,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=dt_bias,
            num_k_heads=h,
            head_k_dim=k_dim,
            head_v_dim=v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )

    q, k, v, g, beta = prep_only()
    q_4d = q.unsqueeze(0)
    k_4d = k.unsqueeze(0)
    v_4d = v.unsqueeze(0)
    g_3d = g.unsqueeze(0)
    beta_3d = beta.unsqueeze(0)

    def chunk_only():
        return chunk_gated_delta_rule(
            q=q_4d,
            k=k_4d,
            v=v_4d,
            g=g_3d,
            beta=beta_3d,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )

    def prep_and_chunk():
        q_i, k_i, v_i, g_i, beta_i = prep_only()
        return chunk_gated_delta_rule(
            q=q_i.unsqueeze(0),
            k=k_i.unsqueeze(0),
            v=v_i.unsqueeze(0),
            g=g_i.unsqueeze(0),
            beta=beta_i.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )

    prep_stats, prep_result = _time_cuda_ms(
        torch,
        prep_only,
        warmups=warmups,
        repeats=repeats,
    )
    chunk_stats, chunk_result = _time_cuda_ms(
        torch,
        chunk_only,
        warmups=warmups,
        repeats=repeats,
    )
    combined_stats, combined_result = _time_cuda_ms(
        torch,
        prep_and_chunk,
        warmups=warmups,
        repeats=repeats,
    )
    output, final_state = combined_result
    return {
        "kind": "prefill",
        "shape": {
            "lengths": lengths,
            "total_tokens": total_tokens,
            "qkv_dim": qkv_dim,
            "num_key_heads": h,
            "num_value_heads": hv,
            "key_dim": k_dim,
            "value_dim": v_dim,
            "qkv_dtype": str(dtype),
            "gate_beta_dtype": "torch.float32",
            "state_dtype": str(initial_state.dtype),
            "state_layout": "[N,HV,V,K]",
            "cu_seqlens_dtype": str(cu_seqlens.dtype),
        },
        "prep_only": prep_stats,
        "chunk_only": chunk_stats,
        "prep_and_chunk": combined_stats,
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "final_state_shape": list(final_state.shape),
            "final_state_dtype": str(final_state.dtype),
            "prep_q_dtype": str(prep_result[0].dtype),
            "prep_gate_dtype": str(prep_result[3].dtype),
            "chunk_output_dtype": str(chunk_result[0].dtype),
        },
    }


def _run_decode_probe(
    torch: Any,
    fused_recurrent_gated_delta_rule_packed_decode: Callable[..., Any],
    *,
    batches: list[int],
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    h = 16
    hv = 16
    k_dim = 128
    v_dim = 128
    qkv_dim = 2 * h * k_dim + hv * v_dim
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = []
    for batch in batches:
        generator = torch.Generator(device=device)
        generator.manual_seed(4321 + batch)
        mixed_qkv = torch.randn(batch, qkv_dim, device=device, dtype=dtype, generator=generator)
        a = torch.randn(batch, hv, device=device, dtype=dtype, generator=generator)
        b = torch.randn(batch, hv, device=device, dtype=dtype, generator=generator)
        a_log = torch.full((hv,), -3.0, device=device, dtype=torch.float32)
        dt_bias = torch.zeros(hv, device=device, dtype=torch.float32)
        # vLLM state index 0 is a null block, so use 1..B for active rows.
        state = torch.zeros(batch + 1, hv, v_dim, k_dim, device=device, dtype=torch.float32)
        out = torch.empty(batch, 1, hv, v_dim, device=device, dtype=dtype)
        state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)

        def decode_once():
            return fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=a_log,
                dt_bias=dt_bias,
                scale=k_dim**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )

        stats, result = _time_cuda_ms(
            torch,
            decode_once,
            warmups=warmups,
            repeats=repeats,
        )
        decoded, new_state = result
        rows.append(
            {
                "kind": "packed_decode",
                "shape": {
                    "batch": batch,
                    "qkv_dim": qkv_dim,
                    "num_key_heads": h,
                    "num_value_heads": hv,
                    "key_dim": k_dim,
                    "value_dim": v_dim,
                    "qkv_dtype": str(dtype),
                    "state_dtype": str(state.dtype),
                    "state_layout": "[slots,HV,V,K]",
                },
                "timing": stats,
                "output": {
                    "shape": list(decoded.shape),
                    "dtype": str(decoded.dtype),
                    "state_shape": list(new_state.shape),
                    "state_dtype": str(new_state.dtype),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default="",
        help="Output path. Defaults to results/vllm_fla_gdn_probe_<timestamp>.json.",
    )
    parser.add_argument(
        "--prefill-lengths",
        type=_parse_int_list,
        default=_parse_int_list("512,1024,1536,2048"),
        help="Comma-separated varlen prefill rows.",
    )
    parser.add_argument(
        "--decode-batches",
        type=_parse_int_list,
        default=_parse_int_list("1,4,8,16"),
        help="Comma-separated packed-decode batch sizes.",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Skip independent Torch recurrent checks against vLLM/FLA kernels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_paths = _configure_runtime_paths()

    import torch
    import triton
    import vllm
    from vllm.model_executor.layers.fla.ops.chunk import (
        chunk_gated_delta_rule,
    )
    from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda is not available; run this probe with GPU access")

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    results: dict[str, Any] = {
        "created_at_unix": time.time(),
        "purpose": "Torch-side vLLM/FLA GDN kernel porting probe, not a JAX serving speed claim.",
        "runtime_paths": env_paths,
        "device": _torch_device_report(torch),
        "versions": {
            "vllm": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "triton": getattr(triton, "__version__", None),
        },
        "vllm_fla_contract": {
            "prefill": (
                "fused_post_conv_prep(conv_output [L,2*H*K+HV*V] bf16, a/b [L,HV]) "
                "-> q/k/v [L,H,D] bf16, g/beta [L,HV] fp32; "
                "chunk_gated_delta_rule consumes [1,L,H,D] with cu_seqlens int32."
            ),
            "decode": (
                "fused_recurrent_gated_delta_rule_packed_decode consumes mixed_qkv "
                "[B,2*H*K+HV*V] bf16, a/b [B,HV], state [slots,HV,V,K] fp32."
            ),
            "fla_chunk_size": FLA_CHUNK_SIZE,
        },
        "prefill": None,
        "decode": [],
        "reference_checks": None,
        "errors": [],
    }

    if not args.skip_reference_check:
        try:
            results["reference_checks"] = _run_reference_checks(
                torch,
                fused_post_conv_prep,
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule_packed_decode,
            )
        except Exception as exc:
            results["errors"].append({"stage": "reference_checks", "error": repr(exc)})

    if not args.skip_prefill:
        try:
            results["prefill"] = _run_prefill_probe(
                torch,
                fused_post_conv_prep,
                chunk_gated_delta_rule,
                lengths=args.prefill_lengths,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        except Exception as exc:
            results["errors"].append({"stage": "prefill", "error": repr(exc)})

    if not args.skip_decode:
        try:
            results["decode"] = _run_decode_probe(
                torch,
                fused_recurrent_gated_delta_rule_packed_decode,
                batches=args.decode_batches,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        except Exception as exc:
            results["errors"].append({"stage": "decode", "error": repr(exc)})

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"vllm_fla_gdn_probe_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(results), indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
