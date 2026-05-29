#!/usr/bin/env python3
"""Audit first divergence point between vLLM FLA GDN and JAX reference."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.microbench_vllm_fla_gdn_real_activation import _build_real_case


LENGTHS = [1, 2, 8, 32, 63, 64, 65, 96, 128, 256, 512]
TRANSFORM_LENGTHS = {64, 65, 128}
OUT_PATH = Path("results/vllm_fla_real_activation_contract_audit_20260528.json")
SUMMARY_PATH = Path("results/vllm_fla_real_activation_contract_audit_20260528.md")


def _configure_runtime_env() -> None:
    os.environ.setdefault("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    os.environ.setdefault("HF_HOME", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/mountpoint/.exp/.cache")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/mountpoint/.exp/.cache/flashinfer")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    if not finite.any():
        return {"max_abs": float("nan"), "mean_abs": float("nan")}
    return {
        "max_abs": float(np.max(diff[finite])),
        "mean_abs": float(np.mean(diff[finite])),
    }


def _to_torch(x: np.ndarray, torch, *, dtype, device: str = "cuda"):
    if dtype == torch.bfloat16:
        x = np.asarray(x, dtype=np.float32)
    return torch.tensor(x, device=device, dtype=dtype)


def _run_vllm(
    *,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gate: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    cu_seqlens: np.ndarray,
    provide_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    q_t = _to_torch(query, torch, dtype=torch.bfloat16)
    k_t = _to_torch(key, torch, dtype=torch.bfloat16)
    v_t = _to_torch(value, torch, dtype=torch.bfloat16)
    g_t = _to_torch(gate, torch, dtype=torch.float32)
    beta_t = _to_torch(beta, torch, dtype=torch.float32)
    init_t = _to_torch(initial_state, torch, dtype=torch.float32)
    cu_t = _to_torch(cu_seqlens, torch, dtype=torch.int64)

    kwargs = {
        "q": q_t,
        "k": k_t,
        "v": v_t,
        "g": g_t,
        "beta": beta_t,
        "initial_state": init_t,
        "output_final_state": True,
        "cu_seqlens": cu_t,
    }
    if provide_scale:
        kwargs["scale"] = float(query.shape[-1] ** -0.5)
    out_t, state_t = _VLLM_CHUNK_KERNEL(**kwargs)
    torch.cuda.synchronize()
    return (
        np.asarray(out_t.to(torch.float32).cpu().numpy()),
        np.asarray(state_t.to(torch.float32).cpu().numpy()),
    )


def _run_jax_reference(
    *,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gate: np.ndarray,
    beta: np.ndarray,
    seq_lens: np.ndarray,
    initial_state: np.ndarray,
    qkv_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        pack_prepared_gdn_prefill_inputs,
        prepare_gdn_fla_prefill_kernel_inputs,
        unpack_prepared_gdn_prefill_output,
        gdn_fla_chunk_gated_delta_rule_packed_reference,
    )

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        jnp.asarray(query),
        jnp.asarray(key),
        jnp.asarray(value),
        jnp.asarray(gate),
        jnp.asarray(beta),
        jnp.asarray(seq_lens, dtype=jnp.int32),
        jnp.asarray(initial_state, dtype=jnp.float32),
        qkv_dtype=qkv_dtype,
    )
    packed_q, packed_k, packed_v, packed_g, packed_beta, cu = pack_prepared_gdn_prefill_inputs(
        prepared.query,
        prepared.key,
        prepared.value,
        prepared.gate,
        prepared.beta,
        prepared.seq_lens,
    )
    out, state = gdn_fla_chunk_gated_delta_rule_packed_reference(
        packed_q,
        packed_k,
        packed_v,
        packed_g,
        packed_beta,
        jnp.asarray(cu, dtype=jnp.int32),
        prepared.initial_state,
        chunk_size=64,
    )
    unpacked = unpack_prepared_gdn_prefill_output(out, cu, query.shape[1])
    return np.asarray(unpacked), np.asarray(state)


def _sweep_case(
    case,
    *,
    qkv_dtype: str = "bfloat16",
    provide_scale: bool = True,
    gate_mode: str = "raw",
    beta_mode: str = "raw",
    state_layout: str = "BHVK",
) -> dict[str, Any]:
    gate = np.asarray(case.gate, dtype=np.float32)
    beta = np.asarray(case.beta, dtype=np.float32)
    init = np.asarray(case.initial_state, dtype=np.float32)

    gate_used = gate if gate_mode == "raw" else np.cumsum(gate, axis=1, dtype=np.float32)
    beta_used = beta if beta_mode == "raw" else (1.0 / (1.0 + np.exp(-beta)))
    init_used = init if state_layout == "BHVK" else np.transpose(init, (0, 1, 3, 2))

    vllm_out, vllm_state = _run_vllm(
        query=np.asarray(case.query),
        key=np.asarray(case.key),
        value=np.asarray(case.value),
        gate=gate_used,
        beta=beta_used,
        initial_state=init_used,
        cu_seqlens=np.asarray(case.cu_seqlens),
        provide_scale=provide_scale,
    )
    ref_out, ref_state = _run_jax_reference(
        query=np.asarray(case.query),
        key=np.asarray(case.key),
        value=np.asarray(case.value),
        gate=np.asarray(case.gate, dtype=np.float32),
        beta=np.asarray(case.beta, dtype=np.float32),
        seq_lens=np.asarray(case.seq_lens, dtype=np.int32),
        initial_state=np.asarray(case.initial_state, dtype=np.float32),
        qkv_dtype=qkv_dtype,
    )
    return {
        "out_diff": _diff_metrics(ref_out, vllm_out),
        "state_diff": _diff_metrics(ref_state, vllm_state),
        "meta": {
            "qkv_dtype": qkv_dtype,
            "provide_scale": provide_scale,
            "gate_mode": gate_mode,
            "beta_mode": beta_mode,
            "state_layout": state_layout,
        },
        "artifacts": {
            "vllm_out": vllm_out,
            "vllm_state": vllm_state,
            "ref_out": ref_out,
            "ref_state": ref_state,
        },
    }


def _chunk_handoff(case) -> dict[str, Any]:
    length = int(case.token_len)
    if length <= 64:
        return {"status": "skipped", "reason": "length <= chunk size"}

    q = np.asarray(case.query)
    k = np.asarray(case.key)
    v = np.asarray(case.value)
    g = np.asarray(case.gate, dtype=np.float32)
    beta = np.asarray(case.beta, dtype=np.float32)
    init = np.asarray(case.initial_state, dtype=np.float32)
    seq_lens = np.asarray(case.seq_lens, dtype=np.int32)

    base = _sweep_case(case)["artifacts"]

    q1, q2 = q[:, :64], q[:, 64:]
    k1, k2 = k[:, :64], k[:, 64:]
    v1, v2 = v[:, :64], v[:, 64:]
    g1, g2 = g[:, :64], g[:, 64:]
    b1, b2 = beta[:, :64], beta[:, 64:]

    cu1 = np.array([0, 64], dtype=np.int32)
    cu2 = np.array([0, length - 64], dtype=np.int32)
    s1 = np.array([64], dtype=np.int32)
    s2 = np.array([length - 64], dtype=np.int32)

    _, vllm_state_64 = _run_vllm(
        query=q1, key=k1, value=v1, gate=g1, beta=b1, initial_state=init, cu_seqlens=cu1, provide_scale=True
    )
    _, ref_state_64 = _run_jax_reference(
        query=q1, key=k1, value=v1, gate=g1, beta=b1, seq_lens=s1, initial_state=init, qkv_dtype="bfloat16"
    )

    vllm_out_2, vllm_state_2 = _run_vllm(
        query=q2,
        key=k2,
        value=v2,
        gate=g2,
        beta=b2,
        initial_state=vllm_state_64,
        cu_seqlens=cu2,
        provide_scale=True,
    )
    ref_out_2, ref_state_2 = _run_jax_reference(
        query=q2,
        key=k2,
        value=v2,
        gate=g2,
        beta=b2,
        seq_lens=s2,
        initial_state=ref_state_64,
        qkv_dtype="bfloat16",
    )
    cross_out_2, cross_state_2 = _run_jax_reference(
        query=q2,
        key=k2,
        value=v2,
        gate=g2,
        beta=b2,
        seq_lens=s2,
        initial_state=vllm_state_64,
        qkv_dtype="bfloat16",
    )

    return {
        "status": "ok",
        "second_chunk_vllm_vs_ref": {
            "out_diff": _diff_metrics(ref_out_2, vllm_out_2),
            "state_diff": _diff_metrics(ref_state_2, vllm_state_2),
        },
        "second_chunk_ref_with_vllm_state_vs_vllm": {
            "out_diff": _diff_metrics(cross_out_2, vllm_out_2),
            "state_diff": _diff_metrics(cross_state_2, vllm_state_2),
        },
        "split_vs_monolithic_ref": {
            "out_diff": _diff_metrics(ref_out_2, base["ref_out"][:, 64:]),
            "state_diff": _diff_metrics(ref_state_2, base["ref_state"]),
        },
        "split_vs_monolithic_vllm": {
            "out_diff": _diff_metrics(vllm_out_2, base["vllm_out"][:, 64:]),
            "state_diff": _diff_metrics(vllm_state_2, base["vllm_state"]),
        },
    }


def _trim_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.pop("artifacts", None)
    return row


def _load_runtime():
    import torch
    from nanovllm_jax.config import Qwen3_5Config
    from nanovllm_jax.load_weights import load_weights_from_hf
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    vllm_path = "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages"
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    module = __import__("vllm.model_executor.layers.fla.ops.chunk", fromlist=["chunk_gated_delta_rule"])

    config = Qwen3_5Config.qwen3_5_0_8b()
    config.dtype = "bfloat16"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    params = load_weights_from_hf("Qwen/Qwen3.5-0.8B", config, verbose=False)
    return config, tokenizer, params, module


def _md_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# vLLM FLA Real Activation Contract Audit (2026-05-28)",
        "",
        f"- Smallest unacceptable length: `{payload['smallest_unacceptable_length']}`",
        f"- Threshold (max abs out): `{payload['unacceptable_threshold_out_max_abs']}`",
        "",
        "## Sweep",
        "",
        "| len | out max | out mean | state max | state mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["length_sweep"]:
        lines.append(
            f"| {row['length']} | {row['out_diff']['max_abs']:.6g} | {row['out_diff']['mean_abs']:.6g} | "
            f"{row['state_diff']['max_abs']:.6g} | {row['state_diff']['mean_abs']:.6g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    _configure_runtime_env()
    config, tokenizer, params, vllm_chunk_module = _load_runtime()
    global _VLLM_CHUNK_KERNEL
    _VLLM_CHUNK_KERNEL = getattr(vllm_chunk_module, "chunk_gated_delta_rule")

    length_rows: list[dict[str, Any]] = []
    transforms: dict[str, list[dict[str, Any]]] = {}
    handoff: dict[str, Any] = {}

    for length in LENGTHS:
        case = _build_real_case(
            params=params,
            config=config,
            tokenizer=tokenizer,
            layer_idx=0,
            prompt="The quick brown fox jumps over the lazy dog.",
            token_len=length,
            qkv_dtype="bfloat16",
        )
        base = _sweep_case(case)
        length_rows.append(
            {
                "length": length,
                "out_diff": base["out_diff"],
                "state_diff": base["state_diff"],
            }
        )

        if length in TRANSFORM_LENGTHS:
            tests = [
                ("baseline", {}),
                ("state_layout_BHKK", {"state_layout": "BHKV"}),
                ("qkv_dtype_fp32_ref", {"qkv_dtype": "float32"}),
                ("scale_omitted", {"provide_scale": False}),
                ("gate_cumsum_to_vllm", {"gate_mode": "cumsum"}),
                ("beta_sigmoid_to_vllm", {"beta_mode": "sigmoid"}),
            ]
            rows = []
            for name, overrides in tests:
                run = _sweep_case(case, **overrides)
                rows.append({"name": name, **_trim_artifacts(run)})
            transforms[str(length)] = rows

        if length >= 65:
            handoff[str(length)] = _chunk_handoff(case)

    unacceptable_threshold = 0.1
    smallest_unacceptable = None
    for row in length_rows:
        if row["out_diff"]["max_abs"] > unacceptable_threshold:
            smallest_unacceptable = row["length"]
            break

    payload = {
        "date_utc": "2026-05-28",
        "model": "Qwen/Qwen3.5-0.8B",
        "layer_idx": 0,
        "lengths": LENGTHS,
        "unacceptable_threshold_out_max_abs": unacceptable_threshold,
        "smallest_unacceptable_length": smallest_unacceptable,
        "length_sweep": length_rows,
        "transforms_64_65_128": transforms,
        "chunk_handoff": handoff,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(_md_summary(payload), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
