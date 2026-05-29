#!/usr/bin/env python3
"""Stage-by-stage audit of vLLM FLA GDN vs local JAX references."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.microbench_vllm_fla_gdn_real_activation import _build_real_case


OUT_JSON = Path("results/vllm_fla_real_activation_stage_audit_20260528.json")
OUT_MD = Path("results/vllm_fla_real_activation_stage_audit_20260528.md")
CHUNK_SIZE = 64
LENGTHS = [128, 256, 512]


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
    return {"max_abs": float(np.max(diff[finite])), "mean_abs": float(np.mean(diff[finite]))}


def _chunk_slices(length: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(s, min(length, s + chunk_size)) for s in range(0, length, chunk_size)]


def _chunked_token_diffs(vllm: np.ndarray, ref: np.ndarray, chunk_size: int) -> list[dict[str, Any]]:
    rows = []
    for idx, (s, e) in enumerate(_chunk_slices(vllm.shape[0], chunk_size)):
        rows.append({"chunk_index": idx, "token_start": s, "token_end": e, **_diff_metrics(vllm[s:e], ref[s:e])})
    return rows


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
        if str(getattr(x, "dtype", "")) == "torch.bfloat16":
            x = x.to(dtype=x.float().dtype)
    return np.asarray(x)


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
    chunk = __import__("vllm.model_executor.layers.fla.ops.chunk", fromlist=["chunk_gated_delta_rule_fwd"])
    cumsum = __import__("vllm.model_executor.layers.fla.ops.cumsum", fromlist=["chunk_local_cumsum"])
    kkt = __import__("vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt", fromlist=["chunk_scaled_dot_kkt_fwd"])
    solve = __import__("vllm.model_executor.layers.fla.ops.solve_tril", fromlist=["solve_tril"])
    wy = __import__("vllm.model_executor.layers.fla.ops.wy_fast", fromlist=["recompute_w_u_fwd"])
    dh = __import__("vllm.model_executor.layers.fla.ops.chunk_delta_h", fromlist=["chunk_gated_delta_rule_fwd_h"])
    out = __import__("vllm.model_executor.layers.fla.ops.chunk_o", fromlist=["chunk_fwd_o"])

    config = Qwen3_5Config.qwen3_5_0_8b()
    config.dtype = "bfloat16"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    params = load_weights_from_hf("Qwen/Qwen3.5-0.8B", config, verbose=False)
    return config, tokenizer, params, torch, chunk, cumsum, kkt, solve, wy, dh, out


def _callable_signatures(modules: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, fn in modules.items():
        out[key] = str(inspect.signature(fn))
    return out


def _run_one_length(case, torch, stage_fns: dict[str, Any]) -> dict[str, Any]:
    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_delta_h_packed_reference,
        gdn_fla_chunk_fwd_o_packed_reference,
        gdn_fla_chunk_local_cumsum_packed_reference,
        gdn_fla_chunk_scaled_dot_kkt_packed_reference,
        gdn_fla_recompute_w_u_packed_reference,
        gdn_fla_solve_tril_packed_reference,
        pack_prepared_gdn_prefill_inputs,
        prepare_gdn_fla_chunk_metadata,
        prepare_gdn_fla_prefill_kernel_inputs,
    )

    query = _to_np(case.query)
    key = _to_np(case.key)
    value = _to_np(case.value)
    gate = _to_np(case.gate).astype(np.float32)
    beta = _to_np(case.beta).astype(np.float32)
    init_state = _to_np(case.initial_state).astype(np.float32)
    cu = _to_np(case.cu_seqlens).astype(np.int32)
    seq_lens = _to_np(case.seq_lens).astype(np.int32)
    length = int(case.token_len)

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        jnp.asarray(query),
        jnp.asarray(key),
        jnp.asarray(value),
        jnp.asarray(gate),
        jnp.asarray(beta),
        jnp.asarray(seq_lens, dtype=jnp.int32),
        jnp.asarray(init_state, dtype=jnp.float32),
        qkv_dtype="bfloat16",
    )
    packed_q, packed_k, packed_v, packed_g, packed_beta, packed_cu = pack_prepared_gdn_prefill_inputs(
        prepared.query,
        prepared.key,
        prepared.value,
        prepared.gate,
        prepared.beta,
        prepared.seq_lens,
    )
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(packed_cu, CHUNK_SIZE)
    num_chunks = int(_to_np(chunk_indices).shape[0])

    q_t = torch.tensor(np.asarray(query, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    k_t = torch.tensor(np.asarray(key, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    v_t = torch.tensor(np.asarray(value, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    g_t = torch.tensor(gate, device="cuda", dtype=torch.float32)
    beta_t = torch.tensor(beta, device="cuda", dtype=torch.float32)
    init_t = torch.tensor(init_state, device="cuda", dtype=torch.float32)
    cu_t = torch.tensor(cu, device="cuda", dtype=torch.int64)

    g_cs_t = stage_fns["chunk_local_cumsum"](g_t, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t)
    A_t = stage_fns["chunk_scaled_dot_kkt_fwd"](k_t, beta=beta_t, g=g_cs_t, cu_seqlens=cu_t, output_dtype=torch.float32)
    Ai_t = stage_fns["solve_tril"](A_t, cu_seqlens=cu_t, output_dtype=k_t.dtype)
    w_t, u_t = stage_fns["recompute_w_u_fwd"](
        k=k_t, v=v_t, beta=beta_t, g_cumsum=g_cs_t, A=Ai_t, cu_seqlens=cu_t
    )
    h_t, vnew_t, final_state_t = stage_fns["chunk_gated_delta_rule_fwd_h"](
        k=k_t,
        w=w_t,
        u=u_t,
        g=g_cs_t,
        initial_state=init_t,
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_t,
    )
    o_t = stage_fns["chunk_fwd_o"](
        q=q_t, k=k_t, v=vnew_t, h=h_t, g=g_cs_t, cu_seqlens=cu_t, chunk_size=CHUNK_SIZE
    )
    torch.cuda.synchronize()

    g_cs_vllm = _to_np(g_cs_t[0, :length])
    A_vllm = _to_np(A_t[0, :length])
    Ai_vllm = _to_np(Ai_t[0, :length])
    w_vllm = _to_np(w_t[0, :length])
    u_vllm = _to_np(u_t[0, :length])
    vnew_vllm = _to_np(vnew_t[0, :length])
    h_vllm = _to_np(h_t[0, :num_chunks])
    out_vllm = _to_np(o_t[0, :length])
    final_state_vllm = _to_np(final_state_t)

    g_cs_ref = _to_np(gdn_fla_chunk_local_cumsum_packed_reference(packed_g, packed_cu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices))
    A_ref = _to_np(gdn_fla_chunk_scaled_dot_kkt_packed_reference(packed_k, packed_beta, g_cs_ref, packed_cu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices))
    Ai_ref = _to_np(gdn_fla_solve_tril_packed_reference(A_ref, packed_cu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices))
    w_ref, u_ref = gdn_fla_recompute_w_u_packed_reference(
        packed_k, packed_v, packed_beta, g_cs_ref, Ai_ref, packed_cu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    h_ref, vnew_ref, final_state_ref = gdn_fla_chunk_delta_h_packed_reference(
        packed_k,
        w_ref,
        u_ref,
        g_cs_ref,
        packed_cu,
        prepared.initial_state,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        output_final_state=True,
        save_new_value=True,
    )
    out_ref = gdn_fla_chunk_fwd_o_packed_reference(
        packed_q, packed_k, vnew_ref, h_ref, g_cs_ref, packed_cu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )

    w_ref = _to_np(w_ref)
    u_ref = _to_np(u_ref)
    h_ref = _to_np(h_ref)
    vnew_ref = _to_np(vnew_ref)
    final_state_ref = _to_np(final_state_ref)
    out_ref = _to_np(out_ref)

    stage_rows = {
        "gate_cumsum": {"global": _diff_metrics(g_cs_vllm, g_cs_ref), "by_chunk": _chunked_token_diffs(g_cs_vllm, g_cs_ref, CHUNK_SIZE)},
        "A": {"global": _diff_metrics(A_vllm, A_ref), "by_chunk": _chunked_token_diffs(A_vllm, A_ref, CHUNK_SIZE)},
        "attention_inverse": {"global": _diff_metrics(Ai_vllm, Ai_ref), "by_chunk": _chunked_token_diffs(Ai_vllm, Ai_ref, CHUNK_SIZE)},
        "w": {"global": _diff_metrics(w_vllm, w_ref), "by_chunk": _chunked_token_diffs(w_vllm, w_ref, CHUNK_SIZE)},
        "u": {"global": _diff_metrics(u_vllm, u_ref), "by_chunk": _chunked_token_diffs(u_vllm, u_ref, CHUNK_SIZE)},
        "h": {"global": _diff_metrics(h_vllm, h_ref), "by_chunk": [{"chunk_index": i, **_diff_metrics(h_vllm[i], h_ref[i])} for i in range(num_chunks)]},
        "v_new": {"global": _diff_metrics(vnew_vllm, vnew_ref), "by_chunk": _chunked_token_diffs(vnew_vllm, vnew_ref, CHUNK_SIZE)},
        "final_state": {"global": _diff_metrics(final_state_vllm, final_state_ref)},
        "output": {"global": _diff_metrics(out_vllm, out_ref), "by_chunk": _chunked_token_diffs(out_vllm, out_ref, CHUNK_SIZE)},
    }

    return {"length": length, "num_chunks": num_chunks, "stage_diffs": stage_rows}


def _first_divergent_stage(length_rows: list[dict[str, Any]]) -> dict[str, Any]:
    order = ["gate_cumsum", "A", "attention_inverse", "w", "u", "h", "v_new", "output"]
    best: dict[str, Any] = {"stage": None, "length": None, "chunk_index": None, "max_abs": 0.0, "mean_abs": 0.0}
    threshold = 0.1
    for row in length_rows:
        for stage in order:
            by_chunk = row["stage_diffs"][stage].get("by_chunk", [])
            for chunk in by_chunk:
                if chunk["max_abs"] > threshold:
                    return {
                        "stage": stage,
                        "length": row["length"],
                        "chunk_index": chunk["chunk_index"],
                        "max_abs": chunk["max_abs"],
                        "mean_abs": chunk["mean_abs"],
                        "threshold": threshold,
                    }
                if chunk["max_abs"] > best["max_abs"]:
                    best = {
                        "stage": stage,
                        "length": row["length"],
                        "chunk_index": chunk["chunk_index"],
                        "max_abs": chunk["max_abs"],
                        "mean_abs": chunk["mean_abs"],
                        "threshold": threshold,
                    }
    return best


def _summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# vLLM FLA Stage Audit (2026-05-28)",
        "",
        f"- First divergent stage: `{payload['first_divergent_stage']['stage']}`",
        f"- At length/chunk: `{payload['first_divergent_stage']['length']}` / `{payload['first_divergent_stage']['chunk_index']}`",
        f"- Max/mean diff: `{payload['first_divergent_stage']['max_abs']:.6g}` / `{payload['first_divergent_stage']['mean_abs']:.6g}`",
        "",
    ]
    for row in payload["length_results"]:
        lines.append(f"## Length {row['length']}")
        lines.append("")
        lines.append("| stage | max_abs | mean_abs |")
        lines.append("|---|---:|---:|")
        for stage, metrics in row["stage_diffs"].items():
            g = metrics["global"]
            lines.append(f"| {stage} | {g['max_abs']:.6g} | {g['mean_abs']:.6g} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    _configure_runtime_env()
    config, tokenizer, params, torch, chunk, cumsum, kkt, solve, wy, dh, out = _load_runtime()
    stage_fns = {
        "chunk_local_cumsum": cumsum.chunk_local_cumsum,
        "chunk_scaled_dot_kkt_fwd": kkt.chunk_scaled_dot_kkt_fwd,
        "solve_tril": solve.solve_tril,
        "recompute_w_u_fwd": wy.recompute_w_u_fwd,
        "chunk_gated_delta_rule_fwd_h": dh.chunk_gated_delta_rule_fwd_h,
        "chunk_fwd_o": out.chunk_fwd_o,
    }

    length_results = []
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
        length_results.append(_run_one_length(case, torch, stage_fns))

    payload = {
        "date_utc": "2026-05-28",
        "model": "Qwen/Qwen3.5-0.8B",
        "layer_idx": 0,
        "chunk_size": CHUNK_SIZE,
        "lengths": LENGTHS,
        "callable_stage_signatures": _callable_signatures(stage_fns),
        "length_results": length_results,
        "first_divergent_stage": _first_divergent_stage(length_results),
        "notes": {
            "layout": "vLLM stage tensors are [B,T,H,*] (or [B,NT,H,V,K] for h); JAX refs are packed [T,H,*] and [num_chunks,H,V,K]. Comparisons use packed-aligned views.",
            "dtype": "q/k/v into vLLM are bf16, gate/beta/state are fp32; stage diffs computed in fp32 numpy.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_summary_md(payload), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
