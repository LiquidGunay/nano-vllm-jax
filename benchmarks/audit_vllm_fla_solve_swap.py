#!/usr/bin/env python3
"""Causality audit: swap only solve_tril between JAX reference and vLLM stages."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.microbench_vllm_fla_gdn_real_activation import _build_real_case


OUT_JSON = Path("results/vllm_fla_solve_swap_audit_20260528.json")
OUT_MD = Path("results/vllm_fla_solve_swap_audit_20260528.md")
LENGTHS = [128, 256, 512]
CHUNK_SIZE = 64


def _configure_runtime_env() -> None:
    os.environ.setdefault("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    os.environ.setdefault("HF_HOME", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/mountpoint/.exp/.cache")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/mountpoint/.exp/.cache/flashinfer")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
        if str(getattr(x, "dtype", "")) == "torch.bfloat16":
            x = x.float()
    return np.asarray(x)


def _diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    finite = np.isfinite(d)
    return {
        "max_abs": float(np.max(d[finite])) if finite.any() else float("nan"),
        "mean_abs": float(np.mean(d[finite])) if finite.any() else float("nan"),
        "nan_count": int(np.size(d) - int(finite.sum())),
    }


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
    return config, tokenizer, params, torch, cumsum, kkt, solve, wy, dh, out


def _run_case(case, torch, cumsum_mod, kkt_mod, solve_mod, wy_mod, dh_mod, out_mod) -> dict[str, Any]:
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
    seq_lens = _to_np(case.seq_lens).astype(np.int32)
    cu = _to_np(case.cu_seqlens).astype(np.int32)
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
    pq, pk, pv, pg, pb, pcu = pack_prepared_gdn_prefill_inputs(
        prepared.query, prepared.key, prepared.value, prepared.gate, prepared.beta, prepared.seq_lens
    )
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(pcu, CHUNK_SIZE)
    num_chunks = int(_to_np(chunk_indices).shape[0])

    q_t = torch.tensor(np.asarray(query, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    k_t = torch.tensor(np.asarray(key, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    v_t = torch.tensor(np.asarray(value, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    g_t = torch.tensor(gate, device="cuda", dtype=torch.float32)
    b_t = torch.tensor(beta, device="cuda", dtype=torch.float32)
    init_t = torch.tensor(init_state, device="cuda", dtype=torch.float32)
    cu_t = torch.tensor(cu, device="cuda", dtype=torch.int64)

    # vLLM up to A + solve
    gcs_v = cumsum_mod.chunk_local_cumsum(g_t, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t)
    A_v = kkt_mod.chunk_scaled_dot_kkt_fwd(k_t, beta=b_t, g=gcs_v, cu_seqlens=cu_t, output_dtype=torch.float32)
    Ai_v_bf16 = solve_mod.solve_tril(A_v, cu_seqlens=cu_t, output_dtype=k_t.dtype)
    Ai_v_fp32 = solve_mod.solve_tril(A_v, cu_seqlens=cu_t, output_dtype=torch.float32)

    # JAX up to A + solve
    gcs_j = gdn_fla_chunk_local_cumsum_packed_reference(pg, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    A_j = gdn_fla_chunk_scaled_dot_kkt_packed_reference(pk, pb, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    Ai_j = gdn_fla_solve_tril_packed_reference(A_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    Ai_j_from_Av_bf16 = gdn_fla_solve_tril_packed_reference(jnp.asarray(_to_np(A_v[0])), pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)

    # Full baselines
    wj, uj = gdn_fla_recompute_w_u_packed_reference(pk, pv, pb, gcs_j, Ai_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    hj, vnewj, fsj = gdn_fla_chunk_delta_h_packed_reference(
        pk, wj, uj, gcs_j, pcu, prepared.initial_state,
        chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, output_final_state=True, save_new_value=True
    )
    oj = gdn_fla_chunk_fwd_o_packed_reference(pq, pk, vnewj, hj, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)

    wv, uv = wy_mod.recompute_w_u_fwd(k_t, v_t, b_t, gcs_v, Ai_v_bf16, cu_t)
    hv, vnewv, fsv = dh_mod.chunk_gated_delta_rule_fwd_h(
        k_t, wv, uv, g=gcs_v, initial_state=init_t, output_final_state=True, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t
    )
    ov = out_mod.chunk_fwd_o(q_t, k_t, vnewv, hv, g=gcs_v, cu_seqlens=cu_t, chunk_size=CHUNK_SIZE)
    torch.cuda.synchronize()

    # Hybrid 1: vLLM gate/A + JAX solve -> vLLM downstream
    Ai_j_t = torch.tensor(_to_np(Ai_j)[None], device="cuda", dtype=torch.bfloat16)
    wv_js, uv_js = wy_mod.recompute_w_u_fwd(k_t, v_t, b_t, gcs_v, Ai_j_t, cu_t)
    hv_js, vnewv_js, fsv_js = dh_mod.chunk_gated_delta_rule_fwd_h(
        k_t, wv_js, uv_js, g=gcs_v, initial_state=init_t, output_final_state=True, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t
    )
    ov_js = out_mod.chunk_fwd_o(q_t, k_t, vnewv_js, hv_js, g=gcs_v, cu_seqlens=cu_t, chunk_size=CHUNK_SIZE)

    # Hybrid 2: JAX gate/A + vLLM solve -> JAX downstream
    Ai_v_for_j = jnp.asarray(_to_np(Ai_v_bf16[0]).astype(np.float32))
    wj_vs, uj_vs = gdn_fla_recompute_w_u_packed_reference(pk, pv, pb, gcs_j, Ai_v_for_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    hj_vs, vnewj_vs, fsj_vs = gdn_fla_chunk_delta_h_packed_reference(
        pk, wj_vs, uj_vs, gcs_j, pcu, prepared.initial_state,
        chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, output_final_state=True, save_new_value=True
    )
    oj_vs = gdn_fla_chunk_fwd_o_packed_reference(pq, pk, vnewj_vs, hj_vs, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)

    # JAX downstream with solve-injection controls
    wj_Avsolve, uj_Avsolve = gdn_fla_recompute_w_u_packed_reference(
        pk, pv, pb, gcs_j, Ai_j_from_Av_bf16, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    hj_Avsolve, vnewj_Avsolve, fsj_Avsolve = gdn_fla_chunk_delta_h_packed_reference(
        pk, wj_Avsolve, uj_Avsolve, gcs_j, pcu, prepared.initial_state,
        chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, output_final_state=True, save_new_value=True
    )
    oj_Avsolve = gdn_fla_chunk_fwd_o_packed_reference(
        pq, pk, vnewj_Avsolve, hj_Avsolve, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )

    # Normalize baselines to packed shapes
    ov_np = _to_np(ov[0, :length]).astype(np.float32)
    fsv_np = _to_np(fsv).astype(np.float32)
    oj_np = _to_np(oj).astype(np.float32)
    fsj_np = _to_np(fsj).astype(np.float32)

    chains = {
        "full_jax": {"out": oj_np, "state": fsj_np},
        "full_vllm": {"out": ov_np, "state": fsv_np},
        "vllm_A_jax_solve_vllm_downstream": {
            "out": _to_np(ov_js[0, :length]).astype(np.float32),
            "state": _to_np(fsv_js).astype(np.float32),
        },
        "jax_A_vllm_solve_jax_downstream": {
            "out": _to_np(oj_vs).astype(np.float32),
            "state": _to_np(fsj_vs).astype(np.float32),
        },
        "jax_downstream_jax_solve_of_vllm_A": {
            "out": _to_np(oj_Avsolve).astype(np.float32),
            "state": _to_np(fsj_Avsolve).astype(np.float32),
        },
    }
    metrics: dict[str, Any] = {}
    for name, vals in chains.items():
        metrics[name] = {
            "vs_full_jax": {"out": _diff(vals["out"], oj_np), "state": _diff(vals["state"], fsj_np)},
            "vs_full_vllm": {"out": _diff(vals["out"], ov_np), "state": _diff(vals["state"], fsv_np)},
        }

    solve_deltas = {
        "Ai_vllm_bf16_vs_jax": _diff(_to_np(Ai_v_bf16[0]).astype(np.float32), _to_np(Ai_j).astype(np.float32)),
        "Ai_vllm_fp32_vs_jax": _diff(_to_np(Ai_v_fp32[0]).astype(np.float32), _to_np(Ai_j).astype(np.float32)),
        "Ai_vllm_bf16_vs_vllm_fp32": _diff(_to_np(Ai_v_bf16[0]).astype(np.float32), _to_np(Ai_v_fp32[0]).astype(np.float32)),
        "Ai_jax_solve_on_vllm_A_vs_jax_A": _diff(_to_np(Ai_j_from_Av_bf16).astype(np.float32), _to_np(Ai_j).astype(np.float32)),
    }

    return {"length": length, "solve_deltas": solve_deltas, "chain_metrics": metrics}


def _summary(payload: dict[str, Any]) -> str:
    lines = ["# vLLM Solve Swap Audit (2026-05-28)", ""]
    for row in payload["results"]:
        lines.append(f"## Length {row['length']}")
        lines.append("")
        lines.append("| chain | out max vs JAX | out max vs vLLM | state max vs JAX | state max vs vLLM |")
        lines.append("|---|---:|---:|---:|---:|")
        for chain, m in row["chain_metrics"].items():
            lines.append(
                f"| {chain} | {m['vs_full_jax']['out']['max_abs']:.6g} | {m['vs_full_vllm']['out']['max_abs']:.6g} | "
                f"{m['vs_full_jax']['state']['max_abs']:.6g} | {m['vs_full_vllm']['state']['max_abs']:.6g} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    _configure_runtime_env()
    config, tokenizer, params, torch, cumsum_mod, kkt_mod, solve_mod, wy_mod, dh_mod, out_mod = _load_runtime()

    results = []
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
        results.append(_run_case(case, torch, cumsum_mod, kkt_mod, solve_mod, wy_mod, dh_mod, out_mod))

    payload = {
        "date_utc": "2026-05-28",
        "model": "Qwen/Qwen3.5-0.8B",
        "chunk_size": CHUNK_SIZE,
        "lengths": LENGTHS,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_summary(payload), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
