#!/usr/bin/env python3
"""Probe vLLM-like packed solve reference against vendored vLLM solve."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.microbench_vllm_fla_gdn_real_activation import _build_real_case


OUT_JSON = Path("results/vllm_like_solve_reference_probe_20260528.json")
OUT_MD = Path("results/vllm_like_solve_reference_probe_20260528.md")
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

    config = Qwen3_5Config.qwen3_5_0_8b()
    config.dtype = "bfloat16"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    params = load_weights_from_hf("Qwen/Qwen3.5-0.8B", config, verbose=False)
    return config, tokenizer, params, torch, cumsum, kkt, solve


def _run_length(case, torch, cumsum_mod, kkt_mod, solve_mod) -> dict[str, Any]:
    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_delta_h_packed_reference,
        gdn_fla_chunk_fwd_o_packed_reference,
        gdn_fla_chunk_local_cumsum_packed_reference,
        gdn_fla_chunk_scaled_dot_kkt_packed_reference,
        gdn_fla_recompute_w_u_packed_reference,
        gdn_fla_solve_tril_packed_reference,
        gdn_fla_solve_tril_packed_vllm_like_reference,
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

    q_t = torch.tensor(np.asarray(query, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    k_t = torch.tensor(np.asarray(key, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    g_t = torch.tensor(gate, device="cuda", dtype=torch.float32)
    b_t = torch.tensor(beta, device="cuda", dtype=torch.float32)
    cu_t = torch.tensor(cu, device="cuda", dtype=torch.int64)

    gcs_v = cumsum_mod.chunk_local_cumsum(g_t, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t)
    A_v = kkt_mod.chunk_scaled_dot_kkt_fwd(k_t, beta=b_t, g=gcs_v, cu_seqlens=cu_t, output_dtype=torch.float32)
    Ai_v = solve_mod.solve_tril(A_v, cu_seqlens=cu_t, output_dtype=k_t.dtype)
    Ai_v_np = _to_np(Ai_v[0]).astype(np.float32)
    A_v_np = _to_np(A_v[0]).astype(np.float32)

    gcs_j = gdn_fla_chunk_local_cumsum_packed_reference(pg, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    A_j = gdn_fla_chunk_scaled_dot_kkt_packed_reference(pk, pb, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    Ai_j = gdn_fla_solve_tril_packed_reference(A_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    Ai_like = gdn_fla_solve_tril_packed_vllm_like_reference(
        jnp.asarray(A_v_np),
        pcu,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        output_dtype="bfloat16",
    )

    # downstream with strict reference solve and vllm-like solve
    w_ref, u_ref = gdn_fla_recompute_w_u_packed_reference(pk, pv, pb, gcs_j, Ai_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    h_ref, vnew_ref, fs_ref = gdn_fla_chunk_delta_h_packed_reference(
        pk, w_ref, u_ref, gcs_j, pcu, prepared.initial_state,
        chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, output_final_state=True, save_new_value=True
    )
    o_ref = gdn_fla_chunk_fwd_o_packed_reference(pq, pk, vnew_ref, h_ref, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)

    w_like, u_like = gdn_fla_recompute_w_u_packed_reference(pk, pv, pb, gcs_j, Ai_like, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)
    h_like, vnew_like, fs_like = gdn_fla_chunk_delta_h_packed_reference(
        pk, w_like, u_like, gcs_j, pcu, prepared.initial_state,
        chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, output_final_state=True, save_new_value=True
    )
    o_like = gdn_fla_chunk_fwd_o_packed_reference(pq, pk, vnew_like, h_like, gcs_j, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices)

    # full vllm output/state
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    v_t = torch.tensor(np.asarray(value, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    init_t = torch.tensor(init_state, device="cuda", dtype=torch.float32)
    o_v, fs_v = chunk_gated_delta_rule(
        q=q_t, k=k_t, v=v_t, g=g_t, beta=b_t, initial_state=init_t, output_final_state=True, cu_seqlens=cu_t, scale=float(query.shape[-1] ** -0.5)
    )
    torch.cuda.synchronize()

    o_v_np = _to_np(o_v[0, :length]).astype(np.float32)
    fs_v_np = _to_np(fs_v).astype(np.float32)
    o_ref_np = _to_np(o_ref).astype(np.float32)
    fs_ref_np = _to_np(fs_ref).astype(np.float32)
    o_like_np = _to_np(o_like).astype(np.float32)
    fs_like_np = _to_np(fs_like).astype(np.float32)

    return {
        "length": length,
        "inverse_diff": {
            "jax_ref_vs_vllm": _diff(_to_np(Ai_j).astype(np.float32), Ai_v_np),
            "vllm_like_vs_vllm": _diff(_to_np(Ai_like).astype(np.float32), Ai_v_np),
        },
        "downstream_diff": {
            "ref_vs_vllm": {
                "out": _diff(o_ref_np, o_v_np),
                "state": _diff(fs_ref_np, fs_v_np),
            },
            "vllm_like_vs_vllm": {
                "out": _diff(o_like_np, o_v_np),
                "state": _diff(fs_like_np, fs_v_np),
            },
            "vllm_like_vs_ref": {
                "out": _diff(o_like_np, o_ref_np),
                "state": _diff(fs_like_np, fs_ref_np),
            },
        },
    }


def _summary(payload: dict[str, Any]) -> str:
    lines = ["# vLLM-like Solve Reference Probe (2026-05-28)", ""]
    for row in payload["results"]:
        lines.append(f"## Length {row['length']}")
        lines.append("")
        inv0 = row["inverse_diff"]["jax_ref_vs_vllm"]["max_abs"]
        inv1 = row["inverse_diff"]["vllm_like_vs_vllm"]["max_abs"]
        out0 = row["downstream_diff"]["ref_vs_vllm"]["out"]["max_abs"]
        out1 = row["downstream_diff"]["vllm_like_vs_vllm"]["out"]["max_abs"]
        st0 = row["downstream_diff"]["ref_vs_vllm"]["state"]["max_abs"]
        st1 = row["downstream_diff"]["vllm_like_vs_vllm"]["state"]["max_abs"]
        lines.append(f"- inverse max diff: ref `{inv0:.6g}` -> vllm_like `{inv1:.6g}`")
        lines.append(f"- output max diff vs vLLM: ref `{out0:.6g}` -> vllm_like `{out1:.6g}`")
        lines.append(f"- state max diff vs vLLM: ref `{st0:.6g}` -> vllm_like `{st1:.6g}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    _configure_runtime_env()
    config, tokenizer, params, torch, cumsum_mod, kkt_mod, solve_mod = _load_runtime()
    rows = []
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
        rows.append(_run_length(case, torch, cumsum_mod, kkt_mod, solve_mod))
    payload = {
        "date_utc": "2026-05-28",
        "model": "Qwen/Qwen3.5-0.8B",
        "lengths": LENGTHS,
        "chunk_size": CHUNK_SIZE,
        "results": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_summary(payload), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
