#!/usr/bin/env python3
"""Tiny vLLM/FLA chunked GDN prefill probe against nano-vllm JAX reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _collect_environment() -> dict[str, Any]:
    import torch
    import jax

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "jax_devices": [str(device) for device in jax.devices()],
        "flashinfer_chunk_gated_delta_rule": None,
    }
    if torch.cuda.is_available():
        report["torch_cuda_device"] = torch.cuda.get_device_name(0)
        report["torch_cuda_capability"] = torch.cuda.get_device_capability(0)
    else:
        report["torch_cuda_device"] = None

    try:
        from flashinfer import gdn_prefill

        report["flashinfer_chunk_gated_delta_rule"] = {
            "ok": True,
            "has_callable": callable(getattr(gdn_prefill, "chunk_gated_delta_rule", None)),
        }
    except Exception as exc:
        report["flashinfer_chunk_gated_delta_rule"] = {"ok": False, "error": repr(exc)}

    return report


def _to_torch_cuda() -> tuple[Any, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    os.environ.setdefault("PYTHONPATH", "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages")
    sys.path.insert(0, "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages")
    return torch, __import__("vllm.model_executor.layers.fla.ops.chunk", fromlist=["chunk_gated_delta_rule"])


def _run_vllm(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch, vllm_chunk = _to_torch_cuda()
    if chunk_size != 64:
        raise ValueError("vLLM chunk_gated_delta_rule kernel path expects chunk size 64")

    q_t = torch.tensor(q[None, ...], device="cuda", dtype=torch.bfloat16)
    k_t = torch.tensor(k[None, ...], device="cuda", dtype=torch.bfloat16)
    v_t = torch.tensor(v[None, ...], device="cuda", dtype=torch.bfloat16)
    g_t = torch.tensor(g[None, ...], device="cuda", dtype=torch.float32)
    beta_t = torch.tensor(beta[None, ...], device="cuda", dtype=torch.float32)
    init_t = torch.tensor(initial_state, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, q.shape[0]], dtype=torch.int64, device="cuda")

    out, final_state = vllm_chunk.chunk_gated_delta_rule(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        initial_state=init_t,
        output_final_state=True,
        cu_seqlens=cu,
        scale=float(q.shape[-1] ** -0.5),
    )
    torch.cuda.synchronize()
    return (
        np.asarray(out.detach().cpu().to(torch.float32).numpy()),
        np.asarray(final_state.detach().cpu().to(torch.float32).numpy()),
    )


def _run_jax_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import jax.numpy as jnp

    from nanovllm_jax.kernels.gdn_fla import gdn_fla_chunk_gated_delta_rule_packed_reference

    cu = np.array([0, q.shape[0]], dtype=np.int32)
    out, state = gdn_fla_chunk_gated_delta_rule_packed_reference(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(g),
        jnp.asarray(beta),
        jnp.asarray(cu, dtype=np.int32),
        jnp.asarray(initial_state),
        chunk_size=chunk_size,
    )
    return np.asarray(out), np.asarray(state)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _case_inputs() -> list[dict[str, Any]]:
    return [
        {
            "name": "t1_gate0_beta1_init0",
            "q": np.array([[[1.0, -0.5], [0.25, 0.75]]], dtype=np.float32),
            "k": np.array([[[1.0, -0.5], [0.25, 0.75]]], dtype=np.float32),
            "v": np.array([[[0.2, -0.4], [0.3, 0.8]]], dtype=np.float32),
            "g": np.array([[0.0, 0.0]], dtype=np.float32),
            "beta": np.array([[1.0, 1.0]], dtype=np.float32),
            "state": np.zeros((1, 2, 2, 2), dtype=np.float32),
        },
        {
            "name": "t2_gate0_beta1_identity",
            "q": np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            "k": np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            "v": np.array(
                [
                    [[0.5, -0.25], [-0.5, 0.75]],
                    [[1.25, 0.2], [0.3, -0.4]],
                ],
                dtype=np.float32,
            ),
            "g": np.zeros((2, 2), dtype=np.float32),
            "beta": np.ones((2, 2), dtype=np.float32),
            "state": np.zeros((1, 2, 2, 2), dtype=np.float32),
        },
        {
            "name": "t2_gate_nonzero_state",
            "q": np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.5, -0.5], [1.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            "k": np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[-1.0, 0.5], [0.25, 1.0]],
                ],
                dtype=np.float32,
            ),
            "v": np.array(
                [
                    [[0.2, 0.7], [0.3, -0.1]],
                    [[-0.2, 0.4], [0.6, 0.25]],
                ],
                dtype=np.float32,
            ),
            "g": np.array([[0.0, 0.0], [0.5, -0.75]], dtype=np.float32),
            "beta": np.array([[0.5, 0.8], [1.2, 0.3]], dtype=np.float32),
            "state": np.array(
                [
                    [
                        [[0.2, 0.1], [0.0, 0.3]],
                        [[0.05, -0.15], [0.2, 0.4]],
                    ]
                ],
                dtype=np.float32,
            ),
        },
    ]


def _run_probe(chunk_size: int) -> list[dict[str, Any]]:
    Transform = tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], bool]

    transforms: list[Transform] = [
        ("vllm_input_gate_raw_beta_raw", lambda g: g, lambda beta: beta, False),
        ("vllm_input_gate_raw_beta_sigmoid", lambda g: g, lambda beta: 1.0 / (1.0 + np.exp(-beta)), False),
        ("vllm_input_gate_cumsum_beta_raw", lambda g: np.cumsum(g, axis=0), lambda beta: beta, False),
        ("vllm_input_gate_exp_beta_raw", lambda g: np.exp(g), lambda beta: beta, False),
        (
            "vllm_input_gate_raw_beta_raw_state_transposed",
            lambda g: g,
            lambda beta: beta,
            True,
        ),
    ]

    report_rows: list[dict[str, Any]] = []

    for case in _case_inputs():
        q = case["q"]
        k = case["k"]
        v = case["v"]
        g = case["g"]
        beta = case["beta"]
        state = case["state"]

        case_row: dict[str, Any] = {
            "name": case["name"],
            "results": [],
        }
        for name, gate_xform, beta_xform, transpose_state in transforms:
            try:
                g_in = gate_xform(g)
                beta_in = beta_xform(beta)
                vllm_out, vllm_state = _run_vllm(
                    q,
                    k,
                    v,
                    g_in,
                    beta_in,
                    state.transpose(0, 1, 3, 2) if transpose_state else state,
                    chunk_size=chunk_size,
                )
                ref_out, ref_state = _run_jax_reference(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    state,
                    chunk_size=chunk_size,
                )

                # Reference output uses packed [T, H, D] and [N, H, V, K];
                # compare vLLM [1, T, H, D] back to packed layout.
                vllm_out = vllm_out[0]
                # vLLM returns [N, H, V, K] final state.
                diff_out = _max_abs_diff(vllm_out, ref_out)
                diff_state = _max_abs_diff(vllm_state, ref_state)
                case_row["results"].append(
                    {
                        "transform": name,
                        "max_abs_diff_out": diff_out,
                        "max_abs_diff_state": diff_state,
                        "vllm_out": vllm_out.tolist(),
                        "ref_out": ref_out.tolist(),
                        "vllm_state": vllm_state.tolist(),
                        "ref_state": ref_state.tolist(),
                    }
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                case_row["results"].append({"transform": name, "error": repr(exc)})

        report_rows.append(case_row)

    return report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = {
        "chunk_size": args.chunk_size,
        "environment": _collect_environment(),
    }

    try:
        payload["results"] = _run_probe(args.chunk_size)
        payload["status"] = "ran"
    except Exception as exc:
        payload["status"] = "blocked"
        payload["error"] = repr(exc)

    print(json.dumps(payload, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["status"] == "ran" else 1


if __name__ == "__main__":
    raise SystemExit(main())
