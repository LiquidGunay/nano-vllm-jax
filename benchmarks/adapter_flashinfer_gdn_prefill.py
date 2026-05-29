#!/usr/bin/env python3
"""Adapter probe for FlashInfer chunked GDN prefill."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path


def _configure_runtime() -> None:
    os.environ.setdefault("TMPDIR", "/mountpoint/.exp/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/mountpoint/.exp/.cache")
    os.environ.setdefault("HF_HOME", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    from runtime_paths import configure_flashinfer_cache

    configure_flashinfer_cache()


def _collect_contract() -> dict[str, object]:
    _configure_runtime()
    try:
        import flashinfer
        from flashinfer import gdn_prefill
    except Exception as exc:  # pragma: no cover - import path is environment-dependent
        return {
            "flashinfer_import": {"ok": False, "error": repr(exc)},
        }

    fn = getattr(gdn_prefill, "chunk_gated_delta_rule", None)
    try:
        signature = str(inspect.signature(fn)) if fn is not None else None
    except (TypeError, ValueError):
        signature = None

    return {
        "flashinfer_import": {"ok": True, "path": getattr(flashinfer, "__file__", "")},
        "chunk_gated_delta_rule": {
            "present": fn is not None,
            "signature": signature,
        },
        "sources": {
            "gdn_prefill_py": str(Path(flashinfer.__file__).with_name("gdn_prefill.py"))
            if getattr(flashinfer, "__file__", "").endswith("__init__.py")
            else "unknown",
            "blackwell_adapter": str(
                Path(flashinfer.__file__).resolve().parents[0] / "gdn_kernels" / "blackwell" / "gdn_prefill.py"
            ),
            "sm90_source": str(
                Path(flashinfer.__file__).resolve().parents[0]
                / "data"
                / "csrc"
                / "prefill_kernel_delta_rule_sm90.cu"
            ),
        },
        "contract": {
            "qkv_layout": "[total_tokens, num_heads, head_dim]",
            "gate_beta_layout": "[total_tokens, num_sab_heads]",
            "state_layout": "[num_rows, num_sab_heads, head_dim, head_dim]",
            "output_layout": "[total_tokens, num_o_heads, head_dim]",
        },
    }


def _run_flashinfer_smoke(chunk_size: int = 64, compare_jax: bool = False) -> dict[str, object]:
    _configure_runtime()

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return {"status": "blocked", "reason": "torch.cuda.is_available() is False"}

    device = torch.device("cuda")
    if chunk_size != 64:
        return {
            "status": "blocked",
            "reason": "flashinfer chunk_gated_delta_rule appears fixed to chunk size 64",
        }

    batch = 4
    heads = 16
    head_dim = 128
    lengths = [0, 64, 128, 192, 256]
    cu_seqlens = torch.tensor(lengths, dtype=torch.int64, device=device)
    total_tokens = int(cu_seqlens[-1].item())
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(0)
    q = torch.randn(total_tokens, heads, head_dim, generator=g, device=device, dtype=dtype)
    k = torch.randn(total_tokens, heads, head_dim, generator=g, device=device, dtype=dtype)
    v = torch.randn(total_tokens, heads, head_dim, generator=g, device=device, dtype=dtype)
    gate = torch.randn(total_tokens, heads, generator=g, device=device, dtype=torch.float32)
    beta = torch.randn(total_tokens, heads, generator=g, device=device, dtype=torch.float32)
    initial_state = torch.zeros(batch - 1, heads, head_dim, head_dim, device=device, dtype=torch.float32)

    out = torch.empty(total_tokens, heads, head_dim, device=device, dtype=dtype)
    final_state = torch.empty(batch - 1, heads, head_dim, head_dim, device=device, dtype=torch.float32)

    from flashinfer import gdn_prefill

    out_actual, state_actual = gdn_prefill.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=out,
        output_state=final_state,
        scale=1.0 / (head_dim ** 0.5),
    )

    result: dict[str, object] = {
        "status": "smoke-ran",
        "out_shape": list(out_actual.shape),
        "state_shape": list(state_actual.shape),
        "out_dtype": str(out_actual.dtype),
        "state_dtype": str(state_actual.dtype),
    }

    if not compare_jax:
        return result

    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_gated_delta_rule_packed_reference,
    )

    q_np = np.asarray(q.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)
    k_np = np.asarray(k.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)
    v_np = np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)
    gate_np = np.asarray(gate.detach().cpu().numpy(), dtype=np.float32)
    beta_np = np.asarray(beta.detach().cpu().numpy(), dtype=np.float32)
    cu_np = np.asarray(cu_seqlens.detach().cpu().numpy(), dtype=np.int64)
    initial_np = np.zeros((batch - 1, heads, head_dim, head_dim), dtype=np.float32)

    out_ref, state_ref = gdn_fla_chunk_gated_delta_rule_packed_reference(
        jnp.asarray(q_np),
        jnp.asarray(k_np),
        jnp.asarray(v_np),
        jnp.asarray(gate_np),
        jnp.asarray(beta_np),
        jnp.asarray(cu_np),
        jnp.asarray(initial_np),
        chunk_size=chunk_size,
    )
    out_ref_np = np.asarray(out_ref)
    state_ref_np = np.asarray(state_ref)
    out_np = np.asarray(out_actual.detach().to(torch.float32).cpu().numpy())
    state_np = np.asarray(state_actual.detach().to(torch.float32).cpu().numpy())

    diff_out = np.max(np.abs(out_np - out_ref_np))
    diff_state = np.max(np.abs(state_np - state_ref_np))
    result.update({
        "jax_compare": {
            "chunk_size": chunk_size,
            "dtype": "float32",
            "max_abs_diff_out": float(diff_out),
            "max_abs_diff_state": float(diff_state),
        }
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--compare-jax", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, object] = {
        "contract": _collect_contract(),
    }
    if payload["contract"].get("flashinfer_import", {}).get("ok"):
        payload["smoke"] = _run_flashinfer_smoke(
            chunk_size=args.chunk_size,
            compare_jax=args.compare_jax,
        )

    print(json.dumps(payload, indent=2, default=str, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
