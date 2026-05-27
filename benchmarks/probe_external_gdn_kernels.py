#!/usr/bin/env python3
"""Probe external GDN kernel feasibility on the current GPU host.

This is a decision artifact, not a benchmark. It records whether the installed
prewritten GDN kernels can be called directly from this repo's JAX serving path,
and why a candidate is blocked when it cannot be used.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_paths import configure_flashinfer_cache, default_runtime_root


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


def _package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False}
    version_names = [name]
    if name == "flashinfer":
        version_names.append("flashinfer-python")
    version = None
    for candidate in version_names:
        try:
            version = importlib.metadata.version(candidate)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "available": True,
        "version": version,
        "origin": spec.origin,
    }


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        return [{"error": repr(exc), "source": "nvidia-smi"}]

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, name, compute_cap = parts
        try:
            major_text, minor_text = compute_cap.split(".", maxsplit=1)
            compute_capability = [int(major_text), int(minor_text)]
        except Exception:
            compute_capability = None
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "compute_cap": compute_cap,
                "compute_capability": compute_capability,
                "source": "nvidia-smi",
            }
        )
    return gpus


def _torch_gpus() -> list[dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        return [{"error": repr(exc), "source": "torch"}]

    try:
        if not torch.cuda.is_available():
            return [{"available": False, "source": "torch"}]
        gpus = []
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            gpus.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_cap": f"{major}.{minor}",
                    "compute_capability": [major, minor],
                    "source": "torch",
                }
            )
        return gpus
    except Exception as exc:
        return [{"error": repr(exc), "source": "torch"}]


def _first_compute_capability(gpus: list[dict[str, Any]]) -> list[int] | None:
    for gpu in gpus:
        capability = gpu.get("compute_capability")
        if isinstance(capability, list) and len(capability) == 2:
            return [int(capability[0]), int(capability[1])]
    return None


def _path_exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def _flashinfer_gdn_status(run_smoke: bool, compute_capability: list[int] | None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "package": _package_status("flashinfer"),
        "jax_tvm_ffi": _package_status("jax_tvm_ffi"),
        "requires": {
            "gpu": "SM90 Hopper or SM100 Blackwell",
            "qkv_dtype": ["float16", "bfloat16"],
            "gate_beta_state_dtype": "float32",
            "cu_seqlens_dtype": "int64",
            "state_layout": "[N, H, V, K]",
        },
        "source_paths": [
            _path_exists(Path(".venv/lib/python3.11/site-packages/flashinfer/gdn_prefill.py")),
            _path_exists(
                Path(".venv/lib/python3.11/site-packages/flashinfer/data/csrc/gdn_prefill_launcher.cu")
            ),
            _path_exists(Path(".venv/lib/python3.11/site-packages/flashinfer/jit/gdn.py")),
        ],
    }
    if compute_capability is None:
        status["direct_runtime_supported"] = False
        status["blocker"] = "Could not determine CUDA compute capability."
    else:
        major = compute_capability[0]
        status["direct_runtime_supported"] = major in (9, 10)
        if major not in (9, 10):
            status["blocker"] = (
                f"Current GPU compute capability {compute_capability[0]}.{compute_capability[1]} "
                "does not satisfy FlashInfer GDN prefill's SM90/SM100 requirement."
            )

    if not run_smoke:
        status["smoke"] = {"ran": False, "reason": "--run-smoke was not set"}
        return status
    if not status.get("direct_runtime_supported"):
        status["smoke"] = {"ran": False, "reason": status.get("blocker")}
        return status
    if not status["package"]["available"]:
        status["smoke"] = {"ran": False, "reason": "flashinfer is not importable"}
        return status

    try:
        import torch
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        total_tokens = 64
        heads = 16
        dim = 128
        q = torch.randn(total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        g = torch.full((total_tokens, heads), 0.9, device="cuda", dtype=torch.float32)
        beta = torch.full((total_tokens, heads), 0.5, device="cuda", dtype=torch.float32)
        initial_state = torch.zeros(1, heads, dim, dim, device="cuda", dtype=torch.float32)
        cu_seqlens = torch.tensor([0, total_tokens], device="cuda", dtype=torch.int64)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        torch.cuda.synchronize()
        status["smoke"] = {
            "ran": True,
            "seconds": time.perf_counter() - started,
            "output_shape": list(output.shape),
            "state_shape": list(final_state.shape),
            "output_dtype": str(output.dtype),
            "state_dtype": str(final_state.dtype),
        }
    except Exception as exc:
        status["smoke"] = {"ran": True, "error": repr(exc)}
    return status


def _vllm_fla_status() -> dict[str, Any]:
    root = Path("/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages")
    paths = {
        "gdn_layer": root / "vllm/model_executor/layers/mamba/gdn_linear_attn.py",
        "fused_post_conv_prep": root
        / "vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py",
        "chunk_prefill": root / "vllm/model_executor/layers/fla/ops/chunk.py",
        "packed_decode": root / "vllm/model_executor/layers/fla/ops/fused_recurrent.py",
    }
    return {
        "vllm_package": {
            "venv": str(root.parents[2]),
            "paths_present": {name: path.exists() for name, path in paths.items()},
        },
        "source_paths": {name: str(path) for name, path in paths.items()},
        "observed_contract": {
            "prefill_pipeline": "causal_conv1d -> fused_post_conv_prep -> chunk_gated_delta_rule",
            "prefill_qkv_layout": "[L,H,D] then [1,L,H,D] for FLA chunk",
            "prefill_state_layout": "[N,H,V,K]",
            "chunk_size": 64,
            "decode_boundary": "mixed_qkv [B, 2*H*K + HV*V], a/b [B,HV], state [*,HV,V,K]",
        },
        "blockers": [
            "Vendored vLLM/FLA kernels are Torch/Triton custom ops, not JAX FFI targets.",
            "The FLA chunk prefill wrapper rejects torch.float32 q/k/v and asks for bfloat16.",
            "Directly calling the Torch path from JAX serving would introduce host/framework boundaries.",
        ],
        "recommendation": (
            "Use vLLM/FLA as the port/fork reference for a JAX-facing kernel boundary. "
            "On this host, direct FlashInfer GDN prefill is blocked by SM capability, "
            "so the practical path is a vLLM/FLA-derived implementation behind the "
            "existing post-conv or packed-decode ABI."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default="",
        help="Path for the probe JSON. Defaults to results/external_gdn_kernel_probe_<timestamp>.json.",
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run a tiny FlashInfer GDN smoke only when the current GPU satisfies the capability gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = default_runtime_root()
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    configure_flashinfer_cache()
    for path in (
        Path(os.environ["TMPDIR"]),
        Path(os.environ["XDG_CACHE_HOME"]),
        Path(os.environ["HF_HOME"]),
    ):
        path.mkdir(parents=True, exist_ok=True)

    nvidia_gpus = _nvidia_smi_gpus()
    torch_gpus = _torch_gpus()
    compute_capability = _first_compute_capability(nvidia_gpus) or _first_compute_capability(torch_gpus)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_path = Path(args.output_json or f"results/external_gdn_kernel_probe_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flashinfer_gdn = _flashinfer_gdn_status(args.run_smoke, compute_capability)
    payload = {
        "timestamp_utc": timestamp,
        "runtime_root": str(root),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TMPDIR",
                "XDG_CACHE_HOME",
                "HF_HOME",
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_CUBIN_DIR",
            )
        },
        "packages": {
            name: _package_status(name)
            for name in ("jax", "jaxlib", "jax_tvm_ffi", "flashinfer", "torch", "triton")
        },
        "gpu": {
            "nvidia_smi": nvidia_gpus,
            "torch": torch_gpus,
            "selected_compute_capability": compute_capability,
        },
        "flashinfer_gdn_prefill": flashinfer_gdn,
        "vllm_fla_gdn": _vllm_fla_status(),
        "decision": {
            "direct_flashinfer_gdn_prefill": "blocked"
            if not flashinfer_gdn.get("direct_runtime_supported")
            else "probeable",
            "next_gdn_path": (
                "vLLM/FLA-derived port or wrapper behind the existing post-conv/packed-decode ABI; "
                "do not pursue direct FlashInfer GDN prefill on SM86."
            ),
        },
    }

    output_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_json": str(output_path), "decision": payload["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
