"""Run one backend against the committed benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
from statistics import median
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    required = {
        "schema_version",
        "claim_id",
        "model",
        "workload",
        "capacity",
        "measurement",
        "frameworks",
    }
    if set(manifest) != required:
        raise ValueError(f"manifest keys must be exactly {sorted(required)}")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest schema")
    workload = manifest["workload"]
    if workload["batch_size"] != 1:
        raise ValueError("this artifact makes one B=1 claim")
    if workload["temperature"] != 0 or not workload["ignore_eos"]:
        raise ValueError("the claim requires greedy fixed-length decode")
    if workload["prefix_cache"]:
        raise ValueError("the claim requires a prefix miss")
    measurement = manifest["measurement"]
    if measurement["warmup_repeats"] < 1 or measurement["repeats"] < 1:
        raise ValueError("warmup and measured repeats must be positive")
    return manifest


def prompt_rows(manifest: dict[str, Any]) -> list[list[int]]:
    workload = manifest["workload"]
    return [
        [
            workload["prompt_start_token"]
            + row * workload["prompt_row_stride"]
            + position
            for position in range(workload["prompt_tokens"])
        ]
        for row in range(workload["batch_size"])
    ]


def _children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return []
    return [int(value) for value in text.split()]


def _process_tree_rss_bytes(root: int) -> int:
    pending = [root]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(_children(pid))
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


def _nvidia_smi(*fields: str) -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("the artifact requires exactly one visible GPU")
    return [part.strip() for part in rows[0].split(",")]


def _device_used_bytes() -> int:
    return int(_nvidia_smi("memory.used")[0]) * 1024 * 1024


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def _output_hash(rows: list[list[int]]) -> str:
    encoded = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reference_rows(
    path: Path | None,
    manifest: dict[str, Any],
) -> list[list[int]] | None:
    if path is None:
        return None
    result = json.loads(path.read_text())
    if result["backend"] != "jax" or not result["valid"]:
        raise ValueError("the reference must be a valid JAX result")
    for field in ("claim_id", "model", "workload", "capacity"):
        if result[field] != manifest[field]:
            raise ValueError(f"reference {field} does not match the manifest")
    return result["correctness"]["output_token_ids"]


def _sample_memory(maximum: dict[str, int]) -> None:
    maximum["rss"] = max(maximum["rss"], _process_tree_rss_bytes(os.getpid()))
    maximum["device"] = max(maximum["device"], _device_used_bytes())


def run(
    backend_name: str,
    manifest: dict[str, Any],
    reference_path: Path | None,
) -> dict[str, Any]:
    prompts = prompt_rows(manifest)
    backend_module = importlib.import_module(f"benchmarks.backends.{backend_name}")
    memory = {"rss": 0, "device": 0}

    init_started = perf_counter()
    backend = backend_module.Backend(manifest, prompts)
    init_seconds = perf_counter() - init_started
    _sample_memory(memory)

    warmup_started = perf_counter()
    warmup = backend.warmup()
    warmup_seconds = perf_counter() - warmup_started
    for _ in range(manifest["measurement"]["warmup_repeats"]):
        control = backend.run_once()
    _sample_memory(memory)

    compiled_before = backend.compile_fingerprint()
    samples = []
    repeat_exact = True
    for _ in range(manifest["measurement"]["repeats"]):
        sample = backend.run_once()
        repeat_exact &= sample["output_token_ids"] == control["output_token_ids"]
        samples.append(
            {
                key: value
                for key, value in sample.items()
                if key != "output_token_ids"
            }
        )
        _sample_memory(memory)
    compiled_after = backend.compile_fingerprint()

    jit_growth = None
    if compiled_before is not None:
        jit_growth = len(set(compiled_after) - set(compiled_before))

    speeds = [sample["decode_tokens_per_second"] for sample in samples]
    ttfts = [sample["ttft_seconds"] for sample in samples]
    median_speed = median(speeds)
    relative_spread = (max(speeds) - min(speeds)) / median_speed
    reference_rows = _reference_rows(reference_path, manifest)
    reference_exact = (
        None
        if reference_rows is None
        else reference_rows == control["output_token_ids"]
    )

    invalid_reasons = []
    expected_decode_tokens = (
        manifest["workload"]["batch_size"]
        * (manifest["workload"]["output_tokens"] - 1)
    )
    if not repeat_exact:
        invalid_reasons.append("measured repeats produced different tokens")
    if reference_exact is False:
        invalid_reasons.append("backend tokens differ from the reference backend")
    if any(sample["decode_tokens"] != expected_decode_tokens for sample in samples):
        invalid_reasons.append("decode token count does not match the contract")
    if any(sample["decode_seconds"] <= 0 for sample in samples):
        invalid_reasons.append("decode duration is not positive")
    if relative_spread > manifest["measurement"]["max_relative_spread"]:
        invalid_reasons.append("decode throughput spread exceeds the contract")
    if backend_name == "jax" and jit_growth != 0:
        invalid_reasons.append("JAX compiled a new route during measurement")

    gpu_name, gpu_uuid, gpu_memory_mib, driver = _nvidia_smi(
        "name", "uuid", "memory.total", "driver_version"
    )
    backend_memory = backend.memory()
    return {
        "schema_version": 1,
        "claim_id": manifest["claim_id"],
        "backend": backend_name,
        "model": manifest["model"],
        "workload": manifest["workload"],
        "capacity": manifest["capacity"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "framework": backend.software(),
            "gpu": {
                "name": gpu_name,
                "uuid": gpu_uuid,
                "memory_bytes": int(gpu_memory_mib) * 1024 * 1024,
                "driver": driver,
            },
            "repository": _git_state(),
        },
        "timing": {
            "initialization_seconds": init_seconds,
            "warmup_seconds": warmup_seconds,
            "samples": samples,
            "median_ttft_seconds": median(ttfts),
            "median_decode_tokens_per_second": median_speed,
            "relative_spread": relative_spread,
            "decode_definition": (
                "tokens after the first / time from first token to completion"
            ),
        },
        "warmup": warmup,
        "correctness": {
            "repeat_exact": repeat_exact,
            "reference_exact": reference_exact,
            "output_sha256": _output_hash(control["output_token_ids"]),
            "output_token_ids": control["output_token_ids"],
            "measured_jit_cache_growth": jit_growth,
        },
        "memory": {
            "max_observed_process_tree_rss_bytes": memory["rss"],
            "max_observed_device_used_bytes": memory["device"],
            **backend_memory,
        },
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("jax", "vllm"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/claim.json")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run(args.backend, load_manifest(args.manifest), args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{args.backend}: {result['timing']['median_decode_tokens_per_second']:.2f} "
        f"decode tok/s, valid={result['valid']}"
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
