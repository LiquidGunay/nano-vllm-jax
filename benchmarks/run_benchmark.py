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
ROUTES = ("base", "mtp")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    required = {
        "schema_version",
        "benchmark_id",
        "model",
        "workload",
        "speculation",
        "capacity",
        "measurement",
        "frameworks",
    }
    if set(manifest) != required:
        raise ValueError(f"manifest keys must be exactly {sorted(required)}")
    if manifest["schema_version"] != 2:
        raise ValueError("unsupported manifest schema")
    workload = manifest["workload"]
    if workload["batch_size"] != 1:
        raise ValueError("this artifact defines one B=1 benchmark")
    if workload["temperature"] != 0 or not workload["ignore_eos"]:
        raise ValueError("the benchmark requires greedy fixed-length decode")
    if workload["prefix_cache"]:
        raise ValueError("the benchmark requires a prefix miss")
    speculation = manifest["speculation"]
    if set(speculation) != {"method", "draft_tokens"}:
        raise ValueError("speculation must define method and draft_tokens")
    if speculation["method"] != "mtp" or speculation["draft_tokens"] < 1:
        raise ValueError("the benchmark requires a positive-width MTP drafter")
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
    gpu = os.environ.get("NANO_VLLM_JAX_BENCHMARK_GPU", "0")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
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
    if result["route"] != "base":
        raise ValueError("the reference must use base JAX decode")
    for field in (
        "benchmark_id",
        "model",
        "workload",
        "speculation",
        "capacity",
    ):
        if result[field] != manifest[field]:
            raise ValueError(f"reference {field} does not match the manifest")
    return result["correctness"]["output_token_ids"]


def _sample_memory(maximum: dict[str, int]) -> None:
    maximum["rss"] = max(maximum["rss"], _process_tree_rss_bytes(os.getpid()))
    maximum["device"] = max(maximum["device"], _device_used_bytes())


def invalid_reasons(
    backend_name: str,
    route: str,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    output_rows: list[list[int]],
    *,
    repeat_exact: bool,
    reference_exact: bool | None,
    route_cache_growth: int | None,
) -> list[str]:
    reasons = []
    expected_tokens = manifest["workload"]["batch_size"] * (
        manifest["workload"]["output_tokens"] - 1
    )
    speeds = [sample["decode_tokens_per_second"] for sample in samples]
    spread = (max(speeds) - min(speeds)) / median(speeds)
    if not repeat_exact:
        reasons.append("measured repeats produced different tokens")
    if reference_exact is False:
        reasons.append("backend tokens differ from the reference backend")
    if any(sample["decode_tokens"] != expected_tokens for sample in samples):
        reasons.append("decode token count does not match the contract")
    workload = manifest["workload"]
    if len(output_rows) != workload["batch_size"] or any(
        len(row) != workload["output_tokens"] for row in output_rows
    ):
        reasons.append("output token count does not match the contract")
    if any(sample["decode_seconds"] <= 0 for sample in samples):
        reasons.append("decode duration is not positive")
    if spread > manifest["measurement"]["max_relative_spread"]:
        reasons.append("decode throughput spread exceeds the contract")
    if backend_name == "jax" and route_cache_growth != 0:
        reasons.append("JAX added an executor route-cache entry during measurement")
    for sample in samples:
        stats = sample["speculation"]
        drafted = stats["draft_tokens"]
        accepted = stats["accepted_draft_tokens"]
        verified = stats["verified_target_tokens"]
        if not 0 <= accepted <= drafted:
            reasons.append("speculative acceptance counters are inconsistent")
            break
        if route == "base" and (drafted or accepted or verified not in (0, None)):
            reasons.append("base decode reported speculative work")
            break
        if route == "mtp" and (
            drafted == 0
            or (backend_name == "jax" and (verified is None or verified == 0))
        ):
            reasons.append("MTP decode did not report target-verified drafts")
            break
    return reasons


def run(
    backend_name: str,
    route: str,
    manifest: dict[str, Any],
    reference_path: Path | None,
) -> dict[str, Any]:
    prompts = prompt_rows(manifest)
    backend_module = importlib.import_module(f"benchmarks.backends.{backend_name}")
    memory = {"rss": 0, "device": 0}

    init_started = perf_counter()
    backend = backend_module.Backend(manifest, prompts, route)
    init_seconds = perf_counter() - init_started
    _sample_memory(memory)

    warmup_started = perf_counter()
    warmup = backend.warmup()
    warmup_seconds = perf_counter() - warmup_started
    for _ in range(manifest["measurement"]["warmup_repeats"]):
        control = backend.run_once()
    _sample_memory(memory)

    route_cache_before = backend.route_cache_fingerprint()
    samples = []
    repeat_exact = True
    for _ in range(manifest["measurement"]["repeats"]):
        sample = backend.run_once()
        repeat_exact &= sample["output_token_ids"] == control["output_token_ids"]
        samples.append(
            {key: value for key, value in sample.items() if key != "output_token_ids"}
        )
        _sample_memory(memory)
    route_cache_after = backend.route_cache_fingerprint()

    route_cache_growth = None
    if route_cache_before is not None:
        route_cache_growth = len(set(route_cache_after) - set(route_cache_before))

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

    reasons = invalid_reasons(
        backend_name,
        route,
        manifest,
        samples,
        control["output_token_ids"],
        repeat_exact=repeat_exact,
        reference_exact=reference_exact,
        route_cache_growth=route_cache_growth,
    )

    gpu_name, gpu_uuid, gpu_memory_mib, driver = _nvidia_smi(
        "name", "uuid", "memory.total", "driver_version"
    )
    backend_memory = backend.memory()
    return {
        "schema_version": 2,
        "benchmark_id": manifest["benchmark_id"],
        "backend": backend_name,
        "route": route,
        "model": manifest["model"],
        "workload": manifest["workload"],
        "speculation": manifest["speculation"],
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
            "measured_executor_route_cache_growth": route_cache_growth,
        },
        "memory": {
            "max_observed_process_tree_rss_bytes": memory["rss"],
            "max_observed_device_used_bytes": memory["device"],
            **backend_memory,
        },
        "valid": not reasons,
        "invalid_reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("jax", "vllm"))
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "benchmarks/benchmark.json"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run(
        args.backend,
        args.route,
        load_manifest(args.manifest),
        args.reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{args.backend}-{args.route}: "
        f"{result['timing']['median_decode_tokens_per_second']:.2f} "
        f"decode tok/s, valid={result['valid']}"
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
