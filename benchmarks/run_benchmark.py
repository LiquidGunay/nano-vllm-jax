"""Run one backend against the committed benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
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
        "parity",
        "frameworks",
    }
    if set(manifest) != required:
        raise ValueError(f"manifest keys must be exactly {sorted(required)}")
    if manifest["schema_version"] != 3:
        raise ValueError("unsupported manifest schema")
    workload = manifest["workload"]
    if workload["batch_size"] != 1:
        raise ValueError("this artifact defines one B=1 benchmark")
    if workload["temperature"] != 0 or not workload["ignore_eos"]:
        raise ValueError("the benchmark requires greedy fixed-length decode")
    if workload["prefix_cache"]:
        raise ValueError("the benchmark requires a prefix miss")
    prompt = workload["prompt_token_ids"]
    if len(prompt) != workload["prompt_tokens"] or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prompt
    ):
        raise ValueError("prompt_token_ids must contain prompt_tokens nonnegative integers")
    speculation = manifest["speculation"]
    if set(speculation) != {"method", "draft_tokens"}:
        raise ValueError("speculation must define method and draft_tokens")
    if speculation["method"] != "mtp" or speculation["draft_tokens"] < 1:
        raise ValueError("the benchmark requires a positive-width MTP drafter")
    measurement = manifest["measurement"]
    if measurement["warmup_repeats"] < 1 or measurement["repeats"] < 1:
        raise ValueError("warmup and measured repeats must be positive")
    parity = manifest["parity"]
    if set(parity) != {"evidence", "sha256"}:
        raise ValueError("parity must define evidence and sha256")
    if Path(parity["evidence"]).name != parity["evidence"]:
        raise ValueError("parity evidence must be a filename beside the manifest")
    if len(parity["sha256"]) != 64:
        raise ValueError("parity evidence must have a SHA-256 digest")
    return manifest


def load_parity_evidence(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the one content-addressed output equivalence admitted by the contract."""

    parity = manifest["parity"]
    evidence_path = path.parent / parity["evidence"]
    if _file_hash(evidence_path) != parity["sha256"]:
        raise ValueError("parity evidence digest does not match the manifest")
    evidence = json.loads(evidence_path.read_text())
    required = {
        "schema_version",
        "benchmark_id",
        "reference_output_sha256",
        "variant_output_sha256",
        "mismatches",
        "diagnostic",
        "limits",
    }
    if set(evidence) != required or evidence["schema_version"] != 1:
        raise ValueError("unsupported parity evidence schema")
    if evidence["benchmark_id"] != manifest["benchmark_id"]:
        raise ValueError("parity evidence benchmark does not match the manifest")
    if evidence["reference_output_sha256"] == evidence["variant_output_sha256"]:
        raise ValueError("parity evidence must describe two different outputs")

    limits = evidence["limits"]
    if set(limits) != {
        "max_bidirectional_kl",
        "max_jensen_shannon",
        "max_total_variation",
        "max_mismatches",
    }:
        raise ValueError("parity evidence limits are incomplete")
    if any(
        not isinstance(limits[name], (int, float))
        or isinstance(limits[name], bool)
        or not math.isfinite(limits[name])
        or limits[name] <= 0
        for name in ("max_bidirectional_kl", "max_jensen_shannon", "max_total_variation")
    ):
        raise ValueError("parity divergence limits must be finite and positive")
    if isinstance(limits["max_mismatches"], bool) or limits["max_mismatches"] < 1:
        raise ValueError("max_mismatches must be positive")

    mismatches = evidence["mismatches"]
    mismatch_fields = {"row", "output_index", "reference_token", "variant_token"}
    if not mismatches or len(mismatches) > limits["max_mismatches"]:
        raise ValueError("parity evidence mismatch count exceeds its limit")
    if any(
        set(item) != mismatch_fields
        or any(isinstance(item[name], bool) or not isinstance(item[name], int) for name in item)
        or any(item[name] < 0 for name in item)
        for item in mismatches
    ):
        raise ValueError("parity mismatches must contain nonnegative integer coordinates and ids")

    diagnostic = evidence["diagnostic"]
    if not diagnostic.get("full_vocabulary"):
        raise ValueError("parity evidence must cover the full vocabulary")
    metrics = {
        "kl_reference_to_variant": limits["max_bidirectional_kl"],
        "kl_variant_to_reference": limits["max_bidirectional_kl"],
        "jensen_shannon": limits["max_jensen_shannon"],
        "total_variation": limits["max_total_variation"],
    }
    for name, upper in metrics.items():
        value = diagnostic.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= upper
        ):
            raise ValueError(f"parity evidence {name} exceeds its limit")

    mismatch = mismatches[0]
    reference = str(mismatch["reference_token"])
    variant = str(mismatch["variant_token"])
    reference_bins = diagnostic["reference_bf16_bins"]
    variant_bins = diagnostic["variant_bf16_bins"]
    if reference_bins[reference] != reference_bins[variant]:
        raise ValueError("reference BF16 logits do not demonstrate a tie")
    if variant_bins[variant] <= variant_bins[reference]:
        raise ValueError("variant BF16 logits do not select the variant token")
    return evidence


def prompt_rows(manifest: dict[str, Any]) -> list[list[int]]:
    workload = manifest["workload"]
    return [list(workload["prompt_token_ids"])]


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
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    return {"commit": commit, "dirty": dirty}


def _output_hash(rows: list[list[int]]) -> str:
    encoded = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_rows(
    path: Path | None,
    manifest: dict[str, Any],
    benchmark_sha256: str,
    repository: dict[str, Any],
) -> list[list[int]] | None:
    if path is None:
        return None
    result = json.loads(path.read_text())
    if result["backend"] != "jax" or not result["valid"]:
        raise ValueError("the reference must be a valid JAX result")
    if result["route"] != "base":
        raise ValueError("the reference must use base JAX decode")
    if result.get("benchmark_sha256") != benchmark_sha256:
        raise ValueError("reference benchmark manifest does not match")
    if result.get("environment", {}).get("repository") != repository:
        raise ValueError("reference implementation commit does not match")
    for field in (
        "benchmark_id",
        "model",
        "workload",
        "speculation",
        "capacity",
        "parity",
    ):
        if result[field] != manifest[field]:
            raise ValueError(f"reference {field} does not match the manifest")
    return result["correctness"]["output_token_ids"]


def adjudicate_reference(
    evidence: dict[str, Any] | None,
    reference_rows: list[list[int]] | None,
    output_rows: list[list[int]],
    *,
    evidence_sha256: str,
) -> dict[str, Any] | None:
    """Accept only the finite, measured token variant named by the evidence."""

    if evidence is None or reference_rows is None:
        return None
    reference_hash = _output_hash(reference_rows)
    output_hash = _output_hash(output_rows)
    allowed_hashes = {
        evidence["reference_output_sha256"],
        evidence["variant_output_sha256"],
    }
    if reference_hash == output_hash or {reference_hash, output_hash} != allowed_hashes:
        return None
    if len(reference_rows) != len(output_rows):
        return None
    observed = []
    for row, (reference, output) in enumerate(zip(reference_rows, output_rows)):
        if len(reference) != len(output):
            return None
        observed.extend(
            {
                "row": row,
                "output_index": index,
                "reference_token": reference_token,
                "variant_token": variant_token,
            }
            for index, (reference_token, variant_token) in enumerate(zip(reference, output))
            if reference_token != variant_token
        )
    expected = evidence["mismatches"]
    if reference_hash == evidence["variant_output_sha256"]:
        expected = [
            {
                **item,
                "reference_token": item["variant_token"],
                "variant_token": item["reference_token"],
            }
            for item in expected
        ]
    if observed != expected:
        return None
    return {
        "evidence_sha256": evidence_sha256,
        "output_sha256": output_hash,
        "equivalent_output_sha256": sorted(allowed_hashes),
        "mismatches": observed,
        "diagnostic": evidence["diagnostic"],
        "limits": evidence["limits"],
    }


def _validate_reference_role(
    backend_name: str,
    route: str,
    reference_path: Path | None,
) -> None:
    is_jax_base = backend_name == "jax" and route == "base"
    if is_jax_base and reference_path is not None:
        raise ValueError("JAX base is the canonical reference")
    if not is_jax_base and reference_path is None:
        raise ValueError("JAX MTP and vLLM routes require a JAX-base reference")


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
    reference_adjudicated: bool = False,
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
    if reference_exact is False and not reference_adjudicated:
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
            drafted == 0 or (backend_name == "jax" and (verified is None or verified == 0))
        ):
            reasons.append("MTP decode did not report target-verified drafts")
            break
    return reasons


def run(
    backend_name: str,
    route: str,
    manifest: dict[str, Any],
    benchmark_sha256: str,
    reference_path: Path | None,
    parity_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_reference_role(backend_name, route, reference_path)
    prompts = prompt_rows(manifest)
    memory = {"rss": 0, "device": 0}
    repository = _git_state()
    if repository["dirty"]:
        raise RuntimeError("the benchmark requires a clean repository")
    reference_rows = _reference_rows(
        reference_path,
        manifest,
        benchmark_sha256,
        repository,
    )
    backend_module = importlib.import_module(f"benchmarks.backends.{backend_name}")

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
        samples.append({key: value for key, value in sample.items() if key != "output_token_ids"})
        _sample_memory(memory)
    route_cache_after = backend.route_cache_fingerprint()

    route_cache_growth = None
    if route_cache_before is not None:
        route_cache_growth = len(set(route_cache_after) - set(route_cache_before))

    speeds = [sample["decode_tokens_per_second"] for sample in samples]
    ttfts = [sample["ttft_seconds"] for sample in samples]
    median_speed = median(speeds)
    relative_spread = (max(speeds) - min(speeds)) / median_speed
    reference_exact = (
        None if reference_rows is None else reference_rows == control["output_token_ids"]
    )
    parity_adjudication = (
        None
        if reference_exact is not False
        else adjudicate_reference(
            parity_evidence,
            reference_rows,
            control["output_token_ids"],
            evidence_sha256=manifest["parity"]["sha256"],
        )
    )

    reasons = invalid_reasons(
        backend_name,
        route,
        manifest,
        samples,
        control["output_token_ids"],
        repeat_exact=repeat_exact,
        reference_exact=reference_exact,
        reference_adjudicated=parity_adjudication is not None,
        route_cache_growth=route_cache_growth,
    )

    gpu_name, gpu_uuid, gpu_memory_mib, driver = _nvidia_smi(
        "name", "uuid", "memory.total", "driver_version"
    )
    backend_memory = backend.memory()
    return {
        "schema_version": 3,
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_sha256": benchmark_sha256,
        "backend": backend_name,
        "route": route,
        "model": manifest["model"],
        "workload": manifest["workload"],
        "speculation": manifest["speculation"],
        "capacity": manifest["capacity"],
        "parity": manifest["parity"],
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
            "repository": repository,
        },
        "timing": {
            "initialization_seconds": init_seconds,
            "warmup_seconds": warmup_seconds,
            "samples": samples,
            "median_ttft_seconds": median(ttfts),
            "median_decode_tokens_per_second": median_speed,
            "relative_spread": relative_spread,
            "decode_definition": ("tokens after the first / time from first token to completion"),
        },
        "warmup": warmup,
        "correctness": {
            "repeat_exact": repeat_exact,
            "reference_exact": reference_exact,
            "reference_adjudicated": parity_adjudication is not None,
            "parity_evidence": parity_adjudication,
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
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/benchmark.json")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    parity_evidence = load_parity_evidence(args.manifest, manifest)
    result = run(
        args.backend,
        args.route,
        manifest,
        _file_hash(args.manifest),
        args.reference,
        parity_evidence,
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
