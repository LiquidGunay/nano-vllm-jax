"""Summarize the fixed base/MTP matrix without enforcing a speed target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_results(
    jax_base: dict[str, Any],
    jax_mtp: dict[str, Any],
    vllm_base: dict[str, Any],
    vllm_mtp: dict[str, Any],
) -> dict[str, Any]:
    results = {
        "jax_base": jax_base,
        "jax_mtp": jax_mtp,
        "vllm_base": vllm_base,
        "vllm_mtp": vllm_mtp,
    }
    for name, result in results.items():
        backend, route = name.split("_")
        if result["backend"] != backend or result["route"] != route:
            raise ValueError(f"{name} has the wrong backend or route")
        if not result["valid"]:
            raise ValueError(f"{name} is not a valid benchmark result")

    reference = jax_base
    fields = (
        "benchmark_id",
        "benchmark_sha256",
        "model",
        "workload",
        "speculation",
        "capacity",
        "parity",
    )
    if any(result[field] != reference[field] for result in results.values() for field in fields):
        raise ValueError("result benchmark contracts differ")

    reference_hash = jax_base["correctness"]["output_sha256"]
    hashes = {reference_hash}
    adjudicated_routes = []
    for name, result in results.items():
        output_hash = result["correctness"]["output_sha256"]
        hashes.add(output_hash)
        if output_hash == reference_hash:
            continue
        parity = result["correctness"].get("parity_evidence")
        if (
            not result["correctness"].get("reference_adjudicated")
            or parity is None
            or parity["output_sha256"] != output_hash
            or parity["evidence_sha256"] != result["parity"]["sha256"]
        ):
            raise ValueError("result tokens differ without parity evidence")
        adjudicated_routes.append(name)

    speeds = {
        name: result["timing"]["median_decode_tokens_per_second"]
        for name, result in results.items()
    }
    gpu_uuids = {result["environment"]["gpu"]["uuid"] for result in results.values()}
    if len(gpu_uuids) != 1:
        raise ValueError("results were not measured on the same GPU")
    repositories = [result["environment"]["repository"] for result in results.values()]
    if (
        any(repository["dirty"] for repository in repositories)
        or len({repository["commit"] for repository in repositories}) != 1
    ):
        raise ValueError("results were not measured at the same clean commit")
    return {
        "schema_version": 3,
        "benchmark_id": reference["benchmark_id"],
        "decode_tokens_per_second": speeds,
        "ratios": {
            "jax_base_over_vllm_base": speeds["jax_base"] / speeds["vllm_base"],
            "jax_mtp_over_jax_base": speeds["jax_mtp"] / speeds["jax_base"],
            "vllm_mtp_over_vllm_base": speeds["vllm_mtp"] / speeds["vllm_base"],
            "jax_mtp_over_vllm_base": speeds["jax_mtp"] / speeds["vllm_base"],
            "jax_mtp_over_vllm_mtp": speeds["jax_mtp"] / speeds["vllm_mtp"],
        },
        "output_sha256": reference_hash,
        "observed_output_sha256": sorted(hashes),
        "output_exact": len(hashes) == 1,
        "adjudicated_routes": adjudicated_routes,
        "gpu_uuid": gpu_uuids.pop(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("jax-base", "jax-mtp", "vllm-base", "vllm-mtp"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_results(
        *(
            json.loads(getattr(args, name).read_text())
            for name in (
                "jax_base",
                "jax_mtp",
                "vllm_base",
                "vllm_mtp",
            )
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        "JAX MTP/base: "
        f"{comparison['ratios']['jax_mtp_over_jax_base']:.3f}x; "
        "JAX MTP/vLLM base: "
        f"{comparison['ratios']['jax_mtp_over_vllm_base']:.3f}x"
    )


if __name__ == "__main__":
    main()
