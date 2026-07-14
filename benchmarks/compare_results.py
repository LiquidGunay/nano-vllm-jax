"""Summarize two valid benchmark results without enforcing a speed target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_results(
    jax_result: dict[str, Any], vllm_result: dict[str, Any]
) -> dict[str, Any]:
    if jax_result["backend"] != "jax" or vllm_result["backend"] != "vllm":
        raise ValueError("comparison requires JAX and vLLM results")
    if not jax_result["valid"] or not vllm_result["valid"]:
        raise ValueError("comparison requires valid benchmark results")
    for field in ("benchmark_id", "model", "workload", "capacity"):
        if jax_result[field] != vllm_result[field]:
            raise ValueError(f"result {field} values differ")
    jax_hash = jax_result["correctness"]["output_sha256"]
    vllm_hash = vllm_result["correctness"]["output_sha256"]
    if jax_hash != vllm_hash:
        raise ValueError("result tokens differ")

    jax_speed = jax_result["timing"]["median_decode_tokens_per_second"]
    vllm_speed = vllm_result["timing"]["median_decode_tokens_per_second"]
    jax_gpu = jax_result["environment"]["gpu"]
    vllm_gpu = vllm_result["environment"]["gpu"]
    return {
        "schema_version": 1,
        "benchmark_id": jax_result["benchmark_id"],
        "jax_decode_tokens_per_second": jax_speed,
        "vllm_decode_tokens_per_second": vllm_speed,
        "jax_over_vllm_decode_ratio": jax_speed / vllm_speed,
        "output_sha256": jax_hash,
        "same_gpu": jax_gpu["uuid"] == vllm_gpu["uuid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_results(
        json.loads(args.jax.read_text()), json.loads(args.vllm.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        "JAX/vLLM decode ratio: "
        f"{comparison['jax_over_vllm_decode_ratio']:.3f}x"
    )


if __name__ == "__main__":
    main()
