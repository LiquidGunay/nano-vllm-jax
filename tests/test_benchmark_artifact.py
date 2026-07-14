import json
from pathlib import Path

import pytest

from benchmarks.compare_results import compare_results
from benchmarks.run_benchmark import (
    _nvidia_smi,
    invalid_reasons,
    load_manifest,
    prompt_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _result(backend: str, speed: float, output_hash: str = "same") -> dict:
    return {
        "backend": backend,
        "valid": True,
        "benchmark_id": "benchmark",
        "model": {"id": "model"},
        "workload": {"batch_size": 1},
        "capacity": {"max_model_len": 8},
        "correctness": {"output_sha256": output_hash},
        "timing": {"median_decode_tokens_per_second": speed},
        "environment": {"gpu": {"uuid": "GPU-0"}},
    }


def test_benchmark_is_one_fixed_greedy_b1_workload():
    manifest = load_manifest(ROOT / "benchmarks/benchmark.json")
    workload = manifest["workload"]
    prompts = prompt_rows(manifest)

    assert workload["batch_size"] == 1
    assert workload["temperature"] == 0
    assert workload["ignore_eos"]
    assert not workload["prefix_cache"]
    assert len(prompts) == 1
    assert len(prompts[0]) == workload["prompt_tokens"]
    assert prompts[0][:4] == [1, 2, 3, 4]


def test_manifest_matches_the_vllm_requirement():
    manifest = load_manifest(ROOT / "benchmarks/benchmark.json")
    requirement = (ROOT / "benchmarks/vllm-requirements.txt").read_text().strip()

    assert requirement == f"vllm=={manifest['frameworks']['vllm']}"


def test_nvidia_smi_uses_the_selected_physical_gpu(monkeypatch):
    command = []

    def check_output(args, *, text):
        command.extend(args)
        return "NVIDIA A10G, GPU-id\n"

    monkeypatch.setenv("NANO_VLLM_JAX_BENCHMARK_GPU", "2")
    monkeypatch.setattr("subprocess.check_output", check_output)

    assert _nvidia_smi("name", "uuid") == ["NVIDIA A10G", "GPU-id"]
    assert command[1:3] == ["-i", "2"]


def test_validity_rejects_unstable_or_nonmatching_tokens():
    manifest = load_manifest(ROOT / "benchmarks/benchmark.json")
    samples = [
        {
            "decode_tokens": 63,
            "decode_seconds": 1.0,
            "decode_tokens_per_second": 63.0,
        }
    ] * 3

    assert not invalid_reasons(
        "jax",
        manifest,
        samples,
        repeat_exact=True,
        reference_exact=None,
        route_cache_growth=0,
    )
    reasons = invalid_reasons(
        "vllm",
        manifest,
        samples,
        repeat_exact=False,
        reference_exact=False,
        route_cache_growth=None,
    )
    assert "measured repeats produced different tokens" in reasons
    assert "backend tokens differ from the reference backend" in reasons
    reasons = invalid_reasons(
        "jax",
        manifest,
        samples,
        repeat_exact=True,
        reference_exact=None,
        route_cache_growth=1,
    )
    assert "JAX added an executor route-cache entry during measurement" in reasons


def test_comparison_reports_speed_and_hardware_without_gating_them():
    vllm = _result("vllm", 60.0)
    vllm["environment"]["gpu"]["uuid"] = "GPU-1"
    comparison = compare_results(_result("jax", 40.0), vllm)

    assert comparison["jax_over_vllm_decode_ratio"] == pytest.approx(2 / 3)
    assert not comparison["same_gpu"]


def test_comparison_requires_matching_outputs():
    with pytest.raises(ValueError, match="tokens differ"):
        compare_results(
            _result("jax", 40.0, "jax"), _result("vllm", 60.0, "vllm")
        )


def test_recorded_result_matches_the_manifest():
    manifest = load_manifest(ROOT / "benchmarks/benchmark.json")
    recorded = json.loads((ROOT / "benchmarks/recorded_result.json").read_text())
    jax = recorded["results"]["jax"]
    vllm = recorded["results"]["vllm"]

    assert recorded["benchmark_id"] == manifest["benchmark_id"]
    assert recorded["comparison"]["output_exact"]
    assert recorded["comparison"]["jax_over_vllm_decode_ratio"] == pytest.approx(
        jax["median_decode_tokens_per_second"]
        / vllm["median_decode_tokens_per_second"]
    )
