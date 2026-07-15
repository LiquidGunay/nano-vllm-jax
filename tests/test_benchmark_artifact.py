import hashlib
import json
from pathlib import Path
from statistics import median

import pytest

from benchmarks.compare_results import compare_results
from benchmarks.run_benchmark import (
    _nvidia_smi,
    _reference_rows,
    _validate_reference_role,
    invalid_reasons,
    load_manifest,
    prompt_rows,
    run,
)


ROOT = Path(__file__).resolve().parents[1]


def _result(
    backend: str,
    route: str,
    speed: float,
    output_hash: str = "same",
) -> dict:
    return {
        "backend": backend,
        "route": route,
        "valid": True,
        "benchmark_id": "benchmark",
        "benchmark_sha256": "manifest",
        "model": {"id": "model"},
        "workload": {"batch_size": 1},
        "speculation": {"method": "mtp", "draft_tokens": 2},
        "capacity": {"max_model_len": 8},
        "correctness": {"output_sha256": output_hash},
        "timing": {"median_decode_tokens_per_second": speed},
        "environment": {
            "gpu": {"uuid": "GPU-0"},
            "repository": {"commit": "commit", "dirty": False},
        },
    }


def _sample(*, drafted: int = 0, accepted: int = 0, verified: int | None = 0):
    return {
        "decode_tokens": 63,
        "decode_seconds": 1.0,
        "decode_tokens_per_second": 63.0,
        "speculation": {
            "draft_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "verified_target_tokens": verified,
        },
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
    assert prompts[0][:4] == [814, 20139, 3069, 63520]
    assert prompts[0][-4:] == [5272, 799, 3010, 13]
    assert manifest["speculation"] == {"method": "mtp", "draft_tokens": 2}


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
    samples = [_sample()] * 3

    assert not invalid_reasons(
        "jax",
        "base",
        manifest,
        samples,
        [[0] * 64],
        repeat_exact=True,
        reference_exact=None,
        route_cache_growth=0,
    )
    reasons = invalid_reasons(
        "vllm",
        "base",
        manifest,
        samples,
        [[0] * 64],
        repeat_exact=False,
        reference_exact=False,
        route_cache_growth=None,
    )
    assert "measured repeats produced different tokens" in reasons
    assert "backend tokens differ from the reference backend" in reasons
    reasons = invalid_reasons(
        "jax",
        "base",
        manifest,
        samples,
        [[0] * 64],
        repeat_exact=True,
        reference_exact=None,
        route_cache_growth=1,
    )
    assert "JAX added an executor route-cache entry during measurement" in reasons


def test_validity_rejects_the_wrong_final_output_length():
    manifest = load_manifest(ROOT / "benchmarks/benchmark.json")
    samples = [_sample()] * 3

    reasons = invalid_reasons(
        "jax",
        "base",
        manifest,
        samples,
        [[0] * 63],
        repeat_exact=True,
        reference_exact=None,
        route_cache_growth=0,
    )
    assert "output token count does not match the contract" in reasons


def test_validity_requires_verified_mtp_drafts():
    manifest = load_manifest(ROOT / "benchmarks/benchmark.json")
    samples = [_sample()] * 3

    reasons = invalid_reasons(
        "jax",
        "mtp",
        manifest,
        samples,
        [[0] * 64],
        repeat_exact=True,
        reference_exact=True,
        route_cache_growth=0,
    )

    assert "MTP decode did not report target-verified drafts" in reasons


def test_comparison_reports_the_base_and_mtp_ratios():
    comparison = compare_results(
        _result("jax", "base", 40.0),
        _result("jax", "mtp", 60.0),
        _result("vllm", "base", 50.0),
        _result("vllm", "mtp", 90.0),
    )

    assert comparison["ratios"]["jax_mtp_over_jax_base"] == pytest.approx(1.5)
    assert comparison["ratios"]["jax_mtp_over_vllm_base"] == pytest.approx(1.2)
    assert comparison["gpu_uuid"] == "GPU-0"


def test_comparison_rejects_mixed_gpus_and_commits():
    results = [
        _result("jax", "base", 40.0),
        _result("jax", "mtp", 60.0),
        _result("vllm", "base", 50.0),
        _result("vllm", "mtp", 90.0),
    ]
    results[-1]["environment"]["gpu"]["uuid"] = "GPU-1"
    with pytest.raises(ValueError, match="same GPU"):
        compare_results(*results)

    results[-1]["environment"]["gpu"]["uuid"] = "GPU-0"
    results[-1]["environment"]["repository"]["commit"] = "other"
    with pytest.raises(ValueError, match="same clean commit"):
        compare_results(*results)


def test_reference_requires_the_same_manifest_and_commit(tmp_path):
    result = _result("jax", "base", 40.0)
    result["correctness"]["output_token_ids"] = [[1, 2]]
    path = tmp_path / "jax-base.json"
    path.write_text(json.dumps(result))
    manifest = {
        name: result[name]
        for name in ("benchmark_id", "model", "workload", "speculation", "capacity")
    }
    repository = {"commit": "commit", "dirty": False}

    assert _reference_rows(path, manifest, "manifest", repository) == [[1, 2]]
    result["route"] = "mtp"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="base JAX"):
        _reference_rows(path, manifest, "manifest", repository)
    result["route"] = "base"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="manifest"):
        _reference_rows(path, manifest, "other", repository)
    with pytest.raises(ValueError, match="commit"):
        _reference_rows(path, manifest, "manifest", {**repository, "commit": "other"})


def test_only_jax_base_can_omit_the_reference(monkeypatch):
    _validate_reference_role("jax", "base", None)
    monkeypatch.setattr(
        "benchmarks.run_benchmark.importlib.import_module",
        lambda _name: pytest.fail("backend imported before reference validation"),
    )
    for backend, route in (("jax", "mtp"), ("vllm", "base"), ("vllm", "mtp")):
        with pytest.raises(ValueError, match="require a JAX-base reference"):
            run(backend, route, {}, "manifest", None)


def test_jax_base_rejects_an_unnecessary_reference(tmp_path):
    with pytest.raises(ValueError, match="canonical reference"):
        run("jax", "base", {}, "manifest", tmp_path / "reference.json")


def test_comparison_requires_matching_outputs():
    with pytest.raises(ValueError, match="tokens differ"):
        compare_results(
            _result("jax", "base", 40.0, "jax"),
            _result("jax", "mtp", 60.0),
            _result("vllm", "base", 50.0),
            _result("vllm", "mtp", 70.0, "vllm"),
        )


def test_recorded_result_matches_the_manifest():
    benchmark_path = ROOT / "benchmarks/benchmark.json"
    manifest = load_manifest(benchmark_path)
    recorded = json.loads((ROOT / "benchmarks/recorded_result.json").read_text())
    results = recorded["results"]

    assert recorded["benchmark_id"] == manifest["benchmark_id"]
    assert (
        recorded["benchmark_sha256"]
        == hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    )
    assert recorded["hardware"]["gpu_uuid"].startswith("GPU-")
    assert recorded["comparison"]["output_exact"]
    speeds = {}
    for name, result in results.items():
        samples = result["samples"]
        sample_speeds = [sample["decode_tokens_per_second"] for sample in samples]
        sample_ttfts = [sample["ttft_seconds"] for sample in samples]
        assert len(samples) == manifest["measurement"]["repeats"]
        assert result["median_decode_tokens_per_second"] == median(sample_speeds)
        assert result["median_ttft_seconds"] == median(sample_ttfts)
        assert result["relative_spread"] == pytest.approx(
            (max(sample_speeds) - min(sample_speeds)) / median(sample_speeds)
        )
        speeds[name] = result["median_decode_tokens_per_second"]

    expected_ratios = {
        "jax_base_over_vllm_base": speeds["jax_base"] / speeds["vllm_base"],
        "jax_mtp_over_jax_base": speeds["jax_mtp"] / speeds["jax_base"],
        "vllm_mtp_over_vllm_base": speeds["vllm_mtp"] / speeds["vllm_base"],
        "jax_mtp_over_vllm_base": speeds["jax_mtp"] / speeds["vllm_base"],
        "jax_mtp_over_vllm_mtp": speeds["jax_mtp"] / speeds["vllm_mtp"],
    }
    for name, ratio in expected_ratios.items():
        assert recorded["comparison"][name] == pytest.approx(ratio)
