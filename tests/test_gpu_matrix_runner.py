"""Tests for GPU matrix summary helpers."""

import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.run_gpu_matrix import (
    CONFIG_DIR,
    PROFILE_NEEDLES,
    REPO_ROOT,
    WORKLOADS,
    _aggregate_repeats,
    _benchmark_acceptance_summary,
    _configured_workload_reference,
    _cuda_device_preflight,
    _reference_for,
    _find_local_vllm_reference,
    _jax_command,
    _runtime_env,
    _validate_summary_shape,
)


def _write_workload_artifact(path, workload):
    path.write_text(
        (
            '{"run_config": {"input_lens": %s, "output_len": %d, '
            '"prompt_suite": "%s"}}\n'
        )
        % (
            [int(part) for part in workload.input_lens.split(",")],
            workload.output_len,
            workload.prompt_suite,
        ),
        encoding="utf-8",
    )


def _minimal_schema():
    return {
        "schema_version": 1,
        "required": [
            "schema_version",
            "created_at_utc",
            "dry_run",
            "repeats",
            "run_dir",
            "output_json",
            "configs",
            "workloads",
            "required_metrics",
            "matrix",
            "vllm_references",
            "comparisons",
            "acceptance",
        ],
    }


def _minimal_summary():
    return {
        "schema_version": 1,
        "created_at_utc": "20260526_000000",
        "dry_run": True,
        "repeats": 1,
        "run_dir": "results/gpu_matrix_runs/test",
        "output_json": "results/gpu_matrix_test.json",
        "configs": ["gpu_paged_default"],
        "workloads": ["hetero8"],
        "required_metrics": [],
        "matrix": {
            "hetero8": {
                "gpu_paged_default": {
                    "config": {},
                    "repeats": [
                        {
                            "repeat": 1,
                            "artifact": "artifact.json",
                            "reference_json": None,
                            "reference_source": "none",
                            "run": {"status": "dry_run"},
                            "metrics": None,
                        }
                    ],
                    "aggregate": {
                        "repeat_count": 1,
                        "tokens_per_second_median": None,
                        "ttft_ms_p50_median": None,
                        "ttft_ms_p95_median": None,
                        "itl_ms_p50_median": None,
                        "itl_ms_p95_median": None,
                        "first_forward_step_token_ids_jit_ms_median": None,
                        "all_correct": False,
                        "all_exact_generated_token_match": False,
                        "all_correctness_checked": False,
                    },
                }
            }
        },
        "vllm_references": {"hetero8": {"source": "none"}},
        "comparisons": {"hetero8": {"gpu_paged_default": {}}},
        "acceptance": {
            "hetero8": {
                "gpu_paged_default": {
                    "checks": {
                        "minimum_repeats": False,
                        "correctness_checked": False,
                        "exact_generated_token_match": False,
                        "jax_performance_present": False,
                        "jax_latency_present": False,
                        "first_forward_step_present": False,
                        "vllm_reference_present": False,
                        "profile_counters_present": False,
                    },
                    "speed_claim_ready": False,
                    "target_vllm_ratio": 0.75,
                    "target_vllm_ratio_met": False,
                    "missing_profile_counters": [],
                    "notes": "not enough evidence for a performance claim",
                }
            }
        },
    }


def test_aggregate_repeats_does_not_treat_missing_correctness_as_correct():
    aggregate = _aggregate_repeats([{"metrics": None}])

    assert not aggregate["all_correct"]
    assert not aggregate["all_exact_generated_token_match"]
    assert not aggregate["all_correctness_checked"]


def test_validate_summary_shape_accepts_minimal_summary():
    _validate_summary_shape(_minimal_summary(), _minimal_schema())


def test_validate_summary_shape_rejects_missing_acceptance():
    summary = _minimal_summary()
    del summary["acceptance"]["hetero8"]["gpu_paged_default"]["checks"]["profile_counters_present"]

    with pytest.raises(ValueError, match="profile_counters_present"):
        _validate_summary_shape(summary, _minimal_schema())


def test_aggregate_repeats_requires_exact_full_length_match():
    aggregate = _aggregate_repeats(
        [
            {
                "metrics": {
                    "performance": {
                        "tokens_per_second": 10.0,
                        "ttft_ms_p50": 1.0,
                        "ttft_ms_p95": 1.5,
                        "itl_ms_p50": 2.0,
                        "itl_ms_p95": 2.5,
                    },
                    "first_forward_step_token_ids_jit_ms": 3.0,
                    "correctness": {
                        "checked": True,
                        "ok": True,
                        "full_length_ok": False,
                        "exact_generated_token_match": False,
                    },
                }
            }
        ]
    )

    assert aggregate["all_correct"]
    assert not aggregate["all_exact_generated_token_match"]
    assert aggregate["all_correctness_checked"]
    assert aggregate["ttft_ms_p95_median"] == 1.5
    assert aggregate["itl_ms_p95_median"] == 2.5
    assert aggregate["first_forward_step_token_ids_jit_ms_median"] == 3.0


def test_cuda_device_preflight_accepts_visible_device_nodes(tmp_path):
    (tmp_path / "nvidiactl").touch()
    (tmp_path / "nvidia0").touch()

    ok, detail = _cuda_device_preflight(
        dev_dir=tmp_path,
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert ok
    assert "nvidia0" in detail


def test_cuda_device_preflight_accepts_nvidia_smi_fallback(tmp_path):
    def fake_runner(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="GPU 0: test gpu\n")

    ok, detail = _cuda_device_preflight(dev_dir=tmp_path, runner=fake_runner)

    assert ok
    assert "GPU 0" in detail


def test_cuda_device_preflight_rejects_missing_gpu(tmp_path):
    def fake_runner(*args, **kwargs):
        return SimpleNamespace(returncode=9, stdout="NVIDIA-SMI has failed\n")

    ok, detail = _cuda_device_preflight(dev_dir=tmp_path, runner=fake_runner)

    assert not ok
    assert "NVIDIA-SMI has failed" in detail


def test_configured_workload_reference_uses_workload_mapping(tmp_path):
    workload = WORKLOADS["long_prefill_512_2048"]
    reference = tmp_path / "long.json"
    legacy = tmp_path / "legacy.json"
    _write_workload_artifact(reference, workload)
    _write_workload_artifact(legacy, WORKLOADS["hetero8"])

    selected = _configured_workload_reference(
        {
            "reference_json": str(legacy),
            "workload_reference_jsons": {"long_prefill_512_2048": str(reference)},
        },
        workload,
        mapping_key="workload_reference_jsons",
        legacy_key="reference_json",
    )

    assert selected == reference


def test_gpu_matrix_configs_have_valid_stored_references():
    required_workloads = ("hetero8", "long_prefill_512_2048")
    config_names = ("gpu_paged_default", "gpu_paged_fast_optin", "gpu_mtp_diagnostics")

    for config_name in config_names:
        config = json.loads((CONFIG_DIR / f"{config_name}.json").read_text(encoding="utf-8"))
        for workload_name in required_workloads:
            workload = WORKLOADS[workload_name]
            jax_reference = _configured_workload_reference(
                config,
                workload,
                mapping_key="workload_reference_jsons",
                legacy_key="reference_json",
            )
            vllm_reference = _find_local_vllm_reference(
                config,
                workload,
                reference_dir=Path("/tmp/nonexistent-gpu-matrix-reference-dir"),
            )

            assert jax_reference is not None, f"{config_name} missing JAX reference for {workload_name}"
            assert vllm_reference is not None, f"{config_name} missing vLLM reference for {workload_name}"
            assert (REPO_ROOT / jax_reference).exists()
            assert (REPO_ROOT / vllm_reference).exists()


def test_reference_for_prefers_stored_long_reference_for_default_repeats(tmp_path):
    workload = WORKLOADS["long_prefill_512_2048"]
    reference = tmp_path / "long.json"
    generated = tmp_path / "generated.json"
    _write_workload_artifact(reference, workload)
    _write_workload_artifact(generated, workload)

    selected, source = _reference_for(
        "gpu_paged_default",
        workload,
        1,
        reference,
        generated_default_reference=generated,
    )

    assert selected == reference
    assert source == "stored_jax_default"


def test_reference_for_uses_live_default_when_no_stored_reference(tmp_path):
    workload = WORKLOADS["long_prefill_512_2048"]
    generated = tmp_path / "generated.json"
    _write_workload_artifact(generated, workload)

    selected, source = _reference_for(
        "gpu_paged_fast_optin",
        workload,
        0,
        stored_workload_reference=None,
        generated_default_reference=generated,
    )

    assert selected == generated
    assert source == "live_jax_default"


def _flag_value(command, flag):
    return command[command.index(flag) + 1]


def test_jax_command_applies_workload_overrides_and_reference(tmp_path):
    config = json.loads((CONFIG_DIR / "gpu_paged_default.json").read_text(encoding="utf-8"))
    workload = WORKLOADS["long_prefill_512_2048"]
    output_json = tmp_path / "out.json"
    reference_json = tmp_path / "ref.json"

    command = _jax_command(
        config,
        workload,
        output_json,
        reference_json,
        "matrix_label",
    )

    assert command[0] == sys.executable
    assert command[1].endswith("benchmarks/benchmark_jax_server_trace.py")
    assert _flag_value(command, "--input-lens") == "512,1024,1536,2048"
    assert _flag_value(command, "--output-len") == "16"
    assert _flag_value(command, "--max-num-seqs") == "4"
    assert _flag_value(command, "--max-num-batched-tokens") == "8192"
    assert _flag_value(command, "--prefill-buckets") == "512,1024,2048"
    assert _flag_value(command, "--reference-json") == str(reference_json)
    assert _flag_value(command, "--output-json") == str(output_json)
    assert _flag_value(command, "--run-label") == "matrix_label"
    assert "--warmup" in command
    assert "--profile" in command


def test_runtime_env_roots_cache_and_temp_under_mountpoint(tmp_path, monkeypatch):
    cache_root = tmp_path / "runtime-root"
    monkeypatch.setenv("NANO_VLLM_JAX_CACHE_ROOT", str(cache_root))

    env = _runtime_env({"JAX_PLATFORMS": "cuda"})

    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["TMPDIR"] == str(cache_root / "tmp")
    assert env["HF_HOME"] == str(cache_root / ".cache" / "huggingface")
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(cache_root / ".cache" / "jax")
    assert env["NANO_VLLM_JAX_COMPILE_CACHE_DIR"] == str(cache_root / ".cache" / "jax")
    assert env["FLASHINFER_CUBIN_DIR"] == str(cache_root / ".cache" / "flashinfer" / "cubins")
    assert env["XLA_FLAGS"] == "--xla_gpu_autotune_level=4"
    assert Path(env["TMPDIR"]).exists()
    assert Path(env["JAX_COMPILATION_CACHE_DIR"]).exists()


def test_benchmark_acceptance_summary_requires_plan_evidence():
    complete_profile = {
        needle: {
            "total_ms": float(index + 1),
            "count": index + 1,
        }
        for index, needle in enumerate(PROFILE_NEEDLES)
    }
    repeats = [
        {
            "metrics": {
                "profile": complete_profile,
                "first_forward_step_token_ids_jit_ms": 9.0,
            }
        },
        {
            "metrics": {
                "profile": complete_profile,
                "first_forward_step_token_ids_jit_ms": 10.0,
            }
        },
    ]
    aggregate = {
        "repeat_count": 2,
        "tokens_per_second_median": 90.0,
        "ttft_ms_p50_median": 1.0,
        "ttft_ms_p95_median": 1.5,
        "itl_ms_p50_median": 2.0,
        "itl_ms_p95_median": 2.5,
        "first_forward_step_token_ids_jit_ms_median": 9.5,
        "all_correctness_checked": True,
        "all_exact_generated_token_match": True,
    }
    comparison = {"jax_over_vllm_throughput": 0.9}
    vllm_metrics = {"performance": {"tokens_per_second": 100.0}}

    acceptance = _benchmark_acceptance_summary(
        repeats,
        aggregate,
        comparison,
        vllm_metrics,
    )

    assert all(acceptance["checks"].values())
    assert acceptance["speed_claim_ready"]
    assert acceptance["target_vllm_ratio_met"]
    assert acceptance["missing_profile_counters"] == []


def test_benchmark_acceptance_summary_rejects_incomplete_evidence():
    acceptance = _benchmark_acceptance_summary(
        [{"metrics": None}],
        {
            "repeat_count": 1,
            "tokens_per_second_median": None,
            "ttft_ms_p50_median": None,
            "ttft_ms_p95_median": None,
            "itl_ms_p50_median": None,
            "itl_ms_p95_median": None,
            "first_forward_step_token_ids_jit_ms_median": None,
            "all_correctness_checked": False,
            "all_exact_generated_token_match": False,
        },
        {"jax_over_vllm_throughput": None},
        None,
    )

    assert not acceptance["speed_claim_ready"]
    assert not acceptance["target_vllm_ratio_met"]
    assert not acceptance["checks"]["minimum_repeats"]
    assert not acceptance["checks"]["profile_counters_present"]
    assert not acceptance["checks"]["jax_latency_present"]
    assert not acceptance["checks"]["first_forward_step_present"]
    assert acceptance["missing_profile_counters"]


def test_benchmark_acceptance_summary_requires_all_profile_counters():
    incomplete_profile = {
        needle: {
            "total_ms": 1.0,
            "count": 1,
        }
        for needle in PROFILE_NEEDLES[:-1]
    }
    acceptance = _benchmark_acceptance_summary(
        [
            {"metrics": {"profile": incomplete_profile}},
            {"metrics": {"profile": incomplete_profile}},
        ],
        {
            "repeat_count": 2,
            "tokens_per_second_median": 90.0,
            "ttft_ms_p50_median": 1.0,
            "ttft_ms_p95_median": 1.5,
            "itl_ms_p50_median": 2.0,
            "itl_ms_p95_median": 2.5,
            "first_forward_step_token_ids_jit_ms_median": 9.5,
            "all_correctness_checked": True,
            "all_exact_generated_token_match": True,
        },
        {"jax_over_vllm_throughput": 0.9},
        {"performance": {"tokens_per_second": 100.0}},
    )

    assert not acceptance["speed_claim_ready"]
    assert not acceptance["checks"]["profile_counters_present"]
    assert acceptance["missing_profile_counters"] == [
        f"repeat1:{PROFILE_NEEDLES[-1]}",
        f"repeat2:{PROFILE_NEEDLES[-1]}",
    ]
