"""Tests for GPU matrix summary helpers."""

import os
import sys
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.run_gpu_matrix import (
    CONFIG_DIR,
    DEFAULT_WORKLOADS,
    PROFILE_NEEDLES,
    REPO_ROOT,
    WORKLOADS,
    _acceptance_failures,
    _aggregate_repeats,
    _artifact_matches_workload,
    _benchmark_acceptance_summary,
    _comparison_summary,
    _configured_workload_reference,
    _cuda_device_preflight,
    _goal_target_failure,
    _should_capture_live_jax_default_reference,
    _reference_for,
    _reference_metrics_for_comparison,
    _find_local_vllm_reference,
    _jax_available,
    _jax_command,
    _runtime_env,
    _selected_matrix_names,
    _stored_reference_gaps,
    _validate_summary_shape,
)


class _FakeTokenizer:
    vocab_size = 128
    eos_token_id = 2

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [min(127, max(1, len(part))) for part in text.split()]}


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
            "goal_target",
            "jax_python",
            "jax_default_references",
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
        "goal_target": {
            "workload": "long_prefill_512_2048",
            "config": "gpu_paged_default",
            "target_vllm_ratio": 0.75,
            "description": "test target",
        },
        "jax_python": {
            "path": sys.executable,
            "available": False,
        },
        "jax_default_references": {"hetero8": {"source": "stored"}},
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
                        "profile_medians": {},
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
                        "runs_succeeded": False,
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


def test_validate_summary_shape_rejects_incomplete_goal_target():
    summary = _minimal_summary()
    del summary["goal_target"]["description"]

    with pytest.raises(ValueError, match="summary.goal_target"):
        _validate_summary_shape(summary, _minimal_schema())


def test_validate_summary_shape_rejects_missing_jax_python():
    summary = _minimal_summary()
    del summary["jax_python"]

    with pytest.raises(ValueError, match="jax_python"):
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
                    "profile": {
                        PROFILE_NEEDLES[0]: {
                            "total_ms": 5.0,
                            "count": 2,
                        }
                    },
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
    assert aggregate["profile_medians"][PROFILE_NEEDLES[0]] == {
        "total_ms_median": 5.0,
        "count_median": 2.0,
    }


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


def test_jax_available_uses_selected_python_without_importing_jax(tmp_path):
    jax_python = tmp_path / "python"
    jax_python.touch()

    def fake_runner(command, **kwargs):
        assert command[0] == str(jax_python)
        assert "find_spec('jax')" in command[2]
        return SimpleNamespace(returncode=0, stdout="")

    assert _jax_available(jax_python, runner=fake_runner)


def test_jax_available_rejects_missing_selected_python(tmp_path):
    assert not _jax_available(tmp_path / "missing-python")


def test_selected_matrix_names_goal_target_only_overrides_matrix_selection():
    args = SimpleNamespace(
        goal_target_only=True,
        configs="gpu_paged_default,gpu_mtp_diagnostics",
        workloads="hetero8,decode_heavy_128x128",
    )

    configs, workloads = _selected_matrix_names(args)

    assert configs == ["gpu_paged_default"]
    assert workloads == ["long_prefill_512_2048"]


def test_selected_matrix_names_uses_requested_matrix_without_goal_target_only():
    args = SimpleNamespace(
        goal_target_only=False,
        configs="gpu_paged_fast_optin,gpu_mtp_diagnostics",
        workloads="hetero8,long_prefill_512_2048",
    )

    configs, workloads = _selected_matrix_names(args)

    assert configs == ["gpu_paged_fast_optin", "gpu_mtp_diagnostics"]
    assert workloads == ["hetero8", "long_prefill_512_2048"]


def test_default_workloads_keep_vllm_random_sidecar_opt_in():
    assert "vllm_random_longprefill" in WORKLOADS
    assert "vllm_random_longprefill" not in DEFAULT_WORKLOADS


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


def test_stored_reference_gaps_reports_uncovered_workloads(tmp_path):
    config = {
        "workload_reference_jsons": {},
        "workload_vllm_reference_jsons": {},
    }

    gaps = _stored_reference_gaps(
        {"gpu_paged_default": config},
        {"short_32_128": WORKLOADS["short_32_128"]},
        reference_dir=tmp_path,
    )

    assert gaps == [
        "short_32_128: missing stored vLLM reference",
        "short_32_128/gpu_paged_default: missing stored JAX reference",
    ]


def test_live_jax_default_capture_needed_when_selected_config_lacks_reference():
    should_capture, missing = _should_capture_live_jax_default_reference(
        selected_configs=["gpu_paged_fast_optin"],
        configs={"gpu_paged_fast_optin": {"workload_reference_jsons": {}}},
        workload=WORKLOADS["short_32_128"],
        default_config={"allow_live_jax_default_if_reference_missing": True},
    )

    assert should_capture
    assert missing == ["gpu_paged_fast_optin"]


def test_live_jax_default_capture_skipped_when_default_runs_first():
    should_capture, missing = _should_capture_live_jax_default_reference(
        selected_configs=["gpu_paged_default", "gpu_paged_fast_optin"],
        configs={
            "gpu_paged_default": {"workload_reference_jsons": {}},
            "gpu_paged_fast_optin": {"workload_reference_jsons": {}},
        },
        workload=WORKLOADS["short_32_128"],
        default_config={"allow_live_jax_default_if_reference_missing": True},
    )

    assert not should_capture
    assert missing == ["gpu_paged_default", "gpu_paged_fast_optin"]


def test_live_jax_default_capture_used_for_sidecar_even_when_default_runs_first():
    should_capture, missing = _should_capture_live_jax_default_reference(
        selected_configs=["gpu_paged_default"],
        configs={"gpu_paged_default": {"workload_reference_jsons": {}}},
        workload=WORKLOADS["vllm_random_longprefill"],
        default_config={"allow_live_jax_default_if_reference_missing": True},
    )

    assert should_capture
    assert missing == ["gpu_paged_default"]


def test_live_jax_default_capture_disabled_by_config():
    should_capture, missing = _should_capture_live_jax_default_reference(
        selected_configs=["gpu_paged_fast_optin"],
        configs={"gpu_paged_fast_optin": {"workload_reference_jsons": {}}},
        workload=WORKLOADS["short_32_128"],
        default_config={"allow_live_jax_default_if_reference_missing": False},
    )

    assert not should_capture
    assert missing == ["gpu_paged_fast_optin"]


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


def test_reference_for_can_plan_unwritten_live_default_in_dry_run(tmp_path):
    workload = WORKLOADS["short_32_128"]
    generated = tmp_path / "planned.json"

    selected, source = _reference_for(
        "gpu_paged_fast_optin",
        workload,
        0,
        stored_workload_reference=None,
        generated_default_reference=generated,
        allow_unverified_generated_default=True,
    )

    assert selected == generated
    assert source == "live_jax_default"


def test_reference_metrics_for_comparison_uses_first_existing_reference(tmp_path):
    workload = WORKLOADS["hetero8"]
    missing = tmp_path / "missing.json"
    reference = tmp_path / "reference.json"
    _write_workload_artifact(reference, workload)
    data = json.loads(reference.read_text(encoding="utf-8"))
    data["performance"] = {"tokens_per_second": 10.0}
    data["profile_counters"] = {
        "ranges": {
            PROFILE_NEEDLES[0]: {
                "total_ms": 5.0,
                "count": 1,
            }
        }
    }
    reference.write_text(json.dumps(data), encoding="utf-8")

    metrics, source, artifact = _reference_metrics_for_comparison(
        [
            {"reference_json": str(missing), "reference_source": "missing"},
            {"reference_json": str(reference), "reference_source": "stored"},
        ]
    )

    assert source == "stored"
    assert artifact == str(reference)
    assert metrics["performance"]["tokens_per_second"] == 10.0
    assert metrics["profile"][PROFILE_NEEDLES[0]] == {
        "total_ms": 5.0,
        "count": 1,
    }


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
        Path("/tmp/jax-python"),
    )

    assert command[0] == "/tmp/jax-python"
    assert command[1].endswith("benchmarks/benchmark_jax_server_trace.py")
    assert _flag_value(command, "--input-lens") == "512,1024,1536,2048"
    assert _flag_value(command, "--output-len") == "16"
    assert _flag_value(command, "--prompt-source") == "tokenized_seed_repeat"
    assert _flag_value(command, "--max-num-seqs") == "4"
    assert _flag_value(command, "--max-num-batched-tokens") == "8192"
    assert _flag_value(command, "--prefill-buckets") == "512,1024,2048"
    assert _flag_value(command, "--reference-json") == str(reference_json)
    assert _flag_value(command, "--output-json") == str(output_json)
    assert _flag_value(command, "--run-label") == "matrix_label"
    assert "--warmup" in command
    assert "--profile" in command


def test_jax_command_wires_vllm_random_sidecar_args(tmp_path):
    config = json.loads((CONFIG_DIR / "gpu_paged_default.json").read_text(encoding="utf-8"))
    workload = WORKLOADS["vllm_random_longprefill"]
    command = _jax_command(
        config,
        workload,
        tmp_path / "out.json",
        None,
        "sidecar",
        Path("/tmp/jax-python"),
    )

    assert _flag_value(command, "--prompt-source") == "vllm_random"
    assert _flag_value(command, "--dataset-name") == "random"
    assert _flag_value(command, "--num-prompts") == "128"
    assert _flag_value(command, "--random-input-len") == "1280"
    assert _flag_value(command, "--random-output-len") == "16"
    assert _flag_value(command, "--random-range-ratio") == '{"input":0.6,"output":0.0}'


def test_artifact_matches_vllm_random_sidecar_metadata(tmp_path):
    workload = WORKLOADS["vllm_random_longprefill"]
    artifact = tmp_path / "random.json"
    artifact.write_text(
        json.dumps(
            {
                "run_config": {
                    "prompt_source": "vllm_random",
                    "dataset_name": "random",
                    "num_prompts": 128,
                    "seed": 0,
                    "random_input_len": 1280,
                    "random_output_len": 16,
                    "random_range_ratio": {"input": 0.6, "output": 0.0},
                }
            }
        ),
        encoding="utf-8",
    )

    assert _artifact_matches_workload(artifact, workload)


def test_prepare_vllm_random_prompts_writes_hashed_manifest(tmp_path):
    from benchmarks.benchmark_vllm_qwen35 import prepare_prompt_rows

    args = SimpleNamespace(
        input_lens="1280",
        output_len=16,
        output_lengths="",
        prompt_suite="mixed",
        prompt_source="vllm_random",
        prompt_manifest_jsonl="",
        prompt_manifest_output_jsonl=str(tmp_path / "prompts.jsonl"),
        dataset_name="random",
        num_prompts=3,
        seed=7,
        random_input_len=8,
        random_output_len=2,
        random_range_ratio='{"input":0.5,"output":0.0}',
        output_json=str(tmp_path / "artifact.json"),
    )

    rows, info = prepare_prompt_rows(_FakeTokenizer(), args)

    assert len(rows) == 3
    assert info["prompt_source"] == "vllm_random"
    assert info["dataset_name"] == "random"
    assert info["num_prompts"] == 3
    assert info["prompt_manifest_sha256"]
    assert Path(info["prompt_manifest_jsonl"]).exists()
    assert all(row["output_len"] == 2 for row in rows)


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
            "run": {"status": "ok"},
            "metrics": {
                "profile": complete_profile,
                "first_forward_step_token_ids_jit_ms": 9.0,
            }
        },
        {
            "run": {"status": "ok"},
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


def test_comparison_summary_reports_gap_to_target():
    comparison = _comparison_summary(
        {
            "tokens_per_second_median": 78.0,
            "ttft_ms_p50_median": 580.0,
            "itl_ms_p50_median": 15.0,
            "profile_medians": {
                PROFILE_NEEDLES[0]: {
                    "total_ms_median": 12.0,
                    "count_median": 3.0,
                }
            },
        },
        {
            "performance": {
                "tokens_per_second": 116.0,
                "ttft_ms_p50": 440.0,
                "itl_ms_p50": 5.0,
            }
        },
        "stored",
        {
            "performance": {
                "tokens_per_second": 70.0,
                "ttft_ms_p50": 600.0,
                "itl_ms_p50": 20.0,
            },
            "profile": {
                PROFILE_NEEDLES[0]: {
                    "total_ms": 10.0,
                    "count": 2,
                }
            },
        },
        "stored_jax_default",
        "reference.json",
    )

    assert comparison["jax_over_vllm_throughput"] == pytest.approx(78.0 / 116.0)
    assert comparison["target_tokens_per_second"] == pytest.approx(87.0)
    assert comparison["tokens_per_second_gap_to_target"] == pytest.approx(9.0)
    assert comparison["required_jax_speedup_to_target"] == pytest.approx(87.0 / 78.0)
    assert comparison["ttft_ms_p50_delta_vs_vllm"] == 140.0
    assert comparison["itl_ms_p50_delta_vs_vllm"] == 10.0
    assert comparison["jax_reference_source"] == "stored_jax_default"
    assert comparison["jax_reference_artifact"] == "reference.json"
    assert comparison["jax_reference_tokens_per_second"] == 70.0
    assert comparison["jax_over_jax_reference_throughput"] == pytest.approx(78.0 / 70.0)
    assert comparison["tokens_per_second_delta_vs_jax_reference"] == 8.0
    assert comparison["ttft_ms_p50_delta_vs_jax_reference"] == -20.0
    assert comparison["itl_ms_p50_delta_vs_jax_reference"] == -5.0
    profile_delta = comparison["profile_delta_vs_jax_reference"][PROFILE_NEEDLES[0]]
    assert profile_delta["total_ms_delta"] == 2.0
    assert profile_delta["total_ms_ratio"] == pytest.approx(1.2)
    assert profile_delta["count_delta"] == 1.0


def test_comparison_summary_clamps_negative_gap_after_target_met():
    comparison = _comparison_summary(
        {"tokens_per_second_median": 100.0},
        {"performance": {"tokens_per_second": 116.0}},
        "stored",
    )

    assert comparison["target_tokens_per_second"] == pytest.approx(87.0)
    assert comparison["tokens_per_second_gap_to_target"] == 0.0
    assert comparison["required_jax_speedup_to_target"] == 1.0


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
    assert not acceptance["checks"]["runs_succeeded"]
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
            {"run": {"status": "ok"}, "metrics": {"profile": incomplete_profile}},
            {"run": {"status": "ok"}, "metrics": {"profile": incomplete_profile}},
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


def test_acceptance_failures_reports_missing_evidence_and_target():
    summary = {
        "workloads": ["long_prefill_512_2048"],
        "configs": ["gpu_paged_fast_optin"],
        "acceptance": {
            "long_prefill_512_2048": {
                "gpu_paged_fast_optin": {
                    "checks": {
                        "minimum_repeats": True,
                        "runs_succeeded": True,
                        "correctness_checked": False,
                        "exact_generated_token_match": False,
                    },
                    "speed_claim_ready": False,
                    "target_vllm_ratio": 0.75,
                    "target_vllm_ratio_met": False,
                    "missing_profile_counters": ["repeat1:gather"],
                }
            }
        },
    }

    failures = _acceptance_failures(summary)

    assert failures == [
        "long_prefill_512_2048/gpu_paged_fast_optin: "
        "failed checks: correctness_checked,exact_generated_token_match; "
        "speed_claim_ready=false; target_vllm_ratio_met=false target=0.75; "
        "missing_profile_counters=1"
    ]


def test_acceptance_failures_empty_when_ready_and_target_met():
    summary = {
        "workloads": ["long_prefill_512_2048"],
        "configs": ["gpu_paged_fast_optin"],
        "acceptance": {
            "long_prefill_512_2048": {
                "gpu_paged_fast_optin": {
                    "checks": {
                        "minimum_repeats": True,
                        "runs_succeeded": True,
                        "correctness_checked": True,
                        "exact_generated_token_match": True,
                    },
                    "speed_claim_ready": True,
                    "target_vllm_ratio": 0.75,
                    "target_vllm_ratio_met": True,
                    "missing_profile_counters": [],
                }
            }
        },
    }

    assert _acceptance_failures(summary) == []


def test_goal_target_failure_reports_missing_target_workload():
    summary = {
        "workloads": ["hetero8"],
        "configs": ["gpu_paged_default"],
        "goal_target": {
            "workload": "long_prefill_512_2048",
            "config": "gpu_paged_default",
        },
        "acceptance": {},
    }

    failure = _goal_target_failure(summary)

    assert "final goal target workload is not in this matrix summary" in failure


def test_goal_target_failure_reports_failed_checks():
    summary = {
        "workloads": ["long_prefill_512_2048"],
        "configs": ["gpu_paged_default"],
        "goal_target": {
            "workload": "long_prefill_512_2048",
            "config": "gpu_paged_default",
        },
        "acceptance": {
            "long_prefill_512_2048": {
                "gpu_paged_default": {
                    "checks": {
                        "minimum_repeats": True,
                        "runs_succeeded": False,
                    },
                    "speed_claim_ready": False,
                    "target_vllm_ratio": 0.75,
                    "target_vllm_ratio_met": False,
                    "missing_profile_counters": ["repeat1:gather"],
                }
            }
        },
    }

    failure = _goal_target_failure(summary)

    assert failure == (
        "long_prefill_512_2048/gpu_paged_default: "
        "failed checks: runs_succeeded; speed_claim_ready=false; "
        "target_vllm_ratio_met=false target=0.75; missing_profile_counters=1"
    )


def test_goal_target_failure_empty_when_target_ready():
    summary = {
        "workloads": ["long_prefill_512_2048"],
        "configs": ["gpu_paged_default"],
        "goal_target": {
            "workload": "long_prefill_512_2048",
            "config": "gpu_paged_default",
        },
        "acceptance": {
            "long_prefill_512_2048": {
                "gpu_paged_default": {
                    "checks": {
                        "minimum_repeats": True,
                        "runs_succeeded": True,
                    },
                    "speed_claim_ready": True,
                    "target_vllm_ratio": 0.75,
                    "target_vllm_ratio_met": True,
                    "missing_profile_counters": [],
                }
            }
        },
    }

    assert _goal_target_failure(summary) is None


def test_run_gpu_matrix_dry_run_writes_markdown_report(tmp_path):
    output_json = tmp_path / "matrix.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_gpu_matrix.py",
            "--goal-target-only",
            "--dry-run",
            "--no-live-vllm",
            "--require-stored-references",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    report_md = output_json.with_suffix(".md")
    assert output_json.exists()
    assert report_md.exists()
    report = report_md.read_text(encoding="utf-8")
    assert "# GPU Matrix Report" in report
    assert "long_prefill_512_2048/gpu_paged_default" in report
