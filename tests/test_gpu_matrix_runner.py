"""Tests for GPU matrix summary helpers."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.run_gpu_matrix import (
    WORKLOADS,
    _aggregate_repeats,
    _configured_workload_reference,
    _cuda_device_preflight,
    _reference_for,
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


def test_aggregate_repeats_does_not_treat_missing_correctness_as_correct():
    aggregate = _aggregate_repeats([{"metrics": None}])

    assert not aggregate["all_correct"]
    assert not aggregate["all_exact_generated_token_match"]
    assert not aggregate["all_correctness_checked"]


def test_aggregate_repeats_requires_exact_full_length_match():
    aggregate = _aggregate_repeats(
        [
            {
                "metrics": {
                    "performance": {"tokens_per_second": 10.0},
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


def test_reference_for_checks_default_first_repeat_against_stored_long_reference(tmp_path):
    workload = WORKLOADS["long_prefill_512_2048"]
    reference = tmp_path / "long.json"
    _write_workload_artifact(reference, workload)

    selected, source = _reference_for(
        "gpu_paged_default",
        workload,
        0,
        reference,
        generated_default_reference=None,
    )

    assert selected == reference
    assert source == "stored_jax_default"
