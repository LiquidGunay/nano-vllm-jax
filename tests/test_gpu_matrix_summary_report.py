"""Tests for GPU matrix Markdown report rendering."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.summarize_gpu_matrix import acceptance_failures, render_markdown


def _summary():
    return {
        "created_at_utc": "20260526_000000",
        "dry_run": False,
        "repeats": 2,
        "run_dir": "results/gpu_matrix_runs/test",
        "output_json": "results/gpu_matrix_test.json",
        "configs": ["gpu_paged_default"],
        "workloads": ["long_prefill_512_2048"],
        "goal_target": {
            "workload": "long_prefill_512_2048",
            "config": "gpu_paged_default",
            "target_vllm_ratio": 0.75,
        },
        "jax_python": {
            "path": "/mountpoint/.exp/.venv/bin/python",
            "available": True,
        },
        "comparisons": {
            "long_prefill_512_2048": {
                "gpu_paged_default": {
                    "jax_tokens_per_second_median": 90.0,
                    "vllm_tokens_per_second": 116.0,
                    "jax_over_vllm_throughput": 90.0 / 116.0,
                    "target_tokens_per_second": 87.0,
                    "tokens_per_second_gap_to_target": 0.0,
                    "jax_reference_tokens_per_second": 78.0,
                    "jax_over_jax_reference_throughput": 90.0 / 78.0,
                    "profile_delta_vs_jax_reference": {
                        "small_bucket": {
                            "current_total_ms_median": 12.0,
                            "reference_total_ms": 10.0,
                            "total_ms_delta": 2.0,
                            "total_ms_ratio": 1.2,
                            "current_count_median": 3.0,
                            "reference_count": 2,
                            "count_delta": 1.0,
                        },
                        "large_bucket": {
                            "current_total_ms_median": 5.0,
                            "reference_total_ms": 20.0,
                            "total_ms_delta": -15.0,
                            "total_ms_ratio": 0.25,
                            "current_count_median": 1.0,
                            "reference_count": 4,
                            "count_delta": -3.0,
                        },
                    },
                }
            }
        },
        "acceptance": {
            "long_prefill_512_2048": {
                "gpu_paged_default": {
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


def test_render_markdown_includes_goal_matrix_and_sorted_profile_deltas():
    report = render_markdown(_summary(), top_profile_deltas=2)

    assert "# GPU Matrix Report" in report
    assert "- target: `long_prefill_512_2048/gpu_paged_default`" in report
    assert "| workload | config | ready | target met | JAX tok/s |" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | yes | yes | 90.00 |" in report
    assert "## Acceptance Failures\n\nNone." in report
    assert report.index("large_bucket") < report.index("small_bucket")


def test_acceptance_failures_reports_failed_checks_and_missing_profiles():
    summary = _summary()
    acceptance = summary["acceptance"]["long_prefill_512_2048"]["gpu_paged_default"]
    acceptance["checks"]["exact_generated_token_match"] = False
    acceptance["speed_claim_ready"] = False
    acceptance["target_vllm_ratio_met"] = False
    acceptance["missing_profile_counters"] = ["repeat1:gather"]

    assert acceptance_failures(summary) == [
        "long_prefill_512_2048/gpu_paged_default: "
        "failed checks: exact_generated_token_match; "
        "speed_claim_ready=false; target_vllm_ratio_met=false target=0.75; "
        "missing_profile_counters=1"
    ]
