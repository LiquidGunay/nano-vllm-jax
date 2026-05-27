"""Tests for GPU matrix Markdown report rendering."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.summarize_gpu_matrix import acceptance_failures, render_markdown

TARGET_VLLM_RATIO = 0.9


def _summary():
    jax_tokens_per_second = 105.0
    vllm_tokens_per_second = 116.0
    target_tokens_per_second = vllm_tokens_per_second * TARGET_VLLM_RATIO
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
            "target_vllm_ratio": TARGET_VLLM_RATIO,
        },
        "jax_python": {
            "path": "/mountpoint/.exp/.venv/bin/python",
            "available": True,
        },
        "comparisons": {
            "long_prefill_512_2048": {
                "gpu_paged_default": {
                    "jax_tokens_per_second_median": jax_tokens_per_second,
                    "vllm_tokens_per_second": vllm_tokens_per_second,
                    "jax_over_vllm_throughput": jax_tokens_per_second / vllm_tokens_per_second,
                    "target_tokens_per_second": target_tokens_per_second,
                    "tokens_per_second_gap_to_target": 0.0,
                    "jax_reference_tokens_per_second": 78.0,
                    "jax_over_jax_reference_throughput": jax_tokens_per_second / 78.0,
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
        "matrix": {
            "long_prefill_512_2048": {
                "gpu_paged_default": {
                    "aggregate": {
                        "scheduler_diagnostics_median": {
                            "available": True,
                            "prefill_step_count": 4,
                            "decode_step_count": 60,
                            "max_prefill_step_sequences": 4,
                            "max_step_tokens": 8192,
                            "prefill_step_seconds_total": 2.4,
                            "decode_step_seconds_total": 0.8,
                        },
                        "profile_scoped_range_medians": {
                            "gpu": {
                                "gemm_fusion": {
                                    "total_ms_median": 25.0,
                                    "count_median": 3.0,
                                }
                            },
                            "cpu": {
                                "forward_step_token_ids_jit": {
                                    "total_ms_median": 30.0,
                                    "count_median": 1.0,
                                }
                            },
                        },
                    },
                    "repeats": [
                        {
                            "repeat": 1,
                            "metrics": {
                                "profile_scoped_top_events_by_total_ms": {
                                    "gpu": [
                                        {
                                            "name": "gemm_fusion_dot",
                                            "total_ms": 25.0,
                                            "count": 3,
                                        },
                                        {
                                            "name": "MemcpyD2D",
                                            "total_ms": 5.0,
                                            "count": 2,
                                        },
                                    ],
                                    "cpu": [
                                        {
                                            "name": "forward_step_token_ids_jit",
                                            "total_ms": 30.0,
                                            "count": 1,
                                        }
                                    ],
                                }
                            },
                        }
                    ],
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
                    "target_vllm_ratio": TARGET_VLLM_RATIO,
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
    assert "| long_prefill_512_2048 | gpu_paged_default | yes | yes | 105.00 |" in report
    assert "## Scheduler Diagnostics" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | 4 | 60 | 4 | 8192 | 2.40 s | 0.80 s |" in report
    assert "## Scoped Profile Range Medians" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 30.00 ms | 1.0 |" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | gpu | gemm_fusion | 25.00 ms | 3.0 |" in report
    assert "## Top Scoped Profile Events" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot | 25.00 ms | 3 |" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | forward_step_token_ids_jit | 30.00 ms | 1 |" in report
    assert "## Acceptance Failures\n\nNone." in report
    assert "## Logbook Entry Template" in report
    assert "- profile movement to explain:" in report
    assert "- interpretation: <explain whether the profile movement supports the claimed change>" in report
    assert "- decision: <keep/reject/follow up, with reason>" in report
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
        f"speed_claim_ready=false; target_vllm_ratio_met=false target={TARGET_VLLM_RATIO}; "
        "missing_profile_counters=1"
    ]


def test_render_markdown_falls_back_to_repeat_scoped_range_medians():
    summary = _summary()
    aggregate = summary["matrix"]["long_prefill_512_2048"]["gpu_paged_default"]["aggregate"]
    del aggregate["profile_scoped_range_medians"]
    repeat = summary["matrix"]["long_prefill_512_2048"]["gpu_paged_default"]["repeats"][0]
    repeat["metrics"]["profile_scoped_ranges"] = {
        "gpu": {
            "gemm_fusion": {
                "total_ms": 25.0,
                "count": 3,
            }
        }
    }

    report = render_markdown(summary)

    assert "## Scoped Profile Range Medians" in report
    assert "| long_prefill_512_2048 | gpu_paged_default | gpu | gemm_fusion | 25.00 ms | 3.0 |" in report
