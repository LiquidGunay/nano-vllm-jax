"""Policy tests for the segmented GDN prefill correctness gate."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_gdn_prefill_kernel import _segmented_reference_policy


def test_segmented_reference_policy_blocks_when_gate_not_checked():
    policy = _segmented_reference_policy({"enabled": False})

    assert policy["status"] == "not_checked"
    assert not policy["cuda_math_allowed"]
    assert not policy["serving_routing_allowed"]


def test_segmented_reference_policy_allows_benchmark_cuda_after_strict_pass():
    policy = _segmented_reference_policy(
        {
            "enabled": True,
            "comparison": {
                "output_max_abs": 1e-6,
                "valid_output_max_abs": 2e-6,
                "state_max_abs": 3e-6,
            },
        }
    )

    assert policy["status"] == "eligible_for_segmented_cuda_math"
    assert policy["cuda_math_allowed"]
    assert not policy["serving_routing_allowed"]


def test_segmented_reference_policy_blocks_rowwise_drift():
    policy = _segmented_reference_policy(
        {
            "enabled": True,
            "comparison": {
                "output_max_abs": 1.431e-5,
                "valid_output_max_abs": 1.431e-5,
                "state_max_abs": 1.678e-4,
            },
            "row_padded_to_seq_len": {
                "comparison": {
                    "output_max_abs": 1.240e-5,
                    "valid_output_max_abs": 1.240e-5,
                    "state_max_abs": 1.831e-4,
                },
            },
        }
    )

    assert policy["status"] == "blocked_on_correctness_policy"
    assert policy["requires_design_decision"]
    assert not policy["cuda_math_allowed"]
    assert not policy["serving_routing_allowed"]
    assert "row-wise decomposition" in policy["reason"]
    assert "full-model real-weight token/logit parity" in policy["design_change_option"]
    assert policy["design_change_required_gate"]["name"] == "real_weight_full_model_token_logit_parity"
    required = policy["design_change_required_gate"]["required_checks"]
    assert required["top1_exact_matches"] == 500
    assert required["ordered_top5_exact_matches"] == 500
    assert required["top5_set_exact_matches"] == 500
    assert required["max_hf_topk_id_logit_diff_lte"] == 2e-5


def test_segmented_reference_policy_reports_gate_error():
    policy = _segmented_reference_policy(
        {
            "enabled": True,
            "error": "RuntimeError: CUDA backend unavailable",
        }
    )

    assert policy["status"] == "gate_error"
    assert not policy["cuda_math_allowed"]
    assert "CUDA backend unavailable" in policy["reason"]
