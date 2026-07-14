import json
from pathlib import Path

from benchmarks.run_claim import load_manifest, prompt_rows


ROOT = Path(__file__).resolve().parents[1]


def test_claim_is_one_fixed_greedy_b1_workload():
    manifest = load_manifest(ROOT / "benchmarks/claim.json")
    workload = manifest["workload"]
    prompts = prompt_rows(manifest)

    assert workload["batch_size"] == 1
    assert workload["temperature"] == 0
    assert workload["ignore_eos"]
    assert not workload["prefix_cache"]
    assert len(prompts) == 1
    assert len(prompts[0]) == workload["prompt_tokens"]
    assert prompts[0][:4] == [1, 2, 3, 4]


def test_environment_metadata_matches_the_manifest():
    manifest = load_manifest(ROOT / "benchmarks/claim.json")
    environments = json.loads((ROOT / "benchmarks/environments.json").read_text())
    requirement = (ROOT / "benchmarks/vllm-requirements.txt").read_text().strip()

    assert environments["vllm"]["requirement"] == requirement
    assert requirement == f"vllm=={manifest['frameworks']['vllm']}"
    assert environments["jax"]["lock"] == "uv.lock"
    assert environments["vllm"]["remove_for_text_only"] == ["torchcodec"]


def test_result_schema_covers_validity_evidence():
    schema = json.loads((ROOT / "benchmarks/result.schema.json").read_text())
    required = set(schema["required"])

    assert {"capacity", "timing", "correctness", "memory", "valid", "invalid_reasons"} <= required
