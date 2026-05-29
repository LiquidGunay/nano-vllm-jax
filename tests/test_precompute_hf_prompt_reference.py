"""CPU-only tests for the HF prompt reference helper."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import precompute_hf_prompt_reference as helper


def test_parse_args_defaults_cover_long_prefill_reference():
    args = helper.parse_args([])

    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.dtype == "float32"
    assert args.weight_dtype == "bfloat16"
    assert args.input_lens == "512,1024,1536,2048"
    assert args.output_len == 16
    assert args.prompt_source == "tokenized_seed_repeat"
    assert args.top_k == 5
    assert args.output_json == helper.DEFAULT_OUTPUT_JSON


@pytest.mark.parametrize("flag", ["--output-len", "--top-k", "--random-input-len", "--random-output-len"])
def test_parse_args_rejects_non_positive_counts(flag):
    with pytest.raises(SystemExit):
        helper.parse_args([flag, "0"])


def test_json_safe_converts_paths_numpy_and_nested_values():
    safe = helper._json_safe(
        {
            "path": Path("results/example.json"),
            "scalar": np.int32(7),
            "array": np.array([1, 2], dtype=np.int64),
            "tuple": (np.float32(1.5), Path("x")),
        }
    )

    assert safe == {
        "path": "results/example.json",
        "scalar": 7,
        "array": [1, 2],
        "tuple": [1.5, "x"],
    }


def test_write_reference_emits_sorted_json(tmp_path):
    output_path = helper.write_reference({"z": np.int64(1), "a": "first"}, str(tmp_path / "ref.json"))

    assert output_path == tmp_path / "ref.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"a": "first", "z": 1}
    assert output_path.read_text(encoding="utf-8").splitlines()[1].startswith('  "a"')
