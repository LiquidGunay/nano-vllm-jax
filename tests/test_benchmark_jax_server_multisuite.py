"""Tests for the shared-process JAX multisuite benchmark helper."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_jax_server_multisuite import (
    _engine_kwargs,
    _random_large_prompt_rows,
    _workload_names,
)
from benchmarks.run_gpu_matrix import CONFIG_DIR


def _args(**overrides):
    values = {
        "serving_envelope": "random_large",
        "model": "Qwen/Qwen3.5-0.8B",
        "backend": "gpu",
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "jax_execution": "jit",
        "temperature": 0.0,
        "top_p": 1.0,
        "sampling_top_k": -1,
        "top_k": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_random_large_envelope_overrides_config_shape_surface():
    config = json.loads(
        (CONFIG_DIR / "gpu_paged_gdn_fla_decode_static_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    kwargs = _engine_kwargs(_args(), config)

    assert kwargs["max_num_seqs"] == 8
    assert kwargs["max_num_resident_seqs"] == 8
    assert kwargs["max_num_batched_tokens"] == 1024
    assert kwargs["prefill_buckets"] == (512, 1024)
    assert kwargs["prefill_token_buckets"] == (512, 1024)
    assert kwargs["batch_size_buckets"] == (1, 2, 3, 4, 5, 6, 7, 8)
    assert kwargs["max_blocks_per_seq"] == 128
    assert kwargs["full_attention_decode_impl"] == "flashinfer_paged"
    assert kwargs["gdn_disable_fallbacks"] is True


def test_workload_names_accepts_pseudo_random_large_and_rejects_unknown_workload():
    assert _workload_names("random_large,hetero8,decode_heavy_128x128") == [
        "random_large",
        "hetero8",
        "decode_heavy_128x128",
    ]

    try:
        _workload_names("hetero8,missing")
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected unknown workload to fail")


def test_random_large_prompt_rows_match_target_envelope(tmp_path):
    class FakeTokenizer:
        vocab_size = 1024
        eos_token_id = 2

        def __len__(self):
            return self.vocab_size

    rows, info = _random_large_prompt_rows(FakeTokenizer(), tmp_path / "random_large.json")

    assert len(rows) == 8
    assert min(row["prompt_length"] for row in rows) >= 512
    assert max(row["prompt_length"] for row in rows) <= 1024
    assert min(row["output_len"] for row in rows) >= 128
    assert max(row["output_len"] for row in rows) <= 256
    assert info["prompt_source"] == "manifest"
    assert Path(info["prompt_manifest_jsonl"]).exists()
