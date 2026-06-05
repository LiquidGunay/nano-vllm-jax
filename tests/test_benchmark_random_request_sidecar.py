"""Unit tests for the random-request benchmark sidecar."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import benchmark_random_request_sidecar as sidecar


def test_parse_args_defaults_set_random_ranges():
    args = sidecar.parse_args(["--output-json", "/tmp/sidecar.json"])

    assert args.min_input_tokens == 512
    assert args.max_input_tokens == 4096
    assert args.min_output_tokens == 256
    assert args.max_output_tokens == 1024
    assert args.min_request_count == 5
    assert args.max_request_count == 15
    assert args.seed == 1234
    assert args.jax_profile is False
    assert args.jax_warmup_mode == "generic"
    assert args.jax_fail_on_jit_cache_growth is False
    assert args.jax_config == ""
    assert args.max_num_resident_seqs == 0
    assert args.decode_block_table_buckets == ""
    assert args.resident_decode_metadata is False
    assert args.full_attention_kv_cache_dtype == "default"
    assert args.full_attention_kv_append_impl == "reference"
    assert args.full_attention_decode_impl == "reference"
    assert args.full_attention_prefill_impl == "reference"
    assert args.vllm_dtype == ""
    assert sidecar._effective_vllm_dtype(args) == "bfloat16"
    assert args.vllm_num_speculative_tokens == 0
    assert args.max_system_ram_percent == 70.0
    assert args.worker_cpu_cores >= 1
    assert args.worker_nice == 10


def test_parse_args_rejects_invalid_ranges():
    with pytest.raises(SystemExit):
        sidecar.parse_args(["--min-input-tokens", "128", "--max-input-tokens", "64"])

    with pytest.raises(SystemExit):
        sidecar.parse_args(["--min-request-count", "16", "--max-request-count", "15"])

    with pytest.raises(SystemExit):
        sidecar.parse_args(["--max-system-ram-percent", "100"])

    with pytest.raises(SystemExit):
        sidecar.parse_args(["--worker-cpu-cores", "-1"])


def test_deterministic_request_rows_have_same_contents():
    first = sidecar.generate_random_request_rows(
        seed=123,
        num_requests=8,
        min_input_tokens=16,
        max_input_tokens=24,
        min_output_tokens=4,
        max_output_tokens=6,
        token_vocab_size=1024,
        eos_token_id=1,
    )
    second = sidecar.generate_random_request_rows(
        seed=123,
        num_requests=8,
        min_input_tokens=16,
        max_input_tokens=24,
        min_output_tokens=4,
        max_output_tokens=6,
        token_vocab_size=1024,
        eos_token_id=1,
    )

    assert first == second


def test_build_random_request_suite_respects_ranges_and_counts():
    parser = sidecar.build_arg_parser()
    args = parser.parse_args(
        [
            "--seed",
            "17",
            "--min-input-tokens",
            "10",
            "--max-input-tokens",
            "12",
            "--min-output-tokens",
            "4",
            "--max-output-tokens",
            "6",
            "--min-request-count",
            "3",
            "--max-request-count",
            "6",
            "--output-json",
            "/tmp/sidecar.json",
        ]
    )
    rows, metadata = sidecar.build_random_request_suite(args, token_vocab_size=1000, eos_token_id=2)

    assert args.min_request_count <= len(rows) <= args.max_request_count
    assert all(10 <= row["prompt_len"] <= 12 for row in rows)
    assert all(4 <= row["output_len"] <= 6 for row in rows)
    assert metadata["request_count"] == len(rows)
    assert metadata["prompt_len_min"] >= 10
    assert metadata["prompt_len_max"] <= 12


def test_baseline_vllm_command_omits_speculative_args():
    args = sidecar.parse_args(["--output-json", "/tmp/sidecar.json"])
    command = sidecar._build_vllm_command(
        args,
        manifest_jsonl=sidecar.Path("/tmp/prompts.jsonl"),
        output_json=sidecar.Path("/tmp/vllm.json"),
    )

    assert "--mode" in command
    assert "baseline" in command
    assert "--dtype" in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--speculative-method" not in command
    assert "--num-speculative-tokens" not in command


def test_vllm_dtype_override_is_used():
    args = sidecar.parse_args(
        ["--output-json", "/tmp/sidecar.json", "--vllm-dtype", "float16"]
    )
    command = sidecar._build_vllm_command(
        args,
        manifest_jsonl=sidecar.Path("/tmp/prompts.jsonl"),
        output_json=sidecar.Path("/tmp/vllm.json"),
    )

    assert command[command.index("--dtype") + 1] == "float16"


def test_jax_command_uses_generic_warmup_controls():
    args = sidecar.parse_args(
        [
            "--output-json",
            "/tmp/sidecar.json",
            "--jax-warmup-mode",
            "generic",
            "--jax-fail-on-jit-cache-growth",
            "--max-num-resident-seqs",
            "16",
            "--decode-block-table-buckets",
            "128,256,320",
            "--resident-decode-metadata",
            "--full-attention-kv-cache-dtype",
            "bf16",
            "--full-attention-kv-append-impl",
            "flashinfer",
            "--full-attention-decode-impl",
            "triton_paged",
        ]
    )
    command = sidecar._build_jax_command(
        args,
        manifest_jsonl=sidecar.Path("/tmp/prompts.jsonl"),
        output_json=sidecar.Path("/tmp/jax.json"),
    )

    assert "--warmup" in command
    assert command[command.index("--warmup-mode") + 1] == "generic"
    assert "--fail-on-jit-cache-growth" in command
    assert command[command.index("--prefill-layout") + 1] == "packed"
    assert command[command.index("--prefill-token-buckets") + 1] == args.prefill_buckets
    assert command[command.index("--max-num-resident-seqs") + 1] == "16"
    assert command[command.index("--decode-block-table-buckets") + 1] == "128,256,320"
    assert "--resident-decode-metadata" in command
    assert command[command.index("--full-attention-kv-cache-dtype") + 1] == "bf16"
    assert command[command.index("--full-attention-kv-append-impl") + 1] == "flashinfer"
    assert command[command.index("--full-attention-decode-impl") + 1] == "triton_paged"


def test_jax_config_projects_runtime_kernel_policy(tmp_path):
    config_path = tmp_path / "random_jax_config.yaml"
    config_path.write_text(
        """
runtime:
  platform: cuda
  xla:
    flags: --xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false
  fastpaths:
    device_token_carry: true
    static_decode_metadata: true
    compact_prefill_token_count_mode: bucket
kernels:
  full_attention:
    kv_cache_dtype: bf16
    decode_impl: triton_paged
    prefill_impl: triton_packed
  gdn:
    disable_fallbacks: true
    packed_decode:
      impl: triton_fla_conv_raw_gates
      qkv_dtype: bf16
""".strip()
    )
    args = sidecar.parse_args(
        [
            "--output-json",
            "/tmp/sidecar.json",
            "--jax-config",
            str(config_path),
        ]
    )

    loaded = sidecar._load_jax_config(args.jax_config)
    command = sidecar._build_jax_command(
        args,
        manifest_jsonl=sidecar.Path("/tmp/prompts.jsonl"),
        output_json=sidecar.Path("/tmp/jax.json"),
        config_engine_overrides=loaded["engine_overrides"],
    )

    assert loaded["env"]["JAX_PLATFORMS"] == "cuda"
    assert loaded["env"]["XLA_FLAGS"] == "--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false"
    assert "--device-token-carry" in command
    assert "--static-decode-metadata" in command
    assert command[command.index("--compact-prefill-token-count-mode") + 1] == "bucket"
    assert command[command.index("--full-attention-kv-cache-dtype") + 1] == "bf16"
    assert command[command.index("--full-attention-decode-impl") + 1] == "triton_paged"
    assert command[command.index("--full-attention-prefill-impl") + 1] == "triton_packed"
    assert "--gdn-disable-fallbacks" in command
    assert command[command.index("--gdn-packed-decode-impl") + 1] == "triton_fla_conv_raw_gates"
    assert command[command.index("--gdn-packed-decode-qkv-dtype") + 1] == "bf16"
    policy = sidecar._effective_jax_kernel_policy(args, loaded["engine_overrides"])
    assert policy["full_attention_decode_impl"] == "triton_paged"
    assert policy["full_attention_prefill_impl"] == "triton_packed"
    assert policy["gdn_disable_fallbacks"] is True


def test_run_command_kills_process_when_ram_guard_trips(monkeypatch):
    monkeypatch.setattr(sidecar, "_system_ram_percent", lambda: 71.0)

    result = sidecar._run_command(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        timeout_seconds=10,
        max_system_ram_percent=70.0,
        worker_cpu_cores=0,
        worker_nice=0,
        resource_poll_seconds=0.05,
    )

    assert result["status"] == "killed_resource_limit"
    assert result["resource_limit_reason"]["kind"] == "system_ram_percent"
    assert result["resource_limit_reason"]["observed_percent"] == 71.0
