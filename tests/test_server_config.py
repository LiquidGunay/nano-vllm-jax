"""Tests for typed runtime/kernel configuration loading."""

import os

import pytest

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.server_config import (
    engine_overrides_from_config,
    load_server_config,
    runtime_env_from_config,
)


def test_speculative_config_legacy_num_tokens_selects_mtp():
    config = Qwen3_5Config(num_speculative_tokens=1)

    assert config.speculative_method == "mtp"
    assert config.num_speculative_tokens == 1
    assert config.draft_sample_method == "greedy"
    assert config.mtp_verifier_impl == "two_decode"


def test_speculative_config_rejects_unimplemented_probabilistic_mtp():
    with pytest.raises(ValueError, match="probabilistic"):
        Qwen3_5Config(
            speculative_method="mtp",
            num_speculative_tokens=1,
            draft_sample_method="probabilistic",
        )


@pytest.mark.parametrize(
    "field",
    ["mtp_unverified_draft_append", "mtp_unverified_fused_append"],
)
def test_speculative_config_rejects_unverified_mtp_append(field):
    with pytest.raises(ValueError, match="Unverified MTP draft append"):
        Qwen3_5Config(
            speculative_method="mtp",
            num_speculative_tokens=1,
            **{field: True},
        )


def test_runtime_and_kernel_sections_translate_to_env():
    env = runtime_env_from_config(
        {
            "runtime": {
                "platform": "cuda",
                "tokenizers_parallelism": False,
                "fastpaths": {
                    "greedy_token": True,
                    "compact_prefill_token_count_mode": "bucket",
                    "device_token_carry": True,
                    "static_decode_metadata": True,
                    "static_decode_seq_lens_carry": True,
                    "resident_decode_metadata": True,
                    "greedy_decode_burst_steps": 2,
                    "trace_token_prefetch": True,
                    "lm_head_decode_act_dtype": "bf16",
                    "lm_head_topk_impl": "flashinfer",
                    "lm_head_greedy_top1_impl": "cutlass",
                    "decode_proj_act_dtype": "bf16_single_seq",
                    "decode_padded_gemm": True,
                    "decode_padded_gemm_gate_up": True,
                    "decode_padded_gemm_rows": 8,
                    "decode_padded_gemm_max_out_dim": 8192,
                    "pallas_decode_rmsnorm": True,
                    "triton_decode_rmsnorm": True,
                    "pallas_gdn_qk_prenorm": True,
                },
                "xla": {
                    "preallocate": False,
                    "gpu_allocator": "cuda_malloc_async",
                    "command_buffer": {
                        "enable_during_profiling": True,
                        "unroll_loops": True,
                        "graph_min_size": 1,
                    },
                },
            },
            "kernels": {
                "backend": "pure_jax",
                "full_attention": {
                    "kv_cache_dtype": "bf16",
                },
                "gdn": {
                    "disable_fallbacks": True,
                    "prefill_post_conv_impl": "triton_fla_padded",
                    "prefill_block_dot": True,
                    "packed_decode": {
                        "impl": "triton_fla",
                        "qkv_dtype": "bf16",
                        "pre_normalize_qk": True,
                        "max_batch": 1,
                        "triton": {
                            "num_warps": 8,
                            "num_stages": 2,
                            "block_v": 32,
                        },
                    },
                },
            },
        }
    )

    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["TOKENIZERS_PARALLELISM"] == "0"
    assert env["XLA_FLAGS"] == (
        "--xla_gpu_autotune_level=4 "
        "--xla_enable_command_buffers_during_profiling=true "
        "--xla_gpu_command_buffer_unroll_loops=true "
        "--xla_gpu_graph_min_graph_size=1"
    )
    assert env["NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH"] == "1"
    assert env["NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE"] == "bucket"
    assert env["NANO_VLLM_JAX_DEVICE_TOKEN_CARRY"] == "1"
    assert env["NANO_VLLM_JAX_STATIC_DECODE_METADATA"] == "1"
    assert env["NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY"] == "1"
    assert env["NANO_VLLM_JAX_RESIDENT_DECODE_METADATA"] == "1"
    assert env["NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS"] == "2"
    assert env["NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH"] == "1"
    assert env["NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE"] == "bf16"
    assert env["NANO_VLLM_JAX_LM_HEAD_TOPK_IMPL"] == "flashinfer"
    assert "NANO_VLLM_JAX_LM_HEAD_GREEDY_TOP1_IMPL" not in env
    assert env["NANO_VLLM_JAX_DECODE_PROJ_ACT_DTYPE"] == "bf16_single_seq"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM"] == "1"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM_GATE_UP"] == "1"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS"] == "8"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM"] == "8192"
    assert env["NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM"] == "1"
    assert env["NANO_VLLM_JAX_TRITON_DECODE_RMSNORM"] == "1"
    assert env["NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM"] == "1"
    assert env["NANO_VLLM_JAX_KERNEL_BACKEND"] == "pure_jax"
    assert env["NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE"] == "bf16"
    assert env["NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL"] == "triton_fla_padded"
    assert env["NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_FWD_O_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL"] == "triton_fla"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE"] == "bf16"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_MAX_BATCH"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_WARPS"] == "8"


def test_legacy_env_section_preserves_raw_env_names_and_short_nano_keys():
    env = runtime_env_from_config(
        {
            "env": {
                "JAX_PLATFORMS": "cuda",
                "GREEDY_TOKEN_FASTPATH": "1",
                "NANO_VLLM_JAX_COMPACT_PREFILL_MLP": "1",
            }
        }
    )

    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH"] == "1"
    assert env["NANO_VLLM_JAX_COMPACT_PREFILL_MLP"] == "1"


def test_runtime_fastpaths_project_to_engine_config(tmp_path, monkeypatch):
    for key in (
        "NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH",
        "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        "NANO_VLLM_JAX_STATIC_DECODE_METADATA",
        "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
        "NANO_VLLM_JAX_RESIDENT_DECODE_METADATA",
        "NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS",
        "NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH",
        "NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD",
        "NANO_VLLM_JAX_COMPACT_PREFILL_MLP",
        "NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE",
        "NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS",
    ):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "server_config.yaml"
    config.write_text(
        """
runtime:
  fastpaths:
    greedy_token: true
    device_token_carry: true
    static_decode_metadata: true
    static_decode_seq_lens_carry: true
    resident_decode_metadata: true
    greedy_decode_burst_steps: 3
    trace_token_prefetch: false
    materialize_tied_lm_head: true
    compact_prefill_mlp: true
    lm_head_decode_act_dtype: bf16
    lm_head_greedy_top1_impl: cutlass
    decode_rms_padded_gemm: true
    decode_padded_gemm_rows: 8
""".strip()
    )

    loaded = load_server_config(config)

    assert loaded.engine["greedy_token_fastpath"] is True
    assert loaded.engine["device_token_carry"] is True
    assert loaded.engine["static_decode_metadata"] is True
    assert loaded.engine["static_decode_seq_lens_carry"] is True
    assert loaded.engine["resident_decode_metadata"] is True
    assert loaded.engine["greedy_decode_burst_steps"] == 3
    assert loaded.engine["trace_token_prefetch"] is False
    assert loaded.engine["materialize_tied_lm_head"] is True
    assert loaded.engine["compact_prefill_mlp"] is True
    assert loaded.engine["lm_head_decode_act_dtype"] == "bf16"
    assert loaded.engine["lm_head_greedy_top1_impl"] == "cutlass"
    assert loaded.engine["decode_rms_padded_gemm"] is True
    assert loaded.engine["decode_padded_gemm_rows"] == 8


def test_engine_config_supports_speculative_fields(tmp_path, monkeypatch):
    for key in (
        "NANO_VLLM_JAX_SPECULATIVE_METHOD",
        "NANO_VLLM_JAX_DRAFT_SAMPLE_METHOD",
        "NANO_VLLM_JAX_MTP_VERIFIER_IMPL",
        "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
        "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
    ):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "server_config.yaml"
    config.write_text(
        """
engine:
  speculative_method: mtp
  num_speculative_tokens: 1
  draft_sample_method: greedy
  mtp_verifier_impl: two_decode
  mtp_batch_accept_policy: rowwise
  mtp_seed_after_bonus: true
""".strip()
    )

    loaded = load_server_config(config)

    assert loaded.engine["speculative_method"] == "mtp"
    assert loaded.engine["num_speculative_tokens"] == 1
    assert loaded.engine["draft_sample_method"] == "greedy"
    assert loaded.engine["mtp_verifier_impl"] == "two_decode"
    assert loaded.engine["mtp_batch_accept_policy"] == "rowwise"
    assert loaded.engine["mtp_seed_after_bonus"] is True


def test_engine_config_supports_resident_sequence_capacity(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_MAX_NUM_RESIDENT_SEQS", raising=False)
    config = tmp_path / "server_config.yaml"
    config.write_text(
        """
engine:
  max_num_seqs: 8
  max_num_resident_seqs: 16
""".strip()
    )

    loaded = load_server_config(config)

    assert loaded.engine["max_num_seqs"] == 8
    assert loaded.engine["max_num_resident_seqs"] == 16

    monkeypatch.setenv("NANO_VLLM_JAX_MAX_NUM_RESIDENT_SEQS", "12")
    loaded = load_server_config(config)

    assert loaded.engine["max_num_resident_seqs"] == 12


def test_engine_overrides_from_config_merges_runtime_fastpaths_and_kernel_policy():
    overrides = engine_overrides_from_config(
        {
            "runtime": {
                "fastpaths": {
                    "materialize_tied_lm_head": True,
                    "trace_token_prefetch": False,
                    "compact_prefill_in_proj_qkv": True,
                    "lm_head_decode_act_dtype": "bf16",
                    "lm_head_topk_impl": "flashinfer",
                    "lm_head_greedy_top1_impl": "cutlass",
                    "decode_padded_gemm": True,
                    "decode_rms_padded_gemm": True,
                },
            },
            "kernels": {
                "full_attention": {
                    "kv_cache_dtype": "bf16",
                    "kv_append_impl": "reference",
                    "decode_impl": "triton_paged",
                    "prefill_impl": "triton_packed",
                },
                "gdn": {
                    "disable_fallbacks": True,
                    "prefill_post_conv_impl": "triton_fla_padded",
                    "packed_decode": {
                        "impl": "triton_fla_conv_raw_gates",
                        "qkv_dtype": "bf16",
                    },
                },
            },
        }
    )

    assert overrides["materialize_tied_lm_head"] is True
    assert overrides["trace_token_prefetch"] is False
    assert overrides["compact_prefill_in_proj_qkv"] is True
    assert overrides["lm_head_decode_act_dtype"] == "bf16"
    assert overrides["lm_head_topk_impl"] == "flashinfer"
    assert overrides["lm_head_greedy_top1_impl"] == "cutlass"
    assert overrides["decode_padded_gemm"] is True
    assert overrides["decode_rms_padded_gemm"] is True
    assert overrides["full_attention_kv_cache_dtype"] == "bf16"
    assert overrides["full_attention_kv_append_impl"] == "reference"
    assert overrides["full_attention_decode_impl"] == "triton_paged"
    assert overrides["full_attention_prefill_impl"] == "triton_packed"
    assert overrides["gdn_disable_fallbacks"] is True
    assert overrides["gdn_prefill_post_conv_impl"] == "triton_fla_padded"
    assert overrides["gdn_packed_decode_impl"] == "triton_fla_conv_raw_gates"
    assert overrides["gdn_packed_decode_qkv_dtype"] == "bf16"


def test_kernel_policy_projects_to_engine_config(tmp_path, monkeypatch):
    for key in (
        "NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE",
        "NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS",
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL",
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE",
    ):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "server_config.yaml"
    config.write_text(
        """
kernels:
  full_attention:
    kv_cache_dtype: bf16
    kv_append_impl: reference
    decode_impl: triton_paged
    prefill_impl: triton_packed
  gdn:
    disable_fallbacks: true
    prefill_post_conv_impl: triton_fla_padded
    prefill_qkv_dtype: bf16
    packed_decode:
      impl: triton_fla_conv_raw_gates
      qkv_dtype: bf16
      max_batch: 8
""".strip()
    )

    loaded = load_server_config(config)

    assert loaded.engine["full_attention_kv_cache_dtype"] == "bf16"
    assert loaded.engine["full_attention_kv_append_impl"] == "reference"
    assert loaded.engine["full_attention_decode_impl"] == "triton_paged"
    assert loaded.engine["full_attention_prefill_impl"] == "triton_packed"
    assert loaded.engine["gdn_disable_fallbacks"] is True
    assert loaded.engine["gdn_prefill_post_conv_impl"] == "triton_fla_padded"
    assert loaded.engine["gdn_prefill_qkv_dtype"] == "bf16"
    assert loaded.engine["gdn_packed_decode_impl"] == "triton_fla_conv_raw_gates"
    assert loaded.engine["gdn_packed_decode_qkv_dtype"] == "bf16"
    assert loaded.engine["gdn_packed_decode_max_batch"] == 8


def test_runtime_config_rejects_local_cuda_probes_by_default():
    with pytest.raises(ValueError, match="Local CUDA/JAX FFI probes are disabled"):
        runtime_env_from_config(
            {
                "kernels": {
                    "gdn": {
                        "prefill_post_conv_impl": "cuda_prep_prefill_fp32",
                    }
                }
            }
        )


def test_runtime_config_allows_explicit_local_cuda_probe_diagnostics():
    env = runtime_env_from_config(
        {
            "kernels": {
                "allow_local_cuda_probes": True,
                "gdn": {
                    "prefill_post_conv_impl": "cuda_prep_prefill_fp32",
                },
            }
        }
    )

    assert env["NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL"] == "cuda_prep_prefill_fp32"


def test_backend_local_cuda_probe_routes_require_explicit_opt_in(monkeypatch):
    from nanovllm_jax import backends

    monkeypatch.delenv("NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES", raising=False)
    with pytest.raises(RuntimeError, match="local CUDA/JAX FFI probe code"):
        backends._require_local_cuda_probe_opt_in("CUDA FP32 packed GDN decode")

    monkeypatch.setenv("NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES", "1")
    backends._require_local_cuda_probe_opt_in("CUDA FP32 packed GDN decode")


def test_load_server_config_applies_typed_runtime_env(tmp_path, monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", raising=False)
    config = tmp_path / "server_config.yaml"
    config.write_text(
        """
server:
  port: 9090
runtime:
  platform: cuda
kernels:
  gdn:
    packed_decode:
      impl: triton_fla
""".strip()
    )

    loaded = load_server_config(config)

    assert loaded.server["port"] == 9090
    assert loaded.env["JAX_PLATFORMS"] == "cuda"
    assert loaded.env["NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL"] == "triton_fla"
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    assert os.environ["NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL"] == "triton_fla"
