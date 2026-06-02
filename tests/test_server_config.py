"""Tests for typed runtime/kernel configuration loading."""

import os

import pytest

from nanovllm_jax.server_config import load_server_config, runtime_env_from_config


def test_runtime_and_kernel_sections_translate_to_env():
    env = runtime_env_from_config(
        {
            "runtime": {
                "platform": "cuda",
                "tokenizers_parallelism": False,
                "fastpaths": {
                    "greedy_token": True,
                    "device_token_carry": True,
                    "static_decode_metadata": True,
                    "lm_head_decode_act_dtype": "bf16",
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
                },
            },
            "kernels": {
                "backend": "pure_jax",
                "gdn": {
                    "disable_fallbacks": True,
                    "prefill_post_conv_impl": "triton_fla_padded",
                    "prefill_block_dot": True,
                    "packed_decode": {
                        "impl": "triton_fla",
                        "qkv_dtype": "bf16",
                        "pre_normalize_qk": True,
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
    assert env["NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH"] == "1"
    assert env["NANO_VLLM_JAX_DEVICE_TOKEN_CARRY"] == "1"
    assert env["NANO_VLLM_JAX_STATIC_DECODE_METADATA"] == "1"
    assert env["NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE"] == "bf16"
    assert env["NANO_VLLM_JAX_DECODE_PROJ_ACT_DTYPE"] == "bf16_single_seq"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM"] == "1"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM_GATE_UP"] == "1"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS"] == "8"
    assert env["NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM"] == "8192"
    assert env["NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM"] == "1"
    assert env["NANO_VLLM_JAX_TRITON_DECODE_RMSNORM"] == "1"
    assert env["NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM"] == "1"
    assert env["NANO_VLLM_JAX_KERNEL_BACKEND"] == "pure_jax"
    assert env["NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL"] == "triton_fla_padded"
    assert env["NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_FWD_O_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT"] == "1"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL"] == "triton_fla"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE"] == "bf16"
    assert env["NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK"] == "1"
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
