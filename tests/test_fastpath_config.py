from pathlib import Path

import yaml

from nanovllm_jax.config import load_engine_config
from nanovllm_jax.fastpath import (
    FASTPATH,
    as_manifest,
    engine_overrides,
    required_modules,
    validate_runtime_dependencies,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fastpath_manifest_is_promoted_cuda_policy():
    manifest = as_manifest()

    assert manifest["compute_dtype"] == "bfloat16"
    assert manifest["weight_dtype"] == "bfloat16"
    assert manifest["full_attention_prefill"] == "triton_packed"
    assert manifest["full_attention_decode"] == "flashinfer_paged"
    assert manifest["gdn_prefill"] == "triton_fla_padded"
    assert manifest["gdn_decode"] == "reference"
    assert manifest["lm_head_greedy"] == "triton"
    assert manifest["device_token_carry"] is True
    assert "speculative_method" not in manifest
    assert "num_speculative_tokens" not in manifest


def test_fastpath_declares_required_optional_modules(monkeypatch):
    assert set(required_modules()) == {"flashinfer", "jax_tvm_ffi", "jax_triton", "triton"}

    monkeypatch.setattr(
        "nanovllm_jax.fastpath.importlib.util.find_spec",
        lambda module: object() if module != "flashinfer" else None,
    )
    try:
        validate_runtime_dependencies()
    except RuntimeError as exc:
        assert "flashinfer" in str(exc)
    else:
        raise AssertionError("expected missing dependency failure")


def test_server_yaml_loads_capacity_without_kernel_policy():
    with (REPO_ROOT / "server.yaml").open() as fh:
        raw = yaml.safe_load(fh)

    assert "fastpath" not in raw
    assert "kernels" not in raw
    assert "runtime" not in raw
    for key in engine_overrides(FASTPATH):
        assert key not in raw["engine"]

    settings = load_engine_config(REPO_ROOT / "server.yaml")

    assert settings.engine.model == "Qwen/Qwen3.5-0.8B"
    assert settings.engine.max_num_seqs == 8
    assert settings.engine.kv_cache_bytes == 3072 * 1024 * 1024
    assert settings.engine.prefill_token_buckets == (64, 128, 256, 512, 1024, 2048, 4096)
    assert settings.engine.decode_block_buckets == (128, 256, 320)

    projected = settings.engine.to_engine_kwargs()
    fastpath = engine_overrides(FASTPATH)
    for key, value in fastpath.items():
        assert projected[key] == value
    assert projected["max_num_batched_tokens"] == 4096
