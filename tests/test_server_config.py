import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from nanovllm_jax.config import (
    EngineConfig,
    ModelConfig,
    ModelSpec,
    RuntimeSpec,
    ServerSettings,
    WarmupConfig,
    load_engine_config,
)
from nanovllm_jax.device_batch import HostBatch
from nanovllm_jax.engine import LLMEngine, _engine_config_from_public_kwargs
from nanovllm_jax.fastpath import KERNEL_PLAN
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams
from tests.runtime_specs import qwen_text_config, runtime_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_checkpoint(path: Path, text: dict) -> None:
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": text}))


def test_server_import_has_no_jax_runtime_side_effects():
    env = dict(os.environ)
    for name in (
        "JAX_PLATFORMS",
        "JAX_COMPILATION_CACHE_DIR",
        "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os, sys, server; "
            "values = {k: os.environ.get(k) for k in "
            "('JAX_PLATFORMS', 'JAX_COMPILATION_CACHE_DIR', "
            "'NANO_VLLM_JAX_COMPILE_CACHE_DIR')}; "
            "values['jax_imported'] = 'jax' in sys.modules; "
            "values['engine_imported'] = 'nanovllm_jax.engine' in sys.modules; "
            "print(json.dumps(values))",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "JAX_PLATFORMS": None,
        "JAX_COMPILATION_CACHE_DIR": None,
        "NANO_VLLM_JAX_COMPILE_CACHE_DIR": None,
        "jax_imported": False,
        "engine_imported": False,
    }


def test_server_app_factory_registers_only_transport_routes():
    import server

    application = server.create_app(ServerSettings(max_tokens_default=17))

    assert application.config["MAX_TOKENS_DEFAULT"] == 17
    assert {rule.rule for rule in application.url_map.iter_rules()} == {
        "/static/<path:filename>",
        "/health",
        "/v1/generate",
        "/v1/generate_stream",
        "/v1/completions",
    }


def test_server_shutdown_stops_service_before_engine(monkeypatch):
    import server

    events = []
    old_service = SimpleNamespace(stop=lambda: events.append("service"))
    old_engine = SimpleNamespace(close=lambda: events.append("engine"))
    monkeypatch.setattr(server, "service", old_service)
    monkeypatch.setattr(server, "engine", old_engine)

    server.shutdown_engine()

    assert events == ["service", "engine"]
    assert server.service is None
    assert server.engine is None


def test_completion_ids_are_unique():
    import server

    results = [{"text": "x", "token_ids": [1], "finish_reason": "length"}]
    first = server._completion_payload(results, [1], 0.1)
    second = server._completion_payload(results, [1], 0.1)

    assert first["id"].startswith("cmpl-")
    assert first["id"] != second["id"]


def test_runtime_config_has_no_speculative_surface():
    config = runtime_spec()

    assert not hasattr(config, "speculative_method")
    assert not hasattr(config, "num_speculative_tokens")
    with pytest.raises(TypeError):
        RuntimeSpec()


def test_materialized_host_batch_has_one_writer():
    assert "hybrid_slot_ids" not in HostBatch.__dataclass_fields__


def test_engine_config_validates_capacity():
    with pytest.raises(ValueError, match="max_num_resident_seqs"):
        EngineConfig(max_num_seqs=8, max_num_resident_seqs=4)

    with pytest.raises(ValueError, match="kv_cache_bytes"):
        EngineConfig(kv_cache_bytes=0)


def test_engine_config_is_strict_and_parses_canonical_capacity():
    with pytest.raises(ValueError, match="unknown engine keys"):
        EngineConfig.from_mapping({"kv_cache_mb": 1024})
    with pytest.raises(ValueError, match="unknown engine keys: max_prefill"):
        EngineConfig.from_mapping({"max_prefill": 128})

    assert not hasattr(EngineConfig(), "max_prefill")

    config = EngineConfig.from_mapping(
        {
            "max_num_seqs": 2,
            "max_num_resident_seqs": 2,
            "max_num_batched_tokens": 128,
            "max_blocks_per_seq": 128,
            "kv_cache_bytes": 1024 * 1024 * 1024,
            "prefill_token_buckets": "64,128",
            "batch_size_buckets": [1, 2],
            "decode_block_buckets": "128,320",
            "warmup": {
                "enabled": False,
                "prefill_token_buckets": "64",
                "batch_size_buckets": "1",
                "decode_block_buckets": [128],
                "include_sampled_routes": False,
            },
        }
    )

    assert config.kv_cache_bytes == 1024 * 1024 * 1024
    assert config.prefill_token_buckets == (64, 128)
    assert config.batch_size_buckets == (1, 2)
    assert config.decode_block_buckets == (128, 320)
    assert config.warmup == WarmupConfig(
        prefill_token_buckets=(64,),
        batch_size_buckets=(1,),
        decode_block_buckets=(128,),
        include_sampled_routes=False,
        enabled=False,
    )


def test_shorthand_bucket_overrides_derive_matching_warmup():
    config = _engine_config_from_public_kwargs(
        "Qwen/Qwen3.5-0.8B",
        {
            "max_num_seqs": 1,
            "max_num_batched_tokens": 64,
            "max_blocks_per_seq": 64,
            "prefill_token_buckets": (64,),
            "batch_size_buckets": (1,),
            "decode_block_buckets": (64,),
        },
    )

    assert config.warmup.prefill_token_buckets == (64,)
    assert config.warmup.batch_size_buckets == (1,)
    assert config.warmup.decode_block_buckets == (64,)


def test_server_yaml_is_the_only_committed_serving_config():
    raw = yaml.safe_load((REPO_ROOT / "server.yaml").read_text()) or {}

    assert "fastpath" not in raw
    assert "kernels" not in raw
    assert "runtime" not in raw
    assert not (REPO_ROOT / "server_config.yaml").exists()
    assert not (REPO_ROOT / "configs").exists()

    settings = load_engine_config(REPO_ROOT / "server.yaml")

    assert settings.host == "127.0.0.1"
    assert settings.port == 6791
    assert settings.max_tokens_default == 128
    assert settings.engine.model == "Qwen/Qwen3.5-0.8B"
    assert settings.engine.max_num_seqs == 8
    assert settings.engine.prefix_cache is True
    assert settings.engine.warmup.enabled is True


def test_runtime_spec_composes_engine_capacity_and_kernel_policy():
    config = EngineConfig(
        max_num_seqs=2,
        max_num_resident_seqs=3,
        max_num_batched_tokens=512,
        max_blocks_per_seq=64,
        kv_cache_bytes=256 * 1024 * 1024,
        prefill_token_buckets=(64, 128, 512),
        batch_size_buckets=(1, 2, 3),
        decode_block_buckets=(64,),
        warmup=WarmupConfig(
            prefill_token_buckets=(64, 128),
            batch_size_buckets=(1, 2),
            decode_block_buckets=(64,),
            include_sampled_routes=False,
        ),
        prefix_cache=False,
    )

    runtime = RuntimeSpec.promoted(ModelConfig(), config)

    assert runtime.kernels == KERNEL_PLAN
    assert runtime.capacity.max_num_seqs == 2
    assert runtime.capacity.max_num_resident_seqs == 3
    assert runtime.capacity.max_num_batched_tokens == 512
    assert runtime.capacity.max_blocks_per_seq == 64
    assert runtime.capacity.max_kv_cache_bytes == 256 * 1024 * 1024
    assert runtime.capacity.prefix_cache is False
    assert runtime.compile.prefill_token_buckets == (64, 128, 512)
    assert runtime.compile.decode_block_table_buckets == (64,)

    import server

    manifest = server._runtime_manifest(runtime)
    assert tuple(manifest) == ("model", "capacity", "compile", "kernels")
    assert manifest["compile"]["dtype"] == "bfloat16"
    assert manifest["kernels"]["full_attention_decode"] == "flashinfer_paged"
    assert "layer_types" not in json.dumps(manifest)
    assert "decode_padded_gemm" not in json.dumps(manifest)


def test_server_request_validation_reads_runtime_capacity(monkeypatch):
    import server

    runtime = runtime_spec(
        capacity={
            "block_size": 4,
            "max_blocks_per_seq": 2,
            "max_num_seqs": 1,
            "max_num_resident_seqs": 1,
        }
    )
    engine = object.__new__(LLMEngine)
    engine.config = runtime
    engine.scheduler = Scheduler(runtime)
    monkeypatch.setattr(server, "engine", engine)

    server._validate_inputs_fit_config(
        [[1, 2]],
        [2],
        [SamplingParams(max_tokens=6)],
    )
    with pytest.raises(ValueError, match="per-sequence capacity is 8"):
        server._validate_inputs_fit_config(
            [[1, 2]],
            [2],
            [SamplingParams(max_tokens=7)],
        )
    with pytest.raises(ValueError, match="exceeding max_num_seqs 1"):
        server._validate_inputs_fit_config(
            [[1], [2]],
            [1, 1],
            [SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        )


def test_server_capacity_validation_is_pairwise(monkeypatch):
    import server

    runtime = runtime_spec(
        capacity={
            "block_size": 1,
            "max_blocks_per_seq": 101,
            "max_num_seqs": 2,
            "max_num_resident_seqs": 2,
        }
    )
    engine = object.__new__(LLMEngine)
    engine.config = runtime
    engine.scheduler = Scheduler(runtime)
    monkeypatch.setattr(server, "engine", engine)

    server._validate_inputs_fit_config(
        [[1], [2]],
        [100, 1],
        [SamplingParams(max_tokens=1), SamplingParams(max_tokens=100)],
    )
    with pytest.raises(ValueError, match=r"request\[1\] needs 102 total tokens"):
        server._validate_inputs_fit_config(
            [[1], [2]],
            [100, 1],
            [SamplingParams(max_tokens=1), SamplingParams(max_tokens=101)],
        )


@pytest.mark.parametrize(
    "data",
    (
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"temperature": float("nan")},
        {"ignore_eos": "false"},
    ),
)
def test_server_sampling_validation_is_strict(data):
    import server

    with pytest.raises(ValueError):
        server._sampling_params(data, 1)


def test_engine_config_rejects_unsorted_or_uncovered_buckets():
    with pytest.raises(ValueError, match="sorted, unique"):
        EngineConfig(prefill_token_buckets=(128, 64))
    with pytest.raises(ValueError, match="cover max_num_seqs"):
        EngineConfig(batch_size_buckets=(1, 4))


@pytest.mark.parametrize("size", ("0.8B", "2B", "4B"))
def test_model_config_is_read_from_checkpoint(tmp_path, size):
    text = qwen_text_config(size)
    _write_checkpoint(tmp_path, text)

    model = ModelConfig.from_checkpoint(tmp_path, model=f"Qwen/Qwen3.5-{size}")

    assert model.hidden_size == text["hidden_size"]
    assert model.num_hidden_layers == text["num_hidden_layers"]
    assert model.linear_num_value_heads == text["linear_num_value_heads"]


def test_only_validated_model_type_parses_checkpoints(tmp_path):
    assert not hasattr(ModelSpec, "from_checkpoint")

    text = qwen_text_config()
    text["use_qk_norm_in_gdn"] = False
    _write_checkpoint(tmp_path, text)

    with pytest.raises(ValueError, match="use_qk_norm_in_gdn=False"):
        ModelConfig.from_checkpoint(tmp_path, model="Qwen/Qwen3.5-4B")


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("vocab_size",), 248000),
        (("head_dim",), 128),
        (("linear_key_head_dim",), 64),
        (("linear_value_head_dim",), 64),
        (("linear_conv_kernel_dim",), 3),
        (("full_attention_interval",), 8),
        (("hidden_act",), "gelu"),
        (("rms_norm_eps",), 1e-5),
        (("attention_dropout",), 0.1),
        (("attention_bias",), True),
        (("attn_output_gate",), False),
        (("mamba_ssm_dtype",), "bfloat16"),
        (("tie_word_embeddings",), False),
        (("max_position_embeddings",), 131072),
        (("rope_parameters", "rope_theta"), 1_000_000),
        (("rope_parameters", "partial_rotary_factor"), 0.5),
        (("rope_parameters", "mrope_section"), [16, 8, 8]),
        (("rope_parameters", "mrope_interleaved"), False),
    ),
)
def test_model_config_rejects_unvalidated_architecture_field(tmp_path, path, value):
    text = qwen_text_config()
    target = text
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    _write_checkpoint(tmp_path, text)

    with pytest.raises(ValueError, match="unsupported Qwen3.5"):
        ModelConfig.from_checkpoint(tmp_path, model="Qwen/Qwen3.5-4B")


def test_model_config_rejects_altered_layer_order(tmp_path):
    text = qwen_text_config()
    text["layer_types"][0] = "full_attention"
    _write_checkpoint(tmp_path, text)

    with pytest.raises(ValueError, match="layer_types"):
        ModelConfig.from_checkpoint(tmp_path, model="Qwen/Qwen3.5-4B")
