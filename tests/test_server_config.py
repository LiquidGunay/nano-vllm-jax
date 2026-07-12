import json
from pathlib import Path

import pytest
import yaml

from nanovllm_jax.config import EngineConfig, ModelConfig, RuntimeConfig, WarmupConfig, load_engine_config
from nanovllm_jax.fastpath import FASTPATH, engine_overrides


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_config_has_no_speculative_surface():
    config = RuntimeConfig()

    assert not hasattr(config, "speculative_method")
    assert not hasattr(config, "num_speculative_tokens")


def test_engine_config_validates_capacity():
    with pytest.raises(ValueError, match="max_num_resident_seqs"):
        EngineConfig(max_num_seqs=8, max_num_resident_seqs=4)

    with pytest.raises(ValueError, match="kv_cache_bytes"):
        EngineConfig(kv_cache_bytes=0)


def test_engine_config_is_strict_and_parses_canonical_capacity():
    with pytest.raises(ValueError, match="unknown engine keys"):
        EngineConfig.from_mapping({"kv_cache_mb": 1024})

    config = EngineConfig.from_mapping(
        {
            "max_prefill": 128,
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


def test_engine_config_projects_only_capacity_plus_fastpath_for_engine():
    config = EngineConfig(
        max_prefill=128,
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

    projected = config.to_engine_kwargs()
    for key, value in engine_overrides(FASTPATH).items():
        assert projected[key] == value

    assert projected["max_num_seqs"] == 2
    assert projected["max_num_resident_seqs"] == 3
    assert projected["max_num_batched_tokens"] == 512
    assert projected["max_blocks_per_seq"] == 64
    assert projected["max_kv_cache_bytes"] == 256 * 1024 * 1024
    assert projected["prefill_token_buckets"] == (64, 128, 512)
    assert projected["decode_block_table_buckets"] == (64,)
    assert projected["prefix_cache"] is False


def test_engine_config_rejects_unsorted_or_uncovered_buckets():
    with pytest.raises(ValueError, match="sorted, unique"):
        EngineConfig(prefill_token_buckets=(128, 64))
    with pytest.raises(ValueError, match="cover max_num_seqs"):
        EngineConfig(batch_size_buckets=(1, 4))


def test_model_config_is_read_from_checkpoint(tmp_path):
    text = {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "layer_types": tuple(
            "linear_attention" if index % 4 != 3 else "full_attention"
            for index in range(32)
        ),
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "attention_bias": False,
        "tie_word_embeddings": True,
        "eos_token_id": 248044,
        "mlp_only_layers": [],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": text})
    )

    model = ModelConfig.from_checkpoint(tmp_path, model="Qwen/Qwen3.5-4B")
    runtime = RuntimeConfig.from_model_config(model)

    assert model.hidden_size == runtime.hidden_size == 2560
    assert model.num_hidden_layers == runtime.num_hidden_layers == 32
    assert model.linear_num_value_heads == runtime.linear_num_value_heads == 32
