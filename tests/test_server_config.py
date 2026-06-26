from pathlib import Path

import pytest
import yaml

from nanovllm_jax.config import EngineConfig, Qwen3_5Config, WarmupConfig, load_engine_config
from nanovllm_jax.fastpath import FASTPATH, engine_overrides


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_qwen_config_has_no_speculative_surface():
    config = Qwen3_5Config()

    assert not hasattr(config, "speculative_method")
    assert not hasattr(config, "num_speculative_tokens")


def test_engine_config_validates_capacity():
    with pytest.raises(ValueError, match="max_num_resident_seqs"):
        EngineConfig(max_num_seqs=8, max_num_resident_seqs=4)

    with pytest.raises(ValueError, match="kv_cache_bytes"):
        EngineConfig(kv_cache_bytes=0)


def test_engine_config_parses_capacity_aliases():
    config = EngineConfig.from_mapping(
        {
            "kv_cache_mb": 1024,
            "prefill_token_buckets": "64,128",
            "batch_size_buckets": [1, 2],
            "decode_block_table_buckets": "128,320",
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
        max_num_seqs=2,
        max_num_resident_seqs=3,
        max_num_batched_tokens=512,
        max_blocks_per_seq=64,
        kv_cache_bytes=256 * 1024 * 1024,
        prefill_token_buckets=(64, 128),
        batch_size_buckets=(1, 2),
        decode_block_buckets=(64,),
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
    assert projected["prefill_token_buckets"] == (64, 128)
    assert projected["decode_block_table_buckets"] == (64,)
    assert projected["prefix_cache"] is False
