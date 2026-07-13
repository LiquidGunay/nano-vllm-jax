import json
from types import SimpleNamespace

import pytest

import nanovllm_jax.engine as engine_module
import nanovllm_jax.weights as weights_module
from nanovllm_jax.config import EngineConfig, ModelConfig, WarmupConfig
from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.step import FinishReason, RunResult
from tests.runtime_specs import qwen_text_config


def _small_engine_config(model: str) -> EngineConfig:
    return EngineConfig(
        model=model,
        max_prefill=1,
        max_num_seqs=1,
        max_num_resident_seqs=1,
        max_num_batched_tokens=1,
        max_blocks_per_seq=4,
        kv_cache_bytes=1 << 20,
        num_kvcache_blocks=4,
        prefill_token_buckets=(1,),
        batch_size_buckets=(1,),
        decode_block_buckets=(4,),
        warmup=WarmupConfig(
            prefill_token_buckets=(1,),
            batch_size_buckets=(1,),
            decode_block_buckets=(4,),
            enabled=False,
        ),
        prefix_cache=False,
    )


def test_engine_stops_on_tokenizer_eos_when_checkpoint_eos_differs(tmp_path, monkeypatch):
    model = "example/Qwen3.5"

    class FakeRunner:
        def __init__(self, config, params):
            self.config = config

        def memory_bytes(self):
            return {}

    monkeypatch.setattr(engine_module, "resolve_checkpoint_metadata", lambda _model: tmp_path)
    monkeypatch.setattr(engine_module, "resolve_checkpoint", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        engine_module.ModelConfig,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint, *, model: ModelConfig()),
    )
    monkeypatch.setattr(
        engine_module,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(eos_token_id=248046)
        ),
    )
    monkeypatch.setattr(engine_module, "load_weights_from_hf_streaming", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(engine_module, "ModelRunner", FakeRunner)

    engine = LLMEngine(model, engine_config=_small_engine_config(model))
    assert engine.config.capacity.eos_token_ids == (248044, 248046)

    seq = engine.add_request(
        [1],
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=False),
    )
    seqs, plan = engine.scheduler.schedule()
    result = engine.commit(
        seqs,
        plan,
        RunResult.from_rows([248046]),
    )

    assert result.finished[0].reason is FinishReason.EOS
    assert seq.is_finished
    assert seq.output.token_ids() == [248046]


def test_unsupported_hub_architecture_fails_before_weight_resolution(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unsupported"}))
    weight_resolutions = []

    monkeypatch.setattr(engine_module, "resolve_checkpoint_metadata", lambda _model: tmp_path)

    def resolve_weights(*args, **kwargs):
        weight_resolutions.append((args, kwargs))
        return tmp_path

    monkeypatch.setattr(engine_module, "resolve_checkpoint", resolve_weights)

    with pytest.raises(ValueError, match="unsupported model_type"):
        LLMEngine(
            "unsupported/model",
            engine_config=_small_engine_config("unsupported/model"),
        )

    assert weight_resolutions == []


def test_unsupported_gdn_norm_fails_before_weight_resolution(tmp_path, monkeypatch):
    text = qwen_text_config("0.8B")
    text["use_qk_norm_in_gdn"] = False
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": text})
    )
    weight_resolutions = []

    monkeypatch.setattr(engine_module, "resolve_checkpoint_metadata", lambda _model: tmp_path)
    monkeypatch.setattr(
        engine_module,
        "resolve_checkpoint",
        lambda *args, **kwargs: weight_resolutions.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="use_qk_norm_in_gdn=False"):
        LLMEngine(
            "unsupported/gdn-norm",
            engine_config=_small_engine_config("unsupported/gdn-norm"),
        )

    assert weight_resolutions == []


def test_hub_resolution_fetches_metadata_before_weights(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(weights_module, "_local_checkpoint", lambda _model: None)

    def download(model, *, cache_dir, allow_patterns, revision=None):
        calls.append((model, allow_patterns, revision))
        return tmp_path

    monkeypatch.setattr(weights_module, "_download_snapshot", download)

    metadata = weights_module.resolve_checkpoint_metadata("example/model")
    checkpoint = weights_module.resolve_checkpoint("example/model", revision="abc123")

    assert metadata == checkpoint == tmp_path
    assert "*.safetensors" not in calls[0][1]
    assert "*.safetensors" in calls[1][1]
    assert calls[1][2] == "abc123"
