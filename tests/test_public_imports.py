import pytest

from nanovllm_jax import EngineConfig, LLM, SamplingParams
from nanovllm_jax.batch import ScheduledBatch
from nanovllm_jax.block_manager import BlockManager
from nanovllm_jax.executor import ModelExecutor
import nanovllm_jax.engine as engine_module
from nanovllm_jax.engine import LLM as PublicLLM
from nanovllm_jax.engine import LLMEngine as PublicLLMEngine
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams as PublicSamplingParams
from nanovllm_jax.sequence import Sequence, SequenceStatus


def test_top_level_public_imports_are_canonical():
    assert LLM is PublicLLM
    assert PublicLLM is PublicLLMEngine
    assert SamplingParams is PublicSamplingParams
    assert EngineConfig.__name__ == "EngineConfig"
    import nanovllm_jax

    assert nanovllm_jax.__all__ == ["EngineConfig", "LLM", "SamplingParams"]
    assert engine_module.__all__ == ["LLM"]
    assert not hasattr(nanovllm_jax, "FASTPATH")
    assert not hasattr(nanovllm_jax, "FastPath")
    assert not hasattr(nanovllm_jax, "LLMEngine")
    assert not hasattr(nanovllm_jax, "ModelConfig")
    assert not hasattr(nanovllm_jax, "Qwen3_5Config")
    assert not hasattr(nanovllm_jax, "ServerSettings")
    assert not hasattr(nanovllm_jax, "ServingOps")
    assert not hasattr(nanovllm_jax, "select_backend")


def test_llm_constructor_rejects_internal_policy_kwargs_before_loading_model():
    with pytest.raises(TypeError, match="workload/capacity"):
        PublicLLMEngine("Qwen/Qwen3.5-0.8B", gdn_prefill_post_conv_impl="reference")

    with pytest.raises(TypeError, match="weight_dtype"):
        PublicLLMEngine("Qwen/Qwen3.5-0.8B", weight_dtype="float32")


def test_internal_imports_use_top_level_modules():
    assert BlockManager.__module__ == "nanovllm_jax.block_manager"
    assert ModelExecutor.__module__ == "nanovllm_jax.executor"
    assert ModelRunner.__module__ == "nanovllm_jax.runner"
    assert ScheduledBatch.__module__ == "nanovllm_jax.batch"
    assert Scheduler.__module__ == "nanovllm_jax.scheduler"
    assert Sequence.__module__ == "nanovllm_jax.sequence"
    assert SequenceStatus.__module__ == "nanovllm_jax.sequence"


def test_old_engine_package_is_not_part_of_clean_main():
    with pytest.raises(ModuleNotFoundError):
        __import__("nanovllm_jax.engine.llm_engine")
