"""Qwen3.5 serving engine in JAX."""

from nanovllm_jax.config import EngineConfig
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.speculation import DrafterConfig

__all__ = [
    "DrafterConfig",
    "EngineConfig",
    "LLM",
    "SamplingParams",
]


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm_jax.engine import LLM

        return LLM
    raise AttributeError(name)
