"""Engine module for Qwen 3.5 JAX inference."""

from nanovllm_jax.engine.sequence import Sequence, SequenceStatus, SamplingParams

__all__ = [
    "Sequence",
    "SequenceStatus",
    "SamplingParams",
    "ScheduledBatch",
    "BlockManager",
    "Scheduler",
    "ModelExecutor",
    "ModelRunner",
    "LLMEngine",
]


def __getattr__(name):
    """Lazy-load engine components with optional dependencies."""
    if name == "BlockManager":
        from nanovllm_jax.engine.block_manager import BlockManager

        return BlockManager
    if name == "Scheduler":
        from nanovllm_jax.engine.scheduler import Scheduler

        return Scheduler
    if name == "ScheduledBatch":
        from nanovllm_jax.engine.scheduled_batch import ScheduledBatch

        return ScheduledBatch
    if name == "ModelExecutor":
        from nanovllm_jax.engine.model_executor import ModelExecutor

        return ModelExecutor
    if name == "ModelRunner":
        from nanovllm_jax.engine.model_runner import ModelRunner

        return ModelRunner
    if name == "LLMEngine":
        from nanovllm_jax.engine.llm_engine import LLMEngine

        return LLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
