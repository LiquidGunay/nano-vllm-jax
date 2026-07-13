from dataclasses import replace

from nanovllm_jax.config import (
    CapacitySpec,
    CompileSpec,
    ModelSpec,
    RuntimeSpec,
)
from nanovllm_jax.fastpath import KernelPlan


def runtime_spec(*, model=None, capacity=None, compile=None, kernels=None) -> RuntimeSpec:
    """Build an explicitly owned test runtime without production adapters."""

    return RuntimeSpec(
        model=replace(ModelSpec(), **(model or {})),
        capacity=replace(CapacitySpec(), **(capacity or {})),
        compile=replace(CompileSpec(), **(compile or {})),
        kernels=replace(KernelPlan(), **(kernels or {})),
    )
