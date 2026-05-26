"""CUDA Gated DeltaNet kernel placeholders."""

from __future__ import annotations

from typing import Any

from nanovllm_jax.kernels.registry import KernelBackendUnavailable, backend_status


def availability():
    return backend_status("gdn_cuda")


def require_available() -> None:
    status = availability()
    if not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)


def gdn_recurrent_decode_step(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("gdn_recurrent_decode_step CUDA wrapper is not implemented yet")


def gdn_segmented_prefill_chunk32(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("gdn_segmented_prefill_chunk32 CUDA wrapper is not implemented yet")
