"""FlashInfer/JAX FFI placeholders.

FlashInfer integration is optional and not yet wired into serving. These stubs
make the ABI boundary explicit while preventing accidental fallback-free use.
"""

from __future__ import annotations

from typing import Any

from nanovllm_jax.kernels.registry import KernelBackendUnavailable, backend_status


def availability():
    return backend_status("flashinfer")


def require_available() -> None:
    status = availability()
    if not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)


def kv_append_paged_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("kv_append_paged_nhd FlashInfer FFI wrapper is not implemented yet")


def paged_decode_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_decode_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")


def paged_prefill_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_prefill_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")
