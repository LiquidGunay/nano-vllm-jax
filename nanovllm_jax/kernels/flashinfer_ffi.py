"""FlashInfer/JAX FFI placeholders.

FlashInfer integration is optional and not yet wired into serving. These stubs
make the ABI boundary explicit while preventing accidental fallback-free use.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nanovllm_jax.kernels.registry import KernelBackendUnavailable, backend_status


def availability():
    return backend_status("flashinfer")


def require_available() -> None:
    status = availability()
    if not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)


def kv_append_paged_nhd_reference(
    append_key,
    append_value,
    batch_indices,
    positions,
    k_cache,
    v_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
):
    """Pure-JAX reference for the planned FlashInfer NHD append ABI.

    This mirrors FlashInfer's NHD contract:
    - append key/value: [nnz_tokens, num_kv_heads, head_dim]
    - cache: [num_pages, page_size, num_kv_heads, head_dim]
    - page table: `kv_indices[kv_indptr[b] + positions[i] // page_size]`
    """

    del kv_last_page_len  # Shape/ABI placeholder; bounds are enforced by metadata tests.
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if append_key.shape != append_value.shape:
        raise ValueError("append_key and append_value must have the same shape")
    if append_key.ndim != 3:
        raise ValueError("append_key must have shape [nnz_tokens, num_kv_heads, head_dim]")
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have NHD shape [num_pages, page_size, num_kv_heads, head_dim]")
    nnz_tokens = append_key.shape[0]
    if batch_indices.shape != (nnz_tokens,) or positions.shape != (nnz_tokens,):
        raise ValueError("batch_indices and positions must both have shape [nnz_tokens]")

    page_size = k_cache.shape[1]
    page_offsets = positions.astype(jnp.int32) // page_size
    slots = positions.astype(jnp.int32) % page_size
    page_table_offsets = kv_indptr[batch_indices.astype(jnp.int32)] + page_offsets
    physical_pages = kv_indices[page_table_offsets]
    k_cache = k_cache.at[physical_pages, slots].set(append_key)
    v_cache = v_cache.at[physical_pages, slots].set(append_value)
    return k_cache, v_cache


def kv_append_paged_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("kv_append_paged_nhd FlashInfer FFI wrapper is not implemented yet")


def paged_decode_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_decode_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")


def paged_prefill_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_prefill_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")
