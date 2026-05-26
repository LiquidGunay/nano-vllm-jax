"""FlashInfer/JAX FFI prototypes.

FlashInfer integration is optional and not yet wired into serving. Prototype
kernels in this module can be exercised directly by focused tests, while the
runtime backend registry keeps the pure-JAX path as the default until integrated
correctness and performance gates pass.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from nanovllm_jax.kernels.registry import KernelBackendUnavailable, backend_status

_APPEND_PAGED_KV_CACHE_TARGET = "nanovllm_jax_flashinfer_append_paged_kv_cache"
_NHD_LAYOUT = 0
_REGISTER_LOCK = threading.Lock()
_APPEND_PAGED_KV_CACHE_REGISTERED = False


def availability():
    return backend_status("flashinfer")


def require_available() -> None:
    status = availability()
    if not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)


def _default_runtime_root() -> Path:
    configured = os.getenv("NANO_VLLM_JAX_CACHE_ROOT")
    if configured:
        return Path(configured)
    mountpoint = Path("/mountpoint/.exp")
    if mountpoint.exists():
        return mountpoint
    mountpath = Path("/mountpath")
    if mountpath.exists():
        return mountpath
    return Path.cwd()


def _configure_flashinfer_cache() -> None:
    root = _default_runtime_root()
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(root))
    os.environ.setdefault(
        "FLASHINFER_CUBIN_DIR",
        str(root / ".cache" / "flashinfer" / "cubins"),
    )
    Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FLASHINFER_CUBIN_DIR"]).mkdir(parents=True, exist_ok=True)


def _require_flashinfer_modules() -> None:
    missing = [
        module
        for module in ("flashinfer", "jax_tvm_ffi")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise KernelBackendUnavailable(
            "FlashInfer/JAX FFI dependencies are unavailable; missing optional modules: "
            + ", ".join(missing)
        )


def _register_append_paged_kv_cache() -> None:
    global _APPEND_PAGED_KV_CACHE_REGISTERED
    if _APPEND_PAGED_KV_CACHE_REGISTERED:
        return

    with _REGISTER_LOCK:
        if _APPEND_PAGED_KV_CACHE_REGISTERED:
            return
        _configure_flashinfer_cache()
        from flashinfer.jit.page import gen_page_module
        from jax_tvm_ffi import register_ffi_target

        module = gen_page_module().build_and_load()
        register_ffi_target(
            _APPEND_PAGED_KV_CACHE_TARGET,
            module.append_paged_kv_cache,
            arg_spec=["args", "attrs.layout"],
            platform="gpu",
            allow_cuda_graph=True,
        )
        _APPEND_PAGED_KV_CACHE_REGISTERED = True


def _as_jax_array(name: str, value: Any):
    try:
        return jnp.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive type error path.
        raise TypeError(f"{name} must be array-like") from exc


def _validate_kv_append_inputs(
    append_key,
    append_value,
    batch_indices,
    positions,
    k_cache,
    v_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
) -> None:
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if append_key.shape != append_value.shape:
        raise ValueError("append_key and append_value must have the same shape")
    if append_key.ndim != 3:
        raise ValueError("append_key must have shape [nnz_tokens, num_kv_heads, head_dim]")
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have NHD shape [num_pages, page_size, num_kv_heads, head_dim]")
    if append_key.shape[1:] != k_cache.shape[2:]:
        raise ValueError("append_key trailing dimensions must match cache [num_kv_heads, head_dim]")

    nnz_tokens = append_key.shape[0]
    if batch_indices.shape != (nnz_tokens,) or positions.shape != (nnz_tokens,):
        raise ValueError("batch_indices and positions must both have shape [nnz_tokens]")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must have shape [total_pages]")
    if kv_indptr.ndim != 1:
        raise ValueError("kv_indptr must have shape [batch + 1]")
    if kv_last_page_len.ndim != 1:
        raise ValueError("kv_last_page_len must have shape [batch]")

    for name, value in (
        ("batch_indices", batch_indices),
        ("positions", positions),
        ("kv_indices", kv_indices),
        ("kv_indptr", kv_indptr),
        ("kv_last_page_len", kv_last_page_len),
    ):
        if value.dtype != jnp.int32:
            raise ValueError(f"{name} must have dtype int32 for the FlashInfer ABI")


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


def kv_append_paged_nhd(
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
    """Append K/V into an NHD paged cache using FlashInfer's JAX FFI path.

    This is a focused prototype for the planned ABI. It aliases the cache inputs
    to the cache outputs so unwritten cache entries are preserved, matching
    FlashInfer's mutating CUDA contract while still returning functional JAX
    values.
    """

    _require_flashinfer_modules()
    _register_append_paged_kv_cache()

    append_key = _as_jax_array("append_key", append_key)
    append_value = _as_jax_array("append_value", append_value)
    batch_indices = _as_jax_array("batch_indices", batch_indices)
    positions = _as_jax_array("positions", positions)
    k_cache = _as_jax_array("k_cache", k_cache)
    v_cache = _as_jax_array("v_cache", v_cache)
    kv_indices = _as_jax_array("kv_indices", kv_indices)
    kv_indptr = _as_jax_array("kv_indptr", kv_indptr)
    kv_last_page_len = _as_jax_array("kv_last_page_len", kv_last_page_len)
    _validate_kv_append_inputs(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )

    call = jax.ffi.ffi_call(
        _APPEND_PAGED_KV_CACHE_TARGET,
        (
            jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
            jax.ShapeDtypeStruct(v_cache.shape, v_cache.dtype),
        ),
        has_side_effect=True,
        input_output_aliases={4: 0, 5: 1},
    )
    return call(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout=_NHD_LAYOUT,
    )


def _static_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must be a static Python integer for FlashInfer append routing"
        ) from exc


def _kv_last_page_len(seq_lens: jnp.ndarray, page_size: int) -> jnp.ndarray:
    seq_lens = seq_lens.astype(jnp.int32)
    return jnp.where(
        seq_lens > 0,
        ((seq_lens - 1) % jnp.asarray(page_size, dtype=jnp.int32)) + 1,
        0,
    ).astype(jnp.int32)


def kv_append_paged_nhd_from_metadata(
    k: jnp.ndarray,
    v: jnp.ndarray,
    k_cache_layer: jnp.ndarray,
    v_cache_layer: jnp.ndarray,
    metadata: Any,
    *,
    page_size: int,
):
    """Append rectangular scheduled K/V tensors using ragged metadata.

    `k` and `v` use the model/backend shape `[batch, query_len, num_kv_heads,
    head_dim]`. Padded query slots are compacted away using
    `metadata.query_start_loc` and the static token counters from the scheduled
    batch.
    """

    if metadata.positions is None:
        raise ValueError("metadata.positions is required for FlashInfer KV append")

    num_tokens = _static_int(
        metadata.num_prefill_tokens,
        "metadata.num_prefill_tokens",
    ) + _static_int(
        metadata.num_decode_tokens,
        "metadata.num_decode_tokens",
    )
    if num_tokens == 0:
        return k_cache_layer, v_cache_layer

    batch, query_len = k.shape[:2]
    query_lens = jnp.diff(metadata.query_start_loc).astype(jnp.int32)
    valid_mask = jnp.arange(query_len, dtype=jnp.int32)[None, :] < query_lens[:, None]
    row_idx, col_idx = jnp.nonzero(valid_mask, size=num_tokens)

    append_key = k[row_idx, col_idx]
    append_value = v[row_idx, col_idx]
    batch_indices = row_idx.astype(jnp.int32)
    positions = metadata.positions[row_idx, col_idx].astype(jnp.int32)
    max_pages_per_sequence = metadata.block_tables.shape[1]
    kv_indices = metadata.block_tables.reshape(-1).astype(jnp.int32)
    kv_indptr = (
        jnp.arange(batch + 1, dtype=jnp.int32)
        * jnp.asarray(max_pages_per_sequence, dtype=jnp.int32)
    )
    kv_last_page_len = _kv_last_page_len(metadata.seq_lens, page_size)
    return kv_append_paged_nhd(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache_layer,
        v_cache_layer,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


def paged_decode_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_decode_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")


def paged_prefill_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_prefill_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")
