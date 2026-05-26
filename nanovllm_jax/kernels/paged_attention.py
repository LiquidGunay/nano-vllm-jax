"""Reference ABIs for paged attention kernels.

These helpers are not optimized serving kernels. They encode the JAX-facing
contracts that future CUDA/FFI implementations must match while preserving the
pure-JAX fallback path.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _as_jax_array(name: str, value: Any) -> jnp.ndarray:
    try:
        return jnp.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive type error path.
        raise TypeError(f"{name} must be array-like") from exc


def kv_last_page_len_from_seq_lens(
    seq_lens: Any,
    page_size: int,
) -> jnp.ndarray:
    """Return FlashInfer/vLLM-style last-page lengths for each sequence."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    seq_lens = _as_jax_array("seq_lens", seq_lens).astype(jnp.int32)
    page_size_array = jnp.asarray(page_size, dtype=jnp.int32)
    return jnp.where(
        seq_lens > 0,
        ((seq_lens - 1) % page_size_array) + 1,
        0,
    ).astype(jnp.int32)


def dense_block_tables_to_kv_indptr(
    block_tables: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert dense `[batch, max_pages]` block tables to flattened page metadata."""

    block_tables = _as_jax_array("block_tables", block_tables)
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [batch, max_pages_per_sequence]")
    batch, max_pages_per_sequence = block_tables.shape
    kv_indices = block_tables.reshape(-1).astype(jnp.int32)
    kv_indptr = (
        jnp.arange(batch + 1, dtype=jnp.int32)
        * jnp.asarray(max_pages_per_sequence, dtype=jnp.int32)
    )
    return kv_indices, kv_indptr


def _validate_paged_decode_inputs(
    q: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_indices: jnp.ndarray,
    kv_last_page_len: jnp.ndarray,
    seq_lens: jnp.ndarray,
    max_pages_per_sequence: int,
) -> None:
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_q_heads, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if k_cache.ndim != 4:
        raise ValueError(
            "k_cache must have NHD shape "
            "[num_pages, page_size, num_kv_heads, head_dim]"
        )
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("q and cache head_dim must match")
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    batch = q.shape[0]
    if kv_indptr.shape != (batch + 1,):
        raise ValueError("kv_indptr must have shape [batch + 1]")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must have shape [total_pages]")
    if kv_last_page_len.shape != (batch,):
        raise ValueError("kv_last_page_len must have shape [batch]")
    if seq_lens.shape != (batch,):
        raise ValueError("seq_lens must have shape [batch]")
    if max_pages_per_sequence <= 0:
        raise ValueError("max_pages_per_sequence must be positive")
    for name, value in (
        ("kv_indptr", kv_indptr),
        ("kv_indices", kv_indices),
        ("kv_last_page_len", kv_last_page_len),
        ("seq_lens", seq_lens),
    ):
        if value.dtype != jnp.int32:
            raise ValueError(f"{name} must have dtype int32")


def paged_decode_attention_gqa_nhd_reference(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    kv_indptr: Any,
    kv_indices: Any,
    kv_last_page_len: Any,
    seq_lens: Any,
    softmax_scale: float,
    *,
    max_pages_per_sequence: int | None = None,
) -> jnp.ndarray:
    """Pure-JAX reference for the planned NHD paged decode attention ABI.

    Args:
        q: Decode query `[batch, num_q_heads, head_dim]`.
        k_cache: NHD key cache `[num_pages, page_size, num_kv_heads, head_dim]`.
        v_cache: NHD value cache with the same shape as `k_cache`.
        kv_indptr: CSR-style page-table offsets `[batch + 1]`.
        kv_indices: Flattened physical page ids `[total_pages]`.
        kv_last_page_len: Valid token count in each sequence's final page.
        seq_lens: Logical sequence lengths `[batch]`.
        softmax_scale: Attention scale.
        max_pages_per_sequence: Static gathered page span. When omitted, this
            assumes dense per-sequence page tables and uses
            `len(kv_indices) // batch`, matching the repo's current block-table
            metadata shape.

    Returns:
        Attention output `[batch, num_q_heads, head_dim]`.
    """

    q = _as_jax_array("q", q)
    k_cache = _as_jax_array("k_cache", k_cache)
    v_cache = _as_jax_array("v_cache", v_cache)
    kv_indptr = _as_jax_array("kv_indptr", kv_indptr)
    kv_indices = _as_jax_array("kv_indices", kv_indices)
    kv_last_page_len = _as_jax_array("kv_last_page_len", kv_last_page_len)
    seq_lens = _as_jax_array("seq_lens", seq_lens)

    batch, num_q_heads, head_dim = q.shape
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    if max_pages_per_sequence is None:
        max_pages_per_sequence = kv_indices.shape[0] // batch
    max_pages_per_sequence = int(max_pages_per_sequence)
    _validate_paged_decode_inputs(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
        max_pages_per_sequence,
    )

    max_kv_len = max_pages_per_sequence * page_size
    token_positions = jnp.arange(max_kv_len, dtype=jnp.int32)
    page_offsets = token_positions // jnp.asarray(page_size, dtype=jnp.int32)
    slot_offsets = token_positions % jnp.asarray(page_size, dtype=jnp.int32)

    page_counts = kv_indptr[1:] - kv_indptr[:-1]
    safe_page_counts = jnp.maximum(page_counts, 1)
    safe_page_offsets = jnp.minimum(
        page_offsets[None, :],
        safe_page_counts[:, None] - 1,
    )
    page_table_offsets = kv_indptr[:-1, None] + safe_page_offsets
    physical_pages = kv_indices[page_table_offsets]

    k_gathered = k_cache[physical_pages, slot_offsets[None, :]]
    v_gathered = v_cache[physical_pages, slot_offsets[None, :]]

    effective_lens = jnp.where(
        page_counts > 0,
        (page_counts - 1) * jnp.asarray(page_size, dtype=jnp.int32)
        + kv_last_page_len,
        0,
    )
    effective_lens = jnp.minimum(effective_lens, seq_lens.astype(jnp.int32))
    valid_tokens = token_positions[None, :] < effective_lens[:, None]

    num_key_value_groups = num_q_heads // num_kv_heads
    q_grouped = q.reshape(batch, num_kv_heads, num_key_value_groups, head_dim)
    k_grouped = k_gathered[:, :, :, None, :]
    v_grouped = v_gathered[:, :, :, None, :]

    attn_scores = jnp.einsum("bkgd,bskgd->bkgs", q_grouped, k_grouped)
    attn_scores = attn_scores * softmax_scale
    attn_scores = jnp.where(valid_tokens[:, None, None, :], attn_scores, -1e10)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum("bkgs,bskgd->bkgd", attn_weights, v_grouped)
    return out.reshape(batch, num_q_heads, head_dim)
