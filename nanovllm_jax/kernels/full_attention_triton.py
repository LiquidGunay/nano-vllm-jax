"""Triton kernels for packed paged full-attention prefill."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


def _runtime_root() -> Path:
    configured = os.environ.get("NANO_VLLM_JAX_CACHE_ROOT")
    if configured:
        return Path(configured)
    mountpoint = Path("/mountpoint/.exp")
    if mountpoint.exists():
        return mountpoint
    mountpath = Path("/mountpath")
    if mountpath.exists():
        return mountpath
    return Path.cwd()


def _configure_triton_runtime() -> None:
    root = _runtime_root()
    triton_cache = root / ".cache" / "triton"
    tmp_dir = root / "tmp"
    triton_cache.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    os.environ.setdefault("TMPDIR", str(tmp_dir))

    triton_root = Path(triton.__file__).resolve().parent
    bundled_ptxas_dir = triton_root / "backends" / "nvidia" / "bin"
    bundled_ptxas = bundled_ptxas_dir / "ptxas"
    if bundled_ptxas.exists():
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        bundled = str(bundled_ptxas_dir)
        if not path_parts or path_parts[0] != bundled:
            os.environ["PATH"] = os.pathsep.join([bundled, *path_parts])


_configure_triton_runtime()


@triton.jit
def _packed_paged_prefill_attention_kernel(
    query,
    k_cache,
    v_cache,
    block_table,
    kv_lens,
    positions,
    query_start_loc,
    scale,
    out,
    token_bucket: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    row_query_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_row = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    row_start = tl.load(query_start_loc + pid_row).to(tl.int32)
    row_end = tl.load(query_start_loc + pid_row + 1).to(tl.int32)
    row_len = row_end - row_start
    token_idx = row_start + offs_m
    valid_m = (offs_m < row_len) & (offs_m < row_query_len) & (token_idx < token_bucket)
    token_pos = tl.load(positions + token_idx, mask=valid_m, other=0).to(tl.int32)
    scale_value = tl.load(scale).to(tl.float32)

    q = tl.load(
        query + (token_idx[:, None] * num_heads + pid_head) * head_dim + offs_d[None, :],
        mask=valid_m[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    kv_head = pid_head // num_key_value_groups
    kv_len = tl.load(kv_lens + pid_row).to(tl.int32)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, max_blocks_per_seq * block_size, BLOCK_N):
        kv_pos = start_n + offs_n
        page_offsets = kv_pos // block_size
        slot_offsets = kv_pos - page_offsets * block_size
        page_ids = tl.load(
            block_table + pid_row * max_blocks_per_seq + page_offsets,
            mask=kv_pos < max_blocks_per_seq * block_size,
            other=0,
        ).to(tl.int32)
        kv_base = (
            ((page_ids[:, None] * block_size + slot_offsets[:, None]) * num_kv_heads + kv_head)
            * head_dim
            + offs_d[None, :]
        )
        valid_n = kv_pos < kv_len
        k = tl.load(
            k_cache + kv_base,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_cache + kv_base,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k).to(q.dtype)) * scale_value
        causal = kv_pos[None, :] <= token_pos[:, None]
        scores = tl.where(valid_m[:, None] & valid_n[None, :] & causal, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new = tl.where(valid_m, m_new, 0.0)
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    acc = acc / tl.maximum(l_i[:, None], 1.0e-20)
    tl.store(
        out + (token_idx[:, None] * num_heads + pid_head) * head_dim + offs_d[None, :],
        acc,
        mask=valid_m[:, None] & (offs_d[None, :] < head_dim) & (l_i[:, None] > 0.0),
    )


def packed_paged_prefill_attention_triton(
    query: jax.Array,
    k_cache_layer: jax.Array,
    v_cache_layer: jax.Array,
    block_table: jax.Array,
    kv_lens: jax.Array,
    positions: jax.Array,
    query_start_loc: jax.Array,
    *,
    block_size: int,
    scale: float,
    num_key_value_groups: int,
    max_query_len: int,
) -> jax.Array:
    """Run packed paged prefill attention and return `[1, token_bucket, hidden]`."""

    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError("query must have shape [1, token_bucket, heads, head_dim]")
    if k_cache_layer.ndim != 4 or v_cache_layer.shape != k_cache_layer.shape:
        raise ValueError("k/v cache layers must have shape [pages, page_size, kv_heads, head_dim]")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [rows, max_blocks_per_seq]")

    _, token_bucket, num_heads, head_dim = query.shape
    row_count, max_blocks_per_seq = block_table.shape
    num_kv_heads = k_cache_layer.shape[2]
    if num_heads != num_kv_heads * num_key_value_groups:
        raise ValueError("num_key_value_groups must match query/KV head counts")
    if head_dim != k_cache_layer.shape[3]:
        raise ValueError("query/cache head dimensions must match")

    row_query_len = max(1, min(int(max_query_len), int(token_bucket)))
    block_m = 16
    block_n = 64
    block_d = max(16, int(jt.next_power_of_2(int(head_dim))))
    if block_d > 256:
        raise ValueError("packed prefill Triton attention supports head_dim <= 256")

    query_for_kernel = (
        query.astype(k_cache_layer.dtype)
        if k_cache_layer.dtype in (jnp.bfloat16, jnp.float16)
        else query
    )
    out_shape = jax.ShapeDtypeStruct((token_bucket, num_heads, head_dim), jnp.float32)
    out = jt.triton_call(
        query_for_kernel.reshape(token_bucket, num_heads, head_dim),
        k_cache_layer,
        v_cache_layer,
        block_table.astype(jnp.int32),
        kv_lens.astype(jnp.int32),
        positions.reshape(token_bucket).astype(jnp.int32),
        query_start_loc.astype(jnp.int32),
        jnp.asarray(scale, dtype=jnp.float32),
        kernel=_packed_paged_prefill_attention_kernel,
        out_shape=out_shape,
        grid=(jt.cdiv(row_query_len, block_m), int(num_heads), int(row_count)),
        token_bucket=int(token_bucket),
        max_blocks_per_seq=int(max_blocks_per_seq),
        block_size=int(block_size),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        num_key_value_groups=int(num_key_value_groups),
        row_query_len=int(row_query_len),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    actual_tokens = query_start_loc[-1].astype(jnp.int32)
    valid = jnp.arange(token_bucket, dtype=jnp.int32) < actual_tokens
    out = jnp.where(valid[:, None, None], out, 0.0)
    return out.reshape(1, token_bucket, num_heads * head_dim)


@triton.jit
def _paged_decode_attention_kernel(
    query,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    scale,
    out,
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_key_value_groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    kv_head = pid_head // num_key_value_groups
    seq_len = tl.load(seq_lens + pid_row).to(tl.int32)
    scale_value = tl.load(scale).to(tl.float32)

    q = tl.load(
        query + (pid_row * num_heads + pid_head) * head_dim + offs_d,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for start_n in range(0, max_blocks_per_seq * block_size, BLOCK_N):
        kv_pos = start_n + offs_n
        page_offsets = kv_pos // block_size
        slot_offsets = kv_pos - page_offsets * block_size
        valid_n = kv_pos < seq_len
        page_ids = tl.load(
            block_table + pid_row * max_blocks_per_seq + page_offsets,
            mask=kv_pos < max_blocks_per_seq * block_size,
            other=0,
        ).to(tl.int32)
        kv_base = (
            ((page_ids[:, None] * block_size + slot_offsets[:, None]) * num_kv_heads + kv_head)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            k_cache + kv_base,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_cache + kv_base,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale_value
        scores = tl.where(valid_n, scores, -float("inf"))
        block_has_valid = tl.max(valid_n.to(tl.int32), axis=0) > 0
        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        m_for_exp = tl.where(block_has_valid | (l_i > 0.0), m_new, 0.0)
        p = tl.where(valid_n, tl.exp(scores - m_for_exp), 0.0)
        alpha = tl.where(l_i > 0.0, tl.exp(m_i - m_for_exp), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = tl.where(l_new > 0.0, m_for_exp, m_i)
        l_i = l_new

    out_value = acc / tl.maximum(l_i, 1.0e-20)
    tl.store(
        out + (pid_row * num_heads + pid_head) * head_dim + offs_d,
        out_value,
        mask=(offs_d < head_dim) & (l_i > 0.0),
    )


def paged_decode_attention_triton(
    query: jax.Array,
    k_cache_layer: jax.Array,
    v_cache_layer: jax.Array,
    block_table: jax.Array,
    seq_lens: jax.Array,
    *,
    block_size: int,
    scale: float,
    num_key_value_groups: int,
) -> jax.Array:
    """Run width-1 paged decode attention and return `[batch, 1, hidden]`."""

    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("decode query must have shape [batch, 1, heads, head_dim]")
    if k_cache_layer.ndim != 4 or v_cache_layer.shape != k_cache_layer.shape:
        raise ValueError("k/v cache layers must have shape [pages, page_size, kv_heads, head_dim]")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_blocks_per_seq]")

    batch, _, num_heads, head_dim = query.shape
    row_count, max_blocks_per_seq = block_table.shape
    if row_count != batch:
        raise ValueError("query and block_table batch sizes must match")
    num_kv_heads = k_cache_layer.shape[2]
    if num_heads != num_kv_heads * num_key_value_groups:
        raise ValueError("num_key_value_groups must match query/KV head counts")
    if head_dim != k_cache_layer.shape[3]:
        raise ValueError("query/cache head dimensions must match")

    block_n = 64
    block_d = max(16, int(jt.next_power_of_2(int(head_dim))))
    if block_d > 256:
        raise ValueError("decode Triton attention supports head_dim <= 256")

    query_for_kernel = (
        query.astype(k_cache_layer.dtype)
        if k_cache_layer.dtype in (jnp.bfloat16, jnp.float16)
        else query
    )
    out_shape = jax.ShapeDtypeStruct((batch, num_heads, head_dim), jnp.float32)
    out = jt.triton_call(
        query_for_kernel.reshape(batch, num_heads, head_dim),
        k_cache_layer,
        v_cache_layer,
        block_table.astype(jnp.int32),
        seq_lens.astype(jnp.int32),
        jnp.asarray(scale, dtype=jnp.float32),
        kernel=_paged_decode_attention_kernel,
        out_shape=out_shape,
        grid=(int(batch), int(num_heads)),
        max_blocks_per_seq=int(max_blocks_per_seq),
        block_size=int(block_size),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        num_key_value_groups=int(num_key_value_groups),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out.reshape(batch, 1, num_heads * head_dim)
