"""JAX/Triton Gated DeltaNet kernels shaped after vLLM/FLA.

These kernels are optional and are intentionally kept behind explicit backend
flags. The correctness reference remains in :mod:`nanovllm_jax.kernels.gdn_fla`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np

import jax
import jax.numpy as jnp
from jax import core
import jax_triton as jt
import triton
import triton.language as tl


def _normalize_int_env(name: str, default: int, *, min_value: int, max_value: int) -> int:
    """Read an integer env override with bounds; return ``default`` when unset."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    if not (min_value <= parsed <= max_value):
        raise ValueError(f"{name} must be between {min_value} and {max_value}; got {parsed}")
    return parsed


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


def _decode_triton_num_warps(value_dim: int) -> int:
    env_name = "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_WARPS"
    default = 1 if value_dim <= 128 else 2
    return _normalize_int_env(env_name, default, min_value=1, max_value=8)


def _decode_triton_num_stages() -> int:
    env_name = "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_STAGES"
    return _normalize_int_env(env_name, 3, min_value=1, max_value=8)


def _decode_triton_block_v(value_dim: int) -> int:
    env_name = "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_BLOCK_V"
    block_v = min(jt.next_power_of_2(value_dim), 32)
    return _normalize_int_env(env_name, block_v, min_value=1, max_value=128)


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
def _gdn_fla_chunk_local_cumsum_packed_kernel(
    gate,
    cu_seqlens,
    chunk_indices,
    out,
    num_heads: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    head_mask = head < num_heads
    chunk_row = tl.load(chunk_indices + pid_chunk * 2 + 0).to(tl.int32)
    chunk_id = tl.load(chunk_indices + pid_chunk * 2 + 1).to(tl.int32)

    row_start = tl.load(cu_seqlens + chunk_row)
    row_end = tl.load(cu_seqlens + chunk_row + 1)
    chunk_start = row_start + chunk_id * chunk_size
    chunk_end = tl.minimum(row_end, row_start + (chunk_id + 1) * chunk_size)
    chunk_len = chunk_end - chunk_start

    # Forward:
    #   input idx : chunk_start + offs
    # Reverse:
    #   input idx : chunk_start + chunk_len - 1 - offs
    #            write at original positions -> store idx same as input idx for reverse output.
    input_offsets = tl.where(
        offs < chunk_len,
        tl.where(REVERSE, chunk_len - 1 - offs, offs),
        0,
    )
    input_offsets = tl.where(input_offsets >= 0, input_offsets, 0)
    input_positions = chunk_start + input_offsets
    chunk_mask = offs < chunk_len
    store_positions = (
        chunk_start + tl.where(REVERSE, chunk_len - 1 - offs, offs)
    )
    gate_offsets = input_positions * num_heads + head
    gate_vals = tl.load(
        gate + gate_offsets,
        mask=chunk_mask & head_mask,
        other=0.0,
    ).to(tl.float32)
    gate_scan = tl.zeros((BLOCK,), dtype=tl.float32)
    running = tl.zeros((), dtype=tl.float32)
    for i in range(BLOCK):
        elem = tl.sum(
            tl.where(
                offs == i,
                gate_vals,
                tl.zeros((BLOCK,), dtype=tl.float32),
            ),
            axis=0,
        )
        running += elem
        gate_scan = tl.where(offs == i, running, gate_scan)
    output_offsets = store_positions * num_heads + head
    tl.store(out + output_offsets, gate_scan, mask=chunk_mask & head_mask)


@triton.jit
def _gdn_fla_chunk_scaled_dot_kkt_packed_kernel(
    key,
    beta,
    gate_cumsum,
    cu_seqlens,
    chunk_indices,
    out,
    num_key_heads: tl.constexpr,
    num_output_heads: tl.constexpr,
    key_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_GATE: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_row = tl.program_id(2)

    head_mask = pid_head < num_output_heads
    col_offsets = tl.arange(0, BLOCK_S)
    active_row = pid_row < chunk_size
    use_row = active_row & head_mask

    chunk_row = tl.load(chunk_indices + pid_chunk * 2 + 0).to(tl.int32)
    chunk_id = tl.load(chunk_indices + pid_chunk * 2 + 1).to(tl.int32)

    row_start = tl.load(cu_seqlens + chunk_row)
    row_end = tl.load(cu_seqlens + chunk_row + 1)
    chunk_start = row_start + chunk_id * chunk_size
    chunk_end = tl.minimum(row_end, row_start + (chunk_id + 1) * chunk_size)
    chunk_len = chunk_end - chunk_start

    row_in_chunk = pid_row
    row_in_bounds = row_in_chunk < chunk_len
    row_global = chunk_start + row_in_chunk
    col_in_bounds = col_offsets < chunk_len
    local_mask = col_offsets < row_in_chunk
    save_mask = use_row & row_in_bounds & col_in_bounds

    if not USE_GATE:
        gate_row = tl.zeros((), dtype=tl.float32)
        gate_col = tl.zeros((BLOCK_S,), dtype=tl.float32)
    else:
        gate_row = tl.load(
            gate_cumsum + row_global * num_output_heads + pid_head,
            mask=use_row & row_in_bounds,
            other=0.0,
        ).to(tl.float32)
        gate_col = tl.load(
            gate_cumsum + (chunk_start + col_offsets) * num_output_heads + pid_head,
            mask=save_mask,
            other=0.0,
        ).to(tl.float32)

    head_group = num_output_heads // num_key_heads
    key_head = pid_head // head_group
    beta_row = tl.load(
        beta + row_global * num_output_heads + pid_head,
        mask=use_row & row_in_bounds,
        other=0.0,
    ).to(tl.float32)
    row_k_base_ptr = key + row_global * num_key_heads * key_dim + key_head * key_dim

    scores = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for k in range(BLOCK_K):
        row_k_scalar = tl.load(
            row_k_base_ptr + k,
            mask=use_row & row_in_bounds & (k < key_dim),
            other=0.0,
        ).to(tl.float32)
        col_k = tl.load(
            key + (chunk_start + col_offsets) * num_key_heads * key_dim + key_head * key_dim + k,
            mask=col_in_bounds & (k < key_dim),
            other=0.0,
        ).to(tl.float32)
        scores += beta_row * row_k_scalar * col_k

    if USE_GATE:
        scores *= tl.exp(gate_row - gate_col)

    scores = scores * tl.where(local_mask, 1.0, 0.0)
    scores = tl.where(col_offsets < chunk_len, scores, 0.0)
    out_offsets = (row_global * num_output_heads + pid_head) * chunk_size + col_offsets
    tl.store(
        out + out_offsets,
        scores,
        mask=use_row & row_in_bounds & (col_offsets < chunk_size),
    )


@triton.jit
def _gdn_fla_solve_tril_packed_kernel(
    attention_matrix,
    cu_seqlens,
    chunk_indices,
    out,
    num_heads: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)

    head_mask = pid_head < num_heads
    offs_col = tl.arange(0, BLOCK)

    chunk_row = tl.load(chunk_indices + pid_chunk * 2 + 0).to(tl.int32)
    chunk_id = tl.load(chunk_indices + pid_chunk * 2 + 1).to(tl.int32)

    row_start = tl.load(cu_seqlens + chunk_row)
    row_end = tl.load(cu_seqlens + chunk_row + 1)
    chunk_start = row_start + chunk_id * chunk_size
    chunk_end = tl.minimum(row_end, row_start + (chunk_id + 1) * chunk_size)
    chunk_len = chunk_end - chunk_start
    row_stride = num_heads * chunk_size
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    inv = tl.where(rows == cols, 1.0, 0.0).to(tl.float32)
    inv = tl.where(
        (rows < chunk_len) & (cols < chunk_len),
        inv,
        0.0,
    )

    for i in range(BLOCK):
        row_mask = i < chunk_len
        running_row = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(BLOCK):
            att_mask = head_mask & row_mask & (k < i) & (k < chunk_len) & (k < chunk_size)
            a_k = tl.load(
                attention_matrix
                + ((chunk_start + i) * num_heads + pid_head) * chunk_size
                + k,
                mask=att_mask,
                other=0.0,
            ).to(tl.float32)
            row_k = tl.sum(
                tl.where(rows == k, inv, 0.0),
                axis=0,
            ).to(tl.float32)
            running_row += a_k * row_k

        row_inv = tl.where(offs_col < i, -running_row, 0.0)
        row_inv = tl.where(offs_col == i, 1.0, row_inv)
        row_inv = tl.where(offs_col < chunk_len, row_inv, 0.0)
        row_inv = tl.where(head_mask & row_mask, row_inv, 0.0)
        inv = tl.where(rows == i, row_inv[None, :], inv)

    for i in range(BLOCK):
        row_to_store = tl.sum(tl.where(rows == i, inv, 0.0), axis=0)
        row_to_store = tl.where(offs_col < chunk_len, row_to_store, 0.0)
        row_to_store = tl.where(i < chunk_len, row_to_store, 0.0)
        row_to_store = tl.where(head_mask, row_to_store, 0.0)
        tl.store(
            out + ((chunk_start + i) * num_heads + pid_head) * chunk_size + offs_col,
            row_to_store,
            mask=(i < chunk_len) & head_mask & (offs_col < chunk_size),
        )


@triton.jit
def _gdn_fla_chunk_delta_h_packed_kernel(
    key,
    w,
    u,
    gate_cumsum,
    initial_state,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    out_h,
    out_v_new,
    out_final_state,
    num_key_heads: tl.constexpr,
    num_output_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    USE_GATE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_head = tl.program_id(1)

    head_mask = pid_head < num_output_heads
    val_offsets = tl.arange(0, BLOCK_V)
    key_offsets = tl.arange(0, BLOCK_K)
    val_mask = val_offsets < value_dim
    key_mask = key_offsets < key_dim

    row_start = tl.load(cu_seqlens + pid_row).to(tl.int32)
    row_end = tl.load(cu_seqlens + pid_row + 1).to(tl.int32)
    row_len = row_end - row_start
    row_chunk_count = (row_len + chunk_size - 1) // chunk_size
    row_chunk_base = tl.load(chunk_offsets + pid_row).to(tl.int32)
    head_group = num_output_heads // num_key_heads
    key_head = pid_head // head_group

    state = tl.load(
        initial_state
        + (pid_row * num_output_heads + pid_head) * (value_dim * key_dim)
        + val_offsets[:, None] * key_dim
        + key_offsets[None, :],
        mask=head_mask & val_mask[:, None] & key_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    state_row_stride = value_dim * key_dim

    for local_chunk in range(MAX_CHUNKS):
        in_row_chunk = local_chunk < row_chunk_count
        has_chunk = head_mask & in_row_chunk
        chunk_start = row_start + local_chunk * chunk_size
        chunk_end = tl.minimum(row_end, row_start + (local_chunk + 1) * chunk_size)
        chunk_len = chunk_end - chunk_start
        has_tokens = (chunk_len > 0) & has_chunk

        output_chunk_id = row_chunk_base + local_chunk
        h_mask = (val_mask[:, None] & key_mask[None, :]) & has_chunk
        h_offsets = ((output_chunk_id * num_output_heads + pid_head) * state_row_stride) + (
            val_offsets[:, None] * key_dim + key_offsets[None, :]
        )
        tl.store(out_h + h_offsets, state, mask=h_mask)

        update_scale = 1.0
        last_gate = 0.0
        if USE_GATE:
            row_last_gate_offset = (chunk_start + (chunk_len - 1)) * num_output_heads + pid_head
            chunk_has_tokens = has_tokens & (chunk_len > 0)
            last_gate = tl.load(
                gate_cumsum + row_last_gate_offset,
                mask=chunk_has_tokens,
                other=0.0,
            ).to(tl.float32)
            update_scale = tl.exp(last_gate)

        state_update = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
        for i in range(chunk_size):
            in_chunk = i < chunk_len
            token_mask = has_tokens & in_chunk
            token_index = chunk_start + i
            token_u = tl.load(
                u + (token_index * num_output_heads + pid_head) * value_dim + val_offsets,
                mask=token_mask & val_mask,
                other=0.0,
            ).to(tl.float32)
            token_w = tl.load(
                w + (token_index * num_output_heads + pid_head) * key_dim + key_offsets,
                mask=token_mask & key_mask,
                other=0.0,
            ).to(tl.float32)

            token_dot = tl.sum(token_w[None, :] * state, axis=1)
            delta = token_u - token_dot

            tl.store(
                out_v_new
                + (token_index * num_output_heads + pid_head) * value_dim
                + val_offsets,
                delta,
                mask=token_mask & val_mask,
            )

            if USE_GATE:
                token_gate = tl.load(
                    gate_cumsum
                    + token_index * num_output_heads
                    + pid_head,
                    mask=token_mask,
                    other=0.0,
                ).to(tl.float32)
                delta = delta * tl.exp(last_gate - token_gate)

            token_k = tl.load(
                key + (token_index * num_key_heads + key_head) * key_dim + key_offsets,
                mask=token_mask & key_mask,
                other=0.0,
            ).to(tl.float32)
            state_update += delta[:, None] * token_k[None, :]

        if USE_GATE:
            state = state * tl.where(has_tokens, update_scale, 1.0)
        state += state_update

    out_final_offsets = (
        (pid_row * num_output_heads + pid_head) * state_row_stride
        + val_offsets[:, None] * key_dim
        + key_offsets[None, :]
    )
    tl.store(
        out_final_state + out_final_offsets,
        state,
        mask=head_mask & val_mask[:, None] & key_mask[None, :],
    )


@triton.jit
def _gdn_fla_recompute_w_packed_kernel(
    key,
    beta,
    gate_cumsum,
    attention_inverse,
    cu_seqlens,
    chunk_indices,
    out,
    num_key_heads: tl.constexpr,
    num_output_heads: tl.constexpr,
    key_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)

    head_mask = pid_head < num_output_heads
    head_group = num_output_heads // num_key_heads
    key_head = pid_head // head_group
    feature_offsets = pid_block * BLOCK_D + tl.arange(0, BLOCK_D)
    feature_mask = feature_offsets < key_dim
    valid_output = head_mask & feature_mask

    chunk_row = tl.load(chunk_indices + pid_chunk * 2 + 0).to(tl.int32)
    chunk_id = tl.load(chunk_indices + pid_chunk * 2 + 1).to(tl.int32)

    row_start = tl.load(cu_seqlens + chunk_row)
    row_end = tl.load(cu_seqlens + chunk_row + 1)
    chunk_start = row_start + chunk_id * chunk_size
    chunk_end = tl.minimum(row_end, row_start + (chunk_id + 1) * chunk_size)
    chunk_len = chunk_end - chunk_start

    for local_row in range(BLOCK_S):
        row_mask = local_row < chunk_len
        row_global = chunk_start + local_row
        running = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k in range(BLOCK_S):
            inv_scale = tl.load(
                attention_inverse
                + (row_global * num_output_heads + pid_head) * chunk_size
                + k,
                mask=head_mask & row_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            beta_val = tl.load(
                beta + (chunk_start + k) * num_output_heads + pid_head,
                mask=head_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            gate_val = tl.load(
                gate_cumsum + (chunk_start + k) * num_output_heads + pid_head,
                mask=head_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            key_val = tl.load(
                key
                + ((chunk_start + k) * num_key_heads + key_head) * key_dim
                + feature_offsets,
                mask=head_mask & feature_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            running += inv_scale * key_val * beta_val * tl.exp(gate_val)
        running = tl.where(feature_mask, running, 0.0)
        running = tl.where(row_mask, running, 0.0)
        out_offsets = (row_global * num_output_heads + pid_head) * key_dim + feature_offsets
        tl.store(
            out + out_offsets,
            running,
            mask=valid_output & row_mask,
        )


@triton.jit
def _gdn_fla_recompute_u_packed_kernel(
    value,
    beta,
    attention_inverse,
    cu_seqlens,
    chunk_indices,
    out,
    num_output_heads: tl.constexpr,
    value_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)

    head_mask = pid_head < num_output_heads
    feature_offsets = pid_block * BLOCK_D + tl.arange(0, BLOCK_D)
    feature_mask = feature_offsets < value_dim
    valid_output = head_mask & feature_mask

    chunk_row = tl.load(chunk_indices + pid_chunk * 2 + 0).to(tl.int32)
    chunk_id = tl.load(chunk_indices + pid_chunk * 2 + 1).to(tl.int32)

    row_start = tl.load(cu_seqlens + chunk_row)
    row_end = tl.load(cu_seqlens + chunk_row + 1)
    chunk_start = row_start + chunk_id * chunk_size
    chunk_end = tl.minimum(row_end, row_start + (chunk_id + 1) * chunk_size)
    chunk_len = chunk_end - chunk_start

    for local_row in range(BLOCK_S):
        row_mask = local_row < chunk_len
        row_global = chunk_start + local_row
        running = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k in range(BLOCK_S):
            inv_scale = tl.load(
                attention_inverse
                + (row_global * num_output_heads + pid_head) * chunk_size
                + k,
                mask=head_mask & row_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            beta_val = tl.load(
                beta + (chunk_start + k) * num_output_heads + pid_head,
                mask=head_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            value_val = tl.load(
                value
                + ((chunk_start + k) * num_output_heads + pid_head) * value_dim
                + feature_offsets,
                mask=head_mask & feature_mask & (k < chunk_len),
                other=0.0,
            ).to(tl.float32)
            running += inv_scale * value_val * beta_val
        running = tl.where(feature_mask, running, 0.0)
        running = tl.where(row_mask, running, 0.0)
        out_offsets = (row_global * num_output_heads + pid_head) * value_dim + feature_offsets
        tl.store(
            out + out_offsets,
            running,
            mask=valid_output & row_mask,
        )


@triton.jit
def _gdn_fla_chunk_fwd_o_packed_kernel(
    query,
    key,
    v_new,
    h,
    gate_cumsum,
    cu_seqlens,
    chunk_indices,
    out,
    num_key_heads: tl.constexpr,
    num_output_heads: tl.constexpr,
    value_dim: tl.constexpr,
    key_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_GATE: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_v_block = tl.program_id(2)

    head_mask = pid_head < num_output_heads
    row_offsets = tl.arange(0, BLOCK_T)
    val_offsets = pid_v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    val_mask = val_offsets < value_dim

    chunk_row = tl.load(chunk_indices + pid_chunk * 2 + 0).to(tl.int32)
    chunk_id = tl.load(chunk_indices + pid_chunk * 2 + 1).to(tl.int32)

    row_start = tl.load(cu_seqlens + chunk_row).to(tl.int32)
    row_end = tl.load(cu_seqlens + chunk_row + 1).to(tl.int32)
    chunk_start = row_start + chunk_id * chunk_size
    chunk_end = tl.minimum(row_end, row_start + (chunk_id + 1) * chunk_size)
    chunk_len = chunk_end - chunk_start

    valid_rows = row_offsets < chunk_len
    row_token_idx = chunk_start + row_offsets

    head_group = num_output_heads // num_key_heads
    key_head = pid_head // head_group

    query_row_ptr = (row_token_idx * num_key_heads + key_head) * key_dim
    state_row_ptr = (
        (pid_chunk * num_output_heads + pid_head) * value_dim * key_dim
    )

    state_out = tl.zeros((BLOCK_T, BLOCK_V), dtype=tl.float32)
    attention = tl.zeros((BLOCK_T, BLOCK_V), dtype=tl.float32)
    if USE_GATE:
        row_gate = tl.load(
            gate_cumsum + row_token_idx * num_output_heads + pid_head,
            mask=valid_rows & head_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        row_gate = 0.0

    for k in range(BLOCK_K):
        state_k = tl.load(
            h + state_row_ptr + val_offsets * key_dim + k,
            mask=val_mask & (k < key_dim),
            other=0.0,
        ).to(tl.float32)
        query_k = tl.load(
            query + query_row_ptr + k,
            mask=valid_rows & head_mask & (k < key_dim),
            other=0.0,
        ).to(tl.float32)
        state_out += query_k[:, None] * state_k[None, :]

        for j in range(BLOCK_T):
            col_mask = (j < chunk_len) & (j <= row_offsets)
            col_pos = chunk_start + j
            key_k = tl.load(
                key + ((col_pos * num_key_heads + key_head) * key_dim + k),
                mask=(j < chunk_len) & head_mask & (k < key_dim),
                other=0.0,
            ).to(tl.float32)
            dot = query_k * key_k
            if USE_GATE:
                col_gate = tl.load(
                    gate_cumsum + col_pos * num_output_heads + pid_head,
                    mask=(j < chunk_len) & head_mask,
                    other=0.0,
                ).to(tl.float32)
                dot *= tl.exp(row_gate - col_gate)
            v_vals = tl.load(
                v_new + (col_pos * num_output_heads + pid_head) * value_dim + val_offsets,
                mask=((j < chunk_len) & head_mask) & val_mask,
                other=0.0,
            ).to(tl.float32)
            attention += tl.where(
                col_mask[:, None],
                dot[:, None] * v_vals[None, :],
                0.0,
            )

    if USE_GATE:
        state_out *= tl.exp(row_gate)[:, None]

    out_vals = (state_out + attention).to(tl.float32)
    out_offsets = (
        (row_token_idx[:, None] * num_output_heads + pid_head) * value_dim
        + val_offsets[None, :]
    )
    store_mask = valid_rows[:, None] & val_mask[None, :] & head_mask
    tl.store(out + out_offsets, out_vals, mask=store_mask)


def gdn_fla_chunk_fwd_o_packed_triton(
    query: jax.Array,
    key: jax.Array,
    v_new: jax.Array,
    h: jax.Array,
    gate_cumsum: jax.Array | None,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
    scale: float | None = None,
) -> jax.Array:
    """Packed stage-6 FLA chunk forward-output over packed varlen tensors."""

    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [nnz_tokens, key_heads, key_dim]")
    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")
    if v_new.ndim != 3:
        raise ValueError("v_new must have shape [nnz_tokens, output_heads, value_dim]")
    if h.ndim != 4:
        raise ValueError("h must have shape [num_chunks, output_heads, value_dim, key_dim]")
    if query.shape[0] != v_new.shape[0]:
        raise ValueError("query and v_new token counts must match")
    if v_new.shape[1] != h.shape[1]:
        raise ValueError("v_new and h output-head dimensions must match")
    if h.shape[2] != v_new.shape[2] or h.shape[3] != query.shape[2]:
        raise ValueError("h value/key dimensions must match v_new/query")
    if gate_cumsum is not None and gate_cumsum.shape != v_new.shape[:2]:
        raise ValueError("gate_cumsum must have shape [nnz_tokens, output_heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if scale is None:
        scale = float(query.shape[-1] ** -0.5)

    key_dim = query.shape[2]
    value_dim = v_new.shape[2]
    num_key_heads = query.shape[1]
    output_heads = v_new.shape[1]
    if output_heads < num_key_heads or output_heads % num_key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    query_fp32 = query.astype(jnp.float32)
    key_fp32 = key.astype(jnp.float32)
    v_new_fp32 = v_new.astype(jnp.float32)
    h_fp32 = h.astype(jnp.float32)
    if gate_cumsum is None:
        gate_fp32 = jnp.zeros((query.shape[0], output_heads), dtype=jnp.float32)
        use_gate = False
        reference_gate = None
    else:
        gate_fp32 = gate_cumsum.astype(jnp.float32)
        use_gate = True
        reference_gate = gate_fp32

    cu = cu_seqlens.astype(jnp.int32)
    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu, chunk_size)

    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    num_chunks = int(chunk_indices.shape[0])
    if num_chunks == 0:
        return jnp.zeros(v_new.shape, dtype=jnp.float32)

    block_t = int(jt.next_power_of_2(int(chunk_size)))
    block_v = int(jt.next_power_of_2(int(value_dim)))
    block_k = int(jt.next_power_of_2(int(key_dim)))
    if block_t > 1024 or block_v > 1024 or block_k > 1024:
        from nanovllm_jax.kernels.gdn_fla import (
            gdn_fla_chunk_fwd_o_packed_reference,
        )

        return gdn_fla_chunk_fwd_o_packed_reference(
            query_fp32,
            key_fp32,
            v_new_fp32,
            h_fp32,
            reference_gate,
            cu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            scale=scale,
        )

    out_shape = jax.ShapeDtypeStruct(v_new.shape, jnp.float32)
    return jt.triton_call(
        query_fp32,
        key_fp32,
        v_new_fp32,
        h_fp32,
        gate_fp32,
        cu,
        chunk_indices,
        kernel=_gdn_fla_chunk_fwd_o_packed_kernel,
        out_shape=out_shape,
        grid=(num_chunks, output_heads, jt.cdiv(value_dim, block_v)),
        num_key_heads=int(num_key_heads),
        num_output_heads=int(output_heads),
        value_dim=value_dim,
        key_dim=int(key_dim),
        chunk_size=chunk_size,
        BLOCK_T=block_t,
        BLOCK_V=block_v,
        BLOCK_K=block_k,
        USE_GATE=_normalize_bool(use_gate),
        num_warps=1,
        num_stages=2,
    ) * float(scale)


def gdn_fla_chunk_gated_delta_rule_packed_triton(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    cu_seqlens: jax.Array,
    initial_state: jax.Array,
    *,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: jax.Array | None = None,
    chunk_offsets: jax.Array | None = None,
    max_row_chunks: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compose Triton FLA chunk stages for packed varlen prefill."""

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must have shape [nnz_tokens, heads, dim]")
    if query.shape != key.shape:
        raise ValueError("query and key must match")
    if value.shape[:2] != query.shape[:2]:
        raise ValueError("value must match token/head shape with query")
    if gate.shape != query.shape[:2] or beta.shape != query.shape[:2]:
        raise ValueError("gate and beta must match [tokens, heads]")
    if initial_state.shape[:2] != (cu_seqlens.shape[0] - 1, query.shape[1]):
        raise ValueError("initial_state batch/head dimensions must match cu_seqlens/query")

    if use_qk_l2norm_in_kernel:
        from nanovllm_jax.kernels.gdn_fla import (
            gdn_fla_chunk_gated_delta_rule_packed_reference,
        )

        return gdn_fla_chunk_gated_delta_rule_packed_reference(
            query,
            key,
            value,
            gate,
            beta,
            cu_seqlens,
            initial_state,
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=True,
        )

    if chunk_indices is None:
        from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    query_fp32 = query.astype(jnp.float32)
    key_fp32 = key.astype(jnp.float32)
    value_fp32 = value.astype(jnp.float32)
    gate_fp32 = gate.astype(jnp.float32)
    beta_fp32 = beta.astype(jnp.float32)
    state_fp32 = initial_state.astype(jnp.float32)

    cu = cu_seqlens.astype(jnp.int32)
    if chunk_indices is None:
        chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(cu, chunk_size)
    if chunk_indices is None:
        raise ValueError("chunk_indices must be provided for the dense/padded path")
    if chunk_offsets is None:
        raise ValueError("chunk_offsets must be provided when chunk_indices is provided")

    gate_cumsum = gdn_fla_chunk_local_cumsum_packed_triton(
        gate_fp32,
        cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    attention_matrix = gdn_fla_chunk_scaled_dot_kkt_packed_triton(
        key_fp32,
        beta_fp32,
        gate_cumsum,
        cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    attention_inverse = gdn_fla_solve_tril_packed_triton(
        attention_matrix,
        cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    w, u = gdn_fla_recompute_w_u_packed_triton(
        key_fp32,
        value_fp32,
        beta_fp32,
        gate_cumsum,
        attention_inverse,
        cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = gdn_fla_chunk_delta_h_packed_triton(
        key_fp32,
        w,
        u,
        gate_cumsum,
        cu,
        state_fp32,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        max_row_chunks=max_row_chunks,
        output_final_state=True,
        save_new_value=True,
    )
    if v_new is None or final_state is None:
        raise AssertionError("chunk delta h should return v_new and final_state")

    output = gdn_fla_chunk_fwd_o_packed_triton(
        query_fp32,
        key_fp32,
        v_new,
        h,
        gate_cumsum,
        cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        scale=float(query.shape[-1] ** -0.5),
    )
    return output, final_state


@triton.jit
def _gdn_packed_decode_kernel(
    mixed_qkv,
    gate,
    beta,
    state,
    out,
    new_state,
    qkv_dim: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    FULL_K: tl.constexpr,
    FULL_V: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    p_state = state + ((i_n * HV + i_hv) * V * K)

    p_mixed = mixed_qkv + i_n * qkv_dim
    if FULL_K:
        q = tl.load(p_mixed + i_h * K + offs_k).to(tl.float32)
        k = tl.load(p_mixed + H * K + i_h * K + offs_k).to(tl.float32)
    else:
        q = tl.load(
            p_mixed + i_h * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            p_mixed + H * K + i_h * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
    if FULL_V:
        v = tl.load(p_mixed + 2 * H * K + i_hv * V + offs_v).to(tl.float32)
    else:
        v = tl.load(
            p_mixed + 2 * H * K + i_hv * V + offs_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        q_scale = SCALE / (tl.sqrt(tl.sum(q * q, axis=0) + 1e-6))
        k_scale = 1.0 / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6)
    else:
        q_scale = SCALE
        k_scale = 1.0

    gate_val = tl.load(gate + i_n * HV + i_hv).to(tl.float32)
    beta_val = tl.load(beta + i_n * HV + i_hv).to(tl.float32)

    gate_scale = tl.exp(gate_val)

    if FULL_K and FULL_V:
        h = tl.load(
            p_state + offs_v[:, None] * K + offs_k[None, :],
            mask=mask_v[:, None] & mask_k[None, :],
        ).to(tl.float32)
    elif FULL_K:
        h = tl.load(
            p_state + offs_v[:, None] * K + offs_k[None, :],
            mask=mask_v[:, None],
            other=0.0,
        ).to(tl.float32)
    elif FULL_V:
        h = tl.load(
            p_state + offs_v[:, None] * K + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        h = tl.load(
            p_state + offs_v[:, None] * K + offs_k[None, :],
            mask=mask_v[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

    h = h * gate_scale
    q = q * q_scale
    k = k * k_scale
    kv_mem = tl.sum(h * k[None, :], axis=1)
    state_dot_q = tl.sum(h * q[None, :], axis=1)
    qk_dot = tl.sum(q * k, axis=0)
    delta = (v - kv_mem) * beta_val
    o = state_dot_q + delta * qk_dot

    h = h + delta[:, None] * k[None, :]

    p_out = out + ((i_n * HV + i_hv) * V) + offs_v
    if FULL_V:
        tl.store(p_out, o)
    else:
        tl.store(p_out, o, mask=mask_v)

    p_new_state = new_state + ((i_n * HV + i_hv) * V * K)
    if FULL_K and FULL_V:
        tl.store(
            p_new_state + offs_v[:, None] * K + offs_k[None, :],
            h,
            mask=mask_v[:, None] & mask_k[None, :],
        )
    elif FULL_K:
        tl.store(
            p_new_state + offs_v[:, None] * K + offs_k[None, :],
            h,
            mask=mask_v[:, None],
        )
    elif FULL_V:
        tl.store(
            p_new_state + offs_v[:, None] * K + offs_k[None, :],
            h,
            mask=mask_k[None, :],
        )
    else:
        tl.store(
            p_new_state + offs_v[:, None] * K + offs_k[None, :],
            h,
            mask=mask_v[:, None] & mask_k[None, :],
        )


@triton.jit
def _gdn_post_conv_prep_kernel(
    conv_out,
    a,
    b,
    decay,
    dt_bias,
    valid_mask,
    query,
    key,
    value,
    gate,
    beta,
    total_tokens: tl.constexpr,
    seq_len: tl.constexpr,
    conv_dim: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HEADS_PER_KEY: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    i_hv = tl.program_id(1)
    offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs < total_tokens
    batch = offs // seq_len
    t = offs - batch * seq_len
    valid = tl.load(valid_mask + offs, mask=mask_t, other=0) > 0

    i_h = i_hv // HEADS_PER_KEY
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    token_base = (batch[:, None] * seq_len + t[:, None]) * conv_dim

    q_offsets = token_base + i_h * K + offs_k[None, :]
    k_offsets = token_base + H * K + i_h * K + offs_k[None, :]
    q = tl.load(conv_out + q_offsets, mask=valid[:, None] & mask_k[None, :], other=0.0).to(
        tl.float32
    )
    k = tl.load(conv_out + k_offsets, mask=valid[:, None] & mask_k[None, :], other=0.0).to(
        tl.float32
    )
    if APPLY_L2NORM:
        q = q / tl.sqrt(tl.sum(q * q, axis=1)[:, None] + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k, axis=1)[:, None] + 1e-6)

    q_out = ((batch[:, None] * seq_len + t[:, None]) * HV + i_hv) * K + offs_k[None, :]
    tl.store(query + q_out, q, mask=valid[:, None] & mask_k[None, :])
    tl.store(key + q_out, k, mask=valid[:, None] & mask_k[None, :])

    v_offsets = token_base + 2 * H * K + i_hv * V + offs_v[None, :]
    v = tl.load(conv_out + v_offsets, mask=valid[:, None] & mask_v[None, :], other=0.0)
    v_out = ((batch[:, None] * seq_len + t[:, None]) * HV + i_hv) * V + offs_v[None, :]
    tl.store(value + v_out, v, mask=valid[:, None] & mask_v[None, :])

    scalar_offsets = (batch * seq_len + t) * HV + i_hv
    a_val = tl.load(a + scalar_offsets, mask=valid, other=0.0).to(tl.float32)
    b_val = tl.load(b + scalar_offsets, mask=valid, other=0.0).to(tl.float32)
    decay_val = tl.load(decay + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(
        x > 0.0,
        x + tl.log(1.0 + tl.exp(-x)),
        tl.log(1.0 + tl.exp(x)),
    )
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, softplus_x, x)
    gate_val = -decay_val * softplus_x
    beta_val = tl.sigmoid(b_val)
    tl.store(gate + scalar_offsets, gate_val, mask=valid)
    tl.store(beta + scalar_offsets, beta_val, mask=valid)


def _normalize_bool(value: bool) -> bool:
    return bool(value)


def gdn_fla_chunk_local_cumsum_packed_triton(
    gate: jax.Array,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
    reverse: bool = False,
) -> jax.Array:
    """Packed stage-1 FLA chunk-local cumsum from Triton.

    The function keeps the same shape contract as
    ``gdn_fla_chunk_local_cumsum_packed_reference`` and returns FP32 output.
    """

    if gate.ndim != 2:
        raise ValueError("gate must have shape [nnz_tokens, heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    gate_fp32 = gate.astype(jnp.float32)
    num_heads = gate_fp32.shape[1]
    if num_heads <= 0:
        return jnp.zeros_like(gate_fp32)

    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    cu = cu_seqlens.astype(jnp.int32)
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu, chunk_size)

    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    num_chunks = int(chunk_indices.shape[0])
    if num_chunks == 0:
        return jnp.zeros(gate_fp32.shape, dtype=jnp.float32)

    block = int(jt.next_power_of_2(int(chunk_size)))
    if block > 1024:
        from nanovllm_jax.kernels.gdn_fla import gdn_fla_chunk_local_cumsum_packed_reference

        return gdn_fla_chunk_local_cumsum_packed_reference(
            gate_fp32,
            cu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            reverse=reverse,
        )

    out_shape = jax.ShapeDtypeStruct(gate_fp32.shape, jnp.float32)
    return jt.triton_call(
        gate_fp32,
        cu,
        chunk_indices,
        kernel=_gdn_fla_chunk_local_cumsum_packed_kernel,
        out_shape=out_shape,
        grid=(num_chunks, int(num_heads)),
        num_heads=num_heads,
        chunk_size=chunk_size,
        BLOCK=block,
        REVERSE=_normalize_bool(reverse),
        num_warps=1,
        num_stages=2,
    )


def gdn_fla_chunk_scaled_dot_kkt_packed_triton(
    key: jax.Array,
    beta: jax.Array,
    gate_cumsum: jax.Array | None,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
) -> jax.Array:
    """Packed stage-2 FLA chunk scaled-dot over packed varlen tensors."""

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if beta.ndim != 2:
        raise ValueError("beta must have shape [nnz_tokens, output_heads]")
    if key.shape[0] != beta.shape[0]:
        raise ValueError("key and beta token counts must match")
    if gate_cumsum is not None and gate_cumsum.shape != beta.shape:
        raise ValueError("gate_cumsum must have the same shape as beta")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    cu = cu_seqlens.astype(jnp.int32)
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu, chunk_size)

    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    num_chunks = int(chunk_indices.shape[0])
    if num_chunks == 0:
        return jnp.zeros((key.shape[0], beta.shape[1], int(chunk_size)), dtype=jnp.float32)

    key_fp32 = key.astype(jnp.float32)
    beta_fp32 = beta.astype(jnp.float32)
    if gate_cumsum is None:
        gate_fp32 = jnp.zeros_like(beta_fp32)
        use_gate = False
        reference_gate = None
    else:
        gate_fp32 = gate_cumsum.astype(jnp.float32)
        use_gate = True
        reference_gate = gate_fp32

    key_heads = key.shape[1]
    output_heads = beta.shape[1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")
    key_dim = key.shape[2]
    if key_dim <= 0:
        raise ValueError("key_dim must be positive")
    if gate_fp32.shape[0] != key.shape[0] or gate_fp32.shape[1] != output_heads:
        raise ValueError("gate_cumsum must have the same shape as beta")

    block_s = int(jt.next_power_of_2(int(chunk_size)))
    block_k = int(jt.next_power_of_2(int(key_dim)))
    if block_s > 1024 or block_k > 1024:
        from nanovllm_jax.kernels.gdn_fla import (
            gdn_fla_chunk_scaled_dot_kkt_packed_reference,
        )

        return gdn_fla_chunk_scaled_dot_kkt_packed_reference(
            key_fp32,
            beta_fp32,
            reference_gate,
            cu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

    out_shape = jax.ShapeDtypeStruct(
        (key.shape[0], output_heads, int(chunk_size)),
        jnp.float32,
    )
    return jt.triton_call(
        key_fp32,
        beta_fp32,
        gate_fp32,
        cu,
        chunk_indices,
        kernel=_gdn_fla_chunk_scaled_dot_kkt_packed_kernel,
        out_shape=out_shape,
        grid=(num_chunks, int(output_heads), block_s),
        num_key_heads=key_heads,
        num_output_heads=output_heads,
        key_dim=key_dim,
        chunk_size=chunk_size,
        BLOCK_K=block_k,
        BLOCK_S=block_s,
        USE_GATE=_normalize_bool(use_gate),
        num_warps=1,
        num_stages=2,
    )


def gdn_fla_solve_tril_packed_triton(
    attention_matrix: jax.Array,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
) -> jax.Array:
    """Packed stage-3 FLA chunk solve over packed varlen tensors."""

    if attention_matrix.ndim != 3:
        raise ValueError("attention_matrix must have shape [nnz_tokens, heads, chunk_size]")
    if attention_matrix.shape[-1] != chunk_size:
        raise ValueError("attention_matrix last dimension must equal chunk_size")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    cu = cu_seqlens.astype(jnp.int32)
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu, chunk_size)

    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    num_chunks = int(chunk_indices.shape[0])
    if num_chunks == 0:
        return jnp.zeros_like(attention_matrix, dtype=jnp.float32)

    attention_matrix_fp32 = attention_matrix.astype(jnp.float32)
    num_heads = attention_matrix.shape[1]
    block_s = int(jt.next_power_of_2(int(chunk_size)))
    if block_s > 1024:
        from nanovllm_jax.kernels.gdn_fla import gdn_fla_solve_tril_packed_reference

        return gdn_fla_solve_tril_packed_reference(
            attention_matrix_fp32,
            cu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

    out_shape = jax.ShapeDtypeStruct(
        (attention_matrix.shape[0], num_heads, int(chunk_size)),
        jnp.float32,
    )
    return jt.triton_call(
        attention_matrix_fp32,
        cu,
        chunk_indices,
        kernel=_gdn_fla_solve_tril_packed_kernel,
        out_shape=out_shape,
        grid=(num_chunks, int(num_heads)),
        num_heads=int(num_heads),
        chunk_size=chunk_size,
        BLOCK=block_s,
        num_warps=1,
        num_stages=2,
    )


def gdn_fla_chunk_delta_h_packed_triton(
    key: jax.Array,
    w: jax.Array,
    u: jax.Array,
    gate_cumsum: jax.Array | None,
    cu_seqlens: jax.Array,
    initial_state: jax.Array | None,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
    chunk_offsets: Any | None = None,
    max_row_chunks: int | None = None,
    output_final_state: bool = True,
    save_new_value: bool = True,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Packed stage-5 FLA chunk delta-h over packed varlen tensors."""

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if w.ndim != 3:
        raise ValueError("w must have shape [nnz_tokens, output_heads, key_dim]")
    if u.ndim != 3:
        raise ValueError("u must have shape [nnz_tokens, output_heads, value_dim]")
    if key.shape[0] != w.shape[0] or key.shape[0] != u.shape[0]:
        raise ValueError("key, w, and u token counts must match")
    if w.shape[:2] != u.shape[:2]:
        raise ValueError("w and u must agree on token and output heads")
    if w.shape[-1] != key.shape[2]:
        raise ValueError("w key dimension must match key")
    if gate_cumsum is not None and gate_cumsum.shape != w.shape[:2]:
        raise ValueError("gate_cumsum must have shape [nnz_tokens, output_heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not output_final_state or not save_new_value:
        from nanovllm_jax.kernels.gdn_fla import (
            gdn_fla_chunk_delta_h_packed_reference,
        )

        return gdn_fla_chunk_delta_h_packed_reference(
            key,
            w,
            u,
            gate_cumsum,
            cu_seqlens,
            initial_state,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            output_final_state=output_final_state,
            save_new_value=save_new_value,
        )

    cu = cu_seqlens.astype(jnp.int32)
    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    if chunk_indices is None or chunk_offsets is None:
        chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(cu, chunk_size)
    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")
    if chunk_offsets.ndim != 1:
        raise ValueError("chunk_offsets must have shape [batch + 1]")

    num_chunks = int(chunk_indices.shape[0])
    if num_chunks == 0:
        return (
            jnp.zeros((0, w.shape[1], u.shape[2], key.shape[2]), dtype=jnp.float32),
            jnp.zeros_like(u, dtype=jnp.float32),
            jnp.zeros(
                (int(chunk_offsets.shape[0]) - 1, w.shape[1], u.shape[2], key.shape[2]),
                dtype=jnp.float32,
            ),
        )

    key_fp32 = key.astype(jnp.float32)
    w_fp32 = w.astype(jnp.float32)
    u_fp32 = u.astype(jnp.float32)
    gate_fp32 = None if gate_cumsum is None else gate_cumsum.astype(jnp.float32)
    if gate_cumsum is None:
        gate_fp32 = jnp.zeros((key.shape[0], w.shape[1]), dtype=jnp.float32)
    if chunk_offsets.shape[0] != cu.shape[0]:
        raise ValueError("chunk_offsets must have shape [batch + 1] matching cu_seqlens")
    batch = int(cu.shape[0] - 1)
    key_heads = key.shape[1]
    output_heads = w.shape[1]
    key_dim = key.shape[2]
    value_dim = u.shape[2]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    if initial_state is None:
        state = jnp.zeros(
            (batch, output_heads, value_dim, key_dim),
            dtype=jnp.float32,
        )
    else:
        if initial_state.shape != (
            batch,
            output_heads,
            value_dim,
            key_dim,
        ):
            raise ValueError("initial_state must have shape [batch, output_heads, value_dim, key_dim]")
        state = initial_state.astype(jnp.float32)

    # Derive fallback thresholds from host materialized lengths only in eager mode.
    # Tracing through JAX should avoid eager host round-trips.
    is_tracer = isinstance(cu, core.Tracer)
    if max_row_chunks is None:
        if is_tracer:
            max_row_chunks = 1024
        else:
            row_lengths = np.asarray(
                jax.device_get(cu[1:] - cu[:-1]), dtype=np.int64
            ).reshape(-1)
            if row_lengths.size == 0:
                max_row_chunks = 0
            else:
                max_row_chunks = int(((row_lengths + chunk_size - 1) // chunk_size).max())
    block_k = int(jt.next_power_of_2(int(key_dim)))
    block_v = int(jt.next_power_of_2(int(value_dim)))
    if (
        (not is_tracer and (max_row_chunks == 0 or max_row_chunks > 256))
        or block_k > 1024
        or block_v > 1024
    ):
        from nanovllm_jax.kernels.gdn_fla import (
            gdn_fla_chunk_delta_h_packed_reference,
        )
        return gdn_fla_chunk_delta_h_packed_reference(
            key_fp32,
            w_fp32,
            u_fp32,
            gate_fp32,
            cu,
            state,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            output_final_state=True,
            save_new_value=True,
        )

    h_shape = jax.ShapeDtypeStruct(
        (num_chunks, output_heads, value_dim, key_dim),
        jnp.float32,
    )
    v_shape = jax.ShapeDtypeStruct(
        (key.shape[0], output_heads, value_dim),
        jnp.float32,
    )
    final_shape = jax.ShapeDtypeStruct(
        (batch, output_heads, value_dim, key_dim),
        jnp.float32,
    )
    return jt.triton_call(
        key_fp32,
        w_fp32,
        u_fp32,
        gate_fp32,
        state,
        cu,
        chunk_indices,
        chunk_offsets,
        kernel=_gdn_fla_chunk_delta_h_packed_kernel,
        out_shape=(h_shape, v_shape, final_shape),
        grid=(batch, output_heads),
        num_key_heads=key_heads,
        num_output_heads=output_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk_size,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        MAX_CHUNKS=max_row_chunks,
        USE_GATE=_normalize_bool(gate_cumsum is not None),
        num_warps=1,
        num_stages=2,
    )


def gdn_fla_recompute_w_u_packed_triton(
    key: jax.Array,
    value: jax.Array,
    beta: jax.Array,
    gate_cumsum: jax.Array,
    attention_inverse: jax.Array,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int,
    chunk_indices: Any | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Packed stage-4 FLA recompute (w, u) over packed varlen tensors."""

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if value.ndim != 3:
        raise ValueError("value must have shape [nnz_tokens, output_heads, value_dim]")
    if beta.ndim != 2 or gate_cumsum.ndim != 2:
        raise ValueError("beta and gate_cumsum must have shape [nnz_tokens, output_heads]")
    if beta.shape != gate_cumsum.shape:
        raise ValueError("beta and gate_cumsum must have the same shape")
    if key.shape[0] != value.shape[0] or key.shape[0] != beta.shape[0]:
        raise ValueError("key, value, and beta token counts must match")
    if value.shape[:2] != beta.shape:
        raise ValueError("value and beta must agree on token and output heads")
    if attention_inverse.shape != (key.shape[0], beta.shape[1], chunk_size):
        raise ValueError("attention_inverse shape must match [tokens, output_heads, chunk_size]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    key_dim = key.shape[2]
    value_dim = value.shape[2]
    num_key_heads = key.shape[1]
    output_heads = beta.shape[1]
    if output_heads < num_key_heads or output_heads % num_key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    key_fp32 = key.astype(jnp.float32)
    value_fp32 = value.astype(jnp.float32)
    beta_fp32 = beta.astype(jnp.float32)
    gate_fp32 = gate_cumsum.astype(jnp.float32)
    inverse_fp32 = attention_inverse.astype(jnp.float32)

    if key_fp32.shape[0] == 0:
        empty_w = jnp.zeros((0, output_heads, key_dim), dtype=jnp.float32)
        empty_u = jnp.zeros((0, output_heads, value_dim), dtype=jnp.float32)
        return empty_w, empty_u

    from nanovllm_jax.kernels.gdn_fla import prepare_gdn_fla_chunk_metadata

    cu = cu_seqlens.astype(jnp.int32)
    if chunk_indices is None:
        chunk_indices, _ = prepare_gdn_fla_chunk_metadata(cu, chunk_size)

    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    num_chunks = int(chunk_indices.shape[0])
    if num_chunks == 0:
        return (
            jnp.zeros((key.shape[0], output_heads, key_dim), dtype=jnp.float32),
            jnp.zeros((key.shape[0], output_heads, value_dim), dtype=jnp.float32),
        )

    block_s = int(jt.next_power_of_2(int(chunk_size)))
    block_k = int(jt.next_power_of_2(int(key_dim)))
    block_u = int(jt.next_power_of_2(int(value_dim)))
    if block_s > 1024 or block_k > 1024 or block_u > 1024:
        from nanovllm_jax.kernels.gdn_fla import gdn_fla_recompute_w_u_packed_reference

        return gdn_fla_recompute_w_u_packed_reference(
            key_fp32,
            value_fp32,
            beta_fp32,
            gate_fp32,
            inverse_fp32,
            cu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

    w_shape = jax.ShapeDtypeStruct(
        (key.shape[0], output_heads, key_dim),
        jnp.float32,
    )
    u_shape = jax.ShapeDtypeStruct(
        (key.shape[0], output_heads, value_dim),
        jnp.float32,
    )

    w = jt.triton_call(
        key_fp32,
        beta_fp32,
        gate_fp32,
        inverse_fp32,
        cu,
        chunk_indices,
        kernel=_gdn_fla_recompute_w_packed_kernel,
        out_shape=w_shape,
        grid=(num_chunks, output_heads, jt.cdiv(key_dim, block_k)),
        num_key_heads=num_key_heads,
        num_output_heads=output_heads,
        key_dim=key_dim,
        chunk_size=chunk_size,
        BLOCK_S=block_s,
        BLOCK_D=block_k,
        num_warps=1,
        num_stages=2,
    )

    u = jt.triton_call(
        value_fp32,
        beta_fp32,
        inverse_fp32,
        cu,
        chunk_indices,
        kernel=_gdn_fla_recompute_u_packed_kernel,
        out_shape=u_shape,
        grid=(num_chunks, output_heads, jt.cdiv(value_dim, block_u)),
        num_output_heads=output_heads,
        value_dim=value_dim,
        chunk_size=chunk_size,
        BLOCK_S=block_s,
        BLOCK_D=block_u,
        num_warps=1,
        num_stages=2,
    )
    return w, u


def gdn_post_conv_prep_bf16(
    conv_out: jax.Array,
    a: jax.Array,
    b: jax.Array,
    decay: jax.Array,
    dt_bias: jax.Array,
    valid_token_mask: jax.Array,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    normalize_qk: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused BF16 post-conv prep for the prepared FLA prefill boundary."""

    if conv_out.ndim != 3:
        raise ValueError("conv_out must have shape [batch, time, conv_dim]")
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must have shape [batch, time, value_heads]")
    if valid_token_mask.ndim != 2:
        raise ValueError("valid_token_mask must have shape [batch, time]")
    batch, seq_len, conv_dim = conv_out.shape
    if a.shape != (batch, seq_len, num_value_heads):
        raise ValueError("a must have shape [batch, time, value_heads]")
    if b.shape != (batch, seq_len, num_value_heads):
        raise ValueError("b must have shape [batch, time, value_heads]")
    if valid_token_mask.shape != (batch, seq_len):
        raise ValueError("valid_token_mask must have shape [batch, time]")
    expected_conv_dim = 2 * num_key_heads * key_head_dim + num_value_heads * value_head_dim
    if conv_dim != expected_conv_dim:
        raise ValueError(f"conv_out last dimension must be {expected_conv_dim}, got {conv_dim}")
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    if decay.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("decay and dt_bias must have shape [value_heads]")

    valid = valid_token_mask.astype(jnp.int32)
    block_t = 8
    block_k = jt.next_power_of_2(key_head_dim)
    block_v = jt.next_power_of_2(value_head_dim)
    out_shape = (
        jax.ShapeDtypeStruct(
            (batch, seq_len, num_value_heads, key_head_dim),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct(
            (batch, seq_len, num_value_heads, key_head_dim),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct(
            (batch, seq_len, num_value_heads, value_head_dim),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct((batch, seq_len, num_value_heads), jnp.float32),
        jax.ShapeDtypeStruct((batch, seq_len, num_value_heads), jnp.float32),
    )
    return jt.triton_call(
        conv_out.astype(jnp.float32),
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        decay.astype(jnp.float32),
        dt_bias.astype(jnp.float32),
        valid.reshape(batch * seq_len),
        kernel=_gdn_post_conv_prep_kernel,
        out_shape=out_shape,
        grid=(jt.cdiv(batch * seq_len, block_t), num_value_heads),
        total_tokens=batch * seq_len,
        seq_len=seq_len,
        conv_dim=conv_dim,
        H=num_key_heads,
        HV=num_value_heads,
        K=key_head_dim,
        V=value_head_dim,
        BK=block_k,
        BV=block_v,
        BLOCK_T=block_t,
        HEADS_PER_KEY=num_value_heads // num_key_heads,
        APPLY_L2NORM=_normalize_bool(normalize_qk),
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=4,
        num_stages=_decode_triton_num_stages(),
        zeroed_outputs=(0, 1, 2, 3, 4),
    )


def gdn_packed_decode_step_bf16(
    mixed_qkv: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run the vLLM/FLA-shaped packed decode step from JAX.

    Inputs:
    - ``mixed_qkv``: ``[B, 2 * H * K + HV * V]`` BF16.
    - ``gate``/``beta``: ``[B, HV]`` FP32, where ``gate = -decay * softplus(a + dt_bias)``.
    - ``state``: ``[B, HV, V, K]`` FP32.

    Returns ``output [B, HV, 1, V]`` and ``new_state [B, HV, V, K]`` in FP32.
    """

    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [batch, packed_dim]")
    if gate.ndim != 2 or beta.ndim != 2:
        raise ValueError("gate and beta must have shape [batch, value_heads]")
    if state.ndim != 4:
        raise ValueError("state must have shape [batch, value_heads, value_dim, key_dim]")
    if mixed_qkv.dtype != jnp.bfloat16:
        raise ValueError("gdn_packed_decode_step_bf16 requires BF16 packed QKV")
    for name, array in (
        ("gate", gate),
        ("beta", beta),
        ("state", state),
    ):
        if array.dtype != jnp.float32:
            raise ValueError(f"{name} must be float32")

    batch, packed_dim = mixed_qkv.shape
    state_batch, value_heads, value_dim, key_dim = state.shape
    if state_batch != batch:
        raise ValueError("state batch must match mixed_qkv batch")
    if gate.shape != (batch, value_heads) or beta.shape != (batch, value_heads):
        raise ValueError("gate and beta must have shape [batch, value_heads]")
    qk_dim = packed_dim - value_heads * value_dim
    if qk_dim <= 0 or qk_dim % (2 * key_dim) != 0:
        raise ValueError("mixed_qkv has an invalid packed Q/K dimension")
    num_q_heads = qk_dim // (2 * key_dim)
    if value_heads % num_q_heads != 0:
        raise ValueError("value_heads must be divisible by num_q_heads")

    block_k = jt.next_power_of_2(key_dim)
    block_v = _decode_triton_block_v(value_dim)
    num_warps = _decode_triton_num_warps(value_dim)
    out_shape = (
        jax.ShapeDtypeStruct((batch, value_heads, 1, value_dim), jnp.float32),
        jax.ShapeDtypeStruct(state.shape, jnp.float32),
    )
    return jt.triton_call(
        mixed_qkv,
        gate,
        beta,
        state,
        kernel=_gdn_packed_decode_kernel,
        out_shape=out_shape,
        grid=(jt.cdiv(value_dim, block_v), batch * value_heads),
        qkv_dim=packed_dim,
        H=num_q_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        FULL_K=key_dim == block_k,
        FULL_V=value_dim % block_v == 0,
        SCALE=1.0 / (key_dim**0.5),
        USE_QK_L2NORM_IN_KERNEL=_normalize_bool(use_qk_l2norm_in_kernel),
        num_warps=num_warps,
        num_stages=_decode_triton_num_stages(),
    )


def available() -> bool:
    try:
        _configure_triton_runtime()
        return True
    except Exception:
        return False


__all__ = [
    "available",
    "gdn_fla_chunk_local_cumsum_packed_triton",
    "gdn_fla_chunk_scaled_dot_kkt_packed_triton",
    "gdn_fla_solve_tril_packed_triton",
    "gdn_fla_chunk_delta_h_packed_triton",
    "gdn_fla_recompute_w_u_packed_triton",
    "gdn_fla_chunk_fwd_o_packed_triton",
    "gdn_fla_chunk_gated_delta_rule_packed_triton",
    "gdn_packed_decode_step_bf16",
    "gdn_post_conv_prep_bf16",
]
