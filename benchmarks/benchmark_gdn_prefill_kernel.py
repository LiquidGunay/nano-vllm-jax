#!/usr/bin/env python3
"""Standalone Gated DeltaNet prefill kernel microbenchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

from run_tracking import RunRecorder
from runtime_paths import configure_compilation_cache, configure_xla_flags

configure_xla_flags()
configure_compilation_cache()

import jax
import jax.numpy as jnp

from nanovllm_jax.model import jax_chunk_gated_delta_rule
from nanovllm_jax.layers import l2norm
from nanovllm_jax.kernels.cuda_fp32_ffi import (
    gdn_prefill_chunk32_normalized_fp32,
    gdn_prefill_chunk32_v64_normalized_fp32,
)
from nanovllm_jax.kernels.cuda_gdn import (
    gdn_segmented_prefill_chunk32_reference,
    pack_padded_gdn_inputs,
    unpack_segmented_gdn_output,
)

jax.config.update("jax_default_matmul_precision", "highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--lengths", default="64,128,192,256,320,384,448,512")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variants", default="current_jax_chunk32_padded")
    parser.add_argument("--output-json", default="results/gdn_prefill_kernel_baseline_hetero8_64_512x32.json")
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=os.environ.get("NANO_VLLM_JAX_PROFILE", "1") not in {"0", "false", "False", "no", "off"},
    )
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--run-label", default="gdn_prefill_kernel_hetero8_64_512x32")
    parser.add_argument(
        "--enable-one-piece-gdn-probe",
        action="store_true",
        help="Opt into the experimental one-piece Pallas GDN prefill probe. Full hetero8 shapes can compile for minutes.",
    )
    parser.add_argument(
        "--check-segmented-reference-gate",
        action="store_true",
        help="Run the pure-JAX packed segmented ABI correctness gate after timed variants.",
    )
    return parser.parse_args()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()
    return value


def _trace_annotation(name: str):
    annotation = getattr(jax.profiler, "TraceAnnotation", None)
    if annotation is None:
        return nullcontext()
    return annotation(name)


def _make_inputs(args: argparse.Namespace, lengths: tuple[int, ...]):
    if len(lengths) != args.batch_size:
        raise ValueError("--lengths must contain exactly --batch-size entries")
    if any(length < 0 or length > args.seq_len for length in lengths):
        raise ValueError("all --lengths entries must be in [0, seq_len]")
    if args.seq_len % args.chunk_size != 0:
        raise ValueError("this benchmark keeps seq_len divisible by chunk_size for stable shape comparisons")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 6)
    shape_k = (args.batch_size, args.num_heads, args.seq_len, args.key_dim)
    shape_v = (args.batch_size, args.num_heads, args.seq_len, args.value_dim)
    state_shape = (args.batch_size, args.num_heads, args.key_dim, args.value_dim)

    query = jax.random.normal(keys[0], shape_k, dtype=jnp.float32)
    key = jax.random.normal(keys[1], shape_k, dtype=jnp.float32)
    value = jax.random.normal(keys[2], shape_v, dtype=jnp.float32)
    g = jax.random.normal(keys[3], (args.batch_size, args.num_heads, args.seq_len), dtype=jnp.float32) * 0.1
    beta = jax.random.uniform(keys[4], (args.batch_size, args.num_heads, args.seq_len), dtype=jnp.float32)
    initial_state = jax.random.normal(keys[5], state_shape, dtype=jnp.float32) * 0.01

    valid = jnp.arange(args.seq_len, dtype=jnp.int32)[None, :] < jnp.asarray(lengths, dtype=jnp.int32)[:, None]
    query = jnp.where(valid[:, None, :, None], query, 0.0)
    key = jnp.where(valid[:, None, :, None], key, 0.0)
    value = jnp.where(valid[:, None, :, None], value, 0.0)
    g = jnp.where(valid[:, None, :], g, 0.0)
    beta = jnp.where(valid[:, None, :], beta, 0.0)
    return query, key, value, g, beta, initial_state, valid


def _active_chunk_plan(args: argparse.Namespace, lengths: tuple[int, ...]) -> dict[str, Any]:
    n_chunks = args.seq_len // args.chunk_size
    length_array = np.asarray(lengths, dtype=np.int32)
    chunk_ids = np.arange(n_chunks, dtype=np.int32)
    chunk_starts = chunk_ids * int(args.chunk_size)
    chunk_ends = chunk_starts + int(args.chunk_size)
    active_mask = chunk_starts[None, :] < length_array[:, None]
    full_mask = chunk_ends[None, :] <= length_array[:, None]
    partial_mask = active_mask & ~full_mask
    active_rows, active_chunks = np.nonzero(active_mask)
    active_starts = active_chunks.astype(np.int32) * int(args.chunk_size)
    if active_rows.size:
        active_token_counts = np.minimum(
            int(args.chunk_size),
            length_array[active_rows] - active_starts,
        ).astype(np.int32)
    else:
        active_token_counts = np.zeros((0,), dtype=np.int32)
    chunks_per_row = active_mask.sum(axis=1).astype(np.int32)
    row_offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int32), np.cumsum(chunks_per_row, dtype=np.int32)]
    )
    return {
        "n_chunks": int(n_chunks),
        "active_chunk_count": int(active_rows.size),
        "total_chunk_count": int(args.batch_size * n_chunks),
        "inactive_chunk_count": int(args.batch_size * n_chunks - active_rows.size),
        "partial_chunk_count": int(partial_mask.sum()),
        "chunks_per_row": chunks_per_row.tolist(),
        "row_offsets": row_offsets.tolist(),
        "active_rows": active_rows.astype(np.int32).tolist(),
        "active_chunks": active_chunks.astype(np.int32).tolist(),
        "active_starts": active_starts.astype(np.int32).tolist(),
        "active_token_counts": active_token_counts.tolist(),
        "active_mask": active_mask.astype(np.int32).tolist(),
        "partial_mask": partial_mask.astype(np.int32).tolist(),
    }


def _pallas_active_chunk_probe_fn(batch_size: int, n_chunks: int, chunk_size: int) -> Callable:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    def kernel(lengths_ref, out_ref):
        row = pl.program_id(0)
        chunk = pl.program_id(1)
        length = lengths_ref[row]
        out_ref[row, chunk] = ((chunk * chunk_size) < length).astype(jnp.int32)

    def run(query_lens):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size, n_chunks), jnp.int32),
            grid=(batch_size, n_chunks),
            name="gdn_active_chunk_probe",
            compiler_params=plt.CompilerParams(num_warps=1),
        )(query_lens)

    return run


def _pallas_active_input_pack_probe_fn(
    active_chunk_count: int,
    num_heads: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> Callable:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    def kernel(
        query_ref,
        key_ref,
        value_ref,
        g_ref,
        beta_ref,
        active_rows_ref,
        active_chunks_ref,
        key_offsets_ref,
        value_offsets_ref,
        query_out_ref,
        key_out_ref,
        value_out_ref,
        g_out_ref,
        beta_out_ref,
    ):
        active = pl.program_id(0)
        head = pl.program_id(1)
        token = pl.program_id(2)
        row = active_rows_ref[active]
        time_index = active_chunks_ref[active] * chunk_size + token
        key_offsets = key_offsets_ref[:]
        value_offsets = value_offsets_ref[:]

        query_out_ref[active, head, token, key_offsets] = query_ref[row, head, time_index, key_offsets]
        key_out_ref[active, head, token, key_offsets] = key_ref[row, head, time_index, key_offsets]
        value_out_ref[active, head, token, value_offsets] = value_ref[row, head, time_index, value_offsets]
        g_out_ref[active, head, token] = g_ref[row, head, time_index]
        beta_out_ref[active, head, token] = beta_ref[row, head, time_index]

    def run(query, key, value, g, beta, active_rows, active_chunks, key_offsets, value_offsets):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size, key_dim), query.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size, key_dim), key.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size, value_dim), value.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size), g.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size), beta.dtype),
            ),
            grid=(active_chunk_count, num_heads, chunk_size),
            name="gdn_active_input_pack_probe",
            compiler_params=plt.CompilerParams(num_warps=1),
        )(query, key, value, g, beta, active_rows, active_chunks, key_offsets, value_offsets)

    return run


def _pallas_active_chunk_local_math_probe_fn(
    active_chunk_count: int,
    num_heads: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> Callable:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    def kernel(
        key_ref,
        value_ref,
        g_ref,
        beta_ref,
        active_rows_ref,
        active_chunks_ref,
        key_offsets_ref,
        value_offsets_ref,
        token_offsets_ref,
        value_transformed_ref,
        k_cumdecay_ref,
        g_cumsum_ref,
        attn_ref,
    ):
        active = pl.program_id(0)
        head = pl.program_id(1)
        row = active_rows_ref[active]
        start = active_chunks_ref[active] * chunk_size
        token_offsets = token_offsets_ref[:]
        key_offsets = key_offsets_ref[:]
        value_offsets = value_offsets_ref[:]

        key_block = key_ref[row, head, start + token_offsets[:, None], key_offsets[None, :]]
        value_block = value_ref[row, head, start + token_offsets[:, None], value_offsets[None, :]]
        g_block = g_ref[row, head, start + token_offsets]
        beta_block = beta_ref[row, head, start + token_offsets]

        g_cumsum = jnp.cumsum(g_block, axis=0)
        lower_mask = token_offsets[:, None] >= token_offsets[None, :]
        strict_lower_mask = token_offsets[:, None] > token_offsets[None, :]
        decay_mask = jnp.exp(g_cumsum[:, None] - g_cumsum[None, :])
        decay_mask = jnp.where(lower_mask, decay_mask, 0.0)

        k_beta = key_block * beta_block[:, None]
        kkt = pl.dot(k_beta, key_block, trans_b=True, allow_tf32=False)
        attn = jnp.where(strict_lower_mask, -(kkt * decay_mask), 0.0)

        for row_index in range(1, chunk_size):
            valid_cols = token_offsets < row_index
            selected_row = token_offsets[:, None] == row_index
            row_values = jnp.sum(jnp.where(selected_row, attn, 0.0), axis=0)
            row_values = jnp.where(valid_cols, row_values, 0.0)
            submatrix = jnp.where(valid_cols[:, None] & valid_cols[None, :], attn, 0.0)
            contribution = jnp.sum(row_values[:, None] * submatrix, axis=0)
            updated_row = jnp.where(valid_cols, row_values + contribution, 0.0)
            attn = jnp.where(selected_row, updated_row[None, :], attn)

        attn_with_identity = attn + (token_offsets[:, None] == token_offsets[None, :]).astype(jnp.float32)
        value_transformed = pl.dot(attn_with_identity, value_block * beta_block[:, None], allow_tf32=False)
        k_cumdecay = pl.dot(
            attn_with_identity,
            k_beta * jnp.exp(g_cumsum)[:, None],
            allow_tf32=False,
        )

        value_transformed_ref[active, head, token_offsets[:, None], value_offsets[None, :]] = value_transformed
        k_cumdecay_ref[active, head, token_offsets[:, None], key_offsets[None, :]] = k_cumdecay
        g_cumsum_ref[active, head, token_offsets] = g_cumsum
        attn_ref[active, head, token_offsets[:, None], token_offsets[None, :]] = attn_with_identity

    def run(
        key,
        value,
        g,
        beta,
        active_rows,
        active_chunks,
        key_offsets,
        value_offsets,
        token_offsets,
    ):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size, value_dim), value.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size, key_dim), key.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size), g.dtype),
                jax.ShapeDtypeStruct((active_chunk_count, num_heads, chunk_size, chunk_size), jnp.float32),
            ),
            grid=(active_chunk_count, num_heads),
            name="gdn_active_chunk_local_math_chunk32_probe",
            compiler_params=plt.CompilerParams(num_warps=4),
        )(key, value, g, beta, active_rows, active_chunks, key_offsets, value_offsets, token_offsets)

    return run


def _pallas_rectangular_chunk_local_math_probe_fn(
    batch_size: int,
    num_heads: int,
    n_chunks: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> Callable:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    def kernel(
        key_ref,
        value_ref,
        g_ref,
        beta_ref,
        key_offsets_ref,
        value_offsets_ref,
        token_offsets_ref,
        value_transformed_ref,
        k_cumdecay_ref,
        g_cumsum_ref,
        attn_ref,
    ):
        row = pl.program_id(0)
        head = pl.program_id(1)
        chunk = pl.program_id(2)
        start = chunk * chunk_size
        token_offsets = token_offsets_ref[:]
        key_offsets = key_offsets_ref[:]
        value_offsets = value_offsets_ref[:]

        key_block = key_ref[row, head, start + token_offsets[:, None], key_offsets[None, :]]
        value_block = value_ref[row, head, start + token_offsets[:, None], value_offsets[None, :]]
        g_block = g_ref[row, head, start + token_offsets]
        beta_block = beta_ref[row, head, start + token_offsets]

        g_cumsum = jnp.cumsum(g_block, axis=0)
        lower_mask = token_offsets[:, None] >= token_offsets[None, :]
        strict_lower_mask = token_offsets[:, None] > token_offsets[None, :]
        decay_mask = jnp.exp(g_cumsum[:, None] - g_cumsum[None, :])
        decay_mask = jnp.where(lower_mask, decay_mask, 0.0)

        k_beta = key_block * beta_block[:, None]
        kkt = pl.dot(k_beta, key_block, trans_b=True, allow_tf32=False)
        attn = jnp.where(strict_lower_mask, -(kkt * decay_mask), 0.0)

        for row_index in range(1, chunk_size):
            valid_cols = token_offsets < row_index
            selected_row = token_offsets[:, None] == row_index
            row_values = jnp.sum(jnp.where(selected_row, attn, 0.0), axis=0)
            row_values = jnp.where(valid_cols, row_values, 0.0)
            submatrix = jnp.where(valid_cols[:, None] & valid_cols[None, :], attn, 0.0)
            contribution = jnp.sum(row_values[:, None] * submatrix, axis=0)
            updated_row = jnp.where(valid_cols, row_values + contribution, 0.0)
            attn = jnp.where(selected_row, updated_row[None, :], attn)

        attn_with_identity = attn + (token_offsets[:, None] == token_offsets[None, :]).astype(jnp.float32)
        value_transformed = pl.dot(attn_with_identity, value_block * beta_block[:, None], allow_tf32=False)
        k_cumdecay = pl.dot(
            attn_with_identity,
            k_beta * jnp.exp(g_cumsum)[:, None],
            allow_tf32=False,
        )

        value_transformed_ref[row, head, chunk, token_offsets[:, None], value_offsets[None, :]] = value_transformed
        k_cumdecay_ref[row, head, chunk, token_offsets[:, None], key_offsets[None, :]] = k_cumdecay
        g_cumsum_ref[row, head, chunk, token_offsets] = g_cumsum
        attn_ref[row, head, chunk, token_offsets[:, None], token_offsets[None, :]] = attn_with_identity

    def run(key, value, g, beta, key_offsets, value_offsets, token_offsets):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((batch_size, num_heads, n_chunks, chunk_size, value_dim), value.dtype),
                jax.ShapeDtypeStruct((batch_size, num_heads, n_chunks, chunk_size, key_dim), key.dtype),
                jax.ShapeDtypeStruct((batch_size, num_heads, n_chunks, chunk_size), g.dtype),
                jax.ShapeDtypeStruct((batch_size, num_heads, n_chunks, chunk_size, chunk_size), jnp.float32),
            ),
            grid=(batch_size, num_heads, n_chunks),
            name="gdn_rectangular_chunk_local_math_chunk32_probe",
            compiler_params=plt.CompilerParams(num_warps=4),
        )(key, value, g, beta, key_offsets, value_offsets, token_offsets)

    return run


def _pallas_one_piece_gdn_prefill_probe_fn(
    batch_size: int,
    num_heads: int,
    n_chunks: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
    block_v: int,
) -> Callable:
    if value_dim % block_v != 0:
        raise ValueError("value_dim must be divisible by block_v")

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    def kernel(
        query_ref,
        key_ref,
        value_ref,
        g_ref,
        beta_ref,
        initial_state_ref,
        key_offsets_ref,
        value_offsets_ref,
        token_offsets_ref,
        output_ref,
        final_state_ref,
    ):
        row = pl.program_id(0)
        head = pl.program_id(1)
        value_block = pl.program_id(2)
        key_offsets = key_offsets_ref[:]
        value_offsets = value_block * block_v + value_offsets_ref[:]
        token_offsets = token_offsets_ref[:]
        lower_mask = token_offsets[:, None] >= token_offsets[None, :]
        strict_lower_mask = token_offsets[:, None] > token_offsets[None, :]

        state = initial_state_ref[row, head, key_offsets[:, None], value_offsets[None, :]].astype(jnp.float32)
        for chunk_index in range(n_chunks):
            start = chunk_index * chunk_size
            query_block = query_ref[row, head, start + token_offsets[:, None], key_offsets[None, :]]
            key_block = key_ref[row, head, start + token_offsets[:, None], key_offsets[None, :]]
            value_block_i = value_ref[row, head, start + token_offsets[:, None], value_offsets[None, :]]
            g_block = g_ref[row, head, start + token_offsets]
            beta_block = beta_ref[row, head, start + token_offsets]

            g_cumsum = jnp.cumsum(g_block, axis=0)
            decay_mask = jnp.exp(g_cumsum[:, None] - g_cumsum[None, :])
            decay_mask = jnp.where(lower_mask, decay_mask, 0.0)

            k_beta = key_block * beta_block[:, None]
            kkt = pl.dot(k_beta, key_block, trans_b=True, allow_tf32=False)
            local_attn = jnp.where(strict_lower_mask, -(kkt * decay_mask), 0.0)

            for row_index in range(1, chunk_size):
                valid_cols = token_offsets < row_index
                selected_row = token_offsets[:, None] == row_index
                row_values = jnp.sum(jnp.where(selected_row, local_attn, 0.0), axis=0)
                row_values = jnp.where(valid_cols, row_values, 0.0)
                submatrix = jnp.where(valid_cols[:, None] & valid_cols[None, :], local_attn, 0.0)
                contribution = jnp.sum(row_values[:, None] * submatrix, axis=0)
                updated_row = jnp.where(valid_cols, row_values + contribution, 0.0)
                local_attn = jnp.where(selected_row, updated_row[None, :], local_attn)

            local_attn = local_attn + (token_offsets[:, None] == token_offsets[None, :]).astype(jnp.float32)
            value_transformed = pl.dot(local_attn, value_block_i * beta_block[:, None], allow_tf32=False)
            k_cumdecay = pl.dot(
                local_attn,
                k_beta * jnp.exp(g_cumsum)[:, None],
                allow_tf32=False,
            )

            query_attn = pl.dot(query_block, key_block, trans_b=True, allow_tf32=False) * decay_mask
            query_attn = jnp.where(lower_mask, query_attn, 0.0)
            v_prime = pl.dot(k_cumdecay, state, allow_tf32=False)
            v_new = value_transformed - v_prime
            attn_inter = pl.dot(query_block * jnp.exp(g_cumsum)[:, None], state, allow_tf32=False)
            attn_v_new = pl.dot(query_attn, v_new, allow_tf32=False)
            output_i = attn_inter + attn_v_new

            g_last = jnp.sum(jnp.where(token_offsets == (chunk_size - 1), g_cumsum, 0.0), axis=0)
            g_last_minus_g = g_last - g_cumsum
            k_weighted = key_block * jnp.exp(g_last_minus_g)[:, None]
            state_update = pl.dot(k_weighted, v_new, trans_a=True, allow_tf32=False)
            state = state * jnp.exp(g_last) + state_update
            output_ref[row, head, start + token_offsets[:, None], value_offsets[None, :]] = output_i

        final_state_ref[row, head, key_offsets[:, None], value_offsets[None, :]] = state

    def run(query, key, value, g, beta, initial_state, key_offsets, value_offsets, token_offsets):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((batch_size, num_heads, n_chunks * chunk_size, value_dim), query.dtype),
                jax.ShapeDtypeStruct((batch_size, num_heads, key_dim, value_dim), jnp.float32),
            ),
            grid=(batch_size, num_heads, value_dim // block_v),
            name=f"gdn_one_piece_gdn_prefill_vblock{block_v}_probe",
            compiler_params=plt.CompilerParams(num_warps=4),
        )(query, key, value, g, beta, initial_state, key_offsets, value_offsets, token_offsets)

    return run


def _active_input_pack_reference(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    active_rows: jnp.ndarray,
    active_chunks: jnp.ndarray,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    heads = jnp.arange(query.shape[1], dtype=jnp.int32)[None, :, None]
    token_offsets = jnp.arange(chunk_size, dtype=jnp.int32)[None, :]
    time_indices = active_chunks[:, None] * int(chunk_size) + token_offsets
    rows = active_rows[:, None, None]
    times = time_indices[:, None, :]
    return (
        query[rows, heads, times, :],
        key[rows, heads, times, :],
        value[rows, heads, times, :],
        g[rows, heads, times],
        beta[rows, heads, times],
    )


def _active_chunk_local_math_reference(
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    active_rows: jnp.ndarray,
    active_chunks: jnp.ndarray,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    heads = jnp.arange(key.shape[1], dtype=jnp.int32)[None, :, None]
    token_offsets = jnp.arange(chunk_size, dtype=jnp.int32)[None, :]
    time_indices = active_chunks[:, None] * int(chunk_size) + token_offsets
    rows = active_rows[:, None, None]
    times = time_indices[:, None, :]

    key_active = key[rows, heads, times, :]
    value_active = value[rows, heads, times, :]
    g_active = g[rows, heads, times]
    beta_active = beta[rows, heads, times]

    g_cumsum = jnp.cumsum(g_active, axis=-1)
    decay_mask = jnp.tril(jnp.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
    kkt = jnp.einsum("ahck,ahjk->ahcj", key_active * beta_active[..., None], key_active)
    attn = -(kkt * decay_mask)
    mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn = jnp.where(mask_upper, 0.0, attn)

    for row_index in range(1, chunk_size):
        valid_cols = jnp.arange(chunk_size) < row_index
        row_values = attn[..., row_index, :] * valid_cols
        submatrix = attn * valid_cols[None, None, :, None] * valid_cols[None, None, None, :]
        contribution = jnp.einsum("ahj,ahjk->ahk", row_values, submatrix)
        updated_row = (row_values + contribution) * valid_cols
        attn = attn.at[..., row_index, :].set(updated_row)

    attn_with_identity = attn + jnp.eye(chunk_size, dtype=jnp.float32)
    value_transformed = jnp.einsum("ahct,ahtv->ahcv", attn_with_identity, value_active * beta_active[..., None])
    k_cumdecay = jnp.einsum(
        "ahct,ahtk->ahck",
        attn_with_identity,
        key_active * beta_active[..., None] * jnp.exp(g_cumsum)[..., None],
    )
    return value_transformed, k_cumdecay, g_cumsum, attn_with_identity


def _rectangular_local_math_reference(
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, num_heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    n_chunks = seq_len // int(chunk_size)
    key_chunks = key.reshape(batch_size, num_heads, n_chunks, chunk_size, key_dim)
    value_chunks = value.reshape(batch_size, num_heads, n_chunks, chunk_size, value_dim)
    g_chunks = g.reshape(batch_size, num_heads, n_chunks, chunk_size)
    beta_chunks = beta.reshape(batch_size, num_heads, n_chunks, chunk_size)

    g_cumsum = jnp.cumsum(g_chunks, axis=-1)
    decay_mask = jnp.tril(jnp.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
    k_beta_chunks = key_chunks * beta_chunks[..., None]
    kkt = jnp.einsum("bhnck,bhnjk->bhncj", k_beta_chunks, key_chunks)
    attn = -(kkt * decay_mask)
    mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn = jnp.where(mask_upper, 0.0, attn)

    for row_index in range(1, chunk_size):
        valid_cols = jnp.arange(chunk_size) < row_index
        row_values = attn[..., row_index, :] * valid_cols
        submatrix = attn * valid_cols[None, None, None, :, None] * valid_cols[None, None, None, None, :]
        contribution = jnp.einsum("bhnj,bhnjk->bhnk", row_values, submatrix)
        updated_row = (row_values + contribution) * valid_cols
        attn = attn.at[..., row_index, :].set(updated_row)

    attn_with_identity = attn + jnp.eye(chunk_size, dtype=jnp.float32)
    value_transformed = jnp.einsum("bhnct,bhntv->bhncv", attn_with_identity, value_chunks * beta_chunks[..., None])
    k_cumdecay = jnp.einsum(
        "bhnct,bhntk->bhnck",
        attn_with_identity,
        k_beta_chunks * jnp.exp(g_cumsum)[..., None],
    )
    return value_transformed, k_cumdecay, g_cumsum, attn_with_identity


def _scatter_active_local_math_to_chunks(
    value_transformed: jnp.ndarray,
    k_cumdecay: jnp.ndarray,
    g_cumsum: jnp.ndarray,
    active_rows: jnp.ndarray,
    active_chunks: jnp.ndarray,
    *,
    batch_size: int,
    num_heads: int,
    n_chunks: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    heads = jnp.arange(num_heads, dtype=jnp.int32)[None, :]
    rows = active_rows[:, None]
    chunks = active_chunks[:, None]
    value_transformed_chunks = jnp.zeros(
        (batch_size, num_heads, n_chunks, chunk_size, value_dim),
        dtype=value_transformed.dtype,
    ).at[rows, heads, chunks, :, :].set(value_transformed)
    k_cumdecay_chunks = jnp.zeros(
        (batch_size, num_heads, n_chunks, chunk_size, key_dim),
        dtype=k_cumdecay.dtype,
    ).at[rows, heads, chunks, :, :].set(k_cumdecay)
    g_cumsum_chunks = jnp.zeros(
        (batch_size, num_heads, n_chunks, chunk_size),
        dtype=g_cumsum.dtype,
    ).at[rows, heads, chunks, :].set(g_cumsum)
    return value_transformed_chunks, k_cumdecay_chunks, g_cumsum_chunks


def _rectangular_output_state_reconstruction_probe_fn(
    batch_size: int,
    num_heads: int,
    n_chunks: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> Callable:
    mask_strict_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)

    def run(
        normalized_query_scaled,
        normalized_key,
        initial_state,
        value_transformed_chunks,
        k_cumdecay_chunks,
        g_cumsum_chunks,
    ):
        query_chunks = normalized_query_scaled.reshape(batch_size, num_heads, n_chunks, chunk_size, key_dim)
        key_chunks = normalized_key.reshape(batch_size, num_heads, n_chunks, chunk_size, key_dim)
        decay_mask = jnp.tril(jnp.exp(g_cumsum_chunks[..., :, None] - g_cumsum_chunks[..., None, :]))
        state = initial_state.astype(jnp.float32)

        def process_chunk(carry, chunk_index):
            state = carry
            q_i = query_chunks[:, :, chunk_index]
            k_i = key_chunks[:, :, chunk_index]
            v_i = value_transformed_chunks[:, :, chunk_index]
            decay_mask_i = decay_mask[:, :, chunk_index]
            k_cumdecay_i = k_cumdecay_chunks[:, :, chunk_index]
            g_cumsum_i = g_cumsum_chunks[:, :, chunk_index]

            attn_i = jnp.einsum("bhck,bhdk->bhcd", q_i, k_i) * decay_mask_i
            attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)
            v_prime = jnp.einsum("bhck,bhkv->bhcv", k_cumdecay_i, state)
            v_new = v_i - v_prime
            attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_i * jnp.exp(g_cumsum_i)[..., None], state)
            attn_v_new = jnp.einsum("bhcd,bhdv->bhcv", attn_i, v_new)
            core_attn_out_i = attn_inter + attn_v_new

            g_last_minus_g = g_cumsum_i[..., -1, None] - g_cumsum_i
            k_weighted = k_i * jnp.exp(g_last_minus_g)[..., None]
            state_update = jnp.einsum("bhck,bhcv->bhkv", k_weighted, v_new)
            state = state * jnp.exp(g_cumsum_i[..., -1, None, None]) + state_update
            return state, core_attn_out_i

        final_state, core_attn_out_chunks = jax.lax.scan(process_chunk, state, jnp.arange(n_chunks))
        core_attn_out = core_attn_out_chunks.transpose(1, 2, 0, 3, 4).reshape(
            batch_size,
            num_heads,
            n_chunks * chunk_size,
            value_dim,
        )
        return core_attn_out.astype(normalized_query_scaled.dtype), final_state

    return run


def _active_output_state_reconstruction_probe_fn(
    batch_size: int,
    num_heads: int,
    n_chunks: int,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> Callable:
    mask_strict_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)

    def run(
        normalized_query_scaled,
        normalized_key,
        initial_state,
        active_rows,
        active_chunks,
        value_transformed,
        k_cumdecay,
        g_cumsum,
    ):
        query_chunks = normalized_query_scaled.reshape(batch_size, num_heads, n_chunks, chunk_size, key_dim)
        key_chunks = normalized_key.reshape(batch_size, num_heads, n_chunks, chunk_size, key_dim)
        value_transformed_chunks, k_cumdecay_chunks, g_cumsum_chunks = _scatter_active_local_math_to_chunks(
            value_transformed,
            k_cumdecay,
            g_cumsum,
            active_rows,
            active_chunks,
            batch_size=batch_size,
            num_heads=num_heads,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        decay_mask = jnp.tril(jnp.exp(g_cumsum_chunks[..., :, None] - g_cumsum_chunks[..., None, :]))
        state = initial_state.astype(jnp.float32)

        def process_chunk(carry, chunk_index):
            state = carry
            q_i = query_chunks[:, :, chunk_index]
            k_i = key_chunks[:, :, chunk_index]
            v_i = value_transformed_chunks[:, :, chunk_index]
            decay_mask_i = decay_mask[:, :, chunk_index]
            k_cumdecay_i = k_cumdecay_chunks[:, :, chunk_index]
            g_cumsum_i = g_cumsum_chunks[:, :, chunk_index]

            attn_i = jnp.einsum("bhck,bhdk->bhcd", q_i, k_i) * decay_mask_i
            attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)
            v_prime = jnp.einsum("bhck,bhkv->bhcv", k_cumdecay_i, state)
            v_new = v_i - v_prime
            attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_i * jnp.exp(g_cumsum_i)[..., None], state)
            attn_v_new = jnp.einsum("bhcd,bhdv->bhcv", attn_i, v_new)
            core_attn_out_i = attn_inter + attn_v_new

            g_last_minus_g = g_cumsum_i[..., -1, None] - g_cumsum_i
            k_weighted = k_i * jnp.exp(g_last_minus_g)[..., None]
            state_update = jnp.einsum("bhck,bhcv->bhkv", k_weighted, v_new)
            state = state * jnp.exp(g_cumsum_i[..., -1, None, None]) + state_update
            return state, core_attn_out_i

        final_state, core_attn_out_chunks = jax.lax.scan(
            process_chunk,
            state,
            jnp.arange(n_chunks),
        )
        core_attn_out = core_attn_out_chunks.transpose(1, 2, 0, 3, 4).reshape(
            batch_size,
            num_heads,
            n_chunks * chunk_size,
            value_dim,
        )
        return core_attn_out.astype(normalized_query_scaled.dtype), final_state

    return run


def _compare_pack_outputs(
    reference: tuple[jnp.ndarray, ...],
    candidate: tuple[jnp.ndarray, ...],
) -> dict[str, float]:
    names = ("query", "key", "value", "g", "beta")
    comparisons: dict[str, float] = {}
    max_abs_values: list[float] = []
    for name, ref, out in zip(names, reference, candidate):
        diff = out.astype(jnp.float32) - ref.astype(jnp.float32)
        element_count = int(np.prod(out.shape))
        if element_count == 0:
            max_abs = 0.0
            mse = 0.0
        else:
            max_abs = float(jnp.max(jnp.abs(diff)))
            mse = float(jnp.mean(jnp.square(diff)))
        comparisons[f"{name}_max_abs"] = max_abs
        comparisons[f"{name}_mse"] = mse
        max_abs_values.append(max_abs)
    comparisons["max_abs"] = max(max_abs_values) if max_abs_values else 0.0
    return comparisons


def _compare_local_math_outputs(
    reference: tuple[jnp.ndarray, ...],
    candidate: tuple[jnp.ndarray, ...],
) -> dict[str, float | bool]:
    names = ("value_transformed", "k_cumdecay", "g_cumsum", "attn")
    comparisons: dict[str, float | bool] = {}
    max_abs_values: list[float] = []
    for name, ref, out in zip(names, reference, candidate):
        diff = out.astype(jnp.float32) - ref.astype(jnp.float32)
        max_abs = float(jnp.max(jnp.abs(diff))) if int(np.prod(out.shape)) else 0.0
        mse = float(jnp.mean(jnp.square(diff))) if int(np.prod(out.shape)) else 0.0
        comparisons[f"{name}_max_abs"] = max_abs
        comparisons[f"{name}_mse"] = mse
        max_abs_values.append(max_abs)
    comparisons["max_abs"] = max(max_abs_values) if max_abs_values else 0.0
    comparisons["passes_1e_5_gate"] = bool(comparisons["max_abs"] <= 1e-5)
    return comparisons


def _padded_chunked_fn(chunk_size: int) -> Callable:
    def run(query, key, value, g, beta, initial_state):
        return jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    return run


def _cuda_fp32_one_piece_chunk32_fn(
    args: argparse.Namespace,
    lengths: tuple[int, ...],
    prefill_fn: Callable = gdn_prefill_chunk32_normalized_fp32,
) -> Callable:
    seq_lens = jnp.asarray(lengths, dtype=jnp.int32)
    scale = 1.0 / jnp.sqrt(args.key_dim)

    def run(query, key, value, g, beta, initial_state):
        query_norm_scaled = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6) * scale
        key_norm = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)
        return prefill_fn(
            query_norm_scaled,
            key_norm,
            value.astype(jnp.float32),
            g.astype(jnp.float32),
            beta.astype(jnp.float32),
            seq_lens,
            initial_state.astype(jnp.float32),
        )

    return run


def _variant_fns(args: argparse.Namespace, lengths: tuple[int, ...]) -> dict[str, Callable]:
    available = {
        "current_jax_chunk32_padded": _padded_chunked_fn(args.chunk_size),
        "cuda_fp32_one_piece_chunk32": _cuda_fp32_one_piece_chunk32_fn(args, lengths),
        "cuda_fp32_one_piece_chunk32_v64": _cuda_fp32_one_piece_chunk32_fn(
            args,
            lengths,
            gdn_prefill_chunk32_v64_normalized_fp32,
        ),
    }
    requested = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(available)}")
    return {name: available[name] for name in requested}


def _compare_outputs(reference: tuple[jnp.ndarray, jnp.ndarray], candidate: tuple[jnp.ndarray, jnp.ndarray], valid: jnp.ndarray) -> dict[str, float]:
    ref_out, ref_state = reference
    out, state = candidate
    output_diff = (out.astype(jnp.float32) - ref_out.astype(jnp.float32))
    state_diff = state.astype(jnp.float32) - ref_state.astype(jnp.float32)
    valid_output_diff = jnp.where(valid[:, None, :, None], output_diff, 0.0)
    valid_count = max(int(valid.sum()) * int(out.shape[1]) * int(out.shape[-1]), 1)
    return {
        "output_max_abs": float(jnp.max(jnp.abs(output_diff))),
        "output_mse": float(jnp.mean(jnp.square(output_diff))),
        "valid_output_max_abs": float(jnp.max(jnp.abs(valid_output_diff))),
        "valid_output_mse": float(jnp.sum(jnp.square(valid_output_diff)) / valid_count),
        "state_max_abs": float(jnp.max(jnp.abs(state_diff))),
        "state_mse": float(jnp.mean(jnp.square(state_diff))),
    }


SEGMENTED_GDN_STANDALONE_GATE_THRESHOLD = 1e-5

SEGMENTED_GDN_FULL_MODEL_GATE = {
    "name": "real_weight_full_model_token_logit_parity",
    "reference_artifact": "results/qwen08_hf_bf16w_fp32act_long_decode_top5_500.npz",
    "required_checks": {
        "exact_generated_token_match": True,
        "top1_exact_matches": 500,
        "ordered_top5_exact_matches": 500,
        "top5_set_exact_matches": 500,
        "max_hf_topk_id_logit_diff_lte": 2e-5,
    },
    "required_command": "benchmark_long_decode_top5.py --max-new-tokens 500",
}


def _passes_segmented_standalone_gate(
    comparison: dict[str, Any] | None,
    *,
    threshold: float = SEGMENTED_GDN_STANDALONE_GATE_THRESHOLD,
) -> bool:
    if not comparison:
        return False
    return bool(
        comparison.get("output_max_abs", float("inf")) <= threshold
        and comparison.get("valid_output_max_abs", float("inf")) <= threshold
        and comparison.get("state_max_abs", float("inf")) <= threshold
    )


def _segmented_reference_policy(gate: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether the planned packed GDN ABI is eligible for CUDA math."""

    threshold = SEGMENTED_GDN_STANDALONE_GATE_THRESHOLD
    if not gate.get("enabled"):
        return {
            "status": "not_checked",
            "cuda_math_allowed": False,
            "serving_routing_allowed": False,
            "threshold": threshold,
            "reason": "segmented reference gate was not requested",
            "required_next_step": "run with --check-segmented-reference-gate before implementing segmented CUDA math",
        }
    if gate.get("error"):
        return {
            "status": "gate_error",
            "cuda_math_allowed": False,
            "serving_routing_allowed": False,
            "threshold": threshold,
            "reason": gate["error"],
            "required_next_step": "fix the standalone segmented reference gate before implementing segmented CUDA math",
        }

    strict_pass = _passes_segmented_standalone_gate(gate.get("comparison"))
    row_padded = gate.get("row_padded_to_seq_len") or {}
    row_padded_pass = _passes_segmented_standalone_gate(row_padded.get("comparison"))
    if strict_pass:
        return {
            "status": "eligible_for_segmented_cuda_math",
            "cuda_math_allowed": True,
            "serving_routing_allowed": False,
            "threshold": threshold,
            "reason": "packed segmented ABI passed the standalone padded-chunk32 output/state gate",
            "required_next_step": "implement CUDA math as benchmark-only, then require integrated exact-token and latency gates before serving routing",
        }

    diagnosis = "packed true-token ABI misses the standalone padded-chunk32 gate"
    if row_padded:
        diagnosis = (
            "row-wise decomposition changes enough FP32 accumulation to miss the "
            "standalone padded-chunk32 gate"
            if not row_padded_pass
            else "actual-length packing misses the gate but row-padded diagnostic passes"
        )
    return {
        "status": "blocked_on_correctness_policy",
        "cuda_math_allowed": False,
        "serving_routing_allowed": False,
        "requires_design_decision": True,
        "threshold": threshold,
        "reason": diagnosis,
        "allowed_without_design_change": (
            "a backend design that preserves the current batched rectangular padded-chunk32 accumulation contract"
        ),
        "design_change_option": (
            "accept a true-token packed ABI only after an explicit full-model real-weight token/logit parity gate"
        ),
        "design_change_required_gate": SEGMENTED_GDN_FULL_MODEL_GATE,
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _device_summary() -> dict[str, Any]:
    devices = jax.devices()
    first = devices[0] if devices else None
    return {
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "device": str(first) if first is not None else None,
        "platform": getattr(first, "platform", None) if first is not None else None,
        "device_kind": getattr(first, "device_kind", None) if first is not None else None,
    }


def _profile_counters(profile_path: Path, profiled_iterations: int, variants: list[str]) -> dict[str, Any]:
    traces = sorted(profile_path.glob("plugins/profile/*/*.trace.json.gz"))
    if not traces:
        return {
            "trace_json_gz": None,
            "profiled_iterations": profiled_iterations,
            "ranges": {},
        }
    trace_path = traces[-1]
    needles = [
        *(f"gdn_prefill/{variant}" for variant in variants),
        "PjRtCApiLoadedExecutable::Execute",
        "jit_compiled:XLA GPU module",
        "command_buffer::execute",
        "command_buffer::update",
        "input_reduce_fusion",
        "loop_dynamic_update_slice_fusion",
        "loop_multiply_fusion",
        "MemcpyD2D",
        "Thunks::Initialize",
        "gdn_prefill/pallas_active_chunk_probe",
        "gdn_active_chunk_probe",
        "gdn_prefill/pallas_active_input_pack_probe",
        "gdn_active_input_pack_probe",
        "gdn_prefill/pallas_active_chunk_local_math_probe",
        "gdn_active_chunk_local_math_chunk32_probe",
        "gdn_prefill/pallas_active_output_state_reconstruction_probe",
        "gdn_prefill/pallas_rectangular_chunk_local_math_probe",
        "gdn_rectangular_chunk_local_math_chunk32_probe",
        "gdn_prefill/rectangular_local_split_reconstruction_probe",
        "gdn_prefill/pallas_rectangular_output_state_reconstruction_probe",
        "gdn_prefill/pallas_one_piece_gdn_prefill_probe",
        "gdn_one_piece_gdn_prefill_vblock64_probe",
        "while",
        "fusion",
        "transpose",
        "gather",
    ]
    aggregate: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    try:
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            events = json.load(handle).get("traceEvents", [])
    except Exception as exc:
        return {
            "trace_json_gz": str(trace_path),
            "profiled_iterations": profiled_iterations,
            "error": f"{type(exc).__name__}: {exc}",
            "ranges": {},
        }
    for event in events:
        duration = event.get("dur")
        name = event.get("name", "")
        if duration is None:
            continue
        duration_ms = float(duration) / 1000.0
        for needle in needles:
            if needle in name:
                aggregate[needle][0] += duration_ms
                aggregate[needle][1] += 1
    denominator = max(int(profiled_iterations), 1)
    return {
        "trace_json_gz": str(trace_path),
        "profiled_iterations": profiled_iterations,
        "ranges": {
            needle: {
                "total_ms": total,
                "count": count,
                "ms_per_iter": total / denominator,
                "count_per_iter": count / denominator,
            }
            for needle, (total, count) in sorted(aggregate.items())
        },
    }


def run_benchmark(args: argparse.Namespace, recorder: RunRecorder) -> dict[str, Any]:
    lengths = _parse_ints(args.lengths)
    query, key, value, g, beta, initial_state, valid = _make_inputs(args, lengths)
    inputs = (query, key, value, g, beta, initial_state)
    variants = _variant_fns(args, lengths)
    true_tokens = int(sum(lengths))
    rectangular_tokens = int(args.batch_size * args.seq_len)
    active_chunks = int(sum((length + args.chunk_size - 1) // args.chunk_size for length in lengths))
    total_chunks = int(args.batch_size * (args.seq_len // args.chunk_size))
    active_chunk_plan = _active_chunk_plan(args, lengths)
    query_lens = jnp.asarray(lengths, dtype=jnp.int32)
    active_rows = jnp.asarray(active_chunk_plan["active_rows"], dtype=jnp.int32)
    active_chunk_indices = jnp.asarray(active_chunk_plan["active_chunks"], dtype=jnp.int32)
    key_offsets = jnp.arange(args.key_dim, dtype=jnp.int32)
    value_offsets = jnp.arange(args.value_dim, dtype=jnp.int32)
    token_offsets = jnp.arange(args.chunk_size, dtype=jnp.int32)
    normalized_query_scaled = l2norm(query.astype(jnp.float32), axis=-1, eps=1e-6) * (1.0 / jnp.sqrt(args.key_dim))
    normalized_key = l2norm(key.astype(jnp.float32), axis=-1, eps=1e-6)

    variant_results: dict[str, dict[str, Any]] = {}
    outputs: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
    pallas_feasibility: dict[str, Any] = {
        "attempted": False,
        "lowering_ok": None,
        "error": None,
        "custom_call_count": None,
        "active_input_pack_probe": {
            "attempted": False,
            "lowering_ok": None,
            "error": None,
            "custom_call_count": None,
        },
        "active_chunk_local_math_probe": {
            "attempted": False,
            "lowering_ok": None,
            "error": None,
            "custom_call_count": None,
        },
        "active_output_state_reconstruction_probe": {
            "attempted": False,
            "lowering_ok": None,
            "error": None,
        },
        "rectangular_chunk_local_math_probe": {
            "attempted": False,
            "lowering_ok": None,
            "error": None,
            "custom_call_count": None,
        },
        "rectangular_local_split_reconstruction_probe": {
            "attempted": False,
            "lowering_ok": None,
            "error": None,
        },
        "one_piece_gdn_prefill_probe": {
            "enabled": bool(args.enable_one_piece_gdn_probe),
            "attempted": False,
            "lowering_ok": None,
            "error": None,
            "custom_call_count": None,
            "block_v": 64,
        },
    }
    pallas_pack_last_output = None
    pallas_pack_reference = None
    pallas_local_math_last_output = None
    pallas_local_math_reference = None
    reconstruction_reference_output = None
    reconstruction_pallas_last_output = None
    rectangular_local_math_reference = None
    pallas_rectangular_local_math_last_output = None
    rectangular_reference_reconstruction_output = None
    rectangular_pallas_reconstruction_output = None
    one_piece_gdn_prefill_probe = None
    one_piece_gdn_prefill_last_output = None

    # Compile before profiling. The trace should focus on warmed kernel behavior.
    compiled_variants: list[tuple[str, Callable, Any, float]] = []
    for name, fn in variants.items():
        started = time.perf_counter()
        compiled = jax.jit(fn).lower(*inputs).compile()
        compiled_variants.append((name, fn, compiled, time.perf_counter() - started))

    pallas_probe = None
    pallas_pack_probe = None
    pallas_local_math_probe = None
    reconstruction_probe = None
    pallas_rectangular_local_math_probe = None
    rectangular_reconstruction_probe = None
    if jax.default_backend() == "gpu":
        pallas_feasibility["attempted"] = True
        try:
            started = time.perf_counter()
            probe_fn = _pallas_active_chunk_probe_fn(
                int(args.batch_size),
                int(args.seq_len // args.chunk_size),
                int(args.chunk_size),
            )
            pallas_probe = jax.jit(probe_fn).lower(query_lens).compile()
            pallas_feasibility["compile_seconds"] = time.perf_counter() - started
            pallas_feasibility["lowering_ok"] = True
        except BaseException as exc:
            pallas_feasibility["lowering_ok"] = False
            pallas_feasibility["error"] = f"{type(exc).__name__}: {exc}"

        pack_info = pallas_feasibility["active_input_pack_probe"]
        if int(active_chunk_plan["active_chunk_count"]) > 0:
            pack_info["attempted"] = True
            try:
                started = time.perf_counter()
                pack_fn = _pallas_active_input_pack_probe_fn(
                    int(active_chunk_plan["active_chunk_count"]),
                    int(args.num_heads),
                    int(args.chunk_size),
                    int(args.key_dim),
                    int(args.value_dim),
                )
                pallas_pack_probe = jax.jit(pack_fn).lower(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    active_rows,
                    active_chunk_indices,
                    key_offsets,
                    value_offsets,
                ).compile()
                pack_info["compile_seconds"] = time.perf_counter() - started
                pack_info["lowering_ok"] = True
                pallas_pack_reference = _block_until_ready(
                    _active_input_pack_reference(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        active_rows,
                        active_chunk_indices,
                        int(args.chunk_size),
                    )
                )
            except BaseException as exc:
                pallas_pack_probe = None
                pallas_pack_reference = None
                pack_info["lowering_ok"] = False
                pack_info["error"] = f"{type(exc).__name__}: {exc}"

            local_math_info = pallas_feasibility["active_chunk_local_math_probe"]
            local_math_info["attempted"] = True
            try:
                started = time.perf_counter()
                local_math_fn = _pallas_active_chunk_local_math_probe_fn(
                    int(active_chunk_plan["active_chunk_count"]),
                    int(args.num_heads),
                    int(args.chunk_size),
                    int(args.key_dim),
                    int(args.value_dim),
                )
                pallas_local_math_probe = jax.jit(local_math_fn).lower(
                    normalized_key,
                    value,
                    g,
                    beta,
                    active_rows,
                    active_chunk_indices,
                    key_offsets,
                    value_offsets,
                    token_offsets,
                ).compile()
                local_math_info["compile_seconds"] = time.perf_counter() - started
                local_math_info["lowering_ok"] = True
                pallas_local_math_reference = _block_until_ready(
                    _active_chunk_local_math_reference(
                        normalized_key,
                        value,
                        g,
                        beta,
                        active_rows,
                        active_chunk_indices,
                        int(args.chunk_size),
                    )
                )
                reconstruction_info = pallas_feasibility["active_output_state_reconstruction_probe"]
                reconstruction_info["attempted"] = True
                reconstruction_started = time.perf_counter()
                reconstruction_fn = _active_output_state_reconstruction_probe_fn(
                    int(args.batch_size),
                    int(args.num_heads),
                    int(args.seq_len // args.chunk_size),
                    int(args.chunk_size),
                    int(args.key_dim),
                    int(args.value_dim),
                )
                reconstruction_probe = jax.jit(reconstruction_fn).lower(
                    normalized_query_scaled,
                    normalized_key,
                    initial_state,
                    active_rows,
                    active_chunk_indices,
                    pallas_local_math_reference[0],
                    pallas_local_math_reference[1],
                    pallas_local_math_reference[2],
                ).compile()
                reconstruction_info["compile_seconds"] = time.perf_counter() - reconstruction_started
                reconstruction_info["lowering_ok"] = True
                reconstruction_reference_output = _block_until_ready(
                    reconstruction_probe(
                        normalized_query_scaled,
                        normalized_key,
                        initial_state,
                        active_rows,
                        active_chunk_indices,
                        pallas_local_math_reference[0],
                        pallas_local_math_reference[1],
                        pallas_local_math_reference[2],
                    )
                )
            except BaseException as exc:
                pallas_local_math_probe = None
                pallas_local_math_reference = None
                reconstruction_probe = None
                reconstruction_reference_output = None
                local_math_info["lowering_ok"] = False
                local_math_info["error"] = f"{type(exc).__name__}: {exc}"
                reconstruction_info = pallas_feasibility["active_output_state_reconstruction_probe"]
                if reconstruction_info.get("attempted"):
                    reconstruction_info["lowering_ok"] = False
                    reconstruction_info["error"] = f"{type(exc).__name__}: {exc}"

            rectangular_info = pallas_feasibility["rectangular_chunk_local_math_probe"]
            rectangular_reconstruction_info = pallas_feasibility["rectangular_local_split_reconstruction_probe"]
            rectangular_info["attempted"] = True
            rectangular_reconstruction_info["attempted"] = True
            try:
                rectangular_local_math_reference = _block_until_ready(
                    _rectangular_local_math_reference(
                        normalized_key,
                        value,
                        g,
                        beta,
                        int(args.chunk_size),
                    )
                )
                rectangular_started = time.perf_counter()
                rectangular_fn = _pallas_rectangular_chunk_local_math_probe_fn(
                    int(args.batch_size),
                    int(args.num_heads),
                    int(args.seq_len // args.chunk_size),
                    int(args.chunk_size),
                    int(args.key_dim),
                    int(args.value_dim),
                )
                pallas_rectangular_local_math_probe = jax.jit(rectangular_fn).lower(
                    normalized_key,
                    value,
                    g,
                    beta,
                    key_offsets,
                    value_offsets,
                    token_offsets,
                ).compile()
                rectangular_info["compile_seconds"] = time.perf_counter() - rectangular_started
                rectangular_info["lowering_ok"] = True

                rectangular_reconstruction_started = time.perf_counter()
                rectangular_reconstruction_fn = _rectangular_output_state_reconstruction_probe_fn(
                    int(args.batch_size),
                    int(args.num_heads),
                    int(args.seq_len // args.chunk_size),
                    int(args.chunk_size),
                    int(args.key_dim),
                    int(args.value_dim),
                )
                rectangular_reconstruction_probe = jax.jit(rectangular_reconstruction_fn).lower(
                    normalized_query_scaled,
                    normalized_key,
                    initial_state,
                    rectangular_local_math_reference[0],
                    rectangular_local_math_reference[1],
                    rectangular_local_math_reference[2],
                ).compile()
                rectangular_reconstruction_info["compile_seconds"] = (
                    time.perf_counter() - rectangular_reconstruction_started
                )
                rectangular_reconstruction_info["lowering_ok"] = True
                rectangular_reference_reconstruction_output = _block_until_ready(
                    rectangular_reconstruction_probe(
                        normalized_query_scaled,
                        normalized_key,
                        initial_state,
                        rectangular_local_math_reference[0],
                        rectangular_local_math_reference[1],
                        rectangular_local_math_reference[2],
                    )
                )
            except BaseException as exc:
                pallas_rectangular_local_math_probe = None
                rectangular_reconstruction_probe = None
                rectangular_local_math_reference = None
                rectangular_info["lowering_ok"] = False
                rectangular_info["error"] = f"{type(exc).__name__}: {exc}"
                rectangular_reconstruction_info["lowering_ok"] = False
                rectangular_reconstruction_info["error"] = f"{type(exc).__name__}: {exc}"

            one_piece_info = pallas_feasibility["one_piece_gdn_prefill_probe"]
            if args.enable_one_piece_gdn_probe:
                one_piece_info["attempted"] = True
                try:
                    block_v = int(one_piece_info["block_v"])
                    one_piece_started = time.perf_counter()
                    one_piece_fn = _pallas_one_piece_gdn_prefill_probe_fn(
                        int(args.batch_size),
                        int(args.num_heads),
                        int(args.seq_len // args.chunk_size),
                        int(args.chunk_size),
                        int(args.key_dim),
                        int(args.value_dim),
                        block_v,
                    )
                    one_piece_gdn_prefill_probe = jax.jit(one_piece_fn).lower(
                        normalized_query_scaled,
                        normalized_key,
                        value,
                        g,
                        beta,
                        initial_state,
                        key_offsets,
                        jnp.arange(block_v, dtype=jnp.int32),
                        token_offsets,
                    ).compile()
                    one_piece_info["compile_seconds"] = time.perf_counter() - one_piece_started
                    one_piece_info["lowering_ok"] = True
                except BaseException as exc:
                    one_piece_gdn_prefill_probe = None
                    one_piece_info["lowering_ok"] = False
                    one_piece_info["error"] = f"{type(exc).__name__}: {exc}"

    recorder.start_jax_profile(enabled=args.profile)
    try:
        if pallas_probe is not None:
            with _trace_annotation("gdn_prefill/pallas_active_chunk_probe"):
                started = time.perf_counter()
                pallas_mask = _block_until_ready(pallas_probe(query_lens))
                pallas_feasibility["run_ms"] = 1000.0 * (time.perf_counter() - started)
            pallas_mask_host = np.asarray(pallas_mask)
            expected_mask = np.asarray(active_chunk_plan["active_mask"], dtype=np.int32)
            pallas_feasibility.update(
                {
                    "mask_matches_plan": bool(np.array_equal(pallas_mask_host, expected_mask)),
                    "active_chunk_count": int(pallas_mask_host.sum()),
                    "inactive_chunk_count": int(pallas_mask_host.size - pallas_mask_host.sum()),
                    "total_chunk_count": int(pallas_mask_host.size),
                }
            )
        if pallas_pack_probe is not None:
            pack_info = pallas_feasibility["active_input_pack_probe"]
            warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/pallas_active_input_pack_probe/warmup"):
                    started = time.perf_counter()
                    pallas_pack_last_output = _block_until_ready(
                        pallas_pack_probe(
                            query,
                            key,
                            value,
                            g,
                            beta,
                            active_rows,
                            active_chunk_indices,
                            key_offsets,
                            value_offsets,
                        )
                    )
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/pallas_active_input_pack_probe/repeat"):
                    started = time.perf_counter()
                    pallas_pack_last_output = _block_until_ready(
                        pallas_pack_probe(
                            query,
                            key,
                            value,
                            g,
                            beta,
                            active_rows,
                            active_chunk_indices,
                            key_offsets,
                            value_offsets,
                        )
                    )
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            pack_info.update(
                {
                    "active_chunk_count": int(active_chunk_plan["active_chunk_count"]),
                    "packed_true_tokens": int(active_chunk_plan["active_chunk_count"]) * int(args.chunk_size),
                    "warmup_ms": warmup_ms,
                    "repeat_ms": repeat_ms,
                    "mean_ms": mean_ms,
                    "p50_ms": _percentile(repeat_ms, 50),
                    "p95_ms": _percentile(repeat_ms, 95),
                    "min_ms": min(repeat_ms) if repeat_ms else None,
                    "max_ms": max(repeat_ms) if repeat_ms else None,
                    "profiled_iterations": int(args.warmups) + int(args.repeats),
                }
            )
        if pallas_local_math_probe is not None:
            local_math_info = pallas_feasibility["active_chunk_local_math_probe"]
            warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/pallas_active_chunk_local_math_probe/warmup"):
                    started = time.perf_counter()
                    pallas_local_math_last_output = _block_until_ready(
                        pallas_local_math_probe(
                            normalized_key,
                            value,
                            g,
                            beta,
                            active_rows,
                            active_chunk_indices,
                            key_offsets,
                            value_offsets,
                            token_offsets,
                        )
                    )
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/pallas_active_chunk_local_math_probe/repeat"):
                    started = time.perf_counter()
                    pallas_local_math_last_output = _block_until_ready(
                        pallas_local_math_probe(
                            normalized_key,
                            value,
                            g,
                            beta,
                            active_rows,
                            active_chunk_indices,
                            key_offsets,
                            value_offsets,
                            token_offsets,
                        )
                    )
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            local_math_info.update(
                {
                    "active_chunk_count": int(active_chunk_plan["active_chunk_count"]),
                    "warmup_ms": warmup_ms,
                    "repeat_ms": repeat_ms,
                    "mean_ms": mean_ms,
                    "p50_ms": _percentile(repeat_ms, 50),
                    "p95_ms": _percentile(repeat_ms, 95),
                    "min_ms": min(repeat_ms) if repeat_ms else None,
                    "max_ms": max(repeat_ms) if repeat_ms else None,
                    "profiled_iterations": int(args.warmups) + int(args.repeats),
                }
            )
        if reconstruction_probe is not None and pallas_local_math_last_output is not None:
            reconstruction_info = pallas_feasibility["active_output_state_reconstruction_probe"]
            warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/pallas_active_output_state_reconstruction_probe/warmup"):
                    started = time.perf_counter()
                    reconstruction_pallas_last_output = _block_until_ready(
                        reconstruction_probe(
                            normalized_query_scaled,
                            normalized_key,
                            initial_state,
                            active_rows,
                            active_chunk_indices,
                            pallas_local_math_last_output[0],
                            pallas_local_math_last_output[1],
                            pallas_local_math_last_output[2],
                        )
                    )
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/pallas_active_output_state_reconstruction_probe/repeat"):
                    started = time.perf_counter()
                    reconstruction_pallas_last_output = _block_until_ready(
                        reconstruction_probe(
                            normalized_query_scaled,
                            normalized_key,
                            initial_state,
                            active_rows,
                            active_chunk_indices,
                            pallas_local_math_last_output[0],
                            pallas_local_math_last_output[1],
                            pallas_local_math_last_output[2],
                        )
                    )
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            reconstruction_info.update(
                {
                    "warmup_ms": warmup_ms,
                    "repeat_ms": repeat_ms,
                    "mean_ms": mean_ms,
                    "p50_ms": _percentile(repeat_ms, 50),
                    "p95_ms": _percentile(repeat_ms, 95),
                    "min_ms": min(repeat_ms) if repeat_ms else None,
                    "max_ms": max(repeat_ms) if repeat_ms else None,
                    "profiled_iterations": int(args.warmups) + int(args.repeats),
                }
            )
        if pallas_rectangular_local_math_probe is not None:
            rectangular_info = pallas_feasibility["rectangular_chunk_local_math_probe"]
            warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/pallas_rectangular_chunk_local_math_probe/warmup"):
                    started = time.perf_counter()
                    pallas_rectangular_local_math_last_output = _block_until_ready(
                        pallas_rectangular_local_math_probe(
                            normalized_key,
                            value,
                            g,
                            beta,
                            key_offsets,
                            value_offsets,
                            token_offsets,
                        )
                    )
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/pallas_rectangular_chunk_local_math_probe/repeat"):
                    started = time.perf_counter()
                    pallas_rectangular_local_math_last_output = _block_until_ready(
                        pallas_rectangular_local_math_probe(
                            normalized_key,
                            value,
                            g,
                            beta,
                            key_offsets,
                            value_offsets,
                            token_offsets,
                        )
                    )
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            rectangular_info.update(
                {
                    "warmup_ms": warmup_ms,
                    "repeat_ms": repeat_ms,
                    "mean_ms": mean_ms,
                    "p50_ms": _percentile(repeat_ms, 50),
                    "p95_ms": _percentile(repeat_ms, 95),
                    "min_ms": min(repeat_ms) if repeat_ms else None,
                    "max_ms": max(repeat_ms) if repeat_ms else None,
                    "profiled_iterations": int(args.warmups) + int(args.repeats),
                }
            )
        if rectangular_reconstruction_probe is not None and rectangular_local_math_reference is not None:
            rectangular_reconstruction_info = pallas_feasibility["rectangular_local_split_reconstruction_probe"]
            warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/rectangular_local_split_reconstruction_probe/warmup"):
                    started = time.perf_counter()
                    rectangular_reference_reconstruction_output = _block_until_ready(
                        rectangular_reconstruction_probe(
                            normalized_query_scaled,
                            normalized_key,
                            initial_state,
                            rectangular_local_math_reference[0],
                            rectangular_local_math_reference[1],
                            rectangular_local_math_reference[2],
                        )
                    )
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/rectangular_local_split_reconstruction_probe/repeat"):
                    started = time.perf_counter()
                    rectangular_reference_reconstruction_output = _block_until_ready(
                        rectangular_reconstruction_probe(
                            normalized_query_scaled,
                            normalized_key,
                            initial_state,
                            rectangular_local_math_reference[0],
                            rectangular_local_math_reference[1],
                            rectangular_local_math_reference[2],
                        )
                    )
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            rectangular_reconstruction_info.update(
                {
                    "warmup_ms": warmup_ms,
                    "repeat_ms": repeat_ms,
                    "mean_ms": mean_ms,
                    "p50_ms": _percentile(repeat_ms, 50),
                    "p95_ms": _percentile(repeat_ms, 95),
                    "min_ms": min(repeat_ms) if repeat_ms else None,
                    "max_ms": max(repeat_ms) if repeat_ms else None,
                    "profiled_iterations": int(args.warmups) + int(args.repeats),
                }
            )
        if rectangular_reconstruction_probe is not None and pallas_rectangular_local_math_last_output is not None:
            rectangular_reconstruction_info = pallas_feasibility["rectangular_local_split_reconstruction_probe"]
            pallas_warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/pallas_rectangular_output_state_reconstruction_probe/warmup"):
                    started = time.perf_counter()
                    rectangular_pallas_reconstruction_output = _block_until_ready(
                        rectangular_reconstruction_probe(
                            normalized_query_scaled,
                            normalized_key,
                            initial_state,
                            pallas_rectangular_local_math_last_output[0],
                            pallas_rectangular_local_math_last_output[1],
                            pallas_rectangular_local_math_last_output[2],
                        )
                    )
                    pallas_warmup_ms.append(1000.0 * (time.perf_counter() - started))
            pallas_repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/pallas_rectangular_output_state_reconstruction_probe/repeat"):
                    started = time.perf_counter()
                    rectangular_pallas_reconstruction_output = _block_until_ready(
                        rectangular_reconstruction_probe(
                            normalized_query_scaled,
                            normalized_key,
                            initial_state,
                            pallas_rectangular_local_math_last_output[0],
                            pallas_rectangular_local_math_last_output[1],
                            pallas_rectangular_local_math_last_output[2],
                        )
                    )
                    pallas_repeat_ms.append(1000.0 * (time.perf_counter() - started))
            pallas_mean_ms = float(sum(pallas_repeat_ms) / len(pallas_repeat_ms)) if pallas_repeat_ms else None
            rectangular_reconstruction_info.update(
                {
                    "pallas_warmup_ms": pallas_warmup_ms,
                    "pallas_repeat_ms": pallas_repeat_ms,
                    "pallas_mean_ms": pallas_mean_ms,
                    "pallas_p50_ms": _percentile(pallas_repeat_ms, 50),
                    "pallas_p95_ms": _percentile(pallas_repeat_ms, 95),
                    "pallas_min_ms": min(pallas_repeat_ms) if pallas_repeat_ms else None,
                    "pallas_max_ms": max(pallas_repeat_ms) if pallas_repeat_ms else None,
                }
            )
        if one_piece_gdn_prefill_probe is not None:
            one_piece_info = pallas_feasibility["one_piece_gdn_prefill_probe"]
            block_v = int(one_piece_info["block_v"])
            value_block_offsets = jnp.arange(block_v, dtype=jnp.int32)
            warmup_ms = []
            for _ in range(args.warmups):
                with _trace_annotation("gdn_prefill/pallas_one_piece_gdn_prefill_probe/warmup"):
                    started = time.perf_counter()
                    one_piece_gdn_prefill_last_output = _block_until_ready(
                        one_piece_gdn_prefill_probe(
                            normalized_query_scaled,
                            normalized_key,
                            value,
                            g,
                            beta,
                            initial_state,
                            key_offsets,
                            value_block_offsets,
                            token_offsets,
                        )
                    )
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation("gdn_prefill/pallas_one_piece_gdn_prefill_probe/repeat"):
                    started = time.perf_counter()
                    one_piece_gdn_prefill_last_output = _block_until_ready(
                        one_piece_gdn_prefill_probe(
                            normalized_query_scaled,
                            normalized_key,
                            value,
                            g,
                            beta,
                            initial_state,
                            key_offsets,
                            value_block_offsets,
                            token_offsets,
                        )
                    )
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            one_piece_info.update(
                {
                    "warmup_ms": warmup_ms,
                    "repeat_ms": repeat_ms,
                    "mean_ms": mean_ms,
                    "p50_ms": _percentile(repeat_ms, 50),
                    "p95_ms": _percentile(repeat_ms, 95),
                    "min_ms": min(repeat_ms) if repeat_ms else None,
                    "max_ms": max(repeat_ms) if repeat_ms else None,
                    "profiled_iterations": int(args.warmups) + int(args.repeats),
                }
            )
        for name, _fn, compiled, compile_seconds in compiled_variants:
            warmup_ms = []
            last_output = None
            for _ in range(args.warmups):
                with _trace_annotation(f"gdn_prefill/{name}/warmup"):
                    started = time.perf_counter()
                    last_output = _block_until_ready(compiled(*inputs))
                    warmup_ms.append(1000.0 * (time.perf_counter() - started))
            repeat_ms = []
            for _ in range(args.repeats):
                with _trace_annotation(f"gdn_prefill/{name}/repeat"):
                    started = time.perf_counter()
                    last_output = _block_until_ready(compiled(*inputs))
                    repeat_ms.append(1000.0 * (time.perf_counter() - started))
            outputs[name] = last_output
            mean_ms = float(sum(repeat_ms) / len(repeat_ms)) if repeat_ms else None
            variant_results[name] = {
                "compile_seconds": compile_seconds,
                "warmup_ms": warmup_ms,
                "repeat_ms": repeat_ms,
                "mean_ms": mean_ms,
                "p50_ms": _percentile(repeat_ms, 50),
                "p95_ms": _percentile(repeat_ms, 95),
                "min_ms": min(repeat_ms) if repeat_ms else None,
                "max_ms": max(repeat_ms) if repeat_ms else None,
                "true_tokens_per_second": (1000.0 * true_tokens / mean_ms) if mean_ms else None,
                "rectangular_tokens_per_second": (1000.0 * rectangular_tokens / mean_ms) if mean_ms else None,
            }
    finally:
        recorder.stop_jax_profile()

    reference_name = "current_jax_chunk32_padded"
    if reference_name not in outputs:
        reference_name = next(iter(outputs))
    comparisons = {}
    for name, output in outputs.items():
        comparisons[f"{name}_vs_{reference_name}"] = _compare_outputs(outputs[reference_name], output, valid)
    segmented_reference_gate: dict[str, Any] = {
        "enabled": bool(args.check_segmented_reference_gate),
        "attempted": False,
        "passes_1e_5_gate": None,
        "comparison": None,
        "run_ms": None,
        "error": None,
    }
    if args.check_segmented_reference_gate:
        segmented_reference_gate["attempted"] = True
        try:
            started = time.perf_counter()
            (
                packed_query,
                packed_key,
                packed_value,
                packed_g,
                packed_beta,
                cu_seqlens,
            ) = pack_padded_gdn_inputs(query, key, value, g, beta, lengths)
            packed_output, segmented_state = gdn_segmented_prefill_chunk32_reference(
                packed_query,
                packed_key,
                packed_value,
                packed_beta,
                packed_g,
                cu_seqlens,
                initial_state,
                chunk_size=int(args.chunk_size),
                use_qk_l2norm_in_kernel=True,
            )
            segmented_output = unpack_segmented_gdn_output(
                packed_output,
                cu_seqlens,
                int(args.seq_len),
            )
            segmented_output = _block_until_ready((segmented_output, segmented_state))
            comparison = _compare_outputs(outputs[reference_name], segmented_output, valid)
            actual_length_run_ms = 1000.0 * (time.perf_counter() - started)
            padded_started = time.perf_counter()
            padded_packed_output, padded_segmented_state = (
                gdn_segmented_prefill_chunk32_reference(
                    packed_query,
                    packed_key,
                    packed_value,
                    packed_beta,
                    packed_g,
                    cu_seqlens,
                    initial_state,
                    chunk_size=int(args.chunk_size),
                    use_qk_l2norm_in_kernel=True,
                    reference_seq_len=int(args.seq_len),
                )
            )
            padded_segmented_output = unpack_segmented_gdn_output(
                padded_packed_output,
                cu_seqlens,
                int(args.seq_len),
            )
            padded_segmented_output = _block_until_ready(
                (padded_segmented_output, padded_segmented_state)
            )
            padded_comparison = _compare_outputs(
                outputs[reference_name],
                padded_segmented_output,
                valid,
            )
            segmented_reference_gate.update(
                {
                    "run_ms": actual_length_run_ms,
                    "total_gate_run_ms": 1000.0 * (time.perf_counter() - started),
                    "comparison": comparison,
                    "row_padded_to_seq_len": {
                        "reference_seq_len": int(args.seq_len),
                        "run_ms": 1000.0 * (time.perf_counter() - padded_started),
                        "comparison": padded_comparison,
                        "passes_1e_5_gate": bool(
                            padded_comparison["output_max_abs"] <= 1e-5
                            and padded_comparison["valid_output_max_abs"] <= 1e-5
                            and padded_comparison["state_max_abs"] <= 1e-5
                        ),
                    },
                    "cu_seqlens": np.asarray(cu_seqlens).tolist(),
                    "nnz_tokens": int(packed_query.shape[0]),
                    "passes_1e_5_gate": bool(
                        comparison["output_max_abs"] <= 1e-5
                        and comparison["valid_output_max_abs"] <= 1e-5
                        and comparison["state_max_abs"] <= 1e-5
                    ),
                }
            )
        except BaseException as exc:
            segmented_reference_gate["error"] = f"{type(exc).__name__}: {exc}"
    segmented_reference_gate["policy"] = _segmented_reference_policy(segmented_reference_gate)
    if pallas_pack_last_output is not None and pallas_pack_reference is not None:
        pallas_feasibility["active_input_pack_probe"]["comparisons"] = _compare_pack_outputs(
            pallas_pack_reference,
            pallas_pack_last_output,
        )
    if pallas_local_math_last_output is not None and pallas_local_math_reference is not None:
        pallas_feasibility["active_chunk_local_math_probe"]["comparisons"] = _compare_local_math_outputs(
            pallas_local_math_reference,
            pallas_local_math_last_output,
        )
    reconstruction_info = pallas_feasibility["active_output_state_reconstruction_probe"]
    if reconstruction_reference_output is not None:
        reference_comparison = _compare_outputs(outputs[reference_name], reconstruction_reference_output, valid)
        reconstruction_info["reference_local_math_comparison"] = reference_comparison
        reconstruction_info["reference_local_math_passes_1e_5_gate"] = bool(
            reference_comparison["output_max_abs"] <= 1e-5
            and reference_comparison["valid_output_max_abs"] <= 1e-5
            and reference_comparison["state_max_abs"] <= 1e-5
        )
    if reconstruction_pallas_last_output is not None:
        pallas_comparison = _compare_outputs(outputs[reference_name], reconstruction_pallas_last_output, valid)
        reconstruction_info["pallas_local_math_comparison"] = pallas_comparison
        reconstruction_info["pallas_local_math_passes_1e_5_gate"] = bool(
            pallas_comparison["output_max_abs"] <= 1e-5
            and pallas_comparison["valid_output_max_abs"] <= 1e-5
            and pallas_comparison["state_max_abs"] <= 1e-5
        )
    rectangular_info = pallas_feasibility["rectangular_chunk_local_math_probe"]
    rectangular_reconstruction_info = pallas_feasibility["rectangular_local_split_reconstruction_probe"]
    if pallas_rectangular_local_math_last_output is not None and rectangular_local_math_reference is not None:
        rectangular_info["comparisons"] = _compare_local_math_outputs(
            rectangular_local_math_reference,
            pallas_rectangular_local_math_last_output,
        )
    if rectangular_reference_reconstruction_output is not None:
        rectangular_reference_comparison = _compare_outputs(
            outputs[reference_name],
            rectangular_reference_reconstruction_output,
            valid,
        )
        rectangular_reconstruction_info["reference_local_math_comparison"] = rectangular_reference_comparison
        rectangular_reconstruction_info["reference_local_math_passes_1e_5_gate"] = bool(
            rectangular_reference_comparison["output_max_abs"] <= 1e-5
            and rectangular_reference_comparison["valid_output_max_abs"] <= 1e-5
            and rectangular_reference_comparison["state_max_abs"] <= 1e-5
        )
    if rectangular_pallas_reconstruction_output is not None:
        rectangular_pallas_comparison = _compare_outputs(
            outputs[reference_name],
            rectangular_pallas_reconstruction_output,
            valid,
        )
        rectangular_reconstruction_info["pallas_local_math_comparison"] = rectangular_pallas_comparison
        rectangular_reconstruction_info["pallas_local_math_passes_1e_5_gate"] = bool(
            rectangular_pallas_comparison["output_max_abs"] <= 1e-5
            and rectangular_pallas_comparison["valid_output_max_abs"] <= 1e-5
            and rectangular_pallas_comparison["state_max_abs"] <= 1e-5
        )
    one_piece_info = pallas_feasibility["one_piece_gdn_prefill_probe"]
    if one_piece_gdn_prefill_last_output is not None:
        one_piece_comparison = _compare_outputs(
            outputs[reference_name],
            one_piece_gdn_prefill_last_output,
            valid,
        )
        one_piece_info["comparison"] = one_piece_comparison
        one_piece_info["passes_1e_5_gate"] = bool(
            one_piece_comparison["output_max_abs"] <= 1e-5
            and one_piece_comparison["valid_output_max_abs"] <= 1e-5
            and one_piece_comparison["state_max_abs"] <= 1e-5
        )
    profiled_iterations = len(variants) * (int(args.warmups) + int(args.repeats))
    profile_counters = _profile_counters(recorder.profile_path, profiled_iterations, list(variants)) if args.profile else None
    if profile_counters is not None:
        ranges = profile_counters.get("ranges", {})
        probe_custom_call = ranges.get("gdn_active_chunk_probe")
        if probe_custom_call is not None:
            pallas_feasibility["custom_call_count"] = int(probe_custom_call.get("count", 0))
        pack_custom_call = ranges.get("gdn_active_input_pack_probe")
        if pack_custom_call is not None:
            pallas_feasibility["active_input_pack_probe"]["custom_call_count"] = int(pack_custom_call.get("count", 0))
        local_math_custom_call = ranges.get("gdn_active_chunk_local_math_chunk32_probe")
        if local_math_custom_call is not None:
            pallas_feasibility["active_chunk_local_math_probe"]["custom_call_count"] = int(local_math_custom_call.get("count", 0))
        rectangular_custom_call = ranges.get("gdn_rectangular_chunk_local_math_chunk32_probe")
        if rectangular_custom_call is not None:
            pallas_feasibility["rectangular_chunk_local_math_probe"]["custom_call_count"] = int(rectangular_custom_call.get("count", 0))
        one_piece_custom_call = ranges.get("gdn_one_piece_gdn_prefill_vblock64_probe")
        if one_piece_custom_call is not None:
            pallas_feasibility["one_piece_gdn_prefill_probe"]["custom_call_count"] = int(one_piece_custom_call.get("count", 0))

    return {
        "run_config": {
            "git_head": _git_head(),
            **_device_summary(),
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "seq_len": args.seq_len,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "chunk_size": args.chunk_size,
            "lengths": lengths,
            "true_tokens": true_tokens,
            "rectangular_tokens": rectangular_tokens,
            "active_chunks": active_chunks,
            "total_chunks": total_chunks,
            "active_chunk_plan": active_chunk_plan,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "variants": list(variants),
            "enable_one_piece_gdn_probe": bool(args.enable_one_piece_gdn_probe),
            "dtype_contract": "fp32 activations/state, output cast follows current jax_chunk_gated_delta_rule",
        },
        "variants": variant_results,
        "comparisons": comparisons,
        "profile_counters": profile_counters,
        "pallas_feasibility": pallas_feasibility,
        "segmented_reference_gate": segmented_reference_gate,
        "historical_negative_controls": [
            "Entry053 static row-chunk ragged GDN prefill",
            "Entry056 static chunk-major GDN prefill",
        ],
        "decision": {
            "promote_to_server_routing": False,
            "reason": "standalone GDN probes only; no candidate was routed into the server",
        },
        "run": recorder.metadata(),
    }


def main() -> None:
    args = parse_args()
    recorder = RunRecorder.create(
        script=Path(__file__).name,
        args=vars(args),
        run_label=args.run_label,
        profile_dir=args.profile_dir or None,
        run_log=args.run_log or None,
    )
    try:
        summary = run_benchmark(args, recorder)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
        recorder.finish(
            status="ok",
            summary={
                "variants": {
                    name: {
                        "mean_ms": result["mean_ms"],
                        "true_tokens_per_second": result["true_tokens_per_second"],
                    }
                    for name, result in summary["variants"].items()
                },
                "output_json": str(output_path),
            },
            learnings=[
                "Standalone GDN microbenchmarks are gates for backend kernels only; server routing still needs full hetero8 proof.",
            ],
            resolution="wrote benchmark summary",
        )
    except BaseException as exc:
        recorder.finish_exception(exc)
        raise


if __name__ == "__main__":
    main()
