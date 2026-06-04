#!/usr/bin/env python3
"""Audit where vLLM/FLA GDN prefill diverges from local references.

This is a numeric investigation harness, not a serving benchmark. It feeds the
same real post-conv GDN tensors through:

* local packed FP32 reference,
* local packed BF16-QKV reference,
* vLLM's vendored FLA chain with its default BF16 triangular-solve output,
* vLLM's vendored FLA chain with FP32 triangular-solve output.

The goal is to isolate dtype drift from kernel-schedule drift while holding the
packed FLA layout and real model activations fixed.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.microbench_vllm_fla_gdn_real_activation import _build_real_case


CHUNK_SIZE = 64
DEFAULT_OUT = Path("results/vllm_fla_correctness_boundary_audit_20260601.json")
DEFAULT_MD = Path("results/vllm_fla_correctness_boundary_audit_20260601.md")
STAGE_ORDER = (
    "gate_cumsum",
    "attention_matrix",
    "attention_inverse",
    "w",
    "u",
    "h",
    "v_new",
    "output",
    "final_state",
)


def _configure_runtime_env() -> None:
    os.environ.setdefault("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp")
    os.environ.setdefault("HF_HOME", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mountpoint/.exp/.cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/mountpoint/.exp/.cache")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/mountpoint/.exp/.cache/flashinfer")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
        if str(getattr(x, "dtype", "")) == "torch.bfloat16":
            x = x.float()
    return np.asarray(x)


def _quantize_bf16(x: Any) -> np.ndarray:
    import jax.numpy as jnp

    return np.asarray(jnp.asarray(x, dtype=jnp.bfloat16), dtype=np.float32)


def _tensor_signature(x: Any) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float32)
    flat = arr.reshape(-1)
    sample_count = min(8, flat.size)
    digest = hashlib.sha1(arr.view(np.float32)).hexdigest()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sample_head": flat[:sample_count].astype(np.float32).tolist(),
        "sum_abs": float(np.sum(np.abs(flat))),
        "mean": float(np.mean(flat)),
        "l2": float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0,
        "fingerprint": digest,
    }


def _build_case_packed_signature(case: Any, *, qkv_dtype: str) -> dict[str, Any]:
    import jax.numpy as jnp

    from nanovllm_jax.kernels.gdn_fla import (
        pack_prepared_gdn_prefill_inputs,
        prepare_gdn_fla_prefill_kernel_inputs,
    )

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        jnp.asarray(case.query),
        jnp.asarray(case.key),
        jnp.asarray(case.value),
        jnp.asarray(case.gate, dtype=jnp.float32),
        jnp.asarray(case.beta, dtype=jnp.float32),
        jnp.asarray(case.seq_lens, dtype=jnp.int32),
        jnp.asarray(case.initial_state, dtype=jnp.float32),
        qkv_dtype=qkv_dtype,
    )
    pq, pk, pv, pg, pb, pcu = pack_prepared_gdn_prefill_inputs(
        prepared.query,
        prepared.key,
        prepared.value,
        prepared.gate,
        prepared.beta,
        prepared.seq_lens,
    )
    if qkv_dtype == "bfloat16":
        quantized_query = _quantize_bf16(_to_np(case.query))
        quantized_key = _quantize_bf16(_to_np(case.key))
        quantized_value = _quantize_bf16(_to_np(case.value))
    elif qkv_dtype == "float32":
        quantized_query = np.asarray(_to_np(case.query), dtype=np.float32)
        quantized_key = np.asarray(_to_np(case.key), dtype=np.float32)
        quantized_value = np.asarray(_to_np(case.value), dtype=np.float32)
    else:
        raise ValueError("qkv_dtype must be bfloat16 or float32")

    return {
        f"case_query_{qkv_dtype}": _tensor_signature(quantized_query),
        f"case_key_{qkv_dtype}": _tensor_signature(quantized_key),
        f"case_value_{qkv_dtype}": _tensor_signature(quantized_value),
        "packed_query": _tensor_signature(_to_np(pq)),
        "packed_key": _tensor_signature(_to_np(pk)),
        "packed_value": _tensor_signature(_to_np(pv)),
        "packed_gate": _tensor_signature(_to_np(pg)),
        "packed_beta": _tensor_signature(_to_np(pb)),
        "cu_seqlens": _tensor_signature(_to_np(pcu)),
    }


def _diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "error": "shape mismatch"}
    d = np.abs(a - b)
    finite = np.isfinite(d)
    if not finite.any():
        return {
            "shape": list(a.shape),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "nan_count": int(d.size),
        }
    max_flat = int(np.argmax(np.where(finite, d, -np.inf)))
    max_index = tuple(int(i) for i in np.unravel_index(max_flat, d.shape))
    row: dict[str, Any] = {
        "shape": list(a.shape),
        "max_abs": float(d[max_index]),
        "mean_abs": float(np.mean(d[finite])),
        "rms": float(np.sqrt(np.mean((a[finite] - b[finite]) ** 2))),
        "nan_count": int(d.size - int(finite.sum())),
        "max_index": list(max_index),
        "a_at_max": float(a[max_index]),
        "b_at_max": float(b[max_index]),
    }
    for threshold in (1e-3, 1e-2, 1e-1):
        hits = np.argwhere(finite & (d > threshold))
        key = f"first_over_{threshold:g}"
        row[key] = hits[0].astype(int).tolist() if hits.size else None
    return row


def _load_runtime(model_id: str):
    import torch
    from nanovllm_jax.config import Qwen3_5Config
    from nanovllm_jax.load_weights import load_weights_from_hf
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    vllm_path = "/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages"
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)

    cumsum = __import__(
        "vllm.model_executor.layers.fla.ops.cumsum",
        fromlist=["chunk_local_cumsum"],
    )
    kkt = __import__(
        "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
        fromlist=["chunk_scaled_dot_kkt_fwd"],
    )
    solve = __import__(
        "vllm.model_executor.layers.fla.ops.solve_tril",
        fromlist=["solve_tril"],
    )
    wy = __import__(
        "vllm.model_executor.layers.fla.ops.wy_fast",
        fromlist=["recompute_w_u_fwd"],
    )
    dh = __import__(
        "vllm.model_executor.layers.fla.ops.chunk_delta_h",
        fromlist=["chunk_gated_delta_rule_fwd_h"],
    )
    out = __import__(
        "vllm.model_executor.layers.fla.ops.chunk_o",
        fromlist=["chunk_fwd_o"],
    )
    chunk = __import__(
        "vllm.model_executor.layers.fla.ops.chunk",
        fromlist=["chunk_gated_delta_rule"],
    )

    load_config = Qwen3_5Config.qwen3_5_0_8b()
    load_config.dtype = "bfloat16"
    config = Qwen3_5Config.qwen3_5_0_8b()
    # Match the benchmark/server correctness contract: BF16 checkpoint weights,
    # FP32 activations unless an explicit BF16 kernel boundary is under test.
    config.dtype = "float32"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    params = load_weights_from_hf(model_id, load_config, verbose=False)
    return config, tokenizer, params, torch, cumsum, kkt, solve, wy, dh, out, chunk


def _run_jax_chain(case, *, qkv_dtype: str) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_delta_h_packed_reference,
        gdn_fla_chunk_fwd_o_packed_reference,
        gdn_fla_chunk_local_cumsum_packed_reference,
        gdn_fla_chunk_scaled_dot_kkt_packed_reference,
        gdn_fla_recompute_w_u_packed_reference,
        gdn_fla_solve_tril_packed_reference,
        pack_prepared_gdn_prefill_inputs,
        prepare_gdn_fla_chunk_metadata,
        prepare_gdn_fla_prefill_kernel_inputs,
    )

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        jnp.asarray(case.query),
        jnp.asarray(case.key),
        jnp.asarray(case.value),
        jnp.asarray(case.gate, dtype=jnp.float32),
        jnp.asarray(case.beta, dtype=jnp.float32),
        jnp.asarray(case.seq_lens, dtype=jnp.int32),
        jnp.asarray(case.initial_state, dtype=jnp.float32),
        qkv_dtype=qkv_dtype,
    )
    pq, pk, pv, pg, pb, pcu = pack_prepared_gdn_prefill_inputs(
        prepared.query,
        prepared.key,
        prepared.value,
        prepared.gate,
        prepared.beta,
        prepared.seq_lens,
    )
    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(pcu, CHUNK_SIZE)
    gcs = gdn_fla_chunk_local_cumsum_packed_reference(
        pg, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    a = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        pk, pb, gcs, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    ai = gdn_fla_solve_tril_packed_reference(
        a, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    w, u = gdn_fla_recompute_w_u_packed_reference(
        pk, pv, pb, gcs, ai, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    h, v_new, final_state = gdn_fla_chunk_delta_h_packed_reference(
        pk,
        w,
        u,
        gcs,
        pcu,
        prepared.initial_state,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        output_final_state=True,
        save_new_value=True,
    )
    if v_new is None or final_state is None:
        raise AssertionError("expected v_new and final_state")
    output = gdn_fla_chunk_fwd_o_packed_reference(
        pq, pk, v_new, h, gcs, pcu, chunk_size=CHUNK_SIZE, chunk_indices=chunk_indices
    )
    return {
        "packed_query": _to_np(pq),
        "packed_key": _to_np(pk),
        "packed_value": _to_np(pv),
        "gate_cumsum": _to_np(gcs),
        "attention_matrix": _to_np(a),
        "attention_inverse": _to_np(ai),
        "w": _to_np(w),
        "u": _to_np(u),
        "h": _to_np(h),
        "v_new": _to_np(v_new),
        "output": _to_np(output),
        "final_state": _to_np(final_state),
    }


def _run_jax_chunk_delta_h_vllm_like(
    key: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    gate_cumsum: np.ndarray,
    cu_seqlens: Any,
    initial_state: np.ndarray,
    *,
    chunk_size: int,
    chunk_indices: Any,
    chunk_offsets: Any,
    output_final_state: bool = True,
    save_new_value: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import jax
    from nanovllm_jax.kernels.gdn_fla import _vllm_like_quantize_stage

    if key.ndim != 3:
        raise ValueError("key must have shape [nnz_tokens, key_heads, key_dim]")
    if w.ndim != 3:
        raise ValueError("w must have shape [nnz_tokens, output_heads, key_dim]")
    if u.ndim != 3:
        raise ValueError("u must have shape [nnz_tokens, output_heads, value_dim]")
    if w.shape[0] != u.shape[0] or w.shape[0] != key.shape[0]:
        raise ValueError("key, w, and u token counts must match")
    if w.shape[:2] != u.shape[:2]:
        raise ValueError("w and u must agree on token and output heads")
    if gate_cumsum.shape != w.shape[:2]:
        raise ValueError("gate_cumsum must have shape [nnz_tokens, output_heads]")

    key_heads = key.shape[1]
    output_heads = w.shape[1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")

    chunk_indices_arr = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    chunk_offsets_arr = np.asarray(jax.device_get(chunk_offsets), dtype=np.int64).reshape(-1)
    if chunk_indices_arr.ndim != 2 or chunk_indices_arr.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")
    if chunk_offsets_arr.shape != (len(offsets),):
        raise ValueError("chunk_offsets must have shape [batch + 1]")

    if initial_state.shape != (
        len(offsets) - 1,
        output_heads,
        u.shape[-1],
        key.shape[-1],
    ):
        raise ValueError("initial_state must match [batch, output_heads, value_dim, key_dim]")

    state = np.array(initial_state, dtype=np.float32)
    key_bf16 = _quantize_bf16(key)
    w_bf16 = _quantize_bf16(w)
    u_bf16 = _quantize_bf16(u)
    final_state = np.zeros_like(state) if output_final_state else None

    h = np.zeros(
        (len(chunk_indices_arr), output_heads, u.shape[-1], key.shape[-1]),
        dtype=np.float32,
    )
    v_new = (
        np.zeros((key.shape[0], output_heads, u.shape[-1]), dtype=np.float32)
        if save_new_value
        else None
    )

    head_group = output_heads // key_heads

    for row in range(len(offsets) - 1):
        row_state = np.array(state[row], copy=True)
        row_start = int(offsets[row])
        row_end = int(offsets[row + 1])
        row_final_state = np.array(row_state, copy=True)
        row_chunks = int(chunk_offsets_arr[row + 1] - chunk_offsets_arr[row])
        for chunk in range(row_chunks):
            flat_chunk = int(chunk_offsets_arr[row]) + chunk
            if tuple(chunk_indices_arr[flat_chunk]) != (row, chunk):
                raise ValueError("chunk_indices and chunk_offsets disagree")

            chunk_start = row_start + chunk * chunk_size
            chunk_end = min(row_end, chunk_start + chunk_size)
            length = chunk_end - chunk_start
            for head in range(output_heads):
                key_head = head // head_group
                state_head = row_state[head]
                state_head_for_dot = _vllm_like_quantize_stage(
                    state_head,
                    output_dtype="bfloat16",
                )
                h[flat_chunk, head] = state_head_for_dot

                chunk_gate = gate_cumsum[chunk_start:chunk_end, head].astype(np.float32)
                chunk_w = w_bf16[chunk_start:chunk_end, head]
                chunk_u = u_bf16[chunk_start:chunk_end, head]
                chunk_key = key_bf16[chunk_start:chunk_end, key_head]

                chunk_w = _vllm_like_quantize_stage(
                    chunk_w,
                    output_dtype="bfloat16",
                )
                chunk_u = _vllm_like_quantize_stage(
                    chunk_u,
                    output_dtype="bfloat16",
                )
                delta = chunk_u - chunk_w @ state_head_for_dot.T

                if v_new is not None:
                    v_new[chunk_start:chunk_end, head] = _vllm_like_quantize_stage(
                        delta,
                        output_dtype="bfloat16",
                    )

                if len(chunk_gate) != length:
                    raise ValueError("gate_cumsum chunk length mismatch")

                update_delta = delta
                if length > 0:
                    last_gate = chunk_gate[length - 1]
                    update_delta = delta * np.exp(last_gate - chunk_gate)[:, None]
                    state_head = state_head_for_dot * np.exp(last_gate)
                else:
                    state_head = state_head_for_dot

                update_delta = _vllm_like_quantize_stage(
                    update_delta,
                    output_dtype="bfloat16",
                )
                chunk_key = _vllm_like_quantize_stage(
                    chunk_key,
                    output_dtype="bfloat16",
                )
                state_head = state_head + update_delta.T @ chunk_key
                row_final_state[head] = state_head
                row_state[head] = _vllm_like_quantize_stage(
                    state_head,
                    output_dtype="bfloat16",
                )

        state[row] = row_state
        if output_final_state and final_state is not None:
            final_state[row] = row_final_state

    if output_final_state:
        return h, v_new, final_state
    return h, v_new, None


def _run_jax_chunk_fwd_o_vllm_like(
    query: np.ndarray,
    key: np.ndarray,
    v_new: np.ndarray,
    h: np.ndarray,
    gate_cumsum: np.ndarray,
    cu_seqlens: Any,
    *,
    chunk_size: int,
    chunk_indices: Any,
    scale: float | None = None,
) -> np.ndarray:
    import jax
    from nanovllm_jax.kernels.gdn_fla import _vllm_like_quantize_stage

    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [nnz_tokens, key_heads, key_dim]")
    if v_new.ndim != 3:
        raise ValueError("v_new must have shape [nnz_tokens, output_heads, value_dim]")
    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")
    if gate_cumsum.shape != v_new.shape[:2]:
        raise ValueError("gate_cumsum must have shape [nnz_tokens, output_heads]")
    if query.shape[0] != v_new.shape[0]:
        raise ValueError("query and v_new token counts must match")
    if h.ndim != 4:
        raise ValueError("h must have shape [num_chunks, output_heads, value_dim, key_dim]")
    if h.shape[1] != v_new.shape[1] or h.shape[2] != v_new.shape[2] or h.shape[3] != query.shape[2]:
        raise ValueError("h dimensions must match v_new and query")

    key_heads = query.shape[1]
    output_heads = v_new.shape[1]
    if output_heads < key_heads or output_heads % key_heads != 0:
        raise ValueError("output_heads must be a multiple of key_heads")

    if scale is None:
        scale = float(query.shape[-1] ** -0.5)

    offsets = np.asarray(jax.device_get(cu_seqlens), dtype=np.int64).reshape(-1)
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("cu_seqlens must start with 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(offsets[-1]) != query.shape[0]:
        raise ValueError("last cu_seqlens entry must equal token count")

    chunk_indices_arr = np.asarray(jax.device_get(chunk_indices), dtype=np.int64)
    if chunk_indices_arr.ndim != 2 or chunk_indices_arr.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")

    query_q = _vllm_like_quantize_stage(np.asarray(query), output_dtype="bfloat16")
    key_q = _vllm_like_quantize_stage(np.asarray(key), output_dtype="bfloat16")
    h_q = _vllm_like_quantize_stage(np.asarray(h), output_dtype="bfloat16")
    v_new_q = _vllm_like_quantize_stage(np.asarray(v_new), output_dtype="bfloat16")

    head_group = output_heads // key_heads
    output = np.zeros(v_new.shape, dtype=np.float32)

    for flat_chunk, (row, chunk) in enumerate(chunk_indices_arr):
        chunk_start = int(offsets[row]) + int(chunk) * chunk_size
        chunk_end = min(int(offsets[row + 1]), chunk_start + chunk_size)
        length = chunk_end - chunk_start
        causal = np.tril(np.ones((length, length), dtype=np.float32), k=0)

        for head in range(output_heads):
            key_head = head // head_group
            chunk_query = query_q[chunk_start:chunk_end, key_head]
            chunk_key = key_q[chunk_start:chunk_end, key_head]
            chunk_gate = gate_cumsum[chunk_start:chunk_end, head].astype(np.float32)
            chunk_state = h_q[flat_chunk, head]

            state_out = chunk_query @ chunk_state.T
            attention = chunk_query @ chunk_key.T
            if len(chunk_gate) != length:
                raise ValueError("chunk gate length mismatch")

            state_out = state_out * np.exp(chunk_gate)[:, None]
            attention = attention * np.exp(chunk_gate[:, None] - chunk_gate[None, :])
            attention = attention * causal
            attention = _vllm_like_quantize_stage(
                attention,
                output_dtype="bfloat16",
            )

            chunk_output = (
                state_out + attention @ v_new_q[chunk_start:chunk_end, head]
            ) * scale
            output[chunk_start:chunk_end, head] = _vllm_like_quantize_stage(
                chunk_output,
                output_dtype="bfloat16",
            )

    return output


def _run_jax_chain_vllm_like(case) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    from nanovllm_jax.kernels.gdn_fla import (
        gdn_fla_chunk_local_cumsum_packed_reference,
        gdn_fla_chunk_scaled_dot_kkt_packed_reference,
        gdn_fla_recompute_w_u_packed_reference,
        gdn_fla_solve_tril_packed_vllm_like_reference,
        pack_prepared_gdn_prefill_inputs,
        prepare_gdn_fla_chunk_metadata,
        prepare_gdn_fla_prefill_kernel_inputs,
    )

    prepared = prepare_gdn_fla_prefill_kernel_inputs(
        jnp.asarray(case.query),
        jnp.asarray(case.key),
        jnp.asarray(case.value),
        jnp.asarray(case.gate, dtype=jnp.float32),
        jnp.asarray(case.beta, dtype=jnp.float32),
        jnp.asarray(case.seq_lens, dtype=jnp.int32),
        jnp.asarray(case.initial_state, dtype=jnp.float32),
        qkv_dtype="bfloat16",
    )
    pq, pk, pv, pg, pb, pcu = pack_prepared_gdn_prefill_inputs(
        prepared.query,
        prepared.key,
        prepared.value,
        prepared.gate,
        prepared.beta,
        prepared.seq_lens,
    )
    pq = jnp.asarray(pq, dtype=jnp.bfloat16)
    pk = jnp.asarray(pk, dtype=jnp.bfloat16)
    pv = jnp.asarray(pv, dtype=jnp.bfloat16)

    chunk_indices, chunk_offsets = prepare_gdn_fla_chunk_metadata(pcu, CHUNK_SIZE)
    gcs = gdn_fla_chunk_local_cumsum_packed_reference(
        pg,
        pcu,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
    )
    a = gdn_fla_chunk_scaled_dot_kkt_packed_reference(
        pk,
        pb,
        gcs,
        pcu,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
    )
    ai = gdn_fla_solve_tril_packed_vllm_like_reference(
        a,
        pcu,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        output_dtype="bfloat16",
    )
    ai = _quantize_bf16(ai)

    w, u = gdn_fla_recompute_w_u_packed_reference(
        pk,
        pv,
        pb,
        gcs,
        ai,
        pcu,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        vllm_like=True,
        stage_output_dtype="bfloat16",
    )
    h, v_new, final_state = _run_jax_chunk_delta_h_vllm_like(
        np.asarray(pk),
        w,
        u,
        _to_np(gcs),
        pcu,
        _to_np(prepared.initial_state),
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        output_final_state=True,
        save_new_value=True,
    )

    if v_new is None or final_state is None:
        raise AssertionError("expected v_new and final_state from vLLM-like delta")

    output = _run_jax_chunk_fwd_o_vllm_like(
        np.asarray(pq),
        np.asarray(pk),
        v_new,
        h,
        _to_np(gcs),
        pcu,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        scale=float(np.asarray(pq).shape[-1] ** -0.5),
    )

    return {
        "packed_query": _to_np(pq),
        "packed_key": _to_np(pk),
        "packed_value": _to_np(pv),
        "input_signature": {
            "case_query_bf16": _tensor_signature(_quantize_bf16(_to_np(case.query))),
            "case_key_bf16": _tensor_signature(_quantize_bf16(_to_np(case.key))),
            "case_value_bf16": _tensor_signature(_quantize_bf16(_to_np(case.value))),
            "case_query": _tensor_signature(_to_np(case.query)),
            "case_key": _tensor_signature(_to_np(case.key)),
            "case_value": _tensor_signature(_to_np(case.value)),
            "case_gate": _tensor_signature(_to_np(case.gate)),
            "case_beta": _tensor_signature(_to_np(case.beta)),
            "case_initial_state": _tensor_signature(_to_np(case.initial_state)),
            "query_bf16_packed": _tensor_signature(_to_np(pq)),
            "key_bf16_packed": _tensor_signature(_to_np(pk)),
            "value_bf16_packed": _tensor_signature(_to_np(pv)),
            "gate_cumsum_input": _tensor_signature(_to_np(pg)),
            "beta_packed": _tensor_signature(_to_np(pb)),
        },
        "gate_cumsum": _to_np(gcs),
        "attention_matrix": _to_np(a),
        "attention_inverse": ai,
        "w": w,
        "u": u,
        "h": h,
        "v_new": v_new,
        "output": _to_np(output),
        "final_state": final_state,
    }


def _run_vllm_chain(
    case,
    torch,
    cumsum_mod,
    kkt_mod,
    solve_mod,
    wy_mod,
    dh_mod,
    out_mod,
    chunk_mod,
    *,
    qkv_torch_dtype: str,
    solve_output_dtype: str,
) -> dict[str, np.ndarray]:
    if qkv_torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif qkv_torch_dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("qkv_torch_dtype must be bfloat16 or float32")
    if solve_output_dtype == "bfloat16":
        solve_dtype = torch.bfloat16
    elif solve_output_dtype == "float32":
        solve_dtype = torch.float32
    else:
        raise ValueError("solve_output_dtype must be bfloat16 or float32")

    q_t = torch.tensor(_to_np(case.query), device="cuda", dtype=dtype)
    k_t = torch.tensor(_to_np(case.key), device="cuda", dtype=dtype)
    v_t = torch.tensor(_to_np(case.value), device="cuda", dtype=dtype)
    g_t = torch.tensor(_to_np(case.gate).astype(np.float32), device="cuda", dtype=torch.float32)
    b_t = torch.tensor(_to_np(case.beta).astype(np.float32), device="cuda", dtype=torch.float32)
    init_t = torch.tensor(
        _to_np(case.initial_state).astype(np.float32), device="cuda", dtype=torch.float32
    )
    q_case_bf16 = _quantize_bf16(_to_np(case.query))
    k_case_bf16 = _quantize_bf16(_to_np(case.key))
    v_case_bf16 = _quantize_bf16(_to_np(case.value))
    cu_t = torch.tensor(_to_np(case.cu_seqlens).astype(np.int64), device="cuda", dtype=torch.int64)

    gcs = cumsum_mod.chunk_local_cumsum(g_t, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t)
    a = kkt_mod.chunk_scaled_dot_kkt_fwd(
        k_t,
        beta=b_t,
        g=gcs,
        cu_seqlens=cu_t,
        chunk_size=CHUNK_SIZE,
        output_dtype=torch.float32,
    )
    ai = solve_mod.solve_tril(a, cu_seqlens=cu_t, output_dtype=solve_dtype)
    w, u = wy_mod.recompute_w_u_fwd(k_t, v_t, b_t, gcs, ai, cu_t)
    h, v_new, final_state = dh_mod.chunk_gated_delta_rule_fwd_h(
        k_t,
        w,
        u,
        g=gcs,
        initial_state=init_t,
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_t,
    )
    output = out_mod.chunk_fwd_o(
        q_t, k_t, v_new, h, g=gcs, cu_seqlens=cu_t, chunk_size=CHUNK_SIZE
    )

    input_signature = {
        "case_query_fp32": _tensor_signature(_to_np(case.query)),
        "case_key_fp32": _tensor_signature(_to_np(case.key)),
        "case_value_fp32": _tensor_signature(_to_np(case.value)),
        "case_query_bf16": _tensor_signature(q_case_bf16),
        "case_key_bf16": _tensor_signature(k_case_bf16),
        "case_value_bf16": _tensor_signature(v_case_bf16),
        "query": _tensor_signature(_to_np(q_t)),
        "key": _tensor_signature(_to_np(k_t)),
        "value": _tensor_signature(_to_np(v_t)),
        "gate": _tensor_signature(_to_np(g_t)),
        "beta": _tensor_signature(_to_np(b_t)),
        "initial_state": _tensor_signature(_to_np(init_t)),
    }

    full_output = None
    full_state = None
    if (
        CHUNK_SIZE == 64
        and qkv_torch_dtype == "bfloat16"
        and solve_output_dtype == "bfloat16"
    ):
        full_output, full_state = chunk_mod.chunk_gated_delta_rule(
            q=q_t,
            k=k_t,
            v=v_t,
            g=g_t,
            beta=b_t,
            initial_state=init_t.clone(),
            output_final_state=True,
            cu_seqlens=cu_t,
            scale=float(case.query.shape[-1] ** -0.5),
        )
    torch.cuda.synchronize()

    length = int(case.token_len)
    row = {
        "input_signature": input_signature,
        "gate_cumsum": _to_np(gcs[0, :length]),
        "attention_matrix": _to_np(a[0, :length]),
        "attention_inverse": _to_np(ai[0, :length]),
        "w": _to_np(w[0, :length]),
        "u": _to_np(u[0, :length]),
        "h": _to_np(h[0]),
        "v_new": _to_np(v_new[0, :length]),
        "output": _to_np(output[0, :length]),
        "final_state": _to_np(final_state),
    }
    if full_output is not None and full_state is not None:
        row["full_output"] = _to_np(full_output[0, :length])
        row["full_state"] = _to_np(full_state)
    return row


def _run_vllm_solve_only(
    case,
    torch,
    cumsum_mod,
    kkt_mod,
    solve_mod,
    *,
    qkv_torch_dtype: str,
    solve_output_dtype: str,
) -> dict[str, np.ndarray]:
    if qkv_torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif qkv_torch_dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("qkv_torch_dtype must be bfloat16 or float32")
    if solve_output_dtype == "bfloat16":
        solve_dtype = torch.bfloat16
    elif solve_output_dtype == "float32":
        solve_dtype = torch.float32
    else:
        raise ValueError("solve_output_dtype must be bfloat16 or float32")

    k_t = torch.tensor(_to_np(case.key), device="cuda", dtype=dtype)
    g_t = torch.tensor(_to_np(case.gate).astype(np.float32), device="cuda", dtype=torch.float32)
    b_t = torch.tensor(_to_np(case.beta).astype(np.float32), device="cuda", dtype=torch.float32)
    cu_t = torch.tensor(_to_np(case.cu_seqlens).astype(np.int64), device="cuda", dtype=torch.int64)

    gcs = cumsum_mod.chunk_local_cumsum(g_t, chunk_size=CHUNK_SIZE, cu_seqlens=cu_t)
    a = kkt_mod.chunk_scaled_dot_kkt_fwd(
        k_t,
        beta=b_t,
        g=gcs,
        cu_seqlens=cu_t,
        chunk_size=CHUNK_SIZE,
        output_dtype=torch.float32,
    )
    ai = solve_mod.solve_tril(a, cu_seqlens=cu_t, output_dtype=solve_dtype)
    torch.cuda.synchronize()
    length = int(case.token_len)
    return {
        "gate_cumsum": _to_np(gcs[0, :length]),
        "attention_matrix": _to_np(a[0, :length]),
        "attention_inverse": _to_np(ai[0, :length]),
    }


def _stage_diffs(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    return {stage: _diff(left[stage], right[stage]) for stage in STAGE_ORDER}


def _signature_match(
    left: dict[str, Any], right: dict[str, Any], left_key: str, right_key: str
) -> bool:
    return bool(
        left.get(left_key, {}).get("fingerprint")
        == right.get(right_key, {}).get("fingerprint")
    )


def _first_stage_over(stage_diffs: dict[str, Any], threshold: float) -> dict[str, Any]:
    for stage in STAGE_ORDER:
        metrics = stage_diffs[stage]
        if "max_abs" in metrics and metrics["max_abs"] > threshold:
            return {
                "stage": stage,
                "threshold": threshold,
                "max_abs": metrics["max_abs"],
                "max_index": metrics.get("max_index"),
                "first_over": metrics.get(f"first_over_{threshold:g}"),
            }
    return {"stage": None, "threshold": threshold}


def _metric_value(metric: Any, key: str = "max_abs") -> str:
    if isinstance(metric, dict):
        if key in metric and isinstance(metric[key], (int, float)) and not isinstance(
            metric[key], bool
        ):
            return f"{float(metric[key]):.6g}"
        if "status" in metric:
            return str(metric["status"])
        if "note" in metric:
            return str(metric["note"])
    return "n/a"


def _run_length(
    *,
    length: int,
    params: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    torch: Any,
    cumsum_mod: Any,
    kkt_mod: Any,
    solve_mod: Any,
    wy_mod: Any,
    dh_mod: Any,
    out_mod: Any,
    chunk_mod: Any,
    skip_vllm: bool = False,
) -> dict[str, Any]:
    case = _build_real_case(
        params=params,
        config=config,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        prompt=prompt,
        token_len=length,
        qkv_dtype="float32",
    )
    case_packed_signature = _build_case_packed_signature(case, qkv_dtype="bfloat16")

    jax_fp32 = _run_jax_chain(case, qkv_dtype="float32")
    jax_bf16 = _run_jax_chain(case, qkv_dtype="bfloat16")
    vllm_like_bf16 = _run_jax_chain_vllm_like(case)
    default_vs_vllm_like = _stage_diffs(vllm_like_bf16, jax_bf16)
    dtype_only = _stage_diffs(jax_bf16, jax_fp32)

    if skip_vllm:
        summary = {
            "dtype_only_output": dtype_only["output"],
            "dtype_only_state": dtype_only["final_state"],
            "vllm_default_vs_jax_bf16_output": {
                "status": "skip_vllm",
            },
            "vllm_default_vs_jax_bf16_state": {
                "status": "skip_vllm",
            },
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32_output": {
                "status": "skip_vllm",
            },
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32_state": {
                "status": "skip_vllm",
            },
            "vllm_default_vs_fp32_output": {"status": "skip_vllm"},
            "vllm_default_vs_fp32_state": {"status": "skip_vllm"},
            "vllm_default_vs_vllm_fp32_qkv_output": {"status": "skip_vllm"},
            "vllm_default_vs_vllm_fp32_qkv_state": {"status": "skip_vllm"},
            "vllm_default_vs_vllm_like_output": default_vs_vllm_like["output"],
            "vllm_default_vs_vllm_like_state": default_vs_vllm_like["final_state"],
            "vllm_bf16_qkv_fp32_solve_inverse_vs_jax_bf16": {
                "status": "skip_vllm",
            },
        }
        return {
            "length": length,
            "input_contract": {
                "case_packed_signature": case_packed_signature,
                "vllm_via_cpu": {
                    "note": "vLLM CUDA path was skipped; only local JAX comparisons were executed."
                },
                "query_bf16_input_matches": {
                    "query_vllm_like_to_case_packed": _signature_match(
                        vllm_like_bf16["input_signature"],
                        case_packed_signature,
                        "query_bf16_packed",
                        "packed_query",
                    ),
                    "key_vllm_like_to_case_packed": _signature_match(
                        vllm_like_bf16["input_signature"],
                        case_packed_signature,
                        "key_bf16_packed",
                        "packed_key",
                    ),
                    "value_vllm_like_to_case_packed": _signature_match(
                        vllm_like_bf16["input_signature"],
                        case_packed_signature,
                        "value_bf16_packed",
                        "packed_value",
                    ),
                },
            },
            "tensor_contract": {
                "packed_q_shape": list(jax_fp32["packed_query"].shape),
                "packed_v_shape": list(jax_fp32["packed_value"].shape),
                "state_shape": list(jax_fp32["final_state"].shape),
                "qkv_vllm_dtype": "bfloat16",
                "gate_beta_state_dtype": "float32",
            },
            "summary": summary,
            "first_stage_over_1e_2": {
                "dtype_only": _first_stage_over(dtype_only, 1e-2),
                "vllm_default_vs_jax_bf16": {
                    "stage": None,
                    "threshold": 1e-2,
                    "note": "skip_vllm",
                },
                "vllm_fp32_qkv_fp32_solve_vs_jax_fp32": {
                    "stage": None,
                    "threshold": 1e-2,
                    "note": "skip_vllm",
                },
                "vllm_default_vs_vllm_like": _first_stage_over(default_vs_vllm_like, 1e-2),
            },
            "first_stage_over_1e_1": {
                "dtype_only": _first_stage_over(dtype_only, 1e-1),
                "vllm_default_vs_jax_bf16": {
                    "stage": None,
                    "threshold": 1e-1,
                    "note": "skip_vllm",
                },
                "vllm_fp32_qkv_fp32_solve_vs_jax_fp32": {
                    "stage": None,
                    "threshold": 1e-1,
                    "note": "skip_vllm",
                },
                "vllm_default_vs_vllm_like": _first_stage_over(default_vs_vllm_like, 1e-1),
            },
            "stage_diffs": {
                "dtype_only_jax_bf16_vs_jax_fp32": dtype_only,
                "vllm_default_vs_jax_bf16": {
                    "status": "skip_vllm",
                },
                "vllm_fp32_qkv_fp32_solve_vs_jax_fp32": {
                    "status": "skip_vllm",
                },
                "vllm_default_vs_vllm_fp32_qkv": {
                    "status": "skip_vllm",
                },
                "vllm_default_vs_vllm_like": default_vs_vllm_like,
            },
            "prefix_only": {
                "note": "skip_vllm",
            },
            "vllm_staged_default_vs_full_chunk_call": {},
        }

    vllm_default = _run_vllm_chain(
        case,
        torch,
        cumsum_mod,
        kkt_mod,
        solve_mod,
        wy_mod,
        dh_mod,
        out_mod,
        chunk_mod,
        qkv_torch_dtype="bfloat16",
        solve_output_dtype="bfloat16",
    )
    vllm_fp32_qkv_fp32_solve = _run_vllm_chain(
        case,
        torch,
        cumsum_mod,
        kkt_mod,
        solve_mod,
        wy_mod,
        dh_mod,
        out_mod,
        chunk_mod,
        qkv_torch_dtype="float32",
        solve_output_dtype="float32",
    )
    vllm_bf16_qkv_fp32_solve_prefix = _run_vllm_solve_only(
        case,
        torch,
        cumsum_mod,
        kkt_mod,
        solve_mod,
        qkv_torch_dtype="bfloat16",
        solve_output_dtype="float32",
    )

    default_vs_bf16 = _stage_diffs(vllm_default, jax_bf16)
    fp32qkv_vs_fp32 = _stage_diffs(vllm_fp32_qkv_fp32_solve, jax_fp32)
    default_vs_vllm_like = _stage_diffs(vllm_default, vllm_like_bf16)
    default_vs_fp32qkv = _stage_diffs(vllm_default, vllm_fp32_qkv_fp32_solve)

    full_match = {}
    if "full_output" in vllm_default:
        full_match = {
            "output": _diff(vllm_default["output"], vllm_default["full_output"]),
            "final_state": _diff(vllm_default["final_state"], vllm_default["full_state"]),
        }

    return {
        "length": length,
        "input_contract": {
            "case_packed_signature": case_packed_signature,
            "vllm_input_signature": vllm_default["input_signature"],
            "jax_vllm_like_signature": vllm_like_bf16["input_signature"],
            "query_bf16_input_matches": {
                "query_vllm_input": _signature_match(
                    vllm_default["input_signature"],
                    vllm_like_bf16["input_signature"],
                    "case_query_bf16",
                    "query",
                ),
                "key_vllm_input": _signature_match(
                    vllm_default["input_signature"],
                    vllm_like_bf16["input_signature"],
                    "case_key_bf16",
                    "key",
                ),
                "value_vllm_input": _signature_match(
                    vllm_default["input_signature"],
                    vllm_like_bf16["input_signature"],
                    "case_value_bf16",
                    "value",
                ),
                "gate_vllm_to_vllm_like": _signature_match(
                    vllm_default["input_signature"],
                    vllm_like_bf16["input_signature"],
                    "gate",
                    "case_gate",
                ),
                "beta_vllm_to_vllm_like": _signature_match(
                    vllm_default["input_signature"],
                    vllm_like_bf16["input_signature"],
                    "beta",
                    "case_beta",
                ),
                "initial_state_vllm_to_vllm_like": _signature_match(
                    vllm_default["input_signature"],
                    vllm_like_bf16["input_signature"],
                    "initial_state",
                    "case_initial_state",
                ),
                "query_packed_matches_case_prep": _signature_match(
                    vllm_like_bf16["input_signature"],
                    case_packed_signature,
                    "query_bf16_packed",
                    "packed_query",
                ),
                "key_packed_matches_case_prep": _signature_match(
                    vllm_like_bf16["input_signature"],
                    case_packed_signature,
                    "key_bf16_packed",
                    "packed_key",
                ),
                "value_packed_matches_case_prep": _signature_match(
                    vllm_like_bf16["input_signature"],
                    case_packed_signature,
                    "value_bf16_packed",
                    "packed_value",
                ),
                "gate_packed_matches_case_prep": _signature_match(
                    vllm_like_bf16["input_signature"],
                    case_packed_signature,
                    "case_gate",
                    "packed_gate",
                ),
                "beta_packed_matches_case_prep": _signature_match(
                    vllm_like_bf16["input_signature"],
                    case_packed_signature,
                    "case_beta",
                    "packed_beta",
                ),
            },
            "signature_match_note": "Compare fingerprints are exact for quantized FP32 snapshots of inputs. `case_*_bf16` are case tensors quantized with bfloat16; packed signatures come from prepare+pack path.",
        },
        "tensor_contract": {
            "packed_q_shape": list(jax_fp32["packed_query"].shape),
            "packed_v_shape": list(jax_fp32["packed_value"].shape),
            "state_shape": list(jax_fp32["final_state"].shape),
            "qkv_vllm_dtype": "bfloat16",
            "gate_beta_state_dtype": "float32",
        },
        "summary": {
            "dtype_only_output": dtype_only["output"],
            "dtype_only_state": dtype_only["final_state"],
            "vllm_default_vs_jax_bf16_output": default_vs_bf16["output"],
            "vllm_default_vs_jax_bf16_state": default_vs_bf16["final_state"],
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32_output": fp32qkv_vs_fp32["output"],
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32_state": fp32qkv_vs_fp32["final_state"],
            "vllm_default_vs_fp32_output": _diff(vllm_default["output"], jax_fp32["output"]),
            "vllm_default_vs_fp32_state": _diff(vllm_default["final_state"], jax_fp32["final_state"]),
            "vllm_default_vs_vllm_fp32_qkv_output": default_vs_fp32qkv["output"],
            "vllm_default_vs_vllm_fp32_qkv_state": default_vs_fp32qkv["final_state"],
            "vllm_default_vs_vllm_like_output": default_vs_vllm_like["output"],
            "vllm_default_vs_vllm_like_state": default_vs_vllm_like["final_state"],
            "vllm_bf16_qkv_fp32_solve_inverse_vs_jax_bf16": _diff(
                vllm_bf16_qkv_fp32_solve_prefix["attention_inverse"],
                jax_bf16["attention_inverse"],
            ),
        },
        "first_stage_over_1e_2": {
            "dtype_only": _first_stage_over(dtype_only, 1e-2),
            "vllm_default_vs_jax_bf16": _first_stage_over(default_vs_bf16, 1e-2),
            "vllm_default_vs_vllm_like": _first_stage_over(default_vs_vllm_like, 1e-2),
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32": _first_stage_over(
                fp32qkv_vs_fp32, 1e-2
            ),
        },
        "first_stage_over_1e_1": {
            "dtype_only": _first_stage_over(dtype_only, 1e-1),
            "vllm_default_vs_jax_bf16": _first_stage_over(default_vs_bf16, 1e-1),
            "vllm_default_vs_vllm_like": _first_stage_over(default_vs_vllm_like, 1e-1),
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32": _first_stage_over(
                fp32qkv_vs_fp32, 1e-1
            ),
        },
        "stage_diffs": {
            "dtype_only_jax_bf16_vs_jax_fp32": dtype_only,
            "vllm_default_vs_jax_bf16": default_vs_bf16,
            "vllm_fp32_qkv_fp32_solve_vs_jax_fp32": fp32qkv_vs_fp32,
            "vllm_default_vs_vllm_fp32_qkv": default_vs_fp32qkv,
            "vllm_default_vs_vllm_like": default_vs_vllm_like,
        },
        "prefix_only": {
            "vllm_bf16_qkv_fp32_solve_vs_jax_bf16": {
                "gate_cumsum": _diff(
                    vllm_bf16_qkv_fp32_solve_prefix["gate_cumsum"],
                    jax_bf16["gate_cumsum"],
                ),
                "attention_matrix": _diff(
                    vllm_bf16_qkv_fp32_solve_prefix["attention_matrix"],
                    jax_bf16["attention_matrix"],
                ),
                "attention_inverse": _diff(
                    vllm_bf16_qkv_fp32_solve_prefix["attention_inverse"],
                    jax_bf16["attention_inverse"],
                ),
            },
            "note": (
                "BF16 QKV + FP32 solve can run the vLLM staged path, but this does not test solve-output"
                "-quantization behavior. Use vllm_like for a closer BF16-stage comparison."
            ),
        },
        "vllm_staged_default_vs_full_chunk_call": full_match,
    }


def _summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# vLLM FLA Correctness Boundary Audit",
        "",
        f"- Model: `{payload['model']}`",
        f"- Layer: `{payload['layer_idx']}`",
        f"- Chunk size: `{payload['chunk_size']}`",
        "",
    ]
    for row in payload["results"]:
        lines.append(f"## Length {row['length']}")
        lines.append("")
        contract = row["tensor_contract"]
        lines.append(
            f"- Contract: packed Q `{contract['packed_q_shape']}`, packed V `{contract['packed_v_shape']}`, "
            f"state `{contract['state_shape']}`"
        )
        for label, firsts in row["first_stage_over_1e_1"].items():
            stage = firsts["stage"]
            max_abs = firsts.get("max_abs")
            if stage is None:
                lines.append(f"- First stage over `0.1` for `{label}`: none")
            else:
                lines.append(
                f"- First stage over `0.1` for `{label}`: `{stage}` "
                    f"(max `{max_abs:.6g}` at `{firsts.get('max_index')}`)"
                )
        lines.append("")
        lines.append("| comparison | output max | state max |")
        lines.append("|---|---:|---:|")
        s = row["summary"]
        lines.append(
            f"| JAX BF16-QKV vs JAX FP32-QKV | {_metric_value(s['dtype_only_output'])} | {_metric_value(s['dtype_only_state'])} |"
        )
        lines.append(
            f"| vLLM default vs JAX BF16-QKV | {_metric_value(s['vllm_default_vs_jax_bf16_output'])} | {_metric_value(s['vllm_default_vs_jax_bf16_state'])} |"
        )
        lines.append(
            f"| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | {_metric_value(s['vllm_fp32_qkv_fp32_solve_vs_jax_fp32_output'])} | {_metric_value(s['vllm_fp32_qkv_fp32_solve_vs_jax_fp32_state'])} |"
        )
        lines.append(
            f"| vLLM default vs JAX FP32-QKV | {_metric_value(s['vllm_default_vs_fp32_output'])} | {_metric_value(s['vllm_default_vs_fp32_state'])} |"
        )
        lines.append(
            f"| vLLM default vs vLLM FP32-QKV/FP32-solve | {_metric_value(s['vllm_default_vs_vllm_fp32_qkv_output'])} | {_metric_value(s['vllm_default_vs_vllm_fp32_qkv_state'])} |"
        )
        lines.append(
            f"| vLLM default vs vLLM-like BF16 path | {_metric_value(s['vllm_default_vs_vllm_like_output'])} | {_metric_value(s['vllm_default_vs_vllm_like_state'])} |"
        )
        inv = s["vllm_bf16_qkv_fp32_solve_inverse_vs_jax_bf16"]
        lines.append(
            f"| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | {_metric_value(inv)} | n/a |"
        )
        input_matches = row.get("input_contract", {}).get("query_bf16_input_matches", {})
        if input_matches:
            if "query_vllm_input" in input_matches:
                lines.append(
                    "- input fingerprint matches (vLLM vs vLLM-like): "
                    f"q={input_matches.get('query_vllm_input', False)}, "
                    f"k={input_matches.get('key_vllm_input', False)}, "
                    f"v={input_matches.get('value_vllm_input', False)}"
                )
                lines.append(
                    "- packed fingerprint matches vs case prepare+pack: "
                    f"q={input_matches.get('query_packed_matches_case_prep', False)}, "
                    f"k={input_matches.get('key_packed_matches_case_prep', False)}, "
                    f"v={input_matches.get('value_packed_matches_case_prep', False)}, "
                    f"gate={input_matches.get('gate_packed_matches_case_prep', False)}, "
                    f"beta={input_matches.get('beta_packed_matches_case_prep', False)}"
                )
            elif "query_vllm_like_to_case_packed" in input_matches:
                lines.append(
                    "- input fingerprint matches (vLLM-like vs case prepare+pack): "
                    f"q={input_matches.get('query_vllm_like_to_case_packed', False)}, "
                    f"k={input_matches.get('key_vllm_like_to_case_packed', False)}, "
                    f"v={input_matches.get('value_vllm_like_to_case_packed', False)}"
                )
        full = row.get("vllm_staged_default_vs_full_chunk_call", {})
        if full:
            lines.append(
                f"| vLLM staged default vs full chunk call | {full['output']['max_abs']:.6g} | {full['final_state']['max_abs']:.6g} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths:
        raise argparse.ArgumentTypeError("expected at least one length")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--lengths", type=_parse_lengths, default=_parse_lengths("64,128,256,512"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, choices=(16, 32, 64))
    parser.add_argument("--output-json", default=str(DEFAULT_OUT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="run only local JAX comparisons; do not import/load the vLLM CUDA path",
    )
    return parser.parse_args()


def main() -> int:
    global CHUNK_SIZE
    _configure_runtime_env()
    args = parse_args()
    CHUNK_SIZE = int(args.chunk_size)
    if args.skip_vllm:
        from nanovllm_jax.config import Qwen3_5Config
        from nanovllm_jax.load_weights import load_weights_from_hf
        from transformers import AutoTokenizer

        config = Qwen3_5Config.qwen3_5_0_8b()
        config.dtype = "float32"
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        params = load_weights_from_hf(args.model, Qwen3_5Config.qwen3_5_0_8b(), verbose=False)
        runtime = None
    else:
        (
            config,
            tokenizer,
            params,
            runtime_torch,
            cumsum_mod,
            kkt_mod,
            solve_mod,
            wy_mod,
            dh_mod,
            out_mod,
            chunk_mod,
        ) = _load_runtime(args.model)
        runtime = runtime_torch, cumsum_mod, kkt_mod, solve_mod, wy_mod, dh_mod, out_mod, chunk_mod

    rows = []
    for length in args.lengths:
        print(f"running length {length}", flush=True)
        if args.skip_vllm:
            rows.append(
                _run_length(
                    length=length,
                    params=params,
                    config=config,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    layer_idx=args.layer_idx,
                    torch=None,
                    cumsum_mod=None,
                    kkt_mod=None,
                    solve_mod=None,
                    wy_mod=None,
                    dh_mod=None,
                    out_mod=None,
                    chunk_mod=None,
                    skip_vllm=True,
                )
            )
        else:
            torch, cumsum_mod, kkt_mod, solve_mod, wy_mod, dh_mod, out_mod, chunk_mod = runtime
            rows.append(
                _run_length(
                    length=length,
                    params=params,
                    config=config,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    layer_idx=args.layer_idx,
                    torch=torch,
                    cumsum_mod=cumsum_mod,
                    kkt_mod=kkt_mod,
                    solve_mod=solve_mod,
                    wy_mod=wy_mod,
                    dh_mod=dh_mod,
                    out_mod=out_mod,
                    chunk_mod=chunk_mod,
                    skip_vllm=False,
                )
            )

    payload = {
        "date_utc": "2026-06-01",
        "model": args.model,
        "layer_idx": args.layer_idx,
        "chunk_size": CHUNK_SIZE,
        "lengths": args.lengths,
        "weight_dtype": "bfloat16",
        "activation_dtype": "float32",
        "skip_vllm": bool(args.skip_vllm),
        "results": rows,
        "interpretation": {
            "same_kernel_check": "vLLM staged default and vLLM full chunk call should match exactly or near-exactly.",
            "dtype_only": "JAX BF16-QKV vs JAX FP32-QKV isolates input activation dtype under the same local algorithm.",
            "schedule_drift": "vLLM default vs JAX BF16-QKV isolates vLLM schedule/kernel numerics after aligning layout and QKV dtype.",
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_summary_md(payload), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
