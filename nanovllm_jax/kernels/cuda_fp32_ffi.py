"""Local CUDA/JAX FFI prototypes for FP32 serving kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from nanovllm_jax.kernels.flashinfer_ffi import (
    kv_append_paged_nhd_reference,
)
from nanovllm_jax.kernels.paged_attention import (
    paged_decode_attention_gqa_nhd_reference,
)
from nanovllm_jax.kernels.registry import backend_status

_TARGET_KV_APPEND = "nanovllm_jax_fp32_kv_append_paged_nhd"
_TARGET_PAGED_DECODE = "nanovllm_jax_fp32_paged_decode_attention_gqa_nhd"
_TARGET_GDN_DECODE = "nanovllm_jax_fp32_gdn_recurrent_decode"
_TARGET_GDN_PACKED_DECODE = "nanovllm_jax_fp32_gdn_packed_decode"
_TARGET_GDN_PREFILL = "nanovllm_jax_fp32_gdn_prefill_chunk32"
_TARGET_GDN_PREFILL_V64 = "nanovllm_jax_fp32_gdn_prefill_chunk32_v64"
_REGISTER_LOCK = threading.Lock()
_REGISTERED = False
_LOADED_LIBS: list[ctypes.CDLL] = []


def availability():
    return backend_status("cuda_fp32")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_root() -> Path:
    configured = os.getenv("NANO_VLLM_JAX_CACHE_ROOT")
    if configured:
        return Path(configured)
    mountpoint = Path("/mountpoint/.exp")
    if mountpoint.exists():
        return mountpoint
    return Path.cwd()


def _build_dir() -> Path:
    return _runtime_root() / ".cache" / "nano-vllm-jax" / "cuda_fp32"


def _cuda_root_from_site_packages() -> Path | None:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return None
    candidate = Path(purelib) / "nvidia" / "cu13"
    return candidate if candidate.exists() else None


def _nvcc_path() -> Path:
    configured = os.getenv("NANO_VLLM_JAX_NVCC")
    if configured:
        return Path(configured)
    cuda_root = _cuda_root_from_site_packages()
    if cuda_root is not None:
        candidate = cuda_root / "bin" / "nvcc"
        if candidate.exists():
            return candidate
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    raise RuntimeError(
        "nvcc was not found; set NANO_VLLM_JAX_NVCC or install the cuda13 extra"
    )


def _cuda_root_from_nvcc(nvcc: Path) -> Path:
    return nvcc.resolve().parents[1]


def _shared_library_path() -> Path:
    ext = ".dll" if sys.platform == "win32" else ".so"
    return _build_dir() / f"libnano_vllm_jax_cuda_fp32_{_cuda_arch()}{ext}"


def _source_path() -> Path:
    return _repo_root() / "nanovllm_jax" / "kernels" / "csrc" / "fp32_kv_append.cu"


def _cuda_arch() -> str:
    return os.getenv("NANO_VLLM_JAX_CUDA_ARCH", "sm_86").strip()


def _cuda_arch_flags() -> list[str]:
    arch = _cuda_arch()
    if not arch.startswith("sm_"):
        raise ValueError(
            "NANO_VLLM_JAX_CUDA_ARCH must use an sm_XX architecture string"
        )
    compute = "compute_" + arch.split("_", 1)[1]
    return [f"--generate-code=arch={compute},code={arch}"]


def _needs_rebuild(output_path: Path, source_path: Path) -> bool:
    if os.getenv("NANO_VLLM_JAX_FORCE_CUDA_FFI_REBUILD", "0") in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    if not output_path.exists():
        return True
    return source_path.stat().st_mtime > output_path.stat().st_mtime


def build_cuda_fp32_kernels() -> Path:
    """Build the local FP32 CUDA FFI shared object under the runtime cache."""

    output_path = _shared_library_path()
    source_path = _source_path()
    if not _needs_rebuild(output_path, source_path):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nvcc = _nvcc_path()
    cuda_root = _cuda_root_from_nvcc(nvcc)
    include_dir = jax.ffi.include_dir()
    command = [
        str(nvcc),
        "-std=c++17",
        "-O3",
        "-shared",
        *_cuda_arch_flags(),
        "-Xcompiler",
        "-fPIC",
        str(source_path),
        "-o",
        str(output_path),
        "-I",
        include_dir,
        "-I",
        str(cuda_root / "include"),
        "-L",
        str(cuda_root / "lib"),
        "-lcudart",
    ]
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to build local FP32 CUDA FFI kernels with nvcc:\n"
            + completed.stdout[-12000:]
        )
    return output_path


def _register_kv_append_target() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        library_path = build_cuda_fp32_kernels()
        library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        handler = library.NanoVllmJaxFp32KvAppend
        decode_handler = library.NanoVllmJaxFp32PagedDecodeAttention
        gdn_decode_handler = library.NanoVllmJaxFp32GdnRecurrentDecode
        gdn_packed_decode_handler = library.NanoVllmJaxFp32GdnPackedDecode
        gdn_prefill_handler = library.NanoVllmJaxFp32GdnPrefillChunk32
        gdn_prefill_v64_handler = library.NanoVllmJaxFp32GdnPrefillChunk32V64
        jax.ffi.register_ffi_target(
            _TARGET_KV_APPEND,
            jax.ffi.pycapsule(handler),
            platform="gpu",
            api_version=1,
        )
        jax.ffi.register_ffi_target(
            _TARGET_PAGED_DECODE,
            jax.ffi.pycapsule(decode_handler),
            platform="gpu",
            api_version=1,
        )
        jax.ffi.register_ffi_target(
            _TARGET_GDN_DECODE,
            jax.ffi.pycapsule(gdn_decode_handler),
            platform="gpu",
            api_version=1,
        )
        jax.ffi.register_ffi_target(
            _TARGET_GDN_PACKED_DECODE,
            jax.ffi.pycapsule(gdn_packed_decode_handler),
            platform="gpu",
            api_version=1,
        )
        jax.ffi.register_ffi_target(
            _TARGET_GDN_PREFILL,
            jax.ffi.pycapsule(gdn_prefill_handler),
            platform="gpu",
            api_version=1,
        )
        jax.ffi.register_ffi_target(
            _TARGET_GDN_PREFILL_V64,
            jax.ffi.pycapsule(gdn_prefill_v64_handler),
            platform="gpu",
            api_version=1,
        )
        _LOADED_LIBS.append(library)
        _REGISTERED = True


def _as_jax_array(name: str, value: Any) -> jnp.ndarray:
    try:
        return jnp.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive type error path.
        raise TypeError(f"{name} must be array-like") from exc


def _validate_fp32_kv_append_inputs(
    append_key: jnp.ndarray,
    append_value: jnp.ndarray,
    batch_indices: jnp.ndarray,
    positions: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    kv_indices: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_last_page_len: jnp.ndarray,
    valid_mask: jnp.ndarray | None = None,
) -> None:
    if append_key.dtype != jnp.float32 or append_value.dtype != jnp.float32:
        raise ValueError("append_key and append_value must be float32")
    if k_cache.dtype != jnp.float32 or v_cache.dtype != jnp.float32:
        raise ValueError("k_cache and v_cache must be float32")
    for name, value in (
        ("batch_indices", batch_indices),
        ("positions", positions),
        ("kv_indices", kv_indices),
        ("kv_indptr", kv_indptr),
        ("kv_last_page_len", kv_last_page_len),
    ):
        if value.dtype != jnp.int32:
            raise ValueError(f"{name} must have dtype int32")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if append_key.shape != append_value.shape:
        raise ValueError("append_key and append_value must have the same shape")
    if append_key.ndim != 3:
        raise ValueError(
            "append_key must have shape [nnz_tokens, num_kv_heads, head_dim]"
        )
    if k_cache.ndim != 4:
        raise ValueError(
            "k_cache must have NHD shape "
            "[num_pages, page_size, num_kv_heads, head_dim]"
        )
    if append_key.shape[1:] != k_cache.shape[2:]:
        raise ValueError(
            "append_key trailing dimensions must match cache "
            "[num_kv_heads, head_dim]"
        )
    nnz_tokens = append_key.shape[0]
    if batch_indices.shape != (nnz_tokens,) or positions.shape != (nnz_tokens,):
        raise ValueError(
            "batch_indices and positions must both have shape [nnz_tokens]"
        )
    if kv_indices.ndim != 1 or kv_indptr.ndim != 1 or kv_last_page_len.ndim != 1:
        raise ValueError("page metadata tensors must be rank-1")
    if valid_mask is not None:
        if valid_mask.dtype != jnp.int32:
            raise ValueError("valid_mask must have dtype int32")
        if valid_mask.shape != (nnz_tokens,):
            raise ValueError("valid_mask must have shape [nnz_tokens]")


def _kv_append_paged_nhd_fp32_call(
    append_key: Any,
    append_value: Any,
    batch_indices: Any,
    positions: Any,
    k_cache: Any,
    v_cache: Any,
    kv_indices: Any,
    kv_indptr: Any,
    kv_last_page_len: Any,
    valid_mask: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    append_key = _as_jax_array("append_key", append_key)
    append_value = _as_jax_array("append_value", append_value)
    batch_indices = _as_jax_array("batch_indices", batch_indices)
    positions = _as_jax_array("positions", positions)
    k_cache = _as_jax_array("k_cache", k_cache)
    v_cache = _as_jax_array("v_cache", v_cache)
    kv_indices = _as_jax_array("kv_indices", kv_indices)
    kv_indptr = _as_jax_array("kv_indptr", kv_indptr)
    kv_last_page_len = _as_jax_array("kv_last_page_len", kv_last_page_len)
    valid_mask = (
        None if valid_mask is None else _as_jax_array("valid_mask", valid_mask)
    )
    _validate_fp32_kv_append_inputs(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        valid_mask,
    )
    _register_kv_append_target()
    call = jax.ffi.ffi_call(
        _TARGET_KV_APPEND,
        (
            jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
            jax.ShapeDtypeStruct(v_cache.shape, v_cache.dtype),
        ),
        has_side_effect=True,
        input_output_aliases={4: 0, 5: 1},
    )
    args = (
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
    if valid_mask is not None:
        args = (*args, valid_mask)
    return call(*args)


def kv_append_paged_nhd_fp32(
    append_key: Any,
    append_value: Any,
    batch_indices: Any,
    positions: Any,
    k_cache: Any,
    v_cache: Any,
    kv_indices: Any,
    kv_indptr: Any,
    kv_last_page_len: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Append FP32 K/V into an NHD paged cache using a local CUDA FFI kernel."""

    return _kv_append_paged_nhd_fp32_call(
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


def _static_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a static Python integer") from exc


def _kv_last_page_len(seq_lens: jnp.ndarray, page_size: int) -> jnp.ndarray:
    seq_lens = seq_lens.astype(jnp.int32)
    return jnp.where(
        seq_lens > 0,
        ((seq_lens - 1) % jnp.asarray(page_size, dtype=jnp.int32)) + 1,
        0,
    ).astype(jnp.int32)


def kv_append_paged_nhd_fp32_from_metadata(
    k: jnp.ndarray,
    v: jnp.ndarray,
    k_cache_layer: jnp.ndarray,
    v_cache_layer: jnp.ndarray,
    metadata: Any,
    *,
    page_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Append rectangular scheduled FP32 K/V tensors using ragged metadata."""

    if metadata.positions is None:
        raise ValueError("metadata.positions is required for CUDA FP32 KV append")

    num_tokens = _static_int(k.shape[0] * k.shape[1], "scheduled token count")
    if num_tokens == 0:
        return k_cache_layer, v_cache_layer

    batch, query_len = k.shape[:2]
    query_lens = jnp.diff(metadata.query_start_loc).astype(jnp.int32)
    valid_mask = jnp.arange(query_len, dtype=jnp.int32)[None, :] < query_lens[:, None]
    row_idx = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), query_len)
    col_idx = jnp.tile(jnp.arange(query_len, dtype=jnp.int32), batch)
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
    return _kv_append_paged_nhd_fp32_call(
        append_key,
        append_value,
        batch_indices,
        positions,
        k_cache_layer,
        v_cache_layer,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        valid_mask.reshape(-1).astype(jnp.int32),
    )


def kv_append_paged_nhd_fp32_reference(*args: Any, **kwargs: Any):
    return kv_append_paged_nhd_reference(*args, **kwargs)


def _validate_fp32_paged_decode_inputs(
    q: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_indices: jnp.ndarray,
    kv_last_page_len: jnp.ndarray,
    seq_lens: jnp.ndarray,
    softmax_scale: jnp.ndarray,
) -> None:
    if q.dtype != jnp.float32:
        raise ValueError("q must be float32")
    if k_cache.dtype != jnp.float32 or v_cache.dtype != jnp.float32:
        raise ValueError("k_cache and v_cache must be float32")
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
    for name, value in (
        ("kv_indptr", kv_indptr),
        ("kv_indices", kv_indices),
        ("kv_last_page_len", kv_last_page_len),
        ("seq_lens", seq_lens),
    ):
        if value.dtype != jnp.int32:
            raise ValueError(f"{name} must have dtype int32")
        if value.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
    if kv_last_page_len.shape != (batch,) or seq_lens.shape != (batch,):
        raise ValueError("kv_last_page_len and seq_lens must have shape [batch]")
    if kv_indices.shape[0] % batch != 0:
        raise ValueError("kv_indices must be dense by batch for this prototype")
    if softmax_scale.dtype != jnp.float32 or softmax_scale.shape != ():
        raise ValueError("softmax_scale must be a float32 scalar")


def paged_decode_attention_gqa_nhd_fp32(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    kv_indptr: Any,
    kv_indices: Any,
    kv_last_page_len: Any,
    seq_lens: Any,
    softmax_scale: Any,
) -> jnp.ndarray:
    """Decode attention over an FP32 NHD paged cache using local CUDA FFI."""

    q = _as_jax_array("q", q)
    k_cache = _as_jax_array("k_cache", k_cache)
    v_cache = _as_jax_array("v_cache", v_cache)
    kv_indptr = _as_jax_array("kv_indptr", kv_indptr)
    kv_indices = _as_jax_array("kv_indices", kv_indices)
    kv_last_page_len = _as_jax_array("kv_last_page_len", kv_last_page_len)
    seq_lens = _as_jax_array("seq_lens", seq_lens)
    softmax_scale_array = jnp.asarray(softmax_scale, dtype=jnp.float32)
    _validate_fp32_paged_decode_inputs(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
        softmax_scale_array,
    )
    _register_kv_append_target()
    call = jax.ffi.ffi_call(
        _TARGET_PAGED_DECODE,
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        has_side_effect=False,
    )
    return call(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_lens,
        softmax_scale_array,
    )


def paged_decode_attention_gqa_nhd_fp32_reference(*args: Any, **kwargs: Any):
    return paged_decode_attention_gqa_nhd_reference(*args, **kwargs)


def _validate_fp32_gdn_decode_inputs(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: jnp.ndarray,
) -> None:
    for name, value_array, rank in (
        ("query", query, 4),
        ("key", key, 4),
        ("value", value, 4),
        ("g", g, 3),
        ("beta", beta, 3),
        ("state", state, 4),
    ):
        if value_array.dtype != jnp.float32:
            raise ValueError(f"{name} must be float32")
        if value_array.ndim != rank:
            raise ValueError(f"{name} must be rank-{rank}")
    if query.shape[:3] != key.shape[:3]:
        raise ValueError("query and key must have matching [batch, heads, time]")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value must have matching [batch, heads, time]")
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("g and beta must have shape [batch, heads, time]")
    if query.shape[2] != 1:
        raise ValueError("FP32 GDN decode prototype only supports time=1")
    if state.shape[:2] != query.shape[:2]:
        raise ValueError("state batch/head dimensions must match query")
    if state.shape[2] != value.shape[3] or state.shape[3] != query.shape[3]:
        raise ValueError("state must have shape [batch, heads, value_dim, key_dim]")


def gdn_recurrent_decode_step_fp32(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    state: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Width-1 GDN recurrent decode with native V,K FP32 state layout."""

    query = _as_jax_array("query", query)
    key = _as_jax_array("key", key)
    value = _as_jax_array("value", value)
    g = _as_jax_array("g", g)
    beta = _as_jax_array("beta", beta)
    state = _as_jax_array("state", state)
    _validate_fp32_gdn_decode_inputs(query, key, value, g, beta, state)
    _register_kv_append_target()
    call = jax.ffi.ffi_call(
        _TARGET_GDN_DECODE,
        (
            jax.ShapeDtypeStruct(value.shape, value.dtype),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
        ),
        has_side_effect=False,
    )
    return call(query, key, value, g, beta, state)


def gdn_recurrent_decode_step_fp32_reference(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    state: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    from nanovllm_jax.model import jax_recurrent_gated_delta_rule

    return jax_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
    )


def _validate_fp32_gdn_packed_decode_inputs(
    mixed_qkv: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    state: jnp.ndarray,
) -> tuple[int, int, int, int, int]:
    for name, value_array, rank in (
        ("mixed_qkv", mixed_qkv, 2),
        ("a", a, 2),
        ("b", b, 2),
        ("a_log", a_log, 1),
        ("dt_bias", dt_bias, 1),
        ("state", state, 4),
    ):
        if value_array.dtype != jnp.float32:
            raise ValueError(f"{name} must be float32")
        if value_array.ndim != rank:
            raise ValueError(f"{name} must be rank-{rank}")
    batch, num_value_heads, value_dim, key_dim = state.shape
    if mixed_qkv.shape[0] != batch:
        raise ValueError("mixed_qkv batch must match state batch")
    if a.shape != (batch, num_value_heads) or b.shape != (batch, num_value_heads):
        raise ValueError("a and b must have shape [batch, value_heads]")
    if a_log.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("a_log and dt_bias must have shape [value_heads]")
    qk_dim = mixed_qkv.shape[1] - num_value_heads * value_dim
    if qk_dim <= 0 or qk_dim % (2 * key_dim) != 0:
        raise ValueError("mixed_qkv has an invalid packed Q/K dimension")
    num_q_heads = qk_dim // (2 * key_dim)
    if num_value_heads % num_q_heads != 0:
        raise ValueError("value head count must be divisible by packed Q/K head count")
    return batch, num_q_heads, num_value_heads, value_dim, key_dim


def gdn_packed_decode_step_fp32(
    mixed_qkv: Any,
    a: Any,
    b: Any,
    a_log: Any,
    dt_bias: Any,
    state: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """vLLM-style packed width-1 GDN decode using local V,K FP32 state layout."""

    mixed_qkv = _as_jax_array("mixed_qkv", mixed_qkv)
    a = _as_jax_array("a", a)
    b = _as_jax_array("b", b)
    a_log = _as_jax_array("a_log", a_log)
    dt_bias = _as_jax_array("dt_bias", dt_bias)
    state = _as_jax_array("state", state)
    batch, _, num_value_heads, value_dim, _ = _validate_fp32_gdn_packed_decode_inputs(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
    )
    _register_kv_append_target()
    call = jax.ffi.ffi_call(
        _TARGET_GDN_PACKED_DECODE,
        (
            jax.ShapeDtypeStruct(
                (batch, num_value_heads, 1, value_dim),
                state.dtype,
            ),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
        ),
        has_side_effect=False,
    )
    return call(mixed_qkv, a, b, a_log, dt_bias, state)


def gdn_packed_decode_step_fp32_reference(*args: Any, **kwargs: Any):
    from nanovllm_jax.kernels.gdn_fla import gdn_packed_decode_reference_local_state

    return gdn_packed_decode_reference_local_state(*args, **kwargs)


def _validate_fp32_gdn_prefill_inputs(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    seq_lens: jnp.ndarray,
    state: jnp.ndarray,
    *,
    value_dim_multiple: int = 32,
) -> None:
    for name, value_array, rank in (
        ("query", query, 4),
        ("key", key, 4),
        ("value", value, 4),
        ("g", g, 3),
        ("beta", beta, 3),
        ("state", state, 4),
    ):
        if value_array.dtype != jnp.float32:
            raise ValueError(f"{name} must be float32")
        if value_array.ndim != rank:
            raise ValueError(f"{name} must be rank-{rank}")
    if seq_lens.dtype != jnp.int32:
        raise ValueError("seq_lens must be int32")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be rank-1")
    if query.shape != key.shape:
        raise ValueError("query and key must have matching shape")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value must have matching [batch, heads, time]")
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("g and beta must have shape [batch, heads, time]")
    if seq_lens.shape != (query.shape[0],):
        raise ValueError("seq_lens must have shape [batch]")
    if query.shape[2] % 32 != 0:
        raise ValueError("FP32 GDN prefill prototype requires time divisible by 32")
    if value.shape[3] % value_dim_multiple != 0:
        raise ValueError(
            "FP32 GDN prefill prototype requires value_dim divisible by "
            f"{value_dim_multiple}"
        )
    if state.shape != (query.shape[0], query.shape[1], value.shape[3], query.shape[3]):
        raise ValueError("state must have shape [batch, heads, value_dim, key_dim]")


def _gdn_prefill_chunk32_normalized_fp32_target(
    target: str,
    value_dim_multiple: int,
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    seq_lens: Any,
    state: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    query = _as_jax_array("query", query)
    key = _as_jax_array("key", key)
    value = _as_jax_array("value", value)
    g = _as_jax_array("g", g)
    beta = _as_jax_array("beta", beta)
    seq_lens = _as_jax_array("seq_lens", seq_lens)
    state = _as_jax_array("state", state)
    _validate_fp32_gdn_prefill_inputs(
        query,
        key,
        value,
        g,
        beta,
        seq_lens,
        state,
        value_dim_multiple=value_dim_multiple,
    )
    _register_kv_append_target()
    call = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(value.shape, value.dtype),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
        ),
        has_side_effect=False,
    )
    return call(query, key, value, g, beta, seq_lens, state)


def gdn_prefill_chunk32_normalized_fp32(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    seq_lens: Any,
    state: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Chunk-32 GDN prefill for already-normalized FP32 q/k tensors.

    The caller owns the model contract: query is L2-normalized and scaled by
    1/sqrt(key_dim), key is L2-normalized, and value/g/beta/state are FP32.
    This is intentionally a benchmark-facing prototype rather than a server
    default.
    """

    return _gdn_prefill_chunk32_normalized_fp32_target(
        _TARGET_GDN_PREFILL,
        32,
        query,
        key,
        value,
        g,
        beta,
        seq_lens,
        state,
    )


def gdn_prefill_chunk32_v64_normalized_fp32(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    seq_lens: Any,
    state: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Chunk-32 GDN prefill prototype using 64 value columns per CUDA block."""

    return _gdn_prefill_chunk32_normalized_fp32_target(
        _TARGET_GDN_PREFILL_V64,
        64,
        query,
        key,
        value,
        g,
        beta,
        seq_lens,
        state,
    )
