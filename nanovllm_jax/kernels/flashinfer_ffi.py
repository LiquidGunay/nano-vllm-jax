"""FlashInfer/JAX FFI wrappers for promoted full-attention decode."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from nanovllm_jax.kernels import KernelUnavailable, missing_modules, require_modules

_APPEND_PAGED_KV_CACHE_TARGET = "nanovllm_jax_flashinfer_append_paged_kv_cache"
_RADIX_TOPK_TARGET = "nanovllm_jax_flashinfer_radix_topk"
_BATCH_DECODE_TARGET_PREFIX = "nanovllm_jax_flashinfer_batch_decode_jax_plan"
_BATCH_DECODE_FUSED_APPEND_TARGET_PREFIX = (
    "nanovllm_jax_flashinfer_batch_decode_fused_append_jax_plan"
)
_NHD_LAYOUT = 0
_SUPPORTED_KV_APPEND_DTYPES = (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))
_SUPPORTED_BATCH_DECODE_DTYPES = (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))
_REGISTER_LOCK = threading.Lock()
_APPEND_PAGED_KV_CACHE_REGISTERED = False
_RADIX_TOPK_REGISTERED = False
_BATCH_DECODE_TARGETS: dict[tuple[str, int], str] = {}
_BATCH_DECODE_FUSED_APPEND_TARGETS: dict[tuple[str, int], str] = {}
_BATCH_DECODE_MODULES: dict[tuple[str, int], Any] = {}
_BATCH_DECODE_PLANS: dict[
    tuple[str, int, int, int, int, int, int],
    tuple[tuple[int, ...], Any, int],
] = {}
_FLASHINFER_NEW_CUB_FLAG = "-DFLASHINFER_CUB_SUBTRACTLEFT_DEFINED"
_FLASHINFER_FLOAT_WORKSPACE_BYTES = 128 * 1024 * 1024
_FLASHINFER_INT_WORKSPACE_BYTES = 8 * 1024 * 1024
_FLASHINFER_BATCH_DECODE_PLAN_FIELDS = 10


def availability() -> dict[str, object]:
    required = ("flashinfer", "jax_tvm_ffi")
    missing = missing_modules(required)
    return {
        "feature": "flashinfer",
        "available": not missing,
        "missing_modules": missing,
    }


def require_available() -> None:
    require_modules(("flashinfer", "jax_tvm_ffi"), "FlashInfer FFI")


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
    extra_cuda_flags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if _FLASHINFER_NEW_CUB_FLAG not in extra_cuda_flags.split():
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (
            f"{extra_cuda_flags} {_FLASHINFER_NEW_CUB_FLAG}".strip()
        )
    Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FLASHINFER_CUBIN_DIR"]).mkdir(parents=True, exist_ok=True)
    _configure_flashinfer_cuda_arch()


def _configure_flashinfer_cuda_arch() -> None:
    if os.environ.get("FLASHINFER_CUDA_ARCH_LIST"):
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pynvml.nvmlShutdown()
    except Exception:
        return
    if major > 0:
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", f"{major}.{minor}")


def _require_flashinfer_modules() -> None:
    require_available()


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


def _register_radix_topk() -> None:
    global _RADIX_TOPK_REGISTERED
    if _RADIX_TOPK_REGISTERED:
        return

    with _REGISTER_LOCK:
        if _RADIX_TOPK_REGISTERED:
            return
        _configure_flashinfer_cache()
        from flashinfer.jit.topk import gen_topk_module
        from jax_tvm_ffi import register_ffi_target

        module = gen_topk_module().build_and_load()
        register_ffi_target(
            _RADIX_TOPK_TARGET,
            module.radix_topk,
            arg_spec=[
                "args",
                "attrs.top_k",
                "attrs.sorted_output",
                "attrs.deterministic",
                "attrs.tie_break",
                "attrs.dsa_graph_safe",
            ],
            platform="gpu",
            allow_cuda_graph=True,
        )
        _RADIX_TOPK_REGISTERED = True


def _dtype_key(dtype: jnp.dtype) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.dtype(jnp.bfloat16):
        return "bf16"
    if dtype == jnp.dtype(jnp.float16):
        return "fp16"
    raise ValueError(f"unsupported FlashInfer dtype {dtype}")


def _torch_dtype_for_key(key: str):
    import torch

    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    raise ValueError(f"unsupported FlashInfer dtype key {key!r}")


def _write_batch_decode_jax_plan_binding(path: Path) -> None:
    from flashinfer.jit.core import write_if_different

    plan_args = ",\n                                    ".join(
        f"int64_t plan_{idx}" for idx in range(_FLASHINFER_BATCH_DECODE_PLAN_FIELDS)
    )
    plan_values = ", ".join(
        f"plan_{idx}" for idx in range(_FLASHINFER_BATCH_DECODE_PLAN_FIELDS)
    )
    source = f"""/*
 * JAX-facing FlashInfer batch decode binding.
 *
 * FlashInfer's stock TVM FFI run function takes plan_info as Array<int64_t>.
 * JAX FFI attributes do not lower Python tuples/lists to that TVM container, and
 * a JAX device tensor has the wrong ABI type. This thin wrapper accepts the ten
 * current DecodePlanInfo fields as scalar attrs and reconstructs the TVM Array
 * before forwarding to FlashInfer's generated implementation.
 */
#include <vector>

#include "batch_decode_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

void BatchDecodeWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                    TensorView int_workspace_buffer,
                                    Array<int64_t> plan_info_vec,
                                    TensorView q,
                                    TensorView paged_k_cache,
                                    TensorView paged_v_cache,
                                    TensorView paged_kv_indptr,
                                    TensorView paged_kv_indices,
                                    TensorView paged_kv_last_page_len,
                                    TensorView o,
                                    Optional<TensorView> maybe_lse,
                                    int64_t kv_layout_code,
                                    int64_t window_left,
                                    bool enable_pdl ADDITIONAL_FUNC_PARAMS);

void BatchDecodeWithPagedKVCacheRunJaxPlan(
    TensorView float_workspace_buffer,
    TensorView int_workspace_buffer,
    TensorView q,
    TensorView paged_k_cache,
    TensorView paged_v_cache,
    TensorView paged_kv_indptr,
    TensorView paged_kv_indices,
    TensorView paged_kv_last_page_len,
    {plan_args},
    TensorView o,
    TensorView lse,
    int64_t kv_layout_code,
    int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS) {{
  std::vector<int64_t> plan_vec = {{{plan_values}}};
  Array<int64_t> plan_info_vec(plan_vec);
  Optional<TensorView> maybe_lse(lse);
  BatchDecodeWithPagedKVCacheRun(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q,
                                 paged_k_cache, paged_v_cache, paged_kv_indptr,
                                 paged_kv_indices, paged_kv_last_page_len, o, maybe_lse,
                                 kv_layout_code, window_left, enable_pdl,
                                 logits_soft_cap, sm_scale, rope_rcp_scale, rope_rcp_theta);
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_jax_plan, BatchDecodeWithPagedKVCacheRunJaxPlan);
"""
    write_if_different(path, source)


def _write_batch_decode_fused_append_jax_plan_binding(path: Path) -> None:
    from flashinfer.jit.core import write_if_different

    plan_args = ",\n    ".join(
        f"int64_t plan_{idx}" for idx in range(_FLASHINFER_BATCH_DECODE_PLAN_FIELDS)
    )
    plan_values = ", ".join(
        f"plan_{idx}" for idx in range(_FLASHINFER_BATCH_DECODE_PLAN_FIELDS)
    )
    source = f"""/*
 * JAX-facing fused append + FlashInfer batch decode binding.
 *
 * This accepts the repo's full cache layout
 * [layers, pages, page_size, kv_heads, head_dim], appends width-1 K/V for one
 * layer in-place, then runs FlashInfer decode attention over that layer. The
 * full cache tensors and output buffer are returned through JAX input/output
 * aliasing, matching the boundary shape of the existing Triton fused kernel.
 */
#include <vector>

#include <flashinfer/page.cuh>
#include <flashinfer/attention/scheduler.cuh>

#include "batch_decode_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;

namespace flashinfer {{

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream);

}}  // namespace flashinfer

using namespace flashinfer;

void BatchDecodeWithPagedKVCacheFusedAppendJaxPlan(
    TensorView float_workspace_buffer,
    TensorView int_workspace_buffer,
    TensorView q,
    TensorView append_key,
    TensorView append_value,
    TensorView paged_k_cache,
    TensorView paged_v_cache,
    TensorView paged_kv_indptr,
    TensorView paged_kv_indices,
    TensorView paged_kv_last_page_len,
    TensorView batch_indices,
    TensorView positions,
    TensorView o,
    {plan_args},
    int64_t layer_id,
    int64_t kv_layout_code,
    int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS) {{
  CHECK_INPUT_TYPE(paged_kv_indptr, dl_int32);
  CHECK_INPUT_TYPE(paged_kv_indices, dl_int32);
  CHECK_INPUT_TYPE(paged_kv_last_page_len, dl_int32);
  CHECK_INPUT_TYPE(batch_indices, dl_int32);
  CHECK_INPUT_TYPE(positions, dl_int32);
  CHECK_DIM(3, q);
  CHECK_DIM(3, append_key);
  CHECK_DIM(3, append_value);
  CHECK_DIM(5, paged_k_cache);
  CHECK_DIM(5, paged_v_cache);

  std::vector<int64_t> plan_vec = {{{plan_values}}};
  DecodePlanInfo plan_info;
  plan_info.FromVector(plan_vec);

  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t num_layers = paged_k_cache.size(0);
  TVM_FFI_ICHECK_GE(layer_id, 0);
  TVM_FFI_ICHECK_LT(layer_id, num_layers);
  TVM_FFI_ICHECK_EQ(paged_v_cache.size(0), num_layers);
  TVM_FFI_ICHECK_EQ(paged_kv_last_page_len.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(paged_kv_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(batch_indices.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(positions.size(0), batch_size);

  int64_t num_kv_heads, page_size;
  if (kv_layout == QKVLayout::kHND) {{
    num_kv_heads = paged_k_cache.size(2);
    page_size = paged_k_cache.size(3);
  }} else {{
    page_size = paged_k_cache.size(2);
    num_kv_heads = paged_k_cache.size(3);
  }}
  uint32_t head_dim_qk = q.size(2);
  uint32_t head_dim_vo = paged_v_cache.size(4);
  TVM_FFI_ICHECK_EQ(head_dim_qk, head_dim_vo);
  TVM_FFI_ICHECK_EQ(append_key.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(append_key.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(append_key.size(2), head_dim_qk);
  TVM_FFI_ICHECK_EQ(append_value.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(append_value.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(append_value.size(2), head_dim_vo);

  auto k_strides_full = paged_k_cache.strides();
  auto v_strides_full = paged_v_cache.strides();
  TVM_FFI_ICHECK_EQ(k_strides_full.size(), 5);
  TVM_FFI_ICHECK_EQ(v_strides_full.size(), 5);
  for (int i = 0; i < 5; ++i) {{
    TVM_FFI_ICHECK_EQ(k_strides_full[i], v_strides_full[i]);
  }}
  int64_t layer_strides[4] = {{
      k_strides_full[1],
      k_strides_full[2],
      k_strides_full[3],
      k_strides_full[4],
  }};

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DTypeKV* k_layer_ptr =
      static_cast<DTypeKV*>(paged_k_cache.data_ptr()) + layer_id * k_strides_full[0];
  DTypeKV* v_layer_ptr =
      static_cast<DTypeKV*>(paged_v_cache.data_ptr()) + layer_id * v_strides_full[0];
  paged_kv_t<DTypeKV, IdType> paged_kv(
        num_kv_heads, page_size, head_dim_qk, batch_size, kv_layout, k_layer_ptr,
        v_layer_ptr, layer_strides, static_cast<int32_t*>(paged_kv_indices.data_ptr()),
        static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
        static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));

  auto append_k_strides = append_key.strides();
  auto append_v_strides = append_value.strides();
  cudaError_t append_status = AppendPagedKVCache(
        paged_kv, static_cast<DTypeKV*>(append_key.data_ptr()),
        static_cast<DTypeKV*>(append_value.data_ptr()),
        static_cast<int32_t*>(batch_indices.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()), batch_size, append_k_strides[0],
        append_k_strides[1], append_v_strides[0], append_v_strides[1], stream);
  TVM_FFI_ICHECK(append_status == cudaSuccess)
        << "AppendPagedKVCache failed with error: " << cudaGetErrorString(append_status);

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());
  auto q_strides = q.strides();

  DISPATCH_context(
        DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
        USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {{
          Params params;
          params.q = static_cast<DTypeQ*>(q.data_ptr());
          params.paged_kv = paged_kv;
          params.o = static_cast<DTypeO*>(o.data_ptr());
          params.lse = nullptr;
          params.padded_batch_size = 0;
          params.num_qo_heads = num_qo_heads;
          params.q_stride_n = q_strides[0];
          params.q_stride_h = q_strides[1];
          params.window_left = window_left;
          params.request_indices = nullptr;
          params.kv_tile_indices = nullptr;
          params.o_indptr = nullptr;
          params.kv_chunk_size_ptr = nullptr;
          params.block_valid_mask = nullptr;
          params.partition_kv = false;
          ADDITIONAL_PARAMS_SETTER

          DTypeO* tmp_v = nullptr;
          float* tmp_s = nullptr;
          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
          params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
          if (plan_info.split_kv) {{
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
            if (plan_info.enable_cuda_graph) {{
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
            }}
          }}
          params.padded_batch_size = plan_info.padded_batch_size;

          cudaError_t status =
              flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM_QK, POS_ENCODING_MODE,
                                                                AttentionVariant>(
                  params, tmp_v, tmp_s, enable_pdl, stream);
          TVM_FFI_ICHECK(status == cudaSuccess)
              << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
          return true;
        }});
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_jax_plan_fused_append,
                              BatchDecodeWithPagedKVCacheFusedAppendJaxPlan);
"""
    write_if_different(path, source)


def _register_batch_decode(dtype: jnp.dtype, head_dim: int) -> str:
    dtype_key = _dtype_key(dtype)
    key = (dtype_key, int(head_dim))
    if key in _BATCH_DECODE_TARGETS:
        return _BATCH_DECODE_TARGETS[key]

    with _REGISTER_LOCK:
        if key in _BATCH_DECODE_TARGETS:
            return _BATCH_DECODE_TARGETS[key]
        _configure_flashinfer_cache()
        torch_dtype = _torch_dtype_for_key(dtype_key)
        import torch
        from flashinfer.jit.attention import gen_customize_batch_decode_module
        from flashinfer.jit.core import gen_jit_spec
        from jax_tvm_ffi import register_ffi_target

        uri = (
            f"nanovllm_jax_batch_decode_jax_plan_{dtype_key}_"
            f"hd{int(head_dim)}"
        )
        base_spec = gen_customize_batch_decode_module(
            uri,
            torch_dtype,
            torch_dtype,
            torch_dtype,
            torch.int32,
            int(head_dim),
            int(head_dim),
            [],
            [],
            ["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
            ["double", "double", "double", "double"],
            "DefaultAttention<false, false, false, false>",
            "#include<flashinfer/attention/variants.cuh>",
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=False,
        )
        custom_binding = Path(base_spec.sources[0]).parent / "batch_decode_jax_plan_binding.cu"
        fused_binding = (
            Path(base_spec.sources[0]).parent
            / "batch_decode_fused_append_jax_plan_binding.cu"
        )
        _write_batch_decode_jax_plan_binding(custom_binding)
        _write_batch_decode_fused_append_jax_plan_binding(fused_binding)
        spec = gen_jit_spec(
            uri + "_jax_plan_binding",
            [*base_spec.sources, custom_binding, fused_binding],
        )
        module = spec.build_and_load()
        target = f"{_BATCH_DECODE_TARGET_PREFIX}_{dtype_key}_hd{int(head_dim)}"
        fused_target = (
            f"{_BATCH_DECODE_FUSED_APPEND_TARGET_PREFIX}_{dtype_key}_hd{int(head_dim)}"
        )
        register_ffi_target(
            target,
            module.run_jax_plan,
            arg_spec=[
                "args",
                *[
                    f"attrs.plan_{idx}"
                    for idx in range(_FLASHINFER_BATCH_DECODE_PLAN_FIELDS)
                ],
                "rets",
                "attrs.kv_layout_code",
                "attrs.window_left",
                "attrs.enable_pdl",
                "attrs.logits_soft_cap",
                "attrs.sm_scale",
                "attrs.rope_rcp_scale",
                "attrs.rope_rcp_theta",
            ],
            platform="gpu",
            allow_cuda_graph=True,
        )
        register_ffi_target(
            fused_target,
            module.run_jax_plan_fused_append,
            arg_spec=[
                "args",
                *[
                    f"attrs.plan_{idx}"
                    for idx in range(_FLASHINFER_BATCH_DECODE_PLAN_FIELDS)
                ],
                "attrs.layer_id",
                "attrs.kv_layout_code",
                "attrs.window_left",
                "attrs.enable_pdl",
                "attrs.logits_soft_cap",
                "attrs.sm_scale",
                "attrs.rope_rcp_scale",
                "attrs.rope_rcp_theta",
            ],
            platform="gpu",
            allow_cuda_graph=True,
        )
        _BATCH_DECODE_MODULES[key] = module
        _BATCH_DECODE_TARGETS[key] = target
        _BATCH_DECODE_FUSED_APPEND_TARGETS[key] = fused_target
        return target


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
    if k_cache.dtype != v_cache.dtype:
        raise ValueError("k_cache and v_cache must have the same dtype")
    if append_key.dtype != k_cache.dtype or append_value.dtype != v_cache.dtype:
        raise ValueError("append_key/value dtype must match the cache dtype")
    if k_cache.dtype not in _SUPPORTED_KV_APPEND_DTYPES:
        raise ValueError(
            "FlashInfer append_paged_kv_cache supports only FP16/BF16 cache tensors "
            f"through this JAX FFI route; got {k_cache.dtype}. The current serving "
            "contract keeps FP32 activation/KV-cache tensors, so this route must "
            "stay disabled unless that dtype policy changes or a FP32 append kernel "
            "is added."
        )

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
    _require_flashinfer_modules()
    _register_append_paged_kv_cache()

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


def radix_topk(
    logits,
    *,
    top_k: int,
    sorted_output: bool = False,
    deterministic: bool = False,
):
    """Return FlashInfer radix top-k values and int32 indices for 2D logits."""

    logits = _as_jax_array("logits", logits)
    if logits.ndim != 2:
        raise ValueError("FlashInfer radix_topk expects logits with shape [batch, vocab]")
    if logits.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        raise ValueError("FlashInfer radix_topk supports FP32/FP16/BF16 logits")
    k = int(top_k)
    if k < 1 or k > int(logits.shape[1]):
        raise ValueError("top_k must be in [1, vocab]")

    _require_flashinfer_modules()
    _register_radix_topk()

    batch = int(logits.shape[0])
    output_indices = jnp.zeros((batch, k), dtype=jnp.int32)
    output_values = jnp.zeros((batch, k), dtype=logits.dtype)
    # FlashInfer's multi-CTA radix path uses a 1 MiB row-state scratch buffer.
    row_states = jnp.zeros((1024 * 1024,), dtype=jnp.uint8)
    call = jax.ffi.ffi_call(
        _RADIX_TOPK_TARGET,
        (
            jax.ShapeDtypeStruct(output_indices.shape, output_indices.dtype),
            jax.ShapeDtypeStruct(output_values.shape, output_values.dtype),
            jax.ShapeDtypeStruct(row_states.shape, row_states.dtype),
        ),
        has_side_effect=True,
        input_output_aliases={1: 0, 2: 1, 3: 2},
    )
    indices, values, _ = call(
        logits,
        output_indices,
        output_values,
        row_states,
        top_k=k,
        sorted_output=bool(sorted_output),
        deterministic=bool(deterministic),
        tie_break=0,
        dsa_graph_safe=False,
    )
    return values, indices.astype(jnp.int32)


def _static_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must be a static Python integer for FlashInfer append routing"
        ) from exc


def _static_scale(value: Any, head_dim: int) -> float:
    try:
        return float(value)
    except Exception:
        return 1.0 / (float(head_dim) ** 0.5)


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

    try:
        num_tokens = _static_int(
            metadata.num_prefill_tokens,
            "metadata.num_prefill_tokens",
        ) + _static_int(
            metadata.num_decode_tokens,
            "metadata.num_decode_tokens",
        )
    except ValueError:
        if k.shape[1] != 1:
            raise
        # Resident decode paths can carry token counters as traced scalars, but
        # width-1 decode has one scheduled token per static batch row.
        num_tokens = int(k.shape[0])
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


def _batch_decode_plan_info(
    *,
    dtype: jnp.dtype,
    batch: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_pages_per_sequence: int,
) -> tuple[tuple[int, ...], Any, int]:
    dtype_key = _dtype_key(dtype)
    cache_key = (
        dtype_key,
        int(batch),
        int(num_qo_heads),
        int(num_kv_heads),
        int(head_dim),
        int(page_size),
        int(max_pages_per_sequence),
    )
    if cache_key in _BATCH_DECODE_PLANS:
        return _BATCH_DECODE_PLANS[cache_key]

    module = _BATCH_DECODE_MODULES[(dtype_key, int(head_dim))]
    import numpy as np
    import torch

    float_workspace = torch.zeros(
        (_FLASHINFER_FLOAT_WORKSPACE_BYTES,),
        dtype=torch.uint8,
        device="cuda",
    )
    int_workspace = torch.empty(
        (_FLASHINFER_INT_WORKSPACE_BYTES,),
        dtype=torch.uint8,
        device="cuda",
    )
    pinned_workspace = torch.empty(
        (_FLASHINFER_INT_WORKSPACE_BYTES,),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    indptr = (
        torch.arange(int(batch) + 1, dtype=torch.int32, device="cpu")
        * int(max_pages_per_sequence)
    )
    torch_dtype = _torch_dtype_for_key(dtype_key)
    empty_q = torch.empty((0,), dtype=torch_dtype)
    empty_kv = torch.empty((0,), dtype=torch_dtype)
    plan_info = tuple(
        int(value)
        for value in module.plan(
            float_workspace,
            int_workspace,
            pinned_workspace,
            indptr,
            int(batch),
            int(num_qo_heads),
            int(num_kv_heads),
            int(page_size),
            False,  # enable_cuda_graph
            -1,  # window_left
            0.0,  # logits_soft_cap
            int(head_dim),
            int(head_dim),
            empty_q,
            empty_kv,
        )
    )
    if len(plan_info) != _FLASHINFER_BATCH_DECODE_PLAN_FIELDS:
        raise ValueError(
            "FlashInfer DecodePlanInfo ABI changed; expected "
            f"{_FLASHINFER_BATCH_DECODE_PLAN_FIELDS} fields, got {len(plan_info)}"
        )
    (
        padded_batch_size,
        _v_offset,
        _s_offset,
        request_indices_offset,
        kv_tile_indices_offset,
        o_indptr_offset,
        block_valid_mask_offset,
        kv_chunk_size_ptr_offset,
        enable_cuda_graph,
        _split_kv,
    ) = plan_info
    planned_int_bytes = max(
        request_indices_offset + max(1, padded_batch_size) * 4,
        kv_tile_indices_offset + max(1, padded_batch_size) * 4,
        o_indptr_offset + (max(1, padded_batch_size) + 1) * 4,
        kv_chunk_size_ptr_offset + 4,
        (
            block_valid_mask_offset + max(1, padded_batch_size)
            if enable_cuda_graph
            else 0
        ),
    )
    # FlashInfer's plan writes scheduler tables into int_workspace; run reads
    # them back using the offsets stored in plan_info. Copy only the populated
    # prefix instead of carrying the full default 8 MiB workspace into XLA.
    int_workspace_bytes = (
        int_workspace[:planned_int_bytes].detach().cpu().numpy().copy().astype(np.uint8)
    )
    if _split_kv:
        dtype_size = jnp.dtype(dtype).itemsize
        planned_float_bytes = max(
            _v_offset + max(1, padded_batch_size) * int(num_qo_heads) * int(head_dim) * dtype_size,
            _s_offset + max(1, padded_batch_size) * int(num_qo_heads) * 4,
        )
    else:
        planned_float_bytes = 1
    result = (plan_info, int_workspace_bytes, int(planned_float_bytes))
    _BATCH_DECODE_PLANS[cache_key] = result
    return result


def paged_decode_attention_gqa_nhd(
    query: jnp.ndarray,
    k_cache_layer: jnp.ndarray,
    v_cache_layer: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_indices: jnp.ndarray,
    kv_last_page_len: jnp.ndarray,
    *,
    scale: float,
) -> jnp.ndarray:
    """Run FlashInfer batch decode attention on an NHD paged KV cache.

    This wrapper intentionally exposes a broad paged-attention ABI: page table
    indptr/indices/last-page-len are passed directly and the FlashInfer plan is
    computed from the static dense shape. It currently allocates FlashInfer's
    workspace inside the compiled program; serving-speed promotion requires
    threading persistent workspace buffers through the executor state.
    """

    query = _as_jax_array("query", query)
    k_cache_layer = _as_jax_array("k_cache_layer", k_cache_layer)
    v_cache_layer = _as_jax_array("v_cache_layer", v_cache_layer)
    kv_indptr = _as_jax_array("kv_indptr", kv_indptr)
    kv_indices = _as_jax_array("kv_indices", kv_indices)
    kv_last_page_len = _as_jax_array("kv_last_page_len", kv_last_page_len)

    if query.ndim != 3:
        raise ValueError("query must have shape [batch, num_heads, head_dim]")
    if k_cache_layer.ndim != 4 or v_cache_layer.shape != k_cache_layer.shape:
        raise ValueError(
            "k_cache_layer/v_cache_layer must have NHD shape "
            "[num_pages, page_size, num_kv_heads, head_dim]"
        )
    if query.dtype != k_cache_layer.dtype or v_cache_layer.dtype != k_cache_layer.dtype:
        raise ValueError("query and KV cache dtypes must match for FlashInfer decode")
    if query.dtype not in _SUPPORTED_BATCH_DECODE_DTYPES:
        raise ValueError(
            "FlashInfer batch decode supports only FP16/BF16 through this JAX FFI route; "
            f"got {query.dtype}"
        )
    if kv_indptr.dtype != jnp.int32 or kv_indices.dtype != jnp.int32 or kv_last_page_len.dtype != jnp.int32:
        raise ValueError("FlashInfer page metadata must use int32 dtype")
    batch, num_qo_heads, head_dim = query.shape
    num_pages, page_size, num_kv_heads, cache_head_dim = k_cache_layer.shape
    if cache_head_dim != head_dim:
        raise ValueError("query/cache head_dim mismatch")
    if kv_indptr.shape != (batch + 1,):
        raise ValueError("kv_indptr must have shape [batch + 1]")
    if kv_last_page_len.shape != (batch,):
        raise ValueError("kv_last_page_len must have shape [batch]")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be 1D")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    _require_flashinfer_modules()
    target = _register_batch_decode(query.dtype, int(head_dim))
    max_pages_per_sequence = max(1, int(kv_indices.shape[0]) // int(batch))
    plan_info, planned_int_workspace, planned_float_workspace_nbytes = _batch_decode_plan_info(
        dtype=query.dtype,
        batch=int(batch),
        num_qo_heads=int(num_qo_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        max_pages_per_sequence=max_pages_per_sequence,
    )
    float_workspace = jnp.zeros((planned_float_workspace_nbytes,), dtype=jnp.uint8)
    int_workspace = jnp.asarray(planned_int_workspace, dtype=jnp.uint8)
    call = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(query.shape, query.dtype),
            jax.ShapeDtypeStruct((batch, num_qo_heads), jnp.float32),
        ),
        has_side_effect=True,
    )
    out, _lse = call(
        float_workspace,
        int_workspace,
        query,
        k_cache_layer,
        v_cache_layer,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        **{f"plan_{idx}": value for idx, value in enumerate(plan_info)},
        kv_layout_code=_NHD_LAYOUT,
        window_left=-1,
        enable_pdl=False,
        logits_soft_cap=0.0,
        sm_scale=_static_scale(scale, int(head_dim)),
        rope_rcp_scale=1.0,
        rope_rcp_theta=1.0e4,
    )
    return out.astype(jnp.float32)


def paged_decode_attention_with_kv_append_gqa_nhd(
    query: jnp.ndarray,
    new_k: jnp.ndarray,
    new_v: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_indices: jnp.ndarray,
    kv_last_page_len: jnp.ndarray,
    positions: jnp.ndarray,
    *,
    layer_id: int,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Append width-1 K/V into the full cache and run FlashInfer decode.

    Args follow the full-cache serving boundary:
    - query: `[batch, num_q_heads, head_dim]`
    - new_k/new_v: `[batch, num_kv_heads, head_dim]`
    - k_cache/v_cache: `[layers, pages, page_size, num_kv_heads, head_dim]`

    Returns `(out, k_cache, v_cache)` where `out` is `[batch, num_q_heads,
    head_dim]` in FP32 and cache outputs alias the input cache buffers.
    """

    query = _as_jax_array("query", query)
    new_k = _as_jax_array("new_k", new_k)
    new_v = _as_jax_array("new_v", new_v)
    k_cache = _as_jax_array("k_cache", k_cache)
    v_cache = _as_jax_array("v_cache", v_cache)
    kv_indptr = _as_jax_array("kv_indptr", kv_indptr)
    kv_indices = _as_jax_array("kv_indices", kv_indices)
    kv_last_page_len = _as_jax_array("kv_last_page_len", kv_last_page_len)
    positions = _as_jax_array("positions", positions)

    if query.ndim != 3:
        raise ValueError("query must have shape [batch, num_heads, head_dim]")
    if new_k.ndim != 3 or new_v.shape != new_k.shape:
        raise ValueError("new_k/new_v must have shape [batch, num_kv_heads, head_dim]")
    if k_cache.ndim != 5 or v_cache.shape != k_cache.shape:
        raise ValueError(
            "k_cache/v_cache must have shape [layers, pages, page_size, num_kv_heads, head_dim]"
        )
    if query.dtype != k_cache.dtype or new_k.dtype != k_cache.dtype or new_v.dtype != k_cache.dtype:
        raise ValueError("query, new_k/new_v, and KV cache dtypes must match")
    if query.dtype not in _SUPPORTED_BATCH_DECODE_DTYPES:
        raise ValueError(
            "FlashInfer fused append+decode supports only FP16/BF16 through this JAX FFI route; "
            f"got {query.dtype}"
        )
    if kv_indptr.dtype != jnp.int32 or kv_indices.dtype != jnp.int32 or kv_last_page_len.dtype != jnp.int32:
        raise ValueError("FlashInfer page metadata must use int32 dtype")
    batch, num_qo_heads, head_dim = query.shape
    num_layers, _num_pages, page_size, num_kv_heads, cache_head_dim = k_cache.shape
    if int(layer_id) < 0 or int(layer_id) >= int(num_layers):
        raise ValueError("layer_id is out of range for KV cache")
    if new_k.shape != (batch, num_kv_heads, head_dim):
        raise ValueError("new_k/new_v trailing dimensions must match query/cache")
    if cache_head_dim != head_dim:
        raise ValueError("query/cache head_dim mismatch")
    if kv_indptr.shape != (batch + 1,):
        raise ValueError("kv_indptr must have shape [batch + 1]")
    if kv_last_page_len.shape != (batch,):
        raise ValueError("kv_last_page_len must have shape [batch]")
    if positions.shape != (batch,):
        raise ValueError("positions must have shape [batch]")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    _require_flashinfer_modules()
    _register_batch_decode(query.dtype, int(head_dim))
    target = _BATCH_DECODE_FUSED_APPEND_TARGETS[(_dtype_key(query.dtype), int(head_dim))]
    max_pages_per_sequence = max(1, int(kv_indices.shape[0]) // int(batch))
    plan_info, planned_int_workspace, planned_float_workspace_nbytes = _batch_decode_plan_info(
        dtype=query.dtype,
        batch=int(batch),
        num_qo_heads=int(num_qo_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        max_pages_per_sequence=max_pages_per_sequence,
    )
    float_workspace = jnp.zeros((planned_float_workspace_nbytes,), dtype=jnp.uint8)
    int_workspace = jnp.asarray(planned_int_workspace, dtype=jnp.uint8)
    batch_indices = jnp.arange(batch, dtype=jnp.int32)
    output_buffer = jnp.zeros(query.shape, dtype=query.dtype)
    call = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(output_buffer.shape, output_buffer.dtype),
            jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
            jax.ShapeDtypeStruct(v_cache.shape, v_cache.dtype),
        ),
        has_side_effect=True,
        input_output_aliases={12: 0, 5: 1, 6: 2},
    )
    out, k_cache_out, v_cache_out = call(
        float_workspace,
        int_workspace,
        query,
        new_k,
        new_v,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        batch_indices,
        positions.astype(jnp.int32),
        output_buffer,
        **{f"plan_{idx}": value for idx, value in enumerate(plan_info)},
        layer_id=int(layer_id),
        kv_layout_code=_NHD_LAYOUT,
        window_left=-1,
        enable_pdl=False,
        logits_soft_cap=0.0,
        sm_scale=_static_scale(scale, int(head_dim)),
        rope_rcp_scale=1.0,
        rope_rcp_theta=1.0e4,
    )
    return out.astype(jnp.float32), k_cache_out, v_cache_out


def paged_prefill_attention_gqa_nhd(*args: Any, **kwargs: Any):
    require_available()
    raise NotImplementedError("paged_prefill_attention_gqa_nhd FlashInfer FFI wrapper is not implemented yet")
