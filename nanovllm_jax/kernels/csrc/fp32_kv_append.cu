// Historical local CUDA/JAX FFI diagnostic probes.
// Do not route or extend this file as the serving kernel path without an
// explicit decision to reopen local CUDA work; prefer Python-facing Pallas,
// CuteDSL, or borrowed/adapted Triton kernels for new lowered kernel work.

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include "xla/ffi/api/c_api.h"

namespace {

XLA_FFI_Error* CreateError(
    const XLA_FFI_Api* api,
    XLA_FFI_Error_Code code,
    const std::string& message) {
  XLA_FFI_Error_Create_Args args;
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.message = message.c_str();
  args.errc = code;
  return api->XLA_FFI_Error_Create(&args);
}

__device__ inline float MultiplyLikeXlaDot(float lhs, float rhs) {
  return lhs * rhs;
}

bool IsBufferArg(XLA_FFI_CallFrame* call_frame, int64_t index) {
  return index >= 0 && index < call_frame->args.size &&
         call_frame->args.types[index] == XLA_FFI_ArgType_BUFFER;
}

bool IsBufferRet(XLA_FFI_CallFrame* call_frame, int64_t index) {
  return index >= 0 && index < call_frame->rets.size &&
         call_frame->rets.types[index] == XLA_FFI_RetType_BUFFER;
}

XLA_FFI_Buffer* ArgBuffer(XLA_FFI_CallFrame* call_frame, int64_t index) {
  return reinterpret_cast<XLA_FFI_Buffer*>(call_frame->args.args[index]);
}

XLA_FFI_Buffer* RetBuffer(XLA_FFI_CallFrame* call_frame, int64_t index) {
  return reinterpret_cast<XLA_FFI_Buffer*>(call_frame->rets.rets[index]);
}

bool HasTypeAndRank(
    const XLA_FFI_Buffer* buffer,
    XLA_FFI_DataType dtype,
    int64_t rank) {
  return buffer != nullptr && buffer->dtype == dtype && buffer->rank == rank;
}

__device__ __forceinline__ float ToFloat(float value) {
  return value;
}

__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

XLA_FFI_Error* CheckCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 kv append only supports execute stage");
  }
  if ((call_frame->args.size != 9 && call_frame->args.size != 10) ||
      call_frame->rets.size != 2) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 kv append expects 9 or 10 arguments and 2 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 kv append arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 kv append results must be buffers");
    }
  }

  const XLA_FFI_Buffer* append_key = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* append_value = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* batch_indices = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* positions = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* k_cache = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* v_cache = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* kv_indices = ArgBuffer(call_frame, 6);
  const XLA_FFI_Buffer* kv_indptr = ArgBuffer(call_frame, 7);
  const XLA_FFI_Buffer* kv_last_page_len = ArgBuffer(call_frame, 8);
  const XLA_FFI_Buffer* valid_mask =
      call_frame->args.size == 10 ? ArgBuffer(call_frame, 9) : nullptr;
  const XLA_FFI_Buffer* k_out = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* v_out = RetBuffer(call_frame, 1);

  if (!HasTypeAndRank(append_key, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(append_value, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(k_cache, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(v_cache, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(k_out, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(v_out, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "append_key/value and cache buffers must be FP32 with "
                       "ranks 3 and 4");
  }
  if (!HasTypeAndRank(batch_indices, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(positions, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(kv_indices, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(kv_indptr, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(kv_last_page_len, XLA_FFI_DataType_S32, 1)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "page metadata buffers must be int32 rank-1 buffers");
  }
  if (valid_mask != nullptr &&
      !HasTypeAndRank(valid_mask, XLA_FFI_DataType_S32, 1)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "valid_mask must be an int32 rank-1 buffer");
  }
  if (append_key->dims[0] != append_value->dims[0] ||
      append_key->dims[1] != append_value->dims[1] ||
      append_key->dims[2] != append_value->dims[2]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "append_key and append_value shapes must match");
  }
  if (append_key->dims[1] != k_cache->dims[2] ||
      append_key->dims[2] != k_cache->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "append_key trailing shape must match cache heads/dim");
  }
  for (int i = 0; i < 4; ++i) {
    if (k_cache->dims[i] != v_cache->dims[i] ||
        k_cache->dims[i] != k_out->dims[i] ||
        k_cache->dims[i] != v_out->dims[i]) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "cache input and output shapes must match");
    }
  }
  if (batch_indices->dims[0] != append_key->dims[0] ||
      positions->dims[0] != append_key->dims[0]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "batch_indices and positions must have nnz length");
  }
  if (valid_mask != nullptr && valid_mask->dims[0] != append_key->dims[0]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "valid_mask must have nnz length");
  }
  return nullptr;
}

XLA_FFI_Error* CheckDecodeCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 paged decode only supports execute stage");
  }
  if (call_frame->args.size != 8 || call_frame->rets.size != 1) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 paged decode expects 8 arguments and 1 result");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 paged decode arguments must be buffers");
    }
  }
  if (!IsBufferRet(call_frame, 0)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 paged decode result must be a buffer");
  }

  const XLA_FFI_Buffer* q = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* k_cache = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* v_cache = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* kv_indptr = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* kv_indices = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* kv_last_page_len = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 6);
  const XLA_FFI_Buffer* softmax_scale = ArgBuffer(call_frame, 7);
  const XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);

  if (!HasTypeAndRank(q, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(k_cache, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(v_cache, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(out, XLA_FFI_DataType_F32, 3)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "q, cache, and output must be FP32 rank-3/4 buffers");
  }
  if (!HasTypeAndRank(kv_indptr, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(kv_indices, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(kv_last_page_len, XLA_FFI_DataType_S32, 1) ||
      !HasTypeAndRank(seq_lens, XLA_FFI_DataType_S32, 1)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "decode page metadata buffers must be int32 rank-1");
  }
  if (!HasTypeAndRank(softmax_scale, XLA_FFI_DataType_F32, 0)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "softmax_scale must be a FP32 scalar buffer");
  }
  if (k_cache->dims[0] != v_cache->dims[0] ||
      k_cache->dims[1] != v_cache->dims[1] ||
      k_cache->dims[2] != v_cache->dims[2] ||
      k_cache->dims[3] != v_cache->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "k_cache and v_cache shapes must match");
  }
  if (q->dims[0] != out->dims[0] || q->dims[1] != out->dims[1] ||
      q->dims[2] != out->dims[2]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "q and output shapes must match");
  }
  if (q->dims[0] != kv_last_page_len->dims[0] ||
      q->dims[0] != seq_lens->dims[0] ||
      kv_indptr->dims[0] != q->dims[0] + 1) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "metadata batch dimensions must match q batch");
  }
  if (q->dims[2] != k_cache->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "q and cache head_dim must match");
  }
  if (k_cache->dims[2] <= 0 || q->dims[1] % k_cache->dims[2] != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "num_q_heads must be divisible by num_kv_heads");
  }
  if (q->dims[0] <= 0 || kv_indices->dims[0] % q->dims[0] != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "kv_indices must be dense by batch for this prototype");
  }
  return nullptr;
}

XLA_FFI_Error* CheckGdnDecodeCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN decode only supports execute stage");
  }
  if (call_frame->args.size != 6 || call_frame->rets.size != 2) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN decode expects 6 arguments and 2 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 GDN decode arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 GDN decode results must be buffers");
    }
  }

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  if (!HasTypeAndRank(query, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(key, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(value, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(out, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "query/key/value/output must be FP32 rank-4 buffers");
  }
  if (!HasTypeAndRank(g, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(beta, XLA_FFI_DataType_F32, 3)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "g and beta must be FP32 rank-3 buffers");
  }
  if (!HasTypeAndRank(state, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(new_state, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "state and new_state must be FP32 rank-4 buffers");
  }
  if (query->dims[0] != key->dims[0] ||
      query->dims[0] != value->dims[0] ||
      query->dims[0] != state->dims[0] ||
      query->dims[1] != key->dims[1] ||
      query->dims[1] != value->dims[1] ||
      query->dims[1] != state->dims[1]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN decode batch/head dimensions must match");
  }
  if (query->dims[2] != 1 || key->dims[2] != 1 || value->dims[2] != 1 ||
      out->dims[2] != 1 || g->dims[2] != 1 || beta->dims[2] != 1) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN decode prototype only supports width-1 decode");
  }
  if (query->dims[3] != key->dims[3] ||
      value->dims[3] != state->dims[2] ||
      query->dims[3] != state->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN decode state must have shape [B,H,V,K]");
  }
  if (out->dims[0] != value->dims[0] ||
      out->dims[1] != value->dims[1] ||
      out->dims[2] != value->dims[2] ||
      out->dims[3] != value->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN decode output shape must match value");
  }
  for (int i = 0; i < 4; ++i) {
    if (new_state->dims[i] != state->dims[i]) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "GDN decode state output shape must match state input");
    }
  }
  return nullptr;
}

XLA_FFI_Error* CheckGdnPackedDecodeCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 packed GDN decode only supports execute stage");
  }
  if (call_frame->args.size != 6 || call_frame->rets.size != 2) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 packed GDN decode expects 6 arguments and 2 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 packed GDN decode arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 packed GDN decode results must be buffers");
    }
  }

  const XLA_FFI_Buffer* mixed_qkv = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* a = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* b = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* a_log = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* dt_bias = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  if (!HasTypeAndRank(mixed_qkv, XLA_FFI_DataType_F32, 2) ||
      !HasTypeAndRank(a, XLA_FFI_DataType_F32, 2) ||
      !HasTypeAndRank(b, XLA_FFI_DataType_F32, 2) ||
      !HasTypeAndRank(a_log, XLA_FFI_DataType_F32, 1) ||
      !HasTypeAndRank(dt_bias, XLA_FFI_DataType_F32, 1) ||
      !HasTypeAndRank(state, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(out, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(new_state, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode buffers must be FP32 with expected ranks");
  }

  int64_t batch = state->dims[0];
  int64_t num_value_heads = state->dims[1];
  int64_t value_dim = state->dims[2];
  int64_t key_dim = state->dims[3];
  if (mixed_qkv->dims[0] != batch ||
      a->dims[0] != batch ||
      b->dims[0] != batch ||
      a->dims[1] != num_value_heads ||
      b->dims[1] != num_value_heads ||
      a_log->dims[0] != num_value_heads ||
      dt_bias->dims[0] != num_value_heads) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode batch/head metadata must match state");
  }
  int64_t qk_dim = mixed_qkv->dims[1] - num_value_heads * value_dim;
  if (qk_dim <= 0 || qk_dim % (2 * key_dim) != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode mixed_qkv has invalid Q/K dimensions");
  }
  int64_t num_q_heads = qk_dim / (2 * key_dim);
  if (num_q_heads <= 0 || num_value_heads % num_q_heads != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode value heads must be divisible by Q heads");
  }
  if (out->dims[0] != batch ||
      out->dims[1] != num_value_heads ||
      out->dims[2] != 1 ||
      out->dims[3] != value_dim) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode output must have shape [B,HV,1,V]");
  }
  for (int i = 0; i < 4; ++i) {
    if (new_state->dims[i] != state->dims[i]) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "packed GDN decode state output shape must match state");
    }
  }
  return nullptr;
}

XLA_FFI_Error* CheckGdnPackedDecodeBf16QkvCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 packed GDN decode only supports execute stage");
  }
  if (call_frame->args.size != 6 || call_frame->rets.size != 2) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 packed GDN decode expects 6 arguments and 2 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 packed GDN decode arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 packed GDN decode results must be buffers");
    }
  }

  const XLA_FFI_Buffer* mixed_qkv = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* a = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* b = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* a_log = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* dt_bias = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  if (!HasTypeAndRank(mixed_qkv, XLA_FFI_DataType_BF16, 2) ||
      !HasTypeAndRank(a, XLA_FFI_DataType_F32, 2) ||
      !HasTypeAndRank(b, XLA_FFI_DataType_F32, 2) ||
      !HasTypeAndRank(a_log, XLA_FFI_DataType_F32, 1) ||
      !HasTypeAndRank(dt_bias, XLA_FFI_DataType_F32, 1) ||
      !HasTypeAndRank(state, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(out, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(new_state, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode buffers have invalid dtypes or ranks");
  }

  int64_t batch = state->dims[0];
  int64_t num_value_heads = state->dims[1];
  int64_t value_dim = state->dims[2];
  int64_t key_dim = state->dims[3];
  if (mixed_qkv->dims[0] != batch ||
      a->dims[0] != batch ||
      b->dims[0] != batch ||
      a->dims[1] != num_value_heads ||
      b->dims[1] != num_value_heads ||
      a_log->dims[0] != num_value_heads ||
      dt_bias->dims[0] != num_value_heads) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode batch/head metadata must match state");
  }
  int64_t qk_dim = mixed_qkv->dims[1] - num_value_heads * value_dim;
  if (qk_dim <= 0 || qk_dim % (2 * key_dim) != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode mixed_qkv has invalid Q/K dimensions");
  }
  int64_t num_q_heads = qk_dim / (2 * key_dim);
  if (num_q_heads <= 0 || num_value_heads % num_q_heads != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode value heads must be divisible by Q heads");
  }
  if (out->dims[0] != batch ||
      out->dims[1] != num_value_heads ||
      out->dims[2] != 1 ||
      out->dims[3] != value_dim) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "packed GDN decode output must have shape [B,HV,1,V]");
  }
  for (int i = 0; i < 4; ++i) {
    if (new_state->dims[i] != state->dims[i]) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "packed GDN decode state output shape must match state");
    }
  }
  return nullptr;
}

XLA_FFI_Error* CheckGdnPrefillCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN prefill only supports execute stage");
  }
  if (call_frame->args.size != 7 || call_frame->rets.size != 2) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN prefill expects 7 arguments and 2 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 GDN prefill arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 GDN prefill results must be buffers");
    }
  }

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 6);
  const XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  if (!HasTypeAndRank(query, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(key, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(value, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(out, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "query/key/value/output must be FP32 rank-4 buffers");
  }
  if (!HasTypeAndRank(g, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(beta, XLA_FFI_DataType_F32, 3)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "g and beta must be FP32 rank-3 buffers");
  }
  if (!HasTypeAndRank(seq_lens, XLA_FFI_DataType_S32, 1)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "seq_lens must be an int32 rank-1 buffer");
  }
  if (!HasTypeAndRank(state, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(new_state, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "state and new_state must be FP32 rank-4 buffers");
  }
  if (query->dims[0] != key->dims[0] ||
      query->dims[0] != value->dims[0] ||
      query->dims[0] != state->dims[0] ||
      query->dims[0] != seq_lens->dims[0] ||
      query->dims[1] != key->dims[1] ||
      query->dims[1] != value->dims[1] ||
      query->dims[1] != state->dims[1]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN prefill batch/head dimensions must match");
  }
  if (query->dims[2] != key->dims[2] ||
      query->dims[2] != value->dims[2] ||
      g->dims[2] != query->dims[2] ||
      beta->dims[2] != query->dims[2] ||
      query->dims[2] % 32 != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN prefill time dimension must match and be divisible by 32");
  }
  if (g->dims[0] != query->dims[0] || beta->dims[0] != query->dims[0] ||
      g->dims[1] != query->dims[1] || beta->dims[1] != query->dims[1]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "g/beta batch/head dimensions must match query");
  }
  if (query->dims[3] != key->dims[3] ||
      value->dims[3] != state->dims[2] ||
      query->dims[3] != state->dims[3] ||
      value->dims[3] % 32 != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN prefill state must have shape [B,H,V,K] and value_dim must be divisible by 32");
  }
  if (out->dims[0] != value->dims[0] ||
      out->dims[1] != value->dims[1] ||
      out->dims[2] != value->dims[2] ||
      out->dims[3] != value->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN prefill output shape must match value");
  }
  for (int i = 0; i < 4; ++i) {
    if (new_state->dims[i] != state->dims[i]) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "GDN prefill state output shape must match state input");
    }
  }
  return nullptr;
}

XLA_FFI_Error* CheckGdnPrefillPreparedCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared fp32 GDN prefill only supports execute stage");
  }
  if (call_frame->args.size != 7 || call_frame->rets.size != 2) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared fp32 GDN prefill expects 7 arguments and 2 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all prepared fp32 GDN prefill arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all prepared fp32 GDN prefill results must be buffers");
    }
  }

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 6);
  const XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  if (!HasTypeAndRank(query, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(key, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(value, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(out, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared query/key/value/output must be FP32 rank-4 buffers");
  }
  if (!HasTypeAndRank(g, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(beta, XLA_FFI_DataType_F32, 3)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared g and beta must be FP32 rank-3 buffers");
  }
  if (!HasTypeAndRank(seq_lens, XLA_FFI_DataType_S32, 1)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared seq_lens must be an int32 rank-1 buffer");
  }
  if (!HasTypeAndRank(state, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(new_state, XLA_FFI_DataType_F32, 4)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared state and new_state must be FP32 rank-4 buffers");
  }

  int64_t batch = query->dims[0];
  int64_t seq_len = query->dims[1];
  int64_t num_heads = query->dims[2];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  if (key->dims[0] != batch || value->dims[0] != batch ||
      state->dims[0] != batch || seq_lens->dims[0] != batch ||
      key->dims[1] != seq_len || value->dims[1] != seq_len ||
      key->dims[2] != num_heads || value->dims[2] != num_heads ||
      key->dims[3] != key_dim) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared GDN prefill q/k/v batch/time/head dimensions must match");
  }
  if (g->dims[0] != batch || beta->dims[0] != batch ||
      g->dims[1] != seq_len || beta->dims[1] != seq_len ||
      g->dims[2] != num_heads || beta->dims[2] != num_heads) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared g/beta dimensions must match query");
  }
  if (seq_len % 32 != 0 || value_dim % 32 != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared GDN prefill requires time and value_dim divisible by 32");
  }
  if (state->dims[1] != num_heads ||
      state->dims[2] != value_dim ||
      state->dims[3] != key_dim) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared state must have shape [B,H,V,K]");
  }
  if (out->dims[0] != batch || out->dims[1] != seq_len ||
      out->dims[2] != num_heads || out->dims[3] != value_dim) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "prepared output shape must match value");
  }
  for (int i = 0; i < 4; ++i) {
    if (new_state->dims[i] != state->dims[i]) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "prepared state output shape must match state input");
    }
  }
  return nullptr;
}

XLA_FFI_Error* CheckGdnPostConvPrepCallFrame(XLA_FFI_CallFrame* call_frame) {
  const XLA_FFI_Api* api = call_frame->api;
  if (call_frame->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN post-conv prep only supports execute stage");
  }
  if (call_frame->args.size != 6 || call_frame->rets.size != 5) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "fp32 GDN post-conv prep expects 6 arguments and 5 results");
  }
  for (int64_t i = 0; i < call_frame->args.size; ++i) {
    if (!IsBufferArg(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 GDN post-conv prep arguments must be buffers");
    }
  }
  for (int64_t i = 0; i < call_frame->rets.size; ++i) {
    if (!IsBufferRet(call_frame, i)) {
      return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                         "all fp32 GDN post-conv prep results must be buffers");
    }
  }

  const XLA_FFI_Buffer* conv_out = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* a = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* b = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* decay = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* dt_bias = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* valid_mask = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* query = RetBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = RetBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = RetBuffer(call_frame, 2);
  const XLA_FFI_Buffer* gate = RetBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = RetBuffer(call_frame, 4);

  if (!HasTypeAndRank(conv_out, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(a, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(b, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(query, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(key, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(value, XLA_FFI_DataType_F32, 4) ||
      !HasTypeAndRank(gate, XLA_FFI_DataType_F32, 3) ||
      !HasTypeAndRank(beta, XLA_FFI_DataType_F32, 3)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN post-conv prep tensors must be FP32 with expected ranks");
  }
  if (!HasTypeAndRank(decay, XLA_FFI_DataType_F32, 1) ||
      !HasTypeAndRank(dt_bias, XLA_FFI_DataType_F32, 1)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "decay and dt_bias must be FP32 rank-1 buffers");
  }
  if (!HasTypeAndRank(valid_mask, XLA_FFI_DataType_S32, 2)) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "valid_mask must be int32 rank-2");
  }

  int64_t batch = conv_out->dims[0];
  int64_t seq_len = conv_out->dims[1];
  int64_t conv_dim = conv_out->dims[2];
  int64_t num_value_heads = query->dims[1];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  int64_t value_part = num_value_heads * value_dim;
  int64_t qk_part = conv_dim - value_part;
  if (qk_part <= 0 || qk_part % (2 * key_dim) != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "conv_out has invalid Q/K dimensions");
  }
  int64_t num_key_heads = qk_part / (2 * key_dim);
  if (num_key_heads <= 0 || num_value_heads % num_key_heads != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "value heads must be divisible by key heads");
  }
  if (a->dims[0] != batch || b->dims[0] != batch ||
      a->dims[1] != seq_len || b->dims[1] != seq_len ||
      a->dims[2] != num_value_heads || b->dims[2] != num_value_heads ||
      decay->dims[0] != num_value_heads || dt_bias->dims[0] != num_value_heads ||
      valid_mask->dims[0] != batch || valid_mask->dims[1] != seq_len) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN post-conv prep input metadata shapes must match outputs");
  }
  if (key->dims[0] != batch || value->dims[0] != batch ||
      gate->dims[0] != batch || beta->dims[0] != batch ||
      query->dims[0] != batch ||
      key->dims[1] != num_value_heads || value->dims[1] != num_value_heads ||
      gate->dims[1] != num_value_heads || beta->dims[1] != num_value_heads ||
      query->dims[2] != seq_len || key->dims[2] != seq_len ||
      value->dims[2] != seq_len || gate->dims[2] != seq_len ||
      beta->dims[2] != seq_len || key->dims[3] != key_dim) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN post-conv prep output shapes must match");
  }
  return nullptr;
}

void MaybeSetMetadata(XLA_FFI_CallFrame* call_frame) {
  XLA_FFI_Extension_Base* extension = call_frame->extension_start;
  if (extension == nullptr ||
      extension->type != XLA_FFI_Extension_Metadata) {
    return;
  }
  XLA_FFI_Metadata_Extension* metadata_extension =
      reinterpret_cast<XLA_FFI_Metadata_Extension*>(extension);
  XLA_FFI_Metadata* metadata = metadata_extension->metadata;
  metadata->api_version.major_version = XLA_FFI_API_MAJOR;
  metadata->api_version.minor_version = XLA_FFI_API_MINOR;
  metadata->traits = XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
  metadata->state_type_id = XLA_FFI_UNKNOWN_TYPE_ID;
}

__global__ void Fp32KvAppendKernel(
    const float* append_key,
    const float* append_value,
    const int32_t* batch_indices,
    const int32_t* positions,
    float* k_cache,
    float* v_cache,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* valid_mask,
    int64_t nnz_tokens,
    int64_t page_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t elements_per_token = num_kv_heads * head_dim;
  int64_t total_elements = nnz_tokens * elements_per_token;
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_elements) {
    return;
  }

  int64_t token = linear / elements_per_token;
  if (valid_mask != nullptr && valid_mask[token] == 0) {
    return;
  }
  int64_t head_offset = linear - token * elements_per_token;
  int64_t kv_head = head_offset / head_dim;
  int64_t dim = head_offset - kv_head * head_dim;
  int32_t batch = batch_indices[token];
  int32_t position = positions[token];
  int32_t page_offset = position / static_cast<int32_t>(page_size);
  int32_t slot = position - page_offset * static_cast<int32_t>(page_size);
  int32_t physical_page = kv_indices[kv_indptr[batch] + page_offset];

  int64_t cache_offset =
      (((static_cast<int64_t>(physical_page) * page_size + slot) *
        num_kv_heads + kv_head) *
       head_dim + dim);
  k_cache[cache_offset] = append_key[linear];
  v_cache[cache_offset] = append_value[linear];
}

__global__ void Fp32PagedDecodeAttentionKernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    const int32_t* seq_lens,
    const float* softmax_scale,
    float* out,
    int64_t batch,
    int64_t num_q_heads,
    int64_t num_pages,
    int64_t page_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_pages_per_sequence,
    int64_t max_kv_len) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + max_kv_len;

  int64_t row = blockIdx.x;
  int64_t batch_idx = row / num_q_heads;
  int64_t q_head = row - batch_idx * num_q_heads;
  int64_t groups = num_q_heads / num_kv_heads;
  int64_t kv_head = q_head / groups;
  int tid = threadIdx.x;
  float scale = *softmax_scale;

  int32_t page_count = kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  int32_t effective_len = 0;
  if (page_count > 0) {
    effective_len = (page_count - 1) * static_cast<int32_t>(page_size) +
                    kv_last_page_len[batch_idx];
  }
  int32_t seq_len = seq_lens[batch_idx];
  if (effective_len > seq_len) {
    effective_len = seq_len;
  }
  if (effective_len > max_kv_len) {
    effective_len = static_cast<int32_t>(max_kv_len);
  }

  float local_max = -FLT_MAX;
  for (int64_t token = tid; token < max_kv_len; token += blockDim.x) {
    float score = -FLT_MAX;
    if (token < effective_len) {
      int32_t page_offset = token / page_size;
      int32_t slot = token - page_offset * static_cast<int32_t>(page_size);
      int32_t physical_page = kv_indices[kv_indptr[batch_idx] + page_offset];
      float dot = 0.0f;
      int64_t q_base = (batch_idx * num_q_heads + q_head) * head_dim;
      int64_t k_base = ((static_cast<int64_t>(physical_page) * page_size + slot) *
                            num_kv_heads +
                        kv_head) *
                       head_dim;
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        dot += MultiplyLikeXlaDot(q[q_base + dim], k_cache[k_base + dim]);
      }
      score = dot * scale;
    }
    scores[token] = score;
    if (score > local_max) {
      local_max = score;
    }
  }
  reduce[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride && reduce[tid + stride] > reduce[tid]) {
      reduce[tid] = reduce[tid + stride];
    }
    __syncthreads();
  }
  float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int64_t token = tid; token < max_kv_len; token += blockDim.x) {
    float exp_score = token < effective_len ? expf(scores[token] - max_score) : 0.0f;
    scores[token] = exp_score;
    local_sum += exp_score;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  float denom = reduce[0];
  if (denom == 0.0f) {
    denom = 1.0f;
  }

  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < effective_len; ++token) {
      int32_t page_offset = token / page_size;
      int32_t slot = token - page_offset * static_cast<int32_t>(page_size);
      int32_t physical_page = kv_indices[kv_indptr[batch_idx] + page_offset];
      int64_t v_offset =
          ((static_cast<int64_t>(physical_page) * page_size + slot) *
               num_kv_heads +
           kv_head) *
              head_dim +
          dim;
      acc += MultiplyLikeXlaDot(scores[token] / denom, v_cache[v_offset]);
    }
    int64_t out_offset = (batch_idx * num_q_heads + q_head) * head_dim + dim;
    out[out_offset] = acc;
  }
}

__global__ void Fp32GdnRecurrentDecodeKernel(
    const float* query,
    const float* key,
    const float* value,
    const float* g,
    const float* beta,
    const float* state,
    float* out,
    float* new_state,
    int64_t batch,
    int64_t num_heads,
    int64_t key_dim,
    int64_t value_dim) {
  extern __shared__ float shared[];
  float* q_norm = shared;
  float* k_norm = q_norm + key_dim;
  float* delta = k_norm + key_dim;
  float* reduce_q = delta + value_dim;
  float* reduce_k = reduce_q + blockDim.x;

  int64_t row = blockIdx.x;
  int64_t batch_idx = row / num_heads;
  int64_t head = row - batch_idx * num_heads;
  int tid = threadIdx.x;

  int64_t q_base = ((batch_idx * num_heads + head) * 1) * key_dim;
  int64_t v_base = ((batch_idx * num_heads + head) * 1) * value_dim;
  int64_t state_base = ((batch_idx * num_heads + head) * value_dim) * key_dim;
  int64_t gate_offset = (batch_idx * num_heads + head) * 1;

  float local_q_sum = 0.0f;
  float local_k_sum = 0.0f;
  for (int64_t dim = tid; dim < key_dim; dim += blockDim.x) {
    float q_value = query[q_base + dim];
    float k_value = key[q_base + dim];
    local_q_sum += q_value * q_value;
    local_k_sum += k_value * k_value;
  }
  reduce_q[tid] = local_q_sum;
  reduce_k[tid] = local_k_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_q[tid] += reduce_q[tid + stride];
      reduce_k[tid] += reduce_k[tid + stride];
    }
    __syncthreads();
  }

  float q_inv_norm = 1.0f / sqrtf(reduce_q[0] + 1.0e-6f);
  float k_inv_norm = 1.0f / sqrtf(reduce_k[0] + 1.0e-6f);
  float query_scale = 1.0f / sqrtf(static_cast<float>(key_dim));
  for (int64_t dim = tid; dim < key_dim; dim += blockDim.x) {
    q_norm[dim] = query[q_base + dim] * q_inv_norm * query_scale;
    k_norm[dim] = key[q_base + dim] * k_inv_norm;
  }
  __syncthreads();

  float gate = expf(g[gate_offset]);
  float beta_value = beta[gate_offset];
  for (int64_t v_dim = tid; v_dim < value_dim; v_dim += blockDim.x) {
    float kv_mem = 0.0f;
    for (int64_t k_dim = 0; k_dim < key_dim; ++k_dim) {
      kv_mem +=
          state[state_base + v_dim * key_dim + k_dim] * gate * k_norm[k_dim];
    }
    delta[v_dim] = (value[v_base + v_dim] - kv_mem) * beta_value;
  }
  __syncthreads();

  int64_t state_elements = key_dim * value_dim;
  for (int64_t linear = tid; linear < state_elements; linear += blockDim.x) {
    int64_t v_dim = linear / key_dim;
    int64_t k_dim = linear - v_dim * key_dim;
    float updated =
        state[state_base + linear] * gate + delta[v_dim] * k_norm[k_dim];
    new_state[state_base + linear] = updated;
  }
  __syncthreads();

  for (int64_t v_dim = tid; v_dim < value_dim; v_dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t k_dim = 0; k_dim < key_dim; ++k_dim) {
      acc += new_state[state_base + v_dim * key_dim + k_dim] * q_norm[k_dim];
    }
    out[v_base + v_dim] = acc;
  }
}

__device__ inline float SoftplusF32(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

__device__ inline float SigmoidF32(float x) {
  if (x >= 0.0f) {
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  float z = expf(x);
  return z / (1.0f + z);
}

__global__ void Fp32GdnPostConvPrepKernel(
    const float* conv_out,
    const float* a,
    const float* b,
    const float* decay,
    const float* dt_bias,
    const int32_t* valid_mask,
    float* query,
    float* key,
    float* value,
    float* gate,
    float* beta,
    int64_t batch,
    int64_t seq_len,
    int64_t conv_dim,
    int64_t num_key_heads,
    int64_t num_value_heads,
    int64_t key_dim,
    int64_t value_dim) {
  extern __shared__ float shared[];
  float* reduce_q = shared;
  float* reduce_k = reduce_q + blockDim.x;

  int64_t batch_idx = blockIdx.x;
  int64_t token = blockIdx.y;
  int64_t value_head = blockIdx.z;
  int64_t repeat = num_value_heads / num_key_heads;
  int64_t key_head = value_head / repeat;
  int tid = threadIdx.x;

  int64_t token_base = (batch_idx * seq_len + token) * conv_dim;
  int64_t query_base = token_base + key_head * key_dim;
  int64_t key_base = token_base + num_key_heads * key_dim + key_head * key_dim;
  int64_t value_base =
      token_base + 2 * num_key_heads * key_dim + value_head * value_dim;
  int64_t out_q_base =
      ((batch_idx * num_value_heads + value_head) * seq_len + token) * key_dim;
  int64_t out_v_base =
      ((batch_idx * num_value_heads + value_head) * seq_len + token) * value_dim;
  int64_t gate_base = (batch_idx * num_value_heads + value_head) * seq_len + token;
  int valid = valid_mask[batch_idx * seq_len + token] != 0;

  float local_q_sum = 0.0f;
  float local_k_sum = 0.0f;
  if (valid) {
    for (int64_t dim = tid; dim < key_dim; dim += blockDim.x) {
      float q_value = conv_out[query_base + dim];
      float k_value = conv_out[key_base + dim];
      local_q_sum += q_value * q_value;
      local_k_sum += k_value * k_value;
    }
  }
  reduce_q[tid] = local_q_sum;
  reduce_k[tid] = local_k_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_q[tid] += reduce_q[tid + stride];
      reduce_k[tid] += reduce_k[tid + stride];
    }
    __syncthreads();
  }

  float q_inv_norm = rsqrtf(reduce_q[0] + 1.0e-6f);
  float k_inv_norm = rsqrtf(reduce_k[0] + 1.0e-6f);
  for (int64_t dim = tid; dim < key_dim; dim += blockDim.x) {
    float q_value = valid ? conv_out[query_base + dim] * q_inv_norm : 0.0f;
    float k_value = valid ? conv_out[key_base + dim] * k_inv_norm : 0.0f;
    query[out_q_base + dim] = q_value;
    key[out_q_base + dim] = k_value;
  }

  for (int64_t dim = tid; dim < value_dim; dim += blockDim.x) {
    value[out_v_base + dim] = valid ? conv_out[value_base + dim] : 0.0f;
  }

  if (tid == 0) {
    int64_t in_gate_base = (batch_idx * seq_len + token) * num_value_heads + value_head;
    if (valid) {
      float gate_raw =
          -decay[value_head] *
          SoftplusF32(a[in_gate_base] + dt_bias[value_head]);
      gate[gate_base] = gate_raw;
      beta[gate_base] = SigmoidF32(b[in_gate_base]);
    } else {
      gate[gate_base] = 0.0f;
      beta[gate_base] = 0.0f;
    }
  }
}

template <typename TQ>
__global__ void Fp32GdnPackedDecodeKernel(
    const TQ* mixed_qkv,
    const float* a,
    const float* b,
    const float* a_log,
    const float* dt_bias,
    const float* state,
    float* out,
    float* new_state,
    int64_t batch,
    int64_t num_q_heads,
    int64_t num_value_heads,
    int64_t key_dim,
    int64_t value_dim,
    int64_t packed_dim,
    bool precompute_gate_beta) {
  extern __shared__ float shared[];
  float* q_norm = shared;
  float* k_norm = q_norm + key_dim;
  float* delta = k_norm + key_dim;
  float* reduce_q = delta + value_dim;
  float* reduce_k = reduce_q + blockDim.x;

  int64_t row = blockIdx.x;
  int64_t batch_idx = row / num_value_heads;
  int64_t value_head = row - batch_idx * num_value_heads;
  int64_t repeat = num_value_heads / num_q_heads;
  int64_t q_head = value_head / repeat;
  int tid = threadIdx.x;

  int64_t mixed_base = batch_idx * packed_dim;
  int64_t query_base = mixed_base + q_head * key_dim;
  int64_t key_base = mixed_base + num_q_heads * key_dim + q_head * key_dim;
  int64_t value_base =
      mixed_base + 2 * num_q_heads * key_dim + value_head * value_dim;
  int64_t gate_base = batch_idx * num_value_heads + value_head;
  int64_t state_base =
      ((batch_idx * num_value_heads + value_head) * value_dim) * key_dim;

  float local_q_sum = 0.0f;
  float local_k_sum = 0.0f;
  for (int64_t dim = tid; dim < key_dim; dim += blockDim.x) {
    float q_value = ToFloat(mixed_qkv[query_base + dim]);
    float k_value = ToFloat(mixed_qkv[key_base + dim]);
    local_q_sum += q_value * q_value;
    local_k_sum += k_value * k_value;
  }
  reduce_q[tid] = local_q_sum;
  reduce_k[tid] = local_k_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_q[tid] += reduce_q[tid + stride];
      reduce_k[tid] += reduce_k[tid + stride];
    }
    __syncthreads();
  }

  float q_inv_norm = 1.0f / sqrtf(reduce_q[0] + 1.0e-6f);
  float k_inv_norm = 1.0f / sqrtf(reduce_k[0] + 1.0e-6f);
  float query_scale = 1.0f / sqrtf(static_cast<float>(key_dim));
  for (int64_t dim = tid; dim < key_dim; dim += blockDim.x) {
    q_norm[dim] = ToFloat(mixed_qkv[query_base + dim]) * q_inv_norm * query_scale;
    k_norm[dim] = ToFloat(mixed_qkv[key_base + dim]) * k_inv_norm;
  }
  __syncthreads();

  float gate;
  float beta_value;
  if (precompute_gate_beta) {
    // For BF16 QKV route, gate/beta are precomputed in the Python caller as:
    // gate = exp(-decay * softplus(a + dt_bias)) and beta = sigmoid(b).
    gate = a[gate_base];
    beta_value = b[gate_base];
  } else {
    float gate_raw =
        -expf(a_log[value_head]) *
        SoftplusF32(a[gate_base] + dt_bias[value_head]);
    gate = expf(gate_raw);
    beta_value = SigmoidF32(b[gate_base]);
  }
  for (int64_t v_dim = tid; v_dim < value_dim; v_dim += blockDim.x) {
    float kv_mem = 0.0f;
    for (int64_t k_dim = 0; k_dim < key_dim; ++k_dim) {
      kv_mem +=
          state[state_base + v_dim * key_dim + k_dim] * gate * k_norm[k_dim];
    }
    delta[v_dim] = (ToFloat(mixed_qkv[value_base + v_dim]) - kv_mem) * beta_value;
  }
  __syncthreads();

  int64_t state_elements = key_dim * value_dim;
  for (int64_t linear = tid; linear < state_elements; linear += blockDim.x) {
    int64_t v_dim = linear / key_dim;
    int64_t k_dim = linear - v_dim * key_dim;
    float updated =
        state[state_base + linear] * gate + delta[v_dim] * k_norm[k_dim];
    new_state[state_base + linear] = updated;
  }
  __syncthreads();

  int64_t out_base = ((batch_idx * num_value_heads + value_head) * 1) * value_dim;
  for (int64_t v_dim = tid; v_dim < value_dim; v_dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t k_dim = 0; k_dim < key_dim; ++k_dim) {
      acc += new_state[state_base + v_dim * key_dim + k_dim] * q_norm[k_dim];
    }
    out[out_base + v_dim] = acc;
  }
}

template <bool kPreparedLayout>
__device__ __forceinline__ int64_t GdnFeatureOffset(
    int64_t batch_idx,
    int64_t head,
    int64_t token,
    int64_t dim,
    int64_t num_heads,
    int64_t seq_len,
    int64_t feature_dim) {
  if constexpr (kPreparedLayout) {
    return ((batch_idx * seq_len + token) * num_heads + head) * feature_dim + dim;
  }
  return ((batch_idx * num_heads + head) * seq_len + token) * feature_dim + dim;
}

template <bool kPreparedLayout>
__device__ __forceinline__ int64_t GdnGateOffset(
    int64_t batch_idx,
    int64_t head,
    int64_t token,
    int64_t num_heads,
    int64_t seq_len) {
  if constexpr (kPreparedLayout) {
    return (batch_idx * seq_len + token) * num_heads + head;
  }
  return (batch_idx * num_heads + head) * seq_len + token;
}

template <int kBlockV, bool kPreparedLayout>
__global__ void Fp32GdnPrefillChunk32Kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* g,
    const float* beta,
    const int32_t* seq_lens,
    const float* initial_state,
    float* out,
    float* final_state,
    int64_t batch,
    int64_t num_heads,
    int64_t seq_len,
    int64_t key_dim,
    int64_t value_dim) {
  constexpr int kChunk = 32;

  extern __shared__ float shared[];
  float* local_attn = shared;                         // [32, 32]
  float* query_attn = local_attn + kChunk * kChunk;   // [32, 32]
  float* g_cumsum = query_attn + kChunk * kChunk;     // [32]
  float* value_work = g_cumsum + kChunk;              // [32, 32]
  float* k_cumdecay = value_work + kChunk * kBlockV;  // [32, key_dim]
  float* state_tile = k_cumdecay + kChunk * key_dim;  // [key_dim, 32]

  int64_t batch_idx = blockIdx.x;
  int64_t head = blockIdx.y;
  int64_t value_block = blockIdx.z;
  int64_t value_start = value_block * kBlockV;
  int tid = threadIdx.x;

  int64_t state_base = ((batch_idx * num_heads + head) * value_dim) * key_dim;
  float query_scale =
      kPreparedLayout ? rsqrtf(static_cast<float>(key_dim)) : 1.0f;

  for (int64_t linear = tid; linear < seq_len * kBlockV; linear += blockDim.x) {
    int64_t token = linear / kBlockV;
    int64_t v_offset = linear - token * kBlockV;
    out[GdnFeatureOffset<kPreparedLayout>(
        batch_idx,
        head,
        token,
        value_start + v_offset,
        num_heads,
        seq_len,
        value_dim)] = 0.0f;
  }
  for (int64_t linear = tid; linear < key_dim * kBlockV; linear += blockDim.x) {
    int64_t k_offset = linear / kBlockV;
    int64_t v_offset = linear - k_offset * kBlockV;
    state_tile[linear] =
        initial_state[state_base + (value_start + v_offset) * key_dim + k_offset];
  }
  __syncthreads();

  int32_t row_len = seq_lens[batch_idx];
  int64_t n_chunks = seq_len / kChunk;
  for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
    int64_t start = chunk * kChunk;
    int32_t active_tokens = 0;
    if (row_len > start) {
      int32_t remaining = row_len - static_cast<int32_t>(start);
      active_tokens = remaining < kChunk ? remaining : kChunk;
    }
    if (active_tokens <= 0) {
      continue;
    }

    if (tid == 0) {
      float running = 0.0f;
      for (int i = 0; i < active_tokens; ++i) {
        running += g[GdnGateOffset<kPreparedLayout>(
            batch_idx, head, start + i, num_heads, seq_len)];
        g_cumsum[i] = running;
      }
      for (int i = active_tokens; i < kChunk; ++i) {
        g_cumsum[i] = running;
      }
    }
    __syncthreads();

    for (int idx = tid; idx < kChunk * kChunk; idx += blockDim.x) {
      int i = idx / kChunk;
      int j = idx - i * kChunk;
      float value_ij = 0.0f;
      if (i < active_tokens && j < i) {
        float dot = 0.0f;
        float beta_i = beta[GdnGateOffset<kPreparedLayout>(
            batch_idx, head, start + i, num_heads, seq_len)];
        int64_t key_i_base = GdnFeatureOffset<kPreparedLayout>(
            batch_idx, head, start + i, 0, num_heads, seq_len, key_dim);
        int64_t key_j_base = GdnFeatureOffset<kPreparedLayout>(
            batch_idx, head, start + j, 0, num_heads, seq_len, key_dim);
        for (int64_t k_offset = 0; k_offset < key_dim; ++k_offset) {
          dot += key[key_i_base + k_offset] * beta_i *
                 key[key_j_base + k_offset];
        }
        float decay = expf(g_cumsum[i] - g_cumsum[j]);
        value_ij = -(dot * decay);
      }
      local_attn[idx] = value_ij;
    }
    __syncthreads();

    for (int row = 1; row < active_tokens; ++row) {
      for (int col = tid; col < kChunk; col += blockDim.x) {
        float updated = 0.0f;
        if (col < row) {
          float row_value = local_attn[row * kChunk + col];
          float contribution = 0.0f;
          for (int j = 0; j < row; ++j) {
            contribution += local_attn[row * kChunk + j] *
                            local_attn[j * kChunk + col];
          }
          updated = row_value + contribution;
        }
        query_attn[row * kChunk + col] = updated;
      }
      __syncthreads();
      for (int col = tid; col < kChunk; col += blockDim.x) {
        if (col < row) {
          local_attn[row * kChunk + col] = query_attn[row * kChunk + col];
        }
      }
      __syncthreads();
    }

    for (int idx = tid; idx < active_tokens * kBlockV; idx += blockDim.x) {
      int i = idx / kBlockV;
      int v_offset = idx - i * kBlockV;
      float acc = 0.0f;
      for (int t = 0; t <= i; ++t) {
        float attn = (i == t) ? 1.0f : local_attn[i * kChunk + t];
        acc += attn *
               value[GdnFeatureOffset<kPreparedLayout>(
                   batch_idx,
                   head,
                   start + t,
                   value_start + v_offset,
                   num_heads,
                   seq_len,
                   value_dim)] *
               beta[GdnGateOffset<kPreparedLayout>(
                   batch_idx, head, start + t, num_heads, seq_len)];
      }
      value_work[i * kBlockV + v_offset] = acc;
    }
    for (int idx = tid; idx < active_tokens * key_dim; idx += blockDim.x) {
      int i = idx / key_dim;
      int k_offset = idx - i * key_dim;
      float acc = 0.0f;
      for (int t = 0; t <= i; ++t) {
        float attn = (i == t) ? 1.0f : local_attn[i * kChunk + t];
        acc += attn *
               key[GdnFeatureOffset<kPreparedLayout>(
                   batch_idx,
                   head,
                   start + t,
                   k_offset,
                   num_heads,
                   seq_len,
                   key_dim)] *
               beta[GdnGateOffset<kPreparedLayout>(
                   batch_idx, head, start + t, num_heads, seq_len)] *
               expf(g_cumsum[t]);
      }
      k_cumdecay[i * key_dim + k_offset] = acc;
    }
    for (int idx = tid; idx < active_tokens * active_tokens; idx += blockDim.x) {
      int i = idx / active_tokens;
      int j = idx - i * active_tokens;
      float acc = 0.0f;
      if (j <= i) {
        int64_t q_i_base = GdnFeatureOffset<kPreparedLayout>(
            batch_idx, head, start + i, 0, num_heads, seq_len, key_dim);
        int64_t k_j_base = GdnFeatureOffset<kPreparedLayout>(
            batch_idx, head, start + j, 0, num_heads, seq_len, key_dim);
        for (int64_t k_offset = 0; k_offset < key_dim; ++k_offset) {
          acc += query[q_i_base + k_offset] * query_scale *
                 key[k_j_base + k_offset];
        }
        acc *= expf(g_cumsum[i] - g_cumsum[j]);
      }
      query_attn[i * kChunk + j] = acc;
    }
    __syncthreads();

    for (int idx = tid; idx < active_tokens * kBlockV; idx += blockDim.x) {
      int i = idx / kBlockV;
      int v_offset = idx - i * kBlockV;
      float v_prime = 0.0f;
      for (int64_t k_offset = 0; k_offset < key_dim; ++k_offset) {
        v_prime += k_cumdecay[i * key_dim + k_offset] *
                   state_tile[k_offset * kBlockV + v_offset];
      }
      value_work[i * kBlockV + v_offset] -= v_prime;
    }
    __syncthreads();

    for (int idx = tid; idx < active_tokens * kBlockV; idx += blockDim.x) {
      int i = idx / kBlockV;
      int v_offset = idx - i * kBlockV;
      float attn_inter = 0.0f;
      for (int64_t k_offset = 0; k_offset < key_dim; ++k_offset) {
        attn_inter += query[GdnFeatureOffset<kPreparedLayout>(
                          batch_idx,
                          head,
                          start + i,
                          k_offset,
                          num_heads,
                          seq_len,
                          key_dim)] *
                      query_scale *
                      expf(g_cumsum[i]) *
                      state_tile[k_offset * kBlockV + v_offset];
      }
      float attn_v_new = 0.0f;
      for (int j = 0; j <= i; ++j) {
        attn_v_new += query_attn[i * kChunk + j] *
                      value_work[j * kBlockV + v_offset];
      }
      out[GdnFeatureOffset<kPreparedLayout>(
          batch_idx,
          head,
          start + i,
          value_start + v_offset,
          num_heads,
          seq_len,
          value_dim)] =
          attn_inter + attn_v_new;
    }
    __syncthreads();

    float g_last = g_cumsum[active_tokens - 1];
    float exp_g_last = expf(g_last);
    for (int idx = tid; idx < key_dim * kBlockV; idx += blockDim.x) {
      int k_offset = idx / kBlockV;
      int v_offset = idx - k_offset * kBlockV;
      float state_update = 0.0f;
      for (int i = 0; i < active_tokens; ++i) {
        state_update +=
            key[GdnFeatureOffset<kPreparedLayout>(
                batch_idx,
                head,
                start + i,
                k_offset,
                num_heads,
                seq_len,
                key_dim)] *
            expf(g_last - g_cumsum[i]) *
            value_work[i * kBlockV + v_offset];
      }
      state_tile[idx] = state_tile[idx] * exp_g_last + state_update;
    }
    __syncthreads();
  }

  for (int64_t linear = tid; linear < key_dim * kBlockV; linear += blockDim.x) {
    int64_t k_offset = linear / kBlockV;
    int64_t v_offset = linear - k_offset * kBlockV;
    final_state[state_base + (value_start + v_offset) * key_dim + k_offset] =
        state_tile[linear];
  }
}

}  // namespace

extern "C" XLA_FFI_Error* NanoVllmJaxFp32KvAppend(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* append_key = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* append_value = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* batch_indices = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* positions = ArgBuffer(call_frame, 3);
  XLA_FFI_Buffer* k_out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* v_out = RetBuffer(call_frame, 1);
  const XLA_FFI_Buffer* kv_indices = ArgBuffer(call_frame, 6);
  const XLA_FFI_Buffer* kv_indptr = ArgBuffer(call_frame, 7);
  const XLA_FFI_Buffer* valid_mask =
      call_frame->args.size == 10 ? ArgBuffer(call_frame, 9) : nullptr;

  int64_t nnz_tokens = append_key->dims[0];
  int64_t page_size = k_out->dims[1];
  int64_t num_kv_heads = k_out->dims[2];
  int64_t head_dim = k_out->dims[3];
  int64_t total_elements = nnz_tokens * num_kv_heads * head_dim;
  if (total_elements == 0) {
    return nullptr;
  }

  int threads = 256;
  int blocks = static_cast<int>((total_elements + threads - 1) / threads);
  Fp32KvAppendKernel<<<blocks, threads, 0, stream>>>(
      static_cast<const float*>(append_key->data),
      static_cast<const float*>(append_value->data),
      static_cast<const int32_t*>(batch_indices->data),
      static_cast<const int32_t*>(positions->data),
      static_cast<float*>(k_out->data),
      static_cast<float*>(v_out->data),
      static_cast<const int32_t*>(kv_indices->data),
      static_cast<const int32_t*>(kv_indptr->data),
      valid_mask == nullptr ? nullptr : static_cast<const int32_t*>(valid_mask->data),
      nnz_tokens,
      page_size,
      num_kv_heads,
      head_dim);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32PagedDecodeAttention(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckDecodeCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* q = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* k_cache = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* v_cache = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* kv_indptr = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* kv_indices = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* kv_last_page_len = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 6);
  const XLA_FFI_Buffer* softmax_scale = ArgBuffer(call_frame, 7);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);

  int64_t batch = q->dims[0];
  int64_t num_q_heads = q->dims[1];
  int64_t head_dim = q->dims[2];
  int64_t num_pages = k_cache->dims[0];
  int64_t page_size = k_cache->dims[1];
  int64_t num_kv_heads = k_cache->dims[2];
  int64_t max_pages_per_sequence = kv_indices->dims[0] / batch;
  int64_t max_kv_len = max_pages_per_sequence * page_size;
  int threads = 256;
  int64_t grid = batch * num_q_heads;
  size_t shared_bytes = static_cast<size_t>(max_kv_len + threads) * sizeof(float);

  Fp32PagedDecodeAttentionKernel<<<static_cast<int>(grid), threads, shared_bytes, stream>>>(
      static_cast<const float*>(q->data),
      static_cast<const float*>(k_cache->data),
      static_cast<const float*>(v_cache->data),
      static_cast<const int32_t*>(kv_indptr->data),
      static_cast<const int32_t*>(kv_indices->data),
      static_cast<const int32_t*>(kv_last_page_len->data),
      static_cast<const int32_t*>(seq_lens->data),
      static_cast<const float*>(softmax_scale->data),
      static_cast<float*>(out->data),
      batch,
      num_q_heads,
      num_pages,
      page_size,
      num_kv_heads,
      head_dim,
      max_pages_per_sequence,
      max_kv_len);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnRecurrentDecode(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnDecodeCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 5);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  int64_t batch = query->dims[0];
  int64_t num_heads = query->dims[1];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  int threads = 256;
  int64_t grid = batch * num_heads;
  size_t shared_bytes =
      static_cast<size_t>(key_dim * 2 + value_dim + threads * 2) *
      sizeof(float);

  Fp32GdnRecurrentDecodeKernel<<<static_cast<int>(grid), threads, shared_bytes, stream>>>(
      static_cast<const float*>(query->data),
      static_cast<const float*>(key->data),
      static_cast<const float*>(value->data),
      static_cast<const float*>(g->data),
      static_cast<const float*>(beta->data),
      static_cast<const float*>(state->data),
      static_cast<float*>(out->data),
      static_cast<float*>(new_state->data),
      batch,
      num_heads,
      key_dim,
      value_dim);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnPackedDecode(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnPackedDecodeCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* mixed_qkv = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* a = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* b = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* a_log = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* dt_bias = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 5);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  int64_t batch = state->dims[0];
  int64_t num_value_heads = state->dims[1];
  int64_t value_dim = state->dims[2];
  int64_t key_dim = state->dims[3];
  int64_t packed_dim = mixed_qkv->dims[1];
  int64_t qk_dim = packed_dim - num_value_heads * value_dim;
  int64_t num_q_heads = qk_dim / (2 * key_dim);
  int threads = 256;
  int64_t grid = batch * num_value_heads;
  size_t shared_bytes =
      static_cast<size_t>(key_dim * 2 + value_dim + threads * 2) *
      sizeof(float);

  Fp32GdnPackedDecodeKernel<float><<<static_cast<int>(grid), threads, shared_bytes, stream>>>(
      static_cast<const float*>(mixed_qkv->data),
      static_cast<const float*>(a->data),
      static_cast<const float*>(b->data),
      static_cast<const float*>(a_log->data),
      static_cast<const float*>(dt_bias->data),
      static_cast<const float*>(state->data),
      static_cast<float*>(out->data),
      static_cast<float*>(new_state->data),
      batch,
      num_q_heads,
      num_value_heads,
      key_dim,
      value_dim,
      packed_dim,
      false);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnPackedDecodeBf16qkv(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnPackedDecodeBf16QkvCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* mixed_qkv = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* a = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* b = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* a_log = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* dt_bias = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 5);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  int64_t batch = state->dims[0];
  int64_t num_value_heads = state->dims[1];
  int64_t value_dim = state->dims[2];
  int64_t key_dim = state->dims[3];
  int64_t packed_dim = mixed_qkv->dims[1];
  int64_t qk_dim = packed_dim - num_value_heads * value_dim;
  int64_t num_q_heads = qk_dim / (2 * key_dim);
  int threads = 256;
  int64_t grid = batch * num_value_heads;
  size_t shared_bytes =
      static_cast<size_t>(key_dim * 2 + value_dim + threads * 2) *
      sizeof(float);

  Fp32GdnPackedDecodeKernel<__nv_bfloat16><<<
      static_cast<int>(grid), threads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(mixed_qkv->data),
      static_cast<const float*>(a->data),
      static_cast<const float*>(b->data),
      static_cast<const float*>(a_log->data),
      static_cast<const float*>(dt_bias->data),
      static_cast<const float*>(state->data),
      static_cast<float*>(out->data),
      static_cast<float*>(new_state->data),
      batch,
      num_q_heads,
      num_value_heads,
      key_dim,
      value_dim,
      packed_dim,
      true);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnPostConvPrep(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnPostConvPrepCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* conv_out = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* a = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* b = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* decay = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* dt_bias = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* valid_mask = ArgBuffer(call_frame, 5);
  XLA_FFI_Buffer* query = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* key = RetBuffer(call_frame, 1);
  XLA_FFI_Buffer* value = RetBuffer(call_frame, 2);
  XLA_FFI_Buffer* gate = RetBuffer(call_frame, 3);
  XLA_FFI_Buffer* beta = RetBuffer(call_frame, 4);

  int64_t batch = conv_out->dims[0];
  int64_t seq_len = conv_out->dims[1];
  int64_t conv_dim = conv_out->dims[2];
  int64_t num_value_heads = query->dims[1];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  int64_t num_key_heads = (conv_dim - num_value_heads * value_dim) / (2 * key_dim);

  int threads = 256;
  dim3 grid(
      static_cast<unsigned int>(batch),
      static_cast<unsigned int>(seq_len),
      static_cast<unsigned int>(num_value_heads));
  size_t shared_bytes = 2 * threads * sizeof(float);
  Fp32GdnPostConvPrepKernel<<<grid, threads, shared_bytes, stream>>>(
      static_cast<const float*>(conv_out->data),
      static_cast<const float*>(a->data),
      static_cast<const float*>(b->data),
      static_cast<const float*>(decay->data),
      static_cast<const float*>(dt_bias->data),
      static_cast<const int32_t*>(valid_mask->data),
      static_cast<float*>(query->data),
      static_cast<float*>(key->data),
      static_cast<float*>(value->data),
      static_cast<float*>(gate->data),
      static_cast<float*>(beta->data),
      batch,
      seq_len,
      conv_dim,
      num_key_heads,
      num_value_heads,
      key_dim,
      value_dim);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnPrefillChunk32(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnPrefillCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 6);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  int64_t batch = query->dims[0];
  int64_t num_heads = query->dims[1];
  int64_t seq_len = query->dims[2];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  constexpr int kChunk = 32;
  constexpr int kBlockV = 32;
  int threads = 256;
  dim3 grid(
      static_cast<unsigned int>(batch),
      static_cast<unsigned int>(num_heads),
      static_cast<unsigned int>(value_dim / kBlockV));
  size_t shared_floats =
      kChunk * kChunk +       // local_attn
      kChunk * kChunk +       // query_attn
      kChunk +                // g_cumsum
      kChunk * kBlockV +      // value_work
      kChunk * key_dim +      // k_cumdecay
      key_dim * kBlockV;      // state_tile
  size_t shared_bytes = shared_floats * sizeof(float);

  Fp32GdnPrefillChunk32Kernel<kBlockV, false><<<grid, threads, shared_bytes, stream>>>(
      static_cast<const float*>(query->data),
      static_cast<const float*>(key->data),
      static_cast<const float*>(value->data),
      static_cast<const float*>(g->data),
      static_cast<const float*>(beta->data),
      static_cast<const int32_t*>(seq_lens->data),
      static_cast<const float*>(state->data),
      static_cast<float*>(out->data),
      static_cast<float*>(new_state->data),
      batch,
      num_heads,
      seq_len,
      key_dim,
      value_dim);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnPrefillChunk32V64(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnPrefillCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 6);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  int64_t batch = query->dims[0];
  int64_t num_heads = query->dims[1];
  int64_t seq_len = query->dims[2];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  constexpr int kChunk = 32;
  constexpr int kBlockV = 64;
  if (value_dim % kBlockV != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN prefill v64 requires value_dim divisible by 64");
  }
  int threads = 256;
  dim3 grid(
      static_cast<unsigned int>(batch),
      static_cast<unsigned int>(num_heads),
      static_cast<unsigned int>(value_dim / kBlockV));
  size_t shared_floats =
      kChunk * kChunk +       // local_attn
      kChunk * kChunk +       // query_attn
      kChunk +                // g_cumsum
      kChunk * kBlockV +      // value_work
      kChunk * key_dim +      // k_cumdecay
      key_dim * kBlockV;      // state_tile
  size_t shared_bytes = shared_floats * sizeof(float);

  cudaError_t attr_error = cudaFuncSetAttribute(
      Fp32GdnPrefillChunk32Kernel<kBlockV, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));
  if (attr_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(attr_error));
  }

  Fp32GdnPrefillChunk32Kernel<kBlockV, false><<<grid, threads, shared_bytes, stream>>>(
      static_cast<const float*>(query->data),
      static_cast<const float*>(key->data),
      static_cast<const float*>(value->data),
      static_cast<const float*>(g->data),
      static_cast<const float*>(beta->data),
      static_cast<const int32_t*>(seq_lens->data),
      static_cast<const float*>(state->data),
      static_cast<float*>(out->data),
      static_cast<float*>(new_state->data),
      batch,
      num_heads,
      seq_len,
      key_dim,
      value_dim);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}

extern "C" XLA_FFI_Error* NanoVllmJaxFp32GdnPrefillChunk32Prepared(
    XLA_FFI_CallFrame* call_frame) {
  MaybeSetMetadata(call_frame);
  if (call_frame->extension_start != nullptr) {
    return nullptr;
  }

  if (XLA_FFI_Error* error = CheckGdnPrefillPreparedCallFrame(call_frame)) {
    return error;
  }

  const XLA_FFI_Api* api = call_frame->api;
  XLA_FFI_Stream_Get_Args stream_args;
  stream_args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  stream_args.extension_start = nullptr;
  stream_args.ctx = call_frame->ctx;
  stream_args.stream = nullptr;
  if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&stream_args)) {
    return error;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_args.stream);

  const XLA_FFI_Buffer* query = ArgBuffer(call_frame, 0);
  const XLA_FFI_Buffer* key = ArgBuffer(call_frame, 1);
  const XLA_FFI_Buffer* value = ArgBuffer(call_frame, 2);
  const XLA_FFI_Buffer* g = ArgBuffer(call_frame, 3);
  const XLA_FFI_Buffer* beta = ArgBuffer(call_frame, 4);
  const XLA_FFI_Buffer* seq_lens = ArgBuffer(call_frame, 5);
  const XLA_FFI_Buffer* state = ArgBuffer(call_frame, 6);
  XLA_FFI_Buffer* out = RetBuffer(call_frame, 0);
  XLA_FFI_Buffer* new_state = RetBuffer(call_frame, 1);

  int64_t batch = query->dims[0];
  int64_t seq_len = query->dims[1];
  int64_t num_heads = query->dims[2];
  int64_t key_dim = query->dims[3];
  int64_t value_dim = value->dims[3];
  constexpr int kChunk = 32;
  constexpr int kBlockV = 32;
  int threads = 256;
  dim3 grid(
      static_cast<unsigned int>(batch),
      static_cast<unsigned int>(num_heads),
      static_cast<unsigned int>(value_dim / kBlockV));
  size_t shared_floats =
      kChunk * kChunk +       // local_attn
      kChunk * kChunk +       // query_attn
      kChunk +                // g_cumsum
      kChunk * kBlockV +      // value_work
      kChunk * key_dim +      // k_cumdecay
      key_dim * kBlockV;      // state_tile
  size_t shared_bytes = shared_floats * sizeof(float);

  Fp32GdnPrefillChunk32Kernel<kBlockV, true><<<grid, threads, shared_bytes, stream>>>(
      static_cast<const float*>(query->data),
      static_cast<const float*>(key->data),
      static_cast<const float*>(value->data),
      static_cast<const float*>(g->data),
      static_cast<const float*>(beta->data),
      static_cast<const int32_t*>(seq_lens->data),
      static_cast<const float*>(state->data),
      static_cast<float*>(out->data),
      static_cast<float*>(new_state->data),
      batch,
      num_heads,
      seq_len,
      key_dim,
      value_dim);
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(launch_error));
  }
  return nullptr;
}
