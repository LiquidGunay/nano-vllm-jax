#include <cmath>
#include <cfloat>
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
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
      query->dims[3] != state->dims[2] ||
      value->dims[3] != state->dims[3]) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN decode head dimensions must match state");
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
      query->dims[3] != state->dims[2] ||
      value->dims[3] != state->dims[3] ||
      value->dims[3] % 32 != 0) {
    return CreateError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                       "GDN prefill dimensions must match state and value_dim must be divisible by 32");
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
  int64_t state_base = ((batch_idx * num_heads + head) * key_dim) * value_dim;
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

  float q_inv_norm = rsqrtf(reduce_q[0] + 1.0e-6f);
  float k_inv_norm = rsqrtf(reduce_k[0] + 1.0e-6f);
  float query_scale = rsqrtf(static_cast<float>(key_dim));
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
          state[state_base + k_dim * value_dim + v_dim] * gate * k_norm[k_dim];
    }
    delta[v_dim] = (value[v_base + v_dim] - kv_mem) * beta_value;
  }
  __syncthreads();

  int64_t state_elements = key_dim * value_dim;
  for (int64_t linear = tid; linear < state_elements; linear += blockDim.x) {
    int64_t k_dim = linear / value_dim;
    int64_t v_dim = linear - k_dim * value_dim;
    float updated =
        state[state_base + linear] * gate + k_norm[k_dim] * delta[v_dim];
    new_state[state_base + linear] = updated;
  }
  __syncthreads();

  for (int64_t v_dim = tid; v_dim < value_dim; v_dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t k_dim = 0; k_dim < key_dim; ++k_dim) {
      acc += new_state[state_base + k_dim * value_dim + v_dim] * q_norm[k_dim];
    }
    out[v_base + v_dim] = acc;
  }
}

template <int kBlockV>
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

  int64_t q_base = ((batch_idx * num_heads + head) * seq_len) * key_dim;
  int64_t v_base = ((batch_idx * num_heads + head) * seq_len) * value_dim;
  int64_t gate_base = (batch_idx * num_heads + head) * seq_len;
  int64_t state_base = ((batch_idx * num_heads + head) * key_dim) * value_dim;

  for (int64_t linear = tid; linear < seq_len * kBlockV; linear += blockDim.x) {
    int64_t token = linear / kBlockV;
    int64_t v_offset = linear - token * kBlockV;
    out[v_base + token * value_dim + value_start + v_offset] = 0.0f;
  }
  for (int64_t linear = tid; linear < key_dim * kBlockV; linear += blockDim.x) {
    int64_t k_offset = linear / kBlockV;
    int64_t v_offset = linear - k_offset * kBlockV;
    state_tile[linear] =
        initial_state[state_base + k_offset * value_dim + value_start + v_offset];
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
        running += g[gate_base + start + i];
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
        float beta_i = beta[gate_base + start + i];
        int64_t key_i_base = q_base + (start + i) * key_dim;
        int64_t key_j_base = q_base + (start + j) * key_dim;
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
        acc += attn * value[v_base + (start + t) * value_dim + value_start + v_offset] *
               beta[gate_base + start + t];
      }
      value_work[i * kBlockV + v_offset] = acc;
    }
    for (int idx = tid; idx < active_tokens * key_dim; idx += blockDim.x) {
      int i = idx / key_dim;
      int k_offset = idx - i * key_dim;
      float acc = 0.0f;
      for (int t = 0; t <= i; ++t) {
        float attn = (i == t) ? 1.0f : local_attn[i * kChunk + t];
        acc += attn * key[q_base + (start + t) * key_dim + k_offset] *
               beta[gate_base + start + t] * expf(g_cumsum[t]);
      }
      k_cumdecay[i * key_dim + k_offset] = acc;
    }
    for (int idx = tid; idx < active_tokens * active_tokens; idx += blockDim.x) {
      int i = idx / active_tokens;
      int j = idx - i * active_tokens;
      float acc = 0.0f;
      if (j <= i) {
        int64_t q_i_base = q_base + (start + i) * key_dim;
        int64_t k_j_base = q_base + (start + j) * key_dim;
        for (int64_t k_offset = 0; k_offset < key_dim; ++k_offset) {
          acc += query[q_i_base + k_offset] * key[k_j_base + k_offset];
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
        attn_inter += query[q_base + (start + i) * key_dim + k_offset] *
                      expf(g_cumsum[i]) *
                      state_tile[k_offset * kBlockV + v_offset];
      }
      float attn_v_new = 0.0f;
      for (int j = 0; j <= i; ++j) {
        attn_v_new += query_attn[i * kChunk + j] *
                      value_work[j * kBlockV + v_offset];
      }
      out[v_base + (start + i) * value_dim + value_start + v_offset] =
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
            key[q_base + (start + i) * key_dim + k_offset] *
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
    final_state[state_base + k_offset * value_dim + value_start + v_offset] =
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

  Fp32GdnPrefillChunk32Kernel<kBlockV><<<grid, threads, shared_bytes, stream>>>(
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
      Fp32GdnPrefillChunk32Kernel<kBlockV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));
  if (attr_error != cudaSuccess) {
    return CreateError(api, XLA_FFI_Error_Code_INTERNAL,
                       cudaGetErrorString(attr_error));
  }

  Fp32GdnPrefillChunk32Kernel<kBlockV><<<grid, threads, shared_bytes, stream>>>(
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
