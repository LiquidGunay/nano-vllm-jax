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
