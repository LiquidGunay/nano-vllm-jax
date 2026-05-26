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
