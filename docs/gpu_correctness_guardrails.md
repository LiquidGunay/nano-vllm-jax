# GPU Correctness Guardrails

This project currently treats HuggingFace Qwen3.5 with BF16 checkpoint values
and FP32 activation math as the reference for GPU correctness.

The default persistent JAX compilation cache is:

`/mountpoint/.exp/.cache/jax`

Override it with `NANO_VLLM_JAX_COMPILE_CACHE_DIR`, `JAX_COMPILATION_CACHE_DIR`,
or `NANO_VLLM_JAX_CACHE_ROOT` if a different mount is needed.

## Invariants

- Standard Qwen3.5 RMSNorm computes the shifted scale in FP32:
  `norm(x.float()) * (1.0 + weight.float())`, then casts to the input dtype.
- Gated DeltaNet RMSNorm uses the raw gated norm weight, not `1 + weight`.
- Gated DeltaNet `A_log` is checkpoint-quantized, but `exp(A_log)` is computed
  in FP32 at runtime.
- Multi-chunk Gated DeltaNet prefill must match HF/PyTorch recurrent and
  chunked references. Until the JAX chunk kernel is repaired, multi-chunk
  prefill routes through the recurrent reference path.
- Server/engine runs that claim the BF16-weight/FP32-activation contract must
  use runtime `dtype=float32` with `weight_dtype=bfloat16`.

## Guard Commands

Run these on GPU with cache and temporary paths rooted under `/mountpoint`:

```bash
env TMPDIR=/mountpoint/.exp/tmp \
XDG_CACHE_HOME=/mountpoint/.exp/.cache \
XDG_DATA_HOME=/mountpoint/.exp/.local/share \
UV_CACHE_DIR=/mountpoint/.exp/.cache/uv \
PIP_CACHE_DIR=/mountpoint/.exp/.cache/pip \
HF_HOME=/mountpoint/.exp/.cache/huggingface \
HF_HUB_CACHE=/mountpoint/.exp/.cache/huggingface/hub \
JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax \
NANO_VLLM_JAX_COMPILE_CACHE_DIR=/mountpoint/.exp/.cache/jax \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
HF_TEST_DEVICE=cuda \
uv run pytest tests/test_real_weight_layerwise_parity.py tests/test_e2e_parity.py tests/test_layer_parity.py -q
```

```bash
env TMPDIR=/mountpoint/.exp/tmp \
XDG_CACHE_HOME=/mountpoint/.exp/.cache \
XDG_DATA_HOME=/mountpoint/.exp/.local/share \
UV_CACHE_DIR=/mountpoint/.exp/.cache/uv \
PIP_CACHE_DIR=/mountpoint/.exp/.cache/pip \
JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax \
NANO_VLLM_JAX_COMPILE_CACHE_DIR=/mountpoint/.exp/.cache/jax \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
uv run pytest tests/test_backend_boundaries.py tests/test_kv_cache.py -q
```

```bash
env TMPDIR=/mountpoint/.exp/tmp \
XDG_CACHE_HOME=/mountpoint/.exp/.cache \
XDG_DATA_HOME=/mountpoint/.exp/.local/share \
UV_CACHE_DIR=/mountpoint/.exp/.cache/uv \
PIP_CACHE_DIR=/mountpoint/.exp/.cache/pip \
HF_HOME=/mountpoint/.exp/.cache/huggingface \
HF_HUB_CACHE=/mountpoint/.exp/.cache/huggingface/hub \
JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax \
NANO_VLLM_JAX_COMPILE_CACHE_DIR=/mountpoint/.exp/.cache/jax \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
HF_TEST_DEVICE=cuda \
uv run python benchmark_long_decode_top5.py --max-new-tokens 500
```

Expected long-decode result:

- `top1_exact_matches = 500`
- `ordered_top5_exact_matches = 500`
- `top5_set_exact_matches = 500`
- `max_hf_topk_id_logit_diff <= 2e-5`

The reusable HF artifact is:

`results/qwen08_hf_bf16w_fp32act_long_decode_top5_500.npz`

## Segmented GDN Override Gate

The default segmented GDN policy requires standalone output and final-state max
abs `<=1e-5` versus the current padded chunk32 reference before CUDA math. If we
make an explicit design decision to accept a true-token packed ABI instead, that
path must first pass the same long-decode real-weight gate above:

- exact generated-token match
- `top1_exact_matches = 500`
- `ordered_top5_exact_matches = 500`
- `top5_set_exact_matches = 500`
- `max_hf_topk_id_logit_diff <= 2e-5`
