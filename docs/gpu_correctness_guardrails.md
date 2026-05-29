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

Expected long-decoder result (default FP32-activation contract):

- `top1_exact_matches = 500`
- `ordered_top5_exact_matches = 500`
- `top5_set_exact_matches = 500`
- `max_hf_topk_id_logit_diff <= 2e-5`

The reusable HF artifact is:

`results/qwen08_hf_bf16w_fp32act_long_decode_top5_500.npz`

Reusable HF long-prefill artifacts are:

- `results/qwen08_hf_bf16w_fp32act_long_prefill_512_2048x16.json`
- `results/qwen08_hf_bf16w_bf16act_long_prefill_512_2048x16.json`

Both cover `Qwen/Qwen3.5-0.8B`, prompt lengths `[512, 1024, 1536, 2048]`,
output length `16`, prompt source `tokenized_seed_repeat`, prompt suite
`mixed`, and `top_k=5`. Use the FP32-activation artifact for the default GPU
correctness contract. Use the BF16-activation artifact only for explicit
BF16-activation diagnostics such as GDN prefill experiments.

BF16 external-kernel lane gate (diagnostic-only, not a contract change):

- `top1_exact_matches = 500`
- `ordered_top5_exact_matches = 500`
- `top5_set_exact_matches = 500`
- exact generated-token identity must hold
- `max_hf_topk_id_logit_diff <= 1e-4`

The BF16 external-kernel lane is an orthogonal boundary. It keeps identity checks
exact and only relaxes the max-logit gate for experimental BF16 q/k/v input
routes (for example packed decode/FLA-FP16-style external kernels). It must not
be used to promote serving.

Latest BF16 packed-decode lane status:

- reference path artifact:
  `results/qwen08_jax_packed_decode_reference_bf16qkv_fp32math_long_decode_top5_compare_20260527.json`
  passes exact identity with `max_hf_topk_id_logit_diff =
  2.6702880859375e-05`; this is a semantic scaffold only.
- local CUDA path artifact:
  `results/qwen08_jax_packed_decode_cuda_bf16qkv_fp32out_long_decode_top5_compare_20260527.json`
  fails the lane (`499/500` top-1, `491/500` ordered top-5,
  `max_hf_topk_id_logit_diff = 0.02606964111328125`). Do not use this route for
  speed benchmarking or serving promotion.
- JAX-Triton packed-decode artifact:
  `results/qwen08_jax_packed_decode_triton_fla_bf16qkv_fp32out_long_decode_top5_compare_20260528.json`
  fails the lane (`499/500` top-1, `493/500` ordered top-5,
  `max_hf_topk_id_logit_diff = 0.029291152954101562`).
- JAX-Triton prefill-prep artifact:
  `results/qwen08_jax_gdn_prefill_triton_fla_prep_bf16_jax_gate_long_decode_top5_compare_20260528.json`
  keeps generated-token/top-1 identity (`500/500`) but fails top-5/logit parity
  (`496/500` ordered top-5, `499/500` top-5 set,
  `max_hf_topk_id_logit_diff = 0.008432388305664062`).

Regenerate the default HF long-prefill JSON on CUDA with:

```bash
env TMPDIR=/mountpoint/.exp/tmp \
XDG_CACHE_HOME=/mountpoint/.exp/.cache \
HF_HOME=/mountpoint/.exp/.cache/huggingface \
HF_HUB_CACHE=/mountpoint/.exp/.cache/huggingface/hub \
TOKENIZERS_PARALLELISM=false \
uv run python benchmarks/precompute_hf_prompt_reference.py \
  --model Qwen/Qwen3.5-0.8B \
  --dtype float32 \
  --weight-dtype bfloat16 \
  --input-lens 512,1024,1536,2048 \
  --output-len 16 \
  --prompt-suite mixed \
  --prompt-source tokenized_seed_repeat \
  --top-k 5 \
  --output-json results/qwen08_hf_bf16w_fp32act_long_prefill_512_2048x16.json
```

For the BF16-activation diagnostic artifact, change `--dtype float32` to
`--dtype bfloat16` and write
`results/qwen08_hf_bf16w_bf16act_long_prefill_512_2048x16.json`.

Latest kernel-phase revalidation:

- artifact: `results/qwen08_jax_bf16w_fp32act_long_decode_top5_compare_20260526_kernel_phase_gate.json`
- git head: `d66b285`
- exact top-1/top-5 ranks: pass, `500/500`
- numeric logit-diff bound: fail narrowly, `max_hf_topk_id_logit_diff =
  2.09808349609375e-05` versus the required `2e-5`

Do not use the segmented GDN override gate for a serving promotion until either
a candidate passes this exact bound or we make an explicit threshold decision.
The "exact bound" here means the default path: exact token/top-k identity and
`max_hf_topk_id_logit_diff <= 2e-5`.

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
