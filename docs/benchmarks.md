# Benchmarks

Local docs cleanup must not run benchmarks. Heavy correctness and speed measurements belong on the TPU VM.

## Valid result criteria

A benchmark result is valid for speed comparison only when:

- emitted tokens exactly match the non-speculative target baseline,
- `first_diff` is `null` or equivalent,
- the run used the same prompts, max tokens, dtype, buckets, and KV capacity for baseline and MTP,
- warmup policy is stated,
- MTP environment overrides are recorded.

Invalid result criteria:

- any emitted-token mismatch,
- missing baseline comparison,
- changed prompt set between baseline and MTP,
- speedup reported from a failed correctness run,
- hidden local compile/setup time mixed into only one side of the comparison without disclosure.

## Warmup rules

Recommended TPU benchmark hygiene:

```text
1. run one untimed baseline generation
2. run one untimed MTP generation with the same shape/configuration
3. run timed baseline
4. run timed MTP
5. report prefill and decode phases separately
```

Warmup invariant:

```text
compile and cache warmup effects must not be mistaken for decode throughput
```

## Metrics to report

Required fields:

- `correct`,
- `first_diff`,
- `baseline_decode_tps`,
- `mtp_decode_tps`,
- `decode_speedup`,
- end-to-end `speedup`,
- baseline and MTP prefill seconds,
- baseline and MTP decode seconds,
- accepted/rejected/fallback token counts,
- bonus token counts,
- step-mode counts.

Interpretation invariant:

```text
decode_speedup is the primary MTP signal; end-to-end speedup can be dominated by prefill
```

## Exact command lines

HF/JAX split comparison from historical docs:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS=--xla_gpu_autotune_level=0

python benchmark_real_kv_hf.py \
  --target hf \
  --max-new-tokens 4 \
  --prompt 'Tell me a joke about compilers.' \
  --output-npz /tmp/qwen_hf_ref.npz

python benchmark_real_kv_hf.py \
  --target jax \
  --jax-execution jit \
  --prefill-bucket 16 \
  --max-new-tokens 4 \
  --max-kv-cache-mb 64 \
  --prompt 'Tell me a joke about compilers.' \
  --compare-npz /tmp/qwen_hf_ref.npz
```

Representative TPU MTP benchmark shape from archived status notes:

```bash
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1 \
NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT=1 \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --batch-size 4 \
  --prompt-lengths 32,64,96,128 \
  --max-tokens 40 \
  --prefill-buckets 256 \
  --num-kvcache-blocks 512 \
  --max-blocks-per-seq 24 \
  --warmup
```

Reuse-fallback attempt shape from archived status notes:

```bash
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1 \
NANO_VLLM_JAX_MTP_FORCE_REUSE_FALLBACK=1 \
NANO_VLLM_JAX_MTP_ENABLE_REUSE_FALLBACK=1 \
NANO_VLLM_JAX_MTP_EMIT_BONUS=1 \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --batch-size 4 \
  --prompt-lengths 32,64,96,128 \
  --max-tokens 40 \
  --prefill-buckets 256 \
  --warmup
```

The reuse-fallback result was historically invalid for throughput because correctness failed. Keep it as a regression/debug shape, not as a speed claim.

## Current interpretation

Archived reports show:

- K=1 can be exact in conservative modes.
- Correctness-preserving MTP has often been slower than baseline decode on tested workloads.
- Larger prefill buckets reduce prefill wave count and expose decode bottlenecks.
- Rowwise acceptance and reuse fallback require correctness repair before speed conclusions.
- K=2 remains under validation.

Historical raw reports are archived under `docs/archive/2026-05-pre-current-state/`.
