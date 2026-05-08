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

## TPU correctness checkpoint: 2026-05-08

All commands below were run on the TPU VM with `--warmup`, `--correctness-only`,
`--require-tpu`, exact token comparison, and HF next-step logit sanity enabled.
Throughput fields are raw diagnostics only because correctness-only runs mark
timed results invalid by design.

### Prefix-safe forced reject, B2

Configuration:

```bash
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1 \
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1 \
NANO_VLLM_JAX_MTP_PREFIX_SAFE=1 \
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise \
NANO_VLLM_JAX_MTP_FORCE_REJECT=1 \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --max-tokens 32 \
  --num-speculative-tokens 1 \
  --compile-mtp-draft \
  --batch-size-buckets 2 \
  --batch-prompts 2 \
  --prompt-lengths 32,64 \
  --prefill-buckets 128 \
  --warmup \
  --correctness-only \
  --check-hf-logits
```

Result summary:

- Exact token match: true
- HF logit sanity: true
- First diff: null
- Drafts proposed: 62
- Drafts accepted: 0
- Drafts rejected: 52
- Accepted/rejected/fallback decode steps: 0 / 26 / 5
- Raw decode TPS: 51.17
- Raw decode speedup: 0.316x
- Rejected inter-token latency p50/p95: 21.45 ms / 21.57 ms
- Warmup: 6 shape runs, 29.91 s startup

### Prefix-safe rowwise, B4 physical bucket with 3 active rows

Configuration:

```bash
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1 \
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1 \
NANO_VLLM_JAX_MTP_PREFIX_SAFE=1 \
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --max-tokens 32 \
  --num-speculative-tokens 1 \
  --compile-mtp-draft \
  --max-num-seqs 4 \
  --batch-size-buckets 4 \
  --batch-prompts 3 \
  --prompt-lengths 32,64,96 \
  --prefill-buckets 128 \
  --warmup \
  --correctness-only \
  --check-hf-logits
```

Result summary:

- Exact token match: true
- HF logit sanity: true
- First diff: null
- Drafts proposed: 49
- Drafts accepted: 22
- Drafts rejected: 8
- Acceptance rate: 44.90%
- Accepted/rejected/fallback decode steps: 10 / 0 / 14
- Raw decode TPS: 96.30
- Raw decode speedup: 0.536x
- Accepted inter-token latency p50/p95: 12.02 ms / 15.08 ms
- Fallback inter-token latency p50/p95: 8.91 ms / 9.71 ms
- Warmup: 8 shape runs, 75.13 s startup

### Prefix-safe rowwise, B2, 128 generated tokens/request

Configuration:

```bash
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1 \
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1 \
NANO_VLLM_JAX_MTP_PREFIX_SAFE=1 \
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --max-tokens 128 \
  --num-speculative-tokens 1 \
  --compile-mtp-draft \
  --batch-size-buckets 2 \
  --batch-prompts 2 \
  --prompt-lengths 32,64 \
  --prefill-buckets 128 \
  --max-blocks-per-seq 24 \
  --warmup \
  --correctness-only \
  --check-hf-logits
```

Result summary:

- Exact token match: true
- HF logit sanity: true
- First diff: null
- Completion tokens: 128 / 128 for both baseline and MTP
- Drafts proposed: 146
- Drafts accepted: 54
- Drafts rejected: 23
- Acceptance rate: 36.99%
- Accepted/rejected/fallback decode steps: 36 / 3 / 63
- Raw decode TPS: 83.05
- Raw decode speedup: 0.543x
- Accepted inter-token latency p50/p95: 13.10 ms / 15.25 ms
- Rejected inter-token latency p50/p95: 22.92 ms / 22.97 ms
- Fallback inter-token latency p50/p95: 10.36 ms / 10.69 ms
- Warmup: 6 shape runs, 35.16 s startup

## Warmed TPU correctness checkpoint: seeded K=1 MTP

Continuous seeded bonus drafts are valid only through the sequential
commit-select verifier as of this checkpoint.

Environment:

```bash
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1
NANO_VLLM_JAX_MTP_PREFIX_SAFE=1
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise
NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1
```

Command shape:

```bash
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --config-preset hf \
  --prompt-suite expanded \
  --max-tokens 128 \
  --num-speculative-tokens 1 \
  --compile-mtp-draft \
  --dtype bfloat16 \
  --backend tpu \
  --jax-execution decode-jit \
  --prefill-buckets 128 \
  --num-kvcache-blocks 512 \
  --max-blocks-per-seq 24 \
  --require-tpu \
  --warmup \
  --correctness-only \
  --check-hf-logits
```

Results:

| batch | prompt lengths | exact match | HF sanity | acceptance | raw decode speedup |
| --- | --- | --- | --- | ---: | ---: |
| 1 | `64` | pass | pass | 42.70% | 0.865x |
| 2 mixed | `32,64` | pass | pass | 44.32% | 0.777x |

The same B=2 mixed run without `--correctness-only` produced valid timed
metrics after the correctness gate passed:

- prefill throughput: 36.73 tok/s
- decode throughput: 115.93 tok/s
- decode speedup versus baseline: 0.769x
- end-to-end speedup versus baseline: 0.887x
- acceptance rate: 44.32%
- fallback count: 30
- host time: 56.03 ms
- runner/device time: 4746.23 ms
- postprocess time: 0.76 ms

The same continuous seeded setup failed with the fused two-token prefix verifier:
B=1 diverged at generated token 101 and B=2 all-or-none also diverged. Treat
those fused one-pass seeded numbers as invalid until slot-0 verifier logits are
shown equivalent to a canonical one-token decode from the same state.
