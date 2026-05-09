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

Adaptive MTP gating note:

```text
adaptive_mtp_gating.predicted_speedup is diagnostic only. Do not use it as an
authoritative serving enable/disable criterion when it disagrees with measured
decode_speedup from the same run. For exact commit-select K=1, prefer measured
decode_speedup under valid correctness gates. Formulaic acceptance-based
predicted speedup is only comparable when baseline and speculative milliseconds
are measured with identical scope.
```

## Serving workload suite wrapper

`benchmarks/benchmark_serving_workloads.py` orchestrates fixed serving-oriented
workload definitions through `benchmark_mtp1_engine.py` and emits one combined
JSON report plus one Markdown table. It keeps correctness gating in the existing
engine harness: timed results are valid only when exact token match passes and
the optional HF next-step logit sanity check passes. Invalid timing is preserved
as raw timing by the engine report and surfaced with `valid=false` plus an
invalid reason in the suite report.

Fixed workload definitions:

- `decode_steady_b1`: one active row, warmup, one normal prefill followed by steady decode.
- `long_output_b1`: one active row with longer prompt and longer generated output.
- `heterogeneous_b4`: four active rows with short, medium, and long prompt lengths in one batch.
- `interleaved_prefill_decode_b4`: four rows, small prefill buckets, and mixed short/long prompts so short rows can decode while long rows continue chunked prefill.
- `mixed_active_inactive_b4`: three active rows in fixed physical B4 buckets to cover inactive-row behavior.

Optional MTP modes:

- `baseline`: non-speculative target baseline with `--num-speculative-tokens 0`.
- `commit_select`: safe sequential commit-select verifier reference, with no unsafe one-pass env. The standalone engine harness supports `--num-speculative-tokens 1` and `--num-speculative-tokens 2`; the serving-workload wrapper currently uses K=1 unless extended.
- `compact_commit_select`: optional env-gated compact commit-select K=1 comparison mode using `NANO_VLLM_JAX_MTP_ENABLE_COMPACT_COMMIT_SELECT=1`; report separately from the safe full-physical-bucket `commit_select` reference.
- `unsafe_one_pass_no_seed`: unsafe one-pass K=1 verifier with `NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1`, one-pass env enabled, and no seed-after-bonus.
- `unsafe_one_pass_seeded_cap2`: unsafe one-pass K=1 verifier with seed-after-bonus enabled and `NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN=2`.
- `unsafe_one_pass_seeded_cap4`: unsafe one-pass K=1 verifier with seed-after-bonus enabled and `NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN=4`.

Unsafe one-pass validity policy:

```text
unsafe_one_pass_* speed fields are invalid unless exact-token match passes and next-step-logit sanity passes.
```

The suite wrapper enforces that policy by marking unsafe one-pass rows invalid
unless `--check-hf-logits` was used and passed. Smoke or correctness-only runs
may still preserve raw timing from the underlying engine report, but missing
throughput cells in the Markdown report are rendered as `n/a (smoke)` instead of
blank cells.

Default correctness-valid benchmark matrix:

- Workloads: `decode_steady_b1`, `heterogeneous_b4`, `long_output_b1`, `interleaved_prefill_decode_b4`.
- Modes: `baseline`, `commit_select`, `compact_commit_select`.
- Unsafe one-pass modes are opt-in diagnostics and should not be mixed into valid throughput summaries.

Metrics normalized into the suite report:

- `prefill_tok_s`
- `decode_tok_s`
- `end_to_end_tok_s`
- `decode_speedup`
- `end_to_end_speedup`
- `acceptance_rate`
- `fallback_count`
- accepted, rejected, and fallback inter-token latency p50/p95
- host, runner/device, and postprocess time
- correctness flags
- `first_diff` when invalid

Bounded TPU smoke command:

```bash
gcloud compute tpus tpu-vm ssh nano-vllm-tpu-2404-run \
  --project=project-b9551f07-5f68-491a-8a0 \
  --zone=europe-west4-a \
  --command='cd /tmp/nano-vllm-jax-validate-2e3fbad && \
    /tmp/nvj-validate-venv-system/bin/python benchmarks/benchmark_serving_workloads.py \
      --smoke \
      --model Qwen/Qwen3.5-2B \
      --backend tpu \
      --output-json /tmp/serving_workloads_smoke.json \
      --output-md /tmp/serving_workloads_smoke.md'
```

TPU-side dry-run command validation without model load or benchmark compute:

```bash
gcloud compute tpus tpu-vm ssh nano-vllm-tpu-2404-run \
  --project=project-b9551f07-5f68-491a-8a0 \
  --zone=europe-west4-a \
  --command='cd /tmp/nano-vllm-jax-validate-2e3fbad && \
    /tmp/nvj-validate-venv-system/bin/python benchmarks/benchmark_serving_workloads.py \
      --dry-run \
      --workload decode_steady_b1 \
      --mode baseline \
      --mode commit_select \
      --mode compact_commit_select \
      --output-json /tmp/serving_workloads_dry_run.json \
      --output-md /tmp/serving_workloads_dry_run.md'
```

Bounded full-suite command:

```bash
gcloud compute tpus tpu-vm ssh nano-vllm-tpu-2404-run \
  --project=project-b9551f07-5f68-491a-8a0 \
  --zone=europe-west4-a \
  --command='cd /tmp/nano-vllm-jax-validate-2e3fbad && \
    /tmp/nvj-validate-venv-system/bin/python benchmarks/benchmark_serving_workloads.py \
      --model Qwen/Qwen3.5-2B \
      --backend tpu \
      --workload decode_steady_b1 \
      --workload heterogeneous_b4 \
      --workload mixed_active_inactive_b4 \
      --mode baseline \
      --mode commit_select \
      --mode compact_commit_select \
      --check-hf-logits \
      --output-json /tmp/serving_workloads.json \
      --output-md /tmp/serving_workloads.md'
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

Additional warmed valid timed runs on the same pushed checkpoint:

| batch | prompt lengths | exact match | acceptance | decode tok/s | decode speedup | end-to-end speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `64` | pass | 42.70% | 62.77 | 0.801x | 0.902x |
| 4 mixed | `32,64,96,128` | pass | 31.95% | 175.64 | 0.688x | 0.913x |

The same continuous seeded setup failed with the fused two-token prefix verifier:
B=1 diverged at generated token 101 and B=2 all-or-none also diverged. A warmed
B=1 control without post-bonus seeding also diverged at generated token 37.
Treat all fused one-pass numbers as invalid until slot-0 verifier logits are
shown equivalent to a canonical one-token decode from the same state.
