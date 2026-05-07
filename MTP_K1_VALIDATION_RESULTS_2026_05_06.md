# MTP K=1 validation results - 2026-05-06

## Scope

- TPU VM: `nano-vllm-tpu-2404-run`
- Model: `Qwen/Qwen3.5-2B`
- Backend: JAX TPU, `decode-jit`, `bfloat16`
- Speculative mode: K=1 MTP, fused verifier, commit-select path, one-pass K=1 disabled
- Validation goal: preserve exact greedy output correctness, then measure decode throughput speedup

## Changes validated

- Added benchmark runtime reset before each generation run.
- Removed the extra measured-variant warmup between baseline and MTP runs.
- Added conservative MTP guards:
  - Mixed logical decode lengths use baseline decode.
  - Partial scheduled batches in a multi-sequence engine use baseline decode.
  - Mixed or partial prefill batches do not seed MTP drafts.

## Correctness validation

### Baseline repeatability

Command shape:

```bash
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --config-preset hf \
  --prompt-suite expanded \
  --max-tokens 40 \
  --num-speculative-tokens 0 \
  --dtype bfloat16 \
  --backend tpu \
  --jax-execution decode-jit \
  --max-num-seqs 4 \
  --batch-size-buckets 4 \
  --batch-prompts 4 \
  --prompt-lengths 32,64,96,128 \
  --max-blocks-per-seq 16 \
  --warmup \
  --repeats 1 \
  --require-tpu \
  --correctness-only
```

Result:

| Metric | Value |
| --- | ---: |
| Exact match | true |
| First diff | none |
| Baseline decode tok/s | 151.53 |
| Repeat decode tok/s | 152.15 |

### Mixed-length MTP correctness

Command shape:

```bash
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1 \
NANO_VLLM_JAX_MTP_COMMIT_SELECT=1 \
NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1=1 \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --config-preset hf \
  --prompt-suite expanded \
  --max-tokens 40 \
  --num-speculative-tokens 1 \
  --dtype bfloat16 \
  --backend tpu \
  --jax-execution decode-jit \
  --max-num-seqs 4 \
  --batch-size-buckets 4 \
  --batch-prompts 4 \
  --prompt-lengths 32,64,96,128 \
  --max-blocks-per-seq 16 \
  --warmup \
  --repeats 1 \
  --require-tpu \
  --correctness-only
```

Result:

| Metric | Value |
| --- | ---: |
| Exact match | true |
| First diff | none |
| Accepted tokens | 0 |
| Rejected tokens | 0 |
| Fallback tokens | 156 |
| Drafts proposed | 0 |
| Baseline decode tok/s | 151.06 |
| MTP-labeled decode tok/s | 150.45 |

Interpretation: mixed-length batches are now correctness-safe by suppressing MTP. This is intentionally conservative.

## Speed validation

### Homogeneous B=4 K=1 MTP

Command shape:

```bash
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1 \
NANO_VLLM_JAX_MTP_COMMIT_SELECT=1 \
NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1=1 \
python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --config-preset hf \
  --prompt "Solve: if 3x + 7 = 31, what is x? Show the arithmetic." \
  --max-tokens 40 \
  --num-speculative-tokens 1 \
  --dtype bfloat16 \
  --backend tpu \
  --jax-execution decode-jit \
  --max-num-seqs 4 \
  --batch-size-buckets 4 \
  --batch-prompts 4 \
  --prompt-lengths 96,96,96,96 \
  --max-blocks-per-seq 16 \
  --warmup \
  --repeats 1 \
  --require-tpu
```

Throughput-valid result:

| Metric | Value |
| --- | ---: |
| Exact match | true |
| Acceptance rate | 50.0% |
| Accepted tokens | 64 |
| Rejected tokens | 20 |
| Fallback tokens | 72 |
| Baseline decode tok/s | 56.15 |
| MTP decode tok/s | 266.29 |
| Decode speedup | 4.74x |
| End-to-end speedup | 1.014x |
| Prefill tokens | 1152 |
| Decode tokens | 156 |
| Prefill fraction | 88.07% |

Interpretation: decode-only speedup is achieved and token-exact. End-to-end speedup is small because this benchmark is prefill-dominated.

## Remaining work

- The current correctness guard suppresses MTP for mixed-length and partial scheduled batches.
- To regain speedup for heterogeneous serving, implement length-grouped batched verification:
  - Partition scheduled decode rows by logical `seq.num_tokens`.
  - Run fused K=1 verification per homogeneous group.
  - Use baseline decode for singleton or unsafe groups.
  - Merge outputs back into scheduler row order.
- Re-enable one-pass K=1 only after it matches the sequential commit-select verifier on the expanded correctness suite.

## Update - real serving policy experiment

The runner was changed to separate the correctness-safe serving policy from experimental mixed-length fused verification:

- Default serving policy:
  - Use fused K=1 MTP only for full homogeneous decode batches.
  - Use baseline decode for mixed-length or partial batches.
  - Do not seed MTP drafts when the next step cannot use the fused fast path.
- Experimental flags:
  - `NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1` enables mixed-length fused MTP experiments.
  - `NANO_VLLM_JAX_MTP_ENABLE_REUSE_FALLBACK=1` enables the exact but slower main-decode reuse verifier fallback.

### Mixed arbitrary lengths, safe default

Prompt lengths: `32,64,96,128`

| Metric | Value |
| --- | ---: |
| Exact match | true |
| First diff | none |
| Accepted tokens | 0 |
| Rejected tokens | 0 |
| Fallback tokens | 156 |
| Drafts proposed | 0 |
| Baseline decode tok/s | 153.57 |
| MTP-enabled decode tok/s | 150.96 |
| Decode speed ratio | 0.983x |

Interpretation: arbitrary mixed lengths are correctness-safe and avoid wasted speculative work by default.

### Mixed arbitrary lengths, reuse fallback diagnostic

Prompt lengths: `32,64,96,128`

| Metric | Value |
| --- | ---: |
| Exact match | true |
| Acceptance rate | 52.48% |
| Accepted tokens | 73 |
| Rejected tokens | 38 |
| Fallback tokens | 45 |
| Baseline decode tok/s | 149.55 |
| MTP reuse decode tok/s | 86.38 |
| Decode speed ratio | 0.578x |

Interpretation: main-decode reuse is exact for arbitrary mixed lengths, but it is slower for this 2B TPU benchmark.

### Homogeneous full batch, safe default fast path

Prompt lengths: `96,96,96,96`

| Metric | Value |
| --- | ---: |
| Exact match | true |
| Acceptance rate | 50.0% |
| Accepted tokens | 64 |
| Rejected tokens | 12 |
| Fallback tokens | 80 |
| Drafts proposed | 64 |
| Baseline decode tok/s | 56.56 |
| MTP decode tok/s | 237.49 |
| Decode speedup | 4.20x |
| End-to-end speedup | 0.998x |

Interpretation: the fast fused path still gives a large decode speedup when the batch shape is correctness-proven. End-to-end is still prefill dominated.

### Current blocker for faster arbitrary-length MTP

`NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1` activates fused mixed-length verification, but it is not yet correctness-safe. The failure appears after accepted MTP steps when visible tokens still match baseline, which points to KV/hybrid commit semantics around speculative bonus tokens and interleaved continuous batching. Until that is fixed, mixed-length serving should keep the safe default policy.

### Mixed-length fused correctness update

Date: 2026-05-07

Changes:

- Added K=1 all-or-none batch acceptance in commit-select. If any row in the active fused decode batch rejects, every row emits only the main-model target token. This fixed a minimal B2 correctness failure where one accepted row caused a rejected row to drift later.
- Added rejected-row masking for the second commit-select decode. Rejected rows are represented as zero-length rows for KV writes and hybrid-state advancement.
- Added a full-physical-batch gate for fused K=1. Partial padded decode batches fall back to baseline decode because accepted bonus tokens in partial buckets still caused delayed divergence after continuous-batching interleavings.
- Disabled MTP draft seeding for partial buckets when fused execution is gated off, avoiding most no-op MTP overhead.

Validated on TPU with `Qwen/Qwen3.5-2B`, prompt lengths `32,64,96,128`, `max_tokens=40`, `batch_size_buckets=4`, and `NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1`:

| Metric | Value |
| --- | ---: |
| Exact match | true |
| First diff | none |
| Accepted tokens | 0 |
| Rejected tokens | 0 |
| Fallback tokens | 156 |
| Acceptance rate | 0.0% |
| Baseline decode tok/s | 149.10 |
| MTP-enabled decode tok/s | 146.92 |
| Decode speed ratio | 0.985x |

Interpretation: mixed-length fused correctness is now protected by conservative gates. This is correctness-safe but does not yet provide mixed-length MTP speedup because the tested mixed workload never reaches a full physical decode bucket after the gates. The next optimization target is a correctness-safe partial-bucket verifier/commit path.
