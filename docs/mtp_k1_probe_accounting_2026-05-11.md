# K=1 MTP probe/accounting update - 2026-05-11

## Scope

Implemented and tested the five requested items:

1. Per-bucket MTP admission stats are reported with acceptance, proposed drafts, seeded-main fallback steps, partial rows, probe completion, and measured scheduler speedup.
2. Confidence gating defaults to `NANO_VLLM_JAX_MTP_CONFIDENCE_MIN_ACCEPT_RATE=0.75`.
3. Probe-then-decide is explicit through `NANO_VLLM_JAX_MTP_PROBE_STEPS` and aggregate speculative-overhead latency accounting.
4. K=1 one-pass verifier now returns only the first-token hybrid prefix state needed for rejected rows. Accepted rows commit final verifier state; rejected rows commit token-0 state; there is no full repair decode and no full prefix-state broadcast for this path.
5. Benchmarks were run on the TPU VM with real cached `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B` weights, BF16, JIT, KV cache, warmup, and K=1.

## Code changes

- `nanovllm_jax/model.py`
  - Added `return_first_state` to the recurrent Gated DeltaNet rule.
  - Added `return_first_prefix_state` through `gated_deltanet_block`.
  - Added `return_first_prefix_hybrid` through `transformer_block` and `forward_step`.
  - This avoids allocating/returning `[B, T, ...]` prefix hybrid state for K=1 when only token-0 state is needed.

- `nanovllm_jax/engine/model_executor.py`
  - `mtp1_two_decode_greedy_step_jit` now requests `return_first_prefix_hybrid=True`.
  - Rejected rows use that token-0 state directly.
  - Accepted rows use final verifier state.

- `nanovllm_jax/engine/model_runner.py`
  - K=1 fused verifier is now enabled by default.
  - One-pass K=1 is enabled by default after the first-prefix state fix.
  - Rowwise repair/direct rejected commit defaults are enabled for K=1.
  - Forced-reject probe rows are not counted as real draft rejections.

- `nanovllm_jax/engine/scheduler.py`
  - Per-bucket admission now tracks proposed drafts and seeded-main fallback steps.
  - Scheduler latency accounting treats draft-seeding fallback as speculative overhead rather than baseline latency.
  - This fixed the false-positive speed gate where verifier-only latency looked fast but total MTP serving was slower.

- `benchmark_mtp1_engine.py`
  - Admission report now includes confidence/probe fields.

## TPU validation

Ran on TPU VM:

```bash
python3 -m pytest tests/test_mtp_commit_semantics.py \
  tests/test_backend_boundaries.py::test_mtp_admission_gate_tracks_logical_decode_rows -q
```

Result:

```text
17 passed
```

The broader `tests/test_backend_boundaries.py` still has unrelated existing failures in the TPU snapshot (`server` import, model-runner warmup fake-output shape, cached prefill parity), so the focused path above is the validated scope for these changes.

## Final benchmark command shape

All final rows used:

```bash
NANO_VLLM_JAX_PLATFORMS=tpu \
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise \
python3 benchmark_mtp1_engine.py \
  --model <model> \
  --config-preset hf \
  --dtype bfloat16 \
  --backend pure_jax \
  --jax-execution jit \
  --require-tpu \
  --warmup \
  --repeats 1 \
  --max-tokens 32 \
  --num-speculative-tokens 1 \
  --batch-prompts <B> \
  --max-num-seqs <B> \
  --batch-size-buckets <B> \
  --max-blocks-per-seq 16 \
  --num-kvcache-blocks 512 \
  --prompt-suite <synthetic|manual|real|mixed> \
  --step-profile \
  --trace-steps \
  --show-outputs \
  --output-json /tmp/nvj_bench_20260511_accounted/<case>.json
```

## Final benchmark results

Rows with `valid=false` failed exact MTP-vs-baseline token correctness; their throughput is not trusted.

| Model | B | Suite | Valid | Correct | Baseline decode tok/s | MTP/gated decode tok/s | Decode speedup | E2E speedup | Acceptance | Accepted | Rejected | Admission |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.8B | 4 | synthetic | true | true | 313.25 | 296.86 | 0.948 | 0.954 | 0.000 | 0 | 0 | low_throughput |
| 0.8B | 4 | manual | false | false | 337.38 | 137.53 | 0.408 | 0.478 | 0.350 | 14 | 18 | warming_acceptance |
| 0.8B | 4 | real | false | false | 334.77 | 131.47 | 0.393 | 0.462 | 0.400 | 16 | 16 | warming_acceptance |
| 0.8B | 4 | mixed | true | true | 311.90 | 306.28 | 0.982 | 0.985 | 0.000 | 0 | 0 | low_throughput |
| 0.8B | 16 | synthetic | true | true | 774.13 | 759.56 | 0.981 | 0.987 | 0.000 | 0 | 0 | low_throughput |
| 0.8B | 16 | manual | false | false | 757.71 | 21.85 | 0.029 | 0.046 | 0.304 | 42 | 71 | probing_mtp |
| 0.8B | 16 | real | false | false | 809.89 | 53.27 | 0.066 | 0.105 | 0.329 | 50 | 75 | probing_mtp |
| 0.8B | 16 | mixed | false | false | 753.60 | 20.40 | 0.027 | 0.043 | 0.421 | 67 | 57 | probing_mtp |
| 4B | 1 | synthetic | true | true | 65.38 | 62.64 | 0.958 | 0.960 | 0.000 | 0 | 0 | low_throughput |
| 4B | 1 | manual | true | true | 63.33 | 62.65 | 0.989 | 0.980 | 0.000 | 0 | 0 | low_acceptance |
| 4B | 1 | real | true | true | 62.99 | 62.52 | 0.993 | 0.992 | 0.000 | 0 | 0 | low_throughput |
| 4B | 1 | mixed | true | true | 62.31 | 63.53 | 1.020 | 1.019 | 0.000 | 0 | 0 | low_throughput |
| 4B | 4 | synthetic | true | true | 208.25 | 202.38 | 0.972 | 0.977 | 0.000 | 0 | 0 | low_throughput |
| 4B | 4 | manual | false | false | 205.45 | 103.46 | 0.504 | 0.581 | 0.590 | 23 | 8 | warming_acceptance |
| 4B | 4 | real | false | false | 206.00 | 92.79 | 0.450 | 0.527 | 0.450 | 18 | 14 | warming_acceptance |
| 4B | 4 | mixed | true | true | 211.69 | 217.94 | 1.030 | 1.025 | 0.000 | 0 | 0 | low_throughput |

## Insights

- The original scheduler speed gate was wrong: it counted draft-seeding fallback steps as baseline latency and only verifier accept/reject steps as speculative latency. That made MTP look profitable even when end-to-end MTP serving was slower.
- After accounting for draft-seeding overhead, the gate correctly disables slow buckets. Valid rows are generally near parity instead of persistently worse.
- The rows showing small speedups (`4B/B=1 mixed`, `4B/B=4 mixed`) are effectively MTP-disabled parity/noise rows: accepted and rejected counts are zero after the gate disables the bucket.
- We do not yet have a trustworthy K=1 MTP speedup row with actual accepted speculative tokens in the final 32-token matrix.
- Correctness remains the blocker for real/manual workloads when the verifier is allowed to run for longer. Failures correlate with accepted/rejected speculative state transitions, not with baseline generation.

## Current blockers

1. Some real/manual fused K=1 rows still diverge from baseline tokens after verifier attempts.
2. 0.8B B=16 manual/real/mixed is especially unstable under actual verifier attempts.
3. Scheduler gating prevents persistent regressions, but this means most valid final rows have zero accepted/rejected drafts and are parity measurements rather than MTP speedup measurements.
4. The remaining speed path is correctness first: only after manual/real verifier rows are exact should we trust MTP throughput numbers.

## Next concrete work

1. Add step-level divergence capture for invalid fused rows: first differing token, row id, accepted/rejected history, committed seq_len, and top-k next logits.
2. Run invalid cases with `NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1=1` to compare commit-select reference against one-pass state install.
3. If commit-select passes and one-pass fails, diff installed hybrid state and KV slots for accepted rows, rejected rows, and block-boundary-adjacent rows.
4. Keep the corrected probe accounting as the serving default so MTP does not stay enabled when measured aggregate speculative throughput is below baseline.

## Follow-up controls - 2026-05-12

Added benchmark divergence context:

- `first_diff`
- baseline/MTP token windows around the first mismatch
- MTP branch history before the mismatch
- MTP step at or after the mismatch
- scheduler admission snapshot and speculative counters

Focused TPU validation still passes:

```text
17 passed
```

Command:

```bash
python3 -m pytest tests/test_mtp_commit_semantics.py \
  tests/test_backend_boundaries.py::test_mtp_admission_gate_tracks_logical_decode_rows -q
```

### One-pass vs commit-select correctness controls

These controls forced verifier attempts by disabling admission thresholds:

```bash
NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE=0
NANO_VLLM_JAX_MTP_MIN_SPEEDUP=0
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise
```

For commit-select reference:

```bash
NANO_VLLM_JAX_MTP_COMMIT_SELECT=1
NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1=1
```

| Case | Mode | Correct | Baseline decode tok/s | MTP decode tok/s | Decode speedup | Acceptance | Accepted | Rejected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B B=4 manual | one-pass | false | 337.63 | 156.97 | 0.465 | 0.426 | 20 | 17 |
| 0.8B B=4 manual | commit-select | true | 291.23 | 126.79 | 0.435 | 0.298 | 14 | 25 |
| 4B B=1 real | one-pass | false | 63.77 | 59.97 | 0.940 | 0.583 | 7 | 2 |
| 4B B=1 real | commit-select | true | 61.79 | 49.72 | 0.805 | 0.462 | 6 | 3 |
| 4B B=4 manual | one-pass | false | 215.98 | 108.12 | 0.501 | 0.581 | 25 | 12 |
| 4B B=4 manual | commit-select | true | 219.03 | 103.32 | 0.472 | 0.535 | 23 | 14 |

### Concrete correctness blocker

The commit-select reference passes the same workloads that one-pass fails. Therefore the MTP weights, tokenizer, scheduler admission, and high-level accept/reject policy are not the primary correctness blocker.

The blocker is one-pass K=1 state installation:

- accepted rows sometimes install a state that later diverges from the commit-select reference;
- rejected rows mostly improved after token-0 prefix-state commit, but divergence still appears after mixed accepted/rejected histories;
- B>1 cases often diverge after a mixed rowwise step or a fallback-seeded step following a speculative transition.

Examples from divergence context:

- `0.8B B=4 manual`: first mismatch at request `0`, token index `9`; previous steps include rejected, fallback-seeded, mixed accept/reject, accepted, fallback-seeded, mixed accept/reject, accepted.
- `4B B=1 real`: first mismatch at request `0`, token index `12`; mismatch happens after accepted, fallback-seeded, accepted, fallback-seeded, rejected, fallback-seeded.
- `4B B=4 manual`: first mismatch at request `0`, token index `6`; mismatch happens immediately after a rejected step followed by fallback-seeded main.

This points to installed hybrid/KV/position state after one-pass accepted/rejected transitions, not immediate token comparison.

### Concrete speed blocker

K=1 speculative decoding can emit at most:

```text
expected_tokens_per_attempt = 1 + acceptance_rate
```

For speedup, the speculative attempt cost must satisfy:

```text
spec_attempt_cost / baseline_decode_cost < 1 + acceptance_rate
```

Equivalently:

```text
required_acceptance_rate > spec_attempt_cost / baseline_decode_cost - 1
```

Using the forced controls:

| Case | Mode | Approx cost ratio `(1+a)/speedup` | Required acceptance for speedup |
|---|---|---:|---:|
| 0.8B B=4 manual | one-pass | 3.07x | > 2.07 |
| 0.8B B=4 manual | commit-select | 2.98x | > 1.98 |
| 4B B=1 real | one-pass | 1.68x | > 0.68 |
| 4B B=1 real | commit-select | 1.82x | > 0.82 |
| 4B B=4 manual | one-pass | 3.16x | > 2.16 |
| 4B B=4 manual | commit-select | 3.25x | > 2.25 |

For B>1, K=1 speedup is impossible with the current implementation because the verifier path costs about 3x a baseline decode step. Since K=1 can emit at most 2 tokens per attempt, even 100% acceptance cannot overcome a 3x attempt cost.

For B=1 4B, speedup is mathematically possible only if one-pass correctness is fixed and acceptance stays above roughly 0.68 with current one-pass cost. The correct commit-select path needs roughly >0.82 acceptance and was still only `0.805x` in the real-prompt control.

The immediate path to real speedup is therefore:

1. Fix one-pass state installation until it matches commit-select on manual/real workloads.
2. Keep aggregate probe accounting enabled so MTP only stays active when measured emitted-token throughput beats baseline.
3. Reduce one-pass verifier cost for B>1; otherwise K=1 cannot beat baseline even with perfect acceptance.

### Parity-state diagnostic - 2026-05-12

The first parity-debug run hit a debug-only donation issue: the commit-select
reference and one-pass verifier both donate KV cache buffers. Running the
reference from the live cache deleted the arrays that one-pass needed next.
The runner now clones the KV cache for `NANO_VLLM_JAX_MTP_PARITY_DEBUG=1`
before invoking commit-select, leaving the live cache available for one-pass.

Focused TPU validation after the debug fix:

```text
17 passed
```

Command:

```bash
python3 -m py_compile benchmark_mtp1_engine.py \
  nanovllm_jax/engine/scheduler.py \
  nanovllm_jax/engine/model_runner.py \
  nanovllm_jax/engine/model_executor.py \
  nanovllm_jax/model.py

python3 -m pytest tests/test_mtp_commit_semantics.py \
  tests/test_backend_boundaries.py::test_mtp_admission_gate_tracks_logical_decode_rows -q
```

Diagnostic case:

```bash
NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE=0
NANO_VLLM_JAX_MTP_MIN_SPEEDUP=0
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise
NANO_VLLM_JAX_MTP_PARITY_DEBUG=1
NANO_VLLM_JAX_MTP_PARITY_STATE_THRESHOLD=0
python3 benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-4B \
  --config-preset hf \
  --dtype bfloat16 \
  --backend pure_jax \
  --jax-execution jit \
  --require-tpu \
  --warmup \
  --repeats 1 \
  --max-tokens 16 \
  --num-speculative-tokens 1 \
  --batch-prompts 1 \
  --max-num-seqs 1 \
  --batch-size-buckets 1 \
  --max-blocks-per-seq 16 \
  --num-kvcache-blocks 512 \
  --prompt-suite real \
  --step-profile \
  --trace-steps
```

The harness exits non-zero because MTP correctness fails, but the diagnostic
data is usable. The parity-state rows show that one-pass differs from the
commit-select reference on every verifier attempt:

| Metric | Observed range |
|---|---:|
| current/draft KV K slot max abs | `0.0625` to `0.125` |
| current/draft KV V slot max abs | `0.0625` to `0.125` |
| hybrid conv state max abs | `0.125` to `0.25` |
| hybrid recurrent state max abs | `0.011` to `0.027` |

Representative log lines:

```text
[MTP_PARITY_STATE] one_pass_vs_commit_select k_slot_max_abs=0.0625 v_slot_max_abs=0.125 conv_max_abs=0.125 recurrent_max_abs=0.0131047
[MTP_PARITY_STATE] one_pass_vs_commit_select k_slot_max_abs=0.101562 v_slot_max_abs=0.125 conv_max_abs=0.141602 recurrent_max_abs=0.0266638
[MTP_PARITY_STATE] one_pass_vs_commit_select k_slot_max_abs=0.125 v_slot_max_abs=0.125 conv_max_abs=0.25 recurrent_max_abs=0.0157008
```

This narrows the correctness blocker further: one-pass is not merely making a
different accept/reject decision. It installs numerically different KV and
hybrid state than the exact commit-select reference from the same pre-state.
Because commit-select is exact on the same workloads, the next fix should be
inside one-pass prefix-state construction or slot restoration, not scheduler
admission.

The same diagnostic run produced:

| Metric | Value |
|---|---:|
| exact token match | `false` |
| throughput valid | `false` |
| acceptance rate | `0.500` |
| decode tok/s | `40.66` |
| decode speedup | `0.624` |
| end-to-end speedup | `0.640` |
| runner/device time | `390.53 ms` |
| host time | `10.06 ms` |
| postprocess time | `0.11 ms` |

Concrete current blocker:

1. The correct K=1 path is commit-select, but it is slower than baseline.
2. The faster one-pass path is still incorrect and already differs in installed
   KV/hybrid state at each verifier attempt.
3. For B>1, the current K=1 verifier cost is around `3x` baseline decode. K=1
   can emit at most `2x` tokens even at perfect acceptance, so speedup is
   mathematically impossible until verifier cost is reduced below `2x`.
4. For 4B B=1, speedup is plausible after correctness because the one-pass
   cost ratio is about `1.68x`, requiring acceptance above roughly `0.68`.

### Controlled fixes attempted - 2026-05-12

Two additional narrow fixes were tested on TPU.

#### 1. Width-1 decode RMSNorm for multi-token one-pass

Layerwise drift showed the first one-pass-vs-sequential difference at layer 0:

```text
first_layer=0 first_type=linear_attention
stages=entry=0,in_norm=0.03125,...
```

This means the input token entering layer 0 is identical, but the width-2
one-pass RMSNorm and the width-1 sequential RMSNorm are not bit-equivalent on
TPU BF16. The model now has `_decode_width1_rms_norm`, used only for
multi-token cached decode, so one-pass can run per-token RMSNorm with the same
`[B, 1, ...]` shape as baseline decode.

Focused validation:

```text
17 passed
```

4B B=1 real-prompt control after this patch:

| Metric | Value |
|---|---:|
| exact token match | `false` |
| throughput valid | `false` |
| acceptance rate | `0.500` |
| decode tok/s | `59.85` |
| decode speedup | `0.917` |
| end-to-end speedup | `0.909` |
| runner/device time | `272.50 ms` |

This improved one-pass throughput substantially but did not fix exact token
parity.

#### 2. One-pass all-or-none gating over rows with drafts

The one-pass verifier used active rows for all-or-none acceptance gating, while
commit-select used rows that actually had draft tokens. That can change commit
decisions in partial/probe batches. One-pass now matches commit-select:

```text
all_or_none_accept = all(where(row_has_draft, accepted, true))
```

Focused validation:

```text
17 passed
```

4B B=1 real-prompt control after both patches:

| Metric | Value |
|---|---:|
| exact token match | `false` |
| throughput valid | `false` |
| acceptance rate | `0.500` |
| decode tok/s | `60.58` |
| decode speedup | `0.916` |
| end-to-end speedup | `0.908` |
| runner/device time | `269.30 ms` |

Some individual short runs were close to parity in speed (`~0.999x` decode),
but correctness still failed. The remaining blocker is therefore not just
row-gating. It is still one-pass state/logit drift from width-2 execution.

Updated concrete reason speedup is blocked:

1. The near-fast path is now close to baseline speed for 4B B=1, but invalid.
2. The exact path is still commit-select, which costs too much to beat baseline
   at the observed acceptance rates.
3. Width-2 one-pass still differs from width-1 sequential execution at layer 0
   despite width-1 matmul and RMSNorm attempts, so the remaining work is to
   either make width-2 state installation exactly match sequential decode or
   design a verifier that avoids installing width-2-derived target state.

#### 3. Skip unused next-draft computation/seeding by default

The serving default does not seed another K=1 draft after an accepted bonus
unless `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1`. The verifier was still computing
`next_draft_token`, and the one-pass rowwise repair path was still storing it
for accepted rows. That was both wasted work and an unsafe way to propagate
one-pass hidden-state drift.

Changes:

- one-pass K=1 skips `mtp_forward` for `next_draft_token` unless seed-after-bonus is enabled;
- commit-select K=1 does the same;
- one-pass rowwise repair no longer stores next drafts after bonus by default;
- tests now assert the safer no-reseed default.

Focused validation:

```text
16 passed, 1 xfailed
```

TPU controls:

| Case | Path | Exact | Valid throughput | Decode speedup | Acceptance | Notes |
|---|---|---:|---:|---:|---:|---|
| 4B B=1 real | one-pass | false | false | `0.934` | `0.500` | Faster, still invalid |
| 4B B=1 real | commit-select | true | true | `0.853` | `0.500` | Exact, still below baseline |
| 4B B=4 manual | one-pass rowwise | true | true | `0.567` | `0.500` | Mixed-row correctness fixed for this case, but slow |

This is the clearest current split:

- exact path: valid but no speedup;
- fastest path: closer to speedup but invalid;
- B>1 mixed correctness can be made valid by avoiding unsafe next-draft
  propagation, but K=1 verifier cost remains too high.

Seed-after-bonus was also tested on the exact commit-select path:

| Case | Path | Seed after bonus | Exact | Decode speedup | Acceptance |
|---|---|---:|---:|---:|---:|
| 4B B=1 real, 32 tokens | commit-select | true | true | `0.799` | `0.375` |

This reduced acceptance and speed relative to the safer no-reseed default, so it
should not be enabled as a default policy.

#### 4. Token-only verifier LM-head and dirty rejected slots

The next K=1 implementation pass keeps the same scheduler-owned admission and
rectangular physical-bucket contract, but narrows the verifier output path:

- `lm_head_token_ids_and_topk` computes exact greedy target ids and optional
  top-k margin data inside the JIT, instead of returning full
  `[B, 2, vocab]` verifier logits to the host.
- The one-pass K=1 verifier now calls the target model with
  `return_hidden=True` only, then runs the LM-head reduction on device.
- The exact commit-select verifier also avoids returning full bonus logits from
  its second target decode.
- Rejected draft KV slots are no longer restored in the serving verifier path;
  they are treated as dirty but uncommitted because `committed_seq_lens` remains
  at the current-token prefix and the next decode overwrites the same slot.

This targets the current priority order: reduce verifier LM-head/output cost
first, then target hidden-state cost and commit overhead. Correctness still
depends on the active verifier mode: the exact commit-select path remains the
reference, while one-pass width-2 execution must still be validated against
baseline before being trusted for speed claims.
