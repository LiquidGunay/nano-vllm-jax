# MTP chain-repair findings

Date: 2026-05-09

All validation and benchmarks in this note were run on the TPU VM, using the remote clone at `/tmp/nano-vllm-jax-validate-2e3fbad`.

## Change summary

Added a bounded seeded-chain control for unsafe one-pass K=1 MTP:

```bash
NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN=<N>
```

When `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1`, this limits how long one-pass fused verifier state can be reused before forcing a normal decode fallback to re-canonicalize KV and hybrid state.

Also enabled KV donation for the one-pass fused verifier JIT, matching the existing commit-select JIT donation pattern.

## TPU validation

Focused MTP semantics tests pass on the TPU VM:

```text
tests/test_mtp_commit_semantics.py: 13 passed
```

## Key benchmark results

Common settings unless otherwise noted:

```bash
--config-preset hf
--prompt-suite expanded
--num-speculative-tokens 1
--compile-mtp-draft
--dtype bfloat16
--backend tpu
--jax-execution decode-jit
--prefill-buckets 128
--num-kvcache-blocks 512
--batch-size-buckets 1
--batch-prompts 1
--prompt-lengths 64
--mtp-token-source generated
NANO_VLLM_JAX_MTP_FUSED_VERIFY=1
NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1
NANO_VLLM_JAX_MTP_PREFIX_SAFE=1
NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1
NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE=1
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise
```

| Model | Mode | Max tokens | Correct | Decode speedup | E2E speedup | Acceptance | Fallbacks | First diff |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-2B | one-pass, no seeded bonus | 64 | yes | 0.895x | 1.055x | 45.45% | 25 | none |
| Qwen3.5-4B | one-pass, no seeded bonus | 64 | yes | 0.929x | 0.921x | 58.62% | 24 | none |
| Qwen3.5-4B | seeded one-pass, uncapped | 64 | no | 1.109x | 0.973x | 65.79% | 9 | token 28 |
| Qwen3.5-4B | seeded one-pass, chain cap 5, repeats=3 | 64 | yes | 1.034x | 1.067x | 59.46% | 11 | none |
| Qwen3.5-4B | seeded one-pass, chain cap 5 | 128 | no | 1.004x | 0.942x | 52.56% | 26 | token 73 |
| Qwen3.5-4B | seeded one-pass, chain cap 2 | 128 | yes | 0.972x | 0.945x | 52.86% | 38 | none |

The first confirmed exact-token decode speedup is:

```text
Qwen/Qwen3.5-4B, max_tokens=64, chain cap 5, repeats=3
decode_speedup_mean = 1.034x
decode_tps_no_spec_mean = 64.58
decode_tps_mtp_mean = 66.78
```

## Interpretation

The speed path exists, but only when seeded one-pass can reuse fused verifier state for several steps.

The correctness blocker is still fused-state drift:

- Non-seeded one-pass stays correct because accepted bonus tokens are followed by normal decode fallback, which repairs state.
- Seeded one-pass is fast because it avoids that fallback, but it persists fused prefix state.
- Persisting that state eventually diverges from the sequential commit-select state.
- A chain cap can trade speed for periodic repair.

For 4B, a cap of 5 is enough for an exact 64-token speedup, but not robust for 128 tokens. A cap of 2 is robust for 128 tokens in this benchmark, but loses decode speedup.

## Current conclusion

We have achieved an exact-token decode speedup for a bounded 4B run, but not yet a robust long-generation serving speedup.

To make this production-grade, the next fix must reduce or eliminate the one-pass fused-state drift rather than only bounding it. The likely work items are:

1. Add per-layer parity instrumentation to find the first layer where one-pass `[current, draft]` diverges from sequential commit-select.
2. Fix that state mismatch if it is an implementation issue in prefix hybrid state, decode metadata, or cached suffix handling.
3. If the drift is unavoidable TPU/XLA BF16 shape numerics, keep one-pass behind adaptive chain repair and only enable it when measured speedup exceeds the correctness repair cost.
4. For serving, gate MTP by model/bucket stats and sequence length: short generations may use cap 5 on 4B, while longer generations need a smaller cap or commit-select until fused-state parity is fixed.

## Added fused-vs-sequential parity hook

Added:

```bash
NANO_VLLM_JAX_MTP_LAYER_PARITY_DEBUG=1
```

When unsafe K=1 one-pass is active, this runs a debug-only executor path from the same pre-state before the donating production one-pass call. The path compares:

- fused decode over `[current, draft]`
- sequential decode over `current`, then `draft`

It prints one compact line per speculative step:

```text
[MTP_LAYER_PARITY] fused_one_pass_vs_seq ...
```

The line includes slot-0 and slot-1 logit/hidden max-abs differences, current/draft KV-slot max-abs differences, final linear-attention conv/recurrent-state max-abs differences, and top-5 token IDs for fused and sequential slot 0/slot 1.

Interpretation:

- `current_*_slot_max_abs` isolates full-attention KV drift at the committed current-token slot.
- `draft_*_slot_max_abs` isolates full-attention KV drift at the draft/bonus slot.
- `conv_state_max_abs` and `recurrent_state_max_abs` isolate linear-attention state drift after the draft token.
- `slot*_hidden_max_abs` with zero KV/state drift points toward numerics in non-stateful projections, MLP, norms, or attention output math.
- matching top-5 IDs with nonzero max-abs means drift is present but not yet token-changing for that step.

## Width-2 fused K=1 verifier parity conclusion (2026-05-09)

Worker A tested the unsafe one-pass fused K=1 path on TPU with Qwen3.5-4B after keeping the only directionally useful numeric patch: casting Gated DeltaNet recurrent q/k to fp32 before l2norm.

Findings:
- Fused width-2 `[current, draft]` remains numerically non-equivalent to sequential width-1 commit-select for slot0/current.
- After backing out worsened norm and full-attention tokenwise experiments, short parity returned to the candidate-1 baseline: step 1 `slot0_logit=0.255972`, `current_k=0.328125`, `current_v=0.257812`, `conv=0.5625`; step 2 `slot0_logit=0.929449`, `current_k=1.0625`, `current_v=1.9375`, `conv=2.125`.
- Compact stage instrumentation shows entry deltas are near zero, then block input RMSNorm can amplify them (`in_norm=0.125` in early linear-attention layers). Tokenwise input RMSNorm worsened parity and was backed out.
- Full-attention cache deltas matched pre-write K/V deltas, so cache scatter/write ordering was not the primary cause in the observed runs.
- Bounded unsafe one-pass seeded cap2 margin experiments on Qwen3.5-4B, prompt_len=64, max_tokens=32 did not restore next-step logit sanity:
  - margin 0: exact tokens true, next-step sanity false, acceptance 56.25%, raw decode speedup 0.170x, E2E 2.887x.
  - margin 0.5: exact tokens true, next-step sanity false, acceptance 56.25%, raw decode speedup 0.0289x, E2E 0.863x.
  - margin 1.0: exact tokens true, next-step sanity false, acceptance 56.25%, raw decode speedup 0.0292x, E2E 0.851x.

Conclusion:
- On TPU for this hybrid Qwen3.5 model, the width-2 fused verifier should be treated as numerically unsafe for exact serving even when visible tokens match.
- Exact serving must use commit-select/sequential repair, or explicitly accept hidden-state/logit drift from the unsafe one-pass path.
- K=1 speedup remains blocked unless a kernel/numerics path can make fused slot0 parity exact, especially around width-dependent RMSNorm/linear-attention amplification.

## Adaptive gating correction (2026-05-09)

The benchmark adaptive gating report now uses the same timing scope as the
reported decode speedup:

```text
measured_decode_speedup =
  mtp_decode_tokens_per_second / baseline_decode_tokens_per_second

should_enable =
  measured_decode_speedup >= 1.0 + adaptive_margin
```

The previous diagnostic formula multiplied a step/token timing ratio by
acceptance rate:

```text
(baseline_step_ms / speculative_step_ms) * (1 + accept_rate)
```

That formula is misleading for exact sequential commit-select because
`speculative_step_ms` is already measured over emitted decode tokens, while
acceptance changes emitted tokens per scheduler step. Multiplying by
`1 + accept_rate` double-counts the emitted-token multiplier and can recommend
enabling MTP even when the measured decode speedup is below 1.

Structural limitation:
- Exact K=1 commit-select performs at least one target-model forward per
  emitted token plus MTP-head and commit bookkeeping overhead.
- It can beat baseline only if batching/fusion lowers per-emitted-token target
  cost enough to offset that overhead.
- Serving gates should therefore be conservative: enable exact commit-select
  only after observed emitted-token throughput exceeds baseline by the
  configured margin.

Latest B=1 exact commit-select timing lower bound observed on TPU:
- Baseline: 63.47 decode tok/s, about 15.8 ms per emitted decode token.
- Exact K=1 commit-select: 53.88 decode tok/s, about 18.6 ms per emitted
  decode token, with 66.7% acceptance.
- Accepted steps were close to baseline per emitted token, but rejected and
  fallback repair steps remained slower because they still require the target
  forward plus MTP draft maintenance.

This means acceptance alone is not sufficient for enabling exact K=1. The
serving decision must use measured emitted-token throughput for the current
model/bucket.

## Conditional exact K=2 status (2026-05-09)

Added a correctness-preserving exact K=2 commit-select path in
`ModelExecutor`. It keeps every target-model verifier decode at width 1, then
uses device-side global `lax.cond` predicates to skip later verifier decodes
when no row accepted the previous draft.

The intended commit semantics are:

- reject first draft: commit target token only,
- accept first draft but reject second draft: commit target plus first accepted
  draft token,
- accept both drafts: commit target plus both accepted draft tokens,
- inactive padded rows remain unchanged.

TPU validation:

- `tests/test_mtp_commit_semantics.py -q`
- result: `13 passed, 1 warning`

Short warmed TPU benchmark:

- model: `Qwen/Qwen3.5-4B`
- batch: `B=1`
- prompt length: `64`
- max generated tokens: `32`
- speculative tokens: `2`
- exact tokens: true
- next-step sanity: true
- acceptance: `43.33%`
- baseline decode: about `15.64 ms/token`
- exact K=2 decode speedup: `0.905x`
- accepted p50/p95: `14.32 / 16.32 ms/token`
- rejected p50/p95: `25.09 / 25.12 ms/token`
- fallback p50/p95: `20.88 / 22.71 ms/token`

Interpretation:

- K=2 reduces host/scheduler pressure and can make accepted-token latency
  competitive with baseline.
- It still does one target-model forward for each emitted token in the exact
  path, so it remains below baseline on the measured 4B/B=1 workload.
- This confirms the main speedup path is still an exact fused multi-token
  verifier, not only deeper sequential commit-select.

## TPU matmul precision diagnostic (2026-05-09)

The unsafe fused K=1 verifier was retested with TPU matmul precision overrides.

With `JAX_DEFAULT_MATMUL_PRECISION=highest` on the short B=1 4B workload:

- exact tokens: true
- next-step sanity: true
- acceptance: `53.33%`
- decode speedup: `0.800x`

This is useful diagnostically because it shows the fused width-2 correctness
issue is strongly tied to TPU BF16 shape-dependent numerics. It is not a
serving solution because the precision mode is slower than exact K=2 on the
same class of workload.

With `JAX_DEFAULT_MATMUL_PRECISION=high`, the quick diagnostic did not complete
within the command window; the MTP warmup alone took about `35s`. Treat it as
not viable until a more targeted precision patch is found.

Current serving conclusion:

- Exact K=1/K=2 commit-select is the correctness reference.
- Unsafe fused one-pass remains opt-in only.
- A future speedup requires either making the fused verifier numerically exact
  at normal TPU precision, or adding a narrower targeted high-precision patch
  around the first layer where width-2 drift appears.
