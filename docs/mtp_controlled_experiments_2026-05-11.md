# K=1 MTP controlled experiments, 2026-05-11

This file records the controlled TPU experiments for the current K=1 MTP speedup work.

## Setup

- Hardware: TPU v6e-1 VM, `nano-vllm-jax-spot-v6e2-1527`
- Backend: pure JAX/XLA on TPU
- Dtype: BF16
- Execution: JIT
- Correctness gates: exact token match and next-step sanity
- Primary workload: `Qwen/Qwen3.5-0.8B`, B=4, prompt length 16, output length 16
- Common MTP mode: K=1, rowwise acceptance, warmed shapes, real weights

Every result below passed exact token match and next-step sanity for its own same-shape baseline.

## Results

| Experiment | Model / shape | Change tested | Baseline decode tok/s | MTP decode tok/s | Speedup | Acceptance | Accepted p50 ms/tok | Rejected p50 ms/tok | Fallback p50 ms/tok |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `safe_forced_0p8b_b4` | 0.8B B=4 | Current exact forced K=1 path | 362.36 | 275.08 | 0.759x | 62.5% | 2.78 | 5.43 | 5.67 |
| `afterdraft_final_0p8b_b4` | 0.8B B=4 | Accepted rows use final verifier state, not token-1 prefix state | 336.49 | 261.69 | 0.778x | 62.5% | 2.87 | 5.66 | 6.20 |
| `prefix_first_only_0p8b_b4` | 0.8B B=4 | Temporary experiment: materialize token-0 prefix state only | 354.29 | 138.04 | 0.390x | 62.5% | 2.99 | 5.84 | 6.10 |
| `native_width2_0p8b_b4` | 0.8B B=4 | Disable width-1 decode math | 351.42 | 279.96 | 0.797x | 62.5% | 2.69 | 5.28 | 5.78 |
| `fast_all_accept_0p8b_b4` | 0.8B B=4 | Cheap fast verifier, repair rejected rows | 362.08 | 239.71 | 0.662x | 62.5% | 2.75 | 9.69 | 5.63 |
| `commit_select_0p8b_b4` | 0.8B B=4 | Explicit commit-select verifier | 362.40 | 273.63 | 0.755x | 62.5% | 2.81 | 5.19 | 5.81 |
| `measured_gate_0p8b_b4` | 0.8B B=4 | Scheduler speed gate at `min_speedup=1.0` | 369.94 | 370.66 | 1.002x | 0.0% | 0.00 | 0.00 | 2.59 |
| `safe_forced_0p8b_b16` | 0.8B B=16 | Current exact forced K=1 at larger batch | 822.18 | 546.17 | 0.664x | 57.1% | 1.17 | 2.29 | 3.21 |
| `safe_forced_4b_b4` | 4B B=4 | Current exact forced K=1 at larger model | 211.09 | 169.39 | 0.802x | 62.5% | 4.68 | 9.04 | 8.56 |
| `native_width2_4b_b4` | 4B B=4 | Larger model with native width-2 verifier math | 210.25 | 188.35 | 0.896x | 62.5% | 4.06 | 7.53 | n/a |
| `native_width2_4b_b1_manual` | 4B B=1 | Small-batch manual prompt, native width-2 | 70.91 | 70.67 | 0.997x | 62.5% | 11.43 | 21.80 | n/a |
| `safe_width1_4b_b1_synthetic` | 4B B=1 | High-acceptance synthetic prompt, safe width-1 | 69.91 | 79.17 | 1.132x | 87.5% | 12.35 | 0.00 | n/a |
| `native_width2_4b_b1_synthetic` | 4B B=1 | High-acceptance synthetic prompt, native width-2 | 70.58 | 84.67 | 1.200x | 87.5% | 11.50 | 0.00 | n/a |
| `safe_width1_4b_b1_synthetic_gate` | 4B B=1 | Same synthetic prompt with default measured gate | 68.82 | 77.85 | 1.131x | 87.5% | n/a | 0.00 | n/a |

## Per-change interpretation

### Current exact forced K=1 path

Result: valid but slower, `0.759x` on 0.8B B=4.

The accepted path emits two tokens and is slightly worse than baseline per emitted token:

```text
baseline B=4 ms/tok ~= 1000 / 362.36 = 2.76
accepted K=1 p50 ms/tok = 2.78
```

Rejected and fallback paths are roughly 2x baseline per emitted token. This makes forced K=1 lose even with `62.5%` acceptance.

### Native width-2 decode math

Result: valid for B=4 and modestly faster than the safe default, `0.797x` vs `0.759x`.

Effect:

```text
relative improvement over safe forced B=4 ~= 279.96 / 275.08 = 1.018x
absolute speedup change = +0.038x
```

This is not a safe serving default. Earlier B=16 tests showed native width-2 decode math can diverge from the same-shape baseline; width-1 decode math is still required for robust exactness across larger physical batch shapes.

### Accepted rows use final verifier state

Result: exact and roughly neutral, `0.778x` in this run.

This change removes the need to read token-1 prefix hybrid state for accepted rows. Accepted rows can use `updated_hybrid_state`, the final state after the two-token verifier. Rejected rows still need token-0 prefix state.

This is retained because it simplifies the invariant:

```text
accepted row: final verifier state
rejected row: token-0 prefix verifier state
```

The measured speed is within run-to-run noise of the safe forced path and does not solve the speedup problem by itself.

### Token-0-only prefix materialization

Result: exact but much slower, `0.390x`; not retained.

The experiment tried to materialize only token-0 prefix state and broadcast it through the prefix-state shape because accepted rows no longer need token-1 prefix state. XLA produced a much worse graph:

```text
safe forced MTP decode = 275.08 tok/s
prefix-first-only MTP decode = 138.04 tok/s
```

This rules out the naive "return less prefix state by broadcasting token 0" implementation. A useful implementation would need a separate verifier return type or kernel path, not shape-compatible dummy prefix state.

### Fast all-accept verifier

Result: exact but slower, `0.662x`.

The accepted p50 latency is comparable to the safe path, but rejected p50 latency is much worse:

```text
safe rejected p50 = 5.43 ms/tok
fast-all-accept rejected p50 = 9.69 ms/tok
```

This confirms the current fast verifier is not useful unless it can also commit rejected rows safely. The repair decode cost dominates.

### Commit-select verifier

Result: exact but not faster, `0.755x`.

Commit-select and current safe forced K=1 are effectively tied:

```text
safe forced speedup = 0.759x
commit-select speedup = 0.755x
```

This suggests explicit sequential current-token state selection is correct, but it does not reduce accepted-step cost enough to beat baseline.

### Measured speed gate

Result: exact and effectively no worse, `1.002x`.

The gate disabled speculative decode after measuring that MTP was slower:

```text
accepted steps = 0
rejected steps = 0
fallback steps = 15
```

This matches the vLLM-style serving principle: speculative decode should not remain enabled when measured throughput is below baseline. It does not provide an MTP speedup; it prevents regression.

### Larger batch B=16

Result: exact but slower, `0.664x`.

Baseline benefits much more from B=16 batching than MTP:

```text
B=4 baseline = 362.36 tok/s
B=16 baseline = 822.18 tok/s
B=4 MTP = 275.08 tok/s
B=16 MTP = 546.17 tok/s
```

The exact verifier path scales, but not enough to keep up with baseline decode. Acceptance also fell from `62.5%` to `57.1%`.

### Larger model 4B

Result: exact but still slower, `0.802x`.

The larger model improves the ratio compared with 0.8B B=4, but not enough. Accepted-step p50 is still around baseline per emitted token, and rejected/fallback paths are still roughly 2x baseline.

### 4B native width-2 verifier math

Result: exact at B=4 and closer to baseline, `0.896x`, but still not a speedup.

Native width-2 verifier math reduces accepted and rejected latency:

```text
safe 4B B=4 accepted p50 = 4.68 ms/tok
native 4B B=4 accepted p50 = 4.06 ms/tok
safe 4B B=4 rejected p50 = 9.04 ms/tok
native 4B B=4 rejected p50 = 7.53 ms/tok
```

This is not robust as a global default. The same native width-2 mode still fails exact token matching at 0.8B B=16. It can be considered only for narrow shapes where same-shape parity is proven.

### First valid speedup: high-acceptance 4B B=1

Result: exact K=1 MTP speedup exists when acceptance is high enough.

The synthetic B=1 4B prompt suite produced `87.5%` acceptance and no rejected steps. Results:

```text
safe width-1:   69.91 -> 79.17 tok/s, 1.132x
native width-2: 70.58 -> 84.67 tok/s, 1.200x
```

The default measured gate also preserved the speedup on this case:

```text
baseline decode = 68.82 tok/s
gated MTP decode = 77.85 tok/s
decode speedup = 1.131x
scheduler measured speedup = 1.229x
accepted decode steps = 7
rejected decode steps = 0
fallback decode steps = 1
```

Manual and real B=1 4B prompts stayed near parity but did not cross it:

```text
manual: 0.997x at 62.5% acceptance
real:   0.991x at 62.5% acceptance
```

This changes the conclusion: K=1 speedup is achievable, but only for high-acceptance small-batch workloads with the current executor. For ordinary prompts around `62.5%` acceptance, the rejected/fallback overhead still erases the gain.

## Break-even calculation

For K=1, using emitted-token latency:

```text
average_spec_ms_per_token = (2 * p * A + (1 - p) * R) / (1 + p)
```

Where:

- `p` is acceptance rate,
- `A` is accepted-step ms per emitted token,
- `R` is rejected-step ms per emitted token.

For the current safe 0.8B B=4 run:

```text
baseline B ~= 2.76 ms/tok
A ~= 2.78 ms/tok
R ~= 5.43 ms/tok
```

Since `A` is already slightly worse than baseline, acceptance alone cannot reliably produce a speedup. Even perfect acceptance would be approximately:

```text
best_case_speedup ~= B / A ~= 2.76 / 2.78 = 0.99x
```

For 4B B=4:

```text
baseline B ~= 4.74 ms/tok
A ~= 4.68 ms/tok
best_case_speedup ~= 1.01x before any fallback/rejected cost
```

The 4B threshold is slightly better but still too tight for real serving because rejected/fallback steps are much slower.

## Conclusion

The controlled experiments show a valid K=1 MTP speedup only in high-acceptance small-batch 4B workloads.

The viable next implementation target is narrow:

```text
Preserve the cheap accepted verifier path, but produce baseline-equivalent
after-current-token KV and hybrid state for rejected rows without repair decode.
```

The current blockers are:

- accepted K=1 emitted-token latency is not meaningfully below baseline,
- rejected/fallback rows are around 2x baseline latency,
- native width-2 math helps only slightly and is not robust across batch shapes,
- larger batch sizes improve baseline more than MTP,
- larger model size helps the ratio, and becomes profitable only when acceptance is high enough.

The current default measured gate is the correct serving behavior until the executor has a cheaper exact verifier path.
