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
| `native_width2_0p8b_b4` | 0.8B B=4 | Disable width-1 decode math | 351.42 | 279.96 | 0.797x | 62.5% | 2.69 | 5.28 | 5.78 |
| `fast_all_accept_0p8b_b4` | 0.8B B=4 | Cheap fast verifier, repair rejected rows | 362.08 | 239.71 | 0.662x | 62.5% | 2.75 | 9.69 | 5.63 |
| `commit_select_0p8b_b4` | 0.8B B=4 | Explicit commit-select verifier | 362.40 | 273.63 | 0.755x | 62.5% | 2.81 | 5.19 | 5.81 |
| `measured_gate_0p8b_b4` | 0.8B B=4 | Scheduler speed gate at `min_speedup=1.0` | 369.94 | 370.66 | 1.002x | 0.0% | 0.00 | 0.00 | 2.59 |
| `safe_forced_0p8b_b16` | 0.8B B=16 | Current exact forced K=1 at larger batch | 822.18 | 546.17 | 0.664x | 57.1% | 1.17 | 2.29 | 3.21 |
| `safe_forced_4b_b4` | 4B B=4 | Current exact forced K=1 at larger model | 211.09 | 169.39 | 0.802x | 62.5% | 4.68 | 9.04 | 8.56 |

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

The controlled experiments do not show a forced K=1 MTP speedup with the current exact implementation.

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
- larger model size helps the ratio but does not make forced K=1 profitable.

The current default measured gate is the correct serving behavior until the executor has a cheaper exact verifier path.

