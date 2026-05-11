# MTP Speculative Decoding

MTP is experimental in this repository. The current serving policy is K=1 one-pass MTP only, with scheduler-owned admission and acceptance plus measured decode-latency EWMA gates.

Do not describe the current MTP path as production-ready.

## Current validated state

Validated on 2026-05-09 through 2026-05-11 repo state:

- hardware: TPU v6e-1,
- models: `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B`, BF16, real weights,
- execution: JIT on the pure JAX/XLA TPU backend,
- correctness test: `tests/test_mtp_commit_semantics.py` passed `16/16` on TPU,
- serving policy: K=1, scheduler-owned admission, latency EWMA gate.

Remaining serving gap:

```text
The measured-latency EWMA gate is global, not per bucket.
```

## Terminology

For K=1:

- target token: the token sampled from the canonical target model for the current decode position,
- draft token: the token proposed by the MTP head from target-model hidden state,
- bonus token: a draft token that is accepted and emitted after the target token in the same scheduler step,
- reject: commit only the target token,
- accept: commit the target token and the bonus token.

Avoid using K=2/K>1 terminology when describing the serving path unless explicitly discussing experimental work.

## K=1 commit invariants

Reject case:

```text
old prefix + target token
```

Accept case:

```text
old prefix + target token + accepted bonus token
```

Required invariants:

- target logits come from the canonical target forward-step contract,
- an accepted bonus token must match target-model verification semantics,
- rejected draft writes must not become logically reachable,
- full-attention KV and hybrid linear-attention state must advance by exactly the committed prefix,
- inactive bucket rows must preserve prior state,
- block allocation must include target and admitted lookahead positions before execution.

## Scheduler-owned admission

The scheduler owns whether MTP is admitted for a row. Admission should consider:

- whether the sequence is in decode, not prefill,
- whether sufficient lookahead KV capacity is allocated,
- recent acceptance behavior,
- measured decode-latency EWMA compared with baseline,
- bucket and shape eligibility.

Current implementation gates with acceptance and measured latency, but the latency EWMA is still global. Per-bucket EWMA remains pending.

## 2026-05-11 K=1 speed status

The corrected K=1 path is exact but does not yet beat baseline when MTP is forced on.

Validated TPU v6e-1 results, `Qwen/Qwen3.5-0.8B`, BF16, real weights, JIT, warmed shapes:

- homogeneous B=4, prompt length 16, output length 16: exact token match passed, next-step sanity passed, baseline decode `347.84-364.65 tok/s`, forced K=1 decode `271.43-282.93 tok/s`, speedup `0.776-0.780x`, acceptance `62.5%`.
- mixed/interleaved B=4, prompt lengths `16,17,31,32`, arrivals `0,0,2,4`, output length 12: exact token match passed, next-step sanity passed, baseline decode `238.71 tok/s`, forced K=1 decode `141.88 tok/s`, speedup `0.594x`, acceptance `38.9%`.
- measured-speed gate with `NANO_VLLM_JAX_MTP_MIN_SPEEDUP=1.0`: exact token match passed and decode throughput was effectively parity, `362.76 tok/s` baseline vs `362.72 tok/s` gated K=1, because admission disabled speculative decode after measured throughput was below threshold.

Current blocker:

```text
Accepted K=1 steps can be slightly faster per emitted token than baseline.
Rejected and fallback K=1 steps are much slower per emitted token, so forced MTP
needs a high acceptance rate to break even.
```

Do not seed a follow-up K=1 draft after a rejected row unless the rejected-row next-draft state invariant is proven. The current safe policy commits the target token and leaves no draft behind for rejected K=1 rows.

The next implementation target is a safe fast verifier that preserves the cheap accepted path while exposing an after-current-token state for rejected rows. Without that prefix state, fast rejected rows must either be repaired or left uncommitted, which removes the expected K=1 speedup.

## Current K=2 status

K=2 is correctness-clean in focused TPU semantics testing, but slower in observed serving benchmarks. Treat it as experimental and non-serving.

K=2 should stay disabled for serving until it has:

- robust long-generation exactness across workloads,
- per-bucket latency evidence showing a real throughput benefit,
- clear state-commit handling for partial acceptance,
- benchmark coverage comparable to K=1.

## Unsafe fused paths

Historical unsafe fused one-pass experiments showed useful diagnostics but are not the serving reference. Width-dependent TPU BF16 numerics and hidden-state drift made those paths unsuitable for exact serving unless guarded by explicit experimental flags and correctness checks.

Current canonical docs should describe the gated K=1 path, not older unsafe fused experiments.
