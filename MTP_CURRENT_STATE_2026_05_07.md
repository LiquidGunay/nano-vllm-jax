# MTP current state and open issues

Date: 2026-05-07

## Repo state

The repo now contains the serving, verifier, benchmark, correctness, and notes changes from the TPU MTP investigation. The current pushed head is `bb4da8b` on `main`.

The committed work includes:

- Mixed-length batching support across scheduler, KV cache, model runner, and executor paths.
- TPU benchmark scripts for baseline decode, MTP K=1, correctness checks, and profiling splits.
- Correctness guardrails that suppress throughput reporting when MTP diverges from the baseline sequence.
- HF/logit comparison support and expanded correctness reporting.
- MTP notes covering K=1 validation, mixed-length fused verifier behavior, vLLM comparison attempts, and the forced reuse-fallback experiment.

## Validated behavior

The baseline path is now treated as the canonical implementation for correctness. MTP runs are only considered throughput-valid when they exactly match baseline token output.

The conservative mixed-length fused K=1 verifier is exact on the tested prompt suites, including mixed prompt lengths. This is the current safe MTP path.

Prefill warmup and bucket selection matter heavily. With full warmup and a larger prefill bucket, prefill time is no longer misleadingly different between baseline and MTP. Remaining decode comparisons should use warmed runs and should separate prefill and decode throughput.

## Current performance picture

On Qwen/Qwen3.5-2B with real weights on the TPU, batch 4, mixed prompt lengths 32/64/96/128, `max_tokens=40`, prefill bucket 256:

- Baseline decode reached about `416 tok/s`.
- Conservative exact MTP K=1 decode remained slower than baseline.
- Forced reuse fallback with bonus emission reached about `212 tok/s` decode, but failed exact token matching and is not a valid throughput result.

The main conclusion is that the correctness-safe K=1 implementation has not yet produced a decode speedup over baseline on the tested small/dense models.

## Main issues

The one-pass verifier can preserve correctness in the conservative all-or-none path, but rowwise acceptance and more aggressive state reuse still drift. The latest observed reuse-fallback failure diverged at request 3, token 25.

K=1 speculative decode has limited theoretical headroom because each successful step can add at most one extra token. If the verifier or cache/state repair path adds non-trivial overhead, the speedup is easily erased, especially for small models where baseline decode is already fast.

The MTP implementation still has correctness risk around accepted-state installation, rejected-row repair, hybrid prefix state handling, and bonus-token emission. These paths need stricter equivalence checks against baseline/HF logits before they should be used for speed measurements.

Mixed-length serving is not yet a production scheduler. The repo has components for homogeneous handling of heterogeneous requests, but a robust serving design still needs continuous batching semantics, warmed shape buckets, prefill/decode interleaving policy, preemption policy when KV is exhausted, and minimized host/device sync.

vLLM TPU comparison is not complete. The repo has notes and partial benchmark attempts, but a clean apples-to-apples vLLM baseline and speculative decode comparison still needs to be run under the same model, prompt distribution, batch size, max tokens, warmup, and correctness constraints.

## Recommended next steps

1. Keep the conservative exact verifier as the correctness reference.
2. Build a focused correctness test for accepted-state installation and rejected-row repair over longer generated sequences.
3. Only re-enable rowwise acceptance after it passes exact token matching and logit sanity checks.
4. Profile decode-only latency by phase after warmup: scheduler, TPU executable time, cache update, host sampling, and release/postprocess.
5. Benchmark larger dense models where verifier overhead is a smaller fraction of baseline decode cost.
6. Run vLLM TPU baseline and speculative benchmarks with the same prompt suite and report decode throughput separately from prefill throughput.
