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
