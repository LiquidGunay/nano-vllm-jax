# MTP reuse-fallback verifier attempt

Date: 2026-05-07

Command shape: Qwen/Qwen3.5-2B, K=1, batch 4, mixed prompt lengths 32/64/96/128, max_tokens=40, prefill bucket 256, full warmup, real HF weights on TPU.

Environment overrides:

- `NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1`
- `NANO_VLLM_JAX_MTP_FORCE_REUSE_FALLBACK=1`
- `NANO_VLLM_JAX_MTP_ENABLE_REUSE_FALLBACK=1`
- `NANO_VLLM_JAX_MTP_EMIT_BONUS=1`

Result: invalid for throughput because exact token matching failed.

- First mismatch: request 3, token 25, baseline token 25, MTP token 1414.
- Baseline decode throughput: 416.10 tok/s.
- MTP decode throughput: 212.21 tok/s.
- Decode speedup: 0.510x.
- End-to-end speedup: 0.909x, but suppressed because correctness failed.
- Baseline prefill: 3.301s.
- MTP prefill: 3.309s.
- Accepted tokens: 129.
- Rejected tokens: 12.
- Fallback tokens: 15.
- Speculative drafts proposed/accepted/rejected: 75 / 41 / 21.
- Bonus tokens emitted: 41.
- Step modes: 1 prefill, 29 decode, 22 accepted, 3 rejected, 4 fallback.

Learning: forcing the canonical reuse fallback path is not yet a safe speed path. With bonus emission enabled it can diverge, and even before considering correctness its decode throughput is still about half of baseline on this 2B run. The valid path remains the conservative mixed fused all-or-none verifier, which is exact but decode-slower than baseline for K=1 on the tested small/dense models.
