# TPU MTP benchmark results

Environment:

- TPU VM: `nano-vllm-tpu-2404-run`
- Zone: `europe-west4-a`
- Project: `project-b9551f07-5f68-491a-8a0`
- Model: `Qwen/Qwen3.5-0.8B`
- Batch: 100 parallel prompts
- Prompt lengths: 32 to 96 tokens
- Output length: 64 tokens
- JAX platform: TPU
- Execution: JIT
- KV cache blocks: 1024
- Per-sequence block-table width: 10 blocks

## Current best exact results

| Mode | E2E tok/s | Decode tok/s | Decode latency p50 | Acceptance | Speedup vs baseline decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1361.3 | 1989.2 | ~0.50 ms/token | n/a | 1.000x |
| MTP K=1 fast all-accept | 1190.9 | 1682.2 | ~0.59 ms/token | 96.875% | 0.846x |
| MTP K=1 prefix-safe | 1362.3 baseline run | 1498.1 | ~0.67 ms/token | 96.875% | 0.752x |
| MTP K=2 chain | 1051.8 | 1422.3 | ~0.70 ms/token | 78.57% | 0.720x |

MTP remains slower than baseline in the exact serving loop, despite high first-token acceptance.

## Improvements made during this pass

- Replaced per-sequence Python/JAX hybrid-state dict rebuilds with a device-resident hybrid-state table.
- Added decode lookahead block allocation so MTP accepted steps have KV block capacity at block boundaries.
- Avoided expensive Python-built compact state batches when storing accepted MTP hybrid state.
- Fixed benchmark warmup so it compiles the fused MTP verifier before measurement.
- Added optional experimental `NANO_VLLM_JAX_MTP_COMMIT_SELECT=1` path for device-side commit selection.
- Added prefix hybrid-state extraction for the two-token verifier so rejected K=1 rows can be committed safely on device.
- Removed hot-path metadata rebuild from speculative decode; `self.kv_state` is now a compatibility snapshot while `cache_storage`, per-step metadata, and the hybrid-state table remain authoritative.
- Added a fast K=1 all-accepted verifier that avoids prefix-state extraction and falls back to the safe path only if a rejection appears.
- Fixed K>1 decode to use cached recurrent/conv state for `seq_len <= 1 + num_speculative_tokens`.
- Added per-draft-position acceptance counters to diagnose recursive MTP chains.

## Before/after snapshot

| State | Baseline decode tok/s | MTP decode tok/s | MTP decode speedup |
| --- | ---: | ---: | ---: |
| Before hybrid table/lookahead | ~1058 | ~801 | ~0.76x |
| After hybrid table | ~1995 | ~1055 | ~0.53x |
| After lookahead + hybrid-store fix + warmup | ~1995 | ~1711 | ~0.86x |
| Prefix-safe commit selection | ~1992 | ~1498 | ~0.75x |
| Fast K=1 all-accept path | ~1989 | ~1682 | ~0.85x |

The baseline now substantially exceeds the earlier vLLM TPU-inference 80% target for this 0.8B benchmark.

## Bottleneck diagnosis

Executor-level component timings for batch 100:

| Component | p50 latency |
| --- | ---: |
| Target T=1 decode, last logits | 17.3 ms |
| Target T=2 decode, all logits | 26.3 ms |
| Full MTP K=1 verifier | 27.7 ms |

Earlier steady fused MTP serving split before prefix-safe commit:

| Step component | Approx latency |
| --- | ---: |
| Hybrid gather | 2.5 ms |
| Fused MTP verifier | 12.6 ms |
| Acceptance scalar sync | 25.6 ms |
| Hybrid state store | 3.1 ms |
| Metadata refresh | 1.3 ms |
| Accepted token transfer | 0.4 ms |

After prefix-safe commit, host acceptance sync was removed, but MTP was still slower because prefix-state extraction increased verifier work. The fast K=1 path restored the cheaper all-accepted case, but exact MTP1 still does more work than baseline.

Current step-profile comparison for a short batch-100 run:

| Mode | Decode tok/s | Runner run avg | Accepted step avg | Notes |
| --- | ---: | ---: | ---: | --- |
| Baseline | 1982.3 | 105.4 ms/step | n/a | One target decode plus one vocab argmax per row |
| MTP K=1 fast | 1549.4 | 220.0 ms/step | 121.9 ms accepted-step run time | Two-token target decode plus target verify argmax, target bonus argmax, and MTP next-draft argmax |

The measured full benchmark is better than the short profile (`1682.2 tok/s`), but still below baseline.

## Commit-select experiment

I added an optional `NANO_VLLM_JAX_MTP_COMMIT_SELECT=1` path that avoids host-side accept gating by running two one-token target decodes inside one JIT and selecting committed hybrid state on device.

Short benchmark result:

- Correct: yes
- Decode tok/s: ~1168
- Decode speedup: ~0.59x

This confirms the design is correctness-preserving, but two one-token decodes cost more than the scalar host sync it removes. The default remains the faster fused two-token verifier.

## K>1 chain diagnosis

Recursive use of the single MTP1 head is not viable for this checkpoint.

Short K=2 acceptance-position counters:

- Draft position 1: `100/100` accepted after fixing cached decode for `seq_len=3`.
- Draft position 2: `0/100` accepted.

So K=2/3/4 cannot produce a speedup with this MTP1 head. Wider speculative verification would need true MTP2+ heads, not recursively applying the same MTP1 head.

## Current conclusion

Exact greedy MTP1 on `Qwen/Qwen3.5-0.8B` is still slower than the baseline pure-JAX TPU path because accepted K=1 steps require three full-vocabulary argmax projections:

- target logits at token 0 to verify the draft
- target logits at token 1 to produce the bonus token
- MTP logits to seed the next draft

Baseline decode needs only one full-vocabulary argmax projection per step.

On this small model, the extra vocab projections and MTP layer dominate the saved scheduler/dispatch overhead. This is a model/algorithm cost issue, not a TPU setup issue.

Likely paths to an actual speedup:

- Use a larger target model where one full target decode is much more expensive relative to the MTP head/vocab projection.
- Use true MTP2+ heads so a single verifier can commit multiple accepted draft positions.
- Add an optimized TPU/Pallas backend for attention and GDN so the target two-token decode amortizes better.
- Explore explicitly approximate modes, for example optimistic acceptance or reduced-vocab draft projection, but those are not exact greedy parity.
