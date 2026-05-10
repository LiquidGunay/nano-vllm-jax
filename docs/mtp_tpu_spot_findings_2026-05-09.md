# MTP TPU Spot Findings - 2026-05-09

## TPU spot status

- Working spot TPU VM: `nano-vllm-jax-spot-v6e2-1527`
- Zone: `europe-west4-a`
- Accelerator: `v6e-1`
- Runtime: `v2-alpha-tpuv6e`
- JAX: `0.6.2`
- Backend probe: `backend tpu`, devices `[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]`

The earlier `tpu-ubuntu2204-base` spot VM reached `READY/HEALTHY` but did not expose TPU hardware. Its driver log reported `No hardware is found` and `/dev/accel*` was absent. The `v2-alpha-tpuv6e` image exposed `/dev/vfio/0` and worked with `jax[tpu]`.

## Correctness

- `tests/test_mtp_commit_semantics.py`: `13 passed`
- All benchmark rows below reported:
  - `mtp_exact_token_match: true`
  - `next_step_logit_sanity: true`
  - `correctness.all_correct: true`

## Local changes under test

- `nanovllm_jax/layers.py`
  - `rms_norm` uses fp32 accumulation and casts back to input dtype.
  - `l2norm` uses fp32 accumulation and casts back to input dtype.
- `nanovllm_jax/model.py`
  - Gated DeltaNet per-head RMSNorm uses stable fp32 accumulation.
  - Gated DeltaNet q/k normalization casts to fp32 before `l2norm`.
  - Width-1 forced matmul path is gated behind `NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_MATH`.
- `nanovllm_jax/engine/model_runner.py`
  - K=2 draft-chain carry now keeps executor-returned next draft chains after rejected/partial commits.
  - Seeded-chain accounting no longer counts the accepted bonus token as speculative carry distance.

## Benchmark configurations

Common flags:

```bash
--model Qwen/Qwen3.5-4B
--config-preset hf
--dtype bfloat16
--platform tpu
--require-tpu
--jax-execution jit
--max-kv-cache-mb 4096
--num-kvcache-blocks 256
--batch-size-buckets 1
--step-profile
--check-next-step-sanity
--warmup
```

Fast K=1 one-pass decode-mode env:

```bash
NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1
NANO_VLLM_JAX_MTP_ENABLE_ONE_PASS_K1=1
NANO_VLLM_JAX_MTP_ONE_PASS_DECODE_MODE=1
NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise
NANO_VLLM_JAX_MTP_COMMIT_SELECT=0
```

Seed-after-bonus env:

```bash
NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1
NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1=1
```

## Results

| workload | batch | K | seed-after-bonus | baseline decode tok/s | MTP decode tok/s | decode speedup | acceptance | fallback decode steps |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| synthetic red-repeat | 1 | 1 | yes | 62.50 | 57.72 | 0.924x | 47.6% | 5 |
| synthetic red-repeat, confirmed decode-mode defaults | 1 | 1 | yes | 61.84 | 61.93 | 1.001x | 47.6% | 5 |
| synthetic red-repeat | 1 | 2 | no | 64.32 | 58.46 | 0.909x | 59.1% | 10 |
| easy numbers | 1 | 1 | yes | 59.85 | 69.62 | 1.163x | 75.0% | 9 |
| easy numbers, fast-all-accept variant | 1 | 1 | yes | 64.44 | 67.46 | 1.047x | 75.0% | 9 |
| easy numbers | 1 | 2 | no | 63.88 | 60.24 | 0.943x | 83.3% | 18 |
| easy numbers | 1 | 2 | yes, after K=2 carry patch | 64.32 | 60.73 | 0.944x | 51.6% | 8 |
| easy numbers | 4 | 1 | yes | 196.05 | 182.00 | 0.928x | 70.3% | 7 |
| easy numbers, fast-all-accept variant | 4 | 1 | yes | 200.34 | 173.45 | 0.866x | 70.3% | 11 |

Easy numbers prompt:

```text
Continue the sequence exactly: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
```

Generated continuation for the speedup row:

```text
 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
```

## Current interpretation

- A correctness-clean MTP speedup exists on the 4B model for a high-acceptance B=1 workload: `1.163x` decode speedup with K=1 seed-after-bonus.
- A marginal correctness-clean MTP speedup also exists on the synthetic B=1 smoke workload when the actual decode-mode one-pass verifier is selected: `1.001x`.
- The speedup is not robust yet:
  - Synthetic B=1 is only barely above baseline and is likely within run-to-run variance.
  - B=4 remains below baseline because baseline decode batching is already much more efficient.
  - K=2 has high acceptance on the easy prompt without seed-after-bonus, but too many fallback/single-token transitions.
- The next bottleneck is draft-chain continuity and verifier overhead, not TPU availability or basic correctness.

## Next work items

1. Keep K=1 one-pass decode-mode as the explicit benchmark/serving fast-path when unsafe one-pass is selected.
2. Rework K=2 next-draft generation so full accepts keep a high-quality chain without reducing acceptance.
3. Add a traced step-mode label to benchmark JSON so fallback, rejected, K=1, and K=2 steps are distinguishable without inferring from token counts.
4. Add adaptive gating by workload/bucket using measured decode speedup, not legacy acceptance-only formulas.

## Adaptive gating update on v6e-1

All runs below used the existing spot `v6e-1` TPU VM, real
`Qwen/Qwen3.5-4B` weights, BF16, JIT execution, warmup enabled, KV caching, and
next-step sanity checks.

| workload | K | gate | baseline decode tok/s | MTP decode tok/s | decode speedup | acceptance | fallback decode steps | correct |
|---|---:|---|---:|---:|---:|---:|---:|---|
| synthetic, 64-token prompt, 32 decode tokens | 1 | min accept 0.6, samples 8 | 63.79 | 54.69 | 0.857x | 41.7% | 17 | yes |
| synthetic, 64-token prompt, 32 decode tokens | 1 | min accept 0.6, samples 4 | 63.62 | 56.35 | 0.886x | 50.0% | 22 | yes |
| synthetic, 64-token prompt, 32 decode tokens | 1 | min accept 0.6, samples 2 | 64.41 | 57.39 | 0.891x | 50.0% | 28 | yes |
| manual counting prompt, 32 decode tokens | 1 | min accept 0.6, samples 4 | 64.39 | 71.09 | 1.104x | 72.2% | 3 | yes |
| manual counting prompt, 32 decode tokens | 1 | min accept 0.6, samples 2 | 61.49 | 54.18 | 0.881x | 50.0% | 28 | yes |
| manual counting prompt, 64 decode tokens | 2 | min accept 0.6, samples 4 | 64.51 | 60.27 | 0.934x | 51.6% | 7 | yes |
| synthetic, scheduler gate + baseline bypass | 1 | min accept 0.6, samples 4 | 63.06 | 60.91 | 0.966x | 50.0% | 22 | yes |
| manual counting, scheduler gate + baseline bypass | 1 | min accept 0.6, samples 4 | 64.05 | 71.09 | 1.110x | 72.2% | 3 | yes |
| synthetic, scheduler gate + baseline bypass | 1 | min accept 0.6, samples 3 | 60.30 | 58.63 | 0.972x | 50.0% | 22 | yes |
| manual counting, scheduler gate + baseline bypass | 1 | min accept 0.6, samples 3 | 62.89 | 69.94 | 1.112x | 72.2% | 3 | yes |
| synthetic, scheduler gate + latency EWMA | 1 | min accept 0.6, samples 3, min speedup 1.0 | 62.56 | 62.28 | 0.996x | 50.0% | 22 | yes |
| manual counting, scheduler gate + latency EWMA | 1 | min accept 0.6, samples 3, min speedup 1.0 | 61.09 | 67.52 | 1.105x | 72.2% | 3 | yes |

The useful setting from this sweep is K=1 with a conservative acceptance gate
around four verified drafts. It preserves a real speedup on the high-acceptance
counting prompt while limiting speculative attempts on synthetic prompts. It
does not make low-acceptance short generations faster than baseline, because the
first few failed speculative steps still dominate a 32-token decode.

K=2 remains correctness-clean but non-winning on this setup. The accepted-token
multiplier is not high enough to pay for the wider verifier and additional
commit bookkeeping.

Current serving implication: MTP should stay adaptive and per bucket. For short
or low-confidence requests, the scheduler should either delay MTP until a prompt
class has prior acceptance statistics or disable it immediately after the first
few rejected drafts. For high-acceptance greedy streams, K=1 one-pass MTP is the
only path currently showing TPU decode speedup.

## Scheduler-owned admission update

The MTP gate now lives in the scheduler instead of only inside
`ModelRunner`. When `NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE` is set, the scheduler:

- tracks cumulative accepted/rejected draft deltas reported by the runner,
- marks each sequence with `mtp_admitted` before prefill/decode execution,
- allocates speculative lookahead slots only for admitted rows,
- suppresses prefill draft seeding when the gate is closed,
- lets the runner bypass all MTP orchestration for fully gated decode batches.

This matches the vLLM-style policy direction: speculative decoding is scheduled
only when current acceptance evidence says it should help. The current policy is
still acceptance-only, not latency-EWMA-based, so it cannot fully guarantee
non-regression. The next step is to add per-bucket latency accounting so the
gate uses measured emitted-token throughput rather than acceptance alone.

Benchmark reset now also clears scheduler MTP admission state. Warmup runs
should compile shapes, not train the adaptive gate for the measured run.

## Latency EWMA admission update

The scheduler gate now also tracks measured decode latency per emitted token:

- speculative decode EWMA is updated when a step verifies at least one draft,
- baseline decode EWMA is updated when a decode step emits tokens without a
  speculative verifier attempt,
- MTP is admitted only if both acceptance and measured speedup gates pass once
  enough samples exist.

The first implementation is global, not yet per-bucket. It still moves the
behavior closer to vLLM's practical policy: low-acceptance synthetic decode is
now within run-to-run noise of baseline (`0.996x`), while the high-acceptance
counting prompt keeps a correctness-clean decode speedup (`1.105x`).

Remaining gap: the latency gate needs bucket keys such as `(batch_size_bucket,
model_size, dtype, max_blocks_per_seq, num_speculative_tokens)`. A single global
EWMA can mix B=1 and larger batch behavior, which is exactly where vLLM-style
serving systems need separate admission decisions.

## Per-bucket gate and mixed-serving update

Follow-up date: 2026-05-10.

The scheduler now owns MTP admission per physical bucket. The bucket key is:

```text
(physical_batch_size, dtype, backend, max_blocks_per_seq, num_speculative_tokens)
```

The runner's legacy acceptance-only gate is inert; rows arrive with
`seq.mtp_admitted` already set by the scheduler. Benchmark JSON now includes the
legacy active-bucket fields plus the scheduler's per-bucket report.

Mixed final prefill rows now seed K=1 drafts even when the physical bucket is
padded or heterogeneous. This fixed the earlier `drafts_proposed = 0` mixed
arrival smoke: the same workload now exercises the verifier path and remains
correct.

Latest TPU spot numbers on `Qwen/Qwen3.5-4B`, BF16, JIT, real weights, warmup
enabled, KV caching, and next-step sanity enabled:

| workload | batch | baseline decode tok/s | MTP decode tok/s | decode speedup | acceptance | fallback decode steps | correct |
|---|---:|---:|---:|---:|---:|---:|---|
| manual counting prompt, 32 decode tokens | 1 | 64.48 | 72.84 | 1.130x | 61.1% | 7 | yes |
| mixed arrivals, prompt lengths 16/17/31/32 | 4 | 104.33 | 52.41 | 0.502x | 36.4% | 3 | yes |
| interleaved prefill/decode, prompt lengths 32/448/48/512 | 4 | 95.25 | 89.66 | 0.941x | 0.0% | 61 | yes |

The manual counting prompt still proves that K=1 MTP can produce a real decode
speedup on TPU when acceptance and measured latency are favorable:

```text
prompt: Continue the sequence exactly: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
baseline:  11, 12, 13, 14, 15, 16, 17, 18,
mtp:       11, 12, 13, 14, 15, 16, 17, 18,
```

The mixed B=4 workload remains slower because accepted speculative steps still
cost about `17 ms/token`, rejected steps cost about `35 ms/token`, and baseline
decode is about `10-12 ms/token` on these shapes. The next optimization target is
the verifier/commit path itself, not correctness.

Forced-reject probe rows were added for K=1 rowwise one-pass decode. A row with
no stored draft can ride along in an existing verifier batch with draft token
`-1`, forcing rejection while committing the target-token state and producing a
next draft. This reduces split verifier-plus-fallback overhead for mixed batches;
on the interleaved B=4 workload the always-on MTP path improved from the earlier
`68.90 tok/s` to `89.66 tok/s`, but it still does not beat the `95.25 tok/s`
baseline when acceptance is `0%`.

## Mixed B>1 layout analysis

Follow-up date: 2026-05-10.

The mixed/heterogeneous serving problem is not that logical rows have different
lengths. The scheduler already keeps a fixed physical bucket and expresses
logical work through row masks, `query_start_loc`, `seq_lens`, and block tables.
The problem is that K=1 MTP has different logical work per row:

- inactive padded row: 0 target tokens,
- baseline/probe row without a stored draft: 1 target token,
- MTP row with a stored draft: 2 target tokens, current plus draft.

The physical JAX shape can remain `[B, 2]`, but the logical verifier length must
be `{0, 1, 2}` per row. The executor now treats forced-reject probe rows
(`draft_token < 0`) as logical length 1:

- `verify_query_lens = row_query_lens + row_has_draft`,
- verifier `seq_lens` advance only for rows with a real draft lane,
- rows without a real draft cannot accept,
- dummy draft slots are not restored over potentially shared inactive slots.

This is the correct layout invariant for mixed serving: physical shape is static
for JIT, while logical work and cache ownership are row-local.

The same K=1 logical-length rule is now applied to the regular one-pass verifier
and the fast two-token verifier path. A regression test covers forced-reject
probe rows explicitly: the verifier sees a physical draft lane with token `-1`,
the row emits only the target token, the forced rejection is not counted as a
failed proposed draft, and the row still receives a next draft for the following
step.

Benchmark outcome: this layout fix is correctness-clean but not a standalone
speedup. The interleaved B=4 workload stayed below baseline within run variance.
An attempted conditional-logits version that skipped second-position vocab logits
on all-reject verifier steps also did not help; XLA control-flow overhead and the
small number of speculative steps outweighed any savings, so it was not kept.

Exact commit-select was also tested as an alternative logical layout: first
decode every row, then run the second target decode only for accepted rows. It is
semantically attractive but too slow on v6e-1 for this engine:

| workload | path | baseline decode tok/s | MTP decode tok/s | speedup | acceptance | correct |
|---|---|---:|---:|---:|---:|---|
| interleaved B=4, low acceptance | commit-select | 99.07 | 89.64 | 0.905x | 0.0% | yes |
| counting B=4, high acceptance | commit-select | 213.99 | 81.28 | 0.380x | 60.6% reported, 80.0% bucket | yes |

Conclusion: for B>1, the one-pass verifier is still the best current path, but
its two-token target-model cost is too close to or above 2x the one-token
baseline cost. A robust B>1 MTP speedup needs real kernel/logit-path work:

- avoid full second-position LM-head work for rows that cannot accept without
  adding expensive dynamic control flow,
- specialize a packed verifier kernel where logical token count, not rectangular
  `[B, 2]` width, drives expensive layer work,
- keep per-bucket adaptive gating so low-acceptance B>1 traffic quickly falls
  back to baseline.
