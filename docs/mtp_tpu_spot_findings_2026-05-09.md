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
| synthetic red-repeat | 1 | 2 | no | 64.32 | 58.46 | 0.909x | 59.1% | 10 |
| easy numbers | 1 | 1 | yes | 59.85 | 69.62 | 1.163x | 75.0% | 9 |
| easy numbers | 1 | 2 | no | 63.88 | 60.24 | 0.943x | 83.3% | 18 |
| easy numbers | 1 | 2 | yes, after K=2 carry patch | 64.32 | 60.73 | 0.944x | 51.6% | 8 |
| easy numbers | 4 | 1 | yes | 196.05 | 182.00 | 0.928x | 70.3% | 7 |

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
- The speedup is not robust yet:
  - Synthetic B=1 remains below baseline even with fewer fallback steps.
  - B=4 remains below baseline because baseline decode batching is already much more efficient.
  - K=2 has high acceptance on the easy prompt without seed-after-bonus, but too many fallback/single-token transitions.
- The next bottleneck is draft-chain continuity and verifier overhead, not TPU availability or basic correctness.

## Next work items

1. Make K=1 one-pass decode-mode the explicit benchmark/serving fast-path when unsafe one-pass is selected.
2. Rework K=2 next-draft generation so full accepts keep a high-quality chain without reducing acceptance.
3. Add a traced step-mode label to benchmark JSON so fallback, rejected, K=1, and K=2 steps are distinguishable without inferring from token counts.
4. Add adaptive gating by workload/bucket using measured decode speedup, not legacy acceptance-only formulas.
