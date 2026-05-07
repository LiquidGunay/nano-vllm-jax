# MTP mixed-length fused notes - 2026-05-07

## Physical-bucket verifier result

Benchmark run on TPU VM:

```text
model: Qwen/Qwen3.5-2B
prompt suite: expanded
prompt lengths: 32,64,96,128
max tokens: 40
batch size bucket: 4
MTP K: 1
mode: mixed-length fused enabled, physical-bucket one-pass verifier for partial rows
```

Result:

```text
correct: true
first_diff: null
accepted_tokens: 64
rejected_tokens: 37
fallback_tokens: 55
baseline_decode_tps: 152.2117424867428
mtp_decode_tps: 49.65618976299122
total_speedup_reported: 0.9845769203832727
mtp_decode_latency_ms_per_token_p50: 28.62532899598591
mtp_decode_latency_ms_per_token_p95: 28.871090480242856
mtp_rejected_latency_ms_per_token_p50: 29.51269448385574
step_mode_counts: prefill=5, decode=80, accepted_decode=22, rejected_decode=21, fallback_decode=37
```

## Current interpretation

The correctness issue from compacting partial mixed-length batches is fixed by keeping the original physical bucket shape and masking inactive rows as zero-length work. This avoids disabling fused mixed-length MTP while preserving a homogeneous shape contract for JIT.

The remaining issue is performance: decode throughput is still lower than baseline. The physical-bucket path improves substantially over the previous compact partial verifier path, but it still pays too much per decode step for verification and fallback handling.

## Uniform one-pass verifier result

Benchmark run on TPU VM with commit-select not forced, so K=1 uses the one-pass verifier path uniformly:

```text
correct: true
first_diff: null
accepted_tokens: 64
rejected_tokens: 37
fallback_tokens: 55
baseline_decode_tps: 144.47715690360562
mtp_decode_tps: 49.55312623669852
total_speedup_reported: 1.0020632780546987
mtp_decode_latency_ms_per_token_p50: 28.636499497224577
mtp_decode_latency_ms_per_token_p95: 28.925271234766115
mtp_rejected_latency_ms_per_token_p50: 29.565038508735597
step_mode_counts: prefill=5, decode=80, accepted_decode=22, rejected_decode=21, fallback_decode=37
```

This is the preferred correctness-preserving mixed-length path so far: it keeps fused mixed-length MTP enabled, uses a single physical bucket abstraction, and avoids switching partial buckets to a separate compact verifier. The reported total throughput is essentially parity with baseline, but decode-only MTP throughput remains the target bottleneck.

## Routing patch validation

After changing the runner so `NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1` chooses the K=1 one-pass verifier even when `NANO_VLLM_JAX_MTP_COMMIT_SELECT=1`, the same benchmark was rerun:

```text
correct: true
first_diff: null
accepted_tokens: 64
rejected_tokens: 37
fallback_tokens: 55
baseline_decode_tps: 151.0915849590644
mtp_decode_tps: 49.98509494125597
total_speedup_reported: 1.002456394783325
mtp_decode_latency_ms_per_token_p50: 28.45126199827064
mtp_decode_latency_ms_per_token_p95: 28.672414046013728
mtp_rejected_latency_ms_per_token_p50: 29.444129002513364
step_mode_counts: prefill=5, decode=80, accepted_decode=22, rejected_decode=21, fallback_decode=37
```

This confirms the environment no longer silently switches full batches to commit-select while partial batches use one-pass under mixed fused mode.

## Decode-speed accounting fix

The reported `speedup` field is end-to-end wall time:

```text
speedup = baseline_seconds / mtp_seconds
```

This can hide decode regressions when prefill dominates wall time or when prefill timing noise differs between baseline and MTP runs. The benchmark now also reports:

```text
decode_speedup = mtp_decode_tps / baseline_decode_tps
baseline_seconds
mtp_seconds
baseline_prefill_seconds
mtp_prefill_seconds
```

Example from the default fast mixed-fused B4 run:

```text
correct: true
baseline_decode_tps: 155.32766136763007
mtp_decode_tps: 103.44991901996691
decode_speedup: 0.6660109223889058
speedup: 1.0877317822179915
baseline_seconds: 19.66772532137111
mtp_seconds: 18.08141091663856
baseline_prefill_seconds: 18.663396803021897
mtp_prefill_seconds: 16.5734348691185
```

Interpretation: the end-to-end wall metric looks faster because the MTP run measured lower prefill time, but decode is still slower than baseline.

## Fast-accept path

Mixed fused K=1 now uses an optimistic fast-accept verifier by default when `NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED=1`. Set `NANO_VLLM_JAX_MTP_PREFIX_SAFE=1` to force the conservative prefix verifier.

The fast path:

1. Runs the cheaper no-prefix two-token verifier.
2. Commits it only if all active rows accept.
3. Falls back to ordinary decode when the batch does not fully accept.

Validated B4 mixed lengths `32,64,96,128`:

```text
correct: true
first_diff: null
accepted_tokens: 64
rejected_tokens: 0
fallback_tokens: 92
baseline_decode_tps: 155.32766136763007
mtp_decode_tps: 103.44991901996691
decode_speedup: 0.6660109223889058
accepted step p50: 6.8463672359939665 ms/token
step_mode_counts: prefill=5, decode=80, accepted_decode=22, rejected_decode=0, fallback_decode=58
```

This improves decode TPS from about `50 tok/s` to about `103 tok/s` while preserving exact token correctness. It is still below baseline decode because too many steps fall back.

## Batch-size scaling blocker

B6 with explicit KV capacity ran correctly but all-or-none acceptance committed no speculative tokens:

```text
correct: true
accepted_tokens: 0
fallback_tokens: 234
baseline_decode_tps: 680.5882581641431
mtp_decode_tps: 146.7867002668149
decode_speedup: 0.2156762161350911
```

The baseline decode path scales strongly with larger batches. MTP does not currently scale because all-or-none acceptance probability collapses as active row count increases.

Rowwise acceptance improves acceptance but is not correct yet:

```text
correct: false
first_diff: request 2, token_index 22, baseline 8160, MTP 90700
accepted_tokens: 77
fallback_tokens: 71
baseline_decode_tps: 154.36286021540425
mtp_decode_tps: 58.340371658673455
```

Trace interpretation: divergence occurs after consecutive rejected rowwise speculative steps, pointing at rejected-row state advancement/prefix-state equivalence rather than token comparison.

## Warmed prefill measurements

The benchmark now performs a full untimed baseline generation and a full untimed MTP generation before collecting timed measurements when `--warmup` is set. This removed the previous misleading prefill difference between baseline and MTP.

With the default batched benchmark prefill bucket, the scheduler used 5 prefill chunks:

```text
prefill bucket: default batched mode, effectively 64
correct: true
baseline_prefill_seconds: 16.297124988981523
mtp_prefill_seconds: 16.630034461035393
baseline_decode_tps: 154.30067456686282
mtp_decode_tps: 103.82522421008393
decode_speedup: 0.6728760227492948
prefill_steps: 5
```

Phase profiling showed the time is inside `runner.run`, not scheduling or postprocess:

```text
baseline prefill run_ms total: 16290.934308955912
baseline prefill schedule_ms total: 4.384349973406643
```

The default chunking was the largest immediate prefill problem. With `--prefill-buckets 128`, prefill dropped to 3 chunks:

```text
correct: true
baseline_prefill_seconds: 9.844265340070706
mtp_prefill_seconds: 9.945150419021957
baseline_decode_tps: 150.73245428683458
mtp_decode_tps: 92.95728978538085
decode_speedup: 0.6167038825526509
prefill_steps: 3
```

With `--prefill-buckets 256 --num-kvcache-blocks 512 --max-blocks-per-seq 24`, the same workload prefills in one wave:

```text
correct: true
baseline_prefill_seconds: 3.347145806008484
mtp_prefill_seconds: 3.323785956017673
baseline_decode_tps: 403.7945170891642
mtp_decode_tps: 126.9346103045544
decode_speedup: 0.3143544672661448
speedup: 0.8200468864808017
accepted_tokens: 32
fallback_tokens: 124
prefill_steps: 1
```

Interpretation: prefill was mostly slow because the benchmark default forced too many prefill waves. A larger prefill bucket makes prefill substantially more sensible, although the pure-JAX prefill kernel is still slow in absolute terms at about `95.6 tok/s` for this 2B model on the single visible TPU device.

The larger prefill bucket also exposes the decode-side issue more clearly: baseline decode improves to `403.8 tok/s`, while MTP remains limited by all-or-none acceptance.

## Next work items

1. Keep the physical bucket shape for all mixed-length serving paths.
2. Use larger prefill buckets for benchmarks and serving warmup; small buckets dominate latency.
3. Fix rowwise correctness without relying on non-canonical accepted-state transitions.
4. Investigate canonical rowwise commit-select for mixed fused batches, because one-pass accepted state can drift enough to change later tokens.
5. For fast path, commit accepted rows only when the accepted state is proven equivalent; otherwise repair/canonicalize accepted rows.
6. Minimize host synchronization in per-step acceptance and token materialization.
7. Move toward a scheduler model where mixed prefill/decode work is represented as fixed-shape slots with zero-length inactive rows, not by switching implementations.
8. Re-check MTP speed after rowwise/groupwise acceptance is correct.
