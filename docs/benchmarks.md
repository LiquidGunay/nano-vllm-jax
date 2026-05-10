# Benchmarks

Benchmark results are valid only when correctness, warmup, and hardware rules are followed. Keep this file concise; store detailed logs in dated findings docs when needed.

## Current benchmark target

Current validated benchmark target:

- TPU v6e-1,
- `Qwen/Qwen3.5-4B`, BF16, real weights,
- JIT execution on the pure JAX/XLA TPU backend,
- K=1 MTP with scheduler-owned admission,
- correctness checks enabled.

CPU runs are useful for smoke tests only. Do not use CPU numbers to claim MTP throughput.

## Required benchmark rules

A result is reportable only if it states:

- model id and dtype,
- hardware and backend,
- JAX execution mode,
- prompt workload,
- batch size and prompt length,
- generated token limit,
- speculative token count,
- MTP policy/env flags when used,
- warmup behavior,
- correctness result,
- baseline decode tokens/sec,
- MTP decode tokens/sec,
- acceptance rate.

## Warmup

JAX compile time must be excluded from steady-state decode throughput. Use warmup for every reported shape.

Valid warmup expectations:

- compile happens before measured decode timing,
- baseline and MTP paths both warm their JIT shapes,
- repeated measurements reuse the same process where practical,
- first-run compile latency is reported separately if relevant.

## Valid results

A result can be used for serving decisions only when:

- real checkpoint weights are loaded,
- target token correctness is true,
- next-step sanity is true when the benchmark supports it,
- baseline and MTP use comparable prompts and max tokens,
- decode tokens/sec excludes prefill and compile time,
- MTP enablement is based on measured emitted-token throughput, not acceptance alone.

## Invalid results

Do not use results for serving decisions when they have any of these properties:

- random or synthetic model weights,
- CPU-only throughput numbers,
- compile time included in decode throughput without disclosure,
- missing baseline,
- missing correctness checks,
- unsafe fused MTP flags used as if they were the serving path,
- acceptance rate used as a substitute for measured decode speedup.

## Latest TPU result table

Latest preserved findings from `docs/mtp_tpu_spot_findings_2026-05-09.md`:

| workload | mode | baseline tok/s | MTP tok/s | speedup | acceptance | correctness |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| synthetic low-acceptance | K=1 MTP | 62.56 | 62.28 | 0.996x | 50.0% | true |
| manual counting high-acceptance | K=1 MTP | 61.09 | 67.52 | 1.105x | 72.2% | true |
| manual counting high-acceptance, per-bucket gate | K=1 MTP | 64.48 | 72.84 | 1.130x | 61.1% | true |
| mixed arrivals B=4 | K=1 MTP | 104.33 | 52.41 | 0.502x | 36.4% | true |
| interleaved B=4 after forced-reject probes | K=1 MTP | 95.25 | 89.66 | 0.941x | 0.0% | true |

Interpretation:

- K=1 can be neutral or faster depending on workload and measured latency.
- Acceptance alone is insufficient; serving gates must use measured decode throughput.
- Mixed/heterogeneous B=4 is still below baseline; the latest forced-reject probe patch reduces fallback-heavy overhead but does not create speedup when acceptance is low.
- K=2 is correctness-clean but slower in observed benchmarks, so it remains experimental/non-serving.

## Where to put detailed logs

Use dated findings files for full logs and one-off investigations. Keep `docs/benchmarks.md` to rules, criteria, and the current compact result table.
