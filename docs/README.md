# Documentation Index

Start here for current documentation. Archived notes may be useful for history, but they are not current serving guidance.

## Canonical docs

- [Architecture](architecture.md): engine loop, scheduler/runner/executor/backend boundaries, state ownership.
- [GPU correctness guardrails](gpu_correctness_guardrails.md): current GPU correctness contract, runtime cache roots, and guard commands.
- [KV cache](kv_cache.md): block tables, slot mapping, masks, block-boundary behavior.
- [MTP](mtp.md): historical TPU K=1 terminology and invariants, with current GPU caveats.
- [Scheduler](scheduler.md): prefill chunks, decode buckets, inactive rows, preemption, MTP lookahead.
- [Benchmarks](benchmarks.md): GPU benchmark rules, warmup, server-shape recipe, current GPU caveats, and historical TPU table.
- [Optimization logbook](optimization_logbook.md): profile-linked optimization runs and decisions.
- [Roadmap](roadmap.md): pending correctness, serving, benchmark, optimization, and cleanup work.

## Current GPU findings

- [GPU correctness guardrails](gpu_correctness_guardrails.md): current BF16-weight/FP32-activation reference and guard commands.

## Historical findings retained

- [MTP TPU spot findings - 2026-05-09](mtp_tpu_spot_findings_2026-05-09.md): historical TPU findings and preserved benchmark numbers; do not use these as current GPU serving evidence.
- [K=1 MTP controlled experiments - 2026-05-11](mtp_controlled_experiments_2026-05-11.md): historical TPU experiments for MTP speedup analysis.

## Archive

- [2026-05 pre-latency-gate archive](archive/2026-05-pre-latency-gate/): obsolete debug plans, historical status notes, and duplicated root documentation.
