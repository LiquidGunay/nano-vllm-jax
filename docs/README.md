# Documentation Index

Start here for current documentation. Archived notes may be useful for history, but they are not current serving guidance.

## Canonical docs

- [Architecture](architecture.md): engine loop, scheduler/runner/executor/backend boundaries, state ownership.
- [KV cache](kv_cache.md): block tables, slot mapping, masks, block-boundary behavior.
- [MTP](mtp.md): K=1 terminology, accept/reject invariants, scheduler-owned admission, K=2 status.
- [Scheduler](scheduler.md): prefill chunks, decode buckets, inactive rows, preemption, MTP lookahead.
- [Benchmarks](benchmarks.md): benchmark rules, warmup, valid/invalid results, latest compact TPU table.
- [Roadmap](roadmap.md): pending correctness, serving, benchmark, optimization, and cleanup work.

## Current findings retained

- [MTP TPU spot findings - 2026-05-09](mtp_tpu_spot_findings_2026-05-09.md): latest detailed TPU findings and preserved benchmark numbers.

## Archive

- [2026-05 pre-latency-gate archive](archive/2026-05-pre-latency-gate/): obsolete debug plans, historical status notes, and duplicated root documentation.
