# Documentation Index

## Stable docs

| File | Purpose |
| --- | --- |
| [README.md](README.md) | Short project overview and entry points |
| [docs/architecture.md](docs/architecture.md) | Engine loop, scheduler, executor, backend boundary, and invariants |
| [docs/kv_cache.md](docs/kv_cache.md) | Block tables, slot mapping, valid masks, and block-boundary examples |
| [docs/mtp.md](docs/mtp.md) | Target/draft/bonus token semantics, accept/reject, and state commit rules |
| [docs/scheduler.md](docs/scheduler.md) | Prefill chunks, decode buckets, zero-length rows, preemption, and postprocess |
| [docs/benchmarks.md](docs/benchmarks.md) | Exact command lines, warmup rules, and valid/invalid result criteria |
| [docs/roadmap.md](docs/roadmap.md) | Correctness-first roadmap |

## Historical status archives

Dated MTP, benchmark, and pre-current-state summary docs are archived in `docs/archive/2026-05-pre-current-state/`.

## Maintenance rules

- Keep current architecture and correctness guidance in `docs/`.
- Keep dated one-off run logs in `docs/archive/2026-05-pre-current-state/`.
- Do not update benchmark claims without a correctness-passing TPU run.
- Do not run local compute-heavy tests or benchmarks for documentation cleanup.
