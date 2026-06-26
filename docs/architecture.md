# Architecture

This document describes the cleaned runtime flow and ownership boundaries.

## Flow

```text
EngineService -> LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> model.forward_step
```

`server.py` is only transport and CLI. It parses requests, validates capacity,
and submits work to `EngineService`.

`EngineService` owns cross-request admission. Handler threads enqueue work; one
worker advances the engine and publishes token events or final results.

`LLMEngine` owns request lifecycle:

- create `Sequence` objects,
- call the scheduler,
- call the runner,
- postprocess finished requests,
- release runner/cache state.

## Scheduling

`Scheduler` owns dynamic Python serving state:

- waiting and running queues,
- prompt chunk selection,
- decode row selection,
- inactive-row padding,
- block allocation and preemption,
- prefix-cache lookup and publication.

`BlockManager` owns physical cache page ids, reference counts, and prefix-cache
metadata.

`ScheduledBatch` is the Python-to-JAX contract. It documents the fixed-shape
arrays that the runner and executor consume.

## Execution

`ModelRunner` owns session state around compiled execution:

- full-attention KV cache arrays,
- GDN hybrid-state slots,
- resident decode metadata,
- device token carry,
- compile-bucket lookup.

`ModelExecutor` owns JIT cache keys and calls into `model.forward_step`.

`model.py` owns the Qwen3.5 layer loop and high-level attention, GDN, and
LM-head helpers. Low-level promoted kernels live under `nanovllm_jax/kernels/`.

## Invariant

```text
Logical length, block-table capacity, full-attention KV writes, and GDN hybrid
state all advance by the same committed prefix.
```

That invariant is the main correctness rule for scheduler, block manager,
runner, executor, and output materialization work.
