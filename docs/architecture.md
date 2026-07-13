# Architecture

This document describes the cleaned runtime flow and ownership boundaries.

## Flow

```text
EngineService -> LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> model.forward_step
```

`server.py` is only transport and CLI. It parses requests, validates capacity,
and submits work to `EngineService`.

`EngineService` owns cross-request admission. Handler threads enqueue work; one
worker advances the engine and publishes output or cancellation transitions.
The total queued-plus-active request count is bounded. Streaming uses a
one-slot wake channel plus an output watermark, so slow clients coalesce work
instead of retaining one Python dictionary per token.

`LLMEngine` owns request lifecycle:

- create `Sequence` objects,
- call the scheduler,
- call the runner,
- commit each `RunResult` exactly once,
- release runner/cache state.

Cancellation is also an engine transition: `cancel_request()` removes a
waiting or running sequence and releases its scheduler, KV, and runner state.
Health is derived from the worker thread rather than model allocation alone,
and shutdown succeeds only after that worker has actually stopped. If a stop
deadline expires, the worker still fails remaining handles when it later exits.

The control boundary is explicit:

```text
Scheduler -> SchedulePlan
ModelRunner -> DeviceBatch -> RunResult
LLMEngine.commit() -> StepResult
```

`Sequence` owns logical positions and cache metadata. `OutputBuffer` owns every
generated token, including deferred device references. Reading logical request
state never synchronizes the accelerator; callers explicitly prefetch and
materialize output snapshots when tokens must reach the host.

## Scheduling

`Scheduler` owns dynamic Python serving state:

- waiting and running queues,
- prompt chunk selection,
- decode row selection,
- whole-request capacity reservation,
- bounded first-fit waiting admission,
- prefix-cache lookup and publication.

`BlockManager` owns physical cache page ids, reference counts, prefix-cache
metadata, and future-capacity credits. Admission reserves enough capacity for
`prompt + max_tokens`, but allocates pages only for the current prompt and as
decode crosses block boundaries. Thus future capacity cannot be stolen, yet an
unwritten completion does not evict a cached prefix or widen its block table.
Admission scans the bounded waiting queue for the first request that fits, so a
large blocked request does not stall smaller requests behind it.

Each reusable prefix is one `PrefixCacheEntry`: its chained hash, exact token
count, physical block chain, and an optional opaque GDN-state handle. The
runner owns the referenced device arrays. Reusing any block removes every
dependent entry and returns its handle to the runner; the state-handle budget
is capped by resident request capacity and included in the startup device
budget.

The scheduler returns a host-only `SchedulePlan`: immutable request rows plus
the selected bucket shape. It neither imports JAX nor allocates device arrays.

## Execution

`ModelRunner` owns session state around compiled execution:

- plan padding and `DeviceBatch` materialization,
- reusable shape-stable device metadata,
- full-attention KV cache arrays,
- GDN hybrid-state slots,
- bounded cached-prefix GDN snapshots,
- resident decode metadata,
- device token carry,
- compile-bucket lookup.

FlashInfer receives a fixed-size page-index buffer for JIT stability, but its
CSR indptr exposes only each row's live page prefix. Static block-table padding
is never treated as attention context. Decode uses a reusable non-split plan,
because FlashInfer split-KV scheduler tables depend on the exact plan-time page
counts. The fused append kernel skips rows whose logical length is zero, so a
padded row cannot write through its placeholder page.

`ModelExecutor` owns JIT cache keys and calls into `model.forward_step`.

`RunResult` contains one emitted-token row per scheduled sequence. The engine
commits those rows into `OutputBuffer`, advances logical/cache state, assigns
EOS or length finish reasons, and returns a `StepResult` of token and finish
events. The service consumes those events directly instead of scanning
sequences for new output.

Checkpoint `config.json` owns model dimensions and layer types. The loader
accepts the validated Qwen3.5 0.8B, 2B, and 4B text configurations, validates
their complete math and tensor-layout architecture before downloading weight
shards, validates every loaded tensor shape, and retains vocabulary weights as
`[V, H]`. Generation terminates on the union of checkpoint and tokenizer EOS
ids because the official chat tokenizer and text config designate different
special tokens.

`model.py` owns parameter structure, the Qwen3.5 layer loop, and the exported
forward entrypoints. The math is split by role:

- `projection.py`: packed linear projection policy and helpers,
- `attention.py`: full-attention prefill/decode wrappers,
- `gdn.py`: Gated DeltaNet projection, recurrence, and state handling,
- `lm_head.py`: final norm, logits, greedy top-1, and temperature sampling.

Low-level promoted kernels live under `nanovllm_jax/kernels/`.

## Invariant

```text
Logical length, block-table capacity, full-attention KV writes, and GDN hybrid
state all advance by the same committed prefix.
```

That invariant is the main correctness rule for scheduler, block manager,
runner, executor, and output materialization work.
