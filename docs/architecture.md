# Architecture

This document records stable runtime invariants. It intentionally avoids one-off benchmark claims and dated status conclusions.

## Engine loop

The serving loop is organized as:

```text
LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> Backend -> model.forward_step
```

Core invariant:

```text
one scheduled batch produces one postprocess decision for every scheduled sequence
```

The engine is responsible for request lifecycle. It repeatedly asks the scheduler for runnable work, calls the runner, and lets scheduler postprocess append emitted tokens and update block metadata.

## Scheduler

The scheduler owns Python-side logical state:

- waiting and running queues,
- prompt chunk selection,
- decode row selection,
- block-table allocation and preemption,
- conversion from `Sequence` objects to `ScheduledBatch`.

Scheduler invariant:

```text
all physical KV slots that a model step may write must be allocated before the executor runs
```

The scheduler does not decide logits, draft acceptance, or target-model correctness. It only decides which logical positions are executable.

## Executor

`ModelExecutor` is the canonical model execution boundary. It builds attention metadata, creates the `KVCacheState` view, calls `model.forward_step`, and returns updated cache and hybrid state.

Executor invariant:

```text
all target logits used for sampling or verification must come from the canonical forward-step contract
```

Verifier paths may request hidden states for MTP seeding, but target token decisions should use canonical logits, not independently reconstructed logits unless the reconstruction is explicitly being tested as equivalent.

## Backend boundary

The backend boundary owns operations that depend on cache layout or accelerator implementation:

- KV cache allocation,
- attention metadata construction,
- KV writes,
- full-attention prefill/decode attention,
- linear-attention recurrent decode helpers.

Backend invariant:

```text
decode attention with query length greater than one must be absolute-position causal
```

For speculative verification, a decode query can contain multiple tokens. The backend must ensure token at absolute position `p` can attend only to keys with position `<= p`.

## Model forward contract

`model.forward_step` returns either logits, hidden states, or `(hidden, logits)` depending on flags.

Important invariant:

```text
return_hidden=True and return_hidden_with_logits=True returns ((hidden, logits), kv_state, hybrid_state)
```

Callers must destructure this exactly. Misdestructuring can cause verifier code to use hidden tuples where arrays are expected or to reference undefined logits.

## State ownership

Runtime state is split across four layers:

- `Sequence` owns emitted token ids and logical length.
- `BlockManager` owns physical block ids and prefix-cache metadata.
- `KVCacheStorage` owns full-attention device KV arrays.
- `HybridLayerState` owns linear-attention recurrent and convolution state.

Global invariant:

```text
logical sequence length, block-table capacity, KV writes, and hybrid state must advance by the same accepted prefix
```
