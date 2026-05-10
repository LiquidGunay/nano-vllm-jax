# Architecture

This document describes the canonical runtime flow and ownership boundaries. It intentionally avoids dated benchmark conclusions.

## Canonical flow

```text
LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> Backend -> model.forward_step
```

One scheduled batch produces one executor call and one scheduler postprocess decision for every scheduled sequence row.

## Engine loop

`LLMEngine` owns request lifecycle:

- accept requests and create `Sequence` objects,
- ask the scheduler for the next runnable batch,
- call the runner,
- return generated tokens,
- finish or abort completed requests.

The engine should not duplicate scheduler decisions or model verification logic.

## Scheduler boundary

The scheduler owns Python-side serving state:

- waiting/running queues,
- prefill chunk selection,
- decode bucket row selection,
- block allocation and preemption,
- inactive-row padding,
- MTP admission and lookahead reservation.

Scheduler invariant:

```text
Every physical KV slot that a model step may write is allocated before execution.
```

For MTP, admission is scheduler-owned. The scheduler decides whether a row may run speculative decode, based on acceptance and measured decode-latency EWMA gates. The current EWMA is global, not per bucket.

## Runner boundary

`ModelRunner` owns runtime/session state around execution:

- JIT entrypoint selection,
- bucketed batch materialization,
- persistent cache and hybrid-state handles,
- MTP draft state passed between scheduler steps,
- compatibility glue around the executor.

The runner may package data for efficient JAX calls, but it should not become a second source of scheduling truth.

## Executor boundary

`ModelExecutor` is the canonical model execution boundary. It builds attention metadata, creates the `KVCacheState` view, calls `model.forward_step`, and returns updated full-attention KV and hybrid linear-attention state.

Executor invariant:

```text
Target logits used for sampling or verification come from the canonical forward-step contract.
```

Verifier paths may request hidden states for MTP seeding, but target token decisions must not depend on independently reconstructed logits unless that path is explicitly being tested as equivalent.

## Backend boundary

The backend owns cache-layout and accelerator-specific operations:

- KV cache allocation,
- slot mapping,
- attention metadata construction,
- KV writes,
- full-attention prefill/decode kernels expressed in JAX/XLA,
- linear-attention recurrent decode helpers.

Current TPU execution is a pure JAX/XLA backend running on TPU. It is not a dedicated TPU kernel backend.

Backend invariant:

```text
Decode attention with query length greater than one must be absolute-position causal.
```

For speculative verification, a decode query may contain multiple logical positions. A token at absolute position `p` may attend only to keys with position `<= p`.

## State ownership

Runtime state is split across layers:

- `Sequence` owns emitted token ids and logical length.
- `BlockManager` owns physical block ids and prefix-cache metadata.
- `KVCacheStorage` owns full-attention device KV arrays.
- `HybridLayerState` owns linear-attention recurrent/convolution state.

Global invariant:

```text
Logical length, block-table capacity, KV writes, and hybrid state advance by the same committed prefix.
```
