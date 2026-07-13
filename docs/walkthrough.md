# Serving Walkthrough

This walkthrough follows one request through the promoted serving path. It is
about ownership and shapes, not historical experiments.

## First Packed Prefill

`server.py` normalizes an HTTP request into token ids and sampling parameters,
then submits it to `EngineService`.

`EngineService` owns online admission. It drains queued arrivals, calls
`LLMEngine.add_request()`, and lets the engine worker call `engine.step()`.

`LLMEngine.step()` asks the scheduler for work. Before a new prompt starts, the
scheduler reserves enough capacity credits for its prompt and maximum
completion. It allocates physical pages only for the prompt and later block
boundaries, so unwritten future tokens do not evict reusable prefixes. If the
capacity reservation cannot be made, the request waits while active requests
continue; a bounded first-fit scan still admits smaller requests behind it. An
admitted request is never later evicted with partial state. The scheduler then
chooses a prefill chunk and returns a host-only `SchedulePlan`.
The cleaned scheduler chooses either prefill work or decode work for a step; it
does not carry a dormant mixed prefill/decode mode.

`ModelRunner` materializes that plan. It pads rows to the selected bucket,
reuses eligible device metadata, and produces the `DeviceBatch` consumed by
compiled execution.

Packed prefill arrays use fixed bucket shapes:

```text
tokens          [1, token_bucket]
positions       [1, token_bucket]
token_row_ids   [1, token_bucket]
query_start_loc [rows + 1]
block_tables    [rows, block_bucket]
seq_lens        [rows]
```

Only the first `num_prefill_tokens` entries are live. The bucket padding is part
of the static JAX contract.

`ModelRunner` installs any cached hybrid state and selects one `ExecutionPlan`
from the route registry. That same route specification drives preparation,
executor dispatch, startup warmup, and the route label. The executor then runs
the Qwen layer loop:

```text
embed
for each layer:
  projection     -> packed helpers in projection.py
  full attention -> attention.py, Triton packed prefill
  or GDN          -> gdn.py, Triton/FLA padded prefill
  MLP
LM head          -> lm_head.py, native `[V, H]` Triton top-1 when greedy
```

After execution, the scheduler records computed prefix blocks and matching GDN
hybrid state. The runner returns a `RunResult`; `LLMEngine.commit()` then
advances logical state once and returns a `StepResult`. The invariant is that
logical tokens, full-attention KV blocks, and GDN hybrid state all advance by
the same committed prefix.

## One Decode Step

On the next `engine.step()`, the scheduler picks running requests and returns a
decode plan. The runner materializes these arrays:

```text
tokens       [batch_bucket, 1]
positions    [batch_bucket, 1]
block_tables [batch_bucket, block_bucket]
seq_lens     [batch_bucket]
seq_ids      [batch_bucket]
```

Inactive padded rows carry sentinel sequence ids and zero lengths. Active rows
reuse resident decode metadata where possible.

`ModelRunner` applies device token carry so the next decode token can stay on
device instead of synchronizing through Python. Full-attention decode calls the
FlashInfer paged route; GDN decode uses the accepted packed BF16 reference
route; greedy LM-head selection uses the Triton top-1 wrapper.

The engine commits emitted tokens into each request's `OutputBuffer` and
publishes the resulting `TokenEvent` and `FinishedRequest` values through a
`StepResult`. Streaming explicitly materializes only event-bearing buffers;
non-streaming output stays on device until the request finishes. Either the
checkpoint EOS or tokenizer EOS ends generation, and the finish reason is
carried to the service response.

The HTTP stream reads those tokens through a one-slot notification channel.
If the client is slow, the channel retains only the latest output watermark and
the next `tokens` event carries the whole unseen range. Closing the stream asks
the engine worker to cancel the request and release its cache state.

## Prefix-Cache Hit

When a later prompt shares complete prompt blocks with an earlier request, the
scheduler hashes full prompt blocks and asks `BlockManager` for reusable pages.
It always leaves at least one prompt token to execute because prefix entries do
not store the logits following a completely cached prompt.

A hit is valid only when both pieces of state match:

```text
full-attention KV blocks for the prefix
GDN conv/recurrent hybrid state at the same prefix boundary
```

The scheduler skips the cached prefix, passes its opaque state handle to the
runner, and schedules only the remaining suffix. The runner checks the entry,
snapshot, and request token counts before installing the state. Reusing a KV
block invalidates the complete entry and releases the snapshot.
