# Serving Walkthrough

This walkthrough follows one request through the promoted serving path. It is
about ownership and shapes, not historical experiments.

## First Packed Prefill

`server.py` normalizes an HTTP request into token ids and sampling parameters,
then submits it to `EngineService`.

`EngineService` owns online admission. It drains queued arrivals, calls
`LLMEngine.add_request()`, and lets the engine worker call `engine.step()`.

`LLMEngine.step()` asks the scheduler for work. For a new prompt, the scheduler
reserves cache blocks through `BlockManager`, chooses a prefill chunk, and
returns a `ScheduledBatch`.

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

`ModelRunner` installs any cached hybrid state, selects the compiled prefill
bucket, and calls `ModelExecutor`. The executor runs the Qwen layer loop:

```text
embed
for each layer:
  full attention -> Triton packed prefill
  or GDN          -> Triton/FLA padded prefill
  MLP
LM head          -> Triton greedy top-1 when greedy
```

After the step, the scheduler records computed prefix blocks and matching GDN
hybrid state. The invariant is that logical tokens, full-attention KV blocks,
and GDN hybrid state all advance by the same committed prefix.

## One Decode Step

On the next `engine.step()`, the scheduler picks running requests and returns a
decode batch:

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

The engine postprocesses emitted tokens, advances logical sequence length, and
publishes token events through the request's service handle. Finished requests
materialize their final token ids and release runner/cache state.

## Prefix-Cache Hit

When a later prompt shares complete prompt blocks with an earlier request, the
scheduler hashes full prompt blocks and asks `BlockManager` for reusable pages.

A hit is valid only when both pieces of state match:

```text
full-attention KV blocks for the prefix
GDN conv/recurrent hybrid state at the same prefix boundary
```

The scheduler skips the cached prefix, schedules only the remaining suffix, and
seeds the runner with the cached hybrid state before prefill. The same invariant
still applies: the request's logical cached-token count, KV block table, and GDN
state all describe the same prefix length.
