# Scheduler

The scheduler converts queued `Sequence` objects into fixed-shape `ScheduledBatch` objects and maintains Python-side block allocation metadata.

## Prefill chunks

Prompt prefill can be chunked by token budget or configured prefill buckets.

Prefill chunk fields:

- `tokens`: prompt tokens for this chunk,
- `positions`: absolute prompt positions,
- `query_lens`: number of active tokens per row,
- `query_start_loc`: flattened query offsets,
- `seq_lens`: logical sequence length after this chunk,
- `prefill_is_final`: whether this chunk finishes the prompt.

Prefill invariant:

```text
non-final prefill chunks update cache only and emit no generated token
```

Final prefill chunks may emit the first generated token from target logits.

## Decode buckets

Decode batches usually have one active token per sequence. With speculative decoding, the scheduler reserves lookahead capacity:

```text
reserved slots = 1 current token + K draft tokens
```

Decode bucket invariant:

```text
block capacity must cover the largest position the executor may write, not only the token Python will append first
```

For K=2, a verifier can write current token plus two draft-token positions.

## Zero-length rows

Fixed-shape batches may contain inactive rows or rows with zero query length. These rows are useful for stable compilation shapes but must be masked throughout metadata and KV writes.

Zero-row invariant:

```text
zero-length rows must not write KV, advance hybrid state, emit tokens, or affect active rows
```

If a batch uses padded rows, `query_start_loc` and valid masks are the source of truth for which columns are active.

## Preemption

When KV capacity is insufficient, the scheduler can preempt running sequences by deallocating their blocks and moving them back to waiting.

Preemption invariant:

```text
a preempted sequence must not retain device cache state as if it were still resident
```

After preemption, prompt/cache reconstruction must proceed through normal scheduling rather than reusing stale physical slots.

## Postprocess

Postprocess appends emitted tokens and updates block metadata.

For a single emitted token:

```text
append token; it has not yet been processed as an input token
```

For multiple emitted tokens:

```text
append each token
for every emitted token except the last, commit_processed_token
```

This matches speculative semantics: accepted drafts have been processed by the verifier, while the final bonus token has not.

## Scheduler/executor boundary

Scheduler guarantees before executor execution:

- every active row has valid tokens and positions,
- block tables include every slot the executor may write,
- query lengths and token counts agree,
- decode rows are fully prefetched through prompt tokens.

Executor guarantees after execution:

- cache storage reflects the model step it ran,
- hybrid state reflects the same accepted/processed prefix,
- logits/hidden outputs correspond to the scheduled tokens.
