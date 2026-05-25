# Scheduler

The scheduler is the Python-side owner of runnable work, block allocation, preemption, and MTP admission.

Current GPU work treats MTP admission as experimental. The TPU-era K=1 policy below is historical unless a GPU benchmark explicitly revalidates it with generated-token parity and measured decode speedup.

## Responsibilities

The scheduler owns:

- waiting and running queues,
- prompt prefill chunking,
- decode bucket selection,
- active/inactive row masks,
- block allocation and block-table updates,
- preemption when cache capacity is insufficient,
- postprocess after executor output,
- MTP admission and lookahead reservation.

The scheduler does not own model logits or target-model verification correctness.

## Prefill chunks

Prefill may split a prompt into chunks. Each chunk must have:

- logical positions for the prompt slice,
- allocated blocks for every written position,
- valid attention metadata for all visible prefix tokens,
- postprocess that advances prompt progress exactly by the chunk length.

Prefill chunks should not run speculative decode. MTP starts only once a sequence is in decode state.

## Decode buckets

Decode batches are bucketed to static shapes for JIT reuse. A bucket can contain active and inactive rows.

Active rows:

- have a sequence assigned,
- have at least one scheduled decode position,
- may be eligible for MTP lookahead.

Inactive rows:

- are shape padding,
- have zero logical scheduled tokens,
- must not change sequence state,
- must not expose logits or cache writes as real work.

## Zero-length inactive rows

Zero-length inactive rows are required for static bucket shapes. They must remain inert through:

- batch materialization,
- attention metadata,
- executor output,
- scheduler postprocess,
- MTP accept/reject accounting.

A common failure mode is treating padded row output as a real token. Masks must prevent that.

## Preemption

If cache capacity is insufficient, the scheduler may preempt lower-priority running sequences and free their blocks.

Preemption invariant:

```text
A scheduled batch must never reference a physical block that was not allocated for that sequence at scheduling time.
```

After preemption, sequence-visible logical state and block-manager state must agree before the sequence is rescheduled.

## MTP admission and lookahead

For K=1, an admitted row needs capacity for the target token and one lookahead/bonus position. The scheduler must reserve this before execution.

Admission is controlled by serving policy:

- K=1 only,
- scheduler-owned admission,
- acceptance gate,
- measured decode-latency EWMA gate,
- backend-specific execution expectations.

Current limitation:

```text
Latency EWMA is global; per-bucket admission is still pending.
```

Postprocess commits:

- reject: target token only,
- accept: target token plus bonus token,
- inactive: no state change.

Postprocess must update sequence length, block metadata, draft carry, and accounting consistently with the committed prefix.
