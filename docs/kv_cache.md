# KV Cache and Hybrid State

This document describes cache invariants rather than historical implementation status.

## Block table

Each sequence has a block table:

```text
logical block index -> physical block id
```

If `block_size = 16`, logical token positions map to logical blocks as:

```text
positions 0..15   -> logical block 0
positions 16..31  -> logical block 1
positions 32..47  -> logical block 2
```

Block-table invariant:

```text
a token position may be scheduled only if its logical block has a physical block id
```

## Slot mapping

`slot_mapping` maps each scheduled token position to a flat physical KV slot:

```text
physical_slot = physical_block_id * block_size + (position % block_size)
```

Example with `block_size = 16` and block table `[7, 3]`:

```text
position 0   -> block 7, offset 0  -> slot 112
position 15  -> block 7, offset 15 -> slot 127
position 16  -> block 3, offset 0  -> slot 48
position 17  -> block 3, offset 1  -> slot 49
```

Slot invariant:

```text
sequential target decode and any fused verifier must write identical logical positions to identical physical slots
```

## Valid masks

Scheduled batches can be padded or contain fixed-shape rows. KV writes must use a valid-token mask derived from query lengths so inactive or padded positions do not overwrite real cache entries.

Valid-mask invariant:

```text
only positions with local index < query_len(row) may write KV
```

For rows with query length zero, all token columns are invalid and must be masked out.

## Block boundary examples

Normal decode at a boundary:

```text
before decode: len(seq) = 16, block table has block 0 full
scheduled token position = 15 as current last_token if already appended
next emitted token will make len(seq) = 17
block manager must allocate block 1 before position 16 can be written
```

Speculative K=2 near a boundary with `block_size = 16`:

```text
current last_token position = 14
verifier tokens = [position 14, position 15, position 16]
required blocks include positions 15 and 16
position 16 requires the next physical block before verifier execution
```

Postprocess after full K=2 accept emits:

```text
[draft_1, draft_2, bonus]
```

The verifier has processed through `draft_2`; the bonus is emitted but not yet in KV. `commit_processed_token` is needed for accepted drafts that complete a block.

## Rejected speculative writes

Rejected verifier writes may exist physically, but they must remain logically unreachable.

Safety invariant:

```text
a rejected draft slot must be restored, overwritten, or masked by logical length before any later attention can read it
```

For correctness-first work, prefer discarding speculative state and repairing from canonical target decode over relying on stale future slots being harmless.

## Hybrid state

Full-attention KV cache is not the only state. Linear-attention layers also maintain:

- convolution state,
- recurrent Gated DeltaNet state.

Hybrid-state invariant:

```text
committed hybrid state must correspond to the same accepted token prefix as committed KV state
```

A multi-token decode is equivalent to sequential decode only if linear-attention state updates scan in token order and the selected final state matches the accepted prefix.
