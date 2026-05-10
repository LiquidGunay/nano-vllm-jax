# KV Cache

This document records the canonical KV-cache invariants for prefill, decode, and MTP lookahead.

## Objects

Full-attention KV state has three logical parts:

- device KV arrays, indexed by physical block and offset,
- per-sequence block tables, mapping logical blocks to physical blocks,
- slot mappings, mapping scheduled token positions to physical write slots.

Linear-attention layers do not use the paged full-attention cache. They carry recurrent/convolution state in `HybridLayerState` and must advance with the same accepted prefix as the full-attention cache.

## Block table

A sequence is divided into fixed-size logical blocks. The block table stores the physical block id for each logical block.

```text
logical_position -> logical_block = position // block_size
logical_position -> block_offset  = position % block_size
block_table[row, logical_block] -> physical_block
physical_slot = physical_block * block_size + block_offset
```

Scheduler invariant:

```text
All logical blocks that may be written by the next scheduled step, including MTP lookahead, are allocated before the executor runs.
```

## Slot mapping

`slot_mapping` is the executor-facing list of physical slots to write for scheduled tokens. It is not equivalent to logical position when paging is active.

Valid slot mappings must satisfy:

- active rows map each scheduled token to an allocated physical slot,
- inactive padded rows do not expose writable slots as real tokens,
- prefill chunks map every prompt token in the chunk,
- decode maps the next target token position,
- MTP lookahead maps the additional draft/bonus positions that may be committed.

## Valid masks

Masks are part of correctness, not only performance.

- Active-row masks decide which rows participate in logits, sampling, and commit updates.
- Attention masks decide which historical positions are visible to each query position.
- Slot/write masks prevent inactive padded rows and rejected lookahead tokens from becoming logically reachable.

Inactive rows in a bucket may have zero scheduled tokens. They must preserve previous sequence, KV, and hybrid state.

## Block boundary behavior

Decode and MTP can cross a block boundary. When the current target token or lookahead token lands at offset `0` of a new logical block, the scheduler must allocate that block before execution.

For K=1 MTP, a row may need capacity for:

- the target token at `current_length`,
- the draft/bonus token at `current_length + 1` when admitted.

A reject commits only the target token. An accept commits the target token plus the bonus token. Rejected lookahead writes must not become reachable through logical length, masks, or future block-table interpretation.

## Commit invariant

After scheduler postprocess:

```text
visible tokens == tokens committed by the target-model semantics
logical length == old length + committed token count
reachable KV/hybrid state == state for that committed prefix
```

This invariant is required even when an executor call produced extra padded rows or speculative lookahead writes.
