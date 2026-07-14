"""Paged block allocation and prefix-cache metadata.

Owns:
    Physical block allocation, reference counts, prefix hashes, and block-table
    snapshots.
Receives:
    Logical sequence state from the scheduler.
Returns:
    Physical cache blocks, capacity reservations, and prefix-cache decisions.
Invariant:
    A published prefix block hash refers to materialized KV for a complete
    logical block.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import xxhash
import numpy as np
from typing import Any, Dict, List, Set, Deque

from nanovllm_jax.sequence import Sequence


@dataclass
class BlockTables:
    """Snapshot of Python-owned allocation state."""

    tables: List[List[int]]
    ref_counts: Any = None
    hashes: Any = None


@dataclass(frozen=True)
class PrefixCacheEntry:
    """One reusable prefix whose KV and optional GDN state share a boundary."""

    prefix_hash: int
    token_count: int
    block_ids: tuple[int, ...]
    hybrid_state_handle: int | None = None


class PrefixCache:
    """Bounded host metadata for materialized prefix state.

    A positive state capacity makes a state handle mandatory for reuse.
    """

    def __init__(self, state_capacity: int = 0):
        self.entries: OrderedDict[int, PrefixCacheEntry] = OrderedDict()
        self.state_capacity = max(0, int(state_capacity))
        self._released_state_handles: Deque[int] = deque()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.state_evictions = 0

    def get(self, prefix_hash: int) -> PrefixCacheEntry | None:
        return self.entries.get(int(prefix_hash))

    def record_access(self, prefix_hash: int | None) -> None:
        if prefix_hash is None:
            self.misses += 1
            return
        self.hits += 1
        self.entries.move_to_end(int(prefix_hash))

    def publish(self, entry: PrefixCacheEntry) -> PrefixCacheEntry:
        current = self.entries.get(entry.prefix_hash)
        if current is not None and current.token_count != entry.token_count:
            raise RuntimeError("one prefix hash has inconsistent token counts")
        if current is not None and entry.hybrid_state_handle is None:
            entry = replace(entry, hybrid_state_handle=current.hybrid_state_handle)
        elif (
            current is not None
            and current.hybrid_state_handle is not None
            and current.hybrid_state_handle != entry.hybrid_state_handle
        ):
            self._released_state_handles.append(current.hybrid_state_handle)
        self.entries[entry.prefix_hash] = entry
        self.entries.move_to_end(entry.prefix_hash)
        return entry

    def make_state_room(self, entries: List[PrefixCacheEntry]) -> None:
        """Make room for new runner-owned state before it is published."""
        new_hashes: set[int] = set()
        for entry in entries:
            current = self.get(entry.prefix_hash)
            if current is None or current.hybrid_state_handle is None:
                new_hashes.add(entry.prefix_hash)
        if len(new_hashes) > self.state_capacity:
            raise RuntimeError("prefix-state batch exceeds its cache capacity")
        while self.num_state_entries + len(new_hashes) > self.state_capacity:
            for prefix_hash, entry in self.entries.items():
                if entry.hybrid_state_handle is not None and prefix_hash not in new_hashes:
                    self._drop_state(prefix_hash)
                    break
            else:
                raise AssertionError("prefix-state capacity accounting is inconsistent")

    def invalidate_block(self, block_id: int) -> None:
        """Remove every prefix whose physical KV chain includes ``block_id``."""
        # Entries are bounded by physical blocks, so a linear scan keeps this
        # lifecycle obvious. Add a reverse index only if reuse profiles show
        # this host-side scan matters at larger cache sizes.
        stale_hashes = [
            prefix_hash
            for prefix_hash, entry in self.entries.items()
            if int(block_id) in entry.block_ids
        ]
        for prefix_hash in stale_hashes:
            entry = self.entries.pop(prefix_hash)
            if entry.hybrid_state_handle is not None:
                self._released_state_handles.append(entry.hybrid_state_handle)
            self.evictions += 1

    def take_released_state_handles(self) -> tuple[int, ...]:
        handles = tuple(self._released_state_handles)
        self._released_state_handles.clear()
        return handles

    @property
    def num_state_entries(self) -> int:
        return sum(
            entry.hybrid_state_handle is not None
            for entry in self.entries.values()
        )

    @property
    def requires_state(self) -> bool:
        return self.state_capacity > 0

    def stats(self) -> dict[str, int]:
        return {
            "prefix_entries": len(self.entries),
            "prefix_state_entries": self.num_state_entries,
            "prefix_state_capacity": self.state_capacity,
            "prefix_hits": self.hits,
            "prefix_misses": self.misses,
            "prefix_evictions": self.evictions,
            "prefix_state_evictions": self.state_evictions,
        }

    def _drop_state(self, prefix_hash: int) -> None:
        entry = self.entries[prefix_hash]
        handle = entry.hybrid_state_handle
        if handle is None:
            return
        self.entries[prefix_hash] = replace(entry, hybrid_state_handle=None)
        self._released_state_handles.append(handle)
        self.state_evictions += 1


class Block:
    """Represents a physical KV cache block."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: List[int] = []

    def update(self, hash_val: int, token_ids: List[int]):
        """Update block with hash and token IDs."""
        self.hash = hash_val
        self.token_ids = token_ids

    def reset(self):
        """Reset block for reuse."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Manages KV cache block allocation with prefix caching.
    
    Features:
    - Reference counting for block sharing
    - Hash-based prefix caching (content-addressable)
    - Simple block_table list per sequence (like nano-vllm)
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        prefix_state_capacity: int = 0,
    ):
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.prefix_cache = PrefixCache(prefix_state_capacity)
        self.free_block_ids: Deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        self._reserved_by_sequence: Dict[int, int] = {}
        self.num_reserved_blocks = 0

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1) -> int:
        """Compute hash for token sequence."""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Allocate a free block."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.prefix_cache.invalidate_block(block_id)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _reuse_cached_block(self, block_id: int) -> Block:
        """Mark a cached free block used without clearing its KV metadata."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.ref_count = 1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        """Free a block."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _num_blocks(self, seq: Sequence) -> int:
        return (seq.num_tokens + self.block_size - 1) // self.block_size

    def _block_tokens(self, seq: Sequence, block_idx: int) -> List[int]:
        start = block_idx * self.block_size
        end = (block_idx + 1) * self.block_size
        return seq.token_ids[start:end]

    def _record_completed_block_hash(
        self,
        seq: Sequence,
        block_idx: int,
        *,
        publish: bool | None = None,
    ) -> int | None:
        if block_idx < 0 or block_idx >= len(seq.block_table):
            return None
        if seq.block_has_unmaterialized_device_tokens(block_idx):
            return None
        token_ids = self._block_tokens(seq, block_idx)
        if len(token_ids) != self.block_size:
            return None
        block = self.blocks[seq.block_table[block_idx]]
        should_publish = seq.prefix_cache_enabled if publish is None else bool(publish)
        if block.hash != -1:
            if should_publish:
                self.prefix_cache.publish(self._prefix_entry(seq, block_idx))
            return block.hash
        prefix = self.blocks[seq.block_table[block_idx - 1]].hash if block_idx > 0 else -1
        h = self.compute_hash(token_ids, prefix)
        block.update(h, token_ids)
        if should_publish:
            self.prefix_cache.publish(self._prefix_entry(seq, block_idx))
        return h

    def _prefix_entry(
        self,
        seq: Sequence,
        block_idx: int,
    ) -> PrefixCacheEntry:
        block = self.blocks[seq.block_table[block_idx]]
        if block.hash == -1:
            raise AssertionError("cannot cache a block before its hash is recorded")
        return PrefixCacheEntry(
            prefix_hash=block.hash,
            token_count=(block_idx + 1) * self.block_size,
            block_ids=tuple(seq.block_table[: block_idx + 1]),
        )

    def _cached_block_id(self, h: int, token_ids: List[int]) -> int:
        entry = self.prefix_cache.get(h)
        if entry is None:
            return -1
        block_id = entry.block_ids[-1]
        block = self.blocks[block_id]
        if block.hash != h or block.token_ids != token_ids:
            return -1
        return block_id

    def cached_prefix_info(
        self,
        seq: Sequence,
    ) -> tuple[int, int | None]:
        """Return the longest contiguous reusable full-block prefix.

        At least one prompt token remains executable because cache entries do
        not store the logits that follow a completely cached prompt.
        """
        h = -1
        best_blocks = 0
        best_hash: int | None = None
        max_blocks = min(
            self._num_blocks(seq),
            max(0, (int(seq.num_prompt_tokens) - 1) // self.block_size),
        )
        for block_idx in range(max_blocks):
            token_ids = self._block_tokens(seq, block_idx)
            if len(token_ids) != self.block_size:
                break
            h = self.compute_hash(token_ids, h)
            if self._cached_block_id(h, token_ids) == -1:
                break
            entry = self.prefix_cache.get(h)
            if entry is None:
                break
            if entry.token_count != (block_idx + 1) * self.block_size:
                raise AssertionError("prefix KV and metadata token counts differ")
            if not self.prefix_cache.requires_state or entry.hybrid_state_handle is not None:
                best_blocks = block_idx + 1
                best_hash = h
        return best_blocks * self.block_size, best_hash

    def can_reserve(
        self,
        seq: Sequence,
        *,
        total_blocks: int | None = None,
        use_prefix_cache: bool = True,
    ) -> bool:
        """Check whether a request's unallocated lifetime can be reserved."""
        available = len(self.free_block_ids) - self.num_reserved_blocks
        return available >= self._num_required_blocks(
            seq,
            total_blocks=total_blocks,
            use_prefix_cache=use_prefix_cache,
        )

    def _num_required_blocks(
        self,
        seq: Sequence,
        *,
        total_blocks: int | None = None,
        use_prefix_cache: bool = True,
    ) -> int:
        """Count physical free blocks needed for allocation.

        Full-block prefix-cache hits that are already in use do not consume a
        free block; cache misses and request-local partial blocks do.
        """
        logical_blocks = self._num_blocks(seq)
        total_blocks = logical_blocks if total_blocks is None else int(total_blocks)
        if total_blocks < logical_blocks:
            raise ValueError("total_blocks cannot be smaller than the prompt")
        if not use_prefix_cache:
            return total_blocks
        cached_tokens, _ = self.cached_prefix_info(seq)
        cached_blocks = cached_tokens // self.block_size
        h = -1
        required = 0
        for block_idx in range(total_blocks):
            token_ids = self._block_tokens(seq, block_idx)
            block_id = -1
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
                if block_idx < cached_blocks:
                    block_id = self._cached_block_id(h, token_ids)
            if block_idx < cached_blocks and block_id in self.used_block_ids:
                continue
            required += 1
        return required

    def reserve(
        self,
        seq: Sequence,
        *,
        total_blocks: int | None = None,
        use_prefix_cache: bool = True,
        initial_lookahead_tokens: int = 0,
    ):
        """Reserve lifetime capacity, then allocate only prompt blocks."""
        if seq.block_table:
            raise ValueError("sequence already has allocated blocks")
        if seq.block_size != self.block_size:
            raise ValueError("sequence and block manager use different block sizes")
        key = id(seq)
        if key in self._reserved_by_sequence:
            raise ValueError("sequence already has a capacity reservation")
        required = self._num_required_blocks(
            seq,
            total_blocks=total_blocks,
            use_prefix_cache=use_prefix_cache,
        )
        if required > len(self.free_block_ids) - self.num_reserved_blocks:
            raise RuntimeError("insufficient free blocks for complete request reservation")
        self._reserved_by_sequence[key] = required
        self.num_reserved_blocks += required
        self.allocate(
            seq,
            use_prefix_cache=use_prefix_cache,
        )
        initial_tokens = seq.num_tokens + max(0, int(initial_lookahead_tokens))
        initial_blocks = (initial_tokens + self.block_size - 1) // self.block_size
        if total_blocks is not None and initial_blocks > int(total_blocks):
            raise ValueError("initial lookahead exceeds reserved request capacity")
        while len(seq.block_table) < initial_blocks:
            seq.block_table.append(self._allocate_reserved_block(seq))

    def _consume_reservation(self, seq: Sequence) -> None:
        key = id(seq)
        remaining = self._reserved_by_sequence.get(key, 0)
        if remaining <= 0:
            raise AssertionError("physical allocation exceeded reserved capacity")
        self._reserved_by_sequence[key] = remaining - 1
        self.num_reserved_blocks -= 1

    def _release_reservation(self, seq: Sequence) -> None:
        remaining = self._reserved_by_sequence.pop(id(seq), 0)
        self.num_reserved_blocks -= remaining

    def _allocate_reserved_block(self, seq: Sequence) -> int:
        if not self.free_block_ids:
            raise AssertionError("reserved capacity has no physical free block")
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        self._consume_reservation(seq)
        return block_id

    def allocate(
        self,
        seq: Sequence,
        *,
        use_prefix_cache: bool = True,
    ) -> None:
        """Allocate prompt pages, reusing a complete cached prefix when possible."""
        assert not seq.block_table
        if seq.block_size != self.block_size:
            raise ValueError("sequence and block manager use different block sizes")
        logical_blocks = self._num_blocks(seq)
        seq.num_cached_tokens = 0
        seq.cached_prefix_hash = None
        seq.cached_prefix_hybrid_seeded = False
        seq.prefix_cache_enabled = bool(use_prefix_cache)
        cached_tokens, cached_hash = (
            self.cached_prefix_info(seq)
            if use_prefix_cache
            else (0, None)
        )
        if use_prefix_cache:
            self.prefix_cache.record_access(cached_hash)
        cached_blocks = cached_tokens // self.block_size
        h = -1

        for i in range(logical_blocks):
            token_ids = self._block_tokens(seq, i)
            block_id = -1

            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
                if i < cached_blocks:
                    block_id = self._cached_block_id(h, token_ids)
                    if block_id == -1:
                        raise AssertionError("cached prefix disappeared during allocation")

            if block_id == -1:
                block_id = self._allocate_reserved_block(seq)
                block = self.blocks[block_id]
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._reuse_cached_block(block_id)
                    self._consume_reservation(seq)

            # Only full blocks are content-addressed. Newly allocated blocks are
            # not published to the prefix-cache map until execution has actually
            # materialized their KV rows.
            if len(token_ids) == self.block_size and h != -1:
                block.update(h, token_ids)

            seq.block_table.append(block_id)
        seq.cached_prefix_hash = cached_hash if seq.num_cached_tokens > 0 else None

    def deallocate(self, seq: Sequence):
        """Free physical pages and unused lifetime capacity."""
        self._release_reservation(seq)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.cached_prefix_hash = None
        seq.cached_prefix_hybrid_seeded = False
        seq.prefix_cache_enabled = False
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a token to sequence."""
        return self.can_append_slots(seq, 1)

    def can_append_slots(self, seq: Sequence, num_slots: int) -> bool:
        """Check if blocks exist/can be allocated for a decode lookahead window.

        ``seq`` already contains the scheduled decode token as ``last_token``.
        ``num_slots`` counts the scheduled token plus any greedy burst tokens
        reserved for the same compiled decode call.
        """
        target_tokens = len(seq) + max(0, int(num_slots) - 1)
        required_blocks = (target_tokens + self.block_size - 1) // self.block_size
        new_blocks = max(0, required_blocks - len(seq.block_table))
        return self._reserved_by_sequence.get(id(seq), 0) >= new_blocks

    def stats(self) -> dict[str, int]:
        """Return allocation counters for scheduler diagnostics."""
        return {
            "total_blocks": len(self.blocks),
            "free_blocks": len(self.free_block_ids),
            "used_blocks": len(self.used_block_ids),
            "reserved_blocks": self.num_reserved_blocks,
            "available_blocks": len(self.free_block_ids) - self.num_reserved_blocks,
            **self.prefix_cache.stats(),
        }

    def take_released_prefix_state_handles(self) -> tuple[int, ...]:
        return self.prefix_cache.take_released_state_handles()

    def snapshot(self, seqs: List[Sequence] | None = None) -> BlockTables:
        """Expose Python-side prefix-cache state without touching JAX arrays."""
        return BlockTables(
            tables=[list(seq.block_table) for seq in seqs] if seqs is not None else [],
            ref_counts=[block.ref_count for block in self.blocks],
            hashes=[block.hash for block in self.blocks],
        )

    def may_append(self, seq: Sequence):
        """Append a token to sequence, allocating new block if needed.
        
        Updates hash for completed blocks.
        """
        self.may_append_slots(seq, 1)

    def may_append_slots(self, seq: Sequence, num_slots: int):
        """Reserve block-table entries for a decode lookahead window.

        The scheduled decode token is already present in ``seq.token_ids``.
        Greedy burst decode can process additional tokens before Python
        postprocess appends them to ``seq``. Their physical cache rows must be
        allocated before the executor writes them.
        """
        block_table = seq.block_table
        current_required_blocks = (len(seq) + self.block_size - 1) // self.block_size
        
        if current_required_blocks > len(block_table):
            # Just crossed block boundary - allocate new block
            last_block_idx = len(block_table) - 1
            self._record_completed_block_hash(seq, last_block_idx)
            last_block = self.blocks[block_table[-1]]
            if last_block.hash == -1 and not seq.block_has_unmaterialized_device_tokens(last_block_idx):
                raise AssertionError("completed block hash was not recorded")
            block_table.append(self._allocate_reserved_block(seq))

        if len(seq) % self.block_size == 0:
            # Completed a block - update its hash
            block_idx = len(seq) // self.block_size - 1
            self._record_completed_block_hash(seq, block_idx)

        target_tokens = len(seq) + max(0, int(num_slots) - 1)
        required_blocks = (target_tokens + self.block_size - 1) // self.block_size
        while len(block_table) < required_blocks:
            block_table.append(self._allocate_reserved_block(seq))

    def commit_processed_token(self, seq: Sequence):
        """Record metadata for an already-processed appended token.

        Greedy burst decode can commit a token in the same compiled call that
        processed it. The block is already allocated and written in the device
        cache, so this only refreshes the Python prefix-cache metadata when
        that token completes a block.
        """
        if len(seq) % self.block_size != 0:
            return
        block_idx = len(seq) // self.block_size - 1
        self._record_completed_block_hash(seq, block_idx)

    def record_computed_prefix(self, seq: Sequence, upto_tokens: int, *, publish: bool) -> int | None:
        """Record completed full prompt blocks through ``upto_tokens``.

        Returns the chained hash for ``upto_tokens`` when it is exactly on a
        full-block boundary, otherwise ``None``.
        """
        upto_tokens = max(0, min(int(upto_tokens), int(seq.num_prompt_tokens)))
        full_blocks = upto_tokens // self.block_size
        final_hash: int | None = None
        for block_idx in range(full_blocks):
            block_hash = self._record_completed_block_hash(seq, block_idx, publish=publish)
            if block_idx == full_blocks - 1:
                final_hash = block_hash
        if upto_tokens > 0 and upto_tokens % self.block_size == 0:
            return final_hash
        return None

    def publish_computed_prefix(
        self,
        seq: Sequence,
        upto_tokens: int,
    ) -> PrefixCacheEntry | None:
        """Publish KV metadata at one exact token boundary."""
        prefix_hash = self.record_computed_prefix(seq, upto_tokens, publish=True)
        if prefix_hash is None:
            return None
        entry = self.prefix_cache.get(prefix_hash)
        if entry is None:
            raise AssertionError("published prefix is missing from the cache")
        return entry
