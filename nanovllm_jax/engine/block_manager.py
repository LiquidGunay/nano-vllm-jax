"""Block manager for KV cache allocation with prefix caching."""

from collections import deque
import xxhash
import numpy as np
from typing import Dict, List, Set, Deque

from nanovllm_jax.kv_cache import BlockTables
from nanovllm_jax.engine.sequence import Sequence


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

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = dict()
        self.free_block_ids: Deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()

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
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
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

    def _record_completed_block_hash(self, seq: Sequence, block_idx: int, *, publish: bool | None = None) -> int | None:
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
                self.hash_to_block_id[block.hash] = block.block_id
            return block.hash
        prefix = self.blocks[seq.block_table[block_idx - 1]].hash if block_idx > 0 else -1
        h = self.compute_hash(token_ids, prefix)
        block.update(h, token_ids)
        if should_publish:
            self.hash_to_block_id[h] = block.block_id
        return h

    def _cached_block_id(self, h: int, token_ids: List[int]) -> int:
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1:
            return -1
        block = self.blocks[block_id]
        if block.hash != h or block.token_ids != token_ids:
            return -1
        return block_id

    def cached_prefix_info(
        self,
        seq: Sequence,
        *,
        cacheable_hashes: Set[int] | None = None,
    ) -> tuple[int, int | None]:
        """Return the longest contiguous reusable full-block prefix.

        When ``cacheable_hashes`` is provided, intermediate KV blocks may be
        reused only up to the longest prefix whose final chained hash is in the
        set. Hybrid models use this to require a matching cached recurrent/GDN
        state for the exact skipped prefix.
        """
        h = -1
        best_blocks = 0
        best_hash: int | None = None
        for block_idx in range(self._num_blocks(seq)):
            token_ids = self._block_tokens(seq, block_idx)
            if len(token_ids) != self.block_size:
                break
            h = self.compute_hash(token_ids, h)
            if self._cached_block_id(h, token_ids) == -1:
                break
            if cacheable_hashes is None or h in cacheable_hashes:
                best_blocks = block_idx + 1
                best_hash = h
        return best_blocks * self.block_size, best_hash

    def can_allocate(
        self,
        seq: Sequence,
        *,
        use_prefix_cache: bool = True,
        cacheable_hashes: Set[int] | None = None,
    ) -> bool:
        """Check if we can allocate blocks for sequence."""
        return len(self.free_block_ids) >= self._num_required_blocks(
            seq,
            use_prefix_cache=use_prefix_cache,
            cacheable_hashes=cacheable_hashes,
        )

    def _num_required_blocks(
        self,
        seq: Sequence,
        *,
        use_prefix_cache: bool = True,
        cacheable_hashes: Set[int] | None = None,
    ) -> int:
        """Count physical free blocks needed for allocation.

        Full-block prefix-cache hits that are already in use do not consume a
        free block; cache misses and request-local partial blocks do.
        """
        if not use_prefix_cache:
            return self._num_blocks(seq)
        cached_tokens, _ = self.cached_prefix_info(seq, cacheable_hashes=cacheable_hashes)
        cached_blocks = cached_tokens // self.block_size
        h = -1
        required = 0
        for block_idx in range(self._num_blocks(seq)):
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

    def allocate(
        self,
        seq: Sequence,
        *,
        use_prefix_cache: bool = True,
        cacheable_hashes: Set[int] | None = None,
    ):
        """Allocate blocks for a sequence.
        
        Implements prefix caching:
        - Computes hash for each full block
        - Reuses existing blocks if hash matches
        - Allocates new blocks for cache misses
        """
        assert not seq.block_table
        seq.block_size = self.block_size
        seq.num_cached_tokens = 0
        seq.cached_prefix_hash = None
        seq.cached_prefix_hybrid_seeded = False
        seq.prefix_cache_enabled = bool(use_prefix_cache)
        cached_tokens, cached_hash = (
            self.cached_prefix_info(seq, cacheable_hashes=cacheable_hashes)
            if use_prefix_cache
            else (0, None)
        )
        cached_blocks = cached_tokens // self.block_size
        h = -1
        
        for i in range(self._num_blocks(seq)):
            token_ids = self._block_tokens(seq, i)
            block_id = -1

            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
                if i < cached_blocks:
                    block_id = self._cached_block_id(h, token_ids)
                    if block_id == -1:
                        raise AssertionError("cached prefix disappeared during allocation")

            if block_id == -1:
                # Allocate new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache hit - reuse existing block
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # Block already in use - increment ref count
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._reuse_cached_block(block_id)
            
            # Only full blocks are content-addressed. Newly allocated blocks are
            # not published to the prefix-cache map until execution has actually
            # materialized their KV rows.
            if len(token_ids) == self.block_size and h != -1:
                block.update(h, token_ids)

            seq.block_table.append(block_id)
        seq.cached_prefix_hash = cached_hash if seq.num_cached_tokens > 0 else None

    def deallocate(self, seq: Sequence):
        """Free blocks for a sequence."""
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
        ``num_slots`` counts that scheduled token plus any speculative draft
        tokens that may be processed by the same target-model call.
        """
        target_tokens = len(seq) + max(0, int(num_slots) - 1)
        required_blocks = (target_tokens + self.block_size - 1) // self.block_size
        return len(self.free_block_ids) >= max(0, required_blocks - len(seq.block_table))

    def stats(self) -> dict[str, int]:
        """Return allocation counters for scheduler diagnostics."""
        return {
            "total_blocks": len(self.blocks),
            "free_blocks": len(self.free_block_ids),
            "used_blocks": len(self.used_block_ids),
        }

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
        Speculative decoding can process one or more additional draft tokens
        before Python postprocess appends them to ``seq``. Their physical cache
        rows must therefore be allocated before the executor writes them.
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
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        if len(seq) % self.block_size == 0:
            # Completed a block - update its hash
            block_idx = len(seq) // self.block_size - 1
            self._record_completed_block_hash(seq, block_idx)

        target_tokens = len(seq) + max(0, int(num_slots) - 1)
        required_blocks = (target_tokens + self.block_size - 1) // self.block_size
        while len(block_table) < required_blocks:
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

    def trim_append_slots(self, seq: Sequence, num_slots: int):
        """Release unused decode lookahead blocks after admission changes."""
        block_table = seq.block_table
        target_tokens = len(seq) + max(0, int(num_slots) - 1)
        required_blocks = (target_tokens + self.block_size - 1) // self.block_size
        while len(block_table) > required_blocks:
            block_id = block_table.pop()
            block = self.blocks[block_id]
            if block.hash != -1 or block.token_ids:
                raise AssertionError("cannot trim a materialized KV block")
            block.ref_count -= 1
            self._deallocate_block(block_id)

    def commit_processed_token(self, seq: Sequence):
        """Record metadata for an already-processed appended token.

        Speculative decoding can commit a draft token in the same target pass
        that processed it. The block is already allocated and written in the
        device cache, so this only refreshes the Python prefix-cache metadata
        when that token completes a block.
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

    def publish_computed_prefix(self, seq: Sequence, upto_tokens: int) -> int | None:
        """Publish an already-recorded full-block prefix to the cache map."""
        return self.record_computed_prefix(seq, upto_tokens, publish=True)
