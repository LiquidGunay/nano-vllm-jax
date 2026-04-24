"""Block manager for KV cache allocation with prefix caching."""

from collections import deque
import xxhash
import numpy as np
from typing import Dict, List, Set, Deque

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
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """Free a block."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate blocks for sequence."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate blocks for a sequence.
        
        Implements prefix caching:
        - Computes hash for each full block
        - Reuses existing blocks if hash matches
        - Allocates new blocks for cache misses
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            
            # Compute hash only for full blocks
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
                
                # Check if we have a cache hit
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    cache_miss = True
                else:
                    cache_miss = False
            else:
                cache_miss = True
            
            if cache_miss:
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
                    # Block is free - allocate it
                    block = self._allocate_block(block_id)
            
            # Update block hash if computed
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Free blocks for a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a token to sequence."""
        # Need new block if current block is full
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Append a token to sequence, allocating new block if needed.
        
        Updates hash for completed blocks.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # Just crossed block boundary - allocate new block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        elif len(seq) % self.block_size == 0:
            # Completed a block - update its hash
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
