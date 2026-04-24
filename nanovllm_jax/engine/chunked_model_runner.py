"""Chunked model runner for JAX vLLM.

Implements chunked prefill with fixed-shape JIT compilation.
"""

import time
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import ModelParams
from nanovllm_jax.kv_cache import (
    KVCacheState,
    init_kv_cache,
    init_linear_attention_states,
)
from nanovllm_jax.layers import rms_norm, get_activation
from nanovllm_jax.chunked_prefill import (
    chunked_full_attention_block,
    pad_to_chunk_size,
)
from nanovllm_jax.model import gated_deltanet_block


class ChunkedModelRunner:
    """Model runner with chunked prefill for fixed-shape compilation."""
    
    def __init__(
        self,
        config: Qwen3_5Config,
        params: ModelParams,
        chunk_size: int = 256,
    ):
        self.config = config
        self.params = params
        self.chunk_size = chunk_size
        
        max_seqs = getattr(config, 'max_num_seqs', 16)
        self.max_blocks_per_seq = config.num_kvcache_blocks // max_seqs
        
        self.kv_state = init_kv_cache(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
            dtype=config.get_dtype(),
        )
        self.kv_state = init_linear_attention_states(
            self.kv_state, config, batch_size=1, dtype=config.get_dtype()
        )
        
        self._compiled_fns = {}
        self._warmup_compiled = False
        
        self.mtp_enabled = params.mtp_params is not None
        if self.mtp_enabled:
            self._mtp_compiled = {}
    
    def warmup_compilation(self, max_batch: int = 1):
        """Pre-compile for fixed chunk size."""
        if self._warmup_compiled:
            return
        
        print(f"Warming up compilation for chunk_size={self.chunk_size}...")
        
        print(f"  Prefill: batch=1, chunk_size={self.chunk_size}...", end=" ", flush=True)
        t0 = time.time()
        self._compile_prefill_fn(batch_size=1)
        print(f"{time.time()-t0:.1f}s")
        
        print(f"  Decode: batch=1...", end=" ", flush=True)
        t0 = time.time()
        self._compile_decode_fn(batch_size=1)
        print(f"{time.time()-t0:.1f}s")
        
        self._warmup_compiled = True
        print("  ✓ Warmup complete")
    
    def _compile_prefill_fn(self, batch_size: int):
        """Compile prefill function for fixed chunk_size."""
        compile_key = ('prefill', batch_size, self.chunk_size)
        
        if compile_key in self._compiled_fns:
            return
        
        config = self.config
        params = self.params
        chunk_size = self.chunk_size
        
        @jax.jit
        def prefill_chunk(
            embeddings: jnp.ndarray,
            positions: jnp.ndarray,
            actual_lens: jnp.ndarray,
            kv_state: KVCacheState,
        ):
            """Process one chunk of tokens."""
            hidden = embeddings
            batch = embeddings.shape[0]
            
            for i, layer in enumerate(params.layers):
                layer_type = config.layer_types[i]
                
                residual = hidden
                hidden = rms_norm(hidden, layer.get("input_norm", params.norm_weight), eps=config.rms_norm_eps)
                
                if layer_type == "full_attention":
                    hidden, kv_state = chunked_full_attention_block(
                        x=hidden,
                        params=layer,
                        positions=positions,
                        config=config,
                        kv_cache_state=kv_state,
                        actual_lens=actual_lens,
                        chunk_size=chunk_size,
                    )
                else:
                    hidden, kv_state = gated_deltanet_block(
                        x=hidden,
                        positions=positions,
                        params=layer,
                        config=config,
                        layer_idx=i,
                        is_prefill=True,
                        kv_cache_state=kv_state,
                    )
                
                hidden = hidden + residual
                
                residual = hidden
                hidden = rms_norm(hidden, layer.get("post_norm", params.norm_weight), eps=config.rms_norm_eps)
                
                gate_proj = layer.get("gate_proj")
                up_proj = layer.get("up_proj")
                down_proj = layer.get("down_proj")
                
                act_fn = get_activation("silu")
                hidden = jnp.dot(act_fn(jnp.dot(hidden, gate_proj)) * jnp.dot(hidden, up_proj), down_proj)
                hidden = hidden + residual
            
            hidden = rms_norm(hidden, params.norm_weight, eps=config.rms_norm_eps)
            
            if params.lm_head is not None:
                logits = jnp.einsum("bsh,hv->bsv", hidden, params.lm_head)
            else:
                logits = jnp.einsum("bsh,vh->bsv", hidden, params.embed_tokens)
            
            return logits, kv_state
        
        self._compiled_fns[compile_key] = prefill_chunk
    
    def _compile_decode_fn(self, batch_size: int):
        """Compile decode function for single token generation."""
        compile_key = ('decode', batch_size)
        
        if compile_key in self._compiled_fns:
            return
        
        config = self.config
        params = self.params
        
        @jax.jit
        def decode_step(
            token: jnp.ndarray,
            position: jnp.ndarray,
            kv_state: KVCacheState,
        ) -> Tuple[jnp.ndarray, KVCacheState]:
            """Generate one token."""
            hidden = params.embed_tokens[token].astype(jnp.bfloat16)
            hidden = hidden[:, None, :]
            
            for i, layer in enumerate(params.layers):
                layer_type = config.layer_types[i]
                
                residual = hidden
                hidden = rms_norm(hidden, layer.get("input_norm", params.norm_weight), eps=config.rms_norm_eps)
                
                if layer_type == "full_attention":
                    from nanovllm_jax.model import full_attention_block
                    hidden, kv_state = full_attention_block(
                        x=hidden,
                        params=layer,
                        positions=position[:, None],
                        mask=None,
                        config=config,
                        kv_cache_state=kv_state,
                        is_prefill=False,
                    )
                else:
                    hidden, kv_state = gated_deltanet_block(
                        x=hidden,
                        positions=position[:, None],
                        params=layer,
                        config=config,
                        layer_idx=i,
                        is_prefill=False,
                        kv_cache_state=kv_state,
                    )
                
                hidden = hidden + residual
                
                residual = hidden
                hidden = rms_norm(hidden, layer.get("post_norm", params.norm_weight), eps=config.rms_norm_eps)
                gate_proj = layer.get("gate_proj")
                up_proj = layer.get("up_proj")
                down_proj = layer.get("down_proj")
                act_fn = get_activation("silu")
                hidden = jnp.dot(act_fn(jnp.dot(hidden.squeeze(1), gate_proj)) * jnp.dot(hidden.squeeze(1), up_proj), down_proj)
                hidden = hidden[:, None, :] + residual
            
            hidden = rms_norm(hidden.squeeze(1), params.norm_weight, eps=config.rms_norm_eps)
            
            if params.lm_head is not None:
                logits = jnp.einsum("bh,hv->bv", hidden, params.lm_head)
            else:
                logits = jnp.einsum("bh,vh->bv", hidden, params.embed_tokens)
            
            return logits, kv_state
        
        self._compiled_fns[compile_key] = decode_step
    
    def prefill_sequence(
        self,
        tokens: jnp.ndarray,
        kv_state: KVCacheState,
    ) -> Tuple[jnp.ndarray, KVCacheState]:
        """Prefill a sequence using chunked processing."""
        batch, seq_len = tokens.shape
        chunk_size = self.chunk_size
        
        compile_key = ('prefill', batch, chunk_size)
        if compile_key not in self._compiled_fns:
            self._compile_prefill_fn(batch)
        prefill_fn = self._compiled_fns[compile_key]
        
        # Set up block_table for contiguous block assignment
        block_table_batch = jnp.broadcast_to(
            jnp.arange(self.max_blocks_per_seq, dtype=jnp.int32)[None, :],
            (batch, self.max_blocks_per_seq)
        )
        
        # Initialize kv_lens to 0
        kv_lens = jnp.zeros(batch, dtype=jnp.int32)
        
        # Update kv_state with batch-specific info
        kv_state = replace(
            kv_state,
            block_table=block_table_batch,
            kv_lens=kv_lens,
        )
        
        last_logits = None
        
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            actual_chunk_len = chunk_end - chunk_start
            
            chunk_tokens = tokens[:, chunk_start:chunk_end]
            
            if actual_chunk_len < chunk_size:
                chunk_tokens, _ = pad_to_chunk_size(
                    chunk_tokens,
                    chunk_size,
                    pad_value=0,
                )
            
            positions = jnp.arange(chunk_start, chunk_start + chunk_size)[None, :]
            positions = jnp.broadcast_to(positions, (batch, chunk_size))
            
            embeddings = self.params.embed_tokens[chunk_tokens].astype(jnp.bfloat16)
            
            # Compute actual_lens for this chunk
            actual_lens = jnp.full((batch,), actual_chunk_len, dtype=jnp.int32)
            
            # Compute slot_mapping for this chunk (positions -> physical slots)
            # Using contiguous block assignment: position -> position
            slot_mapping_chunk = positions
            
            # Update kv_state with slot_mapping for this chunk
            kv_state = replace(
                kv_state,
                slot_mapping=slot_mapping_chunk,
            )
            
            logits, kv_state = prefill_fn(embeddings, positions, actual_lens, kv_state)
            
            if chunk_end == seq_len:
                last_logits = logits[:, actual_chunk_len - 1, :]
        
        return last_logits, kv_state
    
    def decode_step(
        self,
        token: jnp.ndarray,
        position: jnp.ndarray,
        kv_state: KVCacheState,
    ) -> Tuple[jnp.ndarray, KVCacheState]:
        """Generate one token."""
        batch = token.shape[0]
        
        compile_key = ('decode', batch)
        if compile_key not in self._compiled_fns:
            self._compile_decode_fn(batch)
        decode_fn = self._compiled_fns[compile_key]
        
        # Update kv_lens
        kv_lens = kv_state.kv_lens + 1
        
        # Update block_table for this batch (contiguous blocks)
        block_table_batch = jnp.broadcast_to(
            jnp.arange(self.max_blocks_per_seq, dtype=jnp.int32)[None, :],
            (batch, self.max_blocks_per_seq)
        )
        
        # Update slot_mapping for decode
        slot_mapping = position[:, None]
        
        # Update kv_state
        kv_state = replace(
            kv_state,
            block_table=block_table_batch,
            kv_lens=kv_lens,
            slot_mapping=slot_mapping,
        )
        
        logits, kv_state = decode_fn(token, position, kv_state)
        
        return logits, kv_state
