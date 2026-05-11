"""Model runner for JAX inference with paged KV cache."""

import time
import os
import jax
import jax.numpy as jnp
import sys
from typing import List, Tuple, Dict, Optional, Any
from functools import partial
from dataclasses import replace

from nanovllm_jax.backends import select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import ModelParams, full_attention_block, gated_deltanet_block, transformer_block, forward as model_forward
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.kv_cache import (
    KVCacheState, 
    KVCacheSpec,
    cap_num_kv_cache_blocks,
    init_kv_cache, 
    init_linear_attention_states,
    compute_slot_mapping,
)
from nanovllm_jax.mtp.mtp_layer import MTPParams, mtp_forward
from nanovllm_jax.mtp.speculative import generate_draft_tokens, verify_draft_tokens, apply_acceptance


class _LegacyModelRunner:
    """Runs JAX model with paged KV cache.
    
    Handles:
    - KV cache state management
    - Prefill vs decode execution
    - Block table to JAX array conversion
    - Logits computation and sampling
    """

    def __init__(self, config: Qwen3_5Config, params: ModelParams, backend: str = "auto"):
        self.config = config
        self.params = params
        self.backend = select_backend(backend)
        self.block_size = config.block_size
        
        # Initialize KV cache state
        max_seqs = getattr(config, 'max_num_seqs', 16)
        kv_spec = KVCacheSpec(
            num_layers=config.num_hidden_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.get_dtype(),
            max_kv_cache_bytes=config.max_kv_cache_bytes,
        )
        effective_num_blocks = cap_num_kv_cache_blocks(kv_spec)
        if effective_num_blocks != config.num_kvcache_blocks:
            print(
                "KV cache capped: "
                f"{config.num_kvcache_blocks} -> {effective_num_blocks} blocks "
                f"({config.max_kv_cache_bytes} byte cap)"
            )
            config.num_kvcache_blocks = effective_num_blocks
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        if self.max_blocks_per_seq is None:
            self.max_blocks_per_seq = max(1, effective_num_blocks // max_seqs)
            config.max_blocks_per_seq = self.max_blocks_per_seq
        self.execution = getattr(config, "jax_execution", "eager")
        
        self.kv_state = init_kv_cache(
            num_blocks=effective_num_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
            num_layers=config.num_hidden_layers,
            dtype=config.get_dtype(),
            max_kv_cache_bytes=config.max_kv_cache_bytes,
        )
        
        # Initialize linear attention states
        self.kv_state = init_linear_attention_states(
            cache=self.kv_state,
            config=config,
            batch_size=1,
            dtype=config.get_dtype(),
        )
        
        # Create sampling function
        self._sample_fn = jax.jit(self._sample_logits)
        
        # MTP support
        self.mtp_enabled = hasattr(params, 'mtp_params') and params.mtp_params is not None
        if self.mtp_enabled:
            print(f"MTP enabled: {config.mtp_num_hidden_layers} layer(s)")
            self._mtp_forward_fn = None
            self._mtp_forward_key = None
        
        # Compilation cache for different shapes
        # Key: (batch_size, seq_len, is_prefill) -> JIT compiled function
        self._compiled_fns: Dict[Tuple[int, int, bool], callable] = {}
        
        # Pre-compile common shapes during initialization (server-style startup)
        self._warmup_compiled = False
    
    def warmup_compilation(self, max_prefill_len: int = 64, max_batch: int = 1):
        """Pre-compile common shapes for fast inference.
        
        This is like server startup - compile once, serve many requests.
        
        Args:
            max_prefill_len: Maximum prefill sequence length to compile
            max_batch: Maximum batch size to compile
        """
        if self._warmup_compiled:
            return  # Already compiled
        
        print("  Compiling prefill shapes...")
        # Compile prefill shapes - minimal set for testing
        prefill_lens = [16, max_prefill_len]
        for seq_len in prefill_lens:
            print(f"    Prefill: batch=1, seq_len={seq_len}...", end=" ", flush=True)
            t0 = time.time()
            self._compile_and_run_once(batch_size=1, seq_len=seq_len, is_prefill=True)
            print(f"{time.time()-t0:.1f}s")
        
        print("  Compiling decode shapes...")
        # Compile decode shapes (batch sizes)
        for batch_size in range(1, max_batch + 1):
            print(f"    Decode: batch={batch_size}, seq_len=1...", end=" ", flush=True)
            sys.stdout.flush()
            t0 = time.time()
            self._compile_and_run_once(batch_size=batch_size, seq_len=1, is_prefill=False)
            print(f"{time.time()-t0:.1f}s")
            sys.stdout.flush()
        
        # Compile MTP if enabled
        if self.mtp_enabled:
            print("  Compiling MTP speculative decoding...")
            print(f"    MTP Decode: batch=1, seq_len=1...", end=" ", flush=True)
            sys.stdout.flush()
            t0 = time.time()
            self._warmup_mtp_compilation()
            print(f"{time.time()-t0:.1f}s")
            sys.stdout.flush()
        
        self._warmup_compiled = True
        print("  ✓ All compilations complete")
    
    def _compile_and_run_once(
        self,
        batch_size: int,
        seq_len: int,
        is_prefill: bool,
    ):
        """Compile and run forward pass once to force JIT compilation using run()."""
        from nanovllm_jax.engine.sequence import Sequence, SamplingParams
        
        # Create dummy sequence
        token_ids = [0] * seq_len
        sampling_params = SamplingParams(temperature=0.0)
        seq = Sequence(token_ids=token_ids, sampling_params=sampling_params, seq_id=999)
        seq.block_table = list(range(10))
        seq.block_size = self.block_size
        
        # Run to force compilation
        _ = self.run([seq], is_prefill=is_prefill)
    
    def _warmup_mtp_compilation(self):
        """Warmup MTP speculative decoding compilation."""
        # Create a dummy sequence for MTP warmup
        from nanovllm_jax.engine.sequence import Sequence, SamplingParams
        
        warmup_ids = [1]  # Single token
        seq = Sequence(token_ids=warmup_ids.copy(), sampling_params=SamplingParams(temperature=0.0), seq_id=999)
        seq.block_table = list(range(5))
        seq.block_size = self.block_size
        
        # Prefill first
        _ = self.run([seq], is_prefill=True)
        
        # Run speculative decode
        _ = self.run_speculative([seq])
    
    def _get_or_compile_forward_fn(
        self,
        batch_size: int,
        seq_len: int,
        is_prefill: bool,
    ):
        """Get or compile forward function for given shape.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            is_prefill: Whether this is prefill
            
        Returns:
            JIT compiled function
        """
        from nanovllm_jax.layers import rms_norm, get_activation
        from nanovllm_jax.model import full_attention_block, gated_deltanet_block
        
        compile_key = (batch_size, seq_len, is_prefill)
        
        if compile_key not in self._compiled_fns:
            # Create a standalone forward function that captures everything via closure
            # This is critical for fast JIT compilation
            config = self.config
            params = self.params
            norm_weight = params.norm_weight
            layers = params.layers
            embed_tokens = params.embed_tokens
            lm_head = params.lm_head
            rms_norm_eps = config.rms_norm_eps
            layer_types = config.layer_types
            
            @jax.jit
            def forward_fn(embeddings, positions, kv_state):
                batch, seq_len = embeddings.shape[:2]
                hidden = embeddings
                
                # Run through layers
                for i, layer in enumerate(layers):
                    layer_type = layer_types[i]
                    
                    # Input norm + residual
                    residual = hidden
                    hidden = rms_norm(hidden, layer.get("input_norm", norm_weight), eps=rms_norm_eps)
                    
                    if layer_type == "full_attention":
                        input_seq_len = hidden.shape[1]
                        mask = jnp.tril(jnp.ones((input_seq_len, input_seq_len))).astype(jnp.float32)
                        
                        hidden, kv_state = full_attention_block(
                            x=hidden,
                            positions=positions,
                            mask=mask,
                            params=layer,
                            config=config,
                            kv_cache_state=kv_state,
                            is_prefill=is_prefill,
                            layer_idx=i,
                            backend=self.backend,
                        )
                    else:
                        if is_prefill:
                            hidden, kv_state = gated_deltanet_block(
                                x=hidden,
                                positions=positions,
                                params=layer,
                                config=config,
                                layer_idx=i,
                                is_prefill=True,
                                kv_cache_state=kv_state,
                            )
                        else:
                            hidden, kv_state = gated_deltanet_block(
                                x=hidden,
                                positions=positions,
                                params=layer,
                                config=config,
                                layer_idx=i,
                                is_prefill=False,
                                kv_cache_state=kv_state,
                            )
                    
                    hidden = residual + hidden
                    
                    # MLP
                    ffn_norm = layer.get("ffn_norm", norm_weight)
                    gate_proj = layer["gate_proj"]
                    up_proj = layer["up_proj"]
                    down_proj = layer["down_proj"]
                    
                    residual = hidden
                    hidden = rms_norm(hidden, ffn_norm, eps=rms_norm_eps)
                    act_fn = get_activation("silu")
                    hidden = jnp.dot(act_fn(jnp.dot(hidden, gate_proj)) * jnp.dot(hidden, up_proj), down_proj)
                    hidden = hidden + residual
                
                # Final norm
                hidden = rms_norm(hidden, norm_weight, eps=rms_norm_eps)
                
                # LM head
                if lm_head is not None:
                    logits = jnp.einsum("bsh,hv->bsv", hidden, lm_head)
                else:
                    logits = jnp.einsum("bsh,vh->bsv", hidden, embed_tokens)
                
                return logits, kv_state
            
            self._compiled_fns[compile_key] = forward_fn
        
        return self._compiled_fns[compile_key]

    def run(
        self, 
        seqs: List[Sequence], 
        is_prefill: bool,
    ) -> List[int | List[int]]:
        """Run model on scheduled sequences.
        
        Args:
            seqs: Sequences to process
            is_prefill: Whether this is prefill or decode
            
        Returns:
            List of generated token IDs (one per sequence)
        """
        batch_size = len(seqs)
        
        # Gather input tokens and positions
        input_ids_list = []
        positions_list = []
        block_tables = []
        kv_lens = []
        
        if is_prefill:
            max_seq_len = max(len(seq) for seq in seqs)
        else:
            # Decode: always seq_len=1
            max_seq_len = 1
        
        for seq in seqs:
            if is_prefill:
                # Prefill: use all tokens
                tokens = seq.token_ids
                positions = seq.get_absolute_positions()
            else:
                # Decode: use only last token
                tokens = [seq.last_token]
                positions = [seq.num_tokens - 1]
            
            # Pad to max length for batching
            pad_len = max_seq_len - len(tokens)
            tokens = tokens + [0] * pad_len
            positions = positions + [0] * pad_len
            
            input_ids_list.append(tokens)
            positions_list.append(positions)
            block_tables.append(seq.block_table + [0] * (self.max_blocks_per_seq - len(seq.block_table)))
            kv_lens.append(seq.num_tokens)
        
        # Convert to JAX arrays
        input_ids = jnp.array(input_ids_list, dtype=jnp.int32)  # [batch, seq_len]
        positions = jnp.array(positions_list, dtype=jnp.int32)  # [batch, seq_len]
        block_table = jnp.array(block_tables, dtype=jnp.int32)  # [batch, max_blocks]
        kv_lens = jnp.array(kv_lens, dtype=jnp.int32)  # [batch]
        
        # Compute slot_mapping from block_table (real paged attention)
        slot_mapping = compute_slot_mapping(
            positions=positions,
            block_table=block_table,
            block_size=self.block_size,
            is_prefill=is_prefill,
        )
        
        # Update KV cache state
        kv_state = replace(
            self.kv_state,
            block_table=block_table,
            kv_lens=kv_lens,
            slot_mapping=slot_mapping,
        )
        
        # Run forward pass (use pre-compiled function)
        batch_size, seq_len = input_ids.shape
        
        # Get or compile function for this shape
        forward_fn = self._get_or_compile_forward_fn(batch_size, seq_len, is_prefill)
        
        # Compute embeddings BEFORE calling JIT function (critical for compilation speed)
        embeddings = self.params.embed_tokens[input_ids].astype(jnp.bfloat16)
        
        logits, updated_kv_state = forward_fn(embeddings, positions, kv_state)
        
        # Update KV cache state - preserve all fields from updated_kv_state
        self.kv_state = updated_kv_state
        
        # Get last token logits for each sequence
        if is_prefill:
            # Use last position of each sequence
            last_logits = logits[jnp.arange(batch_size), kv_lens - 1]
        else:
            # Only one token in decode
            last_logits = logits[:, 0]
        
        # Sample tokens
        temperatures = jnp.array([seq.temperature for seq in seqs])
        token_ids = self._sample_fn(last_logits, temperatures)
        
        return token_ids

    def _forward_with_params(
        self,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        block_table: jnp.ndarray,
        kv_lens: jnp.ndarray,
        slot_mapping: jnp.ndarray,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        is_prefill: bool,
        conv_state: jnp.ndarray = None,
        recurrent_state: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Forward pass using self.params and self.config (for JIT closure).
        
        Args:
            k_cache: Key cache
            v_cache: Value cache
            block_table: Block table
            kv_lens: Sequence lengths
            slot_mapping: Slot mapping
            input_ids: Token IDs [batch, seq_len]
            positions: Absolute positions [batch, seq_len]
            is_prefill: Whether this is prefill
            conv_state: Conv state for linear attention [batch, num_linear_layers, conv_dim, kernel_size]
            recurrent_state: Recurrent state for linear attention [batch, num_linear_layers, num_heads, k_dim, v_dim]
            
        Returns:
            Tuple of (logits, k_cache, v_cache, conv_state, recurrent_state)
        """
        
        # Reconstruct KVCacheState for compatibility
        kv_state = KVCacheState(
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            kv_lens=kv_lens,
            slot_mapping=slot_mapping,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
        )
        
        batch, seq_len = input_ids.shape
        
        # Get embeddings - cast to bfloat16 to match HF computation
        embeddings = self.params.embed_tokens[input_ids].astype(jnp.bfloat16)  # [batch, seq_len, hidden]
        
        # Run through transformer layers
        hidden = embeddings
        for i, layer in enumerate(self.params.layers):
            layer_type = self.config.layer_types[i]
            
            # Apply input_layernorm and residual (matches transformer_block in model.py)
            from nanovllm_jax.layers import rms_norm
            residual = hidden
            input_norm = layer.get("input_norm", self.params.norm_weight)
            hidden = rms_norm(hidden, input_norm, eps=self.config.rms_norm_eps)
            
            if layer_type == "full_attention":
                # Create causal mask based on mode
                input_seq_len = hidden.shape[1]
                
                # Prefill: standard causal mask [seq_len, seq_len]
                # Decode: mask will be expanded in full_attention_block based on kv_lens
                # For now, use input_seq_len for the mask (will be overridden in decode)
                mask = jnp.tril(jnp.ones((input_seq_len, input_seq_len))).astype(jnp.float32)
                
                # Use paged attention
                hidden, kv_state = full_attention_block(
                    x=hidden,
                    positions=positions,
                    mask=mask,
                    params=layer,
                    config=self.config,
                    kv_cache_state=kv_state,
                    is_prefill=is_prefill,
                    layer_idx=i,
                    backend=self.backend,
                )
            else:
                # Linear attention with decode mode support
                if is_prefill:
                    # Prefill: pass kv_cache_state to initialize and save final state
                    hidden, kv_state = gated_deltanet_block(
                        x=hidden,
                        positions=positions,
                        params=layer,
                        config=self.config,
                        layer_idx=i,
                        is_prefill=True,
                        kv_cache_state=kv_state,
                    )
                else:
                    # Decode mode: pass cache state
                    hidden, kv_state = gated_deltanet_block(
                        x=hidden,
                        positions=positions,
                        params=layer,
                        config=self.config,
                        layer_idx=i,
                        is_prefill=False,
                        kv_cache_state=kv_state,
                    )
            
            # Add residual
            hidden = residual + hidden
            
            # Apply MLP
            ffn_norm = layer.get("ffn_norm", self.params.norm_weight)
            gate_proj = layer["gate_proj"]
            up_proj = layer["up_proj"]
            down_proj = layer["down_proj"]
            
            from nanovllm_jax.layers import get_activation
            residual = hidden
            hidden = rms_norm(hidden, ffn_norm, eps=self.config.rms_norm_eps)
            act_fn = get_activation("silu")
            hidden = jnp.dot(act_fn(jnp.dot(hidden, gate_proj)) * jnp.dot(hidden, up_proj), down_proj)
            hidden = hidden + residual
        
        # Final norm
        hidden = rms_norm(hidden, self.params.norm_weight, eps=self.config.rms_norm_eps)
        
        # LM head
        if self.params.lm_head is not None:
            logits = jnp.einsum("bsh,hv->bsv", hidden, self.params.lm_head)
        else:
            # Tie embeddings
            logits = jnp.einsum("bsh,vh->bsv", hidden, self.params.embed_tokens)
        
        return (logits, kv_state.k_cache, kv_state.v_cache, 
                kv_state.conv_state, kv_state.recurrent_state)
    
    def _forward_impl(
        self,
        params: ModelParams,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        kv_state: KVCacheState,
        is_prefill: bool,
        config: Qwen3_5Config,
    ) -> jnp.ndarray:
        """Forward pass implementation (non-JIT, for reference).
        
        Args:
            params: Model parameters
            input_ids: Token IDs [batch, seq_len]
            positions: Absolute positions [batch, seq_len]
            kv_state: KV cache state
            is_prefill: Whether this is prefill
            config: Model configuration
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        
        # Get embeddings
        embeddings = params.embed_tokens[input_ids]  # [batch, seq_len, hidden]
        
        # Run through transformer layers
        hidden = embeddings
        for i, layer in enumerate(params.layers):
            layer_type = config.layer_types[i]
            
            if layer_type == "full_attention":
                # Use paged attention
                hidden = full_attention_block(
                    hidden_states=hidden,
                    positions=positions,
                    kv_state=kv_state,
                    params=layer,
                    config=config,
                    is_prefill=is_prefill,
                )
            else:
                # Linear attention (doesn't use paged cache yet)
                # TODO: Implement paged linear attention
                hidden = gated_deltanet_block(
                    x=hidden,
                    positions=positions,
                    params=layer,
                    config=config,
                )
            
            # Apply MLP - extract from transformer_block
            ffn_norm = layer.get("ffn_norm", params.norm_weight)
            gate_proj = layer["gate_proj"]
            up_proj = layer["up_proj"]
            down_proj = layer["down_proj"]
            
            from nanovllm_jax.layers import rms_norm, get_activation
            residual = hidden
            hidden = rms_norm(hidden, ffn_norm, eps=config.rms_norm_eps)
            act_fn = get_activation("silu")
            hidden = jnp.dot(act_fn(jnp.dot(hidden, gate_proj)) * jnp.dot(hidden, up_proj), down_proj)
            hidden = hidden + residual
        
        # Final norm
        hidden = rms_norm(hidden, params.norm_weight, eps=config.rms_norm_eps)
        
        # LM head
        if params.lm_head is not None:
            logits = jnp.einsum("bsh,hv->bsv", hidden, params.lm_head)
        else:
            # Tie embeddings
            logits = jnp.einsum("bsh,vh->bsv", hidden, params.embed_tokens)
        
        return logits
    
    def forward(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        kv_state: KVCacheState,
        is_prefill: bool,
    ) -> jnp.ndarray:
        """Forward pass with paged KV cache (calls JIT-compiled version).
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            positions: Absolute positions [batch, seq_len]
            kv_state: KV cache state
            is_prefill: Whether this is prefill
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        return self._forward_fn(
            self.params,
            input_ids,
            positions,
            kv_state,
            is_prefill,
            self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _sample_logits(
        self,
        logits: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample tokens from logits.
        
        Args:
            logits: Logits [batch, vocab_size]
            temperatures: Temperature for each sequence [batch]
            
        Returns:
            Sampled token IDs [batch]
        """
        import jax.lax as lax
        
        def sample_single(logit, temp):
            """Sample a single logit with given temperature."""
            def greedy(_):
                return jnp.argmax(logit)
            
            def sample(_):
                scaled = logit / temp
                return jax.random.categorical(jax.random.PRNGKey(0), scaled)
            
            return lax.cond(temp == 0.0, greedy, sample, None)
        
        # Vectorize over batch
        token_ids = jax.vmap(sample_single)(logits, temperatures)
        return token_ids

    def call(self, method: str, *args):
        """Compatibility method for multiprocessing (like nano-vllm)."""
        if method == "run":
            return self.run(*args)
        elif method == "exit":
            pass
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run_speculative(
        self,
        seqs: List[Sequence],
    ) -> List[int | List[int]]:
        """Run speculative decoding with MTP.
        
        Speculative decoding workflow (K=1):
        1. Run main model on last confirmed token (position t)
           - Returns hidden_state[t] and logits[t] (predicting t+1)
        2. Sample main_token[t+1] from main_logits[t]
        3. Run MTP with hidden_state[t] + main_token[t+1] embedding
           - MTP predicts t+2
        4. Verify MTP's t+2 prediction against main model
           - Run main model on main_token[t+1] to get logits[t+1] (predicting t+2)
           - Compare and accept/reject
        
        For K=1, we simplify:
        - MTP predicts t+1 from hidden_state[t-1] + confirmed_token[t-1]
        - Compare with main model's prediction for t+1
        - This is equivalent to standard speculative decoding but with K=1
        
        Args:
            seqs: Sequences to process (decode mode only)
            
        Returns:
            List of generated token IDs (one per sequence)
        """
        if not self.mtp_enabled:
            # Fallback to normal decode
            return self.run(seqs, is_prefill=False)
        
        # For now, disable MTP for batch > 1 (simplest approach)
        batch_size = len(seqs)
        if batch_size > 1:
            return self.run(seqs, is_prefill=False)
        
        seq = seqs[0]
        
        # In decode mode, we only process the last token
        # The KV cache already contains all previous tokens
        last_token = seq.last_token
        last_position = seq.num_tokens - 1
        
        # Run main model on last token (uses KV cache)
        # We need hidden state for MTP, so use _forward_with_hidden_state
        hidden_state, main_logits = self._forward_with_hidden_state(
            token_ids=[[last_token]],
            positions=[[last_position]],
            is_prefill=False,  # Decode mode - uses KV cache
        )
        
        # hidden_state: [1, 1, hidden_size]
        # main_logits: [1, 1, vocab_size] (prediction for NEXT token)
        
        # Generate draft token with MTP
        # MTP predicts the NEXT token using current hidden state and last confirmed token
        draft_token, draft_logits = self._generate_draft_token(
            hidden_state=hidden_state,  # [1, 1, hidden_size]
            confirmed_token_id=last_token,
            position=last_position + 1,  # Position for the predicted token
        )
        
        # Get main model logits for verification
        # main_logits is already the prediction for the NEXT token
        main_next_logits = main_logits[0, 0]  # [vocab_size]
        
        # Verify draft token
        # Compare MTP's prediction with main model's prediction for the same token
        accepted = self._verify_draft_token(
            main_logits=main_next_logits,  # [vocab_size] - main model's prediction for t+1
            draft_logits=draft_logits[0, 0],  # [vocab_size] - MTP's prediction for t+1
            draft_token=int(draft_token[0, 0]),
            confirmed_token=last_token,
            temperature=seq.temperature,
        )
        
        if accepted:
            # Accept draft token
            return [int(draft_token[0, 0])]
        else:
            # Reject: sample from main model logits
            sampled_token = self._sample_fn(main_next_logits[jnp.newaxis, :], jnp.array([seq.temperature]))
            return [int(sampled_token[0])]
    
    def _forward_with_hidden_state(
        self,
        token_ids: List[int],
        positions: List[int],
        is_prefill: bool,
        use_kv_cache: bool = False,  # Default to False for MTP prefill
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning both hidden state and logits.
        
        For MTP speculative decoding, we need the FINAL hidden state
        (output of last layer, before final norm) to feed into the MTP head.
        
        Args:
            token_ids: Token IDs [batch, seq_len]
            positions: Positions [batch, seq_len]
            is_prefill: Whether this is prefill
            use_kv_cache: Whether to use/update self.kv_state (default False for MTP)
            
        Returns:
            Tuple of (hidden_state_for_mtp, logits)
            - hidden_state_for_mtp: Final hidden state after last layer [batch, seq_len, hidden_size]
            - logits: Output logits [batch, seq_len, vocab_size]
        """
        # For MTP prefill, just use the simple model.forward function
        # This avoids KV cache complexity
        from nanovllm_jax.model import forward as model_forward
        
        input_ids = jnp.array([token_ids] if isinstance(token_ids[0], int) else token_ids, dtype=jnp.int32)
        
        # Get hidden state
        hidden_pre, _ = model_forward(
            input_ids, 
            self.params, 
            self.config, 
            kv_cache_state=None,
            return_hidden=True,
        )
        
        # Get logits
        logits, _ = model_forward(
            input_ids,
            self.params,
            self.config,
            kv_cache_state=None,
            return_hidden=False,
        )
        
        return hidden_pre, logits
    
    def _generate_draft_token(
        self,
        hidden_state: jnp.ndarray,
        confirmed_token_id: int,
        position: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate draft token using MTP head.
        
        MTP predicts the NEXT token (t+1) using:
        - hidden_state from position t (output of main model's final layer)
        - embedding of confirmed token at position t
        
        This follows the Qwen3.5 MTP architecture where the MTP head
        uses both the context (hidden state) and the current token
        to predict the next token.
        
        Args:
            hidden_state: Pre-norm hidden state [1, 1, hidden_size]
            confirmed_token_id: Confirmed token ID at position t
            position: Position ID for the NEXT token (t+1)
            
        Returns:
            Tuple of (draft_token, draft_logits)
        """
        # Get embedding of confirmed token (at position t)
        # Shape: [1, 1, hidden_size]
        confirmed_embed = self.params.embed_tokens[jnp.array([[confirmed_token_id]])]
        
        # MTP forward pass
        # Predicts token at position t+1
        # Signature: mtp_forward(hidden_state, next_token_ids, embed_tokens, params, config, positions)
        # Returns (logits, hidden_state) tuple
        draft_logits, _ = mtp_forward(
            hidden_state=hidden_state,
            next_token_ids=jnp.array([[confirmed_token_id]]),
            embed_tokens=self.params.embed_tokens,
            params=self.params.mtp_params,
            config=self.config,
            positions=jnp.array([[position]]),
        )
        
        # Sample draft token (greedy for now)
        draft_token = jnp.argmax(draft_logits, axis=-1)
        
        return draft_token, draft_logits
    
def _verify_draft_token(
        self,
        main_logits: jnp.ndarray,
        draft_logits: jnp.ndarray,
        draft_token: int,
        confirmed_token: int,
        temperature: float,
    ) -> bool:
        """Verify draft token against main model.
        
        Args:
            main_logits: Main model logits [vocab_size]
            draft_logits: MTP logits [vocab_size]
            draft_token: Draft token ID
            confirmed_token: Confirmed token ID (for fallback)
            temperature: Sampling temperature
            
        Returns:
            True if draft token is accepted
        """
        # Compute probabilities
        main_probs = jax.nn.softmax(main_logits / temperature)
        draft_probs = jax.nn.softmax(draft_logits / temperature)
        
        # Get probability of draft token under main model
        draft_prob_main = main_probs[draft_token]
        draft_prob_draft = draft_probs[draft_token]
        
        # Acceptance probability
        acceptance_prob = jnp.minimum(1.0, draft_prob_main / (draft_prob_draft + 1e-10))
        
        # Sample acceptance
        if temperature > 0:
            acceptance_noise = jax.random.uniform(jax.random.PRNGKey(int(time.time() * 1000) % 2**31))
            return float(acceptance_noise) < float(acceptance_prob)
        else:
            # Greedy: accept if draft matches main argmax
            main_token = jnp.argmax(main_logits)
            return int(main_token) == int(draft_token)


from nanovllm_jax.engine.model_executor import ModelExecutor
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import HybridLayerState, KVCacheStorage, init_hybrid_state


class CanonicalModelRunner:
    """Canonical engine runner built around ModelExecutor.forward_step()."""

    def __init__(self, config: Qwen3_5Config, params: ModelParams, backend: str = "auto"):
        self.config = config
        self.params = params
        self.backend = select_backend(backend)
        self.executor = ModelExecutor(config, params, self.backend)
        self.block_size = config.block_size

        max_seqs = getattr(config, "max_num_seqs", 16)
        kv_spec = KVCacheSpec(
            num_layers=config.num_hidden_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.get_dtype(),
            max_kv_cache_bytes=config.max_kv_cache_bytes,
        )
        effective_num_blocks = cap_num_kv_cache_blocks(kv_spec)
        if effective_num_blocks != config.num_kvcache_blocks:
            print(
                "KV cache capped: "
                f"{config.num_kvcache_blocks} -> {effective_num_blocks} blocks "
                f"({config.max_kv_cache_bytes} byte cap)"
            )
            config.num_kvcache_blocks = effective_num_blocks
        self.max_blocks_per_seq = getattr(config, "max_blocks_per_seq", None)
        if self.max_blocks_per_seq is None:
            self.max_blocks_per_seq = max(1, effective_num_blocks // max_seqs)
            config.max_blocks_per_seq = self.max_blocks_per_seq
        self.execution = getattr(config, "jax_execution", "eager")

        self.cache_storage = self.backend.allocate_kv_cache(
            replace(kv_spec, num_blocks=effective_num_blocks),
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
        )
        self.hybrid_states: Dict[int, HybridLayerState] = {}
        self._max_hybrid_slots = max_seqs
        self._hybrid_slots: Dict[int, int] = {}
        self._free_hybrid_slots: List[int] = list(range(max_seqs))

        self.kv_state = init_kv_cache(
            num_blocks=effective_num_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
            num_layers=config.num_hidden_layers,
            dtype=config.get_dtype(),
            max_kv_cache_bytes=config.max_kv_cache_bytes,
        )
        self.kv_state = init_linear_attention_states(
            cache=self.kv_state,
            config=config,
            batch_size=1,
            dtype=config.get_dtype(),
        )
        self._empty_hybrid_state = self.kv_state.hybrid_state
        self._hybrid_state_table = init_hybrid_state(
            self.config,
            batch_size=max_seqs,
            dtype=self.config.get_dtype(),
        )
        self._sample_fn = jax.jit(self._sample_logits)
        self.mtp_enabled = hasattr(params, "mtp_params") and params.mtp_params is not None
        self.num_speculative_tokens = int(getattr(config, "num_speculative_tokens", 0) or 0)
        self.mtp1_enabled = self.mtp_enabled and self.num_speculative_tokens > 0
        self._mtp1_forward_jit = None
        self._mtp1_token_jit = None
        self._hidden_token_jit = None
        self._mtp1_drafts: Dict[int, int] = {}
        self._mtp1_seeded_chain: Dict[int, int] = {}
        self._mtp1_draft_debug: Dict[int, Dict[str, Any]] = {}
        self._mtp1_debug_events: List[Dict[str, Any]] = []
        self.reset_speculative_stats()
        self._warmup_compiled = False

    def reset_speculative_stats(self):
        self.speculative_stats = {
            "enabled": False,
            "drafts_proposed": 0,
            "drafts_accepted": 0,
            "drafts_rejected": 0,
            "bonus_tokens": 0,
            "fallback_steps": 0,
            "fallback_gated_no_spec_steps": 0,
            "fallback_seeded_main_steps": 0,
            "fallback_partial_rows": 0,
            "draft_position_proposed": [],
            "draft_position_accepted": [],
        }
        if not hasattr(self, "_mtp1_debug_events"):
            self._mtp1_debug_events = []
        if not hasattr(self, "_mtp1_draft_debug"):
            self._mtp1_draft_debug = {}
        self._mtp1_debug_events.clear()
        self._mtp1_draft_debug.clear()

    def get_speculative_stats(self) -> Dict[str, int | bool | float]:
        if not hasattr(self, "speculative_stats"):
            self.reset_speculative_stats()
        stats = dict(self.speculative_stats)
        stats["enabled"] = bool(self.mtp1_enabled)
        proposed = stats["drafts_proposed"]
        stats["acceptance_rate"] = stats["drafts_accepted"] / proposed if proposed else 0.0
        stats["debug_events"] = list(getattr(self, "_mtp1_debug_events", [])[-16:])
        return stats

    def _speculative_stats(self) -> Dict[str, int | bool]:
        if not hasattr(self, "speculative_stats"):
            self.reset_speculative_stats()
        return self.speculative_stats

    def _mtp_adaptive_gated(self) -> bool:
        # Adaptive MTP admission is scheduler-owned so it can be keyed by the
        # active physical bucket. Keep this legacy runner hook inert; rows that
        # are gated by the scheduler arrive with ``seq.mtp_admitted = False``.
        return False

    @staticmethod
    def _seq_mtp_admitted(seq: Sequence) -> bool:
        return bool(getattr(seq, "mtp_admitted", True))

    def _clear_mtp1_drafts_for_rows(self, seqs: List[Sequence], rows: List[int]) -> None:
        for row in rows:
            seq = seqs[row]
            self._mtp1_drafts.pop(seq.seq_id, None)
            self._mtp1_seeded_chain.pop(seq.seq_id, None)
            self._mtp1_debug_state()[0].pop(seq.seq_id, None)

    def _record_draft_position_acceptance(self, accepted_matrix: List[List[bool]]):
        if not accepted_matrix:
            return
        stats = self._speculative_stats()
        max_width = max(len(row) for row in accepted_matrix)
        proposed = stats.setdefault("draft_position_proposed", [])
        accepted = stats.setdefault("draft_position_accepted", [])
        while len(proposed) < max_width:
            proposed.append(0)
            accepted.append(0)
        for row in accepted_matrix:
            for idx, value in enumerate(row):
                proposed[idx] += 1
                accepted[idx] += int(bool(value))

    def _mtp1_debug_state(self) -> tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        if not hasattr(self, "_mtp1_draft_debug"):
            self._mtp1_draft_debug = {}
        if not hasattr(self, "_mtp1_debug_events"):
            self._mtp1_debug_events = []
        return self._mtp1_draft_debug, self._mtp1_debug_events

    def warmup_compilation(self, max_prefill_len: int = 64, max_batch: int = 1):
        """Compile configured static shapes through the canonical executor path."""
        if self._warmup_compiled:
            return
        if self.execution not in {"decode-jit", "jit"}:
            self._warmup_compiled = True
            return

        prefill_buckets = tuple(getattr(self.config, "prefill_buckets", ())) or (max_prefill_len,)
        batch_buckets = tuple(getattr(self.config, "batch_size_buckets", ())) or (max_batch,)

        for prefill_len in prefill_buckets:
            if self.execution != "jit":
                break
            for batch_size in batch_buckets:
                batch = self._dummy_batch(batch_size=batch_size, query_len=prefill_len, is_prefill=True)
                hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                output = self.executor.forward_step_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    last_logits_only=True,
                )
                output.activations.block_until_ready()
                self.cache_storage = output.cache_storage

        for batch_size in batch_buckets:
            batch = self._dummy_batch(batch_size=batch_size, query_len=1, is_prefill=False)
            hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
            output = self.executor.forward_step_jit(
                batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                last_logits_only=True,
            )
            output.activations.block_until_ready()
            self.cache_storage = output.cache_storage
            self._sample_fn(
                jnp.zeros((batch_size, self.config.vocab_size), dtype=jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            ).block_until_ready()
        self._warmup_compiled = True

    def _dummy_batch(self, *, batch_size: int, query_len: int, is_prefill: bool) -> ScheduledBatch:
        block_tables = []
        for row in range(batch_size):
            start = row * self.max_blocks_per_seq
            block_tables.append(list(range(start, start + self.max_blocks_per_seq)))
        query_lens = [query_len if is_prefill else 1] * batch_size
        query_start_loc = [0]
        for qlen in query_lens:
            query_start_loc.append(query_start_loc[-1] + qlen)
        positions = [list(range(query_len)) for _ in range(batch_size)]
        return ScheduledBatch(
            tokens=jnp.zeros((batch_size, query_len), dtype=jnp.int32),
            positions=jnp.array(positions, dtype=jnp.int32),
            seq_ids=jnp.arange(batch_size, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else batch_size,
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.full((batch_size,), query_len if is_prefill else 1, dtype=jnp.int32),
        )

    def release(self, seq_ids: List[int]):
        """Release per-sequence hybrid state once a request is finished."""
        for seq_id in seq_ids:
            self.hybrid_states.pop(seq_id, None)
            slot = self._hybrid_slots.pop(seq_id, None)
            if slot is not None:
                self._zero_hybrid_slot(slot)
                self._free_hybrid_slots.append(slot)
            self._mtp1_drafts.pop(seq_id, None)

    def _build_scheduled_batch(self, seqs: List[Sequence], is_prefill: bool) -> ScheduledBatch:
        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        if self.max_blocks_per_seq is not None:
            if max_blocks > self.max_blocks_per_seq:
                raise ValueError(
                    f"scheduled block table needs {max_blocks} blocks but bucket has {self.max_blocks_per_seq}"
                )
            max_blocks = self.max_blocks_per_seq
        for seq in seqs:
            if is_prefill:
                start = seq.num_cached_tokens
                tokens = seq.token_ids[start:]
                positions = list(range(start, seq.num_tokens))
            else:
                tokens = [seq.last_token]
                positions = [seq.num_tokens - 1]
            if not tokens:
                raise ValueError(f"Scheduled sequence {seq.seq_id} has no executable tokens")
            query_tokens.append(tokens)
            query_positions.append(positions)
            block_tables.append(seq.block_table + [0] * (max_blocks - len(seq.block_table)))
            seq_lens.append(seq.num_tokens)
            query_lens.append(len(tokens))

        max_query_len = max(query_lens)
        query_len_bucket = max_query_len
        prefill_buckets = tuple(getattr(self.config, "prefill_buckets", ()))
        if is_prefill and prefill_buckets:
            query_len_bucket = self._select_bucket(max_query_len, prefill_buckets, "prefill")

        batch_size_bucket = len(seqs)
        batch_size_buckets = tuple(getattr(self.config, "batch_size_buckets", ()))
        if batch_size_buckets:
            batch_size_bucket = self._select_bucket(len(seqs), batch_size_buckets, "batch")

        padded_tokens = [tokens + [0] * (query_len_bucket - len(tokens)) for tokens in query_tokens]
        padded_positions = [positions + [0] * (query_len_bucket - len(positions)) for positions in query_positions]
        query_start_loc = [0]
        for qlen in query_lens:
            query_start_loc.append(query_start_loc[-1] + qlen)
        for _ in range(batch_size_bucket - len(seqs)):
            padded_tokens.append([0] * query_len_bucket)
            padded_positions.append([0] * query_len_bucket)
            block_tables.append([0] * max_blocks)
            seq_lens.append(0)
            query_lens.append(0)
            query_start_loc.append(query_start_loc[-1])

        return ScheduledBatch(
            tokens=jnp.array(padded_tokens, dtype=jnp.int32),
            positions=jnp.array(padded_positions, dtype=jnp.int32),
            seq_ids=jnp.array([seq.seq_id for seq in seqs] + [-1] * (batch_size_bucket - len(seqs)), dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else sum(query_lens),
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
        )

    @staticmethod
    def _select_bucket(size: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in sorted(buckets):
            if size <= bucket:
                return bucket
        raise ValueError(f"{name} size {size} exceeds configured buckets {buckets}")

    def _zero_hybrid_slot(self, slot: int):
        if slot < 0:
            return
        self._hybrid_state_table = HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state.at[slot].set(
                jnp.zeros_like(self._hybrid_state_table.conv_state[slot])
            )
            if self._hybrid_state_table.conv_state is not None
            else None,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot].set(
                jnp.zeros_like(self._hybrid_state_table.recurrent_state[slot])
            )
            if self._hybrid_state_table.recurrent_state is not None
            else None,
        )

    def _ensure_hybrid_slot(self, seq_id: int) -> int:
        if seq_id < 0:
            return -1
        slot = self._hybrid_slots.get(seq_id)
        if slot is not None:
            return slot
        if not self._free_hybrid_slots:
            raise RuntimeError("No free hybrid-state slots; max_num_seqs is exhausted")
        if seq_id < self._max_hybrid_slots and seq_id in self._free_hybrid_slots:
            slot = seq_id
            self._free_hybrid_slots.remove(slot)
        else:
            slot = self._free_hybrid_slots.pop()
        self._zero_hybrid_slot(slot)
        self._hybrid_slots[seq_id] = slot
        return slot

    def _get_hybrid_state(self, seq_id: int) -> HybridLayerState:
        if seq_id < 0:
            return self._empty_hybrid_state
        slot = self._ensure_hybrid_slot(seq_id)
        return HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state[slot : slot + 1]
            if self._hybrid_state_table.conv_state is not None
            else None,
            recurrent_state=self._hybrid_state_table.recurrent_state[slot : slot + 1]
            if self._hybrid_state_table.recurrent_state is not None
            else None,
        )

    def _set_hybrid_state(self, seq_id: int, state: HybridLayerState | None):
        if state is None or seq_id < 0:
            return
        slot = self._ensure_hybrid_slot(seq_id)
        self._hybrid_state_table = HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state.at[slot].set(state.conv_state[0])
            if self._hybrid_state_table.conv_state is not None and state.conv_state is not None
            else self._hybrid_state_table.conv_state,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot].set(state.recurrent_state[0])
            if self._hybrid_state_table.recurrent_state is not None and state.recurrent_state is not None
            else self._hybrid_state_table.recurrent_state,
        )

    def _slice_batch(self, batch: ScheduledBatch, idx: int) -> ScheduledBatch:
        query_len = int(batch.query_lens[idx])
        return ScheduledBatch(
            tokens=batch.tokens[idx : idx + 1, :query_len],
            positions=batch.positions[idx : idx + 1, :query_len],
            seq_ids=batch.seq_ids[idx : idx + 1],
            query_start_loc=jnp.array([0, query_len], dtype=jnp.int32),
            is_prefill=batch.is_prefill,
            num_prefill_tokens=query_len if batch.is_prefill else 0,
            num_decode_tokens=0 if batch.is_prefill else 1,
            block_tables=batch.block_tables[idx : idx + 1],
            seq_lens=batch.seq_lens[idx : idx + 1],
        )

    def _masked_decode_batch(
        self,
        batch: ScheduledBatch,
        rows: List[int],
        *,
        token_values: List[int] | None = None,
        position_values: List[int] | None = None,
        seq_len_values: List[int] | None = None,
    ) -> ScheduledBatch:
        if not rows:
            raise ValueError("rows must not be empty")
        if batch.is_prefill:
            raise ValueError("masked decode batches require a decode batch")
        batch_size = int(batch.tokens.shape[0])
        row_ids = jnp.array(rows, dtype=jnp.int32)
        active = jnp.zeros((batch_size,), dtype=bool).at[row_ids].set(True)
        tokens = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        seq_lens = jnp.zeros((batch_size,), dtype=jnp.int32)
        if token_values is None:
            tokens = tokens.at[row_ids, 0].set(batch.tokens[row_ids, 0])
        else:
            tokens = tokens.at[row_ids, 0].set(jnp.array(token_values, dtype=jnp.int32))
        if position_values is None:
            positions = positions.at[row_ids, 0].set(batch.positions[row_ids, 0])
        else:
            positions = positions.at[row_ids, 0].set(jnp.array(position_values, dtype=jnp.int32))
        if seq_len_values is None:
            seq_lens = seq_lens.at[row_ids].set(batch.seq_lens[row_ids])
        else:
            seq_lens = seq_lens.at[row_ids].set(jnp.array(seq_len_values, dtype=jnp.int32))
        query_lens = active.astype(jnp.int32)
        return ScheduledBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=jnp.where(active, batch.seq_ids, jnp.full_like(batch.seq_ids, -1)),
            query_start_loc=jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens),
                ]
            ),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=len(rows),
            block_tables=jnp.where(active[:, None], batch.block_tables, jnp.zeros_like(batch.block_tables)),
            seq_lens=seq_lens,
        )

    def _compact_decode_batch(
        self,
        batch: ScheduledBatch,
        rows: List[int],
        *,
        token_values: List[int] | None = None,
        position_values: List[int] | None = None,
        seq_len_values: List[int] | None = None,
    ) -> ScheduledBatch:
        if not rows:
            raise ValueError("rows must not be empty")
        if batch.is_prefill:
            raise ValueError("compact decode batches require a decode batch")

        row_ids = jnp.array(rows, dtype=jnp.int32)
        if token_values is None:
            tokens = batch.tokens[row_ids, :1]
        else:
            tokens = jnp.array(token_values, dtype=jnp.int32)[:, None]
        if position_values is None:
            positions = batch.positions[row_ids, :1]
        else:
            positions = jnp.array(position_values, dtype=jnp.int32)[:, None]
        if seq_len_values is None:
            seq_lens = batch.seq_lens[row_ids]
        else:
            seq_lens = jnp.array(seq_len_values, dtype=jnp.int32)

        compact_size = len(rows)
        return ScheduledBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=batch.seq_ids[row_ids],
            query_start_loc=jnp.arange(compact_size + 1, dtype=jnp.int32),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=compact_size,
            block_tables=batch.block_tables[row_ids],
            seq_lens=seq_lens,
        )

    def _batch_hybrid_state(self, batch: ScheduledBatch) -> HybridLayerState:
        slot_values = [self._ensure_hybrid_slot(int(seq_id)) for seq_id in batch.seq_ids.tolist()]
        slot_ids = jnp.array(slot_values, dtype=jnp.int32)
        safe_slot_ids = jnp.maximum(slot_ids, 0)
        valid = slot_ids >= 0
        conv_state = None
        recurrent_state = None
        if self._hybrid_state_table.conv_state is not None:
            conv_state = self._hybrid_state_table.conv_state[safe_slot_ids]
            conv_state = jnp.where(
                valid.reshape((valid.shape[0],) + (1,) * (conv_state.ndim - 1)),
                conv_state,
                jnp.zeros_like(conv_state),
            )
        if self._hybrid_state_table.recurrent_state is not None:
            recurrent_state = self._hybrid_state_table.recurrent_state[safe_slot_ids]
            recurrent_state = jnp.where(
                valid.reshape((valid.shape[0],) + (1,) * (recurrent_state.ndim - 1)),
                recurrent_state,
                jnp.zeros_like(recurrent_state),
            )
        return HybridLayerState(conv_state=conv_state, recurrent_state=recurrent_state)

    def _store_batch_hybrid_state(self, batch: ScheduledBatch, state: HybridLayerState | None):
        if state is None:
            return
        valid_rows: List[int] = []
        slot_values: List[int] = []
        query_lens = [int(x) for x in batch.query_lens.tolist()]
        for row, seq_id in enumerate([int(x) for x in batch.seq_ids.tolist()]):
            if seq_id < 0 or (not batch.is_prefill and query_lens[row] <= 0):
                continue
            valid_rows.append(row)
            slot_values.append(self._ensure_hybrid_slot(seq_id))
        if not valid_rows:
            return
        row_ids = jnp.array(valid_rows, dtype=jnp.int32)
        slot_ids = jnp.array(slot_values, dtype=jnp.int32)
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and state.conv_state is not None
            and state.recurrent_state is not None
            and len(valid_rows) == len(slot_values) == state.conv_state.shape[0]
            and state.conv_state.shape[0] == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
        ):
            self._hybrid_state_table = state
            return
        self._hybrid_state_table = HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state.at[slot_ids].set(state.conv_state[row_ids])
            if self._hybrid_state_table.conv_state is not None and state.conv_state is not None
            else self._hybrid_state_table.conv_state,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot_ids].set(state.recurrent_state[row_ids])
            if self._hybrid_state_table.recurrent_state is not None and state.recurrent_state is not None
            else self._hybrid_state_table.recurrent_state,
        )

    def _refresh_kv_snapshot(self, batch: ScheduledBatch, hybrid_state: HybridLayerState | None = None):
        if hybrid_state is None:
            hybrid_state = self._batch_hybrid_state(batch)
        metadata = self.backend.build_attention_metadata(
            positions=batch.positions,
            block_tables=batch.block_tables,
            seq_lens=batch.seq_lens,
            block_size=self.config.block_size,
            is_prefill=batch.is_prefill,
            query_start_loc=batch.query_start_loc,
            num_prefill_tokens=batch.num_prefill_tokens,
            num_decode_tokens=batch.num_decode_tokens,
        )
        self.kv_state = KVCacheState(
            k_cache=self.cache_storage.k_cache,
            v_cache=self.cache_storage.v_cache,
            block_table=batch.block_tables,
            kv_lens=batch.seq_lens,
            slot_mapping=metadata.slot_mapping,
            conv_state=hybrid_state.conv_state,
            recurrent_state=hybrid_state.recurrent_state,
        )

    def _record_kv_snapshot(self, batch: ScheduledBatch, hybrid_state: HybridLayerState | None = None):
        """Update the legacy KV snapshot without rebuilding attention metadata.

        The canonical serving path uses ``cache_storage`` plus scheduled
        per-step metadata, not ``self.kv_state``, for execution. This snapshot is
        kept for compatibility/introspection only; rebuilding slot metadata here
        is avoidable hot-path work for speculative decode.
        """
        if hybrid_state is None:
            hybrid_state = self._batch_hybrid_state(batch)
        slot_mapping = getattr(self.kv_state, "slot_mapping", None)
        if slot_mapping is None:
            slot_mapping = jnp.zeros_like(batch.positions, dtype=jnp.int32)
        self.kv_state = KVCacheState(
            k_cache=self.cache_storage.k_cache,
            v_cache=self.cache_storage.v_cache,
            block_table=batch.block_tables,
            kv_lens=batch.seq_lens,
            slot_mapping=slot_mapping,
            conv_state=hybrid_state.conv_state,
            recurrent_state=hybrid_state.recurrent_state,
        )

    def _step_fn(self, batch: ScheduledBatch):
        execution = getattr(self, "execution", "eager")
        if execution == "jit" or (execution == "decode-jit" and not batch.is_prefill):
            return self.executor.forward_step_jit
        return self.executor.forward_step

    def _mtp1_verifier_step_fn(self):
        if getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            return self.executor.forward_step_jit
        return self.executor.forward_step

    def _logits_from_hidden(self, hidden: jnp.ndarray) -> jnp.ndarray:
        hidden = self._final_norm_hidden(hidden)
        return self._logits_from_normed_hidden(hidden)

    def _logits_from_normed_hidden(self, hidden: jnp.ndarray) -> jnp.ndarray:
        if self.params.lm_head is not None:
            return jnp.dot(hidden, self.params.lm_head)
        return jnp.dot(hidden, self.params.embed_tokens.T)

    def _final_norm_hidden(self, hidden: jnp.ndarray) -> jnp.ndarray:
        from nanovllm_jax.layers import rms_norm

        return rms_norm(hidden, self.params.norm_weight, self.config.rms_norm_eps).astype(jnp.float32)

    def _hidden_for_mtp(self, hidden: jnp.ndarray) -> jnp.ndarray:
        if getattr(self, "mtp_hidden_source", "final_normed") == "final_normed":
            return self._final_norm_hidden(hidden)
        return hidden

    @staticmethod
    def _topk_debug(logits: jnp.ndarray, k: int = 5) -> Dict[str, List[int] | List[float]]:
        values, ids = jax.lax.top_k(logits.astype(jnp.float32), min(k, logits.shape[-1]))
        return {
            "ids": [int(x) for x in ids.tolist()],
            "values": [float(x) for x in values.tolist()],
        }

    @staticmethod
    def _token_rank(logits: jnp.ndarray, token_id: int) -> int:
        logits = logits.astype(jnp.float32)
        token_logit = logits[token_id]
        return int((jnp.sum(logits > token_logit) + 1).item())

    def _mtp1_params_tree(self):
        params = self.params.mtp_params
        return (
            params.eh_proj,
            tuple(params.layers),
            params.pre_fc_norm_hidden,
            params.pre_fc_norm_embedding,
            params.final_norm,
            params.lm_head,
        )

    @staticmethod
    def _mtp1_params_from_tree(tree) -> MTPParams:
        eh_proj, layers, pre_fc_norm_hidden, pre_fc_norm_embedding, final_norm, lm_head = tree
        return MTPParams(
            eh_proj=eh_proj,
            layers=list(layers),
            pre_fc_norm_hidden=pre_fc_norm_hidden,
            pre_fc_norm_embedding=pre_fc_norm_embedding,
            final_norm=final_norm,
            lm_head=lm_head,
        )

    def _mtp1_logits(self, hidden_state: jnp.ndarray, token_ids: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        def forward(hidden_arg, token_arg, position_arg, embed_tokens_arg, mtp_params_tree):
            logits, _ = mtp_forward(
                hidden_state=hidden_arg,
                next_token_ids=token_arg,
                embed_tokens=embed_tokens_arg,
                params=self._mtp1_params_from_tree(mtp_params_tree),
                config=self.config,
                positions=position_arg,
            )
            return logits

        mtp_params_tree = self._mtp1_params_tree()
        if getattr(self, "mtp_compile_draft", False) and getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            if getattr(self, "_mtp1_forward_jit", None) is None:
                self._mtp1_forward_jit = jax.jit(forward)
            return self._mtp1_forward_jit(hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)
        return forward(hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)

    def _mtp1_draft_token(self, hidden_state: jnp.ndarray, token_ids: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        def forward(hidden_arg, token_arg, position_arg, embed_tokens_arg, mtp_params_tree):
            logits, _ = mtp_forward(
                hidden_state=hidden_arg,
                next_token_ids=token_arg,
                embed_tokens=embed_tokens_arg,
                params=self._mtp1_params_from_tree(mtp_params_tree),
                config=self.config,
                positions=position_arg,
            )
            return jnp.argmax(logits[:, 0], axis=-1).astype(jnp.int32)

        mtp_params_tree = self._mtp1_params_tree()
        if getattr(self, "mtp_compile_draft", False) and getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            if getattr(self, "_mtp1_token_jit", None) is None:
                self._mtp1_token_jit = jax.jit(forward)
            return self._mtp1_token_jit(hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)
        return forward(hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)

    def _mtp1_draft_chain(
        self,
        hidden_state: jnp.ndarray,
        token_ids: jnp.ndarray,
        positions: jnp.ndarray,
        draft_len: int,
    ) -> jnp.ndarray:
        def forward(hidden_arg, token_arg, position_arg, embed_tokens_arg, mtp_params_tree):
            mtp_params = self._mtp1_params_from_tree(mtp_params_tree)
            current_hidden = hidden_arg
            current_token = token_arg
            current_position = position_arg
            drafts = []
            for _ in range(draft_len):
                logits, current_hidden = mtp_forward(
                    hidden_state=current_hidden,
                    next_token_ids=current_token,
                    embed_tokens=embed_tokens_arg,
                    params=mtp_params,
                    config=self.config,
                    positions=current_position,
                )
                current_token = jnp.argmax(logits[:, 0], axis=-1).astype(jnp.int32)[:, None]
                drafts.append(current_token[:, 0])
                current_position = current_position + 1
            return jnp.stack(drafts, axis=1)

        mtp_params_tree = self._mtp1_params_tree()
        if getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            if not hasattr(self, "_mtp1_chain_jit"):
                self._mtp1_chain_jit = {}
            if draft_len not in self._mtp1_chain_jit:
                self._mtp1_chain_jit[draft_len] = jax.jit(forward)
            return self._mtp1_chain_jit[draft_len](hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)
        return forward(hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)

    def _mtp1_draft_chain_with_margin(
        self,
        hidden_state: jnp.ndarray,
        token_ids: jnp.ndarray,
        positions: jnp.ndarray,
        draft_len: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def forward(hidden_arg, token_arg, position_arg, embed_tokens_arg, mtp_params_tree):
            mtp_params = self._mtp1_params_from_tree(mtp_params_tree)
            current_hidden = hidden_arg
            current_token = token_arg
            current_position = position_arg
            drafts = []
            first_margin = None
            for idx in range(draft_len):
                logits, current_hidden = mtp_forward(
                    hidden_state=current_hidden,
                    next_token_ids=current_token,
                    embed_tokens=embed_tokens_arg,
                    params=mtp_params,
                    config=self.config,
                    positions=current_position,
                )
                if idx == 0:
                    top2, _ = jax.lax.top_k(logits[:, 0].astype(jnp.float32), 2)
                    first_margin = top2[:, 0] - top2[:, 1]
                current_token = jnp.argmax(logits[:, 0], axis=-1).astype(jnp.int32)[:, None]
                drafts.append(current_token[:, 0])
                current_position = current_position + 1
            if first_margin is None:
                first_margin = jnp.zeros((hidden_arg.shape[0],), dtype=jnp.float32)
            return jnp.stack(drafts, axis=1), first_margin

        mtp_params_tree = self._mtp1_params_tree()
        if getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            if not hasattr(self, "_mtp1_chain_margin_jit"):
                self._mtp1_chain_margin_jit = {}
            if draft_len not in self._mtp1_chain_margin_jit:
                self._mtp1_chain_margin_jit[draft_len] = jax.jit(forward)
            return self._mtp1_chain_margin_jit[draft_len](
                hidden_state,
                token_ids,
                positions,
                self.params.embed_tokens,
                mtp_params_tree,
            )
        return forward(hidden_state, token_ids, positions, self.params.embed_tokens, mtp_params_tree)

    def _greedy_tokens_from_hidden(self, hidden: jnp.ndarray) -> jnp.ndarray:
        from nanovllm_jax.layers import rms_norm

        output_weight = self.params.lm_head if self.params.lm_head is not None else self.params.embed_tokens.T

        def forward(hidden_arg, norm_weight_arg, output_weight_arg):
            hidden_norm = rms_norm(hidden_arg, norm_weight_arg, self.config.rms_norm_eps).astype(jnp.float32)
            logits = jnp.dot(hidden_norm, output_weight_arg)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        if getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            if getattr(self, "_hidden_token_jit", None) is None:
                self._hidden_token_jit = jax.jit(forward)
            return self._hidden_token_jit(hidden, self.params.norm_weight, output_weight)
        return forward(hidden, self.params.norm_weight, output_weight)

    @staticmethod
    def _last_query_activations(activations: jnp.ndarray, batch: ScheduledBatch, num_seqs: int) -> jnp.ndarray:
        query_lens = batch.query_lens[:num_seqs]
        gather_idx = jnp.clip(query_lens - 1, 0, activations.shape[1] - 1).astype(jnp.int32)
        return activations[jnp.arange(num_seqs), gather_idx]

    def _run_main_and_sample(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        *,
        seed_mtp1: bool,
    ) -> List[int | List[int]]:
        def _replay_copy(value):
            if value is None:
                return None
            copied = jnp.array(value, copy=True)
            copied.block_until_ready()
            return copied

        def _replay_copy_tree(value):
            if value is None:
                return None
            if hasattr(value, "k_cache") and hasattr(value, "v_cache"):
                return type(value)(
                    k_cache=_replay_copy(value.k_cache),
                    v_cache=_replay_copy(value.v_cache),
                )
            if hasattr(value, "conv_state") and hasattr(value, "recurrent_state"):
                return type(value)(
                    conv_state=_replay_copy(value.conv_state),
                    recurrent_state=_replay_copy(value.recurrent_state),
                )
            return jax.tree_util.tree_map(_replay_copy, value)

        hybrid_state = self._batch_hybrid_state(batch)
        if batch.is_prefill:
            prefill_final_flags = list(batch.prefill_final_flags)[: len(seqs)]
            if len(prefill_final_flags) < len(seqs):
                prefill_final_flags.extend([True] * (len(seqs) - len(prefill_final_flags)))
        else:
            prefill_final_flags = [True] * len(seqs)

        return_hidden_for_seed = bool(seed_mtp1)
        output = self._step_fn(batch)(
            batch,
            cache_storage=self.cache_storage,
            hybrid_state=hybrid_state,
            return_hidden=return_hidden_for_seed,
            return_hidden_with_logits=return_hidden_for_seed,
            last_logits_only=True,
        )
        self.cache_storage = output.cache_storage
        self._store_batch_hybrid_state(batch, output.hybrid_state)
        self._refresh_kv_snapshot(batch, output.hybrid_state)

        last_hidden = None
        seed_hidden = None
        if return_hidden_for_seed:
            hidden_activations, logits = output.activations
            last_logits = logits[: len(seqs), 0]
            last_hidden = self._last_query_activations(hidden_activations, batch, len(seqs))
            seed_hidden = self._hidden_for_mtp(last_hidden[:, None, :])[:, 0]
        else:
            last_logits = output.activations[: len(seqs), 0]
        query_lens = [int(x) for x in batch.query_lens[: len(seqs)].tolist()]
        active_rows = [
            row
            for row, query_len in enumerate(query_lens)
            if query_len > 0 and int(batch.seq_ids[row]) >= 0
        ]
        if batch.is_prefill:
            prefill_logits_by_seq = getattr(self, "_last_prefill_logits_by_seq", None)
            if prefill_logits_by_seq is None:
                prefill_logits_by_seq = {}
                self._last_prefill_logits_by_seq = prefill_logits_by_seq
            for row, seq in enumerate(seqs):
                if row in active_rows and row < len(prefill_final_flags) and prefill_final_flags[row]:
                    prefill_logits_by_seq[int(seq.seq_id)] = last_logits[row]

        token_by_row: dict[int, int] = {}
        if active_rows:
            active_idx = jnp.array(active_rows, dtype=jnp.int32)
            temperatures = jnp.array([seqs[row].temperature for row in active_rows], dtype=jnp.float32)
            token_ids = self._sample_fn(last_logits[active_idx], temperatures)
            token_by_row = {
                row: int(token_id)
                for row, token_id in zip(active_rows, token_ids.tolist())
            }

        outputs: List[int | List[int]] = []
        for row, seq in enumerate(seqs):
            if row not in token_by_row:
                outputs.append([])
                continue
            token_id = token_by_row[row]
            if batch.is_prefill and not prefill_final_flags[row]:
                outputs.append([])
                continue

            outputs.append(token_id)

            if seed_hidden is not None:
                self._seed_mtp1_drafts([seq], seed_hidden[row : row + 1], [token_id])

        return outputs

    def _run_main_and_sample_with_mtp1_reuse(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        *,
        seed_mtp1: bool,
        force_emit_bonus: bool = False,
    ) -> List[int | List[int]]:
        """Decode once with the target model and verify any stored MTP1 drafts.

        The previous MTP1 serving path verified a draft by running a two-token
        target-model prefill on ``[last_token, draft]``. That recomputed the
        existing decode forward for ``last_token``. This path instead uses the
        logits from the normal scheduled decode step to verify the stored draft.

        For accepted rows, this then runs one normal target decode on the draft
        token to produce the speculative bonus token. That preserves the usual
        K=1 speculative contract without recomputing the current token.
        """
        def _replay_copy(value):
            if value is None:
                return None
            copied = jnp.array(value, copy=True)
            copied.block_until_ready()
            return copied

        def _replay_copy_tree(value):
            if value is None:
                return None
            if hasattr(value, "k_cache") and hasattr(value, "v_cache"):
                return type(value)(
                    k_cache=_replay_copy(value.k_cache),
                    v_cache=_replay_copy(value.v_cache),
                )
            if hasattr(value, "conv_state") and hasattr(value, "recurrent_state"):
                return type(value)(
                    conv_state=_replay_copy(value.conv_state),
                    recurrent_state=_replay_copy(value.recurrent_state),
                )
            return jax.tree_util.tree_map(_replay_copy, value)

        hybrid_state = self._batch_hybrid_state(batch)
        return_hidden_for_seed = bool(seed_mtp1)
        output = self._step_fn(batch)(
            batch,
            cache_storage=self.cache_storage,
            hybrid_state=hybrid_state,
            return_hidden=return_hidden_for_seed,
            return_hidden_with_logits=return_hidden_for_seed,
            last_logits_only=True,
        )
        self.cache_storage = output.cache_storage
        self._store_batch_hybrid_state(batch, output.hybrid_state)
        self._refresh_kv_snapshot(batch, output.hybrid_state)

        last_hidden = None
        seed_hidden = None
        if return_hidden_for_seed:
            hidden_activations, logits = output.activations
            last_logits = logits[: len(seqs), 0]
            last_hidden = self._last_query_activations(hidden_activations, batch, len(seqs))
            seed_hidden = self._hidden_for_mtp(last_hidden[:, None, :])[:, 0]
        else:
            last_logits = output.activations[: len(seqs), 0]

        temperatures = jnp.array([seq.temperature for seq in seqs], dtype=jnp.float32)
        token_ids = self._sample_fn(last_logits, temperatures)
        target_tokens = [int(token_id) for token_id in token_ids.tolist()]

        main_lookahead_all = os.environ.get("NANO_VLLM_JAX_MAIN_LOOKAHEAD_ALL", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        if (
            main_lookahead_all
            and not batch.is_prefill
            and all(seq.temperature == 0 for seq in seqs)
            and all(seq.num_completion_tokens + 2 <= seq.max_tokens for seq in seqs)
            and all(int(length) == 1 for length in batch.query_lens[: len(seqs)].tolist())
        ):
            lookahead_rows = list(range(len(seqs)))
            lookahead_batch = self._masked_decode_batch(
                batch,
                lookahead_rows,
                token_values=[int(token_id) for token_id in token_ids.tolist()],
                position_values=[int(batch.positions[row, 0]) + 1 for row in lookahead_rows],
                seq_len_values=[int(batch.seq_lens[row]) + 1 for row in lookahead_rows],
            )
            lookahead_hybrid = self._batch_hybrid_state(lookahead_batch)
            lookahead_output = self._step_fn(lookahead_batch)(
                lookahead_batch,
                cache_storage=self.cache_storage,
                hybrid_state=lookahead_hybrid,
                return_hidden=False,
                last_logits_only=True,
            )
            self.cache_storage = lookahead_output.cache_storage
            self._store_batch_hybrid_state(lookahead_batch, lookahead_output.hybrid_state)
            self._refresh_kv_snapshot(lookahead_batch, lookahead_output.hybrid_state)
            lookahead_logits = lookahead_output.activations[: len(seqs), 0]
            bonus_token_ids = self._sample_fn(lookahead_logits, temperatures)
            bonus_tokens = [int(token_id) for token_id in bonus_token_ids.tolist()]
            return [[target_tokens[row], bonus_tokens[row]] for row in range(len(seqs))]

        outputs: List[int | List[int] | None] = [None] * len(seqs)
        stats = self._speculative_stats()
        debug_by_seq, debug_events = self._mtp1_debug_state()
        accepted_rows: List[int] = []
        accepted_drafts: List[int] = []
        draft_chains_by_row: dict[int, List[int]] = {}
        seed_normal_rows: List[int] = []
        seed_normal_tokens: List[int] = []

        for row, (seq, target_token) in enumerate(zip(seqs, target_tokens)):
            draft_value = self._mtp1_drafts.pop(seq.seq_id, None)
            draft_tokens = (
                [int(token) for token in draft_value]
                if isinstance(draft_value, list)
                else ([int(draft_value)] if draft_value is not None else [])
            )
            first_draft = draft_tokens[0] if draft_tokens else None
            # Accepted MTP emits the draft plus a bonus token. The verifier
            # writes KV only through the draft token, but Python advances the
            # sequence by both emitted tokens before the next scheduled decode.
            required_blocks = (seq.num_tokens + 2 + self.block_size - 1) // self.block_size
            unsafe_bonus_boundary = (seq.num_tokens + 2) % self.block_size == 0
            can_verify = (
                first_draft is not None
                and self.mtp1_enabled
                and not batch.is_prefill
                and seq.temperature == 0
                and int(batch.query_lens[row]) == 1
                and seq.num_completion_tokens + 2 <= seq.max_tokens
                and len(seq.block_table) >= required_blocks
                and not unsafe_bonus_boundary
            )

            if not can_verify:
                if self.mtp1_enabled and not batch.is_prefill:
                    stats["fallback_steps"] += 1
                outputs[row] = target_token
                seed_normal_rows.append(row)
                seed_normal_tokens.append(target_token)
                continue

            draft_chains_by_row[row] = draft_tokens
            accepted = int(target_token) == int(first_draft)
            if accepted:
                stats["drafts_accepted"] += 1
                accepted_rows.append(row)
                accepted_drafts.append(int(first_draft))
            else:
                stats["drafts_rejected"] += 1
                outputs[row] = target_token
                seed_normal_rows.append(row)
                seed_normal_tokens.append(target_token)

            if getattr(self, "mtp_debug", False):
                draft_debug = debug_by_seq.pop(seq.seq_id, {})
                verify_logits = last_logits[row]
                debug_events.append(
                    {
                        **draft_debug,
                        "target_token": int(target_token),
                        "accepted": bool(accepted),
                        "draft_rank_in_main": self._token_rank(verify_logits, int(first_draft)),
                        "main_top": self._topk_debug(verify_logits),
                        "target_in_mtp_top5": int(target_token) in draft_debug.get("mtp_top", {}).get("ids", []),
                        "verifier_reused_decode": True,
                    }
                )

        emit_bonus = force_emit_bonus or os.environ.get("NANO_VLLM_JAX_MTP_EMIT_BONUS", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        if accepted_rows and not emit_bonus:
            for local_row, row in enumerate(accepted_rows):
                outputs[row] = accepted_drafts[local_row]
            accepted_rows = []

        if accepted_rows:
            draft_batch = self._compact_decode_batch(
                batch,
                accepted_rows,
                token_values=accepted_drafts,
                position_values=[int(batch.positions[row, 0]) + 1 for row in accepted_rows],
                seq_len_values=[int(batch.seq_lens[row]) + 1 for row in accepted_rows],
            )
            draft_hybrid_state = self._batch_hybrid_state(draft_batch)
            draft_output = self._step_fn(draft_batch)(
                draft_batch,
                cache_storage=self.cache_storage,
                hybrid_state=draft_hybrid_state,
                return_hidden=bool(seed_mtp1),
                return_hidden_with_logits=bool(seed_mtp1),
                last_logits_only=True,
            )
            self.cache_storage = draft_output.cache_storage
            self._store_batch_hybrid_state(draft_batch, draft_output.hybrid_state)
            self._refresh_kv_snapshot(draft_batch, draft_output.hybrid_state)

            bonus_hidden = None
            if seed_mtp1:
                bonus_hidden, bonus_logits_all = draft_output.activations
                bonus_logits = bonus_logits_all[:, 0]
            else:
                bonus_logits = draft_output.activations[:, 0]
            bonus_temperatures = jnp.array([seqs[row].temperature for row in accepted_rows], dtype=jnp.float32)
            bonus_token_ids = self._sample_fn(bonus_logits, bonus_temperatures)
            bonus_tokens = [int(token_id) for token_id in bonus_token_ids.tolist()]
            second_accept_rows: List[int] = []
            second_accept_drafts: List[int] = []
            enable_second_accept = False

            # Build second-token acceptance using the draft chains collected
            # during first-token verification.
            for local_row, row in enumerate(accepted_rows):
                seq = seqs[row]
                draft_chain = draft_chains_by_row.get(row, [])
                if (
                    enable_second_accept
                    and len(draft_chain) >= 2
                    and seq.num_completion_tokens + 3 <= seq.max_tokens
                    and int(bonus_tokens[local_row]) == int(draft_chain[1])
                ):
                    stats["drafts_accepted"] += 1
                    second_accept_rows.append(row)
                    second_accept_drafts.append(int(draft_chain[1]))
                else:
                    if len(draft_chain) >= 2:
                        stats["drafts_rejected"] += 1
                    outputs[row] = [accepted_drafts[local_row], bonus_tokens[local_row]]
                    stats["bonus_tokens"] += 1

            if second_accept_rows:
                second_batch = self._compact_decode_batch(
                    batch,
                    second_accept_rows,
                    token_values=second_accept_drafts,
                    position_values=[int(batch.positions[row, 0]) + 2 for row in second_accept_rows],
                    seq_len_values=[int(batch.seq_lens[row]) + 2 for row in second_accept_rows],
                )
                second_hybrid_state = self._batch_hybrid_state(second_batch)
                second_output = self._step_fn(second_batch)(
                    second_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=second_hybrid_state,
                    return_hidden=True,
                    last_logits_only=False,
                )
                self.cache_storage = second_output.cache_storage
                self._store_batch_hybrid_state(second_batch, second_output.hybrid_state)
                self._refresh_kv_snapshot(second_batch, second_output.hybrid_state)
                second_bonus_hidden = second_output.activations[:, 0]
                second_bonus_logits = self._logits_from_hidden(second_bonus_hidden[:, None, :])[:, 0]
                second_bonus_temps = jnp.array([seqs[row].temperature for row in second_accept_rows], dtype=jnp.float32)
                second_bonus_ids = self._sample_fn(second_bonus_logits, second_bonus_temps)
                second_bonus_tokens = [int(token_id) for token_id in second_bonus_ids.tolist()]
                accepted_draft_by_row = dict(zip(accepted_rows, accepted_drafts))
                for local_row, row in enumerate(second_accept_rows):
                    chain = draft_chains_by_row[row]
                    outputs[row] = [accepted_draft_by_row[row], chain[1], second_bonus_tokens[local_row]]
                    stats["bonus_tokens"] += 1
                if seed_mtp1:
                    self._seed_mtp1_drafts(
                        [seqs[row] for row in second_accept_rows],
                        self._hidden_for_mtp(second_bonus_hidden[:, None, :])[:, 0],
                        second_bonus_tokens,
                    )

            first_only_rows = [
                row
                for row in accepted_rows
                if row not in second_accept_rows
            ]
            if seed_mtp1 and first_only_rows:
                if bonus_hidden is not None:
                    first_only_indices = jnp.array([accepted_rows.index(row) for row in first_only_rows], dtype=jnp.int32)
                    bonus_seed_hidden = (
                        self._hidden_for_mtp(bonus_hidden)[:, 0]
                        if bonus_hidden.ndim == 3
                        else self._hidden_for_mtp(bonus_hidden[:, None, :])[:, 0]
                    )
                    self._seed_mtp1_drafts(
                        [seqs[row] for row in first_only_rows],
                        bonus_seed_hidden[first_only_indices],
                        [bonus_tokens[accepted_rows.index(row)] for row in first_only_rows],
                        positions=[
                            int(batch.positions[row, 0]) + 2
                            for row in first_only_rows
                        ],
                    )

        if seed_hidden is not None and seed_normal_rows:
            normal_row_index = jnp.array(seed_normal_rows, dtype=jnp.int32)
            self._seed_mtp1_drafts(
                [seqs[row] for row in seed_normal_rows],
                seed_hidden[normal_row_index],
                seed_normal_tokens,
            )

        resolved_outputs: List[int | List[int]] = []
        for output_token in outputs:
            if output_token is None:
                raise RuntimeError("MTP decode reuse path produced no output for a scheduled sequence")
            resolved_outputs.append(output_token)

        return resolved_outputs

    def _seed_mtp1_drafts(
        self,
        seqs: List[Sequence],
        hidden: jnp.ndarray,
        confirmed_token_ids: List[int],
        positions: List[int] | None = None,
    ):
        seed_rows: List[int] = []
        token_values: List[int] = []
        position_values: List[int] = []
        adaptive_gated = getattr(self, "_mtp_adaptive_gated", lambda: False)
        for row, seq in enumerate(seqs):
            confirmed_token_id = confirmed_token_ids[row]
            position = int(positions[row]) if positions is not None else seq.num_tokens
            if getattr(self, "mtp_token_source", "generated") == "current":
                confirmed_token_id = seq.last_token
                position = seq.num_tokens - 1
            if (
                not self.mtp1_enabled
                or seq.temperature != 0
                or not self._seq_mtp_admitted(seq)
                or adaptive_gated()
            ):
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_debug_state()[0].pop(seq.seq_id, None)
                continue
            if seq.num_completion_tokens + 1 >= seq.max_tokens:
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_debug_state()[0].pop(seq.seq_id, None)
                continue
            seed_rows.append(row)
            token_values.append(int(confirmed_token_id))
            position_values.append(int(position + int(getattr(self, "mtp_position_offset", 0))))

        if not seed_rows:
            return

        row_index = jnp.array(seed_rows, dtype=jnp.int32)
        hidden_input = hidden[row_index][:, None, :]
        token_input = jnp.array(token_values, dtype=jnp.int32)[:, None]
        position_input = jnp.array(position_values, dtype=jnp.int32)[:, None]
        draft_len = max(1, int(getattr(self, "num_speculative_tokens", 1) or 1))
        draft_margin_threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_DRAFT_MARGIN", "0") or "0")
        draft_margin_values = None
        if getattr(self, "mtp_debug", False):
            draft_logits = self._mtp1_logits(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
            )
            draft_tokens = jnp.argmax(draft_logits[:, 0], axis=-1).astype(jnp.int32)
            draft_chains = draft_tokens[:, None]
            if draft_margin_threshold > 0:
                draft_top2, _ = jax.lax.top_k(draft_logits[:, 0].astype(jnp.float32), 2)
                draft_margin_values = [float(value) for value in (draft_top2[:, 0] - draft_top2[:, 1]).tolist()]
        elif draft_margin_threshold > 0:
            draft_logits = None
            draft_chains, draft_margins = self._mtp1_draft_chain_with_margin(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
                draft_len=draft_len,
            )
            draft_margin_values = [float(value) for value in draft_margins.tolist()]
        else:
            draft_logits = None
            draft_chains = self._mtp1_draft_chain(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
                draft_len=draft_len,
            )

        draft_chain_list = [[int(token) for token in chain] for chain in draft_chains.tolist()]
        for local_row, row in enumerate(seed_rows):
            seq = seqs[row]
            if (
                draft_margin_values is not None
                and draft_margin_values[local_row] < draft_margin_threshold
            ):
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
                continue
            draft_chain = draft_chain_list[local_row]
            self._mtp1_drafts[seq.seq_id] = draft_chain if len(draft_chain) > 1 else draft_chain[0]
            self._mtp1_seeded_chain[seq.seq_id] = 0
            if getattr(self, "mtp_debug", False):
                draft_token = draft_chain[0]
                draft_vector = draft_logits[local_row, 0]
                draft_debug, _ = self._mtp1_debug_state()
                draft_debug[seq.seq_id] = {
                    "confirmed_token_id": int(token_values[local_row]),
                    "draft_token": draft_token,
                    "position": int(position_values[local_row] - int(getattr(self, "mtp_position_offset", 0))),
                    "position_offset": int(getattr(self, "mtp_position_offset", 0)),
                    "token_source": str(getattr(self, "mtp_token_source", "generated")),
                    "hidden_source": str(getattr(self, "mtp_hidden_source", "final_normed")),
                    "mtp_top": self._topk_debug(draft_vector),
                }
            self._speculative_stats()["drafts_proposed"] += len(draft_chain)

    def _seed_mtp1_draft(
        self,
        seq: Sequence,
        hidden: jnp.ndarray,
        confirmed_token_id: int,
        position: int,
    ):
        adaptive_gated = getattr(self, "_mtp_adaptive_gated", lambda: False)
        if (
            not self.mtp1_enabled
            or seq.temperature != 0
            or not self._seq_mtp_admitted(seq)
            or adaptive_gated()
        ):
            self._mtp1_drafts.pop(seq.seq_id, None)
            self._mtp1_debug_state()[0].pop(seq.seq_id, None)
            return
        if seq.num_completion_tokens + 1 >= seq.max_tokens:
            self._mtp1_drafts.pop(seq.seq_id, None)
            self._mtp1_debug_state()[0].pop(seq.seq_id, None)
            return

        hidden_input = hidden[None, None, :]
        token_input = jnp.array([[confirmed_token_id]], dtype=jnp.int32)
        position_input = jnp.array([[position + int(getattr(self, "mtp_position_offset", 0))]], dtype=jnp.int32)
        draft_margin_threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_DRAFT_MARGIN", "0") or "0")
        draft_margin = None
        if getattr(self, "mtp_debug", False):
            draft_logits = self._mtp1_logits(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
            )
            draft_vector = draft_logits[0, 0]
            draft_token = int(jnp.argmax(draft_vector))
            if draft_margin_threshold > 0:
                top2, _ = jax.lax.top_k(draft_vector.astype(jnp.float32), 2)
                draft_margin = float(top2[0] - top2[1])
        elif draft_margin_threshold > 0:
            draft_chain, draft_margins = self._mtp1_draft_chain_with_margin(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
                draft_len=1,
            )
            draft_token = int(draft_chain[0, 0])
            draft_margin = float(draft_margins[0])
        else:
            draft_token = int(self._mtp1_draft_token(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
            )[0])
        if draft_margin is not None and draft_margin < draft_margin_threshold:
            self._mtp1_drafts.pop(seq.seq_id, None)
            self._mtp1_seeded_chain.pop(seq.seq_id, None)
            return
        self._mtp1_drafts[seq.seq_id] = draft_token
        self._mtp1_seeded_chain[seq.seq_id] = 0
        if getattr(self, "mtp_debug", False):
            draft_debug, _ = self._mtp1_debug_state()
            draft_debug[seq.seq_id] = {
                "confirmed_token_id": int(confirmed_token_id),
                "draft_token": draft_token,
                "position": int(position),
                "position_offset": int(getattr(self, "mtp_position_offset", 0)),
                "token_source": str(getattr(self, "mtp_token_source", "generated")),
                "hidden_source": str(getattr(self, "mtp_hidden_source", "final_normed")),
                "mtp_top": self._topk_debug(draft_vector),
            }
        self._speculative_stats()["drafts_proposed"] += 1

    def _can_run_mtp1(self, seqs: List[Sequence], batch: ScheduledBatch) -> bool:
        if not self.mtp1_enabled or batch.is_prefill or len(seqs) != 1:
            if self.mtp1_enabled and not batch.is_prefill:
                self._speculative_stats()["fallback_steps"] += 1
            return False
        seq = seqs[0]
        if (
            seq.temperature != 0
            or not self._seq_mtp_admitted(seq)
            or seq.seq_id not in self._mtp1_drafts
        ):
            self._speculative_stats()["fallback_steps"] += 1
            return False
        if seq.num_completion_tokens + 2 > seq.max_tokens:
            self._speculative_stats()["fallback_steps"] += 1
            return False
        if int(batch.query_lens[0]) != 1:
            self._speculative_stats()["fallback_steps"] += 1
            return False
        # MTP K=1 can emit [draft, bonus]. Even though the verifier only writes
        # KV through the draft token, Sequence.num_tokens advances by both
        # emitted tokens before the next scheduled decode.
        required_blocks = (seq.num_tokens + 2 + self.block_size - 1) // self.block_size
        can_run = len(seq.block_table) >= required_blocks
        if not can_run:
            self._speculative_stats()["fallback_steps"] += 1
        return can_run

    def _can_run_mtp1_for_row(self, seq: Sequence, batch: ScheduledBatch, row: int) -> bool:
        if not self.mtp1_enabled or batch.is_prefill:
            if self.mtp1_enabled and not batch.is_prefill:
                self._speculative_stats()["fallback_steps"] += 1
            return False

        if (
            seq.temperature != 0
            or not self._seq_mtp_admitted(seq)
            or seq.seq_id not in self._mtp1_drafts
        ):
            self._speculative_stats()["fallback_steps"] += 1
            return False

        if seq.num_completion_tokens + 2 > seq.max_tokens:
            self._speculative_stats()["fallback_steps"] += 1
            return False

        if int(batch.query_lens[row]) != 1:
            self._speculative_stats()["fallback_steps"] += 1
            return False

        # MTP K=1 can emit [draft, bonus]. Even though the verifier only writes
        # KV through the draft token, Sequence.num_tokens advances by both
        # emitted tokens before the next scheduled decode.
        required_blocks = (seq.num_tokens + 2 + self.block_size - 1) // self.block_size
        can_run = len(seq.block_table) >= required_blocks
        if not can_run:
            self._speculative_stats()["fallback_steps"] += 1
        return can_run

    def _mtp1_verification_batch(self, seq: Sequence, batch: ScheduledBatch, draft_token: int) -> ScheduledBatch:
        return ScheduledBatch(
            tokens=jnp.array([[seq.last_token, draft_token]], dtype=jnp.int32),
            positions=jnp.array([[seq.num_tokens - 1, seq.num_tokens]], dtype=jnp.int32),
            seq_ids=jnp.array([seq.seq_id], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 2], dtype=jnp.int32),
            is_prefill=True,
            num_prefill_tokens=2,
            num_decode_tokens=0,
            block_tables=batch.block_tables[:1],
            seq_lens=jnp.array([seq.num_tokens + 1], dtype=jnp.int32),
        )

    def _mtp1_verification_batch_for_rows(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        rows: List[int],
        draft_tokens: List[int],
    ) -> ScheduledBatch:
        tokens = []
        positions = []
        seq_ids = []
        seq_lens = []
        block_tables = []
        query_start_loc = [0]
        for seq, row, draft_token in zip(seqs, rows, draft_tokens):
            tokens.append([seq.last_token, draft_token])
            positions.append([seq.num_tokens - 1, seq.num_tokens])
            seq_ids.append(seq.seq_id)
            seq_lens.append(seq.num_tokens + 1)
            block_tables.append(batch.block_tables[row].tolist())
            query_start_loc.append(query_start_loc[-1] + 2)

        return ScheduledBatch(
            tokens=jnp.array(tokens, dtype=jnp.int32),
            positions=jnp.array(positions, dtype=jnp.int32),
            seq_ids=jnp.array(seq_ids, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=True,
            num_prefill_tokens=2 * len(rows),
            num_decode_tokens=0,
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
        )

    def _run_mtp1_batched(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        rows: List[int],
        forced_reject_rows: set[int] | None = None,
    ) -> dict[int, List[int] | int] | None:
        if not rows:
            return {}
        forced_reject_rows = set(forced_reject_rows or ())
        for row in rows:
            if not bool(getattr(seqs[row], "mtp_admitted", True)):
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
                return None
        if getattr(self, "_mtp_adaptive_gated", lambda: False)():
            for row in rows:
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
            return None
        # K=1 MTP may run one target-model token ahead of the canonical
        # decode stream. Avoid doing that on the same step that starts from a
        # just-completed KV block; the ordinary decode path will process the
        # boundary token, refresh block metadata, and MTP can resume next step.
        relax_start_boundary = (
            os.environ.get("NANO_VLLM_JAX_MTP_RELAX_START_BOUNDARY", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        if (
            not relax_start_boundary
            and any(seqs[row].num_tokens % self.block_size == 0 for row in rows)
        ):
            return None
        if os.environ.get("NANO_VLLM_JAX_MTP_FUSED_VERIFY", "0") not in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            return None
        profile_mtp = os.environ.get("NANO_VLLM_JAX_PROFILE_MTP_RUN", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }

        def _ready(value):
            if not profile_mtp:
                return
            for leaf in jax.tree_util.tree_leaves(value):
                ready = getattr(leaf, "block_until_ready", None)
                if ready is not None:
                    ready()

        def _mark(label: str, start: float) -> float:
            if profile_mtp:
                now = time.perf_counter()
                print(f"[MTP_RUN] {label}={(now - start) * 1000:.3f}ms", flush=True)
                return now
            return start

        t_profile = time.perf_counter()
        use_debug = getattr(self, "mtp_debug", False)
        force_scalar_mtp = os.environ.get("NANO_VLLM_JAX_MTP_FORCE_SCALAR", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        physical_batch_size = int(batch.tokens.shape[0])
        partial_physical_batch = rows != list(range(physical_batch_size))
        debug_verifier_enabled = any(
            os.environ.get(name, "0") in {"1", "true", "yes", "on", "True"}
            for name in (
                "NANO_VLLM_JAX_MTP_PARITY_DEBUG",
                "NANO_VLLM_JAX_MTP_LAYER_PARITY_DEBUG",
                "NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_DEBUG",
            )
        )
        use_fused_step = (
            not use_debug
            and not force_scalar_mtp
            and getattr(self, "execution", "eager") in {"decode-jit", "jit"}
            and hasattr(self.executor, "mtp1_commit_select_greedy_step_jit")
            and all(int(batch.query_lens[row]) == 1 for row in rows)
        )
        if not use_fused_step:
            if int(batch.tokens.shape[0]) != 1 or rows != [0]:
                return None
            for row in rows:
                draft_value = self._mtp1_drafts.get(seqs[row].seq_id)
                if isinstance(draft_value, list) and len(draft_value) > 1:
                    return None
            return {
                row: self._run_mtp1([seqs[row]], self._slice_batch(batch, row))[0]
                for row in rows
            }

        mtp_seqs = [seqs[row] for row in rows]
        draft_chains: List[List[int]] = []
        for row, seq in zip(rows, mtp_seqs):
            if row in forced_reject_rows:
                draft_chains.append([-1])
                continue
            draft_value = self._mtp1_drafts[seq.seq_id]
            if isinstance(draft_value, list):
                draft_chains.append([int(token) for token in draft_value])
            else:
                draft_chains.append([int(draft_value)])
        draft_len = min(len(chain) for chain in draft_chains)
        draft_len = min(draft_len, max(1, int(getattr(self, "num_speculative_tokens", 1) or 1)))
        if draft_len < 1:
            return {}
        relax_start_boundary = (
            os.environ.get("NANO_VLLM_JAX_MTP_RELAX_START_BOUNDARY", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        if (
            not relax_start_boundary
            and any((seqs[row].num_tokens + draft_len + 1) % self.block_size == 0 for row in rows)
        ):
            return None
        draft_token_chains = [chain[:draft_len] for chain in draft_chains]
        draft_tokens = [chain[0] for chain in draft_token_chains]
        compact_verifier_enabled = (
            os.environ.get("NANO_VLLM_JAX_MTP_COMPACT_VERIFIER", "1")
            in {"1", "true", "yes", "on", "True"}
        )
        use_compact_verifier = (
            partial_physical_batch
            and not use_debug
            and not debug_verifier_enabled
            and compact_verifier_enabled
            and (
                draft_len == 1
                or (
                    draft_len == 2
                    and hasattr(self.executor, "mtp2_commit_select_greedy_step_jit")
                )
            )
        )
        decode_batch = (
            self._compact_decode_batch(batch, rows)
            if use_compact_verifier
            else self._masked_decode_batch(batch, rows)
        )
        verifier_physical_batch_size = int(decode_batch.tokens.shape[0])
        verifier_index_for_local = (
            list(range(len(rows))) if use_compact_verifier else list(rows)
        )
        draft_token_chains_for_batch = [
            [0 for _ in range(draft_len)]
            for _ in range(verifier_physical_batch_size)
        ]
        for local_row, row in enumerate(rows):
            draft_token_chains_for_batch[verifier_index_for_local[local_row]] = draft_token_chains[local_row]
        verifier_draft_tokens = draft_tokens
        if os.environ.get("NANO_VLLM_JAX_MTP_FORCE_REJECT", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            verifier_draft_tokens = [-1 for _ in draft_tokens]
        hybrid_state = self._batch_hybrid_state(decode_batch)
        _ready(hybrid_state)
        t_profile = _mark("batch_hybrid_state", t_profile)
        verifier_draft_tokens_for_batch = [0 for _ in range(verifier_physical_batch_size)]
        next_mtp_positions_for_batch = [0 for _ in range(verifier_physical_batch_size)]
        for local_row, row in enumerate(rows):
            verifier_idx = verifier_index_for_local[local_row]
            verifier_draft_tokens_for_batch[verifier_idx] = verifier_draft_tokens[local_row]
            next_mtp_positions_for_batch[verifier_idx] = (
                mtp_seqs[local_row].num_tokens
                + draft_len
                + int(getattr(self, "mtp_position_offset", 0))
            )
        next_mtp_positions = jnp.array(next_mtp_positions_for_batch, dtype=jnp.int32)
        force_commit_select = (
            os.environ.get("NANO_VLLM_JAX_MTP_COMMIT_SELECT", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        disable_one_pass_k1 = (
            os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        enable_one_pass_k1 = (
            os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_ONE_PASS_K1", "1")
            in {"1", "true", "yes", "on", "True"}
        )
        allow_unsafe_one_pass_k1 = (
            os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        allow_mixed_fused_k1 = (
            os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        seed_after_bonus_enabled = (
            os.environ.get("NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        allow_seeded_one_pass_k1 = (
            os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        block_seeded_one_pass_k1 = seed_after_bonus_enabled and not allow_seeded_one_pass_k1
        enable_fast_all_accept = (
            (
                allow_mixed_fused_k1
                or os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT", "0")
                in {"1", "true", "yes", "on", "True"}
            )
            and not force_commit_select
            and os.environ.get("NANO_VLLM_JAX_MTP_PREFIX_SAFE", "0")
            not in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_two_decode_greedy_fast_step_jit")
        )
        batch_accept_policy = os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none")
        enable_rowwise_repair = (
            os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_ROWWISE_REPAIR", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        use_one_pass_k1 = (
            draft_len == 1
            and not enable_fast_all_accept
            and not force_commit_select
            and not disable_one_pass_k1
            and not block_seeded_one_pass_k1
            and allow_unsafe_one_pass_k1
            and (enable_one_pass_k1 or allow_mixed_fused_k1 or partial_physical_batch)
            and hasattr(self.executor, "mtp1_two_decode_greedy_step_jit")
        )
        use_commit_select = (
            draft_len == 1
            and not use_one_pass_k1
            and not enable_fast_all_accept
            and hasattr(self.executor, "mtp1_commit_select_greedy_step_jit")
        )
        verifier_full_physical_batch = (not partial_physical_batch) or use_compact_verifier
        use_mtp2_commit_select = (
            draft_len == 2
            and verifier_full_physical_batch
            and hasattr(self.executor, "mtp2_commit_select_greedy_step_jit")
        )
        use_fast_all_accept = (
            draft_len == 1
            and not use_one_pass_k1
            and not use_commit_select
            and enable_fast_all_accept
        )
        enable_rowwise_repair = enable_rowwise_repair or (
            batch_accept_policy == "rowwise" and use_fast_all_accept
        )
        use_prefix_two_decode = (
            draft_len == 1
            and not use_one_pass_k1
            and not use_commit_select
            and os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_PREFIX_TWO_DECODE", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_two_decode_greedy_step_jit")
        )
        parity_debug_one_pass = (
            draft_len == 1
            and use_one_pass_k1
            and os.environ.get("NANO_VLLM_JAX_MTP_PARITY_DEBUG", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_commit_select_greedy_step_jit")
        )
        layer_parity_debug_one_pass = (
            draft_len == 1
            and use_one_pass_k1
            and os.environ.get("NANO_VLLM_JAX_MTP_LAYER_PARITY_DEBUG", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_layer_parity_debug_jit")
        )
        layerwise_drift_debug_one_pass = (
            draft_len == 1
            and use_one_pass_k1
            and os.environ.get("NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_DEBUG", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_layerwise_drift_debug_jit")
        )
        if profile_mtp:
            if use_one_pass_k1:
                verifier_mode = "mtp1_one_pass_prefix"
            elif use_commit_select:
                verifier_mode = "mtp1_commit_select"
            elif use_fast_all_accept:
                verifier_mode = "mtp1_two_decode_fast"
            elif draft_len == 1 and use_prefix_two_decode:
                verifier_mode = "mtp1_two_decode"
            elif use_mtp2_commit_select:
                verifier_mode = "mtp2_commit_select"
            elif draft_len == 1:
                verifier_mode = "fallback_k1_no_verifier"
            elif partial_physical_batch and not use_compact_verifier:
                verifier_mode = "fallback_k_gt1_partial_physical"
            else:
                verifier_mode = "mtp_k_decode"
            print(
                "[MTP_RUN] verifier "
                f"mode={verifier_mode} draft_len={draft_len} "
                f"rows={rows} physical_batch={physical_batch_size} "
                f"verifier_batch={verifier_physical_batch_size} "
                f"partial_physical={partial_physical_batch} "
                f"compact={use_compact_verifier}",
                flush=True,
            )
        parity_output = None
        layer_parity_output = None
        layerwise_drift_output = None
        if use_one_pass_k1:
            if layerwise_drift_debug_one_pass:
                layerwise_drift_output = self.executor.mtp1_layerwise_drift_debug_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                )
                _ready(layerwise_drift_output)
                t_profile = _mark("executor_mtp1_layerwise_drift_debug", t_profile)
            if layer_parity_debug_one_pass:
                layer_parity_output = self.executor.mtp1_layer_parity_debug_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                    next_mtp_position=next_mtp_positions,
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                _ready(layer_parity_output)
                t_profile = _mark("executor_mtp1_layer_parity_debug", t_profile)
            output = self.executor.mtp1_two_decode_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                next_mtp_position=next_mtp_positions,
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_one_pass_prefix", t_profile)
            if parity_debug_one_pass:
                parity_output = self.executor.mtp1_commit_select_greedy_step_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                    next_mtp_position=next_mtp_positions,
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                _ready(parity_output)
                t_profile = _mark("executor_mtp1_parity_commit_select", t_profile)
        elif use_commit_select:
            output = self.executor.mtp1_commit_select_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                next_mtp_position=next_mtp_positions,
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_commit_select", t_profile)
        elif use_fast_all_accept:
            output = self.executor.mtp1_two_decode_greedy_fast_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                next_mtp_position=next_mtp_positions,
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_two_decode_fast", t_profile)
            accepted_all = bool(jnp.all(output.accepted).item())
        elif draft_len == 1 and use_prefix_two_decode:
            output = self.executor.mtp1_two_decode_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=jnp.array(verifier_draft_tokens_for_batch, dtype=jnp.int32),
                next_mtp_position=next_mtp_positions,
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_two_decode", t_profile)
        elif draft_len == 1:
            return None
        elif use_mtp2_commit_select:
            output = self.executor.mtp2_commit_select_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=jnp.array(draft_token_chains_for_batch, dtype=jnp.int32),
                next_mtp_position=next_mtp_positions,
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp2_commit_select", t_profile)
            accepted_all = bool(jnp.all(output.accepted).item())
        else:
            if partial_physical_batch:
                return None
            output = self.executor.mtp_k_decode_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=jnp.array(draft_token_chains_for_batch, dtype=jnp.int32),
                next_mtp_position=next_mtp_positions,
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp_k_decode", t_profile)
            accepted_all = bool(jnp.all(output.accepted).item())
        t_profile = _mark("host_result_transfer", t_profile)

        if draft_len > 1:
            output_acceptance_matrix = output.accepted.tolist()
            accepted_all = all(
                all(
                    bool(value)
                    for value in output_acceptance_matrix[
                        verifier_index_for_local[local_row]
                    ]
                )
                for local_row, _row in enumerate(rows)
            )
        else:
            output_acceptance_for_rows = [bool(x) for x in output.accepted.tolist()]
            accepted_all = all(
                output_acceptance_for_rows[verifier_index_for_local[local_row]]
                for local_row, _row in enumerate(rows)
            )
        if (
            enable_rowwise_repair
            and (use_fast_all_accept or use_one_pass_k1)
            and draft_len == 1
            and not accepted_all
        ):
            output_acceptance = [bool(x) for x in output.accepted.tolist()]
            accepted_flags_local = [
                output_acceptance[verifier_index_for_local[local_row]]
                for local_row, row in enumerate(rows)
            ]
            commit_rejected_directly = (
                use_one_pass_k1
                and os.environ.get("NANO_VLLM_JAX_MTP_K1_COMMIT_REJECTED", "0")
                in {"1", "true", "yes", "on", "True"}
            )

            committed_batch = replace(decode_batch, seq_lens=output.committed_seq_lens)
            self.cache_storage = output.cache_storage
            self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
            _ready(self._hybrid_state_table)
            self._record_kv_snapshot(committed_batch, output.hybrid_state)

            target_values = [int(value) for value in output.target_token.tolist()]
            bonus_values = [int(value) for value in output.bonus_token.tolist()]
            next_draft_values = [int(value) for value in output.next_draft_token.tolist()]
            outputs: dict[int, List[int] | int] = {}
            repair_rows: List[int] = []
            stats = self._speculative_stats()
            for local_row, row in enumerate(rows):
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                if accepted_flags_local[local_row]:
                    idx = verifier_index_for_local[local_row]
                    stats["drafts_accepted"] += 1
                    stats["bonus_tokens"] += 1
                    emitted_len = 2
                    outputs[row] = draft_token_chains[local_row] + [bonus_values[idx]]
                    if (
                        self.mtp1_enabled
                        and seq.temperature == 0
                        and seq.num_completion_tokens + emitted_len < seq.max_tokens
                        and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                    ):
                        self._mtp1_drafts[seq.seq_id] = next_draft_values[idx]
                        stats["drafts_proposed"] += 1
                else:
                    stats["drafts_rejected"] += 1
                    if commit_rejected_directly:
                        idx = verifier_index_for_local[local_row]
                        outputs[row] = target_values[idx]
                    else:
                        repair_rows.append(row)

            if repair_rows:
                repair_batch = self._masked_decode_batch(batch, repair_rows)
                repair_outputs = self._run_main_and_sample(
                    seqs,
                    repair_batch,
                    seed_mtp1=False,
                )
                for row in repair_rows:
                    outputs[row] = repair_outputs[row]
            return outputs

        if (
            batch_accept_policy == "rowwise"
            and not enable_rowwise_repair
            and not accepted_all
            and not (use_one_pass_k1 or use_commit_select or use_mtp2_commit_select)
        ):
            return None

        if not accepted_all and not (use_one_pass_k1 or use_commit_select) and draft_len == 1:
            # Correctness first: the verifier may have physically written KV /
            # hybrid state for rejected draft slots. Do not install those side
            # effects. Let the canonical main-model reuse path verify the
            # stored drafts against an ordinary decode from the original state.
            return None

        if draft_len > 1 and not use_mtp2_commit_select and not use_commit_select and not accepted_all:
            accepted_matrix = [
                [bool(value) for value in row_acceptance]
                for row_acceptance in output.accepted.tolist()
            ]
            self._record_draft_position_acceptance(accepted_matrix)
            full_accept_locals = [
                local_row
                for local_row, row_acceptance in enumerate(accepted_matrix)
                if all(row_acceptance)
            ]
            repair_locals = [
                local_row
                for local_row, row_acceptance in enumerate(accepted_matrix)
                if not all(row_acceptance)
            ]

            # Physical verifier writes for repair rows are safe: those slots
            # remain logically unreachable and K=1 repair rewrites the current
            # token / first draft before they can be read.
            self.cache_storage = output.cache_storage

            outputs: dict[int, List[int] | int] = {}
            stats = self._speculative_stats()
            if full_accept_locals:
                full_accept_rows = [rows[local_row] for local_row in full_accept_locals]
                full_accept_idx = jnp.array(full_accept_locals, dtype=jnp.int32)
                accepted_batch = ScheduledBatch(
                    tokens=decode_batch.tokens[full_accept_idx],
                    positions=decode_batch.positions[full_accept_idx],
                    seq_ids=decode_batch.seq_ids[full_accept_idx],
                    query_start_loc=jnp.arange(len(full_accept_locals) + 1, dtype=jnp.int32),
                    is_prefill=False,
                    num_prefill_tokens=0,
                    num_decode_tokens=len(full_accept_locals),
                    block_tables=decode_batch.block_tables[full_accept_idx],
                    seq_lens=decode_batch.seq_lens[full_accept_idx] + draft_len,
                )
                accepted_hybrid = HybridLayerState(
                    conv_state=output.hybrid_state.conv_state[full_accept_idx]
                    if output.hybrid_state.conv_state is not None
                    else None,
                    recurrent_state=output.hybrid_state.recurrent_state[full_accept_idx]
                    if output.hybrid_state.recurrent_state is not None
                    else None,
                )
                self._store_batch_hybrid_state(accepted_batch, accepted_hybrid)
                self._record_kv_snapshot(accepted_batch, accepted_hybrid)

                bonus_values = [int(value) for value in output.bonus_token.tolist()]
                next_draft_values = [
                    [int(token) for token in chain]
                    for chain in output.next_draft_token.tolist()
                ]
                for local_row, row in zip(full_accept_locals, full_accept_rows):
                    seq = seqs[row]
                    self._mtp1_drafts.pop(seq.seq_id, None)
                    stats["drafts_accepted"] += draft_len
                    stats["bonus_tokens"] += 1
                    emitted_len = draft_len + 1
                    outputs[row] = draft_token_chains[local_row] + [bonus_values[local_row]]
                    if (
                        self.mtp1_enabled
                        and seq.temperature == 0
                        and seq.num_completion_tokens + emitted_len < seq.max_tokens
                        and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                    ):
                        next_chain = next_draft_values[local_row]
                        self._mtp1_drafts[seq.seq_id] = next_chain if len(next_chain) > 1 else next_chain[0]
                        stats["drafts_proposed"] += len(next_chain)

            if repair_locals:
                repair_rows = [rows[local_row] for local_row in repair_locals]
                for local_row, row in zip(repair_locals, repair_rows):
                    seq = seqs[row]
                    self._mtp1_drafts[seq.seq_id] = draft_token_chains[local_row][0]
                repair_outputs = self._run_mtp1_batched(seqs, batch, repair_rows)
                if repair_outputs is None:
                    return None
                outputs.update(repair_outputs)
            return outputs

        self.cache_storage = output.cache_storage
        committed_batch = decode_batch
        if getattr(output, "committed_seq_lens", None) is not None:
            committed_batch = replace(decode_batch, seq_lens=output.committed_seq_lens)
        self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
        _ready(self._hybrid_state_table)
        t_profile = _mark("store_hybrid_state", t_profile)
        self._record_kv_snapshot(committed_batch, output.hybrid_state)
        t_profile = _mark("record_kv_snapshot", t_profile)
        target_token_values = output.target_token.tolist()
        if draft_len > 1:
            target_token_rows = [[int(token) for token in row] for row in target_token_values]
            target_tokens = [row[0] for row in target_token_rows]
        else:
            target_token_rows = [[int(x)] for x in target_token_values]
            target_tokens = [row[0] for row in target_token_rows]
        bonus_tokens = [int(x) for x in output.bonus_token.tolist()]
        if draft_len > 1:
            accepted_matrix = [
                [bool(value) for value in row_acceptance]
                for row_acceptance in output.accepted.tolist()
            ]
            prefix_lengths = []
            for row_acceptance in accepted_matrix:
                prefix_len = 0
                for value in row_acceptance:
                    if not value:
                        break
                    prefix_len += 1
                prefix_lengths.append(prefix_len)
            accepted_flags = [prefix_len == draft_len for prefix_len in prefix_lengths]
            self._record_draft_position_acceptance(
                accepted_matrix
            )
        else:
            accepted_flags = [bool(x) for x in output.accepted.tolist()]
            prefix_lengths = [1 if accepted else 0 for accepted in accepted_flags]
        if draft_len == 1:
            next_draft_chains = [[int(x)] for x in output.next_draft_token.tolist()]
        else:
            next_draft_chains = [[int(token) for token in chain] for chain in output.next_draft_token.tolist()]
        if layerwise_drift_output is not None:
            threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_THRESHOLD", "0.1"))
            hidden_vals = [float(x) for x in layerwise_drift_output.hidden_max_abs.tolist()]
            k_vals = [float(x) for x in layerwise_drift_output.k_slot_max_abs.tolist()]
            v_vals = [float(x) for x in layerwise_drift_output.v_slot_max_abs.tolist()]
            conv_vals = [float(x) for x in layerwise_drift_output.conv_state_max_abs.tolist()]
            rec_vals = [float(x) for x in layerwise_drift_output.recurrent_state_max_abs.tolist()]
            pre_k_vals = [float(x) for x in layerwise_drift_output.k_prewrite_max_abs.tolist()]
            pre_v_vals = [float(x) for x in layerwise_drift_output.v_prewrite_max_abs.tolist()]
            stage_vals = [
                [float(v) for v in row]
                for row in layerwise_drift_output.block_stage_max_abs.tolist()
            ]
            stage_names = ["entry", "in_norm", "attn", "attn_resid", "ffn_norm", "mlp", "out"]
            layer_rows = []
            first_idx = None
            for layer_idx, layer_type in enumerate(self.config.layer_types):
                score = max(
                    hidden_vals[layer_idx],
                    k_vals[layer_idx],
                    v_vals[layer_idx],
                    conv_vals[layer_idx],
                    rec_vals[layer_idx],
                    pre_k_vals[layer_idx],
                    pre_v_vals[layer_idx],
                    max(stage_vals[layer_idx]),
                )
                if first_idx is None and score > threshold:
                    first_idx = layer_idx
                limit_idx = 3 if first_idx is None else first_idx + 1
                if layer_idx <= limit_idx:
                    layer_rows.append(
                        f"{layer_idx}:{layer_type}:h={hidden_vals[layer_idx]:.6g},"
                        f"k={k_vals[layer_idx]:.6g},v={v_vals[layer_idx]:.6g},"
                        f"pre_k={pre_k_vals[layer_idx]:.6g},pre_v={pre_v_vals[layer_idx]:.6g},"
                        f"conv={conv_vals[layer_idx]:.6g},rec={rec_vals[layer_idx]:.6g},"
                        f"stages="
                        f"{','.join(f'{name}={stage_vals[layer_idx][idx]:.6g}' for idx, name in enumerate(stage_names))}"
                    )
            if first_idx is None:
                first_layer = "none"
                first_type = "none"
            else:
                first_layer = str(first_idx)
                first_type = self.config.layer_types[first_idx]
            print(
                "[MTP_LAYERWISE_DRIFT] fused_one_pass_vs_seq "
                f"threshold={threshold:.6g} "
                f"first_layer={first_layer} first_type={first_type} "
                f"layers={';'.join(layer_rows)}",
                flush=True,
            )
        if layer_parity_output is not None:
            print(
                "[MTP_LAYER_PARITY] fused_one_pass_vs_seq "
                f"slot0_logit_max_abs={float(layer_parity_output.slot0_logit_max_abs.item()):.6g} "
                f"slot1_logit_max_abs={float(layer_parity_output.slot1_logit_max_abs.item()):.6g} "
                f"slot0_hidden_max_abs={float(layer_parity_output.slot0_hidden_max_abs.item()):.6g} "
                f"slot1_hidden_max_abs={float(layer_parity_output.slot1_hidden_max_abs.item()):.6g} "
                f"current_k_slot_max_abs={float(layer_parity_output.current_k_slot_max_abs.item()):.6g} "
                f"draft_k_slot_max_abs={float(layer_parity_output.draft_k_slot_max_abs.item()):.6g} "
                f"current_v_slot_max_abs={float(layer_parity_output.current_v_slot_max_abs.item()):.6g} "
                f"draft_v_slot_max_abs={float(layer_parity_output.draft_v_slot_max_abs.item()):.6g} "
                f"conv_state_max_abs={float(layer_parity_output.conv_state_max_abs.item()):.6g} "
                f"recurrent_state_max_abs={float(layer_parity_output.recurrent_state_max_abs.item()):.6g} "
                f"fused_target={layer_parity_output.fused_target_token.tolist()} "
                f"seq_target={layer_parity_output.seq_target_token.tolist()} "
                f"fused_bonus={layer_parity_output.fused_bonus_token.tolist()} "
                f"seq_bonus={layer_parity_output.seq_bonus_token.tolist()} "
                f"fused_top5_slot0={layer_parity_output.fused_top5_slot0.tolist()} "
                f"seq_top5_slot0={layer_parity_output.seq_top5_slot0.tolist()} "
                f"fused_top5_slot1={layer_parity_output.fused_top5_slot1.tolist()} "
                f"seq_top5_slot1={layer_parity_output.seq_top5_slot1.tolist()}",
                flush=True,
            )
        if parity_output is not None:
            parity_targets = [int(x) for x in parity_output.target_token.tolist()]
            parity_bonus = [int(x) for x in parity_output.bonus_token.tolist()]
            parity_accepted = [bool(x) for x in parity_output.accepted.tolist()]
            parity_next = [int(x) for x in parity_output.next_draft_token.tolist()]
            state_threshold = float(
                os.environ.get("NANO_VLLM_JAX_MTP_PARITY_STATE_THRESHOLD", "0")
            )

            def _state_max_abs(left, right) -> float:
                if left is None or right is None:
                    return 0.0
                return float(
                    jnp.max(
                        jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
                    ).item()
                )

            def _slot_max_abs(left, right, slots) -> float:
                leading_shape = left.shape[:-4] if left.ndim == 5 else left.shape[:-3]
                flat_left = left.reshape(leading_shape + (-1,) + left.shape[-2:])
                flat_right = right.reshape(leading_shape + (-1,) + right.shape[-2:])
                left_values = flat_left[..., slots, :, :].astype(jnp.float32)
                right_values = flat_right[..., slots, :, :].astype(jnp.float32)
                return float(jnp.max(jnp.abs(left_values - right_values)).item())

            slot_current = compute_slot_mapping(
                positions=decode_batch.positions,
                block_table=decode_batch.block_tables,
                block_size=self.config.block_size,
                is_prefill=False,
            )[:, 0]
            slot_draft = compute_slot_mapping(
                positions=decode_batch.positions + 1,
                block_table=decode_batch.block_tables,
                block_size=self.config.block_size,
                is_prefill=False,
            )[:, 0]
            parity_slots = jnp.stack([slot_current, slot_draft], axis=1).reshape(-1)
            k_slot_diff = _slot_max_abs(
                output.cache_storage.k_cache,
                parity_output.cache_storage.k_cache,
                parity_slots,
            )
            v_slot_diff = _slot_max_abs(
                output.cache_storage.v_cache,
                parity_output.cache_storage.v_cache,
                parity_slots,
            )
            conv_diff = _state_max_abs(
                output.hybrid_state.conv_state,
                parity_output.hybrid_state.conv_state,
            )
            recurrent_diff = _state_max_abs(
                output.hybrid_state.recurrent_state,
                parity_output.hybrid_state.recurrent_state,
            )
            state_diff = max(k_slot_diff, v_slot_diff, conv_diff, recurrent_diff)
            if state_diff > state_threshold:
                print(
                    "[MTP_PARITY_STATE] one_pass_vs_commit_select "
                    f"k_slot_max_abs={k_slot_diff:.6g} "
                    f"v_slot_max_abs={v_slot_diff:.6g} "
                    f"conv_max_abs={conv_diff:.6g} "
                    f"recurrent_max_abs={recurrent_diff:.6g}",
                    flush=True,
                )
                if os.environ.get("NANO_VLLM_JAX_MTP_PARITY_STOP_STATE", "0") in {
                    "1",
                    "true",
                    "yes",
                    "on",
                    "True",
                }:
                    raise RuntimeError("MTP one-pass parity state mismatch")
            for local_row, row in enumerate(rows):
                idx = verifier_index_for_local[local_row]
                mismatch = (
                    target_tokens[idx] != parity_targets[idx]
                    or accepted_flags[idx] != parity_accepted[idx]
                    or (
                        accepted_flags[idx]
                        and parity_accepted[idx]
                        and bonus_tokens[idx] != parity_bonus[idx]
                    )
                    or (
                        accepted_flags[idx] == parity_accepted[idx]
                        and next_draft_chains[idx][0] != parity_next[idx]
                    )
                )
                if mismatch:
                    seq = seqs[row]
                    print(
                        "[MTP_PARITY] one_pass_vs_commit_select "
                        f"seq_id={seq.seq_id} row={row} "
                        f"seq_tokens={seq.num_tokens} completion={seq.num_completion_tokens} "
                        f"draft={draft_token_chains[local_row][0]} "
                        f"target_one={target_tokens[idx]} target_commit={parity_targets[idx]} "
                        f"bonus_one={bonus_tokens[idx]} bonus_commit={parity_bonus[idx]} "
                        f"accepted_one={accepted_flags[idx]} accepted_commit={parity_accepted[idx]} "
                        f"next_one={next_draft_chains[idx][0]} next_commit={parity_next[idx]}",
                        flush=True,
                    )
                    if os.environ.get("NANO_VLLM_JAX_MTP_PARITY_STOP", "0") in {
                        "1",
                        "true",
                        "yes",
                        "on",
                        "True",
                    }:
                        raise RuntimeError("MTP one-pass parity mismatch")
                    break
        t_profile = _mark("accepted_result_transfer", t_profile)
        outputs: dict[int, List[int] | int] = {}
        stats = self._speculative_stats()
        seed_after_bonus = os.environ.get("NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        if not hasattr(self, "_mtp1_seeded_chain"):
            self._mtp1_seeded_chain = {}
        max_seeded_chain = int(os.environ.get("NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN", "0") or "0")
        disable_bonus = (
            draft_len == 1
            and os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_BONUS", "0")
            in {"1", "true", "yes", "on", "True"}
        )

        for local_row, row in enumerate(rows):
            seq = seqs[row]
            self._mtp1_drafts.pop(seq.seq_id, None)
            idx = verifier_index_for_local[local_row]
            forced_reject_probe = row in forced_reject_rows

            accepted = accepted_flags[idx]
            prefix_len = prefix_lengths[idx]
            if draft_len > 1 and prefix_len < draft_len:
                if not forced_reject_probe:
                    stats["drafts_accepted"] += prefix_len
                    stats["drafts_rejected"] += 1
                emitted_len = prefix_len + 1
                if prefix_len == 0:
                    outputs[row] = target_token_rows[idx][0]
                else:
                    outputs[row] = (
                        draft_token_chains[local_row][:prefix_len]
                        + [target_token_rows[idx][prefix_len]]
                    )
            elif accepted:
                stats["drafts_accepted"] += draft_len
                if disable_bonus:
                    emitted_len = draft_len
                    outputs[row] = draft_token_chains[local_row]
                else:
                    stats["bonus_tokens"] += 1
                    emitted_len = draft_len + 1
                    outputs[row] = draft_token_chains[local_row] + [bonus_tokens[idx]]
            else:
                if not forced_reject_probe:
                    stats["drafts_rejected"] += 1
                emitted_len = 1
                outputs[row] = target_tokens[idx]

            emitted_bonus = accepted and not disable_bonus
            if draft_len == 1:
                # The rejected-row next-draft invariant is not proven for K=1:
                # direct rejected commit is correct only when the following
                # step falls back to a normal decode. Seeding the verifier's
                # rejected-row next_draft_token causes visible token drift.
                can_seed_next_chain = accepted and seed_after_bonus
            else:
                can_seed_next_chain = prefix_len < draft_len or seed_after_bonus
            if (
                self.mtp1_enabled
                and seq.temperature == 0
                and seq.num_completion_tokens + emitted_len < seq.max_tokens
                and can_seed_next_chain
                and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                and (
                    max_seeded_chain <= 0
                    or self._mtp1_seeded_chain.get(seq.seq_id, 0) < max_seeded_chain
                )
            ):
                next_chain = next_draft_chains[idx]
                self._mtp1_drafts[seq.seq_id] = next_chain if len(next_chain) > 1 else next_chain[0]
                seeded_increment = max(
                    1,
                    min(draft_len, emitted_len - (1 if emitted_bonus else 0)),
                )
                self._mtp1_seeded_chain[seq.seq_id] = (
                    self._mtp1_seeded_chain.get(seq.seq_id, 0) + seeded_increment
                )
                stats["drafts_proposed"] += len(next_chain)
            else:
                self._mtp1_seeded_chain.pop(seq.seq_id, None)

        return outputs

    def _run_mtp1(self, seqs: List[Sequence], batch: ScheduledBatch) -> List[List[int] | int]:
        seq = seqs[0]
        draft_token = self._mtp1_drafts.pop(seq.seq_id)
        draft_debug_by_seq, debug_events = self._mtp1_debug_state()
        draft_debug = draft_debug_by_seq.pop(seq.seq_id, {}) if getattr(self, "mtp_debug", False) else {}
        verifier_batch = self._mtp1_verification_batch(seq, batch, draft_token)
        hybrid_state = self._batch_hybrid_state(verifier_batch)
        use_debug = getattr(self, "mtp_debug", False)
        use_fused_step = (
            not use_debug
            and getattr(self, "execution", "eager") in {"decode-jit", "jit"}
            and hasattr(self.executor, "mtp1_greedy_step_jit")
        )
        if use_fused_step:
            output = self.executor.mtp1_greedy_step_jit(
                verifier_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=draft_token,
                next_mtp_position=seq.num_tokens + 1 + int(getattr(self, "mtp_position_offset", 0)),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            logits = None
            verify_logits = None
            token_ids = None
            target_token = int(output.target_token[0].item())
            accepted = bool(output.accepted[0].item())
        else:
            output = self._mtp1_verifier_step_fn()(
                verifier_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                return_hidden=True,
                last_logits_only=False,
            )
            use_direct_tokens = not use_debug and hasattr(self, "params")
            if use_direct_tokens:
                token_ids = self._greedy_tokens_from_hidden(output.activations)
                logits = None
                verify_logits = None
                target_token = int(token_ids[0, 0])
            else:
                token_ids = None
                logits = self._logits_from_hidden(output.activations)
                verify_logits = logits[0, 0]
                target_token = int(jnp.argmax(verify_logits))
            accepted = target_token == draft_token

        if use_debug:
            logits = self._logits_from_hidden(output.activations)
            verify_logits = logits[0, 0]
            target_token = int(jnp.argmax(verify_logits))
            accepted = target_token == draft_token
            event = {
                **draft_debug,
                "target_token": target_token,
                "accepted": bool(accepted),
                "draft_rank_in_main": self._token_rank(verify_logits, draft_token),
                "main_top": self._topk_debug(verify_logits),
                "target_in_mtp_top5": target_token in draft_debug.get("mtp_top", {}).get("ids", []),
            }
            debug_events.append(event)
        if not accepted:
            self._speculative_stats()["drafts_rejected"] += 1
            return self._run_main_and_sample(seqs, batch, seed_mtp1=True)

        self.cache_storage = output.cache_storage
        self._store_batch_hybrid_state(verifier_batch, output.hybrid_state)
        self._refresh_kv_snapshot(verifier_batch, output.hybrid_state)

        stats = self._speculative_stats()
        stats["drafts_accepted"] += 1
        stats["bonus_tokens"] += 1

        if use_fused_step:
            bonus_token = int(output.bonus_token[0].item())
            if (
                self.mtp1_enabled
                and seq.temperature == 0
                and seq.num_completion_tokens + 1 < seq.max_tokens
                and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
            ):
                self._mtp1_drafts[seq.seq_id] = int(output.next_draft_token[0].item())
                self._speculative_stats()["drafts_proposed"] += 1
            else:
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_debug_state()[0].pop(seq.seq_id, None)
        elif logits is not None:
            bonus_token = int(self._sample_fn(logits[:, 1], jnp.array([seq.temperature], dtype=jnp.float32))[0])
            self._seed_mtp1_draft(
                seq,
                self._hidden_for_mtp(output.activations[:, 1:2, :])[0, 0],
                bonus_token,
                position=seq.num_tokens + 1,
            )
        else:
            bonus_token = int(token_ids[0, 1])
            self._seed_mtp1_draft(
                seq,
                self._hidden_for_mtp(output.activations[:, 1:2, :])[0, 0],
                bonus_token,
                position=seq.num_tokens + 1,
            )
        return [[draft_token, bonus_token]]

    def run(
        self,
        seqs: List[Sequence],
        is_prefill: bool | None = None,
        *,
        batch: ScheduledBatch | None = None,
    ) -> List[int | List[int]]:
        """Run one engine step through the canonical executor path."""
        if batch is None:
            if is_prefill is None:
                raise ValueError("Either is_prefill or batch must be provided")
            batch = self._build_scheduled_batch(seqs, is_prefill=is_prefill)

        seed_mtp1 = True
        force_commit_select = os.environ.get("NANO_VLLM_JAX_MTP_COMMIT_SELECT", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        disable_one_pass_k1 = os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        allow_unsafe_one_pass_k1 = os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        exact_commit_select_available = (
            hasattr(self.executor, "mtp1_commit_select_greedy_step_jit")
            and (force_commit_select or disable_one_pass_k1 or not allow_unsafe_one_pass_k1)
        )
        if batch.is_prefill:
            prefill_final_flags = [
                bool(flag) for flag in (batch.prefill_final_flags[: len(seqs)] if len(batch.prefill_final_flags) >= len(seqs) else batch.prefill_final_flags)
            ]
            # Draft seeding after prefill is read-only with respect to target
            # KV/hybrid state and is already row-gated in _run_main_and_sample:
            # non-final prompt chunks emit no token and are skipped.  Do not tie
            # this to verifier shape policy; bucket-padded or heterogeneous
            # final prefill rows still need initial drafts for the following
            # decode step to exercise scheduler-owned MTP admission.
            seed_mtp1 = any(prefill_final_flags)
            if os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_PREFILL_SEED", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }:
                seed_mtp1 = False

        admitted_mtp_rows = [
            row for row, seq in enumerate(seqs) if self._seq_mtp_admitted(seq)
        ]
        if self.mtp1_enabled and not batch.is_prefill:
            non_admitted_rows = [
                row for row, seq in enumerate(seqs) if not self._seq_mtp_admitted(seq)
            ]
            if non_admitted_rows:
                self._clear_mtp1_drafts_for_rows(seqs, non_admitted_rows)
        if self.mtp1_enabled and not batch.is_prefill and admitted_mtp_rows:
            fused_rows: List[int] = []
            probe_candidate_rows: List[int] = []
            profile_mtp = os.environ.get("NANO_VLLM_JAX_PROFILE_MTP_RUN", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            not_fused_reasons: Dict[str, int] = {}
            for row, seq in enumerate(seqs):
                draft_value = self._mtp1_drafts.get(seq.seq_id)
                draft_len = len(draft_value) if isinstance(draft_value, list) else (1 if draft_value is not None else 0)
                draft_len = min(draft_len, max(1, int(getattr(self, "num_speculative_tokens", 1) or 1)))
                # MTP can emit all accepted drafts plus one bonus token. Require
                # block-table capacity for the post-emit logical length, not
                # just the verifier-written draft positions.
                verifier_width = max(1, draft_len)
                required_blocks = (seq.num_tokens + verifier_width + 1 + self.block_size - 1) // self.block_size
                unsafe_bonus_boundary = (seq.num_tokens + verifier_width + 1) % self.block_size == 0
                can_fuse = (
                    draft_value is not None
                    and draft_len > 0
                    and self._seq_mtp_admitted(seq)
                    and seq.temperature == 0
                    and seq.num_completion_tokens + draft_len + 1 <= seq.max_tokens
                    and int(batch.query_lens[row]) == 1
                    and len(seq.block_table) >= required_blocks
                    and not unsafe_bonus_boundary
                )
                if can_fuse:
                    fused_rows.append(row)
                elif (
                    draft_value is None
                    and self.num_speculative_tokens == 1
                    and self._seq_mtp_admitted(seq)
                    and seq.temperature == 0
                    and seq.num_completion_tokens + 1 <= seq.max_tokens
                    and int(batch.query_lens[row]) == 1
                    and len(seq.block_table) >= required_blocks
                    and not unsafe_bonus_boundary
                ):
                    probe_candidate_rows.append(row)
                elif profile_mtp:
                    if draft_value is None:
                        reason = "missing_draft"
                    elif draft_len <= 0:
                        reason = "empty_draft"
                    elif not self._seq_mtp_admitted(seq):
                        reason = "scheduler_gate"
                    elif seq.temperature != 0:
                        reason = "temperature"
                    elif seq.num_completion_tokens + draft_len + 1 > seq.max_tokens:
                        reason = "max_tokens"
                    elif int(batch.query_lens[row]) != 1:
                        reason = "query_len"
                    elif len(seq.block_table) < required_blocks:
                        reason = "blocks"
                    elif unsafe_bonus_boundary:
                        reason = "bonus_boundary"
                    else:
                        reason = "other"
                    not_fused_reasons[reason] = not_fused_reasons.get(reason, 0) + 1

            allow_mixed_fused = os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            batch_accept_policy = os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none")
            force_reuse_fallback = os.environ.get("NANO_VLLM_JAX_MTP_FORCE_REUSE_FALLBACK", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            enable_rowwise_repair = os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_ROWWISE_REPAIR", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            seed_after_bonus_enabled = os.environ.get("NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            allow_seeded_one_pass_k1 = os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            allow_unsafe_one_pass_k1 = os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            one_pass_available_for_partial = (
                hasattr(self.executor, "mtp1_two_decode_greedy_step_jit")
                and os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1", "0")
                not in {"1", "true", "yes", "on", "True"}
                and allow_unsafe_one_pass_k1
                and (not seed_after_bonus_enabled or allow_seeded_one_pass_k1)
            )
            homogeneous_full_batch = (
                len(seqs) == batch.tokens.shape[0]
                and len({seq.num_tokens for seq in seqs}) == 1
            )
            full_physical_batch = len(seqs) == batch.tokens.shape[0]
            allow_exact_commit_select_mixed = exact_commit_select_available
            partial_prefix_verifier = (
                not full_physical_batch
                and (
                    one_pass_available_for_partial
                    or hasattr(self.executor, "mtp1_commit_select_greedy_step_jit")
                )
            )
            allow_partial_commit_select = (
                not allow_exact_commit_select_mixed
                or os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_PARTIAL_COMMIT_SELECT", "0")
                in {"1", "true", "yes", "on", "True"}
            )
            allow_verifier_for_batch_shape = full_physical_batch or (
                partial_prefix_verifier and allow_partial_commit_select
            )
            can_seed_for_decode_shape = (
                allow_verifier_for_batch_shape
                and (
                    allow_mixed_fused
                    or allow_exact_commit_select_mixed
                    or homogeneous_full_batch
                    or one_pass_available_for_partial
                )
            )
            one_pass_expected = (
                one_pass_available_for_partial
                and (not force_commit_select or allow_mixed_fused or not full_physical_batch)
            )
            k1_commit_rejected_enabled = os.environ.get("NANO_VLLM_JAX_MTP_K1_COMMIT_REJECTED", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            allow_unsafe_forced_reject_probes = os.environ.get(
                "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_FORCED_REJECT_PROBES", "0"
            ) in {"1", "true", "yes", "on", "True"}
            can_commit_forced_reject_rows = force_commit_select or (
                allow_unsafe_forced_reject_probes
                and one_pass_expected
                and k1_commit_rejected_enabled
            )
            allow_forced_reject_probes = (
                os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_FORCED_REJECT_PROBES", "0")
                in {"1", "true", "yes", "on", "True"}
                and batch_accept_policy == "rowwise"
                and can_commit_forced_reject_rows
            )
            probe_rows = probe_candidate_rows if allow_forced_reject_probes and fused_rows else []
            verifier_row_set = set(fused_rows) | set(probe_rows)
            verifier_rows = [
                row for row in range(len(seqs))
                if row in verifier_row_set
            ]
            compact_commit_select = (
                allow_exact_commit_select_mixed
                and batch_accept_policy == "rowwise"
                and len(seqs) > 1
                and allow_verifier_for_batch_shape
                and os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_COMPACT_COMMIT_SELECT", "0")
                in {"1", "true", "yes", "on", "True"}
            )
            if compact_commit_select:
                legacy_compact_reuse = os.environ.get(
                    "NANO_VLLM_JAX_MTP_ENABLE_LEGACY_COMPACT_REUSE", "0"
                ) in {"1", "true", "yes", "on", "True"}
                if legacy_compact_reuse:
                    return self._run_main_and_sample_with_mtp1_reuse(
                        seqs,
                        batch,
                        seed_mtp1=self.mtp1_enabled and seed_mtp1,
                        force_emit_bonus=True,
                    )
            can_run_fused_batch = (
                not force_reuse_fallback
                and
                fused_rows
                and allow_verifier_for_batch_shape
                and can_seed_for_decode_shape
            )
            if can_run_fused_batch:
                fused_outputs = self._run_mtp1_batched(
                    seqs,
                    batch,
                    verifier_rows,
                    forced_reject_rows=set(probe_rows),
                )
                if fused_outputs is not None:
                    outputs: List[int | List[int] | None] = [None] * len(seqs)
                    for row, value in fused_outputs.items():
                        outputs[row] = value
                    fallback_rows = [row for row in range(len(seqs)) if outputs[row] is None]
                    if fallback_rows:
                        stats = self._speculative_stats()
                        stats["fallback_partial_rows"] += len(fallback_rows)
                        stats["fallback_gated_no_spec_steps"] += 1
                        fallback_batch = self._masked_decode_batch(batch, fallback_rows)
                        fallback_outputs = self._run_main_and_sample(
                            seqs,
                            fallback_batch,
                            seed_mtp1=False,
                        )
                        for row in fallback_rows:
                            outputs[row] = fallback_outputs[row]
                    return [outputs[row] for row in range(len(seqs))]  # type: ignore[list-item]
            elif profile_mtp:
                print(
                    f"[MTP_RUN] fused_rows={len(fused_rows)}/{len(seqs)} "
                    f"batch_shape={tuple(batch.tokens.shape)} reasons={not_fused_reasons}",
                    flush=True,
                )

            if os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_REUSE_FALLBACK", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }:
                return self._run_main_and_sample_with_mtp1_reuse(
                    seqs,
                    batch,
                    seed_mtp1=self.mtp1_enabled and seed_mtp1,
                )

            if not can_seed_for_decode_shape:
                self._clear_mtp1_drafts_for_rows(seqs, admitted_mtp_rows)

            main_seed_mtp1 = (
                self.mtp1_enabled
                and seed_mtp1
                and can_seed_for_decode_shape
            )
            stats = self._speculative_stats()
            if main_seed_mtp1:
                stats["fallback_seeded_main_steps"] += 1
            elif self.mtp1_enabled and not batch.is_prefill:
                stats["fallback_gated_no_spec_steps"] += 1
            return self._run_main_and_sample(
                seqs,
                batch,
                seed_mtp1=main_seed_mtp1,
            )

        if self.mtp1_enabled and not batch.is_prefill and not admitted_mtp_rows:
            self._speculative_stats()["fallback_gated_no_spec_steps"] += 1
        return self._run_main_and_sample(
            seqs,
            batch,
            seed_mtp1=self.mtp1_enabled and seed_mtp1 and bool(admitted_mtp_rows),
        )

    def forward(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        kv_state: KVCacheState,
        is_prefill: bool,
    ) -> jnp.ndarray:
        batch = ScheduledBatch(
            tokens=input_ids,
            positions=positions,
            seq_ids=jnp.arange(input_ids.shape[0], dtype=jnp.int32),
            query_start_loc=jnp.arange(input_ids.shape[0] + 1, dtype=jnp.int32) * input_ids.shape[1],
            is_prefill=is_prefill,
            num_prefill_tokens=int(input_ids.size) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else input_ids.shape[0],
            block_tables=kv_state.block_table[: input_ids.shape[0]],
            seq_lens=kv_state.kv_lens[: input_ids.shape[0]],
        )
        output = self.executor.forward_step(
            batch,
            cache_storage=KVCacheStorage(kv_state.k_cache, kv_state.v_cache),
            hybrid_state=kv_state.hybrid_state,
        )
        return output.activations

    @partial(jax.jit, static_argnums=(0,))
    def _sample_logits(
        self,
        logits: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        import jax.lax as lax

        def sample_single(logit, temp):
            def greedy(_):
                return jnp.argmax(logit)

            def sample(_):
                scaled = logit / temp
                return jax.random.categorical(jax.random.PRNGKey(0), scaled)

            return lax.cond(temp == 0.0, greedy, sample, None)

        return jax.vmap(sample_single)(logits, temperatures)

    def call(self, method: str, *args):
        if method == "run":
            return self.run(*args)
        if method == "exit":
            return None
        raise ValueError(f"Unknown method: {method}")

    def run_speculative(
        self,
        seqs: List[Sequence],
    ) -> List[int | List[int]]:
        return self.run(seqs, is_prefill=False)


class ModelRunner(CanonicalModelRunner):
    """Backward-compatible facade over the canonical executor implementation."""

    def __init__(self, config: Qwen3_5Config, params: ModelParams, backend: str = "auto"):
        super().__init__(config=config, params=params, backend=backend)
