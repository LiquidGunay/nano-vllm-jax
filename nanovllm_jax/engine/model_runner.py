"""Model runner for JAX inference with paged KV cache."""

import time
import jax
import jax.numpy as jnp
import sys
from typing import List, Tuple, Dict, Optional, Any
from functools import partial
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import ModelParams, full_attention_block, gated_deltanet_block, transformer_block, forward as model_forward
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.kv_cache import (
    KVCacheState, 
    init_kv_cache, 
    init_linear_attention_states,
    compute_slot_mapping,
    update_kv_cache,
    paged_attention,
    paged_attention_decode,
)
from nanovllm_jax.mtp.mtp_layer import MTPParams, mtp_forward
from nanovllm_jax.mtp.speculative import generate_draft_tokens, verify_draft_tokens, apply_acceptance


class ModelRunner:
    """Runs JAX model with paged KV cache.
    
    Handles:
    - KV cache state management
    - Prefill vs decode execution
    - Block table to JAX array conversion
    - Logits computation and sampling
    """

    def __init__(self, config: Qwen3_5Config, params: ModelParams):
        self.config = config
        self.params = params
        self.block_size = config.block_size
        
        # Initialize KV cache state
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
    ) -> List[int]:
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
    ) -> List[int]:
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
        draft_logits = mtp_forward(
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
