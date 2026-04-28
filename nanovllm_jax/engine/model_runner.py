"""Model runner for JAX inference with paged KV cache."""

import time
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


class ModelRunner:
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
        self._sample_fn = jax.jit(self._sample_logits)
        self.mtp_enabled = hasattr(params, "mtp_params") and params.mtp_params is not None
        self.num_speculative_tokens = int(getattr(config, "num_speculative_tokens", 0) or 0)
        self.mtp1_enabled = self.mtp_enabled and self.num_speculative_tokens == 1
        self._mtp1_forward_jit = None
        self._mtp1_token_jit = None
        self._hidden_token_jit = None
        self._mtp1_drafts: Dict[int, int] = {}
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
                _ = self.executor.forward_step_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    last_logits_only=True,
                ).activations.block_until_ready()

        for batch_size in batch_buckets:
            batch = self._dummy_batch(batch_size=batch_size, query_len=1, is_prefill=False)
            hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
            _ = self.executor.forward_step_jit(
                batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                last_logits_only=True,
            ).activations.block_until_ready()
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
            num_decode_tokens=0 if is_prefill else len(seqs),
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
        )

    @staticmethod
    def _select_bucket(size: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in sorted(buckets):
            if size <= bucket:
                return bucket
        raise ValueError(f"{name} size {size} exceeds configured buckets {buckets}")

    def _get_hybrid_state(self, seq_id: int) -> HybridLayerState:
        if seq_id < 0:
            zero_state = init_hybrid_state(self.config, batch_size=1, dtype=self.config.get_dtype())
            return zero_state
        if seq_id not in self.hybrid_states:
            zero_state = init_hybrid_state(self.config, batch_size=1, dtype=self.config.get_dtype())
            self.hybrid_states[seq_id] = HybridLayerState(
                conv_state=zero_state.conv_state[0],
                recurrent_state=zero_state.recurrent_state[0],
            )
        state = self.hybrid_states[seq_id]
        return HybridLayerState(
            conv_state=state.conv_state[None, ...] if state.conv_state is not None else None,
            recurrent_state=state.recurrent_state[None, ...] if state.recurrent_state is not None else None,
        )

    def _set_hybrid_state(self, seq_id: int, state: HybridLayerState | None):
        if state is None or seq_id < 0:
            return
        self.hybrid_states[seq_id] = HybridLayerState(
            conv_state=state.conv_state[0] if state.conv_state is not None else None,
            recurrent_state=state.recurrent_state[0] if state.recurrent_state is not None else None,
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

    def _batch_hybrid_state(self, batch: ScheduledBatch) -> HybridLayerState:
        conv_states = []
        recurrent_states = []
        for seq_id in [int(x) for x in batch.seq_ids.tolist()]:
            state = self._get_hybrid_state(seq_id)
            conv_states.append(state.conv_state[0])
            recurrent_states.append(state.recurrent_state[0])
        return HybridLayerState(
            conv_state=jnp.stack(conv_states, axis=0),
            recurrent_state=jnp.stack(recurrent_states, axis=0),
        )

    def _store_batch_hybrid_state(self, batch: ScheduledBatch, state: HybridLayerState | None):
        if state is None:
            return
        for row, seq_id in enumerate([int(x) for x in batch.seq_ids.tolist()]):
            if seq_id < 0:
                continue
            self.hybrid_states[seq_id] = HybridLayerState(
                conv_state=state.conv_state[row] if state.conv_state is not None else None,
                recurrent_state=state.recurrent_state[row] if state.recurrent_state is not None else None,
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
    ) -> List[int]:
        hybrid_state = self._batch_hybrid_state(batch)
        output = self._step_fn(batch)(
            batch,
            cache_storage=self.cache_storage,
            hybrid_state=hybrid_state,
            return_hidden=seed_mtp1,
            last_logits_only=not seed_mtp1,
        )
        self.cache_storage = output.cache_storage
        self._store_batch_hybrid_state(batch, output.hybrid_state)
        self._refresh_kv_snapshot(batch, output.hybrid_state)

        if seed_mtp1:
            last_hidden = self._last_query_activations(output.activations, batch, len(seqs))
            last_logits = self._logits_from_hidden(last_hidden[:, None, :])[:, 0]
        else:
            last_hidden = None
            last_logits = output.activations[: len(seqs), 0]

        temperatures = jnp.array([seq.temperature for seq in seqs], dtype=jnp.float32)
        token_ids = self._sample_fn(last_logits, temperatures)
        token_list = [int(token_id) for token_id in token_ids.tolist()]
        if seed_mtp1 and last_hidden is not None:
            self._seed_mtp1_drafts(seqs, self._hidden_for_mtp(last_hidden[:, None, :])[:, 0], token_list)
        return token_list

    def _seed_mtp1_drafts(
        self,
        seqs: List[Sequence],
        hidden: jnp.ndarray,
        confirmed_token_ids: List[int],
    ):
        for row, seq in enumerate(seqs):
            confirmed_token_id = confirmed_token_ids[row]
            position = seq.num_tokens
            if getattr(self, "mtp_token_source", "generated") == "current":
                confirmed_token_id = seq.last_token
                position = seq.num_tokens - 1
            self._seed_mtp1_draft(
                seq,
                hidden[row],
                confirmed_token_id,
                position=position,
            )

    def _seed_mtp1_draft(
        self,
        seq: Sequence,
        hidden: jnp.ndarray,
        confirmed_token_id: int,
        position: int,
    ):
        if not self.mtp1_enabled or seq.temperature != 0:
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
        if getattr(self, "mtp_debug", False):
            draft_logits = self._mtp1_logits(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
            )
            draft_vector = draft_logits[0, 0]
            draft_token = int(jnp.argmax(draft_vector))
        else:
            draft_token = int(self._mtp1_draft_token(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
            )[0])
        self._mtp1_drafts[seq.seq_id] = draft_token
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
        if seq.temperature != 0 or seq.seq_id not in self._mtp1_drafts:
            self._speculative_stats()["fallback_steps"] += 1
            return False
        if seq.num_completion_tokens + 2 > seq.max_tokens:
            self._speculative_stats()["fallback_steps"] += 1
            return False
        if int(batch.query_lens[0]) != 1:
            self._speculative_stats()["fallback_steps"] += 1
            return False
        required_blocks = (seq.num_tokens + 1 + self.block_size - 1) // self.block_size
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
            target_token = int(output.target_token.item())
            accepted = bool(output.accepted.item())
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

        stats = self._speculative_stats()
        stats["drafts_accepted"] += 1
        stats["bonus_tokens"] += 1
        self.cache_storage = output.cache_storage
        self._store_batch_hybrid_state(verifier_batch, output.hybrid_state)
        self._refresh_kv_snapshot(verifier_batch, output.hybrid_state)

        if use_fused_step:
            bonus_token = int(output.bonus_token.item())
            if self.mtp1_enabled and seq.temperature == 0 and seq.num_completion_tokens + 1 < seq.max_tokens:
                self._mtp1_drafts[seq.seq_id] = int(output.next_draft_token.item())
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

        if self._can_run_mtp1(seqs, batch):
            return self._run_mtp1(seqs, batch)
        return self._run_main_and_sample(seqs, batch, seed_mtp1=self.mtp1_enabled)

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


ModelRunner = CanonicalModelRunner
