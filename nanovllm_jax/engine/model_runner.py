"""Model runner for JAX inference with paged KV cache."""

import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
from typing import List, Tuple, Dict, Optional, Any
from functools import partial
from dataclasses import replace

from nanovllm_jax.backends import select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import ModelParams, full_attention_block, gated_deltanet_block, transformer_block, forward as model_forward
from nanovllm_jax.engine.sequence import DeviceTokenRef, Sequence
from nanovllm_jax.kv_cache import (
    KVCacheState,
    KVCacheSpec,
    cap_num_kv_cache_blocks,
    init_kv_cache,
    init_hybrid_state,
    init_linear_attention_states,
    compute_slot_mapping,
)
from nanovllm_jax.mtp.mtp_layer import (
    MTPParams,
    mtp_forward,
    mtp_forward_last,
    mtp_forward_last_token_ids,
    mtp_forward_token_ids,
)
from nanovllm_jax.mtp.speculative import generate_draft_tokens, verify_draft_tokens, apply_acceptance

_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}
_UNVERIFIED_MTP_APPEND_ERROR = (
    "Unverified MTP draft append is not supported. "
    "All MTP benchmark and serving paths must verify drafts with the target model."
)


def _block_until_ready_tree(value: object) -> None:
    ready = getattr(value, "block_until_ready", None)
    if callable(ready):
        ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready_tree(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready_tree(item)
        return
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if dataclass_fields is not None:
        for name in dataclass_fields:
            _block_until_ready_tree(getattr(value, name))
        return
    for leaf in jax.tree_util.tree_leaves(value):
        leaf_ready = getattr(leaf, "block_until_ready", None)
        if callable(leaf_ready):
            leaf_ready()


def _config_or_env_flag(config: Qwen3_5Config | None, attr: str, env_name: str, *, default: bool = False) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value in _TRUE_ENV_VALUES
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return bool(default)


def _config_or_env_int(config: Qwen3_5Config | None, attr: str, env_name: str, *, default: int = 0) -> int:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value or default)
    if config is not None and hasattr(config, attr):
        return int(getattr(config, attr) or default)
    return int(default)


def _unverified_mtp_append_enabled(config: Qwen3_5Config | None, attr: str, env_name: str) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value in _TRUE_ENV_VALUES:
        raise ValueError(_UNVERIFIED_MTP_APPEND_ERROR)
    if config is not None and bool(getattr(config, attr, False)):
        raise ValueError(_UNVERIFIED_MTP_APPEND_ERROR)
    return False


def _int32_device_vector(value) -> jnp.ndarray:
    """Return a 1D int32 device vector without re-wrapping existing int32 arrays."""

    if hasattr(value, "dtype") and getattr(value, "dtype", None) == jnp.dtype(jnp.int32):
        if getattr(value, "ndim", None) == 1:
            return value
        return value.reshape(-1)
    return jnp.asarray(value, dtype=jnp.int32).reshape(-1)


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
        self.backend = select_backend(backend, config=config)
        self.block_size = config.block_size

        # Initialize KV cache state
        max_seqs = int(
            getattr(config, "max_num_resident_seqs", None)
            or getattr(config, "max_num_seqs", 16)
        )
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
        self.decode_block_table_buckets = tuple(
            getattr(config, "decode_block_table_buckets", ()) or ()
        )
        self.execution = getattr(config, "jax_execution", "eager")
        self.greedy_token_fastpath = _config_or_env_flag(
            config,
            "greedy_token_fastpath",
            "NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH",
            default=True,
        )
        self.sampled_token_fastpath = _config_or_env_flag(
            config,
            "sampled_token_fastpath",
            "NANO_VLLM_JAX_SAMPLED_TOKEN_FASTPATH",
            default=True,
        )
        self.device_token_carry = _config_or_env_flag(
            config,
            "device_token_carry",
            "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        )
        self.static_decode_metadata = _config_or_env_flag(
            config,
            "static_decode_metadata",
            "NANO_VLLM_JAX_STATIC_DECODE_METADATA",
        )
        self.resident_decode_metadata = bool(
            getattr(config, "resident_decode_metadata", False)
        )
        self.static_decode_seq_lens_carry = _config_or_env_flag(
            config,
            "static_decode_seq_lens_carry",
            "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
        )

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

    def warmup_compilation(
        self,
        max_prefill_len: int = 64,
        max_batch: int = 1,
        *,
        include_sampled_routes: bool = True,
    ):
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
            recurrent_state: Recurrent state for linear attention [batch, num_linear_layers, num_heads, v_dim, k_dim]

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
        self.backend = select_backend(backend, config=config)
        self.executor = ModelExecutor(config, params, self.backend)
        self.block_size = config.block_size

        max_seqs = int(
            getattr(config, "max_num_resident_seqs", None)
            or getattr(config, "max_num_seqs", 16)
        )
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
        self.greedy_token_fastpath = _config_or_env_flag(
            config,
            "greedy_token_fastpath",
            "NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH",
            default=True,
        )
        self.device_token_carry = _config_or_env_flag(
            config,
            "device_token_carry",
            "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        )
        self.static_decode_metadata = _config_or_env_flag(
            config,
            "static_decode_metadata",
            "NANO_VLLM_JAX_STATIC_DECODE_METADATA",
        )
        self.resident_decode_metadata = bool(
            getattr(config, "resident_decode_metadata", False)
        )
        self.static_decode_seq_lens_carry = _config_or_env_flag(
            config,
            "static_decode_seq_lens_carry",
            "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
        )

        self.cache_storage = self.backend.allocate_kv_cache(
            replace(kv_spec, num_blocks=effective_num_blocks),
            max_seqs=max_seqs,
            max_blocks_per_seq=self.max_blocks_per_seq,
        )
        self.full_attention_nhd_cache = self.backend.allocate_full_attention_nhd_kv_cache(
            replace(kv_spec, num_blocks=effective_num_blocks),
            full_attention_layers=tuple(
                layer_id
                for layer_id, layer_type in enumerate(config.layer_types)
                if layer_type == "full_attention"
            ),
        )
        self.hybrid_states: Dict[int, HybridLayerState] = {}
        self._max_hybrid_slots = max_seqs
        self._hybrid_slots: Dict[int, int] = {}
        self._free_hybrid_slots: List[int] = list(range(max_seqs))
        self._zeroed_hybrid_slots: set[int] = set(range(max_seqs))

        empty_hybrid_state = init_hybrid_state(
            config=config,
            batch_size=1,
            dtype=config.get_dtype(),
        )
        self.kv_state = KVCacheState(
            k_cache=self.cache_storage.k_cache,
            v_cache=self.cache_storage.v_cache,
            block_table=jnp.zeros((max_seqs, self.max_blocks_per_seq), dtype=jnp.int32),
            kv_lens=jnp.zeros(max_seqs, dtype=jnp.int32),
            slot_mapping=jnp.zeros((max_seqs, 1), dtype=jnp.int32),
            conv_state=empty_hybrid_state.conv_state,
            recurrent_state=empty_hybrid_state.recurrent_state,
        )
        self._empty_hybrid_state = self.kv_state.hybrid_state
        self._hybrid_state_table = init_hybrid_state(
            self.config,
            batch_size=max_seqs,
            dtype=self.config.get_dtype(),
        )
        self._resident_block_tables = jnp.zeros(
            (max_seqs, self.max_blocks_per_seq),
            dtype=jnp.int32,
        )
        self._resident_seq_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_block_tables_host: list[tuple[int, ...]] = [
            tuple(0 for _ in range(self.max_blocks_per_seq))
            for _ in range(max_seqs)
        ]
        self._resident_block_counts_host: list[int] = [0 for _ in range(max_seqs)]
        self._resident_seq_lens_host: list[int] = [0 for _ in range(max_seqs)]
        self._resident_last_tokens = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_rng_counters = jnp.zeros((max_seqs,), dtype=jnp.int32)
        self._resident_rng_counter_reset_slots: set[int] = set()
        self._sample_fn = jax.jit(self._sample_logits)
        self.speculative_method = str(getattr(config, "speculative_method", "none") or "none").lower()
        requested_mtp = (
            self.speculative_method == "mtp"
            and int(getattr(config, "num_speculative_tokens", 0) or 0) > 0
        )
        self.mtp_enabled = requested_mtp and hasattr(params, "mtp_params") and params.mtp_params is not None
        if requested_mtp and not self.mtp_enabled:
            raise ValueError("speculative_method='mtp' requires loaded mtp.* weights")
        self.num_speculative_tokens = (
            int(getattr(config, "num_speculative_tokens", 0) or 0)
            if requested_mtp
            else 0
        )
        self.mtp1_enabled = self.mtp_enabled and self.num_speculative_tokens > 0
        self.draft_sample_method = str(getattr(config, "draft_sample_method", "greedy") or "greedy").lower()
        self.mtp_verifier_impl = str(getattr(config, "mtp_verifier_impl", "packed_prefix") or "packed_prefix").lower()
        self.mtp_batch_accept_policy = str(
            getattr(config, "mtp_batch_accept_policy", "rowwise") or "rowwise"
        ).lower()
        self.mtp_seed_after_bonus = bool(getattr(config, "mtp_seed_after_bonus", False))
        self.mtp_bonus_margin = float(getattr(config, "mtp_bonus_margin", 0.0) or 0.0)
        self.mtp_draft_margin = float(getattr(config, "mtp_draft_margin", 0.0) or 0.0)
        self.mtp_hidden_source = str(getattr(config, "mtp_hidden_source", "pre_norm") or "pre_norm").lower()
        self.mtp_chain_hidden_source = str(
            getattr(config, "mtp_chain_hidden_source", "raw") or "raw"
        ).lower()
        self.mtp_chain_mode = str(
            getattr(config, "mtp_chain_mode", "recursive") or "recursive"
        ).lower()
        self.mtp_token_source = str(getattr(config, "mtp_token_source", "generated") or "generated").lower()
        self.mtp_burst_groups = max(1, int(getattr(config, "mtp_burst_groups", 1) or 1))
        self.mtp_max_active_rows = max(0, int(getattr(config, "mtp_max_active_rows", 0) or 0))
        if self.mtp1_enabled and self.draft_sample_method != "greedy":
            raise ValueError("MTP probabilistic draft sampling is not implemented yet")
        self.mtp_cache_storage = None
        if self.mtp1_enabled:
            mtp_kv_spec = replace(
                kv_spec,
                num_layers=max(1, int(getattr(config, "mtp_num_hidden_layers", 1) or 1)),
                num_blocks=effective_num_blocks,
            )
            self.mtp_cache_storage = self.backend.allocate_kv_cache(
                mtp_kv_spec,
                max_seqs=max_seqs,
                max_blocks_per_seq=self.max_blocks_per_seq,
            )
        # FlashInfer paged decode handles verifier width > 1 through the
        # append/decode loop in the backend. Width>1 GDN verification uses a
        # static token loop inside the compiled model path, so strict fallback
        # checks live at the kernel callsite where the exact packed
        # implementation and projection availability are known.
        compile_mtp_draft_default = os.environ.get("NANO_VLLM_JAX_MTP_COMPILE_DRAFT", "1") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        self.mtp_compile_draft = bool(
            self.mtp1_enabled
            and self.execution in {"decode-jit", "jit"}
            and compile_mtp_draft_default
        )
        self._mtp1_forward_jit = None
        self._mtp1_token_jit = None
        self._hidden_token_jit = None
        self._mtp1_drafts: Dict[int, int] = {}
        self._mtp1_seeded_chain: Dict[int, int] = {}
        self._mtp1_draft_debug: Dict[int, Dict[str, Any]] = {}
        self._mtp1_debug_events: List[Dict[str, Any]] = []
        self._device_token_carry_seq_ids: tuple[int, ...] | None = None
        self._device_token_carry_tokens: jnp.ndarray | None = None
        self._device_token_carry_by_seq_id: dict[int, DeviceTokenRef] = {}
        self._device_seq_lens_carry_seq_ids: tuple[int, ...] | None = None
        self._device_seq_lens_carry: jnp.ndarray | None = None
        self._resident_last_tokens_stale_seq_ids: set[int] = set()
        self._hybrid_slot_ids_device_cache: dict[tuple[int, ...], jnp.ndarray] = {}
        self._prefill_final_flags_device_cache: dict[tuple[bool, ...], jnp.ndarray] = {}
        self.reset_speculative_stats()
        self._warmup_compiled = False

    def _strict_k_mtp_verifier_enabled(self) -> bool:
        """Fail closed for K>1 MTP verifier routes.

        K>1 `commit_select` is debug-only; the runtime path must exercise the
        grouped verifier boundary instead of repairing with sequential decode.
        """
        return (
            self.mtp1_enabled
            and int(getattr(self, "num_speculative_tokens", 0) or 0) > 1
            and str(getattr(self, "mtp_verifier_impl", "two_decode") or "two_decode")
            in {
                "k_decode",
                "generic_k",
                "expanded",
                "packed_prefix",
                "packed_prefill",
                "prefill_packed",
            }
        )

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
        debug_event_limit = int(os.environ.get("NANO_VLLM_JAX_MTP_DEBUG_EVENT_LIMIT", "16") or "16")
        if debug_event_limit <= 0:
            stats["debug_events"] = list(getattr(self, "_mtp1_debug_events", []))
        else:
            stats["debug_events"] = list(
                getattr(self, "_mtp1_debug_events", [])[-debug_event_limit:]
            )
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

    def warmup_compilation(
        self,
        max_prefill_len: int = 64,
        max_batch: int = 1,
        *,
        include_sampled_routes: bool = True,
        prefill_token_buckets: tuple[int, ...] | None = None,
        batch_size_buckets: tuple[int, ...] | None = None,
        decode_block_table_buckets: tuple[int, ...] | None = None,
    ):
        """Compile configured static shapes through the canonical executor path."""
        def _block_until_ready(value: object) -> None:
            _block_until_ready_tree(value)

        def _cache_entries() -> int | None:
            cache = getattr(getattr(self, "executor", None), "_jit_cache", None)
            return len(cache) if cache is not None else None

        summary: dict[str, Any] = {
            "mode": "generic_bucket_startup",
            "execution": getattr(self, "execution", "eager"),
            "prefill_buckets": [],
            "batch_size_buckets": [],
            "prefill_runs": [],
            "prefill_skipped": [],
            "decode_runs": [],
            "decode_block_table_buckets": [],
            "resident_metadata_scatter_runs": [],
            "sampled_token_fastpath_runs": [],
            "include_sampled_routes": bool(include_sampled_routes),
            "jit_cache_entries_before": _cache_entries(),
            "jit_cache_entries_after": None,
            "already_warmed": bool(self._warmup_compiled),
        }
        if self._warmup_compiled:
            summary["jit_cache_entries_after"] = _cache_entries()
            return summary
        if self.execution not in {"decode-jit", "jit"}:
            self._warmup_compiled = True
            summary["jit_cache_entries_after"] = _cache_entries()
            return summary

        prefill_buckets = tuple(int(bucket) for bucket in (prefill_token_buckets or ())) or (
            tuple(getattr(self.config, "prefill_token_buckets", ()))
            or tuple(getattr(self.config, "prefill_buckets", ()))
            or (max_prefill_len,)
        )
        batch_buckets = tuple(int(bucket) for bucket in (batch_size_buckets or ())) or (
            tuple(getattr(self.config, "batch_size_buckets", ())) or (max_batch,)
        )
        decode_block_table_buckets = tuple(
            int(bucket) for bucket in (decode_block_table_buckets or ())
        ) or (
            tuple(getattr(self.config, "decode_block_table_buckets", ()) or ())
            or (int(self.max_blocks_per_seq),)
        )
        sorted_batch_buckets = tuple(sorted(int(bucket) for bucket in batch_buckets))
        padded_decode_buckets = {
            bucket
            for index, bucket in enumerate(sorted_batch_buckets)
            if (sorted_batch_buckets[index - 1] + 1 if index > 0 else 1) < bucket
        }
        summary["prefill_buckets"] = list(prefill_buckets)
        summary["batch_size_buckets"] = list(batch_buckets)
        summary["decode_block_table_buckets"] = [int(width) for width in decode_block_table_buckets]
        row_prefill_buckets = tuple(getattr(self.config, "prefill_buckets", ()) or ())
        use_greedy_token_fastpath = bool(
            getattr(
                self,
                "greedy_token_fastpath",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "greedy_token_fastpath",
                    "NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH",
                    default=True,
                ),
            )
        ) and hasattr(self.executor, "forward_step_token_ids_jit")
        use_sampled_token_fastpath = (
            bool(include_sampled_routes)
            and bool(
                getattr(
                    self,
                    "sampled_token_fastpath",
                    _config_or_env_flag(
                        getattr(self, "config", None),
                        "sampled_token_fastpath",
                        "NANO_VLLM_JAX_SAMPLED_TOKEN_FASTPATH",
                        default=True,
                    ),
                )
            )
            and hasattr(self.executor, "forward_step_sampled_token_ids_jit")
        )
        hybrid_state_table = getattr(self, "_hybrid_state_table", None)
        use_hybrid_table_decode = (
            use_greedy_token_fastpath
            and hybrid_state_table is not None
            and hybrid_state_table.conv_state is not None
            and hybrid_state_table.recurrent_state is not None
            and (
                hasattr(self.executor, "forward_step_token_ids_table_jit")
                or hasattr(self.executor, "forward_greedy_decode_burst_table_jit")
            )
        )
        use_hybrid_table_prefill = (
            use_greedy_token_fastpath
            and hybrid_state_table is not None
            and hybrid_state_table.conv_state is not None
            and hybrid_state_table.recurrent_state is not None
            and hasattr(self.executor, "forward_prefill_token_ids_table_jit")
        )
        use_prefill_slot_carry_table = (
            use_hybrid_table_prefill
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
            and hasattr(self.executor, "forward_prefill_token_ids_slot_carry_table_jit")
        )
        use_sampled_resident_dense_decode = (
            use_sampled_token_fastpath
            and hybrid_state_table is not None
            and hybrid_state_table.conv_state is not None
            and hybrid_state_table.recurrent_state is not None
            and bool(getattr(self, "resident_decode_metadata", False))
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
            and hasattr(self, "_resident_rng_counters")
            and hasattr(self.executor, "forward_step_sampled_token_ids_resident_dense_slot_carry_jit")
        )
        use_slot_carry_table_decode = (
            use_hybrid_table_decode
            and bool(getattr(self, "device_token_carry", False))
            and bool(getattr(self, "static_decode_metadata", False))
            and not bool(getattr(self, "resident_decode_metadata", False))
            and hasattr(self.executor, "forward_step_token_ids_slot_carry_table_jit")
        )
        seed_mtp1 = bool(getattr(self, "mtp1_enabled", False))
        mtp_max_active_rows = max(
            0,
            int(
                os.environ.get(
                    "NANO_VLLM_JAX_MTP_MAX_ACTIVE_ROWS",
                    str(getattr(self.config, "mtp_max_active_rows", 0)),
                )
                or "0"
            ),
        )
        prefill_seed_mtp1 = seed_mtp1 and bool(getattr(self.config, "mtp_prefill_seed", False))
        if os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_PREFILL_SEED", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            prefill_seed_mtp1 = False
        greedy_decode_burst_steps = max(
            1,
            _config_or_env_int(
                getattr(self, "config", None),
                "greedy_decode_burst_steps",
                "NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS",
                default=1,
            ),
        )
        mtp_static_batch_size = (
            self._mtp_static_batch_size(1)
            if seed_mtp1 and mtp_max_active_rows > 0
            else 0
        )
        prefill_warmup_batch_buckets = tuple(int(bucket) for bucket in batch_buckets)
        if mtp_static_batch_size > 0 and str(getattr(self.config, "prefill_layout", "packed")).lower() == "packed":
            prefill_warmup_batch_buckets = tuple(
                sorted(
                    {
                        mtp_static_batch_size
                        if int(bucket) <= mtp_max_active_rows
                        else int(bucket)
                        for bucket in batch_buckets
                    }
                )
            )
        decode_warmup_batch_buckets = tuple(
            sorted(
                set(int(bucket) for bucket in batch_buckets)
                | ({mtp_static_batch_size} if mtp_static_batch_size > 0 else set())
            )
        )

        for prefill_len in prefill_buckets:
            if self.execution != "jit":
                break
            for batch_size in prefill_warmup_batch_buckets:
                prefill_seed_for_batch = prefill_seed_mtp1 and (
                    mtp_max_active_rows <= 0 or int(batch_size) <= mtp_max_active_rows
                )
                dense_prefill_tokens = int(batch_size) * int(prefill_len)
                max_batched_tokens = int(getattr(self.config, "max_num_batched_tokens", 0) or 0)
                packed_prefill_layout = (
                    str(getattr(self.config, "prefill_layout", "packed")).lower()
                    == "packed"
                )
                if (
                    not packed_prefill_layout
                    and max_batched_tokens > 0
                    and dense_prefill_tokens > max_batched_tokens
                ):
                    summary["prefill_skipped"].append(
                        {
                            "batch_size": int(batch_size),
                            "query_len": int(prefill_len),
                            "dense_prefill_tokens": dense_prefill_tokens,
                            "max_num_batched_tokens": max_batched_tokens,
                            "reason": "dense_prefill_tokens_exceed_budget",
                        }
                    )
                    continue
                if packed_prefill_layout and row_prefill_buckets:
                    max_row_tokens = int(max(row_prefill_buckets))
                    max_reachable_tokens = int(batch_size) * max_row_tokens
                    if int(prefill_len) > max_reachable_tokens:
                        summary["prefill_skipped"].append(
                            {
                                "batch_size": int(batch_size),
                                "query_len": int(prefill_len),
                                "max_row_tokens": max_row_tokens,
                                "max_reachable_tokens": max_reachable_tokens,
                                "reason": "packed_token_bucket_exceeds_row_bucket_capacity",
                            }
                        )
                        continue
                batch = self._dummy_batch(batch_size=batch_size, query_len=prefill_len, is_prefill=True)
                hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                if use_prefill_slot_carry_table and not prefill_seed_for_batch:
                    hybrid_slot_ids = jnp.arange(int(batch_size), dtype=jnp.int32)
                    batch.hybrid_slot_ids_host = tuple(range(int(batch_size)))
                    output = self.executor.forward_prefill_token_ids_slot_carry_table_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state_table=self._hybrid_state_table,
                        hybrid_slot_ids=hybrid_slot_ids,
                        prefill_final_flags=self._prefill_final_flags_device(batch),
                        resident_last_tokens=self._resident_last_tokens,
                    )
                    self._hybrid_state_table = output.hybrid_state
                    if output.resident_last_tokens is not None:
                        self._resident_last_tokens = output.resident_last_tokens
                    route = "forward_prefill_token_ids_slot_carry_table_jit:prefill"
                elif use_hybrid_table_prefill:
                    hybrid_slot_ids = jnp.arange(int(batch_size), dtype=jnp.int32)
                    batch.hybrid_slot_ids_host = tuple(range(int(batch_size)))
                    output = self.executor.forward_prefill_token_ids_table_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        mtp_cache_storage=self.mtp_cache_storage if prefill_seed_for_batch else None,
                        hybrid_state_table=self._hybrid_state_table,
                        hybrid_slot_ids=hybrid_slot_ids,
                        return_mtp_draft=prefill_seed_for_batch,
                    )
                    self._hybrid_state_table = output.hybrid_state
                    if output.mtp_cache_storage is not None:
                        self.mtp_cache_storage = output.mtp_cache_storage
                    route = (
                        "forward_prefill_token_ids_table_jit:prefill-mtp-hidden-seed"
                        if prefill_seed_for_batch
                        else "forward_prefill_token_ids_table_jit:prefill"
                    )
                elif use_greedy_token_fastpath and not prefill_seed_for_batch:
                    output = self.executor.forward_step_token_ids_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=hybrid_state,
                    )
                    route = "forward_step_token_ids_jit:prefill"
                else:
                    output = self.executor.forward_step_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=hybrid_state,
                        return_hidden=prefill_seed_for_batch,
                        return_hidden_with_logits=False,
                        last_logits_only=True,
                    )
                    route = "forward_step_jit:prefill-mtp-hidden-seed" if prefill_seed_for_batch else "forward_step_jit:prefill"
                _block_until_ready(output.activations)
                self.cache_storage = output.cache_storage
                summary["prefill_runs"].append(
                    {
                        "batch_size": int(batch_size),
                        "query_len": int(prefill_len),
                        "tokens_shape": list(batch.tokens.shape),
                        "block_tables_shape": list(batch.block_tables.shape),
                        "num_prefill_tokens": int(batch.num_prefill_tokens),
                        "route": route,
                    }
                )
                if use_sampled_token_fastpath and not prefill_seed_for_batch:
                    sampled_hybrid_state = init_hybrid_state(
                        self.config,
                        batch_size=batch_size,
                        dtype=self.config.get_dtype(),
                    )
                    temperatures = jnp.ones((int(batch_size),), dtype=jnp.float32)
                    rng_slots = jnp.arange(int(batch_size), dtype=jnp.int32)
                    rng_counters = jnp.zeros((int(batch_size),), dtype=jnp.int32)
                    sampled_output = self.executor.forward_step_sampled_token_ids_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=sampled_hybrid_state,
                        temperatures=temperatures,
                        rng_counters=rng_counters,
                        rng_slots=rng_slots,
                    )
                    _block_until_ready(sampled_output.activations)
                    self.cache_storage = sampled_output.cache_storage
                    summary["prefill_runs"].append(
                        {
                            "batch_size": int(batch_size),
                            "query_len": int(prefill_len),
                            "tokens_shape": list(batch.tokens.shape),
                            "block_tables_shape": list(batch.block_tables.shape),
                            "num_prefill_tokens": int(batch.num_prefill_tokens),
                            "route": "forward_step_sampled_token_ids_jit:prefill",
                        }
                    )
                    summary["sampled_token_fastpath_runs"].append(
                        {
                            "kind": "prefill",
                            "batch_size": int(batch_size),
                            "query_len": int(prefill_len),
                            "route": "forward_step_sampled_token_ids_jit:prefill",
                        }
                    )

        for batch_size in decode_warmup_batch_buckets:
            for block_table_width in decode_block_table_buckets:
                batch = self._dummy_batch(
                    batch_size=batch_size,
                    query_len=1,
                    is_prefill=False,
                    max_blocks_per_seq=int(block_table_width),
                )
                warm_decode_mtp = seed_mtp1 and (
                    (
                        mtp_static_batch_size > 0
                        and int(batch_size) == mtp_static_batch_size
                    )
                    or (
                        mtp_static_batch_size <= 0
                        and (mtp_max_active_rows <= 0 or int(batch_size) <= mtp_max_active_rows)
                    )
                )

                def _record_decode_warmup(output, route: str, decode_steps: int = 1) -> None:
                    # MTP and resident-state routes return important cache,
                    # hybrid, draft-token, and commit-selection leaves outside
                    # ``activations``. Blocking only token activations leaves
                    # first-use route work in the measured request even though
                    # the executor JIT cache is already populated.
                    _block_until_ready(output)
                    self.cache_storage = output.cache_storage
                    if getattr(output, "mtp_cache_storage", None) is not None:
                        self.mtp_cache_storage = output.mtp_cache_storage
                    self._sample_fn(
                        jnp.zeros((batch_size, self.config.vocab_size), dtype=jnp.float32),
                        jnp.zeros((batch_size,), dtype=jnp.float32),
                    ).block_until_ready()
                    summary["decode_runs"].append(
                        {
                            "batch_size": int(batch_size),
                            "tokens_shape": list(batch.tokens.shape),
                            "block_tables_shape": list(batch.block_tables.shape),
                            "num_decode_tokens": int(batch.num_decode_tokens),
                            "route": route,
                            "decode_steps": int(decode_steps),
                        }
                    )

                if (
                    use_greedy_token_fastpath
                    and not warm_decode_mtp
                    and greedy_decode_burst_steps > 1
                ):
                    for burst_steps in range(2, greedy_decode_burst_steps + 1):
                        if use_hybrid_table_decode and hasattr(self.executor, "forward_greedy_decode_burst_table_jit"):
                            hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                            output = self.executor.forward_greedy_decode_burst_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                decode_steps=burst_steps,
                            )
                            self._hybrid_state_table = output.hybrid_state
                            _record_decode_warmup(
                                output,
                                "forward_greedy_decode_burst_table_jit:decode",
                                burst_steps,
                            )
                        elif hasattr(self.executor, "forward_greedy_decode_burst_jit"):
                            hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                            output = self.executor.forward_greedy_decode_burst_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=hybrid_state,
                                decode_steps=burst_steps,
                            )
                            _record_decode_warmup(
                                output,
                                "forward_greedy_decode_burst_jit:decode",
                                burst_steps,
                            )
                if use_hybrid_table_decode:
                    hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                    batch.hybrid_slot_ids_host = tuple(range(batch_size))
                    if (
                        self.resident_decode_metadata
                        and hasattr(self.executor, "forward_step_token_ids_resident_dense_slot_carry_jit")
                    ):
                        self._sync_resident_decode_metadata(
                            batch,
                            list(batch.hybrid_slot_ids_host),
                            sync_seq_lens=True,
                        )
                        output = self.executor.forward_step_token_ids_resident_dense_slot_carry_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                            resident_last_tokens=self._resident_last_tokens,
                        )
                        if output.resident_seq_lens is not None:
                            self._resident_seq_lens = output.resident_seq_lens
                            self._advance_resident_seq_lens_host(
                                list(batch.hybrid_slot_ids_host),
                                active_rows=list(range(batch_size)),
                                steps=1,
                            )
                        if output.resident_last_tokens is not None:
                            self._resident_last_tokens = output.resident_last_tokens
                        route = "forward_step_token_ids_resident_dense_slot_carry_jit:decode"
                    elif (
                        self.resident_decode_metadata
                        and hasattr(self.executor, "forward_step_token_ids_resident_slot_carry_jit")
                    ):
                        self._sync_resident_decode_metadata(
                            batch,
                            list(batch.hybrid_slot_ids_host),
                            sync_seq_lens=True,
                        )
                        output = self.executor.forward_step_token_ids_resident_slot_carry_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                            resident_last_tokens=self._resident_last_tokens,
                        )
                        if output.resident_seq_lens is not None:
                            self._resident_seq_lens = output.resident_seq_lens
                            self._advance_resident_seq_lens_host(
                                list(batch.hybrid_slot_ids_host),
                                active_rows=list(range(batch_size)),
                                steps=1,
                            )
                        if output.resident_last_tokens is not None:
                            self._resident_last_tokens = output.resident_last_tokens
                        route = "forward_step_token_ids_resident_slot_carry_jit:decode"
                    elif self.resident_decode_metadata and hasattr(self.executor, "forward_step_token_ids_resident_jit"):
                        self._sync_resident_decode_metadata(
                            batch,
                            list(batch.hybrid_slot_ids_host),
                            sync_seq_lens=True,
                        )
                        output = self.executor.forward_step_token_ids_resident_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                        )
                        if output.resident_seq_lens is not None:
                            self._resident_seq_lens = output.resident_seq_lens
                            self._advance_resident_seq_lens_host(
                                list(batch.hybrid_slot_ids_host),
                                active_rows=list(range(batch_size)),
                                steps=1,
                            )
                        route = "forward_step_token_ids_resident_jit:decode"
                    elif use_slot_carry_table_decode:
                        output = self.executor.forward_step_token_ids_slot_carry_table_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_last_tokens=self._resident_last_tokens,
                        )
                        if output.resident_last_tokens is not None:
                            self._resident_last_tokens = output.resident_last_tokens
                        route = "forward_step_token_ids_slot_carry_table_jit:decode"
                    else:
                        output = self.executor.forward_step_token_ids_table_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                        )
                        route = "forward_step_token_ids_table_jit:decode"
                    self._hybrid_state_table = output.hybrid_state
                    _record_decode_warmup(output, route)
                    if (
                        use_slot_carry_table_decode
                        and not self.resident_decode_metadata
                        and hasattr(self.executor, "forward_step_token_ids_table_jit")
                    ):
                        # The first decode after prefill can arrive before every
                        # row has a resident last-token slot, so the serving hot
                        # path falls back to the plain table boundary once and
                        # then returns to slot-carry table decode. Warm both
                        # generic routes so measured serving never compiles this
                        # real fallback from live request data.
                        table_output = self.executor.forward_step_token_ids_table_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                        )
                        self._hybrid_state_table = table_output.hybrid_state
                        _record_decode_warmup(
                            table_output,
                            "forward_step_token_ids_table_jit:decode-fallback",
                    )
                    if (
                        seed_mtp1
                        and self.resident_decode_metadata
                        and hasattr(self.executor, "forward_step_token_ids_resident_jit")
                    ):
                        # MTP can still fall back to the older resident
                        # metadata decode boundary on seeded-main/boundary
                        # steps. The normal capped-MTP path can also use this
                        # route when dense token carry is not ready, so warm it
                        # for resident metadata configs regardless of whether
                        # this batch will compile an MTP verifier.
                        self._sync_resident_decode_metadata(
                            batch,
                            list(batch.hybrid_slot_ids_host),
                            sync_seq_lens=True,
                        )
                        resident_output = self.executor.forward_step_token_ids_resident_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                        )
                        self._hybrid_state_table = resident_output.hybrid_state
                        if resident_output.resident_seq_lens is not None:
                            self._resident_seq_lens = resident_output.resident_seq_lens
                            self._advance_resident_seq_lens_host(
                                list(batch.hybrid_slot_ids_host),
                                active_rows=list(range(batch_size)),
                                steps=1,
                            )
                        _record_decode_warmup(
                            resident_output,
                            "forward_step_token_ids_resident_jit:decode-resident-fallback",
                        )
                    if (
                        int(batch_size) in padded_decode_buckets
                        and self.resident_decode_metadata
                        and hasattr(self.executor, "forward_step_token_ids_resident_slot_carry_jit")
                    ):
                        inactive_row = int(batch_size) - 1
                        padded_slot_values = tuple(
                            list(range(inactive_row)) + [-1]
                        )
                        padded_batch = self._masked_decode_batch(
                            batch,
                            list(range(inactive_row)),
                        )
                        padded_batch.hybrid_slot_ids_host = padded_slot_values
                        padded_hybrid_slot_ids = jnp.asarray(
                            padded_slot_values,
                            dtype=jnp.int32,
                        )
                        self._sync_resident_decode_metadata(
                            padded_batch,
                            list(padded_batch.hybrid_slot_ids_host),
                            sync_seq_lens=True,
                        )
                        padded_output = self.executor.forward_step_token_ids_resident_slot_carry_jit(
                            padded_batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=padded_hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                            resident_last_tokens=self._resident_last_tokens,
                        )
                        self._hybrid_state_table = padded_output.hybrid_state
                        if padded_output.resident_seq_lens is not None:
                            self._resident_seq_lens = padded_output.resident_seq_lens
                            self._advance_resident_seq_lens_host(
                                list(padded_batch.hybrid_slot_ids_host),
                                active_rows=list(range(inactive_row)),
                                steps=1,
                            )
                        if padded_output.resident_last_tokens is not None:
                            self._resident_last_tokens = padded_output.resident_last_tokens
                        _record_decode_warmup(
                            padded_output,
                            "forward_step_token_ids_resident_slot_carry_jit:decode-inactive-row",
                        )
                else:
                    hybrid_state = init_hybrid_state(self.config, batch_size=batch_size, dtype=self.config.get_dtype())
                    if use_greedy_token_fastpath and not warm_decode_mtp:
                        output = self.executor.forward_step_token_ids_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=hybrid_state,
                        )
                        _record_decode_warmup(output, "forward_step_token_ids_jit:decode")
                    else:
                        output = self.executor.forward_step_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=hybrid_state,
                            return_hidden=warm_decode_mtp,
                            return_hidden_with_logits=False,
                            last_logits_only=True,
                        )
                        _record_decode_warmup(output, "forward_step_jit:decode")
                if warm_decode_mtp and self.num_speculative_tokens >= 1:
                    warm_unverified_fused_append = _unverified_mtp_append_enabled(
                        getattr(self, "config", None),
                        "mtp_unverified_fused_append",
                        "NANO_VLLM_JAX_MTP_UNVERIFIED_FUSED_APPEND",
                    )
                    if warm_unverified_fused_append:
                        if (
                            use_hybrid_table_decode
                            and bool(getattr(self, "resident_decode_metadata", False))
                            and hasattr(self.executor, "forward_step_token_ids_mtp_draft_resident_table_jit")
                        ):
                            hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                            fused_append_output = self.executor.forward_step_token_ids_mtp_draft_resident_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_last_tokens=self._resident_last_tokens,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            self._hybrid_state_table = fused_append_output.hybrid_state
                            if fused_append_output.resident_last_tokens is not None:
                                self._resident_last_tokens = fused_append_output.resident_last_tokens
                            _record_decode_warmup(
                                fused_append_output,
                                "forward_step_token_ids_mtp_draft_resident_table_jit:decode",
                            )
                        elif use_hybrid_table_decode and hasattr(self.executor, "forward_step_token_ids_mtp_draft_table_jit"):
                            hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                            fused_append_output = self.executor.forward_step_token_ids_mtp_draft_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_last_tokens=self._resident_last_tokens,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            self._hybrid_state_table = fused_append_output.hybrid_state
                            if fused_append_output.resident_last_tokens is not None:
                                self._resident_last_tokens = fused_append_output.resident_last_tokens
                            _record_decode_warmup(
                                fused_append_output,
                                "forward_step_token_ids_mtp_draft_table_jit:decode",
                            )
                        elif hasattr(self.executor, "forward_step_token_ids_mtp_draft_jit"):
                            mtp_hybrid_state = init_hybrid_state(
                                self.config,
                                batch_size=batch_size,
                                dtype=self.config.get_dtype(),
                            )
                            fused_append_output = self.executor.forward_step_token_ids_mtp_draft_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            _record_decode_warmup(
                                fused_append_output,
                                "forward_step_token_ids_mtp_draft_jit:decode",
                            )
                        continue
                    seed_output = self.executor.forward_step_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=init_hybrid_state(
                            self.config,
                            batch_size=batch_size,
                            dtype=self.config.get_dtype(),
                        ),
                        return_hidden=True,
                        return_hidden_with_logits=False,
                        last_logits_only=True,
                    )
                    _record_decode_warmup(
                        seed_output,
                        "forward_step_jit:decode-mtp-seed-or-repair",
                    )
                    if self.num_speculative_tokens == 1:
                        reuse_seed_output = self.executor.forward_step_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=init_hybrid_state(
                                self.config,
                                batch_size=batch_size,
                                dtype=self.config.get_dtype(),
                            ),
                            return_hidden=True,
                            return_hidden_with_logits=True,
                            last_logits_only=True,
                        )
                        _record_decode_warmup(
                            reuse_seed_output,
                            "forward_step_jit:decode-mtp-reuse-main",
                        )
                    dummy_hidden = jnp.zeros(
                        (int(batch_size), 1, int(self.config.hidden_size)),
                        dtype=self.config.get_dtype(),
                    )
                    hidden_tokens = self._greedy_tokens_from_hidden(dummy_hidden)
                    _block_until_ready(hidden_tokens)
                    dummy_seed_hidden = self._hidden_for_mtp(dummy_hidden)
                    draft_len = max(1, int(getattr(self, "num_speculative_tokens", 1) or 1))
                    dummy_tokens = jnp.zeros((int(batch_size), 1), dtype=jnp.int32)
                    dummy_positions = jnp.ones((int(batch_size), 1), dtype=jnp.int32)
                    draft_output = self._mtp1_draft_chain(
                        hidden_state=dummy_seed_hidden,
                        token_ids=dummy_tokens,
                        positions=dummy_positions,
                        draft_len=draft_len,
                    )
                    _block_until_ready(draft_output)
                    summary["decode_runs"].append(
                        {
                            "batch_size": int(batch_size),
                            "tokens_shape": [int(batch_size), 1],
                            "block_tables_shape": list(batch.block_tables.shape),
                            "num_decode_tokens": int(batch.num_decode_tokens),
                            "route": "mtp1_seed_helpers:decode",
                            "decode_steps": 1,
                        }
                    )
                    mtp_hybrid_state = init_hybrid_state(
                        self.config,
                        batch_size=batch_size,
                        dtype=self.config.get_dtype(),
                    )
                    verifier_drafts = jnp.zeros((int(batch_size),), dtype=jnp.int32)
                    next_positions = jnp.full((int(batch_size),), 2, dtype=jnp.int32)
                    mtp_burst_groups = max(
                        1,
                        int(
                            os.environ.get(
                                "NANO_VLLM_JAX_MTP_BURST_GROUPS",
                                str(getattr(self, "mtp_burst_groups", 1)),
                            )
                            or "1"
                        ),
                    )
                    if (
                        _unverified_mtp_append_enabled(
                            getattr(self, "config", None),
                            "mtp_unverified_fused_append",
                            "NANO_VLLM_JAX_MTP_UNVERIFIED_FUSED_APPEND",
                        )
                    ):
                        if (
                            use_hybrid_table_decode
                            and bool(getattr(self, "resident_decode_metadata", False))
                            and hasattr(self.executor, "forward_step_token_ids_mtp_draft_resident_table_jit")
                        ):
                            hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                            fused_append_output = self.executor.forward_step_token_ids_mtp_draft_resident_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_last_tokens=self._resident_last_tokens,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            self._hybrid_state_table = fused_append_output.hybrid_state
                            if fused_append_output.resident_last_tokens is not None:
                                self._resident_last_tokens = fused_append_output.resident_last_tokens
                            _record_decode_warmup(
                                fused_append_output,
                                "forward_step_token_ids_mtp_draft_resident_table_jit:decode",
                            )
                        elif use_hybrid_table_decode and hasattr(self.executor, "forward_step_token_ids_mtp_draft_table_jit"):
                            hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                            fused_append_output = self.executor.forward_step_token_ids_mtp_draft_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=hybrid_slot_ids,
                                resident_last_tokens=self._resident_last_tokens,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            self._hybrid_state_table = fused_append_output.hybrid_state
                            if fused_append_output.resident_last_tokens is not None:
                                self._resident_last_tokens = fused_append_output.resident_last_tokens
                            _record_decode_warmup(
                                fused_append_output,
                                "forward_step_token_ids_mtp_draft_table_jit:decode",
                            )
                        elif hasattr(self.executor, "forward_step_token_ids_mtp_draft_jit"):
                            fused_append_output = self.executor.forward_step_token_ids_mtp_draft_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            _record_decode_warmup(
                                fused_append_output,
                                "forward_step_token_ids_mtp_draft_jit:decode",
                            )
                    if hasattr(self.executor, "forward_step_token_ids_mtp_draft_chain_jit"):
                        fused_chain_seed_output = self.executor.forward_step_token_ids_mtp_draft_chain_jit(
                            batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                draft_len=draft_len,
                                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                            )
                        _record_decode_warmup(
                            fused_chain_seed_output,
                            "forward_step_token_ids_mtp_draft_chain_jit:decode",
                        )
                    warmed_mtp_k_decode = False
                    mtp_verifier_impl = str(
                        getattr(self, "mtp_verifier_impl", "two_decode") or "two_decode"
                    ).lower()
                    use_packed_prefix_warmup = mtp_verifier_impl in {
                        "packed_prefix",
                        "packed_prefill",
                        "prefill_packed",
                    }
                    use_generic_k_warmup = (
                        mtp_verifier_impl in {"k_decode", "generic_k", "expanded"}
                        or use_packed_prefix_warmup
                        or os.environ.get(
                            "NANO_VLLM_JAX_MTP_FORCE_GENERIC_K",
                            "0",
                        )
                        in {"1", "true", "yes", "on", "True"}
                    )
                    use_packed_prefix_row_warmup = (
                        use_packed_prefix_warmup
                        and hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
                    )
                    use_packed_prefix_table_warmup = (
                        use_packed_prefix_warmup
                        and bool(getattr(self, "resident_decode_metadata", False))
                        and getattr(self, "_hybrid_state_table", None) is not None
                        and self._hybrid_state_table.conv_state is not None
                        and self._hybrid_state_table.recurrent_state is not None
                        and hasattr(self.executor, "mtp_k_packed_prefix_table_greedy_step_jit")
                    )
                    if use_packed_prefix_warmup and not use_packed_prefix_row_warmup:
                        raise RuntimeError(
                            "mtp_verifier_impl=packed_prefix requires the row-state "
                            "packed verifier; no sequential fallback is allowed"
                        )

                    def _warm_packed_prefix_table_verifier(
                        warm_batch: ScheduledBatch,
                        *,
                        draft_width: int,
                        next_positions_arg: jnp.ndarray,
                        route_suffix: str,
                        emit_bonus: bool = True,
                        burst_groups_arg: int | None = None,
                    ):
                        warm_burst_groups = (
                            mtp_burst_groups
                            if burst_groups_arg is None
                            else max(1, int(burst_groups_arg))
                        )
                        table_hybrid_slot_ids = self._batch_hybrid_slot_ids(warm_batch)
                        table_output = self.executor.mtp_k_packed_prefix_table_greedy_step_jit(
                            warm_batch,
                            cache_storage=self.cache_storage,
                            mtp_cache_storage=self.mtp_cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=table_hybrid_slot_ids,
                            draft_tokens=jnp.zeros(
                                (int(warm_batch.tokens.shape[0]), int(draft_width)),
                                dtype=jnp.int32,
                            ),
                            next_mtp_position=next_positions_arg,
                            mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                            mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                            burst_groups=warm_burst_groups,
                            emit_bonus=emit_bonus,
                            resident_seq_lens=self._resident_seq_lens,
                        )
                        self._hybrid_state_table = table_output.hybrid_state
                        if table_output.mtp_cache_storage is not None:
                            self.mtp_cache_storage = table_output.mtp_cache_storage
                            _record_decode_warmup(
                                table_output,
                                (
                                    "mtp_k_packed_prefix_table_greedy_step_jit:"
                                    f"{route_suffix}:burst{warm_burst_groups}"
                                ),
                            )
                        return table_output
                    if (
                        draft_len == 1
                        and mtp_verifier_impl == "commit_select"
                        and not use_generic_k_warmup
                    ):
                        output = self.executor.mtp1_commit_select_greedy_step_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=mtp_hybrid_state,
                            draft_token=verifier_drafts,
                            next_mtp_position=next_positions,
                            mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                        )
                        _record_decode_warmup(output, "mtp1_commit_select_greedy_step_jit:decode")
                        warm_fast_table_verifier = (
                            os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT", "0")
                            in {"1", "true", "yes", "on", "True"}
                            or os.environ.get("NANO_VLLM_JAX_MTP_GREEDY_BURST_VERIFY_TABLE", "0")
                            in {"1", "true", "yes", "on", "True"}
                            or os.environ.get("NANO_VLLM_JAX_MTP_BURST_VERIFY_TABLE", "0")
                            in {"1", "true", "yes", "on", "True"}
                        )
                        if warm_fast_table_verifier:
                            table_hybrid_slot_ids = self._batch_hybrid_slot_ids(batch)
                            if (
                                os.environ.get("NANO_VLLM_JAX_MTP_BURST_VERIFY_TABLE", "0")
                                in {"1", "true", "yes", "on", "True"}
                                and hasattr(self.executor, "mtp1_burst_verify_table_jit")
                            ):
                                burst_verify_output = self.executor.mtp1_burst_verify_table_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state_table=self._hybrid_state_table,
                                    hybrid_slot_ids=table_hybrid_slot_ids,
                                    draft_token=verifier_drafts,
                                    next_mtp_position=next_positions,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                )
                                _record_decode_warmup(
                                    burst_verify_output,
                                    "mtp1_burst_verify_table_jit:decode",
                                )
                            if (
                                os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT", "0")
                                in {"1", "true", "yes", "on", "True"}
                                and os.environ.get("NANO_VLLM_JAX_MTP_BURST_VERIFY_TABLE", "0")
                                not in {"1", "true", "yes", "on", "True"}
                                and os.environ.get("NANO_VLLM_JAX_MTP_GREEDY_BURST_VERIFY_TABLE", "0")
                                not in {"1", "true", "yes", "on", "True"}
                                and hasattr(self.executor, "mtp1_two_decode_greedy_fast_table_jit")
                            ):
                                fast_table_output = self.executor.mtp1_two_decode_greedy_fast_table_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state_table=self._hybrid_state_table,
                                    hybrid_slot_ids=table_hybrid_slot_ids,
                                    draft_token=verifier_drafts,
                                    next_mtp_position=next_positions,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                )
                                _record_decode_warmup(
                                    fast_table_output,
                                    "mtp1_two_decode_greedy_fast_table_jit:decode",
                                )
                            if hasattr(self.executor, "mtp1_greedy_burst_table_jit"):
                                greedy_burst_output = self.executor.mtp1_greedy_burst_table_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state_table=self._hybrid_state_table,
                                    hybrid_slot_ids=table_hybrid_slot_ids,
                                    draft_token=verifier_drafts,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                )
                                self._hybrid_state_table = greedy_burst_output.hybrid_state
                                _record_decode_warmup(
                                    greedy_burst_output,
                                    "mtp1_greedy_burst_table_jit:decode",
                                )
                    elif (
                        draft_len == 1
                        and use_generic_k_warmup
                        and (
                            hasattr(self.executor, "mtp_k_decode_greedy_step_jit")
                            or hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
                            or (
                                mtp_burst_groups > 1
                                and hasattr(self.executor, "mtp_k_burst_greedy_step_jit")
                            )
                        )
                    ):
                        if (
                            mtp_burst_groups > 1
                            and not use_packed_prefix_warmup
                            and hasattr(self.executor, "mtp_k_burst_greedy_step_jit")
                        ):
                            burst_output = self.executor.mtp_k_burst_greedy_step_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                draft_tokens=jnp.zeros((int(batch_size), 1), dtype=jnp.int32),
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                burst_groups=mtp_burst_groups,
                            )
                            _record_decode_warmup(
                                burst_output,
                                "mtp_k_burst_greedy_step_jit:decode",
                            )
                        if use_packed_prefix_warmup:
                            if use_packed_prefix_table_warmup:
                                output = _warm_packed_prefix_table_verifier(
                                    batch,
                                    draft_width=1,
                                    next_positions_arg=next_positions,
                                    route_suffix="decode",
                                )
                            else:
                                output = self.executor.mtp_k_packed_prefix_greedy_step_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state=mtp_hybrid_state,
                                    draft_tokens=jnp.zeros((int(batch_size), 1), dtype=jnp.int32),
                                    next_mtp_position=next_positions,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                    mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                    mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                    burst_groups=mtp_burst_groups,
                                )
                                _record_decode_warmup(
                                    output,
                                    "mtp_k_packed_prefix_greedy_step_jit:decode",
                                )
                        else:
                            output = self.executor.mtp_k_decode_greedy_step_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                draft_tokens=jnp.zeros((int(batch_size), 1), dtype=jnp.int32),
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                            )
                            _record_decode_warmup(
                                output,
                                "mtp_k_decode_greedy_step_jit:decode",
                            )
                        warmed_mtp_k_decode = True
                    elif draft_len == 1:
                        warmed_exact_table = False
                        if (
                            mtp_verifier_impl == "two_decode"
                            and getattr(self, "_hybrid_state_table", None) is not None
                            and self._hybrid_state_table.conv_state is not None
                            and self._hybrid_state_table.recurrent_state is not None
                            and hasattr(self.executor, "mtp1_two_decode_greedy_table_step_jit")
                        ):
                            table_hybrid_slot_ids = self._batch_hybrid_slot_ids(batch)
                            if (
                                mtp_burst_groups > 1
                                and hasattr(self.executor, "mtp1_two_decode_greedy_table_burst_step_jit")
                            ):
                                burst_output = self.executor.mtp1_two_decode_greedy_table_burst_step_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state_table=self._hybrid_state_table,
                                    hybrid_slot_ids=table_hybrid_slot_ids,
                                    draft_token=verifier_drafts,
                                    next_mtp_position=next_positions,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                    burst_groups=mtp_burst_groups,
                                )
                                _record_decode_warmup(
                                    burst_output,
                                    "mtp1_two_decode_greedy_table_burst_step_jit:decode",
                                )
                                self._hybrid_state_table = burst_output.hybrid_state
                                self._mark_hybrid_slots_written(list(batch.hybrid_slot_ids_host or ()))
                            output = self.executor.mtp1_two_decode_greedy_table_step_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=table_hybrid_slot_ids,
                                draft_token=verifier_drafts,
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            _record_decode_warmup(
                                output,
                                "mtp1_two_decode_greedy_table_step_jit:decode",
                            )
                            self._hybrid_state_table = output.hybrid_state
                            self._mark_hybrid_slots_written(list(batch.hybrid_slot_ids_host or ()))
                            warmed_exact_table = True
                        if not warmed_exact_table:
                            output = self.executor.mtp1_two_decode_greedy_step_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                draft_token=verifier_drafts,
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            _record_decode_warmup(output, "mtp1_two_decode_greedy_step_jit:decode")
                        if (
                            os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT", "0")
                            in {"1", "true", "yes", "on", "True"}
                            and hasattr(self.executor, "mtp1_two_decode_greedy_fast_table_jit")
                        ):
                            table_hybrid_slot_ids = self._batch_hybrid_slot_ids(batch)
                            if hasattr(self.executor, "mtp1_greedy_burst_table_jit"):
                                greedy_burst_output = self.executor.mtp1_greedy_burst_table_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state_table=self._hybrid_state_table,
                                    hybrid_slot_ids=table_hybrid_slot_ids,
                                    draft_token=verifier_drafts,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                )
                                self._hybrid_state_table = greedy_burst_output.hybrid_state
                                _record_decode_warmup(
                                    greedy_burst_output,
                                    "mtp1_greedy_burst_table_jit:decode",
                                )
                            table_fast_output = self.executor.mtp1_two_decode_greedy_fast_table_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state_table=self._hybrid_state_table,
                                hybrid_slot_ids=table_hybrid_slot_ids,
                                draft_token=verifier_drafts,
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            _record_decode_warmup(
                                table_fast_output,
                                "mtp1_two_decode_greedy_fast_table_jit:decode",
                            )
                        if (
                            os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT", "0")
                            in {"1", "true", "yes", "on", "True"}
                            and hasattr(self.executor, "mtp1_two_decode_greedy_fast_step_jit")
                        ):
                            fast_output = self.executor.mtp1_two_decode_greedy_fast_step_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=init_hybrid_state(
                                    self.config,
                                    batch_size=batch_size,
                                    dtype=self.config.get_dtype(),
                                ),
                                draft_token=verifier_drafts,
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            )
                            _record_decode_warmup(
                                fast_output,
                                "mtp1_two_decode_greedy_fast_step_jit:decode",
                            )
                    elif (
                        draft_len == 2
                        and mtp_burst_groups <= 1
                        and mtp_verifier_impl == "commit_select"
                        and not use_generic_k_warmup
                        and hasattr(self.executor, "mtp2_commit_select_greedy_step_jit")
                    ):
                        output = self.executor.mtp2_commit_select_greedy_step_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=mtp_hybrid_state,
                            draft_tokens=jnp.zeros((int(batch_size), 2), dtype=jnp.int32),
                            next_mtp_position=next_positions,
                            mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                            mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                        )
                        _record_decode_warmup(output, "mtp2_commit_select_greedy_step_jit:decode")
                    elif (
                        hasattr(self.executor, "mtp_k_decode_greedy_step_jit")
                        or hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
                        or (
                            mtp_burst_groups > 1
                            and not use_packed_prefix_warmup
                            and hasattr(self.executor, "mtp_k_burst_greedy_step_jit")
                        )
                    ):
                        if (
                            (mtp_burst_groups <= 1 or use_packed_prefix_warmup)
                            and (
                                hasattr(self.executor, "mtp_k_decode_greedy_step_jit")
                                or hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
                            )
                        ):
                            if use_packed_prefix_warmup:
                                if use_packed_prefix_table_warmup:
                                    output = _warm_packed_prefix_table_verifier(
                                        batch,
                                        draft_width=draft_len,
                                        next_positions_arg=next_positions,
                                        route_suffix="decode",
                                    )
                                    if draft_len > 1:
                                        if mtp_burst_groups > 1:
                                            _warm_packed_prefix_table_verifier(
                                                batch,
                                                draft_width=draft_len,
                                                next_positions_arg=next_positions,
                                                route_suffix="decode-tail-burst1",
                                                emit_bonus=True,
                                                burst_groups_arg=1,
                                            )
                                        for tail_width in range(1, draft_len + 1):
                                            _warm_packed_prefix_table_verifier(
                                                batch,
                                                draft_width=tail_width,
                                                next_positions_arg=next_positions,
                                                route_suffix=f"decode-no-bonus-k{tail_width}",
                                                emit_bonus=False,
                                                burst_groups_arg=1,
                                            )
                                else:
                                    output = self.executor.mtp_k_packed_prefix_greedy_step_jit(
                                        batch,
                                        cache_storage=self.cache_storage,
                                        hybrid_state=mtp_hybrid_state,
                                        draft_tokens=jnp.zeros((int(batch_size), draft_len), dtype=jnp.int32),
                                        next_mtp_position=next_positions,
                                        mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                        mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                        mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                        burst_groups=mtp_burst_groups,
                                    )
                                    _record_decode_warmup(
                                        output,
                                        "mtp_k_packed_prefix_greedy_step_jit:decode",
                                    )
                            else:
                                output = self.executor.mtp_k_decode_greedy_step_jit(
                                    batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state=mtp_hybrid_state,
                                    draft_tokens=jnp.zeros((int(batch_size), draft_len), dtype=jnp.int32),
                                    next_mtp_position=next_positions,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                    mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                    mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                )
                                _record_decode_warmup(
                                    output,
                                    "mtp_k_decode_greedy_step_jit:decode",
                                )
                            warmed_mtp_k_decode = True
                        if (
                            mtp_burst_groups > 1
                            and not use_packed_prefix_warmup
                            and hasattr(self.executor, "mtp_k_burst_greedy_step_jit")
                        ):
                            burst_output = self.executor.mtp_k_burst_greedy_step_jit(
                                batch,
                                cache_storage=self.cache_storage,
                                hybrid_state=mtp_hybrid_state,
                                draft_tokens=jnp.zeros((int(batch_size), draft_len), dtype=jnp.int32),
                                next_mtp_position=next_positions,
                                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                burst_groups=mtp_burst_groups,
                            )
                            _record_decode_warmup(
                                burst_output,
                                "mtp_k_burst_greedy_step_jit:decode",
                            )
                    if (
                        draft_len > 1
                        and use_generic_k_warmup
                        and int(batch_size) > 1
                        and (
                            hasattr(self.executor, "mtp_k_decode_greedy_step_jit")
                            or hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
                        )
                    ):
                        compact_row_counts = tuple(range(1, int(batch_size)))
                        if use_packed_prefix_warmup:
                            # The strict packed-prefix serving path masks
                            # inactive tail rows inside the physical bucket.
                            # Warm that exact shape rather than a compact row
                            # batch, otherwise measured serving will compile
                            # on the final partial step.
                            compact_row_counts = (int(batch_size) - 1,)
                        for compact_rows in compact_row_counts:
                            if use_packed_prefix_warmup:
                                masked_tail_batch = self._masked_decode_batch(
                                    batch,
                                    list(range(compact_rows)),
                                )
                                masked_tail_positions = jnp.zeros((int(batch_size),), dtype=jnp.int32)
                                masked_tail_positions = masked_tail_positions.at[:compact_rows].set(2)
                                if use_packed_prefix_table_warmup:
                                    _warm_packed_prefix_table_verifier(
                                        masked_tail_batch,
                                        draft_width=draft_len,
                                        next_positions_arg=masked_tail_positions,
                                        route_suffix=f"decode-masked-tail-{compact_rows}",
                                    )
                                    if draft_len > 1:
                                        if mtp_burst_groups > 1:
                                            _warm_packed_prefix_table_verifier(
                                                masked_tail_batch,
                                                draft_width=draft_len,
                                                next_positions_arg=masked_tail_positions,
                                                route_suffix=(
                                                    f"decode-masked-tail-{compact_rows}-"
                                                    "tail-burst1"
                                                ),
                                                emit_bonus=True,
                                                burst_groups_arg=1,
                                            )
                                        for tail_width in range(1, draft_len + 1):
                                            _warm_packed_prefix_table_verifier(
                                                masked_tail_batch,
                                                draft_width=tail_width,
                                                next_positions_arg=masked_tail_positions,
                                                route_suffix=(
                                                    f"decode-masked-tail-{compact_rows}-"
                                                    f"no-bonus-k{tail_width}"
                                                ),
                                                emit_bonus=False,
                                                burst_groups_arg=1,
                                            )
                                else:
                                    masked_tail_hybrid_state = self._batch_hybrid_state(
                                        masked_tail_batch
                                    )
                                    masked_tail_output = self.executor.mtp_k_packed_prefix_greedy_step_jit(
                                        masked_tail_batch,
                                        cache_storage=self.cache_storage,
                                        hybrid_state=masked_tail_hybrid_state,
                                        draft_tokens=jnp.zeros(
                                            (int(batch_size), draft_len),
                                            dtype=jnp.int32,
                                        ),
                                        next_mtp_position=masked_tail_positions,
                                        mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                        mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                        mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                        burst_groups=mtp_burst_groups,
                                    )
                                    _record_decode_warmup(
                                        masked_tail_output,
                                        f"mtp_k_packed_prefix_greedy_step_jit:decode-masked-tail-{compact_rows}",
                                    )
                            else:
                                compact_batch = self._compact_decode_batch(
                                    batch,
                                    list(range(compact_rows)),
                                )
                                compact_hybrid_state = init_hybrid_state(
                                    self.config,
                                    batch_size=compact_rows,
                                    dtype=self.config.get_dtype(),
                                )
                                compact_next_positions = jnp.full(
                                    (compact_rows,),
                                    2,
                                    dtype=jnp.int32,
                                )
                                compact_output = self.executor.mtp_k_decode_greedy_step_jit(
                                    compact_batch,
                                    cache_storage=self.cache_storage,
                                    hybrid_state=compact_hybrid_state,
                                    draft_tokens=jnp.zeros(
                                        (compact_rows, draft_len),
                                        dtype=jnp.int32,
                                    ),
                                    next_mtp_position=compact_next_positions,
                                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                                    mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                                    mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                                )
                                _record_decode_warmup(
                                    compact_output,
                                    f"mtp_k_decode_greedy_step_jit:decode-compact-{compact_rows}",
                                )
                    if (
                        draft_len > 1
                        and not warmed_mtp_k_decode
                        and not use_packed_prefix_warmup
                        and hasattr(self.executor, "mtp_k_decode_greedy_step_jit")
                    ):
                        # Warm the generic grouped verifier for K>1 shapes even
                        # when a more specific verifier was warmed above.
                        generic_output = self.executor.mtp_k_decode_greedy_step_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state=mtp_hybrid_state,
                            draft_tokens=jnp.zeros((int(batch_size), draft_len), dtype=jnp.int32),
                            next_mtp_position=next_positions,
                            mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                            mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                            mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                        )
                        _record_decode_warmup(
                            generic_output,
                            "mtp_k_decode_greedy_step_jit:decode-generic",
                        )
                if use_sampled_token_fastpath and not warm_decode_mtp:
                    temperatures = jnp.ones((int(batch_size),), dtype=jnp.float32)
                    sampled_hybrid_state = init_hybrid_state(
                        self.config,
                        batch_size=batch_size,
                        dtype=self.config.get_dtype(),
                    )
                    rng_slots = jnp.arange(int(batch_size), dtype=jnp.int32)
                    rng_counters = jnp.zeros((int(batch_size),), dtype=jnp.int32)
                    sampled_output = self.executor.forward_step_sampled_token_ids_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state=sampled_hybrid_state,
                        temperatures=temperatures,
                        rng_counters=rng_counters,
                        rng_slots=rng_slots,
                    )
                    sampled_route = "forward_step_sampled_token_ids_jit:decode"
                    _record_decode_warmup(sampled_output, sampled_route)
                    summary["sampled_token_fastpath_runs"].append(
                        {
                            "kind": "decode",
                            "batch_size": int(batch_size),
                            "block_tables_shape": list(batch.block_tables.shape),
                            "route": sampled_route,
                        }
                    )
                    if use_sampled_resident_dense_decode:
                        hybrid_slot_ids = jnp.arange(batch_size, dtype=jnp.int32)
                        batch.hybrid_slot_ids_host = tuple(range(batch_size))
                        self._sync_resident_decode_metadata(
                            batch,
                            list(batch.hybrid_slot_ids_host),
                            sync_seq_lens=True,
                        )
                        sampled_output = self.executor.forward_step_sampled_token_ids_resident_dense_slot_carry_jit(
                            batch,
                            cache_storage=self.cache_storage,
                            hybrid_state_table=self._hybrid_state_table,
                            hybrid_slot_ids=hybrid_slot_ids,
                            resident_block_tables=self._resident_block_tables,
                            resident_seq_lens=self._resident_seq_lens,
                            resident_last_tokens=self._resident_last_tokens,
                            resident_rng_counters=self._resident_rng_counters,
                            temperatures=temperatures,
                        )
                        self._hybrid_state_table = sampled_output.hybrid_state
                        if sampled_output.resident_seq_lens is not None:
                            self._resident_seq_lens = sampled_output.resident_seq_lens
                            self._advance_resident_seq_lens_host(
                                list(batch.hybrid_slot_ids_host),
                                active_rows=list(range(batch_size)),
                                steps=1,
                            )
                        if sampled_output.resident_last_tokens is not None:
                            self._resident_last_tokens = sampled_output.resident_last_tokens
                        if sampled_output.resident_rng_counters is not None:
                            self._resident_rng_counters = sampled_output.resident_rng_counters
                        sampled_route = "forward_step_sampled_token_ids_resident_dense_slot_carry_jit:decode"
                        _record_decode_warmup(sampled_output, sampled_route)
                        summary["sampled_token_fastpath_runs"].append(
                            {
                                "kind": "decode",
                                "batch_size": int(batch_size),
                                "block_tables_shape": list(batch.block_tables.shape),
                                "route": sampled_route,
                            }
                        )
        if bool(getattr(self, "resident_decode_metadata", False)):
            for row_count in range(1, int(max(batch_buckets)) + 1):
                if row_count > int(self._resident_block_tables.shape[0]):
                    break
                slots = jnp.arange(row_count, dtype=jnp.int32)
                block_rows = jnp.zeros(
                    (row_count, int(self._resident_block_tables.shape[1])),
                    dtype=jnp.int32,
                )
                seq_lens = jnp.zeros((row_count,), dtype=jnp.int32)
                token_rows = jnp.arange(row_count, dtype=jnp.int32)
                last_tokens = jnp.zeros((row_count, 1), dtype=jnp.int32)
                self._resident_block_tables = self._scatter_resident_block_table_rows(
                    self._resident_block_tables,
                    slots,
                    block_rows,
                )
                self._resident_seq_lens = self._scatter_resident_seq_lens(
                    self._resident_seq_lens,
                    slots,
                    seq_lens,
                )
                self._resident_last_tokens = self._scatter_resident_last_tokens(
                    self._resident_last_tokens,
                    slots,
                    last_tokens,
                    token_rows,
                )
                _block_until_ready(self._resident_block_tables)
                _block_until_ready(self._resident_seq_lens)
                _block_until_ready(self._resident_last_tokens)
                summary["resident_metadata_scatter_runs"].append(
                    {
                        "row_count": int(row_count),
                        "block_rows_shape": list(block_rows.shape),
                        "seq_lens_shape": list(seq_lens.shape),
                        "last_tokens_shape": list(last_tokens.shape),
                    }
                )
        self._reset_runtime_state_after_warmup()
        if hasattr(self, "_hybrid_state_table"):
            _block_until_ready(self._hybrid_state_table)
        if hasattr(self, "_resident_block_tables"):
            _block_until_ready(self._resident_block_tables)
        if hasattr(self, "_resident_seq_lens"):
            _block_until_ready(self._resident_seq_lens)
        if hasattr(self, "_resident_last_tokens"):
            _block_until_ready(self._resident_last_tokens)
        if hasattr(self, "_resident_rng_counters"):
            _block_until_ready(self._resident_rng_counters)
        summary["state_reset_after_warmup"] = True
        self._warmup_compiled = True
        summary["jit_cache_entries_after"] = _cache_entries()
        return summary

    def _dummy_batch(
        self,
        *,
        batch_size: int,
        query_len: int,
        is_prefill: bool,
        max_blocks_per_seq: int | None = None,
    ) -> ScheduledBatch:
        block_tables = []
        num_blocks = max(1, int(getattr(self.config, "num_kvcache_blocks", 1) or 1))
        block_table_width = int(max_blocks_per_seq or self.max_blocks_per_seq)
        for row in range(batch_size):
            start = row * block_table_width
            block_tables.append(
                [
                    (start + offset) % num_blocks
                    for offset in range(block_table_width)
                ]
            )
        if is_prefill and str(getattr(self.config, "prefill_layout", "packed")).lower() == "packed":
            token_bucket = int(query_len)
            base = token_bucket // batch_size
            rem = token_bucket % batch_size
            query_lens = [base + (1 if row < rem else 0) for row in range(batch_size)]
            query_start_loc = [0]
            packed_positions = []
            token_row_ids = []
            for row, qlen in enumerate(query_lens):
                query_start_loc.append(query_start_loc[-1] + qlen)
                packed_positions.extend(range(qlen))
                token_row_ids.extend([row] * qlen)
            return ScheduledBatch(
                tokens=jnp.zeros((1, token_bucket), dtype=jnp.int32),
                positions=jnp.array([packed_positions], dtype=jnp.int32),
                seq_ids=jnp.arange(batch_size, dtype=jnp.int32),
                query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
                is_prefill=True,
                num_prefill_tokens=token_bucket,
                num_decode_tokens=0,
                block_tables=jnp.array(block_tables, dtype=jnp.int32),
                seq_lens=jnp.array(query_lens, dtype=jnp.int32),
                seq_ids_host=tuple(range(batch_size)),
                query_lens_host=tuple(query_lens),
                seq_lens_host=tuple(query_lens),
                block_tables_host=tuple(tuple(int(block) for block in row) for row in block_tables),
                packed_prefill=True,
                token_row_ids=jnp.array([token_row_ids], dtype=jnp.int32),
            )

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
            seq_ids_host=tuple(range(batch_size)),
            query_lens_host=tuple(query_lens),
            seq_lens_host=tuple([query_len if is_prefill else 1] * batch_size),
            block_tables_host=tuple(tuple(int(block) for block in row) for row in block_tables),
        )

    def release(self, seq_ids: List[int]):
        """Release per-sequence hybrid state once a request is finished."""
        released_slots: list[int] = []
        for seq_id in seq_ids:
            self.hybrid_states.pop(seq_id, None)
            slot = self._hybrid_slots.pop(seq_id, None)
            if slot is not None:
                self._free_hybrid_slots.append(slot)
                released_slots.append(int(slot))
                if hasattr(self, "_resident_block_tables_host"):
                    self._resident_block_tables_host[slot] = tuple(
                        0 for _ in range(self.max_blocks_per_seq)
                    )
                if hasattr(self, "_resident_block_counts_host"):
                    self._resident_block_counts_host[slot] = 0
                if hasattr(self, "_resident_seq_lens_host"):
                    self._resident_seq_lens_host[slot] = 0
                if hasattr(self, "_resident_rng_counters"):
                    reset_slots = getattr(
                        self,
                        "_resident_rng_counter_reset_slots",
                        None,
                    )
                    if reset_slots is None:
                        reset_slots = set()
                        self._resident_rng_counter_reset_slots = reset_slots
                    reset_slots.add(int(slot))
            self._mtp1_drafts.pop(seq_id, None)
            if hasattr(self, "_resident_last_tokens_stale_seq_ids"):
                self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        carry_by_seq_id = getattr(self, "_device_token_carry_by_seq_id", {})
        if carry_by_seq_id and any(seq_id in carry_by_seq_id for seq_id in seq_ids):
            finished_seq_ids = {int(seq_id) for seq_id in seq_ids}
            remaining_carry = {
                int(seq_id): token_ref
                for seq_id, token_ref in carry_by_seq_id.items()
                if int(seq_id) not in finished_seq_ids
            }
            if remaining_carry:
                self._device_token_carry_seq_ids = None
                self._device_token_carry_tokens = None
                self._device_token_carry_by_seq_id = remaining_carry
            else:
                self._clear_device_token_carry()

    def _reset_runtime_state_after_warmup(self) -> None:
        """Drop dummy warmup sequence state while keeping compiled executables."""
        if hasattr(self, "hybrid_states"):
            self.hybrid_states.clear()
        if hasattr(self, "_hybrid_slots"):
            self._hybrid_slots.clear()
        if hasattr(self, "_max_hybrid_slots"):
            self._free_hybrid_slots = list(range(self._max_hybrid_slots))
            self._zeroed_hybrid_slots = set()
            self._zero_hybrid_slots(tuple(range(self._max_hybrid_slots)))
            self._zeroed_hybrid_slots = set(range(self._max_hybrid_slots))
        if hasattr(self, "_mtp1_drafts"):
            self._mtp1_drafts.clear()
        if hasattr(self, "_mtp1_seeded_chain"):
            self._mtp1_seeded_chain.clear()
        if hasattr(self, "_clear_device_token_carry"):
            self._clear_device_token_carry()
        if hasattr(self, "cache_storage"):
            self.cache_storage = KVCacheStorage(
                k_cache=jnp.zeros_like(self.cache_storage.k_cache),
                v_cache=jnp.zeros_like(self.cache_storage.v_cache),
            )
            if hasattr(self, "kv_state"):
                self.kv_state = replace(
                    self.kv_state,
                    k_cache=self.cache_storage.k_cache,
                    v_cache=self.cache_storage.v_cache,
                )
        if hasattr(self, "mtp_cache_storage") and self.mtp_cache_storage is not None:
            self.mtp_cache_storage = KVCacheStorage(
                k_cache=jnp.zeros_like(self.mtp_cache_storage.k_cache),
                v_cache=jnp.zeros_like(self.mtp_cache_storage.v_cache),
            )
        if hasattr(self, "_resident_block_tables"):
            self._resident_block_tables = jnp.zeros_like(self._resident_block_tables)
        if hasattr(self, "_resident_seq_lens"):
            self._resident_seq_lens = jnp.zeros_like(self._resident_seq_lens)
        if hasattr(self, "_resident_last_tokens"):
            self._resident_last_tokens = jnp.zeros_like(self._resident_last_tokens)
        if hasattr(self, "_resident_rng_counters"):
            self._resident_rng_counters = jnp.zeros_like(self._resident_rng_counters)
        if hasattr(self, "_resident_rng_counter_reset_slots"):
            self._resident_rng_counter_reset_slots.clear()
        if hasattr(self, "_resident_last_tokens_stale_seq_ids"):
            self._resident_last_tokens_stale_seq_ids.clear()
        if hasattr(self, "_resident_block_tables_host") and hasattr(self, "_max_hybrid_slots"):
            self._resident_block_tables_host = [
                tuple(0 for _ in range(self.max_blocks_per_seq))
                for _ in range(self._max_hybrid_slots)
            ]
        if hasattr(self, "_resident_block_counts_host") and hasattr(self, "_max_hybrid_slots"):
            self._resident_block_counts_host = [0 for _ in range(self._max_hybrid_slots)]
        if hasattr(self, "_resident_seq_lens_host") and hasattr(self, "_max_hybrid_slots"):
            self._resident_seq_lens_host = [0 for _ in range(self._max_hybrid_slots)]

    def _clear_device_token_carry(self) -> None:
        self._device_token_carry_seq_ids = None
        self._device_token_carry_tokens = None
        self._device_token_carry_by_seq_id = {}
        self._device_seq_lens_carry_seq_ids = None
        self._device_seq_lens_carry = None

    @staticmethod
    def _active_decode_rows_host(batch: ScheduledBatch) -> List[int]:
        if batch.seq_ids_host is None or batch.query_lens_host is None:
            return []
        return [
            row
            for row, (seq_id, query_len) in enumerate(zip(batch.seq_ids_host, batch.query_lens_host))
            if int(seq_id) >= 0 and int(query_len) > 0
        ]

    def _maybe_apply_device_token_carry(self, batch: ScheduledBatch) -> ScheduledBatch:
        static_decode_metadata = bool(getattr(batch, "uses_static_decode_metadata", False))
        active_rows = self._active_decode_rows_host(batch)
        carry_enabled = bool(
            getattr(
                self,
                "device_token_carry",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "device_token_carry",
                    "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
                ),
            )
        )
        if (
            carry_enabled
            and batch.is_prefill
            and batch.packed_prefill
            and getattr(batch, "mixed_prefill_decode", False)
            and getattr(self, "_device_token_carry_by_seq_id", {})
            and batch.seq_ids_host is not None
            and batch.query_lens_host is not None
            and batch.tokens.shape[0] == 1
        ):
            tokens = batch.tokens
            offset = 0
            applied = False
            for row, (seq_id, query_len) in enumerate(zip(batch.seq_ids_host, batch.query_lens_host)):
                query_len = int(query_len)
                if int(seq_id) >= 0 and query_len == 1:
                    token_ref = self._device_token_carry_by_seq_id.get(int(seq_id))
                    if token_ref is not None:
                        token_array = jnp.asarray(token_ref.tokens, dtype=jnp.int32)
                        if token_array.ndim == 2 and token_array.shape[1] == 1:
                            token_value = token_array[int(token_ref.row), 0]
                        else:
                            token_value = _int32_device_vector(token_array)[int(token_ref.row)]
                        tokens = tokens.at[0, offset].set(token_value)
                        applied = True
                offset += query_len
            if applied:
                return replace(batch, tokens=tokens)

        if (
            not carry_enabled
            or batch.is_prefill
            or not getattr(self, "_device_token_carry_by_seq_id", {})
            or batch.seq_ids_host is None
            or batch.tokens.shape[1] != 1
        ):
            if static_decode_metadata:
                raise RuntimeError("static decode metadata requires a device-token carry for every active row")
            return batch

        carried_seq_ids = getattr(self, "_device_token_carry_seq_ids", None)
        carried_tokens = getattr(self, "_device_token_carry_tokens", None)
        carried_seq_lens_ids = getattr(self, "_device_seq_lens_carry_seq_ids", None)
        carried_seq_lens = getattr(self, "_device_seq_lens_carry", None)
        use_seq_lens_carry = bool(
            getattr(
                self,
                "static_decode_seq_lens_carry",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "static_decode_seq_lens_carry",
                    "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
                ),
            )
        )
        tokens = batch.tokens
        seq_lens = batch.seq_lens

        if (
            carried_seq_ids is not None
            and tuple(batch.seq_ids_host) == carried_seq_ids
            and carried_tokens is not None
        ):
            token_array = jnp.asarray(carried_tokens, dtype=jnp.int32)
            if tuple(token_array.shape) == tuple(tokens.shape):
                tokens = token_array
                applied = True
            else:
                token_vector = _int32_device_vector(token_array)
                if token_vector.shape[0] == int(tokens.shape[0]):
                    tokens = jnp.reshape(token_vector, tokens.shape)
                    applied = True
                else:
                    applied = False
        else:
            applied = False

        missing_static_rows: List[int] = []
        if not applied:
            for row, seq_id in enumerate(batch.seq_ids_host):
                token_ref = self._device_token_carry_by_seq_id.get(int(seq_id))
                if token_ref is None:
                    if static_decode_metadata and row in active_rows:
                        missing_static_rows.append(row)
                    continue
                token_array = jnp.asarray(token_ref.tokens, dtype=jnp.int32)
                if token_array.ndim == 2 and token_array.shape[1] == 1:
                    tokens = tokens.at[row, 0].set(token_array[int(token_ref.row), 0])
                else:
                    token_vector = _int32_device_vector(token_array)
                    tokens = tokens.at[row, 0].set(token_vector[int(token_ref.row)])
                applied = True
        if missing_static_rows:
            raise RuntimeError(
                "static decode metadata is missing device-token carry rows "
                f"{tuple(missing_static_rows)}"
            )
        if not applied:
            if static_decode_metadata:
                raise RuntimeError("static decode metadata did not apply any device-token carry")
            return batch

        seq_lens_applied = False
        if use_seq_lens_carry:
            if (
                carried_seq_lens_ids is not None
                and tuple(batch.seq_ids_host) == carried_seq_lens_ids
                and carried_seq_lens is not None
            ):
                seq_lens_vector = _int32_device_vector(carried_seq_lens)
                if seq_lens_vector.shape[0] == int(seq_lens.shape[0]):
                    seq_lens = seq_lens_vector
                    seq_lens_applied = True
            if not seq_lens_applied and carried_seq_lens is not None and carried_seq_lens_ids is not None:
                seq_lens_vector = _int32_device_vector(carried_seq_lens)
                seq_id_to_row = {int(seq_id): row for row, seq_id in enumerate(carried_seq_lens_ids)}
                for row, seq_id in enumerate(batch.seq_ids_host):
                    source_row = seq_id_to_row.get(int(seq_id))
                    if source_row is None:
                        continue
                    seq_lens = seq_lens.at[row].set(seq_lens_vector[source_row])
                    seq_lens_applied = True
        if (
            static_decode_metadata
            and use_seq_lens_carry
            and not seq_lens_applied
            and carried_seq_lens is not None
        ):
            raise RuntimeError("static decode metadata requires carried device seq_lens")
        return replace(batch, tokens=tokens, seq_lens=seq_lens)

    def _resident_slot_token_decode_ready(
        self,
        batch: ScheduledBatch,
        *,
        active_rows: list[int],
    ) -> bool:
        required_method = (
            "forward_step_token_ids_resident_slot_carry_jit"
            if bool(getattr(self, "resident_decode_metadata", False))
            else "forward_step_token_ids_slot_carry_table_jit"
        )
        if (
            batch.is_prefill
            or not bool(getattr(batch, "uses_static_decode_metadata", False))
            or batch.seq_ids_host is None
            or not active_rows
            or not hasattr(self, "_resident_last_tokens")
            or not hasattr(self.executor, required_method)
        ):
            return False
        carry_by_seq_id = getattr(self, "_device_token_carry_by_seq_id", {})
        if not carry_by_seq_id:
            return False
        hybrid_slots = getattr(self, "_hybrid_slots", {})
        stale_seq_ids = getattr(self, "_resident_last_tokens_stale_seq_ids", set())
        for row in active_rows:
            seq_id = int(batch.seq_ids_host[row])
            if seq_id in stale_seq_ids:
                return False
            if seq_id < 0 or seq_id not in carry_by_seq_id or seq_id not in hybrid_slots:
                return False
        return True

    def _resident_slot_token_dense_decode_ready(
        self,
        batch: ScheduledBatch,
        *,
        active_rows: list[int],
    ) -> bool:
        if not self._resident_slot_token_decode_ready(batch, active_rows=active_rows):
            return False
        batch_size = int(batch.tokens.shape[0])
        if active_rows != list(range(batch_size)):
            return False
        query_lens = (
            list(batch.query_lens_host)
            if batch.query_lens_host is not None
            else [int(x) for x in batch.query_lens[:batch_size].tolist()]
        )
        if len(query_lens) < batch_size or any(int(query_lens[row]) != 1 for row in range(batch_size)):
            return False
        seq_ids = list(batch.seq_ids_host or ())
        if len(seq_ids) != batch_size:
            return False
        hybrid_slots = getattr(self, "_hybrid_slots", {})
        slot_values = [int(hybrid_slots.get(int(seq_id), -1)) for seq_id in seq_ids]
        return all(slot >= 0 for slot in slot_values) and len(set(slot_values)) == len(slot_values)

    def _record_resident_last_tokens(
        self,
        batch: ScheduledBatch,
        token_ids: jnp.ndarray,
        *,
        eligible_rows: list[int],
        active_row_to_token_row: dict[int, int],
        full_batch_tokens: bool,
    ) -> None:
        if (
            not eligible_rows
            or not hasattr(self, "_resident_last_tokens")
            or batch.seq_ids_host is None
        ):
            return
        slot_values = list(batch.hybrid_slot_ids_host or ())
        if not slot_values:
            slot_values = [
                self._hybrid_slots.get(int(seq_id), -1)
                for seq_id in batch.seq_ids_host
            ]
        slots: list[int] = []
        token_rows: list[int] = []
        for row in eligible_rows:
            if row >= len(slot_values):
                continue
            slot = int(slot_values[row])
            if slot < 0:
                continue
            token_row = row if full_batch_tokens else active_row_to_token_row[row]
            slots.append(slot)
            token_rows.append(int(token_row))
        if not slots:
            return
        self._resident_last_tokens = self._scatter_resident_last_tokens(
            self._resident_last_tokens,
            self._resident_update_slots_device(slots),
            token_ids,
            jax.device_put(np.asarray(token_rows, dtype=np.int32)),
        )

    def _record_device_token_carry(
        self,
        batch: ScheduledBatch,
        token_ids: jnp.ndarray,
        *,
        active_rows: list[int],
        prefill_final_flags: list[bool],
        seqs: List[Sequence],
        update_resident_tokens: bool = True,
        resident_tokens_already_current: bool = False,
    ) -> None:
        if (
            not bool(
                getattr(
                    self,
                    "device_token_carry",
                    _config_or_env_flag(
                        getattr(self, "config", None),
                        "device_token_carry",
                        "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
                    ),
                )
            )
            or batch.seq_ids_host is None
            or not active_rows
            or any(row >= len(seqs) or not seqs[row].ignore_eos for row in active_rows)
        ):
            self._clear_device_token_carry()
            return
        eligible_rows = active_rows
        if batch.is_prefill:
            eligible_rows = [
                row
                for row in active_rows
                if row < len(prefill_final_flags) and prefill_final_flags[row]
            ]
            if not eligible_rows:
                return
        if getattr(token_ids, "dtype", None) != jnp.dtype(jnp.int32):
            token_ids = jnp.asarray(token_ids, dtype=jnp.int32)
        full_batch_tokens = int(token_ids.shape[0]) == int(batch.tokens.shape[0])
        active_row_to_token_row = {row: index for index, row in enumerate(active_rows)}
        carry_by_seq_id: dict[int, DeviceTokenRef] = dict(
            getattr(self, "_device_token_carry_by_seq_id", {})
        )
        new_carry_by_seq_id: dict[int, DeviceTokenRef] = {}
        for row in eligible_rows:
            seq_id = int(batch.seq_ids_host[row])
            if seq_id < 0:
                continue
            token_row = row if full_batch_tokens else active_row_to_token_row[row]
            token_ref = DeviceTokenRef(tokens=token_ids, row=token_row)
            carry_by_seq_id[seq_id] = token_ref
            new_carry_by_seq_id[seq_id] = token_ref
        if not new_carry_by_seq_id:
            return
        if update_resident_tokens:
            self._record_resident_last_tokens(
                batch,
                token_ids,
                eligible_rows=eligible_rows,
                active_row_to_token_row=active_row_to_token_row,
                full_batch_tokens=full_batch_tokens,
            )
            if hasattr(self, "_resident_last_tokens_stale_seq_ids"):
                for seq_id in new_carry_by_seq_id:
                    self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        elif resident_tokens_already_current and hasattr(
            self,
            "_resident_last_tokens_stale_seq_ids",
        ):
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.discard(int(seq_id))
        elif hasattr(self, "_resident_last_tokens_stale_seq_ids"):
            for seq_id in new_carry_by_seq_id:
                self._resident_last_tokens_stale_seq_ids.add(int(seq_id))
        self._device_token_carry_seq_ids = (
            tuple(int(seq_id) for seq_id in batch.seq_ids_host)
            if full_batch_tokens
            else tuple(new_carry_by_seq_id)
        )
        self._device_token_carry_tokens = token_ids
        self._device_token_carry_by_seq_id = carry_by_seq_id
        use_seq_lens_carry = bool(
            getattr(
                self,
                "static_decode_seq_lens_carry",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "static_decode_seq_lens_carry",
                    "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
                ),
            )
        )
        if batch.is_prefill:
            self._device_seq_lens_carry_seq_ids = None
            self._device_seq_lens_carry = None
        elif use_seq_lens_carry:
            self._device_seq_lens_carry_seq_ids = tuple(int(seq_id) for seq_id in batch.seq_ids_host)
            if active_rows == list(range(int(batch.tokens.shape[0]))):
                self._device_seq_lens_carry = batch.seq_lens.astype(jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
            else:
                active_mask = jnp.zeros((int(batch.tokens.shape[0]),), dtype=bool)
                active_mask = active_mask.at[jnp.asarray(active_rows, dtype=jnp.int32)].set(True)
                self._device_seq_lens_carry = jnp.where(
                    active_mask,
                    batch.seq_lens.astype(jnp.int32) + jnp.asarray(1, dtype=jnp.int32),
                    batch.seq_lens.astype(jnp.int32),
                )
        else:
            self._device_seq_lens_carry_seq_ids = None
            self._device_seq_lens_carry = None

    def _materialize_static_decode_metadata_batch(
        self,
        batch: ScheduledBatch,
    ) -> ScheduledBatch:
        """Build a concrete decode batch from static/resident scheduler metadata.

        Static resident decode batches intentionally carry placeholder token
        metadata because the normal resident fast path gathers block tables,
        sequence lengths, positions, and last tokens inside the compiled
        boundary. MTP commit-select still consumes a conventional
        ``ScheduledBatch``. Until it has a resident-metadata verifier variant,
        give it concrete metadata so it verifies the same prefix as the
        non-speculative path.
        """
        if (
            batch.is_prefill
            or not bool(getattr(batch, "uses_static_decode_metadata", False))
            or batch.block_tables_host is None
            or batch.seq_lens_host is None
        ):
            return batch

        batch_size = int(batch.tokens.shape[0])
        query_width = int(batch.tokens.shape[1])
        seq_lens_host = tuple(int(x) for x in batch.seq_lens_host)
        block_tables_host = tuple(
            tuple(int(block) for block in row)
            for row in batch.block_tables_host
        )
        seq_ids_host = tuple(
            int(seq_id)
            for seq_id in (
                batch.seq_ids_host
                if batch.seq_ids_host is not None
                else tuple(int(x) for x in jax.device_get(batch.seq_ids).tolist())
            )
        )
        query_lens_host = tuple(
            int(query_len)
            for query_len in (
                batch.query_lens_host
                if batch.query_lens_host is not None
                else tuple(int(x) for x in jax.device_get(batch.query_lens).tolist())
            )
        )

        positions = np.zeros((batch_size, query_width), dtype=np.int32)
        for row in range(min(batch_size, len(seq_lens_host), len(query_lens_host))):
            if query_lens_host[row] > 0:
                positions[row, 0] = max(seq_lens_host[row] - 1, 0)

        return replace(
            batch,
            positions=jax.device_put(positions),
            seq_ids=jax.device_put(np.asarray(seq_ids_host, dtype=np.int32)),
            block_tables=jax.device_put(np.asarray(block_tables_host, dtype=np.int32)),
            seq_lens=jax.device_put(np.asarray(seq_lens_host, dtype=np.int32)),
            uses_static_decode_metadata=False,
        )

    def _record_mtp_output_token_carry(
        self,
        batch: ScheduledBatch,
        seqs: List[Sequence],
        outputs: dict[int, List[int] | int],
        *,
        update_resident_tokens: bool = True,
    ) -> None:
        """Record final emitted MTP tokens for the next resident decode step."""
        if (
            not outputs
            or not bool(
                getattr(
                    self,
                    "device_token_carry",
                    _config_or_env_flag(
                        getattr(self, "config", None),
                        "device_token_carry",
                        "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
                    ),
                )
            )
            or batch.seq_ids_host is None
        ):
            self._clear_device_token_carry()
            return

        active_rows: list[int] = []
        token_values: list[object] = []
        for row in sorted(outputs):
            row = int(row)
            if row < 0 or row >= len(seqs) or not seqs[row].ignore_eos:
                continue
            emitted = outputs[row]
            if isinstance(emitted, list):
                if not emitted:
                    continue
                emitted_count = len(emitted)
                token = emitted[-1]
            else:
                emitted_count = 1
                token = emitted
            if seqs[row].num_completion_tokens + emitted_count >= seqs[row].max_tokens:
                continue
            active_rows.append(row)
            token_values.append(token)
        if not active_rows:
            self._clear_device_token_carry()
            return

        if all(isinstance(token, DeviceTokenRef) for token in token_values):
            first_ref = token_values[0]
            assert isinstance(first_ref, DeviceTokenRef)
            first_tokens = first_ref.tokens
            if all(
                isinstance(token, DeviceTokenRef) and token.tokens is first_tokens
                for token in token_values
            ):
                if not update_resident_tokens:
                    carry_by_seq_id: dict[int, DeviceTokenRef] = dict(
                        getattr(self, "_device_token_carry_by_seq_id", {})
                    )
                    new_carry_by_seq_id: dict[int, DeviceTokenRef] = {}
                    for row, token in zip(active_rows, token_values):
                        assert isinstance(token, DeviceTokenRef)
                        seq_id = int(batch.seq_ids_host[row])
                        if seq_id < 0:
                            continue
                        carry_by_seq_id[seq_id] = token
                        new_carry_by_seq_id[seq_id] = token
                    if new_carry_by_seq_id:
                        self._device_token_carry_seq_ids = None
                        self._device_token_carry_tokens = None
                        self._device_token_carry_by_seq_id = carry_by_seq_id
                        self._device_seq_lens_carry_seq_ids = None
                        self._device_seq_lens_carry = None
                        return
                token_matrix = jnp.asarray(first_tokens, dtype=jnp.int32)
                batch_size = int(batch.tokens.shape[0])
                if token_matrix.ndim == 2 and int(token_matrix.shape[0]) == batch_size:
                    width = int(token_matrix.shape[1])
                    flat_rows = [int(token.row) for token in token_values]  # type: ignore[union-attr]
                    columns = [flat_row % width for flat_row in flat_rows]
                    source_rows = [flat_row // width for flat_row in flat_rows]
                    if (
                        columns
                        and len(set(columns)) == 1
                        and source_rows == active_rows
                    ):
                        token_vector = token_matrix[:, columns[0]]
                        self._record_device_token_carry(
                            batch,
                            token_vector,
                            active_rows=active_rows,
                            prefill_final_flags=[True for _ in range(len(seqs))],
                            seqs=seqs,
                            update_resident_tokens=update_resident_tokens,
                        )
                        return

        if len(active_rows) == 1 and int(batch.tokens.shape[0]) == 1:
            token = token_values[0]
            if isinstance(token, DeviceTokenRef):
                token_array = jnp.asarray(token.tokens, dtype=jnp.int32).reshape(-1)
                token_vector = token_array[int(token.row) : int(token.row) + 1].astype(jnp.int32)
            else:
                token_vector = jnp.asarray([int(token)], dtype=jnp.int32)
            self._record_device_token_carry(
                batch,
                token_vector,
                active_rows=active_rows,
                prefill_final_flags=[True for _ in range(len(seqs))],
                seqs=seqs,
                update_resident_tokens=update_resident_tokens,
            )
            return

        token_leaves = []
        for token in token_values:
            if isinstance(token, DeviceTokenRef):
                token_array = jnp.asarray(token.tokens, dtype=jnp.int32).reshape(-1)
                token_leaves.append(token_array[int(token.row)])
            else:
                token_leaves.append(jnp.asarray(int(token), dtype=jnp.int32))
        token_vector = jnp.stack(token_leaves).astype(jnp.int32)
        full_batch_tokens = jnp.zeros((int(batch.tokens.shape[0]),), dtype=jnp.int32)
        full_batch_tokens = full_batch_tokens.at[
            jnp.asarray(active_rows, dtype=jnp.int32)
        ].set(token_vector)
        self._record_device_token_carry(
            batch,
            full_batch_tokens,
            active_rows=active_rows,
            prefill_final_flags=[True for _ in range(len(seqs))],
            seqs=seqs,
            update_resident_tokens=update_resident_tokens,
        )

    def _device_token_carry_enabled(self) -> bool:
        return bool(
            getattr(
                self,
                "device_token_carry",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "device_token_carry",
                    "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
                ),
            )
        )

    @staticmethod
    def _materialize_device_token_outputs(
        outputs: dict[int, List[object] | object],
    ) -> dict[int, List[int] | int]:
        """Resolve deferred token refs for non-device-carry execution paths."""
        resolved_arrays: dict[int, np.ndarray] = {}

        def resolve_token(token: object) -> int:
            if isinstance(token, DeviceTokenRef):
                key = id(token.tokens)
                if key not in resolved_arrays:
                    resolved_arrays[key] = np.asarray(jax.device_get(token.tokens)).reshape(-1)
                return int(resolved_arrays[key][int(token.row)])
            if hasattr(token, "dtype") and hasattr(token, "shape"):
                return int(np.asarray(jax.device_get(token)).reshape(-1)[0])
            return int(token)  # type: ignore[arg-type]

        materialized: dict[int, List[int] | int] = {}
        for row, value in outputs.items():
            if isinstance(value, list):
                materialized[int(row)] = [resolve_token(token) for token in value]
            else:
                materialized[int(row)] = resolve_token(value)
        return materialized

    def _build_scheduled_batch(self, seqs: List[Sequence], is_prefill: bool) -> ScheduledBatch:
        query_tokens: List[List[int]] = []
        query_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        actual_max_blocks = max(1, max(len(seq.block_table) for seq in seqs))
        max_blocks = actual_max_blocks
        if self.max_blocks_per_seq is not None:
            if actual_max_blocks > self.max_blocks_per_seq:
                raise ValueError(
                    f"scheduled block table needs {actual_max_blocks} blocks but bucket has {self.max_blocks_per_seq}"
                )
            max_blocks = self.max_blocks_per_seq
        if not is_prefill:
            decode_block_table_buckets = tuple(getattr(self.config, "decode_block_table_buckets", ()) or ())
            if decode_block_table_buckets:
                max_blocks = self._select_bucket(actual_max_blocks, decode_block_table_buckets, "decode block table")
                if self.max_blocks_per_seq is not None and max_blocks > self.max_blocks_per_seq:
                    raise ValueError(
                        f"decode block table bucket {max_blocks} exceeds max_blocks_per_seq {self.max_blocks_per_seq}"
                    )
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

        seq_ids_host = tuple([seq.seq_id for seq in seqs] + [-1] * (batch_size_bucket - len(seqs)))
        query_lens_host = tuple(query_lens)
        return ScheduledBatch(
            tokens=jnp.array(padded_tokens, dtype=jnp.int32),
            positions=jnp.array(padded_positions, dtype=jnp.int32),
            seq_ids=jnp.array(seq_ids_host, dtype=jnp.int32),
            query_start_loc=jnp.array(query_start_loc, dtype=jnp.int32),
            is_prefill=is_prefill,
            num_prefill_tokens=sum(query_lens) if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else sum(query_lens),
            block_tables=jnp.array(block_tables, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=tuple(seq_lens),
        )

    @staticmethod
    def _select_bucket(size: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in sorted(buckets):
            if size <= bucket:
                return bucket
        raise ValueError(f"{name} size {size} exceeds configured buckets {buckets}")

    def _mtp_static_batch_size(self, size: int) -> int:
        """Return the reusable physical row count for MTP shapes.

        When ``mtp_max_active_rows`` is configured, the MTP serving path pads
        smaller decode verifier batches to that row bucket.  This keeps K=1/K>1
        verifier JIT keys stable as requests finish at different times.
        """

        size = int(size)
        if (
            not bool(getattr(self, "mtp1_enabled", False))
            or self.mtp_max_active_rows <= 0
            or size > self.mtp_max_active_rows
        ):
            return size
        target = int(self.mtp_max_active_rows)
        batch_buckets = tuple(getattr(self.config, "batch_size_buckets", ()) or ())
        if batch_buckets:
            return self._select_bucket(target, batch_buckets, "batch")
        return target

    def _zero_hybrid_slot(self, slot: int):
        self._zero_hybrid_slots([slot])

    def _zero_hybrid_slots(self, slots: List[int] | Tuple[int, ...]):
        slots = tuple(int(slot) for slot in slots if int(slot) >= 0)
        if not slots:
            return
        if not hasattr(self, "_zeroed_hybrid_slots"):
            self._zeroed_hybrid_slots = set()
        slots_to_zero = tuple(
            slot for slot in slots if slot not in self._zeroed_hybrid_slots
        )
        if not slots_to_zero:
            return
        conv_state = self._hybrid_state_table.conv_state
        recurrent_state = self._hybrid_state_table.recurrent_state
        if (
            conv_state is not None
            and recurrent_state is not None
            and len(slots_to_zero) == int(conv_state.shape[0])
            and slots_to_zero == tuple(range(int(conv_state.shape[0])))
        ):
            next_conv_state = jnp.zeros_like(conv_state)
            next_recurrent_state = jnp.zeros_like(recurrent_state)
        else:
            slot_ids = jnp.asarray(slots_to_zero, dtype=jnp.int32)
            next_conv_state = (
                conv_state.at[slot_ids].set(
                    jnp.zeros(
                        (len(slots_to_zero),) + conv_state.shape[1:],
                        dtype=conv_state.dtype,
                    )
                )
                if conv_state is not None
                else None
            )
            next_recurrent_state = (
                recurrent_state.at[slot_ids].set(
                    jnp.zeros(
                        (len(slots_to_zero),) + recurrent_state.shape[1:],
                        dtype=recurrent_state.dtype,
                    )
                )
                if recurrent_state is not None
                else None
            )
        self._hybrid_state_table = HybridLayerState(
            conv_state=next_conv_state,
            recurrent_state=next_recurrent_state,
        )
        self._zeroed_hybrid_slots.update(slots_to_zero)

    def _mark_hybrid_slots_written(self, slots: List[int] | Tuple[int, ...]):
        if not hasattr(self, "_zeroed_hybrid_slots"):
            self._zeroed_hybrid_slots = set()
        for slot in slots:
            if int(slot) >= 0:
                self._zeroed_hybrid_slots.discard(int(slot))

    def _assign_hybrid_slot(self, seq_id: int, preferred_slot: int | None = None) -> tuple[int, bool]:
        if seq_id < 0:
            return -1, False
        slot = self._hybrid_slots.get(seq_id)
        if slot is not None:
            return slot, False
        if not self._free_hybrid_slots:
            raise RuntimeError("No free hybrid-state slots; max_num_resident_seqs is exhausted")
        if (
            preferred_slot is not None
            and 0 <= preferred_slot < self._max_hybrid_slots
            and preferred_slot in self._free_hybrid_slots
        ):
            slot = preferred_slot
            self._free_hybrid_slots.remove(slot)
        elif seq_id < self._max_hybrid_slots and seq_id in self._free_hybrid_slots:
            slot = seq_id
            self._free_hybrid_slots.remove(slot)
        else:
            slot = self._free_hybrid_slots.pop()
        self._hybrid_slots[seq_id] = slot
        return slot, True

    def _ensure_hybrid_slot(self, seq_id: int, preferred_slot: int | None = None) -> int:
        slot, allocated = self._assign_hybrid_slot(seq_id, preferred_slot=preferred_slot)
        if allocated:
            self._zero_hybrid_slots([slot])
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

    def hybrid_state_for_sequence(self, seq_id: int) -> HybridLayerState | None:
        if seq_id < 0 or getattr(self, "_hybrid_state_table", None) is None:
            return None
        if self._hybrid_state_table.conv_state is None and self._hybrid_state_table.recurrent_state is None:
            return None
        if seq_id not in self._hybrid_slots:
            return None
        return self._get_hybrid_state(seq_id)

    def hybrid_states_for_sequences(self, seqs: List[Sequence]) -> dict[int, HybridLayerState]:
        states: dict[int, HybridLayerState] = {}
        for seq in seqs:
            state = self.hybrid_state_for_sequence(int(seq.seq_id))
            if state is not None:
                states[int(seq.seq_id)] = state
        return states

    def install_cached_prefix_hybrid_states(
        self,
        seqs: List[Sequence],
        prefix_states: dict[int, HybridLayerState] | None,
    ) -> None:
        if not prefix_states:
            return
        for seq in seqs:
            prefix_hash = getattr(seq, "cached_prefix_hash", None)
            if prefix_hash is None or int(getattr(seq, "num_cached_tokens", 0)) <= 0:
                continue
            if getattr(seq, "cached_prefix_hybrid_seeded", False):
                continue
            state = prefix_states.get(int(prefix_hash))
            if state is None:
                raise RuntimeError(
                    f"missing hybrid prefix state for cached prefix hash {int(prefix_hash)}"
                )
            self._set_hybrid_state(int(seq.seq_id), state)
            seq.cached_prefix_hybrid_seeded = True

    def _slice_batch(self, batch: ScheduledBatch, idx: int) -> ScheduledBatch:
        query_len = int(batch.query_lens[idx])
        block_tables_host = None
        if batch.block_tables_host is not None:
            block_tables_host = (tuple(batch.block_tables_host[idx]),)
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
            block_tables_host=block_tables_host,
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
        row_set = set(int(row) for row in rows)
        block_tables_host = None
        if batch.block_tables_host is not None:
            zero_row = tuple(0 for _ in batch.block_tables_host[0])
            block_tables_host = tuple(
                tuple(batch.block_tables_host[row]) if row in rows else zero_row
                for row in range(batch_size)
            )
        hybrid_slot_ids_host = None
        if batch.hybrid_slot_ids_host is not None:
            hybrid_slot_ids_host = tuple(
                int(batch.hybrid_slot_ids_host[row]) if row in row_set else -1
                for row in range(batch_size)
            )
        seq_ids_host = None
        if batch.seq_ids_host is not None:
            seq_ids_host = tuple(
                int(batch.seq_ids_host[row]) if row in row_set else -1
                for row in range(batch_size)
            )
        query_lens_host = tuple(1 if row in row_set else 0 for row in range(batch_size))
        seq_lens_host = None
        if seq_len_values is not None:
            row_to_seq_len = {
                int(row): int(value)
                for row, value in zip(rows, seq_len_values)
            }
            seq_lens_host = tuple(
                row_to_seq_len[row] if row in row_to_seq_len else 0
                for row in range(batch_size)
            )
        elif batch.seq_lens_host is not None:
            seq_lens_host = tuple(
                int(batch.seq_lens_host[row]) if row in row_set else 0
                for row in range(batch_size)
            )
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
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            hybrid_slot_ids_host=hybrid_slot_ids_host,
            uses_static_decode_metadata=False,
        )

    def _pad_decode_batch_to_rows(self, batch: ScheduledBatch, target_rows: int) -> ScheduledBatch:
        """Pad a decode batch with inactive rows to stabilize verifier shapes."""

        if batch.is_prefill:
            raise ValueError("decode batch padding requires a decode batch")
        target_rows = int(target_rows)
        current_rows = int(batch.tokens.shape[0])
        if target_rows <= current_rows:
            return batch
        query_width = int(batch.tokens.shape[1])
        block_width = int(batch.block_tables.shape[1])
        pad_rows = target_rows - current_rows

        query_lens = jnp.concatenate(
            [
                batch.query_lens.astype(jnp.int32),
                jnp.zeros((pad_rows,), dtype=jnp.int32),
            ]
        )
        zero_tokens = jnp.zeros((pad_rows, query_width), dtype=batch.tokens.dtype)
        zero_block_tables = jnp.zeros((pad_rows, block_width), dtype=batch.block_tables.dtype)

        seq_ids_host = None
        if batch.seq_ids_host is not None:
            seq_ids_host = tuple(int(x) for x in batch.seq_ids_host) + tuple(-1 for _ in range(pad_rows))
        query_lens_host = None
        if batch.query_lens_host is not None:
            query_lens_host = tuple(int(x) for x in batch.query_lens_host) + tuple(0 for _ in range(pad_rows))
        seq_lens_host = None
        if batch.seq_lens_host is not None:
            seq_lens_host = tuple(int(x) for x in batch.seq_lens_host) + tuple(0 for _ in range(pad_rows))
        block_tables_host = None
        if batch.block_tables_host is not None:
            zero_row = tuple(0 for _ in range(block_width))
            block_tables_host = tuple(tuple(int(block) for block in row) for row in batch.block_tables_host) + tuple(
                zero_row for _ in range(pad_rows)
            )
        hybrid_slot_ids_host = None
        if batch.hybrid_slot_ids_host is not None:
            hybrid_slot_ids_host = tuple(int(x) for x in batch.hybrid_slot_ids_host) + tuple(-1 for _ in range(pad_rows))

        return replace(
            batch,
            tokens=jnp.concatenate([batch.tokens, zero_tokens], axis=0),
            positions=jnp.concatenate([batch.positions, jnp.zeros_like(zero_tokens)], axis=0),
            seq_ids=jnp.concatenate(
                [
                    batch.seq_ids.astype(jnp.int32),
                    jnp.full((pad_rows,), -1, dtype=jnp.int32),
                ]
            ),
            query_start_loc=jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens),
                ]
            ),
            block_tables=jnp.concatenate([batch.block_tables, zero_block_tables], axis=0),
            seq_lens=jnp.concatenate(
                [
                    batch.seq_lens.astype(jnp.int32),
                    jnp.zeros((pad_rows,), dtype=jnp.int32),
                ]
            ),
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            hybrid_slot_ids_host=hybrid_slot_ids_host,
            uses_static_decode_metadata=False,
        )

    def _with_committed_seq_lens(
        self,
        batch: ScheduledBatch,
        committed_seq_lens: jnp.ndarray | None,
    ) -> ScheduledBatch:
        if committed_seq_lens is None:
            return batch
        active = (batch.seq_ids >= 0) & (batch.query_lens > 0)
        seq_lens = jnp.where(
            active,
            jnp.asarray(committed_seq_lens, dtype=jnp.int32),
            batch.seq_lens.astype(jnp.int32),
        )
        seq_lens_host = batch.seq_lens_host
        if seq_lens_host is not None:
            committed_host = np.asarray(jax.device_get(committed_seq_lens), dtype=np.int32).reshape(-1)
            seq_lens_values = [int(value) for value in seq_lens_host]
            seq_ids_host = batch.seq_ids_host
            query_lens_host = batch.query_lens_host
            for row in range(min(len(seq_lens_values), int(committed_host.shape[0]))):
                active_host = True
                if seq_ids_host is not None and row < len(seq_ids_host):
                    active_host = active_host and int(seq_ids_host[row]) >= 0
                if query_lens_host is not None and row < len(query_lens_host):
                    active_host = active_host and int(query_lens_host[row]) > 0
                if active_host:
                    seq_lens_values[row] = int(committed_host[row])
            seq_lens_host = tuple(seq_lens_values)
        return replace(batch, seq_lens=seq_lens, seq_lens_host=seq_lens_host)

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
        block_tables_host = None
        if batch.block_tables_host is not None:
            block_tables_host = tuple(tuple(batch.block_tables_host[row]) for row in rows)
        seq_ids_host = None
        if batch.seq_ids_host is not None:
            seq_ids_host = tuple(int(batch.seq_ids_host[row]) for row in rows)
        query_lens_host = None
        if batch.query_lens_host is not None:
            query_lens_host = tuple(int(batch.query_lens_host[row]) for row in rows)
        seq_lens_host = None
        if seq_len_values is not None:
            seq_lens_host = tuple(int(value) for value in seq_len_values)
        elif batch.seq_lens_host is not None:
            seq_lens_host = tuple(int(batch.seq_lens_host[row]) for row in rows)
        hybrid_slot_ids_host = None
        if batch.hybrid_slot_ids_host is not None:
            hybrid_slot_ids_host = tuple(int(batch.hybrid_slot_ids_host[row]) for row in rows)

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
            seq_ids_host=seq_ids_host,
            query_lens_host=query_lens_host,
            seq_lens_host=seq_lens_host,
            block_tables_host=block_tables_host,
            hybrid_slot_ids_host=hybrid_slot_ids_host,
        )

    def _batch_hybrid_state(self, batch: ScheduledBatch) -> HybridLayerState:
        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(seq_ids) == self._hybrid_state_table.conv_state.shape[0]
        ):
            direct_slots = True
            for row, seq_id in enumerate(seq_ids):
                if seq_id < 0 or self._hybrid_slots.get(int(seq_id)) != row:
                    direct_slots = False
                    break
            if direct_slots:
                batch.hybrid_slot_ids_host = tuple(range(len(seq_ids)))
                return self._hybrid_state_table
        if batch.hybrid_slot_ids_host is not None:
            slot_values = [int(slot) for slot in batch.hybrid_slot_ids_host]
            newly_allocated = [False for _ in slot_values]
        else:
            slot_allocations = [
                self._assign_hybrid_slot(int(seq_id), preferred_slot=row)
                for row, seq_id in enumerate(seq_ids)
            ]
            slot_values = [slot for slot, _ in slot_allocations]
            newly_allocated = [allocated for _, allocated in slot_allocations]
            self._zero_hybrid_slots(
                [slot for slot, allocated in slot_allocations if allocated]
            )
        batch.hybrid_slot_ids_host = tuple(slot_values)
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(slot_values) == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
            and all(newly_allocated)
        ):
            return self._hybrid_state_table
        if (
            self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and len(slot_values) == self._hybrid_state_table.conv_state.shape[0]
            and slot_values == list(range(len(slot_values)))
            and not any(newly_allocated)
        ):
            return self._hybrid_state_table
        slot_ids = jnp.array(slot_values, dtype=jnp.int32)
        safe_slot_ids = jnp.maximum(slot_ids, 0)
        valid_seq = jnp.asarray([seq_id >= 0 for seq_id in seq_ids], dtype=bool)
        valid = (
            (slot_ids >= 0)
            & valid_seq
            & jnp.logical_not(jnp.array(newly_allocated, dtype=bool))
        )
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
        query_lens = (
            list(batch.query_lens_host)
            if batch.query_lens_host is not None
            else [int(x) for x in batch.query_lens.tolist()]
        )
        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(x) for x in batch.seq_ids.tolist()]
        )
        slot_values_all = (
            list(batch.hybrid_slot_ids_host)
            if batch.hybrid_slot_ids_host is not None
            else [self._ensure_hybrid_slot(seq_id) for seq_id in seq_ids]
        )
        slot_values: List[int] = []
        for row, seq_id in enumerate(seq_ids):
            if seq_id < 0 or (not batch.is_prefill and query_lens[row] <= 0):
                continue
            valid_rows.append(row)
            slot_values.append(slot_values_all[row])
        if not valid_rows:
            return
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
            self._mark_hybrid_slots_written(slot_values)
            return
        row_ids = jnp.array(valid_rows, dtype=jnp.int32)
        slot_ids = jnp.array(slot_values, dtype=jnp.int32)
        self._hybrid_state_table = HybridLayerState(
            conv_state=self._hybrid_state_table.conv_state.at[slot_ids].set(state.conv_state[row_ids])
            if self._hybrid_state_table.conv_state is not None and state.conv_state is not None
            else self._hybrid_state_table.conv_state,
            recurrent_state=self._hybrid_state_table.recurrent_state.at[slot_ids].set(state.recurrent_state[row_ids])
            if self._hybrid_state_table.recurrent_state is not None and state.recurrent_state is not None
                else self._hybrid_state_table.recurrent_state,
        )
        self._mark_hybrid_slots_written(slot_values)

    def _batch_hybrid_slot_ids(self, batch: ScheduledBatch) -> jnp.ndarray:
        """Assign hybrid slots for a batch without gathering the state table."""

        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        slot_values: List[int] = []
        for row, seq_id in enumerate(seq_ids):
            slot, allocated = self._assign_hybrid_slot(int(seq_id), preferred_slot=row)
            if allocated:
                self._zero_hybrid_slots([slot])
            slot_values.append(slot)
        batch.hybrid_slot_ids_host = tuple(slot_values)
        slot_key = tuple(slot_values)
        cache = getattr(self, "_hybrid_slot_ids_device_cache", None)
        if cache is None:
            cache = {}
            self._hybrid_slot_ids_device_cache = cache
        cached = cache.get(slot_key)
        if cached is None:
            cached = jax.device_put(np.asarray(slot_key, dtype=np.int32))
            cache[slot_key] = cached
        return cached

    def _prefill_final_flags_device(self, batch: ScheduledBatch) -> jnp.ndarray:
        rows = max(0, int(batch.query_start_loc.shape[0]) - 1)
        flags = [bool(flag) for flag in list(batch.prefill_final_flags)[:rows]]
        if len(flags) < rows:
            flags.extend([False] * (rows - len(flags)))
        key = tuple(flags)
        cache = getattr(self, "_prefill_final_flags_device_cache", None)
        if cache is None:
            cache = {}
            self._prefill_final_flags_device_cache = cache
        cached = cache.get(key)
        if cached is None:
            cached = jax.device_put(np.asarray(key, dtype=bool))
            cache[key] = cached
        return cached

    def _resident_metadata_scatter_fn(self, kind: str, table_shape: tuple[int, ...], update_shape: tuple[int, ...]):
        cache = getattr(self, "_resident_metadata_scatter_cache", None)
        if cache is None:
            cache = {}
            self._resident_metadata_scatter_cache = cache
        key = (kind, tuple(int(x) for x in table_shape), tuple(int(x) for x in update_shape))
        fn = cache.get(key)
        if fn is None:

            def scatter_rows(table, slots, rows):
                return table.at[slots].set(rows)

            fn = jax.jit(scatter_rows, donate_argnums=(0,))
            cache[key] = fn
        return fn

    def _scatter_resident_block_table_rows(
        self,
        table: jnp.ndarray,
        slots: jnp.ndarray,
        rows: jnp.ndarray,
    ) -> jnp.ndarray:
        rows = jnp.asarray(rows, dtype=jnp.int32)
        slots = _int32_device_vector(slots)
        if rows.ndim != 2:
            raise ValueError("resident block-table row updates must be rank-2")
        if int(rows.shape[1]) != int(table.shape[1]):
            raise ValueError(
                "resident block-table update width must match the resident table width"
            )
        fn = self._resident_metadata_scatter_fn(
            "block_tables",
            tuple(int(x) for x in table.shape),
            tuple(int(x) for x in rows.shape),
        )
        return fn(table, slots, rows)

    def _scatter_resident_seq_lens(
        self,
        table: jnp.ndarray,
        slots: jnp.ndarray,
        seq_lens: jnp.ndarray,
    ) -> jnp.ndarray:
        seq_lens = _int32_device_vector(seq_lens)
        slots = _int32_device_vector(slots)
        fn = self._resident_metadata_scatter_fn(
            "seq_lens",
            tuple(int(x) for x in table.shape),
            tuple(int(x) for x in seq_lens.shape),
        )
        return fn(table, slots, seq_lens)

    def _resident_last_tokens_scatter_fn(
        self,
        table_shape: tuple[int, ...],
        slots_shape: tuple[int, ...],
        token_shape: tuple[int, ...],
        token_rows_shape: tuple[int, ...],
    ):
        cache = getattr(self, "_resident_metadata_scatter_cache", None)
        if cache is None:
            cache = {}
            self._resident_metadata_scatter_cache = cache
        key = (
            "last_tokens_from_rows",
            tuple(int(x) for x in table_shape),
            tuple(int(x) for x in slots_shape),
            tuple(int(x) for x in token_shape),
            tuple(int(x) for x in token_rows_shape),
        )
        fn = cache.get(key)
        if fn is None:

            def scatter_tokens(table, slots, token_ids, token_rows):
                token_vector = jnp.asarray(token_ids, dtype=jnp.int32).reshape(-1)
                token_rows = jnp.asarray(token_rows, dtype=jnp.int32).reshape(-1)
                slots = jnp.asarray(slots, dtype=jnp.int32).reshape(-1)
                return table.at[slots].set(token_vector[token_rows])

            fn = jax.jit(scatter_tokens, donate_argnums=(0,))
            cache[key] = fn
        return fn

    def _scatter_resident_last_tokens(
        self,
        table: jnp.ndarray,
        slots: jnp.ndarray,
        token_ids: jnp.ndarray,
        token_rows: jnp.ndarray,
    ) -> jnp.ndarray:
        slots = _int32_device_vector(slots)
        token_ids = jnp.asarray(token_ids, dtype=jnp.int32)
        token_rows = _int32_device_vector(token_rows)
        fn = self._resident_last_tokens_scatter_fn(
            tuple(int(x) for x in table.shape),
            tuple(int(x) for x in slots.shape),
            tuple(int(x) for x in token_ids.shape),
            tuple(int(x) for x in token_rows.shape),
        )
        return fn(table, slots, token_ids, token_rows)

    def _resident_update_slots_device(self, slots: List[int] | Tuple[int, ...]) -> jnp.ndarray:
        key = tuple(int(slot) for slot in slots)
        cache = getattr(self, "_resident_update_slots_device_cache", None)
        if cache is None:
            cache = {}
            self._resident_update_slots_device_cache = cache
        cached = cache.get(key)
        if cached is None:
            cached = jax.device_put(np.asarray(key, dtype=np.int32))
            cache[key] = cached
        return cached

    def _sync_resident_decode_metadata(
        self,
        batch: ScheduledBatch,
        slot_values: List[int] | Tuple[int, ...],
        *,
        sync_seq_lens: bool,
    ) -> None:
        """Refresh resident per-slot paging metadata from scheduler-owned rows.

        Block allocations are still owned by the Python scheduler/block manager.
        This method mirrors only the changed rows into device-resident tables so
        decode can gather compact metadata by slot id inside the JIT boundary.
        """

        if batch.block_tables_host is None:
            return
        seq_ids = (
            list(batch.seq_ids_host)
            if batch.seq_ids_host is not None
            else [int(seq_id) for seq_id in batch.seq_ids.tolist()]
        )
        query_lens = (
            list(batch.query_lens_host)
            if batch.query_lens_host is not None
            else [int(query_len) for query_len in batch.query_lens.tolist()]
        )
        seq_lens = (
            list(batch.seq_lens_host)
            if batch.seq_lens_host is not None
            else [int(seq_len) for seq_len in batch.seq_lens.tolist()]
        )
        if not hasattr(self, "_resident_block_counts_host"):
            self._resident_block_counts_host = [
                0 for _ in range(len(self._resident_block_tables_host))
            ]
        block_size = int(
            getattr(
                self,
                "block_size",
                getattr(getattr(self, "config", None), "block_size", 16),
            )
        )
        changed_block_slots: list[int] = []
        changed_block_rows: list[tuple[int, ...]] = []
        changed_seq_lens_slots: list[int] = []
        changed_seq_lens: list[int] = []
        for row, slot in enumerate(slot_values):
            slot = int(slot)
            if slot < 0 or row >= len(seq_ids) or int(seq_ids[row]) < 0:
                continue
            if (not batch.is_prefill) and row < len(query_lens) and int(query_lens[row]) <= 0:
                continue

            next_block_count = None
            if row < len(seq_lens):
                seq_len_for_blocks = max(0, int(seq_lens[row]))
                next_block_count = (seq_len_for_blocks + block_size - 1) // block_size
            skip_block_row_check = (
                not batch.is_prefill
                and next_block_count is not None
                and self._resident_block_counts_host[slot] == next_block_count
            )
            if not skip_block_row_check:
                source_row = tuple(int(block) for block in batch.block_tables_host[row])
                if len(source_row) < self.max_blocks_per_seq:
                    source_row = source_row + tuple(
                        0 for _ in range(self.max_blocks_per_seq - len(source_row))
                    )
                elif len(source_row) > self.max_blocks_per_seq:
                    source_row = source_row[: self.max_blocks_per_seq]
                if self._resident_block_tables_host[slot] != source_row:
                    self._resident_block_tables_host[slot] = source_row
                    changed_block_slots.append(slot)
                    changed_block_rows.append(source_row)
                if next_block_count is not None:
                    self._resident_block_counts_host[slot] = next_block_count

            if sync_seq_lens and row < len(seq_lens):
                seq_len = int(seq_lens[row])
                if self._resident_seq_lens_host[slot] != seq_len:
                    self._resident_seq_lens_host[slot] = seq_len
                    changed_seq_lens_slots.append(slot)
                    changed_seq_lens.append(seq_len)

        if changed_block_slots:
            self._resident_block_tables = self._scatter_resident_block_table_rows(
                self._resident_block_tables,
                self._resident_update_slots_device(changed_block_slots),
                jax.device_put(np.asarray(changed_block_rows, dtype=np.int32)),
            )
        if changed_seq_lens_slots:
            self._resident_seq_lens = self._scatter_resident_seq_lens(
                self._resident_seq_lens,
                self._resident_update_slots_device(changed_seq_lens_slots),
                jax.device_put(np.asarray(changed_seq_lens, dtype=np.int32)),
            )

    def _advance_resident_seq_lens_host(
        self,
        slot_values: List[int] | Tuple[int, ...],
        *,
        active_rows: List[int],
        steps: int,
    ) -> None:
        if steps <= 0:
            return
        active = set(int(row) for row in active_rows)
        for row, slot in enumerate(slot_values):
            slot = int(slot)
            if slot >= 0 and row in active:
                self._resident_seq_lens_host[slot] += int(steps)

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
            token_row_ids=batch.token_row_ids if batch.packed_prefill else None,
            max_query_len=(
                max(tuple(getattr(self.config, "prefill_buckets", ()) or ()))
                if batch.packed_prefill
                and tuple(getattr(self.config, "prefill_buckets", ()) or ())
                else None
            ),
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
        is avoidable hot-path work for generation.
        """
        if hybrid_state is None:
            hybrid_state = self._batch_hybrid_state(batch)
        kv_state = getattr(self, "kv_state", None)
        if kv_state is None:
            return
        slot_mapping = getattr(kv_state, "slot_mapping", None)
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

    def _record_resident_committed_seq_lens(self, batch: ScheduledBatch) -> None:
        """Mirror committed per-row decode lengths into resident metadata."""
        if (
            not bool(getattr(self, "resident_decode_metadata", False))
            or not hasattr(self, "_resident_seq_lens")
            or batch.hybrid_slot_ids_host is None
            or batch.seq_ids_host is None
            or batch.query_lens_host is None
        ):
            return
        slots: list[int] = []
        rows: list[int] = []
        for row, (slot, seq_id, query_len) in enumerate(
            zip(batch.hybrid_slot_ids_host, batch.seq_ids_host, batch.query_lens_host)
        ):
            if int(slot) < 0 or int(seq_id) < 0 or int(query_len) <= 0:
                continue
            slots.append(int(slot))
            rows.append(row)
        if not slots:
            return
        row_ids = jnp.asarray(rows, dtype=jnp.int32)
        committed_lens = batch.seq_lens.astype(jnp.int32)[row_ids]
        self._resident_seq_lens = self._scatter_resident_seq_lens(
            self._resident_seq_lens,
            self._resident_update_slots_device(slots),
            committed_lens,
        )
        if hasattr(self, "_resident_seq_lens_host") and batch.seq_lens_host is not None:
            for slot, row in zip(slots, rows):
                if row < len(batch.seq_lens_host):
                    self._resident_seq_lens_host[int(slot)] = int(batch.seq_lens_host[row])

    def _record_resident_committed_seq_lens_host(
        self,
        batch: ScheduledBatch,
        row_to_committed_len: dict[int, int],
    ) -> None:
        """Mirror committed per-row decode lengths into the resident host cache."""
        if (
            not row_to_committed_len
            or not bool(getattr(self, "resident_decode_metadata", False))
            or not hasattr(self, "_resident_seq_lens_host")
            or batch.hybrid_slot_ids_host is None
            or batch.seq_ids_host is None
            or batch.query_lens_host is None
        ):
            return
        for row, committed_len in row_to_committed_len.items():
            row = int(row)
            if (
                row < 0
                or row >= len(batch.hybrid_slot_ids_host)
                or row >= len(batch.seq_ids_host)
                or row >= len(batch.query_lens_host)
            ):
                continue
            slot = int(batch.hybrid_slot_ids_host[row])
            if (
                slot < 0
                or int(batch.seq_ids_host[row]) < 0
                or int(batch.query_lens_host[row]) <= 0
            ):
                continue
            self._resident_seq_lens_host[slot] = int(committed_len)

    def _step_fn(self, batch: ScheduledBatch):
        execution = getattr(self, "execution", "eager")
        if execution == "jit" or (execution == "decode-jit" and not batch.is_prefill):
            return self.executor.forward_step_jit
        return self.executor.forward_step

    def _can_use_greedy_token_fastpath(self, seqs: List[Sequence], batch: ScheduledBatch, *, seed_mtp1: bool) -> bool:
        if seed_mtp1 and not batch.is_prefill:
            return False
        if not bool(
            getattr(
                self,
                "greedy_token_fastpath",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "greedy_token_fastpath",
                    "NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH",
                    default=True,
                ),
            )
        ):
            return False
        execution = getattr(self, "execution", "eager")
        if execution != "jit" and not (execution == "decode-jit" and not batch.is_prefill):
            return False
        if batch.is_prefill and bool(getattr(self, "_capture_prefill_logits", False)):
            return False
        for seq in seqs:
            if float(getattr(seq, "temperature", 0.0)) != 0.0:
                return False
        return hasattr(self.executor, "forward_step_token_ids_jit")

    def _can_use_sampled_token_fastpath(self, seqs: List[Sequence], batch: ScheduledBatch, *, seed_mtp1: bool) -> bool:
        if seed_mtp1:
            return False
        if not bool(
            getattr(
                self,
                "sampled_token_fastpath",
                _config_or_env_flag(
                    getattr(self, "config", None),
                    "sampled_token_fastpath",
                    "NANO_VLLM_JAX_SAMPLED_TOKEN_FASTPATH",
                    default=True,
                ),
            )
        ):
            return False
        execution = getattr(self, "execution", "eager")
        if execution != "jit" and not (execution == "decode-jit" and not batch.is_prefill):
            return False
        if batch.is_prefill and bool(getattr(self, "_capture_prefill_logits", False)):
            return False
        if not hasattr(self.executor, "forward_step_sampled_token_ids_jit"):
            return False
        has_sampling = False
        for seq in seqs:
            temperature = float(getattr(seq, "temperature", 0.0))
            if temperature < 0.0:
                return False
            if temperature > 0.0:
                has_sampling = True
            if float(getattr(seq, "top_p", 1.0)) < 1.0:
                return False
            if int(getattr(seq, "top_k", -1)) > 0:
                return False
        return has_sampling

    def _sample_temperatures_device(self, seqs: List[Sequence], batch: ScheduledBatch) -> jnp.ndarray:
        row_count = len(seqs) if batch.is_prefill and batch.packed_prefill else int(batch.tokens.shape[0])
        values = [0.0 for _ in range(row_count)]
        active_limit = min(len(seqs), row_count)
        for row in range(active_limit):
            if batch.query_lens_host is not None and int(batch.query_lens_host[row]) <= 0:
                continue
            values[row] = float(getattr(seqs[row], "temperature", 0.0))
        return jnp.asarray(values, dtype=jnp.float32)

    def _sample_rng_slots_and_counters_device(
        self,
        batch: ScheduledBatch,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self._flush_resident_rng_counter_resets()
        row_count = (
            len(batch.query_lens_host)
            if batch.is_prefill and batch.packed_prefill and batch.query_lens_host is not None
            else int(batch.tokens.shape[0])
        )
        slot_values = list(batch.hybrid_slot_ids_host or ())
        if len(slot_values) < row_count:
            slot_values.extend([-1] * (row_count - len(slot_values)))
        safe_slots = [max(0, int(slot)) for slot in slot_values[:row_count]]
        slot_ids = jnp.asarray(safe_slots, dtype=jnp.int32)
        if hasattr(self, "_resident_rng_counters"):
            counters = self._resident_rng_counters[slot_ids]
        else:
            counters = jnp.zeros((row_count,), dtype=jnp.int32)
        return slot_ids, counters.astype(jnp.int32)

    def _flush_resident_rng_counter_resets(self) -> None:
        """Apply deferred sampled-RNG counter resets before sampled paths read them."""

        reset_slots = getattr(self, "_resident_rng_counter_reset_slots", set())
        if not reset_slots or not hasattr(self, "_resident_rng_counters"):
            return
        slots = tuple(sorted(int(slot) for slot in reset_slots if int(slot) >= 0))
        reset_slots.clear()
        if not slots:
            return
        slot_ids = jnp.asarray(slots, dtype=jnp.int32)
        self._resident_rng_counters = self._resident_rng_counters.at[slot_ids].set(
            jnp.zeros((len(slots),), dtype=jnp.int32)
        )

    def _record_resident_rng_counters(
        self,
        batch: ScheduledBatch,
        updated_counters: jnp.ndarray | None,
        *,
        active_rows: list[int],
        prefill_final_flags: list[bool],
    ) -> None:
        if updated_counters is None or not hasattr(self, "_resident_rng_counters"):
            return
        self._flush_resident_rng_counter_resets()
        slot_values = list(batch.hybrid_slot_ids_host or ())
        if not slot_values:
            return
        slots: list[int] = []
        rows: list[int] = []
        for row in active_rows:
            if row >= len(slot_values):
                continue
            if batch.is_prefill and (row >= len(prefill_final_flags) or not prefill_final_flags[row]):
                continue
            slot = int(slot_values[row])
            if slot < 0:
                continue
            slots.append(slot)
            rows.append(row)
        if not slots:
            return
        self._resident_rng_counters = self._resident_rng_counters.at[
            jnp.asarray(slots, dtype=jnp.int32)
        ].set(updated_counters[jnp.asarray(rows, dtype=jnp.int32)].astype(jnp.int32))

    def _greedy_decode_burst_steps(self, seqs: List[Sequence], batch: ScheduledBatch) -> int:
        if batch.is_prefill:
            return 1
        configured_steps = max(
            1,
            _config_or_env_int(
                getattr(self, "config", None),
                "greedy_decode_burst_steps",
                "NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS",
                default=1,
            ),
        )
        if configured_steps <= 1:
            return 1
        if not hasattr(self.executor, "forward_greedy_decode_burst_jit"):
            return 1
        if getattr(batch, "decode_step_count_host", 1) <= 1:
            return 1
        if batch.query_lens_host is not None:
            active_query_lens = list(batch.query_lens_host[: len(seqs)])
            if any(int(length) != 1 for length in active_query_lens):
                return 1
        for seq in seqs:
            if seq.temperature != 0 or not seq.ignore_eos:
                return 1
        remaining = [seq.max_tokens - seq.num_completion_tokens for seq in seqs]
        if not remaining or min(remaining) <= 1:
            return 1
        return max(
            1,
            min(
                configured_steps,
                int(getattr(batch, "decode_step_count_host", 1)),
                min(remaining),
            ),
        )

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
            mtp_token_ids, _ = mtp_forward_token_ids(
                hidden_state=hidden_arg,
                next_token_ids=token_arg,
                embed_tokens=embed_tokens_arg,
                params=self._mtp1_params_from_tree(mtp_params_tree),
                config=self.config,
                positions=position_arg,
            )
            return mtp_token_ids[:, 0].astype(jnp.int32)

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
            return_normed_hidden = (
                getattr(self.config, "mtp_chain_hidden_source", "raw") == "final_normed"
            )
            chain_mode = getattr(self.config, "mtp_chain_mode", "recursive")
            hidden_seq = current_hidden
            token_seq = current_token
            position_seq = current_position
            drafts = []
            for _ in range(draft_len):
                if chain_mode == "sequence":
                    current_token, current_hidden = mtp_forward_last_token_ids(
                        hidden_state=hidden_seq,
                        next_token_ids=token_seq,
                        embed_tokens=embed_tokens_arg,
                        params=mtp_params,
                        config=self.config,
                        positions=position_seq,
                        return_normed_hidden=return_normed_hidden,
                    )
                else:
                    current_token, current_hidden = mtp_forward_token_ids(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=embed_tokens_arg,
                        params=mtp_params,
                        config=self.config,
                        positions=current_position,
                        return_normed_hidden=return_normed_hidden,
                    )
                drafts.append(current_token[:, 0])
                current_position = current_position + 1
                if chain_mode == "sequence":
                    hidden_seq = jnp.concatenate([hidden_seq, current_hidden], axis=1)
                    token_seq = jnp.concatenate([token_seq, current_token], axis=1)
                    position_seq = jnp.concatenate([position_seq, current_position], axis=1)
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
            return_normed_hidden = (
                getattr(self.config, "mtp_chain_hidden_source", "raw") == "final_normed"
            )
            chain_mode = getattr(self.config, "mtp_chain_mode", "recursive")
            hidden_seq = current_hidden
            token_seq = current_token
            position_seq = current_position
            drafts = []
            first_margin = None
            for idx in range(draft_len):
                if chain_mode == "sequence":
                    logits, current_hidden = mtp_forward_last(
                        hidden_state=hidden_seq,
                        next_token_ids=token_seq,
                        embed_tokens=embed_tokens_arg,
                        params=mtp_params,
                        config=self.config,
                        positions=position_seq,
                        return_normed_hidden=return_normed_hidden,
                    )
                    step_logits = logits[:, 0, :]
                else:
                    logits, current_hidden = mtp_forward(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=embed_tokens_arg,
                        params=mtp_params,
                        config=self.config,
                        positions=current_position,
                        return_normed_hidden=return_normed_hidden,
                    )
                    step_logits = logits[:, 0, :]
                if idx == 0:
                    top2, _ = jax.lax.top_k(step_logits.astype(jnp.float32), 2)
                    first_margin = top2[:, 0] - top2[:, 1]
                current_token = jnp.argmax(step_logits, axis=-1).astype(jnp.int32)[:, None]
                drafts.append(current_token[:, 0])
                current_position = current_position + 1
                if chain_mode == "sequence":
                    hidden_seq = jnp.concatenate([hidden_seq, current_hidden], axis=1)
                    token_seq = jnp.concatenate([token_seq, current_token], axis=1)
                    position_seq = jnp.concatenate([position_seq, current_position], axis=1)
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

    def _mtp1_draft_chain_with_topk(
        self,
        hidden_state: jnp.ndarray,
        token_ids: jnp.ndarray,
        positions: jnp.ndarray,
        draft_len: int,
        *,
        top_k: int = 5,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def forward(hidden_arg, token_arg, position_arg, embed_tokens_arg, mtp_params_tree):
            mtp_params = self._mtp1_params_from_tree(mtp_params_tree)
            current_hidden = hidden_arg
            current_token = token_arg
            current_position = position_arg
            return_normed_hidden = (
                getattr(self.config, "mtp_chain_hidden_source", "raw") == "final_normed"
            )
            chain_mode = getattr(self.config, "mtp_chain_mode", "recursive")
            hidden_seq = current_hidden
            token_seq = current_token
            position_seq = current_position
            drafts = []
            top_ids = []
            top_values = []
            for _idx in range(draft_len):
                if chain_mode == "sequence":
                    logits, current_hidden = mtp_forward_last(
                        hidden_state=hidden_seq,
                        next_token_ids=token_seq,
                        embed_tokens=embed_tokens_arg,
                        params=mtp_params,
                        config=self.config,
                        positions=position_seq,
                        return_normed_hidden=return_normed_hidden,
                    )
                    step_logits = logits[:, 0, :]
                else:
                    logits, current_hidden = mtp_forward(
                        hidden_state=current_hidden,
                        next_token_ids=current_token,
                        embed_tokens=embed_tokens_arg,
                        params=mtp_params,
                        config=self.config,
                        positions=current_position,
                        return_normed_hidden=return_normed_hidden,
                    )
                    step_logits = logits[:, 0, :]
                values, ids = jax.lax.top_k(step_logits.astype(jnp.float32), top_k)
                current_token = jnp.argmax(step_logits, axis=-1).astype(jnp.int32)[:, None]
                drafts.append(current_token[:, 0])
                top_ids.append(ids.astype(jnp.int32))
                top_values.append(values.astype(jnp.float32))
                current_position = current_position + 1
                if chain_mode == "sequence":
                    hidden_seq = jnp.concatenate([hidden_seq, current_hidden], axis=1)
                    token_seq = jnp.concatenate([token_seq, current_token], axis=1)
                    position_seq = jnp.concatenate([position_seq, current_position], axis=1)
            return (
                jnp.stack(drafts, axis=1),
                jnp.stack(top_ids, axis=1),
                jnp.stack(top_values, axis=1),
            )

        mtp_params_tree = self._mtp1_params_tree()
        if getattr(self, "execution", "eager") in {"decode-jit", "jit"}:
            if not hasattr(self, "_mtp1_chain_topk_jit"):
                self._mtp1_chain_topk_jit = {}
            key = (int(draft_len), int(top_k))
            if key not in self._mtp1_chain_topk_jit:
                self._mtp1_chain_topk_jit[key] = jax.jit(forward)
            return self._mtp1_chain_topk_jit[key](
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
        profile_prefill_seed = (
            batch.is_prefill
            and seed_mtp1
            and os.environ.get("NANO_VLLM_JAX_PROFILE_PREFILL_SEED", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        t_prefill_seed = time.perf_counter()

        def _prefill_seed_mark(label: str) -> None:
            nonlocal t_prefill_seed
            if profile_prefill_seed:
                now = time.perf_counter()
                print(f"[PREFILL_SEED] {label}={(now - t_prefill_seed) * 1000:.3f}ms", flush=True)
                t_prefill_seed = now

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

        if batch.is_prefill:
            prefill_final_flags = list(batch.prefill_final_flags)[: len(seqs)]
            if len(prefill_final_flags) < len(seqs):
                prefill_final_flags.extend([True] * (len(seqs) - len(prefill_final_flags)))
        else:
            prefill_final_flags = [True] * len(seqs)
        return_hidden_for_seed = bool(seed_mtp1)
        use_greedy_token_fastpath = self._can_use_greedy_token_fastpath(
            seqs,
            batch,
            seed_mtp1=return_hidden_for_seed,
        )
        use_sampled_token_fastpath = (
            not use_greedy_token_fastpath
            and self._can_use_sampled_token_fastpath(
                seqs,
                batch,
                seed_mtp1=return_hidden_for_seed,
            )
        )
        decode_burst_steps = (
            self._greedy_decode_burst_steps(seqs, batch)
            if use_greedy_token_fastpath
            else 1
        )
        use_hybrid_table_decode = (
            use_greedy_token_fastpath
            and not batch.is_prefill
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and (
                (decode_burst_steps <= 1 and hasattr(self.executor, "forward_step_token_ids_table_jit"))
                or (
                    decode_burst_steps > 1
                    and hasattr(self.executor, "forward_greedy_decode_burst_table_jit")
                )
            )
        )
        use_hybrid_table_prefill = (
            use_greedy_token_fastpath
            and batch.is_prefill
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and hasattr(self.executor, "forward_prefill_token_ids_table_jit")
        )
        use_sampled_hybrid_table_decode = (
            use_sampled_token_fastpath
            and not batch.is_prefill
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and bool(getattr(self, "resident_decode_metadata", False))
            and hasattr(self.executor, "forward_step_sampled_token_ids_resident_dense_slot_carry_jit")
        )
        use_prefill_slot_carry_table = (
            use_hybrid_table_prefill
            and not return_hidden_for_seed
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
            and hasattr(self.executor, "forward_prefill_token_ids_slot_carry_table_jit")
        )
        resident_slot_token_decode = (
            use_hybrid_table_decode
            and decode_burst_steps <= 1
            and not bool(getattr(self, "resident_decode_metadata", False))
            and self._resident_slot_token_decode_ready(
                batch,
                active_rows=self._active_decode_rows_host(batch),
            )
        )
        resident_slot_token_metadata_decode = (
            use_hybrid_table_decode
            and decode_burst_steps <= 1
            and bool(getattr(self, "resident_decode_metadata", False))
            and hasattr(self.executor, "forward_step_token_ids_resident_slot_carry_jit")
            and self._resident_slot_token_decode_ready(
                batch,
                active_rows=self._active_decode_rows_host(batch),
            )
        )
        resident_dense_slot_token_metadata_decode = (
            resident_slot_token_metadata_decode
            and hasattr(self.executor, "forward_step_token_ids_resident_dense_slot_carry_jit")
            and self._resident_slot_token_dense_decode_ready(
                batch,
                active_rows=self._active_decode_rows_host(batch),
            )
        )
        sampled_resident_dense_slot_token_metadata_decode = (
            use_sampled_hybrid_table_decode
            and bool(getattr(self, "device_token_carry", False))
            and hasattr(self, "_resident_last_tokens")
            and hasattr(self, "_resident_rng_counters")
            and self._resident_slot_token_dense_decode_ready(
                batch,
                active_rows=self._active_decode_rows_host(batch),
            )
        )
        if not (
            resident_slot_token_decode
            or resident_slot_token_metadata_decode
            or sampled_resident_dense_slot_token_metadata_decode
        ):
            batch = self._maybe_apply_device_token_carry(batch)
        if batch.query_lens_host is not None:
            query_lens = [int(x) for x in batch.query_lens_host[: len(seqs)]]
        else:
            query_lens = [int(x) for x in batch.query_lens[: len(seqs)].tolist()]
        if batch.seq_ids_host is not None:
            seq_ids_host = [int(x) for x in batch.seq_ids_host[: len(seqs)]]
        else:
            seq_ids_host = [int(batch.seq_ids[row]) for row in range(len(seqs))]
        active_rows = [
            row
            for row, query_len in enumerate(query_lens)
            if query_len > 0 and seq_ids_host[row] >= 0
        ]
        hidden_seed_without_logits = bool(
            return_hidden_for_seed
            and active_rows
            and all(float(getattr(seqs[row], "temperature", 0.0) or 0.0) == 0.0 for row in active_rows)
        )
        if use_hybrid_table_decode or use_hybrid_table_prefill or sampled_resident_dense_slot_token_metadata_decode:
            hybrid_slot_ids = self._batch_hybrid_slot_ids(batch)
            hybrid_slot_values = list(batch.hybrid_slot_ids_host or ())
            hybrid_state = self._hybrid_state_table
        else:
            hybrid_slot_ids = None
            hybrid_slot_values = list(batch.hybrid_slot_ids_host or ())
            hybrid_state = self._batch_hybrid_state(batch)
            hybrid_slot_values = list(batch.hybrid_slot_ids_host or hybrid_slot_values)
        use_resident_slot_decode = (
            use_hybrid_table_decode
            and decode_burst_steps <= 1
            and bool(getattr(self, "resident_decode_metadata", False))
            and not resident_slot_token_metadata_decode
            and hasattr(self.executor, "forward_step_token_ids_resident_jit")
        )
        if (
            use_resident_slot_decode
            or resident_slot_token_metadata_decode
            or sampled_resident_dense_slot_token_metadata_decode
        ):
            self._sync_resident_decode_metadata(
                batch,
                hybrid_slot_values,
                sync_seq_lens=True,
        )
        prefill_resident_tokens_seeded = False
        if decode_burst_steps > 1:
            if use_hybrid_table_decode:
                output = self.executor.forward_greedy_decode_burst_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    decode_steps=decode_burst_steps,
                )
            else:
                output = self.executor.forward_greedy_decode_burst_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    decode_steps=decode_burst_steps,
                )
        elif use_greedy_token_fastpath:
            if use_prefill_slot_carry_table:
                output = self.executor.forward_prefill_token_ids_slot_carry_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    prefill_final_flags=self._prefill_final_flags_device(batch),
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
                    prefill_resident_tokens_seeded = True
            elif use_hybrid_table_prefill:
                output = self.executor.forward_prefill_token_ids_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    mtp_cache_storage=self.mtp_cache_storage if return_hidden_for_seed else None,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    return_mtp_draft=return_hidden_for_seed,
                )
            elif resident_dense_slot_token_metadata_decode:
                output = self.executor.forward_step_token_ids_resident_dense_slot_carry_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
            elif resident_slot_token_metadata_decode:
                output = self.executor.forward_step_token_ids_resident_slot_carry_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
            elif use_resident_slot_decode:
                output = self.executor.forward_step_token_ids_resident_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                )
            elif resident_slot_token_decode:
                output = self.executor.forward_step_token_ids_slot_carry_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_last_tokens=self._resident_last_tokens,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
            elif use_hybrid_table_decode:
                output = self.executor.forward_step_token_ids_table_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                )
            else:
                output = self.executor.forward_step_token_ids_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                )
        elif use_sampled_token_fastpath:
            temperatures = self._sample_temperatures_device(seqs, batch)
            if sampled_resident_dense_slot_token_metadata_decode:
                self._flush_resident_rng_counter_resets()
                output = self.executor.forward_step_sampled_token_ids_resident_dense_slot_carry_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=hybrid_state,
                    hybrid_slot_ids=hybrid_slot_ids,
                    resident_block_tables=self._resident_block_tables,
                    resident_seq_lens=self._resident_seq_lens,
                    resident_last_tokens=self._resident_last_tokens,
                    resident_rng_counters=self._resident_rng_counters,
                    temperatures=temperatures,
                )
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
                if output.resident_rng_counters is not None:
                    self._resident_rng_counters = output.resident_rng_counters
            else:
                rng_slots, rng_counters = self._sample_rng_slots_and_counters_device(batch)
                output = self.executor.forward_step_sampled_token_ids_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    temperatures=temperatures,
                    rng_counters=rng_counters,
                    rng_slots=rng_slots,
                )
        else:
            output = self._step_fn(batch)(
                batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                return_hidden=return_hidden_for_seed,
                return_hidden_with_logits=return_hidden_for_seed and not hidden_seed_without_logits,
                last_logits_only=True,
            )
        _prefill_seed_mark("executor_dispatch")
        self.cache_storage = output.cache_storage
        if output.mtp_cache_storage is not None:
            self.mtp_cache_storage = output.mtp_cache_storage
        if use_hybrid_table_decode or use_hybrid_table_prefill or sampled_resident_dense_slot_token_metadata_decode:
            self._hybrid_state_table = output.hybrid_state
            self._mark_hybrid_slots_written(list(batch.hybrid_slot_ids_host or ()))
            if (
                (
                    use_resident_slot_decode
                    or resident_slot_token_metadata_decode
                    or sampled_resident_dense_slot_token_metadata_decode
                )
                and output.resident_seq_lens is not None
            ):
                self._resident_seq_lens = output.resident_seq_lens
                self._advance_resident_seq_lens_host(
                    hybrid_slot_values,
                    active_rows=active_rows,
                    steps=1,
                )
        else:
            self._store_batch_hybrid_state(batch, output.hybrid_state)
            if use_sampled_token_fastpath:
                self._record_resident_rng_counters(
                    batch,
                    output.resident_rng_counters,
                    active_rows=active_rows,
                    prefill_final_flags=prefill_final_flags,
                )
        if batch.is_prefill and bool(getattr(self, "resident_decode_metadata", False)):
            self._sync_resident_decode_metadata(
                batch,
                list(batch.hybrid_slot_ids_host or ()),
                sync_seq_lens=True,
            )
        snapshot_batch = batch
        if decode_burst_steps > 1:
            active = batch.active_decode_rows
            processed_seq_lens = jnp.where(
                active,
                batch.seq_lens + jnp.asarray(decode_burst_steps - 1, dtype=batch.seq_lens.dtype),
                batch.seq_lens,
            )
            seq_lens_host = None
            if batch.seq_lens_host is not None and batch.query_lens_host is not None:
                seq_lens_host = tuple(
                    int(length) + (decode_burst_steps - 1)
                    if idx < len(seqs) and int(batch.query_lens_host[idx]) > 0
                    else int(length)
                    for idx, length in enumerate(batch.seq_lens_host)
                )
            snapshot_batch = replace(
                batch,
                seq_lens=processed_seq_lens,
                seq_lens_host=seq_lens_host,
            )
        if os.environ.get("NANO_VLLM_JAX_REFRESH_KV_SNAPSHOT", "0") in {"1", "true", "yes", "on", "True"}:
            self._refresh_kv_snapshot(snapshot_batch, output.hybrid_state)
        else:
            self._record_kv_snapshot(snapshot_batch, output.hybrid_state)

        last_hidden = None
        seed_hidden = None
        token_ids_all = None
        prefill_mtp_draft_tokens = None
        if decode_burst_steps > 1:
            token_ids_all = output.activations[: len(seqs), :decode_burst_steps]
            last_logits = None
        elif use_greedy_token_fastpath or use_sampled_token_fastpath:
            if isinstance(output.activations, tuple):
                token_ids_all, aux_tokens_or_hidden = output.activations
                if int(token_ids_all.shape[0]) != len(seqs):
                    token_ids_all = token_ids_all[: len(seqs)]
                    aux_tokens_or_hidden = aux_tokens_or_hidden[: len(seqs)]
                if (
                    getattr(aux_tokens_or_hidden, "dtype", None) is not None
                    and jnp.issubdtype(aux_tokens_or_hidden.dtype, jnp.integer)
                    and getattr(aux_tokens_or_hidden, "ndim", 0) in {1, 2}
                ):
                    prefill_mtp_draft_tokens = aux_tokens_or_hidden
                else:
                    last_hidden = aux_tokens_or_hidden
                    seed_hidden = self._hidden_for_mtp(last_hidden[:, None, :])[:, 0]
            else:
                token_ids_all = output.activations
            if int(token_ids_all.shape[0]) != len(seqs):
                token_ids_all = token_ids_all[: len(seqs)]
            last_logits = None
        elif return_hidden_for_seed:
            if isinstance(output.activations, tuple):
                hidden_activations, logits = output.activations
                last_logits = logits[: len(seqs), 0]
            else:
                hidden_activations = output.activations
                last_logits = None
            last_hidden = self._last_query_activations(hidden_activations, batch, len(seqs))
            seed_hidden = self._hidden_for_mtp(last_hidden[:, None, :])[:, 0]
            if last_logits is None:
                token_ids_all = self._greedy_tokens_from_hidden(last_hidden[:, None, :])
        else:
            last_logits = output.activations[: len(seqs), 0]
        carry_device_tokens = (
            (use_greedy_token_fastpath or use_sampled_token_fastpath)
            and bool(
                getattr(
                    self,
                    "device_token_carry",
                    _config_or_env_flag(
                        getattr(self, "config", None),
                        "device_token_carry",
                        "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
                    ),
                )
            )
            and all(seqs[row].ignore_eos for row in active_rows)
        )
        if (use_greedy_token_fastpath or use_sampled_token_fastpath) and decode_burst_steps <= 1:
            carry_tokens = token_ids_all if token_ids_all is not None else output.activations
            resident_tokens_already_current = (
                resident_slot_token_decode
                or resident_slot_token_metadata_decode
                or sampled_resident_dense_slot_token_metadata_decode
                or prefill_resident_tokens_seeded
            )
            self._record_device_token_carry(
                batch,
                carry_tokens,
                active_rows=active_rows,
                prefill_final_flags=prefill_final_flags,
                seqs=seqs,
                update_resident_tokens=not resident_tokens_already_current,
                resident_tokens_already_current=resident_tokens_already_current,
            )
        elif use_greedy_token_fastpath and decode_burst_steps > 1 and carry_device_tokens:
                self._record_device_token_carry(
                    snapshot_batch,
                    output.activations[:, -1:],
                    active_rows=active_rows,
                    prefill_final_flags=prefill_final_flags,
                    seqs=seqs,
                )
        else:
            self._clear_device_token_carry()
        if batch.is_prefill and last_logits is not None:
            prefill_logits_by_seq = getattr(self, "_last_prefill_logits_by_seq", None)
            if prefill_logits_by_seq is None:
                prefill_logits_by_seq = {}
                self._last_prefill_logits_by_seq = prefill_logits_by_seq
            for row, seq in enumerate(seqs):
                if row in active_rows and row < len(prefill_final_flags) and prefill_final_flags[row]:
                    prefill_logits_by_seq[int(seq.seq_id)] = last_logits[row]

        token_by_row: dict[int, Any] = {}
        token_list_by_row: dict[int, list[int]] = {}
        if active_rows:
            if decode_burst_steps > 1:
                token_rows = token_ids_all
                if active_rows != list(range(len(seqs))):
                    token_rows = token_rows[jnp.array(active_rows, dtype=jnp.int32)]
                if carry_device_tokens:
                    burst_width = int(token_rows.shape[1])
                    token_list_by_row = {
                        row: [
                            DeviceTokenRef(tokens=token_rows, row=index * burst_width + step)
                            for step in range(burst_width)
                        ]
                        for index, row in enumerate(active_rows)
                    }
                else:
                    token_list_by_row = {
                        row: [int(token_id) for token_id in token_row]
                        for row, token_row in zip(active_rows, token_rows.tolist())
                    }
            elif use_greedy_token_fastpath or use_sampled_token_fastpath:
                if active_rows == list(range(len(seqs))):
                    token_ids = token_ids_all
                else:
                    token_ids = token_ids_all[jnp.array(active_rows, dtype=jnp.int32)]
            elif token_ids_all is not None:
                if active_rows == list(range(len(seqs))):
                    token_ids = token_ids_all
                else:
                    token_ids = token_ids_all[jnp.array(active_rows, dtype=jnp.int32)]
            else:
                active_idx = jnp.array(active_rows, dtype=jnp.int32)
                temperatures = jnp.array([seqs[row].temperature for row in active_rows], dtype=jnp.float32)
                token_ids = self._sample_fn(last_logits[active_idx], temperatures)
            if decode_burst_steps <= 1:
                if carry_device_tokens:
                    token_by_row = {
                        row: DeviceTokenRef(tokens=token_ids, row=index)
                        for index, row in enumerate(active_rows)
                    }
                else:
                    host_token_ids = (
                        token_ids[:, 0]
                        if getattr(token_ids, "ndim", 0) == 2 and int(token_ids.shape[1]) == 1
                        else token_ids
                    )
                    token_by_row = {
                        row: int(token_id)
                        for row, token_id in zip(active_rows, host_token_ids.tolist())
                    }
                    _prefill_seed_mark("token_host_transfer")

        outputs: List[int | List[int]] = []
        for row, seq in enumerate(seqs):
            if row not in token_by_row and row not in token_list_by_row:
                outputs.append([])
                continue
            if batch.is_prefill and not prefill_final_flags[row]:
                outputs.append([])
                continue

            emitted = token_list_by_row[row] if row in token_list_by_row else token_by_row[row]
            outputs.append(emitted)

            if seed_hidden is not None:
                token_id = emitted[0] if isinstance(emitted, list) else emitted
                self._seed_mtp1_drafts([seq], seed_hidden[row : row + 1], [token_id])
                _prefill_seed_mark("seed_mtp1_drafts")
            elif prefill_mtp_draft_tokens is not None:
                if (
                    self.mtp1_enabled
                    and seq.temperature == 0
                    and self._seq_mtp_admitted(seq)
                    and seq.num_completion_tokens + 1 < seq.max_tokens
                    and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                ):
                    draft_row = prefill_mtp_draft_tokens[row]
                    if getattr(prefill_mtp_draft_tokens, "ndim", 0) == 1:
                        draft_value = int(draft_row.item())
                        draft_count = 1
                    else:
                        draft_chain = [int(token) for token in draft_row.tolist()]
                        draft_value = draft_chain if len(draft_chain) > 1 else draft_chain[0]
                        draft_count = len(draft_chain)
                    self._mtp1_drafts[seq.seq_id] = draft_value
                    self._mtp1_seeded_chain[seq.seq_id] = 0
                    self._speculative_stats()["drafts_proposed"] += draft_count
                else:
                    self._mtp1_drafts.pop(seq.seq_id, None)
                    self._mtp1_seeded_chain.pop(seq.seq_id, None)
                _prefill_seed_mark("store_fused_mtp_draft")

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

        def _draft_token_to_int(token: object) -> int:
            if isinstance(token, DeviceTokenRef):
                token_array = jnp.asarray(token.tokens, dtype=jnp.int32).reshape(-1)
                return int(jax.device_get(token_array[int(token.row)]))
            return int(token)

        for row, (seq, target_token) in enumerate(zip(seqs, target_tokens)):
            draft_value = self._mtp1_drafts.pop(seq.seq_id, None)
            draft_tokens = (
                [_draft_token_to_int(token) for token in draft_value]
                if isinstance(draft_value, list)
                else ([_draft_token_to_int(draft_value)] if draft_value is not None else [])
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

    def _run_main_and_append_unverified_mtp1_draft(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        admitted_rows: List[int],
    ) -> List[int | List[int]]:
        """Emit one unverified MTP draft after the normal target token.

        This is an experimental speed probe, not exact speculative decoding: the
        appended MTP token is not verified by the target model before emission.
        The cache/hybrid state is still consistent for the next step because the
        target decode has committed state through the first emitted token.
        """
        use_fused_append = _unverified_mtp_append_enabled(
            getattr(self, "config", None),
            "mtp_unverified_fused_append",
            "NANO_VLLM_JAX_MTP_UNVERIFIED_FUSED_APPEND",
        )
        if (
            use_fused_append
            and not batch.is_prefill
            and all(
                row < len(seqs)
                and seqs[row].temperature == 0
                and int(batch.query_lens[row]) == 1
                for row in admitted_rows
            )
        ):
            profile_unverified = os.environ.get(
                "NANO_VLLM_JAX_PROFILE_MTP_UNVERIFIED",
                "0",
            ) in {"1", "true", "yes", "on", "True"}
            t_unverified = time.perf_counter()

            def _mark_unverified(label: str) -> None:
                nonlocal t_unverified
                if not profile_unverified:
                    return
                now = time.perf_counter()
                print(
                    f"[MTP_UNVERIFIED] {label}={(now - t_unverified) * 1000:.3f}ms",
                    flush=True,
                )
                t_unverified = now

            use_resident_table_append = (
                bool(getattr(self, "resident_decode_metadata", False))
                and hasattr(self.executor, "forward_step_token_ids_mtp_draft_resident_table_jit")
                and getattr(self, "_hybrid_state_table", None) is not None
                and self._hybrid_state_table.conv_state is not None
                and self._hybrid_state_table.recurrent_state is not None
                and hasattr(self, "_resident_last_tokens")
            )
            if not use_resident_table_append:
                batch = self._maybe_apply_device_token_carry(batch)
            _mark_unverified("apply_device_token_carry")
            use_table_append = (
                (
                    use_resident_table_append
                    or hasattr(self.executor, "forward_step_token_ids_mtp_draft_table_jit")
                )
                and getattr(self, "_hybrid_state_table", None) is not None
                and self._hybrid_state_table.conv_state is not None
                and self._hybrid_state_table.recurrent_state is not None
            )
            if use_table_append:
                hybrid_slot_ids = self._batch_hybrid_slot_ids(batch)
                _mark_unverified("batch_hybrid_slot_ids")
                if use_resident_table_append:
                    output = self.executor.forward_step_token_ids_mtp_draft_resident_table_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state_table=self._hybrid_state_table,
                        hybrid_slot_ids=hybrid_slot_ids,
                        resident_last_tokens=self._resident_last_tokens,
                        mtp_hidden_final_normed=(self.mtp_hidden_source == "final_normed"),
                    )
                else:
                    output = self.executor.forward_step_token_ids_mtp_draft_table_jit(
                        batch,
                        cache_storage=self.cache_storage,
                        hybrid_state_table=self._hybrid_state_table,
                        hybrid_slot_ids=hybrid_slot_ids,
                        resident_last_tokens=self._resident_last_tokens,
                        mtp_hidden_final_normed=(self.mtp_hidden_source == "final_normed"),
                    )
                _mark_unverified("executor_table")
            elif hasattr(self.executor, "forward_step_token_ids_mtp_draft_jit"):
                hybrid_state = self._batch_hybrid_state(batch)
                _mark_unverified("batch_hybrid_state")
                output = self.executor.forward_step_token_ids_mtp_draft_jit(
                    batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    mtp_hidden_final_normed=(self.mtp_hidden_source == "final_normed"),
                )
                _mark_unverified("executor")
            else:
                return self._run_main_and_sample(seqs, batch, seed_mtp1=True)
            self.cache_storage = output.cache_storage
            if use_table_append:
                self._hybrid_state_table = output.hybrid_state
                resident_tokens_updated = output.resident_last_tokens is not None
                if output.resident_last_tokens is not None:
                    self._resident_last_tokens = output.resident_last_tokens
                self._mark_hybrid_slots_written(list(batch.hybrid_slot_ids_host or ()))
                _mark_unverified("store_table")
            else:
                resident_tokens_updated = False
                self._store_batch_hybrid_state(batch, output.hybrid_state)
                _mark_unverified("store_hybrid")
            self._record_kv_snapshot(batch, output.hybrid_state)
            _mark_unverified("record_kv_snapshot")

            token_rows = (
                output.activations
                if getattr(output.activations, "dtype", None) == jnp.dtype(jnp.int32)
                else output.activations.astype(jnp.int32)
            )
            outputs_by_row: dict[int, List[int] | int] = {}
            stats = self._speculative_stats()
            token_width = int(token_rows.shape[1]) if getattr(token_rows, "ndim", 0) == 2 else 1
            for row in admitted_rows:
                if row >= len(seqs):
                    continue
                seq = seqs[row]
                if (
                    int(batch.query_lens[row]) <= 0
                    or seq.temperature != 0
                    or seq.num_completion_tokens >= seq.max_tokens
                ):
                    continue
                emit_width = min(
                    token_width,
                    max(1, int(seq.max_tokens - seq.num_completion_tokens)),
                )
                token_refs = [
                    DeviceTokenRef(tokens=token_rows, row=row * token_width + offset)
                    for offset in range(emit_width)
                ]
                if emit_width > 1:
                    outputs_by_row[row] = token_refs
                    stats["drafts_proposed"] += emit_width - 1
                    stats["bonus_tokens"] += emit_width - 1
                else:
                    outputs_by_row[row] = token_refs[0]
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)

            if outputs_by_row:
                self._record_mtp_output_token_carry(
                    batch,
                    seqs,
                    outputs_by_row,
                    update_resident_tokens=not resident_tokens_updated,
                )
                _mark_unverified("record_token_carry")
            _mark_unverified("build_outputs")
            return [
                outputs_by_row[row] if row in outputs_by_row else []
                for row in range(len(seqs))
            ]

        outputs = self._run_main_and_sample(seqs, batch, seed_mtp1=True)
        if not admitted_rows:
            return outputs
        stats = self._speculative_stats()
        for row in admitted_rows:
            if row >= len(seqs) or row >= len(outputs):
                continue
            seq = seqs[row]
            emitted = outputs[row]
            if emitted == []:
                continue
            if (
                seq.temperature != 0
                or seq.num_completion_tokens + 2 > seq.max_tokens
                or int(batch.query_lens[row]) != 1
            ):
                self._mtp1_drafts.pop(seq.seq_id, None)
                continue
            draft_value = self._mtp1_drafts.pop(seq.seq_id, None)
            if draft_value is None:
                continue
            if isinstance(draft_value, list):
                if not draft_value:
                    continue
                draft_token = draft_value[0]
            else:
                draft_token = draft_value
            if isinstance(emitted, list):
                outputs[row] = emitted + [draft_token]  # type: ignore[list-item]
            else:
                outputs[row] = [emitted, draft_token]
            stats["bonus_tokens"] += 1
        return outputs

    def _run_mtp1_seed_then_table_burst(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        admitted_rows: List[int],
    ) -> dict[int, List[int] | int] | None:
        """Run the initial MTP seed and exact K=1 verifier groups in one JIT."""
        if (
            batch.is_prefill
            or not admitted_rows
            or self.num_speculative_tokens != 1
            or str(getattr(self, "mtp_verifier_impl", "two_decode") or "two_decode") != "two_decode"
            or not hasattr(self.executor, "mtp1_seed_then_table_burst_step_jit")
            or getattr(self, "_hybrid_state_table", None) is None
            or self._hybrid_state_table.conv_state is None
            or self._hybrid_state_table.recurrent_state is None
        ):
            return None

        query_lens_host = batch.query_lens_host
        configured_burst_groups = max(
            1,
            int(
                os.environ.get(
                    "NANO_VLLM_JAX_MTP_BURST_GROUPS",
                    str(getattr(self, "mtp_burst_groups", 1)),
                )
                or "1"
            ),
        )
        if configured_burst_groups <= 1:
            return None
        if os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_MULTI_GROUP_BURST", "0") not in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            raise RuntimeError(
                "MTP seed-then-table burst_groups>1 is not correctness-safe "
                "without trace-step materialization; use mtp_burst_groups=1 "
                "or set NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_MULTI_GROUP_BURST=1 "
                "for diagnostics only."
            )
        seed_burst_groups = max(1, configured_burst_groups - 1)
        max_emit_tokens = 1 + 2 * seed_burst_groups
        relax_bonus_boundary = os.environ.get(
            "NANO_VLLM_JAX_MTP_RELAX_BONUS_BOUNDARY",
            "0",
        ) in {"1", "true", "yes", "on", "True"}
        active_rows: List[int] = []
        for row in admitted_rows:
            if row < 0 or row >= len(seqs):
                continue
            seq = seqs[row]
            query_len = (
                int(query_lens_host[row])
                if query_lens_host is not None and row < len(query_lens_host)
                else int(batch.query_lens[row])
            )
            required_blocks = (seq.num_tokens + max_emit_tokens - 1 + self.block_size - 1) // self.block_size
            unsafe_bonus_boundary = (
                not relax_bonus_boundary
                and (seq.num_tokens + max_emit_tokens) % self.block_size == 0
            )
            if (
                seq.seq_id not in self._mtp1_drafts
                and seq.temperature == 0
                and seq.ignore_eos
                and self._seq_mtp_admitted(seq)
                and query_len == 1
                and seq.num_completion_tokens + max_emit_tokens <= seq.max_tokens
                and len(seq.block_table) >= required_blocks
                and not unsafe_bonus_boundary
            ):
                active_rows.append(row)
        if not active_rows:
            return None

        profile_mtp = os.environ.get("NANO_VLLM_JAX_PROFILE_MTP_RUN", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        t_profile = time.perf_counter()

        def _mark(label: str) -> None:
            nonlocal t_profile
            if not profile_mtp:
                return
            now = time.perf_counter()
            print(f"[MTP_RUN] seed_table_burst_{label}={(now - t_profile) * 1000:.3f}ms", flush=True)
            t_profile = now

        batch = self._materialize_static_decode_metadata_batch(batch)
        _mark("materialize_static_decode_metadata")
        if active_rows != list(range(int(batch.tokens.shape[0]))):
            decode_batch = self._masked_decode_batch(batch, active_rows)
            verifier_index_for_local = list(range(len(active_rows)))
            _mark("masked_batch")
        else:
            decode_batch = batch
            verifier_index_for_local = list(active_rows)
        decode_batch = self._maybe_apply_device_token_carry(decode_batch)
        _mark("apply_device_token_carry")
        hybrid_slot_ids = self._batch_hybrid_slot_ids(decode_batch)
        if profile_mtp:
            _block_until_ready_tree(hybrid_slot_ids)
        _mark("batch_hybrid_slot_ids")
        output = self.executor.mtp1_seed_then_table_burst_step_jit(
            decode_batch,
            cache_storage=self.cache_storage,
            hybrid_state_table=self._hybrid_state_table,
            hybrid_slot_ids=hybrid_slot_ids,
            mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            burst_groups=seed_burst_groups,
        )
        if profile_mtp:
            _block_until_ready_tree(output)
        _mark("executor")

        self.cache_storage = output.cache_storage
        committed_batch = self._with_committed_seq_lens(
            decode_batch,
            output.committed_seq_lens,
        )
        self._record_resident_committed_seq_lens(committed_batch)
        self._hybrid_state_table = output.hybrid_state
        self._mark_hybrid_slots_written(list(decode_batch.hybrid_slot_ids_host or ()))
        _mark("install_hybrid_table")
        self._record_kv_snapshot(committed_batch, output.hybrid_state)
        _mark("record_kv_snapshot")

        verifier_batch_size = int(decode_batch.tokens.shape[0])
        emitted_width = 1 + 2 * seed_burst_groups
        emitted_tokens = output.emitted_tokens.astype(jnp.int32).reshape(
            (verifier_batch_size, emitted_width)
        )
        next_draft_tokens = output.next_draft_token.astype(jnp.int32)
        compact_summary = getattr(output, "compact_summary", None)
        if compact_summary is not None:
            compact_summary_host = jax.device_get(compact_summary)
            emitted_totals_host = compact_summary_host[:, 0]
            accepted_totals_host = compact_summary_host[:, 1]
            rejected_totals_host = compact_summary_host[:, 2]
            bonus_totals_host = compact_summary_host[:, 3]
            accepted_bitmask_host = compact_summary_host[:, 4]
        else:
            emitted_counts_host, accepted_counts_host = jax.device_get(
                (
                    output.emitted_counts.reshape((verifier_batch_size, seed_burst_groups)),
                    output.accepted_counts.reshape((verifier_batch_size, seed_burst_groups)),
                )
            )
            emitted_totals_host = 1 + np.sum(emitted_counts_host, axis=1)
            accepted_totals_host = np.sum(accepted_counts_host, axis=1)
            rejected_totals_host = np.sum((accepted_counts_host < 1).astype(np.int32), axis=1)
            bonus_totals_host = accepted_totals_host
            bit_values = np.left_shift(
                np.ones((seed_burst_groups,), dtype=np.int32),
                np.arange(seed_burst_groups, dtype=np.int32),
            )
            accepted_bitmask_host = np.sum((accepted_counts_host > 0).astype(np.int32) * bit_values[None, :], axis=1)
        _mark("host_burst_summary_transfer")

        outputs: dict[int, List[int] | int] = {}
        stats = self._speculative_stats()
        accepted_matrix: list[list[bool]] = []
        max_seeded_chain = int(os.environ.get("NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN", "0") or "0")
        row_to_committed_len_host: dict[int, int] = {}
        seq_lens_host = decode_batch.seq_lens_host
        for local_row, row in enumerate(active_rows):
            seq = seqs[row]
            verifier_idx = verifier_index_for_local[local_row]
            self._mtp1_drafts.pop(seq.seq_id, None)
            emitted_total = max(
                0,
                min(emitted_width, int(emitted_totals_host[verifier_idx])),
            )
            accepted_total = max(
                0,
                min(seed_burst_groups, int(accepted_totals_host[verifier_idx])),
            )
            rejected_total = max(
                0,
                min(seed_burst_groups, int(rejected_totals_host[verifier_idx])),
            )
            bonus_total = max(
                0,
                min(seed_burst_groups, int(bonus_totals_host[verifier_idx])),
            )
            row_outputs: list[object] = [
                DeviceTokenRef(tokens=emitted_tokens, row=verifier_idx * emitted_width + offset)
                for offset in range(emitted_total)
            ]
            bitmask = int(accepted_bitmask_host[verifier_idx])
            accepted_matrix.extend(
                [bool(bitmask & (1 << group_idx))]
                for group_idx in range(seed_burst_groups)
            )
            outputs[row] = row_outputs
            if seq_lens_host is not None and verifier_idx < len(seq_lens_host):
                row_to_committed_len_host[verifier_idx] = (
                    int(seq_lens_host[verifier_idx]) + emitted_total
                )
            stats["drafts_proposed"] += seed_burst_groups
            stats["drafts_accepted"] += accepted_total
            stats["drafts_rejected"] += rejected_total
            stats["bonus_tokens"] += bonus_total
            if (
                self.mtp1_enabled
                and seq.temperature == 0
                and seq.num_completion_tokens + emitted_total < seq.max_tokens
                and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                and (
                    max_seeded_chain <= 0
                    or self._mtp1_seeded_chain.get(seq.seq_id, 0) < max_seeded_chain
                )
            ):
                self._mtp1_drafts[seq.seq_id] = DeviceTokenRef(
                    tokens=next_draft_tokens,
                    row=verifier_idx,
                )
                self._mtp1_seeded_chain[seq.seq_id] = (
                    self._mtp1_seeded_chain.get(seq.seq_id, 0) + seed_burst_groups
                )
                stats["drafts_proposed"] += 1
            else:
                self._mtp1_seeded_chain.pop(seq.seq_id, None)

        self._record_draft_position_acceptance(accepted_matrix)
        if not ModelRunner._device_token_carry_enabled(self):
            outputs = ModelRunner._materialize_device_token_outputs(outputs)
        self._record_resident_committed_seq_lens_host(
            committed_batch,
            row_to_committed_len_host,
        )
        self._record_mtp_output_token_carry(committed_batch, seqs, outputs)
        self._mtp_carry_recorded_this_call = True
        _mark("commit")
        return outputs

    def _run_main_and_seed_mtp_chain_fused(
        self,
        seqs: List[Sequence],
        batch: ScheduledBatch,
        admitted_rows: List[int],
    ) -> List[int | List[int]] | None:
        """Decode one target token and seed MTP drafts without a host hidden round-trip."""
        if (
            batch.is_prefill
            or not admitted_rows
            or not hasattr(self.executor, "forward_step_token_ids_mtp_draft_chain_jit")
        ):
            return None
        draft_len = max(1, int(getattr(self, "num_speculative_tokens", 1) or 1))
        if draft_len < 1:
            return None
        query_lens_host = batch.query_lens_host
        active_rows = [
            row
            for row in admitted_rows
            if row < len(seqs)
            and seqs[row].temperature == 0
            and self._seq_mtp_admitted(seqs[row])
            and (
                int(query_lens_host[row])
                if query_lens_host is not None and row < len(query_lens_host)
                else int(batch.query_lens[row])
            ) == 1
            and seqs[row].num_completion_tokens + 1 < seqs[row].max_tokens
        ]
        if not active_rows:
            return None

        profile_mtp = os.environ.get("NANO_VLLM_JAX_PROFILE_MTP_RUN", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        t_profile = time.perf_counter()

        def _mark(label: str) -> None:
            nonlocal t_profile
            if not profile_mtp:
                return
            now = time.perf_counter()
            print(f"[MTP_RUN] fused_seed_{label}={(now - t_profile) * 1000:.3f}ms", flush=True)
            t_profile = now

        batch = self._materialize_static_decode_metadata_batch(batch)
        _mark("materialize_static_decode_metadata")
        token_row_for_row = {int(row): int(row) for row in active_rows}
        if active_rows != list(range(int(batch.tokens.shape[0]))):
            original_active_rows = list(active_rows)
            batch = self._compact_decode_batch(batch, active_rows)
            token_row_for_row = {
                int(row): int(local_row)
                for local_row, row in enumerate(original_active_rows)
            }
            _mark("compact_batch")
        batch = self._maybe_apply_device_token_carry(batch)
        _mark("apply_device_token_carry")
        hybrid_state = self._batch_hybrid_state(batch)
        _mark("batch_hybrid_state")
        parity_output = None
        seed_parity_debug = (
            os.environ.get("NANO_VLLM_JAX_MTP_SEED_PARITY_DEBUG", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "forward_step_token_ids_jit")
        )
        if seed_parity_debug:
            parity_cache_storage = KVCacheStorage(
                self.cache_storage.k_cache.copy(),
                self.cache_storage.v_cache.copy(),
            )
            parity_hybrid_state = HybridLayerState(
                conv_state=hybrid_state.conv_state.copy()
                if hybrid_state.conv_state is not None
                else None,
                recurrent_state=hybrid_state.recurrent_state.copy()
                if hybrid_state.recurrent_state is not None
                else None,
            )
            parity_output = self.executor.forward_step_token_ids_jit(
                batch,
                cache_storage=parity_cache_storage,
                hybrid_state=parity_hybrid_state,
            )
            _block_until_ready_tree(parity_output)
            _mark("parity_executor")
        output = self.executor.forward_step_token_ids_mtp_draft_chain_jit(
            batch,
            cache_storage=self.cache_storage,
            hybrid_state=hybrid_state,
            mtp_hidden_final_normed=(self.mtp_hidden_source == "final_normed"),
            mtp_chain_return_normed=(self.mtp_chain_hidden_source == "final_normed"),
            draft_len=draft_len,
            mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
        )
        if profile_mtp:
            _block_until_ready_tree(output)
        _mark("executor")
        if parity_output is not None:
            seed_targets = output.resident_last_tokens
            if seed_targets is None:
                seed_targets = output.activations[:, :1]
            seed_target_vec = jnp.asarray(seed_targets, dtype=jnp.int32).reshape(-1)
            parity_target_vec = jnp.asarray(parity_output.activations, dtype=jnp.int32).reshape(-1)
            token_mismatch = jnp.any(seed_target_vec != parity_target_vec)

            slot_positions = jnp.maximum(batch.seq_lens - 1, 0).astype(jnp.int32)[:, None]
            seed_slots = compute_slot_mapping(
                positions=slot_positions,
                block_table=batch.block_tables,
                block_size=self.config.block_size,
                is_prefill=False,
            )[:, 0]

            def _slot_max_abs(left, right, slots) -> float:
                leading_shape = left.shape[:-4] if left.ndim == 5 else left.shape[:-3]
                flat_left = left.reshape(leading_shape + (-1,) + left.shape[-2:])
                flat_right = right.reshape(leading_shape + (-1,) + right.shape[-2:])
                left_values = flat_left[..., slots, :, :].astype(jnp.float32)
                right_values = flat_right[..., slots, :, :].astype(jnp.float32)
                return float(jnp.max(jnp.abs(left_values - right_values)).item())

            def _state_max_abs(left, right) -> float:
                if left is None or right is None:
                    return 0.0
                return float(jnp.max(jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))).item())

            k_slot_diff = _slot_max_abs(
                output.cache_storage.k_cache,
                parity_output.cache_storage.k_cache,
                seed_slots,
            )
            v_slot_diff = _slot_max_abs(
                output.cache_storage.v_cache,
                parity_output.cache_storage.v_cache,
                seed_slots,
            )
            conv_diff = _state_max_abs(
                output.hybrid_state.conv_state,
                parity_output.hybrid_state.conv_state,
            )
            recurrent_diff = _state_max_abs(
                output.hybrid_state.recurrent_state,
                parity_output.hybrid_state.recurrent_state,
            )
            print(
                "[MTP_SEED_PARITY] "
                f"rows={active_rows} seq_ids={tuple(int(x) for x in (batch.seq_ids_host or ())) } "
                f"tokens_seed={seed_target_vec.tolist()} "
                f"tokens_normal={parity_target_vec.tolist()} "
                f"token_mismatch={bool(token_mismatch.item())} "
                f"k_slot_max_abs={k_slot_diff:.6g} "
                f"v_slot_max_abs={v_slot_diff:.6g} "
                f"conv_max_abs={conv_diff:.6g} "
                f"recurrent_max_abs={recurrent_diff:.6g}",
                flush=True,
            )
        self.cache_storage = output.cache_storage
        committed_batch = self._with_committed_seq_lens(
            batch,
            batch.seq_lens + batch.query_lens.astype(jnp.int32),
        )
        self._record_resident_committed_seq_lens(committed_batch)
        self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
        _mark("store_hybrid_state")
        self._record_kv_snapshot(committed_batch, output.hybrid_state)
        _mark("record_kv_snapshot")

        token_rows = output.activations
        width = int(token_rows.shape[1])
        target_tokens = output.resident_last_tokens
        if target_tokens is None:
            target_tokens = token_rows[:, :1]
        chain_debug_payload = getattr(output, "debug_payload", None)
        chain_debug_host = None
        token_rows_host = None
        if chain_debug_payload is not None:
            chain_debug_host = jax.device_get(chain_debug_payload)
            token_rows_host = jax.device_get(token_rows)
        carry_by_seq_id: dict[int, DeviceTokenRef] = dict(
            getattr(self, "_device_token_carry_by_seq_id", {})
        )
        stale_seq_ids = getattr(self, "_resident_last_tokens_stale_seq_ids", None)
        for row in active_rows:
            token_row = token_row_for_row[int(row)]
            seq_id = int(batch.seq_ids_host[token_row]) if batch.seq_ids_host is not None else -1
            if seq_id < 0:
                continue
            carry_by_seq_id[seq_id] = DeviceTokenRef(tokens=target_tokens, row=token_row)
            if stale_seq_ids is not None:
                stale_seq_ids.add(seq_id)
        self._device_token_carry_seq_ids = None
        self._device_token_carry_tokens = None
        self._device_token_carry_by_seq_id = carry_by_seq_id
        self._device_seq_lens_carry_seq_ids = None
        self._device_seq_lens_carry = None
        _mark("record_device_token_carry")

        outputs: List[int | List[int]] = [[] for _ in range(len(seqs))]
        stats = self._speculative_stats()
        for row in active_rows:
            seq = seqs[row]
            token_row = token_row_for_row[int(row)]
            outputs[row] = DeviceTokenRef(tokens=target_tokens, row=token_row)
            chain_refs = [
                DeviceTokenRef(tokens=token_rows, row=token_row * width + offset)
                for offset in range(1, width)
            ]
            can_verify_next_chain = (
                bool(chain_refs)
                and seq.num_completion_tokens + draft_len + 2 <= seq.max_tokens
            )
            if can_verify_next_chain:
                self._mtp1_drafts[seq.seq_id] = chain_refs if len(chain_refs) > 1 else chain_refs[0]
                self._mtp1_seeded_chain[seq.seq_id] = 0
                stats["drafts_proposed"] += len(chain_refs)
                if profile_mtp:
                    _draft_debug, debug_events = self._mtp1_debug_state()
                    debug_events.append(
                        {
                            "kind": "mtp_seeded_main",
                            "seq_id": int(seq.seq_id),
                            "row": int(row),
                            "seq_tokens": int(seq.num_tokens),
                            "completion_tokens": int(seq.num_completion_tokens),
                            "target_ref_row": int(token_row),
                            "draft_chain": [
                                {"device_ref_row": int(ref.row)}
                                for ref in chain_refs
                            ],
                        }
                    )
                if chain_debug_host is not None and token_rows_host is not None:
                    draft_top_ids, draft_top_values = chain_debug_host
                    draft_debug, _ = self._mtp1_debug_state()
                    draft_debug[seq.seq_id] = {
                        "confirmed_token_id": int(token_rows_host[token_row, 0]),
                        "draft_chain": [
                            int(token_rows_host[token_row, offset])
                            for offset in range(1, width)
                        ],
                        "position": int(seqs[row].num_tokens),
                        "position_offset": int(getattr(self, "mtp_position_offset", 0)),
                        "token_source": str(getattr(self, "mtp_token_source", "generated")),
                        "hidden_source": str(getattr(self, "mtp_hidden_source", "final_normed")),
                        "mtp_chain_top": [
                            {
                                "ids": [
                                    int(value)
                                    for value in draft_top_ids[token_row, pos].tolist()
                                ],
                                "values": [
                                    float(value)
                                    for value in draft_top_values[token_row, pos].tolist()
                                ],
                            }
                            for pos in range(len(chain_refs))
                        ],
                    }
            else:
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
        _mark("build_outputs")
        if not ModelRunner._device_token_carry_enabled(self):
            output_by_row = {
                row: outputs[row]
                for row in active_rows
                if outputs[row] != []
            }
            materialized = ModelRunner._materialize_device_token_outputs(output_by_row)
            for row, value in materialized.items():
                outputs[row] = value
        return outputs

    def _seed_mtp1_drafts(
        self,
        seqs: List[Sequence],
        hidden: jnp.ndarray,
        confirmed_token_ids: List[Any],
        positions: List[int] | None = None,
    ):
        seed_rows: List[int] = []
        token_values: List[int] = []
        token_inputs: List[object] = []
        position_values: List[int] = []
        adaptive_gated = getattr(self, "_mtp_adaptive_gated", lambda: False)
        for row, seq in enumerate(seqs):
            confirmed_token_id = confirmed_token_ids[row]
            position = int(positions[row]) if positions is not None else seq.num_tokens
            if getattr(self, "mtp_token_source", "generated") == "current":
                confirmed_token_id = seq.last_token
                position = seq.num_tokens - 1
            if isinstance(confirmed_token_id, DeviceTokenRef):
                token_array = jnp.asarray(confirmed_token_id.tokens, dtype=jnp.int32).reshape(-1)
                token_input = token_array[int(confirmed_token_id.row)]
                token_debug_value = 0
            else:
                token_debug_value = int(confirmed_token_id)
                token_input = jnp.asarray(token_debug_value, dtype=jnp.int32)
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
            token_values.append(token_debug_value)
            token_inputs.append(token_input)
            position_values.append(int(position + int(getattr(self, "mtp_position_offset", 0))))

        if not seed_rows:
            return

        row_index = jnp.array(seed_rows, dtype=jnp.int32)
        hidden_input = hidden[row_index][:, None, :]
        token_input = jnp.stack([jnp.asarray(token, dtype=jnp.int32) for token in token_inputs])[:, None]
        position_input = jnp.array(position_values, dtype=jnp.int32)[:, None]
        draft_len = max(1, int(getattr(self, "num_speculative_tokens", 1) or 1))
        draft_margin_threshold = float(
            os.environ.get(
                "NANO_VLLM_JAX_MTP_DRAFT_MARGIN",
                getattr(self, "mtp_draft_margin", 0.0),
            )
            or "0"
        )
        draft_margin_values = None
        chain_logit_debug = os.environ.get(
            "NANO_VLLM_JAX_MTP_CHAIN_LOGIT_DEBUG",
            "0",
        ) in {"1", "true", "yes", "on", "True"}
        draft_top_ids = None
        draft_top_values = None
        if chain_logit_debug:
            draft_logits = None
            draft_chains, draft_top_ids, draft_top_values = self._mtp1_draft_chain_with_topk(
                hidden_state=hidden_input,
                token_ids=token_input,
                positions=position_input,
                draft_len=draft_len,
                top_k=5,
            )
        elif getattr(self, "mtp_debug", False):
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

        keep_drafts_on_device = (
            not getattr(self, "mtp_debug", False)
            and not chain_logit_debug
            and draft_margin_threshold <= 0
        )
        draft_chain_list = (
            None
            if keep_drafts_on_device
            else [[int(token) for token in chain] for chain in draft_chains.tolist()]
        )
        for local_row, row in enumerate(seed_rows):
            seq = seqs[row]
            if (
                draft_margin_values is not None
                and draft_margin_values[local_row] < draft_margin_threshold
            ):
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
                continue
            if keep_drafts_on_device:
                draft_chain = [
                    DeviceTokenRef(tokens=draft_chains, row=local_row * draft_len + pos)
                    for pos in range(draft_len)
                ]
            else:
                assert draft_chain_list is not None
                draft_chain = draft_chain_list[local_row]
            self._mtp1_drafts[seq.seq_id] = draft_chain if len(draft_chain) > 1 else draft_chain[0]
            self._mtp1_seeded_chain[seq.seq_id] = 0
            if chain_logit_debug:
                assert draft_top_ids is not None
                assert draft_top_values is not None
                draft_debug, _ = self._mtp1_debug_state()
                draft_debug[seq.seq_id] = {
                    "confirmed_token_id": int(token_values[local_row]),
                    "draft_chain": [int(token) for token in draft_chain],
                    "position": int(position_values[local_row] - int(getattr(self, "mtp_position_offset", 0))),
                    "position_offset": int(getattr(self, "mtp_position_offset", 0)),
                    "token_source": str(getattr(self, "mtp_token_source", "generated")),
                    "hidden_source": str(getattr(self, "mtp_hidden_source", "final_normed")),
                    "mtp_chain_top": [
                        {
                            "ids": [
                                int(value)
                                for value in draft_top_ids[local_row, draft_pos].tolist()
                            ],
                            "values": [
                                float(value)
                                for value in draft_top_values[local_row, draft_pos].tolist()
                            ],
                        }
                        for draft_pos in range(draft_len)
                    ],
                }
            elif getattr(self, "mtp_debug", False):
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
        draft_margin_threshold = float(
            os.environ.get(
                "NANO_VLLM_JAX_MTP_DRAFT_MARGIN",
                getattr(self, "mtp_draft_margin", 0.0),
            )
            or "0"
        )
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
        strict_k_verifier = self._strict_k_mtp_verifier_enabled()

        def _none_or_strict_error(reason: str) -> None:
            if strict_k_verifier:
                raise RuntimeError(
                    "K>1 MTP verifier fallback is disabled; grouped verifier "
                    f"could not run: {reason}"
                )
            return None

        forced_reject_rows = set(forced_reject_rows or ())
        for row in rows:
            if not bool(getattr(seqs[row], "mtp_admitted", True)):
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
                return _none_or_strict_error("row is not MTP-admitted")
        if getattr(self, "_mtp_adaptive_gated", lambda: False)():
            for row in rows:
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
            return _none_or_strict_error("adaptive MTP gate is active")
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
            _block_until_ready_tree(value)

        def _mark(label: str, start: float) -> float:
            if profile_mtp:
                now = time.perf_counter()
                print(f"[MTP_RUN] {label}={(now - start) * 1000:.3f}ms", flush=True)
                return now
            return start

        t_profile = time.perf_counter()
        batch = self._maybe_apply_device_token_carry(batch)
        t_profile = _mark("apply_device_token_carry", t_profile)
        batch = self._materialize_static_decode_metadata_batch(batch)
        t_profile = _mark("materialize_static_decode_metadata", t_profile)
        mtp_max_active_rows = int(
            getattr(
                self,
                "mtp_max_active_rows",
                getattr(getattr(self, "config", None), "mtp_max_active_rows", 0),
            )
            or 0
        )
        use_static_verifier_batch = (
            mtp_max_active_rows > 0 and len(seqs) <= mtp_max_active_rows
        )
        static_verifier_rows = (
            self._mtp_static_batch_size(len(seqs))
            if hasattr(self, "_mtp_static_batch_size")
            else len(seqs)
        )
        if static_verifier_rows > int(batch.tokens.shape[0]):
            batch = self._pad_decode_batch_to_rows(batch, static_verifier_rows)
            t_profile = _mark("pad_static_verifier_batch", t_profile)
        if os.environ.get("NANO_VLLM_JAX_MTP_FUSED_VERIFY", "1") not in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }:
            return _none_or_strict_error("NANO_VLLM_JAX_MTP_FUSED_VERIFY is disabled")
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
                return _none_or_strict_error("fused verifier unavailable for batch shape")
            for row in rows:
                draft_value = self._mtp1_drafts.get(seqs[row].seq_id)
                if isinstance(draft_value, list) and len(draft_value) > 1:
                    return _none_or_strict_error("scalar MTP path received a K-chain draft")
            return {
                row: self._run_mtp1([seqs[row]], self._slice_batch(batch, row))[0]
                for row in rows
            }

        mtp_seqs = [seqs[row] for row in rows]
        draft_chains: List[List[object]] = []
        for row, seq in zip(rows, mtp_seqs):
            if row in forced_reject_rows:
                draft_chains.append([-1])
                continue
            draft_value = self._mtp1_drafts[seq.seq_id]
            if isinstance(draft_value, list):
                draft_chains.append([
                    token if isinstance(token, DeviceTokenRef) else int(token)
                    for token in draft_value
                ])
            elif isinstance(draft_value, DeviceTokenRef):
                draft_chains.append([draft_value])
            else:
                draft_chains.append([int(draft_value)])
        draft_len = min(len(chain) for chain in draft_chains)
        draft_len = min(draft_len, max(1, int(getattr(self, "num_speculative_tokens", 1) or 1)))
        if draft_len < 1:
            return {}
        emit_bonus = True
        if strict_k_verifier and draft_len > 1:
            remaining_tokens = [
                max(0, int(seq.max_tokens - seq.num_completion_tokens))
                for seq in mtp_seqs
            ]
            min_remaining_tokens = min(remaining_tokens) if remaining_tokens else 0
            if min_remaining_tokens <= 0:
                return {}
            bonus_boundary_no_bonus = any(
                (seq.num_tokens + draft_len + 1) % self.block_size == 0
                for seq in mtp_seqs
            )
            if min_remaining_tokens < draft_len + 1 or bonus_boundary_no_bonus:
                draft_len = min(draft_len, min_remaining_tokens)
                emit_bonus = False
            if draft_len < 1:
                return {}
            if any(remaining < draft_len + (1 if emit_bonus else 0) for remaining in remaining_tokens):
                return _none_or_strict_error(
                    "MTP verifier would exceed remaining max-token budget"
                )
        t_profile = _mark("draft_setup", t_profile)
        required_blocks_by_row = {
            row: (seqs[row].num_tokens + draft_len + self.block_size - 1) // self.block_size
            for row in rows
        }
        missing_capacity = [
            (row, required_blocks, len(seqs[row].block_table))
            for row, required_blocks in required_blocks_by_row.items()
            if len(seqs[row].block_table) < required_blocks
        ]
        if missing_capacity:
            return _none_or_strict_error(
                "scheduler did not allocate MTP verifier lookahead blocks "
                f"{missing_capacity}"
            )
        max_required_blocks = max(required_blocks_by_row.values()) if required_blocks_by_row else 0
        if int(batch.block_tables.shape[1]) < max_required_blocks:
            return _none_or_strict_error(
                "scheduled decode batch block-table width "
                f"{int(batch.block_tables.shape[1])} is smaller than verifier "
                f"requirement {max_required_blocks}"
            )
        draft_token_chains = [chain[:draft_len] for chain in draft_chains]
        draft_tokens = [chain[0] for chain in draft_token_chains]
        force_reject_mtp = os.environ.get("NANO_VLLM_JAX_MTP_FORCE_REJECT", "0") in {
            "1",
            "true",
            "yes",
            "on",
            "True",
        }
        if force_reject_mtp:
            draft_token_chains = [[-1 for _ in range(draft_len)] for _ in draft_token_chains]
            draft_tokens = [-1 for _ in draft_tokens]
        verifier_impl = str(getattr(self, "mtp_verifier_impl", "two_decode") or "two_decode")
        use_packed_prefix_verifier = verifier_impl in {
            "packed_prefix",
            "packed_prefill",
            "prefill_packed",
        }
        force_generic_k_verifier = os.environ.get(
            "NANO_VLLM_JAX_MTP_FORCE_GENERIC_K",
            "0",
        ) in {"1", "true", "yes", "on", "True"}
        use_generic_k_verifier = (
            force_generic_k_verifier
            or verifier_impl in {"k_decode", "generic_k", "expanded"}
            or use_packed_prefix_verifier
        )
        compact_verifier_enabled = (
            os.environ.get("NANO_VLLM_JAX_MTP_COMPACT_VERIFIER", "1")
            in {"1", "true", "yes", "on", "True"}
        )
        if strict_k_verifier and draft_len > 1:
            # Compacting strict K rows changes verifier row identity and has
            # shown state drift on GPU. Keep the physical batch shape and mask
            # inactive rows instead; this still exercises the grouped verifier.
            compact_verifier_enabled = False
        if (
            use_static_verifier_batch
            and static_verifier_rows > len(rows)
            and not (draft_len > 1 and use_generic_k_verifier)
        ):
            compact_verifier_enabled = False
        use_compact_verifier = (
            partial_physical_batch
            and not use_debug
            and not debug_verifier_enabled
            and compact_verifier_enabled
            and (
                draft_len == 1
                or (
                    draft_len > 1
                    and use_generic_k_verifier
                    and (
                        hasattr(self.executor, "mtp_k_decode_greedy_step_jit")
                        or hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
                    )
                )
                or (
                    draft_len == 2
                    and verifier_impl == "commit_select"
                    and hasattr(self.executor, "mtp2_commit_select_greedy_step_jit")
                )
            )
        )
        if use_compact_verifier:
            decode_batch = self._compact_decode_batch(batch, rows)
        elif rows == list(range(physical_batch_size)):
            decode_batch = batch
        else:
            decode_batch = self._masked_decode_batch(batch, rows)
        t_profile = _mark("decode_batch_setup", t_profile)
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
        next_mtp_positions = None

        def _next_mtp_positions_device():
            nonlocal next_mtp_positions
            if next_mtp_positions is None:
                next_mtp_positions = jnp.asarray(next_mtp_positions_for_batch, dtype=jnp.int32)
            return next_mtp_positions

        verifier_draft_tokens_device = None

        def _verifier_draft_tokens_device():
            nonlocal verifier_draft_tokens_device
            if verifier_draft_tokens_device is None:
                values = jnp.zeros((verifier_physical_batch_size,), dtype=jnp.int32)
                for idx, token in enumerate(verifier_draft_tokens_for_batch):
                    if isinstance(token, DeviceTokenRef):
                        token_array = jnp.asarray(token.tokens, dtype=jnp.int32).reshape(-1)
                        token_value = token_array[int(token.row)]
                    else:
                        token_value = jnp.asarray(int(token), dtype=jnp.int32)
                    values = values.at[idx].set(token_value)
                verifier_draft_tokens_device = values
            return verifier_draft_tokens_device

        verifier_draft_token_chains_device = None

        def _verifier_draft_token_chains_device():
            nonlocal verifier_draft_token_chains_device
            if verifier_draft_token_chains_device is None:
                dense_tokens = None
                dense_refs = draft_len > 0
                for idx, chain in enumerate(draft_token_chains_for_batch):
                    for pos, token in enumerate(chain[:draft_len]):
                        if not isinstance(token, DeviceTokenRef):
                            dense_refs = False
                            break
                        if dense_tokens is None:
                            dense_tokens = token.tokens
                        elif token.tokens is not dense_tokens:
                            dense_refs = False
                            break
                        if int(token.row) != idx * draft_len + pos:
                            dense_refs = False
                            break
                    if not dense_refs:
                        break
                if dense_refs and dense_tokens is not None:
                    dense_values = jnp.asarray(dense_tokens, dtype=jnp.int32).reshape(
                        -1,
                        draft_len,
                    )
                    if int(dense_values.shape[0]) >= verifier_physical_batch_size:
                        verifier_draft_token_chains_device = dense_values[
                            :verifier_physical_batch_size,
                            :,
                        ]
                        return verifier_draft_token_chains_device
                values = jnp.zeros(
                    (verifier_physical_batch_size, draft_len),
                    dtype=jnp.int32,
                )
                for idx, chain in enumerate(draft_token_chains_for_batch):
                    for pos, token in enumerate(chain[:draft_len]):
                        if isinstance(token, DeviceTokenRef):
                            token_array = jnp.asarray(token.tokens, dtype=jnp.int32).reshape(-1)
                            token_value = token_array[int(token.row)]
                        else:
                            token_value = jnp.asarray(int(token), dtype=jnp.int32)
                        values = values.at[idx, pos].set(token_value)
                verifier_draft_token_chains_device = values
            return verifier_draft_token_chains_device

        t_profile = _mark("verifier_inputs_host_setup", t_profile)
        force_commit_select = (
            os.environ.get(
                "NANO_VLLM_JAX_MTP_COMMIT_SELECT",
                "1" if verifier_impl == "commit_select" else "0",
            )
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
            os.environ.get(
                "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1",
                "1" if verifier_impl == "two_decode" else "0",
            )
            in {"1", "true", "yes", "on", "True"}
        )
        allow_mixed_fused_k1 = (
            os.environ.get(
                "NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED",
                "1" if verifier_impl == "two_decode" else "0",
            )
            in {"1", "true", "yes", "on", "True"}
        )
        seed_after_bonus_enabled = (
            os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
            )
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
        batch_accept_policy = os.environ.get(
            "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
            str(getattr(self, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
        )
        enable_rowwise_repair = (
            os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_ROWWISE_REPAIR", "1")
            in {"1", "true", "yes", "on", "True"}
        )
        use_one_pass_table_k1 = (
            draft_len == 1
            and verifier_impl == "two_decode"
            and not use_generic_k_verifier
            and not force_commit_select
            and not disable_one_pass_k1
            and enable_one_pass_k1
            and getattr(self, "_hybrid_state_table", None) is not None
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and hasattr(self.executor, "mtp1_two_decode_greedy_table_step_jit")
        )
        use_one_pass_k1 = (
            draft_len == 1
            and not use_one_pass_table_k1
            and not enable_fast_all_accept
            and not use_generic_k_verifier
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
            and not use_generic_k_verifier
            and hasattr(self.executor, "mtp1_commit_select_greedy_step_jit")
        )
        verifier_full_physical_batch = (
            (not partial_physical_batch)
            or use_compact_verifier
            or use_static_verifier_batch
        )
        mtp_burst_groups = max(
            1,
            int(
                os.environ.get(
                    "NANO_VLLM_JAX_MTP_BURST_GROUPS",
                    str(getattr(self, "mtp_burst_groups", 1)),
                )
                or "1"
            ),
        )
        if (
            mtp_burst_groups > 1
            and not use_packed_prefix_verifier
            and os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_MULTI_GROUP_BURST", "0")
            not in {"1", "true", "yes", "on", "True"}
        ):
            raise RuntimeError(
                "MTP broad verifier burst_groups>1 is not correctness-safe "
                "without trace-step materialization; use mtp_burst_groups=1 "
                "or set "
                "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_MULTI_GROUP_BURST=1 for "
                "diagnostics only."
            )
        use_mtp2_commit_select = (
            draft_len == 2
            and mtp_burst_groups <= 1
            and verifier_impl == "commit_select"
            and not use_generic_k_verifier
            and verifier_full_physical_batch
            and hasattr(self.executor, "mtp2_commit_select_greedy_step_jit")
        )
        burst_emit_tokens = mtp_burst_groups * (draft_len + 1)
        burst_required_blocks = [
            (seq.num_tokens + burst_emit_tokens - 1 + self.block_size - 1) // self.block_size
            for seq in mtp_seqs
        ]
        burst_final_bonus_boundary = any(
            (seq.num_tokens + burst_emit_tokens) % self.block_size == 0
            for seq in mtp_seqs
        )
        burst_fits_completion_budget = all(
            seq.num_completion_tokens + burst_emit_tokens <= seq.max_tokens
            for seq in mtp_seqs
        )
        burst_fits_with_final_bonus_clamp = all(
            seq.num_completion_tokens + burst_emit_tokens - 1 <= seq.max_tokens
            for seq in mtp_seqs
        )
        packed_prefix_burst_groups = 1
        if (
            use_packed_prefix_verifier
            and mtp_burst_groups > 1
            and bool(getattr(self, "resident_decode_metadata", False))
            and getattr(self, "_hybrid_state_table", None) is not None
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and hasattr(self.executor, "mtp_k_packed_prefix_table_greedy_step_jit")
            and verifier_full_physical_batch
            and (
                burst_fits_completion_budget
                or burst_fits_with_final_bonus_clamp
            )
            and all(
                len(seq.block_table) >= required_blocks
                for seq, required_blocks in zip(mtp_seqs, burst_required_blocks)
            )
            and not burst_final_bonus_boundary
        ):
            packed_prefix_burst_groups = mtp_burst_groups
        if not emit_bonus:
            packed_prefix_burst_groups = 1
        table_burst_groups = 1
        if (
            use_one_pass_table_k1
            and mtp_burst_groups > 1
            and hasattr(self.executor, "mtp1_two_decode_greedy_table_burst_step_jit")
            and all(
                seq.num_completion_tokens + burst_emit_tokens <= seq.max_tokens
                for seq in mtp_seqs
            )
            and all(
                len(seq.block_table) >= required_blocks
                for seq, required_blocks in zip(mtp_seqs, burst_required_blocks)
            )
            and not burst_final_bonus_boundary
        ):
            table_burst_groups = mtp_burst_groups
        use_k_burst = (
            mtp_burst_groups > 1
            and (draft_len > 1 or use_generic_k_verifier)
            and not use_mtp2_commit_select
            and not use_packed_prefix_verifier
            and verifier_full_physical_batch
            and all(
                seq.num_completion_tokens + burst_emit_tokens <= seq.max_tokens
                for seq in mtp_seqs
            )
            and all(
                len(seq.block_table) >= required_blocks
                for seq, required_blocks in zip(mtp_seqs, burst_required_blocks)
            )
            and not burst_final_bonus_boundary
            and hasattr(self.executor, "mtp_k_burst_greedy_step_jit")
        )
        use_fast_all_accept = (
            draft_len == 1
            and not use_one_pass_k1
            and not use_one_pass_table_k1
            and not use_commit_select
            and enable_fast_all_accept
        )
        use_fast_table_verifier = (
            use_fast_all_accept
            and hasattr(self.executor, "mtp1_two_decode_greedy_fast_table_jit")
            and getattr(self, "_hybrid_state_table", None) is not None
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
        )
        use_burst_table_verifier = (
            use_fast_table_verifier
            and os.environ.get("NANO_VLLM_JAX_MTP_BURST_VERIFY_TABLE", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_burst_verify_table_jit")
        )
        use_greedy_burst_table_verifier = (
            use_fast_table_verifier
            and os.environ.get("NANO_VLLM_JAX_MTP_GREEDY_BURST_VERIFY_TABLE", "0")
            in {"1", "true", "yes", "on", "True"}
            and hasattr(self.executor, "mtp1_greedy_burst_table_jit")
        )
        use_packed_prefix_table_verifier = (
            use_packed_prefix_verifier
            and bool(getattr(self, "resident_decode_metadata", False))
            and getattr(self, "_hybrid_state_table", None) is not None
            and self._hybrid_state_table.conv_state is not None
            and self._hybrid_state_table.recurrent_state is not None
            and hasattr(self.executor, "mtp_k_packed_prefix_table_greedy_step_jit")
        )
        use_packed_prefix_row_verifier = (
            use_packed_prefix_verifier
            and hasattr(self.executor, "mtp_k_packed_prefix_greedy_step_jit")
        )
        if use_packed_prefix_verifier and not use_packed_prefix_row_verifier:
            raise RuntimeError(
                "mtp_verifier_impl=packed_prefix requires the row-state packed "
                "verifier; no sequential fallback is allowed"
            )
        t_profile = _mark("verifier_route_setup", t_profile)
        hybrid_state = None
        hybrid_slot_ids = None
        if (
            use_one_pass_table_k1
            or use_fast_table_verifier
            or use_burst_table_verifier
            or use_packed_prefix_table_verifier
        ):
            hybrid_slot_ids = self._batch_hybrid_slot_ids(decode_batch)
            _ready(hybrid_slot_ids)
            t_profile = _mark("batch_hybrid_slot_ids", t_profile)
        else:
            hybrid_state = self._batch_hybrid_state(decode_batch)
            _ready(hybrid_state)
            t_profile = _mark("batch_hybrid_state", t_profile)
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
            if use_one_pass_table_k1:
                verifier_mode = "mtp1_one_pass_prefix_table"
            elif use_one_pass_k1:
                verifier_mode = "mtp1_one_pass_prefix"
            elif use_commit_select:
                verifier_mode = "mtp1_commit_select"
            elif use_fast_all_accept:
                verifier_mode = "mtp1_two_decode_fast"
            elif draft_len == 1 and use_prefix_two_decode:
                verifier_mode = "mtp1_two_decode"
            elif use_mtp2_commit_select:
                verifier_mode = "mtp2_commit_select"
            elif use_k_burst:
                verifier_mode = "mtp_k_burst"
            elif use_packed_prefix_table_verifier:
                verifier_mode = "mtp_k_packed_prefix_table"
            elif use_packed_prefix_row_verifier:
                verifier_mode = "mtp_k_packed_prefix"
            elif use_packed_prefix_verifier:
                verifier_mode = "mtp_k_packed_prefix_invalid"
            elif draft_len == 1 and not use_generic_k_verifier:
                verifier_mode = "fallback_k1_no_verifier"
            elif (
                partial_physical_batch
                and not use_compact_verifier
                and draft_len > 1
                and use_generic_k_verifier
            ):
                verifier_mode = "mtp_k_decode_masked_partial"
            elif partial_physical_batch and not use_compact_verifier:
                verifier_mode = "fallback_k_gt1_partial_physical"
            else:
                verifier_mode = "mtp_k_decode"
            print(
                "[MTP_RUN] verifier "
                f"mode={verifier_mode} draft_len={draft_len} "
                f"burst_groups={packed_prefix_burst_groups if use_packed_prefix_verifier else (table_burst_groups if use_one_pass_table_k1 else mtp_burst_groups)} "
                f"rows={rows} physical_batch={physical_batch_size} "
                f"verifier_batch={verifier_physical_batch_size} "
                f"partial_physical={partial_physical_batch} "
                f"compact={use_compact_verifier}",
                flush=True,
            )
        parity_output = None
        k_parity_output = None
        layer_parity_output = None
        layerwise_drift_output = None
        if use_one_pass_table_k1:
            if table_burst_groups > 1:
                output = self.executor.mtp1_two_decode_greedy_table_burst_step_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=self._hybrid_state_table,
                    hybrid_slot_ids=hybrid_slot_ids,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                    burst_groups=table_burst_groups,
                )
                route_label = "executor_mtp1_one_pass_prefix_table_burst"
            else:
                output = self.executor.mtp1_two_decode_greedy_table_step_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=self._hybrid_state_table,
                    hybrid_slot_ids=hybrid_slot_ids,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                route_label = "executor_mtp1_one_pass_prefix_table"
            _ready(output)
            t_profile = _mark(route_label, t_profile)
        elif use_one_pass_k1:
            if layerwise_drift_debug_one_pass:
                layerwise_drift_output = self.executor.mtp1_layerwise_drift_debug_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=_verifier_draft_tokens_device(),
                )
                _ready(layerwise_drift_output)
                t_profile = _mark("executor_mtp1_layerwise_drift_debug", t_profile)
            if layer_parity_debug_one_pass:
                layer_parity_output = self.executor.mtp1_layer_parity_debug_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                _ready(layer_parity_output)
                t_profile = _mark("executor_mtp1_layer_parity_debug", t_profile)
            if parity_debug_one_pass:
                # Run the commit-select parity reference before the one-pass
                # verifier call. Both executors donate KV cache buffers, so the
                # parity reference must consume an explicit copy and leave the
                # live cache available for the production one-pass call.
                parity_cache_storage = KVCacheStorage(
                    self.cache_storage.k_cache.copy(),
                    self.cache_storage.v_cache.copy(),
                )
                parity_output = self.executor.mtp1_commit_select_greedy_step_jit(
                    decode_batch,
                    cache_storage=parity_cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                _ready(parity_output)
                t_profile = _mark("executor_mtp1_parity_commit_select", t_profile)
            output = self.executor.mtp1_two_decode_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=_verifier_draft_tokens_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_one_pass_prefix", t_profile)
        elif use_commit_select:
            output = self.executor.mtp1_commit_select_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=_verifier_draft_tokens_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_commit_select", t_profile)
        elif use_fast_all_accept:
            if use_greedy_burst_table_verifier:
                output = self.executor.mtp1_greedy_burst_table_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=self._hybrid_state_table,
                    hybrid_slot_ids=hybrid_slot_ids,
                    draft_token=_verifier_draft_tokens_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                route_label = "executor_mtp1_greedy_burst_table"
            elif use_burst_table_verifier:
                output = self.executor.mtp1_burst_verify_table_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=self._hybrid_state_table,
                    hybrid_slot_ids=hybrid_slot_ids,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                route_label = "executor_mtp1_burst_verify_table"
            elif use_fast_table_verifier:
                output = self.executor.mtp1_two_decode_greedy_fast_table_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state_table=self._hybrid_state_table,
                    hybrid_slot_ids=hybrid_slot_ids,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                route_label = "executor_mtp1_two_decode_fast_table"
            else:
                output = self.executor.mtp1_two_decode_greedy_fast_step_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_token=_verifier_draft_tokens_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                )
                route_label = "executor_mtp1_two_decode_fast"
            _ready(output)
            t_profile = _mark(route_label, t_profile)
        elif draft_len == 1 and use_prefix_two_decode:
            output = self.executor.mtp1_two_decode_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_token=_verifier_draft_tokens_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
            )
            _ready(output)
            t_profile = _mark("executor_mtp1_two_decode", t_profile)
        elif draft_len == 1 and not use_generic_k_verifier:
            return None
        elif use_mtp2_commit_select:
            output = self.executor.mtp2_commit_select_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=_verifier_draft_token_chains_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
            )
            _ready(output)
            t_profile = _mark("executor_mtp2_commit_select", t_profile)
        elif use_packed_prefix_table_verifier:
            if (
                draft_len > 1
                and os.environ.get("NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_DEBUG", "0")
                in {"1", "true", "yes", "on", "True"}
            ):
                raise RuntimeError(
                    "packed-prefix table verifier does not support layerwise "
                    "drift debug; run a diagnostic verifier route explicitly"
                )
            output = self.executor.mtp_k_packed_prefix_table_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                mtp_cache_storage=self.mtp_cache_storage,
                hybrid_state_table=self._hybrid_state_table,
                hybrid_slot_ids=hybrid_slot_ids,
                draft_tokens=_verifier_draft_token_chains_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                burst_groups=packed_prefix_burst_groups,
                emit_bonus=emit_bonus,
                resident_seq_lens=self._resident_seq_lens,
            )
            _ready(output)
            t_profile = _mark("executor_mtp_k_packed_prefix_table", t_profile)
        elif use_packed_prefix_verifier:
            if (
                draft_len > 1
                and os.environ.get("NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_DEBUG", "0")
                in {"1", "true", "yes", "on", "True"}
            ):
                raise RuntimeError(
                    "packed-prefix verifier does not support layerwise drift "
                    "debug; run a diagnostic verifier route explicitly"
                )
            output = self.executor.mtp_k_packed_prefix_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=_verifier_draft_token_chains_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                burst_groups=packed_prefix_burst_groups,
            )
            _ready(output)
            t_profile = _mark("executor_mtp_k_packed_prefix", t_profile)
        elif use_k_burst:
            output = self.executor.mtp_k_burst_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=_verifier_draft_token_chains_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                burst_groups=mtp_burst_groups,
            )
            _ready(output)
            t_profile = _mark("executor_mtp_k_burst", t_profile)
        else:
            if (
                partial_physical_batch
                and not use_compact_verifier
                and not (strict_k_verifier and draft_len > 1 and use_generic_k_verifier)
            ):
                return _none_or_strict_error(
                    "partial physical batch requires compact grouped verification"
                )
            if (
                draft_len > 1
                and use_generic_k_verifier
                and os.environ.get("NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_DEBUG", "0")
                in {"1", "true", "yes", "on", "True"}
                and hasattr(self.executor, "mtp_k_layerwise_drift_debug_jit")
            ):
                layerwise_drift_output = self.executor.mtp_k_layerwise_drift_debug_jit(
                    decode_batch,
                    cache_storage=self.cache_storage,
                    hybrid_state=hybrid_state,
                    draft_tokens=_verifier_draft_token_chains_device(),
                )
                _ready(layerwise_drift_output)
                t_profile = _mark("executor_mtp_k_layerwise_drift_debug", t_profile)
            if (
                draft_len == 2
                and use_generic_k_verifier
                and os.environ.get("NANO_VLLM_JAX_MTP_K_PARITY_DEBUG", "0")
                in {"1", "true", "yes", "on", "True"}
                and hasattr(self.executor, "mtp2_commit_select_greedy_step_jit")
            ):
                parity_cache_storage = KVCacheStorage(
                    self.cache_storage.k_cache.copy(),
                    self.cache_storage.v_cache.copy(),
                )
                k_parity_output = self.executor.mtp2_commit_select_greedy_step_jit(
                    decode_batch,
                    cache_storage=parity_cache_storage,
                    hybrid_state=hybrid_state,
                    draft_tokens=_verifier_draft_token_chains_device(),
                    next_mtp_position=_next_mtp_positions_device(),
                    mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                    mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                    mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
                )
                _ready(k_parity_output)
                t_profile = _mark("executor_mtp_k_parity_commit_select", t_profile)
            output = self.executor.mtp_k_decode_greedy_step_jit(
                decode_batch,
                cache_storage=self.cache_storage,
                hybrid_state=hybrid_state,
                draft_tokens=_verifier_draft_token_chains_device(),
                next_mtp_position=_next_mtp_positions_device(),
                mtp_hidden_final_normed=getattr(self, "mtp_hidden_source", "final_normed") == "final_normed",
                mtp_chain_return_normed=getattr(self, "mtp_chain_hidden_source", "raw") == "final_normed",
                mtp_chain_mode=getattr(self, "mtp_chain_mode", "recursive"),
            )
            _ready(output)
            t_profile = _mark("executor_mtp_k_decode", t_profile)

        device_burst_commit = (
            use_greedy_burst_table_verifier
            and draft_len == 1
            and os.environ.get("NANO_VLLM_JAX_MTP_DEVICE_BURST_COMMIT", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        if device_burst_commit:
            self.cache_storage = output.cache_storage
            committed_batch = decode_batch
            if getattr(output, "committed_seq_lens", None) is not None:
                committed_batch = self._with_committed_seq_lens(
                    decode_batch,
                    output.committed_seq_lens,
                )
            self._record_resident_committed_seq_lens(committed_batch)
            self._hybrid_state_table = output.hybrid_state
            self._mark_hybrid_slots_written(list(decode_batch.hybrid_slot_ids_host or ()))
            self._record_kv_snapshot(committed_batch, output.hybrid_state)

            token_rows = jnp.stack(
                [
                    output.target_token.astype(jnp.int32),
                    output.bonus_token.astype(jnp.int32),
                ],
                axis=1,
            )
            outputs: dict[int, List[int] | int] = {}
            stats = self._speculative_stats()
            seed_after_bonus = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
            ) in {"1", "true", "yes", "on", "True"}
            for local_row, row in enumerate(rows):
                verifier_idx = verifier_index_for_local[local_row]
                outputs[row] = [
                    DeviceTokenRef(tokens=token_rows, row=verifier_idx * 2),
                    DeviceTokenRef(tokens=token_rows, row=verifier_idx * 2 + 1),
                ]
                stats["bonus_tokens"] += 1
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                if (
                    seed_after_bonus
                    and self.mtp1_enabled
                    and seq.temperature == 0
                    and seq.num_completion_tokens + 2 < seq.max_tokens
                    and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                ):
                    self._mtp1_drafts[seq.seq_id] = DeviceTokenRef(
                        tokens=output.next_draft_token.astype(jnp.int32),
                        row=verifier_idx,
                    )
                    stats["drafts_proposed"] += 1
            t_profile = _mark("device_burst_commit", t_profile)
            return outputs

        device_assume_accept_commit = (
            use_burst_table_verifier
            and draft_len == 1
            and os.environ.get("NANO_VLLM_JAX_MTP_BURST_ASSUME_ALL_ACCEPT", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        if device_assume_accept_commit:
            self.cache_storage = output.cache_storage
            committed_batch = decode_batch
            if getattr(output, "committed_seq_lens", None) is not None:
                committed_batch = self._with_committed_seq_lens(
                    decode_batch,
                    output.committed_seq_lens,
                )
            self._record_resident_committed_seq_lens(committed_batch)
            self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
            self._record_kv_snapshot(committed_batch, output.hybrid_state)

            token_rows = jnp.stack(
                [
                    _verifier_draft_tokens_device().astype(jnp.int32),
                    output.bonus_token.astype(jnp.int32),
                ],
                axis=1,
            )
            outputs: dict[int, List[int] | int] = {}
            stats = self._speculative_stats()
            seed_after_bonus = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
            ) in {"1", "true", "yes", "on", "True"}
            for local_row, row in enumerate(rows):
                verifier_idx = verifier_index_for_local[local_row]
                outputs[row] = [
                    DeviceTokenRef(tokens=token_rows, row=verifier_idx * 2),
                    DeviceTokenRef(tokens=token_rows, row=verifier_idx * 2 + 1),
                ]
                stats["drafts_accepted"] += 1
                stats["bonus_tokens"] += 1
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                if (
                    seed_after_bonus
                    and self.mtp1_enabled
                    and seq.temperature == 0
                    and seq.num_completion_tokens + 2 < seq.max_tokens
                    and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                ):
                    self._mtp1_drafts[seq.seq_id] = DeviceTokenRef(
                        tokens=output.next_draft_token.astype(jnp.int32),
                        row=verifier_idx,
                    )
                    stats["drafts_proposed"] += 1
            t_profile = _mark("device_assume_accept_commit", t_profile)
            return outputs

        assume_all_accept_k = (
            draft_len > 1
            and not use_k_burst
            and not use_mtp2_commit_select
            and not use_commit_select
            and os.environ.get("NANO_VLLM_JAX_MTP_ASSUME_ALL_ACCEPT_K", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        if assume_all_accept_k:
            self.cache_storage = output.cache_storage
            committed_batch = decode_batch
            if getattr(output, "committed_seq_lens", None) is not None:
                committed_batch = self._with_committed_seq_lens(
                    decode_batch,
                    output.committed_seq_lens,
                )
            self._record_resident_committed_seq_lens(committed_batch)
            self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
            self._record_kv_snapshot(committed_batch, output.hybrid_state)

            bonus_token_refs = output.bonus_token.astype(jnp.int32)
            next_draft_token_refs = output.next_draft_token.astype(jnp.int32)
            seed_after_bonus = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
            ) in {"1", "true", "yes", "on", "True"}
            disable_bonus = (
                os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_BONUS", "0")
                in {"1", "true", "yes", "on", "True"}
            )
            max_seeded_chain = int(os.environ.get("NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN", "0") or "0")

            outputs: dict[int, List[int] | int] = {}
            stats = self._speculative_stats()
            self._record_draft_position_acceptance([[True for _ in range(draft_len)] for _ in rows])
            for local_row, row in enumerate(rows):
                seq = seqs[row]
                verifier_idx = verifier_index_for_local[local_row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                emitted_len = draft_len
                row_outputs: list[object] = list(draft_token_chains[local_row])
                stats["drafts_accepted"] += draft_len
                if not disable_bonus:
                    row_outputs.append(DeviceTokenRef(tokens=bonus_token_refs, row=verifier_idx))
                    stats["bonus_tokens"] += 1
                    emitted_len += 1
                outputs[row] = row_outputs

                can_seed_next_chain = (
                    self.mtp1_enabled
                    and seq.temperature == 0
                    and seq.num_completion_tokens + emitted_len < seq.max_tokens
                    and (seed_after_bonus or disable_bonus)
                    and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                    and (
                        max_seeded_chain <= 0
                        or self._mtp1_seeded_chain.get(seq.seq_id, 0) < max_seeded_chain
                    )
                )
                if can_seed_next_chain:
                    next_chain = [
                        DeviceTokenRef(
                            tokens=next_draft_token_refs,
                            row=verifier_idx * draft_len + pos,
                        )
                        for pos in range(draft_len)
                    ]
                    self._mtp1_drafts[seq.seq_id] = next_chain if len(next_chain) > 1 else next_chain[0]
                    self._mtp1_seeded_chain[seq.seq_id] = (
                        self._mtp1_seeded_chain.get(seq.seq_id, 0) + draft_len
                    )
                    stats["drafts_proposed"] += len(next_chain)
                else:
                    self._mtp1_seeded_chain.pop(seq.seq_id, None)

            self._record_mtp_output_token_carry(batch, seqs, outputs)
            t_profile = _mark("assume_all_accept_k_commit", t_profile)
            return outputs

        use_resident_verifier_commit = (
            use_k_burst
            or (
                getattr(output, "emitted_counts", None) is not None
                and (draft_len == 1 or use_generic_k_verifier)
            )
        )
        if use_resident_verifier_commit:
            if getattr(output, "emitted_counts", None) is None:
                raise RuntimeError(
                    "MTP resident verifier commit requires emitted_counts; "
                    "rowwise host repair is disabled for this path"
                )
            resident_burst_groups = int(
                getattr(output, "burst_groups", None)
                or (mtp_burst_groups if use_k_burst else 1)
            )

            self.cache_storage = output.cache_storage
            if output.mtp_cache_storage is not None:
                self.mtp_cache_storage = output.mtp_cache_storage
            committed_batch = self._with_committed_seq_lens(
                decode_batch,
                output.committed_seq_lens,
            )
            if k_parity_output is not None:
                (
                    k_target_host,
                    k_bonus_host,
                    k_next_host,
                    k_accepted_host,
                    parity_target_host,
                    parity_bonus_host,
                    parity_next_host,
                    parity_accepted_host,
                ) = jax.device_get(
                    (
                        output.target_token,
                        output.bonus_token,
                        output.next_draft_token,
                        output.accepted,
                        k_parity_output.target_token,
                        k_parity_output.bonus_token,
                        k_parity_output.next_draft_token,
                        k_parity_output.accepted,
                    )
                )

                def _state_max_abs(left, right) -> float:
                    if left is None or right is None:
                        return 0.0
                    return float(
                        jnp.max(
                            jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
                        ).item()
                    )

                def _state_layer_max_abs(left, right) -> list[float]:
                    if left is None or right is None:
                        return []
                    diff = jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
                    if diff.ndim < 2:
                        return [float(jnp.max(diff).item())]
                    if diff.ndim >= 5:
                        # Hybrid table: [resident, rows, linear_layers, ...].
                        reduce_axes = tuple(axis for axis in range(diff.ndim) if axis != 2)
                        values = jnp.max(diff, axis=reduce_axes)
                    elif diff.ndim >= 4:
                        # Layerwise resident state: [rows, linear_layers, ...].
                        reduce_axes = tuple(axis for axis in range(diff.ndim) if axis != 1)
                        values = jnp.max(diff, axis=reduce_axes)
                    else:
                        values = jnp.asarray([jnp.max(diff)])
                    return [float(x) for x in jax.device_get(values).reshape(-1)]

                def _slot_max_abs(left, right, slots) -> float:
                    leading_shape = left.shape[:-4] if left.ndim == 5 else left.shape[:-3]
                    flat_left = left.reshape(leading_shape + (-1,) + left.shape[-2:])
                    flat_right = right.reshape(leading_shape + (-1,) + right.shape[-2:])
                    left_values = flat_left[..., slots, :, :].astype(jnp.float32)
                    right_values = flat_right[..., slots, :, :].astype(jnp.float32)
                    return float(jnp.max(jnp.abs(left_values - right_values)).item())

                def _slot_layer_max_abs(left, right, slots) -> list[float]:
                    leading_shape = left.shape[:-4] if left.ndim == 5 else left.shape[:-3]
                    flat_left = left.reshape(leading_shape + (-1,) + left.shape[-2:])
                    flat_right = right.reshape(leading_shape + (-1,) + right.shape[-2:])
                    diff = jnp.abs(
                        flat_left[..., slots, :, :].astype(jnp.float32)
                        - flat_right[..., slots, :, :].astype(jnp.float32)
                    )
                    if len(leading_shape) == 0:
                        values = jnp.asarray([jnp.max(diff)])
                    else:
                        reduce_axes = tuple(range(1, diff.ndim))
                        values = jnp.max(diff, axis=reduce_axes)
                    return [float(x) for x in jax.device_get(values).reshape(-1)]

                slot_columns = [
                    compute_slot_mapping(
                        positions=decode_batch.positions + offset,
                        block_table=decode_batch.block_tables,
                        block_size=self.config.block_size,
                        is_prefill=False,
                    )[:, 0]
                    for offset in range(draft_len + 1)
                ]
                parity_slots = jnp.stack(slot_columns, axis=1).reshape(-1)
                k_slot_diff = _slot_max_abs(
                    output.cache_storage.k_cache,
                    k_parity_output.cache_storage.k_cache,
                    parity_slots,
                )
                v_slot_diff = _slot_max_abs(
                    output.cache_storage.v_cache,
                    k_parity_output.cache_storage.v_cache,
                    parity_slots,
                )
                conv_diff = _state_max_abs(
                    output.hybrid_state.conv_state,
                    k_parity_output.hybrid_state.conv_state,
                )
                recurrent_diff = _state_max_abs(
                    output.hybrid_state.recurrent_state,
                    k_parity_output.hybrid_state.recurrent_state,
                )
                k_slot_layer_diff = _slot_layer_max_abs(
                    output.cache_storage.k_cache,
                    k_parity_output.cache_storage.k_cache,
                    parity_slots,
                )
                v_slot_layer_diff = _slot_layer_max_abs(
                    output.cache_storage.v_cache,
                    k_parity_output.cache_storage.v_cache,
                    parity_slots,
                )
                conv_layer_diff = _state_layer_max_abs(
                    output.hybrid_state.conv_state,
                    k_parity_output.hybrid_state.conv_state,
                )
                recurrent_layer_diff = _state_layer_max_abs(
                    output.hybrid_state.recurrent_state,
                    k_parity_output.hybrid_state.recurrent_state,
                )
                if profile_mtp:
                    _draft_debug, debug_events = self._mtp1_debug_state()
                    debug_events.append(
                        {
                            "kind": "mtp_k_parity",
                            "target_k": k_target_host.tolist(),
                            "target_commit": parity_target_host.tolist(),
                            "bonus_k": k_bonus_host.tolist(),
                            "bonus_commit": parity_bonus_host.tolist(),
                            "accepted_k": k_accepted_host.tolist(),
                            "accepted_commit": parity_accepted_host.tolist(),
                            "next_k": k_next_host.tolist(),
                            "next_commit": parity_next_host.tolist(),
                            "k_slot_max_abs": k_slot_diff,
                            "v_slot_max_abs": v_slot_diff,
                            "conv_max_abs": conv_diff,
                            "recurrent_max_abs": recurrent_diff,
                            "k_slot_layer_max_abs": k_slot_layer_diff,
                            "v_slot_layer_max_abs": v_slot_layer_diff,
                            "conv_layer_max_abs": conv_layer_diff,
                            "recurrent_layer_max_abs": recurrent_layer_diff,
                        }
                    )
                print(
                    "[MTP_K_PARITY] k_decode_vs_commit_select "
                    f"target_k={k_target_host.tolist()} target_commit={parity_target_host.tolist()} "
                    f"bonus_k={k_bonus_host.tolist()} bonus_commit={parity_bonus_host.tolist()} "
                    f"accepted_k={k_accepted_host.tolist()} accepted_commit={parity_accepted_host.tolist()} "
                    f"next_k={k_next_host.tolist()} next_commit={parity_next_host.tolist()} "
                    f"k_slot_max_abs={k_slot_diff:.6g} "
                    f"v_slot_max_abs={v_slot_diff:.6g} "
                    f"conv_max_abs={conv_diff:.6g} "
                    f"recurrent_max_abs={recurrent_diff:.6g} "
                    f"k_slot_layer_max_abs={k_slot_layer_diff} "
                    f"v_slot_layer_max_abs={v_slot_layer_diff} "
                    f"conv_layer_max_abs={conv_layer_diff} "
                    f"recurrent_layer_max_abs={recurrent_layer_diff}",
                    flush=True,
                )
            if layerwise_drift_output is not None:
                threshold = float(os.environ.get("NANO_VLLM_JAX_MTP_LAYERWISE_DRIFT_THRESHOLD", "0.1"))
                hidden_vals = [float(x) for x in layerwise_drift_output.hidden_max_abs.tolist()]
                k_vals = [float(x) for x in layerwise_drift_output.k_slot_max_abs.tolist()]
                v_vals = [float(x) for x in layerwise_drift_output.v_slot_max_abs.tolist()]
                conv_vals = [
                    float(x)
                    for x in layerwise_drift_output.conv_state_max_abs.tolist()
                ]
                rec_vals = [
                    float(x)
                    for x in layerwise_drift_output.recurrent_state_max_abs.tolist()
                ]
                pre_k_vals = [
                    float(x)
                    for x in layerwise_drift_output.k_prewrite_max_abs.tolist()
                ]
                pre_v_vals = [
                    float(x)
                    for x in layerwise_drift_output.v_prewrite_max_abs.tolist()
                ]
                stage_vals = [
                    [float(v) for v in row]
                    for row in layerwise_drift_output.block_stage_max_abs.tolist()
                ]
                gdn_input_vals = []
                if getattr(layerwise_drift_output, "gdn_input_max_abs", None) is not None:
                    gdn_input_vals = [
                        float(v)
                        for v in layerwise_drift_output.gdn_input_max_abs.tolist()
                    ]
                gdn_input_names = [
                    "mixed_qkv",
                    "z",
                    "a",
                    "b",
                    "conv_out",
                    "query",
                    "key",
                    "value",
                    "gate",
                    "beta",
                ]
                stage_names = [
                    "entry",
                    "in_norm",
                    "attn",
                    "attn_resid",
                    "ffn_norm",
                    "mlp",
                    "out",
                ]
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
                first_layer = "none" if first_idx is None else str(first_idx)
                first_type = "none" if first_idx is None else self.config.layer_types[first_idx]
                if profile_mtp:
                    _draft_debug, debug_events = self._mtp1_debug_state()
                    debug_events.append(
                        {
                            "kind": "mtp_k_layerwise_drift",
                            "threshold": threshold,
                            "first_layer": first_layer,
                            "first_type": first_type,
                            "layers": layer_rows,
                            "gdn0_inputs": {
                                name: gdn_input_vals[idx]
                                for idx, name in enumerate(gdn_input_names)
                            }
                            if gdn_input_vals
                            else {},
                        }
                    )
                print(
                    "[MTP_LAYERWISE_DRIFT] grouped_k_current_vs_seq "
                    f"threshold={threshold:.6g} "
                    f"first_layer={first_layer} first_type={first_type} "
                    f"layers={';'.join(layer_rows)} "
                    f"gdn0_inputs={','.join(f'{name}={gdn_input_vals[idx]:.6g}' for idx, name in enumerate(gdn_input_names)) if gdn_input_vals else 'n/a'}",
                    flush=True,
                )
            if getattr(output, "resident_seq_lens", None) is not None:
                self._resident_seq_lens = output.resident_seq_lens
            else:
                self._record_resident_committed_seq_lens(committed_batch)
            if getattr(output, "hybrid_state_is_table", False):
                self._hybrid_state_table = output.hybrid_state
                self._mark_hybrid_slots_written(list(decode_batch.hybrid_slot_ids_host or ()))
                t_profile = _mark("install_resident_burst_hybrid_table", t_profile)
            else:
                self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
                t_profile = _mark("store_resident_burst_hybrid_state", t_profile)
            self._record_kv_snapshot(committed_batch, output.hybrid_state)
            t_profile = _mark("record_resident_burst_kv_snapshot", t_profile)

            emitted_width = resident_burst_groups * (draft_len + 1)
            emitted_tokens = output.emitted_tokens.astype(jnp.int32).reshape(
                (verifier_physical_batch_size, emitted_width)
            )
            next_draft_tokens = output.next_draft_token.astype(jnp.int32)

            compact_burst_output = (
                draft_len == 1
                and getattr(output, "emitted_totals", None) is not None
                and getattr(output, "accepted_totals", None) is not None
                and getattr(output, "accepted_bitmask", None) is not None
            )
            if compact_burst_output:
                compact_summary = getattr(output, "compact_summary", None)
                if compact_summary is not None:
                    compact_summary_host = jax.device_get(compact_summary)
                    emitted_totals_host = compact_summary_host[:, 0]
                    accepted_totals_host = compact_summary_host[:, 1]
                    rejected_totals_host = compact_summary_host[:, 2]
                    bonus_totals_host = compact_summary_host[:, 3]
                    accepted_bitmask_host = compact_summary_host[:, 4]
                else:
                    (
                        emitted_totals_host,
                        accepted_totals_host,
                        rejected_totals_host,
                        bonus_totals_host,
                        accepted_bitmask_host,
                    ) = jax.device_get(
                        (
                            output.emitted_totals,
                            output.accepted_totals,
                            output.rejected_totals,
                            output.bonus_totals,
                            output.accepted_bitmask,
                        )
                    )
                t_profile = _mark("host_burst_summary_transfer", t_profile)
                accepted_matrix = []
                for local_row, _row in enumerate(rows):
                    verifier_idx = verifier_index_for_local[local_row]
                    bitmask = int(accepted_bitmask_host[verifier_idx])
                    accepted_matrix.extend(
                        [bool(bitmask & (1 << group_idx))]
                        for group_idx in range(resident_burst_groups)
                    )
            else:
                emitted_counts_host = jax.device_get(output.emitted_counts).reshape(
                    (verifier_physical_batch_size, resident_burst_groups)
                )
                if getattr(output, "accepted_counts", None) is not None:
                    accepted_counts_host = jax.device_get(output.accepted_counts).reshape(
                        (verifier_physical_batch_size, resident_burst_groups)
                    )
                else:
                    accepted_counts_host = emitted_counts_host - 1
                t_profile = _mark("host_burst_output_count_transfer", t_profile)
                accepted_matrix = [
                    [pos < int(accepted_counts_host[verifier_index_for_local[local_row], group_idx]) for pos in range(draft_len)]
                    for local_row, _row in enumerate(rows)
                    for group_idx in range(resident_burst_groups)
                ]
            self._record_draft_position_acceptance(accepted_matrix)
            future_draft_top_ids_dbg = None
            future_draft_top_values_dbg = None
            if getattr(output, "debug_payload", None) is not None:
                (
                    draft_top_ids_dbg,
                    draft_top_values_dbg,
                    verifier_top_ids_dbg,
                    verifier_top_values_dbg,
                    draft_tokens_dbg,
                    target_tokens_dbg,
                ) = jax.device_get(output.debug_payload)
                future_draft_top_ids_dbg = draft_top_ids_dbg
                future_draft_top_values_dbg = draft_top_values_dbg
                seed_debug_by_seq, debug_events = self._mtp1_debug_state()
                for local_row, row in enumerate(rows):
                    verifier_idx = verifier_index_for_local[local_row]
                    seq_id = int(seqs[row].seq_id)
                    seed_debug = seed_debug_by_seq.get(seq_id, {})
                    seed_chain_top = seed_debug.get("mtp_chain_top", [])
                    for group_idx in range(resident_burst_groups):
                        for draft_pos in range(draft_len):
                            mtp_top_ids = []
                            verifier_top_ids = [
                                int(value)
                                for value in verifier_top_ids_dbg[
                                    verifier_idx, group_idx, draft_pos
                                ].tolist()
                            ]
                            target_token_dbg = int(
                                target_tokens_dbg[verifier_idx, group_idx, draft_pos]
                            )
                            draft_token_dbg = int(
                                draft_tokens_dbg[verifier_idx, group_idx, draft_pos]
                            )
                            mtp_top_source = "unavailable"
                            mtp_top_values = []
                            if group_idx == 0 and draft_pos < len(seed_chain_top):
                                seed_top = seed_chain_top[draft_pos]
                                mtp_top_ids = [int(value) for value in seed_top.get("ids", [])]
                                mtp_top_values = [
                                    float(value)
                                    for value in seed_top.get("values", [])
                                ]
                                mtp_top_source = str(
                                    seed_debug.get("top_source", "seed_time")
                                )
                            debug_events.append(
                                {
                                    "kind": "mtp_k_logit_debug",
                                    "seq_id": seq_id,
                                    "row": int(row),
                                    "verifier_row": int(verifier_idx),
                                    "burst_group": int(group_idx),
                                    "draft_position": int(draft_pos),
                                    "draft_token": draft_token_dbg,
                                    "target_token": target_token_dbg,
                                    "accepted": bool(
                                        accepted_matrix[
                                            local_row * resident_burst_groups + group_idx
                                        ][draft_pos]
                                    ),
                                    "target_in_mtp_top5": target_token_dbg in mtp_top_ids,
                                    "mtp_top_source": mtp_top_source,
                                    "mtp_top": {
                                        "ids": mtp_top_ids,
                                        "values": mtp_top_values,
                                    },
                                    "verifier_top": {
                                        "ids": verifier_top_ids,
                                        "values": [
                                            float(value)
                                            for value in verifier_top_values_dbg[
                                                verifier_idx, group_idx, draft_pos
                                            ].tolist()
                                        ],
                                    },
                                }
                            )

            outputs: dict[int, List[int] | int] = {}
            stats = self._speculative_stats()
            max_seeded_chain = int(os.environ.get("NANO_VLLM_JAX_MTP_MAX_SEEDED_CHAIN", "0") or "0")
            row_to_committed_len_host: dict[int, int] = {}
            seq_lens_host = decode_batch.seq_lens_host
            for local_row, row in enumerate(rows):
                row = rows[local_row]
                seq = seqs[row]
                verifier_idx = verifier_index_for_local[local_row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                row_outputs: list[object] = []
                if compact_burst_output:
                    emitted_total = max(
                        0,
                        min(
                            emitted_width,
                            int(emitted_totals_host[verifier_idx]),
                        ),
                    )
                    accepted_total = max(
                        0,
                        min(
                            resident_burst_groups * draft_len,
                            int(accepted_totals_host[verifier_idx]),
                        ),
                    )
                    rejected_total = max(
                        0,
                        min(
                            resident_burst_groups,
                            int(rejected_totals_host[verifier_idx]),
                        ),
                    )
                    bonus_total = max(
                        0,
                        min(
                            resident_burst_groups,
                            int(bonus_totals_host[verifier_idx]),
                        ),
                    )
                    row_outputs.extend(
                        DeviceTokenRef(
                            tokens=emitted_tokens,
                            row=verifier_idx * emitted_width + offset,
                        )
                        for offset in range(emitted_total)
                    )
                else:
                    emitted_total = 0
                    accepted_total = 0
                    rejected_total = 0
                    bonus_total = 0
                    remaining_output_budget = max(
                        0,
                        int(seq.max_tokens - seq.num_completion_tokens),
                    )
                    for group_idx in range(resident_burst_groups):
                        emitted_count = int(emitted_counts_host[verifier_idx, group_idx])
                        accepted_count = int(accepted_counts_host[verifier_idx, group_idx])
                        emitted_count = max(0, min(draft_len + 1, emitted_count))
                        accepted_count = max(0, min(draft_len, accepted_count))
                        allowed_count = max(
                            0,
                            min(
                                emitted_count,
                                remaining_output_budget - emitted_total,
                            ),
                        )
                        group_offset = group_idx * (draft_len + 1)
                        row_outputs.extend(
                            DeviceTokenRef(
                                tokens=emitted_tokens,
                                row=verifier_idx * emitted_width + group_offset + offset,
                            )
                            for offset in range(allowed_count)
                        )
                        emitted_total += allowed_count
                        accepted_in_group = min(accepted_count, allowed_count)
                        accepted_total += accepted_in_group
                        if allowed_count <= accepted_count:
                            continue
                        if accepted_count < draft_len:
                            rejected_total += 1
                        else:
                            bonus_total += 1
                outputs[row] = row_outputs
                if seq_lens_host is not None and verifier_idx < len(seq_lens_host):
                    row_to_committed_len_host[verifier_idx] = (
                        int(seq_lens_host[verifier_idx]) + emitted_total
                    )
                if profile_mtp:
                    _draft_debug, debug_events = self._mtp1_debug_state()
                    debug_events.append(
                        {
                            "kind": "mtp_resident_commit",
                            "seq_id": int(seq.seq_id),
                            "row": int(row),
                            "verifier_row": int(verifier_idx),
                            "emitted_count": int(emitted_total),
                            "accepted_count": int(accepted_total),
                            "bonus_count": int(bonus_total),
                            "emitted_ref_rows": [
                                int(token.row)
                                for token in row_outputs
                                if isinstance(token, DeviceTokenRef)
                            ],
                            "committed_seq_len": int(
                                row_to_committed_len_host.get(verifier_idx, -1)
                            ),
                        }
                    )
                stats["drafts_proposed"] += max(0, (resident_burst_groups - 1) * draft_len)
                stats["drafts_accepted"] += accepted_total
                stats["drafts_rejected"] += rejected_total
                stats["bonus_tokens"] += bonus_total
                if (
                    self.mtp1_enabled
                    and seq.temperature == 0
                    and seq.num_completion_tokens + emitted_total < seq.max_tokens
                    and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                    and (
                        max_seeded_chain <= 0
                        or self._mtp1_seeded_chain.get(seq.seq_id, 0) < max_seeded_chain
                    )
                ):
                    next_chain = [
                        DeviceTokenRef(
                            tokens=next_draft_tokens,
                            row=verifier_idx * draft_len + pos,
                        )
                        for pos in range(draft_len)
                    ]
                    self._mtp1_drafts[seq.seq_id] = next_chain if len(next_chain) > 1 else next_chain[0]
                    self._mtp1_seeded_chain[seq.seq_id] = (
                        self._mtp1_seeded_chain.get(seq.seq_id, 0) + resident_burst_groups * draft_len
                    )
                    stats["drafts_proposed"] += len(next_chain)
                    if (
                        profile_mtp
                        and future_draft_top_ids_dbg is not None
                        and future_draft_top_values_dbg is not None
                    ):
                        draft_debug, _ = self._mtp1_debug_state()
                        draft_debug[seq.seq_id] = {
                            "top_source": "resident_table_next_seed",
                            "draft_chain": [
                                int(
                                    future_draft_top_ids_dbg[
                                        verifier_idx,
                                        0,
                                        pos,
                                        0,
                                    ]
                                )
                                for pos in range(draft_len)
                            ],
                            "mtp_chain_top": [
                                {
                                    "ids": [
                                        int(value)
                                        for value in future_draft_top_ids_dbg[
                                            verifier_idx,
                                            0,
                                            pos,
                                        ].tolist()
                                    ],
                                    "values": [
                                        float(value)
                                        for value in future_draft_top_values_dbg[
                                            verifier_idx,
                                            0,
                                            pos,
                                        ].tolist()
                                    ],
                                }
                                for pos in range(draft_len)
                            ],
                        }
                else:
                    self._mtp1_seeded_chain.pop(seq.seq_id, None)
                    self._mtp1_debug_state()[0].pop(seq.seq_id, None)
            if not ModelRunner._device_token_carry_enabled(self):
                outputs = ModelRunner._materialize_device_token_outputs(outputs)
            self._record_resident_committed_seq_lens_host(
                committed_batch,
                row_to_committed_len_host,
            )
            if use_compact_verifier:
                compact_seqs = [seqs[row] for row in rows]
                compact_outputs = {
                    local_row: outputs[row]
                    for local_row, row in enumerate(rows)
                    if row in outputs
                }
                self._record_mtp_output_token_carry(
                    committed_batch,
                    compact_seqs,
                    compact_outputs,
                )
            else:
                self._record_mtp_output_token_carry(committed_batch, seqs, outputs)
            self._mtp_carry_recorded_this_call = True
            t_profile = _mark("resident_burst_commit", t_profile)
            return outputs

        host_payload = getattr(output, "host_payload", None)
        if host_payload is not None:
            payload_host = jax.device_get(host_payload)
            if draft_len > 1:
                target_token_host = payload_host[:, :draft_len]
                bonus_token_host = payload_host[:, draft_len]
                next_draft_token_host = payload_host[:, draft_len + 1 : draft_len + 1 + draft_len]
                accepted_host = payload_host[:, draft_len + 1 + draft_len : draft_len + 1 + draft_len + draft_len].astype(bool)
            else:
                target_token_host = payload_host[:, 0]
                bonus_token_host = payload_host[:, 1]
                next_draft_token_host = payload_host[:, 2]
                accepted_host = payload_host[:, 3].astype(bool)
        else:
            (
                accepted_host,
                target_token_host,
                bonus_token_host,
                next_draft_token_host,
            ) = jax.device_get(
                (
                    output.accepted,
                    output.target_token,
                    output.bonus_token,
                    output.next_draft_token,
                )
            )
        if draft_len == 1:
            if getattr(accepted_host, "ndim", 0) > 1:
                accepted_host = accepted_host[:, 0]
            if getattr(target_token_host, "ndim", 0) > 1:
                target_token_host = target_token_host[:, 0]
            if getattr(bonus_token_host, "ndim", 0) > 1:
                bonus_token_host = bonus_token_host[:, 0]
            if getattr(next_draft_token_host, "ndim", 0) > 1:
                next_draft_token_host = next_draft_token_host[:, 0]
        t_profile = _mark("host_result_transfer", t_profile)

        if draft_len > 1:
            output_acceptance_matrix = accepted_host.tolist()
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
            output_acceptance_for_rows = [bool(x) for x in accepted_host.tolist()]
            accepted_all = all(
                output_acceptance_for_rows[verifier_index_for_local[local_row]]
                for local_row, _row in enumerate(rows)
            )
        if use_fast_all_accept and draft_len == 1 and not accepted_all:
            # This verifier intentionally avoids materializing token-0 prefix
            # state. Its output is therefore safe to commit only when every
            # verified row accepts. On any rejection, leave the live cache and
            # hybrid state untouched and let the ordinary main-model path run
            # from the original pre-verifier state.
            stats = self._speculative_stats()
            stats["drafts_rejected"] += len(rows)
            for row in rows:
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                self._mtp1_seeded_chain.pop(seq.seq_id, None)
            return None
        if (
            enable_rowwise_repair
            and (use_fast_all_accept or use_one_pass_k1)
            and draft_len == 1
            and not accepted_all
        ):
            output_acceptance = [bool(x) for x in accepted_host.tolist()]
            accepted_flags_local = [
                output_acceptance[verifier_index_for_local[local_row]]
                for local_row, row in enumerate(rows)
            ]
            commit_rejected_directly = (
                use_one_pass_k1
                and os.environ.get("NANO_VLLM_JAX_MTP_K1_COMMIT_REJECTED", "1")
                in {"1", "true", "yes", "on", "True"}
            )

            committed_batch = self._with_committed_seq_lens(
                decode_batch,
                output.committed_seq_lens,
            )
            self.cache_storage = output.cache_storage
            self._record_resident_committed_seq_lens(committed_batch)
            self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
            self._record_kv_snapshot(committed_batch, output.hybrid_state)

            target_values = [int(value) for value in target_token_host.tolist()]
            bonus_values = [int(value) for value in bonus_token_host.tolist()]
            next_draft_values = [int(value) for value in next_draft_token_host.tolist()]
            outputs: dict[int, List[int] | int] = {}
            repair_rows: List[int] = []
            stats = self._speculative_stats()
            seed_after_bonus = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            for local_row, row in enumerate(rows):
                seq = seqs[row]
                self._mtp1_drafts.pop(seq.seq_id, None)
                if accepted_flags_local[local_row]:
                    idx = verifier_index_for_local[local_row]
                    stats["drafts_accepted"] += 1
                    stats["bonus_tokens"] += 1
                    emitted_len = 2
                    outputs[row] = [target_values[idx], bonus_values[idx]]
                    if (
                        self.mtp1_enabled
                        and seq.temperature == 0
                        and seq.num_completion_tokens + emitted_len < seq.max_tokens
                        and seed_after_bonus
                        and not getattr(self, "_mtp_adaptive_gated", lambda: False)()
                    ):
                        self._mtp1_drafts[seq.seq_id] = next_draft_values[idx]
                        stats["drafts_proposed"] += 1
                else:
                    if row not in forced_reject_rows:
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

        if (
            not accepted_all
            and not (use_one_pass_k1 or use_commit_select)
            and draft_len == 1
            and not use_generic_k_verifier
        ):
            # Correctness first: the verifier may have physically written KV /
            # hybrid state for rejected draft slots. Do not install those side
            # effects. Let the canonical main-model reuse path verify the
            # stored drafts against an ordinary decode from the original state.
            return None

        self.cache_storage = output.cache_storage
        committed_batch = decode_batch
        if getattr(output, "committed_seq_lens", None) is not None:
            committed_batch = self._with_committed_seq_lens(
                decode_batch,
                output.committed_seq_lens,
            )
        self._record_resident_committed_seq_lens(committed_batch)
        self._store_batch_hybrid_state(committed_batch, output.hybrid_state)
        t_profile = _mark("store_hybrid_state", t_profile)
        self._record_kv_snapshot(committed_batch, output.hybrid_state)
        t_profile = _mark("record_kv_snapshot", t_profile)
        target_token_values = target_token_host.tolist()
        if draft_len > 1:
            target_token_rows = [[int(token) for token in row] for row in target_token_values]
            target_tokens = [row[0] for row in target_token_rows]
        else:
            target_token_rows = [[int(x)] for x in target_token_values]
            target_tokens = [row[0] for row in target_token_rows]
        bonus_tokens = [int(x) for x in bonus_token_host.tolist()]
        if draft_len > 1:
            accepted_matrix = [
                [bool(value) for value in row_acceptance]
                for row_acceptance in accepted_host.tolist()
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
            accepted_flags = [bool(x) for x in accepted_host.tolist()]
            prefix_lengths = [1 if accepted else 0 for accepted in accepted_flags]
        if draft_len == 1:
            next_draft_chains = [[int(x)] for x in next_draft_token_host.tolist()]
        else:
            next_draft_chains = [[int(token) for token in chain] for chain in next_draft_token_host.tolist()]
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
            gdn_input_vals = []
            if getattr(layerwise_drift_output, "gdn_input_max_abs", None) is not None:
                gdn_input_vals = [
                    float(v)
                    for v in layerwise_drift_output.gdn_input_max_abs.tolist()
                ]
            gdn_input_names = [
                "mixed_qkv",
                "z",
                "a",
                "b",
                "conv_out",
                "query",
                "key",
                "value",
                "gate",
                "beta",
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
                f"layers={';'.join(layer_rows)} "
                f"gdn0_inputs={','.join(f'{name}={gdn_input_vals[idx]:.6g}' for idx, name in enumerate(gdn_input_names)) if gdn_input_vals else 'n/a'}",
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
        seed_after_bonus = os.environ.get(
            "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
            "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
        ) in {
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
            os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_BONUS", "0")
            in {"1", "true", "yes", "on", "True"}
        )
        rejected_lookahead_rows: List[int] = []
        rejected_lookahead_targets: List[int] = []

        def _debug_token_value(value: object) -> object:
            if isinstance(value, DeviceTokenRef):
                return {"device_ref_row": int(value.row)}
            try:
                return int(value)
            except Exception:
                return str(type(value).__name__)

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
                    outputs[row] = target_token_rows[idx][: prefix_len + 1]
            elif accepted:
                stats["drafts_accepted"] += draft_len
                if disable_bonus:
                    emitted_len = draft_len
                    outputs[row] = target_token_rows[idx][:draft_len]
                else:
                    stats["bonus_tokens"] += 1
                    emitted_len = draft_len + 1
                    outputs[row] = target_token_rows[idx][:draft_len] + [bonus_tokens[idx]]
            else:
                if not forced_reject_probe:
                    stats["drafts_rejected"] += 1
                emitted_len = 1
                outputs[row] = target_tokens[idx]
                lookahead_required_blocks = (
                    seq.num_tokens + 1 + self.block_size - 1
                ) // self.block_size
                can_rejected_lookahead = (
                    draft_len == 1
                    and use_commit_select
                    and batch_accept_policy == "rowwise"
                    and not disable_bonus
                    and not forced_reject_probe
                    and self.mtp1_enabled
                    and seq.temperature == 0
                    and seq.ignore_eos
                    and seq.num_completion_tokens + 2 <= seq.max_tokens
                    and int(batch.query_lens[row]) == 1
                    and len(seq.block_table) >= lookahead_required_blocks
                )
                if can_rejected_lookahead:
                    rejected_lookahead_rows.append(row)
                    rejected_lookahead_targets.append(int(target_tokens[idx]))

            if profile_mtp:
                _draft_debug, debug_events = self._mtp1_debug_state()
                emitted_debug = outputs.get(row, [])
                if not isinstance(emitted_debug, list):
                    emitted_debug = [emitted_debug]
                debug_events.append(
                    {
                        "kind": "mtp_k_commit",
                        "seq_id": int(seq.seq_id),
                        "row": int(row),
                        "verifier_row": int(idx),
                        "verifier_mode": str(verifier_mode),
                        "seq_tokens": int(seq.num_tokens),
                        "completion_tokens": int(seq.num_completion_tokens),
                        "draft_chain": [
                            _debug_token_value(token)
                            for token in draft_token_chains[local_row]
                        ],
                        "target_row": [
                            int(token)
                            for token in target_token_rows[idx][:draft_len]
                        ],
                        "bonus_token": int(bonus_tokens[idx]),
                        "accepted": [
                            bool(value)
                            for value in (
                                accepted_matrix[idx]
                                if draft_len > 1
                                else [accepted_flags[idx]]
                            )
                        ],
                        "prefix_len": int(prefix_len),
                        "emitted": [
                            _debug_token_value(token)
                            for token in emitted_debug
                        ],
                    }
                )

            emitted_bonus = accepted and not disable_bonus
            if draft_len == 1:
                # The rejected-row next-draft invariant is not proven for K=1:
                # direct rejected commit is correct only when the following
                # step falls back to a normal decode. Seeding the verifier's
                # rejected-row next_draft_token causes visible token drift.
                seed_rejected_k1 = os.environ.get(
                    "NANO_VLLM_JAX_MTP_SEED_REJECTED_K1",
                    "0",
                ) in {"1", "true", "yes", "on", "True"}
                seed_rejected_k1 = seed_rejected_k1 or bool(use_generic_k_verifier)
                can_seed_next_chain = seed_after_bonus and (accepted or seed_rejected_k1)
            else:
                seed_partial_k = os.environ.get(
                    "NANO_VLLM_JAX_MTP_SEED_PARTIAL_K",
                    "0",
                ) in {"1", "true", "yes", "on", "True"}
                can_seed_next_chain = (accepted and seed_after_bonus) or (
                    seed_partial_k and prefix_len < draft_len
                )
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

        if rejected_lookahead_rows:
            lookahead_batch = self._masked_decode_batch(
                batch,
                rejected_lookahead_rows,
                token_values=rejected_lookahead_targets,
                position_values=[
                    int(batch.positions[row, 0]) + 1
                    for row in rejected_lookahead_rows
                ],
                seq_len_values=[
                    int(batch.seq_lens[row]) + 1
                    for row in rejected_lookahead_rows
                ],
            )
            # The verifier started from the previous step's carried token. The
            # rejected-row lookahead must consume the just-selected target
            # token, so clear stale carry state before the normal decode helper
            # has a chance to substitute it back into this synthetic batch.
            self._clear_device_token_carry()
            lookahead_outputs = self._run_main_and_seed_mtp_chain_fused(
                seqs,
                lookahead_batch,
                rejected_lookahead_rows,
            )
            if lookahead_outputs is None:
                lookahead_outputs = self._run_main_and_sample(
                    seqs,
                    lookahead_batch,
                    seed_mtp1=False,
                )
            target_by_row = dict(zip(rejected_lookahead_rows, rejected_lookahead_targets))
            for row in rejected_lookahead_rows:
                if row >= len(lookahead_outputs):
                    continue
                lookahead_value = lookahead_outputs[row]
                if lookahead_value == []:
                    continue
                if isinstance(lookahead_value, list):
                    if not lookahead_value:
                        continue
                    lookahead_token = lookahead_value[0]
                else:
                    lookahead_token = lookahead_value
                outputs[row] = [target_by_row[row], lookahead_token]

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

        seed_mtp1 = bool(self.mtp1_enabled)
        force_commit_select = os.environ.get(
            "NANO_VLLM_JAX_MTP_COMMIT_SELECT",
            "1" if getattr(self, "mtp_verifier_impl", "two_decode") == "commit_select" else "0",
        ) in {
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
        allow_unsafe_one_pass_k1 = os.environ.get(
            "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1",
            "0",
        ) in {
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
            seed_mtp1 = bool(
                self.mtp1_enabled
                and bool(getattr(self.config, "mtp_prefill_seed", False))
                and any(prefill_final_flags)
            )
            mtp_max_active_rows = max(
                0,
                int(
                    os.environ.get(
                        "NANO_VLLM_JAX_MTP_MAX_ACTIVE_ROWS",
                        str(getattr(self.config, "mtp_max_active_rows", 0)),
                    )
                    or "0"
                ),
            )
            if mtp_max_active_rows > 0 and len(seqs) > mtp_max_active_rows:
                seed_mtp1 = False
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
            if _unverified_mtp_append_enabled(
                getattr(self, "config", None),
                "mtp_unverified_draft_append",
                "NANO_VLLM_JAX_MTP_UNVERIFIED_DRAFT_APPEND",
            ):
                return self._run_main_and_append_unverified_mtp1_draft(
                    seqs,
                    batch,
                    admitted_mtp_rows,
                )
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
            query_lens_host = batch.query_lens_host
            mtp_max_active_rows = int(
                getattr(
                    self,
                    "mtp_max_active_rows",
                    getattr(getattr(self, "config", None), "mtp_max_active_rows", 0),
                )
                or 0
            )
            use_static_verifier_batch = (
                mtp_max_active_rows > 0 and len(seqs) <= mtp_max_active_rows
            )
            relax_bonus_boundary = os.environ.get(
                "NANO_VLLM_JAX_MTP_RELAX_BONUS_BOUNDARY",
                "0",
            ) in {"1", "true", "yes", "on", "True"}
            verifier_impl = str(getattr(self, "mtp_verifier_impl", "two_decode") or "two_decode")
            strict_k_verifier = self._strict_k_mtp_verifier_enabled()
            row_draft_lens: Dict[int, int] = {}
            for row, seq in enumerate(seqs):
                query_len = (
                    int(query_lens_host[row])
                    if query_lens_host is not None and row < len(query_lens_host)
                    else int(batch.query_lens[row])
                )
                draft_value = self._mtp1_drafts.get(seq.seq_id)
                draft_len = len(draft_value) if isinstance(draft_value, list) else (1 if draft_value is not None else 0)
                draft_len = min(draft_len, max(1, int(getattr(self, "num_speculative_tokens", 1) or 1)))
                row_draft_lens[row] = draft_len
                # The verifier physically writes the current token plus the
                # accepted draft positions. The bonus token is emitted but not
                # written to KV until the next decode step, so requiring block
                # capacity for it here creates avoidable boundary fallbacks.
                remaining_tokens = max(0, int(seq.max_tokens - seq.num_completion_tokens))
                tail_no_bonus = (
                    strict_k_verifier
                    and draft_len > 1
                    and remaining_tokens > 0
                    and (
                        remaining_tokens < draft_len + 1
                        or (
                            not relax_bonus_boundary
                            and remaining_tokens >= draft_len
                            and (seq.num_tokens + draft_len + 1) % self.block_size == 0
                        )
                    )
                )
                verifier_width = max(
                    1,
                    min(draft_len, remaining_tokens) if tail_no_bonus else draft_len,
                )
                required_blocks = (seq.num_tokens + verifier_width + self.block_size - 1) // self.block_size
                unsafe_bonus_boundary = (
                    not relax_bonus_boundary
                    and not tail_no_bonus
                    and (seq.num_tokens + verifier_width + 1) % self.block_size == 0
                )
                can_fuse = (
                    draft_value is not None
                    and draft_len > 0
                    and self._seq_mtp_admitted(seq)
                    and seq.temperature == 0
                    and (
                        seq.num_completion_tokens + draft_len + 1 <= seq.max_tokens
                        or tail_no_bonus
                    )
                    and query_len == 1
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
                    and query_len == 1
                    and len(seq.block_table) >= required_blocks
                    and not unsafe_bonus_boundary
                ):
                    probe_candidate_rows.append(row)
                elif profile_mtp:
                    if draft_value is None:
                        reason = "missing_draft"
                        _draft_debug, debug_events = self._mtp1_debug_state()
                        debug_events.append(
                            {
                                "kind": "mtp_missing_draft",
                                "seq_id": int(seq.seq_id),
                                "row": int(row),
                                "num_tokens": int(seq.num_tokens),
                                "completion_tokens": int(seq.num_completion_tokens),
                                "draft_keys": [
                                    int(key)
                                    for key in sorted(self._mtp1_drafts.keys())
                                ],
                            }
                        )
                    elif draft_len <= 0:
                        reason = "empty_draft"
                    elif not self._seq_mtp_admitted(seq):
                        reason = "scheduler_gate"
                    elif seq.temperature != 0:
                        reason = "temperature"
                    elif (
                        seq.num_completion_tokens + draft_len + 1 > seq.max_tokens
                        and not tail_no_bonus
                    ):
                        reason = "max_tokens"
                    elif query_len != 1:
                        reason = "query_len"
                    elif len(seq.block_table) < required_blocks:
                        reason = "blocks"
                    elif unsafe_bonus_boundary:
                        reason = "bonus_boundary"
                    else:
                        reason = "other"
                    not_fused_reasons[reason] = not_fused_reasons.get(reason, 0) + 1

            strict_k_draft_rows = [
                row
                for row in admitted_mtp_rows
                if row_draft_lens.get(row, 0) > 1
                and max(0, seqs[row].max_tokens - seqs[row].num_completion_tokens) > 0
                and not (
                    not relax_bonus_boundary
                    and max(0, seqs[row].max_tokens - seqs[row].num_completion_tokens)
                    >= row_draft_lens[row] + 1
                    and (
                        seqs[row].num_tokens
                        + max(1, row_draft_lens[row])
                        + 1
                    )
                    % self.block_size
                    == 0
                )
            ]
            strict_k_missing_draft_rows: List[int] = []
            if strict_k_verifier:
                draft_budget = max(
                    1,
                    int(getattr(self, "num_speculative_tokens", 1) or 1),
                )
                for row in admitted_mtp_rows:
                    if row < 0 or row >= len(seqs):
                        continue
                    if row_draft_lens.get(row, 0) > 0:
                        continue
                    seq = seqs[row]
                    query_len = (
                        int(query_lens_host[row])
                        if query_lens_host is not None and row < len(query_lens_host)
                        else int(batch.query_lens[row])
                    )
                    if (
                        query_len == 1
                        and seq.temperature == 0
                        and seq.num_completion_tokens + draft_budget + 1 <= seq.max_tokens
                    ):
                        strict_k_missing_draft_rows.append(row)
                # Missing strict-K drafts are bootstrapped by the same fused
                # main-decode + MTP-seed boundary used by the hot path. This is
                # not verifier repair: no draft token is emitted before target
                # verification, and ordinary/sequential decode fallbacks remain
                # disallowed for strict K.
            allow_mixed_fused = os.environ.get(
                "NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED",
                "1" if verifier_impl == "two_decode" else "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            batch_accept_policy = os.environ.get(
                "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
                str(getattr(self, "mtp_batch_accept_policy", "rowwise") or "rowwise"),
            )
            force_reuse_fallback = os.environ.get("NANO_VLLM_JAX_MTP_FORCE_REUSE_FALLBACK", "0") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            enable_rowwise_repair = os.environ.get("NANO_VLLM_JAX_MTP_ENABLE_ROWWISE_REPAIR", "1") in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            seed_after_bonus_enabled = os.environ.get(
                "NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS",
                "1" if getattr(self, "mtp_seed_after_bonus", False) else "0",
            ) in {
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
            allow_unsafe_one_pass_k1 = os.environ.get(
                "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1",
                "0",
            ) in {
                "1",
                "true",
                "yes",
                "on",
                "True",
            }
            exact_table_one_pass_available = (
                verifier_impl == "two_decode"
                and hasattr(self.executor, "mtp1_two_decode_greedy_table_step_jit")
                and getattr(self, "_hybrid_state_table", None) is not None
                and self._hybrid_state_table.conv_state is not None
                and self._hybrid_state_table.recurrent_state is not None
                and os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1", "0")
                not in {"1", "true", "yes", "on", "True"}
            )
            one_pass_available_for_partial = (
                exact_table_one_pass_available
                or (
                    hasattr(self.executor, "mtp1_two_decode_greedy_step_jit")
                    and os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1", "0")
                    not in {"1", "true", "yes", "on", "True"}
                    and allow_unsafe_one_pass_k1
                    and (not seed_after_bonus_enabled or allow_seeded_one_pass_k1)
                )
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
            allow_verifier_for_batch_shape = (
                full_physical_batch
                or (
                    use_static_verifier_batch
                    and strict_k_verifier
                    and int(getattr(self, "num_speculative_tokens", 1) or 1) > 1
                )
                or (partial_prefix_verifier and allow_partial_commit_select)
            )
            can_seed_for_decode_shape = (
                allow_verifier_for_batch_shape
                and (
                    allow_mixed_fused
                    or allow_exact_commit_select_mixed
                    or homogeneous_full_batch
                    or one_pass_available_for_partial
                    or (
                        strict_k_verifier
                        and int(getattr(self, "num_speculative_tokens", 1) or 1) > 1
                    )
                )
            )
            can_seed_mtp_chain_for_decode_shape = can_seed_for_decode_shape
            k_seed_budget_rows = admitted_mtp_rows
            if int(getattr(self, "num_speculative_tokens", 1) or 1) > 1:
                draft_budget = max(1, int(getattr(self, "num_speculative_tokens", 1) or 1))
                k_seed_budget_rows = []
                for row in admitted_mtp_rows:
                    if row < 0 or row >= len(seqs):
                        continue
                    query_len = (
                        int(query_lens_host[row])
                        if query_lens_host is not None and row < len(query_lens_host)
                        else int(batch.query_lens[row])
                    )
                    seq = seqs[row]
                    if (
                        query_len == 1
                        and seq.temperature == 0
                        and self._seq_mtp_admitted(seq)
                        and seq.num_completion_tokens + draft_budget + 2 <= seq.max_tokens
                    ):
                        k_seed_budget_rows.append(row)
                if set(k_seed_budget_rows) != set(admitted_mtp_rows):
                    can_seed_mtp_chain_for_decode_shape = False
            if strict_k_verifier and strict_k_missing_draft_rows:
                if not can_seed_mtp_chain_for_decode_shape:
                    raise RuntimeError(
                        "K>1 MTP bootstrap could not run for missing-draft rows "
                        f"{strict_k_missing_draft_rows}: "
                        f"allow_verifier_for_batch_shape={allow_verifier_for_batch_shape}, "
                        f"can_seed_mtp_chain_for_decode_shape={can_seed_mtp_chain_for_decode_shape}, "
                        f"reasons={not_fused_reasons}"
                    )
                stats = self._speculative_stats()
                stats["mtp_bootstrap_main_seed_steps"] = (
                    stats.get("mtp_bootstrap_main_seed_steps", 0) + 1
                )
                seeded_outputs = self._run_main_and_seed_mtp_chain_fused(
                    seqs,
                    batch,
                    strict_k_missing_draft_rows,
                )
                missing_after_bootstrap = [
                    row
                    for row in strict_k_missing_draft_rows
                    if seeded_outputs is None
                    or row >= len(seeded_outputs)
                    or seeded_outputs[row] == []
                ]
                if missing_after_bootstrap:
                    raise RuntimeError(
                        "K>1 MTP bootstrap failed to seed strict rows "
                        f"{missing_after_bootstrap}"
                    )
                return seeded_outputs
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
                    if strict_k_verifier and strict_k_draft_rows:
                        raise RuntimeError(
                            "K>1 MTP verifier fallback is disabled; legacy "
                            "compact reuse fallback was requested for rows "
                            f"{strict_k_draft_rows}"
                        )
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
            if strict_k_verifier and strict_k_draft_rows and not can_run_fused_batch:
                raise RuntimeError(
                    "K>1 MTP verifier fallback is disabled; grouped verifier "
                    "could not run for rows "
                    f"{strict_k_draft_rows}. fused_rows={fused_rows}, "
                    f"allow_verifier_for_batch_shape={allow_verifier_for_batch_shape}, "
                    f"can_seed_for_decode_shape={can_seed_for_decode_shape}, "
                    f"force_reuse_fallback={force_reuse_fallback}, "
                    f"reasons={not_fused_reasons}"
                )
            if can_run_fused_batch:
                self._mtp_carry_recorded_this_call = False
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
                    missing_strict_rows = [
                        row
                        for row in strict_k_draft_rows
                        if outputs[row] is None
                    ]
                    if strict_k_verifier and missing_strict_rows:
                        raise RuntimeError(
                            "K>1 MTP verifier fallback is disabled; grouped "
                            "verifier returned no output for rows "
                            f"{missing_strict_rows}"
                        )
                    fallback_rows = [row for row in range(len(seqs)) if outputs[row] is None]
                    if fallback_rows:
                        stats = self._speculative_stats()
                        stats["fallback_partial_rows"] += len(fallback_rows)
                        unresolved_rows = list(fallback_rows)
                        seedable_rows = [
                            row
                            for row in fallback_rows
                            if row in admitted_mtp_rows
                        ]
                        if (
                            self.mtp1_enabled
                            and seed_mtp1
                            and seedable_rows
                            and not strict_k_verifier
                        ):
                            seed_then_outputs = self._run_mtp1_seed_then_table_burst(
                                seqs,
                                batch,
                                seedable_rows,
                            )
                            if seed_then_outputs is not None:
                                for row, value in seed_then_outputs.items():
                                    outputs[row] = value
                                unresolved_rows = [
                                    row for row in unresolved_rows if outputs[row] is None
                                ]
                            if unresolved_rows:
                                stats["fallback_seeded_main_steps"] += 1
                                seeded_outputs = self._run_main_and_seed_mtp_chain_fused(
                                    seqs,
                                    batch,
                                    unresolved_rows,
                                )
                                if seeded_outputs is not None:
                                    next_unresolved = []
                                    for row in unresolved_rows:
                                        value = seeded_outputs[row] if row < len(seeded_outputs) else []
                                        if value == []:
                                            next_unresolved.append(row)
                                        else:
                                            outputs[row] = value
                                    unresolved_rows = next_unresolved
                        if unresolved_rows:
                            stats["fallback_gated_no_spec_steps"] += 1
                            fallback_batch = self._masked_decode_batch(batch, unresolved_rows)
                            fallback_outputs = self._run_main_and_sample(
                                seqs,
                                fallback_batch,
                                seed_mtp1=False,
                            )
                            for row in unresolved_rows:
                                outputs[row] = fallback_outputs[row]
                    if not getattr(self, "_mtp_carry_recorded_this_call", False):
                        self._record_mtp_output_token_carry(
                            batch,
                            seqs,
                            {
                                row: value
                                for row, value in enumerate(outputs)
                                if value is not None
                            },
                        )
                    return [outputs[row] for row in range(len(seqs))]  # type: ignore[list-item]
                if strict_k_verifier and strict_k_draft_rows:
                    raise RuntimeError(
                        "K>1 MTP verifier fallback is disabled; grouped verifier "
                        f"returned None for rows {strict_k_draft_rows}"
                    )
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
                if strict_k_verifier and strict_k_draft_rows:
                    raise RuntimeError(
                        "K>1 MTP verifier fallback is disabled; reuse fallback "
                        f"was requested for rows {strict_k_draft_rows}"
                    )
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
                and not strict_k_verifier
                and can_seed_mtp_chain_for_decode_shape
            )
            stats = self._speculative_stats()
            if main_seed_mtp1:
                self._mtp_carry_recorded_this_call = False
                seed_then_outputs = (
                    None
                    if strict_k_verifier
                    else self._run_mtp1_seed_then_table_burst(
                        seqs,
                        batch,
                        admitted_mtp_rows,
                    )
                )
                if seed_then_outputs is not None:
                    outputs: List[int | List[int] | None] = [None] * len(seqs)
                    for row, value in seed_then_outputs.items():
                        outputs[row] = value
                    unresolved_rows = [
                        row for row in range(len(seqs)) if outputs[row] is None
                    ]
                    if unresolved_rows:
                        stats["fallback_partial_rows"] += len(unresolved_rows)
                        stats["fallback_seeded_main_steps"] += 1
                        seeded_outputs = self._run_main_and_seed_mtp_chain_fused(
                            seqs,
                            batch,
                            unresolved_rows,
                        )
                        if seeded_outputs is not None:
                            for row in unresolved_rows:
                                outputs[row] = (
                                    seeded_outputs[row]
                                    if row < len(seeded_outputs)
                                    else []
                                )
                            unresolved_rows = [
                                row for row in unresolved_rows if outputs[row] == []
                            ]
                    if unresolved_rows:
                        stats["fallback_gated_no_spec_steps"] += 1
                        fallback_batch = self._masked_decode_batch(batch, unresolved_rows)
                        fallback_outputs = self._run_main_and_sample(
                            seqs,
                            fallback_batch,
                            seed_mtp1=False,
                        )
                        for row in unresolved_rows:
                            outputs[row] = fallback_outputs[row]
                    if not getattr(self, "_mtp_carry_recorded_this_call", False):
                        self._record_mtp_output_token_carry(
                            batch,
                            seqs,
                            {
                                row: value
                                for row, value in enumerate(outputs)
                                if value is not None
                            },
                        )
                    return [outputs[row] for row in range(len(seqs))]  # type: ignore[list-item]
                stats["fallback_seeded_main_steps"] += 1
                fused_seed_outputs = self._run_main_and_seed_mtp_chain_fused(
                    seqs,
                    batch,
                    admitted_mtp_rows,
                )
                if fused_seed_outputs is not None:
                    return fused_seed_outputs
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
