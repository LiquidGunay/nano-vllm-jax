"""Configuration for Qwen 3.5 model."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass(eq=True, frozen=False)
class Qwen3_5Config:
    """Configuration for Qwen 3.5 model.
    
    Default values match Qwen3.5-0.8B architecture.
    
    Note: This dataclass is made hashable for JAX JIT compilation by:
    1. Using eq=True (default)
    2. Converting lists to tuples for hashing
    3. Implementing __hash__ method
    """
    
    # Model architecture
    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    
    # Gated DeltaNet (linear attention) config
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_size: int = 4
    linear_chunk_size: int = 32
    linear_recurrent_prefill_threshold: int = 8
    use_qk_norm_in_gdn: bool = True
    
    # RoPE config
    rope_theta: float = 10_000_000
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144
    mrope_section: tuple = field(default_factory=lambda: (11, 11, 10))
    
    # Layer types (hybrid architecture)
    # Pattern: 3x linear_attention + 1x full_attention
    layer_types: Optional[tuple] = None
    linear_attn_layers: Optional[tuple] = None
    
    # Other config
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    
    # Computation dtype (bfloat16 for CPU/CUDA, float32 for Metal)
    dtype: str = "float32"  # Options: "bfloat16", "float16", "float32"
    
    # MTP config
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    speculative_method: str = "none"
    draft_sample_method: str = "greedy"
    mtp_verifier_impl: str = "two_decode"
    mtp_batch_accept_policy: str = "rowwise"
    mtp_seed_after_bonus: bool = False
    mtp_bonus_margin: float = 0.0
    mtp_draft_margin: float = 0.0
    mtp_hidden_source: str = "pre_norm"
    mtp_chain_hidden_source: str = "raw"
    mtp_chain_mode: str = "recursive"
    mtp_token_source: str = "generated"
    mtp_position_offset: int = 0
    mtp_lm_head_greedy_top1_impl: str = "jax"
    num_speculative_tokens: int = 0
    mtp_burst_groups: int = 1
    mtp_max_active_rows: int = 0
    mtp_prefill_seed: bool = False
    mtp_unverified_draft_append: bool = False
    mtp_unverified_fused_append: bool = False
    
    # KV cache config (for vLLM paging)
    block_size: int = 16
    num_kvcache_blocks: int = 1024
    max_kv_cache_bytes: int = 512 * 1024 * 1024
    
    # Scheduler config
    max_num_seqs: int = 16
    max_num_resident_seqs: Optional[int] = None
    max_num_batched_tokens: int = 2048
    eos: Optional[int] = None
    prefill_buckets: tuple = field(default_factory=tuple)
    prefill_token_buckets: tuple = field(default_factory=tuple)
    prefill_layout: str = "packed"
    batch_size_buckets: tuple = field(default_factory=tuple)
    max_blocks_per_seq: Optional[int] = None
    decode_block_table_buckets: tuple = field(default_factory=tuple)
    prefix_cache: bool = True
    jax_execution: str = "eager"
    greedy_token_fastpath: bool = True
    sampled_token_fastpath: bool = True
    device_token_carry: bool = False
    static_decode_metadata: bool = False
    static_decode_seq_lens_carry: bool = False
    resident_decode_metadata: bool = False
    greedy_decode_burst_steps: int = 1
    trace_token_prefetch: bool = True

    # Accepted serving fast paths. Environment variables remain supported as
    # compatibility overrides, but server/benchmark configs should set these
    # fields directly.
    materialize_tied_lm_head: bool = False
    compact_prefill_in_proj_qkv: bool = False
    compact_prefill_gdn_z: bool = False
    compact_prefill_full_attn_proj: bool = False
    compact_prefill_mlp: bool = False
    compact_prefill_token_count_mode: str = "exact"
    lm_head_decode_act_dtype: str = "fp32"
    lm_head_topk_impl: str = "jax"
    lm_head_greedy_top1_impl: str = "jax"
    decode_proj_act_dtype: str = "fp32"
    decode_padded_gemm: bool = False
    decode_padded_gemm_gate_up: bool = False
    decode_rms_padded_gemm: bool = False
    decode_padded_gemm_rows: int = 8
    decode_padded_gemm_max_out_dim: int = 300000

    # Kernel policy carried by config. Low-level diagnostic CUDA switches stay
    # env-only; accepted serving kernels should flow through these fields.
    full_attention_kv_cache_dtype: str = "default"
    full_attention_kv_append_impl: str = "reference"
    full_attention_decode_impl: str = "reference"
    full_attention_prefill_impl: str = "reference"
    gdn_disable_fallbacks: bool = False
    gdn_prefill_post_conv_impl: str = "off"
    gdn_prefill_qkv_dtype: str = "fp32"
    gdn_prefill_post_conv_output_dtype: str = "fp32"
    gdn_packed_decode_impl: str = "off"
    gdn_packed_decode_qkv_dtype: str = "fp32"
    gdn_packed_decode_pre_normalize_qk: bool = False
    gdn_packed_decode_max_batch: Optional[int] = None
    
    # Vision config (for multimodal)
    vision_depth: int = 12
    vision_hidden_size: int = 768
    vision_num_heads: int = 12
    vision_patch_size: int = 16
    vision_out_hidden_size: int = 1024
    
    def __post_init__(self):
        """Initialize layer_types if not provided."""
        max_num_seqs = max(1, int(self.max_num_seqs or 1))
        object.__setattr__(self, "max_num_seqs", max_num_seqs)
        resident_raw = self.max_num_resident_seqs
        if resident_raw is None or int(resident_raw) <= 0:
            max_num_resident_seqs = max_num_seqs
        else:
            max_num_resident_seqs = int(resident_raw)
        if max_num_resident_seqs < max_num_seqs:
            raise ValueError("max_num_resident_seqs must be >= max_num_seqs")
        object.__setattr__(self, "max_num_resident_seqs", max_num_resident_seqs)

        for field_name in (
            "prefill_buckets",
            "prefill_token_buckets",
            "batch_size_buckets",
            "decode_block_table_buckets",
        ):
            value = getattr(self, field_name)
            if isinstance(value, str):
                parsed = tuple(int(part) for part in value.split(",") if part.strip())
                object.__setattr__(self, field_name, parsed)
            elif value is None:
                object.__setattr__(self, field_name, ())
            elif not isinstance(value, tuple):
                object.__setattr__(self, field_name, tuple(value))
        prefill_layout = str(self.prefill_layout or "packed").strip().lower()
        if prefill_layout not in {"packed", "dense"}:
            raise ValueError("prefill_layout must be 'packed' or 'dense'")
        object.__setattr__(self, "prefill_layout", prefill_layout)
        object.__setattr__(
            self,
            "greedy_decode_burst_steps",
            max(1, int(self.greedy_decode_burst_steps or 1)),
        )
        num_speculative_tokens = max(0, int(self.num_speculative_tokens or 0))
        if num_speculative_tokens > 8:
            raise ValueError("MTP speculative decoding currently supports num_speculative_tokens <= 8")
        speculative_method = str(self.speculative_method or "none").strip().lower()
        if speculative_method == "none" and num_speculative_tokens > 0:
            # Legacy configs selected MTP by setting only num_speculative_tokens.
            speculative_method = "mtp"
        if speculative_method not in {"none", "mtp"}:
            raise ValueError("speculative_method must be 'none' or 'mtp'")
        draft_sample_method = str(self.draft_sample_method or "greedy").strip().lower()
        if draft_sample_method not in {"greedy", "probabilistic"}:
            raise ValueError("draft_sample_method must be 'greedy' or 'probabilistic'")
        if speculative_method == "mtp" and num_speculative_tokens < 1:
            raise ValueError("speculative_method='mtp' requires num_speculative_tokens >= 1")
        if speculative_method == "mtp" and draft_sample_method != "greedy":
            raise ValueError("MTP probabilistic draft sampling is not implemented yet")
        mtp_verifier_impl = str(self.mtp_verifier_impl or "two_decode").strip().lower()
        if mtp_verifier_impl in {"generic_k", "expanded"}:
            mtp_verifier_impl = "k_decode"
        if mtp_verifier_impl not in {"two_decode", "commit_select", "k_decode"}:
            raise ValueError("mtp_verifier_impl must be 'two_decode', 'commit_select', or 'k_decode'")
        if (
            speculative_method == "mtp"
            and num_speculative_tokens > 1
            and not (
                mtp_verifier_impl == "k_decode"
                or (mtp_verifier_impl == "commit_select" and num_speculative_tokens == 2)
            )
        ):
            raise ValueError("MTP K>1 requires mtp_verifier_impl='k_decode' or K=2 commit_select")
        mtp_batch_accept_policy = str(self.mtp_batch_accept_policy or "rowwise").strip().lower()
        if mtp_batch_accept_policy not in {"rowwise", "all_or_none"}:
            raise ValueError("mtp_batch_accept_policy must be 'rowwise' or 'all_or_none'")
        object.__setattr__(self, "num_speculative_tokens", num_speculative_tokens)
        object.__setattr__(self, "speculative_method", speculative_method)
        object.__setattr__(self, "draft_sample_method", draft_sample_method)
        object.__setattr__(self, "mtp_verifier_impl", mtp_verifier_impl)
        object.__setattr__(self, "mtp_batch_accept_policy", mtp_batch_accept_policy)
        object.__setattr__(self, "mtp_bonus_margin", max(0.0, float(self.mtp_bonus_margin or 0.0)))
        object.__setattr__(self, "mtp_draft_margin", max(0.0, float(self.mtp_draft_margin or 0.0)))
        mtp_hidden_source = str(self.mtp_hidden_source or "pre_norm").strip().lower()
        if mtp_hidden_source not in {"pre_norm", "final_normed"}:
            raise ValueError("mtp_hidden_source must be 'pre_norm' or 'final_normed'")
        mtp_chain_hidden_source = str(self.mtp_chain_hidden_source or "raw").strip().lower()
        if mtp_chain_hidden_source not in {"raw", "final_normed"}:
            raise ValueError("mtp_chain_hidden_source must be 'raw' or 'final_normed'")
        mtp_chain_mode = str(self.mtp_chain_mode or "recursive").strip().lower()
        if mtp_chain_mode not in {"recursive", "sequence"}:
            raise ValueError("mtp_chain_mode must be 'recursive' or 'sequence'")
        mtp_token_source = str(self.mtp_token_source or "generated").strip().lower()
        if mtp_token_source not in {"generated", "current"}:
            raise ValueError("mtp_token_source must be 'generated' or 'current'")
        mtp_lm_head_greedy_top1_impl = str(
            self.mtp_lm_head_greedy_top1_impl or "jax"
        ).strip().lower()
        if mtp_lm_head_greedy_top1_impl not in {
            "jax",
            "triton",
            "triton_tensorcore",
            "triton_top1",
            "cutlass",
            "cutlass_top1",
            "cutlass_fused_gemm",
            "fused_gemm",
        }:
            raise ValueError(
                "mtp_lm_head_greedy_top1_impl must be jax, triton, or cutlass"
            )
        object.__setattr__(self, "mtp_hidden_source", mtp_hidden_source)
        object.__setattr__(self, "mtp_chain_hidden_source", mtp_chain_hidden_source)
        object.__setattr__(self, "mtp_chain_mode", mtp_chain_mode)
        object.__setattr__(self, "mtp_token_source", mtp_token_source)
        object.__setattr__(self, "mtp_position_offset", int(self.mtp_position_offset or 0))
        object.__setattr__(
            self,
            "mtp_lm_head_greedy_top1_impl",
            mtp_lm_head_greedy_top1_impl,
        )
        object.__setattr__(self, "mtp_burst_groups", max(1, int(self.mtp_burst_groups or 1)))
        object.__setattr__(self, "mtp_max_active_rows", max(0, int(self.mtp_max_active_rows or 0)))
        object.__setattr__(self, "mtp_prefill_seed", bool(self.mtp_prefill_seed))
        if bool(self.mtp_unverified_draft_append) or bool(self.mtp_unverified_fused_append):
            raise ValueError(
                "Unverified MTP draft append is not supported. "
                "All MTP benchmark and serving paths must verify drafts with the target model."
            )
        object.__setattr__(self, "mtp_unverified_draft_append", bool(self.mtp_unverified_draft_append))
        object.__setattr__(self, "mtp_unverified_fused_append", bool(self.mtp_unverified_fused_append))
        object.__setattr__(
            self,
            "compact_prefill_token_count_mode",
            str(self.compact_prefill_token_count_mode or "exact").strip().lower(),
        )
        object.__setattr__(
            self,
            "lm_head_decode_act_dtype",
            str(self.lm_head_decode_act_dtype or "fp32").strip().lower(),
        )
        object.__setattr__(
            self,
            "lm_head_topk_impl",
            str(self.lm_head_topk_impl or "jax").strip().lower(),
        )
        object.__setattr__(
            self,
            "lm_head_greedy_top1_impl",
            str(self.lm_head_greedy_top1_impl or "jax").strip().lower(),
        )
        object.__setattr__(
            self,
            "decode_proj_act_dtype",
            str(self.decode_proj_act_dtype or "fp32").strip().lower(),
        )
        object.__setattr__(
            self,
            "decode_padded_gemm_rows",
            max(1, int(self.decode_padded_gemm_rows or 1)),
        )
        object.__setattr__(
            self,
            "decode_padded_gemm_max_out_dim",
            max(1, int(self.decode_padded_gemm_max_out_dim or 1)),
        )
        object.__setattr__(
            self,
            "full_attention_kv_cache_dtype",
            str(self.full_attention_kv_cache_dtype or "default").strip().lower(),
        )
        object.__setattr__(
            self,
            "full_attention_kv_append_impl",
            str(self.full_attention_kv_append_impl or "reference").strip().lower(),
        )
        object.__setattr__(
            self,
            "full_attention_decode_impl",
            str(self.full_attention_decode_impl or "reference").strip().lower(),
        )
        object.__setattr__(
            self,
            "full_attention_prefill_impl",
            str(self.full_attention_prefill_impl or "reference").strip().lower(),
        )
        object.__setattr__(
            self,
            "gdn_prefill_post_conv_impl",
            str(self.gdn_prefill_post_conv_impl or "off").strip().lower(),
        )
        object.__setattr__(
            self,
            "gdn_prefill_qkv_dtype",
            str(self.gdn_prefill_qkv_dtype or "fp32").strip().lower(),
        )
        object.__setattr__(
            self,
            "gdn_prefill_post_conv_output_dtype",
            str(self.gdn_prefill_post_conv_output_dtype or "fp32").strip().lower(),
        )
        object.__setattr__(
            self,
            "gdn_packed_decode_impl",
            str(self.gdn_packed_decode_impl or "off").strip().lower(),
        )
        object.__setattr__(
            self,
            "gdn_packed_decode_qkv_dtype",
            str(self.gdn_packed_decode_qkv_dtype or "fp32").strip().lower(),
        )
        if self.gdn_packed_decode_max_batch is not None:
            max_batch = int(self.gdn_packed_decode_max_batch)
            object.__setattr__(
                self,
                "gdn_packed_decode_max_batch",
                max_batch if max_batch > 0 else None,
            )

        if self.layer_types is None:
            # Default pattern: 3 linear + 1 full attention
            interval = 4
            layer_types = tuple(
                "linear_attention" if (i % interval) != 3 else "full_attention"
                for i in range(self.num_hidden_layers)
            )
            object.__setattr__(self, 'layer_types', layer_types)
        
        # Build list of layer indices that use linear attention
        if self.linear_attn_layers is None:
            linear_attn_layers = tuple(
                i for i, lt in enumerate(self.layer_types)
                if lt == "linear_attention"
            )
            object.__setattr__(self, 'linear_attn_layers', linear_attn_layers)
    
    def __hash__(self):
        """Make config hashable for JAX JIT."""
        # Hash all fields that affect computation
        return hash((
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_size,
            self.linear_chunk_size,
            self.linear_recurrent_prefill_threshold,
            self.use_qk_norm_in_gdn,
            self.rope_theta,
            self.partial_rotary_factor,
            self.max_position_embeddings,
            self.mrope_section,
            self.layer_types,
            self.linear_attn_layers,
            self.hidden_act,
            self.rms_norm_eps,
            self.dtype,
            self.mtp_num_hidden_layers,
            self.mtp_use_dedicated_embeddings,
            self.speculative_method,
            self.draft_sample_method,
            self.mtp_verifier_impl,
            self.mtp_batch_accept_policy,
            self.mtp_seed_after_bonus,
            self.mtp_bonus_margin,
            self.mtp_draft_margin,
            self.mtp_hidden_source,
            self.mtp_chain_hidden_source,
            self.mtp_chain_mode,
            self.mtp_token_source,
            self.mtp_position_offset,
            self.mtp_lm_head_greedy_top1_impl,
            self.num_speculative_tokens,
            self.mtp_burst_groups,
            self.mtp_max_active_rows,
            self.mtp_prefill_seed,
            self.mtp_unverified_draft_append,
            self.mtp_unverified_fused_append,
            self.block_size,
            self.max_kv_cache_bytes,
            self.prefill_token_buckets,
            self.prefill_layout,
            self.decode_block_table_buckets,
            self.prefix_cache,
            self.greedy_token_fastpath,
            self.sampled_token_fastpath,
            self.device_token_carry,
            self.static_decode_metadata,
            self.static_decode_seq_lens_carry,
            self.resident_decode_metadata,
            self.greedy_decode_burst_steps,
            self.trace_token_prefetch,
            self.materialize_tied_lm_head,
            self.compact_prefill_in_proj_qkv,
            self.compact_prefill_gdn_z,
            self.compact_prefill_full_attn_proj,
            self.compact_prefill_mlp,
            self.compact_prefill_token_count_mode,
            self.lm_head_decode_act_dtype,
            self.lm_head_topk_impl,
            self.lm_head_greedy_top1_impl,
            self.decode_proj_act_dtype,
            self.decode_padded_gemm,
            self.decode_padded_gemm_gate_up,
            self.decode_rms_padded_gemm,
            self.decode_padded_gemm_rows,
            self.decode_padded_gemm_max_out_dim,
            self.full_attention_kv_cache_dtype,
            self.full_attention_kv_append_impl,
            self.full_attention_decode_impl,
            self.full_attention_prefill_impl,
            self.gdn_disable_fallbacks,
            self.gdn_prefill_post_conv_impl,
            self.gdn_prefill_qkv_dtype,
            self.gdn_prefill_post_conv_output_dtype,
            self.gdn_packed_decode_impl,
            self.gdn_packed_decode_qkv_dtype,
            self.gdn_packed_decode_pre_normalize_qk,
            self.gdn_packed_decode_max_batch,
        ))
    
    def get_dtype(self):
        """Get JAX dtype from config."""
        import jax.numpy as jnp
        dtype_map = {
            "bfloat16": jnp.bfloat16,
            "float16": jnp.float16,
            "float32": jnp.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"Unknown dtype: {self.dtype}. Options: {list(dtype_map.keys())}")
        return dtype_map[self.dtype]
    
    @classmethod
    def qwen3_5_0_8b(cls) -> "Qwen3_5Config":
        """Qwen3.5-0.8B configuration."""
        return cls(
            vocab_size=248320,
            hidden_size=1024,
            intermediate_size=3584,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_size=4,
            linear_chunk_size=32,
            use_qk_norm_in_gdn=True,
            rope_theta=10_000_000,
            max_position_embeddings=262144,
            tie_word_embeddings=True,
        )
    
    @classmethod
    def qwen3_5_2b(cls) -> "Qwen3_5Config":
        """Qwen3.5-2B configuration."""
        return cls(
            vocab_size=248320,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_size=4,
            linear_chunk_size=32,
            use_qk_norm_in_gdn=True,
            rope_theta=10_000_000,
            max_position_embeddings=262144,
            tie_word_embeddings=True,
        )
    
    @classmethod
    def qwen3_5_27b(cls) -> "Qwen3_5Config":
        """Qwen3.5-27B configuration."""
        return cls(
            vocab_size=248320,
            hidden_size=4608,
            intermediate_size=12032,
            num_hidden_layers=64,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=256,
            linear_num_key_heads=32,
            linear_num_value_heads=64,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_size=4,
            linear_chunk_size=32,
            use_qk_norm_in_gdn=True,
            rope_theta=1_000_000,
            max_position_embeddings=262144,
            tie_word_embeddings=False,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_size": self.linear_conv_kernel_size,
            "linear_chunk_size": self.linear_chunk_size,
            "linear_recurrent_prefill_threshold": self.linear_recurrent_prefill_threshold,
            "use_qk_norm_in_gdn": self.use_qk_norm_in_gdn,
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_types": self.layer_types,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_dropout": self.attention_dropout,
            "attention_bias": self.attention_bias,
            "tie_word_embeddings": self.tie_word_embeddings,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "mtp_use_dedicated_embeddings": self.mtp_use_dedicated_embeddings,
            "speculative_method": self.speculative_method,
            "draft_sample_method": self.draft_sample_method,
            "mtp_verifier_impl": self.mtp_verifier_impl,
            "mtp_batch_accept_policy": self.mtp_batch_accept_policy,
            "mtp_seed_after_bonus": self.mtp_seed_after_bonus,
            "mtp_bonus_margin": self.mtp_bonus_margin,
            "mtp_draft_margin": self.mtp_draft_margin,
            "mtp_hidden_source": self.mtp_hidden_source,
            "mtp_chain_hidden_source": self.mtp_chain_hidden_source,
            "mtp_chain_mode": self.mtp_chain_mode,
            "mtp_token_source": self.mtp_token_source,
            "mtp_position_offset": self.mtp_position_offset,
            "mtp_lm_head_greedy_top1_impl": self.mtp_lm_head_greedy_top1_impl,
            "num_speculative_tokens": self.num_speculative_tokens,
            "mtp_burst_groups": self.mtp_burst_groups,
            "mtp_max_active_rows": self.mtp_max_active_rows,
            "mtp_prefill_seed": self.mtp_prefill_seed,
            "mtp_unverified_draft_append": self.mtp_unverified_draft_append,
            "mtp_unverified_fused_append": self.mtp_unverified_fused_append,
            "max_kv_cache_bytes": self.max_kv_cache_bytes,
            "max_num_seqs": self.max_num_seqs,
            "max_num_resident_seqs": self.max_num_resident_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "prefill_buckets": self.prefill_buckets,
            "prefill_token_buckets": self.prefill_token_buckets,
            "prefill_layout": self.prefill_layout,
            "batch_size_buckets": self.batch_size_buckets,
            "max_blocks_per_seq": self.max_blocks_per_seq,
            "decode_block_table_buckets": self.decode_block_table_buckets,
            "jax_execution": self.jax_execution,
            "greedy_token_fastpath": self.greedy_token_fastpath,
            "sampled_token_fastpath": self.sampled_token_fastpath,
            "device_token_carry": self.device_token_carry,
            "static_decode_metadata": self.static_decode_metadata,
            "static_decode_seq_lens_carry": self.static_decode_seq_lens_carry,
            "resident_decode_metadata": self.resident_decode_metadata,
            "greedy_decode_burst_steps": self.greedy_decode_burst_steps,
            "trace_token_prefetch": self.trace_token_prefetch,
            "materialize_tied_lm_head": self.materialize_tied_lm_head,
            "compact_prefill_in_proj_qkv": self.compact_prefill_in_proj_qkv,
            "compact_prefill_gdn_z": self.compact_prefill_gdn_z,
            "compact_prefill_full_attn_proj": self.compact_prefill_full_attn_proj,
            "compact_prefill_mlp": self.compact_prefill_mlp,
            "compact_prefill_token_count_mode": self.compact_prefill_token_count_mode,
            "lm_head_decode_act_dtype": self.lm_head_decode_act_dtype,
            "lm_head_topk_impl": self.lm_head_topk_impl,
            "lm_head_greedy_top1_impl": self.lm_head_greedy_top1_impl,
            "decode_proj_act_dtype": self.decode_proj_act_dtype,
            "decode_padded_gemm": self.decode_padded_gemm,
            "decode_padded_gemm_gate_up": self.decode_padded_gemm_gate_up,
            "decode_rms_padded_gemm": self.decode_rms_padded_gemm,
            "decode_padded_gemm_rows": self.decode_padded_gemm_rows,
            "decode_padded_gemm_max_out_dim": self.decode_padded_gemm_max_out_dim,
            "full_attention_kv_cache_dtype": self.full_attention_kv_cache_dtype,
            "full_attention_kv_append_impl": self.full_attention_kv_append_impl,
            "full_attention_decode_impl": self.full_attention_decode_impl,
            "full_attention_prefill_impl": self.full_attention_prefill_impl,
            "gdn_disable_fallbacks": self.gdn_disable_fallbacks,
            "gdn_prefill_post_conv_impl": self.gdn_prefill_post_conv_impl,
            "gdn_prefill_qkv_dtype": self.gdn_prefill_qkv_dtype,
            "gdn_prefill_post_conv_output_dtype": self.gdn_prefill_post_conv_output_dtype,
            "gdn_packed_decode_impl": self.gdn_packed_decode_impl,
            "gdn_packed_decode_qkv_dtype": self.gdn_packed_decode_qkv_dtype,
            "gdn_packed_decode_pre_normalize_qk": self.gdn_packed_decode_pre_normalize_qk,
            "gdn_packed_decode_max_batch": self.gdn_packed_decode_max_batch,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3_5Config":
        """Create config from dictionary."""
        return cls(**config_dict)
