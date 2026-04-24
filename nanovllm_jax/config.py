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
    linear_chunk_size: int = 64
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
    
    # Computation dtype (bfloat16 for CPU/CUDA, float16 for Metal)
    dtype: str = "bfloat16"  # Options: "bfloat16", "float16", "float32"
    
    # MTP config
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    
    # KV cache config (for vLLM paging)
    block_size: int = 16
    num_kvcache_blocks: int = 1024
    
    # Scheduler config
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    eos: Optional[int] = None
    
    # Vision config (for multimodal)
    vision_depth: int = 12
    vision_hidden_size: int = 768
    vision_num_heads: int = 12
    vision_patch_size: int = 16
    vision_out_hidden_size: int = 1024
    
    def __post_init__(self):
        """Initialize layer_types if not provided."""
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
            self.block_size,
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
            linear_chunk_size=64,
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
            intermediate_size=5632,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_size=4,
            linear_chunk_size=64,
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
            linear_chunk_size=64,
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
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3_5Config":
        """Create config from dictionary."""
        return cls(**config_dict)
