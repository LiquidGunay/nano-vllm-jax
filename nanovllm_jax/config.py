"""Configuration for Qwen 3.5 serving.

The small immutable configs below are the mainline boundary: architecture and
serving capacity are configurable, implementation policy is not.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from nanovllm_jax.fastpath import FASTPATH, engine_overrides


def _int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    try:
        parsed = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain integers") from exc
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{field_name} must contain positive integers")
    return parsed


def _bool_value(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True)
class ModelConfig:
    """Architectural values read from the checkpoint config."""

    model: str = "Qwen/Qwen3.5-0.8B"
    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_size: int = 4
    linear_chunk_size: int = 32
    rope_theta: float = 10_000_000
    max_position_embeddings: int = 262144

    @classmethod
    def from_qwen_config(cls, config: "Qwen3_5Config", model: str = "Qwen/Qwen3.5-0.8B") -> "ModelConfig":
        return cls(
            model=model,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            linear_num_key_heads=config.linear_num_key_heads,
            linear_num_value_heads=config.linear_num_value_heads,
            linear_key_head_dim=config.linear_key_head_dim,
            linear_value_head_dim=config.linear_value_head_dim,
            linear_conv_kernel_size=config.linear_conv_kernel_size,
            linear_chunk_size=config.linear_chunk_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )


@dataclass(frozen=True)
class WarmupConfig:
    """Static buckets compiled at startup before the HTTP server is ready."""

    prefill_token_buckets: tuple[int, ...] = (64, 128)
    batch_size_buckets: tuple[int, ...] = (1, 4)
    decode_block_buckets: tuple[int, ...] = (128, 320)
    include_sampled_routes: bool = True
    enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "WarmupConfig":
        raw = raw or {}
        return cls(
            prefill_token_buckets=_int_tuple(
                raw.get("prefill_token_buckets", cls.prefill_token_buckets),
                "warmup.prefill_token_buckets",
            ),
            batch_size_buckets=_int_tuple(
                raw.get("batch_size_buckets", cls.batch_size_buckets),
                "warmup.batch_size_buckets",
            ),
            decode_block_buckets=_int_tuple(
                raw.get("decode_block_buckets", cls.decode_block_buckets),
                "warmup.decode_block_buckets",
            ),
            include_sampled_routes=_bool_value(raw.get("include_sampled_routes", True), "warmup.include_sampled_routes"),
            enabled=_bool_value(raw.get("enabled", True), "warmup.enabled"),
        )


@dataclass(frozen=True)
class EngineConfig:
    """Workload and capacity config for the serving engine."""

    model: str = "Qwen/Qwen3.5-0.8B"
    max_prefill: int = 4096
    max_num_seqs: int = 8
    max_num_resident_seqs: int = 8
    max_num_batched_tokens: int = 4096
    max_blocks_per_seq: int = 320
    kv_cache_bytes: int = 3072 * 1024 * 1024
    num_kvcache_blocks: int = 2048
    prefill_token_buckets: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)
    batch_size_buckets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    decode_block_buckets: tuple[int, ...] = (128, 256, 320)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    prefix_cache: bool = True

    def __post_init__(self):
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_resident_seqs < self.max_num_seqs:
            raise ValueError("max_num_resident_seqs must be >= max_num_seqs")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_blocks_per_seq <= 0:
            raise ValueError("max_blocks_per_seq must be positive")
        if self.kv_cache_bytes <= 0:
            raise ValueError("kv_cache_bytes must be positive")
        if self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "EngineConfig":
        raw = raw or {}
        kv_cache_bytes = raw.get("kv_cache_bytes")
        if kv_cache_bytes is None and raw.get("kv_cache_mb") is not None:
            kv_cache_bytes = int(raw["kv_cache_mb"]) * 1024 * 1024
        if kv_cache_bytes is None and raw.get("max_kv_cache_mb") is not None:
            kv_cache_bytes = int(raw["max_kv_cache_mb"]) * 1024 * 1024
        if kv_cache_bytes is None:
            kv_cache_bytes = cls.kv_cache_bytes
        return cls(
            model=str(raw.get("model", cls.model)),
            max_prefill=int(raw.get("max_prefill", cls.max_prefill)),
            max_num_seqs=int(raw.get("max_num_seqs", cls.max_num_seqs)),
            max_num_resident_seqs=int(
                raw.get("max_num_resident_seqs", raw.get("max_num_seqs", cls.max_num_resident_seqs))
            ),
            max_num_batched_tokens=int(raw.get("max_num_batched_tokens", cls.max_num_batched_tokens)),
            max_blocks_per_seq=int(raw.get("max_blocks_per_seq", cls.max_blocks_per_seq)),
            kv_cache_bytes=int(kv_cache_bytes),
            num_kvcache_blocks=int(raw.get("num_kvcache_blocks", raw.get("num_kv_cache_blocks", cls.num_kvcache_blocks))),
            prefill_token_buckets=_int_tuple(
                raw.get("prefill_token_buckets", cls.prefill_token_buckets),
                "prefill_token_buckets",
            ),
            batch_size_buckets=_int_tuple(
                raw.get("batch_size_buckets", cls.batch_size_buckets),
                "batch_size_buckets",
            ),
            decode_block_buckets=_int_tuple(
                raw.get("decode_block_buckets", raw.get("decode_block_table_buckets", cls.decode_block_buckets)),
                "decode_block_buckets",
            ),
            warmup=WarmupConfig.from_mapping(raw.get("warmup")),
            prefix_cache=_bool_value(raw.get("prefix_cache", True), "prefix_cache"),
        )

    def to_engine_kwargs(self) -> dict[str, Any]:
        """Project public capacity plus canonical policy into engine kwargs."""

        kwargs = dict(engine_overrides(FASTPATH))
        kwargs.update(
            {
                "max_prefill": self.max_prefill,
                "num_kvcache_blocks": self.num_kvcache_blocks,
                "max_kv_cache_bytes": self.kv_cache_bytes,
                "max_num_seqs": self.max_num_seqs,
                "max_num_resident_seqs": self.max_num_resident_seqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_blocks_per_seq": self.max_blocks_per_seq,
                "prefill_buckets": self.prefill_token_buckets,
                "prefill_token_buckets": self.prefill_token_buckets,
                "batch_size_buckets": self.batch_size_buckets,
                "decode_block_table_buckets": self.decode_block_buckets,
                "prefix_cache": self.prefix_cache,
            }
        )
        return kwargs


@dataclass(frozen=True)
class ServerSettings:
    """Transport settings plus the engine capacity config."""

    host: str = "127.0.0.1"
    port: int = 6791
    max_tokens_default: int = 128
    engine: EngineConfig = field(default_factory=EngineConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ServerSettings":
        raw = raw or {}
        server = raw.get("server", {}) or {}
        return cls(
            host=str(server.get("host", cls.host)),
            port=int(server.get("port", cls.port)),
            max_tokens_default=int(server.get("max_tokens_default", cls.max_tokens_default)),
            engine=EngineConfig.from_mapping(raw.get("engine")),
        )


def load_engine_config(path: str | Path = "server.yaml") -> ServerSettings:
    """Load the new serving config without projecting policy from YAML."""

    import yaml

    config_path = Path(path)
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return ServerSettings.from_mapping(raw)


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
    
    # Computation dtype for the promoted JAX/CUDA path.
    dtype: str = "float32"  # Options: "bfloat16", "float16", "float32"
    
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

    # Internal implementation policy projected from nanovllm_jax.fastpath.
    # These fields stay here because the model and executor read one config.
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

    # Kernel policy projected from nanovllm_jax.fastpath, not user config.
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
