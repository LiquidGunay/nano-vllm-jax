"""Configuration for Qwen 3.5 serving.

The small immutable configs below are the mainline boundary: architecture and
serving capacity are configurable, implementation policy is not.
"""

from dataclasses import dataclass, field
import json
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
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(f"{field_name} must be sorted and unique")
    return parsed


def _strict_keys(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(unknown)}")


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


def _layer_pattern(count: int) -> tuple[str, ...]:
    return tuple(
        "linear_attention" if index % 4 != 3 else "full_attention"
        for index in range(count)
    )


_COMMON_ARCHITECTURE = {
    "vocab_size": 248320,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_size": 4,
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "mrope_section": (11, 11, 10),
    "mrope_interleaved": True,
    "max_position_embeddings": 262144,
    "full_attention_interval": 4,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "attention_bias": False,
    "attn_output_gate": True,
    "mamba_ssm_dtype": "float32",
    "tie_word_embeddings": True,
}

_SUPPORTED_ARCHITECTURES = {
    (1024, 24): {
        "size": "0.8B",
        "intermediate_size": 3584,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 16,
    },
    (2048, 24): {
        "size": "2B",
        "intermediate_size": 6144,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 16,
    },
    (2560, 32): {
        "size": "4B",
        "intermediate_size": 9216,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "linear_num_value_heads": 32,
    },
}


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
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, ...] = (11, 11, 10)
    max_position_embeddings: int = 262144
    layer_types: tuple[str, ...] = _layer_pattern(24)
    full_attention_interval: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    attn_output_gate: bool = True
    mrope_interleaved: bool = True
    mamba_ssm_dtype: str = "float32"
    tie_word_embeddings: bool = True
    eos_token_id: int | None = 248044

    def __post_init__(self) -> None:
        key = (self.hidden_size, self.num_hidden_layers)
        variant = _SUPPORTED_ARCHITECTURES.get(key)
        if variant is None:
            raise ValueError(
                "unsupported Qwen3.5 dense text shape "
                f"hidden_size={self.hidden_size}, num_hidden_layers={self.num_hidden_layers}; "
                "validated sizes are 0.8B, 2B, and 4B"
            )
        size = str(variant["size"])
        expected = {
            **_COMMON_ARCHITECTURE,
            **{name: value for name, value in variant.items() if name != "size"},
            "hidden_size": key[0],
            "num_hidden_layers": key[1],
            "layer_types": _layer_pattern(key[1]),
        }
        for name, expected_value in expected.items():
            actual = getattr(self, name)
            if actual != expected_value:
                raise ValueError(
                    f"unsupported Qwen3.5-{size} architecture: "
                    f"{name}={actual!r}, expected {expected_value!r}"
                )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, model: str) -> "ModelConfig":
        config_path = Path(checkpoint) / "config.json"
        try:
            raw = json.loads(config_path.read_text())
        except FileNotFoundError as exc:
            raise ValueError(f"checkpoint has no config.json: {checkpoint}") from exc
        if raw.get("model_type") != "qwen3_5":
            raise ValueError(f"unsupported model_type={raw.get('model_type')!r}; expected 'qwen3_5'")
        text = raw.get("text_config")
        if not isinstance(text, Mapping) or text.get("model_type") != "qwen3_5_text":
            raise ValueError("checkpoint does not contain a Qwen3.5 text_config")
        if text.get("mlp_only_layers", []) not in ([], None):
            raise ValueError("mlp-only layers are not supported")
        rope = text.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("only default Qwen3.5 RoPE is supported")
        return cls(
            model=model,
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_conv_kernel_size=int(text["linear_conv_kernel_dim"]),
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(rope["partial_rotary_factor"]),
            mrope_section=tuple(int(value) for value in rope["mrope_section"]),
            mrope_interleaved=bool(rope["mrope_interleaved"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            layer_types=tuple(str(value) for value in text["layer_types"]),
            full_attention_interval=int(text["full_attention_interval"]),
            hidden_act=str(text["hidden_act"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            attention_dropout=float(text["attention_dropout"]),
            attention_bias=bool(text["attention_bias"]),
            attn_output_gate=bool(text["attn_output_gate"]),
            mamba_ssm_dtype=str(text["mamba_ssm_dtype"]),
            tie_word_embeddings=bool(text["tie_word_embeddings"]),
            eos_token_id=(
                int(text["eos_token_id"])
                if text.get("eos_token_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class WarmupConfig:
    """Static buckets compiled at startup before the HTTP server is ready."""

    prefill_token_buckets: tuple[int, ...] = (64, 128)
    batch_size_buckets: tuple[int, ...] = (1, 4)
    decode_block_buckets: tuple[int, ...] = (128, 320)
    include_sampled_routes: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("prefill_token_buckets", "batch_size_buckets", "decode_block_buckets"):
            value = getattr(self, name)
            if value != tuple(sorted(set(value))) or any(item <= 0 for item in value):
                raise ValueError(f"warmup.{name} must be sorted, unique, and positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "WarmupConfig":
        raw = raw or {}
        _strict_keys(
            raw,
            {
                "prefill_token_buckets",
                "batch_size_buckets",
                "decode_block_buckets",
                "include_sampled_routes",
                "enabled",
            },
            "warmup",
        )
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


def _derived_warmup(
    prefill: tuple[int, ...],
    batches: tuple[int, ...],
    decode: tuple[int, ...],
) -> WarmupConfig:
    def subset(configured: tuple[int, ...], preferred: tuple[int, ...]) -> tuple[int, ...]:
        selected = tuple(value for value in preferred if value in configured)
        return selected or configured[:1]

    return WarmupConfig(
        prefill_token_buckets=subset(prefill, WarmupConfig.prefill_token_buckets),
        batch_size_buckets=subset(batches, WarmupConfig.batch_size_buckets),
        decode_block_buckets=subset(decode, WarmupConfig.decode_block_buckets),
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
        if self.max_prefill <= 0:
            raise ValueError("max_prefill must be positive")
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
        for name in ("prefill_token_buckets", "batch_size_buckets", "decode_block_buckets"):
            value = getattr(self, name)
            if not value or value != tuple(sorted(set(value))) or any(item <= 0 for item in value):
                raise ValueError(f"{name} must be sorted, unique, and positive")
        if max(self.prefill_token_buckets) < self.max_prefill:
            raise ValueError("prefill_token_buckets must cover max_prefill")
        if max(self.prefill_token_buckets) < self.max_num_batched_tokens:
            raise ValueError("prefill_token_buckets must cover max_num_batched_tokens")
        if max(self.batch_size_buckets) < self.max_num_seqs:
            raise ValueError("batch_size_buckets must cover max_num_seqs")
        if max(self.decode_block_buckets) < self.max_blocks_per_seq:
            raise ValueError("decode_block_buckets must cover max_blocks_per_seq")
        if not set(self.warmup.prefill_token_buckets).issubset(self.prefill_token_buckets):
            raise ValueError("warmup prefill buckets must be configured engine buckets")
        if not set(self.warmup.batch_size_buckets).issubset(self.batch_size_buckets):
            raise ValueError("warmup batch buckets must be configured engine buckets")
        if not set(self.warmup.decode_block_buckets).issubset(self.decode_block_buckets):
            raise ValueError("warmup decode buckets must be configured engine buckets")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "EngineConfig":
        raw = raw or {}
        _strict_keys(
            raw,
            {
                "model",
                "max_prefill",
                "max_num_seqs",
                "max_num_resident_seqs",
                "max_num_batched_tokens",
                "max_blocks_per_seq",
                "kv_cache_bytes",
                "num_kvcache_blocks",
                "prefill_token_buckets",
                "batch_size_buckets",
                "decode_block_buckets",
                "warmup",
                "prefix_cache",
            },
            "engine",
        )
        prefill_buckets = _int_tuple(
            raw.get("prefill_token_buckets", cls.prefill_token_buckets),
            "prefill_token_buckets",
        )
        batch_buckets = _int_tuple(
            raw.get("batch_size_buckets", cls.batch_size_buckets),
            "batch_size_buckets",
        )
        decode_buckets = _int_tuple(
            raw.get("decode_block_buckets", cls.decode_block_buckets),
            "decode_block_buckets",
        )
        warmup = _derived_warmup(prefill_buckets, batch_buckets, decode_buckets)
        if "warmup" in raw:
            warmup_raw = dict(raw.get("warmup") or {})
            warmup_raw.setdefault("prefill_token_buckets", warmup.prefill_token_buckets)
            warmup_raw.setdefault("batch_size_buckets", warmup.batch_size_buckets)
            warmup_raw.setdefault("decode_block_buckets", warmup.decode_block_buckets)
            warmup = WarmupConfig.from_mapping(warmup_raw)
        return cls(
            model=str(raw.get("model", cls.model)),
            max_prefill=int(raw.get("max_prefill", cls.max_prefill)),
            max_num_seqs=int(raw.get("max_num_seqs", cls.max_num_seqs)),
            max_num_resident_seqs=int(
                raw.get("max_num_resident_seqs", raw.get("max_num_seqs", cls.max_num_resident_seqs))
            ),
            max_num_batched_tokens=int(raw.get("max_num_batched_tokens", cls.max_num_batched_tokens)),
            max_blocks_per_seq=int(raw.get("max_blocks_per_seq", cls.max_blocks_per_seq)),
            kv_cache_bytes=int(raw.get("kv_cache_bytes", cls.kv_cache_bytes)),
            num_kvcache_blocks=int(raw.get("num_kvcache_blocks", cls.num_kvcache_blocks)),
            prefill_token_buckets=prefill_buckets,
            batch_size_buckets=batch_buckets,
            decode_block_buckets=decode_buckets,
            warmup=warmup,
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
        _strict_keys(raw, {"server", "engine"}, "top-level")
        server = raw.get("server", {}) or {}
        _strict_keys(server, {"host", "port", "max_tokens_default"}, "server")
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
class RuntimeConfig:
    """Private merged runtime config for Qwen 3.5 serving.
    
    Default architecture values match Qwen3.5-0.8B.
    Public workload and capacity config lives in ``EngineConfig``; this object
    is the internal bridge consumed by model, scheduler, runner, and executor.
    
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
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
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
    gdn_width1_packed_input_projection: bool = False

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
        object.__setattr__(
            self,
            "eos_token_ids",
            tuple(sorted({int(token_id) for token_id in self.eos_token_ids})),
        )

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
            self.gdn_width1_packed_input_projection,
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
    def qwen3_5_0_8b(cls) -> "RuntimeConfig":
        """Small deterministic config used by focused model tests."""
        return cls.from_model_config(ModelConfig())

    @classmethod
    def from_model_config(
        cls,
        model: ModelConfig,
        **runtime: Any,
    ) -> "RuntimeConfig":
        """Combine checkpoint architecture with serving/runtime policy."""
        architecture = {
            "vocab_size": model.vocab_size,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "num_hidden_layers": model.num_hidden_layers,
            "num_attention_heads": model.num_attention_heads,
            "num_key_value_heads": model.num_key_value_heads,
            "head_dim": model.head_dim,
            "linear_num_key_heads": model.linear_num_key_heads,
            "linear_num_value_heads": model.linear_num_value_heads,
            "linear_key_head_dim": model.linear_key_head_dim,
            "linear_value_head_dim": model.linear_value_head_dim,
            "linear_conv_kernel_size": model.linear_conv_kernel_size,
            "linear_chunk_size": model.linear_chunk_size,
            "rope_theta": model.rope_theta,
            "partial_rotary_factor": model.partial_rotary_factor,
            "mrope_section": model.mrope_section,
            "max_position_embeddings": model.max_position_embeddings,
            "layer_types": model.layer_types,
            "linear_attn_layers": tuple(
                index
                for index, layer_type in enumerate(model.layer_types)
                if layer_type == "linear_attention"
            ),
            "hidden_act": model.hidden_act,
            "rms_norm_eps": model.rms_norm_eps,
            "attention_dropout": model.attention_dropout,
            "attention_bias": model.attention_bias,
            "tie_word_embeddings": model.tie_word_embeddings,
            "eos_token_ids": (
                (model.eos_token_id,)
                if model.eos_token_id is not None
                else ()
            ),
        }
        overlap = set(architecture) & set(runtime)
        if overlap:
            raise ValueError(
                "runtime policy cannot override checkpoint architecture: "
                + ", ".join(sorted(overlap))
            )
        return cls(**architecture, **runtime)
    
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
            "eos_token_ids": self.eos_token_ids,
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
            "gdn_width1_packed_input_projection": self.gdn_width1_packed_input_projection,
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RuntimeConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
