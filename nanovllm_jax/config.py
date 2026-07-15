"""Configuration for Qwen 3.5 serving.

The small immutable configs below are the mainline boundary: architecture and
serving capacity are configurable, implementation policy is not.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from nanovllm_jax.fastpath import KERNEL_PLAN, KernelPlan
from nanovllm_jax.speculation import DrafterConfig


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


def _checkpoint_bool(
    raw: Mapping[str, Any],
    name: str,
    default: bool | None = None,
) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


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
    "use_qk_norm_in_gdn": True,
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
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
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


def _checkpoint_model_fields(checkpoint: str | Path, model: str) -> dict[str, Any]:
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
    use_qk_norm = _checkpoint_bool(text, "use_qk_norm_in_gdn", True)
    return {
        "model": model,
        "vocab_size": int(text["vocab_size"]),
        "hidden_size": int(text["hidden_size"]),
        "intermediate_size": int(text["intermediate_size"]),
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "linear_num_key_heads": int(text["linear_num_key_heads"]),
        "linear_num_value_heads": int(text["linear_num_value_heads"]),
        "linear_key_head_dim": int(text["linear_key_head_dim"]),
        "linear_value_head_dim": int(text["linear_value_head_dim"]),
        "linear_conv_kernel_size": int(text["linear_conv_kernel_dim"]),
        "use_qk_norm_in_gdn": use_qk_norm,
        "rope_theta": float(rope["rope_theta"]),
        "partial_rotary_factor": float(rope["partial_rotary_factor"]),
        "mrope_section": tuple(int(value) for value in rope["mrope_section"]),
        "mrope_interleaved": _checkpoint_bool(rope, "mrope_interleaved"),
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "layer_types": tuple(str(value) for value in text["layer_types"]),
        "full_attention_interval": int(text["full_attention_interval"]),
        "hidden_act": str(text["hidden_act"]),
        "rms_norm_eps": float(text["rms_norm_eps"]),
        "attention_dropout": float(text["attention_dropout"]),
        "attention_bias": _checkpoint_bool(text, "attention_bias"),
        "attn_output_gate": _checkpoint_bool(text, "attn_output_gate"),
        "mamba_ssm_dtype": str(text["mamba_ssm_dtype"]),
        "tie_word_embeddings": _checkpoint_bool(text, "tie_word_embeddings"),
        "mtp_num_hidden_layers": int(text.get("mtp_num_hidden_layers", 1)),
        "mtp_use_dedicated_embeddings": _checkpoint_bool(
            text,
            "mtp_use_dedicated_embeddings",
            False,
        ),
        "eos_token_id": (
            int(text["eos_token_id"])
            if text.get("eos_token_id") is not None
            else None
        ),
    }


@dataclass(frozen=True)
class ModelSpec:
    """Qwen3.5 text architecture consumed by model code."""

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
    use_qk_norm_in_gdn: bool = True
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
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    eos_token_id: int | None = 248044

    @property
    def linear_attn_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "linear_attention"
        )


@dataclass(frozen=True)
class ModelConfig(ModelSpec):
    """Validated architecture read from a supported checkpoint."""

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, model: str) -> "ModelConfig":
        return cls(**_checkpoint_model_fields(checkpoint, model))

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
        for name in ("prefill_token_buckets", "batch_size_buckets", "decode_block_buckets"):
            value = getattr(self, name)
            if not value or value != tuple(sorted(set(value))) or any(item <= 0 for item in value):
                raise ValueError(f"{name} must be sorted, unique, and positive")
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


@dataclass(frozen=True)
class CapacitySpec:
    """Request and cache capacity owned by the engine."""

    block_size: int = 16
    num_kvcache_blocks: int = 1024
    max_kv_cache_bytes: int = 512 * 1024 * 1024
    max_num_seqs: int = 16
    max_num_resident_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_blocks_per_seq: int = 64
    eos_token_ids: tuple[int, ...] = ()
    prefix_cache: bool = True

    def __post_init__(self) -> None:
        for name in (
            "block_size",
            "num_kvcache_blocks",
            "max_kv_cache_bytes",
            "max_num_seqs",
            "max_num_resident_seqs",
            "max_num_batched_tokens",
            "max_blocks_per_seq",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_num_resident_seqs < self.max_num_seqs:
            raise ValueError("max_num_resident_seqs must be >= max_num_seqs")
        object.__setattr__(
            self,
            "eos_token_ids",
            tuple(sorted({int(token_id) for token_id in self.eos_token_ids})),
        )


@dataclass(frozen=True)
class CompileSpec:
    """Dtypes and static bucket shapes owned by compiled execution."""

    dtype: str = "float32"
    weight_dtype: str = "float32"
    execution: str = "eager"
    prefill_layout: str = "packed"
    prefill_token_buckets: tuple[int, ...] = ()
    batch_size_buckets: tuple[int, ...] = ()
    decode_block_table_buckets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unknown compute dtype: {self.dtype}")
        if self.weight_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unknown weight dtype: {self.weight_dtype}")
        if self.execution not in {"eager", "decode-jit", "jit"}:
            raise ValueError(f"unknown execution mode: {self.execution}")
        if self.prefill_layout not in {"packed", "dense"}:
            raise ValueError("prefill_layout must be 'packed' or 'dense'")
        for name in (
            "prefill_token_buckets",
            "batch_size_buckets",
            "decode_block_table_buckets",
        ):
            value = getattr(self, name)
            if value != tuple(sorted(set(value))) or any(item <= 0 for item in value):
                raise ValueError(f"{name} must be sorted, unique, and positive")

    def jax_dtype(self):
        import jax.numpy as jnp

        return {
            "bfloat16": jnp.bfloat16,
            "float16": jnp.float16,
            "float32": jnp.float32,
        }[self.dtype]


@dataclass(frozen=True)
class RuntimeSpec:
    """Frozen internal aggregate composed once by the engine."""

    model: ModelSpec
    capacity: CapacitySpec
    compile: CompileSpec
    kernels: KernelPlan
    drafter: DrafterConfig | None = None

    def __post_init__(self) -> None:
        if self.drafter is None:
            return
        if self.model.mtp_num_hidden_layers != 1:
            raise ValueError("persistent MTP requires exactly one predictor layer")
        if self.model.mtp_use_dedicated_embeddings or not self.model.tie_word_embeddings:
            raise ValueError("persistent MTP requires the checkpoint's tied embeddings")
        rotary_dim = int(self.model.head_dim * self.model.partial_rotary_factor)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError("persistent MTP requires a positive even rotary dimension")
        if self.compile.execution != "jit" or self.compile.prefill_layout != "packed":
            raise ValueError("persistent MTP requires JIT execution and packed prefill")
        if self.capacity.prefix_cache:
            raise ValueError("persistent MTP currently requires prefix_cache=False")
        missing_batch_sizes = tuple(
            size
            for size in range(1, self.capacity.max_num_seqs + 1)
            if size not in self.compile.batch_size_buckets
        )
        if missing_batch_sizes:
            raise ValueError(
                "persistent MTP requires exact batch_size_buckets for every "
                f"admitted batch size; missing {missing_batch_sizes}"
            )
        required = {
            "greedy_token_fastpath": self.kernels.greedy_token_fastpath,
            "device_token_carry": self.kernels.device_token_carry,
            "static_decode_metadata": self.kernels.static_decode_metadata,
            "resident_decode_metadata": self.kernels.resident_decode_metadata,
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "persistent MTP requires " + ", ".join(missing)
            )

    @classmethod
    def promoted(
        cls,
        model: ModelConfig,
        engine: EngineConfig,
        kernels: KernelPlan = KERNEL_PLAN,
        drafter: DrafterConfig | None = None,
    ) -> "RuntimeSpec":
        eos_token_ids = (
            (model.eos_token_id,)
            if model.eos_token_id is not None
            else ()
        )
        return cls(
            model=model,
            capacity=CapacitySpec(
                num_kvcache_blocks=engine.num_kvcache_blocks,
                max_kv_cache_bytes=engine.kv_cache_bytes,
                max_num_seqs=engine.max_num_seqs,
                max_num_resident_seqs=engine.max_num_resident_seqs,
                max_num_batched_tokens=engine.max_num_batched_tokens,
                max_blocks_per_seq=engine.max_blocks_per_seq,
                eos_token_ids=eos_token_ids,
                prefix_cache=engine.prefix_cache,
            ),
            compile=CompileSpec(
                dtype="bfloat16",
                weight_dtype="bfloat16",
                execution="jit",
                prefill_layout="packed",
                prefill_token_buckets=engine.prefill_token_buckets,
                batch_size_buckets=engine.batch_size_buckets,
                decode_block_table_buckets=engine.decode_block_buckets,
            ),
            kernels=kernels,
            drafter=drafter,
        )
