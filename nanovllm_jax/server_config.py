"""YAML-based runtime configuration with environment-variable overrides.

Loads ``server_config.yaml`` (or a path given by ``NANO_VLLM_JAX_SERVER_CONFIG``)
at startup.  Server/engine keys keep their existing env-var overrides, and
runtime/kernel env vars already set in the shell win over values translated from
the config file.

The config is split into two sections:

* ``server`` – Flask / network knobs (host, port, max_tokens_default).
* ``engine`` – LLMEngine / scheduler knobs (model, dtype, buckets, KV cache …).
* ``runtime`` – Process/JAX startup knobs such as platform, allocator, cache,
                and serving fast-path toggles.
* ``kernels`` – Kernel policy in typed, readable form.  This is translated to
                the legacy env vars before JAX/XLA init.
* ``env``     – Compatibility escape hatch for low-level process variables and
                old configs.  Prefer ``runtime``/``kernels`` for new configs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_FILENAME = "server_config.yaml"


def _default_config_path() -> Path | None:
    """Return the default config file path, or *None* if none exists."""
    explicit = os.getenv("NANO_VLLM_JAX_SERVER_CONFIG")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit server config not found: {p}")

    # Search next to server.py and in the package directory
    candidates = [
        Path(__file__).resolve().parent.parent / _DEFAULT_CONFIG_FILENAME,  # repo root
        Path(__file__).resolve().parent / _DEFAULT_CONFIG_FILENAME,         # package dir
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Env-var override helpers
# ---------------------------------------------------------------------------

_ENV_PREFIX = "NANO_VLLM_JAX_"

# Mapping: config-key -> (env-var-name, type-coerce-fn)
# Only keys that have a meaningful env-var override are listed.
_SERVER_ENV_MAP: dict[str, tuple[str, type]] = {
    "host": ("NANO_VLLM_JAX_HOST", str),
    "port": ("NANO_VLLM_JAX_PORT", int),
    "max_tokens_default": ("NANO_VLLM_JAX_MAX_TOKENS_DEFAULT", int),
}

_ENGINE_ENV_MAP: dict[str, tuple[str, type]] = {
    "model": ("NANO_VLLM_JAX_MODEL", str),
    "backend": ("NANO_VLLM_JAX_BACKEND", str),
    "dtype": ("NANO_VLLM_JAX_DTYPE", str),
    "weight_dtype": ("NANO_VLLM_JAX_WEIGHT_DTYPE", str),
    "jax_execution": ("NANO_VLLM_JAX_JAX_EXECUTION", str),
    "prefill_buckets": ("NANO_VLLM_JAX_PREFILL_BUCKETS", str),
    "prefill_token_buckets": ("NANO_VLLM_JAX_PREFILL_TOKEN_BUCKETS", str),
    "prefill_layout": ("NANO_VLLM_JAX_PREFILL_LAYOUT", str),
    "batch_size_buckets": ("NANO_VLLM_JAX_BATCH_SIZE_BUCKETS", str),
    "decode_block_table_buckets": ("NANO_VLLM_JAX_DECODE_BLOCK_TABLE_BUCKETS", str),
    "max_prefill": ("NANO_VLLM_JAX_MAX_PREFILL", int),
    "max_kv_cache_mb": ("NANO_VLLM_JAX_MAX_KV_CACHE_MB", int),
    "num_kvcache_blocks": ("NANO_VLLM_JAX_NUM_KVCACHE_BLOCKS", int),
    "max_num_seqs": ("NANO_VLLM_JAX_MAX_NUM_SEQS", int),
    "max_num_resident_seqs": ("NANO_VLLM_JAX_MAX_NUM_RESIDENT_SEQS", int),
    "max_num_batched_tokens": ("NANO_VLLM_JAX_MAX_NUM_BATCHED_TOKENS", int),
    "num_speculative_tokens": ("NANO_VLLM_JAX_NUM_SPECULATIVE_TOKENS", int),
    "greedy_token_fastpath": ("NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH", bool),
    "device_token_carry": ("NANO_VLLM_JAX_DEVICE_TOKEN_CARRY", bool),
    "static_decode_metadata": ("NANO_VLLM_JAX_STATIC_DECODE_METADATA", bool),
    "static_decode_seq_lens_carry": ("NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY", bool),
    "greedy_decode_burst_steps": ("NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS", int),
    "materialize_tied_lm_head": ("NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD", bool),
    "compact_prefill_in_proj_qkv": ("NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV", bool),
    "compact_prefill_gdn_z": ("NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z", bool),
    "compact_prefill_full_attn_proj": ("NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ", bool),
    "compact_prefill_mlp": ("NANO_VLLM_JAX_COMPACT_PREFILL_MLP", bool),
    "compact_prefill_token_count_mode": ("NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE", str),
    "lm_head_decode_act_dtype": ("NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE", str),
    "decode_proj_act_dtype": ("NANO_VLLM_JAX_DECODE_PROJ_ACT_DTYPE", str),
    "decode_padded_gemm": ("NANO_VLLM_JAX_DECODE_PADDED_GEMM", bool),
    "decode_padded_gemm_gate_up": ("NANO_VLLM_JAX_DECODE_PADDED_GEMM_GATE_UP", bool),
    "decode_padded_gemm_rows": ("NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS", int),
    "decode_padded_gemm_max_out_dim": ("NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM", int),
    "full_attention_kv_cache_dtype": ("NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE", str),
    "gdn_disable_fallbacks": ("NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS", bool),
    "gdn_prefill_post_conv_impl": ("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", str),
    "gdn_prefill_qkv_dtype": ("NANO_VLLM_JAX_GDN_PREFILL_QKV_DTYPE", str),
    "gdn_prefill_post_conv_output_dtype": ("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE", str),
    "gdn_packed_decode_impl": ("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", str),
    "gdn_packed_decode_qkv_dtype": ("NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE", str),
    "gdn_packed_decode_pre_normalize_qk": ("NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK", bool),
    "gdn_packed_decode_max_batch": ("NANO_VLLM_JAX_GDN_PACKED_DECODE_MAX_BATCH", int),
    "skip_compile": ("NANO_VLLM_JAX_SKIP_COMPILE", bool),
}

# Non-NANO env vars that are accepted verbatim in the legacy ``env`` section.
_UNPREFIXED_ENV_KEYS: set[str] = {
    "TOKENIZERS_PARALLELISM",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "TF_GPU_ALLOCATOR",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PLATFORMS",
    "HF_HUB_OFFLINE",
    "XLA_FLAGS",
}

_DIRECT_ENV_PREFIXES = (
    _ENV_PREFIX,
    "JAX_",
    "XLA_",
    "TF_",
    "HF_",
    "TOKENIZERS_",
    "CUDA_",
    "NVIDIA_",
    "XDG_",
    "UV_",
    "PIP_",
)

_LOCAL_CUDA_PROBE_ENV_KEYS = {
    "NANO_VLLM_JAX_CUDA_FP32_KV_APPEND",
    "NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN",
    "NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE",
}

_LOCAL_CUDA_GDN_PREFILL_IMPLS = {
    "cuda_fla_chunk32_fp32",
    "cuda_prep_fp32",
    "cuda_prep_prefill_fp32",
}


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(value: Any) -> str:
    return "1" if _config_bool(value) else "0"


def _put_env(env: dict[str, str], key: str, value: Any) -> None:
    if value is not None:
        env[key] = _env_bool(value) if isinstance(value, bool) else str(value)


def _coerce(value: str, target_type: type) -> Any:
    if target_type is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    return target_type(value)


def _apply_env_overrides(section: dict, env_map: dict[str, tuple[str, type]]) -> dict:
    """Return a copy of *section* with env-var overrides applied."""
    result = dict(section)
    for key, (env_name, target_type) in env_map.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            result[key] = _coerce(env_value, target_type)
    return result


def _normalize_env_section_key(key: Any) -> str:
    """Normalize legacy ``env`` keys.

    Full env var names are kept verbatim. Short nano-vllm keys such as
    ``DEVICE_TOKEN_CARRY`` are expanded to ``NANO_VLLM_JAX_DEVICE_TOKEN_CARRY``.
    """

    normalized = str(key).strip()
    if normalized in _UNPREFIXED_ENV_KEYS or normalized.startswith(_DIRECT_ENV_PREFIXES):
        return normalized
    return f"{_ENV_PREFIX}{normalized}"


def _env_section_to_env(env_section: dict) -> dict[str, str]:
    """Return env vars from the legacy raw ``env`` config section."""
    env: dict[str, str] = {}
    for key, yaml_value in env_section.items():
        if yaml_value is not None:
            env[_normalize_env_section_key(key)] = str(yaml_value)
    return env


def _runtime_fastpaths_to_engine(runtime_section: dict) -> dict[str, Any]:
    """Project accepted serving fastpaths into engine config fields."""
    fastpaths = runtime_section.get("fastpaths", {}) or {}
    mapping = {
        "greedy_token": "greedy_token_fastpath",
        "device_token_carry": "device_token_carry",
        "static_decode_metadata": "static_decode_metadata",
        "static_decode_seq_lens_carry": "static_decode_seq_lens_carry",
        "greedy_decode_burst_steps": "greedy_decode_burst_steps",
        "decode_block_table_buckets": "decode_block_table_buckets",
        "materialize_tied_lm_head": "materialize_tied_lm_head",
        "compact_prefill_in_proj_qkv": "compact_prefill_in_proj_qkv",
        "compact_prefill_gdn_z": "compact_prefill_gdn_z",
        "compact_prefill_full_attn_proj": "compact_prefill_full_attn_proj",
        "compact_prefill_mlp": "compact_prefill_mlp",
        "compact_prefill_token_count_mode": "compact_prefill_token_count_mode",
        "lm_head_decode_act_dtype": "lm_head_decode_act_dtype",
        "decode_proj_act_dtype": "decode_proj_act_dtype",
        "decode_padded_gemm": "decode_padded_gemm",
        "decode_padded_gemm_gate_up": "decode_padded_gemm_gate_up",
        "decode_padded_gemm_rows": "decode_padded_gemm_rows",
        "decode_padded_gemm_max_out_dim": "decode_padded_gemm_max_out_dim",
    }
    return {
        engine_key: fastpaths[fastpath_key]
        for fastpath_key, engine_key in mapping.items()
        if fastpath_key in fastpaths and fastpaths[fastpath_key] is not None
    }


def _kernels_to_engine_config(kernels_section: dict) -> dict[str, Any]:
    """Project accepted kernel policy into engine/model config fields."""
    engine: dict[str, Any] = {}
    full_attention = kernels_section.get("full_attention", {}) or {}
    if full_attention.get("kv_cache_dtype") is not None:
        engine["full_attention_kv_cache_dtype"] = full_attention["kv_cache_dtype"]

    gdn = kernels_section.get("gdn", {}) or {}
    mapping = {
        "disable_fallbacks": "gdn_disable_fallbacks",
        "prefill_post_conv_impl": "gdn_prefill_post_conv_impl",
        "prefill_qkv_dtype": "gdn_prefill_qkv_dtype",
        "prefill_post_conv_output_dtype": "gdn_prefill_post_conv_output_dtype",
    }
    for source, target in mapping.items():
        if gdn.get(source) is not None:
            engine[target] = gdn[source]

    packed_decode = gdn.get("packed_decode", {}) or {}
    packed_mapping = {
        "impl": "gdn_packed_decode_impl",
        "qkv_dtype": "gdn_packed_decode_qkv_dtype",
        "pre_normalize_qk": "gdn_packed_decode_pre_normalize_qk",
        "max_batch": "gdn_packed_decode_max_batch",
    }
    for source, target in packed_mapping.items():
        if packed_decode.get(source) is not None:
            engine[target] = packed_decode[source]
    return engine


def _runtime_section_to_env(runtime_section: dict) -> dict[str, str]:
    env: dict[str, str] = {}
    _put_env(env, "JAX_PLATFORMS", runtime_section.get("platform"))
    _put_env(env, "NANO_VLLM_JAX_CACHE_ROOT", runtime_section.get("cache_root"))
    _put_env(
        env,
        "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
        runtime_section.get("compile_cache_dir"),
    )
    _put_env(
        env,
        "JAX_COMPILATION_CACHE_DIR",
        runtime_section.get("jax_compilation_cache_dir"),
    )
    _put_env(
        env,
        "TOKENIZERS_PARALLELISM",
        runtime_section.get("tokenizers_parallelism"),
    )

    xla = runtime_section.get("xla", {}) or {}
    _put_env(env, "XLA_PYTHON_CLIENT_PREALLOCATE", xla.get("preallocate"))
    _put_env(env, "TF_GPU_ALLOCATOR", xla.get("gpu_allocator"))
    _put_env(env, "NANO_VLLM_JAX_XLA_GPU_AUTOTUNE_LEVEL", xla.get("autotune_level"))
    _put_env(env, "XLA_FLAGS", xla.get("flags"))

    fastpaths = runtime_section.get("fastpaths", {}) or {}
    fastpath_map = {
        "greedy_token": "NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH",
        "materialize_tied_lm_head": "NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD",
        "compact_prefill_in_proj_qkv": "NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV",
        "compact_prefill_gdn_z": "NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z",
        "compact_prefill_full_attn_proj": "NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ",
        "compact_prefill_mlp": "NANO_VLLM_JAX_COMPACT_PREFILL_MLP",
        "compact_prefill_token_count_mode": "NANO_VLLM_JAX_COMPACT_PREFILL_TOKEN_COUNT_MODE",
        "device_token_carry": "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        "static_decode_metadata": "NANO_VLLM_JAX_STATIC_DECODE_METADATA",
        "static_decode_seq_lens_carry": "NANO_VLLM_JAX_STATIC_DECODE_SEQ_LENS_CARRY",
        "greedy_decode_burst_steps": "NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS",
        "trace_token_prefetch": "NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH",
        "lm_head_decode_act_dtype": "NANO_VLLM_JAX_LM_HEAD_DECODE_ACT_DTYPE",
        "decode_proj_act_dtype": "NANO_VLLM_JAX_DECODE_PROJ_ACT_DTYPE",
        "decode_padded_gemm": "NANO_VLLM_JAX_DECODE_PADDED_GEMM",
        "decode_padded_gemm_gate_up": "NANO_VLLM_JAX_DECODE_PADDED_GEMM_GATE_UP",
        "decode_padded_gemm_rows": "NANO_VLLM_JAX_DECODE_PADDED_GEMM_ROWS",
        "decode_padded_gemm_max_out_dim": "NANO_VLLM_JAX_DECODE_PADDED_GEMM_MAX_OUT_DIM",
        "pallas_decode_rmsnorm": "NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM",
        "triton_decode_rmsnorm": "NANO_VLLM_JAX_TRITON_DECODE_RMSNORM",
        "pallas_gdn_qk_prenorm": "NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM",
    }
    for key, env_key in fastpath_map.items():
        _put_env(env, env_key, fastpaths.get(key))
    return env


def _kernels_section_to_env(kernels_section: dict) -> dict[str, str]:
    env: dict[str, str] = {}
    _put_env(env, "NANO_VLLM_JAX_KERNEL_BACKEND", kernels_section.get("backend"))
    _put_env(
        env,
        "NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES",
        kernels_section.get("allow_local_cuda_probes"),
    )

    full_attention = kernels_section.get("full_attention", {}) or {}
    full_attention_map = {
        "nhd_full_attention_kv_cache": "NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE",
        "kv_cache_dtype": "NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE",
        "flashinfer_kv_append": "NANO_VLLM_JAX_FLASHINFER_KV_APPEND",
        "cuda_fp32_kv_append": "NANO_VLLM_JAX_CUDA_FP32_KV_APPEND",
        "cuda_fp32_decode_attention": "NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN",
    }
    for key, env_key in full_attention_map.items():
        _put_env(env, env_key, full_attention.get(key))

    gdn = kernels_section.get("gdn", {}) or {}
    _put_env(env, "NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE", gdn.get("cuda_fp32_decode"))
    _put_env(env, "NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS", gdn.get("disable_fallbacks"))
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
        gdn.get("prefill_post_conv_impl"),
    )
    _put_env(env, "NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE", gdn.get("prefill_act_dtype"))
    _put_env(env, "NANO_VLLM_JAX_GDN_PREFILL_QKV_DTYPE", gdn.get("prefill_qkv_dtype"))
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE",
        gdn.get("prefill_post_conv_output_dtype"),
    )
    prefill_block_dot = gdn.get("prefill_block_dot")
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT",
        prefill_block_dot if prefill_block_dot is not None else gdn.get("kkt_block_dot"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_FWD_O_BLOCK_DOT",
        prefill_block_dot if prefill_block_dot is not None else gdn.get("fwd_o_block_dot"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT",
        prefill_block_dot if prefill_block_dot is not None else gdn.get("delta_h_block_dot"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT",
        prefill_block_dot if prefill_block_dot is not None else gdn.get("recompute_block_dot"),
    )

    packed_decode = gdn.get("packed_decode", {}) or {}
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL",
        packed_decode.get("impl"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE",
        packed_decode.get("qkv_dtype"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK",
        packed_decode.get("pre_normalize_qk"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_MAX_BATCH",
        packed_decode.get("max_batch"),
    )

    triton_decode = packed_decode.get("triton", {}) or {}
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_WARPS",
        triton_decode.get("num_warps"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_STAGES",
        triton_decode.get("num_stages"),
    )
    _put_env(
        env,
        "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_BLOCK_V",
        triton_decode.get("block_v"),
    )
    return env


def _uses_local_cuda_probe(env: dict[str, str]) -> bool:
    for key in _LOCAL_CUDA_PROBE_ENV_KEYS:
        if _config_bool(env.get(key)):
            return True
    packed_decode = env.get("NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL", "").lower()
    if packed_decode == "cuda_fp32":
        return True
    prefill_impl = env.get("NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL", "").lower()
    return prefill_impl in _LOCAL_CUDA_GDN_PREFILL_IMPLS


def _validate_local_cuda_probe_policy(env: dict[str, str]) -> None:
    if not _uses_local_cuda_probe(env):
        return
    if _config_bool(env.get("NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES")):
        return
    raise ValueError(
        "Local CUDA/JAX FFI probes are disabled for runtime configs. "
        "Use Pallas/CuteDSL or borrowed/adapted Triton kernels for serving paths. "
        "Set kernels.allow_local_cuda_probes=true only for explicit diagnostic probes."
    )


def runtime_env_from_config(config: dict) -> dict[str, str]:
    """Translate typed ``runtime``/``kernels`` plus legacy ``env`` to env vars."""
    env: dict[str, str] = {}
    env.update(_runtime_section_to_env(config.get("runtime", {}) or {}))
    env.update(_kernels_section_to_env(config.get("kernels", {}) or {}))
    env.update(_env_section_to_env(config.get("env", {}) or {}))
    _validate_local_cuda_probe_policy(env)
    return env


def engine_overrides_from_config(config: dict) -> dict[str, Any]:
    """Return engine/config fields implied by typed runtime and kernel policy."""

    return {
        **_runtime_fastpaths_to_engine(config.get("runtime", {}) or {}),
        **_kernels_to_engine_config(config.get("kernels", {}) or {}),
    }


def _apply_runtime_env(env: dict[str, str]) -> None:
    """Set os.environ from resolved runtime config; existing env vars still win."""
    for key, value in env.items():
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ServerConfig:
    """Resolved server, engine, and runtime/kernel configuration.

    Attributes:
        server:  dict of Flask / network settings.
        engine:  dict of LLMEngine / scheduler settings.
        runtime: dict of typed process/JAX startup settings.
        kernels: dict of typed kernel policy settings.
        env:     dict of env vars derived from runtime/kernels/env sections.
        source:  Path to the loaded config file, or ``"<defaults>"``.
    """

    def __init__(
        self,
        server: dict,
        engine: dict,
        runtime: dict,
        kernels: dict,
        env: dict[str, str],
        source: str,
    ):
        self.server = server
        self.engine = engine
        self.runtime = runtime
        self.kernels = kernels
        self.env = env
        self.source = source

    def __repr__(self) -> str:
        return (
            f"ServerConfig(source={self.source!r}, "
            f"server={self.server}, engine={self.engine}, "
            f"runtime={self.runtime}, kernels={self.kernels})"
        )


def load_server_config(path: str | Path | None = None) -> ServerConfig:
    """Load server config from YAML, apply env overrides, return resolved config.

    If *path* is *None*, searches for the default config file.  If no config
    file is found, returns built-in defaults (still subject to env overrides).
    """
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            # Explicit path missing -> fall back to defaults (don't crash server startup)
            config_path = None
    else:
        config_path = _default_config_path()

    if config_path is not None:
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
        source = str(config_path)
    else:
        raw = {}
        source = "<defaults>"

    server_defaults: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 8080,
        "max_tokens_default": 16,
    }
    engine_defaults: dict[str, Any] = {
        "model": "Qwen/Qwen3.5-0.8B",
        "backend": "auto",
        "dtype": "float16",
        "weight_dtype": None,
        "jax_execution": "jit",
        "prefill_buckets": "16",
        "prefill_token_buckets": "16",
        "prefill_layout": "packed",
        "batch_size_buckets": "1",
        "max_prefill": 16,
        "max_kv_cache_mb": 64,
        "num_kvcache_blocks": 8,
        "max_num_seqs": 1,
        "max_num_resident_seqs": None,
        "max_num_batched_tokens": 16,
        "num_speculative_tokens": 0,
        "greedy_token_fastpath": True,
        "device_token_carry": False,
        "static_decode_metadata": False,
        "static_decode_seq_lens_carry": False,
        "greedy_decode_burst_steps": 1,
        "materialize_tied_lm_head": False,
        "compact_prefill_in_proj_qkv": False,
        "compact_prefill_gdn_z": False,
        "compact_prefill_full_attn_proj": False,
        "compact_prefill_mlp": False,
        "compact_prefill_token_count_mode": "exact",
        "lm_head_decode_act_dtype": "fp32",
        "decode_proj_act_dtype": "fp32",
        "decode_padded_gemm": False,
        "decode_padded_gemm_gate_up": False,
        "decode_padded_gemm_rows": 8,
        "decode_padded_gemm_max_out_dim": 8192,
        "full_attention_kv_cache_dtype": "default",
        "gdn_disable_fallbacks": False,
        "gdn_prefill_post_conv_impl": "off",
        "gdn_prefill_qkv_dtype": "fp32",
        "gdn_prefill_post_conv_output_dtype": "fp32",
        "gdn_packed_decode_impl": "off",
        "gdn_packed_decode_qkv_dtype": "fp32",
        "gdn_packed_decode_pre_normalize_qk": False,
        "gdn_packed_decode_max_batch": None,
        "skip_compile": False,
    }

    server = _apply_env_overrides({**server_defaults, **raw.get("server", {})}, _SERVER_ENV_MAP)
    runtime = dict(raw.get("runtime", {}) or {})
    explicit_engine = dict(raw.get("engine", {}) or {})
    engine_raw = {
        **engine_defaults,
        **engine_overrides_from_config(raw),
        **explicit_engine,
    }
    engine = _apply_env_overrides(engine_raw, _ENGINE_ENV_MAP)

    kernels = dict(raw.get("kernels", {}) or {})
    env = runtime_env_from_config(raw)
    _apply_runtime_env(env)

    return ServerConfig(
        server=server,
        engine=engine,
        runtime=runtime,
        kernels=kernels,
        env=env,
        source=source,
    )
