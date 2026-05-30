"""YAML-based server configuration with environment-variable overrides.

Loads ``server_config.yaml`` (or a path given by ``NANO_VLLM_JAX_SERVER_CONFIG``)
at startup.  Every key can be overridden by an environment variable of the same
name (upper-cased, ``NANO_VLLM_JAX_`` prefix).  Env vars always win, preserving
backward compatibility.

The config is split into two sections:

* ``server`` – Flask / network knobs (host, port, max_tokens_default).
* ``engine`` – LLMEngine / scheduler knobs (model, dtype, buckets, KV cache …).
* ``env``    – Runtime environment knobs that were previously only set via
               ``os.environ`` (XLA flags, kernel backend, MTP policy, etc.).
               These are applied to ``os.environ`` *before* JAX / XLA init so
               that the existing ``os.getenv`` call-sites continue to work
               unchanged.
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
    "batch_size_buckets": ("NANO_VLLM_JAX_BATCH_SIZE_BUCKETS", str),
    "max_prefill": ("NANO_VLLM_JAX_MAX_PREFILL", int),
    "max_kv_cache_mb": ("NANO_VLLM_JAX_MAX_KV_CACHE_MB", int),
    "num_kvcache_blocks": ("NANO_VLLM_JAX_NUM_KVCACHE_BLOCKS", int),
    "max_num_seqs": ("NANO_VLLM_JAX_MAX_NUM_SEQS", int),
    "max_num_batched_tokens": ("NANO_VLLM_JAX_MAX_NUM_BATCHED_TOKENS", int),
    "num_speculative_tokens": ("NANO_VLLM_JAX_NUM_SPECULATIVE_TOKENS", int),
    "skip_compile": ("NANO_VLLM_JAX_SKIP_COMPILE", bool),
}

# Env-section keys: these are applied to os.environ before JAX init.
# The YAML key is the env-var name (without the NANO_VLLM_JAX_ prefix for
# brevity, but the full env-var name is also accepted).
_ENV_SECTION_KEYS: list[str] = [
    "TOKENIZERS_PARALLELISM",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "TF_GPU_ALLOCATOR",
    "NANO_VLLM_JAX_XLA_GPU_AUTOTUNE_LEVEL",
    "NANO_VLLM_JAX_CACHE_ROOT",
    "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
    "JAX_COMPILATION_CACHE_DIR",
    "NANO_VLLM_JAX_KERNEL_BACKEND",
    "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
    "NANO_VLLM_JAX_OVERLAPPED_STREAMING_TOKEN_PREFETCH",
    "NANO_VLLM_JAX_OFFLINE_STREAMING_TOKEN_EVENTS",
    "NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE",
    "NANO_VLLM_JAX_FLASHINFER_KV_APPEND",
    "NANO_VLLM_JAX_CUDA_FP32_KV_APPEND",
    "NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN",
    "NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE",
    "NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL",
    "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
    "NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE",
    "NANO_VLLM_JAX_GDN_PREFILL_QKV_DTYPE",
    "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_OUTPUT_DTYPE",
    "NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE",
    "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_WARPS",
    "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_STAGES",
    "NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_BLOCK_V",
    "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY",
    "NANO_VLLM_JAX_MTP_COMMIT_SELECT",
    "NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1",
    "NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1",
    "NANO_VLLM_JAX_MTP_ENABLE_REUSE_FALLBACK",
    "NANO_VLLM_JAX_MTP_CHECK_NEXT_STEP_SANITY",
    "NANO_VLLM_JAX_EXEC_LOG_STEPS",
    "NANO_VLLM_JAX_FORCE_CUDA_FFI_REBUILD",
    "NANO_VLLM_JAX_NVCC",
    "NANO_VLLM_JAX_CUDA_ARCH",
    "NANO_VLLM_JAX_PROFILE",
    "NANO_VLLM_JAX_JAX_PLATFORMS",
    "NANO_VLLM_JAX_PLATFORMS",
    "HF_HUB_OFFLINE",
    "XLA_FLAGS",
]


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


def _apply_env_section(env_section: dict) -> None:
    """Set os.environ from the ``env`` config section (env vars still win)."""
    for key in _ENV_SECTION_KEYS:
        # Accept both short key (e.g. "TOKENIZERS_PARALLELISM") and full
        # env-var name in the YAML.
        yaml_value = env_section.get(key)
        if yaml_value is None:
            # Try with NANO_VLLM_JAX_ prefix stripped for short-form keys
            if key.startswith(_ENV_PREFIX):
                short = key[len(_ENV_PREFIX):]
                yaml_value = env_section.get(short)
        if yaml_value is not None:
            # Env var takes precedence over config file
            os.environ.setdefault(key, str(yaml_value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ServerConfig:
    """Resolved server + engine + env configuration.

    Attributes:
        server:  dict of Flask / network settings.
        engine:  dict of LLMEngine / scheduler settings.
        source:  Path to the loaded config file, or ``"<defaults>"``.
    """

    def __init__(self, server: dict, engine: dict, source: str):
        self.server = server
        self.engine = engine
        self.source = source

    def __repr__(self) -> str:
        return (
            f"ServerConfig(source={self.source!r}, "
            f"server={self.server}, engine={self.engine})"
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
        "batch_size_buckets": "1",
        "max_prefill": 16,
        "max_kv_cache_mb": 64,
        "num_kvcache_blocks": 8,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 16,
        "num_speculative_tokens": 0,
        "skip_compile": False,
    }

    server = _apply_env_overrides({**server_defaults, **raw.get("server", {})}, _SERVER_ENV_MAP)
    engine = _apply_env_overrides({**engine_defaults, **raw.get("engine", {})}, _ENGINE_ENV_MAP)

    # Apply env-section *before* returning so callers can run JAX init after.
    env_section = raw.get("env", {})
    if env_section:
        _apply_env_section(env_section)

    return ServerConfig(server=server, engine=engine, source=source)
