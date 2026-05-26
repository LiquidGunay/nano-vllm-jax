"""Runtime path defaults for local benchmark/server entry points."""

from __future__ import annotations

import os
from pathlib import Path


def default_runtime_root() -> Path:
    configured = os.getenv("NANO_VLLM_JAX_CACHE_ROOT")
    if configured:
        return Path(configured)
    mountpoint = Path("/mountpoint/.exp")
    if mountpoint.exists():
        return mountpoint
    mountpath = Path("/mountpath")
    if mountpath.exists():
        return mountpath
    return Path.cwd()


def configure_compilation_cache() -> str:
    cache_dir = Path(
        os.getenv("NANO_VLLM_JAX_COMPILE_CACHE_DIR")
        or os.getenv("JAX_COMPILATION_CACHE_DIR")
        or (default_runtime_root() / ".cache" / "jax")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NANO_VLLM_JAX_COMPILE_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    return str(cache_dir)


def configure_flashinfer_cache() -> str:
    """Set FlashInfer JIT/cache roots under the runtime root."""

    root = default_runtime_root()
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(root))
    cubin_dir = root / ".cache" / "flashinfer" / "cubins"
    os.environ.setdefault("FLASHINFER_CUBIN_DIR", str(cubin_dir))
    Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FLASHINFER_CUBIN_DIR"]).mkdir(parents=True, exist_ok=True)
    return os.environ["FLASHINFER_WORKSPACE_BASE"]


def configure_xla_flags(default_gpu_autotune_level: int = 4) -> str:
    """Set GPU-oriented XLA defaults while preserving explicit user flags."""
    if "XLA_FLAGS" not in os.environ:
        autotune_level = os.getenv(
            "NANO_VLLM_JAX_XLA_GPU_AUTOTUNE_LEVEL",
            str(default_gpu_autotune_level),
        )
        os.environ["XLA_FLAGS"] = f"--xla_gpu_autotune_level={autotune_level}"
    return os.environ["XLA_FLAGS"]
