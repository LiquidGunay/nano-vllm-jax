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
