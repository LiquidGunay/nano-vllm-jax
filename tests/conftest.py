"""Shared test isolation for runtime environment overrides."""

from __future__ import annotations

import os

import pytest


_ENV_NAMES = {
    "NANO_VLLM_JAX_SERVER_CONFIG",
    "NANO_VLLM_JAX_CACHE_ROOT",
    "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
}

_ENV_PREFIXES = (
    "JAX_",
    "XLA_",
    "TF_",
    "TOKENIZERS_",
    "FLASHINFER_",
    "TRITON_",
)


@pytest.fixture(autouse=True)
def restore_process_env():
    before = {
        key: value
        for key, value in os.environ.items()
        if key in _ENV_NAMES or key.startswith(_ENV_PREFIXES)
    }
    yield
    for key in list(os.environ):
        if (key in _ENV_NAMES or key.startswith(_ENV_PREFIXES)) and key not in before:
            os.environ.pop(key, None)
    for key, value in before.items():
        os.environ[key] = value
