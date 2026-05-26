"""Registry for optional GPU serving kernels.

This module deliberately probes dependency availability without importing the
heavy optional packages. Kernel call sites should use the returned status to
decide whether an external backend is enabled, and must keep a pure-JAX fallback.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass


class KernelBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested optional kernel backend cannot run."""


@dataclass(frozen=True)
class KernelBackendSpec:
    name: str
    aliases: tuple[str, ...]
    required_modules: tuple[str, ...]
    provided_kernels: tuple[str, ...]
    implemented: bool
    description: str


@dataclass(frozen=True)
class KernelBackendStatus:
    requested: str
    selected: str
    available: bool
    implemented: bool
    external_kernels_enabled: bool
    missing_modules: tuple[str, ...] = ()
    provided_kernels: tuple[str, ...] = ()
    fallback: str | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "available": self.available,
            "implemented": self.implemented,
            "external_kernels_enabled": self.external_kernels_enabled,
            "missing_modules": list(self.missing_modules),
            "provided_kernels": list(self.provided_kernels),
            "fallback": self.fallback,
            "reason": self.reason,
        }


_SPECS: dict[str, KernelBackendSpec] = {
    "pure_jax": KernelBackendSpec(
        name="pure_jax",
        aliases=("pure_jax", "jax", "none", "off"),
        required_modules=(),
        provided_kernels=(),
        implemented=True,
        description="Reference implementation using ordinary JAX/XLA operations.",
    ),
    "flashinfer": KernelBackendSpec(
        name="flashinfer",
        aliases=("flashinfer", "flashinfer_ffi", "jax_tvm_ffi"),
        required_modules=("flashinfer", "jax_tvm_ffi"),
        provided_kernels=(
            "kv_append_paged_nhd",
            "paged_decode_attention_gqa_nhd",
            "paged_prefill_attention_gqa_nhd",
        ),
        implemented=False,
        description="FlashInfer kernels called from JAX through jax-tvm-ffi.",
    ),
    "gdn_cuda": KernelBackendSpec(
        name="gdn_cuda",
        aliases=("gdn_cuda", "cuda_gdn"),
        required_modules=(),
        provided_kernels=(
            "gdn_recurrent_decode_step",
            "gdn_segmented_prefill_chunk32",
        ),
        implemented=False,
        description="Future CUDA/ported Gated DeltaNet kernels.",
    ),
}

_ALIASES = {
    alias: name
    for name, spec in _SPECS.items()
    for alias in spec.aliases
}


def _normalize(name: str | None) -> str:
    value = (name or os.environ.get("NANO_VLLM_JAX_KERNEL_BACKEND") or "pure_jax").strip().lower()
    if value == "auto":
        return "auto"
    try:
        return _ALIASES[value]
    except KeyError as exc:
        valid = ", ".join(sorted([*_ALIASES, "auto"]))
        raise ValueError(f"Unknown kernel backend {name!r}; expected one of: {valid}") from exc


def _missing_modules(required_modules: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(module for module in required_modules if importlib.util.find_spec(module) is None)


def _status_for_spec(requested: str, spec: KernelBackendSpec) -> KernelBackendStatus:
    missing = _missing_modules(spec.required_modules)
    available = not missing
    if spec.name == "pure_jax":
        return KernelBackendStatus(
            requested=requested,
            selected="pure_jax",
            available=True,
            implemented=True,
            external_kernels_enabled=False,
            reason="pure-JAX correctness backend selected",
        )
    if available and spec.implemented:
        return KernelBackendStatus(
            requested=requested,
            selected=spec.name,
            available=True,
            implemented=True,
            external_kernels_enabled=True,
            provided_kernels=spec.provided_kernels,
            reason=f"{spec.name} backend selected",
        )
    if not available:
        reason = f"{spec.name} backend unavailable; missing optional modules: {', '.join(missing)}"
    else:
        reason = f"{spec.name} backend is available but not implemented or accepted yet"
    return KernelBackendStatus(
        requested=requested,
        selected="pure_jax",
        available=available,
        implemented=spec.implemented,
        external_kernels_enabled=False,
        missing_modules=missing,
        provided_kernels=spec.provided_kernels,
        fallback="pure_jax",
        reason=reason,
    )


def backend_status(name: str) -> KernelBackendStatus:
    """Return availability status for a concrete backend name."""

    normalized = _normalize(name)
    if normalized == "auto":
        return select_kernel_backend("auto")
    return _status_for_spec(normalized, _SPECS[normalized])


def select_kernel_backend(name: str | None = None, *, strict: bool = False) -> KernelBackendStatus:
    """Resolve the requested optional kernel backend.

    `auto` intentionally resolves to `pure_jax` until an external backend has an
    implemented and accepted kernel path. Explicit external requests can set
    `strict=True` to fail instead of silently falling back.
    """

    normalized = _normalize(name)
    if normalized == "auto":
        for candidate in ("flashinfer", "gdn_cuda"):
            status = _status_for_spec("auto", _SPECS[candidate])
            if status.external_kernels_enabled:
                return status
        return KernelBackendStatus(
            requested="auto",
            selected="pure_jax",
            available=True,
            implemented=True,
            external_kernels_enabled=False,
            fallback="pure_jax",
            reason="auto resolved to pure_jax because no external backend has passed acceptance gates",
        )

    status = _status_for_spec(normalized, _SPECS[normalized])
    if strict and normalized != "pure_jax" and not status.external_kernels_enabled:
        raise KernelBackendUnavailable(status.reason)
    return status


def list_kernel_backends() -> dict[str, KernelBackendStatus]:
    """Return status for all registered concrete kernel backends."""

    return {name: backend_status(name) for name in _SPECS}


def kernel_backend_is_enabled(name: str | None = None) -> bool:
    """Convenience predicate for guarded call sites."""

    return select_kernel_backend(name).external_kernels_enabled
