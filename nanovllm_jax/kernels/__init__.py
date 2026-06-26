"""Small helpers shared by optional serving-kernel wrappers."""

from __future__ import annotations

import importlib.util


class KernelUnavailable(RuntimeError):
    """Raised when a promoted optional kernel dependency cannot run."""


def missing_modules(modules: tuple[str, ...]) -> tuple[str, ...]:
    """Return optional Python modules that are not importable."""

    return tuple(module for module in modules if importlib.util.find_spec(module) is None)


def require_modules(modules: tuple[str, ...], feature: str) -> None:
    missing = missing_modules(modules)
    if missing:
        raise KernelUnavailable(
            f"{feature} requires optional modules: {', '.join(missing)}"
        )


__all__ = ["KernelUnavailable", "missing_modules", "require_modules"]
