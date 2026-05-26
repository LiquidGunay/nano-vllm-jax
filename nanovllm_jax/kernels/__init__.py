"""Optional serving-kernel registry.

The pure-JAX backend remains the correctness path. Modules in this package only
describe and gate optional external kernels until each one passes the roadmap's
correctness and integrated-performance gates.
"""

from nanovllm_jax.kernels.registry import (
    KernelBackendStatus,
    KernelBackendUnavailable,
    backend_status,
    list_kernel_backends,
    select_kernel_backend,
)

__all__ = [
    "KernelBackendStatus",
    "KernelBackendUnavailable",
    "backend_status",
    "list_kernel_backends",
    "select_kernel_backend",
]
