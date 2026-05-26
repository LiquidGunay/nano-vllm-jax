"""Optional kernel registry tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import pytest

from nanovllm_jax.backends import select_backend
from nanovllm_jax.kernels.registry import (
    KernelBackendUnavailable,
    list_kernel_backends,
    select_kernel_backend,
)


def test_kernel_registry_auto_keeps_pure_jax_until_kernels_are_accepted(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "auto")

    status = select_kernel_backend()

    assert status.requested == "auto"
    assert status.selected == "pure_jax"
    assert status.fallback == "pure_jax"
    assert not status.external_kernels_enabled


def test_runtime_auto_rejects_explicit_unaccepted_external_backend(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "gdn_cuda")

    with pytest.raises(KernelBackendUnavailable):
        select_backend("auto")


def test_kernel_registry_records_planned_external_backends():
    statuses = list_kernel_backends()

    assert statuses["pure_jax"].available
    assert statuses["pure_jax"].implemented
    assert "kv_append_paged_nhd" in statuses["flashinfer"].provided_kernels
    assert "paged_decode_attention_gqa_nhd" in statuses["flashinfer"].provided_kernels
    assert "gdn_recurrent_decode_step" in statuses["gdn_cuda"].provided_kernels
    assert not statuses["flashinfer"].external_kernels_enabled
    assert not statuses["gdn_cuda"].external_kernels_enabled


def test_explicit_unaccepted_kernel_backend_fails_strict():
    with pytest.raises(KernelBackendUnavailable):
        select_kernel_backend("flashinfer", strict=True)

    with pytest.raises(KernelBackendUnavailable):
        select_kernel_backend("gdn_cuda", strict=True)


def test_gpu_runtime_backend_keeps_auto_kernel_backend_on_pure_jax(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "auto")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    backend = select_backend("gpu")

    assert backend.name == "gpu"
    assert backend.kernel_backend.selected == "pure_jax"
    assert not backend.kernel_backend.external_kernels_enabled


def test_gpu_runtime_backend_rejects_explicit_unaccepted_external_backend(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "flashinfer")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    with pytest.raises(KernelBackendUnavailable):
        select_backend("gpu")
