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
    if status.available:
        assert status.selected == "gdn_fla"
        assert not status.fallback
        assert status.external_kernels_enabled
    else:
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
    assert "kv_append_paged_nhd" in statuses["cuda_fp32"].provided_kernels
    assert "paged_decode_attention_gqa_nhd" in statuses["cuda_fp32"].provided_kernels
    assert "gdn_recurrent_decode_step" in statuses["gdn_cuda"].provided_kernels
    assert "gdn_recurrent_decode_step" in statuses["gdn_fla"].provided_kernels
    assert "gdn_segmented_prefill_chunk32" in statuses["gdn_fla"].provided_kernels
    assert not statuses["flashinfer"].external_kernels_enabled
    assert not statuses["cuda_fp32"].external_kernels_enabled
    assert not statuses["gdn_cuda"].external_kernels_enabled
    if statuses["gdn_fla"].available:
        assert statuses["gdn_fla"].external_kernels_enabled
    else:
        assert not statuses["gdn_fla"].external_kernels_enabled


def test_explicit_unaccepted_kernel_backend_fails_strict():
    with pytest.raises(KernelBackendUnavailable):
        select_kernel_backend("flashinfer", strict=True)

    with pytest.raises(KernelBackendUnavailable):
        select_kernel_backend("cuda_fp32", strict=True)

    with pytest.raises(KernelBackendUnavailable):
        select_kernel_backend("gdn_cuda", strict=True)

    if list_kernel_backends()["gdn_fla"].available:
        select_kernel_backend("gdn_fla", strict=True)
    else:
        with pytest.raises(KernelBackendUnavailable):
            select_kernel_backend("gdn_fla", strict=True)


def test_kernel_registry_recognizes_fla_aliases():
    for alias in ("gdn_fla", "fla_gdn", "vllm_fla", "flash_linear_attention"):
        status = select_kernel_backend(alias)
        assert status.requested == "gdn_fla"
        if status.available:
            assert status.selected == "gdn_fla"
        else:
            assert status.selected == "pure_jax"
        assert "gdn_recurrent_decode_step" in status.provided_kernels
        assert status.external_kernels_enabled == status.available


def test_gpu_runtime_backend_keeps_auto_kernel_backend_on_pure_jax(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "auto")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    backend = select_backend("gpu")

    assert backend.name == "gpu"
    if backend.kernel_backend.available:
        assert backend.kernel_backend.selected == "gdn_fla"
        assert backend.kernel_backend.external_kernels_enabled
    else:
        assert backend.kernel_backend.selected == "pure_jax"
        assert not backend.kernel_backend.external_kernels_enabled


def test_gpu_runtime_backend_rejects_explicit_unaccepted_external_backend(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "flashinfer")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    with pytest.raises(KernelBackendUnavailable):
        select_backend("gpu")


def test_gpu_runtime_backend_rejects_explicit_unaccepted_gdn_fla(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_KERNEL_BACKEND", "gdn_fla")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    if list_kernel_backends()["gdn_fla"].available:
        backend = select_backend("gpu")
        assert backend.kernel_backend.selected == "gdn_fla"
        assert backend.kernel_backend.external_kernels_enabled
    else:
        with pytest.raises(KernelBackendUnavailable):
            select_backend("gpu")
