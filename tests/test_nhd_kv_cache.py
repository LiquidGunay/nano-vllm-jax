"""NHD full-attention KV cache allocation tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import pytest

from nanovllm_jax.backends import PureJAXBackend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.kernels.flashinfer_ffi import kv_append_paged_nhd_reference
from nanovllm_jax.kv_cache import (
    KVCacheSpec,
    full_attention_nhd_kv_cache_shape,
    init_full_attention_nhd_kv_cache,
    update_kv_cache,
)
from nanovllm_jax.model import init_params


def _spec() -> KVCacheSpec:
    return KVCacheSpec(
        num_layers=24,
        num_blocks=8,
        block_size=16,
        num_kv_heads=2,
        head_dim=256,
        dtype=jnp.float32,
    )


def _tiny_full_attention_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        block_size=2,
        num_kvcache_blocks=4,
        dtype="float32",
        tie_word_embeddings=True,
        layer_types=("full_attention",),
        linear_attn_layers=(),
        max_num_seqs=1,
        max_blocks_per_seq=2,
        max_kv_cache_bytes=4 * 2 * 2 * 1 * 8 * 4 * 2,
    )


def test_nhd_full_attention_cache_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE", raising=False)

    backend = PureJAXBackend()
    cache = backend.allocate_full_attention_nhd_kv_cache(_spec(), full_attention_layers=(3, 7))

    assert cache is None


def test_nhd_full_attention_cache_shape_when_enabled(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE", "1")
    spec = _spec()

    backend = PureJAXBackend()
    nhd_cache = backend.allocate_full_attention_nhd_kv_cache(
        spec,
        full_attention_layers=(3, 7, 11, 15, 19, 23),
    )
    canonical_cache = backend.allocate_kv_cache(spec, max_seqs=4, max_blocks_per_seq=8)

    assert nhd_cache is not None
    assert nhd_cache.layout == "NHD"
    assert nhd_cache.page_size == spec.block_size
    assert nhd_cache.layer_indices == (3, 7, 11, 15, 19, 23)
    assert nhd_cache.k_cache.shape == (6, 8, 16, 2, 256)
    assert nhd_cache.v_cache.shape == (6, 8, 16, 2, 256)
    assert nhd_cache.k_cache.dtype == jnp.float32
    assert canonical_cache.k_cache.shape == (24, 8, 16, 2, 256)
    assert canonical_cache.v_cache.shape == (24, 8, 16, 2, 256)


def test_nhd_full_attention_cache_uses_main_cache_block_cap(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE", "1")
    spec = KVCacheSpec(
        num_layers=4,
        num_blocks=8,
        block_size=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=jnp.float32,
        max_kv_cache_bytes=2 * 4 * 2 * 1 * 4 * 4 * 2,
    )

    backend = PureJAXBackend()
    nhd_cache = backend.allocate_full_attention_nhd_kv_cache(
        spec,
        full_attention_layers=(1, 3),
    )
    canonical_cache = backend.allocate_kv_cache(spec, max_seqs=1, max_blocks_per_seq=2)

    assert nhd_cache is not None
    assert nhd_cache.k_cache.shape == (2, 2, 2, 1, 4)
    assert canonical_cache.k_cache.shape == (4, 2, 2, 1, 4)


def test_model_runner_sidecar_does_not_replace_canonical_cache(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_NHD_FULL_ATTN_KV_CACHE", "1")
    config = _tiny_full_attention_config()
    params = init_params(jax.random.PRNGKey(0), config)

    runner = ModelRunner(config, params, backend="auto")

    assert runner.cache_storage.k_cache.shape == (1, 4, 2, 1, 8)
    assert runner.cache_storage.v_cache.shape == (1, 4, 2, 1, 8)
    assert runner.full_attention_nhd_cache is not None
    assert runner.full_attention_nhd_cache.k_cache.shape == (1, 4, 2, 1, 8)
    assert runner.full_attention_nhd_cache.layer_indices == (0,)


def test_nhd_full_attention_shape_helper_does_not_allocate():
    shape = full_attention_nhd_kv_cache_shape(
        _spec(),
        full_attention_layers=(3, 7, 11, 15, 19, 23),
    )

    assert shape == (6, 8, 16, 2, 256)


def test_nhd_full_attention_cache_requires_full_attention_layers():
    with pytest.raises(ValueError, match="full_attention_layers"):
        init_full_attention_nhd_kv_cache(_spec(), full_attention_layers=())


def test_kv_append_paged_nhd_reference_matches_canonical_update():
    page_size = 4
    num_pages = 8
    num_kv_heads = 1
    head_dim = 2
    block_tables = jnp.array([[3, 1], [2, 4]], dtype=jnp.int32)
    positions = jnp.array([[0, 4], [1, 6]], dtype=jnp.int32)
    slot_mapping = block_tables[
        jnp.arange(2, dtype=jnp.int32)[:, None],
        positions // page_size,
    ] * page_size + (positions % page_size)
    new_k = jnp.arange(2 * 2 * num_kv_heads * head_dim, dtype=jnp.float32).reshape(
        2,
        2,
        num_kv_heads,
        head_dim,
    )
    new_v = new_k + 100.0
    canonical_k = jnp.zeros((1, num_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32)
    canonical_v = jnp.zeros_like(canonical_k)

    canonical_k, canonical_v = update_kv_cache(
        canonical_k,
        canonical_v,
        slot_mapping=slot_mapping,
        new_k=new_k,
        new_v=new_v,
        layer_idx=0,
    )
    nhd_k, nhd_v = kv_append_paged_nhd_reference(
        append_key=new_k.reshape(-1, num_kv_heads, head_dim),
        append_value=new_v.reshape(-1, num_kv_heads, head_dim),
        batch_indices=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        positions=positions.reshape(-1),
        k_cache=jnp.zeros((num_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32),
        v_cache=jnp.zeros((num_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32),
        kv_indices=jnp.array([3, 1, 2, 4], dtype=jnp.int32),
        kv_indptr=jnp.array([0, 2, 4], dtype=jnp.int32),
        kv_last_page_len=jnp.array([1, 3], dtype=jnp.int32),
    )

    assert nhd_k.shape == (num_pages, page_size, num_kv_heads, head_dim)
    assert nhd_v.shape == (num_pages, page_size, num_kv_heads, head_dim)
    assert jnp.array_equal(nhd_k, canonical_k[0])
    assert jnp.array_equal(nhd_v, canonical_v[0])
