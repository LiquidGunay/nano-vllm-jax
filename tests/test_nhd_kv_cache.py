"""Canonical NHD KV layout and ownership tests."""

import jax
import jax.numpy as jnp

from nanovllm_jax.cache import update_kv_cache
from nanovllm_jax.kernels.flashinfer_ffi import kv_append_paged_nhd_reference
from nanovllm_jax.model import init_params
from nanovllm_jax.mtp import init_mtp_params
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.speculation import DrafterConfig
from tests.runtime_specs import runtime_spec


def _tiny_full_attention_config(*, mtp: bool = False):
    return runtime_spec(
        model={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "tie_word_embeddings": True,
            "layer_types": ("full_attention",),
        },
        capacity={
            "block_size": 2,
            "num_kvcache_blocks": 4,
            "max_num_seqs": 1,
            "max_num_resident_seqs": 1,
            "max_blocks_per_seq": 2,
            "max_kv_cache_bytes": 2 * 4 * 2 * 1 * 8 * 4 * 2,
            "prefix_cache": False,
        },
        compile={
            "dtype": "float32",
            "execution": "jit",
            "prefill_token_buckets": (2,),
            "batch_size_buckets": (1,),
            "decode_block_table_buckets": (2,),
        },
        kernels={
            "full_attention_decode": "flashinfer_paged",
            "device_token_carry": True,
            "static_decode_metadata": True,
            "resident_decode_metadata": True,
        },
        drafter=DrafterConfig.mtp(1) if mtp else None,
    )


def test_runner_has_one_capped_physical_owner_per_kv_layer():
    config = _tiny_full_attention_config(mtp=True)
    params = init_params(jax.random.PRNGKey(0), config.model)
    mtp_params = init_mtp_params(jax.random.PRNGKey(1), config)

    runner = ModelRunner(config, params, mtp_params=mtp_params)

    allocations = runner.persistent_kv_bytes()
    assert tuple(allocations) == ("target_kv", "predictor_kv")
    assert sum(allocations.values()) == config.capacity.max_kv_cache_bytes
    assert runner.cache_storage.k_cache.shape == (1, 4, 2, 1, 8)
    assert runner.mtp_state.cache_storage.k_cache.shape == (1, 4, 2, 1, 8)
    assert "full_attention_kv" not in runner.memory_bytes()


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
    canonical_k = jnp.zeros(
        (1, num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    )
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
        k_cache=jnp.zeros(
            (num_pages, page_size, num_kv_heads, head_dim),
            dtype=jnp.float32,
        ),
        v_cache=jnp.zeros(
            (num_pages, page_size, num_kv_heads, head_dim),
            dtype=jnp.float32,
        ),
        kv_indices=jnp.array([3, 1, 2, 4], dtype=jnp.int32),
        kv_indptr=jnp.array([0, 2, 4], dtype=jnp.int32),
        kv_last_page_len=jnp.array([1, 3], dtype=jnp.int32),
    )

    assert jnp.array_equal(nhd_k, canonical_k[0])
    assert jnp.array_equal(nhd_v, canonical_v[0])
