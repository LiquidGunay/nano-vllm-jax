import jax
import jax.numpy as jnp
import sys
import types

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.mtp.mtp_layer import (
    _mtp_forward_hidden,
    _mtp_greedy_top1_token_ids,
    init_mtp_params,
    mtp_forward,
    mtp_forward_token_ids,
)


def _tiny_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        block_size=8,
        num_kvcache_blocks=4,
        dtype="float32",
        layer_types=("full_attention",),
        mtp_num_hidden_layers=1,
    )


def test_mtp_forward_returns_raw_hidden_for_chaining():
    config = _tiny_config()
    params = init_mtp_params(jax.random.PRNGKey(0), config)
    embed_tokens = jax.random.normal(jax.random.PRNGKey(1), (config.vocab_size, config.hidden_size))
    hidden = jax.random.normal(jax.random.PRNGKey(2), (2, 1, config.hidden_size))
    token_ids = jnp.array([[3], [7]], dtype=jnp.int32)
    positions = jnp.array([[4], [11]], dtype=jnp.int32)

    normed_hidden, raw_hidden = _mtp_forward_hidden(
        hidden_state=hidden,
        next_token_ids=token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )
    logits, chained_hidden = mtp_forward(
        hidden_state=hidden,
        next_token_ids=token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )
    draft_tokens, chained_hidden_from_top1 = mtp_forward_token_ids(
        hidden_state=hidden,
        next_token_ids=token_ids,
        embed_tokens=embed_tokens,
        params=params,
        config=config,
        positions=positions,
    )

    assert logits.shape == (2, 1, config.vocab_size)
    assert draft_tokens.shape == (2, 1)
    assert jnp.allclose(chained_hidden, raw_hidden)
    assert jnp.allclose(chained_hidden_from_top1, raw_hidden)
    assert not jnp.allclose(chained_hidden, normed_hidden)


def test_mtp_triton_top1_casts_hidden_to_weight_dtype(monkeypatch):
    seen = {}

    def fake_top1(hidden_norm, output_weight):
        seen["hidden_dtype"] = hidden_norm.dtype
        seen["weight_dtype"] = output_weight.dtype
        return jnp.zeros((hidden_norm.shape[0], hidden_norm.shape[1]), dtype=jnp.int32)

    monkeypatch.setitem(
        sys.modules,
        "nanovllm_jax.kernels.lm_head_triton",
        types.SimpleNamespace(lm_head_greedy_top1_triton=fake_top1),
    )
    config = types.SimpleNamespace(mtp_lm_head_greedy_top1_impl="triton")
    hidden = jnp.ones((2, 1, 8), dtype=jnp.float32)
    weight = jnp.ones((8, 16), dtype=jnp.bfloat16)

    token_ids = _mtp_greedy_top1_token_ids(hidden, weight, config)

    assert token_ids.shape == (2, 1)
    assert seen == {
        "hidden_dtype": jnp.dtype(jnp.bfloat16),
        "weight_dtype": jnp.dtype(jnp.bfloat16),
    }
