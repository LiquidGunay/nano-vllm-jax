"""LM-head projection, greedy top-1, and temperature sampling."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nanovllm_jax.layers import rms_norm
from nanovllm_jax.projection import (
    _can_use_decode_padded_gemm,
    _decode_padded_gemm_dot,
    _force_width1_decode_math,
    _lm_head_decode_activation_dtype,
    _lm_head_greedy_top1_impl,
    _lm_head_topk_impl,
    _tokenwise_decode_dot,
)

def _lm_head_normed_hidden_and_weight(
    hidden: jnp.ndarray,
    params: Any,
    config,
    *,
    hidden_is_normed: bool = False,
    is_prefill: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if hidden_is_normed:
        hidden_norm = hidden
    else:
        if not is_prefill:
            from nanovllm_jax.kernels.decode_reductions import (
                decode_rms_norm,
                lowered_decode_rms_norm_enabled,
            )

            hidden_norm = (
                decode_rms_norm(hidden, params.norm_weight, config.rms_norm_eps)
                if lowered_decode_rms_norm_enabled()
                else rms_norm(hidden, params.norm_weight, config.rms_norm_eps)
            )
        else:
            hidden_norm = rms_norm(hidden, params.norm_weight, config.rms_norm_eps)
    hidden_norm = hidden_norm.astype(
        _lm_head_decode_activation_dtype(config) if not is_prefill else jnp.float32
    )
    output_weight = params.lm_head if params.lm_head is not None else params.embed_tokens.T
    return hidden_norm, output_weight


def _lm_head_logits_from_normed(
    hidden_norm: jnp.ndarray,
    output_weight: jnp.ndarray,
    config,
    *,
    is_prefill: bool = True,
):
    if _can_use_decode_padded_gemm(hidden_norm, output_weight, config):
        logits = _decode_padded_gemm_dot(hidden_norm, output_weight, config)
    else:
        logits = _tokenwise_decode_dot(
            hidden_norm,
            output_weight,
            force_width1=(not is_prefill) and hidden_norm.ndim == 3 and hidden_norm.shape[1] > 1 and _force_width1_decode_math(),
        )
    return logits


def _lm_head_logits(
    hidden: jnp.ndarray,
    params: Any,
    config,
    *,
    hidden_is_normed: bool = False,
    is_prefill: bool = True,
):
    hidden_norm, output_weight = _lm_head_normed_hidden_and_weight(
        hidden,
        params,
        config,
        hidden_is_normed=hidden_is_normed,
        is_prefill=is_prefill,
    )
    return _lm_head_logits_from_normed(
        hidden_norm,
        output_weight,
        config,
        is_prefill=is_prefill,
    )


def _lm_head_greedy_top1_token_ids(
    hidden_norm: jnp.ndarray,
    output_weight: jnp.ndarray,
    config,
) -> jnp.ndarray:
    impl = _lm_head_greedy_top1_impl(config)
    if impl == "jax":
        logits = _lm_head_logits_from_normed(
            hidden_norm,
            output_weight,
            config,
            is_prefill=False,
        )
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if impl == "triton":
        from nanovllm_jax.kernels.lm_head_triton import lm_head_greedy_top1_triton

        return lm_head_greedy_top1_triton(hidden_norm, output_weight)
    raise AssertionError(f"unexpected LM-head greedy top1 impl: {impl!r}")


def lm_head_token_ids_and_topk(
    hidden: jnp.ndarray,
    params: Any,
    config,
    *,
    hidden_is_normed: bool = False,
    is_prefill: bool = True,
    top_k: int = 0,
):
    """Return greedy LM-head token ids and optional top-k values on device.

    Decode needs exact target token ids and sometimes a top-k summary, but
    returning full `[B, width, vocab]` logits from the JIT is unnecessarily
    expensive. Keep the dense LM-head computation inside the compiled graph
    and return only small products.
    """
    if (
        (not is_prefill)
        and top_k == 0
        and hidden.ndim == 3
        and _lm_head_greedy_top1_impl(config) != "jax"
    ):
        hidden_norm, output_weight = _lm_head_normed_hidden_and_weight(
            hidden,
            params,
            config,
            hidden_is_normed=hidden_is_normed,
            is_prefill=False,
        )
        batch, width, hidden_dim = hidden_norm.shape
        flat_hidden = hidden_norm.reshape(batch * width, 1, hidden_dim)
        token_ids = _lm_head_greedy_top1_token_ids(flat_hidden, output_weight, config)
        return token_ids.reshape(batch, width), None, None

    logits = _lm_head_logits(
        hidden,
        params,
        config,
        hidden_is_normed=hidden_is_normed,
        is_prefill=is_prefill,
    )
    token_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if top_k > 0:
        top_values, top_indices = jax.lax.top_k(logits.astype(jnp.float32), top_k)
        return token_ids, top_values, top_indices.astype(jnp.int32)
    return token_ids, None, None


def lm_head_sample_token_ids(
    hidden: jnp.ndarray,
    params: Any,
    config,
    *,
    temperatures: jnp.ndarray,
    rng_keys: jnp.ndarray,
    hidden_is_normed: bool = False,
    is_prefill: bool = True,
) -> jnp.ndarray:
    """Sample token ids from LM-head logits without returning full logits.

    This intentionally handles the hot full-vocab temperature-sampling case.
    Top-k/top-p filtering should be provided by a dedicated sampler kernel
    before this becomes the general sampling boundary.
    """
    logits = _lm_head_logits(
        hidden,
        params,
        config,
        hidden_is_normed=hidden_is_normed,
        is_prefill=is_prefill,
    )
    logits = logits.astype(jnp.float32)
    row_logits = logits[:, 0, :] if logits.ndim == 3 else logits
    temperatures = temperatures.astype(jnp.float32)

    def sample_one(key, logit, temperature):
        def greedy(_):
            return jnp.argmax(logit).astype(jnp.int32)

        def sample(_):
            scaled = logit / jnp.maximum(temperature, jnp.asarray(1e-6, dtype=jnp.float32))
            return jax.random.categorical(key, scaled, axis=-1).astype(jnp.int32)

        return jax.lax.cond(temperature <= 0.0, greedy, sample, operand=None)

    return jax.vmap(sample_one)(rng_keys, row_logits, temperatures)
