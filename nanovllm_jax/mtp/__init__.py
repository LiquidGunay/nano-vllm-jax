"""MTP (Multi-Token Prediction) module for Qwen3.5 speculative decoding."""

from nanovllm_jax.mtp.mtp_layer import (
    MTPParams,
    MTPConfig,
    init_mtp_params,
    mtp_forward,
)
from nanovllm_jax.mtp.speculative import (
    SpeculativeState,
    generate_draft_tokens,
    verify_draft_tokens,
    apply_acceptance,
)

__all__ = [
    "MTPParams",
    "MTPConfig",
    "init_mtp_params",
    "mtp_forward",
    "SpeculativeState",
    "generate_draft_tokens",
    "verify_draft_tokens",
    "apply_acceptance",
]
