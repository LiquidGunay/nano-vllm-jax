from dataclasses import replace

from nanovllm_jax.config import (
    CapacitySpec,
    CompileSpec,
    ModelSpec,
    RuntimeSpec,
)
from nanovllm_jax.fastpath import KernelPlan
from nanovllm_jax.speculation import DrafterConfig


def qwen_text_config(size: str = "4B") -> dict[str, object]:
    variants = {
        "0.8B": (1024, 3584, 24, 8, 2, 16),
        "2B": (2048, 6144, 24, 8, 2, 16),
        "4B": (2560, 9216, 32, 16, 4, 32),
    }
    hidden, intermediate, layers, heads, kv_heads, linear_value_heads = variants[size]
    return {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": linear_value_heads,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "max_position_embeddings": 262144,
        "layer_types": [
            "linear_attention" if index % 4 != 3 else "full_attention"
            for index in range(layers)
        ],
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "attention_bias": False,
        "attn_output_gate": True,
        "mamba_ssm_dtype": "float32",
        "tie_word_embeddings": True,
        "eos_token_id": 248044,
        "mlp_only_layers": [],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    }


def runtime_spec(
    *,
    model=None,
    capacity=None,
    compile=None,
    kernels=None,
    drafter: DrafterConfig | None = None,
) -> RuntimeSpec:
    """Build an explicitly owned test runtime without production adapters."""

    return RuntimeSpec(
        model=replace(ModelSpec(), **(model or {})),
        capacity=replace(CapacitySpec(), **(capacity or {})),
        compile=replace(CompileSpec(), **(compile or {})),
        kernels=replace(KernelPlan(), **(kernels or {})),
        drafter=drafter,
    )
