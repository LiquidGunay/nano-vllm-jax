"""Canonical serving implementation policy.

Owns:
    The promoted CUDA/JAX operation choices for main.
Receives:
    No user configuration and no environment variables.
Returns:
    One frozen kernel plan.
Invariant:
    Workload and capacity live in config; implementation choices live here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util


@dataclass(frozen=True)
class KernelPlan:
    """Operation implementations selected before model execution."""

    kv_cache_dtype: str = "default"

    full_attention_prefill: str = "reference"
    full_attention_decode: str = "reference"
    full_attention_kv_append: str = "reference"

    gdn_prefill: str = "off"
    gdn_prefill_qkv_dtype: str = "fp32"
    gdn_prefill_output_dtype: str = "fp32"
    gdn_decode: str = "off"
    gdn_decode_qkv_dtype: str = "fp32"
    gdn_disable_fallbacks: bool = False
    gdn_decode_max_batch: int | None = None
    gdn_recurrent_prefill_threshold: int = 8

    greedy_token_fastpath: bool = True
    sampled_token_fastpath: bool = True
    lm_head_greedy: str = "jax"
    lm_head_sampled: str = "jax"
    lm_head_decode_act_dtype: str = "fp32"

    device_token_carry: bool = False
    static_decode_metadata: bool = False
    static_decode_seq_lens_carry: bool = False
    resident_decode_metadata: bool = False

    compact_prefill_in_proj_qkv: bool = False
    compact_prefill_gdn_z: bool = False
    compact_prefill_full_attn_proj: bool = False
    compact_prefill_mlp: bool = False
    compact_prefill_token_count_mode: str = "exact"

    decode_proj_act_dtype: str = "fp32"
    decode_padded_gemm: bool = False
    decode_padded_gemm_gate_up: bool = False
    decode_rms_padded_gemm: bool = False
    decode_padded_gemm_rows: int = 8
    decode_padded_gemm_max_out_dim: int = 300000
    gdn_width1_packed_input_projection: bool = False
    greedy_decode_burst_steps: int = 1


KERNEL_PLAN = KernelPlan(
    kv_cache_dtype="bf16",
    full_attention_prefill="triton_packed",
    full_attention_decode="flashinfer_paged",
    gdn_prefill="triton_fla_padded",
    gdn_decode="reference",
    gdn_decode_qkv_dtype="bf16",
    gdn_disable_fallbacks=True,
    lm_head_greedy="triton",
    lm_head_decode_act_dtype="bf16",
    device_token_carry=True,
    static_decode_metadata=True,
    resident_decode_metadata=True,
    compact_prefill_in_proj_qkv=True,
    compact_prefill_gdn_z=True,
    compact_prefill_full_attn_proj=True,
    compact_prefill_mlp=True,
    compact_prefill_token_count_mode="bucket",
    decode_proj_act_dtype="bf16",
    decode_padded_gemm=True,
    decode_padded_gemm_gate_up=True,
    gdn_width1_packed_input_projection=True,
)


def as_manifest(plan: KernelPlan = KERNEL_PLAN) -> dict[str, object]:
    return asdict(plan)


def format_manifest(plan: KernelPlan = KERNEL_PLAN) -> str:
    pairs = as_manifest(plan)
    return "\n".join(f"{key}: {value}" for key, value in pairs.items())


def required_modules(plan: KernelPlan = KERNEL_PLAN) -> tuple[str, ...]:
    modules: set[str] = set()
    if plan.full_attention_prefill.startswith("triton"):
        modules.update({"triton", "jax_triton"})
    if plan.full_attention_decode.startswith("flashinfer"):
        modules.update({"flashinfer", "jax_tvm_ffi"})
    if plan.gdn_prefill.startswith("triton"):
        modules.update({"triton", "jax_triton"})
    if plan.lm_head_greedy.startswith("triton"):
        modules.update({"triton", "jax_triton"})
    return tuple(sorted(modules))


def missing_required_modules(plan: KernelPlan = KERNEL_PLAN) -> tuple[str, ...]:
    return tuple(
        module
        for module in required_modules(plan)
        if importlib.util.find_spec(module) is None
    )


def validate_runtime_dependencies(plan: KernelPlan = KERNEL_PLAN) -> None:
    missing = missing_required_modules(plan)
    if missing:
        raise RuntimeError(
            "Promoted fast path dependencies are missing: "
            + ", ".join(missing)
            + ". Install the serving extras with "
            "`pip install -e \".[cuda13,flashinfer-ffi,gdn-fla-triton]\"`."
        )
