"""Canonical serving implementation policy.

Owns:
    The promoted CUDA/JAX operation choices for main.
Receives:
    No user configuration and no environment variables.
Returns:
    A frozen manifest plus engine/runtime projections.
Invariant:
    Workload and capacity live in config; implementation choices live here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util


@dataclass(frozen=True)
class FastPath:
    compute_dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    kv_cache_dtype: str = "bf16"
    execution: str = "jit"
    prefill_layout: str = "packed"

    full_attention_prefill: str = "triton_packed"
    full_attention_decode: str = "flashinfer_paged"
    full_attention_kv_append: str = "reference"

    gdn_prefill: str = "triton_fla_padded"
    gdn_prefill_qkv_dtype: str = "fp32"
    gdn_prefill_output_dtype: str = "fp32"
    gdn_decode: str = "reference"
    gdn_decode_qkv_dtype: str = "bf16"
    gdn_disable_fallbacks: bool = True

    greedy_token_fastpath: bool = True
    sampled_token_fastpath: bool = True
    lm_head_greedy: str = "triton"
    lm_head_sampled: str = "jax"
    lm_head_decode_act_dtype: str = "bf16"

    device_token_carry: bool = True
    static_decode_metadata: bool = True
    static_decode_seq_lens_carry: bool = False
    resident_decode_metadata: bool = True

    materialize_tied_lm_head: bool = True
    compact_prefill_in_proj_qkv: bool = True
    compact_prefill_gdn_z: bool = True
    compact_prefill_full_attn_proj: bool = True
    compact_prefill_mlp: bool = True
    compact_prefill_token_count_mode: str = "bucket"

    decode_proj_act_dtype: str = "bf16"
    decode_padded_gemm: bool = True
    decode_padded_gemm_gate_up: bool = True
    decode_padded_gemm_rows: int = 8
    decode_padded_gemm_max_out_dim: int = 300000


FASTPATH = FastPath()


def as_manifest(fastpath: FastPath = FASTPATH) -> dict[str, object]:
    return asdict(fastpath)


def format_manifest(fastpath: FastPath = FASTPATH) -> str:
    pairs = as_manifest(fastpath)
    return "\n".join(f"{key}: {value}" for key, value in pairs.items())


def engine_overrides(fastpath: FastPath = FASTPATH) -> dict[str, object]:
    """Return current ``Qwen3_5Config`` fields for the promoted path.

    Workload and capacity remain in public config; implementation policy is
    centralized here so serving config cannot fork the execution path.
    """

    return {
        "dtype": fastpath.compute_dtype,
        "weight_dtype": fastpath.weight_dtype,
        "jax_execution": fastpath.execution,
        "prefill_layout": fastpath.prefill_layout,
        "greedy_token_fastpath": fastpath.greedy_token_fastpath,
        "sampled_token_fastpath": fastpath.sampled_token_fastpath,
        "device_token_carry": fastpath.device_token_carry,
        "static_decode_metadata": fastpath.static_decode_metadata,
        "static_decode_seq_lens_carry": fastpath.static_decode_seq_lens_carry,
        "resident_decode_metadata": fastpath.resident_decode_metadata,
        "materialize_tied_lm_head": fastpath.materialize_tied_lm_head,
        "compact_prefill_in_proj_qkv": fastpath.compact_prefill_in_proj_qkv,
        "compact_prefill_gdn_z": fastpath.compact_prefill_gdn_z,
        "compact_prefill_full_attn_proj": fastpath.compact_prefill_full_attn_proj,
        "compact_prefill_mlp": fastpath.compact_prefill_mlp,
        "compact_prefill_token_count_mode": fastpath.compact_prefill_token_count_mode,
        "lm_head_decode_act_dtype": fastpath.lm_head_decode_act_dtype,
        "lm_head_topk_impl": fastpath.lm_head_sampled,
        "lm_head_greedy_top1_impl": fastpath.lm_head_greedy,
        "decode_proj_act_dtype": fastpath.decode_proj_act_dtype,
        "decode_padded_gemm": fastpath.decode_padded_gemm,
        "decode_padded_gemm_gate_up": fastpath.decode_padded_gemm_gate_up,
        "decode_padded_gemm_rows": fastpath.decode_padded_gemm_rows,
        "decode_padded_gemm_max_out_dim": fastpath.decode_padded_gemm_max_out_dim,
        "full_attention_kv_cache_dtype": fastpath.kv_cache_dtype,
        "full_attention_kv_append_impl": fastpath.full_attention_kv_append,
        "full_attention_decode_impl": fastpath.full_attention_decode,
        "full_attention_prefill_impl": fastpath.full_attention_prefill,
        "gdn_disable_fallbacks": fastpath.gdn_disable_fallbacks,
        "gdn_prefill_post_conv_impl": fastpath.gdn_prefill,
        "gdn_prefill_qkv_dtype": fastpath.gdn_prefill_qkv_dtype,
        "gdn_prefill_post_conv_output_dtype": fastpath.gdn_prefill_output_dtype,
        "gdn_packed_decode_impl": fastpath.gdn_decode,
        "gdn_packed_decode_qkv_dtype": fastpath.gdn_decode_qkv_dtype,
    }


def required_modules(fastpath: FastPath = FASTPATH) -> tuple[str, ...]:
    modules: set[str] = set()
    if fastpath.full_attention_prefill.startswith("triton"):
        modules.update({"triton", "jax_triton"})
    if fastpath.full_attention_decode.startswith("flashinfer"):
        modules.update({"flashinfer", "jax_tvm_ffi"})
    if fastpath.gdn_prefill.startswith("triton"):
        modules.update({"triton", "jax_triton"})
    if fastpath.lm_head_greedy.startswith("triton"):
        modules.update({"triton", "jax_triton"})
    return tuple(sorted(modules))


def missing_required_modules(fastpath: FastPath = FASTPATH) -> tuple[str, ...]:
    return tuple(
        module
        for module in required_modules(fastpath)
        if importlib.util.find_spec(module) is None
    )


def validate_runtime_dependencies(fastpath: FastPath = FASTPATH) -> None:
    missing = missing_required_modules(fastpath)
    if missing:
        raise RuntimeError(
            "Promoted fast path dependencies are missing: "
            + ", ".join(missing)
            + ". Install the serving extras with "
            "`pip install -e \".[cuda13,flashinfer-ffi,gdn-fla-triton]\"`."
        )
