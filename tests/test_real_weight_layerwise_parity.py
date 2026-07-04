"""Prompt-driven layer parity against HuggingFace Qwen3.5 real weights."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm_jax.config import RuntimeConfig
from nanovllm_jax.layers import rms_norm
from nanovllm_jax.weights import load_weights_from_hf
from nanovllm_jax.model import _stable_rmsnorm_fp32


jax.config.update("jax_default_matmul_precision", "highest")

MODEL_NAME = os.getenv("HF_PARITY_MODEL", "Qwen/Qwen3.5-0.8B")
PROMPTS = (
    "The future of artificial intelligence is poised to revolutionize",
    "Mathematical proofs require rigorous logical reasoning and careful",
)
STAGE_NAMES = (
    "block_input",
    "input_norm",
    "mixer_out",
    "attn_residual",
    "ffn_norm",
    "mlp_out",
    "block_output",
)


@dataclass
class PromptTrace:
    prompt: str
    token_ids: np.ndarray
    stages: np.ndarray
    final_norm: np.ndarray
    logits: np.ndarray
    standard_norm: dict[str, np.ndarray]
    gated_norm: dict[str, np.ndarray]


@dataclass
class RealWeightArtifacts:
    config: RuntimeConfig
    params: object
    traces: tuple[PromptTrace, ...]


def _require_cuda_device() -> torch.device:
    requested = os.getenv("HF_TEST_DEVICE", "cuda").lower()
    if requested not in {"cuda", "gpu"}:
        pytest.skip("Real-weight layer parity requires HF_TEST_DEVICE=cuda")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("CUDA is unavailable for HuggingFace parity capture")
    if jax.default_backend() != "gpu":
        pytest.skip(f"JAX default backend is {jax.default_backend()!r}, expected 'gpu'")
    return torch.device("cuda")


def _to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _as_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _capture_hf_traces(model, tokenizer, device: torch.device) -> tuple[PromptTrace, ...]:
    traces: list[PromptTrace] = []

    for prompt in PROMPTS:
        captures = {
            layer_idx: {}
            for layer_idx in range(len(model.model.layers))
        }
        final_capture = {}
        handles = []

        def add_hook(module, fn):
            handles.append(module.register_forward_hook(fn))

        def add_pre_hook(module, fn):
            handles.append(module.register_forward_pre_hook(fn))

        for layer_idx, layer in enumerate(model.model.layers):
            add_pre_hook(
                layer,
                lambda module, inputs, idx=layer_idx: captures[idx].__setitem__(
                    "block_input", _to_np(inputs[0])
                ),
            )
            add_hook(
                layer.input_layernorm,
                lambda module, inputs, output, idx=layer_idx: captures[idx].__setitem__(
                    "input_norm", _to_np(output)
                ),
            )
            mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            add_hook(
                mixer,
                lambda module, inputs, output, idx=layer_idx: captures[idx].__setitem__(
                    "mixer_out", _to_np(_as_tensor(output))
                ),
            )
            add_hook(
                layer.post_attention_layernorm,
                lambda module, inputs, output, idx=layer_idx: captures[idx].__setitem__(
                    "ffn_norm", _to_np(output)
                ),
            )
            add_hook(
                layer.mlp,
                lambda module, inputs, output, idx=layer_idx: captures[idx].__setitem__(
                    "mlp_out", _to_np(output)
                ),
            )
            add_hook(
                layer,
                lambda module, inputs, output, idx=layer_idx: captures[idx].__setitem__(
                    "block_output", _to_np(_as_tensor(output))
                ),
            )

        add_hook(model.model.norm, lambda module, inputs, output: final_capture.__setitem__("final_norm", _to_np(output)))

        first_linear_layer = model.model.layers[0]
        linear_norm_capture = {}
        handles.append(
            first_linear_layer.linear_attn.norm.register_forward_hook(
                lambda module, inputs, output: linear_norm_capture.update(
                    {
                        "input": _to_np(inputs[0]),
                        "gate": _to_np(inputs[1]),
                        "output": _to_np(output),
                        "weight": _to_np(module.weight),
                    }
                )
            )
        )

        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)

        for handle in handles:
            handle.remove()

        for layer_idx, layer_capture in captures.items():
            layer_capture["attn_residual"] = layer_capture["block_input"] + layer_capture["mixer_out"]

        stages = np.stack(
            [
                np.stack([captures[layer_idx][name] for name in STAGE_NAMES], axis=0)
                for layer_idx in range(len(model.model.layers))
            ],
            axis=0,
        )

        traces.append(
            PromptTrace(
                prompt=prompt,
                token_ids=encoded["input_ids"].detach().cpu().numpy().astype(np.int32),
                stages=stages,
                final_norm=final_capture["final_norm"],
                logits=_to_np(outputs.logits),
                standard_norm={
                    "input": captures[0]["block_input"],
                    "output": captures[0]["input_norm"],
                    "weight": _to_np(model.model.layers[0].input_layernorm.weight),
                },
                gated_norm=linear_norm_capture,
            )
        )

    return tuple(traces)


@pytest.fixture(scope="session")
def real_weight_artifacts() -> RealWeightArtifacts:
    device = _require_cuda_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Load checkpoint values through BF16, then run HF math in FP32.  This
    # mirrors the intended JAX contract: BF16 weights with FP32 activations.
    hf_model.float().to(device)
    hf_model.eval()

    traces = _capture_hf_traces(hf_model, tokenizer, device)

    del hf_model
    torch.cuda.empty_cache()

    load_config = RuntimeConfig.qwen3_5_0_8b()
    load_config.dtype = "bfloat16"
    params = load_weights_from_hf(MODEL_NAME, load_config)

    runtime_config = RuntimeConfig.qwen3_5_0_8b()
    runtime_config.dtype = "float32"
    return RealWeightArtifacts(config=runtime_config, params=params, traces=traces)


def _mse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean((actual.astype(np.float32) - expected.astype(np.float32)) ** 2))


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))


def test_real_weight_rmsnorm_offset_semantics(real_weight_artifacts: RealWeightArtifacts):
    """Standard Qwen3.5 RMSNorm uses 1 + weight; gated DeltaNet RMSNorm does not."""
    trace = real_weight_artifacts.traces[0]

    standard = trace.standard_norm
    standard_out = np.array(
        rms_norm(
            jnp.array(standard["input"]),
            jnp.array(standard["weight"]),
            real_weight_artifacts.config.rms_norm_eps,
        )
    )
    raw_weight_out = np.array(
        _stable_rmsnorm_fp32(
            jnp.array(standard["input"]),
            jnp.array(standard["weight"]),
            real_weight_artifacts.config.rms_norm_eps,
        )
    )

    assert _mse(standard_out, standard["output"]) < 1e-5
    assert _mse(raw_weight_out, standard["output"]) > 1e-2

    gated = trace.gated_norm
    gated_normed = _stable_rmsnorm_fp32(
        jnp.array(gated["input"]),
        jnp.array(gated["weight"]),
        real_weight_artifacts.config.rms_norm_eps,
    )
    gated_out = np.array(gated_normed * jax.nn.silu(jnp.array(gated["gate"])))
    gated_off_by_one = np.array(
        rms_norm(
            jnp.array(gated["input"]),
            jnp.array(gated["weight"]),
            real_weight_artifacts.config.rms_norm_eps,
        )
        * jax.nn.silu(jnp.array(gated["gate"]))
    )

    assert _mse(gated_out, gated["output"]) < 1e-5
    assert _mse(gated_off_by_one, gated["output"]) > 1e-3
