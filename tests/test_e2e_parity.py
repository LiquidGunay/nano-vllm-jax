"""End-to-end parity tests against HuggingFace Qwen3.5 model.

Tests the complete model forward pass and generation:
- Top 5 predicted logits match exactly for 10 different sequences
- Each sequence length >= 10
- Total MSE < 1e-4 for bfloat16 weights + fp32 activations
- Generated tokens match HF exactly

Success Criteria:
- Top 5 logits match exactly for all test prompts
- MSE < 1e-4
- Token generation matches for 10+ tokens
"""

import sys
import os
# Ensure we import from the correct location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights import load_weights_from_hf
from nanovllm_jax.model import forward
 
jax.config.update("jax_default_matmul_precision", "highest")

MODEL_NAME = os.getenv("HF_PARITY_MODEL", "Qwen/Qwen3.5-0.8B")


def resolve_hf_device() -> torch.device:
    requested = os.getenv("HF_TEST_DEVICE", "cpu").lower()
    if requested in {"auto", "cuda", "gpu"}:
        if not torch.cuda.is_available() or torch.version.cuda is None:
            print(
                "HF_TEST_DEVICE requested CUDA/GPU but CUDA is unavailable; "
                "falling back to CPU for parity tests."
            )
            return torch.device("cpu")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    pytest.skip(f"Unknown HF_TEST_DEVICE={requested}")


def load_models(model_name: str = MODEL_NAME):
    """Load both HF and JAX models."""
    hf_device = resolve_hf_device()
    print(f"\nLoading models from {model_name}...")
    
    # Load HF model
    print("  Loading HF model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_model.float().to(hf_device)
    except ValueError as exc:
        # Newer/older transformers versions gate Qwen3.5 support differently.
        msg = str(exc).lower()
        if (
            "does not recognize" in msg
            or "does not recogniz" in msg
            or "not recognized" in msg
            or "model_type" in msg
        ):
            pytest.skip(
                f"HF config mapping missing for {model_name}; update transformers version or set HF_PARITY_MODEL to a supported checkpoint."
            )
        raise
    except OSError as exc:
        pytest.skip(f"Could not load local HF artifacts for {model_name}: {exc}")
    hf_model.eval()
    
    # Load JAX model with HF weights
    print("  Loading JAX model...")
    config = Qwen3_5Config.qwen3_5_0_8b()
    config.dtype = "bfloat16"
    params = load_weights_from_hf(model_name, config)
    config.dtype = "float32"
    
    print("  ✓ Models loaded")
    return hf_model, tokenizer, config, params, hf_device


@pytest.fixture(scope="session")
def e2e_models():
    """Load shared HF/JAX artifacts for the module."""
    return load_models()


def _default_prompts() -> List[str]:
    return [
        "The future of artificial intelligence is poised to revolutionize",
        "In the beginning, there was nothing but an infinite void of darkness",
        "The most important discovery in physics was the theory of relativity",
        "When we consider the nature of consciousness, we must ask fundamental",
        "The economic implications of climate change are profound and far-reaching",
        "Programming languages have evolved significantly since the early days of",
        "The human brain processes information through complex neural networks",
        "Mathematical proofs require rigorous logical reasoning and careful",
        "The history of civilization demonstrates that human societies continually",
        "Scientific progress depends on careful observation and systematic analysis",
    ]


def test_logits_parity(
    e2e_models,
    prompts: Optional[List[str]] = None,
):
    """Test that top 5 logits match exactly between HF and JAX."""
    hf_model, tokenizer, config, params, hf_device = e2e_models
    prompts = prompts or _default_prompts()
    print("\n" + "=" * 80)
    print("TESTING LOGITS PARITY")
    print("=" * 80)

    total_mse = 0.0
    num_tests = 0
    
    for i, prompt in enumerate(prompts):
        print(
            f"\n[Prompt {i+1}/{len(prompts)}] \"{prompt[:50]}...\""
            if len(prompt) > 50
            else f"\n[Prompt {i+1}/{len(prompts)}] \"{prompt}\""
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_device)
        input_ids = inputs["input_ids"]
        
        if input_ids.shape[1] < 10:
            print(f"  ⚠ Skipping: sequence length {input_ids.shape[1]} < 10")
            continue
        
        # HF forward
        with torch.no_grad():
            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits[0, -1, :].float().cpu().numpy()  # Last token logits
        
        # JAX forward (prefill mode) - no KV cache for simple test
        input_ids_jax = jnp.array(input_ids.cpu().numpy())
        
        # Forward pass (no KV cache for simple logits test)
        logits_jax, _ = forward(
            input_ids_jax,
            params,
            config,
            kv_cache_state=None,  # No cache for simple forward pass
            is_prefill=True,
        )
        jax_logits = np.array(logits_jax[0, -1, :])  # Last token logits
        
        # Compare logits
        mse = np.mean((hf_logits - jax_logits) ** 2)
        total_mse += mse
        
        # Get top 5 tokens
        hf_top5 = np.argsort(hf_logits)[-5:][::-1].copy()
        jax_top5 = np.argsort(jax_logits)[-5:][::-1].copy()
        
        top5_match = np.array_equal(hf_top5, jax_top5)
        
        print(f"  HF top 5 tokens: {hf_top5}")
        print(f"  JAX top 5 tokens: {jax_top5}")
        print(f"  MSE: {mse:.2e}")
        print(f"  Top 5 match: {'✓' if top5_match else '✗'}")

        assert top5_match, f"Top 5 tokens do not match for prompt {i}"
        assert mse < 1e-8, f"MSE {mse:.2e} >= 1e-8 for prompt {i}"
        num_tests += 1
    
    avg_mse = total_mse / num_tests if num_tests > 0 else 0.0
    
    print("\n" + "-" * 80)
    print(f"Average MSE: {avg_mse:.2e}")
    print(f"Target: MSE < 1e-4")
    assert num_tests > 0


def test_generation_parity(
    e2e_models,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 10,
):
    """Test that generated tokens match between HF and JAX."""
    hf_model, tokenizer, config, params, hf_device = e2e_models
    prompts = prompts or _default_prompts()[:3]
    print("\n" + "=" * 80)
    print("TESTING GENERATION PARITY")
    print("=" * 80)

    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{len(prompts)}] \"{prompt[:50]}...\"" if len(prompt) > 50 else f"\n[Prompt {i+1}/{len(prompts)}] \"{prompt}\"")
        
        # HF generation
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_device)
        with torch.no_grad():
            hf_output_ids = hf_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_generated = tokenizer.decode(hf_output_ids[0], skip_special_tokens=True)
        
        # JAX generation (simplified greedy decoding)
        input_ids = inputs["input_ids"].cpu().numpy()
        generated_ids = list(input_ids[0])
        
        # Prefill (no KV cache for simplicity)
        input_ids_jax = jnp.array([generated_ids])
        logits, _ = forward(
            input_ids_jax,
            params,
            config,
            kv_cache_state=None,  # No cache for simple test
            is_prefill=True,
        )
        
        # Decode tokens
        for _ in range(max_new_tokens):
            next_token = int(jnp.argmax(logits[0, -1, :]))
            generated_ids.append(next_token)
            
            # Decode step (forward full sequence each time - no cache)
            input_ids_jax = jnp.array([generated_ids])
            logits, _ = forward(
                input_ids_jax,
                params,
                config,
                kv_cache_state=None,
                is_prefill=True,  # Treat as prefill since no cache
            )
        
        jax_generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compare
        match = hf_generated == jax_generated
        print(f"  HF:  \"{hf_generated}\"")
        print(f"  JAX: \"{jax_generated}\"")
        print(f"  Match: {'✓' if match else '✗'}")
        assert match
