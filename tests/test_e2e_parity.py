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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights import load_weights_from_hf
from nanovllm_jax.model import forward, ModelParams
from nanovllm_jax.kv_cache import init_kv_cache, init_linear_attention_states


def load_models(model_name="Qwen/Qwen3.5-0.8B"):
    """Load both HF and JAX models."""
    print(f"\nLoading models from {model_name}...")
    
    # Load HF model
    print("  Loading HF model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    hf_model.eval()
    
    # Load JAX model with HF weights
    print("  Loading JAX model...")
    config = Qwen3_5Config.qwen3_5_0_8b()
    params = load_weights_from_hf(model_name, config)
    
    print("  ✓ Models loaded")
    return hf_model, tokenizer, config, params


def test_logits_parity(
    hf_model,
    tokenizer,
    config: Qwen3_5Config,
    params: ModelParams,
    prompts: List[str],
):
    """Test that top 5 logits match exactly between HF and JAX."""
    print("\n" + "=" * 80)
    print("TESTING LOGITS PARITY")
    print("=" * 80)
    
    all_pass = True
    total_mse = 0.0
    num_tests = len(prompts)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{num_tests}] \"{prompt[:50]}...\"" if len(prompt) > 50 else f"\n[Prompt {i+1}/{num_tests}] \"{prompt}\"")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        if input_ids.shape[1] < 10:
            print(f"  ⚠ Skipping: sequence length {input_ids.shape[1]} < 10")
            continue
        
        # HF forward
        with torch.no_grad():
            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits[0, -1, :].float().numpy()  # Last token logits
        
        # JAX forward (prefill mode) - no KV cache for simple test
        input_ids_jax = jnp.array(input_ids.numpy())
        
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
        
        hf_top5_probs = torch.nn.functional.softmax(torch.from_numpy(hf_logits.copy()), dim=0)[hf_top5]
        jax_top5_probs = jax.nn.softmax(jnp.array(jax_logits))[jax_top5]
        
        top5_match = np.array_equal(hf_top5, jax_top5)
        
        print(f"  HF top 5 tokens: {hf_top5}")
        print(f"  JAX top 5 tokens: {jax_top5}")
        print(f"  MSE: {mse:.2e}")
        print(f"  Top 5 match: {'✓' if top5_match else '✗'}")
        
        if not top5_match:
            all_pass = False
            print(f"  ✗ FAIL: Top 5 tokens do not match")
        elif mse >= 1e-4:
            all_pass = False
            print(f"  ✗ FAIL: MSE {mse:.2e} >= 1e-4")
        else:
            print(f"  ✓ PASS")
    
    avg_mse = total_mse / num_tests if num_tests > 0 else 0.0
    
    print("\n" + "-" * 80)
    print(f"Average MSE: {avg_mse:.2e}")
    print(f"Target: MSE < 1e-4")
    
    return all_pass and avg_mse < 1e-4


def test_generation_parity(
    hf_model,
    tokenizer,
    config: Qwen3_5Config,
    params: ModelParams,
    prompts: List[str],
    max_new_tokens: int = 10,
):
    """Test that generated tokens match between HF and JAX."""
    print("\n" + "=" * 80)
    print("TESTING GENERATION PARITY")
    print("=" * 80)
    
    all_pass = True
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{len(prompts)}] \"{prompt[:50]}...\"" if len(prompt) > 50 else f"\n[Prompt {i+1}/{len(prompts)}] \"{prompt}\"")
        
        # HF generation
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            hf_output_ids = hf_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_generated = tokenizer.decode(hf_output_ids[0], skip_special_tokens=True)
        
        # JAX generation (simplified greedy decoding)
        input_ids = inputs["input_ids"].numpy()
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
        
        if not match:
            all_pass = False
            print(f"  ✗ FAIL: Generated text does not match")
        else:
            print(f"  ✓ PASS")
    
    return all_pass


def run_all_tests():
    """Run all end-to-end parity tests."""
    print("=" * 80)
    print("END-TO-END PARITY TESTS")
    print("=" * 80)
    
    # Load models
    hf_model, tokenizer, config, params = load_models()
    
    # Test prompts (at least 10 tokens each after tokenization)
    test_prompts = [
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
    
    # Run logits parity test
    logits_pass = test_logits_parity(hf_model, tokenizer, config, params, test_prompts)
    
    # Run generation parity test (use fewer prompts for speed)
    generation_pass = test_generation_parity(
        hf_model, tokenizer, config, params, 
        test_prompts[:3],  # Test first 3 prompts
        max_new_tokens=10,
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Logits parity (top 5 exact match): {'✓ PASS' if logits_pass else '✗ FAIL'}")
    print(f"  Generation parity: {'✓ PASS' if generation_pass else '✗ FAIL'}")
    
    all_pass = logits_pass and generation_pass
    
    if all_pass:
        print("\n✅ All end-to-end parity tests PASSED!")
        return True
    else:
        print("\n❌ Some tests FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
