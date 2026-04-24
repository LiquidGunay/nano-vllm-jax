"""Speculative decoding with MTP (Multi-Token Prediction)."""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from dataclasses import replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kv_cache import KVCacheState


def speculative_decode_step(
    current_token,
    params,
    config,
    kv_cache_state,
    forward_fn: Callable,
    draft_forward_fn: Callable,
    num_spec_tokens: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[jnp.ndarray, KVCacheState, int]:
    """Single speculative decoding step.
    
    Args:
        current_token: Current token [batch, 1]
        params: Model parameters
        config: Model config
        kv_cache_state: KV cache state
        forward_fn: Main model forward function (tokens, params, config, kv_cache_state, is_prefill, return_hidden)
        draft_forward_fn: MTP draft forward function (hidden_state, next_token_ids, embed_tokens, params, config)
        num_spec_tokens: Number of tokens to speculatively generate
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
    
    Returns:
        Tuple of:
        - accepted_tokens: Accepted token IDs [batch, num_accepted]
        - new_kv_cache_state: Updated KV cache
        - num_accepted: Number of accepted tokens
    """
    batch_size = current_token.shape[0]
    
    # Step 1: Get hidden state from main model
    hidden_state, kv_cache_state = forward_fn(
        current_token, params, config, kv_cache_state, is_prefill=False, return_hidden=True
    )
    
    # Step 2: Generate draft tokens with MTP
    # MTP generates draft tokens one at a time, using the main model's hidden state
    draft_tokens = []
    draft_probs = []
    
    draft_token = current_token
    for i in range(num_spec_tokens):
        # MTP forward pass
        draft_logits, _ = draft_forward_fn(
            hidden_state=hidden_state,
            next_token_ids=draft_token,
            embed_tokens=params.embed_tokens,
            params=params.mtp_params,
            config=config,
        )
        
        # Store draft probabilities
        draft_probs.append(draft_logits)
        
        # Sample draft token
        if temperature > 0:
            draft_logits_temp = draft_logits / temperature
            probs = jax.nn.softmax(draft_logits_temp[:, -1, :], axis=-1)
            
            # Top-p sampling
            sorted_probs = jnp.sort(probs, axis=-1)[:, ::-1]
            cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
            cutoff_idx = jnp.sum(cumsum_probs <= top_p, axis=-1)
            mask = probs >= sorted_probs[jnp.arange(batch_size), cutoff_idx][:, None]
            probs = jnp.where(mask, probs, 0.0)
            probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
            
            draft_token = jax.random.categorical(
                jax.random.PRNGKey(42 + i), jnp.log(probs + 1e-10)
            )[:, None]
        else:
            # Greedy sampling
            draft_token = jnp.argmax(draft_logits[:, -1, :], axis=-1)[:, None]
        
        draft_tokens.append(draft_token[:, 0])
    
    # Step 3: Verify draft tokens with main model (autoregressively)
    accepted_tokens = []
    num_accepted = 0
    
    for i in range(num_spec_tokens):
        draft_token_i = draft_tokens[i]
        
        # Run main model on this draft token
        main_logits, kv_cache_state = forward_fn(
            draft_token_i[:, None], params, config, kv_cache_state, is_prefill=False, return_hidden=False
        )
        
        # Get main model probability for draft token
        main_prob = jax.nn.softmax(main_logits[:, 0, :], axis=-1)
        draft_prob = jax.nn.softmax(draft_probs[i][:, 0, :], axis=-1)
        
        # Get probability of draft token under both models
        p_main = main_prob[jnp.arange(batch_size), draft_token_i]
        p_draft = draft_prob[jnp.arange(batch_size), draft_token_i]
        
        # Acceptance probability: min(1, p_main / p_draft)
        acceptance_prob = jnp.minimum(1.0, p_main / (p_draft + 1e-10))
        
        # Sample uniform random for acceptance test
        r = jax.random.uniform(jax.random.PRNGKey(100 + i), shape=(batch_size,))
        
        # Accept if r < acceptance_prob
        accepted = r < acceptance_prob
        
        if jnp.all(accepted):
            accepted_tokens.append(draft_token_i)
            num_accepted += 1
        else:
            # Reject - sample from adjusted distribution
            # p_adjusted = max(0, p_main - p_draft)
            p_adjusted = jnp.maximum(0.0, main_prob - draft_prob)
            p_adjusted = p_adjusted / (jnp.sum(p_adjusted, axis=-1, keepdims=True) + 1e-10)
            
            # Sample from adjusted distribution
            if temperature > 0:
                sampled_token = jax.random.categorical(
                    jax.random.PRNGKey(200 + i), jnp.log(p_adjusted + 1e-10)
                )
            else:
                sampled_token = jnp.argmax(p_adjusted, axis=-1)
            
            accepted_tokens.append(sampled_token)
            break
    
    # If all draft tokens accepted, sample one more token from main model
    if num_accepted == num_spec_tokens:
        # Use the last main logits to sample bonus token
        if temperature > 0:
            bonus_token = jax.random.categorical(
                jax.random.PRNGKey(300), jnp.log(main_prob / temperature + 1e-10)
            )
        else:
            bonus_token = jnp.argmax(main_prob, axis=-1)
        
        accepted_tokens.append(bonus_token)
        num_accepted += 1
    
    return jnp.stack(accepted_tokens, axis=1), kv_cache_state, num_accepted


def speculative_decode(
    prompt_tokens,
    params,
    config,
    forward_fn: Callable,
    draft_forward_fn: Callable,
    max_new_tokens: int = 100,
    num_spec_tokens: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    kv_cache_state: Optional[KVCacheState] = None,
) -> jnp.ndarray:
    """Speculative decoding for text generation.
    
    Args:
        prompt_tokens: Prompt token IDs [batch, prompt_len]
        params: Model parameters
        config: Model config
        forward_fn: Main model forward function
        draft_forward_fn: MTP draft forward function
        max_new_tokens: Maximum number of new tokens to generate
        num_spec_tokens: Number of tokens to speculatively generate per step
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        kv_cache_state: Optional initial KV cache state
    
    Returns:
        Generated tokens [batch, prompt_len + num_generated]
    """
    batch_size, prompt_len = prompt_tokens.shape
    
    # Initialize KV cache if not provided
    if kv_cache_state is None:
        from nanovllm_jax.kv_cache import init_kv_cache, init_linear_attention_states
        num_blocks = (prompt_len + max_new_tokens + config.block_size - 1) // config.block_size + 10
        
        kv_cache_state = init_kv_cache(
            num_blocks=num_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seqs=batch_size,
            max_blocks_per_seq=num_blocks,
            dtype=config.get_dtype(),
        )
        kv_cache_state = init_linear_attention_states(kv_cache_state, config, batch_size=batch_size, dtype=config.get_dtype())
        
        # Set up block_table
        block_table = jnp.arange(num_blocks)[None, :]
        kv_cache_state = replace(
            kv_cache_state,
            block_table=block_table,
        )
    
    # Prefill prompt
    positions = jnp.arange(prompt_len)[None, :]
    kv_cache_state = replace(
        kv_cache_state,
        slot_mapping=positions,
    )
    
    _, kv_cache_state = forward_fn(
        prompt_tokens, params, config, kv_cache_state, is_prefill=True
    )
    
    # Generate tokens with speculative decoding
    generated_tokens = [prompt_tokens]
    total_generated = 0
    total_accepted = 0
    total_spec_steps = 0
    
    current_token = prompt_tokens[:, -1:]
    
    while total_generated < max_new_tokens:
        # Speculative decode step
        accepted, kv_cache_state, num_accepted = speculative_decode_step(
            current_token,
            params,
            config,
            kv_cache_state,
            forward_fn,
            draft_forward_fn,
            num_spec_tokens=num_spec_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        generated_tokens.append(accepted)
        total_generated += accepted.shape[1]
        total_accepted += num_accepted
        total_spec_steps += 1
        
        current_token = accepted[:, -1:]
        
        # Stop if EOS token
        if config.eos is not None and jnp.any(current_token == config.eos):
            break
    
    # Concatenate all tokens
    all_tokens = jnp.concatenate(generated_tokens, axis=1)
    
    # Print stats
    acceptance_rate = total_accepted / (total_spec_steps * num_spec_tokens) if total_spec_steps > 0 else 0.0
    print(f"\nSpeculative Decoding Stats:")
    print(f"  Total tokens generated: {total_generated}")
    print(f"  Spec steps: {total_spec_steps}")
    print(f"  Acceptance rate: {acceptance_rate:.2%}")
    print(f"  Effective speedup: {1 + acceptance_rate * num_spec_tokens:.2f}x")
    
    return all_tokens
