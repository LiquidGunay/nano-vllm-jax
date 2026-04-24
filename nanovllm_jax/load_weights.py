"""Load pretrained weights from HuggingFace and compare logits."""

import sys
import time
import json
from pathlib import Path

import torch
import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import Qwen3_5, ModelParams
from nanovllm_jax.mtp.mtp_layer import MTPParams


def download_hf_weights(model_name: str, cache_dir: str = None):
    """Load model path from HF cache."""
    from huggingface_hub import scan_cache_dir
    print(f"Loading {model_name} from cache...")
    cache = scan_cache_dir()
    # Find the repo in the cache
    for repo in cache.repos:
        if repo.repo_id == model_name:
            path = list(repo.revisions)[0].snapshot_path
            print(f"Using cached snapshot: {path}")
            return Path(path)
    raise ValueError(f"Model {model_name} not found in cache")

def load_safetensors(model_path: Path):
    """Load weights from safetensors files."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors"])
        from safetensors import safe_open
    
    weights = {}
    for st_file in model_path.glob("*.safetensors"):
        print(f"  Loading {st_file.name}...")
        # Use torch framework to properly handle bfloat16
        with safe_open(st_file, framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    return weights


def convert_hf_to_jax(hf_weights: dict, config: Qwen3_5Config, verbose: bool = False) -> ModelParams:
    """Convert HuggingFace weights to JAX format for Qwen 3.5."""
    print("Converting weights...")
    if not verbose:
        import warnings
        warnings.filterwarnings('ignore')
    
    # Import ml_dtypes for bfloat16 support in JAX
    import ml_dtypes
    
    # Extract text model weights (prefix "model.language_model." or "model.")
    # HF loads ALL weights as bfloat16, so we keep them as bfloat16 in JAX
    text_weights = {}
    
    for key, value in hf_weights.items():
        # Get as bfloat16 numpy directly
        if hasattr(value, 'cpu'):  # torch tensor
            # Convert to bfloat16 if not already
            if value.dtype != torch.bfloat16:
                value = value.bfloat16()
            # Convert via float32 intermediate (torch can't directly numpy() bfloat16)
            value_np = value.float().cpu().numpy()
            # Convert back to bfloat16 numpy using ml_dtypes
            value_np = value_np.astype(ml_dtypes.bfloat16)
        else:
            # Already numpy (possibly ml_dtypes.bfloat16 from safetensors)
            value_np = np.asarray(value)
            if value_np.dtype != np.dtype('bfloat16'):
                # Convert to bfloat16
                value_torch = torch.from_numpy(value_np).bfloat16()
                value_np = value_torch.float().numpy().astype(ml_dtypes.bfloat16)
        
        value_jax = jnp.array(value_np, dtype=jnp.bfloat16)
        
        if key.startswith("model.language_model."):
            new_key = key[21:]  # Remove "model.language_model." prefix
            text_weights[new_key] = value_jax
        elif key.startswith("model."):
            new_key = key[6:]  # Remove "model." prefix
            text_weights[new_key] = value_jax
    
    # Embeddings
    embed_tokens = text_weights.get("embed_tokens.weight")
    if embed_tokens is None:
        raise ValueError("embed_tokens.weight not found")
    
    # Convert layers
    layers = []
    for i in range(config.num_hidden_layers):
        layer_prefix = f"layers.{i}."
        layer_params = {}
        
        layer_type = config.layer_types[i]
        
        if layer_type == "full_attention":
            # Full attention layer
            layer_params["q_proj"] = text_weights[f"{layer_prefix}self_attn.q_proj.weight"].T
            layer_params["k_proj"] = text_weights[f"{layer_prefix}self_attn.k_proj.weight"].T
            layer_params["v_proj"] = text_weights[f"{layer_prefix}self_attn.v_proj.weight"].T
            layer_params["o_proj"] = text_weights[f"{layer_prefix}self_attn.o_proj.weight"].T
            
            # Norms (no shift needed - HF checkpoint is already sanitized)
            layer_params["q_norm"] = text_weights[f"{layer_prefix}self_attn.q_norm.weight"]
            layer_params["k_norm"] = text_weights[f"{layer_prefix}self_attn.k_norm.weight"]
            layer_params["input_norm"] = text_weights[f"{layer_prefix}input_layernorm.weight"]
            layer_params["post_attn_norm"] = text_weights[f"{layer_prefix}post_attention_layernorm.weight"]
            
            # MLP (SwiGLU)
            layer_params["gate_proj"] = text_weights[f"{layer_prefix}mlp.gate_proj.weight"].T
            layer_params["up_proj"] = text_weights[f"{layer_prefix}mlp.up_proj.weight"].T
            layer_params["down_proj"] = text_weights[f"{layer_prefix}mlp.down_proj.weight"].T
            layer_params["ffn_norm"] = text_weights[f"{layer_prefix}post_attention_layernorm.weight"]
            
        else:
            # Gated DeltaNet layer
            linear_prefix = f"{layer_prefix}linear_attn."
            
            # in_proj_qkv: [6144, 1024] -> split into q, k, v after loading
            layer_params["in_proj_qkv"] = text_weights[f"{linear_prefix}in_proj_qkv.weight"].T  # [1024, 6144]
            
            # in_proj_a: [16, 1024] -> [1024, 16]
            layer_params["in_proj_a"] = text_weights[f"{linear_prefix}in_proj_a.weight"].T
            
            # in_proj_b: [16, 1024] -> [1024, 16]
            layer_params["in_proj_b"] = text_weights[f"{linear_prefix}in_proj_b.weight"].T
            
            # in_proj_z: [2048, 1024] -> [1024, 2048]
            layer_params["in_proj_z"] = text_weights[f"{linear_prefix}in_proj_z.weight"].T
            
            # conv1d: [6144, 1, 4] -> need to squeeze and transpose to [6144, 4]
            conv_weight = text_weights[f"{linear_prefix}conv1d.weight"]
            if conv_weight.ndim == 3:
                conv_weight = jnp.squeeze(conv_weight, axis=1)  # [6144, 4]
            layer_params["conv1d_weight"] = conv_weight
            
            # dt_bias: [16]
            layer_params["dt_bias"] = text_weights[f"{linear_prefix}dt_bias"]
            
            # A_log: [16] -> exp to get A (HF computes g = -exp(A_log) * softplus(...))
            # We store A = exp(A_log) and compute g = -A * softplus(...) in gated_deltanet_block
            layer_params["A"] = jnp.exp(text_weights[f"{linear_prefix}A_log"])
            
            # norm: [128] - RMSNorm over head groups (no shift needed)
            layer_params["norm_weight"] = text_weights[f"{linear_prefix}norm.weight"]
            
            # out_proj: [1024, 2048] -> [2048, 1024]
            layer_params["out_proj"] = text_weights[f"{linear_prefix}out_proj.weight"].T
            
            # Layer norms - use correct names for transformer_block (no shift needed)
            layer_params["input_norm"] = text_weights[f"{layer_prefix}input_layernorm.weight"]
            layer_params["ffn_norm"] = text_weights[f"{layer_prefix}post_attention_layernorm.weight"]
            
            # MLP (SwiGLU)
            layer_params["gate_proj"] = text_weights[f"{layer_prefix}mlp.gate_proj.weight"].T
            layer_params["up_proj"] = text_weights[f"{layer_prefix}mlp.up_proj.weight"].T
            layer_params["down_proj"] = text_weights[f"{layer_prefix}mlp.down_proj.weight"].T
        
        layers.append(layer_params)
    
    # Final norm (no shift needed)
    norm_weight = text_weights.get("norm.weight", jnp.ones(config.hidden_size))
    
    # LM head (check if tied)
    lm_head = None
    if "lm_head.weight" in text_weights:
        lm_head = text_weights["lm_head.weight"].T
    # Also check without language_model prefix
    elif "lm_head.weight" in hf_weights:
        lm_head_val = hf_weights["lm_head.weight"]
        if hasattr(lm_head_val, 'cpu'):
            lm_head_val = lm_head_val.cpu().float().numpy()
        lm_head = jnp.array(lm_head_val).T
    # Tie weights if no separate LM head
    elif config.tie_word_embeddings:
        # embed_tokens is [vocab_size, hidden_size], lm_head needs [hidden_size, vocab_size]
        lm_head = embed_tokens.T
    
    return ModelParams(
        embed_tokens=embed_tokens,
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
    )


def load_weights_from_hf(params, model_name: str, config: Qwen3_5Config = None, verbose: bool = False, load_mtp: bool = False) -> ModelParams:
    """Load weights from HuggingFace for a given model.
    
    Args:
        params: Existing ModelParams (ignored, kept for API compatibility)
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3.5-0.8B")
        config: Model configuration (required)
        verbose: Whether to print detailed weight info
        load_mtp: Whether to load MTP weights for speculative decoding
        
    Returns:
        ModelParams with loaded weights (and optional mtp_params attribute)
    """
    print(f"Loading weights for {model_name}...")
    
    if config is None:
        config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Download from HF
    hf_path = download_hf_weights(model_name)
    
    # Load safetensors
    hf_weights = load_safetensors(hf_path)
    
    # Convert to JAX format
    jax_params = convert_hf_to_jax(hf_weights, config, verbose=verbose)
    
    # Load MTP weights if requested
    if load_mtp:
        print("Loading MTP weights...")
        mtp_params = load_mtp_weights_from_hf(hf_weights, config, verbose=verbose)
        # Share LM head from main model (tied embeddings)
        mtp_params.lm_head = jax_params.lm_head
        jax_params.mtp_params = mtp_params  # Attach MTP params to main params
        print(f"✓ Loaded MTP head: {config.mtp_num_hidden_layers} layer(s)")
    
    print(f"✓ Loaded weights: {len(jax_params.layers)} layers")
    return jax_params


def load_mtp_weights_from_hf(hf_weights: dict, config: Qwen3_5Config, verbose: bool = False) -> MTPParams:
    """Load MTP weights from HuggingFace checkpoint.
    
    Qwen3.5 MTP weight naming:
    - mtp.fc.weight -> eh_proj (input fusion)
    - mtp.pre_fc_norm_hidden.weight
    - mtp.pre_fc_norm_embedding.weight  
    - mtp.layers.0.* -> MTP transformer layer
    - mtp.norm.weight -> final norm
    
    Args:
        hf_weights: Dictionary of HF weights
        config: Model config
        verbose: Whether to print detailed info
        
    Returns:
        MTPParams with loaded weights
    """
    # Extract MTP weights
    mtp_weights = {}
    for key, value in hf_weights.items():
        if key.startswith("model.mtp."):
            mtp_weights[key[12:]] = value
        elif key.startswith("mtp."):
            mtp_weights[key[4:]] = value
    
    if not mtp_weights:
        raise ValueError("No MTP weights found in checkpoint")
    
    if verbose:
        print(f"  Found {len(mtp_weights)} MTP weight tensors")
    
    # Convert to JAX format (handle bfloat16)
    def convert_tensor(value):
        if hasattr(value, 'cpu'):
            value = value.cpu().float().numpy()
        else:
            # Handle ml_dtypes.bfloat16 from safetensors
            value = np.asarray(value, dtype=np.float32)
        # Simulate HF's bfloat16 loading
        value = torch.from_numpy(value).bfloat16().float().numpy()
        return jnp.array(value)
    
    # Input fusion projection (mtp.fc.weight in HF)
    eh_proj = convert_tensor(mtp_weights["fc.weight"]).T  # [hidden_size*2, hidden_size]
    
    # Pre-norms (NO shift: HF stores RMS norms directly for Qwen3.5)
    pre_fc_norm_hidden = convert_tensor(mtp_weights["pre_fc_norm_hidden.weight"])
    pre_fc_norm_embedding = convert_tensor(mtp_weights["pre_fc_norm_embedding.weight"])
    
    # MTP layers
    layers = []
    for i in range(config.mtp_num_hidden_layers):
        prefix = f"layers.{i}."
        layer = {}
        
        # Self-attention
        layer["q_proj"] = convert_tensor(mtp_weights[f"{prefix}self_attn.q_proj.weight"]).T
        layer["k_proj"] = convert_tensor(mtp_weights[f"{prefix}self_attn.k_proj.weight"]).T
        layer["v_proj"] = convert_tensor(mtp_weights[f"{prefix}self_attn.v_proj.weight"]).T
        layer["o_proj"] = convert_tensor(mtp_weights[f"{prefix}self_attn.o_proj.weight"]).T
        
        # Norms (NO shift: HF stores RMS norms directly for Qwen3.5)
        layer["q_norm"] = convert_tensor(mtp_weights[f"{prefix}self_attn.q_norm.weight"])
        layer["k_norm"] = convert_tensor(mtp_weights[f"{prefix}self_attn.k_norm.weight"])
        layer["input_norm"] = convert_tensor(mtp_weights[f"{prefix}input_layernorm.weight"])
        layer["post_attn_norm"] = convert_tensor(mtp_weights[f"{prefix}post_attention_layernorm.weight"])
        
        # MLP (note: post_attention_layernorm is applied before MLP in Pre-LN architecture)
        layer["gate_proj"] = convert_tensor(mtp_weights[f"{prefix}mlp.gate_proj.weight"]).T
        layer["up_proj"] = convert_tensor(mtp_weights[f"{prefix}mlp.up_proj.weight"]).T
        layer["down_proj"] = convert_tensor(mtp_weights[f"{prefix}mlp.down_proj.weight"]).T
        layer["ffn_norm"] = convert_tensor(mtp_weights[f"{prefix}post_attention_layernorm.weight"])
        
        layers.append(layer)
    
    # Final norm (mtp.norm.weight) - NO shift for Qwen3.5
    final_norm = convert_tensor(mtp_weights["norm.weight"])
    
    # LM head (shared with main model, not in MTP weights)
    lm_head = None
    
    return MTPParams(
        eh_proj=eh_proj,
        layers=layers,
        pre_fc_norm_hidden=pre_fc_norm_hidden,
        pre_fc_norm_embedding=pre_fc_norm_embedding,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def test_with_text(model_name: str = "Qwen/Qwen3.5-0.8B"):
    """Test model with actual text input."""
    print("=" * 60)
    print(f"Testing {model_name}")
    print("=" * 60)
    
    # Load config
    config = Qwen3_5Config.qwen3_5_0_8b()
    
    # Download and load weights
    hf_path = download_hf_weights(model_name)
    hf_weights = load_safetensors(hf_path)
    jax_params = convert_hf_to_jax(hf_weights, config)
    
    # Create model
    model = Qwen3_5(config, jax.random.PRNGKey(0))
    model.params = jax_params
    
    # Test with sample input
    test_input = jnp.array([[1, 2, 3, 4, 5]])
    output = model.forward(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (first 5 tokens): {output[0, :5]}")
