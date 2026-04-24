"""Load weights with float16 for Metal compatibility."""

import os
import json
import numpy as np
import jax.numpy as jnp
from safetensors import safe_open
from typing import Dict, Any
import torch


def load_config(model_name: str) -> Dict[str, Any]:
    """Load model config from HuggingFace cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    for name in os.listdir(cache_dir):
        if name.startswith(f"models--{model_name.replace('/', '--')}"):
            model_dir = os.path.join(cache_dir, name)
            break
    else:
        raise ValueError(f"Model {model_name} not found in cache")
    
    refs_path = os.path.join(model_dir, "refs", "main")
    if os.path.exists(refs_path):
        with open(refs_path) as f:
            snapshot = f.read().strip()
    else:
        snapshot = "main"
    
    snapshot_dir = os.path.join(model_dir, "snapshots", snapshot)
    config_path = os.path.join(snapshot_dir, "config.json")
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config, snapshot_dir


def convert_hf_to_jax_float16(hf_weights: Dict[str, torch.Tensor], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Convert HuggingFace weights to JAX format with float16."""
    jax_params = {}
    
    for key, value_torch in hf_weights.items():
        if verbose and len(hf_weights) < 50:
            print(f"  Converting {key}: {value_torch.shape} {value_torch.dtype}")
        
        # Convert to float32 then float16 (handles bfloat16 input)
        if value_torch.dtype == torch.bfloat16:
            value_np = value_torch.float().numpy().astype(np.float16)
        elif value_torch.dtype == torch.float32:
            value_np = value_torch.numpy().astype(np.float16)
        elif value_torch.dtype == torch.float16:
            value_np = value_torch.numpy()
        else:
            # Keep other dtypes as-is (int, etc.)
            value_np = value_torch.numpy()
        
        value_jax = jnp.array(value_np)
        jax_params[key] = value_jax
    
    return jax_params


def load_weights_from_hf_float16(model_name: str = "Qwen/Qwen3.5-0.8B", verbose: bool = True):
    """Load weights from HuggingFace cache with float16.
    
    This is a compatibility wrapper that returns raw weight dict and config.
    For ModelParams, use load_weights_from_hf() instead.
    
    Args:
        model_name: HuggingFace model identifier
        verbose: Print loading progress
        
    Returns:
        Tuple of (jax_weights_dict, config_dict)
    """
    config_dict, snapshot_dir = load_config(model_name)
    
    if verbose:
        print(f"Loading {model_name} with float16...")
        print(f"Snapshot: {snapshot_dir}")
    
    # Load safetensors
    safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        safetensors_path = os.path.join(snapshot_dir, "model.safetensors-00001-of-00001.safetensors")
    
    if verbose:
        print(f"Loading {os.path.basename(safetensors_path)}...")
    
    hf_weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            hf_weights[key] = f.get_tensor(key)  # Keep as torch tensor
    
    if verbose:
        print(f"Loaded {len(hf_weights)} tensors")
    
    # Convert to JAX format
    jax_params = convert_hf_to_jax_float16(hf_weights, config_dict, verbose=verbose)
    
    return jax_params, config_dict


if __name__ == "__main__":
    params, config = load_weights_from_hf_float16("Qwen/Qwen3.5-0.8B", verbose=True)
    print(f"\nLoaded {len(params)} parameters")
    print(f"Sample key: {list(params.keys())[0]}")
    print(f"Sample dtype: {params[list(params.keys())[0]].dtype}")
