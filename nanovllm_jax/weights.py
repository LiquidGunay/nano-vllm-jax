"""Load pretrained Hugging Face weights into the serving parameter tree."""

import time
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.model import ModelParams


GDN_DECODE_IN_PROJ_PACKED_KEY = "in_proj_qkv_abz"
FULL_ATTN_DECODE_QKV_PACKED_KEY = "qkv_proj_decode"
MLP_GATE_UP_PACKED_KEY = "gate_up_proj"


def _add_gdn_decode_packed_in_proj(layer_params: dict[str, jnp.ndarray]) -> None:
    """Add a decode-only packed GDN input projection weight."""
    layer_params[GDN_DECODE_IN_PROJ_PACKED_KEY] = jnp.concatenate(
        [
            layer_params["in_proj_qkv"],
            layer_params["in_proj_a"],
            layer_params["in_proj_b"],
            layer_params["in_proj_z"],
        ],
        axis=1,
    )


def _add_full_attention_decode_packed_qkv(layer_params: dict[str, jnp.ndarray]) -> None:
    """Add a decode-only packed full-attention Q/K/V projection weight."""
    layer_params[FULL_ATTN_DECODE_QKV_PACKED_KEY] = jnp.concatenate(
        [
            layer_params["q_proj"],
            layer_params["k_proj"],
            layer_params["v_proj"],
        ],
        axis=1,
    )


def _add_mlp_packed_gate_up(layer_params: dict[str, jnp.ndarray]) -> None:
    """Add a packed SwiGLU gate/up projection weight."""
    layer_params[MLP_GATE_UP_PACKED_KEY] = jnp.concatenate(
        [
            layer_params["gate_proj"],
            layer_params["up_proj"],
        ],
        axis=1,
    )


def _materialize_tied_lm_head_enabled(config: Qwen3_5Config | None = None) -> bool:
    """Materialize tied embeddings as a separate [hidden, vocab] LM-head leaf."""
    if config is not None and hasattr(config, "materialize_tied_lm_head"):
        return bool(config.materialize_tied_lm_head)
    return False


def download_hf_weights(model_name: str, cache_dir: str = None):
    """Download or reuse a Hugging Face snapshot."""
    from huggingface_hub import snapshot_download

    print(f"Resolving {model_name} from Hugging Face cache...")
    path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "tokenizer.*",
            "vocab.*",
            "merges.txt",
            "generation_config.json",
        ],
    )
    print(f"Using snapshot: {path}")
    return Path(path)

def load_safetensors(model_path: Path):
    """Load weights from safetensors files."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to load model weights. Install the package "
            "with `pip install -e .` or `pip install safetensors`."
        ) from exc

    weights = {}
    for st_file in model_path.glob("*.safetensors"):
        print(f"  Loading {st_file.name}...")
        with safe_open(st_file, framework="np") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    return weights


class _SafeTensorReader:
    """Small random-access wrapper that does not keep all checkpoint tensors live."""

    def __init__(self, model_path: Path):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load model weights. Install the package "
                "with `pip install -e .` or `pip install safetensors`."
            ) from exc

        self._safe_open = safe_open
        self._key_to_file = {}
        for st_file in model_path.glob("*.safetensors"):
            with safe_open(st_file, framework="np") as f:
                for key in f.keys():
                    normalized = _normalize_hf_key(key)
                    self._key_to_file[normalized] = (st_file, key)
                    self._key_to_file[key] = (st_file, key)

    def get(self, key: str):
        try:
            st_file, real_key = self._key_to_file[key]
        except KeyError as exc:
            raise KeyError(f"Weight {key!r} not found in safetensors checkpoint") from exc
        with self._safe_open(st_file, framework="np") as f:
            return f.get_tensor(real_key)

    def has(self, key: str) -> bool:
        return key in self._key_to_file


def _normalize_hf_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return key[21:]
    if key.startswith("model."):
        return key[6:]
    return key


def _checkpoint_dtypes(config: Qwen3_5Config):
    import ml_dtypes

    target_dtype = config.get_dtype()
    if target_dtype == jnp.bfloat16:
        return ml_dtypes.bfloat16, jnp.bfloat16
    if target_dtype == jnp.float16:
        return np.float16, jnp.float16
    return np.float32, jnp.float32


def _to_jax_weight(
    reader: _SafeTensorReader,
    key: str,
    config: Qwen3_5Config,
    *,
    transpose: bool = False,
    squeeze_axis: int | None = None,
    exp: bool = False,
):
    np_dtype, jax_dtype = _checkpoint_dtypes(config)
    value = np.asarray(reader.get(key))
    if squeeze_axis is not None and value.ndim > squeeze_axis:
        value = np.squeeze(value, axis=squeeze_axis)
    if transpose:
        value = value.T
    if value.dtype != np_dtype:
        if jax_dtype == jnp.bfloat16:
            value = value.astype(np.float32).astype(np_dtype)
        else:
            value = value.astype(np_dtype)
    if transpose and not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    if exp:
        # HF stores A_log in the checkpoint dtype but computes
        # A_log.float().exp() at runtime.  Keep that derived parameter in FP32
        # so BF16 checkpoint loading does not round exp(A_log) itself.
        return jnp.exp(jnp.array(value, dtype=jax_dtype).astype(jnp.float32))
    arr = jnp.array(value, dtype=jax_dtype)
    return arr


def load_weights_from_hf_streaming(
    model_name: str,
    config: Qwen3_5Config,
    *,
    verbose: bool = False,
    cache_dir: str = None,
) -> ModelParams:
    """Load HF weights one tensor at a time to keep peak memory bounded."""
    if config is None:
        raise ValueError("config is required - cannot be None")

    print(f"Loading weights for {model_name}...")
    hf_path = download_hf_weights(model_name, cache_dir=cache_dir)
    reader = _SafeTensorReader(hf_path)

    print("Converting weights...")
    embed_tokens = _to_jax_weight(reader, "embed_tokens.weight", config)

    layers = []
    for i in range(config.num_hidden_layers):
        layer_prefix = f"layers.{i}."
        layer_params = {}
        layer_type = config.layer_types[i]

        if layer_type == "full_attention":
            layer_params["q_proj"] = _to_jax_weight(reader, f"{layer_prefix}self_attn.q_proj.weight", config, transpose=True)
            layer_params["k_proj"] = _to_jax_weight(reader, f"{layer_prefix}self_attn.k_proj.weight", config, transpose=True)
            layer_params["v_proj"] = _to_jax_weight(reader, f"{layer_prefix}self_attn.v_proj.weight", config, transpose=True)
            layer_params["o_proj"] = _to_jax_weight(reader, f"{layer_prefix}self_attn.o_proj.weight", config, transpose=True)
            layer_params["q_norm"] = _to_jax_weight(reader, f"{layer_prefix}self_attn.q_norm.weight", config)
            layer_params["k_norm"] = _to_jax_weight(reader, f"{layer_prefix}self_attn.k_norm.weight", config)
            layer_params["input_norm"] = _to_jax_weight(reader, f"{layer_prefix}input_layernorm.weight", config)
            layer_params["post_attn_norm"] = _to_jax_weight(reader, f"{layer_prefix}post_attention_layernorm.weight", config)
            layer_params["gate_proj"] = _to_jax_weight(reader, f"{layer_prefix}mlp.gate_proj.weight", config, transpose=True)
            layer_params["up_proj"] = _to_jax_weight(reader, f"{layer_prefix}mlp.up_proj.weight", config, transpose=True)
            layer_params["down_proj"] = _to_jax_weight(reader, f"{layer_prefix}mlp.down_proj.weight", config, transpose=True)
            layer_params["ffn_norm"] = _to_jax_weight(reader, f"{layer_prefix}post_attention_layernorm.weight", config)
            _add_full_attention_decode_packed_qkv(layer_params)
            _add_mlp_packed_gate_up(layer_params)
        else:
            linear_prefix = f"{layer_prefix}linear_attn."
            layer_params["in_proj_qkv"] = _to_jax_weight(reader, f"{linear_prefix}in_proj_qkv.weight", config, transpose=True)
            layer_params["in_proj_a"] = _to_jax_weight(reader, f"{linear_prefix}in_proj_a.weight", config, transpose=True)
            layer_params["in_proj_b"] = _to_jax_weight(reader, f"{linear_prefix}in_proj_b.weight", config, transpose=True)
            layer_params["in_proj_z"] = _to_jax_weight(reader, f"{linear_prefix}in_proj_z.weight", config, transpose=True)
            layer_params["conv1d_weight"] = _to_jax_weight(reader, f"{linear_prefix}conv1d.weight", config, squeeze_axis=1)
            layer_params["dt_bias"] = _to_jax_weight(reader, f"{linear_prefix}dt_bias", config)
            layer_params["A"] = _to_jax_weight(reader, f"{linear_prefix}A_log", config, exp=True)
            layer_params["norm_weight"] = _to_jax_weight(reader, f"{linear_prefix}norm.weight", config)
            layer_params["out_proj"] = _to_jax_weight(reader, f"{linear_prefix}out_proj.weight", config, transpose=True)
            layer_params["input_norm"] = _to_jax_weight(reader, f"{layer_prefix}input_layernorm.weight", config)
            layer_params["ffn_norm"] = _to_jax_weight(reader, f"{layer_prefix}post_attention_layernorm.weight", config)
            layer_params["gate_proj"] = _to_jax_weight(reader, f"{layer_prefix}mlp.gate_proj.weight", config, transpose=True)
            layer_params["up_proj"] = _to_jax_weight(reader, f"{layer_prefix}mlp.up_proj.weight", config, transpose=True)
            layer_params["down_proj"] = _to_jax_weight(reader, f"{layer_prefix}mlp.down_proj.weight", config, transpose=True)
            _add_gdn_decode_packed_in_proj(layer_params)
            _add_mlp_packed_gate_up(layer_params)

        layers.append(layer_params)
        if verbose:
            print(f"  converted layer {i}: {layer_type}")

    norm_weight = _to_jax_weight(reader, "norm.weight", config) if reader.has("norm.weight") else jnp.ones(config.hidden_size)
    if reader.has("lm_head.weight"):
        lm_head = _to_jax_weight(reader, "lm_head.weight", config, transpose=True)
    elif config.tie_word_embeddings and _materialize_tied_lm_head_enabled(config):
        print("Materializing tied LM head as a separate [hidden, vocab] weight...")
        lm_head = _to_jax_weight(reader, "embed_tokens.weight", config, transpose=True)
    else:
        lm_head = None

    print(f"✓ Loaded weights: {len(layers)} layers")
    return ModelParams(
        embed_tokens=embed_tokens,
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
    )


def convert_hf_to_jax(hf_weights: dict, config: Qwen3_5Config, verbose: bool = False) -> ModelParams:
    """Convert HuggingFace weights to JAX format for Qwen 3.5."""
    print("Converting weights...")
    if not verbose:
        import warnings
        warnings.filterwarnings('ignore')

    import ml_dtypes

    # Get target dtype from config
    target_dtype = config.get_dtype()
    if target_dtype == jnp.bfloat16:
        # Use ml_dtypes for bfloat16 support
        np_dtype = ml_dtypes.bfloat16
        jax_dtype = jnp.bfloat16
    elif target_dtype == jnp.float16:
        np_dtype = np.float16
        jax_dtype = jnp.float16
    else:
        np_dtype = np.float32
        jax_dtype = jnp.float32

    # Extract text model weights (prefix "model.language_model." or "model.")
    text_weights = {}

    for key, value in hf_weights.items():
        value_np = np.asarray(value)

        # Convert to target dtype
        if value_np.dtype != np_dtype:
            # Convert via float32 intermediate for bfloat16
            if np_dtype == ml_dtypes.bfloat16:
                value_np = value_np.astype(np.float32).astype(np_dtype)
            else:
                value_np = value_np.astype(np_dtype)

        value_jax = jnp.array(value_np, dtype=jax_dtype)

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
            _add_full_attention_decode_packed_qkv(layer_params)
            _add_mlp_packed_gate_up(layer_params)

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

            # A_log: [16] -> exp to get A. HF computes A_log.float().exp()
            # at runtime, so the derived A stays FP32 even for BF16 weights.
            layer_params["A"] = jnp.exp(text_weights[f"{linear_prefix}A_log"].astype(jnp.float32))

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
            _add_gdn_decode_packed_in_proj(layer_params)
            _add_mlp_packed_gate_up(layer_params)

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
        # Keep tied weights implicit by default. Materializing embed_tokens.T costs
        # about 485 MiB for Qwen3.5-0.8B, but can be profiled as an opt-in layout.
        lm_head = jnp.array(embed_tokens.T, copy=True) if _materialize_tied_lm_head_enabled(config) else None

    return ModelParams(
        embed_tokens=embed_tokens,
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
    )


def load_weights_from_hf(
    model_name: str,
    config: Qwen3_5Config,
    *,
    verbose: bool = False,
    cache_dir: str = None,
) -> ModelParams:
    """Load weights from HuggingFace for Qwen3.5 model.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3.5-0.8B")
        config: Model configuration (REQUIRED)
        verbose: Whether to print detailed weight info
        cache_dir: Optional Hugging Face cache directory

    Returns:
        ModelParams with loaded serving weights

    Raises:
        ValueError: If model not found in cache or weights invalid
        RuntimeError: If weight conversion fails
    """
    if config is None:
        raise ValueError("config is required - cannot be None")

    print(f"Loading weights for {model_name}...")

    # Download from HF
    hf_path = download_hf_weights(model_name, cache_dir=cache_dir)

    # Load safetensors
    hf_weights = load_safetensors(hf_path)

    # Convert to JAX format
    jax_params = convert_hf_to_jax(hf_weights, config, verbose=verbose)

    print(f"✓ Loaded weights: {len(jax_params.layers)} layers")
    return jax_params
