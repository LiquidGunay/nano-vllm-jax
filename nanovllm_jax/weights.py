"""Load pretrained Hugging Face weights into the serving parameter tree."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nanovllm_jax.config import RuntimeConfig
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
    gate = layer_params.pop("gate_proj")
    up = layer_params.pop("up_proj")
    layer_params[MLP_GATE_UP_PACKED_KEY] = jnp.concatenate([gate, up], axis=1)


_METADATA_PATTERNS = (
    "*.json",
    "*.model",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
)


def _local_checkpoint(model: str | Path) -> Path | None:
    local = Path(model).expanduser()
    if not local.exists():
        return None
    if not local.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {local}")
    return local.resolve()


def _download_snapshot(
    model: str | Path,
    *,
    cache_dir: str | None,
    allow_patterns: tuple[str, ...],
    revision: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=str(model),
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )
    return Path(path)


def resolve_checkpoint_metadata(model: str | Path, cache_dir: str | None = None) -> Path:
    """Resolve config/tokenizer files without downloading weight shards."""
    local = _local_checkpoint(model)
    if local is not None:
        return local
    print(f"Resolving {model} metadata from Hugging Face cache...")
    path = _download_snapshot(
        model,
        cache_dir=cache_dir,
        allow_patterns=_METADATA_PATTERNS,
    )
    print(f"Using metadata snapshot: {path}")
    return path


def resolve_checkpoint(
    model: str | Path,
    cache_dir: str | None = None,
    *,
    revision: str | None = None,
) -> Path:
    """Resolve a local checkpoint or download one validated Hub revision."""
    local = _local_checkpoint(model)
    if local is not None:
        return local

    print(f"Resolving {model} weights from Hugging Face cache...")
    path = _download_snapshot(
        model,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=(
            *_METADATA_PATTERNS,
            "*.safetensors",
        ),
    )
    print(f"Using snapshot: {path}")
    return path


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
        if not self._key_to_file:
            raise ValueError(f"checkpoint contains no safetensors weights: {model_path}")

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


def _checkpoint_dtypes(config: RuntimeConfig):
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
    config: RuntimeConfig,
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


def _expect_shape(name: str, value: jnp.ndarray, expected: tuple[int, ...]) -> None:
    actual = tuple(int(size) for size in value.shape)
    if actual != expected:
        raise ValueError(f"{name} has shape {actual}, expected {expected}")


def _validate_params(params: ModelParams, config: RuntimeConfig) -> None:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    mixed_dim = 2 * key_dim + value_dim
    query_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    _expect_shape("embed_tokens", params.embed_tokens, (config.vocab_size, hidden))
    _expect_shape("norm_weight", params.norm_weight, (hidden,))
    if params.lm_head is not None:
        _expect_shape("lm_head", params.lm_head, (config.vocab_size, hidden))
    if len(params.layers) != config.num_hidden_layers:
        raise ValueError("checkpoint layer count does not match model config")

    for index, (layer_type, layer) in enumerate(zip(config.layer_types, params.layers)):
        prefix = f"layers.{index}"
        common = {
            "input_norm": (hidden,),
            "ffn_norm": (hidden,),
            "down_proj": (intermediate, hidden),
            MLP_GATE_UP_PACKED_KEY: (hidden, 2 * intermediate),
        }
        expected = dict(common)
        if layer_type == "full_attention":
            expected.update(
                {
                    "q_proj": (hidden, 2 * query_dim),
                    "k_proj": (hidden, kv_dim),
                    "v_proj": (hidden, kv_dim),
                    "o_proj": (query_dim, hidden),
                    "q_norm": (config.head_dim,),
                    "k_norm": (config.head_dim,),
                    FULL_ATTN_DECODE_QKV_PACKED_KEY: (hidden, 2 * query_dim + 2 * kv_dim),
                }
            )
        else:
            expected.update(
                {
                    "in_proj_qkv": (hidden, mixed_dim),
                    "in_proj_a": (hidden, config.linear_num_value_heads),
                    "in_proj_b": (hidden, config.linear_num_value_heads),
                    "in_proj_z": (hidden, value_dim),
                    "conv1d_weight": (mixed_dim, config.linear_conv_kernel_size),
                    "dt_bias": (config.linear_num_value_heads,),
                    "A": (config.linear_num_value_heads,),
                    "norm_weight": (config.linear_value_head_dim,),
                    "out_proj": (value_dim, hidden),
                    GDN_DECODE_IN_PROJ_PACKED_KEY: (
                        hidden,
                        mixed_dim + 2 * config.linear_num_value_heads + value_dim,
                    ),
                }
            )
        for name, shape in expected.items():
            if name not in layer:
                raise ValueError(f"{prefix} is missing {name}")
            _expect_shape(f"{prefix}.{name}", layer[name], shape)


def load_weights_from_hf_streaming(
    model: str | Path,
    config: RuntimeConfig,
    *,
    verbose: bool = False,
    cache_dir: str = None,
) -> ModelParams:
    """Load HF weights one tensor at a time to keep peak memory bounded."""
    if config is None:
        raise ValueError("config is required - cannot be None")

    hf_path = resolve_checkpoint(model, cache_dir=cache_dir)
    print(f"Loading weights from {hf_path}...")
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

    norm_weight = _to_jax_weight(reader, "norm.weight", config)
    if reader.has("lm_head.weight"):
        lm_head = _to_jax_weight(reader, "lm_head.weight", config)
    elif config.tie_word_embeddings:
        lm_head = None
    else:
        raise ValueError("untied checkpoint is missing lm_head.weight")

    print(f"✓ Loaded weights: {len(layers)} layers")
    params = ModelParams(
        embed_tokens=embed_tokens,
        layers=layers,
        norm_weight=norm_weight,
        lm_head=lm_head,
    )
    _validate_params(params, config)
    return params
