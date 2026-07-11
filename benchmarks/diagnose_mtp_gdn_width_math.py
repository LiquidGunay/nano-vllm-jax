#!/usr/bin/env python3
"""Compare grouped and width-1 inputs to the first Qwen3.5 GDN layer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kv_cache import HybridLayerState, init_hybrid_state
from nanovllm_jax.layers import causal_mask
from nanovllm_jax.load_weights import load_weights_from_hf
from nanovllm_jax.model import (
    _decode_width1_rms_norm,
    _force_width1_decode_norms,
    _force_width1_full_attention_projections,
    _force_width1_gdn_input_projections,
    _force_width1_gdn_output_projections,
    _force_width1_mlp_projections,
    gdn_decode_recurrent_input_max_abs,
    transformer_block,
)


_VALUE_NAMES = (
    "input_norm",
    "mixed_qkv",
    "z",
    "a",
    "b",
    "conv_out",
    "query",
    "key",
    "value",
    "gate",
    "beta",
)
_STAGE_NAMES = (
    "block_input",
    "input_norm",
    "mixer_out",
    "attn_residual",
    "ffn_norm",
    "mlp_out",
    "block_output",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--force-width1", choices=("0", "1"), default="0")
    parser.add_argument("--force-width1-norms", choices=("", "0", "1"), default="")
    parser.add_argument("--include-layer", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.width < 2:
        raise ValueError("batch-size must be positive and width must be at least 2")
    os.environ["NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_MATH"] = args.force_width1
    if args.force_width1_norms:
        os.environ["NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_NORMS"] = (
            args.force_width1_norms
        )
    # This probe never projects to vocabulary, so avoid materializing the tied
    # 0.8B LM head while loading the real checkpoint.
    os.environ["NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD"] = "0"

    config = Qwen3_5Config.qwen3_5_0_8b()
    config.dtype = "bfloat16"
    config.num_speculative_tokens = args.width - 1
    config.decode_proj_act_dtype = "bf16"
    config.decode_padded_gemm = True
    config.decode_padded_gemm_gate_up = True
    config.decode_padded_gemm_rows = 8
    config.decode_padded_gemm_max_out_dim = 300000
    config.gdn_packed_decode_impl = "reference"
    config.gdn_packed_decode_qkv_dtype = "bf16"

    load_started = time.perf_counter()
    params = load_weights_from_hf(args.model, config, load_mtp=False, verbose=False)
    layer_params = params.layers[0]
    token_ids = (
        jnp.arange(args.batch_size * args.width, dtype=jnp.int32)
        .reshape(args.batch_size, args.width)
        + jnp.asarray(128, dtype=jnp.int32)
    )
    grouped_x = params.embed_tokens[token_ids].astype(config.get_dtype())
    grouped_x.block_until_ready()
    del params
    gc.collect()
    load_seconds = time.perf_counter() - load_started

    hybrid_state = init_hybrid_state(
        config,
        batch_size=args.batch_size,
        dtype=config.get_dtype(),
    )
    force_width1 = args.force_width1 == "1"
    force_width1_norms = _force_width1_decode_norms()

    def probe(
        layer,
        grouped,
        single,
        conv_state,
        recurrent_state,
    ):
        grouped_norm = _decode_width1_rms_norm(
            grouped,
            layer["input_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norms,
        )
        single_norm = _decode_width1_rms_norm(
            single,
            layer["input_norm"],
            config.rms_norm_eps,
            force_width1=force_width1_norms,
        )
        norm_diff = jnp.max(
            jnp.abs(
                grouped_norm[:, 0].astype(jnp.float32)
                - single_norm[:, 0].astype(jnp.float32)
            )
        )
        gdn_diffs = gdn_decode_recurrent_input_max_abs(
            grouped,
            single,
            layer,
            config,
            layer_idx=0,
            hybrid_state=HybridLayerState(conv_state, recurrent_state),
        )
        return jnp.concatenate([norm_diff[None], gdn_diffs])

    compiled_probe = jax.jit(probe)
    compile_started = time.perf_counter()
    values = compiled_probe(
        layer_params,
        grouped_x,
        grouped_x[:, :1],
        hybrid_state.conv_state,
        hybrid_state.recurrent_state,
    )
    values.block_until_ready()
    compile_seconds = time.perf_counter() - compile_started
    values_host = [float(value) for value in values.tolist()]

    layer_result = None
    if args.include_layer:
        def layer_probe(layer, grouped, conv_state, recurrent_state):
            batch_size, width, _ = grouped.shape
            positions_2d = jnp.broadcast_to(
                jnp.arange(width, dtype=jnp.int32)[None, :],
                (batch_size, width),
            )
            positions = jnp.stack([positions_2d, positions_2d, positions_2d], axis=0)
            initial_prefix = HybridLayerState(
                conv_state=jnp.broadcast_to(
                    conv_state[:, None, ...],
                    (batch_size, width) + conv_state.shape[1:],
                ),
                recurrent_state=jnp.broadcast_to(
                    recurrent_state[:, None, ...],
                    (batch_size, width) + recurrent_state.shape[1:],
                ),
            )
            broad = transformer_block(
                grouped,
                layer,
                positions,
                causal_mask(width, width),
                layer_idx=0,
                config=config,
                hybrid_state=HybridLayerState(conv_state, recurrent_state),
                prefix_hybrid_state=initial_prefix,
                is_prefill=False,
                return_prefix_hybrid=True,
                return_layer_stages=True,
            )
            broad_state = broad[2]
            broad_prefix = broad[3]
            broad_stages = broad[4]

            seq_state = HybridLayerState(conv_state, recurrent_state)
            seq_stages = []
            seq_conv_prefix = []
            seq_recurrent_prefix = []
            for token_idx in range(width):
                token_position = positions[:, :, token_idx : token_idx + 1]
                step = transformer_block(
                    grouped[:, token_idx : token_idx + 1],
                    layer,
                    token_position,
                    causal_mask(1, 1),
                    layer_idx=0,
                    config=config,
                    hybrid_state=seq_state,
                    is_prefill=False,
                    return_layer_stages=True,
                )
                seq_state = step[2]
                seq_stages.append(step[3])
                seq_conv_prefix.append(seq_state.conv_state)
                seq_recurrent_prefix.append(seq_state.recurrent_state)
            seq_stages_value = jnp.concatenate(seq_stages, axis=2)
            seq_conv_prefix_value = jnp.stack(seq_conv_prefix, axis=1)
            seq_recurrent_prefix_value = jnp.stack(seq_recurrent_prefix, axis=1)

            stage_diff = jnp.abs(
                broad_stages.astype(jnp.float32)
                - seq_stages_value.astype(jnp.float32)
            )
            current_stage_max = jnp.max(stage_diff[:, :, :1, :], axis=(1, 2, 3))
            all_stage_max = jnp.max(stage_diff, axis=(1, 2, 3))
            final_conv_max = jnp.max(
                jnp.abs(
                    broad_state.conv_state.astype(jnp.float32)
                    - seq_state.conv_state.astype(jnp.float32)
                )
            )
            final_recurrent_max = jnp.max(
                jnp.abs(
                    broad_state.recurrent_state.astype(jnp.float32)
                    - seq_state.recurrent_state.astype(jnp.float32)
                )
            )
            prefix_conv_max = jnp.max(
                jnp.abs(
                    broad_prefix.conv_state.astype(jnp.float32)
                    - seq_conv_prefix_value.astype(jnp.float32)
                )
            )
            prefix_recurrent_max = jnp.max(
                jnp.abs(
                    broad_prefix.recurrent_state.astype(jnp.float32)
                    - seq_recurrent_prefix_value.astype(jnp.float32)
                )
            )
            return (
                current_stage_max,
                all_stage_max,
                final_conv_max,
                final_recurrent_max,
                prefix_conv_max,
                prefix_recurrent_max,
            )

        compiled_layer_probe = jax.jit(layer_probe)
        layer_started = time.perf_counter()
        (
            current_stage_max,
            all_stage_max,
            final_conv_max,
            final_recurrent_max,
            prefix_conv_max,
            prefix_recurrent_max,
        ) = compiled_layer_probe(
            layer_params,
            grouped_x,
            hybrid_state.conv_state,
            hybrid_state.recurrent_state,
        )
        current_stage_max.block_until_ready()
        layer_result = {
            "compile_and_run_seconds": time.perf_counter() - layer_started,
            "current_token_stage_max_abs": dict(
                zip(_STAGE_NAMES, [float(value) for value in current_stage_max.tolist()])
            ),
            "all_token_stage_max_abs": dict(
                zip(_STAGE_NAMES, [float(value) for value in all_stage_max.tolist()])
            ),
            "final_conv_state_max_abs": float(final_conv_max.item()),
            "final_recurrent_state_max_abs": float(final_recurrent_max.item()),
            "prefix_conv_state_max_abs": float(prefix_conv_max.item()),
            "prefix_recurrent_state_max_abs": float(prefix_recurrent_max.item()),
        }

    result = {
        "model": args.model,
        "batch_size": args.batch_size,
        "width": args.width,
        "force_width1": force_width1,
        "force_width1_norms": force_width1_norms,
        "force_width1_projection_policy": {
            "gdn_input": _force_width1_gdn_input_projections(),
            "gdn_output": _force_width1_gdn_output_projections(),
            "full_attention": _force_width1_full_attention_projections(),
            "mlp": _force_width1_mlp_projections(),
        },
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "load_seconds": load_seconds,
        "compile_and_run_seconds": compile_seconds,
        "max_abs": dict(zip(_VALUE_NAMES, values_host)),
        "layer0": layer_result,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
