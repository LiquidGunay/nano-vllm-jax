"""Long decode top-5 logits parity against a cached HuggingFace reference.

The HuggingFace pass is intentionally separate and stored on disk so repeated
JAX correctness runs do not need to reload and regenerate the reference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from runtime_paths import configure_compilation_cache

configure_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.model_executor import ModelExecutor
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import KVCacheSpec, init_hybrid_state
from nanovllm_jax.load_weights import load_weights_from_hf


jax.config.update("jax_default_matmul_precision", "highest")

REQUIRED_MAX_HF_TOPK_ID_LOGIT_DIFF = 2e-5


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="The future of artificial intelligence is poised to revolutionize")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--hf-reference",
        default="results/qwen08_hf_bf16w_fp32act_long_decode_top5_500.npz",
        help="Path for cached HF generated tokens and top-k logits.",
    )
    parser.add_argument(
        "--compare-json",
        default="results/qwen08_jax_bf16w_fp32act_long_decode_top5_compare.json",
        help="Path for JAX-vs-HF comparison summary.",
    )
    parser.add_argument("--refresh-hf", action="store_true")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--backend", default="gpu")
    return parser.parse_args()


def _metadata_array(metadata: dict) -> np.ndarray:
    return np.array(json.dumps(metadata, sort_keys=True))


def _load_metadata(npz) -> dict:
    return json.loads(str(npz["metadata"]))


def precompute_hf_reference(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("HF long-decode reference requires CUDA")

    output_path = Path(args.hf_reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.float().to("cuda")
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            encoded["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started

    top_ids = []
    top_values = []
    for score in output.scores:
        values, ids = torch.topk(score.float(), k=args.top_k, dim=-1)
        top_ids.append(ids[0].cpu().numpy().astype(np.int32))
        top_values.append(values[0].cpu().numpy().astype(np.float32))

    sequences = output.sequences[0].cpu().numpy().astype(np.int32)
    prompt_ids = encoded["input_ids"][0].detach().cpu().numpy().astype(np.int32)
    generated_ids = sequences[len(prompt_ids) :]

    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "hf_weight_dtype": "bfloat16",
        "hf_activation_dtype": "float32",
        "elapsed_seconds": elapsed,
        "created_unix": time.time(),
    }
    np.savez_compressed(
        output_path,
        metadata=_metadata_array(metadata),
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        hf_top_ids=np.stack(top_ids, axis=0),
        hf_top_values=np.stack(top_values, axis=0),
    )
    print(f"wrote_hf_reference={output_path}")
    print(f"hf_elapsed_seconds={elapsed:.3f}")
    print(f"generated_tokens={len(generated_ids)}")
    del model
    torch.cuda.empty_cache()


def _qwen08_config() -> Qwen3_5Config:
    return Qwen3_5Config.qwen3_5_0_8b()


def _make_batch(
    *,
    tokens: np.ndarray | list[int],
    positions: np.ndarray | list[int],
    block_table: np.ndarray,
    seq_len: int,
    is_prefill: bool,
) -> ScheduledBatch:
    tokens_np = np.asarray(tokens, dtype=np.int32)[None, :]
    positions_np = np.asarray(positions, dtype=np.int32)[None, :]
    query_len = int(tokens_np.shape[1])
    return ScheduledBatch(
        tokens=jnp.array(tokens_np, dtype=jnp.int32),
        positions=jnp.array(positions_np, dtype=jnp.int32),
        seq_ids=jnp.array([0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, query_len], dtype=jnp.int32),
        is_prefill=is_prefill,
        num_prefill_tokens=query_len if is_prefill else 0,
        num_decode_tokens=0 if is_prefill else query_len,
        block_tables=jnp.array(block_table[None, :], dtype=jnp.int32),
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
        prefill_is_final=[True] if is_prefill else None,
    )


def compare_jax_to_hf(args: argparse.Namespace) -> None:
    reference_path = Path(args.hf_reference)
    if not reference_path.exists():
        raise FileNotFoundError(f"HF reference does not exist: {reference_path}")

    with np.load(reference_path, allow_pickle=False) as ref:
        metadata = _load_metadata(ref)
        prompt_ids = ref["prompt_ids"].astype(np.int32)
        generated_ids = ref["generated_ids"].astype(np.int32)
        hf_top_ids = ref["hf_top_ids"].astype(np.int32)
        hf_top_values = ref["hf_top_values"].astype(np.float32)

    if len(generated_ids) < args.max_new_tokens:
        raise ValueError(
            f"HF reference has {len(generated_ids)} tokens, requested {args.max_new_tokens}"
        )

    load_config = _qwen08_config()
    load_config.dtype = "bfloat16"
    params = load_weights_from_hf(args.model, load_config)

    config = _qwen08_config()
    config.dtype = "float32"
    config.max_num_seqs = 1
    total_tokens = int(len(prompt_ids) + args.max_new_tokens)
    max_blocks = (total_tokens + config.block_size - 1) // config.block_size + 1
    config.max_blocks_per_seq = max_blocks
    config.num_kvcache_blocks = max_blocks
    config.max_kv_cache_bytes = None

    executor = ModelExecutor(config, params, backend=args.backend)
    kv_spec = KVCacheSpec(
        num_layers=config.num_hidden_layers,
        num_blocks=config.num_kvcache_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.get_dtype(),
        max_kv_cache_bytes=config.max_kv_cache_bytes,
    )
    cache_storage = executor.backend.allocate_kv_cache(
        kv_spec,
        max_seqs=1,
        max_blocks_per_seq=max_blocks,
    )
    hybrid_state = init_hybrid_state(config, batch_size=1, dtype=config.get_dtype())
    block_table = np.arange(max_blocks, dtype=np.int32)

    def topk_from_logits(logits):
        values, ids = jax.lax.top_k(logits[0, 0].astype(jnp.float32), args.top_k)
        return np.array(ids), np.array(values)

    started = time.perf_counter()
    mismatches = []
    top1_mismatches = []
    topk_set_mismatches = []
    max_top1_abs_logit_diff = 0.0
    max_ranked_logit_diff = 0.0
    max_hf_topk_id_logit_diff = 0.0

    prefill = _make_batch(
        tokens=prompt_ids,
        positions=np.arange(len(prompt_ids), dtype=np.int32),
        block_table=block_table,
        seq_len=len(prompt_ids),
        is_prefill=True,
    )
    output = executor.forward_step_jit(
        prefill,
        cache_storage=cache_storage,
        hybrid_state=hybrid_state,
        last_logits_only=True,
    )
    cache_storage = output.cache_storage
    hybrid_state = output.hybrid_state

    for step in range(args.max_new_tokens):
        if step > 0:
            token = int(generated_ids[step - 1])
            pos = len(prompt_ids) + step - 1
            decode = _make_batch(
                tokens=[token],
                positions=[pos],
                block_table=block_table,
                seq_len=len(prompt_ids) + step,
                is_prefill=False,
            )
            output = executor.forward_step_jit(
                decode,
                cache_storage=cache_storage,
                hybrid_state=hybrid_state,
                last_logits_only=True,
            )
            cache_storage = output.cache_storage
            hybrid_state = output.hybrid_state

        jax_ids, jax_values = topk_from_logits(output.activations)
        hf_ids = hf_top_ids[step]
        hf_values = hf_top_values[step]
        jax_values_at_hf_ids = np.array(
            output.activations[0, 0, jnp.array(hf_ids, dtype=jnp.int32)].astype(jnp.float32)
        )
        ranked_diff = float(np.max(np.abs(jax_values - hf_values)))
        hf_id_diff = float(np.max(np.abs(jax_values_at_hf_ids - hf_values)))
        max_ranked_logit_diff = max(max_ranked_logit_diff, ranked_diff)
        max_hf_topk_id_logit_diff = max(max_hf_topk_id_logit_diff, hf_id_diff)
        if int(jax_ids[0]) == int(hf_ids[0]):
            max_top1_abs_logit_diff = max(max_top1_abs_logit_diff, abs(float(jax_values[0] - hf_values[0])))
        else:
            top1_mismatches.append(
                {
                    "step": step,
                    "prefix_len": int(len(prompt_ids) + step),
                    "hf_top_id": int(hf_ids[0]),
                    "jax_top_id": int(jax_ids[0]),
                    "hf_top_value": float(hf_values[0]),
                    "jax_top_value": float(jax_values[0]),
                }
            )
        same_top5_set = set(map(int, jax_ids.tolist())) == set(map(int, hf_ids.tolist()))
        if not same_top5_set:
            topk_set_mismatches.append(
                {
                    "step": step,
                    "prefix_len": int(len(prompt_ids) + step),
                    "hf_top_ids": hf_ids.tolist(),
                    "jax_top_ids": jax_ids.tolist(),
                }
            )
        if not np.array_equal(jax_ids, hf_ids):
            mismatches.append(
                {
                    "step": step,
                    "prefix_len": int(len(prompt_ids) + step),
                    "hf_top_ids": hf_ids.tolist(),
                    "jax_top_ids": jax_ids.tolist(),
                    "hf_top_values": hf_values.tolist(),
                    "jax_top_values": jax_values.tolist(),
                    "ranked_logit_max_abs_diff": ranked_diff,
                    "hf_top_id_logit_max_abs_diff": hf_id_diff,
                    "same_top5_set": same_top5_set,
                }
            )

        if (step + 1) % 50 == 0:
            print(f"checked_steps={step + 1} mismatches={len(mismatches)}")

    elapsed = time.perf_counter() - started
    top1_exact_matches = int(args.max_new_tokens - len(top1_mismatches))
    ordered_top5_exact_matches = int(args.max_new_tokens - len(mismatches))
    top5_set_exact_matches = int(args.max_new_tokens - len(topk_set_mismatches))
    exact_generated_token_match = bool(top1_exact_matches == args.max_new_tokens)
    ordered_top5_match = bool(ordered_top5_exact_matches == args.max_new_tokens)
    top5_set_match = bool(top5_set_exact_matches == args.max_new_tokens)
    logit_diff_within_gate = bool(
        max_hf_topk_id_logit_diff <= REQUIRED_MAX_HF_TOPK_ID_LOGIT_DIFF
    )
    summary = {
        "model": args.model,
        "prompt": metadata.get("prompt", args.prompt),
        "steps_checked": int(args.max_new_tokens),
        "top_k": int(args.top_k),
        "top1_exact_matches": top1_exact_matches,
        "top1_mismatches": int(len(top1_mismatches)),
        "ordered_top5_exact_matches": ordered_top5_exact_matches,
        "ordered_top5_mismatches": int(len(mismatches)),
        "top5_set_exact_matches": top5_set_exact_matches,
        "top5_set_mismatches": int(len(topk_set_mismatches)),
        "first_ordered_top5_mismatches": mismatches[:20],
        "first_top1_mismatches": top1_mismatches[:20],
        "first_top5_set_mismatches": topk_set_mismatches[:20],
        "max_top1_abs_logit_diff_when_id_matches": max_top1_abs_logit_diff,
        "max_ranked_topk_logit_diff": max_ranked_logit_diff,
        "max_hf_topk_id_logit_diff": max_hf_topk_id_logit_diff,
        "elapsed_seconds": elapsed,
        "jax_weight_dtype": "bfloat16",
        "jax_activation_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "gdn_kernel_flags": {
            "prefill_post_conv_impl": os.environ.get(
                "NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL",
                "off",
            ),
            "prefill_act_dtype": os.environ.get(
                "NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE",
                "fp32",
            ),
        },
        "git_head": _git_head(),
        "hf_reference": str(reference_path),
        "guardrail": {
            "exact_generated_token_match": exact_generated_token_match,
            "top1_exact_matches_required": int(args.max_new_tokens),
            "top1_exact_match": exact_generated_token_match,
            "ordered_top5_exact_matches_required": int(args.max_new_tokens),
            "ordered_top5_exact_match": ordered_top5_match,
            "top5_set_exact_matches_required": int(args.max_new_tokens),
            "top5_set_exact_match": top5_set_match,
            "max_hf_topk_id_logit_diff_lte": REQUIRED_MAX_HF_TOPK_ID_LOGIT_DIFF,
            "max_hf_topk_id_logit_diff_within_gate": logit_diff_within_gate,
            "passes_required_gate": bool(
                exact_generated_token_match
                and ordered_top5_match
                and top5_set_match
                and logit_diff_within_gate
            ),
        },
    }

    compare_path = Path(args.compare_json)
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_compare_json={compare_path}")


def main() -> None:
    args = parse_args()
    reference_path = Path(args.hf_reference)
    if args.refresh_hf or not reference_path.exists():
        precompute_hf_reference(args)
    else:
        print(f"using_hf_reference={reference_path}")
    if not args.skip_jax:
        compare_jax_to_hf(args)


if __name__ == "__main__":
    main()
