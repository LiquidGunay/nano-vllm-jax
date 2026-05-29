#!/usr/bin/env python3
"""Precompute HuggingFace greedy references for benchmark prompt rows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_paths import default_runtime_root

DEFAULT_OUTPUT_JSON = "results/qwen08_hf_bf16w_fp32act_long_prefill_512_2048x16.json"


def configure_hf_paths(root: Path | None = None) -> dict[str, str]:
    root = root or default_runtime_root()
    values = {
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / ".cache"),
        "HF_HOME": str(root / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(root / ".cache" / "huggingface" / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)
    for key, value in values.items():
        if key != "TOKENIZERS_PARALLELISM":
            Path(value).mkdir(parents=True, exist_ok=True)
    return {key: os.environ[key] for key in values}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--weight-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--input-lens", default="512,1024,1536,2048")
    parser.add_argument("--output-len", type=_positive_int, default=16)
    parser.add_argument("--output-lengths", default="")
    parser.add_argument("--prompt-suite", choices=["synthetic", "real", "mixed", "server_shapes"], default="mixed")
    parser.add_argument(
        "--prompt-source",
        choices=["tokenized_seed_repeat", "manifest", "vllm_random"],
        default="tokenized_seed_repeat",
    )
    parser.add_argument("--prompt-manifest-jsonl", default="")
    parser.add_argument("--prompt-manifest-output-jsonl", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--num-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-input-len", type=_positive_int, default=1280)
    parser.add_argument("--random-output-len", type=_positive_int, default=16)
    parser.add_argument("--random-range-ratio", default='{"input":0.0,"output":0.0}')
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _torch_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _pad_token_id(tokenizer) -> int | None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return None


def _topk_rows(scores, top_k: int) -> list[list[dict[str, Any]]]:
    import torch

    rows = []
    for score in scores:
        logits = score.float()
        values, ids = torch.topk(logits, k=int(top_k), dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, ids)
        rows.append(
            [
                {
                    "token_id": int(token_id),
                    "logprob": float(logprob),
                    "logit": float(logit),
                }
                for token_id, logprob, logit in zip(
                    ids[0].detach().cpu().tolist(),
                    log_probs[0].detach().cpu().tolist(),
                    values[0].detach().cpu().tolist(),
                )
            ]
        )
    return rows


def _prepare_prompt_rows(tokenizer, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from benchmarks.benchmark_vllm_qwen35 import prepare_prompt_rows

    return prepare_prompt_rows(tokenizer, args)


def generate_reference(args: argparse.Namespace) -> dict[str, Any]:
    env_paths = configure_hf_paths()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HF prompt reference generation")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_rows, prompt_info = _prepare_prompt_rows(tokenizer, args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.weight_dtype),
        trust_remote_code=True,
    )
    if args.dtype == "float32":
        model.float()
    elif args.dtype != args.weight_dtype:
        model.to(dtype=_torch_dtype(args.dtype))
    model.eval().to("cuda")

    rows = []
    total_tokens = 0
    started = time.perf_counter()
    pad_token_id = _pad_token_id(tokenizer)
    with torch.no_grad():
        for prompt in prompt_rows:
            input_ids = torch.tensor(
                [prompt["input_ids"]],
                dtype=torch.long,
                device="cuda",
            )
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(prompt["output_len"]),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_token_id,
            )
            generated = (
                output.sequences[0, input_ids.shape[1] :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
                .tolist()
            )
            total_tokens += len(generated)
            rows.append(
                {
                    "name": prompt["name"],
                    "prompt_length": int(prompt["prompt_length"]),
                    "output_len": int(prompt["output_len"]),
                    "generated_token_ids": [int(token) for token in generated],
                    "generated_tokens": len(generated),
                    "topk_logprobs_by_step": _topk_rows(output.scores, args.top_k),
                }
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    summary = {
        "environment": {
            "env_paths": env_paths,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "run_config": {
            "model": args.model,
            "dtype": args.dtype,
            "weight_dtype": args.weight_dtype,
            **prompt_info,
            "top_k": int(args.top_k),
            "reference_source": "huggingface_generate_per_request",
        },
        "performance": {
            "seconds": elapsed,
            "generated_tokens": total_tokens,
            "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        },
        "rows": rows,
    }
    return _json_safe(summary)


def write_reference(summary: dict[str, Any], output_json: str) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = generate_reference(args)
    output_path = write_reference(summary, args.output_json)
    performance = summary["performance"]
    print(f"wrote_json={output_path}")
    print(f"generated_tokens={performance['generated_tokens']}")
    print(f"seconds={performance['seconds']:.3f}")


if __name__ == "__main__":
    main()
