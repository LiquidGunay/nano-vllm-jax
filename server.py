#!/usr/bin/env python3
"""Flask API server backed by the canonical LLMEngine path."""

from __future__ import annotations

import argparse
import os
from threading import Lock
import time
import traceback
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

from flask import Flask, jsonify, request

from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.sequence import SamplingParams


app = Flask(__name__)
app.config.setdefault("MAX_TOKENS_DEFAULT", 16)
engine: LLMEngine | None = None
engine_lock = Lock()


def _parse_buckets(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    buckets = tuple(int(part) for part in value.split(",") if part.strip())
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("bucket sizes must be positive integers")
    return buckets


def _validate_server_args(args):
    prefill_buckets = _parse_buckets(args.prefill_buckets)
    batch_size_buckets = _parse_buckets(args.batch_size_buckets)
    if batch_size_buckets and max(batch_size_buckets) > args.max_num_seqs:
        raise ValueError("--batch-size-buckets cannot exceed --max-num-seqs")
    if getattr(args, "max_tokens_default", app.config["MAX_TOKENS_DEFAULT"]) <= 0:
        raise ValueError("--max-tokens-default must be positive")
    return prefill_buckets, batch_size_buckets


def _is_token_ids(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(token, int) for token in value)


def _is_batched_token_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and bool(row) and all(isinstance(token, int) for token in row) for row in value)
    )


def _normalize_generation_inputs(data: dict) -> tuple[list[str | list[int]], bool]:
    prompt = data.get("prompt")
    input_ids = data.get("input_ids")
    if prompt is None and input_ids is None:
        raise ValueError("provide either prompt or input_ids")
    if prompt is not None and input_ids is not None:
        raise ValueError("provide only one of prompt or input_ids")

    if input_ids is not None:
        if _is_token_ids(input_ids):
            return [input_ids], False
        if _is_batched_token_ids(input_ids):
            return input_ids, True
        raise ValueError("input_ids must be a non-empty list of token ids or list of token-id lists")

    if isinstance(prompt, str):
        if not prompt:
            raise ValueError("prompt must be non-empty")
        return [prompt], False
    if isinstance(prompt, list) and prompt and all(isinstance(item, str) and item for item in prompt):
        return prompt, True
    raise ValueError("prompt must be a non-empty string or list of strings")


def _token_counts(inputs: list[str | list[int]]) -> list[int]:
    if engine is None:
        raise RuntimeError("Model is not loaded")
    return [len(item) if isinstance(item, list) else len(engine._tokenize(item)) for item in inputs]


def _validate_inputs_fit_config(inputs: list[str | list[int]], prompt_tokens: list[int], max_tokens: int):
    if engine is None:
        raise RuntimeError("Model is not loaded")

    if prompt_tokens:
        max_total_tokens = max(prompt_tokens) + max_tokens
        max_blocks_per_seq = getattr(engine.config, "max_blocks_per_seq", None)
        if max_blocks_per_seq is not None:
            max_tokens_per_seq = max_blocks_per_seq * engine.config.block_size
            if max_total_tokens > max_tokens_per_seq:
                raise ValueError(
                    f"request needs {max_total_tokens} total tokens, exceeding per-sequence KV capacity "
                    f"{max_tokens_per_seq}"
                )

    prefill_buckets = tuple(getattr(engine.config, "prefill_buckets", ()))
    if prefill_buckets and prompt_tokens:
        max_prompt_tokens = max(prompt_tokens)
        max_prefill_bucket = max(prefill_buckets)
        if max_prompt_tokens > max_prefill_bucket:
            raise ValueError(
                f"prompt has {max_prompt_tokens} tokens, exceeding largest prefill bucket {max_prefill_bucket}"
            )

    max_num_seqs = getattr(engine.config, "max_num_seqs", None)
    if max_num_seqs is not None and len(inputs) > max_num_seqs:
        raise ValueError(f"request has {len(inputs)} prompts, exceeding max_num_seqs {max_num_seqs}")
    batch_size_buckets = tuple(getattr(engine.config, "batch_size_buckets", ()))
    if batch_size_buckets and len(inputs) > max(batch_size_buckets):
        raise ValueError(
            f"request has {len(inputs)} prompts, exceeding largest batch-size bucket {max(batch_size_buckets)}"
        )


def _sampling_from_request(data: dict) -> tuple[int, float]:
    try:
        max_tokens = int(data.get("max_tokens", app.config["MAX_TOKENS_DEFAULT"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens must be an integer") from exc
    try:
        temperature = float(data.get("temperature", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a number") from exc
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    return max_tokens, temperature


def load_engine(args) -> LLMEngine:
    global engine
    prefill_buckets, batch_size_buckets = _validate_server_args(args)
    max_kv_cache_bytes = int(args.max_kv_cache_mb * 1024 * 1024)
    engine = LLMEngine(
        args.model,
        backend=args.backend,
        dtype=args.dtype,
        max_kv_cache_bytes=max_kv_cache_bytes,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_buckets=prefill_buckets,
        batch_size_buckets=batch_size_buckets,
        jax_execution=args.jax_execution,
        num_speculative_tokens=args.num_speculative_tokens,
    )
    if not args.skip_compile:
        engine.model_runner.warmup_compilation(
            max_prefill_len=max(prefill_buckets or (args.max_prefill,)),
            max_batch=max(batch_size_buckets or (args.max_num_seqs,)),
        )
    return engine


def _run_generation(inputs: list[str | list[int]], max_tokens: int, temperature: float):
    if engine is None:
        raise RuntimeError("Model is not loaded")
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        ignore_eos=False,
    )
    with engine_lock:
        return engine.generate(inputs, sampling_params=sampling, use_tqdm=False)


def _generation_payload(results, prompt_tokens: list[int], elapsed: float, is_batch: bool):
    completion_tokens = [len(result["token_ids"]) for result in results]
    items = [
        {
            "text": result["text"],
            "token_ids": result["token_ids"],
            "new_tokens": result["token_ids"],
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": completion_count,
                "total_tokens": prompt_count + completion_count,
            },
        }
        for result, prompt_count, completion_count in zip(results, prompt_tokens, completion_tokens)
    ]
    stats = {
        "generation_time_ms": int(elapsed * 1000),
        "tokens_per_second": sum(completion_tokens) / elapsed if elapsed > 0 else 0.0,
        "jit_cache_entries": len(engine.model_runner.executor._jit_cache) if engine is not None else 0,
    }
    if engine is not None and hasattr(engine.model_runner, "get_speculative_stats"):
        stats["speculative"] = engine.model_runner.get_speculative_stats()
    if not is_batch:
        payload = dict(items[0])
        payload["stats"] = stats
        return payload
    return {
        "results": items,
        "usage": {
            "prompt_tokens": sum(prompt_tokens),
            "completion_tokens": sum(completion_tokens),
            "total_tokens": sum(prompt_tokens) + sum(completion_tokens),
        },
        "stats": stats,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy" if engine is not None else "loading",
            "model_loaded": engine is not None,
            "jax_execution": getattr(engine.config, "jax_execution", None) if engine is not None else None,
            "jit_cache_entries": len(engine.model_runner.executor._jit_cache) if engine is not None else 0,
        }
    )


@app.route("/v1/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    try:
        inputs, is_batch = _normalize_generation_inputs(data)
        max_tokens, temperature = _sampling_from_request(data)
        prompt_tokens = _token_counts(inputs)
        _validate_inputs_fit_config(inputs, prompt_tokens, max_tokens)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        t0 = time.perf_counter()
        results = _run_generation(inputs, max_tokens=max_tokens, temperature=temperature)
        elapsed = time.perf_counter() - t0
        return jsonify(_generation_payload(results, prompt_tokens, elapsed, is_batch))
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.get_json(force=True) or {}
    try:
        inputs, _ = _normalize_generation_inputs({"prompt": data.get("prompt", "")})
        max_tokens, temperature = _sampling_from_request(data)
        prompt_tokens = _token_counts(inputs)
        _validate_inputs_fit_config(inputs, prompt_tokens, max_tokens)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        t0 = time.perf_counter()
        results = _run_generation(inputs, max_tokens=max_tokens, temperature=temperature)
        elapsed = time.perf_counter() - t0
        completion_tokens = [len(result["token_ids"]) for result in results]
        return jsonify(
            {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": engine.config.__class__.__name__ if engine is not None else "unknown",
                "choices": [
                    {"text": result["text"], "index": idx, "finish_reason": "length"}
                    for idx, result in enumerate(results)
                ],
                "usage": {
                    "prompt_tokens": sum(prompt_tokens),
                    "completion_tokens": sum(completion_tokens),
                    "total_tokens": sum(prompt_tokens) + sum(completion_tokens),
                },
                "stats": {
                    "generation_time_ms": int(elapsed * 1000),
                    "tokens_per_second": sum(completion_tokens) / elapsed if elapsed > 0 else 0.0,
                    "jit_cache_entries": len(engine.model_runner.executor._jit_cache),
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


def main():
    parser = argparse.ArgumentParser(description="nano-vllm-jax LLMEngine API server")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="jit")
    parser.add_argument("--prefill-buckets", default="16")
    parser.add_argument("--batch-size-buckets", default="1")
    parser.add_argument("--max-prefill", type=int, default=16)
    parser.add_argument("--max-kv-cache-mb", type=int, default=64)
    parser.add_argument("--num-kvcache-blocks", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16)
    parser.add_argument("--max-tokens-default", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, choices=[0, 1], default=0)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    app.config["MAX_TOKENS_DEFAULT"] = args.max_tokens_default

    print("nano-vllm-jax LLMEngine server")
    print(f"model={args.model} dtype={args.dtype} execution={args.jax_execution}")
    print(f"prefill_buckets={args.prefill_buckets} batch_size_buckets={args.batch_size_buckets}")
    load_engine(args)
    print(f"server_ready=http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
