#!/usr/bin/env python3
"""Flask API server backed by the canonical LLMEngine path."""

from __future__ import annotations

import argparse
import json
import os
from threading import Lock
import time
import traceback
from typing import Any

from runtime_paths import configure_compilation_cache, configure_xla_flags
from nanovllm_jax.server_config import load_server_config


def _initial_config_path_from_argv() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    return args.config


# Load YAML config before importing the engine/JAX stack.
_server_cfg = load_server_config(_initial_config_path_from_argv())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
configure_xla_flags()
configure_compilation_cache()

from flask import Flask, Response, jsonify, request, stream_with_context

from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.sequence import SamplingParams


app = Flask(__name__)
app.config.setdefault("MAX_TOKENS_DEFAULT", 16)
engine: LLMEngine | None = None
engine_lock = Lock()

_ENGINE_CLI_FIELDS = (
    "model",
    "backend",
    "dtype",
    "weight_dtype",
    "jax_execution",
    "prefill_buckets",
    "prefill_token_buckets",
    "prefill_layout",
    "batch_size_buckets",
    "startup_warmup_prefill_token_buckets",
    "startup_warmup_batch_size_buckets",
    "startup_warmup_decode_block_table_buckets",
    "startup_warmup_include_sampled_routes",
    "max_prefill",
    "max_kv_cache_mb",
    "num_kvcache_blocks",
    "max_num_seqs",
    "max_num_resident_seqs",
    "max_num_batched_tokens",
    "max_blocks_per_seq",
    "prefix_cache",
    "decode_block_table_buckets",
    "speculative_method",
    "draft_sample_method",
    "mtp_verifier_impl",
    "mtp_batch_accept_policy",
    "mtp_seed_after_bonus",
    "mtp_hidden_source",
    "mtp_chain_hidden_source",
    "mtp_chain_mode",
    "mtp_lm_head_greedy_top1_impl",
    "num_speculative_tokens",
    "mtp_burst_groups",
    "mtp_prefill_seed",
    "greedy_token_fastpath",
    "sampled_token_fastpath",
    "device_token_carry",
    "static_decode_metadata",
    "static_decode_seq_lens_carry",
    "resident_decode_metadata",
    "trace_token_prefetch",
    "materialize_tied_lm_head",
    "compact_prefill_in_proj_qkv",
    "compact_prefill_gdn_z",
    "compact_prefill_full_attn_proj",
    "compact_prefill_mlp",
    "compact_prefill_token_count_mode",
    "lm_head_decode_act_dtype",
    "lm_head_topk_impl",
    "lm_head_greedy_top1_impl",
    "decode_proj_act_dtype",
    "decode_padded_gemm",
    "decode_padded_gemm_gate_up",
    "decode_rms_padded_gemm",
    "decode_padded_gemm_rows",
    "decode_padded_gemm_max_out_dim",
    "full_attention_kv_cache_dtype",
    "full_attention_kv_append_impl",
    "full_attention_decode_impl",
    "full_attention_prefill_impl",
    "gdn_disable_fallbacks",
    "gdn_prefill_post_conv_impl",
    "gdn_prefill_qkv_dtype",
    "gdn_prefill_post_conv_output_dtype",
    "gdn_packed_decode_impl",
    "gdn_packed_decode_qkv_dtype",
    "gdn_packed_decode_pre_normalize_qk",
    "gdn_packed_decode_max_batch",
    "skip_compile",
)


def _parse_buckets(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    buckets = tuple(int(part) for part in value.split(",") if part.strip())
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("bucket sizes must be positive integers")
    return buckets


def _resolved_engine_config(args) -> dict[str, Any]:
    resolved = dict(_server_cfg.engine)
    for key in _ENGINE_CLI_FIELDS:
        if hasattr(args, key):
            value = getattr(args, key)
            if key in {"max_num_resident_seqs", "max_blocks_per_seq", "gdn_packed_decode_max_batch"}:
                if value is not None and int(value) <= 0:
                    value = None
            resolved[key] = value
    return resolved


def _validate_server_config(engine_config: dict[str, Any], max_tokens_default: int):
    prefill_buckets = _parse_buckets(str(engine_config.get("prefill_buckets") or ""))
    prefill_token_buckets = _parse_buckets(str(engine_config.get("prefill_token_buckets") or ""))
    batch_size_buckets = _parse_buckets(str(engine_config.get("batch_size_buckets") or ""))
    max_num_seqs = int(engine_config.get("max_num_seqs") or 1)
    if batch_size_buckets and max(batch_size_buckets) > max_num_seqs:
        raise ValueError("--batch-size-buckets cannot exceed --max-num-seqs")
    if max_tokens_default <= 0:
        raise ValueError("--max-tokens-default must be positive")
    return prefill_buckets, prefill_token_buckets, batch_size_buckets


def _is_token_ids(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(token, int) for token in value)


def _is_batched_token_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and bool(row) and all(isinstance(token, int) for token in row) for row in value)
    )


def _is_chat_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("role"), str)
        and isinstance(value.get("content"), str)
        and value["role"] in {"system", "user", "assistant", "tool"}
    )


def _is_chat_messages(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_chat_message(item) for item in value)


def _is_batched_chat_messages(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_chat_messages(item) for item in value)


def _chat_messages_to_token_ids(messages: list[dict[str, str]]) -> list[int]:
    if engine is None:
        raise RuntimeError("Model is not loaded")
    tokenizer = engine.tokenizer
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("loaded tokenizer does not provide a chat template")
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "get") and token_ids.get("input_ids") is not None:
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("chat template returned an unexpected batch")
        token_ids = token_ids[0]
    if not _is_token_ids(token_ids):
        raise ValueError("chat template returned no token ids")
    return [int(token) for token in token_ids]


def _normalize_generation_inputs(data: dict) -> tuple[list[str | list[int]], bool]:
    prompt = data.get("prompt")
    input_ids = data.get("input_ids")
    messages = data.get("messages")
    provided = sum(value is not None for value in (prompt, input_ids, messages))
    if provided == 0:
        raise ValueError("provide one of prompt, input_ids, or messages")
    if provided > 1:
        raise ValueError("provide only one of prompt, input_ids, or messages")

    if messages is not None:
        if _is_chat_messages(messages):
            return [_chat_messages_to_token_ids(messages)], False
        if _is_batched_chat_messages(messages):
            return [_chat_messages_to_token_ids(item) for item in messages], True
        raise ValueError(
            "messages must be a non-empty list of chat messages or list of chat-message lists"
        )

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


def _sampling_params_for_inputs(data: dict, inputs: list[str | list[int]]) -> tuple[list[SamplingParams], int | list[int], float]:
    try:
        temperature = float(data.get("temperature", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a number") from exc
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    raw_max_tokens = data.get("max_tokens", app.config["MAX_TOKENS_DEFAULT"])
    if isinstance(raw_max_tokens, list):
        if len(raw_max_tokens) != len(inputs):
            raise ValueError("max_tokens list length must match number of prompts")
        try:
            token_limits = [int(value) for value in raw_max_tokens]
        except (TypeError, ValueError) as exc:
            raise ValueError("max_tokens entries must be integers") from exc
        if any(value <= 0 for value in token_limits):
            raise ValueError("max_tokens entries must be positive")
        params = [
            SamplingParams(temperature=temperature, max_tokens=value, ignore_eos=bool(data.get("ignore_eos", False)))
            for value in token_limits
        ]
        return params, token_limits, temperature
    try:
        max_tokens = int(raw_max_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens must be an integer or a list of integers") from exc
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return (
        [
            SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=bool(data.get("ignore_eos", False)))
            for _ in inputs
        ],
        max_tokens,
        temperature,
    )


def load_engine(args) -> LLMEngine:
    global engine
    engine_config = _resolved_engine_config(args)
    prefill_buckets, prefill_token_buckets, batch_size_buckets = _validate_server_config(
        engine_config,
        int(args.max_tokens_default),
    )
    model = str(engine_config.pop("model"))
    backend = str(engine_config.pop("backend"))
    max_prefill = int(engine_config.pop("max_prefill", 16) or 16)
    skip_compile = bool(engine_config.pop("skip_compile", False))
    startup_warmup_prefill_token_buckets = _parse_buckets(
        str(engine_config.pop("startup_warmup_prefill_token_buckets", "") or "")
    )
    startup_warmup_batch_size_buckets = _parse_buckets(
        str(engine_config.pop("startup_warmup_batch_size_buckets", "") or "")
    )
    startup_warmup_decode_block_table_buckets = _parse_buckets(
        str(engine_config.pop("startup_warmup_decode_block_table_buckets", "") or "")
    )
    startup_warmup_include_sampled_routes = bool(
        engine_config.pop("startup_warmup_include_sampled_routes", True)
    )
    max_kv_cache_mb = int(engine_config.pop("max_kv_cache_mb"))
    engine_config["max_kv_cache_bytes"] = int(max_kv_cache_mb * 1024 * 1024)
    engine_config["prefill_buckets"] = prefill_buckets
    engine_config["prefill_token_buckets"] = prefill_token_buckets
    engine_config["batch_size_buckets"] = batch_size_buckets
    engine = LLMEngine(
        model,
        backend=backend,
        **engine_config,
    )
    if not skip_compile:
        warmup_summary = engine.warmup_compilation(
            max_prefill_len=max(prefill_token_buckets or prefill_buckets or (max_prefill,)),
            max_batch=max(batch_size_buckets or (int(engine.config.max_num_seqs),)),
            prefill_token_buckets=startup_warmup_prefill_token_buckets or None,
            batch_size_buckets=startup_warmup_batch_size_buckets or None,
            decode_block_table_buckets=startup_warmup_decode_block_table_buckets or None,
            include_sampled_routes=startup_warmup_include_sampled_routes,
        )
        print(
            "Generic startup warmup complete: "
            f"{warmup_summary.get('seconds', 0.0):.2f}s, "
            f"jit_cache_entries={warmup_summary.get('jit_cache_entries_after')}"
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


def _summarize_trace(trace: dict, prompt_tokens: list[int]) -> dict:
    token_events = [event for event in trace["events"] if event.get("event") == "token"]
    by_request: dict[int, list[float]] = {}
    for event in token_events:
        by_request.setdefault(int(event["request_index"]), []).append(float(event["elapsed_seconds"]))
    itls_ms = []
    ttfts_ms = []
    for timestamps in by_request.values():
        if timestamps:
            ttfts_ms.append(1000.0 * timestamps[0])
            itls_ms.extend(1000.0 * (right - left) for left, right in zip(timestamps, timestamps[1:]))
    elapsed = max(
        [float(event.get("elapsed_seconds", 0.0)) for event in trace["events"]]
        or [0.0]
    )
    completion_tokens = [len(result["token_ids"]) for result in trace["results"]]
    return {
        "generation_time_ms": int(elapsed * 1000),
        "tokens_per_second": sum(completion_tokens) / elapsed if elapsed > 0 else 0.0,
        "ttft_ms_mean": float(sum(ttfts_ms) / len(ttfts_ms)) if ttfts_ms else None,
        "itl_ms_mean": float(sum(itls_ms) / len(itls_ms)) if itls_ms else None,
        "itl_ms_p50": float(_percentile(itls_ms, 50)) if itls_ms else None,
        "itl_ms_p95": float(_percentile(itls_ms, 95)) if itls_ms else None,
        "prompt_tokens": sum(prompt_tokens),
        "completion_tokens": sum(completion_tokens),
        "jit_cache_entries": len(engine.model_runner.executor._jit_cache) if engine is not None else 0,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


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


@app.route("/v1/generate_trace", methods=["POST"])
def generate_trace():
    data = request.get_json(force=True) or {}
    try:
        inputs, is_batch = _normalize_generation_inputs(data)
        sampling_params, max_tokens_for_validation, _ = _sampling_params_for_inputs(data, inputs)
        prompt_tokens = _token_counts(inputs)
        validation_max_tokens = max(max_tokens_for_validation) if isinstance(max_tokens_for_validation, list) else max_tokens_for_validation
        _validate_inputs_fit_config(inputs, prompt_tokens, validation_max_tokens)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        if engine is None:
            raise RuntimeError("Model is not loaded")
        with engine_lock:
            trace = engine.generate_with_trace(inputs, sampling_params=sampling_params)
        stats = _summarize_trace(trace, prompt_tokens)
        payload = {
            "results": trace["results"] if is_batch else trace["results"][0],
            "events": trace["events"],
            "stats": stats,
        }
        if engine is not None and hasattr(engine.model_runner, "get_speculative_stats"):
            payload["stats"]["speculative"] = engine.model_runner.get_speculative_stats()
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/v1/generate_stream", methods=["POST"])
def generate_stream():
    data = request.get_json(force=True) or {}
    try:
        inputs, _ = _normalize_generation_inputs(data)
        sampling_params, max_tokens_for_validation, _ = _sampling_params_for_inputs(data, inputs)
        prompt_tokens = _token_counts(inputs)
        validation_max_tokens = max(max_tokens_for_validation) if isinstance(max_tokens_for_validation, list) else max_tokens_for_validation
        _validate_inputs_fit_config(inputs, prompt_tokens, validation_max_tokens)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    def event_stream():
        if engine is None:
            yield f"data: {json.dumps({'event': 'error', 'error': 'Model is not loaded'})}\n\n"
            return
        try:
            with engine_lock:
                for event in engine.iter_generate(inputs, sampling_params=sampling_params):
                    yield f"data: {json.dumps(event, sort_keys=True)}\n\n"
        except Exception as exc:
            payload = {"event": "error", "error": str(exc), "traceback": traceback.format_exc()}
            yield f"data: {json.dumps(payload, sort_keys=True)}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


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
    global _server_cfg
    cfg = _server_cfg  # loaded at module init (before JAX)
    parser = argparse.ArgumentParser(description="nano-vllm-jax LLMEngine API server")
    parser.add_argument("--model", default=cfg.engine["model"])
    parser.add_argument("--host", default=cfg.server["host"])
    parser.add_argument("--port", type=int, default=cfg.server["port"])
    parser.add_argument("--backend", default=cfg.engine["backend"])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=cfg.engine["dtype"])
    parser.add_argument("--weight-dtype", choices=["float16", "bfloat16", "float32"], default=cfg.engine.get("weight_dtype"))
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default=cfg.engine["jax_execution"])
    parser.add_argument("--prefill-buckets", default=cfg.engine["prefill_buckets"])
    parser.add_argument("--prefill-token-buckets", default=cfg.engine["prefill_token_buckets"])
    parser.add_argument("--prefill-layout", choices=["packed", "dense"], default=cfg.engine["prefill_layout"])
    parser.add_argument("--batch-size-buckets", default=cfg.engine["batch_size_buckets"])
    parser.add_argument("--startup-warmup-prefill-token-buckets", default=cfg.engine.get("startup_warmup_prefill_token_buckets", ""))
    parser.add_argument("--startup-warmup-batch-size-buckets", default=cfg.engine.get("startup_warmup_batch_size_buckets", ""))
    parser.add_argument("--startup-warmup-decode-block-table-buckets", default=cfg.engine.get("startup_warmup_decode_block_table_buckets", ""))
    parser.add_argument("--startup-warmup-include-sampled-routes", action=argparse.BooleanOptionalAction, default=cfg.engine.get("startup_warmup_include_sampled_routes", True))
    parser.add_argument("--max-prefill", type=int, default=cfg.engine["max_prefill"])
    parser.add_argument("--max-kv-cache-mb", type=int, default=cfg.engine["max_kv_cache_mb"])
    parser.add_argument("--num-kvcache-blocks", type=int, default=cfg.engine["num_kvcache_blocks"])
    parser.add_argument("--max-num-seqs", type=int, default=cfg.engine["max_num_seqs"])
    parser.add_argument("--max-num-resident-seqs", type=int, default=cfg.engine.get("max_num_resident_seqs"))
    parser.add_argument("--max-num-batched-tokens", type=int, default=cfg.engine["max_num_batched_tokens"])
    parser.add_argument("--max-blocks-per-seq", type=int, default=cfg.engine.get("max_blocks_per_seq"))
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=cfg.engine.get("prefix_cache", True))
    parser.add_argument("--decode-block-table-buckets", default=cfg.engine.get("decode_block_table_buckets", ""))
    parser.add_argument("--max-tokens-default", type=int, default=cfg.server["max_tokens_default"])
    parser.add_argument("--speculative-method", choices=["none", "mtp"], default=cfg.engine["speculative_method"])
    parser.add_argument("--draft-sample-method", choices=["greedy", "probabilistic"], default=cfg.engine["draft_sample_method"])
    parser.add_argument(
        "--mtp-verifier-impl",
        choices=[
            "two_decode",
            "commit_select",
            "k_decode",
            "generic_k",
            "expanded",
            "packed_prefix",
            "packed_prefill",
            "prefill_packed",
        ],
        default=cfg.engine["mtp_verifier_impl"],
    )
    parser.add_argument("--mtp-batch-accept-policy", choices=["rowwise", "all_or_none"], default=cfg.engine["mtp_batch_accept_policy"])
    parser.add_argument("--mtp-seed-after-bonus", action=argparse.BooleanOptionalAction, default=cfg.engine["mtp_seed_after_bonus"])
    parser.add_argument("--mtp-hidden-source", choices=["pre_norm", "final_normed"], default=cfg.engine["mtp_hidden_source"])
    parser.add_argument("--mtp-chain-hidden-source", choices=["raw", "final_normed"], default=cfg.engine["mtp_chain_hidden_source"])
    parser.add_argument("--mtp-chain-mode", choices=["recursive", "sequence"], default=cfg.engine["mtp_chain_mode"])
    parser.add_argument("--mtp-lm-head-greedy-top1-impl", default=cfg.engine["mtp_lm_head_greedy_top1_impl"])
    parser.add_argument("--num-speculative-tokens", type=int, choices=list(range(0, 9)), default=cfg.engine["num_speculative_tokens"])
    parser.add_argument("--mtp-burst-groups", type=int, default=cfg.engine["mtp_burst_groups"])
    parser.add_argument("--mtp-prefill-seed", action=argparse.BooleanOptionalAction, default=cfg.engine["mtp_prefill_seed"])
    parser.add_argument("--greedy-token-fastpath", action=argparse.BooleanOptionalAction, default=cfg.engine["greedy_token_fastpath"])
    parser.add_argument("--sampled-token-fastpath", action=argparse.BooleanOptionalAction, default=cfg.engine["sampled_token_fastpath"])
    parser.add_argument("--device-token-carry", action=argparse.BooleanOptionalAction, default=cfg.engine["device_token_carry"])
    parser.add_argument("--static-decode-metadata", action=argparse.BooleanOptionalAction, default=cfg.engine["static_decode_metadata"])
    parser.add_argument("--static-decode-seq-lens-carry", action=argparse.BooleanOptionalAction, default=cfg.engine["static_decode_seq_lens_carry"])
    parser.add_argument("--resident-decode-metadata", action=argparse.BooleanOptionalAction, default=cfg.engine["resident_decode_metadata"])
    parser.add_argument("--trace-token-prefetch", action=argparse.BooleanOptionalAction, default=cfg.engine["trace_token_prefetch"])
    parser.add_argument("--materialize-tied-lm-head", action=argparse.BooleanOptionalAction, default=cfg.engine["materialize_tied_lm_head"])
    parser.add_argument("--compact-prefill-in-proj-qkv", action=argparse.BooleanOptionalAction, default=cfg.engine["compact_prefill_in_proj_qkv"])
    parser.add_argument("--compact-prefill-gdn-z", action=argparse.BooleanOptionalAction, default=cfg.engine["compact_prefill_gdn_z"])
    parser.add_argument("--compact-prefill-full-attn-proj", action=argparse.BooleanOptionalAction, default=cfg.engine["compact_prefill_full_attn_proj"])
    parser.add_argument("--compact-prefill-mlp", action=argparse.BooleanOptionalAction, default=cfg.engine["compact_prefill_mlp"])
    parser.add_argument("--compact-prefill-token-count-mode", default=cfg.engine["compact_prefill_token_count_mode"])
    parser.add_argument("--lm-head-decode-act-dtype", default=cfg.engine["lm_head_decode_act_dtype"])
    parser.add_argument("--lm-head-topk-impl", default=cfg.engine["lm_head_topk_impl"])
    parser.add_argument("--lm-head-greedy-top1-impl", default=cfg.engine["lm_head_greedy_top1_impl"])
    parser.add_argument("--decode-proj-act-dtype", default=cfg.engine["decode_proj_act_dtype"])
    parser.add_argument("--decode-padded-gemm", action=argparse.BooleanOptionalAction, default=cfg.engine["decode_padded_gemm"])
    parser.add_argument("--decode-padded-gemm-gate-up", action=argparse.BooleanOptionalAction, default=cfg.engine["decode_padded_gemm_gate_up"])
    parser.add_argument("--decode-rms-padded-gemm", action=argparse.BooleanOptionalAction, default=cfg.engine["decode_rms_padded_gemm"])
    parser.add_argument("--decode-padded-gemm-rows", type=int, default=cfg.engine["decode_padded_gemm_rows"])
    parser.add_argument("--decode-padded-gemm-max-out-dim", type=int, default=cfg.engine["decode_padded_gemm_max_out_dim"])
    parser.add_argument("--full-attention-kv-cache-dtype", default=cfg.engine["full_attention_kv_cache_dtype"])
    parser.add_argument("--full-attention-kv-append-impl", default=cfg.engine["full_attention_kv_append_impl"])
    parser.add_argument("--full-attention-decode-impl", default=cfg.engine["full_attention_decode_impl"])
    parser.add_argument("--full-attention-prefill-impl", default=cfg.engine["full_attention_prefill_impl"])
    parser.add_argument("--gdn-disable-fallbacks", action=argparse.BooleanOptionalAction, default=cfg.engine["gdn_disable_fallbacks"])
    parser.add_argument("--gdn-prefill-post-conv-impl", default=cfg.engine["gdn_prefill_post_conv_impl"])
    parser.add_argument("--gdn-prefill-qkv-dtype", default=cfg.engine["gdn_prefill_qkv_dtype"])
    parser.add_argument("--gdn-prefill-post-conv-output-dtype", default=cfg.engine["gdn_prefill_post_conv_output_dtype"])
    parser.add_argument("--gdn-packed-decode-impl", default=cfg.engine["gdn_packed_decode_impl"])
    parser.add_argument("--gdn-packed-decode-qkv-dtype", default=cfg.engine["gdn_packed_decode_qkv_dtype"])
    parser.add_argument("--gdn-packed-decode-pre-normalize-qk", action=argparse.BooleanOptionalAction, default=cfg.engine["gdn_packed_decode_pre_normalize_qk"])
    parser.add_argument("--gdn-packed-decode-max-batch", type=int, default=cfg.engine.get("gdn_packed_decode_max_batch") or 0)
    parser.add_argument("--skip-compile", action=argparse.BooleanOptionalAction, default=cfg.engine.get("skip_compile", False))
    parser.add_argument("--config", default=None, help="Path to server_config.yaml (overrides NANO_VLLM_JAX_SERVER_CONFIG)")
    args = parser.parse_args()
    engine_config = _resolved_engine_config(args)

    app.config["MAX_TOKENS_DEFAULT"] = args.max_tokens_default

    print("nano-vllm-jax LLMEngine server")
    print(f"config_source={_server_cfg.source}")
    print(
        f"model={engine_config['model']} dtype={engine_config['dtype']} "
        f"weight_dtype={engine_config.get('weight_dtype') or engine_config['dtype']} "
        f"execution={engine_config['jax_execution']}"
    )
    print(
        f"prefill_layout={engine_config['prefill_layout']} "
        f"prefill_token_buckets={engine_config['prefill_token_buckets']} "
        f"prefill_buckets={engine_config['prefill_buckets']} "
        f"batch_size_buckets={engine_config['batch_size_buckets']} "
        f"max_blocks_per_seq={engine_config.get('max_blocks_per_seq')}"
    )
    print(
        f"startup_warmup=prefill_token_buckets:"
        f"{engine_config.get('startup_warmup_prefill_token_buckets') or '<serving>'} "
        f"batch_size_buckets:{engine_config.get('startup_warmup_batch_size_buckets') or '<serving>'} "
        f"decode_block_table_buckets:{engine_config.get('startup_warmup_decode_block_table_buckets') or '<serving>'} "
        f"include_sampled_routes:{engine_config.get('startup_warmup_include_sampled_routes')}"
    )
    print(
        f"serving_fastpaths=greedy_token:{engine_config['greedy_token_fastpath']} "
        f"sampled_token:{engine_config['sampled_token_fastpath']} "
        f"device_token_carry:{engine_config['device_token_carry']} "
        f"static_decode_metadata:{engine_config['static_decode_metadata']} "
        f"resident_decode_metadata:{engine_config['resident_decode_metadata']}"
    )
    print(
        f"kernels=full_attention_decode:{engine_config['full_attention_decode_impl']} "
        f"full_attention_prefill:{engine_config['full_attention_prefill_impl']} "
        f"gdn_prefill:{engine_config['gdn_prefill_post_conv_impl']} "
        f"gdn_decode:{engine_config['gdn_packed_decode_impl']}"
    )
    load_engine(args)
    print(f"server_ready=http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
