#!/usr/bin/env python3
"""Thin HTTP transport for the continuous-batching engine."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any

from nanovllm_jax.config import EngineConfig, ServerSettings, WarmupConfig, load_engine_config
from nanovllm_jax.fastpath import format_manifest, validate_runtime_dependencies


_DEFAULT_XLA_FLAGS = "--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false"
_CHAT_ROLES = {"system", "user", "assistant", "tool"}


def _initial_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=os.getenv("NANO_VLLM_JAX_SERVER_CONFIG", "server.yaml"))
    args, _ = parser.parse_known_args()
    path = Path(args.config or "server.yaml")
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent / path
    return path


_CONFIG_PATH = _initial_config_path()
_SETTINGS = load_engine_config(_CONFIG_PATH)


def _runtime_root() -> Path:
    configured = os.getenv("NANO_VLLM_JAX_CACHE_ROOT")
    if configured:
        return Path(configured)
    mountpoint = Path("/mountpoint/.exp")
    return mountpoint if mountpoint.exists() else Path.cwd()


def _apply_runtime_config() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("XLA_FLAGS", _DEFAULT_XLA_FLAGS)

    cache_dir = Path(
        os.getenv("NANO_VLLM_JAX_COMPILE_CACHE_DIR")
        or os.getenv("JAX_COMPILATION_CACHE_DIR")
        or (_runtime_root() / ".cache" / "jax")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NANO_VLLM_JAX_COMPILE_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))

    try:
        import jax

        jax.config.update("jax_enable_compilation_cache", True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except AttributeError:
        print("[server] this JAX build does not support jax_compilation_cache_dir")
    except Exception as exc:
        print(f"[server] could not configure the JAX compile cache: {type(exc).__name__}: {exc}")


# Configure JAX before importing Flask or the engine stack.
_apply_runtime_config()

from flask import Flask, Response, jsonify, request, stream_with_context

from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.service import EngineService


app = Flask(__name__)
app.config["MAX_TOKENS_DEFAULT"] = _SETTINGS.max_tokens_default
engine: LLMEngine | None = None
service: EngineService | None = None
engine_lock = Lock()


def _parse_buckets(value: str | tuple[int, ...] | list[int] | None, name: str) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    parts = [part.strip() for part in value.split(",") if part.strip()] if isinstance(value, str) else value
    try:
        buckets = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"{name} values must be positive")
    return buckets


def _settings_from_args(args: argparse.Namespace) -> ServerSettings:
    warmup = WarmupConfig(
        enabled=not bool(args.skip_compile),
        prefill_token_buckets=_parse_buckets(args.warmup_prefill_token_buckets, "warmup-prefill-token-buckets"),
        batch_size_buckets=_parse_buckets(args.warmup_batch_size_buckets, "warmup-batch-size-buckets"),
        decode_block_buckets=_parse_buckets(args.warmup_decode_block_buckets, "warmup-decode-block-buckets"),
        include_sampled_routes=bool(args.warmup_sampled_routes),
    )
    engine_config = EngineConfig(
        model=args.model,
        max_prefill=args.max_prefill,
        max_num_seqs=args.max_num_seqs,
        max_num_resident_seqs=args.max_num_resident_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_blocks_per_seq=args.max_blocks_per_seq,
        kv_cache_bytes=args.kv_cache_bytes,
        num_kvcache_blocks=args.num_kvcache_blocks,
        prefill_token_buckets=_parse_buckets(args.prefill_token_buckets, "prefill-token-buckets"),
        batch_size_buckets=_parse_buckets(args.batch_size_buckets, "batch-size-buckets"),
        decode_block_buckets=_parse_buckets(args.decode_block_buckets, "decode-block-buckets"),
        warmup=warmup,
        prefix_cache=bool(args.prefix_cache),
    )
    return replace(
        _SETTINGS,
        host=args.host,
        port=args.port,
        max_tokens_default=args.max_tokens_default,
        engine=engine_config,
    )


def _validate_settings(settings: ServerSettings) -> None:
    if settings.max_tokens_default <= 0:
        raise ValueError("--max-tokens-default must be positive")
    if max(settings.engine.batch_size_buckets) > settings.engine.max_num_seqs:
        raise ValueError("--batch-size-buckets cannot exceed --max-num-seqs")


def _is_token_ids(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(token, int) for token in value)


def _is_token_id_batch(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_token_ids(row) for row in value)


def _is_chat_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("role"), str)
        and isinstance(value.get("content"), str)
        and value["role"] in _CHAT_ROLES
    )


def _is_chat(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_chat_message(item) for item in value)


def _is_chat_batch(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_chat(item) for item in value)


def _chat_to_token_ids(messages: list[dict[str, str]]) -> list[int]:
    if engine is None:
        raise RuntimeError("model is not loaded")
    tokenizer = engine.tokenizer
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("loaded tokenizer does not provide a chat template")

    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(token_ids, "get"):
        token_ids = token_ids.get("input_ids", token_ids)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("chat template returned an unexpected batch")
        token_ids = token_ids[0]
    if not _is_token_ids(token_ids):
        raise ValueError("chat template returned no token ids")
    return [int(token) for token in token_ids]


def _inputs_from_request(data: dict[str, Any]) -> tuple[list[str | list[int]], bool]:
    prompt = data.get("prompt")
    input_ids = data.get("input_ids")
    messages = data.get("messages")
    provided = sum(value is not None for value in (prompt, input_ids, messages))
    if provided != 1:
        raise ValueError("provide exactly one of prompt, input_ids, or messages")

    if messages is not None:
        if _is_chat(messages):
            return [_chat_to_token_ids(messages)], False
        if _is_chat_batch(messages):
            return [_chat_to_token_ids(item) for item in messages], True
        raise ValueError("messages must be chat messages or a batch of chat messages")

    if input_ids is not None:
        if _is_token_ids(input_ids):
            return [[int(token) for token in input_ids]], False
        if _is_token_id_batch(input_ids):
            return [[int(token) for token in row] for row in input_ids], True
        raise ValueError("input_ids must be token ids or a batch of token ids")

    if isinstance(prompt, str) and prompt:
        return [prompt], False
    if isinstance(prompt, list) and prompt and all(isinstance(item, str) and item for item in prompt):
        return prompt, True
    raise ValueError("prompt must be a non-empty string or list of strings")


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _sampling_params(data: dict[str, Any], count: int) -> tuple[list[SamplingParams], int]:
    try:
        temperature = float(data.get("temperature", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a number") from exc
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    raw_limits = data.get("max_tokens", app.config["MAX_TOKENS_DEFAULT"])
    limits = raw_limits if isinstance(raw_limits, list) else [raw_limits] * count
    if len(limits) != count:
        raise ValueError("max_tokens list length must match number of prompts")
    limits = [_positive_int(limit, "max_tokens") for limit in limits]
    ignore_eos = bool(data.get("ignore_eos", False))
    return (
        [SamplingParams(temperature=temperature, max_tokens=limit, ignore_eos=ignore_eos) for limit in limits],
        max(limits),
    )


def _token_counts(inputs: list[str | list[int]]) -> list[int]:
    if engine is None:
        raise RuntimeError("model is not loaded")
    return [len(item) if isinstance(item, list) else len(engine._tokenize(item)) for item in inputs]


def _validate_inputs_fit_config(inputs: list[str | list[int]], prompt_tokens: list[int], max_tokens: int) -> None:
    if engine is None:
        raise RuntimeError("model is not loaded")

    max_blocks_per_seq = getattr(engine.config, "max_blocks_per_seq", None)
    if max_blocks_per_seq is not None:
        capacity = int(max_blocks_per_seq) * int(engine.config.block_size)
        needed = max(prompt_tokens) + max_tokens
        if needed > capacity:
            raise ValueError(f"request needs {needed} tokens, exceeding per-sequence KV capacity {capacity}")

    max_num_seqs = getattr(engine.config, "max_num_seqs", None)
    if max_num_seqs is not None and len(inputs) > int(max_num_seqs):
        raise ValueError(f"request has {len(inputs)} prompts, exceeding max_num_seqs {max_num_seqs}")


def _prepare_generation(data: dict[str, Any]):
    inputs, is_batch = _inputs_from_request(data)
    sampling_params, max_tokens = _sampling_params(data, len(inputs))
    prompt_tokens = _token_counts(inputs)
    _validate_inputs_fit_config(inputs, prompt_tokens, max_tokens)
    return inputs, sampling_params, prompt_tokens, is_batch


def load_engine(settings: ServerSettings) -> LLMEngine:
    global engine, service
    _validate_settings(settings)
    engine = LLMEngine(settings.engine.model, engine_config=settings.engine)

    if settings.engine.warmup.enabled:
        warmup = settings.engine.warmup
        summary = engine.warmup_compilation(
            max_prefill_len=max(settings.engine.prefill_token_buckets),
            max_batch=max(settings.engine.batch_size_buckets),
            prefill_token_buckets=warmup.prefill_token_buckets,
            batch_size_buckets=warmup.batch_size_buckets,
            decode_block_table_buckets=warmup.decode_block_buckets,
            include_sampled_routes=warmup.include_sampled_routes,
        )
        print(f"startup_warmup_complete seconds={summary.get('seconds', 0.0):.2f}")

    service = EngineService(engine, engine_lock=engine_lock)
    service.start()
    return engine


def _run_generation(inputs: list[str | list[int]], sampling_params: list[SamplingParams]):
    if engine is None:
        raise RuntimeError("model is not loaded")
    if service is None:
        return engine.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    return [
        {"text": result.text, "token_ids": result.token_ids}
        for result in service.generate_many(inputs, sampling_params)
    ]


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
    }
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


def _completion_payload(results, prompt_tokens: list[int], elapsed: float):
    completion_tokens = [len(result["token_ids"]) for result in results]
    created = int(time.time())
    model_name = engine.config.__class__.__name__ if engine is not None else "unknown"
    return {
        "id": f"cmpl-{created}",
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {"text": result["text"], "index": index, "finish_reason": "length"}
            for index, result in enumerate(results)
        ],
        "usage": {
            "prompt_tokens": sum(prompt_tokens),
            "completion_tokens": sum(completion_tokens),
            "total_tokens": sum(prompt_tokens) + sum(completion_tokens),
        },
        "stats": {
            "generation_time_ms": int(elapsed * 1000),
            "tokens_per_second": sum(completion_tokens) / elapsed if elapsed > 0 else 0.0,
        },
    }


def _json_error(exc: Exception, status: int):
    return jsonify({"error": str(exc)}), status


@app.route("/health", methods=["GET"])
def health():
    loaded = engine is not None
    return jsonify({"status": "healthy" if loaded else "loading", "model_loaded": loaded})


@app.route("/v1/generate", methods=["POST"])
def generate():
    try:
        inputs, sampling_params, prompt_tokens, is_batch = _prepare_generation(request.get_json(force=True) or {})
        started = time.perf_counter()
        results = _run_generation(inputs, sampling_params)
        return jsonify(_generation_payload(results, prompt_tokens, time.perf_counter() - started, is_batch))
    except ValueError as exc:
        return _json_error(exc, 400)
    except RuntimeError as exc:
        return _json_error(exc, 503)
    except Exception as exc:
        return _json_error(exc, 500)


@app.route("/v1/generate_stream", methods=["POST"])
def generate_stream():
    try:
        inputs, sampling_params, _, _ = _prepare_generation(request.get_json(force=True) or {})
    except ValueError as exc:
        return _json_error(exc, 400)
    except RuntimeError as exc:
        return _json_error(exc, 503)

    if len(inputs) != 1:
        return _json_error(ValueError("streaming accepts one prompt per request"), 400)

    def events():
        if service is None:
            yield f"data: {json.dumps({'event': 'error', 'error': 'model is not loaded'})}\n\n"
            return
        try:
            handle = service.submit(inputs[0], sampling_params[0], stream=True)
            for event in handle.events():
                event.setdefault("request_index", 0)
                yield f"data: {json.dumps(event, sort_keys=True)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'error': str(exc)}, sort_keys=True)}\n\n"

    return Response(stream_with_context(events()), mimetype="text/event-stream")


@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.get_json(force=True) or {}
    try:
        inputs, sampling_params, prompt_tokens, _ = _prepare_generation({"prompt": data.get("prompt", ""), **data})
        started = time.perf_counter()
        results = _run_generation(inputs, sampling_params)
        return jsonify(_completion_payload(results, prompt_tokens, time.perf_counter() - started))
    except ValueError as exc:
        return _json_error(exc, 400)
    except RuntimeError as exc:
        return _json_error(exc, 503)
    except Exception as exc:
        return _json_error(exc, 500)


def _build_parser(settings: ServerSettings) -> argparse.ArgumentParser:
    cfg = settings.engine
    parser = argparse.ArgumentParser(description="nano-vllm-jax serving API")
    parser.add_argument("--config", default=str(_CONFIG_PATH), help="Path to server.yaml")
    parser.add_argument("--model", default=cfg.model)
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--max-tokens-default", type=int, default=settings.max_tokens_default)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=cfg.prefix_cache)
    parser.add_argument("--skip-compile", action=argparse.BooleanOptionalAction, default=not cfg.warmup.enabled)
    parser.add_argument("--warmup-sampled-routes", action=argparse.BooleanOptionalAction, default=cfg.warmup.include_sampled_routes)

    for name in (
        "max_prefill",
        "max_num_seqs",
        "max_num_resident_seqs",
        "max_num_batched_tokens",
        "max_blocks_per_seq",
        "kv_cache_bytes",
        "num_kvcache_blocks",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(cfg, name))

    for flag, values in (
        ("prefill-token-buckets", cfg.prefill_token_buckets),
        ("batch-size-buckets", cfg.batch_size_buckets),
        ("decode-block-buckets", cfg.decode_block_buckets),
        ("warmup-prefill-token-buckets", cfg.warmup.prefill_token_buckets),
        ("warmup-batch-size-buckets", cfg.warmup.batch_size_buckets),
        ("warmup-decode-block-buckets", cfg.warmup.decode_block_buckets),
    ):
        parser.add_argument(f"--{flag}", default=",".join(map(str, values)))
    return parser


def main() -> None:
    args = _build_parser(_SETTINGS).parse_args()
    settings = _settings_from_args(args)
    app.config["MAX_TOKENS_DEFAULT"] = settings.max_tokens_default

    print("nano-vllm-jax serving engine")
    print(f"config_source={Path(args.config)}")
    print(f"model={settings.engine.model}")
    print(
        "capacity="
        f"max_num_seqs:{settings.engine.max_num_seqs} "
        f"max_num_batched_tokens:{settings.engine.max_num_batched_tokens} "
        f"max_blocks_per_seq:{settings.engine.max_blocks_per_seq}"
    )
    print("fastpath_manifest:\n" + format_manifest())
    validate_runtime_dependencies()
    load_engine(settings)
    print(f"server_ready=http://{settings.host}:{settings.port}")
    app.run(host=settings.host, port=settings.port, threaded=True)


if __name__ == "__main__":
    main()
