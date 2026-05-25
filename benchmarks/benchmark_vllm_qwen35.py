#!/usr/bin/env python3
"""vLLM benchmark harness for Qwen3.5 0.8B.

The script is intentionally import-gated: it records a run-journal issue when
vLLM is not installed instead of importing vLLM at module import time.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_tracking import RunRecorder
from runtime_paths import default_runtime_root


def configure_vllm_paths() -> dict[str, str]:
    root = default_runtime_root()
    values = {
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / ".cache"),
        "HF_HOME": str(root / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(root / ".cache" / "huggingface" / "hub"),
        "VLLM_CACHE_ROOT": str(root / ".cache" / "vllm"),
        "VLLM_XLA_CACHE_PATH": str(root / ".cache" / "vllm" / "xla_cache"),
        "VLLM_RPC_BASE_PATH": str(root / "tmp" / "vllm-rpc"),
        "VLLM_DO_NOT_TRACK": "1",
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)
    for key, value in values.items():
        if key != "VLLM_DO_NOT_TRACK":
            Path(value).mkdir(parents=True, exist_ok=True)
    return {key: os.environ[key] for key in values}


configure_vllm_paths()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "mtp"])
    parser.add_argument("--execution", default="offline", choices=["offline", "async"])
    parser.add_argument("--speculative-method", default="mtp")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--input-lens", default="16,64,128")
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--prompt-suite", choices=["synthetic", "real", "mixed", "server_shapes"], default="mixed")
    parser.add_argument("--output-json", default="results/qwen08_vllm_benchmark.json")
    parser.add_argument("--reference-json", default="", help="Optional benchmark_server_shapes.py JSON to compare generated IDs.")
    parser.add_argument("--profile", action="store_true", default=False, help="Reserved for parity with JAX harness; vLLM profiling is backend-specific.")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


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
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    return repr(value)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _timing_metrics(rows: list[dict[str, Any]], elapsed: float, total_tokens: int, source: str) -> dict[str, Any]:
    ttfts = []
    itls = []
    for row in rows:
        timestamps = [float(event["elapsed_seconds"]) for event in row.get("token_events", [])]
        if timestamps:
            ttfts.append(1000.0 * timestamps[0])
            itls.extend(1000.0 * (right - left) for left, right in zip(timestamps, timestamps[1:]))
    return {
        "seconds": elapsed,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "ttft_ms_mean": float(sum(ttfts) / len(ttfts)) if ttfts else None,
        "ttft_ms_p50": _percentile(ttfts, 50),
        "ttft_ms_p95": _percentile(ttfts, 95),
        "itl_ms_mean": float(sum(itls) / len(itls)) if itls else None,
        "itl_ms_p50": _percentile(itls, 50),
        "itl_ms_p95": _percentile(itls, 95),
        "itl_source": source,
    }


def _first_diff(left: list[int], right: list[int]) -> dict | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if int(lhs) != int(rhs):
            return {"index": index, "left": int(lhs), "right": int(rhs)}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "left": int(left[index]) if index < len(left) else None,
            "right": int(right[index]) if index < len(right) else None,
            "length_mismatch": True,
        }
    return None


def _prompt_seed_suite(name: str) -> list[str]:
    synthetic = [
        "red red red red red red red red",
        "1 1 1 1 1 1 1 1 1 1 1 1",
        "the the the the the the the the",
    ]
    real = [
        "Explain the key tradeoffs in paged attention cache management for inference.",
        "Write a careful proof sketch for binary search over a sorted array.",
        "Summarize speculative decoding risks for deterministic generation.",
    ]
    server_shapes = [
        "The future of artificial intelligence is poised to transform software systems.",
        "Explain the key tradeoffs in paged attention cache management for inference.",
        "Write a concise proof sketch for binary search over a sorted array.",
        "Summarize speculative decoding risks for deterministic generation.",
    ]
    if name == "synthetic":
        return synthetic
    if name == "real":
        return real
    if name == "server_shapes":
        return server_shapes
    return synthetic + real


def make_prompt_ids(tokenizer, length: int, seed: str) -> list[int]:
    ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    out: list[int] = []
    while len(out) < length:
        out.extend(ids)
    return [int(token) for token in out[:length]]


def make_prompts(tokenizer, lengths: list[int], suite: str) -> list[dict[str, Any]]:
    seeds = _prompt_seed_suite(suite)
    return [
        {
            "name": f"len_{length}",
            "prompt_length": int(length),
            "input_ids": make_prompt_ids(tokenizer, length, seeds[index % len(seeds)]),
        }
        for index, length in enumerate(lengths)
    ]


def _extract_logprob_topk(completion, top_k: int) -> list[dict[str, Any]]:
    rows = []
    for item in getattr(completion, "logprobs", None) or []:
        # vLLM versions differ: entries can be dict-like token_id -> object or
        # list-like objects. Keep only stable token ids and logprobs.
        if isinstance(item, dict):
            entries = list(item.items())
        else:
            entries = [(None, value) for value in list(item or [])]
        packed = []
        for key, entry in entries[:top_k]:
            numeric_id = getattr(entry, "token_id", None)
            if numeric_id is None and key is not None:
                numeric_id = key
            if numeric_id is None and isinstance(entry, tuple) and entry:
                numeric_id = entry[0]
            decoded_token = getattr(entry, "decoded_token", None)
            logprob = getattr(entry, "logprob", None)
            if logprob is None and isinstance(entry, tuple) and len(entry) > 1:
                logprob = entry[1]
            packed.append({"token_id": numeric_id, "decoded_token": decoded_token, "logprob": logprob})
        rows.append(packed)
    return rows


def run_vllm(args: argparse.Namespace, recorder: RunRecorder) -> dict:
    if args.execution == "async":
        return asyncio.run(run_vllm_async(args, recorder))

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:
        recorder.record_issue(
            summary="vLLM is not installed or cannot be imported",
            severity="error",
            status="open",
            details={"error": f"{type(exc).__name__}: {exc}"},
            learnings=["The vLLM benchmark requires an isolated vLLM install under /mountpoint/.exp."],
            resolution="Install vLLM in the project environment, then rerun this harness on GPU.",
        )
        raise

    env = configure_vllm_paths()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = make_prompts(tokenizer, _parse_ints(args.input_lens), args.prompt_suite)

    speculative_config = None
    if args.mode == "mtp":
        speculative_config = {
            "method": args.speculative_method,
            "num_speculative_tokens": int(args.num_speculative_tokens),
        }

    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "trust_remote_code": bool(args.trust_remote_code),
        "enforce_eager": bool(args.enforce_eager),
    }
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config

    load_t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_t0

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(args.output_len),
        ignore_eos=True,
        logprobs=int(args.top_k),
    )
    prompt_token_ids = [{"prompt_token_ids": row["input_ids"]} for row in prompts]

    # Warmup one prompt before timing.
    _ = llm.generate([prompt_token_ids[0]], sampling)

    started = time.perf_counter()
    outputs = llm.generate(prompt_token_ids, sampling)
    elapsed = time.perf_counter() - started

    rows = []
    total_tokens = 0
    for prompt, output in zip(prompts, outputs):
        completion = output.outputs[0]
        token_ids = [int(token) for token in completion.token_ids]
        total_tokens += len(token_ids)
        rows.append(
            {
                "name": prompt["name"],
                "prompt_length": prompt["prompt_length"],
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "topk_logprobs_by_step": _extract_logprob_topk(completion, int(args.top_k)),
            }
        )

    # Offline LLM.generate exposes end-to-end timing, not true streaming ITL.
    # Keep the ITL fields explicit so downstream reports do not confuse them
    # with server-stream token timestamps.
    performance = {
        "seconds": elapsed,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "itl_ms_mean": None,
        "itl_ms_p50": None,
        "itl_ms_p95": None,
        "itl_source": "unavailable_from_offline_llm_generate",
    }
    if len(rows) == 1 and rows[0]["generated_tokens"] > 1:
        synthetic_itl = 1000.0 * elapsed / max(1, rows[0]["generated_tokens"])
        performance["offline_avg_ms_per_token"] = synthetic_itl

    del llm
    gc.collect()

    return {
        "environment": {
            "vllm_version": importlib.metadata.version("vllm"),
            "env_paths": env,
        },
        "run_config": {
            "model": args.model,
            "dtype": args.dtype,
            "mode": args.mode,
            "execution": args.execution,
            "speculative_config": speculative_config,
            "input_lens": _parse_ints(args.input_lens),
            "output_len": args.output_len,
            "top_k": args.top_k,
        },
        "load_seconds": load_seconds,
        "performance": performance,
        "rows": rows,
    }


async def run_vllm_async(args: argparse.Namespace, recorder: RunRecorder) -> dict:
    try:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        from vllm.sampling_params import RequestOutputKind
    except Exception as exc:
        recorder.record_issue(
            summary="vLLM async engine is not installed or cannot be imported",
            severity="error",
            status="open",
            details={"error": f"{type(exc).__name__}: {exc}"},
            learnings=["True vLLM ITL uses AsyncLLMEngine.generate streaming outputs."],
            resolution="Install vLLM in the isolated environment, then rerun with --execution async.",
        )
        raise

    env = configure_vllm_paths()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = make_prompts(tokenizer, _parse_ints(args.input_lens), args.prompt_suite)

    speculative_config = None
    if args.mode == "mtp":
        speculative_config = {
            "method": args.speculative_method,
            "num_speculative_tokens": int(args.num_speculative_tokens),
        }

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        trust_remote_code=bool(args.trust_remote_code),
        enforce_eager=bool(args.enforce_eager),
        speculative_config=speculative_config,
        disable_log_stats=True,
    )
    load_t0 = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    load_seconds = time.perf_counter() - load_t0
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(args.output_len),
        ignore_eos=True,
        logprobs=int(args.top_k),
        output_kind=RequestOutputKind.DELTA,
    )

    async def consume_prompt(prompt: dict[str, Any], request_id: str, start: float) -> dict[str, Any]:
        token_events = []
        token_ids = []
        topk_rows = []
        async for output in engine.generate(
            {"prompt_token_ids": prompt["input_ids"]},
            sampling,
            request_id=request_id,
        ):
            now = time.perf_counter()
            completion = output.outputs[0]
            delta_ids = [int(token) for token in completion.token_ids]
            metrics = getattr(output, "metrics", None)
            metric_last_token_ts = getattr(metrics, "last_token_ts", None) if metrics is not None else None
            metric_first_token_ts = getattr(metrics, "first_token_ts", None) if metrics is not None else None
            for offset, token_id in enumerate(delta_ids):
                token_events.append(
                    {
                        "completion_index": len(token_ids) + offset,
                        "token_id": int(token_id),
                        "elapsed_seconds": now - start,
                        "delta_size": len(delta_ids),
                        "vllm_first_token_ts": metric_first_token_ts,
                        "vllm_last_token_ts": metric_last_token_ts,
                    }
                )
            token_ids.extend(delta_ids)
            topk_rows.extend(_extract_logprob_topk(completion, int(args.top_k)))
        return {
            "name": prompt["name"],
            "prompt_length": prompt["prompt_length"],
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "topk_logprobs_by_step": topk_rows,
            "token_events": token_events,
        }

    async def drain_prompt(prompt: dict[str, Any], request_id: str, warm_sampling: SamplingParams) -> None:
        async for _ in engine.generate(
            {"prompt_token_ids": prompt["input_ids"]},
            warm_sampling,
            request_id=request_id,
        ):
            pass

    # Warm up the full request shape so measured TTFT does not include
    # first-use per-shape Triton/JIT kernels.
    warm_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max(1, min(2, int(args.output_len))),
        ignore_eos=True,
        logprobs=int(args.top_k),
        output_kind=RequestOutputKind.DELTA,
    )
    await asyncio.gather(
        *[
            drain_prompt(prompt, f"warmup-{index}", warm_sampling)
            for index, prompt in enumerate(prompts)
        ]
    )

    started = time.perf_counter()
    rows = await asyncio.gather(
        *[
            consume_prompt(prompt, f"bench-{index}", started)
            for index, prompt in enumerate(prompts)
        ]
    )
    elapsed = time.perf_counter() - started
    total_tokens = sum(int(row["generated_tokens"]) for row in rows)
    performance = _timing_metrics(rows, elapsed, total_tokens, "vllm_async_generate")

    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        maybe_result = shutdown()
        if asyncio.iscoroutine(maybe_result):
            await maybe_result
    del engine
    gc.collect()

    return {
        "environment": {
            "vllm_version": importlib.metadata.version("vllm"),
            "env_paths": env,
        },
        "run_config": {
            "model": args.model,
            "dtype": args.dtype,
            "mode": args.mode,
            "execution": args.execution,
            "speculative_config": speculative_config,
            "input_lens": _parse_ints(args.input_lens),
            "output_len": args.output_len,
            "top_k": args.top_k,
        },
        "load_seconds": load_seconds,
        "performance": performance,
        "rows": rows,
    }


def compare_reference(summary: dict, reference_json: str) -> dict:
    if not reference_json:
        return {"checked": False, "ok": True}
    reference = json.loads(Path(reference_json).read_text())
    ref_by_len = {}
    ref_by_shape = {}
    for row in reference.get("rows", []):
        lengths = row.get("prompt_lengths", [])
        topk_rows = row.get("jax_paged", {}).get("topk_by_request") or []
        refs = [
            {
                "generated_tokens": tokens,
                "first_step_topk": (topk_rows[index] if index < len(topk_rows) else {}).get("ids", []),
            }
            for index, tokens in enumerate(row.get("jax_paged", {}).get("token_ids_by_request", []))
        ]
        if refs:
            ref_by_shape[tuple(int(length) for length in lengths)] = refs
        if len(lengths) == 1:
            ref_by_len[int(lengths[0])] = refs[0] if refs else {"generated_tokens": [], "first_step_topk": []}
    summary_shape = tuple(int(length) for length in summary.get("run_config", {}).get("input_lens", []))
    shape_refs = ref_by_shape.get(summary_shape)
    rows = []
    for index, row in enumerate(summary.get("rows", [])):
        ref = shape_refs[index] if shape_refs is not None and index < len(shape_refs) else None
        ref_source = "exact_shape" if ref is not None else "single_length"
        if ref is None:
            ref = ref_by_len.get(int(row["prompt_length"]))
        if ref is None:
            rows.append({"name": row["name"], "checked": False, "reason": "missing_reference_length"})
            continue
        ref_tokens = [int(token) for token in ref["generated_tokens"]]
        generated_tokens = [int(token) for token in row["generated_token_ids"]]
        compare_len = min(len(ref_tokens), len(generated_tokens))
        diff = _first_diff(ref_tokens[:compare_len], generated_tokens[:compare_len])
        generated_prefix_match = diff is None
        generated_full_match = generated_prefix_match and len(ref_tokens) == len(generated_tokens)
        first_step_topk = [
            int(item["token_id"])
            for item in (row.get("topk_logprobs_by_step") or [[]])[0]
            if item.get("token_id") is not None
        ]
        ref_topk = [int(token) for token in ref.get("first_step_topk", [])]
        topk_overlap = len(set(first_step_topk) & set(ref_topk)) if first_step_topk and ref_topk else None
        rows.append(
            {
                "name": row["name"],
                "checked": True,
                "reference_source": ref_source,
                "comparison_length": compare_len,
                "generated_length": len(generated_tokens),
                "reference_length": len(ref_tokens),
                "generated_prefix_match": generated_prefix_match,
                "generated_full_match": generated_full_match,
                "first_diff": diff,
                "first_step_top1_match": bool(first_step_topk[:1] == ref_topk[:1]) if first_step_topk and ref_topk else None,
                "first_step_topk_ordered_match": bool(first_step_topk == ref_topk[: len(first_step_topk)])
                if first_step_topk and ref_topk
                else None,
                "first_step_topk_overlap": topk_overlap,
                "reference_first_step_topk": ref_topk,
                "vllm_first_step_topk": first_step_topk,
            }
        )
    checked = [row for row in rows if row.get("checked")]
    return {
        "checked": bool(checked),
        "ok": all(row.get("generated_prefix_match", False) for row in checked),
        "full_length_ok": all(row.get("generated_full_match", False) for row in checked),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    recorder = RunRecorder.create(
        script=Path(__file__).name,
        args=vars(args),
        run_label=args.run_label or f"vllm_{args.mode}",
        profile_dir=args.profile_dir or None,
        run_log=args.run_log or None,
    )
    try:
        summary = run_vllm(args, recorder)
        correctness = compare_reference(summary, args.reference_json)
        summary["correctness"] = correctness
        summary["run"] = recorder.metadata()
        if correctness.get("checked") and not correctness.get("ok"):
            recorder.record_issue(
                summary="vLLM generated tokens diverged from the reference JSON",
                severity="error",
                status="open",
                details=correctness,
                learnings=["vLLM throughput should be reported only with generated-token correctness."],
                resolution="pending",
            )
        if summary["performance"].get("itl_ms_mean") is None:
            recorder.record_issue(
                summary="vLLM offline API does not expose true ITL",
                severity="info",
                status="open",
                details={"itl_source": summary["performance"]["itl_source"]},
                learnings=["Use a streaming vLLM server or AsyncLLMEngine path for real ITL."],
                resolution="pending streaming harness",
            )
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
        recorder.finish(
            status="ok" if correctness.get("ok", True) else "failed_correctness",
            summary={
                "performance": summary["performance"],
                "correctness": correctness,
                "mode": args.mode,
            },
            learnings=["vLLM generated-token correctness and throughput are recorded in one JSON artifact."],
            resolution="Use streaming mode next for true ITL.",
        )
    except Exception as exc:
        recorder.finish_exception(exc)
        raise


if __name__ == "__main__":
    main()
