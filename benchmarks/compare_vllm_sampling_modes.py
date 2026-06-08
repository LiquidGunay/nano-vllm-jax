#!/usr/bin/env python3
"""Compare vLLM greedy and temperature sampling on a token-id manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    normalized = []
    for index, row in enumerate(rows):
        token_ids = row.get("prompt_token_ids") or row.get("input_ids")
        output_len = row.get("output_len")
        if token_ids is None or output_len is None:
            raise ValueError(f"Manifest row {index} must contain prompt_token_ids/input_ids and output_len")
        normalized.append(
            {
                "request_id": str(row.get("request_id", index)),
                "prompt_token_ids": [int(token) for token in token_ids],
                "output_len": int(output_len),
            }
        )
    return normalized


def _gpu_memory_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    return max(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _make_sampling_params(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    logprobs: int,
) -> SamplingParams:
    kwargs: dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "ignore_eos": True,
        "output_kind": RequestOutputKind.DELTA,
    }
    if logprobs > 0:
        kwargs["logprobs"] = int(logprobs)
    return SamplingParams(**kwargs)


async def _drain_prompt(
    engine: AsyncLLMEngine,
    row: dict[str, Any],
    sampling: SamplingParams,
    request_id: str,
) -> None:
    async for _ in engine.generate(
        {"prompt_token_ids": row["prompt_token_ids"]},
        sampling,
        request_id=request_id,
    ):
        pass


async def _consume_prompt(
    engine: AsyncLLMEngine,
    row: dict[str, Any],
    sampling: SamplingParams,
    request_id: str,
    started: float,
) -> dict[str, Any]:
    token_ids: list[int] = []
    token_timestamps: list[float] = []
    async for output in engine.generate(
        {"prompt_token_ids": row["prompt_token_ids"]},
        sampling,
        request_id=request_id,
    ):
        now = time.perf_counter()
        completion = output.outputs[0]
        delta_ids = [int(token) for token in completion.token_ids]
        token_ids.extend(delta_ids)
        token_timestamps.extend([now - started] * len(delta_ids))
    return {"generated_tokens": len(token_ids), "token_timestamps": token_timestamps}


async def _run_mode(
    engine: AsyncLLMEngine,
    rows: list[dict[str, Any]],
    *,
    name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    logprobs: int,
) -> dict[str, Any]:
    warmup_sampling = [
        _make_sampling_params(
            max_tokens=max(1, min(2, int(row["output_len"]))),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
        )
        for row in rows
    ]
    sampling = [
        _make_sampling_params(
            max_tokens=int(row["output_len"]),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
        )
        for row in rows
    ]
    await asyncio.gather(
        *[
            _drain_prompt(engine, row, warmup_sampling[index], f"warm-{name}-{index}")
            for index, row in enumerate(rows)
        ]
    )
    memory_before = _gpu_memory_mb()
    started = time.perf_counter()
    outputs = await asyncio.gather(
        *[
            _consume_prompt(engine, row, sampling[index], f"bench-{name}-{index}", started)
            for index, row in enumerate(rows)
        ]
    )
    elapsed = time.perf_counter() - started
    memory_after = _gpu_memory_mb()
    total_tokens = sum(int(output["generated_tokens"]) for output in outputs)
    ttfts = []
    itls = []
    for output in outputs:
        timestamps = [float(value) for value in output["token_timestamps"]]
        if timestamps:
            ttfts.append(1000.0 * timestamps[0])
            itls.extend(1000.0 * (right - left) for left, right in zip(timestamps, timestamps[1:]))
    return {
        "name": name,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "logprobs": int(logprobs),
        "seconds": elapsed,
        "generated_tokens": total_tokens,
        "output_tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "ttft_ms_p50": _percentile(ttfts, 50),
        "itl_ms_p50": _percentile(itls, 50),
        "gpu_memory_mb_before": memory_before,
        "gpu_memory_mb_after": memory_after,
    }


async def async_main() -> dict[str, Any]:
    args = parse_args()
    rows = _read_manifest(Path(args.manifest_jsonl))
    baseline_memory = _gpu_memory_mb()
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        trust_remote_code=True,
        disable_log_stats=True,
    )
    load_started = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    load_seconds = time.perf_counter() - load_started
    memory_after_load = _gpu_memory_mb()
    modes = [
        {
            "name": "greedy_plain",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": int(args.top_k),
            "logprobs": 0,
        },
        {
            "name": "temperature_plain",
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "logprobs": 0,
        },
    ]
    if int(args.logprobs) > 0:
        modes.extend(
            [
                {
                    "name": f"greedy_logprobs{int(args.logprobs)}",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": int(args.top_k),
                    "logprobs": int(args.logprobs),
                },
                {
                    "name": f"temperature_logprobs{int(args.logprobs)}",
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "logprobs": int(args.logprobs),
                },
            ]
        )
    results = []
    for mode in modes:
        result = await _run_mode(engine, rows, **mode)
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
    summary = {
        "manifest_jsonl": str(Path(args.manifest_jsonl)),
        "request_count": len(rows),
        "model": args.model,
        "dtype": args.dtype,
        "load_seconds": load_seconds,
        "gpu_memory_mb_baseline": baseline_memory,
        "gpu_memory_mb_after_load": memory_after_load,
        "gpu_memory_mb_final": _gpu_memory_mb(),
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    summary = asyncio.run(async_main())
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
