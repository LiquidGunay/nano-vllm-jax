#!/usr/bin/env python3
"""Real-checkpoint MTP1 correctness and throughput smoke test.

The script uses the public LLMEngine path, loads real MTP weights, then runs
greedy baseline generation and greedy MTP1 generation on the same prompts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.sequence import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="eager")
    parser.add_argument("--max-kv-cache-mb", type=int, default=512)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--prefill-buckets", default="")
    parser.add_argument("--batch-size-buckets", default="1")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compile-mtp-draft", action="store_true")
    parser.add_argument("--debug-spec", action="store_true")
    parser.add_argument("--mtp-position-offset", type=int, default=0)
    parser.add_argument("--mtp-token-source", choices=["generated", "current"], default="generated")
    parser.add_argument("--mtp-hidden-source", choices=["pre_norm", "final_normed"], default="final_normed")
    parser.add_argument("--sweep-alignments", action="store_true")
    return parser.parse_args()


def parse_buckets(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def run_generation(
    engine: LLMEngine,
    prompt: str,
    *,
    mtp1: bool,
    max_tokens: int,
    mtp_position_offset: int = 0,
    mtp_token_source: str = "generated",
    mtp_hidden_source: str = "final_normed",
    compile_mtp_draft: bool = False,
    debug_spec: bool = False,
):
    runner = engine.model_runner
    runner.mtp1_enabled = bool(mtp1 and runner.mtp_enabled and runner.num_speculative_tokens == 1)
    runner.mtp_position_offset = mtp_position_offset
    runner.mtp_token_source = mtp_token_source
    runner.mtp_hidden_source = mtp_hidden_source
    runner.mtp_compile_draft = compile_mtp_draft
    runner.mtp_debug = debug_spec
    runner._mtp1_drafts.clear()
    runner.reset_speculative_stats()

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    result = engine.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    elapsed = time.perf_counter() - t0
    token_ids = list(result["token_ids"])
    return {
        "text": result["text"],
        "token_ids": token_ids,
        "seconds": elapsed,
        "tokens_per_second": len(token_ids) / elapsed if elapsed > 0 else 0.0,
        "speculative": runner.get_speculative_stats(),
    }


def main():
    args = parse_args()
    prompts = args.prompt or ["Tell me a joke about compilers."]
    prefill_buckets = parse_buckets(args.prefill_buckets)
    batch_size_buckets = parse_buckets(args.batch_size_buckets)

    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")
    print("loading_engine_with_mtp_weights=true")
    load_t0 = time.perf_counter()
    engine = LLMEngine(
        args.model,
        backend=args.backend,
        dtype=args.dtype,
        max_kv_cache_bytes=args.max_kv_cache_mb * 1024 * 1024,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_num_seqs=1,
        max_num_batched_tokens=max(prefill_buckets or (64,)),
        prefill_buckets=prefill_buckets,
        batch_size_buckets=batch_size_buckets,
        jax_execution=args.jax_execution,
        num_speculative_tokens=1,
    )
    load_seconds = time.perf_counter() - load_t0
    print(f"load_seconds={load_seconds:.3f}")

    variants = [(args.mtp_token_source, args.mtp_position_offset, args.mtp_hidden_source)]
    if args.sweep_alignments:
        variants = [
            ("generated", -1, "pre_norm"),
            ("generated", 0, "pre_norm"),
            ("generated", 1, "pre_norm"),
            ("current", -1, "pre_norm"),
            ("current", 0, "pre_norm"),
            ("current", 1, "pre_norm"),
            ("generated", -1, "final_normed"),
            ("generated", 0, "final_normed"),
            ("current", 0, "final_normed"),
        ]

    rows = []
    for prompt in prompts:
        if args.warmup:
            warmup_tokens = min(args.max_tokens, 2)
            _ = run_generation(engine, prompt, mtp1=False, max_tokens=warmup_tokens)

        baseline_runs = [
            run_generation(engine, prompt, mtp1=False, max_tokens=args.max_tokens)
            for _ in range(args.repeats)
        ]
        baseline = baseline_runs[-1]
        variant_rows = []
        for token_source, position_offset, hidden_source in variants:
            if args.warmup and args.max_tokens >= 3:
                _ = run_generation(
                    engine,
                    prompt,
                    mtp1=True,
                    max_tokens=3,
                    mtp_position_offset=position_offset,
                    mtp_token_source=token_source,
                    mtp_hidden_source=hidden_source,
                    compile_mtp_draft=args.compile_mtp_draft,
                    debug_spec=args.debug_spec,
                )
            speculative_runs = [
                run_generation(
                    engine,
                    prompt,
                    mtp1=True,
                    max_tokens=args.max_tokens,
                    mtp_position_offset=position_offset,
                    mtp_token_source=token_source,
                    mtp_hidden_source=hidden_source,
                    compile_mtp_draft=args.compile_mtp_draft,
                    debug_spec=args.debug_spec,
                )
                for _ in range(args.repeats)
            ]
            speculative = speculative_runs[-1]
            variant_rows.append({
                "token_source": token_source,
                "position_offset": position_offset,
                "hidden_source": hidden_source,
                "mtp1": speculative,
                "baseline_runs": baseline_runs,
                "mtp1_runs": speculative_runs,
                "correct": baseline["token_ids"] == speculative["token_ids"],
                "speedup": (
                    baseline["seconds"] / speculative["seconds"]
                    if speculative["seconds"] > 0
                    else 0.0
                ),
            })
        rows.append({
            "prompt": prompt,
            "prompt_tokens": len(engine._tokenize(prompt)),
            "baseline": baseline,
            "variants": variant_rows,
        })

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "jax_backend": jax.default_backend(),
        "jax_execution": args.jax_execution,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "compile_mtp_draft": args.compile_mtp_draft,
        "debug_spec": args.debug_spec,
        "load_seconds": load_seconds,
        "all_correct": all(variant["correct"] for row in rows for variant in row["variants"]),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))

    del engine
    gc.collect()
    raise SystemExit(0 if summary["all_correct"] else 1)


if __name__ == "__main__":
    main()
