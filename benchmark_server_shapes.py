#!/usr/bin/env python3
"""Server-style shape benchmark for HF, JAX paged attention, and MTP1."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from runtime_paths import configure_compilation_cache, configure_xla_flags
from run_tracking import RunRecorder

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
configure_xla_flags()
configure_compilation_cache()

import jax
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_mtp1_engine import run_generation_batch
from nanovllm_jax.engine.llm_engine import LLMEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--weight-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="jit")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenario names to run; defaults to all.",
    )
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--output-json", default="results/qwen08_server_shape_speed.json")
    parser.add_argument("--max-kv-cache-mb", type=int, default=1024)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--prefill-buckets", default="16,32,64,128")
    parser.add_argument("--batch-size-buckets", default="1,2,4")
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=os.environ.get("NANO_VLLM_JAX_PROFILE", "1") not in {"0", "false", "False", "no", "off"},
        help="Write a JAX profiler trace under /mountpoint/.exp/profiles by default.",
    )
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--profile-dir", default="", help="Directory for per-run JAX profiler traces.")
    parser.add_argument("--run-log", default="", help="Append-only JSONL run journal path.")
    parser.add_argument("--run-label", default="", help="Short label included in the profiler run directory name.")
    return parser.parse_args()


def _parse_ints(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def _torch_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _topk(logits: np.ndarray, k: int) -> dict[str, list]:
    ids = np.argsort(logits.astype(np.float32))[-k:][::-1].astype(np.int32)
    return {
        "ids": ids.tolist(),
        "values": logits[ids].astype(np.float32).tolist(),
    }


def _first_diff(left: list[int], right: list[int]) -> dict | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return {"index": idx, "left": int(a), "right": int(b)}
    if len(left) != len(right):
        idx = min(len(left), len(right))
        return {
            "index": idx,
            "left": int(left[idx]) if idx < len(left) else None,
            "right": int(right[idx]) if idx < len(right) else None,
            "length_mismatch": True,
        }
    return None


def _make_prompt_ids(tokenizer, length: int, seed: str) -> list[int]:
    ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    repeated: list[int] = []
    while len(repeated) < length:
        repeated.extend(ids)
    return [int(token) for token in repeated[:length]]


def _scenarios(tokenizer) -> list[dict]:
    seeds = [
        "The future of artificial intelligence is poised to transform software systems.",
        "Explain the key tradeoffs in paged attention cache management for inference.",
        "Write a concise proof sketch for binary search over a sorted array.",
        "Summarize speculative decoding risks for deterministic generation.",
    ]
    specs = [
        ("single_16x16", [16], 16),
        ("single_64x24", [64], 24),
        ("single_128x24", [128], 24),
        ("batch4_mixed_16_32_64_128x24", [16, 32, 64, 128], 24),
    ]
    scenarios = []
    for name, lengths, max_new_tokens in specs:
        prompts = [
            _make_prompt_ids(tokenizer, length, seeds[index % len(seeds)])
            for index, length in enumerate(lengths)
        ]
        scenarios.append(
            {
                "name": name,
                "prompt_lengths": lengths,
                "max_new_tokens": max_new_tokens,
                "prompts": prompts,
            }
        )
    return scenarios


def _run_hf_batch(model, prompts: list[list[int]], *, max_new_tokens: int, pad_token_id: int, top_k: int) -> dict:
    max_len = max(len(row) for row in prompts)
    input_rows = []
    mask_rows = []
    for row in prompts:
        pad = max_len - len(row)
        input_rows.append([pad_token_id] * pad + row)
        mask_rows.append([0] * pad + [1] * len(row))
    input_ids = torch.tensor(input_rows, dtype=torch.long, device="cuda")
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device="cuda")

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    first_logits = output.scores[0].detach().float().cpu().numpy()
    generated = output.sequences[:, max_len:].detach().cpu().numpy().astype(np.int32).tolist()
    return {
        "seconds": elapsed,
        "generated_tokens": int(len(prompts) * max_new_tokens),
        "tokens_per_second": float(len(prompts) * max_new_tokens / max(elapsed, 1e-9)),
        "token_ids_by_request": generated,
        "topk_by_request": [_topk(first_logits[row], top_k) for row in range(len(prompts))],
        "first_logits_by_request": first_logits,
    }


def _run_jax(engine, scenario: dict, *, mtp1: bool, top_k: int) -> dict:
    result = run_generation_batch(
        engine=engine,
        prompts=scenario["prompts"],
        mtp1=mtp1,
        max_tokens=int(scenario["max_new_tokens"]),
        output_lengths=[int(scenario["max_new_tokens"])] * len(scenario["prompts"]),
        step_profile=True,
        return_step_records=False,
        return_prefill_logits=True,
    )
    logits_rows = result.get("prefill_logits_by_request", [])
    return {
        "seconds": float(result["seconds"]),
        "generated_tokens": int(result["tokens"]),
        "tokens_per_second": float(result["tokens_per_second"]),
        "decode_tokens_per_second": float(result["decode_tokens_per_second"]),
        "prefill_tokens_per_second": float(result["prefill_tokens_per_second"]),
        "token_ids_by_request": result["token_ids_by_request"],
        "topk_by_request": [_topk(np.asarray(row, dtype=np.float32), top_k) for row in logits_rows],
        "first_logits_by_request": [np.asarray(row, dtype=np.float32) for row in logits_rows],
        "speculative_counts": result.get("speculative_counts", {}),
        "speculative": result.get("speculative", {}),
        "step_profile": result.get("step_profile", {}),
    }


def _compare_against_hf(hf: dict, other: dict, *, top_k: int) -> dict:
    rows = []
    generated_matches = []
    for idx, hf_tokens in enumerate(hf["token_ids_by_request"]):
        other_tokens = other["token_ids_by_request"][idx]
        hf_logits = hf["first_logits_by_request"][idx].astype(np.float32)
        other_logits = other["first_logits_by_request"][idx].astype(np.float32)
        diff = other_logits - hf_logits
        hf_top = hf["topk_by_request"][idx]["ids"]
        other_top = other["topk_by_request"][idx]["ids"]
        generated_match = hf_tokens == other_tokens
        generated_matches.append(generated_match)
        rows.append(
            {
                "request_index": idx,
                "prefill_logits_mse": float(np.mean(diff * diff)),
                "prefill_logits_max_abs": float(np.max(np.abs(diff))),
                "top1_match": bool(hf_top[:1] == other_top[:1]),
                "topk_ordered_match": bool(hf_top[:top_k] == other_top[:top_k]),
                "topk_overlap": int(len(set(hf_top[:top_k]) & set(other_top[:top_k]))),
                "generated_match": bool(generated_match),
                "first_generated_diff": _first_diff(hf_tokens, other_tokens),
            }
        )
    return {
        "all_generated_match": bool(all(generated_matches)),
        "all_top1_match": bool(all(row["top1_match"] for row in rows)),
        "all_topk_ordered_match": bool(all(row["topk_ordered_match"] for row in rows)),
        "max_prefill_logits_mse": float(max(row["prefill_logits_mse"] for row in rows)),
        "max_prefill_logits_max_abs": float(max(row["prefill_logits_max_abs"] for row in rows)),
        "rows": rows,
    }


def _compare_jax_vs_mtp(jax_result: dict, mtp_result: dict) -> dict:
    rows = []
    for idx, jax_tokens in enumerate(jax_result["token_ids_by_request"]):
        mtp_tokens = mtp_result["token_ids_by_request"][idx]
        rows.append(
            {
                "request_index": idx,
                "generated_match": bool(jax_tokens == mtp_tokens),
                "first_generated_diff": _first_diff(jax_tokens, mtp_tokens),
            }
        )
    return {
        "all_generated_match": bool(all(row["generated_match"] for row in rows)),
        "rows": rows,
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if key != "first_logits_by_request"}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _main_impl(args: argparse.Namespace, recorder: RunRecorder) -> dict:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu for this benchmark, got {jax.default_backend()!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HF benchmark")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    scenarios = _scenarios(tokenizer)
    if args.scenarios:
        wanted = {name.strip() for name in args.scenarios.split(",") if name.strip()}
        scenarios = [scenario for scenario in scenarios if scenario["name"] in wanted]
        missing = wanted - {scenario["name"] for scenario in scenarios}
        if missing:
            raise ValueError(f"Unknown scenario(s): {sorted(missing)}")
    if not scenarios:
        raise ValueError("No benchmark scenarios selected")
    max_prompt_len = max(max(item["prompt_lengths"]) for item in scenarios)
    max_batch = max(len(item["prompts"]) for item in scenarios)

    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")
    print(f"torch_device={torch.cuda.get_device_name(0)}")
    print(f"dtype={args.dtype} weight_dtype={args.weight_dtype}")
    print(f"compile_cache={os.environ.get('NANO_VLLM_JAX_COMPILE_CACHE_DIR')}")

    print("loading_hf=true")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.weight_dtype),
        trust_remote_code=True,
    )
    if args.dtype == "float32":
        hf_model.float()
    hf_model.eval().to("cuda")

    hf_results: dict[str, dict] = {}
    for scenario in scenarios:
        if args.warmup:
            _ = _run_hf_batch(
                hf_model,
                scenario["prompts"],
                max_new_tokens=min(2, int(scenario["max_new_tokens"])),
                pad_token_id=pad_token_id,
                top_k=args.top_k,
            )
        hf_results[scenario["name"]] = _run_hf_batch(
            hf_model,
            scenario["prompts"],
            max_new_tokens=int(scenario["max_new_tokens"]),
            pad_token_id=pad_token_id,
            top_k=args.top_k,
        )
        print(
            f"hf {scenario['name']} "
            f"tps={hf_results[scenario['name']]['tokens_per_second']:.2f} "
            f"seconds={hf_results[scenario['name']]['seconds']:.3f}"
        )

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("loading_jax_engine=true")
    engine = LLMEngine(
        args.model,
        backend=args.backend,
        dtype=args.dtype,
        weight_dtype=args.weight_dtype,
        max_kv_cache_bytes=args.max_kv_cache_mb * 1024 * 1024,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_num_seqs=max(args.max_num_seqs, max_batch),
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_buckets=_parse_ints(args.prefill_buckets),
        batch_size_buckets=_parse_ints(args.batch_size_buckets),
        max_blocks_per_seq=max(1, (max_prompt_len + max(item["max_new_tokens"] for item in scenarios) + 15) // 16 + 1),
        jax_execution=args.jax_execution,
        num_speculative_tokens=1,
    )
    engine.model_runner.warmup_compilation(max_prefill_len=max_prompt_len, max_batch=max_batch)

    rows = []
    for scenario in scenarios:
        if args.warmup:
            _ = _run_jax(engine, scenario, mtp1=False, top_k=args.top_k)
            _ = _run_jax(engine, scenario, mtp1=True, top_k=args.top_k)
        jax_result = _run_jax(engine, scenario, mtp1=False, top_k=args.top_k)
        mtp_result = _run_jax(engine, scenario, mtp1=True, top_k=args.top_k)
        hf_result = hf_results[scenario["name"]]
        jax_vs_hf = _compare_against_hf(hf_result, jax_result, top_k=args.top_k)
        mtp_vs_hf = _compare_against_hf(hf_result, mtp_result, top_k=args.top_k)
        mtp_vs_jax = _compare_jax_vs_mtp(jax_result, mtp_result)
        speedup_jax_vs_hf = jax_result["tokens_per_second"] / max(hf_result["tokens_per_second"], 1e-9)
        speedup_mtp_vs_jax = mtp_result["tokens_per_second"] / max(jax_result["tokens_per_second"], 1e-9)
        row = {
            "name": scenario["name"],
            "prompt_lengths": scenario["prompt_lengths"],
            "max_new_tokens": scenario["max_new_tokens"],
            "hf": hf_result,
            "jax_paged": jax_result,
            "mtp1": mtp_result,
            "speedup": {
                "jax_vs_hf": float(speedup_jax_vs_hf),
                "mtp1_vs_jax": float(speedup_mtp_vs_jax),
                "mtp1_vs_hf": float(mtp_result["tokens_per_second"] / max(hf_result["tokens_per_second"], 1e-9)),
            },
            "correctness": {
                "jax_vs_hf": jax_vs_hf,
                "mtp1_vs_hf": mtp_vs_hf,
                "mtp1_vs_jax": mtp_vs_jax,
            },
        }
        rows.append(row)
        if not jax_vs_hf["all_generated_match"]:
            recorder.record_issue(
                summary=f"JAX generated tokens diverged from HF for {scenario['name']}",
                severity="error",
                status="open",
                details=jax_vs_hf,
                learnings=["Main-model correctness must remain the first gate before speed work."],
                resolution="pending",
            )
        if not mtp_vs_jax["all_generated_match"]:
            recorder.record_issue(
                summary=f"MTP1 generated tokens diverged from regular JAX for {scenario['name']}",
                severity="error",
                status="open",
                details=mtp_vs_jax,
                learnings=[
                    "MTP correctness must be judged against the regular JAX target path.",
                    "Unsafe one-pass K=1 verifier state can produce visible drift.",
                ],
                resolution="Use exact commit-select by default; only opt into unsafe one-pass for isolated profiling.",
            )
        if speedup_mtp_vs_jax < 1.0:
            recorder.record_issue(
                summary=f"MTP1 is slower than regular JAX for {scenario['name']}",
                severity="info",
                status="open",
                details={
                    "mtp1_vs_jax_speedup": float(speedup_mtp_vs_jax),
                    "jax_tokens_per_second": jax_result["tokens_per_second"],
                    "mtp1_tokens_per_second": mtp_result["tokens_per_second"],
                    "speculative_counts": mtp_result.get("speculative_counts", {}),
                },
                learnings=["Correct MTP1 must beat the regular JAX decode path before it is useful for serving."],
                resolution="pending after correctness is locked",
            )
        print(
            f"{scenario['name']} "
            f"hf={hf_result['tokens_per_second']:.2f} tok/s "
            f"jax={jax_result['tokens_per_second']:.2f} tok/s "
            f"mtp1={mtp_result['tokens_per_second']:.2f} tok/s "
            f"jax_ok={jax_vs_hf['all_generated_match']} "
            f"mtp_ok={mtp_vs_jax['all_generated_match']}"
        )

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "weight_dtype": args.weight_dtype,
        "top_k": args.top_k,
        "jax_backend": jax.default_backend(),
        "jax_execution": args.jax_execution,
        "run": recorder.metadata(),
        "rows": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(f"wrote_json={output_path}")

    del engine
    gc.collect()
    return summary


def _summary_for_journal(summary: dict) -> dict:
    rows = summary.get("rows", [])
    return {
        "model": summary.get("model"),
        "dtype": summary.get("dtype"),
        "weight_dtype": summary.get("weight_dtype"),
        "row_count": len(rows),
        "jax_all_generated_match": all(
            row.get("correctness", {}).get("jax_vs_hf", {}).get("all_generated_match", False)
            for row in rows
        ),
        "mtp_all_generated_match": all(
            row.get("correctness", {}).get("mtp1_vs_jax", {}).get("all_generated_match", False)
            for row in rows
        ),
        "speedups": {
            row.get("name"): row.get("speedup", {})
            for row in rows
        },
    }


def main() -> None:
    args = parse_args()
    recorder = RunRecorder.create(
        script=Path(__file__).name,
        args=vars(args),
        run_label=args.run_label,
        profile_dir=args.profile_dir or None,
        run_log=args.run_log or None,
    )
    recorder.start_jax_profile(enabled=args.profile)
    try:
        summary = _main_impl(args, recorder)
        recorder.stop_jax_profile()
        recorder.finish(
            status="ok",
            summary=_summary_for_journal(summary),
            learnings=[
                "HF, regular JAX paged attention, and MTP1 were measured on the same prompt shapes.",
                "Generated-token and top-k correctness are recorded alongside throughput.",
            ],
            resolution="Open MTP issues are recorded in the run journal when divergence or slowdown is observed.",
        )
    except Exception as exc:
        recorder.stop_jax_profile()
        recorder.finish_exception(exc)
        raise
    finally:
        recorder.stop_jax_profile()


if __name__ == "__main__":
    main()
