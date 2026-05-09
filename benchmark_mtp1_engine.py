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
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
_platform = os.getenv("NANO_VLLM_JAX_JAX_PLATFORMS") or os.getenv("NANO_VLLM_JAX_PLATFORMS")
if _platform:
    os.environ["JAX_PLATFORMS"] = _platform

from nanovllm_jax.engine.llm_engine import LLMEngine
from nanovllm_jax.engine.sequence import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--config-preset",
        choices=["auto", "hf", "qwen3_5_0_8b", "qwen3_5_2b", "qwen3_5_27b"],
        default="auto",
        help=(
            "Model architecture preset. auto infers from --model when possible; "
            "hf reads the model's Hugging Face config.json text_config."
        ),
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument(
        "--prompt-suite",
        choices=["manual", "synthetic", "real", "mixed", "expanded"],
        default="manual",
        help="Use a built-in prompt suite when --prompt is not supplied.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32", "auto"], default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="jit")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-kv-cache-mb", type=int, default=512)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument(
        "--max-blocks-per-seq",
        type=int,
        default=None,
        help=(
            "Static per-sequence block-table width. This bounds decode attention's "
            "compiled KV window independently from total physical cache blocks."
        ),
    )
    parser.add_argument("--prefill-buckets", default="")
    parser.add_argument("--batch-size-buckets", default="1")
    parser.add_argument("--batch-prompts", type=int, default=1,
                        help="Run one batched benchmark with this many prompts")
    parser.add_argument("--prompt-lengths", default="",
                        help="Comma-separated prompt lengths for batched mode")
    parser.add_argument("--prompt-length-min", type=int, default=32)
    parser.add_argument("--prompt-length-max", type=int, default=192)
    parser.add_argument("--warmup", action="store_true", help="Enable compile/timing warmup passes")
    parser.add_argument(
        "--platform",
        default="auto",
        help="JAX platform override (cpu|gpu|tpu). Uses NANO_VLLM_JAX_PLATFORMS env when set.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--compile-mtp-draft", action="store_true")
    parser.add_argument("--debug-spec", action="store_true")
    parser.add_argument("--mtp-position-offset", type=int, default=0)
    parser.add_argument("--mtp-token-source", choices=["generated", "current"], default="generated")
    parser.add_argument("--mtp-hidden-source", choices=["pre_norm", "final_normed"], default="final_normed")
    parser.add_argument("--sweep-alignments", action="store_true")
    parser.add_argument("--exec-log-steps", action="store_true", help="Enable executor per-step logging")
    parser.add_argument("--step-profile", action="store_true", help="Collect per-step decode profiling")
    parser.add_argument("--adaptive-margin", type=float, default=0.0,
                        help="Margin for documented adaptive MTP gating decisions")
    parser.add_argument("--trace-steps", action="store_true",
                        help="Include per-step emitted token deltas in JSON output for correctness debugging")
    parser.add_argument("--hf-offline", action="store_true")
    parser.add_argument("--check-hf-logits", action="store_true",
                        help="Compare HF final-prefill logits against the JAX baseline model before trusting throughput")
    parser.add_argument("--check-next-step-sanity", action="store_true",
                        help="Run an extra max_tokens+1 baseline/MTP pass to verify the next token after the reported MTP boundary")
    parser.add_argument("--hf-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-topk", type=int, default=5)
    parser.add_argument("--hf-max-prompts", type=int, default=1)
    parser.add_argument("--hf-logits-mse-threshold", type=float, default=None)
    parser.add_argument("--hf-logits-cache", default="",
                        help="Optional .npz path for cached full HF final-prefill logits")
    parser.add_argument("--hf-logits-cache-mode", choices=["auto", "read", "refresh"], default="auto",
                        help="HF logits cache mode: auto reads/writes misses, read forbids HF recompute, refresh overwrites")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Run correctness checks but suppress throughput summaries")
    parser.add_argument("--output-json", default="", help="Optional path to write the machine-readable summary")
    parser.add_argument("--show-outputs", action="store_true",
                        help="Include decoded baseline/MTP outputs in each variant row")
    return parser.parse_args()


def choose_dtype(requested: str, backend: str) -> str:
    if requested != "auto":
        return requested
    if backend == "tpu":
        return "bfloat16"
    if backend == "gpu":
        return "bfloat16"
    return "float16"


def parse_buckets(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def parse_prompt_lengths(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def prompt_suite(name: str) -> list[str]:
    synthetic = [
        "red red red red red red red red",
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
        "the the the the the the the the the the the the the the the the",
        "A B C A B C A B C A B C",
    ]
    real = [
        "Write a Python function that parses a CSV file with quoted fields.",
        "Explain why the sky is blue in one paragraph.",
        "Solve: if 3x + 7 = 31, what is x? Show the arithmetic.",
        "Translate to French: The meeting was moved to Thursday afternoon.",
        "Summarize the risks of speculative decoding for deterministic generation.",
        "Complete this JSON object: {\"name\": \"compiler\", \"features\": [",
        "Here is a short story opening: The old radio crackled at midnight...",
        "List three differences between TCP and UDP.",
    ]
    expanded = real + [
        "Analyze this SQL query plan and explain where an index would help: SELECT users.id, COUNT(*) FROM users JOIN events ON users.id = events.user_id GROUP BY users.id;",
        "Write a careful proof sketch for why binary search terminates on a sorted finite array.",
        "Compare greedy decoding, beam search, and speculative decoding for latency-sensitive serving.",
        "Given a Rust enum representing parser states, describe how to handle invalid transitions safely.",
        "Draft a concise incident report for a cache stampede caused by synchronized token refresh.",
        "Explain how rotary positional embeddings affect attention scores at long context lengths.",
        "Convert the following requirements into test cases for an HTTP retry policy with jitter.",
        "Summarize this design tradeoff: store KV cache as paged blocks versus contiguous per-request buffers.",
    ]
    if name == "synthetic":
        return synthetic
    if name == "real":
        return real
    if name == "mixed":
        return synthetic + real
    if name == "expanded":
        return expanded
    return ["Tell me a joke about compilers."]


def _first_token_diff(
    baseline_rows: list[list[int]],
    mtp_rows: list[list[int]],
) -> dict | None:
    for request_index, (baseline, mtp) in enumerate(zip(baseline_rows, mtp_rows)):
        limit = min(len(baseline), len(mtp))
        for token_index in range(limit):
            if baseline[token_index] != mtp[token_index]:
                return {
                    "request_index": request_index,
                    "token_index": token_index,
                    "baseline_token": int(baseline[token_index]),
                    "mtp_token": int(mtp[token_index]),
                }
        if len(baseline) != len(mtp):
            return {
                "request_index": request_index,
                "token_index": limit,
                "baseline_token": int(baseline[limit]) if limit < len(baseline) else None,
                "mtp_token": int(mtp[limit]) if limit < len(mtp) else None,
                "length_mismatch": True,
            }
    if len(baseline_rows) != len(mtp_rows):
        return {
            "request_index": min(len(baseline_rows), len(mtp_rows)),
            "token_index": 0,
            "baseline_token": None,
            "mtp_token": None,
            "row_count_mismatch": True,
        }
    return None


def _torch_dtype(dtype: str):
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _topk_ids(values, k: int) -> list[int]:
    import numpy as np

    return [int(index) for index in np.argsort(values)[-k:][::-1]]


def run_hf_logits_check(
    engine: LLMEngine,
    prompts: list[str | list[int]],
    *,
    args: argparse.Namespace,
    dtype: str,
) -> dict:
    if not args.check_hf_logits:
        return {"checked": False, "ok": True}

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from nanovllm_jax.model import forward
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "error": f"failed to import HF/JAX parity dependencies: {exc}",
        }

    device = "cuda" if args.hf_device == "auto" and torch.cuda.is_available() else args.hf_device
    if device == "auto":
        device = "cpu"
    mse_threshold = (
        float(args.hf_logits_mse_threshold)
        if args.hf_logits_mse_threshold is not None
        else (1e-2 if dtype in {"bfloat16", "float16"} else 1e-4)
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=bool(args.hf_offline),
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=_torch_dtype(dtype),
            trust_remote_code=True,
            local_files_only=bool(args.hf_offline),
        )
        hf_model.eval()
        hf_model.to(device)

        checks = []
        params = engine.model_runner.params
        config = engine.config
        for index, prompt in enumerate(prompts[: max(1, args.hf_max_prompts)]):
            if isinstance(prompt, str):
                encoded = tokenizer(prompt, return_tensors="pt")
                torch_ids = encoded["input_ids"].to(device)
                jax_ids_np = encoded["input_ids"].cpu().numpy()
            else:
                jax_ids_np = np.array([prompt], dtype=np.int64)
                torch_ids = torch.tensor(jax_ids_np, dtype=torch.long, device=device)

            with torch.no_grad():
                hf_logits = hf_model(torch_ids).logits[0, -1, :].float().cpu().numpy()

            jax_logits, _ = forward(
                jnp.array(jax_ids_np, dtype=jnp.int32),
                params,
                config,
                kv_cache_state=None,
                is_prefill=True,
            )
            jax.block_until_ready(jax_logits)
            baseline_logits = np.array(jax_logits[0, -1, :], dtype=np.float32)

            mse = float(np.mean((hf_logits - baseline_logits) ** 2))
            max_abs = float(np.max(np.abs(hf_logits - baseline_logits)))
            hf_top = _topk_ids(hf_logits, args.hf_topk)
            baseline_top = _topk_ids(baseline_logits, args.hf_topk)
            top1_match = bool(hf_top[:1] == baseline_top[:1])
            topk_match = bool(hf_top == baseline_top)
            checks.append({
                "prompt_index": index,
                "prompt_tokens": int(jax_ids_np.shape[1]),
                "mse": mse,
                "max_abs": max_abs,
                "hf_topk": hf_top,
                "baseline_topk": baseline_top,
                "top1_match": top1_match,
                "topk_match": topk_match,
                "topk_overlap": len(set(hf_top) & set(baseline_top)),
                "ok": bool(top1_match and mse <= mse_threshold),
            })

        del hf_model
        if device == "cuda":
            torch.cuda.empty_cache()
        ok = all(item["ok"] for item in checks)
        return {
            "checked": True,
            "ok": bool(ok),
            "device": device,
            "dtype": dtype,
            "topk": args.hf_topk,
            "mse_threshold": mse_threshold,
            "mse_threshold_source": "cli" if args.hf_logits_mse_threshold is not None else "dtype_default",
            "checks": checks,
        }
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "device": device,
            "dtype": dtype,
            "error": str(exc),
        }


def run_hf_logits_check(
    engine: LLMEngine,
    prompts: list[str | list[int]],
    *,
    args: argparse.Namespace,
    dtype: str,
) -> dict:
    if not args.check_hf_logits:
        return {"checked": False, "ok": True}

    try:
        import hashlib
        import json as _json
        import os
        from pathlib import Path

        import jax
        import jax.numpy as jnp
        import numpy as np
        from nanovllm_jax.model import forward
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "error": f"failed to import JAX parity dependencies: {exc}",
        }

    def _prompt_token_ids(prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            return [int(token) for token in engine._tokenize(prompt)]
        return [int(token) for token in prompt]

    def _cache_key(token_ids: list[int]) -> str:
        digest = hashlib.sha256()
        digest.update(str(args.model).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(token_ids, dtype=np.int32).tobytes())
        return digest.hexdigest()

    mse_threshold = (
        float(args.hf_logits_mse_threshold)
        if args.hf_logits_mse_threshold is not None
        else (1e-2 if dtype in {"bfloat16", "float16"} else 1e-4)
    )
    cache_path = str(getattr(args, "hf_logits_cache", "") or "")
    cache_mode = str(getattr(args, "hf_logits_cache_mode", "auto") or "auto")
    prompt_limit = max(1, int(args.hf_max_prompts))
    prompt_records = []
    for index, prompt in enumerate(prompts[:prompt_limit]):
        token_ids = _prompt_token_ids(prompt)
        prompt_records.append({
            "index": index,
            "token_ids": token_ids,
            "ids_array": np.asarray([token_ids], dtype=np.int64),
            "key": _cache_key(token_ids),
        })

    cache_entries: dict[str, dict] = {}
    cache_error = None
    if cache_path and cache_mode != "refresh" and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            metadata = _json.loads(str(loaded["metadata"].tolist()))
            if metadata.get("model") != args.model or metadata.get("dtype") != dtype:
                cache_error = (
                    f"cache metadata mismatch: expected model={args.model} dtype={dtype}, "
                    f"found model={metadata.get('model')} dtype={metadata.get('dtype')}"
                )
            else:
                for entry in metadata.get("entries", []):
                    key = str(entry["key"])
                    cache_entries[key] = {
                        "token_ids": np.asarray(loaded[entry["ids_array"]], dtype=np.int64),
                        "logits": np.asarray(loaded[entry["logits_array"]], dtype=np.float32),
                    }
            loaded.close()
        except Exception as exc:
            cache_error = f"failed to load HF logits cache: {exc}"

    if cache_mode == "read" and cache_error:
        return {
            "checked": True,
            "ok": False,
            "device": "cache",
            "dtype": dtype,
            "error": cache_error,
            "cache": {
                "path": cache_path,
                "mode": cache_mode,
                "hits": 0,
                "misses": len(prompt_records),
                "written": False,
            },
        }

    cache_hits = 0
    missing_records = []
    for record in prompt_records:
        cached = cache_entries.get(record["key"])
        if cached is not None and np.array_equal(cached["token_ids"], np.asarray(record["token_ids"], dtype=np.int64)):
            cache_hits += 1
        else:
            missing_records.append(record)

    device = "cache"
    if missing_records:
        if cache_mode == "read":
            return {
                "checked": True,
                "ok": False,
                "device": "cache",
                "dtype": dtype,
                "error": f"HF logits cache is missing {len(missing_records)} prompt(s)",
                "missing_keys": [record["key"] for record in missing_records],
                "cache": {
                    "path": cache_path,
                    "mode": cache_mode,
                    "hits": cache_hits,
                    "misses": len(missing_records),
                    "written": False,
                },
            }
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except Exception as exc:
            return {
                "checked": True,
                "ok": False,
                "device": "unavailable",
                "dtype": dtype,
                "error": f"failed to import HF dependencies for cache misses: {exc}",
            }

        device = "cuda" if args.hf_device == "auto" and torch.cuda.is_available() else args.hf_device
        if device == "auto":
            device = "cpu"
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=_torch_dtype(dtype),
                trust_remote_code=True,
                local_files_only=bool(args.hf_offline),
            )
            hf_model.eval()
            hf_model.to(device)
            for record in missing_records:
                torch_ids = torch.tensor(record["ids_array"], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = hf_model(torch_ids).logits[0, -1, :].float().cpu().numpy()
                cache_entries[record["key"]] = {
                    "token_ids": np.asarray(record["token_ids"], dtype=np.int64),
                    "logits": np.asarray(logits, dtype=np.float32),
                }
            del hf_model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            return {
                "checked": True,
                "ok": False,
                "device": device,
                "dtype": dtype,
                "error": str(exc),
            }

    cache_written = False
    if cache_path and missing_records and cache_mode in {"auto", "refresh"}:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            arrays = {}
            entries = []
            for slot, (key, entry) in enumerate(cache_entries.items()):
                ids_name = f"ids_{slot}"
                logits_name = f"logits_{slot}"
                arrays[ids_name] = np.asarray(entry["token_ids"], dtype=np.int64)
                arrays[logits_name] = np.asarray(entry["logits"], dtype=np.float32)
                entries.append({
                    "key": key,
                    "prompt_tokens": int(arrays[ids_name].shape[0]),
                    "ids_array": ids_name,
                    "logits_array": logits_name,
                })
            metadata = {
                "version": 1,
                "model": args.model,
                "dtype": dtype,
                "entries": entries,
            }
            np.savez_compressed(
                cache_path,
                metadata=np.asarray(_json.dumps(metadata)),
                **arrays,
            )
            cache_written = True
        except Exception as exc:
            return {
                "checked": True,
                "ok": False,
                "device": device,
                "dtype": dtype,
                "error": f"failed to write HF logits cache: {exc}",
            }

    try:
        baseline_probe = run_generation_batch(
            engine,
            [record["token_ids"] for record in prompt_records],
            mtp1=False,
            max_tokens=1,
            return_step_records=False,
            return_prefill_logits=True,
        )
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "device": device,
            "dtype": dtype,
            "error": f"failed to collect engine prefill logits: {exc}",
            "cache": {
                "path": cache_path,
                "mode": cache_mode,
                "hits": cache_hits,
                "misses": len(missing_records),
                "written": cache_written,
                "entries": len(cache_entries),
                "load_warning": cache_error,
            },
        }

    checks = []
    baseline_logits_rows = baseline_probe.get("prefill_logits_by_request", [])
    baseline_token_rows = baseline_probe.get("token_ids_by_request", [])
    for record in prompt_records:
        prompt_index = int(record["index"])
        hf_logits = cache_entries[record["key"]]["logits"]
        baseline_logits = (
            baseline_logits_rows[prompt_index]
            if prompt_index < len(baseline_logits_rows)
            else None
        )
        baseline_tokens = (
            baseline_token_rows[prompt_index]
            if prompt_index < len(baseline_token_rows)
            else []
        )
        if baseline_logits is None:
            checks.append({
                "prompt_index": prompt_index,
                "prompt_tokens": int(len(record["token_ids"])),
                "hf_topk": _topk_ids(hf_logits, args.hf_topk),
                "baseline_topk": [],
                "top1_match": False,
                "topk_match": False,
                "topk_overlap": 0,
                "baseline_first_token": None,
                "baseline_first_token_match": False,
                "ok": False,
                "error": "missing engine prefill logits",
                "cache_key": record["key"],
            })
            continue

        mse = float(np.mean((hf_logits - baseline_logits) ** 2))
        max_abs = float(np.max(np.abs(hf_logits - baseline_logits)))
        hf_top = _topk_ids(hf_logits, args.hf_topk)
        baseline_top = _topk_ids(baseline_logits, args.hf_topk)
        top1_match = bool(hf_top[:1] == baseline_top[:1])
        topk_match = bool(hf_top == baseline_top)
        baseline_first_token = int(baseline_tokens[0]) if baseline_tokens else None
        baseline_first_token_match = bool(baseline_first_token == hf_top[0]) if hf_top else False
        checks.append({
            "prompt_index": prompt_index,
            "prompt_tokens": int(len(record["token_ids"])),
            "mse": mse,
            "max_abs": max_abs,
            "hf_topk": hf_top,
            "baseline_topk": baseline_top,
            "top1_match": top1_match,
            "topk_match": topk_match,
            "topk_overlap": len(set(hf_top) & set(baseline_top)),
            "baseline_first_token": baseline_first_token,
            "baseline_first_token_match": baseline_first_token_match,
            "ok": bool(top1_match and baseline_first_token_match and mse <= mse_threshold),
            "cache_key": record["key"],
        })

    ok = all(item["ok"] for item in checks)
    return {
        "checked": True,
        "ok": bool(ok),
        "device": device,
        "dtype": dtype,
        "topk": args.hf_topk,
        "mse_threshold": mse_threshold,
        "mse_threshold_source": "cli" if args.hf_logits_mse_threshold is not None else "dtype_default",
        "checks": checks,
        "cache": {
            "path": cache_path,
            "mode": cache_mode,
            "hits": cache_hits,
            "misses": len(missing_records),
            "written": cache_written,
            "entries": len(cache_entries),
            "load_warning": cache_error,
        },
    }


def apply_correctness_gate(summary: dict, *, args: argparse.Namespace, hf_logits_check: dict) -> dict:
    mtp_exact = bool(summary.get("all_correct", False))
    hf_checked = bool(hf_logits_check.get("checked", False))
    hf_ok = bool(hf_logits_check.get("ok", False))
    next_step_checked = False
    next_step_values = []
    for row in summary.get("rows", []):
        for variant in row.get("variants", []):
            check = variant.get("next_step_sanity_check", {})
            if bool(check.get("checked", False)):
                next_step_checked = True
                next_step_values.append(bool(check.get("ok", False)))
    next_step_ok = bool(all(next_step_values)) if next_step_checked else hf_ok
    all_correct = bool(mtp_exact and hf_ok and next_step_ok)
    timed_results_valid = bool(all_correct and not args.correctness_only)
    summary["mtp_exact_token_match"] = mtp_exact
    summary["hf_logits_check"] = hf_logits_check
    summary["all_correct"] = all_correct
    summary["correctness"] = {
        "hf_logits_checked": hf_checked,
        "hf_logits_ok": hf_ok,
        "next_step_logit_sanity_checked": next_step_checked,
        "next_step_logit_sanity": next_step_ok,
        "baseline_greedy_is_canonical": hf_ok,
        "mtp_exact_token_match": mtp_exact,
        "all_correct": all_correct,
    }
    summary["throughput_valid"] = timed_results_valid
    summary["timed_results_valid"] = timed_results_valid
    for row in summary.get("rows", []):
        for variant in row.get("variants", []):
            check = variant.get("next_step_sanity_check", {})
            variant_next_ok = (
                bool(check.get("ok", False))
                if bool(check.get("checked", False))
                else next_step_ok
            )
            variant["next_step_logit_sanity"] = variant_next_ok
            variant["exact_token_match"] = bool(variant.get("correct", False))
            variant["timed_results_valid"] = bool(
                variant.get("correct", False)
                and variant_next_ok
                and hf_ok
                and not args.correctness_only
            )
            variant["throughput_valid"] = variant["timed_results_valid"]
            if "timed_results" in variant:
                variant["timed_results"]["valid"] = variant["timed_results_valid"]
                variant["timed_results"]["next_step_logit_sanity"] = variant_next_ok
                variant["timed_results"]["exact_token_match"] = bool(variant.get("correct", False))
    if not timed_results_valid:
        reason = "correctness_only" if args.correctness_only and all_correct else "correctness_failed"
        summary["throughput_suppressed_reason"] = reason
        summary["timed_results_invalid_reason"] = reason
        if "throughput_summary" in summary:
            summary["raw_throughput_summary"] = summary["throughput_summary"]
            summary["throughput_summary"] = None
        for row in summary.get("rows", []):
            for variant in row.get("variants", []):
                variant["throughput_valid"] = False
                variant["timed_results_valid"] = False
                variant["throughput_suppressed_reason"] = reason
                variant["timed_results_invalid_reason"] = reason
                if "timed_results" in variant:
                    variant["raw_timed_results"] = variant["timed_results"]
                    variant["timed_results"] = {
                        "valid": False,
                        "invalid_reason": reason,
                    }
    return summary


def infer_config_preset(model_name: str, requested: str) -> dict:
    if requested == "auto":
        lowered = model_name.lower()
        if "27b" in lowered:
            requested = "qwen3_5_27b"
        elif "2b" in lowered:
            requested = "qwen3_5_2b"
        elif "0.8b" in lowered or "0_8b" in lowered:
            requested = "qwen3_5_0_8b"
        else:
            requested = "hf"

    from dataclasses import asdict
    from nanovllm_jax.config import Qwen3_5Config

    runtime_fields = {
        "dtype",
        "block_size",
        "num_kvcache_blocks",
        "max_kv_cache_bytes",
        "max_num_seqs",
        "max_num_batched_tokens",
        "eos",
        "prefill_buckets",
        "batch_size_buckets",
        "max_blocks_per_seq",
        "jax_execution",
        "num_speculative_tokens",
    }

    if requested == "hf":
        from pathlib import Path
        from huggingface_hub import snapshot_download

        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        snapshot = Path(
            snapshot_download(
                repo_id=model_name,
                allow_patterns=["config.json"],
                local_files_only=local_only,
            )
        )
        hf_config = json.loads((snapshot / "config.json").read_text())
        text = hf_config.get("text_config", hf_config)
        vision = hf_config.get("vision_config", {})
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}

        mapped = {
            "vocab_size": text.get("vocab_size"),
            "hidden_size": text.get("hidden_size"),
            "intermediate_size": text.get("intermediate_size"),
            "num_hidden_layers": text.get("num_hidden_layers"),
            "num_attention_heads": text.get("num_attention_heads"),
            "num_key_value_heads": text.get("num_key_value_heads"),
            "head_dim": text.get("head_dim"),
            "linear_num_key_heads": text.get("linear_num_key_heads"),
            "linear_num_value_heads": text.get("linear_num_value_heads"),
            "linear_key_head_dim": text.get("linear_key_head_dim"),
            "linear_value_head_dim": text.get("linear_value_head_dim"),
            "linear_conv_kernel_size": text.get("linear_conv_kernel_dim", text.get("linear_conv_kernel_size")),
            "use_qk_norm_in_gdn": text.get("use_qk_norm_in_gdn", True),
            "rope_theta": rope.get("rope_theta", text.get("rope_theta")),
            "partial_rotary_factor": rope.get("partial_rotary_factor", text.get("partial_rotary_factor")),
            "max_position_embeddings": text.get("max_position_embeddings"),
            "mrope_section": tuple(rope["mrope_section"]) if "mrope_section" in rope else None,
            "layer_types": tuple(text["layer_types"]) if "layer_types" in text else None,
            "hidden_act": text.get("hidden_act"),
            "rms_norm_eps": text.get("rms_norm_eps"),
            "attention_dropout": text.get("attention_dropout"),
            "attention_bias": text.get("attention_bias"),
            "tie_word_embeddings": hf_config.get(
                "tie_word_embeddings",
                text.get("tie_word_embeddings"),
            ),
            "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
            "mtp_use_dedicated_embeddings": text.get("mtp_use_dedicated_embeddings"),
            "vision_depth": vision.get("depth"),
            "vision_hidden_size": vision.get("hidden_size"),
            "vision_num_heads": vision.get("num_heads"),
            "vision_patch_size": vision.get("patch_size"),
            "vision_out_hidden_size": vision.get("out_hidden_size"),
        }
        fields = set(Qwen3_5Config.__dataclass_fields__)
        return {
            key: value
            for key, value in mapped.items()
            if key in fields and key not in runtime_fields and value is not None
        }

    factory = getattr(Qwen3_5Config, requested)
    config = factory()
    fields = set(Qwen3_5Config.__dataclass_fields__)
    return {
        key: value
        for key, value in asdict(config).items()
        if key in fields and key not in runtime_fields
    }


def make_prompt_lengths(
    requested: tuple[int, ...],
    count: int,
    min_tokens: int,
    max_tokens: int,
) -> list[int]:
    if requested:
        lengths = list(requested)
        if len(lengths) < count:
            repeats = count // len(lengths)
            tail = count % len(lengths)
            lengths = lengths * repeats + lengths[:tail]
        elif len(lengths) > count:
            lengths = lengths[:count]
    else:
        if count <= 1:
            lengths = [min_tokens]
        else:
            span = max(1, max_tokens - min_tokens)
            lengths = [
                min_tokens + (i * span) // (count - 1)
                for i in range(count)
            ]
    return [max(1, int(v)) for v in lengths]


def make_token_prompts_from_lengths(engine, lengths: list[int], seed_text: str | list[str]) -> list[list[int]]:
    seed_texts = seed_text if isinstance(seed_text, list) else [seed_text]
    seed_token_rows = [engine._tokenize(text) for text in seed_texts]
    seed_token_rows = [tokens for tokens in seed_token_rows if tokens]
    if not seed_token_rows:
        seed_token_rows = [[1]]

    block_size = getattr(engine, "block_size", None) or getattr(engine.config, "block_size", 1)
    max_blocks_per_seq = getattr(engine.config, "max_blocks_per_seq", None)
    if max_blocks_per_seq is None:
        max_prompt_tokens = max(lengths)
    else:
        max_prompt_tokens = max_blocks_per_seq * block_size
    prompts: list[list[int]] = []
    for index, length in enumerate(lengths):
        seed_tokens = seed_token_rows[index % len(seed_token_rows)]
        capped_len = min(length, max_prompt_tokens)
        repeats = (capped_len + len(seed_tokens) - 1) // len(seed_tokens)
        prompts.append((seed_tokens * repeats)[:capped_len])
    return prompts


def _model_size_label(model: str) -> str:
    lowered = model.lower()
    for marker in ("27b", "14b", "7b", "4b", "2b", "1.5b", "0.8b"):
        if marker in lowered:
            return marker
    return model.rsplit("/", 1)[-1]


def _stats_key(args: argparse.Namespace, *, dtype: str, backend: str) -> dict:
    batch_buckets = parse_buckets(getattr(args, "batch_size_buckets", "") or "")
    batch_bucket = max(batch_buckets) if batch_buckets else max(1, int(getattr(args, "batch_prompts", 1)))
    return {
        "batch_bucket": int(batch_bucket),
        "model_size": _model_size_label(str(args.model)),
        "dtype": dtype,
        "backend": backend,
        "max_blocks_per_seq": getattr(args, "max_blocks_per_seq", None),
    }


def _adaptive_gating_decision(baseline: dict, speculative: dict, *, margin: float) -> dict:
    baseline_decode_tps = float(baseline.get("decode_tokens_per_second", 0.0))
    speculative_decode_tps = float(speculative.get("decode_tokens_per_second", 0.0))
    measured_decode_speedup = speculative_decode_tps / max(1e-9, baseline_decode_tps)
    baseline_decode_ms = 1000.0 / max(1e-9, baseline_decode_tps)
    speculative_decode_ms = 1000.0 / max(1e-9, speculative_decode_tps)
    baseline_step_ms = float(baseline.get("step_profile", {}).get("all_steps", {}).get("seconds_per_token", 0.0)) * 1000.0
    speculative_step_ms = float(speculative.get("step_profile", {}).get("all_steps", {}).get("seconds_per_token", 0.0)) * 1000.0
    stats = speculative.get("speculative", {})
    accept_rate = float(stats.get("drafts_accepted", 0)) / max(1.0, float(stats.get("drafts_proposed", 0)))
    legacy_acceptance_formula = (
        baseline_step_ms / max(1e-9, speculative_step_ms)
    ) * (1.0 + accept_rate)
    should_enable = measured_decode_speedup >= 1.0 + float(margin)
    return {
        "formula": "measured_decode_speedup = mtp_decode_tokens_per_second / baseline_decode_tokens_per_second; should_enable = measured_decode_speedup >= 1.0 + margin",
        "timing_scope": "decode_tokens_per_second",
        "baseline_decode_tokens_per_second": baseline_decode_tps,
        "speculative_decode_tokens_per_second": speculative_decode_tps,
        "baseline_decode_ms_per_token": baseline_decode_ms,
        "speculative_decode_ms_per_token": speculative_decode_ms,
        "baseline_step_ms_per_token": baseline_step_ms,
        "speculative_step_ms_per_token": speculative_step_ms,
        "accept_rate": accept_rate,
        "margin": float(margin),
        "measured_decode_speedup": measured_decode_speedup,
        "predicted_speedup": measured_decode_speedup,
        "legacy_acceptance_formula_speedup": legacy_acceptance_formula,
        "legacy_formula_note": "diagnostic only; it can overestimate exact commit-select by multiplying an already-emitted-token timing ratio by acceptance",
        "should_enable": bool(should_enable),
        "structural_limit": "Exact sequential commit-select K=1 needs at least one target-model forward for each emitted token plus MTP overhead; it should be enabled only when measured emitted-token throughput beats baseline by the configured margin.",
    }


def _benchmark_modes_report() -> list[dict]:
    return [
        {"mode": "baseline", "status": "measured", "description": "No speculative decoding."},
        {"mode": "all-or-none MTP", "status": "measured_by_current_mtp_row", "description": "Fused verifier result is only useful when the configured MTP path accepts as implemented by the runner."},
        {"mode": "rowwise MTP", "status": "default_for_exact_commit_select_serving", "description": "K=1 exact commit-select serving uses NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY=rowwise by default unless explicitly overridden."},
        {"mode": "groupwise g=2", "status": "not_implemented_in_harness", "description": "Harness reports the required mode but does not yet split speculative batches into fixed groups of 2."},
        {"mode": "groupwise g=4", "status": "not_implemented_in_harness", "description": "Harness reports the required mode but does not yet split speculative batches into fixed groups of 4."},
        {"mode": "adaptive MTP gating", "status": "documented_and_reported", "description": "Reports measured decode-token throughput gating; exact commit-select enables only when observed decode speedup beats margin."},
    ]


def _percentile_ms(values: list[float], p: float) -> float:
    """Return a percentile in milliseconds for a list of seconds values."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0] * 1000.0
    values_ms = sorted(v * 1000.0 for v in values)
    idx = (len(values_ms) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values_ms) - 1)
    if lo == hi:
        return values_ms[int(idx)]
    alpha = idx - lo
    return values_ms[lo] * (1.0 - alpha) + values_ms[hi] * alpha


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[int(idx)]
    alpha = idx - lo
    return sorted_values[lo] * (1.0 - alpha) + sorted_values[hi] * alpha


def _summarize_branch(step_records: list[dict]) -> dict:
    count = len(step_records)
    if not count:
        empty_phase_ms = {
            "schedule_ms": 0.0,
            "run_ms": 0.0,
            "postprocess_ms": 0.0,
            "release_ms": 0.0,
        }
        empty_rollup = {
            "host_ms": 0.0,
            "device_ms": 0.0,
            "postprocess_ms": 0.0,
            "release_ms": 0.0,
        }
        return {
            "count": 0,
            "steps_seconds_total": 0.0,
            "tokens": 0,
            "decode_tokens": 0,
            "decode_seconds": 0.0,
            "seconds_per_step": 0.0,
            "seconds_per_token": 0.0,
            "decode_tokens_per_second": 0.0,
            "ms_per_step_p50": 0.0,
            "ms_per_step_p95": 0.0,
            "ms_per_token_p50": 0.0,
            "ms_per_token_p95": 0.0,
            "ms_per_decode_token_p50": 0.0,
            "ms_per_decode_token_p95": 0.0,
            "inter_token_latency_ms_p50": 0.0,
            "inter_token_latency_ms_p95": 0.0,
            "phase_ms": empty_phase_ms,
            "phase_ms_avg_per_step": empty_phase_ms,
            "host_device_postprocess_ms": empty_rollup,
            "phase_ms_rollup": empty_rollup,
        }

    step_seconds = [r["seconds"] for r in step_records]
    token_counts = [abs(r["num_tokens"]) for r in step_records]
    decode_records = [r for r in step_records if r["num_tokens"] < 0]
    decode_seconds = [r["seconds"] for r in decode_records]
    decode_tokens = [abs(r["num_tokens"]) for r in decode_records]
    decode_total_seconds = sum(decode_seconds)
    decode_total_tokens = sum(decode_tokens)
    decode_token_ms = [seconds * 1000.0 / max(1, tokens) for seconds, tokens in zip(decode_seconds, decode_tokens)]
    total_seconds = sum(step_seconds)
    total_tokens = sum(token_counts)
    token_ms = [seconds * 1000.0 / max(1, tokens) for seconds, tokens in zip(step_seconds, token_counts)]
    phase_ms = {
        "schedule_ms": sum(item.get("phase_ms_schedule", 0.0) for item in step_records),
        "run_ms": sum(item.get("phase_ms_run", 0.0) for item in step_records),
        "postprocess_ms": sum(item.get("phase_ms_postprocess", 0.0) for item in step_records),
        "release_ms": sum(item.get("phase_ms_release", 0.0) for item in step_records),
    }
    phase_rollup = {
        "host_ms": phase_ms["schedule_ms"] + phase_ms["postprocess_ms"] + phase_ms["release_ms"],
        "device_ms": phase_ms["run_ms"],
        "postprocess_ms": phase_ms["postprocess_ms"],
        "release_ms": phase_ms["release_ms"],
    }
    decode_latency_p50 = _percentile(decode_token_ms, 0.50)
    decode_latency_p95 = _percentile(decode_token_ms, 0.95)

    return {
        "count": count,
        "steps_seconds_total": total_seconds,
        "tokens": total_tokens,
        "decode_tokens": decode_total_tokens,
        "decode_seconds": decode_total_seconds,
        "seconds_per_step": total_seconds / count,
        "seconds_per_token": total_seconds / max(1, total_tokens),
        "decode_tokens_per_second": decode_total_tokens / max(1e-9, decode_total_seconds),
        "ms_per_step_p50": _percentile_ms(step_seconds, 0.50),
        "ms_per_step_p95": _percentile_ms(step_seconds, 0.95),
        "ms_per_token_p50": _percentile(token_ms, 0.50),
        "ms_per_token_p95": _percentile(token_ms, 0.95),
        "ms_per_decode_token_p50": decode_latency_p50,
        "ms_per_decode_token_p95": decode_latency_p95,
        "inter_token_latency_ms_p50": decode_latency_p50,
        "inter_token_latency_ms_p95": decode_latency_p95,
        "phase_ms": phase_ms,
        "phase_ms_avg_per_step": {
            "schedule_ms": phase_ms["schedule_ms"] / count,
            "run_ms": phase_ms["run_ms"] / count,
            "postprocess_ms": phase_ms["postprocess_ms"] / count,
            "release_ms": phase_ms["release_ms"] / count,
        },
        "host_device_postprocess_ms": phase_rollup,
        "phase_ms_rollup": phase_rollup,
    }


def reset_engine_runtime(engine: LLMEngine):
    """Reset per-request runtime state while preserving loaded weights/JIT cache."""
    from itertools import count

    from nanovllm_jax.engine.block_manager import BlockManager
    from nanovllm_jax.engine.sequence import Sequence

    scheduler = engine.scheduler
    scheduler.waiting.clear()
    scheduler.running.clear()
    scheduler.block_manager = BlockManager(
        engine.config.num_kvcache_blocks,
        engine.config.block_size,
    )
    scheduler.last_num_generated_tokens = 0
    Sequence.counter = count()

    runner = engine.model_runner
    runner._mtp1_drafts.clear()
    runner._last_prefill_logits_by_seq = {}
    runner.reset_speculative_stats()
    if hasattr(runner, "hybrid_states"):
        runner.hybrid_states.clear()
    if hasattr(runner, "_hybrid_slots"):
        runner._hybrid_slots.clear()
    if hasattr(runner, "_max_hybrid_slots"):
        runner._free_hybrid_slots = list(range(runner._max_hybrid_slots))
        for slot in range(runner._max_hybrid_slots):
            runner._zero_hybrid_slot(slot)


def run_generation_batch(
    engine: LLMEngine,
    prompts: list[str | list[int]],
    *,
    mtp1: bool,
    max_tokens: int,
    mtp_position_offset: int = 0,
    mtp_token_source: str = "generated",
    mtp_hidden_source: str = "final_normed",
    compile_mtp_draft: bool = False,
    debug_spec: bool = False,
    step_profile: bool = False,
    return_step_records: bool = True,
    return_prefill_logits: bool = False,
):
    reset_engine_runtime(engine)
    runner = engine.model_runner
    runner.mtp1_enabled = bool(mtp1 and runner.mtp_enabled and runner.num_speculative_tokens > 0)
    runner.mtp_position_offset = mtp_position_offset
    runner.mtp_token_source = mtp_token_source
    runner.mtp_hidden_source = mtp_hidden_source
    runner.mtp_compile_draft = compile_mtp_draft
    runner.mtp_debug = debug_spec
    runner._mtp1_drafts.clear()
    runner._last_prefill_logits_by_seq = {}
    runner.reset_speculative_stats()

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    if not prompts:
        raise ValueError("prompts list must not be empty")

    scheduler = engine.scheduler
    phase_deltas = {
        "schedule": [],
        "run": [],
        "postprocess": [],
        "release": [],
    }

    def _wrap_with_timing(method_name: str, func, phase: str):
        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                phase_deltas[phase].append((time.perf_counter() - t0) * 1000.0)

        _wrapped.__name__ = method_name
        return _wrapped

    orig_schedule = scheduler.schedule
    orig_run = runner.run
    orig_postprocess = scheduler.postprocess
    orig_release = runner.release

    if step_profile:
        scheduler.schedule = _wrap_with_timing("schedule", orig_schedule, "schedule")
        runner.run = _wrap_with_timing("run", orig_run, "run")
        scheduler.postprocess = _wrap_with_timing("postprocess", orig_postprocess, "postprocess")
        runner.release = _wrap_with_timing("release", orig_release, "release")
    else:
        scheduler.schedule = orig_schedule
        runner.run = orig_run
        scheduler.postprocess = orig_postprocess
        runner.release = orig_release

    def _safe_pop_ms(key: str) -> float:
        values = phase_deltas[key]
        if not values:
            return 0.0
        return values.pop(0)

    def _active_trace_outputs():
        seen: set[int] = set()
        trace_outputs = []
        for attr in ("running", "waiting", "finished"):
            collection = getattr(scheduler, attr, None)
            if collection is None:
                continue
            values = collection.values() if isinstance(collection, dict) else collection
            for seq in values:
                seq_id = getattr(seq, "seq_id", getattr(seq, "id", None))
                completion = getattr(seq, "completion_token_ids", None)
                if completion is None:
                    completion = getattr(seq, "output_token_ids", None)
                if seq_id is None or completion is None:
                    continue
                seq_id = int(seq_id)
                if seq_id in seen:
                    continue
                seen.add(seq_id)
                trace_outputs.append((seq_id, list(completion)))
        return trace_outputs

    try:
        for prompt in prompts:
            engine.add_request(prompt, sampling)

        step_records = []
        elapsed = 0.0
        completion_by_seq: dict[int, list[int]] = {}
        last_completion_lens: dict[int, int] = {}
        accepted_steps: list[dict] = []
        rejected_steps: list[dict] = []
        fallback_steps: list[dict] = []
        prefill_steps: list[dict] = []

        while not engine.is_finished():
            pre = runner.get_speculative_stats()
            t0 = time.perf_counter()
            _outputs, num_tokens = engine.step()
            step_elapsed = time.perf_counter() - t0
            elapsed += step_elapsed
            post = runner.get_speculative_stats()

            schedule_ms = _safe_pop_ms("schedule")
            run_ms = _safe_pop_ms("run")
            postprocess_ms = _safe_pop_ms("postprocess")
            release_ms = _safe_pop_ms("release")

            num_tokens = int(num_tokens)
            step_record = {
                "num_tokens": num_tokens,
                "seconds": step_elapsed,
                "seconds_per_token": step_elapsed / max(1, abs(num_tokens)),
                "prefill": bool(num_tokens > 0),
                "phase_ms_schedule": schedule_ms,
                "phase_ms_run": run_ms,
                "phase_ms_postprocess": postprocess_ms,
                "phase_ms_release": release_ms,
            }

            if num_tokens < 0:
                delta_accepted = post["drafts_accepted"] - pre["drafts_accepted"]
                delta_rejected = post["drafts_rejected"] - pre["drafts_rejected"]
                if delta_accepted > 0:
                    branch = "mtp_accepted"
                    accepted_steps.append(step_record)
                elif delta_rejected > 0:
                    branch = "mtp_rejected"
                    rejected_steps.append(step_record)
                else:
                    branch = "fallback"
                    fallback_steps.append(step_record)
                step_record["mtp_branch"] = branch
                step_record["drafts_accepted_delta"] = delta_accepted
                step_record["drafts_rejected_delta"] = delta_rejected
                step_record["drafts_proposed_delta"] = post["drafts_proposed"] - pre["drafts_proposed"]
                step_record["bonus_tokens_delta"] = post["bonus_tokens"] - pre["bonus_tokens"]
            else:
                step_record["mtp_branch"] = "prefill"
                prefill_steps.append(step_record)

            if return_step_records:
                step_outputs = []
                trace_outputs = _active_trace_outputs()
                if not trace_outputs:
                    trace_outputs = [(int(seq_id), list(completion)) for seq_id, completion in _outputs]
                for seq_id, completion in trace_outputs:
                    seq_id = int(seq_id)
                    previous_len = last_completion_lens.get(seq_id, 0)
                    current_completion = list(completion)
                    step_outputs.append({
                        "seq_id": seq_id,
                        "completion_len": len(current_completion),
                        "delta_tokens": current_completion[previous_len:],
                    })
                    last_completion_lens[seq_id] = len(current_completion)
                step_record["outputs"] = step_outputs

            step_records.append(step_record)

            for seq_id, completion in _outputs:
                completion_by_seq[int(seq_id)] = list(completion)
    finally:
        scheduler.schedule = orig_schedule
        runner.run = orig_run
        scheduler.postprocess = orig_postprocess
        runner.release = orig_release

    if len(completion_by_seq) != len(prompts):
        raise RuntimeError(f"generation produced {len(completion_by_seq)} outputs for {len(prompts)} prompts")

    ordered_ids = sorted(completion_by_seq)
    token_id_rows = [completion_by_seq[seq_id] for seq_id in ordered_ids]
    text_rows = [engine._detokenize(token_ids) for token_ids in token_id_rows]
    prefill_logits_by_request = []
    if return_prefill_logits:
        import jax
        import numpy as np

        captured_logits = getattr(runner, "_last_prefill_logits_by_seq", {})
        for seq_id in ordered_ids:
            logits = captured_logits.get(int(seq_id))
            prefill_logits_by_request.append(
                None if logits is None else np.asarray(jax.device_get(logits), dtype=np.float32)
            )
    completion_counts = [len(token_ids) for token_ids in token_id_rows]
    completion_total = sum(completion_counts)
    decode_steps = [r for r in step_records if r["num_tokens"] < 0]
    prefill_steps = [r for r in step_records if r["num_tokens"] > 0]
    decode_seconds = sum(r["seconds"] for r in decode_steps)
    prefill_steps_seconds = sum(r["seconds"] for r in prefill_steps)
    all_steps_summary = _summarize_branch(step_records)
    accepted_summary = _summarize_branch(accepted_steps)
    rejected_summary = _summarize_branch(rejected_steps)
    fallback_summary = _summarize_branch(fallback_steps)
    prefill_summary = _summarize_branch(prefill_steps)
    decode_tokens = sum(abs(r["num_tokens"]) for r in decode_steps)
    prefill_tokens = prefill_summary["tokens"]
    accepted_tokens = sum(abs(r["num_tokens"]) for r in accepted_steps)
    rejected_tokens = sum(abs(r["num_tokens"]) for r in rejected_steps)
    fallback_tokens = sum(abs(r["num_tokens"]) for r in fallback_steps)
    e2e_tps = completion_total / elapsed if elapsed > 0 else 0.0
    decode_tps = decode_tokens / max(1e-9, decode_seconds)
    prefill_tps = prefill_tokens / max(1e-9, prefill_steps_seconds)
    phase_ms_total = all_steps_summary["phase_ms"]
    host_device_postprocess_ms = {
        "host_ms": phase_ms_total["schedule_ms"] + phase_ms_total["postprocess_ms"] + phase_ms_total["release_ms"],
        "device_ms": phase_ms_total["run_ms"],
        "postprocess_ms": phase_ms_total["postprocess_ms"],
        "release_ms": phase_ms_total["release_ms"],
    }
    speculative_counts = runner.get_speculative_stats()

    return {
        "texts": text_rows,
        "token_ids": token_id_rows[0] if len(token_id_rows) == 1 else token_id_rows,
        "token_ids_by_request": token_id_rows,
        "completion_tokens_by_request": completion_counts,
        "tokens": completion_total,
        "total_requests": len(prompts),
        "seconds": elapsed,
        "tokens_per_second": e2e_tps,
        "end_to_end_tokens_per_second": e2e_tps,
        "decode_tokens_per_second": decode_tps,
        "prefill_tokens_per_second": prefill_tps,
        "decode_tokens": decode_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_seconds": decode_seconds,
        "accepted_tokens": accepted_tokens,
        "rejected_tokens": rejected_tokens,
        "fallback_tokens": fallback_tokens,
        "prefill_seconds": prefill_steps_seconds,
        "speculative": speculative_counts,
        "speculative_counts": {
            "drafts_proposed": speculative_counts.get("drafts_proposed", 0),
            "drafts_accepted": speculative_counts.get("drafts_accepted", 0),
            "drafts_rejected": speculative_counts.get("drafts_rejected", 0),
            "bonus_tokens": speculative_counts.get("bonus_tokens", 0),
            "fallback_steps": speculative_counts.get("fallback_steps", 0),
            "accepted_decode_steps": len(accepted_steps),
            "rejected_decode_steps": len(rejected_steps),
            "fallback_decode_steps": len(fallback_steps),
        },
        "return_step_records": return_step_records,
        "prefill_logits_by_request": prefill_logits_by_request,
        "step_record_count": len(step_records),
        "host_device_postprocess_ms": host_device_postprocess_ms,
        "phase_ms_total": phase_ms_total,
        "accepted_inter_token_latency_ms": {
            "p50": accepted_summary["inter_token_latency_ms_p50"],
            "p95": accepted_summary["inter_token_latency_ms_p95"],
        },
        "rejected_inter_token_latency_ms": {
            "p50": rejected_summary["inter_token_latency_ms_p50"],
            "p95": rejected_summary["inter_token_latency_ms_p95"],
        },
        "fallback_inter_token_latency_ms": {
            "p50": fallback_summary["inter_token_latency_ms_p50"],
            "p95": fallback_summary["inter_token_latency_ms_p95"],
        },
        "step_profile": {
            "steps": step_records if return_step_records else [],
            "all_steps": all_steps_summary,
            "accepted": accepted_summary,
            "rejected": rejected_summary,
            "fallback": fallback_summary,
            "prefill": prefill_summary,
            "phase_ms_avg_per_step": {
                "schedule_ms": all_steps_summary["phase_ms"]["schedule_ms"] / max(1, len(step_records)),
                "run_ms": all_steps_summary["phase_ms"]["run_ms"] / max(1, len(step_records)),
                "postprocess_ms": all_steps_summary["phase_ms"]["postprocess_ms"] / max(1, len(step_records)),
                "release_ms": all_steps_summary["phase_ms"]["release_ms"] / max(1, len(step_records)),
            },
            "phase_ms_total": all_steps_summary["phase_ms"],
            "host_device_postprocess_ms": host_device_postprocess_ms,
            "mode_counts": {
                "prefill_steps": len(prefill_steps),
                "decode_steps": len(decode_steps),
                "accepted_decode_steps": len(accepted_steps),
                "rejected_decode_steps": len(rejected_steps),
                "fallback_decode_steps": len(fallback_steps),
            },
        },
    }


def run_generation(
    engine: LLMEngine,
    prompt: str | list[int],
    *,
    mtp1: bool,
    max_tokens: int,
    mtp_position_offset: int = 0,
    mtp_token_source: str = "generated",
    mtp_hidden_source: str = "final_normed",
    compile_mtp_draft: bool = False,
    debug_spec: bool = False,
    step_profile: bool = False,
):
    result = run_generation_batch(
        engine,
        [prompt],
        mtp1=mtp1,
        max_tokens=max_tokens,
        mtp_position_offset=mtp_position_offset,
        mtp_token_source=mtp_token_source,
        mtp_hidden_source=mtp_hidden_source,
        compile_mtp_draft=compile_mtp_draft,
        debug_spec=debug_spec,
        step_profile=step_profile,
    )

    return {
        "text": result["texts"][0],
        "token_ids": result["token_ids"],
        "tokens": result["tokens"],
        "seconds": result["seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "decode_tokens_per_second": result["decode_tokens_per_second"],
        "decode_tokens": result["decode_tokens"],
        "prefill_seconds": result["prefill_seconds"],
        "speculative": result["speculative"],
        "step_profile": result["step_profile"],
        "completion_tokens_by_request": result["completion_tokens_by_request"],
    }


def _run_shape_warmups(
    engine: LLMEngine,
    prompt_lengths: list[int],
    *,
    prefill_lengths: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    max_tokens: int,
    mtp1: bool,
    mtp_position_offset: int = 0,
    mtp_token_source: str = "generated",
    mtp_hidden_source: str = "final_normed",
    compile_mtp_draft: bool = False,
    debug_spec: bool = False,
    step_profile: bool = False,
) -> dict[str, float]:
    if not prompt_lengths:
        return {}

    # Prefill final chunks emit the first completion token. A K=1 fused MTP
    # decode then needs room for the accepted draft plus the target bonus token,
    # so two generated tokens is not enough to exercise/compile that path.
    speculative_k = int(getattr(engine.config, "num_speculative_tokens", 1) or 1)
    warmup_min_tokens = max(2, speculative_k + 1)
    if mtp1:
        warmup_min_tokens = max(warmup_min_tokens, speculative_k + 3)
    warmup_tokens = min(max_tokens, warmup_min_tokens)
    seed = "benchmark prompt"
    total_prompts = len(prompt_lengths)
    warmup_summary = {
        "variant": ("mtp" if mtp1 else "nospec"),
        "runs": 0,
        "total_elapsed": 0.0,
    }

    max_batch = max(batch_sizes) if batch_sizes else total_prompts
    for prefill_len in prefill_lengths:
        for batch_size in sorted(set(batch_sizes or (max_batch,))):
            target_batch = min(int(batch_size), engine.config.max_num_seqs)
            if target_batch <= 0:
                continue
            prompts = make_token_prompts_from_lengths(engine, [prefill_len] * target_batch, seed_text=seed)
            run_t0 = time.perf_counter()
            run_generation_batch(
                engine=engine,
                prompts=prompts,
                mtp1=mtp1,
                max_tokens=warmup_tokens,
                mtp_position_offset=mtp_position_offset,
                mtp_token_source=mtp_token_source,
                mtp_hidden_source=mtp_hidden_source,
                compile_mtp_draft=compile_mtp_draft,
                debug_spec=debug_spec,
                step_profile=False,
                return_step_records=False,
            )
            warmup_summary["runs"] += 1
            warmup_summary["total_elapsed"] += time.perf_counter() - run_t0

    # Additional mixed-length warmup to force prefill chunked decode scheduling.
    mixed_prompts = make_token_prompts_from_lengths(
        engine,
        prompt_lengths[: engine.config.max_num_seqs],
        seed_text=seed,
    )
    if len(mixed_prompts) > 1:
        run_t0 = time.perf_counter()
        run_generation_batch(
            engine=engine,
            prompts=mixed_prompts,
            mtp1=mtp1,
            max_tokens=warmup_tokens,
            mtp_position_offset=mtp_position_offset,
            mtp_token_source=mtp_token_source,
            mtp_hidden_source=mtp_hidden_source,
            compile_mtp_draft=compile_mtp_draft,
            debug_spec=debug_spec,
            step_profile=False,
            return_step_records=False,
        )
        warmup_summary["runs"] += 1
        warmup_summary["total_elapsed"] += time.perf_counter() - run_t0

    return warmup_summary


def _run_benchmark_for_prompts(
    engine: LLMEngine,
    prompts: list[str | list[int]],
    *,
    args: argparse.Namespace,
    mtp1_variant: tuple[str, int, str] | None,
    repeats: int,
    mtp_position_offset: int = 0,
    mtp_token_source: str = "generated",
    mtp_hidden_source: str = "final_normed",
    step_profile: bool = False,
):
    runs = []
    for _ in range(repeats):
        run = run_generation_batch(
            engine=engine,
            prompts=prompts,
            mtp1=bool(mtp1_variant),
            max_tokens=args.max_tokens,
            mtp_position_offset=mtp_position_offset,
            mtp_token_source=mtp_token_source,
            mtp_hidden_source=mtp_hidden_source,
            compile_mtp_draft=args.compile_mtp_draft,
            debug_spec=args.debug_spec,
            step_profile=step_profile,
            return_step_records=bool(getattr(args, "trace_steps", False)),
        )
        runs.append(run)
    return runs[-1]


def _should_check_next_step_sanity(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "check_next_step_sanity", False)
        or getattr(args, "correctness_only", False)
        or os.environ.get("NANO_VLLM_JAX_MTP_CHECK_NEXT_STEP_SANITY", "0")
        in {"1", "true", "yes", "on", "True"}
    )


def _run_next_step_sanity_check(
    engine: LLMEngine,
    prompts: list[str | list[int]],
    *,
    args: argparse.Namespace,
    mtp_position_offset: int,
    mtp_token_source: str,
    mtp_hidden_source: str,
) -> dict:
    """Compare baseline and MTP one token past the reported generation boundary.

    This checks the actual public engine continuation path: if MTP emits exactly
    ``args.max_tokens`` visible tokens but leaves cache/hybrid state or Python
    length bookkeeping wrong, the extra token in this ``max_tokens + 1`` run
    should diverge from the baseline.
    """
    if int(args.max_tokens) < 1:
        return {
            "checked": False,
            "ok": True,
            "reason": "max_tokens_less_than_one",
        }

    sanity_max_tokens = int(args.max_tokens) + 1
    sanity_t0 = time.perf_counter()
    baseline_t0 = time.perf_counter()
    baseline_next = run_generation_batch(
        engine=engine,
        prompts=prompts,
        mtp1=False,
        max_tokens=sanity_max_tokens,
        mtp_position_offset=0,
        mtp_token_source=args.mtp_token_source,
        mtp_hidden_source=args.mtp_hidden_source,
        compile_mtp_draft=args.compile_mtp_draft,
        debug_spec=args.debug_spec,
        step_profile=False,
        return_step_records=False,
    )
    baseline_sanity_seconds = time.perf_counter() - baseline_t0
    mtp_t0 = time.perf_counter()
    mtp_next = run_generation_batch(
        engine=engine,
        prompts=prompts,
        mtp1=True,
        max_tokens=sanity_max_tokens,
        mtp_position_offset=mtp_position_offset,
        mtp_token_source=mtp_token_source,
        mtp_hidden_source=mtp_hidden_source,
        compile_mtp_draft=args.compile_mtp_draft,
        debug_spec=args.debug_spec,
        step_profile=False,
        return_step_records=False,
    )
    mtp_sanity_seconds = time.perf_counter() - mtp_t0
    sanity_seconds = time.perf_counter() - sanity_t0
    first_diff = _first_token_diff(
        baseline_next["token_ids_by_request"],
        mtp_next["token_ids_by_request"],
    )
    prompt_lengths = _prompt_lengths_for_inputs(engine, prompts)
    block_size = int(getattr(engine.config, "block_size", 1) or 1)
    boundary_rows = []
    for request_index, prompt_len in enumerate(prompt_lengths):
        baseline_tokens = baseline_next["token_ids_by_request"][request_index]
        mtp_tokens = mtp_next["token_ids_by_request"][request_index]
        visible_count = min(int(args.max_tokens), len(baseline_tokens), len(mtp_tokens))
        next_input_position = int(prompt_len + max(0, visible_count - 1))
        next_write_position = int(prompt_len + visible_count)
        committed_seq_len = int(prompt_len + visible_count)
        boundary_rows.append({
            "request_index": int(request_index),
            "prompt_tokens": int(prompt_len),
            "visible_token_count": int(visible_count),
            "baseline_visible_tail": [int(x) for x in baseline_tokens[:visible_count][-8:]],
            "mtp_visible_tail": [int(x) for x in mtp_tokens[:visible_count][-8:]],
            "next_input_token_baseline": int(baseline_tokens[visible_count - 1]) if visible_count > 0 else None,
            "next_input_token_mtp": int(mtp_tokens[visible_count - 1]) if visible_count > 0 else None,
            "next_token_baseline": int(baseline_tokens[visible_count]) if visible_count < len(baseline_tokens) else None,
            "next_token_mtp": int(mtp_tokens[visible_count]) if visible_count < len(mtp_tokens) else None,
            "committed_seq_len": committed_seq_len,
            "next_input_position": next_input_position,
            "next_input_block": int(next_input_position // block_size),
            "next_input_slot": int(next_input_position % block_size),
            "next_write_position": next_write_position,
            "next_write_block": int(next_write_position // block_size),
            "next_write_slot": int(next_write_position % block_size),
        })

    return {
        "checked": True,
        "ok": first_diff is None,
        "max_tokens_checked": sanity_max_tokens,
        "seconds": sanity_seconds,
        "baseline_seconds": baseline_sanity_seconds,
        "mtp_seconds": mtp_sanity_seconds,
        "first_diff": first_diff,
        "boundary": boundary_rows,
        "baseline_speculative_counts": baseline_next.get("speculative_counts", {}),
        "mtp_speculative_counts": mtp_next.get("speculative_counts", {}),
    }


def _prompt_lengths_for_inputs(engine: LLMEngine, prompts: list[str | list[int]]) -> list[int]:
    lengths: list[int] = []
    for prompt in prompts:
        if isinstance(prompt, str):
            lengths.append(len(engine._tokenize(prompt)))
        else:
            lengths.append(len(prompt))
    return lengths


def _build_variant_rows(
    engine: LLMEngine,
    prompts: list[str | list[int]],
    args: argparse.Namespace,
    variants: list[tuple[str, int, str]],
    repeats: int,
    warmup: bool,
    step_profile: bool,
):
    if warmup:
        _run_benchmark_for_prompts(
            engine,
            prompts,
            args=args,
            mtp1_variant=None,
            repeats=1,
            mtp_position_offset=0,
            mtp_token_source=args.mtp_token_source,
            mtp_hidden_source=args.mtp_hidden_source,
            step_profile=False,
        )
    baseline = _run_benchmark_for_prompts(
        engine,
        prompts,
        args=args,
        mtp1_variant=None,
        repeats=repeats,
        mtp_position_offset=0,
        mtp_token_source=args.mtp_token_source,
        mtp_hidden_source=args.mtp_hidden_source,
        step_profile=step_profile,
    )

    variant_rows = []
    for token_source, position_offset, hidden_source in variants:
        if warmup:
            _run_benchmark_for_prompts(
                engine,
                prompts,
                args=args,
                mtp1_variant=(token_source, position_offset, hidden_source),
                repeats=1,
                mtp_position_offset=position_offset,
                mtp_token_source=token_source,
                mtp_hidden_source=hidden_source,
                step_profile=False,
            )
        speculative = _run_benchmark_for_prompts(
            engine,
            prompts,
            args=args,
            mtp1_variant=(token_source, position_offset, hidden_source),
            repeats=repeats,
            mtp_position_offset=position_offset,
            mtp_token_source=token_source,
            mtp_hidden_source=hidden_source,
            step_profile=step_profile,
        )

        speculative_profile = speculative["step_profile"]
        accepted_profile = speculative_profile.get("accepted", {})
        rejected_profile = speculative_profile.get("rejected", {})
        fallback_profile = speculative_profile.get("fallback", {})
        prefill_profile = speculative_profile.get("prefill", {})
        baseline_profile = baseline["step_profile"]
        baseline_prefill_profile = baseline_profile.get("prefill", {})
        first_diff = _first_token_diff(
            baseline["token_ids_by_request"],
            speculative["token_ids_by_request"],
        )
        correct = first_diff is None
        next_step_check = {"checked": False, "ok": True}
        if correct and _should_check_next_step_sanity(args):
            next_step_check = _run_next_step_sanity_check(
                engine,
                prompts,
                args=args,
                mtp_position_offset=position_offset,
                mtp_token_source=token_source,
                mtp_hidden_source=hidden_source,
            )
        next_step_ok = bool(next_step_check.get("ok", True))
        output_samples = None
        if args.show_outputs:
            output_samples = {
                "baseline": baseline["texts"],
                "mtp": speculative["texts"],
            }

        row = {
            "token_source": token_source,
            "position_offset": position_offset,
            "hidden_source": hidden_source,
            "benchmark_mode": "all-or-none MTP",
            "mtp1": speculative,
            "baseline": baseline,
            "mtp1_runs": repeats,
            "mtp_tokens_per_second": speculative["tokens_per_second"],
            "mtp_end_to_end_tps": speculative["end_to_end_tokens_per_second"],
            "baseline_end_to_end_tps": baseline["end_to_end_tokens_per_second"],
            "mtp_prefill_tps": speculative["prefill_tokens_per_second"],
            "baseline_prefill_tps": baseline["prefill_tokens_per_second"],
            "mtp_decode_tps": speculative["decode_tokens_per_second"],
            "mtp_decode_latency_ms_per_token_p50": accepted_profile.get("ms_per_decode_token_p50", 0.0),
            "mtp_decode_latency_ms_per_token_p95": accepted_profile.get("ms_per_decode_token_p95", 0.0),
            "mtp_rejected_decode_tokens_per_second": rejected_profile.get("decode_tokens_per_second", 0.0),
            "mtp_rejected_latency_ms_per_token_p50": rejected_profile.get("ms_per_decode_token_p50", 0.0),
            "mtp_fallback_decode_tokens_per_second": fallback_profile.get("decode_tokens_per_second", 0.0),
            "baseline_decode_tps": baseline["decode_tokens_per_second"],
            "decode_speedup": speculative["decode_tokens_per_second"] / max(1e-9, baseline["decode_tokens_per_second"]),
            "end_to_end_speedup": baseline["seconds"] / max(1e-9, speculative["seconds"]),
            "acceptance_rate": speculative["speculative"].get("drafts_accepted", 0) / max(1, speculative["speculative"].get("drafts_proposed", 0)),
            "fallback_count": speculative["speculative_counts"].get("fallback_decode_steps", 0),
            "host_time_ms": speculative["host_device_postprocess_ms"]["host_ms"],
            "runner_device_time_ms": speculative["host_device_postprocess_ms"]["device_ms"],
            "postprocess_time_ms": speculative["host_device_postprocess_ms"]["postprocess_ms"],
            "baseline_prefill_ms_per_token_p50": baseline_prefill_profile.get("ms_per_token_p50", 0.0),
            "mtp_prefill_ms_per_token_p50": prefill_profile.get("ms_per_token_p50", 0.0),
            "baseline_seconds": baseline["seconds"],
            "mtp_seconds": speculative["seconds"],
            "baseline_prefill_seconds": baseline.get("prefill_seconds", 0.0),
            "mtp_prefill_seconds": speculative.get("prefill_seconds", 0.0),
            "speedup": baseline["seconds"] / max(1e-9, speculative["seconds"]),
            "correct": correct,
            "first_diff": first_diff,
            "throughput_valid": bool(correct and next_step_ok),
            "timed_results_valid": bool(correct and next_step_ok),
            "next_step_logit_sanity": bool(next_step_ok),
            "next_step_sanity_check": next_step_check,
            "throughput_suppressed_reason": None if correct and next_step_ok else (
                "mtp_next_step_diverged" if correct else "mtp_tokens_diverged"
            ),
            "accepted_tokens": speculative["accepted_tokens"],
            "rejected_tokens": speculative["rejected_tokens"],
            "fallback_tokens": speculative["fallback_tokens"],
            "decode_tokens": speculative["decode_tokens"],
            "prefill_tokens": speculative["step_profile"]["prefill"]["tokens"],
            "step_mode_counts": speculative["step_profile"]["mode_counts"],
            "speculative": speculative["speculative"],
            "decode": {
                "accepted": accepted_profile,
                "rejected": rejected_profile,
                "fallback": fallback_profile,
            },
            "inter_token_latency_ms": {
                "accepted": speculative["accepted_inter_token_latency_ms"],
                "rejected": speculative["rejected_inter_token_latency_ms"],
                "fallback": speculative["fallback_inter_token_latency_ms"],
            },
            "host_device_postprocess_ms": speculative["host_device_postprocess_ms"],
            "timed_results": {
                "valid": bool(correct and next_step_ok),
                "prefill_tok_s": speculative["prefill_tokens_per_second"],
                "decode_tok_s": speculative["decode_tokens_per_second"],
                "end_to_end_tok_s": speculative["end_to_end_tokens_per_second"],
                "decode_speedup": speculative["decode_tokens_per_second"] / max(1e-9, baseline["decode_tokens_per_second"]),
                "end_to_end_speedup": baseline["seconds"] / max(1e-9, speculative["seconds"]),
                "acceptance_rate": speculative["speculative"].get("drafts_accepted", 0) / max(1, speculative["speculative"].get("drafts_proposed", 0)),
                "fallback_count": speculative["speculative_counts"].get("fallback_decode_steps", 0),
                "host_time_ms": speculative["host_device_postprocess_ms"]["host_ms"],
                "runner_device_time_ms": speculative["host_device_postprocess_ms"]["device_ms"],
                "postprocess_time_ms": speculative["host_device_postprocess_ms"]["postprocess_ms"],
            },
            "adaptive_mtp_gating": _adaptive_gating_decision(
                baseline,
                speculative,
                margin=float(getattr(args, "adaptive_margin", 0.0)),
            ),
        }
        if output_samples is not None:
            row["outputs"] = output_samples
        variant_rows.append(row)

    return baseline, variant_rows


def main():
    args = parse_args()
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not args.hf_offline:
        os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["NANO_VLLM_JAX_EXEC_LOG_STEPS"] = "1" if args.exec_log_steps else "0"
    exact_commit_select_serving = (
        args.num_speculative_tokens > 0
        and (
            os.environ.get("NANO_VLLM_JAX_MTP_COMMIT_SELECT", "0") in {"1", "true", "yes", "on", "True"}
            or os.environ.get("NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1", "0") in {"1", "true", "yes", "on", "True"}
            or os.environ.get("NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1", "0") not in {"1", "true", "yes", "on", "True"}
        )
    )
    if exact_commit_select_serving and "NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY" not in os.environ:
        os.environ["NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY"] = "rowwise"
    if args.platform and args.platform != "auto":
        os.environ["JAX_PLATFORMS"] = args.platform

    import jax

    if args.require_tpu and jax.default_backend() != "tpu":
        raise SystemExit("require_tpu was set but JAX backend is not TPU")

    prompts = args.prompt or prompt_suite(args.prompt_suite)
    prefill_buckets = parse_buckets(args.prefill_buckets)
    batch_size_buckets = parse_buckets(args.batch_size_buckets)
    dtype = choose_dtype(args.dtype, jax.default_backend())
    batch_prompts = max(1, int(args.batch_prompts))
    requested_prompt_lengths = parse_prompt_lengths(args.prompt_lengths)
    use_batch = batch_prompts > 1 or bool(requested_prompt_lengths)
    if use_batch:
        batch_size_buckets = tuple(sorted(set(batch_size_buckets + (batch_prompts,))))
    if use_batch and not prefill_buckets:
        prefill_buckets = (64,)
    max_num_batched_tokens = max(prefill_buckets or (64,))
    if use_batch:
        max_num_batched_tokens = max_num_batched_tokens * batch_prompts

    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")
    print("loading_engine_with_mtp_weights=true")
    load_t0 = time.perf_counter()
    preset_config_kwargs = infer_config_preset(args.model, args.config_preset)
    engine = LLMEngine(
        args.model,
        backend=args.backend,
        **preset_config_kwargs,
        dtype=dtype,
        max_kv_cache_bytes=args.max_kv_cache_mb * 1024 * 1024,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_num_seqs=max(args.max_num_seqs, batch_prompts),
        max_num_batched_tokens=max_num_batched_tokens,
        prefill_buckets=prefill_buckets,
        batch_size_buckets=batch_size_buckets,
        max_blocks_per_seq=args.max_blocks_per_seq,
        jax_execution=args.jax_execution,
        num_speculative_tokens=args.num_speculative_tokens,
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

    if prefill_buckets:
        warmup_prefill_lengths = prefill_buckets
    else:
        warmup_prefill_lengths = (32, 64, args.prompt_length_max)

    if use_batch:
        batch_prompt_lengths = make_prompt_lengths(
            requested_prompt_lengths,
            count=batch_prompts,
            min_tokens=args.prompt_length_min,
            max_tokens=args.prompt_length_max,
        )
        seed_text = prompts if prompts else ["benchmark prompt"]
        benchmark_prompts: list[str | list[int]] = make_token_prompts_from_lengths(
            engine,
            lengths=batch_prompt_lengths,
            seed_text=seed_text,
        )
    else:
        batch_prompt_lengths = [len(engine._tokenize(prompt)) if isinstance(prompt, str) else len(prompt) for prompt in prompts]
        benchmark_prompts = prompts
        if not benchmark_prompts:
            benchmark_prompts = ["Tell me a joke about compilers."]

    hf_logits_check = run_hf_logits_check(
        engine,
        benchmark_prompts,
        args=args,
        dtype=dtype,
    )

    runner = engine.model_runner
    max_prefill_len = max(batch_prompt_lengths) if batch_prompt_lengths else args.prompt_length_min
    max_batch = len(benchmark_prompts)
    runner.warmup_compilation(max_prefill_len=max_prefill_len, max_batch=max_batch)

    warmup_summary = {
        "startup_seconds": 0.0,
        "shape_runs": 0,
        "shape_variant": "nospec",
        "warmed_shape_phase": {
            "enabled": bool(args.warmup),
            "nospec": None,
            "mtp": None,
            "total_runs": 0,
            "total_elapsed": 0.0,
        },
    }
    if args.warmup:
        warmup_t0 = time.perf_counter()
        nospec_warmup = _run_shape_warmups(
            engine=engine,
            prompt_lengths=batch_prompt_lengths,
            prefill_lengths=tuple(sorted(set([length for length in warmup_prefill_lengths]))),
            batch_sizes=tuple(sorted(set((1, batch_prompts, max_batch) + batch_size_buckets))),
            max_tokens=args.max_tokens,
            mtp1=False,
            mtp_position_offset=args.mtp_position_offset,
            mtp_token_source=args.mtp_token_source,
            mtp_hidden_source=args.mtp_hidden_source,
            compile_mtp_draft=args.compile_mtp_draft,
            debug_spec=args.debug_spec,
            step_profile=False,
        )
        warmup_summary["warmed_shape_phase"]["nospec"] = nospec_warmup

        if args.num_speculative_tokens > 0:
            first_mtp_variant = variants[0]
            mtp_warmup = _run_shape_warmups(
                engine=engine,
                prompt_lengths=batch_prompt_lengths,
                prefill_lengths=tuple(sorted(set([length for length in warmup_prefill_lengths]))),
                batch_sizes=tuple(sorted(set((1, batch_prompts, max_batch) + batch_size_buckets))),
                max_tokens=args.max_tokens,
                mtp1=True,
                mtp_position_offset=first_mtp_variant[1],
                mtp_token_source=first_mtp_variant[0],
                mtp_hidden_source=first_mtp_variant[2],
                compile_mtp_draft=args.compile_mtp_draft,
                debug_spec=args.debug_spec,
                step_profile=False,
            )
            warmup_summary["warmed_shape_phase"]["mtp"] = mtp_warmup
            warmup_summary["shape_variant"] = "nospec+mtp"
        warmup_summary["startup_seconds"] = time.perf_counter() - warmup_t0
        warmup_summary["shape_runs"] = sum(
            int(item.get("runs", 0))
            for item in (
                warmup_summary["warmed_shape_phase"]["nospec"],
                warmup_summary["warmed_shape_phase"]["mtp"],
            )
            if item
        )
        warmup_summary["warmed_shape_phase"]["total_runs"] = warmup_summary["shape_runs"]
        warmup_summary["warmed_shape_phase"]["total_elapsed"] = sum(
            float(item.get("total_elapsed", 0.0))
            for item in (
                warmup_summary["warmed_shape_phase"]["nospec"],
                warmup_summary["warmed_shape_phase"]["mtp"],
            )
            if item
        )

    if use_batch:
        benchmark_start_t0 = time.perf_counter()
        baseline, variant_rows = _build_variant_rows(
            engine=engine,
            prompts=benchmark_prompts,
            args=args,
            variants=variants,
            repeats=args.repeats,
            warmup=args.warmup,
            step_profile=args.step_profile,
        )
        benchmark_seconds = time.perf_counter() - benchmark_start_t0

        all_steps = baseline["step_profile"]["all_steps"]
        prefill_tokens = baseline["step_profile"]["prefill"]["tokens"]
        prefill_seconds = baseline["prefill_seconds"]
        decode_tokens = baseline["decode_tokens"]
        decode_seconds = all_steps["decode_seconds"]
        rows = []
        rows.append({
            "prompt_count": len(benchmark_prompts),
            "prompt_lengths": batch_prompt_lengths,
            "mode": "parallel_batch",
            "baseline": baseline,
            "variants": variant_rows,
            "total_elapsed_seconds": baseline["seconds"],
            "throughput_tokens_per_second": baseline["tokens_per_second"],
            "decode_tokens_per_second": decode_tokens / max(1e-9, decode_seconds),
            "prefill_tokens_per_second": prefill_tokens / max(1e-9, prefill_seconds),
            "prefill_decode_split": {
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "prefill_fraction": prefill_tokens / max(1, prefill_tokens + decode_tokens),
                "decode_fraction": decode_tokens / max(1, prefill_tokens + decode_tokens),
            },
        })
        summary = {
            "model": args.model,
            "dtype": args.dtype,
            "resolved_dtype": dtype,
            "jax_backend": jax.default_backend(),
            "jax_execution": args.jax_execution,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "compile_mtp_draft": args.compile_mtp_draft,
            "debug_spec": args.debug_spec,
            "step_profile": args.step_profile,
            "load_seconds": load_seconds,
            "warmup": warmup_summary,
            "warmed_shape_phase": warmup_summary["warmed_shape_phase"],
            "benchmark_modes": _benchmark_modes_report(),
            "mtp_batch_accept_policy": os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none"),
            "adaptive_mtp_gating_formula": "measured_decode_speedup = mtp_decode_tokens_per_second / baseline_decode_tokens_per_second; should_enable = measured_decode_speedup >= 1.0 + margin",
            "adaptive_margin": float(args.adaptive_margin),
            "stats_key": _stats_key(args, dtype=dtype, backend=jax.default_backend()),
            "benchmark_seconds": benchmark_seconds,
            "all_correct": all(
                variant["correct"] for variant in variant_rows
            ),
            "batch_prompts": batch_prompts,
            "rows": rows,
            "acceptance_rate_mean": (
                sum(
                    variant["speculative"]["drafts_accepted"]
                    / max(1, variant["speculative"]["drafts_proposed"])
                    for variant in variant_rows
                )
                / max(1, len(variant_rows))
            ),
            "decode_tps_no_spec_mean": sum(
                row["variants"][0]["baseline_decode_tps"] for row in rows
            ) / max(1, len(rows)),
            "decode_tps_mtp_mean": sum(
                row["variants"][0]["mtp_decode_tps"] for row in rows if row["variants"]
            ) / max(1, len(rows)),
            "decode_speedup_mean": sum(
                row["variants"][0]["mtp_decode_tps"] / max(1e-9, row["variants"][0]["baseline_decode_tps"])
                for row in rows
                if row["variants"] and row["variants"][0]["baseline_decode_tps"] > 0
            ) / max(1, len(rows)),
            "throughput_summary": {
                "parallel_batch_tps": baseline["tokens_per_second"],
                "decode_tps": decode_tokens / max(1e-9, baseline["seconds"] - prefill_seconds),
                "prefill_tps": prefill_tokens / max(1e-9, prefill_seconds),
                "prefill_decode_balance": rows[0]["prefill_decode_split"],
                "num_prefill_steps": baseline["step_profile"]["mode_counts"]["prefill_steps"],
                "num_decode_steps": baseline["step_profile"]["mode_counts"]["decode_steps"],
            },
        }
    else:
        rows = []
        for prompt in prompts:
            baseline, variant_rows = _build_variant_rows(
                engine=engine,
                prompts=[prompt],
                args=args,
                variants=variants,
                repeats=args.repeats,
                warmup=args.warmup,
                step_profile=args.step_profile,
            )
            rows.append({
                "prompt": prompt,
                "prompt_tokens": len(engine._tokenize(prompt)),
                "baseline": baseline,
                "variants": variant_rows,
            })

        mtp_variant_rows = [
            variant for row in rows for variant in row["variants"] if variant["speculative"]["drafts_proposed"] > 0
        ]
        acceptance_rate_mean = (
            sum(
                variant["speculative"]["drafts_accepted"]
                / max(1e-9, variant["speculative"]["drafts_proposed"])
                for variant in mtp_variant_rows
            )
            / max(1, len(mtp_variant_rows))
        )
        summary = {
            "model": args.model,
            "dtype": args.dtype,
            "resolved_dtype": dtype,
            "jax_backend": jax.default_backend(),
            "jax_execution": args.jax_execution,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "compile_mtp_draft": args.compile_mtp_draft,
            "debug_spec": args.debug_spec,
            "step_profile": args.step_profile,
            "load_seconds": load_seconds,
            "all_correct": all(variant["correct"] for row in rows for variant in row["variants"]),
            "acceptance_rate_mean": acceptance_rate_mean,
            "decode_tps_no_spec_mean": (
                sum(row["baseline"]["decode_tokens_per_second"] for row in rows) / max(1, len(rows))
            ),
            "decode_tps_mtp_mean": (
                sum(row["variants"][0]["mtp_decode_tps"] for row in rows if row["variants"]) / max(1, len(rows))
            ),
            "decode_speedup_mean": (
                sum(
                    row["variants"][0]["mtp_decode_tps"] / max(1e-9, row["baseline"]["decode_tokens_per_second"])
                    for row in rows
                    if row["variants"] and row["baseline"]["decode_tokens_per_second"] > 0
                )
                / max(1, len(rows))
            ),
            "rows": rows,
            "warmup": warmup_summary,
            "warmed_shape_phase": warmup_summary["warmed_shape_phase"],
            "benchmark_modes": _benchmark_modes_report(),
            "mtp_batch_accept_policy": os.environ.get("NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY", "all_or_none"),
            "adaptive_mtp_gating_formula": "measured_decode_speedup = mtp_decode_tokens_per_second / baseline_decode_tokens_per_second; should_enable = measured_decode_speedup >= 1.0 + margin",
            "adaptive_margin": float(args.adaptive_margin),
            "stats_key": _stats_key(args, dtype=dtype, backend=jax.default_backend()),
        }

    primary_variant = None
    for row in summary.get("rows", []):
        variants_for_row = row.get("variants", [])
        if variants_for_row:
            primary_variant = variants_for_row[0]
            break
    if primary_variant is not None:
        summary["required_metrics"] = {
            "prefill_tok_s": primary_variant["mtp_prefill_tps"],
            "decode_tok_s": primary_variant["mtp_decode_tps"],
            "decode_speedup": primary_variant["decode_speedup"],
            "end_to_end_speedup": primary_variant["end_to_end_speedup"],
            "acceptance_rate": primary_variant["acceptance_rate"],
            "fallback_count": primary_variant["fallback_count"],
            "host_time_ms": primary_variant["host_time_ms"],
            "runner_device_time_ms": primary_variant["runner_device_time_ms"],
            "postprocess_time_ms": primary_variant["postprocess_time_ms"],
            "stats_key": summary.get("stats_key"),
        }

    summary = apply_correctness_gate(summary, args=args, hf_logits_check=hf_logits_check)

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n")

    del engine
    gc.collect()
    raise SystemExit(0 if summary.get("all_correct", False) else 1)


if __name__ == "__main__":
    main()
