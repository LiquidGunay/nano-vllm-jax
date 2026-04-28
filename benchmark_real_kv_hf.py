#!/usr/bin/env python3
"""Real-weight KV-cache parity and throughput benchmark.

This script intentionally uses the canonical ModelExecutor path instead of the
older experimental JIT modules.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from nanovllm_jax.backends import select_backend
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.model_executor import ModelExecutor
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.kv_cache import KVCacheSpec, init_hybrid_state
from nanovllm_jax.load_weights import load_weights_from_hf_streaming


@dataclass
class RunResult:
    ids: list[int]
    prefill_logits: np.ndarray
    prefill_seconds: float
    decode_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Tell me a joke about compilers and coffee.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--hf-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-kv-cache-mb", type=int, default=96)
    parser.add_argument("--warmup", action="store_true", help="Run one untimed JAX and HF pass first.")
    parser.add_argument("--jax-execution", choices=["eager", "decode-jit", "jit"], default="eager")
    parser.add_argument("--prefill-bucket", type=int, help="Pad JAX prefill to this static token length.")
    parser.add_argument("--target", choices=["both", "hf", "jax"], default="both")
    parser.add_argument("--output-npz", type=Path, help="Write target result arrays and timing metadata.")
    parser.add_argument("--compare-npz", type=Path, help="Compare JAX result against a saved HF result.")
    return parser.parse_args()


def choose_dtype(requested: str, target: str) -> str:
    if requested != "auto":
        return requested
    if target == "jax":
        if jax.default_backend() != "gpu":
            return "float32"
        try:
            import torch

            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                return "bfloat16" if major >= 8 else "float16"
        except ImportError:
            pass
        return "float16"
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return "bfloat16" if major >= 8 else "float16"
    return "float32"


def torch_dtype(dtype: str):
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def make_batch(
    token_ids: list[int],
    *,
    position_start: int,
    seq_len: int,
    max_blocks: int,
    is_prefill: bool,
    query_len_bucket: int | None = None,
) -> ScheduledBatch:
    query_len = len(token_ids)
    padded_len = query_len_bucket or query_len
    if query_len > padded_len:
        raise ValueError(f"query length {query_len} exceeds bucket {padded_len}")
    positions = list(range(position_start, position_start + len(token_ids)))
    token_ids = token_ids + [0] * (padded_len - query_len)
    positions = positions + [0] * (padded_len - query_len)
    return ScheduledBatch(
        tokens=jnp.array([token_ids], dtype=jnp.int32),
        positions=jnp.array([positions], dtype=jnp.int32),
        seq_ids=jnp.array([0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, query_len], dtype=jnp.int32),
        is_prefill=is_prefill,
        num_prefill_tokens=query_len if is_prefill else 0,
        num_decode_tokens=0 if is_prefill else query_len,
        block_tables=jnp.arange(max_blocks, dtype=jnp.int32)[None, :],
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
    )


class JaxBenchModel:
    def __init__(
        self,
        model_name: str,
        *,
        prompt_tokens: int,
        max_new_tokens: int,
        dtype: str,
        max_kv_cache_mb: int,
        execution: str,
        prefill_bucket: int | None,
    ):
        self.config = Qwen3_5Config.qwen3_5_0_8b()
        self.config.dtype = dtype
        self.prefill_bucket = prefill_bucket
        if self.prefill_bucket is not None and prompt_tokens > self.prefill_bucket:
            raise ValueError(f"prompt has {prompt_tokens} tokens but --prefill-bucket is {self.prefill_bucket}")
        max_total_len = prompt_tokens + max_new_tokens + 1
        self.max_blocks = max(4, math.ceil(max_total_len / self.config.block_size) + 1)
        self.config.num_kvcache_blocks = self.max_blocks
        self.config.max_kv_cache_bytes = max_kv_cache_mb * 1024 * 1024

        params = load_weights_from_hf_streaming(model_name, self.config)
        self.backend = select_backend("auto")
        self.executor = ModelExecutor(self.config, params, self.backend)
        self.execution = execution
        self.spec = KVCacheSpec(
            num_layers=self.config.num_hidden_layers,
            num_blocks=self.max_blocks,
            block_size=self.config.block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.get_dtype(),
            max_kv_cache_bytes=self.config.max_kv_cache_bytes,
        )

    @property
    def compiled_entries(self) -> int:
        return len(self.executor._jit_cache)

    def compile_startup(self, input_ids: list[int], *, compile_decode: bool):
        if self.execution not in {"decode-jit", "jit"}:
            return
        compile_tokens = 2 if compile_decode else 1
        _ = self.generate(input_ids, max_new_tokens=compile_tokens)

    def generate(self, input_ids: list[int], *, max_new_tokens: int) -> RunResult:
        cache = self.backend.allocate_kv_cache(self.spec, max_seqs=1, max_blocks_per_seq=self.max_blocks)
        hybrid = init_hybrid_state(self.config, batch_size=1, dtype=self.config.get_dtype())

        generated = list(input_ids)
        prefill_batch = make_batch(
            generated,
            position_start=0,
            seq_len=len(generated),
            max_blocks=self.max_blocks,
            is_prefill=True,
            query_len_bucket=self.prefill_bucket,
        )
        t0 = time.perf_counter()
        prefill_step = self.executor.forward_step_jit if self.execution == "jit" else self.executor.forward_step
        decode_step = self.executor.forward_step_jit if self.execution in {"decode-jit", "jit"} else self.executor.forward_step
        out = prefill_step(
            prefill_batch,
            cache_storage=cache,
            hybrid_state=hybrid,
            last_logits_only=True,
        )
        out.activations.block_until_ready()
        prefill_seconds = time.perf_counter() - t0
        prefill_logits = np.array(out.activations[0, -1], dtype=np.float32)
        next_token = int(jnp.argmax(out.activations[0, -1]))
        generated.append(next_token)

        decode_seconds = 0.0
        cache = out.cache_storage
        hybrid = out.hybrid_state
        for _ in range(max_new_tokens - 1):
            pos = len(generated) - 1
            decode_batch = make_batch(
                [next_token],
                position_start=pos,
                seq_len=len(generated),
                max_blocks=self.max_blocks,
                is_prefill=False,
            )
            t0 = time.perf_counter()
            out = decode_step(
                decode_batch,
                cache_storage=cache,
                hybrid_state=hybrid,
                last_logits_only=True,
            )
            out.activations.block_until_ready()
            decode_seconds += time.perf_counter() - t0
            cache = out.cache_storage
            hybrid = out.hybrid_state
            next_token = int(jnp.argmax(out.activations[0, -1]))
            generated.append(next_token)

        return RunResult(generated, prefill_logits, prefill_seconds, decode_seconds)


def load_hf_model(model_name: str, dtype: str, device_arg: str):
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg != "auto":
        device = device_arg
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype(dtype),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, torch.device(device)


def run_hf(
    model,
    device,
    input_ids: list[int],
    *,
    max_new_tokens: int,
) -> RunResult:
    import torch

    generated = list(input_ids)
    input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(input_tensor, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - t0
        past_kv = outputs.past_key_values
        prefill_logits = outputs.logits[0, -1].detach().float().cpu().numpy()
        next_token = int(torch.argmax(outputs.logits[0, -1]))
        generated.append(next_token)

        decode_seconds = 0.0
        for _ in range(max_new_tokens - 1):
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(input_tensor, past_key_values=past_kv, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            decode_seconds += time.perf_counter() - t0
            past_kv = outputs.past_key_values
            next_token = int(torch.argmax(outputs.logits[0, -1]))
            generated.append(next_token)

    return RunResult(generated, prefill_logits, prefill_seconds, decode_seconds)


def free_torch_model(model):
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def save_result(path: Path, result: RunResult, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array(result.ids, dtype=np.int32),
        prefill_logits=result.prefill_logits.astype(np.float32),
        prefill_seconds=np.array(result.prefill_seconds, dtype=np.float64),
        decode_seconds=np.array(result.decode_seconds, dtype=np.float64),
        metadata=np.array(json.dumps(metadata)),
    )


def load_result(path: Path) -> tuple[RunResult, dict]:
    with np.load(path, allow_pickle=False) as data:
        result = RunResult(
            ids=data["ids"].astype(np.int32).tolist(),
            prefill_logits=data["prefill_logits"].astype(np.float32),
            prefill_seconds=float(data["prefill_seconds"]),
            decode_seconds=float(data["decode_seconds"]),
        )
        metadata = json.loads(str(data["metadata"]))
    return result, metadata


def print_metrics(name: str, result: RunResult, prompt_tokens: int, max_new_tokens: int):
    decode_tokens = max(0, max_new_tokens - 1)
    prefill_tps = prompt_tokens / result.prefill_seconds if result.prefill_seconds else float("inf")
    decode_tps = decode_tokens / result.decode_seconds if result.decode_seconds else float("inf")
    e2e_tps = max_new_tokens / result.total_seconds if result.total_seconds else float("inf")
    print(
        f"{name:<18} prefill={prefill_tps:8.2f} tok/s "
        f"decode={decode_tps:8.2f} tok/s e2e={e2e_tps:8.2f} tok/s "
        f"time={result.total_seconds:7.3f}s"
    )


def main():
    args = parse_args()
    dtype = choose_dtype(args.dtype, args.target)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    if not input_ids:
        raise ValueError("Prompt tokenized to an empty sequence")

    print(f"model={args.model}")
    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")
    if args.target in {"both", "hf"}:
        import torch

        print(f"torch_cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"dtype={dtype} prompt_tokens={len(input_ids)} max_new_tokens={args.max_new_tokens}")
    print(f"jax_execution={args.jax_execution}")
    if args.prefill_bucket is not None:
        print(f"prefill_bucket={args.prefill_bucket}")
    print(f"target={args.target}")
    print(f"prompt={args.prompt!r}")

    if args.warmup:
        print("warmup=enabled")

    hf_result = None
    if args.target in {"both", "hf"}:
        hf_model, hf_device = load_hf_model(args.model, dtype, args.hf_device)
        if args.warmup:
            _ = run_hf(hf_model, hf_device, input_ids, max_new_tokens=max(1, min(args.max_new_tokens, 2)))
        hf_result = run_hf(hf_model, hf_device, input_ids, max_new_tokens=args.max_new_tokens)
        free_torch_model(hf_model)

        print("\nthroughput")
        print_metrics("HF cache", hf_result, len(input_ids), args.max_new_tokens)
        if args.output_npz is not None:
            save_result(
                args.output_npz,
                hf_result,
                {
                    "target": "hf",
                    "model": args.model,
                    "prompt": args.prompt,
                    "dtype": dtype,
                    "prompt_tokens": len(input_ids),
                    "max_new_tokens": args.max_new_tokens,
                },
            )
            print(f"saved_result={args.output_npz}")

    if args.target == "hf":
        return

    if args.compare_npz is not None:
        hf_result, hf_metadata = load_result(args.compare_npz)
        if hf_metadata.get("prompt") != args.prompt or int(hf_metadata.get("max_new_tokens", -1)) != args.max_new_tokens:
            raise ValueError("--compare-npz prompt/max_new_tokens metadata does not match this run")

    jax_model = JaxBenchModel(
        args.model,
        prompt_tokens=len(input_ids),
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        max_kv_cache_mb=args.max_kv_cache_mb,
        execution=args.jax_execution,
        prefill_bucket=args.prefill_bucket,
    )
    if args.jax_execution in {"decode-jit", "jit"}:
        print("jax_compile_startup=begin")
        jax_model.compile_startup(input_ids, compile_decode=args.max_new_tokens > 1)
        print(f"jax_compile_startup=done cache_entries={jax_model.compiled_entries}")
    elif args.warmup:
        _ = jax_model.generate(input_ids, max_new_tokens=max(1, min(args.max_new_tokens, 2)))

    compiled_before = jax_model.compiled_entries
    jax_result = jax_model.generate(input_ids, max_new_tokens=args.max_new_tokens)
    compiled_after = jax_model.compiled_entries

    if args.output_npz is not None:
        save_result(
            args.output_npz,
            jax_result,
            {
                "target": "jax",
                "model": args.model,
                "prompt": args.prompt,
                "dtype": dtype,
                "prompt_tokens": len(input_ids),
                "max_new_tokens": args.max_new_tokens,
                "prefill_bucket": args.prefill_bucket,
                "jax_execution": args.jax_execution,
                "jit_cache_entries_before": compiled_before,
                "jit_cache_entries_after": compiled_after,
            },
        )
        print(f"saved_result={args.output_npz}")

    print("\nthroughput")
    if hf_result is not None:
        print_metrics("HF cache", hf_result, len(input_ids), args.max_new_tokens)
    print_metrics("JAX cache", jax_result, len(input_ids), args.max_new_tokens)
    if args.jax_execution in {"decode-jit", "jit"}:
        print(f"jit_cache_entries_before={compiled_before} after={compiled_after}")
        print(f"jit_compiled_during_measure={compiled_after != compiled_before}")

    if hf_result is not None:
        diff = jax_result.prefill_logits - hf_result.prefill_logits
        mse = float(np.mean(diff * diff))
        max_abs = float(np.max(np.abs(diff)))
        jax_top5 = np.argsort(jax_result.prefill_logits)[-5:][::-1]
        hf_top5 = np.argsort(hf_result.prefill_logits)[-5:][::-1]
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(jax_result.ids, hf_result.ids)) if a != b),
            None,
        )

        print("\ncorrectness")
        print(f"prefill_logits_mse={mse:.6e} max_abs={max_abs:.6e}")
        print(f"top1_match={int(jax_top5[0]) == int(hf_top5[0])} top5_match={np.array_equal(jax_top5, hf_top5)}")
        print(f"hf_top5={hf_top5.tolist()}")
        print(f"jax_top5={jax_top5.tolist()}")
        print(f"generated_match={jax_result.ids == hf_result.ids}")
        if first_diff is not None:
            print(f"first_generated_diff_index={first_diff} hf={hf_result.ids[first_diff]} jax={jax_result.ids[first_diff]}")
        print(f"hf_ids={hf_result.ids}")
        print(f"jax_ids={jax_result.ids}")


if __name__ == "__main__":
    main()
