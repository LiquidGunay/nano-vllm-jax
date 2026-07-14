"""Nano-VLLM-JAX adapter for the benchmark claim."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any

from huggingface_hub import snapshot_download

from nanovllm_jax.config import EngineConfig, WarmupConfig
from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.fastpath import validate_runtime_dependencies
from nanovllm_jax.output import OutputBuffer
from nanovllm_jax.sequence import SamplingParams


def _versions(*packages: str) -> dict[str, str]:
    found = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


class Backend:
    def __init__(self, manifest: dict[str, Any], prompts: list[list[int]]):
        validate_runtime_dependencies()
        self.manifest = manifest
        self.prompts = prompts
        model = manifest["model"]
        checkpoint = snapshot_download(model["id"], revision=model["revision"])
        workload = manifest["workload"]
        capacity = manifest["capacity"]
        batch_size = workload["batch_size"]
        prefill_tokens = batch_size * workload["prompt_tokens"]
        blocks = capacity["max_blocks_per_seq"]
        warmup = WarmupConfig(
            prefill_token_buckets=(prefill_tokens,),
            batch_size_buckets=(batch_size,),
            decode_block_buckets=(blocks,),
            include_sampled_routes=False,
        )
        config = EngineConfig(
            model=checkpoint,
            max_num_seqs=batch_size,
            max_num_resident_seqs=batch_size,
            max_num_batched_tokens=prefill_tokens,
            max_blocks_per_seq=blocks,
            kv_cache_bytes=capacity["kv_cache_bytes"],
            num_kvcache_blocks=capacity["num_kvcache_blocks"],
            prefill_token_buckets=(prefill_tokens,),
            batch_size_buckets=(batch_size,),
            decode_block_buckets=(blocks,),
            warmup=warmup,
            prefix_cache=False,
        )
        self.engine = LLMEngine(checkpoint, engine_config=config)
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=workload["output_tokens"],
            ignore_eos=True,
        )

    def warmup(self) -> dict[str, Any]:
        workload = self.manifest["workload"]
        capacity = self.manifest["capacity"]
        summary = self.engine.warmup_compilation(
            prefill_token_buckets=(workload["prompt_tokens"],),
            batch_size_buckets=(workload["batch_size"],),
            decode_block_table_buckets=(capacity["max_blocks_per_seq"],),
            include_sampled_routes=False,
        )
        return {
            "routes": summary["runner"]["warmed_routes"],
            "compilation_seconds": summary["seconds"],
        }

    def run_once(self) -> dict[str, Any]:
        seqs = [
            self.engine.add_request(prompt, self.sampling)
            for prompt in self.prompts
        ]
        started = perf_counter()
        first_token_time = None
        decode_tokens = 0
        while not self.engine.is_finished():
            result = self.engine.step()
            if result.phase == "prefill":
                OutputBuffer.materialize_many(seq.output for seq in seqs)
                first_token_time = perf_counter()
            else:
                decode_tokens += result.num_emitted_tokens
        outputs = OutputBuffer.materialize_many(seq.output for seq in seqs)
        finished = perf_counter()
        if first_token_time is None:
            raise RuntimeError("request completed without a prefill step")
        decode_seconds = finished - first_token_time
        return {
            "ttft_seconds": first_token_time - started,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / decode_seconds,
            "output_token_ids": outputs,
        }

    def compile_fingerprint(self) -> tuple[str, ...]:
        cache = self.engine.model_runner.executor._jit_cache
        return tuple(sorted(repr(key) for key in cache))

    def software(self) -> dict[str, str]:
        return _versions(
            "nano-vllm-jax",
            "jax",
            "jaxlib",
            "transformers",
            "triton",
            "flashinfer-python",
            "jax-tvm-ffi",
        )

    def memory(self) -> dict[str, Any]:
        budget = self.engine.startup_device_budget_bytes
        return {
            "declared_startup_device_bytes": sum(budget.values()),
            "declared_startup_device_breakdown": budget,
        }
