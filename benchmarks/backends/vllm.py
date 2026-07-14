"""vLLM adapter for the benchmark claim."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

from vllm import LLM, SamplingParams


class Backend:
    def __init__(self, manifest: dict[str, Any], prompts: list[list[int]]):
        expected = manifest["frameworks"]["vllm"]
        actual = version("vllm")
        if actual != expected:
            raise RuntimeError(f"claim requires vllm=={expected}, found {actual}")
        self.manifest = manifest
        self.prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
        model = manifest["model"]
        workload = manifest["workload"]
        capacity = manifest["capacity"]
        self.llm = LLM(
            model=model["id"],
            revision=model["revision"],
            dtype=model["dtype"],
            seed=0,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            language_model_only=True,
            mm_processor_cache_gb=0,
            max_model_len=capacity["max_model_len"],
            max_num_seqs=workload["batch_size"],
            max_num_batched_tokens=workload["prompt_tokens"],
            enable_prefix_caching=False,
            kv_cache_memory_bytes=capacity["kv_cache_bytes"],
            generation_config="vllm",
            disable_log_stats=False,
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=workload["output_tokens"],
            ignore_eos=True,
            detokenize=False,
        )

    def warmup(self) -> dict[str, Any]:
        return {
            "cuda_graphs": True,
            "language_model_only": True,
            "speculative_decoding": False,
            "flashinfer_sampler": False,
        }

    def run_once(self) -> dict[str, Any]:
        outputs = self.llm.generate(self.prompts, self.sampling, use_tqdm=False)
        rows = [list(output.outputs[0].token_ids) for output in outputs]
        if len(outputs) != 1 or outputs[0].metrics is None:
            raise RuntimeError("vLLM did not return B=1 request timing metrics")
        metrics = outputs[0].metrics
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        decode_tokens = len(rows[0]) - 1
        if metrics.first_token_latency <= 0 or decode_seconds <= 0:
            raise RuntimeError("vLLM returned incomplete request timing metrics")
        return {
            "ttft_seconds": metrics.first_token_latency,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / decode_seconds,
            "output_token_ids": rows,
        }

    def compile_fingerprint(self) -> None:
        return None

    def software(self) -> dict[str, str]:
        packages = ("vllm", "torch", "triton", "transformers")
        return {package: version(package) for package in packages}

    def memory(self) -> dict[str, Any]:
        return {
            "configured_kv_cache_bytes": self.manifest["capacity"][
                "kv_cache_bytes"
            ]
        }
