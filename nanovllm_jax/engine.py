"""Request lifecycle engine for Qwen 3.5 JAX serving."""

import atexit
from time import perf_counter
from typing import Any, List, Dict, Optional, Union
from tqdm.auto import tqdm
from dataclasses import replace

from nanovllm_jax.config import EngineConfig, Qwen3_5Config
from nanovllm_jax.cache import KVCacheSpec, cap_num_kv_cache_blocks
from nanovllm_jax.model import ModelParams
from nanovllm_jax.weights import load_weights_from_hf_streaming
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import Sequence, SamplingParams

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


_PUBLIC_ENGINE_KWARGS = {
    "max_prefill",
    "max_num_seqs",
    "max_num_resident_seqs",
    "max_num_batched_tokens",
    "max_blocks_per_seq",
    "kv_cache_bytes",
    "kv_cache_mb",
    "max_kv_cache_mb",
    "num_kvcache_blocks",
    "num_kv_cache_blocks",
    "prefill_token_buckets",
    "batch_size_buckets",
    "decode_block_buckets",
    "decode_block_table_buckets",
    "prefix_cache",
}


def _engine_config_from_public_kwargs(model_path: str, kwargs: dict[str, Any]) -> EngineConfig:
    unknown = sorted(set(kwargs) - _PUBLIC_ENGINE_KWARGS)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(
            "LLM accepts workload/capacity kwargs only; "
            f"unsupported policy or unknown kwargs: {names}"
        )
    return EngineConfig.from_mapping({"model": model_path, **kwargs})


def _qwen_config_from_engine_config(engine_config: EngineConfig) -> tuple[Qwen3_5Config, str]:
    engine_kwargs = engine_config.to_engine_kwargs()
    weight_dtype = str(engine_kwargs.pop("weight_dtype", engine_kwargs.get("dtype", "bfloat16")))
    qwen_fields = set(Qwen3_5Config.__dataclass_fields__)
    qwen_kwargs = {key: value for key, value in engine_kwargs.items() if key in qwen_fields}
    return Qwen3_5Config(**qwen_kwargs), weight_dtype


class LLMEngine:
    """Request lifecycle engine for the promoted Qwen 3.5 serving path."""

    def __init__(
        self,
        model_path: str,
        *,
        engine_config: EngineConfig | None = None,
        **kwargs,
    ):
        if engine_config is not None and kwargs:
            raise TypeError("Pass either engine_config or workload/capacity kwargs, not both")
        if engine_config is None:
            engine_config = _engine_config_from_public_kwargs(model_path, kwargs)
        elif engine_config.model != model_path:
            engine_config = replace(engine_config, model=model_path)

        self.config, self.weight_dtype = _qwen_config_from_engine_config(engine_config)
        kv_spec = KVCacheSpec(
            num_layers=self.config.num_hidden_layers,
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.get_dtype(),
            max_kv_cache_bytes=self.config.max_kv_cache_bytes,
        )
        self.config.num_kvcache_blocks = cap_num_kv_cache_blocks(kv_spec)
        if self.config.max_blocks_per_seq is None:
            resident_capacity = int(
                getattr(self.config, "max_num_resident_seqs", None)
                or self.config.max_num_seqs
            )
            self.config.max_blocks_per_seq = max(1, self.config.num_kvcache_blocks // resident_capacity)

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required; install it with the package dependencies")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.config.eos is None:
            self.config.eos = self.tokenizer.eos_token_id

        print(f"Loading pretrained weights from {model_path}...")
        load_config = replace(self.config, dtype=self.weight_dtype)
        self.params = load_weights_from_hf_streaming(
            model_path,
            load_config,
        )
        print("✓ Using pretrained weights")

        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config, self.params)
        atexit.register(self.exit)

    def warmup_compilation(
        self,
        *,
        max_prefill_len: int | None = None,
        max_batch: int | None = None,
        include_sampled_routes: bool = True,
        prefill_token_buckets: tuple[int, ...] | None = None,
        batch_size_buckets: tuple[int, ...] | None = None,
        decode_block_table_buckets: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        """Compile configured serving buckets without using live request data."""
        if max_prefill_len is None:
            max_prefill_len = max(
                tuple(getattr(self.config, "prefill_token_buckets", ()) or ())
                or
                tuple(getattr(self.config, "prefill_buckets", ()) or ())
                or (int(getattr(self.config, "max_num_batched_tokens", 64) or 64),)
            )
        if max_batch is None:
            max_batch = max(
                tuple(getattr(self.config, "batch_size_buckets", ()) or ())
                or (int(getattr(self.config, "max_num_seqs", 1) or 1),)
            )

        started = perf_counter()
        runner_summary = self.model_runner.warmup_compilation(
            max_prefill_len=int(max_prefill_len),
            max_batch=int(max_batch),
            include_sampled_routes=bool(include_sampled_routes),
            prefill_token_buckets=prefill_token_buckets,
            batch_size_buckets=batch_size_buckets,
            decode_block_table_buckets=decode_block_table_buckets,
        )
        elapsed = perf_counter() - started
        return {"enabled": True, "seconds": elapsed, "runner": runner_summary}

    def exit(self):
        del self.model_runner

    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
    ) -> Sequence:
        if isinstance(prompt, str):
            prompt = self._tokenize(prompt)

        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if sampling_params.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if sampling_params.temperature < 0:
            raise ValueError("temperature must be non-negative")

        # Create sequence
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def _prepare_generation_sequences(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        *,
        require_greedy_ignore_eos: bool = False,
    ) -> List[Sequence]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        elif len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts length")

        request_inputs: List[List[int]] = []
        for prompt in prompts:
            token_ids = self._tokenize(prompt) if isinstance(prompt, str) else list(prompt)
            if not token_ids:
                raise ValueError("prompt must contain at least one token")
            request_inputs.append(token_ids)

        for sp in sampling_params:
            if sp.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            if sp.temperature < 0:
                raise ValueError("temperature must be non-negative")
            if require_greedy_ignore_eos and (sp.temperature != 0 or not sp.ignore_eos):
                raise ValueError(
                    "device_token_carry requires greedy sampling "
                    "with ignore_eos=True"
                )

        return [self.add_request(prompt, sp) for prompt, sp in zip(request_inputs, sampling_params)]

    def step(self, *, materialize_finished_outputs: bool = True) -> tuple[List[tuple], int]:
        seqs, scheduled_batch = self.scheduler.schedule()
        prefill_chunk_lengths: list[int] | None = None
        if scheduled_batch.is_prefill:
            if scheduled_batch.query_lens_host is not None:
                prefill_chunk_lengths = [int(x) for x in scheduled_batch.query_lens_host[:len(seqs)]]
            else:
                prefill_chunk_lengths = [int(x) for x in scheduled_batch.query_lens.tolist()[:len(seqs)]]

        self.model_runner.install_cached_prefix_hybrid_states(
            seqs,
            getattr(self.scheduler, "prefix_cache_hybrid_states", None),
        )

        token_ids = self.model_runner.run(seqs, batch=scheduled_batch)
        mixed_prefill_decode = bool(getattr(scheduled_batch, "mixed_prefill_decode", False))

        if scheduled_batch.is_prefill:
            prefix_states_by_seq = None
            if (
                getattr(self.scheduler, "enable_prefix_cache_execution", False)
                and getattr(self.scheduler, "prefix_cache_requires_hybrid_state", False)
            ):
                prefix_states_by_seq = self.model_runner.hybrid_states_for_sequences(seqs)
            self.scheduler.record_computed_prefix_states(
                seqs,
                prefill_chunk_lengths or [],
                prefix_states_by_seq,
            )
        finished_flags = self.scheduler.postprocess(seqs, token_ids, prefill_chunk_lengths=prefill_chunk_lengths)
        finished_seq_ids = [seq.seq_id for seq, is_finished in zip(seqs, finished_flags) if is_finished]
        if finished_seq_ids:
            self.model_runner.release(finished_seq_ids)

        if materialize_finished_outputs:
            outputs = [
                (seq.seq_id, seq.completion_token_ids)
                for seq in seqs
                if seq.is_finished
            ]
        else:
            outputs = [(seq.seq_id, []) for seq in seqs if seq.is_finished]

        if scheduled_batch.is_prefill and not mixed_prefill_decode:
            num_tokens = scheduled_batch.num_prefill_tokens
        else:
            num_tokens = -getattr(self.scheduler, "last_num_generated_tokens", scheduled_batch.num_decode_tokens)

        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, any]]:
        seqs = self._prepare_generation_sequences(prompts, sampling_params)
        if use_tqdm:
            pbar = tqdm(total=len(seqs), desc="Generating", dynamic_ncols=True)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = [
            {"text": self._detokenize(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return results

    def iter_generate(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        *,
        include_text: bool = True,
    ):
        seqs = self._prepare_generation_sequences(prompts, sampling_params)
        seq_to_request = {seq.seq_id: index for index, seq in enumerate(seqs)}
        seen_completion_lengths = {seq.seq_id: 0 for seq in seqs}
        emitted_finish = set()
        stream_start = perf_counter()

        while not self.is_finished():
            step_start = perf_counter()
            _, num_tokens = self.step()
            step_end = perf_counter()
            for seq in seqs:
                request_index = seq_to_request[seq.seq_id]
                completion = seq.completion_token_ids
                previous_length = seen_completion_lengths[seq.seq_id]
                if len(completion) > previous_length:
                    for offset, token_id in enumerate(completion[previous_length:]):
                        completion_index = previous_length + offset
                        token_event = {
                            "event": "token",
                            "seq_id": seq.seq_id,
                            "request_index": request_index,
                            "completion_index": completion_index,
                            "token_id": int(token_id),
                            "elapsed_seconds": step_end - stream_start,
                            "step_seconds": step_end - step_start,
                            "step_start_seconds": step_start - stream_start,
                            "step_end_seconds": step_end - stream_start,
                            "scheduler_step_tokens": int(abs(num_tokens)),
                            "scheduler_step_is_decode": bool(num_tokens < 0),
                        }
                        if include_text:
                            token_event["text"] = self._detokenize([int(token_id)])
                        yield token_event
                    seen_completion_lengths[seq.seq_id] = len(completion)
                if seq.is_finished and seq.seq_id not in emitted_finish:
                    emitted_finish.add(seq.seq_id)
                    yield {
                        "event": "finished",
                        "seq_id": seq.seq_id,
                        "request_index": request_index,
                        "elapsed_seconds": step_end - stream_start,
                        "completion_tokens": len(seq.completion_token_ids),
                    }

        yield {
            "event": "done",
            "elapsed_seconds": perf_counter() - stream_start,
            "results": [
                {
                    "request_index": index,
                    "text": self._detokenize(seq.completion_token_ids) if include_text else "",
                    "token_ids": [int(token) for token in seq.completion_token_ids],
                }
                for index, seq in enumerate(seqs)
            ],
        }

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using Qwen tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs using Qwen tokenizer."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


LLM = LLMEngine

__all__ = ["LLM"]
