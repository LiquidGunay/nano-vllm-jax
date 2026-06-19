"""LLM Engine for Qwen 3.5 JAX with continuous batching."""

import atexit
import os
from time import perf_counter
from typing import List, Dict, Optional, Union
from tqdm.auto import tqdm
from dataclasses import dataclass, replace

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.kv_cache import KVCacheSpec, cap_num_kv_cache_blocks
from nanovllm_jax.engine.sequence import DeviceTokenSlot, Sequence, SamplingParams
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.model import ModelParams
from nanovllm_jax.load_weights import load_weights_from_hf_streaming

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "True"}


def _config_or_env_flag(config: Qwen3_5Config | None, attr: str, env_name: str, *, default: bool = False) -> bool:
    if config is not None and hasattr(config, attr):
        return bool(getattr(config, attr))
    return os.environ.get(env_name, "1" if default else "0") in _TRUE_ENV_VALUES


def _overlapped_streaming_token_prefetch_enabled() -> bool:
    return os.environ.get("NANO_VLLM_JAX_OVERLAPPED_STREAMING_TOKEN_PREFETCH", "0") in _TRUE_ENV_VALUES


def _offline_streaming_token_events_enabled() -> bool:
    return os.environ.get("NANO_VLLM_JAX_OFFLINE_STREAMING_TOKEN_EVENTS", "0") in _TRUE_ENV_VALUES


def _trace_token_prefetch_enabled(config: Qwen3_5Config | None = None) -> bool:
    if "NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH" in os.environ:
        return os.environ.get("NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH", "0") in _TRUE_ENV_VALUES
    return _config_or_env_flag(
        config,
        "trace_token_prefetch",
        "NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH",
        default=True,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _deferred_trace_timing_summary(
    *,
    ttfts_ms: list[float],
    itls_ms: list[float],
    last_token_elapsed_seconds: float | None,
    generated_tokens: int,
    source: str,
    extra: dict | None = None,
) -> dict:
    summary = {
        "source": source,
        "generated_tokens": int(generated_tokens),
        "last_token_elapsed_seconds": last_token_elapsed_seconds,
        "ttft_ms_mean": float(sum(ttfts_ms) / len(ttfts_ms)) if ttfts_ms else None,
        "ttft_ms_p50": _percentile(ttfts_ms, 50),
        "ttft_ms_p95": _percentile(ttfts_ms, 95),
        "itl_ms_mean": float(sum(itls_ms) / len(itls_ms)) if itls_ms else None,
        "itl_ms_p50": _percentile(itls_ms, 50),
        "itl_ms_p95": _percentile(itls_ms, 95),
    }
    if extra:
        summary.update(extra)
    return summary


def _engine_step_profile_enabled() -> bool:
    return os.environ.get("NANO_VLLM_JAX_PROFILE_ENGINE_STEP", "0") in _TRUE_ENV_VALUES


@dataclass(frozen=True)
class _DeferredTokenStreamRecord:
    slots: tuple
    target_completion_lengths: dict[int, int]
    step_seconds: float
    step_start_seconds: float
    step_end_seconds: float
    scheduler_step_tokens: int
    scheduler_step_is_decode: bool


class LLMEngine:
    """LLM Engine for Qwen 3.5 JAX.
    
    Features:
    - Continuous batching (prefill + decode in same batch)
    - Paged KV cache with prefix caching
    - Simple Python-based scheduling (like nano-vllm)
    
    Usage:
        engine = LLMEngine("qwen-3.5-0.8b")
        outputs = engine.generate(
            prompts=["Hello, how are you?"],
            sampling_params=SamplingParams(max_tokens=100),
        )
    """

    def __init__(
        self, 
        model_path: str,
        backend: str = "auto",
        **kwargs,
    ):
        """Initialize LLM engine.
        
        Args:
            model_path: Path to model weights or model identifier
            **kwargs: Additional config parameters
        """
        # Create config
        weight_dtype = kwargs.pop("weight_dtype", None)
        config_fields = {f.name for f in Qwen3_5Config.__dataclass_fields__.values()}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Qwen3_5Config(**config_kwargs)
        self.weight_dtype = weight_dtype or self.config.dtype
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
        
        # Initialize tokenizer
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set EOS token from tokenizer
        if self.config.eos is None:
            self.config.eos = self.tokenizer.eos_token_id
        
        # Initialize model parameters - load from HF (no silent fallback)
        print(f"Loading pretrained weights from {model_path}...")
        load_config = replace(self.config, dtype=self.weight_dtype)
        load_mtp = (
            self.config.speculative_method == "mtp"
            and self.config.num_speculative_tokens > 0
        )
        self.params = load_weights_from_hf_streaming(
            model_path,
            load_config,
            load_mtp=load_mtp,
        )
        print("✓ Using pretrained weights")
        
        # Initialize components
        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config, self.params, backend=backend)
        self.scheduler.set_mtp_backend(self.model_runner.backend)
        
        # Register cleanup
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

        runner = self.model_runner
        executor = getattr(runner, "executor", None)
        cache = getattr(executor, "_jit_cache", None)
        cache_entries_before = len(cache) if cache is not None else None
        started = perf_counter()
        runner_summary = runner.warmup_compilation(
            max_prefill_len=int(max_prefill_len),
            max_batch=int(max_batch),
            include_sampled_routes=bool(include_sampled_routes),
            prefill_token_buckets=prefill_token_buckets,
            batch_size_buckets=batch_size_buckets,
            decode_block_table_buckets=decode_block_table_buckets,
        )
        elapsed = perf_counter() - started
        cache_entries_after = len(cache) if cache is not None else None
        return {
            "enabled": True,
            "mode": "generic_bucket_startup",
            "seconds": elapsed,
            "max_prefill_len": int(max_prefill_len),
            "max_batch": int(max_batch),
            "prefill_buckets": list(getattr(self.config, "prefill_buckets", ()) or ()),
            "prefill_token_buckets": list(getattr(self.config, "prefill_token_buckets", ()) or ()),
            "prefill_layout": str(getattr(self.config, "prefill_layout", "packed")),
            "batch_size_buckets": list(getattr(self.config, "batch_size_buckets", ()) or ()),
            "decode_block_table_buckets": list(getattr(self.config, "decode_block_table_buckets", ()) or ()),
            "startup_warmup_prefill_token_buckets": list(prefill_token_buckets or ()),
            "startup_warmup_batch_size_buckets": list(batch_size_buckets or ()),
            "startup_warmup_decode_block_table_buckets": list(decode_block_table_buckets or ()),
            "include_sampled_routes": bool(include_sampled_routes),
            "jit_cache_entries_before": cache_entries_before,
            "jit_cache_entries_after": cache_entries_after,
            "jit_cache_entries_added": (
                cache_entries_after - cache_entries_before
                if cache_entries_before is not None and cache_entries_after is not None
                else None
            ),
            "runner": runner_summary,
        }

    def exit(self):
        """Cleanup on exit."""
        del self.model_runner

    def add_request(
        self, 
        prompt: Union[str, List[int]], 
        sampling_params: SamplingParams,
    ) -> Sequence:
        """Add a generation request.
        
        Args:
            prompt: Prompt text or token IDs
            sampling_params: Sampling parameters
        """
        # Tokenize if string
        if isinstance(prompt, str):
            # TODO: Use proper tokenizer
            # For now, simple placeholder
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
                    "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY requires greedy sampling "
                    "with ignore_eos=True"
                )

        return [self.add_request(prompt, sp) for prompt, sp in zip(request_inputs, sampling_params)]

    def step(self, *, materialize_finished_outputs: bool = True) -> tuple[List[tuple], int]:
        """Execute one scheduling step.
        
        Returns:
            Tuple of (outputs, num_tokens)
            - outputs: List of (seq_id, completion_tokens) for finished sequences
            - num_tokens: Number of tokens processed (positive for prefill, negative for decode)
        """
        profile_step = _engine_step_profile_enabled()
        step_t = perf_counter()
        # Schedule sequences
        seqs, scheduled_batch = self.scheduler.schedule()
        schedule_t = perf_counter()
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

        # Run model
        token_ids = self.model_runner.run(seqs, batch=scheduled_batch)
        runner_t = perf_counter()
        emitted_tokens = 0
        mixed_prefill_decode = bool(getattr(scheduled_batch, "mixed_prefill_decode", False))
        if (not scheduled_batch.is_prefill) or mixed_prefill_decode:
            for token_id in token_ids:
                emitted_tokens += len(token_id) if isinstance(token_id, list) else 1

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
        postprocess_t = perf_counter()
        finished_seq_ids = [seq.seq_id for seq, is_finished in zip(seqs, finished_flags) if is_finished]
        if finished_seq_ids:
            self.model_runner.release(finished_seq_ids)
        release_t = perf_counter()
        step_elapsed = perf_counter() - step_t
        self.scheduler.update_mtp_admission(
            self.model_runner.get_speculative_stats(),
            is_decode=(not scheduled_batch.is_prefill) or mixed_prefill_decode,
            elapsed_seconds=step_elapsed,
            emitted_tokens=emitted_tokens,
            batch=scheduled_batch,
        )
        admission_t = perf_counter()
        
        # Collect outputs
        if materialize_finished_outputs:
            outputs = [
                (seq.seq_id, seq.completion_token_ids)
                for seq in seqs
                if seq.is_finished
            ]
        else:
            outputs = [(seq.seq_id, []) for seq in seqs if seq.is_finished]
        outputs_t = perf_counter()
        
        # Track throughput
        if scheduled_batch.is_prefill and not mixed_prefill_decode:
            num_tokens = scheduled_batch.num_prefill_tokens
        else:
            num_tokens = -getattr(self.scheduler, "last_num_generated_tokens", scheduled_batch.num_decode_tokens)

        if profile_step:
            is_decode = (not scheduled_batch.is_prefill) or mixed_prefill_decode
            print(
                "[ENGINE_STEP] "
                f"decode={int(is_decode)} emitted={int(emitted_tokens)} "
                f"num_tokens={int(abs(num_tokens))} seqs={len(seqs)} "
                f"schedule_ms={(schedule_t - step_t) * 1000:.3f} "
                f"runner_ms={(runner_t - schedule_t) * 1000:.3f} "
                f"post_ms={(postprocess_t - runner_t) * 1000:.3f} "
                f"release_ms={(release_t - postprocess_t) * 1000:.3f} "
                f"admit_ms={(admission_t - release_t) * 1000:.3f} "
                f"outputs_ms={(outputs_t - admission_t) * 1000:.3f} "
                f"total_ms={(outputs_t - step_t) * 1000:.3f}",
                flush=True,
            )

        return outputs, num_tokens

    def _step_without_finished_output_materialization(self) -> tuple[List[tuple], int]:
        """Run one step without pulling final deferred token IDs to host."""
        try:
            return self.step(materialize_finished_outputs=False)
        except TypeError as exc:
            if "materialize_finished_outputs" not in str(exc):
                raise
            return self.step()

    def is_finished(self) -> bool:
        """Check if all requests are complete."""
        return self.scheduler.is_finished()

    def get_mtp_admission_report(self) -> dict[str, object]:
        """Return JSON-friendly per-bucket MTP admission stats."""
        return self.scheduler.get_mtp_admission_report()

    def generate(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, any]]:
        """Generate completions for prompts.
        
        Args:
            prompts: List of prompts (text or token IDs)
            sampling_params: Sampling parameters (single or list)
            use_tqdm: Show progress bar
            
        Returns:
            List of dicts with 'text' and 'token_ids' keys
        """
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
        
        # Setup progress bar
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # Add all requests
        for prompt, sp in zip(request_inputs, sampling_params):
            self.add_request(prompt, sp)
        
        # Generation loop
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # Update progress
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })
            
            # Collect finished outputs
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Sort by seq_id and decode
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
        """Yield token events as requests make progress through the scheduler."""
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

        seqs = [self.add_request(prompt, sp) for prompt, sp in zip(request_inputs, sampling_params)]
        if _offline_streaming_token_events_enabled():
            yield from self._iter_generate_offline_token_events(
                seqs,
                include_text=include_text,
            )
            return
        device_token_carry = _config_or_env_flag(
            getattr(self, "config", None),
            "device_token_carry",
            "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        )
        if device_token_carry and _overlapped_streaming_token_prefetch_enabled():
            yield from self._iter_generate_deferred_tokens(
                seqs,
                include_text=include_text,
            )
            return

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

    def _iter_generate_offline_token_events(
        self,
        seqs: List[Sequence],
        *,
        include_text: bool = True,
    ):
        """Yield deferred token events while deferring host token IDs until finish.

        This mode intentionally avoids materializing generated tokens every step,
        trading per-step streaming IDs for exact final completion output.
        """
        seq_to_request = {seq.seq_id: index for index, seq in enumerate(seqs)}
        seen_completion_lengths = {seq.seq_id: 0 for seq in seqs}
        emitted_finish = set()
        stream_start = perf_counter()

        while not self.is_finished():
            step_start = perf_counter()
            _, num_tokens = self._step_without_finished_output_materialization()
            step_end = perf_counter()

            for seq in seqs:
                request_index = seq_to_request[seq.seq_id]
                previous_length = seen_completion_lengths[seq.seq_id]
                current_length = seq.num_completion_tokens
                if current_length > previous_length:
                    for completion_index in range(previous_length, current_length):
                        token_event = {
                            "event": "token",
                            "seq_id": seq.seq_id,
                            "request_index": request_index,
                            "completion_index": completion_index,
                            "token_id": None,
                            "elapsed_seconds": step_end - stream_start,
                            "step_seconds": step_end - step_start,
                            "step_start_seconds": step_start - stream_start,
                            "step_end_seconds": step_end - stream_start,
                            "scheduler_step_tokens": int(abs(num_tokens)),
                            "scheduler_step_is_decode": bool(num_tokens < 0),
                        }
                        if include_text:
                            token_event["text"] = ""
                        yield token_event
                    seen_completion_lengths[seq.seq_id] = current_length
                if seq.is_finished and seq.seq_id not in emitted_finish:
                    emitted_finish.add(seq.seq_id)
                    yield {
                        "event": "finished",
                        "seq_id": seq.seq_id,
                        "request_index": request_index,
                        "elapsed_seconds": step_end - stream_start,
                        "completion_tokens": current_length,
                    }

        Sequence.materialize_device_tokens_for_sequences(seqs)
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

    def _iter_generate_deferred_tokens(
        self,
        seqs: List[Sequence],
        *,
        include_text: bool = True,
    ):
        """Yield streaming events after overlapping device-token host copies.

        The newest token snapshot remains deferred while the next scheduler
        step runs. Once that next step completes, the older prefetched snapshot
        is materialized and emitted. This keeps the behavior default-off and
        preserves generation order while avoiding the normal per-step
        ``completion_token_ids`` sync.
        """
        seq_to_request = {seq.seq_id: index for index, seq in enumerate(seqs)}
        seen_completion_lengths = {seq.seq_id: 0 for seq in seqs}
        emitted_finish = set()
        stream_start = perf_counter()
        pending_records: list[_DeferredTokenStreamRecord] = []
        snapshotted_completion_lengths = {seq.seq_id: 0 for seq in seqs}

        while not self.is_finished():
            step_start = perf_counter()
            _, num_tokens = self._step_without_finished_output_materialization()
            step_end = perf_counter()

            slots = Sequence.snapshot_new_device_token_slots_for_sequences(
                seqs,
                snapshotted_completion_lengths,
            )
            if slots:
                slots = Sequence.prefetch_device_token_slots(slots)
            target_completion_lengths = {
                int(seq.seq_id): int(seq.num_completion_tokens)
                for seq in seqs
            }
            snapshotted_completion_lengths.update(target_completion_lengths)
            pending_records.append(
                _DeferredTokenStreamRecord(
                    slots=slots,
                    target_completion_lengths=target_completion_lengths,
                    step_seconds=step_end - step_start,
                    step_start_seconds=step_start - stream_start,
                    step_end_seconds=step_end - stream_start,
                    scheduler_step_tokens=int(abs(num_tokens)),
                    scheduler_step_is_decode=bool(num_tokens < 0),
                )
            )

            while len(pending_records) > 1:
                record = pending_records.pop(0)
                yield from self._emit_deferred_stream_record(
                    seqs,
                    record,
                    seq_to_request=seq_to_request,
                    seen_completion_lengths=seen_completion_lengths,
                    emitted_finish=emitted_finish,
                    include_text=include_text,
                )

        while pending_records:
            record = pending_records.pop(0)
            yield from self._emit_deferred_stream_record(
                seqs,
                record,
                seq_to_request=seq_to_request,
                seen_completion_lengths=seen_completion_lengths,
                emitted_finish=emitted_finish,
                include_text=include_text,
            )
        Sequence.materialize_device_tokens_for_sequences(seqs)
        final_record = _DeferredTokenStreamRecord(
            slots=(),
            target_completion_lengths={
                int(seq.seq_id): int(seq.num_completion_tokens)
                for seq in seqs
            },
            step_seconds=0.0,
            step_start_seconds=perf_counter() - stream_start,
            step_end_seconds=perf_counter() - stream_start,
            scheduler_step_tokens=0,
            scheduler_step_is_decode=False,
        )
        yield from self._emit_deferred_stream_record(
            seqs,
            final_record,
            seq_to_request=seq_to_request,
            seen_completion_lengths=seen_completion_lengths,
            emitted_finish=emitted_finish,
            include_text=include_text,
        )
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

    def _emit_deferred_stream_record(
        self,
        seqs: List[Sequence],
        record: _DeferredTokenStreamRecord,
        *,
        seq_to_request: dict[int, int],
        seen_completion_lengths: dict[int, int],
        emitted_finish: set[int],
        include_text: bool,
    ):
        Sequence.materialize_device_token_slots(record.slots)
        for seq in seqs:
            seq_id = int(seq.seq_id)
            request_index = seq_to_request[seq_id]
            materialized = seq.materialized_completion_token_ids()
            target_length = min(
                int(record.target_completion_lengths.get(seq_id, len(materialized))),
                len(materialized),
            )
            previous_length = int(seen_completion_lengths[seq_id])
            if target_length > previous_length:
                for offset, token_id in enumerate(materialized[previous_length:target_length]):
                    completion_index = previous_length + offset
                    token_event = {
                        "event": "token",
                        "seq_id": seq.seq_id,
                        "request_index": request_index,
                        "completion_index": completion_index,
                        "token_id": int(token_id),
                        "elapsed_seconds": record.step_end_seconds,
                        "step_seconds": record.step_seconds,
                        "step_start_seconds": record.step_start_seconds,
                        "step_end_seconds": record.step_end_seconds,
                        "scheduler_step_tokens": record.scheduler_step_tokens,
                        "scheduler_step_is_decode": record.scheduler_step_is_decode,
                    }
                    if include_text:
                        token_event["text"] = self._detokenize([int(token_id)])
                    yield token_event
                seen_completion_lengths[seq_id] = target_length
            if (
                seq.is_finished
                and seq_id not in emitted_finish
                and not seq.has_unmaterialized_device_tokens
                and seen_completion_lengths[seq_id] >= seq.num_completion_tokens
            ):
                emitted_finish.add(seq_id)
                yield {
                    "event": "finished",
                    "seq_id": seq.seq_id,
                    "request_index": request_index,
                    "elapsed_seconds": record.step_end_seconds,
                    "completion_tokens": seq.num_completion_tokens,
                }

    def generate_with_trace(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        *,
        include_text: bool = True,
        trace_events: bool = True,
    ) -> dict:
        """Generate requests and return server-side per-token timing events."""
        device_token_carry = _config_or_env_flag(
            getattr(self, "config", None),
            "device_token_carry",
            "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY",
        )
        if device_token_carry and self._trace_can_defer_device_tokens(prompts, sampling_params):
            return self._generate_with_trace_deferred_tokens(
                prompts,
                sampling_params=sampling_params,
                include_text=include_text,
                trace_events=trace_events,
            )
        events = []
        results = []
        for event in self.iter_generate(
            prompts,
            sampling_params=sampling_params,
            include_text=include_text,
        ):
            events.append(event)
            if event.get("event") == "done":
                results = event.get("results", [])
        return {
            "results": results,
            "events": events,
        }

    def _trace_can_defer_device_tokens(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams], None],
    ) -> bool:
        """The deferred trace path avoids token readback and cannot stop on EOS."""
        if sampling_params is None:
            return False
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                return False
            params = sampling_params
        else:
            params = [sampling_params] * len(prompts)
        return all(bool(sp.ignore_eos) for sp in params)

    def _generate_with_trace_deferred_tokens(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        *,
        include_text: bool = True,
        trace_events: bool = True,
    ) -> dict:
        """Trace greedy device-token-carry runs without per-step token readback."""
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

        sampled_token_fastpath = _config_or_env_flag(
            getattr(self, "config", None),
            "sampled_token_fastpath",
            "NANO_VLLM_JAX_SAMPLED_TOKEN_FASTPATH",
            default=True,
        )
        for sp in sampling_params:
            if sp.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            if sp.temperature < 0:
                raise ValueError("temperature must be non-negative")
            if not sp.ignore_eos:
                raise ValueError(
                    "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY requires ignore_eos=True"
                )
            if sp.temperature != 0:
                if not sampled_token_fastpath:
                    raise ValueError(
                        "NANO_VLLM_JAX_DEVICE_TOKEN_CARRY with temperature sampling "
                        "requires sampled_token_fastpath=True"
                    )
                if sp.top_p < 1.0 or sp.top_k > 0:
                    raise ValueError(
                        "NANO_VLLM_JAX_SAMPLED_TOKEN_FASTPATH currently supports "
                        "full-vocab temperature sampling only; keep top_p=1.0 and top_k<=0"
                    )

        seqs = [self.add_request(prompt, sp) for prompt, sp in zip(request_inputs, sampling_params)]
        seq_to_request = {seq.seq_id: index for index, seq in enumerate(seqs)}
        seen_completion_lengths = {seq.seq_id: 0 for seq in seqs}
        last_token_elapsed_by_seq: dict[int, float] = {}
        ttfts_ms: list[float] = []
        itls_ms: list[float] = []
        last_token_elapsed_seconds: float | None = None
        emitted_finish = set()
        stream_start = perf_counter()
        events = []
        config = getattr(self, "config", None)
        trace_token_prefetch_enabled = _trace_token_prefetch_enabled(config)
        prefetch_trace_tokens = trace_events and trace_token_prefetch_enabled
        prefetch_finished_tokens = (not trace_events) and trace_token_prefetch_enabled
        snapshotted_completion_lengths = {seq.seq_id: 0 for seq in seqs}
        pending_prefetch_slots: tuple[DeviceTokenSlot, ...] = ()
        pending_finished_slots: tuple[DeviceTokenSlot, ...] = ()
        prefetched_finished_seq_ids: set[int] = set()

        while not self.is_finished():
            step_start = perf_counter()
            _, num_tokens = self._step_without_finished_output_materialization()
            step_end = perf_counter()
            if prefetch_finished_tokens and pending_finished_slots:
                Sequence.materialize_device_token_slots(pending_finished_slots)
                pending_finished_slots = ()
            if prefetch_trace_tokens:
                if pending_prefetch_slots:
                    Sequence.prefetch_device_token_slots(pending_prefetch_slots)
                slots = Sequence.snapshot_new_device_token_slots_for_sequences(
                    seqs,
                    snapshotted_completion_lengths,
                )
                pending_prefetch_slots = slots
                snapshotted_completion_lengths.update(
                    {
                        int(seq.seq_id): int(seq.num_completion_tokens)
                        for seq in seqs
                    }
                )
            for seq in seqs:
                request_index = seq_to_request[seq.seq_id]
                previous_length = seen_completion_lengths[seq.seq_id]
                current_length = seq.num_completion_tokens
                if current_length > previous_length:
                    token_elapsed = step_end - stream_start
                    for completion_index in range(previous_length, current_length):
                        previous_token_elapsed = last_token_elapsed_by_seq.get(seq.seq_id)
                        if previous_token_elapsed is None:
                            ttfts_ms.append(1000.0 * token_elapsed)
                        else:
                            itls_ms.append(1000.0 * (token_elapsed - previous_token_elapsed))
                        last_token_elapsed_by_seq[seq.seq_id] = token_elapsed
                        last_token_elapsed_seconds = (
                            token_elapsed
                            if last_token_elapsed_seconds is None
                            else max(last_token_elapsed_seconds, token_elapsed)
                        )
                        if trace_events:
                            token_event = {
                                "event": "token",
                                "seq_id": seq.seq_id,
                                "request_index": request_index,
                                "completion_index": completion_index,
                                "token_id": None,
                                "elapsed_seconds": token_elapsed,
                                "step_seconds": step_end - step_start,
                                "step_start_seconds": step_start - stream_start,
                                "step_end_seconds": step_end - stream_start,
                                "scheduler_step_tokens": int(abs(num_tokens)),
                                "scheduler_step_is_decode": bool(num_tokens < 0),
                            }
                            events.append(token_event)
                    seen_completion_lengths[seq.seq_id] = current_length
                if trace_events and seq.is_finished and seq.seq_id not in emitted_finish:
                    emitted_finish.add(seq.seq_id)
                    events.append(
                        {
                            "event": "finished",
                            "seq_id": seq.seq_id,
                            "request_index": request_index,
                            "elapsed_seconds": step_end - stream_start,
                            "completion_tokens": seq.num_completion_tokens,
                        }
                    )
            if prefetch_finished_tokens:
                newly_finished = [
                    seq
                    for seq in seqs
                    if seq.is_finished and int(seq.seq_id) not in prefetched_finished_seq_ids
                ]
                if newly_finished:
                    slots = Sequence.snapshot_device_token_slots_for_sequences(newly_finished)
                    prefetched_finished_seq_ids.update(int(seq.seq_id) for seq in newly_finished)
                    if slots:
                        pending_finished_slots = slots
                        Sequence.prefetch_device_token_slots(slots)

        if prefetch_trace_tokens and pending_prefetch_slots:
            Sequence.prefetch_device_token_slots(pending_prefetch_slots)
        final_post_loop_start = perf_counter()
        final_pending_materialize_seconds = 0.0
        final_device_token_materialize_seconds = 0.0
        final_result_build_seconds = 0.0
        final_slots_before = sum(len(seq._device_token_slots) for seq in seqs)
        if prefetch_finished_tokens and pending_finished_slots:
            final_pending_start = perf_counter()
            Sequence.materialize_device_token_slots(pending_finished_slots)
            final_pending_materialize_seconds = perf_counter() - final_pending_start
        final_slots_after_pending = sum(len(seq._device_token_slots) for seq in seqs)
        final_device_materialize_start = perf_counter()
        Sequence.materialize_device_tokens_for_sequences(seqs)
        final_device_token_materialize_seconds = perf_counter() - final_device_materialize_start
        final_slots_after_device_materialize = sum(len(seq._device_token_slots) for seq in seqs)
        final_result_build_start = perf_counter()
        results = []
        for index, seq in enumerate(seqs):
            token_ids = [int(token) for token in seq.completion_token_ids]
            results.append(
                {
                    "request_index": index,
                    "text": self._detokenize(token_ids) if include_text else "",
                    "token_ids": token_ids,
                }
            )
        final_result_build_seconds = perf_counter() - final_result_build_start
        final_post_loop_seconds = perf_counter() - final_post_loop_start
        tokens_by_request = {
            int(result["request_index"]): list(result["token_ids"])
            for result in results
        }
        if trace_events:
            for event in events:
                if event.get("event") != "token":
                    continue
                request_tokens = tokens_by_request.get(int(event["request_index"]), [])
                completion_index = int(event["completion_index"])
                if completion_index < len(request_tokens):
                    token_id = int(request_tokens[completion_index])
                    event["token_id"] = token_id
                    if include_text:
                        event["text"] = self._detokenize([token_id])
            events.append(
                {
                    "event": "done",
                    "elapsed_seconds": perf_counter() - stream_start,
                    "results": results,
                }
            )
        return {
            "results": results,
            "events": events,
            "timing_summary": _deferred_trace_timing_summary(
                ttfts_ms=ttfts_ms,
                itls_ms=itls_ms,
                last_token_elapsed_seconds=last_token_elapsed_seconds,
                generated_tokens=sum(len(result["token_ids"]) for result in results),
                source=(
                    "jax_server_step_trace"
                    if trace_events
                    else "jax_server_step_summary"
                ),
                extra={
                    "final_post_loop_seconds": final_post_loop_seconds,
                    "final_pending_materialize_seconds": final_pending_materialize_seconds,
                    "final_device_token_materialize_seconds": final_device_token_materialize_seconds,
                    "final_result_build_seconds": final_result_build_seconds,
                    "final_device_token_slots_before": final_slots_before,
                    "final_device_token_slots_after_pending": final_slots_after_pending,
                    "final_device_token_slots_after_device_materialize": final_slots_after_device_materialize,
                },
            ),
        }

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using Qwen tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs using Qwen tokenizer."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
