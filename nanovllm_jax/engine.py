"""Request lifecycle engine for Qwen 3.5 JAX serving."""

import atexit
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Union
from dataclasses import replace

import jax

from nanovllm_jax.config import EngineConfig, ModelConfig, RuntimeSpec
from nanovllm_jax.cache import KVCacheSpec
from nanovllm_jax.ops import resolve_kv_cache_spec
from nanovllm_jax.batch import SchedulePlan
from nanovllm_jax.weights import (
    load_mtp_weights_from_hf_streaming,
    load_weights_from_hf_streaming,
    resolve_checkpoint,
    resolve_checkpoint_metadata,
)
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.output import OutputBuffer, is_device_token
from nanovllm_jax.sequence import Sequence, SequenceStatus, SamplingParams
from nanovllm_jax.speculation import DrafterConfig
from nanovllm_jax.step import (
    FinishedRequest,
    FinishReason,
    RunResult,
    StepResult,
    TokenEvent,
)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


_PUBLIC_ENGINE_KWARGS = {
    "max_num_seqs",
    "max_num_resident_seqs",
    "max_num_batched_tokens",
    "max_blocks_per_seq",
    "kv_cache_bytes",
    "num_kvcache_blocks",
    "prefill_token_buckets",
    "batch_size_buckets",
    "decode_block_buckets",
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


def _runtime_spec_from_engine_config(
    engine_config: EngineConfig,
    model_config: ModelConfig,
    drafter: DrafterConfig | None,
) -> RuntimeSpec:
    return RuntimeSpec.promoted(model_config, engine_config, drafter=drafter)


def _tree_nbytes(value: object) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(value)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


class LLMEngine:
    """Request lifecycle engine for the promoted Qwen 3.5 serving path."""

    def __init__(
        self,
        model_path: str,
        *,
        engine_config: EngineConfig | None = None,
        drafter: DrafterConfig | None = None,
        **kwargs,
    ):
        if engine_config is not None and kwargs:
            raise TypeError("Pass either engine_config or workload/capacity kwargs, not both")
        if engine_config is None:
            engine_config = _engine_config_from_public_kwargs(model_path, kwargs)
        elif engine_config.model != model_path:
            engine_config = replace(engine_config, model=model_path)

        self.model_id = model_path
        metadata_path = resolve_checkpoint_metadata(model_path)
        self.model_config = ModelConfig.from_checkpoint(
            metadata_path,
            model=model_path,
        )
        self.checkpoint_path = (
            metadata_path
            if Path(model_path).expanduser().exists()
            else resolve_checkpoint(model_path, revision=metadata_path.name)
        )
        self.config = _runtime_spec_from_engine_config(
            engine_config,
            self.model_config,
            drafter,
        )
        requested_kv_blocks = self.config.capacity.num_kvcache_blocks
        kv_spec = resolve_kv_cache_spec(
            KVCacheSpec(
                num_layers=self.config.model.num_hidden_layers,
                num_blocks=requested_kv_blocks,
                block_size=self.config.capacity.block_size,
                num_kv_heads=self.config.model.num_key_value_heads,
                head_dim=self.config.model.head_dim,
                dtype=self.config.compile.jax_dtype(),
                max_kv_cache_bytes=self.config.capacity.max_kv_cache_bytes,
            ),
            self.config.kernels,
        )
        effective_blocks = kv_spec.num_blocks
        if effective_blocks != requested_kv_blocks:
            print(
                "KV cache capped: "
                f"{requested_kv_blocks} -> {effective_blocks} blocks "
                f"({self.config.capacity.max_kv_cache_bytes} byte cap)"
            )
        self.config = replace(
            self.config,
            capacity=replace(
                self.config.capacity,
                num_kvcache_blocks=effective_blocks,
            ),
        )

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required; install it with the package dependencies")

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        eos_token_ids = tuple(
            sorted({
                int(token_id)
                for token_id in (
                    *self.config.capacity.eos_token_ids,
                    self.tokenizer.eos_token_id,
                )
                if token_id is not None
            })
        )
        self.config = replace(
            self.config,
            capacity=replace(self.config.capacity, eos_token_ids=eos_token_ids),
        )

        print(f"Loading pretrained weights from {self.checkpoint_path}...")
        self.params = load_weights_from_hf_streaming(
            self.checkpoint_path,
            self.config.model,
            self.config.compile.weight_dtype,
        )
        self.mtp_params = (
            load_mtp_weights_from_hf_streaming(
                self.checkpoint_path,
                self.config.model,
                self.config.compile.weight_dtype,
            )
            if self.config.drafter is not None
            else None
        )
        print("✓ Using pretrained weights")

        self.scheduler = Scheduler(self.config)
        runner_kwargs = (
            {"mtp_params": self.mtp_params}
            if self.mtp_params is not None
            else {}
        )
        self.model_runner = ModelRunner(self.config, self.params, **runner_kwargs)
        self.startup_device_budget_bytes = {
            "parameters": _tree_nbytes(self.params),
            "draft_parameters": _tree_nbytes(self.mtp_params),
            **self.model_runner.memory_bytes(),
        }
        memory_mib = sum(self.startup_device_budget_bytes.values()) / (1024 * 1024)
        print(f"Startup device budget: {memory_mib:.1f} MiB")
        self._next_seq_id = 0
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
        if not self.scheduler.is_pristine():
            raise RuntimeError(
                "warmup_compilation must run before requests or prefix-cache state"
            )
        if max_prefill_len is None:
            max_prefill_len = max(
                self.config.compile.prefill_token_buckets
                or (self.config.capacity.max_num_batched_tokens,)
            )
        if max_batch is None:
            max_batch = max(
                self.config.compile.batch_size_buckets
                or (self.config.capacity.max_num_seqs,)
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

        seq = Sequence(
            prompt,
            sampling_params,
            seq_id=self._next_seq_id,
            block_size=self.config.capacity.block_size,
        )
        self._next_seq_id += 1
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

    def commit(
        self,
        seqs: List[Sequence],
        schedule_plan: SchedulePlan,
        run_result: RunResult,
    ) -> StepResult:
        """Commit one runner result to logical request state."""
        if len(run_result.rows) != len(seqs):
            raise ValueError("run result rows must align with scheduled sequences")

        prefill_chunk_lengths = (
            schedule_plan.prefill_chunk_lengths
            if schedule_plan.is_prefill
            else (0,) * len(seqs)
        )
        if len(prefill_chunk_lengths) != len(seqs):
            raise ValueError("prefill chunk lengths must align with scheduled sequences")

        emitted: list[TokenEvent] = []
        finished: list[FinishedRequest] = []
        for seq, tokens, prefill_chunk_len in zip(
            seqs,
            run_result.rows,
            prefill_chunk_lengths,
        ):
            if prefill_chunk_len:
                seq.num_cached_tokens = min(
                    seq.num_prompt_tokens,
                    seq.num_cached_tokens + int(prefill_chunk_len),
                )

            for index, token in enumerate(tokens):
                deferred = (
                    self.scheduler.device_token_carry
                    and seq.ignore_eos
                    and is_device_token(token)
                )
                if deferred:
                    completion_index = seq.output.append_device(token)
                    is_eos = False
                    event_token = token
                else:
                    event_token = int(token)
                    completion_index = seq.output.append(event_token)
                    is_eos = event_token in self.scheduler.eos_token_ids
                emitted.append(TokenEvent(seq.seq_id, completion_index, event_token))

                if index < len(tokens) - 1:
                    self.scheduler.block_manager.commit_processed_token(seq)

                reason = None
                if not seq.ignore_eos and is_eos:
                    reason = FinishReason.EOS
                elif seq.num_completion_tokens >= seq.max_tokens:
                    reason = FinishReason.LENGTH
                if reason is not None:
                    seq.status = SequenceStatus.FINISHED
                    self.scheduler.release(seq)
                    finished.append(FinishedRequest(seq.seq_id, reason))
                    break

        return StepResult(
            phase="prefill" if schedule_plan.is_prefill else "decode",
            scheduled_tokens=int(schedule_plan.num_scheduled_tokens),
            emitted_tokens=tuple(emitted),
            finished=tuple(finished),
            verified_target_tokens=run_result.verified_target_tokens,
            draft_tokens=run_result.draft_tokens,
            accepted_draft_tokens=run_result.accepted_draft_tokens,
        )

    def _release_invalidated_prefix_states(self) -> None:
        handles = self.scheduler.take_released_prefix_state_handles()
        if handles:
            self.model_runner.release_prefix_hybrid_states(handles)

    def step(self) -> StepResult:
        seqs, schedule_plan = self.scheduler.schedule()
        prefill_chunk_lengths = (
            list(schedule_plan.prefill_chunk_lengths)
            if schedule_plan.is_prefill
            else None
        )

        self._release_invalidated_prefix_states()
        self.model_runner.install_cached_prefix_hybrid_states(
            seqs,
            self.scheduler.cached_prefix_entries(seqs),
        )

        device_batch = self.model_runner.materialize(schedule_plan)
        run_result = self.model_runner.execute(seqs, device_batch)

        if schedule_plan.is_prefill:
            pending = self.scheduler.record_computed_prefixes(
                seqs,
                prefill_chunk_lengths or [],
            )
            self._release_invalidated_prefix_states()
            if pending:
                handles = self.model_runner.cache_prefix_hybrid_states(
                    pending,
                )
                self.scheduler.publish_prefix_states(pending, handles)
        step_result = self.commit(seqs, schedule_plan, run_result)
        finished_seq_ids = [request.seq_id for request in step_result.finished]
        if finished_seq_ids:
            self.model_runner.release(finished_seq_ids)
        return step_result

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def cancel_request(self, seq: Sequence) -> bool:
        """Commit cancellation and release all state owned by one request."""
        if seq.is_finished:
            return False
        self.scheduler.release(seq)
        self.model_runner.release([seq.seq_id])
        seq.status = SequenceStatus.FINISHED
        return True

    def generate(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        seqs = self._prepare_generation_sequences(prompts, sampling_params)
        seqs_by_id = {seq.seq_id: seq for seq in seqs}
        if use_tqdm:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise ImportError(
                    "tqdm is required only for progress bars; install the "
                    "`progress` extra or call generate(..., use_tqdm=False)"
                ) from exc
            pbar = tqdm(total=len(seqs), desc="Generating", dynamic_ncols=True)

        outputs = {}
        finish_reasons = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            step_result = self.step()

            if use_tqdm:
                if step_result.phase == "prefill":
                    prefill_throughput = step_result.scheduled_tokens / (perf_counter() - t)
                else:
                    decode_throughput = step_result.num_emitted_tokens / (perf_counter() - t)

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })

            finished_seqs = [
                seqs_by_id[request.seq_id]
                for request in step_result.finished
            ]
            token_rows = OutputBuffer.materialize_many(seq.output for seq in finished_seqs)
            for request, seq, token_ids in zip(step_result.finished, finished_seqs, token_rows):
                outputs[seq.seq_id] = token_ids
                finish_reasons[seq.seq_id] = request.reason.value
                if use_tqdm:
                    pbar.update(1)

        ordered_ids = sorted(outputs)
        results = [
            {
                "text": self._detokenize(outputs[seq_id]),
                "token_ids": outputs[seq_id],
                "finish_reason": finish_reasons[seq_id],
            }
            for seq_id in ordered_ids
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
        seqs_by_id = {seq.seq_id: seq for seq in seqs}
        stream_start = perf_counter()

        while not self.is_finished():
            step_start = perf_counter()
            step_result = self.step()
            step_end = perf_counter()
            event_buffers = {
                event.seq_id: seqs_by_id[event.seq_id].output
                for event in step_result.emitted_tokens
            }
            OutputBuffer.snapshot_many(event_buffers.values()).prefetch().materialize()
            for event in step_result.emitted_tokens:
                seq = seqs_by_id[event.seq_id]
                token_id = seq.output.token_id(event.completion_index)
                token_event = {
                    "event": "token",
                    "seq_id": event.seq_id,
                    "request_index": seq_to_request[event.seq_id],
                    "completion_index": event.completion_index,
                    "token_id": token_id,
                    "elapsed_seconds": step_end - stream_start,
                    "step_seconds": step_end - step_start,
                    "step_start_seconds": step_start - stream_start,
                    "step_end_seconds": step_end - stream_start,
                    "scheduler_step_tokens": step_result.scheduled_tokens,
                    "scheduler_step_is_decode": step_result.is_decode,
                    "verified_target_tokens": step_result.verified_target_tokens,
                    "draft_tokens": step_result.draft_tokens,
                    "accepted_draft_tokens": step_result.accepted_draft_tokens,
                }
                if include_text:
                    token_event["text"] = self._detokenize([token_id])
                yield token_event
            for request in step_result.finished:
                seq = seqs_by_id[request.seq_id]
                yield {
                    "event": "finished",
                    "seq_id": request.seq_id,
                    "request_index": seq_to_request[request.seq_id],
                    "finish_reason": request.reason.value,
                    "elapsed_seconds": step_end - stream_start,
                    "completion_tokens": seq.num_completion_tokens,
                }

        OutputBuffer.snapshot_many(seq.output for seq in seqs).prefetch().materialize()

        yield {
            "event": "done",
            "elapsed_seconds": perf_counter() - stream_start,
            "results": [
                {
                    "request_index": index,
                    "text": self._detokenize(seq.output.token_ids()) if include_text else "",
                    "token_ids": seq.output.token_ids(),
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
