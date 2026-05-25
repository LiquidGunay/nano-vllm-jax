# Nano-vLLM-JAX

A compact vLLM-style JAX runtime for Qwen3.5-family checkpoints, with paged KV cache, hybrid linear/full attention, scheduler-driven batching, and experimental MTP speculative decoding.

This is a correctness-focused research/prototype codebase. Current GPU work treats HuggingFace Qwen3.5 with BF16 checkpoint values and FP32 activation math as the correctness reference. Older TPU MTP notes remain useful history, but they are not current GPU serving guidance.

## Current validated state

- Hardware/backend: CUDA GPU through JAX backend `gpu`.
- Model: `Qwen/Qwen3.5-0.8B`, real weights.
- Runtime contract: `dtype=float32` with `weight_dtype=bfloat16`.
- Long-decode correctness: the current BF16-weight/FP32-activation artifact checks 500 decode steps against HF top-5 with no top-1, ordered top-5, or top-5 set mismatches.
- Server-shape benchmark: `benchmark_server_shapes.py` compares HF, JAX paged attention, and MTP1. Current GPU artifacts show JAX paged generation and exact commit-select MTP1 both match HF/JAX generated tokens across the server-shape suite, but MTP1 is still slower than the regular JAX paged baseline.

MTP remains experimental. Do not claim current GPU MTP speedups until MTP1 beats the JAX paged baseline on the same correctness-checked benchmark shape suite.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [GPU correctness guardrails](docs/gpu_correctness_guardrails.md)
- [KV cache](docs/kv_cache.md)
- [MTP speculative decoding](docs/mtp.md)
- [Scheduler](docs/scheduler.md)
- [Benchmarks](docs/benchmarks.md)
- [Roadmap](docs/roadmap.md)
- [Historical TPU findings](docs/mtp_tpu_spot_findings_2026-05-09.md)

Historical and obsolete notes are archived under [docs/archive/2026-05-pre-latency-gate](docs/archive/2026-05-pre-latency-gate/).

## Runtime path

```text
LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> Backend -> model.forward_step
```

`ModelExecutor` is the canonical execution boundary. The scheduler owns runnable work, block allocation, and MTP admission.

## GPU runtime environment

Use workspace-scoped cache and temporary paths for GPU runs:

```bash
export NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp
export TMPDIR=/mountpoint/.exp/tmp
export XDG_CACHE_HOME=/mountpoint/.exp/.cache
export XDG_DATA_HOME=/mountpoint/.exp/.local/share
export UV_CACHE_DIR=/mountpoint/.exp/.cache/uv
export PIP_CACHE_DIR=/mountpoint/.exp/.cache/pip
export HF_HOME=/mountpoint/.exp/.cache/huggingface
export HF_HUB_CACHE=/mountpoint/.exp/.cache/huggingface/hub
export JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax
export NANO_VLLM_JAX_COMPILE_CACHE_DIR=/mountpoint/.exp/.cache/jax
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS=--xla_gpu_autotune_level=4
```

Profiler traces and run journals are also workspace-scoped by default:

- JAX profiler traces: `/mountpoint/.exp/profiles/<run-id>/`
- append-only run journal: `/mountpoint/.exp/run_logs/run_journal.jsonl`

Benchmark entry points using `RunRecorder` create run IDs in the form `<utc timestamp>-<pid>-<run label>`, write `run_manifest.json` under the profile directory, and append `start`, `profile_start`, `issue`, `profile_stop`, and `finish` events to the JSONL journal. Use `--run-label`, `--profile-dir`, and `--run-log` when a run needs explicit naming or non-default paths, while keeping those paths under `/mountpoint/.exp`.

The current vLLM comparison harness uses an isolated environment under `/mountpoint/.exp/vllm-venv` and vLLM caches under `/mountpoint/.exp/.cache/vllm`. Its offline `LLM.generate()` path records throughput and logprobs, but not true streaming ITL.

For true ITL, use:

- JAX server path: `benchmarks/benchmark_jax_server_trace.py`, or the Flask `/v1/generate_trace` and `/v1/generate_stream` routes.
- vLLM path: `benchmarks/benchmark_vllm_qwen35.py --execution async`, which uses `AsyncLLMEngine` delta outputs.

The Flask trace routes accept the same prompt forms as `/v1/generate`. `/v1/generate_trace` returns final results, per-token events, TTFT/ITL summary stats, and speculative counters. `/v1/generate_stream` returns server-sent events from the same iterator. For heterogeneous batches, `max_tokens` may be either one integer for all prompts or a list with one positive integer per prompt.

## Quick start

```bash
pip install -e .
```

Minimal usage:

```python
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights import load_weights_from_hf
from nanovllm_jax.engine.model_runner import ModelRunner

config = Qwen3_5Config.qwen3_5_0_8b()
params = load_weights_from_hf("Qwen/Qwen3.5-0.8B", config)
runner = ModelRunner(config, params)
```
