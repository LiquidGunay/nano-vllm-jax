# Nano-vLLM-JAX

A compact vLLM-style JAX runtime for Qwen3.5-family checkpoints, with paged KV cache, hybrid linear/full attention, scheduler-driven batching, and experimental MTP speculative decoding.

This repository is a correctness-focused research/prototype codebase. Heavy validation and benchmark work should run on the TPU VM, not locally.

## Current documentation

- [Architecture](docs/architecture.md)
- [KV cache and hybrid state](docs/kv_cache.md)
- [MTP speculative decoding](docs/mtp.md)
- [Scheduler](docs/scheduler.md)
- [Benchmarks](docs/benchmarks.md)
- [Roadmap](docs/roadmap.md)

Archived historical MTP, benchmark, and stale status notes live in [docs/archive/2026-05-pre-current-state](docs/archive/2026-05-pre-current-state/).

## Runtime path

```text
LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> Backend -> model.forward_step
```

`ModelExecutor` is the canonical execution path. `ModelRunner` owns runtime/session state and compatibility helpers around the executor.

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

## Notes

- MTP is experimental and limited to K=1 or K=2 work.
- K=1 safe mode is the correctness baseline; speed remains workload-dependent.
- K=2 needs TPU validation before being treated as correct.
- Do not treat archived status reports as current state without rerunning the TPU correctness gates.
