# Nano-vLLM-JAX

A compact vLLM-style JAX runtime for Qwen3.5-family checkpoints, with paged KV cache, hybrid linear/full attention, scheduler-driven batching, and experimental MTP speculative decoding.

This is a correctness-focused research/prototype codebase. Current TPU execution is a pure JAX/XLA backend running on TPU, not a dedicated TPU kernel backend.

## Current validated state

- Hardware: TPU v6e-1.
- Model: `Qwen/Qwen3.5-4B`, BF16, real weights.
- Execution: JIT on TPU.
- MTP policy: K=1 one-pass only, scheduler-owned admission, acceptance plus measured decode-latency EWMA gates.
- Correctness: `tests/test_mtp_commit_semantics.py` passes `13/13` on TPU.
- Remaining gap: latency EWMA is global, not per bucket.

K=2 is correctness-clean in focused testing but slower in observed benchmarks, so it is experimental and non-serving.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [KV cache](docs/kv_cache.md)
- [MTP speculative decoding](docs/mtp.md)
- [Scheduler](docs/scheduler.md)
- [Benchmarks](docs/benchmarks.md)
- [Roadmap](docs/roadmap.md)
- [Latest TPU findings](docs/mtp_tpu_spot_findings_2026-05-09.md)

Historical and obsolete notes are archived under [docs/archive/2026-05-pre-latency-gate](docs/archive/2026-05-pre-latency-gate/).

## Runtime path

```text
LLMEngine -> Scheduler -> ModelRunner -> ModelExecutor -> Backend -> model.forward_step
```

`ModelExecutor` is the canonical execution boundary. The scheduler owns runnable work, block allocation, and MTP admission.

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
