# Nano-VLLM-JAX

Nano-VLLM-JAX is a compact CUDA/JAX serving engine for Qwen3.5-family text
checkpoints. The cleaned mainline is intentionally narrow: it executes and
explains one accepted serving path instead of exposing the full experimental
search space as runtime configuration.

The promoted target is `Qwen/Qwen3.5-0.8B` with BF16 weights and compute,
packed prefill, paged decode, prefix caching, device token carry, resident
decode metadata, and queue-driven continuous batching.

## Start The Server

```bash
pip install -e ".[cuda13,flashinfer-ffi,gdn-fla-triton]"
python server.py
```

[server.yaml](server.yaml) controls model id, serving capacity, bucket sizes,
KV budget, warmup buckets, and prefix-cache enablement.

[nanovllm_jax/fastpath.py](nanovllm_jax/fastpath.py) owns implementation
policy: dtypes, attention/GDN routes, LM-head route, device token carry, and
metadata residency. Users should not switch kernels through YAML on the cleaned
branch.

## API Smoke

```bash
curl http://127.0.0.1:6791/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Write one sentence about JAX serving.","max_tokens":32}'
```

Streaming:

```bash
curl http://127.0.0.1:6791/v1/generate_stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Reply with one short sentence.","max_tokens":32,"temperature":0}'
```

HTTP handlers submit work to `EngineService`. A single worker admits queued
requests, calls `LLMEngine.step()`, and publishes per-request results so
independent clients can batch together.

## Runtime Path

```text
server.py
  -> EngineService
  -> LLMEngine
  -> Scheduler -> BlockManager
  -> ScheduledBatch
  -> ModelRunner
  -> ModelExecutor
  -> Qwen3.5 model
  -> attention / GDN / LM-head kernels
```

The central invariant is:

```text
logical sequence length, allocated block capacity, full-attention KV state,
and GDN hybrid state advance by the same committed prefix.
```

## Reading Path

1. [nanovllm_jax/fastpath.py](nanovllm_jax/fastpath.py) - promoted operation policy.
2. [server.yaml](server.yaml) and [nanovllm_jax/config.py](nanovllm_jax/config.py) - capacity and buckets.
3. [nanovllm_jax/service.py](nanovllm_jax/service.py) - online request queue.
4. [nanovllm_jax/engine.py](nanovllm_jax/engine.py) - request lifecycle and step loop.
5. [nanovllm_jax/scheduler.py](nanovllm_jax/scheduler.py) and [nanovllm_jax/block_manager.py](nanovllm_jax/block_manager.py) - work selection and cache pages.
6. [nanovllm_jax/batch.py](nanovllm_jax/batch.py) - Python-to-JAX batch contract.
7. [nanovllm_jax/runner.py](nanovllm_jax/runner.py) and [nanovllm_jax/executor.py](nanovllm_jax/executor.py) - persistent device state and compiled calls.
8. [nanovllm_jax/model.py](nanovllm_jax/model.py) - parameter structure, layer loop, and forward entrypoints.
9. [nanovllm_jax/projection.py](nanovllm_jax/projection.py), [nanovllm_jax/attention.py](nanovllm_jax/attention.py), [nanovllm_jax/gdn.py](nanovllm_jax/gdn.py), and [nanovllm_jax/lm_head.py](nanovllm_jax/lm_head.py) - the model math split by role.
10. [nanovllm_jax/cache.py](nanovllm_jax/cache.py) and [nanovllm_jax/kernels](nanovllm_jax/kernels) - cache layout and low-level routes.

## Development

Generated results, profiles, and benchmark artifacts are not part of the
cleaned branch. Keep ad hoc diagnostics under `/mountpoint/.exp/diagnostics` or
another external scratch path.

CPU-safe control-plane checks:

```bash
PYTHONPATH=$PWD python tests/ram_guard.py -- pytest -q tests/test_fastpath_config.py tests/test_service.py tests/test_server_config.py tests/test_public_imports.py
```

For GPU correctness, verify CUDA visibility first and run JAX with
`JAX_PLATFORMS=cuda`; do not hide missing GPU access with CPU fallback.
