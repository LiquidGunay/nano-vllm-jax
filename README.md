# Nano-VLLM-JAX

Nano-VLLM-JAX is a compact CUDA/JAX serving engine for Qwen3.5-family text
checkpoints. The cleaned mainline is intentionally narrow: it executes and
explains one accepted serving path instead of exposing the full experimental
search space as runtime configuration.

The default target is `Qwen/Qwen3.5-0.8B`; the validated dense text sizes are
0.8B, 2B, and 4B. Architecture is read from each checkpoint and checked before
weights are loaded. The promoted path uses BF16 weights and compute, packed
prefill, paged decode, prefix caching, device token carry, resident decode
metadata, and queue-driven continuous batching.

## Start The Server

```bash
pip install -e ".[cuda13,flashinfer-ffi,gdn-fla-triton]"
python server.py
```

[server.yaml](server.yaml) controls model id, serving capacity, bucket sizes,
KV budget, warmup buckets, and prefix-cache enablement.

Requests reserve worst-case KV capacity credits before prefill, while physical
pages are allocated only as tokens need them. This keeps active requests safe
without evicting cached prefixes for unwritten future tokens. A request that
can never fit is rejected. Tied vocabulary weights remain in checkpoint-native
`[V, H]` layout instead of allocating a second transposed copy.

[nanovllm_jax/config.py](nanovllm_jax/config.py) composes model, capacity, and
compile specs once at startup. [nanovllm_jax/fastpath.py](nanovllm_jax/fastpath.py)
owns attention/GDN routes, LM-head policy, device token carry, and metadata
residency. Users should not switch implementations through YAML on the cleaned
branch.

The offline `LLM.generate(..., use_tqdm=True)` progress bar uses the optional
`progress` extra. Serving does not require it.

## Run The Benchmark

The repository makes one narrow claim: Qwen3.5-4B greedy B=1 decode with a
64-token prompt and 64 generated tokens. The exact checkpoint, workload,
framework version, and validity gates live in
[benchmarks/benchmark.json](benchmarks/benchmark.json). The committed
[recorded result](benchmarks/recorded_result.json) is one historical A10G
observation, not a pass threshold for new runs.

```bash
./scripts/run_benchmark.sh both
```

JAX and vLLM run in separate environments because their CUDA Python packages
conflict. vLLM is therefore optional. Environments, model cache, and raw JSON
results derive from `NANO_VLLM_JAX_BENCHMARK_ROOT`, which defaults to
`/mountpoint/.exp`; the script stops at 80% system RAM use by default.
[docs/benchmark.md](docs/benchmark.md) defines the measured window and reports
the committed result.

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

Streaming emits coalesced `tokens` events with `completion_start`, `token_ids`,
and an append-only `text` delta. The final `done` event carries the canonical
text, token ids, and finish reason.

HTTP handlers submit work to `EngineService`. A single worker admits queued
requests, calls `LLMEngine.step()`, and publishes per-request results so
independent clients can batch together.

## Runtime Path

```text
server.py
  -> EngineService
  -> LLMEngine
  -> Scheduler -> BlockManager
  -> SchedulePlan (host)
  -> ModelRunner
  -> DeviceBatch + HostBatch
  -> ModelExecutor
  -> Qwen3.5 model
  -> attention / GDN / LM-head kernels
  -> RunResult
  -> LLMEngine.commit()
  -> StepResult
```

The central invariant is:

```text
logical sequence length, allocated block capacity, full-attention KV state,
and GDN hybrid state advance by the same committed prefix.
```

## Reading Path

Core engine:

1. [nanovllm_jax/engine.py](nanovllm_jax/engine.py),
   [nanovllm_jax/step.py](nanovllm_jax/step.py), and
   [nanovllm_jax/output.py](nanovllm_jax/output.py) - request lifecycle and commit.
2. [nanovllm_jax/scheduler.py](nanovllm_jax/scheduler.py) - prefill and decode selection.
3. [nanovllm_jax/block_manager.py](nanovllm_jax/block_manager.py) - paged capacity and prefix reuse.
4. [nanovllm_jax/batch.py](nanovllm_jax/batch.py) and
   [nanovllm_jax/device_batch.py](nanovllm_jax/device_batch.py) - host planning
   and runner-owned device materialization.
5. [nanovllm_jax/runner.py](nanovllm_jax/runner.py) and
   [nanovllm_jax/executor.py](nanovllm_jax/executor.py) - device state and the selected compiled call.
6. [nanovllm_jax/model.py](nanovllm_jax/model.py) - one model transition.

Advanced serving:

- [nanovllm_jax/service.py](nanovllm_jax/service.py), [server.yaml](server.yaml),
  and [nanovllm_jax/config.py](nanovllm_jax/config.py) - online queues and capacity.
- [nanovllm_jax/fastpath.py](nanovllm_jax/fastpath.py) - promoted operation policy.
- [nanovllm_jax/projection.py](nanovllm_jax/projection.py),
  [nanovllm_jax/attention.py](nanovllm_jax/attention.py),
  [nanovllm_jax/gdn.py](nanovllm_jax/gdn.py), and
  [nanovllm_jax/lm_head.py](nanovllm_jax/lm_head.py) - model operations.
- [nanovllm_jax/cache.py](nanovllm_jax/cache.py) and
  [nanovllm_jax/kernels](nanovllm_jax/kernels) - cache layout and low-level routes.

## Development

Generated results and profiles are not part of the cleaned branch. The fixed
benchmark contract and concise report are committed; raw runs and ad hoc
diagnostics stay under `/mountpoint/.exp`.

[docs/style.md](docs/style.md) defines the repository's lightweight complexity
budget and review checks.

Ownership and configuration contract checks:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=$PWD python tests/ram_guard.py -- pytest -q \
  tests/test_engine_initialization.py \
  tests/test_fastpath_config.py \
  tests/test_public_imports.py \
  tests/test_server_config.py
```

For GPU correctness, verify CUDA visibility first and run JAX with
`JAX_PLATFORMS=cuda`; do not hide missing GPU access with CPU fallback.
