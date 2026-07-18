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
uv run --frozen --python 3.11 \
  --extra cuda13 --extra flashinfer-ffi --extra gdn-fla-triton \
  python server.py
```

The lockfile and Python 3.11 are the reproducible environment contract.
The current lock resolves JAX/JAXlib and the CUDA 13 plugin to 0.10.0,
FlashInfer to 0.6.11.post3, JAX-Triton to 0.3.1, and JAX TVM FFI to 0.1.3.
`uv.lock`, rather than these descriptive version notes, remains authoritative.

An editable pip install remains convenient for development, but it does not
pin the transitive CUDA stack:

```bash
pip install -e ".[cuda13,flashinfer-ffi,gdn-fla-triton]"
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
`SamplingParams` defaults to greedy decoding (`temperature=0`), matching HTTP;
explicit token-id prompts are checked against the checkpoint vocabulary before
the request enters the scheduler.

## Run The Benchmark

The repository makes one narrow claim: Qwen3.5-4B greedy B=1 decode with a
64-token prompt, 64 generated tokens, and optional K=2 MTP. The exact
checkpoint, workload, framework version, and validity gates live in
[benchmarks/benchmark.json](benchmarks/benchmark.json). The committed
[recorded result](benchmarks/recorded_result.json) is one historical A10G
observation, not a pass threshold for new runs.

| Framework | Base decode tok/s | MTP decode tok/s | MTP/base |
| --- | ---: | ---: | ---: |
| Nano-VLLM-JAX | 53.99 | 82.72 | 1.532x |
| vLLM 0.25.1 | 50.34 | 86.60 | 1.720x |

JAX MTP is 1.643x vLLM without MTP and 0.955x vLLM with MTP on this fixed
workload. The four routes resolve one BF16 near-tie in either direction; the
two exact output hashes and full-vocabulary KL/JS evidence are content-addressed
by the benchmark contract. Any other output still fails validation.

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
independent clients can batch together. The service owns engine stepping until
`stop()` returns. Offline `generate()` owns an otherwise idle engine for one
call; `add_request()` plus `step()` is the explicit manual lifecycle. A service
lease rejects manual mutation, and unfinished manual work prevents lease
acquisition.

The bundled Flask server is intentionally a local pedagogical transport, not a
production WSGI deployment recipe.

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
Logical output, resident target length, and selected GDN state advance only by
the committed prefix. Physical target and predictor KV writes may run ahead,
but resident length bounds attention visibility and reserved block capacity
covers the farthest speculative write.
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

- [docs/speculative-verification.md](docs/speculative-verification.md) and
  [nanovllm_jax/mtp.py](nanovllm_jax/mtp.py) - experimental constructor-time
  persistent MTP with packed target verification.
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

The local control-plane check does not require a GPU and does not create a CI
workflow:

```bash
./scripts/check.sh
```

The check creates or refreshes `.venv` from the frozen lock, then runs the
CPU-safe ownership, admission, configuration, server, route, commit, and
benchmark-contract suites under the 70% RAM guard. The benchmark execution
remains a separate explicit command.

GPU correctness matrix:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=$PWD python tests/ram_guard.py -- pytest -q \
  tests/test_engine_initialization.py \
  tests/test_fastpath_config.py \
  tests/test_public_imports.py \
  tests/test_server_config.py
```

For GPU correctness, verify CUDA visibility first and run JAX with
`JAX_PLATFORMS=cuda`; do not hide missing GPU access with CPU fallback.

The project is available under the [MIT License](LICENSE).
