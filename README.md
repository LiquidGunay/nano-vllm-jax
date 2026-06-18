# Nano-vLLM-JAX

Nano-vLLM-JAX is a compact, pedagogical serving runtime for Qwen3.5-family checkpoints on JAX/CUDA. It keeps the serving stack small enough to inspect while implementing the core pieces expected from a modern LLM server: paged KV cache, chunked/ragged prefill, continuous batching, reusable prefix cache, optimized attention/GDN routes, and a local chat UI.

The current promoted artifact is a non-MTP GPU serving path for `Qwen/Qwen3.5-0.8B`. MTP speculative decoding remains in the repo as an experimental path, but it is not the default serving configuration and should not be used for speed claims until it beats the same non-MTP baseline with target-model verification enabled.

## One-Command Startup

From the repository root:

```bash
pip install -e ".[cuda13,flashinfer-ffi,gdn-fla-triton]"
tools/start_local_chat.sh
```

This starts:

- model server: `http://127.0.0.1:6791`
- chat UI: `http://127.0.0.1:6789`
- logs: `/mountpoint/.exp/run_logs/nvj_model_server_6791.log` and `/mountpoint/.exp/run_logs/nvj_chat_ui_6789.log`

The launcher uses [configs/server/gpu_optimal.yaml](configs/server/gpu_optimal.yaml), waits for the model server to finish loading and generic compilation warmup, then starts the browser UI. The root [server_config.yaml](server_config.yaml) mirrors the same promoted config, so `python server.py` uses the same path.

The promoted config can serve the full configured bucket set, but startup only
warms a bounded common profile by default:

- prefill token buckets: `64,128`
- batch-size buckets: `1,4`
- decode block-table buckets: `128,320`
- greedy and full-vocab temperature sampled trace routes

This keeps first startup practical on a 0.8B model while still compiling real
serving paths before the UI is exposed. On the current A10G test host, this
compiled startup completed in `216.51s` on the first bounded run and `76.92s`
on the next run with the persistent JAX cache populated, with `26` JIT cache
entries.

For a faster installation smoke test that skips startup compilation:

```bash
NANO_VLLM_JAX_SERVER_CONFIG=configs/server/gpu_minimal_pure_jax.yaml \
NANO_VLLM_JAX_SKIP_COMPILE_STARTUP=1 \
tools/start_local_chat.sh
```

With `--skip-compile`, startup mostly pays model loading; the first request compiles whatever shape it uses. With the compiled path, startup is slower, but the server is warmed for the startup profile before the UI is exposed. Persistent JAX compilation cache entries are stored under `/mountpoint/.exp/.cache/jax` by default, so repeated startups with the same model, JAX/XLA version, hardware, dtype, bucket shapes, and kernel/config choices should reuse compiled artifacts where JAX can cache them.

## API Smoke

Raw completion-style request:

```bash
curl http://127.0.0.1:6791/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Write one sentence about JAX serving.","max_tokens":32}'
```

Chat-template request:

```bash
curl http://127.0.0.1:6791/v1/generate_trace \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with one short sentence about JAX."}],"max_tokens":32,"temperature":0}'
```

The Flask routes accept raw `prompt`, pre-tokenized `input_ids`, or chat `messages`. Chat messages are formatted on the server with the loaded tokenizer's chat template and `add_generation_prompt=True`. EOS stopping is enabled by default unless `ignore_eos` is explicitly set.

`/v1/generate_trace` has two trace modes in the promoted config:

- `ignore_eos: false` uses the EOS-compatible trace path and reports TTFT/TPS.
- `ignore_eos: true` uses the faster device-token-carry trace path that avoids per-token readback and runs to `max_tokens`.

The local chat UI uses the EOS-compatible mode by default.

## Current Capabilities

- CUDA/JAX serving for `Qwen/Qwen3.5-0.8B` real weights.
- Config-owned server startup through YAML rather than a large pile of required environment variables.
- Paged full-attention KV cache and block-table scheduling.
- Chunked/ragged packed prefill with generic bucket warmup.
- Continuous batching for prefill, decode, and mixed prefill/decode serving.
- Prefix caching for reusable full-block prompt prefixes. For Qwen3.5 hybrid GDN layers, a prefix is skipped only when both paged full-attention KV blocks and the matching GDN hybrid state for that exact prefix hash are recorded.
- Server-side chat template support and EOS termination.
- Local browser chat UI showing TTFT and tokens/sec after each response.
- Optional optimized kernel routes in the promoted config: FlashInfer paged full-attention decode, Triton packed full-attention prefill, Triton/FLA GDN prefill, Triton greedy LM-head top-1, BF16 decode projection paths, and accepted compact prefill routes.
- Experimental MTP-1 speculative decoding kept separate in [configs/server/mtp_experimental.yaml](configs/server/mtp_experimental.yaml). It keeps target-model verification enabled and unverified draft append disabled.
- Benchmark and profiling harnesses for JAX server traces, vLLM comparison, random workloads, and optimization log summaries.

## Current Artifact Configuration

The promoted path is [configs/server/gpu_optimal.yaml](configs/server/gpu_optimal.yaml):

- model: `Qwen/Qwen3.5-0.8B`
- model server port: `6791`
- UI port: `6789`
- dtype and weights: BF16
- execution: JAX JIT
- max resident sequences: `8`
- max prefill: `4096`
- KV budget: `3072 MiB`
- startup warmup profile: prefill `64,128`, batch `1,4`, decode block tables `128,320`
- prefix cache: enabled
- MTP: off
- local CUDA probe kernels: disabled

Useful overrides:

```bash
NANO_VLLM_JAX_SERVER_CONFIG=configs/server/gpu_minimal_pure_jax.yaml tools/start_local_chat.sh
NANO_VLLM_JAX_SKIP_COMPILE_STARTUP=1 tools/start_local_chat.sh
NANO_VLLM_JAX_PORT=6792 NANO_VLLM_JAX_CHAT_UI_PORT=6793 tools/start_local_chat.sh
NANO_VLLM_JAX_PREFIX_CACHE=0 tools/start_local_chat.sh
NANO_VLLM_JAX_STARTUP_WARMUP_PREFILL_TOKEN_BUCKETS=64,128,512 tools/start_local_chat.sh
NANO_VLLM_JAX_STARTUP_WARMUP_BATCH_SIZE_BUCKETS=1,4,8 tools/start_local_chat.sh
NANO_VLLM_JAX_COMPILE_CACHE_DIR=/mountpoint/.exp/.cache/jax tools/start_local_chat.sh
```

## Vision

The goal is a performant and understandable serving library, not a black-box production server. The target shape is:

- close enough to vLLM-style serving to teach the important systems ideas;
- small and readable enough to follow scheduler, cache, kernel, and model-state ownership end to end;
- correctness-first for hybrid Qwen3.5 models, where full-attention KV state and GDN recurrent/conv state must advance together;
- benchmarkable against vLLM with clear baselines and no benchmark-specific warmup tricks;
- extensible across the Qwen3.5 family through config and bucket choices rather than model/GPU-specific hand tuning;
- honest about experimental routes, especially MTP speculative decoding and borrowed/custom kernels.

## Repository Structure

```text
nanovllm_jax/
  config.py                  Model and serving config dataclass
  server_config.py           YAML config loader and env override projection
  engine/
    llm_engine.py            Public generation loop
    scheduler.py             Continuous batching, prefix cache, block allocation
    block_manager.py         Paged block tables, refcounts, prefix hashes
    model_runner.py          Serving fast-path selection and mutable runtime state
    model_executor.py        JIT/executor boundary and compiled step variants
    scheduled_batch.py       Scheduler-to-executor batch contract
  model.py                   Qwen3.5 forward pass and hybrid state logic
  kernels/                   FlashInfer/Triton/Pallas/CUDA-probe integration points
configs/server/
  gpu_optimal.yaml           Promoted non-MTP CUDA serving path
  gpu_minimal_pure_jax.yaml  Small installation and endpoint smoke path
  mtp_experimental.yaml      Verified MTP diagnostic path
benchmarks/
  benchmark_jax_server_trace.py  JAX server TTFT/ITL tracing
  benchmark_vllm_qwen35.py       vLLM comparison harness
  run_gpu_matrix.py              Matrix benchmark runner
tools/
  start_local_chat.sh        One-command model server + chat UI launcher
  chat_ui_server.py          Local browser UI and backend proxy
docs/
  serving_paths.md           Current serving recipes and path separation
  architecture.md            Engine/scheduler/runner/executor architecture
  gpu_correctness_guardrails.md  Correctness and cache-root rules
  optimization_logbook.md    Historical optimization experiments and decisions
```

## Runtime Paths And Cache

GPU runs should keep caches and temporary files under `/mountpoint/.exp`:

```bash
export NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp
export TMPDIR=/mountpoint/.exp/tmp
export XDG_CACHE_HOME=/mountpoint/.exp/.cache
export HF_HOME=/mountpoint/.exp/.cache/huggingface
export HF_HUB_CACHE=/mountpoint/.exp/.cache/huggingface/hub
export JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax
export NANO_VLLM_JAX_COMPILE_CACHE_DIR=/mountpoint/.exp/.cache/jax
```

The launcher and server set sensible CUDA/JAX defaults from the YAML config, including `JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and `TF_GPU_ALLOCATOR=cuda_malloc_async`. Runtime env vars remain available for compatibility and quick experiments, but promoted serving behavior should live in config files.

## Documentation

- [Documentation index](docs/README.md)
- [Serving paths](docs/serving_paths.md)
- [Architecture](docs/architecture.md)
- [GPU correctness guardrails](docs/gpu_correctness_guardrails.md)
- [KV cache](docs/kv_cache.md)
- [MTP speculative decoding](docs/mtp.md)
- [Scheduler](docs/scheduler.md)
- [Benchmarks](docs/benchmarks.md)
- [Roadmap](docs/roadmap.md)

Historical and obsolete notes are archived under [docs/archive/2026-05-pre-latency-gate](docs/archive/2026-05-pre-latency-gate/).

## Programmatic Minimal Usage

```python
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights import load_weights_from_hf
from nanovllm_jax.engine.model_runner import ModelRunner

config = Qwen3_5Config.qwen3_5_0_8b()
params = load_weights_from_hf("Qwen/Qwen3.5-0.8B", config)
runner = ModelRunner(config, params)
```
