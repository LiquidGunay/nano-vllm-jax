# Benchmarks

Benchmark results are valid only when correctness, warmup, and hardware rules are followed. Keep this file concise; store detailed logs in dated findings docs when needed.

## Current GPU benchmark target

Current validated GPU benchmark target:

- CUDA GPU through JAX backend `gpu`,
- `Qwen/Qwen3.5-0.8B`, real weights,
- BF16 checkpoint values with FP32 activation math: `dtype=float32`, `weight_dtype=bfloat16`,
- JIT execution,
- correctness checks enabled.

Do not run JAX CPU-only paths for the current GPU validation workflow. CPU numbers are not current correctness or MTP throughput evidence.

Older TPU benchmark findings are retained below for history only. They are not current GPU serving evidence.

## Required benchmark rules

A result is reportable only if it states:

- model id and dtype,
- hardware and backend,
- JAX execution mode,
- prompt workload,
- batch size and prompt length,
- generated token limit,
- speculative token count,
- MTP policy/env flags when used,
- warmup behavior,
- correctness result,
- baseline decode tokens/sec,
- MTP decode tokens/sec,
- acceptance rate.

## GPU runtime environment

Root GPU runtime state under `/mountpoint/.exp`, including the persistent JAX compilation cache:

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
export XLA_FLAGS=--xla_gpu_autotune_level=0
```

Profiler and journal output should stay under the same root:

- JAX profiler traces: `/mountpoint/.exp/profiles/<run-id>/`
- run journal: `/mountpoint/.exp/run_logs/run_journal.jsonl`
- vLLM cache root: `/mountpoint/.exp/.cache/vllm`

`RunRecorder` creates run IDs as `<utc timestamp>-<pid>-<run label>`, writes a `run_manifest.json` inside the profile directory, and appends structured JSONL events to the run journal. Normal profiled runs emit `start`, `profile_start`, zero or more `issue` rows, `profile_stop`, and `finish`. Use `--run-label` to make artifacts searchable; use `--profile-dir` and `--run-log` only when the replacement paths still live under `/mountpoint/.exp`.

## Warmup

JAX compile time must be excluded from steady-state decode throughput. Use warmup for every reported shape.

Valid warmup expectations:

- compile happens before measured decode timing,
- baseline and MTP paths both warm their JIT shapes,
- repeated measurements reuse the same process where practical,
- first-run compile latency is reported separately if relevant.

## Valid results

A result can be used for serving decisions only when:

- real checkpoint weights are loaded,
- GPU Qwen3.5 runs that claim the current correctness contract use `dtype=float32` and `weight_dtype=bfloat16`,
- target token correctness is true,
- next-step sanity is true when the benchmark supports it,
- baseline and MTP use comparable prompts and max tokens,
- decode tokens/sec excludes prefill and compile time,
- MTP enablement is based on measured emitted-token throughput, not acceptance alone.

## Invalid results

Do not use results for serving decisions when they have any of these properties:

- random or synthetic model weights,
- CPU-only throughput numbers,
- compile time included in decode throughput without disclosure,
- missing baseline,
- missing correctness checks,
- unsafe fused MTP flags used as if they were the serving path,
- acceptance rate used as a substitute for measured decode speedup.

## GPU server-shape benchmark

Use `benchmark_server_shapes.py` for the current HF vs JAX paged vs MTP1 check:

```bash
uv run python benchmark_server_shapes.py \
  --model Qwen/Qwen3.5-0.8B \
  --backend gpu \
  --jax-execution jit \
  --dtype float32 \
  --weight-dtype bfloat16 \
  --max-kv-cache-mb 1024 \
  --num-kvcache-blocks 64 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --prefill-buckets 16,32,64,128 \
  --batch-size-buckets 1,4 \
  --output-json results/qwen08_server_shape_current_all_mtp.json \
  --run-label server_shape_current_all_mtp
```

The script loads HF and the JAX engine in one process, runs each scenario through HF, JAX paged attention, and MTP1, then records generated-token parity, prefill top-k parity, throughput, decode throughput, and speculative stats.

Current GPU artifact: `results/qwen08_server_shape_current_all_mtp.json` (`server_shape_current_all_mtp`, 2026-05-25). It wrote the benchmark JSON and issue rows successfully; the process then exited with code 137 during profiler finalization, before the final `profile_stop`/`finish` journal rows. That shutdown issue is recorded in the run journal.

| scenario | JAX paged vs HF | MTP1 vs JAX/HF | throughput note |
| --- | --- | --- | --- |
| `single_16x16` | generated tokens and ordered top-5 match | generated tokens match | MTP1 is slower than JAX paged (`0.790x`, 18.39 vs 23.27 tok/s). |
| `single_64x24` | generated tokens and ordered top-5 match | generated tokens match | MTP1 is slower than JAX paged (`0.804x`, 17.72 vs 22.06 tok/s). |
| `single_128x24` | generated tokens and ordered top-5 match | generated tokens match | MTP1 is slower than JAX paged (`0.744x`, 15.59 vs 20.96 tok/s). |
| `batch4_mixed_16_32_64_128x24` | generated tokens and ordered top-5 match | generated tokens match | MTP1 is slower than JAX paged (`0.502x`, 44.81 vs 89.30 tok/s). |

Interpretation for current GPU work:

- JAX paged attention is the current GPU serving baseline for the server-shape suite.
- MTP1 is a correctness and benchmark target, not a current GPU speedup claim.
- MTP1 generated-token parity currently holds on the server-shape suite, but throughput is below the JAX paged baseline. Do not report MTP1 as a serving speedup until it beats that baseline on the same shapes.

## Focused MTP1 benchmark

Use `benchmark_mtp1_engine.py` when isolating speculative decoding behavior, next-step sanity, and acceptance accounting:

```bash
uv run python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-0.8B \
  --backend gpu \
  --jax-execution jit \
  --dtype float32 \
  --weight-dtype bfloat16 \
  --batch-prompts 1 \
  --prompt-lengths 64 \
  --max-tokens 24 \
  --max-kv-cache-mb 512 \
  --num-kvcache-blocks 32 \
  --max-num-seqs 1 \
  --prefill-buckets 64 \
  --batch-size-buckets 1 \
  --max-blocks-per-seq 8 \
  --warmup \
  --step-profile \
  --trace-steps \
  --check-next-step-sanity \
  --output-json results/qwen08_mtp1_current_single64.json \
  --run-label mtp1_current_single64
```

Keep `NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=0` for correctness checks. The exact commit-select verifier is the current K=1 correctness reference.

## vLLM GPU comparison

Use `benchmarks/benchmark_vllm_qwen35.py` from the isolated vLLM environment when comparing against vLLM. The async execution mode records true client-observed per-token stream timing without running a separate HTTP server:

```bash
/mountpoint/.exp/vllm-venv/bin/python benchmarks/benchmark_vllm_qwen35.py \
  --model Qwen/Qwen3.5-0.8B \
  --mode baseline \
  --execution async \
  --dtype bfloat16 \
  --input-lens 16,32,64,128 \
  --output-len 24 \
  --prompt-suite server_shapes \
  --max-model-len 256 \
  --gpu-memory-utilization 0.55 \
  --top-k 5 \
  --reference-json results/qwen08_server_shape_current_all_mtp.json \
  --output-json results/qwen08_vllm_baseline_server_shapes_batch4x24.json
```

For vLLM MTP1, add:

```bash
--mode mtp --speculative-method mtp --num-speculative-tokens 1
```

`--execution offline` uses vLLM `LLM.generate()` and records output tokens, first-step top-k logprobs, and offline tokens/sec. It cannot report true ITL, so `itl_ms_mean`, `itl_ms_p50`, and `itl_ms_p95` are intentionally `null` in offline artifacts.

`--execution async` uses `AsyncLLMEngine.generate(..., output_kind=RequestOutputKind.DELTA)` and stores per-token events. If a speculative backend emits multiple tokens in one delta, the artifact records `delta_size` so the same-step tokens are explicit.

Use `benchmarks/benchmark_jax_server_trace.py` for the matching JAX server-path timing artifact:

```bash
uv run python benchmarks/benchmark_jax_server_trace.py \
  --model Qwen/Qwen3.5-0.8B \
  --backend gpu \
  --jax-execution jit \
  --dtype float32 \
  --weight-dtype bfloat16 \
  --input-lens 16,32,64,128 \
  --output-len 24 \
  --prompt-suite server_shapes \
  --max-kv-cache-mb 1024 \
  --num-kvcache-blocks 64 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --prefill-buckets 16,32,64,128 \
  --batch-size-buckets 1,2,4 \
  --max-blocks-per-seq 16 \
  --reference-json results/qwen08_server_shape_current_all_mtp.json \
  --output-json results/qwen08_jax_server_trace_batch4x24.json
```

Server endpoint timing can also be collected from Flask:

- `/v1/generate_trace` returns `results`, per-token `events`, TTFT/ITL stats, total tokens/sec, JIT cache count, and speculative counters.
- `/v1/generate_stream` emits server-sent events from `LLMEngine.iter_generate`; token events include request index, token id, text, elapsed time, and finished/done state.
- `max_tokens` may be one integer for all prompts or a list matching the number of prompts, which is enough for heterogeneous output lengths in mixed batches.

Current GPU artifacts:

| workload | implementation | generated-token check vs JAX reference | tok/s | TTFT p50 ms | ITL p50 ms | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `len16x8` | JAX server trace | prefix match | 27.64 | 49.25 | 33.82 | profiled under `/mountpoint/.exp/profiles`. |
| `len16x8` | vLLM async baseline | prefix match | 127.04 | 33.80 | 4.17 | top-1 matches; top-5 has 4/5 overlap due tie/order differences. |
| `batch4_16_32_64_128x24` | JAX server trace | full match | 69.12 | 339.75 | 44.46 | profiled server-path artifact. |
| `batch4_16_32_64_128x24` | vLLM async baseline | diverges on `len_64` at generated index 6 | 549.03 | 63.09 | 4.87 | first-step top-1 matches all rows; correctness issue is logged in the run journal. |
| `batch4_16_32_64_128x24` | vLLM offline MTP1 | same `len_64` divergence as vLLM baseline | 32.08 | n/a | n/a | slower than vLLM offline baseline; offline API has no true ITL. |

## Historical TPU result table

Preserved TPU findings from `docs/mtp_tpu_spot_findings_2026-05-09.md`; keep these for historical comparison only:

| workload | mode | baseline tok/s | MTP tok/s | speedup | acceptance | correctness |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| synthetic low-acceptance | K=1 MTP | 62.56 | 62.28 | 0.996x | 50.0% | true |
| manual counting high-acceptance | K=1 MTP | 61.09 | 67.52 | 1.105x | 72.2% | true |
| manual counting high-acceptance, per-bucket gate | K=1 MTP | 64.48 | 72.84 | 1.130x | 61.1% | true |
| mixed arrivals B=4 | K=1 MTP | 104.33 | 52.41 | 0.502x | 36.4% | true |
| interleaved B=4 after forced-reject probes | K=1 MTP | 95.25 | 89.66 | 0.941x | 0.0% | true |

Interpretation:

- K=1 can be neutral or faster depending on workload and measured latency.
- Acceptance alone is insufficient; serving gates must use measured decode throughput.
- Mixed/heterogeneous B=4 is still below baseline; the latest forced-reject probe patch reduces fallback-heavy overhead but does not create speedup when acceptance is low.
- K=2 is correctness-clean but slower in observed benchmarks, so it remains experimental/non-serving.

## Where to put detailed logs

Use dated findings files for full logs and one-off investigations. Keep `docs/benchmarks.md` to rules, criteria, and the current compact result table.
