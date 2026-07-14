# Benchmark

This repository makes one deliberately small performance claim:

> On an NVIDIA A10G, Nano-VLLM-JAX reaches 53.98 decode tok/s on the pinned
> Qwen3.5-4B B=1 workload, 1.075x vLLM 0.25.0's 50.23 decode tok/s. Neither
> backend uses speculative decoding.

The executable contract is [benchmark.json](../benchmarks/benchmark.json). It fixes the
model revision, BF16 weights and activations, one 64-token ID prompt, 64 output
tokens, greedy sampling, ignored EOS, and a prefix-cache miss. Initialization,
model download, compilation, graph capture, and one control request are outside
the measured window. TTFT is reported separately. Decode throughput is the 63
tokens after the first token divided by time from first token to completion.

Each backend runs three measured repeats in its own process and environment. A
result is invalid if repeats change tokens, a new JAX executor route-cache entry
appears during measurement, JAX and vLLM differ on any output token, or
throughput spread exceeds 10%. Hardware, software, and repository state are
recorded as provenance; they do not make a correctly executed benchmark
invalid. Raw JSON results remain outside the repository.

## Run

```bash
./scripts/run_benchmark.sh both
```

Run only the required side with `jax` or `vllm`; the vLLM side requires an
existing JAX result for exact-token comparison. When both run, the script also
writes `comparison.json` and prints the observed ratio; it does not enforce the
historical ratio. The script creates isolated Python 3.11 environments from
[uv.lock](../uv.lock) and
[vllm-requirements.txt](../benchmarks/vllm-requirements.txt). It checks one
explicit CUDA GPU and runs each backend through the process-tree RAM guard. The
default 80% system-RAM ceiling is paired with a 10 GiB process-tree limit and a
2 GiB available-memory floor. Set `NANO_VLLM_JAX_MAX_SYSTEM_RAM_PERCENT=70` for
a stricter host.

All environments, caches, and outputs derive from
`NANO_VLLM_JAX_BENCHMARK_ROOT`, which defaults to `/mountpoint/.exp`. Select a
physical GPU index or UUID with `NANO_VLLM_JAX_BENCHMARK_GPU`; it defaults to
`CUDA_VISIBLE_DEVICES`, then GPU 0.

The vLLM environment removes its optional TorchCodec package: vLLM's text path
does not use it, while importing it requires host FFmpeg video libraries. This
keeps the optional baseline directory-scoped and avoids changing the server.
The adapter also uses vLLM's `language_model_only` mode, matching the JAX
engine's use of the dense text model without allocating a vision encoder cache.
FlashInfer sampling is disabled because its optional JIT extension is
incompatible with this host CUDA toolkit; greedy decoding uses vLLM's native
argmax path. Other vLLM model-execution and compilation defaults are unchanged.

## Recorded Result

These values come from the committed
[recorded result](../benchmarks/recorded_result.json), measured at implementation
commit `d8c50ab` on 2026-07-14. Both backends produced the same tokens on every
repeat and the same output hash,
`352a694746b191d9b1cbb50e43499729b3fc3903048745cad7d79b8b8d85ca5a`.

| Backend | Version | Median decode tok/s | Median TTFT | Spread | Output parity |
| --- | --- | ---: | ---: | ---: | --- |
| Nano-VLLM-JAX | JAX 0.10.0 | 53.98 | 122.8 ms | 0.059% | exact |
| vLLM | 0.25.0 | 50.23 | 54.5 ms | 0.065% | exact |

The GPU was an NVIDIA A10G (22.5 GiB) with driver 580.159.03. JAX added no
executor route-cache entries during the measured repeats. The RAM guard
observed peak process-tree RSS of 2.74 GiB for JAX and 4.61 GiB for vLLM. The
largest sampled device use was 17.39 GiB and 8.89 GiB respectively; these
sparse observations are not execution peaks. Raw result and guard JSON remain
outside the repository under
`/mountpoint/.exp/artifacts/nano-vllm-jax/benchmark/results`.

This is a steady-state decode claim, not a TTFT or end-to-end latency claim.
JAX's TTFT was 2.25x vLLM's in this run. The result also does not claim an MTP
win; it establishes the non-speculative artifact contract that later MTP work
must beat without changing the workload.
