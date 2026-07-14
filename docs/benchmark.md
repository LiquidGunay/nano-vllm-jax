# Benchmark Claim

This repository makes one deliberately small performance claim:

> On an NVIDIA A10G, Nano-VLLM-JAX reaches 53.98 decode tok/s on the pinned
> Qwen3.5-4B B=1 workload, 1.075x vLLM 0.25.0's 50.23 decode tok/s. Neither
> backend uses speculative decoding.

The executable contract is [claim.json](../benchmarks/claim.json). It fixes the
model revision, BF16 weights and activations, one 64-token ID prompt, 64 output
tokens, greedy sampling, ignored EOS, and a prefix-cache miss. Initialization,
model download, compilation, graph capture, and one control request are outside
the measured window. TTFT is reported separately. Decode throughput is the 63
tokens after the first token divided by time from first token to completion.

Each backend runs three measured repeats in its own process and environment. A
result is invalid if repeats change tokens, JAX compiles during measurement,
JAX and vLLM differ on any output token, or throughput spread exceeds 10%.
Raw JSON results remain under `/mountpoint/.exp`.

## Reproduce

```bash
./scripts/reproduce_claim.sh both
```

Run only the required side with `jax` or `vllm`; the vLLM side requires an
existing JAX result for exact-token comparison. The script creates isolated
environments from [uv.lock](../uv.lock) and
[vllm-requirements.txt](../benchmarks/vllm-requirements.txt). It checks one
visible CUDA GPU and runs each backend through the process-tree RAM guard. The
default 80% system-RAM ceiling is paired with a 10 GiB process-tree limit and a
2 GiB available-memory floor. Set `NANO_VLLM_JAX_MAX_SYSTEM_RAM_PERCENT=70` for
a stricter host.

The vLLM environment removes its optional TorchCodec package: vLLM's text path
does not use it, while importing it requires host FFmpeg video libraries. This
keeps the optional baseline directory-scoped and avoids changing the server.
The adapter also uses vLLM's `language_model_only` mode, matching the JAX
engine's use of the dense text model without allocating a vision encoder cache.
FlashInfer sampling is disabled because its optional JIT extension is
incompatible with this host CUDA toolkit; greedy decoding uses vLLM's native
argmax path, while model execution and CUDA graphs remain enabled.

## Recorded Result

These values come from a clean run at implementation commit `d8c50ab` on
2026-07-14. Both backends produced the same tokens on every repeat and the
same output hash, `352a694746b191d9b1cbb50e43499729b3fc3903048745cad7d79b8b8d85ca5a`.

| Backend | Version | Median decode tok/s | Median TTFT | Spread | Output parity |
| --- | --- | ---: | ---: | ---: | --- |
| Nano-VLLM-JAX | JAX 0.10.0 | 53.98 | 122.8 ms | 0.059% | exact |
| vLLM | 0.25.0 | 50.23 | 54.5 ms | 0.065% | exact |

The GPU was an NVIDIA A10G (22.5 GiB) with driver 580.159.03. JAX reported no
measured-phase JIT cache growth. The RAM guard observed peak process-tree RSS
of 2.74 GiB for JAX and 4.61 GiB for vLLM; peak device use was 17.39 GiB and
8.89 GiB respectively. Raw result and guard JSON remain outside the repository
under `/mountpoint/.exp/artifacts/nano-vllm-jax/claim/results`.

This is a steady-state decode claim, not a TTFT or end-to-end latency claim.
JAX's TTFT was 2.25x vLLM's in this run. The result also does not claim an MTP
win; it establishes the non-speculative artifact contract that later MTP work
must beat without changing the workload.
