# Benchmark Claim

This repository makes one deliberately small performance claim:

> On the recorded GPU and software stack, Nano-VLLM-JAX serves the pinned
> Qwen3.5-4B checkpoint at the reported median B=1 greedy decode rate relative
> to vLLM without speculative decoding.

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

The result table is populated only from a clean, valid run of this exact
contract. Until then, the repository makes no numeric speed claim.

| Backend | Version | Median decode tok/s | Median TTFT | Spread | Output parity |
| --- | --- | ---: | ---: | ---: | --- |
| Nano-VLLM-JAX | pending | pending | pending | pending | pending |
| vLLM | 0.25.0 | pending | pending | pending | pending |

Hardware, driver, memory observations, repository commit, and output hash will
be recorded beside the populated table.
