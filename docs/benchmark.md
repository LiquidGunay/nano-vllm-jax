# Benchmark

This repository makes one deliberately small performance claim:

> On an NVIDIA A10G, persistent K=2 MTP raises Nano-VLLM-JAX from 53.99
> to 82.72 decode tok/s on the pinned Qwen3.5-4B B=1 workload. That is
> 1.532x its base route and 1.643x vLLM 0.25.1 without MTP. vLLM with
> K=2 MTP reaches 86.60 decode tok/s on the same workload.

The four routes produced one of two content-addressed outputs. They differ only
at output index 44, where BF16 rounding can select token 5129 or 8343. The
[parity evidence](../benchmarks/parity_evidence.json) records the full-vocabulary
KL/JS check and is itself pinned by [benchmark.json](../benchmarks/benchmark.json).
The evidence also pins a digest of the serving source tree, both benchmark
backend adapters, `uv.lock`, and the vLLM requirement, so a later runtime or
frozen-environment change cannot silently reuse the exception. That executable
contract pins the model revision, BF16 weights and activations, one committed
64-token natural-language prompt, 64 output tokens, greedy sampling, ignored
EOS, a prefix-cache miss, and two MTP draft positions.

For orientation, the committed token IDs decode to:

> Explain why speculative decoding can improve single-request inference
> latency, and describe the tradeoff between draft accuracy and verification
> cost. Discuss memory bandwidth, batch size, compilation overhead, and the
> verifier's packed execution. Give a concrete intuition for why fewer
> target-model steps can still lose in end-to-end serving. Use one example.

The token IDs, not this explanatory text, are the executable prompt contract.

Initialization, model download, compilation, graph capture, and one control
request are outside the measured window. TTFT is reported separately. Decode
throughput is the 63 tokens after the first token divided by time from first
token to completion.

Each route runs three measured repeats in its own process. A result is invalid
if repeats change tokens, output rows are not 64 tokens, output falls outside
the two named hashes, JAX adds an executor route-cache entry during measurement,
speculative counters are inconsistent, or throughput spread exceeds 10%.
This is a finite equivalence, not fuzzy token comparison: any other token,
position, row, or second mismatch fails.
The runner requires a clean checkout. Standalone routes also require their
JAX-base reference to have the same manifest digest and implementation commit,
and cross-framework ratios require one physical GPU.

## Run

```bash
./scripts/run_benchmark.sh both
```

Use `jax` or `vllm` to run both routes for one framework. The `jax-base`,
`jax-mtp`, `vllm-base`, and `vllm-mtp` targets run one route; MTP and vLLM
targets require an existing JAX-base result for parity. A complete run writes
four raw results plus `comparison.json` and prints observed ratios. It does not
enforce the historical result.

The script creates isolated Python 3.11 environments from
[uv.lock](../uv.lock) and
[vllm-requirements.txt](../benchmarks/vllm-requirements.txt). It checks one
explicit CUDA GPU and guards every backend process. The default 80% system-RAM
ceiling is paired with a 10 GiB process-tree limit and a 2 GiB available-memory
floor. Set `NANO_VLLM_JAX_MAX_SYSTEM_RAM_PERCENT=70` for a stricter host.

Environments, caches, and outputs derive from
`NANO_VLLM_JAX_BENCHMARK_ROOT`, which defaults to `/mountpoint/.exp`. Select a
physical GPU index or UUID with `NANO_VLLM_JAX_BENCHMARK_GPU`; it defaults to
`CUDA_VISIBLE_DEVICES`, then GPU 0.

JAX and vLLM use separate environments because their CUDA Python dependencies
conflict. The vLLM environment removes optional TorchCodec, uses text-only
model loading, resolves the already-cached pinned revision offline, and disables
the optional FlashInfer sampler extension; vLLM's model execution, MTP,
compilation, and CUDA-graph defaults remain enabled.
Draft and accepted-token counts come from vLLM's public metrics API. vLLM does
not expose the number of target positions evaluated, so that field remains
unreported rather than inferred.

## Recorded Result

These values come from the committed
[recorded result](../benchmarks/recorded_result.json), measured at implementation
commit `9d6e890` on 2026-07-18. The two admitted output hashes are
`9d0a61c9...2c32bde` and `7823dab0...5b163c`.

| Framework | Route | Decode tok/s | Own-base ratio | vLLM-base ratio | Median TTFT | Accepted / drafted | Parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Nano-VLLM-JAX | base | 53.99 | 1.000x | 1.073x | 124.2 ms | — | reference |
| Nano-VLLM-JAX | MTP K=2 | 82.72 | 1.532x | 1.643x | 129.9 ms | 36 / 50 | near-tie variant |
| vLLM 0.25.1 | base | 50.34 | 1.000x | 1.000x | 51.9 ms | — | near-tie variant |
| vLLM 0.25.1 | MTP K=2 | 86.60 | 1.720x | 1.720x | 58.5 ms | 36 / 51 | reference |

JAX MTP is 0.955x vLLM MTP on this lane. JAX evaluated 75 target positions
per repeat and added no executor route-cache entries during measurement. The
different drafted totals reflect different tail grouping; both accepted 36
draft tokens.

At the one mismatch, the measured bidirectional KL is at most
`1.50e-4`, JS is `3.73e-5`, and total variation is `0.0080`. The base
full-precision logits favor token 8343 by only `0.0796`; both candidates round
to `24.375` in its BF16 reduction. The packed distribution rounds them to
`24.375` and `24.25`. Fresh compiled processes can therefore choose either
named output without changing the accepted-prefix or state-commit contract.

The GPU was an NVIDIA A10G (22.5 GiB) with driver 580.159.03. The RAM guard
observed peak process-tree RSS of 2.80, 2.98, 6.22, and 6.10 GiB
for JAX base, JAX MTP, vLLM base, and vLLM MTP respectively. The corresponding
largest sampled device use was 17.39, 17.40, 8.89, and 9.10 GiB. These device
samples are not execution peaks. JAX's declared persistent allocation grows by
about 231 MiB when MTP is enabled. Peak guarded system use was 6.06 GiB.

This is a steady-state decode-throughput claim, not a TTFT or end-to-end
latency claim. Raw result and guard JSON stay outside the repository under
`/mountpoint/.exp/artifacts/nano-vllm-jax/benchmark/results`.

## Smaller Checkpoints

The same B=1/P64/O64/K2 contract was also run as a non-claim diagnostic on the
0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` and 2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. These were guarded three-repeat
fresh-process runs from clean implementation commit `f991b63` on the same
physical A10G and driver as the primary result, using JAX 0.10.0 and vLLM
0.25.1. Their raw
JSON remains external, so they are diagnostic context rather than additional
artifact claims:

| Model | JAX base | JAX MTP | JAX ratio | vLLM base | vLLM MTP | vLLM ratio | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B | 208.91 | 238.57 | 1.142x | 243.24 | 277.84* | 1.142x* | JAX exact; vLLM final token differs |
| Qwen3.5-2B | 113.50 | 149.74 | 1.319x | 112.06 | 167.06 | 1.491x | exact |
| Qwen3.5-4B | 53.95 | 83.94 | 1.556x | 50.32 | 86.82 | 1.725x | exact |

`*` The 0.8B vLLM-MTP timing is diagnostic only because its final token differed
from both base routes; no KL was captured, so it is not accepted as a parity
result.

The scaling is consistent with two costs. First, fixed launch, metadata, and
packed-boundary work consumes a larger share of a small target model. Second,
the 0.8B JAX run accepted 33 of 58 drafts and evaluated 87 target positions,
while 2B and 4B accepted 36 of 50 and evaluated 75. Lower acceptance therefore
creates more verifier groups exactly where framework overhead is already least
amortized. vLLM's CUDA-graph runtime has a clearer advantage on 0.8B; the JAX
gap narrows as target-model work grows.
