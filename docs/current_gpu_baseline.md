# Current GPU Baseline

This is the current accepted non-speculative GPU serving baseline for the
optimization plan in `docs/gpu_optimization_next_goal_plan.md`.

## Current Accepted Default

- model: `Qwen/Qwen3.5-0.8B`
- hardware: `NVIDIA A10G`
  - provenance: the Entry 045 server artifact records `backend=gpu`; the
    standalone GDN CUDA artifact for the same tracked GPU target records
    `device_kind=NVIDIA A10G`.
- dtype contract: BF16 checkpoint weights, FP32 activation math
- platform: JAX CUDA
- baseline entry: Entry 045, `Accepted GDN Prefill Chunk Size 32`
- benchmark artifact:
  `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`
- profile directory:
  `/mountpoint/.exp/profiles/20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat`
- Perfetto trace:
  `/mountpoint/.exp/profiles/20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat/plugins/profile/2026_05_26_02_20_50/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane:
  `/mountpoint/.exp/profiles/20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat/plugins/profile/2026_05_26_02_20_50/INDCS0291.atrapa.deloitte.com.xplane.pb`

## Workload

- prompt suite: `mixed`
- input lengths: `[64,128,192,256,320,384,448,512]`
- output length: `32`
- generated tokens: `256`
- scheduler envelope:
  - `max_kv_cache_mb=3072`
  - `num_kvcache_blocks=256`
  - `max_num_batched_tokens=4096`
  - `max_num_seqs=8`
  - `max_blocks_per_seq=40`
  - `prefill_buckets=64,128,256,384,512`
  - `batch_size_buckets=1,2,4,8`

## Metrics

| metric | value |
| --- | ---: |
| throughput | `367.80 tok/s` |
| TTFT p50 | `289.98 ms` |
| ITL p50 | `13.14 ms` |
| ITL p95 | `13.59 ms` |

## vLLM Comparison

- vLLM artifact:
  `results/qwen08_vllm_async_delta_baseline_hetero8_64_512x32.json`
- vLLM async throughput: `864.18 tok/s`
- JAX/vLLM throughput ratio: `0.426x`
- target ratio for this plan: `>=0.75x` on long heterogeneous mixed-shape
  requests before moving to MTP speed work

## Enabled Accepted Fast Flags

The accepted fastest non-MTP default uses:

```text
NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH=1
NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD=1
NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV=1
NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z=1
NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ=1
NANO_VLLM_JAX_COMPACT_PREFILL_MLP=1
```

It does not use MTP and does not use FFI/custom kernels.

## Correctness Evidence

The Entry 045 artifact has `correctness.ok=true` and exact generated-token
matches for all 8 rows over all 32 generated tokens. Future performance claims
must keep exact generated-token parity against this artifact or a later accepted
baseline.
