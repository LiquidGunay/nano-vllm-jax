# Benchmark/Profile Strategy Crosscheck - 2026-06-01

Scope: Worker E benchmark/profile harness and external strategy crosscheck. This
note does not change model/runtime kernels.

## Current Evidence

Decode anchor:

| artifact | workload/config | exact tokens | tok/s | vLLM tok/s | ratio | target tok/s | gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/gpu_matrix_decode_static_metadata_r3_20260601.json` | `decode_heavy_128x128/gpu_paged_gdn_fla_decode_static_metadata` | yes | 165.81 | 213.54 | 0.776x | 192.18 | 26.38 |
| `results/gpu_matrix_decode_padded_gemm_current_control_r3_20260601.json` | `decode_heavy_128x128/gpu_paged_gdn_fla_decode_padded_gemm` | yes | 161.72 | 213.54 | 0.757x | 192.18 | 30.46 |
| `results/gpu_matrix_decode_padded_gemm_r3_20260601.json` | `decode_heavy_128x128/gpu_paged_gdn_fla_decode_padded_gemm` | yes | 162.10 | 213.54 | 0.759x | 192.18 | 30.09 |
| `results/gpu_matrix_decode_pallas_reductions_r3_20260601.json` | `decode_heavy_128x128/gpu_paged_gdn_fla_decode_pallas_reductions` | yes | 157.01 | 213.54 | 0.735x | 192.18 | 35.18 |

Padded GEMM moved the strict decode anchor by +5.09 tok/s over the prior
Pallas-reduction anchor. Static decode metadata then moved the current
three-repeat median to `165.81 tok/s`, but the remaining target still requires
roughly another `26.38 tok/s`, or about `105 ms` less end-to-end time on the
128-token decode run.

Long-prefill evidence is split:

| artifact | workload/config | exact tokens | repeats | tok/s | vLLM tok/s | ratio | target tok/s | gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.json` | `long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv` | yes | 2 | 63.98 | 116.37 | 0.550x | 104.74 | 40.76 |
| `results/gpu_matrix_goal_long_prefill_decode_bf16qkv.json` | same | yes | 1 | 93.60 | 116.37 | 0.804x | 104.74 | 11.14 |
| `results/gpu_matrix_long_prefill_current_best_bf16_r1_20260601.json` | same | no | 1 | 23.81 | 116.37 | 0.205x | 104.74 | 80.93 |

Only the two-repeat strict artifact is speed-claim-ready for correctness/repeats.
The one-repeat 93.60 tok/s artifact is the recovery target to reproduce under
the current harness.

## External Strategy Surface

- Upstream nano-vLLM advertises prefix caching, tensor parallelism, Torch
  compilation, and CUDA graphs as its small-codebase optimization surface, and
  its Qwen3 model uses merged QKV and gate/up projections:
  [README](https://github.com/GeeeekExplorer/nano-vllm),
  [model_runner.py](https://raw.githubusercontent.com/GeeeekExplorer/nano-vllm/main/nanovllm/engine/model_runner.py),
  [qwen3.py](https://raw.githubusercontent.com/GeeeekExplorer/nano-vllm/main/nanovllm/models/qwen3.py).
- Upstream nano-vLLM attention uses slot-mapped KV writes plus FlashAttention
  varlen prefill and `flash_attn_with_kvcache` decode:
  [attention.py](https://raw.githubusercontent.com/GeeeekExplorer/nano-vllm/main/nanovllm/layers/attention.py).
- vLLM's public strategy surface is PagedAttention, continuous batching/chunked
  prefill, prefix caching, piecewise/full CUDA graphs, FlashAttention/FlashInfer/
  Triton attention, optimized GEMM/MoE kernels, quantization, and torch.compile:
  [vLLM README](https://github.com/vllm-project/vllm).
- vLLM's current CUDA graph design distinguishes full, piecewise, decode-only,
  and backend-limited graph modes; attention backends advertise graph support,
  with FlashInfer currently described as single-token decode graph compatible:
  [CUDA graphs design](https://vllm.website.cncfstack.com/design/cuda_graphs/).
- vLLM's PagedAttention design keeps KV in paged blocks and depends on a custom
  memory layout/read path in the attention kernel:
  [PagedAttention design](https://docs.vllm.ai/en/v0.21.0/design/paged_attention/).

## Prioritized Checklist

1. Reduce per-step decode projection/reduction kernel cost before more prefill work.
   Why: padded GEMM improved integrated decode by 5.09 tok/s, and static
   metadata improved the current measured decode best to 165.81 tok/s. The
   remaining decode profile is still dominated by repeated GPU projection/
   reduction events:
   `gemm_fusion_dot_199` 126.43 ms/127, `gemm_fusion_dot_175` 101.47 ms/3066,
   `gemm_fusion_dot_general_337` 64.71 ms/2286, `gemm_fusion_dot_200`
   51.45 ms/3048, plus `_gdn_fla_chunk_fwd_o_packed_kernel` 50.10 ms/18.
   Measurement: rerun `decode_heavy_128x128` with 3 repeats, exact-token checks,
   and scoped profiles. A useful change must close at least 26 tok/s or reduce
   aggregate wall time by about 105 ms, with visible drops in those top GPU
   events rather than only microbench wins.

2. Add static replay/CUDA-graph-style validation before accepting launch-count work.
   Why: the padded-GEMM anchor still has one JIT forward per scheduler step, but
   many command-buffer subevents per step: `PjRt Execute` 2.21 count/step,
   `command_buffer::execute` 19.09 count/step, and `command_buffer::update`
   18.70 count/step. This is the clearest harness-side proxy for whether a
   static replay design is actually reducing host launch/update churn.
   Measurement: use the new `Host Replay Diagnostics` report section from
   `benchmarks/summarize_gpu_matrix.py`; compare count/step and ms/step for
   `PjRtCApiLoadedExecutable::Execute`, `command_buffer::execute`, and
   `command_buffer::update` against the same exact-token decode workload.

3. Reproduce the 93.60 tok/s long-prefill artifact under current strict harness.
   Why: current strict long prefill is 63.98 tok/s, but an older one-repeat run
   hit 93.60 tok/s with exact tokens. That difference is large enough to change
   the long-prefill plan: 93.60 is only 11.14 tok/s short of target, while 63.98
   is 40.76 tok/s short.
   Measurement: rerun the exact `long_prefill_512_2048` config with 3 repeats,
   current Python/env, stored vLLM reference, exact generated-token match, TTFT/
   ITL, scheduler diagnostics, and scoped top CPU/GPU events. Promote only if
   median throughput is reproduced and speed-claim-ready.

4. Keep the FLA/GDN prefill-kernel path on a correctness-gated leash.
   Why: the non-ready 23.81 tok/s artifact has exact-token failure and is
   dominated by `_gdn_fla_chunk_fwd_o_packed_kernel` 928.94 ms,
   `_gdn_fla_chunk_delta_h_packed_kernel` 854.29 ms, and
   `_gdn_fla_chunk_scaled_dot_kkt_packed_kernel` 306.00 ms. The strict CUDA
   artifact replaces that with `Fp32GdnPrefillChunk32Kernel` around 444 ms but
   still lands at 63.98 tok/s.
   Measurement: for every GDN prefill candidate, require exact generated-token
   match, 2+ repeats, and a top-event table showing whether the GDN prefill
   kernel family drops below roughly 250 ms total. If it does not, it cannot
   recover long prefill to 0.9x by itself.

5. Track scheduler/device-put movement explicitly for long prefill.
   Why: long-prefill scoped top CPU events show scheduler/build/device-put time
   at 869 ms in the strict 63.98 tok/s run and 2547 ms in the non-ready
   23.81 tok/s run, while the existing required profile needles emphasize PJRT
   and command buffers. This is a high-risk blind spot for prefill recovery.
   Measurement: for long-prefill profiles, record top CPU events and compare
   `$scheduler.py:151 schedule`, `$scheduler.py:303 build_scheduled_batch`,
   `$scheduler.py:16 _device_int32_arrays`, and `$api.py:2106 device_put`.
   Recovery requires those combined scheduler/device-put events to stay near the
   93.60 tok/s artifact's profile, not the 63.98/23.81 tok/s profiles.

6. Do not spend target-seeking time on rejected Triton matvec/reduction variants.
   Why: integrated decode results show `gpu_matrix_decode_triton_reductions...`
   at 113.77 tok/s and `gpu_matrix_decode_triton_matvec...` at 82.89 tok/s,
   far below both Pallas reductions and padded GEMM. They are useful only as
   negative evidence unless their kernel family changes materially.
   Measurement: any resurrection must first beat 157.01 tok/s with exact tokens
   on the same 3-repeat decode harness before it enters the target queue.

## Harness Change Made

`benchmarks/summarize_gpu_matrix.py` now emits `Host Replay Diagnostics`, which
normalizes key host buckets by scheduler step and includes reference-normalized
counts when the summary has JAX-reference deltas. This exposes bucket movement
that was previously hidden in total counts and helps compare static replay
experiments against ordinary JAX command-buffer churn.

Validated with:

```bash
python3 -m pytest tests/test_gpu_matrix_summary_report.py tests/test_gpu_matrix_runner.py
```
