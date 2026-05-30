# GPU Matrix Report

- created_at_utc: `20260529_180252`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260529_180252`
- output_json: `results/gpu_matrix_20260529_cuda_fla_chunk32_fp32_r1.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.553x (target 0.900x)
- JAX tok/s: 64.31
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 40.43

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 64.31 | 116.37 | 0.553x | 104.74 | 40.43 | 78.02 | 0.824x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.03 s | 0.96 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 53.26 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 53.22 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 33.55 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 20.66 ms | 253.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.47 ms | 237.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 12.75 ms | 208.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 11.60 ms | 194.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.62 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void (anonymous namespace)::Fp32GdnPrefillChunk32Kernel<32, true>(float const*, float const*, float const*, float const*, float const*, int const*, float const*, float*, float*, long, long, long, long, long) | 446.40 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.19 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 993.31 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 993.25 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 992.38 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 899.89 ms | 144 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $scheduler.py:151 schedule | 876.56 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 33.55 ms | 427.55 ms | -394.01 ms | 0.078x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 53.22 ms | 289.24 ms | -236.01 ms | 0.184x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 53.26 ms | 280.56 ms | -227.30 ms | 0.190x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 12.75 ms | 229.21 ms | -216.46 ms | 0.056x | 208.0 | 1936 | -1728.0 |
| transpose | 20.66 ms | 47.30 ms | -26.64 ms | 0.437x | 253.0 | 312 | -59.0 |
| MemcpyD2D | 17.68 ms | 30.39 ms | -12.70 ms | 0.582x | 456.0 | 655 | -199.0 |
| command_buffer::update | 11.60 ms | 10.46 ms | 1.14 ms | 1.109x | 194.0 | 195 | -1.0 |
| gather | 15.30 ms | 14.55 ms | 0.74 ms | 1.051x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260529_cuda_fla_chunk32_fp32_r1.json`
- report: `results/gpu_matrix_20260529_cuda_fla_chunk32_fp32_r1.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.553x
- JAX/reference: 0.824x
- TTFT delta vs reference: -553.42 ms
- ITL delta vs reference: -5.83 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 33.55 ms, reference 427.55 ms, delta -394.01 ms, ratio 0.078x, count delta 48.0
- `PjRtCApiLoadedExecutable::Execute`: current 53.22 ms, reference 289.24 ms, delta -236.01 ms, ratio 0.184x, count delta -81.0
- `forward_step_token_ids_jit`: current 53.26 ms, reference 280.56 ms, delta -227.30 ms, ratio 0.190x, count delta 0.0
- `command_buffer::execute`: current 12.75 ms, reference 229.21 ms, delta -216.46 ms, ratio 0.056x, count delta -1728.0
- `transpose`: current 20.66 ms, reference 47.30 ms, delta -26.64 ms, ratio 0.437x, count delta -59.0
- `MemcpyD2D`: current 17.68 ms, reference 30.39 ms, delta -12.70 ms, ratio 0.582x, count delta -199.0
- `command_buffer::update`: current 11.60 ms, reference 10.46 ms, delta 1.14 ms, ratio 1.109x, count delta -1.0
- `gather`: current 15.30 ms, reference 14.55 ms, delta 0.74 ms, ratio 1.051x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
