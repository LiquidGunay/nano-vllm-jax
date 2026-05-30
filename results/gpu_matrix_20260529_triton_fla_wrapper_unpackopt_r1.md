# GPU Matrix Report

- created_at_utc: `20260529_165519`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260529_165519`
- output_json: `results/gpu_matrix_20260529_triton_fla_wrapper_unpackopt_r1.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.750x (target 0.900x)
- JAX tok/s: 87.28
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 17.46

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 87.28 | 116.37 | 0.750x | 104.74 | 17.46 | 78.02 | 1.119x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.27 s | 0.46 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 305.19 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 304.40 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 227.66 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 73.09 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 33.14 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.93 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 14.96 ms | 194.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 14.84 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.27 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.53 ms | 17 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.82 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 731.38 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 731.31 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 730.33 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 606.69 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 379.88 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 33.14 ms | 427.55 ms | -394.41 ms | 0.078x | 64.0 | 16 | 48.0 |
| transpose | 73.09 ms | 47.30 ms | 25.79 ms | 1.545x | 1547.0 | 312 | 1235.0 |
| forward_step_token_ids_jit | 304.40 ms | 280.56 ms | 23.85 ms | 1.085x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 305.19 ms | 289.24 ms | 15.96 ms | 1.055x | 59.0 | 140 | -81.0 |
| gather | 19.51 ms | 14.55 ms | 4.96 ms | 1.341x | 103.0 | 103 | 0.0 |
| command_buffer::update | 14.96 ms | 10.46 ms | 4.50 ms | 1.430x | 194.0 | 195 | -1.0 |
| MemcpyD2D | 27.86 ms | 30.39 ms | -2.52 ms | 0.917x | 493.0 | 655 | -162.0 |
| command_buffer::execute | 227.66 ms | 229.21 ms | -1.55 ms | 0.993x | 1954.0 | 1936 | 18.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_unpackopt_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_unpackopt_r1.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.750x
- JAX/reference: 1.119x
- TTFT delta vs reference: -316.24 ms
- ITL delta vs reference: -5.80 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 33.14 ms, reference 427.55 ms, delta -394.41 ms, ratio 0.078x, count delta 48.0
- `transpose`: current 73.09 ms, reference 47.30 ms, delta 25.79 ms, ratio 1.545x, count delta 1235.0
- `forward_step_token_ids_jit`: current 304.40 ms, reference 280.56 ms, delta 23.85 ms, ratio 1.085x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 305.19 ms, reference 289.24 ms, delta 15.96 ms, ratio 1.055x, count delta -81.0
- `gather`: current 19.51 ms, reference 14.55 ms, delta 4.96 ms, ratio 1.341x, count delta 0.0
- `command_buffer::update`: current 14.96 ms, reference 10.46 ms, delta 4.50 ms, ratio 1.430x, count delta -1.0
- `MemcpyD2D`: current 27.86 ms, reference 30.39 ms, delta -2.52 ms, ratio 0.917x, count delta -162.0
- `command_buffer::execute`: current 227.66 ms, reference 229.21 ms, delta -1.55 ms, ratio 0.993x, count delta 18.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
