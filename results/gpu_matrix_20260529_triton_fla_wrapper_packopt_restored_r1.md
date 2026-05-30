# GPU Matrix Report

- created_at_utc: `20260529_170437`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260529_170437`
- output_json: `results/gpu_matrix_20260529_triton_fla_wrapper_packopt_restored_r1.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.762x (target 0.900x)
- JAX tok/s: 88.66
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 16.07

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 88.66 | 116.37 | 0.762x | 104.74 | 16.07 | 78.02 | 1.136x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.25 s | 0.47 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 316.99 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 315.30 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 234.89 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 73.10 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 28.29 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 22.34 ms | 189.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.93 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 15.46 ms | 219.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.30 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.52 ms | 17 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.64 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.82 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 718.59 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 718.53 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 717.21 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 630.97 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 343.39 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 28.29 ms | 427.55 ms | -399.26 ms | 0.066x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 316.99 ms | 280.56 ms | 36.43 ms | 1.130x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 315.30 ms | 289.24 ms | 26.07 ms | 1.090x | 59.0 | 140 | -81.0 |
| transpose | 73.10 ms | 47.30 ms | 25.80 ms | 1.545x | 1547.0 | 312 | 1235.0 |
| command_buffer::update | 22.34 ms | 10.46 ms | 11.87 ms | 2.135x | 189.0 | 195 | -6.0 |
| command_buffer::execute | 234.89 ms | 229.21 ms | 5.68 ms | 1.025x | 1954.0 | 1936 | 18.0 |
| MemcpyD2D | 31.39 ms | 30.39 ms | 1.00 ms | 1.033x | 493.0 | 655 | -162.0 |
| gather | 15.13 ms | 14.55 ms | 0.57 ms | 1.039x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_packopt_restored_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_packopt_restored_r1.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.762x
- JAX/reference: 1.136x
- TTFT delta vs reference: -332.52 ms
- ITL delta vs reference: -5.85 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 28.29 ms, reference 427.55 ms, delta -399.26 ms, ratio 0.066x, count delta 48.0
- `forward_step_token_ids_jit`: current 316.99 ms, reference 280.56 ms, delta 36.43 ms, ratio 1.130x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 315.30 ms, reference 289.24 ms, delta 26.07 ms, ratio 1.090x, count delta -81.0
- `transpose`: current 73.10 ms, reference 47.30 ms, delta 25.80 ms, ratio 1.545x, count delta 1235.0
- `command_buffer::update`: current 22.34 ms, reference 10.46 ms, delta 11.87 ms, ratio 2.135x, count delta -6.0
- `command_buffer::execute`: current 234.89 ms, reference 229.21 ms, delta 5.68 ms, ratio 1.025x, count delta 18.0
- `MemcpyD2D`: current 31.39 ms, reference 30.39 ms, delta 1.00 ms, ratio 1.033x, count delta -162.0
- `gather`: current 15.13 ms, reference 14.55 ms, delta 0.57 ms, ratio 1.039x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
