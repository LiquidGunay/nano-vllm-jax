# GPU Matrix Report

- created_at_utc: `20260527_064234`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_064234`
- output_json: `results/gpu_matrix_20260527_device_token_carry_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.768x (target 0.900x)
- JAX tok/s: 89.35
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 15.39

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 89.35 | 116.37 | 0.768x | 104.74 | 15.39 | 78.02 | 1.145x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.25 s | 0.46 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 559.19 ms | 375.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 270.12 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 222.99 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 68.82 ms | 57.5 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.05 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 16.31 ms | 492.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 9.96 ms | 194.5 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 6.02 ms | 437.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.59 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 746.88 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 746.83 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 745.97 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(stack) | 647.15 ms | 38 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjRtCApiLoadedExecutable::Execute | 556.49 ms | 375 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.95 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:373 generate_with_trace | 688.15 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 688.09 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:142 step | 687.05 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | PjRtCApiLoadedExecutable::Execute | 561.89 ms | 375 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | PJRT_LoadedExecutable_Execute | 559.57 ms | 375 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 0.94 ms | 427.55 ms | -426.61 ms | 0.002x | 4.0 | 16 | -12.0 |
| PjRtCApiLoadedExecutable::Execute | 559.19 ms | 289.24 ms | 269.95 ms | 1.933x | 375.0 | 140 | 235.0 |
| gather | 73.48 ms | 14.55 ms | 58.93 ms | 5.049x | 138.5 | 103 | 35.5 |
| forward_step_token_ids_jit | 270.12 ms | 280.56 ms | -10.44 ms | 0.963x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 22.32 ms | 30.39 ms | -8.06 ms | 0.735x | 929.0 | 655 | 274.0 |
| command_buffer::execute | 222.99 ms | 229.21 ms | -6.22 ms | 0.973x | 1936.0 | 1936 | 0.0 |
| transpose | 45.06 ms | 47.30 ms | -2.24 ms | 0.953x | 313.5 | 312 | 1.5 |
| command_buffer::update | 9.96 ms | 10.46 ms | -0.50 ms | 0.952x | 194.5 | 195 | -0.5 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.768x
- JAX/reference: 1.145x
- TTFT delta vs reference: -328.58 ms
- ITL delta vs reference: -7.17 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 0.94 ms, reference 427.55 ms, delta -426.61 ms, ratio 0.002x, count delta -12.0
- `PjRtCApiLoadedExecutable::Execute`: current 559.19 ms, reference 289.24 ms, delta 269.95 ms, ratio 1.933x, count delta 235.0
- `gather`: current 73.48 ms, reference 14.55 ms, delta 58.93 ms, ratio 5.049x, count delta 35.5
- `forward_step_token_ids_jit`: current 270.12 ms, reference 280.56 ms, delta -10.44 ms, ratio 0.963x, count delta 0.0
- `MemcpyD2D`: current 22.32 ms, reference 30.39 ms, delta -8.06 ms, ratio 0.735x, count delta 274.0
- `command_buffer::execute`: current 222.99 ms, reference 229.21 ms, delta -6.22 ms, ratio 0.973x, count delta 0.0
- `transpose`: current 45.06 ms, reference 47.30 ms, delta -2.24 ms, ratio 0.953x, count delta 1.5
- `command_buffer::update`: current 9.96 ms, reference 10.46 ms, delta -0.50 ms, ratio 0.952x, count delta -0.5
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
