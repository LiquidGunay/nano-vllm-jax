# GPU Matrix Report

- created_at_utc: `20260527_075527`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_075527`
- output_json: `results/gpu_matrix_20260527_device_token_carry_batched_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.800x (target 0.900x)
- JAX tok/s: 93.06
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 11.67

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 93.06 | 116.37 | 0.800x | 104.74 | 11.67 | 78.02 | 1.193x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.26 s | 0.43 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 360.39 ms | 255.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 273.44 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 224.22 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.08 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 26.98 ms | 347.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 16.12 ms | 402.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 10.32 ms | 195.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.04 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.90 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 687.74 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 687.68 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 686.66 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 620.67 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 619.91 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.61 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:373 generate_with_trace | 687.67 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 687.62 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:142 step | 686.67 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:3703 run | 621.36 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:1865 _run_main_and_sample | 620.74 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 0.53 ms | 427.55 ms | -427.03 ms | 0.001x | 4.0 | 16 | -12.0 |
| PjRtCApiLoadedExecutable::Execute | 360.39 ms | 289.24 ms | 71.16 ms | 1.246x | 255.0 | 140 | 115.0 |
| MemcpyD2D | 43.10 ms | 30.39 ms | 12.71 ms | 1.418x | 749.0 | 655 | 94.0 |
| forward_step_token_ids_jit | 273.44 ms | 280.56 ms | -7.12 ms | 0.975x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 224.22 ms | 229.21 ms | -4.99 ms | 0.978x | 1936.0 | 1936 | 0.0 |
| transpose | 45.08 ms | 47.30 ms | -2.22 ms | 0.953x | 312.0 | 312 | 0.0 |
| gather | 14.70 ms | 14.55 ms | 0.15 ms | 1.010x | 103.0 | 103 | 0.0 |
| command_buffer::update | 10.32 ms | 10.46 ms | -0.15 ms | 0.986x | 195.0 | 195 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_batched_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_batched_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.800x
- JAX/reference: 1.193x
- TTFT delta vs reference: -328.24 ms
- ITL delta vs reference: -7.18 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 0.53 ms, reference 427.55 ms, delta -427.03 ms, ratio 0.001x, count delta -12.0
- `PjRtCApiLoadedExecutable::Execute`: current 360.39 ms, reference 289.24 ms, delta 71.16 ms, ratio 1.246x, count delta 115.0
- `MemcpyD2D`: current 43.10 ms, reference 30.39 ms, delta 12.71 ms, ratio 1.418x, count delta 94.0
- `forward_step_token_ids_jit`: current 273.44 ms, reference 280.56 ms, delta -7.12 ms, ratio 0.975x, count delta 0.0
- `command_buffer::execute`: current 224.22 ms, reference 229.21 ms, delta -4.99 ms, ratio 0.978x, count delta 0.0
- `transpose`: current 45.08 ms, reference 47.30 ms, delta -2.22 ms, ratio 0.953x, count delta 0.0
- `gather`: current 14.70 ms, reference 14.55 ms, delta 0.15 ms, ratio 1.010x, count delta 0.0
- `command_buffer::update`: current 10.32 ms, reference 10.46 ms, delta -0.15 ms, ratio 0.986x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
