# GPU Matrix Report

- created_at_utc: `20260527_064813`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_064813`
- output_json: `results/gpu_matrix_20260527_device_token_carry_vector_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.799x (target 0.900x)
- JAX tok/s: 93.01
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 11.72

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 93.01 | 116.37 | 0.799x | 104.74 | 11.72 | 78.02 | 1.192x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 360.71 ms | 255.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 273.08 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 223.11 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.06 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 29.65 ms | 347.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 16.12 ms | 402.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.83 ms | 22.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 10.05 ms | 195.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.44 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.89 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.69 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.03 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 685.33 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 685.28 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 684.33 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 622.45 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 621.83 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.63 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:373 generate_with_trace | 690.85 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 690.79 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:142 step | 689.88 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:3703 run | 625.97 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:1865 _run_main_and_sample | 625.32 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 0.58 ms | 427.55 ms | -426.97 ms | 0.001x | 4.0 | 16 | -12.0 |
| PjRtCApiLoadedExecutable::Execute | 360.71 ms | 289.24 ms | 71.47 ms | 1.247x | 255.0 | 140 | 115.0 |
| MemcpyD2D | 45.77 ms | 30.39 ms | 15.38 ms | 1.506x | 749.0 | 655 | 94.0 |
| forward_step_token_ids_jit | 273.08 ms | 280.56 ms | -7.47 ms | 0.973x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 223.11 ms | 229.21 ms | -6.10 ms | 0.973x | 1936.0 | 1936 | 0.0 |
| transpose | 45.06 ms | 47.30 ms | -2.24 ms | 0.953x | 312.0 | 312 | 0.0 |
| gather | 15.50 ms | 14.55 ms | 0.94 ms | 1.065x | 103.0 | 103 | 0.0 |
| command_buffer::update | 10.05 ms | 10.46 ms | -0.41 ms | 0.961x | 195.0 | 195 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_vector_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_vector_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.799x
- JAX/reference: 1.192x
- TTFT delta vs reference: -325.85 ms
- ITL delta vs reference: -7.17 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 0.58 ms, reference 427.55 ms, delta -426.97 ms, ratio 0.001x, count delta -12.0
- `PjRtCApiLoadedExecutable::Execute`: current 360.71 ms, reference 289.24 ms, delta 71.47 ms, ratio 1.247x, count delta 115.0
- `MemcpyD2D`: current 45.77 ms, reference 30.39 ms, delta 15.38 ms, ratio 1.506x, count delta 94.0
- `forward_step_token_ids_jit`: current 273.08 ms, reference 280.56 ms, delta -7.47 ms, ratio 0.973x, count delta 0.0
- `command_buffer::execute`: current 223.11 ms, reference 229.21 ms, delta -6.10 ms, ratio 0.973x, count delta 0.0
- `transpose`: current 45.06 ms, reference 47.30 ms, delta -2.24 ms, ratio 0.953x, count delta 0.0
- `gather`: current 15.50 ms, reference 14.55 ms, delta 0.94 ms, ratio 1.065x, count delta 0.0
- `command_buffer::update`: current 10.05 ms, reference 10.46 ms, delta -0.41 ms, ratio 0.961x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
