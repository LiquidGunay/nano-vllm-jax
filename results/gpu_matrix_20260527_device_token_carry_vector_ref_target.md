# GPU Matrix Report

- created_at_utc: `20260527_075853`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_075853`
- output_json: `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.818x (target 0.900x)
- JAX tok/s: 95.14
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 9.59

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 95.14 | 116.37 | 0.818x | 104.74 | 9.59 | 78.02 | 1.220x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.24 s | 0.43 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 307.17 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 295.14 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 251.24 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.06 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 27.66 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.95 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 11.54 ms | 22.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 8.33 ms | 189.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.66 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 674.76 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 674.69 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 673.82 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 590.98 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 329.07 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.47 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.94 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.63 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:373 generate_with_trace | 670.57 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 670.51 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:142 step | 669.68 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | PjitFunction(compiled) | 585.68 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $profiler.py:424 wrapper | 326.77 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 27.66 ms | 427.55 ms | -399.89 ms | 0.065x | 64.0 | 16 | 48.0 |
| command_buffer::execute | 251.24 ms | 229.21 ms | 22.03 ms | 1.096x | 1936.0 | 1936 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 307.17 ms | 289.24 ms | 17.93 ms | 1.062x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 295.14 ms | 280.56 ms | 14.58 ms | 1.052x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.62 ms | 30.39 ms | -11.76 ms | 0.613x | 493.0 | 655 | -162.0 |
| transpose | 45.06 ms | 47.30 ms | -2.24 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.33 ms | 10.46 ms | -2.13 ms | 0.796x | 189.0 | 195 | -6.0 |
| gather | 16.20 ms | 14.55 ms | 1.65 ms | 1.113x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.818x
- JAX/reference: 1.220x
- TTFT delta vs reference: -340.65 ms
- ITL delta vs reference: -7.17 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 27.66 ms, reference 427.55 ms, delta -399.89 ms, ratio 0.065x, count delta 48.0
- `command_buffer::execute`: current 251.24 ms, reference 229.21 ms, delta 22.03 ms, ratio 1.096x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 307.17 ms, reference 289.24 ms, delta 17.93 ms, ratio 1.062x, count delta -81.0
- `forward_step_token_ids_jit`: current 295.14 ms, reference 280.56 ms, delta 14.58 ms, ratio 1.052x, count delta 0.0
- `MemcpyD2D`: current 18.62 ms, reference 30.39 ms, delta -11.76 ms, ratio 0.613x, count delta -162.0
- `transpose`: current 45.06 ms, reference 47.30 ms, delta -2.24 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.33 ms, reference 10.46 ms, delta -2.13 ms, ratio 0.796x, count delta -6.0
- `gather`: current 16.20 ms, reference 14.55 ms, delta 1.65 ms, ratio 1.113x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
