# GPU Matrix Report

- created_at_utc: `20260527_142121`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_142121`
- output_json: `results/gpu_matrix_20260527_device_token_carry_direct_vector_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.823x (target 0.900x)
- JAX tok/s: 95.75
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 8.98

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 95.75 | 116.37 | 0.823x | 104.74 | 8.98 | 78.02 | 1.227x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 305.38 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 294.39 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 251.11 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.07 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 27.44 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.94 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 9.99 ms | 22.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 8.37 ms | 184.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.93 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.59 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 668.40 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 668.34 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 667.54 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 586.75 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:424 wrapper | 324.72 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 27.44 ms | 427.55 ms | -400.11 ms | 0.064x | 64.0 | 16 | 48.0 |
| command_buffer::execute | 251.11 ms | 229.21 ms | 21.90 ms | 1.096x | 1936.0 | 1936 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 305.38 ms | 289.24 ms | 16.14 ms | 1.056x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 294.39 ms | 280.56 ms | 13.83 ms | 1.049x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.71 ms | 30.39 ms | -11.68 ms | 0.616x | 493.0 | 655 | -162.0 |
| transpose | 45.07 ms | 47.30 ms | -2.23 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.37 ms | 10.46 ms | -2.09 ms | 0.800x | 184.0 | 195 | -11.0 |
| gather | 14.64 ms | 14.55 ms | 0.09 ms | 1.006x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_direct_vector_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_direct_vector_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.823x
- JAX/reference: 1.227x
- TTFT delta vs reference: -343.42 ms
- ITL delta vs reference: -7.12 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 27.44 ms, reference 427.55 ms, delta -400.11 ms, ratio 0.064x, count delta 48.0
- `command_buffer::execute`: current 251.11 ms, reference 229.21 ms, delta 21.90 ms, ratio 1.096x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 305.38 ms, reference 289.24 ms, delta 16.14 ms, ratio 1.056x, count delta -81.0
- `forward_step_token_ids_jit`: current 294.39 ms, reference 280.56 ms, delta 13.83 ms, ratio 1.049x, count delta 0.0
- `MemcpyD2D`: current 18.71 ms, reference 30.39 ms, delta -11.68 ms, ratio 0.616x, count delta -162.0
- `transpose`: current 45.07 ms, reference 47.30 ms, delta -2.23 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.37 ms, reference 10.46 ms, delta -2.09 ms, ratio 0.800x, count delta -11.0
- `gather`: current 14.64 ms, reference 14.55 ms, delta 0.09 ms, ratio 1.006x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
