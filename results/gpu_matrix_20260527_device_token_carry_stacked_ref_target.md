# GPU Matrix Report

- created_at_utc: `20260527_081126`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_081126`
- output_json: `results/gpu_matrix_20260527_device_token_carry_stacked_ref_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.790x (target 0.900x)
- JAX tok/s: 91.94
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 12.79

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 91.94 | 116.37 | 0.790x | 104.74 | 12.79 | 78.02 | 1.178x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.24 s | 0.46 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 317.06 ms | 63.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 297.44 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 251.67 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.08 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.95 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.72 ms | 27.5 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 9.57 ms | 194.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | gather | 4.65 ms | 81.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.94 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 720.60 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 720.54 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 719.57 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 599.16 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:424 wrapper | 358.75 ms | 29 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.94 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:373 generate_with_trace | 673.18 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:402 _generate_with_trace_deferred_tokens | 673.12 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:142 step | 672.27 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | PjitFunction(compiled) | 586.80 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:3703 run | 322.84 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 1.04 ms | 427.55 ms | -426.51 ms | 0.002x | 4.0 | 16 | -12.0 |
| PjRtCApiLoadedExecutable::Execute | 317.06 ms | 289.24 ms | 27.83 ms | 1.096x | 63.0 | 140 | -77.0 |
| command_buffer::execute | 251.67 ms | 229.21 ms | 22.46 ms | 1.098x | 1936.0 | 1936 | 0.0 |
| forward_step_token_ids_jit | 297.44 ms | 280.56 ms | 16.89 ms | 1.060x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.63 ms | 30.39 ms | -11.76 ms | 0.613x | 493.0 | 655 | -162.0 |
| transpose | 45.09 ms | 47.30 ms | -2.21 ms | 0.953x | 313.5 | 312 | 1.5 |
| command_buffer::update | 9.57 ms | 10.46 ms | -0.89 ms | 0.915x | 194.0 | 195 | -1.0 |
| gather | 15.38 ms | 14.55 ms | 0.82 ms | 1.056x | 108.5 | 103 | 5.5 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_stacked_ref_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_stacked_ref_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.790x
- JAX/reference: 1.178x
- TTFT delta vs reference: -342.13 ms
- ITL delta vs reference: -7.28 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 1.04 ms, reference 427.55 ms, delta -426.51 ms, ratio 0.002x, count delta -12.0
- `PjRtCApiLoadedExecutable::Execute`: current 317.06 ms, reference 289.24 ms, delta 27.83 ms, ratio 1.096x, count delta -77.0
- `command_buffer::execute`: current 251.67 ms, reference 229.21 ms, delta 22.46 ms, ratio 1.098x, count delta 0.0
- `forward_step_token_ids_jit`: current 297.44 ms, reference 280.56 ms, delta 16.89 ms, ratio 1.060x, count delta 0.0
- `MemcpyD2D`: current 18.63 ms, reference 30.39 ms, delta -11.76 ms, ratio 0.613x, count delta -162.0
- `transpose`: current 45.09 ms, reference 47.30 ms, delta -2.21 ms, ratio 0.953x, count delta 1.5
- `command_buffer::update`: current 9.57 ms, reference 10.46 ms, delta -0.89 ms, ratio 0.915x, count delta -1.0
- `gather`: current 15.38 ms, reference 14.55 ms, delta 0.82 ms, ratio 1.056x, count delta 5.5
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
