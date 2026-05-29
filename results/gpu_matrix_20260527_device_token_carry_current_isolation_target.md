# GPU Matrix Report

- created_at_utc: `20260527_141442`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_141442`
- output_json: `results/gpu_matrix_20260527_device_token_carry_current_isolation_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.729x (target 0.900x)
- JAX tok/s: 84.84
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 19.89

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 84.84 | 116.37 | 0.729x | 104.74 | 19.89 | 78.02 | 1.087x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.24 s | 0.51 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 695.08 ms | 142.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 338.03 ms | 584.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 275.52 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 224.61 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.08 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 26.46 ms | 624.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 16.45 ms | 679.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 10.80 ms | 195.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 754.35 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 754.29 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 753.27 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3724 run | 717.83 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1886 _run_main_and_sample | 717.19 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 699.74 ms | 14.55 ms | 685.19 ms | 48.076x | 223.0 | 103 | 120.0 |
| np.asarray(jax.Array) | 3.00 ms | 427.55 ms | -424.56 ms | 0.007x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 338.03 ms | 289.24 ms | 48.79 ms | 1.169x | 584.0 | 140 | 444.0 |
| MemcpyD2D | 42.91 ms | 30.39 ms | 12.52 ms | 1.412x | 1303.0 | 655 | 648.0 |
| forward_step_token_ids_jit | 275.52 ms | 280.56 ms | -5.03 ms | 0.982x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 224.61 ms | 229.21 ms | -4.60 ms | 0.980x | 1936.0 | 1936 | 0.0 |
| transpose | 45.08 ms | 47.30 ms | -2.22 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 10.80 ms | 10.46 ms | 0.34 ms | 1.032x | 195.0 | 195 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_current_isolation_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_current_isolation_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.729x
- JAX/reference: 1.087x
- TTFT delta vs reference: -343.33 ms
- ITL delta vs reference: -1.28 ms
- profile movement to explain:
- `gather`: current 699.74 ms, reference 14.55 ms, delta 685.19 ms, ratio 48.076x, count delta 120.0
- `np.asarray(jax.Array)`: current 3.00 ms, reference 427.55 ms, delta -424.56 ms, ratio 0.007x, count delta 48.0
- `PjRtCApiLoadedExecutable::Execute`: current 338.03 ms, reference 289.24 ms, delta 48.79 ms, ratio 1.169x, count delta 444.0
- `MemcpyD2D`: current 42.91 ms, reference 30.39 ms, delta 12.52 ms, ratio 1.412x, count delta 648.0
- `forward_step_token_ids_jit`: current 275.52 ms, reference 280.56 ms, delta -5.03 ms, ratio 0.982x, count delta 0.0
- `command_buffer::execute`: current 224.61 ms, reference 229.21 ms, delta -4.60 ms, ratio 0.980x, count delta 0.0
- `transpose`: current 45.08 ms, reference 47.30 ms, delta -2.22 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 10.80 ms, reference 10.46 ms, delta 0.34 ms, ratio 1.032x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
