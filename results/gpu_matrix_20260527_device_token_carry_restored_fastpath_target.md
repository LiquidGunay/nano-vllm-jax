# GPU Matrix Report

- created_at_utc: `20260527_141900`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_141900`
- output_json: `results/gpu_matrix_20260527_device_token_carry_restored_fastpath_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.821x (target 0.900x)
- JAX tok/s: 95.50
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 9.23

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 95.50 | 116.37 | 0.821x | 104.74 | 9.23 | 78.02 | 1.224x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 666.20 ms | 52.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 570.18 ms | 104.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 303.93 ms | 249.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 265.29 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 221.56 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.06 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 18.49 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.99 ms | 304.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.89 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 670.12 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 670.07 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 669.28 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3738 run | 616.99 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1900 _run_main_and_sample | 616.39 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 670.86 ms | 14.55 ms | 656.31 ms | 46.092x | 133.0 | 103 | 30.0 |
| np.asarray(jax.Array) | 18.49 ms | 427.55 ms | -409.07 ms | 0.043x | 64.0 | 16 | 48.0 |
| MemcpyD2D | 319.91 ms | 30.39 ms | 289.53 ms | 10.528x | 553.0 | 655 | -102.0 |
| PjRtCApiLoadedExecutable::Execute | 570.18 ms | 289.24 ms | 280.94 ms | 1.971x | 104.0 | 140 | -36.0 |
| forward_step_token_ids_jit | 265.29 ms | 280.56 ms | -15.26 ms | 0.946x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 221.56 ms | 229.21 ms | -7.66 ms | 0.967x | 1936.0 | 1936 | 0.0 |
| transpose | 45.06 ms | 47.30 ms | -2.24 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.52 ms | 10.46 ms | -1.94 ms | 0.815x | 190.0 | 195 | -5.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_device_token_carry_restored_fastpath_target.json`
- report: `results/gpu_matrix_20260527_device_token_carry_restored_fastpath_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.821x
- JAX/reference: 1.224x
- TTFT delta vs reference: -342.17 ms
- ITL delta vs reference: -7.23 ms
- profile movement to explain:
- `gather`: current 670.86 ms, reference 14.55 ms, delta 656.31 ms, ratio 46.092x, count delta 30.0
- `np.asarray(jax.Array)`: current 18.49 ms, reference 427.55 ms, delta -409.07 ms, ratio 0.043x, count delta 48.0
- `MemcpyD2D`: current 319.91 ms, reference 30.39 ms, delta 289.53 ms, ratio 10.528x, count delta -102.0
- `PjRtCApiLoadedExecutable::Execute`: current 570.18 ms, reference 289.24 ms, delta 280.94 ms, ratio 1.971x, count delta -36.0
- `forward_step_token_ids_jit`: current 265.29 ms, reference 280.56 ms, delta -15.26 ms, ratio 0.946x, count delta 0.0
- `command_buffer::execute`: current 221.56 ms, reference 229.21 ms, delta -7.66 ms, ratio 0.967x, count delta 0.0
- `transpose`: current 45.06 ms, reference 47.30 ms, delta -2.24 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.52 ms, reference 10.46 ms, delta -1.94 ms, ratio 0.815x, count delta -5.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
