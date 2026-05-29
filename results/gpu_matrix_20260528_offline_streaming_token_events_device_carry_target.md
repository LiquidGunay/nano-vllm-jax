# GPU Matrix Report

- created_at_utc: `20260528_082505`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260528_082505`
- output_json: `results/gpu_matrix_20260528_offline_streaming_token_events_device_carry_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.820x (target 0.900x)
- JAX tok/s: 95.37
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 9.36

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 95.37 | 116.37 | 0.820x | 104.74 | 9.36 | 78.02 | 1.222x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 303.93 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 293.13 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 249.14 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.07 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 27.92 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.95 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 9.68 ms | 22.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 8.58 ms | 194.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 670.88 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 670.82 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 669.94 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 587.48 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 325.45 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.95 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:632 generate_with_trace | 671.21 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 671.16 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:161 step | 670.25 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | PjitFunction(compiled) | 581.27 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $profiler.py:424 wrapper | 326.90 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 27.92 ms | 427.55 ms | -399.63 ms | 0.065x | 64.0 | 16 | 48.0 |
| command_buffer::execute | 249.14 ms | 229.21 ms | 19.92 ms | 1.087x | 1936.0 | 1936 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 303.93 ms | 289.24 ms | 14.69 ms | 1.051x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 293.13 ms | 280.56 ms | 12.58 ms | 1.045x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.55 ms | 30.39 ms | -11.83 ms | 0.611x | 493.0 | 655 | -162.0 |
| transpose | 45.07 ms | 47.30 ms | -2.23 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.58 ms | 10.46 ms | -1.89 ms | 0.820x | 194.0 | 195 | -1.0 |
| gather | 14.33 ms | 14.55 ms | -0.22 ms | 0.985x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260528_offline_streaming_token_events_device_carry_target.json`
- report: `results/gpu_matrix_20260528_offline_streaming_token_events_device_carry_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.820x
- JAX/reference: 1.222x
- TTFT delta vs reference: -342.47 ms
- ITL delta vs reference: -7.17 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 27.92 ms, reference 427.55 ms, delta -399.63 ms, ratio 0.065x, count delta 48.0
- `command_buffer::execute`: current 249.14 ms, reference 229.21 ms, delta 19.92 ms, ratio 1.087x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 303.93 ms, reference 289.24 ms, delta 14.69 ms, ratio 1.051x, count delta -81.0
- `forward_step_token_ids_jit`: current 293.13 ms, reference 280.56 ms, delta 12.58 ms, ratio 1.045x, count delta 0.0
- `MemcpyD2D`: current 18.55 ms, reference 30.39 ms, delta -11.83 ms, ratio 0.611x, count delta -162.0
- `transpose`: current 45.07 ms, reference 47.30 ms, delta -2.23 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.58 ms, reference 10.46 ms, delta -1.89 ms, ratio 0.820x, count delta -1.0
- `gather`: current 14.33 ms, reference 14.55 ms, delta -0.22 ms, ratio 0.985x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
