# GPU Matrix Report

- created_at_utc: `20260527_142213`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_142213`
- output_json: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_direct_vector_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.820x (target 0.900x)
- JAX tok/s: 95.44
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 9.29

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 95.44 | 116.37 | 0.820x | 104.74 | 9.29 | 78.02 | 1.223x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 306.30 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 295.03 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 251.25 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.07 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 26.24 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.95 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.03 ms | 22.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 8.68 ms | 194.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.93 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.67 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 670.56 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 670.51 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 669.71 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 588.27 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 325.70 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 26.24 ms | 427.55 ms | -401.32 ms | 0.061x | 64.0 | 16 | 48.0 |
| command_buffer::execute | 251.25 ms | 229.21 ms | 22.03 ms | 1.096x | 1936.0 | 1936 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 306.30 ms | 289.24 ms | 17.06 ms | 1.059x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 295.03 ms | 280.56 ms | 14.48 ms | 1.052x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.73 ms | 30.39 ms | -11.66 ms | 0.616x | 493.0 | 655 | -162.0 |
| transpose | 45.07 ms | 47.30 ms | -2.23 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.68 ms | 10.46 ms | -1.79 ms | 0.829x | 194.0 | 195 | -1.0 |
| gather | 14.70 ms | 14.55 ms | 0.14 ms | 1.010x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_direct_vector_target.json`
- report: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_direct_vector_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.820x
- JAX/reference: 1.223x
- TTFT delta vs reference: -341.80 ms
- ITL delta vs reference: -7.10 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 26.24 ms, reference 427.55 ms, delta -401.32 ms, ratio 0.061x, count delta 48.0
- `command_buffer::execute`: current 251.25 ms, reference 229.21 ms, delta 22.03 ms, ratio 1.096x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 306.30 ms, reference 289.24 ms, delta 17.06 ms, ratio 1.059x, count delta -81.0
- `forward_step_token_ids_jit`: current 295.03 ms, reference 280.56 ms, delta 14.48 ms, ratio 1.052x, count delta 0.0
- `MemcpyD2D`: current 18.73 ms, reference 30.39 ms, delta -11.66 ms, ratio 0.616x, count delta -162.0
- `transpose`: current 45.07 ms, reference 47.30 ms, delta -2.23 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.68 ms, reference 10.46 ms, delta -1.79 ms, ratio 0.829x, count delta -1.0
- `gather`: current 14.70 ms, reference 14.55 ms, delta 0.14 ms, ratio 1.010x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
