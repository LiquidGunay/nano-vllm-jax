# GPU Matrix Report

- created_at_utc: `20260527_141012`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_141012`
- output_json: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.707x (target 0.900x)
- JAX tok/s: 82.24
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 22.50

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 82.24 | 116.37 | 0.707x | 104.74 | 22.50 | 78.02 | 1.054x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.24 s | 0.53 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 705.80 ms | 142.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 348.00 ms | 584.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 281.17 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 226.49 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.07 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 27.09 ms | 624.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 16.45 ms | 679.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 12.71 ms | 184.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.95 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.64 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 778.21 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 778.01 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 774.90 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3724 run | 733.47 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1886 _run_main_and_sample | 732.81 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 710.46 ms | 14.55 ms | 695.90 ms | 48.813x | 223.0 | 103 | 120.0 |
| np.asarray(jax.Array) | 4.71 ms | 427.55 ms | -422.85 ms | 0.011x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 348.00 ms | 289.24 ms | 58.77 ms | 1.203x | 584.0 | 140 | 444.0 |
| MemcpyD2D | 43.54 ms | 30.39 ms | 13.16 ms | 1.433x | 1303.0 | 655 | 648.0 |
| command_buffer::execute | 226.49 ms | 229.21 ms | -2.73 ms | 0.988x | 1936.0 | 1936 | 0.0 |
| command_buffer::update | 12.71 ms | 10.46 ms | 2.24 ms | 1.214x | 184.0 | 195 | -11.0 |
| transpose | 45.07 ms | 47.30 ms | -2.23 ms | 0.953x | 312.0 | 312 | 0.0 |
| forward_step_token_ids_jit | 281.17 ms | 280.56 ms | 0.62 ms | 1.002x | 16.0 | 16 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_target.json`
- report: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.707x
- JAX/reference: 1.054x
- TTFT delta vs reference: -340.04 ms
- ITL delta vs reference: 0.78 ms
- profile movement to explain:
- `gather`: current 710.46 ms, reference 14.55 ms, delta 695.90 ms, ratio 48.813x, count delta 120.0
- `np.asarray(jax.Array)`: current 4.71 ms, reference 427.55 ms, delta -422.85 ms, ratio 0.011x, count delta 48.0
- `PjRtCApiLoadedExecutable::Execute`: current 348.00 ms, reference 289.24 ms, delta 58.77 ms, ratio 1.203x, count delta 444.0
- `MemcpyD2D`: current 43.54 ms, reference 30.39 ms, delta 13.16 ms, ratio 1.433x, count delta 648.0
- `command_buffer::execute`: current 226.49 ms, reference 229.21 ms, delta -2.73 ms, ratio 0.988x, count delta 0.0
- `command_buffer::update`: current 12.71 ms, reference 10.46 ms, delta 2.24 ms, ratio 1.214x, count delta -11.0
- `transpose`: current 45.07 ms, reference 47.30 ms, delta -2.23 ms, ratio 0.953x, count delta 0.0
- `forward_step_token_ids_jit`: current 281.17 ms, reference 280.56 ms, delta 0.62 ms, ratio 1.002x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
