# GPU Matrix Report

- created_at_utc: `20260527_141349`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_141349`
- output_json: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_trace_fixed_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.726x (target 0.900x)
- JAX tok/s: 84.54
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 20.20

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 84.54 | 116.37 | 0.726x | 104.74 | 20.20 | 78.02 | 1.084x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 696.82 ms | 142.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 340.26 ms | 584.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 277.67 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 224.93 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.08 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 26.37 ms | 624.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 16.46 ms | 679.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 9.89 ms | 184.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 757.05 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 756.99 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 755.98 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3724 run | 720.42 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1886 _run_main_and_sample | 719.77 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 701.48 ms | 14.55 ms | 686.92 ms | 48.195x | 223.0 | 103 | 120.0 |
| np.asarray(jax.Array) | 3.03 ms | 427.55 ms | -424.52 ms | 0.007x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 340.26 ms | 289.24 ms | 51.02 ms | 1.176x | 584.0 | 140 | 444.0 |
| MemcpyD2D | 42.83 ms | 30.39 ms | 12.44 ms | 1.409x | 1303.0 | 655 | 648.0 |
| command_buffer::execute | 224.93 ms | 229.21 ms | -4.28 ms | 0.981x | 1936.0 | 1936 | 0.0 |
| forward_step_token_ids_jit | 277.67 ms | 280.56 ms | -2.89 ms | 0.990x | 16.0 | 16 | 0.0 |
| transpose | 45.08 ms | 47.30 ms | -2.22 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 9.89 ms | 10.46 ms | -0.57 ms | 0.945x | 184.0 | 195 | -11.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_trace_fixed_target.json`
- report: `results/gpu_matrix_20260527_overlapped_streaming_prefetch_trace_fixed_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.726x
- JAX/reference: 1.084x
- TTFT delta vs reference: -339.28 ms
- ITL delta vs reference: -1.60 ms
- profile movement to explain:
- `gather`: current 701.48 ms, reference 14.55 ms, delta 686.92 ms, ratio 48.195x, count delta 120.0
- `np.asarray(jax.Array)`: current 3.03 ms, reference 427.55 ms, delta -424.52 ms, ratio 0.007x, count delta 48.0
- `PjRtCApiLoadedExecutable::Execute`: current 340.26 ms, reference 289.24 ms, delta 51.02 ms, ratio 1.176x, count delta 444.0
- `MemcpyD2D`: current 42.83 ms, reference 30.39 ms, delta 12.44 ms, ratio 1.409x, count delta 648.0
- `command_buffer::execute`: current 224.93 ms, reference 229.21 ms, delta -4.28 ms, ratio 0.981x, count delta 0.0
- `forward_step_token_ids_jit`: current 277.67 ms, reference 280.56 ms, delta -2.89 ms, ratio 0.990x, count delta 0.0
- `transpose`: current 45.08 ms, reference 47.30 ms, delta -2.22 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 9.89 ms, reference 10.46 ms, delta -0.57 ms, ratio 0.945x, count delta -11.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
