# GPU Matrix Report

- created_at_utc: `20260527_043741`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_043741`
- output_json: `results/gpu_matrix_20260527_scoped_profile_target.json`
- jax_python: `.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.780x (target 0.900x)
- JAX tok/s: 90.81
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 13.93

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | yes | no | 90.81 | 116.37 | 0.780x | 104.74 | 13.93 | 78.02 | 1.164x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.53 s | 0.17 s |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | generate_with_trace | 704.78 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | _run_main_and_sample | 678.91 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | fusion | 507.03 ms | 23935.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 397.54 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 397.42 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 270.34 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 268.43 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | jit_compiled:XLA GPU module | 252.87 ms | 16.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.89 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:365 generate_with_trace | 704.50 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:279 iter_generate | 704.39 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:134 step | 703.50 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3547 run | 679.80 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1778 _run_main_and_sample | 678.90 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.44 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.03 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:365 generate_with_trace | 705.05 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:279 iter_generate | 704.94 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:134 step | 703.93 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:3547 run | 679.80 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:1778 _run_main_and_sample | 678.92 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array.py:325 tolist | 397.54 ms | 427.68 ms | -30.14 ms | 0.930x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 397.42 ms | 427.55 ms | -30.14 ms | 0.930x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 268.43 ms | 289.24 ms | -20.81 ms | 0.928x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.29 ms | 30.39 ms | -12.10 ms | 0.602x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 270.34 ms | 280.56 ms | -10.22 ms | 0.964x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 223.64 ms | 229.21 ms | -5.57 ms | 0.976x | 1936.0 | 1936 | 0.0 |
| transpose | 45.06 ms | 47.30 ms | -2.24 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.85 ms | 10.46 ms | -1.62 ms | 0.845x | 181.0 | 195 | -14.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_scoped_profile_target.json`
- report: `results/gpu_matrix_20260527_scoped_profile_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.780x
- JAX/reference: 1.164x
- TTFT delta vs reference: -49.63 ms
- ITL delta vs reference: -4.47 ms
- profile movement to explain:
- `array.py:325 tolist`: current 397.54 ms, reference 427.68 ms, delta -30.14 ms, ratio 0.930x, count delta 0.0
- `np.asarray(jax.Array)`: current 397.42 ms, reference 427.55 ms, delta -30.14 ms, ratio 0.930x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 268.43 ms, reference 289.24 ms, delta -20.81 ms, ratio 0.928x, count delta -96.0
- `MemcpyD2D`: current 18.29 ms, reference 30.39 ms, delta -12.10 ms, ratio 0.602x, count delta -192.0
- `forward_step_token_ids_jit`: current 270.34 ms, reference 280.56 ms, delta -10.22 ms, ratio 0.964x, count delta 0.0
- `command_buffer::execute`: current 223.64 ms, reference 229.21 ms, delta -5.57 ms, ratio 0.976x, count delta 0.0
- `transpose`: current 45.06 ms, reference 47.30 ms, delta -2.24 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.85 ms, reference 10.46 ms, delta -1.62 ms, ratio 0.845x, count delta -14.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
