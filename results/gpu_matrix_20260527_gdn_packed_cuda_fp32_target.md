# GPU Matrix Report

- created_at_utc: `20260527_060446`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_060446`
- output_json: `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.760x (target 0.900x)
- JAX tok/s: 88.41
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 16.33

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | yes | no | 88.41 | 116.37 | 0.760x | 104.74 | 16.33 | 78.02 | 1.133x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.54 s | 0.19 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 412.41 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 412.29 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 273.71 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 271.58 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 224.95 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.08 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.90 ms | 259.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 9.61 ms | 184.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.57 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.59 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:365 generate_with_trace | 722.93 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:279 iter_generate | 722.82 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:134 step | 721.92 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3547 run | 696.73 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1778 _run_main_and_sample | 695.90 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.89 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.63 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.03 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:365 generate_with_trace | 724.92 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:279 iter_generate | 724.80 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:134 step | 723.88 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:3547 run | 698.32 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $model_runner.py:1778 _run_main_and_sample | 697.50 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 271.58 ms | 289.24 ms | -17.65 ms | 0.939x | 44.0 | 140 | -96.0 |
| array.py:325 tolist | 412.41 ms | 427.68 ms | -15.27 ms | 0.964x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 412.29 ms | 427.55 ms | -15.26 ms | 0.964x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.41 ms | 30.39 ms | -11.97 ms | 0.606x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 273.71 ms | 280.56 ms | -6.85 ms | 0.976x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 224.95 ms | 229.21 ms | -4.26 ms | 0.981x | 1936.0 | 1936 | 0.0 |
| transpose | 45.08 ms | 47.30 ms | -2.22 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 9.61 ms | 10.46 ms | -0.85 ms | 0.918x | 184.0 | 195 | -11.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.json`
- report: `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.760x
- JAX/reference: 1.133x
- TTFT delta vs reference: -46.29 ms
- ITL delta vs reference: -3.39 ms
- profile movement to explain:
- `PjRtCApiLoadedExecutable::Execute`: current 271.58 ms, reference 289.24 ms, delta -17.65 ms, ratio 0.939x, count delta -96.0
- `array.py:325 tolist`: current 412.41 ms, reference 427.68 ms, delta -15.27 ms, ratio 0.964x, count delta 0.0
- `np.asarray(jax.Array)`: current 412.29 ms, reference 427.55 ms, delta -15.26 ms, ratio 0.964x, count delta 0.0
- `MemcpyD2D`: current 18.41 ms, reference 30.39 ms, delta -11.97 ms, ratio 0.606x, count delta -192.0
- `forward_step_token_ids_jit`: current 273.71 ms, reference 280.56 ms, delta -6.85 ms, ratio 0.976x, count delta 0.0
- `command_buffer::execute`: current 224.95 ms, reference 229.21 ms, delta -4.26 ms, ratio 0.981x, count delta 0.0
- `transpose`: current 45.08 ms, reference 47.30 ms, delta -2.22 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 9.61 ms, reference 10.46 ms, delta -0.85 ms, ratio 0.918x, count delta -11.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
