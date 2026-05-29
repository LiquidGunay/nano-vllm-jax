# GPU Matrix Report

- created_at_utc: `20260527_141536`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_141536`
- output_json: `results/gpu_matrix_20260527_current_head_no_device_carry_control_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.778x (target 0.900x)
- JAX tok/s: 90.52
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 14.21

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 90.52 | 116.37 | 0.778x | 104.74 | 14.21 | 78.02 | 1.160x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.53 s | 0.17 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 397.69 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 397.56 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 271.96 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 269.93 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 223.80 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 45.09 ms | 312.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.90 ms | 259.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 9.76 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.47 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 706.99 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:302 iter_generate | 706.82 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 705.62 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3724 run | 681.65 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1886 _run_main_and_sample | 680.85 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 397.56 ms | 427.55 ms | -30.00 ms | 0.930x | 16.0 | 16 | 0.0 |
| array.py:325 tolist | 397.69 ms | 427.68 ms | -29.99 ms | 0.930x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 269.93 ms | 289.24 ms | -19.31 ms | 0.933x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.26 ms | 30.39 ms | -12.13 ms | 0.601x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 271.96 ms | 280.56 ms | -8.60 ms | 0.969x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 223.80 ms | 229.21 ms | -5.41 ms | 0.976x | 1936.0 | 1936 | 0.0 |
| transpose | 45.09 ms | 47.30 ms | -2.21 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.94 ms | 10.46 ms | -1.52 ms | 0.855x | 183.0 | 195 | -12.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_current_head_no_device_carry_control_target.json`
- report: `results/gpu_matrix_20260527_current_head_no_device_carry_control_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.778x
- JAX/reference: 1.160x
- TTFT delta vs reference: -49.54 ms
- ITL delta vs reference: -4.38 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 397.56 ms, reference 427.55 ms, delta -30.00 ms, ratio 0.930x, count delta 0.0
- `array.py:325 tolist`: current 397.69 ms, reference 427.68 ms, delta -29.99 ms, ratio 0.930x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 269.93 ms, reference 289.24 ms, delta -19.31 ms, ratio 0.933x, count delta -96.0
- `MemcpyD2D`: current 18.26 ms, reference 30.39 ms, delta -12.13 ms, ratio 0.601x, count delta -192.0
- `forward_step_token_ids_jit`: current 271.96 ms, reference 280.56 ms, delta -8.60 ms, ratio 0.969x, count delta 0.0
- `command_buffer::execute`: current 223.80 ms, reference 229.21 ms, delta -5.41 ms, ratio 0.976x, count delta 0.0
- `transpose`: current 45.09 ms, reference 47.30 ms, delta -2.21 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.94 ms, reference 10.46 ms, delta -1.52 ms, ratio 0.855x, count delta -12.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
