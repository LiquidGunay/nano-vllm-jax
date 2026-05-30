# GPU Matrix Report

- created_at_utc: `20260529_171324`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260529_171324`
- output_json: `results/gpu_matrix_20260529_triton_fla_wrapper_triton_decode_r1.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.756x (target 0.900x)
- JAX tok/s: 87.99
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 16.75

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 87.99 | 116.37 | 0.756x | 104.74 | 16.75 | 78.02 | 1.128x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.25 s | 0.47 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 377.32 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 375.51 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 245.24 ms | 2224.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 73.14 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 26.04 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 25.54 ms | 219.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 23.92 ms | 454.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.93 ms | 274.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.24 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.55 ms | 17 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.82 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 751.15 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 724.63 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 724.57 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 723.07 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 407.19 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 26.04 ms | 427.55 ms | -401.52 ms | 0.061x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 377.32 ms | 280.56 ms | 96.76 ms | 1.345x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 375.51 ms | 289.24 ms | 86.28 ms | 1.298x | 59.0 | 140 | -81.0 |
| transpose | 73.14 ms | 47.30 ms | 25.84 ms | 1.546x | 1547.0 | 312 | 1235.0 |
| command_buffer::execute | 245.24 ms | 229.21 ms | 16.03 ms | 1.070x | 2224.0 | 1936 | 288.0 |
| command_buffer::update | 23.92 ms | 10.46 ms | 13.46 ms | 2.287x | 454.0 | 195 | 259.0 |
| MemcpyD2D | 41.47 ms | 30.39 ms | 11.09 ms | 1.365x | 493.0 | 655 | -162.0 |
| gather | 14.82 ms | 14.55 ms | 0.26 ms | 1.018x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_triton_decode_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_triton_decode_r1.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.756x
- JAX/reference: 1.128x
- TTFT delta vs reference: -330.75 ms
- ITL delta vs reference: -5.68 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 26.04 ms, reference 427.55 ms, delta -401.52 ms, ratio 0.061x, count delta 48.0
- `forward_step_token_ids_jit`: current 377.32 ms, reference 280.56 ms, delta 96.76 ms, ratio 1.345x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 375.51 ms, reference 289.24 ms, delta 86.28 ms, ratio 1.298x, count delta -81.0
- `transpose`: current 73.14 ms, reference 47.30 ms, delta 25.84 ms, ratio 1.546x, count delta 1235.0
- `command_buffer::execute`: current 245.24 ms, reference 229.21 ms, delta 16.03 ms, ratio 1.070x, count delta 288.0
- `command_buffer::update`: current 23.92 ms, reference 10.46 ms, delta 13.46 ms, ratio 2.287x, count delta 259.0
- `MemcpyD2D`: current 41.47 ms, reference 30.39 ms, delta 11.09 ms, ratio 1.365x, count delta -162.0
- `gather`: current 14.82 ms, reference 14.55 ms, delta 0.26 ms, ratio 1.018x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
