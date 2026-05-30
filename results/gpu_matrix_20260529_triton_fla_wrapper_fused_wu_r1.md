# GPU Matrix Report

- created_at_utc: `20260529_212450`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260529_212450`
- output_json: `results/gpu_matrix_20260529_triton_fla_wrapper_fused_wu_r1.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.764x (target 0.900x)
- JAX tok/s: 88.93
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 15.81

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 88.93 | 116.37 | 0.764x | 104.74 | 15.81 | 78.02 | 1.140x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 313.90 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 311.90 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 235.09 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 73.11 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 28.31 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 21.28 ms | 194.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.92 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 15.50 ms | 219.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.25 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.51 ms | 17 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.82 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 717.41 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 717.34 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 716.06 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 624.83 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 347.73 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 28.31 ms | 427.55 ms | -399.24 ms | 0.066x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 313.90 ms | 280.56 ms | 33.35 ms | 1.119x | 16.0 | 16 | 0.0 |
| transpose | 73.11 ms | 47.30 ms | 25.81 ms | 1.546x | 1547.0 | 312 | 1235.0 |
| PjRtCApiLoadedExecutable::Execute | 311.90 ms | 289.24 ms | 22.67 ms | 1.078x | 59.0 | 140 | -81.0 |
| command_buffer::update | 21.28 ms | 10.46 ms | 10.82 ms | 2.034x | 194.0 | 195 | -1.0 |
| command_buffer::execute | 235.09 ms | 229.21 ms | 5.88 ms | 1.026x | 1954.0 | 1936 | 18.0 |
| MemcpyD2D | 31.42 ms | 30.39 ms | 1.03 ms | 1.034x | 493.0 | 655 | -162.0 |
| gather | 15.54 ms | 14.55 ms | 0.98 ms | 1.067x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_fused_wu_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_fused_wu_r1.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.764x
- JAX/reference: 1.140x
- TTFT delta vs reference: -333.18 ms
- ITL delta vs reference: -5.80 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 28.31 ms, reference 427.55 ms, delta -399.24 ms, ratio 0.066x, count delta 48.0
- `forward_step_token_ids_jit`: current 313.90 ms, reference 280.56 ms, delta 33.35 ms, ratio 1.119x, count delta 0.0
- `transpose`: current 73.11 ms, reference 47.30 ms, delta 25.81 ms, ratio 1.546x, count delta 1235.0
- `PjRtCApiLoadedExecutable::Execute`: current 311.90 ms, reference 289.24 ms, delta 22.67 ms, ratio 1.078x, count delta -81.0
- `command_buffer::update`: current 21.28 ms, reference 10.46 ms, delta 10.82 ms, ratio 2.034x, count delta -1.0
- `command_buffer::execute`: current 235.09 ms, reference 229.21 ms, delta 5.88 ms, ratio 1.026x, count delta 18.0
- `MemcpyD2D`: current 31.42 ms, reference 30.39 ms, delta 1.03 ms, ratio 1.034x, count delta -162.0
- `gather`: current 15.54 ms, reference 14.55 ms, delta 0.98 ms, ratio 1.067x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
