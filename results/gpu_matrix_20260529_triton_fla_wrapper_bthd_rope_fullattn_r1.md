# GPU Matrix Report

- created_at_utc: `20260529_203749`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260529_203749`
- output_json: `results/gpu_matrix_20260529_triton_fla_wrapper_bthd_rope_fullattn_r1.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.764x (target 0.900x)
- JAX tok/s: 88.96
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 15.78

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 88.96 | 116.37 | 0.764x | 104.74 | 15.78 | 78.02 | 1.140x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 314.36 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 312.32 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 234.65 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 73.07 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 29.35 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 22.19 ms | 189.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.92 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 15.53 ms | 219.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.24 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.54 ms | 17 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 716.10 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 716.03 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 714.80 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 625.90 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 347.80 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 29.35 ms | 427.55 ms | -398.20 ms | 0.069x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 314.36 ms | 280.56 ms | 33.81 ms | 1.121x | 16.0 | 16 | 0.0 |
| transpose | 73.07 ms | 47.30 ms | 25.77 ms | 1.545x | 1547.0 | 312 | 1235.0 |
| PjRtCApiLoadedExecutable::Execute | 312.32 ms | 289.24 ms | 23.09 ms | 1.080x | 59.0 | 140 | -81.0 |
| command_buffer::update | 22.19 ms | 10.46 ms | 11.73 ms | 2.121x | 189.0 | 195 | -6.0 |
| command_buffer::execute | 234.65 ms | 229.21 ms | 5.44 ms | 1.024x | 1954.0 | 1936 | 18.0 |
| MemcpyD2D | 31.44 ms | 30.39 ms | 1.06 ms | 1.035x | 493.0 | 655 | -162.0 |
| gather | 14.80 ms | 14.55 ms | 0.24 ms | 1.017x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_bthd_rope_fullattn_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_bthd_rope_fullattn_r1.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.764x
- JAX/reference: 1.140x
- TTFT delta vs reference: -334.77 ms
- ITL delta vs reference: -5.76 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 29.35 ms, reference 427.55 ms, delta -398.20 ms, ratio 0.069x, count delta 48.0
- `forward_step_token_ids_jit`: current 314.36 ms, reference 280.56 ms, delta 33.81 ms, ratio 1.121x, count delta 0.0
- `transpose`: current 73.07 ms, reference 47.30 ms, delta 25.77 ms, ratio 1.545x, count delta 1235.0
- `PjRtCApiLoadedExecutable::Execute`: current 312.32 ms, reference 289.24 ms, delta 23.09 ms, ratio 1.080x, count delta -81.0
- `command_buffer::update`: current 22.19 ms, reference 10.46 ms, delta 11.73 ms, ratio 2.121x, count delta -6.0
- `command_buffer::execute`: current 234.65 ms, reference 229.21 ms, delta 5.44 ms, ratio 1.024x, count delta 18.0
- `MemcpyD2D`: current 31.44 ms, reference 30.39 ms, delta 1.06 ms, ratio 1.035x, count delta -162.0
- `gather`: current 14.80 ms, reference 14.55 ms, delta 0.24 ms, ratio 1.017x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
