# GPU Matrix Report

- created_at_utc: `20260530_041855`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_041855`
- output_json: `results/gdn_fla_check.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.771x (target 0.900x)
- JAX tok/s: 89.75
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 14.98

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 89.75 | 116.37 | 0.771x | 104.74 | 14.98 | 78.02 | 1.150x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.25 s | 0.46 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 289.84 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 289.57 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 228.48 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 73.12 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 33.69 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.92 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 14.07 ms | 219.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 13.61 ms | 188.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.32 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.55 ms | 17 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 711.81 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 711.76 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 710.90 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 577.82 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 383.16 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 33.69 ms | 427.55 ms | -393.86 ms | 0.079x | 64.0 | 16 | 48.0 |
| transpose | 73.12 ms | 47.30 ms | 25.83 ms | 1.546x | 1547.0 | 312 | 1235.0 |
| forward_step_token_ids_jit | 289.84 ms | 280.56 ms | 9.28 ms | 1.033x | 16.0 | 16 | 0.0 |
| command_buffer::update | 13.61 ms | 10.46 ms | 3.15 ms | 1.301x | 188.0 | 195 | -7.0 |
| command_buffer::execute | 228.48 ms | 229.21 ms | -0.73 ms | 0.997x | 1954.0 | 1936 | 18.0 |
| MemcpyD2D | 30.00 ms | 30.39 ms | -0.39 ms | 0.987x | 493.0 | 655 | -162.0 |
| PjRtCApiLoadedExecutable::Execute | 289.57 ms | 289.24 ms | 0.34 ms | 1.001x | 59.0 | 140 | -81.0 |
| gather | 14.38 ms | 14.55 ms | -0.18 ms | 0.988x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gdn_fla_check.json`
- report: `results/gdn_fla_check.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.771x
- JAX/reference: 1.150x
- TTFT delta vs reference: -334.58 ms
- ITL delta vs reference: -5.87 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 33.69 ms, reference 427.55 ms, delta -393.86 ms, ratio 0.079x, count delta 48.0
- `transpose`: current 73.12 ms, reference 47.30 ms, delta 25.83 ms, ratio 1.546x, count delta 1235.0
- `forward_step_token_ids_jit`: current 289.84 ms, reference 280.56 ms, delta 9.28 ms, ratio 1.033x, count delta 0.0
- `command_buffer::update`: current 13.61 ms, reference 10.46 ms, delta 3.15 ms, ratio 1.301x, count delta -7.0
- `command_buffer::execute`: current 228.48 ms, reference 229.21 ms, delta -0.73 ms, ratio 0.997x, count delta 18.0
- `MemcpyD2D`: current 30.00 ms, reference 30.39 ms, delta -0.39 ms, ratio 0.987x, count delta -162.0
- `PjRtCApiLoadedExecutable::Execute`: current 289.57 ms, reference 289.24 ms, delta 0.34 ms, ratio 1.001x, count delta -81.0
- `gather`: current 14.38 ms, reference 14.55 ms, delta -0.18 ms, ratio 0.988x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
