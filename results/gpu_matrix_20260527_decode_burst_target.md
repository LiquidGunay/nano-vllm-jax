# GPU Matrix Report

- created_at_utc: `20260527_062303`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_062303`
- output_json: `results/gpu_matrix_20260527_decode_burst_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.150x (target 0.900x)
- JAX tok/s: 17.46
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 87.28

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 17.46 | 116.37 | 0.150x | 104.74 | 87.28 | 78.02 | 0.224x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 1.0 | 4.0 | 5120.0 | 0.54 s | 3.13 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 294.41 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 294.40 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 231.69 ms | 29.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 227.25 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 207.61 ms | 1741.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 134.27 ms | 242.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | transpose | 40.23 ms | 471.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 34.71 ms | 132.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.95 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.63 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 6444.71 ms | 4 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $traceback_util.py:191 reraise_with_filtered_traceback | 4797.04 ms | 2220 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:365 generate_with_trace | 3670.78 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:279 iter_generate | 3670.46 ms | 5 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:134 step | 3670.20 ms | 2 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.65 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | PjitFunction(compiled) | 6440.36 ms | 4 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $traceback_util.py:191 reraise_with_filtered_traceback | 4811.08 ms | 2220 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:365 generate_with_trace | 3661.83 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:279 iter_generate | 3661.54 ms | 5 |
| long_prefill_512_2048 | gpu_paged_default | 2 | cpu | $llm_engine.py:134 step | 3661.30 ms | 2 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=2

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array.py:325 tolist | 294.41 ms | 427.68 ms | -133.27 ms | 0.688x | 1.0 | 16 | -15.0 |
| np.asarray(jax.Array) | 294.40 ms | 427.55 ms | -133.16 ms | 0.689x | 1.0 | 16 | -15.0 |
| gather | 138.87 ms | 14.55 ms | 124.32 ms | 9.541x | 293.0 | 103 | 190.0 |
| PjRtCApiLoadedExecutable::Execute | 231.69 ms | 289.24 ms | -57.54 ms | 0.801x | 29.0 | 140 | -111.0 |
| forward_step_token_ids_jit | 227.25 ms | 280.56 ms | -53.30 ms | 0.810x | 1.0 | 16 | -15.0 |
| transpose | 74.94 ms | 47.30 ms | 27.64 ms | 1.584x | 603.0 | 312 | 291.0 |
| MemcpyD2D | 6.72 ms | 30.39 ms | -23.66 ms | 0.221x | 103.0 | 655 | -552.0 |
| command_buffer::execute | 207.61 ms | 229.21 ms | -21.60 ms | 0.906x | 1741.0 | 1936 | -195.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_decode_burst_target.json`
- report: `results/gpu_matrix_20260527_decode_burst_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.150x
- JAX/reference: 0.224x
- TTFT delta vs reference: -45.19 ms
- ITL delta vs reference: -15.82 ms
- profile movement to explain:
- `array.py:325 tolist`: current 294.41 ms, reference 427.68 ms, delta -133.27 ms, ratio 0.688x, count delta -15.0
- `np.asarray(jax.Array)`: current 294.40 ms, reference 427.55 ms, delta -133.16 ms, ratio 0.689x, count delta -15.0
- `gather`: current 138.87 ms, reference 14.55 ms, delta 124.32 ms, ratio 9.541x, count delta 190.0
- `PjRtCApiLoadedExecutable::Execute`: current 231.69 ms, reference 289.24 ms, delta -57.54 ms, ratio 0.801x, count delta -111.0
- `forward_step_token_ids_jit`: current 227.25 ms, reference 280.56 ms, delta -53.30 ms, ratio 0.810x, count delta -15.0
- `transpose`: current 74.94 ms, reference 47.30 ms, delta 27.64 ms, ratio 1.584x, count delta 291.0
- `MemcpyD2D`: current 6.72 ms, reference 30.39 ms, delta -23.66 ms, ratio 0.221x, count delta -552.0
- `command_buffer::execute`: current 207.61 ms, reference 229.21 ms, delta -21.60 ms, ratio 0.906x, count delta -195.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
