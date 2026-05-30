# GPU Matrix Report

- created_at_utc: `20260530_043905`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_043905`
- output_json: `results/final_probe_bf16prep.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.733x (target 0.900x)
- JAX tok/s: 85.29
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 19.45

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 85.29 | 116.37 | 0.733x | 104.74 | 19.45 | 78.02 | 1.093x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.29 s | 0.45 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 388.12 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 385.76 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 287.19 ms | 1972.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 55.84 ms | 1515.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 27.87 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 25.84 ms | 194.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 14.84 ms | 22.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 11.14 ms | 256.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.29 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.64 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | loop_add_fusion_18 | 24.67 ms | 1152 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 771.42 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 748.35 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 748.26 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 746.86 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 417.36 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: exact_generated_token_match, minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 27.87 ms | 427.55 ms | -399.69 ms | 0.065x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 388.12 ms | 280.56 ms | 107.57 ms | 1.383x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 385.76 ms | 289.24 ms | 96.52 ms | 1.334x | 59.0 | 140 | -81.0 |
| command_buffer::execute | 287.19 ms | 229.21 ms | 57.98 ms | 1.253x | 1972.0 | 1936 | 36.0 |
| command_buffer::update | 25.84 ms | 10.46 ms | 15.38 ms | 2.470x | 194.0 | 195 | -1.0 |
| MemcpyD2D | 16.04 ms | 30.39 ms | -14.35 ms | 0.528x | 475.0 | 655 | -180.0 |
| transpose | 55.84 ms | 47.30 ms | 8.54 ms | 1.181x | 1515.0 | 312 | 1203.0 |
| gather | 19.62 ms | 14.55 ms | 5.07 ms | 1.348x | 121.0 | 103 | 18.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/final_probe_bf16prep.json`
- report: `results/final_probe_bf16prep.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.733x
- JAX/reference: 1.093x
- TTFT delta vs reference: -289.34 ms
- ITL delta vs reference: -5.96 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 27.87 ms, reference 427.55 ms, delta -399.69 ms, ratio 0.065x, count delta 48.0
- `forward_step_token_ids_jit`: current 388.12 ms, reference 280.56 ms, delta 107.57 ms, ratio 1.383x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 385.76 ms, reference 289.24 ms, delta 96.52 ms, ratio 1.334x, count delta -81.0
- `command_buffer::execute`: current 287.19 ms, reference 229.21 ms, delta 57.98 ms, ratio 1.253x, count delta 36.0
- `command_buffer::update`: current 25.84 ms, reference 10.46 ms, delta 15.38 ms, ratio 2.470x, count delta -1.0
- `MemcpyD2D`: current 16.04 ms, reference 30.39 ms, delta -14.35 ms, ratio 0.528x, count delta -180.0
- `transpose`: current 55.84 ms, reference 47.30 ms, delta 8.54 ms, ratio 1.181x, count delta 1203.0
- `gather`: current 19.62 ms, reference 14.55 ms, delta 5.07 ms, ratio 1.348x, count delta 18.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
