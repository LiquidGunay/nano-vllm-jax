# GPU Matrix Report

- created_at_utc: `20260527_090034`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_090034`
- output_json: `results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.768x (target 0.900x)
- JAX tok/s: 89.37
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 15.37

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 89.37 | 116.37 | 0.768x | 104.74 | 15.37 | 78.02 | 1.145x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.54 s | 0.17 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 398.85 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 398.72 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 274.61 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 272.73 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 226.40 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.90 ms | 259.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 15.36 ms | 276.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.96 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.90 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.61 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.63 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 716.13 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:287 iter_generate | 715.99 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 711.47 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 685.54 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 684.74 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transpose | 15.36 ms | 47.30 ms | -31.94 ms | 0.325x | 276.0 | 312 | -36.0 |
| array.py:325 tolist | 398.85 ms | 427.68 ms | -28.83 ms | 0.933x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 398.72 ms | 427.55 ms | -28.83 ms | 0.933x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 272.73 ms | 289.24 ms | -16.51 ms | 0.943x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.39 ms | 30.39 ms | -12.00 ms | 0.605x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 274.61 ms | 280.56 ms | -5.94 ms | 0.979x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 226.40 ms | 229.21 ms | -2.81 ms | 0.988x | 1936.0 | 1936 | 0.0 |
| gather | 15.62 ms | 14.55 ms | 1.07 ms | 1.073x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.768x
- JAX/reference: 1.145x
- TTFT delta vs reference: -39.18 ms
- ITL delta vs reference: -4.47 ms
- profile movement to explain:
- `transpose`: current 15.36 ms, reference 47.30 ms, delta -31.94 ms, ratio 0.325x, count delta -36.0
- `array.py:325 tolist`: current 398.85 ms, reference 427.68 ms, delta -28.83 ms, ratio 0.933x, count delta 0.0
- `np.asarray(jax.Array)`: current 398.72 ms, reference 427.55 ms, delta -28.83 ms, ratio 0.933x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 272.73 ms, reference 289.24 ms, delta -16.51 ms, ratio 0.943x, count delta -96.0
- `MemcpyD2D`: current 18.39 ms, reference 30.39 ms, delta -12.00 ms, ratio 0.605x, count delta -192.0
- `forward_step_token_ids_jit`: current 274.61 ms, reference 280.56 ms, delta -5.94 ms, ratio 0.979x, count delta 0.0
- `command_buffer::execute`: current 226.40 ms, reference 229.21 ms, delta -2.81 ms, ratio 0.988x, count delta 0.0
- `gather`: current 15.62 ms, reference 14.55 ms, delta 1.07 ms, ratio 1.073x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
