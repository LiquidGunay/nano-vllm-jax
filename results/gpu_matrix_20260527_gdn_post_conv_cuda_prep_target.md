# GPU Matrix Report

- created_at_utc: `20260527_071901`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_071901`
- output_json: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.755x (target 0.900x)
- JAX tok/s: 87.80
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 16.93

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 87.80 | 116.37 | 0.755x | 104.74 | 16.93 | 78.02 | 1.125x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.56 s | 0.17 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 410.36 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 410.24 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 280.81 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 278.53 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 234.51 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 15.22 ms | 225.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 11.13 ms | 241.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 9.33 ms | 195.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_742 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.57 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 728.88 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:287 iter_generate | 728.75 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 727.77 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 702.20 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 701.41 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transpose | 15.22 ms | 47.30 ms | -32.08 ms | 0.322x | 225.0 | 312 | -87.0 |
| array.py:325 tolist | 410.36 ms | 427.68 ms | -17.32 ms | 0.959x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 410.24 ms | 427.55 ms | -17.31 ms | 0.960x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 13.42 ms | 30.39 ms | -16.97 ms | 0.442x | 445.0 | 655 | -210.0 |
| PjRtCApiLoadedExecutable::Execute | 278.53 ms | 289.24 ms | -10.70 ms | 0.963x | 44.0 | 140 | -96.0 |
| command_buffer::execute | 234.51 ms | 229.21 ms | 5.30 ms | 1.023x | 1954.0 | 1936 | 18.0 |
| command_buffer::update | 9.33 ms | 10.46 ms | -1.13 ms | 0.892x | 195.0 | 195 | 0.0 |
| gather | 13.83 ms | 14.55 ms | -0.73 ms | 0.950x | 121.0 | 103 | 18.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.755x
- JAX/reference: 1.125x
- TTFT delta vs reference: -24.59 ms
- ITL delta vs reference: -4.54 ms
- profile movement to explain:
- `transpose`: current 15.22 ms, reference 47.30 ms, delta -32.08 ms, ratio 0.322x, count delta -87.0
- `array.py:325 tolist`: current 410.36 ms, reference 427.68 ms, delta -17.32 ms, ratio 0.959x, count delta 0.0
- `np.asarray(jax.Array)`: current 410.24 ms, reference 427.55 ms, delta -17.31 ms, ratio 0.960x, count delta 0.0
- `MemcpyD2D`: current 13.42 ms, reference 30.39 ms, delta -16.97 ms, ratio 0.442x, count delta -210.0
- `PjRtCApiLoadedExecutable::Execute`: current 278.53 ms, reference 289.24 ms, delta -10.70 ms, ratio 0.963x, count delta -96.0
- `command_buffer::execute`: current 234.51 ms, reference 229.21 ms, delta 5.30 ms, ratio 1.023x, count delta 18.0
- `command_buffer::update`: current 9.33 ms, reference 10.46 ms, delta -1.13 ms, ratio 0.892x, count delta 0.0
- `gather`: current 13.83 ms, reference 14.55 ms, delta -0.73 ms, ratio 0.950x, count delta 18.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
