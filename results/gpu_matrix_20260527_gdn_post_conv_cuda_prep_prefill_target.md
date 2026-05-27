# GPU Matrix Report

- created_at_utc: `20260527_072219`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_072219`
- output_json: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.520x (target 0.900x)
- JAX tok/s: 60.46
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 44.28

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 60.46 | 116.37 | 0.520x | 104.74 | 44.28 | 78.02 | 0.775x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.89 s | 0.17 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 965.76 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 965.64 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 55.85 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 53.49 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 16.31 ms | 208.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 15.21 ms | 225.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 11.07 ms | 205.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 9.89 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void (anonymous namespace)::Fp32GdnPrefillChunk32Kernel<64>(float const*, float const*, float const*, float const*, float const*, int const*, float const*, float*, float*, long, long, long, long, long) | 500.69 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_469 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.92 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_459 | 36.60 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 1058.56 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:287 iter_generate | 1058.40 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 1057.42 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 1033.00 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 1032.16 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 965.64 ms | 427.55 ms | 538.08 ms | 2.259x | 16.0 | 16 | 0.0 |
| array.py:325 tolist | 965.76 ms | 427.68 ms | 538.08 ms | 2.258x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 53.49 ms | 289.24 ms | -235.75 ms | 0.185x | 44.0 | 140 | -96.0 |
| forward_step_token_ids_jit | 55.85 ms | 280.56 ms | -224.71 ms | 0.199x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 16.31 ms | 229.21 ms | -212.90 ms | 0.071x | 208.0 | 1936 | -1728.0 |
| transpose | 15.21 ms | 47.30 ms | -32.09 ms | 0.321x | 225.0 | 312 | -87.0 |
| MemcpyD2D | 13.27 ms | 30.39 ms | -17.12 ms | 0.437x | 409.0 | 655 | -246.0 |
| command_buffer::update | 8.54 ms | 10.46 ms | -1.92 ms | 0.816x | 181.0 | 195 | -14.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.520x
- JAX/reference: 0.775x
- TTFT delta vs reference: 305.08 ms
- ITL delta vs reference: -4.59 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 965.64 ms, reference 427.55 ms, delta 538.08 ms, ratio 2.259x, count delta 0.0
- `array.py:325 tolist`: current 965.76 ms, reference 427.68 ms, delta 538.08 ms, ratio 2.258x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 53.49 ms, reference 289.24 ms, delta -235.75 ms, ratio 0.185x, count delta -96.0
- `forward_step_token_ids_jit`: current 55.85 ms, reference 280.56 ms, delta -224.71 ms, ratio 0.199x, count delta 0.0
- `command_buffer::execute`: current 16.31 ms, reference 229.21 ms, delta -212.90 ms, ratio 0.071x, count delta -1728.0
- `transpose`: current 15.21 ms, reference 47.30 ms, delta -32.09 ms, ratio 0.321x, count delta -87.0
- `MemcpyD2D`: current 13.27 ms, reference 30.39 ms, delta -17.12 ms, ratio 0.437x, count delta -246.0
- `command_buffer::update`: current 8.54 ms, reference 10.46 ms, delta -1.92 ms, ratio 0.816x, count delta -14.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
