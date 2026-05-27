# GPU Matrix Report

- created_at_utc: `20260527_092620`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_092620`
- output_json: `results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.548x (target 0.900x)
- JAX tok/s: 63.82
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 40.92

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 63.82 | 116.37 | 0.548x | 104.74 | 40.92 | 78.02 | 0.818x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.83 s | 0.17 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 912.08 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 911.97 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 51.68 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 50.01 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 16.53 ms | 208.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.76 ms | 223.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 15.22 ms | 222.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 9.86 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void (anonymous namespace)::Fp32GdnPrefillChunk32Kernel<32, true>(float const*, float const*, float const*, float const*, float const*, int const*, float const*, float*, float*, long, long, long, long, long) | 452.44 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_471 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.94 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_459 | 36.61 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.63 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 1002.86 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:287 iter_generate | 1002.73 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 1001.68 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 975.41 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 974.57 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 911.97 ms | 427.55 ms | 484.42 ms | 2.133x | 16.0 | 16 | 0.0 |
| array.py:325 tolist | 912.08 ms | 427.68 ms | 484.40 ms | 2.133x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 50.01 ms | 289.24 ms | -239.22 ms | 0.173x | 44.0 | 140 | -96.0 |
| forward_step_token_ids_jit | 51.68 ms | 280.56 ms | -228.87 ms | 0.184x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 16.53 ms | 229.21 ms | -212.68 ms | 0.072x | 208.0 | 1936 | -1728.0 |
| transpose | 15.22 ms | 47.30 ms | -32.08 ms | 0.322x | 222.0 | 312 | -90.0 |
| MemcpyD2D | 17.98 ms | 30.39 ms | -12.41 ms | 0.592x | 427.0 | 655 | -228.0 |
| command_buffer::update | 8.53 ms | 10.46 ms | -1.93 ms | 0.815x | 181.0 | 195 | -14.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.548x
- JAX/reference: 0.818x
- TTFT delta vs reference: 250.76 ms
- ITL delta vs reference: -4.62 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 911.97 ms, reference 427.55 ms, delta 484.42 ms, ratio 2.133x, count delta 0.0
- `array.py:325 tolist`: current 912.08 ms, reference 427.68 ms, delta 484.40 ms, ratio 2.133x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 50.01 ms, reference 289.24 ms, delta -239.22 ms, ratio 0.173x, count delta -96.0
- `forward_step_token_ids_jit`: current 51.68 ms, reference 280.56 ms, delta -228.87 ms, ratio 0.184x, count delta 0.0
- `command_buffer::execute`: current 16.53 ms, reference 229.21 ms, delta -212.68 ms, ratio 0.072x, count delta -1728.0
- `transpose`: current 15.22 ms, reference 47.30 ms, delta -32.08 ms, ratio 0.322x, count delta -90.0
- `MemcpyD2D`: current 17.98 ms, reference 30.39 ms, delta -12.41 ms, ratio 0.592x, count delta -228.0
- `command_buffer::update`: current 8.53 ms, reference 10.46 ms, delta -1.93 ms, ratio 0.815x, count delta -14.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
