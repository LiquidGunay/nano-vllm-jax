# GPU Matrix Report

- created_at_utc: `20260527_134735`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_134735`
- output_json: `results/gpu_matrix_20260527_gdn_prefill_bf16qkv_fp32out_reference_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.770x (target 0.900x)
- JAX tok/s: 89.60
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 15.13

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 89.60 | 116.37 | 0.770x | 104.74 | 15.13 | 78.02 | 1.148x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 399.24 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 399.11 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 274.58 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 272.26 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 226.21 ms | 1936.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.90 ms | 259.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 15.35 ms | 276.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.09 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.91 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:377 generate_with_trace | 714.26 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:291 iter_generate | 714.12 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:146 step | 712.88 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3724 run | 685.86 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1886 _run_main_and_sample | 685.05 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transpose | 15.35 ms | 47.30 ms | -31.95 ms | 0.325x | 276.0 | 312 | -36.0 |
| np.asarray(jax.Array) | 399.11 ms | 427.55 ms | -28.45 ms | 0.933x | 16.0 | 16 | 0.0 |
| array.py:325 tolist | 399.24 ms | 427.68 ms | -28.44 ms | 0.934x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 272.26 ms | 289.24 ms | -16.98 ms | 0.941x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.27 ms | 30.39 ms | -12.12 ms | 0.601x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 274.58 ms | 280.56 ms | -5.97 ms | 0.979x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 226.21 ms | 229.21 ms | -3.00 ms | 0.987x | 1936.0 | 1936 | 0.0 |
| command_buffer::update | 9.48 ms | 10.46 ms | -0.98 ms | 0.906x | 183.0 | 195 | -12.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_prefill_bf16qkv_fp32out_reference_target.json`
- report: `results/gpu_matrix_20260527_gdn_prefill_bf16qkv_fp32out_reference_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.770x
- JAX/reference: 1.148x
- TTFT delta vs reference: -42.30 ms
- ITL delta vs reference: -4.40 ms
- profile movement to explain:
- `transpose`: current 15.35 ms, reference 47.30 ms, delta -31.95 ms, ratio 0.325x, count delta -36.0
- `np.asarray(jax.Array)`: current 399.11 ms, reference 427.55 ms, delta -28.45 ms, ratio 0.933x, count delta 0.0
- `array.py:325 tolist`: current 399.24 ms, reference 427.68 ms, delta -28.44 ms, ratio 0.934x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 272.26 ms, reference 289.24 ms, delta -16.98 ms, ratio 0.941x, count delta -96.0
- `MemcpyD2D`: current 18.27 ms, reference 30.39 ms, delta -12.12 ms, ratio 0.601x, count delta -192.0
- `forward_step_token_ids_jit`: current 274.58 ms, reference 280.56 ms, delta -5.97 ms, ratio 0.979x, count delta 0.0
- `command_buffer::execute`: current 226.21 ms, reference 229.21 ms, delta -3.00 ms, ratio 0.987x, count delta 0.0
- `command_buffer::update`: current 9.48 ms, reference 10.46 ms, delta -0.98 ms, ratio 0.906x, count delta -12.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
