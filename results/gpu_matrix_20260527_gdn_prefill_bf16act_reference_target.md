# GPU Matrix Report

- created_at_utc: `20260527_095335`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_095335`
- output_json: `results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.771x (target 0.900x)
- JAX tok/s: 89.74
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 14.99

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 89.74 | 116.37 | 0.771x | 104.74 | 14.99 | 78.02 | 1.150x |

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
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 400.83 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 400.70 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 273.25 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 270.94 ms | 44.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 225.77 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.90 ms | 259.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 15.33 ms | 276.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.30 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_742 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.93 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.61 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.64 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:373 generate_with_trace | 713.16 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:287 iter_generate | 713.02 ms | 70 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:142 step | 711.91 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3703 run | 686.06 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:1865 _run_main_and_sample | 685.20 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transpose | 15.33 ms | 47.30 ms | -31.97 ms | 0.324x | 276.0 | 312 | -36.0 |
| array.py:325 tolist | 400.83 ms | 427.68 ms | -26.86 ms | 0.937x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 400.70 ms | 427.55 ms | -26.85 ms | 0.937x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 270.94 ms | 289.24 ms | -18.29 ms | 0.937x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.30 ms | 30.39 ms | -12.09 ms | 0.602x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 273.25 ms | 280.56 ms | -7.31 ms | 0.974x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 225.77 ms | 229.21 ms | -3.44 ms | 0.985x | 1954.0 | 1936 | 18.0 |
| command_buffer::update | 9.60 ms | 10.46 ms | -0.87 ms | 0.917x | 181.0 | 195 | -14.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.json`
- report: `results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.771x
- JAX/reference: 1.150x
- TTFT delta vs reference: -41.83 ms
- ITL delta vs reference: -4.50 ms
- profile movement to explain:
- `transpose`: current 15.33 ms, reference 47.30 ms, delta -31.97 ms, ratio 0.324x, count delta -36.0
- `array.py:325 tolist`: current 400.83 ms, reference 427.68 ms, delta -26.86 ms, ratio 0.937x, count delta 0.0
- `np.asarray(jax.Array)`: current 400.70 ms, reference 427.55 ms, delta -26.85 ms, ratio 0.937x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 270.94 ms, reference 289.24 ms, delta -18.29 ms, ratio 0.937x, count delta -96.0
- `MemcpyD2D`: current 18.30 ms, reference 30.39 ms, delta -12.09 ms, ratio 0.602x, count delta -192.0
- `forward_step_token_ids_jit`: current 273.25 ms, reference 280.56 ms, delta -7.31 ms, ratio 0.974x, count delta 0.0
- `command_buffer::execute`: current 225.77 ms, reference 229.21 ms, delta -3.44 ms, ratio 0.985x, count delta 18.0
- `command_buffer::update`: current 9.60 ms, reference 10.46 ms, delta -0.87 ms, ratio 0.917x, count delta -14.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
