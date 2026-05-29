# GPU Matrix Report

- created_at_utc: `20260528_072208`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260528_072208`
- output_json: `results/gpu_matrix_20260528_gdn_prefill_triton_fla_prep_bf16_device_carry_diag_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.797x (target 0.900x)
- JAX tok/s: 92.80
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 11.94

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 92.80 | 116.37 | 0.797x | 104.74 | 11.94 | 78.02 | 1.189x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.27 s | 0.42 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 318.96 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 298.41 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 245.48 ms | 1954.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 32.21 ms | 261.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 27.79 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | MemcpyD2D | 23.47 ms | 219.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 11.18 ms | 256.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 10.23 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.46 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.93 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:554 generate_with_trace | 689.67 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:583 _generate_with_trace_deferred_tokens | 689.62 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:157 step | 688.80 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 594.96 ms | 32 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 338.75 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: exact_generated_token_match, minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 27.79 ms | 427.55 ms | -399.76 ms | 0.065x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 318.96 ms | 289.24 ms | 29.72 ms | 1.103x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 298.41 ms | 280.56 ms | 17.86 ms | 1.064x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 245.48 ms | 229.21 ms | 16.27 ms | 1.071x | 1954.0 | 1936 | 18.0 |
| transpose | 32.21 ms | 47.30 ms | -15.09 ms | 0.681x | 261.0 | 312 | -51.0 |
| MemcpyD2D | 34.65 ms | 30.39 ms | 4.27 ms | 1.140x | 475.0 | 655 | -180.0 |
| command_buffer::update | 8.97 ms | 10.46 ms | -1.49 ms | 0.858x | 183.0 | 195 | -12.0 |
| gather | 14.95 ms | 14.55 ms | 0.40 ms | 1.027x | 121.0 | 103 | 18.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260528_gdn_prefill_triton_fla_prep_bf16_device_carry_diag_target.json`
- report: `results/gpu_matrix_20260528_gdn_prefill_triton_fla_prep_bf16_device_carry_diag_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.797x
- JAX/reference: 1.189x
- TTFT delta vs reference: -311.54 ms
- ITL delta vs reference: -7.07 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 27.79 ms, reference 427.55 ms, delta -399.76 ms, ratio 0.065x, count delta 48.0
- `PjRtCApiLoadedExecutable::Execute`: current 318.96 ms, reference 289.24 ms, delta 29.72 ms, ratio 1.103x, count delta -81.0
- `forward_step_token_ids_jit`: current 298.41 ms, reference 280.56 ms, delta 17.86 ms, ratio 1.064x, count delta 0.0
- `command_buffer::execute`: current 245.48 ms, reference 229.21 ms, delta 16.27 ms, ratio 1.071x, count delta 18.0
- `transpose`: current 32.21 ms, reference 47.30 ms, delta -15.09 ms, ratio 0.681x, count delta -51.0
- `MemcpyD2D`: current 34.65 ms, reference 30.39 ms, delta 4.27 ms, ratio 1.140x, count delta -180.0
- `command_buffer::update`: current 8.97 ms, reference 10.46 ms, delta -1.49 ms, ratio 0.858x, count delta -12.0
- `gather`: current 14.95 ms, reference 14.55 ms, delta 0.40 ms, ratio 1.027x, count delta 18.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
