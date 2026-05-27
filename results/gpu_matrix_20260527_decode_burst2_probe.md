# GPU Matrix Report

- created_at_utc: `20260527_062510`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_062510`
- output_json: `results/gpu_matrix_20260527_decode_burst2_probe.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.048x (target 0.900x)
- JAX tok/s: 5.57
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 99.17

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 5.57 | 116.37 | 0.048x | 104.74 | 99.17 | 78.02 | 0.071x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 8.0 | 4.0 | 5120.0 | 0.54 s | 10.95 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | array.py:325 tolist | 294.41 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 294.39 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 227.98 ms | 29.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 225.57 ms | 1.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 207.67 ms | 1741.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 134.12 ms | 242.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | transpose | 40.01 ms | 471.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 34.69 ms | 132.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 57.45 ms | 48 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 36.90 ms | 30 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 36.58 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 26.04 ms | 24 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 22010.79 ms | 4 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $traceback_util.py:191 reraise_with_filtered_traceback | 12595.34 ms | 2220 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:365 generate_with_trace | 11489.87 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:279 iter_generate | 11354.61 ms | 5 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:134 step | 11354.37 ms | 2 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array.py:325 tolist | 294.41 ms | 427.68 ms | -133.28 ms | 0.688x | 1.0 | 16 | -15.0 |
| np.asarray(jax.Array) | 294.39 ms | 427.55 ms | -133.16 ms | 0.689x | 1.0 | 16 | -15.0 |
| gather | 138.73 ms | 14.55 ms | 124.17 ms | 9.531x | 293.0 | 103 | 190.0 |
| PjRtCApiLoadedExecutable::Execute | 227.98 ms | 289.24 ms | -61.26 ms | 0.788x | 29.0 | 140 | -111.0 |
| forward_step_token_ids_jit | 225.57 ms | 280.56 ms | -54.99 ms | 0.804x | 1.0 | 16 | -15.0 |
| transpose | 74.69 ms | 47.30 ms | 27.39 ms | 1.579x | 603.0 | 312 | 291.0 |
| MemcpyD2D | 6.79 ms | 30.39 ms | -23.60 ms | 0.223x | 103.0 | 655 | -552.0 |
| command_buffer::execute | 207.67 ms | 229.21 ms | -21.54 ms | 0.906x | 1741.0 | 1936 | -195.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_decode_burst2_probe.json`
- report: `results/gpu_matrix_20260527_decode_burst2_probe.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.048x
- JAX/reference: 0.071x
- TTFT delta vs reference: -48.27 ms
- ITL delta vs reference: 0.43 ms
- profile movement to explain:
- `array.py:325 tolist`: current 294.41 ms, reference 427.68 ms, delta -133.28 ms, ratio 0.688x, count delta -15.0
- `np.asarray(jax.Array)`: current 294.39 ms, reference 427.55 ms, delta -133.16 ms, ratio 0.689x, count delta -15.0
- `gather`: current 138.73 ms, reference 14.55 ms, delta 124.17 ms, ratio 9.531x, count delta 190.0
- `PjRtCApiLoadedExecutable::Execute`: current 227.98 ms, reference 289.24 ms, delta -61.26 ms, ratio 0.788x, count delta -111.0
- `forward_step_token_ids_jit`: current 225.57 ms, reference 280.56 ms, delta -54.99 ms, ratio 0.804x, count delta -15.0
- `transpose`: current 74.69 ms, reference 47.30 ms, delta 27.39 ms, ratio 1.579x, count delta 291.0
- `MemcpyD2D`: current 6.79 ms, reference 30.39 ms, delta -23.60 ms, ratio 0.223x, count delta -552.0
- `command_buffer::execute`: current 207.67 ms, reference 229.21 ms, delta -21.54 ms, ratio 0.906x, count delta -195.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
