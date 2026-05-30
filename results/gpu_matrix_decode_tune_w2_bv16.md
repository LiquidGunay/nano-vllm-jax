# GPU Matrix Report

- created_at_utc: `20260530_080256`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/decode_tune_w2_bv16`
- output_json: `results/gpu_matrix_decode_tune_w2_bv16.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: -
- target_vllm_ratio_met: -
- JAX/vLLM: - (target 0.900x)
- JAX tok/s: -
- vLLM tok/s: -
- target tok/s: -
- gap to target tok/s: -

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | no | no | 88.63 | 116.37 | 0.762x | 104.74 | 16.11 | 78.02 | 1.136x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1.0 | 15.0 | 4.0 | 5120.0 | 0.25 s | 0.47 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 360.50 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 357.55 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 237.39 ms | 2224.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | gpu | transpose | 73.12 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 33.75 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 25.24 ms | 219.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 19.12 ms | 454.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | gpu | MemcpyD2D | 15.94 ms | 274.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.28 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.52 ms | 17 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.65 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.82 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 720.63 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 720.57 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 718.05 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 717.95 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 384.60 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 33.75 ms | 427.55 ms | -393.81 ms | 0.079x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 360.50 ms | 280.56 ms | 79.94 ms | 1.285x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 357.55 ms | 289.24 ms | 68.31 ms | 1.236x | 59.0 | 140 | -81.0 |
| transpose | 73.12 ms | 47.30 ms | 25.82 ms | 1.546x | 1547.0 | 312 | 1235.0 |
| MemcpyD2D | 41.18 ms | 30.39 ms | 10.79 ms | 1.355x | 493.0 | 655 | -162.0 |
| command_buffer::update | 19.12 ms | 10.46 ms | 8.66 ms | 1.828x | 454.0 | 195 | 259.0 |
| command_buffer::execute | 237.39 ms | 229.21 ms | 8.18 ms | 1.036x | 2224.0 | 1936 | 288.0 |
| gather | 14.48 ms | 14.55 ms | -0.07 ms | 0.995x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_w2_bv16.json`
- report: `results/gpu_matrix_decode_tune_w2_bv16.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: -
- target_vllm_ratio_met: -
- JAX/vLLM: -
- JAX/reference: -
- TTFT delta vs reference: -
- ITL delta vs reference: -
- profile movement to explain:
- No profile delta rows are available.
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
