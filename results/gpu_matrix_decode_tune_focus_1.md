# GPU Matrix Report

- created_at_utc: `20260530_080111`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_focused_tune_1`
- output_json: `results/gpu_matrix_decode_tune_focus_1.json`
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
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | no | no | 88.28 | 116.37 | 0.759x | 104.74 | 16.45 | 78.02 | 1.132x |

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
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 365.31 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 363.38 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 241.09 ms | 2224.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | gpu | transpose | 73.13 ms | 1547.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 27.72 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 26.07 ms | 219.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 21.52 ms | 460.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | gpu | MemcpyD2D | 15.92 ms | 274.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.27 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.56 ms | 17 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.59 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 727.67 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 722.90 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 722.84 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 721.60 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 391.29 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 27.72 ms | 427.55 ms | -399.84 ms | 0.065x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 365.31 ms | 280.56 ms | 84.75 ms | 1.302x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 363.38 ms | 289.24 ms | 74.15 ms | 1.256x | 59.0 | 140 | -81.0 |
| transpose | 73.13 ms | 47.30 ms | 25.83 ms | 1.546x | 1547.0 | 312 | 1235.0 |
| command_buffer::execute | 241.09 ms | 229.21 ms | 11.88 ms | 1.052x | 2224.0 | 1936 | 288.0 |
| MemcpyD2D | 41.99 ms | 30.39 ms | 11.60 ms | 1.382x | 493.0 | 655 | -162.0 |
| command_buffer::update | 21.52 ms | 10.46 ms | 11.06 ms | 2.057x | 460.0 | 195 | 265.0 |
| gather | 14.31 ms | 14.55 ms | -0.24 ms | 0.983x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_focus_1.json`
- report: `results/gpu_matrix_decode_tune_focus_1.md`
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
