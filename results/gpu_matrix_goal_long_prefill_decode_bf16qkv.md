# GPU Matrix Report

- created_at_utc: `20260530_081738`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_081738`
- output_json: `results/gpu_matrix_goal_long_prefill_decode_bf16qkv.json`
- jax_python: `/root/miniconda3/bin/python` (available: yes)

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
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | no | no | 93.60 | 116.37 | 0.804x | 104.74 | 11.14 | 78.02 | 1.200x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 15.0 | 4.0 | 5120.0 | 0.26 s | 0.42 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 338.80 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 338.00 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 253.74 ms | 2230.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 28.20 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | MemcpyD2D | 15.92 ms | 274.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | transpose | 15.35 ms | 276.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 11.80 ms | 34.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 10.46 ms | 448.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_general_744 | 57.31 ms | 48 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_general_729 | 36.59 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.91 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.90 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_2 | 26.05 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 682.28 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 682.23 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 681.26 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 675.41 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 360.80 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 28.20 ms | 427.55 ms | -399.35 ms | 0.066x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 338.80 ms | 280.56 ms | 58.25 ms | 1.208x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 338.00 ms | 289.24 ms | 48.76 ms | 1.169x | 59.0 | 140 | -81.0 |
| transpose | 15.35 ms | 47.30 ms | -31.94 ms | 0.325x | 276.0 | 312 | -36.0 |
| command_buffer::execute | 253.74 ms | 229.21 ms | 24.53 ms | 1.107x | 2230.0 | 1936 | 294.0 |
| MemcpyD2D | 23.76 ms | 30.39 ms | -6.63 ms | 0.782x | 493.0 | 655 | -162.0 |
| gather | 16.46 ms | 14.55 ms | 1.91 ms | 1.131x | 115.0 | 103 | 12.0 |
| command_buffer::update | 10.46 ms | 10.46 ms | -0.00 ms | 1.000x | 448.0 | 195 | 253.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_goal_long_prefill_decode_bf16qkv.json`
- report: `results/gpu_matrix_goal_long_prefill_decode_bf16qkv.md`
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
