# GPU Matrix Report

- created_at_utc: `20260530_123848`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_moveon_quick2_novllm`
- output_json: `results/gpu_matrix_decode_moveon_quick2_novllm.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | no | no | 130.89 | - | - | - | - | 176.01 | 0.744x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1.0 | 127.0 | 1.0 | 128.0 | 0.05 s | 0.92 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | forward_step_token_ids_jit | 589.96 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | PjRtCApiLoadedExecutable::Execute | 543.78 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | command_buffer::execute | 169.93 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | command_buffer::update | 148.59 ms | 2400.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | gather | 16.09 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | np.asarray(jax.Array) | 9.48 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | gpu | gather | 5.82 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | MemcpyD2D | 4.71 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | gemm_fusion_dot_265 | 150.00 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_92 | 108.77 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_62 | 79.03 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_116 | 66.24 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_140 | 39.86 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | PjitFunction(compiled) | 1160.76 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:632 generate_with_trace | 973.60 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 973.55 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:161 step | 968.15 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $model_runner.py:3737 run | 734.15 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4: failed checks: minimum_repeats, vllm_reference_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 589.96 ms | 394.16 ms | 195.80 ms | 1.497x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 543.78 ms | 373.51 ms | 170.27 ms | 1.456x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 169.93 ms | 111.82 ms | 58.11 ms | 1.520x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 148.59 ms | 111.57 ms | 37.02 ms | 1.332x | 2400.0 | 126 | 2274.0 |
| np.asarray(jax.Array) | 9.48 ms | 7.25 ms | 2.24 ms | 1.309x | 128.0 | 128 | 0.0 |
| gather | 21.91 ms | 19.82 ms | 2.09 ms | 1.106x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.71 ms | 5.16 ms | 0.54 ms | 1.105x | 401.0 | 384 | 17.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.995x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 589.96 ms | 394.16 ms | 195.80 ms | 1.497x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 543.78 ms | 373.51 ms | 170.27 ms | 1.456x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 169.93 ms | 111.82 ms | 58.11 ms | 1.520x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 148.59 ms | 111.57 ms | 37.02 ms | 1.332x | 2400.0 | 126 | 2274.0 |
| cpu | np.asarray(jax.Array) | 9.48 ms | 7.25 ms | 2.24 ms | 1.309x | 128.0 | 128 | 0.0 |
| cpu | gather | 16.09 ms | 14.48 ms | 1.61 ms | 1.111x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 4.71 ms | 4.19 ms | 0.52 ms | 1.123x | 175.0 | 175 | 0.0 |
| gpu | gather | 5.82 ms | 5.34 ms | 0.48 ms | 1.091x | 1585.0 | 1585 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_moveon_quick2_novllm.json`
- report: `results/gpu_matrix_decode_moveon_quick2_novllm.md`
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
