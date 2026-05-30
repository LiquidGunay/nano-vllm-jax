# GPU Matrix Report

- created_at_utc: `20260530_124038`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_moveon_quick3`
- output_json: `results/gpu_matrix_decode_moveon_quick3.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | no | no | 87.42 | 213.54 | 0.409x | 192.18 | 104.76 | 176.68 | 0.495x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1.0 | 127.0 | 1.0 | 128.0 | 0.05 s | 1.41 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | forward_step_token_ids_jit | 686.11 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | PjRtCApiLoadedExecutable::Execute | 646.95 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | command_buffer::execute | 201.18 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | command_buffer::update | 175.02 ms | 2413.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | np.asarray(jax.Array) | 30.61 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | gpu | gather | 12.57 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | gather | 11.36 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | MemcpyD2D | 4.81 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | gemm_fusion_dot_265 | 227.81 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_92 | 158.47 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_62 | 118.24 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_116 | 99.46 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_140 | 83.99 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1459.27 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1459.22 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:161 step | 1452.97 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | PjitFunction(compiled) | 1351.88 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $model_runner.py:3737 run | 847.99 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 686.11 ms | 390.22 ms | 295.90 ms | 1.758x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 646.95 ms | 371.41 ms | 275.53 ms | 1.742x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 201.18 ms | 119.11 ms | 82.07 ms | 1.689x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 175.02 ms | 105.38 ms | 69.64 ms | 1.661x | 2413.0 | 127 | 2286.0 |
| np.asarray(jax.Array) | 30.61 ms | 6.36 ms | 24.25 ms | 4.809x | 128.0 | 128 | 0.0 |
| gather | 23.93 ms | 17.30 ms | 6.64 ms | 1.384x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.82 ms | 7.30 ms | -1.48 ms | 0.797x | 401.0 | 384 | 17.0 |
| transpose | 0.07 ms | 0.08 ms | -0.00 ms | 0.979x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 686.11 ms | 390.22 ms | 295.90 ms | 1.758x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 646.95 ms | 371.41 ms | 275.53 ms | 1.742x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 201.18 ms | 119.11 ms | 82.07 ms | 1.689x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 175.02 ms | 105.38 ms | 69.64 ms | 1.661x | 2413.0 | 127 | 2286.0 |
| cpu | np.asarray(jax.Array) | 30.61 ms | 6.36 ms | 24.25 ms | 4.809x | 128.0 | 128 | 0.0 |
| gpu | gather | 12.57 ms | 5.33 ms | 7.24 ms | 2.357x | 1585.0 | 1585 | 0.0 |
| cpu | MemcpyD2D | 4.81 ms | 6.33 ms | -1.51 ms | 0.761x | 175.0 | 175 | 0.0 |
| cpu | gather | 11.36 ms | 11.96 ms | -0.60 ms | 0.950x | 18.0 | 18 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_moveon_quick3.json`
- report: `results/gpu_matrix_decode_moveon_quick3.md`
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
