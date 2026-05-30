# GPU Matrix Report

- created_at_utc: `20260530_122130`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/decode_focus_moveon`
- output_json: `results/gpu_matrix_decode_focus_moveon.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | no | no | 136.26 | 218.84 | 0.623x | 196.96 | 60.70 | 113.69 | 1.199x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1.0 | 127.0 | 1.0 | 128.0 | 0.06 s | 0.88 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | forward_step_token_ids_jit | 600.22 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | PjRtCApiLoadedExecutable::Execute | 577.03 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | command_buffer::execute | 160.12 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | command_buffer::update | 150.97 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | gather | 12.59 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | np.asarray(jax.Array) | 5.55 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | gpu | gather | 5.33 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | cpu | MemcpyD2D | 4.56 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_116 | 64.08 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | input_reduce_fusion_62 | 62.80 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | gpu | fusion_922 | 37.26 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | PjitFunction(compiled) | 1183.26 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $threading.py:604 wait | 1130.20 ms | 2 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $threading.py:288 wait | 1130.19 ms | 2 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:632 generate_with_trace | 936.56 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4 | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 936.52 ms | 1 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 600.22 ms | 645.02 ms | -44.80 ms | 0.931x | 128.0 | 128 | 0.0 |
| command_buffer::update | 150.97 ms | 194.34 ms | -43.37 ms | 0.777x | 2394.0 | 126 | 2268.0 |
| command_buffer::execute | 160.12 ms | 191.86 ms | -31.74 ms | 0.835x | 2517.0 | 231 | 2286.0 |
| PjRtCApiLoadedExecutable::Execute | 577.03 ms | 607.03 ms | -30.00 ms | 0.951x | 283.0 | 283 | 0.0 |
| gather | 17.92 ms | 15.86 ms | 2.06 ms | 1.130x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.54 ms | 6.62 ms | -1.08 ms | 0.837x | 401.0 | 384 | 17.0 |
| np.asarray(jax.Array) | 5.55 ms | 5.36 ms | 0.19 ms | 1.036x | 128.0 | 128 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.998x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_gdn_backend_bf16_w8_b32_s4`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 600.22 ms | 645.02 ms | -44.80 ms | 0.931x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::update | 150.97 ms | 194.34 ms | -43.37 ms | 0.777x | 2394.0 | 126 | 2268.0 |
| cpu | command_buffer::execute | 160.12 ms | 191.86 ms | -31.74 ms | 0.835x | 2517.0 | 231 | 2286.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 577.03 ms | 607.03 ms | -30.00 ms | 0.951x | 283.0 | 283 | 0.0 |
| cpu | gather | 12.59 ms | 10.56 ms | 2.04 ms | 1.193x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 4.56 ms | 5.64 ms | -1.08 ms | 0.808x | 175.0 | 175 | 0.0 |
| cpu | np.asarray(jax.Array) | 5.55 ms | 5.36 ms | 0.19 ms | 1.036x | 128.0 | 128 | 0.0 |
| gpu | gather | 5.33 ms | 5.30 ms | 0.02 ms | 1.005x | 1585.0 | 1585 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_focus_moveon.json`
- report: `results/gpu_matrix_decode_focus_moveon.md`
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
