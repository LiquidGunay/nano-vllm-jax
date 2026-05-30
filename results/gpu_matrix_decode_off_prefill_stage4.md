# GPU Matrix Report

- created_at_utc: `20260530_092709`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_090958`
- output_json: `results/gpu_matrix_decode_off_prefill_stage4.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | no | no | 106.09 | 218.67 | 0.485x | 196.80 | 90.71 | 129.08 | 0.822x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.05 s | 1.14 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 644.17 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 617.71 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 196.85 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 162.62 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 93.57 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 16.13 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 5.47 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 4.24 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.55 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.47 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.05 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.69 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.45 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $threading.py:604 wait | 1522.77 ms | 2 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $threading.py:288 wait | 1522.76 ms | 2 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 1267.64 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1199.15 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1199.11 ms | 1 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 93.57 ms | 6.73 ms | 86.84 ms | 13.893x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 617.71 ms | 538.79 ms | 78.93 ms | 1.146x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 644.17 ms | 573.28 ms | 70.89 ms | 1.124x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 196.85 ms | 159.51 ms | 37.34 ms | 1.234x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 162.62 ms | 172.36 ms | -9.75 ms | 0.943x | 2394.0 | 126 | 2268.0 |
| gather | 21.60 ms | 20.35 ms | 1.24 ms | 1.061x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.05 ms | 6.15 ms | -1.10 ms | 0.821x | 384.0 | 384 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.012x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | np.asarray(jax.Array) | 93.57 ms | 6.73 ms | 86.84 ms | 13.893x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 617.71 ms | 538.79 ms | 78.93 ms | 1.146x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 644.17 ms | 573.28 ms | 70.89 ms | 1.124x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 196.85 ms | 159.51 ms | 37.34 ms | 1.234x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 162.62 ms | 172.36 ms | -9.75 ms | 0.943x | 2394.0 | 126 | 2268.0 |
| cpu | gather | 16.13 ms | 14.89 ms | 1.23 ms | 1.083x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 4.24 ms | 5.17 ms | -0.93 ms | 0.820x | 175.0 | 175 | 0.0 |
| gpu | MemcpyD2D | 0.81 ms | 0.98 ms | -0.17 ms | 0.828x | 209.0 | 209 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_off_prefill_stage4.json`
- report: `results/gpu_matrix_decode_off_prefill_stage4.md`
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
