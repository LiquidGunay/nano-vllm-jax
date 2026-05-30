# GPU Matrix Report

- created_at_utc: `20260530_073033`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_smoke`
- output_json: `results/gla_decode_heavy_smoke_check.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python3` (available: yes)

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 166.31 | - | - | - | - | 149.98 | 1.109x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.72 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 368.47 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 348.58 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 122.51 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 78.49 ms | 2413.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 10.25 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 6.21 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.45 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.48 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.24 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.33 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.80 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.08 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 766.90 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 766.85 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 763.35 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 723.51 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 465.10 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats, vllm_reference_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 368.47 ms | 290.86 ms | 77.61 ms | 1.267x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 348.58 ms | 273.82 ms | 74.76 ms | 1.273x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 122.51 ms | 98.57 ms | 23.94 ms | 1.243x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 78.49 ms | 65.75 ms | 12.73 ms | 1.194x | 2413.0 | 126 | 2287.0 |
| np.asarray(jax.Array) | 6.21 ms | 3.49 ms | 2.72 ms | 1.778x | 128.0 | 128 | 0.0 |
| transpose | 0.97 ms | 0.61 ms | 0.36 ms | 1.591x | 187.0 | 170 | 17.0 |
| gather | 15.70 ms | 15.60 ms | 0.10 ms | 1.006x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.55 ms | 4.59 ms | -0.04 ms | 0.991x | 401.0 | 384 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 368.47 ms | 290.86 ms | 77.61 ms | 1.267x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 348.58 ms | 273.82 ms | 74.76 ms | 1.273x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 122.51 ms | 98.57 ms | 23.94 ms | 1.243x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 78.49 ms | 65.75 ms | 12.73 ms | 1.194x | 2413.0 | 126 | 2287.0 |
| cpu | np.asarray(jax.Array) | 6.21 ms | 3.49 ms | 2.72 ms | 1.778x | 128.0 | 128 | 0.0 |
| gpu | transpose | 0.60 ms | 0.26 ms | 0.33 ms | 2.265x | 119.0 | 102 | 17.0 |
| cpu | MemcpyD2D | 3.48 ms | 3.63 ms | -0.14 ms | 0.961x | 175.0 | 175 | 0.0 |
| gpu | MemcpyD2D | 1.06 ms | 0.96 ms | 0.10 ms | 1.104x | 226.0 | 209 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gla_decode_heavy_smoke_check.json`
- report: `results/gla_decode_heavy_smoke_check.md`
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
