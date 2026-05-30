# GPU Matrix Report

- created_at_utc: `20260530_122308`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_122308`
- output_json: `results/gpu_matrix_decode_tune_env_w4_b16.json`
- jax_python: `/root/miniconda3/bin/python3` (available: yes)

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | yes | no | 89.09 | 218.91 | 0.407x | 197.02 | 107.94 | 161.70 | 0.551x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.07 s | 1.37 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 666.69 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 645.12 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 195.13 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 168.82 ms | 2413.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 14.83 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 9.51 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 6.48 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 4.29 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.08 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.80 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.29 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 1708.54 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1392.97 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1392.92 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 1386.86 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 1082.46 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | gemm_fusion_dot_265 | 285.88 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_92 | 183.79 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_116 | 147.95 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_62 | 124.05 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_140 | 82.45 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:632 generate_with_trace | 1475.25 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1475.21 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:161 step | 1471.33 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | PjitFunction(compiled) | 915.72 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $scheduler.py:151 schedule | 841.44 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 645.12 ms | 393.30 ms | 251.83 ms | 1.640x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 666.69 ms | 418.12 ms | 248.57 ms | 1.594x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 195.13 ms | 121.61 ms | 73.52 ms | 1.605x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 168.82 ms | 120.03 ms | 48.80 ms | 1.407x | 2413.0 | 127 | 2286.0 |
| np.asarray(jax.Array) | 9.51 ms | 5.31 ms | 4.21 ms | 1.793x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 6.37 ms | 5.35 ms | 1.02 ms | 1.191x | 384.0 | 384 | 0.0 |
| gather | 21.31 ms | 22.30 ms | -0.99 ms | 0.956x | 1603.0 | 1603 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.993x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 645.12 ms | 393.30 ms | 251.83 ms | 1.640x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 666.69 ms | 418.12 ms | 248.57 ms | 1.594x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 195.13 ms | 121.61 ms | 73.52 ms | 1.605x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 168.82 ms | 120.03 ms | 48.80 ms | 1.407x | 2413.0 | 127 | 2286.0 |
| cpu | np.asarray(jax.Array) | 9.51 ms | 5.31 ms | 4.21 ms | 1.793x | 128.0 | 128 | 0.0 |
| cpu | gather | 14.83 ms | 17.04 ms | -2.21 ms | 0.870x | 18.0 | 18 | 0.0 |
| gpu | gather | 6.48 ms | 5.26 ms | 1.22 ms | 1.232x | 1585.0 | 1585 | 0.0 |
| gpu | MemcpyD2D | 2.09 ms | 0.98 ms | 1.10 ms | 2.121x | 209.0 | 209 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_env_w4_b16.json`
- report: `results/gpu_matrix_decode_tune_env_w4_b16.md`
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
