# GPU Matrix Report

- created_at_utc: `20260530_092054`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_092054`
- output_json: `results/gpu_matrix_decode_tune_env_w1_b16.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | yes | no | 105.15 | 208.92 | 0.503x | 188.02 | 82.88 | 172.30 | 0.610x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.06 s | 1.15 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 676.61 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 640.24 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 197.00 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 170.72 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 12.68 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 8.16 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 6.57 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 4.64 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.50 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.53 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.14 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.78 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.39 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 1368.68 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1171.08 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1171.02 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 1164.40 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 860.93 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | gemm_fusion_dot_265 | 198.35 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_92 | 136.97 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_116 | 106.51 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_62 | 96.71 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_140 | 58.20 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | PjitFunction(compiled) | 1294.76 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:632 generate_with_trace | 1246.33 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1246.28 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:161 step | 1238.80 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $model_runner.py:3737 run | 816.82 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 676.61 ms | 377.78 ms | 298.83 ms | 1.791x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 640.24 ms | 355.30 ms | 284.94 ms | 1.802x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 197.00 ms | 110.97 ms | 86.03 ms | 1.775x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 170.72 ms | 106.14 ms | 64.57 ms | 1.608x | 2394.0 | 127 | 2267.0 |
| np.asarray(jax.Array) | 8.16 ms | 5.10 ms | 3.05 ms | 1.598x | 128.0 | 128 | 0.0 |
| gather | 19.25 ms | 16.39 ms | 2.87 ms | 1.175x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.47 ms | 4.73 ms | 0.74 ms | 1.156x | 384.0 | 384 | 0.0 |
| transpose | 0.08 ms | 0.07 ms | 0.00 ms | 1.002x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 676.61 ms | 377.78 ms | 298.83 ms | 1.791x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 640.24 ms | 355.30 ms | 284.94 ms | 1.802x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 197.00 ms | 110.97 ms | 86.03 ms | 1.775x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 170.72 ms | 106.14 ms | 64.57 ms | 1.608x | 2394.0 | 127 | 2267.0 |
| cpu | np.asarray(jax.Array) | 8.16 ms | 5.10 ms | 3.05 ms | 1.598x | 128.0 | 128 | 0.0 |
| cpu | gather | 12.68 ms | 11.12 ms | 1.56 ms | 1.141x | 18.0 | 18 | 0.0 |
| gpu | gather | 6.57 ms | 5.27 ms | 1.30 ms | 1.247x | 1585.0 | 1585 | 0.0 |
| cpu | MemcpyD2D | 4.64 ms | 3.75 ms | 0.89 ms | 1.237x | 175.0 | 175 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_env_w1_b16.json`
- report: `results/gpu_matrix_decode_tune_env_w1_b16.md`
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
