# GPU Matrix Report

- created_at_utc: `20260530_092854`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_090958`
- output_json: `results/gpu_matrix_decode_off_prefill_stage3_default.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | no | no | 113.84 | 218.67 | 0.521x | 196.80 | 82.96 | 142.84 | 0.797x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.07 s | 1.05 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 677.60 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 648.65 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 187.70 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 173.62 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 13.34 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 8.43 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 5.29 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 4.52 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.40 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.42 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.14 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.77 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.41 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 1334.69 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1120.09 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1120.05 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 1113.01 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 845.53 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 677.60 ms | 525.66 ms | 151.94 ms | 1.289x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 648.65 ms | 500.82 ms | 147.83 ms | 1.295x | 283.0 | 283 | 0.0 |
| command_buffer::update | 173.62 ms | 149.27 ms | 24.35 ms | 1.163x | 2394.0 | 126 | 2268.0 |
| command_buffer::execute | 187.70 ms | 173.55 ms | 14.15 ms | 1.082x | 2517.0 | 231 | 2286.0 |
| np.asarray(jax.Array) | 8.43 ms | 10.94 ms | -2.51 ms | 0.771x | 128.0 | 128 | 0.0 |
| gather | 18.63 ms | 19.48 ms | -0.85 ms | 0.956x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.31 ms | 5.23 ms | 0.07 ms | 1.014x | 384.0 | 384 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.998x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 677.60 ms | 525.66 ms | 151.94 ms | 1.289x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 648.65 ms | 500.82 ms | 147.83 ms | 1.295x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::update | 173.62 ms | 149.27 ms | 24.35 ms | 1.163x | 2394.0 | 126 | 2268.0 |
| cpu | command_buffer::execute | 187.70 ms | 173.55 ms | 14.15 ms | 1.082x | 2517.0 | 231 | 2286.0 |
| cpu | np.asarray(jax.Array) | 8.43 ms | 10.94 ms | -2.51 ms | 0.771x | 128.0 | 128 | 0.0 |
| cpu | gather | 13.34 ms | 14.19 ms | -0.85 ms | 0.940x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 4.52 ms | 4.27 ms | 0.25 ms | 1.059x | 175.0 | 175 | 0.0 |
| gpu | MemcpyD2D | 0.79 ms | 0.97 ms | -0.18 ms | 0.816x | 209.0 | 209 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_off_prefill_stage3_default.json`
- report: `results/gpu_matrix_decode_off_prefill_stage3_default.md`
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
