# GPU Matrix Report

- created_at_utc: `20260530_074543`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260530_decode_tune2`
- output_json: `results/gpu_matrix_decode_tune2.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | yes | no | 166.95 | 219.11 | 0.762x | 197.20 | 30.25 | 149.48 | 1.117x |

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 350.45 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 331.60 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 116.67 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 74.16 ms | 2413.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 9.85 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.41 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 5.38 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.00 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.32 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.27 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.47 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.07 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 764.77 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 764.72 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 761.48 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 690.85 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 444.97 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | gemm_fusion_dot_265 | 163.24 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_92 | 103.28 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_116 | 63.77 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_62 | 62.47 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_140 | 37.04 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:632 generate_with_trace | 762.41 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 762.36 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:161 step | 759.27 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | PjitFunction(compiled) | 685.58 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $model_runner.py:3737 run | 441.09 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 350.45 ms | 281.80 ms | 68.65 ms | 1.244x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 331.60 ms | 265.04 ms | 66.55 ms | 1.251x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 116.67 ms | 93.77 ms | 22.91 ms | 1.244x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 74.16 ms | 64.55 ms | 9.61 ms | 1.149x | 2413.0 | 127 | 2286.0 |
| gather | 15.26 ms | 16.63 ms | -1.37 ms | 0.918x | 1603.0 | 1603 | 0.0 |
| transpose | 0.96 ms | 0.62 ms | 0.35 ms | 1.559x | 187.0 | 170 | 17.0 |
| MemcpyD2D | 4.05 ms | 4.36 ms | -0.32 ms | 0.927x | 401.0 | 384 | 17.0 |
| np.asarray(jax.Array) | 5.38 ms | 5.43 ms | -0.05 ms | 0.990x | 128.0 | 128 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 350.45 ms | 281.80 ms | 68.65 ms | 1.244x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 331.60 ms | 265.04 ms | 66.55 ms | 1.251x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 116.67 ms | 93.77 ms | 22.91 ms | 1.244x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 74.16 ms | 64.55 ms | 9.61 ms | 1.149x | 2413.0 | 127 | 2286.0 |
| cpu | gather | 9.85 ms | 11.27 ms | -1.42 ms | 0.874x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.00 ms | 3.40 ms | -0.40 ms | 0.883x | 175.0 | 175 | 0.0 |
| gpu | transpose | 0.59 ms | 0.26 ms | 0.33 ms | 2.267x | 119.0 | 102 | 17.0 |
| gpu | MemcpyD2D | 1.05 ms | 0.97 ms | 0.08 ms | 1.085x | 226.0 | 209 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune2.json`
- report: `results/gpu_matrix_decode_tune2.md`
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
