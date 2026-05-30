# GPU Matrix Report

- created_at_utc: `20260530_073838`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_tune_baseline`
- output_json: `results/gpu_matrix_decode_tune_baseline.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 142.50 | 218.88 | 0.651x | 197.00 | 54.49 | 149.97 | 0.950x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.05 s | 0.85 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 481.75 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 456.59 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 157.91 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 105.59 ms | 2413.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 23.37 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 10.93 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.41 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.48 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.31 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.33 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.52 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.07 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 947.01 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 893.25 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 893.21 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 888.73 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 598.26 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 481.75 ms | 283.17 ms | 198.57 ms | 1.701x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 456.59 ms | 267.27 ms | 189.32 ms | 1.708x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 157.91 ms | 94.84 ms | 63.07 ms | 1.665x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 105.59 ms | 63.64 ms | 41.94 ms | 1.659x | 2413.0 | 126 | 2287.0 |
| np.asarray(jax.Array) | 23.37 ms | 1.38 ms | 21.99 ms | 16.963x | 128.0 | 128 | 0.0 |
| gather | 16.34 ms | 14.97 ms | 1.37 ms | 1.091x | 1603.0 | 1603 | 0.0 |
| transpose | 0.95 ms | 0.64 ms | 0.31 ms | 1.493x | 187.0 | 170 | 17.0 |
| MemcpyD2D | 4.48 ms | 4.32 ms | 0.16 ms | 1.037x | 401.0 | 384 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 481.75 ms | 283.17 ms | 198.57 ms | 1.701x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 456.59 ms | 267.27 ms | 189.32 ms | 1.708x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 157.91 ms | 94.84 ms | 63.07 ms | 1.665x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 105.59 ms | 63.64 ms | 41.94 ms | 1.659x | 2413.0 | 126 | 2287.0 |
| cpu | np.asarray(jax.Array) | 23.37 ms | 1.38 ms | 21.99 ms | 16.963x | 128.0 | 128 | 0.0 |
| cpu | gather | 10.93 ms | 9.66 ms | 1.26 ms | 1.131x | 18.0 | 18 | 0.0 |
| gpu | transpose | 0.59 ms | 0.26 ms | 0.33 ms | 2.258x | 119.0 | 102 | 17.0 |
| cpu | MemcpyD2D | 3.48 ms | 3.36 ms | 0.12 ms | 1.037x | 175.0 | 175 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_baseline.json`
- report: `results/gpu_matrix_decode_tune_baseline.md`
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
