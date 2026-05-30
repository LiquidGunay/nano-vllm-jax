# GPU Matrix Report

- created_at_utc: `20260530_094723`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_094723`
- output_json: `results/gpu_matrix_decode_tune_focus_default.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | yes | no | 178.48 | 218.20 | 0.818x | 196.38 | 17.90 | 182.87 | 0.976x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.67 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 397.12 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 378.92 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 111.88 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 100.47 ms | 2403.5 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 10.37 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.33 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 4.23 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.31 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.81 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | fusion_922 | 37.30 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 786.39 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 718.58 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 718.55 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 715.22 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 496.82 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | gemm_fusion_dot_265 | 127.47 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_92 | 103.46 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_62 | 62.80 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | fusion_922 | 37.33 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | PjitFunction(compiled) | 776.58 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:632 generate_with_trace | 709.27 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 709.23 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:161 step | 705.55 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $model_runner.py:3737 run | 489.50 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 378.92 ms | 295.82 ms | 83.10 ms | 1.281x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 397.12 ms | 317.61 ms | 79.51 ms | 1.250x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 111.88 ms | 94.93 ms | 16.94 ms | 1.178x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 100.47 ms | 89.36 ms | 11.12 ms | 1.124x | 2403.5 | 126 | 2277.5 |
| gather | 15.70 ms | 15.11 ms | 0.59 ms | 1.039x | 1603.0 | 1603 | 0.0 |
| np.asarray(jax.Array) | 4.23 ms | 3.69 ms | 0.54 ms | 1.145x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 4.36 ms | 4.56 ms | -0.19 ms | 0.958x | 401.0 | 384 | 17.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.017x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 378.92 ms | 295.82 ms | 83.10 ms | 1.281x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 397.12 ms | 317.61 ms | 79.51 ms | 1.250x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 111.88 ms | 94.93 ms | 16.94 ms | 1.178x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 100.47 ms | 89.36 ms | 11.12 ms | 1.124x | 2403.5 | 126 | 2277.5 |
| cpu | gather | 10.37 ms | 9.77 ms | 0.60 ms | 1.061x | 18.0 | 18 | 0.0 |
| cpu | np.asarray(jax.Array) | 4.23 ms | 3.69 ms | 0.54 ms | 1.145x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 3.31 ms | 3.57 ms | -0.26 ms | 0.926x | 175.0 | 175 | 0.0 |
| gpu | MemcpyD2D | 1.06 ms | 0.98 ms | 0.07 ms | 1.074x | 226.0 | 209 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_focus_default.json`
- report: `results/gpu_matrix_decode_tune_focus_default.md`
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
