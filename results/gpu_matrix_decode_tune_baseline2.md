# GPU Matrix Report

- created_at_utc: `20260530_074149`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260530_decode_tune_baseline2`
- output_json: `results/gpu_matrix_decode_tune_baseline2.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | yes | no | 166.86 | 218.94 | 0.762x | 197.05 | 30.19 | 147.54 | 1.131x |

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 364.65 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 345.00 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 121.96 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 77.04 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 10.38 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.42 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 4.58 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.07 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.25 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.34 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.07 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 764.55 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 764.51 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 761.22 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 721.11 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 464.63 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | gemm_fusion_dot_265 | 163.25 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_92 | 103.34 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_116 | 63.79 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_140 | 37.07 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:632 generate_with_trace | 764.24 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 764.19 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:161 step | 760.90 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | PjitFunction(compiled) | 711.51 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $model_runner.py:3737 run | 456.27 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 345.00 ms | 332.83 ms | 12.17 ms | 1.037x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 364.65 ms | 355.02 ms | 9.63 ms | 1.027x | 128.0 | 128 | 0.0 |
| command_buffer::update | 77.04 ms | 83.64 ms | -6.60 ms | 0.921x | 2394.0 | 126 | 2268.0 |
| command_buffer::execute | 121.96 ms | 116.93 ms | 5.03 ms | 1.043x | 2517.0 | 231 | 2286.0 |
| np.asarray(jax.Array) | 4.58 ms | 5.84 ms | -1.26 ms | 0.784x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 4.12 ms | 5.15 ms | -1.03 ms | 0.799x | 401.0 | 384 | 17.0 |
| gather | 15.80 ms | 14.91 ms | 0.89 ms | 1.060x | 1603.0 | 1603 | 0.0 |
| transpose | 0.99 ms | 0.61 ms | 0.37 ms | 1.605x | 187.0 | 170 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 345.00 ms | 332.83 ms | 12.17 ms | 1.037x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 364.65 ms | 355.02 ms | 9.63 ms | 1.027x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::update | 77.04 ms | 83.64 ms | -6.60 ms | 0.921x | 2394.0 | 126 | 2268.0 |
| cpu | command_buffer::execute | 121.96 ms | 116.93 ms | 5.03 ms | 1.043x | 2517.0 | 231 | 2286.0 |
| cpu | np.asarray(jax.Array) | 4.58 ms | 5.84 ms | -1.26 ms | 0.784x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 3.07 ms | 4.19 ms | -1.12 ms | 0.733x | 175.0 | 175 | 0.0 |
| cpu | gather | 10.38 ms | 9.57 ms | 0.81 ms | 1.084x | 18.0 | 18 | 0.0 |
| gpu | transpose | 0.59 ms | 0.26 ms | 0.33 ms | 2.253x | 119.0 | 102 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_baseline2.json`
- report: `results/gpu_matrix_decode_tune_baseline2.md`
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
