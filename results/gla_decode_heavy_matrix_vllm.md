# GPU Matrix Report

- created_at_utc: `20260530_071407`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260530_073500`
- output_json: `results/gla_decode_heavy_matrix_vllm.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | yes | no | 162.90 | 213.54 | 0.763x | 192.18 | 29.29 | 150.11 | 1.085x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.74 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 378.52 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 358.95 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 125.64 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 81.53 ms | 2403.5 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 10.32 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.42 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 4.23 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.12 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.23 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.32 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.09 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 761.91 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 761.87 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 758.76 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 693.09 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 444.15 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | gemm_fusion_dot_265 | 163.23 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_92 | 103.32 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_140 | 37.09 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:632 generate_with_trace | 804.43 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 804.33 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:161 step | 800.30 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | PjitFunction(compiled) | 795.16 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $model_runner.py:3737 run | 510.40 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 358.95 ms | 304.31 ms | 54.63 ms | 1.180x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 378.52 ms | 326.38 ms | 52.14 ms | 1.160x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 125.64 ms | 110.70 ms | 14.95 ms | 1.135x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 81.53 ms | 72.67 ms | 8.86 ms | 1.122x | 2403.5 | 126 | 2277.5 |
| np.asarray(jax.Array) | 4.23 ms | 2.70 ms | 1.53 ms | 1.568x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 4.16 ms | 4.64 ms | -0.48 ms | 0.896x | 401.0 | 384 | 17.0 |
| gather | 15.75 ms | 15.30 ms | 0.44 ms | 1.029x | 1603.0 | 1603 | 0.0 |
| transpose | 0.97 ms | 0.68 ms | 0.29 ms | 1.435x | 187.0 | 170 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 358.95 ms | 304.31 ms | 54.63 ms | 1.180x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 378.52 ms | 326.38 ms | 52.14 ms | 1.160x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 125.64 ms | 110.70 ms | 14.95 ms | 1.135x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 81.53 ms | 72.67 ms | 8.86 ms | 1.122x | 2403.5 | 126 | 2277.5 |
| cpu | np.asarray(jax.Array) | 4.23 ms | 2.70 ms | 1.53 ms | 1.568x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 3.12 ms | 3.68 ms | -0.56 ms | 0.848x | 175.0 | 175 | 0.0 |
| cpu | gather | 10.32 ms | 9.99 ms | 0.34 ms | 1.034x | 18.0 | 18 | 0.0 |
| gpu | transpose | 0.59 ms | 0.26 ms | 0.33 ms | 2.258x | 119.0 | 102 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gla_decode_heavy_matrix_vllm.json`
- report: `results/gla_decode_heavy_matrix_vllm.md`
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
