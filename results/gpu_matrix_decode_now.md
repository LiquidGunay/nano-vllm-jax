# GPU Matrix Report

- created_at_utc: `20260530_075054`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_075054`
- output_json: `results/gpu_matrix_decode_now.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 143.34 | 218.34 | 0.657x | 196.50 | 53.16 | 149.83 | 0.957x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.84 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 469.59 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 443.92 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 164.42 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 99.67 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 12.01 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 6.30 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.41 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.43 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.23 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.32 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.77 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.08 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 922.92 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 887.50 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 887.40 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 882.75 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 585.69 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 469.59 ms | 310.23 ms | 159.36 ms | 1.514x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 443.92 ms | 291.41 ms | 152.51 ms | 1.523x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 164.42 ms | 100.14 ms | 64.28 ms | 1.642x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 99.67 ms | 73.14 ms | 26.53 ms | 1.363x | 2394.0 | 126 | 2268.0 |
| gather | 17.42 ms | 15.48 ms | 1.93 ms | 1.125x | 1603.0 | 1603 | 0.0 |
| transpose | 0.97 ms | 0.61 ms | 0.36 ms | 1.590x | 187.0 | 170 | 17.0 |
| np.asarray(jax.Array) | 6.30 ms | 6.58 ms | -0.27 ms | 0.958x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 4.44 ms | 4.68 ms | -0.24 ms | 0.948x | 401.0 | 384 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 469.59 ms | 310.23 ms | 159.36 ms | 1.514x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 443.92 ms | 291.41 ms | 152.51 ms | 1.523x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 164.42 ms | 100.14 ms | 64.28 ms | 1.642x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 99.67 ms | 73.14 ms | 26.53 ms | 1.363x | 2394.0 | 126 | 2268.0 |
| cpu | gather | 12.01 ms | 10.15 ms | 1.85 ms | 1.182x | 18.0 | 18 | 0.0 |
| gpu | transpose | 0.60 ms | 0.26 ms | 0.33 ms | 2.265x | 119.0 | 102 | 17.0 |
| cpu | MemcpyD2D | 3.43 ms | 3.72 ms | -0.29 ms | 0.923x | 175.0 | 175 | 0.0 |
| cpu | np.asarray(jax.Array) | 6.30 ms | 6.58 ms | -0.27 ms | 0.958x | 128.0 | 128 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_now.json`
- report: `results/gpu_matrix_decode_now.md`
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
