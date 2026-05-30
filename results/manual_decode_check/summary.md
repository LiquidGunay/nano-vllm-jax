# GPU Matrix Report

- created_at_utc: `20260530_124507`
- dry_run: no
- repeats: 1
- run_dir: `results/manual_decode_check`
- output_json: `results/manual_decode_check/summary.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 179.67 | 213.54 | 0.841x | 192.18 | 12.51 | 169.97 | 1.057x |

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 409.35 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 390.50 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 112.70 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 104.24 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 11.56 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.33 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.26 ms | 175.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 3.20 ms | 128.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.81 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | fusion_922 | 37.31 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 805.87 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 708.52 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 708.47 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 704.89 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 508.16 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| command_buffer::update | 104.24 ms | 115.70 ms | -11.46 ms | 0.901x | 2394.0 | 126 | 2268.0 |
| np.asarray(jax.Array) | 3.20 ms | 9.10 ms | -5.90 ms | 0.351x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 112.70 ms | 118.23 ms | -5.54 ms | 0.953x | 2517.0 | 231 | 2286.0 |
| forward_step_token_ids_jit | 409.35 ms | 411.73 ms | -2.38 ms | 0.994x | 128.0 | 128 | 0.0 |
| gather | 16.89 ms | 15.38 ms | 1.51 ms | 1.098x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.32 ms | 5.42 ms | -1.10 ms | 0.797x | 401.0 | 384 | 17.0 |
| PjRtCApiLoadedExecutable::Execute | 390.50 ms | 389.90 ms | 0.60 ms | 1.002x | 283.0 | 283 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.001x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | command_buffer::update | 104.24 ms | 115.70 ms | -11.46 ms | 0.901x | 2394.0 | 126 | 2268.0 |
| cpu | np.asarray(jax.Array) | 3.20 ms | 9.10 ms | -5.90 ms | 0.351x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 112.70 ms | 118.23 ms | -5.54 ms | 0.953x | 2517.0 | 231 | 2286.0 |
| cpu | forward_step_token_ids_jit | 409.35 ms | 411.73 ms | -2.38 ms | 0.994x | 128.0 | 128 | 0.0 |
| cpu | gather | 11.56 ms | 10.01 ms | 1.55 ms | 1.155x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.26 ms | 4.44 ms | -1.18 ms | 0.734x | 175.0 | 175 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 390.50 ms | 389.90 ms | 0.60 ms | 1.002x | 283.0 | 283 | 0.0 |
| gpu | MemcpyD2D | 1.06 ms | 0.98 ms | 0.08 ms | 1.081x | 226.0 | 209 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/manual_decode_check/summary.json`
- report: `None`
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
