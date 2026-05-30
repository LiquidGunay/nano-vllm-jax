# GPU Matrix Report

- created_at_utc: `20260530_124217`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_moveon_quick4`
- output_json: `results/gpu_matrix_decode_moveon_quick4.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 140.65 | 213.54 | 0.659x | 192.18 | 51.54 | 160.90 | 0.874x |

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 558.11 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 525.73 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 168.93 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 142.66 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 13.68 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 7.00 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.41 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.81 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 127.49 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.49 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 64.04 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.67 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | fusion_922 | 37.42 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 1097.46 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 905.66 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 905.59 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 900.02 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 690.26 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 558.11 ms | 449.30 ms | 108.81 ms | 1.242x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 525.73 ms | 419.54 ms | 106.19 ms | 1.253x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 168.93 ms | 131.80 ms | 37.13 ms | 1.282x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 142.66 ms | 129.05 ms | 13.61 ms | 1.105x | 2394.0 | 126 | 2268.0 |
| np.asarray(jax.Array) | 7.00 ms | 2.40 ms | 4.60 ms | 2.920x | 128.0 | 128 | 0.0 |
| gather | 19.09 ms | 15.82 ms | 3.27 ms | 1.207x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.79 ms | 5.40 ms | -0.61 ms | 0.887x | 401.0 | 384 | 17.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.998x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 558.11 ms | 449.30 ms | 108.81 ms | 1.242x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 525.73 ms | 419.54 ms | 106.19 ms | 1.253x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 168.93 ms | 131.80 ms | 37.13 ms | 1.282x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 142.66 ms | 129.05 ms | 13.61 ms | 1.105x | 2394.0 | 126 | 2268.0 |
| cpu | np.asarray(jax.Array) | 7.00 ms | 2.40 ms | 4.60 ms | 2.920x | 128.0 | 128 | 0.0 |
| cpu | gather | 13.68 ms | 10.53 ms | 3.15 ms | 1.299x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.81 ms | 4.43 ms | -0.62 ms | 0.860x | 175.0 | 175 | 0.0 |
| gpu | gather | 5.41 ms | 5.29 ms | 0.12 ms | 1.023x | 1585.0 | 1585 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_moveon_quick4.json`
- report: `results/gpu_matrix_decode_moveon_quick4.md`
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
