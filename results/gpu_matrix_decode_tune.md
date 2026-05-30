# GPU Matrix Report

- created_at_utc: `20260530_073526`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260530_decode_tune`
- output_json: `results/gpu_matrix_decode_tune.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 151.96 | 218.71 | 0.695x | 196.84 | 44.88 | 147.06 | 1.033x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.08 s | 0.76 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 449.94 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 431.77 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 142.91 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 91.98 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | gather | 15.60 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | np.asarray(jax.Array) | 10.32 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | gpu | gather | 5.40 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | MemcpyD2D | 3.61 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.19 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.28 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.79 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.46 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.06 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 884.98 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 839.42 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 839.34 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 835.07 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 564.18 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 449.94 ms | 283.63 ms | 166.32 ms | 1.586x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 431.77 ms | 267.45 ms | 164.32 ms | 1.614x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 142.91 ms | 97.73 ms | 45.17 ms | 1.462x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 91.98 ms | 62.82 ms | 29.16 ms | 1.464x | 2394.0 | 126 | 2268.0 |
| gather | 21.00 ms | 15.70 ms | 5.30 ms | 1.338x | 1603.0 | 1603 | 0.0 |
| np.asarray(jax.Array) | 10.32 ms | 13.80 ms | -3.48 ms | 0.748x | 128.0 | 128 | 0.0 |
| transpose | 1.17 ms | 0.63 ms | 0.54 ms | 1.864x | 187.0 | 170 | 17.0 |
| MemcpyD2D | 4.65 ms | 4.43 ms | 0.22 ms | 1.050x | 401.0 | 384 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 449.94 ms | 283.63 ms | 166.32 ms | 1.586x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 431.77 ms | 267.45 ms | 164.32 ms | 1.614x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 142.91 ms | 97.73 ms | 45.17 ms | 1.462x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 91.98 ms | 62.82 ms | 29.16 ms | 1.464x | 2394.0 | 126 | 2268.0 |
| cpu | gather | 15.60 ms | 10.37 ms | 5.22 ms | 1.503x | 18.0 | 18 | 0.0 |
| cpu | np.asarray(jax.Array) | 10.32 ms | 13.80 ms | -3.48 ms | 0.748x | 128.0 | 128 | 0.0 |
| gpu | transpose | 0.59 ms | 0.26 ms | 0.33 ms | 2.263x | 119.0 | 102 | 17.0 |
| cpu | transpose | 0.57 ms | 0.36 ms | 0.21 ms | 1.576x | 68.0 | 68 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune.json`
- report: `results/gpu_matrix_decode_tune.md`
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
