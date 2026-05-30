# GPU Matrix Report

- created_at_utc: `20260530_122438`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/decode_focus_base`
- output_json: `results/gpu_matrix_decode_focus_base.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | no | no | 151.77 | 216.82 | 0.700x | 195.14 | 43.37 | 180.00 | 0.843x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.05 s | 0.79 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 513.05 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 489.21 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 154.68 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 133.26 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 14.04 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 8.25 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | gather | 5.53 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | MemcpyD2D | 3.71 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 145.01 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_92 | 112.68 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_116 | 71.21 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_62 | 68.48 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_140 | 41.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 1009.70 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 839.38 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 839.34 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 834.83 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 633.50 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 513.05 ms | 339.36 ms | 173.68 ms | 1.512x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 489.21 ms | 317.96 ms | 171.26 ms | 1.539x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 154.68 ms | 98.65 ms | 56.03 ms | 1.568x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 133.26 ms | 95.95 ms | 37.31 ms | 1.389x | 2394.0 | 126 | 2268.0 |
| np.asarray(jax.Array) | 8.25 ms | 10.94 ms | -2.69 ms | 0.754x | 128.0 | 128 | 0.0 |
| gather | 19.57 ms | 16.96 ms | 2.61 ms | 1.154x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.75 ms | 4.72 ms | 0.03 ms | 1.007x | 401.0 | 384 | 17.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.008x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 513.05 ms | 339.36 ms | 173.68 ms | 1.512x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 489.21 ms | 317.96 ms | 171.26 ms | 1.539x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 154.68 ms | 98.65 ms | 56.03 ms | 1.568x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 133.26 ms | 95.95 ms | 37.31 ms | 1.389x | 2394.0 | 126 | 2268.0 |
| cpu | np.asarray(jax.Array) | 8.25 ms | 10.94 ms | -2.69 ms | 0.754x | 128.0 | 128 | 0.0 |
| cpu | gather | 14.04 ms | 11.67 ms | 2.37 ms | 1.203x | 18.0 | 18 | 0.0 |
| gpu | gather | 5.53 ms | 5.29 ms | 0.24 ms | 1.046x | 1585.0 | 1585 | 0.0 |
| gpu | MemcpyD2D | 1.05 ms | 0.98 ms | 0.07 ms | 1.071x | 226.0 | 209 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_focus_base.json`
- report: `results/gpu_matrix_decode_focus_base.md`
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
