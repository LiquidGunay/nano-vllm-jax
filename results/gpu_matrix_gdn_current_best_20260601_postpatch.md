# GPU Matrix Report

- created_at_utc: `20260601_034201`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260601_gdn_current_best_postpatch`
- output_json: `results/gpu_matrix_gdn_current_best_20260601_postpatch.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | yes | no | 163.62 | 213.54 | 0.766x | 192.18 | 28.56 | 151.84 | 1.078x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.74 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 371.85 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 352.76 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 126.59 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 79.89 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 10.87 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 10.78 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | gather | 5.43 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | MemcpyD2D | 3.19 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 163.20 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_110 | 103.22 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_134 | 63.77 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_80 | 62.41 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_158 | 36.88 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 768.57 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 768.52 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 765.23 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 705.10 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 451.46 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | gemm_fusion_dot_265 | 163.23 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_110 | 103.24 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_134 | 63.79 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_80 | 62.40 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_158 | 36.89 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:632 generate_with_trace | 788.69 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 788.64 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:161 step | 785.41 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | PjitFunction(compiled) | 756.75 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $model_runner.py:3737 run | 486.34 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 371.85 ms | 489.15 ms | -117.30 ms | 0.760x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 352.76 ms | 462.11 ms | -109.35 ms | 0.763x | 283.0 | 283 | 0.0 |
| command_buffer::update | 79.89 ms | 143.82 ms | -63.93 ms | 0.555x | 2394.0 | 127 | 2267.0 |
| command_buffer::execute | 126.59 ms | 145.58 ms | -18.99 ms | 0.870x | 2517.0 | 231 | 2286.0 |
| np.asarray(jax.Array) | 10.78 ms | 14.85 ms | -4.07 ms | 0.726x | 128.0 | 128 | 0.0 |
| gather | 16.30 ms | 19.45 ms | -3.15 ms | 0.838x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.23 ms | 5.56 ms | -1.33 ms | 0.761x | 401.0 | 384 | 17.0 |
| transpose | 0.98 ms | 0.07 ms | 0.90 ms | 13.228x | 187.0 | 30 | 157.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 371.85 ms | 489.15 ms | -117.30 ms | 0.760x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 352.76 ms | 462.11 ms | -109.35 ms | 0.763x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::update | 79.89 ms | 143.82 ms | -63.93 ms | 0.555x | 2394.0 | 127 | 2267.0 |
| cpu | command_buffer::execute | 126.59 ms | 145.58 ms | -18.99 ms | 0.870x | 2517.0 | 231 | 2286.0 |
| cpu | np.asarray(jax.Array) | 10.78 ms | 14.85 ms | -4.07 ms | 0.726x | 128.0 | 128 | 0.0 |
| cpu | gather | 10.87 ms | 14.15 ms | -3.27 ms | 0.769x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.19 ms | 4.59 ms | -1.40 ms | 0.694x | 175.0 | 175 | 0.0 |
| gpu | transpose | 0.59 ms | 0.07 ms | 0.52 ms | 8.059x | 119.0 | 30 | 89.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_gdn_current_best_20260601_postpatch.json`
- report: `results/gpu_matrix_gdn_current_best_20260601_postpatch.md`
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
