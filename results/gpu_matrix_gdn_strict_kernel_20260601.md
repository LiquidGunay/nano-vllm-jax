# GPU Matrix Report

- created_at_utc: `20260601_045019`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260601_gdn_strict_kernel`
- output_json: `results/gpu_matrix_gdn_strict_kernel_20260601.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | no | no | 150.78 | 213.54 | 0.706x | 192.18 | 41.41 | 151.84 | 0.993x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.02 s | 0.82 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 488.01 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 468.05 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 130.31 ms | 2444.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 93.43 ms | 2413.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 10.62 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 5.69 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | gather | 5.49 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | MemcpyD2D | 3.33 ms | 172.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 163.25 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_110 | 103.22 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_134 | 63.72 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_80 | 62.42 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | _gdn_fla_chunk_fwd_o_packed_kernel | 50.07 ms | 18 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 961.54 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 846.19 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 846.15 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 842.33 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 594.11 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| command_buffer::update | 93.43 ms | 143.82 ms | -50.39 ms | 0.650x | 2413.0 | 127 | 2286.0 |
| command_buffer::execute | 130.31 ms | 145.58 ms | -15.27 ms | 0.895x | 2444.0 | 231 | 2213.0 |
| np.asarray(jax.Array) | 5.69 ms | 14.85 ms | -9.16 ms | 0.383x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 468.05 ms | 462.11 ms | 5.94 ms | 1.013x | 283.0 | 283 | 0.0 |
| gather | 16.11 ms | 19.45 ms | -3.34 ms | 0.828x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.34 ms | 5.56 ms | -1.22 ms | 0.780x | 362.0 | 384 | -22.0 |
| forward_step_token_ids_jit | 488.01 ms | 489.15 ms | -1.14 ms | 0.998x | 128.0 | 128 | 0.0 |
| transpose | 0.08 ms | 0.07 ms | 0.00 ms | 1.066x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | command_buffer::update | 93.43 ms | 143.82 ms | -50.39 ms | 0.650x | 2413.0 | 127 | 2286.0 |
| cpu | command_buffer::execute | 130.31 ms | 145.58 ms | -15.27 ms | 0.895x | 2444.0 | 231 | 2213.0 |
| cpu | np.asarray(jax.Array) | 5.69 ms | 14.85 ms | -9.16 ms | 0.383x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 468.05 ms | 462.11 ms | 5.94 ms | 1.013x | 283.0 | 283 | 0.0 |
| cpu | gather | 10.62 ms | 14.15 ms | -3.52 ms | 0.751x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.33 ms | 4.59 ms | -1.27 ms | 0.724x | 172.0 | 175 | -3.0 |
| cpu | forward_step_token_ids_jit | 488.01 ms | 489.15 ms | -1.14 ms | 0.998x | 128.0 | 128 | 0.0 |
| gpu | gather | 5.49 ms | 5.31 ms | 0.18 ms | 1.034x | 1585.0 | 1585 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_gdn_strict_kernel_20260601.json`
- report: `results/gpu_matrix_gdn_strict_kernel_20260601.md`
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
