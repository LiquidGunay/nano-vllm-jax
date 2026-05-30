# GPU Matrix Report

- created_at_utc: `20260530_112119`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_112119`
- output_json: `results/gpu_matrix_decode_moveon_w4_bv64.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | no | no | 166.06 | 219.15 | 0.758x | 197.24 | 31.18 | 183.16 | 0.907x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.72 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 472.68 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 453.10 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 127.90 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 125.78 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 10.01 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 6.12 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | gather | 5.30 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | MemcpyD2D | 4.11 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_92 | 103.46 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_116 | 64.08 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_62 | 62.81 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | fusion_922 | 37.26 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 930.59 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 768.18 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 768.14 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 763.93 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 583.35 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 472.68 ms | 317.55 ms | 155.13 ms | 1.489x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 453.10 ms | 300.79 ms | 152.31 ms | 1.506x | 283.0 | 283 | 0.0 |
| command_buffer::update | 125.78 ms | 89.01 ms | 36.77 ms | 1.413x | 2394.0 | 126 | 2268.0 |
| command_buffer::execute | 127.90 ms | 93.00 ms | 34.90 ms | 1.375x | 2517.0 | 231 | 2286.0 |
| np.asarray(jax.Array) | 6.12 ms | 4.61 ms | 1.52 ms | 1.329x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 5.13 ms | 4.64 ms | 0.49 ms | 1.104x | 401.0 | 384 | 17.0 |
| gather | 15.30 ms | 15.05 ms | 0.26 ms | 1.017x | 1603.0 | 1603 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.004x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 472.68 ms | 317.55 ms | 155.13 ms | 1.489x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 453.10 ms | 300.79 ms | 152.31 ms | 1.506x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::update | 125.78 ms | 89.01 ms | 36.77 ms | 1.413x | 2394.0 | 126 | 2268.0 |
| cpu | command_buffer::execute | 127.90 ms | 93.00 ms | 34.90 ms | 1.375x | 2517.0 | 231 | 2286.0 |
| cpu | np.asarray(jax.Array) | 6.12 ms | 4.61 ms | 1.52 ms | 1.329x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 4.11 ms | 3.67 ms | 0.44 ms | 1.119x | 175.0 | 175 | 0.0 |
| cpu | gather | 10.01 ms | 9.71 ms | 0.30 ms | 1.031x | 18.0 | 18 | 0.0 |
| gpu | MemcpyD2D | 1.02 ms | 0.97 ms | 0.05 ms | 1.050x | 226.0 | 209 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_moveon_w4_bv64.json`
- report: `results/gpu_matrix_decode_moveon_w4_bv64.md`
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
