# GPU Matrix Report

- created_at_utc: `20260530_090958`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_090958`
- output_json: `results/gpu_matrix_decode_off_prefill_bf16_qkv.json`
- jax_python: `/root/miniconda3/bin/python3` (available: yes)

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | yes | no | 178.38 | 218.67 | 0.816x | 196.80 | 18.42 | 182.36 | 0.978x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.67 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 398.91 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 380.05 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 111.68 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 100.84 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 10.01 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 5.32 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 4.66 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 3.28 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.44 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.79 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.34 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 785.30 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 713.04 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 712.99 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 709.39 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 494.38 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | gemm_fusion_dot_265 | 127.46 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_92 | 103.44 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_116 | 64.06 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_62 | 62.79 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | fusion_922 | 37.32 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | PjitFunction(compiled) | 785.15 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:632 generate_with_trace | 715.86 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 715.81 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:161 step | 712.56 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $model_runner.py:3737 run | 494.36 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 380.05 ms | 290.29 ms | 89.76 ms | 1.309x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 398.91 ms | 309.71 ms | 89.20 ms | 1.288x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 111.68 ms | 90.11 ms | 21.57 ms | 1.239x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 100.84 ms | 85.20 ms | 15.64 ms | 1.184x | 2394.0 | 127 | 2267.0 |
| np.asarray(jax.Array) | 4.66 ms | 3.18 ms | 1.48 ms | 1.467x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 4.26 ms | 4.42 ms | -0.16 ms | 0.964x | 384.0 | 384 | 0.0 |
| gather | 15.33 ms | 15.29 ms | 0.04 ms | 1.003x | 1603.0 | 1603 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.003x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 380.05 ms | 290.29 ms | 89.76 ms | 1.309x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 398.91 ms | 309.71 ms | 89.20 ms | 1.288x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 111.68 ms | 90.11 ms | 21.57 ms | 1.239x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 100.84 ms | 85.20 ms | 15.64 ms | 1.184x | 2394.0 | 127 | 2267.0 |
| cpu | np.asarray(jax.Array) | 4.66 ms | 3.18 ms | 1.48 ms | 1.467x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 3.28 ms | 3.44 ms | -0.16 ms | 0.952x | 175.0 | 175 | 0.0 |
| cpu | gather | 10.01 ms | 9.92 ms | 0.09 ms | 1.009x | 18.0 | 18 | 0.0 |
| gpu | gather | 5.32 ms | 5.37 ms | -0.05 ms | 0.991x | 1585.0 | 1585 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_off_prefill_bf16_qkv.json`
- report: `results/gpu_matrix_decode_off_prefill_bf16_qkv.md`
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
