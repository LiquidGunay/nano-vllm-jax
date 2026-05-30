# GPU Matrix Report

- created_at_utc: `20260530_085754`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_085754`
- output_json: `results/gpu_matrix_decode_only_2repeat.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | yes | no | 146.36 | 218.62 | 0.669x | 196.76 | 50.40 | 166.57 | 0.879x |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | yes | no | 177.74 | 218.62 | 0.813x | 196.76 | 19.02 | 166.57 | 1.067x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.06 s | 0.81 s |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.67 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 496.40 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 471.98 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 391.68 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 373.48 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 139.60 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 129.31 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 109.20 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 97.92 ms | 2394.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_116 | 64.05 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_62 | 62.80 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | fusion_922 | 37.27 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 1046.04 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 897.98 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 897.93 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 892.49 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 662.70 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | gemm_fusion_dot_265 | 127.44 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_62 | 62.80 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | fusion_922 | 37.33 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | PjitFunction(compiled) | 906.88 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:632 generate_with_trace | 845.18 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 845.06 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:161 step | 840.47 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $model_runner.py:3737 run | 571.42 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.78 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.31 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 758.17 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 714.64 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 714.60 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 711.50 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 481.75 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | gemm_fusion_dot_265 | 127.46 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_116 | 64.09 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_62 | 62.78 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | fusion_922 | 37.34 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | PjitFunction(compiled) | 782.97 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:632 generate_with_trace | 719.09 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 719.05 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:161 step | 715.59 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $model_runner.py:3737 run | 496.07 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: target_vllm_ratio_met=false target=0.9
- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 496.40 ms | 359.88 ms | 136.52 ms | 1.379x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 471.98 ms | 335.64 ms | 136.34 ms | 1.406x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 139.60 ms | 105.70 ms | 33.90 ms | 1.321x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 129.31 ms | 103.36 ms | 25.94 ms | 1.251x | 2394.0 | 126 | 2268.0 |
| np.asarray(jax.Array) | 9.27 ms | 2.53 ms | 6.74 ms | 3.663x | 128.0 | 128 | 0.0 |
| gather | 17.43 ms | 16.20 ms | 1.24 ms | 1.076x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.65 ms | 4.82 ms | -0.17 ms | 0.964x | 401.0 | 384 | 17.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.998x | 30.0 | 30 | 0.0 |

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 373.48 ms | 335.64 ms | 37.84 ms | 1.113x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 391.68 ms | 359.88 ms | 31.80 ms | 1.088x | 128.0 | 128 | 0.0 |
| command_buffer::update | 97.92 ms | 103.36 ms | -5.45 ms | 0.947x | 2394.0 | 126 | 2268.0 |
| command_buffer::execute | 109.20 ms | 105.70 ms | 3.50 ms | 1.033x | 2517.0 | 231 | 2286.0 |
| np.asarray(jax.Array) | 4.35 ms | 2.53 ms | 1.81 ms | 1.717x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 4.22 ms | 4.82 ms | -0.60 ms | 0.876x | 384.0 | 384 | 0.0 |
| gather | 16.42 ms | 16.20 ms | 0.22 ms | 1.014x | 1603.0 | 1603 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.003x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 496.40 ms | 359.88 ms | 136.52 ms | 1.379x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 471.98 ms | 335.64 ms | 136.34 ms | 1.406x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 139.60 ms | 105.70 ms | 33.90 ms | 1.321x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 129.31 ms | 103.36 ms | 25.94 ms | 1.251x | 2394.0 | 126 | 2268.0 |
| cpu | np.asarray(jax.Array) | 9.27 ms | 2.53 ms | 6.74 ms | 3.663x | 128.0 | 128 | 0.0 |
| cpu | gather | 12.11 ms | 10.83 ms | 1.28 ms | 1.118x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.66 ms | 3.84 ms | -0.18 ms | 0.953x | 175.0 | 175 | 0.0 |
| gpu | gather | 5.33 ms | 5.37 ms | -0.04 ms | 0.992x | 1585.0 | 1585 | 0.0 |

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 373.48 ms | 335.64 ms | 37.84 ms | 1.113x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 391.68 ms | 359.88 ms | 31.80 ms | 1.088x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::update | 97.92 ms | 103.36 ms | -5.45 ms | 0.947x | 2394.0 | 126 | 2268.0 |
| cpu | command_buffer::execute | 109.20 ms | 105.70 ms | 3.50 ms | 1.033x | 2517.0 | 231 | 2286.0 |
| cpu | np.asarray(jax.Array) | 4.35 ms | 2.53 ms | 1.81 ms | 1.717x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 3.25 ms | 3.84 ms | -0.59 ms | 0.845x | 175.0 | 175 | 0.0 |
| cpu | gather | 11.11 ms | 10.83 ms | 0.28 ms | 1.026x | 18.0 | 18 | 0.0 |
| gpu | gather | 5.32 ms | 5.37 ms | -0.05 ms | 0.990x | 1585.0 | 1585 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_only_2repeat.json`
- report: `results/gpu_matrix_decode_only_2repeat.md`
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
