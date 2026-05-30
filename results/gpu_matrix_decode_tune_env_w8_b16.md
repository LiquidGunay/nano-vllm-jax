# GPU Matrix Report

- created_at_utc: `20260530_123746`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_123746`
- output_json: `results/gpu_matrix_decode_tune_env_w8_b16.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | yes | no | 126.66 | 213.17 | 0.594x | 191.86 | 65.19 | 182.51 | 0.694x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.06 s | 0.95 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 622.32 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 594.59 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 169.78 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 159.81 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 13.96 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 7.93 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 5.41 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 4.26 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.46 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.78 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.31 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 1147.55 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 925.51 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 925.47 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 920.39 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 721.06 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | gemm_fusion_dot_265 | 127.48 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_92 | 103.44 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_62 | 62.82 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | fusion_922 | 37.29 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | PjitFunction(compiled) | 1303.47 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:632 generate_with_trace | 1100.13 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1100.07 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:161 step | 1093.78 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $model_runner.py:3737 run | 824.40 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 622.32 ms | 324.31 ms | 298.01 ms | 1.919x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 594.59 ms | 305.04 ms | 289.55 ms | 1.949x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 169.78 ms | 94.61 ms | 75.17 ms | 1.795x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 159.81 ms | 91.97 ms | 67.84 ms | 1.738x | 2394.0 | 127 | 2267.0 |
| np.asarray(jax.Array) | 7.93 ms | 2.66 ms | 5.28 ms | 2.988x | 128.0 | 128 | 0.0 |
| gather | 19.38 ms | 16.04 ms | 3.34 ms | 1.208x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.14 ms | 4.57 ms | 0.57 ms | 1.124x | 384.0 | 384 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | 0.00 ms | 1.001x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 622.32 ms | 324.31 ms | 298.01 ms | 1.919x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 594.59 ms | 305.04 ms | 289.55 ms | 1.949x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 169.78 ms | 94.61 ms | 75.17 ms | 1.795x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 159.81 ms | 91.97 ms | 67.84 ms | 1.738x | 2394.0 | 127 | 2267.0 |
| cpu | np.asarray(jax.Array) | 7.93 ms | 2.66 ms | 5.28 ms | 2.988x | 128.0 | 128 | 0.0 |
| cpu | gather | 13.96 ms | 10.69 ms | 3.27 ms | 1.306x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 4.26 ms | 3.59 ms | 0.67 ms | 1.187x | 175.0 | 175 | 0.0 |
| gpu | MemcpyD2D | 0.87 ms | 0.98 ms | -0.11 ms | 0.892x | 209.0 | 209 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_env_w8_b16.json`
- report: `results/gpu_matrix_decode_tune_env_w8_b16.md`
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
