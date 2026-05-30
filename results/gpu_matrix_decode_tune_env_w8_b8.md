# GPU Matrix Report

- created_at_utc: `20260530_123400`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_123400`
- output_json: `results/gpu_matrix_decode_tune_env_w8_b8.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | yes | no | 129.47 | 218.26 | 0.593x | 196.44 | 66.97 | 181.87 | 0.712x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1.0 | 127.0 | 1.0 | 128.0 | 0.08 s | 0.96 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | forward_step_token_ids_jit | 629.12 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | PjRtCApiLoadedExecutable::Execute | 601.93 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::update | 163.90 ms | 2403.5 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | command_buffer::execute | 162.13 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | gather | 35.79 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | np.asarray(jax.Array) | 15.49 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | gpu | gather | 5.31 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | cpu | MemcpyD2D | 4.34 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_92 | 103.45 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | input_reduce_fusion_62 | 62.79 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | gpu | fusion_922 | 37.28 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | PjitFunction(compiled) | 1555.91 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1300.20 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1300.14 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $llm_engine.py:161 step | 1293.28 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 1 | cpu | $model_runner.py:3737 run | 982.96 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | gemm_fusion_dot_265 | 127.45 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_92 | 103.44 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_116 | 64.07 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | input_reduce_fusion_62 | 62.78 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | gpu | fusion_922 | 37.30 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | PjitFunction(compiled) | 919.97 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:632 generate_with_trace | 791.65 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 791.61 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $llm_engine.py:161 step | 787.10 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_off_prefill | 2 | cpu | $model_runner.py:3737 run | 582.09 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 629.12 ms | 325.89 ms | 303.23 ms | 1.930x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 601.93 ms | 306.79 ms | 295.14 ms | 1.962x | 283.0 | 283 | 0.0 |
| command_buffer::update | 163.90 ms | 93.85 ms | 70.05 ms | 1.746x | 2403.5 | 126 | 2277.5 |
| command_buffer::execute | 162.13 ms | 96.45 ms | 65.68 ms | 1.681x | 2517.0 | 231 | 2286.0 |
| gather | 41.09 ms | 15.18 ms | 25.91 ms | 2.706x | 1603.0 | 1603 | 0.0 |
| np.asarray(jax.Array) | 15.49 ms | 6.91 ms | 8.58 ms | 2.242x | 128.0 | 128 | 0.0 |
| MemcpyD2D | 5.23 ms | 4.89 ms | 0.34 ms | 1.070x | 384.0 | 384 | 0.0 |
| transpose | 0.07 ms | 0.07 ms | -0.00 ms | 0.994x | 30.0 | 30 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_off_prefill`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 629.12 ms | 325.89 ms | 303.23 ms | 1.930x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 601.93 ms | 306.79 ms | 295.14 ms | 1.962x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::update | 163.90 ms | 93.85 ms | 70.05 ms | 1.746x | 2403.5 | 126 | 2277.5 |
| cpu | command_buffer::execute | 162.13 ms | 96.45 ms | 65.68 ms | 1.681x | 2517.0 | 231 | 2286.0 |
| cpu | gather | 35.79 ms | 9.84 ms | 25.95 ms | 3.638x | 18.0 | 18 | 0.0 |
| cpu | np.asarray(jax.Array) | 15.49 ms | 6.91 ms | 8.58 ms | 2.242x | 128.0 | 128 | 0.0 |
| cpu | MemcpyD2D | 4.34 ms | 3.91 ms | 0.43 ms | 1.110x | 175.0 | 175 | 0.0 |
| gpu | MemcpyD2D | 0.89 ms | 0.98 ms | -0.09 ms | 0.911x | 209.0 | 209 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_tune_env_w8_b8.json`
- report: `results/gpu_matrix_decode_tune_env_w8_b8.md`
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
