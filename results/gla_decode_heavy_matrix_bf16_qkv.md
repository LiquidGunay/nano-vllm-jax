# GPU Matrix Report

- created_at_utc: `20260530_072554`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260530_075700`
- output_json: `results/gla_decode_heavy_matrix_bf16_qkv.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | yes | no | 151.38 | 218.25 | 0.694x | 196.43 | 45.05 | 149.76 | 1.011x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.80 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 423.06 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 401.30 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 137.84 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 89.71 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 10.09 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 8.79 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | gather | 5.40 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | MemcpyD2D | 3.70 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 163.23 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_92 | 103.33 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_116 | 63.79 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_140 | 37.09 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 882.76 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 874.32 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 874.27 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 870.01 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 559.57 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | gemm_fusion_dot_265 | 163.25 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_92 | 103.33 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_116 | 63.77 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_62 | 62.50 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_140 | 37.09 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $threading.py:604 wait | 1058.36 ms | 2 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $threading.py:288 wait | 1058.35 ms | 2 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:632 generate_with_trace | 810.63 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 810.58 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:161 step | 806.46 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 423.06 ms | 309.40 ms | 113.65 ms | 1.367x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 401.30 ms | 292.44 ms | 108.86 ms | 1.372x | 283.0 | 283 | 0.0 |
| command_buffer::execute | 137.84 ms | 98.47 ms | 39.36 ms | 1.400x | 2517.0 | 231 | 2286.0 |
| command_buffer::update | 89.71 ms | 71.79 ms | 17.92 ms | 1.250x | 2394.0 | 126 | 2268.0 |
| np.asarray(jax.Array) | 8.79 ms | 4.65 ms | 4.14 ms | 1.892x | 128.0 | 128 | 0.0 |
| gather | 15.49 ms | 14.96 ms | 0.53 ms | 1.035x | 1603.0 | 1603 | 0.0 |
| transpose | 0.95 ms | 0.67 ms | 0.28 ms | 1.411x | 187.0 | 170 | 17.0 |
| MemcpyD2D | 4.73 ms | 4.60 ms | 0.14 ms | 1.030x | 401.0 | 384 | 17.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 423.06 ms | 309.40 ms | 113.65 ms | 1.367x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 401.30 ms | 292.44 ms | 108.86 ms | 1.372x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::execute | 137.84 ms | 98.47 ms | 39.36 ms | 1.400x | 2517.0 | 231 | 2286.0 |
| cpu | command_buffer::update | 89.71 ms | 71.79 ms | 17.92 ms | 1.250x | 2394.0 | 126 | 2268.0 |
| cpu | np.asarray(jax.Array) | 8.79 ms | 4.65 ms | 4.14 ms | 1.892x | 128.0 | 128 | 0.0 |
| cpu | gather | 10.09 ms | 9.61 ms | 0.48 ms | 1.050x | 18.0 | 18 | 0.0 |
| gpu | transpose | 0.59 ms | 0.26 ms | 0.33 ms | 2.253x | 119.0 | 102 | 17.0 |
| cpu | MemcpyD2D | 3.70 ms | 3.63 ms | 0.07 ms | 1.020x | 175.0 | 175 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gla_decode_heavy_matrix_bf16_qkv.json`
- report: `results/gla_decode_heavy_matrix_bf16_qkv.md`
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
