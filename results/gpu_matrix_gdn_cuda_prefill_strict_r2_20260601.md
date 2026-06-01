# GPU Matrix Report

- created_at_utc: `20260601_050053`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260601_gdn_cuda_prefill_strict_r2`
- output_json: `results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.json`
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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | yes | no | 165.88 | 213.54 | 0.777x | 192.18 | 26.30 | 151.84 | 1.092x |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | yes | no | 63.98 | 116.37 | 0.550x | 104.74 | 40.76 | 78.02 | 0.820x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 127.0 | 1.0 | 128.0 | 0.02 s | 0.74 s |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 15.0 | 4.0 | 5120.0 | 0.03 s | 0.97 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 365.40 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 346.08 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 120.60 ms | 2443.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 86.72 ms | 2413.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 61.25 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 61.04 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 34.68 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | transpose | 20.71 ms | 256.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_265 | 163.27 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_110 | 103.24 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_134 | 63.77 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_80 | 62.39 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | input_reduce_fusion_158 | 36.92 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 770.38 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 770.33 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 766.54 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | PjitFunction(compiled) | 704.84 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $model_runner.py:3737 run | 460.92 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | gemm_fusion_dot_265 | 163.22 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_110 | 103.24 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_134 | 63.75 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_80 | 62.41 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | input_reduce_fusion_158 | 36.91 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:632 generate_with_trace | 767.05 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 766.99 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:161 step | 763.45 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | PjitFunction(compiled) | 730.22 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $model_runner.py:3737 run | 473.83 ms | 128 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void (anonymous namespace)::Fp32GdnPrefillChunk32Kernel<64, false>(float const*, float const*, float const*, float const*, float const*, int const*, float const*, float*, float*, long, long, long, long, long) | 443.81 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.16 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.61 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 998.17 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 998.11 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 997.13 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $profiler.py:381 wrapper | 891.55 ms | 144 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $scheduler.py:151 schedule | 869.06 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | void (anonymous namespace)::Fp32GdnPrefillChunk32Kernel<64, false>(float const*, float const*, float const*, float const*, float const*, int const*, float const*, float*, float*, long, long, long, long, long) | 444.20 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.20 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:632 generate_with_trace | 998.69 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 998.61 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $llm_engine.py:161 step | 997.74 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $profiler.py:381 wrapper | 896.64 ms | 144 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 2 | cpu | $scheduler.py:151 schedule | 872.41 ms | 16 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv: target_vllm_ratio_met=false target=0.9
- long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 365.40 ms | 489.15 ms | -123.75 ms | 0.747x | 128.0 | 128 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 346.08 ms | 462.11 ms | -116.03 ms | 0.749x | 283.0 | 283 | 0.0 |
| command_buffer::update | 86.72 ms | 143.82 ms | -57.10 ms | 0.603x | 2413.0 | 127 | 2286.0 |
| command_buffer::execute | 120.60 ms | 145.58 ms | -24.97 ms | 0.828x | 2443.0 | 231 | 2212.0 |
| np.asarray(jax.Array) | 0.97 ms | 14.85 ms | -13.88 ms | 0.066x | 128.0 | 128 | 0.0 |
| gather | 15.89 ms | 19.45 ms | -3.57 ms | 0.817x | 1621.0 | 1603 | 18.0 |
| MemcpyD2D | 4.21 ms | 5.56 ms | -1.35 ms | 0.757x | 343.0 | 384 | -41.0 |
| transpose | 0.08 ms | 0.07 ms | 0.00 ms | 1.040x | 33.0 | 30 | 3.0 |

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 34.68 ms | 427.55 ms | -392.87 ms | 0.081x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 61.04 ms | 289.24 ms | -228.20 ms | 0.211x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 61.25 ms | 280.56 ms | -219.31 ms | 0.218x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 15.99 ms | 229.21 ms | -213.23 ms | 0.070x | 478.0 | 1936 | -1458.0 |
| transpose | 20.71 ms | 47.30 ms | -26.59 ms | 0.438x | 256.0 | 312 | -56.0 |
| MemcpyD2D | 13.10 ms | 30.39 ms | -17.29 ms | 0.431x | 438.0 | 655 | -217.0 |
| command_buffer::update | 13.03 ms | 10.46 ms | 2.57 ms | 1.245x | 457.0 | 195 | 262.0 |
| gather | 15.16 ms | 14.55 ms | 0.61 ms | 1.042x | 121.0 | 103 | 18.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode_bf16_qkv`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | forward_step_token_ids_jit | 365.40 ms | 489.15 ms | -123.75 ms | 0.747x | 128.0 | 128 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 346.08 ms | 462.11 ms | -116.03 ms | 0.749x | 283.0 | 283 | 0.0 |
| cpu | command_buffer::update | 86.72 ms | 143.82 ms | -57.10 ms | 0.603x | 2413.0 | 127 | 2286.0 |
| cpu | command_buffer::execute | 120.60 ms | 145.58 ms | -24.97 ms | 0.828x | 2443.0 | 231 | 2212.0 |
| cpu | np.asarray(jax.Array) | 0.97 ms | 14.85 ms | -13.88 ms | 0.066x | 128.0 | 128 | 0.0 |
| cpu | gather | 10.40 ms | 14.15 ms | -3.74 ms | 0.736x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.28 ms | 4.59 ms | -1.32 ms | 0.713x | 171.0 | 175 | -4.0 |
| gpu | gather | 5.48 ms | 5.31 ms | 0.18 ms | 1.033x | 1603.0 | 1585 | 18.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.json`
- report: `results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.md`
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
