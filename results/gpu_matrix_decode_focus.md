# GPU Matrix Report

- created_at_utc: `20260530_075643`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_075643`
- output_json: `results/gpu_matrix_decode_focus.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

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
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | no | no | 131.14 | - | - | - | - | 137.25 | 0.955x |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | yes | no | 88.63 | 116.37 | 0.762x | 104.74 | 16.11 | 78.02 | 1.136x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1.0 | 127.0 | 1.0 | 128.0 | 0.06 s | 0.92 s |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1.0 | 15.0 | 4.0 | 5120.0 | 0.25 s | 0.47 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 549.73 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 520.73 ms | 283.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | forward_step_token_ids_jit | 357.99 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | PjRtCApiLoadedExecutable::Execute | 356.21 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 236.10 ms | 2224.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::execute | 181.25 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | cpu | command_buffer::update | 118.91 ms | 2394.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | gpu | transpose | 73.14 ms | 1547.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_265 | 163.28 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_92 | 103.33 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | gpu | input_reduce_fusion_140 | 37.08 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 1273.96 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 1072.33 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 1072.28 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 1064.89 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 1 | cpu | $model_runner.py:3737 run | 820.22 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | gemm_fusion_dot_265 | 163.21 ms | 127 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_92 | 103.34 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | gpu | input_reduce_fusion_140 | 37.09 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:632 generate_with_trace | 885.10 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 885.05 ms | 1 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | PjitFunction(compiled) | 882.29 ms | 256 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:161 step | 880.16 ms | 128 |
| decode_heavy_128x128 | gpu_paged_gdn_fla_decode | 2 | cpu | $model_runner.py:3737 run | 565.17 ms | 128 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.22 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.53 ms | 17 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.62 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.82 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | PjitFunction(compiled) | 761.37 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:632 generate_with_trace | 721.67 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 721.61 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | $llm_engine.py:161 step | 720.24 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 1 | cpu | MemcpyH2D | 616.87 ms | 106 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.27 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | gpu | input_multiply_reduce_select_transpose_fusion_16 | 31.54 ms | 17 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.55 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params) | 26.83 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:632 generate_with_trace | 718.22 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 718.16 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | cpu | $llm_engine.py:161 step | 717.24 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | cpu | PjitFunction(compiled) | 665.20 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode | 2 | cpu | $model_runner.py:3737 run | 353.22 ms | 16 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_gdn_fla_decode: failed checks: vllm_reference_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9
- long_prefill_512_2048/gpu_paged_gdn_fla_decode: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 520.73 ms | 451.12 ms | 69.61 ms | 1.154x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 549.73 ms | 480.78 ms | 68.95 ms | 1.143x | 128.0 | 128 | 0.0 |
| command_buffer::execute | 181.25 ms | 160.83 ms | 20.42 ms | 1.127x | 2517.0 | 231 | 2286.0 |
| np.asarray(jax.Array) | 16.46 ms | 10.26 ms | 6.20 ms | 1.604x | 128.0 | 128 | 0.0 |
| gather | 18.77 ms | 15.54 ms | 3.23 ms | 1.208x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 5.09 ms | 6.16 ms | -1.07 ms | 0.826x | 401.0 | 384 | 17.0 |
| command_buffer::update | 118.91 ms | 118.38 ms | 0.53 ms | 1.004x | 2394.0 | 126 | 2268.0 |
| transpose | 1.10 ms | 0.66 ms | 0.44 ms | 1.665x | 187.0 | 170 | 17.0 |

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 31.54 ms | 427.55 ms | -396.01 ms | 0.074x | 64.0 | 16 | 48.0 |
| forward_step_token_ids_jit | 357.99 ms | 280.56 ms | 77.44 ms | 1.276x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 356.21 ms | 289.24 ms | 66.98 ms | 1.232x | 59.0 | 140 | -81.0 |
| transpose | 73.14 ms | 47.30 ms | 25.84 ms | 1.546x | 1547.0 | 312 | 1235.0 |
| MemcpyD2D | 38.92 ms | 30.39 ms | 8.53 ms | 1.281x | 493.0 | 655 | -162.0 |
| command_buffer::update | 18.38 ms | 10.46 ms | 7.91 ms | 1.756x | 457.0 | 195 | 262.0 |
| command_buffer::execute | 236.10 ms | 229.21 ms | 6.89 ms | 1.030x | 2224.0 | 1936 | 288.0 |
| gather | 15.13 ms | 14.55 ms | 0.58 ms | 1.040x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_gdn_fla_decode`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 520.73 ms | 451.12 ms | 69.61 ms | 1.154x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 549.73 ms | 480.78 ms | 68.95 ms | 1.143x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::execute | 181.25 ms | 160.83 ms | 20.42 ms | 1.127x | 2517.0 | 231 | 2286.0 |
| cpu | np.asarray(jax.Array) | 16.46 ms | 10.26 ms | 6.20 ms | 1.604x | 128.0 | 128 | 0.0 |
| cpu | gather | 13.35 ms | 10.20 ms | 3.16 ms | 1.310x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 4.12 ms | 5.19 ms | -1.08 ms | 0.793x | 175.0 | 175 | 0.0 |
| cpu | command_buffer::update | 118.91 ms | 118.38 ms | 0.53 ms | 1.004x | 2394.0 | 126 | 2268.0 |
| gpu | transpose | 0.60 ms | 0.26 ms | 0.33 ms | 2.275x | 119.0 | 102 | 17.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_decode_focus.json`
- report: `results/gpu_matrix_decode_focus.md`
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
