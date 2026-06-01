# GPU Matrix Report

- created_at_utc: `20260601_045601`
- dry_run: no
- repeats: 1
- run_dir: `results/gpu_matrix_runs/20260601_gdn_strict_kernel_longprefill`
- output_json: `results/gpu_matrix_gdn_strict_kernel_longprefill_20260601.json`
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
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | no | no | 22.80 | 116.37 | 0.196x | 104.74 | 81.93 | 78.02 | 0.292x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1.0 | 15.0 | 4.0 | 5120.0 | 0.03 s | 2.77 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | forward_step_token_ids_jit | 78.38 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | PjRtCApiLoadedExecutable::Execute | 77.13 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | np.asarray(jax.Array) | 34.05 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | transpose | 20.71 ms | 253.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::execute | 18.34 ms | 496.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | command_buffer::update | 16.06 ms | 460.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | gpu | MemcpyD2D | 15.55 ms | 237.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | cpu | gather | 10.89 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | _gdn_fla_chunk_fwd_o_packed_kernel | 931.80 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | _gdn_fla_chunk_delta_h_packed_kernel | 842.48 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | _gdn_fla_chunk_scaled_dot_kkt_packed_kernel | 302.92 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.10 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | gpu | _gdn_fla_recompute_w_packed_kernel | 85.32 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:632 generate_with_trace | 2802.70 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 2802.64 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $llm_engine.py:161 step | 2801.67 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $profiler.py:381 wrapper | 2678.10 ms | 144 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_bf16_qkv | 1 | cpu | $scheduler.py:151 schedule | 2655.63 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode_bf16_qkv`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 34.05 ms | 427.55 ms | -393.50 ms | 0.080x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 77.13 ms | 289.24 ms | -212.11 ms | 0.267x | 59.0 | 140 | -81.0 |
| command_buffer::execute | 18.34 ms | 229.21 ms | -210.87 ms | 0.080x | 496.0 | 1936 | -1440.0 |
| forward_step_token_ids_jit | 78.38 ms | 280.56 ms | -202.17 ms | 0.279x | 16.0 | 16 | 0.0 |
| transpose | 20.71 ms | 47.30 ms | -26.59 ms | 0.438x | 253.0 | 312 | -59.0 |
| MemcpyD2D | 18.20 ms | 30.39 ms | -12.19 ms | 0.599x | 456.0 | 655 | -199.0 |
| command_buffer::update | 16.06 ms | 10.46 ms | 5.60 ms | 1.535x | 460.0 | 195 | 265.0 |
| gather | 15.57 ms | 14.55 ms | 1.02 ms | 1.070x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_gdn_strict_kernel_longprefill_20260601.json`
- report: `results/gpu_matrix_gdn_strict_kernel_longprefill_20260601.md`
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
