# GPU Matrix Report

- created_at_utc: `20260528_101213`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260528_101213`
- output_json: `results/gpu_matrix_20260528_triton_fla_padded_jit_fixed.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.196x (target 0.900x)
- JAX tok/s: 22.76
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 81.97

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 22.76 | 116.37 | 0.196x | 104.74 | 81.97 | 78.02 | 0.292x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.03 s | 2.78 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | cpu | forward_step_token_ids_jit | 54.91 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 54.89 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | np.asarray(jax.Array) | 33.53 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | transpose | 20.66 ms | 253.0 |
| long_prefill_512_2048 | gpu_paged_default | gpu | MemcpyD2D | 15.54 ms | 237.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::execute | 13.30 ms | 226.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | command_buffer::update | 12.38 ms | 194.0 |
| long_prefill_512_2048 | gpu_paged_default | cpu | gather | 9.67 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | _gdn_fla_chunk_fwd_o_packed_kernel | 931.89 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | _gdn_fla_chunk_delta_h_packed_kernel | 860.83 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | _gdn_fla_chunk_scaled_dot_kkt_packed_kernel | 303.99 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.16 ms | 72 |
| long_prefill_512_2048 | gpu_paged_default | 1 | gpu | _gdn_fla_recompute_w_packed_kernel | 85.33 ms | 18 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | MemcpyH2D | 7778.68 ms | 106 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 2810.33 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 2810.27 ms | 1 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 2809.46 ms | 16 |
| long_prefill_512_2048 | gpu_paged_default | 1 | cpu | $profiler.py:381 wrapper | 2718.34 ms | 144 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats, profile_counters_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9; missing_profile_counters=1

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 33.53 ms | 427.55 ms | -394.02 ms | 0.078x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 54.89 ms | 289.24 ms | -234.35 ms | 0.190x | 59.0 | 140 | -81.0 |
| forward_step_token_ids_jit | 54.91 ms | 280.56 ms | -225.64 ms | 0.196x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 13.30 ms | 229.21 ms | -215.91 ms | 0.058x | 226.0 | 1936 | -1710.0 |
| transpose | 20.66 ms | 47.30 ms | -26.64 ms | 0.437x | 253.0 | 312 | -59.0 |
| MemcpyD2D | 17.77 ms | 30.39 ms | -12.62 ms | 0.585x | 456.0 | 655 | -199.0 |
| command_buffer::update | 12.38 ms | 10.46 ms | 1.92 ms | 1.183x | 194.0 | 195 | -1.0 |
| gather | 14.35 ms | 14.55 ms | -0.20 ms | 0.986x | 103.0 | 103 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260528_triton_fla_padded_jit_fixed.json`
- report: `results/gpu_matrix_20260528_triton_fla_padded_jit_fixed.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.196x
- JAX/reference: 0.292x
- TTFT delta vs reference: -555.44 ms
- ITL delta vs reference: -5.82 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 33.53 ms, reference 427.55 ms, delta -394.02 ms, ratio 0.078x, count delta 48.0
- `PjRtCApiLoadedExecutable::Execute`: current 54.89 ms, reference 289.24 ms, delta -234.35 ms, ratio 0.190x, count delta -81.0
- `forward_step_token_ids_jit`: current 54.91 ms, reference 280.56 ms, delta -225.64 ms, ratio 0.196x, count delta 0.0
- `command_buffer::execute`: current 13.30 ms, reference 229.21 ms, delta -215.91 ms, ratio 0.058x, count delta -1710.0
- `transpose`: current 20.66 ms, reference 47.30 ms, delta -26.64 ms, ratio 0.437x, count delta -59.0
- `MemcpyD2D`: current 17.77 ms, reference 30.39 ms, delta -12.62 ms, ratio 0.585x, count delta -199.0
- `command_buffer::update`: current 12.38 ms, reference 10.46 ms, delta 1.92 ms, ratio 1.183x, count delta -1.0
- `gather`: current 14.35 ms, reference 14.55 ms, delta -0.20 ms, ratio 0.986x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
