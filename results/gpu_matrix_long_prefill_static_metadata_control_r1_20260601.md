# GPU Matrix Report

- created_at_utc: `20260601_141241`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260601_141241`
- output_json: `results/gpu_matrix_long_prefill_static_metadata_control_r1_20260601.json`
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
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | no | no | 22.70 | 116.37 | 0.195x | 104.74 | 82.04 | 78.02 | 0.291x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1.0 | 15.0 | 4.0 | 5120.0 | 0.03 s | 2.79 s |

## Host Replay Diagnostics

| workload | config | bucket | steps | decode steps | count | count/step | ms/step | ref count | ref count/step | ref ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | forward_step_token_ids_jit | 16.0 | 15.0 | 16.0 | 1.00 | 165.39 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | PjRtCApiLoadedExecutable::Execute | 16.0 | 15.0 | 59.0 | 3.69 | 165.38 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | command_buffer::execute | 16.0 | 15.0 | 496.0 | 31.00 | 1.03 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | command_buffer::update | 16.0 | 15.0 | 465.0 | 29.06 | 0.79 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | np.asarray(jax.Array) | 16.0 | 15.0 | 64.0 | 4.00 | 6.93 ms | - | - | - |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | cpu | forward_step_token_ids_jit | 2646.18 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | cpu | PjRtCApiLoadedExecutable::Execute | 2646.04 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | cpu | np.asarray(jax.Array) | 110.83 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | gpu | transpose | 20.71 ms | 253.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | cpu | command_buffer::execute | 16.43 ms | 496.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | gpu | MemcpyD2D | 15.54 ms | 237.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | cpu | command_buffer::update | 12.59 ms | 465.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | cpu | gather | 10.23 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | gpu | _gdn_fla_chunk_fwd_o_packed_kernel | 931.77 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | gpu | _gdn_fla_chunk_delta_h_packed_kernel | 863.15 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | gpu | _gdn_fla_chunk_scaled_dot_kkt_packed_kernel | 304.47 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.20 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | gpu | _gdn_fla_recompute_w_packed_kernel | 85.33 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | cpu | PjitFunction(compiled) | 5290.37 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | cpu | $llm_engine.py:632 generate_with_trace | 2818.28 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 2818.23 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | cpu | $llm_engine.py:161 step | 2817.62 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_static_metadata | 1 | cpu | $model_runner.py:3766 run | 2665.86 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_gdn_fla_decode_static_metadata: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode_static_metadata`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 2646.18 ms | 280.56 ms | 2365.62 ms | 9.432x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 2646.04 ms | 289.24 ms | 2356.81 ms | 9.148x | 59.0 | 140 | -81.0 |
| np.asarray(jax.Array) | 110.83 ms | 427.55 ms | -316.72 ms | 0.259x | 64.0 | 16 | 48.0 |
| command_buffer::execute | 16.43 ms | 229.21 ms | -212.78 ms | 0.072x | 496.0 | 1936 | -1440.0 |
| transpose | 20.71 ms | 47.30 ms | -26.59 ms | 0.438x | 253.0 | 312 | -59.0 |
| MemcpyD2D | 17.77 ms | 30.39 ms | -12.62 ms | 0.585x | 456.0 | 655 | -199.0 |
| command_buffer::update | 12.59 ms | 10.46 ms | 2.12 ms | 1.203x | 465.0 | 195 | 270.0 |
| gather | 14.88 ms | 14.55 ms | 0.33 ms | 1.022x | 88.0 | 103 | -15.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_long_prefill_static_metadata_control_r1_20260601.json`
- report: `results/gpu_matrix_long_prefill_static_metadata_control_r1_20260601.md`
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
