# GPU Matrix Report

- created_at_utc: `20260601_142616`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260601_142616`
- output_json: `results/gpu_matrix_long_prefill_kkt_fwd_o_delta_block_dot_r1_20260601.json`
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
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | no | no | 82.37 | 116.37 | 0.708x | 104.74 | 22.36 | 78.02 | 1.056x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1.0 | 15.0 | 4.0 | 5120.0 | 0.04 s | 0.74 s |

## Host Replay Diagnostics

| workload | config | bucket | steps | decode steps | count | count/step | ms/step | ref count | ref count/step | ref ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | forward_step_token_ids_jit | 16.0 | 15.0 | 16.0 | 1.00 | 37.35 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | PjRtCApiLoadedExecutable::Execute | 16.0 | 15.0 | 59.0 | 3.69 | 37.26 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | command_buffer::execute | 16.0 | 15.0 | 496.0 | 31.00 | 1.12 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | command_buffer::update | 16.0 | 15.0 | 465.0 | 29.06 | 0.92 ms | - | - | - |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | np.asarray(jax.Array) | 16.0 | 15.0 | 64.0 | 4.00 | 6.94 ms | - | - | - |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | tokenized_seed_repeat | tokenized_seed_repeat | 4 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | cpu | forward_step_token_ids_jit | 597.55 ms | 16.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | cpu | PjRtCApiLoadedExecutable::Execute | 596.15 ms | 59.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | cpu | np.asarray(jax.Array) | 111.12 ms | 64.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | gpu | transpose | 20.69 ms | 253.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | cpu | command_buffer::execute | 17.98 ms | 496.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | gpu | MemcpyD2D | 15.48 ms | 237.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | cpu | command_buffer::update | 14.69 ms | 465.0 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | cpu | gather | 10.32 ms | 22.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params) | 103.06 ms | 72 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | gpu | _gdn_fla_recompute_w_packed_kernel | 85.34 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | gpu | _gdn_fla_recompute_u_packed_kernel | 82.42 ms | 18 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 27.60 ms | 12 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | gpu | gemm_fusion_dot_2 | 27.32 ms | 24 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | cpu | PjitFunction(compiled) | 1192.55 ms | 32 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | cpu | $llm_engine.py:632 generate_with_trace | 774.85 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 774.78 ms | 1 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | cpu | $llm_engine.py:161 step | 774.01 ms | 16 |
| long_prefill_512_2048 | gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot | 1 | cpu | $model_runner.py:3766 run | 618.34 ms | 16 |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 597.55 ms | 280.56 ms | 317.00 ms | 2.130x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 111.12 ms | 427.55 ms | -316.43 ms | 0.260x | 64.0 | 16 | 48.0 |
| PjRtCApiLoadedExecutable::Execute | 596.15 ms | 289.24 ms | 306.91 ms | 2.061x | 59.0 | 140 | -81.0 |
| command_buffer::execute | 17.98 ms | 229.21 ms | -211.23 ms | 0.078x | 496.0 | 1936 | -1440.0 |
| transpose | 20.69 ms | 47.30 ms | -26.61 ms | 0.437x | 253.0 | 312 | -59.0 |
| MemcpyD2D | 17.73 ms | 30.39 ms | -12.66 ms | 0.583x | 456.0 | 655 | -199.0 |
| command_buffer::update | 14.69 ms | 10.46 ms | 4.22 ms | 1.404x | 465.0 | 195 | 270.0 |
| gather | 14.96 ms | 14.55 ms | 0.41 ms | 1.028x | 88.0 | 103 | -15.0 |


## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_long_prefill_kkt_fwd_o_delta_block_dot_r1_20260601.json`
- report: `results/gpu_matrix_long_prefill_kkt_fwd_o_delta_block_dot_r1_20260601.md`
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
