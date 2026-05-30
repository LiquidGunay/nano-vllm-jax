# GPU Matrix Report

- created_at_utc: `20260530_070947`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260530_070947`
- output_json: `results/fla_decode_heavy_gdn_matrix.json`
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
| decode_heavy_128x128 | gpu_paged_default | no | no | 166.67 | - | - | - | - | - | - |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.72 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | cpu | forward_step_token_ids_jit | 368.14 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 349.69 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | command_buffer::execute | 122.14 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | command_buffer::update | 82.08 ms | 2394.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | gather | 10.62 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_default | gpu | gather | 5.40 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | np.asarray(jax.Array) | 4.32 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | MemcpyD2D | 3.12 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_265 | 163.30 ms | 127 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_92 | 103.33 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_116 | 63.79 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_62 | 62.52 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_140 | 37.09 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 765.35 ms | 1 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 765.31 ms | 1 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 762.15 ms | 128 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 723.22 ms | 256 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 462.02 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_default: failed checks: correctness_checked, exact_generated_token_match, minimum_repeats, vllm_reference_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

No profile deltas available.

## Scoped Profile Deltas Vs JAX Reference

No scoped profile deltas available.

## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/fla_decode_heavy_gdn_matrix.json`
- report: `results/fla_decode_heavy_gdn_matrix.md`
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
