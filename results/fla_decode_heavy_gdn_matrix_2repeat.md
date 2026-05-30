# GPU Matrix Report

- created_at_utc: `20260530_071135`
- dry_run: no
- repeats: 2
- run_dir: `results/gpu_matrix_runs/20260530_071300`
- output_json: `results/fla_decode_heavy_gdn_matrix_2repeat.json`
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
| decode_heavy_128x128 | gpu_paged_default | no | no | 165.29 | - | - | - | - | 165.58 | 0.998x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | 1.0 | 127.0 | 1.0 | 128.0 | 0.04 s | 0.73 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | tokenized_seed_repeat | tokenized_seed_repeat | 1 | 0 | 1280 | 16 | {"input":0.0,"output":0.0} | - | - | - |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | cpu | forward_step_token_ids_jit | 346.23 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 328.26 ms | 283.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | command_buffer::execute | 114.91 ms | 2517.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | command_buffer::update | 73.78 ms | 2403.5 |
| decode_heavy_128x128 | gpu_paged_default | cpu | gather | 10.41 ms | 18.0 |
| decode_heavy_128x128 | gpu_paged_default | gpu | gather | 5.43 ms | 1585.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | np.asarray(jax.Array) | 4.79 ms | 128.0 |
| decode_heavy_128x128 | gpu_paged_default | cpu | MemcpyD2D | 3.01 ms | 175.0 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | gemm_fusion_dot_265 | 163.20 ms | 127 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_92 | 103.34 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_default | 1 | gpu | input_reduce_fusion_140 | 37.07 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $llm_engine.py:632 generate_with_trace | 769.15 ms | 1 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 769.10 ms | 1 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $llm_engine.py:161 step | 765.90 ms | 128 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | PjitFunction(compiled) | 687.88 ms | 256 |
| decode_heavy_128x128 | gpu_paged_default | 1 | cpu | $model_runner.py:3737 run | 442.56 ms | 128 |
| decode_heavy_128x128 | gpu_paged_default | 2 | gpu | gemm_fusion_dot_265 | 163.26 ms | 127 |
| decode_heavy_128x128 | gpu_paged_default | 2 | gpu | input_reduce_fusion_92 | 103.34 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_default | 2 | gpu | input_reduce_fusion_116 | 63.78 ms | 3048 |
| decode_heavy_128x128 | gpu_paged_default | 2 | gpu | input_reduce_fusion_62 | 62.51 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_default | 2 | gpu | input_reduce_fusion_140 | 37.07 ms | 2286 |
| decode_heavy_128x128 | gpu_paged_default | 2 | cpu | $llm_engine.py:632 generate_with_trace | 771.89 ms | 1 |
| decode_heavy_128x128 | gpu_paged_default | 2 | cpu | $llm_engine.py:661 _generate_with_trace_deferred_tokens | 771.84 ms | 1 |
| decode_heavy_128x128 | gpu_paged_default | 2 | cpu | $llm_engine.py:161 step | 768.65 ms | 128 |
| decode_heavy_128x128 | gpu_paged_default | 2 | cpu | PjitFunction(compiled) | 671.67 ms | 256 |
| decode_heavy_128x128 | gpu_paged_default | 2 | cpu | $model_runner.py:3737 run | 434.43 ms | 128 |

## Acceptance Failures

- decode_heavy_128x128/gpu_paged_default: failed checks: correctness_checked, exact_generated_token_match, vllm_reference_present; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 328.26 ms | 332.32 ms | -4.06 ms | 0.988x | 283.0 | 283 | 0.0 |
| forward_step_token_ids_jit | 346.23 ms | 350.18 ms | -3.95 ms | 0.989x | 128.0 | 128 | 0.0 |
| np.asarray(jax.Array) | 4.79 ms | 2.36 ms | 2.43 ms | 2.028x | 128.0 | 128 | 0.0 |
| command_buffer::update | 73.78 ms | 74.76 ms | -0.98 ms | 0.987x | 2403.5 | 2413 | -9.5 |
| command_buffer::execute | 114.91 ms | 115.20 ms | -0.29 ms | 0.998x | 2517.0 | 2517 | 0.0 |
| gather | 15.85 ms | 16.09 ms | -0.24 ms | 0.985x | 1603.0 | 1603 | 0.0 |
| MemcpyD2D | 4.06 ms | 4.01 ms | 0.05 ms | 1.013x | 401.0 | 401 | 0.0 |
| transpose | 0.94 ms | 0.93 ms | 0.00 ms | 1.005x | 187.0 | 187 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `decode_heavy_128x128/gpu_paged_default`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | PjRtCApiLoadedExecutable::Execute | 328.26 ms | 332.32 ms | -4.06 ms | 0.988x | 283.0 | 283 | 0.0 |
| cpu | forward_step_token_ids_jit | 346.23 ms | 350.18 ms | -3.95 ms | 0.989x | 128.0 | 128 | 0.0 |
| cpu | np.asarray(jax.Array) | 4.79 ms | 2.36 ms | 2.43 ms | 2.028x | 128.0 | 128 | 0.0 |
| cpu | command_buffer::update | 73.78 ms | 74.76 ms | -0.98 ms | 0.987x | 2403.5 | 2413 | -9.5 |
| cpu | command_buffer::execute | 114.91 ms | 115.20 ms | -0.29 ms | 0.998x | 2517.0 | 2517 | 0.0 |
| cpu | gather | 10.41 ms | 10.65 ms | -0.23 ms | 0.978x | 18.0 | 18 | 0.0 |
| cpu | MemcpyD2D | 3.01 ms | 2.96 ms | 0.05 ms | 1.018x | 175.0 | 175 | 0.0 |
| cpu | transpose | 0.34 ms | 0.34 ms | 0.01 ms | 1.019x | 68.0 | 68 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/fla_decode_heavy_gdn_matrix_2repeat.json`
- report: `results/fla_decode_heavy_gdn_matrix_2repeat.md`
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
