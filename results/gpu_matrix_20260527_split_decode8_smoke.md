# GPU Matrix Report

- created_at_utc: `20260527_034250`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_034250`
- output_json: `results/gpu_matrix_20260527_split_decode8_smoke.json`
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
| vllm_random_longprefill_smoke | gpu_paged_default | no | no | 68.25 | 299.56 | 0.228x | 269.61 | 201.36 | 69.89 | 0.976x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vllm_random_longprefill_smoke | gpu_paged_default | 5.0 | 45.0 | 4.0 | 6409.0 | 2.48 s | 1.26 s |

## Acceptance Failures

- vllm_random_longprefill_smoke/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `vllm_random_longprefill_smoke/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 1607.85 ms | 1389.02 ms | 218.84 ms | 1.158x | 2905.0 | 2953 | -48.0 |
| PjRtCApiLoadedExecutable::Execute | 1665.04 ms | 1591.80 ms | 73.24 ms | 1.046x | 2966.0 | 2966 | 0.0 |
| forward_step_token_ids_jit | 1316.72 ms | 1300.62 ms | 16.11 ms | 1.012x | 50.0 | 50 | 0.0 |
| command_buffer::execute | 965.94 ms | 957.42 ms | 8.52 ms | 1.009x | 8570.0 | 8570 | 0.0 |
| MemcpyD2D | 226.06 ms | 221.07 ms | 4.99 ms | 1.023x | 4076.0 | 4076 | 0.0 |
| np.asarray(jax.Array) | 1329.63 ms | 1332.09 ms | -2.46 ms | 0.998x | 50.0 | 50 | 0.0 |
| array.py:325 tolist | 1330.19 ms | 1332.61 ms | -2.42 ms | 0.998x | 50.0 | 50 | 0.0 |
| command_buffer::update | 61.45 ms | 60.32 ms | 1.13 ms | 1.019x | 392.0 | 392 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_split_decode8_smoke.json`
- report: `results/gpu_matrix_20260527_split_decode8_smoke.md`
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
