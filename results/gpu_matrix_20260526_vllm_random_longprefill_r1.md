# GPU Matrix Report

- created_at_utc: `20260526_172108`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260526_172108`
- output_json: `results/gpu_matrix_20260526_vllm_random_longprefill_r1.json`
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
| vllm_random_longprefill | gpu_paged_default | no | no | 84.45 | 354.24 | 0.238x | 318.81 | 234.37 | 83.68 | 1.009x |

## Acceptance Failures

- vllm_random_longprefill/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `vllm_random_longprefill/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward_step_token_ids_jit | 4095.62 ms | 4216.17 ms | -120.55 ms | 0.971x | 214.0 | 214 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 4075.13 ms | 4178.10 ms | -102.97 ms | 0.975x | 606.0 | 606 | 0.0 |
| array.py:325 tolist | 5860.71 ms | 5816.76 ms | 43.95 ms | 1.008x | 213.0 | 213 | 0.0 |
| np.asarray(jax.Array) | 5858.85 ms | 5814.93 ms | 43.92 ms | 1.008x | 213.0 | 213 | 0.0 |
| command_buffer::update | 148.82 ms | 180.65 ms | -31.83 ms | 0.824x | 2427.0 | 2429 | -2.0 |
| command_buffer::execute | 3249.11 ms | 3268.31 ms | -19.20 ms | 0.994x | 26967.0 | 26965 | 2.0 |
| gather | 230.42 ms | 211.80 ms | 18.62 ms | 1.088x | 1421.0 | 1421 | 0.0 |
| MemcpyD2D | 381.36 ms | 382.43 ms | -1.07 ms | 0.997x | 6224.0 | 6222 | 2.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260526_vllm_random_longprefill_r1.json`
- report: `results/gpu_matrix_20260526_vllm_random_longprefill_r1.md`
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
