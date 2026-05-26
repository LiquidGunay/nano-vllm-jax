# GPU Matrix Report

- created_at_utc: `20260526_170052`
- dry_run: no
- repeats: 1
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260526_170052`
- output_json: `results/gpu_matrix_20260526_vk_cuda_gdn_decode_probe.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.757x (target 0.900x)
- JAX tok/s: 88.07
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 16.66

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | no | no | 88.07 | 116.37 | 0.757x | 104.74 | 16.66 | 78.02 | 1.129x |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: failed checks: minimum_repeats; speed_claim_ready=false; target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PjRtCApiLoadedExecutable::Execute | 273.86 ms | 289.24 ms | -15.38 ms | 0.947x | 44.0 | 140 | -96.0 |
| array.py:325 tolist | 413.02 ms | 427.68 ms | -14.66 ms | 0.966x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 412.90 ms | 427.55 ms | -14.65 ms | 0.966x | 16.0 | 16 | 0.0 |
| MemcpyD2D | 18.24 ms | 30.39 ms | -12.14 ms | 0.600x | 463.0 | 655 | -192.0 |
| command_buffer::execute | 223.13 ms | 229.21 ms | -6.08 ms | 0.973x | 1936.0 | 1936 | 0.0 |
| forward_step_token_ids_jit | 275.56 ms | 280.56 ms | -4.99 ms | 0.982x | 16.0 | 16 | 0.0 |
| transpose | 45.05 ms | 47.30 ms | -2.25 ms | 0.952x | 312.0 | 312 | 0.0 |
| command_buffer::update | 9.31 ms | 10.46 ms | -1.15 ms | 0.890x | 181.0 | 195 | -14.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260526_vk_cuda_gdn_decode_probe.json`
- report: `results/gpu_matrix_20260526_vk_cuda_gdn_decode_probe.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: no
- target_vllm_ratio_met: no
- JAX/vLLM: 0.757x
- JAX/reference: 1.129x
- TTFT delta vs reference: -42.62 ms
- ITL delta vs reference: -3.52 ms
- profile movement to explain:
- `PjRtCApiLoadedExecutable::Execute`: current 273.86 ms, reference 289.24 ms, delta -15.38 ms, ratio 0.947x, count delta -96.0
- `array.py:325 tolist`: current 413.02 ms, reference 427.68 ms, delta -14.66 ms, ratio 0.966x, count delta 0.0
- `np.asarray(jax.Array)`: current 412.90 ms, reference 427.55 ms, delta -14.65 ms, ratio 0.966x, count delta 0.0
- `MemcpyD2D`: current 18.24 ms, reference 30.39 ms, delta -12.14 ms, ratio 0.600x, count delta -192.0
- `command_buffer::execute`: current 223.13 ms, reference 229.21 ms, delta -6.08 ms, ratio 0.973x, count delta 0.0
- `forward_step_token_ids_jit`: current 275.56 ms, reference 280.56 ms, delta -4.99 ms, ratio 0.982x, count delta 0.0
- `transpose`: current 45.05 ms, reference 47.30 ms, delta -2.25 ms, ratio 0.952x, count delta 0.0
- `command_buffer::update`: current 9.31 ms, reference 10.46 ms, delta -1.15 ms, ratio 0.890x, count delta -14.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
