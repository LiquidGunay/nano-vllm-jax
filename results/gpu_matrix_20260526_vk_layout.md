# GPU Matrix Report

- created_at_utc: `20260526_164219`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260526_164219`
- output_json: `results/gpu_matrix_20260526_vk_layout.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.779x (target 0.900x)
- JAX tok/s: 90.65
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 14.08

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | yes | no | 90.65 | 116.37 | 0.779x | 104.74 | 14.08 | 78.02 | 1.162x |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array.py:325 tolist | 397.19 ms | 427.68 ms | -30.50 ms | 0.929x | 16.0 | 16 | 0.0 |
| np.asarray(jax.Array) | 397.07 ms | 427.55 ms | -30.48 ms | 0.929x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 269.72 ms | 289.24 ms | -19.52 ms | 0.933x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.24 ms | 30.39 ms | -12.14 ms | 0.600x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 271.43 ms | 280.56 ms | -9.12 ms | 0.967x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 223.76 ms | 229.21 ms | -5.45 ms | 0.976x | 1936.0 | 1936 | 0.0 |
| gather | 17.20 ms | 14.55 ms | 2.64 ms | 1.181x | 103.0 | 103 | 0.0 |
| transpose | 45.03 ms | 47.30 ms | -2.27 ms | 0.952x | 312.0 | 312 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260526_vk_layout.json`
- report: `results/gpu_matrix_20260526_vk_layout.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.779x
- JAX/reference: 1.162x
- TTFT delta vs reference: -48.03 ms
- ITL delta vs reference: -4.53 ms
- profile movement to explain:
- `array.py:325 tolist`: current 397.19 ms, reference 427.68 ms, delta -30.50 ms, ratio 0.929x, count delta 0.0
- `np.asarray(jax.Array)`: current 397.07 ms, reference 427.55 ms, delta -30.48 ms, ratio 0.929x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 269.72 ms, reference 289.24 ms, delta -19.52 ms, ratio 0.933x, count delta -96.0
- `MemcpyD2D`: current 18.24 ms, reference 30.39 ms, delta -12.14 ms, ratio 0.600x, count delta -192.0
- `forward_step_token_ids_jit`: current 271.43 ms, reference 280.56 ms, delta -9.12 ms, ratio 0.967x, count delta 0.0
- `command_buffer::execute`: current 223.76 ms, reference 229.21 ms, delta -5.45 ms, ratio 0.976x, count delta 0.0
- `gather`: current 17.20 ms, reference 14.55 ms, delta 2.64 ms, ratio 1.181x, count delta 0.0
- `transpose`: current 45.03 ms, reference 47.30 ms, delta -2.27 ms, ratio 0.952x, count delta 0.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
