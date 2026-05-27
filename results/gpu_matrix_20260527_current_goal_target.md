# GPU Matrix Report

- created_at_utc: `20260527_035342`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_035342`
- output_json: `results/gpu_matrix_20260527_current_goal_target.json`
- jax_python: `/mountpoint/.exp/nano-vllm-jax/.venv/bin/python` (available: yes)

## Goal Target

- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.781x (target 0.900x)
- JAX tok/s: 90.87
- vLLM tok/s: 116.37
- target tok/s: 104.74
- gap to target tok/s: 13.86

## Matrix

| workload | config | ready | target met | JAX tok/s | vLLM tok/s | JAX/vLLM | target tok/s | gap tok/s | JAX ref tok/s | JAX/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | yes | no | 90.87 | 116.37 | 0.781x | 104.74 | 13.86 | 78.02 | 1.165x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_prefill_512_2048 | gpu_paged_default | 1.0 | 15.0 | 4.0 | 5120.0 | 0.53 s | 0.17 s |

## Acceptance Failures

- long_prefill_512_2048/gpu_paged_default: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `long_prefill_512_2048/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 397.79 ms | 427.55 ms | -29.77 ms | 0.930x | 16.0 | 16 | 0.0 |
| array.py:325 tolist | 397.93 ms | 427.68 ms | -29.75 ms | 0.930x | 16.0 | 16 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 266.74 ms | 289.24 ms | -22.50 ms | 0.922x | 44.0 | 140 | -96.0 |
| MemcpyD2D | 18.45 ms | 30.39 ms | -11.94 ms | 0.607x | 463.0 | 655 | -192.0 |
| forward_step_token_ids_jit | 268.76 ms | 280.56 ms | -11.80 ms | 0.958x | 16.0 | 16 | 0.0 |
| command_buffer::execute | 223.46 ms | 229.21 ms | -5.75 ms | 0.975x | 1936.0 | 1936 | 0.0 |
| transpose | 45.07 ms | 47.30 ms | -2.23 ms | 0.953x | 312.0 | 312 | 0.0 |
| command_buffer::update | 8.59 ms | 10.46 ms | -1.87 ms | 0.821x | 183.0 | 195 | -12.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_current_goal_target.json`
- report: `results/gpu_matrix_20260527_current_goal_target.md`
- target: `long_prefill_512_2048/gpu_paged_default`
- speed_claim_ready: yes
- target_vllm_ratio_met: no
- JAX/vLLM: 0.781x
- JAX/reference: 1.165x
- TTFT delta vs reference: -48.83 ms
- ITL delta vs reference: -4.59 ms
- profile movement to explain:
- `np.asarray(jax.Array)`: current 397.79 ms, reference 427.55 ms, delta -29.77 ms, ratio 0.930x, count delta 0.0
- `array.py:325 tolist`: current 397.93 ms, reference 427.68 ms, delta -29.75 ms, ratio 0.930x, count delta 0.0
- `PjRtCApiLoadedExecutable::Execute`: current 266.74 ms, reference 289.24 ms, delta -22.50 ms, ratio 0.922x, count delta -96.0
- `MemcpyD2D`: current 18.45 ms, reference 30.39 ms, delta -11.94 ms, ratio 0.607x, count delta -192.0
- `forward_step_token_ids_jit`: current 268.76 ms, reference 280.56 ms, delta -11.80 ms, ratio 0.958x, count delta 0.0
- `command_buffer::execute`: current 223.46 ms, reference 229.21 ms, delta -5.75 ms, ratio 0.975x, count delta 0.0
- `transpose`: current 45.07 ms, reference 47.30 ms, delta -2.23 ms, ratio 0.953x, count delta 0.0
- `command_buffer::update`: current 8.59 ms, reference 10.46 ms, delta -1.87 ms, ratio 0.821x, count delta -12.0
- interpretation: <explain whether the profile movement supports the claimed change>
- decision: <keep/reject/follow up, with reason>
