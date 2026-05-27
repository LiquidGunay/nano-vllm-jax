# GPU Matrix Report

- created_at_utc: `20260527_045735`
- dry_run: no
- repeats: 2
- run_dir: `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260527_045735`
- output_json: `results/gpu_matrix_20260527_vllm_random_longprefill_r2.json`
- jax_python: `.venv/bin/python` (available: yes)

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
| vllm_random_longprefill | gpu_paged_default | yes | no | 84.60 | 353.91 | 0.239x | 318.52 | 233.92 | 84.15 | 1.005x |

## Scheduler Diagnostics

| workload | config | prefill steps | decode steps | max prefill seqs | max step tokens | prefill step s | decode step s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vllm_random_longprefill | gpu_paged_default | 32.0 | 480.0 | 4.0 | 7267.0 | 17.74 s | 6.30 s |

## Prompt Provenance

| workload | config | source | dataset | prompts | seed | random input | random output | range ratio | current manifest | vLLM manifest | manifest match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vllm_random_longprefill | gpu_paged_default | vllm_random | random | 128 | 0 | 1280 | 16 | {"input":0.6,"output":0.0} | 9f98c47fe18b | 9f98c47fe18b | yes |

## Scoped Profile Range Medians

| workload | config | scope | bucket | median total | median count |
| --- | --- | --- | --- | --- | --- |
| vllm_random_longprefill | gpu_paged_default | cpu | array.py:325 tolist | 5855.57 ms | 213.0 |
| vllm_random_longprefill | gpu_paged_default | cpu | np.asarray(jax.Array) | 5853.70 ms | 213.0 |
| vllm_random_longprefill | gpu_paged_default | cpu | forward_step_token_ids_jit | 4075.78 ms | 213.5 |
| vllm_random_longprefill | gpu_paged_default | cpu | PjRtCApiLoadedExecutable::Execute | 4040.18 ms | 605.0 |
| vllm_random_longprefill | gpu_paged_default | cpu | command_buffer::execute | 3250.11 ms | 26961.0 |
| vllm_random_longprefill | gpu_paged_default | gpu | transpose | 651.69 ms | 4236.0 |
| vllm_random_longprefill | gpu_paged_default | gpu | MemcpyD2D | 348.77 ms | 3494.0 |
| vllm_random_longprefill | gpu_paged_default | cpu | command_buffer::update | 154.37 ms | 2419.5 |

## Top Scoped Profile Events

| workload | config | repeat | scope | event | total | count |
| --- | --- | --- | --- | --- | --- | --- |
| vllm_random_longprefill | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_744 | 812.60 ms | 672 |
| vllm_random_longprefill | gpu_paged_default | 1 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 471.81 ms | 168 |
| vllm_random_longprefill | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_729 | 375.76 ms | 162 |
| vllm_random_longprefill | gpu_paged_default | 1 | gpu | gemm_fusion_dot_general_746 | 373.38 ms | 312 |
| vllm_random_longprefill | gpu_paged_default | 1 | gpu | gemm_fusion_dot_2 | 364.65 ms | 336 |
| vllm_random_longprefill | gpu_paged_default | 1 | cpu | $llm_engine.py:365 generate_with_trace | 24229.54 ms | 1 |
| vllm_random_longprefill | gpu_paged_default | 1 | cpu | $threading.py:604 wait | 14221.42 ms | 2 |
| vllm_random_longprefill | gpu_paged_default | 1 | cpu | $threading.py:288 wait | 14221.41 ms | 2 |
| vllm_random_longprefill | gpu_paged_default | 1 | cpu | $llm_engine.py:279 iter_generate | 10519.75 ms | 905 |
| vllm_random_longprefill | gpu_paged_default | 1 | cpu | $llm_engine.py:134 step | 10451.98 ms | 214 |
| vllm_random_longprefill | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_744 | 812.59 ms | 672 |
| vllm_random_longprefill | gpu_paged_default | 2 | gpu | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params) | 471.67 ms | 168 |
| vllm_random_longprefill | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_729 | 375.75 ms | 162 |
| vllm_random_longprefill | gpu_paged_default | 2 | gpu | gemm_fusion_dot_general_746 | 373.35 ms | 312 |
| vllm_random_longprefill | gpu_paged_default | 2 | gpu | gemm_fusion_dot_2 | 364.59 ms | 336 |
| vllm_random_longprefill | gpu_paged_default | 2 | cpu | $llm_engine.py:365 generate_with_trace | 24187.81 ms | 1 |
| vllm_random_longprefill | gpu_paged_default | 2 | cpu | $threading.py:604 wait | 17807.97 ms | 2 |
| vllm_random_longprefill | gpu_paged_default | 2 | cpu | $threading.py:288 wait | 17807.94 ms | 2 |
| vllm_random_longprefill | gpu_paged_default | 2 | cpu | $llm_engine.py:279 iter_generate | 10506.24 ms | 905 |
| vllm_random_longprefill | gpu_paged_default | 2 | cpu | $llm_engine.py:134 step | 10437.35 ms | 214 |

## Acceptance Failures

- vllm_random_longprefill/gpu_paged_default: target_vllm_ratio_met=false target=0.9

## Top Profile Deltas Vs JAX Reference

### `vllm_random_longprefill/gpu_paged_default`

| bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np.asarray(jax.Array) | 5853.70 ms | 5818.43 ms | 35.28 ms | 1.006x | 213.0 | 213 | 0.0 |
| array.py:325 tolist | 5855.57 ms | 5820.54 ms | 35.02 ms | 1.006x | 213.0 | 213 | 0.0 |
| PjRtCApiLoadedExecutable::Execute | 4040.18 ms | 4072.01 ms | -31.84 ms | 0.992x | 605.0 | 605 | 0.0 |
| forward_step_token_ids_jit | 4075.78 ms | 4106.93 ms | -31.15 ms | 0.992x | 213.5 | 213 | 0.5 |
| command_buffer::execute | 3250.11 ms | 3267.29 ms | -17.19 ms | 0.995x | 26961.0 | 26961 | 0.0 |
| command_buffer::update | 154.37 ms | 165.99 ms | -11.61 ms | 0.930x | 2419.5 | 2585 | -165.5 |
| gather | 205.99 ms | 211.82 ms | -5.83 ms | 0.972x | 1420.0 | 1420 | 0.0 |
| MemcpyD2D | 380.96 ms | 383.63 ms | -2.66 ms | 0.993x | 6218.0 | 6218 | 0.0 |


## Scoped Profile Deltas Vs JAX Reference

### `vllm_random_longprefill/gpu_paged_default`

| scope | bucket | current | reference | delta | ratio | current count | reference count | count delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | np.asarray(jax.Array) | 5853.70 ms | 5818.43 ms | 35.28 ms | 1.006x | 213.0 | 213 | 0.0 |
| cpu | array.py:325 tolist | 5855.57 ms | 5820.54 ms | 35.02 ms | 1.006x | 213.0 | 213 | 0.0 |
| cpu | PjRtCApiLoadedExecutable::Execute | 4040.18 ms | 4072.01 ms | -31.84 ms | 0.992x | 605.0 | 605 | 0.0 |
| cpu | forward_step_token_ids_jit | 4075.78 ms | 4106.93 ms | -31.15 ms | 0.992x | 213.5 | 213 | 0.5 |
| cpu | command_buffer::execute | 3250.11 ms | 3267.29 ms | -17.19 ms | 0.995x | 26961.0 | 26961 | 0.0 |
| cpu | command_buffer::update | 154.37 ms | 165.99 ms | -11.61 ms | 0.930x | 2419.5 | 2585 | -165.5 |
| cpu | gather | 139.78 ms | 145.62 ms | -5.84 ms | 0.960x | 308.0 | 308 | 0.0 |
| cpu | MemcpyD2D | 32.20 ms | 34.88 ms | -2.68 ms | 0.923x | 2724.0 | 2724 | 0.0 |


## Logbook Entry Template

Copy this into `docs/optimization_logbook.md` after replacing the interpretation and decision text.

- artifact: `results/gpu_matrix_20260527_vllm_random_longprefill_r2.json`
- report: `results/gpu_matrix_20260527_vllm_random_longprefill_r2.md`
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
