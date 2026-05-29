# vLLM Solve Swap Audit (2026-05-28)

## Length 128

| chain | out max vs JAX | out max vs vLLM | state max vs JAX | state max vs vLLM |
|---|---:|---:|---:|---:|
| full_jax | 0 | 0.0062623 | 0 | 0.0723245 |
| full_vllm | 0.0062623 | 0 | 0.0723245 | 0 |
| vllm_A_jax_solve_vllm_downstream | 0.0103393 | 0.0166016 | 0.171438 | 0.232065 |
| jax_A_vllm_solve_jax_downstream | 0.00686239 | 0.00170914 | 0.0644131 | 0.0189079 |
| jax_downstream_jax_solve_of_vllm_A | 0.00357267 | 0.00609687 | 0.0247762 | 0.0916103 |

## Length 256

| chain | out max vs JAX | out max vs vLLM | state max vs JAX | state max vs vLLM |
|---|---:|---:|---:|---:|
| full_jax | 0 | 0.0462755 | 0 | 0.782248 |
| full_vllm | 0.0462755 | 0 | 0.782248 | 0 |
| vllm_A_jax_solve_vllm_downstream | 0.0338139 | 0.0644531 | 0.909188 | 1.53387 |
| jax_A_vllm_solve_jax_downstream | 0.0538408 | 0.010095 | 0.922255 | 0.140007 |
| jax_downstream_jax_solve_of_vllm_A | 0.0138747 | 0.0463875 | 0.274166 | 0.952312 |

## Length 512

| chain | out max vs JAX | out max vs vLLM | state max vs JAX | state max vs vLLM |
|---|---:|---:|---:|---:|
| full_jax | 0 | 5.21796 | 0 | 34.5676 |
| full_vllm | 5.21796 | 0 | 34.5676 | 0 |
| vllm_A_jax_solve_vllm_downstream | 1.16131 | 5.6875 | 13.0038 | 44.3721 |
| jax_A_vllm_solve_jax_downstream | 5.79299 | 0.681554 | 36.7842 | 4.72333 |
| jax_downstream_jax_solve_of_vllm_A | 0.955322 | 6.17328 | 5.52141 | 40.089 |

