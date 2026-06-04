# vLLM FLA Correctness Boundary Audit

- Model: `Qwen/Qwen3.5-0.8B`
- Layer: `0`
- Chunk size: `64`

## Length 64

- Contract: packed Q `[64, 16, 128]`, packed V `[64, 16, 128]`, state `[1, 16, 128, 128]`
- First stage over `0.1` for `dtype_only`: `attention_inverse` (max `0.434067` at `[63, 1, 6]`)
- First stage over `0.1` for `vllm_default_vs_jax_bf16`: none
- First stage over `0.1` for `vllm_default_vs_vllm_like`: none
- First stage over `0.1` for `vllm_fp32_qkv_fp32_solve_vs_jax_fp32`: `attention_inverse` (max `0.102924` at `[63, 1, 16]`)

| comparison | output max | state max |
|---|---:|---:|
| JAX BF16-QKV vs JAX FP32-QKV | 0.00326104 | 0.0343365 |
| vLLM default vs JAX BF16-QKV | 0.00330181 | 0.0677405 |
| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | 0.00138017 | 0.017447 |
| vLLM default vs JAX FP32-QKV | 0.00646404 | 0.102077 |
| vLLM default vs vLLM FP32-QKV/FP32-solve | 0.00522521 | 0.0927584 |
| vLLM default vs vLLM-like BF16 path | 0.00402832 | 0.0686681 |
| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | 0.0532913 | n/a |
- input fingerprint matches (vLLM vs vLLM-like): q=False, k=False, v=False
- packed fingerprint matches vs case prepare+pack: q=True, k=True, v=True, gate=True, beta=True
| vLLM staged default vs full chunk call | 0 | 0 |
## Length 128

- Contract: packed Q `[128, 16, 128]`, packed V `[128, 16, 128]`, state `[1, 16, 128, 128]`
- First stage over `0.1` for `dtype_only`: `attention_inverse` (max `0.434129` at `[123, 1, 2]`)
- First stage over `0.1` for `vllm_default_vs_jax_bf16`: `v_new` (max `0.250106` at `[114, 1, 35]`)
- First stage over `0.1` for `vllm_default_vs_vllm_like`: `v_new` (max `0.220703` at `[114, 1, 35]`)
- First stage over `0.1` for `vllm_fp32_qkv_fp32_solve_vs_jax_fp32`: `attention_inverse` (max `0.130129` at `[124, 1, 2]`)

| comparison | output max | state max |
|---|---:|---:|
| JAX BF16-QKV vs JAX FP32-QKV | 0.0067365 | 0.0565874 |
| vLLM default vs JAX BF16-QKV | 0.0150403 | 0.206221 |
| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | 0.00262409 | 0.0156543 |
| vLLM default vs JAX FP32-QKV | 0.0197993 | 0.217503 |
| vLLM default vs vLLM FP32-QKV/FP32-solve | 0.0207908 | 0.221146 |
| vLLM default vs vLLM-like BF16 path | 0.00756836 | 0.290734 |
| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | 0.0532913 | n/a |
- input fingerprint matches (vLLM vs vLLM-like): q=False, k=False, v=False
- packed fingerprint matches vs case prepare+pack: q=True, k=True, v=True, gate=True, beta=True
| vLLM staged default vs full chunk call | 0 | 0 |

## Length 256

- Contract: packed Q `[256, 16, 128]`, packed V `[256, 16, 128]`, state `[1, 16, 128, 128]`
- First stage over `0.1` for `dtype_only`: `attention_inverse` (max `0.434129` at `[123, 1, 2]`)
- First stage over `0.1` for `vllm_default_vs_jax_bf16`: `h` (max `0.209935` at `[2, 1, 35, 126]`)
- First stage over `0.1` for `vllm_default_vs_vllm_like`: `h` (max `0.84375` at `[3, 1, 35, 126]`)
- First stage over `0.1` for `vllm_fp32_qkv_fp32_solve_vs_jax_fp32`: `attention_inverse` (max `0.130129` at `[124, 1, 2]`)

| comparison | output max | state max |
|---|---:|---:|
| JAX BF16-QKV vs JAX FP32-QKV | 0.0259935 | 0.328157 |
| vLLM default vs JAX BF16-QKV | 0.0301856 | 0.969607 |
| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | 0.00653535 | 0.131717 |
| vLLM default vs JAX FP32-QKV | 0.0413183 | 1.07286 |
| vLLM default vs vLLM FP32-QKV/FP32-solve | 0.046002 | 1.15385 |
| vLLM default vs vLLM-like BF16 path | 0.108154 | 1.38389 |
| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | 0.0532913 | n/a |
- input fingerprint matches (vLLM vs vLLM-like): q=False, k=False, v=False
- packed fingerprint matches vs case prepare+pack: q=True, k=True, v=True, gate=True, beta=True
| vLLM staged default vs full chunk call | 0 | 0 |
