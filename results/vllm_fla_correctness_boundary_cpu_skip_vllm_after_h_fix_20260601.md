# vLLM FLA Correctness Boundary Audit

- Model: `Qwen/Qwen3.5-0.8B`
- Layer: `0`
- Chunk size: `64`

## Length 64

- Contract: packed Q `[64, 16, 128]`, packed V `[64, 16, 128]`, state `[1, 16, 128, 128]`
- First stage over `0.1` for `dtype_only`: `attention_inverse` (max `0.433928` at `[63, 1, 6]`)
- First stage over `0.1` for `vllm_default_vs_jax_bf16`: none
- First stage over `0.1` for `vllm_fp32_qkv_fp32_solve_vs_jax_fp32`: none
- First stage over `0.1` for `vllm_default_vs_vllm_like`: none

| comparison | output max | state max |
|---|---:|---:|
| JAX BF16-QKV vs JAX FP32-QKV | 0.00318132 | 0.0347214 |
| vLLM default vs JAX BF16-QKV | skip_vllm | skip_vllm |
| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | skip_vllm | skip_vllm |
| vLLM default vs JAX FP32-QKV | skip_vllm | skip_vllm |
| vLLM default vs vLLM FP32-QKV/FP32-solve | skip_vllm | skip_vllm |
| vLLM default vs vLLM-like BF16 path | 0.00563416 | 0.0815895 |
| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | skip_vllm | n/a |
- input fingerprint matches (vLLM-like vs case prepare+pack): q=True, k=True, v=True
## Length 128

- Contract: packed Q `[128, 16, 128]`, packed V `[128, 16, 128]`, state `[1, 16, 128, 128]`
- First stage over `0.1` for `dtype_only`: `attention_inverse` (max `0.434023` at `[123, 1, 2]`)
- First stage over `0.1` for `vllm_default_vs_jax_bf16`: none
- First stage over `0.1` for `vllm_fp32_qkv_fp32_solve_vs_jax_fp32`: none
- First stage over `0.1` for `vllm_default_vs_vllm_like`: `v_new` (max `0.252207` at `[104, 1, 35]`)

| comparison | output max | state max |
|---|---:|---:|
| JAX BF16-QKV vs JAX FP32-QKV | 0.00671354 | 0.0535412 |
| vLLM default vs JAX BF16-QKV | skip_vllm | skip_vllm |
| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | skip_vllm | skip_vllm |
| vLLM default vs JAX FP32-QKV | skip_vllm | skip_vllm |
| vLLM default vs vLLM FP32-QKV/FP32-solve | skip_vllm | skip_vllm |
| vLLM default vs vLLM-like BF16 path | 0.0143083 | 0.213898 |
| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | skip_vllm | n/a |
- input fingerprint matches (vLLM-like vs case prepare+pack): q=True, k=True, v=True

## Length 256

- Contract: packed Q `[256, 16, 128]`, packed V `[256, 16, 128]`, state `[1, 16, 128, 128]`
- First stage over `0.1` for `dtype_only`: `attention_inverse` (max `0.434023` at `[123, 1, 2]`)
- First stage over `0.1` for `vllm_default_vs_jax_bf16`: none
- First stage over `0.1` for `vllm_fp32_qkv_fp32_solve_vs_jax_fp32`: none
- First stage over `0.1` for `vllm_default_vs_vllm_like`: `h` (max `0.813834` at `[3, 1, 35, 126]`)

| comparison | output max | state max |
|---|---:|---:|
| JAX BF16-QKV vs JAX FP32-QKV | 0.0240848 | 0.319001 |
| vLLM default vs JAX BF16-QKV | skip_vllm | skip_vllm |
| vLLM FP32-QKV/FP32-solve vs JAX FP32-QKV | skip_vllm | skip_vllm |
| vLLM default vs JAX FP32-QKV | skip_vllm | skip_vllm |
| vLLM default vs vLLM FP32-QKV/FP32-solve | skip_vllm | skip_vllm |
| vLLM default vs vLLM-like BF16 path | 0.118502 | 1.06638 |
| vLLM BF16-QKV/FP32-solve inverse vs JAX BF16-QKV inverse | skip_vllm | n/a |
- input fingerprint matches (vLLM-like vs case prepare+pack): q=True, k=True, v=True
