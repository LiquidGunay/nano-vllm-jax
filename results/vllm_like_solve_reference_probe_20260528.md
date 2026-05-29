# vLLM-like Solve Reference Probe (2026-05-28)

## Length 128

- inverse max diff: ref `0.112078` -> vllm_like `0.46875`
- output max diff vs vLLM: ref `0.0062623` -> vllm_like `0.0258411`
- state max diff vs vLLM: ref `0.0723245` -> vllm_like `0.310912`

## Length 256

- inverse max diff: ref `0.112078` -> vllm_like `1.0625`
- output max diff vs vLLM: ref `0.0462755` -> vllm_like `0.189461`
- state max diff vs vLLM: ref `0.782248` -> vllm_like `1.86608`

## Length 512

- inverse max diff: ref `0.112078` -> vllm_like `1.28125`
- output max diff vs vLLM: ref `5.21796` -> vllm_like `6.95813`
- state max diff vs vLLM: ref `34.5676` -> vllm_like `68.6585`

