# TPU Qwen3.5 2B/4B and vLLM Benchmark Notes

Date: 2026-05-06

Host: `nano-vllm-tpu-2404-run`

Project: `project-b9551f07-5f68-491a-8a0`

Zone: `europe-west4-a`

Accelerator: single `v6e-1` TPU VM

## nano-vllm-jax: Qwen/Qwen3.5-2B

Command:

```bash
JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 .bench-venv/bin/python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-2B \
  --config-preset qwen3_5_2b \
  --platform tpu --require-tpu \
  --dtype bfloat16 --jax-execution jit --backend auto \
  --batch-prompts 50 --max-num-seqs 50 \
  --prompt-length-min 32 --prompt-length-max 96 \
  --max-tokens 64 \
  --num-kvcache-blocks 1024 --max-kv-cache-mb 12288 \
  --prefill-buckets 64 --batch-size-buckets 1,50 \
  --warmup --repeats 1 \
  --num-speculative-tokens 1
```

Result:

| Model | Mode | Decode tok/s | E2E tok/s | Prefill tok/s | MTP acceptance | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-2B` | baseline | 1641.05 | 1242.16 | 4836.62 | n/a | 1.000x |
| `Qwen/Qwen3.5-2B` | MTP K=1 | 1549.83 | 1145.22 | measured separately in MTP path | 96.875% | 0.944x |

Interpretation: MTP K=1 still does not beat baseline on the 2B checkpoint. Acceptance is high, but the implementation remains slower because the accepted speculative step performs extra full-vocab projections and bookkeeping relative to the baseline decode step.

## nano-vllm-jax: Qwen/Qwen3.5-4B

Command:

```bash
JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 .bench-venv/bin/python benchmark_mtp1_engine.py \
  --model Qwen/Qwen3.5-4B \
  --config-preset hf \
  --platform tpu --require-tpu \
  --dtype bfloat16 --jax-execution jit --backend auto \
  --batch-prompts 24 --max-num-seqs 24 \
  --prompt-length-min 32 --prompt-length-max 96 \
  --max-tokens 64 \
  --num-kvcache-blocks 768 --max-kv-cache-mb 12288 \
  --prefill-buckets 64 --batch-size-buckets 1,24 \
  --warmup --repeats 1 \
  --num-speculative-tokens 1
```

Result:

| Model | Mode | Decode tok/s | E2E tok/s | Prefill tok/s | MTP acceptance | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-4B` | baseline | 594.89 | 487.70 | 2508.83 | n/a | 1.000x |
| `Qwen/Qwen3.5-4B` | MTP K=1 | 776.55 | 585.63 | measured separately in MTP path | 96.875% | 1.305x |

Interpretation: this is the first measured speedup. On 4B, the saved target-model step outweighs the extra MTP work and Python bookkeeping, so MTP K=1 improves decode throughput by about 30.5% on this fixed benchmark shape.

## nano-vllm-jax: Qwen/Qwen3.5-0.8B reference

Previously measured on the same TPU VM:

| Model | Mode | Decode tok/s | E2E tok/s | MTP acceptance | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-0.8B` | baseline | 1989.16 | 1361.33 | n/a | 1.000x |
| `Qwen/Qwen3.5-0.8B` | MTP K=1 fast path | 1682.16 | 1190.93 | 96.875% | 0.846x |

## vLLM TPU attempts

Installed:

```bash
python3 -m venv ~/vllm-tpu-venv
~/vllm-tpu-venv/bin/python -m pip install -q -U pip uv
~/vllm-tpu-venv/bin/uv pip install --python ~/vllm-tpu-venv/bin/python vllm-tpu
```

Installed package: `vllm-tpu==0.19.0`.

The installed TPU backend reports native architectures including `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, and `Qwen2ForCausalLM`, but not `Qwen3_5ForConditionalGeneration` or `Qwen3_5ForCausalLM`.

### Official Qwen3.5 checkpoint

Command shape:

```bash
HF_HUB_OFFLINE=1 JAX_PLATFORMS=tpu,cpu ~/vllm-tpu-venv/bin/vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen3.5-0.8B \
  --dataset-name random \
  --random-input-len 64 \
  --random-output-len 32 \
  --num-prompts 50 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 256 \
  --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

Result: no throughput number. vLLM reaches engine initialization, disables multimodal inputs, then falls back because Qwen3.5 is not registered in `tpu-inference`, and the fallback OpenXLA compilation fails with HBM allocation errors on v6e-1.

### Text-only shim attempt

I also created a local causal-only shim from the official 0.8B checkpoint by:

1. Using `text_config` as top-level `config.json`.
2. Setting architecture to `Qwen3_5ForCausalLM`.
3. Rewriting safetensor keys from `model.language_model.*` to `model.*`.
4. Disabling multimodal inputs with `--limit-mm-per-prompt '{"image":0,"video":0}'`.

Result: no throughput number. vLLM still routes through the Qwen3.5 fallback path and expects conditional-wrapper-style parameter names, so the shim is not a reliable vLLM baseline.

## vLLM MTP status

The `vllm bench throughput` CLI accepts `--speculative-config`, including MTP syntax such as:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

However, vLLM TPU did not reach a runnable baseline for Qwen3.5 on this VM, so there is no valid vLLM Qwen3.5 MTP throughput number from this run.

Also, the official vLLM TPU feature matrix lists TPU speculative decoding support for Ngram and Eagle3, not Qwen3.5 MTP. Treat Qwen3.5 MTP on `vllm-tpu==0.19.0` as unsupported/unvalidated unless a newer TPU build explicitly registers this architecture and speculative method.

## Code changes made for this benchmark

1. `benchmark_mtp1_engine.py` now has `--config-preset` so larger Qwen3.5 checkpoints do not accidentally use the 0.8B default architecture.
2. `Qwen3_5Config.qwen3_5_2b()` was corrected to match the downloaded HF `Qwen/Qwen3.5-2B` text config:
   `intermediate_size=6144`, `num_attention_heads=8`, `num_key_value_heads=2`.
3. `--config-preset hf` reads architecture dimensions from the HF `config.json` `text_config`, which was needed for `Qwen/Qwen3.5-4B`.
4. The hybrid GDN recurrent-state table now uses `linear_num_value_heads` instead of `linear_num_key_heads`. This fixes models such as 4B where value-head count exceeds key-head count (`linear_num_key_heads=16`, `linear_num_value_heads=32`).
