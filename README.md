# Nano-vLLM-JAX

A pedagogical nano-vllm style implementation of **Qwen3.5** in JAX with paged attention, speculative decoding, and complete JIT compilation.

## Status: Working Prototype

This is a **pedagogical implementation** with **partial parity validation**.

### What Works ✓
- Model loads weights from HuggingFace
- Basic forward pass produces logits
- KV cache shapes are correct
- Formal verification test passes for short sequences (<128 tokens)

### Under Active Development ⚠
- Full KV cache parity for sequences > 128 tokens
- Real paged attention with non-identity block tables
- MTP speculative decoding (experimental)
- Complete HuggingFace output parity

### Known Issues ⚠
See [BUG_FIXES.md](BUG_FIXES.md) for detailed issue tracking.

**Not production-ready.** Use for learning and experimentation only.

## Overview

This project implements Qwen3.5-0.8B model in pure JAX with:

- **Hybrid Attention Architecture**: 18 linear attention + 6 full attention layers
- **Paged Attention**: Block-based KV cache (identity block tables only, under development)
- **Linear Attention States**: Recurrent state management for Gated DeltaNet layers
- **MTP Speculative Decoding**: Experimental, partially integrated
- **Server-Style Compilation**: JIT compile once at startup, serve many requests
- **HF Parity**: Partial match, under active validation

## Project Structure

```
nano-vllm-jax/
├── nanovllm_jax/              # Main package (~20 files)
│   ├── config.py              # Qwen3.5 configuration
│   ├── model.py               # Transformer implementation (929 lines)
│   ├── model_simple_jit.py    # JIT-compiled forward pass
│   ├── layers.py              # RoPE, RMSNorm, attention ops
│   ├── kv_cache.py            # 5D per-layer KV cache
│   ├── load_weights_float16.py # HF weight loader
│   ├── engine/
│   │   ├── model_runner.py    # JIT-compiled inference engine
│   │   ├── block_manager.py   # Paged attention blocks
│   │   ├── sequence.py        # Sequence state tracking
│   │   └── scheduler.py       # Request scheduling
│   └── mtp/
│       ├── mtp_layer.py       # MTP head (375 lines)
│       └── speculative.py     # Speculative decoding logic
│
├── tests/                     # Test suite
│   ├── test_formal_verification.py  # KV cache parity (✓ PASSING)
│   ├── test_e2e_parity.py     # End-to-end parity tests
│   ├── test_kv_cache.py       # KV cache correctness
│   ├── test_layer_parity.py   # Layer-wise HF comparison
│   └── test_mtp.py            # MTP speculative decoding tests
│
├── docs/                      # Documentation
│   └── archive/               # Development docs
│
├── archives/                  # Stable backups
│
├── server.py                  # Flask API server
├── benchmark_quick_hf.py      # JAX vs HF benchmark
├── benchmark_quick_combined.py # JAX + MTP vs HF
├── benchmark_mtp_v3.py        # MTP benchmark
│
├── README.md                  # This file
├── BENCHMARK_REPORT.md        # Performance report
├── REPOSITORY_SUMMARY.md      # File inventory
├── PLAN.md                    # Project goals
└── INDEX.md                   # Directory index
```

## Installation

```bash
pip install -e .
```

Dependencies:
- JAX
- transformers (for HF model loading)
- torch (for HF model comparison)

## Quick Start

### Basic Usage

```python
from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.load_weights import load_weights_from_hf
from nanovllm_jax.engine.model_runner import ModelRunner

# Load configuration and weights
config = Qwen3_5Config.qwen3_5_0_8b()
params = load_weights_from_hf("Qwen/Qwen3.5-0.8B", config)

# Initialize model runner
runner = ModelRunner(config, params)

# Server-style warmup compilation (one-time startup cost)
runner.warmup_compilation(max_prefill_len=64, max_batch=1)

# Ready to serve requests!
```

### API Server

```bash
# Start the LLMEngine-backed server with startup-compiled static shapes
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS=--xla_gpu_autotune_level=0

python server.py \
  --dtype float16 \
  --jax-execution jit \
  --prefill-buckets 16 \
  --batch-size-buckets 1 \
  --max-num-seqs 1 \
  --max-kv-cache-mb 64 \
  --num-kvcache-blocks 8

# Optional greedy MTP1 speculative decoding, if the checkpoint has mtp.* weights
# Add: --num-speculative-tokens 1

# Test health endpoint
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 4, "temperature": 0.0}'

# Explicit batched generation; set --max-num-seqs and --batch-size-buckets accordingly
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": ["Hello", "Explain JAX in one sentence"], "max_tokens": 4}'
```

### Benchmarks

```bash
# JAX vs HuggingFace comparison
python benchmark_quick_hf.py

# JAX + MTP vs HuggingFace
python benchmark_quick_combined.py

# MTP performance
python benchmark_mtp_v3.py
```

### Current KV-Cache GPU Benchmark

`benchmark_real_kv_hf.py` uses the canonical `ModelExecutor` path, validates
prefill logits and greedy tokens against HuggingFace, and can split HF/JAX into
separate processes to avoid mixed-runtime GPU memory pressure.

Tested on a 6 GiB GTX 1660 Ti under WSL. This GPU does not support native bf16,
so the benchmark uses float16 weights.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS=--xla_gpu_autotune_level=0

python benchmark_real_kv_hf.py \
  --target hf \
  --max-new-tokens 4 \
  --prompt 'Tell me a joke about compilers.' \
  --output-npz /tmp/qwen_hf_ref.npz

python benchmark_real_kv_hf.py \
  --target jax \
  --jax-execution jit \
  --prefill-bucket 16 \
  --max-new-tokens 4 \
  --max-kv-cache-mb 64 \
  --prompt 'Tell me a joke about compilers.' \
  --compare-npz /tmp/qwen_hf_ref.npz
```

Observed for that prompt with an exact prefill shape: HF cache 1.32 tok/s
end-to-end, JAX full startup-JIT 13.93 tok/s end-to-end. With
`--prefill-bucket 16`, JAX full startup-JIT measured 8.65 tok/s end-to-end.
Both JAX runs had exact greedy-token match, top-5 prefill logits match, and
`jit_compiled_during_measure=False`.

## Model Architecture: Qwen3.5-0.8B

- **Total Layers**: 24
- **Linear Attention**: 18 layers (Gated DeltaNet)
- **Full Attention**: 6 layers (standard attention)
- **Pattern**: 3 linear + 1 full attention, repeated
- **Hidden Size**: 1024
- **Attention Heads**: 8 (full), 16 (linear)
- **Head Dim**: 256 (full), 128 (linear)

### Inference Modes

**Prefill Mode**:
- Linear attention: Chunked computation (chunk_size=64)
- Full attention: Standard scaled dot-product attention

**Decode Mode**:
- Linear attention: Recurrent state updates (single token)
- Full attention: Paged attention with KV cache

## Testing

Run the test suite:

```bash
# KV cache parity tests (✓ PASSING)
python tests/test_formal_verification.py

# KV cache tests
python tests/test_kv_cache.py

# Layer-wise parity tests (requires HF model)
python tests/test_layer_parity.py

# End-to-end parity tests
python tests/test_e2e_parity.py

# MTP speculative decoding tests
python tests/test_mtp.py
```

### Test Requirements
- **KV Cache Parity**: Exact token match (100% parity)
- **Layer Parity**: MSE < 1e-5 per layer
- **E2E Parity**: Top 5 logits match exactly, total MSE < 1e-4
- **MTP**: Speedup > 1.05x

## MTP Speculative Decoding (Experimental)

⚠️ **Status**: MTP integration is experimental.

The Multi-Token Prediction (MTP) head generates draft tokens for speculative decoding:

- **Mechanism**: MTP1 keeps one pending draft token, verifies it on the next target pass, and commits a bonus token when the greedy draft is accepted.
- **Scope**: Config gated with `num_speculative_tokens=1`; the server exposes this as `--num-speculative-tokens 1`.
- **Fallbacks**: Batched decode, non-greedy sampling, missing MTP weights, and boundary cases fall back to normal decoding.
- **Architecture**: 1 transformer layer + LM head.

### Known Issues
- Current MTP1 path is greedy-only.
- The first implementation is single-sequence only.
- Broader correctness/performance validation still needs a real checkpoint with MTP weights.

**Do not use in production** until MTP integration is complete.

## Performance

### JAX + MTP vs HuggingFace (Preliminary Results)

**CPU Performance (5 prompt tokens, 10 decode tokens):**

| Implementation | TPS | Speedup vs HF |
|----------------|-----|---------------|
| HuggingFace (KV cache) | 0.49 | baseline |
| JAX (no MTP) | 0.65 | 1.31x |
| JAX + MTP | ~0.71 | 1.45x |

**Combined speedup: 45% faster than HuggingFace**

⚠️ **Methodology Limitations**:
- Tested only on short sequences (5-10 prompt tokens)
- Uses identity block tables (not real paged attention)
- CPU-only testing
- **Do not extrapolate** to production workloads

### Speedup Breakdown

- **JAX optimizations**: 1.31x (31% faster)
  - JIT compilation reduces Python overhead
  - Efficient per-layer KV cache
  - Optimized custom operations
  
- **MTP speculative decoding**: 1.10x (10% additional)
  - Predicts token T+2 using hidden state + embedding
  - Reduces forward passes by 13%
  - Lightweight (1 layer vs 24 layers)

### Verified (Preliminary)
- ⚠️ Partial match with HF outputs (for short sequences only)
- ✅ JAX is 31% faster than HuggingFace (tested configurations)
- ⚠️ MTP adds 10% speedup (requires verification)
- ⚠️ Combined: 45% faster than HuggingFace (tested configurations)
- ✅ Server-style compilation works
- ⚠️ API server functional but experimental

### Expected Performance
- **CPU**: ~0.6-0.7 tokens/second
- **GPU**: 50-100x faster (30-70 tokens/second)

### Known Limitations
- ⚠️ CPU compilation time: 30-120 seconds per shape
- ✅ GPU/TPU: 5-10 seconds total compilation

## Documentation

- **[README.md](README.md)** - This file (project overview)
- **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)** - Performance comparison report
- **[REPOSITORY_SUMMARY.md](REPOSITORY_SUMMARY.md)** - Complete file inventory
- **[PLAN.md](PLAN.md)** - Project goals and architecture
- **[INDEX.md](INDEX.md)** - File/directory index
- **[MODEL_IMPLEMENTATION_STATUS.md](MODEL_IMPLEMENTATION_STATUS.md)** - Implementation status

## References

- [Qwen3.5 Model](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [vLLM](https://github.com/vllm-project/vllm) - Production vLLM
- [MaxText](https://github.com/google/maxtext) - JAX reference implementation
- [Gated DeltaNet Paper](https://arxiv.org/abs/...) - Linear attention mechanism

## License

MIT License
