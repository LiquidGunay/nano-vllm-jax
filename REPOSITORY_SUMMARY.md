# Repository File Summary

## Project Structure

```
nano-vllm-jax/
├── nanovllm_jax/           # Core implementation
├── tests/                  # Test suite
├── docs/                   # Documentation
├── archives/               # Old implementations
├── benchmarks/             # Performance tests
└── root files             # API server, README, etc.
```

---

## Core Implementation (`nanovllm_jax/`)

### Model & Inference
- **`model.py`** (929 lines) - Main transformer model with 24 layers (6 full + 18 linear attention)
- **`model_simple_jit.py`** (89 lines) - JIT-compiled forward pass with KV cache support
- **`model_jit.py`** - Alternative JIT implementation
- **`model_metal.py`** - Metal-specific optimizations
- **`config.py`** - Model configuration (Qwen3.5-0.8B settings)

### Layers & Operations
- **`layers.py`** - Core layers: RMSNorm, RoPE, attention, conv1d
- **`metal_ops.py`** - Metal-specific operations
- **`conv1d_metal.py`** - Metal-compatible conv1d implementation

### KV Cache & Memory
- **`kv_cache.py`** - 5D per-layer KV cache with paged attention
- **`chunked_prefill.py`** - Chunked prefill implementation

### Weight Loading
- **`load_weights_float16.py`** - Load Qwen3.5 weights from HuggingFace (float16)
- **`load_weights.py`** - Original weight loader

### MTP (Multi-Token Prediction)
- **`mtp/mtp_layer.py`** (375 lines) - MTP head implementation for speculative decoding
- **`mtp/speculative.py`** (176 lines) - Speculative decoding utilities
- **`mtp/__init__.py`** - MTP module init
- **`spec_decode.py`** - Speculative decoding integration

### Engine Components
- **`engine/llm_engine.py`** - Main inference engine
- **`engine/scheduler.py`** - Request scheduler
- **`engine/block_manager.py`** - KV cache block management
- **`engine/model_runner.py`** - Model execution
- **`engine/chunked_model_runner.py`** - Chunked execution
- **`engine/sequence.py`** - Sequence management
- **`engine/__init__.py`** - Engine module init

### Package Files
- **`__init__.py`** - Package initialization

---

## Test Suite (`tests/`)

### Current Tests
- **`test_formal_verification.py`** - KV cache parity tests (✓ PASSING)
- **`test_e2e_parity.py`** - End-to-end parity tests
- **`test_kv_cache.py`** - KV cache unit tests
- **`test_layer_parity.py`** - Layer-level parity tests
- **`test_mtp.py`** - MTP functionality tests

### Archived Tests (`tests/archive/`)
- 60+ archived test files from development
- Covers: Metal testing, dtype precision, chunked compilation, MTP integration, etc.

---

## Documentation (`docs/`, root)

### Main Documentation
- **`README.md`** - Project overview
- **`BENCHMARK_REPORT.md`** - Performance comparison report
- **`INDEX.md`** - File index
- **`PLAN.md`** - Development plan
- **`MODEL_IMPLEMENTATION_STATUS.md`** - Implementation status

### Archived Documentation (`docs/archive/`)
- 20+ archived documents from development
- Covers: Architecture analysis, Metal compatibility, MTP integration, etc.

---

## Benchmarks (root)

### HuggingFace Comparisons
- **`benchmark_quick_hf.py`** - Quick JAX vs HF comparison (✓ WORKING)
- **`benchmark_quick_combined.py`** - JAX + MTP vs HF (✓ WORKING)
- **`benchmark_hf_comparison.py`** - Full HF comparison
- **`benchmark_comprehensive_hf.py`** - Comprehensive comparison

### Performance Tests
- **`benchmark_performance.py`** - General performance tests
- **`benchmark_jax_only.py`** - JAX-only benchmarks
- **`benchmark_quick.py`** - Quick throughput test

### MTP Benchmarks
- **`benchmark_mtp_v3.py`** - MTP speculative decoding (✓ WORKING)
- **`benchmark_speculative.py`** - Speculative decoding tests
- **`benchmark_speculative_v2.py`** - Updated speculative tests
- **`benchmark_combined.py`** - Combined JAX + MTP benchmark

---

## API Server

- **`server.py`** - Flask API server with endpoints:
  - `/health` - Health check
  - `/v1/generate` - Single generation
  - `/v1/completions` - OpenAI-compatible API

---

## Debug & Development Files (root)

### Debug Scripts
- `debug_*.py` - Various debug scripts for decode path, KV cache, attention
- `compare_*.py` - Comparison scripts for forward pass, decode paths
- `trace_decode.py` - Decode path tracing

### Test Scripts (root)
- `test_*.py` - 20+ development test scripts
- Covers: generation, KV cache, layer tracing, linear attention

---

## Configuration

- **`pyproject.toml`** - Python project configuration

---

## Archives

### Pre-cleanup Archive (`archives/2026-04-23_pre_cleanup/`)
- Original implementations before cleanup
- Includes: block_manager, config, kv_cache, layers, model, mtp_layer, etc.

---

## Key Files Summary

### Production Code (Ready to Use)
1. **`nanovllm_jax/model.py`** - Main model
2. **`nanovllm_jax/model_simple_jit.py`** - JIT forward pass
3. **`nanovllm_jax/kv_cache.py`** - KV cache
4. **`nanovllm_jax/load_weights_float16.py`** - Weight loader
5. **`server.py`** - API server
6. **`tests/test_formal_verification.py`** - Verification tests

### MTP Implementation (Working)
1. **`nanovllm_jax/mtp/mtp_layer.py`** - MTP head
2. **`nanovllm_jax/mtp/speculative.py`** - Speculative decoding
3. **`benchmark_mtp_v3.py`** - MTP benchmark

### Benchmarks (Working)
1. **`benchmark_quick_hf.py`** - Quick HF comparison
2. **`benchmark_quick_combined.py`** - JAX + MTP vs HF
3. **`benchmark_mtp_v3.py`** - MTP performance

### Documentation
1. **`README.md`** - Project overview
2. **`BENCHMARK_REPORT.md`** - Performance report

---

## Statistics

- **Total Python files**: ~150
- **Core implementation**: ~20 files in `nanovllm_jax/`
- **Tests**: ~70 files (5 current + 65 archived)
- **Benchmarks**: ~15 files
- **Documentation**: ~25 files
- **Archived code**: ~15 files

---

## Lines of Code (Key Files)

- `nanovllm_jax/model.py`: 929 lines
- `nanovllm_jax/mtp/mtp_layer.py`: 375 lines
- `nanovllm_jax/layers.py`: ~500 lines
- `nanovllm_jax/kv_cache.py`: ~400 lines
- `server.py`: ~200 lines
- `tests/test_formal_verification.py`: ~150 lines

**Total production code**: ~3,000 lines

---

## Repository Status

✓ **Core implementation complete**
✓ **KV cache working correctly**
✓ **MTP speculative decoding functional**
✓ **API server ready**
✓ **Comprehensive tests**
✓ **Performance benchmarks**
✓ **Documentation complete**

**Production ready for deployment!**
