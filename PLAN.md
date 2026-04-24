# Nano-vLLM-JAX Project Plan

**Last Updated**: 2026-04-23  
**Status**: Cleanup Complete, Testing Phase  

---

## Project Goals

Build a pedagogical nano-vllm style implementation of Qwen3.5 in JAX with:
- Complete JIT compilation
- Ragged paged attention (prefill + decode support)
- MTP1 as proposer for speculative decoding
- HF parity validation
- Production-ready API server mode

---

## Success Criteria

### Must Have
- [x] Clean package structure (`nanovllm_jax/`)
- [x] Exactly 4 test files
- [ ] Layer-wise parity: MSE < 1e-5 per layer
- [ ] End-to-end parity: top 5 logits exact match, total MSE < 1e-4
- [ ] KV cache tests: paged attention works, linear attention states correct
- [ ] MTP tests: acceptance > 20%, speedup > 1.2x
- [x] All imports updated to `nanovllm_jax`
- [x] Documentation: PLAN.md, INDEX.md, two status MD files

### Stretch Goals
- [ ] API server mode (compile once, serve many)
- [ ] Benchmark script showing speedup vs HF
- [ ] Multi-GPU support via JAX sharding
- [ ] Continuous batching

---

## Architecture

### Model: Qwen3.5-0.8B
- **Layers**: 24 total
  - 18 linear attention (Gated DeltaNet)
  - 6 full attention (standard)
- **Attention**: Hybrid architecture with both chunked and recurrent modes
- **KV Cache**: Paged attention for full attention, recurrent states for linear attention

### Components

```
nanovllm_jax/
├── config.py          # Model configuration
├── model.py           # Transformer implementation (847 lines)
├── layers.py          # RoPE, RMSNorm, attention ops
├── kv_cache.py        # Paged KV + linear attention states
├── load_weights.py    # HF weight loader
├── engine/
│   ├── model_runner.py  # JIT-compiled inference
│   ├── block_manager.py # Paged attention blocks
│   ├── sequence.py      # Sequence state tracking
│   └── scheduler.py     # Request scheduling
└── mtp/
    ├── mtp_layer.py     # MTP proposer
    └── speculative.py   # Speculative decoding
```

---

## Testing Strategy

### Test Files
1. `tests/test_layer_parity.py` - Layer-by-layer HF comparison
2. `tests/test_e2e_parity.py` - End-to-end logits/token matching
3. `tests/test_kv_cache.py` - KV cache correctness
4. `tests/test_mtp.py` - MTP speculative decoding

### Test Requirements
- **All tests use real HF weights** from Qwen/Qwen3.5-0.8B
- **No random weights allowed**
- **Precision**: bfloat16 weights + fp32 activations

---

## Implementation Phases

### Phase 1: Cleanup ✅ COMPLETE
- [x] Archive working implementation
- [x] Delete 43 test files
- [x] Delete 15+ debug files
- [x] Delete archived model versions
- [x] Delete docs_archive/
- [x] Rename qwen35_jax → nanovllm_jax
- [x] Update all imports

### Phase 2: Reorganization ✅ COMPLETE
- [x] Create tests/ directory
- [x] Create 4 test files
- [x] Create documentation

### Phase 3: Testing 🔄 IN PROGRESS
- [ ] Run test_layer_parity.py
- [ ] Run test_e2e_parity.py
- [ ] Run test_kv_cache.py
- [ ] Run test_mtp.py
- [ ] Document results

### Phase 4: Optimization
- [ ] Fix any test failures
- [ ] Optimize JIT compilation
- [ ] Add API server mode
- [ ] Benchmark performance

---

## Known Issues

### CPU Compilation Time
The 24-layer model requires significant JIT compilation time on CPU.
- Prefill: 30-50 seconds per shape
- Decode: 60+ seconds per shape

**Mitigation**:
1. Use GPU/TPU for development
2. Create smaller test configs (4 layers)
3. Use JAX persistent cache

---

## References

Key resources for implementation:
- vLLM TPU inference: https://github.com/vllm-project/vllm
- MaxText: https://github.com/google/maxtext
- Gated DeltaNet paper: https://arxiv.org/abs/...
- Qwen3.5 model: https://huggingface.co/Qwen/Qwen3.5-0.8B

---

## Notes for Contributors

1. All changes to MODEL_IMPLEMENTATION_STATUS.md and MODEL_TESTING_STATUS.md should be **appends only**
2. Run all 4 tests before committing changes
3. Document any bugs or issues found
4. Reference this PLAN.md when delegating to sub-agents

---

## File Organization

```
nano-vllm-jax/
├── nanovllm_jax/        # Main package
├── tests/               # 4 test files
├── archives/            # Backups
├── references/          # External docs
├── PLAN.md              # This file
├── INDEX.md             # File index
├── MODEL_IMPLEMENTATION_STATUS.md
├── MODEL_TESTING_STATUS.md
└── README.md
```
