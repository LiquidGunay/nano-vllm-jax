# Nano-vLLM-JAX Documentation Index

**Last Updated**: 2026-04-24  
**Status**: Cleaned and reorganized

---

## Essential Documentation

| File | Purpose |
|------|---------|
| [PLAN.md](PLAN.md) | Project goals, architecture, execution plan |
| [README.md](README.md) | Project overview and quick start |
| [MODEL_IMPLEMENTATION_STATUS.md](MODEL_IMPLEMENTATION_STATUS.md) | Implementation progress (append-only) |

---

## Package Structure (`nanovllm_jax/`)

### Core Modules
- `config.py` - Qwen3.5 model configuration
- `model.py` - Transformer implementation (linear + full attention)
- `model_simple_jit.py` - JIT-compiled forward pass
- `layers.py` - RoPE, RMSNorm, attention ops
- `kv_cache.py` - Paged KV cache + linear attention states
- `load_weights.py` / `load_weights_float16.py` - HF weight loaders

### Engine (`engine/`)
- `model_runner.py` - JIT-compiled inference engine
- `block_manager.py` - Paged attention block management
- `sequence.py` - Sequence state tracking
- `scheduler.py` - Request scheduling
- `llm_engine.py` - High-level engine API

### MTP (`mtp/`)
- `mtp_layer.py` - MTP proposer implementation
- `speculative.py` - Speculative decoding logic

---

## Test Suite (`tests/`)

| File | Purpose | Criteria |
|------|---------|----------|
| `test_layer_parity.py` | Layer-wise HF comparison | MSE < 1e-5 |
| `test_e2e_parity.py` | End-to-end logits matching | MSE < 1e-4 |
| `test_kv_cache.py` | KV cache correctness | MSE < 1e-6 |
| `test_mtp.py` | MTP speculative decoding | Acceptance > 20% |

### Run Tests
```bash
JAX_PLATFORMS=cpu python -m pytest tests/ -v
```

---

## Project Structure

```
nano-vllm-jax/
├── nanovllm_jax/          # Main package
│   ├── config.py
│   ├── model.py
│   ├── model_simple_jit.py
│   ├── layers.py
│   ├── kv_cache.py
│   ├── load_weights*.py
│   ├── engine/
│   └── mtp/
│
├── tests/                  # Test suite (4 files)
│   ├── test_layer_parity.py
│   ├── test_e2e_parity.py
│   ├── test_kv_cache.py
│   ├── test_mtp.py
│   └── archive/            # Historical test files
│
├── docs/
│   └── archive/            # Session summaries, work logs
│
├── archives/               # Stable backups
├── references/             # External docs
│
├── PLAN.md
├── INDEX.md
├── README.md
└── MODEL_IMPLEMENTATION_STATUS.md
```

---

## Maintenance Notes

- All changes to `MODEL_IMPLEMENTATION_STATUS.md` should be **appends only**
- Run all 4 tests before committing: `JAX_PLATFORMS=cpu python -m pytest tests/ -v`
- Update `INDEX.md` when adding/removing files
- Archive old files to `docs/archive/` or `tests/archive/`
