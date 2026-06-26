# Agent Instructions

This branch is the cleaned serving mainline. Keep it compact, executable, and
pedagogical.

## Serving Path

- `server.yaml` is the only committed serving config.
- `nanovllm_jax/config.py` owns model architecture plus workload/capacity
  settings.
- `nanovllm_jax/fastpath.py` owns implementation policy. Do not add YAML or
  environment switches for alternative kernels on this branch.
- The supported runtime is the promoted CUDA/JAX Qwen3.5 text-serving path.
- Speculative decoding, Metal/TPU alternates, benchmark harnesses, result
  dumps, and historical optimization notes do not belong on cleaned main.

## Validation

- Use `rg` for search.
- Run `python -m compileall -q server.py nanovllm_jax tests` after structural
  edits.
- Wrap long test runs with the RAM guard so this shared server does not run out
  of memory:

```bash
PYTHONPATH=$PWD python tests/ram_guard.py -- pytest -q
```

- For CPU-safe control-plane changes, run:

```bash
PYTHONPATH=$PWD python tests/ram_guard.py -- pytest -q tests/test_fastpath_config.py tests/test_service.py tests/test_server_config.py tests/test_public_imports.py
```

- For GPU work, first verify CUDA visibility:

```bash
nvidia-smi
JAX_PLATFORMS=cuda PYTHONPATH=$PWD python - <<'PY'
import jax
print(jax.devices("gpu"))
PY
```

- Use `JAX_PLATFORMS=cuda` for CUDA correctness tests. Do not silently fall
  back to CPU for GPU validation.

## Artifacts

- Keep diagnostics, profiles, and benchmark outputs outside the repo, preferably
  under `/mountpoint/.exp/diagnostics`.
- Do not commit generated `results/`, profile dumps, server run logs, caches, or
  model checkpoints.
- If a benchmark or diagnostic script is needed, keep it in a sibling worktree
  or external scratch area rather than restoring it to cleaned main.

## Editing

- Prefer existing module boundaries and naming.
- Keep public configuration limited to workload and capacity.
- Keep comments focused on ownership, shapes, and invariants.
- Do not reintroduce `nanovllm_jax.engine` compatibility imports.
