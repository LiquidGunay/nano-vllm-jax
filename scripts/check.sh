#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

uv sync --frozen --python 3.11
python=$root/.venv/bin/python

"$python" -m compileall -q nanovllm_jax server.py tests
"$python" -m ruff format --check .
"$python" -m ruff check .
if matches=$(grep -En "(getattr|hasattr)\\(self,[[:space:]]*['\"]_" nanovllm_jax/runner.py); then
  echo "runner-owned private state must be initialized and accessed directly" >&2
  echo "$matches" >&2
  exit 1
fi
"$python" tests/ram_guard.py \
  --max-system-ram-percent 70 \
  --rss-gib 4 \
  --min-available-gib 2 \
  -- "$python" -m pytest -q \
  tests/test_benchmark_artifact.py \
  tests/test_engine_initialization.py \
  tests/test_fastpath_config.py \
  tests/test_public_imports.py \
  tests/test_routes.py \
  tests/test_scheduler_capacity.py \
  tests/test_server_config.py \
  tests/test_service.py \
  tests/test_step_results.py
