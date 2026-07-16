#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

python -m compileall -q nanovllm_jax server.py tests
python -m ruff format --check .
python -m ruff check .
if matches=$(grep -En "(getattr|hasattr)\\(self,[[:space:]]*['\"]_" nanovllm_jax/runner.py); then
  echo "runner-owned private state must be initialized and accessed directly" >&2
  echo "$matches" >&2
  exit 1
fi
python tests/ram_guard.py \
  --max-system-ram-percent 70 \
  --rss-gib 4 \
  --min-available-gib 2 \
  -- python -m pytest -q tests/test_service.py
