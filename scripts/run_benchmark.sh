#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-both}
case "$target" in
  jax|vllm|both) ;;
  *) echo "usage: $0 [jax|vllm|both]" >&2; exit 2 ;;
esac

scratch_root=${NANO_VLLM_JAX_BENCHMARK_ROOT:-/mountpoint/.exp}
artifact_root=$scratch_root/artifacts/nano-vllm-jax/benchmark
jax_env=$scratch_root/envs/nano-vllm-jax-benchmark-jax
vllm_env=$scratch_root/envs/nano-vllm-jax-benchmark-vllm
ram_percent=${NANO_VLLM_JAX_MAX_SYSTEM_RAM_PERCENT:-80}
gpu=${NANO_VLLM_JAX_BENCHMARK_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
[[ "$gpu" != *,* ]] || {
  echo "select exactly one GPU with NANO_VLLM_JAX_BENCHMARK_GPU" >&2
  exit 2
}

export NANO_VLLM_JAX_BENCHMARK_GPU=$gpu
export CUDA_VISIBLE_DEVICES=$gpu
export HF_HOME=$scratch_root/.cache/huggingface
export UV_CACHE_DIR=$scratch_root/.cache/uv
export VLLM_CACHE_ROOT=$scratch_root/.cache/vllm
export VLLM_CONFIG_ROOT=$scratch_root/.config/vllm
export FLASHINFER_WORKSPACE_BASE=$scratch_root
export VLLM_USE_FLASHINFER_SAMPLER=0
export UV_NO_PROGRESS=1
mkdir -p "$artifact_root/results" "$HF_HOME"

setup_jax() {
  UV_PROJECT_ENVIRONMENT="$jax_env" uv sync --python 3.11 \
    --project "$root" --frozen --no-dev \
    --extra cuda13 --extra flashinfer-ffi --extra gdn-fla-triton
}

setup_vllm() {
  if [[ ! -x "$vllm_env/bin/python" ]]; then
    uv venv --python 3.11 "$vllm_env"
  fi
  "$vllm_env/bin/python" -c \
    'import sys; assert sys.version_info[:2] == (3, 11)'
  uv pip install --python "$vllm_env/bin/python" \
    --requirements "$root/benchmarks/vllm-requirements.txt"
  # vLLM treats TorchCodec as optional, but its import raises when FFmpeg video
  # libraries are absent. This benchmark is text-only.
  uv pip uninstall --python "$vllm_env/bin/python" torchcodec >/dev/null
}

preflight() {
  nvidia-smi -i "$gpu" \
    --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
  JAX_PLATFORMS=cuda "$jax_env/bin/python" -c \
    'import jax, jax.numpy as jnp; assert len(jax.devices("gpu")) == 1; jnp.ones(1).block_until_ready()'
}

guard() {
  local summary=$1
  shift
  "$jax_env/bin/python" "$root/tests/ram_guard.py" \
    --rss-gib 10 --min-available-gib 2 \
    --max-system-ram-percent "$ram_percent" --summary-json "$summary" -- "$@"
}

run_jax() {
  JAX_PLATFORMS=cuda PYTHONPATH="$root" guard \
    "$artifact_root/results/jax.ram.json" \
    "$jax_env/bin/python" -m benchmarks.run_benchmark jax \
    --output "$artifact_root/results/jax.json"
}

run_vllm() {
  [[ -f "$artifact_root/results/jax.json" ]] || {
    echo "run the JAX side first so vLLM has an exact-token reference" >&2
    exit 2
  }
  PYTHONPATH="$root" guard \
    "$artifact_root/results/vllm.ram.json" \
    "$vllm_env/bin/python" -m benchmarks.run_benchmark vllm \
    --reference "$artifact_root/results/jax.json" \
    --output "$artifact_root/results/vllm.json"
}

setup_jax
preflight
if [[ "$target" == jax || "$target" == both ]]; then
  run_jax
fi
if [[ "$target" == vllm || "$target" == both ]]; then
  setup_vllm
  run_vllm
fi
if [[ "$target" == both ]]; then
  "$jax_env/bin/python" -m benchmarks.compare_results \
    --jax "$artifact_root/results/jax.json" \
    --vllm "$artifact_root/results/vllm.json" \
    --output "$artifact_root/results/comparison.json"
fi
