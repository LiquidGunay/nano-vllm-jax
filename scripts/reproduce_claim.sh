#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-both}
case "$target" in
  jax|vllm|both) ;;
  *) echo "usage: $0 [jax|vllm|both]" >&2; exit 2 ;;
esac

artifact_root=${NANO_VLLM_JAX_ARTIFACT_ROOT:-/mountpoint/.exp/artifacts/nano-vllm-jax/claim}
jax_env=${NANO_VLLM_JAX_CLAIM_JAX_ENV:-/mountpoint/.exp/envs/nano-vllm-jax-claim-jax}
vllm_env=${NANO_VLLM_JAX_CLAIM_VLLM_ENV:-/mountpoint/.exp/envs/nano-vllm-jax-claim-vllm}
ram_percent=${NANO_VLLM_JAX_MAX_SYSTEM_RAM_PERCENT:-80}
mkdir -p "$artifact_root/results" /mountpoint/.exp/.cache/huggingface

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-/mountpoint/.exp/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mountpoint/.exp/.cache/uv}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/mountpoint/.exp/.cache/vllm}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/mountpoint/.exp/.config/vllm}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/mountpoint/.exp}
export VLLM_USE_FLASHINFER_SAMPLER=0
export UV_NO_PROGRESS=1

setup_jax() {
  UV_PROJECT_ENVIRONMENT="$jax_env" uv sync --project "$root" --frozen --no-dev \
    --extra cuda13 --extra flashinfer-ffi --extra gdn-fla-triton
}

setup_vllm() {
  if [[ ! -x "$vllm_env/bin/python" ]]; then
    uv venv --python 3.11 "$vllm_env"
  fi
  uv pip install --python "$vllm_env/bin/python" \
    --requirements "$root/benchmarks/vllm-requirements.txt"
  # vLLM treats TorchCodec as optional, but its import raises when FFmpeg video
  # libraries are absent. This benchmark is text-only.
  uv pip uninstall --python "$vllm_env/bin/python" torchcodec >/dev/null
}

preflight() {
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
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
    "$jax_env/bin/python" -m benchmarks.run_claim jax \
    --output "$artifact_root/results/jax.json"
}

run_vllm() {
  [[ -f "$artifact_root/results/jax.json" ]] || {
    echo "run the JAX side first so vLLM has an exact-token reference" >&2
    exit 2
  }
  PYTHONPATH="$root" guard \
    "$artifact_root/results/vllm.ram.json" \
    "$vllm_env/bin/python" -m benchmarks.run_claim vllm \
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
