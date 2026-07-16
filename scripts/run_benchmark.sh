#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-both}
case "$target" in
  jax|jax-base|jax-mtp|vllm|vllm-base|vllm-mtp|both) ;;
  *) echo "usage: $0 [jax|jax-base|jax-mtp|vllm|vllm-base|vllm-mtp|both]" >&2; exit 2 ;;
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

clear_routes() {
  local route
  for route in "$@"; do
    rm -f "$artifact_root/results/$route.json" \
      "$artifact_root/results/$route.ram.json"
  done
}

rm -f "$artifact_root/results/comparison.json"
case "$target" in
  jax-base|jax-mtp|vllm-base|vllm-mtp) clear_routes "$target" ;;
  jax) clear_routes jax-base jax-mtp ;;
  vllm) clear_routes vllm-base vllm-mtp ;;
  both) clear_routes jax-base jax-mtp vllm-base vllm-mtp ;;
esac

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

gpu_preflight() {
  nvidia-smi -i "$gpu" \
    --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
}

jax_preflight() {
  JAX_PLATFORMS=cuda "$jax_env/bin/python" -c \
    'import jax, jax.numpy as jnp; assert len(jax.devices("gpu")) == 1; jnp.ones(1).block_until_ready()'
}

guard_with() {
  local python=$1
  local summary=$2
  shift 2
  "$python" "$root/tests/ram_guard.py" \
    --rss-gib 10 --min-available-gib 2 \
    --max-system-ram-percent "$ram_percent" --summary-json "$summary" -- "$@"
}

run_jax() {
  local route=$1
  local reference=()
  if [[ "$route" == mtp ]]; then
    reference=(--reference "$artifact_root/results/jax-base.json")
  fi
  JAX_PLATFORMS=cuda PYTHONPATH="$root" guard_with \
    "$jax_env/bin/python" \
    "$artifact_root/results/jax-$route.ram.json" \
    "$jax_env/bin/python" -m benchmarks.run_benchmark jax \
    --route "$route" "${reference[@]}" \
    --output "$artifact_root/results/jax-$route.json"
}

run_vllm() {
  local route=$1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$root" guard_with \
    "$vllm_env/bin/python" \
    "$artifact_root/results/vllm-$route.ram.json" \
    "$vllm_env/bin/python" -m benchmarks.run_benchmark vllm \
    --route "$route" \
    --reference "$artifact_root/results/jax-base.json" \
    --output "$artifact_root/results/vllm-$route.json"
}

gpu_preflight
case "$target" in
  jax-base|jax-mtp)
    [[ "$target" != jax-mtp || -f "$artifact_root/results/jax-base.json" ]] || {
      echo "run jax-base first so JAX MTP has a parity reference" >&2
      exit 2
    }
    setup_jax
    jax_preflight
    run_jax "${target#jax-}"
    ;;
  jax)
    setup_jax
    jax_preflight
    run_jax base
    run_jax mtp
    ;;
  vllm-base|vllm-mtp)
    [[ -f "$artifact_root/results/jax-base.json" ]] || {
      echo "run jax-base first so vLLM has a parity reference" >&2
      exit 2
    }
    setup_vllm
    run_vllm "${target#vllm-}"
    ;;
  vllm)
    [[ -f "$artifact_root/results/jax-base.json" ]] || {
      echo "run the JAX side first so vLLM has a base-token reference" >&2
      exit 2
    }
    setup_vllm
    run_vllm base
    run_vllm mtp
    ;;
  both)
    setup_jax
    jax_preflight
    run_jax base
    run_jax mtp
    setup_vllm
    run_vllm base
    run_vllm mtp
    "$jax_env/bin/python" -m benchmarks.compare_results \
      --jax-base "$artifact_root/results/jax-base.json" \
      --jax-mtp "$artifact_root/results/jax-mtp.json" \
      --vllm-base "$artifact_root/results/vllm-base.json" \
      --vllm-mtp "$artifact_root/results/vllm-mtp.json" \
      --output "$artifact_root/results/comparison.json"
    ;;
esac
