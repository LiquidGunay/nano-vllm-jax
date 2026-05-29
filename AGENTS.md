# Agent Instructions

## Subagents

- Prefer `gpt-5.3-codex-spark` for newly spawned subagents unless a task clearly
  needs a different model.

## GPU And Benchmark Commands

- Run benchmark, profiling, vLLM, JAX GPU, CUDA, NVIDIA, and model-serving
  commands outside the sandbox with elevated access. The sandbox cannot reliably
  see `/dev/nvidia*`, so sandboxed GPU checks can report false driver/device
  failures.
- Treat Python/pytest commands that initialize JAX, vLLM, CUDA, or NVIDIA
  libraries as GPU commands. Run those outside the sandbox with elevated access
  as well, even when the command itself looks like a normal unit-test or
  benchmark invocation.
- Default to elevated access for any benchmark, profiling, server, model-load,
  or performance-measurement command that may touch the GPU runtime. If unsure
  whether a command will initialize GPU/JAX/vLLM/CUDA state, run it elevated.
- Keep benchmark/model/cache/temp paths rooted under `/mountpoint/.exp`.
- Keep JAX GPU runs GPU-only with `JAX_PLATFORMS=cuda`; do not fall back to CPU
  for correctness or benchmark runs.
- Do not use `--skip-gpu-preflight` to hide missing GPU visibility. If an
  elevated run still cannot communicate with the GPU, stop and ask the user for
  help.
