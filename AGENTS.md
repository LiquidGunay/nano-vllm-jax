# Agent Instructions

## Subagents

- Do not use `realmai_worker` for new delegated work.
- Prefer `gpt-5.3-codex-spark` for newly spawned subagents unless a task clearly
  needs a different model.

## GPU And Benchmark Commands

- Run benchmark, profiling, vLLM, JAX GPU, CUDA, NVIDIA, and model-serving
  commands with GPU visibility verified up front. In sessions that support
  elevated access, use it for GPU work; in unrestricted sessions where approvals
  are disabled, do not request elevation and instead verify with `nvidia-smi`
  plus a CUDA-only JAX smoke check.
- Treat Python/pytest commands that initialize JAX, vLLM, CUDA, or NVIDIA
  libraries as GPU commands. Apply the same visibility/elevation rule even when
  the command itself looks like a normal unit-test or benchmark invocation.
- Default to elevated access for any benchmark, profiling, server, model-load,
  or performance-measurement command that may touch the GPU runtime. If unsure
  whether a command will initialize GPU/JAX/vLLM/CUDA state, run it elevated.
- Keep benchmark/model/cache/temp paths rooted under `/mountpoint/.exp`.
- Keep JAX GPU runs GPU-only with `JAX_PLATFORMS=cuda`; do not fall back to CPU
  for correctness or benchmark runs.
- Do not use `--skip-gpu-preflight` to hide missing GPU visibility. If an
  elevated run still cannot communicate with the GPU, stop and ask the user for
  help.

## Random Decode Benchmark Safety

- Treat new JIT boundaries on the random sidecar as crash-risk until proven
  otherwise. Run a scaled JAX-only diagnostic first with small request and token
  ranges, a finite timeout, CUDA-only execution, and constrained KV/cache
  capacity.
- Keep the random sidecar resource guard enabled for GPU benchmark subprocesses:
  default `--max-system-ram-percent 70`, bounded `--worker-cpu-cores`, and
  positive `--worker-nice`. If a run dies with `killed_resource_limit`, scale
  the random ranges or cache capacity down before retrying.
- Promote diagnostics in stages: small random run, medium random run, then the
  full seed-1234 random decode graph. Do not run the full graph after a
  boundary-changing edit unless the smaller run shows acceptable correctness,
  no measured-phase JIT cache growth, and reasonable memory behavior.
- Use the medium and large random envelopes as the normal optimization lanes.
  They are representative enough for hill climbing and run much faster than the
  full seed-1234 graph. Treat the full random graph as an occasional safety or
  release validation, not the default iteration benchmark.
- Do not run live vLLM on every iteration. Use stored same-envelope vLLM
  denominators for JAX-only A/B work, and rerun live vLLM only after benchmark
  contract changes, runtime/library/hardware changes, or before promoting a
  new best result.
- Keep benchmark artifacts under `/mountpoint/.exp/diagnostics` or another
  mountpoint path, but commit only summaries, configs, docs, and tests. Do not
  stage `results/*` or full profile/artifact dumps.
- Push checkpoint commits to the remote periodically while working on long GPU
  optimization passes so the current best state is not only local.

## Current Random Decode Speed Scope

- Stay on the random decode graph as the main hill-climb target. Do not switch to
  shape-specific microbench tuning unless it is needed to debug a random-graph
  regression.
- Keep these work items in order: accepted FA/FLA policy validation, broader
  resident/scheduler decode metadata reduction, coarse GDN decode/prefill
  kernels, and model-family-general batched GEMM/fusion improvements.
- Treat standalone attention, append, GDN, and GEMM probes as diagnostic only
  until they improve integrated random decode throughput with correctness and no
  measured-phase JIT growth.
- Current rejected random-speed routes: standalone FA decode Triton, resident
  decode metadata v1/placeholders with scatter sync, static seq-lens carry,
  shared-gather token-carry fallback, physical-row token refs for the full
  greedy vector, and source-level greedy decode bursts. Reopen any of these
  only with a broader boundary that removes a whole per-step operation.
- Current accepted large-random token-carry boundary: packed prefill seeds
  resident slot tokens inside `forward_prefill_token_ids_slot_carry_table_jit`,
  and static decode carries them inside
  `forward_step_token_ids_slot_carry_table_jit`. The runner keeps immutable
  generated-token refs for final materialization, but the resident per-slot
  table owns the next decode input state. This is the preferred route unless a
  broader resident metadata/kernel boundary replaces it.
- Current accepted FA/FLA kernel policy: GDN keeps strict
  `triton_fla_padded` prefill plus packed-projection
  `triton_fla_conv_raw_gates` decode, while full-attention uses
  `triton_packed` prefill and `triton_paged_fused_append` decode. Standalone
  FA decode remains rejected; the accepted FA route is the broader
  packed-prefill plus fused append/decode boundary.
- Current accepted resident metadata route: static decode placeholders are
  shape/active-row keyed, resident block/seq tables are synchronized from host
  mirrors with full-table `device_put` on actual changes, and
  `forward_step_token_ids_resident_slot_carry_jit` owns block table, seq-len,
  token, hybrid-state, KV, and greedy-token updates inside the decode boundary.
- Latest accepted large-random hill-climb result: `495.10 output tok/s` against
  the stored vLLM denominator (`0.560x`), zero measured-phase JIT growth, with
  `_sync_resident_decode_metadata` reduced from `689 ms` to `140 ms` in the
  profiled route. The next bottleneck is model-side decode GPU work
  (GEMMs/attention/GDN) plus remaining PJRT execution overhead, not another
  token-carry rewrite.
