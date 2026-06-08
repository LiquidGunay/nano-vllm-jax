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
- Do not pass an existing diagnostic `*.prompts.jsonl` file back into the
  random sidecar unless it has first been copied to a throwaway path. The
  sidecar may regenerate/rewrite prompt manifests, so exact-envelope A/B runs
  should either call the JAX trace benchmark directly or use a fresh unique
  sidecar output prefix.
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
  greedy vector, source-level greedy decode bursts, resident slot-carry greedy
  decode bursts, standalone Triton LM-head argmax, and standalone FlashInfer
  radix top-k for LM-head selection. Reopen any of these only with a
  backend-owned boundary that reduces model work instead of scanning the full
  model in JAX.
- LM-head fused-GEMM status: `lm_head_greedy_top1_impl` is a config-file-only
  boundary for projection-plus-top1 work. `cutlass` intentionally
  raises until a true CuTeDSL/CUTLASS or library-backed
  `[B,H]x[H,V]->token_ids` backend is implemented. Do not replace it with
  another standalone top-k/argmax selector. A 2026-06-08 scalar CuTeDSL
  no-logits probe matched tokens but took `~100.21 ms` versus XLA
  `dot+argmax` `~1.04 ms` on the representative `B=8,H=1024,V=248064` BF16
  shape; standalone CUTLASS full GEMM was also neutral at `~1035 us`. A later
  JAX-Triton tensor-core top-1 route matched JAX and improved the focused
  microbench (`1.213 ms -> 1.174 ms`) but missed the integrated large-random
  anchor (`753.80 output tok/s` vs accepted `756-758`). Keep
  `lm_head_greedy_top1_impl=triton` diagnostic-only and keep optimized configs
  on `jax` for the older Triton-attention route. On the accepted FlashInfer
  attention route, the same backend is a small integrated win
  (`818.91 output tok/s`, `0.802x` stored vLLM) and is promoted as current
  best, but it is not a large enough lever by itself.
- Current GEMM fragmentation evidence: use
  `benchmarks/summarize_profile_trace.py` with its `Top GEMM Kernels` section
  before changing GEMM code. The usable random-large profile shows GPU `gemm`
  about `992.61 ms / 24735`, `fusion` about `636.04 ms / 79893`,
  `_paged_decode_attention` about `197.94 ms / 1488`, full-vocab LM-head grid
  `1940,1,1`, repeated model GEMM grids such as `56,1,1` and `65,1,1`, and
  `11904` split-K reducers. Current grid totals rank the main GEMM families as
  `1940,1,1` LM-head (`250.23 ms / 248`), `56,1,1` packed MLP gate/up
  (`211.03 ms / 5952`), `65,1,1` packed GDN input projection
  (`175.71 ms / 4464`), and `8,2,5` projection/down/out kernels
  (`152.73 ms / 9192`). Rejected flag routes include
  `--xla_gpu_experimental_force_split_k=1`, XLA Triton GEMM, and disabling
  cuBLASLt; BF16 3/6-way GEMM and CUDA-graph flags are unavailable in the
  current jaxlib. Entry 282 also rejected applying row-padded GEMM only to
  GDN's packed decode input projection: same-manifest large random reached
  `752.99 output tok/s`, below the `753.07` repeat and `756-758` accepted
  anchor. The next route is code-level coarsening of repeated MLP/GDN
  GEMM/fusion groups, not more XLA GEMM flag tuning or single-projection shape
  changes.
- Entry 283 rejected the latest neutral-composition pass. Prefetched token-ref
  direct materialization reached only `752.55 output tok/s`; composing it with
  the Triton LM-head diagnostic reached `753.62 output tok/s`; lowering the
  worker CPU cap to 2 cores reached `751.02 output tok/s`; and widening the
  random-large prefill-token cap to `1536` or `2048` hit the 70% RAM guard
  before measurement. Keep the current 4-core, `max_num_batched_tokens=1024`
  random envelope until memory/compile footprint is reduced, and do not retry
  token materialization as the primary speed lever.
- Entry 284 rejected extending row-padded decode GEMMs to nearby output
  projection sites and a standalone Triton MLP `silu(gate) * up -> down`
  helper. Output projection padding reached only `753.57-753.72 output tok/s`,
  and the MLP helper regressed integrated large-random throughput to
  `723.27 output tok/s` despite one focused microbench tile beating XLA. Do not
  accept microbench-only MLP middle wins; the next MLP/GDN attempt must coarsen
  a larger model-side boundary or replace a full repeated kernel family.
- Entry 285 rejected the latest larger decode-boundary pass. GDN tail-fused
  decode looked neutral on a small B4 probe but regressed target large random to
  `618.30 output tok/s`; table burst-2 reached only `326.19`; temporary
  static-unrolled burst-2 failed the small gate at `266.23`; and temporary
  resident-dense burst-2 reached only `591.44` on large random with a much
  larger warmup surface. Do not retry full-model JAX decode bursts or the
  existing GDN tail custom-call path as primary routes. Revisit only with
  runtime graph replay or a backend-owned boundary that avoids full-model
  `lax.scan` token loops.
- Entry 286 rejected standalone fused FFN RMSNorm plus packed MLP gate/up
  projection. The Triton helper passed focused correctness, but target large
  random fell to `742.07 output tok/s`; final drain improved to `27.28 ms`
  while token-event throughput dropped to `751.69`, so the custom-call route
  hurt model-side decode work. Do not retry this as a standalone boundary.
- Current accepted large-random token-carry boundary: packed prefill seeds
  resident slot tokens inside `forward_prefill_token_ids_slot_carry_table_jit`,
  and compact active decode rows use
  `forward_step_token_ids_resident_dense_slot_carry_jit`. The runner keeps
  immutable generated-token refs for final materialization, but resident
  per-slot tables own next-token, seq-len, block-table, and hybrid-state decode
  state.
- Current accepted sampling boundary: full-vocab temperature sampling can stay
  on the deferred device-token path through `sampled_token_fastpath`. Startup
  must generically warm sampled prefill, generic sampled decode, and resident
  sampled dense decode keys. Top-p/top-k filtering is not part of this boundary
  yet; use a dedicated FlashInfer/Triton sampler kernel before enabling those
  modes on the fast path.
- FlashInfer status: the JAX FFI top-k wrapper is available only as an
  explicit experiment via `lm_head_topk_impl=flashinfer`; large random regressed
  badly, so optimized configs stay on `lm_head_topk_impl=jax`. FlashInfer batch
  decode attention is now promoted for full-attention decode through
  `full_attention_decode_impl=flashinfer_paged` after the persistent-plan,
  fused append/decode route reached `816.58 output tok/s`, `0.799x` of the
  stored large-random vLLM denominator, with zero measured JIT-cache growth.
  Do not retry the old Entry 277 FlashInfer path; that route paid first-use and
  workspace/planning stalls before the current implementation. Also do not
  retry removing the FlashInfer FP32 output cast or coarsening large-random
  decode buckets to powers of two as speed routes. A focused A10G B8 attention
  microbench showed FlashInfer tensor-core decode planning slower than the
  current non-tensor-core path (`0.0855 ms` versus `0.0763 ms`; fixed/disabled
  split-KV variants `0.079-0.081 ms`), so do not add a tensor-core attention
  JAX FFI unless new hardware/profile evidence changes that.
- Borrowed FlashInfer GDN decode status: the installed `flashinfer.gdn_decode`
  CuTe kernels expose the desired pool-indexed state boundary, but they require
  SM90+ while the current GPU is A10G/SM86. They are therefore a blocked
  borrowed-kernel route on this machine, not an integrated serving path.
- Current accepted FA/FLA kernel policy: GDN keeps strict
  `triton_fla_padded` prefill plus explicit packed-projection `reference`
  decode, while full-attention uses `triton_packed` prefill and
  `flashinfer_paged` fused append/decode. Standalone FA decode remains
  rejected; the accepted FA route is the broader packed-prefill plus fused
  append/decode boundary.
- Current accepted resident metadata route: static decode placeholders are
  shape/active-row keyed, resident block/seq host mirrors only synchronize rows
  on actual KV-page/seq changes, small update tensors are built with direct
  `jax.device_put(np.asarray(...))`, and
  `forward_step_token_ids_resident_dense_slot_carry_jit` owns block-table,
  seq-len, token, hybrid-state, KV, and greedy-token updates inside the decode
  boundary.
- Latest accepted large-random hill-climb result: FlashInfer paged
  full-attention decode plus deferred resident RNG-counter resets and the
  promoted Triton greedy LM-head top-1 path reached
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_lm_triton_bn256_r1.json`,
  `818.91 output tok/s`, `0.802x` of the stored vLLM denominator
  (`1021.59 output tok/s`), with `1582` generated tokens, generic warmup, and
  zero measured JIT-cache growth. The remaining gap is GPU-side decode work,
  especially B8 decode model work, not another token materialization tweak.
- Latest broad checkpoint: Entry 291 reran the current checkout on the
  seed-1234 random-large envelope and got
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/broad_benchmark_20260608/broad_random_large_current_r1.json`,
  `812.23 output tok/s`, `0.795x` of the same stored vLLM denominator.
  `decode_heavy_128x128` completed at `179.89 output tok/s`, `0.842x` vLLM and
  `1.185x` stored JAX baseline. The combined matrix and guarded `hetero8` slice
  hit the 70% RAM guard during compile/warmup, and `long_prefill_512_2048`
  currently fails because the promoted config's packed prefill-token buckets
  stop at `4096` while the workload needs `5120`. Treat this as a broad
  checkpoint and coverage gap, not as a new accepted speed record.
- Do not retry direct JAX `.lower().compile()` executable caching as "graph
  replay". The 2026-06-05 guarded smoke stayed CPU-bound in compile/warmup for
  more than six minutes before measurement. Use XLA/runtime graph replay or a
  backend-owned decode boundary if replay is revisited.
- Do not retry Entry 272 rejected branches as primary routes: resident
  prefix-slot compaction, decode block-table buckets `32,64,128`, scalar
  block-entry scatter, in-JIT resident metadata delta application, and
  per-shape grouped final token materialization all lost integrated large-random
  throughput or profile health.
- Do not retry output-tiled single-kernel full MLP fusion. The 2026-06-08
  actual-shape prototype had to recompute packed gate/up activations per
  output-column tile and measured `371.37 ms` median versus `1.76 ms` for the
  JAX/XLA BF16 decode MLP. Future MLP work needs gate/up reuse, grouped/layer
  batching, or a serving-proven GEMM plan.
- Do not retry power-of-two random decode batch buckets as a speed route. The
  inactive-row warmup support is useful for non-exact bucket policies, but the
  target large-random run with `1,2,4,8` buckets reached only `784.07 output
  tok/s` versus the accepted exact-bucket FlashInfer route.
- Do not remove the FlashInfer attention output FP32 cast as a speed route.
  Returning BF16 from the wrapper passed focused FFI tests but regressed the
  integrated large-random target to `815.22 output tok/s`.
