# Agent Instructions

## Subagents

- Do not use `realmai_worker` for new delegated work.
- Prefer `gpt-5.3-codex-spark` for newly spawned subagents unless a task clearly
  needs a different model.

## GPU And Benchmark Commands

- Use `configs/server/gpu_optimal.yaml` as the promoted non-MTP server path.
  The root `server_config.yaml` intentionally mirrors it. Keep optional MTP,
  local CUDA probes, raw GDN experiments, and route-specific warmups in
  separate diagnostic configs/docs rather than adding them to the default
  server path.
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
- As of the 2026-06-18 full-random sweep, the accepted random sidecar prefill
  envelope is `max_num_batched_tokens=1024` with prefill/token buckets
  `128,256,512,1024`. Do not restore the 2048/4096 prefill envelope as a
  default unless a new generic server-style run beats the 1024 envelope without
  extra measured-phase compilation.
- As of the 2026-06-19 reset, the default random sidecar request count is fixed
  at `8` requests. Treat the old 15-request stress as a future B16/B32 scaling
  lane, not the active B8 target. The fixed-8 lane is close to target:
  `770.12 output tok/s` (`0.769x` vLLM) total and `807.16 output tok/s`
  (`0.806x` vLLM) token-event throughput, with the remaining miss mostly final
  device-token materialization/drain.
- XLA low-memory allocator/platform flags are diagnostic only: they can reduce
  GPU memory to about `5 GiB`, but they regressed random/hetero throughput in
  the accepted benchmark lane. XLA Triton GEMM and B16-capacity diagnostics
  showed runtime potential but remain compile-heavy and are not promoted
  defaults.
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
- MTP speed work must be an overlay on the current accepted best serving config
  only. Do not change random-large batch buckets, cache/block geometry,
  FlashInfer full-attention policy, Triton greedy LM-head policy, or
  resident/static decode metadata when testing MTP unless the same change has
  already been promoted by a non-MTP best-path run.
  Corrected K=3 unverified MTP stack test reached only `683.53 output tok/s`,
  below the accepted non-MTP best (`818.91 output tok/s`), so do not promote
  the old collapsed-bucket route. The K=8 unverified diagnostic:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k8_unverified_repeat_r2.json`
  reached `1018.75 output tok/s` with no measured JIT growth, about `1.244x`
  the accepted non-MTP best and `0.997x` the stored vLLM denominator, but this
  is rejected as a valid MTP speedup. It emitted MTP draft tokens without
  target-model verification (`drafts_accepted=0`, `drafts_rejected=0`,
  `bonus_tokens=1397`) and matched only short prefixes against the exact
  best-path output (`0/8` full matches, average prefix `2.375` tokens).
  Unverified MTP append is no longer a valid config surface; stale configs must
  fail rather than run. All MTP benchmarks must use target-model verification
  and report accepted/rejected drafts.
  Keep `NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=0` and
  `NANO_VLLM_JAX_MTP_K1_COMMIT_REJECTED=0` for correctness and speed runs
  unless the run is explicitly labeled as an unsafe diagnostic.
  Verified MTP should not pre-seed drafts during prefill by default
  (`mtp_prefill_seed=false`); on the 2026-06-14 short verified diagnostic this
  reduced TTFT from `335.3 ms` to about `74 ms`. When MTP is scheduler-gated
  off, the decode batch must still use the static/resident metadata hot path;
  otherwise "MTP enabled but gated" remains slower than the accepted no-MTP
  path and benchmark results become misleading. Current exact verified MTP is
  still not a speed path: the same short diagnostic improved from `26.13` to
  `57.87 output tok/s` after the guardrail fixes, but the no-MTP control is
  `84.01 output tok/s`.
  Follow-up exact K>1 GPU diagnostics on 2026-06-14 also failed to produce a
  speed path. The warmup reset fix made K=2 packed-prefill verification
  correctness-clean on the short two-request probe
  (`short8_verified_k2_prefill_generic_after_reset_r4.json` matched the
  no-MTP rows exactly), but throughput was only `15.81 output tok/s` with
  `25%` acceptance. Generic K=2 decode verification with a packed in-JIT host
  payload stayed correctness-clean but reached only `16.75 output tok/s` on
  the same mixed short probe. A high-acceptance B=1 stream
  (`single64_mtp_k2_generic_decode_24_r1.json`) accepted `87.5%` of drafts
  and still reached only `34.69 output tok/s` versus the no-MTP control's
  `63.99 output tok/s`. K=4 was worse on the 40-token B=1 control:
  exact K=4 reached `37.50 output tok/s` versus `109.85 output tok/s`
  no-MTP, and the assume-accepted upper bound reached only `52.53 output
  tok/s`. Do not retry K=2/K=4 generic decode or packed-prefill verification
  as a speed route unless the verifier boundary avoids per-step host decisions
  and the MTP-head/main-width verifier work is materially reduced.
  On 2026-06-15, the verified K=1 path was promoted from the
  `NANO_VLLM_JAX_MTP_FORCE_GENERIC_K=1` diagnostic to explicit
  `mtp_verifier_impl=k_decode`. On the two-request smoke manifest it matched
  the diagnostic acceptance (`11/13`, `84.6%`) with zero measured JIT growth,
  reaching about `28.55-30.41 output tok/s` depending on run variance, still
  below the no-MTP control (`47.25 output tok/s`). K=1 burst verification now
  supports `mtp_burst_groups>1`; burst-2 reached `29.83 output tok/s` with the
  same acceptance and zero JIT growth, so it is only a small host-sync
  amortization win, not a final speed path. A K=1 logit-debug burst run showed
  seed-time MTP top-1 matching verifier top-1 for `5/6` first-group drafts and
  target-in-MTP-top5 for `5/6`; the remaining rejection was a real logit
  disagreement, not a draft bookkeeping bug. A 2026-06-15 follow-up fixed the
  cached GDN `return_first_prefix_state` return path: first-prefix state now
  matches full-prefix token-0 gather in focused coverage, and the one-pass
  two-decode smoke is correctness-clean with `12/13` accepted drafts (`92.3%`).
  It is still not a speed path: reference/packed-projection one-pass reached
  only `22.52 output tok/s`, raw-tail Triton GDN reached `22.60 output tok/s`,
  and conv-tail Triton GDN collapsed to `2.20 output tok/s`, all below
  `k_decode` (`28.97`) and no-MTP (`47.25`). Do not spend more iterations on
  per-token/per-layer GDN kernel swaps for MTP; the remaining verifier blocker
  is the coarse width-2 target-model boundary.
  Later on 2026-06-15, the verified K=1 burst boundary was widened: the
  executor now compacts emitted burst tokens on device and returns per-row
  emitted/accepted/rejected/bonus totals plus an acceptance bitmask, so Python
  no longer walks the `[batch, burst_groups]` count matrix to commit K=1 burst
  outputs. Unit coverage includes mixed accept/reject K=1 burst ordering. A
  forced two-request GPU smoke with reference attention had zero measured JIT
  growth, no backend fallbacks, and sane counters (`7/19` accepted drafts,
  `7/16` position accepts), but remained slow (`19.29 output tok/s`) because
  verifier work and low acceptance still dominate. This validates the widened
  boundary, not a promoted speed path.
  The same day, FlashInfer was made usable in this environment by installing a
  CUDA Torch build and restoring JAX-compatible `triton>=3.6` plus
  `nvidia-cudnn-cu12>=9.8`; keep this package constraint in mind because
  PyTorch 2.5.1's metadata pins older Triton/cuDNN that break JAX. Runtime
  setup now derives `FLASHINFER_CUDA_ARCH_LIST` from NVML when Torch cannot
  discover CUDA. vLLM's relevant pattern is uniform speculative decode metadata
  plus persistent CUDA-graph buffers; MaxText's relevant pattern is a donated
  JIT `generate` transition over persistent decode state. A B=2 packed-prefill
  verifier was added as a diagnostic route and its packed `kv_lens` was fixed
  to include current plus all draft tokens, but it is not correctness-clean yet:
  on the non-boundary two-request FlashInfer smoke it reached `26.87 output
  tok/s` with zero measured JIT growth, close to the decode verifier's `26.78`,
  but diverged from no-MTP at row `len_129` token index 9. The decode verifier
  stayed exact against no-MTP on that same non-boundary smoke while no-MTP
  reached `77.34 output tok/s`. Do not promote packed-prefill verification
  until verifier logits/acceptance match decode verifier under the same inputs.
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
- Borrowed GDN decode status, corrected 2026-06-12: FlashInfer GDN prefill is
  still SM90/SM100-gated on this A10G/SM86 host, but FlashInfer GDN decode and
  vLLM FLA packed recurrent decode both run as raw kernels on A10G. Focused
  Qwen3.5-0.8B shape microbench (`H=HV=16,K=V=128`, FP32 state) recorded
  `results/gdn_decode_kernel_microbench_a10g_qwen08_b1_b2_b4_b8.json`: at B8,
  JAX reference `0.858 ms` p50, local JAX-Triton raw-gate `0.256 ms`, vLLM FLA
  `0.084 ms`, FlashInfer pretranspose `0.086 ms`; output differs from JAX only
  at BF16-output scale (`max_abs <= 5.5e-5`) and FP32 state matches closely
  enough (`max_abs <= 4.5e-8`) for serving diagnostics. The follow-up state-pool
  conv+projection+tail GDN boundary is correct and removes the caller-side
  layer scatter for that route, but it did not produce a stable random-large
  speed win (`776.98` then `749.59 output tok/s` versus `818.17` reference).
  Treat it as graph-replay scaffolding, not a promoted serving path, unless a
  fresh profile shows GDN decode becoming dominant again.
  Do not call Torch/FlashInfer from the JAX serving loop;
  that would introduce a framework/host boundary. The viable borrowed-kernel
  path is a JAX FFI/custom call or a port of the vLLM/FlashInfer kernel behind
  the resident state ABI.
- Do not promote the existing non-conv `triton_fla_raw_gates` route solely from
  the raw microbench. The integrated random-large follow-up
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_decode_microbench_followup_r1.json`
  used the accepted FlashInfer full-attention baseline plus only
  `gdn_packed_decode_impl=triton_fla_raw_gates`; it hit the 70% RAM guard during
  generic warmup/compile after `473 s`, before measurement. Treat the existing
  JAX-Triton raw-gate route as a useful parity/per-kernel probe, not an accepted
  serving route.
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
  exposed a packed-token overpack bug later fixed by Entry 292. Treat Entry 291
  as a broad checkpoint and historical coverage gap, not as a new accepted speed
  record.
- Entry 292 fixed the long-prefill packed-token overpack by making pure prefill
  scheduling respect the same effective compiled token budget used by mixed
  prefill/decode. The current long-prefill slice now chunks into `4096 + 1024`
  packed prefill tokens and completes at
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/long_prefill_fix_20260608/gpu_matrix_long_prefill_bucket_cap_fix_r1.json`,
  `176.77 output tok/s`, `1.519x` stored vLLM. Do not widen the promoted
  random/hetero config to `8192` token buckets unless a new profile proves that
  the extra compile/memory surface is acceptable.
- Hetero8 RAM pressure is bucket-surface pressure, not live-request-specific
  compilation. Generic warmup compiles the selected process's configured
  prefill-token, row, batch, decode, resident metadata, sampled, and inactive-row
  buckets. The matrix runner starts a fresh JAX process per workload, so
  `hetero8` can still hit the 70% RAM guard even if narrower lanes complete.
- Entry 293 added `benchmarks/benchmark_jax_server_multisuite.py` for the
  long-lived-server comparison. With the accepted config policy plus the
  random-large serving envelope, one process warmed `56` JIT keys, then measured
  `random_large` at `818.36 output tok/s` and `hetero8` at `478.64 output
  tok/s`, both with JIT cache `56 -> 56`. This proves the random-large envelope
  covers hetero8 without measured recompilation in one server process, but it is
  not a hetero speed win because the random-large `1024` packed-token cap chunks
  hetero prefill more than the hetero-specialized envelope.
- Do not retry direct JAX `.lower().compile()` executable caching as "graph
  replay". The 2026-06-05 guarded smoke stayed CPU-bound in compile/warmup for
  more than six minutes before measurement. Use XLA/runtime graph replay or a
  backend-owned decode boundary if replay is revisited.
- XLA command-buffer graph-replay knobs are now config-owned through
  `runtime.xla.command_buffer`. The accepted startup flags in this jaxlib are
  `--xla_enable_command_buffers_during_profiling=true`,
  `--xla_gpu_command_buffer_unroll_loops=true`, and
  `--xla_gpu_graph_min_graph_size=<int>`; direct
  `--xla_gpu_enable_cuda_graphs=true` and tested
  `--xla_gpu_enable_command_buffer=...` values are unavailable/invalid. Do not
  promote these knobs blindly: on 2026-06-12, forcing graph min size to `1`
  regressed large random to `777.67 output tok/s` and `239.62 s` warmup, while
  unroll-only tied throughput (`818.58` vs `818.17 output tok/s`) but still
  inflated warmup to `234.69 s`. The current JAX/XLA path already has
  command-buffer regions; further replay work needs a backend-owned decode loop
  or materially broader compiled boundary, not flag-only tuning.
- NVIDIA JAX-Toolbox flag A/B, 2026-06-12: do not promote the tested flags on
  this A10G/SM86 `jax==0.10.0` stack. `JAX_OPTIMIZATION_LEVEL=O1` tied the
  optimized path (`818.27` vs `818.17 output tok/s`) and regressed the plain
  JAX baseline (`180.85` vs `227.72`). `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL`
  regressed optimized random-large to `691.55 output tok/s` and `233.48 s`
  warmup. `--xla_gpu_cudnn_gemm_fusion_level=1` tied throughput (`818.49`) but
  also inflated warmup (`235.44 s`). The optimized path is already about
  `3.59x` the plain JAX baseline on random-large; the remaining gap is
  structural GPU/kernel/scheduler work, not these XLA flag knobs.
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
- Entry 296 tightened request-specific benchmark warmup so it replays the full
  per-request output lengths. The earlier short request warmup compiled only
  the first couple decode steps and missed later active-batch shrink shapes.
  With the patched diagnostic path, the raw-GDN decode config completed the
  reduced 4-request random slice with JIT cache `6 -> 6`, peak child RSS
  `3.48 GB`, and peak system RAM `57.2%`. Reduced generic warmup for the same
  slice also completed with JIT cache `22 -> 22`, but took `177 s` and peaked
  at `4.61 GB` child RSS. This confirms the failed random-large raw-GDN run was
  compile/warmup surface pressure, not KV cache memory for a few requests.
  Repeating the request-specific slice with BF16 activations did not materially
  reduce memory (`8675 MB` GPU after warmup versus `8673 MB` for FP32
  activations), so the dominant retained footprint is runtime/executable
  workspace, not live activation dtype.
- Entry 297 restored the actual random-large benchmark envelope and added
  route-aware generic warmup for greedy runs. Do not compare random-large runs
  with `num_kvcache_blocks=320` against the stored `0.80x` result: the stored
  fast run used `2048` KV blocks, and 320 blocks cannot keep all 8 random-large
  prompts resident. With 320 blocks the current scheduler decodes at max B6 and
  drops to about `0.57x` vLLM. With 2048 blocks and route-aware warmup, current
  reference GDN reaches
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_2048blocks_routeaware_warmup_r2.json`,
  `818.47 output tok/s`, `0.801x` vLLM, JIT cache `24 -> 24`, warmup `61.6 s`,
  peak child RSS `4.56 GB`, peak system RAM `62.1%`. The old full warmup
  compiled sampled routes unnecessarily for greedy and used `56` JIT entries,
  `138.8 s`, and `6.07 GB` child RSS. Raw GDN decode remains rejected for
  serving: the matching route-aware 2048-block run reached only `778.41 output
  tok/s`, `0.762x` vLLM.
- Entry 298 tested broader GDN decode kernel boundaries on the actual
  FlashInfer random-large path using the project venv
  (`.venv/bin/python`; the conda Python lacks `flashinfer`/`jax_tvm_ffi`).
  The route-aware greedy multisuite run reproduced the accepted reference at
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_routeaware_venv_r1.json`,
  `818.33 output tok/s`, JIT `24 -> 24`, warmup `60.59 s`. GDN kernel
  substitutions lost: raw recurrent
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_decode_flashinfer_routeaware_venv_r1.json`
  hit `779.37 output tok/s` (`0.952x` reference), raw recurrent plus separate
  Triton tail epilogue
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_split_tail_decode_flashinfer_routeaware_venv_r1.json`
  hit `759.25 output tok/s` (`0.928x`), and monolithic conv/recurrent/tail
  variants were already slower. Do not retry isolated GDN decode replacement as
  the 0.9x lever on A10G; the next speed route needs a broader serving boundary
  than per-layer GDN decode or a different bottleneck.
- Entry 299 tightened strict GDN fallback behavior for prefill. With
  `gdn_disable_fallbacks=True`/`NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS=1`, packed
  and non-packed GDN prefill now error before the slow recurrent/chunked branch
  if `return_prefix_state`, `return_first_prefix_state`, recurrent prefill, or
  disabled post-conv kernels would prevent the post-conv kernel path. This
  intentionally rejects the packed-prefill MTP verifier route until a
  kernel-backed prefix-state boundary exists; do not re-enable this fallback
  for benchmark runs.
- Entry 329 restored the best exact MTP route after rejecting decode-side
  seed-plus-table-burst experiments. The accepted exact route is still the
  resident-table K=1 two-decode verifier with burst2 steady groups:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verifier_20260615/entry329_mtp_table_burst2_restored_profile_b2_active2.json`
  reached `31.22 output tok/s`, `12/13` accepted, JIT `20 -> 20`, matching
  Entry 324. Do not retry prefill seeding (`29.37 output tok/s`), seed plus
  two verifier groups (`27.70`), or seed plus one verifier group (`29.11`) as
  default routes. Also do not make max-prefill-length decode warmup a default:
  warming the current MTP table-burst verifier at long `seq_lens` crashed in a
  Triton custom call with `CUDA_ERROR_ILLEGAL_ADDRESS`. The remaining exact-MTP
  blockers are the cold first fused seed execution (~493 ms), a measured
  scheduler gap before the first steady burst in short smokes, and the
  target-model verifier cost itself.
