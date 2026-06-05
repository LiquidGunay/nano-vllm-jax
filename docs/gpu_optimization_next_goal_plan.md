# GPU Optimization Next Goal Plan

This file is the source-of-truth planning artifact for the next goal. It captures
the proposed cleanup, benchmark discipline, and kernel roadmap before further
optimization work starts.

## Resolved Decisions

1. FlashInfer and `jax-tvm-ffi` may be used as optional dependencies for opt-in
   kernel backends. They must not become mandatory for the pure-JAX fallback.
2. `gpu_paged_default` means the fastest accepted non-MTP serving configuration,
   not the most conservative historical path. Experimental FFI/custom kernels
   are still excluded until they pass the gates and are promoted.
3. This file may be committed and revised as the next-goal contract evolves.
4. The near-term implementation order should favor easier kernel wins first:
   paged KV append, paged decode attention, and integration scaffolding. GDN
   remains critical, but the next GDN implementation should use the Qwen 3 Next
   vLLM/Flash Linear Attention kernels as references rather than repeating
   source-level JAX or compile-heavy Pallas attempts.
5. If kernel integration is difficult, add a minimal optional kernel-backend
   strategy and ABI validation path before forcing it through the repo's final
   paging layout.
6. Phase 0 should create the baseline docs and benchmark config JSONs in the
   same commit. Treat docs plus config as the first deliverable; the benchmark
   matrix runner can follow in the next commit.
7. The matrix runner should use stored local vLLM/JAX reference artifacts when
   available. If no stored local reference exists for a requested comparison,
   it should run live where possible and store the resulting artifact: live
   vLLM for missing vLLM references and live `gpu_paged_default` for missing
   JAX default references. `--require-stored-references` remains an explicit
   opt-in gate when live fallback is not acceptable.
8. Parity targets are staged. The no-kernel `gpu_paged_default >= 0.75x` vLLM
   gate on the long heterogeneous exact-token benchmark is now met by the
   accepted 2026-05-26 goal-target artifact. The next active serving target is
   a correctness-gated kernel-backed path at `>=0.9x` vLLM on the same
   benchmark discipline. MTP speed work remains out of scope until the
   non-speculative kernel path meets that bar.
9. The existing `long_prefill_512_2048` workload is a valid exact-token,
   shape-synthetic gate, but it is not enough for an external vLLM benchmark
   workload claim. Add a sidecar lane that records vLLM-comparable metadata:
   random/custom-manifest prompts, larger `num_prompts`, explicit seed,
   prompt-manifest hash, output-token throughput, total-token throughput, and
   request throughput. ShareGPT-style serving should be a separate comparability
   track, not a replacement for the long-prefill correctness gate.
10. GDN serving state should move to the kernel-native V,K layout. The canonical
    recurrent-state shape for serving should become
    `[batch, linear_layer, value_heads, value_dim, key_dim]`, with per-layer
    state `[B,HV,V,K]`. The old JAX `[B,HV,K,V]` path is not a reason to keep
    the serving ABI if usable Qwen/vLLM/FlashInfer-style kernels prefer V,K.
    Adapt the pure-JAX fallback/reference math to the V,K layout instead of
    paying hot-path state transposes. Any transition must remain
    correctness-gated and must not silently change activation/state dtype.
11. Keep BF16 GDN prefill activations as a later opt-in experiment, not part of
    the V,K layout migration. The default dtype contract remains BF16 checkpoint
    weights with FP32 activation/state math until a separate BF16-prefill
    experiment passes the full correctness and speed gates.
12. A broad BF16 activation diagnostic is not a shortcut to the target: it
    reached `108.76 tok/s` (`0.935x` the stored vLLM reference) but failed
    exact generated-token parity on the `1536`-token row at the first generated
    token. Any BF16 lane must therefore stay default-off and narrower than a
    whole-model dtype change until it passes full token/logit gates.
13. The narrow vLLM-style BF16 GDN prefill activation diagnostic is also not
    promotable yet. It keeps gate/beta/state FP32 and only casts prepared GDN
    prefill q/k/v plus core output to BF16, but the 500-token top-5 guardrail
    fails (`498/500` ordered top-5, `499/500` top-5 set,
    `max_hf_topk_id_logit_diff=0.010448455810546875`). Keep this path
    default-off unless a true external BF16-input kernel passes the full gates.
14. Stop investing in the local CUDA BF16 packed-decode prototype as an
    optimization route. The BF16 packed-decode reference boundary is correct
    enough for the diagnostic lane, but the local CUDA BF16 implementation
    fails the 500-token HF top-5 guardrail (`499/500` top-1,
    `491/500` ordered top-5, `max_hf_topk_id_logit_diff=0.02606964111328125`).
    Use the BF16 reference route as the semantic contract and go directly to a
    vLLM/FLA-derived kernel route for GDN.
15. JAX-Triton is viable as an optional bridge only if Triton's bundled CUDA
    12.8 `ptxas` is first on `PATH`; the system CUDA 12.0 `ptxas` cannot
    assemble Triton's PTX 8.7. Keep `jax-triton` optional and default-off.
16. Do not promote the current JAX-Triton BF16 GDN decode or prefill-prep-only
    routes. They compile and pass focused tests, but fail the full top-5/logit
    guardrails and remain below target (`0.793x` and `0.797x` vLLM
    diagnostics). The next GDN kernel attempt must port the full vLLM/FLA
    chunk-body schedule behind the existing prepared boundary; split/norm/prep
    alone is too narrow.
17. For the full FLA chunk-body port, start with `chunk_local_cumsum`. It is
    scalar, independently testable, and every later FLA stage consumes its
    cumulative gate output. Porting later stages first makes failures harder to
    localize.

## Active Random Request Contract - 2026-06-04

The active broad serving target is now the seed-`1234` random request sidecar,
with generic startup warmup and no measured-phase JIT growth. This lane is more
representative than the old fixed `hetero8` shape because request count, prompt
lengths, and output lengths vary within a declared range.

Current best:

- artifact:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_noprofile_after_driver_fix_20260604.json`;
- workload: `15` requests, `30506` input tokens, `11602` output tokens,
  prompt lengths `1077..4022`, output lengths `425..1007`;
- generic warmup compiled all batch buckets `1,2,4,8`, prefill token buckets
  `128,256,512,1024,2048`, and decode block-table buckets `128,256,320`;
- generated lengths complete and measured-phase JIT growth was zero
  (`32 -> 32`);
- throughput `437.63 output tok/s`, fresh same-manifest vLLM reference
  `1541.89 output tok/s`, ratio `0.284x`;
- the `0.9x` target requires about `1387.70 output tok/s`, so the remaining
  speedup needed is about `3.17x`.
- the previous accepted run
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_b8_320width_static_cache_split_decode_block_buckets_128_256_320_r1_20260604.json`
  remains equivalent evidence at `436.24 output tok/s`.
- Entry 244 tested mixed packed backfill and greedy decode burst diagnostics.
  Mixed backfill is rejected for automatic scheduling (`165.24` then
  `113.64 output tok/s`) because it delays live decode rows behind prompt
  chunks. Greedy burst has dtype-stable scans and honest generic warmup now,
  but burst4/burst8 resident runs only reached `282.37`/`284.73 output tok/s`
  and remain below the B8 no-burst anchor.

Rules for this lane:

- Do not specialize to seed `1234`, exact request lengths, or Qwen3.5-0.8B
  dimensions.
- Use serving-envelope buckets, not benchmark-specific warmup. Any promoted run
  must have zero measured-phase JIT cache growth.
- Keep committed artifacts to summaries/docs/configs. Full JSON/profile
  artifacts stay under `/mountpoint/.exp/diagnostics` or
  `/mountpoint/.exp/profiles`.
- Treat decode block-table bucketing as accepted but minor. The next material
  path must reduce full decode-step cost, not only metadata width.
- Treat the random gap as a serving-architecture gap, not a single isolated
  kernel gap. The target run must improve arbitrary-batch decode and
  prefill/decode interleaving without overfitting to the seed-`1234` manifest.

Next implementation order:

1. Expand the decode boundary around resident slots:
   - keep a device-resident table for active slot metadata (`block_tables`,
     `seq_lens`, and GDN state), keyed by logical serving slots rather than by
     the transient scheduler row;
   - pass compact active slot ids plus current-token references into the JIT
     decode step, then gather paged-attention metadata, update KV/GDN state,
     run the model, run greedy LM-head selection, and scatter updated resident
     metadata inside that one boundary;
   - preserve paged attention, packed chunked/ragged prefill, and normal decode
     semantics. The generated token is cached only when it becomes a scheduled
     input on the next step.
2. Keep active decode rows compacted without seed-specific scheduling:
   - use general batch buckets such as `1,2,4,8,16` when they fit memory, but do
     not specialize to seed `1234`, exact request lengths, Qwen3.5-0.8B hidden
     sizes, or a specific GPU;
   - avoid widening resident/execution capacity by itself as an optimization:
     Entry 243 showed resident16/B8 regressed because the old executor still ran
     prefill-only or decode-only steps.
3. Integrate production kernels only inside the wider serving boundary:
   - full-attention decode should use a production paged decode-attention path
     only when it consumes the resident-slot paged layout directly;
   - GDN decode should continue toward the vLLM/FLA-style fused boundary with
     BF16 activations and FP32 recurrent state where required;
   - do not promote attention-only or prep-only kernels if the integrated random
     run does not improve.
4. Fuse or narrow LM-head sampling for greedy serving:
   - compute only the greedy token on the serving path unless diagnostics ask
     for logits/top-k;
   - pursue a model-family-general fused matmul/reduction path rather than
     hand-tuned vocabulary or tile constants for this model/GPU.
5. Re-enable chunked/ragged prefill backfill only after decode gets cheaper:
   - keep the explicit mixed packed ABI as groundwork;
   - automatic mixed backfill remains rejected until a latency-aware policy plus
     faster decode/prefill kernels can improve integrated output throughput
     without starving live decode rows.
6. Keep generic startup warmup and benchmark discipline:
   - warm serving-envelope buckets only: batch buckets, packed prefill token
     buckets, and decode block-table widths;
   - after any new decode JIT boundary, run a scaled JAX-only random diagnostic
     before the full seed-`1234` sidecar to avoid repeating the unsafe full
     compile path;
   - run random sidecar experiments through `--jax-config` so the sidecar uses
     the same typed runtime/kernel policy as the server and GPU matrix runner;
   - promoted random runs must keep full output lengths, acceptable correctness,
     and zero measured-phase JIT cache growth;
   - reprofile only after a structural change and compare against the fresh
     random anchor, the Entry 240 anchor, and the live vLLM reference.

Status note, 2026-06-05: resident decode metadata v1 is correct and
cache-stable on a tiny random A/B, but it is slower than the current
table/static metadata path (`67.75` vs `104.33 output tok/s`). Keep it opt-in
as scaffolding only; do not scale it to medium/full random runs until the
gather/scatter/table-update overhead is folded into a larger useful decode
boundary or removed.

Status note, 2026-06-05 r2: the random sidecar now has a subprocess resource
guard (`70%` system-RAM kill threshold by default, bounded worker CPU cores,
and positive worker niceness). The table-prefill boundary now participates in
generic warmup, so packed prefill table shapes are not first compiled during
measurement. The scaled guarded JAX-only random run
`random_config_table_prefill_guarded_scaled_r1` completed with zero
measured-phase JIT growth (`10 -> 10`) and `132.48 output tok/s`, versus the
same tiny accepted-config control at `104.33 output tok/s`. This is only a
promotion signal for the next medium run, not a replacement for the full random
anchor above.

Status note, 2026-06-05 r3: the first guarded medium rung
(`4` requests, `1787` input tokens, `290` output tokens) completed with zero
measured-phase JIT growth. Live same-envelope vLLM measured
`511.94 output tok/s`. After preserving full static decode row alignment in
the device-token carry and avoiding prefill resident-metadata sync when the
resident path is disabled, the JAX medium rung reached `355.14 output tok/s`
without profiling, about `0.69x` that vLLM denominator. The follow-up profile
confirmed CPU `gather` fell from `376.6 ms` to `50.9 ms`; remaining work is
per-step PJRT/device-token carry overhead plus real GPU GEMM/fusion time.

Status note, 2026-06-05 r4: the larger guarded rung
(`8` requests, `6240` input tokens, `1351` output tokens) completed with zero
measured-phase JIT growth and `400.42 output tok/s`; live same-envelope vLLM
was `884.03 output tok/s`, so this rung is `0.453x`. Resource pressure stayed
safe (`<47%` system RAM for the live comparison). Large-rung profiling shows
the next gap is mixed: per-step PJRT/device-token carry/scheduler metadata plus
real GPU GEMM/fusion time. Two follow-ups were rejected on this path:
`static_decode_seq_lens_carry=true` regressed to `341.31 output tok/s`, and a
shared-gather token-carry fallback regressed to `370.84 output tok/s`.

Status note, 2026-06-05 r5: the full seed-`1234` random graph now completes
under the resource guard with zero measured-phase JIT growth (`36 -> 36`) and
safe RAM (`53%` peak system RAM). With `--worker-cpu-cores 2`, JAX measured
`239.35 output tok/s` for the full `15`-request, `30506` input-token,
`11602` output-token workload. Treat this as a safety validation, not a new
performance baseline: the two-core cap likely depresses measured serving
throughput because scheduler/PJRT work remains CPU-visible.

Status note, 2026-06-05 r6: the random decode speed work is now constrained to
five non-negotiable items until the `0.9x` vLLM target is reached:

1. Make full-attention kernel policy explicit and prevent silent fallback in
   benchmark/perf configs. Packed prefill should select a production paged
   attention kernel instead of relying on hidden GPU auto-routing, and benchmark
   artifacts must report the effective kernel policy.
2. Validate and repair paged decode attention on the integrated random workload.
   Standalone exactness is not enough; promotion requires random-graph
   throughput improvement, correctness, and zero measured-phase JIT growth.
3. Reduce per-step host/PJRT metadata overhead with a broader resident decode
   boundary. Do not repeat rejected seq-lens-carry or shared-gather variants
   unless the new design removes an entire per-step host/device operation.
4. Advance coarse GDN decode/prefill kernels using vLLM/FLA-like boundaries that
   own conv, gates, q/k norm, recurrent-state update, and layout together.
5. Improve batched decode GEMM/fusion in a Qwen3.5-dense-family-general way.
   Avoid GPU/model-specific tile hand tuning as the primary path.

First implementation slice: `full_attention.prefill_impl` now controls reference
versus packed Triton prefill routing, `server_config.yaml` and the FA/GDN
benchmark config select `prefill_impl=triton_packed`, and the random sidecar
records the effective kernel policy after config overrides.

Medium random attention A/B, 2026-06-05:

| variant | FA prefill | FA decode | output tok/s | JIT growth | decision |
| --- | --- | --- | ---: | ---: | --- |
| reference control | reference | reference | `358.09` | `0` | baseline |
| prefill-only Triton | triton_packed | reference | `360.10` | `0` | keep as small win |
| decode-only Triton | reference | triton_paged | `351.03` | `0` | reject standalone |
| prefill+decode Triton | triton_packed | triton_paged | `347.40` | `0` | reject standalone |

Interpretation: packed prefill attention is worth keeping because it removes a
hidden fallback and slightly improves the integrated random medium rung. The
current width-1 paged decode Triton kernel is not an integrated serving win even
though it is algorithmically closer to vLLM than materialized JAX attention. Do
not promote it as the default. The next decode-attention attempt must broaden
the boundary so cache append, metadata ownership, and attention read/setup are
not paid as separate per-layer/per-step work.

Resident boundary A/B, 2026-06-05:

- implementation slice: when `resident_decode_metadata=true`, static decode
  metadata now keeps true block tables and sequence lengths in host mirrors for
  resident-table synchronization, but passes reusable zero placeholders for the
  device `ScheduledBatch.block_tables` and `ScheduledBatch.seq_lens` fields
  ignored by `forward_step_token_ids_resident_jit`.
- medium random result:
  `random_resident_placeholder_medium_r1` completed the same `4`-request,
  `1787` input-token, `290` output-token envelope at `354.32 output tok/s`,
  zero measured-phase JIT growth, and `45.0%` peak system RAM.
- comparison: the current promoted medium route,
  `random_fa_prefill_triton_medium_r1`, reached `360.10 output tok/s` on the
  same envelope.
- decision: keep the placeholder behavior as safer resident-path scaffolding,
  but do not promote `resident_decode_metadata` for the random serving path.
  The old resident route's main cost is not just the ignored device metadata
  arrays; its gather/scatter/table-update boundary must remove a larger
  per-step operation before it can beat table/static metadata.

GDN/GEMM boundary audit, 2026-06-05:

- the larger random profile still shows real GPU work: `fusion` plus `cutlass`
  totals about `1064 ms`; `_gdn_conv_packed_decode_raw_gate_kernel` is
  `112.35 ms` over `4983` calls, matching one fused GDN decode kernel per GDN
  layer per decode step.
- the active GDN decode route already fuses conv update, q/k normalization,
  raw gate/beta math, and recurrent-state update inside the Triton
  `triton_fla_conv_raw_gates` boundary.
- do not repeat the previously rejected source-level greedy decode-burst path:
  burst-16 and burst-2 were exact but much slower because a full-model JAX scan
  increased gather/transpose/PJRT overhead.
- next GDN/GEMM work must be structurally coarser, for example
  projection-plus-GDN or a backend-owned multi-layer/decode boundary, or a
  model-family-general greedy LM-head matmul+argmax epilogue. Do not spend the
  next pass sweeping padded-GEMM rows or narrow GDN launch parameters.

Random benchmark policy, 2026-06-05:

- use the medium and large random envelopes as the normal hill-climb lanes.
  They exercise the same serving mechanisms that matter for the full random
  graph, but keep compile/run time low enough for iteration.
- the full seed-`1234` random graph remains a safety/release validation, not
  the default benchmark after every edit.
- use stored same-envelope vLLM denominators for JAX-only A/B work. Rerun live
  vLLM only after benchmark-contract changes, runtime/library/hardware changes,
  or before promoting a new best result.
- current stored medium denominator:
  `random_promoted_medium_with_vllm_r1`, `4` requests, `1787` input tokens,
  `290` output tokens, vLLM `471.06 output tok/s`, JAX
  `355.77 output tok/s`, JAX/vLLM `0.755x`, zero measured-phase JIT growth
  (`15 -> 15`).
- current stored large denominator remains
  `random_config_table_prefill_token_carry_large_with_vllm_r1`: `8` requests,
  `6240` input tokens, `1351` output tokens, vLLM `884.03 output tok/s`, JAX
  `400.42 output tok/s`, JAX/vLLM `0.453x`, zero measured-phase JIT growth.

Micro-burst selection model:

- A width-`K` greedy burst replaces `K` host/PJRT scheduler executions with one
  compiled execution that loops or scans `K` decode iterations on device. It
  reduces fixed per-step overhead but increases on-device work per call and may
  delay scheduler decisions.
- Use a simple runtime score for each warmed burst bucket:
  `estimated_seconds(K) = fixed_step_overhead + K * device_decode_seconds +
  wasted_masked_seconds(K) + scheduling_delay_penalty(K)`.
- Estimate `fixed_step_overhead` from burst-`1` timing and profile counters;
  estimate `device_decode_seconds` from recent same-batch decode medians; bound
  `wasted_masked_seconds` by the number of rows that will finish before `K`;
  bound `scheduling_delay_penalty` by waiting-prefill/backfill pressure.
- Select the largest warmed `K` whose predicted win over `K` separate burst-`1`
  steps is positive and whose delay penalty is below a configured service
  bound. This makes the policy portable across GPUs: different hardware changes
  the measured coefficients, not the algorithm.
- The default admissible bucket set should stay small and model-family
  general. Expanding it is a benchmarked policy change, not a hand-tuned tile
  sweep.

## Active Hetero8 Contract - 2026-06-03

This section supersedes the older decode-heavy worker plan below. The active
target is the fair, generic-warmup `hetero8` lane for
`gpu_paged_gdn_fla_decode_static_metadata`, not the older no-profile
diagnostic lane and not the already-met decode-heavy micro-target.

Current anchor:

- artifact:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/typed_projection_matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
- exact generated-token parity and full generated lengths;
- generic bucket startup warmup compiled `26` entries, and measurement had
  zero JIT cache growth (`26 -> 26`);
- throughput `514.16 tok/s`, stored vLLM reference `864.18 tok/s`, ratio
  `0.595x`; the `0.9x` target requires about `777.76 tok/s`, so the remaining
  required speedup is about `1.51x`.

Rules for the current pass:

- Only work directly unless the user explicitly re-enables subagents. If
  subagents are re-enabled, use the `AGENTS.md` guidance and prefer
  `gpt-5.3-codex-spark`.
- Keep accepted serving controls in config/runtime/kernel policy. Do not add
  new default-on hot-path environment-variable gates.
- Keep benchmark artifacts under `/mountpoint/.exp/diagnostics/...` unless the
  benchmark tool's existing output contract explicitly writes elsewhere.
- Require exact generated-token parity, full lengths, and zero measured-phase
  JIT growth for every promoted serving-path result.
- Avoid parameter-only sweeps. The next useful changes must be structural and
  model-family general across Qwen3.5 dense sizes.
- Do not retry the rejected Entry 225 metadata warmup variants: long-context
  decode warmup, static seq-lens carry alone, scheduler metadata warmup, or
  prefill-seeded seq-lens placeholders. They were exact but slower and moved
  latency rather than eliminating it.
- Do not retry Entry 226/227 materialization or seq-lens carry variants:
  host-side stacked `DeviceTokenRef` materialization and graph-returned
  `next_seq_lens` were both exact but slower. A future metadata route must
  remove more of the serving boundary, not only move token or length values.
- Do not retry Entry 228's direct full-active table decode specialization.
  Removing source-level hybrid table gather/scatter for row-aligned full
  batches was exact but slower; future full-active work needs profile evidence
  that a concrete lowered bucket disappears.

Next implementation order:

1. Keep the packed paged chunked-prefill ABI as the main serving grammar:
   prefill shape is total token bucket plus ragged metadata; decode shape is a
   finite batch bucket.
2. Replace reference prefill bodies behind that ABI with production-shaped
   kernels only after each boundary is exact against the dense/reference path:
   packed full-attention prefill first, then vLLM/FLA-style varlen GDN chunk
   body.
3. For hetero8 decode, target a broader fused model-side decode boundary that
   reduces launch count or materialization across projection/MLP/GDN/LM-head
   work. Static-metadata toggles, token-carry shape movement, source-only
   gate/up packing, and single-query Triton attention are already rejected in
   the logbook.
4. Make request metadata/block tables device-owned across the prefill-to-decode
   boundary, or integrate metadata updates into the compiled decode boundary, so
   scheduler hot-path work passes small slot ids instead of rebuilding device
   arrays every step.
5. Reprofile only after a structural change. Treat CPU buckets that move
   without end-to-end throughput movement as synchronization/accounting
   evidence, not standalone wins.

## Active Decode Plan - 2026-06-02

The decode-heavy `0.9x` vLLM goal is now met by the Entry 210 composed route:
strict block-dot FLA prefill plus explicit BF16-QKV packed-reference GDN decode
with static metadata and padded decode GEMMs. The current no-profile
`decode_heavy_128x128` repeats are exact at `197.91`, `196.38`, and
`197.96 tok/s`, median `197.91 tok/s`. Against the fresh same-shape vLLM async
baseline `219.03 tok/s`, the median ratio is `0.904x`; the fresh `0.9x` target
is about `197.12 tok/s`.

This is a decode-heavy target hit, not a blanket serving win. The remaining
work is to preserve this config as the baseline, validate broader workloads,
and only then attack the remaining decode GEMM/reduction buckets if they block
the broader claim.

Full no-profile sanity after promotion:

- `hetero8`: exact, median `317.35 tok/s`, `0.367x` stored vLLM;
- `short_32_128`: exact, median `386.52 tok/s`, `0.680x` stored vLLM;
- `long_prefill_512_2048`: exact, median `106.08 tok/s`, `0.912x` stored vLLM;
- `decode_heavy_128x128`: exact, median `197.60 tok/s`, `0.902x` fresh vLLM.

The config is correctness-clean across the sanity set. The target claim is
decode-heavy plus long-prefill; hetero8 remains open.

Next implementation order:

1. Preserve and remeasure the Entry 210 baseline. Use
   `gpu_paged_gdn_fla_decode_static_metadata` as the strict decode-heavy
   baseline: `prefill_block_dot=true`, `packed_decode.impl=reference`,
   `qkv_dtype=bf16`, static metadata, padded decode GEMMs, and fallbacks
   disabled.
2. Broaden the claim to hetero8 and long-prefill using the same composition.
   Entry 197 already hit `0.901x` vLLM on long-prefill with all block-dot FLA
   stages, while hetero8 remains below target and decode-dominated. Do not
   infer hetero8 from the decode-heavy win.
3. Decode projection/GEMM buckets remain the next model-side targets only if
   broader workloads fail. The current profiled config smoke still shows
   LM-head `gemm_fusion_dot_199` around `126 ms`, MLP gate/up around `109 ms`,
   GDN QKV around `65 ms`, and MLP down around `51 ms`.
4. Greedy LM-head top-1 should only be revisited as a materially different
   single-call GEMM/matvec-plus-argmax epilogue. Source-level
   `top_k(logits, 1)` and two-stage Pallas argmax are rejected.
5. Device-owned cache/state metadata should only be changed where the profile
   confirms an integrated serving win. Entry 209 rejected another batch of
   source-boundary and XLA-command-buffer spelling changes.
6. Do not continue target work on narrow GDN decode-core boundary widening.
   Entry 201 and Entry 203 showed that raw gates, in-kernel Q/K norm changes,
   layer-internal state-table routing, and fusing width-1 conv with recurrent
   decode are correctness-clean or close but do not move integrated
   decode-heavy throughput.

Benchmark discipline for every slice:

- use strict no-fallback configs for kernel-route claims;
- record exact generated-token parity, vLLM ratio, top CPU/GPU profile events,
  and whether the change targets decode, long prefill, or both;
- keep microbenchmarks as diagnostics only until the integrated GPU matrix
  confirms the win.

Orchestration assignments:

- Worker A owns static decode execution and replay. Target the CPU-side
  `forward_step_token_ids_jit`, `PjRtCApiLoadedExecutable::Execute`,
  `command_buffer::execute`, and `command_buffer::update` buckets by reducing
  repeated host metadata rebuilds and keeping fixed decode-bucket state on
  device.
- Worker B owns greedy LM-head top-1 and logits materialization. Target the
  remaining top `gemm_fusion_dot_199` bucket without changing exact
  `jnp.argmax` tie behavior; keep full logits only where diagnostics or
  non-greedy modes require them.
- Worker C owns decode projection structure. Target remaining projection GEMM
  launch count and shape quality through safe gate/up and QKV packing or
  merging, while keeping the accepted padded-GEMM route compatible.
- Worker D owns strict GDN/FLA prefill and decode kernels. Target the packed
  GDN decode bucket and the much larger long-prefill gap using Pallas, Triton,
  CuteDSL, or adapted external FLA schedules, with local CUDA probes remaining
  diagnostics only.
- Worker E owns benchmark/profile strategy. Keep the plan aligned with
  nano-vLLM/vLLM structural advantages: merged projections, CUDA-graph-like
  replay, paged/cache metadata discipline, FlashAttention/FlashInfer-style
  kernels, reduced host sync, and optimized GEMM only where profile evidence
  shows a real shape or fusion problem.

Post-pass orchestration rule: avoid further fan-out unless there is a clear
need for independent read-only checks. Use one reusable worker for bounded
follow-up slices, refresh that worker's brief with the current anchor artifacts,
and close/reset it when its context becomes noisy.

Do-not-repeat guardrail for the current decode pass:

- Do not retry static `seq_lens` device carry for static metadata. It regressed
  the confirmed run to `165.31 tok/s` and worsened `MemcpyD2D`/`np.asarray`
  buckets.
- Do not retry `_gdn_fla_chunk_fwd_o_packed_kernel` `num_warps=4`; it raised
  the `chunk_fwd_o` bucket to about `80 ms` and regressed throughput to
  `159.43 tok/s`.
- Do not continue packed-GDN decode launch-param-only sweeps such as
  `w4/s2/b32` or `w8/s3/b32`; they did not reduce `_gdn_packed_decode_kernel`
  or the dominant GEMM/GDN buckets. Reopen only with a kernel-body or launch
  structure change.
- Do not retry runtime/source-level MLP gate/up concatenation or persistent
  duplicated gate/up weight leaves in the rejected forms already recorded in
  the logbook. A future gate/up proposal must explain the material difference
  from those failures before implementation.
- Do not retry token-carry shape movement by itself. The executor carry-column
  and table-decode carried-vector variants were exact but slower on packed
  generic hetero8 (`508.41` and `507.66 tok/s` versus the `512.56 tok/s`
  anchor) because lower recorded decode-step time was offset by higher TTFT and
  final materialization gap.
- Do not promote the single-query Triton full-attention decode probe. It passed
  focused parity but regressed packed generic hetero8 to `503.32 tok/s` and had
  a long first compile; future attention work needs a materially different
  boundary, such as a borrowed production paged-attention kernel or broader
  cache/layout change.

## Next Goal Handoff

Use this file as the main goal contract. The first goal slice should stay
documentation/configuration-first:

1. Keep the docs plus benchmark config JSONs as the first deliverable.
2. Treat `gpu_paged_default` as the fastest accepted non-MTP default, not the
   most conservative historical baseline.
3. Keep FlashInfer, `jax-tvm-ffi`, and FLA/vLLM-derived external kernels as
   optional dependencies behind backend flags with pure-JAX fallbacks. The
   latest implementation decision explicitly allows invasive kernel work as
   long as each slice keeps one reference implementation and one fastest
   implementation. Local CUDA/JAX FFI may therefore be used as an opt-in fast
   implementation for a production-shaped boundary, but it must remain
   default-off until integrated correctness and speed gates pass.
4. Prefer stored local artifacts for JAX and vLLM comparisons. If a selected
   workload has no stored local artifact, run the missing comparison live when a
   GPU and dependency stack are available, then store that artifact for future
   comparisons.
5. Treat the no-kernel long heterogeneous `0.75x` vLLM target as achieved by
   `results/gpu_matrix_20260526_141130.json`: two repeats, exact generated-token
   parity, full profile-counter coverage, and `0.764x` vLLM.
6. The active kernel-phase goal is now `0.9x` vLLM. The `0.9x` target applies
   only to correctness-gated kernel-backed non-speculative serving paths, not to
   MTP speculative decoding.
7. Before treating `0.75x` or `0.9x` as an externally comparable vLLM-benchmark
   claim, run a diverse prompt-manifest sidecar as well as the existing
   exact-token long-prefill gate. The current repeated-seed prompt suite remains
   useful for regression control because it freezes token IDs and shapes.
8. Treat V,K as the next GDN layout target. The first implementation slice
   should update the JAX recurrence/chunk references and hybrid-state
   initialization/tests for `[B,L,HV,V,K]`, then route external kernels against
   that layout. Do not add per-token or per-layer K,V<->V,K adapters in the
   server hot path.
9. Do not combine the V,K layout migration with BF16 GDN prefill activation
   changes. If FLA/FlashInfer integration suggests BF16 prefill is valuable,
   add it later behind an explicit opt-in flag such as
   `NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE=bf16`. This is a speed experiment for
   external-kernel compatibility, not a correctness-contract change for the
   default path.
10. Do not promote the historical standalone local CUDA GDN probes as the
    serving default. The local CUDA/JAX FFI GDN implementations are now
    diagnostics only unless a later design explicitly revives one for a focused
    replay fixture. The next real kernel route should use FlashInfer for paged
    KV/attention and vLLM/Flash Linear Attention references for GDN.
11. Treat direct FlashInfer GDN prefill as blocked on the current A10G/SM86
    host. The local FlashInfer GDN path requires SM90/SM100, so the actionable
    GDN route is a vLLM/FLA-derived JAX-facing port.
12. Run GPU, benchmark, profiling, vLLM, CUDA, NVIDIA, and model-serving
    commands outside the sandbox with elevated access. The sandbox can miss
    `/dev/nvidia*` and report false GPU communication failures. Keep all
    benchmark/model/cache/temp paths under `/mountpoint/.exp`.

Current GDN status: serving GDN is still expressed as JAX and lowered by XLA to
GPU work; there is no accepted hand-owned GDN kernel in the default path.
The GDN bottleneck is profile-backed by XLA/PjRt trace buckets and integrated
benchmark deltas, especially the accepted chunk-size-32 movement and later
row/chunk regressions. It is not yet proven by a separate standalone HLO-only
audit. The next GDN kernel attempt should use Qwen 3 Next vLLM and Flash Linear
Attention as implementation references. The GDN state-layout decision is now to
move serving state to V,K `[B,L,HV,V,K]` and adapt the pure-JAX fallback to that
layout, rather than preserving `[B,L,HV,K,V]` as the long-term ABI.
The current external-GDN audit says direct reuse is not a drop-in FP32 route:
the installed vLLM/FLA prefill path rejects FP32 activation tensors, and
FlashInfer GDN kernels are half/BF16 oriented. The packed decode port/fork
against the `gdn_fla` reference boundary was implemented and rejected for
promotion. The first coarser post-conv prefill reference boundary now exists:
`NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference` routes split, gate
construction, valid-token masking, GQA repeat, layout packing, and chunked
prefill through `backend.gated_delta_prefill_post_conv` while preserving the
pure-JAX FP32/V,K correctness path. The next real GDN step is to replace work
behind that same post-conv boundary with a vLLM/FLA-derived fast implementation,
not to add another model-call-site rewrite.
The latest external-kernel audit narrowed this further: vLLM FLA post-conv prep
emits q/k/v in the input activation dtype, vLLM FLA chunk explicitly rejects
FP32 q/k/v, and FlashInfer GDN prefill accepts BF16/FP16 q/k/v with FP32
g/beta/state. Preserving the default contract requires a FP32 port/fork of the
FLA chunk schedule behind the existing post-conv boundary. A FlashInfer BF16
prefill experiment is allowed only as a separate diagnostic lane. The current
GPU host is NVIDIA A10G / SM86, while FlashInfer GDN prefill requires SM90 or
SM100, so direct FlashInfer GDN prefill is blocked locally even before the dtype
contract question.
A Torch-side vLLM/FLA microprobe now confirms the vendored FLA kernels are
runnable on this SM86 host. On the model-shaped varlen prefill
`[512,1024,1536,2048]` (`5120` total tokens), vLLM's BF16
`fused_post_conv_prep + chunk_gated_delta_rule` takes `1.45 ms` p50 per GDN
body, with FP32 gate/beta/state and V,K state `[N,HV,V,K]`. Packed BF16 decode
takes about `0.11-0.16 ms` p50 for batch sizes `1,4,8,16`. This is a porting
decision artifact, not a correctness or serving speed claim: the default
contract remains FP32 activation/state math until a JAX-facing path passes the
token/logit gates. The evidence says the next useful implementation should
port/fork the FLA schedule, not keep adapting the old local FP32 chunk body.
The same artifact now includes independent Torch recurrent reference checks:
packed decode matches BF16 output exactly and FP32 state within
`1.2e-7` max abs, while ragged prefill `[17,64,65]` differs by one BF16 output
quantum (`4.88e-4` max abs) and `4.45e-3` max abs in final FP32 state.
Use these as upstream-kernel semantic targets for a port/fork; integrated JAX
promotion still requires the stricter model-level token/logit gates.
A sidecar reuse audit confirms direct Torch/vLLM Triton reuse from inside JAX is
not a low-risk path on this stack: the repo venv has no `jax_triton`, the vLLM
venv has no JAX/`jax_tvm_ffi`, vLLM exposes Torch/Triton wrappers rather than a
stable non-Torch ABI, and this JAX build does not expose an easy JIT-safe DLPack
export path. Treat vLLM/FLA as a golden-reference implementation and port/fork
the needed schedule behind the existing JAX-facing post-conv or packed-decode
boundary.
The concrete vLLM/FLA prefill port surface is now identified. After
`fused_post_conv_prep`, vLLM's `chunk_gated_delta_rule` runs a deterministic FLA
pipeline: local cumulative decay, scaled dot KKT, triangular solve, WY
recompute, state/chunk update, and output materialization. The relevant source
functions are `chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`, `solve_tril`,
`recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h`, and `chunk_fwd_o`, plus
`prepare_chunk_indices` and `prepare_chunk_offsets` for varlen chunk metadata.
These math kernels are the preferred port/fork surface. Do not port Torch
autograd wrappers, `input_guard`, vLLM runtime context, or autotune/env plumbing
as part of the first JAX path. On A10G/SM86, vLLM uses the non-TMA path; the
same route is the local target.
The post-conv reference now exposes an explicit FLA-shaped FP32 prep helper:
`prepare_gdn_post_conv_prefill_fla_inputs_from_decay` returns query/key/value in
`[B,T,H,D]`, gate/beta in `[B,T,H]`, and row lengths, with optional q/k L2
normalization. The existing reference path still transposes into the current
chunk body, so this is an ABI scaffold rather than a speed claim.
The prepared-body reference is also explicit:
`gdn_fla_prefill_chunk32_fp32_reference` consumes normalized q/k, value,
gate/beta, row lengths, and state in the future fast-body layout, masks padded
tokens from `seq_lens`, then falls back to the current chunk rule. This is the
correctness target for the next FP32 CUDA/FLA-derived chunk body.
The JAX-side FLA varlen contract is now explicit too:
`pack_prepared_gdn_prefill_inputs` converts prepared `[B,T,H,D]` tensors into
the upstream-style `[nnz,H,D] + cu_seqlens` ABI, and
`gdn_fla_prefill_varlen_reference` round-trips through the segmented reference
and unpacks to `[B,T,H,V]`. This gives a focused boundary for a future
FLA-schedule port without forcing hot-path K,V/V,K state adapters.
The FLA varlen chunk metadata helper now exists as well:
`prepare_gdn_fla_chunk_metadata` emits active chunk indices and per-row chunk
offsets for the future FLA chunk-body port. It deliberately preserves original
row ids across zero-length padded rows, unlike the upstream helper's
pre-filtered-row assumption.
The first scalar FLA math-stage reference is now explicit:
`gdn_fla_chunk_local_cumsum_packed_reference` implements vLLM's scalar
`chunk_local_cumsum` semantics over packed `[nnz,H]` gates, including per-chunk
reset behavior and reverse cumulative sums. It is a correctness reference for a
future kernel stage, not a serving speed path.
The second scalar FLA math-stage reference is also explicit:
`gdn_fla_chunk_scaled_dot_kkt_packed_reference` implements the strict-lower
`beta * K * K^T` chunk matrix over packed varlen keys, including optional
`exp(g_i - g_j)` scaling and output-head to key-head grouping. This locks down
the next port/fork stage before `solve_tril`.
The third scalar FLA math-stage reference is now explicit:
`gdn_fla_solve_tril_packed_reference` implements the per-active-chunk
`(I + A)^-1` solve over packed varlen `[nnz,H,BT]` matrices, preserving padded
zero columns outside each ragged chunk. This locks down the triangular solve
stage before `recompute_w_u_fwd`.
The fourth FLA math-stage reference is now explicit:
`gdn_fla_recompute_w_u_packed_reference` implements vLLM/FLA's
`recompute_w_u_fwd` semantics over packed varlen tensors, producing `w` and
`u` from the solved chunk matrix, beta, local gate cumsums, grouped keys, and
values.
The fifth FLA math-stage reference is now explicit:
`gdn_fla_chunk_delta_h_packed_reference` implements vLLM/FLA's
`chunk_gated_delta_rule_fwd_h` state/value update over packed varlen tensors.
It uses `chunk_offsets` for per-sequence chunk traversal, stores `v_new` before
gate rescaling, and applies gate-rescaled deltas to the FP32 recurrent state.
The final scalar FLA chunk-body reference is now explicit:
`gdn_fla_chunk_fwd_o_packed_reference` implements vLLM/FLA's `chunk_fwd_o`
output stage over packed varlen tensors, combining query-to-prior-state output
with causal intra-chunk attention over ungated `v_new`.
The composed FLA packed-body reference is now explicit too:
`gdn_fla_chunk_gated_delta_rule_packed_reference` runs the full audited FLA
stage order over packed ragged tensors and matches the existing segmented JAX
reference on a ragged normalized-Q/K parity test.
The BF16 packed-decode reference boundary is also useful but only as a semantic
contract: it passes the diagnostic long-decode lane, while the local CUDA BF16
packed-decode implementation fails full-model top-5 parity. Do not spend more
optimization effort on that local CUDA route. The next implementation should
replace `gdn_fla_chunk_gated_delta_rule_packed_reference` or the prepared
post-conv body with a vLLM/FLA-derived lowered implementation and keep the
reference path as the correctness oracle.

Immediate kernel implementation checkpoint: the latest elevated long-prefill
target artifact, `results/gpu_matrix_20260527_current_goal_target.json`, is
speed-claim-ready and exact at `90.87 tok/s`, while the stored vLLM reference is
`116.37 tok/s`; the active `0.9x` target is `104.74 tok/s`, leaving a
`13.86 tok/s` gap. Scheduler diagnostics show one prefill step at about
`0.53 s` and 15 decode steps totaling about `0.17 s`. The packed-GDN decode
route now has a selected first slice: `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL`
routes width-1 cached decode through a vLLM-shaped packed boundary with
`reference` and `cuda_fp32` implementations. The integrated exact-token
benchmark rejected that path for promotion. A separate greedy decode-burst
experiment also rejected a device-side multi-token decode loop: it reduced
token readback count but lowered the full-model scan poorly and regressed
throughput badly. A narrower `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1` experiment
kept token IDs on device between greedy decode steps and is exact, with the
current fastest local target throughput at `93.01 tok/s` (`0.799x` vLLM), but
it changes streaming TTFT/ITL semantics and is still below the active target,
so it remains default-off. The next step should return to coarser GDN
post-conv/kernel boundaries, not larger source-level JAX decode loops.

## Baseline And Best-Run Tracking

Keep two records for each tracked workload:

- accepted baseline: the current default path that is correctness-gated and
  eligible for comparison by future PRs.
- fastest achieved run: the fastest locally observed run, even if it is not yet
  accepted because it is experimental, missing a gate, or uses opt-in kernels.

Each record must include artifact path, date, model, hardware, workload shape,
env/kernel flags, generated-token parity, top-k/logit guardrail status when
available, throughput, TTFT p50/p95, ITL p50/p95, and vLLM ratio when a vLLM
reference exists. The accepted baseline remains the comparison anchor; the
fastest achieved run is a progress marker and must not replace the baseline
unless it passes the acceptance gates.

Current tracked records:

- Entry 045 hetero8 accepted baseline:
  `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`,
  `367.80 tok/s`, `0.426x` the stored vLLM async reference.
- Long-prefill accepted no-kernel best/default after the GDN V,K JAX layout
  migration:
  `results/gpu_matrix_20260526_vk_layout.json`, `90.65 tok/s`, `0.779x` the
  stored vLLM reference, exact generated-token parity over two repeats.
- Current long-prefill elevated revalidation:
  `results/gpu_matrix_20260527_current_goal_target.json`, `90.87 tok/s`,
  `0.781x` the stored vLLM reference, exact generated-token parity over two
  repeats, speed-claim-ready, but still below the active `0.9x` gate. The gap is
  `13.86 tok/s`, or about `1.153x` required JAX speedup.
- Current scoped-profile long-prefill revalidation:
  `results/gpu_matrix_20260527_scoped_profile_target.json`, `90.81 tok/s`,
  `0.780x` the stored vLLM reference, exact generated-token parity over two
  repeats, speed-claim-ready, and still below the active `0.9x` gate. This is
  the first target artifact whose matrix report includes scoped GPU/CPU top
  profile events.
- Current rejected packed-GDN CUDA FP32 target:
  `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.json`, `88.41
  tok/s`, `0.760x` the stored vLLM reference, exact generated-token parity over
  two repeats, speed-claim-ready, but slower than the current accepted/scoped
  default. It records `run_config.gdn_kernel_flags.packed_decode_impl =
  cuda_fp32` and remains default-off.
- Current rejected greedy decode-burst target:
  `results/gpu_matrix_20260527_decode_burst_target.json`, `17.46 tok/s`,
  `0.150x` the stored vLLM reference, exact generated-token parity over two
  repeats, but not speed-claim-ready and far slower than default. It records
  `run_config.serving_fastpath_flags.greedy_decode_burst_steps = 16` and remains
  default-off. The one-repeat burst-2 probe was even slower at `5.57 tok/s`.
- Current fastest default-off device-token-carry target:
  `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.json`,
  `95.14 tok/s`, `0.818x` the stored vLLM reference, exact generated-token
  parity over two repeats, and `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1`. This is a
  fastest-run marker only, not the accepted baseline: it defers token
  materialization until the end of greedy `ignore_eos` generation, so per-token
  streaming TTFT/ITL measurements are no longer equivalent to the default
  server path. The vector-ref materializer reduces the prior device-carry
  `PjRt Execute` count from `255` to `59` and `MemcpyD2D` count from `749` to
  `493`, but the route remains below the `0.9x` gate and not speed-claim-ready
  under the current profile-counter policy. A stacked final materialization
  follow-up regressed to `91.94 tok/s`, so keep the Entry 127 tuple
  materializer as the current marker and return optimization effort to
  model/kernel boundaries.
- Current default-off GDN post-conv reference boundary:
  `results/gpu_matrix_20260527_gdn_post_conv_reference_target.json`, `90.15
  tok/s`, `0.775x` the stored vLLM reference, exact generated-token parity on
  the one-repeat integrated route, and
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference`. This is not
  speed-claim-ready because it has only one repeat and is below the `0.9x`
  target. It is an accepted boundary/correctness scaffold for the next
  vLLM/FLA-derived post-conv fast implementation.
- Current rejected GDN CUDA post-conv prep target:
  `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.json`, `87.80
  tok/s`, `0.755x` the stored vLLM reference, exact generated-token parity on
  the one-repeat integrated route, and
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_fp32`. The vLLM-inspired
  prep-only CUDA FFI reduces transpose and D2D profile buckets but adds command
  buffer work and loses integrated throughput, so it remains a rejected
  default-off diagnostic. The next candidate must also replace the chunked GDN
  prefill body behind the same boundary before it can plausibly help TTFT.
- Current rejected GDN CUDA post-conv prep+chunk target:
  `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.json`,
  `60.46 tok/s`, `0.520x` the stored vLLM reference, exact generated-token
  parity on the one-repeat integrated route, and
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_prefill_fp32`. This route
  proves the existing local FP32 chunk32/V64 body is not serving-viable: the
  top GPU event is `Fp32GdnPrefillChunk32Kernel<64>` at about `500.69 ms`
  across 18 launches, and TTFT regresses badly. Do not pursue this local chunk
  body further as the production GDN prefill path.
- Current rejected broad BF16 activation diagnostic:
  `results/qwen08_jax_bf16_activation_longprefill_probe.json`, `108.76 tok/s`,
  `0.935x` the stored vLLM reference, but not correct: the `len_1536` row
  diverged at generated token `0` (`279` vs reference `1719`). This confirms
  that BF16 activations may expose enough speed for upstream kernels, but a
  whole-model dtype change violates the current correctness gate.
- Current default-off prepared-FLA GDN prefill reference route:
  `results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.json`,
  `89.37 tok/s`, `0.768x` the stored vLLM reference, exact generated-token
  parity on the one-repeat integrated route, and
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32`. This is not
  speed-claim-ready and is below the accepted/scoped default, but it proves the
  prepared-body FLA ABI can be exercised through the server path.
- Current rejected prepared-FLA CUDA chunk32 route:
  `results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.json`,
  `63.82 tok/s`, `0.548x` the stored vLLM reference, exact generated-token
  parity on the one-repeat integrated route, and
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_fla_chunk32_fp32`. The top GPU
  event is `Fp32GdnPrefillChunk32Kernel<32, true>` at about `452.44 ms` across
  18 launches. This is rejected for promotion; a direct layout-adapted port of
  the old local chunk body is still not the right GDN prefill kernel schedule.
- Current rejected narrow BF16 GDN prefill activation diagnostic:
  `results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.json`,
  `89.74 tok/s`, `0.771x` the stored vLLM reference, exact generated-token
  parity on the one-repeat integrated route, and
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32` with
  `NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE=bf16`. The 500-token top-5 guardrail
  artifact
  `results/qwen08_jax_bf16_prefillact_long_decode_top5_compare_20260527.json`
  failed promotion: top-1 exact `500/500`, ordered top-5 `498/500`, top-5 set
  `499/500`, and `max_hf_topk_id_logit_diff=0.010448455810546875`.
- Current BF16 packed-decode boundary scaffold result:
  `results/qwen08_jax_packed_decode_bf16qkv_fp32out_long_decode_top5_compare_20260527.json`
  keeps exact generated-token, top-1, and top-5 identity (`500/500` for all),
  with `max_hf_topk_id_logit_diff=2.6702880859375e-05`. This misses the
  default FP32-activation gate of `<=2e-5` but passes the BF16 external-kernel
  boundary threshold (`<=1e-4`). Keep it as a default-off BF16-lane scaffold,
  not a promoted speed path, until it also wins a benchmark target.
- Current rejected local CUDA BF16 packed-decode route:
  `results/qwen08_jax_packed_decode_cuda_bf16qkv_fp32out_long_decode_top5_compare_20260527.json`
  passes focused CUDA parity but fails the full 500-token HF top-5 guardrail:
  generated-token/top-1 exact `499/500`, ordered top-5 `491/500`, top-5 set
  `499/500`, and `max_hf_topk_id_logit_diff=0.02606964111328125`. Do not run
  speed claims for this variant; use it only as a diagnostic replay source.
- Current external GDN kernel feasibility probe:
  `results/external_gdn_kernel_probe_20260527_sm86.json` records installed
  FlashInfer `0.6.11.post3`, `jax-tvm-ffi 0.1.3`, torch `2.12.0`, Triton
  `3.7.0`, GPU `NVIDIA A10G` with compute capability `8.6`, and all expected
  local vLLM/FLA source paths present. It marks direct FlashInfer GDN prefill
  blocked by the SM90/SM100 requirement; use vLLM/FLA as a port/fork reference
  behind the existing post-conv or packed-decode ABI.
- Current vLLM/FLA Torch-side GDN microprobe:
  `results/vllm_fla_gdn_probe_20260527_sm86.json` records vLLM `0.21.0`, torch
  `2.11.0+cu130`, Triton `3.6.0`, and GPU `NVIDIA A10G` with compute
  capability `8.6`. It is not a JAX serving speed claim. It shows the actual
  vLLM BF16 FLA prefill body is fast on SM86: `prep_only` p50 `0.40 ms`,
  `chunk_only` p50 `1.31 ms`, and combined prep+chunk p50 `1.45 ms` for
  `[512,1024,1536,2048]` (`5120` total tokens), with FP32 gate/beta/state and
  V,K state. Packed decode p50 is `0.116 ms` at batch 1, `0.116 ms` at batch 4,
  `0.119 ms` at batch 8, and `0.161 ms` at batch 16. This supports a real
  FLA-schedule port/fork or JAX-facing external-kernel route; it does not
  promote BF16 activations by itself. Independent recurrent reference checks in
  the artifact record packed-decode exact BF16 output match with FP32 state max
  abs `1.19e-7`, and ragged prefill output max abs `4.88e-4` with final-state
  max abs `4.45e-3`. A sidecar reuse audit concludes that direct Torch/vLLM
  Triton reuse from JAX is not viable on the current stack, so this remains a
  port/fork target rather than an imported runtime dependency.
- Current vLLM-inspired random-token manifest sidecar:
  `results/gpu_matrix_20260527_vllm_random_longprefill_r2.json`,
  `84.60 tok/s`, live vLLM `353.91 tok/s`, `0.239x` vLLM, exact generated-token
  match over two repeats, and speed-claim-ready in the matrix sense. This
  sidecar is a broader comparability lane, not the frozen exact-token goal gate.
  Scheduler diagnostics show 32 prefill waves at max 4 active sequences, 480
  decode steps, about `17.74 s` total prefill-step time, and about `6.30 s`
  total decode-step time, so the sidecar gap is heavily
  TTFT/static-concurrency driven.
- Active target: fastest accepted kernel-backed non-speculative serving at
  `>=0.9x` vLLM on the same benchmark discipline, with MTP remaining
  diagnostic-only.

If paged-layout kernels remain hard to integrate cleanly, insert a smaller
kernel-integration probe before more complicated paging work. GEMM-shaped or
simple fused kernels may be used to prove optional dependency loading, ABI
registration, profiling, fallback behavior, and correctness gates, but they
should not be promoted as speed work unless the integrated server benchmark
improves.

## Context

The current validated GPU baseline is the JAX paged serving path for
`Qwen/Qwen3.5-0.8B`. The optimization log shows the accepted Entry 045 baseline
at `367.80 tok/s`, `TTFT p50 289.98 ms`, and `ITL p50 13.14 ms`, compared with
the tracked vLLM async baseline of `864.18 tok/s`. JAX is currently about
`0.426x` vLLM on the tracked hetero8 workload.

The target model is structurally unusual: `Qwen3.5-0.8B` has 24 layers arranged
as `6 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`, so most
layers are Gated DeltaNet / linear-attention layers rather than ordinary
full-attention layers. The model card lists GDN with 16 QK/V heads, head dim
128, full/gated attention with 8 Q heads, 2 KV heads, head dim 256, tied LM
output, and MTP training.

The current lesson from the logbook is that source-level JAX rewrites are mostly
exhausted. Several changes preserved exact generated-token correctness but
regressed integrated performance. The next serious wins should come from a
cleaner benchmark/docs surface and backend-owned kernels, not more ad hoc JAX
reshaping.

## Phase 0 - Repository And Docs Cleanup

### Goal

Make the current state unambiguous before adding new kernels. The agent should
not start by changing kernel code.

### Tasks

Create these files:

- ~~`docs/current_gpu_baseline.md`~~
- ~~`docs/rejected_optimization_index.md`~~
- ~~`docs/kernel_roadmap.md`~~
- ~~`benchmarks/configs/gpu_paged_default.json`~~
- ~~`benchmarks/configs/gpu_paged_fast_optin.json`~~
- ~~`benchmarks/configs/gpu_mtp_diagnostics.json`~~

### `docs/current_gpu_baseline.md`

Include:

```text
Current accepted default:
- model: Qwen/Qwen3.5-0.8B
- hardware: exact GPU name from benchmark artifact
- dtype contract: BF16 checkpoint weights, FP32 activation math
- platform: JAX CUDA
- baseline entry: Entry 045
- benchmark: hetero8 input lengths [64,128,192,256,320,384,448,512], output length 32
- metrics:
  - throughput: 367.80 tok/s
  - TTFT p50: 289.98 ms
  - ITL p50: 13.14 ms
  - ITL p95: 13.59 ms
- vLLM comparison:
  - vLLM async: 864.18 tok/s
  - JAX/vLLM: 0.426x
```

Split configs into two profiles:

```text
gpu_paged_default:
- fastest accepted default flags
- no experimental compact-prefill flags unless already default
- no MTP
- no FFI/custom kernels

gpu_paged_fast_optin:
- accepted opt-in compact prefill/layout flags
- still no MTP
- still correctness-gated
```

### `docs/rejected_optimization_index.md`

Summarize every rejected experiment in a compact table:

```text
| Entry | Experiment | Correct? | Perf result | Decision | Lesson |
|---|---|---:|---:|---|---|
```

Must include at least:

```text
- width-1 GDN scan bypass
- decode slot-mapping reuse
- Pallas paged decode attention
- Pallas LM-head argmax
- GDN chunk-size 16
- head-major paged attention prefill
- flat KV cache allocation
- packed full-attention K/V projection
- skip-unused hidden return norm
- static chunk-major GDN prefill
- default MTP1 serving
```

For MTP, explicitly record that the log rejects default MTP1 serving because the
run was unusably slower and output tokens remained exact only because all
speculative drafts were rejected.

### `docs/kernel_roadmap.md`

Use the kernel priority list from Phase 2 below. Each kernel entry must include:

```text
- motivation
- proposed source/reference implementation
- JAX-facing ABI
- fallback path
- correctness gate
- performance gate
- do-not-merge conditions
```

## Phase 1 - Benchmark And Correctness Discipline

### Goal

Before implementing kernels, make benchmark comparisons reproducible and hard to
game.

### Tasks

Add a benchmark matrix runner:

- ~~`benchmarks/run_gpu_matrix.py`~~

It should run:

```text
1. gpu_paged_default
2. gpu_paged_fast_optin
3. gpu_mtp_diagnostics
```

Across at least:

```text
- hetero8: [64,128,192,256,320,384,448,512] x 32 output
- short_32_128: [32,64,96,128] x 32 output
- long_prefill_512_2048: [512,1024,1536,2048] x 16 output
- decode_heavy_128x128: [128] x 128 output
```

Output one summary file:

```text
results/gpu_matrix_<timestamp>.json
```

Required metrics:

```text
- total tok/s
- TTFT p50/p95
- ITL p50/p95
- exact generated-token match vs baseline
- PjRt Execute total/count
- command_buffer::execute total/count
- command_buffer::update total/count
- forward_step_token_ids_jit total/count
- first forward_step_token_ids_jit
- gather total
- transpose total
- MemcpyD2D total
- tolist / np.asarray sync attribution
```

### Acceptance Rule For Any Performance PR

A PR can only claim a speedup if:

```text
- exact generated-token match holds
- the same benchmark config is used
- at least 2 repeat runs are reported
- median result improves
- no major TTFT/ITL regression is hidden by throughput
- profile bucket movement is explained
```

This matters because the log already shows experiments where a named trace bucket
improved but integrated throughput regressed. For example, smaller GDN chunking
improved one reduce bucket but worsened first prefill, GEMM/module execution, and
end-to-end throughput.

### Status - 2026-05-26

- Matrix configs and `benchmarks/run_gpu_matrix.py` are in place. The runner
  uses stored local references when available. When no workload-specific local
  artifact exists, it can run live vLLM for the vLLM reference and can use the
  live `gpu_paged_default` artifact as the JAX default reference for subsequent
  comparisons.
- The first `long_prefill_512_2048` slice had no stored local vLLM reference, so
  it ran live vLLM and stored
  `results/gpu_matrix_runs/20260526_104818/references/vllm_long_prefill_512_2048.json`.
- One-repeat results: vLLM async `116.37 tok/s`, JAX `gpu_paged_default`
  `78.02 tok/s` (`0.670x` vLLM), and JAX `gpu_paged_fast_optin`
  `78.27 tok/s` (`0.673x` vLLM). Fast opt-in matched the live JAX default
  exactly for all four rows; default was the baseline capture for this slice.
- This is not a speed-claim artifact because it is one repeat and the default
  row is not checked against a same-workload token reference. It is useful as
  current target evidence: long-prefill is closer to vLLM than hetero8, but
  still below the `0.75x` next-goal target.
- The JAX gap remains visible in both TTFT and ITL. The fast-optin profile shows
  first `forward_step_token_ids_jit=237.75 ms`,
  `PjRtCApiLoadedExecutable::Execute=290.66 ms / 140`,
  `command_buffer::execute=228.13 ms / 1936`, and `np.asarray(jax.Array)=
  429.63 ms / 16`.
- ~~A two-repeat `hetero8,long_prefill_512_2048` matrix attempt could not
  produce benchmark evidence because the sandboxed command session had no
  visible NVIDIA device nodes.~~ The GPU was reachable outside that sandbox, and
  the long-prefill goal-target slice has now been rerun with GPU access.
- The matrix configs now carry workload-specific stored JAX and vLLM reference
  paths for `hetero8` and `long_prefill_512_2048`. A dry run verified that both
  repeats of `gpu_paged_default` and `gpu_paged_fast_optin` will compare against
  stored JAX references from repeat one instead of leaving the default
  long-prefill baseline unchecked. Stored references are preferred over
  generated same-run defaults so exact-token gates use stable baselines.
- Matrix summaries now include an `acceptance` section that makes the plan's
  speed-claim evidence check explicit: successful subprocesses, minimum
  repeats, exact correctness, JAX/vLLM performance presence, TTFT/ITL p50/p95
  latency, first `forward_step_token_ids_jit`, profile counters, and the
  active vLLM ratio target. The active target is now `0.9x`; it was `0.75x`
  for the closed no-kernel milestone. All configured profile-counter buckets
  must be present for every repeat. The runner validates the summary shape
  before writing it.
  Human explanation of profile bucket movement still belongs in the logbook.
- `benchmarks/run_gpu_matrix.py --require-speed-claim-ready` can be used for
  the final benchmark command. It still writes the summary, then exits nonzero
  if any selected workload/config is not speed-claim-ready or misses the active
  vLLM target ratio.
- Matrix summaries now record the final thread-goal target explicitly:
  `long_prefill_512_2048/gpu_paged_default` must be speed-claim-ready and reach
  at least `0.9x` vLLM. `--require-goal-target-ready` writes the summary and
  exits nonzero unless that specific long heterogeneous non-speculative target
  is present, correctness-gated, profile-covered, and at/above the target
  throughput ratio.
- `--goal-target-only` selects exactly `long_prefill_512_2048/gpu_paged_default`
  regardless of the generic `--configs`/`--workloads` defaults. The intended
  final non-speculative benchmark command should combine it with
  `--require-goal-target-ready` and, when stable local references are required,
  `--require-stored-references`.
- Comparison rows now include the concrete throughput target and remaining gap:
  `target_tokens_per_second`, `tokens_per_second_gap_to_target`, and
  `required_jax_speedup_to_target`. With the active `0.9x` target and the
  stored vLLM reference at `116.37 tok/s`, the target is about
  `104.73 tok/s`; the accepted no-kernel default at `90.50 tok/s` is still
  about `14.23 tok/s` short before kernel work.
- `--require-stored-references` can be used before benchmark launch to fail
  fast when selected workloads/configs lack stored JAX or vLLM references.
- Focused tests also verify that all GPU matrix configs have valid stored JAX
  and vLLM references for `hetero8` and `long_prefill_512_2048`, so the next
  GPU-visible two-repeat run should not silently fall back to unchecked
  baselines for the tracked workloads.
- The live JAX fallback is now explicit for uncovered workloads. If a selected
  config lacks a stored JAX reference and `gpu_paged_default` is not the first
  selected config, the runner first captures a live `gpu_paged_default`
  artifact under the run's `references/` directory and uses it as the
  correctness baseline for the selected configs. Dry runs record that planned
  reference command without importing JAX.
- The same focused suite verifies command construction and runtime environment
  defaults for matrix runs: workload overrides, `--reference-json`, warmup,
  profile, `JAX_PLATFORMS=cuda`, and cache/temp roots under the configured
  `/mountpoint` runtime root.
- The matrix runner now selects the JAX subprocess interpreter explicitly with
  `--jax-python` or `NANO_VLLM_JAX_PYTHON`, records it in the summary, and
  checks package visibility with `importlib.util.find_spec("jax")` before real
  runs. This avoids accidentally launching live GPU benchmarks with the base
  shell Python when that interpreter has no JAX installed; the preflight does
  not import JAX or select a CPU backend.
- Matrix summaries now aggregate profile-counter medians per selected
  workload/config and add JAX-reference comparison fields:
  throughput/TTFT/ITL deltas versus the selected JAX correctness reference plus
  per-bucket profile deltas for the required trace needles. This does not
  replace the required human logbook explanation, but it makes profile movement
  concrete in the benchmark artifact instead of leaving it to ad hoc manual
  inspection.
- `benchmarks/summarize_gpu_matrix.py` renders a concise Markdown report from a
  matrix summary, including final-target status, acceptance failures, JAX/vLLM
  and JAX/reference throughput rows, and the largest profile-bucket deltas
  versus the selected JAX reference. Use it after real GPU matrix runs to seed
  the required logbook explanation without hand-reading the raw JSON.
- `benchmarks/summarize_profile_trace.py` renders a raw Chrome/Perfetto trace
  event summary when the matrix-level buckets are too coarse. Use it to inspect
  top GPU/CPU/all-scope events, selected substring totals, and trace-provided
  `hlo_module`/`hlo_op`/kernel-detail rows without hand-parsing
  `*.trace.json.gz`.
- `benchmarks/benchmark_jax_server_trace.py` now records scoped GPU/CPU profile
  ranges and top raw events directly in benchmark artifacts. Matrix summaries
  preserve those fields per repeat, so future speed claims can cite the
  artifact without rerunning a separate trace parser.
- `benchmarks/summarize_gpu_matrix.py` renders the scoped GPU/CPU top events
  when a matrix artifact contains them. Older matrix JSONs predate this field
  and will omit the section; new profiled runs should expose it in the default
  Markdown report.
- Matrix aggregation now computes scoped GPU/CPU profile-range medians across
  repeats, and the Markdown report can fall back to per-repeat scoped ranges for
  older artifacts. This gives future speed claims a stable scoped median table
  plus per-repeat top-event detail.
- Matrix comparisons now compute scoped GPU/CPU profile deltas versus the JAX
  reference when both artifacts contain scoped profile fields. If the selected
  reference predates scoped fields, reports explicitly say scoped deltas are
  unavailable instead of implying no movement.
- `benchmarks/run_gpu_matrix.py` now writes that Markdown report by default next
  to the summary JSON, or to `--report-md` when supplied. `--no-report-md`
  keeps report generation opt-out for controlled runs.
- The Markdown report includes a logbook-entry template with the target
  artifact paths, JAX/vLLM and JAX/reference ratios, TTFT/ITL deltas, and the
  largest profile bucket deltas. It is a copyable starting point for the
  required profile-movement explanation; the final interpretation and
  keep/reject/follow-up decision still must be written by the reviewer.
- Real two-repeat goal-target run:
  `results/gpu_matrix_20260526_131031.json` and
  `results/gpu_matrix_20260526_131031.md`. The run is speed-claim-ready in the
  matrix-gate sense: both repeats succeeded, exact generated-token match held,
  and all required latency/profile buckets were present. It still fails the
  final speed target: JAX median throughput is `77.17 tok/s`, vLLM reference is
  `116.37 tok/s`, JAX/vLLM is `0.663x`, and the `0.75x` target requires
  `87.28 tok/s`, leaving a `10.11 tok/s` gap or about `1.13x` required JAX
  speedup.
- Accepted scheduler host-handoff optimization:
  `results/gpu_matrix_20260526_132210.json` and
  `results/gpu_matrix_20260526_132210.md`. The scheduler now batches the six
  `int32` metadata arrays for each `ScheduledBatch` through one
  `jax.device_put` of NumPy host arrays instead of separate `jnp.array` calls.
  The two-repeat goal-target matrix remains speed-claim-ready and improves to
  `85.98 tok/s`, `0.739x` vLLM, with exact generated-token parity. The final
  target is still not met: it needs `87.28 tok/s`, leaving a `1.30 tok/s` gap
  or about `1.015x` required JAX speedup.
- ~~Updated parity ladder: keep pushing the pure-JAX/no-custom-kernel default
  across the `0.75x` vLLM gate first.~~ This staged gate is now closed on the
  deterministic long-prefill target. The active runner gate is `0.9x` vLLM for
  correctness-gated kernel-backed non-speculative serving. MTP remains
  diagnostic-only until the non-speculative path has met its staged targets.
- vLLM benchmark audit: upstream vLLM now exposes `vllm bench serve` for
  online/server throughput and `vllm bench throughput` for offline engine
  throughput. Its benchmark datasets include ShareGPT, random, custom JSONL, HF
  datasets, prefix repetition, BurstGPT, Spec Bench, and SPEED-Bench. The repo's
  current vLLM comparison is a local exact-token harness, not the upstream
  benchmark CLI, and the current prompt generator repeats tiny tokenized seed
  prompts to force exact input lengths. Keep this harness for deterministic
  correctness and shape comparisons, but label it as `tokenized_seed_repeat` or
  `shape_synthetic`.
- Add a vLLM-comparable sidecar benchmark lane before making broader throughput
  claims. The first sidecar should use either upstream vLLM random sampling
  semantics or a shared custom JSONL manifest such as one row per request with
  `request_id`, `prompt_token_ids`, `prompt_len`, and `output_len`. Artifacts
  should record `prompt_source`, `dataset_name`, `num_prompts`, `seed`,
  `prompt_manifest_jsonl`, `prompt_manifest_sha256`, `total_input_tokens`,
  `total_output_tokens`, `request_throughput`, `output_token_throughput`, and
  `total_token_throughput`. A reasonable first long-prefill sidecar is
  128 prompts, random input length around 1280, output length 16,
  input range ratio 0.6, output range ratio 0.0, request rate `inf`, and
  `ignore_eos=true`.
- Sidecar harness support added:
  ~~JAX and vLLM benchmark artifacts record `prompt_source`, `dataset_name`,
  `num_prompts`, `seed`, prompt manifest path/hash, request throughput,
  output-token throughput, and total-token throughput.~~ The opt-in
  `vllm_random_longprefill` matrix workload now wires a deterministic
  vLLM-inspired random-token manifest with 128 prompts, input length centered at
  1280 with range ratio 0.6, fixed output length 16, and `sidecar_only`
  acceptance scope. This local generator is not upstream
  `vllm bench --dataset-name random` semantics; comparability comes from the
  shared prompt-token JSONL manifest and matching manifest SHA. This does not
  replace the exact-token
  `long_prefill_512_2048/gpu_paged_default` gate.
- Sidecar harness verification: `py_compile` passes for the touched benchmark
  scripts; `tests/test_gpu_matrix_runner.py` and
  `tests/test_gpu_matrix_summary_report.py` pass with 43 tests; a dry run of
  `benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads
  vllm_random_longprefill --repeats 2` validates command construction, summary
  schema, the opt-in workload metadata, and live JAX default reference planning
  for sidecar correctness comparison.
- Post-sidecar goal-target verification:
  `results/gpu_matrix_20260526_135134.json` and
  `results/gpu_matrix_20260526_135134.md`. The no-kernel default remains
  speed-claim-ready with exact generated-token parity, but still misses the
  final target: `85.21 tok/s`, `0.732x` vLLM, target `87.28 tok/s`, gap
  `2.07 tok/s`, required JAX speedup `1.024x`. The run confirms the benchmark
  harness change did not itself close the remaining gap.
- Accepted benchmark trace text-materialization cleanup:
  `results/gpu_matrix_20260526_135552.json` and
  `results/gpu_matrix_20260526_135552.md`. The trace benchmark now disables
  per-token and final text detokenization inside the timed `generate_with_trace`
  path while preserving token IDs and default engine text behavior for normal
  callers. The two-repeat goal-target matrix stays speed-claim-ready and exact,
  improving to `86.26 tok/s`, `0.741x` vLLM. The final `0.75x` target is still
  not met: target `87.28 tok/s`, gap `1.02 tok/s`, required JAX speedup
  `1.012x`.
- Rejected local micro-probe: switching the block-manager first-free allocation
  path to `popleft()` and skipping non-speculative MTP admission bookkeeping did
  not beat the text-cleanup run in the integrated goal-target matrix, so those
  changes were not kept.
- Rejected local workload-envelope probe: adding a `1536` prefill bucket was
  exact but slower (`85.29 tok/s`) and was not kept.
- Local envelope probe result: lowering only `max_blocks_per_seq` to `129` was
  exact but marginal (`86.43 tok/s`). The accepted form pairs
  `max_blocks_per_seq=129` with `num_kvcache_blocks=384`, which better matches
  the long-target request envelope while preserving correctness.
- Accepted no-kernel `0.75x` target closure:
  `results/gpu_matrix_20260526_141130.json` and
  `results/gpu_matrix_20260526_141130.md`. The change keeps the benchmark trace
  text-materialization cleanup, adds a direct-row hybrid-state fast path for the
  common static-slot case, and tightens the long-target KV/block-table envelope
  to `num_kvcache_blocks=384` and `max_blocks_per_seq=129`. The two-repeat
  goal-target matrix is speed-claim-ready with exact generated-token parity,
  JAX median `88.91 tok/s`, vLLM `116.37 tok/s`, and JAX/vLLM `0.764x`, clearing
  the `0.75x` gate.
- Updated parity ladder: the no-kernel/non-speculative default has now cleared
  the staged `0.75x` gate. The next active target is `0.9x` vLLM for
  correctness-gated kernel-backed non-speculative serving on the same
  long-prefill exact-token benchmark discipline. MTP remains diagnostic-only
  until that kernel target is met or explicitly reprioritized.
- Sidecar smoke validation:
  `results/gpu_matrix_20260526_143600.json` and
  `results/gpu_matrix_20260526_143600.md`. A 16-prompt
  `vllm_random_longprefill_smoke` lane now validates the vLLM-random prompt
  manifest path quickly before the heavier 128-prompt sidecar. It is
  speed-claim-ready in the matrix sense and exact against its live JAX default
  reference, but it is not close to vLLM: JAX `82.57 tok/s`, vLLM
  `299.31 tok/s`, JAX/vLLM `0.276x`. This confirms that the exact-token
  long-prefill gate must not be described as a broad vLLM-random serving parity
  claim.
- Multi-wave scheduler fix: the scheduler no longer admits new waiting prompts
  while the active running set already occupies `max_num_seqs`. This fixes the
  sidecar's hybrid-state slot exhaustion class when many prompts are queued with
  `max_num_seqs=4`.
- Goal-target revalidation after the sidecar scheduler fix:
  `results/gpu_matrix_20260526_144104.json` and
  `results/gpu_matrix_20260526_144104.md`. The exact long-prefill target remains
  speed-claim-ready and exact, improving to JAX `90.50 tok/s`, vLLM
  `116.37 tok/s`, JAX/vLLM `0.778x`.
- Executable goal gate update: `benchmarks/run_gpu_matrix.py` now uses
  `TARGET_VLLM_RATIO=0.9`. This aligns `--require-goal-target-ready`,
  target-token/s math, and acceptance summaries with the kernel-phase target in
  this plan. The earlier `0.75x` threshold remains historical evidence for the
  closed no-kernel milestone, not the current acceptance bar.
- Scheduler diagnostics added to matrix summaries: `benchmarks/run_gpu_matrix.py`
  now derives unique JAX scheduler steps from token events and
  `benchmarks/summarize_gpu_matrix.py` reports median prefill/decode step
  counts, max active prefill sequences, max step tokens, and total prefill/decode
  step seconds. Existing artifacts show the 16-prompt sidecar smoke ran 4
  prefill waves plus 60 decode steps, while the 128-prompt sidecar ran 32
  prefill waves plus 480 decode steps. Attempts to raise sidecar
  `max_num_seqs` to 5, 6, or 8 OOMed during warmup with static allocations of
  about `10.05 GiB`, `10.66 GiB`, and `12.57 GiB`; no result JSONs were
  produced. Treat this as evidence for a static-shape concurrency/TTFT problem
  before assuming a narrow kernel swap will close the vLLM-random sidecar gap.
- Rejected split prefill/decode batch-bucket scheduler experiment:
  `results/gpu_matrix_20260527_split_decode8_smoke.json` and
  `results/gpu_matrix_20260527_split_decode8_smoke.md`. The attempt capped
  prefill at 4 active rows while allowing decode to use an 8-row batch. It
  reduced sidecar smoke decode steps from 60 to 45 but regressed integrated
  output throughput to `68.25 tok/s`, `0.228x` vLLM, with `ITL p50 27.74 ms`
  and worse `gather`/`PjRt Execute` profile buckets. Do not keep split
  prefill/decode physical batch buckets as a default path without a new profile
  reason and a sidecar throughput win.
- Current elevated goal-target revalidation:
  `results/gpu_matrix_20260527_current_goal_target.json` and
  `results/gpu_matrix_20260527_current_goal_target.md`. The default non-
  speculative path remains speed-claim-ready and exact over two repeats, with
  JAX `90.87 tok/s`, stored vLLM `116.37 tok/s`, and JAX/vLLM `0.781x`. This is
  useful baseline evidence but not goal completion; the active `0.9x` target
  still requires `104.74 tok/s`, leaving a `13.86 tok/s` gap.
- Scoped-profile goal-target revalidation:
  `results/gpu_matrix_20260527_scoped_profile_target.json` and
  `results/gpu_matrix_20260527_scoped_profile_target.md`. The default non-
  speculative path remains speed-claim-ready and exact over two repeats, with
  JAX `90.81 tok/s`, stored vLLM `116.37 tok/s`, JAX/vLLM `0.780x`, target
  `104.74 tok/s`, and gap `13.93 tok/s`. Scheduler diagnostics remain one
  prefill step plus 15 decode steps; the scoped report shows the top GPU events
  are GEMM/CUTLASS buckets, led by `gemm_fusion_dot_general_744` at about
  `57.45 ms` per repeat. This artifact verifies the new scoped profile-event
  reporting path and does not change the current speed target status.
- Packed-GDN CUDA FP32 target rejection:
  `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.json` and
  `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.md`. The opt-in
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=cuda_fp32` route is exact over two
  repeats and speed-claim-ready, but reaches only `88.41 tok/s`, `0.760x` vLLM,
  with a `16.33 tok/s` target gap. It reduces some named profile buckets versus
  the older configured JAX reference but regresses against the current
  accepted/scoped default, so it stays default-off.
- Greedy decode-burst rejection:
  `results/gpu_matrix_20260527_decode_burst_target.json` and
  `results/gpu_matrix_20260527_decode_burst_target.md`. The opt-in
  `NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS=16` route is exact over two repeats,
  reduces token readback counts from 16 to 1, and runs only one decode scheduler
  step, but throughput collapses to `17.46 tok/s`, `0.150x` vLLM. The decode
  step takes about `3.13 s`, `gather` and `transpose` counts rise sharply, and
  required profile counters are missing. A one-repeat burst-2 probe was worse
  at `5.57 tok/s`. This rejects source-level full-model decode scans as the
  host-sync solution.
- Device-token-carry fastest-run marker:
  `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.json` and
  `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.md`. The
  opt-in `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1` route keeps greedy token vectors
  on device between scheduled decode calls, now stores deferred completion
  tokens as vector references instead of per-row scalar slices, and materializes
  token IDs after fixed-length `ignore_eos` generation. It is exact over two
  repeats and raises the local long-prefill target to `95.14 tok/s`, `0.818x`
  vLLM, but it is not an accepted default because it changes streaming timing
  semantics, is not speed-claim-ready under the current profile-counter gate,
  and still leaves a `9.59 tok/s` gap to the `0.9x` target.
- Current raw GPU trace summary:
  `results/profile_trace_20260527_current_goal_target_gpu.json` and
  `results/profile_trace_20260527_current_goal_target_gpu.md`. Across both
  repeats, the GPU-scope profile is dominated by GEMM/fusion work:
  `gemm_fusion` about `243.6 ms`, `cutlass` about `64.5 ms`, `transpose` about
  `45.1 ms`, `input_reduce_fusion` about `34.7 ms`, and
  `wrapped_concatenate` about `17.4 ms`. This supports the current caution:
  packed GDN decode is the smallest FP32 vLLM/FLA-shaped kernel boundary, but
  the stored traces do not prove it can close the full `0.9x` gap alone.
- Full vLLM-random sidecar revalidation:
  `results/gpu_matrix_20260527_vllm_random_longprefill_r2.json` and
  `results/gpu_matrix_20260527_vllm_random_longprefill_r2.md`. The 128-prompt
  sidecar is now two-repeat and exact against a live JAX default reference, with
  JAX `84.60 tok/s`, live vLLM `353.91 tok/s`, and JAX/vLLM `0.239x`. It is
  speed-claim-ready in the matrix sense but not close to vLLM. TTFT p50 is
  `12385.99 ms` for JAX versus `2953.88 ms` for vLLM, while JAX ITL p50 is
  `13.39 ms`. Scheduler diagnostics again show 32 prefill waves at max 4 active
  sequences plus 480 decode steps, so this remains evidence for a static-shape
  concurrency/TTFT gap separate from the frozen exact-token long-prefill goal
  gate.
- Prompt provenance is now explicit in matrix metrics and Markdown reports:
  future matrix summaries preserve the artifact `run_config` prompt source,
  dataset, seed, random length settings, and prompt-manifest SHA. Reports show a
  `Prompt Provenance` table with current-vs-vLLM manifest hashes, so the
  vLLM-random sidecar can prove when JAX and vLLM used the same token IDs. This
  lane remains local-harness `vllm_random`, not an upstream `vllm bench`
  dataset run. `vllm_random` stored-reference matching now requires both a
  prompt-manifest path and SHA so older/random-only metadata cannot be mistaken
  for shared-token evidence.
- Profile dashboard status: `tools/profile_dashboard.py` is the visual
  leaderboard for matrix/profile artifacts. Direct Perfetto iframe embedding is
  not the plan because the official UI is designed around HTTPS trace URLs or
  opening `ui.perfetto.dev` and passing trace bytes with `postMessage`; browser
  origin and popup rules make iframe/local embedding fragile. Keep in-dashboard
  charts for summary comparison, use the `Load in Perfetto` handoff for local
  `*.trace.json.gz` files, and keep raw trace download/open-file as the
  fallback.

## Phase 2 - Kernel Roadmap

### Kernel Priority Table

```text
P0:
1. kv_append_paged_nhd
2. paged_decode_attention_gqa_nhd

P1:
3. gdn_recurrent_decode_step
4. gdn_segmented_prefill_chunk32

P2:
5. paged_prefill_attention_gqa_nhd
6. qk_norm_rope_kv_append_fused

P3:
7. topk_logits / sampling / logprob
8. silu_and_mul / RMSNorm smoke-test kernels only if needed for FFI validation
```

The important ordering is: first own KV layout and decode attention, then attack
GDN decode/prefill using external kernel references. Do not start with MTP or
top-k.

### Current Execution Plan - FA/FLA Integration

As of 2026-06-04, the next speed path is the FA/FLA kernel integration plan:

1. Make full-attention and GDN kernel policy first-class config, not loose
   benchmark environment. Artifacts must report the selected FA KV append,
   FA decode, FA prefill, GDN prefill, and GDN decode implementations.
2. Promote the existing paged full-attention decode boundary into a typed route:
   `full_attention.decode_impl=triton_paged`. This must consume the current
   NHD paged cache directly and fail loudly when the shape/dtype contract is not
   supported, instead of silently falling back to JAX.
3. Keep `full_attention.kv_append_impl=reference` until a matching FlashInfer or
   Triton append path is paired with the paged decode reader. Standalone append
   kernels have already regressed integrated serving.
4. Use the selected GDN route as a component, but do not expect the narrow GDN
   recurrent kernel alone to close the random/decode gap. The useful FLA work is
   a broader GDN decode boundary that owns conv, gate/beta math, q/k norm,
   recurrent-state read/update/write, and surrounding layout.
5. Re-run focused parity first, then integrated `decode_heavy_128x128`,
   `hetero8`, `long_prefill_512_2048`, and the random sidecar. Promotion still
   requires exact generated-token parity where the reference is exact, zero
   measured-phase JIT growth after generic warmup, and a real integrated
   throughput improvement against the accepted JAX baseline and vLLM.

Status after the first execution pass: the config route and focused parity
tests are implemented, but the narrow FA decode routes are rejected for
promotion. `full_attention.decode_impl=triton_paged` was exact but slower than
the current static route on `decode_heavy_128x128`; fused append+decode and B1
packed-QKV probes also regressed. Continue with broader decode graph work
rather than retrying standalone FA attention replacement.

## P0.1 - `kv_append_paged_nhd`

### Motivation

This defines the full-attention KV-cache physical layout. Without this, every
later attention kernel risks layout conversion overhead.

### Reference

Use FlashInfer's `append_paged_kv_cache` as the first target. It appends ragged
K/V tensors into a paged KV cache and supports `NHD` layout with cache shape
`[max_num_pages, page_size, num_kv_heads, head_dim]`, or a combined 5-D cache
`[max_num_pages, 2, page_size, num_kv_heads, head_dim]`.

Also study vLLM's `reshape_and_cache_flash_kernel`; it writes token K/V into
paged cache using `slot_mapping`, supports NHD/HND-style layout, and is close to
the desired CUDA contract.

### Proposed ABI

```python
kv_append_paged_nhd(
    append_key,        # [nnz_tokens, num_kv_heads, head_dim]
    append_value,      # [nnz_tokens, num_kv_heads, head_dim]
    batch_indices,     # [nnz_tokens]
    positions,         # [nnz_tokens]
    k_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    kv_indices,        # [total_pages]
    kv_indptr,         # [batch + 1]
    kv_last_page_len,  # [batch]
) -> updated_cache
```

For this model's full-attention layers:

```text
num_q_heads = 8
num_kv_heads = 2
head_dim = 256
initial page_size = 16
secondary page_size sweep = 32
```

### Implementation Path

```text
1. Add full-attention KV cache allocation in NHD layout.
2. Keep existing JAX cache path as fallback.
3. Add FlashInfer/JAX FFI wrapper.
4. Route only full-attention layers through the new cache append.
5. Run exact-token parity.
```

FlashInfer documents calling its GPU kernels from JAX through `jax-tvm-ffi`;
that should be the first integration route rather than writing CUDA from
scratch.

### Acceptance Gate

```text
- exact generated-token match vs Entry 045 reference
- no TTFT regression
- no ITL p50/p95 regression
- lower or equal cache write/gather/transpose overhead
- no new host sync around FFI workspace setup
```

### Status

- Focused CUDA FFI proof passed on 2026-05-26 for separate NHD K/V caches,
  BF16 cache tensors, `head_dim=128` and `head_dim=256`, and a non-contiguous
  page table.
- The working JAX FFI registration uses `arg_spec=["args", "attrs.layout"]`
  and `input_output_aliases={4: 0, 5: 1}` so FlashInfer can mutate cache inputs
  while JAX receives functional cache outputs with unwritten entries preserved.
- The opt-in route `NANO_VLLM_JAX_FLASHINFER_KV_APPEND=1` updates canonical
  per-layer cache slices through FlashInfer and then returns the existing cache
  layout to the pure-JAX attention path.
- A full hetero8 server run with the opt-in route failed during warmup before
  producing a benchmark artifact because FlashInfer's append dispatcher does not
  support FP32 cache tensors. The current accepted contract keeps BF16 weights
  and FP32 activation/KV-cache tensors, so this route is rejected for default
  serving unless the dtype policy changes or a FP32 append kernel is added.
- A local CUDA/JAX FFI FP32 append smoke kernel now builds under
  `/mountpoint/.exp/.cache`, uses the same NHD append ABI and cache-output
  aliasing contract, and passes a focused CUDA parity test against the pure-JAX
  NHD append reference. This proves the FP32 custom-call toolchain but is not
  routed into serving or accepted as a speed path yet.
- Routing that local FP32 append kernel through `write_kv` behind
  `NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1` preserved exact generated-token
  correctness on hetero8, but regressed throughput to `193.62 tok/s` and ITL
  p50 to `31.43 ms`. Keep the kernel as a toolchain smoke proof; do not promote
  standalone append routing without a paired attention/layout consumer or
  integrated profile evidence that removes the added overhead.
- This is not exact-token/integrated-benchmark accepted and is not a default
  backend.

## P0.2 - `paged_decode_attention_gqa_nhd`

### Motivation

This is the closest equivalent to vLLM's core paged decode attention path.

### Reference

vLLM's PagedAttention design relies on a specialized memory layout and access
method for reading paged KV cache efficiently.

FlashInfer provides `BatchDecodeWithPagedKVCacheWrapper`, which is specifically
for batched decode attention with paged KV cache and supports `NHD` layout.

### Proposed ABI

```python
paged_decode_attention_gqa_nhd(
    q,                 # [batch, num_q_heads, head_dim]
    k_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    kv_indptr,         # [batch + 1]
    kv_indices,        # [total_pages]
    kv_last_page_len,  # [batch]
    seq_lens,          # [batch]
    softmax_scale,     # float
) -> out               # [batch, num_q_heads, head_dim]
```

### Acceptance Gate

```text
- exact generated-token match
- top-5 parity on focused decode tests
- ITL p50 improves
- ITL p95 does not regress
- PjRt Execute improves or stays flat
- command_buffer update/execute does not regress
```

### Do Not Merge If

```text
- FFI planning/setup happens inside the decode loop
- layout conversion is needed every layer
- benchmark only wins in a microbenchmark but loses integrated server throughput
```

### Status

- FlashInfer's batch decode/prefill JIT dtype maps do not include FP32 for the
  Q/KV/O attention path, and the KV dtype map is BF16/FP16/low-precision only.
- Under the current BF16-weights/FP32-activation contract, do not start a
  FlashInfer `paged_decode_attention_gqa_nhd` wrapper as the next serving path.
  The viable options are a FP32-capable custom kernel, or an explicit separate
  decision to change the KV-cache/attention dtype policy.
- A pure-JAX FP32 NHD ABI reference now exists for
  `paged_decode_attention_gqa_nhd`, with focused parity tests against the
  current decode attention path. This is an ABI/correctness target for a future
  CUDA custom-call, not an accepted performance kernel.
- A local FP32 CUDA/JAX FFI implementation now exists for the same ABI. It
  passes focused CUDA parity against the pure-JAX reference, including the
  model's full-attention `8q/2kv/head_dim=256` shape, under the same
  `jax_default_matmul_precision=highest` correctness mode used by the long
  decode top-5 harness.
- The local FP32 CUDA decode attention route is available behind
  `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1`, but the first integrated hetero8 run
  was slower than Entry 045 despite exact generated-token parity. Keep it
  default-off as a diagnostic route; do not promote it to default or fast
  opt-in.
- Pairing the local FP32 append route with the local FP32 decode attention route
  on the long-prefill target preserved exact generated-token parity, and trace
  inspection confirmed both `Fp32KvAppendKernel` and
  `Fp32PagedDecodeAttentionKernel` executed. Integrated performance regressed to
  `52.94 tok/s` (`0.455x` vLLM, below the accepted pure-JAX `90.50 tok/s`).
  This rejects the narrow paired P0 route as a serving strategy.

## P1.1 - `gdn_recurrent_decode_step`

### Motivation

This is probably the highest-leverage model-specific kernel. The model has 18
GDN layers and only 6 full-attention layers, so full-attention kernels alone will
not close the vLLM gap.

vLLM's Qwen3-Next support explicitly calls out hybrid attention with Gated
DeltaNet plus full attention, and says the implementation integrates Triton
kernels from Flash Linear Attention for Gated DeltaNet.

### Reference

Study:

```text
- vLLM Qwen3-Next implementation path
- Flash Linear Attention Gated DeltaNet kernels
- current JAX GDN recurrence implementation
```

Flash Linear Attention provides hardware-efficient building blocks for linear
attention, sparse attention, state-space models, and hybrid LLM architectures.

Concrete reference mapping:

```text
Local current path:
- nanovllm_jax/model.py::gated_deltanet_block
- nanovllm_jax/backends.py::PureJAXBackend.gated_delta_decode
- nanovllm_jax/model.py::jax_recurrent_gated_delta_rule
- pre-change tensors: q/k/v [B,H,T,D], g/beta [B,H,T], state [B,H,K,V]
- target serving tensors: q/k/v [B,H,T,D], g/beta [B,H,T],
  state [B,H,V,K]

vLLM Qwen GDN path:
- vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::QwenGatedDeltaNetAttention
- vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::ChunkGatedDeltaRule
- vllm/model_executor/layers/fla/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule_packed_decode
- packed decode tensors: mixed_qkv [B, 2*H*K + HV*V], a/b [B,HV],
  output [B,1,HV,V], state in-place through ssm_state_indices

FLA reference path:
- fla/ops/gated_delta_rule/fused_recurrent.py::fused_recurrent_gated_delta_rule
- fla/ops/gated_delta_rule/chunk.py::chunk_gated_delta_rule
- tensors: q/k [B,T,H,K], v [B,T,HV,V], g/beta [B,T,HV],
  optional cu_seqlens [N+1]
```

Layout decision: make k-last/V-first state canonical for GDN serving. The
target persistent shape is `[B,L,HV,V,K]`, matching the Qwen/vLLM/FlashInfer
direction and avoiding hot-path state transposes. The old `[B,L,HV,K,V]` JAX
layout should remain only as a short-term migration/reference aid. If a fallback
uses it internally, conversion must happen outside the decode hot path or be
removed before promotion.

### Proposed ABI

```python
gdn_recurrent_decode_step(
    q,          # [batch, gdn_heads, head_dim], fp32 activation
    k,          # [batch, gdn_heads, head_dim]
    v,          # [batch, gdn_heads, head_dim]
    beta,       # [batch, gdn_heads] or [batch, gdn_heads, 1]
    gate,       # [batch, gdn_heads] or [batch, gdn_heads, 1]
    state,      # [batch, gdn_heads, value_dim, key_dim], fp32 state
) -> (
    out,        # [batch, gdn_heads, head_dim]
    new_state,  # same shape as state
)
```

For Qwen3.5-0.8B GDN:

```text
gdn_heads = 16
head_dim = 128
state dtype = fp32 unless proven safe otherwise
```

### Implementation Path

```text
1. Write isolated reference test comparing current JAX recurrence vs proposed kernel.
2. Change hybrid recurrent-state initialization to `[B,L,HV,V,K]`.
3. Adapt `jax_recurrent_gated_delta_rule` and `jax_chunk_gated_delta_rule` to
   consume and return V,K state directly.
4. Update state-table, prefix-state, MTP, and parity tests for the new shape.
5. Integrate against Qwen 3 Next vLLM and Flash Linear Attention Gated DeltaNet
   kernels where practical, preserving their natural V,K state layout.
6. Do not promote the local width-1 CUDA GDN probe as the serving path. If a
   native custom-call is needed later, it must be a broader FLA/vLLM-shaped
   kernel boundary with an integrated serving win, not the standalone probe.
7. Keep pure-JAX recurrence as the fallback/reference, but in the same V,K
   serving layout.
8. Integrate behind a named optional backend flag such as
   NANO_VLLM_JAX_KERNEL_BACKEND=gdn_fla.
```

Pallas is still worth studying, but JAX documents Pallas as a custom-kernel
language for GPU/TPU and marks it with experimental caveats. Given the repo's
previous Pallas regressions, it should not be the first production path for GDN.

### Status

- Current serving GDN decode/prefill is still pure JAX/XLA at the backend
  boundary. `PureJAXBackend.gated_delta_decode` calls
  `jax_recurrent_gated_delta_rule`; `PureJAXBackend.gated_delta_prefill` calls
  `jax_chunk_gated_delta_rule`.
- A local FP32 CUDA/JAX FFI width-1 `gdn_recurrent_decode_step` prototype now
  passes focused parity against `jax_recurrent_gated_delta_rule`, including the
  model's `batch=2`, `gdn_heads=16`, `head_dim=128` shape.
- The local FP32 CUDA GDN decode route is available behind
  `NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE=1`, but the first integrated hetero8 run
  was slower than Entry 045 despite exact generated-token parity. After the V,K
  native decode cleanup, a one-repeat long-prefill probe also missed the active
  target: `88.07 tok/s`, `0.757x` vLLM, exact generated-token parity, below the
  accepted V,K baseline of `90.65 tok/s`. Keep it default-off as a diagnostic
  route; do not promote it to default, fast opt-in, or the next kernel
  implementation path.
- A pure-JAX vLLM-style packed decode reference exists in the neutral planned
  backend module `nanovllm_jax/kernels/gdn_fla.py` as
  `gdn_packed_decode_reference_local_state`. It accepts packed
  `mixed_qkv [B, 2*H*K + HV*V]` plus raw `a/b/A_log/dt_bias`, computes the
  same gate/beta transform as vLLM's packed decode path, and calls the V,K
  recurrent rule with `[B,HV,V,K]` state. The companion
  `gdn_packed_decode_reference_from_decay` accepts local loaded `A=exp(A_log)`
  weights so model call sites do not depend on checkpoint naming.
- External GDN reference audit confirms vLLM/FLA's natural Qwen GDN state layout
  is k-last/V-first `[*,HV,V,K]`, and the vLLM/FLA prefill path is
  BF16-activation oriented. The explicit decision is to switch persistent GDN
  state layout to V,K while preserving the repo's BF16-weight/FP32-activation
  contract unless a separate dtype decision is made.
- The latest external implementation audit found no direct FP32 drop-in for
  serving GDN: vLLM/FLA prefill is BF16-oriented, FlashInfer GDN kernels are not
  a FP32 activation route, and the useful upstream shape is vLLM's packed decode
  kernel. Treat packed decode as the first port/fork boundary; do not start with
  segmented prefill or fused projection+GDN.
- A local CUDA/JAX FFI packed FP32 GDN decode core now exists as
  `gdn_packed_decode_step_fp32`. It accepts vLLM-style
  `mixed_qkv + a/b/A_log/dt_bias`, consumes native `[B,HV,V,K]` state,
  and passes focused CUDA parity against the pure-JAX packed reference for both
  same-head and GVA q/k repetition cases.
- The model now has an experimental width-1 cached-decode packed GDN route
  selected by `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL`. `reference` calls the
  pure-JAX packed reference from local `A` weights; `cuda_fp32` calls the local
  CUDA/JAX FFI packed core after converting `A` to `A_log` inside the backend.
  The default remains off. The first integrated long-prefill target run with
  `cuda_fp32` was exact and speed-claim-ready but slower than the current
  accepted/scoped default: `88.41 tok/s`, `0.760x` vLLM, versus `90.81 tok/s`,
  `0.780x` vLLM. Keep the route as an implementation tool, not a promoted
  serving path.
- The BF16 packed-decode reference variant
  (`NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=reference`,
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE=bf16`) passes the BF16 diagnostic
  long-decode gate. The corresponding local CUDA variant fails the same
  full-model guardrail (`499/500` top-1, `491/500` ordered top-5,
  `max_hf_topk_id_logit_diff=0.02606964111328125`). Do not continue optimizing
  that local CUDA variant for speed; use the reference variant as the oracle for
  FLA/vLLM-derived kernels.
- The optional backend registry now recognizes `gdn_fla` and aliases
  `fla_gdn`, `vllm_fla`, and `flash_linear_attention`. These requests resolve
  to the pure-JAX fallback until a vLLM/FLA-shaped kernel path is implemented
  and accepted. This keeps the planned production route explicit while leaving
  local CUDA GDN probes as diagnostics.
- The `gdn_fla` module now owns the FP32 packed decode ABI reference and
  fallback-only availability wrapper. Focused tests confirm it remains
  unimplemented/default-off and that the packed decode reference matches the
  current recurrent rule.
- The same neutral module owns the planned segmented prefill ABI reference and
  pack/unpack helpers for `[nnz,H,D]` tensors plus `cu_seqlens`. Existing
  segmented reference tests and the standalone prefill benchmark now import this
  contract from `gdn_fla` instead of the local CUDA diagnostic module.

### Acceptance Gate

```text
- recurrent unit test matches current JAX output within strict tolerance
- full 500-step generated-token parity remains exact
- ITL p50 improves on hetero8
- ITL p95 does not regress
- forward_step_token_ids_jit decreases
- no activation/state dtype downgrade
```

## P1.2 - `gdn_segmented_prefill_chunk32`

### Motivation

Entry 045 establishes chunk size 32 as the best verified point for the tracked
hetero8 workload. The rejected static chunk-major GDN prefill experiment shows
that source-level ragged scans explode into many small dynamic-slice/update
kernels and badly regress first prefill.

Therefore, the next GDN prefill attempt must be backend-owned: one coarse op
that internally handles segmentation and chunking.

### Proposed ABI

```python
gdn_segmented_prefill_chunk32(
    q,              # [nnz_tokens, gdn_heads, head_dim]
    k,              # [nnz_tokens, gdn_heads, head_dim]
    v,              # [nnz_tokens, gdn_heads, head_dim]
    beta,           # [nnz_tokens, gdn_heads]
    gate,           # [nnz_tokens, gdn_heads]
    cu_seqlens,     # [batch + 1]
    initial_state,  # [batch, gdn_heads, head_dim, head_dim]
    chunk_size=32,
) -> (
    y,              # [nnz_tokens, gdn_heads, head_dim]
    final_state,    # [batch, gdn_heads, head_dim, head_dim]
)
```

External prefill references use a related but not identical ABI:

```text
FLA chunk prefill:
- fla/ops/gated_delta_rule/chunk.py::chunk_gated_delta_rule
- q/k [B,T,H,K], v [B,T,HV,V], g/beta [B,T,HV], cu_seqlens [N+1]
- default chunk size in that path is 64, while this repo's accepted Qwen3.5
  point is chunk size 32.

FlashInfer GDN prefill traces:
- flashinfer.gdn_prefill.chunk_gated_delta_rule
- q/k/v [total_seq_len, heads, head_size], cu_seqlens [num_seqs+1]
- state is documented as k-last [N,H,V,K] with FP32 state and BF16 q/k/v.
- local 2026-05-26 audit found this path is Torch-only in the installed vLLM
  environment and gated to newer CUDA targets (`sm90/sm100`) than the current
  A10G baseline.
```

Under the current BF16-weight/FP32-activation contract, treat FLA/FlashInfer
prefill as algorithm and layout references first. Directly adopting their BF16
activation or k-last state contract is a design change and must go through the
full-model real-weight token/logit gate before any serving promotion.
The installed vLLM/FLA chunk path explicitly rejects FP32 activation tensors,
so using that prefill kernel directly requires either a BF16-prefill design
decision or a separate FP32-capable port.

### Acceptance Gate

```text
- exact generated-token match
- first forward_step_token_ids_jit improves
- total TTFT p50 improves
- PjRt Execute does not regress
- command_buffer execute/update does not regress
- no explosion in dynamic_slice/dynamic_update_slice or tiny command buffers
```

### Do Not Merge If

```text
- kernel only improves a microbenchmark
- first prefill gets slower
- chunking creates many more command-buffer executions
```

### Status - 2026-05-26

- GDN prefill is a profile-backed target, not just an architecture guess. Entry
  045's chunk-size change moved the intended `input_reduce_fusion` bucket from
  `59.30 ms / 2512` to `28.65 ms / 1936` and improved the accepted integrated
  server baseline, while later row/chunk source rewrites showed matching
  regressions in first prefill, `PjRt Execute`, and command-buffer work.
- First local CUDA/JAX FFI prototype added as
  `gdn_prefill_chunk32_normalized_fp32` and benchmark variant
  `cuda_fp32_one_piece_chunk32`.
- Focused CUDA FFI tests pass, and the reduced `B=2,H=2,T=64,K=32,V=32`
  benchmark is faster with small drift.
- The benchmark-only CUDA/JAX FFI prefill prototype and standalone GDN prefill
  benchmark now use native V,K state. A non-square smoke shape
  `B=2,H=2,T=64,K=32,V=64` passes focused CUDA parity and standalone output/state
  comparisons: `output_max_abs=1.49e-07`, `state_max_abs=1.073e-06`.
- Full hetero8 model-shape microbenchmark still rejects this prototype after
  the native V,K cleanup: `cuda_fp32_one_piece_chunk32` p50 `10.43 ms` and V64
  p50 `10.46 ms` versus `5.60 ms` for current JAX chunk32, with
  `state_max_abs=2.441e-04`.
- A follow-up V64 value-block variant reduced full-shape p50 from `11.56 ms`
  to `8.60 ms`, confirming value-block/grid overhead is real, but it still lost
  to current JAX `5.44 ms` p50 and kept the same `state_max_abs=2.441e-04`.
- A pure-JAX packed segmented/nnz ABI gate was added before CUDA math. It passes
  reduced shape, but full hetero8 true-token packing fails the strict standalone
  gate versus current padded chunk32: output max `1.431e-05`, state max
  `1.678e-04`. Do not implement CUDA math for this packed ABI until the
  correctness contract is resolved.
- A row-padded diagnostic then padded each packed row back to rectangular
  `T=512` before applying the chunk rule. It still failed (`state_max_abs=
  1.831e-04`), so the drift is not just shorter per-row sequence length; the
  row-wise decomposition itself changes enough FP32 accumulation to miss the
  current gate.
- Correctness policy is now encoded in the GDN prefill microbenchmark output.
  The default policy keeps the strict padded chunk32 output/state gate as the
  only route to segmented CUDA math. When the packed or row-padded ABI misses
  that gate, the benchmark reports `blocked_on_correctness_policy`,
  `cuda_math_allowed=false`, and `requires_design_decision=true`. A true-token
  packed ABI can only proceed after an explicit design decision and a separate
  real-weight full-model token/logit parity gate. That override gate is also
  machine-readable now: exact generated-token match, 500/500 top-1 match,
  500/500 ordered top-5 match, 500/500 top-5 set match, and
  `max_hf_topk_id_logit_diff <= 2e-5` against the stored HF long-decode
  artifact.
  A separate BF16 external-kernel lane keeps identity exact and can use
  `max_hf_topk_id_logit_diff <= 1e-4` when explicitly scoped.
- Kernel-phase guardrail revalidation on 2026-05-26 produced
  `results/qwen08_jax_bf16w_fp32act_long_decode_top5_compare_20260526_kernel_phase_gate.json`
  at git head `d66b285`. It kept exact `500/500` top-1, ordered top-5, and
  top-5-set matches, but the numeric `max_hf_topk_id_logit_diff` was
  `2.09808349609375e-05`, narrowly above the `2e-5` override bound. Treat the
  full-model override as not currently passed; do not relax the threshold
  without an explicit correctness decision.
- Keep it default-off and benchmark-only. Do not route into serving. The next
  attempt should move closer to the true segmented/nnz ABI and preserve FP32
  accumulation more closely before a server run.

## P2.1 - `paged_prefill_attention_gqa_nhd`

### Motivation

Useful for the 6 full-attention layers, but lower priority than GDN because this
architecture is GDN-heavy.

### Reference

FlashInfer provides `BatchPrefillWithPagedKVCacheWrapper` for batched
prefill/append attention over paged KV cache.

MaxText is useful as a design reference: its docs say serving attention uses
paged/ragged kernels to fetch non-contiguous KV cache pages in high-throughput
inference.

### Proposed ABI

```python
paged_prefill_attention_gqa_nhd(
    q,                 # [nnz_q, num_q_heads, head_dim]
    k_cache,
    v_cache,
    qo_indptr,          # [batch + 1]
    kv_indptr,          # [batch + 1]
    kv_indices,         # [total_pages]
    kv_last_page_len,   # [batch]
    causal=True,
) -> out                # [nnz_q, num_q_heads, head_dim]
```

### Acceptance Gate

```text
- exact-token match
- TTFT p50 improves
- no ITL regression
- no per-layer layout conversion
```

### Status - 2026-05-26

- Current stored traces do not justify starting this kernel yet. In the stored
  `long_prefill_512_2048/gpu_paged_default` profile, the whole
  `generate_with_trace` range is about `820 ms`; the attention-shaped
  `triton_softmax_*` buckets total only about `10 ms`, while larger buckets are
  host readback/sync (`np.asarray(jax.Array)` about `428 ms`), GEMM/fusion
  (`gemm_fusion` about `247 ms`), recurrent/GDN-shaped `while` work about
  `210 ms`, command-buffer execute about `229 ms`, transpose about `47 ms`, and
  `input_reduce_fusion` about `38 ms`. Keep Commit 9 blocked until a repeatable
  profile shows paged-prefill attention itself is a material TTFT bottleneck.
- Host token readback is a sync label on prior GPU work, not currently a proven
  D2H copy bottleneck. Entry 044 already rejected `copy_to_host_async()` on the
  token-id result. A follow-up subagent audit found no smaller local cleanup
  likely to reduce sync count without changing scheduler semantics. Reducing the
  sync count further likely requires a device-side multi-token greedy decode
  loop or similar scheduler dependency change; the first opt-in greedy
  decode-burst attempt proved this is not enough if implemented as a source-level
  JAX scan around the full model. It stayed exact and reduced readback count, but
  lowered into a much slower graph. A follow-up device-token-carry route that
  keeps one-token greedy outputs on device between scheduler steps is exact and
  modestly faster for offline fixed-length throughput, but it defers token
  materialization and therefore is not equivalent to the streaming server timing
  contract. Do not continue host-sync work as the primary path without a
  backend-owned loop/kernel boundary or a deliberate offline-throughput API.

## P2.2 - `qk_norm_rope_kv_append_fused`

### Motivation

Only attempt this after P0 kernels are stable. The useful fusion is not just
RoPE; it is:

```text
QKV projection output
-> split Q/K/V
-> Q/K norm if present
-> RoPE on Q/K
-> append K/V into paged cache
-> return Q for attention
```

### Acceptance Gate

```text
- exact-token match
- lower decode-layer overhead
- no extra layout conversions
- no host-side metadata changes
```

### Do Not Start Before

```text
- kv_append_paged_nhd is stable
- paged_decode_attention_gqa_nhd is stable
- profile still shows norm/RoPE/cache-write overhead
```

## P3 - Sampling/Top-k/Logprob Kernels

### Motivation

These are useful for features and diagnostics, but not the next throughput lever
for greedy serving.

FlashInfer exposes top-k, sampling, logits processor, norm, RoPE, and activation
APIs, so these can be used later as FFI smoke tests or feature work.

### Possible APIs

```python
topk_logits(
    logits,  # [batch, vocab]
    k: int,
) -> (values, indices)
```

```python
sample_topk_topp(
    logits,
    temperature,
    top_k,
    top_p,
    rng_state,
) -> token_ids
```

### Do Not Prioritize Until

```text
- decode ITL is closer to vLLM
- non-greedy sampling/logprobs are product requirements
- MTP diagnostics need draft-rank/top-k efficiently
```

## Phase 3 - MTP Policy

### Goal

Keep MTP as diagnostics/correctness work, not speed work.

### Tasks

Add MTP diagnostics, but do not optimize MTP kernels yet:

```text
- draft token rank under target model
- accepted/rejected draft count
- acceptance rate by prompt bucket
- verifier overhead
- commit-select JIT/cache-miss overhead
- warmed vs cold commit-select timing
```

The current logbook says MTP1 repeat was unusably slower, dominated by
verifier/commit-select/tracing overhead, and exact output was preserved only
because drafts were rejected.

### MTP Can Be Revisited Only When

```text
- acceptance rate is meaningfully nonzero on at least one realistic prompt suite
- commit-select shapes are warmed before timing
- MTP beats paged baseline on a controlled benchmark
```

## Phase 4 - What To Borrow

### From vLLM

Borrow:

```text
- paged KV layout discipline
- slot/page table contract
- decode attention ABI
- cache append kernel contract
- CUDA graph/static-shape mindset
- hybrid Qwen attention/GDN implementation references
```

Do not borrow:

```text
- full scheduler architecture
- PyTorch-specific graph assumptions
- MTP speed path before acceptance quality is proven
```

vLLM's paged attention docs are the right reference for the core decode attention
memory-access pattern.

### From FlashInfer

Borrow first, because it is the most practical CUDA kernel route from JAX:

```text
- append_paged_kv_cache
- BatchDecodeWithPagedKVCacheWrapper
- BatchPrefillWithPagedKVCacheWrapper
- later: top-k/sampling/RoPE/RMSNorm helpers
```

FlashInfer's docs describe using FlashInfer kernels from JAX through
`jax-tvm-ffi`, with CUDA/JAX/FlashInfer setup requirements.

### From MaxText

Borrow design patterns, not GPU drop-ins:

```text
- profile first
- use custom kernels for bandwidth-bound or irregular/ragged work
- use auxiliary metadata
- validate integrated model performance, not only microbenchmarks
- think in decode/prefill/mixed specialized kernels
```

MaxText's Pallas guide says custom kernels are appropriate for irregular compute
and memory-access-bound attention, but also warns to profile first and prefer
XLA when standard dense ops already perform well.

The Ragged Paged Attention paper is useful conceptually: it uses fine-grained
ragged tiling, fuses KV cache update with attention computation, and specializes
decode/prefill/mixed kernels for TPU serving. Treat it as a design reference for
kernel boundaries, not a CUDA implementation to copy.

## Implementation Order

Commit 1:

- ~~Add `docs/current_gpu_baseline.md`~~
- ~~Add `docs/rejected_optimization_index.md`~~
- ~~Add `docs/kernel_roadmap.md`~~
- ~~Add benchmark config JSONs~~

Commit 2:

- ~~Add benchmark matrix runner~~
- ~~Add summary JSON output schema~~

Commit 3:

- ~~Add kernel backend registry:~~

- ~~`nanovllm_jax/kernels/__init__.py`~~
- ~~`nanovllm_jax/kernels/registry.py`~~
- ~~`nanovllm_jax/kernels/flashinfer_ffi.py`~~
- ~~`nanovllm_jax/kernels/cuda_gdn.py`~~

Commit 4:

- ~~Add NHD full-attention KV cache allocation behind a flag~~
- ~~Keep existing pure-JAX cache fallback~~

Commit 5:

- ~~Add `kv_append_paged_nhd` prototype via FlashInfer/JAX FFI~~
- ~~Add focused CUDA parity test against the pure-JAX NHD append reference~~
- ~~Route full-attention layers through NHD append behind an opt-in flag~~
- ~~Run live integrated attempt and record rejection under the current FP32 KV-cache contract~~

Interim ABI validation:

- ~~Add pure-JAX `kv_append_paged_nhd` ABI reference for the exact NHD append contract~~
- ~~Add focused parity test against the canonical `update_kv_cache` path~~
- ~~Add local CUDA/JAX FFI FP32 `kv_append_paged_nhd` smoke kernel~~
- ~~Add focused CUDA parity test against the pure-JAX NHD append reference~~
- ~~Route local FP32 append through `write_kv` behind an opt-in flag~~
- ~~Run live integrated attempt and record rejection of standalone append routing~~

Commit 6:

- Do not add `paged_decode_attention_gqa_nhd` via FlashInfer/JAX FFI under the
  current FP32 activation/KV-cache contract.
- ~~Add pure-JAX FP32 `paged_decode_attention_gqa_nhd` ABI reference~~
- ~~Add focused parity tests against the current decode path~~
- ~~Add FP32-capable CUDA/custom-call decode implementation with focused CUDA
  parity tests~~
- ~~Route FP32-capable CUDA/custom-call decode implementation behind an opt-in
  backend~~
- ~~Run live integrated attempt and record rejection of standalone decode
  routing~~
- ~~Run live integrated paired append+decode attempt and record rejection of the
  narrow local FP32 pair~~
- Route only full-attention decode layers as an accepted path after integrated
  gates pass

Commit 7:

- ~~Add gdn_recurrent_decode_step prototype~~
- ~~First isolated tests~~
- ~~Add vLLM-style packed GDN decode ABI reference while preserving local
  `[B,H,K,V]` state layout~~
- ~~Audit vLLM/FLA GDN references and record that packed decode with local
  state is the smallest non-design-changing next target~~
- ~~Add local CUDA/JAX FFI packed FP32 GDN decode core with focused parity
  tests~~
- ~~Route through `gated_delta_decode` behind an opt-in flag~~
- ~~Run integrated decode benchmark and record rejection of standalone GDN
  decode routing~~

Commit 7b - GDN V,K layout migration:

- ~~Add a layout migration note/tests for the pre-change `[B,L,HV,K,V]` state and
  target `[B,L,HV,V,K]` state.~~
- ~~Change `init_hybrid_state` and recurrent-state shape expectations to
  `[batch, linear_layer, value_heads, value_dim, key_dim]`.~~
- ~~Adapt `jax_recurrent_gated_delta_rule` and `jax_chunk_gated_delta_rule` so the
  pure-JAX fallback consumes and returns V,K state natively.~~
- ~~Update state-table, MTP commit-select, and focused parity tests for the new
  shape.~~
- ~~Make the default-off local CUDA FFI recurrent, packed decode, and chunked
  prefill probes consume V,K state natively, with no Python K,V compatibility
  transpose.~~ Validation: focused recurrent/packed CUDA decode selection
  `4 passed, 9 deselected`; focused prefill selection `2 passed, 11
  deselected`; full CUDA FFI suite `13 passed`; focused GDN CUDA/segmented suite
  `18 passed`.
- ~~Run layer parity, cached prefill/decode equivalence, 500-token top-5/logit
  guardrail, and integrated server benchmark before promoting the layout.~~
  Validation: focused CUDA suite `12 passed`; CUDA FFI suite `13 passed`; MTP
  commit-state suite `15 passed, 1 xfailed`; long-decode top-5 guardrail passed
  `500/500` with max HF top-k-id logit diff `1.9073486328125e-05`;
  integrated long-prefill goal target is speed-claim-ready at `90.65 tok/s`,
  `0.779x` vLLM, exact generated-token parity over two repeats.

Commit 7c - Optional BF16 GDN prefill activation experiment:

- Defer this until after the V,K FP32 path has a stable correctness and speed
  baseline. Do not make BF16 prefill part of the default migration.
- ~~Add `NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE=bf16` or equivalent opt-in flag.~~
- ~~Limit the first experiment to GDN prefill activations; keep recurrent state
  and decode activation math FP32 unless a separate decision changes that
  contract.~~
- ~~Compare against the V,K FP32-prefill baseline, not against the old K,V
  layout.~~
- Treat FLA/FlashInfer BF16 prefill as an integration/performance hypothesis:
  useful only if it reduces integrated TTFT or throughput bottlenecks after
  full-model gates, not merely because the external kernel prefers BF16 inputs.
- ~~Run an integrated one-repeat goal-target diagnostic and long-decode top-5
  guardrail for the first narrow BF16 reference lane.~~ Result: default-off path
  was exact for the one-repeat integrated 16-token benchmark but failed the
  long-decode top-5/logit gate, so it is rejected for promotion.
- Require layer/state drift checks, cached prefill plus long-decode top-5/logit
  guardrails, exact generated-token parity, and an integrated TTFT/throughput
  win before any future BF16 external-kernel promotion.
- If the BF16 path only helps an external-kernel microbenchmark but fails the
  full-model gates, keep it rejected/default-off.
- ~~Add a reproducible external GDN kernel feasibility probe for the local host
  and record the FlashInfer/vLLM/FLA constraints.~~ Validation:
  `benchmarks/probe_external_gdn_kernels.py --run-smoke` wrote
  `results/external_gdn_kernel_probe_20260527_sm86.json`, showing A10G SM86 and
  direct FlashInfer GDN prefill blocked by the SM90/SM100 requirement.
- ~~Add a Torch-side vLLM/FLA GDN microprobe on SM86 to verify the real upstream
  kernels run and to record their model-shaped BF16 timing and independent
  recurrent-reference deltas before porting.~~
  Validation:
  `benchmarks/probe_vllm_fla_gdn.py --warmups 2 --repeats 5` wrote
  `results/vllm_fla_gdn_probe_20260527_sm86.json`; the updated artifact was
  rerun with reference checks. vLLM FLA prefill prep+chunk p50 was `1.45 ms`
  for `5120` total tokens and packed decode p50 was `0.11-0.16 ms` for batch
  sizes `1,4,8,16`. Packed-decode reference output matched exactly; ragged
  prefill reference output max abs was `4.88e-4`.

Commit 8:

- ~~Add first gdn_segmented_prefill_chunk32 prototype~~
- Keep first CUDA one-piece chunk32 prototype default-off and benchmark-only;
  do not route it into serving
- ~~Run value-block-width follow-up and record that V64 improves V32 but still
  misses the full-shape microbenchmark gate~~
- ~~Add packed segmented/nnz ABI correctness gate before CUDA math~~
- ~~Run row-padded segmented reference diagnostic and record that row-wise
  decomposition still misses the full-shape gate~~
- ~~Resolve packed-ABI correctness policy after full hetero8 gate failure before
  implementing segmented CUDA math~~
- ~~Make the full-model token/logit override gate machine-readable in
  `benchmark_long_decode_top5.py`~~
- ~~Fix the standalone GDN prefill benchmark/probes to generate and reconstruct
  native V,K state, including non-square K/V smoke coverage.~~
- ~~Add `gdn_fla`/vLLM-FLA aliases to the optional backend registry so the
  planned GDN production route is explicit and still falls back to pure JAX
  until accepted.~~ Validation: `tests/test_kernel_registry.py` `8 passed`.
- ~~Add neutral `nanovllm_jax.kernels.gdn_fla` FP32 packed-decode ABI reference
  so the planned vLLM/FLA contract is separate from local CUDA diagnostics.~~
  Validation: `tests/test_gdn_packed_decode_reference.py`
  `tests/test_kernel_registry.py` `12 passed`.
- ~~Move the planned segmented GDN prefill ABI reference imports to
  `nanovllm_jax.kernels.gdn_fla` so both decode and prefill FLA-shaped
  contracts live outside the local CUDA diagnostic module.~~ Validation:
  `tests/test_gdn_segmented_reference.py`
  `tests/test_gdn_packed_decode_reference.py`
  `tests/test_kernel_registry.py` `13 passed`.
- ~~Audit the installed vLLM/FLA and FlashInfer GDN routes for direct FP32 reuse
  vs port/fork requirements.~~ Result: direct reuse is blocked for the FP32
  activation contract; packed decode is the smallest next port/fork boundary.
- ~~Audit whether vLLM's vendored Triton/FLA GDN kernels can be called directly
  from JAX without rewriting.~~ Result: direct Torch/Triton reuse is not a
  viable low-risk path on this stack (`jax_triton` absent, vLLM venv lacks
  JAX/`jax_tvm_ffi`, no stable non-Torch ABI, no simple JIT-safe DLPack export).
  Use vLLM/FLA as golden reference and port/fork behind the existing JAX-facing
  GDN boundaries.
- ~~Route width-1 cached GDN decode through the vLLM-shaped packed boundary
  behind `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL`, with `reference` and
  `cuda_fp32` implementations.~~ Validation: elevated
  `JAX_PLATFORMS=cuda` focused selection
  `tests/test_gdn_packed_decode_reference.py tests/test_cuda_fp32_ffi.py -k
  packed_decode` passed `9 passed, 11 deselected`.
- ~~Add a default-off GDN post-conv prefill reference boundary behind
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference`, routing split,
  beta/gate construction, valid-token masking, GQA repeat, layout packing, and
  chunked prefill through `backend.gated_delta_prefill_post_conv`.~~
  Validation: elevated CUDA focused suite
  `tests/test_gdn_post_conv_prefill_reference.py
  tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py
  tests/test_kernel_registry.py` passed `16 passed`; one-repeat integrated
  long-prefill route was exact at `90.15 tok/s`, `0.775x` vLLM.
- ~~Factor the post-conv prefill prep into an explicit FLA-shaped helper
  returning `[B,T,H,D]` q/k/v, `[B,T,H]` gate/beta, and row lengths while
  preserving the existing reference fallback behavior.~~ Validation: elevated
  CUDA focused suite
  `tests/test_gdn_post_conv_prefill_reference.py
  tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py
  tests/test_kernel_registry.py` passed `19 passed`.
- ~~Add the rectangular prepared-body FP32 reference
  `gdn_fla_prefill_chunk32_fp32_reference` for q/k/v `[B,T,H,D]`, gate/beta
  `[B,T,H]`, `seq_lens [B]`, and state `[B,H,V,K]`.~~ Validation: elevated
  CUDA focused suite
  `tests/test_gdn_post_conv_prefill_reference.py
  tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py
  tests/test_kernel_registry.py` passed `21 passed`.
- ~~Add the prepared-layout varlen packing/reference contract for vLLM/FLA
  prefill: `[B,T,H,D] -> [nnz,H,D] + cu_seqlens -> [B,T,H,V]`.~~ Validation:
  elevated CUDA focused selection
  `tests/test_gdn_post_conv_prefill_reference.py -k 'varlen or prepared_fla_chunk32_reference_matches_post_conv_reference or masks_padded_rows'`
  passed `3 passed, 7 deselected`.
- ~~Route `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32`
  through the model/server path using the prepared-body reference.~~
  Validation: elevated CUDA focused suite passed `22 passed`; one-repeat
  integrated long-prefill route was exact at `89.37 tok/s`, `0.768x` vLLM.
- ~~Add a default-off prepared-layout FP32 CUDA FFI target
  `gdn_prefill_chunk32_prepared_fp32` and route it through
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_fla_chunk32_fp32`.~~
  Validation: elevated CUDA focused suite passed `38 passed`; one-repeat
  integrated long-prefill route was exact but slow at `63.82 tok/s`,
  `0.548x` vLLM. Reject this old-body layout adaptation for promotion.
- Add the fast vLLM/FLA-derived implementation behind the same post-conv
  boundary. The first fast attempt should either fuse post-conv prep into
  chunked prefill or call a FLA-shaped kernel without adding hot-path
  K,V/V,K transposes or per-layer layout conversions.
- Use the audited vLLM FLA pipeline as the implementation decomposition:
  `fused_post_conv_prep`, `chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`,
  `solve_tril`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h`, and
  `chunk_fwd_o`. Keep Torch autograd/runtime wrappers out of the JAX port.
  Preserve FP32 gate/beta/state. BF16 q/k/v remains an opt-in diagnostic unless
  it passes the long-decode top-logit gates.
- Use the existing BF16 packed-decode reference as the decode-side semantic
  oracle, but do not continue the local CUDA BF16 packed-decode route after its
  500-token guardrail failure. If decode is revisited, adapt the vLLM/FLA
  packed decode semantics directly instead of repairing the historical local
  prototype.
- ~~Add the FLA varlen chunk metadata helper for `chunk_indices` and
  `chunk_offsets`, preserving original row ids when bucket padding creates
  zero-length rows.~~ Validation: elevated CUDA focused selection
  `tests/test_gdn_segmented_reference.py -k 'chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `3 passed`.
- ~~Add scalar FLA chunk-local cumsum packed reference over `[nnz,H]` gates,
  including reverse mode and chunk-reset tests.~~ Validation: elevated CUDA
  focused selection
  `tests/test_gdn_segmented_reference.py -k 'chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `4 passed`.
- ~~Add scalar FLA chunk-scaled-dot-KKT packed reference over `[nnz,Hg,K]` keys
  and `[nnz,H]` beta/gate tensors, including strict-lower masking and
  grouped-head tests.~~ Validation: elevated CUDA focused selection
  `tests/test_gdn_segmented_reference.py -k 'chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `5 passed`.
- ~~Add scalar FLA solve-tril packed reference over `[nnz,H,BT]` chunk
  matrices, including ragged partial chunks and `(I + A)^-1` tests.~~
  Validation: elevated CUDA focused selection
  `tests/test_gdn_segmented_reference.py -k 'solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `6 passed`.
- ~~Add FLA recompute-w/u packed reference over `[nnz,H,D]` values and grouped
  `[nnz,Hg,K]` keys, including gate-exp scaling and grouped-head tests.~~
  Validation: elevated CUDA focused selection
  `tests/test_gdn_segmented_reference.py -k 'recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `7 passed`.
- ~~Add FLA chunk-delta-h packed reference for `chunk_gated_delta_rule_fwd_h`,
  including `chunk_offsets`, prior-state `h`, ungated `v_new`, gate-rescaled
  state updates, and final-state tests.~~ Validation: elevated CUDA focused
  selection
  `tests/test_gdn_segmented_reference.py -k 'chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `8 passed`.
- ~~Add FLA chunk-fwd-o packed reference for `chunk_fwd_o`, including
  query-to-prior-state output, causal intra-chunk attention over ungated
  `v_new`, gate scaling, grouped heads, and explicit output scale.~~
  Validation: elevated CUDA focused selection
  `tests/test_gdn_segmented_reference.py -k 'chunk_fwd_o or chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `9 passed`.
- ~~Add composed FLA chunk-gated-delta packed reference that runs
  `chunk_local_cumsum`, KKT, solve-tril, recompute-w/u, chunk-delta-h, and
  chunk-fwd-o in vLLM order and compares against the segmented JAX
  reference.~~ Validation: elevated CUDA focused selection
  `tests/test_gdn_segmented_reference.py -k 'chunk_gated_delta_rule_packed or chunk_fwd_o or chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'`
  passed `10 passed`.
- ~~Add a vLLM `fused_post_conv_prep`-inspired CUDA FP32 prep-only
  implementation behind `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_fp32`.~~
  Validation: elevated CUDA focused suite passed `18 passed`; one-repeat
  integrated long-prefill route was exact but slower at `87.80 tok/s`,
  `0.755x` vLLM, so this is rejected for promotion.
- Add a guarded prep+chunk32 implementation behind the same boundary, using the
  local FP32/V,K chunk32 or V64 path only when shape guards pass, and compare it
  against the post-conv reference boundary before any promotion.
- ~~Add a guarded prep+chunk32 implementation behind the same boundary using
  local FP32/V,K chunk32 or V64 when shape guards pass.~~ Validation:
  one-repeat integrated long-prefill route was exact but much slower at
  `60.46 tok/s`, `0.520x` vLLM, dominated by
  `Fp32GdnPrefillChunk32Kernel<64>` at about `500.69 ms`; reject this local
  chunk body for serving.
- Compare a revised segmented prefill candidate against Entry 045 chunk-32
  baseline after it beats the full-shape GDN microbenchmark gate

Commit 9:

- Add paged_prefill_attention_gqa_nhd only if traces justify it

Commit 10:

- Add MTP diagnostics; no MTP speed optimization yet

If FlashInfer/JAX FFI integration is harder than expected, insert a small ABI
validation commit before Commit 4. That commit should prove an optional external
kernel can be discovered, gated, called, profiled, and bypassed without touching
the pure-JAX correctness path.

## Current Decode-Heavy Split Diagnostic

Status as of 2026-06-02:

- Current exact-token decode-heavy diagnostic:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_split_20260602/nvj_decode_split_diag_jax_20260602.json`.
- Fresh same-shape vLLM async reference:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_split_20260602/nvj_decode_split_diag_vllm_async_20260602.json`.
- vLLM graph-mode CUDA-event split:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_split_20260602/nvj_decode_split_diag_vllm_offline_cuda_events_20260602.json`.
- Current result:
  - JAX `decode_heavy_128x128/gpu_paged_gdn_fla_decode_static_metadata`:
    exact generated-token match, `167.65 output tok/s`, `0.7635 s`;
  - vLLM async baseline, same `128 -> 128` shape with `logprobs=5`:
    `219.03 output tok/s`, `0.5844 s`;
  - current JAX ratio: `0.766x`; `0.9x` target requires about `197.12 tok/s`,
    or roughly `114 ms` less wall time on this 128-token run.
- LM-head/logits finding:
  - JAX LM-head/logits bucket: `127.44 ms` across 127 decode calls;
  - vLLM `logits_processor.forward`: `137.86 ms` across 128 calls;
  - vLLM sampler after logits: `29.79 ms`, including top-k logprob gather.
- Interpretation:
  - vLLM is not faster because it avoids logits materialization; it pays a
    similar or larger logits bucket in this benchmark;
  - logits materialization is a shared floor, not the end-to-end upper bound;
  - the remaining JAX gap is dominated by replay/launch/static metadata and
    non-LM model kernels, not by the GDN decode core or LM-head alone.
- Next optimization priority:
  - reduce PJRT/command-buffer/static-metadata churn (`PjRT Execute`
    `428.69 ms`, command-buffer execute/update `130.11/90.99 ms`,
    scheduler/build/static decode arrays around `100 ms` each);
  - reduce non-LM projection/reduction buckets before spending more target
    time on LM-head-only kernels.
- PjRT overlap audit:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/pjrt_split_20260602/pjrt_split_20260602.md`;
  - `PjRT Execute` is not standalone CPU arithmetic in this trace:
    `407.21 ms` of `428.69 ms` overlaps GPU-active intervals (`95.0%`);
  - same-thread PjRT-exclusive time after subtracting nested children is only
    `3.18 ms` total;
  - grouping the `410` PjRT events shows the main contribution is still the
    decode executable, not incidental metadata conversions: `127`
    `forward_step_token_ids_table_jit` events account for `380.82 ms`, while
    `convert_element_type` and `broadcast_in_dim` account for `19.81 ms` and
    `12.23 ms`;
  - the decode step boundary still shows replay pressure: `127`
    `forward_step_token_ids_table_jit` rows contain `2413`
    command-buffer executes and `2413` updates, about `19` of each per token.
- Updated decode-heavy baseline after Entry 210:
  - `gpu_paged_gdn_fla_decode_static_metadata` now uses explicit
    `kernels.gdn.packed_decode.impl=reference` with BF16 QKV for decode and
    enables all block-dot FLA prefill stages through
    `kernels.gdn.prefill_block_dot=true`;
  - paired no-profile smokes: `triton_fla` decode was `169.88` and
    `170.58 tok/s`, while packed `reference` was `174.09` and
    `173.88 tok/s`, all exact;
  - after the slot/token conversion cache, no-profile decode-heavy is exact at
    `173.99 tok/s`;
  - after composing block-dot prefill with the reference decode route,
    no-profile decode-heavy is exact at `197.91`, `196.38`, and
    `197.96 tok/s`, median `197.91 tok/s`;
  - against the fresh vLLM async baseline `219.03 tok/s`, the median ratio is
    `0.904x`, which meets the fresh `0.9x` target of about `197.12 tok/s`;
  - target artifacts:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_blockdot_prefill_noprofile.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_blockdot_prefill_noprofile_r2.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_blockdot_prefill_noprofile_r3.json`;
  - profile structure is cleaner (`PjRT Execute` `198.22 ms / 283`,
    command-buffer execute/update `79.09/35.26 ms`, `MemcpyD2D`
    `4.18 ms / 362`) after Entry 207, but that cleanup should not be counted
    as a standalone throughput win;
  - the config-driven profiled smoke after Entry 210 is exact and records all
    four block-dot flags as true, but profiling lowers throughput to
    `191.23 tok/s`; use it for bucket movement, not for the no-profile speed
    claim.
- Entry 211 full no-profile sanity:
  - all four sanity workloads passed exact parity;
  - `long_prefill_512_2048` median `106.08 tok/s`, `0.912x` stored vLLM;
  - `decode_heavy_128x128` median `197.60 tok/s`, `0.902x` fresh vLLM;
  - `hetero8` and `short_32_128` remain below target at `0.367x` and `0.680x`
    stored vLLM respectively.
- Entry 212 hetero8 hill-climb:
  - B>1 BF16 decode projections and decode output-projection casts are exact
    and raise direct no-profile hetero8 from the Entry 211 `317.35 tok/s`
    median to the mid-`340 tok/s` range;
  - disabling XLA Triton GEMM while preserving autotune level 4 is exact and
    raises direct no-profile hetero8 to `441.10` and `438.98 tok/s`, median
    `440.04 tok/s`;
  - canonical profiled matrix:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260602/gpu_matrix_hetero8_no_triton_gemm_r2_20260602.json`;
  - profiled matrix median: `414.45 tok/s`, `1.127x` stored Entry 045 JAX
    reference, `0.480x` stored vLLM. This is a real hetero8 improvement but
    still not a `0.9x` vLLM result.
- Entry 213/214 hetero8 hill-climb:
  - layerwise GDN table-state routing is exact and raises the no-profile anchor
    to `457.56 tok/s`;
  - packed GDN input projection, packed full-attention Q/K/V decode, and
    canonical packed MLP gate/up remove the runtime weight-concat buckets and
    raise no-profile hetero8 to `543.27 tok/s`;
  - conv-fused Triton FLA GDN decode becomes positive after the packed
    projection/MLP changes and is exact over two no-profile repeats,
    `552.86` and `552.59 tok/s`, median `552.72 tok/s`;
  - current hetero8 ratio is `0.640x` stored vLLM (`864.18 tok/s`), so the
    `0.9x` hetero8 target remains open.

Profile interpretation guardrails:

- Compare only same workload, same config family, same repeat policy, and same
  profile scope before declaring a bucket moved. Do not compare decode-heavy,
  hetero8, and long-prefill bucket totals directly.
- Treat CPU `PjRT Execute`, `command_buffer::*`, scheduler, and `device_put`
  labels as inclusive host-side wrapper/proxy buckets. They overlap GPU work
  and are not additive with GPU kernel totals.
- Do not call `PjRT Execute` a CPU-compute bottleneck unless an overlap split
  shows substantial same-thread exclusive time. In the current split it is
  primarily host wrapper/wait/replay wall time around GPU work.
- Treat GPU top events as the first exclusive kernel targets, but require an
  integrated wall-time drop. A single-kernel reduction is not enough if PJRT or
  command-buffer counts rise.
- For replay/launch work, the acceptance signal is lower per-step
  `PjRT Execute`/`command_buffer` counts and lower wall time on exact-token
  decode-heavy, not just a renamed profile bucket.

## Current Hetero8 Bottleneck Tracker

Status as of 2026-06-03:

- Current accepted local hetero8 route under investigation:
  `gpu_paged_gdn_fla_decode_static_metadata` with block-dot GDN prefill,
  packed prefill/decode projections, BF16 model compute, BF16 full-attention
  KV cache, BF16 decode output projections, canonical packed MLP gate/up,
  conv-fused Triton FLA GDN decode, and
  `XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false"`.
- Current profiled artifact:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/packed_prefill_proj/gpu_matrix_hetero8_packed_prefill_proj_r1_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`.
- Current result: exact generated-token match, `562.02 tok/s` profiled,
  no-profile repeats `593.98` and `588.51 tok/s`, median `591.24 tok/s`,
  `0.684x` stored vLLM reference.
- Current best hetero8 run after Entry 229 BF16 model compute:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_compute_hetero8/gpu_matrix_hetero8_bf16_compute_r1_20260603.json`;
  run artifact:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_compute_hetero8/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_bf16_compute_repeat1.json`;
  exact generated-token match, full generated lengths, zero measured-phase JIT
  growth, `651.18 tok/s`, `0.754x` stored vLLM. This is the current fastest
  hetero8 result, but it is one repeat and still below the `0.9x` target
  (`777.76 tok/s`).
- Entry 239 cleanup check:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/skip_unused_final_norm/gpu_matrix_hetero8_skip_unused_final_norm_r1_20260603.json`;
  run artifact:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/skip_unused_final_norm/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  exact, full-length, zero measured-phase JIT growth, `650.62 tok/s`.
  Keep the final-norm cleanup as code cleanup, but do not treat it as a speed
  improvement over Entry 229.
- Dominant shape: decode, not prefill. One prefill step is about `0.02 s`;
  31 decode steps total about `0.43-0.44 s` in the current no-profile runs.
- Current main buckets:
  - compiled decode execution: PJRT execute about `269 ms / 91` in the selected
    BF16-cache profiled matrix and about `260 ms / 91` after packed-prefill
    projection reuse;
  - GPU GEMM events remain dominant: CUTLASS/BF16 GEMM rows around
    `68.3 ms / 72`, `55.6 ms / 1488`, `31.3 ms / 31`, and `24.5 ms / 1488`;
  - full-attention/cache fusions remain visible:
    `loop_slice_fusion_23` about `22.2 ms / 32` and
    `input_scatter_fusion_15` about `19.2 ms / 31`;
  - conv-fused GDN decode kernel: `_gdn_conv_packed_decode_raw_gate_kernel`
    about `20.6 ms / 558`;
  - root `MemcpyD2D` has been reduced by BF16 KV cache
    (`55.10 ms -> 12.32 ms` versus the prior profile), so cache dtype is no
    longer the biggest obvious copy bucket;
  - host token materialization labels remain visible (`np.asarray(jax.Array)`
    about `69.4 ms / 32`), but they overlap GPU work and are not an
    independent CPU-compute bucket.
- Recently accepted optimizations:
  - layerwise hybrid-state routing;
  - canonical packed MLP gate/up with separate gate/up leaves omitted from the
    JIT parameter tree;
  - packed GDN/full-attention decode projections;
  - packed GDN/full-attention prefill projection reuse through the existing
    compact-prefill controls;
  - conv-fused Triton FLA GDN decode after the packed projection/MLP changes;
  - BF16 physical full-attention KV cache with explicit K/V cache-write casts.
- Recently rejected/not-promoted optimizations:
  - naive closed-params decode lowering captured about `2.01 GB` constants and
    was killed;
  - layerwise KV-cache internal path, generalized padded GEMM, `packed_decode=off`,
    and RoPE duplicate-removal did not beat the layerwise anchor;
  - non-conv `triton_fla_raw_gates` GDN decode is exact but neutral/slower
    (`543.04 tok/s`) versus canonical packed MLP with reference GDN decode
    (`543.27 tok/s`);
  - `all_rows_valid` KV-write sentinel-skip and trace-token prefetch were exact
    but slower in the no-profile lane (`547.59` and `548.44 tok/s`);
  - compact full-attention cache layer packing was exact but neutral
    (`586.70 tok/s`) versus the BF16-cache median, so it was reverted;
  - FlashInfer KV append with BF16 cache passed the focused backend test and
    ran exact integrated hetero8, but it was slower (`578.89 tok/s`) than the
    selected BF16-cache route and remains off;
  - FlashInfer decode attention is not currently a JAX drop-in. The installed
    decode path is Torch custom-op oriented, while the local JAX FFI only
    covers paged KV append;
  - the old duplicate-leaf packed MLP rejections still stand. Only the canonical
    representation is accepted.
  - whole-model BF16 compute is now accepted as a speed route after one exact
    hetero8 repeat; it still needs broader correctness and repeat/profile
    evidence before a speed claim.
  - Entry 239 reference GDN decode direct no-profile run (`651.95 tok/s`) tied
    the selected conv-fused Triton GDN decode direct run (`650.25 tok/s`) but
    did not prove a matrix-level median win. Do not switch the selected config
    on this evidence alone.
  - Entry 239 current-BF16 MTP1 diagnostic did not reach measurement after
    about three minutes of warmup/compilation. Do not use MTP as the near-term
    hetero8 path without a dedicated speculative warmup and acceptance redesign.

Next hetero8 experiments should reduce one of the current dominant buckets and
must compare against both stored Entry 045 and stored vLLM. Avoid source-level
rewrites unless the profile shows that the rewrite removes a dominant bucket in
the integrated server trace. Entry 216's packed-prefill reuse is accepted as a
small cleanup, but it did not change the `1488`-count decode GEMM buckets; the
next material speedup needs a decode-side structural boundary. Current evidence
points to three credible directions: reduce full-model decode executions,
replace XLA's small-B decode projection GEMM path with a better external
backend, or group the regular `3 x GDN + 1 x full-attention` layer pattern into
a broader compiled boundary. Do not spend more primary-target time on isolated
RMSNorm, token materialization ordering, seq-lens carry, or GDN-core-only
changes.

Random-request sidecar status as of Entry 240:

- The seed `1234` random suite (`512-4096` input tokens, `256-1024` output
  tokens, `15` requests, `30506/11602` input/output tokens) is now the broad
  random-serving stress target. The manifest is frozen for repeatability, but
  optimizations must not specialize to this seed, these exact request lengths,
  or Qwen3.5-0.8B hidden/head dimensions.
- Latest current route artifact:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/current_all_bench_20260604/random/random_current_optimized_r1_20260604.json`.
- Current route settings are BF16 activations/weights, packed paged prefill,
  `max_num_batched_tokens=2048`, `num_kvcache_blocks=2048`,
  `max_blocks_per_seq=512`, static decode metadata, BF16 decode projections and
  LM head, BF16 full-attention KV cache, and Triton FLA GDN prefill/decode.
  Throughput is `383.96 output tok/s` with zero measured-phase JIT growth.
- Live vLLM BF16 on the same manifest reaches `1531.33 output tok/s`, so the
  random-suite ratio is `0.251x`. This is much worse than hetero8 and should be
  treated as a separate stress benchmark, not as a solved serving claim.
- Generated lengths match. Generated-token parity is approximate rather than
  exact: the current route matches vLLM on 11 of 15 rows, and earlier
  diagnostics showed that some early divergences are close-logit/tie-sensitive
  rather than obvious state corruption. For this random lane, use approximate
  parity as the gate: complete generated lengths, stable request accounting,
  most rows matching vLLM, and any divergent rows inspected enough to rule out
  capacity/state bleed. Keep exact generated-token parity as the gate for
  deterministic `hetero8`, `short_32_128`, `decode_heavy_128x128`, and
  long-prefill correctness runs.
- Random timing is decode dominated: latest current run spends about `2.35 s`
  in prefill scheduler steps, `26.62 s` in decode scheduler steps, and `0.51 s`
  in final drain. The main hill-climb target is therefore decode throughput and
  tail-bucket efficiency, not seed-specific prefill tuning.
- Rejected random routes from the current pass:
  - device-resident block-table table:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_block_table_table_r1_20260604.json`,
    `255.11 output tok/s`, `0.166x` live vLLM. It removed host block-table
    rebuild pressure but made the integrated decode path much slower.
  - larger B16 serving capacity:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_b16_3072blocks_320width_r1_20260604.json`,
    `229.87 output tok/s`, `0.150x` live vLLM. Fewer waves did not compensate
    for the larger decode bucket and wider capacity envelope.
  - waiting-admission reordering by declared output budget:
    `random_lpt_admission_r1_20260604.json` reached `267.57 output tok/s`
    and `random_spt_admission_r1_20260604.json` reached
    `249.39 output tok/s`. The policies changed active-batch timelines but
    made prefill/context-dependent decode slower, so FIFO remains selected.
  - cached full-attention `BTHD` layout:
    `random_cached_attention_bthd_r1_20260604.json` reached
    `259.21 output tok/s`. Avoiding source transposes around RoPE/KV write did
    not improve the lowered XLA graph and made prefill/decode wall time worse.
- The run exposed and fixed real harness/server blockers: false scheduler
  capacity exhaustion when waiting admission was blocked, duplicate K/V snapshot
  allocation, prompt-tail block hash recording, and carry-map clearing when a
  finished row was released.
- Warmup requirement: use the standard generic server warmup over the configured
  serving grammar. Do not use request-specific warmup for speed claims, and do
  not compile only the exact shapes of a benchmark manifest.
- Generality requirement: accepted changes must be expressed in terms of
  serving grammar, batch buckets, token buckets, block tables, model config
  metadata, or backend capability. Do not hard-code request lengths, seed
  `1234`, active-batch timelines, Qwen3.5-0.8B dimensions, or A10G-specific
  tile constants as the default route. If a kernel needs tile choices, route
  them through a model/hardware-independent selection policy with safe fallback
  diagnostics, not benchmark-specific flags.
- Next execution order:
  1. Profile the latest random route with generic warmup and no measured-phase
     JIT growth.
  2. Attribute the `26.62 s` decode bucket into model GEMMs, GDN decode,
     full-attention decode, LM head, scheduler/table movement, and PjRT/CPU
     gaps.
  3. Implement the largest general decode-side change first. Candidate classes
     are broader decode fusion/grouped layer boundaries, better small-batch
     projection/LM-head lowering, and scheduler/backfill changes that keep
     active decode batches fuller without specializing to a manifest.
  4. Re-run the random stress target plus `hetero8`/`decode_heavy_128x128` to
     catch regressions on deterministic lanes.

## Current Main Path: Packed Paged Chunked Prefill

This is now the primary optimization path, not a side experiment.

The serving ABI should represent chunked prefill as packed ragged query tokens
plus paged-cache metadata:

```text
prefill:
  tokens:          [1, token_bucket]
  positions:       [1, token_bucket]
  token_row_ids:   [1, token_bucket]
  query_start_loc: [request_bucket + 1]
  block_tables:    [request_bucket, max_blocks_per_seq]
  seq_lens:        [request_bucket]
  slot_mapping:    [1, token_bucket]

decode:
  tokens:          [batch_bucket, 1]
  positions:       [batch_bucket, 1]
  block_tables:    [batch_bucket, max_blocks_per_seq]
  seq_lens:        [batch_bucket]
```

Dense prefill remains useful as a correctness comparison path, but it is no
longer the target serving ABI. The old dense shape grammar compiled a Cartesian
product of `batch_bucket * max_query_bucket`, which caused padded work, large
JIT cache surfaces, and benchmark-specific warmup pressure. The packed ABI makes
prefill shape depend on total scheduled chunk tokens and metadata rows instead.

Execution order:

1. Land the scheduler/executor/model contract for packed paged prefill while
   preserving the existing `BlockManager`, block tables, slot mapping, physical
   KV cache, and decode fast paths.
2. Make reference packed full-attention prefill and segmented GDN prefill exact
   against the old dense path on mixed prompt chunks.
3. Warm only the finite serving grammar: `prefill_token_buckets` for packed
   prefill and `batch_size_buckets` for decode. Runtime JIT cache growth during
   benchmarks should be treated as a failure.
4. Replace reference bodies behind the same ABI:
   - full attention: packed query + paged KV prefill kernel;
   - GDN: vLLM/FLA-style varlen packed chunk kernel using `query_start_loc`;
   - decode: existing fixed-width paged decode buckets and GDN state-table path.
5. Benchmark hetero8 and random sidecar against live vLLM and the latest stored
   accepted baseline. Do not interpret a random-suite loss as a win just because
   a microbenchmark improved.

Current validation:

- Entry 221 ran a packed ABI GPU smoke with generic warmup and
  `--fail-on-jit-cache-growth`. Warmup compiled `8` entries and the measured
  phase created `0` additional JIT entries.
- The follow-up config-first smoke passed `--greedy-token-fastpath` and
  `--device-token-carry` as normal benchmark/engine config fields. It again
  had `0` measured-phase JIT cache growth, with `greedy_token_fastpath=True`
  and `device_token_carry=True` recorded in the artifact run config.
- The smoke confirms the finite packed grammar is executable through the normal
  config path. It is not a speed claim: the packed GDN and full-attention
  prefill bodies are still reference implementations, so the next optimization
  work must replace those bodies behind the same ABI.
- Accepted serving controls should be expressed as engine config fields
  (`greedy_token_fastpath`, `device_token_carry`, `static_decode_metadata`,
  `static_decode_seq_lens_carry`) rather than new hot-path env gates. Legacy env
  overrides remain only for compatibility and diagnostics.

Model-specific assumptions to track:

- The ABI is model-family general across Qwen3.5 dense sizes because it depends
  on token buckets, block size, head dimensions, and layer metadata, not hand
  tuned GEMM parameters.
- Kernel implementations still need model-shape validation for the 0.8B, 4B,
  and 27B dense configurations because head counts and hidden sizes affect
  tile choices and available memory.
- GDN semantics must stay segmented per request. Packed tokens may be adjacent
  in memory, but recurrence and convolution history cannot bleed across
  `query_start_loc` boundaries.

## Hard Rules For The Agent

```text
1. Do not merge deterministic-lane speed changes without exact generated-token
   parity. For the random sidecar only, approximate parity is acceptable when
   generated lengths are complete, most rows match vLLM, and divergent rows are
   consistent with known close-logit/tie-sensitive behavior rather than
   capacity/state bugs.
2. Do not compare against stale baselines; compare against Entry 045 or the latest accepted baseline.
3. Do not optimize MTP for speed yet.
4. Do not implement more source-level JAX rewrites unless HLO/profile evidence says they target a real bottleneck.
5. Do not accept microbenchmark-only wins.
6. Do not add per-layer layout conversions to use an external kernel.
7. Keep every external kernel behind a backend flag and fallback path.
8. Record rejected experiments. Rejected experiments are useful evidence, not failure.
9. Maintain both accepted-baseline and fastest-achieved records for tracked workloads.
10. For GDN, target V,K serving state; do not preserve K,V as the serving ABI merely because the old JAX path used it.
11. Keep BF16 GDN prefill activations as a separate opt-in experiment after V,K correctness is established.
12. Do not use local CUDA probes as the optimization path. They are historical diagnostics only. Use FlashInfer for paged KV/attention and vLLM/FLA-derived kernels for GDN unless those routes are explicitly blocked.
```

## Expected Strategic Outcome

The path is:

```text
Clean baseline
-> stable benchmark matrix
-> GDN V,K state migration with pure-JAX fallback updated
-> FlashInfer paged KV/attention where dtype/layout gates allow
-> vLLM/FLA-shaped GDN decode recurrence
-> vLLM/FLA-shaped segmented GDN prefill
-> only then revisit MTP or finer fusions
```

The core bet is: keep the JAX model and correctness harness, but align the GDN
state ABI with kernel-native V,K layout, then replace the few serving kernels
where vLLM/FlashInfer/FLA have structural advantage. Local CUDA probes are
historical diagnostics only; production speed work should come from FlashInfer
paged-attention/KV kernels and vLLM/FLA-style GDN kernels.

## Reference Links From The Proposal

- Optimization logbook:
  <https://raw.githubusercontent.com/LiquidGunay/nano-vllm-jax/main/docs/optimization_logbook.md>
- Qwen/Qwen3.5-0.8B model card:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B>
- FlashInfer `append_paged_kv_cache`:
  <https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html>
- vLLM cache kernels:
  <https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/cache_kernels.cu>
- FlashInfer on JAX with TVM FFI:
  <https://docs.flashinfer.ai/tutorials/generated/jax_tvm_ffi/index.html>
- vLLM paged attention:
  <https://docs.vllm.ai/en/latest/design/paged_attention/>
- FlashInfer attention kernels:
  <https://docs.flashinfer.ai/api/attention.html>
- vLLM Qwen3-Next support note:
  <https://vllm.ai/blog/2025-09-11-qwen3-next>
- Flash Linear Attention:
  <https://github.com/fla-org/flash-linear-attention>
- JAX Pallas docs:
  <https://docs.jax.dev/en/latest/pallas/index.html>
- MaxText Pallas performance guide:
  <https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html>
- Ragged Paged Attention paper:
  <https://arxiv.org/abs/2604.15464>
