# MTP Speculative Decoding

MTP is experimental in this repository. This page records the current GPU serving posture plus TPU-era K=1 semantics and benchmark findings. For current GPU correctness, use [GPU correctness guardrails](gpu_correctness_guardrails.md) and [benchmarks](benchmarks.md).

## Current MTP Direction (2026-06-22)

The speed target is a vLLM-style packed verifier route:

- verify `[current, draft_1, ..., draft_K]` as a single target-model packed forward;
- gather verifier and bonus logits by index and compute accept/reject on-device;
- commit only the accepted prefix and corresponding KV/GDN/full-attention state;
- emit only verified tokens.

Any route that emits draft tokens without target verification, repairs accepted-prefix
state on host with sequential decode, or reports host-side repair/materialization
as progress is a correctness diagnostic only.

Do not describe the current MTP path as production-ready. Current GPU
server-shape artifacts show a real verified speedup only once decode length is
large enough to amortize MTP prefill seeding and the cold first speculative
burst. Treat historical K=1 TPU results as non-normative for current GPU work.

## Current Plan (2026-06-22)

- Keep `k_decode` as the exact same-width verifier oracle for parity only.
- Use packed-prefix row-state verification as the only candidate serving route for
  speed progress: one packed `[current, draft...]` forward, on-device accepted-prefix
  decisions, and no host-side accepted-reject branching during verification.
- Keep `commit_select`, `two_decode`, table-burst checkpoints, and broad/repair-mode
  variants tagged as diagnostic-only until full state parity is proven and the packed
  row-state route is both exact and faster than the accepted non-MTP best path.

## No-Host-Sync MTP Target

The historical implementation checkpoint is a resident speculative decode boundary, not
more Python-side repair of verifier outputs. The hot path should launch a fixed
compiled step that:

- verifies the current draft chain;
- computes the accepted prefix length on device;
- selects the committed KV/hybrid/GDN state at that prefix;
- emits `draft_prefix + target_or_bonus` into a fixed device token matrix;
- regenerates the next MTP draft chain from the committed token;
- advances resident sequence lengths and paging metadata using masks.

The host must not read `accepted`, split rows into accepted/rejected groups, or
run a repair decode before the next speculative group can execute. The current
Python engine may still drain the final emitted token matrix after a burst so it
can update `Sequence` objects and stream results; that drain is not allowed to
control intra-burst commit. A future resident output ring should remove even
that post-burst length drain from the decode scheduling loop.

The concrete speed boundary target is packed-prefix K-token verification:
each admitted row forms `[current, draft_1, ..., draft_K]`, the target model
verifies that short prefix in one packed prefill-shaped forward pass, and the
compiled step selects the committed prefix state per row. This is the only
route that can amortize verifier work enough to be a valid speed target. The
sequential `commit_select` route stays as a diagnostic oracle for correctness and
drift debugging.

2026-06-20 historical checkpoint: resident
seed-then-table burst for K=1. The compiled step seeds the first draft when no
draft is already stored, verifies fixed table-burst groups, computes
accept/reject on device, commits KV plus hybrid state, compacts emitted tokens
into a fixed device matrix, and seeds the next draft before returning to
Python. The host drains one compact per-row summary and deferred token refs
after the burst; those reads no longer decide verification before the next
group can be formed. A B=1 synthetic GPU smoke was exact and exercised this
path, but it remained slower than the no-MTP control, so this is an
architecture/correctness checkpoint rather than a speed claim.

## Historical GPU checkpoint

As of 2026-06-21, the conservative correctness checkpoint used exact K=2
`commit_select` verification. Packed-prefix verification remains the intended
speed boundary, but it is not the serving config because it is not yet
correctness-clean under the strict no-fallback GDN policy. MTP remains an
explicit opt-in serving mode. The server config fields are:

- `speculative_method`: `none` or `mtp`;
- `num_speculative_tokens`: `0` for regular serving; K=2 is the largest
  currently tested true-K checkpoint for diagnostics. K=3 should wait until K=2 has useful
  second-position draft quality;
- `draft_sample_method`: currently only `greedy` is implemented for MTP;
- `mtp_verifier_impl`: `commit_select` for the exact K=1/K=2 sequential
  diagnostic path, `packed_prefix` for the current speed boundary direction,
  `k_decode` for same-width decode parity, and `two_decode`
  for the historical width-2 K=1 verifier;
- `mtp_batch_accept_policy`: `rowwise` or `all_or_none`;
- `mtp_seed_after_bonus`: default `false`.
- `mtp_prefill_seed`: default `false`. Prefill draft seeding is now opt-in
  because the current verified GPU path paid a large TTFT cost without beating
  the no-MTP decode baseline.

Legacy configs that set only `num_speculative_tokens=1` are interpreted as
`speculative_method=mtp` for compatibility. Invalid or unimplemented MTP modes
fail at startup instead of silently taking a partial speculative path.

The current GPU route is correctness-first:

- the target model remains canonical;
- the scheduler marks admitted speculative rows in `ScheduledBatch`;
- generic warmup compiles the configured MTP verifier buckets;
- accept/reject and prefix-state selection happen inside the compiled verifier;
- Python may drain compact emitted-token metadata after a verifier call, but it
  must not choose the accepted prefix before the next device commit is valid;
- packed-prefix verification borrows the prefill-shaped full-attention route,
  not the width-1 decode route;
- the strict MTP diagnostic path disables GDN fallbacks and uses explicit
  packed GDN decode `reference` with BF16 QKV rather than silently selecting
  `off`;
- `benchmarks/configs/gpu_mtp1_two_decode.json` uses the reference
  full-attention decode backend for the width-2 verifier until a real width-2
  FlashInfer or equivalent verifier exists.

Do not use the new MTP config as a speed-claim artifact yet. It is the first
clean GPU implementation target for measuring acceptance, verifier cost, and
whether a width-2 verifier can beat the current non-speculative kernel path.

2026-06-21 follow-up: K=2 `commit_select` is now conservative for partial
acceptance. A raw `[accept, reject]` K=2 row is exposed to the runner as a
one-token target commit rather than a two-token partial-prefix commit. The
partial-prefix skip was the source of row drift in the two-request smoke: the
fused seed-main path matched the normal greedy path exactly on emitted tokens,
current KV slots, and hybrid state, but partial K=2 commits later diverged.
With partial commits disabled, the same two-request Qwen3.5-0.8B smoke is
exact through 12 and 16 output tokens:

- `commit_select_fullonly_jax_12.json`: exact through 12 tokens, `15.71`
  output tok/s, acceptance `0.091`;
- `commit_select_fullonly_jax_16.json`: exact through 16 tokens, `15.96`
  output tok/s, acceptance `0.214`.

This restores a correctness oracle; it is not a speed path.

The same pass also isolated the broader verifier candidates:

- `k_decode_force_reject_fixed_jax_8.json` is exact when every K=2 draft is
  forced to reject, so the decode-shaped K verifier can reproduce the next
  target token for one-token commits.
- `k_decode_retest_fixed_jax_16.json` still diverges when accepted K=2 draft
  tokens are committed. The first row drifts at token index 9, which means the
  width-3 verifier's accepted-prefix state is not equivalent to sequential
  decode state.
- `packed_prefix_force_reject_fixed_jax_8.json` diverges even with zero accepted
  drafts. Row 0 expects `12` at index 5 but the packed-prefix target pass emits
  `220`; `12` is only second in the packed verifier top-k.
- A packed-prefix run with config-owned reference GDN prefill
  (`packed_prefix_force_reject_gdn_ref_config_jax_8.json`) also diverges, so the
  issue is not limited to the Triton packed-prefix kernel.

Historical implication: use sequential `commit_select` as the exact K=2 diagnostic oracle.
Do not promote packed-prefix or width>1 decode accepted-prefix commits until the
target logits and committed GDN/full-attention state match sequential decode.

2026-06-21 bridge checkpoint update: the temporary exact repair guard was
removed. K=2 `mtp_verifier_impl=k_decode` and
`mtp_verifier_impl=packed_prefix` must now exercise the actual grouped verifier
boundary. `commit_select` remains the correctness oracle for diagnostics, but
it is no longer a runtime fallback for K=2 grouped verification. The current
known failure is grouped-state drift after accepted groups: the grouped verifier
can match accept decisions while KV, conv, and recurrent state still differ
from sequential decode. Packed-prefix also uses prefill-shaped projection/GDN
math, so it can diverge earlier than decode-shaped `k_decode` until that math
is aligned with width-1 decode or intentionally treated as approximate.

Historical bridge plan (diagnostic):

1. Treat `commit_select` as the correctness oracle because it runs the target
   model as sequential decode steps and is exact on the small GPU smoke.
2. Make `k_decode` a same-width decode oracle, not a speed path. For K=2 it
   has the same logical verifier boundary as packed-prefix
   (`[current, draft_1, draft_2]`) while still using decode-style cache/state
   semantics.
3. Add a focused parity probe that compares sequential `commit_select`,
   `k_decode`, and packed-prefix from the same pre-state. The probe must report
   target tokens/top-k, hidden-state differences, KV slot differences, GDN
   conv/recurrent state differences, accepted-prefix length, emitted tokens, and
   committed seq-lens.
4. Fix `k_decode` until its accepted-prefix committed state matches
   `commit_select`. This establishes the exact same-width contract.
5. Use that contract to drive packed-prefix parity work. Packed-prefix is only viable as a speed path once
   its target logits and selected prefix state match the same-width decode
   oracle; until then, keep accepted-prefix repair and broad-mode experiments
   diagnostic-only.

Strict runtime check: `k2_strict_runtime_no_parity_jax_12.json` ran K=2 MTP
with parity/layerwise probes disabled. The profile only shows
`executor_mtp_k_decode` for K verifier steps; there are no `commit_select`
labels in the runtime path. Counters were `fallback_steps=0`,
`fallback_partial_rows=0`, four resident grouped commits, and acceptance
`4/12`. The remaining fallback counters are initial missing-draft seeding and
the final `max_tokens` tail where there is no token budget for K+bonus
speculation, not sequential verifier repair.

2026-06-21 strict-K follow-up: `k_decode` now keeps K verification inside one
compiled boundary while statically unrolling width-1 target decode steps inside
that boundary. This removes the Python/sequential verifier fallback while
matching ordinary decode state semantics for full physical batches. Scheduler
adaptive admission is disabled for strict K>1 verifier configs so a verifier
development run cannot silently gate itself off after one probe.

The focused B=2, 16-token smoke now has an exact strict-K artifact:
`k2_kdecode_exact_unroll_skip_partial_b2_len16.json` matches the same-shape
no-MTP reference exactly with no measured-phase JIT cache growth. The actual
K verifier runs on full B=2 steps and reports acceptance `10/32` (`31.25%`).
Partial mixed tail K verifier batches are intentionally skipped for now:
`k2_kdecode_exact_unroll_admit_b2_len16.json` showed that the compact partial
K commit corrupts the following single-row state at row 1 token index 11. This
skip is a correctness guard, not a speed path. Throughput remains far below
baseline (`31.44 output tok/s` vs `110.32` no-MTP on this smoke), so the next
real speed blockers are verifier cost and draft quality, not fallback removal.

2026-06-21 no-fallback update: compacting was not the only bug. A masked
physical partial-batch grouped verifier still diverged at the same row/index
when it committed a partial K prefix. The current strict grouped K decode
contract is therefore conservative: the executor still runs
`mtp_k_decode_greedy_step_jit`, but for K>1 it commits accepted drafts only
when the whole K-token block verifies. Partial accepts emit one target-model
token and leave the remaining state to the next decode step. The exact GPU
checkpoint is `k2_kdecode_fullonly_no_fallback_b2_len16.json`: it matches the
same no-MTP reference through both 16-token rows, shows `fallback_steps=0`, and
profiles masked partial rows as `mtp_k_decode_masked_partial` rather than
`commit_select` or sequential repair. It is slower (`19.73 output tok/s`,
`21.05%` committed draft acceptance) than the skip-partial guard and much
slower than no-MTP (`110.32 output tok/s`), so this is the fail-closed
correctness bridge for building a real packed verifier, not a speed claim.

2026-06-21 current-reference update: after verifier/config changes, the old
`no_mtp_same_shape_b2_len16_after_kv_reset.json` reference no longer represented
the current runtime. Regenerate a same-code no-MTP reference before judging MTP
correctness. With `no_mtp_current_b2_len16.json` as the reference:

- `kdecode_prefill_seed_strict_force_reject_currentref_b2_len16.json` is exact
  under forced rejection, has `fallback_seeded_main_steps=0`, and keeps the
  K=2 verifier on `mtp_k_decode`. This is the diagnostic fail-closed oracle that
  proves missing-draft seeded-main repair is gone.
- `kdecode_prefill_seed_verified_currentref_b2_len16.json` is exact but the
  default `pre_norm/raw/recursive` draft convention accepts only `2/52` drafts.
- `kdecode_prefill_seed_verified_finalnorm_sequence_currentref_b2_len16.json`
  is exact and improves draft quality to `12/36` accepted drafts with six bonus
  tokens. It is still not a speed path: once rows desynchronize, masked partial
  verifier batches spend large time in host/hybrid-state gather/store work.
- `k1_twodecode_prefill_seed_verified_finalnorm_currentref_b2_len16_nowarmup.json`
  diverges at row 1 token 7. Do not promote the K=1 `two_decode`/table route
  until its prefix-state commit path is parity-proven.
- `kdecode_prefill_seed_verified_finalnorm_sequence_slotfix_currentref_b2_len16.json`
  keeps exactness after preserving `hybrid_slot_ids_host` through masked
  verifier batches. It improves the smoke modestly (`4.22 -> 4.54 output
  tok/s`, mean ITL `85.1 -> 63.0 ms`) but does not remove the bottleneck:
  masked partial-row batch construction and hybrid-state gather/store are still
  large.

Historical implication: keep `configs/server/mtp_experimental.yaml` checkpoints on
K=2 `k_decode`, prefill draft seeding, and final-normed sequence drafts for
diagnostics only. The next performance task is not another fallback removal;
it is a partial-row verifier boundary that consumes resident slot ids/state
directly, or a packed verifier whose state matches the `k_decode` oracle.

## GPU 2026-06-22 packed verifier plan

The current packed-verifier speed direction is:

1. Treat `k_decode` as the exact semantic oracle for target-model verification.
2. Make `mtp_verifier_impl=packed_prefix` exact before optimizing it. The route
   must use an on-device scanned verifier that repeats the width-1 decode cell
   inside one JIT boundary, with no Python fallback and no host-side accept
   decision.
3. Keep broad packed decode/prefill verification diagnostic-only until accepted
   rows commit GDN and KV state exactly like the scanned oracle. Forced-reject
   repair is not sufficient; full-accept state parity is the blocker.
4. Once the exact scanned packed route is correctness-clean, replace individual
   expensive pieces with packed work only when parity remains intact. Candidate
   pieces are verifier logits, KV slot commit, and GDN state commit.
5. Report speed only against the current best non-MTP path and the exact scanned
   MTP route. A broad verifier result that diverges from `k_decode` is not a
   valid throughput win.

Implementation checkpoint:

- `mtp_verifier_impl=packed_prefix` now routes through an exact on-device scan
  verifier (`verify_mode="scan"`) rather than the broad accepted-state verifier.
- The exact scan keeps the K width-1 decode rollout inside one JIT boundary and
  skips the broad-route reject repair, because scan state is already exact for
  full rejects and full accepts.
- `packed_prefix_exact_scan_verified_b2_len16.json` fixed the earlier token-10
  accepted-state divergence. `packed_prefix_exact_scan_norepair_donate_b2_len16.json`
  is also exact, has no measured JIT growth, accepts `12/36` drafts, emits six
  bonus tokens, and reaches `4.38 output tok/s` on the small B=2 len-16 smoke.
- This is correctness progress, not a serving speed win. The route is still
  close to, but slightly below, the prior exact `k_decode` artifact
  (`4.54 output tok/s`) and far below no-MTP on the same small smoke. The next
  speed task is to replace parts of the scanned verifier with parity-proven
  packed work, not to re-enable the divergent broad accepted-state commit.

## GPU 2026-06-20 true-K status

The exact true-K route now has the first strict B=2 smoke checkpoint:

- `mtp_verifier_impl=k_decode`, `num_speculative_tokens=2`,
  `mtp_burst_groups=1`, generic warmup, strict GDN no-fallback settings, and
  compact-tail verifier rows are correctness-clean against the no-MTP reference;
- compact verifier batches now preserve host seq/slot metadata and record
  device-token carry with local row ids, fixing the previous tail-row
  divergence;
- the forced K=2 smoke is exact and has no measured-phase JIT growth, but is
  slow: `22.16 output tok/s` versus `100.63` no-MTP on the same B=2 synthetic
  smoke;
- forced K=1 on the same route is also exact and faster than K=2, but still
  loses to no-MTP: `43.41 output tok/s`;
- K=2 draft quality is the current blocker, not an obvious verifier
  correctness bug. Logit debug showed position 0 target in MTP top-5 for `6/6`
  events, but position 1 target in top-5 for `0/6`; offsets `-1` and `+1` did
  not change acceptance.

2026-06-20 follow-up:

- the chain input bug was real: feeding final-normed MTP hidden forward raised
  the strict K=2 B=2 smoke from `22.78` to `26.00 output tok/s` and made the
  second draft position nonzero (`[4, 0] / [10, 10]` to `[4, 1] / [9, 9]`);
- sequence-context chaining improved the draft contract further by running the
  MTP layer over the accumulated draft sequence and top-1ing only the final
  position. Strict K=2 B=2 reached `26.77 output tok/s`, exact parity, and
  position acceptance `[3, 3] / [7, 7]`;
- K=3 sequence-context did not help. Debug acceptance dropped to `3/21`, so
  this checkpoint should stay at K=2;
- K=2 `commit_select` with final-normed sequence drafts is the best exact route
  from this pass: strict B=2 len-12 reached `27.47 output tok/s`, and len-64
  reached `70.31 output tok/s` with exact parity. The len-64 no-MTP reference
  remained much faster at `361.05 output tok/s`;
- the concrete speed blocker is verifier cost, not output materialization. A
  profiled K=2 commit-select run showed steady verifier calls around `25 ms`
  and fused seed around `10 ms`, while adaptive admission measured the B=2
  speculative bucket at `14.9 ms/token` versus `2.1 ms/token` baseline and
  disabled the bucket for low throughput;
- packed-prefix verification is not currently promotable under strict
  no-fallback GDN policy because it needs prefix-state output from the packed
  prefill GDN route and would otherwise use the slow recurrent scan.

Treat K=2 as research-only until the verifier boundary is substantially
cheaper. For Qwen/Qwen3.5-0.8B, exact MTP does not provide a throughput win
over the current non-speculative path even after the draft-chain correctness
fixes. Scheduler admission should remain enabled for serving so slow buckets
fall back to ordinary decode.

The current GPU correctness contract is `dtype=float32` with `weight_dtype=bfloat16` for Qwen3.5 real weights. Under that contract:

- the long-decode top-5 guardrail matches HF for 500/500 decode steps,
- JAX paged generation matches HF across the current server-shape suite,
- exact commit-select MTP1 generated tokens match JAX/HF across the current server-shape suite,
- MTP1 throughput should not be reported as a serving win until it beats the JAX paged baseline on the same correctness-checked shapes.

Current default posture:

- regular serving uses `num_speculative_tokens=0` unless a caller opts into MTP,
- unverified draft append is not a valid serving or benchmark mode; configs
  that request it fail at startup,
- the width-2 K=1 verifier is selected through config rather than
  `NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1`,
- the exact commit-select path remains the correctness reference for focused
  MTP1 tests and diagnostics,
- scheduler-gated MTP decode falls back to the ordinary static/resident
  metadata hot path when no rows are admitted,
- forced MTP benchmarks are diagnostics, not serving guidance, until they pass generated-token parity and measured decode speedup.

## Historical TPU validated state

Validated on 2026-05-09 through 2026-05-11 repo state:

- hardware: TPU v6e-1,
- models: `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B`, BF16, real weights,
- execution: JIT on the pure JAX/XLA TPU backend,
- correctness test: `tests/test_mtp_commit_semantics.py` passed `16/16` on TPU,
- serving policy: K=1, scheduler-owned admission, latency EWMA gate.

Remaining serving gap:

```text
The measured-latency EWMA gate is global, not per bucket.
```

## Terminology

For K=1:

- target token: the token sampled from the canonical target model for the current decode position,
- draft token: the token proposed by the MTP head from target-model hidden state,
- bonus token: a draft token that is accepted and emitted after the target token in the same scheduler step,
- reject: commit only the target token,
- accept: commit the target token and the bonus token.

Avoid using K=2/K>1 terminology when describing the serving path unless explicitly discussing experimental work.

## K=1 commit invariants

Reject case:

```text
old prefix + target token
```

Accept case:

```text
old prefix + target token + accepted bonus token
```

Required invariants:

- target logits come from the canonical target forward-step contract,
- an accepted bonus token must match target-model verification semantics,
- rejected draft writes must not become logically reachable,
- full-attention KV and hybrid linear-attention state must advance by exactly the committed prefix,
- inactive bucket rows must preserve prior state,
- block allocation must include target and admitted lookahead positions before execution.

## Scheduler-owned admission

The scheduler owns whether MTP is admitted for a row. Admission should consider:

- whether the sequence is in decode, not prefill,
- whether sufficient lookahead KV capacity is allocated,
- recent acceptance behavior,
- measured decode-latency EWMA compared with baseline,
- bucket and shape eligibility.

The TPU implementation described below gated with acceptance and measured latency, but its latency EWMA was still global. Treat per-bucket EWMA as a historical TPU follow-up item unless it is revalidated on GPU.

## Historical 2026-05-11 K=1 speed status

The corrected K=1 path is exact but does not yet beat baseline when MTP is forced on.

Validated TPU v6e-1 results, `Qwen/Qwen3.5-0.8B`, BF16, real weights, JIT, warmed shapes:

- homogeneous B=4, prompt length 16, output length 16: exact token match passed, next-step sanity passed, baseline decode `353.57 tok/s`, forced K=1 decode `265.27 tok/s`, speedup `0.750x`, acceptance `62.5%`.
- homogeneous B=16, prompt length 16, output length 16: exact token match passed, next-step sanity passed, baseline decode `772.84 tok/s`, forced K=1 decode `510.14 tok/s`, speedup `0.660x`, acceptance `57.1%`.
- homogeneous B=4, `Qwen/Qwen3.5-4B`, prompt length 16, output length 16: exact token match passed, next-step sanity passed, baseline decode `209.33 tok/s`, forced K=1 decode `161.44 tok/s`, speedup `0.771x`, acceptance `62.5%`.
- high-acceptance B=1, `Qwen/Qwen3.5-4B`, synthetic prompt suite, prompt length 16, output length 16: exact token match passed, next-step sanity passed, baseline decode `69.91 tok/s`, forced K=1 decode `79.17 tok/s`, speedup `1.132x`, acceptance `87.5%`.
- same high-acceptance B=1 4B case with measured-speed gate enabled: exact token match passed, baseline decode `68.82 tok/s`, gated K=1 decode `77.85 tok/s`, speedup `1.131x`; the scheduler kept MTP enabled with measured speedup `1.229x`.
- mixed/interleaved B=4, prompt lengths `16,17,31,32`, arrivals `0,0,2,4`, output length 12: exact token match passed, next-step sanity passed, baseline decode `238.71 tok/s`, forced K=1 decode `141.88 tok/s`, speedup `0.594x`, acceptance `38.9%`.
- measured-speed gate with `NANO_VLLM_JAX_MTP_MIN_SPEEDUP=1.0`: exact token match passed and decode throughput was effectively parity, `362.76 tok/s` baseline vs `362.72 tok/s` gated K=1, because admission disabled speculative decode after measured throughput was below threshold.

Historical blocker:

```text
Accepted K=1 steps can be faster per emitted token on 4B/B=1.
Rejected and fallback K=1 steps are much slower per emitted token, so forced MTP
needs a high acceptance rate to break even.
```

Do not seed a follow-up K=1 draft after a rejected row unless the rejected-row next-draft state invariant is proven. The historical safe policy commits the target token and leaves no draft behind for rejected K=1 rows.

The historical MTP research target at this stage was a coarse safe verifier whose
width-2 target-model forward is materially cheaper than two width-1 forwards.
The after-current-token prefix state is now exposed by
`return_first_prefix_hybrid` in focused coverage, but the corrected one-pass GPU
smoke is still slower than both `k_decode` and no-MTP.

## GPU 2026-06-15 resident-table K=1 verifier

That checkpoint's active K=1 verifier is an exact resident-table two-decode
boundary.
Instead of returning a compact per-row hybrid state and asking Python to store
it, the compiled verifier:

- gathers the active rows from the resident hybrid/GDN state table;
- runs the target model on `[current_token, draft_token]`;
- compares the verified target token with the draft token on device;
- selects the after-current state for rejected rows and after-draft state for
  accepted rows;
- scatters the selected state back into the resident table inside the JIT;
- emits a fixed two-token row summary plus accepted counts for the scheduler.

This route keeps target-model verification exact and removes the old Python
hybrid-state writeback boundary. On the two-request smoke manifest with
`final_normed` MTP hidden state and Triton MTP top-1, it is exact,
JIT-cache-stable, and accepts `12/13` drafts. Throughput improved from
`22.52 output tok/s` for the compact one-pass verifier to `28.23 output tok/s`,
roughly matching `k_decode` at `28.97 output tok/s`.

That is not a serving win yet: the no-MTP control on the same smoke is
`47.25 output tok/s`. The next blocker is therefore the surrounding serving
loop, not another narrow GDN kernel swap. The current profiling target is to
remove or move host-facing work around verified MTP groups: token prefetch and
materialization, `Sequence`/block-manager commit bookkeeping, admission updates,
and first-use/final-drain synchronizations. Any future speed claim must stay on
the exact verified route and keep zero measured JIT-cache growth.

Prefill seeding is not a speed route in this checkpoint. The `entry325` diagnostic moved
the seed cost into prefill (`~628 ms` TTFT) and reached only
`29.37 output tok/s`, below the non-prefill-seeded exact table burst2 run at
`31.45 output tok/s`. Decode-side seed-plus-table-burst was tested next and is
also rejected as a default: seed plus two verifier groups reached
`27.70 output tok/s`, and seed plus one verifier group reached
`29.11 output tok/s`. The checkpoint route remains the non-prefill-seeded exact
resident-table burst verifier while the next pass isolates the first seed
execution and steady-state verifier model cost.

Break-even direction:

- With the historical safe path, increasing from 0.8B to 4B did not materially lower the threshold. The 4B accepted-step per-token latency was approximately baseline latency, while rejected rows remained about 2x baseline latency.
- Therefore the historical 4B forced-K=1 break-even acceptance threshold was effectively at or above 100%. A larger model alone was not enough; the accepted verifier path had to become cheaper than two baseline decode steps and rejected/fallback rows had to be gated or repaired cheaply.

Batch-shape correctness note:

- B=16 homogeneous forced K=1 diverged from the B=16 baseline when width-2 decode matmuls used their native shape.
- Setting `NANO_VLLM_JAX_FORCE_WIDTH1_DECODE_MATH=1` restored exact token match and next-step sanity at B=16.
- This is now the default for multi-token decode. It preserves the canonical width-1 baseline contract but raises the forced-MTP speedup threshold.
- Hybrid prefix-cache execution is disabled, so allocation must also avoid sharing content-addressed full blocks. Otherwise repeated prompts in the same prefill wave can write duplicate rows into one physical KV block. The scheduler now passes `use_prefix_cache=False` to allocation when prefix-cache execution is disabled; blocks still keep local hashes for decode append bookkeeping, but they are not inserted into the shared prefix-cache map.

## Historical K=2 status

K=2 is correctness-clean in focused TPU semantics testing, but slower in observed serving benchmarks. Treat it as experimental and non-serving.

K=2 should stay disabled for serving until it has:

- robust long-generation exactness across workloads,
- per-bucket latency evidence showing a real throughput benefit,
- clear state-commit handling for partial acceptance,
- benchmark coverage comparable to K=1.

## GPU 2026-06-14 K>1 status

Exact K>1 verification is correctness-clean on the focused GPU probes but is
not a speed path.

- Warmup state reset fixed the short K=2 packed-prefill first-token drift:
  `short8_verified_k2_prefill_generic_after_reset_r4.json` matched the no-MTP
  short8 rows exactly.
- The same K=2 packed-prefill run was only `15.81 output tok/s` at `25%`
  acceptance; the no-MTP short8 control was `74.69 output tok/s`.
- Generic K=2 decode verification with in-JIT result packing stayed
  correctness-clean but reached only `16.75 output tok/s` on the mixed short
  probe.
- On a high-acceptance B=1 repeated-token stream, exact K=2 accepted `87.5%`
  of drafts and still reached only `34.69 output tok/s` versus `63.99 output
  tok/s` no-MTP.
- Exact K=4 on the same B=1 stream reached `37.50 output tok/s` versus
  `109.85 output tok/s` no-MTP. Even the assume-accepted upper bound, valid
  only as a diagnostic on the zero-rejection stream, reached only `52.53
  output tok/s`.

Current diagnosis: the verifier boundary is dominated by width-K target-model
work, MTP-head chaining, hybrid state gather/store, and first seed/probe cost.
Avoiding the small host acceptance transfer is not enough. Do not promote or
retry K=2/K=4 generic decode/packed-prefill verification as serving speed
routes without a new device-side fixed-output/repair boundary and evidence that
the verifier plus MTP-head work is cheaper than ordinary decode.

## GPU 2026-06-14 random-large K=1/K=2 status

The random-large benchmark currently blocks before a useful verified MTP
speedup comparison:

- no-MTP under the accepted random-large envelope reaches `817.85 output
  tok/s`, `0.801x` of the stored vLLM denominator, with no JIT-cache growth;
- best-path verified K=1 with materialized tied LM head trips the RAM guard
  before measurement, at `80.1-82.2%` system RAM and `7.47-7.86 GB` child RSS;
- disabling materialized tied LM head makes K=1 memory-safe but leaves it in
  CPU-side warmup/compile for about ten minutes with `0%` GPU utilization;
- diagnostic K=2 without materialized tied LM head timed out after `300 s`
  before measurement, with output ending after weight load.

This means random-large MTP is currently blocked by startup/compile and host
memory surface before acceptance-rate economics can help. Treat K=1/K=2 random
serving as non-promoted until the MTP verifier compilation boundary and
materialized LM-head duplication are reduced.

## GPU 2026-06-15 no-host-sync K-burst status

The first no-host-sync K-burst commit boundary is implemented, but it is not a
serving speed path yet.

- The compiled burst now computes accepted prefix length, selects prefix
  KV/hybrid/GDN state, emits fixed token/count tensors, regenerates drafts, and
  advances committed sequence lengths without Python row repair between burst
  groups.
- Focused tests pass, including mixed-reject K=2 semantics without repair.
- A guarded two-row random smoke with exact K=2, burst groups `2`, and
  `final_normed` hidden was cache-stable but slow:
  `23.11 output tok/s` versus `96.15` no-MTP (`0.240x`), with only `2/42`
  accepted drafts.
- Repeating the same smoke with `pre_norm` hidden kept acceptance at `2/42` and
  was slower (`19.62 output tok/s`).

Current diagnosis: intra-burst host-sync removal worked, but random-request
MTP is dominated by low draft acceptance plus width-K target verification and
MTP-head chaining. Keep MTP off the best random/hetero serving path until a
better draft contract/model or confidence gate makes the verifier cheaper than
ordinary decode for the active bucket.

## GPU 2026-06-22 packed verifier status

`k_decode` remains the exact oracle for K>1 grouped verification. The latest
packed/broad verifier pass established a narrower correctness boundary:

- Forced rejection can be made exact without host fallback by repairing rejected
  rows inside the JIT with one exact width-1 target decode. This fixes the
  committed current-token KV/GDN state for rejected rows.
- True flattened packed prefill is slower on the smoke (`~47 ms` steady
  verifier step) than decode-shaped broad verification (`~36-37 ms`).
- Verified mode is still not exact. When a row fully accepts, the broad
  verifier commits broad GDN/KV state; that state drifts from sequential
  width-1 decode and later changes the token stream.

Current artifacts:

- `packed_decode_reject_repair_force_reject_b2_len8.json`: exact forced-reject
  decode-shaped broad verifier.
- `packed_decode_reject_repair_verified_b2_len16.json`: diverges at row 1
  token index 10 with `10/36` accepted drafts.
- `packed_prefill_reject_repair_force_reject_b2_len8.json`: exact forced-reject
  flattened packed-prefill verifier, but slower.
- `packed_prefill_reject_repair_verified_b2_len16.json`: same full-accept
  state divergence.

Do not claim a packed-prefix speed win from these runs. The next required fix
is full-accept state parity against `k_decode`. Repairing accepted rows with an
exact sequential rollout is correctness-safe but removes the expected speed
advantage, so it is a diagnostic fallback, not the target design.

Follow-up on 2026-06-22: exact packed-prefix verification first routed through
the scanned verifier and remained correctness-clean after the burst-loop
assembly fix. The smoke
`packed_prefix_exact_scan_burst1_after_loopfix_b2_len8.json` matched the
no-MTP reference prefix and measured `2.11 output tok/s`, in line with the
earlier exact scan smoke. Exact-scan `mtp_burst_groups=2` remained too slow on
the B=2 length-16 smoke and should not be used as a speed route. The active
strict route has since moved to the resident-table packed-prefix verifier with
`NANO_VLLM_JAX_MTP_TABLE_TARGET_MODE=decode_rectangular`; burst groups are
allowed there only when generic warmup covers full-burst and masked-tail routes
and `--fail-on-jit-cache-growth` confirms the measured phase does not compile.

Second follow-up on 2026-06-22: exact K verification now has a no-bonus mode
enabled by `NANO_VLLM_JAX_MTP_DISABLE_BONUS=1` in
`configs/server/mtp_experimental.yaml`. In this mode the verifier emits only
accepted draft tokens or the fallback target token. It does not compute or emit
the speculative bonus token. More importantly, accepted K-token blocks now use
normal decode carry semantics: the last emitted token is carried as the next
decode input, so the verifier only runs K target forwards and commits KV/GDN
state through token `K-1` of the verifier input stream. The previous no-bonus
prototype still ran K+1 target forwards and over-committed state through the
last emitted token.

Current artifacts:

- `no_mtp_b2_len8_current_for_mtp_no_bonus.json`: current no-MTP reference,
  `78.60 output tok/s` on the tiny B=2 len-8 smoke.
- `mtp_packed_prefix_exact_scan_no_bonus_b2_len8.json`: first no-bonus smoke
  before the carried-token state fix, `15.21 output tok/s`, unchecked against a
  reference.
- `mtp_packed_prefix_exact_scan_no_bonus_kforward_b2_len8.json`: corrected
  no-bonus K-forward verifier, exact generated-token match against the no-MTP
  reference, `16.26 output tok/s`, `6/18` accepted drafts, `0` bonus tokens.

This is a real verifier-cost reduction for the exact path, but it is still not
a serving speed win. The B=2 smoke remains far slower than no-MTP because draft
acceptance is low and the exact verifier still runs target-model forwards plus
MTP-head chaining. The next speed route remains broad packed-prefix parity:
use the tiny packed GDN prefix-state kernel only when full-accept KV/GDN state
matches the exact K-forward oracle.

Third follow-up on 2026-06-22: broad packed-prefix decode verification can now
be selected explicitly with `NANO_VLLM_JAX_MTP_K_VERIFY_MODE=decode` while
`mtp_verifier_impl=packed_prefix` remains the entry point. In no-bonus K=2
mode, the broad verifier uses the same K-forward carried-token boundary as the
exact scanned oracle. The B=2 len-8 smoke
`mtp_packed_prefix_broad_decode_no_bonus_firstprefix_b2_len8.json` matched the
exact no-bonus reference and measured `17.10 output tok/s`; the full-prefix
variant measured `17.35 output tok/s`. This is a small reduction versus exact
scan (`16.26 output tok/s`) but not a serving win, and it raised warmup GPU
memory to about `9.2 GB`.

K=1 verified-bonus MTP was also retested on the same tiny smoke. The gated run
`mtp_packed_prefix_broad_decode_k1_bonus_b2_len8.json` measured `40.85 output
tok/s`, but admission disabled MTP after the low-throughput probe, so that is
mostly fallback decode and not a valid speculation speedup. With admission
forced (`NANO_VLLM_JAX_MTP_MIN_ACCEPT_RATE=0` and
`NANO_VLLM_JAX_MTP_MIN_SPEEDUP=0`), packed-prefix K=1 measured `17.70 output
tok/s` with `4/10` accepted drafts and `4` verified bonus tokens. The existing
resident-table two-decode K=1 verifier measured `23.14 output tok/s` with
`1/13` accepted drafts. Both exact K=1 routes remain well below the `78.60
output tok/s` no-MTP reference on this smoke.

## Unsafe fused paths

Historical unsafe fused one-pass experiments showed useful diagnostics but are not the serving reference. Width-dependent TPU BF16 numerics and hidden-state drift made those paths unsuitable for exact serving unless guarded by explicit experimental flags and correctness checks.

Current GPU docs should describe the BF16-weight/FP32-activation baseline, the exact commit-select MTP1 parity result, and the remaining MTP1 speed gap, not older unsafe fused TPU experiments.

Fourth follow-up on 2026-06-22: the packed-prefix resident-table verifier is
again the active strict speed route, but only through the rectangular
target-verifier boundary. Earlier table-scatter variants were too copy-heavy;
the current route keeps verification, accept/reject, compact token emission,
and next-draft seeding inside the compiled boundary and tolerates stale
speculative KV beyond `committed_seq_lens` because future attention is bounded
by committed lengths. The exact hill-climb point is K=3 with
`mtp_burst_groups=4`, persistent MTP KV, prefill MTP cache seeding, recursive
draft chaining, and `NANO_VLLM_JAX_MTP_TABLE_TARGET_MODE=decode_rectangular`.
Broad packed-prefill attention modes still drift and remain diagnostics only.

Fifth follow-up on 2026-06-22: the current verified speedup comes from masked
tail bursting, not from MTP projection packing. When a K=3, burst-4 verifier
call would exceed `max_tokens` only by the final speculative bonus, the runner
now keeps the tail inside one resident-table verifier call, clamps
host-visible emitted token refs to the remaining output budget, and skips
device-token carry for rows that finish in that call. This avoids four
separate burst-1 verifier calls at the end of the B=2 len-64/128 smoke. The
temporary cached-MTP `qkv_proj_decode`/`gate_up_proj` experiment was neutral to
slightly slower and was removed from the runtime parameter tree.

Current strict speed checkpoint:

- no-MTP reference:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_strict_20260622/no_mtp_b2_len128_ref_r1.json`
  at `240.76 output tok/s`;
- verified MTP:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_strict_20260622/mtp_k3_decode_rectangular_burst4_tailclamp_clean_b2_len128_r1.json`
  at `285.10 output tok/s`;
- exact generated-token match, `100%` draft acceptance, `0` fallbacks, and `0`
  measured JIT cache growth.

The output-64 smoke is near parity (`196.83` MTP versus `197.58` no-MTP) and
is still dominated by prefill MTP cache seeding plus the cold first
speculative burst.

Broader same-shape sweep on 2026-06-23:

| output len | no-MTP output tok/s | strict MTP output tok/s | ratio | MTP correctness | MTP accept rate | measured JIT growth |
|---:|---:|---:|---:|---|---:|---:|
| 32 | 135.68 | 119.78 | 0.883x | exact | 100.0% | 0 |
| 64 | 191.06 | 198.56 | 1.039x | exact | 100.0% | 0 |
| 128 | 230.10 | 285.46 | 1.241x | exact | 100.0% | 0 |
| 256 | 264.57 | 333.71 | 1.261x | exact | 99.0% | 0 |

Artifacts live under
`/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_broad_20260623/`. The sweep
uses B=2 prompts `64,128`, generic warmup, rectangular packed-prefix target
verification, K=3, `mtp_burst_groups=4`, and reference checking against the
matching no-MTP artifact for each output length.
