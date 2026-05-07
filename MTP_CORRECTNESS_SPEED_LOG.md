# MTP correctness and speed log

## Goal

- Correctness first: MTP speculative decode must exactly match baseline greedy decode.
- Speed second: after correctness is robust, get MTP throughput above baseline or document why a speedup is mathematically impossible for the measured configuration.
- Constraint: speculation depth limited to K=1 or K=2.
- Benchmarks/tests run on the TPU VM only.

## Current known-good baseline

- Baseline engine was previously checked against HF greedy outputs for longer sequences.
- Earlier long decode check: expanded prompt set, 128 generated tokens, baseline vs HF exact token match `16/16`.
- BF16 logits MSE is small but still suspected to be higher than ideal; target is to identify implementation issues that could reduce MSE further.

## Correctness findings

### K=1 commit-select

- Correct for Qwen3.5-2B, active batch `2`, prompt length `128`, max tokens `512`.
- Correct for Qwen3.5-2B, active batch `2`, prompt length `128`, max tokens `640`.
- Failed for Qwen3.5-2B, active batch `4`, prompt length `128`, max tokens `512`.
  - First diff: request `3`, token index `499`, baseline token `36602`, MTP token `577`.
- Failed for Qwen3.5-2B, active batch `2`, queued `8` prompts, prompt length `128`, max tokens `512`.
  - First diff: request `0`, token index `497`, baseline token `2002`, MTP token `869`.

### K=2

- Correct for Qwen3.5-2B, active batch `32`, queued `128`, max tokens `64`, but no speedup.
- Failed for some low-batch mixed cases, including active batch `2` and `4` in the low-batch grid.
- Treat K=2 as unsafe until the mixed accept/reject path is fixed.

### Single-pass K=1 verifier

- A single-pass verifier over `[last_token, draft_token]` was explored.
- It was faster in some cases but failed exact greedy correctness on mixed accept/reject rows.
- Restoring rejected draft KV slots did not fix the mismatch.
- Disabled mixed use of this path for correctness; correctness-safe path remains K=1 commit-select.

## Speed findings

### Correct speedup found, but not robust yet

- Qwen3.5-2B, active batch `2`, prompt length `128`, K=1, max tokens `512`:
  - Baseline decode: `173.99 tok/s`
  - MTP decode: `186.82 tok/s`
  - Speedup: `1.058x`
  - Acceptance: `66.18%`
  - Correct: `true`
- Qwen3.5-2B, active batch `2`, prompt length `128`, K=1, max tokens `640`:
  - Baseline decode: `175.00 tok/s`
  - MTP decode: `187.30 tok/s`
  - Speedup: `1.057x`
  - Acceptance: `64.69%`
  - Correct: `true`

### Non-speedup cases

- Qwen3.5-2B, active batch `32`, queued `128`, K=1:
  - Baseline decode around `1145 tok/s`
  - MTP decode around `417 tok/s`
  - Speedup around `0.43x`
- Qwen3.5-2B, active batch `32`, queued `128`, K=2:
  - MTP decode around `491 tok/s`
  - Speedup around `0.47x`
- Longer prompt lengths `256` and `320` reduced acceptance to about `32%` and throughput to about `0.49x`.

## Active hypotheses

- Main correctness risk: mismatch between logical committed sequence length, physical KV writes, hybrid state, and `Sequence.num_tokens` after emitting multiple tokens from one MTP step.
- K=1 commit-select may update only enough internal state for one committed token while the scheduler/sequence object advances by two emitted tokens on accepted steps.
- K=2 partial/mixed paths may commit a state corresponding to a different prefix than the emitted token list.
- Remaining MSE gap may come from Gated DeltaNet prefill/decode alignment, cached suffix handling, or BF16 recurrent-state update differences.

## Workitems

- Inspect K=1 accepted-step state transitions: emitted token list, `seq.num_tokens`, `seq_lens`, KV slot writes, hybrid state table, and snapshot table.
- Fix exact correctness for K=1 in active batch `4` and queued active batch `2`.
- Fix or disable unsafe K=2 mixed paths until exact correctness is demonstrated.
- After correctness is robust, rerun K=1 speed at active batch `2`, `4`, and queued workloads.
- If speedup cannot be made robust, derive an explicit cost model comparing baseline decode cost, verifier cost, MTP head cost, acceptance rate, and emitted tokens per step.

## Patch log

### 2026-05-06: MTP block-capacity eligibility

- Issue: MTP eligibility checked block capacity only through verifier-written draft positions.
- Risk: accepted K=1 emits `[draft, bonus]`, so Python advances `Sequence.num_tokens` by two tokens before the next decode. Near block boundaries this can leave the logical sequence length ahead of allocated block-table capacity.
- Patch: require capacity for the full possible emitted token list:
  - K=1 singleton paths require `seq.num_tokens + 2`.
  - Batched K path requires `seq.num_tokens + draft_len + 1`.
- Expected effect: MTP falls back near insufficient-capacity boundaries instead of writing/advancing into a state the next scheduler step cannot represent correctly.
- TPU validation:
  - Active `B=2`, queued `8`, prompt length `128`, max tokens `512`: fixed; exact correctness `true`.
  - Active `B=4`, prompt length `128`, max tokens `512`: still failing; first diff remains request `3`, token index `499`, baseline token `36602`, MTP token `577`.
  - Speed regressed for queued active-2 because the patch forces many fallback steps near capacity boundaries; correctness remains the priority.

### 2026-05-06: Disable non-fused MTP reuse fallback

- Issue: after the capacity patch, active `B=4` still diverged late while queued active-2 became correct.
- Observation: fallback/reuse stats diverged (`drafts_accepted` far exceeded `bonus_tokens`) because the reuse path can accept a draft without emitting/processing a bonus, while fused commit-select uses different state-advance semantics.
- Patch: when not all scheduled rows can run the fused verifier, use canonical `_run_main_and_sample(...)` and only seed future MTP drafts. Do not use `_run_main_and_sample_with_mtp1_reuse(...)` for mixed/fallback batches.
- Expected effect: remove late-sequence divergence from mixed fallback semantics. Throughput may drop until a safer fallback reuse path is reintroduced.
- TPU validation:
  - Active `B=4`, prompt length `128`, max tokens `512`: still failing at the same request/token.
  - Active `B=2`, queued `8`: regressed to a late failure. This patch is not sufficient alone.

### 2026-05-06: Restore rejected draft KV slots in commit-select

- Issue: K=1 commit-select wrote draft-position KV for all rows, then selected hybrid/seq_lens for rejected rows while leaving rejected draft KV physically present.
- Previous assumption: rejected draft slots are hidden by `seq_lens` and overwritten by the next decode.
- Risk: stale rejected draft KV can leak through later mixed scheduling/block-boundary behavior.
- Patch: for rejected rows, restore draft-position KV slots from `kv_after_current`; accepted rows keep `kv_after_draft`.
- Expected effect: make physical KV match committed logical prefix exactly for both accepted and rejected rows.
- TPU validation:
  - Active `B=4`, prompt length `128`, max tokens `512`: still incorrect. Acceptance `0.4905`, baseline decode `321.39 tok/s`, MTP decode `183.99 tok/s`, speedup `0.572x`.
  - Active `B=2`, queued `8`, prompt length `128`, max tokens `512`: still incorrect. First diff request `0`, token index `492`, baseline token `3433`, MTP token `1923`. Acceptance `0.2741`, baseline decode `587.79 tok/s`, MTP decode `202.45 tok/s`, speedup `0.344x`.
- Conclusion: rejected draft KV leakage was plausible, but this patch alone does not fix the remaining correctness blocker.

### 2026-05-06: Revert canonical fallback regression

- Issue: the canonical `_run_main_and_sample(...)` mixed/fallback path regressed the queued active-2 case that the capacity patch had fixed.
- Patch: restore fallback dispatch to `_run_main_and_sample_with_mtp1_reuse(...)`.
- Expected effect: regain the previously fixed queued active-2 correctness while continuing to isolate the remaining active `B=4` late divergence.
- TPU validation:
  - Active `B=4`: still incorrect at request `3`, token index `499`, baseline token `36602`, MTP token `577`.
  - Active `B=2`, queued `8`: still incorrect at request `0`, token index `492`, baseline token `3433`, MTP token `1923`.
- Follow-up finding: the restored fallback return was indented under the profiling-only `elif profile_mtp` branch, so normal non-fused runs still fell through to plain `_run_main_and_sample(...)`. Fixed in the next patch.

### 2026-05-06: Gate unsafe fast all-accept verifier and fix fallback indentation

- Issue: with commit-select disabled, active `B=4` diverged early at token index `52` when the fast all-accept verifier was allowed. Disabling fused verify made the same case correct, indicating the fast two-token decode verifier is unsafe for batched hybrid-state decode.
- Issue: the non-fused fallback return was accidentally nested under the profile-only branch, preventing the intended state-preserving reuse path from running in normal benchmarks.
- Patch:
  - Require explicit `NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT=1` before using `mtp1_two_decode_greedy_fast_step_jit`.
  - Dedent the non-fused fallback return so `_run_main_and_sample_with_mtp1_reuse(...)` runs whenever the full batch cannot use fused verification.
- TPU validation: pending.

### 2026-05-06: Use scheduled batch metadata for compact bonus decode

- Issue: safe reuse with compact one-token bonus decode still diverged early when bonus emission was enabled: active `B=4`, request `0`, token index `52`, baseline token `318`, MTP token `27718`.
- Observation: no-bonus reuse remains correct, so the issue is specifically in the bonus decode/commit path.
- Patch: build accepted-row bonus decode batches from `batch.positions`, `batch.seq_lens`, `batch.seq_ids`, and `batch.block_tables` instead of host `Sequence.num_tokens` and `.tolist()` block-table reconstruction.
- Expected effect: remove any stale host-length or block-table mismatch between the scheduled target decode and the compact accepted-row bonus decode.
- TPU validation:
  - Active `B=4`, safe reuse + compact bonus: still incorrect at request `0`, token index `52`, baseline token `318`, MTP token `27718`. Acceptance `0.6212`, baseline decode `325.13 tok/s`, MTP decode `130.65 tok/s`, speedup `0.402x`.
- Conclusion: stale host length / `.tolist()` block-table reconstruction was not the root cause.

### 2026-05-06: Fix accepted-bonus MTP reseed position

- Issue: after safe reuse accepted a draft and emitted `[draft, bonus]`, it reseeded the next MTP draft with `position=seq.num_tokens`, even though the confirmed bonus token is at `seq.num_tokens + 1`.
- Patch: allow `_seed_mtp1_drafts(...)` to take explicit positions and pass `batch.positions[row, 0] + 2` for accepted-bonus first-only rows.
- Expected effect: prevent the next draft from being generated with a stale position id after bonus emission. This should reduce suspicious over-acceptance and eliminate one source of post-bonus drift.
- TPU validation:
  - Short active `B=4`, max tokens `80`, safe reuse + compact bonus: still incorrect at request `0`, token index `52`, baseline token `318`, MTP token `27718`. Acceptance `0.6410`, baseline decode `314.04 tok/s`, MTP decode `126.48 tok/s`, speedup `0.412x`.
- Conclusion: the reseed position bug is real but not the first-divergence root cause.

### 2026-05-06: Decode verifier-confirmed target token in commit-select second pass

- Issue: commit-select's second target-model decode used the MTP draft token for every row. For accepted rows this equals the verifier target, but for rejected rows it is an uncommitted token. Although rejected rows are later restored, using unaccepted tokens in the batched second pass is a correctness risk and diverges from ordinary target decode.
- Patch: compute `target_token` immediately after the first decode, then feed `target_token` into the second decode for every row. Accepted rows still commit the second-step state and emit `[draft, bonus]`; rejected rows restore second-step KV/state and emit only `target_token`.
- Expected effect: keep commit-select's second pass aligned with the canonical target-model trajectory while preserving on-device commit selection.
- TPU validation:
  - Short active `B=4`, max tokens `80`: correct. Acceptance `0.6615`, baseline decode `312.49 tok/s`, MTP decode `167.23 tok/s`, speedup `0.531x`.
  - Full active `B=4`, max tokens `512`: still incorrect at request `3`, token index `499`, baseline token `36602`, MTP token `577`. Acceptance `0.6014`, baseline decode `320.96 tok/s`, MTP decode `162.35 tok/s`, speedup `0.506x`.
  - Active `B=2`, queued `8`, max tokens `512`: correct. Acceptance `0.4139`, baseline decode `592.86 tok/s`, MTP decode `241.68 tok/s`, speedup `0.410x`.
- Conclusion: this fixes the early bonus-state failure and the queued mixed/fallback correctness gate, but the active `B=4` late divergence remains.

### 2026-05-06: Add scalar MTP verifier isolation switch

- Issue: active `B=4` remains correct through 480 generated tokens and diverges at token index `499`, suggesting accumulated batched two-step state drift.
- Patch: add `NANO_VLLM_JAX_MTP_FORCE_SCALAR=1` to bypass batched fused verification inside `_run_mtp1_batched(...)` and use the existing scalar verifier per row.
- Expected effect: determine whether correctness failure is in the batched verifier/commit path or in scalar MTP semantics.
- TPU validation:
  - Active `B=4`, max tokens `512`, scalar verifier: incorrect at request `0`, token index `52`, baseline token `318`, MTP token `27718`.
  - Single prompt `0`, max tokens `512`, scalar verifier: incorrect at token index `143`, baseline token `1445`, MTP token `464`.
  - Single prompt `0`, max tokens `256`, safe one-token reuse + compact bonus: correct. Acceptance `0.4971`, baseline decode `108.36 tok/s`, MTP decode `53.24 tok/s`, speedup `0.492x`.
- Conclusion: the scalar cached-prefill verifier is not exact; the one-token reuse bonus path is exact for a single sequence but not for active `B=4`, so the remaining issue is batch-shape/state interaction in bonus emission rather than prompt content alone.

### 2026-05-06: Add full-batch main lookahead diagnostic

- Issue: conditional/compact bonus emission is exact for one sequence but diverges for active `B=4`.
- Patch: add `NANO_VLLM_JAX_MAIN_LOOKAHEAD_ALL=1`, which runs a second ordinary full-batch main-model decode on the target tokens and emits `[target, bonus]` for every active row. This preserves batch shape and avoids conditional accepted-row compaction.
- Purpose: distinguish "two-step decode is inherently drifting" from "conditional/compact speculative bonus commit is drifting." This path is a diagnostic/lookahead baseline, not an MTP acceptance speed path.
- TPU validation:
  - Active `B=4`, max tokens `512`, full-batch main lookahead with hidden-derived bonus logits: incorrect at request `0`, token index `52`, baseline token `318`, lookahead token `27718`.
- Follow-up finding: the diagnostic and compact bonus path computed bonus logits from `return_hidden=True` activations via `_logits_from_hidden(...)`, while baseline greedy decode uses the canonical `return_hidden=False,last_logits_only=True` logits path. This can change greedy top-1 enough to diverge.

### 2026-05-06: Use canonical logits path for bonus emission

- Issue: bonus tokens were sampled from hidden-derived logits rather than the same logits path used by baseline decode.
- Patch: for compact accepted bonus decode and full-batch lookahead diagnostic, run the second decode with `return_hidden=False,last_logits_only=True` and sample bonus tokens directly from executor logits. MTP reseeding after accepted bonus is temporarily skipped until a canonical hidden+logits path is available.
- Expected effect: exact bonus tokens should match baseline target decode for the same prefix.
- TPU validation: pending.

### 2026-05-06: Commit-select canonical bonus logits, no post-bonus reseed by default

- Issue: commit-select still derived `bonus_token` from `return_hidden=True` activations inside the fused path, and then seeded the next MTP draft from the accepted bonus path. Hidden materialization can differ from canonical baseline logits enough to flip greedy top-1.
- Patch:
  - In `mtp1_commit_select_greedy_step_jit`, run the second decode with `return_hidden=False,last_logits_only=True` and sample `bonus_token` from canonical logits.
  - Generate `next_draft_token` only from the first decode state, which is valid for rejected rows.
  - In the runner, do not seed MTP after accepted bonus emission unless `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1` is explicitly set.
- Expected effect: correctness first; accepted steps still emit one bonus token, but accepted rows do not immediately propose another MTP draft until a later ordinary decode supplies canonical hidden/logits.
- TPU validation:
  - Active `B=4`, max tokens `512`: correct. Acceptance `0.5932`, baseline decode `318.18 tok/s`, MTP decode `155.68 tok/s`, speedup `0.489x`.
  - Active `B=2`, queued `8`, max tokens `512`: correct. Acceptance `0.4156`, baseline decode `587.60 tok/s`, MTP decode `241.32 tok/s`, speedup `0.413x`.
- Conclusion: correctness is fixed for the main K=1 gates, but speed is poor because accepted rows do not immediately reseed MTP.

### 2026-05-06: Return canonical logits and hidden from one decode

- Issue: to recover speed, accepted rows need to reseed MTP after bonus emission, but bonus sampling must still use canonical logits.
- Patch:
  - Add `return_hidden_with_logits` to `model_forward_step(...)` / `forward(...)`.
  - In commit-select, request `(hidden, logits)` from the second decode, sample `bonus_token` from canonical logits, and use the returned hidden only for the next MTP draft.
- Expected effect: allow `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1` to restore continuous speculative proposals without reintroducing hidden-derived bonus-token drift.
- TPU validation:
  - With `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1`, active `B=4`, max tokens `512`: incorrect at request `3`, token index `499`, baseline token `36602`, MTP token `577`.
  - With `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1`, active `B=2`, queued `8`: correct.
  - With default no post-bonus reseed, active `B=4`, max tokens `512`: correct. Acceptance `0.5932`, baseline decode `324.76 tok/s`, MTP decode `154.43 tok/s`, speedup `0.476x`.
- Conclusion: hidden returned for MTP reseed after accepted bonus remains unsafe for continuous speculation. The default must keep `NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=0` for correctness.

### Current speed interpretation

- K=1 correctness is now reliable in the safe mode, but it cannot beat baseline in the compute-dominated regime: accepted steps require the current target decode plus a second target decode to produce the bonus, yielding at most two emitted tokens for roughly two target decodes.
- Measured safe K=1 active `B=4`: `154.43 tok/s` MTP vs `324.76 tok/s` baseline (`0.476x`).
- Next speed path: test K=2, where multiple drafted tokens can be verified in a single wider target pass and may amortize target-decode cost better than K=1.

### 2026-05-06: K=2 canonical logits patch

- Observation before patch:
  - K=2 active `B=2`, max tokens `256`: incorrect at request `0`, token index `52`, baseline token `318`, MTP token `27718`; speedup `0.386x`.
  - K=2 active `B=4`, max tokens `256`: incorrect at request `0`, token index `52`, baseline token `318`, MTP token `27718`; speedup `0.340x`.
- Patch: in `mtp_k_decode_greedy_step_jit`, request `(hidden, logits)` from `model_forward_step(..., return_hidden_with_logits=True)` and sample target/bonus tokens from canonical logits instead of recomputing logits from materialized hidden.
- TPU validation: pending.

### 2026-05-06: Compact fallback accepted-draft decode rows

- Issue: the non-fused reuse fallback decoded accepted draft tokens using a full active-batch tensor with real inactive sequence IDs, dummy token `0`, position `0`, and zero query length for inactive rows.
- Risk: if any backend write/hybrid path does not fully mask zero-query real rows, rejected or non-MTP rows can receive dummy KV/hybrid updates. This matches the queued/mixed late-divergence risk after fallback semantics changed.
- Patch:
  - Decode accepted draft rows as a compact accepted-only `ScheduledBatch`.
  - Store/refresh only accepted rows after the draft decode.
  - Return hidden for accepted draft decodes and seed the next MTP draft from the accepted logical prefix.
  - Make rejected-draft KV-slot restore preserve layer dimensions for both 5D block cache and 4D flat cache layouts.
- Expected effect: fallback decode mutates exactly the rows whose logical prefix advanced, while commit-select rejected-slot restoration is layout-safe.
- TPU validation pending.
## K=2 verifier canonical-logits probe: K=1 repair contract failure

- TPU run: Qwen/Qwen3.5-2B, K=2, B=2/B=4, 256 decode tokens, JIT, KV cache.
- Result: benchmark failed before producing correctness numbers.
- Failure: K=2 partial-reject repair entered the K=1 repair path, where `model_forward_step(return_hidden=True)` was destructured as `(hidden, verify_logits)` even though `return_hidden_with_logits=True` was not set.
- Fix applied: restored the K=1 repair callsite to receive only `hidden`; K=2 verifier still uses `return_hidden_with_logits=True` and canonical `verify_logits`.

## K=2 canonical-logits rerun still diverged at token 52

- TPU run: Qwen/Qwen3.5-2B, K=2, B=2/B=4, 256 decode tokens, JIT, KV cache.
- B=2: correctness failed at request 0 token 52, baseline token `318`, MTP token `27718`; acceptance `0.4016`; decode speedup `0.386x`.
- B=4: correctness failed at request 0 token 52, baseline token `318`, MTP token `27718`; acceptance `0.3831`; decode speedup `0.339x`.
- Diagnosis: this is the same divergence signature previously seen when target/bonus tokens were sampled from hidden-derived logits instead of canonical forward logits.
- Fix applied: all remaining MTP verifier callsites in `model_executor.py` now use `model_forward_step(return_hidden=True, return_hidden_with_logits=True)` and sample target/bonus tokens from returned canonical `verify_logits`, not from `lm_head(rms_norm(hidden))`.

## K=2 correctness fix: sequential commit-select verifier

- Additional finding: the K=2 common postprocess used `bool(row_acceptance)` for a row of booleans, which treats every non-empty K=2 row as fully accepted.
- Additional finding: the all-at-once K=2 verifier still diverged after canonical logits, which points to multi-token verifier state semantics for the hybrid model rather than logits only.
- Fix applied: added a K=2 sequential commit-select executor path. It runs three one-token target decodes, computes accepted prefix length `0/1/2`, selects KV/hybrid state on device, restores uncommitted KV slots, and returns canonical target tokens.
- Fix applied: K=2 runner postprocess now emits accepted prefix plus the first target-model mismatch, not always full draft chain plus bonus.
- Safety choice: default seeding after accepted bonus remains off; K=2 only reseeds by default after prefix length `0`, matching the known-correct K=1 no-post-bonus-reseed policy.

## K=1/K=2 correctness gate after sequential K=2 verifier

- TPU run: Qwen/Qwen3.5-2B, 256 decode tokens, expanded prompts, real weights, JIT, KV cache.
- K=1, B=4: `correct=True`, first diff `None`, acceptance `0.6198`, baseline decode `321.40 tok/s`, MTP decode `157.01 tok/s`, speedup `0.489x`.
- K=2, B=2: `correct=True`, first diff `None`, acceptance `0.4167`, per-position accepts `[81, 41]`, baseline decode `174.59 tok/s`, MTP decode `99.45 tok/s`, speedup `0.568x`.
- K=2, B=4: `correct=True`, first diff `None`, acceptance `0.3656`, per-position accepts `[54, 33]`, baseline decode `315.71 tok/s`, MTP decode `146.35 tok/s`, speedup `0.465x`.
- Status: correctness is restored for K=1 and K=2 on this gate. Speed is still below baseline; next work is optimizing or proving why this correctness-safe path cannot beat baseline for this model/backend.

## B=50 throughput gate before seeding replay removal

- TPU run: Qwen/Qwen3.5-2B, 50 concurrent sequences, mixed prompt lengths `[32,64,128,256]`, 128 decode tokens, real weights, JIT, KV cache.
- K=1: `correct=True`, acceptance `0.5217`, baseline decode `1057.26 tok/s`, MTP decode `417.33 tok/s`, speedup `0.414x`, accepted-latency p50 `2.40 ms/token`.
- K=2: `correct=True`, acceptance `0.2768`, per-position accepts `[7, 0]`, baseline decode `1061.71 tok/s`, MTP decode `428.29 tok/s`, speedup `0.421x`, accepted-latency p50 `2.32 ms/token`.
- Interpretation: batching improves absolute throughput, but sequential verifier paths still lose because they perform extra target-model decodes per emitted token.

## Speed fix: remove normal decode replay used for MTP seeding

- Issue: when normal target decoding seeded a future MTP draft, the runner first ran logits-only decode, then replayed the same target forward to recover hidden state.
- Fix applied: `ModelExecutor.forward_step` and `forward_step_jit` now expose `return_hidden_with_logits`, and the runner requests `(hidden, logits)` from the same canonical forward when `seed_mtp1=True`.
- Correctness rationale: token sampling still uses canonical logits returned by the target forward; MTP seeding uses the matching hidden from the same forward, so no baseline output should change.

## Correctness/speed gate after seeding replay removal

- TPU run: Qwen/Qwen3.5-2B, B=4, 256 decode tokens, expanded prompts, real weights, JIT, KV cache.
- K=1: `correct=True`, first diff `None`, acceptance `0.6198`, baseline decode `321.51 tok/s`, MTP decode `210.70 tok/s`, speedup `0.663x`, accepted-latency p50 `4.79 ms/token`.
- K=2: `correct=True`, first diff `None`, acceptance `0.3656`, per-position accepts `[54, 33]`, baseline decode `326.05 tok/s`, MTP decode `202.95 tok/s`, speedup `0.630x`, accepted-latency p50 `4.99 ms/token`.
- Effect: K=1 improved from `0.489x` to `0.663x`; K=2 improved from `0.465x` to `0.630x`. Still below baseline, but the biggest replay overhead is removed.

## B=50 throughput after seeding replay removal

- TPU run: Qwen/Qwen3.5-2B, 50 concurrent sequences, mixed prompt lengths `[32,64,128,256]`, 128 decode tokens, real weights, JIT, KV cache.
- K=1: `correct=True`, acceptance `0.5217`, baseline decode `1066.62 tok/s`, MTP decode `587.42 tok/s`, speedup `0.606x`, accepted-latency p50 `1.70 ms/token`.
- K=2: `correct=True`, acceptance `0.2768`, per-position accepts `[7, 0]`, baseline decode `1064.69 tok/s`, MTP decode `589.47 tok/s`, speedup `0.607x`, accepted-latency p50 `1.67 ms/token`.
- Effect: K=1 improved from `0.414x` to `0.606x`; K=2 improved from `0.421x` to `0.607x`. Remaining gap is dominated by sequential target-verifier cost and low K=2 second-position acceptance.

## Speed attempt: switch K=1 two-token verifier from cached prefill to decode metadata

- Finding: `mtp1_two_decode_greedy_step_jit` already selects prefix hybrid state for accepted vs rejected rows, but it ran the verifier batch with `is_prefill=True`.
- Risk: previous K=1 non-commit-select failures at token 52 may have come from using cached prefill semantics for what should be a decode verifier.
- Fix applied: K=1 two-token verifier now uses `is_prefill=False`, `num_prefill_tokens=0`, and `num_decode_tokens=2 * batch_size`, while still requesting prefix hybrid state.
- Expected speed effect if correct: one width-2 target verifier can replace the two sequential target decodes in commit-select, making K=1 speedup possible if width-2 cost is less than `1 + acceptance` baseline decode equivalents.

## K=1 two-token verifier remains unsafe

- TPU run: Qwen/Qwen3.5-2B, K=1 two-token verifier, `NANO_VLLM_JAX_MTP_COMMIT_SELECT=0`.
- B=4: `correct=False`, first diff request 0 token 52, baseline token `318`, MTP token `27718`; speedup `0.451x`.
- B=50: `correct=True`, but only because all-accepted batches were effectively absent; speedup `0.458x` due wasted verifier attempts before fallback.
- Conclusion: the width-2 verifier still cannot be committed safely for this hybrid model, even in decode mode. It is now gated behind `NANO_VLLM_JAX_MTP_ENABLE_PREFIX_TWO_DECODE=1`; the default K=1 fused path falls back unless commit-select is enabled.

## Main-decode-reuse bonus path remains unsafe

- TPU run: Qwen/Qwen3.5-2B, K=1, `NANO_VLLM_JAX_MTP_COMMIT_SELECT=0`, `NANO_VLLM_JAX_MTP_EMIT_BONUS=1`.
- B=4: `correct=False`, first diff request 1 token 142, baseline token `35801`, MTP token `9670`; speedup `0.534x`.
- B=50: `correct=False`, first diff request 3 token 80, baseline token `424`, MTP token `2503`; speedup `0.575x`.
- Conclusion: emitting a bonus from the main-decode-reuse path is not yet scheduler/state correct. Keep `NANO_VLLM_JAX_MTP_EMIT_BONUS=0` unless the cache-length/state accounting for emitted-but-not-cached bonus tokens is fixed.

## Current speed conclusion for correctness-safe Qwen3.5 hybrid K<=2

- Correct paths:
  - K=1 sequential commit-select: correct, best measured speedup `0.663x` at B=4 and `0.606x` at B=50 after replay removal.
  - K=2 sequential commit-select: correct, best measured speedup `0.630x` at B=4 and `0.607x` at B=50 after replay removal.
- Incorrect/unsafe paths:
  - K=1 one-pass prefix/two-token verifier: incorrect on B=4.
  - K=1 main-decode-reuse with emitted bonus: incorrect on B=4 and B=50.
  - K=2 all-at-once verifier: incorrect at token 52 before sequential commit-select.
- Math: for a sequential verifier with K drafts, each speculative step costs `K+1` target decodes and emits at most `1 + accepted_drafts` tokens. For K=1 the ideal zero-overhead ceiling is `(1+a)/2 <= 1`; with measured `a=0.62`, ceiling is `0.81x`. For K=2 the ideal ceiling is `(1+a0+a1)/3 <= 1`; with measured B=4 accepted positions `[54,33]` over 80 proposals, ceiling is about `(1+0.675+0.4125)/3 = 0.696x`. Any real overhead pushes below that. Therefore a true speedup is impossible for the currently correct sequential K<=2 verifier; speedup requires a correct one-pass verifier with selectable hybrid prefix states or a dense-only model path without hybrid state.

## K=1 one-pass verifier implementation with per-token hybrid prefix states

- Change: `forward_step(..., return_prefix_hybrid=True)` now returns token-indexed hybrid prefix states with shape `[batch, query_token, linear_layer, ...]` instead of only a single token-0 snapshot.
- Change: `mtp1_two_decode_greedy_step_jit` now runs a single two-token decode verifier over `[current_token, draft_token]` with decode metadata, then selects token-0 hybrid state for rejected rows and token-1 hybrid state for accepted rows.
- Change: rejected rows keep the current-token KV write while restoring the rejected draft KV slot; accepted rows keep both verifier KV writes and commit `seq_lens + 1`.
- Change: the batched K=1 fused runner now prefers the one-pass prefix verifier by default. The older sequential commit-select verifier is still available by setting `NANO_VLLM_JAX_MTP_COMMIT_SELECT=1`; the new path can be disabled with `NANO_VLLM_JAX_MTP_DISABLE_ONE_PASS_K1=1`.
- Status: not benchmarked in this edit pass. The next TPU correctness gate should compare long greedy outputs against baseline before measuring speedup.

## K=1 bonus-margin gate

- Finding: Qwen/Qwen3.5-2B diverged on expanded 4-prompt validation at request `2`, token `32`, with baseline token `6093` and MTP token `4087`. The same divergence appeared with the older sequential commit-select path, so the issue is not isolated to the new token-indexed hybrid prefix extraction.
- Hypothesis: emitting verifier bonus tokens on low-margin logits lets tiny numerical/state differences compound into a different greedy branch. Treating those accepts as one-token emissions preserves the ordinary next decode for the risky position.
- Change: K=1 one-pass and sequential commit-select now support `NANO_VLLM_JAX_MTP_BONUS_MARGIN=<float>`. If set above `0`, an otherwise accepted draft only commits/emits the bonus when the verifier bonus logit margin `top1 - top2` meets the threshold; otherwise it emits only the target/draft token and commits state after the current token.
- Status: threshold sweep pending on TPU.

## K=1 one-pass verifier correctness diagnosis

- Added `--trace-steps` to `benchmark_mtp1_engine.py` so TPU-side correctness runs can record per-step emitted token deltas and branch labels.
- Targeted 2B prompt `"Complete this JSON object: {\"name\": \"compiler\", \"features\": ["` diverged at generated token 7:
  - baseline: `[16, 11, 220, 17, 11, 220, 18, 1089, ...]`
  - MTP K=1 one-pass: `[16, 11, 220, 17, 11, 220, 18, 13587, ...]`
  - the divergent token was emitted by an `mtp_rejected` step after the visible prefix still matched, so earlier accepted two-token verifier steps had drifted internal model state.
- Patch: the K=1 one-pass verifier now runs the two-token target suffix as cached prefill (`is_prefill=True`, `num_prefill_tokens=2 * batch`) instead of decode metadata. This is intended to make full-attention KV for the draft token causal/exact while still returning per-token hybrid prefix states for accept/reject commit selection.
