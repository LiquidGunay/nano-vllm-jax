# Mainline Cleanup and Speculative Decoding Plan

Last updated: 2026-07-13

## Goal

Build the next mainline in three stages:

1. Make the non-speculative engine correct, explicit, and easy to teach.
2. Make its single performance claim reproducible from a clean checkout.
3. Add speculative decoding behind a generic drafter/verifier boundary, with
   Qwen3.5 MTP as the first drafter.

Every merged PR must leave `main` usable on its own. We will open and review
one PR at a time, then build the next PR from the newly merged `main`.

## Source Revisions

- Clean mainline: `origin/main@850cd3e`
- Experimental evidence: `experimental/mtp-prefill-verifier-speed@da91504`
- Cleanup review: `/mountpoint/.exp/cleanup_and_diagnosis.md`

The clean mainline is the implementation base. The experimental branch is a
source of small, proven ideas and tests; its runner, executor, configuration
surface, and diagnostics will not be merged or cherry-picked wholesale.

## Working Decisions

- The base engine loads model architecture from the checkpoint, then validates
  it. It does not infer architecture from a model-name substring.
- The initially validated dense checkpoints are Qwen3.5 0.8B, 2B, and 4B.
  Larger models remain rejected until they have a real-weight test on suitable
  hardware.
- The proposed headline workload is Qwen3.5-4B, BF16, batch 1, 64 prompt
  tokens, 64 greedy output tokens, on one A10G. This is provisional until it is
  freshly reproduced on the cleaned code and a pinned current vLLM release.
- B=1 is the speculative latency target. B=8 remains a non-regression lane for
  the ordinary engine; no B=1 optimization may silently replace the B=8 path.
- JAX shape specialization is explicit in compile buckets and route keys. It
  is not hidden benchmark-specific compilation.
- A tied vocabulary projection stays in its checkpoint-native `[V, H]`
  layout. The base engine must not allocate a second BF16 `[H, V]` copy.
- Fixed token-ID vocabulary prefixes are diagnostics only. Shipped drafting is
  full-vocabulary.
- Speculation is optional. When disabled, MTP weights, MTP caches, speculative
  routes, and their compilation surface do not exist.
- The only production verifier is strict target-model packed-prefix
  verification. Sequential repair, unverified append, and forced acceptance
  remain test or diagnostic tools and cannot support a speed claim.

## Repository Style Contract

Issue [#10](https://github.com/LiquidGunay/nano-vllm-jax/issues/10) identified
the main remaining risk: optimizations accumulating as flags, optional fields,
and history-encoded route names until documentation has to explain accidental
complexity. `docs/style.md` is the review contract once its PR merges.

The governing rule is conceptual compression, not a line-count target:

- Keep one ordinary serving path and one typed representation of each choice.
- Put state and transitions at their documented owner; do not add compatibility
  guards for partially initialized internal objects.
- Prefer concept-level names and narrow specs over boolean matrices and a flat
  configuration message bus.
- Keep serving policy outside model math and keep host/device mirrors behind one
  explicit materialization boundary.
- Extract a file only when it owns a coherent transition; facade-only modules
  make the reading path worse.
- Every optimized operation keeps visible reference semantics and a focused
  parity check.

The first reading path stays centered on `engine.py`, `scheduler.py`,
`block_manager.py`, batch materialization, `runner.py`, and `model.py`. Service,
warmup, implementation policy, and specialized kernels are the advanced path.
Each PR review records which concept it adds, which older concept it removes or
consolidates, and why the ordinary path remains easy to trace.

## Target Base ABI

The scheduler should describe work; it should not create or cache JAX arrays.

```text
Scheduler.plan()
    -> SchedulePlan          host-only logical work
Runner.materialize()
    -> DeviceBatch           JAX arrays and resident-slot ids
Runner.execute()
    -> RunResult             device outputs and state transition
Engine.commit()
    -> StepResult            token events, finishes, and counters
Service
    -> explicit output materialization and streaming
```

| Object | Owner | Contains |
| --- | --- | --- |
| `ModelSpec` | checkpoint loader | Validated Qwen3.5 text architecture |
| `CapacitySpec` | engine | KV/request capacity and block geometry |
| `CompileSpec` | runner | Sorted, unique shape buckets |
| `KernelPlan` | internal policy | One promoted implementation per route |
| `RuntimeSpec` | engine | Frozen aggregate of the four specs above |
| `SchedulePlan` | scheduler | Host rows, phase, lengths, block ids, bucket |
| `DeviceBatch` | runner | Device arrays only; no host mirrors |
| `ExecutionPlan` | route registry | Route kind and static execution signature |
| `RunResult` | runner | Device token buffer, counts, and committed state |
| `StepResult` | engine | Explicit logical token and finish events |
| `OutputBuffer` | output layer | Deferred device references and host tokens |

Only the minimal `ExecutionSignature` needed by a compiled function is a JIT
cache key. The complete engine configuration is not hashed and passed through
model code.

### State ownership

- `Sequence` owns request identity, prompt metadata, logical lengths, sampling
  parameters, and status. It contains no JAX values and performs no implicit
  materialization.
- `BlockManager` owns physical KV block ids and reservations.
- `PrefixCache` owns complete cache entries and cache statistics.
- `ModelRunner` owns all accelerator arrays: target KV, GDN state, resident
  metadata, token carry, and later drafter state.
- `OutputBuffer` owns generated token values and deferred token references.
- `Engine.commit()` is the single place that advances logical length, block
  state, target state, output position, and finish reason.

### Capacity pressure

The first cleaned implementation uses conservative reservation, not recompute
preemption:

1. Compute worst-case blocks from `prompt_tokens + max_tokens`.
2. Reject a request that can never fit the configured engine.
3. Admit a waiting request only when its complete capacity-credit reservation
   can be held; allocate physical pages only as tokens need them.
4. Use bounded first-fit admission so a blocked large waiter does not stall a
   smaller request that fits.
5. Never evict a request after generation begins.
6. Release physical pages and unused credits on finish, cancellation, or error.

Recompute preemption can return later only with an explicit replay/reset ABI
and its own correctness PR.

### Prefix cache lifecycle

```python
@dataclass(frozen=True)
class PrefixCacheEntry:
    prefix_hash: int
    token_count: int
    block_ids: tuple[int, ...]
    hybrid_state_handle: int | None
```

KV metadata may be published before a GDN snapshot is attached, but a hybrid
prefix is reusable only when both represent the same complete boundary.
Reusing any physical block in the entry atomically removes the entry, releases
its runner-owned state handle, and updates statistics. The cache has an
explicit entry/state budget and a churn test.

### Model and weight ABI

- Resolve one checkpoint snapshot and use its config, tokenizer, and weights.
- Extract the text config when the Hugging Face config wraps it.
- Validate architecture type, layer types, dimensions, head relationships,
  RoPE fields, tying, and every required tensor shape before serving.
- Keep vocabulary weights as `[V, H]` for both tied embeddings and untied
  heads. JAX reference math may use a transpose view; optimized kernels accept
  the native layout directly.
- Create derived packed projection weights only when the promoted route needs
  them, and include them in the startup memory estimate.
- Keep only the streaming safetensors loader.

The native vocabulary layout ports the useful base-JAX ABI change from the
experimental branch and removes the extra tied-head allocation that made 4B
unnecessarily tight.

## Target Speculative ABI

Drafter and verifier are separate components, but they compose inside one JAX
transition. There is no Python callback, token materialization, or host decision
between proposal and verification.

```python
class Drafter(Protocol):
    def propose(self, context, state) -> DraftProposal: ...
    def commit(self, proposal, verdict) -> DraftState: ...

class Verifier(Protocol):
    def verify(self, context, proposal) -> VerificationResult: ...
```

The device pytrees have these semantic fields:

- `DraftProposal`: full-vocabulary token ids `[B, K]`, valid mask, and opaque
  drafter checkpoints.
- `VerificationResult`: accepted count per row, emitted target-approved tokens
  `[B, K+1]`, emitted count, bonus/correction token, and selected target state.
- `DraftState`: owned by the drafter. The orchestrator passes it back only to
  that drafter's `commit()` method.

The target verifier knows nothing about MTP weights or caches. The MTP drafter
knows nothing about target KV/GDN commit mechanics. A small speculative
orchestrator performs:

```text
proposal = drafter.propose(...)
verdict = target_verifier.verify(..., proposal)
draft_state = drafter.commit(proposal, verdict)
return target-approved tokens and both committed states
```

The production verifier packs `[current, draft_1, ..., draft_K]` into one target
forward, compares tokens on device, selects the accepted-prefix KV/GDN state on
device, and returns compact token references. Target KV written beyond the
committed length may remain stale because committed lengths bound attention and
future writes replace those slots.

MTP is then one implementation of `Drafter`:

- recursive K-token proposals;
- persistent MTP KV state;
- MTP state seeded during target prefill;
- BF16 full-vocabulary proposal as the reference;
- full-vocabulary INT8 proposal projection as a performance option, promoted
  only after acceptance, margin-conditioned flips, and KL are reported;
- no reduced token-ID vocabulary and no unverified route.

## Pull Request Sequence

### Preparation: fresh external baseline

Status: [x] complete with a guarded baseline limitation

Unmodified `main` could not complete either the 0.8B B=8 or 4B B=1 lane
inside the 70% system-RAM guard because it retained the materialized tied head.
That failure is the baseline result; the cleanup branch completed both lanes
without measured-phase JIT growth. Raw artifacts remain outside the repository.

Before changing `main`, capture a fresh external baseline for the exact
manifests used by the cleanup gates. This is not a public claim and no result
dump is committed.

- 0.8B B=8 regression lane.
- 4B B=1 candidate headline lane.
- Exact output rows and measured-phase compilation count.
- Peak system RAM, peak device memory, TTFT, and decode throughput.
- Three repeats where runtime permits.

All processes use CUDA-only JAX, GPU preflight, the 70% system-RAM sidecar,
bounded CPU cores, positive nice level, and mountpoint-owned caches/artifacts.

### PR 1: correct capacity, configuration, and base model ABI

Status: [x] merged as PR [#7](https://github.com/LiquidGunay/nano-vllm-jax/pull/7)
at main commit `0b6aa72`

Branch: `agent/base-correctness-abi` at `ef1ceab`

Purpose: make the ordinary target model truthful and safe before moving state
between components.

Scope:

- Remove recompute preemption and add worst-case block reservation.
- Make sequence ids and block size engine-local.
- Introduce frozen checkpoint and workload configs. Keep the private flat
  `RuntimeConfig` as a compatibility bridge until PR 2 extracts the complete
  capacity/compile/kernel aggregate and step ABI.
- Reject unknown YAML keys and invalid or uncovered buckets.
- Resolve and validate architecture from the checkpoint.
- Support only real-weight-validated Qwen3.5 dense sizes.
- Standardize the native `[V, H]` vocabulary ABI and remove the materialized
  tied-head copy.
- Keep streaming weight loading and remove the duplicate full-memory loader.
- Add startup parameter, derived-weight, KV, and headroom accounting.

Merge gates:

- Tiny-capacity tests cannot preempt or corrupt a partially generated request.
- Two engines with different block sizes and id spaces do not interfere.
- Config/model shape failures are early and precise.
- Fresh real-weight short parity for 0.8B, 2B, and 4B, run sequentially under
  the RAM guard.
- The 4B base route fits with documented memory headroom.
- No measured-phase JIT growth on the primary B=1 and B=8 checks.

### PR 2: explicit step, cache, route, and output ownership

Status: [ ] split in progress; PRs 2a-2c are merged, PR 2d is open for review
as PR [#12](https://github.com/LiquidGunay/nano-vllm-jax/pull/12), and PR 2e
remains planned from issue #10

PR 2a branch: `agent/step-ownership-abi` at `5cee688`; merged as PR
[#8](https://github.com/LiquidGunay/nano-vllm-jax/pull/8) at main commit
`534f039`

PR 2a establishes the review boundary recommended by the cleanup audit:

- `Scheduler.schedule()` returns immutable host-only `ScheduledRow` and
  `SchedulePlan` values with an explicit `BucketShape`.
- Runner-owned `BatchMaterializer` performs padding, packed layout,
  `device_put`, and reusable decode-array lookup, then returns `DeviceBatch`.
- `LLMEngine.step()` now exposes `schedule -> materialize -> execute`.
- Importing scheduler and sequence in a clean process does not import JAX.
- Exact guarded controls against the approved PR 1 head matched all output
  tokens with no measured JIT growth: 0.8B B=8 measured `434.41` versus
  `434.21` decode tok/s; 4B B=1 measured `49.92` versus `49.89`.

PR 2b branch: `agent/step-commit-output`; merged as PR
[#9](https://github.com/LiquidGunay/nano-vllm-jax/pull/9) at main commit
`850cd3e`

PR 2b completes the execute/commit/output half of the ABI:

- `ModelRunner.execute()` returns `RunResult`.
- `LLMEngine.commit()` is the one logical transition and `step()` returns
  `StepResult` rather than a signed integer tuple.
- `OutputBuffer` owns every generated host token and device reference.
- Service streaming consumes explicit output watermarks, materializes output,
  and emits bounded coalesced token chunks.
- Cancellation is committed by the engine worker and releases scheduler,
  runner, and KV state for waiting or running requests.
- Queued plus active requests are bounded; disconnects cancel, health follows
  the worker, and shutdown verifies that the worker actually stopped.
- EOS/length/cancelled finish reasons and the configured model id reach server
  output.

PR 2c branch: `agent/prefix-cache-lifecycle`; merged as PR
[#11](https://github.com/LiquidGunay/nano-vllm-jax/pull/11) at main commit
`0f6fbbc`

PR 2c keeps the prefix-cache follow-up narrow:

- Replace split KV/hash and GDN-state dictionaries with one bounded
  `PrefixCacheEntry` lifecycle.
- Keep device snapshots runner-owned behind opaque handles.
- Invalidate complete entries on physical block reuse and evict state by an
  explicit LRU budget.
- Keep admission and cached-state seeding in one engine step; a full prefill
  token budget must stop before another waiter is reserved.
- Treat compilation warmup as startup-only, and bind each runner snapshot to
  both its prefix hash and token count.
- Leave at least one prompt token executable because entries do not cache the
  following logits.
- Prove cache-hit parity on a tiny CUDA model and a real Qwen3.5 checkpoint;
  add churn and capacity tests.
- Add the style guide and put the core engine before advanced serving policy in
  the README reading path.

PR 2d branch: `agent/route-registry` at `f55f349`; open as draft PR
[#12](https://github.com/LiquidGunay/nano-vllm-jax/pull/12)

PR 2d is the route-ownership compression pass:

- Replace the runner boolean matrix with one `RouteKind`/`RouteSpec` registry
  shared by selection, preparation, dispatch, warmup, validation, and metrics.
- Make `ExecutionPlan` carry one route kind plus only per-step values, with no
  parallel route booleans.
- Run warmup through the same execution-plan dispatch used by serving and
  report the warmed route kinds directly.
- Remove the unreachable resident sampled warmup path; sampled requests do not
  use the scheduler's greedy-only static token-carry contract.
- Add a host-only executable trace of schedule, materialize, execute, and
  commit.

PR 2e completes the remaining configuration and model-policy compression:

- Split the flat `RuntimeConfig` into narrow model, capacity, compile, and
  kernel specs without compatibility adapters in the ordinary path.
- Group host batch metadata at the materialization boundary instead of growing
  optional parallel `DeviceBatch` fields.
- Move GDN serving-policy selection out of model math, retaining an explicit
  reference implementation and parity tests.
- Split runner/executor files only where an extracted module owns a complete
  state transition.

Purpose: implement the host/device ABI that speculation will extend later.

Completed and remaining scope across PR 2a-2e:

- Add `ScheduledRow`, `SchedulePlan`, `DeviceBatch`, `ExecutionPlan`,
  `RunResult`, `StepResult`, and `TokenEvent`.
- Move padding, `device_put`, and device-array caches from scheduler to runner.
- Replace the runner boolean matrix with a `RouteKind` registry.
- Use the same registry for selection, dependency checks, warmup, and metrics.
- Make `Engine.commit()` the single logical state transition.
- Move every generated token/ref from `Sequence` to `OutputBuffer`.
- Publish explicit token events; remove property-triggered synchronization.
- Add bounded/coalesced streaming, cancellation, finish reasons, truthful
  model id, worker-aware health, and verified shutdown.
- Replace split prefix dictionaries with the bounded `PrefixCacheEntry`
  lifecycle and runner-owned handles.
- Separate GDN reference, layout, and serving policy without changing math.
- Make `ServingOps` a thin adapter over `KernelPlan`.
- Broaden Ruff rules and add light typing for control-plane modules.

Merge gates:

- Importing/testing scheduler and sequence does not initialize JAX.
- No `device_put`, `device_get`, or JAX array lives in scheduler/sequence.
- `reachable promoted routes == warmed promoted routes` for every bucket.
- Forced pressure, cancellation, and prefix churn preserve base output parity.
- Prefix metadata, hybrid handles, host RAM, and device RAM remain bounded.
- Service tests cover slow consumers, disconnects, Unicode boundaries,
  failure health, and shutdown.
- Fresh primary and B=8 report-only comparisons remain within measured run
  variance; any material regression blocks merge.

The work is deliberately split at coherent ownership boundaries. Every part
must leave `main` fully working and reduce or consolidate concepts; a structural
change does not earn scope merely because it appears in issue #10.

### PR 3: reproducible artifact and one benchmark claim

Status: [ ] blocked by PR 2e

Purpose: make the repository able to support and reproduce one honest claim.

Committed pieces:

- One deterministic workload manifest and metric schema.
- One compact benchmark driver shared by JAX and vLLM adapters.
- A small `scripts/reproduce_claim.sh` entry point driven by the manifest.
- Locked environment metadata and an optional, isolated vLLM dependency.
- A concise benchmark report with exact model revision, hardware, software,
  measurement window, parity result, median, spread, and memory use.

The JAX and vLLM environments remain isolated because their Torch, Triton, and
cuDNN constraints can conflict. The reproduction script owns both environments
and writes raw results outside the repository. vLLM comparison is optional;
the JAX run and contract remain usable without installing vLLM.

Headline contract, subject to the fresh baseline:

- Qwen3.5-4B, BF16, B=1, prompt 64, output 64, greedy, prefix miss.
- Initialization, download, and compilation excluded from measurement.
- TTFT reported separately.
- Generated-token decode throughput is the single headline metric.
- Three measured repeats, exact output comparison, zero measured-phase JIT.

The script performs GPU/CUDA preflight and enforces the same 70% RAM guard. A
claim is not written if parity, memory, compilation, or variance checks fail.

### PR 4: generic drafter plus cheap target verifier

Status: [ ] blocked by PR 3

Purpose: prove cheap verification independently of MTP draft quality.

Scope:

- Add the JAX-traceable drafter/verifier protocols and pytrees.
- Add the speculative orchestrator as a new route-registry entry.
- Implement strict resident packed-prefix target verification and device-side
  accepted-prefix commit.
- Add an internal supplied-draft drafter for tests and diagnostics only.
- Test rejection, partial acceptance, full acceptance, bonus emission, tails,
  cancellation, stale speculative KV, and GDN/KV prefix parity.
- Measure freshly supplied high-accuracy drafts to isolate verifier cost.

Merge gates:

- Speculative output is token-for-token identical to the same base JAX target.
- One packed target pass is used per speculative group.
- No sequential target repair, ordinary-decode fallback, or host accept loop.
- Persistent state and compact outputs stay device-owned through the step.
- A 99%-accurate verified-draft diagnostic beats base decode on the primary
  lane. If it does not, stop here and profile the verifier ABI before adding
  MTP.

The supplied-draft adapter is never a public serving method and never supports
a production speed claim.

### PR 5: Qwen3.5 MTP drafter and speculative promotion

Status: [ ] blocked by PR 4

Purpose: attach MTP through the generic drafter ABI and determine where it is a
real speed win.

Scope:

- Load MTP parameters only when the optional drafter is configured.
- Seed persistent MTP KV during prefill and keep recursive proposal state on
  device.
- Implement the BF16 full-vocabulary reference proposal.
- Port the full-vocabulary INT8 proposal projection without changing target
  weights, target logits, verifier math, or output correctness.
- Expose one small typed configuration, conceptually:

  ```yaml
  speculation:
    drafter: mtp
    num_draft_tokens: 2
    proposal_dtype: int8
  ```

  Packed-prefix target verification is the only production verifier and is not
  a user-selectable implementation string.

- Compare clean JAX base, JAX MTP, pinned vLLM base, and vLLM MTP on the same
  single workload contract.
- Run non-claim correctness/quality diagnostics on 0.8B and 2B to investigate
  the known width-sensitive numerical drift.

Merge/promotion gates:

- Exact target output on a multi-prompt B=1 manifest.
- Acceptance/rejection counters account for every proposed token.
- No fallback labels and no measured-phase compilation.
- INT8 reports proposal KL, top-1 flips with BF16 margins, and acceptance delta;
  near-tie flips are acceptable, unexplained quality loss is not.
- Peak host RAM remains below the 70% guard and device memory retains explicit
  headroom.
- JAX MTP beats the freshly measured clean JAX base and vLLM without MTP on the
  primary workload before the README calls it a speedup.
- vLLM MTP is reported as a framework comparison, not used to relax the JAX
  correctness or speed gates.

If only 4B passes the speed gate, MTP is documented as a 4B/B=1 route. Smaller
models remain supported by base decode and their speculative limitation is
reported rather than hidden.

## Experimental Transplant Map

Reimplement against the clean ABI:

- native vocab-major Triton top-1 for tied weights;
- no materialized tied-head copy;
- optional B=1 packed GDN input projection, only if fresh integrated A/B keeps
  the B=8 route healthy;
- strict resident packed-prefix target verification;
- persistent MTP KV with prefill seeding;
- full-vocabulary INT8 proposal weight and row scale;
- exact counters, no-fallback assertions, and multi-prompt parity tests.

Do not transplant:

- the experimental `ModelRunner` or `ModelExecutor` wholesale;
- flat MTP fields on the target model config;
- environment aliases and verifier implementation switches;
- adaptive or diagnostic verifier fallbacks;
- fixed token-ID vocabulary prefixes;
- sequential repair, forced acceptance, or unverified token append;
- benchmark dumps, profile dumps, and optimization-log history.

## Validation and Resource Policy

- GPU visibility: `nvidia-smi` plus a CUDA-only JAX smoke before GPU work.
- JAX correctness/performance: `JAX_PLATFORMS=cuda`; never CPU fallback.
- System RAM: guarded subprocesses, default hard stop at 70%.
- GPU memory: estimate before load, record peak, and retain a documented safety
  margin rather than merely avoiding OOM.
- Large models run one process at a time; caches and artifacts stay under
  `/mountpoint/.exp`.
- New JIT boundaries pass small and medium diagnostics before the full lane.
- A miss-heavy prefix-publication diagnostic reports GDN snapshot copy and
  allocation cost before performance promotion. Move snapshots to a
  preallocated slot table only if integrated profiling shows material overhead.
- Raw results/profiles stay external. Only manifests, schemas, scripts, concise
  summaries, and tests are committed.
- Every performance comparison records exact commit, model revision, package
  lock, hardware, config hash, repeat count, parity, and JIT growth.

## Progress Log

- [x] Read the cleanup diagnosis and inspect current `main` and the experimental
  branch at the revisions above.
- [x] Define the target base and speculative ABIs.
- [x] Define PR boundaries and merge gates.
- [x] Confirm the proposed supported model set and headline workload.
- [x] Capture the fresh external baseline (guarded mainline failure recorded).
- [x] Open PR 1 as draft PR #7.
- [x] Address PR 1 review: dual-EOS termination, complete architecture gate,
  capacity credits, first-fit admission, derived shorthand warmup, metadata-first
  Hub resolution, and live-page FlashInfer metadata.
- [x] Address PR 1 follow-up: use a bucket-independent non-split FlashInfer
  decode plan, mask inactive fused-append rows in the CUDA binding, and cover
  128-page sparse-live metadata plus padded-row cache integrity. Same-envelope
  B=8 before/after output was exact with no measured JIT growth or speed loss;
  the guarded validation passed.
- [x] Merge PR 1 as #7.
- [x] Open, review, and merge host-only scheduling/device materialization PR
  #8.
- [x] Open typed step/commit/output ownership draft PR #9. Guarded control,
  CUDA device-carry, kernel/GDN/layer, real-weight, and live 0.8B engine-smoke
  checks passed; combined heavyweight parity was split at the RAM floor.
- [x] Expand PR #9 with bounded/coalesced streaming, total-request pressure,
  cancellation, worker-aware health, and verified shutdown. The expanded
  control suite passes 57 tests, 205 tests collect, and a fresh guarded 0.8B
  CUDA smoke remains correct.
- [x] Address both PR #9 review rounds and merge it at main commit `850cd3e`.
- [x] Review issue #10 and turn its critique into an explicit conceptual-
  compression contract and a core-versus-advanced reading path.
- [x] Open prefix-lifecycle draft PR #11 from
  `agent/prefix-cache-lifecycle`: bounded KV/GDN entries, runner-owned
  snapshots, issue #10 style guide, and guarded tiny/real-model cache-hit
  parity. The 0.8B smoke matched exact tokens with zero measured JIT growth;
  the guarded control suite passed 69 tests and 214 tests collect.
- [x] Address PR #11 lifecycle review: prevent zero-token admission, reject
  late warmup, validate snapshot hash plus token count, exercise LRU pressure,
  and keep validation local to the relevant change. The expanded guarded
  suite passes 71 tests, 33 runner/cache tests, and 217 tests collect; the real
  0.8B cache-hit smoke remains exact with zero measured JIT growth.
- [x] Start PR 2d locally: one typed route registry now drives selection,
  dispatch, validation, warmup, and labels; it removes the unreachable
  resident sampled warmup route. Tiny greedy/sampled and real 0.8B checks are
  exact with zero measured compilation. B=1 is within `+0.7%` of PR #11 and
  B=8 within `-0.5%`, both inside observed variance and under the RAM guard.
- [x] Merge PR #11 at main commit `0f6fbbc`, rebase the route-registry change,
  and open draft PR #12 at `f55f349`. The rebased focused suite passes under a
  3 GiB RAM guard; the patch remains net-negative in size.
- [ ] Review and merge PR #12, then complete PR 2e's config/model-policy
  compression before the reproducibility artifact or speculative routes add
  new execution choices.
