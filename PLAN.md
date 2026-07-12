# Mainline Cleanup and Speculative Decoding Plan

Last updated: 2026-07-12

## Goal

Build the next mainline in three stages:

1. Make the non-speculative engine correct, explicit, and easy to teach.
2. Make its single performance claim reproducible from a clean checkout.
3. Add speculative decoding behind a generic drafter/verifier boundary, with
   Qwen3.5 MTP as the first drafter.

Every merged PR must leave `main` usable on its own. We will open and review
one PR at a time, then build the next PR from the newly merged `main`.

## Source Revisions

- Clean mainline: `origin/main@ef0eead`
- Experimental evidence: `experimental/mtp-prefill-verifier-speed@7ced216`
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
3. Admit a waiting request only when its complete reservation can be held.
4. Never evict a request after generation begins.
5. Release the reservation on finish, cancellation, or error.

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

An entry is published only when target KV and GDN state represent the same
complete prefix. Reusing any physical block in the entry atomically removes
the entry, releases its runner-owned state handle, and updates statistics.
The cache has an explicit entry/state budget and a churn test.

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

Status: [ ] not started

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

Status: [ ] not started

Purpose: make the ordinary target model truthful and safe before moving state
between components.

Scope:

- Remove recompute preemption and add worst-case block reservation.
- Make sequence ids and block size engine-local.
- Introduce frozen model/capacity/compile/kernel/runtime specs.
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

Status: [ ] blocked by PR 1

Purpose: implement the host/device ABI that speculation will extend later.

Scope:

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

If review size requires it, this PR may be split after the type extraction,
but both halves must leave `main` fully working and use the same final ABI.

### PR 3: reproducible artifact and one benchmark claim

Status: [ ] blocked by PR 2

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
- Raw results/profiles stay external. Only manifests, schemas, scripts, concise
  summaries, and tests are committed.
- Every performance comparison records exact commit, model revision, package
  lock, hardware, config hash, repeat count, parity, and JIT growth.

## Progress Log

- [x] Read the cleanup diagnosis and inspect current `main` and the experimental
  branch at the revisions above.
- [x] Define the target base and speculative ABIs.
- [x] Define PR boundaries and merge gates.
- [ ] Confirm the proposed supported model set and headline workload.
- [ ] Capture the fresh external baseline.
- [ ] Open PR 1.
