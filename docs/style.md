# Repository Style

Nano-VLLM-JAX is both a real serving engine and an explanation of one. Its
style target is **conceptual compression**: a reader should need few objects,
few routes, and one visible implementation of each idea to follow a token from
admission to commit.

This is not code golf. Correctness checks, explicit ownership, validation at
trust boundaries, and hardware-specific calibration stay even when they cost
lines. We remove accidental machinery, not necessary guarantees.

## Rules For Every Change

1. **Trace the existing path first.** Put the change at the owner of the state
   or transition it affects. Fix a shared cause once instead of adding guards
   at its callers.
2. **Fit an existing concept or replace it.** Do not add an independent flag,
   optional field, or specialized method when the behavior is another case of
   an existing route, state transition, or result type.
3. **Prefer deletion and reuse.** Reuse the repository's types, Python/JAX
   primitives, and installed dependencies. New abstractions must remove more
   branching or duplication than they introduce.
4. **Keep one ordinary path.** Clean main exposes one promoted serving policy.
   Alternative kernels and diagnostic routes stay outside the public config
   and outside the core reading path.
5. **Make illegal states difficult to express.** Internal concrete objects are
   fully initialized and use direct attribute access. Do not add `getattr()` or
   `hasattr()` to tolerate partial test objects or historical object shapes.
6. **Name the concept, not its optimization history.** Prefer names such as
   `PACKED_PREFILL` and `RESIDENT_DECODE` over names that concatenate every
   storage, sampling, and token-carry detail.
7. **Leave one small executable check.** Non-trivial control flow needs a
   focused test. An optimized mathematical path needs parity with its reference
   semantics. Tests should construct valid objects rather than weaken
   production invariants.

The smallest correct diff wins only after the full request and state flow is
understood. A short patch in the wrong layer is future complexity.

## Boundaries

The core transition remains:

```text
schedule -> materialize -> execute -> commit
```

- The scheduler owns host-side work selection and capacity.
- The runner owns device materialization and accelerator state.
- The executor performs compiled model transitions.
- The engine owns the logical commit and user-visible result.

A control-plane function should not select a route, prepare its batch, execute
it, and commit it. If one function crosses these boundaries, split by state
transition rather than by arbitrary file length.

Host and device copies of the same fact need an explicit owner and update
point. Do not add another optional `*_host` mirror merely because one caller
needs a Python value; carry the typed host plan or group host metadata at a
single boundary.

Extract a module only when it owns a coherent concept or transition. A facade
that forwards calls without reducing concepts makes the reading path longer
and should not exist.

## Routes And Configuration

A route is one typed choice, not a mode plus a parallel boolean matrix.
`routes.py` owns the shared route specification used by selection, preparation,
execution, warmup, executor validation, and route labels. New execution work
must extend or consolidate that registry instead of adding a route-specific
boolean.

Configuration follows the same rule:

- Public configuration describes workload and capacity, not kernel search.
- Implementation policy has one internal promoted plan.
- Components receive the narrowest spec they need.
- Do not add a field to the flat `RuntimeConfig` as a convenient message bus.

The target shape is a small aggregate of model, capacity, compile, and kernel
specs, with callees receiving only their relevant part. This is a structural
refactor, not a reason to build compatibility adapters in unrelated PRs.

## Model And Performance Code

Model functions should read like the model: project, normalize, attend or
recur, gate, and project out. Route eligibility, fallback policy, bucket
selection, and kernel naming belong in the runner or operation layer.

Every promoted optimization has three visible pieces:

```text
reference semantics -> promoted implementation -> parity test
```

The reference path favors clarity. The promoted path may be low-level or
hardware-specific, but its adapter should expose the same operation rather than
leak kernel policy through the model. Experiments become promoted code only
when they replace or consolidate a path; they do not accumulate beside it.

Comments explain ownership, shapes, synchronization, or a non-obvious
invariant. They do not narrate implementation history. A deliberate simple
algorithm with a real scaling ceiling should name that ceiling and the event
that would justify replacing it.

## Reading Path

The first reading path is limited to the generation algorithm:

1. `engine.py` — request lifecycle and commit.
2. `scheduler.py` — prefill/decode selection.
3. `block_manager.py` — paged capacity and prefix reuse.
4. `batch.py` and `device_batch.py` — host plans and JAX shapes.
5. `runner.py` — route selection and device state.
6. `model.py` — one model transition.

Service, configuration, warmup, resident metadata, and specialized kernels are
an advanced path. Documentation should not require those concepts before a
reader can explain one generated token.

## Tracked Structural Work

These are repository-level refactors, not requirements to expand the scope of
every feature PR:

- Split the flat runtime configuration into narrow model, capacity, compile,
  and kernel specs.
- Replace optional parallel host fields in `DeviceBatch` with one explicit
  plan/host-metadata relationship.
- Move serving policy out of GDN and other model-math functions.
- Split runner/executor code only at coherent state transitions; create no
  forwarding-only modules.

Until each lands, nearby changes should avoid deepening the corresponding
debt.

## Review Check

Before merging, a reviewer should be able to answer yes to these questions:

- Can one concept be removed for every concept added?
- Is there one owner and one update point for new state?
- Does the ordinary route remain obvious without reading deployment policy?
- Did the change avoid new route booleans, defensive internal attribute access,
  and duplicated host/device facts?
- Does model code still describe model math?
- Is the smallest relevant correctness or parity check present?
- Would a newcomer know which six modules to read first?

These are review heuristics, not line-count gates. Complexity that is necessary
for correctness or measured performance is allowed; it must remain localized,
named, and testable.
