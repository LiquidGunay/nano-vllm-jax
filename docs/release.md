# v0.1 Release Decisions

Issue #17 is the v0.1 hardening checklist. This note records the bounded
structural decisions that are easy to lose in a large cleanup diff.

## Landed

- Persistent KV memory has one target owner and one optional predictor owner;
  the removed NHD sidecar had no reads.
- Offline batch admission is atomic, HTTP capacity is validated per request,
  and strict token and sampling checks happen before scheduler mutation.
- `EngineService`, offline generation, and the manual stepping API have an
  explicit single-writer contract. Engine cleanup is idempotent and releases
  heavy references without a process-lifetime `atexit` root.
- Warmup uses disjoint physical block tables. Impossible bucket combinations
  are reported as skipped instead of aliasing cache blocks.
- Server import constructs only the HTTP transport. CUDA/JAX configuration,
  model loading, service startup, and shutdown happen in `main()`.
- Python 3.11 plus `uv.lock` is the reproducible environment. Local checks are
  intentionally not CI.

## Audited And Retained

- `KVCacheState` is the typed reference-model input that couples cache storage
  with one call's block table, lengths, and optional hybrid state. It is not a
  second runner-owned cache.
- The route registry retains its distinct state transitions. Logits, sampled,
  table-state, resident-metadata, burst, and speculative calls have different
  executor ABIs or commit obligations; merging their names would hide those
  differences rather than remove code.
- `RuntimeSpec` remains the startup aggregate because runner construction
  coordinates model, capacity, compilation, kernels, and the optional drafter.
  Operation helpers continue to receive narrower specs where they own only one
  concern.

## Explicitly Declined For v0.1

- Splitting `runner.py` or `executor.py` by file size. Their current boundary is
  state ownership versus compiled execution; another layer would mostly
  forward calls. A later split must move a complete state transition and delete
  existing coordination.
- A broad model-policy rewrite. The dead attention branch and reused PRNG key
  were removed; `ServingOps` already keeps backend selection outside model
  math. Further movement needs a concrete duplicated policy to delete.
- Renaming or collapsing routes solely to reduce their count. The registry,
  selection, warmup, validation, and tests already share the same typed names.

The `v0.1.0` tag is created only from reviewed, merged `main` after the guarded
CUDA suite and four-route benchmark have been recorded.
