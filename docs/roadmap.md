# Roadmap

The near-term roadmap is correctness first, then speed. MTP work should stay constrained to K=1 or K=2 until the target-model equivalence story is fully validated.

## Priority 1: preserve canonical target decode

- Keep `ModelExecutor` as the canonical execution boundary.
- Keep verifier logits sourced from canonical target logits.
- Avoid hidden-derived verifier logits except for diagnostics.
- Preserve exact token parity against non-speculative baseline before measuring speed.

## Priority 2: validate K=2

- Prove multi-token verifier decode equals sequential one-token target decode for logits and committed cache state.
- Test full accept, first-token reject, second-token reject, and mixed-row batches.
- Confirm full-attention decode masking and linear-attention recurrent state are position-causal for query length greater than one.
- Keep K=2 disabled or experimental until TPU validation passes.

## Priority 3: repair partial acceptance

- Prefer discard-and-repair from canonical state for rejected rows until rowwise commit selection is proven.
- If rowwise acceptance is reintroduced, make KV and hybrid state selection explicit per accepted prefix.
- Do not let rejected verifier writes become logically reachable.

## Priority 4: improve speed after correctness

- Reduce fallback rate before optimizing microseconds.
- Use larger prefill buckets in benchmark configurations where capacity allows.
- Minimize host synchronization in token materialization and acceptance accounting.
- Explore groupwise acceptance only after rowwise state correctness is solved.

## Non-goals for this cleanup

- No source-code edits.
- No local benchmark or test runs.
- No changes to executor, runner, tests, or benchmark scripts.
