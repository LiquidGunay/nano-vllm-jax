# Persistent MTP

Persistent MTP is an experimental, constructor-time overlay on the promoted
greedy path:

```python
from nanovllm_jax import DrafterConfig, LLM, SamplingParams

llm = LLM(
    "Qwen/Qwen3.5-4B",
    prefix_cache=False,
    drafter=DrafterConfig.mtp(width=3),
)
outputs = llm.generate(
    ["The future of small serving engines"],
    SamplingParams(temperature=0, max_tokens=64, ignore_eos=True),
)
```

`DrafterConfig` is frozen. Its width owns compiled verifier shape, scheduler
lookahead, request-capacity padding, the resident proposal table, and warmup.
There is no post-construction drafter installation or host `propose()` call.

For draft width `K`, the scheduler derives three physical allowances:

- prefill allocates `K - 1` future slots because the first draft comes directly
  from the final prompt predictor state and only the remaining drafts write KV;
- decode allocates `2K` lookahead slots for current-plus-`K` target verification
  followed by as many as `K - 1` recursive predictor writes;
- lifetime reservation adds `max(0, K - 2)` tokens because the final eligible
  `K + 1` output group can leave that many predictor writes beyond the logical
  end.

```text
packed prefill
  -> target hidden + shifted prompt tokens
  -> persistent one-layer MTP KV
  -> recursive drafts[B, K]
  -> resident draft table

decode transition
  -> read resident drafts
  -> one packed target pass over [current, draft_1, ..., draft_K]
  -> device accept and target GDN-prefix selection
  -> advance persistent MTP KV at the selected prefix
  -> recursively create and store the next drafts
  -> return compact verified tokens and counts
```

The packed target transition consumes only draft ids and target state. MTP
refresh is composed around that boundary inside the same JIT, so another
device-resident drafter can reuse verification without a host callback.

The checkpoint's MTP layer has its own projection, attention, MLP, and norm
weights. It shares the target embedding/vocabulary weight instead of allocating
a second `[V, H]` table. Prefill seeds MTP KV before the first decode step;
decode refreshes MTP KV and proposals inside the same JIT boundary as target
verification. Python sees only compact emitted/accepted counts and deferred
token references.

For each row, target logit position `i` predicts the token after input position
`i`. If `n` drafts match, the transition emits `drafts[:n] + target[n]`. The
last token is a correction when `n < K` and a bonus when `n == K`. The selected
GDN state and resident length advance by `n + 1`. Rejected full-attention KV
writes may remain beyond that committed length because later attention cannot
observe them and subsequent writes replace them.

`RunResult` and `StepResult` report draft tokens, accepted draft tokens, and
the number of target positions evaluated by verification separately from the
scheduler's input-token count. Streaming attaches those step-wide counters to
the first token event from the step and uses zero for later events from it.

## Limits and correctness

- MTP is not part of `server.yaml`; omitting `drafter` leaves the promoted base
  server unchanged.
- It currently requires JIT packed prefill, prefix caching disabled, greedy
  sampling, ignored EOS, dense resident decode rows, device token carry, and
  resident metadata. Static incompatibilities fail during construction and
  incompatible requests fail at admission.
- Exact decode buckets are required for every admitted batch cardinality, so
  configuring MTP cannot silently select a padded ordinary route.
- `kv_cache_bytes` caps the combined canonical target and predictor KV arrays;
  enabling MTP can therefore reduce their shared block count.
- Tail eligibility is batch-wide. If any row has fewer than `K + 1` output
  tokens remaining, the whole batch uses the already-warmed ordinary route and
  discards its proposals; longer rows do not resume MTP after the short row
  finishes.
- `K <= 15`; the packed target width includes one additional current token.
- The loader targets Qwen3.5 dense checkpoints with one tied-embedding MTP
  layer; model-size support follows that checkpoint contract rather than a
  size-specific verifier path.

Packed BF16 verification can reorder near-tied logits relative to width-1
decode even when the distributions are close. Validation therefore treats
exact token parity as the strongest result, but permits isolated top-1 changes
only when an explicit full-vocabulary KL/JS and logit-margin check shows that
they are numerical near ties. Acceptance, state advancement, and emitted-token
accounting must still follow the packed target distribution exactly.

The committed B=1 claim has one such content-addressed equivalence in
[`parity_evidence.json`](../benchmarks/parity_evidence.json): two 64-token
outputs differing only at index 44. The benchmark accepts those two hashes in
either direction and rejects every other mismatch; this is not a general
tolerance for token drift.

Because of that numerical limitation, MTP remains experimental even when a
checkpoint passes an exact generation run. Promotion checks use identical base
and MTP prompts, compare every emitted token, and inspect both full-vocabulary
distributions at the first mismatch rather than accepting token drift alone.
