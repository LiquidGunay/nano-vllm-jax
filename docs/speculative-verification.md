# Persistent MTP

MTP is an optional constructor-time overlay on the promoted greedy path:

```python
from nanovllm_jax import DrafterConfig, LLM

llm = LLM(
    "Qwen/Qwen3.5-4B",
    prefix_cache=False,
    drafter=DrafterConfig.mtp(width=3),
)
```

`DrafterConfig` is frozen. Its width owns compiled verifier shape, scheduler
lookahead, request-capacity padding, the resident proposal table, and warmup.
There is no post-construction drafter installation or host `propose()` call.

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
scheduler's input-token count.

## Limits and correctness

- MTP is not part of `server.yaml`; omitting `drafter` leaves the promoted base
  server unchanged.
- It currently requires JIT packed prefill, prefix caching disabled, greedy
  sampling, ignored EOS, dense resident decode rows, device token carry, and
  resident metadata. Static incompatibilities fail during construction.
- A tail with fewer than `K + 1` output tokens remaining uses the already-warmed
  ordinary greedy route.
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
