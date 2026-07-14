# Speculative Verification

The optional speculative route separates proposals from target-model
verification. A drafter produces only full-vocabulary token ids; it does not
decide acceptance or modify target state.

```text
Drafter -> DraftProposal[B, K] -> packed target verifier
                                      |
                                      v
                           VerificationResult[B, K+1]
```

For each row, the verifier runs `[current, draft_1, ..., draft_K]` through one
prefill-shaped target forward. Logit position `i` predicts the token after
input position `i`, so positions `0..K-1` verify the drafts and position `K`
provides the bonus token. If `n` drafts match, the result emits
`drafts[:n] + target[n]`. The last token is a correction when `n < K` and a
bonus when `n == K`.

Acceptance and state selection happen on device. The verifier selects GDN
prefix state `n`, advances the resident length by `n + 1`, and leaves the last
emitted token unprocessed for the next step, matching ordinary decode. Full-
attention KV writes beyond the committed length may remain dirty: attention is
bounded by the resident length and later writes replace those invisible slots.

`DraftProposal` and `VerificationResult` are JAX pytrees. The latter contains
the padded emitted-token buffer, emitted and accepted counts, and the next
resident token. Python reads only the compact counts needed by the logical
commit; generated token values remain deferred device references.

## Current Limits

- The route is optional and has no server-YAML surface.
- It currently requires greedy decoding with ignored EOS, dense resident rows,
  device token carry, and resident metadata.
- `K` is at most 15 because the promoted packed-prefix GDN kernel covers at
  most 16 input positions including `current`.
- A request tail shorter than `K + 1` uses the ordinary decode route.
- `SuppliedDrafter` is a test/diagnostic adapter, not a serving method or an
  MTP speed claim.

Packed BF16 execution is mathematically the same target computation but need
not be bitwise identical to width-1 decode. On the 0.8B diagnostic, later
packed and width-1 distributions retained the same top-1 with KL `0.00252`
and JS `0.00063`; near-tied tokens can still flip. Primary promotion therefore
checks exact 4B output parity and reports KL when a smaller model differs.

The next pass will implement Qwen3.5 MTP behind this proposal boundary. Its
persistent draft KV state, prefill seed, and recursive proposal must stay on
device and compose with verification inside the compiled transition.
