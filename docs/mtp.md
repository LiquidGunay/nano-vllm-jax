# MTP Speculative Decoding

MTP is experimental. K is limited to 1 or 2, and correctness is the gating criterion for all speed work.

## Terms

Target token:

```text
the token selected from canonical target-model logits for the current position
```

Draft token:

```text
a token proposed by the MTP head and verified against target-model logits
```

Bonus token:

```text
the target-model token sampled after an accepted draft prefix; emitted now, cached on the next decode step
```

Accept:

```text
greedy target argmax for the verified position equals the draft token
```

Reject:

```text
greedy target argmax differs from the draft token; emit the target token and do not commit the rejected draft state
```

## Canonical greedy acceptance

For greedy decoding:

```text
accepted_i = argmax(target_logits_i) == draft_i
```

Target logits must come from `model.forward_step` through `ModelExecutor`. Hidden states are allowed for MTP seeding, not as a substitute for verifier logits unless explicitly validated.

## K=1 state flow

Initial state:

```text
sequence already contains last_token
KV/hybrid state is committed through the previous scheduled model step
```

Verifier flow:

```text
1. target decode last_token -> target token for next position
2. compare target token with draft_1
3. if reject: emit target token, commit only canonical last_token decode state
4. if accept: decode draft_1 -> bonus token
5. commit state through draft_1
6. emit [draft_1, bonus]
```

K=1 commit invariant:

```text
accepted path commits through draft_1; rejected path commits through last_token only
```

## K=2 state flow

Intended verifier input:

```text
[last_token, draft_1, draft_2]
```

Logit interpretation:

```text
position 0 logits verify draft_1
position 1 logits verify draft_2
position 2 logits produce bonus if both drafts accept
```

K=2 full-accept commit invariant:

```text
commit KV/hybrid state through draft_2, emit [draft_1, draft_2, bonus]
```

K=2 rejection invariant:

```text
commit only the accepted target prefix; rejected and later verifier state must not become logically reachable
```

For a second-token reject, a correctness-first implementation may repair by falling back to canonical sequential decode rather than trying to partially select fused verifier state.

## State commit rules

Accepted draft tokens are both emitted and already processed by the verifier. They require Python-side block metadata commit if they complete a block.

The bonus token is emitted but not processed by the target model as an input token yet. It becomes `last_token` for the next scheduled decode.

Commit invariant:

```text
number of processed emitted tokens = len(emitted_tokens) - 1
```

## Current correctness posture

- K=1 safe mode is the baseline when post-bonus reseeding is disabled.
- K=1 optimized paths require exact token parity before their speed numbers are meaningful.
- K=2 is under validation and must prove equivalence to sequential target decode for logits and committed state.
- Post-bonus reseeding and rowwise partial acceptance are correctness risks unless specifically validated on TPU.

## Minimum K=2 validation

A K=2 implementation should pass these checks on TPU before speed conclusions:

- position 0 verifier logits match normal target decode of `last_token`,
- position 1 verifier logits match sequential target decode after `draft_1`,
- full-accept KV/hybrid state matches sequential target decode through `draft_2`,
- first-token rejection continuation matches baseline,
- second-token rejection continuation matches baseline,
- mixed batch rows do not leak accepted or rejected state across rows.
## Current K=1 correctness policy

The continuous seeded path (`NANO_VLLM_JAX_MTP_SEED_AFTER_BONUS=1`) must use
the sequential commit-select verifier by default. Warmed TPU probes showed that
the fused two-token prefix verifier can keep visible tokens matched for many
steps and then diverge after a visibly correct accepted bonus, which means the
next verifier can accept against a drifted or draft-contaminated state.

The runner therefore blocks `mtp1_two_decode_greedy_step_jit` by default, unless
`NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1` is set explicitly for parity and
performance experiments. For seeded bonus experiments, the additional
`NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1=1` opt-in is also required. The
correctness path is
`mtp1_commit_select_greedy_step_jit`: decode the current token first, compare it
with the MTP draft, decode the accepted draft only for accepted rows, and select
the committed KV/hybrid state on device.

Validated TPU checkpoint:

- `Qwen/Qwen3.5-2B`, bf16, warmed, decode-jit, B=1, 128 generated tokens:
  exact MTP/baseline token match and HF logit sanity passed with continuous
  seeded bonus drafts.
- `Qwen/Qwen3.5-2B`, bf16, warmed, decode-jit, mixed B=2, prompt lengths
  `32,64`, 128 generated tokens/request: exact MTP/baseline token match and HF
  logit sanity passed with continuous seeded bonus drafts and rowwise acceptance.

The fused one-pass K=1 verifier remains available only as an explicitly unsafe
experiment for seeded bonus drafts. It needs a dedicated equivalence test that
compares verifier slot-0 logits against a one-token baseline decode from the
same state before it can be re-enabled for serving.

Attempted fix that did not resolve the issue: switching the fused verifier's
two-token block from cached-prefill metadata to decode metadata still reproduced
the same seeded B=1 divergence at generated token 101. That points away from a
simple attention metadata mode bug and toward prefix-state/logit equivalence
drift in the fused path under consecutive accepted steps.

Additional control: warmed B=1 without post-bonus seeding still diverged with
the fused one-pass verifier at generated token 37 (`22513` baseline versus
`73982` MTP). That makes the fused verifier unsafe as a default even outside the
continuous seeded path.
