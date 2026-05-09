# MTP fused K=1 debug plan

## Current status

The correctness-safe seeded K=1 path is `mtp1_commit_select_greedy_step_jit`.
It passed warmed TPU exact-token checks against baseline and HF logit sanity for:

- B=1, prompt length 64, 128 generated tokens
- mixed B=2, prompt lengths `32,64`, 128 generated tokens/request
- mixed B=4, prompt lengths `32,64,96,128`, 128 generated tokens/request

The safe path is still slower than baseline:

- B=1 decode speedup: 0.801x
- mixed B=2 decode speedup: 0.769x
- mixed B=4 decode speedup: 0.688x

This is expected for commit-select. With acceptance rate `a`, commit-select runs
one current-token target decode and an additional draft-token target decode for
accepted rows. If that second decode costs about the same as the first, the
ideal model-only speedup is approximately:

```text
speedup ~= (1 + a) / (1 + a) = 1.0x
```

Any MTP head, selection, host, or scheduler overhead pushes it below 1.0x.

## Fused one-pass failure

The fused one-pass path is unsafe today and is blocked by default. It can only
be enabled for experiments with:

```text
NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1
```

Seeded experiments additionally require:

```text
NANO_VLLM_JAX_MTP_ALLOW_SEEDED_ONE_PASS_K1=1
```

Observed warmed TPU failures:

- B=1 diverged at generated token 101: baseline token `871`, MTP token `11`
- mixed B=2 diverged at generated token 79 for request 1: baseline token
  `7359`, MTP token `68146`
- B=1 without post-bonus seeding diverged at generated token 37: baseline token
  `22513`, MTP token `73982`

The B=1 trace matched visible tokens through token 100, then the next accepted
step emitted the wrong first token. That means either:

- the committed fused state after a prior accepted draft differs from sequential
  decode state, or
- the fused verifier slot-0 logits differ from a canonical one-token decode from
  the same pre-step state.

Switching the fused verifier's two-token block from cached-prefill metadata to
decode metadata did not resolve the B=1 divergence, so the bug is not fixed by a
simple prefill/decode metadata toggle.

## Required parity diagnostic

The first diagnostic hook is implemented in the runner:

```text
NANO_VLLM_JAX_MTP_ALLOW_UNSAFE_ONE_PASS_K1=1
NANO_VLLM_JAX_MTP_PARITY_DEBUG=1
```

When unsafe one-pass K=1 is selected, the runner also executes
`mtp1_commit_select_greedy_step_jit` from the same pre-step cache/hybrid state
and prints the first target/bonus/acceptance/next-draft mismatch:

```text
[MTP_PARITY] one_pass_vs_commit_select ...
```

Use `NANO_VLLM_JAX_MTP_PARITY_STOP=1` to stop at the first mismatch. This is
intentionally a diagnostic-only path and is not used for throughput.

The same hook also reports state drift between the fused and commit-select
outputs:

```text
[MTP_PARITY_STATE] one_pass_vs_commit_select ...
```

It compares:

- current and draft KV slots in `k_cache`
- current and draft KV slots in `v_cache`
- full selected convolution state
- full selected recurrent state

Use `NANO_VLLM_JAX_MTP_PARITY_STOP_STATE=1` to stop on the first state mismatch.
Use `NANO_VLLM_JAX_MTP_PARITY_STATE_THRESHOLD=<float>` to ignore small numeric
differences.

The next, deeper harness should pause on accepted seeded K=1 steps and compare
the fused one-pass verifier against sequential commit-select from the same
pre-step state.

For each accepted row, compare:

```text
target_token_fused == target_token_commit_select
bonus_token_fused == bonus_token_commit_select
next_draft_fused == next_draft_commit_select
topk(logits_fused_slot0, 10) == topk(logits_commit_current, 10)
topk(logits_fused_slot1, 10) == topk(logits_commit_draft, 10)
KV[current_slot]_fused ~= KV[current_slot]_commit_select
KV[draft_slot]_fused ~= KV[draft_slot]_commit_select
hybrid_after_current_fused ~= hybrid_after_current_commit_select
hybrid_after_draft_fused ~= hybrid_after_draft_commit_select
```

The first failing comparison determines the fix:

- slot-0 logits differ: attention or linear prefix semantics are not equivalent
  for multi-token fused decode.
- slot-1 logits differ while slot-0 matches: draft-position state is wrong.
- logits match but KV differs: cache write/restore semantics are wrong.
- KV/logits match but hybrid differs: prefix hybrid state extraction is wrong.

## Work items

1. Add a `--mtp-parity-debug` benchmark mode that runs baseline, commit-select,
   and fused one-pass from cloned pre-step cache/hybrid state.
2. Capture only the first accepted seeded step and the first step after two
   consecutive accepted seeded steps to keep the diagnostic small.
3. Report max-abs and MSE for logits, KV slots, conv state, and recurrent state.
4. Keep throughput claims invalid until exact token match and parity diagnostics
   pass.
5. After fused parity passes, re-enable seeded one-pass by default and rerun
   warmed valid B=1/B=2/B=4 throughput.

## Speed target

For fused one-pass K=1 with acceptance rate `a`, expected speedup is:

```text
speedup ~= (1 + a) / fused_two_token_decode_cost
```

At the observed B=2 acceptance rate of 44.32%, fused one-pass can beat baseline
if the two-token verifier costs less than about 1.44x a baseline decode,
including overhead. That is the only current K=1 path with realistic headroom.
