# MTP reference semantics

Date: 2026-05-08

Scope: research notes only. This document records reference behavior from primary sources and separates it from behavior inferred from this repo's Qwen/Qwen3.5 MTP experiments. It is not an implementation plan and does not validate any local benchmark result.

## Terms

- `target model`: the canonical model whose output distribution must be preserved.
- `draft` or `proposal`: one or more speculative tokens predicted before target verification.
- `bonus token`: the first target-model token after a fully accepted speculative prefix.
- `K`: number of proposed speculative tokens per verification step.
- `dense-only model`: model whose generation state is only attention KV cache plus ordinary sequence metadata.
- `hybrid model`: model with attention KV plus non-attention recurrent/convolutional state, such as Qwen3-Next/Qwen3.5-style Gated DeltaNet layers.

## Canonical behavior from sources

### Speculative decoding must preserve target-model output

The canonical speculative decoding contract is losslessness: target sampling with speculative decoding should match target sampling without speculative decoding, up to numerical precision. The original speculative decoding paper states that the method can compute several tokens in parallel without changing the output distribution, and vLLM documents this as the intended guarantee for its speculative framework.

For greedy decoding in this repo, the practical equivalence contract is stricter: the speculative path must produce exactly the same token IDs as the baseline target decode for the same prompt, model weights, dtype/backend, batch-shape policy, and sampling settings.

Sources: [Speculative Decoding paper](https://arxiv.org/abs/2211.17192), [vLLM speculative decoding lossless guarantees](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding).

### vLLM speculative decoding separates proposer, scorer, and sampler

vLLM's older `SpecDecodeWorker` API describes three roles:

- a proposer that produces speculative tokens,
- a scorer that computes target-model probabilities for those tokens,
- an acceptance sampler that decides which speculative tokens are accepted.

vLLM's public docs also expose `method: "mtp"` and `num_speculative_tokens` in `speculative_config`; MTP can often omit a separate draft `model` because the target model may contain native MTP support.

Sources: [vLLM `SpecDecodeWorker` API](https://docs.vllm.ai/en/v0.9.0/api/vllm/spec_decode/spec_decode_worker.html), [vLLM speculative config docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

### vLLM multi-step workers require scheduler KV lookahead

vLLM's `MultiStepWorker` is documented as a worker variant that can run multiple forward passes in one call when the scheduler has already allocated space for the extra KV. This maps directly to a local requirement: any speculative verifier that can physically write draft/bonus positions must reserve block-table capacity for the maximum positions it can touch or commit.

Source: [vLLM `MultiStepWorker` API](https://docs.vllm.ai/en/v0.9.0.1/api/vllm/spec_decode/multi_step_worker.html).

### Qwen3-Next has hybrid attention and native MTP

Qwen and Hugging Face describe Qwen3-Next as a hybrid model using Gated DeltaNet plus full/gated attention, with native Multi-Token Prediction. The Qwen blog says the native MTP module is intended to provide high speculative-decoding acceptance, and vLLM states it supports Qwen3-Next MTP mode.

Sources: [Qwen3-Next blog](https://qwen.ai/blog?id=qwen3-next), [Hugging Face Qwen3-Next docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_next), [vLLM Qwen3-Next blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html).

### Qwen/vLLM MTP head shape

The vLLM Qwen3-Next MTP source implements an inference-only `Qwen3NextMTP` model. Its MTP predictor embeds the previous token, normalizes both token embedding and previous hidden state, concatenates them, projects back to hidden size, runs one of the MTP decoder layers selected by `spec_step_idx`, normalizes, then applies an LM head. The source also uses `num_nextn_predict_layers` to choose the number of MTP layers.

This supports the local interpretation that a Qwen-style MTP head should be seeded with the target model's hidden state at the already-confirmed prefix and with the most recently confirmed token ID and position.

Source: [vLLM `qwen3_next_mtp.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py).

## Inferred behavior for this repo

The following items are inferred from the local implementation notes and model semantics, not directly specified by a Qwen or vLLM public contract.

### Dense-only all-at-once verification should be correctness-preserving

For dense-only autoregressive models, the mutable decode state is attention KV plus sequence length/block metadata. A verifier that evaluates `[last_token, draft_1, ..., draft_K]` in one target pass can be exact if it observes these rules:

- verifier logits used for acceptance must come from the same target logits path as baseline decode,
- positions and causal masks must match the sequential decode positions,
- KV slots for tokens past the accepted prefix must be restored or made unreachable,
- committed logical length must advance by exactly the accepted prefix length, plus one for the mismatch/bonus token that is actually emitted,
- scheduler capacity must cover every position physically written or emitted.

Because dense-only state is slot-addressed, selecting or restoring by KV slot is enough to return to the canonical prefix after partial rejection. This is an inference from standard KV-cache semantics and vLLM's lookahead requirement, not an independently sourced Qwen claim.

### Hybrid layers make all-at-once verification harder

For Qwen3.5/Qwen3-Next-style hybrid layers, full attention KV is not the only mutable state. Gated DeltaNet has convolutional and recurrent state that advances token by token. If an all-at-once verifier computes state after all proposed tokens but only a prefix is accepted, the final recurrent/conv state is generally the wrong committed state.

A hybrid all-at-once verifier is only safe if it can produce and select the exact hybrid prefix state for every possible accepted prefix. For K=2, that means the verifier must be able to commit state after:

- current target token only, when zero drafts are accepted,
- first draft, when one draft is accepted,
- second draft, when both drafts are accepted.

The current local correctness notes point to this failure mode: all-at-once K=2 diverged after canonical-logits fixes, while sequential commit-select became correct by materializing each one-token state and selecting among them.

Local notes: [MTP current state](../MTP_CURRENT_STATE_2026_05_07.md), [MTP mixed-length fused notes](../MTP_MIXED_LENGTH_FUSED_NOTES_2026_05_07.md), [MTP correctness/speed log](../MTP_CORRECTNESS_SPEED_LOG.md).

### Bonus-token hidden state is a separate correctness boundary

The local notes show repeated failures when bonus tokens or next drafts were derived from a hidden/logit path that was not exactly the baseline target logits path. The inferred local rule is:

- sample emitted target tokens from canonical target logits,
- only use hidden states for MTP seeding after proving that the hidden state corresponds to the same committed prefix as the emitted token,
- do not reseed from accepted bonus hidden state by default until equivalence is proven for the hybrid backend and batch shape.

This is stricter than the public speculative-decoding contract because local greedy exactness can fail from small numerical or state-order differences even when the high-level algorithm is lossless in theory.

### Rowwise acceptance is a state-selection problem, not just a token-comparison problem

Rowwise acceptance is attractive because all-or-none acceptance probability collapses as batch size grows. However, for hybrid models, rowwise acceptance means different rows may commit different prefix lengths from the same verifier call. Correctness then requires rowwise selection of both:

- attention KV slots and sequence lengths,
- Gated DeltaNet conv/recurrent prefix states.

The repo's current notes report that all-or-none fast paths can be exact but slow, while rowwise acceptance has drifted. The likely root cause is incomplete per-row canonicalization of rejected or partially accepted hybrid state.

## Practical reference rules for future changes

1. Treat baseline target decode as canonical for greedy correctness.
2. Use canonical target logits for every emitted target token, including bonus tokens.
3. Reserve scheduler/block capacity for the maximum verifier write window before executing speculation.
4. For dense-only models, all-at-once verification can be a valid speed path if uncommitted KV slots and logical lengths are restored exactly.
5. For hybrid models, assume sequential verifier state updates are required unless the verifier returns selectable hybrid prefix states for each possible accepted length.
6. Keep all-or-none hybrid fast paths separate from rowwise hybrid paths. All-or-none can commit a single final state only when every active row accepts the full prefix.
7. Record throughput only when exact token parity with baseline passes; otherwise report correctness failure first.

## Source index

- Qwen Team: [Qwen3-Next blog](https://qwen.ai/blog?id=qwen3-next).
- Hugging Face Transformers: [Qwen3-Next model docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_next).
- Hugging Face Transformers source: [Qwen3-Next modeling source](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py).
- vLLM: [Speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/).
- vLLM: [SpecDecodeWorker API](https://docs.vllm.ai/en/v0.9.0/api/vllm/spec_decode/spec_decode_worker.html).
- vLLM: [MultiStepWorker API](https://docs.vllm.ai/en/v0.9.0.1/api/vllm/spec_decode/multi_step_worker.html).
- vLLM: [Qwen3-Next support blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html).
- vLLM source: [Qwen3-Next MTP model](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py).
- Paper: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).
- Paper: [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737).

## Additional verification notes from 2026-05-08 review

This section treats the review claims as hypotheses and records what was verified from primary sources versus what remains an inference for this repo.

### Qwen3.5 and Qwen MTP source status

Canonical source behavior:

- Qwen's Qwen3.5 model card for `Qwen/Qwen3.5-2B` lists a hybrid language-model layout built from Gated DeltaNet layers plus gated attention layers, and states that MTP is trained with multi-steps.
- The same Qwen3.5 model card recommends vLLM MTP serving with `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
- The Qwen3.5 launch blog says Qwen3.5 uses the Qwen3-Next architecture family, including high-sparsity MoE, Gated DeltaNet plus Gated Attention hybrid attention, training-stability optimizations, and multi-token prediction.
- The Qwen3.5-Omni technical report confirms a related Qwen3.5-family Hybrid Attention MoE architecture, but it is not a speculative-decoding algorithm reference.
- vLLM exposes `qwen3_next_mtp` as an MTP speculative method, and its Qwen3Next MTP model source is an inference-only MTP module. The source builds the MTP predictor from the previous hidden state plus the current input embedding, then runs a Qwen3Next decoder layer selected by `spec_step_idx % num_mtp_layers`.

Inferred repo consequence:

- For Qwen3.5/Qwen3-Next-like models, MTP proposals are not a separate dense draft model. They are native target-family next-token predictor layers attached to a hybrid architecture. Correctness should therefore be judged against target-model emitted tokens, not against standalone drafter quality.
- I did not find a separate Qwen3.5 MTP paper that defines acceptance semantics beyond the Qwen model cards/blog and vLLM implementation. The acceptance semantics below come from vLLM and the speculative decoding paper, not from a Qwen-specific paper.

### K=1 MTP speculative step

Review hypothesis:

```text
main model produces target token
MTP head proposes one draft/bonus token
verifier checks whether target == draft
accepted row emits [target, bonus]
rejected row emits [target]
```

Verified for vLLM V1 GPU greedy/top-1 speculative decoding:

- The vLLM GPU `rejection_sample` kernel is row-indexed by request. For each row, it stores the target-sampled token at the current speculative position, compares that target sample with the corresponding draft token, and marks the row rejected on mismatch.
- If no draft token was rejected, the same kernel appends one final target-sampled token. For K=1, this means a matching row emits two target-family tokens, while a mismatching row emits only the first target-family token.
- vLLM's higher-level rejection sampler defines the final output as accepted tokens plus recovered tokens plus bonus tokens, and states that the bonus token is added only when all proposed tokens are accepted and is sampled from target probabilities.

Canonical boundary:

- The exact `[target, bonus]` / `[target]` statement is canonical for greedy/top-1 equality checking as implemented in vLLM's GPU kernel.
- For non-greedy rejection sampling, a rejected draft can be replaced by a recovered token sampled from the adjusted target-minus-draft distribution. In that case, the simplified equality form is a greedy special case, not the full probabilistic algorithm.

### Expected tokens for K drafts

Review hypothesis:

```text
E[tokens] = 1 + p + p^2 + ... + p^K
          = (1 - p^(K+1)) / (1 - p)
```

Status: inferred speed math, not a source-level guarantee.

- The formula follows from rowwise speculative decoding if each draft position has the same independent conditional acceptance probability `p` and a bonus target token is emitted only after all K drafts are accepted.
- vLLM source supports the structural part of the formula: a row can emit up to `K + 1` tokens, and the extra final token is emitted only on full acceptance.
- Real acceptance is position-dependent and workload-dependent, so use the formula only as a planning estimate.

Useful planning examples:

```text
K=1, p=0.8: E = 1.80 tokens/row/step
K=2, p=0.8: E = 2.44 tokens/row/step
K=3, p=0.8: E = 2.952 tokens/row/step
```

### All-or-none versus rowwise acceptance

Review hypothesis:

```text
All-or-none B=4/B=6 is too brittle; rowwise or small group acceptance is expected to matter.
```

Verified and inferred:

- vLLM's GPU kernel is rowwise: each request row has its own rejection flag and its own emitted-token count. This is canonical vLLM behavior for GPU speculative decoding.
- Batch-level all-or-none is therefore not the canonical vLLM acceptance model. It is an implementation simplification.
- If an all-or-none group of size `g` requires every row to accept all K drafts, the full-group success probability is approximately `p^(g*K)`. If fallback emits one token per row, expected tokens per row are approximately `1 + K * p^(g*K)`.
- With `p=0.8`, K=1, and group size 4, all-or-none gives about `1.41` tokens/row/step versus rowwise `1.80`. At group size 6, it gives about `1.26`.
- With `p=0.8`, K=2, and group size 4, all-or-none gives about `1.34` tokens/row/step versus rowwise `2.44`. At group size 6, it gives about `1.14`.

Repo consequence:

- If full rowwise state selection is hard, small groups are a reasonable intermediate design. They preserve more expected tokens than whole-batch all-or-none while keeping state-management complexity below fully rowwise acceptance.
- Group size 1 is rowwise. Larger groups should be treated as a tunable performance/correctness engineering compromise, not as canonical speculative decoding semantics.

### Dense-only all-at-once versus hybrid state updates

Canonical source behavior:

- vLLM speculative decoding aims to be lossless up to numerical limits, with greedy equality validated against non-speculative decoding.
- vLLM's native MTP docs describe MTP as speculative decoding where the target model itself has native MTP capability and no separate draft model is required.
- Qwen3.5/Qwen3-Next sources identify the relevant models as hybrid, with Gated DeltaNet/linear-attention state in addition to gated attention/KV state.

Inferred dense-only rule:

- Dense-only all-at-once verification can be correct if the verifier computes the same target logits as sequential target decoding and can restore or commit KV cache, sequence lengths, masks, and positions exactly to each row's accepted prefix.
- For dense-only transformers, the needed mutable decode state is primarily position-indexed KV plus scheduler metadata. That makes all-at-once verification plausible because rejected suffix KV can be ignored or truncated per row.

Inferred hybrid rule:

- Hybrid Qwen3.5/Qwen3-Next-like models add recurrent linear-attention/Gated DeltaNet state. After verifying K drafts all at once, the final recurrent state corresponds to the full draft path, not necessarily to the shorter accepted prefix for rejected rows.
- Therefore hybrid rowwise acceptance needs either sequential verifier state updates or a verifier that materializes selectable per-prefix hybrid states for each row.
- Without selectable per-prefix hybrid states, all-at-once hybrid verification can produce correct logits for comparison but still commit the wrong future state after partial acceptance. This matches the observed failure mode where token parity can diverge later rather than immediately.

### TPU-specific vLLM findings

Canonical source behavior:

- vLLM's public TPU optimization docs focus on TPU workload sizing, XLA compilation/caching, static shape behavior, padding buckets, `max_num_seqs`, `max_num_batch_tokens`, quantization, and TPU parallelization.
- I did not find TPU-specific vLLM speculative-decoding semantics in the public docs reviewed here.
- The concrete vLLM speculative GPU references are under `vllm/v1/worker/gpu/spec_decode/` and use Triton kernels. Those GPU kernels should not be assumed to run unchanged on TPU.

Inferred repo consequence:

- For a JAX/TPU implementation, preserve vLLM's semantic model but design around static shapes: fixed K, fixed batch/group windows, mask-based row acceptance, and XLA-friendly state selection.
- TPU speedups are likely to come less from dynamic control flow and more from avoiding extra host scheduling, avoiding shape churn, and keeping verifier/drafter work in a small number of compiled graphs.

### Source links added for this section

- Qwen3.5-2B model card: https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen3.5 launch blog: https://qwen.ai/blog?id=qwen3.5
- Qwen3.5-Omni technical report: https://arxiv.org/abs/2604.15804
- vLLM speculative decoding docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM MTP docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- vLLM rejection sampler docs/source: https://docs.vllm.ai/en/v0.17.1/api/vllm/v1/sample/rejection_sampler/
- vLLM GPU greedy rejection sample source: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/spec_decode/rejection_sample.py
- vLLM Qwen3Next MTP source: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py
- vLLM TPU optimization docs: https://docs.vllm.ai/en/latest/configuration/tpu/
- Speculative decoding paper: https://arxiv.org/abs/2211.17192
