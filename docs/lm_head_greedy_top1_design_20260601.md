# LM-Head Greedy Top-1 Design Note

Date: 2026-06-01

## Current Path

The decode-heavy profile anchor is
`results/gpu_matrix_decode_padded_gemm_r3_20260601.json`: `decode_heavy_128x128`
is `162.10 tok/s`, `0.759x` vLLM, and the top GPU event is the LM-head GEMM
`gemm_fusion_dot_199` at about `126.45 ms / 127` decode calls.

The greedy token fast path already avoids returning full logits to Python:

- `nanovllm_jax/engine/model_executor.py::forward_step_token_ids_jit`
  calls `model_forward_step(..., return_hidden=True)`.
- It gathers the final hidden row and calls
  `nanovllm_jax/model.py::lm_head_token_ids_and_topk(..., top_k=0)`.
- `lm_head_token_ids_and_topk` applies the final RMSNorm, runs the dense
  LM-head dot, and returns `jnp.argmax(logits, axis=-1)`.

So the remaining issue is not host-side logits transfer. It is the dense
`[B, 1, hidden] x [hidden, vocab]` LM-head work and the in-graph logits
materialization needed before `argmax`.

## Feasibility Findings

High-level JAX does not provide an efficient exact streaming top-1 replacement
for this LM head:

- `jax.lax.top_k(logits, 1)` still requires full logits first. Entry 024 in
  `docs/optimization_logbook.md` tested this spelling and rejected it because
  it preserved generated tokens but did not reduce the LM-head GEMM bucket.
- A pure-JAX tiled scan can preserve `jnp.argmax` tie semantics, but its HLO
  lowers to a `while` with a `dot_general` in the loop body and a
  `dynamic-slice` of the weight tile. On GPU this is structurally a sequence of
  tile GEMMs, not one fused streaming GEMM-plus-argmax. For Qwen3.5-0.8B
  `vocab_size=248320`; even an `8192` vocab tile means 31 GEMM-shaped steps per
  decode token, while smaller tiles quickly become launch-bound.
- Entry 037 already tested a Pallas Triton LM-head argmax. It removed the dense
  GEMM event, preserved exact generated tokens, but the replacement
  `lm_head_tile_argmax` cost was roughly the same as the removed GEMM in the
  integrated server trace and throughput regressed.
- The tied-embedding layout improvement is already present as the opt-in
  `NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD=1`, surfaced through
  `runtime.fastpaths.materialize_tied_lm_head`. The current decode configs use
  it.
- Full logits are still required for non-greedy diagnostics, prefill logit
  comparisons, and `top_k > 0` verifier margin paths. The safe specialization
  boundary is greedy decode only with `top_k == 0`.

## Exact Argmax Contract

Any replacement must match `jnp.argmax(logits, axis=-1)` for finite logits:

- Reduce each vocab tile with `jnp.argmax` or equivalent first-index selection.
- Carry `(best_value, best_index)` across tiles.
- Update on `candidate_value > best_value`, not `>=`, so ties keep the lowest
  vocab index from the earlier tile.
- Return `int32` token ids with shape `[batch, width]` to match
  `lm_head_token_ids_and_topk`.
- If NaN parity is required beyond the current finite-logit guardrails, treat
  the first NaN as winning, matching dense argmax behavior on current JAX.

Do not use `lax.top_k(..., 1)` as the token source for this contract. It is not
the same tie contract as `argmax` and it does not avoid logits materialization.

## Patchable Target

The next viable patch should be a single custom-call or library-backed kernel
that combines LM-head matvec/GEMM and first-index top-1 reduction. A pure JAX
scan should not be added as a runtime option because it is predictably worse
than the current dense GEMM.

Exact file and function targets:

1. Add a helper in `nanovllm_jax/model.py`, next to
   `lm_head_token_ids_and_topk`:
   `lm_head_greedy_top1_token_ids(hidden_norm, output_weight, *, is_prefill)`.
2. Gate it from `lm_head_token_ids_and_topk` only when all of these are true:
   `not is_prefill`, `top_k == 0`, `hidden_norm.ndim == 3`,
   `hidden_norm.shape[1] == 1`, and `output_weight.ndim == 2`.
3. Keep dense fallback as the default and for all top-k/full-logit diagnostic
   paths.
4. Add config-file hooks in
   `nanovllm_jax/server_config.py::_runtime_section_to_env`:
   `runtime.fastpaths.lm_head_greedy_top1_impl` to
   `NANO_VLLM_JAX_LM_HEAD_GREEDY_TOP1_IMPL`, plus backend-specific block-size
   knobs only after a real kernel exists.
5. Add focused tests in `tests/test_lm_head_helpers.py`:
   exact token parity against dense logits, first-index tie behavior across
   tile boundaries, tied and untied output weights, and fallback when
   `top_k > 0` or `is_prefill=True`.

Kernel constraints:

- Inputs: hidden `[B, 1, H]`, weight `[H, V]`, both in the same dtype path used
  by the dense helper after final RMSNorm.
- Output: token ids `[B, 1]`; optionally top value `[B, 1]` for diagnostics.
- The implementation must not introduce runtime CUDA source compilation. Use a
  JAX/Pallas/Triton/CuTeDSL path only if it lowers to one coarse custom call for
  the whole top-1 operation, or use a library-backed path with an argmax
  epilogue if one is available.
- A two-stage tile-kernel plus second reduction should not be promoted unless
  it beats the current dense GEMM in the integrated decode profile, because
  Entry 037 already showed the straightforward version regresses.

## Recommendation

Do not patch `lm_head_token_ids_and_topk` with a pure-JAX tiled top-1 path.
It can be exact, but it is not an efficient route to remove
`gemm_fusion_dot_199`. The next patchable step is a real single-call
GEMM/matvec-plus-argmax backend behind
`runtime.fastpaths.lm_head_greedy_top1_impl`, with the helper and tests listed
above.
