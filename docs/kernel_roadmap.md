# Kernel Roadmap

This roadmap is the kernel-focused slice of
`docs/gpu_optimization_next_goal_plan.md`. Every external or lowered kernel must
remain optional until it passes the correctness and integrated-performance
gates. The pure-JAX path is the fallback and correctness reference.

Local CUDA/JAX FFI probes recorded below are historical diagnostics only. Do not
use, extend, or route them as the next optimization path unless the user
explicitly reopens that direction. The planned production direction remains
FlashInfer for paged KV/attention where dtype/layout gates allow, and
vLLM/Flash Linear Attention-shaped GDN kernels with pure-JAX fallbacks.

## Priority Order

| Priority | Kernel | Purpose |
| --- | --- | --- |
| P0.1 | `kv_append_paged_nhd` | Establish full-attention KV physical layout. |
| P0.2 | `paged_decode_attention_gqa_nhd` | Replace the core full-attention decode read path. |
| P1.1 | `gdn_recurrent_decode_step` | Own the GDN decode recurrence and state update. |
| P1.2 | `gdn_segmented_prefill_chunk32` | Replace rectangular padded chunked GDN prefill with a coarse segmented kernel. |
| P2.1 | `paged_prefill_attention_gqa_nhd` | Improve full-attention prefill after P0 layout is stable. |
| P2.2 | `qk_norm_rope_kv_append_fused` | Fuse full-attention decode projection post-processing and cache append. |
| P3 | `topk_logits` / sampling / logprob | Feature and diagnostics kernels after decode is closer to vLLM. |
| P3 | `silu_and_mul` / RMSNorm smoke kernels | Optional FFI validation helpers only if needed. |

## P0.1 - `kv_append_paged_nhd`

- motivation: define the full-attention paged KV layout so later attention
  kernels do not pay layout conversion overhead.
- proposed source/reference implementation: FlashInfer `append_paged_kv_cache`
  and vLLM `reshape_and_cache_flash_kernel`.
- JAX-facing ABI:

```python
kv_append_paged_nhd(
    append_key,        # [nnz_tokens, num_kv_heads, head_dim]
    append_value,      # [nnz_tokens, num_kv_heads, head_dim]
    batch_indices,     # [nnz_tokens]
    positions,         # [nnz_tokens]
    k_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    kv_indices,        # [total_pages]
    kv_indptr,         # [batch + 1]
    kv_last_page_len,  # [batch]
) -> updated_cache
```

- fallback path: existing pure-JAX `update_kv_cache`.
- correctness gate: exact generated-token match vs the current accepted baseline
  and focused cache-layout parity tests.
- performance gate: no TTFT or ITL p50/p95 regression; lower or flat cache
  write/gather/transpose overhead; no new host sync around FFI setup.
- do-not-merge conditions: per-layer layout conversions, FFI planning in the
  decode loop, or an integrated regression hidden by a microbenchmark win.
- current status: FlashInfer's append route is ABI-proven for BF16/FP16 cache
  tensors, but rejected for the accepted FP32 activation/KV-cache serving
  contract. A future serving path needs either a FP32-capable append kernel or
  an explicit dtype-policy change.
- current FP32 status: a local CUDA/JAX FFI append smoke kernel now preserves
  the FP32 cache contract and matches the NHD reference in a focused CUDA test.
  Routed standalone append preserved exact hetero8 generated tokens, but
  regressed integrated throughput to `193.62 tok/s`; keep it as a toolchain
  smoke proof, not a serving candidate.
- paired-route status: combining the local FP32 append route with the local FP32
  decode-attention route on the long-prefill target preserved exact tokens but
  regressed to `52.94 tok/s`. The trace confirmed both CUDA kernels ran. Do not
  continue this narrow pair as the P0 serving strategy; the next P0 attempt must
  own a broader fused/layout boundary or move to the GDN roadmap.

## P0.2 - `paged_decode_attention_gqa_nhd`

- motivation: match the vLLM-style paged decode attention memory access pattern
  for full-attention layers.
- proposed source/reference implementation: FlashInfer
  `BatchDecodeWithPagedKVCacheWrapper` and vLLM PagedAttention.
- JAX-facing ABI:

```python
paged_decode_attention_gqa_nhd(
    q,                 # [batch, num_q_heads, head_dim]
    k_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    kv_indptr,         # [batch + 1]
    kv_indices,        # [total_pages]
    kv_last_page_len,  # [batch]
    seq_lens,          # [batch]
    softmax_scale,     # float
) -> out               # [batch, num_q_heads, head_dim]
```

- fallback path: existing pure-JAX `paged_attention_decode`.
- correctness gate: exact generated-token match and focused top-5 decode parity.
- performance gate: ITL p50 improves, ITL p95 does not regress, PjRt Execute and
  command-buffer update/execute stay flat or improve.
- do-not-merge conditions: FFI setup inside the decode loop, per-layer layout
  conversion, or microbenchmark-only improvement.
- current status: FlashInfer decode/prefill attention JIT dtype maps are
  BF16/FP16/low-precision oriented and do not satisfy the accepted FP32
  activation/KV-cache contract. Do not implement this FlashInfer wrapper for
  serving until there is a FP32-capable path or an explicit dtype-policy change.
- current ABI status: a pure-JAX FP32 NHD reference and focused parity tests now
  define the decode-attention contract a future CUDA/custom-call path must
  satisfy before it can be routed into serving.
- current CUDA status: a local FP32 CUDA/JAX FFI implementation satisfies the
  focused ABI parity gate, including the Qwen3.5-0.8B full-attention
  `8q/2kv/head_dim=256` shape. The backend route is available behind
  `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1`, but the first integrated hetero8 run
  preserved exact tokens while regressing throughput/ITL. Keep this route
  default-off as a diagnostic until it is paired with a broader layout/attention
  strategy that passes the integrated gate.
- paired-route status: pairing this local FP32 decode route with the local FP32
  append route still regressed the integrated long-prefill gate to
  `52.94 tok/s` despite exact generated-token parity. Do not promote the
  current narrow pair.

## P1.1 - `gdn_recurrent_decode_step`

- motivation: Qwen3.5-0.8B has 18 GDN layers and only 6 full-attention layers;
  GDN decode remains a major model-specific target.
- proposed source/reference implementation: Qwen 3 Next vLLM GDN path and Flash
  Linear Attention Gated DeltaNet kernels.
- implementation note: direct reuse is not a drop-in FP32 path. The installed
  vLLM/FLA prefill path rejects FP32 activations, and FlashInfer GDN kernels are
  half/BF16 oriented. Use vLLM's packed decode kernel shape as the first
  port/fork target, then revisit segmented prefill after packed decode passes
  strict parity.
- route decision: the first selected slice is a vLLM/FLA-shaped packed decode
  boundary with two implementations under one backend switch: pure-JAX
  `reference` and local CUDA/JAX FFI `cuda_fp32`. Keep it default-off until an
  integrated exact-token benchmark proves a speed win.
- JAX-facing ABI:

```python
gdn_recurrent_decode_step(
    q,      # [batch, gdn_heads, head_dim], fp32 activation
    k,      # [batch, gdn_heads, head_dim]
    v,      # [batch, gdn_heads, head_dim]
    beta,   # [batch, gdn_heads] or [batch, gdn_heads, 1]
    gate,   # [batch, gdn_heads] or [batch, gdn_heads, 1]
    state,  # [batch, gdn_heads, value_dim, key_dim], fp32 V,K layout
) -> (out, new_state)
```

- fallback path: `jax_recurrent_gated_delta_rule`.
- correctness gate: strict recurrent unit parity, exact 500-step generated-token
  parity, no activation/state dtype downgrade.
- performance gate: hetero8 ITL p50 improves, ITL p95 does not regress,
  `forward_step_token_ids_jit` decreases.
- do-not-merge conditions: state drift above gate, hidden dtype downgrade, or
  integrated server regression despite a microbenchmark win.
- local probe status: a local FP32 CUDA/JAX FFI width-1 recurrent decode
  prototype passes focused parity against `jax_recurrent_gated_delta_rule`,
  including the model's `16` GDN heads and `128`-wide state shape. The backend
  route is available behind `NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE=1`, but the
  first integrated hetero8 run preserved exact tokens while regressing
  throughput/ITL, and the V,K-native one-repeat long-prefill probe reached only
  `88.07 tok/s`, `0.757x` vLLM, below the accepted V,K baseline of
  `90.65 tok/s`. Keep this route default-off as a diagnostic; do not treat a
  standalone width-1 recurrence custom call as an accepted serving kernel or as
  the next production implementation path.
- layout decision: vLLM/FlashInfer Qwen GDN uses k-last/V-first recurrent state,
  and the repo will move canonical serving GDN state to `[B,L,HV,V,K]` instead
  of preserving the old JAX `[B,L,HV,K,V]` ABI. The pure-JAX fallback should be
  adapted to V,K state so external kernels do not require hot-path state
  transposes. BF16-oriented external prefill activations remain a separate dtype
  decision and are not implied by the layout change. Test BF16 GDN prefill
  activations later as an opt-in experiment after V,K FP32 correctness is
  established.
- packed-core status: `gdn_packed_decode_step_fp32` implements that boundary as
  a local CUDA/JAX FFI target and passes focused CUDA parity for same-head and
  GVA q/k repetition shapes using native V,K state. The model can now route
  width-1 cached decode through
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=reference|cuda_fp32`. This is still
  experimental and default-off. The first integrated long-prefill target run
  with `cuda_fp32` was exact and speed-claim-ready but regressed to
  `88.41 tok/s`, `0.760x` vLLM, versus the current accepted/scoped default at
  `90.81 tok/s`, `0.780x` vLLM. Keep it as a tool, not a promoted path.
- backend-selection status: the registry recognizes `gdn_fla` plus
  `fla_gdn`, `vllm_fla`, and `flash_linear_attention` aliases. They are
  intentionally unimplemented and fall back to pure JAX until a vLLM/FLA-shaped
  GDN path passes the correctness and benchmark gates.
- ABI-reference status: `nanovllm_jax/kernels/gdn_fla.py` owns the FP32
  packed-decode reference boundary for `mixed_qkv + a/b/A_log/dt_bias` with
  native `[B,HV,V,K]` state. It also exposes a local-decay variant that accepts
  loaded `A=exp(A_log)` weights, and owns the planned segmented prefill
  `[nnz,H,D] + cu_seqlens` pack/reference/unpack helpers. This keeps the planned
  FLA/vLLM contract separate from implementation-specific fast paths.
- V,K migration status: the pure-JAX fallback now consumes and returns V,K GDN
  state directly, and the local recurrent/packed CUDA decode probes now accept
  V,K without Python-side K,V transposes. Focused CUDA tests, CUDA FFI tests, MTP
  commit-state tests, the 500-token top-5 guardrail, and the long-prefill
  integrated goal target passed. The integrated result is `90.65 tok/s`,
  `0.779x` vLLM, so this is an accepted correctness/layout migration, not the
  final `0.9x` speed target.
- vLLM/FLA timing probe status: `results/vllm_fla_gdn_probe_20260527_sm86.json`
  confirms the vendored vLLM BF16 FLA kernels run on A10G/SM86. For the
  model-shaped varlen prefill `[512,1024,1536,2048]`, `fused_post_conv_prep`
  plus `chunk_gated_delta_rule` is `1.45 ms` p50 with FP32 gate/beta/state and
  V,K state `[N,HV,V,K]`. Packed BF16 decode is `0.11-0.16 ms` p50 for batches
  `1,4,8,16`. This is a porting target, not a serving promotion: it still
  requires a JAX-facing path plus exact-token and long-logit gates. The same
  artifact includes independent recurrent reference checks: packed decode
  matches BF16 output exactly and FP32 state within `1.19e-7`; ragged prefill
  `[17,64,65]` has BF16 output max abs `4.88e-4` and final-state max abs
  `4.45e-3`.
- direct-reuse audit status: direct vLLM Torch/Triton reuse from JAX is not a
  low-risk path in the current environment. There is no `jax_triton` package in
  the repo venv, the vLLM venv lacks JAX/`jax_tvm_ffi`, vLLM's GDN kernels are
  exposed through Torch/Triton wrappers rather than a stable non-Torch ABI, and
  the available JAX DLPack/host-callback surface is not enough for a JIT-safe
  framework bridge. Port/fork the FLA schedule behind the JAX-facing boundary.
- varlen contract status: the JAX-side prepared-layout FLA prefill boundary now
  accepts q/k/v `[B,T,H,D]`, gate/beta `[B,T,H]`, and `seq_lens [B]`, packs the
  valid tokens into upstream-style `[nnz,H,D] + cu_seqlens`, runs the segmented
  FP32 reference, and unpacks output to `[B,T,H,V]`. This is an ABI scaffold for
  a future FLA-schedule port, not a serving speed claim.
- port-map status: the vLLM prefill implementation decomposes into
  `fused_post_conv_prep`, then `chunk_local_cumsum`,
  `chunk_scaled_dot_kkt_fwd`, `solve_tril`, `recompute_w_u_fwd`,
  `chunk_gated_delta_rule_fwd_h`, and `chunk_fwd_o`. These math kernels plus
  `prepare_chunk_indices`/`prepare_chunk_offsets` are the port/fork surface.
  Do not port Torch autograd wrappers, `input_guard`, vLLM forward context, or
  global autotune/runtime plumbing into the first JAX path.

## P1.2 - `gdn_segmented_prefill_chunk32`

- motivation: Entry 045 established chunk size 32 as the best verified point;
  source-JAX row/chunk segmentation regressed badly, so the next attempt must be
  backend-owned.
- proposed source/reference implementation: current padded
  `jax_chunk_gated_delta_rule` as the reference; Qwen 3 Next vLLM / Flash Linear
  Attention as implementation references.
- JAX-facing ABI:

```python
gdn_segmented_prefill_chunk32(
    q,              # [nnz_tokens, gdn_heads, head_dim]
    k,              # [nnz_tokens, gdn_heads, head_dim]
    v,              # [nnz_tokens, gdn_heads, head_dim]
    beta,           # [nnz_tokens, gdn_heads]
    gate,           # [nnz_tokens, gdn_heads]
    cu_seqlens,     # [batch + 1]
    initial_state,  # [batch, gdn_heads, value_dim, key_dim]
    chunk_size=32,
) -> (y, final_state)
```

- fallback path: current padded chunk32 JAX implementation.
- correctness gate: exact generated-token match before serving promotion;
  standalone output and final-state max abs `<=1e-5` vs current padded chunk32.
- performance gate: first `forward_step_token_ids_jit` and TTFT p50 improve;
  PjRt Execute and command-buffer execute/update do not regress.
- do-not-merge conditions: first prefill gets slower, dynamic-slice/update or
  tiny-command-buffer count explodes, or the win exists only in a microbenchmark.
- current status: first local CUDA/JAX FFI one-piece chunk32 prototype is
  benchmark-only and rejected for serving promotion. It passed focused CUDA
  tests and a reduced-shape smoke benchmark. The benchmark-only FFI/probe
  boundary now uses native V,K state and passes a non-square
  `B=2,H=2,T=64,K=32,V=64` smoke comparison against current padded chunk32
  (`output_max_abs=1.49e-07`, `state_max_abs=1.073e-06`). The post-V,K full
  hetero8 model-shape microbenchmark is still rejected: V32 p50 `10.43 ms` and
  V64 p50 `10.46 ms` versus current JAX chunk32 `5.60 ms`, with
  `state_max_abs=2.441e-04`. The next candidate should move closer to the
  segmented/nnz ABI instead of only widening the rectangular value block. The
  first pure-JAX packed segmented ABI
  correctness gate passes reduced shape but fails the full hetero8 standalone
  `1e-5` gate versus current padded chunk32, so CUDA math for that ABI is
  deferred pending a correctness-contract decision. A row-padded diagnostic
  that keeps each packed row at `T=512` still fails the gate, so the issue is
  row-wise decomposition/accumulation order rather than only variable-length
  chunk count. The benchmark now emits a machine-readable policy summary:
  segmented CUDA math is allowed only after the strict padded chunk32
  output/state gate passes. Failed packed/row-padded gates are reported as
  `blocked_on_correctness_policy`; a true-token packed ABI requires an explicit
  design decision plus a separate real-weight full-model token/logit parity gate
  before CUDA math. The required override gate is exact generated-token match,
  500/500 top-1 match, 500/500 ordered top-5 match, 500/500 top-5 set match,
  and `max_hf_topk_id_logit_diff <= 2e-5` against the stored HF long-decode
  artifact. The 2026-05-26 kernel-phase revalidation kept all 500 top-1/top-5
  matches but missed the numeric bound slightly (`2.09808349609375e-05`), so the
  override gate is not currently passed.
- dtype experiment note: FLA/FlashInfer-oriented BF16 GDN prefill activations
  are allowed only as a later opt-in experiment, not as part of the V,K layout
  migration. Keep recurrent state and decode activation math FP32 for the first
  experiment, and require exact generated-token parity plus long-decode top-5
  guardrails and integrated TTFT/throughput improvement before promotion.
- compatibility note: the installed vLLM/FLA chunk prefill path rejects FP32
  activation tensors, and FlashInfer GDN prefill is Torch-only plus gated to
  newer CUDA targets than the A10G baseline. The local probe
  `results/external_gdn_kernel_probe_20260527_sm86.json` records A10G SM86, so
  direct FlashInfer GDN prefill is not runnable on this host. Direct reuse
  therefore needs a BF16-prefill design decision on supported hardware or a
  FP32-capable vLLM/FLA-derived port for this machine.
- prepared-varlen status: `pack_prepared_gdn_prefill_inputs`,
  `unpack_prepared_gdn_prefill_output`, and
  `gdn_fla_prefill_varlen_reference` define the next port/fork seam for
  vLLM/FLA-style prefill. The focused CUDA test verifies empty rows, ragged
  lengths `[0,5,17,32]`, `cu_seqlens=[0,0,5,22,54]`, packed tensor shapes, and
  parity with the rectangular prepared FP32 reference.
- chunk-metadata status: `prepare_gdn_fla_chunk_metadata` now defines the
  future FLA chunk-body metadata contract: active chunk rows
  `[sequence_index, chunk_index_in_sequence]` plus per-row `chunk_offsets`.
  It preserves original sequence indices when padded rows have zero length, so
  future kernels can read `cu_seqlens` and state rows without silently
  compressing batch-row identity.
- chunk-local-cumsum status: `gdn_fla_chunk_local_cumsum_packed_reference`
  defines the first scalar FLA math-stage reference over packed `[nnz,H]`
  gates. It covers forward and reverse cumsum, resets accumulation at every
  active chunk row, and uses `prepare_gdn_fla_chunk_metadata` so future kernels
  can match vLLM semantics without assuming padded rows were pre-filtered.
- chunk-scaled-dot-KKT status:
  `gdn_fla_chunk_scaled_dot_kkt_packed_reference` defines the next scalar FLA
  math-stage reference over packed `[nnz,Hg,K]` keys and `[nnz,H]` beta/gate
  tensors. It covers strict-lower masking, optional `exp(g_i - g_j)` decay
  scaling, and grouped output-head to key-head mapping.
- solve-tril status: `gdn_fla_solve_tril_packed_reference` defines the scalar
  FLA triangular-solve stage over packed `[nnz,H,BT]` matrices. It computes
  `(I + A)^-1` per active ragged chunk and leaves columns outside each partial
  chunk zero, matching the shape contract expected before `recompute_w_u_fwd`.
- recompute-w/u status: `gdn_fla_recompute_w_u_packed_reference` defines the
  next FLA chunk-body stage over packed varlen tensors. It applies the solved
  chunk matrix to beta-weighted values for `u` and to
  `beta * exp(g_cumsum)` weighted grouped keys for `w`.
- chunk-delta-h status: `gdn_fla_chunk_delta_h_packed_reference` defines the
  FLA recurrent state/value update stage. It uses `chunk_offsets` rather than
  `chunk_indices` for per-sequence chunk traversal, stores prior chunk states
  in `h`, stores ungated `v_new`, and applies gate-rescaled deltas to the FP32
  final state.
- SM86 port note: on the current A10G host, vLLM's Hopper/TMA branches are not
  active. The local target should therefore mirror the non-TMA FLA path first.
  Preserve FP32 gate/beta/state; vLLM rejects FP32 q/k/v in its Torch wrapper,
  so a FP32 JAX port must intentionally own that dtype contract rather than
  calling the upstream wrapper directly.

## P2.1 - `paged_prefill_attention_gqa_nhd`

- motivation: improve the 6 full-attention layers' prefill path after P0 owns
  the cache layout.
- proposed source/reference implementation: FlashInfer
  `BatchPrefillWithPagedKVCacheWrapper`; MaxText paged/ragged attention design
  patterns as non-drop-in references.
- JAX-facing ABI:

```python
paged_prefill_attention_gqa_nhd(
    q,                 # [nnz_q, num_q_heads, head_dim]
    k_cache,
    v_cache,
    qo_indptr,          # [batch + 1]
    kv_indptr,          # [batch + 1]
    kv_indices,         # [total_pages]
    kv_last_page_len,   # [batch]
    causal=True,
) -> out                # [nnz_q, num_q_heads, head_dim]
```

- fallback path: existing pure-JAX `paged_attention_prefill`.
- correctness gate: exact generated-token match and focused attention parity.
- performance gate: TTFT p50 improves with no ITL regression.
- do-not-merge conditions: per-layer layout conversion or integrated prefill
  regression.

## P2.2 - `qk_norm_rope_kv_append_fused`

- motivation: fuse the useful full-attention decode post-projection chain:
  Q/K/V split, Q/K norm, RoPE, KV append, and returning Q for attention.
- proposed source/reference implementation: local pure-JAX full-attention path,
  FlashInfer RoPE/cache helpers where useful.
- JAX-facing ABI: to be finalized after P0 stabilizes the NHD cache contract.
- fallback path: current separate JAX norm/RoPE/cache append steps.
- correctness gate: exact generated-token match and no host-side metadata change.
- performance gate: lower decode-layer overhead after P0 kernels are stable.
- do-not-merge conditions: starts before P0 is stable, adds layout conversion, or
  profile no longer shows norm/RoPE/cache-write overhead.

## P3 - Sampling, Top-k, Logprob, And Smoke Kernels

- motivation: features and diagnostics, not the next greedy throughput lever.
- proposed source/reference implementation: FlashInfer logits/sampling/norm/RoPE
  APIs or minimal local FFI smoke kernels.
- JAX-facing ABI examples:

```python
topk_logits(logits, k: int) -> (values, indices)
sample_topk_topp(logits, temperature, top_k, top_p, rng_state) -> token_ids
```

- fallback path: existing JAX logits, top-k, and sampling code.
- correctness gate: exact greedy parity where applicable; distribution tests for
  non-greedy sampling when product requirements exist.
- performance gate: only considered after decode ITL is closer to vLLM.
- do-not-merge conditions: distracts from P0/P1, or adds complexity without a
  product requirement or MTP diagnostic need.
