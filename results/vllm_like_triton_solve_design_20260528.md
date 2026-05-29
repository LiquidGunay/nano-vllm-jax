# vLLM-like Triton Solve Design (2026-05-28)

## Outcome
I did **not** implement `gdn_fla_solve_tril_packed_vllm_like_triton` in this bounded pass.

Reason: local `jax_triton` solve path is currently a single-kernel forward substitution over `[T,H,BT]`, while vLLM solve uses a **multi-kernel block-inverse pipeline** with explicit intermediate store/downcast behavior. A faithful port requires adding multiple Triton kernels and a staged launcher pipeline, not a small edit.

## Source comparison

### vLLM vendored solve (`vllm/.../ops/solve_tril.py`)
- Entry: `solve_tril(A, cu_seqlens, chunk_indices, output_dtype)`
- Kernel family:
  - `solve_tril_16x16_kernel`
  - `merge_16x16_to_32x32_inverse_kernel`
  - `merge_16x16_to_64x64_inverse_kernel`
- BT selection:
  - `BT=16`: 16x16 kernel
  - `BT=32`: merge 16->32
  - `BT=64`: merge 16->64
- Numeric behavior:
  - Internal math in fp32
  - `tl.store(... to(dtype.element_ty, fp_downcast_rounding="rtne"))` at stage boundaries
  - Dot precision knob via `FLA_TRIL_PRECISION` (`ieee/tf32/tf32x3`)
  - Output dtype defaults to `k.dtype` in `chunk.py` chain (typically bf16)

### local solve (`nanovllm_jax/kernels/gdn_fla_triton.py`)
- Entry: `gdn_fla_solve_tril_packed_triton(...)`
- Kernel: `_gdn_fla_solve_tril_packed_kernel`
- Behavior:
  - One kernel does row-wise inversion logic for each chunk
  - Output always `jnp.float32`
  - No staged 16->32/64 merge
  - No explicit RTNE downcast-store points
  - No equivalent dot-precision controls

## Why this is a blocker for a bounded patch
To match vLLM numerics closely, we must mirror:
1. Same block decomposition (16 base + merge kernels for 32/64),
2. Same intermediate rounding points (RTNE on stores),
3. Same output dtype path (bf16 vs fp32) used by downstream recompute kernels.

Current local path diverges on all three. Replacing one kernel with another row-solver is insufficient.

## Required implementation plan

### New probe-only API
- Add in `nanovllm_jax/kernels/gdn_fla_triton.py`:
  - `gdn_fla_solve_tril_packed_vllm_like_triton(attention_matrix, cu_seqlens, *, chunk_size, chunk_indices=None, output_dtype="bfloat16", dot_precision="ieee")`
- Do **not** alter default `gdn_fla_chunk_gated_delta_rule_packed_triton` path.

### Required kernels
1. `_gdn_fla_solve_tril_16x16_kernel_vllm_like`
2. `_gdn_fla_merge_16_to_32_inverse_kernel_vllm_like`
3. `_gdn_fla_merge_16_to_64_inverse_kernel_vllm_like`

All three must:
- operate on packed varlen layout `[tokens, heads, BT]` via `cu_seqlens/chunk_indices`,
- keep fp32 math internal,
- explicitly cast+store to target output dtype (bf16/fp32) at merge boundaries.

### Launcher behavior
- Allocate staged buffers like vLLM (`Ai` scratch/output).
- Dispatch by `BT` exactly as vendored solve:
  - 16 -> base kernel
  - 32 -> merge32
  - 64 -> merge64
- Preserve chunk metadata mapping from `prepare_gdn_fla_chunk_metadata`.

### Downstream compatibility
- For parity probing, ensure recompute receives inverse dtype matching vLLM chain (`bf16`).
- Keep existing fp32 reference solve untouched.

## Acceptance gates
Evaluate on real activation `A` at lengths 128/256/512:
1. Inverse parity vs vLLM solve:
   - improve over current baseline max diff (~0.112 from prior audits).
2. Downstream parity (JAX recompute+delta+o using injected inverse):
   - reduce state/output max diff vs vLLM relative to current reference path.
3. Stability:
   - no regression to default server/reference behavior,
   - probe-only invocation.

## Next exact coding task
Implement the three probe-only Triton kernels + staged dispatcher in `gdn_fla_triton.py`, then add a dedicated probe script:
- `benchmarks/probe_vllm_like_triton_solve_reference.py`
- output:
  - `results/vllm_like_triton_solve_reference_probe_20260528.json`
  - `results/vllm_like_triton_solve_reference_probe_20260528.md`
