# Roadmap

This roadmap is grouped by work type. It is not a production-readiness claim.

## Correctness

- Keep the GPU BF16-weight/FP32-activation contract aligned with HF: runtime `dtype=float32`, `weight_dtype=bfloat16`.
- Keep the long-decode top-5 guardrail at 500/500 matches against HF before making GPU correctness claims.
- Keep JAX paged server-shape generation and ordered top-5 parity against HF.
- Preserve current exact commit-select MTP1 server-shape generated-token parity while optimizing or gating the slow speculative path.
- Preserve K=1 accept/reject commit invariants across block boundaries and inactive rows in MTP research.
- Keep target logits sourced from the canonical executor forward-step contract.
- Keep older TPU MTP semantics tests useful as historical regression coverage, but do not treat them as current GPU validation.

## Serving

- Keep current GPU serving guidance on the JAX paged baseline unless MTP1 beats that baseline on the full server-shape correctness suite.
- Keep MTP admission scheduler-owned.
- Gate on acceptance and measured decode-latency EWMA.
- Keep per-bucket/physical-bucket admission reporting visible in benchmarks.
- Document all serving flags with exact default behavior.

## Benchmarks

- Preserve the current `benchmark_server_shapes.py` GPU recipe for HF vs JAX paged vs MTP1 with `dtype=float32` and `weight_dtype=bfloat16`.
- Keep correctness checks mandatory for any speed claim.
- Report both total tokens/sec and decode tokens/sec, separated from prefill and compile time.
- Keep the long-decode top-5 GPU guardrail artifact reproducible.
- Treat the TPU v6e-1 Qwen/Qwen3.5-4B BF16 findings as historical for current GPU work.
- Add repeatable GPU benchmark recipes for low-acceptance and high-acceptance prompts now that exact commit-select MTP1 server-shape parity is restored.

## Optimization

- Reduce host synchronization in token materialization and MTP accounting.
- K=1 burst MTP accounting now uses compact device-side emitted-token output
  and per-row summary metadata; continue reducing verifier work rather than
  revisiting per-group host count parsing.
- Improve bucket-specific admission decisions.
- Investigate K=1 overhead in rejected and fallback steps.
- Implement a coarse safe K=1 verifier whose width-2 target-model forward is
  materially cheaper than two separate width-1 target forwards. The
  after-current-token prefix state is fixed in focused coverage and the
  corrected one-pass smoke is exact with `12/13` accepted drafts, but the
  verifier graph is still slower than `k_decode` and no-MTP.
- Keep `mtp_verifier_impl=k_decode` as the conservative verified K=1 route.
  The corrected one-pass route is now a diagnostic speed target, not a
  promoted serving path, until it beats both `k_decode` and no-MTP under the
  same generic warmup and correctness envelope.
- Do not retry per-token/per-layer GDN kernel swaps as the main MTP speed
  lever: on the same smoke, reference/packed-projection and raw-tail Triton GDN
  one-pass were neutral (`22.52-22.60 output tok/s`), while conv-tail Triton
  GDN was much slower (`2.20 output tok/s`).
- Keep packed-prefill K verifier work diagnostic-only until its verifier logits
  and accept/reject decisions match the decode verifier. The 2026-06-15 B=2
  non-boundary FlashInfer smoke validated the packed shape and fixed packed
  `kv_lens` visibility, but packed-prefill still diverged from no-MTP at one
  generated token while decode verification remained exact.
- Strict GDN fallback mode must error on prefill prefix-state verifier routes
  until there is a kernel-backed prefix-state boundary. Do not let
  `return_prefix_hybrid` or `return_first_prefix_hybrid` silently select the
  slow JAX recurrent/chunked path in GDN prefill.
- Use the vLLM/MaxText references for the next boundary design: vLLM flattens
  speculative metadata and pads uniform speculative decode graph keys, while
  MaxText donates persistent decode state into a compiled `generate` step.
- Make the next MTP verifier route a hybrid of those references: scheduler
  builds vLLM-style flattened verifier metadata, the compiled JAX step consumes
  resident decode/GDN state with MaxText-style donation, and the step returns
  compact per-row commit metadata rather than logits or prefix-state tensors.
- Treat verifier speed as the MTP release gate. MTP should stay disabled or
  scheduler-gated unless verified rows amortize target verification, draft
  generation, and commit bookkeeping enough to beat the accepted no-MTP
  random-large path on the same warmup and correctness envelope.
- Treat K=2 as optimization research only until it beats K=1/baseline in valid benchmarks.
- Avoid accelerator-kernel claims unless a dedicated backend exists; current GPU path is JAX/XLA on CUDA.

## Cleanup

- Keep canonical docs in `docs/architecture.md`, `docs/gpu_correctness_guardrails.md`, `docs/kv_cache.md`, `docs/mtp.md`, `docs/scheduler.md`, `docs/benchmarks.md`, and this file.
- Keep `docs/mtp_tpu_spot_findings_2026-05-09.md` accessible as a historical TPU findings record.
- Archive obsolete debug/status markdown under `docs/archive/2026-05-pre-latency-gate/`.
- Avoid duplicating full benchmark logs in canonical docs.
