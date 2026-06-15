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
- Implement a safe fast K=1 verifier that can commit rejected rows from after-current-token state without full repair decode.
- Keep `mtp_verifier_impl=k_decode` as the verified K=1 optimization route; K=1 burst verification is available for host-sync amortization, but it still needs a cheaper exact prefix-state path before it can beat the no-MTP baseline.
- Do not promote `return_first_prefix_hybrid` as that cheaper path until it matches full prefix-hybrid selection layer-by-layer; the 2026-06-15 smoke reduced acceptance from 11/13 to 6/13.
- Keep packed-prefill K verifier work diagnostic-only until its verifier logits
  and accept/reject decisions match the decode verifier. The 2026-06-15 B=2
  non-boundary FlashInfer smoke validated the packed shape and fixed packed
  `kv_lens` visibility, but packed-prefill still diverged from no-MTP at one
  generated token while decode verification remained exact.
- Use the vLLM/MaxText references for the next boundary design: vLLM flattens
  speculative metadata and pads uniform speculative decode graph keys, while
  MaxText donates persistent decode state into a compiled `generate` step.
- Treat K=2 as optimization research only until it beats K=1/baseline in valid benchmarks.
- Avoid accelerator-kernel claims unless a dedicated backend exists; current GPU path is JAX/XLA on CUDA.

## Cleanup

- Keep canonical docs in `docs/architecture.md`, `docs/gpu_correctness_guardrails.md`, `docs/kv_cache.md`, `docs/mtp.md`, `docs/scheduler.md`, `docs/benchmarks.md`, and this file.
- Keep `docs/mtp_tpu_spot_findings_2026-05-09.md` accessible as a historical TPU findings record.
- Archive obsolete debug/status markdown under `docs/archive/2026-05-pre-latency-gate/`.
- Avoid duplicating full benchmark logs in canonical docs.
