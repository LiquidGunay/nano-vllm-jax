# Roadmap

This roadmap is grouped by work type. It is not a production-readiness claim.

## Correctness

- Keep `tests/test_mtp_commit_semantics.py` passing on TPU.
- Preserve K=1 accept/reject commit invariants across block boundaries and inactive rows.
- Preserve the K=1 rejected-row no-reseed invariant until rejected-row next-draft state is formally proven.
- Keep target logits sourced from the canonical executor forward-step contract.
- Add coverage for per-bucket MTP admission behavior once EWMA is bucketed.
- Keep K=2 correctness tests, but do not promote K=2 to serving policy without speed evidence.

## Serving

- Keep current serving policy at K=1 one-pass MTP only.
- Keep MTP admission scheduler-owned.
- Gate on acceptance and measured decode-latency EWMA.
- Replace global latency EWMA with per-bucket EWMA.
- Document all serving flags with exact default behavior.

## Benchmarks

- Preserve latest TPU v6e-1 Qwen/Qwen3.5-4B BF16 findings.
- Add repeatable benchmark recipes for low-acceptance and high-acceptance prompts.
- Validate mixed/heterogeneous benchmark recipes on TPU once the v6e-1 VM is reachable again.
- Add vLLM TPU baseline/speculative comparison for Qwen 0.8B, or document exact blockers if unsupported.
- Report decode tokens/sec separately from prefill and compile time.
- Keep correctness checks mandatory for any speed claim.
- Add compact per-bucket reporting once per-bucket EWMA exists.

## Optimization

- Reduce host synchronization in token materialization and MTP accounting.
- Improve bucket-specific admission decisions.
- Investigate K=1 overhead in rejected and fallback steps.
- Implement a safe fast K=1 verifier that can commit rejected rows from after-current-token state without full repair decode.
- Treat K=2 as optimization research only until it beats K=1/baseline in valid benchmarks.
- Avoid dedicated TPU-kernel claims; current accelerator path is JAX/XLA on TPU.

## Cleanup

- Keep canonical docs in `docs/architecture.md`, `docs/kv_cache.md`, `docs/mtp.md`, `docs/scheduler.md`, `docs/benchmarks.md`, and this file.
- Keep `docs/mtp_tpu_spot_findings_2026-05-09.md` accessible as the latest detailed findings record.
- Archive obsolete debug/status markdown under `docs/archive/2026-05-pre-latency-gate/`.
- Avoid duplicating full benchmark logs in canonical docs.
