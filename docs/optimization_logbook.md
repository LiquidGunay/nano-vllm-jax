# Optimization Logbook

This file is the compact optimization record. Keep one entry per profiled JAX run and link the exact profile directory plus benchmark artifact. Profiles live under `/mountpoint/.exp/profiles` and can be opened with:

```bash
tensorboard --logdir /mountpoint/.exp/profiles
```

Perfetto can open the `*.trace.json.gz` file inside each profile directory.

## Entry 001 - Heterogeneous Batch Baseline

- run id: `20260525-180419-1816768-jax_hetero8_64_512x32_baseline`
- commit: `40f7e3a7c6e4b3995e2f80e0935bf74690f4635d`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_baseline.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-180419-1816768-jax_hetero8_64_512x32_baseline`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-180419-1816768-jax_hetero8_64_512x32_baseline/plugins/profile/2026_05_25_18_05_55/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-180419-1816768-jax_hetero8_64_512x32_baseline/plugins/profile/2026_05_25_18_05_55/INDCS0291.atrapa.deloitte.com.xplane.pb`
- shape: input lengths `[64, 128, 192, 256, 320, 384, 448, 512]`, output length `32`, batch size `8`
- runtime shape controls: prefill buckets `[64, 128, 256, 384, 512]`, batch buckets `[1, 2, 4, 8]`, `max_blocks_per_seq=40`
- correctness: not reference-checked inside the JAX artifact; manually compared against vLLM async baseline below and all generated token rows matched.
- JAX timing: `81.05 tok/s`, TTFT p50 `1416.66 ms`, ITL p50 `51.87 ms`, ITL p95 `57.99 ms`
- matching vLLM artifact: `results/qwen08_vllm_async_delta_baseline_hetero8_64_512x32.json`
- vLLM timing: `864.18 tok/s`, TTFT p50 `127.80 ms`, ITL p50 `5.46 ms`, ITL p95 `6.40 ms`
- gap: JAX is `0.094x` of vLLM total tokens/sec on this shape.

Profile-backed hypotheses:

- The heterogeneous prefill is padded to an effective `[8, 512]` rectangular shape even though the true prompt tokens total `2304`. This likely dominates TTFT.
- Decode remains far slower than vLLM: JAX ITL p50 is about `9.5x` vLLM. Candidate sources are full-vocab logits transfer/sample separation, hybrid-state store/load movement, and unfused model kernels.
- vLLM logs show optimized kernels for this architecture: Triton/FLA GDN prefill, FlashAttention 2, torch.compile, and CUDA graph capture. The equivalent lowered paths should remain optional in this repo.

Decision:

- First JAX experiment: add an optional greedy-token fast path for temperature-0 generation so the compiled step returns token IDs instead of full vocab logits and avoids the separate sampling JIT. This is narrow, model-side, and preserves the existing logits path for correctness diagnostics.
- Next evidence to capture after the change: same hetero8 profile with `NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH=1`, exact generated-token match against this baseline, and the token/sec/TTFT/ITL delta.

## Entry 002 - Optional Greedy Token Fast Path

- run id: `20260525-181837-1821856-jax_hetero8_64_512x32_greedy_token_fastpath`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_greedy_token_fastpath.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-181837-1821856-jax_hetero8_64_512x32_greedy_token_fastpath`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-181837-1821856-jax_hetero8_64_512x32_greedy_token_fastpath/plugins/profile/2026_05_25_18_19_02/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-181837-1821856-jax_hetero8_64_512x32_greedy_token_fastpath/plugins/profile/2026_05_25_18_19_02/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: `NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH=1` enables `ModelExecutor.forward_step_token_ids_jit` for temperature-0, non-MTP generation when prefill logits are not being captured.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- JAX timing: `88.90 tok/s`, TTFT p50 `1282.83 ms`, ITL p50 `50.09 ms`, ITL p95 `51.84 ms`
- delta vs Entry 001: `1.097x` total tok/s, TTFT p50 `0.905x`, ITL p50 `0.966x`
- gap vs vLLM Entry 001 comparison: JAX is now `0.103x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 001 ms | Entry 002 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 3158.64 | 2879.48 | end-to-end measured region |
| `_run_main_and_sample` | 2975.89 | 2705.56 | main runner hot path |
| `forward_step_jit` / `forward_step_token_ids_jit` | 1446.33 | 1403.90 | modest direct reduction |
| `_store_batch_hybrid_state` | 783.26 | 659.05 | still large; likely next model-runner state-movement target |
| `MemcpyD2D` | 796.65 | 793.84 | essentially unchanged |
| `__getitem__`/`rewriting_take` cluster | about 612+ ms | about 529+ ms | still visible host/JAX indexing overhead |

Decision:

- Keep the fast path optional for now. It is correctness-clean on the hetero8 reference and gives a useful but insufficient win.
- Next JAX experiment should target hybrid-state movement and indexing overhead, while preserving the existing ragged scheduled batch contract.
- Prefill padding remains the largest TTFT problem, but it needs a separate grouped-prefill study before changing the algorithm.

## Entry 003 - Hybrid-State Host Metadata Fast Path

- run id: `20260525-182354-1824651-jax_hetero8_64_512x32_statefast_greedy_token`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_statefast_greedy_token.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-182354-1824651-jax_hetero8_64_512x32_statefast_greedy_token`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-182354-1824651-jax_hetero8_64_512x32_statefast_greedy_token/plugins/profile/2026_05_25_18_24_20/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-182354-1824651-jax_hetero8_64_512x32_statefast_greedy_token/plugins/profile/2026_05_25_18_24_20/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: scheduled batches carry host-side `seq_ids` and query lengths, and `ModelRunner` skips gather/scatter when hybrid slots cover the full table contiguously.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- JAX timing: `96.25 tok/s`, TTFT p50 `1337.44 ms`, ITL p50 `41.38 ms`, ITL p95 `43.20 ms`
- delta vs Entry 001: `1.188x` total tok/s
- delta vs Entry 002: `1.083x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.111x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 002 ms | Entry 003 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 2879.48 | 2659.69 | end-to-end measured region |
| `_run_main_and_sample` | 2705.56 | 2489.42 | main runner hot path |
| `forward_step_token_ids_jit` | 1403.90 | 1397.74 | unchanged; this experiment targeted runner state movement |
| `_store_batch_hybrid_state` | 659.05 | 301.62 | clear improvement from full-table replacement |
| `np.asarray(jax.Array)` | 489.62 | 132.54 | host metadata avoids repeated host array materialization |
| `MemcpyD2D` | 793.84 | 786.27 | still effectively unchanged |
| `__getitem__`/`rewriting_take` cluster | about 529+ ms | about 548+ ms | still visible and not solved by this path |

Decision:

- Keep the host metadata/full-table fast path. It is correctness-clean on the hetero8 reference and materially improves decode ITL.
- Add a boundary test for the contiguous full-table path and the inactive-row non-replacement case, because stale host metadata would be a high-risk correctness failure.
- Next profile-backed work should investigate remaining `__getitem__`/`rewriting_take` and D2D movement before changing scheduler behavior.
