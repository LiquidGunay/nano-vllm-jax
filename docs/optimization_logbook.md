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

## Entry 004 - Hybrid Slot Row Locality

- run id: `20260525-183457-1828944-jax_hetero8_64_512x32_row_slot_greedy_token`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_row_slot_greedy_token.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-183457-1828944-jax_hetero8_64_512x32_row_slot_greedy_token`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-183457-1828944-jax_hetero8_64_512x32_row_slot_greedy_token/plugins/profile/2026_05_25_18_35_22/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-183457-1828944-jax_hetero8_64_512x32_row_slot_greedy_token/plugins/profile/2026_05_25_18_35_22/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: newly allocated hybrid-state slots prefer the current physical batch row when that slot is free. This keeps the profiled post-warmup batch in slot order `[0..7]` even when sequence IDs have advanced beyond `max_num_seqs`.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- JAX timing: `112.82 tok/s`, TTFT p50 `1234.67 ms`, ITL p50 `31.37 ms`, ITL p95 `40.19 ms`
- delta vs Entry 003: `1.172x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.131x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 003 ms | Entry 004 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 2659.69 | 2269.10 | end-to-end measured region |
| `_run_main_and_sample` | 2489.42 | 2084.29 | main runner hot path |
| `forward_step_token_ids_jit` | 1397.74 | 1387.97 | model step mostly unchanged |
| `_batch_hybrid_state` | 311.40 | 45.43 | full-table path now actually triggers after warmup |
| `_store_batch_hybrid_state` | 301.62 | 0.74 | contiguous full-table state replacement avoids scatter |
| `__getitem__`/`rewriting_take` cluster | about 548 ms | about 319 ms | remaining indexing moved elsewhere |
| `MemcpyD2D` | 786.27 | 689.14 | improved but still large |

Decision:

- Keep the row-local hybrid slot assignment. It is correctness-clean and removes most of the intended hybrid-state gather/scatter overhead.
- Add a boundary guard that new sequence IDs beyond `max_num_seqs` still choose physical row slots when available.
- The next visible Python/JAX overhead is active-row bookkeeping in `_run_main_and_sample`, which still materializes `batch.query_lens` and gathers all active token IDs.

## Entry 005 - Host Active-Row Bookkeeping

- run id: `20260525-183802-1830207-jax_hetero8_64_512x32_host_active_rows_greedy_token`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_host_active_rows_greedy_token.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-183802-1830207-jax_hetero8_64_512x32_host_active_rows_greedy_token`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-183802-1830207-jax_hetero8_64_512x32_host_active_rows_greedy_token/plugins/profile/2026_05_25_18_38_26/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-183802-1830207-jax_hetero8_64_512x32_host_active_rows_greedy_token/plugins/profile/2026_05_25_18_38_26/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: `_run_main_and_sample` uses host `query_lens`/`seq_ids` metadata for active-row selection and skips the token-ID gather when all logical rows are active.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- JAX timing: `132.22 tok/s`, TTFT p50 `1218.24 ms`, ITL p50 `21.60 ms`, ITL p95 `24.62 ms`
- delta vs Entry 004: `1.172x` total tok/s
- delta vs Entry 003: `1.374x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.153x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 004 ms | Entry 005 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 2269.10 | 1936.15 | end-to-end measured region |
| `_run_main_and_sample` | 2084.29 | 1764.35 | main runner hot path |
| `forward_step_token_ids_jit` | 1387.97 | 1357.39 | modest improvement from less surrounding JAX work |
| active-row list comprehension | 200.60 | 0.06 | host metadata avoids `query_lens` materialization |
| `__getitem__`/`rewriting_take` cluster | about 319 ms | about 112 ms | remaining indexing mostly metadata/snapshot work |
| `compute_slot_mapping` | 125.82 | 115.90 | still rebuilt each step |
| `MemcpyD2D` | 689.14 | 679.62 | still the largest named GPU-side copy bucket |

Decision:

- Keep this path. It is correctness-clean and removes the largest remaining Python-side active-row overhead.
- Remaining model-side targets are now inside the compiled step and cache metadata/snapshot work: `MemcpyD2D`, `compute_slot_mapping`, `_refresh_kv_snapshot`, and paged-attention gathers.

## Entry 006 - Rejected Width-1 Gated DeltaNet Scan Bypass

- run id: `20260525-184154-1841799-jax_hetero8_64_512x32_gdn_width1_host_active`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_width1_host_active.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-184154-1841799-jax_hetero8_64_512x32_gdn_width1_host_active`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-184154-1841799-jax_hetero8_64_512x32_gdn_width1_host_active/plugins/profile/2026_05_25_18_42_31/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-184154-1841799-jax_hetero8_64_512x32_gdn_width1_host_active/plugins/profile/2026_05_25_18_42_31/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: explicit `time_dim == 1` branch in `jax_recurrent_gated_delta_rule` using the same `step_one` math instead of a one-iteration `lax.scan`.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused recurrent tests also passed on GPU.
- JAX timing: `124.61 tok/s`, TTFT p50 `1223.34 ms`, ITL p50 `21.95 ms`, ITL p95 `61.90 ms`
- delta vs Entry 005: `0.942x` total tok/s, so the change regressed throughput.

Top trace ranges, total inclusive time:

| range | Entry 005 ms | Entry 006 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1936.15 | 2054.43 | regressed |
| `_run_main_and_sample` | 1764.35 | 1833.71 | regressed |
| `forward_step_token_ids_jit` | 1357.39 | 1439.53 | regressed despite simpler source |
| `MemcpyD2D` | 679.62 | 652.55 | copy bucket improved slightly |
| `PjRtCApiLoadedExecutable::Execute` | 1344.00 | 1440.03 | compiled execution cost increased |
| `compute_slot_mapping` | 115.90 | 139.61 | noisy/regressed |

Decision:

- Do not keep the width-1 recurrent source rewrite. It reduced the D2D copy bucket slightly but increased compiled execution time enough to hurt end-to-end throughput.
- Revisit Gated DeltaNet only with HLO/kernel evidence or an optional lowered backend path. The pure JAX source-level bypass is not a win on this GPU profile.
- Next better target: cache metadata/snapshot work (`compute_slot_mapping`, `_refresh_kv_snapshot`) or paged decode attention key-slot reuse across full-attention layers.
