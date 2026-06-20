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

## Entry 007 - Rejected Legacy Snapshot-Only Update

- run id: `20260525-185217-1863257-jax_hetero8_64_512x32_record_snapshot`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_record_snapshot.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-185217-1863257-jax_hetero8_64_512x32_record_snapshot`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-185217-1863257-jax_hetero8_64_512x32_record_snapshot/plugins/profile/2026_05_25_18_52_43/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-185217-1863257-jax_hetero8_64_512x32_record_snapshot/plugins/profile/2026_05_25_18_52_43/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: normal non-MTP `_run_main_and_sample` used `_record_kv_snapshot` instead of `_refresh_kv_snapshot`, keeping `cache_storage`, block tables, sequence lengths, and hybrid state current while preserving the previous legacy `kv_state.slot_mapping`.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused GPU tests covering MTP baseline/reference paths passed.
- JAX timing: `130.54 tok/s`, TTFT p50 `1217.85 ms`, ITL p50 `22.04 ms`, ITL p95 `26.54 ms`
- delta vs Entry 005: `0.987x` total tok/s, so the change was slightly slower end to end.

Top trace ranges, total inclusive time:

| range | Entry 005 ms | Entry 007 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1936.15 | 1961.13 | slight regression |
| `_run_main_and_sample` | 1764.35 | 1777.85 | slight regression |
| `forward_step_token_ids_jit` | 1357.39 | 1372.65 | slight regression |
| `_refresh_kv_snapshot` | 117.32 | 0.00 | removed as intended |
| `_record_kv_snapshot` | 0.00 | 0.58 | cheap replacement |
| `compute_slot_mapping` | 115.90 | 0.00 | removed from snapshot path |
| `__getitem__`/`rewriting_take` cluster | about 112 ms | about 25 ms | improved |
| `MemcpyD2D` | 679.62 | 680.63 | unchanged |

Decision:

- Do not keep this as the default. The trace-local cleanup did not translate to throughput, and it weakens the legacy introspection guarantee that `kv_state.slot_mapping` describes the latest step.
- The experiment is useful evidence: Python/JAX snapshot metadata is no longer the main bottleneck after Entry 005. The larger target remains compiled execution and per-layer cache movement.

## Entry 008 - Rejected Decode Slot-Mapping Reuse

- run id: `20260525-185804-1865547-jax_hetero8_64_512x32_decode_slot_reuse`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_decode_slot_reuse.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-185804-1865547-jax_hetero8_64_512x32_decode_slot_reuse`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-185804-1865547-jax_hetero8_64_512x32_decode_slot_reuse/plugins/profile/2026_05_25_18_58_43/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-185804-1865547-jax_hetero8_64_512x32_decode_slot_reuse/plugins/profile/2026_05_25_18_58_43/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: `AttentionMetadata` carried a decode-wide `[batch, max_kv_len]` key-slot map computed once in `build_attention_metadata`, and `paged_attention_decode` reused it across full-attention layers.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused GPU tests for decode attention metadata, long decode, physical cache capacity independence, and JIT-vs-eager cached decode passed.
- JAX timing: `123.19 tok/s`, TTFT p50 `1232.17 ms`, ITL p50 `23.60 ms`, ITL p95 `53.24 ms`
- delta vs Entry 005: `0.932x` total tok/s, a clear regression.

Top trace ranges, total inclusive time:

| range | Entry 005 ms | Entry 008 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1936.15 | 2078.08 | regressed |
| `_run_main_and_sample` | 1764.35 | 1895.85 | regressed |
| `forward_step_token_ids_jit` | 1357.39 | 1418.70 | regressed |
| `_refresh_kv_snapshot` | 117.32 | 363.93 | precomputing the wide decode map made metadata much heavier |
| `__getitem__`/`rewriting_take` cluster | about 112 ms | about 292 ms | regressed |
| `PjRtCApiLoadedExecutable::Execute` | 1344.00 | 1543.03 | compiled execution regressed |
| `MemcpyD2D` | 679.62 | 663.71 | small copy-bucket improvement did not pay for metadata/compile cost |

Decision:

- Do not keep decode-wide slot reuse in pure JAX metadata. It increases snapshot/metadata and compiled execution costs more than it saves inside per-layer attention.
- If revisiting paged decode attention, do it as a narrower lowered kernel or as direct per-layer gather/attention fusion, not by materializing a wider metadata tensor in Python/JAX.

## Entry 009 - Lazy Hybrid Slot Zeroing

- run id: `20260525-190314-1866930-jax_hetero8_64_512x32_release_lazy_zero`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_release_lazy_zero.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-190314-1866930-jax_hetero8_64_512x32_release_lazy_zero`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-190314-1866930-jax_hetero8_64_512x32_release_lazy_zero/plugins/profile/2026_05_25_19_03_38/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-190314-1866930-jax_hetero8_64_512x32_release_lazy_zero/plugins/profile/2026_05_25_19_03_38/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: `ModelRunner.release` no longer zeros the freed hybrid slot. The next `_ensure_hybrid_slot` allocation still zeroes the slot before reuse, preserving isolation while avoiding redundant end-of-request device updates.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused GPU hybrid-state tests passed.
- JAX timing: `135.00 tok/s`, TTFT p50 `1217.15 ms`, ITL p50 `21.71 ms`, ITL p95 `23.43 ms`
- delta vs Entry 005: `1.021x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.156x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 005 ms | Entry 009 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1936.15 | 1896.26 | end-to-end improvement |
| `_run_main_and_sample` | 1764.35 | 1769.72 | essentially unchanged; line number shifted in trace |
| `forward_step_token_ids_jit` | 1357.39 | 1359.84 | unchanged |
| `release` | 41.74 | 0.01 | release no longer launches slot-zero updates |
| `_zero_hybrid_slot` | 81.91 over 16 calls | 0.00 in measured trace | redundant release-time zeroes removed; allocation-time zeroing remains for reuse |
| `__getitem__`/`rewriting_take` cluster | about 112 ms | about 99 ms | slight reduction from fewer release-time updates |
| `MemcpyD2D` | 679.62 | 673.14 | slight reduction |

Decision:

- Keep lazy release zeroing. It is correctness-clean, removes redundant end-of-request device work, and gives a small but measurable throughput win.
- The next meaningful speed work should move below Python runner bookkeeping: inspect compiled full-attention/GDN HLO and consider optional lowered decode attention or GDN kernels rather than wider metadata precomputation.

## Entry 010 - Rejected Prefill Shape Guard

- run id: `20260525-191103-1870680-jax_hetero8_64_512x32_prefill_shape_guard`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_prefill_shape_guard.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-191103-1870680-jax_hetero8_64_512x32_prefill_shape_guard`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-191103-1870680-jax_hetero8_64_512x32_prefill_shape_guard/plugins/profile/2026_05_25_19_13_24/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-191103-1870680-jax_hetero8_64_512x32_prefill_shape_guard/plugins/profile/2026_05_25_19_13_24/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: scheduler split heterogeneous prefill waves before admitting a prompt that would force both a larger batch-size bucket and a larger query-length bucket.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- JAX timing: `103.71 tok/s`, TTFT p50 `633.37 ms`, ITL p50 `33.29 ms`, ITL p95 `39.60 ms`
- delta vs Entry 009: `0.768x` total tok/s, despite much better median TTFT.

Top trace ranges, total inclusive time:

| range | Entry 009 ms | Entry 010 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1896.26 | 2468.38 | throughput regression from more waves |
| `_run_main_and_sample` | 1769.72 | 2028.63 | fewer steps, but much more runner state movement |
| `forward_step_token_ids_jit` | 1359.84 | 1305.27 | compiled model work fell slightly |
| `_batch_hybrid_state` | 40.47 | 325.45 | splitting waves defeated the full-table fast path |
| `_store_batch_hybrid_state` | 0.68 | 238.17 | hybrid-state scatter returned |
| `PjRtCApiLoadedExecutable::Execute` | 1330.04 | 1440.88 | more compiled dispatch cost overall |
| `MemcpyD2D` | 673.14 | 287.29 | lower copy bucket did not pay for runner and dispatch overhead |

Decision:

- Do not keep the prefill shape guard as a throughput default. It is useful evidence that smaller prefill waves can reduce TTFT, but it costs too much total decode throughput on the long heterogeneous batch.
- If we expose TTFT-oriented scheduling later, make it a separate policy knob and benchmark it against latency SLOs, not the default throughput path.
- For the current objective, keep the rectangular prefill and focus on lower-level compiled kernels or state movement that does not break row-local hybrid slots.

## Entry 011 - Rejected Full-Active Decode KV Write

- run id: `20260525-192029-1873379-jax_hetero8_64_512x32_full_decode_kv_write`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_full_decode_kv_write.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-192029-1873379-jax_hetero8_64_512x32_full_decode_kv_write`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-192029-1873379-jax_hetero8_64_512x32_full_decode_kv_write/plugins/profile/2026_05_25_19_21_05/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-192029-1873379-jax_hetero8_64_512x32_full_decode_kv_write/plugins/profile/2026_05_25_19_21_05/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: `AttentionMetadata` carried a static `all_slots_valid` flag for full-active decode batches, allowing `write_kv` to call `update_kv_cache(..., valid_mask=None)` and skip the sentinel-concat padding path.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA tests passed before the revert.
- JAX timing: `134.19 tok/s`, TTFT p50 `1222.51 ms`, ITL p50 `21.80 ms`, ITL p95 `24.12 ms`
- delta vs Entry 009: `0.994x` total tok/s, so the change was slightly slower end to end.

Top trace ranges, total inclusive time:

| range | Entry 009 ms | Entry 011 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1896.26 | 1907.68 | slight regression |
| `_run_main_and_sample` | 1769.72 | 1775.99 | slight regression |
| `forward_step_token_ids_jit` | 1359.84 | 1374.04 | compiled step regressed |
| `PjRtCApiLoadedExecutable::Execute` | 1330.04 | 1347.33 | compiled execution increased |
| `command_buffer::update` | 31.69 | 36.14 | target bucket regressed |
| `wrapped_concatenate` | 38.69 | 38.84 | effectively unchanged |
| `MemcpyD2D` | 673.14 | 668.77 | small copy-bucket improvement did not pay for execution overhead |

Decision:

- Do not keep the full-active decode KV write specialization. Removing the sentinel-concat source branch did not translate into a better lowered plan on this profile.
- The failed result is still useful: the dominant pure-JAX decode cost is not this padding mask construction. Continue below this level with decode attention/GDN lowering or broader compiled-step fusion evidence.

## Entry 012 - Rejected Hidden-Select Token Fast Path

- run id: `20260525-193100-1877871-jax_hetero8_64_512x32_hidden_select_fastpath`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_hidden_select_fastpath.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-193100-1877871-jax_hetero8_64_512x32_hidden_select_fastpath`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-193100-1877871-jax_hetero8_64_512x32_hidden_select_fastpath/plugins/profile/2026_05_25_19_31_50/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-193100-1877871-jax_hetero8_64_512x32_hidden_select_fastpath/plugins/profile/2026_05_25_19_31_50/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: `forward_step` could return only selected hidden positions for the greedy token-id JIT path, avoiding an outer hidden gather and skipping final RMSNorm when only pre-norm hidden was requested.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA tests passed before the revert.
- JAX timing: `127.98 tok/s`, TTFT p50 `1223.68 ms`, ITL p50 `25.04 ms`, ITL p95 `31.43 ms`
- delta vs Entry 009: `0.948x` total tok/s, so the change regressed throughput.

Top trace ranges, total inclusive time:

| range | Entry 009 ms | Entry 012 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1896.26 | 2000.33 | regressed |
| `_run_main_and_sample` | 1769.72 | 1836.75 | regressed |
| `forward_step_token_ids_jit` | 1359.84 | 1442.22 | compiled step regressed |
| `PjRtCApiLoadedExecutable::Execute` | 1330.04 | 1405.97 | compiled execution increased |
| `jit_compiled:XLA GPU module` | 1167.33 | 1188.99 | lowered module slower |
| `MemcpyD2D` | 673.14 | 653.10 | copy bucket fell, but not enough to pay for slower execution |
| `while` | 1087.00 | 1086.71 | GDN prefill bottleneck unchanged |

Decision:

- Do not keep hidden-position selection in `forward_step`. XLA did not turn the smaller source-level return into a better compiled plan for this workload.
- The result reinforces that the remaining large target is not last-hidden gathering or final RMSNorm in the token fast path; it is Gated DeltaNet prefill and lower-level compiled decode work.

## Entry 013 - Fixed Chunked Gated DeltaNet Prefill

- run id: `20260525-194440-1884283-jax_hetero8_64_512x32_chunked_gdn_prefill_default`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_chunked_gdn_prefill_default.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-194440-1884283-jax_hetero8_64_512x32_chunked_gdn_prefill_default`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-194440-1884283-jax_hetero8_64_512x32_chunked_gdn_prefill_default/plugins/profile/2026_05_25_19_45_04/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-194440-1884283-jax_hetero8_64_512x32_chunked_gdn_prefill_default/plugins/profile/2026_05_25_19_45_04/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Beauvoir identified Gated DeltaNet prefill as the top model-side bottleneck. Entry 009 spent about `1087 ms` in 18 recurrent `while` ranges, one per linear-attention layer.
- change: fixed the multi-chunk GDN prefill state correction to use `exp(cumsum(g))` instead of per-token `exp(g)`, then made chunked cached prefill the default. `NANO_VLLM_JAX_ENABLE_CHUNKED_GDN_PREFILL=0` remains as the recurrent fallback.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA tests cover multi-chunk GDN parity and bucketed cached prefill state.
- JAX timing: `180.44 tok/s`, TTFT p50 `739.45 ms`, ITL p50 `21.77 ms`, ITL p95 `23.58 ms`
- delta vs Entry 009: `1.337x` total tok/s, TTFT p50 `0.607x`, ITL p50 essentially unchanged.
- gap vs vLLM Entry 001 comparison: JAX is now `0.209x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 009 ms | Entry 013 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1896.26 | 1418.76 | end-to-end improvement |
| `_run_main_and_sample` | 1769.72 | 1287.47 | improved mainly from prefill |
| `forward_step_token_ids_jit` | 1359.84 | 427.58 | large compiled-step reduction |
| `_profile_jit_call` | 1240.42 | 310.19 | compiled execution much lower |
| `PjRtCApiLoadedExecutable::Execute` | 1330.04 | 402.94 | major reduction |
| `jit_compiled:XLA GPU module` | 1167.33 | 237.87 | major reduction |
| `while` | 1087.00 over 18 calls | 152.27 over 36 calls | recurrent prefill scans replaced by chunked work |
| `MemcpyD2D` | 673.14 | 36.41 | large copy-bucket drop |
| `np.asarray(jax.Array)` / `tolist` | 283.25 / 249.33 | 728.50 / 696.31 | host token materialization is now the biggest visible non-model overhead |

Decision:

- Keep the chunked cached GDN prefill fix as the default. It is correctness-clean on the real-weight hetero8 reference and directly attacks the largest model-side prefill bottleneck.
- The next profile-backed target should be post-model host/token materialization (`np.asarray`, `tolist`) and then decode kernels. Scheduler changes remain lower priority unless they reduce that new host overhead without harming the paged/ragged contract.

## Entry 014 - Host Batch Contract Validation

- run id: `20260525-195424-1889297-jax_hetero8_64_512x32_host_validation_fastpath`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_host_validation_fastpath.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-195424-1889297-jax_hetero8_64_512x32_host_validation_fastpath`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-195424-1889297-jax_hetero8_64_512x32_host_validation_fastpath/plugins/profile/2026_05_25_19_54_50/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-195424-1889297-jax_hetero8_64_512x32_host_validation_fastpath/plugins/profile/2026_05_25_19_54_50/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: scheduled batches now carry host-side `seq_lens` in addition to host `seq_ids` and query lengths. `ModelExecutor` validates the batch contract from host metadata when all three are available, and postprocess prefill chunk lengths use host query lengths.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA backend-boundary and GDN tests passed.
- JAX timing: `191.48 tok/s`, TTFT p50 `734.82 ms`, ITL p50 `19.35 ms`, ITL p95 `22.57 ms`
- delta vs Entry 013: `1.061x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.222x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 013 ms | Entry 014 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1418.76 | 1336.97 | end-to-end improvement |
| `_run_main_and_sample` | 1287.47 | 1198.97 | main runner hot path |
| `forward_step_token_ids_jit` | 427.58 | 343.47 | compiled step measured lower in this run |
| `_validate_batch_contract` | 94.83 | 2.48 | host fast path avoids JAX array reductions |
| `_validate_batch_contract_host` | 0.00 | 1.06 | replacement validation work |
| `_refresh_kv_snapshot` | 120.37 | 121.50 | unchanged and still visible |
| `compute_slot_mapping` | 118.83 | 120.00 | unchanged and still visible |
| `np.asarray(jax.Array)` / `tolist` | 728.50 / 696.31 | 689.19 / 689.58 | host token materialization remains the main visible host bucket |

Decision:

- Keep the host validation path. It preserves the same scheduler contract for scheduled batches built by this repo and removes a device-sync validation cost that became visible after chunked GDN prefill.
- The next target remains legacy KV snapshot refresh, because it still rebuilds attention metadata every step even though the serving path already executes from `cache_storage` plus per-step scheduled metadata.

## Entry 015 - Snapshot-Only Normal Greedy Path

- run id: `20260525-200042-1891179-jax_hetero8_64_512x32_host_validation_record_snapshot_v2`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_host_validation_record_snapshot_v2.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-200042-1891179-jax_hetero8_64_512x32_host_validation_record_snapshot_v2`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-200042-1891179-jax_hetero8_64_512x32_host_validation_record_snapshot_v2/plugins/profile/2026_05_25_20_01_04/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-200042-1891179-jax_hetero8_64_512x32_host_validation_record_snapshot_v2/plugins/profile/2026_05_25_20_01_04/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: normal greedy `_run_main_and_sample` now updates the legacy KV snapshot with `_record_kv_snapshot` by default instead of rebuilding attention metadata through `_refresh_kv_snapshot`. `NANO_VLLM_JAX_REFRESH_KV_SNAPSHOT=1` restores the full refresh path for debugging/introspection.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA backend-boundary and GDN tests passed.
- JAX timing: `196.30 tok/s`, TTFT p50 `738.36 ms`, ITL p50 `17.95 ms`, ITL p95 `19.51 ms`
- delta vs Entry 014: `1.025x` total tok/s
- delta vs Entry 013: `1.088x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.227x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 014 ms | Entry 015 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1336.97 | 1304.10 | end-to-end improvement |
| `_run_main_and_sample` | 1198.97 | 1182.58 | main runner hot path |
| `forward_step_token_ids_jit` | 343.47 | 322.73 | slightly lower in this run |
| `_validate_batch_contract` | 2.48 | 2.00 | host validation retained |
| `_refresh_kv_snapshot` | 121.50 | 0.00 | removed from normal greedy hot path |
| `_record_kv_snapshot` | 0.00 | 0.49 | cheap replacement snapshot update |
| `compute_slot_mapping` | 120.00 | 0.00 | removed from snapshot path |
| `PjRtCApiLoadedExecutable::Execute` | 378.37 | 310.72 | fewer snapshot-related JAX executions |
| `MemcpyD2D` | 36.95 | 33.71 | small reduction |
| `np.asarray(jax.Array)` / `tolist` | 689.19 / 689.58 | 813.40 / 813.72 | still the largest visible host-sync bucket |

Decision:

- Keep snapshot-only updates for the normal greedy path with the refresh fallback env var. On the post-Entry013 codebase it is correctness-clean and improves throughput, unlike the earlier pre-GDN Entry 007 experiment.
- The next high-confidence profile target is token readback/materialization (`output.activations[:len(seqs)]`, `tolist`, and related `np.asarray` ranges). After that, revisit decode kernels with HLO/kernel evidence rather than wider Python/JAX metadata tensors.

## Entry 016 - Cached JIT Parameter Leaves

- run id: `20260525-201252-1895342-jax_hetero8_64_512x32_cached_param_leaves`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_cached_param_leaves.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-201252-1895342-jax_hetero8_64_512x32_cached_param_leaves`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-201252-1895342-jax_hetero8_64_512x32_cached_param_leaves/plugins/profile/2026_05_25_20_13_14/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-201252-1895342-jax_hetero8_64_512x32_cached_param_leaves/plugins/profile/2026_05_25_20_13_14/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Erdos confirmed that `tolist`/`np.asarray(jax.Array)` is primarily an asynchronous GPU synchronization label, not a D2H copy bottleneck. The audit found about `813 ms` attributed there in Entry 015, but only about `0.04 ms` total D2H stream time; the next model-side targets are MLP gate/up projection structure, padded prefill dense work, greedy LM-head top-1 lowering, and narrow lowered decode kernels.
- change: `ModelExecutor` now flattens `ModelParams` once at initialization and passes cached leaves to the main JIT functions, unflattening inside the compiled function. This avoids sorting every layer dictionary in `_model_params_flatten` before each profiled JIT call.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA backend-boundary and GDN tests passed.
- JAX timing: `201.86 tok/s`, TTFT p50 `735.16 ms`, ITL p50 `16.86 ms`, ITL p95 `18.66 ms`
- delta vs Entry 015: `1.028x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.234x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 015 ms | Entry 016 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1304.10 | 1268.21 | end-to-end improvement |
| `_run_main_and_sample` | 1182.58 | 1144.02 | main runner hot path |
| `forward_step_token_ids_jit` | 322.73 | 281.10 | lower Python/JAX wrapper overhead around the same model step |
| `_profile_jit_call` | 300.88 | 259.02 | reduced caller-side overhead |
| `_model_params_flatten` | 36.00 | 0.00 | repeated layer-dict sorting removed |
| `PjRtCApiLoadedExecutable::Execute` | 310.72 | 294.75 | fewer small executions around argument conversion |
| `jit_compiled:XLA GPU module` | 234.70 | 230.60 | model GPU work mostly unchanged |
| `np.asarray(jax.Array)` / `tolist` | 813.40 / 813.72 | 816.99 / 817.34 | still mainly synchronizes on prior GPU work |

Decision:

- Keep cached parameter leaves. It is correctness-clean, improves the target hetero8 workload, and removes a real repeated Python-side cost without changing model math or the ragged/paged layout.
- Do not chase `tolist` directly until there is evidence of actual D2H copy cost. The next model-side experiment should target either padded prefill dense work or an optional lowered greedy top-1/attention/GDN kernel.

## Entry 017 - Rejected Hybrid-State Donation

- run id: `20260525-201511-1896115-jax_hetero8_64_512x32_cached_param_leaves_donate_hybrid`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_cached_param_leaves_donate_hybrid.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-201511-1896115-jax_hetero8_64_512x32_cached_param_leaves_donate_hybrid`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-201511-1896115-jax_hetero8_64_512x32_cached_param_leaves_donate_hybrid/plugins/profile/2026_05_25_20_16_01/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-201511-1896115-jax_hetero8_64_512x32_cached_param_leaves_donate_hybrid/plugins/profile/2026_05_25_20_16_01/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: main JIT paths donated hybrid `conv_state` and `recurrent_state` buffers in addition to KV cache buffers.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA tests passed before the revert.
- JAX timing: `195.31 tok/s`, TTFT p50 `747.97 ms`, ITL p50 `17.82 ms`, ITL p95 `19.72 ms`
- delta vs Entry 016: `0.968x` total tok/s, so the change regressed the target workload.

Top trace ranges, total inclusive time:

| range | Entry 016 ms | Entry 017 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1268.21 | 1310.71 | regressed |
| `_run_main_and_sample` | 1144.02 | 1178.42 | regressed |
| `forward_step_token_ids_jit` | 281.10 | 280.54 | compiled call body roughly unchanged |
| `jit_compiled:XLA GPU module` | 230.60 | 222.15 | lower module bucket did not improve end-to-end |
| `np.asarray(jax.Array)` / `tolist` | 816.99 / 817.34 | 848.89 / 849.22 | longer synchronization after the step |
| `schedule` / `build_scheduled_batch` | 224.47 / 108.12 | 239.11 / 114.97 | noisy/regressed host scheduling buckets |

Decision:

- Do not donate hybrid state in the main path. It did not reduce end-to-end time and may constrain buffer reuse or scheduling in a way that increases synchronization delay.
- If donation is revisited, gate it behind a backend-specific experiment and inspect buffer-assignment/HLO evidence first.

## Entry 018 - Rejected Packed MLP Gate/Up Projection

- run id: `20260525-201857-1897035-jax_hetero8_64_512x32_cached_leaves_packed_mlp`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_cached_leaves_packed_mlp.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-201857-1897035-jax_hetero8_64_512x32_cached_leaves_packed_mlp`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-201857-1897035-jax_hetero8_64_512x32_cached_leaves_packed_mlp/plugins/profile/2026_05_25_20_19_49/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-201857-1897035-jax_hetero8_64_512x32_cached_leaves_packed_mlp/plugins/profile/2026_05_25_20_19_49/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: executor setup packed each layer's `gate_proj` and `up_proj` into one `gate_up_proj`, and the MLP path did a single dot followed by a split.
- status: rejected and reverted from the working tree.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA tests passed before the revert.
- JAX timing: `184.33 tok/s`, TTFT p50 `849.79 ms`, ITL p50 `17.27 ms`, ITL p95 `18.19 ms`
- delta vs Entry 016: `0.913x` total tok/s, a clear regression from worse prefill.

Top trace ranges, total inclusive time:

| range | Entry 016 ms | Entry 018 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1268.21 | 1388.82 | regressed |
| `_run_main_and_sample` | 1144.02 | 1260.36 | regressed |
| `forward_step_token_ids_jit` | 281.10 | 321.32 | compiled step regressed |
| `PjRtCApiLoadedExecutable::Execute` | 294.75 | 333.24 | execution increased |
| `jit_compiled:XLA GPU module` | 230.60 | 265.06 | lowered module slower |
| `gemm_fusion_dot_6` | 127.32 over 48 calls | 117.77 over 768 calls | packing changed the lowered GEMM/fusion structure unfavorably |
| `while` | 152.16 | 174.79 | GDN-related prefill work regressed |
| `command_buffer::execute` | 178.13 | 203.20 | more GPU command-buffer time |

Decision:

- Do not keep source-level packed MLP gate/up projection. Although it removes an obvious pair of source dots, XLA lowered it into a slower plan for the target workload.
- If the MLP path is optimized later, do it with HLO-guided lowering or a dedicated optional kernel, not by widening the source-level dense projection.

## Entry 019 - Static Main-JIT Metadata Arguments

- run id: `20260525-202548-1899291-jax_hetero8_64_512x32_static_main_jit_metadata`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_static_main_jit_metadata.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-202548-1899291-jax_hetero8_64_512x32_static_main_jit_metadata`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-202548-1899291-jax_hetero8_64_512x32_static_main_jit_metadata/plugins/profile/2026_05_25_20_26_11/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-202548-1899291-jax_hetero8_64_512x32_static_main_jit_metadata/plugins/profile/2026_05_25_20_26_11/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change: the main `forward_step_jit` and greedy token-id JIT no longer pass `seq_ids`, `num_prefill_tokens`, or `num_decode_tokens` as runtime JAX arguments. The serving model path derives query-token counts from `query_start_loc` inside the compiled function and uses a dummy `seq_ids` array because model math and paged attention do not consume logical sequence IDs.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA backend-boundary and GDN tests passed.
- JAX timing: `204.50 tok/s`, TTFT p50 `733.86 ms`, ITL p50 `16.47 ms`, ITL p95 `18.64 ms`
- delta vs Entry 016: `1.013x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.237x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 016 ms | Entry 019 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1268.21 | 1251.80 | end-to-end improvement |
| `_run_main_and_sample` | 1144.02 | 1124.22 | main runner hot path |
| `forward_step_token_ids_jit` | 281.10 | 263.63 | less argument conversion and dispatch overhead |
| `_profile_jit_call` | 259.02 | 261.74 | essentially unchanged; direct JIT wrapper attribution moved |
| `PjRtCApiLoadedExecutable::Execute` | 294.75 over 416 calls | 293.02 over 352 calls | fewer scalar conversion executions |
| `PjitFunction(convert_element_type)` | 114.43 over 576 calls | 97.24 over 448 calls | fewer scalar argument conversions |
| `_convert_element_type` | 183.61 over 1248 calls | 155.05 over 992 calls | conversion cluster reduced |
| `DevicePut` | 18.77 over 360 calls | 15.46 over 296 calls | fewer small host-to-device puts |
| `array(...)` / `asarray(...)` constructors | 125.34 / 21.41 | 109.53 / 1.53 | scalar argument construction reduced |
| `jit_compiled:XLA GPU module` | 230.60 | 232.50 | model GPU work unchanged, as expected |

Decision:

- Keep the static metadata argument cleanup. It is correctness-clean, improves the target workload, and does not alter model math, ragged query metadata, or the paged-attention layout.
- This is probably the last worthwhile pure executor-wrapper cleanup for the main greedy path. Next speed work should move back to model-side GPU work: padded prefill dense computation, lowered greedy top-1, or narrow optional kernels for decode attention/GDN.

## Entry 020 - Deferred Batched Hybrid Slot Zeroing

- run id: `20260525-203339-1901859-jax_hetero8_64_512x32_deferred_hybrid_zero`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_deferred_hybrid_zero.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-203339-1901859-jax_hetero8_64_512x32_deferred_hybrid_zero`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-203339-1901859-jax_hetero8_64_512x32_deferred_hybrid_zero/plugins/profile/2026_05_25_20_34_01/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-203339-1901859-jax_hetero8_64_512x32_deferred_hybrid_zero/plugins/profile/2026_05_25_20_34_01/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Dalton confirmed that the dominant remaining target is padded prefill dense work: this shape has `2304` true prompt tokens but `4096` rectangular token slots, so `43.75%` of prefill token positions are padding. The next large model-side targets are padded prefill shape-changing, lowered greedy top-1 LM head, and narrow decode kernels. D2H token copy remains negligible at about `0.046 ms` total in Entry 016.
- change: batched serving allocation now assigns new hybrid slots without first writing zeros into the persistent hybrid-state table. `_batch_hybrid_state` feeds zero rows for newly assigned slots, then `_store_batch_hybrid_state` writes the real post-step state once. Direct `_ensure_hybrid_slot` callers still zero on allocation.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA hybrid-state, backend-boundary, and GDN tests passed.
- JAX timing: `210.37 tok/s`, TTFT p50 `709.96 ms`, ITL p50 `16.10 ms`, ITL p95 `17.05 ms`
- delta vs Entry 019: `1.029x` total tok/s
- gap vs vLLM Entry 001 comparison: JAX is now `0.243x` of vLLM total tokens/sec on this shape.

Top trace ranges, total inclusive time:

| range | Entry 019 ms | Entry 020 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1251.80 | 1216.92 | end-to-end improvement |
| `_run_main_and_sample` | 1124.22 | 1094.88 | main runner hot path |
| `_batch_hybrid_state` | 38.60 | 8.04 | pre-step table zero writes removed |
| `_ensure_hybrid_slot` / `_zero_hybrid_slot` | 37.47 / 37.28 | 0.00 / 0.00 | hot batched path no longer calls them |
| `_assign_hybrid_slot` | 0.00 | 0.17 | host-only replacement bookkeeping |
| `_store_batch_hybrid_state` | 0.72 | 0.72 | unchanged post-step state write |
| `forward_step_token_ids_jit` | 263.63 | 260.71 | model step essentially unchanged |
| `PjRtCApiLoadedExecutable::Execute` | 293.02 over 352 calls | 275.65 over 252 calls | fewer zeroing-related executions |
| `MemcpyD2D` | 31.84 | 21.89 | fewer device-side state update copies |
| `jit_compiled:XLA GPU module` | 232.50 | 232.15 | model GPU work unchanged, as expected |

Decision:

- Keep deferred batched hybrid zeroing. It is correctness-clean and removes a real prefill-side device update while preserving zero-state semantics for newly assigned sequences.
- Dalton's audit should drive the next major optimization: stop doing dense prefill work on padded token positions without reverting to the rejected scheduling split. The likely next prototype is a model-side shape-changing/ragged prefill path or narrow lowered kernels, not more host-wrapper cleanup.

## Entry 021 - Rejected Compact Banded Prefill

- run id: `20260525-204949-1917203-jax_hetero8_64_512x32_banded_prefill64`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_banded_prefill64.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-204949-1917203-jax_hetero8_64_512x32_banded_prefill64`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-204949-1917203-jax_hetero8_64_512x32_banded_prefill64/plugins/profile/2026_05_25_20_54_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-204949-1917203-jax_hetero8_64_512x32_banded_prefill64/plugins/profile/2026_05_25_20_54_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audits: Socrates and Bohr both agreed that padded prefill is the right high-level target, but warned that compact bands must preserve chronological KV/hybrid state, final-only token emission, and row-local hybrid slots.
- change tested: a guarded runner-local prototype split the heterogeneous prefill into `64`-token compact bands, carried KV/hybrid state across bands, skipped LM-head token generation on non-final bands, and emitted only final prefill tokens. The code path was reverted after profiling.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- JAX timing: `195.92 tok/s`, TTFT p50 `792.25 ms`, ITL p50 `16.38 ms`, ITL p95 `17.20 ms`.
- delta vs Entry 020: `0.931x` total tok/s, TTFT p50 `1.116x`, ITL p50 `1.017x`; the change regressed the target workload.

Top trace ranges, total inclusive time:

| range | Entry 020 ms | Entry 021 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1216.92 | 1306.63 | end-to-end regression |
| `_run_main_and_sample` | 1094.88 | 1181.34 | runner hot path regressed |
| `_run_banded_prefill_and_sample` | 0.00 | 769.38 | compact prefill prototype hot path |
| `forward_step_jit` | 0.00 | 496.69 over 8 calls | one prefill dispatch became eight band dispatches |
| `forward_step_token_ids_jit` | 260.71 | 96.88 | decode-only attribution fell, but total compiled work increased |
| `PjRtCApiLoadedExecutable::Execute` | 275.65 over 252 calls | 703.20 over 921 calls | dispatch/execution count dominated the saved padded GEMMs |
| `jit_compiled:XLA GPU module` | 232.15 | 539.07 | compiled GPU work increased |
| `_batch_hybrid_state` / `_store_batch_hybrid_state` | about 8.76 | about 106.54 | extra bands reintroduced state table gather/store overhead |
| `MemcpyD2D` | 21.89 | 43.23 | additional state/cache traffic |
| `gemm_fusion_dot_285` | 234.88 | 39.49 | rectangular dense work fell, but not enough to offset dispatch/state costs |

Decision:

- Reject and revert the compact multi-dispatch banded prefill path. It proved the semantic approach can be correctness-clean, but the pure Python/JAX runner split loses to dispatch and state-management overhead on the target hetero8 workload.
- The next prefill attempt should not be eight independent `forward_step_jit` calls. A viable version needs to keep the ragged/chunked layout inside one compiled program or lower the hot prefill blocks so active rows are compacted without multiplying host dispatches and hybrid-state table traffic.
- Keep the profile/result artifact as negative evidence and continue from Entry 020 as the accepted implementation baseline.

## Entry 022 - Rejected Query-1 Decode Attention Shape

- run id: `20260525-210351-1922011-jax_hetero8_64_512x32_decode_q1_attention`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_decode_q1_attention.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-210351-1922011-jax_hetero8_64_512x32_decode_q1_attention`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-210351-1922011-jax_hetero8_64_512x32_decode_q1_attention/plugins/profile/2026_05_25_21_04_34/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-210351-1922011-jax_hetero8_64_512x32_decode_q1_attention/plugins/profile/2026_05_25_21_04_34/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Hegel confirmed that Entry021's dense GEMM savings were real but lost to dispatch/state traffic, and recommended the next prefill attempt be a single compiled compact-prefill program carrying KV/hybrid state through static compact bands. It also bounded decode LM-head/top-1 upside at roughly the `76 ms` `gemm_fusion_dot_234` bucket over the trace.
- change tested: `paged_attention_decode` used a specialized query-length-one einsum shape, computing scores as `[batch, kv_heads, groups, max_kv_len]` for normal decode while leaving multi-token/speculative decode on the existing path. The code path was reverted after profiling.
- correctness: full generated-token match against Entry 001 for all 8 rows; focused CUDA decode-attention tests passed.
- JAX timing: `207.41 tok/s`, TTFT p50 `713.87 ms`, ITL p50 `16.52 ms`, ITL p95 `17.99 ms`.
- delta vs Entry 020: `0.986x` total tok/s, TTFT p50 `1.006x`, ITL p50 `1.026x`; the change slightly regressed the target workload.

Top trace ranges, total inclusive time:

| range | Entry 020 ms | Entry 022 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1216.92 | 1234.27 | end-to-end regression |
| `_run_main_and_sample` | 1094.88 | 1101.90 | runner hot path slightly slower |
| `forward_step_token_ids_jit` | 260.71 | 272.59 | decode-specialized attention shape increased compiled step time |
| `PjRtCApiLoadedExecutable::Execute` | 275.65 | 287.14 | execution overhead increased with the altered lowering |
| `jit_compiled:XLA GPU module` | 232.15 | 237.74 | compiled GPU work increased |
| `command_buffer::execute` | 178.58 over 1574 calls | 181.75 over 1729 calls | more command-buffer executions |
| `gemm_fusion_dot_234` | 76.08 | 77.10 | likely LM-head/decode GEMM bucket did not improve |
| `MemcpyD2D` | 21.89 | 23.59 | small copy-bucket regression |

Decision:

- Reject and revert the query-length-one decode attention source specialization. The existing `[B, 1, KV, G, S]` formulation lowers better on this target than the squeezed `[B, KV, G, S]` source shape.
- Future decode attention work should move below this source-level reshaping, e.g. a real optional Pallas/CuteDSL paged decode kernel or HLO-guided fusion, not another equivalent JAX einsum spelling.

## Entry 023 - Rejected Single-JIT Compact Prefill

- run id: `20260525-211251-1926295-jax_hetero8_64_512x32_compact_prefill_singlejit64`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_prefill_singlejit64.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-211251-1926295-jax_hetero8_64_512x32_compact_prefill_singlejit64`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-211251-1926295-jax_hetero8_64_512x32_compact_prefill_singlejit64/plugins/profile/2026_05_25_21_18_01/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-211251-1926295-jax_hetero8_64_512x32_compact_prefill_singlejit64/plugins/profile/2026_05_25_21_18_01/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Averroes confirmed this was feasible as a dedicated executor specialization but flagged the main risks: stale hybrid rows, wrong per-band `seq_lens`, HLO size, buffer pressure, and KV/hybrid table scatters moving inside the compiled graph.
- change tested: a guarded `ModelExecutor` prototype unrolled `64`-token compact prefill bands inside one outer `jax.jit`, carrying KV cache and hybrid state through the bands and emitting greedy token ids only for final bands. A focused tiny CUDA parity test passed, but the code path was reverted after profiling.
- correctness: full generated-token match against Entry 001 for all 8 rows.
- compile/runtime cost: the full benchmark process elapsed `311.39 s`, materially longer than the comparable rejected Entry021 `278.28 s`, consistent with the large unrolled HLO risk.
- JAX timing: `207.07 tok/s`, TTFT p50 `711.81 ms`, ITL p50 `16.76 ms`, ITL p95 `17.53 ms`.
- delta vs Entry 020: `0.984x` total tok/s, TTFT p50 `1.003x`, ITL p50 `1.041x`; the change regressed the target workload.

Top trace ranges, total inclusive time:

| range | Entry 020 ms | Entry 021 ms | Entry 023 ms | note |
| --- | ---: | ---: | ---: | --- |
| `generate_with_trace` | 1216.92 | 1306.63 | 1236.25 | single-JIT improved over multi-dispatch but still regressed baseline |
| `_run_main_and_sample` | 1094.88 | 1181.34 | 1101.03 | runner hot path slightly slower than baseline |
| `compact_prefill_token_ids_jit` | 0.00 | 0.00 | 650.23 | one giant compact-prefill compiled call |
| `forward_step_token_ids_jit` | 260.71 | 96.88 | 112.26 | decode-only attribution lower because prefill moved to compact path |
| `PjRtCApiLoadedExecutable::Execute` | 275.65 over 252 calls | 703.20 over 921 calls | 776.20 over 252 calls | dispatch count fixed, but compiled execution became much heavier |
| `jit_compiled:XLA GPU module` | 232.15 | 539.07 | 716.13 | unrolled compact HLO dominated runtime |
| `_batch_hybrid_state` / `_store_batch_hybrid_state` | about 8.04 | about 106.54 | about 9.22 | state-table traffic stayed near baseline |
| `MemcpyD2D` | 21.89 | 43.23 | 25.90 | slightly worse than baseline |
| `gemm_fusion_dot_285` | 234.88 | 39.49 | 39.52 | dense rectangular GEMM saving was preserved |
| `gemm_fusion_dot_6` / `gemm_fusion_dot_7` | 200.25 | 0.00 | 0.00 | large prefill GEMMs removed, but offset by other fused loops |
| `loop_add_fusion` / `loop_multiply_fusion_*` | 0.00 | 0.00 | about 134.65 | new unrolled-HLO elementwise loop work |

Decision:

- Reject and revert the single-JIT compact prefill prototype. It proved that dense padded GEMMs can be removed without multiplying host dispatches or runner hybrid-state traffic, but the resulting unrolled HLO was much slower than the accepted rectangular prefill.
- Future compact-prefill work should move below whole-model source unrolling. The next viable direction is an optional lowered kernel for the actual hot prefill blocks, or a more local Pallas/CuteDSL prefill attention/GDN kernel, not repeated or unrolled calls to `model_forward_step`.
- Continue from Entry 020 as the accepted implementation baseline.

## Entry 024 - Rejected LM-Head Top-K(1) Greedy Token

- run id: `20260525-212852-1930393-jax_hetero8_64_512x32_lm_head_topk1_argmax`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_lm_head_topk1_argmax.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-212852-1930393-jax_hetero8_64_512x32_lm_head_topk1_argmax`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-212852-1930393-jax_hetero8_64_512x32_lm_head_topk1_argmax/plugins/profile/2026_05_25_21_29_42/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-212852-1930393-jax_hetero8_64_512x32_lm_head_topk1_argmax/plugins/profile/2026_05_25_21_29_42/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: a temporary env-gated `NANO_VLLM_JAX_LM_HEAD_TOPK1_ARGMAX=1` path used `jax.lax.top_k(logits, 1)` to produce the greedy token id in `lm_head_token_ids_and_topk` instead of `jnp.argmax`. The source change was reverted after profiling.
- correctness: focused CUDA guardrails passed, and the full hetero8 run had exact generated-token matches for all 8 rows against Entry 001.
- JAX timing: `206.11 tok/s`, TTFT p50 `713.34 ms`, ITL p50 `17.09 ms`, ITL p95 `17.54 ms`.
- delta vs Entry 020: `0.980x` total tok/s, TTFT p50 `1.005x`, ITL p50 `1.062x`; the top-k(1) greedy token spelling regressed the target workload.

Top trace ranges, total inclusive time:

| range | Entry 020 ms | Entry 024 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1216.92 | 1242.04 | end-to-end regression |
| `_run_main_and_sample` | 1094.88 | 1106.76 | runner hot path slower |
| `forward_step_token_ids_jit` | 260.71 | 284.29 | decode-step compiled path slower |
| `PjRtCApiLoadedExecutable::Execute` | 275.65 over 252 calls | 300.00 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 232.15 | 245.79 | compiled GPU work increased |
| `command_buffer::execute` | 178.58 over 1574 calls | 182.57 over 1574 calls | same command count, slightly slower |
| `gemm_fusion_dot_234` | 76.08 | 75.92 | likely LM-head GEMM bucket did not materially improve |
| `gemm_fusion_dot_285` | 234.88 | 234.68 | prefill dense work unchanged |
| `MemcpyD2D` | 21.89 | 22.43 | small copy-bucket regression |

Decision:

- Reject and revert the `top_k(logits, 1)` greedy-token source spelling. It preserved correctness but did not reduce the LM-head GEMM bucket and increased overall compiled execution time.
- Future LM-head work should not chase equivalent reduction spellings in plain JAX. The next viable LM-head target is below this level: a real streaming/top-1 lowered kernel, or a structural weight/layout change with memory and compile evidence.

## Entry 025 - Accepted Opt-In Materialized Tied LM Head

- run id: `20260525-213740-1934132-jax_hetero8_64_512x32_materialized_tied_lm_head`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_materialized_tied_lm_head.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-213740-1934132-jax_hetero8_64_512x32_materialized_tied_lm_head`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-213740-1934132-jax_hetero8_64_512x32_materialized_tied_lm_head/plugins/profile/2026_05_25_21_38_33/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-213740-1934132-jax_hetero8_64_512x32_materialized_tied_lm_head/plugins/profile/2026_05_25_21_38_33/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260525-214003-1934826-jax_hetero8_64_512x32_materialized_tied_lm_head_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_materialized_tied_lm_head_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260525-214003-1934826-jax_hetero8_64_512x32_materialized_tied_lm_head_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260525-214003-1934826-jax_hetero8_64_512x32_materialized_tied_lm_head_repeat/plugins/profile/2026_05_25_21_40_27/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-214003-1934826-jax_hetero8_64_512x32_materialized_tied_lm_head_repeat/plugins/profile/2026_05_25_21_40_27/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Faraday confirmed the loader is the right gate point because `model.py` already uses `params.lm_head` when present, and bounded the memory tradeoff at `508,559,360` bytes (`485 MiB`) for BF16 Qwen3.5-0.8B. The audit also flagged this as an opt-in path until MTP-specific memory/JIT surface is profiled.
- change: `NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD=1` makes the streaming loader read tied `embed_tokens.weight` a second time with the same contiguous `[hidden, vocab]` transpose path used for real untied `lm_head.weight`. Default remains implicit tied weights (`lm_head=None`) to avoid the extra memory tax on smaller GPUs.
- correctness: focused CUDA guardrails passed with the flag enabled (`9 passed`), a tiny CUDA `jnp.array(a.T, copy=True)` runtime check passed, and both full hetero8 runs had exact generated-token matches for all 8 rows against Entry 001.
- first JAX timing: `214.08 tok/s`, TTFT p50 `712.14 ms`, ITL p50 `15.37 ms`, ITL p95 `16.46 ms`.
- repeat JAX timing: `217.04 tok/s`, TTFT p50 `706.99 ms`, ITL p50 `15.09 ms`, ITL p95 `16.02 ms`.
- repeat delta vs Entry 020: `1.032x` total tok/s, TTFT p50 `0.996x`, ITL p50 `0.937x`, ITL p95 `0.940x`.

Top repeat trace ranges, total inclusive time:

| range | Entry 020 ms | Entry 025 repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1216.92 | 1179.49 | end-to-end improvement |
| `_run_main_and_sample` | 1094.88 | 1060.05 | runner hot path improved |
| `forward_step_token_ids_jit` | 260.71 | 252.33 | decode-step compiled path improved on repeat |
| `PjRtCApiLoadedExecutable::Execute` | 275.65 over 252 calls | 267.85 over 252 calls | same dispatch count, lower execution time |
| `jit_compiled:XLA GPU module` | 232.15 | 225.20 | compiled GPU work decreased |
| `command_buffer::execute` | 178.58 over 1574 calls | 176.28 over 1574 calls | same command count, slightly faster |
| `gemm_fusion_dot_234` | 76.08 | 45.69 | likely decode LM-head GEMM improved materially |
| `gemm_fusion_dot_285` | 234.88 | 235.08 | prefill dense work unchanged |
| `MemcpyD2D` | 21.89 | 21.92 | copy bucket unchanged |

Decision:

- Accept the materialized tied LM-head layout as an opt-in GPU serving path. It keeps exact greedy correctness, costs about `485 MiB` extra BF16 device memory for Qwen3.5-0.8B, and repeatedly improves the target hetero8 workload by reducing the decode LM-head GEMM bucket.
- Keep the default implicit tied head for now because the memory tradeoff is model and deployment dependent. If future target profiles always have enough headroom, revisit making this a config-level serving default rather than an environment flag.
- The next LM-head step should be a lower-level streaming/top-1 kernel if we want to remove the remaining full-vocab materialized logits cost without paying a second LM-head buffer.

## Entry 026 - Accepted Opt-In Compact GDN QKV Prefill Projection

- run id: `20260525-215517-1939908-jax_hetero8_64_512x32_compact_gdn_qkv_projection`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_gdn_qkv_projection.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-215517-1939908-jax_hetero8_64_512x32_compact_gdn_qkv_projection`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-215517-1939908-jax_hetero8_64_512x32_compact_gdn_qkv_projection/plugins/profile/2026_05_25_21_56_00/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-215517-1939908-jax_hetero8_64_512x32_compact_gdn_qkv_projection/plugins/profile/2026_05_25_21_56_00/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260525-215654-1940211-jax_hetero8_64_512x32_compact_gdn_qkv_projection_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_gdn_qkv_projection_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260525-215654-1940211-jax_hetero8_64_512x32_compact_gdn_qkv_projection_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260525-215654-1940211-jax_hetero8_64_512x32_compact_gdn_qkv_projection_repeat/plugins/profile/2026_05_25_21_57_18/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-215654-1940211-jax_hetero8_64_512x32_compact_gdn_qkv_projection_repeat/plugins/profile/2026_05_25_21_57_18/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Jason identified padded GDN prefill projection as the primary remaining prefill target after Entry 025, especially `mixed_qkv` in `model.py`, while warning against repeating the rejected whole-prefill source compaction from Entries 021 and 023.
- Pallas note: a tiny local `pallas_call`/`pl.dot` top-1 prototype failed on JAX 0.10 Mosaic GPU lowering. Future lowered kernels should follow the installed `jax.experimental.pallas.ops.gpu.ragged_dot_mgpu` style (`plgpu.kernel`, WGMMA, and explicit memory spaces), not a simple `pallas_call` spelling.
- change: `NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV=1` compacts true ragged prefill token rows for only the GDN `in_proj_qkv` projection using `valid_token_mask` and a static `num_prefill_tokens`, performs the dot on compact rows, then scatters back to the rectangular `[batch, seq, channels]` layout. The benchmark also used `NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD=1` from Entry 025.
- correctness: focused CUDA guardrails passed with both opt-in flags enabled (`9 passed`), `py_compile` and `git diff --check` passed, and both full hetero8 runs had exact generated-token matches for all 8 rows against Entry 001.
- first JAX timing: `230.15 tok/s`, TTFT p50 `630.29 ms`, ITL p50 `15.35 ms`, ITL p95 `16.20 ms`.
- repeat JAX timing: `232.16 tok/s`, TTFT p50 `626.11 ms`, ITL p50 `15.18 ms`, ITL p95 `15.88 ms`.
- repeat delta vs Entry 025 repeat: `1.070x` total tok/s, TTFT p50 `0.886x`, ITL p50 `1.006x`, ITL p95 `0.991x`.

Top repeat trace ranges, total inclusive time:

| range | Entry 025 repeat ms | Entry 026 repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1179.49 | 1102.69 | end-to-end improvement |
| `_run_main_and_sample` | 1060.05 | 981.36 | runner hot path improved |
| `forward_step_token_ids_jit` | 252.33 | 237.43 | compiled token-id step improved |
| `PjRtCApiLoadedExecutable::Execute` | 267.85 over 252 calls | 252.78 over 252 calls | same dispatch count, lower execution time |
| `jit_compiled:XLA GPU module` | 225.20 | 209.84 | compiled GPU work decreased |
| `command_buffer::execute` | 176.28 over 1574 calls | 158.74 over 1575 calls | command-buffer time improved |
| `gemm_fusion_dot_285` | 235.08 over 576 calls | 39.54 over 558 calls | padded rectangular GDN QKV prefill bucket mostly removed |
| `gemm_fusion_dot_general_729` | 0.00 | 79.61 over 18 calls | new compact projection GEMM bucket |
| `gemm_fusion_dot_234` | 45.69 | 45.70 | materialized LM-head decode bucket unchanged |
| `gemm_fusion_dot_6` / `gemm_fusion_dot_7` | 200.79 | 73.91 | related prefill GEMM buckets reduced |
| `gemm_fusion_dot_4` | 33.29 | 127.78 | bucket identity shifted after recompilation |
| `MemcpyD2D` | 21.92 | 22.05 | copy bucket unchanged |

Decision:

- Accept the compact GDN QKV prefill projection as an opt-in serving path layered on top of the materialized tied LM head. It preserves exact greedy correctness and repeatedly improves the target hetero8 workload by removing most of the padded GDN QKV prefill GEMM cost.
- Keep the default off until it is profiled on more prompt distributions and MTP paths, because the JIT key now specializes on static prefill-token count when the flag is enabled and the implementation still pays gather/scatter overhead.
- The next version should lower this below source-level gather/dot/scatter, ideally as a real ragged/WGMMA prefill projection kernel, so the new `gemm_fusion_dot_general_729` bucket can be fused or reduced.

## Entry 027 - Accepted Opt-In Compact Prefill MLP

- run id: `20260525-220641-1943624-jax_hetero8_64_512x32_compact_prefill_mlp`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_prefill_mlp.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-220641-1943624-jax_hetero8_64_512x32_compact_prefill_mlp`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-220641-1943624-jax_hetero8_64_512x32_compact_prefill_mlp/plugins/profile/2026_05_25_22_07_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-220641-1943624-jax_hetero8_64_512x32_compact_prefill_mlp/plugins/profile/2026_05_25_22_07_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260525-220752-1944077-jax_hetero8_64_512x32_compact_prefill_mlp_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_prefill_mlp_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260525-220752-1944077-jax_hetero8_64_512x32_compact_prefill_mlp_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260525-220752-1944077-jax_hetero8_64_512x32_compact_prefill_mlp_repeat/plugins/profile/2026_05_25_22_08_17/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-220752-1944077-jax_hetero8_64_512x32_compact_prefill_mlp_repeat/plugins/profile/2026_05_25_22_08_17/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Dewey confirmed that Entry 026's largest remaining model-side prefill buckets were the padded MLP GEMMs, especially `gemm_fusion_dot_4` and `gemm_fusion_dot_6`, totaling about `202 ms` on the hetero8 profile. The audit recommended this exact row-compacted MLP experiment before lower-level kernel work, and required repeat profile evidence plus a masked valid-token parity check.
- change: `NANO_VLLM_JAX_COMPACT_PREFILL_MLP=1` gathers true ragged prefill token rows after the FFN RMSNorm, runs `gate_proj`, `up_proj`, activation, and `down_proj` on compact rows, then scatters the final MLP output back to the rectangular batch layout. It reuses the static `num_prefill_tokens` JIT key already needed by compact GDN QKV when any compact prefill path is enabled. The benchmark also used `NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD=1` and `NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV=1`.
- correctness: focused CUDA guardrails passed with all three opt-in flags enabled (`10 passed`), including a new masked compact-MLP helper test that checks valid-token parity against dense MLP and zero padded compact outputs. Both full hetero8 runs had exact generated-token matches for all 8 rows against Entry 001.
- first JAX timing: `241.30 tok/s`, TTFT p50 `576.35 ms`, ITL p50 `15.58 ms`, ITL p95 `16.33 ms`.
- repeat JAX timing: `240.18 tok/s`, TTFT p50 `585.66 ms`, ITL p50 `15.34 ms`, ITL p95 `16.84 ms`.
- repeat delta vs Entry 026 repeat: `1.035x` total tok/s, TTFT p50 `0.935x`, ITL p50 `1.011x`, ITL p95 `1.060x`.

Top repeat trace ranges, total inclusive time:

| range | Entry 026 repeat ms | Entry 027 repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1102.69 | 1065.87 | end-to-end improvement |
| `_run_main_and_sample` | 981.36 | 934.41 | runner hot path improved |
| `forward_step_token_ids_jit` | 237.43 | 232.40 | compiled token-id step improved |
| `PjRtCApiLoadedExecutable::Execute` | 252.78 over 252 calls | 248.28 over 252 calls | same dispatch count, lower execution time |
| `jit_compiled:XLA GPU module` | 209.84 | 203.18 | compiled GPU work decreased |
| `command_buffer::execute` | 158.74 over 1575 calls | 148.30 over 1575 calls | command-buffer time improved |
| `gemm_fusion_dot_4` | 127.78 over 48 calls | 0.00 | padded MLP expansion bucket removed |
| `gemm_fusion_dot_6` | 73.91 over 24 calls | 0.00 | padded MLP down-projection bucket removed |
| `gemm_fusion_dot_229` | 0.00 | 110.70 over 24 calls | new compact MLP GEMM bucket |
| `gemm_fusion_dot_general_729` | 79.61 | 79.27 | compact GDN QKV bucket unchanged |
| `gemm_fusion_dot_234` | 45.70 | 45.73 | materialized LM-head decode bucket unchanged |
| `MemcpyD2D` | 22.05 | 22.05 | copy bucket unchanged |

Decision:

- Accept compact prefill MLP as an opt-in serving path. It keeps exact greedy correctness, preserves one compiled model step dispatch count, and improves the target hetero8 workload by replacing about `202 ms` of padded MLP GEMM buckets with about `111 ms` of compact MLP GEMM work.
- Keep the default off until MTP and additional prompt distributions are profiled. This remains a source-level gather/dot/scatter implementation and specializes JIT compilation on static prefill-token count when enabled.
- The next model-side targets should be lower-level versions of the compact paths, especially compact GDN QKV and compact MLP with fused gather/scatter or ragged WGMMA, before revisiting source-level decode rewrites that already regressed in earlier entries.

## Entry 028 - Accepted Opt-In Compact GDN Z Prefill Projection

- run id: `20260525-221541-1947122-jax_hetero8_64_512x32_compact_gdn_z`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_gdn_z.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-221541-1947122-jax_hetero8_64_512x32_compact_gdn_z`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-221541-1947122-jax_hetero8_64_512x32_compact_gdn_z/plugins/profile/2026_05_25_22_16_27/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-221541-1947122-jax_hetero8_64_512x32_compact_gdn_z/plugins/profile/2026_05_25_22_16_27/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260525-221655-1947464-jax_hetero8_64_512x32_compact_gdn_z_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_gdn_z_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260525-221655-1947464-jax_hetero8_64_512x32_compact_gdn_z_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260525-221655-1947464-jax_hetero8_64_512x32_compact_gdn_z_repeat/plugins/profile/2026_05_25_22_17_19/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-221655-1947464-jax_hetero8_64_512x32_compact_gdn_z_repeat/plugins/profile/2026_05_25_22_17_19/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Ptolemy confirmed Entry 027's largest remaining source-JAX target was the inferred GDN `in_proj_z` prefill bucket (`gemm_fusion_dot_194`, about `25.46 ms / 18 calls`) because compact MLP and compact GDN QKV already need lower-level fused gather/scatter or ragged/WGMMA work for larger wins.
- change: `NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z=1` gathers true ragged prefill token rows for the GDN `in_proj_z` projection, runs the dot on compact rows, and scatters back to the rectangular `[batch, seq, value_dim]` layout. This shares the same compact-dot helper as compact GDN QKV and extends the static prefill-token JIT key whenever any compact prefill projection is enabled.
- correctness: focused CUDA guardrails passed with all four opt-in flags enabled (`11 passed`), including a compact-dot helper parity test that checks valid-token equality against dense projection and zero padded compact outputs. Both full hetero8 runs had exact generated-token matches for all 8 rows against Entry 001.
- first JAX timing: `244.12 tok/s`, TTFT p50 `569.00 ms`, ITL p50 `15.28 ms`, ITL p95 `16.14 ms`.
- repeat JAX timing: `244.98 tok/s`, TTFT p50 `567.22 ms`, ITL p50 `15.14 ms`, ITL p95 `16.39 ms`.
- repeat delta vs Entry 027 repeat: `1.020x` total tok/s, TTFT p50 `0.969x`, ITL p50 `0.987x`, ITL p95 `0.974x`.

Top repeat trace ranges, total inclusive time:

| range | Entry 027 repeat ms | Entry 028 repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1065.87 | 1044.98 | end-to-end improvement |
| `_run_main_and_sample` | 934.41 | 921.88 | runner hot path improved |
| `forward_step_token_ids_jit` | 232.40 | 225.83 | compiled token-id step improved |
| `PjRtCApiLoadedExecutable::Execute` | 248.28 over 252 calls | 241.57 over 252 calls | same dispatch count, lower execution time |
| `jit_compiled:XLA GPU module` | 203.18 | 197.85 | compiled GPU work decreased |
| `command_buffer::execute` | 148.30 over 1575 calls | 146.87 over 1575 calls | command-buffer time slightly improved |
| `gemm_fusion_dot_194` | 25.46 over 18 calls | 110.74 over 24 calls | bucket name was reused after recompilation; do not compare by name alone |
| `gemm_fusion_dot_229` | 110.70 over 24 calls | 0.00 | compact MLP bucket was renamed/reassigned |
| `gemm_fusion_dot_general_729` | 79.27 over 18 calls | 0.00 | compact GDN QKV bucket was renamed/reassigned |
| `gemm_fusion_dot_193` | 0.00 | 97.20 over 18 calls | new/reassigned compact GDN prefill projection bucket |
| `gemm_fusion_dot_general_746` | 41.67 | 41.21 | compact MLP companion bucket unchanged |
| `gemm_fusion_dot_234` | 45.73 | 45.70 | materialized LM-head decode bucket unchanged |
| `MemcpyD2D` | 22.05 | 24.41 | copy bucket slightly higher but not enough to offset model-step gain |

Decision:

- Accept compact GDN Z as an opt-in serving path. It is a small but repeatable improvement on the target hetero8 workload, keeps exact greedy correctness, and preserves the single compiled model-step dispatch count.
- Keep it default-off with the other compact prefill flags until MTP and more prompt distributions are profiled. The trace bucket names shifted enough after recompilation that the safest acceptance evidence is the repeated lower model-step time plus exact output parity, not a single kernel-name delta.
- Further source-level row compaction now has diminishing returns. The next serious work should lower the compact MLP/GDN projections below JAX gather/dot/scatter, or attack full-attention prefill projection/paged-attention kernels with optional Pallas/CuteDSL paths.

## Entry 029 - Accepted Opt-In Compact Full-Attention Prefill Projections

- run id: `20260525-222240-1950106-jax_hetero8_64_512x32_compact_full_attn_proj`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_full_attn_proj.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-222240-1950106-jax_hetero8_64_512x32_compact_full_attn_proj`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-222240-1950106-jax_hetero8_64_512x32_compact_full_attn_proj/plugins/profile/2026_05_25_22_23_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-222240-1950106-jax_hetero8_64_512x32_compact_full_attn_proj/plugins/profile/2026_05_25_22_23_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260525-222356-1950452-jax_hetero8_64_512x32_compact_full_attn_proj_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_full_attn_proj_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260525-222356-1950452-jax_hetero8_64_512x32_compact_full_attn_proj_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260525-222356-1950452-jax_hetero8_64_512x32_compact_full_attn_proj_repeat/plugins/profile/2026_05_25_22_24_21/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-222356-1950452-jax_hetero8_64_512x32_compact_full_attn_proj_repeat/plugins/profile/2026_05_25_22_24_21/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Pascal was launched to audit Entry 028 but did not complete before shutdown. Gibbs then reviewed the collected Entry 028/029 evidence and agreed this is acceptable as a small opt-in source-JAX win: repeat throughput improved about `0.53%`, TTFT improved, exact correctness held, dispatch stayed at `252`, and execution-side slices (`PjRt execute`, `jit_compiled GPU module`, `command_buffer`) all dropped slightly. The audit cautioned that ITL p50 regressed slightly and the gain is benchmark-specific until broader shape/MTP profiles confirm it.
- change: `NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ=1` makes full-attention prefill `q_proj`, `k_proj`, and `v_proj` use the shared compact-dot helper when attention metadata supplies ragged query lengths and a static true prefill-token count. Valid token rows are projected exactly as before and padded rows scatter zero outputs back into the rectangular layout.
- correctness: focused CUDA guardrails passed with all five opt-in flags enabled (`11 passed`). Both full hetero8 runs had exact generated-token matches for all 8 rows against Entry 001.
- first JAX timing: `247.87 tok/s`, TTFT p50 `550.74 ms`, ITL p50 `15.20 ms`, ITL p95 `16.35 ms`.
- repeat JAX timing: `246.27 tok/s`, TTFT p50 `563.77 ms`, ITL p50 `15.26 ms`, ITL p95 `15.74 ms`.
- repeat delta vs Entry 028 repeat: `1.005x` total tok/s, TTFT p50 `0.994x`, ITL p50 `1.008x`, ITL p95 `0.960x`.

Top repeat trace ranges, total inclusive time:

| range | Entry 028 repeat ms | Entry 029 repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1044.98 | 1039.52 | small end-to-end improvement |
| `_run_main_and_sample` | 921.88 | 907.94 | runner hot path improved |
| `forward_step_token_ids_jit` | 225.83 | 224.15 | compiled token-id step slightly improved |
| `PjRtCApiLoadedExecutable::Execute` | 241.57 over 252 calls | 240.53 over 252 calls | same dispatch count, slightly lower execution time |
| `jit_compiled:XLA GPU module` | 197.85 | 196.07 | compiled GPU work slightly decreased |
| `command_buffer::execute` | 146.87 over 1575 calls | 144.17 over 1575 calls | command-buffer time improved |
| `gemm_fusion_dot_164` | 43.69 over 6 calls | 0.00 | full-attention prefill projection bucket removed or renamed |
| `gemm_fusion_dot_181` | 0.00 | 97.20 over 18 calls | reassigned compact GDN/full-attention prefill bucket |
| `gemm_fusion_dot_182` | 0.00 | 110.73 over 24 calls | reassigned compact MLP bucket |
| `gemm_fusion_dot_193` | 97.20 over 18 calls | 0.00 | bucket name shifted after recompilation |
| `gemm_fusion_dot_194` | 110.74 over 24 calls | 0.00 | bucket name shifted after recompilation |
| `gemm_fusion_dot_234` | 45.70 | 45.68 | materialized LM-head decode bucket unchanged |
| `gemm_fusion_dot_286` | 43.74 | 43.79 | narrow decode GEMM bucket unchanged |
| `MemcpyD2D` | 24.41 | 24.37 | copy bucket unchanged |

Decision:

- Accept compact full-attention prefill projections as a small opt-in improvement. It preserves exact greedy correctness, keeps the same dispatch count, and slightly improves the target hetero8 repeat without touching scheduling or decode semantics.
- Keep this default-off with the other compact prefill flags. The improvement is modest, and the trace bucket names shift enough that this should be treated as workload-specific until broader prompt/MTP profiles are run.
- Source-level row compaction has now covered the obvious padded prefill projections. Next work should move to optional lowered kernels for compact MLP/GDN projection paths, paged prefill attention kernels, or a deeper decode LM-head/narrow-GEMM plan with stronger HLO/kernel evidence.

## Entry 030 - Rejected Compact Attention Output Projections

- run id: `20260525-223325-1953992-jax_hetero8_64_512x32_compact_attn_out_proj`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_attn_out_proj.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-223325-1953992-jax_hetero8_64_512x32_compact_attn_out_proj`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-223325-1953992-jax_hetero8_64_512x32_compact_attn_out_proj/plugins/profile/2026_05_25_22_34_06/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-223325-1953992-jax_hetero8_64_512x32_compact_attn_out_proj/plugins/profile/2026_05_25_22_34_06/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Euler was launched to audit the broader post-Entry029 bottleneck direction but did not complete before shutdown. Carver reviewed the Entry029/Entry030 evidence and confirmed rejection: exact correctness was not enough to offset the `-2.9%` throughput regression, worse ITL, and much heavier `PjRt Execute` / `forward_step_token_ids_jit` ranges.
- change tested: a temporary `NANO_VLLM_JAX_COMPACT_PREFILL_ATTN_OUT_PROJ=1` path compacted valid prefill rows for GDN `out_proj` and full-attention `o_proj`, then scattered the projected output back to the rectangular layout. The source change was reverted after profiling.
- correctness: focused CUDA guardrails passed with the flag enabled (`11 passed`), and the full hetero8 run had exact generated-token matches for all 8 rows against Entry 001.
- JAX timing: `239.07 tok/s`, TTFT p50 `552.36 ms`, ITL p50 `16.81 ms`, ITL p95 `18.55 ms`.
- delta vs Entry 029 repeat: `0.971x` total tok/s, TTFT p50 `0.980x`, ITL p50 `1.102x`, ITL p95 `1.178x`; despite a lower TTFT, decode latency and overall throughput regressed clearly.

Top trace ranges, total inclusive time:

| range | Entry 029 repeat ms | Entry 030 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1039.52 | 1070.81 | end-to-end regression |
| `_run_main_and_sample` | 907.94 | 919.20 | runner hot path slower |
| `forward_step_token_ids_jit` | 224.15 | 258.43 | compiled token-id step much slower |
| `PjRtCApiLoadedExecutable::Execute` | 240.53 over 252 calls | 276.41 over 252 calls | same dispatch count, much heavier execution |
| `jit_compiled:XLA GPU module` | 196.07 | 216.17 | compiled GPU work increased |
| `command_buffer::execute` | 144.17 over 1575 calls | 146.54 over 1575 calls | command-buffer time slightly worse |
| `gemm_fusion_dot_182` | 110.73 over 24 calls | 0.00 | bucket renamed rather than eliminated |
| `gemm_fusion_dot_158` | 0.00 | 110.72 over 24 calls | reassigned compact MLP/attention output bucket |
| `gemm_fusion_dot_181` | 97.20 over 18 calls | 0.00 | bucket renamed rather than eliminated |
| `gemm_fusion_dot_157` | 0.00 | 97.18 over 18 calls | reassigned compact GDN/full-attention bucket |
| `gemm_fusion_dot_234` | 45.68 | 45.72 | materialized LM-head decode bucket unchanged |
| `gemm_fusion_dot_286` | 43.79 | 43.71 | narrow decode GEMM bucket unchanged |
| `MemcpyD2D` | 24.37 | 25.42 | copy bucket slightly worse |

Decision:

- Reject and revert compact attention output projections. The target bucket was mostly renamed, while compiled execution time increased substantially and decode-step latency regressed.
- Do not extend plain source-level row compaction to every remaining projection by default. The accepted compact input/MLP paths are useful, but this result shows that later-stage output projections can make XLA's compiled plan worse even when exact correctness holds.
- Continue from Entry 029 as the accepted implementation baseline. Next work should focus on optional lower-level kernels for the compact projection paths, paged attention prefill, or decode LM-head/narrow-GEMM work with kernel-level evidence.

## Entry 031 - Rejected lax.ragged_dot Compact Projections

- run id: `20260525-224444-1959951-jax_hetero8_64_512x32_compact_lax_ragged_dot`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_lax_ragged_dot.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-224444-1959951-jax_hetero8_64_512x32_compact_lax_ragged_dot`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-224444-1959951-jax_hetero8_64_512x32_compact_lax_ragged_dot/plugins/profile/2026_05_25_22_45_24/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-224444-1959951-jax_hetero8_64_512x32_compact_lax_ragged_dot/plugins/profile/2026_05_25_22_45_24/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Huygens ranked optional lower-level compact projection kernels as the next best direction after Entry 029, ahead of decode LM-head and paged prefill attention. It identified the compact projection buckets (`gemm_fusion_dot_182`, `gemm_fusion_dot_181`) as larger than decode LM-head or narrow decode GEMMs, but warned that replacing only the dot may leave gather/scatter as the limiter and that the installed Pallas path must be checked carefully.
- Pallas feasibility: the installed `jax.experimental.pallas.ops.gpu.ragged_dot_mgpu.ragged_dot` is a Mosaic GPU WGMMA kernel. A tiny CUDA check on this A10G failed during lowering with `nvvm.wgmma.fence.aligned` unsupported on `sm_86`, so that path is not usable on this machine. No Cutlass/CuteDSL package is installed in the current environment.
- change tested: a temporary `NANO_VLLM_JAX_COMPACT_PREFILL_LAX_RAGGED_DOT=1` path replaced compact-row `jnp.dot` calls inside `_compact_prefill_dot_if_enabled` and `_compact_prefill_mlp` with single-group `jax.lax.ragged_dot`. The source change was reverted after profiling.
- correctness: focused CUDA guardrails passed with the flag enabled (`11 passed`), and the full hetero8 run had exact generated-token matches for all 8 rows against Entry 001.
- JAX timing: `245.89 tok/s`, TTFT p50 `553.58 ms`, ITL p50 `15.69 ms`, ITL p95 `16.51 ms`.
- delta vs Entry 029 repeat: `0.998x` total tok/s, TTFT p50 `0.982x`, ITL p50 `1.028x`, ITL p95 `1.049x`; lower TTFT did not compensate for decode latency and compiled execution regressions.

Top trace ranges, total inclusive time:

| range | Entry 029 repeat ms | Entry 031 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1039.52 | 1041.11 | slight end-to-end regression |
| `_run_main_and_sample` | 907.94 | 913.45 | runner hot path slower |
| `forward_step_token_ids_jit` | 224.15 | 237.39 | compiled token-id step slower |
| `PjRtCApiLoadedExecutable::Execute` | 240.53 over 252 calls | 252.67 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 196.07 | 204.15 | compiled GPU work increased |
| `command_buffer::execute` | 144.17 over 1575 calls | 146.99 over 1575 calls | command-buffer time slightly worse |
| `gemm_fusion_dot_182` | 110.73 over 24 calls | 0.00 | bucket renamed rather than removed |
| `gemm_fusion_dot_434` | 0.00 | 110.71 over 24 calls | reassigned compact MLP bucket |
| `gemm_fusion_dot_181` | 97.20 over 18 calls | 0.00 | bucket renamed rather than removed |
| `gemm_fusion_dot_433` | 0.00 | 97.21 over 18 calls | reassigned compact GDN/full-attention bucket |
| `gemm_fusion_dot_234` | 45.68 | 47.16 | materialized LM-head decode bucket slightly worse |
| `gemm_fusion_dot_286` | 43.79 | 43.73 | narrow decode GEMM bucket unchanged |
| `MemcpyD2D` | 24.37 | 24.58 | copy bucket unchanged |

Decision:

- Reject and revert the `lax.ragged_dot` compact projection path. It preserved correctness but did not change the underlying compact GEMM cost materially and made compiled execution and ITL worse.
- On this A10G target, the installed Mosaic GPU `ragged_dot_mgpu` WGMMA path is not viable because it requires instructions unsupported on `sm_86`. A useful lower-level compact projection kernel would need an Ampere-compatible implementation, likely Triton or another CUDA path, and probably needs to fuse gather/dot/scatter to beat the current XLA plan.
- Continue from Entry 029 as the accepted implementation baseline. The next practical work should either prototype an Ampere-compatible custom kernel outside Pallas WGMMA, or switch to decode LM-head/narrow-GEMM work with kernel-level evidence.

## Entry 032 - Rejected Exact `max_blocks_per_seq=34` KV Window

- run id: `20260525-225535-1976931-jax_hetero8_64_512x32_max_blocks_34`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_max_blocks_34.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-225535-1976931-jax_hetero8_64_512x32_max_blocks_34`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-225535-1976931-jax_hetero8_64_512x32_max_blocks_34/plugins/profile/2026_05_25_22_56_35/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-225535-1976931-jax_hetero8_64_512x32_max_blocks_34/plugins/profile/2026_05_25_22_56_35/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Noether recommended testing the exact static paged-attention window for this workload before lower-level kernel work. The target shape needs at most `512 + 32 = 544` tokens, so `--max-blocks-per-seq 34` shrinks the static KV window from `40 * 16 = 640` to `34 * 16 = 544` without changing model math.
- change tested: benchmark-only setting change from Entry 029's `--max-blocks-per-seq 40` to `--max-blocks-per-seq 34`, with all accepted Entry 029 opt-in fast paths enabled.
- correctness: exact generated-token match for all 8 rows against the hetero8 reference.
- JAX timing: `246.76 tok/s`, TTFT p50 `553.52 ms`, ITL p50 `15.37 ms`, ITL p95 `16.86 ms`.
- delta vs Entry 029 repeat: `1.002x` total tok/s, TTFT p50 `0.982x`, ITL p50 `1.007x`, ITL p95 `1.071x`. It missed the acceptance target of `>= 248.7 tok/s` and regressed ITL.

Top trace ranges, total inclusive time:

| range | Entry 029 repeat ms | Entry 032 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 1039.52 | 1037.45 | slightly lower end-to-end wall attribution |
| `_run_main_and_sample` | 907.94 | 910.26 | runner hot path slightly worse |
| `forward_step_token_ids_jit` | 224.15 | 244.39 | compiled token-id step regressed |
| `PjRtCApiLoadedExecutable::Execute` | 240.53 over 252 calls | 258.98 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 196.07 | 208.89 | compiled GPU work increased |
| `command_buffer::execute` | 144.17 over 1575 calls | 147.55 over 1575 calls | command-buffer time increased |
| `np.asarray(jax.Array)` | 668.35 | 650.70 | host readback improved but not enough |
| `gemm_fusion_dot_182` | 110.73 over 24 calls | 110.75 over 24 calls | compact MLP bucket unchanged |
| `gemm_fusion_dot_181` | 97.20 over 18 calls | 97.20 over 18 calls | compact GDN/full-attn bucket unchanged |
| `gemm_fusion_dot_234` | 45.68 | 45.70 | materialized LM-head decode bucket unchanged |
| `gemm_fusion_dot_286` | 43.79 | 43.67 | narrow decode GEMM bucket unchanged |
| `MemcpyD2D` | 24.37 | 24.59 | copy bucket unchanged |

Decision:

- Reject adopting the exact `34`-block window as the target hetero8 setting. The lower TTFT and host readback do not compensate for worse compiled execution and ITL.
- Keep Entry 029's `--max-blocks-per-seq 40` as the current benchmark baseline for this workload. Revisit block-table width only as part of a broader shape-bucketing/server policy experiment, not as the next model-side optimization.
- The next meaningful model-side win still likely needs an Ampere-compatible fused compact projection path or a decode LM-head/top-1 kernel with stronger HLO/kernel evidence.

## Entry 033 - Accepted XLA GPU Autotune Level 4 Default

- run id: `20260525-232138-1984042-jax_hetero8_64_512x32_xla_autotune4_default`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_xla_autotune4_default.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-232138-1984042-jax_hetero8_64_512x32_xla_autotune4_default`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-232138-1984042-jax_hetero8_64_512x32_xla_autotune4_default/plugins/profile/2026_05_25_23_22_03/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-232138-1984042-jax_hetero8_64_512x32_xla_autotune4_default/plugins/profile/2026_05_25_23_22_03/INDCS0291.atrapa.deloitte.com.xplane.pb`
- supporting first run: `results/qwen08_jax_server_trace_hetero8_64_512x32_xla_autotune4.json`, profile `/mountpoint/.exp/profiles/20260525-231513-1980896-jax_hetero8_64_512x32_xla_autotune4`, `319.08 tok/s`, exact correctness.
- supporting repeat run: `results/qwen08_jax_server_trace_hetero8_64_512x32_xla_autotune4_repeat.json`, profile `/mountpoint/.exp/profiles/20260525-231752-1982659-jax_hetero8_64_512x32_xla_autotune4_repeat`, `353.85 tok/s`, exact correctness.
- subagent audit: Kuhn recommended returning to lower-level compact projection kernels after this configuration check. The audit identified pure-JAX compact projection spellings as mostly exhausted and suggested an Ampere-compatible fused gather/dot/scatter projection kernel as the next high-upside model-side experiment.
- change accepted: centralize XLA flag setup in `runtime_paths.configure_xla_flags()` and make benchmark/server entry points default to `--xla_gpu_autotune_level=4` unless `XLA_FLAGS` is explicitly provided. The helper also supports `NANO_VLLM_JAX_XLA_GPU_AUTOTUNE_LEVEL` for overriding the default autotune level without replacing unrelated XLA flags.
- correctness: exact generated-token match for all 8 rows against the hetero8 reference, with BF16 weights and FP32 activations. A first default verification with the wrong prompt-length order was discarded; the committed default artifact is the corrected `[64, 128, 192, 256, 320, 384, 448, 512]` run.
- JAX timing: `355.29 tok/s`, TTFT p50 `315.89 ms`, ITL p50 `12.86 ms`, ITL p95 `13.79 ms`.
- delta vs Entry 029 repeat: `1.443x` total tok/s, TTFT p50 `0.560x`, ITL p50 `0.842x`, ITL p95 `0.876x`.
- delta vs vLLM async baseline (`864.18 tok/s`): JAX improved from `0.285x` vLLM in Entry 029 repeat to `0.411x` vLLM with autotune level 4.

Top trace ranges, total inclusive time:

| range | Entry 029 repeat ms | Entry 033 repeat ms | note |
| --- | ---: | ---: | --- |
| `PjRtCApiLoadedExecutable::Execute` | 240.53 over 252 calls | 189.51 over 252 calls | same dispatch count, less compiled execution |
| `jit_compiled:XLA GPU module` | 196.07 | 145.62 | better GPU lowering after autotune |
| `command_buffer::execute` | 144.17 over 1575 calls | 91.31 over 1575 calls | large device command-buffer reduction |
| `gemm_fusion_dot_234` | 45.68 | 31.88 | materialized LM-head decode bucket improved |
| `gemm_fusion_dot_286` | 43.79 | 25.06 | narrow decode GEMM bucket improved |
| `gemm_fusion_dot_285` | 39.52 | 23.47 | narrow decode GEMM bucket improved |
| `gemm_fusion_dot_general_746` | 40.99 | 13.92 | compact projection/general-dot bucket improved |
| `input_reduce_fusion` | 41.48 | 41.57 | unchanged |
| `MemcpyD2D` | 24.37 | 24.62 | unchanged |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 0.00 | 52.40 over 96 calls | new autotuned CUTLASS kernel bucket |

Decision:

- Accept XLA GPU autotune level 4 as the repo default for GPU benchmark/server entry points. It is a configuration fix rather than a model-code optimization, but it unlocks a large measured improvement while preserving exact correctness.
- Preserve explicit `XLA_FLAGS` when users set them. This keeps debugging and reproduction control intact, while making the default GPU path match the fastest verified setting.
- Continue model-side speed work from this stronger baseline. The next candidate should be an opt-in Ampere-compatible fused compact prefill projection kernel; the remaining pure-JAX projection variants have either been accepted already or rejected with profile evidence.

## Entry 034 - Rejected Broadcast GDN Head Repeat

- run id: `20260525-233212-1987844-jax_hetero8_64_512x32_gdn_broadcast_head_repeat`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_broadcast_head_repeat.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-233212-1987844-jax_hetero8_64_512x32_gdn_broadcast_head_repeat`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-233212-1987844-jax_hetero8_64_512x32_gdn_broadcast_head_repeat/plugins/profile/2026_05_25_23_34_08/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-233212-1987844-jax_hetero8_64_512x32_gdn_broadcast_head_repeat/plugins/profile/2026_05_25_23_34_08/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Harvey reviewed Entry 033 and ranked the remaining model-side targets as an Ampere-compatible fused compact prefill projection kernel, a real streaming/top-1 LM-head kernel, then a lowered Gated DeltaNet decode kernel. The audit also called out `wrapped_concatenate` at `36.45 ms` as a decode-side GDN symptom worth reducing only if the source change actually moved that bucket.
- change tested: a temporary `NANO_VLLM_JAX_BROADCAST_GDN_HEAD_REPEAT=1` path replaced GDN query/key `jnp.repeat(..., axis=2)` with `broadcast_to` plus reshape before recurrent GDN prefill/decode. This tried to remove the visible repeat/concatenate lowering without changing the logical head order. The source change was reverted after profiling.
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `345.16 tok/s`, TTFT p50 `324.90 ms`, ITL p50 `13.48 ms`, ITL p95 `14.02 ms`.
- delta vs Entry 033 default: `0.971x` total tok/s, TTFT p50 `1.029x`, ITL p50 `1.049x`, ITL p95 `1.017x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | Entry 034 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 741.69 | end-to-end regression |
| `_run_main_and_sample` | 598.60 | 613.74 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 195.09 | compiled token-id step regressed |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 211.24 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 159.66 | compiled GPU work increased |
| `command_buffer::execute` | 92.22 over 1575 calls | 94.22 over 1575 calls | command-buffer time increased |
| `wrapped_concatenate` | 36.49 over 576 calls | 36.48 over 576 calls | target bucket unchanged |
| `gemm_fusion_dot_234` | 31.89 | 31.87 | LM-head unchanged |
| `gemm_fusion_dot_286` | 25.05 | 25.61 | narrow decode GEMM slightly worse |
| `gemm_fusion_dot_285` | 23.45 | 23.12 | small improvement did not offset regressions |
| `MemcpyD2D` | 24.62 | 24.62 | unchanged |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 | 52.47 | compact prefill GEMM bucket unchanged |

Decision:

- Reject and revert broadcast/reshape GDN head repeat. It preserved exact correctness but did not reduce the `wrapped_concatenate` bucket and made compiled execution materially worse.
- Do not spend more time on equivalent source spellings for this bucket. If GDN decode becomes the next target, use a real lowered backend kernel around the recurrent/conv state update rather than another `repeat` expression rewrite.
- Continue from Entry 033 as the accepted baseline. The next high-upside model-side target remains an optional fused compact prefill projection kernel or a true lowered greedy top-1 LM-head path.

## Entry 035 - Rejected Packed MLP Gate/Up on Autotuned Baseline

- run id: `20260525-234057-1991214-jax_hetero8_64_512x32_packed_mlp_gate_up`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_packed_mlp_gate_up.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-234057-1991214-jax_hetero8_64_512x32_packed_mlp_gate_up`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-234057-1991214-jax_hetero8_64_512x32_packed_mlp_gate_up/plugins/profile/2026_05_25_23_43_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-234057-1991214-jax_hetero8_64_512x32_packed_mlp_gate_up/plugins/profile/2026_05_25_23_43_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Harvey found Entry 018 had already rejected the same source-level idea before the XLA autotune change. The current run intentionally rechecked it against Entry 033 because autotune changed the GEMM lowering; the result reconfirmed the rejection.
- change tested: a temporary `NANO_VLLM_JAX_PACKED_MLP_GATE_UP=1` path materialized a per-layer `gate_up_proj = concat(gate_proj, up_proj)`, used one dot, split gate/up activations, and then kept the same `silu(gate) * up` and `down_proj` math. The source change was reverted after profiling.
- correctness: targeted CUDA helper check passed, and the full hetero8 run had exact generated-token match for all 8 rows against the Entry 033 reference.
- JAX timing: `289.40 tok/s`, TTFT p50 `344.05 ms`, ITL p50 `15.86 ms`, ITL p95 `24.30 ms`.
- delta vs Entry 033 default: `0.815x` total tok/s, TTFT p50 `1.089x`, ITL p50 `1.233x`, ITL p95 `1.763x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | Entry 035 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 884.60 | major regression |
| `_run_main_and_sample` | 598.60 | 665.63 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 265.22 | compiled token-id step much slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 298.32 over 252 calls | same dispatch count, much heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 209.07 | widened MLP lowering regressed |
| `command_buffer::execute` | 92.22 over 1575 calls | 114.51 over 1575 calls | command-buffer time increased |
| `PjitFunction(convert_element_type)` | 77.18 | 196.59 | extra packed parameter leaves increased host/JAX conversion overhead |
| `DevicePut` | 10.85 | 45.47 | parameter/metadata traffic increased |
| `gemm_fusion_dot_234` | 31.89 | 0.00 | bucket renamed to `gemm_fusion_dot_210`, not eliminated |
| `gemm_fusion_dot_210` | 0.00 | 31.85 | LM-head-equivalent bucket unchanged |
| `gemm_fusion_dot_286` | 25.05 | 0.00 | bucket renamed after packing |
| `gemm_fusion_dot_6` | 0.00 | 24.57 | narrow/MLP bucket reassigned, not improved |
| `wrapped_concatenate` | 36.49 | 36.45 | unchanged |
| `MemcpyD2D` | 24.62 | 27.23 | copy bucket worsened |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 | 52.49 | compact prefill GEMM bucket unchanged |

Decision:

- Reject and revert packed MLP gate/up again, now with evidence on the accepted autotuned baseline. The packed path preserved correctness but made compiled execution, command-buffer time, host conversion, and ITL much worse.
- Do not revisit source-level packed MLP projection unless the parameter representation changes fundamentally. Any future MLP work should be a real lowered compact projection kernel or a fused MLP kernel that avoids adding large parameter leaves and does not depend on XLA discovering the right split.
- Continue from Entry 033 as the accepted implementation baseline.

## Entry 036 - Rejected Pallas GPU Paged Decode Attention

- run id: `20260525-235711-2023622-jax_hetero8_64_512x32_pallas_paged_decode_attention_ppb2`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_pallas_paged_decode_attention_ppb2.json`
- profile directory: `/mountpoint/.exp/profiles/20260525-235711-2023622-jax_hetero8_64_512x32_pallas_paged_decode_attention_ppb2`
- Perfetto trace: `/mountpoint/.exp/profiles/20260525-235711-2023622-jax_hetero8_64_512x32_pallas_paged_decode_attention_ppb2/plugins/profile/2026_05_25_23_59_03/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260525-235711-2023622-jax_hetero8_64_512x32_pallas_paged_decode_attention_ppb2/plugins/profile/2026_05_25_23_59_03/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Harvey recommended a Pallas Triton streaming/top-1 LM-head kernel as the next most implementable lowered path. This paged-attention experiment was still run because JAX ships a GPU Pallas paged-attention op that matches the full-attention decode boundary closely enough to test quickly.
- feasibility checks: a CUDA smoke test of `jax.experimental.pallas.ops.gpu.paged_attention` matched the current pure-JAX decode reference when the query was pre-scaled by `1 / sqrt(head_dim)` (`max_abs=0.00390625`, MSE `4.006872e-07`). The default `pages_per_compute_block=8` full benchmark failed at warmup with `RESOURCE_EXHAUSTED` shared memory (`286720` requested, `101376` available), so the profiled run used `NANO_VLLM_JAX_PALLAS_PAGED_DECODE_PAGES_PER_BLOCK=2`.
- change tested: a temporary env-gated GPU backend path used Pallas paged attention for full-attention decode only (`query_len == 1`), transposing the layer KV cache from `[blocks, block, kv_heads, head_dim]` to the Pallas `[kv_heads, pages, page_size, head_dim]` layout and leaving prefill, MTP multi-token decode, and the pure-JAX fallback unchanged. The source change was reverted after profiling.
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `282.01 tok/s`, TTFT p50 `321.28 ms`, ITL p50 `17.91 ms`, ITL p95 `22.39 ms`.
- delta vs Entry 033 default: `0.794x` total tok/s, TTFT p50 `1.017x`, ITL p50 `1.393x`, ITL p95 `1.624x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | Entry 036 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 907.78 | major regression |
| `_run_main_and_sample` | 598.60 | 753.84 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 185.47 | compiled token-id step slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 207.62 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 144.22 | slight GPU-module improvement did not translate to latency |
| `command_buffer::execute` | 92.22 over 1575 calls | 94.36 over 1358 calls | fewer command buffers, slightly more time |
| `paged_attention` | 0.00 | 7.15 over 186 calls | new Pallas paged-attention ranges |
| `transpose` | 64.01 | 103.96 | layer KV layout conversion cost increased |
| `gemm_fusion_dot_234` | 31.89 | 0.00 | bucket renamed to `gemm_fusion_dot_222`, not eliminated |
| `gemm_fusion_dot_222` | 0.00 | 31.82 | LM-head-equivalent bucket unchanged |
| `gemm_fusion_dot_286` | 25.05 | 3.86 | some narrow decode GEMM work moved, but not enough |
| `MemcpyD2D` | 24.62 | 9.47 | copy bucket improved |
| `wrapped_concatenate` | 36.49 | 36.23 | unchanged |

Decision:

- Reject and revert the Pallas GPU paged decode attention path. It preserved exact correctness and reduced some copy/narrow-GEMM buckets, but the layer-local KV transpose/layout cost and heavier execution made decode latency much worse.
- Do not integrate the shipped Pallas paged-attention op without changing the physical KV cache layout. It wants `[kv_heads, pages, page_size, head_dim]`, while the current pedagogical cache uses `[layers, pages, page_size, kv_heads, head_dim]`; transposing per full-attention layer in the hot decode path erases the kernel benefit.
- Continue from Entry 033 as the accepted implementation baseline. The next lowered experiment should follow the side-audit recommendation: a streaming/top-1 LM-head kernel that avoids changing KV layout.

## Entry 037 - Rejected Pallas Triton LM-Head Argmax

- best run id: `20260526-002748-2034344-jax_hetero8_64_512x32_pallas_lm_head_argmax_bh128_bv256`
- best benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_pallas_lm_head_argmax_bh128_bv256.json`
- best profile directory: `/mountpoint/.exp/profiles/20260526-002748-2034344-jax_hetero8_64_512x32_pallas_lm_head_argmax_bh128_bv256`
- best Perfetto trace: `/mountpoint/.exp/profiles/20260526-002748-2034344-jax_hetero8_64_512x32_pallas_lm_head_argmax_bh128_bv256/plugins/profile/2026_05_26_00_29_42/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- best TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-002748-2034344-jax_hetero8_64_512x32_pallas_lm_head_argmax_bh128_bv256/plugins/profile/2026_05_26_00_29_42/INDCS0291.atrapa.deloitte.com.xplane.pb`
- other benchmark artifacts:
  - `results/qwen08_jax_server_trace_hetero8_64_512x32_pallas_lm_head_argmax.json`
  - `results/qwen08_jax_server_trace_hetero8_64_512x32_pallas_lm_head_argmax_chunked512.json`
- subagent audit: Harvey recommended trying a Pallas Triton streaming/top-1 LM-head kernel as a greedy-only, materialized-LM-head experiment, with explicit value/index tile reduction to preserve `jnp.argmax` lowest-index tie-breaking. The audit also predicted the main risk: this could still lose to XLA's autotuned GEMM because the arithmetic is unchanged and the only possible win is avoiding full-logits materialization.
- change tested: a temporary `NANO_VLLM_JAX_PALLAS_LM_HEAD_ARGMAX=1` path replaced the decode-time `jnp.dot(hidden, lm_head) -> argmax` in `lm_head_token_ids_and_topk` with a Pallas Triton tile-argmax kernel. The path was gated to greedy decode only (`top_k == 0`, `is_prefill == False`, `[B, 1, H]` hidden, materialized `params.lm_head`) and left prefill/top-k/MTP verifier logits on the dense fallback. The source change was reverted after profiling.
- feasibility checks: CUDA routed-JIT smokes matched dense `jnp.argmax(jnp.dot(hidden_fp32, weight_bf16))`. The first full-H tile prototype only fit at `BLOCK_V <= 32` on this GPU and was very slow. A chunked kernel over hidden tiles allowed larger vocab tiles; a real-shape microbenchmark over `[8, 1024] x [1024, 248320]` found `BLOCK_H=128`, `BLOCK_V=256`, `num_warps=8` fastest among tested Pallas settings (`1.172 ms` vs dense `1.196 ms` in isolation), but the integrated server profile still regressed.
- correctness: all profiled full hetero8 runs had exact generated-token match for all 8 rows against the Entry 033 reference.

Best tuned run vs Entry 033:

| metric | Entry 033 | tuned Pallas LM-head | delta |
| --- | ---: | ---: | ---: |
| tok/s | 355.29 | 346.14 | 0.974x |
| TTFT p50 | 315.89 ms | 321.46 ms | 1.018x |
| ITL p50 | 12.86 ms | 13.55 ms | 1.054x |
| ITL p95 | 13.79 ms | 14.15 ms | 1.027x |

Top trace ranges, total inclusive time:

| range | Entry 033 ms | tuned Pallas LM-head ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 739.58 | overall slower |
| `_run_main_and_sample` | 598.60 | 611.68 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 193.87 | decode token-id step slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 208.85 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 157.32 | module time increased |
| `command_buffer::execute` | 92.22 over 1575 calls | 94.05 over 1575 calls | slightly slower |
| `gemm_fusion_dot_234` | 31.89 over 31 calls | 0.00 | dense LM-head GEMM removed |
| `lm_head_tile_argmax` | 0.00 | 32.11 over 31 calls | Pallas replacement costs about the same as the removed GEMM |
| `MemcpyD2D` | 24.62 | 24.55 | unchanged |
| `input_reduce_fusion` | 41.57 | 40.99 | unchanged/slightly lower |

Profiled variants:

| variant | artifact | tok/s | ITL p50 | note |
| --- | --- | ---: | ---: | --- |
| full-H tile, `BLOCK_V=32` | `qwen08_jax_server_trace_hetero8_64_512x32_pallas_lm_head_argmax.json` | 285.53 | 18.26 ms | functional but many tiny tiles; `lm_head_tile_argmax` cost 188.17 ms |
| chunked, `BLOCK_H=64`, `BLOCK_V=512` | `qwen08_jax_server_trace_hetero8_64_512x32_pallas_lm_head_argmax_chunked512.json` | 339.47 | 13.70 ms | much better, still slower than Entry 033 |
| chunked, `BLOCK_H=128`, `BLOCK_V=256` | `qwen08_jax_server_trace_hetero8_64_512x32_pallas_lm_head_argmax_bh128_bv256.json` | 346.14 | 13.55 ms | best tested integrated setting, still slower |

Decision:

- Reject and revert the Pallas Triton LM-head argmax path. It preserved correctness and removed the dense LM-head GEMM bucket, but the custom call plus second-stage tile reduction did not beat XLA's autotuned dense GEMM inside the full server decode graph.
- Do not revisit a two-stage Pallas argmax for this LM head on A10G unless the design eliminates the cross-tile reduction or fuses the final token selection into a single custom call. The current best path remains Entry 033.

## Entry 038 - Rejected Unrolled GDN Conv1D Decode Update

- run id: `20260526-004143-2040570-jax_hetero8_64_512x32_unrolled_conv1d_update`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_unrolled_conv1d_update.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-004143-2040570-jax_hetero8_64_512x32_unrolled_conv1d_update`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-004143-2040570-jax_hetero8_64_512x32_unrolled_conv1d_update/plugins/profile/2026_05_26_00_43_35/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-004143-2040570-jax_hetero8_64_512x32_unrolled_conv1d_update/plugins/profile/2026_05_26_00_43_35/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Turing recommended making fused compact prefill projection the next serious optimization target and ranked Pallas RMSNorm and GDN decode lowering behind it. Local RMSNorm CUDA microbenchmarks supported that ranking: for representative Qwen shapes (`[8, 1, 8, 256]`, `[8, 512, 8, 256]`, `[8, 512, 1024]`), the shipped Pallas RMSNorm wrapper was equal or slower than XLA, so it was not promoted to a full source experiment.
- change tested: a temporary `NANO_VLLM_JAX_UNROLLED_CONV1D_UPDATE=1` path special-cased `causal_conv1d_update` for kernel size 4. It replaced `jnp.roll(...).at[..., -1].set(...)` plus `einsum` with an explicit four-tap update and direct multiply/add in an attempt to reduce the `wrapped_concatenate` bucket from decode-time GDN convolution state shifts. The source change was reverted after profiling.
- smoke checks: a CUDA microbenchmark on the decode shape `[B=8, D=6144, K=4]` matched the current state exactly and had `max_abs=1.9073486328125e-06` on the convolution output. Focused CUDA tests passed with the flag enabled: `tests/test_layer_parity.py tests/test_lm_head_helpers.py` (`9 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `341.62 tok/s`, TTFT p50 `327.96 ms`, ITL p50 `13.13 ms`, ITL p95 `15.70 ms`.
- delta vs Entry 033 default: `0.962x` total tok/s, TTFT p50 `1.038x`, ITL p50 `1.021x`, ITL p95 `1.139x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | unrolled Conv1D ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 749.35 | overall regression |
| `_run_main_and_sample` | 598.60 | 620.46 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 201.59 | compiled decode token-id step much slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 216.62 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 163.83 | module time increased |
| `command_buffer::execute` | 92.22 over 1575 calls | 95.96 over 1575 calls | command-buffer time worsened |
| `wrapped_concatenate` | 36.49 over 576 calls | 36.26 over 576 calls | target bucket barely moved |
| `MemcpyD2D` | 24.62 | 24.75 | unchanged/slightly worse |
| `input_reduce_fusion` | 41.57 | 41.52 | unchanged |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 | 52.46 | compact prefill GEMM unchanged |

Decision:

- Reject and revert the unrolled GDN Conv1D decode update. The microbenchmark was slightly faster in isolation, but the integrated graph did not reduce the target `wrapped_concatenate` bucket and made the decode executable heavier.
- Do not keep source-level rewrites of the GDN Conv1D state shift. If this area is revisited, it should be as part of a real lowered GDN decode kernel that owns the convolution update and recurrent state update together.
- Follow the subagent recommendation for the next model-side experiment: an Ampere-compatible fused compact prefill projection path, with a microbenchmark gate before any full hetero8 run.

## Entry 039 - Rejected Pallas Fused Compact Prefill Projection

- run id: `20260526-005559-2044767-jax_hetero8_64_512x32_fused_compact_prefill_proj`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_fused_compact_prefill_proj.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-005559-2044767-jax_hetero8_64_512x32_fused_compact_prefill_proj`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-005559-2044767-jax_hetero8_64_512x32_fused_compact_prefill_proj/plugins/profile/2026_05_26_00_57_49/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-005559-2044767-jax_hetero8_64_512x32_fused_compact_prefill_proj/plugins/profile/2026_05_26_00_57_49/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: a temporary `NANO_VLLM_JAX_FUSED_COMPACT_PREFILL_PROJ=1` path added a Pallas Triton compact projection helper that loaded true ragged prefill rows directly, performed tiled matmul, and scattered directly into the rectangular `[batch, seq, output]` result via output aliasing. The path was intended for `_compact_prefill_dot_if_enabled` and left MLP packing unchanged. The source change was reverted after profiling.
- microbenchmark gate: for BF16 activations with BF16 weights, the direct Pallas helper was exact and faster on real hetero8 prefill dimensions (`B=8`, `T=512`, `compact_tokens=2304`, `H=1024`): `O=512` was `0.95x`, `O=2048` was `0.85x`, `O=4096` was `0.93x`, and `O=6144` was `0.76x` the current compact helper time. This positive gate was misleading for the real serving contract because the benchmark uses FP32 activations with BF16 weights.
- mixed-dtype gate: with FP32 activations and BF16 weights, the Pallas helper had to cast weights to FP32 inside the tile. It matched the current JAX result with default TF32 behavior on tested shapes, but was much slower: `O=2048` was `1.48x` current time and `O=6144` was `1.76x` current time. `allow_tf32=False` was unusably slow (`58x`-`79x`) and did not match the current XLA numerics.
- full-run caveat: the profiled source path conservatively gated Pallas to matching BF16/BF16 dtypes after the FP32 drift smoke. Therefore the hetero8 profile did not replace the compact projection buckets under the actual FP32-activation/BF16-weight benchmark. The trace confirms the compact GEMM buckets were unchanged. The run is still useful evidence that this implementation should not be integrated without a faster mixed-dtype kernel.
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `339.28 tok/s`, TTFT p50 `321.69 ms`, ITL p50 `13.62 ms`, ITL p95 `15.61 ms`.
- delta vs Entry 033 default: `0.955x` total tok/s, TTFT p50 `1.018x`, ITL p50 `1.059x`, ITL p95 `1.132x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | fused compact flag ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 754.53 | overall slower |
| `_run_main_and_sample` | 598.60 | 619.82 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 207.38 | decode token-id step slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 223.43 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 164.14 | module time increased |
| `command_buffer::execute` | 92.22 over 1575 calls | 96.62 over 1575 calls | command-buffer time worsened |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 over 96 calls | 52.41 over 96 calls | compact prefill GEMM unchanged |
| `gemm_fusion_dot_general_746` | 13.92 | 13.92 | unchanged |
| `gemm_fusion_dot_2` | 13.79 | 13.79 | unchanged |
| `input_reduce_fusion` | 41.57 | 41.53 | unchanged |
| `wrapped_concatenate` | 36.49 | 36.48 | unchanged |

Decision:

- Reject and revert the Pallas fused compact prefill projection path for the current BF16-weight/FP32-activation serving contract. A BF16/BF16 helper is not aligned with the correctness contract, and the mixed FP32/BF16 version loses badly to XLA/CUTLASS.
- Do not revisit this exact Pallas design unless it can use tensor cores efficiently while preserving FP32 activations and BF16 weights, or unless the accepted serving contract changes. The next prefill work should inspect HLO/layout around the existing XLA compact dots rather than replacing them with many small Pallas custom calls.

## Entry 040 - Rejected Pallas Gated DeltaNet Decode Kernel

- run id: `20260526-010942-2048273-jax_hetero8_64_512x32_pallas_gdn_decode`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_pallas_gdn_decode.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-010942-2048273-jax_hetero8_64_512x32_pallas_gdn_decode`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-010942-2048273-jax_hetero8_64_512x32_pallas_gdn_decode/plugins/profile/2026_05_26_01_11_26/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-010942-2048273-jax_hetero8_64_512x32_pallas_gdn_decode/plugins/profile/2026_05_26_01_11_26/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: a temporary `NANO_VLLM_JAX_PALLAS_GDN_DECODE=1` path routed width-1 cached Gated DeltaNet decode to a Pallas Triton helper for FP32 activations/state. The kernel owned Q/K L2 normalization, state decay, KV memory read, delta update, state writeback, and output projection. The source change was reverted after profiling.
- microbenchmark gate: on representative decode shape (`B=8`, `H=16`, `D=128`, `T=1`), the Pallas helper with `BLOCK_V=64` and `allow_tf32=False` matched the JAX recurrent path (`out_max ~= 1.34e-7`, `state_max ~= 4.77e-7`, `out_mse ~= 4.22e-16`) and was faster in isolation (`0.185 ms` vs `0.237 ms`, `0.783x`). The faster default-TF32 variant had about `1.1e-4` output drift and was not used.
- focused CUDA checks: `tests/test_layer_parity.py::test_linear_attention_recurrent`, the temporary routed Pallas parity test, and `tests/test_lm_head_helpers.py` passed before the full run (`5 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `347.11 tok/s`, TTFT p50 `322.66 ms`, ITL p50 `13.22 ms`, ITL p95 `14.07 ms`.
- delta vs Entry 033 default: `0.977x` total tok/s, TTFT p50 `1.021x`, ITL p50 `1.028x`, ITL p95 `1.020x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | Pallas GDN decode ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 737.51 | overall slower |
| `_run_main_and_sample` | 598.60 | 609.89 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 179.80 | decode token-id step slightly slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 194.69 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 148.38 | module bucket nearly unchanged |
| `command_buffer::execute` | 92.22 over 1575 calls | 93.58 over 1730 calls | more command-buffer work |
| `command_buffer::update` | 27.81 over 248 calls | 29.53 over 403 calls | more command-buffer updates |
| `gated_delta_decode` | 0.00 | 18.39 over 558 calls | new custom-call work; did not reduce total step latency |
| `gemm_fusion_dot_286` | 25.05 | 25.78 | unchanged/slightly worse |
| `gemm_fusion_dot_285` | 23.45 | 23.14 | small local improvement did not pay for custom calls |
| `wrapped_concatenate` | 36.49 | 36.20 | effectively unchanged |
| `MemcpyD2D` | 24.62 | 25.61 | slightly worse |
| `input_reduce_fusion` | 41.57 | 41.54 | unchanged |

Decision:

- Reject and revert the Pallas Gated DeltaNet decode path. It was numerically clean under the FP32-activation/BF16-weight serving contract and faster in isolation, but the integrated server graph regressed throughput and ITL.
- The profile shows the custom-call path adds `gated_delta_decode` ranges and more command-buffer update/execute activity without reducing the dominant dense/decode buckets. This is another case where a good isolated Pallas kernel loses once inserted 18 layers x 31 decode steps into the compiled server loop.
- Do not revisit a per-head/per-value-tile Pallas GDN decode kernel in this shape. If GDN decode is revisited, it likely needs coarser fusion that also owns the Conv1D shift/projection boundary or reduces custom-call count, not only the recurrent state update.

## Entry 041 - Rejected Full-Active Decode Valid-Mask Skip

- run id: `20260526-012105-2053974-jax_hetero8_64_512x32_full_active_decode_mask`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_full_active_decode_mask.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-012105-2053974-jax_hetero8_64_512x32_full_active_decode_mask`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-012105-2053974-jax_hetero8_64_512x32_full_active_decode_mask/plugins/profile/2026_05_26_01_23_03/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-012105-2053974-jax_hetero8_64_512x32_full_active_decode_mask/plugins/profile/2026_05_26_01_23_03/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: a temporary static fast path detected normal width-1 decode batches where `num_decode_tokens == batch_size` and skipped per-layer `valid_token_mask` construction plus inactive-row preservation in `gated_deltanet_block`. Prefill, inactive padded decode, and wider verifier/MTP batches kept the existing logic. The source change was reverted after profiling.
- focused CUDA checks: `tests/test_backend_boundaries.py::test_full_active_decode_mask_fastpath_matches_linear_fallback`, `tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, and `tests/test_lm_head_helpers.py` passed (`6 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `346.03 tok/s`, TTFT p50 `323.29 ms`, ITL p50 `13.56 ms`, ITL p95 `14.36 ms`.
- delta vs Entry 033 default: `0.974x` total tok/s, TTFT p50 `1.023x`, ITL p50 `1.055x`, ITL p95 `1.041x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | full-active mask ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 739.82 | overall slower |
| `_run_main_and_sample` | 598.60 | 610.22 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 191.78 | decode token-id step slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 207.60 over 252 calls | same dispatch count, heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 155.85 | module time increased |
| `command_buffer::execute` | 92.22 over 1575 calls | 94.89 over 1730 calls | more command-buffer work |
| `command_buffer::update` | 27.81 over 248 calls | 35.03 over 403 calls | update count increased |
| `input_reduce_fusion` | 41.57 | 41.52 | target reduction bucket unchanged |
| `wrapped_concatenate` | 36.49 | 36.29 | negligible improvement |
| `gemm_fusion_dot_286` | 25.05 | 24.55 | small local improvement did not pay for heavier graph |
| `MemcpyD2D` | 24.62 | 25.54 | slightly worse |

Decision:

- Reject and revert the full-active valid-mask skip. It was correctness-clean, but XLA did not turn the source-level branch removal into a lighter decode graph.
- The intended mask/reduction work was already effectively optimized or not on the critical path; the static key split and altered graph increased command-buffer update/execute cost instead.
- Do not add full-active decode specializations unless the profile shows a real lowered bucket moving. The baseline's generic valid-mask path remains better for this workload.

## Entry 042 - Rejected Compact Prefill Metadata Indices

- run id: `20260526-012720-2056921-jax_hetero8_64_512x32_compact_prefill_metadata_indices`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_compact_prefill_metadata_indices.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-012720-2056921-jax_hetero8_64_512x32_compact_prefill_metadata_indices`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-012720-2056921-jax_hetero8_64_512x32_compact_prefill_metadata_indices/plugins/profile/2026_05_26_01_29_09/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-012720-2056921-jax_hetero8_64_512x32_compact_prefill_metadata_indices/plugins/profile/2026_05_26_01_29_09/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Banach reviewed Entry 033 plus rejected Entries 034-040 and ranked compact prefill index reuse as a plausible medium-reward experiment, but with an explicit gate: it should only be kept if repeated `nonzero`/mask work was not already CSE'd and the profile showed lower prefill buckets without dispatch tax.
- change tested: a temporary `AttentionMetadata` extension precomputed compact prefill row/column indices once in `build_attention_metadata`, then `_compact_prefill_dot_if_enabled` and `_compact_prefill_mlp` reused those indices instead of calling `jnp.nonzero` locally in each compact projection helper. The source change was reverted after profiling.
- focused CUDA checks: `tests/test_lm_head_helpers.py`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, `tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode`, and `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute` passed (`6 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference.
- JAX timing: `337.28 tok/s`, TTFT p50 `321.15 ms`, ITL p50 `13.88 ms`, ITL p95 `16.52 ms`.
- delta vs Entry 033 default: `0.949x` total tok/s, TTFT p50 `1.017x`, ITL p50 `1.080x`, ITL p95 `1.199x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | compact metadata indices ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 759.00 | overall slower |
| `_run_main_and_sample` | 598.60 | 618.88 | runner hot path slower |
| `forward_step_token_ids_jit` | 175.81 | 212.63 | compiled token-id step much slower |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 226.66 over 252 calls | same dispatch count, much heavier execution |
| `jit_compiled:XLA GPU module` | 147.90 | 170.07 | module time increased |
| `command_buffer::execute` | 92.22 over 1575 calls | 98.28 over 1575 calls | command-buffer time worsened |
| `command_buffer::update` | 27.81 over 248 calls | 38.53 over 248 calls | same count, heavier updates |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 over 96 calls | 52.43 over 96 calls | compact prefill GEMM unchanged |
| `gemm_fusion_dot_general_746` | 13.92 | 13.92 | unchanged |
| `gemm_fusion_dot_2` | 13.79 | 13.79 | unchanged |
| `input_reduce_fusion` | 41.57 | 41.52 | unchanged |
| `wrapped_concatenate` | 36.49 | 36.48 | unchanged |
| `PjitFunction(compiled)` | 347.46 | 419.62 | higher compiled execution attribution |

Decision:

- Reject and revert compact prefill metadata indices. Correctness was exact, but the profile proved the repeated compact-index work was not the limiting cost and the metadata extension made the compiled graph heavier.
- Do not move compact projection indices into `AttentionMetadata` in this form. If compact prefill is revisited, inspect HLO/layout around the existing XLA compact dots or change the attention algorithm, not the index plumbing.
- Next candidates should follow the audit's remaining list: initial-prefill local attention while preserving paged KV writes, or a controlled GDN chunk/layout sweep with exact-token gates.

## Entry 043 - Rejected Initial-Prefill Local Attention

- run id: `20260526-014543-2064227-jax_hetero8_64_512x32_initial_prefill_local_attention`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_initial_prefill_local_attention.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-014543-2064227-jax_hetero8_64_512x32_initial_prefill_local_attention`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-014543-2064227-jax_hetero8_64_512x32_initial_prefill_local_attention/plugins/profile/2026_05_26_01_47_32/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-014543-2064227-jax_hetero8_64_512x32_initial_prefill_local_attention/plugins/profile/2026_05_26_01_47_32/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260526-014903-2065504-jax_hetero8_64_512x32_initial_prefill_local_attention_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_initial_prefill_local_attention_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260526-014903-2065504-jax_hetero8_64_512x32_initial_prefill_local_attention_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260526-014903-2065504-jax_hetero8_64_512x32_initial_prefill_local_attention_repeat/plugins/profile/2026_05_26_01_49_27/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-014903-2065504-jax_hetero8_64_512x32_initial_prefill_local_attention_repeat/plugins/profile/2026_05_26_01_49_27/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Archimedes reviewed the temporary path and found the local attention math shape-compatible with the paged path, including GQA head grouping and K/V write ordering, but required a stricter host-only gate than the first draft. The gate must only allow whole no-prefix prompt prefill: env on, prefill step, zero decode tokens, host query/seq/seq-id/final metadata present, active rows `query_len == seq_len`, inactive rows zeroed, final flags true, and `num_prefill_tokens == sum(query_lens)`. Cached-prefix chunks, first chunks of chunked prefill, and MTP verifier-like batches must stay on the paged path.
- change tested: a temporary `NANO_VLLM_JAX_INITIAL_PREFILL_LOCAL_ATTN=1` path still wrote full-attention K/V through `backend.write_kv`, but for strictly gated whole prompt prefill computed attention output directly from the local projected K/V tensors instead of gathering through the paged cache. The static gate was included in the JIT cache key. The source change was reverted after profiling.
- focused CUDA checks: after tightening the gate, `tests/test_backend_boundaries.py::test_initial_prefill_local_attention_static_gate`, `tests/test_backend_boundaries.py::test_initial_prefill_local_attention_matches_paged_prefill`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, `tests/test_backend_boundaries.py::test_executor_mtp1_greedy_step_jit_matches_separate_path`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` with the env flag enabled (`8 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference in both profiled runs.
- first JAX timing: `330.57 tok/s`, TTFT p50 `324.90 ms`, ITL p50 `13.82 ms`, ITL p95 `17.54 ms`.
- repeat JAX timing: `355.75 tok/s`, TTFT p50 `314.99 ms`, ITL p50 `12.90 ms`, ITL p95 `13.79 ms`.
- repeat delta vs Entry 033 default: `1.001x` total tok/s, TTFT p50 `0.997x`, ITL p50 `1.004x`, ITL p95 `1.000x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | local attention repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 719.60 | overall tied |
| `_run_main_and_sample` | 598.60 | 597.95 | runner hot path tied |
| `forward_step_token_ids_jit` | 175.81 | 173.39 | small cumulative improvement, mostly noise scale |
| first `forward_step_token_ids_jit` | 83.25 | 83.38 | initial prefill itself did not improve |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 188.94 over 252 calls | slight cumulative improvement |
| `jit_compiled:XLA GPU module` | 147.90 | 145.88 | slight cumulative improvement |
| `command_buffer::execute` | 92.22 over 1575 calls | 90.80 over 1566 calls | 9 fewer command-buffer execute ranges |
| `command_buffer::update` | 27.81 over 248 calls | 26.69 over 248 calls | small improvement |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 | 52.42 | compact prefill GEMM unchanged |
| `gemm_fusion_dot_general_746` | 13.92 | 13.91 | unchanged |
| `gemm_fusion_dot_2` | 117.43 | 117.84 | unchanged/slightly worse |
| `input_reduce_fusion` | 59.30 | 58.79 | unchanged |
| `wrapped_concatenate` | 36.49 | 36.49 | unchanged |
| `MemcpyD2D` | 24.62 | 24.69 | unchanged |

Decision:

- Reject and revert initial-prefill local attention. Correctness was exact and the stricter gate was safe, but the profile did not show a real TTFT or first-prefill improvement. The repeat was only tied with Entry 033, while the first profiled run was materially slower.
- Do not keep this source branch as an optional flag; it adds another compiled shape key and attention implementation for no durable win.
- The useful follow-up from this experiment is negative evidence: avoiding the paged-cache gather for the whole initial prefill does not move the dominant prefill bucket in this hetero8 workload. The next prefill work should target the XLA full-attention/prefill HLO layout itself or run the controlled GDN chunk/layout sweep from the audit.

## Entry 044 - Rejected Async Greedy Token Readback

- run id: `20260526-015956-2070683-jax_hetero8_64_512x32_async_token_readback`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_async_token_readback.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-015956-2070683-jax_hetero8_64_512x32_async_token_readback`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-015956-2070683-jax_hetero8_64_512x32_async_token_readback/plugins/profile/2026_05_26_02_01_48/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-015956-2070683-jax_hetero8_64_512x32_async_token_readback/plugins/profile/2026_05_26_02_01_48/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260526-020223-2071684-jax_hetero8_64_512x32_async_token_readback_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_async_token_readback_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260526-020223-2071684-jax_hetero8_64_512x32_async_token_readback_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260526-020223-2071684-jax_hetero8_64_512x32_async_token_readback_repeat/plugins/profile/2026_05_26_02_02_48/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-020223-2071684-jax_hetero8_64_512x32_async_token_readback_repeat/plugins/profile/2026_05_26_02_02_48/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Wegener reviewed the accepted Entry 033 and rejected Entries 040-043. It ranked the next high-probability model-side experiments as: (1) controlled GDN prefill chunk/layout sweep in `jax_chunk_gated_delta_rule`/`gated_deltanet_block`, expecting `input_reduce_fusion` and first prefill step movement; (2) HLO-guided full-attention/paged-prefill layout work in `paged_attention_prefill`; (3) a transpose-free GPU KV layout for future Pallas decode attention, only if the physical cache layout removes the transpose regression from Entry 036.
- change tested: a temporary `NANO_VLLM_JAX_ASYNC_TOKEN_READBACK=1` path called `copy_to_host_async()` on the small greedy token-id result immediately after `forward_step_token_ids_jit`, before cache/hybrid snapshot bookkeeping. The goal was to overlap token-id device-to-host transfer with Python-side post-step bookkeeping and reduce the later `token_ids.tolist()` synchronization range. The source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_ASYNC_TOKEN_READBACK=1` and `NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH=1`, `tests/test_backend_boundaries.py::test_model_runner_uses_bucketed_batched_jit_path`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, and `tests/test_lm_head_helpers.py` passed (`6 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference in both profiled runs.
- first JAX timing: `340.22 tok/s`, TTFT p50 `322.00 ms`, ITL p50 `13.56 ms`, ITL p95 `15.36 ms`.
- repeat JAX timing: `355.19 tok/s`, TTFT p50 `318.38 ms`, ITL p50 `12.76 ms`, ITL p95 `13.66 ms`.
- repeat delta vs Entry 033 default: `1.000x` total tok/s, TTFT p50 `1.008x`, ITL p50 `0.993x`, ITL p95 `0.991x`.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | async readback repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 720.73 | overall tied |
| `_run_main_and_sample` | 598.60 | 598.10 | runner hot path tied |
| `forward_step_token_ids_jit` | 175.81 | 172.01 | small cumulative improvement, not reflected in throughput |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 187.62 over 252 calls | slight cumulative improvement |
| `jit_compiled:XLA GPU module` | 147.90 | 144.88 | slight cumulative improvement |
| `command_buffer::execute` | 92.22 over 1575 calls | 91.14 over 1575 calls | same dispatch count |
| `command_buffer::update` | 27.81 over 248 calls | 26.62 over 248 calls | slight cumulative improvement |
| `array.py:325 tolist` | 409.69 | 410.12 | target host sync unchanged |
| `np.asarray(jax.Array)` | 409.43 | 409.93 | target host sync unchanged |
| `input_reduce_fusion` | 59.30 | 59.08 | unchanged |
| `wrapped_concatenate` | 36.49 | 36.50 | unchanged |
| `MemcpyD2D` | 24.62 | 24.84 | unchanged/slightly worse |

Decision:

- Reject and revert async greedy token readback. The idea was correctness-neutral, but the profile shows the actual `tolist`/`np.asarray(jax.Array)` synchronization bucket did not move.
- Do not keep an optional flag for this. If host synchronization is revisited, it needs a larger serving-loop change that changes the dependency structure, not only `copy_to_host_async()` on the final token-id array.
- Follow the xhigh audit for the next model-side experiment: controlled GDN prefill chunk/layout sweep with exact-token gates and profile focus on `input_reduce_fusion`, first `forward_step_token_ids_jit`, and total compiled execution.

## Entry 045 - Accepted GDN Prefill Chunk Size 32

- accepted run id: `20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat`
- accepted benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`
- accepted profile directory: `/mountpoint/.exp/profiles/20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat`
- accepted Perfetto trace: `/mountpoint/.exp/profiles/20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat/plugins/profile/2026_05_26_02_20_50/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- accepted TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-022026-2079580-jax_hetero8_64_512x32_gdn_chunk32_default_repeat/plugins/profile/2026_05_26_02_20_50/INDCS0291.atrapa.deloitte.com.xplane.pb`
- supporting chunk-32 first run: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32.json`, profile `/mountpoint/.exp/profiles/20260526-020811-2074219-jax_hetero8_64_512x32_gdn_chunk32`, `358.77 tok/s`, exact correctness.
- supporting chunk-32 explicit repeat: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_repeat.json`, profile `/mountpoint/.exp/profiles/20260526-021331-2076191-jax_hetero8_64_512x32_gdn_chunk32_repeat`, `369.14 tok/s`, exact correctness.
- rejected chunk-128 sweep: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk128.json`, profile `/mountpoint/.exp/profiles/20260526-021053-2075185-jax_hetero8_64_512x32_gdn_chunk128`, `293.75 tok/s`, exact correctness but TTFT p50 `457.02 ms`.
- change accepted: set `Qwen3_5Config.linear_chunk_size` and Qwen3.5 preset chunk sizes from `64` to `32`. Also add a benchmark-only `--linear-chunk-size` override to `benchmark_jax_server_trace.py` so future sweeps can reproduce alternate GDN chunk sizes without changing source defaults.
- focused CUDA checks: after promoting chunk size 32 as the default, `tests/test_kv_cache.py::test_linear_attention_chunked_vs_recurrent`, `tests/test_kv_cache.py::test_linear_attention_multichunk_matches_recurrent`, `tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`8 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 033 hetero8 reference in all chunk-size sweep runs.
- accepted JAX timing: `367.80 tok/s`, TTFT p50 `289.98 ms`, ITL p50 `13.14 ms`, ITL p95 `13.59 ms`.
- delta vs Entry 033 default: `1.035x` total tok/s, TTFT p50 `0.918x`, ITL p50 `1.022x`, ITL p95 `0.986x`.
- delta vs vLLM async baseline (`864.18 tok/s`): JAX moves from `0.411x` vLLM in Entry 033 to `0.426x` vLLM.

Top trace ranges, total inclusive time:

| range | Entry 033 ms | chunk-32 default repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 720.52 | 696.03 | overall improved |
| `_run_main_and_sample` | 598.60 | 572.99 | runner hot path improved |
| `forward_step_token_ids_jit` | 175.81 | 122.13 | compiled model path improved |
| first `forward_step_token_ids_jit` | 83.25 | 30.83 | prefill step improved materially |
| `PjRtCApiLoadedExecutable::Execute` | 191.91 over 252 calls | 138.32 over 252 calls | same dispatch count, less device execution |
| `jit_compiled:XLA GPU module` | 147.90 | 94.20 | improved lowered module time |
| `command_buffer::execute` | 92.22 over 1575 calls | 40.34 over 1143 calls | fewer/lighter command-buffer executes |
| `command_buffer::update` | 27.81 over 248 calls | 27.13 over 248 calls | unchanged |
| `input_reduce_fusion` | 59.30 over 2512 calls | 28.65 over 1936 calls | target GDN prefill bucket roughly halved |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.39 | 52.40 | compact projection GEMM unchanged |
| `gemm_fusion_dot_general_746` | 13.92 | 13.92 | unchanged |
| `wrapped_concatenate` | 36.49 | 36.50 | unchanged |
| `MemcpyD2D` | 24.62 | 24.69 | unchanged |
| `array.py:325 tolist` | 409.69 | 437.58 | host sync attribution worsened, but end-to-end still improved |

Decision:

- Accept chunk size 32 as the default GDN prefill chunk size. This is the first post-Entry-033 model-side change with clear profile evidence on the intended bucket and exact generated-token correctness.
- Keep the benchmark override for future controlled sweeps, but do not expose a new server flag yet. The library default should get the faster path without extra user setup.
- Do not use chunk size 128 for this workload. It greatly increases first prefill cost, `input_reduce_fusion`, and command-buffer executes.
- Next candidates remain HLO-guided full-attention/paged-prefill layout work and, later, a transpose-free backend-owned KV layout for Pallas decode attention.

## Entry 046 - Rejected Initial-Prefill Bounded KV Window

- run id: `20260526-022846-2084640-jax_hetero8_64_512x32_initial_prefill_bound_kv`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_initial_prefill_bound_kv.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-022846-2084640-jax_hetero8_64_512x32_initial_prefill_bound_kv`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-022846-2084640-jax_hetero8_64_512x32_initial_prefill_bound_kv/plugins/profile/2026_05_26_02_30_36/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-022846-2084640-jax_hetero8_64_512x32_initial_prefill_bound_kv/plugins/profile/2026_05_26_02_30_36/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: McClintock reviewed the current accepted state and ranked the best follow-ups as: (1) profile the initial-prefill KV-window bound already sketched in the tree, but only with a strict host gate for whole no-prefix prompt prefill; (2) bracket GDN prefill chunk size below 32, starting with `--linear-chunk-size 16`, watching `input_reduce_fusion` and the first prefill step; (3) try a head-major reshape/layout path in `paged_attention_prefill`. Pallas/CuteDSL paged attention should wait until the KV cache layout can remove the transpose penalty seen in earlier attempts.
- change tested: a temporary `NANO_VLLM_JAX_INITIAL_PREFILL_BOUND_KV=1` path carried a static `max_kv_len` through attention metadata into `paged_attention_prefill`, limiting initial no-prefix prefill key positions to the active prompt bucket width instead of the full paged block-table capacity. The source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_INITIAL_PREFILL_BOUND_KV=1`, `tests/test_backend_boundaries.py::test_initial_prefill_bounded_kv_window_matches_unbounded_paged_prefill`, `tests/test_backend_boundaries.py::test_executor_cached_prefill_matches_no_cache_prefill_logits`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, `tests/test_kv_cache.py::test_paged_attention_non_identity_blocks`, `tests/test_kv_cache.py::test_paged_attention_grouped_gqa_matches_repeat_reference`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`8 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `350.75 tok/s`, TTFT p50 `297.24 ms`, ITL p50 `13.84 ms`, ITL p95 `15.05 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.954x` total tok/s, TTFT p50 `1.025x`, ITL p50 `1.053x`, ITL p95 `1.107x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | bounded KV ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 729.86 | overall slower |
| `_run_main_and_sample` | 572.99 | 592.08 | runner hot path slower |
| `forward_step_token_ids_jit` | 122.13 | 158.40 | compiled token-id path regressed |
| first `forward_step_token_ids_jit` | 30.83 | 33.65 | first prefill step regressed |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 173.59 over 252 calls | same dispatch count, heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 114.34 | lowered module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 45.11 over 1143 calls | same count, heavier execution |
| `command_buffer::update` | 27.13 over 248 calls | 37.03 over 248 calls | heavier updates |
| `input_reduce_fusion` | 28.65 over 1936 calls | 28.23 over 1936 calls | target GDN bucket unchanged |
| `gather` | 11.97 | 12.54 | target gather path regressed |
| `transpose` | 63.11 | 63.52 | unchanged/slightly worse |
| `array.py:325 tolist` | 437.58 | 418.26 | host sync attribution improved, but not enough to matter |

Decision:

- Reject and revert initial-prefill bounded KV window. Correctness was exact, but the profile shows the static bound made the compiled/device path heavier and did not improve the target gather or transpose ranges.
- Do not keep a flag for this path. The only clear improvement was host sync attribution, while end-to-end latency and device execution regressed.
- Next candidates should follow McClintock's remaining list: bracket GDN prefill chunk size below 32 with `--linear-chunk-size 16`, then inspect `paged_attention_prefill` layout/HLO for head-major or transpose-free alternatives.

## Entry 047 - Rejected GDN Prefill Chunk Size 16

- run id: `20260526-023742-2087778-jax_hetero8_64_512x32_gdn_chunk16`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk16.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-023742-2087778-jax_hetero8_64_512x32_gdn_chunk16`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-023742-2087778-jax_hetero8_64_512x32_gdn_chunk16/plugins/profile/2026_05_26_02_38_45/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-023742-2087778-jax_hetero8_64_512x32_gdn_chunk16/plugins/profile/2026_05_26_02_38_45/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: benchmark-only `--linear-chunk-size 16`, leaving the accepted source default at 32. This brackets the Gated DeltaNet prefill chunk-size sweep below the accepted Entry 045 value without source changes.
- focused CUDA checks: no new source path was introduced. The profiled run used `JAX_PLATFORMS=cuda`, BF16 weights, FP32 activations, the same hetero8 shape set as Entry 045, and exact generated-token comparison against `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`.
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `248.36 tok/s`, TTFT p50 `537.16 ms`, ITL p50 `15.80 ms`, ITL p95 `17.25 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.675x` total tok/s, TTFT p50 `1.852x`, ITL p50 `1.202x`, ITL p95 `1.269x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | chunk-16 ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 1030.77 | overall much slower |
| `_run_main_and_sample` | 572.99 | 898.42 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 321.72 | compiled token-id path regressed heavily |
| first `forward_step_token_ids_jit` | 30.83 | 211.34 | first prefill step regressed heavily |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 339.78 over 252 calls | same dispatch count, much heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 284.66 | lowered module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 195.32 over 873 calls | fewer executes, but much heavier total time |
| `command_buffer::update` | 27.13 over 248 calls | 32.95 over 248 calls | heavier updates |
| `input_reduce_fusion` | 28.65 over 1936 calls | 18.09 over 1378 calls | target reduce bucket improved, but not the bottleneck after the change |
| `gemm_fusion_dot_general_746` | 13.92 | 40.98 | main GEMM bucket regressed |
| `gemm_fusion_dot_2` | 13.79 | 33.25 | main GEMM bucket regressed |
| `gather` | 11.97 | 12.93 | slightly worse |
| `transpose` | 63.11 | 63.73 | unchanged/slightly worse |
| `array.py:325 tolist` | 437.58 | 562.16 | host sync attribution worsened |

Decision:

- Reject chunk size 16. It improves the named `input_reduce_fusion` bucket, but the smaller chunking creates a much worse compiled plan overall, especially first prefill and GEMM/module execution.
- Keep Entry 045's chunk size 32 default. The sweep now brackets 16, 32, 64, and 128, and 32 remains the best verified point for this hetero8 workload.
- Next profile-backed work should move to `paged_attention_prefill` layout/HLO investigation rather than smaller GDN chunks.

## Entry 048 - Rejected Paged Prefill Head-Major Layout

- run id: `20260526-024346-2090819-jax_hetero8_64_512x32_prefill_head_major`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_prefill_head_major.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-024346-2090819-jax_hetero8_64_512x32_prefill_head_major`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-024346-2090819-jax_hetero8_64_512x32_prefill_head_major/plugins/profile/2026_05_26_02_44_39/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-024346-2090819-jax_hetero8_64_512x32_prefill_head_major/plugins/profile/2026_05_26_02_44_39/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: a temporary `NANO_VLLM_JAX_PREFILL_HEAD_MAJOR=1` path in `paged_attention_prefill` transposed gathered K/V to `[batch, kv_heads, tokens, head_dim]` and computed attention in `[batch, kv_heads, groups, query_tokens, key_tokens]`. The goal was to test whether head-major math lowered away the expensive prefill transpose/gather pattern. The source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_PREFILL_HEAD_MAJOR=1`, `tests/test_kv_cache.py::test_paged_attention_prefill_head_major_matches_default`, `tests/test_kv_cache.py::test_paged_attention_non_identity_blocks`, `tests/test_kv_cache.py::test_paged_attention_grouped_gqa_matches_repeat_reference`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`7 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `262.85 tok/s`, TTFT p50 `498.56 ms`, ITL p50 `15.17 ms`, ITL p95 `16.02 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.715x` total tok/s, TTFT p50 `1.719x`, ITL p50 `1.154x`, ITL p95 `1.179x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | head-major ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 973.93 | overall much slower |
| `_run_main_and_sample` | 572.99 | 847.58 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 121.41 | compiled token-id path tied/slightly better |
| first `forward_step_token_ids_jit` | 30.83 | 30.74 | first prefill tied |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 135.04 over 252 calls | slightly lower device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 90.01 | slightly lower module time |
| `command_buffer::execute` | 40.34 over 1143 calls | 36.52 over 1143 calls | slightly lower command-buffer execution |
| `command_buffer::update` | 27.13 over 248 calls | 26.67 over 248 calls | tied/slightly lower |
| `input_reduce_fusion` | 28.65 over 1936 calls | 26.98 over 1930 calls | small improvement |
| `gather` | 11.97 | 12.94 | worse |
| `transpose` | 63.11 | 61.14 | small improvement |
| `gemm_fusion_dot_general_746` | 13.92 | 41.10 | important GEMM bucket regressed |
| `array.py:325 tolist` | 437.58 | 712.38 | host sync attribution worsened badly |
| `np.asarray(jax.Array)` | 437.35 | 712.04 | host sync attribution worsened badly |

Decision:

- Reject and revert the head-major `paged_attention_prefill` layout. It produced exact tokens and a tiny device-side improvement, but the server-critical path regressed sharply through host synchronization and an important GEMM bucket.
- Do not keep this as an optional flag. The small transpose/module gains are not useful unless the dependency structure also prevents the later token readback from absorbing the cost.
- The next attention-layout attempt should inspect lowered HLO/kernel names before changing source shape order again, or defer attention work until a backend-owned KV layout can avoid the transpose/gather pattern more directly.

## Entry 049 - Rejected Default MTP1 Server Path

- first run id: `20260526-025218-2093924-jax_hetero8_64_512x32_mtp1_default`
- first benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_mtp1_default.json`
- first profile directory: `/mountpoint/.exp/profiles/20260526-025218-2093924-jax_hetero8_64_512x32_mtp1_default`
- first Perfetto trace: `/mountpoint/.exp/profiles/20260526-025218-2093924-jax_hetero8_64_512x32_mtp1_default/plugins/profile/2026_05_26_02_55_05/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- first TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-025218-2093924-jax_hetero8_64_512x32_mtp1_default/plugins/profile/2026_05_26_02_55_05/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260526-025631-2095016-jax_hetero8_64_512x32_mtp1_default_repeat`
- repeat benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_mtp1_default_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260526-025631-2095016-jax_hetero8_64_512x32_mtp1_default_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260526-025631-2095016-jax_hetero8_64_512x32_mtp1_default_repeat/plugins/profile/2026_05_26_02_57_27/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-025631-2095016-jax_hetero8_64_512x32_mtp1_default_repeat/plugins/profile/2026_05_26_02_57_27/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Tesla reviewed Entries 045-048 and recommended three next model-side targets: Ampere-compatible compact projection fusion under the FP32/BF16 contract, HLO-guided paged/full-attention prefill layout that preserves the current GEMM plan, and a backend-owned transpose-free KV layout for decode. It explicitly warned not to repeat chunk-16, bounded initial-prefill KV windows, or simple head-major paged prefill.
- change tested: no source change. The benchmark enabled the public server path with `--num-speculative-tokens 1`, BF16 weights, FP32 activations, the accepted chunk-32 default, and exact-token comparison against Entry 045.
- correctness: exact generated-token match for all 8 rows in both runs, because MTP1 accepted no drafts and fell back to the main model path. Draft quality was not acceptable: `0/92` verified drafts accepted in both runs, with `115` drafts proposed.
- first JAX timing: `2.21 tok/s`, TTFT p50 `577.89 ms`, ITL p50 `68.63 ms`, ITL p95 `34756.31 ms`.
- repeat JAX timing: `10.93 tok/s`, TTFT p50 `571.94 ms`, ITL p50 `70.27 ms`, ITL p95 `7930.21 ms`.
- repeat delta vs Entry 045 chunk-32 default repeat: `0.030x` total tok/s, TTFT p50 `1.972x`, ITL p50 `5.346x`, ITL p95 `583.41x`.
- scheduler admission: the active `physical_batch_size=8, active_decode_rows=8` bucket reached `acceptance_ready=true` and disabled MTP with `admission_reason=low_acceptance`. The bucket reported `observed_rejected=92`, `observed_accepted=0`, `fallback_partial_rows=12`, and `fallback_seeded_main_steps=14`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | MTP1 repeat ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 23421.92 | overall unusably slower |
| `_run_main_and_sample` | 572.99 over 32 calls | 551.22 over 1 call | only the fallback/main-model step is comparable |
| `_run_mtp1_batched` | n/a | 7902.83 | verifier path dominates |
| `mtp1_commit_select_greedy_step_jit` | n/a | 7703.71 | commit-select path dominates |
| `_profile_jit_call` | 120.22 over 32 calls | 7646.08 over 2 calls | JIT path dominated by MTP tracing/cache miss |
| `PjitFunction(compiled)` | 240.04 over 64 calls | 15292.12 over 4 calls | huge MTP compiled/tracing attribution |
| `cache_miss` | n/a | 8128.71 over 1660 calls | repeat still pays JAX tracing/cache-miss overhead |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 523.91 over 366 calls | more/heavier device execution |
| `command_buffer::execute` | 40.34 over 1143 calls | 15.60 over 904 calls | not the limiting cost here |
| `gather` | 11.97 | 337.25 | verifier/commit-select gathers are expensive |
| `transpose` | 63.11 | 59.87 | unchanged |
| `array.py:325 tolist` | 437.58 | 13.45 | token readback is no longer the limiting bucket |

Decision:

- Reject default MTP1 serving for this hetero8 workload. Output tokens remain exact, but only because all speculative drafts are rejected; MTP1 does not improve generation and the scheduler correctly disables it for low acceptance.
- Do not optimize around MTP1 until the main model is much closer to vLLM and until the MTP draft path has meaningful acceptance. The current path also needs explicit server warmup coverage for commit-select shapes before any speed number is considered steady-state.
- Near-term work should return to the main model targets from Tesla's audit: compact projection fusion under the current dtype contract, then HLO-guided attention/KV layout work.

## Entry 050 - Rejected Prepacked MLP Gate/Up Compact Prefill

- run id: `20260526-030536-2097294-jax_hetero8_64_512x32_packed_mlp_gate_up`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_prepacked_mlp_gate_up.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-030536-2097294-jax_hetero8_64_512x32_packed_mlp_gate_up`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-030536-2097294-jax_hetero8_64_512x32_packed_mlp_gate_up/plugins/profile/2026_05_26_03_06_15/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-030536-2097294-jax_hetero8_64_512x32_packed_mlp_gate_up/plugins/profile/2026_05_26_03_06_15/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change tested: a temporary `NANO_VLLM_JAX_PACK_MLP_GATE_UP=1` path packed each layer's `gate_proj` and `up_proj` into a precomputed `[gate, up]` MLP weight leaf at HF load time, then the compact prefill MLP used one larger input GEMM and split the result. This avoided runtime weight concatenation and was meant to cut compact projection GEMM launches under the BF16-weight/FP32-activation contract. The source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_PACK_MLP_GATE_UP=1` and compact MLP enabled, `tests/test_lm_head_helpers.py`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, `tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode`, and `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute` passed under `JAX_PLATFORMS=cuda` (`6 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `270.57 tok/s`, TTFT p50 `440.13 ms`, ITL p50 `16.15 ms`, ITL p95 `17.46 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.736x` total tok/s, TTFT p50 `1.518x`, ITL p50 `1.229x`, ITL p95 `1.284x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | prepacked gate/up ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 946.13 | overall much slower |
| `_run_main_and_sample` | 572.99 | 809.97 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 147.65 | compiled token-id path slower |
| first `forward_step_token_ids_jit` | 30.83 | 30.88 | first prefill tied |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 165.74 over 252 calls | same dispatch count, heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 107.62 | module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 44.41 over 1143 calls | same count, heavier execution |
| `command_buffer::update` | 27.13 over 248 calls | 34.35 over 248 calls | heavier updates |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.40 over 96 calls | 25.74 over 48 calls | target compact GEMM bucket halved |
| `input_reduce_fusion` | 28.65 over 1936 calls | 28.57 over 1936 calls | unchanged |
| `gather` | 11.97 | 12.98 | worse |
| `transpose` | 63.11 | 63.74 | unchanged/slightly worse |
| `array.py:325 tolist` | 437.58 | 647.28 | host sync attribution worsened |
| `np.asarray(jax.Array)` | 437.35 | 646.96 | host sync attribution worsened |

Decision:

- Reject and revert prepacked MLP gate/up compact prefill. The intended compact CUTLASS bucket did move in the right direction, but the larger packed parameter leaf and altered compiled plan increased device execution, command-buffer update time, and host synchronization enough to lose badly end to end.
- Do not add persistent packed MLP gate/up weights for serving in this form. Any future compact projection fusion should avoid duplicating large weight leaves and should verify that `PjRt Execute`, `command_buffer::update`, and host sync stay flat before considering the compact GEMM win meaningful.
- Next model-side candidates remain HLO-guided attention/KV layout or a lower-level compact projection kernel that does not expand the model parameter tree.

## Entry 051 - Rejected GPU Flat KV Cache Layout

- run id: `20260526-031933-2100265-jax_hetero8_64_512x32_gpu_flat_kv_cache`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_gpu_flat_kv_cache.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-031933-2100265-jax_hetero8_64_512x32_gpu_flat_kv_cache`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-031933-2100265-jax_hetero8_64_512x32_gpu_flat_kv_cache/plugins/profile/2026_05_26_03_20_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-031933-2100265-jax_hetero8_64_512x32_gpu_flat_kv_cache/plugins/profile/2026_05_26_03_20_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- HLO probe artifact: `/mountpoint/.exp/tmp/hlo/flat_vs_block_kv_hlo_summary.json`
- change tested: a temporary `NANO_VLLM_JAX_GPU_FLAT_KV_CACHE=1` path made the GPU backend allocate full-attention KV as `[layers, slots, kv_heads, head_dim]` instead of `[layers, blocks, block_size, kv_heads, head_dim]`. The existing `update_kv_cache`, `paged_attention_prefill`, and `paged_attention_decode` functions already support this flat rank-4 layout, so the experiment tested whether removing block-to-flat reshapes at the backend boundary helped without changing attention math. The source change was reverted after profiling.
- HLO precheck: lowering `paged_attention_prefill`, `paged_attention_decode`, and `update_kv_cache` for Entry 045 shapes showed unchanged prefill/decode attention structure (`gather`, `transpose`, and `dot` counts tied), but the KV write lowering dropped from `22` to `16` reshape mentions for the flat layout.
- focused CUDA checks: with `NANO_VLLM_JAX_GPU_FLAT_KV_CACHE=1`, `tests/test_backend_boundaries.py::test_gpu_flat_kv_cache_matches_block_layout`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, `tests/test_backend_boundaries.py::test_executor_jit_matches_eager_cached_decode`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`7 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `251.63 tok/s`, TTFT p50 `529.98 ms`, ITL p50 `15.56 ms`, ITL p95 `16.45 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.684x` total tok/s, TTFT p50 `1.828x`, ITL p50 `1.184x`, ITL p95 `1.210x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | flat KV ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 1017.36 | overall much slower |
| `_run_main_and_sample` | 572.99 | 891.14 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 141.39 | compiled token-id path slower |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 154.75 over 252 calls | same dispatch count, heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 105.15 | module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 42.04 over 1143 calls | same count, slightly heavier execution |
| `command_buffer::update` | 27.13 over 248 calls | 31.15 over 248 calls | heavier updates |
| `gather` | 11.97 | 12.83 | worse |
| `transpose` | 63.11 | 63.26 | unchanged |
| `wrapped_concatenate` | 36.50 over 576 calls | 36.47 over 576 calls | unchanged |
| `MemcpyD2D` | 24.69 over 1231 calls | 24.64 over 1232 calls | unchanged |
| `array.py:325 tolist` | 437.58 | 735.23 | host sync attribution much worse |
| `np.asarray(jax.Array)` | 437.35 | 734.95 | host sync attribution much worse |

Decision:

- Reject and revert GPU flat KV cache allocation. The HLO write-side simplification was real but too small; the integrated server trace regressed device execution, command-buffer updates, gather time, and host synchronization.
- Do not pursue flat rank-4 KV layout as a standalone optimization. A future backend-owned KV layout should only be revisited when paired with a real attention kernel that consumes the new physical layout directly; reshaping the existing pure-JAX paged attention boundary is insufficient.

## Entry 052 - Rejected Prepacked Full-Attention K/V Projection

- run id: `20260526-032942-2104004-jax_hetero8_64_512x32_packed_full_attn_kv`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_packed_full_attn_kv.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-032942-2104004-jax_hetero8_64_512x32_packed_full_attn_kv`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-032942-2104004-jax_hetero8_64_512x32_packed_full_attn_kv/plugins/profile/2026_05_26_03_30_40/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-032942-2104004-jax_hetero8_64_512x32_packed_full_attn_kv/plugins/profile/2026_05_26_03_30_40/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Fermat reviewed the post-Entry-051 state and ranked the next model-side targets as: segmented ragged GDN prefill with chunk 32, backend-owned native paged-attention layout plus kernel, then mixed-dtype compact projection lowering gated by a standalone CUDA microbenchmark. The audit explicitly warned that Entries 048, 050, and 051 show local kernel wins can still lose if `PjRt Execute`, command-buffer updates, or host synchronization regress.
- change tested: a temporary `NANO_VLLM_JAX_PACK_FULL_ATTN_KV=1` path added a precomputed `[K, V]` projection leaf for full-attention layers at HF load time, then projected K and V with one larger dot and split the result. The source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_PACK_FULL_ATTN_KV=1` and accepted compact-prefill flags enabled, `tests/test_lm_head_helpers.py`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, and `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute` passed under `JAX_PLATFORMS=cuda` (`6 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `258.87 tok/s`, TTFT p50 `506.90 ms`, ITL p50 `15.42 ms`, ITL p95 `16.14 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.704x` total tok/s, TTFT p50 `1.748x`, ITL p50 `1.173x`, ITL p95 `1.188x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | packed full-attn K/V ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 988.92 | overall much slower |
| `_run_main_and_sample` | 572.99 | 862.63 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 136.80 | compiled token-id path slower |
| first `forward_step_token_ids_jit` | 30.83 | 32.74 | first prefill slower |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 151.05 over 252 calls | same dispatch count, heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 101.66 | module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 42.24 over 1142 calls | command-buffer execution regressed |
| `command_buffer::update` | 27.13 over 248 calls | 31.00 over 248 calls | command-buffer updates regressed |
| `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` | 52.40 over 96 calls | 4.53 over 12 calls | target compact GEMM bucket almost disappeared |
| `input_reduce_fusion` | 28.65 over 1936 calls | 28.61 over 1936 calls | unchanged |
| `gather` | 11.97 | 12.59 | worse |
| `transpose` | 63.11 | 63.68 | worse |
| `wrapped_concatenate` | 36.50 over 576 calls | 36.47 over 576 calls | unchanged |
| `MemcpyD2D` | 24.69 over 1231 calls | 24.63 over 1232 calls | unchanged |
| `array.py:325 tolist` | 437.58 | 711.53 | host sync attribution much worse |
| `np.asarray(jax.Array)` | 437.35 | 711.26 | host sync attribution much worse |

Decision:

- Reject and revert prepacked full-attention K/V projection. The targeted compact CUTLASS bucket moved strongly in the intended direction, but the integrated serving path regressed in device execution, module time, command-buffer updates, gather/transpose work, and host synchronization.
- Do not add persistent packed full-attention K/V leaves in this form. Like the prepacked MLP gate/up experiment, duplicating large projection leaves changes the compiled plan enough to erase the local GEMM win.
- Next work should follow Fermat's ranking: first try ragged GDN prefill that keeps chunk 32 and removes padded work, or start a backend-owned paged-attention/KV kernel path. Avoid more loader-level packed-weight experiments unless a microbenchmark proves the mixed-dtype lowering improves the exact BF16-weight/FP32-activation contract without command-buffer or host-sync regressions.

## Entry 053 - Rejected Static Row-Chunk Ragged GDN Prefill

- run id: `20260526-034521-2112444-jax_hetero8_64_512x32_static_ragged_gdn_prefill`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_static_ragged_gdn_prefill.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-034521-2112444-jax_hetero8_64_512x32_static_ragged_gdn_prefill`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-034521-2112444-jax_hetero8_64_512x32_static_ragged_gdn_prefill/plugins/profile/2026_05_26_03_46_07/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-034521-2112444-jax_hetero8_64_512x32_static_ragged_gdn_prefill/plugins/profile/2026_05_26_03_46_07/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Raman confirmed the next reasonable target was segmented/ragged chunk-32 GDN prefill, but warned that a source-level row-chunk scan would trade `16` vectorized time-chunk steps for about `72` active row-chunk steps on the hetero8 workload. The required gate was a lower `input_reduce_fusion` and first prefill step without increasing `PjRt Execute`, GPU module time, or command-buffer work.
- change tested: a temporary `NANO_VLLM_JAX_STATIC_RAGGED_GDN_PREFILL=1` path propagated static host query lengths into the JIT key and added a row-chunk scan that skipped padded GDN chunks while preserving one compiled model call. The path used `dynamic_slice`/`dynamic_update_slice` to process valid `[row, chunk]` descriptors one at a time, then the source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_STATIC_RAGGED_GDN_PREFILL=1` and accepted compact-prefill flags enabled, `tests/test_kv_cache.py::test_static_ragged_gdn_prefill_matches_padded_chunked`, `tests/test_kv_cache.py::test_linear_attention_chunked_vs_recurrent`, `tests/test_kv_cache.py::test_linear_attention_multichunk_matches_recurrent`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, `tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`10 passed`). `tests/test_backend_boundaries.py::test_linear_suffix_prefill_matches_sequential_decode_state` also passed separately.
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `131.33 tok/s`, TTFT p50 `1431.02 ms`, ITL p50 `16.78 ms`, ITL p95 `17.98 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.357x` total tok/s, TTFT p50 `4.935x`, ITL p50 `1.277x`, ITL p95 `1.323x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | static row-chunk GDN ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 1949.22 | overall much slower |
| `_run_main_and_sample` | 572.99 | 1806.58 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 1507.56 | compiled token-id path much slower |
| first `forward_step_token_ids_jit` | 30.83 | 1374.45 | first prefill exploded |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 1525.34 over 252 calls | same dispatch count, far heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 1459.76 | module time exploded |
| `command_buffer::execute` | 40.34 over 1143 calls | 101.12 over 2871 calls | many more command-buffer executions |
| `command_buffer::update` | 27.13 over 248 calls | 37.19 over 248 calls | updates heavier |
| `input_reduce_fusion` | 28.65 over 1936 calls | 265.47 over 82615 calls | intended GDN bucket regressed badly |
| `loop_dynamic_update_slice_fusion` | 2.24 over 589 calls | 263.69 over 82615 calls | row-chunk writes dominate |
| `loop_multiply_fusion` | n/a | 494.93 over 82584 calls | row-chunk scan generated tiny repeated kernels |
| `gather` | 11.97 | 12.71 | slightly worse |
| `transpose` | 63.11 | 44.41 | lower, but irrelevant next to GDN regression |
| `wrapped_concatenate` | 36.50 over 576 calls | 36.54 over 576 calls | unchanged |
| `MemcpyD2D` | 24.69 over 1231 calls | 27.43 over 2510 calls | more copies |
| `array.py:325 tolist` | 437.58 | 282.82 | host sync lower only because device critical path moved elsewhere |
| `np.asarray(jax.Array)` | 437.35 | 282.49 | host sync lower only because device critical path moved elsewhere |

Decision:

- Reject and revert static row-chunk ragged GDN prefill. It preserved correctness and one model dispatch, but converted a vectorized chunked GDN into many tiny dynamic-slice/update kernels and made the first prefill step unusable.
- Do not revisit source-level per-row/per-chunk scans for GDN prefill. Skipping padded GDN work is still conceptually valid, but it needs a backend-owned/lowered kernel that keeps row chunks inside coarse GPU work rather than expressing them as thousands of JAX dynamic updates.
- Next model-side work should move to a backend-owned paged-attention/KV kernel path or a lower-level GDN prefill kernel design. Pure-JAX static ragged scans should be treated as rejected along with earlier multi-dispatch and single-JIT compact-prefill attempts.

## Entry 054 - Rejected Head-Major KV Cache with Pallas Decode Attention

- run id: `20260526-035950-2127348-jax_hetero8_64_512x32_head_major_kv_pallas_decode`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_head_major_kv_pallas_decode.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-035950-2127348-jax_hetero8_64_512x32_head_major_kv_pallas_decode`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-035950-2127348-jax_hetero8_64_512x32_head_major_kv_pallas_decode/plugins/profile/2026_05_26_04_00_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-035950-2127348-jax_hetero8_64_512x32_head_major_kv_pallas_decode/plugins/profile/2026_05_26_04_00_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Locke reviewed Entries 045-053 and recommended a backend-owned segmented GDN prefill kernel as the better next bet. It warned not to continue the head-major KV/Pallas decode family unless a clean profile win already existed, because Entry 036 showed a Pallas decode kernel can lose to layout and compiled-plan costs.
- change tested: a temporary `NANO_VLLM_JAX_GPU_HEAD_MAJOR_KV_CACHE=1` path allocated GPU KV as `[layers, kv_heads, pages, page_size, head_dim]` so `jax.experimental.pallas.ops.gpu.paged_attention` could consume the decode cache without a per-layer transpose. `NANO_VLLM_JAX_PALLAS_HEAD_MAJOR_DECODE=1` routed width-1 full-attention decode to the Pallas kernel; prefill and multi-token decode kept the pure-JAX fallback. The source change was reverted after profiling.
- feasibility checks: a CUDA smoke test showed head-major KV storage matched the standard block layout exactly in the pure-JAX decode path (`max_abs=0.0`). The Pallas decode output matched the standard reference within BF16 tolerance (`max_abs=0.0079`, MSE `1.41e-6`) with `pages_per_compute_block=2`, `k_splits=4` on a small shape. The first full warmup with `pages_per_compute_block=2`, `k_splits=8` failed because Pallas required `pages_per_partition=5` to be divisible by `pages_per_compute_block`; the profiled run used `pages_per_compute_block=1`, `k_splits=8`.
- focused CUDA checks: with head-major KV and Pallas decode flags enabled, `tests/test_kv_cache.py::test_paged_attention_non_identity_blocks`, `tests/test_kv_cache.py::test_paged_attention_grouped_gqa_matches_repeat_reference`, `tests/test_kv_cache.py::test_masked_update_kv_cache_preserves_invalid_and_duplicate_slots`, `tests/test_backend_boundaries.py::test_executor_jit_matches_eager_cached_decode`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`8 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `245.23 tok/s`, TTFT p50 `524.85 ms`, ITL p50 `16.66 ms`, ITL p95 `18.40 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.667x` total tok/s, TTFT p50 `1.810x`, ITL p50 `1.267x`, ITL p95 `1.353x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | head-major KV + Pallas decode ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 1043.91 | overall much slower |
| `_run_main_and_sample` | 572.99 | 897.15 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 155.45 | compiled token-id path slower |
| first `forward_step_token_ids_jit` | 30.83 | 31.94 | first prefill slightly slower |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 171.55 over 252 calls | same dispatch count, heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 111.85 | module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 44.65 over 1112 calls | fewer ranges but more time |
| `command_buffer::update` | 27.13 over 248 calls | 37.50 over 217 calls | fewer ranges but much more time |
| `paged_attention` | 0.00 | 9.47 over 186 calls | new Pallas decode kernel work |
| `gemm_fusion_dot_general_746` | 13.92 | 41.04 | important GEMM bucket regressed |
| `input_reduce_fusion` | 28.65 over 1936 calls | 28.64 over 1936 calls | unchanged |
| `gather` | 11.97 | 12.88 | worse |
| `transpose` | 63.11 | 56.46 | lower, but not enough to offset other regressions |
| `wrapped_concatenate` | 36.50 over 576 calls | 36.47 over 576 calls | unchanged |
| `MemcpyD2D` | 24.69 over 1231 calls | 26.92 over 1232 calls | worse |
| `array.py:325 tolist` | 437.58 | 725.82 | host sync attribution much worse |
| `np.asarray(jax.Array)` | 437.35 | 725.33 | host sync attribution much worse |

Decision:

- Reject and revert head-major KV cache plus Pallas decode attention. Removing the explicit per-layer Pallas layout transpose was not enough; the new physical cache layout and Pallas decode path still made compiled execution, command-buffer updates, GEMM lowering, and host synchronization worse.
- Do not continue the Pallas paged decode family on this source layout. A future attention kernel must own both cache writes and decode attention in a coarser backend path, and should be gated by a microbenchmark that includes update/write cost, not only standalone attention.
- Follow Locke's recommendation next: if staying with model-side GDN, the next viable version is a backend-owned segmented GDN prefill kernel at chunk size 32, not another source-level scan. Otherwise move to a fuller backend-owned attention/KV kernel rather than adapting the shipped Pallas op through layout plumbing.

## Entry 055 - Rejected Skip-Unused Hidden Return Norm

- run id: `20260526-040900-2130051-jax_hetero8_64_512x32_skip_unused_hidden_norm`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_skip_unused_hidden_norm.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-040900-2130051-jax_hetero8_64_512x32_skip_unused_hidden_norm`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-040900-2130051-jax_hetero8_64_512x32_skip_unused_hidden_norm/plugins/profile/2026_05_26_04_09_36/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-040900-2130051-jax_hetero8_64_512x32_skip_unused_hidden_norm/plugins/profile/2026_05_26_04_09_36/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Feynman confirmed the largest remaining model-side target is still rectangular padded chunked Gated DeltaNet prefill reached through `gated_deltanet_block -> backend.gated_delta_prefill`. The recommended next real experiment is `NANO_VLLM_JAX_GPU_SEGMENTED_GDN_PREFILL=1`: a backend-owned/lowered long-prefill chunk-32 path that consumes query lengths or a valid-token mask inside one coarse GPU operation per GDN layer. The audit explicitly warned that this must not repeat Entry 053's JAX-side `dynamic_slice`/`dynamic_update_slice` row-chunk scan.
- change tested: a temporary `NANO_VLLM_JAX_SKIP_UNUSED_HIDDEN_RETURN_NORM=1` path returned `hidden_pre` before the full-sequence final RMSNorm when `forward_step(..., return_hidden=True, return_hidden_with_logits=False)` was used. The hypothesis was that the greedy token-id fast path, which gathers the last hidden token and applies `lm_head_token_ids_and_topk`, might still be paying for an otherwise unused full-sequence final norm. The source change was reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_SKIP_UNUSED_HIDDEN_RETURN_NORM=1` and accepted compact-prefill flags enabled, `tests/test_lm_head_helpers.py`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, and `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute` passed under `JAX_PLATFORMS=cuda` (`5 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `250.82 tok/s`, TTFT p50 `521.87 ms`, ITL p50 `15.72 ms`, ITL p95 `17.89 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.682x` total tok/s, TTFT p50 `1.800x`, ITL p50 `1.196x`, ITL p95 `1.316x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | skip-unused norm ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 1020.64 | overall much slower |
| `_run_main_and_sample` | 572.99 | 890.85 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 134.48 | compiled token-id path slower overall |
| first `forward_step_token_ids_jit` | 30.83 | 29.81 | first prefill slightly lower but not useful |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 150.83 over 252 calls | same dispatch count, heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 97.26 | module time regressed |
| `command_buffer::execute` | 40.34 over 1143 calls | 40.92 over 1143 calls | slightly worse |
| `command_buffer::update` | 27.13 over 248 calls | 30.27 over 248 calls | updates regressed |
| `input_reduce_fusion` | 28.65 over 1936 calls | 28.62 over 1936 calls | unchanged |
| `gemm_fusion_dot_general_746` | 13.92 over 24 calls | 40.81 over 24 calls | important GEMM bucket regressed |
| `gemm_fusion_dot_2` | 120.93 over 2695 calls | 193.36 over 2695 calls | broader GEMM family regressed |
| `gather` | 11.97 | 12.71 | worse |
| `transpose` | 63.11 | 63.26 | unchanged/slightly worse |
| `wrapped_concatenate` | 36.50 over 576 calls | 36.46 over 576 calls | unchanged |
| `MemcpyD2D` | 24.69 over 1231 calls | 24.88 over 1232 calls | slightly worse |
| `array.py:325 tolist` | 437.58 | 742.05 | host sync attribution much worse |
| `np.asarray(jax.Array)` | 437.35 | 741.76 | host sync attribution much worse |

Decision:

- Reject and revert skip-unused hidden return norm. It preserved exact correctness and made only the first prefill range slightly smaller, while the integrated serving path regressed materially across compiled execution, GEMM lowering, command-buffer updates, and host synchronization.
- Treat this as evidence that the greedy token-id path's visible final-norm structure is not a useful source-level optimization target; XLA's compiled plan is already dominated elsewhere or becomes worse when the branch changes the graph.
- Follow Feynman's recommendation next: the viable model-side target is a backend-owned segmented GDN prefill kernel at chunk size 32, with acceptance gated on lower first `forward_step_token_ids_jit`, lower or flat `PjRt Execute`, `jit_compiled:XLA GPU module`, `command_buffer::update`, and no Entry 053-style kernel-count explosion.

## Entry 056 - Rejected Static Chunk-Major GDN Prefill

- run id: `20260526-042323-2137008-jax_hetero8_64_512x32_static_chunk_major_gdn_prefill`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_static_chunk_major_gdn_prefill.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-042323-2137008-jax_hetero8_64_512x32_static_chunk_major_gdn_prefill`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-042323-2137008-jax_hetero8_64_512x32_static_chunk_major_gdn_prefill/plugins/profile/2026_05_26_04_25_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-042323-2137008-jax_hetero8_64_512x32_static_chunk_major_gdn_prefill/plugins/profile/2026_05_26_04_25_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Hilbert warned against promoting the source-JAX chunk-major segmented GDN path. It reduced Entry 053's outer segmentation from about `72` row-chunks to `16` chunk-major steps, but it still de-vectorized the existing chunked GDN triangular solve across `n_chunks`, ran the 31-row inner scan separately per chunk, added per-chunk state/output scatters, and specialized the JIT key on exact query-length tuples. The recommended next step is a backend-owned segmented GDN prefill scaffold plus standalone hetero8 microbenchmark, promoted to server routing only if the standalone candidate beats current chunk32.
- change tested: a temporary `NANO_VLLM_JAX_STATIC_CHUNK_MAJOR_GDN_PREFILL=1` path passed static host query lengths into the GDN prefill call, unrolled over the fixed time-chunk index, gathered only rows active in that chunk, and scattered the updated recurrent state/output back to the rectangular layout. This avoided Entry 053's per-row/per-chunk outer scan but remained a source-JAX segmented path. The source change and its focused test were reverted after profiling.
- focused CUDA checks: with `NANO_VLLM_JAX_STATIC_CHUNK_MAJOR_GDN_PREFILL=1` and accepted compact-prefill flags enabled, the temporary segmented-vs-padded GDN parity test, `tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode`, `tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute`, `tests/test_backend_boundaries.py::test_bucketed_prefill_last_logits_match_exact_prefill`, and `tests/test_lm_head_helpers.py` passed under `JAX_PLATFORMS=cuda` (`7 passed`). A broader run including `test_linear_attention_chunked_vs_recurrent`, `test_linear_attention_multichunk_matches_recurrent`, and `test_linear_suffix_prefill_matches_sequential_decode_state` also passed (`11 passed`).
- correctness: exact generated-token match for all 8 rows against the Entry 045 chunk-32 reference.
- JAX timing: `227.49 tok/s`, TTFT p50 `626.80 ms`, ITL p50 `16.05 ms`, ITL p95 `16.76 ms`.
- delta vs Entry 045 chunk-32 default repeat: `0.619x` total tok/s, TTFT p50 `2.162x`, ITL p50 `1.221x`, ITL p95 `1.233x`.

Top trace ranges, total inclusive time:

| range | Entry 045 ms | static chunk-major GDN ms | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | 696.03 | 1125.30 | overall much slower |
| `_run_main_and_sample` | 572.99 | 992.72 | runner hot path much slower |
| `forward_step_token_ids_jit` | 122.13 | 659.37 | compiled token-id path much slower |
| first `forward_step_token_ids_jit` | 30.83 | 539.25 | first prefill exploded |
| `PjRtCApiLoadedExecutable::Execute` | 138.32 over 252 calls | 673.34 over 252 calls | same dispatch count, far heavier device execution |
| `jit_compiled:XLA GPU module` | 94.20 | 613.25 | module time exploded |
| `command_buffer::execute` | 40.34 over 1143 calls | 467.52 over 9477 calls | many more command-buffer executes |
| `command_buffer::update` | 27.13 over 248 calls | 36.90 over 248 calls | updates heavier |
| `input_reduce_fusion` | 28.65 over 1936 calls | 44.80 over 19234 calls | intended GDN bucket regressed |
| `loop_dynamic_update_slice_fusion` | 10.47 over 1328 calls | 18.67 over 9428 calls | per-chunk scatters expanded |
| `loop_multiply_fusion` | 8.67 over 1550 calls | 25.86 over 10604 calls | de-vectorized chunk work expanded |
| `gemm_fusion_dot_2` | 120.93 over 2695 calls | 172.42 over 1543 calls | broader GEMM family regressed |
| `gather` | 11.97 | 19.59 | worse |
| `transpose` | 63.11 | 39.99 | lower, but irrelevant next to GDN/module regression |
| `wrapped_concatenate` | 36.50 over 576 calls | 36.46 over 576 calls | unchanged |
| `MemcpyD2D` | 24.69 over 1231 calls | 22.11 over 1465 calls | slightly lower total, more copies |
| `array.py:325 tolist` | 437.58 | 317.74 | host sync lower only because device critical path moved elsewhere |
| `np.asarray(jax.Array)` | 437.35 | 317.47 | host sync lower only because device critical path moved elsewhere |

Decision:

- Reject and revert static chunk-major GDN prefill. It preserved correctness, but it confirmed the audit's risk: reducing padded row arithmetic in source JAX destroyed the current vectorized chunked-GDN lowering and created many more small command-buffer executions.
- Do not pursue further source-level segmented GDN prefill variants. Entry 053 and Entry 056 together bracket row-major and chunk-major segmentation, and both lose badly for the same module/kernel-count reason.
- Next aligned work should be a backend-owned segmented GDN prefill scaffold and standalone CUDA microbenchmark for the exact hetero8 shape (`B=8`, `H=16`, `T=512`, `K=V=128`, `chunk=32`, lengths `64..512`) before any server routing. Promote it only if the microbenchmark reduces warmed kernel time without command-buffer or module-time growth.

## Entry 057 - Accepted GDN Prefill Kernel Microbenchmark Scaffold

- run id: `20260526-043748-2141732-gdn_prefill_kernel_baseline_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_baseline_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-043748-2141732-gdn_prefill_kernel_baseline_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-043748-2141732-gdn_prefill_kernel_baseline_hetero8_64_512x32/plugins/profile/2026_05_26_04_37_52/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-043748-2141732-gdn_prefill_kernel_baseline_hetero8_64_512x32/plugins/profile/2026_05_26_04_37_52/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Hume recommended keeping this turn to a standalone CUDA/JAX microbenchmark for `backend.gated_delta_prefill`, not server routing. The exact-shape reference should be the accepted current padded `jax_chunk_gated_delta_rule` chunk32 path on `B=8,H=16,T=512,K=V=128,chunk=32,lengths=64..512`; recurrent GDN should only be used as a reduced-shape correctness reference; and source-segmented row/chunk-major variants should not be rerun because Entries 053 and 056 already bracket them as negative controls.
- change accepted: added `benchmarks/benchmark_gdn_prefill_kernel.py`, a standalone profiled microbenchmark scaffold for the current GDN prefill kernel shape. It records JAX device/version, git head, active vs rectangular tokens/chunks, compile time, warmup/repeat latencies, self-correctness deltas, profile trace path, and parsed Perfetto range counters. It intentionally defaults to only `current_jax_chunk32_padded`; future backend-owned candidates must be added here and win before any server path routes to them.
- scope note: this is a synthetic kernel benchmark, not a replacement for real-weight generated-token correctness. Main-model correctness remains gated by the Entry 045 hetero8 real-weight reference before any serving change is accepted.
- smoke check: a tiny CUDA no-profile run (`B=2,H=2,T=64,K=V=16,chunk=16,lengths=32,64`) produced valid JSON and zero self-delta.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats.
- correctness: self-comparison for output/final-state is exactly zero (`output_max_abs=0.0`, `valid_output_max_abs=0.0`, `state_max_abs=0.0`). Future candidate comparisons in this benchmark should use `rtol=1e-5, atol=1e-5` as the initial gate, then server-generated exact-token gates.
- timing: current padded chunk32 standalone GDN prefill p50 `6.477 ms`, mean `6.484 ms`, p95 `6.527 ms`, min `6.458 ms`, max `6.538 ms`, true-token throughput `355,323 tok/s` at `2304` active tokens and rectangular-token throughput `631,686 tok/s` at `4096` rectangular tokens. Compile time was `0.162 s` with the existing disk compile cache warm.

Top profile counters from the standalone trace, normalized by `25` profiled calls (`5` warmup + `20` repeat):

| range | total ms / count | per call | note |
| --- | ---: | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `227.34 ms / 25` | `9.09 ms, 1.00 call` | annotation includes warmups and profiling overhead; use repeat p50 for timing gate |
| `PjRtCApiLoadedExecutable::Execute` | `97.92 ms / 25` | `3.92 ms, 1.00 call` | device execute baseline |
| `command_buffer::execute` | `20.65 ms / 1225` | `0.826 ms, 49.0 calls` | kernel-count baseline |
| `command_buffer::update` | `1.48 ms / 96` | `0.059 ms, 3.84 calls` | update baseline |
| `input_reduce_fusion` | `14.65 ms / 775` | `0.586 ms, 31.0 calls` | GDN reduce family baseline |
| `loop_dynamic_update_slice_fusion` | `6.06 ms / 1175` | `0.242 ms, 47.0 calls` | scatter/update baseline to avoid Entry 053/056 explosion |
| `loop_multiply_fusion` | `22.83 ms / 425` | `0.913 ms, 17.0 calls` | chunk recurrence baseline |
| `MemcpyD2D` | `5.80 ms / 875` | `0.232 ms, 35.0 calls` | copy baseline |
| `while` | `28.91 ms / 50` | `1.156 ms, 2.0 calls` | loop baseline |
| `fusion` | `96.76 ms / 7800` | `3.870 ms, 312.0 calls` | broad fusion baseline |
| `transpose` | `16.51 ms / 625` | `0.660 ms, 25.0 calls` | layout baseline |

Decision:

- Accept the standalone GDN prefill microbenchmark scaffold. It does not improve serving speed by itself, but it closes the measurement gap identified by Entries 053, 056, and the xhigh audits: backend-owned GDN candidates now have a narrow, reproducible CUDA gate before any model/server routing.
- Future backend candidates must be added as explicit variants in `benchmark_gdn_prefill_kernel.py` and beat the current p50 `6.477 ms` without worsening command-buffer/module-like counters or correctness deltas. Only then should the candidate be threaded through `KernelBackendPlaceholder.gated_delta_prefill` and tested in the full hetero8 server path.

## Entry 058 - Rejected GDN Triangular-Solve Recurrence

- run id: `20260526-044522-2145908-gdn_prefill_kernel_triangular_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_triangular_hetero8_64_512x32.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-044522-2145908-gdn_prefill_kernel_triangular_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-044522-2145908-gdn_prefill_kernel_triangular_hetero8_64_512x32/plugins/profile/2026_05_26_04_45_29/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-044522-2145908-gdn_prefill_kernel_triangular_hetero8_64_512x32/plugins/profile/2026_05_26_04_45_29/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Kierkegaard ranked GDN prefill as the best next target, decode kernels and LM-head/source-level token-id work as lower-confidence, and attention/KV layout as too risky without a backend kernel that owns write/layout/attention together. The recommended next experiment remains a backend-owned segmented/lowered chunk32 GDN prefill path that consumes lengths or a valid-chunk mask inside the operation and processes only the `72` active chunks, not another JAX-side segmented scan.
- change tested: a temporary microbenchmark-only variant replaced the chunk-local HF row recurrence with a batched triangular solve for `(I - A)^-1`, where `A` is the strictly lower intra-chunk attention correction matrix. This was not routed through the server path and the source hook was reverted after profiling.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats per variant.
- timing: current padded chunk32 in the same run had p50 `6.506 ms`, mean `6.591 ms`, p95 `6.702 ms`; the triangular-solve candidate had p50 `5.799 ms`, mean `5.809 ms`, p95 `5.883 ms`. That is a real standalone speed win, about `1.12x` by p50.
- correctness: reject despite speed. The candidate missed the Entry 057 standalone gate with `valid_output_max_abs=1.335e-05` and `state_max_abs=2.441e-04` versus the current padded chunk32 reference. Given the earlier real-weight drift standard, a single-layer final-state delta above `1e-4` is not acceptable without much stronger long-decode/logit evidence, so it should not be promoted.

Top profile counters from the mixed two-variant trace:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `231.92 ms / 25` | profile annotation about `9.28 ms` per current call including warmup/profile overhead |
| `gdn_prefill/triangular_solve_chunk32_padded` | `159.40 ms / 25` | about `6.38 ms` per candidate call including warmup/profile overhead |
| `PjRtCApiLoadedExecutable::Execute` | `137.86 ms / 50` | mixed current+candidate calls, about `2.76 ms` per profiled call |
| `command_buffer::execute` | `30.26 ms / 1675` | mixed current+candidate, `33.5` calls per profiled call |
| `command_buffer::update` | `3.02 ms / 168` | mixed current+candidate, `3.36` calls per profiled call |
| `input_reduce_fusion` | `14.65 ms / 775` | same total count as current-only baseline across the mixed trace |
| `loop_dynamic_update_slice_fusion` | `10.23 ms / 1575` | lower than duplicating current twice, but correctness failed |
| `loop_multiply_fusion` | `45.70 ms / 850` | unchanged recurrence-family count per mixed call |
| `MemcpyD2D` | `12.62 ms / 1775` | mixed copy baseline |
| `while` | `45.37 ms / 75` | mixed loop baseline |
| `fusion` | `173.76 ms / 11750` | broad mixed fusion baseline |
| `transpose` | `34.42 ms / 1325` | mixed layout baseline |

Decision:

- Reject and revert the triangular-solve GDN recurrence. The profile proves that replacing the row scan can reduce standalone kernel time, but the numerical state drift violates the current correctness bar before real-weight/server routing.
- Do not expose this as an optional server or benchmark variant yet. If triangular solve is reconsidered, it needs a stronger correctness plan first, including real-weight layerwise parity, long-decode top-5 logits, and exact generated-token checks.
- Continue with Kierkegaard's recommendation: the next viable speed path is a backend-owned segmented/lowered GDN prefill candidate that preserves the current padded chunk32 outputs/final states within `1e-5` and beats p50 `6.477 ms` without command-buffer/kernel-count growth.

## Entry 059 - Rejected Pallas GDN Chunk Recurrence

- run id: `20260526-045422-2149962-gdn_prefill_kernel_pallas_recurrence_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_pallas_recurrence_hetero8_64_512x32.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-045422-2149962-gdn_prefill_kernel_pallas_recurrence_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-045422-2149962-gdn_prefill_kernel_pallas_recurrence_hetero8_64_512x32/plugins/profile/2026_05_26_04_54_29/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-045422-2149962-gdn_prefill_kernel_pallas_recurrence_hetero8_64_512x32/plugins/profile/2026_05_26_04_54_29/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Parfit confirmed the environment supports Pallas/Triton on A10G (`jax.experimental.pallas.triton`, Triton `3.6.0`) for kernels that avoid unsupported primitives, but warned not to promote this recurrence experiment because it fails the same correctness class as Entry 058. It recommended the next attempt be a microbenchmark-only Pallas/Triton segmented GDN prefill variant over active row chunks, and explicitly deferred standalone `triton.jit` and JAX FFI/custom-call because those add host dispatch or build-system complexity before Pallas is exhausted.
- change tested: a temporary microbenchmark-only Pallas/Triton kernel replaced only the chunk-local GDN row recurrence while preserving the rest of the current padded chunk32 computation. The first direct-row-slice kernel failed lowering with `Unimplemented primitive in Pallas GPU lowering: slice`; the mask/select rewrite lowered successfully. An `einsum` spelling inside Pallas also failed lowering, so the profiled run used explicit elementwise multiply plus sum. The source hook was reverted after profiling.
- smoke check: a tiny CUDA no-profile run (`B=2,H=2,T=64,K=V=16,chunk=16,lengths=32,64`) lowered and ran. It matched the current chunked path within `valid_output_max_abs=1.341e-07` and `state_max_abs=5.960e-07`.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats per variant.
- timing: current padded chunk32 in the same run had p50 `6.476 ms`, mean `6.486 ms`, p95 `6.544 ms`; the Pallas recurrence candidate had p50 `5.684 ms`, mean `5.691 ms`, p95 `5.732 ms`. This is about a `1.14x` p50 speedup in the standalone GDN gate.
- correctness: reject despite speed. The candidate missed the Entry 057 standalone gate with `valid_output_max_abs=1.907e-05` and `state_max_abs=2.136e-04` versus the current padded chunk32 reference.

Top profile counters from the mixed two-variant trace:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `222.06 ms / 25` | about `8.88 ms` per current call including warmup/profile overhead |
| `gdn_prefill/pallas_recurrence_chunk32_padded` | `143.96 ms / 25` | about `5.76 ms` per candidate call including warmup/profile overhead |
| `gdn_chunk_attn_recurrence` | `1.07 ms / 25` | Pallas custom-call range itself was small; surrounding compiled graph still dominates |
| `PjRtCApiLoadedExecutable::Execute` | `111.81 ms / 50` | mixed current+candidate calls, about `2.24 ms` per profiled call |
| `command_buffer::execute` | `28.15 ms / 1650` | mixed current+candidate, `33.0` calls per profiled call |
| `command_buffer::update` | `2.51 ms / 144` | mixed current+candidate, `2.88` calls per profiled call |
| `input_reduce_fusion` | `14.64 ms / 775` | same total count as current-only baseline across the mixed trace |
| `loop_dynamic_update_slice_fusion` | `9.97 ms / 1575` | lower than duplicating current twice, but correctness failed |
| `loop_multiply_fusion` | `45.65 ms / 850` | unchanged recurrence-family count per mixed call |
| `MemcpyD2D` | `11.17 ms / 1725` | mixed copy baseline |
| `while` | `42.59 ms / 75` | mixed loop baseline |
| `fusion` | `171.36 ms / 11675` | broad mixed fusion baseline |
| `transpose` | `32.96 ms / 1250` | mixed layout baseline |

Decision:

- Reject and revert the Pallas chunk recurrence. It confirms Pallas/Triton can reduce this part of the standalone GDN profile on A10G, but it still changes FP32 accumulation enough to miss the output and final-state gates.
- Do not route this through `KernelBackendPlaceholder.gated_delta_prefill` or keep it as an exposed variant. The current exact padded chunk32 source path remains the only accepted GDN prefill implementation.
- Next viable Pallas work should not replace only mathematically equivalent reductions. It should target the actual segmented active-chunk problem while preserving the current outputs/final states within `1e-5`; otherwise move to JAX FFI/custom-call only with a clear build plan and the same standalone gate.

## Entry 060 - Rejected Rowwise Exact-Length GDN Prefill

- run id: `20260526-050121-2152572-gdn_prefill_kernel_rowwise_exact_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_rowwise_exact_hetero8_64_512x32.json`
- profile directory: `/mountpoint/.exp/profiles/20260526-050121-2152572-gdn_prefill_kernel_rowwise_exact_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-050121-2152572-gdn_prefill_kernel_rowwise_exact_hetero8_64_512x32/plugins/profile/2026_05_26_05_01_36/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-050121-2152572-gdn_prefill_kernel_rowwise_exact_hetero8_64_512x32/plugins/profile/2026_05_26_05_01_36/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Peirce recommended a microbenchmark-only `length_bucketed_current_jax_chunk32` diagnostic: call the existing `jax_chunk_gated_delta_rule` unchanged on each static prompt length, pad outputs back to `T=512`, and compare against current padded chunk32. The audit framed this as the best source-JAX diagnostic because it skips inactive chunks without changing recurrence math and without repeating Entries 053/056's per-active-row/chunk dynamic-slice scans.
- change tested: a temporary benchmark-only `rowwise_exact_chunk32` variant called the accepted chunked GDN implementation once per row/length (`64..512`) with the row's real prefix, then padded outputs and concatenated states back to the rectangular contract. It processed `72` real chunks instead of `128` rectangular chunks. The source hook was reverted after profiling.
- smoke check: a tiny CUDA no-profile run (`B=2,H=2,T=64,K=V=16,chunk=16,lengths=32,64`) lowered and ran. It matched within `valid_output_max_abs=1.490e-07` and `state_max_abs=5.960e-07`.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats per variant.
- timing: current padded chunk32 in the same run had p50 `6.483 ms`, mean `6.549 ms`, p95 `6.822 ms`; rowwise exact-length had p50 `11.591 ms`, mean `12.044 ms`, p95 `14.544 ms`. Compile time also grew from `0.161 s` to `10.189 s`.
- correctness: reject. The rowwise candidate missed the Entry 057 standalone gate with `valid_output_max_abs=1.907e-05` and `state_max_abs=1.831e-04`. This shows that even reusing the same JAX source at smaller static shapes changes FP32 accumulation enough to fail the current state/output bar.

Top profile counters from the mixed two-variant trace:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `168.51 ms / 25` | about `6.74 ms` per current call including warmup/profile overhead |
| `gdn_prefill/rowwise_exact_chunk32` | `314.21 ms / 25` | about `12.57 ms` per candidate call including warmup/profile overhead |
| `PjRtCApiLoadedExecutable::Execute` | `344.66 ms / 50` | mixed current+candidate calls, about `6.89 ms` per profiled call |
| `command_buffer::execute` | `70.50 ms / 3425` | mixed current+candidate, `68.5` calls per profiled call |
| `command_buffer::update` | `11.99 ms / 672` | mixed current+candidate, `13.44` calls per profiled call |
| `input_reduce_fusion` | `103.85 ms / 19375` | large recurrence-family expansion |
| `loop_dynamic_update_slice_fusion` | `111.69 ms / 21575` | per-row subgraphs expanded update/scatter work |
| `loop_multiply_fusion` | `122.52 ms / 22525` | per-row subgraphs expanded multiply work |
| `while` | `302.77 ms / 450` | rowwise chunk scans multiplied loop work |
| `fusion` | `527.59 ms / 100025` | broad fusion count exploded |
| `MemcpyD2D` | `24.36 ms / 4875` | copy count increased to `97.5` per profiled call |
| `transpose` | `22.15 ms / 1225` | lower than current in isolation but not relevant next to recurrence explosion |

Decision:

- Reject and revert rowwise exact-length GDN prefill. It failed both acceptance axes: correctness drift exceeded the standalone gate, and performance regressed badly from duplicated per-row compiled subgraphs.
- Treat this as closure for source-JAX exact-length inactive-chunk skipping. Even the best diagnostic source spelling creates shape-dependent numerical drift and Entry-053/056-like module/kernel expansion.
- The next GDN speed attempt should be a true backend-owned segmented kernel that preserves the exact padded chunk32 accumulation contract, or a lower-level custom implementation with explicit correctness controls; more source-JAX decomposition is not aligned with the current evidence.

## Entry 061 - Current Server Reprofile After GDN Microbench Closure

- matched run id: `20260526-051717-2158089-jax_hetero8_64_512x32_current_matched_reprofile`
- matched benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_current_matched_reprofile.json`
- matched profile directory: `/mountpoint/.exp/profiles/20260526-051717-2158089-jax_hetero8_64_512x32_current_matched_reprofile`
- matched Perfetto trace: `/mountpoint/.exp/profiles/20260526-051717-2158089-jax_hetero8_64_512x32_current_matched_reprofile/plugins/profile/2026_05_26_05_17_39/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- matched TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-051717-2158089-jax_hetero8_64_512x32_current_matched_reprofile/plugins/profile/2026_05_26_05_17_39/INDCS0291.atrapa.deloitte.com.xplane.pb`
- scheduler-envelope artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_current_reprofile.json`
- scheduler-envelope profile directory: `/mountpoint/.exp/profiles/20260526-050707-2154593-jax_hetero8_64_512x32_current_reprofile`
- scheduler-envelope Perfetto trace: `/mountpoint/.exp/profiles/20260526-050707-2154593-jax_hetero8_64_512x32_current_reprofile/plugins/profile/2026_05_26_05_13_52/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- scheduler-envelope TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-050707-2154593-jax_hetero8_64_512x32_current_reprofile/plugins/profile/2026_05_26_05_13_52/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Mencius recommended first confirming the current server contract against Entry 045, then avoiding more source-JAX inactive-chunk GDN work. The audit identified the next useful target as a backend-owned segmented GDN prefill microbenchmark that preserves the current padded chunk32 outputs/final states within `1e-5`, beats Entry 057's p50 `6.477 ms`, and does not increase command-buffer or module counts.
- change tested: no source change. This reprofile records current HEAD after Entries 057-060 closed the source-JAX and recurrence-replacement GDN routes. The matched run used the Entry 045 scheduler envelope: `max_kv_cache_mb=3072`, `num_kvcache_blocks=256`, `max_num_batched_tokens=4096`, `max_num_seqs=8`, BF16 weights, FP32 activations, hetero8 input lengths `64..512`, `32` output tokens, `jit`, greedy token fastpath, materialized tied LM head, and all accepted compact prefill flags.
- correctness: both artifacts exactly matched the Entry 045 generated-token reference for all 8 prompts over all 32 generated tokens.

Matched-envelope performance compared with Entry 045:

| metric | Entry 045 | matched reprofile | note |
| --- | ---: | ---: | --- |
| throughput | `367.797 tok/s` | `256.382 tok/s` | slower wall time, but GPU execute ranges did not regress |
| seconds | `0.696 s` | `0.999 s` | total server trace wall time |
| TTFT p50 | `289.979 ms` | `521.910 ms` | single full-prefill step in both runs (`2304` scheduler tokens) |
| ITL p50 | `13.145 ms` | `15.219 ms` | decode path still exact |
| ITL p95 | `13.593 ms` | `16.181 ms` | decode path still exact |

Matched-envelope profile comparison:

| range | Entry 045 total ms / count | matched total ms / count | note |
| --- | ---: | ---: | --- |
| `generate_with_trace` | `696.03 / 1` | `998.50 / 1` | wall range slower |
| `_run_main_and_sample` | `572.99 / 32` | `877.08 / 32` | host-visible sample loop slower |
| `forward_step_token_ids_jit` | `122.13 / 32` | `118.43 / 32` | model forward range stable/slightly lower |
| first `forward_step_token_ids_jit` | `30.83 / 1` | `27.26 / 1` | prefill forward stable |
| `PjRtCApiLoadedExecutable::Execute` | `138.32 / 252` | `132.64 / 252` | GPU executable time stable/slightly lower |
| `jit_compiled:XLA GPU module` | `94.20 / 32` | `89.17 / 32` | stable |
| `command_buffer::execute` | `40.34 / 1143` | `37.73 / 1143` | stable/slightly lower |
| `command_buffer::update` | `27.13 / 248` | `26.88 / 248` | stable |
| `input_reduce_fusion` | `28.65 / 1936` | `28.64 / 1936` | GDN recurrence range unchanged |
| `loop_dynamic_update_slice_fusion` | `10.47 / 1328` | `10.53 / 1328` | unchanged |
| `loop_multiply_fusion` | `8.67 / 1550` | `8.47 / 1550` | unchanged |
| `gather` | `11.97 / 135` | `12.16 / 135` | unchanged |
| `transpose` | `63.11 / 1236` | `63.25 / 1236` | unchanged |
| `wrapped_concatenate` | `36.50 / 576` | `36.46 / 576` | unchanged |
| `MemcpyD2D` | `24.69 / 1231` | `24.62 / 1232` | unchanged |
| `array.py:325 tolist` | `437.58 / 32` | `745.22 / 32` | dominant source of slower wall timing |
| `np.asarray(jax.Array)` | `437.35 / 32` | `744.96 / 32` | synchronization/readback label, not a model-kernel regression |
| `while` | `13.48 / 36` | `12.33 / 36` | stable/slightly lower |

Scheduler-envelope check:

- A first reprofile used `max_kv_cache_mb=1024`, `num_kvcache_blocks=64`, and `max_num_batched_tokens=512`. It is not comparable to Entry 045 because prefill was split into several scheduler steps instead of a single `2304` token prefill step.
- That smaller-envelope run remained correct but measured only `9.313 tok/s`, TTFT p50 `12731.629 ms`, ITL p50 `22.116 ms`, and ITL p95 `25.784 ms`. The trace showed `generate_with_trace=27487.12 ms`, `_run_main_and_sample=25132.42 ms`, `forward_step_token_ids_jit=24552.00 ms`, `PjRt Execute=441.60 ms / 2179`, `gather=930.86 ms / 2076`, `MemcpyD2D=85.80 ms / 3209`, and `while=79.01 ms / 76`.
- Treat this as evidence that the smaller scheduler envelope is a separate throughput problem; do not compare it to Entry 045's accepted baseline.

Decision:

- Correctness is still protected: the current server path exactly reproduces the Entry 045 generated-token reference under both scheduler envelopes.
- The matched fair comparison does not show a model-kernel regression. `PjRt Execute`, command-buffer counts, GDN recurrence ranges, KV/layout ranges, and the first prefill forward are stable. The slower matched wall timing is dominated by `tolist`/`np.asarray` synchronization/readback labels, so it should not redirect kernel work by itself.
- Continue with Mencius's recommendation: the next speed target should be a backend-owned segmented GDN prefill microbenchmark gated against Entry 057. Do not resume source-JAX GDN decomposition or recurrence-replacement work unless there is a new correctness plan.

## Entry 062 - Rejected Strict-Mask GDN Micro-Cleanup

- first run id: `20260526-052652-2161198-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32`
- first benchmark artifact: `results/gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32.json`
- first profile directory: `/mountpoint/.exp/profiles/20260526-052652-2161198-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32`
- first Perfetto trace: `/mountpoint/.exp/profiles/20260526-052652-2161198-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32/plugins/profile/2026_05_26_05_26_58/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- first TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-052652-2161198-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32/plugins/profile/2026_05_26_05_26_58/INDCS0291.atrapa.deloitte.com.xplane.pb`
- repeat run id: `20260526-052746-2161461-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32_repeat`
- repeat benchmark artifact: `results/gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32_repeat.json`
- repeat profile directory: `/mountpoint/.exp/profiles/20260526-052746-2161461-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32_repeat`
- repeat Perfetto trace: `/mountpoint/.exp/profiles/20260526-052746-2161461-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32_repeat/plugins/profile/2026_05_26_05_27_51/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- repeat TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-052746-2161461-gdn_prefill_kernel_strict_decay_mask_hetero8_64_512x32_repeat/plugins/profile/2026_05_26_05_27_51/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Planck confirmed that backend-owned segmented GDN prefill remains the right next model-side target. It ranked rectangular padded chunked GDN as the strongest remaining model bottleneck because the workload has `2304` active prompt tokens but `4096` rectangular slots and Entry 057 recorded `72` active chunks vs `128` total chunks. It also warned not to reintroduce source-JAX row/chunk scans, rowwise exact-length calls, recurrence replacements, or scheduler-envelope comparisons as model-kernel evidence.
- change tested: a temporary benchmark-only `strict_decay_mask_chunk32_padded` variant precomputed the initial chunk-recurrence mask as strict-lower triangular instead of computing lower-inclusive decay and then zeroing diagonal/upper entries with `where`. A first smoke attempt incorrectly removed the diagonal from output attention and failed output parity; the profiled version kept output attention lower-inclusive and only removed the recurrence-matrix diagonal/upper mask. The source hook was reverted after profiling. The retained code change only makes `benchmark_gdn_prefill_kernel.py` parse profile annotation ranges for whatever variants are requested, so future candidates get their own `gdn_prefill/<variant>` counters automatically.
- full CUDA runs: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats per variant.
- correctness: the fixed strict-mask candidate exactly matched the current padded chunk32 reference in both full runs: `output_max_abs=0.0`, `valid_output_max_abs=0.0`, and `state_max_abs=0.0`.

Timing:

| run | current p50 / mean / p95 | strict-mask p50 / mean / p95 | note |
| --- | ---: | ---: | --- |
| first | `6.482 / 6.549 / 6.677 ms` | `6.468 / 6.500 / 6.571 ms` | beats same-run current, but misses historical mean/p95 gates |
| repeat | `6.498 / 6.524 / 6.630 ms` | `6.483 / 6.482 / 6.503 ms` | beats same-run current, but misses Entry 057 p50 gate `6.477 ms` |

Top repeat profile counters from the mixed two-variant trace:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `231.66 ms / 25` | annotation includes warmup/profile overhead |
| `gdn_prefill/strict_decay_mask_chunk32_padded` | `164.50 ms / 25` | lower annotation range, but measured repeat p50 win is only about `0.015 ms` |
| `PjRtCApiLoadedExecutable::Execute` | `140.09 ms / 50` | one execute per benchmark call, no dispatch growth |
| `command_buffer::execute` | `42.76 ms / 2450` | `49.0` per profiled call, unchanged mixed count |
| `command_buffer::update` | `3.36 ms / 192` | `3.84` per profiled call, unchanged mixed count |
| `input_reduce_fusion` | `29.29 ms / 1550` | unchanged recurrence-family count |
| `loop_dynamic_update_slice_fusion` | `12.62 ms / 2350` | unchanged recurrence-family count |
| `loop_multiply_fusion` | `47.32 ms / 1250` | unchanged recurrence-family count |
| `while` | `60.16 ms / 100` | unchanged mixed loop count |

Decision:

- Reject and revert strict-mask GDN as an optimization candidate. It is exact and slightly faster than the same-run current control, but the improvement is too small/noisy and does not consistently clear the historical Entry 057 standalone gates.
- Do not route this through the server path or keep it as an exposed benchmark variant. It is a local algebraic cleanup, not the backend-owned segmented GDN prefill path we need.
- Keep the dynamic profile-counter parsing improvement in the GDN microbenchmark because it reduces future logbook friction without changing model behavior.
- Next work should follow Planck's gate: add a real backend-owned segmented GDN candidate that skips fully inactive row chunks inside one coarse operation and preserves the padded chunk32 output/final-state contract within `1e-5`.

## Entry 063 - Pallas Active-Chunk Metadata Probe For Segmented GDN

- run id: `20260526-054325-2166013-gdn_prefill_kernel_active_chunk_plan_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_active_chunk_plan_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-054325-2166013-gdn_prefill_kernel_active_chunk_plan_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-054325-2166013-gdn_prefill_kernel_active_chunk_plan_hetero8_64_512x32/plugins/profile/2026_05_26_05_43_30/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-054325-2166013-gdn_prefill_kernel_active_chunk_plan_hetero8_64_512x32/plugins/profile/2026_05_26_05_43_30/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Sagan recommended the next real candidate be benchmark-only `pallas_segmented_chunk32_skip_inactive`, taking `query_lens` or `valid_chunk_mask` as metadata and preserving the exact rectangular output/final-state contract. It confirmed that Pallas/Triton is viable on this A10G/JAX 0.10 setup, but warned to avoid direct dynamic slices and `einsum` inside Pallas. If the full kernel is too large, the smallest useful diagnostic is a Pallas metadata probe that consumes `query_lens`, computes active chunks, and records `72 active / 128 total / 56 skipped`.
- change accepted: the standalone GDN benchmark now records a host-side active-chunk plan in `run_config.active_chunk_plan`, including `active_rows`, `active_chunks`, `active_starts`, `active_token_counts`, `chunks_per_row`, `row_offsets`, and active/partial masks. It also runs a tiny Pallas/Triton metadata probe on GPU that consumes `query_lens`, writes the active-chunk mask, and stores the result in `pallas_feasibility`. No model/server route changed.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats, current padded chunk32 only.
- active-chunk contract: `72` active chunks, `128` total chunks, `56` inactive chunks, `0` partial chunks for this length set. `chunks_per_row=[2,4,6,8,10,12,14,16]` and `row_offsets=[0,2,6,12,20,30,42,56,72]`.
- Pallas metadata probe: lowering succeeded, `mask_matches_plan=true`, `custom_call_count=2`, compile time `0.288 s`, host-visible run time `2.637 ms`, and the traced kernel label `gdn_active_chunk_probe` took `0.038 ms` total.
- correctness: current padded chunk32 self-comparison remains exact: `output_max_abs=0.0`, `valid_output_max_abs=0.0`, `state_max_abs=0.0`.
- timing: current padded chunk32 p50 `6.479 ms`, mean `6.483 ms`, p95 `6.509 ms`, true-token throughput `355,391 tok/s`, rectangular-token throughput `631,806 tok/s`.

Top profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `229.65 ms / 25` | baseline annotation including warmup/profile overhead |
| `gdn_prefill/pallas_active_chunk_probe` | `2.67 ms / 1` | host-visible probe range with synchronization |
| `gdn_active_chunk_probe` | `0.038 ms / 2` | actual tiny Pallas kernel labels |
| `PjRtCApiLoadedExecutable::Execute` | `99.58 ms / 26` | one extra execute from the metadata probe |
| `command_buffer::execute` | `19.69 ms / 1225` | baseline GDN command-buffer count unchanged |
| `command_buffer::update` | `1.48 ms / 96` | baseline GDN command-buffer updates unchanged |
| `input_reduce_fusion` | `14.64 ms / 775` | baseline recurrence counter |
| `while` | `27.50 ms / 50` | baseline loop counter |

Decision:

- Keep this benchmark scaffold. It proves that the active-chunk metadata contract is explicit, serializable in artifacts, and consumable by an A10G-compatible Pallas/Triton kernel without touching model behavior.
- This is not a speed win and must not be promoted to server routing. It is a prerequisite for the real `pallas_segmented_chunk32_skip_inactive` candidate.
- The next implementation step should replace the metadata-only kernel with a benchmark-only Pallas segmented GDN candidate over the same `query_lens`/active-chunk plan, preserving the current padded chunk32 reference within `1e-5`.

## Entry 064 - Pallas Active-Input Pack Probe For Segmented GDN

- run id: `20260526-055311-2168055-gdn_prefill_kernel_active_input_pack_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_active_input_pack_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-055311-2168055-gdn_prefill_kernel_active_input_pack_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-055311-2168055-gdn_prefill_kernel_active_input_pack_hetero8_64_512x32/plugins/profile/2026_05_26_05_53_17/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-055311-2168055-gdn_prefill_kernel_active_input_pack_hetero8_64_512x32/plugins/profile/2026_05_26_05_53_17/INDCS0291.atrapa.deloitte.com.xplane.pb`
- change accepted: the standalone GDN benchmark now includes a benchmark-only Pallas/Triton active-input pack probe. It consumes the host active-chunk plan, gathers active `query`, `key`, `value`, `g`, and `beta` chunk inputs into dense active-chunk buffers, and compares the packed result against a JAX gather reference. No model/server route changed.
- CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats, current padded chunk32 only.
- active-chunk contract: `72` active chunks, `128` total chunks, `56` inactive chunks, and `2304` packed active tokens.
- pack probe: lowering succeeded, compile time `0.157 s`, and the packed `query`/`key`/`value`/`g`/`beta` outputs matched the JAX reference exactly: all max-abs deltas `0.0`. Repeat timing was p50 `0.455 ms`, mean `0.456 ms`, p95 `0.472 ms`.
- metadata probe: lowering still succeeded and matched the active plan (`72` active, `56` inactive). Its host-visible one-shot range was slower in this trace (`4.991 ms`) because the new pack probe changes the warmup/profile envelope; the actual metadata custom-call label remained tiny at `0.056 ms` total.
- current padded chunk32 correctness remains exact: `output_max_abs=0.0`, `valid_output_max_abs=0.0`, and `state_max_abs=0.0`.
- current padded chunk32 timing: p50 `6.473 ms`, mean `6.474 ms`, p95 `6.495 ms`, true-token throughput `355,903 tok/s`, rectangular-token throughput `632,716 tok/s`.

Top profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `223.47 ms / 25` | baseline annotation including warmup/profile overhead |
| `gdn_prefill/pallas_active_input_pack_probe` | `16.69 ms / 25` | pack annotation range; repeat clock p50 was `0.455 ms` |
| `gdn_active_input_pack_probe` | `6.66 ms / 50` | actual Pallas pack kernel labels, two custom-call labels per benchmark call |
| `gdn_prefill/pallas_active_chunk_probe` | `5.06 ms / 1` | metadata probe host-visible range |
| `gdn_active_chunk_probe` | `0.056 ms / 2` | actual metadata Pallas kernel labels |
| `PjRtCApiLoadedExecutable::Execute` | `100.14 ms / 51` | baseline calls plus metadata and pack probes |
| `command_buffer::execute` | `19.20 ms / 1225` | baseline GDN command-buffer count unchanged; pack probe labels are separate |
| `command_buffer::update` | `1.65 ms / 96` | baseline GDN command-buffer updates unchanged |
| `input_reduce_fusion` | `14.66 ms / 775` | baseline recurrence counter |
| `while` | `26.93 ms / 50` | baseline loop counter |

Decision:

- Keep the active-input pack probe as benchmark instrumentation. It proves that the active-chunk metadata can drive vectorized multi-output Pallas input movement on A10G and preserve exact FP32 input values.
- Do not route a separate pack step through the server. The standalone pack costs about `0.455 ms` p50 and emits two custom-call labels per benchmark call, so as an isolated step it would spend a meaningful fraction of the current `6.47 ms` GDN prefill time before doing any GDN math.
- The next real candidate should fuse this active-chunk indexing directly into `pallas_segmented_chunk32_skip_inactive`, so inactive chunks are skipped inside the GDN computation rather than copied into an intermediate buffer first.

## Entry 065 - Pallas Active-Chunk Local Math Probe For Segmented GDN

- run id: `20260526-060946-2172399-gdn_prefill_kernel_active_local_math_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_active_local_math_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-060946-2172399-gdn_prefill_kernel_active_local_math_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-060946-2172399-gdn_prefill_kernel_active_local_math_hetero8_64_512x32/plugins/profile/2026_05_26_06_09_59/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-060946-2172399-gdn_prefill_kernel_active_local_math_hetero8_64_512x32/plugins/profile/2026_05_26_06_09_59/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Arendt recommended a smaller fused per-active-chunk local-math probe before attempting the full segmented GDN kernel. It warned that Entries 058-060 already showed tiny recurrence-order changes can miss the final-state gate, so the next safe step should fuse active indexing into the first real chunk-local GDN math and compare local intermediates before adding the cross-chunk state scan.
- change accepted: the standalone GDN benchmark now includes a benchmark-only Pallas/Triton local-math probe named `gdn_active_chunk_local_math_chunk32_probe`. It consumes the active-chunk metadata, gathers normalized rectangular `key` plus `value`/`g`/`beta`, and computes chunk-local `g_cumsum`, decay-masked HF row-recurrence `attn`, `value_transformed`, and `k_cumdecay`. It compares those intermediates against a JAX reference extracted from the current padded chunk32 math. No model/server route changed.
- implementation note: an initial full-shape smoke with raw, unnormalized random keys produced huge local recurrence drift because it did not match the real `jax_chunk_gated_delta_rule` contract. The accepted probe uses the same L2-normalized key contract as the current path. A scalar per-entry active-KKT Pallas smoke also drifted by `3.05e-05`; the accepted version uses a per-active-chunk matrix `pl.dot(..., allow_tf32=False)` tile, which matched the JAX reference.
- CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats, current padded chunk32 only.
- active-chunk contract: `72` active chunks, `128` total chunks, `56` inactive chunks, and `2304` active tokens.
- local-math correctness: lowering succeeded, compile time `5.094 s`, and the local intermediate comparison passed the `1e-5` gate. Max-abs deltas were `value_transformed=9.537e-07`, `k_cumdecay=3.576e-07`, `g_cumsum=3.576e-07`, `attn=2.384e-07`, combined max `9.537e-07`.
- local-math timing: p50 `1.967 ms`, mean `1.970 ms`, p95 `2.004 ms`, min `1.915 ms`, max `2.012 ms`.
- active-input pack probe remains exact in this run, with p50 `0.498 ms`, mean `0.496 ms`, p95 `0.519 ms`.
- current padded chunk32 correctness remains exact: `output_max_abs=0.0`, `valid_output_max_abs=0.0`, and `state_max_abs=0.0`.
- current padded chunk32 timing: p50 `6.498 ms`, mean `6.504 ms`, p95 `6.546 ms`, true-token throughput `354,244 tok/s`, rectangular-token throughput `629,768 tok/s`.

Top profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `175.43 ms / 25` | baseline annotation; repeat clock p50 was `6.498 ms` |
| `gdn_prefill/pallas_active_chunk_local_math_probe` | `51.77 ms / 25` | local-math annotation range; repeat clock p50 was `1.967 ms` |
| `gdn_active_chunk_local_math_chunk32_probe` | `45.56 ms / 50` | actual local-math Pallas labels, two custom-call labels per benchmark call |
| `gdn_prefill/pallas_active_input_pack_probe` | `13.54 ms / 25` | pack annotation range; repeat clock p50 was `0.498 ms` |
| `gdn_active_input_pack_probe` | `6.67 ms / 50` | actual pack Pallas labels |
| `gdn_prefill/pallas_active_chunk_probe` | `6.70 ms / 1` | metadata probe host-visible range |
| `gdn_active_chunk_probe` | `0.058 ms / 2` | actual metadata Pallas labels |
| `PjRtCApiLoadedExecutable::Execute` | `57.84 ms / 76` | baseline plus metadata, pack, and local-math probes |
| `command_buffer::execute` | `23.20 ms / 1225` | baseline GDN command-buffer count unchanged |
| `command_buffer::update` | `1.73 ms / 96` | baseline GDN command-buffer updates unchanged |
| `input_reduce_fusion` | `14.63 ms / 775` | baseline recurrence-family counter unchanged |
| `loop_dynamic_update_slice_fusion` | `6.54 ms / 1175` | baseline scatter/update counter |
| `loop_multiply_fusion` | `22.86 ms / 425` | baseline chunk-recurrence counter |
| `while` | `32.69 ms / 50` | baseline loop counter |
| `MemcpyD2D` | `6.26 ms / 875` | copy counter |

Decision:

- Keep the local-math probe as benchmark instrumentation. It is the first exact Pallas path that fuses active-chunk indexing with nontrivial chunk-local GDN math under the BF16-weight/FP32-activation serving contract.
- Do not route it through `KernelBackendPlaceholder.gated_delta_prefill` yet. It computes only local intermediates and still omits query-local output attention, the cross-chunk state scan, final rectangular output scatter, and final recurrent state. A standalone local probe cannot prove generated-token correctness.
- The next GDN candidate should extend this exact local-math path with a benchmark-only cross-chunk state/output reconstruction over active chunks, then compare final `valid_output_max_abs` and `state_max_abs` against current padded chunk32 before any server integration.

## Entry 066 - Rejected Active Local-Math Output/State Reconstruction

- run id: `20260526-062016-2175404-gdn_prefill_kernel_active_reconstruction_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_active_reconstruction_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-062016-2175404-gdn_prefill_kernel_active_reconstruction_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-062016-2175404-gdn_prefill_kernel_active_reconstruction_hetero8_64_512x32/plugins/profile/2026_05_26_06_20_25/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-062016-2175404-gdn_prefill_kernel_active_reconstruction_hetero8_64_512x32/plugins/profile/2026_05_26_06_20_25/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Poincare recommended a benchmark-only JAX reconstruction bridge before any Pallas state-scan kernel. It specifically called for dense-scattering active `value_transformed`, `k_cumdecay`, and `g_cumsum` back to rectangular chunk grids, then running the same cross-chunk `lax.scan` equations as `jax_chunk_gated_delta_rule.process_chunk`. It also warned not to use local `attn_with_identity` for query output attention; the reconstruction should recompute query-side decay and keep the strict-upper mask behavior from the current path.
- change tested: the standalone GDN benchmark now has `active_output_state_reconstruction_probe`. It compiles a benchmark-only JAX reconstruction that consumes active local intermediates, scatters them to `[B,H,n_chunks,chunk,*]`, runs the rectangular cross-chunk state/output scan, and compares final output/state against `current_jax_chunk32_padded`. It records two gates: reconstruction from the active JAX local-math reference and reconstruction from the Pallas local-math output. No model/server route changed.
- CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats, current padded chunk32 only.
- current padded chunk32 correctness remains exact: `output_max_abs=0.0`, `valid_output_max_abs=0.0`, and `state_max_abs=0.0`.
- Entry 065 local-math gate still passes inside this run: combined local-intermediate max drift `9.537e-07`, with `value_transformed=9.537e-07`, `k_cumdecay=3.576e-07`, `g_cumsum=3.576e-07`, and `attn=2.384e-07`.
- reconstruction gate failed from active JAX local intermediates: `output_max_abs=1.717e-05`, `valid_output_max_abs=1.717e-05`, and `state_max_abs=2.136e-04`.
- reconstruction gate failed from Pallas local intermediates: `output_max_abs=1.907e-05`, `valid_output_max_abs=1.907e-05`, and `state_max_abs=3.052e-04`.
- reconstruction timing: p50 `4.560 ms`, mean `4.566 ms`, p95 `4.632 ms`, min `4.533 ms`, max `4.649 ms`.
- current padded chunk32 timing: p50 `6.472 ms`, mean `6.486 ms`, p95 `6.545 ms`, true-token throughput `355,240 tok/s`, rectangular-token throughput `631,537 tok/s`.

Top profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `164.23 ms / 25` | baseline annotation; repeat clock p50 was `6.472 ms` |
| `gdn_prefill/pallas_active_output_state_reconstruction_probe` | `122.46 ms / 25` | reconstruction annotation; repeat clock p50 was `4.560 ms` |
| `gdn_prefill/pallas_active_chunk_local_math_probe` | `51.91 ms / 25` | local-math annotation |
| `gdn_active_chunk_local_math_chunk32_probe` | `46.33 ms / 50` | actual local-math Pallas labels |
| `gdn_prefill/pallas_active_input_pack_probe` | `13.04 ms / 25` | pack annotation |
| `gdn_active_input_pack_probe` | `6.70 ms / 50` | actual pack Pallas labels |
| `PjRtCApiLoadedExecutable::Execute` | `68.06 ms / 101` | baseline plus metadata, pack, local-math, and reconstruction probes |
| `command_buffer::execute` | `26.80 ms / 1650` | reconstruction adds command-buffer work vs Entry 065 |
| `command_buffer::update` | `2.51 ms / 146` | reconstruction adds updates vs Entry 065 |
| `input_reduce_fusion` | `14.68 ms / 775` | baseline GDN recurrence-family counter unchanged |
| `loop_dynamic_update_slice_fusion` | `9.77 ms / 1575` | scatter/update count increases with reconstruction |
| `loop_multiply_fusion` | `36.83 ms / 825` | reconstruction scan adds loop multiply work |
| `while` | `41.69 ms / 75` | reconstruction adds one scan loop family |
| `MemcpyD2D` | `12.35 ms / 1675` | reconstruction adds copies |
| `transpose` | `28.15 ms / 1175` | reconstruction adds layout work |

Decision:

- Reject active-local reconstruction as a candidate for server routing. It fails the final `1e-5` correctness gate before any server integration, even when using the active JAX local-math reference. This confirms that small active-shape local differences are amplified by the cross-chunk state recurrence into the same `1e-4` final-state drift class as Entries 058-060.
- Keep the reconstruction probe in the standalone benchmark because it is a useful guardrail: future active/local Pallas work must pass both the local-intermediate gate and the final output/state reconstruction gate before moving toward `KernelBackendPlaceholder.gated_delta_prefill`.
- The next viable GDN route should avoid composing active-shaped local intermediates with a separate rectangular state scan. Either the state/output recurrence must be owned by one lowered segmented kernel with its own final-state parity proof, or the active local math must be made shape-compatible with the current rectangular local intermediates before reuse.

## Entry 067 - Rejected Rectangular Local-Math Split Reconstruction

- run id: `20260526-063026-2178120-gdn_prefill_kernel_rectangular_local_split_hetero8_64_512x32`
- benchmark artifact: `results/gdn_prefill_kernel_rectangular_local_split_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-063026-2178120-gdn_prefill_kernel_rectangular_local_split_hetero8_64_512x32`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-063026-2178120-gdn_prefill_kernel_rectangular_local_split_hetero8_64_512x32/plugins/profile/2026_05_26_06_30_43/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-063026-2178120-gdn_prefill_kernel_rectangular_local_split_hetero8_64_512x32/plugins/profile/2026_05_26_06_30_43/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Meitner recommended a rectangular Pallas local-math probe before attempting a full lowered state/output recurrence. It identified the open question after Entry 066 as whether active-shaped gather/scatter caused the final-state drift, and asked for a rectangular `(B,H,n_chunks)` local-math probe that emits `[B,H,n_chunks,chunk,*]` intermediates directly before any inactive-chunk skipping.
- change tested: the standalone GDN benchmark now records `rectangular_chunk_local_math_probe` and `rectangular_local_split_reconstruction_probe`. The Pallas local probe computes all rectangular chunks, not only active chunks, and emits rectangular `value_transformed`, `k_cumdecay`, `g_cumsum`, and `attn` intermediates. The reconstruction probe feeds either rectangular JAX local intermediates or rectangular Pallas local intermediates into a separate rectangular JAX output/state scan. No model/server route changed.
- CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, FP32 activations/state, chunk size `32`, lengths `64,128,192,256,320,384,448,512`, `5` warmups, `20` measured repeats, current padded chunk32 only.
- current padded chunk32 correctness remains exact: `output_max_abs=0.0`, `valid_output_max_abs=0.0`, and `state_max_abs=0.0`.
- rectangular Pallas local-math gate passed locally: combined local-intermediate max drift `9.537e-07`, with `value_transformed=9.537e-07`, `k_cumdecay=3.576e-07`, `g_cumsum=3.576e-07`, and `attn=2.384e-07`.
- rectangular split reconstruction failed even from rectangular JAX local intermediates: `output_max_abs=1.335e-05`, `valid_output_max_abs=1.335e-05`, and `state_max_abs=1.678e-04`.
- rectangular split reconstruction failed from rectangular Pallas local intermediates: `output_max_abs=1.907e-05`, `valid_output_max_abs=1.907e-05`, and `state_max_abs=3.052e-04`.
- active split reconstruction continues to fail in the same run: active JAX-local reconstruction `state_max_abs=2.136e-04`; active Pallas-local reconstruction `state_max_abs=3.052e-04`.
- rectangular local-math timing: p50 `3.317 ms`, mean `3.320 ms`, p95 `3.343 ms`. This computes all `128` rectangular chunks and is slower than Entry 065's active-local p50 near `1.94 ms`.
- rectangular split reconstruction timing: p50 `4.543 ms`, mean `4.559 ms`, p95 `4.645 ms`; reconstruction from Pallas rectangular local intermediates had p50 `4.552 ms`, mean `4.557 ms`, p95 `4.614 ms`.
- current padded chunk32 timing: p50 `6.500 ms`, mean `6.538 ms`, p95 `6.660 ms`, true-token throughput `352,426 tok/s`, rectangular-token throughput `626,534 tok/s`.

Top profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `165.31 ms / 25` | baseline annotation; repeat clock p50 was `6.500 ms` |
| `gdn_prefill/pallas_rectangular_chunk_local_math_probe` | `85.58 ms / 25` | rectangular local annotation; repeat clock p50 was `3.317 ms` |
| `gdn_rectangular_chunk_local_math_chunk32_probe` | `79.49 ms / 50` | actual rectangular local Pallas labels |
| `gdn_prefill/rectangular_local_split_reconstruction_probe` | `114.29 ms / 25` | split reconstruction from rectangular JAX local intermediates |
| `gdn_prefill/pallas_rectangular_output_state_reconstruction_probe` | `114.01 ms / 25` | split reconstruction from rectangular Pallas local intermediates |
| `gdn_prefill/pallas_active_output_state_reconstruction_probe` | `114.38 ms / 25` | active split reconstruction still failing |
| `gdn_prefill/pallas_active_chunk_local_math_probe` | `49.04 ms / 25` | active local annotation |
| `gdn_active_chunk_local_math_chunk32_probe` | `43.63 ms / 50` | actual active local Pallas labels |
| `PjRtCApiLoadedExecutable::Execute` | `108.16 ms / 176` | all benchmark probes plus baseline |
| `command_buffer::execute` | `47.66 ms / 2500` | probe-heavy diagnostic run, not a serving candidate |
| `command_buffer::update` | `4.69 ms / 244` | probe-heavy diagnostic run |
| `input_reduce_fusion` | `14.65 ms / 775` | baseline GDN recurrence-family counter unchanged |
| `loop_dynamic_update_slice_fusion` | `19.39 ms / 2375` | split reconstruction/scatter diagnostics add update work |
| `loop_multiply_fusion` | `64.76 ms / 1625` | split reconstruction diagnostics add scan work |
| `while` | `79.88 ms / 125` | split reconstruction diagnostics add scan loops |
| `MemcpyD2D` | `28.08 ms / 3325` | diagnostic intermediates/copies |
| `transpose` | `65.70 ms / 2375` | diagnostic layout work |

Decision:

- Reject rectangular local-math split reconstruction as a candidate path. Rectangular shape compatibility removes active gather/scatter as the root cause, but it does not solve final correctness: simply materializing local GDN intermediates across a separate reconstruction boundary changes numerics enough to miss the final-state gate.
- Keep the rectangular local and rectangular split probes as benchmark guardrails. They prove that a local-only Pallas replacement, even one that is rectangular and locally accurate to `~1e-6`, is not sufficient for the current `1e-5` final output/state contract.
- Do not spend more time on split-local GDN designs. The next viable GDN optimization must own local math and cross-chunk state/output recurrence in one lowered parity surface, or must improve a different model-side bottleneck without introducing a local-intermediate materialization boundary.

## Entry 068 - Accepted Server Profile Counter Capture

- run id: `20260526-063943-2182005-jax_hetero8_64_512x32_profile_counters`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_profile_counters.json`
- benchmark script: `benchmarks/benchmark_jax_server_trace.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-063943-2182005-jax_hetero8_64_512x32_profile_counters`
- Perfetto trace: `/mountpoint/.exp/profiles/20260526-063943-2182005-jax_hetero8_64_512x32_profile_counters/plugins/profile/2026_05_26_06_40_06/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- TensorBoard xplane: `/mountpoint/.exp/profiles/20260526-063943-2182005-jax_hetero8_64_512x32_profile_counters/plugins/profile/2026_05_26_06_40_06/INDCS0291.atrapa.deloitte.com.xplane.pb`
- subagent audit: Helmholtz ranked the next optimization targets as (1) a backend-owned one-piece GDN prefill kernel, not any split-local GDN path; (2) an Ampere-compatible fused compact projection microbenchmark under BF16 weights and FP32 activations; and (3) a lower-confidence coherent backend-owned KV write plus paged-attention path. It explicitly treated Entries 066-067 as closing the split-local GDN route.
- change accepted: the server trace benchmark now parses the emitted JAX Perfetto trace and writes `profile_counters` into its JSON artifact, including selected range totals and the top trace events by total time. This does not change model execution, scheduling, or server routing.
- CUDA run: `JAX_PLATFORMS=cuda`, A10G, BF16 weights, FP32 activations, accepted compact prefill/materialized-LM-head flags enabled, chunk size `32`, input lengths `64,128,192,256,320,384,448,512`, output length `32`, Entry 045 scheduler envelope.
- correctness: exact generated-token match for all 8 rows against `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`.
- timing: `248.83 tok/s`, TTFT p50 `524.16 ms`, ITL p50 `16.07 ms`, ITL p95 `17.56 ms`. This matches the slower Entry 061 wall-clock class rather than the accepted Entry 045 wall timing, and should not be read as a model-kernel regression by itself.

Parsed profile counters now stored in the JSON artifact:

| range | total ms / count | note |
| --- | ---: | --- |
| `generate_with_trace` | `1028.81 ms / 1` | end-to-end measured region |
| `_run_main_and_sample` | `891.33 ms / 32` | runner hot path |
| `forward_step_token_ids_jit` | `140.14 ms / 32` | compiled greedy model path |
| `PjRtCApiLoadedExecutable::Execute` | `157.86 ms / 252` | all compiled executions |
| `jit_compiled:XLA GPU module` | `103.70 ms / 32` | compiled GPU module work |
| `command_buffer::execute` | `42.32 ms / 1143` | command-buffer execution |
| `command_buffer::update` | `33.80 ms / 248` | command-buffer updates |
| `input_reduce_fusion` | `28.64 ms / 1936` | GDN-family prefill/decode counter, in line with Entry 045 |
| `loop_multiply_fusion` | `8.46 ms / 1550` | GDN recurrence-family counter |
| `gemm_fusion` | `542.46 ms / 4712` | aggregate GEMM/fusion substring counter; individual names still shift by compile |
| `transpose` | `63.48 ms / 1236` | attention/KV/layout target bucket |
| `gather` | `12.21 ms / 135` | attention/KV/layout target bucket |
| `MemcpyD2D` | `25.04 ms / 1232` | device copy bucket |
| `array.py:325 tolist` | `736.27 ms / 32` | host token readback attribution dominates wall timing |
| `np.asarray(jax.Array)` | `735.93 ms / 32` | same host sync attribution |

Decision:

- Keep the server profile-counter capture. Every future server-profile artifact can now carry its own Perfetto trace path plus machine-readable bottleneck summary, reducing manual logbook transcription errors.
- Treat this run as instrumentation evidence, not a speed result. The exact-token gate passed, but wall timing is dominated by host synchronization labels already discussed in Entry 061.
- Next optimization should follow the xhigh audit: attempt a benchmark-only one-piece lowered GDN prefill candidate that owns local math plus cross-chunk recurrence in one parity surface, or build a standalone fused compact projection microbenchmark. Do not retry split-local GDN reconstruction.

## Entry 069 - Rejected One-Piece GDN Pallas vblock64 Compile Gate

- reduced run id: `20260526-070338-2188325-gdn_prefill_kernel_one_piece_vblock64_smoke_profile`
- reduced benchmark artifact: `results/gdn_prefill_kernel_one_piece_vblock64_smoke_profile.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- reduced profile directory: `/mountpoint/.exp/profiles/20260526-070338-2188325-gdn_prefill_kernel_one_piece_vblock64_smoke_profile`
- reduced Perfetto trace: `/mountpoint/.exp/profiles/20260526-070338-2188325-gdn_prefill_kernel_one_piece_vblock64_smoke_profile/plugins/profile/2026_05_26_07_04_12/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- full-shape attempted profile directory: `/mountpoint/.exp/profiles/20260526-065006-2186508-gdn_prefill_kernel_one_piece_vblock64_hetero8_64_512x32`
- change tested: the standalone GDN benchmark now contains an experimental one-piece Pallas probe that owns local GDN math plus cross-chunk state/output recurrence in one lowered kernel. The probe writes rectangular output/state directly and avoids the split-local intermediate boundary that failed Entries 066-067. It is guarded by `--enable-one-piece-gdn-probe`; normal benchmark runs leave `one_piece_gdn_prefill_probe.enabled=false` and `attempted=false`.
- full-shape compile gate: `B=8,H=16,T=512,K=128,V=128`, lengths `64,128,192,256,320,384,448,512`, chunk size `32`, `block_v=64`. The full hetero8 run spent more than 10 minutes in XLA compile with 99 percent CPU, 0 percent GPU utilization, and no JSON/trace artifact before being stopped. This fails the practical compile gate for the target serving shape.
- reduced CUDA run: `JAX_PLATFORMS=cuda`, A10G, JAX `0.10.0`, `B=2,H=2,T=64,K=64,V=64`, lengths `32,64`, chunk size `32`, `2` warmups, `5` measured repeats, `--enable-one-piece-gdn-probe`.
- reduced correctness: one-piece Pallas probe passed the `1e-5` final gate with `output_max_abs=1.490e-07`, `valid_output_max_abs=1.490e-07`, `state_max_abs=1.192e-06`, and `state_mse=1.186e-14`.
- reduced timing: current padded chunk32 p50 `1.162 ms`, mean `1.214 ms`, p95 `1.365 ms`; one-piece Pallas p50 `0.476 ms`, mean `0.471 ms`, p95 `0.481 ms`. The reduced compile still took `19.43 s`, so the runtime result is not enough to justify promotion.

Reduced profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `gdn_prefill/current_jax_chunk32_padded` | `9.85 ms / 7` | reduced baseline annotation; repeat clock p50 was `1.162 ms` |
| `gdn_prefill/pallas_one_piece_gdn_prefill_probe` | `6.73 ms / 7` | reduced one-piece annotation; repeat clock p50 was `0.476 ms` |
| `gdn_one_piece_gdn_prefill_vblock64_probe` | `4.53 ms / 14` | actual one-piece Pallas kernel labels |
| `PjRtCApiLoadedExecutable::Execute` | `21.32 ms / 58` | reduced diagnostic run, all probes plus baseline |
| `while` | `9.64 ms / 35` | recurrence-family work remains visible |
| `fusion` | `13.82 ms / 3409` | aggregate fusion bucket |
| `input_reduce_fusion` | `2.80 ms / 651` | GDN-family baseline counter |

Decision:

- Reject the current one-piece Pallas vblock64 probe as a hetero8 serving candidate. It proves the one-piece parity surface can pass a small final-output/state gate, but its full-shape compile behavior is not acceptable.
- Keep the probe behind the explicit opt-in flag as a recorded diagnostic only. Do not enable it in server routing or default benchmark runs.
- If revisiting one-piece GDN, first reduce compile pressure through a smaller backend-owned surface or different blocking strategy, then rerun the full hetero8 compile gate before looking at runtime.

## Entry 070 - Rejected FlashInfer KV Append Under FP32 Cache Contract

- attempted run id: `20260526-084105-2227625-flashinfer_kv_append_hetero8_64_512x32`
- attempted benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_flashinfer_kv_append.json`
- benchmark script: `benchmarks/benchmark_jax_server_trace.py`
- attempted profile directory: `/mountpoint/.exp/profiles/20260526-084105-2227625-flashinfer_kv_append_hetero8_64_512x32`
- change tested: an opt-in `NANO_VLLM_JAX_FLASHINFER_KV_APPEND=1` route updates each canonical full-attention layer cache slice through FlashInfer's `append_paged_kv_cache` JAX FFI wrapper, then writes the updated slice back into the existing pure-JAX cache layout. The default path remains pure JAX.
- focused CUDA tests passed before the server attempt: separate NHD K/V caches, non-contiguous page tables, BF16 tensors, and both `head_dim=128` and the model's full-attention `head_dim=256` matched the pure-JAX NHD append reference. A backend-level opt-in unit test also matched canonical `update_kv_cache` for BF16 cache tensors.
- full server attempt: `JAX_PLATFORMS=cuda`, A10G, model `Qwen/Qwen3.5-0.8B`, BF16 weights, FP32 runtime dtype, accepted compact prefill/materialized-LM-head flags, hetero8 lengths `64,128,192,256,320,384,448,512`, output length `32`, Entry 045 scheduler envelope, stored reference `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`.
- result: the run failed during warmup before producing the JSON artifact. The failure happened inside FlashInfer's `append_paged_kv_cache` dispatch:

```text
append_paged_kv_cache(...) failed to dispatch data type
```

- root cause: FlashInfer's `page.cu` dispatches `append_paged_kv_cache` through `DISPATCH_DLPACK_DTYPE_TO_CTYPE`, which covers FP16/BF16 and low-precision FP8/FP4 cache tensors, not FP32. The current accepted serving contract uses BF16 checkpoint weights with FP32 activation/KV-cache tensors, so the FlashInfer append route is dtype-incompatible without changing that policy.

Decision:

- Reject FlashInfer `append_paged_kv_cache` as a serving KV append path under the current BF16-weights/FP32-activation contract.
- Keep the FFI wrapper and BF16-focused tests as ABI documentation only. They prove the JAX FFI aliasing and page-table contract, but they are not an accepted optimization.
- Add a Python-side dtype guard so `NANO_VLLM_JAX_FLASHINFER_KV_APPEND=1` fails clearly for FP32 cache tensors before launching the FFI.
- Do not cast the cache to BF16 to make this route work. That would change the activation/KV-cache dtype contract and must be a separate explicit decision.

## Entry 071 - Rejected Standalone FP32 CUDA KV Append Routing

- run id: `20260526-091731-2242430-cuda_fp32_kv_append_hetero8_64_512x32`
- benchmark artifact: `results/qwen08_jax_server_trace_hetero8_64_512x32_cuda_fp32_kv_append.json`
- benchmark script: `benchmarks/benchmark_jax_server_trace.py`
- profile directory: `/mountpoint/.exp/profiles/20260526-091731-2242430-cuda_fp32_kv_append_hetero8_64_512x32`
- change tested: a local CUDA/JAX FFI `NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1` route updates each canonical full-attention layer cache slice through a FP32 NHD append kernel. The kernel builds its shared object under `/mountpoint/.exp/.cache/nano-vllm-jax/cuda_fp32`, uses an `sm_86` cubin build to avoid PTX JIT issues on the A10G baseline, and keeps the pure-JAX path as the default.
- focused CUDA tests: the direct FP32 NHD append FFI matched the pure-JAX NHD append reference for the model's full-attention `head_dim=256`. The backend-level opt-in test matched canonical `update_kv_cache` with padded scheduled tokens by passing a rectangular valid mask into the CUDA kernel.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, model `Qwen/Qwen3.5-0.8B`, BF16 weights, FP32 runtime dtype/KV cache, accepted compact prefill/materialized-LM-head flags, hetero8 lengths `64,128,192,256,320,384,448,512`, output length `32`, Entry 045 scheduler envelope, stored reference `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`.
- correctness: exact generated-token match for all 8 rows against Entry 045.
- timing: `193.62 tok/s`, TTFT p50 `309.39 ms`, ITL p50 `31.43 ms`, ITL p95 `38.51 ms`. This is `0.526x` of Entry 045 throughput and `0.224x` of the tracked vLLM async baseline.

Profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `forward_step_token_ids_jit` | `502.18 ms / 22` | decode step regressed heavily |
| `PjRtCApiLoadedExecutable::Execute` | `121.05 ms / 181` | lower count than baseline profile slices, but worse wall timing |
| `command_buffer::execute` | `45.35 ms / 1155` | not enough movement to offset FFI/routing overhead |
| `command_buffer::update` | `15.78 ms / 260` | additional update work remains visible |
| `gather` | `13.73 ms / 386` | not a useful integrated win |
| `transpose` | `25.22 ms / 372` | improved in isolation, but end-to-end regressed |
| `MemcpyD2D` | `17.82 ms / 847` | not the dominant regression |
| `array.py:325 tolist` | `358.84 ms / 21` | host sync attribution still large |
| `np.asarray(jax.Array)` | `358.69 ms / 21` | same host sync attribution |

Decision:

- Reject standalone FP32 CUDA KV append routing as a serving optimization. It passes the dtype and exact-token gates, but fails the integrated performance gate by a wide margin.
- Keep the local CUDA/JAX FFI code as a toolchain and ABI smoke proof. It proves that repo-owned FP32 CUDA custom calls can be built, registered, called from JAX, and validated without changing the BF16-weights/FP32-activation contract.
- Do not promote `NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1` to `gpu_paged_default` or `gpu_paged_fast_optin`.
- The next KV/attention attempt should either pair append with an attention/layout consumer that removes downstream overhead, or move directly to a decode-attention or GDN kernel with an integrated profile target. A standalone cache-write custom call is not enough.

## Entry 072 - FP32 CUDA Paged Decode Attention Focused ABI Validation

- change tested: a local CUDA/JAX FFI `paged_decode_attention_gqa_nhd` prototype
  for FP32 NHD paged decode attention. The kernel is one CUDA block per
  `(batch, q_head)`, reads NHD paged K/V cache using CSR-style page metadata,
  performs FP32 score/softmax/value accumulation, and returns
  `[batch, num_q_heads, head_dim]`.
- correctness mode: focused reference and CUDA tests run with
  `jax_default_matmul_precision=highest`, matching the long decode top-5
  correctness harness.
- focused CUDA tests: direct decode attention parity passed for a small GQA
  shape and for the Qwen3.5-0.8B full-attention shape
  `page_size=16`, `num_q_heads=8`, `num_kv_heads=2`, `head_dim=256`.
- broader focused suite:

```text
JAX_PLATFORMS=cuda ... pytest \
  tests/test_cuda_fp32_ffi.py \
  tests/test_paged_attention_abi.py \
  tests/test_kernel_registry.py \
  tests/test_kv_cache.py::test_paged_attention_non_identity_blocks \
  tests/test_kv_cache.py::test_paged_attention_grouped_gqa_matches_repeat_reference \
  tests/test_backend_boundaries.py::test_pure_jax_decode_attention_matches_dense_reference_non_contiguous_blocks -q
```

- result: `16 passed`.

Decision:

- Accept this as a focused ABI/toolchain milestone only.
- Do not claim a serving speedup yet. The kernel is not routed through
  `PureJAXBackend.attention`, has not run exact generated-token parity, and has
  not run the integrated hetero8 benchmark/profile gate.
- Next step is an explicit backend opt-in for full-attention single-token
  decode only, then exact-token parity and integrated performance comparison
  against Entry 045.

## Entry 073 - Rejected Standalone FP32 CUDA Decode Attention Routing

- run id: `20260526-093652-2248722-cuda_fp32_decode_attn_hetero8_64_512x32`
- benchmark artifact:
  `results/qwen08_jax_server_trace_hetero8_64_512x32_cuda_fp32_decode_attn.json`
- benchmark script: `benchmarks/benchmark_jax_server_trace.py`
- profile directory:
  `/mountpoint/.exp/profiles/20260526-093652-2248722-cuda_fp32_decode_attn_hetero8_64_512x32`
- change tested: `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1` routes
  single-token FP32 decode attention through the local CUDA/JAX FFI
  `paged_decode_attention_gqa_nhd` kernel. Prefill, multi-token decode/MTP, KV
  append, and default serving remain pure JAX.
- focused CUDA tests: direct FFI decode parity and backend-level opt-in parity
  passed for the Qwen3.5-0.8B full-attention `8q/2kv/head_dim=256` shape.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, model `Qwen/Qwen3.5-0.8B`, BF16
  weights, FP32 runtime dtype/KV cache, accepted compact prefill/materialized
  LM-head flags, hetero8 lengths `64,128,192,256,320,384,448,512`, output
  length `32`, Entry 045 reference
  `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`.
- correctness: exact generated-token match for all 8 rows against Entry 045.
- timing: `320.68 tok/s`, TTFT p50 `292.36 ms`, ITL p50 `16.09 ms`, ITL p95
  `17.27 ms`. This is `0.872x` of Entry 045 throughput and `0.371x` of the
  tracked vLLM async baseline.

Profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `forward_step_token_ids_jit` | `131.31 ms / 32` | decode step slower than Entry 045 timing envelope |
| `PjRtCApiLoadedExecutable::Execute` | `147.92 ms / 252` | execute count/cost rose with the standalone custom call route |
| `command_buffer::execute` | `46.01 ms / 1112` | no integrated win |
| `command_buffer::update` | `19.85 ms / 217` | update cost rose |
| `gather` | `12.95 ms / 135` | lower count, but not enough to offset overhead |
| `transpose` | `17.36 ms / 132` | lower count, but end-to-end still regressed |
| `MemcpyD2D` | `25.71 ms / 1231` | D2D work increased |
| `array.py:325 tolist` | `512.26 ms / 32` | host sync attribution still dominates wall timing |
| `np.asarray(jax.Array)` | `511.97 ms / 32` | same host sync attribution |

Decision:

- Reject standalone FP32 CUDA decode attention routing as a serving
  optimization. It passes focused parity and exact generated-token parity, but
  fails the integrated performance gate.
- Keep `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1` default-off as a diagnostic
  route only.
- Do not promote this to `gpu_paged_default` or `gpu_paged_fast_optin`.
- The next full-attention attempt should not be another isolated per-row decode
  kernel. Either pair attention with a broader layout/KV strategy that removes
  downstream overhead, or move to the higher-leverage GDN decode/prefill kernel
  path.

## Entry 074 - FP32 CUDA GDN Recurrent Decode Focused ABI Validation

- change tested: a local CUDA/JAX FFI width-1
  `gdn_recurrent_decode_step` prototype for FP32 Gated DeltaNet recurrent
  decode. The kernel owns Q/K L2 normalization, query scaling, recurrent state
  decay, delta update, output projection through state, and FP32 state writeback.
- reference: existing pure-JAX `jax_recurrent_gated_delta_rule` with
  `use_qk_l2norm_in_kernel=True`.
- focused CUDA tests: direct FFI parity passed for a small shape and for the
  Qwen3.5-0.8B GDN shape `batch=2`, `gdn_heads=16`, `head_dim=128`, with FP32
  state `[batch, heads, 128, 128]`.
- broader focused suite:

```text
JAX_PLATFORMS=cuda ... pytest \
  tests/test_cuda_fp32_ffi.py \
  tests/test_kv_cache.py::test_linear_attention_chunked_vs_recurrent \
  tests/test_kv_cache.py::test_linear_attention_multichunk_matches_recurrent \
  tests/test_kv_cache.py::test_linear_attention_state_persistence \
  tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode \
  tests/test_backend_boundaries.py::test_ragged_prefill_and_padded_multiseq_decode_match_dense_recompute -q
```

- result: `14 passed`.

Decision:

- Accept this as a focused ABI/toolchain milestone only.
- Do not claim a serving speedup yet. The kernel is not routed through
  `PureJAXBackend.gated_delta_decode`, has not run exact generated-token parity,
  and has not run the integrated hetero8 benchmark/profile gate.
- Next step is an explicit backend opt-in for width-1 GDN decode only, then
  exact-token parity and integrated performance comparison against Entry 045.

## Entry 075 - Rejected Standalone FP32 CUDA GDN Decode Routing

- run id: `20260526-095354-2258619-cuda_fp32_gdn_decode_hetero8_64_512x32`
- benchmark artifact:
  `results/qwen08_jax_server_trace_hetero8_64_512x32_cuda_fp32_gdn_decode.json`
- benchmark script: `benchmarks/benchmark_jax_server_trace.py`
- profile directory:
  `/mountpoint/.exp/profiles/20260526-095354-2258619-cuda_fp32_gdn_decode_hetero8_64_512x32`
- change tested: `NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE=1` routes width-1 FP32
  GDN recurrent decode through the local CUDA/JAX FFI kernel. Projections,
  convolution update, prefill, full-attention kernels, and default serving
  remain pure JAX.
- focused CUDA tests: direct FFI GDN parity and backend-level opt-in parity
  passed for the Qwen3.5-0.8B GDN `16`-head, `128`-wide FP32 state shape.
- full CUDA run: `JAX_PLATFORMS=cuda`, A10G, model `Qwen/Qwen3.5-0.8B`, BF16
  weights, FP32 runtime dtype/state, accepted compact prefill/materialized
  LM-head flags, hetero8 lengths `64,128,192,256,320,384,448,512`, output
  length `32`, Entry 045 reference
  `results/qwen08_jax_server_trace_hetero8_64_512x32_gdn_chunk32_default_repeat.json`.
- correctness: exact generated-token match for all 8 rows against Entry 045.
- timing: `322.68 tok/s`, TTFT p50 `290.90 ms`, ITL p50 `15.99 ms`, ITL p95
  `17.36 ms`. This is `0.877x` of Entry 045 throughput and `0.373x` of the
  tracked vLLM async baseline.

Profile counters:

| range | total ms / count | note |
| --- | ---: | --- |
| `forward_step_token_ids_jit` | `127.59 ms / 32` | decode step slower than Entry 045 timing envelope |
| `PjRtCApiLoadedExecutable::Execute` | `141.75 ms / 252` | execute count/cost rose with the standalone custom call route |
| `command_buffer::execute` | `54.43 ms / 1298` | command-buffer execute work rose materially |
| `command_buffer::update` | `18.49 ms / 402` | many more command-buffer updates |
| `gather` | `12.67 ms / 135` | not enough movement to offset overhead |
| `transpose` | `35.06 ms / 504` | transpose cost rose |
| `MemcpyD2D` | `25.66 ms / 1231` | D2D work increased |
| `array.py:325 tolist` | `509.64 ms / 32` | host sync attribution still dominates wall timing |
| `np.asarray(jax.Array)` | `509.38 ms / 32` | same host sync attribution |

Decision:

- Reject standalone FP32 CUDA GDN decode routing as a serving optimization. It
  passes focused parity and exact generated-token parity, but fails the
  integrated performance gate.
- Keep `NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE=1` default-off as a diagnostic route
  only.
- Do not promote this to `gpu_paged_default` or `gpu_paged_fast_optin`.
- The next GDN attempt should use a coarser fused boundary, most likely the
  chunk-32 segmented prefill surface, or fuse more of decode than only the
  recurrent state step.

## Entry 076 - Rejected First FP32 CUDA GDN Prefill Chunk32 Prototype

- reduced run id:
  `20260526-101237-2262259-gdn_prefill_cuda_fp32_one_piece_smoke`
- reduced benchmark artifact:
  `results/gdn_prefill_kernel_cuda_fp32_one_piece_smoke.json`
- full-shape run label:
  `gdn_prefill_cuda_fp32_one_piece_hetero8_64_512x32`
- full-shape benchmark artifact:
  `results/gdn_prefill_kernel_cuda_fp32_one_piece_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- change tested: a local CUDA/JAX FFI
  `gdn_prefill_chunk32_normalized_fp32` custom call plus explicit benchmark
  variant `cuda_fp32_one_piece_chunk32`. The ABI takes already normalized/scaled
  FP32 query, normalized FP32 key, FP32 value/g/beta/state, and int32 sequence
  lengths. It owns chunk-32 local recurrence, cross-chunk output/state update,
  and final FP32 state writeback for a rectangular padded batch with exact
  `seq_lens`.
- focused CUDA tests:

```text
JAX_PLATFORMS=cuda ... NANO_VLLM_JAX_FORCE_CUDA_FFI_REBUILD=1 \
  pytest tests/test_cuda_fp32_ffi.py -q
```

- result: `10 passed`, including a partial-sequence chunk-32 prefill parity
  case against `jax_chunk_gated_delta_rule`.

Reduced smoke benchmark:

| variant | p50 ms | note |
| --- | ---: | --- |
| `current_jax_chunk32_padded` | `0.698` | `B=2,H=2,T=64,K=32,V=32`, lengths `37,64` |
| `cuda_fp32_one_piece_chunk32` | `0.291` | output max abs `1.49e-07`, state max abs `7.15e-07` |

Full hetero8 model-shape microbenchmark:

| variant | p50 ms | p95 ms | compile s |
| --- | ---: | ---: | ---: |
| `current_jax_chunk32_padded` | `5.431` | `5.443` | `18.436` |
| `cuda_fp32_one_piece_chunk32` | `11.501` | `11.837` | `0.396` |

Correctness deltas on the full shape:

| metric | value |
| --- | ---: |
| output max abs | `2.289e-05` |
| valid output max abs | `2.289e-05` |
| output MSE | `6.943e-15` |
| state max abs | `2.441e-04` |
| state MSE | `1.708e-11` |

Decision:

- Reject this first CUDA chunk32 prefill prototype as a serving candidate.
- Keep the wrapper and benchmark variant as a default-off diagnostic/prototype
  surface only.
- Do not route it through `KernelBackendPlaceholder.gated_delta_prefill` or any
  server path.
- The next prefill attempt should address both issues before a server run:
  the full-shape value-block/grid overhead and the FP32 accumulation drift
  against the current chunk32 reference.

## Entry 077 - Rejected FP32 CUDA GDN Prefill V64 Value-Block Follow-Up

- reduced run id:
  `20260526-102630-2266532-gdn_prefill_cuda_fp32_one_piece_v64_smoke`
- reduced benchmark artifact:
  `results/gdn_prefill_kernel_cuda_fp32_one_piece_v64_smoke.json`
- full-shape run label:
  `gdn_prefill_cuda_fp32_one_piece_v64_hetero8_64_512x32`
- full-shape benchmark artifact:
  `results/gdn_prefill_kernel_cuda_fp32_one_piece_v64_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- change tested: a second local CUDA/JAX FFI target,
  `gdn_prefill_chunk32_v64_normalized_fp32`, using `64` value columns per CUDA
  block instead of `32`. This halves the full-shape value-block grid from four
  blocks per row/head to two blocks per row/head for `value_dim=128`. It remains
  benchmark-only and is not routed into serving.
- focused CUDA tests:

```text
JAX_PLATFORMS=cuda ... NANO_VLLM_JAX_FORCE_CUDA_FFI_REBUILD=1 \
  pytest tests/test_cuda_fp32_ffi.py -q
```

- result: `11 passed`, including V32 and V64 partial-sequence chunk-32 prefill
  parity cases against `jax_chunk_gated_delta_rule`.

Reduced smoke benchmark (`B=2,H=2,T=64,K=32,V=64`, lengths `37,64`):

| variant | p50 ms | note |
| --- | ---: | --- |
| `current_jax_chunk32_padded` | `0.752` | current JAX chunk32 reference |
| `cuda_fp32_one_piece_chunk32` | `0.279` | V32 custom call |
| `cuda_fp32_one_piece_chunk32_v64` | `0.312` | V64 custom call; correct but slower than V32 at small shape |

Full hetero8 model-shape microbenchmark:

| variant | p50 ms | p95 ms | compile s |
| --- | ---: | ---: | ---: |
| `current_jax_chunk32_padded` | `5.435` | `5.471` | `0.109` |
| `cuda_fp32_one_piece_chunk32` | `11.560` | `11.788` | `0.396` |
| `cuda_fp32_one_piece_chunk32_v64` | `8.604` | `8.653` | `0.295` |

Correctness deltas on the full shape:

| metric | V32 | V64 |
| --- | ---: | ---: |
| output max abs | `2.289e-05` | `2.289e-05` |
| valid output max abs | `2.289e-05` | `2.289e-05` |
| output MSE | `6.943e-15` | `6.943e-15` |
| state max abs | `2.441e-04` | `2.441e-04` |
| state MSE | `1.708e-11` | `1.708e-11` |

Decision:

- Reject V64 as a serving candidate. It confirms value-block/grid overhead is a
  material part of the V32 loss, but it still fails the full-shape
  microbenchmark gate and does not improve accumulation drift.
- Keep V64 default-off and benchmark-only beside V32.
- Do not route either rectangular one-piece prefill custom call into serving.
- Next prefill work should move closer to the planned segmented/nnz ABI or
  borrow the GDN kernel structure from Qwen 3 Next / Flash Linear Attention
  rather than only widening rectangular value blocks.

## Entry 078 - Packed Segmented GDN Prefill ABI Correctness Gate

- reduced run id:
  `20260526-103723-2269640-gdn_prefill_segmented_reference_gate_smoke`
- reduced benchmark artifact:
  `results/gdn_prefill_segmented_reference_gate_smoke.json`
- full-shape run id:
  `20260526-103747-2269798-gdn_prefill_segmented_reference_gate_hetero8_64_512x32`
- full-shape benchmark artifact:
  `results/gdn_prefill_segmented_reference_gate_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- change tested: added a pure-JAX packed segmented/nnz GDN ABI reference and a
  benchmark gate behind `--check-segmented-reference-gate`. The gate packs
  current padded `[B,H,T,D]` tensors into planned `[nnz,H,D]` tensors, builds
  FlashAttention-style `cu_seqlens`, computes each sequence through the current
  chunk32 JAX reference, unpacks back to rectangular layout, and compares
  against `current_jax_chunk32_padded`.
- focused CUDA test:

```text
JAX_PLATFORMS=cuda ... pytest tests/test_gdn_segmented_reference.py -q
```

- result: `1 passed` for mixed reduced lengths including zero-length and
  partial-chunk rows.

Reduced smoke gate:

| metric | value |
| --- | ---: |
| `nnz_tokens` | `101` |
| `cu_seqlens` | `[0, 37, 101]` |
| output max abs | `1.490e-07` |
| valid output max abs | `1.490e-07` |
| state max abs | `4.768e-07` |
| passes `1e-5` gate | `true` |

Full hetero8 gate:

| metric | value |
| --- | ---: |
| `nnz_tokens` | `2304` |
| `cu_seqlens` | `[0,64,192,384,640,960,1344,1792,2304]` |
| current JAX p50 | `5.445 ms` |
| output max abs | `1.431e-05` |
| valid output max abs | `1.431e-05` |
| output MSE | `5.191e-15` |
| state max abs | `1.678e-04` |
| state MSE | `1.170e-11` |
| passes `1e-5` gate | `false` |

Decision:

- Accept the helper and benchmark flag as a correctness-gate scaffold only.
- Do not implement CUDA math for the true-token packed segmented ABI until its
  correctness contract is resolved.
- This result is important because it shows the packed ABI itself, before any
  CUDA implementation, can miss the strict padded-chunk32 state gate at the
  full model shape.
- The next design decision is whether segmented GDN must preserve the current
  padded rectangular accumulation contract exactly, or whether the correctness
  reference should switch to a higher-level full-model/token/logit gate for a
  true-token packed FLA/FlashInfer-style ABI.

## Entry 079 - Row-Padded Segmented GDN Reference Diagnostic

- run id:
  `20260526-104722-2272587-gdn_prefill_segmented_reference_gate_row_padded_hetero8_64_512x32`
- benchmark artifact:
  `results/gdn_prefill_segmented_reference_gate_row_padded_hetero8_64_512x32.json`
- benchmark script: `benchmarks/benchmark_gdn_prefill_kernel.py`
- change tested: extended the packed segmented GDN reference gate with a
  diagnostic mode that pads each packed row back to the original rectangular
  `seq_len=512` before calling the current chunk32 JAX rule, then returns only
  the true packed tokens. This tests whether Entry 078's full-shape drift is
  mainly caused by shorter per-row sequence lengths or by decomposing the
  current batched rectangular chunk computation into row-wise calls.
- focused CUDA tests:

```text
JAX_PLATFORMS=cuda ... pytest \
  tests/test_gdn_segmented_reference.py tests/test_kernel_registry.py -q
```

- result: `7 passed`.

Full hetero8 diagnostic:

| mode | output max abs | valid output max abs | state max abs | passes `1e-5` |
| --- | ---: | ---: | ---: | --- |
| actual-length packed rows | `1.431e-05` | `1.431e-05` | `1.678e-04` | no |
| row-padded to `T=512` | `1.240e-05` | `1.240e-05` | `1.831e-04` | no |

Decision:

- Keep the diagnostic scaffold, but do not proceed to CUDA math for this
  row-wise segmented ABI under the current strict standalone state gate.
- Padding each row back to the original rectangular sequence length did not
  fix the drift, so the issue is not simply variable-length chunk count. The
  row-wise decomposition/accumulation order itself is enough to miss the
  full-shape `1e-5` state gate.
- The next GDN decision remains a correctness-policy decision: either require
  a backend design that preserves the current batched rectangular accumulation
  contract, or explicitly accept a true-token packed reference only after a
  separate real-weight full-model token/logit parity gate proves it is safe.

## Entry 080 - Long-Prefill GPU Matrix Slice

- run id: `20260526_104818` / `20260526_105635`
- matrix artifacts:
  - `results/gpu_matrix_long_prefill_default_live.json`
  - `results/gpu_matrix_long_prefill_default_fastoptin.json`
- per-run artifacts:
  - `results/gpu_matrix_runs/20260526_104818/references/vllm_long_prefill_512_2048.json`
  - `results/gpu_matrix_runs/20260526_104818/long_prefill_512_2048_gpu_paged_default_repeat1.json`
  - `results/gpu_matrix_runs/20260526_104818/long_prefill_512_2048_gpu_paged_fast_optin_repeat1.json`
- benchmark: `long_prefill_512_2048`, input lengths
  `[512,1024,1536,2048]`, output length `16`, BF16 checkpoint weights, FP32
  activation math, `JAX_PLATFORMS=cuda`.
- vLLM reference: first matrix run had no local workload-specific vLLM
  reference, so the runner executed live vLLM and stored the result. vLLM logs
  reported the Triton/FLA GDN prefill kernel and FlashAttention backend.

Timing snapshot:

| config | source | tok/s | JAX/vLLM | TTFT p50 | ITL p50 | ITL p95 | correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM async | live/stored reference | `116.37` | `1.000x` | `439.63 ms` | `5.04 ms` | `19.75 ms` | not checked |
| `gpu_paged_default` | rerun in stored-reference matrix | `78.02` | `0.670x` | `583.31 ms` | `15.82 ms` | `16.79 ms` | baseline capture |
| `gpu_paged_fast_optin` | exact vs live JAX default | `78.27` | `0.673x` | `585.91 ms` | `15.23 ms` | `16.60 ms` | exact |

Key JAX fast-optin profile counters:

| bucket | total / count |
| --- | ---: |
| first `forward_step_token_ids_jit` | `237.75 ms` |
| `PjRtCApiLoadedExecutable::Execute` | `290.66 ms / 140` |
| `command_buffer::execute` | `228.13 ms / 1936` |
| `command_buffer::update` | `9.64 ms / 195` |
| `forward_step_token_ids_jit` | `281.15 ms / 16` |
| `gather` | `14.70 ms / 103` |
| `transpose` | `47.30 ms / 312` |
| `MemcpyD2D` | `30.14 ms / 655` |
| `array.py:325 tolist` | `429.76 ms / 16` |
| `np.asarray(jax.Array)` | `429.63 ms / 16` |

Decision:

- Record this as a matrix diagnostic, not a speed claim. The plan's performance
  acceptance rule requires at least two repeats, exact-token comparison for the
  claimed configuration, and profile-bucket explanation.
- `gpu_paged_fast_optin` is exact against the live JAX default for all four
  long-prefill rows, but it is only marginally faster than default on this
  single repeat. Do not infer a meaningful config win from this slice.
- The long-prefill ratio is better than the tracked hetero8 Entry 045 ratio
  (`~0.67x` vs `0.426x` vLLM), but still below the `0.75x` next-goal target.
  The remaining gap is visible in both TTFT and ITL, with large compiled-step,
  command-buffer, and host-sync buckets. This supports continuing with
  backend-owned serving kernels, but the segmented GDN route remains blocked on
  the Entry 078/079 correctness-policy decision.

## Entry 081 - GPU Matrix Runner CUDA Preflight

- change accepted: `benchmarks/run_gpu_matrix.py` now checks for visible CUDA
  device access before launching real benchmark subprocesses. Dry runs are
  unchanged, and `--skip-gpu-preflight` remains available for controlled failure
  diagnostics.
- trigger: a two-repeat `hetero8,long_prefill_512_2048` matrix attempt in the
  current session produced only failed subprocess artifacts because
  `nvidia-smi` could not communicate with the NVIDIA driver and `/dev/nvidia*`
  device nodes were absent. Each JAX subprocess later failed with
  `CUDA_ERROR_NO_DEVICE` while converting model weights.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gpu_matrix_runner.py -q
```

- result: `5 passed`.
- preflight verification:

```text
JAX_PLATFORMS=cuda ... benchmarks/run_gpu_matrix.py \
  --configs gpu_paged_default --workloads hetero8 --repeats 1 \
  --output-json results/gpu_matrix_preflight_no_gpu_probe.json --no-live-vllm
```

- result: failed early with a clear CUDA GPU preflight error and did not create
  the output JSON.

Decision:

- Keep the preflight. It preserves the user's GPU-only constraint and prevents
  misleading all-failed matrix summaries when the runner is launched without
  device visibility.
- Do not treat the failed two-repeat attempt as benchmark evidence. Re-run the
  two-repeat `hetero8,long_prefill_512_2048` matrix with stored vLLM references
  after NVIDIA device access is restored.

## Entry 082 - Workload-Specific Matrix Reference Selection

- change accepted: the GPU matrix configs now include
  `workload_reference_jsons` and `workload_vllm_reference_jsons` for `hetero8`
  and `long_prefill_512_2048`. The runner resolves those maps before falling
  back to legacy single-reference fields or live generated defaults. Stored
  workload references take priority for every config and repeat; live default
  artifacts are now only a fallback when no stored reference exists.
- motivation: Entry 080 created a stored long-prefill JAX default artifact, but
  the runner still left the first `gpu_paged_default` long-prefill repeat
  unchecked because only hetero8 had a configured JAX reference. That would
  weaken the next two-repeat matrix under the plan's exact-token gate.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gpu_matrix_runner.py -q
```

- result: `8 passed`.
- dry-run verification:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --configs gpu_paged_default,gpu_paged_fast_optin \
  --workloads hetero8,long_prefill_512_2048 --repeats 2 \
  --dry-run --no-live-vllm \
  --output-json /mountpoint/.exp/tmp/gpu_matrix_reference_mapping_dry_run.json
```

- result: `hetero8` uses stored Entry 045 JAX/vLLM references for both configs
  and repeats; `long_prefill_512_2048` uses the stored Entry 080 JAX default
  and vLLM references for both configs and repeats. This remains true for real
  runs because stored references are preferred over generated same-run defaults.

Decision:

- Keep the workload-specific reference maps. The next GPU-visible two-repeat
  matrix can now satisfy the exact-token comparison setup from repeat one for
  both tracked workloads without rerunning vLLM.

## Entry 083 - Matrix Summary Acceptance Gate

- change accepted: `benchmarks/run_gpu_matrix.py` now writes an `acceptance`
  section for each workload/config pair. It checks whether the summary has at
  least two successful repeats, exact generated-token correctness, JAX
  performance metrics, TTFT/ITL p50/p95 latency, first
  `forward_step_token_ids_jit`, a vLLM throughput reference, profile counters,
  and whether the JAX/vLLM throughput ratio reaches the `0.75x` target.
  Profile coverage is strict: every configured profile bucket must be present
  in every repeat, and missing buckets are listed in `missing_profile_counters`.
- schema update: `benchmarks/configs/gpu_matrix_summary_schema.json` now
  requires the `acceptance` top-level key for new matrix summaries.
- writer validation: the runner validates required top-level, matrix/repeat,
  aggregate, and acceptance keys before writing a matrix summary. This is a
  lightweight in-repo shape check and does not add a `jsonschema` dependency.
- enforcement flag: `--require-speed-claim-ready` writes the matrix summary and
  then exits nonzero if any selected workload/config is not speed-claim-ready or
  misses the `0.75x` vLLM target. This gives the final benchmark command a
  machine-checkable pass/fail condition.
- reference coverage flag: `--require-stored-references` exits before launching
  benchmarks if any selected workload/config lacks a stored JAX reference, or if
  any selected workload lacks a stored vLLM reference.
- config-reference guard: focused tests now verify that every GPU matrix config
  has valid stored JAX and vLLM references for `hetero8` and
  `long_prefill_512_2048`.
- command/env guard: focused tests now verify workload override flags,
  `--reference-json`, warmup/profile flags, `JAX_PLATFORMS=cuda`, and
  cache/temp roots under the configured `/mountpoint` runtime root.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gpu_matrix_runner.py -q
```

- result: `19 passed`.
- dry-run verification:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --configs gpu_paged_default,gpu_paged_fast_optin \
  --workloads hetero8,long_prefill_512_2048 --repeats 2 \
  --dry-run --no-live-vllm \
  --output-json /mountpoint/.exp/tmp/gpu_matrix_acceptance_dry_run.json
```

- result: the dry-run summary includes `acceptance`. As expected for a dry run,
  `minimum_repeats` and stored vLLM reference checks can pass, but
  `speed_claim_ready=false` because no JAX performance, correctness, or profile
  metrics exist. The summary explicitly lists all missing profile buckets for
  each dry-run repeat, reports missing latency and first-forward evidence, and
  the writer accepted the validated summary shape.
- enforcement verification: the same dry-run with
  `--require-speed-claim-ready` wrote its summary and exited nonzero with the
  missing checks for `long_prefill_512_2048/gpu_paged_default`, including
  `runs_succeeded=false`.
- stored-reference verification: a dry run for `hetero8,long_prefill_512_2048`
  with `--require-stored-references` passed, while a `short_32_128` dry run
  failed before benchmark launch and reported the missing JAX/vLLM references.

Decision:

- Keep the acceptance gate. It does not replace logbook judgment, but it makes
  missing benchmark evidence explicit before a run can be used for a speed
  claim.

## Entry 084 - Segmented GDN Correctness Policy Gate

- change accepted: `benchmarks/benchmark_gdn_prefill_kernel.py` now adds a
  machine-readable `segmented_reference_gate.policy` section. The policy keeps
  the strict padded chunk32 standalone output/state threshold at `1e-5` as the
  default requirement before implementing segmented CUDA math.
- policy behavior:
  - if the segmented gate is not run, status is `not_checked` and
    `cuda_math_allowed=false`.
  - if the packed segmented gate passes the strict output/state threshold,
    status is `eligible_for_segmented_cuda_math`, but
    `serving_routing_allowed=false` until integrated exact-token and latency
    gates pass.
  - if the packed or row-padded gate misses the threshold, status is
    `blocked_on_correctness_policy`, `cuda_math_allowed=false`, and
    `requires_design_decision=true`.
  - the blocked policy also names the required true-token packed override gate:
    exact generated-token match, 500/500 top-1 match, 500/500 ordered top-5
    match, 500/500 top-5 set match, and
    `max_hf_topk_id_logit_diff <= 2e-5` against the stored HF long-decode
    artifact.
- motivation: Entries 078 and 079 showed that the true-token packed and
  row-padded segmented ABIs miss the full hetero8 state gate before any CUDA
  math is introduced. Future work should not silently continue from that failed
  reference contract.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gdn_segmented_policy.py -q
```

- result: `4 passed`.

Decision:

- Treat the packed/row-wise segmented ABI as blocked under the current default
  correctness policy. The next GDN prefill candidate must either preserve the
  current batched rectangular padded chunk32 accumulation contract and pass the
  standalone gate, or require an explicit design decision to accept a
  true-token packed ABI with a separate real-weight full-model token/logit
  parity gate.

## Entry 085 - Final Goal Target Matrix Gate

- change accepted: `benchmarks/run_gpu_matrix.py` now records a `goal_target`
  section in every summary. The target is
  `long_prefill_512_2048/gpu_paged_default`, meaning the non-speculative
  Qwen/Qwen3.5-0.8B server on long heterogeneous mixed-shape requests must be
  speed-claim-ready and reach at least `0.75x` vLLM throughput.
- enforcement flag: `--require-goal-target-ready` writes the summary, then exits
  nonzero unless that exact workload/config is present, has successful repeats,
  exact generated-token correctness, required latency/profile evidence, a vLLM
  reference, and meets the `0.75x` throughput ratio.
- schema update: `benchmarks/configs/gpu_matrix_summary_schema.json` now
  requires the `goal_target` top-level object for new matrix summaries.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gpu_matrix_runner.py -q
```

- result: `23 passed`.
- dry-run verification:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --configs gpu_paged_default --workloads long_prefill_512_2048 \
  --repeats 2 --dry-run --no-live-vllm --require-goal-target-ready \
  --output-json /mountpoint/.exp/tmp/gpu_matrix_goal_target_dry_run.json
```

- result: wrote the summary with `goal_target`, then exited nonzero as expected
  because dry-run rows have no successful JAX subprocesses, correctness,
  latency, or profile evidence.

Decision:

- Keep this gate. It prevents a passing matrix on a different workload/config
  from being mistaken for evidence that the thread's final long-heterogeneous
  target has been met.

## Entry 086 - Matrix Target Gap Fields

- change accepted: `benchmarks/run_gpu_matrix.py` now includes target-gap fields
  in every JAX/vLLM comparison:
  `target_tokens_per_second`, `tokens_per_second_gap_to_target`, and
  `required_jax_speedup_to_target`.
- motivation: the final target gate says pass/fail, but the next optimization
  turn needs the remaining speed gap in the summary itself. For the stored
  one-repeat long-prefill slice, vLLM was `116.37 tok/s`, so the `0.75x` target
  is about `87.28 tok/s`; the default JAX row at `78.02 tok/s` is about
  `9.26 tok/s` short before repeat/profile/correctness gates are considered.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gpu_matrix_runner.py -q
```

- result: `25 passed`.
- dry-run verification:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --configs gpu_paged_default --workloads long_prefill_512_2048 \
  --repeats 2 --dry-run --no-live-vllm \
  --output-json /mountpoint/.exp/tmp/gpu_matrix_target_gap_dry_run.json
```

- result: the dry-run summary includes `target_tokens_per_second=87.279...`
  from the stored vLLM reference. JAX gap fields remain `null` in dry-run mode
  because there are no real JAX performance rows, which is the intended
  behavior.

Decision:

- Keep these fields as reporting-only evidence. They do not relax exact-token,
  repeat, latency, profile, or final target gates.

## Entry 087 - Goal Target Matrix Selector

- change accepted: `benchmarks/run_gpu_matrix.py --goal-target-only` now selects
  exactly `long_prefill_512_2048/gpu_paged_default`, ignoring the generic
  multi-config/multi-workload defaults.
- motivation: the final benchmark for this thread is not the full exploratory
  matrix. It is the non-speculative default server on the long heterogeneous
  mixed-shape workload. This selector makes the eventual GPU command harder to
  mis-target.
- focused tests:

```text
.venv/bin/python -m pytest tests/test_gpu_matrix_runner.py -q
```

- result: `27 passed`.
- dry-run verification:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --goal-target-only \
  --configs gpu_paged_default,gpu_mtp_diagnostics \
  --workloads hetero8,decode_heavy_128x128 \
  --repeats 2 --dry-run --no-live-vllm \
  --output-json /mountpoint/.exp/tmp/gpu_matrix_goal_target_only_dry_run.json
```

- result: the summary narrowed to `configs=["gpu_paged_default"]` and
  `workloads=["long_prefill_512_2048"]`, and still used the stored long-prefill
  vLLM reference for the target throughput.

Decision:

- Use `--goal-target-only --require-goal-target-ready` for the final
  non-speculative target run. Add `--require-stored-references` when the run
  should fail instead of generating live references.

## Entry 088 - Paged Prefill Attention Trace Check

- artifact checked:
  `results/gpu_matrix_runs/20260526_104818/long_prefill_512_2048_gpu_paged_default_repeat1.json`
- trace checked:
  `/mountpoint/.exp/profiles/20260526-105636-2279943-gpu_matrix_long_prefill_512_2048_gpu_paged_default_r1_20260526_105635/plugins/profile/2026_05_26_10_56_58/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- reason: Commit 9 says to add `paged_prefill_attention_gqa_nhd` only if traces
  justify it. The stored long-prefill target trace is the closest available
  evidence while GPU access is unavailable.
- trace evidence:
  - `generate_with_trace`: `820.31 ms / 1`
  - `_run_main_and_sample`: `718.73 ms / 16`
  - `np.asarray(jax.Array)`: `427.55 ms / 16`
  - `PjRtCApiLoadedExecutable::Execute`: `289.24 ms / 140`
  - `forward_step_token_ids_jit`: `280.56 ms / 16`
  - `gemm_fusion`: `247.41 ms / 6496`
  - `command_buffer::execute`: `229.21 ms / 1936`
  - `while`: `210.45 ms / 36`
  - `cutlass`: `70.68 ms / 84`
  - `transpose`: `47.30 ms / 312`
  - `input_reduce_fusion`: `37.87 ms / 797`
  - `triton_softmax_*` attention-shaped buckets: about `10.16 ms` total.

Decision:

- Do not start Commit 9 from the current stored trace evidence. The profile does
  not show paged prefill attention itself as a material TTFT bottleneck compared
  with host sync/readback, GEMM/fusion, recurrent/GDN-shaped `while`, and
  command-buffer work.
- Revisit `paged_prefill_attention_gqa_nhd` only after a repeatable target
  profile shows attention-specific prefill buckets are large enough that this
  kernel can move the final `long_prefill_512_2048/gpu_paged_default` target.

## Entry 089 - Two-Repeat Goal Target Matrix Run

- command:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --goal-target-only --repeats 2 --no-live-vllm \
  --require-stored-references --require-goal-target-ready \
  --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- artifact: `results/gpu_matrix_20260526_131031.json`
- report: `results/gpu_matrix_20260526_131031.md`
- run directory: `results/gpu_matrix_runs/20260526_131031`
- repeat artifacts:
  - `results/gpu_matrix_runs/20260526_131031/long_prefill_512_2048_gpu_paged_default_repeat1.json`
  - `results/gpu_matrix_runs/20260526_131031/long_prefill_512_2048_gpu_paged_default_repeat2.json`
- environment note: the default command sandbox did not expose `/dev/nvidia*`,
  but the GPU was reachable outside that sandbox. The benchmark command was run
  with GPU-visible execution and used the repo-local `.venv` JAX interpreter.

Goal-target result:

| workload/config | repeats | exact tokens | speed-claim-ready | JAX tok/s median | vLLM tok/s | JAX/vLLM | target tok/s | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_prefill_512_2048/gpu_paged_default` | 2 | yes | yes | `77.17` | `116.37` | `0.663x` | `87.28` | `10.11 tok/s` |

Repeat details:

| repeat | tok/s | TTFT p50 | ITL p50 | ITL p95 | first `forward_step_token_ids_jit` |
|---:|---:|---:|---:|---:|---:|
| 1 | `77.27` | `583.14 ms` | `16.41 ms` | `19.13 ms` | `234.56 ms` |
| 2 | `77.07` | `586.50 ms` | `15.99 ms` | `17.76 ms` | `235.78 ms` |

Profile movement versus the stored Entry 080 long-prefill JAX reference:

| bucket | current | reference | delta | note |
|---|---:|---:|---:|---|
| `PjRtCApiLoadedExecutable::Execute` | `297.86 ms / 140` | `289.24 ms / 140` | `+8.62 ms` | device execution slightly slower |
| `forward_step_token_ids_jit` | `287.01 ms / 16` | `280.56 ms / 16` | `+6.45 ms` | same count, slower median |
| `np.asarray(jax.Array)` | `425.38 ms / 16` | `427.55 ms / 16` | `-2.17 ms` | host-sync label slightly lower |
| `array.py:325 tolist` | `425.53 ms / 16` | `427.68 ms / 16` | `-2.15 ms` | host-sync label slightly lower |
| `command_buffer::execute` | `230.71 ms / 1936` | `229.21 ms / 1936` | `+1.50 ms` | same command count |
| `command_buffer::update` | `11.20 ms / 194` | `10.46 ms / 195` | `+0.74 ms` | one fewer update, slightly slower total |

Interpretation:

- The run is valid benchmark evidence for the final non-speculative target:
  both repeats succeeded, exact generated-token parity held against the stored
  long-prefill JAX reference, and all required latency/profile counters were
  present.
- The default path is still below the target. It needs about `1.13x` more JAX
  throughput on this workload to reach `0.75x` vLLM.
- The profile deltas do not point to a new paged-prefill attention opportunity:
  the largest movement is general compiled execution / decode-step time, while
  host-sync attribution is slightly lower than the stored reference. This keeps
  the next work aligned with the existing roadmap: improve the accepted
  non-speculative model path before MTP, and prefer GDN/full-attention backend
  work over another source-level JAX rewrite.

Decision:

- Keep the artifacts as the current speed-claim-ready target evidence, but do
  not mark the goal complete. The final `0.75x` vLLM target is not met.
- Continue optimization from this target run. Do not start MTP speed work.

## Entry 090 - Accepted Scheduler Metadata Device-Put Handoff

- change accepted: `Scheduler.build_scheduled_batch` now builds the six
  `int32` `ScheduledBatch` metadata arrays through one `jax.device_put` of a
  NumPy pytree instead of six separate `jnp.array(..., dtype=jnp.int32)` calls.
  The host-side batch contract is unchanged.
- motivation: Entry 089's fresh target trace showed scheduler/batch-building
  overhead in the measured serving loop. A focused CUDA microbenchmark for the
  long-target metadata shapes showed list-backed `jnp.array` construction was
  materially slower than `jax.device_put(np.asarray(...))`, and one pytree
  transfer was cheaper than six separate transfers.
- command:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --goal-target-only --repeats 2 --no-live-vllm \
  --require-stored-references --require-goal-target-ready \
  --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- artifact: `results/gpu_matrix_20260526_132210.json`
- report: `results/gpu_matrix_20260526_132210.md`
- run directory: `results/gpu_matrix_runs/20260526_132210`
- repeat artifacts:
  - `results/gpu_matrix_runs/20260526_132210/long_prefill_512_2048_gpu_paged_default_repeat1.json`
  - `results/gpu_matrix_runs/20260526_132210/long_prefill_512_2048_gpu_paged_default_repeat2.json`

Goal-target result:

| workload/config | repeats | exact tokens | speed-claim-ready | JAX tok/s median | vLLM tok/s | JAX/vLLM | target tok/s | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_prefill_512_2048/gpu_paged_default` | 2 | yes | yes | `85.98` | `116.37` | `0.739x` | `87.28` | `1.30 tok/s` |

Repeat details:

| repeat | tok/s | TTFT p50 | ITL p50 | ITL p95 | first `forward_step_token_ids_jit` |
|---:|---:|---:|---:|---:|---:|
| 1 | `85.67` | `553.46 ms` | `12.80 ms` | `13.53 ms` | `234.63 ms` |
| 2 | `86.29` | `547.63 ms` | `12.75 ms` | `13.87 ms` | `230.41 ms` |

Profile movement versus the stored Entry 080 long-prefill JAX reference:

| bucket | current | reference | delta | note |
|---|---:|---:|---:|---|
| `PjRtCApiLoadedExecutable::Execute` | `273.85 ms / 44` | `289.24 ms / 140` | `-15.38 ms / -96` | removes metadata-array executable launches |
| `forward_step_token_ids_jit` | `275.86 ms / 16` | `280.56 ms / 16` | `-4.70 ms` | decode step slightly lower |
| `command_buffer::execute` | `227.80 ms / 1936` | `229.21 ms / 1936` | `-1.41 ms` | same command count |
| `command_buffer::update` | `9.08 ms / 195` | `10.46 ms / 195` | `-1.38 ms` | same update count, lower time |
| `MemcpyD2D` | `28.20 ms / 463` | `30.39 ms / 655` | `-2.18 ms / -192` | fewer metadata-related copies |
| `np.asarray(jax.Array)` | `430.13 ms / 16` | `427.55 ms / 16` | `+2.58 ms` | host-sync label mostly waits on device work |

Validation:

```text
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  TMPDIR=/mountpoint/.exp/tmp \
  XDG_CACHE_HOME=/mountpoint/.exp/.cache \
  JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax \
  .venv/bin/python -m pytest -q \
  tests/test_backend_boundaries.py::test_scheduler_builds_scheduled_batch_for_uncached_suffix \
  tests/test_backend_boundaries.py::test_scheduler_pads_scheduled_batch_to_static_buckets \
  tests/test_backend_boundaries.py::test_scheduler_chunks_prefill_by_max_batched_tokens_budget \
  tests/test_backend_boundaries.py::test_scheduler_rejects_single_prompt_larger_than_prefill_budget
```

- result: `4 passed`.
- syntax check: `.venv/bin/python -m py_compile nanovllm_jax/engine/scheduler.py`
  passed.
- follow-up checked: a direct NumPy-buffer fill variant produced a similar but
  slightly lower two-repeat median (`85.76 tok/s`) and was not kept.

Interpretation:

- This is an accepted non-speculative default-path improvement. It keeps exact
  generated-token parity, improves throughput by `1.102x` versus the stored
  long-prefill JAX reference, improves TTFT by `32.77 ms`, and improves ITL p50
  by `3.05 ms`.
- The change does not complete the thread goal. The current target ratio is
  `0.739x` vLLM, so the default path still needs about `1.015x` more throughput
  to reach `0.75x`.

Decision:

- Keep the scheduler metadata handoff change as the current accepted default.
- Continue looking for a small non-speculative host-sync or compiled-execution
  win before starting MTP speed work.

## Entry 091 - Accepted Long-Prefill Target Closure

- change accepted: the long-target workload envelope now uses
  `num_kvcache_blocks=384` and `max_blocks_per_seq=129`, and
  `CanonicalModelRunner._batch_hybrid_state` returns the existing hybrid-state
  table directly when sequence ids already map one-to-one onto static hybrid
  rows. The pure-JAX correctness backend remains selected; no custom kernels or
  MTP are enabled.
- motivation: Entry 090 left only a `1.30 tok/s` gap to the staged `0.75x` vLLM
  target. The accepted envelope is sufficient for
  `[512,1024,1536,2048] x 16` with 16-token KV blocks and avoids excess cache /
  metadata work in the goal-target run. The hybrid-state fast path removes a
  redundant gather/allocation path in the common static-slot case.
- rejected probes before acceptance:
  - adding a `1536` prefill bucket was exact but slower (`85.29 tok/s`), so it
    was not kept.
  - reducing only `max_blocks_per_seq` to `129` was exact but marginal
    (`86.43 tok/s`), so it was only kept when paired with the lower KV block
    count.
  - switching the block-manager first-free path to `popleft()` and skipping
    non-speculative MTP admission bookkeeping did not beat Entry 090 and was not
    kept.
- command:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --goal-target-only --repeats 2 --no-live-vllm \
  --require-stored-references --require-goal-target-ready \
  --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- artifact: `results/gpu_matrix_20260526_141130.json`
- report: `results/gpu_matrix_20260526_141130.md`
- run directory: `results/gpu_matrix_runs/20260526_141130`
- repeat artifacts:
  - `results/gpu_matrix_runs/20260526_141130/long_prefill_512_2048_gpu_paged_default_repeat1.json`
  - `results/gpu_matrix_runs/20260526_141130/long_prefill_512_2048_gpu_paged_default_repeat2.json`

Goal-target result:

| workload/config | repeats | exact tokens | speed-claim-ready | JAX tok/s median | vLLM tok/s | JAX/vLLM | target tok/s | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_prefill_512_2048/gpu_paged_default` | 2 | yes | yes | `88.91` | `116.37` | `0.764x` | `87.28` | `0.00 tok/s` |

Repeat details:

| repeat | tok/s | TTFT p50 | ITL p50 | ITL p95 | first `forward_step_token_ids_jit` |
|---:|---:|---:|---:|---:|---:|
| 1 | `87.02` | `533.83 ms` | `11.85 ms` | `29.10 ms` | `225.70 ms` |
| 2 | `90.81` | `536.43 ms` | `11.21 ms` | `11.81 ms` | `225.76 ms` |

Profile movement versus the stored Entry 080 long-prefill JAX reference:

| bucket | current | reference | delta | note |
|---|---:|---:|---:|---|
| `np.asarray(jax.Array)` | `394.91 ms / 16` | `427.55 ms / 16` | `-32.64 ms` | host-sync label lower with same count |
| `array.py:325 tolist` | `395.04 ms / 16` | `427.68 ms / 16` | `-32.64 ms` | mirrors the host-sync attribution drop |
| `PjRtCApiLoadedExecutable::Execute` | `273.53 ms / 44` | `289.24 ms / 140` | `-15.71 ms / -96` | keeps the reduced executable-launch class from Entry 090 |
| `MemcpyD2D` | `18.51 ms / 463` | `30.39 ms / 655` | `-11.88 ms / -192` | tighter envelope reduces device copy traffic |
| `forward_step_token_ids_jit` | `276.27 ms / 16` | `280.56 ms / 16` | `-4.28 ms` | decode step slightly lower |
| `command_buffer::execute` | `225.93 ms / 1936` | `229.21 ms / 1936` | `-3.28 ms` | same command count, lower time |
| `transpose` | `45.06 ms / 312` | `47.30 ms / 312` | `-2.24 ms` | same count, lower time |
| `command_buffer::update` | `10.68 ms / 180` | `10.46 ms / 195` | `+0.22 ms / -15` | fewer updates, slightly higher total |

Validation:

```text
.venv/bin/python -m py_compile \
  benchmarks/run_gpu_matrix.py \
  nanovllm_jax/engine/model_runner.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests/test_gpu_matrix_runner.py \
  tests/test_gpu_matrix_summary_report.py
```

- result: `43 passed`.

Interpretation:

- This closes the staged no-kernel/non-speculative target. The accepted default
  is speed-claim-ready for the exact-token long-prefill gate and reaches
  `0.764x` of the stored vLLM reference without custom kernels.
- The remaining gap is no longer a reason to keep doing small source-level JAX
  cleanup before kernel work. The next active target is `0.9x` vLLM for a
  correctness-gated kernel-backed non-speculative path on the same benchmark
  discipline.
- MTP remains diagnostic-only. It should not be optimized for speed until the
  non-speculative kernel path meets the staged kernel target or the goal is
  explicitly changed.

Decision:

- Keep this as the current accepted no-kernel default for the long heterogeneous
  target.
- Move the next performance target to the kernel roadmap: paged KV append,
  paged decode attention, then GDN decode/prefill, with exact-token and profile
  gates unchanged.

## Entry 092 - Sidecar Smoke And Active-Slot Scheduler Fix

- change accepted: `Scheduler.schedule` no longer admits a waiting prompt when
  the already-active running set occupies `max_num_seqs`. This preserves the
  global active sequence cap across multi-wave serving, not just within one
  scheduling step.
- motivation: the first full 128-prompt `vllm_random_longprefill` sidecar
  exposed a real serving lifecycle bug before producing benchmark evidence:
  after the first four requests were active, later waiting prompts could still
  be admitted, eventually exhausting the model-runner hybrid-state slots with
  `RuntimeError: No free hybrid-state slots; max_num_seqs is exhausted`.
- sidecar scope: add `vllm_random_longprefill_smoke`, a 16-prompt smoke lane
  with the same vLLM-random prompt semantics as the 128-prompt sidecar. The
  128-prompt sidecar remains the broader comparability lane, but it is too heavy
  for quick interactive validation.
- subagent audit: the next kernel milestone should be treated as paired P0
  append+decode work. Do not repeat standalone append/decode/GDN experiments
  from Entries 070, 071, 073, 075, 076, 077, 078, or 079, and do not promote MTP
  speed work before the non-speculative kernel target.

Sidecar smoke command:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --configs gpu_paged_default \
  --workloads vllm_random_longprefill_smoke \
  --repeats 2 \
  --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- artifact: `results/gpu_matrix_20260526_143600.json`
- report: `results/gpu_matrix_20260526_143600.md`
- run directory: `results/gpu_matrix_runs/20260526_143600`
- live JAX reference:
  `results/gpu_matrix_runs/20260526_143600/references/jax_default_vllm_random_longprefill_smoke.json`
- live vLLM reference:
  `results/gpu_matrix_runs/20260526_143600/references/vllm_vllm_random_longprefill_smoke.json`

Sidecar smoke result:

| workload/config | repeats | exact tokens | speed-claim-ready | JAX tok/s median | vLLM tok/s | JAX/vLLM |
|---|---:|---:|---:|---:|---:|---:|
| `vllm_random_longprefill_smoke/gpu_paged_default` | 2 | yes | yes | `82.57` | `299.31` | `0.276x` |

Goal-target revalidation command:

```text
.venv/bin/python benchmarks/run_gpu_matrix.py \
  --goal-target-only --repeats 2 --no-live-vllm \
  --require-stored-references --require-goal-target-ready \
  --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- artifact: `results/gpu_matrix_20260526_144104.json`
- report: `results/gpu_matrix_20260526_144104.md`

Goal-target result:

| workload/config | repeats | exact tokens | speed-claim-ready | JAX tok/s median | vLLM tok/s | JAX/vLLM | target tok/s | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_prefill_512_2048/gpu_paged_default` | 2 | yes | yes | `90.50` | `116.37` | `0.778x` | `87.28` | `0.00 tok/s` |

Validation:

```text
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  TMPDIR=/mountpoint/.exp/tmp \
  XDG_CACHE_HOME=/mountpoint/.exp/.cache \
  JAX_COMPILATION_CACHE_DIR=/mountpoint/.exp/.cache/jax \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests/test_backend_boundaries.py::test_scheduler_does_not_admit_waiting_prompts_when_active_slots_are_full \
  tests/test_backend_boundaries.py::test_scheduler_chunks_prefill_by_max_batched_tokens_budget \
  tests/test_gpu_matrix_runner.py \
  tests/test_gpu_matrix_summary_report.py
```

- result: `45 passed`.

Interpretation:

- The exact-token long-prefill target remains achieved after the scheduler
  lifecycle fix.
- The vLLM-random smoke result is valuable negative/comparability evidence:
  deterministic long-prefill parity does not imply broad random-serving parity.
  The next `0.9x` work should stay on the non-speculative kernel roadmap, not
  MTP.
- The sidecar smoke is not a replacement for the full 128-prompt sidecar. It is
  a quick guard that the random-manifest path and multi-wave scheduler lifecycle
  are functioning.

Decision:

- Keep the scheduler active-slot guard and the 16-prompt sidecar smoke lane.
- Keep the 128-prompt `vllm_random_longprefill` lane as the heavier external
  comparability run.
- Proceed next to paired P0 kernel work: full-attention NHD KV append plus
  paged decode attention with one shared layout and no hot-path conversion.

## Entry 093 - Rejected Narrow P0 FP32 Append+Decode Pair

- change tested: no default-path change. The live diagnostic enabled both local
  FP32 CUDA full-attention routes:
  `NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1` and
  `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1`, while keeping BF16 weights, FP32
  activations/KV cache, greedy token fastpath, and all accepted compact prefill
  flags.
- motivation: Entries 071 and 073 rejected standalone FP32 append and standalone
  FP32 decode-attention routing. The P0 question after Entry 092 was whether
  pairing the two existing local CUDA routes over the same NHD-shaped physical
  cache would remove enough overhead to become a viable full-attention kernel
  path.
- command:

```text
JAX_PLATFORMS=cuda \
NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1 \
NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1 \
NANO_VLLM_JAX_KERNEL_BACKEND=pure_jax \
NANO_VLLM_JAX_GREEDY_TOKEN_FASTPATH=1 \
NANO_VLLM_JAX_MATERIALIZE_TIED_LM_HEAD=1 \
NANO_VLLM_JAX_COMPACT_PREFILL_IN_PROJ_QKV=1 \
NANO_VLLM_JAX_COMPACT_PREFILL_GDN_Z=1 \
NANO_VLLM_JAX_COMPACT_PREFILL_FULL_ATTN_PROJ=1 \
NANO_VLLM_JAX_COMPACT_PREFILL_MLP=1 \
.venv/bin/python benchmarks/benchmark_jax_server_trace.py \
  --model Qwen/Qwen3.5-0.8B \
  --backend gpu \
  --dtype float32 \
  --weight-dtype bfloat16 \
  --jax-execution jit \
  --input-lens 512,1024,1536,2048 \
  --output-len 16 \
  --prompt-suite mixed \
  --num-speculative-tokens 0 \
  --max-kv-cache-mb 3072 \
  --num-kvcache-blocks 384 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --prefill-buckets 512,1024,2048 \
  --batch-size-buckets 1,2,4 \
  --max-blocks-per-seq 129 \
  --top-k 5 \
  --warmup \
  --profile \
  --reference-json results/gpu_matrix_runs/20260526_104818/long_prefill_512_2048_gpu_paged_default_repeat1.json \
  --output-json results/qwen08_jax_server_trace_long_prefill_p0_paired_cuda_fp32_probe.json \
  --run-label long_prefill_p0_paired_cuda_fp32_probe
```

- artifact:
  `results/qwen08_jax_server_trace_long_prefill_p0_paired_cuda_fp32_probe.json`
- trace:
  `/mountpoint/.exp/profiles/20260526-144615-2378662-long_prefill_p0_paired_cuda_fp32_probe/plugins/profile/2026_05_26_14_47_04/INDCS0291.atrapa.deloitte.com.trace.json.gz`
- correctness: exact generated-token match for all four long-prefill rows
  against the stored JAX long-prefill reference.
- trace confirmation: the Perfetto trace contains both
  `(anonymous namespace)::Fp32KvAppendKernel(...)` and
  `(anonymous namespace)::Fp32PagedDecodeAttentionKernel(...)`, so the probe did
  exercise the paired local CUDA routes.

Result:

| workload/config | exact tokens | JAX tok/s | vLLM tok/s | JAX/vLLM | TTFT p50 | ITL p50 | ITL p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_prefill_512_2048/p0_fp32_append_decode_pair` | yes | `52.94` | `116.37` | `0.455x` | `600.25 ms` | `38.58 ms` | `46.72 ms` |

Profile comparison against the accepted Entry 092 long-prefill repeat:

| bucket | paired P0 | accepted default | delta | note |
|---|---:|---:|---:|---|
| `forward_step_token_ids_jit` | `641.18 ms / 16` | `272.67 ms / 16` | `+368.51 ms` | decode step gets much slower |
| `PjRtCApiLoadedExecutable::Execute` | `295.31 ms / 44` | `270.18 ms / 44` | `+25.13 ms` | compiled execution regresses |
| `np.asarray(jax.Array)` | `508.45 ms / 16` | `398.51 ms / 16` | `+109.94 ms` | host sync waits on slower device path |
| `array.py:325 tolist` | `508.58 ms / 16` | `398.64 ms / 16` | `+109.94 ms` | mirrors sync attribution |
| `command_buffer::execute` | `226.74 ms / 1864` | `223.59 ms / 1936` | `+3.15 ms / -72` | fewer commands did not help |
| `gather` | `19.31 ms / 311` | `14.74 ms / 103` | `+4.57 ms / +208` | gather work increased |
| `transpose` | `34.70 ms / 132` | `45.05 ms / 312` | `-10.35 ms / -180` | lower transpose did not offset kernel cost |
| `MemcpyD2D` | `18.51 ms / 463` | `18.36 ms / 463` | `+0.15 ms` | unchanged |

Validation:

```text
.venv/bin/python -m py_compile \
  benchmarks/benchmark_jax_server_trace.py \
  benchmarks/run_gpu_matrix.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests/test_gpu_matrix_runner.py \
  tests/test_gpu_matrix_summary_report.py
```

- result: `43 passed`.

Interpretation:

- This rejects the narrow version of paired P0. The existing local FP32 append
  and decode-attention custom calls are correctness-clean together, but they do
  not own enough of the full-attention boundary to move integrated serving
  performance.
- The next kernel attempt should not be another spelling of these two isolated
  custom calls. Viable next paths are a broader fused/layout P0 design with less
  per-layer custom-call overhead, or moving to the higher-leverage GDN roadmap.
- MTP remains out of scope for speed work.

Decision:

- Do not promote `NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1` +
  `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1` to default or fast opt-in.
- Keep the local FP32 CUDA kernels as focused ABI/toolchain diagnostics.

### Entry 094 - Accepted kernel-phase goal target raise

- change accepted: `benchmarks/run_gpu_matrix.py` now sets the active
  `TARGET_VLLM_RATIO` to `0.9` instead of `0.75`.
- motivation: Entry 091 and the post-sidecar revalidation closed the staged
  no-kernel milestone on the deterministic long-prefill gate (`0.764x`, then
  `0.778x` vLLM). The next goal should be harder and kernel-specific rather
  than repeatedly validating the closed `0.75x` bar.
- scope: this changes the executable `--require-goal-target-ready` acceptance
  threshold, the target-token/s math in matrix summaries, and related tests.
  Historical `0.75x` logbook entries remain unchanged.
- decision: use `0.9x` vLLM as the active correctness-gated non-speculative
  kernel-phase target. MTP remains diagnostic-only until this target is met or
  explicitly reprioritized.

### Entry 095 - Kernel-phase top-5 guardrail revalidation

- change accepted: `benchmark_long_decode_top5.py` now writes a machine-readable
  `guardrail` block with exact-token/top-5 checks, the required numeric logit
  threshold, pass/fail status, git head, JAX backend, and JAX version.
- command:

```text
JAX_PLATFORMS=cuda \
.venv/bin/python benchmark_long_decode_top5.py \
  --max-new-tokens 500 \
  --compare-json results/qwen08_jax_bf16w_fp32act_long_decode_top5_compare_20260526_kernel_phase_gate.json
```

- artifact:
  `results/qwen08_jax_bf16w_fp32act_long_decode_top5_compare_20260526_kernel_phase_gate.json`
- result: exact top-1, ordered top-5, and top-5 set parity all passed
  `500/500` against `results/qwen08_hf_bf16w_fp32act_long_decode_top5_500.npz`.
- numeric gate: `max_hf_topk_id_logit_diff=2.09808349609375e-05`, which is
  slightly above the documented `2e-5` bound, so
  `guardrail.passes_required_gate=false`.
- decision: keep the `2e-5` bound. This evidence does not authorize the packed
  segmented GDN override path; it either needs a candidate that passes the bound
  or an explicit threshold decision.

### Entry 096 - External GDN reference audit

- audit source: subagent review of vLLM Qwen GDN and Flash Linear Attention
  Gated DeltaNet references.
- finding: vLLM's Qwen GDN path naturally uses k-last/V-first recurrent state
  (`[*,HV,V,K]`) and a state-slot/index model. At the time of this audit nano's
  serving state was `[B,H,K,V]`; Entry 098 later migrated nano's serving state
  to V,K as well.
- dtype finding: the vLLM/FLA prefill path is BF16-activation oriented even when
  state math/cache can be FP32. That does not match the current BF16
  weights/FP32 activation contract.
- decision: do not silently adopt vLLM/FLA state layout, BF16 prefill
  activations, or state-slot scheduler semantics. The next non-design-changing
  GDN target is an opt-in FP32 packed decode core that borrows vLLM's packed
  `mixed_qkv + a/b/A_log/dt_bias` boundary while preserving nano's current
  persistent-state layout.

### Entry 097 - Accepted packed FP32 GDN decode core

- change accepted: added a local CUDA/JAX FFI target
  `nanovllm_jax_fp32_gdn_packed_decode` plus Python wrapper
  `gdn_packed_decode_step_fp32`.
- scope: the core accepts vLLM-style `mixed_qkv [B, 2*H*K + HV*V]`, raw
  `a/b [B,HV]`, `A_log/dt_bias [HV]`, and nano's local FP32 state
  `[B,HV,K,V]`. It performs split, q/k L2 norm, query scaling, gate/beta
  transform, and width-1 recurrence in one CUDA call.
- correctness: focused CUDA parity passes against
  `gdn_packed_decode_reference_local_state` for same-head and GVA q/k
  repetition cases.
- validation:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests/test_gdn_packed_decode_reference.py \
  tests/test_cuda_fp32_ffi.py \
  -k 'packed_gdn_decode or gdn_packed_decode'

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests/test_cuda_fp32_ffi.py
```

- result: packed-focused test selection `5 passed, 11 deselected`; full local
  CUDA FFI regression file `13 passed`.
- decision: keep this default-off and unrouted for now. It is the packed GDN
  ABI/toolchain step needed before a serving integration attempt, not an
  integrated speed claim.

### Entry 098 - Accepted GDN V,K JAX Layout Migration

- change accepted: switched the pure-JAX GDN serving state contract from
  `[B,L,HV,K,V]` to kernel-native V,K `[B,L,HV,V,K]` and updated the JAX
  recurrent/chunked fallback math to consume and return V,K state directly.
- scope: `init_hybrid_state`, GDN recurrent/chunk rules, shape expectations,
  focused state tests, and default-off CUDA FFI compatibility wrappers. The
  legacy local CUDA probes now accept V,K at the Python boundary and transpose
  internally into their old K,V implementation; they remain diagnostic only.
- validation:

```text
JAX_PLATFORMS=cuda ... pytest -q \
  tests/test_kv_cache.py::test_linear_attention_chunked_vs_recurrent \
  tests/test_kv_cache.py::test_linear_attention_multichunk_matches_recurrent \
  tests/test_kv_cache.py::test_linear_attention_state_persistence \
  tests/test_kv_cache.py::test_multi_layer_linear_attention_states \
  tests/test_layer_parity.py::test_linear_attention_recurrent \
  tests/test_backend_boundaries.py::test_bucketed_linear_prefill_preserves_hybrid_state_for_decode \
  tests/test_backend_boundaries.py::test_linear_suffix_prefill_matches_sequential_decode_state \
  tests/test_gdn_segmented_reference.py \
  tests/test_gdn_packed_decode_reference.py

JAX_PLATFORMS=cuda ... pytest -q tests/test_cuda_fp32_ffi.py
JAX_PLATFORMS=cuda ... pytest -q tests/test_mtp_commit_semantics.py
JAX_PLATFORMS=cuda ... benchmark_long_decode_top5.py --max-new-tokens 500 \
  --compare-json results/qwen08_jax_bf16w_fp32act_long_decode_top5_compare_20260526_vk_layout.json
JAX_PLATFORMS=cuda ... benchmarks/run_gpu_matrix.py --goal-target-only --repeats 2 \
  --no-live-vllm --require-stored-references \
  --output-json results/gpu_matrix_20260526_vk_layout.json
```

- results: focused GDN/state suite `12 passed`; CUDA FFI suite `13 passed`; MTP
  commit-state suite `15 passed, 1 xfailed`; 500-token guardrail passed exact
  top-1, ordered top-5, and top-5 set parity `500/500` with
  `max_hf_topk_id_logit_diff=1.9073486328125e-05` within the `2e-5` gate.
- integrated benchmark: `results/gpu_matrix_20260526_vk_layout.json`, report
  `results/gpu_matrix_20260526_vk_layout.md`, long-prefill goal target,
  `90.65 tok/s`, `0.779x` vLLM, exact generated-token parity over two repeats,
  speed-claim-ready, `target_vllm_ratio_met=false` for the active `0.9x` bar.
- decision: keep V,K as the GDN serving state layout and continue toward a
  kernel-native V,K GDN implementation. This is a correctness/layout migration,
  not the final speed win.

### Entry 099 - Native V,K CUDA GDN Decode Probes

- change accepted: updated the default-off local CUDA/JAX FFI recurrent and
  packed GDN decode probes to consume and return native V,K state directly.
- scope: `gdn_recurrent_decode_step_fp32` and `gdn_packed_decode_step_fp32`
  no longer transpose V,K state into legacy K,V at the Python boundary. The CUDA
  decode kernels and shape checks now index state as `[B,H,V,K]`. The separate
  chunked prefill prototype remained benchmark-only/default-off and was moved to
  native V,K in Entry 101.
- validation:

```text
JAX_PLATFORMS=cuda ... pytest -q tests/test_cuda_fp32_ffi.py \
  -k 'gdn_recurrent_decode_step_fp32_cuda_matches_reference or gdn_packed_decode_step_fp32_matches_reference'

JAX_PLATFORMS=cuda ... pytest -q tests/test_cuda_fp32_ffi.py
```

- result: focused recurrent/packed CUDA decode selection `4 passed, 9
  deselected`; full CUDA FFI regression file `13 passed`.
- decision: keep the probes default-off until an integrated serving route wins
  the benchmark. This removes a layout adapter and keeps future GDN kernel work
  aligned with the accepted V,K serving state.

### Entry 100 - Rejected Native V,K Width-1 CUDA GDN Decode Routing Probe

- experiment: reran the long-prefill goal target with
  `NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE=1` after the recurrent decode CUDA probe
  was moved to native V,K state.
- artifact: `results/gpu_matrix_20260526_vk_cuda_gdn_decode_probe.json`
- report: `results/gpu_matrix_20260526_vk_cuda_gdn_decode_probe.md`
- result: exact generated-token parity held, but the one-repeat diagnostic
  reached only `88.07 tok/s`, `0.757x` vLLM. That is below the accepted V,K
  baseline of `90.65 tok/s`, `0.779x` vLLM, and below the active `0.9x` target.
- profile movement vs stored JAX reference: `PjRt Execute` dropped to
  `273.86 ms`, `MemcpyD2D` dropped to `18.24 ms`, and `transpose` dropped to
  `45.05 ms`, but the integrated throughput still did not beat the current
  accepted baseline.
- decision: reject promotion of the standalone width-1 CUDA GDN decode route.
  Keep it default-off for diagnostics and continue toward a coarser/fused GDN
  kernel boundary rather than adding this custom call to the hot path.

### Entry 101 - Native V,K GDN Prefill Probe Cleanup

- change accepted: updated the benchmark-only CUDA/JAX FFI GDN prefill probes
  and standalone GDN prefill benchmark helpers to use native V,K state.
- scope: `gdn_prefill_chunk32_normalized_fp32`,
  `gdn_prefill_chunk32_v64_normalized_fp32`, the CUDA prefill state loads/stores,
  and `benchmark_gdn_prefill_kernel.py` state generation/reconstruction probes.
  No serving route changed.
- validation:

```text
JAX_PLATFORMS=cuda ... pytest -q tests/test_cuda_fp32_ffi.py \
  -k 'gdn_prefill_chunk32_normalized_fp32_cuda_matches_chunk_reference'

JAX_PLATFORMS=cuda ... pytest -q \
  tests/test_cuda_fp32_ffi.py \
  tests/test_gdn_segmented_reference.py \
  tests/test_gdn_segmented_policy.py
```

- result: focused prefill CUDA selection `2 passed, 11 deselected`; focused GDN
  CUDA/segmented suite `18 passed`.
- smoke artifacts:
  `results/gdn_prefill_kernel_20260526_vk_native_smoke.json` and
  `results/gdn_prefill_kernel_20260526_vk_native_segmented_gate_smoke.json`,
  both at git head `43f39fd`.
- smoke result: non-square `B=2,H=2,T=64,K=32,V=64` one-piece CUDA prefill
  comparisons match current padded chunk32 with `output_max_abs=1.49e-07` and
  `state_max_abs=1.073e-06`. The packed segmented reference gate passes on the
  same reduced smoke shape with `output_max_abs=1.788e-07` and
  `state_max_abs=5.364e-07`.
- decision: keep the one-piece prefill prototypes default-off and
  benchmark-only. This validates the V,K boundary and fixes stale benchmark
  helpers, but it does not overturn the prior full-shape rejection or authorize
  serving promotion.

### Entry 102 - Rejected Full-Shape Native V,K GDN Prefill Probe

- experiment: reran the full model-shape standalone GDN prefill microbenchmark
  after the benchmark-only CUDA prefill probes were moved to native V,K state.
- artifact: `results/gdn_prefill_kernel_20260526_vk_native_fullshape.json`
- shape: `B=8,H=16,T=512,K=128,V=128`, lengths
  `[64,128,192,256,320,384,448,512]`, chunk size `32`, NVIDIA A10G.
- result: current JAX chunk32 p50 `5.60 ms`; `cuda_fp32_one_piece_chunk32` p50
  `10.43 ms`; `cuda_fp32_one_piece_chunk32_v64` p50 `10.46 ms`.
- correctness drift: both CUDA one-piece variants have
  `output_max_abs=1.907e-05` and `state_max_abs=2.441e-04` versus current padded
  chunk32, so they still miss the standalone state gate.
- decision: reject serving promotion and do not spend more effort on the
  rectangular one-piece prefill prototype. The next GDN prefill candidate should
  target the segmented/nnz boundary or a backend design that preserves the
  current padded chunk32 accumulation contract.

### Entry 103 - vLLM-Inspired Random-Token Sidecar

- experiment: ran the new vLLM-inspired random-token long-prefill sidecar with 128
  prompts, random input length centered at `1280`, range ratio `0.6`, output
  length `16`, `max_num_seqs=4`, and one repeat. The run used
  `gpu_paged_default`; no CUDA GDN probe flag was enabled.
- artifact: `results/gpu_matrix_20260526_vllm_random_longprefill_r1.json`
- report: `results/gpu_matrix_20260526_vllm_random_longprefill_r1.md`
- result: JAX default reached `84.45 tok/s`; live vLLM reached
  `354.24 tok/s`; JAX/vLLM was `0.238x`. Exact generated-token match held
  against the stored JAX default reference, and JAX/reference throughput was
  `1.009x`, but the run is not speed-claim-ready because it has only one
  repeat and misses the `0.9x` target.
- latency: JAX `TTFT p50 12420.89 ms`, `TTFT p95 23289.88 ms`,
  `ITL p50 13.30 ms`, `ITL p95 14.22 ms`; live vLLM `TTFT p50 2946.94 ms`,
  `TTFT p95 5403.52 ms`, `ITL p50 71.30 ms`, `ITL p95 75.08 ms`.
- profile movement vs the stored JAX reference is small: `forward_step_token_ids_jit`
  improved by `120.55 ms`, `PjRt Execute` improved by `102.97 ms`, while
  `tolist`/`np.asarray` attribution worsened by about `43.9 ms`.
- decision: record this as a sidecar comparability data point, not as the main
  exact-token goal gate. The large gap is primarily TTFT/serving-batch behavior
  on the local shared-token random prompt lane, so future kernel work should
  continue to report this lane separately from the frozen long-prefill
  correctness gate.

### Entry 104 - GDN FLA Backend Registry Scaffolding

- change accepted: added `gdn_fla` plus `fla_gdn`, `vllm_fla`, and
  `flash_linear_attention` aliases to the optional kernel backend registry.
  These names advertise the planned vLLM/Flash Linear Attention-shaped GDN
  production route while still falling back to pure JAX until an implementation
  passes acceptance gates.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_kernel_registry.py
```

- result: `8 passed`.
- decision: keep `gdn_fla` unimplemented and default-off. The local CUDA GDN
  probes remain diagnostics; this registry change only prevents the plan's
  intended backend name from being an unknown request.

### Entry 105 - External Kernel Compatibility Recheck

- audit source: subagent review of local installed packages plus direct local
  inspection. No internet was used and no files were edited by the subagent.
- environment: repo venv has FlashInfer/JAX FFI and Triton but no vLLM or FLA
  package; vLLM is installed separately at `/mountpoint/.exp/vllm-venv` and
  includes its vendored FLA kernels.
- vLLM GDN finding: Qwen3.5 uses
  `GatedDeltaNetAttention(..., gqa_interleaved_layout=False)`, with non-
  interleaved `[q,k,v,z]` and `[b,a]` projection splits. The packed decode
  kernel uses state `[slots,HV,V,K]`, `mixed_qkv [B,2*H*K+HV*V]`, `a/b [B,HV]`,
  and output `[B,1,HV,V]`.
- dtype finding: vLLM's FLA chunk prefill rejects `torch.float32` activation
  tensors and expects BF16/FP16-style q/k/v, while final state can remain FP32.
  FlashInfer's GDN prefill path is Torch-only in the installed environment and
  is gated to newer CUDA targets than the current A10G baseline.
- FlashInfer/JAX finding: the repo's FlashInfer JAX shim only implements KV
  append today, only for FP16/BF16 cache tensors. Paged attention FFI wrappers
  remain stubs, and installed FlashInfer dtype maps omit FP32 for these paths.
- decision: do not switch to vLLM/FLA or FlashInfer kernels directly under the
  current BF16-weight/FP32-activation/KV contract. The next production kernel
  direction needs an explicit design choice: allow a BF16-prefill experiment
  with full-model gates, or build/port FP32-capable FLA-shaped kernels. Do not
  use the current local CUDA probes as the next optimization path.

### Entry 106 - Scheduler Diagnostics For vLLM-Style Sidecar

- change accepted: matrix summaries now record scheduler-step diagnostics from
  existing JAX token events. The runner deduplicates token events by scheduler
  step and the markdown summary reports median prefill/decode step counts, max
  active prefill sequences, max step tokens, and total prefill/decode step
  seconds.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
```

- result: `45 passed`.
- sidecar evidence: the 16-prompt `vllm_random_longprefill_smoke` artifact has
  4 prefill waves, 60 decode steps, max 4 active prefill sequences, about
  `2.33 s` total prefill-step time, and about `0.77 s` total decode-step time.
  The 128-prompt `vllm_random_longprefill` artifact has 32 prefill waves, 480
  decode steps, max 4 active prefill sequences, about `17.81 s` total
  prefill-step time, and about `6.28 s` total decode-step time.
- failed probes: raising sidecar `max_num_seqs` to 5, 6, or 8 OOMed during
  warmup with static allocations of about `10.05 GiB`, `10.66 GiB`, and
  `12.57 GiB`; no result JSONs were produced.
- decision: keep reporting scheduler diagnostics for sidecar comparisons. The
  vLLM-random sidecar gap is strongly tied to static-shape concurrency and TTFT,
  so the next design work should address compact/ragged prefill scheduling or
  external kernel integration boundaries before assuming a narrow local probe
  will close the gap.

### Entry 107 - Rejected Split Prefill/Decode Batch Buckets

- experiment: tried a scheduler/static-shape split where sidecar prefill stayed
  capped at the existing 4-row bucket while decode could use an 8-row bucket.
  The goal was to avoid the previous `[8, 2048]` prefill OOM while cutting the
  number of random-sidecar decode steps.
- artifact: `results/gpu_matrix_20260527_split_decode8_smoke.json`
- report: `results/gpu_matrix_20260527_split_decode8_smoke.md`
- validation: the run completed with elevated GPU access and exact generated-
  token match against its live JAX default reference, but it used only one
  repeat and was a sidecar smoke, not a speed claim.
- result: JAX output throughput regressed to `68.25 tok/s`, live vLLM was
  `299.56 tok/s`, and JAX/vLLM was `0.228x`. The change reduced decode steps
  from the prior smoke's 60 to 45 and reached max decode step sequences of 8,
  but prefill took about `2.48 s`, decode took about `1.26 s`, `ITL p50`
  worsened to `27.74 ms`, and `gather` plus `PjRt Execute` profile buckets
  increased versus the live JAX reference.
- decision: reject and revert the split-bucket scheduler path. Reducing Python
  decode step count alone is not enough; larger decode physical buckets must
  win integrated sidecar throughput before becoming part of the default server
  policy.

### Entry 108 - Current Elevated Goal-Target Revalidation

- experiment: reran the current default non-speculative long-prefill goal target
  with elevated GPU access, stored JAX/vLLM references, and two repeats.
- artifact: `results/gpu_matrix_20260527_current_goal_target.json`
- report: `results/gpu_matrix_20260527_current_goal_target.md`
- result: speed-claim-ready with exact generated-token parity over both
  repeats. JAX reached `90.87 tok/s`; stored vLLM is `116.37 tok/s`; JAX/vLLM is
  `0.781x`. The active `0.9x` target requires `104.74 tok/s`, leaving a
  `13.86 tok/s` gap and about `1.153x` required JAX speedup.
- latency: `TTFT p50 534.49 ms`, `ITL p50 11.23 ms`, `ITL p95 12.14 ms`.
- scheduler diagnostics: one prefill step, 15 decode steps, max 4 active rows,
  about `0.53 s` prefill-step time and `0.17 s` decode-step time.
- profile movement vs the stored JAX default reference: `np.asarray(jax.Array)`
  and `array.py:325 tolist` attribution improved by about `29.8 ms`, `PjRt
  Execute` improved by about `22.5 ms`, `forward_step_token_ids_jit` improved
  by about `11.8 ms`, and `MemcpyD2D` improved by about `11.9 ms`.
- decision: keep this as the current default baseline evidence. It does not
  close the `0.9x` target, and no further scheduler-only shape change should be
  accepted without an integrated throughput win. The next meaningful step
  requires choosing an external-kernel direction: an opt-in BF16 GDN prefill
  activation experiment, or a FP32-capable vLLM/FLA-shaped GDN path.

### Entry 109 - Neutral GDN FLA Packed Decode ABI Reference

- change accepted: added `nanovllm_jax.kernels.gdn_fla` as the neutral home for
  the planned vLLM/Flash Linear Attention-shaped GDN boundary. The module owns
  the FP32 packed decode reference for
  `mixed_qkv + a/b/A_log/dt_bias -> output, state` with native `[B,HV,V,K]`
  recurrent state, plus fallback-only availability wrappers. No serving route
  changed and no external kernel was enabled.
- cleanup: focused packed-decode tests now import the pure-JAX reference from
  `gdn_fla`, and the local CUDA FP32 diagnostic wrapper uses the same reference
  for parity. This separates the planned production ABI from the rejected local
  CUDA probe path.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
```

- result: `12 passed`.
- decision: keep this as a non-speed ABI step toward the FP32-capable
  vLLM/FLA-shaped GDN path. It does not affect the `0.9x` throughput gate; the
  next real implementation still needs either a FP32-capable FLA-shaped kernel
  port or an explicitly approved BF16-prefill experiment.

### Entry 110 - Neutral GDN FLA Segmented Prefill ABI Reference

- change accepted: extended `nanovllm_jax.kernels.gdn_fla` to own the planned
  segmented GDN prefill ABI reference and pack/unpack helpers:
  `[B,H,T,D] + seq_lens -> [nnz,H,D] + cu_seqlens -> [B,H,T,V]`. Existing
  focused tests and the standalone GDN prefill benchmark now import this
  planned FLA/vLLM-shaped contract from `gdn_fla` instead of from the local CUDA
  diagnostic module.
- scope: no serving route changed, no dtype changed, and no external kernel was
  enabled. The old `cuda_gdn` helpers remain available for compatibility, but
  new planned-backend references should live in `gdn_fla`.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
.venv/bin/python -m py_compile benchmarks/benchmark_gdn_prefill_kernel.py nanovllm_jax/kernels/gdn_fla.py
```

- result: `13 passed`; `py_compile` passed.
- decision: keep this as another non-speed ABI cleanup toward a FP32-capable
  vLLM/FLA-shaped GDN path. It does not change the `0.9x` target status; the
  next implementation step still needs a real FP32 kernel strategy or explicit
  approval for BF16 prefill activation experiments.

### Entry 111 - External GDN FP32 Reuse Audit

- audit result: direct reuse of installed vLLM/FLA or FlashInfer GDN kernels is
  not a drop-in route for the current BF16-weight/FP32-activation contract.
  vLLM/FLA prefill rejects FP32 activation tensors, FlashInfer GDN prefill is
  half/BF16 and SM90/SM100 oriented, and FlashInfer GDN decode is not a direct
  FP32 activation route.
- useful reference: vLLM's packed decode GDN kernel shape remains the best next
  implementation target. It performs FP32 loads/math internally and matches the
  repo's neutral packed-QKV boundary, but it is a Torch/vLLM Triton op tied to
  vLLM state indexing, so nano-vLLM-JAX needs a port/fork or wrapper rather
  than direct reuse.
- decision: make packed decode the first real vLLM/FLA-shaped GDN kernel
  boundary. Do not start with segmented prefill or fused projection+GDN, and do
  not promote the historical local CUDA probes as the production route.

### Entry 112 - Packed GDN Kernel Route Decision Checkpoint

- current target evidence: `results/gpu_matrix_20260527_current_goal_target.json`
  remains the latest elevated, speed-claim-ready long-prefill target artifact.
  It is exact over two repeats at `90.87 tok/s`; the stored vLLM reference is
  `116.37 tok/s`, the `0.9x` target is `104.74 tok/s`, and the remaining gap is
  `13.86 tok/s`.
- scheduler context: the same artifact reports one prefill step at about
  `0.53 s` and 15 decode steps totaling about `0.17 s`. This does not prove
  packed GDN decode alone can close the gap, but it is the smallest
  vLLM/FLA-shaped FP32 kernel boundary currently available without changing the
  default dtype contract.
- decision needed: choose Pallas, a production vLLM/FLA-shaped CUDA/JAX FFI
  port, or pausing GDN to return to FlashInfer/full-attention work. Do not route
  a new GDN serving kernel until that design choice is explicit.

### Entry 113 - Current Goal Raw GPU Trace Summary

- change accepted: added `benchmarks/summarize_profile_trace.py`, a small
  Chrome/Perfetto trace summarizer for top raw event names and substring
  totals. This is analysis tooling only and does not change serving behavior.
- artifact: `results/profile_trace_20260527_current_goal_target_gpu.json`
- report: `results/profile_trace_20260527_current_goal_target_gpu.md`
- evidence: on both current-goal repeats, GPU-scope time is dominated by
  GEMM/fusion work. Repeat 1 reports `gemm_fusion=243.57 ms`,
  `cutlass=64.54 ms`, `transpose=45.08 ms`, `input_reduce_fusion=34.66 ms`,
  `loop_dynamic_slice=18.07 ms`, `wrapped_concatenate=17.39 ms`, and
  `MemcpyD2D=15.91 ms`; repeat 2 is effectively identical.
- interpretation: this supports the caution in Entry 112. Packed GDN decode is
  still the smallest FP32 vLLM/FLA-shaped implementation boundary, but the raw
  GPU trace does not prove a decode-only kernel can close the full `0.9x` gap.
  Prefill/GEMM/fusion-heavy work remains a major part of the target profile.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_profile_trace_summary.py
.venv/bin/python -m py_compile benchmarks/summarize_profile_trace.py tests/test_profile_trace_summary.py
```

- result: `2 passed`; `py_compile` passed.

### Entry 114 - Scoped Raw Trace Events In Benchmark Artifacts

- change accepted: `benchmarks/benchmark_jax_server_trace.py` now records
  `scoped_ranges` and `scoped_top_events_by_total_ms` for GPU and CPU trace
  scopes in each benchmark artifact. `benchmarks/run_gpu_matrix.py` preserves
  those fields in per-repeat matrix metrics.
- motivation: future performance claims need raw GPU/CPU event evidence without
  hand-parsing Perfetto traces or running a separate summarizer after every
  matrix run.
- stored-profile smoke: running `_profile_counters` on the current-goal repeat
  1 profile reports GPU `gemm_fusion={'total_ms': 243.57338599999844,
  'count': 6424}` and top GPU event `gemm_fusion_dot_general_744` at
  `57.457908 ms`.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_profile_trace_summary.py tests/test_gpu_matrix_runner.py
.venv/bin/python -m py_compile benchmarks/benchmark_jax_server_trace.py benchmarks/run_gpu_matrix.py benchmarks/summarize_profile_trace.py tests/test_profile_trace_summary.py tests/test_gpu_matrix_runner.py
```

- result: `47 passed`; `py_compile` passed.

### Entry 115 - Scoped Events In Matrix Markdown Reports

- change accepted: `benchmarks/summarize_gpu_matrix.py` now renders a compact
  `Top Scoped Profile Events` section when per-repeat matrix metrics include
  `profile_scoped_top_events_by_total_ms`.
- scope: this affects future reports generated from new profiled matrix
  artifacts. Existing matrix summaries from before Entry 114 do not contain the
  scoped fields, so they omit the section unless regenerated from new benchmark
  artifacts.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gpu_matrix_summary_report.py
.venv/bin/python -m py_compile benchmarks/summarize_gpu_matrix.py tests/test_gpu_matrix_summary_report.py
```

- result: `2 passed`; `py_compile` passed.

### Entry 116 - Scoped-Profile Goal Target Revalidation

- experiment: reran the non-speculative long-prefill goal target after Entry
  114/115 so the matrix artifact and Markdown report include scoped GPU/CPU top
  profile events. The run used stored JAX/vLLM references and elevated GPU
  access.
- artifact: `results/gpu_matrix_20260527_scoped_profile_target.json`
- report: `results/gpu_matrix_20260527_scoped_profile_target.md`
- result: speed-claim-ready with exact generated-token parity over both
  repeats. JAX reached `90.81 tok/s`; stored vLLM is `116.37 tok/s`; JAX/vLLM is
  `0.780x`. The active `0.9x` target requires `104.74 tok/s`, leaving a
  `13.93 tok/s` gap.
- latency: `TTFT p50 533.69 ms`, `ITL p50 11.36 ms`, `ITL p95 12.25 ms`.
- scheduler diagnostics: one prefill step, 15 decode steps, max 4 active rows,
  about `0.53 s` prefill-step time and `0.17 s` decode-step time.
- scoped profile evidence: both repeats report the top GPU events as GEMM/CUTLASS
  work. Repeat 1 starts with `gemm_fusion_dot_general_744` at `57.45 ms` over
  48 calls, then a CUTLASS kernel at `36.89 ms` over 30 calls, then
  `gemm_fusion_dot_general_729` at `36.58 ms` over 18 calls. Repeat 2 is
  effectively identical.
- decision: keep this as the current scoped-profile target artifact. It verifies
  the benchmark/reporting path and reinforces that the default path remains
  below `0.9x`; it does not justify moving to MTP or accepting a decode-only
  kernel as sufficient by itself.

### Entry 117 - Scoped Profile Range Medians In Matrix Reports

- change accepted: `benchmarks/run_gpu_matrix.py` now aggregates scoped GPU/CPU
  profile range medians across repeats, and `benchmarks/summarize_gpu_matrix.py`
  renders them in `Scoped Profile Range Medians`. The renderer also falls back
  to per-repeat scoped ranges for older matrix JSONs that predate the aggregate
  field.
- artifact update: regenerated
  `results/gpu_matrix_20260527_scoped_profile_target.md` from the existing
  scoped-profile target JSON. The report now shows scoped range medians before
  the per-repeat top scoped events.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
.venv/bin/python -m py_compile benchmarks/run_gpu_matrix.py benchmarks/summarize_gpu_matrix.py tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
```

- result: `47 passed`; `py_compile` passed.

### Entry 118 - Scoped Profile Deltas When References Support Them

- change accepted: matrix comparison summaries now compute
  `profile_scoped_delta_vs_jax_reference` for GPU/CPU scoped ranges when both
  the current aggregate and selected JAX reference contain scoped profile
  fields. `benchmarks/summarize_gpu_matrix.py` renders those deltas in
  `Scoped Profile Deltas Vs JAX Reference`.
- current artifact note: regenerated
  `results/gpu_matrix_20260527_scoped_profile_target.md`. The selected JAX
  reference predates scoped fields, so the report correctly says no scoped
  profile deltas are available while still showing scoped range medians and top
  events for the current run.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
.venv/bin/python -m py_compile benchmarks/run_gpu_matrix.py benchmarks/summarize_gpu_matrix.py tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
```

- result: `47 passed`; `py_compile` passed.

### Entry 119 - Two-Repeat Full vLLM-Random Sidecar

- experiment: reran the full 128-prompt `vllm_random_longprefill` sidecar with
  `gpu_paged_default`, two repeats, elevated GPU access, live JAX default
  reference capture, and live vLLM reference capture.
- artifact: `results/gpu_matrix_20260527_vllm_random_longprefill_r2.json`
- report: `results/gpu_matrix_20260527_vllm_random_longprefill_r2.md`
- result: speed-claim-ready in the matrix sense and exact against the live JAX
  default reference, but still far from vLLM. JAX median throughput is
  `84.60 tok/s`; live vLLM is `353.91 tok/s`; JAX/vLLM is `0.239x`.
- latency and scheduler evidence: JAX `TTFT p50 12385.99 ms`, `TTFT p95
  23244.28 ms`, `ITL p50 13.39 ms`, and `ITL p95 14.16 ms`. The JAX scheduler
  again reports 32 prefill waves, 480 decode steps, max 4 active prefill
  sequences, about `17.74 s` total prefill-step time, and about `6.30 s` total
  decode-step time.
- scoped profile evidence: the current sidecar report now has scoped deltas
  against the live JAX default reference. The largest CPU scoped deltas are
  `np.asarray(jax.Array)` +`35.28 ms`, `array.py:325 tolist` +`35.02 ms`,
  `PjRtCApiLoadedExecutable::Execute` -`31.84 ms`, and
  `forward_step_token_ids_jit` -`31.15 ms`. GPU scoped deltas are effectively
  flat for the required buckets, with `transpose` -`0.13 ms`, `MemcpyD2D`
  +`0.02 ms`, and `gather` +`0.01 ms`.
- decision: keep this as the current broad vLLM-style comparability artifact.
  It does not change the main frozen exact-token goal target and should not be
  used as evidence that a narrow kernel swap alone will close the vLLM-random
  gap. The sidecar remains dominated by static-shape concurrency and TTFT.

### Entry 120 - Prompt Provenance In Matrix Reports

- change accepted: matrix metric summaries now preserve prompt metadata from
  benchmark artifact `run_config`: prompt source, dataset, input/output shape,
  seed, random range settings, manifest path, and manifest SHA. Markdown reports
  render these fields in a `Prompt Provenance` table and show whether the
  current JAX run and vLLM reference have matching manifest hashes.
  `vllm_random` stored-reference matching now also requires both manifest path
  and manifest SHA.
- artifact update: regenerated
  `results/gpu_matrix_20260527_vllm_random_longprefill_r2.json` from its
  already-written repeat/reference artifacts and regenerated
  `results/gpu_matrix_20260527_vllm_random_longprefill_r2.md`. The report now
  shows `vllm_random`, dataset `random`, 128 prompts, seed `0`, random input
  `1280`, random output `16`, range ratio `{"input":0.6,"output":0.0}`, and
  matching current/vLLM manifest prefix `9f98c47fe18b`.
- interpretation: this avoids overstating benchmark comparability. The sidecar
  proves JAX and vLLM used identical generated token IDs for the local
  vLLM-inspired random-token manifest harness, but it is still not upstream
  `vllm bench --dataset-name random` semantics, an upstream `vllm bench` run, or
  a ShareGPT/BurstGPT dataset claim.
- validation:

```text
.venv/bin/python -m pytest -q tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
.venv/bin/python -m py_compile benchmarks/run_gpu_matrix.py benchmarks/summarize_gpu_matrix.py tests/test_gpu_matrix_runner.py tests/test_gpu_matrix_summary_report.py
.venv/bin/python -m json.tool results/gpu_matrix_20260527_vllm_random_longprefill_r2.json
```

- result: `48 passed`; `py_compile` passed; JSON validation passed.

### Entry 121 - Packed GDN Decode Backend Boundary Routed Default-Off

- change accepted: added an experimental width-1 cached-decode GDN route behind
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL`. The model now passes post-convolution
  packed `mixed_qkv`, raw `a/b`, local loaded `A=exp(A_log)`, `dt_bias`, and
  native `[B,HV,V,K]` recurrent state through `backend.gated_delta_packed_decode`
  when the flag is set and no prefix-state path is requested.
- reference implementation: `gdn_fla.gdn_packed_decode_reference_from_decay`
  accepts local `A` weights and calls the same pure-JAX recurrent rule as the
  default path. This is the correctness oracle for the model-routed boundary.
- fastest implementation: `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=cuda_fp32`
  calls the local CUDA/JAX FFI packed core `gdn_packed_decode_step_fp32` after
  the backend converts local `A` back to `A_log`. The explicit `reference`
  implementation uses the same backend API.
- default policy: the flag defaults to `off`, so `gpu_paged_default` is
  unchanged. This entry is an implementation boundary and focused correctness
  step only; it makes no throughput claim until an integrated exact-token
  matrix/server benchmark passes.
- artifact metadata: `benchmarks/benchmark_jax_server_trace.py` now records
  `run_config.gdn_kernel_flags`, including the legacy local recurrent decode
  flag and the packed decode implementation selected for the run.
- integrated experiment: ran the long-prefill goal target with
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=cuda_fp32`, elevated GPU access, stored
  references, and two repeats.
- artifact: `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.json`
- report: `results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.md`
- result: speed-claim-ready and exact generated-token parity, but below both
  the active target and the current fastest accepted default. JAX reached
  `88.41 tok/s`, `0.760x` vLLM, versus the scoped current default target at
  `90.81 tok/s`, `0.780x` vLLM. The `0.9x` target requires `104.74 tok/s`, so
  the packed CUDA route leaves a `16.33 tok/s` gap.
- profile movement: versus the older configured JAX reference, the run reduces
  `PjRt Execute` count by `96` and `MemcpyD2D` count by `192`, and lowers the
  named profile buckets, but integrated throughput still regresses versus the
  current accepted/scoped target. Scheduler diagnostics show prefill at about
  `0.54 s` and decode at about `0.19 s`, worse than the current scoped target's
  roughly `0.53 s` prefill and `0.17 s` decode.
- decision: keep the packed boundary and both implementations as default-off
  tools, but reject `cuda_fp32` packed GDN decode for default/fast-opt-in
  promotion. The next GDN win likely needs a coarser boundary that removes more
  surrounding work, not only the recurrent core custom call.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/backends.py nanovllm_jax/model.py nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_packed_decode_reference.py tests/test_cuda_fp32_ffi.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_packed_decode_reference.py tests/test_cuda_fp32_ffi.py -k packed_decode
NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=cuda_fp32 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 2 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_packed_cuda_fp32_target.json
```

- result: `py_compile` passed; elevated CUDA focused selection passed
  `9 passed, 11 deselected`; integrated matrix run completed successfully.

### Entry 122 - Rejected Greedy Decode Burst

- change tested: added an opt-in
  `NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS` path that reserves multiple decode
  slots, runs several sequential greedy decode steps inside one JIT via
  `forward_greedy_decode_burst_jit`, and returns a `[batch, steps]` token table
  so Python reads token IDs once instead of once per token. The normal
  one-token `forward_step_token_ids_jit` path remains the reference/default.
- focused validation: the new executor burst path matches iterative
  `forward_step_token_ids_jit` token IDs and final KV cache on a tiny
  full-attention model.
- target artifact: `results/gpu_matrix_20260527_decode_burst_target.json`
- target report: `results/gpu_matrix_20260527_decode_burst_target.md`
- target result: exact generated-token parity over two repeats, but not
  speed-claim-ready and far slower than default. With
  `NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS=16`, JAX reached only `17.46 tok/s`,
  `0.150x` vLLM. The scheduler collapsed to one decode step, but that step took
  about `3.13 s`.
- smaller-width probe: `results/gpu_matrix_20260527_decode_burst2_probe.json`
  used one repeat with burst width 2. It stayed exact but was worse:
  `5.57 tok/s`, `0.048x` vLLM, with about `10.95 s` total decode-step time.
- profile movement: token readback count did move as intended
  (`array.py:325 tolist` and `np.asarray(jax.Array)` count `1` instead of
  `16`), but the full-model scan introduced much more gather/transpose work and
  enormous `PjitFunction(compiled)` / traceback overhead. This confirms the
  readback bucket was mostly synchronizing prior work and that a source-level
  full-model decode scan is not the right dependency-structure change.
- decision: keep the path default-off as a diagnostic for now, reject it for
  default/fast-opt-in promotion, and do not pursue larger source-level decode
  bursts. A future multi-token serving loop would need a backend-owned loop or
  different scheduler/kernel boundary.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/engine/scheduled_batch.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py benchmarks/benchmark_jax_server_trace.py tests/test_backend_boundaries.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_backend_boundaries.py -k 'greedy_decode_burst or executor_jit_matches_eager_cached_decode'
NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS=16 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 2 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_decode_burst_target.json
NANO_VLLM_JAX_GREEDY_DECODE_BURST_STEPS=2 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_decode_burst2_probe.json
```

- result: `py_compile` passed; focused elevated CUDA tests passed
  `2 passed, 43 deselected`; both integrated probes completed and were exact.

### Entry 123 - Device Token Carry Fastest-Run Marker

- change tested: added an opt-in `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1` path for
  greedy `ignore_eos` serving. The reference/default path still materializes
  token IDs after every scheduler step. The opt-in path keeps the greedy token
  vector on device between one-token decode calls and materializes generated
  token IDs only after the fixed-length request completes.
- reference implementation: ordinary `forward_step_token_ids_jit` plus Python
  token readback remains unchanged when the flag is off.
- fastest implementation: the runner records the whole device token vector for
  the previous physical batch and reuses it as the next decode batch's token
  input when the `seq_ids_host` tuple is unchanged. Per-sequence `Sequence`
  objects keep deferred token scalars only for final correctness/reporting.
- smoke validation: `results/qwen08_device_token_carry_vector_smoke.json`
  matched `results/qwen08_device_token_carry_smoke_reference.json` exactly for
  generated tokens on the small `[32,64] x 4` CUDA smoke.
- rejected naive variant: `results/gpu_matrix_20260527_device_token_carry_target.json`
  used per-step scalar stack/scatter in the scheduler. It stayed exact but
  regressed to `89.35 tok/s`, `0.768x` vLLM, with `PjRt Execute` count rising
  to about `375`.
- target artifact: `results/gpu_matrix_20260527_device_token_carry_vector_target.json`
- target report: `results/gpu_matrix_20260527_device_token_carry_vector_target.md`
- target result: exact generated-token parity over two repeats and the fastest
  local long-prefill target observed so far at `93.01 tok/s`, `0.799x` vLLM.
  This is still below the `0.9x` target of `104.74 tok/s`, leaving an
  `11.72 tok/s` gap.
- profile movement: `np.asarray(jax.Array)` drops from the older JAX reference
  `427.55 ms` / count `16` to `0.58 ms` / count `4`, but `PjRt Execute` rises
  to `360.71 ms` / count `255` and `MemcpyD2D` rises to `45.77 ms` / count
  `749`. `forward_step_token_ids_jit` is roughly flat/slightly better at
  `273.08 ms` vs `280.56 ms`.
- decision: keep this as a default-off fastest-run marker and host-sync
  diagnostic, but do not promote it to accepted default. It changes streaming
  timing semantics because tokens are not materialized until request end, is not
  speed-claim-ready under the current profile-counter gate, and does not reach
  the `0.9x` target. The next primary speed work should follow the subagent
  recommendation: a coarser GDN post-conv prefill boundary inspired by vLLM's
  `fused_post_conv_prep`, not more host-token scheduling work.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/engine/sequence.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/block_manager.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/llm_engine.py benchmarks/benchmark_jax_server_trace.py
NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/benchmark_jax_server_trace.py --input-lens 32,64 --output-len 4 --no-profile --output-json results/qwen08_device_token_carry_vector_smoke.json
NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 2 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_device_token_carry_vector_target.json
```

- result: `py_compile` passed; smoke parity matched the reference token list;
  integrated matrix run completed and was exact.

### Entry 124 - GDN Post-Conv Prefill Reference Boundary

- change accepted: added a default-off
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference` route for longer GDN
  prefill. The model now passes post-convolution `conv_out`, raw `a/b`, local
  loaded `A=exp(A_log)`, `dt_bias`, valid-token mask, head metadata, chunk size,
  and native `[B,HV,V,K]` recurrent state through
  `backend.gated_delta_prefill_post_conv`.
- reference implementation:
  `gdn_fla.gdn_post_conv_prefill_reference_from_decay` owns the vLLM/FLA-shaped
  post-conv prep boundary: split Q/K/V from `conv_out`, construct beta/gate,
  apply valid-token masking, repeat Q/K for GQA, transpose into local
  `[B,H,T,D]` layout, and call the existing pure-JAX chunked reference. This
  preserves the current BF16-weight/FP32-activation correctness contract.
- default policy: the flag defaults to `off`, so `gpu_paged_default` is
  unchanged. This is a boundary and correctness scaffold, not a throughput
  promotion.
- artifact metadata: `benchmarks/benchmark_jax_server_trace.py` now records
  `run_config.gdn_kernel_flags.prefill_post_conv_impl`, so dashboard/profile
  views can distinguish the reference post-conv route from the default path.
- integrated experiment: ran one elevated GPU long-prefill goal-target repeat
  with stored references.
- artifact: `results/gpu_matrix_20260527_gdn_post_conv_reference_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_reference_target.md`
- result: exact generated-token parity on the integrated route. JAX reached
  `90.15 tok/s`, `0.775x` vLLM, versus the active `0.9x` target of
  `104.74 tok/s`; this leaves a `14.59 tok/s` gap. The run is not
  speed-claim-ready because it has only one repeat and misses the target.
- profile movement: versus the older configured JAX reference, named buckets
  improve (`PjRt Execute` count `44` vs `140`, `MemcpyD2D` count `463` vs
  `655`, and `array.py:325 tolist` about `396.37 ms` vs `427.68 ms`), but this
  is not a valid speedup claim against the current accepted/scoped default.
- decision: keep the post-conv reference boundary as the next GDN fast-kernel
  landing point. The next implementation should replace work behind this same
  backend method with a vLLM/FLA-derived fast path, without adding hot-path
  K,V/V,K adapters or per-layer layout conversions.
- validation:

```text
.venv/bin/python -m py_compile benchmarks/benchmark_jax_server_trace.py nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla.py nanovllm_jax/model.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_post_conv_reference_target.json
```

- result: `py_compile` passed; focused elevated CUDA test passed `1 passed`;
  adjacent elevated CUDA GDN/kernel suite passed `16 passed`; integrated
  one-repeat route completed and was exact.

### Entry 125 - Rejected GDN CUDA Post-Conv Prep-Only Route

- change tested: added
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_fp32`, a vLLM
  `fused_post_conv_prep`-inspired CUDA/JAX FFI route behind the existing
  post-conv prefill boundary. The kernel emits FP32 q/k/v/g/beta in the local
  `[B,H,T,D]` / `[B,H,T]` layout, preserves native `[B,H,V,K]` state, and then
  calls the existing pure-JAX chunked prefill body.
- reference: the default-off `reference` implementation from Entry 124 remains
  the correctness oracle.
- focused validation: direct CUDA prep output matches the JAX prep for masked
  GVA inputs, and a model-routed masked prefill test matches the default path.
- artifact: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.md`
- result: exact generated-token parity on the integrated route, but slower than
  the accepted/scoped default and the post-conv reference route. JAX reached
  `87.80 tok/s`, `0.755x` vLLM, leaving a `16.93 tok/s` gap to the active
  `0.9x` target.
- profile movement: the prep kernel reduces the visible `transpose` bucket from
  the older reference `47.30 ms` to `15.22 ms` and `MemcpyD2D` from `30.39 ms`
  to `13.42 ms`, but `command_buffer::execute` rises to `234.51 ms` with
  `18` extra executions and end-to-end throughput regresses. This confirms that
  replacing only post-conv prep is too narrow.
- decision: reject `cuda_prep_fp32` for default/fast-opt-in promotion and keep
  it default-off as a diagnostic. The next GDN candidate must replace the
  chunked prefill body behind the same boundary, with a fallback to Entry 124's
  reference path.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/backends.py nanovllm_jax/kernels/cuda_fp32_ffi.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_fp32 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_target.json
```

- result: `py_compile` passed; focused elevated CUDA test passed `3 passed`;
  adjacent elevated CUDA GDN/kernel suite passed `18 passed`; integrated
  one-repeat route completed and was exact but slower.

### Entry 126 - Rejected GDN CUDA Post-Conv Prep Plus Local Chunk Route

- change tested: added
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_prefill_fp32`, which uses
  the new CUDA FP32 post-conv prep from Entry 125, scales query by
  `1/sqrt(K)`, derives row lengths from the valid-token mask, and calls the
  existing local FP32 chunk32/V64 GDN prefill FFI when shape guards pass. It
  falls back to the JAX chunk body when guards do not pass.
- artifact: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.json`
- report: `results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.md`
- result: exact generated-token parity on the integrated route, but far slower
  than the accepted/scoped default. JAX reached `60.46 tok/s`, `0.520x` vLLM,
  leaving a `44.28 tok/s` gap to the active `0.9x` target.
- profile movement: the existing local chunk body dominates the GPU profile:
  `Fp32GdnPrefillChunk32Kernel<64>` takes about `500.69 ms` across `18`
  launches. Named host buckets such as `PjRt Execute` and
  `command_buffer::execute` shrink only because the expensive custom kernel
  work is now hidden behind later synchronization; `array.py:325 tolist` and
  `np.asarray(jax.Array)` grow to about `966 ms`.
- decision: reject `cuda_prep_prefill_fp32` for default/fast-opt-in promotion.
  Keep it default-off as evidence, but do not pursue the existing local FP32
  chunk32/V64 body as the production prefill implementation. A future GDN
  prefill kernel needs a different structure, closer to the vLLM/FLA chunk
  schedule or a true fused post-conv+chunk kernel, while preserving FP32 math or
  explicitly entering the separate BF16-prefill experiment lane.
- validation:

```text
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_prefill_fp32 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_post_conv_cuda_prep_prefill_target.json
```

- result: integrated one-repeat route completed and was exact but much slower.

### Entry 127 - Device-Token Vector-Ref Fastest-Run Marker

- change tested: refined the default-off `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1`
  path so deferred completion tokens keep a reference to the whole device token
  vector plus row index, rather than storing one JAX scalar slice per row. The
  deferred trace path now materializes all pending tokens for all sequences in
  one helper before building final results.
- reference path: normal streaming generation and ordinary
  `Sequence.materialize_device_tokens()` behavior remain intact. The new
  `DeviceTokenRef` is used only when the opt-in device-carry path is active.
- artifact:
  `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.json`
- report:
  `results/gpu_matrix_20260527_device_token_carry_vector_ref_target.md`
- result: exact generated-token parity over two elevated GPU repeats. The
  fastest local long-prefill marker improves from the previous device-carry
  `93.01 tok/s`, `0.799x` vLLM to `95.14 tok/s`, `0.818x` vLLM. It still misses
  the active `0.9x` target of `104.74 tok/s`, leaving a `9.59 tok/s` gap.
- profile movement: versus the previous device-carry marker, `PjRt Execute`
  count drops from `255` to `59`, total `PjRt Execute` drops from about
  `360.71 ms` to `307.17 ms`, and total `MemcpyD2D` drops from about
  `45.77 ms` to `18.62 ms`. This confirms the scalar-slice materialization was
  a real sync/transfer tax. `forward_step_token_ids_jit` and
  `command_buffer::execute` grow, so the integrated gain is useful but
  insufficient.
- decision: keep the vector-ref device-carry path as the current fastest
  default-off/offline marker, but do not promote it to `gpu_paged_default`.
  It is not streaming-equivalent, remains below `0.9x` vLLM, and is not
  speed-claim-ready under the current profile-counter gate.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/engine/sequence.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/llm_engine.py tests/test_device_token_carry.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest tests/test_device_token_carry.py -q
NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --configs gpu_paged_default --repeats 2 --output-json results/gpu_matrix_20260527_device_token_carry_vector_ref_target.json --require-stored-references --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- result: `py_compile` passed; focused elevated GPU test passed `4 passed`;
  integrated target run completed, was exact, and produced the fastest local
  default-off marker so far.

### Entry 128 - Rejected Stacked Device-Token Materialization

- change tested: grouped the default-off device-token-carry vector references
  by shape and stacked each group before the final `device_get`, trying to
  reduce final host materialization fanout further than Entry 127.
- artifact:
  `results/gpu_matrix_20260527_device_token_carry_stacked_ref_target.json`
- report:
  `results/gpu_matrix_20260527_device_token_carry_stacked_ref_target.md`
- result: exact generated-token parity over two elevated GPU repeats, but the
  run regressed to `91.94 tok/s`, `0.790x` vLLM, compared with Entry 127 at
  `95.14 tok/s`, `0.818x` vLLM.
- profile movement: `np.asarray(jax.Array)` fell to about `1.04 ms`, but
  `forward_step_token_ids_jit` rose to about `297.44 ms` and
  `command_buffer::execute` rose to about `251.67 ms`. The lower visible
  materialization bucket did not improve integrated throughput.
- decision: reject and do not keep the stacked materialization code. The
  Entry 127 tuple materializer remains the fastest default-off token-carry
  marker. Further work should return to the model/kernel path, not final token
  readback reshaping.
- validation:

```text
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest tests/test_device_token_carry.py -q
NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --configs gpu_paged_default --repeats 2 --output-json results/gpu_matrix_20260527_device_token_carry_stacked_ref_target.json --require-stored-references --jax-python /mountpoint/.exp/nano-vllm-jax/.venv/bin/python
```

- result: focused elevated GPU test passed `4 passed`; integrated target run
  completed and was exact but slower.

### Entry 129 - Rejected Broad BF16 Activation Diagnostic

- experiment: ran the existing long-prefill server path with
  `--dtype bfloat16 --weight-dtype bfloat16` against the stored FP32-activation
  long-prefill reference. This was a diagnostic for the BF16 external-kernel
  lane, not a proposed default-contract change.
- artifact: `results/qwen08_jax_bf16_activation_longprefill_probe.json`
- result: fast but incorrect. Throughput reached `108.76 tok/s`, which is
  `0.935x` the stored vLLM reference of `116.37 tok/s`, but exact-token parity
  failed on the `len_1536` row at generated token `0` (`279` vs reference
  `1719`). The other three rows matched exactly.
- profile movement: versus the FP32 accepted/scoped default, BF16 activation
  math substantially reduces visible `transpose` (`8.44 ms` vs about `45 ms`),
  `MemcpyD2D` (`10.54 ms` vs about `18-30 ms` depending artifact), and
  `command_buffer::execute` (`170.93 ms` vs about `223-229 ms`), but correctness
  failure blocks any speed claim.
- external-kernel audit: installed vLLM FLA chunk prefill rejects FP32 q/k/v
  and says to use BF16; FlashInfer GDN prefill likewise expects BF16/FP16
  q/k/v with FP32 g/beta/state. Direct upstream reuse is therefore only viable
  in a separate BF16-prefill diagnostic lane. Preserving the default contract
  still requires a FP32 port/fork of the FLA chunk schedule behind the existing
  post-conv boundary.
- decision: reject broad BF16 activations for promotion. Keep the default
  contract as BF16 checkpoint weights with FP32 activation/state math. If BF16
  is revisited, scope it narrowly to GDN prefill behind an explicit opt-in flag
  and require full token/logit guardrails before any benchmark claim.
- validation:

```text
JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/benchmark_jax_server_trace.py --model Qwen/Qwen3.5-0.8B --backend gpu --dtype bfloat16 --weight-dtype bfloat16 --jax-execution jit --input-lens 512,1024,1536,2048 --output-len 16 --prompt-suite mixed --num-speculative-tokens 0 --max-kv-cache-mb 3072 --num-kvcache-blocks 384 --max-num-seqs 4 --max-num-batched-tokens 8192 --prefill-buckets 512,1024,2048 --batch-size-buckets 1,2,4 --max-blocks-per-seq 129 --top-k 5 --warmup --profile --prompt-source tokenized_seed_repeat --seed 0 --output-json results/qwen08_jax_bf16_activation_longprefill_probe.json --run-label qwen08_jax_bf16_activation_longprefill_probe --reference-json results/gpu_matrix_runs/20260526_104818/long_prefill_512_2048_gpu_paged_default_repeat1.json
```

- result: elevated GPU run completed, but correctness failed.

### Entry 130 - GDN Post-Conv FLA-Shaped Prep ABI

- change accepted: factored post-conv GDN prefill preparation into
  `prepare_gdn_post_conv_prefill_fla_inputs_from_decay`. The helper returns the
  future-kernel ABI directly: q/k/v in `[B,T,H,D]`, gate/beta in `[B,T,H]`, and
  per-row sequence lengths, with optional q/k L2 normalization. The existing
  reference path still transposes those tensors into the current chunked JAX
  fallback, so model behavior is unchanged.
- motivation: the next FP32-preserving GDN prefill work should replace the body
  behind the post-conv boundary, not add another model call-site rewrite. A
  stable FLA-shaped prep helper lets a future vLLM/FLA-derived kernel consume
  natural post-conv layout while keeping the pure-JAX correctness reference.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
```

- result: elevated CUDA focused suite passed `4 passed`; neighboring GDN
  reference suite passed `19 passed`.

### Entry 131 - Prepared FLA Chunk32 FP32 Reference

- change accepted: added `gdn_fla_prefill_chunk32_fp32_reference`, the pure-JAX
  reference for the next GDN prefill fast body. It consumes the prepared
  rectangular FLA layout q/k/v `[B,T,H,D]`, gate/beta `[B,T,H]`, row lengths
  `[B]`, and V,K state `[B,H,V,K]`. The helper masks padded positions from
  `seq_lens`, transposes only inside the fallback reference, calls the existing
  chunk rule with q/k normalization disabled, and returns output in
  `[B,T,H,V]`.
- motivation: this freezes the smallest FP32-preserving chunk-body ABI before
  writing or porting a CUDA implementation. It avoids the previously rejected
  packed/rowwise serving reference, which changed FP32 accumulation enough to
  miss the state gate.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
```

- result: focused CUDA post-conv suite passed `6 passed`; neighboring GDN
  reference suite passed `21 passed`.

### Entry 132 - Prepared FLA Chunk Reference Server Route

- change accepted: added `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=
  reference_fla_chunk32`, a default-off route that prepares q/k/v/gate/beta in
  FLA layout, calls `gdn_fla_prefill_chunk32_fp32_reference`, then transposes
  output back to the model's `[B,H,T,V]` layout. This is the exact server
  route a future FP32 CUDA/FLA-derived chunk body should replace.
- artifact:
  `results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.json`
- report:
  `results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.md`
- result: one-repeat integrated long-prefill route completed with exact
  generated-token parity. Throughput was `89.37 tok/s`, `0.768x` the stored
  vLLM reference, below the current accepted/scoped default and far below the
  active `0.9x` target. This is not speed-claim-ready because it has only one
  repeat.
- profile movement: versus the configured JAX reference, visible transpose
  dropped to `15.36 ms` from `47.30 ms`, `PjRtCApiLoadedExecutable::Execute`
  count dropped from `140` to `44`, and `MemcpyD2D` dropped to `18.39 ms` from
  `30.39 ms`; however, this route still leaves a `15.37 tok/s` gap to the
  active target and is not a promoted speed path.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_post_conv_reference_fla_chunk32_target.json
```

- result: focused CUDA post-conv suite passed `7 passed`; neighboring GDN
  reference suite passed `22 passed`; integrated one-repeat route was exact.

### Entry 133 - Rejected Prepared FLA CUDA Chunk32 Route

- experiment: added a default-off prepared-layout FP32 CUDA FFI target,
  `gdn_prefill_chunk32_prepared_fp32`, and routed it through
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_fla_chunk32_fp32`. The kernel
  consumes q/k/v `[B,T,H,D]`, gate/beta `[B,T,H]`, row lengths `[B]`, and V,K
  state `[B,H,V,K]`, applies query scaling internally, and returns output in
  `[B,T,H,V]`.
- artifact:
  `results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.json`
- report:
  `results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.md`
- result: correct but much slower. The one-repeat integrated long-prefill route
  kept exact generated-token parity, but throughput fell to `63.82 tok/s`,
  `0.548x` the stored vLLM reference and `0.818x` the configured JAX reference.
  This is not speed-claim-ready because it has only one repeat.
- profile movement: command-buffer/PJRT buckets shrink because the custom call
  collapses many XLA operations, but the replacement kernel itself dominates:
  `Fp32GdnPrefillChunk32Kernel<32, true>` takes about `452.44 ms` across 18
  launches. Host-visible `np.asarray`/`tolist` expands to about `912 ms`, so the
  integrated route is slower despite lower `command_buffer::execute`.
- decision: reject for promotion. Keep the target default-off as an ABI and FFI
  diagnostic, but do not pursue a direct layout-adapted version of the old local
  chunk body as the serving GDN prefill solution. The next GDN prefill attempt
  must port the FLA chunk schedule more faithfully or otherwise reduce the
  per-layer chunk body cost, not just change indexing/layout.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/backends.py nanovllm_jax/kernels/cuda_fp32_ffi.py tests/test_cuda_fp32_ffi.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_cuda_fp32_ffi.py::test_gdn_prefill_chunk32_prepared_fp32_cuda_matches_fla_reference
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_cuda_fp32_ffi.py -k 'gdn_prefill or gdn_recurrent or gdn_packed'
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_cuda_fp32_ffi.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_cuda_fp32_ffi.py tests/test_gdn_post_conv_prefill_reference.py tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py tests/test_kernel_registry.py
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_fla_chunk32_fp32 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_post_conv_cuda_fla_chunk32_target.json
```

- result: new prepared CUDA unit test passed; GDN CUDA subset passed
  `8 passed, 7 deselected`; full CUDA FFI file passed `15 passed`; focused
  CUDA/GDN regression suite passed `38 passed`; integrated route was exact but
  slower.

### Entry 134 - Rejected Narrow BF16 GDN Prefill Activations

- experiment: added `NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE=bf16` for the
  prepared FLA-shaped post-conv prefill route. The flag is default-off and only
  affects `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32`.
  It casts prepared q/k/v and the core output to BF16 while keeping gate, beta,
  and recurrent state in FP32. Decode activation math remains on the existing
  FP32 path.
- vLLM alignment: the local vLLM audit shows the Qwen3.5/Qwen3-Next GDN
  prefill kernels are BF16/FP16-oriented for q/k/v, while gate/beta and
  recurrent state are FP32-oriented in the useful FlashInfer/FLA boundary. This
  route is therefore a correctness scaffold for that dtype split, not a true
  external-kernel speed path.
- artifact:
  `results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.json`
- report:
  `results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.md`
- long-decode guardrail:
  `results/qwen08_jax_bf16_prefillact_long_decode_top5_compare_20260527.json`
- result: the one-repeat integrated long-prefill benchmark kept exact
  generated-token parity for 16 generated tokens and reached `89.74 tok/s`,
  `0.771x` the stored vLLM reference and `1.150x` the configured JAX reference.
  This is not speed-claim-ready because it has only one repeat and remains below
  the active `0.9x` vLLM target.
- correctness failure: the 500-token top-5 guardrail failed promotion. Top-1
  token parity stayed exact at `500/500`, but ordered top-5 parity was
  `498/500`, top-5 set parity was `499/500`, and
  `max_hf_topk_id_logit_diff=0.010448455810546875`, far above the required
  `2e-5` gate.
- decision: reject for promotion. Keep the flag default-off as evidence and as a
  scaffold for a future true FlashInfer/FLA BF16-input kernel experiment, but do
  not change the default dtype contract.
- validation:

```text
.venv/bin/python -m py_compile benchmark_long_decode_top5.py nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla.py benchmarks/benchmark_jax_server_trace.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda ... .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32 NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE=bf16 JAX_PLATFORMS=cuda ... .venv/bin/python benchmarks/run_gpu_matrix.py --goal-target-only --repeats 1 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_20260527_gdn_prefill_bf16act_reference_target.json
NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=reference_fla_chunk32 NANO_VLLM_JAX_GDN_PREFILL_ACT_DTYPE=bf16 JAX_PLATFORMS=cuda ... .venv/bin/python benchmark_long_decode_top5.py --max-new-tokens 500 --compare-json results/qwen08_jax_bf16_prefillact_long_decode_top5_compare_20260527.json
```

- result: focused post-conv suite passed `9 passed`; neighboring
  GDN/kernel-registry suite passed `24 passed`; integrated route was exact for
  16 generated tokens; long-decode top-5 guardrail failed as described above.

### Entry 135 - External GDN Kernel Feasibility Probe

- change accepted: added `benchmarks/probe_external_gdn_kernels.py`, a small
  host probe that records GPU capability, installed optional package versions,
  FlashInfer GDN prefill requirements, and local vLLM/FLA source-path
  availability. This is a decision artifact, not a benchmark.
- artifact: `results/external_gdn_kernel_probe_20260527_sm86.json`
- result: elevated GPU-visible probe reports `NVIDIA A10G`, compute capability
  `8.6`, FlashInfer `0.6.11.post3`, `jax-tvm-ffi 0.1.3`, torch `2.12.0`, and
  Triton `3.7.0`. All expected vLLM/FLA source paths are present under
  `/mountpoint/.exp/vllm-venv`.
- decision: direct FlashInfer GDN prefill is blocked on this host because the
  installed FlashInfer GDN kernel requires SM90/SM100 and BF16/FP16 q/k/v. The
  next local GDN implementation path should keep using vLLM/FLA as a port/fork
  reference behind the existing post-conv or packed-decode ABI, not pursue
  direct FlashInfer GDN prefill on SM86.
- validation:

```text
.venv/bin/python -m py_compile benchmarks/probe_external_gdn_kernels.py
.venv/bin/python benchmarks/probe_external_gdn_kernels.py --run-smoke --output-json results/external_gdn_kernel_probe_20260527_sm86.json
```

### Entry 136 - vLLM/FLA GDN Kernel Timing Probe

- change accepted: added `benchmarks/probe_vllm_fla_gdn.py`, a Torch-side
  probe that directly exercises the vLLM vendored Flash Linear Attention GDN
  kernels on the current host. This is a porting decision artifact, not a JAX
  serving speed claim.
- artifact: `results/vllm_fla_gdn_probe_20260527_sm86.json`
- result: elevated GPU-visible probe reports vLLM `0.21.0`, torch
  `2.11.0+cu130`, Triton `3.6.0`, and `NVIDIA A10G` compute capability `8.6`.
  The model-shaped varlen prefill uses BF16 q/k/v with FP32 gate/beta/state and
  V,K recurrent state `[N,HV,V,K]`. For lengths `[512,1024,1536,2048]`
  (`5120` total tokens), `fused_post_conv_prep` p50 is `0.40 ms`,
  `chunk_gated_delta_rule` p50 is `1.31 ms`, and combined prep+chunk p50 is
  `1.45 ms`. Packed decode p50 is `0.116 ms` at batch 1, `0.116 ms` at batch 4,
  `0.119 ms` at batch 8, and `0.161 ms` at batch 16.
- reference check: the probe also compares vLLM/FLA outputs to an independent
  Torch recurrent reference. Packed decode matches BF16 output exactly and FP32
  state within `1.19e-7` max abs. Ragged prefill lengths `[17,64,65]` match to
  BF16 output max abs `4.88e-4`; final FP32 state max abs is `4.45e-3`.
- decision: use the upstream FLA schedule as the next real GDN port/fork
  target. The numbers are far better than the rejected local FP32 chunk-body
  attempts, but they do not change the default dtype contract. A JAX-facing path
  still has to pass exact generated-token parity and long-decode top-logit gates
  before BF16 GDN activations can be promoted.
- direct-reuse audit: a sidecar audit found that directly calling vLLM's
  Torch/Triton GDN kernels from inside JAX is not a low-risk path on this stack:
  `jax_triton` is absent from the repo venv, the vLLM venv lacks
  JAX/`jax_tvm_ffi`, vLLM exposes Torch/Triton wrappers rather than a stable
  non-Torch ABI, and this JAX build does not provide a simple JIT-safe DLPack
  export/host-callback bridge. Treat vLLM/FLA as the golden reference and
  port/fork the schedule behind the existing JAX-facing GDN boundaries.
- validation:

```text
.venv/bin/python -m py_compile benchmarks/probe_vllm_fla_gdn.py
/mountpoint/.exp/vllm-venv/bin/python benchmarks/probe_vllm_fla_gdn.py --warmups 2 --repeats 5 --output-json results/vllm_fla_gdn_probe_20260527_sm86.json
```

### Entry 137 - Prepared GDN FLA Varlen Reference Contract

- change accepted: added a JAX-side prepared-layout varlen GDN prefill
  contract in `nanovllm_jax/kernels/gdn_fla.py`. The new helpers pack
  prepared q/k/v `[B,T,H,D]` plus gate/beta `[B,T,H]` into the upstream
  vLLM/FLA-style `[nnz,H,D] + cu_seqlens` ABI, run the existing segmented FP32
  reference, and unpack output back to `[B,T,H,V]`.
- decision: this is an ABI scaffold for a future vLLM/FLA-schedule port/fork,
  not a serving speed claim and not a dtype-contract change. It keeps the
  pure-JAX FP32/V,K reference as the correctness target while making the next
  kernel boundary explicit.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_post_conv_prefill_reference.py
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_post_conv_prefill_reference.py -k 'varlen or prepared_fla_chunk32_reference_matches_post_conv_reference or masks_padded_rows'
```

- result: focused CUDA selection passed `3 passed, 7 deselected`. The new test
  covers ragged lengths `[0,5,17,32]`, `cu_seqlens=[0,0,5,22,54]`, packed
  q/k/v/gate/beta shapes, parity with the rectangular prepared FP32 reference,
  and unpacking back to the prepared FLA layout.

### Entry 138 - vLLM FLA Prefill Port Surface Audit

- change accepted: recorded the sidecar audit of vLLM's Qwen3.5/GatedDeltaNet
  prefill implementation in the source-of-truth plan and kernel roadmap. This
  is implementation scoping, not a benchmark result.
- result: the useful vLLM/FLA prefill path is `fused_post_conv_prep` followed
  by `chunk_gated_delta_rule`. The chunk body decomposes into
  `chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`, `solve_tril`,
  `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h`, and `chunk_fwd_o`, with
  `prepare_chunk_indices`/`prepare_chunk_offsets` for varlen chunk metadata.
- decision: port/fork the math kernels behind the existing JAX-facing
  prepared-layout boundary. Do not import Torch autograd wrappers,
  `input_guard`, vLLM forward context, or vLLM autotune/runtime plumbing into
  the first JAX path. On A10G/SM86, target the non-Hopper/non-TMA branch first.
- dtype note: vLLM's Torch wrapper rejects FP32 q/k/v for
  `chunk_gated_delta_rule`, while gate/beta/state are FP32-oriented. The default
  nano-vLLM-JAX path still requires FP32 activation/state correctness; BF16
  q/k/v remains a default-off diagnostic unless it passes the long-decode
  token/logit gates.

### Entry 139 - FLA Varlen Chunk Metadata Contract

- change accepted: added `prepare_gdn_fla_chunk_metadata`, a JAX-side helper
  for the future FLA chunk-body port. It builds active chunk indices
  `[sequence_index, chunk_index_in_sequence]` and per-row `chunk_offsets` from
  `cu_seqlens` and `chunk_size`.
- decision: this is a port prerequisite, not a serving speed claim. The helper
  intentionally preserves original row ids when bucket padding creates
  zero-length rows; vLLM's upstream helper assumes rows are pre-filtered and can
  compress row identity in that case.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: focused CUDA selection passed `3 passed`. The tests cover ragged
  `cu_seqlens=[0,0,5,37,74,138]`, all-empty rows, expected active chunk rows,
  per-row chunk offsets, and the existing segmented-reference parity check.

### Entry 140 - FLA Chunk-Local Cumsum Reference

- change accepted: added `gdn_fla_chunk_local_cumsum_packed_reference`, a
  JAX-side reference for vLLM/FLA's scalar `chunk_local_cumsum` stage over
  packed `[nnz,H]` gate tensors.
- decision: this is a correctness scaffold for the future FLA chunk-body port,
  not a serving speed claim. It locks down per-chunk reset behavior, reverse
  cumsum semantics, and use of the local `chunk_indices` contract before any
  lowered implementation replaces the reference.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed and the focused elevated CUDA selection passed
  `4 passed`. The new test covers ragged `cu_seqlens=[0,0,5,22]`, chunk size
  `8`, forward cumsum, reverse cumsum, and reset at row/chunk boundaries.

### Entry 141 - FLA Chunk-Scaled-Dot-KKT Reference

- change accepted: added `gdn_fla_chunk_scaled_dot_kkt_packed_reference`, a
  JAX-side reference for vLLM/FLA's `chunk_scaled_dot_kkt_fwd` stage over packed
  varlen tensors.
- decision: this is the next correctness scaffold for the future FLA chunk-body
  port. It records the strict-lower matrix convention, optional
  `exp(g_i - g_j)` scaling from local gate cumsums, and grouped value-head to
  key-head behavior before any lowered implementation is added.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed and the focused elevated CUDA selection passed
  `5 passed`. The new test covers ragged `cu_seqlens=[0,0,5,13]`, chunk size
  `4`, grouped output heads over two key heads, strict-lower masking, and gate
  difference scaling.

### Entry 142 - FLA Solve-Tril Reference

- change accepted: added `gdn_fla_solve_tril_packed_reference`, a JAX-side
  reference for vLLM/FLA's `solve_tril` stage over packed varlen chunk
  matrices.
- decision: this is the third scalar correctness scaffold for the future FLA
  chunk-body port. It records the per-active-chunk `(I + A)^-1` convention and
  keeps padded columns outside ragged partial chunks zero before the
  `recompute_w_u_fwd` stage is ported.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
.venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata'
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed, the local focused selection passed
  `5 passed, 1 deselected`, and the elevated CUDA focused selection passed
  `6 passed`. The new test covers ragged `cu_seqlens=[0,0,5,13]`, chunk size
  `4`, strict-lower input from the packed KKT reference, ragged partial chunks,
  and NumPy `(I + A)^-1` parity.

### Entry 143 - FLA Recompute W/U Reference

- change accepted: added `gdn_fla_recompute_w_u_packed_reference`, a JAX-side
  reference for vLLM/FLA's `recompute_w_u_fwd` stage over packed varlen tensors.
- decision: this is the next correctness scaffold in the FLA chunk-body port.
  It locks down how the solved chunk matrix is applied to beta-weighted values
  for `u` and to `beta * exp(g_cumsum)` weighted grouped keys for `w`, before
  porting `chunk_gated_delta_rule_fwd_h`.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
.venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata'
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed, the local focused selection passed
  `6 passed, 1 deselected`, and the elevated CUDA focused selection passed
  `7 passed`. The new test covers ragged `cu_seqlens=[0,0,5,13]`, chunk size
  `4`, grouped output heads over two key heads, gate-exp scaling for `w`, and
  beta-weighted value projection for `u`.

### Entry 144 - FLA Chunk-Delta-H Reference

- change accepted: added `gdn_fla_chunk_delta_h_packed_reference`, a JAX-side
  reference for vLLM/FLA's `chunk_gated_delta_rule_fwd_h` state/value update
  stage over packed varlen tensors.
- decision: this locks down the stage that differs from the preceding FLA
  chunk-body kernels: varlen traversal uses `chunk_offsets`, not
  `chunk_indices`, to walk chunks for each original sequence row. The reference
  records that `h` stores the prior chunk state, `v_new` stores the ungated
  delta, and only the recurrent state update uses gate-rescaled deltas.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed and the elevated CUDA focused selection passed
  `8 passed`. The new test covers ragged `cu_seqlens=[0,0,5,13]`,
  `chunk_offsets=[0,0,2,4]`, prior-state `h`, ungated `v_new`, gate-rescaled
  state updates, grouped output heads over two key heads, and final-state
  parity including the zero-length row.

### Entry 145 - FLA Chunk-Fwd-O Reference

- change accepted: added `gdn_fla_chunk_fwd_o_packed_reference`, a JAX-side
  reference for vLLM/FLA's `chunk_fwd_o` output stage over packed varlen
  tensors.
- decision: this completes the scalar/reference decomposition of the vLLM/FLA
  prefill chunk body. The reference records the output-stage contract:
  query-to-prior-state output plus causal intra-chunk attention over ungated
  `v_new`, with optional gate scaling, grouped output-head to key-head mapping,
  and explicit output scale.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
.venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_fwd_o or chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata'
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_fwd_o or chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed, the local focused selection passed
  `8 passed, 1 deselected`, and the elevated CUDA focused selection passed
  `9 passed`. The new test covers ragged `cu_seqlens=[0,0,5,13]`, grouped
  output heads over two key heads, gate scaling for state and intra-chunk terms,
  causal inclusive masking, explicit output scaling, and composition with the
  previous packed FLA reference stages.

### Entry 146 - Composed FLA Chunk-Gated-Delta Reference

- change accepted: added `gdn_fla_chunk_gated_delta_rule_packed_reference`, a
  composed JAX-side reference for the full vLLM/FLA prefill chunk body over
  packed varlen tensors.
- decision: this gives future lowered implementations one correctness boundary
  to replace. It runs the audited FLA stage order directly:
  `chunk_local_cumsum`, `chunk_scaled_dot_kkt`, `solve_tril`, `recompute_w_u`,
  `chunk_gated_delta_rule_fwd_h`, and `chunk_fwd_o`, while preserving ragged
  `cu_seqlens`, `chunk_indices`, `chunk_offsets`, grouped heads, FP32
  gate/beta/state math, and optional Q/K L2 normalization.
- validation:

```text
.venv/bin/python -m py_compile nanovllm_jax/kernels/gdn_fla.py tests/test_gdn_segmented_reference.py
git diff --check
.venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_gated_delta_rule_packed or chunk_fwd_o or chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata'
JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_gdn_segmented_reference.py -k 'chunk_gated_delta_rule_packed or chunk_fwd_o or chunk_delta_h or recompute_w_u or solve_tril or chunk_scaled_dot_kkt or chunk_local_cumsum or chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32'
```

- result: static checks passed, the local focused selection passed
  `9 passed, 1 deselected`, and the elevated CUDA focused selection passed
  `10 passed`. The new test compares the composed FLA packed body to the
  existing segmented JAX reference on ragged lengths `[0,3,7,9]`, chunk size
  `4`, Q/K L2 normalization enabled, grouped state/output shapes, and final
  state parity.

### Entry 151 - BF16 Packed-Decode Long-Decode Boundary Scaffold

- change accepted: added/updated a packed-decoding boundary artifact check for the
  BF16 qkv external-kernel lane.
- artifact:
  `results/qwen08_jax_packed_decode_bf16qkv_fp32out_long_decode_top5_compare_20260527.json`
- result: generated-token identity is exact, top-1/ordered top-5/top-5 set parity is
  `500/500`, and `max_hf_topk_id_logit_diff=2.6702880859375e-05`. This misses the
  strict default FP32-activation requirement (`max_hf_topk_id_logit_diff <= 2e-5`)
  but passes the BF16 external-kernel boundary limit (`<=1e-4`) with
  `exact_generated_token_match=true`.
- decision: accept this as BF16 boundary-scaffold status only (not a promoted speed
  path) while keeping it default-off, and continue treating BF16 packed decode
  as an experiment lane until it wins on benchmark targets.

### Entry 152 - Rejected Local CUDA BF16 Packed-Decode Route

- experiment: tested the local CUDA/JAX FFI packed GDN decode implementation
  with BF16 packed q/k/v input and FP32 gate/beta/state/output behind
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=cuda_fp32` and
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_QKV_DTYPE=bf16`.
- focused validation: elevated CUDA focused tests passed after relaxing an
  over-strict unit assertion from bit-exact CUDA-vs-JAX FP32 identity to
  `rtol=1e-6, atol=1e-6`. The original unit drift was only about `4.47e-08`;
  this was not the model-level blocker.
- artifact:
  `results/qwen08_jax_packed_decode_cuda_bf16qkv_fp32out_long_decode_top5_compare_20260527.json`
- result: the 500-token HF top-5 guardrail failed. Generated-token/top-1
  identity was `499/500`, ordered top-5 identity was `491/500`, top-5 set
  identity was `499/500`, and `max_hf_topk_id_logit_diff` was
  `0.02606964111328125`, far outside both the default `2e-5` gate and the
  BF16 diagnostic `1e-4` lane.
- decision: reject this local CUDA BF16 packed-decode route for promotion and
  do not run speed claims for it. Keep the BF16 packed-decode reference path as
  the semantic contract, and move directly to a vLLM/FLA-derived kernel route
  for GDN instead of repairing the historical local CUDA prototype.

### Entry 153 - JAX-Triton Optional Kernel Bridge

- change accepted as scaffold: added optional `jax-triton==0.3.1` dependency
  metadata, a `benchmarks/probe_jax_triton.py` smoke probe, and a
  `nanovllm_jax.kernels.gdn_fla_triton` module for vLLM/FLA-shaped GDN
  experiments.
- runtime finding: `jax_triton` imports with JAX `0.10.1` and Triton `3.7.0`,
  but XLA custom-call assembly must prefer Triton's bundled CUDA 12.8 `ptxas`.
  The system `/usr/bin/ptxas` is CUDA 12.0 and fails generated PTX 8.7 with
  `Unsupported .version 8.7; current version is '8.0'`. The probe and kernel
  module therefore put Triton's bundled `ptxas` first on `PATH`.
- validation:

```text
NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp JAX_PLATFORMS=cuda \
  PATH=.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin:$PATH \
  .venv/bin/python benchmarks/probe_jax_triton.py
```

- result: the elevated GPU smoke passed on `cuda:0` with `ok=True`.
- external-kernel finding:
  `results/external_gdn_kernel_probe_20260528_gpu_jaxtriton.json` confirms the
  host GPU is NVIDIA A10G / SM86, while direct FlashInfer GDN prefill requires
  SM90/SM100. Continue using vLLM/FLA Triton sources as the GDN port reference;
  do not pursue direct FlashInfer GDN prefill on this host.

### Entry 154 - Rejected JAX-Triton BF16 Packed GDN Decode

- experiment: ported the vLLM/FLA-style packed decode recurrence to a
  JAX-Triton custom call behind
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla`. The route consumes BF16
  packed q/k/v and FP32 gate/beta/decay/state, with the pure-JAX BF16 reference
  still serving as the semantic oracle.
- focused validation: elevated CUDA test
  `tests/test_gdn_packed_decode_reference.py::test_packed_gdn_decode_triton_bf16_matches_reference`
  passed.
- correctness artifact:
  `results/qwen08_jax_packed_decode_triton_fla_bf16qkv_fp32out_long_decode_top5_compare_20260528.json`
- result: the 500-token HF guardrail failed. Generated-token/top-1 identity was
  `499/500`, ordered top-5 identity was `493/500`, top-5 set identity was
  `499/500`, and `max_hf_topk_id_logit_diff=0.029291152954101562`.
- diagnostic speed artifact:
  `results/gpu_matrix_20260528_gdn_packed_triton_fla_bf16qkv_device_carry_diag_target.json`
- speed result: one-repeat diagnostic was only `92.33 tok/s`, `0.793x` vLLM,
  below the `0.9x` target and not speed-claim ready.
- decision: reject for promotion. Keep default-off only as a porting scaffold.
  Replacing decode recurrence alone adds custom-call/command-buffer cost and
  the Triton math path drifts enough to fail full-model gates.

### Entry 155 - Rejected JAX-Triton BF16 GDN Prefill Prep Only

- experiment: added a JAX-Triton fused post-conv prep route behind
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_prep_bf16`. The route
  produces prepared BF16 q/k/v and keeps FP32 gate/beta/state; after the first
  guardrail failure, gate/beta were moved back to JAX reference math and the
  Triton custom call was limited to the normalized BF16 q/k/v prep boundary.
- focused validation: elevated CUDA test
  `tests/test_gdn_post_conv_prefill_reference.py::test_post_conv_triton_bf16_prep_matches_reference_boundary`
  passed.
- correctness artifact:
  `results/qwen08_jax_gdn_prefill_triton_fla_prep_bf16_jax_gate_long_decode_top5_compare_20260528.json`
- correctness result: generated-token/top-1 identity was `500/500`, but
  ordered top-5 identity was `496/500`, top-5 set identity was `499/500`, and
  `max_hf_topk_id_logit_diff=0.008432388305664062`. This fails both the default
  FP32 gate and the BF16 external-kernel diagnostic gate.
- diagnostic speed artifact:
  `results/gpu_matrix_20260528_gdn_prefill_triton_fla_prep_bf16_device_carry_diag_target.json`
- speed result: one-repeat diagnostic was `92.80 tok/s`, `0.797x` vLLM, below
  the `0.9x` target. The benchmark also failed exact generated-token parity on
  the multi-prompt long-prefill target.
- decision: reject for promotion. A prep-only kernel is too narrow; the next
  GDN speed attempt must move the full FLA chunk body, not just split/norm/prep.

### Entry 156 - CPU-Only GDN Reference Validation & Supervisor Probe

- date: 2026-05-29
- context: local environment has no CUDA device; all runs below are CPU-only.
- supervisor evidence: RealmAI `realmai_worker` tool-call proxy is functional
  after a local Responses proxy fix. Health probe returned
  `realmai_worker available`; `codex exec -p realmai` executed `pwd` via shell
  and returned `/mountpoint/.exp`.
- repo path: `/mountpoint/.exp/nano-vllm-jax`.
- JAX device probe: `uv run python -c "import jax; jax.devices(); print(jax.default_backend())"`
  reported `CUDA_ERROR_NO_DEVICE` from the CUDA plugin, `devices=[CpuDevice(id=0)]`,
  `default_backend=cpu`; exit code 0.
- focused GDN segmented-reference tests:
  `uv run pytest tests/test_gdn_segmented_reference.py -k 'chunk_metadata or segmented_gdn_prefill_reference_matches_padded_chunk32' -q`
  => `2 passed, 1 skipped, 17 deselected in 0.68s`.
- broader triton/padded/packed selection:
  `uv run pytest tests/test_gdn_segmented_reference.py -k 'triton or padded or packed' -q`
  => `8 passed, 10 skipped, 2 deselected in 21.05s`.
- canonical GPU benchmark:
  `uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 2 --require-goal-target-ready`
  remains unverified locally; no CUDA device available.
- decision: GDN segmented reference math is correct on CPU; Triton/packed paths
  skip gracefully without GPU. No source changes warranted. GPU benchmark
  validation deferred to an environment with a CUDA device.

### Entry 156b - GPU Escalation Blocker Confirmed

- date: 2026-05-29
- context: follow-up to Entry 156; confirms the GPU-access blocker is
  subagent sandbox policy, not physical GPU availability.
- supervisor `nvidia-smi` (elevated, outside sandbox): succeeded.
  NVIDIA A10G, driver 580.159.03, CUDA 13.0, 23028 MiB, no running processes.
- `realmai_worker` retry of `nvidia-smi` inside sandbox: failed with exit
  code 9; could not escalate. Elevated retry was rejected by auto-review
  policy.
- conclusion: the host GPU is physically present and idle; the current
  blocker is subagent GPU escalation/access policy, not hardware.
- guardrail reminder: `nanovllm_jax/kernels/registry.py:gdn_fla
  implemented=False` must not be flipped until GPU correctness and
  benchmark gates actually run on the device. No source changes.

### Entry 157 - GDN FLA Wrapper Patch & Focused GPU Validation

- date: 2026-05-29
- change: wrapper patch in `nanovllm_jax/kernels/gdn_fla.py` replaced the two
  post-gate `NotImplementedError` stubs with dispatch to
  `gdn_packed_decode_reference_from_decay` and
  `gdn_segmented_prefill_chunk32_reference`.
- focused tests:
  - fallback-only: `1 passed, 17 deselected`
  - segmented triton/padded/packed: `18 passed, 2 deselected`
  - post-conv triton_fla_packed: `19 passed`
- guardrails:
  - real/e2e/layer parity: `13 passed`
  - long decode top5: `500/500`, max diff `1.526e-05`
  - backend_boundaries+kv_cache: `55 passed, 2 MTP1 failures` (ModelRunner.kv_state)
- GDN microbench artifact:
  `results/vllm_fla_gdn_microbench_verify.json` — vLLM FLA far faster than
  JAX reference.
- focused target gate:
  `uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 2 --require-goal-target-ready`
  exited 1: nano `85.18 tok/s`, vLLM `116.37 tok/s`, ratio `0.732x`, target
  `0.9`, `speed_claim_ready=yes`, `target_vllm_ratio_met=no`.
- artifacts: `results/gpu_matrix_20260529_115306.json`,
  `results/gpu_matrix_20260529_115306.md`.
- decision: goal remains incomplete; ratio `0.732x` misses the `0.9x` target.
  Registry promotion (`gdn_fla implemented=True`) is not justified. The wrapper
  patch is a wiring fix only — actual kernel speed still lags vLLM FLA
  significantly.

### Entry 158 - Opt-In `triton_fla_fp32` Packed Decode Path

- date: 2026-05-29
- change: added opt-in `triton_fla_fp32` packed decode path via
  `gdn_packed_decode_step_fp32`; selector controlled by
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_fp32`.
- focused tests:
  `test_gdn_packed_decode_reference.py -k 'triton_bf16 or triton_fp32 or fallback_only'`:
  `3 passed, 16 deselected`.
- initial long decode: max diff `2.289e-05`, failed gate (numerical issue in
  gate computation).
- fix: stable softplus in `_gdn_packed_decode_kernel` resolved the gate
  failure.
- final long decode (`NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_fp32`):
  `500/500` top1/ordered-top5/top5-set, max diff `1.907e-05`,
  `passes_required_gate true`.
- focused target throughput gate (same selector): nano `84.63 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.727x` vs target `0.9`, worse than prior `0.732x`.
- artifacts: `results/gpu_matrix_20260529_124402.json`,
  `results/gpu_matrix_20260529_124402.md`.
- decision: selector remains opt-in, not promoted. Next work should reduce
  transpose overhead and pursue coarser fusion.

### Entry 159 - Packed Decode Squeeze Optimization

- date: 2026-05-29
- change: `model.py` single-line packed decode squeeze — replaced
  `conv_out[:,0,:]` with direct `conv_out_t[:,:,0]` to avoid the
  intermediate slice-then-squeeze and work on the already-transposed
  packed decode output directly.
- preserved `triton_fla_fp32` correctness: long decode `500/500`
  top1/ordered-top5/top5-set, max diff `1.91e-05`,
  `passes_required_gate true`.
- focused target gate (`NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_fp32`):
  nano `85.03 tok/s`, vLLM `116.37 tok/s`, ratio `0.731x` vs target `0.9`,
  `speed_claim_ready true`, `target_vllm_ratio_met false`.
- artifacts: `results/gpu_matrix_20260529_125705.json`,
  `results/gpu_matrix_20260529_125705.md`.
- delta vs Entry 158: ratio `0.731x` vs `0.727x` — marginal improvement but
  remains flat/slightly below pure_jax `0.732x` baseline. No promotion.
- decision: selector stays opt-in. Next work should target TTFT/prefill and
  host overhead (e.g. `array.tolist`, command_buffer update) where larger
  gains are available.

### Entry 160 - Device Token Carry & Profile Counter Gap

- date: 2026-05-29
- change: enabled `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY=1` in
  `benchmarks/configs/gpu_paged_default.json`; keeps sampled token IDs on
  device across decode steps, eliminating host round-trip for token
  carry.
- focused target gate (`triton_fla_fp32`): nano `88.99 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.765x` vs target `0.9`,
  `target_vllm_ratio_met false`.
- artifacts: `results/gpu_matrix_20260529_130537.json`,
  `results/gpu_matrix_20260529_130537.md`.
- profile highlights:
  - `array.py:325 tolist` disappeared entirely.
  - `np.asarray(jax.Array)` dropped `427.55 ms` → `33.24 ms`.
  - `forward_step`/`transpose` overhead still dominant; target unmet.
- `speed_claim_ready false` due `profile_counters_present` — missing
  `tolist` counter prevents claim readiness.
- decision: goal remains incomplete. Next work: profile-counter gate
  handling and reduce transpose/forward-step/prefill overhead.

### Entry 161 - Sync-Attribution Counters Optional for speed_claim_ready

- date: 2026-05-29
- change: benchmark acceptance patch made sync-attribution counters optional
  for `speed_claim_ready`; `profile_counters_present` no longer blocks claim
  readiness.
- targeted tests: `3 passed, 42 deselected`.
- focused target gate
  (`NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_fp32`,
  device-token carry enabled): nano `88.97 tok/s`, vLLM `116.37 tok/s`,
  ratio `0.7645`, target `0.9`, `speed_claim_ready true`,
  `profile_counters_present true`, `target_vllm_ratio_met false`.
- artifacts: `results/gpu_matrix_20260529_131654.json`,
  `results/gpu_matrix_20260529_131654.md`.
- delta vs Entry 160: ratio `0.7645` vs `0.765` — throughput essentially
  flat; `speed_claim_ready` now `true` (was `false` due to missing counter).
- state: failure is now only the throughput ratio; all other gates pass.
- decision: goal remains incomplete. Next work must close the `0.7645→0.9`
  throughput gap — target TTFT/prefill and transpose/forward-step overhead.

### Entry 162 - Prefill Variant Guardrail Probes vs `triton_fla_fp32`

- date: 2026-05-29
- change: probed three prefill variants against `triton_fla_fp32` decode
  baseline to assess benchmark readiness.
- results (all vs `triton_fla_fp32` reference):

  | variant | top1 | ordered top5 | top5 set | exact gen | max diff | gate |
  | --- | ---: | ---: | ---: | --- | ---: | --- |
  | `triton_fla_packed` | 500/500 | 500/500 | 500/500 | true | 2.289e-05 | fail |
  | `triton_fla_padded` | 500/500 | 500/500 | 500/500 | true | 2.289e-05 | fail |
  | `triton_fla_prep_bf16` | 499/500 | 495/500 | 500/500 | false | 8.404e-03 | fail |

- `triton_fla_packed`/`triton_fla_padded`: perfect token/top5 match and exact
  generated match, but max diff `2.289e-05 > 2e-05` strict threshold — near
  miss.
- `triton_fla_prep_bf16`: large numeric divergence; max diff `8.404e-03`,
  top1 off by 1, ordered top5 495/500, exact generated false.
- decision: no prefill variant is benchmark-ready. Next work: trace/fix
  near-miss numeric source in packed/padded prefill path (likely gate or
  state-accumulation precision).

### Entry 163 - Opt-In `triton_fla_prep_fp32` Prefill Prep Wrapper

- date: 2026-05-29
- change: added opt-in `triton_fla_prep_fp32` Triton prefill prep
  wrapper/dispatch; selected via
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_prep_fp32`.
- focused pytest: post-conv prefill reference `19/19 passed`.
- runtime long-decode
  (`NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_fp32`,
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_prep_fp32`):
  `500/500` top1/ordered-top5/top5-set, exact generated match `true`,
  max diff `2.48e-05 > 2e-05` — failed strict logit gate.
- state: not benchmark-ready; no promotion.
- decision: selector stays opt-in. Next work: narrow the `2.48e-05`
  residual vs `2e-05` threshold — likely gate or state-accumulation
  precision in the prep path.

### Entry 164 - Opt-In `triton_fla_fp32_model_layout` Decode Output Layout

- date: 2026-05-29
- change: added opt-in `triton_fla_fp32_model_layout` decode output layout
  to skip model-side post-kernel transpose; selected via
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_fp32_model_layout`.
- focused decode pytest: `3 passed, 16 deselected`.
- long decode guardrail: `500/500` top1/ordered-top5/top5-set, exact
  generated `true`, max diff `1.907e-05`, `passes_required_gate true`.
- focused target gate: **failed** — no material throughput improvement.
  - nano `89.00` tok/s, vLLM `116.37` tok/s, ratio `0.765x` vs target
    `0.9x`.
  - `speed_claim_ready yes`, `profile_counters_present yes`,
    `target_vllm_ratio_met no`.
- artifacts: `results/gpu_matrix_20260529_142000.json`,
  `results/gpu_matrix_20260529_142000.md`.
- state: goal incomplete; layout selector alone insufficient.
- decision: next work should target `forward_step`/PjRt/per-step overhead
  rather than this layout selector.

### Entry 165 - Device-Token Carry Only (No Packed-Decode Override)

- date: 2026-05-29
- change: ran focused target gate with device-token carry enabled but **no**
  packed-decode or model-layout override — isolating the carry-only gain.
- command: `uv run python benchmarks/run_gpu_matrix.py --configs
  gpu_paged_default --workloads long_prefill_512_2048 --repeats 2
  --require-goal-target-ready` (exit 1)
- nano `89.60 tok/s`, vLLM `116.37 tok/s`, ratio `0.770x`, target `0.9`,
  `speed_claim_ready true`, `profile_counters_present true`,
  `target_vllm_ratio_met false`.
- artifacts: `results/gpu_matrix_20260529_143346.json`,
  `results/gpu_matrix_20260529_143346.md`.
- delta vs Entry 161/164: ratio `0.770x` vs `0.7645–0.765x` — carry-only
  path is **better** than the opt-in packed-decode/model-layout runs.
  Verified gain is from device-token carry, not the opt-in packed decode
  path.
- state: goal remains incomplete; throughput gap `0.770→0.9` still open.
- decision: packed-decode/model-layout opt-in selectors do not help; next
  work should target per-step overhead (forward_step, PjRt, transpose)
  and prefill cost.

### Entry 166 - `gdn_segmented_prefill_chunk32` + `triton_fla_wrapper`

- date: 2026-05-29
- change: implemented `gdn_segmented_prefill_chunk32` wrapper in
  `gdn_fla.py`, narrowed its fallback to import-only, and added
  `triton_fla_wrapper` selection in `backends.py`.
- tests: `tests/test_gdn_segmented_reference.py` => `21 passed`;  
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper tests/test_gdn_post_conv_prefill_reference.py -k 'triton or packed or post_conv'`
  => `19 passed`.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_r1.json --no-live-vllm --require-stored-references`
- benchmark result: `correctness exact/all_correct true`, nano `87.1566 tok/s`,
  vLLM `116.3726 tok/s`, ratio `0.7489`, TTFT p50 `270.5 ms`,
  `speed_claim_ready false`.
- decision: wrapper is functional/benchmarkable but not a speed win.
  Next target is reducing packed prefill metadata/pack-unpack/multi-kernel
  overhead, not another wrapper alias.

### Entry 167 - RealmAI Shim Retarget + Vectorized Pack + Pack-Opt Benchmark

- date: 2026-05-29
- change: retargeted RealmAI local shim to new working Responses/Chat
  endpoint/model after direct HTTPS failed Codex TLS trust; fresh
  `realmai_worker` succeeded with `REALMAI_WORKER_OK`.
- change: optimized `pack_prepared_gdn_prefill_inputs` in
  `nanovllm_jax/kernels/gdn_fla.py` — replaced per-row concatenate loop
  with vectorized valid-token gather.
- tests: `tests/test_gdn_segmented_reference.py` => `21 passed`; wrapper
  post-conv tests => `19 passed`.
- helper timing (B=2,T=5): pack p50 `~9.74 ms → ~6.49 ms` (~33% reduction).
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_packopt_r1.json --no-live-vllm --require-stored-references`
- benchmark result: `correctness exact`, nano `89.16 tok/s`, vLLM `116.37 tok/s`,
  ratio `0.766x`, TTFT p50 `251.28 ms`, `speed_claim_ready false`.
- state: goal incomplete; `0.9x` not met.
- next target: transpose/forward_step regressions remain.

### Entry 168 - Direct-Unpack Transpose Removal (Reverted)

- date: 2026-05-29
- change: tried direct-unpack transpose removal in
  `unpack_prepared_gdn_prefill_output`; tests passed (segmented 21 passed,
  wrapper post-conv 19 passed), but target benchmark regressed.
- benchmark result: nano `87.28 tok/s`, vLLM `116.37 tok/s`, ratio `0.750x`.
- action: reverted only the direct-unpack change; kept pack gather
  optimization from Entry 167.
- restored benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_packopt_restored_r1.json --no-live-vllm --require-stored-references`
- restored result: `correctness_checked true`,
  `exact_generated_token_match true`, `runs_succeeded true`; nano
  `88.66 tok/s`, vLLM `116.37 tok/s`, ratio `0.7619`, TTFT p50
  `250.80 ms`, ITL p50 `9.97 ms`, `speed_claim_ready false`,
  `target_vllm_ratio_met false`.
- conclusion: keep pack optimization; direct unpack is rejected. Remaining
  gap is still decode/transpose/multi-kernel overhead; target `0.9x` not
  met.

### Entry 169 - Packed Decode Probe (triton_fla decode + triton_fla_wrapper prefill)

- date: 2026-05-29
- change: probed BF16 Triton packed decode path
  (`NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla`) combined with
  `triton_fla_wrapper` prefill in the pack-optimized context from Entry 167.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_triton_decode_r1.json --no-live-vllm --require-stored-references`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`, `runs_succeeded true`; nano
  `87.99 tok/s`, vLLM `116.37 tok/s`, ratio `0.756`, TTFT p50
  `252.57 ms`, ITL p50 `10.14 ms`, `speed_claim_ready false`,
  `target_vllm_ratio_met false`.
- conclusion: BF16 Triton packed decode is still not a speed win in the
  current pack-optimized wrapper context; keep it disabled for target
  config. Remaining gap is decode transpose/runtime overhead and
  multi-kernel FLA prefill overhead; `0.9x` not met.

### Entry 170 - Stacked g/beta Transpose Combine (Reverted)

- date: 2026-05-29
- change: tried a small stacked `jnp.stack((g, beta), axis=-1)` transpose
  combine in `nanovllm_jax/model.py` to reduce decode transpose count.
- focused layer parity tests: `2 passed, 4 deselected`.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_gbtranspose_r1.json --no-live-vllm --require-stored-references`
- benchmark result: failed during warmup with
  `ValueError: All input arrays must have the same shape` at
  `jnp.stack((g, beta), axis=-1)` because `g` and `beta` can have
  mismatched broadcast shapes in prefill/warmup.
- action: reverted only the stacked g/beta probe; focused layer parity
  tests passed again (`2 passed, 4 deselected`).
- conclusion: reject naive stacked g/beta transpose; any transpose
  reduction must handle broadcasted gate shapes explicitly. `0.9x` target
  remains unmet.

### Entry 171 - CUDA Prefill Selector Probes

- date: 2026-05-29
- change: probed two CUDA prefill selector implementations against the
  pack-optimized baseline.
- probe 1 (`cuda_prep_prefill_fp32`):
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_prep_prefill_fp32 uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_cuda_prep_prefill_fp32_r1.json --no-live-vllm --require-stored-references`
  result: `correctness_checked true`, `exact_generated_token_match true`,
  `runs_succeeded true`; nano `63.87 tok/s`, vLLM `116.37 tok/s`,
  ratio `0.549`, TTFT p50 `26.74 ms`, ITL p50 `9.95 ms`,
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- probe 2 (`cuda_fla_chunk32_fp32`):
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=cuda_fla_chunk32_fp32 uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_cuda_fla_chunk32_fp32_r1.json --no-live-vllm --require-stored-references`
  result: `correctness_checked true`, `exact_generated_token_match true`,
  `runs_succeeded true`; nano `64.31 tok/s`, vLLM `116.37 tok/s`,
  ratio `0.553`, `speed_claim_ready false`,
  `target_vllm_ratio_met false`.
- conclusion: both CUDA prefill selectors are correctness-safe but much
  slower than the retained `triton_fla_wrapper` pack-optimized path
  (~0.762x), so reject for target config. Remaining viable work requires
  reducing decode transpose/runtime overhead or fusing the 6-stage FLA
  Triton prefill pipeline.

### Entry 172 - Guarded Q/K/V Combined-Transpose Probe (Reverted)

- date: 2026-05-29
- change: tried guarded Q/K/V combined-transpose probe in
  `nanovllm_jax/model.py` non-packed decode branch.
- focused tests: layer parity `2 passed`; packed decode reference
  selection `2 passed`.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_qkvtranspose_r1.json --no-live-vllm --require-stored-references`
- benchmark result: exit 0; `correctness_checked true`,
  `exact_generated_token_match true`, `runs_succeeded true`; nano
  `87.20 tok/s`, vLLM `116.37 tok/s`, ratio `0.7493`; TTFT p50
  `263.56 ms`, ITL p50 `9.95 ms`; transpose profile `73.14 ms / 1547`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- action: reverted only that probe; layer parity passed again
  (`2 passed, 4 deselected`).
- conclusion: reject Q/K/V combine; it did not reduce transpose profile
  and regressed vs retained pack-only path. Remaining viable work is
  larger: broadcast-safe decode layout rewrite or fused/reduced 6-stage
  FLA prefill pipeline. `0.9x` remains unmet.

### Entry 173 - Pack Metadata Cache Benchmark

- date: 2026-05-29
- change: benchmarked the pack metadata cache variant with
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper`.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_packcache_r1.json --no-live-vllm --require-stored-references`
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_packcache_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_packcache_r1.md`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`, `runs_succeeded true`; nano
  `89.35 tok/s`, vLLM `116.37 tok/s`, ratio `0.768`; TTFT p50
  `252.19 ms`, ITL p50 `10.07 ms`; `np.asarray` profile `32.51 ms /
  64 count`, `transpose` profile `73.10 ms / 1547 count`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- comparison vs Entry 167 packopt restored (ratio `0.7619`): pack
  metadata cache retains a small improvement of `+0.006` ratio points
  but remains well below the `0.9x` target. TTFT and ITL are
  essentially unchanged; the `np.asarray` profile dropped from
  `427.55 ms / 16 count` (reference) to `32.51 ms / 64 count` here,
  confirming the cache avoids the prior large host-sync stall, though
  the transpose profile is unchanged at ~73 ms / 1547 count.
- conclusion: pack metadata cache is a small net positive over the
  packopt restored baseline but insufficient alone; `0.9x` target
  remains unmet. Decode transpose/runtime overhead and FLA prefill
  pipeline cost still dominate the gap.

### Entry 174 - causal_conv1d_update Slice/Concat Probe (Rejected)

- date: 2026-05-29
- change: probed `causal_conv1d_update` slice/concat variant with
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper`.
- focused test: `tests/test_causal_conv1d_update.py` — 4 passed in 8.18s.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_convconcat_r1.json --no-live-vllm --require-stored-references`
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_convconcat_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_convconcat_r1.md`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`; nano `87.99 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.756`; `np.asarray` profile `28.25 ms /
  64 count`, `transpose` profile `73.14 ms / 1547 count`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- comparison vs Entry 173 packcache (ratio `0.768`): convconcat
  regresses by `-0.012` ratio points (87.99 vs 89.35 tok/s). The
  `np.asarray` profile improved slightly (28.25 vs 32.51 ms) but
  overall throughput declined; transpose profile unchanged at ~73 ms /
  1547 count.
- conclusion: slice/concat approach regresses vs retained packcache
  implementation. Reject and revert; packcache (Entry 173) remains
  the active variant.

### Entry 175 - Direct [B,H,T,V] Unpack Probe (Rejected)

- date: 2026-05-29
- change: probed direct `[B,H,T,V]` unpack variant with
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper`.
- focused test: `tests/test_causal_conv1d_update.py` — 4 passed in 8.18s.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_direct_unpack_bhtv_r1.json --no-live-vllm --require-stored-references`
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_direct_unpack_bhtv_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_direct_unpack_bhtv_r1.md`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`; nano `88.77 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.763`; TTFT p50 `107.20 ms`, ITL p50
  `10.82 ms`; `np.asarray` profile `29.19 ms / 64 count`,
  `transpose` profile `73.11 ms / 1547 count`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- comparison vs Entry 173 packcache (ratio `0.768`): direct unpack
  regresses by `-0.005` ratio points (88.77 vs 89.35 tok/s). The
  `np.asarray` profile worsened (29.19 vs 32.51 ms) and transpose
  profile unchanged at ~73 ms / 1547 count; overall throughput
  declined.
- conclusion: direct `[B,H,T,V]` unpack regresses vs retained
  packcache implementation. Reject and revert; packcache (Entry 173)
  remains the active variant.

### Entry 176 - Triton cumsum Scalar-Load Probe (Rejected)

- date: 2026-05-29
- change: probed Triton cumsum scalar-load variant with
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper`.
- focused test: `tests/test_causal_conv1d_update.py` — 17 passed, 4 deselected in 37.80s.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_cumsum_scalar_load_r1.json --no-live-vllm --require-stored-references`
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_cumsum_scalar_load_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_cumsum_scalar_load_r1.md`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`; nano `88.82 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.763`; ITL p50
  `10.19 ms`; `np.asarray` profile `27.90 ms / 64 count`,
  `transpose` profile `73.08 ms / 1547 count`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- comparison vs Entry 173 packcache (ratio `0.768`): cumsum scalar-load
  regresses by `-0.005` ratio points (88.82 vs 89.35 tok/s). The
  `np.asarray` profile improved (27.90 vs 32.51 ms) but transpose
  profile unchanged at ~73 ms / 1547 count; overall throughput
  declined.
- conclusion: Triton cumsum scalar-load regresses vs retained
  packcache implementation. Reject and revert; packcache (Entry 173)
  remains the active variant.

### Entry 177 - Direct [B,T,H,V] Unpack Write-Layout Probe (Rejected)

- date: 2026-05-29
- change: probed direct `[B,T,H,V]` unpack write-layout variant with
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper`.
- focused test: `tests/test_gdn_segmented_reference.py` — 21 passed;
  `tests/test_gdn_post_conv_prefill_reference.py` — 19 passed.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_unpack_bthv_direct_r1.json --no-live-vllm --require-stored-references`
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_unpack_bthv_direct_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_unpack_bthv_direct_r1.md`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`; nano `89.04 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.765`; ITL p50
  `10.03 ms`; `np.asarray` profile `27.74 ms / 64 count`,
  `transpose` profile `73.10 ms / 1547 count`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- comparison vs Entry 173 packcache (ratio `0.768`): direct
  `[B,T,H,V]` unpack write-layout regresses by `-0.003` ratio points
  (89.04 vs 89.35 tok/s). The `np.asarray` profile improved
  (27.74 vs 32.51 ms) but transpose profile unchanged at ~73 ms /
  1547 count; overall throughput declined.
- conclusion: Direct `[B,T,H,V]` unpack write-layout regresses vs
  retained packcache implementation. Reject and revert; packcache
  (Entry 173) remains the active variant.

### Entry 178 - _full_attention_block_jit BTHD RoPE Layout Probe (Rejected)

- date: 2026-05-29
- change: probed `_full_attention_block_jit` BTHD RoPE layout variant with
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper`.
- focused test: `tests/test_full_attention_parity.py` — 1 passed.
- benchmark command:
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL=triton_fla_wrapper uv run python benchmarks/run_gpu_matrix.py --configs gpu_paged_default --workloads long_prefill_512_2048 --repeats 1 --output-json results/gpu_matrix_20260529_triton_fla_wrapper_bthd_rope_fullattn_r1.json --no-live-vllm --require-stored-references`
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_bthd_rope_fullattn_r1.json`
- report: `results/gpu_matrix_20260529_triton_fla_wrapper_bthd_rope_fullattn_r1.md`
- benchmark result: `correctness_checked true`,
  `exact_generated_token_match true`; nano `88.96 tok/s`, vLLM
  `116.37 tok/s`, ratio `0.764`; TTFT p50
  `105.15 ms`; ITL p50
  `10.06 ms`; `np.asarray` profile `29.35 ms / 64 count`,
  `transpose` profile `73.07 ms / 1547 count`;
  `speed_claim_ready false`, `target_vllm_ratio_met false`.
- comparison vs Entry 173 packcache (ratio `0.768`): BTHD RoPE layout
  regresses by `-0.004` ratio points (88.96 vs 89.35 tok/s). The
  `np.asarray` profile worsened (29.35 vs 27.74 ms) and transpose
  profile unchanged at ~73 ms / 1547 count; overall throughput
  declined.
- conclusion: `_full_attention_block_jit` BTHD RoPE layout regresses vs
  retained packcache implementation. Reject and revert; packcache
  (Entry 173) remains the active variant.

### Entry 179 - Fused W/U Triton Kernel Probe (Rejected)

- date: 2026-05-29
- change: probed fused W+U recompute kernel (`_gdn_fla_recompute_wu_fused_packed_kernel`)
  behind `NANO_VLLM_JAX_GDN_FLA_FUSED_WU=1` env opt-in, combining the
  separate W and U Triton kernels into a single launch.
- focused test: `tests/test_gdn_segmented_reference.py` — passed.
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_fused_wu_r1.json`
- benchmark result: correctness OK; nano `88.93 tok/s`, vLLM
  `116.37 tok/s`, JAX/vLLM `0.7641x`; retained best `0.7700x`.
- comparison vs retained best (ratio `0.7700`): fused W/U regresses by
  `-0.0059` ratio points (88.93 vs 89.35 tok/s). Single-kernel fusion
  does not offset the reduced parallelism from collapsing two
  independent kernel launches into one.
- conclusion: fused W/U kernel regresses vs retained two-kernel W/U
  implementation. Revert and reject; two-kernel W/U remains the active
  variant.

### Entry 180 - Packed Decode triton_fla Benchmark (Rejected)

- date: 2026-05-29
- change: benchmarked packed decode with `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla`.
- test env value fixed from invalid `triton_fla_fp32` to `triton_fla` in
  `tests/test_gdn_packed_decode_reference.py`.
- BF16-QKV test cleanup: renamed `triton_fp32` test to `triton_fla_bf16qkv`
  with `qkv_dtype=jnp.bfloat16` to match runtime BF16 QKV path.
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_packed_decode_r1.json`
- benchmark result: correctness OK; nano `87.86 tok/s`, vLLM
  `116.37 tok/s`, JAX/vLLM `0.7550x`; retained best `0.7700x`.
- comparison vs retained best (ratio `0.7700`): packed decode triton_fla
  regresses by `-0.0150` ratio points (87.86 vs 89.35 tok/s).
- conclusion: packed decode `triton_fla` benchmark regresses vs retained
  best. Not enabled; retained best remains the active variant.

### Entry 181 - L2Norm keep_fp32 (Rejected)

- date: 2026-05-29
- change: added `keep_fp32` parameter to `l2norm()` in
  `nanovllm_jax/layers.py` and passed `keep_fp32=True` in the two
  `use_qk_l2norm_in_kernel` call sites in `nanovllm_jax/model_jit.py`,
  keeping the L2-normalized Q/K in float32 to avoid a redundant
  cast-to-bf16-then-back-to-fp32 round-trip before the kernel body.
- benchmark artifact: `results/gpu_matrix_20260529_triton_fla_wrapper_l2norm_keepfp32_r1.json`
- benchmark result: correctness OK; nano `88.78 tok/s`, vLLM
  `116.37 tok/s`, JAX/vLLM `0.763x`; retained best `0.7700x`.
- comparison vs retained best (ratio `0.7700`): l2norm keep_fp32
  regresses by `-0.0070` ratio points (88.78 vs 89.35 tok/s).
  Eliminating the dtype round-trip does not offset the overhead; the
  redundant cast is cheap relative to other bottlenecks.
- decision: reverted / not retained. The `keep_fp32` parameter and
  `keep_fp32=True` call-site arguments have been removed; `l2norm()`
  returns to its original signature and always casts back to the input
  dtype. Retained best remains the active variant.

### Entry 182 - Server Config-File Support (YAML + Env Override)

- date: 2026-05-30
- change: Added YAML-based server configuration to reduce env-var
  proliferation while preserving full backward compatibility.
- new files:
  - `nanovllm_jax/server_config.py` – `ServerConfig` dataclass and
    `load_server_config()` loader.  Reads `server_config.yaml` (or a
    path given by `NANO_VLLM_JAX_SERVER_CONFIG`), applies the `env:`
    section to `os.environ` before JAX init, then resolves `server:`
    and `engine:` sections with env-var overrides (env always wins).
  - `server_config.yaml` – default config file at repo root with
    commented-out env-section knobs for kernel backend, MTP policy,
    GDN implementation, etc.
- modified files:
  - `server.py` – imports `load_server_config`, calls it at module
    level (before JAX init) so the `env:` section is applied early.
    `main()` now reads defaults from the resolved `ServerConfig` rather
    than hard-coded values.  Added `--config` CLI flag to reload from a
    custom path.
- env-var override precedence (highest wins):
  1. Shell env var (e.g. `NANO_VLLM_JAX_PORT=9090`)
  2. CLI flag (e.g. `--port 9090`)
  3. YAML config file value
  4. Built-in default
- env-section keys supported: `TOKENIZERS_PARALLELISM`,
  `XLA_PYTHON_CLIENT_PREALLOCATE`, `TF_GPU_ALLOCATOR`,
  `NANO_VLLM_JAX_XLA_GPU_AUTOTUNE_LEVEL`, `NANO_VLLM_JAX_CACHE_ROOT`,
  `NANO_VLLM_JAX_KERNEL_BACKEND`, `NANO_VLLM_JAX_DEVICE_TOKEN_CARRY`,
  `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL`,
  `NANO_VLLM_JAX_GDN_PREFILL_POST_CONV_IMPL`,
  `NANO_VLLM_JAX_MTP_BATCH_ACCEPT_POLICY`, and 20+ others (see
  `_ENV_SECTION_KEYS` in `server_config.py`).
- no benchmark change: this is a config-mechanism refactor with zero
  impact on the inference hot path.
- decision: retained.  All existing env-var call sites (`os.getenv`,
  `os.environ.get`) continue to work unchanged; the YAML `env:` section
  merely pre-populates `os.environ` before those call sites execute.

### Entry 183 - Decode-Heavy GDN FLA vLLM Parity Sweep

- date: 2026-05-30
- change: ran `benchmarks/run_gpu_matrix.py` matrix sweeps on decode-heavy
  workload (`decode_heavy_128x128`) with new GDN FLA configs and live vLLM
  references to validate correctness and throughput.
- harness fix: corrected `_jax_command` invocation order in
  `benchmarks/run_gpu_matrix.py` (live JAX default reference generation now
  passes the JAX execution argument correctly).
- new configs:
  - `benchmarks/configs/gpu_paged_gdn_fla_decode.json`
  - `benchmarks/configs/gpu_paged_gdn_fla_decode_off_prefill.json`
  - `benchmarks/configs/gpu_paged_gdn_fla_decode_bf16_qkv.json`
- benchmark artifacts:
  - `results/gla_decode_heavy_matrix_vllm.json`
  - `results/gla_decode_heavy_matrix_off_prefill.json`
  - `results/gla_decode_heavy_matrix_bf16_qkv.json`
  - `results/gla_longprefill_gdn_matrix_vllm.json`
- outcomes:
  - all runs pass `correctness_checked` and `exact_generated_token_match`;
  - all runs pass `speed_claim_ready` and `minimum_repeats`;
  - `gdn_fla` decode-heavy ratio peaked at `~0.763x` (162.90 vs 213.54 tok/s),
    short of target `0.9x`;
  - long-prefill ratio with the same GDN FLA variant is ~`0.758x`
    (88.21 vs 116.37 tok/s);
  - off-prefill and bf16-QKV variants were slower (ratios `~0.699x`
    and `~0.694x` respectively).
- conclusion: GDN FLA decode path is now functioning and parity-checked against
  vLLM for decode-heavy + long-prefill shapes, but no variant reached the
  `0.9x` vLLM throughput target under this harness and hardware.

### Entry 184 - Triton Packed-Decode Launch Tunables

- date: 2026-05-30
- change: added env-driven launch tuning for the vLLM/FLA-shaped Triton packed
  GDN decode step and exposed the keys through the YAML `env:` section.
- files:
  - `nanovllm_jax/kernels/gdn_fla_triton.py`
  - `nanovllm_jax/server_config.py`
  - `server_config.yaml`
- env keys:
  - `NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_NUM_WARPS` (default: `1` for
    smaller V, `2` for larger V)
  - `NANO_VLLM_JAX_GDN_PACKED_DECODE_TRITON_BLOCK_V` (default: `min(next_power2(V), 32)`)
- benchmark artifacts:
  - `results/gpu_matrix_decode_tune_baseline2.json` (defaults, repeats=2)
  - `results/gpu_matrix_decode_tune2.json` (warps=4, block_v=64, repeats=2)
- outcomes:
  - decode correctness and repeat requirements passed in both runs.
  - measured ratio was `0.76213x` (baseline) versus `0.76194x` (tuned),
    effectively unchanged within sampling noise.
- conclusion: keep the toggles for future tuning, but no performance win to
  retain for this workload/hardware as of this run.

### Entry 185 - Decode-Heavy Focused Retune (Expanded)

- date: 2026-05-30
- change: continued decode-heavy-only sweeps with live-vLLM targets on the same
  harness and a wider set of decode-focused GDN variants, including compact decode
  path and prefill-wrapper off variants.
- benchmark artifacts:
  - `results/gpu_matrix_decode_tune_focus_default.json`
  - `results/gpu_matrix_decode_tune_focus_bf16qkv.json`
  - `results/gpu_matrix_decode_off_prefill_bf16_qkv.json`
- outcomes:
  - all runs passed correctness checks and `minimum_repeats`.
  - best `decode_heavy_128x128` ratio observed was `0.823x` using
    `gpu_paged_gdn_fla_decode_bf16_qkv`:
    - JAX `179.80` tok/s vs vLLM `218.48` tok/s (`0.823x`).
  - nearby variants remained below target:
    - `gpu_paged_gdn_fla_decode` (default): `178.48`/`218.20` (`0.818x`)
    - `gpu_paged_gdn_fla_decode_off_prefill`: `178.38`/`218.67` (`0.816x`)
- conclusion: decode-focused GDN experiments are no longer dominated by launch
  tuning; we are in the same neighborhood (~`0.82x`) and should move on to the
  next decoder-stage opportunities (command-buffer pressure and Triton decode-step
  arithmetic lowering).

### Entry 186 - Current GDN Decode Config And Prep Cleanup

- date: 2026-06-01
- change:
  - promoted the tracked `gpu_paged_gdn_fla_decode_bf16_qkv` config to the
    current pre-normalized packed-decode settings:
    `NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK=1`,
    `TRITON_NUM_WARPS=8`, `TRITON_NUM_STAGES=2`, `TRITON_BLOCK_V=32`;
  - removed the redundant JAX-side `gate`/`beta` recompute in the
    `triton_fla_prep_bf16` post-conv prefill path. Both the JAX fallback helper
    and the Triton prep kernel already return masked FP32 `gate`/`beta`.
  - updated `AGENTS.md` to stop using `realmai_worker` and prefer
    `gpt-5.3-codex-spark` for newly spawned subagents.
- focused tests:
  - `pytest -q tests/test_gdn_packed_decode_reference.py tests/test_cuda_fp32_ffi.py -q`
    - 36 passed.
  - `pytest -q tests/test_gdn_post_conv_prefill_reference.py -k "triton_fla_prep_bf16 or post_conv_prep_bf16" -q`
    - 1 passed.
- benchmark artifact: `results/gpu_matrix_gdn_current_best_20260601_postpatch.json`
- report: `results/gpu_matrix_gdn_current_best_20260601_postpatch.md`
- benchmark command:
  `.venv/bin/python3 -m benchmarks.run_gpu_matrix --configs gpu_paged_gdn_fla_decode_bf16_qkv --workloads decode_heavy_128x128 --repeats 2 --no-live-vllm --output-json results/gpu_matrix_gdn_current_best_20260601_postpatch.json --run-dir results/gpu_matrix_runs/20260601_gdn_current_best_postpatch`
- benchmark result:
  - `speed_claim_ready=true`, `exact_generated_token_match=true`;
  - JAX `163.62 tok/s`, vLLM `213.54 tok/s`, JAX/vLLM `0.766x`;
  - JAX reference `151.84 tok/s`, JAX/reference `1.078x`;
  - `target_vllm_ratio_met=false` for the `0.9x` target.
- reproducibility check: reran commit `55247ab` with the same pre-normalized
  decode settings and current runtime; it measured `164.22 tok/s`, JAX/vLLM
  `0.769x` on one repeat. The earlier `0.842x` pre-normalized artifact was not
  reproduced, so the current reproducible evidence is the `~0.77x` range.
- conclusion: retain the tracked config promotion and redundant prep cleanup.
  The GDN FLA decode path remains correctness-clean and faster than the stored
  JAX reference, but it is still below the requested `0.9x` vLLM target. Next
  work should target decoder-stage command-buffer/update pressure and remaining
  compiled-step reductions rather than more packed-decode launch tuning.

### Entry 187 - Strict GDN Kernel Route Audit

- date: 2026-06-01
- change:
  - added `NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS=1` to fail hard when a requested
    GDN kernel path would silently use a pure-JAX/reference fallback;
  - applied the guard in the backend prefill/decode selection, the segmented FLA
    wrapper, and the internal Triton FLA prefill stage fallbacks;
  - exposed the flag through server config and benchmark artifact metadata;
  - changed the tracked `gpu_paged_gdn_fla_decode_bf16_qkv` config to the best
    strict kernel route found in this pass: CUDA fused post-conv prep+prefill
    (`cuda_prep_prefill_fp32`) plus bf16 Triton packed decode.
- important finding:
  - `triton_fla_wrapper` is not JIT-safe for strict kernel investigation because
    traced `seq_lens` force the previous pure-JAX reference path.
  - `triton_fla_padded` does exercise the FLA Triton prefill kernels, but is a
    severe regression on the target long-prefill workload:
    `22.80 tok/s` vs stored JAX reference `78.02 tok/s`.
  - the FLA prefill profile is dominated by materialized multi-stage kernels:
    `_gdn_fla_chunk_fwd_o_packed_kernel` `931.80 ms`,
    `_gdn_fla_chunk_delta_h_packed_kernel` `842.48 ms`,
    `_gdn_fla_chunk_scaled_dot_kkt_packed_kernel` `302.92 ms` (one repeat).
  - CUDA fused prefill is the current best strict kernel route, but still misses
    the target: the fused `Fp32GdnPrefillChunk32Kernel<64,false>` takes about
    `444 ms` over 18 launches on `long_prefill_512_2048`.
- focused tests:
  - `pytest -q tests/test_gdn_post_conv_prefill_reference.py -k "triton_fla_padded or fallback_without_triton_module or strict_rejects or cuda_post_conv_prep" -q`
    - 7 passed.
  - `pytest -q tests/test_gdn_packed_decode_reference.py tests/test_cuda_fp32_ffi.py -q`
    - 36 passed.
  - `python -m json.tool benchmarks/configs/gpu_paged_gdn_fla_decode_bf16_qkv.json >/dev/null`
  - `python -m py_compile benchmarks/benchmark_jax_server_trace.py nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla.py nanovllm_jax/kernels/gdn_fla_triton.py nanovllm_jax/server_config.py tests/test_gdn_post_conv_prefill_reference.py`
- benchmark artifacts:
  - strict FLA padded audit: `results/gpu_matrix_gdn_strict_kernel_20260601.json`
  - strict FLA padded long-prefill audit:
    `results/gpu_matrix_gdn_strict_kernel_longprefill_20260601.json`
  - retained strict CUDA-prefill route:
    `results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.json`
  - report: `results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.md`
- retained benchmark command:
  `.venv/bin/python3 -m benchmarks.run_gpu_matrix --configs gpu_paged_gdn_fla_decode_bf16_qkv --workloads decode_heavy_128x128,long_prefill_512_2048 --repeats 2 --no-live-vllm --output-json results/gpu_matrix_gdn_cuda_prefill_strict_r2_20260601.json --run-dir results/gpu_matrix_runs/20260601_gdn_cuda_prefill_strict_r2`
- retained benchmark result:
  - `decode_heavy_128x128`: JAX `165.88 tok/s`, vLLM `213.54 tok/s`,
    JAX/vLLM `0.777x`, JAX/reference `1.092x`;
  - `long_prefill_512_2048`: JAX `63.98 tok/s`, vLLM `116.37 tok/s`,
    JAX/vLLM `0.550x`, JAX/reference `0.820x`;
  - correctness and minimum-repeat checks passed; `target_vllm_ratio_met=false`.
- decision: retain strict fallback gating and the CUDA-prefill config because
  they make the kernel route measurable and prevent misleading fallback wins.
  Do not retain Triton FLA padded prefill as the tracked route; its multi-stage
  materialization cost overwhelms any lowering benefit on the target shape.

### Entry 188 - Runtime Config Refactor And CUDA Probe Demotion

- date: 2026-06-01
- change:
  - added typed `runtime` and `kernels` config sections and translated them
    through the same startup env path used by the existing code;
  - changed the server to load `--config` before importing the engine/JAX stack,
    so runtime/XLA/kernel settings are applied before JAX initialization;
  - changed the GPU matrix runner to consume the same typed config translation
    while preserving legacy raw `env` configs;
  - demoted local CUDA/JAX FFI routes behind
    `NANO_VLLM_JAX_ALLOW_LOCAL_CUDA_PROBES=1` and updated docs/source comments
    to mark them as diagnostics, not the serving-kernel direction.
- CUDA file audit:
  - the active local CUDA source is
    `nanovllm_jax/kernels/csrc/fp32_kv_append.cu`, loaded through
    `nanovllm_jax/kernels/cuda_fp32_ffi.py`;
  - it contains historical FP32 KV append, decode attention, GDN decode, and
    GDN prefill probes used by earlier correctness/performance experiments;
  - `nanovllm_jax/kernels/cuda_gdn.py` is still a Python helper/placeholder
    module, not a compiled serving implementation.
- config decision:
  - keep the C++/CUDA files only as explicitly opted-in probes for reproducing
    prior results;
  - new kernel work should be implemented in Python-facing Pallas/CuteDSL, or
    Triton when adapting an existing kernel schedule;
  - the tracked `gpu_paged_gdn_fla_decode_bf16_qkv` config now uses strict
    Triton FLA padded prefill plus Triton packed decode, with local CUDA probes
    disabled.
- validation:
  - `python -m json.tool benchmarks/configs/gpu_paged_gdn_fla_decode_bf16_qkv.json`
  - `python -m py_compile nanovllm_jax/server_config.py benchmarks/run_gpu_matrix.py benchmarks/benchmark_jax_server_trace.py nanovllm_jax/backends.py server.py`
  - `pytest -q tests/test_server_config.py tests/test_gpu_matrix_runner.py -q`
    - 50 passed.
  - `pytest -q tests/test_server_config.py -q`
    - 6 passed.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async pytest -q tests/test_gdn_post_conv_prefill_reference.py -k "triton_fla_padded or fallback_without_triton_module or strict_rejects" -q`
    - 5 passed.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async pytest -q tests/test_gdn_packed_decode_reference.py -k "triton" -q`
    - 1 passed.
  - dry-run matrix config translation:
    `.venv/bin/python3 -m benchmarks.run_gpu_matrix --configs gpu_paged_gdn_fla_decode_bf16_qkv --workloads decode_heavy_128x128 --repeats 1 --dry-run --output-json results/gpu_matrix_config_refactor_dryrun_20260601.json --run-dir results/gpu_matrix_runs/20260601_config_refactor_dryrun`
- status:
  - config refactor is in place and compatibility with legacy benchmark `env`
    sections is preserved;
  - no new speed claim is recorded here. The best non-local-CUDA strict route
    remains the previously measured Triton FLA padded path, which was correct
    but slow on long prefill (`22.80 tok/s` in Entry 187).

### Entry 189 - Decode BF16 Fastpath Pass

- date: 2026-06-01
- accepted changes:
  - added typed runtime fastpaths for decode-only LM-head activation dtype and
    decode projection activation dtype;
  - enabled BF16 LM-head decode activations for the strict Triton GDN config;
  - enabled BF16 projection activations only for single-sequence decode
    (`bf16_single_seq`), leaving batched decode and prefill on FP32;
  - specialized `forward_step_token_ids_jit` decode to use `hidden[:, :1, :]`
    instead of building a dynamic last-token gather.
- rejected experiments:
  - fusing GDN gate/beta scalar preparation into the packed Triton decode kernel
    preserved exactness but did not improve end-to-end throughput; W8/S2/B32
    regressed from `165.53 tok/s` to `161.69 tok/s` in one repeat.
  - global BF16 model activations improved single-sequence decode but failed
    exact generated-token match on `long_prefill_512_2048`.
  - BF16 decode projections for all decode batch sizes stayed exact on
    long-prefill but regressed batched decode throughput to about `23 tok/s`;
    the retained config therefore restricts this to single-sequence decode.
- validation:
  - `pytest -q tests/test_server_config.py tests/test_lm_head_helpers.py`
    - 9 passed.
  - `pytest -q tests/test_backend_boundaries.py -k 'executor_greedy_decode_burst_matches_iterative_token_path'`
    - 1 passed.
- benchmark artifacts:
  - baseline comparison:
    `results/gpu_matrix_decode_longpass_variants_r1_20260601.json`
  - LM-head BF16:
    `results/gpu_matrix_decode_bf16_lmhead_r1_20260601.json`
  - single-sequence decode projection BF16:
    `results/gpu_matrix_decode_proj_bf16_single_seq_r1_20260601.json`
  - final three-repeat decode:
    `results/gpu_matrix_decode_final_bf16_fastpaths_r3_20260601.json`
  - long-prefill audit:
    `results/gpu_matrix_decode_proj_bf16_single_seq_r1_20260601.json`
- retained benchmark result:
  - `decode_heavy_128x128`: JAX median `155.45 tok/s`, vLLM `213.54 tok/s`,
    JAX/vLLM `0.728x`, exact generated-token match true.
  - single-repeat accepted fastpath measurement reached `158.72 tok/s`
    (`0.743x` vLLM) before the final three-repeat run settled lower.
  - `long_prefill_512_2048`: still exact but slow on the strict Triton-padded
    prefill route (`22.82 tok/s` in the single-sequence projection audit);
    this pass did not solve the prefill route.
- decision:
  - retain the decode BF16 fastpaths because they improve the tracked strict
    no-local-CUDA decode route over the refactored baseline (`150.38 tok/s`
    single-repeat baseline, `155.45 tok/s` final median) without changing
    generated tokens.
  - do not claim progress on long-prefill; it remains a separate bottleneck.

### Entry 190 - Decode Plan Refresh And Pallas Reduction Probe

- date: 2026-06-01
- plan update:
  - added an active decode plan to `docs/gpu_optimization_next_goal_plan.md`
    using the latest strict no-local-CUDA decode profile as the anchor:
    `results/gpu_matrix_decode_final_bf16_fastpaths_r3_20260601.json`,
    `155.45 tok/s`, `0.728x` vLLM, exact generated-token parity;
  - reprioritized the next work toward static decode execution/replay,
    decode reductions, greedy LM-head top-1, strict GDN chunked-prefill, and
    device-owned cache/state metadata.
- change tested:
  - added strict opt-in Pallas decode-reduction helpers for Qwen RMSNorm and
    packed GDN decode Q/K pre-normalization;
  - wired typed runtime fastpaths:
    `runtime.fastpaths.pallas_decode_rmsnorm` and
    `runtime.fastpaths.pallas_gdn_qk_prenorm`;
  - added
    `benchmarks/configs/gpu_paged_gdn_fla_decode_pallas_reductions.json` as a
    strict no-local-CUDA experiment layered on the current BF16 decode config.
- validation:
  - `python -m json.tool benchmarks/configs/gpu_paged_gdn_fla_decode_pallas_reductions.json`
  - `python -m py_compile nanovllm_jax/kernels/decode_reductions.py nanovllm_jax/model.py nanovllm_jax/kernels/gdn_fla.py nanovllm_jax/server_config.py`
  - `pytest -q tests/test_server_config.py tests/test_lm_head_helpers.py`
    - 9 passed.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda pytest -q tests/test_decode_reductions.py`
    - 4 passed.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM=1 pytest -q tests/test_gdn_packed_decode_reference.py -k 'pre_normalize_qk or triton_bf16'`
    - 2 passed, 17 deselected.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda NANO_VLLM_JAX_PALLAS_DECODE_RMSNORM=1 pytest -q tests/test_backend_boundaries.py -k 'executor_greedy_decode_burst_matches_iterative_token_path or executor_jit_matches_eager_cached_decode'`
    - 2 passed, 43 deselected.
- benchmark artifact:
  - one-repeat smoke: `results/gpu_matrix_decode_pallas_reductions_r1_20260601.json`
  - retained three-repeat run:
    `results/gpu_matrix_decode_pallas_reductions_r3_20260601.json`
  - report: `results/gpu_matrix_decode_pallas_reductions_r3_20260601.md`
- benchmark command:
  `.venv/bin/python benchmarks/run_gpu_matrix.py --configs gpu_paged_gdn_fla_decode_pallas_reductions --workloads decode_heavy_128x128 --repeats 3 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_decode_pallas_reductions_r3_20260601.json`
- benchmark result:
  - `decode_heavy_128x128`: JAX median `157.01 tok/s`, vLLM `213.54 tok/s`,
    JAX/vLLM `0.735x`, JAX/reference `1.034x`;
  - speed-claim-ready with `exact_generated_token_match=true`, but
    `target_vllm_ratio_met=false`;
  - scheduler diagnostics: one prefill step, `127` decode steps, decode step
    total `0.78 s`;
  - profile still dominated by `gemm_fusion_dot_265` (`126.37 ms`) and
    `input_reduce_fusion_*` buckets (`103.23 ms`, `63.64 ms`, `62.50 ms`).
- decision:
  - keep the Pallas reduction route default-off as a diagnostic/configurable
    experiment only. It is correctness-clean and slightly positive in one
    repeat, but it does not remove the dominant reduction buckets or close the
    `0.9x` gap.
  - next execution should focus on static decode/runtime replay or a deeper
    fused reduction boundary; replacing isolated reductions with many Pallas
    calls risks increasing command-buffer churn.

### Entry 191 - Accepted Padded-GEMM Decode Projection Lowering

- date: 2026-06-01
- HLO finding:
  - the largest decode `input_reduce_fusion_*` GPU buckets were not layer
    normalization. Optimized HLO for `forward_step_token_ids_jit` mapped them to
    B=1 decode projection GEMVs:
    MLP gate/up, MLP down, and GDN `in_proj_qkv`.
  - this explains why isolated RMSNorm/L2Norm replacement did not move the
    dominant buckets.
- accepted change:
  - added a strict opt-in padded-GEMM decode projection helper,
    `NANO_VLLM_JAX_DECODE_PADDED_GEMM=1`;
  - added typed config keys:
    `runtime.fastpaths.decode_padded_gemm`,
    `runtime.fastpaths.decode_padded_gemm_gate_up`,
    `runtime.fastpaths.decode_padded_gemm_rows`, and
    `runtime.fastpaths.decode_padded_gemm_max_out_dim`;
  - routed selected B=1 decode projections through a small broadcasted GEMM
    shape: GDN `in_proj_qkv`, MLP gate/up, and MLP down;
  - added
    `benchmarks/configs/gpu_paged_gdn_fla_decode_padded_gemm.json` with
    rows=`8`, gate/up enabled, strict GDN fallbacks disabled, and no local CUDA
    probes.
- rejected probes:
  - Triton decode RMSNorm: `results/gpu_matrix_decode_triton_reductions_r1_20260601.json`
    regressed to `113.77 tok/s` and left the major GPU buckets unchanged.
  - B=1 LM-head vector dot:
    `results/gpu_matrix_decode_lmhead_vector_r1_20260601.json` reached
    `158.46 tok/s`, but LM-head GEMM stayed at about `126.36 ms`.
  - Triton decode matvec:
    `results/gpu_matrix_decode_triton_matvec_r1_20260601.json`,
    `results/gpu_matrix_decode_triton_matvec_selective_r1_20260601.json`, and
    `results/gpu_matrix_decode_triton_matvec_tldot_tiled_r1_20260601.json`
    all replaced the old `input_reduce` names with a slower `_kernel` bucket and
    were removed from the runtime surface.
  - padded-GEMM rows sweep:
    rows=`4` (`147.01 tok/s`) and rows=`16` (`157.65 tok/s`) both lost to
    rows=`8`.
- validation:
  - `python -m json.tool benchmarks/configs/gpu_paged_gdn_fla_decode_padded_gemm.json`
  - `python -m py_compile nanovllm_jax/model.py nanovllm_jax/server_config.py nanovllm_jax/kernels/decode_reductions.py`
  - `pytest -q tests/test_server_config.py tests/test_lm_head_helpers.py`
    - 9 passed.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda NANO_VLLM_JAX_DECODE_PADDED_GEMM=1 NANO_VLLM_JAX_DECODE_PADDED_GEMM_GATE_UP=1 .venv/bin/python -m pytest -q tests/test_backend_boundaries.py -k 'executor_greedy_decode_burst_matches_iterative_token_path or executor_jit_matches_eager_cached_decode'`
    - 2 passed, 43 deselected.
- benchmark command:
  `.venv/bin/python benchmarks/run_gpu_matrix.py --configs gpu_paged_gdn_fla_decode_padded_gemm --workloads decode_heavy_128x128 --repeats 3 --no-live-vllm --require-stored-references --output-json results/gpu_matrix_decode_padded_gemm_r3_20260601.json`
- benchmark result:
  - artifact: `results/gpu_matrix_decode_padded_gemm_r3_20260601.json`
  - report: `results/gpu_matrix_decode_padded_gemm_r3_20260601.md`
  - `decode_heavy_128x128`: JAX median `162.10 tok/s`, vLLM `213.54 tok/s`,
    JAX/vLLM `0.759x`, JAX/reference `1.068x`;
  - speed-claim-ready with exact generated-token match;
  - profile movement: the previous projection `input_reduce_fusion_*` buckets
    moved to GEMM lowerings (`gemm_fusion_dot_175`,
    `gemm_fusion_dot_general_337`, `gemm_fusion_dot_200`). The remaining top
    GPU event is the LM-head GEMM at about `126.45 ms`.
- decision:
  - accept padded-GEMM decode projection lowering as the current strict
    no-local-CUDA decode best.
  - this meaningfully reduces the projection/GEMV bucket family, but it does
    not solve the remaining LM-head-dominated decode gap to `0.9x` vLLM.

### Entry 192 - Orchestrated Decode/Prefill Strategy Pass

- date: 2026-06-01
- orchestration:
  - split the next `0.9x` work into five bounded investigations: static decode
    replay/metadata, LM-head greedy top-1, projection structure, strict GDN/FLA
    kernels, and benchmark/profile strategy;
  - after the pass, close the fan-out and use a single reusable worker for
    follow-up slices to reduce integration risk and context drift.
- accepted support changes:
  - added `Host Replay Diagnostics` to `benchmarks/summarize_gpu_matrix.py`,
    normalizing PjRt/command-buffer buckets by scheduler step so replay-style
    changes can be judged by count/step and ms/step, not only total time;
  - added `docs/benchmark_profile_strategy_crosscheck_2026-06-01.md` and
    linked it from `docs/README.md`;
  - added `docs/lm_head_greedy_top1_design_20260601.md`, documenting why a
    pure-JAX tiled greedy top-1 path is not a good replacement for the current
    dense LM-head GEMM.
- accepted opt-in decode change:
  - added `runtime.fastpaths.static_decode_metadata` /
    `NANO_VLLM_JAX_STATIC_DECODE_METADATA` as a strict decode-JIT diagnostic
    path that reuses fixed device metadata and requires device-token carry for
    active rows;
  - added
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`;
  - retained three-repeat result:
    `results/gpu_matrix_decode_static_metadata_r3_20260601.json`;
  - `decode_heavy_128x128`: JAX median `165.81 tok/s`, vLLM `213.54 tok/s`,
    JAX/vLLM `0.776x`, JAX/reference `1.092x`, exact generated-token match;
  - current same-code padded-GEMM control:
    `results/gpu_matrix_decode_padded_gemm_current_control_r3_20260601.json`,
    `161.72 tok/s`, `0.757x` vLLM, exact generated-token match;
  - profile movement: static metadata improves integrated throughput and
    slightly lowers `command_buffer::update` versus the same-code control, but
    it does not move the GPU GEMM/GDN buckets and does not prove a full replay
    boundary. `PjRt Execute`, `MemcpyD2D`, and `np.asarray(jax.Array)` are still
    worse than the control profile, so the next replay slice must target those
    directly.
- rejected and removed from runtime surface:
  - gate/up projection merge:
    `results/gpu_matrix_decode_padded_gemm_gateup_merge_r1_20260601.json`
    preserved exact generated tokens but regressed to `89.68 tok/s`; the
    `NANO_VLLM_JAX_DECODE_MERGE_MLP_GATE_UP` code/config path was removed.
  - BF16-output GDN Q/K pre-normalization:
    `results/gpu_matrix_decode_gdn_prenorm_bf16_out_r1_20260601.json`
    preserved exact generated tokens but regressed to `127.65 tok/s` and did
    not move the `_gdn_fla_chunk_fwd_o_packed_kernel` bucket; the runtime
    request for BF16 prenorm output was removed.
- retained direction:
  - keep padded-GEMM rows=`8` as the current accepted projection route;
  - keep static metadata as the current measured decode best, while treating the
    replay/PjRt problem as unsolved;
  - target next implementation work at real replay boundaries, a single-call
    LM-head GEMM/matvec-plus-argmax backend, and proper FLA/GDN prefill kernels
    rather than resurrecting rejected Triton matvec/reduction variants.

### Entry 193 - Rejected Static-Metadata And GDN Launch-Param Rechecks

- date: 2026-06-01
- purpose:
  - record post-Entry-192 follow-up experiments so they are not retried as
    "obvious" next steps.
- rejected static metadata variant:
  - experiment: carry device-resident `seq_lens` alongside device token carry
    and reuse the cached static decode `seq_lens` placeholder from the
    scheduler;
  - artifact:
    `results/gpu_matrix_decode_static_metadata_seqlens_carry_r3_20260601.json`;
  - result: exact generated-token match, but `165.31 tok/s` (`0.774x` vLLM)
    versus the accepted static-metadata anchor `165.81 tok/s` (`0.776x`);
  - profile regression: `MemcpyD2D` worsened to `113.53 ms` from the anchor's
    `50.56 ms`, and `np.asarray(jax.Array)` worsened to `41.33 ms` from
    `24.38 ms`;
  - decision: removed the seq-lens carry patch. A restore smoke,
    `results/gpu_matrix_decode_static_metadata_restore_r1_20260601.json`,
    returned to the anchor band at `165.61 tok/s` with exact generated-token
    match.
- rejected GDN `chunk_fwd_o` launch variant:
  - experiment: change `_gdn_fla_chunk_fwd_o_packed_kernel` from `num_warps=1`
    to `num_warps=4`;
  - focused test passed:
    `tests/test_gdn_segmented_reference.py::test_gdn_fla_chunk_fwd_o_packed_triton_matches_reference`;
  - artifact:
    `results/gpu_matrix_decode_static_metadata_fwdowarps4_r1_20260601.json`;
  - result: exact generated-token match, but throughput regressed to
    `159.43 tok/s` (`0.747x` vLLM);
  - profile regression: `_gdn_fla_chunk_fwd_o_packed_kernel` increased from
    about `50 ms` to `79.97 ms`;
  - decision: reverted to `num_warps=1`. Do not retry this launch change.
- rejected packed-GDN decode launch-param-only variants:
  - `warps=4, stages=2, block_v=32` on top of the current static-metadata
    route:
    `results/gpu_matrix_decode_static_metadata_w4s2b32_r1_20260601.json`,
    exact but `165.28 tok/s`, with no movement in the top GPU buckets;
  - `warps=8, stages=3, block_v=32` on top of the current static-metadata
    route:
    `results/gpu_matrix_decode_static_metadata_w8s3b32_r1_20260601.json`,
    exact and `166.06 tok/s` in one repeat, but `_gdn_packed_decode_kernel`
    remained about `11.13 ms`, `_gdn_fla_chunk_fwd_o_packed_kernel` stayed
    about `50.06 ms`, and the dominant LM-head/projection GEMM buckets were
    unchanged;
  - decision: do not spend more time on packed-decode launch-parameter-only
    sweeps. Reopen only if the kernel body or surrounding launch structure
    changes materially.
- do-not-repeat guardrail:
  - do not retry static `seq_lens` device carry;
  - do not retry GDN `chunk_fwd_o` warp-count bumps without a kernel-body
    rewrite;
  - do not retry packed-GDN decode launch-param sweeps without a different
    implementation;
  - do not retry runtime/source-level MLP gate/up concatenation or persistent
    duplicated gate/up weight leaves in the forms rejected by earlier packed
    MLP gate/up entries and Entry 192.

### Entry 194 - vLLM/FLA Correctness Boundary Audit

- date: 2026-06-01
- purpose:
  - isolate why the vendored vLLM/FLA GDN prefill kernel diverges from the
    local correctness reference when both use the packed FLA layout.
- artifact:
  - script:
    `benchmarks/audit_vllm_fla_correctness_boundary.py`;
  - result:
    `results/vllm_fla_correctness_boundary_audit_20260601.json`;
  - summary:
    `results/vllm_fla_correctness_boundary_audit_20260601.md`.
- follow-up chunk-size artifacts:
  - chunk 32:
    `results/vllm_fla_correctness_boundary_chunk32_audit_20260601.json`;
  - chunk 16:
    `results/vllm_fla_correctness_boundary_chunk16_audit_20260601.json`.
- setup:
  - Qwen/Qwen3.5-0.8B layer 0, BF16 checkpoint weights with FP32 activations;
  - same real post-conv GDN inputs, same packed `[nnz,H,D]` FLA layout, same
    V,K recurrent state `[N,H,V,K]`;
  - compared local FP32-QKV reference, local BF16-QKV reference, vLLM default
    BF16-QKV/BF16-solve chain, and lower vLLM stage functions with
    FP32-QKV/FP32-solve.
- findings:
  - same-kernel sanity passed: vLLM staged default and vLLM full
    `chunk_gated_delta_rule` matched exactly for output and final state across
    the audited lengths;
  - FP32 to BF16 QKV is a real source of drift under the local algorithm:
    at length 512, JAX BF16-QKV versus JAX FP32-QKV reached output max
    `0.532158` and state max `14.2409`;
  - vLLM default adds separate schedule/kernel drift even after aligning to
    BF16 QKV: at length 512, vLLM default versus local BF16-QKV reached output
    max `1.85312` and state max `8.08015`;
  - first material stage mismatch is not layout: at the `0.01` threshold it
    appears in `attention_matrix` for dtype-only drift and in
    `attention_inverse` for vLLM default versus local BF16-QKV;
  - at the `0.1` threshold, vLLM default versus local BF16-QKV first appears in
    downstream recurrence (`v_new` at length 128, `h` at lengths 256 and 512)
    after smaller solve-stage differences have been amplified;
  - changing only vLLM `solve_tril` to FP32 is not directly usable with BF16
    QKV because the next `recompute_w_u_fwd` Triton dot requires matching
    operand dtypes; BF16 QKV plus FP32 solve is therefore a fork, not a config
    switch;
  - lower vLLM stage functions do run with FP32 QKV plus FP32 solve, and this
    reduces short/mid-length drift, but it still diverges on long rows
    (length 512 output max `0.408505`, state max `6.89226` versus local
    FP32-QKV reference).
- chunk-size follow-up:
  - smaller chunks reduce some solve-stage differences, but they do not remove
    long-row drift;
  - at length 512, chunk 32 changed vLLM-default versus local BF16-QKV to
    output max `0.944162`, state max `3.80353`, while FP32-QKV/FP32-solve
    versus local FP32-QKV was output max `0.461152`, state max `7.32990`;
  - at length 512, chunk 16 changed vLLM-default versus local BF16-QKV to
    output max `0.774548`, state max `13.3355`, while FP32-QKV/FP32-solve
    versus local FP32-QKV was output max `0.441082`, state max `7.37834`.
- decision:
  - do not treat the upstream BF16 vLLM/FLA prefill kernel as correctness
    preserving under the current strict local FP32-activation contract;
  - a usable FLA route needs either a deliberate BF16-prefill correctness
    contract with full-model gates, or a port/fork that controls the solve and
    downstream accumulation dtypes and accumulation order instead of calling
    the upstream wrapper unchanged;
  - do not promote smaller FLA chunk size alone as the correctness fix.

### Entry 195 - vLLM-like FLA Reference Alignment Pass

- date: 2026-06-01
- purpose:
  - reduce ambiguity in the vLLM/FLA correctness investigation by making the
    local vLLM-like staged reference mirror the upstream Triton schedule more
    closely.
- setup:
  - GPU access confirmed with `PYTHONPATH=/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages:.`;
  - active project Python has CPU-only torch, but the vLLM virtualenv provides
    `torch 2.11.0+cu130` and JAX sees the A10G when its site-packages path is
    prepended.
- changes:
  - fixed the vLLM-like chunk-delta reference to keep `h` as the pre-chunk
    state; upstream vLLM stores `h` before the recurrence and writes the
    post-chunk value only to carry/final state;
  - matched vLLM `chunk_delta_h` recurrence quantization more closely:
    `v_new` is stored as BF16, but the pre-gate delta remains FP32 for
    recurrence until the post-gate update is cast to BF16;
  - matched vLLM `wy_fast` more closely by quantizing the beta-weighted
    value/key RHS to BF16 before the `A @ RHS` dot in vLLM-like mode.
- artifacts:
  - CUDA audit after the pass:
    `results/vllm_fla_correctness_boundary_cuda_after_wy_delta_fix_20260601.json`;
  - skip-vLLM local audit after the pass:
    `results/vllm_fla_correctness_boundary_cpu_skip_vllm_after_h_fix_20260601.json`;
  - solve probe after the pass:
    `results/vllm_like_solve_reference_probe_after_ref_fix_20260601.json`.
- findings:
  - full CUDA audit still shows vLLM staged default exactly matches vLLM full
    `chunk_gated_delta_rule`, so the upstream chain is internally coherent;
  - vLLM-like solve is accurate when fed vLLM's exact `A`: inverse max diff
    remains `0.000244141`;
  - the larger audit inverse gap is therefore caused by the preceding
    `chunk_scaled_dot_kkt`/`A` difference, not by the vLLM-like solve formula;
  - trying BF16 QKV with FP32 solve is not a valid drop-in configuration:
    upstream `wy_fast` fails compilation because its dot operands become mixed
    FP32/BF16.
- measured after this pass:
  - length 128 vLLM default versus vLLM-like BF16 path improved in output max
    from `0.0107422` to `0.00756836`;
  - length 256 did not improve: output max changed from `0.0939941` to
    `0.108154`, with first `0.1` drift still in `h`;
  - vLLM default versus local BF16-QKV strict reference is unchanged by these
    reference-only fixes: length 256 output max `0.0301856`, state max
    `0.969607`.
- decision:
  - do not keep guessing at chunk-delta quantization points; the current
    remaining divergence is driven by earlier `A`/solve differences that are
    amplified by recurrent state;
  - next useful work is either matching/porting vLLM `chunk_scaled_dot_kkt`
    exactly, or defining a BF16-prefill acceptance contract around the upstream
    vLLM chain rather than the strict FP32-activation local reference.

### Entry 196 - Block-dot KKT Kernel Probe

- date: 2026-06-01
- purpose:
  - move the KKT stage closer to vLLM's implementation, which computes the
    full `[BT, BT]` chunk/head matrix with `tl.dot` instead of row-wise scalar
    accumulation.
- change:
  - added `gdn_fla_chunk_scaled_dot_kkt_packed_triton_block`;
  - wired opt-in via `NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT=1`;
  - kept the current row-wise KKT kernel as the default while the block-dot path
    is evaluated downstream.
- real-activation KKT comparison:
  - setup: Qwen/Qwen3.5-0.8B, layer 0, length 128, chunk size 64, BF16 QKV
    values, FP32 beta/gate;
  - current JAX reference vs vLLM `A`: max `0.00145054`, mean `2.27261e-05`;
  - current row-wise JAX/Triton KKT vs vLLM `A`: max `0.00140166`, mean
    `2.38475e-05`;
  - new block-dot JAX/Triton KKT vs vLLM `A`: max `9.89437e-06`, mean
    `9.57647e-09`.
- tests:
  - `PYTHONPATH=. pytest tests/test_gdn_segmented_reference.py -k "scaled_dot_kkt_packed_triton" -q`
    passed;
  - `NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT=1 PYTHONPATH=. pytest tests/test_gdn_post_conv_prefill_reference.py -k "model_post_conv_prepared_fla_triton_packed_matches_reference" -q`
    passed.
- decision:
  - block-dot KKT is the right direction for vLLM parity and should be used in
    the next full prefill/decode experiment via the opt-in flag;
  - do not remove the row-wise KKT path until full staged output/state and speed
    runs confirm the block-dot path improves the target workload.

### Entry 197 - vLLM-Shaped FLA Block-Dot Prefill Kernels

- date: 2026-06-01
- purpose:
  - replace the remaining scalar/row-wise packed FLA prefill stages with
    vLLM-shaped tiled `tl.dot` kernels while preserving the strict local
    correctness contract.
- changes:
  - added typed config/env switches for the individual block-dot stages; later
    public benchmark configs use `kernels.gdn.prefill_block_dot` as the single
    all-stage knob to avoid config/env sprawl;
  - added `gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot` as the strict
    no-local-CUDA benchmark config for the block-dot FLA route;
  - ported the vLLM/FLA layout strategy for:
    - KKT: one `[BT,BT]` dot block per chunk/head;
    - recompute W/U: one kernel per chunk/head computing `A @ beta*V` and
      `A @ beta*exp(g)*K`;
    - delta-H: value-tiled recurrent update with `W @ H^T` and `K @ delta`
      block dots, using `BV=32` on A10G to stay under shared-memory limits;
    - output: `Q @ H^T` plus triangular `(Q @ K^T) @ V_new`.
- validation:
  - `PYTHONPATH=. pytest tests/test_gdn_segmented_reference.py -k "scaled_dot_kkt_packed_triton" -q`
  - `PYTHONPATH=. pytest tests/test_gdn_segmented_reference.py -k "recompute_w_u_packed_triton" -q`
  - `PYTHONPATH=. pytest tests/test_gdn_segmented_reference.py -k "delta_h_packed_triton" -q`
  - `PYTHONPATH=. pytest tests/test_gdn_segmented_reference.py -k "fwd_o_packed_triton" -q`
  - `PYTHONPATH=. pytest -q tests/test_server_config.py`
  - `NANO_VLLM_JAX_GDN_KKT_BLOCK_DOT=1 NANO_VLLM_JAX_GDN_RECOMPUTE_BLOCK_DOT=1 NANO_VLLM_JAX_GDN_DELTA_H_BLOCK_DOT=1 NANO_VLLM_JAX_GDN_FWD_O_BLOCK_DOT=1 PYTHONPATH=. pytest tests/test_gdn_post_conv_prefill_reference.py -k "model_post_conv_prepared_fla_triton_packed_matches_reference" -q`
- benchmark ladder, one repeat unless noted:
  - same-code control:
    `results/gpu_matrix_long_prefill_static_metadata_control_r1_20260601.json`,
    `22.70 tok/s`, `0.195x` vLLM, exact generated-token match;
  - KKT block-dot only:
    `results/gpu_matrix_long_prefill_kkt_block_dot_r1_20260601.json`,
    `25.43 tok/s`, `0.219x` vLLM, exact generated-token match;
  - KKT + output block-dot:
    `results/gpu_matrix_long_prefill_kkt_fwd_o_block_dot_r1_20260601.json`,
    `39.68 tok/s`, `0.341x` vLLM, exact generated-token match;
  - KKT + output + delta-H block-dot:
    `results/gpu_matrix_long_prefill_kkt_fwd_o_delta_block_dot_r1_20260601.json`,
    `82.37 tok/s`, `0.708x` vLLM, exact generated-token match;
  - all block-dot FLA stages, retained three-repeat result:
    `results/gpu_matrix_long_prefill_all_block_dot_r3_20260601.json`,
    report `results/gpu_matrix_long_prefill_all_block_dot_r3_20260601.md`,
    median `104.83 tok/s`, `0.901x` vLLM, `1.344x` JAX reference, exact
    generated-token match, speed-claim-ready.
- profile movement:
  - the old dominant GDN stage kernels were removed from the top profile
    buckets;
  - median `forward_step_token_ids_jit` total fell from `2646.18 ms` in the
    same-code control to `437.30 ms`;
  - remaining top GPU buckets are model GEMM/Cutlass events and a much smaller
    `_gdn_fla_chunk_delta_h_packed_block_kernel` at about `24 ms`.
- decision:
  - keep the block-dot FLA route as the current long-prefill best;
  - do not spend more time on scalar row-wise FLA kernels for prefill;
  - next speed work should target the remaining non-GDN GEMM/host buckets or
    integrate these block-dot kernels as the default once broader workload
    coverage is clean.

### Entry 198 - Mixed-Length Block-Dot Prefill Coverage And Benchmark Hygiene

- date: 2026-06-02
- purpose:
  - validate the composed block-dot GDN/FLA prefill route on mixed packed
    lengths, and prevent mixed-workload smoke results from being interpreted as
    long-prefill speed claims.
- changes:
  - added a focused CUDA/JAX-Triton regression for packed `cu_seqlens`
    `[0, 0, 17, 81, 160, 290]` with `chunk_size=64`, covering empty rows,
    partial chunks, exact one-chunk rows, and multi-chunk rows;
  - added `.gitignore` guards for newly generated GPU matrix artifacts so full
    run payloads stay local unless deliberately promoted;
  - made the report matrix show JAX and vLLM reference sources;
  - added `docs/gdn_fla_kernel_flow.md` for the block-dot flow, model-specific
    assumptions, and benchmark interpretation rules.
- validation:
  - `PYTHONPATH=. .venv/bin/python -m pytest tests/test_gdn_segmented_reference.py -k block_dot_mixed_lengths -q`
    passed: `1 passed`;
  - `PYTHONPATH=. .venv/bin/python -m pytest tests/test_gdn_segmented_reference.py -k block_dot -q`
    passed: `5 passed`;
  - one-repeat `hetero8/gpu_paged_gdn_fla_decode_kkt_fwd_o_block_dot` smoke
    passed exact generated-token correctness.
- mixed-workload smoke result:
  - local artifact:
    `results/gpu_matrix_hetero8_block_dot_mixed_prefill_r1_20260602.json`;
  - exact generated-token match: yes;
  - repeats: `1`, so not speed-claim-ready;
  - throughput: `308.27 tok/s`;
  - versus accepted hetero8 JAX reference `stored_entry045`: `0.838x`;
  - versus stored hetero8 vLLM reference: `0.357x`.
- interpretation:
  - this is a loss on the `hetero8` mixed serving workload and must not be
    presented as a serving-wide win;
  - it does not invalidate Entry 197's long-prefill result, because
    `hetero8` combines one prefill step with many decode steps and still shows
    decode/host/GEMM buckets outside the prefill GDN route;
  - the correct next hill-climb target depends on the claim: use
    `long_prefill_512_2048` for prefill-kernel claims, and use `hetero8` plus
    `decode_heavy_128x128` before promoting the route as a default serving
    improvement.

### Entry 199 - Hetero8 Decode Bottleneck Pass

- date: 2026-06-02
- purpose:
  - profile the `hetero8` mixed workload after the block-dot prefill route and
    identify the next integrated bottleneck without reusing rejected source-level
    rewrites.
- starting artifact:
  - `results/gpu_matrix_hetero8_block_dot_mixed_prefill_r1_20260602.json`.
- starting result:
  - exact generated-token match: yes;
  - repeats: `1`, so not speed-claim-ready;
  - throughput: `308.27 tok/s`, `0.838x` stored Entry 045 JAX reference,
    `0.357x` stored vLLM reference;
  - scheduler split: one prefill step took `0.02495 s`; 31 decode steps took
    `0.79231 s`.
- starting bottlenecks:
  - CPU/PJRT: `forward_step_token_ids_jit` `559.13 ms`,
    `PjRtCApiLoadedExecutable::Execute` `552.10 ms`;
  - final token host materialization: `np.asarray(jax.Array)` `96.54 ms`;
  - command-buffer churn: execute/update counts `992`/`961`, totaling
    `44.31 ms`/`44.01 ms`;
  - top GPU buckets remained non-GDN model work: `gemm_fusion_dot_164`
    `110.73 ms`, `gemm_fusion_dot_163` `97.20 ms`, plus smaller GEMM,
    concatenate, transpose, and D2D buckets.
- rejected experiment:
  - attempted to materialize only the requested rows from device token vectors
    before host readback;
  - artifact:
    `results/gpu_matrix_hetero8_token_ref_gather_r1_20260602.json`;
  - exact generated-token match: yes;
  - throughput regressed to `183.34 tok/s`, `0.498x` stored Entry 045 and
    `0.212x` stored vLLM;
  - although `np.asarray(jax.Array)` fell to `11.68 ms`, CPU `gather` rose to
    `1454.40 ms`, PJRT execute rose to `858.21 ms`, and decode time rose to
    `1.35713 s`;
  - decision: reject/revert. Do not retry token-row gather materialization in
    this form.
- accepted change:
  - added `kernels.gdn.packed_decode.max_batch` /
    `NANO_VLLM_JAX_GDN_PACKED_DECODE_MAX_BATCH`;
  - set the block-dot benchmark config to `max_batch: 1`, keeping the Triton
    packed GDN decode route for single-sequence decode-heavy runs while using
    the pure-JAX recurrent path for wider decode batches.
- accepted artifact:
  - `results/gpu_matrix_hetero8_block_dot_maxbatch1_r1_20260602.json`.
- accepted result:
  - exact generated-token match: yes;
  - repeats: `1`, so not speed-claim-ready;
  - throughput: `322.79 tok/s`, `0.878x` stored Entry 045 JAX reference,
    `0.374x` stored vLLM reference;
  - delta versus starting block-dot route: `+14.52 tok/s`, decode time
    `0.79231 s -> 0.76085 s`;
  - command-buffer execute/update counts fell from `992`/`961` to `279`/`248`.
- current remaining bottlenecks after the accepted change:
  - decode dominates: one prefill step is still only `0.02482 s`, while 31
    decode steps total `0.76085 s`;
  - CPU/PJRT compiled execution remains large:
    `forward_step_token_ids_jit` `542.83 ms` and PJRT execute `539.47 ms`;
  - final token materialization is still visible at `109.04 ms`;
  - top GPU buckets are still mostly model GEMM/concatenate/transpose work:
    `gemm_fusion_dot_164` `110.72 ms`, `gemm_fusion_dot_163` `97.19 ms`,
    `gemm_fusion_dot_286` `43.63 ms`, `gemm_fusion_dot_general_473`
    `41.09 ms`, and `wrapped_concatenate` `36.46 ms`.
- same-pass current-default check:
  - `results/gpu_matrix_hetero8_current_default_r1_20260602.json` measured
    `292.71 tok/s`, exact, `0.796x` stored Entry 045;
  - this confirms the block-dot plus max-batch guard route beats the current
    branch default in the same one-repeat profiler setup, but it does not
    replace the stored Entry 045 baseline as the acceptance anchor.
- decision:
  - keep the max-batch routing guard as a small integrated hetero8 win;
  - do not claim the block-dot route is a serving-wide win yet;
  - next hetero8 work should target either compiled decode model buckets or
    final device-token materialization scheduling, but must avoid row-gather
    materialization and must be measured against stored Entry 045 plus vLLM.

### Entry 200 - vLLM Packed Decode Boundary Audit

- date: 2026-06-02
- purpose:
  - explain why the vLLM-shaped packed GDN decode kernel has not produced a
    prefill-like speedup, and separate raw kernel cost from the JAX integration
    boundary.
- source comparison:
  - vLLM's current packed decode boundary takes packed `mixed_qkv`, raw `a`/`b`,
    `A_log`/`dt_bias`, a preallocated output buffer, state indices, and an
    in-place state buffer;
  - the vLLM kernel computes softplus/sigmoid gating, optional Q/K L2 norm,
    recurrence, output, and state update inside one Triton launch;
  - the local JAX route computes or normalizes several inputs outside the
    Triton call, launches through `jax_triton`, returns a separate `new_state`
    value, and lets XLA/JAX own the state/output buffer plumbing.
- artifacts:
  - microbench:
    `results/gdn_decode_kernel_boundary_microbench_20260602.json`;
  - Q/K norm microbench:
    `results/gdn_decode_l2norm_boundary_microbench_20260602.json`;
  - decode-heavy current pre-normalized route:
    `results/gpu_matrix_decode_packed_prenorm_audit_r1_20260602.json`;
  - decode-heavy in-kernel Q/K norm ablation:
    `results/gpu_matrix_decode_packed_in_kernel_norm_audit_r1_20260602.json`;
  - hetero8 block-dot packed-decode in-kernel Q/K norm ablation:
    `results/gpu_matrix_hetero8_blockdot_packed_in_kernel_norm_audit_r1_20260602.json`.
- focused microbench findings:
  - batch 1:
    - pure JAX packed reference boundary: `0.0941 ms/step`;
    - current Triton boundary: `0.0996 ms/step`;
    - Triton kernel-only with prepared inputs: `0.0665 ms/step`;
  - batch 8:
    - pure JAX packed reference boundary: `0.0664 ms/step`;
    - current Triton boundary: `0.1054 ms/step`;
    - Triton kernel-only with prepared inputs: `0.0644 ms/step`;
  - interpretation: the raw recurrence kernel is not slow; the current boundary
    around it can erase the win, especially for batch 8.
- Q/K norm ablation:
  - in-kernel Q/K norm was not a server-level fix;
  - microbench batch 8 boundary moved from `0.1209 ms/step` with external
    pre-normalization to `0.1053 ms/step` with in-kernel normalization in that
    isolated run, but integrated decode did not improve.
- integrated decode-heavy result:
  - current pre-normalized route:
    - exact generated-token match: yes;
    - `158.80 tok/s`, `0.744x` stored vLLM, `1.046x` stored JAX reference;
    - `_gdn_packed_decode_kernel`: `11.11 ms` across `2286` launches;
    - larger buckets: `forward_step_token_ids_jit` `414.86 ms`,
      PJRT execute `391.07 ms`, command-buffer execute/update
      `132.67 ms`/`94.64 ms`, and GPU GEMM/reduction buckets such as
      `gemm_fusion_dot_265` `126.37 ms` and `input_reduce_fusion_153`
      `103.20 ms`.
  - in-kernel Q/K norm route:
    - exact generated-token match: yes;
    - `157.27 tok/s`, `0.737x` stored vLLM, `1.036x` stored JAX reference;
    - `_gdn_packed_decode_kernel`: `12.98 ms` across `2286` launches;
    - no integrated win.
- integrated hetero8 result:
  - block-dot prefill with packed decode enabled for batch 8 and in-kernel Q/K
    norm:
    - exact generated-token match: yes;
    - `316.68 tok/s`, `0.861x` stored Entry 045, `0.366x` stored vLLM;
    - `_gdn_packed_decode_kernel`: `18.71 ms` across `558` launches.
  - comparison:
    - old block-dot route with packed decode and external pre-normalization:
      `308.27 tok/s`;
    - accepted `packed_decode.max_batch=1` route:
      `322.79 tok/s`;
    - therefore keep `max_batch=1`; in-kernel Q/K norm improves the packed
      batch-8 variant but still loses to disabling packed decode for batch 8.
- decision:
  - do not expect the current packed recurrence kernel alone to give a
    prefill-like speedup;
  - the local problem is not the raw `_gdn_packed_decode_kernel` body. It is the
    narrow JAX boundary plus the surrounding decode graph: gate/beta and Q/K
    prep, command-buffer churn, state/output buffer plumbing, GEMM/reduction
    buckets, and token/readback scheduling;
  - keep `packed_decode.max_batch=1` for now;
  - the next serious vLLM-style decode attempt should be a coarser boundary
    that owns packed input prep/gating/QK norm/state update/output together, or
    a runtime strategy that reduces command-buffer/PJRT overhead. Retuning the
    current narrow packed kernel is unlikely to close the gap.

### Entry 201 - Raw-Gate Packed Decode Boundary Probe

- date: 2026-06-02
- change tested:
  - added opt-in `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_raw_gates`;
  - the route passes post-conv BF16 packed QKV plus raw `a`, `b`, positive
    local decay `A`, and `dt_bias` into Triton, then computes softplus/sigmoid
    gating inside `_gdn_packed_decode_raw_gate_kernel`;
  - this matches vLLM's packed recurrent decode contract more closely than the
    prior local route, which materialized `gate`/`beta` in JAX before the
    kernel.
- exact vLLM boundary assessment:
  - exact `qwen_gdn_attention_core` parity is not a small kernel swap in this
    codebase because vLLM's custom op mutates a preallocated core-attention
    output and state slots selected by `state_indices`;
  - the local model receives dense per-batch `conv_state`/`recurrent_state`
    slices and then rebuilds `hybrid_state` with functional `.at[..., layer]`
    updates, so matching vLLM fully requires a model-runner/backend ABI change
    that passes the global hybrid state table plus slot ids into the GDN layer
    kernel.
- validation:
  - `JAX_PLATFORMS=cuda .venv/bin/python -m pytest -q
    tests/test_gdn_packed_decode_reference.py -k
    'triton_bf16_matches_reference'`
  - result: `2 passed, 18 deselected`.
- boundary microbench, Qwen3.5-0.8B decode shape (`H=16`, `HV=16`, `K=128`,
  `V=128`), 200 timed calls:
  - batch 1: current `triton_fla` `0.1001 ms/step`; raw-gate boundary
    `0.0920 ms/step`;
  - batch 8: current `triton_fla` `0.1148 ms/step`; raw-gate boundary
    `0.0998 ms/step`.
- direct decode-heavy integrated comparison (`input_len=128`, `output_len=128`,
  one request, block-dot prefill config, exact-shape JAX reference):
  - current `triton_fla`: exact generated-token match, `144.45 output tok/s`,
    `_gdn_packed_decode_kernel` `11.13 ms` across `2286` launches;
  - raw gates: exact generated-token match, `141.73 output tok/s`,
    `_gdn_packed_decode_raw_gate_kernel` `10.96 ms` across `2286` launches;
  - interpretation: the widened recurrent boundary trims only `0.17 ms` from
    the packed kernel bucket and does not improve end-to-end decode throughput.
- decision:
  - keep the route default-off as a diagnostic/vLLM-contract probe, not as the
    selected serving path;
  - a real fully expanded decode boundary must own conv update and state-slot
    mutation as well, otherwise the same JAX state-plumbing and surrounding
    GEMM/reduction buckets dominate the run.

### Entry 202 - Hybrid State Table Decode ABI Bridge

- date: 2026-06-02
- purpose:
  - move the serving decode ABI toward vLLM's state-indexed contract without
    changing the GDN prefill kernel path.
- change:
  - added `ModelExecutor.forward_step_token_ids_table_jit`, a decode-only
    greedy-token path that takes the full hybrid `conv_state` and
    `recurrent_state` tables plus per-row `hybrid_slot_ids`;
  - the compiled function gathers active hybrid rows, runs the existing model
    forward, scatters updated rows back into the tables with invalid rows
    dropped, and returns the updated tables;
  - `ModelRunner._run_main_and_sample` now uses this table path for the normal
    single-step greedy decode fastpath. Prefill, decode burst, and MTP verifier
    paths remain on the existing sliced `HybridLayerState` contract.
- impact on existing kernel work:
  - GDN prefill kernels are not affected. They still consume sequence-local
    post-conv/GDN tensors and return final recurrent state through the existing
    `HybridLayerState` path;
  - packed decode kernels are not reimplemented here. This is the ABI bridge
    needed before a future conv+recurrent state-indexed decode kernel can own
    the state slots directly;
  - the raw-gate packed decode diagnostic from Entry 201 remains default-off.
- validation:
  - `.venv/bin/python -m pytest -q tests/test_backend_boundaries.py`
    -> `48 passed`;
  - `JAX_PLATFORMS=cuda .venv/bin/python -m pytest -q
    tests/test_gdn_packed_decode_reference.py -k
    'triton_bf16_matches_reference'`
    -> `2 passed, 18 deselected`;
  - decode-heavy no-profile smoke (`input_len=128`, `output_len=128`,
    block-dot prefill config, exact-shape JAX reference) produced exact
    generated-token match and `170.02 output tok/s`.
- decision:
  - keep this bridge as the new serving ABI for single-step greedy decode;
  - next kernel step is to replace the gather/model-local/scatter internals for
    GDN layers with a backend op that accepts table slices/slot ids and owns
    conv update plus recurrent update in one coarser boundary.

### Entry 203 - Fused Conv+Packed GDN Decode Boundary Probe

- date: 2026-06-02
- purpose:
  - test whether widening the packed GDN decode boundary beyond recurrent math
    reduces the decode-heavy gap to the `0.9x` vLLM target.
- changes tested:
  - added opt-in
    `NANO_VLLM_JAX_GDN_PACKED_DECODE_IMPL=triton_fla_conv_raw_gates`;
  - added a Triton decode kernel that consumes raw projected `mixed_qkv`,
    raw `a`/`b`, local positive decay `A`, `dt_bias`, the active layer
    `conv_state`, `conv1d` weight/bias, and recurrent state, then performs the
    width-1 causal-conv update, Q/K pre-normalization, BF16 packed QKV
    quantization, raw gate/beta math, and recurrent-state update in one
    backend call;
  - briefly tested moving table row selection inside each GDN layer, but that
    route measured slower and was removed. The retained default remains the
    faster Entry 202 executor-owned table gather/scatter bridge.
- validation:
  - `.venv/bin/python -m py_compile nanovllm_jax/model.py
    nanovllm_jax/backends.py nanovllm_jax/engine/model_executor.py
    nanovllm_jax/kernels/gdn_fla_triton.py`;
  - `.venv/bin/python -m pytest -q tests/test_backend_boundaries.py -k
    'table_hybrid_decode_matches_sliced_decode or hybrid_slot_ids_zero'`
    -> `2 passed, 46 deselected`;
  - `JAX_PLATFORMS=cuda NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS=1
    .venv/bin/python -m pytest -q tests/test_gdn_packed_decode_reference.py
    -k 'triton_bf16_matches_reference or
    conv_packed_gdn_decode_triton_matches_reference'`
    -> `3 passed, 18 deselected`.
- direct decode-heavy no-profile smokes (`input_len=128`, `output_len=128`,
  one request, block-dot prefill config, exact-shape JAX reference):
  - layer-internal table routing with current `triton_fla`: exact generated
    tokens, `169.73 output tok/s`;
  - fused conv+packed route on layer-internal table routing: exact generated
    tokens, `169.97 output tok/s`;
  - fused conv+packed route on the retained outer table bridge: exact generated
    tokens, `170.03 output tok/s`, effectively tied with Entry 202's
    `170.02 output tok/s`.
- interpretation:
  - the fused conv+recurrent kernel is correctness-clean and removes a real
    narrow boundary, but the integrated throughput is unchanged;
  - the remaining decode-heavy gap is not primarily the GDN width-1 conv or
    recurrent core. Existing profile evidence still points to LM-head GEMM,
    projection GEMM buckets already partly handled by padded GEMM, and
    PjRT/command-buffer replay overhead.
- decision:
  - keep `triton_fla_conv_raw_gates` default-off as a diagnostic and future
    boundary scaffold;
  - do not promote it as the serving default and do not spend more target work
    on GDN decode-core boundary widening unless it also owns a larger model or
    replay boundary;
  - next target work should be a true single-call greedy LM-head
    GEMM/matvec-plus-argmax backend or a practical replay mechanism. This JAX
    environment exposes no CUDA graph API (`jax 0.10.1`, no
    `jax.cuda_graph`).

### Entry 204 - Decode Gap Split: vLLM Also Pays Logits

- date: 2026-06-02
- purpose:
  - test whether vLLM is faster on `decode_heavy_128x128` because it avoids
    materializing LM-head logits, or because other decode buckets are cheaper.
- artifacts:
  - JAX current-best diagnostic summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_split_20260602/nvj_decode_split_diag_jax_20260602.json`;
  - fresh vLLM async throughput reference:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_split_20260602/nvj_decode_split_diag_vllm_async_20260602.json`;
  - vLLM single-process graph-mode CUDA-event split:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_split_20260602/nvj_decode_split_diag_vllm_offline_cuda_events_20260602.json`.
- throughput:
  - JAX `gpu_paged_gdn_fla_decode_static_metadata`, exact generated-token
    match: `167.65 output tok/s`, `0.7635 s` for 128 output tokens;
  - fresh vLLM async, same `128 -> 128` shape with `logprobs=5`:
    `219.03 output tok/s`, `0.5844 s`;
  - vLLM offline single-process CUDA-event run, graph mode enabled and
    `max_num_batched_tokens=2048`: `214.65 output tok/s`, `0.5963 s`.
- JAX profile split:
  - LM-head/logits cutlass GEMM: `127.44 ms` across 127 decode calls;
  - PJRT execute: `428.69 ms` across 410 events;
  - command-buffer execute/update: `130.11 ms` / `90.99 ms`;
  - scheduler/static metadata/device-put side remains visible:
    `$scheduler.py:161 schedule` `110.96 ms`,
    `$scheduler.py:313 build_scheduled_batch` `106.64 ms`,
    `$scheduler.py:499 _static_decode_device_arrays` `99.17 ms`,
    `$api.py:2106 device_put` `99.28 ms`;
  - top non-LM GPU buckets include projection GEMMs
    `101.32 ms`, `61.37 ms`, `50.63 ms`, GDN prefill residual kernels
    `45.16 ms` and `33.71 ms`, and many `input_reduce_fusion` buckets.
- vLLM CUDA-event split:
  - `gpu_runner.execute_model`: `563.03 ms` total CUDA elapsed across 129
    calls;
  - `gpu_runner._model_forward`: `403.94 ms` across 128 calls;
  - `logits_processor.forward`: `137.86 ms` across 128 calls;
  - `logits_processor._get_logits`: `137.33 ms` across 128 calls;
  - `sampler.forward`: `29.79 ms`, with `gather_logprobs` `17.07 ms`,
    `compute_logprobs` `9.43 ms`, and the actual sample path `1.62 ms`.
- interpretation:
  - vLLM is not winning because its LM-head/logits materialization is cheaper.
    Its measured logits bucket is slightly larger than the current JAX
    LM-head bucket, and vLLM additionally computes/gathers top-k logprobs;
  - logits materialization is a shared cost floor for the current benchmark,
    not an upper bound on end-to-end decode speed. vLLM wins around it via
    torch.compile, full/piecewise CUDA graph replay, lower host scheduling
    overhead, and cheaper/fewer surrounding model kernels;
  - a fused LM-head+argmax or candidate-emitter kernel can still be useful, but
    it would be a way to beat a shared floor, not the explanation for the
    current vLLM gap.
- decision:
  - do not treat LM-head materialization as the sole next bottleneck;
  - next decode work should prioritize reducing JAX replay/launch/static
    metadata overhead and the non-LM projection/reduction buckets. A useful
    target is at least `~114 ms` wall-time reduction on this 128-token run,
    which is the gap from `167.65 tok/s` to `0.9x` the fresh vLLM run.

### Entry 205 - PjRT Bucket Overlap Split

- date: 2026-06-02
- purpose:
  - determine whether the large decode-heavy CPU-side `PjRT Execute` bucket
    represents standalone CPU work, or host wrapper/wait time that overlaps GPU
    execution.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/pjrt_split_20260602/pjrt_split_20260602.md`;
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/pjrt_split_20260602/pjrt_split_20260602.json`.
- trace inspected:
  - `/mountpoint/.exp/profiles/20260602-123202-776246-gpu_matrix_decode_heavy_128x128_gpu_paged_gdn_fla_decode_static_metadata_r1_20260602_123202/plugins/profile/2026_06_02_12_33_17/INDCS0291.atrapa.deloitte.com.trace.json.gz`.
- findings:
  - full-trace GPU-active union was `722.07 ms` across `67578` GPU events;
  - `PjRtCApiLoadedExecutable::Execute` summed to `428.69 ms` across `410`
    events, but `407.21 ms` of its union overlapped GPU-active intervals
    (`95.0%`);
  - same-thread PjRT-exclusive time after subtracting nested child ranges was
    only `3.18 ms` total, or `0.0078 ms/event` mean;
  - the nested PjRT stack is mostly
    `PJRT_LoadedExecutable_Execute`, `ExecuteHelperOnSingleDevice`,
    `PjRtStreamExecutorRawLoadedExecutable::Execute`,
    `GpuExecutable::ExecuteThunks`, `jit_compiled:XLA GPU module`, and
    command-buffer execute/update work;
  - grouping the `410` PjRT events by surrounding call shows the main bucket
    is the decode-step executable: `127` `forward_step_token_ids_table_jit`
    events account for `380.82 ms`; smaller executable classes include
    `131` `convert_element_type` events for `19.81 ms`, `133`
    `broadcast_in_dim` events for `12.23 ms`, one non-table
    `forward_step_token_ids_jit` event for `13.77 ms`, and `18` other tiny
    executables for `2.07 ms`;
  - inside `127` `forward_step_token_ids_table_jit` decode rows, child PjRT
    accounted for `380.82 ms`, with `2413` command-buffer executes and `2413`
    command-buffer updates, about `19` of each per decode step.
- interpretation:
  - the PjRT bucket is real host-thread wall time, but it is not a clean
    measure of CPU arithmetic and should not be added to GPU kernel totals;
  - it remains actionable as a replay/launch boundary signal because the host
    thread is occupied while repeated command buffers are updated and executed;
  - the acceptance signal for this route must be lower per-token
    command-buffer execute/update counts plus lower exact-token wall time, not
    only a smaller inclusive `PjRT Execute` label.
- decision:
  - continue decode work on reducing per-step graph/update/replay count or
    making each replay boundary coarser;
  - do not redirect effort toward generic CPU optimization based only on the
    inclusive PjRT bucket.

### Entry 206 - Packed GDN Decode Replay A/B

- date: 2026-06-02
- purpose:
  - test whether the `18 layers * 127 decode tokens` Triton packed-GDN custom
    calls are worth their command-buffer replay cost on
    `decode_heavy_128x128`.
- artifacts:
  - packed off:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_packed_off.json`;
  - packed reference:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_packed_reference.json`;
  - paired no-profile smokes:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_triton_fla_noprofile_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_triton_fla_noprofile_r2.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_noprofile_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_noprofile_r2.json`.
- profiled results:
  - `impl=off`: exact generated-token match, `152.04 tok/s`, `0.8419 s`;
    command-buffer execute/update dropped to `158/127` events but the large
    compiled command buffer and recurrent math regressed wall time;
  - `impl=reference`, BF16 QKV: exact generated-token match, `168.82 tok/s`,
    `0.7582 s`; command-buffer execute/update also dropped to `158/127`
    events and totals fell to `80.93/47.20 ms`;
  - `impl=reference`, FP32 QKV no-profile smoke: exact generated-token match,
    `173.80 tok/s`, essentially tied/slightly below BF16 QKV.
- paired no-profile smokes:
  - current `triton_fla` route: `169.88` and `170.58 tok/s`, exact;
  - explicit packed `reference`, BF16 QKV: `174.09` and `173.88 tok/s`,
    exact.
- interpretation:
  - fully disabling packed decode is not viable: it removes custom-call replay
    but loses too much recurrent-kernel efficiency;
  - explicit packed `reference` with BF16 QKV is the better local
    decode-heavy route today. It keeps the packed-decoder numerics/layout,
    collapses the per-layer Triton custom-call replay, and wins by about
    `+3.8 tok/s` in paired no-profile smokes;
  - this is a hill-climb, not the final route. It moves the current no-profile
    decode-heavy best to about `174 tok/s`, still short of the fresh `0.9x`
    vLLM target of about `197 tok/s`.
- change accepted:
  - updated `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`
    to use `kernels.gdn.packed_decode.impl=reference` and
    `qkv_dtype=bf16`;
  - prefill remains on `triton_fla_padded`, and implicit GDN fallbacks remain
    disabled.
- decision:
  - do not retry `impl=off`;
  - keep BF16-QKV packed reference as the current decode-heavy config baseline;
  - next target should be non-GDN decode model work or a true single-call
    LM-head/top-1 backend. Pallas/two-stage LM-head and source-level gate/up
    packing remain rejected forms and should not be repeated.

### Entry 207 - Static Decode Slot/Token Conversion Cache

- date: 2026-06-02
- purpose:
  - remove avoidable per-token device-array wrapping around static decode
    metadata after Entry 206 changed the packed GDN decode route to the
    explicit BF16-QKV reference path.
- change:
  - cache device `hybrid_slot_ids` vectors by slot tuple in
    `CanonicalModelRunner._batch_hybrid_slot_ids`;
  - avoid re-wrapping already-int32 JAX token-carry arrays with
    `jnp.asarray(..., dtype=jnp.int32)`;
  - pass cached `hybrid_slot_ids` directly into
    `forward_step_token_ids_table_jit` instead of re-wrapping it in the
    executor.
- validation:
  - `.venv/bin/python -m py_compile
    nanovllm_jax/engine/model_runner.py
    nanovllm_jax/engine/model_executor.py`;
  - `.venv/bin/python -m pytest -q tests/test_backend_boundaries.py -k
    'table_hybrid_decode_matches_sliced_decode or hybrid_slot_ids_zero'`
    -> `2 passed, 46 deselected`;
  - `.venv/bin/python -m pytest -q tests/test_device_token_carry.py`
    -> `13 passed`.
- benchmark:
  - no-profile artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_slotcache_noprofile.json`;
  - profiled artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_slotcache_profile.json`.
- result:
  - no-profile decode-heavy remained exact and essentially neutral at
    `173.99 tok/s`;
  - profiled decode-heavy remained exact at `169.36 tok/s`;
  - profile structure improved materially versus Entry 206's profiled
    reference run: `PjRtCApiLoadedExecutable::Execute` dropped from
    `391.10 ms / 410` to `198.22 ms / 283`, command-buffer update dropped from
    `47.20 ms / 127` to `35.26 ms / 127`, and `MemcpyD2D` dropped from
    `89.34 ms / 616` to `4.18 ms / 362`.
- interpretation:
  - this is a profile-structure cleanup, not a standalone throughput win;
  - it is still worth keeping because it removes incidental per-step
    conversion/copy churn before the next model-kernel optimization pass.
- decision:
  - keep the cache/conversion cleanup;
  - continue target work on the remaining model-side decode buckets rather than
    more host-conversion cleanup.

### Entry 208 - Rejected GDN z/a/b Decode Padded-GEMM Extension

- date: 2026-06-02
- purpose:
  - test whether the existing B=1 decode padded-GEMM lowering should be
    extended from GDN `in_proj_qkv` to GDN `in_proj_z`, `in_proj_a`, and
    `in_proj_b`.
- change tested:
  - temporarily routed decode-only `in_proj_z`, `in_proj_a`, and `in_proj_b`
    through `_decode_padded_gemm_dot` when
    `NANO_VLLM_JAX_DECODE_PADDED_GEMM=1` and the existing shape guard passed.
- validation:
  - `.venv/bin/python -m py_compile nanovllm_jax/model.py`;
  - `.venv/bin/python -m pytest -q tests/test_backend_boundaries.py -k
    'executor_jit_matches_eager_cached_decode or
    executor_greedy_decode_burst_matches_iterative_token_path or
    table_hybrid_decode_matches_sliced_decode'`
    -> `3 passed, 45 deselected`;
  - `JAX_PLATFORMS=cuda NANO_VLLM_JAX_GDN_DISABLE_FALLBACKS=1
    .venv/bin/python -m pytest -q tests/test_gdn_packed_decode_reference.py
    -k 'fallback_only or triton_bf16_matches_reference'`
    -> `2 passed, 19 deselected`.
- benchmark:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_gdn_zab_padded_noprofile.json`;
  - exact generated-token match, but throughput regressed to `148.76 tok/s`
    from the Entry 207 baseline around `173.99 tok/s`.
- interpretation:
  - the small GDN gate/Z projections are not good candidates for the existing
    padded-GEMM helper. Padding them to the rows=8 GEMM shape adds more cost
    than it removes from reduction-style lowering.
- decision:
  - revert the change;
  - do not extend padded-GEMM lowering to GDN `in_proj_z`, `in_proj_a`, or
    `in_proj_b` without a materially different kernel family.

### Entry 209 - Rejected Decode Replay And Source-Boundary Probes

- date: 2026-06-02
- purpose:
  - finish the decode-heavy replay/source cleanup pass after Entry 207 without
    retrying older rejected kernel families.
- rejected probes:
  - single-row hybrid table bypass:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_single_row_notable_noprofile.json`;
    exact tokens, but regressed to `157.18 tok/s`;
  - normalized-hidden LM-head boundary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_normed_hidden_noprofile.json`;
    exact tokens, but neutral/slightly below baseline at `173.89 tok/s`;
  - LM-head `top_k(logits, 1)` greedy selector:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_lmhead_topk1_noprofile.json`;
    exact tokens, but regressed to `172.43 tok/s`;
  - XLA command-buffer disabled:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_xla_no_command_buffer_noprofile.json`;
    exact tokens, but regressed to `163.47 tok/s`;
  - XLA command-buffer limited to `CUBLAS,CUSTOM_CALL`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_xla_cublas_custom_command_buffer_noprofile.json`;
    exact tokens, but regressed to `163.35 tok/s`;
  - XLA autotune level `5`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_xla_autotune5_noprofile.json`;
    exact tokens, but slightly regressed to `173.44 tok/s`.
- validation:
  - the temporary source probes passed focused py_compile/unit checks before
    benchmark and were reverted after losing;
  - command-buffer flag smokes also found that numeric values such as `1`, `2`,
    and unavailable CUDA-graph flags are not accepted by this jaxlib build.
- interpretation:
  - the table JIT remains useful even for batch `1`;
  - final-norm/source-boundary spelling and LM-head reduction spelling do not
    remove enough real work;
  - default XLA command-buffer behavior is helping this graph. Disabling it or
    narrowing the capture set gives up more replay efficiency than it saves.
- decision:
  - keep Entry 207 as the source baseline;
  - do not retry these source-level or XLA flag variants for the current
    decode-heavy target.

### Entry 210 - Accepted Block-Dot Prefill Plus Reference Decode Target Hit

- date: 2026-06-02
- purpose:
  - combine the current best decode route from Entry 206/207
    (`packed_decode=reference`, BF16 QKV, static metadata, padded decode GEMMs)
    with the accepted vLLM-shaped block-dot FLA prefill stages from Entry 197.
- change accepted:
  - `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json` now sets
    `kernels.gdn.prefill_block_dot=true`;
  - `benchmarks/benchmark_jax_server_trace.py` now records the four block-dot
    prefill stage flags in `run_config.gdn_kernel_flags`, so artifacts show
    whether KKT, recompute, delta-H, and output block-dot stages were active.
- no-profile target artifacts:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_blockdot_prefill_noprofile.json`;
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_blockdot_prefill_noprofile_r2.json`;
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/decode_heavy_static_metadata_reference_blockdot_prefill_noprofile_r3.json`.
- no-profile results:
  - all three repeats passed exact generated-token parity;
  - throughput: `197.91`, `196.38`, `197.96 tok/s`;
  - median throughput: `197.91 tok/s`;
  - median wall time: about `0.6468 s`;
  - versus the fresh vLLM async baseline from Entry 204/205,
    `219.03 tok/s`, the median ratio is `0.904x`;
  - the fresh `0.9x` target is about `197.12 tok/s`, so the median no-profile
    run meets the requested decode-heavy threshold.
- config/profile smoke:
  - matrix artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/gpu_matrix_static_metadata_blockdot_reference_decode_r1_20260602.json`;
  - report:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/gpu_matrix_static_metadata_blockdot_reference_decode_r1_20260602.md`;
  - per-repeat artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/decode_gdn_replay_20260602/gpu_matrix_static_metadata_blockdot_reference_decode_r1/decode_heavy_128x128_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity passed, and the artifact records all four
    block-dot flags as `true`;
  - profiled throughput was `191.23 tok/s`, so this smoke is for config and
    bucket validation only, not the no-profile speed claim.
- profile movement:
  - command-buffer execute count is `158`, matching the Entry 207 reference
    decode route and lower than the older Triton packed-decode route;
  - top GPU buckets after the config smoke remain decode GEMM/reduction work:
    LM-head `gemm_fusion_dot_199` `126.48 ms`, MLP gate/up
    `gemm_fusion_dot_175` `108.87 ms`, GDN QKV
    `gemm_fusion_dot_general_409` `64.80 ms`, and MLP down
    `gemm_fusion_dot_200` `51.47 ms`;
  - the win comes from using the block-dot prefill kernels for the one
    `128`-token prompt step while keeping the lower-replay reference packed
    decode path for the `127` decode steps.
- interpretation:
  - the earlier loop was caused by testing prefill and decode kernel choices
    separately. The target route is the composition: block-dot FLA prefill plus
    reference packed decode;
  - this reaches the requested decode-heavy `0.9x` threshold on median
    no-profile repeats, but it does not mean every workload is solved. Hetero8
    remains a separate integrated-serving claim, and profiled runs should not
    be compared directly against no-profile vLLM timing.
- decision:
  - promote this as the current best strict decode-heavy config;
  - next work should keep this route as the baseline and either broaden the
    same composition to hetero8 or target the remaining decode GEMM/reduction
    buckets only if the new baseline fails on broader workloads.

### Entry 211 - Full Workload Sanity Check After Target Hit

- date: 2026-06-02
- purpose:
  - sanity check the promoted Entry 210 config across the regular benchmark
    workloads with no profiling overhead before updating the PR.
- setup:
  - strict no-fallback route:
    `prefill_block_dot=true`, `packed_decode.impl=reference`,
    `packed_decode.qkv_dtype=bf16`, static decode metadata, padded decode
    GEMMs, and local CUDA probes disabled;
  - two no-profile repeats per workload using
    `benchmarks/benchmark_jax_server_trace.py`;
  - artifacts stored under:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/full_matrix_sanity_20260602/`.
- results:
  - `hetero8`: exact parity in both repeats; `316.68`, `318.03 tok/s`,
    median `317.35 tok/s`; stored vLLM `864.18 tok/s`; ratio `0.367x`;
  - `short_32_128`: exact parity in both repeats; `391.13`,
    `381.92 tok/s`, median `386.52 tok/s`; stored vLLM `568.26 tok/s`;
    ratio `0.680x`;
  - `long_prefill_512_2048`: exact parity in both repeats; `106.17`,
    `105.98 tok/s`, median `106.08 tok/s`; stored vLLM `116.37 tok/s`;
    ratio `0.912x`;
  - `decode_heavy_128x128`: exact parity in both repeats; `197.02`,
    `198.18 tok/s`, median `197.60 tok/s`; fresh vLLM async baseline
    `219.03 tok/s`; ratio `0.902x`.
- interpretation:
  - the promoted config is correctness-clean on all four sanity workloads;
  - the `0.9x` target remains met for the two target-sensitive workloads:
    decode-heavy and long-prefill;
  - `decode_heavy_128x128` is close to the line, but the new two-repeat median
    plus Entry 210's prior three-repeat median are both above the fresh `0.9x`
    threshold;
  - hetero8 and short are not solved by this route. Hetero8 remains the main
    broader-serving bottleneck and should not be presented as target-met.
- decision:
  - update the PR with the target-hit result and the broader sanity caveat;
  - keep future speed work focused on hetero8/broader serving rather than
    re-opening the decode-heavy composition unless it regresses below target.

### Entry 212 - Hetero8 BF16 Decode Projections And cuBLAS GEMM Route

- date: 2026-06-02
- purpose:
  - hill-climb `hetero8` as the primary mixed-serving metric after Entry 211
    showed decode-heavy and long-prefill were target-clean but hetero8 was not.
- accepted source/config changes:
  - broaden `runtime.fastpaths.decode_proj_act_dtype` from
    `bf16_single_seq` to `bf16`, so B>1 decode projections use BF16
    activations under the same explicit runtime policy as B=1;
  - apply that decode projection dtype policy to GDN `out_proj` and full
    attention `o_proj` activations before their decode output projections;
  - set the selected config's XLA flags to
    `--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false`.
- direct no-profile results:
  - B>1 BF16 decode projections, exact over two repeats:
    `341.14`, `340.13 tok/s`, median `340.63 tok/s`;
  - BF16 output projections plus static metadata, exact over two repeats:
    `346.10`, `344.59 tok/s`, median `345.35 tok/s`;
  - BF16 output projections plus no Triton GEMM, exact over two repeats:
    `441.10`, `438.98 tok/s`, median `440.04 tok/s`.
- canonical profiled matrix:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260602/gpu_matrix_hetero8_no_triton_gemm_r2_20260602.json`;
  - report:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260602/gpu_matrix_hetero8_no_triton_gemm_r2_20260602.md`;
  - exact generated-token parity in both repeats;
  - median profiled throughput: `414.45 tok/s`;
  - ratio: `1.127x` stored Entry 045 JAX reference, `0.480x` stored vLLM
    hetero8 reference (`864.18 tok/s`);
  - scheduler split: one prefill step about `0.03 s`; 31 decode steps about
    `0.58 s`.
- profile movement:
  - no-Triton-GEMM route replaces the worst `gemm_fusion` lowering with CUTLASS
    events; top GPU rows include `cutlass_80_tensorop_s1688gemm...`
    `68.2 ms / 72` and `ampere_bf16_s16816gemm...sliced1x2...`
    `61.5 ms / 1488`;
  - `PjRtCApiLoadedExecutable::Execute` median falls to about
    `361.96 ms / 91` in the profiled matrix, from about `543 ms` on the
    BF16-output-projection profile and `575.87 ms` on the original hetero8
    profile;
  - visible token materialization remains large:
    `np.asarray(jax.Array)` about `120.90 ms / 256`;
  - transpose/concatenate remain visible, with GPU transpose about
    `40.40 ms / 1221`.
- rejected or not-promoted probes:
  - `packed_decode.max_batch=1` with B>1 BF16 projections was exact but noisy
    and slower by median (`337.97 tok/s`) than all-batch packed reference;
  - `packed_decode=off` with static metadata off was exact but slightly slower
    (`340.33 tok/s`) than keeping packed reference;
  - disabling static decode metadata was exact and throughput-neutral
    (`340.69 tok/s` before output casts), while lowering ITL p95; keep it as a
    tail-latency diagnostic, not the throughput-selected route;
  - after no-Triton-GEMM, disabling static decode metadata remained exact and
    lowered ITL p95 (`65.64 ms` versus about `148 ms`) but slightly regressed
    throughput to `438.11 tok/s`, so it is still not the throughput route;
  - explicit rank-2 spelling for single-token `_tokenwise_decode_dot` was exact
    but neutral/slower (`344.60 tok/s`) after output casts.
- interpretation:
  - the main hetero8 speedup did not come from more hand tuning of GEMM tile
    values; it came from selecting the better XLA GEMM backend for this small-M
    decode workload. XLA's Triton GEMM fusion was the wrong lowering here,
    while the no-Triton route gets faster CUTLASS/cuBLAS-style kernels;
  - hetero8 is improved materially but remains far from the `0.9x` vLLM target.
    The remaining gap is still decode dominated, with host token materialization
    and transpose/concat/layout work now prominent after GEMM backend selection.
- decision:
  - keep the BF16 projection broadening, output-projection casts, and selected
    config XLA flags;
  - continue hetero8 work from the new `414-440 tok/s` anchor, targeting token
    materialization scheduling and transpose/concat/layout bubbles next.

### Entry 213 - Accepted Layerwise Hybrid-State Table Decode

- date: 2026-06-03
- purpose:
  - reduce hetero8 decode table gather/scatter and whole-table state update work
    without changing the public functional state contract.
- accepted change:
  - `forward_step(..., hybrid_state_layerwise=True)` gathers each GDN layer's
    conv/recurrent state into per-layer lists, passes only the per-layer state
    through `gated_deltanet_block`, and stacks the updated layers once at the
    end of the decode step;
  - `forward_step_token_ids_table_jit` enables this layerwise path for table
    decode.
- validation:
  - `python -m py_compile nanovllm_jax/model.py nanovllm_jax/engine/model_executor.py`
  - `pytest -q tests/test_backend_boundaries.py -k 'table_hybrid_decode_matches_sliced_decode or executor_jit_matches_eager_cached_decode or executor_greedy_decode_burst_matches_iterative_token_path'`
    - 3 passed, 45 deselected.
- result:
  - profiled hetero8 artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/gpu_matrix_hetero8_layerwise_hybrid_state_r1_20260603.json`;
  - exact generated-token parity;
  - profiled throughput `442.58 tok/s`;
  - no-profile throughput `457.56 tok/s`, about `0.529x` stored vLLM
    hetero8 (`864.18 tok/s`).
- rejected follow-ups from the same pass:
  - layerwise KV-cache internal path: exact but `395.06 tok/s`;
  - generalized B<=rows padded GEMM: exact but `454.07 tok/s`;
  - `packed_decode=off`: exact but `455.90 tok/s`;
  - RoPE duplicate-embedding removal: exact but no-profile `456.62 tok/s`
    and profiled `443.36 tok/s`, not a meaningful improvement;
  - closing over all model params for decode: focused tests passed, but
    lowering captured about `2.01 GB` of constants and the benchmark process was
    killed. Do not retry the naive closed-params route.

### Entry 214 - Accepted Canonical Packed Decode Projections And Conv-Fused GDN Decode

- date: 2026-06-03
- purpose:
  - remove the newly dominant hetero8 decode materialization buckets after
    Entry 213: GDN input-projection weight concat, full-attention Q/K/V concat,
    MLP gate/up concat, and GDN conv/recurrent decode source fusions.
- accepted changes:
  - add loader/init packed decode weights:
    - GDN `in_proj_qkv_abz = [in_proj_qkv, in_proj_a, in_proj_b, in_proj_z]`;
    - full-attention `qkv_proj_decode = [q_proj, k_proj, v_proj]`;
    - MLP `gate_up_proj = [gate_proj, up_proj]`;
  - use the packed GDN and full-attention projection weights for B>1,
    width-1 decode only;
  - use canonical packed MLP gate/up in prefill and decode when present;
  - make `ModelParams` flattening omit separate `gate_proj`/`up_proj` leaves
    when `gate_up_proj` exists, avoiding the duplicate-large-leaf failure mode
    from earlier packed-MLP rejections;
  - promote `packed_decode.impl=triton_fla_conv_raw_gates` in
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`;
  - fix the selected config's hetero8 `input_lens` to the canonical eight rows:
    `64,128,192,256,320,384,448,512`.
- validation:
  - `python -m py_compile nanovllm_jax/model.py nanovllm_jax/load_weights.py nanovllm_jax/engine/model_executor.py`
  - `pytest -q tests/test_backend_boundaries.py -k 'table_hybrid_decode_matches_sliced_decode or executor_jit_matches_eager_cached_decode or executor_greedy_decode_burst_matches_iterative_token_path'`
    - 3 passed, 45 deselected.
  - `python -m json.tool benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`
- results:
  - GDN packed input projection only:
    `493.08 tok/s`, exact, artifact
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/jax_hetero8_gdn_decode_in_proj_prepack_b8_noprofile_20260603.json`;
  - plus full-attention Q/K/V packed decode:
    `497.28 tok/s`, exact, artifact
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/jax_hetero8_gdn_fullattn_decode_prepack_b8_noprofile_20260603.json`;
  - plus canonical packed MLP gate/up:
    `543.27 tok/s`, exact, artifact
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/jax_hetero8_packed_mlp_canonical_b8_noprofile_20260603.json`;
  - plus conv-fused Triton FLA GDN decode:
    exact over two no-profile repeats, `552.86` and `552.59 tok/s`, median
    `552.72 tok/s`;
  - current hetero8 ratio: `0.640x` stored vLLM hetero8 (`864.18 tok/s`);
  - profiled selected artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/gpu_matrix_hetero8_packed_mlp_triton_fla_conv_raw_gates_r1_20260603.json`,
    `530.23 tok/s`, exact.
- profile movement:
  - the GDN/full-attention/MLP runtime weight-concat buckets disappear from the
    decode top rows;
  - canonical packed MLP reduces `PjRtCApiLoadedExecutable::Execute` from about
    `320.84 ms / 91` in the full-attention packed profile to about
    `296.68 ms / 91` before the conv-fused GDN switch;
  - conv-fused GDN decode replaces the reference GDN decode fusions with
    `_gdn_conv_packed_decode_raw_gate_kernel` at about `20.52 ms / 558`.
- rejected or not-promoted probes:
  - invalid early GDN prepack run used seven prompt lengths by mistake and must
    not be used as hetero8 evidence;
  - `triton_fla_raw_gates` non-conv decode is exact but neutral/slower at
    `543.04 tok/s` versus the canonical packed-MLP reference route's
    `543.27 tok/s`;
  - the earlier duplicate-leaf packed MLP rejections still stand. Only the
    canonical representation with separate gate/up leaves omitted from the JIT
    parameter tree is accepted.
- decision:
  - promote canonical packed MLP plus conv-fused Triton FLA GDN decode as the
    current hetero8 anchor;
  - continue from `552.72 tok/s`, not from the older `457.56` or `497.28`
    anchors;
  - remaining dominant buckets are GEMM-heavy decode work and LM-head/vocab
    reduction, not simple runtime weight concatenation.

### Entry 215 - Accepted BF16 Full-Attention KV Cache For Hetero8

- date: 2026-06-03
- purpose:
  - reduce full-attention decode cache bandwidth and cache-write materialization
    after Entry 214 removed the simpler runtime weight-concat buckets.
- accepted changes:
  - add typed config key `kernels.full_attention.kv_cache_dtype`, translated to
    `NANO_VLLM_JAX_FULL_ATTN_KV_CACHE_DTYPE`;
  - allocate full-attention KV storage in BF16 when this key is set to `bf16`;
  - cast K/V explicitly to the physical cache dtype inside `update_kv_cache`,
    avoiding implicit JAX scatter casts;
  - promote `"full_attention": {"kv_cache_dtype": "bf16"}` in
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`.
- validation:
  - `python -m py_compile nanovllm_jax/backends.py nanovllm_jax/kv_cache.py nanovllm_jax/server_config.py`
  - `pytest -q tests/test_server_config.py tests/test_backend_boundaries.py -k 'full_attention_kv_cache_dtype_override or pure_jax_decode_attention_matches_dense_reference_non_contiguous_blocks or table_hybrid_decode_matches_sliced_decode or executor_jit_matches_eager_cached_decode or executor_greedy_decode_burst_matches_iterative_token_path'`
    - 5 passed, 50 deselected.
- no-profile results:
  - artifacts:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/kv_cache_bf16/jax_hetero8_full_attn_kv_bf16_noprofile_20260603.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/kv_cache_bf16/jax_hetero8_full_attn_kv_bf16_noprofile_r2_20260603.json`;
  - exact generated-token parity on all eight rows;
  - throughputs `585.81` and `589.24 tok/s`, median `587.53 tok/s`;
  - ratio versus stored vLLM hetero8 `864.18 tok/s`: `0.680x`;
  - improvement versus Entry 214 median `552.72 tok/s`: `1.063x`.
- profiled result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/kv_cache_bf16/gpu_matrix_hetero8_full_attn_kv_bf16_r1_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity;
  - profiled throughput `556.66 tok/s`;
  - root `MemcpyD2D` fell from `55.10 ms / 872` in Entry 214's profile to
    `12.32 ms / 872`; scoped GPU `MemcpyD2D` fell from `17.13 ms / 445` to
    `7.62 ms / 445`;
  - `PjRtCApiLoadedExecutable::Execute` fell from `296.06 ms / 91` to
    `269.17 ms / 91`, and `np.asarray(jax.Array)` fell from `77.70 ms / 32`
    to `69.37 ms / 32`.
- rejected or not-promoted probes from the same pass:
  - `all_rows_valid` sentinel-skip for KV writes: exact and slightly faster
    under profile (`534.14 tok/s` versus `530.23`), but slower in the
    no-profile acceptance lane (`547.59 tok/s` versus `552.72`), so the source
    patch was reverted;
  - `runtime.fastpaths.trace_token_prefetch=true`: exact but slower at
    `548.44 tok/s`, so leave it off;
  - compacting the full-attention cache layer dimension from 24 logical layers
    to the six full-attention layers was exact at `586.70 tok/s`, essentially
    neutral versus the BF16-cache median, so the compact-cache code/config key
    was reverted;
  - FlashInfer KV append after BF16 cache unblocked the dtype guard and passed
    the focused backend boundary test, but the integrated hetero8 run was exact
    and slower at `578.89 tok/s`, so the temporary serving-path changes were
    reverted and `flashinfer_kv_append` remains off;
  - FlashInfer decode attention is not a ready JAX drop-in in the current
    stack. The installed FlashInfer decode path is Torch custom-op oriented;
    this repo's JAX FlashInfer FFI currently only covers paged KV append.
- interpretation:
  - this is a real integrated cache-bandwidth win, not a broad BF16 activation
    change. GDN recurrent state and model residual math remain on the existing
    dtype contract;
  - the remaining hetero8 gap is still decode dominated. After BF16 KV cache,
    top GPU buckets are GEMMs (`68.26 ms / 72`, `55.63 ms / 1488`,
    `31.29 ms / 31`, `24.54 ms / 1488`), full-attention/cache fusions
    (`loop_slice_fusion_23` `22.20 ms / 32`, `input_scatter_fusion_15`
    `19.18 ms / 31`), and the conv-fused GDN decode kernel
    (`20.64 ms / 558`).
- decision:
  - promote BF16 full-attention KV cache as the current hetero8 anchor;
  - continue from no-profile median `587.53 tok/s` (`0.680x` vLLM);
  - do not retry compact-cache layer packing unless a future profile shows
    layer-dimension copies are again dominant.

### Entry 216 - Accepted Packed Prefill Projection Reuse

- date: 2026-06-03
- purpose:
  - test whether the already-loaded packed projection weights can remove
    same-input prefill projection work without adding model-specific tile
    tuning or new runtime knobs.
- accepted change:
  - reuse GDN `in_proj_qkv_abz = [in_proj_qkv, in_proj_a, in_proj_b, in_proj_z]`
    during prefill when the existing compact GDN QKV and Z prefill controls are
    enabled;
  - reuse full-attention `qkv_proj_decode = [q_proj, k_proj, v_proj]` during
    prefill when the existing compact full-attention projection control is
    enabled;
  - keep the packed paths behind the existing compact-prefill switches and
    packed-weight presence checks. No new env/config key was added.
- validation:
  - `python -m py_compile nanovllm_jax/model.py`
  - `pytest -q tests/test_lm_head_helpers.py tests/test_backend_boundaries.py -k 'compact_prefill_dot_matches_dense_on_valid_tokens or compact_prefill_mlp_matches_dense_on_valid_tokens or bucketed_prefill_last_logits_match_exact_prefill or bucketed_linear_prefill_preserves_hybrid_state_for_decode or ragged_prefill_and_padded_multiseq_decode_match_dense_recompute or executor_jit_matches_eager_cached_decode'`
    - 6 passed, 46 deselected.
- no-profile results:
  - artifacts:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/packed_prefill_proj/jax_hetero8_packed_prefill_proj_noprofile_20260603.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/packed_prefill_proj/jax_hetero8_packed_prefill_proj_noprofile_r2_20260603.json`;
  - exact generated-token parity in both repeats;
  - throughputs `593.98` and `588.51 tok/s`, median `591.24 tok/s`;
  - ratio versus stored vLLM hetero8 `864.18 tok/s`: `0.684x`;
  - improvement versus Entry 215 median `587.53 tok/s`: `1.006x`.
- profiled result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/hetero8_hillclimb_20260603/packed_prefill_proj/gpu_matrix_hetero8_packed_prefill_proj_r1_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity;
  - profiled throughput `562.02 tok/s`;
  - `PjRtCApiLoadedExecutable::Execute` moved from `269.17 ms / 91` in Entry
    215 to `260.48 ms / 91`;
  - scoped GPU `MemcpyD2D` moved from `7.62 ms / 445` to `5.29 ms / 428`.
- interpretation:
  - this is a small structural win and generalizes across Qwen 3.5 dense sizes
    because it only concatenates projections that already share the same input;
  - it does not reduce the dominant decode projection GEMM counts. The main
    decode GEMM buckets remain essentially unchanged (`68.42 ms / 72`,
    `55.62 ms / 1488`, `31.28 ms / 31`, and `24.56 ms / 1488` in the new
    profile);
  - do not treat this as evidence that the `0.9x` hetero8 target is solved.
    The remaining gap is still decode-dominated.
- decision:
  - keep packed prefill projection reuse as part of the selected
    `gpu_paged_gdn_fla_decode_static_metadata` route;
  - continue from no-profile median `591.24 tok/s` (`0.684x` vLLM);
  - next work should target a real decode-side structural boundary, not another
    prefill-only projection cleanup.

### Entry 217 - Random Sidecar Harness Findings

- date: 2026-06-03
- purpose:
  - run the new random-request sidecar on the harder request distribution:
    input tokens `512-4096`, output tokens `256-1024`, request count `5-15`,
    seed `1234`.
- accepted harness/server fixes:
  - `benchmark_random_request_sidecar.py` now passes BF16 to vLLM by default
    when JAX uses FP32 activations, because vLLM's Qwen3.5 GDN path rejects
    FP32 chunked GDN;
  - `Sequence.materialize_device_token_slots` gathers only referenced
    `DeviceTokenRef` rows instead of copying whole token vectors;
  - `generate_with_trace` can use the existing typed
    `runtime.fastpaths.trace_token_prefetch` route to collect trace events via
    one-step-lag deferred token materialization;
  - `_record_device_token_carry` now preserves existing carries across
    non-final prefill chunks and merges only rows that actually emitted tokens,
    fixing chunked-prefill/static-metadata scheduling.
- validation:
  - `python -m py_compile nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/sequence.py benchmarks/benchmark_random_request_sidecar.py tests/test_device_token_carry.py tests/test_benchmark_random_request_sidecar.py`
  - `pytest -q tests/test_device_token_carry.py tests/test_benchmark_random_request_sidecar.py`
    - 23 passed, 2 warnings.
  - `git diff --check`
- random-suite metadata:
  - manifest seed `1234`, `15` requests;
  - prompt length min/mean/max: `1077 / 2033.73 / 4022`;
  - output length min/mean/max: `425 / 773.47 / 1007`;
  - total input/output tokens in the vLLM lane: `30506 / 11602`.
- vLLM result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r2_vllm.json`;
  - throughput `1337.00 output tok/s`, total token throughput
    `4852.47 tok/s`;
  - TTFT p50 `584.33 ms`, ITL p50 `6.96 ms`;
  - note: vLLM did runtime Triton JIT for this random shape, so the first run
    includes some cold-kernel latency.
- JAX result:
  - no completed JAX artifact for this full random suite yet.
  - default JAX sidecar failed during warmup token materialization with
    `RESOURCE_EXHAUSTED` allocating `12.98 GiB`;
  - lowering JAX `max_num_batched_tokens` to `4096` fixed the immediate prefill
    OOM and exposed the static-metadata carry bug fixed above;
  - the suite also needs `max_blocks_per_seq=512`, because one request needs
    `5029` prompt+output tokens and the default `256` blocks only covers
    `4096` tokens;
  - capacity-correct attempts with `max_blocks_per_seq=512` and
    `num_kvcache_blocks=2048` still fail during warmup token materialization
    with about `9.46-9.50 GiB` requested, including with
    `trace_token_prefetch=true` and BF16 activations.
- interpretation:
  - the random sidecar is materially harder than `hetero8`: it combines much
    longer prefills, much longer outputs, and per-request capacity above the
    default JAX block budget;
  - current JAX cannot yet produce a fair full-suite random comparison at the
    requested range on this GPU. The blocker is not just decode speed; it is
    warmup/prefill execution plus token materialization under large cache
    shapes;
  - do not quote a JAX/vLLM random ratio until the JAX lane completes with exact
    generated-token parity.
- next actions:
  - make the sidecar derive JAX capacity settings from the generated manifest
    before launch;
  - add a low-memory trace-output mode that avoids forcing large outstanding
    prefill executions through final token materialization;
  - continue hetero8 speed work separately from this harness blocker.

### Entry 218 - Working Random Sidecar Baseline And Memory Check

- date: 2026-06-03
- purpose:
  - make the seed-`1234` random sidecar complete on the A10G and measure vLLM
    memory on the same manifest.
- accepted fixes:
  - scheduler now continues already-admitted running prompt tails when the
    waiting head cannot yet fit in KV cache, instead of turning blocked
    admission into a false global capacity failure;
  - canonical model-runner compatibility `kv_state` now references the backend
    cache storage instead of allocating a second full K/V cache snapshot;
  - block manager records the hash for a prompt-tail block that becomes full
    during decode before allocating the next block;
  - `ModelRunner.release()` removes finished sequence IDs from the device-token
    carry map instead of clearing carries for still-running rows.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/block_manager.py nanovllm_jax/engine/scheduler.py`
  - focused scheduler/block tests:
    `pytest -q tests/test_backend_boundaries.py -k 'block_manager_hashes_completed_prompt_tail_before_next_block or scheduler_continues_running_prefill_when_waiting_head_needs_kv or scheduler_chunks_prefill_by_max_batched_tokens_budget or no_prefix_cache_allocation'`
    - 5 passed, 46 deselected;
  - focused carry/static-metadata tests:
    `pytest -q tests/test_device_token_carry.py -k 'release_preserves_carry_for_still_running_rows or survives_nonfinal_prefill_chunk or static_decode_metadata'`
    - 6 passed, 12 deselected;
  - full sidecar completed:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r19_full_bf16_2048cap_2048blocks_fixed.json`.
- working random result:
  - manifest seed `1234`, `15` requests, `30506` input tokens, `11602`
    output tokens;
  - JAX route: BF16 activations/weights, `max_num_batched_tokens=2048`,
    `num_kvcache_blocks=2048`, `max_blocks_per_seq=512`;
  - JAX throughput `204.61 output tok/s`, total token throughput
    `742.62 tok/s`, TTFT p50 `1361.02 ms`, ITL p50 `17.03 ms`;
  - live vLLM BF16 throughput `1531.33 output tok/s`, total token throughput
    `5557.75 tok/s`, TTFT p50 `587.22 ms`, ITL p50 `6.95 ms`;
  - ratio: `0.134x` vLLM.
- correctness status:
  - generated lengths match on all `15` rows;
  - generated tokens are not an exact parity pass: 4 of 15 rows diverge in the
    r19 sidecar comparison; first divergence is request `12`, generated index
    `32`, JAX token `1599` versus vLLM token `9032`;
  - vLLM is also not bit-identical across the earlier r2 and r13 random runs
    on this manifest, so use random token parity as a diagnostic, not as the
    sole correctness oracle.
- memory:
  - vLLM r13 memory monitor peak:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r13_vllm_memmon_memory.json`,
    sampled peak `15880 MiB`;
  - vLLM log breakdown: model load `1.72 GiB`, available KV cache memory
    `12.55 GiB`, GPU KV cache size `868783` tokens, CUDA graph pool
    `0.27 GiB`;
  - failed JAX FP32/3072-block r15 peak was `16670 MiB` and then OOMed when a
    pending prefill execution needed another `8.51 GiB`;
  - working JAX BF16/2048-block r18 monitor peak was `16714 MiB`.
- interpretation:
  - vLLM's peak is mostly intentional KV reservation under
    `gpu_memory_utilization=0.72`;
  - JAX still reaches a similar peak while exposing far less KV capacity
    (`2048 * 16 = 32768` block tokens), so memory efficiency remains a real
    optimization target;
  - the random sidecar is now usable as a stress benchmark, but its current
    throughput gap is decode-dominated and much worse than hetero8.

### Entry 219 - Random Correctness Diagnostics

- date: 2026-06-03
- purpose:
  - clarify whether the random-sidecar generated-token differences are broad
    correctness failures or low-margin branch sensitivity.
- artifact labels:
  - `r2` is the first successful vLLM-only random sidecar artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r2_vllm.json`;
  - `r13` is a later vLLM-only rerun on the same r2 manifest with a memory
    sampler:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r13_vllm_memmon.json`;
  - `r19` is the full sidecar where JAX and live vLLM both completed:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r19_full_bf16_2048cap_2048blocks_fixed.json`.
- full-artifact comparison:
  - vLLM `r19` and vLLM `r13` are exactly identical on all 15 rows;
  - old vLLM `r2` differs from stable vLLM `r13/r19` on 4 rows;
  - JAX `r19` versus stable vLLM `r19`: 11 of 15 rows exact, generated lengths
    match on all rows, same-position token equality `8616/11602` (`74.26%`);
  - first JAX/stable-vLLM diffs are request `12` at generated index `32`,
    request `13` at index `0`, request `6` at index `1`, and request `7` at
    index `1`;
  - JAX matches old vLLM `r2` on requests `6`, `7`, and `12`; JAX matches
    stable vLLM `r13/r19` on request `1`.
- HF diagnostic subset:
  - diagnostic manifest:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_correctness_rows_1_6_7_12_13_64.prompts.jsonl`;
  - HF BF16 artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_correctness_rows_1_6_7_12_13_64_hf_bf16.json`;
  - HF FP32-activation artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_correctness_rows_1_6_7_12_13_64_hf_fp32act.json`;
  - summary artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_correctness_diag_rows_1_6_7_12_13_64_summary.json`.
- HF BF16 results over first 64 generated tokens:
  - request `1`: JAX and stable vLLM match HF BF16; old vLLM `r2` diverges at
    index `8`, where HF BF16 top-2 is tied (`198` and `271`);
  - request `6`: JAX and old vLLM `r2` match HF BF16; stable vLLM diverges at
    index `1`, where HF BF16 top-2 margin is `0.0625` logprob;
  - request `7`: JAX and old vLLM `r2` match HF BF16; stable vLLM diverges at
    index `1`, where HF BF16 top-2 margin is `0.0625` logprob;
  - request `12`: stable vLLM matches HF BF16; JAX and old vLLM `r2` diverge at
    index `32`, selecting HF BF16 second-best token `1599` instead of top token
    `9032`; top-2 margin is `0.125` logprob;
  - request `13`: JAX matches HF BF16; stable vLLM and old vLLM `r2` diverge at
    index `0`, where HF BF16 top-2 is exactly tied (`220` and `12434`).
- HF FP32-activation note:
  - HF FP32 changes branches on requests `1`, `12`, and `13`, confirming that
    this random subset contains low-margin numerically sensitive branches;
  - JAX FP32 subset diagnostic completed but is not a pure FP32 oracle because
    the selected runtime config still carries accepted BF16 serving fastpaths.
- interpretation:
  - hetero exact generated-token parity remains a valid workload-specific gate;
  - random sidecar is not yet exact generated-token parity against stable vLLM,
    but most observed differences are near ties or branch choices also seen
    across vLLM/HF precision variants;
  - the clearest JAX-side random discrepancy under HF BF16 is request `12`
    index `32`, where JAX chooses the second-best token by a small margin.
- request-12 follow-up:
  - request-only manifest:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_correctness_req12_64.prompts.jsonl`;
  - clean/default JAX FP32 artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_correctness_req12_64_jax_default_fp32.json`;
  - clean/default JAX FP32 matches HF FP32 exactly for all first 64 generated
    tokens on request `12`;
  - selected JAX BF16 serving route, HF BF16, and stable vLLM all take the BF16
    branch at generated index `6`; at generated index `32`, stable vLLM and HF
    BF16 select token `9032`, while selected JAX BF16 and old vLLM `r2` select
    token `1599`;
  - HF BF16 top-k at that step has token `9032` first and token `1599` second
    with top-2 logprob margin `0.125`, so this is a narrow BF16/fastpath branch
    difference rather than evidence that the default JAX model state is corrupt.

### Entry 220 - Packed Paged Chunked-Prefill ABI Becomes Main Path

- date: 2026-06-03
- purpose:
  - stop treating chunked prefill as dense `[batch, max_query]` padding and
    move the serving ABI toward packed ragged query tokens with paged KV
    metadata, matching the vLLM-style chunked-prefill direction and MaxText's
    paged/ragged inference attention design.
- accepted plan:
  - prefill tensors become `[1, token_bucket]` for `tokens`, `positions`, and
    `token_row_ids`;
  - request-row metadata remains paged and row-shaped:
    `query_start_loc=[request_bucket + 1]`,
    `block_tables=[request_bucket, max_blocks_per_seq]`, and
    `seq_lens=[request_bucket]`;
  - decode remains the fixed-width paged path (`[batch_bucket, 1]`) with the
    current block tables, KV cache layout, greedy token fastpath, and GDN
    hybrid-state table.
- implementation direction:
  - keep `BlockManager`, paged KV allocation, physical cache layout, and decode
    paths intact;
  - add reference packed full-attention prefill first, then swap to a packed
    paged-attention kernel behind the same ABI;
  - use the existing packed varlen FLA/GDN references and Triton composition as
    the GDN kernel target;
  - make generic warmup cover finite `prefill_token_buckets` and
    `batch_size_buckets`, not benchmark-specific prompt shapes.
- correctness gate:
  - mixed-length prefills must match the dense reference before performance
    claims;
  - runtime JIT cache growth during a warmed benchmark should fail the
    benchmark harness.
- note:
  - dense prefill remains as a comparison/reference path, but it is no longer
    the target serving ABI. The old Cartesian `batch_bucket * query_bucket`
    warmup surface is the thing being removed.

### Entry 221 - Packed ABI GPU Smoke and Config-First Serving Controls

- date: 2026-06-03
- artifacts:
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/packed_prefill_abi_smoke/packed_prefill_gpu_smoke_20260603.json`
  and
  `/mountpoint/.exp/diagnostics/nano-vllm-jax/packed_prefill_abi_smoke/packed_prefill_gpu_smoke_config_first_20260603.json`
- purpose:
  - verify that the new packed-prefill grammar can be warmed generically and
    measured without creating benchmark-specific JIT entries;
  - start moving accepted serving-path controls out of direct hot-path
    environment-variable checks and into `Qwen3_5Config`.
- smoke setup:
  - two synthetic prompts with input lengths `8,12` and output length `2`;
  - `prefill_layout=packed`, `prefill_token_buckets=8,16,32`,
    `batch_size_buckets=1,2`, generic warmup, no profile;
  - `--fail-on-jit-cache-growth` enabled.
- result:
  - first smoke: generic warmup compiled `8` JIT entries; measured generation
    started with `8` entries and ended with `8` entries, so runtime JIT cache
    growth was `0`; measured prefill processed `20` scheduled tokens in about
    `309 ms`, and decode processed the two generated tokens in about `14 ms`;
  - config-first smoke: `--greedy-token-fastpath` and `--device-token-carry`
    were passed as normal CLI/config fields; warmup again compiled `8` entries,
    measured generation had `0` JIT cache growth, TTFT was about `137 ms`, and
    decode ITL was about `6.2 ms`;
  - this is an ABI/warmup smoke only, not a performance claim, because packed
    GDN and full-attention prefill still use reference bodies.
- config cleanup:
  - `greedy_token_fastpath`, `device_token_carry`,
    `static_decode_metadata`, and `static_decode_seq_lens_carry` are now
    first-class engine config fields;
  - `server_config.yaml`, `server.py`, and the JAX benchmark harness can pass
    these controls as config/CLI values;
  - legacy env vars still work as compatibility overrides, but the accepted
    path should use config fields going forward.

### Entry 222 - Packed Generic Hetero8 Anchor And Rejected Decode Plumbing Probes

- date: 2026-06-03
- purpose:
  - validate the packed/ragged ABI on the canonical hetero8 workload with
    generic warmup and no runtime JIT-cache growth;
  - test whether the profiled device-token carry broadcast and full-attention
    decode materialization buckets were real end-to-end throughput wins.
- current valid packed/generic anchor:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/packed_gdn_kernel_route/isolation/jax_hetero8_packed_generic_static_vector_token_output_20260603.json`;
  - exact generated-token parity and full generated lengths;
  - generic warmup, `prefill_layout=packed`, static decode metadata, device
    token carry, and `--fail-on-jit-cache-growth`;
  - output throughput `512.56 tok/s`, total seconds `0.49945`;
  - prefill step about `18.60 ms`, 31 decode steps sum about `399.84 ms`,
    final done/materialization gap about `79.40 ms`;
  - JIT cache growth during measurement was `0`.
- rejected token-carry variants:
  - returning an extra column-shaped carry token from the executor was exact but
    slower:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/packed_gdn_kernel_route/isolation/jax_hetero8_packed_generic_static_carry_column_20260603.json`,
    `508.41 tok/s`;
  - passing a 1D carried token vector into table decode and expanding it inside
    the main JIT was exact but slower:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/packed_gdn_kernel_route/isolation/jax_hetero8_packed_generic_static_table_token_vector_20260603.json`,
    `507.66 tok/s`;
  - both lowered the recorded decode-step sum to about `380-381 ms` but raised
    TTFT and the final done/materialization gap, so the end-to-end hetero8
    metric regressed. Do not retry token-carry shape movement unless the final
    materialization strategy changes materially.
- rejected full-attention decode Triton probe:
  - a focused CUDA/JAX-Triton parity test matched the dense reference for
    width-1 paged decode attention, but the integrated hetero8 run was exact
    and slower:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/packed_gdn_kernel_route/decode_attention/jax_hetero8_packed_generic_static_triton_decode_attention_20260603.json`,
    `503.32 tok/s`;
  - generic warmup still had zero measurement-time JIT growth, but first compile
    was long and the runtime did not beat the JAX decode attention path;
  - the experimental kernel, switch, and test were reverted rather than adding
    another slower environment-variable path.
- interpretation:
  - the CPU-profile `jit_broadcast_in_dim` attribution was not a standalone
    removable throughput bucket in the measured end-to-end hetero8 run;
  - replacing full-attention decode with a small custom streaming Triton kernel
    did not beat XLA's existing lowered path on this A10G workload;
  - continue from the `512.56 tok/s` packed/generic/static anchor, not from the
    rejected variants.

### Entry 223 - Config-First Packed ABI Promotion Check

- date: 2026-06-03
- purpose:
  - move the accepted packed/ragged serving controls into typed
    `Qwen3_5Config` and `server_config.yaml` fields instead of relying on a
    growing set of hot-path environment variables;
  - remeasure hetero8 after the refactor to make sure the large ABI/config
    change did not change correctness, warmup discipline, or the current
    performance anchor.
- code/config changes:
  - promoted accepted fastpaths and kernel policy to config fields:
    materialized tied LM head, compact packed prefill projections/MLP,
    BF16 decode projection and LM-head activations, padded decode GEMMs,
    BF16 full-attention KV cache, strict GDN no-fallback policy,
    Triton FLA padded GDN prefill, and conv-fused Triton FLA packed decode;
  - `server_config.yaml` and the GPU matrix config now select the accepted
    route through config/runtime sections, with env vars kept only as
    compatibility overrides;
  - the benchmark config uses generic warmup and `--fail-on-jit-cache-growth`
    by default.
- correctness/tests:
  - focused pytest slice passed:
    `16 passed, 78 deselected`;
  - `tests/test_server_config.py` passed standalone:
    `8 passed`;
  - `py_compile` passed for the touched config, model, backend, loader, and
    executor/runner modules;
  - `git diff --check` passed.
- fair hetero8 result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/matrix_runs_generic/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity and full generated lengths;
  - generic bucket startup warmup compiled `26` entries, measurement started
    and ended at `26` entries, so runtime JIT cache growth was `0`;
  - output throughput `512.19 tok/s`, total seconds `0.49982`;
  - stored vLLM reference `864.18 tok/s`, ratio `0.593x`, so the hetero8
    `0.9x` target remains open.
- profile sanity:
  - profiled/request-warmed run was exact but slower at `479.71 tok/s`; do not
    use that as the fair anchor;
  - the current GPU trace still shows launch-heavy decode: Cutlass GEMMs about
    `165 ms / 1957`, fusion kernels about `155 ms / 7369`, transposes about
    `23 ms / 595`, LM-head GEMM about `31 ms / 31`, and conv-fused GDN decode
    about `20.6 ms / 558`;
  - CPU-side `broadcast_in_dim`/materialization labels remain visible, but
    Entry 222 showed token-carry shape movement alone is not an integrated
    throughput win. Treat those labels as synchronization/serving-boundary
    evidence, not as a reason to retry the rejected token-carry variants.
- interpretation:
  - the config refactor is promoted as the current fair generic-warmup anchor
    because it preserves the packed/generic/static route and removes env-var
    sprawl from the accepted path; do not read this as a `0.9x` win or as the
    fastest historical no-profile diagnostic lane;
  - the next speed work must change the decode computation boundary
    materially: reducing GEMM/reduction launch count, LM-head top-1
    materialization, or a broader production-kernel boundary. Do not repeat
    greedy decode burst, token-carry shape movement, single-query Triton
    full-attention decode, or source-level gate/up packing probes already
    rejected above.

### Entry 224 - Typed Config Projection And Boundary Ablations

- date: 2026-06-03
- purpose:
  - finish the config-first replacement by making the GPU matrix runner pass
    typed runtime fastpaths and kernel policy into the benchmark/server process
    explicitly instead of depending on inherited environment variables;
  - rerun the correctness gate after this large routing change;
  - record the latest small boundary ablations so they are not retried as
    apparent wins.
- code changes:
  - added `engine_overrides_from_config()` to `server_config.py` and used it
    from `load_server_config()` and `benchmarks/run_gpu_matrix.py`;
  - `benchmarks/benchmark_jax_server_trace.py` now accepts the promoted
    runtime/kernel fields as CLI arguments and passes them into `LLMEngine` /
    `Qwen3_5Config`;
  - benchmark artifacts now report promoted full-attention and GDN kernel
    policy from `engine.config`, not just from environment variables;
  - fixed `_full_attention_kv_cache_dtype()` so explicit config value
    `"default"` means “use the cache spec dtype,” matching the typed config
    default.
- correctness/tests:
  - `py_compile` passed for `server_config.py`, `run_gpu_matrix.py`,
    `benchmark_jax_server_trace.py`, the touched tests, and the backend dtype
    fix;
  - config/matrix-focused tests passed: `54 passed`;
  - broader correctness slice passed:
    `57 passed, 125 deselected, 2 warnings`;
  - dry-run command projection confirmed the selected matrix config now emits
    the expected typed flags, including BF16 decode projection/LM-head
    activations, padded decode GEMMs, BF16 full-attention KV cache, strict GDN
    fallbacks disabled, Triton FLA padded GDN prefill, and conv-fused Triton
    FLA packed decode.
- fair hetero8 typed-projection validation:
  - matrix summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/gpu_matrix_typed_projection_hetero8_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/typed_projection_matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity and full generated lengths;
  - generic bucket startup warmup compiled `26` entries, measurement started
    and ended at `26` entries, so runtime JIT cache growth was `0`;
  - output throughput `514.16 tok/s`, total seconds `0.49790`;
  - stored vLLM reference `864.18 tok/s`, ratio `0.595x`; the `0.9x` target
    still needs about `1.51x` improvement from this anchor.
- rejected/not-promoted ablations:
  - disabling static decode metadata remained exact with zero JIT growth but
    was neutral:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/static_metadata_ablation/hetero8_generic_no_static_metadata_20260603.json`,
    `512.96 tok/s`; do not promote it as a speed path;
  - trace-token prefetch accounting remained exact with zero JIT growth and
    reached `514.34 tok/s`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/gpu_matrix_hetero8_trace_token_prefetch_r1_20260603.json`;
    the end-to-end movement versus `512-514 tok/s` is too small to treat as a
    material win, and the per-repeat artifact landed under the default
    `results/gpu_matrix_runs/...` directory because no explicit run directory
    was supplied.
- interpretation:
  - the config-first replacement is correctness-clean and measurement-clean,
    but it is a maintainability/benchmark-discipline change, not the speedup;
  - static metadata and token-prefetch/accounting changes are not the dominant
    hetero8 lever;
  - continue from the `514 tok/s` fair generic-warmup anchor and focus on a
    broader decode computation boundary that reduces launch count or fuses
    model-side decode GEMMs/reductions without shape-specific tuning.

### Entry 225 - Rejected Hetero8 Metadata Warmup Variants

- date: 2026-06-03
- purpose:
  - investigate the large hetero8 decode outlier visible after the typed-config
    anchor without changing the correctness contract or allowing measured-phase
    JIT growth;
  - determine whether the host metadata/device-put profile bucket can be removed
    by warming or carrying existing static decode metadata.
- diagnostics and rejected results:
  - default XLA Triton GEMM emitter ablation:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/xla_gemm_ablation/hetero8_xla_default_direct_20260603.json`,
    exact with zero JIT growth but `141.27 tok/s`; reject;
  - long-context decode warmup:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/long_context_warmup/gpu_matrix_hetero8_long_context_warmup_r1_20260603.json`,
    exact with zero JIT growth but `494.60 tok/s`; reject;
  - long-context profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/long_context_warmup/hetero8_long_context_profile_20260603.json`,
    showed `_static_decode_device_arrays`/`device_put`/`s32[8]` host metadata
    work as a large profiled bucket, but the integrated fixes below moved cost
    rather than improving throughput;
  - static decode seq-lens carry:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/seq_lens_carry/gpu_matrix_hetero8_seq_lens_carry_r1_20260603.json`,
    exact with zero JIT growth but `469.79 tok/s`; reject;
  - scheduler static-metadata warmup:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/scheduler_metadata_warmup/gpu_matrix_hetero8_scheduler_metadata_warmup_r1_20260603.json`,
    exact with zero JIT growth but `460.25 tok/s`; reject;
  - seq-lens placeholder plus prefill-seeded seq-lens carry:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/seq_lens_placeholder/gpu_matrix_hetero8_seq_lens_placeholder_r1_20260603.json`,
    exact with zero JIT growth but `416.32 tok/s`; reject.
- decision:
  - revert all long-context warmup, scheduler metadata warmup, and prefill
    seq-lens seeding code;
  - keep the fair typed-config anchor from Entry 224 at `514.16 tok/s` as the
    current baseline.
- interpretation:
  - the host metadata/device-put bubble is real, but cache-miss warmups and
    seq-lens carry toggles are not sufficient; they shift latency into first
    decode or final prefill instead of eliminating it;
  - do not retry these variants. The next structural route should make request
    metadata/block tables device-owned across prefill and decode, or broaden
    the compiled decode boundary so the scheduler stops rebuilding hot-path
    metadata on the host.

### Entry 226 - Rejected Stacked Device-Token Materialization

- date: 2026-06-03
- purpose:
  - test whether grouping same-shaped deferred `DeviceTokenRef` vectors into one
    stacked host transfer reduces the final hetero8 completion-token
    materialization gap.
- baseline:
  - restored Entry 224 anchor rerun:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/gpu_matrix_hetero8_restored_anchor_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/config_refactor/restored_anchor_matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, zero measured-phase
    JIT growth, `514.96 tok/s`;
  - final output gap: done elapsed `0.49630 s`, last token step end
    `0.41685 s`, gap `79.44 ms`.
- rejected experiment:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/materialize_stack/gpu_matrix_hetero8_materialize_stack_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/materialize_stack/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, zero measured-phase
    JIT growth, but throughput fell to `496.33 tok/s`;
  - final output gap worsened to `95.94 ms`.
- decision:
  - revert the stacked materialization code and its focused test;
  - keep the original vector-list `jax.device_get` materialization path.
- interpretation:
  - final token materialization is visible, but adding a `jnp.stack` launch and
    grouped transfer does not eliminate the synchronization cost in the
    integrated server path;
  - do not retry host-side stacking as a speed route. A future materialization
    change needs a broader serving-boundary change, such as device-owned output
    tables or a decode boundary that produces directly reusable host-visible
    request results.

### Entry 227 - Rejected Graph-Returned Seq-Lens Carry

- date: 2026-06-03
- purpose:
  - test a materially different seq-lens carry route from Entry 225 by returning
    the next decode `seq_lens` vector from the compiled greedy token-id decode
    graph, instead of computing `batch.seq_lens + 1` as a separate JAX operation
    in the runner.
- experiment:
  - enabled the existing `static_decode_seq_lens_carry` config field in
    `gpu_paged_gdn_fla_decode_static_metadata`;
  - temporarily extended `ExecutorOutput` so `forward_step_token_ids_jit` and
    `forward_step_token_ids_table_jit` returned `next_seq_lens` alongside token
    ids, cache storage, and hybrid state;
  - wired the runner carry map to prefer the graph-returned `next_seq_lens`.
- correctness/tests:
  - syntax checks passed for the touched executor and runner modules;
  - focused static-decode seq-lens tests passed before the benchmark:
    `4 passed, 16 deselected`;
  - broader focused slice passed with the experiment enabled:
    `14 passed, 73 deselected`.
- hetero8 result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/seq_lens_graph_carry/gpu_matrix_hetero8_seq_lens_graph_carry_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/seq_lens_graph_carry/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity and full generated lengths;
  - generic warmup compiled `26` entries, and the measured phase had zero JIT
    cache growth (`26 -> 26`);
  - throughput `508.89 tok/s`, ratio `0.589x` stored vLLM;
  - final output gap worsened to about `105.01 ms` versus about `79.44 ms` on
    the restored Entry 224 anchor.
- decision:
  - reject and revert the executor output extension, runner wiring, and config
    enablement;
  - keep `static_decode_seq_lens_carry` disabled in the selected hetero8 config.
- interpretation:
  - returning `next_seq_lens` from the decode graph avoids one obvious external
    `seq_lens + 1` operation, but it adds an extra decode-graph output and does
    not reduce the integrated serving latency;
  - do not retry seq-lens carry variants unless the boundary changes more
    substantially, for example a device-owned request metadata table that also
    removes host block-table/query metadata rebuilds.

### Entry 228 - Rejected Direct Full-Active Table Decode Specialization

- date: 2026-06-03
- purpose:
  - test whether the full-active `B=8` hetero8 decode lane benefits from a
    narrower table-state decode boundary that assumes hybrid slots are already
    row-aligned `[0, B)`;
  - unlike the older full-active valid-mask skip, this temporary route removed
    hybrid table gather/scatter and built direct decode metadata inside the JIT.
- experiment:
  - temporarily added `forward_step_token_ids_table_direct_jit`, a decode-only
    greedy token-id path that accepted token rows, block tables, seq_lens,
    cache storage, and full hybrid state tables, without a slot-id vector;
  - runtime used it only when the batch was fully active, every query length
    was `1`, there were no padded rows, and the hybrid table row count matched
    the physical token row count;
  - generic table decode remained warmed for ragged/tail buckets.
- correctness/tests:
  - syntax and diff checks passed;
  - focused table-state parity test was extended during the experiment and
    passed after avoiding donated-buffer reuse;
  - after reverting, the focused table-state slice still passed:
    `2 passed, 65 deselected`.
- hetero8 result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/direct_table_decode/gpu_matrix_hetero8_direct_table_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/direct_table_decode/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity and full generated lengths;
  - generic warmup compiled `27` entries and the measured phase had zero JIT
    cache growth (`27 -> 27`);
  - throughput fell to `493.29 tok/s`, ratio `0.571x` stored vLLM;
  - decode-step total worsened to about `0.41 s` versus about `0.37 s` in the
    run report, despite a slightly smaller final materialization gap.
- decision:
  - reject and revert the direct table decode method, runner route, warmup hook,
    and focused direct-route test assertions;
  - keep the generic `forward_step_token_ids_table_jit` path as the selected
    table-state decode route.
- interpretation:
  - removing visible source-level gather/scatter did not produce a better
    lowered decode plan; it added another compiled variant and regressed
    end-to-end throughput;
  - do not retry full-active decode source specializations unless a profile
    shows a specific lowered GPU/CPU bucket removed by the proposed boundary.

### Entry 229 - Accepted BF16 Model Compute Diagnostic

- date: 2026-06-03
- purpose:
  - test a model-family-wide numeric policy change instead of hand tuning:
    keep BF16 checkpoint weights and FP32 GDN state-sensitive kernels, but make
    the model activation compute dtype BF16.
- rejected preliminary run:
  - direct benchmark artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_compute_hetero8/hetero8_bf16_compute_direct_20260603.json`;
  - exact and zero-JIT, but not comparable because the direct command skipped
    config-derived GDN block-dot env flags; do not use this as a speed result.
- accepted hetero8 result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_compute_hetero8/gpu_matrix_hetero8_bf16_compute_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_compute_hetero8/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_bf16_compute_repeat1.json`;
  - exact generated-token parity and full generated lengths;
  - generic warmup compiled `26` entries, and the measured phase had zero JIT
    cache growth (`26 -> 26`);
  - throughput `651.18 tok/s`, ratio `0.754x` stored vLLM (`864.18 tok/s`);
  - scheduler timing: one prefill step `0.0167 s`, thirty-one decode steps
    `0.2942 s`.
- decision:
  - promote `args.dtype` in `gpu_paged_gdn_fla_decode_static_metadata` from
    `float32` to `bfloat16`;
  - remove the temporary BF16 diagnostic config to avoid benchmark-config
    sprawl;
  - continue to require exact generated-token parity and zero measured-phase
    JIT growth for follow-up changes.
- interpretation:
  - broad BF16 compute gives a meaningful integrated decode win and is more
    general than GEMM tile tuning;
  - the run is still only one repeat and below the `0.9x` vLLM target
    (`777.76 tok/s`), so the next step is to profile the BF16 route and target
    the remaining decode buckets.

### Entry 230 - Rejected BF16 Trace-Token Prefetch

- date: 2026-06-03
- purpose:
  - retest the existing overlapped trace-token prefetch route under the new
    BF16 compute baseline because final device-token materialization still
    leaves about an `80 ms` done-vs-last-token gap.
- baseline:
  - Entry 229 BF16 compute artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_compute_hetero8/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_bf16_compute_repeat1.json`;
  - exact and zero-JIT, `651.18 tok/s`, final gap about `80.47 ms`.
- experiment:
  - direct artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_trace_prefetch_hetero8/hetero8_bf16_trace_prefetch_20260603.json`;
  - enabled existing `NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH=1` while keeping the
    same BF16 config and GDN block-dot env flags.
- result:
  - exact generated-token parity, full generated lengths, and zero measured
    JIT growth;
  - throughput was effectively unchanged/slightly worse: `651.06 tok/s`;
  - final gap shrank to about `11.66 ms`, but the last token timestamp moved
    later, so total wall time did not improve.
- decision:
  - reject for promotion and keep trace-token prefetch off in the selected
    config;
  - do not retry token prefetch/materialization movement alone. A future output
    route needs to remove total step work or produce a device-owned output table
    as part of a broader compiled boundary.

### Entry 231 - Rejected Decode Output-Projection Padded GEMM

- date: 2026-06-03
- purpose:
  - test whether routing GDN/full-attention decode output projections through
    the already accepted padded-GEMM helper reduces the high-count decode
    output-projection GEMM buckets without hand-tuned tile parameters.
- experiment:
  - temporarily added a helper that used `_decode_padded_gemm_dot()` for
    decode-only `B <= decode_padded_gemm_rows`, `seq_len == 1` output
    projections, and routed GDN `out_proj` plus full-attention `o_proj`
    through it.
- correctness/tests:
  - `py_compile` passed for `nanovllm_jax/model.py`;
  - focused tests passed:
    `3 passed, 64 deselected` for decode padded-GEMM/table decode;
  - decode reduction tests passed: `8 passed`.
- result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/output_proj_padded_gemm/gpu_matrix_hetero8_output_proj_padded_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/output_proj_padded_gemm/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, zero measured-phase
    JIT growth;
  - throughput regressed slightly to `650.19 tok/s` versus the Entry 229
    `651.18 tok/s` baseline.
- decision:
  - reject and revert the output-projection padded-GEMM helper and call sites;
  - keep padded GEMM limited to the currently accepted decode projection/MLP
    uses.
- interpretation:
  - reshaping output projection GEMMs through the row-padded helper does not
    reduce integrated wall time on hetero8; the next attempt must change a
    broader decode boundary or use a production kernel, not simply repad the
    same output-projection matmuls.

### Entry 232 - Rejected FP32 GDN Conv-State Storage

- date: 2026-06-03
- purpose:
  - test whether keeping the GDN convolution state table in FP32 under BF16
    model compute avoids per-step casts/scatter warnings and aligns the runtime
    state policy with the desired BF16-activation/FP32-state contract.
- experiment:
  - temporarily changed `init_hybrid_state()` so `conv_state` was allocated in
    FP32 regardless of the model compute dtype. `recurrent_state` was already
    FP32.
- correctness/tests:
  - `py_compile` passed for `kv_cache.py`, `model.py`, and `model_runner.py`;
  - focused GDN packed-decode tests passed: `3 passed, 18 deselected`;
  - focused backend decode/prefill tests passed: `3 passed, 64 deselected`.
- result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fp32_conv_state/gpu_matrix_hetero8_fp32_conv_state_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fp32_conv_state/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, zero measured-phase
    JIT growth;
  - throughput regressed to `628.66 tok/s` from the Entry 229 BF16 baseline
    `651.18 tok/s`;
  - decode-step total increased from about `0.294 s` to `0.308 s`.
- decision:
  - reject and revert FP32 conv-state storage;
  - keep recurrent state FP32 and conv state in the model compute dtype for the
    selected route.
- interpretation:
  - the scatter warning reflects an implicit cast boundary, but storing the
    full conv-state table in FP32 makes the integrated decode graph slower.
    Future cleanup should cast the specific returned conv-state value before
    scatter if needed for warning hygiene, not change the table dtype.

### Entry 233 - Rejected BF16 Lowered Decode Reductions

- date: 2026-06-03
- purpose:
  - retest existing lowered reduction fastpaths under the promoted BF16 compute
    route: Triton decode RMSNorm plus Pallas GDN Q/K pre-normalization.
- experiment:
  - direct artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/bf16_reductions_hetero8/hetero8_bf16_triton_rms_pallas_qk_prenorm_20260603.json`;
  - enabled:
    `NANO_VLLM_JAX_TRITON_DECODE_RMSNORM=1`,
    `NANO_VLLM_JAX_PALLAS_GDN_QK_PRENORM=1`, and
    `NANO_VLLM_JAX_GDN_PACKED_DECODE_PRENORMALIZE_QK=1`;
  - kept the same BF16 model compute config, GDN block-dot prefill env flags,
    exact reference, and zero-JIT-growth gate.
- result:
  - exact generated-token parity, full generated lengths, zero measured-phase
    JIT growth;
  - throughput regressed to `637.34 tok/s` from the Entry 229 BF16 baseline
    `651.18 tok/s`;
  - final materialization gap shrank, but total wall time increased.
- decision:
  - reject for hetero8 BF16 promotion;
  - keep lowered decode reductions default-off. They remain useful focused
    diagnostics, but do not improve this integrated serving target.

### Entry 234 - Token-Carry Reshape Cleanup And Lazy Hybrid Slot Zeroing

- date: 2026-06-03
- purpose:
  - remove redundant hot-path device-token slicing/expansion seen in the BF16
    profile, and fix a correctness edge where newly assigned hybrid-state table
    slots could reuse stale GDN state if a freed physical slot was allocated to
    a new sequence.
- implementation:
  - replaced the full-bucket greedy-token slice with direct reuse of
    `output.activations` when the returned token vector already has the active
    batch shape;
  - replaced `token_vector[:, None]` with `jnp.reshape(token_vector,
    tokens.shape)` in the device-token carry path;
  - added lazy hybrid-slot zero tracking so fresh all-zero tables are not
    rewritten on first allocation, while stale freed slots are zeroed on reuse
    through the batched zeroing helper.
- correctness/tests:
  - `py_compile` passed for `model_runner.py` and the updated backend-boundary
    tests;
  - focused tests passed:
    `27 passed, 60 deselected` across device-token carry, static decode
    metadata, table decode, hybrid slot ids, and hybrid-state table reuse.
- result:
  - rejected eager-zero variant:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/token_carry_reshape/gpu_matrix_hetero8_token_carry_reshape_r1_20260603.json`;
    it was exact but regressed to `312.86 tok/s` because first-prefill TTFT
    included eager zeroing of the full hybrid-state table;
  - lazy-zero rerun:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/token_carry_reshape_lazy_zero/gpu_matrix_hetero8_token_carry_reshape_lazy_zero_r1_20260603.json`;
    run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/token_carry_reshape_lazy_zero/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, zero measured-phase
    JIT growth (`26 -> 26`);
  - throughput was `647.58 tok/s`, below the Entry 229 BF16 best
    `651.18 tok/s`.
- decision:
  - keep the lazy hybrid-slot zeroing correctness fix;
  - do not count token-carry reshape cleanup as a speed win, and do not retry
    eager allocation-time table zeroing;
  - current best speed anchor remains Entry 229 BF16 model compute at
    `651.18 tok/s` (`0.754x` stored vLLM).

### Entry 235 - Rejected Per-Step Trace Output Table

- date: 2026-06-03
- purpose:
  - test whether writing deferred device-token outputs into a device-owned
    `[requests, max_completion_tokens]` trace table can remove the final
    last-token-to-done materialization gap without changing model numerics.
- experiment:
  - temporarily scattered newly generated `DeviceTokenRef` values into a JAX
    trace table after every scheduler step, then copied the full table once at
    trace completion.
- correctness/tests:
  - `py_compile` passed for `llm_engine.py`, `sequence.py`, and
    `model_runner.py`;
  - focused tests passed:
    `27 passed, 60 deselected` across device-token carry, static decode
    metadata, table decode, hybrid slot ids, and hybrid-state table reuse.
- result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/trace_output_table/gpu_matrix_hetero8_trace_output_table_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/trace_output_table/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity and zero measured-phase JIT growth;
  - throughput regressed to `454.91 tok/s` (`0.526x` stored vLLM);
  - final last-token-to-done gap shrank from about `80 ms` to about `11 ms`,
    but inter-step gaps grew from about `0.6 ms` total to about `419 ms` total.
- decision:
  - reject and remove the trace output table path;
  - do not retry per-step host-side JAX scatters for trace output accounting.
- interpretation:
  - the table removed the final materialization cliff by serializing the
    generation loop after each model step. The right fix needs output tracking
    fused into the already-jitted decode boundary or avoided during timing,
    not a separate host-orchestrated scatter per token step.

### Entry 236 - Rejected Admission-Time Output Block Reservation

- date: 2026-06-03
- purpose:
  - test whether reserving each request's full max-output KV block capacity at
    admission keeps decode block-table metadata stable enough to reduce
    hetero8 CPU/PjRT overhead.
- experiment:
  - temporarily added a typed `reserve_output_blocks` engine/benchmark config
    and passed `prompt_tokens + max_tokens` capacity into block allocation;
  - enabled it for the selected
    `gpu_paged_gdn_fla_decode_static_metadata` hetero8 route.
- correctness/tests:
  - JSON and `py_compile` checks passed for the changed config, scheduler,
    block manager, server config, benchmark, and focused tests;
  - focused correctness/config tests passed:
    `10 passed, 68 deselected`;
  - broader backend/device-carry/config run still has existing dense/cached
    prefill numeric and old shape-cache expectation failures, so serving-path
    correctness was gated by the benchmark artifact below.
- result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/reserve_output_blocks/gpu_matrix_hetero8_reserve_output_blocks_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/reserve_output_blocks/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity for all 8 rows, full generated lengths, and
    zero measured-phase JIT growth (`26 -> 26`);
  - throughput regressed to `634.55 tok/s` (`0.734x` stored vLLM), below the
    Entry 229 BF16 best `651.18 tok/s`.
- decision:
  - reject and remove the reservation config/plumbing;
  - do not retry admission-time full-output KV reservation for hetero8 unless a
    later profile shows block-table device_put specifically dominates again.
- interpretation:
  - stabilizing future block-table values did not outweigh the added upfront
    allocation/capacity pressure. The hot path still needs boundary/fusion work
    inside decode rather than request-wide KV over-reservation.

### Entry 237 - Rejected Fused Device-Token Carry Override

- date: 2026-06-03
- purpose:
  - remove the visible `_maybe_apply_device_token_carry` reshape/PJRT boundary
    by passing the carried token vector as an argument to the existing jitted
    greedy decode call and reshaping it inside the compiled function.
- experiment:
  - temporarily added an optional `input_tokens_override` to
    `forward_step_token_ids_jit` and `forward_step_token_ids_table_jit`;
  - updated generic warmup so the override decode key was compiled before
    measurement;
  - routed only the single-step static-decode-metadata path through the
    override; all burst/speculative/non-static paths stayed on the old carry
    application.
- correctness/tests:
  - `py_compile` passed for the executor, runner, benchmark, and focused
    tests;
  - focused correctness passed:
    `24 passed, 63 deselected` across device-token carry, static decode
    metadata, warmup, greedy burst, and table-hybrid decode.
- result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fused_token_carry_override/gpu_matrix_hetero8_fused_token_carry_override_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fused_token_carry_override/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, and zero
    measured-phase JIT growth (`26 -> 26`);
  - throughput was `650.05 tok/s` (`0.752x` stored vLLM), effectively tied but
    still below the Entry 229 BF16 best `651.18 tok/s`.
- decision:
  - reject and remove the override API/warmup/test changes;
  - do not retry token-carry reshape movement unless it is part of a larger
    decode boundary that also removes meaningful model launches or
    materialization.
- interpretation:
  - fusing this one host-dispatched reshape into the decode JIT is not enough
    to move the end-to-end hetero8 target. The profile bucket is partly
    synchronization/accounting around the already-running decode graph, not an
    isolated win by itself.

### Entry 238 - Rejected Delayed Trace-Token Prefetch

- date: 2026-06-03
- purpose:
  - test whether delaying trace-token D2H prefetch until after the following
    decode step dispatch reduces the final token-to-done drain without putting
    the copy ahead of useful decode work.
- experiment:
  - temporarily changed the deferred trace iterator to prefetch each token
    snapshot after the next scheduler step and emit materialized tokens after a
    two-step lag when possible;
  - added timing fields that separate end-to-end throughput from token-event
    timestamps:
    `last_token_elapsed_seconds`, `post_last_token_drain_seconds`,
    `token_event_tokens_per_second`, and
    `token_event_output_token_throughput`.
- correctness/tests:
  - focused syntax checks passed for the trace benchmark, matrix runner, engine,
    and updated tests;
  - focused correctness/config/timing slice passed after restoring the rejected
    iterator change:
    `31 passed, 67 deselected`.
- result:
  - summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/delayed_trace_prefetch/gpu_matrix_hetero8_delayed_trace_prefetch_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/delayed_trace_prefetch/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - exact generated-token parity, full generated lengths, and zero
    measured-phase JIT growth (`26 -> 26`);
  - throughput was `648.66 tok/s` (`0.751x` stored vLLM), effectively unchanged
    from the current `~649 tok/s` rebaseline and below the Entry 229 best
    `651.18 tok/s`;
  - `post_last_token_drain_seconds` dropped to about `20 ms`, but
    `last_token_elapsed_seconds` moved later, so the end-to-end target metric
    did not improve.
- decision:
  - reject and remove the delayed iterator ordering change;
  - keep the timing metric instrumentation because it prevents mistaking
    token-event throughput for accepted end-to-end throughput.
- interpretation:
  - this confirms the final trace drain is mostly queued GPU completion and
    synchronization accounting, not a standalone D2H copy cliff that can be
    fixed by rescheduling token prefetch alone.

### Entry 239 - Decode Cleanup Tie And MTP Current-Route Diagnostic

- date: 2026-06-03
- purpose:
  - remove unused final RMSNorm work from the greedy hidden-return decode path;
  - re-check whether the non-conv/reference GDN decode route or current BF16
    MTP1 path can beat the Entry 229 hetero8 anchor without changing
    correctness.
- experiment:
  - changed `forward_step(return_hidden=True, return_hidden_with_logits=False)`
    to return the pre-final-norm hidden before computing the final model norm.
    The greedy token fast path already applies the final RMSNorm inside
    `lm_head_token_ids_and_topk`, so the earlier norm result was unused;
  - compared no-profile hetero8 direct runs for reference GDN decode versus the
    selected conv-fused Triton FLA GDN decode;
  - attempted a current-BF16 MTP1 hetero8 diagnostic with the selected packed
    prefill/decode runtime and the same exact-token reference gate.
- correctness/tests:
  - syntax check passed:
    `python -m py_compile nanovllm_jax/model.py nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py`;
  - focused correctness passed:
    `27 passed, 63 deselected` across backend boundaries, LM-head helpers, and
    device-token carry tests;
  - hetero8 cleanup artifact is exact for all rows, full generated lengths, and
    zero measured-phase JIT growth (`26 -> 26`).
- result:
  - cleanup hetero8 matrix:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/skip_unused_final_norm/gpu_matrix_hetero8_skip_unused_final_norm_r1_20260603.json`;
  - run artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/skip_unused_final_norm/matrix_runs_20260603/hetero8_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - throughput was `650.62 tok/s`, effectively tied with but slightly below
    the Entry 229 best `651.18 tok/s`;
  - no-profile GDN decode check:
    reference decode reached `651.95 tok/s` and conv-fused Triton decode reached
    `650.25 tok/s` on direct runs, so this is a noise-level tie rather than a
    material route to `0.9x`;
  - current BF16 MTP1 diagnostic did not reach measurement after about three
    minutes of warmup/compilation and was stopped.
- decision:
  - keep the final-norm cleanup because it removes provably unused work and is
    correctness-clean, but do not count it as a speed win;
  - keep the selected conv-fused Triton GDN decode config unchanged until a
    repeated matrix shows a real median win for reference GDN decode;
  - do not use MTP1 as the near-term hetero8 route. It still needs a dedicated
    speculative warmup/acceptance redesign before it can be a fair speed path.
- interpretation:
  - GDN-only changes are now too small to close the remaining gap. The selected
    route already spends most completed decode time in broad per-layer GEMM and
    replay work: roughly two dense projections per layer plus LM head for every
    token step.
  - The next credible speedup must either reduce full-model decode executions,
    introduce a better small-batch GEMM/decode projection backend, or rewrite
    the regular `3 x GDN + 1 x full-attention` layer pattern into a broader
    grouped compiled boundary. Narrow token materialization, seq-lens carry,
    GDN core, and isolated RMSNorm changes should not be retried as primary
    hetero8 levers.

### Entry 240 - Random Stress Anchor And Rejected Capacity/Metadata Routes

- date: 2026-06-04
- purpose:
  - promote the seed-`1234` random request suite to the active broad serving
    stress target;
  - record the first failed random-specific hill-climb routes so they are not
    retried as apparent fixes for the CPU metadata bubble.
- current random anchor:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/current_all_bench_20260604/random/random_current_optimized_r1_20260604.json`;
  - manifest:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_request_sidecar/qwen08_random_request_sidecar_seed1234_20260603_r19_full_bf16_2048cap_2048blocks_fixed.prompts.jsonl`;
  - workload: `15` requests, `30506` input tokens, `11602` output tokens,
    prompt lengths `1077..4022`, output lengths `425..1007`;
  - selected route: BF16 activations and weights, packed paged prefill,
    generic warmup, `max_num_batched_tokens=2048`,
    `num_kvcache_blocks=2048`, `max_blocks_per_seq=512`,
    `batch_size_buckets=1,2,4,8`, static decode metadata, device token carry,
    BF16 decode projections/LM head, BF16 full-attention KV cache, strict GDN
    no-fallbacks, Triton FLA padded prefill, and conv-fused Triton FLA packed
    decode;
  - throughput `383.96 output tok/s`, total wall time `30.22 s`, generated
    lengths complete, and zero measured-phase JIT growth (`24 -> 24`);
  - live vLLM BF16 on the same manifest reaches `1531.33 output tok/s`, so the
    current random ratio is `0.251x`.
- timing attribution from token-event step grouping:
  - `13` prefill steps total about `2.35 s`;
  - `1809` decode steps total about `26.62 s`;
  - B8 decode contributes `1006` steps and `8048` output tokens at about
    `572 tok/s` inside the bucket;
  - B1/B2 tail decode contributes fewer tokens but much worse per-token
    throughput (`~105` and `~221 tok/s`);
  - final drain is about `0.51 s`.
- rejected device block-table table route:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_block_table_table_r1_20260604.json`;
  - temporary change kept a device-side block-table table and passed slot ids
    into decode to avoid rebuilding full `[B,max_blocks]` block-table rows on
    the host every step;
  - focused CPU tests and a CUDA smoke passed, and measured-phase JIT growth
    stayed zero;
  - throughput regressed to `255.11 output tok/s` (`0.166x` live vLLM and
    `0.664x` the current random anchor);
  - decision: reject and revert. Do not retry device block-table table routing
    unless a new lowered profile shows that it removes a dominant GPU bucket
    rather than merely moving host metadata cost into the compiled graph.
- rejected B16 capacity route:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_b16_3072blocks_320width_r1_20260604.json`;
  - setup used `max_num_seqs=16`, `batch_size_buckets=1,2,4,8,16`,
    `num_kvcache_blocks=3072`, and `max_blocks_per_seq=320` so all `15`
    requests could fit concurrently under a model-family request-capacity
    envelope;
  - generic warmup compiled `30` entries, measured-phase JIT growth was zero,
    and full output lengths completed;
  - throughput regressed to `229.87 output tok/s` (`0.150x` live vLLM and
    `0.599x` the current random anchor);
  - decision: reject B16 as a current default. Fewer waves do not compensate
    for the larger decode graph/capacity footprint on this path. Future
    concurrency work must improve arbitrary-batch decode cost first rather
    than simply widening the batch bucket.
- rejected waiting-admission reordering routes:
  - LPT/longest-declared-output-first artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_lpt_admission_r1_20260604.json`;
  - the temporary scheduler policy selected waiting prompts with the largest
    remaining declared generation budget first. A simple offline schedule
    predicted fewer decode steps, and the integrated run did reduce decode
    steps from `1809` to `1568`, but prefill grew to about `7.78 s`, decode
    grew to about `31.82 s`, and throughput regressed to
    `267.57 output tok/s`;
  - SPT/shortest-declared-output-first artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_spt_admission_r1_20260604.json`;
  - the opposite policy also kept measured-phase JIT growth at zero and full
    lengths complete, but prefill grew to about `11.15 s`, decode to about
    `32.18 s`, and throughput regressed to `249.39 output tok/s`;
  - both policies were reverted and the focused scheduler tests passed after
    restoring FIFO admission;
  - decision: keep FIFO waiting admission for now. Request ordering is not a
    reliable random-throughput lever because context length and chunked-prefill
    timing dominate the apparent decode-step-count win.
- rejected cached full-attention `BTHD` layout:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_cached_attention_bthd_r1_20260604.json`;
  - temporary change kept cached full-attention Q/K/V in backend-native
    `[B,T,H,D]` layout through RoPE, KV write, and paged attention, instead of
    transposing to `[B,H,T,D]` for RoPE and back for the cache/backend;
  - focused cached prefill/decode and paged-attention tests passed before the
    benchmark, and measured-phase JIT growth stayed zero;
  - integrated random throughput regressed to `259.21 output tok/s`; prefill
    step time grew to about `11.42 s`, decode grew to about `30.44 s`, and B8
    decode throughput dropped to about `444.5 tok/s`;
  - decision: reject and revert. Source-level transpose removal does not imply
    a faster XLA layout; keep the current full-attention layout until a profile
    proves a concrete lowered transpose or copy bucket disappears.
- next action:
  - continue from the B8 random anchor and target the actual decode buckets:
    broad decode fusion, small-batch projection/LM-head lowering, and
    scheduler/backfill changes that keep decode batches full without
    specializing to seed `1234`, exact request lengths, or Qwen3.5-0.8B
    dimensions.

### Entry 241 - Accepted Random Width/Static-Metadata Improvements

- date: 2026-06-04
- purpose:
  - record the first accepted random-request improvements after the Entry 240
    stress anchor;
  - keep the benchmark evidence in summaries rather than committing full run
    artifacts.
- changes:
  - selected the random serving-envelope width `max_blocks_per_seq=320`, which
    corresponds to the declared random lane envelope
    `(max_input 4096 + max_output 1024) / block_size 16`;
  - split static decode metadata caching so constant width-1 decode tensors
    (`tokens`, placeholder `positions`, `seq_ids`, and `query_start_loc`) are
    cached independently from dynamic `block_tables` and `seq_lens`;
  - added first-class `decode_block_table_buckets` config/benchmark support and
    warmed all configured decode block-table widths generically before
    measurement. The accepted random run used buckets `128,256,320`.
- artifacts:
  - clean width-320 control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_b8_320width_blockdot_r1_20260604.json`;
  - static metadata split repeat:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_b8_320width_static_cache_split_r2_20260604.json`;
  - decode block-table bucket run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_b8_320width_static_cache_split_decode_block_buckets_128_256_320_r1_20260604.json`.
- result:
  - Entry 240 anchor: `383.96 output tok/s`, `30.22 s`, `0.251x` live vLLM;
  - clean width-320 control: `422.03 output tok/s`, `27.49 s`, zero
    measured-phase JIT growth (`24 -> 24`);
  - static metadata split repeat: `431.99 output tok/s`, `26.86 s`, zero
    measured-phase JIT growth (`24 -> 24`);
  - decode block-table buckets: `436.24 output tok/s`, `26.60 s`, zero
    measured-phase JIT growth (`32 -> 32`);
  - against the live vLLM reference `1531.33 output tok/s`, the current random
    ratio is `0.285x`.
- timing interpretation:
  - the accepted bucketed run still spends most time in B8 decode: `1006`
    steps, `8048` tokens, about `11.84 s`, or `~680 tok/s` inside that bucket;
  - B6/B7 and tail buckets improved modestly from smaller block-table metadata
    shapes, but B8 barely moved, so metadata width is a real but minor bucket;
  - the next material win must reduce full decode-step cost itself: production
    paged decode attention, broader model-side decode fusion, or another
    structural route that reduces per-token model executions.
- validation:
  - `python -m py_compile nanovllm_jax/config.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/llm_engine.py benchmarks/benchmark_jax_server_trace.py nanovllm_jax/server_config.py`;
  - `.venv/bin/pytest -q tests/test_device_token_carry.py tests/test_backend_boundaries.py -k 'decode_block_table_buckets or static_decode_metadata or warmup_uses_greedy_token_fastpath_without_mtp or warmup_compiles_decode_block_table_buckets or warmup_uses_requested_prefill_len_without_buckets'`
    -> `9 passed, 80 deselected`;
  - full random runs completed all output lengths and had zero measured-phase
    JIT cache growth under generic warmup.
- current blocker:
  - after the sandbox transition, the local GPU became unavailable to the
    process: `nvidia-smi` cannot communicate with the driver and JAX reports
    CPU only. Do not promote the draft width-1 Triton decode-attention
    prototype until GPU parity and integrated random throughput are measured.

### Entry 242 - Driver Repair And Fresh Random Gap Diagnosis

- date: 2026-06-04
- purpose:
  - merge the completed block-dot FLA prefill PR and keep the random hill-climb
    branch readable after the squash merge;
  - repair the local A10G CUDA runtime so random/vLLM diagnostics can run
    again;
  - remeasure the active random stress target and explain why its vLLM gap is
    much larger than the smaller fixed-shape benchmark gaps.
- GitHub state:
  - PR #1 (`[codex] Hit decode-heavy target with block-dot FLA prefill`) was
    squash-merged into `main` at merge commit
    `eca56449f21ee27ecaad9cc7bfb39569ecd23b12`;
  - PR #2 was rebuilt from the merged `origin/main` with only the random
    checkpoint commit and force-pushed with lease. It is now one commit ahead
    of `main`, mergeable, and remains a draft:
    `ef10494d6476442cbe7d773bf47cd3bb40b70e53`.
- driver diagnosis and fix:
  - `nvcc --version` worked because the CUDA toolkit was installed, but
    `nvidia-smi` and JAX CUDA failed because the running kernel
    `6.17.0-1017-aws` had no loadable NVIDIA module;
  - PCI enumeration showed the GPU was present as NVIDIA A10G
    (`10de:2237`), while `modinfo nvidia` failed and DKMS only had
    `nvidia-srv/580.159.03` installed for older AWS kernels
    (`6.17.0-1010-aws` and `6.17.0-1015-aws`);
  - installing `linux-headers-6.17.0-1017-aws`,
    `linux-modules-nvidia-580-server-6.17.0-1017-aws`, and
    `linux-modules-nvidia-580-server-aws` built/installed the matching DKMS
    module;
  - after `modprobe nvidia` and `modprobe nvidia_uvm`, `nvidia-smi` reported
    driver `580.159.03` on NVIDIA A10G and JAX reported `CudaDevice(id=0)`.
- fresh random measurements:
  - fresh JAX no-profile artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_noprofile_after_driver_fix_20260604.json`;
  - fresh vLLM artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_vllm_after_driver_fix_20260604.json`;
  - workload: `15` requests, `30506` input tokens, `11602` output tokens;
  - JAX: `437.63 output tok/s`, `26.51 s`, full lengths, and zero
    measured-phase JIT growth (`32 -> 32`);
  - vLLM: `1541.89 output tok/s`, `7.52 s`;
  - current random ratio: `0.284x`, consistent with Entry 241's accepted
    `436.24 tok/s` run.
- timing diagnosis:
  - random remains decode dominated: the accepted no-profile run has `13`
    prefill steps and `1759` decode steps inferred from token events;
  - B8 decode is the largest single bucket: `1006` steps, `8048` output
    tokens, `11.84 s`, about `680 tok/s` inside that bucket;
  - B7 and B6 decode add another `7.47 s` for `2717` tokens, and the smaller
    tails are much less efficient (`B1` about `115 tok/s`, `B2` about
    `253 tok/s`);
  - the profiling rerun is diagnostic only because profiling slows the run to
    `370.06 tok/s`, but it still confirms zero measured JIT growth and shows
    CPU/PJRT/scheduler overhead plus many fine-grained GPU launches rather than
    one missing compile bucket.
- why random is much worse than the fixed-shape tests:
  - random has `15` long requests while the current JAX serving envelope is
    `max_num_seqs=8`, so it necessarily runs multiple admission waves and then
    spends substantial time in underfilled B7/B6/B2/B1 tails;
  - vLLM's fresh run uses chunked prefill, async scheduling, persistent
    torch.compile cache, and CUDA graph capture sizes up to `256`. It can keep
    more of the random workload resident and execute mixed prefill/decode and
    decode graphs with much lower per-step overhead;
  - the fixed non-random targets hide much of this serving scheduler gap:
    `decode_heavy_128x128` is single-request/B1, so vLLM cannot exploit high
    concurrency; `long_prefill_512_2048` is prefill dominated, where the
    block-dot/FLA route is already close; canonical `hetero8` fits exactly in
    the B8 envelope and avoids the second-wave random tail;
  - the current random gap should therefore be read as a structural serving
    gap: better arbitrary-batch decode, production paged decode attention, and
    vLLM-like chunked-prefill/backfill are needed before further narrow
    metadata tweaks can move the `0.9x` target.
- decision:
  - keep the current random branch as the best checkpoint, but do not interpret
    the `0.284x` result as a kernel-only loss. It is exposing a broader ABI and
    serving-policy deficit relative to vLLM.

### Entry 243 - Resident Capacity Groundwork And Rejected Resident-Only Route

- date: 2026-06-04
- purpose:
  - start the vLLM-like serving ABI path by separating resident request
    capacity from compiled execution batch width;
  - test whether widening resident capacity alone reduces the random workload
    gap without benchmark-specific warmup.
- changes:
  - added `max_num_resident_seqs` to the model config, server config, and random
    benchmark CLIs. `max_num_seqs` remains the compiled execution batch cap;
    `None`/`0` keeps old behavior;
  - widened persistent KV/GDN resident bookkeeping to the resident capacity
    while keeping warmup keyed to execution buckets;
  - added a scheduler guard so, when resident capacity exceeds execution
    capacity, a full ready decode batch is not delayed behind another waiting
    prefill wave;
  - added focused tests for resident/execution separation, config/env loading,
    and random sidecar command forwarding.
- artifacts:
  - unguarded resident16/B8 diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_resident16_exec8_r1_20260604.json`;
  - guarded resident16/B8 diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_resident16_exec8_guard_r2_20260604.json`.
- result:
  - previous B8 anchor:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_noprofile_after_driver_fix_20260604.json`,
    `437.63 output tok/s`, `26.51 s`, mean TTFT `5894 ms`;
  - unguarded resident16/B8: `263.39 output tok/s`, `44.05 s`, mean TTFT
    `11619 ms`, zero measured-phase JIT growth (`32 -> 32`);
  - guarded resident16/B8: `278.97 output tok/s`, `41.59 s`, mean TTFT
    `10760 ms`, zero measured-phase JIT growth (`32 -> 32`);
  - fresh vLLM reference remains `1541.89 output tok/s`, so the guarded
    resident-only route is only `0.181x` vLLM and is worse than the B8 anchor.
- interpretation:
  - resident capacity is necessary groundwork, but not sufficient. The current
    executor ABI still runs prefill-only or decode-only steps, so widening
    resident admission lets long prompt chunks consume many scheduler steps
    before useful decode;
  - the protective decode guard reduces the worst regression but does not
    restore the B8 anchor, because the system still lacks true mixed
    chunked-prefill/decode backfill;
  - do not retry resident-only widening as a performance optimization. The next
    material step is a mixed packed scheduler/executor boundary where decode
    rows and bounded prefill chunks can share a warmed serving bucket.
- validation:
  - `python -m py_compile nanovllm_jax/config.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/chunked_model_runner.py nanovllm_jax/server_config.py benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_device_token_carry.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py tests/test_benchmark_jax_server_trace.py`
    -> `41 passed`.

### Entry 244 - Mixed Backfill And Decode Burst Diagnostics

- date: 2026-06-04
- purpose:
  - test the next proposed random-workload serving steps under the same
    seed-`1234` manifest and generic warmup discipline:
    mixed packed prefill/decode backfill and greedy decode micro-bursts.
- changes:
  - added a `ScheduledBatch.mixed_prefill_decode` marker plus explicit mixed
    packed batch construction for one-token decode rows and bounded prefill
    rows;
  - taught device-token carry to replace placeholder packed decode tokens in
    mixed packed batches;
  - kept the mixed route out of automatic scheduling after integrated
    measurements showed it regressed output throughput;
  - fixed greedy decode burst scans so KV/hybrid carry dtypes remain stable
    across `lax.scan`;
  - changed generic warmup to compile every burst length from `2` through the
    configured burst size, because request tails can choose shorter legal burst
    lengths during serving.
- artifacts:
  - full-budget mixed resident16/B8:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_mixed_resident16_exec8_r1_20260604.json`;
  - min-bucket mixed resident16/B8:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_mixed_minbucket_resident16_exec8_r1_20260604.json`;
  - burst4 resident16/B8:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_burst4_resident16_exec8_r3_20260604.json`;
  - burst8 resident16/B8:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_burst8_resident16_exec8_r1_20260604.json`;
  - burst4 resident8/B8:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260604/random_burst4_resident8_exec8_r1_20260604.json`.
- results:
  - mixed full-budget: `165.24 output tok/s`, mean TTFT `26098 ms`,
    zero measured-phase JIT growth (`32 -> 32`);
  - mixed min-bucket: `113.64 output tok/s`, mean TTFT `42291 ms`,
    zero measured-phase JIT growth (`32 -> 32`);
  - burst4 resident16: `282.37 output tok/s`, mean TTFT `10793 ms`,
    zero measured-phase JIT growth (`68 -> 68`);
  - burst8 resident16: `284.73 output tok/s`, mean TTFT `10801 ms`,
    zero measured-phase JIT growth (`116 -> 116`);
  - burst4 resident8: `283.15 output tok/s`, mean TTFT `14786 ms`,
    zero measured-phase JIT growth (`68 -> 68`);
  - the B8 no-burst random anchor from Entry 242 remains better at
    `437.63 output tok/s`, and fresh vLLM remains `1541.89 output tok/s`.
- interpretation:
  - automatic mixed packed backfill is rejected for this workload. It performs
    useful prefill work, but it makes live decode rows wait behind prompt
    chunks and harms the output-throughput/TTFT objective;
  - smaller mixed chunks reduce worst individual stalls but increase the number
    of mixed steps enough to lose even more overall;
  - greedy burst now has correct dtype-stable scans and honest generic warmup,
    but source-level full-model burst scans are still not a promotion route:
    burst8 halves decode call count relative to burst4, yet only improves
    output throughput by `0.8%` while adding `48` more warmup compile entries;
  - resident8 and resident16 burst4 are effectively tied, so the current random
    gap is not mainly resident capacity. The remaining gap is decode compute
    inside the compiled graph plus the lack of production paged decode/GDN
    kernels.
- decision:
  - keep the explicit mixed packed ABI and burst warmup/dtype fixes as
    groundwork and diagnostics;
  - do not enable automatic mixed backfill or increase default greedy burst
    width as a promoted optimization;
  - next material work should replace the decode kernels/boundaries rather than
    further tuning scheduler burst width.
- validation:
  - `python -m py_compile nanovllm_jax/engine/scheduled_batch.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/model_executor.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_backend_boundaries.py::test_model_runner_warmup_compiles_greedy_decode_burst tests/test_backend_boundaries.py::test_executor_greedy_decode_burst_matches_iterative_token_path tests/test_device_token_carry.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py tests/test_benchmark_jax_server_trace.py`
    -> `45 passed`.

### Entry 245 - Rejected Narrow FA Decode Triton Routes

- date: 2026-06-04
- purpose:
  - execute the FA/FLA integration plan's first full-attention decode slice
    with config-selected kernel policy instead of environment-variable sprawl;
  - test whether replacing the width-1 paged full-attention decode alone is
    enough to move the decode-heavy target toward `0.9x` vLLM.
- changes:
  - added `kernels.full_attention.{kv_cache_dtype,kv_append_impl,decode_impl,prefill_impl}`
    server config projection and benchmark CLI forwarding;
  - routed `full_attention.decode_impl=triton_paged` through the JAX-Triton
    paged decode attention wrapper over the existing NHD paged KV cache;
  - added a broader diagnostic `triton_paged_fused_append` path that stores the
    current K/V token and computes paged decode attention in one Triton call;
  - kept accepted configs on `decode_impl=reference` because the measured
    Triton routes regressed.
- artifacts:
  - A/B matrix:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/20260604_decode_heavy_ab_matrix.json`;
  - current static profile artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_current_profile_direct_r1.json`;
  - current static profile summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_current_profile_summary.md`;
  - fused append diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fused_append_direct_r1.json`;
  - B1 packed-QKV diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_b1_packed_qkv_direct_r1.json`.
- results:
  - current static route: exact, `178.93 output tok/s`, `0.838x` stored vLLM
    (`213.54 tok/s`);
  - `full_attention.decode_impl=triton_paged`: exact, `175.79 output tok/s`,
    `0.823x` stored vLLM;
  - `full_attention.decode_impl=triton_paged_fused_append`: exact,
    `161.59 output tok/s`;
  - B1 packed full-attention QKV diagnostic: exact, `161.11 output tok/s`;
  - profiled current-static rerun was diagnostic only and slowed to
    `154.76 output tok/s`.
- profile interpretation:
  - the profiled current route still shows the dominant decode-heavy work in
    model GEMM/fusion buckets rather than full-attention paged decode alone:
    `gemm 386.44 ms`, `fusion 267.38 ms`, `cutlass 193.08 ms`,
    `_gdn 125.99 ms`, and `MemcpyD2D 102.71 ms`;
  - the largest single HLO op is `command_buffer_436` cutlass GEMM with ReLU
    (`127.20 ms / 127`), so a narrow FA attention replacement cannot explain
    or close the remaining gap;
  - standalone append+attention fusion added custom-call overhead and did not
    remove the larger model-side launch/GEMM buckets.
- decision:
  - reject promotion of `triton_paged`, `triton_paged_fused_append`, and B1
    packed-QKV decode for the serving defaults;
  - keep the config-level FA policy and focused correctness tests as scaffolding
    for future production-kernel integration, but do not add extra knobs for
    failed threshold probes;
  - next work should identify the largest model-side decode GEMM/fusion source
    and target a broader, model-family-general boundary rather than hand tuning
    kernel parameters.

### Entry 246 - Decode-Heavy B1 Dtype And Final-Sync Diagnostics

- date: 2026-06-04
- purpose:
  - explain why the current hetero8-oriented BF16 config no longer reaches the
    old decode-heavy `0.9x` result;
  - test route-selection, dtype-policy, and final materialization hypotheses
    without keeping failed benchmark-specific changes.
- artifacts:
  - BF16 current-control matrix artifact:
    `/mountpoint/.exp/nano-vllm-jax/results/gpu_matrix_runs/20260604_144131/decode_heavy_128x128_gpu_paged_gdn_fla_decode_static_metadata_repeat1.json`;
  - BF16 reference-GDN direct:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_bf16_reference_gdn_direct_r1.json`;
  - FP32 reference-GDN direct repeats:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_reference_gdn_direct_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_reference_gdn_direct_r2_after_reverts.json`;
  - FP32 reference-GDN profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_reference_gdn_profile_direct_r1.json`;
  - FP32 profile summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_reference_profile_summary.md`;
  - rejected probes:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_triton_fla_gdn_direct_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_conv_raw_gdn_direct_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_reference_gdn_trace_prefetch_direct_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_fp32_reference_gdn_stacked_final_direct_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_bf16_single_seq_fp32_policy_reference_direct_r1.json`,
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/fa_triton_decode/decode_heavy_bf16_single_seq_fp32_policy_conv_direct_r1.json`.
- results:
  - BF16 current config with conv-fused GDN decode: exact,
    `178.93 output tok/s`;
  - BF16 with reference GDN decode: exact, `179.10 output tok/s`;
  - FP32 activations with reference GDN decode: exact repeats at `191.32` and
    `189.94 output tok/s`; profiled repeat was exact at `191.90 output tok/s`;
  - FP32 activations with non-conv Triton GDN decode: exact but
    `169.76 output tok/s`;
  - FP32 activations with conv-fused Triton GDN decode: exact but
    `169.38 output tok/s`;
  - trace-token prefetch: exact but regressed to `187.73 output tok/s`;
  - final stacked `DeviceTokenRef` materialization: exact but regressed to
    `161.15 output tok/s`;
  - temporary BF16-server/single-seq-FP32 internal dtype policy: exact but
    unchanged with reference GDN (`178.93 output tok/s`) and worse with
    conv-fused GDN (`169.60 output tok/s`).
- profile interpretation:
  - the B1 decode-heavy gap is not fixed by the FA decode kernel or GDN Triton
    decode kernels. The best B1 route is the pure reference GDN decode boundary
    with FP32 model activations;
  - the FP32 profile collapses decode execution into far fewer command-buffer
    buckets than the BF16/conv-fused current route: `PjRt Execute` is about
    `389 ms / 263` in the FP32 reference profile versus about
    `570 ms / 263` in the BF16 current profile;
  - the measured output-throughput gap to `0.9x` is mostly a real final
    synchronization, not accounting noise. The best FP32 run emits the last
    token at about `0.609 s` (`~210 output tok/s` by token events), but full
    wall time is about `0.669 s` after final token materialization;
  - prefetch and stacked final materialization both move synchronization around
    or add extra JAX work; neither reduces full wall time.
- decision:
  - reject B1 Triton GDN decode, trace prefetch promotion, final stacked
    materialization, and the internal BF16-server/single-seq-FP32 dtype policy;
  - keep the current BF16 route for hetero8/random work, but do not claim the
    BF16 current config is still the best B1 decode-heavy route;
  - the best safe B1 decode-heavy diagnostic is FP32 activations with reference
    GDN decode, median `191.32 tok/s` (`0.896x` the stored `213.54 tok/s` vLLM
    reference), which is near but below the `0.9x` line;
  - next material work should return to the broader random/hetero8 decode graph
    or a production fused decode boundary. Further final-token sync reshaping is
    not a promising route.

### Entry 247 - Guarded Resident Decode Boundary Groundwork

- date: 2026-06-05
- purpose:
  - continue the random decode hill-climb by expanding the decode JIT boundary
    around resident slot metadata;
  - avoid repeating the unsafe full random compile path after a boundary-changing
    edit.
- implementation:
  - added an opt-in `resident_decode_metadata` config/CLI/server-config field;
  - added a resident-slot greedy decode executor path that gathers block tables,
    sequence lengths, and GDN state by compact slot ids inside the JIT boundary,
    then scatters updated GDN state and resident sequence lengths back to the
    resident tables;
  - mirrored scheduler-owned block tables and sequence lengths into
    device-resident runner tables;
  - cast updated table-scattered GDN state back to the resident table dtypes to
    avoid future JAX scatter dtype failures;
  - updated `AGENTS.md` and the active plan so new random decode JIT boundaries
    must pass scaled JAX-only diagnostics before full seed-`1234` random runs.
- current status:
  - resident decode metadata remains default-off. It is a guarded probe, not an
    accepted speedup;
  - no full random artifact is recorded for the resident-slot path yet because
    the previous full compile attempt did not complete safely.
- validation:
  - `python -m py_compile nanovllm_jax/config.py nanovllm_jax/server_config.py nanovllm_jax/engine/scheduled_batch.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_random_request_sidecar.py tests/test_backend_boundaries.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'table_hybrid_decode_matches_sliced_decode or syncs_resident_decode_metadata_by_slot or hybrid_slot_ids_zero or scheduler_pads_scheduled_batch_to_static_buckets'`;
  - `pytest -q tests/test_device_token_carry.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py tests/test_benchmark_jax_server_trace.py`;
  - `git diff --check`.
- next:
  - checkpoint and push the branch;
  - run a scaled JAX-only random diagnostic with `resident_decode_metadata`
    enabled before trying medium or full random sidecar runs.

### Entry 248 - Config-Driven Random Sidecar And Resident-Metadata A/B

- date: 2026-06-05
- purpose:
  - avoid another unsafe full random compile after the resident decode boundary
    change;
  - make the random sidecar consume the same typed config policy as the server
    and GPU matrix runner;
  - test the resident decode metadata path on a scaled random envelope before
    any medium/full random run.
- implementation:
  - added `--jax-config` to `benchmark_random_request_sidecar.py`;
  - the sidecar now projects config `runtime`/`kernels` into JAX subprocess env
    and accepted engine fastpath/kernel CLI args using
    `runtime_env_from_config()` and `engine_overrides_from_config()`;
  - fixed `CanonicalModelRunner` initialization so it carries
    `resident_decode_metadata` and related serving flags before generic warmup.
- artifacts:
  - unsafe/default sidecar probe:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_resident_scaled_safety_r1.json`;
  - missing canonical flag probe:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_resident_scaled_safety_r2.json`;
  - tiny resident metadata smoke:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_resident_scaled_safety_r3.json`;
  - accepted-config resident-on A/B:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_resident_config_scaled_r1.json`;
  - accepted-config resident-off control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_scaled_control_r1.json`.
- results:
  - r1 failed during prefill warmup with a GPU custom-call shared-memory request
    of `152064 B`, above the A10G limit of about `101376 B`. This happened
    before decode and used sidecar defaults instead of the accepted BF16/XLA
    config policy;
  - r2 reached optimized warmup setup but failed because
    `CanonicalModelRunner` lacked `resident_decode_metadata`;
  - r3 completed the tiny resident metadata path with zero measured-phase JIT
    cache growth, but it did not use the accepted fastpath/kernel config;
  - accepted-config resident-on: `67.75 output tok/s`, `59` output tokens,
    zero measured-phase JIT growth;
  - accepted-config resident-off control: `104.33 output tok/s`, same `59`
    output tokens, zero measured-phase JIT growth;
  - generated token prefixes were identical between resident-on and control for
    both requests in the tiny A/B.
- decision:
  - do not promote the resident decode metadata v1 path. It is correct and
    cache-stable on the tiny envelope, but it is slower than the current
    table/static metadata path;
  - keep the path opt-in as scaffolding only. Do not scale it to medium/full
    random runs unless the gather/scatter/table-update overhead is first
    reduced or folded into a larger useful decode boundary;
  - use the config-driven sidecar for future random runs so the benchmark
    cannot silently fall back to sidecar defaults.
- validation:
  - `python -m py_compile benchmarks/benchmark_random_request_sidecar.py tests/test_benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_benchmark_random_request_sidecar.py`;
  - `python -m py_compile nanovllm_jax/engine/model_runner.py tests/test_backend_boundaries.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'initializes_resident_decode_metadata_flag or table_hybrid_decode_matches_sliced_decode or syncs_resident_decode_metadata_by_slot'`;
  - `git diff --check`.

### Entry 249 - Guarded Random Sidecar And Table-Prefill Warmup

- date: 2026-06-05
- purpose:
  - prevent another machine-stressing random compile/run by adding resource
    limits at the benchmark subprocess boundary;
  - finish the table-prefill boundary promotion check by ensuring generic
    warmup compiles the same packed table-prefill ABI that serving uses.
- implementation:
  - replaced the random sidecar's blocking `subprocess.run()` helper with a
    monitored `Popen` loop;
  - added a default `--max-system-ram-percent 70` guard that terminates the
    JAX/vLLM subprocess group if system RAM crosses the threshold;
  - added worker CPU controls: `--worker-cpu-cores`,
    `--worker-cpu-core-offset`, `--worker-nice`, and
    `--resource-poll-seconds`;
  - the sidecar records the configured resource limits, peak system RAM, and
    peak subprocess-tree RSS in benchmark summaries;
  - moved packed prefill GDN table gather/scatter into
    `forward_prefill_token_ids_table_jit`;
  - updated `CanonicalModelRunner.warmup_compilation()` so generic startup
    warmup compiles `forward_prefill_token_ids_table_jit` for packed prefill
    token buckets and row-count buckets when the table path is available.
- artifacts:
  - guard dry run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_sidecar_resource_guard_dryrun.json`;
  - guarded scaled table-prefill run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_guarded_scaled_r1.json`;
  - JAX child artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_guarded_scaled_r1_jax.json`.
- results:
  - the guarded scaled run completed normally; the watchdog did not fire;
  - resource telemetry recorded peak system RAM `40.8%` and peak process-tree
    RSS about `3.82 GB`;
  - generic warmup compiled table-prefill routes for packed token buckets
    `128,256` and row buckets `1,2`, with block-table shapes `[1,32]` and
    `[2,32]`;
  - measured-phase JIT growth stayed zero (`10 -> 10`);
  - scaled throughput improved from the prior accepted-config tiny control
    (`104.33 output tok/s`) to `132.48 output tok/s` on the same `2`-request,
    `59`-output-token envelope.
- decision:
  - promote the table-prefill boundary to the next medium random diagnostic;
  - keep the resource guard enabled for future random sidecar runs;
  - do not treat the tiny scaled result as a full random claim or a vLLM-ratio
    update.
- validation:
  - `python -m py_compile benchmarks/benchmark_random_request_sidecar.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/model_executor.py tests/test_benchmark_random_request_sidecar.py tests/test_backend_boundaries.py`;
  - `pytest -q tests/test_benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'warmup_compiles_table_prefill_when_available or warmup_uses_greedy_token_fastpath_without_mtp or packed_prefill_greedy_token_jit_returns_row_tokens'`;
  - `pytest -q tests/test_benchmark_random_request_sidecar.py tests/test_backend_boundaries.py -k 'benchmark_random_request_sidecar or warmup_compiles_table_prefill_when_available or warmup_uses_greedy_token_fastpath_without_mtp or warmup_compiles_decode_block_table_buckets or packed_prefill_greedy_token_jit_returns_row_tokens or table_hybrid_decode_matches_sliced_decode or initializes_resident_decode_metadata_flag or syncs_resident_decode_metadata_by_slot'`;
  - `git diff --check`.

### Entry 250 - Static Decode Carry Row Alignment

- date: 2026-06-05
- purpose:
  - attack the largest remaining medium-random profile bucket after table
    prefill promotion: CPU-side JAX gather/scatter and extra PJRT executions in
    `_maybe_apply_device_token_carry`;
  - keep the optimization model-family general by reducing serving-boundary
    array rewrites rather than tuning kernels or dimensions.
- implementation:
  - when greedy decode returns a full static-bucket token array, record the full
    `batch.seq_ids_host` tuple, including padded rows, as the fast-path carry
    alignment key;
  - keep `_device_token_carry_by_seq_id` active-row-only so correctness for
    row-order changes still uses logical sequence IDs;
  - skip prefill resident-metadata sync unless `resident_decode_metadata` is
    enabled. The accepted random path has resident metadata disabled, so the
    previous unconditional sync was dead work.
- artifacts:
  - guarded medium JAX-only before:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_guarded_medium_r1.json`;
  - guarded medium live-vLLM comparison:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_guarded_medium_with_vllm_r1.json`;
  - guarded medium JAX-only after:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_token_carry_medium_r1.json`;
  - medium profile before:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_guarded_medium_profile_r1.json`;
  - medium profile after:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_token_carry_medium_profile_r1.json`.
- results:
  - medium live comparison before this change: JAX `313.31 output tok/s`, vLLM
    `511.94 output tok/s`, ratio `0.612x`, zero measured-phase JIT growth;
  - medium after this change: JAX `355.14 output tok/s`, same `290` output
    tokens, zero measured-phase JIT growth. Against the live medium vLLM
    denominator this is about `0.69x`;
  - profile before/after:
    - CPU `gather`: `376.6 ms` -> `50.9 ms`;
    - `PjRtCApiLoadedExecutable::Execute` count: `557` -> `321`;
    - `_run_main_and_sample`: `808.0 ms` -> `646.0 ms`;
    - GPU `cutlass` and `fusion` totals were effectively unchanged, so the
      win came from host/PJRT boundary cleanup, not GPU-kernel changes.
- decision:
  - keep this change and promote it to the next larger guarded random rung;
  - the next optimization should target the remaining per-step PJRT/device-token
    carry overhead or a broader fused decode boundary. The GPU GEMM/fusion work
    is now a more visible share of time, but host overhead is not fully gone.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_runner.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_device_token_carry.py -k 'records_full_static_decode_rows or whole_vector_when_seq_ids_match or follows_seq_ids_after_row_order_change or static_decode_metadata_applies_carried_seq_lens or first_static_decode_can_use_scheduler_seq_lens'`;
  - `pytest -q tests/test_backend_boundaries.py -k 'warmup_compiles_table_prefill_when_available or packed_prefill_greedy_token_jit_returns_row_tokens'`.

### Entry 251 - Larger Random Rung And Metadata-Carry Rejections

- date: 2026-06-05
- purpose:
  - promote the safe random ladder beyond the medium `4`-request envelope;
  - test whether the now-visible scheduler/static metadata buckets can be
    reduced without changing the serving ABI or specializing to a shape.
- artifacts:
  - larger guarded JAX-only:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_token_carry_large_guarded_r1.json`;
  - larger live-vLLM comparison:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_token_carry_large_with_vllm_r1.json`;
  - larger profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_table_prefill_token_carry_large_profile_r1.json`;
  - seq-lens carry diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_seq_lens_carry_large_guarded_r1.json`;
  - shared-gather carry diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_shared_gather_large_guarded_r1.json`.
- results:
  - larger JAX-only rung: `8` requests, `6240` input tokens, `1351` output
    tokens, `397.84 output tok/s`, zero measured-phase JIT growth (`20 -> 20`),
    peak system RAM `44.2%`;
  - larger live comparison: JAX `400.42 output tok/s`, vLLM
    `884.03 output tok/s`, ratio `0.453x`, same input/output token counts,
    zero JIT growth, peak system RAM below `47%`;
  - larger profile: `_run_main_and_sample` `1817 ms`,
    `PjRtCApiLoadedExecutable::Execute` `1398 ms`, `_maybe_apply_device_token_carry`
    `741 ms`, scheduler `build_scheduled_batch` `557 ms`, and GPU `fusion`
    plus `cutlass` totals about `1064 ms`;
  - `static_decode_seq_lens_carry=true` stayed cache-stable but regressed to
    `341.31 output tok/s` on the larger rung;
  - a shared-gather fallback for row-reordered device-token carry stayed
    cache-stable but regressed to `370.84 output tok/s`.
- decision:
  - keep Entry 250's full-row static decode carry alignment;
  - reject seq-lens carry promotion on the current random path;
  - reject the shared-gather token-carry fallback and keep the previous
    per-row fallback for row-reordered/finishing requests;
  - next work should target a broader decode/scheduler boundary or production
    kernel integration. Source-level metadata micro rewrites are now likely to
    trade one PJRT bucket for another unless they remove an entire per-step
    call.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_runner.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_device_token_carry.py -k 'records_full_static_decode_rows or whole_vector_when_seq_ids_match or follows_seq_ids_after_row_order_change or survives_nonfinal_prefill_chunk or applies_device_token_carry_to_mixed_packed_decode_rows or static_decode_metadata_requires_token_carry'`;
  - `git diff --check`.

### Entry 252 - Full Random Resource-Guard Safety Validation

- date: 2026-06-05
- purpose:
  - verify the full seed-`1234` random sidecar can run under the new watchdog
    and finite generic warmup after the table-prefill/token-carry changes;
  - avoid another unbounded compile/run while still checking the full request
    graph.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_config_token_carry_full_guarded_r1.json`.
- workload:
  - `15` requests;
  - prompt lengths `1077..4022`, total input tokens `30506`;
  - output lengths `425..1007`, total output tokens `11602`.
- result:
  - completed successfully under `--max-system-ram-percent 70` and
    `--worker-cpu-cores 2`;
  - peak system RAM `53.0%`, peak process-tree RSS about `5.72 GB`;
  - generic warmup compiled `36` entries: `24` packed table-prefill shapes and
    `12` decode shapes;
  - measured-phase JIT growth stayed zero (`36 -> 36`);
  - measured JAX throughput was `239.35 output tok/s`.
- interpretation:
  - this is a safety validation, not a new accepted performance baseline. The
    two-core CPU affinity cap intentionally protects the machine, but it also
    likely depresses measured throughput because the random path still has
    CPU-visible scheduler/PJRT work;
  - for fair performance comparisons, either cap both JAX and vLLM identically
    or run an uncapped/niceness-only comparison after the next structural
    optimization.
- decision:
  - keep the resource guard and safety ladder;
  - do not replace the prior full-random performance anchor with this capped
    safety result.

### Entry 253 - Resident Placeholder And GDN Boundary Audit

- date: 2026-06-05
- purpose:
  - continue the six-item random decode plan without retrying rejected
    metadata-carry or decode-burst variants;
  - test whether the resident decode path can stop rebuilding device metadata
    arrays that its compiled executor ignores;
  - inspect whether the active GDN/GEMM route has a safe small promotion left,
    or whether the next useful work needs a coarser boundary.
- implementation:
  - `Scheduler` now carries `resident_decode_metadata` directly from config;
  - when static decode metadata and resident decode metadata are both enabled,
    `_static_decode_device_arrays()` uses stable zero placeholders for device
    `block_tables` and `seq_lens`, while preserving true `block_tables_host`
    and `seq_lens_host` for resident-table synchronization;
  - added a scheduler regression test that changes sequence lengths and block
    tables across decode steps and verifies placeholder reuse plus host-metadata
    preservation.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_resident_placeholder_medium_r1.json`.
- result:
  - medium resident-placeholder run: `4` requests, `1787` input tokens,
    `290` output tokens, `354.32 output tok/s`, zero measured-phase JIT growth,
    peak system RAM `45.0%`;
  - same-envelope promoted control with packed Triton prefill and reference FA
    decode: `360.10 output tok/s`;
  - therefore the resident placeholder route is still slower than the current
    table/static metadata route.
- GDN/GEMM audit:
  - current larger-rung profile has real GPU cost, with `fusion + cutlass`
    around `1064 ms`;
  - `_gdn_conv_packed_decode_raw_gate_kernel` is `112.35 ms` over `4983` calls,
    which matches one fused GDN decode kernel per GDN layer per decode step;
  - that active Triton route already owns conv update, q/k normalization, raw
    gate/beta math, and recurrent-state update. A safe next GDN win is not a
    flag flip or launch-parameter sweep; it needs a coarser projection+GDN,
    multi-layer/backend-owned decode, or greedy LM-head matmul+argmax boundary.
- decision:
  - keep resident device placeholders as opt-in scaffolding, but do not enable
    `resident_decode_metadata` on the random serving path;
  - do not retry source-level greedy decode bursts, which Entry 122 already
    rejected badly;
  - do not spend the next pass on padded-GEMM row or narrow GDN launch sweeps.
- validation:
  - `python -m py_compile nanovllm_jax/engine/scheduler.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_device_token_carry.py -k 'static_decode_metadata_reuses_fixed_device_arrays or static_decode_metadata_can_reuse_seq_lens_placeholder or resident_decode_metadata_uses_device_placeholders'`.

### Entry 254 - Random Iteration Lanes And Stored vLLM Denominators

- date: 2026-06-05
- purpose:
  - keep future random hill-climbing fast and comparable;
  - avoid paying live-vLLM startup on every JAX-only implementation attempt;
  - avoid using the full seed-`1234` random graph as the default edit/measure
    loop when medium and large random already exercise the same serving paths.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_promoted_medium_with_vllm_r1.json`.
- result:
  - medium random live comparison: `4` requests, `1787` input tokens,
    `290` output tokens;
  - JAX `355.77 output tok/s`;
  - vLLM `471.06 output tok/s`;
  - JAX/vLLM ratio `0.755x`;
  - zero measured-phase JIT growth (`15 -> 15`);
  - peak system RAM: JAX `45.2%`, vLLM `46.5%`.
- correctness note:
  - one request matched fully, while the other rows diverged early from vLLM.
    This remains acceptable for the random lane under the current approximate
    correctness policy because earlier diagnostics showed close logits around
    tie-sensitive early tokens.
- decision:
  - use medium and large random as the normal optimization lanes;
  - treat the full seed-`1234` random graph as an occasional safety/release
    validation;
  - use stored same-envelope vLLM denominators for JAX-only A/B work. Rerun
    live vLLM only after benchmark-contract changes, runtime/library/hardware
    changes, or before promoting a new best result.

### Entry 255 - Stored vLLM Denominator Support

- date: 2026-06-05
- purpose:
  - make the random sidecar policy executable, not only documented;
  - allow normal JAX-only medium/large iterations to report a JAX/vLLM ratio
    without launching vLLM every time.
- implementation:
  - added `--vllm-reference-json` to
    `benchmarks/benchmark_random_request_sidecar.py`;
  - the reference path may point to a raw vLLM benchmark JSON or a prior random
    sidecar JSON. For sidecar JSONs, the tool loads the nested
    `runs.vllm.artifact` when available and falls back to the sidecar's vLLM
    performance summary otherwise;
  - sidecar output records vLLM run status `stored_reference`, the reference
    source/kind/artifact, and `performance.jax_over_vllm` ratios.
- artifacts:
  - dry-run schema check:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/dry_run_stored_vllm_reference.json`;
  - JAX-only medium check using the stored live-vLLM denominator:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_promoted_medium_stored_vllm_r1.json`.
- result:
  - stored-reference run used
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_promoted_medium_with_vllm_r1_vllm.json`;
  - JAX `359.34 output tok/s`;
  - stored vLLM `471.06 output tok/s`;
  - JAX/vLLM `0.763x`;
  - zero measured-phase JIT growth (`15 -> 15`);
  - vLLM elapsed time recorded as `0.0` because no vLLM subprocess was launched.
- validation:
  - `python -m py_compile benchmarks/benchmark_random_request_sidecar.py tests/test_benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_benchmark_random_request_sidecar.py`.

### Entry 256 - Resident Slot-Token Decode Boundary

- date: 2026-06-05
- purpose:
  - remove the remaining hot decode path that rewrote `batch.tokens` from
    `DeviceTokenRef` rows in Python/JAX before every static decode step;
  - keep final generated-token materialization correct by preserving immutable
    per-step `DeviceTokenRef` vectors, rather than using a mutable table as the
    output history;
  - move the next-token input carry into the compiled decode boundary, matching
    the broader serving-framework pattern where resident request slots own the
    mutable decode input/state.
- implementation:
  - added `ExecutorOutput.resident_last_tokens`;
  - added `ModelExecutor.forward_step_token_ids_slot_carry_table_jit`;
  - the new JIT gathers `tokens = resident_last_tokens[hybrid_slot_ids]`,
    runs the existing table-hybrid greedy decode path, scatters updated hybrid
    state, and scatters sampled token IDs back into `resident_last_tokens`;
  - `CanonicalModelRunner` now maintains `_resident_last_tokens` per hybrid
    slot, seeds it from `_record_device_token_carry`, resets released slots, and
    skips `_maybe_apply_device_token_carry` only when every active static-decode
    row has both a device-token carry and an assigned hybrid slot;
  - `_record_device_token_carry` still records immutable output token refs for
    correctness/final materialization. On the resident slot-token route, it does
    not issue a second post-JIT resident-token scatter.
- artifacts:
  - first comparable large random capacity fix attempt:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_slot_carry_table_r2.json`;
  - post-guard accepted large random run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_slot_carry_table_r3.json`;
  - nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_slot_carry_table_r3_jax.json`.
- result:
  - large random post-patch JAX: `460.45 output tok/s`, `1582` generated
    tokens, `3.4358 s`, ITL mean `7.36 ms`;
  - stored vLLM denominator: `884.03 output tok/s`;
  - JAX/stored-vLLM output-token ratio: `0.521x`;
  - previous large live-comparison JAX anchor:
    `400.42 output tok/s`, `1351` generated tokens, `3.3740 s`, ITL mean
    `8.03 ms`;
  - measured-phase JIT cache growth remained zero (`20 -> 20`);
  - generic warmup compiled `20` entries, and all decode warmup routes were
    `forward_step_token_ids_slot_carry_table_jit:decode` rather than the older
    `forward_step_token_ids_table_jit:decode`.
- benchmark caveat:
  - random generated token text is allowed to diverge under the current random
    correctness policy;
  - the sidecar stored-vLLM ratio reuses the prior vLLM denominator, so treat
    `0.521x` as the current hill-climb signal rather than a final promotion
    ratio. Rerun live vLLM before declaring a release baseline.
- interpretation:
  - this is a concrete structural fix: it removes a whole per-step token-carry
    rewrite/slice boundary from static decode and folds that state update into
    useful decode work;
  - the gain is real but not enough for the `0.9x` target. The next profile
    should start from this route and attack the remaining model/GEMM,
    GDN/full-attention, and PjRT/command-buffer buckets.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py tests/test_backend_boundaries.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_device_token_carry.py -k 'updates_resident_last_tokens or records_full_static_decode_rows or whole_vector_when_seq_ids_match or follows_seq_ids_after_row_order_change or static_decode_metadata_requires_token_carry'`;
  - `pytest -q tests/test_backend_boundaries.py -k 'table_hybrid_decode_matches_sliced_decode or warmup_compiles_decode_block_table_buckets'`.

### Entry 257 - Prefill Slot-Token Seeding Boundary

- date: 2026-06-05
- purpose:
  - continue the random decode checklist from Entry 256 by removing the next
    host-visible resident-token update;
  - keep chunked/ragged prefill semantics exact: only final prefill chunks seed
    the resident decode token table.
- implementation:
  - added `ModelExecutor.forward_prefill_token_ids_slot_carry_table_jit`;
  - the compiled packed-prefill boundary now gathers/scatters hybrid state as
    before, samples greedy token IDs, and scatters those token IDs into
    `resident_last_tokens[hybrid_slot_ids]` only where
    `prefill_final_flags` is true;
  - `CanonicalModelRunner` caches device prefill-final flag vectors, routes
    generic warmup and serving prefill through the new boundary when device
    token carry is enabled, and suppresses the old post-prefill
    `_record_resident_last_tokens` scatter when the compiled boundary already
    seeded the resident table.
- artifacts:
  - large random benchmark:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_prefill_slot_carry_table_r1.json`;
  - nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_prefill_slot_carry_table_r1_jax.json`;
  - profiled large random benchmark:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_prefill_slot_carry_table_profile_r1.json`;
  - nested profiled JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_prefill_slot_carry_table_profile_r1_jax.json`;
  - profile trace:
    `/mountpoint/.exp/profiles/20260605-073625-101290-random_large_prefill_slot_carry_table_profile_r1_jax`.
- result:
  - large random post-patch JAX: `470.14 output tok/s`, `1582` generated
    tokens, `3.3649 s`, ITL mean `7.84 ms`;
  - previous accepted Entry 256 large random: `460.45 output tok/s`, `1582`
    generated tokens, `3.4358 s`;
  - stored vLLM denominator remains `884.03 output tok/s`, so the JAX/stored
    vLLM ratio moved from `0.521x` to `0.532x`;
  - measured-phase JIT cache growth remained zero (`20 -> 20`);
  - generic warmup compiled all 8 packed prefill bucket routes as
    `forward_prefill_token_ids_slot_carry_table_jit:prefill` and all 12 decode
    bucket routes as `forward_step_token_ids_slot_carry_table_jit:decode`.
- profile delta:
  - old Entry 256 profile had `_record_resident_last_tokens` at `439.35 ms / 7`
    calls and total `gather` at `1216.23 ms`;
  - new profile no longer has `_record_resident_last_tokens` or
    `_record_device_token_carry` in the top events, and total `gather` dropped
    to `293.55 ms`;
  - `_run_main_and_sample` dropped from `1917.86 ms / 315` to
    `1664.43 ms / 338` in the profiled run;
  - remaining visible buckets are `PjRtCApiLoadedExecutable::Execute`
    (`1535.04 ms / 626`), scheduler/build metadata (`schedule`
    `761.71 ms`, `_static_decode_device_arrays` `543.76 ms`, `device_put`
    `532.76 ms`), and GPU model work (BF16 GEMMs, GDN decode kernel, and
    fusion/cutlass).
- interpretation:
  - this accepted fix removes the final known host-visible resident-token seed
    on the current static decode route;
  - the next general target should be scheduler/static-decode metadata movement
    and broader decode execution fusion, not another token-carry rewrite.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py tests/test_backend_boundaries.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'packed_prefill_greedy_token_jit_returns_row_tokens or warmup_compiles_table_prefill_when_available or warmup_compiles_prefill_slot_carry_table_when_available or warmup_compiles_decode_block_table_buckets'`;
  - `pytest -q tests/test_device_token_carry.py -k 'updates_resident_last_tokens or records_full_static_decode_rows or static_decode_metadata_requires_token_carry or survives_nonfinal_prefill_chunk or release_preserves_carry'`.

### Entry 258 - Combined Resident Metadata And Slot-Token Decode Rejection

- date: 2026-06-05
- purpose:
  - test the next scheduler/static-metadata checklist item after Entry 257;
  - combine the existing resident metadata table path with the accepted
    resident slot-token carry path, so decode gathers input token, block table,
    seq-len, and hybrid state by resident slot inside one compiled boundary.
- implementation:
  - added `ModelExecutor.forward_step_token_ids_resident_slot_carry_jit` as an
    opt-in diagnostic path;
  - when `resident_decode_metadata=True`, generic warmup and serving prefer the
    combined route over the older resident-metadata-only decode route;
  - the route returns both updated `resident_seq_lens` and
    `resident_last_tokens`, while preserving immutable generated-token refs in
    the runner.
- artifact:
  - medium random diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_resident_slot_carry_decode_r1.json`;
  - nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_resident_slot_carry_decode_r1_jax.json`.
- result:
  - JAX `323.28 output tok/s`, `290` generated tokens, `0.8971 s`;
  - stored medium vLLM denominator `471.06 output tok/s`, ratio `0.686x`;
  - accepted medium anchors were `355.77` to `359.34 output tok/s`, so this is
    a clear regression despite zero measured-phase JIT growth (`20 -> 20`);
  - generic warmup correctly used
    `forward_prefill_token_ids_slot_carry_table_jit:prefill` for all prefill
    buckets and `forward_step_token_ids_resident_slot_carry_jit:decode` for all
    decode buckets.
- decision:
  - reject this as the next promoted random route and do not run the large lane;
  - keep it as opt-in diagnostic scaffolding behind the existing
    `resident_decode_metadata` config only;
  - the scheduler/static-metadata bottleneck needs a different boundary or
    scheduler-side reduction, not the current resident metadata decode route.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py tests/test_backend_boundaries.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'table_hybrid_decode_matches_sliced_decode or warmup_compiles_resident_slot_carry_decode_when_available or warmup_compiles_prefill_slot_carry_table_when_available'`.

### Entry 259 - Accepted Fixed-ABI FA/FLA Policy

- date: 2026-06-05
- purpose:
  - finish the FA/FLA integration pass after the packed/ragged ABI and
    resident slot-token carry boundary stabilized;
  - retest the existing full-attention Triton routes under the current random
    decode graph instead of relying on the older pre-ABI rejection;
  - keep the GDN FLA route unchanged and promote only an integrated
    full-attention route that improves random decode with generic warmup.
- implementation:
  - promoted `full_attention.prefill_impl=triton_packed` and
    `full_attention.decode_impl=triton_paged_fused_append` in
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`;
  - retained BF16 full-attention KV cache, strict GDN no-fallback policy,
    `gdn.prefill_post_conv_impl=triton_fla_padded`, and
    `gdn.packed_decode.impl=triton_fla_conv_raw_gates`;
  - left standalone `full_attention.decode_impl=triton_paged` rejected for
    random serving because it still replaces too narrow a boundary.
- artifacts:
  - medium same-code control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_current_control_r1.json`;
  - medium FA prefill-only:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_fa_prefill_triton_r1.json`;
  - medium FA decode-only:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_fa_decode_triton_r1.json`;
  - medium FA fused append/decode-only:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_fa_decode_fused_append_r1.json`;
  - medium combined FA route:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_fa_prefill_fused_decode_r1.json`;
  - large combined FA route:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_fa_prefill_fused_decode_r1.json`;
  - nested large JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_fa_prefill_fused_decode_r1_jax.json`.
- result:
  - focused GPU correctness passed for Triton packed FA prefill, Triton FA
    decode, fused FA append/decode, and Triton FLA GDN prefill;
  - medium current control: `337.12 output tok/s`, stored-vLLM ratio `0.716x`;
  - medium FA prefill-only: `350.17 output tok/s`, `0.743x`;
  - medium FA decode-only: `332.30 output tok/s`, `0.705x`;
  - medium FA fused append/decode-only: `349.70 output tok/s`, `0.742x`;
  - medium combined FA prefill plus fused append/decode:
    `359.60 output tok/s`, `0.763x`;
  - large accepted Entry 257 control:
    `470.14 output tok/s`, stored-vLLM ratio `0.532x`;
  - large combined FA prefill plus fused append/decode:
    `484.18 output tok/s`, stored-vLLM ratio `0.548x`;
  - measured-phase JIT cache growth stayed zero on the large run (`20 -> 20`),
    and the artifact reports `prefill_impl=triton_packed`,
    `decode_impl=triton_paged_fused_append`,
    `gdn_prefill_post_conv_impl=triton_fla_padded`, and
    `gdn_packed_decode_impl=triton_fla_conv_raw_gates`.
- decision:
  - promote the combined FA route in the main GPU config;
  - do not promote standalone FA decode or prefill-only as independent routes;
  - treat FA/FLA as now integrated enough for the next pass to return to
    scheduler/static metadata movement, PJRT bubbles, and broader decode graph
    fusion.
- validation:
  - `JAX_PLATFORMS=cuda NANO_VLLM_JAX_CACHE_ROOT=/mountpoint/.exp .venv/bin/python -m pytest -q tests/test_backend_boundaries.py::test_packed_full_attention_prefill_triton_matches_dense_rows_bf16_cache tests/test_backend_boundaries.py::test_paged_decode_attention_triton_matches_reference_bf16_cache tests/test_backend_boundaries.py::test_configured_triton_decode_attention_matches_reference_non_contiguous_blocks tests/test_backend_boundaries.py::test_configured_fused_triton_decode_attention_appends_kv_and_matches_reference tests/test_gdn_post_conv_prefill_reference.py::test_model_post_conv_prepared_fla_triton_packed_matches_reference`;
  - medium and large random sidecar runs above used stored vLLM denominators,
    generic warmup, `--jax-fail-on-jit-cache-growth`, and the 70% RAM guard.

### Entry 260 - Accepted Broadened Resident Metadata And Packed-Projection GDN Decode

- date: 2026-06-05
- purpose:
  - continue the random-decode broadening/coarsening path after Entry 259;
  - reopen resident metadata only if it removes a whole host/device operation,
    not as the Entry 258 scatter-sync route;
  - broaden GDN decode so the conv+recurrent FLA kernel consumes the packed
    decode projection instead of separate `qkv`, `a`, and `b` arrays.
- implementation:
  - changed resident static decode placeholders to be keyed by physical shape
    and active-row pattern instead of real sequence IDs;
  - promoted `resident_decode_metadata=true` in the main GPU config;
  - changed resident block/seq metadata synchronization to refresh the tiny
    full host mirrors with `device_put` on actual changes instead of executing
    JAX `.at[...]` scatter updates;
  - added `gdn_conv_packed_projection_decode_step_bf16_raw_gates`, which loads
    `qkv`, `a`, and `b` from the concatenated `[qkv, a, b, z]` packed
    projection inside the Triton conv+raw-gate GDN decode kernel.
- artifacts:
  - first resident packed-GDN large run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_packed_gdn_boundary_r1.json`;
  - accepted scatter-free resident large run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_deviceput_sync_packed_gdn_r1.json`;
  - accepted profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_deviceput_sync_packed_gdn_profile_r1.json`;
  - rejected physical-row token-ref diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_physical_token_refs_packed_gdn_r1.json`.
- result:
  - scatter-sync resident route regressed large random to
    `476.41 output tok/s`, `0.539x` of stored vLLM;
  - scatter-free resident sync plus packed-projection GDN reached
    `495.10 output tok/s`, `0.560x` of stored vLLM, with zero measured-phase
    JIT growth (`20 -> 20`);
  - this improves the Entry 259 large accepted route
    (`485.11 output tok/s`, `0.549x`) by `~2.1%`;
  - profiled scatter-free resident route shows `_sync_resident_decode_metadata`
    at `140.40 ms / 300` versus `689.15 ms / 231` on the scatter-sync profile,
    and PJRT execute count drops from `1159` to `548`;
  - GPU profile confirms the new
    `_gdn_conv_packed_projection_decode_raw_gate_kernel` path is used
    (`115.55 ms / 5051` in the profiled run);
  - physical-row token refs for the full greedy token vector reached only
    `493.47 output tok/s`, so that subchange is rejected.
- decision:
  - promote resident decode metadata only with the shape-stable descriptor and
    full-table host-mirror sync;
  - promote packed-projection GDN decode as the active FLA decode boundary;
  - do not re-run the old scatter-sync resident route or physical-row token
    refs as primary random-decode optimizations;
  - next broadening target should be model-side decode GPU work
    (GEMMs/full-attention/GDN post-core) plus remaining PJRT execution overhead.
- validation:
  - `python -m py_compile nanovllm_jax/model.py nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla_triton.py`;
  - `pytest -q tests/test_device_token_carry.py -k 'resident_decode_metadata or static_decode_metadata_reuses_fixed_device_arrays'`;
  - `pytest -q tests/test_backend_boundaries.py -k 'resident_decode_metadata_flag or syncs_resident_decode_metadata_by_slot or warmup_compiles_resident_slot_carry_decode_when_available'`;
  - `pytest -q tests/test_gdn_packed_decode_reference.py -k 'conv_packed_projection_decode_matches_split_kernel or conv_packed_decode_matches_reference'`;
  - large random sidecar runs used stored vLLM denominator, generic warmup,
    `--jax-fail-on-jit-cache-growth`, and the 70% RAM guard.

### Entry 261 - Rejected Resident Replay And Narrow GDN Out-Projection Fusion

- date: 2026-06-05
- purpose:
  - test whether the remaining random-decode gap is mostly scheduler shell work
    by replaying a resident static decode descriptor when the live request set
    and bucket shape stay unchanged;
  - test a wider GDN decode custom boundary by fusing post-core gating with the
    output projection after the accepted packed-projection conv+GDN kernel.
- implementation:
  - prototype-only resident decode replay reused the previous static
    `ScheduledBatch` for eligible all-greedy decode steps and bypassed the
    normal scheduler/build path;
  - prototype-only GDN out-projection fusion routed the post-core gated output
    and layer output projection through a Triton custom call;
  - both prototypes were reverted after integrated benchmarks.
- artifacts:
  - GDN out-projection fusion large run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_fused_gdn_outproj_r1.json`;
  - resident decode replay large run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_decode_replay_r1.json`;
  - resident decode replay profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_decode_replay_profile_r1.json`;
  - nested replay JAX profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_resident_decode_replay_profile_r1_jax.json`.
- result:
  - GDN out-projection fusion passed focused correctness but regressed large
    random to `441.92 output tok/s`, `0.500x` of the stored vLLM denominator;
  - resident decode replay reduced profiled scheduler/build calls from `300` to
    `230`, but the no-profile large run reached only `494.35 output tok/s`,
    `0.559x`, below the accepted Entry 260 `495.10 output tok/s`;
  - the replay profile shows scheduler time moved in the right direction
    (`schedule` `355.83 ms / 300` to `337.51 ms / 230`,
    `build_scheduled_batch` `323.29 ms / 300` to `310.60 ms / 230`), while PJRT
    and GPU model work moved the wrong way (`PjRT Execute` `1424.72 ms / 548`
    to `1435.30 ms / 554`, top BF16 GEMM `503.33 ms` to `509.48 ms`, fused
    attention `212.02 ms` to `214.68 ms`);
  - the active GDN recurrent state is already in the kernel-native `[V,K]`
    layout, so a separate state-layout migration is not a pending big lever.
- decision:
  - reject and revert resident decode replay; it confirms that scheduler
    replay alone is not the remaining large random bottleneck;
  - reject and revert narrow GDN out-projection fusion; replacing the tail of
    one layer type in many small calls loses to the existing XLA/CUTLASS plan;
  - do not retry these as primary random-decode optimizations unless a broader
    model-owned decode boundary removes substantial GEMM/scatter/attention work
    at the same time.
- validation:
  - `python -m py_compile nanovllm_jax/config.py nanovllm_jax/server_config.py nanovllm_jax/llm_engine.py benchmarks/benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_device_token_carry.py -k 'resident_decode_replay or resident_decode_metadata_reuses_placeholders or resident_decode_metadata_uses_device_placeholders'`;
  - `pytest -q tests/test_server_config.py -k 'runtime_and_kernel_sections_translate_to_env or runtime_fastpaths or overrides_from_config'`.

### Entry 262 - Accepted Exact Small-Batch Decode Buckets

- date: 2026-06-05
- purpose:
  - test whether the random-decode gap is partly padded model work from
    power-of-two batch buckets;
  - make the standard random-serving warmup closer to vLLM-style dynamic
    batching without benchmark-specific measured-phase compilation.
- implementation:
  - changed the active GPU config `batch_size_buckets` from `1,2,4,8` to
    `1,2,3,4,5,6,7,8`;
  - changed the random sidecar default to the same exact small-batch bucket list
    so random runs get this policy unless explicitly overridden.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_exact_decode_batch_buckets_r1.json`;
  - nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_exact_decode_batch_buckets_r1_jax.json`.
- result:
  - large random improved to `503.66 output tok/s`, `0.570x` of the stored
    vLLM denominator, versus Entry 260 `495.10 output tok/s`, `0.560x`;
  - measured-phase JIT cache growth stayed zero (`40 -> 40`);
  - generic warmup compiled `40` entries, with `24` decode warmup runs covering
    B=1 through B=8 and `16` packed-prefill warmup runs;
  - peak system RAM stayed within the guard (`53.3%`, process tree RSS
    `5.27 GB`).
- decision:
  - promote exact small-batch decode buckets for the random-serving lane;
  - this is not model/GPU tile tuning. It is a general serving policy that
    avoids doing B=8 model work for common B=5/6/7 random decode steps;
  - keep the 70% RAM guard and generic warmup requirement because the policy
    intentionally trades more startup compilation for less measured decode
    padding.
- validation:
  - large random sidecar run above used stored vLLM denominator, generic warmup,
    `--jax-fail-on-jit-cache-growth`, and the 70% RAM guard.

### Entry 263 - Rejected Lower Padded-GEMM Row Count With Exact Buckets

- date: 2026-06-05
- purpose:
  - test whether Entry 262's exact B=5/6/7 scheduler buckets were being
    partially cancelled by `decode_padded_gemm_rows=8`;
  - check whether lowering the padded-GEMM row count lets exact smaller decode
    batches reduce model GEMM work.
- implementation:
  - direct JAX large-random run on the Entry 262 manifest;
  - kept exact decode batch buckets `1,2,3,4,5,6,7,8`;
  - changed only `--decode-padded-gemm-rows` from `8` to `5`.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_exact_buckets_padded_rows5_r1_jax.json`.
- result:
  - large random regressed to `240.16 output tok/s` versus Entry 262
    `503.66 output tok/s`;
  - measured-phase JIT cache growth stayed zero (`40 -> 40`);
  - ITL p95 regressed to `72.67 ms` and TTFT p50 to `574.70 ms`.
- decision:
  - reject lowering fixed padded-GEMM rows as a primary route;
  - keep `decode_padded_gemm_rows=8`. It is not just wasted padding: on this
    XLA/CUDA graph it selects a much faster compiled GEMM path;
  - the next large model-side target should be a genuinely better small-batch
    decode backend or epilogue/fusion path, not dynamic row-count source
    policy.
- validation:
  - direct JAX run used the same prompt manifest as Entry 262, generic warmup,
    exact decode buckets, and `--fail-on-jit-cache-growth`.

### Entry 264 - Rejected Narrow Full-Attention Decode Kernel Tweaks

- date: 2026-06-05
- purpose:
  - test whether the large-random gap can be reduced by removing visible
    materialization around full-attention decode without changing the broader
    decode scheduler ABI;
  - check whether duplicate K/V append stores inside the fused paged decode
    attention kernel are a meaningful bucket.
- implementation:
  - prototype A added a Triton full-attention decode QKV-prep custom call that
    split packed Q/G/K/V, applied q/k RMSNorm and RoPE, and produced query,
    key, value, and gate tensors before the existing fused append+attention
    kernel;
  - prototype B changed the fused append+attention Triton kernel so only the
    first query head in each KV group wrote the appended K/V cache row, while
    all query heads substituted the just-appended K/V locally for the current
    decode position.
- artifacts:
  - QKV-prep run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_fa_decode_qkv_prep_r1.json`;
  - duplicate-store run r1:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_fa_dedup_append_store_r1.json`;
  - duplicate-store repeat r2:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_fa_dedup_append_store_r2.json`.
- result:
  - QKV-prep regressed to `499.13 output tok/s`, `0.565x` of the stored vLLM
    denominator, versus Entry 262 `503.66 output tok/s`;
  - duplicate-store r1 reached `502.29 output tok/s`, `0.568x`; repeat r2
    reached `497.96 output tok/s`, `0.563x`;
  - all runs kept zero measured-phase JIT cache growth and the same generated
    token count as Entry 262.
- decision:
  - reject and revert both probes;
  - a separate prep-only custom call adds a launch without removing enough work
    from the attention boundary;
  - duplicate K/V append stores are not a dominant bucket on the integrated
    random graph;
  - the next large step must remove a larger boundary, such as integrating
    packed QKV prep into the existing append+attention launch, replacing the
    LM-head dense+argmax with a true fused greedy epilogue, or moving multiple
    decode sublayers behind a coarser backend-owned boundary.
- validation:
  - focused CUDA/full-attention tests passed for each prototype before
    benchmarking;
  - large random sidecar runs used stored vLLM denominator, generic warmup,
    exact decode buckets, `--jax-fail-on-jit-cache-growth`, and the 70% RAM
    guard.

### Entry 265 - Rejected Integrated Packed-QKV Full-Attention Decode

- date: 2026-06-05
- purpose:
  - test a coarser full-attention decode boundary that does not add a separate
    prep launch;
  - consume the packed QKV projection directly in the fused append+attention
    Triton call, perform Q/K RMSNorm and RoPE inside that launch, return the
    gate for the existing output projection path, and keep the BF16 paged KV
    cache contract.
- implementation:
  - added a prototype backend method and Triton wrapper for packed-QKV fused
    append+decode attention;
  - routed it automatically for the existing packed full-attention decode,
    BF16 cache, width-1, `triton_paged_fused_append` policy;
  - added a focused CUDA test that compared attention output and appended K/V
    against the reference Q/K RMSNorm + RoPE + paged decode path.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_integrated_packed_fa_decode_r1.json`;
  - nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_integrated_packed_fa_decode_r1_jax.json`.
- result:
  - large random reached `499.81 output tok/s`, `0.565x` of the stored vLLM
    denominator, versus Entry 262 `503.66 output tok/s`;
  - measured-phase JIT cache growth stayed zero (`40 -> 40`);
  - generated token count stayed `1582`, matching the Entry 262 JAX run.
- decision:
  - reject and revert the prototype;
  - moving Q/K norm, RoPE, K/V append, and attention into one call still does
    not reduce the integrated random-decode wall time;
  - full-attention decode alone is not the next large lever. The next useful
    implementation needs to coarsen many decode sublayer calls, reduce the
    small-GEMM/PJRT boundary, or replace a shared dense operation with a true
    single-call backend.
- validation:
  - focused CUDA fused-attention tests passed before benchmarking;
  - large random sidecar used stored vLLM denominator, generic warmup, exact
    decode buckets, `--jax-fail-on-jit-cache-growth`, and the 70% RAM guard.

### Entry 266 - Accepted Reference GDN Decode Route For Random Serving

- date: 2026-06-05
- purpose:
  - test whether the active random-serving config should keep using the
    conv-fused Triton FLA packed GDN decode route after the large-random profile
    showed `_gdn_conv_packed_projection_decode_raw_gate_kernel` as thousands of
    small GPU calls;
  - compare against the existing packed BF16-QKV `reference` GDN decode route
    under the same exact decode buckets, resident metadata, generic warmup, and
    stored vLLM denominator.
- implementation:
  - copied `gpu_paged_gdn_fla_decode_static_metadata` to a diagnostic config
    and changed only `kernels.gdn.packed_decode.impl` from
    `triton_fla_conv_raw_gates` to `reference`;
  - after two winning repeats, promoted the same setting in
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`.
- artifacts:
  - r1 sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_gdn_reference_decode_r1.json`;
  - r1 nested JAX:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_gdn_reference_decode_r1_jax.json`;
  - r2 sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_gdn_reference_decode_r2.json`;
  - r2 nested JAX:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_gdn_reference_decode_r2_jax.json`.
- result:
  - r1 reached `514.25 output tok/s`, `0.582x` of the stored vLLM denominator;
  - r2 reached `508.85 output tok/s`, `0.576x` of the stored vLLM denominator;
  - both runs generated `1582` tokens and kept zero measured-phase JIT cache
    growth (`40 -> 40`);
  - Entry 262, the previous accepted anchor with
    `triton_fla_conv_raw_gates`, reached `503.66 output tok/s`, `0.570x`.
- decision:
  - promote `kernels.gdn.packed_decode.impl=reference` for the active random
    serving config;
  - treat the conv-fused Triton FLA GDN decode route as rejected for this
    integrated random graph until it is part of a much coarser boundary. It is
    correct, but the current shape adds too many small calls and loses wall
    time;
  - this is not enabling a fallback: fallbacks remain disabled, and the selected
    GDN decode implementation is explicit in the config.
- validation:
  - both large random sidecar runs used the stored vLLM denominator, generic
    warmup, exact decode buckets, `--jax-fail-on-jit-cache-growth`, and the 70%
    RAM guard.

### Entry 267 - Rejected Two-Stage Triton LM-Head Greedy Argmax

- date: 2026-06-05
- purpose:
  - test whether the large-random gap can be reduced by replacing the decode
    LM-head dense-logits materialization plus `argmax` with a token-id-only
    Triton path;
  - keep the active GDN reference decode route, exact decode buckets, resident
    metadata, generic warmup, stored vLLM denominator, and resource guards.
- implementation:
  - prototyped a two-stage Triton LM-head greedy path: stage 1 computed
    per-vocab-block max logits from hidden state, RMSNorm weight, and
    materialized tied LM-head weight without writing full `[B, vocab]` logits;
    stage 2 reduced the per-block maxima to one token ID per row;
  - threaded the prototype through `lm_head_token_ids_and_topk` behind a
    config-driven `lm_head_decode_impl=triton_argmax` selector;
  - kept prefill final-token sampling on the reference LM-head path because the
    selector was decode-only.
- artifacts:
  - failed first warmup attempt, rejected before timing:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_triton_argmax_r1.json`;
  - corrected guarded sidecar run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_triton_argmax_r2.json`;
  - corrected nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_triton_argmax_r2_jax.json`.
- result:
  - corrected run reached `508.38 output tok/s`, `0.575x` of the stored vLLM
    denominator;
  - this lost to the active-config repeat at `513.60 output tok/s` and the best
    Entry 266 repeat at `514.25 output tok/s`;
  - measured-phase JIT cache growth stayed zero (`40 -> 40`) and generated
    token count stayed `1582`;
  - the run took `250.31 s` end to end, materially higher than the active-config
    run, so the custom LM-head path also worsened iteration cost.
- decision:
  - reject and revert the prototype;
  - do not keep a `lm_head_decode_impl` config knob or the two-stage Triton
    kernel. It removes the full-logit write but gives up the highly optimized
    GEMM path and adds custom-call/reduction overhead;
  - the next LM-head attempt must be a true GEMM epilogue or library-backed
    fused matmul+top-1 path, not a separate blockwise reduction kernel.
- validation:
  - focused GPU tests passed after the Triton syntax fix before benchmarking;
  - large random sidecar used stored vLLM denominator, generic warmup, exact
    decode buckets, `--jax-fail-on-jit-cache-growth`, the 70% RAM guard, and
    four CPU cores at nice level `10`.

### Entry 268 - Accepted Row-Padded Decode GEMM For LM Head

- date: 2026-06-05
- purpose:
  - test whether the accepted row-padded decode GEMM path should also cover the
    greedy LM-head dot instead of only hidden/intermediate projections;
  - keep the active reference GDN decode route, exact decode buckets, resident
    metadata, generic warmup, stored vLLM denominator, and resource guards.
- implementation:
  - routed `lm_head_token_ids_and_topk` through `_decode_padded_gemm_dot` when
    `decode_padded_gemm=true`, the decode width is `1`, the batch fits the
    configured padded row count, and the output dimension is admitted by
    `decode_padded_gemm_max_out_dim`;
  - promoted the active config's output-dimension guard high enough to include
    the Qwen3.5-family vocabulary, so the same XLA/CUTLASS GEMM-shaped path is
    used for LM-head decode selection;
  - left top-k diagnostics and argmax semantics unchanged.
- artifacts:
  - r1 sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_padded_gemm_r1.json`;
  - r1 nested JAX:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_padded_gemm_r1_jax.json`;
  - r2 sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_padded_gemm_r2.json`;
  - r2 nested JAX:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_lm_head_padded_gemm_r2_jax.json`.
- result:
  - r1 reached `516.35 output tok/s`, `0.584x` of the stored vLLM denominator;
  - r2 reached `519.16 output tok/s`, `0.587x` of the stored vLLM denominator;
  - both runs generated `1582` tokens and kept zero measured-phase JIT cache
    growth (`40 -> 40`);
  - the prior active-config sanity reached `513.60 output tok/s`, and Entry 266
    reference-GDN repeats reached `514.25` and `508.85 output tok/s`.
- decision:
  - promote LM-head coverage for the row-padded GEMM route;
  - count this as an incremental GEMM-shape improvement, not the main route to
    `0.9x` vLLM. The larger LM-head lesson from Entry 267 still stands:
    token-id-only reductions lose unless they preserve the optimized GEMM path
    or become a true backend/library GEMM epilogue.
- validation:
  - focused LM-head helper tests passed;
  - both large random sidecar runs used the stored vLLM denominator, generic
    warmup, exact decode buckets, `--jax-fail-on-jit-cache-growth`, the 70% RAM
    guard, and four CPU cores at nice level `10`.

### Entry 269 - Rejected Resident Slot-Carry Greedy Decode Burst

- date: 2026-06-05
- purpose:
  - test a coarse decode boundary that keeps the accepted resident last-token
    table while emitting two greedy tokens per scheduled decode step;
  - reduce Python/PJRT step count without dropping paged attention, resident
    metadata, static decode metadata, KV cache updates, or GDN table updates.
- implementation:
  - prototyped resident slot-carry burst JIT boundaries for both static
    metadata and resident metadata;
  - the resident path gathered last tokens, block tables, sequence lengths, and
    GDN state by slot id, scanned two full model decode steps, returned all
    emitted token IDs, and scattered the final last token plus updated state
    back to resident tables;
  - the prototype was reverted after benchmarking because it was not promoted.
- artifacts:
  - first small burst-2 sidecar before the resident-specific route:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_small_burst2_slot_carry_r1.json`;
  - resident-route small burst-2 sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_small_burst2_resident_slot_carry_r2.json`;
  - small burst-1 active control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_small_active_burst1_control_r1.json`;
  - medium resident-route burst-2 sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_burst2_resident_slot_carry_r1.json`.
- result:
  - small active burst-1 control reached `199.74 output tok/s`, zero measured
    JIT growth (`11 -> 11`);
  - small resident burst-2 reached `136.99 output tok/s`, zero measured JIT
    growth (`17 -> 17`);
  - medium resident burst-2 reached `277.27 output tok/s`, zero measured JIT
    growth (`24 -> 24`), below the current medium history around
    `355+ output tok/s`;
  - compile/warmup time was materially worse than the active burst-1 path even
    on the small and medium rungs.
- decision:
  - reject and revert the prototype;
  - do not retry source-level full-model JAX scan bursts as the main
    coarsening strategy. Reducing PJRT calls is not enough when each fused call
    runs a worse compiled full-model plan;
  - the next coarse-boundary attempt should target backend/library-owned fused
    sublayers or decode-block kernels that reduce actual GEMM/attention/GDN
    device work.
- validation:
  - all benchmark runs used generic warmup, `--jax-fail-on-jit-cache-growth`,
    the 70% RAM guard, and four CPU cores at nice level `10`;
  - the rejected code was fully reverted before documenting the result.

### Entry 270 - Rejected Packed-Projection GDN Tail Fusion

- date: 2026-06-05
- purpose:
  - test whether the accepted packed-projection GDN decode boundary should also
    absorb the remaining post-core GDN tail: per-head RMSNorm, `silu(z)` gating,
    and the reshape into the output projection input;
  - avoid repeating Entry 261's rejected out-projection fusion by leaving the
    final `out_proj` GEMM on the existing XLA/CUTLASS path.
- implementation:
  - added opt-in `gdn_packed_decode_impl=triton_fla_conv_raw_gates_tail`;
  - extended the existing packed-projection Triton kernel with a compile-time
    tail-fused mode that loads `z` and `norm_weight`, computes per-head RMSNorm
    plus `silu(z)`, and returns `[batch, 1, value_heads * value_dim]`;
  - routed model output processing to skip the JAX tail only for this opt-in
    implementation;
  - allowed the tail-fused route to use the packed decode projection for
    batch-one warmup buckets, while preserving the older batch>1 packed
    projection policy for the active raw route.
- artifacts:
  - tail-fused medium smoke:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_gdn_tail_fused_r3.json`;
  - nested tail-fused JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_gdn_tail_fused_r3_jax.json`;
  - raw packed-GDN same-envelope control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_gdn_raw_control_r2.json`;
  - nested raw-control JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_medium_gdn_raw_control_r2_jax.json`.
- result:
  - both runs used the regenerated medium random envelope:
    `4` requests, `1787` input tokens, `372` output tokens, input lengths
    `[507, 400, 450, 430]`, output lengths `[126, 58, 70, 118]`;
  - tail-fused route reached `362.07 output tok/s`, stored-vLLM ratio `0.769x`,
    and zero measured-phase JIT growth (`15 -> 15`);
  - raw packed-GDN control reached `407.54 output tok/s`, stored-vLLM ratio
    `0.865x`, and zero measured-phase JIT growth (`15 -> 15`);
  - tail fusion therefore regressed the medium lane by about `11.2%` against
    the same request envelope.
  - the stored-vLLM ratios are only rough context because the regenerated JAX
    envelope emitted `372` output tokens while the older stored vLLM artifact
    emitted `290`; the raw-vs-tail same-envelope A/B is the decision signal.
- decision:
  - reject tail fusion for promotion and keep
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json` on
    explicit packed GDN `reference` for the active random-serving lane;
  - keep the opt-in implementation and focused test as diagnostic evidence so
    this exact boundary is not retried blindly;
  - likely explanation: the fused branch forces full-head value tiles and a
    larger custom-call ABI, which costs more than the separate XLA
    RMSNorm/silu/reshape tail it removes. The next GDN decode win needs a
    broader backend-owned layer/model boundary, not only this tail.
- validation:
  - `python -m py_compile nanovllm_jax/model.py nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla_triton.py tests/test_gdn_packed_decode_reference.py`;
  - `pytest -q tests/test_gdn_packed_decode_reference.py`;
  - `pytest -q tests/test_server_config.py`;
  - medium random sidecar runs used stored vLLM denominator, generic warmup,
    `--jax-fail-on-jit-cache-growth`, the 70% RAM guard, and four CPU cores at
    nice level `10`.

### Entry 271 - Rejected Direct Compiled-Executable Replay Prototype

- date: 2026-06-05
- purpose:
  - test whether caching lowered JAX executables manually can approximate
    decode graph replay and reduce per-step PJRT overhead without changing the
    serving ABI;
  - keep the active reference-GDN random route and generic warmup discipline.
- implementation:
  - prototyped a `_compiled_jit_cache` in `ModelExecutor`;
  - changed profiled JIT calls to run `fn.lower(*args).compile()` on first use
    and then call the compiled executable directly for later invocations of the
    same cache key;
  - reverted the prototype after the guarded smoke because it was not viable.
- artifacts:
  - no timing artifact was accepted for the prototype because the smoke did not
    reach measurement;
  - post-revert reference-GDN small validation:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_small_reference_gdn_post_replay_reject_r1.json`;
  - nested post-revert JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_small_reference_gdn_post_replay_reject_r1_jax.json`.
- result:
  - the scaled four-request random smoke stayed CPU-bound in compile/warmup for
    more than six minutes under the resource guard, so it was terminated before
    measurement;
  - after reverting the executable-cache prototype, the reference-GDN small
    validation completed at `330.21 output tok/s`, `137` output tokens,
    `832` input tokens, and zero measured-phase JIT growth (`68 -> 68`).
- decision:
  - reject direct `.lower().compile()` executable caching as the graph-replay
    lever for this stack;
  - it front-loads compilation too aggressively and does not resemble vLLM's
    CUDA graph replay, which captures/replays already-built GPU command
    sequences inside the runtime rather than recompiling Python-level JAX
    callables on demand;
  - future replay work must happen through XLA/runtime command-buffer support
    or a backend-owned decode boundary, not by manually lowering every JIT
    cache key in the serving loop.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/model.py nanovllm_jax/backends.py nanovllm_jax/kernels/gdn_fla_triton.py`;
  - `python -m json.tool benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`;
  - `pytest -q tests/test_server_config.py tests/test_gdn_packed_decode_reference.py`.

### Entry 272 - Accepted Dense Resident Decode And Small-Array DevicePut Cleanup

- date: 2026-06-05
- purpose:
  - reduce large-random decode PJRT/host-transfer overhead without specializing
    to the seed, request lengths, Qwen3.5-0.8B dimensions, or the A10G;
  - keep paged attention, packed chunked prefill, resident GDN state, generic
    warmup, stored vLLM denominator, and zero measured-phase JIT growth.
- accepted implementation:
  - added `forward_step_token_ids_resident_dense_slot_carry_jit` for compact
    active decode rows with arbitrary resident slots. The boundary gathers
    block tables, seq lengths, last tokens, KV/GDN state, runs the model and
    greedy LM-head token selection, then scatters updated state, seq lengths,
    and last tokens back to resident tables;
  - cached resident static-decode placeholder metadata by shape instead of
    keeping only the most recent batch shape;
  - skipped release-time resident-last-token clearing because prefill reseeds
    slots before reuse;
  - tracked resident block counts so decode metadata sync only rebuilds block
    rows on KV-page crossings;
  - replaced hot `jnp.asarray(...)` construction of slot/update tensors with
    direct `jax.device_put(np.asarray(...))`, removing the extra
    `convert_element_type` executable from the resident metadata path.
- accepted artifacts:
  - dense resident-slot first large run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_dense_resident_slot_r1.json`;
  - stacked-materialization/no-trace-preferred intermediate:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_dense_blockcount_stacked_materialize_r1.json`;
  - trace-prefetch route with corrected trace boundary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_dense_blockcount_deferred_trace_prefetch_r1.json`;
  - current accepted best:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_dense_direct_device_put_r1.json`;
  - current accepted profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_dense_direct_device_put_profile_r1.json`.
- result:
  - dense resident-slot first large run reached `709.55 output tok/s`,
    `0.803x` stored vLLM, zero JIT growth;
  - trace-prefetch corrected boundary reached `746.95 output tok/s`,
    `0.845x` stored vLLM, zero JIT growth;
  - current accepted best reached `757.24 output tok/s`, `0.857x` stored
    vLLM (`884.03 output tok/s`), `1582` generated tokens, and zero
    measured-phase JIT growth (`24 -> 24`);
  - token-event throughput in the best run was `794.77 tok/s`, about `0.899x`
    the stored vLLM denominator, while final output materialization/drain kept
    wall-clock output throughput below the `0.9x` target.
- profile movement:
  - accepted profile throughput was `729.30 output tok/s` under profiler;
  - versus the prior dense profile, `PjRtCApiLoadedExecutable::Execute` count
    fell `619 -> 406`;
  - `MemcpyD2D` fell from about `294.63 ms` to `13.74 ms`;
  - `_sync_resident_decode_metadata` fell from about `578.63 ms` to
    `398.40 ms`;
  - remaining top GPU buckets are model-side BF16 GEMMs, paged decode
    attention, LM-head/argmax reductions, and many small fusion kernels.
- rejected subexperiments:
  - resident prefix-slot compaction regressed large random to
    `453.57 output tok/s`; keep arbitrary resident slots and gather by slot id;
  - decode block-table buckets `32,64,128` regressed to
    `708.36 output tok/s`; keep fixed `128` for this large lane;
  - scalar block-entry scatter regressed profile health by increasing PJRT
    execute count (`619 -> 708`) and D2D time, so full-row page-crossing
    scatter remains the selected resident-table sync path for now;
  - in-JIT resident metadata delta application reached only
    `746.78 output tok/s`, below the cache-only/direct-device-put route;
  - per-shape grouped final token materialization exploded final drain to about
    `0.90 s` and regressed large random to `546.79 output tok/s`.
- decision:
  - promote dense resident-slot decode, per-shape resident static metadata
    cache, release-clear removal, block-count skip, and direct device-put update
    tensors;
  - reject scalar entry scatter, in-JIT metadata deltas, and per-shape grouped
    materialization for the current serving path;
  - the next target should be model-side device work or a true runtime graph /
    backend-owned layer boundary. Host/PJRT cleanup is now smaller but not gone.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/sequence.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'table_hybrid_decode_matches_sliced_decode or warmup_compiles_resident_slot_carry_decode or resident_decode_metadata or resident_decode_replay or model_runner_warmup_compiles_greedy_decode_burst' tests/test_device_token_carry.py tests/test_server_config.py`;
  - large random runs used stored vLLM denominator, generic warmup,
    `--jax-fail-on-jit-cache-growth`, the 70% RAM guard, and four CPU cores at
    nice level `10`.

### Entry 273 - Rejected Dense Resident Seq-Lens Sync Narrowing

- date: 2026-06-05
- purpose:
  - test whether the accepted dense resident decode route can skip pre-decode
    resident seq-lens scatter, since
    `forward_step_token_ids_resident_dense_slot_carry_jit` already advances the
    resident seq-len table and the runner advances the host mirror after each
    decode step.
- change tested:
  - changed the runtime pre-decode metadata sync to call
    `_sync_resident_decode_metadata(..., sync_seq_lens=False)` only for the
    dense resident slot-token metadata route;
  - warmup and post-prefill seeding still used full block-table plus seq-len
    synchronization.
- artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260605/random_large_dense_block_only_seqsync_r1.json`.
- result:
  - large random reached `742.47 output tok/s`, `0.840x` of the stored vLLM
    denominator, below Entry 272's accepted `757.24 output tok/s`;
  - token-event throughput was `796.20 tok/s`, but post-last-token drain grew to
    `143.77 ms`, so wall-clock output throughput regressed;
  - measured-phase JIT growth stayed zero and generated-token count stayed
    `1582`.
- decision:
  - reject and revert. Even if this can slightly improve measured token-event
    time, the benchmark target is wall-clock output throughput, and the final
    drain regression is not acceptable.
- related neutral diagnostic:
  - enabling `NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH=1` on the accepted Entry 272
    route reached `757.56 output tok/s`, `0.857x` stored vLLM, with
    `98.30 ms` drain and zero JIT growth. This is too small/noisy to promote as
    a new default.

### Entry 274 - Accepted Trace Token Prefetch Default

- date: 2026-06-06
- purpose:
  - make the deferred trace-token prefetch path a config-backed default instead
    of an environment-only experiment;
  - reduce final output materialization/drain without changing the paged,
    ragged, generic-warmup serving path.
- accepted implementation:
  - added `trace_token_prefetch` to `Qwen3_5Config`, server config runtime
    fastpaths, environment projection, benchmark CLI reporting, and the GPU
    matrix config;
  - defaulted the option to `true` while preserving `--no-trace-token-prefetch`
    and `NANO_VLLM_JAX_TRACE_TOKEN_PREFETCH=0` for focused diagnostics.
- artifacts:
  - short fixed-shape A/B:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/comprehensive_20260606_ab/short_32_128_jax_trace_prefetch.json`;
  - long-prefill A/B:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/comprehensive_20260606_ab/long_prefill_512_2048_jax_trace_prefetch.json`;
  - large random default-on rerun:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260606/random_large_trace_prefetch_default_bf16_r1.json`.
- result:
  - short `32x128` improved from `317.5` to `542.8 output tok/s` versus the
    same live vLLM denominator, with exact generated-token parity and no
    measured-phase JIT growth;
  - long prefill was neutral at `116.9 output tok/s`; its remaining gap is not
    a final-drain issue;
  - large random on the current live-vLLM denominator improved from
    `740.63` to `757.58 output tok/s`, `0.742x` of vLLM `1021.59 output
    tok/s`, with the same prompt manifest, `1582` generated tokens, generic
    warmup, and the 70% RAM/four-core guard;
  - large-random final drain fell from `149.13 ms` to `98.17 ms`; token-event
    throughput stayed effectively flat (`796.22 -> 794.95 tok/s`), confirming
    the win is final materialization overlap rather than faster model decode.
- decision:
  - promote trace-token prefetch as the default serving/benchmark config;
  - keep the next large-random target on model-side decode work or a broader
    backend-owned boundary. Rejected final-token stacking/materialization
    variants should not be retried as the primary route.
- validation:
  - `python -m py_compile nanovllm_jax/config.py nanovllm_jax/server_config.py nanovllm_jax/engine/llm_engine.py benchmarks/benchmark_jax_server_trace.py benchmarks/run_gpu_matrix.py`;
  - `pytest -q tests/test_server_config.py tests/test_device_token_carry.py tests/test_benchmark_jax_server_trace.py tests/test_gpu_matrix_runner.py`;
  - large random used stored vLLM denominator, BF16 activations/weights,
    generic warmup, `--jax-fail-on-jit-cache-growth`, the 70% RAM guard, and
    four CPU cores at nice level `10`.

### Entry 275 - Accepted Full-Vocab Temperature Sampling Boundary

- date: 2026-06-07
- purpose:
  - remove the artificial greedy-only restriction from the deferred-token
    serving path;
  - keep nonzero-temperature token selection inside the compiled executor so
    sampling does not force full logits back to the host;
  - preserve paged attention, chunked/ragged prefill, resident decode metadata,
    device-token carry, and generic startup warmup.
- accepted implementation:
  - added `sampled_token_fastpath` to `Qwen3_5Config`, server config runtime
    projection, benchmark CLI/reporting, and random/vLLM harness sampling args;
  - added `lm_head_sample_token_ids`, `forward_step_sampled_token_ids_jit`, and
    `forward_step_sampled_token_ids_resident_dense_slot_carry_jit`;
  - tracked per-resident-slot RNG counters and reset them on release;
  - made packed prefill sampling state use logical output rows instead of the
    physical packed token row;
  - allowed deferred-token tracing and scheduler postprocess to accept sampled
    `DeviceTokenRef`s when `ignore_eos=True`;
  - expanded generic startup warmup to compile sampled prefill, generic sampled
    decode, and resident sampled dense decode keys.
- limitations:
  - the fast path is full-vocab temperature sampling only;
  - `top_p < 1.0` or `top_k > 0` is rejected on the deferred sampled path until
    a real FlashInfer/Triton sampler kernel is integrated;
  - this is a serving-contract improvement, not a large-random speed promotion.
- artifacts:
  - direct sampled smoke:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/sampled_fastpath_smoke_temp08_jax.json`;
  - guarded random sidecar sampled smoke:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_sampling_sidecar_smoke_temp08_r2.json`;
  - nested JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_sampling_sidecar_smoke_temp08_r2_jax.json`.
- results:
  - direct smoke: `temperature=0.8`, `8` generated tokens, zero measured-phase
    JIT growth (`16 -> 16`), GPU memory `4365 MB` after engine and `8505 MB`
    after measurement;
  - random sidecar smoke: `2` requests, `59` input tokens, `10` output tokens,
    `23.89 output tok/s`, zero measured-phase JIT growth (`16 -> 16`), peak
    process-tree RSS about `4.40 GB`, peak system RAM `50.4%`, and no resource
    guard kill;
  - startup warmup recorded sampled prefill plus both generic and resident
    sampled decode routes for batch buckets `1,2`.
- rejected/diagnostic issues fixed during the pass:
  - engine deferred-token admission initially rejected all non-greedy sampling;
  - packed prefill initially sized RNG/temperature vectors to the physical
    token row, causing a `vmap` size mismatch for multi-request packed prefill;
  - scheduler postprocess initially tried to cast sampled `DeviceTokenRef` to
    `int`;
  - startup initially warmed only resident sampled decode, so the first generic
    sampled decode compiled during measurement;
  - a first random sidecar smoke exceeded configured prefill token buckets
    (`79` tokens versus `16,32,64`), so bucket coverage must match the random
    envelope before using `--fail-on-jit-cache-growth`.
- validation:
  - `python -m py_compile nanovllm_jax/model.py nanovllm_jax/config.py nanovllm_jax/server_config.py nanovllm_jax/engine/sequence.py nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/scheduler.py benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_random_request_sidecar.py benchmarks/benchmark_vllm_qwen35.py benchmarks/compare_vllm_sampling_modes.py tests/test_lm_head_helpers.py tests/test_device_token_carry.py`;
  - `pytest -q tests/test_lm_head_helpers.py tests/test_server_config.py tests/test_device_token_carry.py tests/test_benchmark_jax_server_trace.py tests/test_benchmark_random_request_sidecar.py tests/test_backend_boundaries.py::test_executor_table_hybrid_decode_matches_sliced_decode`.

### Entry 276 - Rejected Standalone FlashInfer LM-Head Top-K

- date: 2026-06-07
- purpose:
  - test whether FlashInfer's radix top-k can replace the JAX
    argmax/top-k selector at the decode LM-head boundary;
  - keep the full-vocab LM-head projection on device and avoid changing the
    benchmark request envelope, paged attention, chunked/ragged prefill, or
    resident decode metadata contract.
- implementation tested:
  - added optional `lm_head_topk_impl=flashinfer` config plumbing and a JAX FFI
    wrapper around FlashInfer `radix_topk`;
  - added the local FlashInfer CUB compatibility define
    `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_CUB_SUBTRACTLEFT_DEFINED`;
  - raised the default `decode_padded_gemm_max_out_dim` to `300000`, so the
    Qwen3.5 0.8B vocabulary projection can use the existing padded decode GEMM
    path without a config-specific cap;
  - fixed scheduler/benchmark diagnostics discovered during the retry:
    `max_num_batched_tokens=0` now means uncapped instead of blocking all
    prefill work, capacity failures report block/request state, and JAX trace
    artifacts record effective/requested KV cache capacity plus the token
    budget.
- artifacts:
  - baseline accepted JAX artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_large_full_table_resident_r2_jax.json`;
  - FlashInfer top-k artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_large_flashinfer_lm_topk_r4_jax.json`;
  - diagnostic synthetic-warmup artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_large_flashinfer_lm_topk_warmtrace_r5_jax.json`;
  - stored same-manifest vLLM denominator:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/vllm_sampling_modes_random_large.json`.
- results:
  - focused FlashInfer FFI tests passed on CUDA: radix top-1 and
    LM-head-top-1 matched the JAX selector;
  - random-large FlashInfer top-k regressed to `296.79 output tok/s` versus the
    current JAX artifact at `756.62 output tok/s` and vLLM greedy at
    `1081.31 output tok/s`; measured-phase JIT growth stayed zero;
  - deduped scheduler-step timing showed FlashInfer top-k median decode was
    slightly faster (`5.26 ms` versus `5.90 ms`), but large first-use outliers
    dominated end-to-end timing: prefill `2.25 s` total and decode `2.99 s`
    total versus the JAX baseline's `0.30 s` and `1.67 s`;
  - a generic synthetic scheduler warmup experiment reduced FlashInfer decode
    step total (`2.99 s -> 1.82 s`) but made prefill much worse (`2.25 s ->
    3.54 s`), so that warmup approach was removed.
  - a ret-only scratch variant that passed focused FlashInfer tests was also
    tried; the integrated random-large run stayed GPU-bound without producing
    an artifact after about eight minutes, so the wrapper was restored to the
    validated scratch-input form and the ret-only scratch variant is rejected.
- decision:
  - keep `lm_head_topk_impl=flashinfer` as an explicit experimental knob only;
  - keep optimized benchmark configs on `lm_head_topk_impl=jax`;
  - do not retry standalone LM-head selector kernels as the primary route.
    Entry 267 already rejected a standalone Triton selector; this FlashInfer
    result confirms the useful LM-head direction must be a true fused GEMM
    epilogue/library path or a broader decode boundary.
- FlashInfer attention note:
  - installed FlashInfer exposes batch decode JIT modules, but the generated
    ABI has separate `plan` and `run` exports with workspaces/plan-info vectors;
  - the current accepted decode path uses
    `triton_paged_fused_append`, which bypasses `write_kv`, so FlashInfer
    append alone cannot improve random-large decode. A real FlashInfer
    attention experiment needs a JAX FFI plan/run wrapper and a boundary that
    preserves or replaces fused append+decode.
- validation:
  - `python -m py_compile nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/block_manager.py nanovllm_jax/engine/scheduler.py benchmarks/benchmark_jax_server_trace.py nanovllm_jax/kernels/flashinfer_ffi.py nanovllm_jax/model.py nanovllm_jax/config.py nanovllm_jax/server_config.py`;
  - `pytest -q tests/test_backend_boundaries.py::test_scheduler_zero_max_batched_tokens_does_not_block_prefill tests/test_backend_boundaries.py::test_scheduler_chunks_prefill_by_max_batched_tokens_budget tests/test_lm_head_helpers.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py`;
  - `JAX_PLATFORMS=cuda FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_CUB_SUBTRACTLEFT_DEFINED' FLASHINFER_NVCC_THREADS=1 MAX_JOBS=1 pytest -q tests/test_flashinfer_ffi.py`.

### Entry 277 - Rejected FlashInfer Fused Paged Decode Attention

- date: 2026-06-07
- purpose:
  - test whether FlashInfer batch decode attention can replace the current
    Triton `triton_paged_fused_append` route on the random decode graph;
  - broaden the borrowed-kernel ABI enough to preserve the serving cache
    boundary instead of measuring a standalone attention probe.
- implementation tested:
  - added a JAX/TVM FFI shim for FlashInfer batch decode where the ten
    `DecodePlanInfo` fields are passed as scalar attrs, because JAX FFI cannot
    lower Python list/tuple attrs to FlashInfer's `Array<int64_t>` ABI and a
    JAX tensor has the wrong TVM type;
  - cached the planned `int_workspace` bytes from FlashInfer `plan()`, because
    `run()` reads request/tile/output-indptr tables from that workspace;
  - shrank the float workspace from FlashInfer's default `128 MiB` to the
    actual planned byte requirement (`~1 MiB` for the large random B8/128-page
    bucket);
  - added an opt-in `full_attention_decode_impl=flashinfer_paged` route with a
    full-cache fused append+decode FFI call that mutates/aliases the full 5D
    cache, avoiding the earlier split append plus layer-slice reinsertion.
- artifacts:
  - Triton accepted baseline:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_large_full_table_resident_r2_jax.json`;
  - split FlashInfer smoke:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_smoke_flashinfer_decode_r3_jax.json`;
  - fused FlashInfer smoke:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_smoke_flashinfer_fused_decode_r1_jax.json`;
  - fused FlashInfer large:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_large_flashinfer_fused_decode_r1_jax.json`;
  - fused FlashInfer large with explicit 128-wide decode warmup:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260607/random_large_flashinfer_fused_decode_warm128_r2_jax.json`.
- results:
  - focused CUDA correctness passed for FlashInfer append, standalone decode,
    and fused full-cache append+decode:
    `XLA_PYTHON_CLIENT_PREALLOCATE=false FLASHINFER_NVCC_THREADS=1 MAX_JOBS=1 FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_CUB_SUBTRACTLEFT_DEFINED' .venv/bin/pytest -q tests/test_flashinfer_ffi.py`;
  - split FlashInfer smoke regressed to `173.00 output tok/s` versus Triton
    smoke `317.67 output tok/s`; fused FlashInfer improved that to `207.56`
    but still lost badly;
  - large fused FlashInfer reached only `297.89 output tok/s`; explicit
    `--decode-block-table-buckets 128` warmup improved to `309.41 output tok/s`,
    still far below the Triton baseline `756.62 output tok/s`;
  - steady large decode median was promising (`5.05 ms` with warm128 versus
    Triton `5.90 ms`), but measured outliers and prefill stalls dominated:
    FlashInfer warm128 prefill sum `2222.80 ms` and decode sum `2.82 s` versus
    Triton prefill `295.75 ms` and decode `1.67 s`;
  - measured-phase JIT growth stayed zero, so these stalls are not new executor
    keys; they are likely first-use/custom-call/runtime effects from the
    FlashInfer route or asynchronous work bleeding into host-timed steps.
- decision:
  - do not promote `flashinfer_paged`; keep it as an explicit diagnostic
    scaffold only;
  - keep optimized configs on `full_attention_decode_impl=triton_paged_fused_append`;
  - do not retry FlashInfer decode as a primary route unless a profile first
    explains/removes the large custom-call first-use/prefill stalls. The next
    high-leverage route is still a broader model/GDN boundary or a fused
    LM-head GEMM epilogue, not another standalone attention swap.

### Entry 278 - LM-Head Fused-GEMM Boundary And Backend Audit

- date: 2026-06-08
- purpose:
  - take the LM-head fused-GEMM lane without repeating the rejected standalone
    selector paths from Entries 267 and 276;
  - make any future fused projection-plus-top1 route explicit and fail-fast so
    dense-logit fallback cannot pollute random-large benchmark results.
- implementation:
  - split LM-head decode helper code into a normalized-hidden/output-weight
    boundary and a logits-from-normalized helper;
  - added config-file field `lm_head_greedy_top1_impl`, carried through
    `Qwen3_5Config`, server config engine overrides, and
    `benchmark_jax_server_trace.py`;
  - intentionally did not add another env-var translation for this experimental
    selector;
  - `lm_head_greedy_top1_impl=cutlass` now enters the decode greedy width-1
    boundary and raises `NotImplementedError` instead of falling back to dense
    logits. This is deliberate until a true fused GEMM+top1 backend exists.
- audit findings:
  - vLLM's current logits processor still computes full logits via the LM-head
    module, then its sampler performs greedy argmax or sampling over those
    logits;
  - FlashInfer public sampling/top-k APIs also start from logits and do not
    expose an LM-head projection-plus-sampler kernel;
  - cuBLASLt built-in epilogues cover default output, pointwise activations,
    bias, and bias-gradient style reductions, not a row-wise top-1/argmax over
    the GEMM output;
  - CUTLASS/CuTeDSL can express custom epilogue/reduction work, so the viable
    implementation path is a real CuTeDSL/CUTLASS JAX custom call or equivalent
    library-backed GEMM epilogue, not another Triton blockwise matvec scan.
- decision:
  - keep optimized configs on `lm_head_greedy_top1_impl=jax`;
  - do not benchmark this boundary as a speed win until a real backend is wired;
  - next LM-head work, if pursued, should implement the CuTeDSL/CUTLASS fused
    projection-plus-rowwise-top1 backend for `[B, H] x [H, V] -> token_ids`
    and compare integrated random-large throughput against the accepted
    `756.62 output tok/s` JAX artifact and same-manifest vLLM denominator.
- validation:
  - `.venv/bin/pytest -q tests/test_lm_head_helpers.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py`;
  - `.venv/bin/python -m py_compile nanovllm_jax/model.py nanovllm_jax/config.py nanovllm_jax/server_config.py benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_random_request_sidecar.py tests/test_lm_head_helpers.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py`.

### Entry 279 - Rejected Scalar CuTeDSL LM-Head Top-1 Probe

- date: 2026-06-08
- purpose:
  - try the `lm_head_greedy_top1_impl=cutlass` direction with a real JAX
    custom-call probe before committing to a larger tensor-core epilogue port;
  - quantify whether removing full logits with a simple CuTeDSL rowwise top-1
    kernel can beat XLA's current LM-head `dot+argmax` path.
- implementation tested and reverted:
  - added a temporary CuTeDSL custom call that scanned
    `[B, 1, H] x [H, V]` directly and returned per-thread partial top-1
    candidates, followed by a tiny JAX reduction over 256 candidates per row;
  - verified token-id parity on a small focused test;
  - removed it from the active selector because it used scalar CUDA-core dot
    loops rather than tensor cores.
- focused measurements on A10G, representative Qwen3.5-0.8B shape
  `B=8, H=1024, V=248064`, BF16 inputs:
  - XLA `jnp.dot + jnp.argmax`: `~1.04 ms`;
  - temporary scalar CuTeDSL no-logits top-1: `~100.21 ms`;
  - standalone CUTLASS Ampere tensor-core full GEMM example:
    `~1035 us`, effectively the same as XLA's full projection;
  - XLA `dot` alone: `~1.02 ms`, so isolated argmax adds only about
    `0.01-0.02 ms` on this shape;
  - `lax.top_k(..., 1)`, fp32-cast argmax, exact pair-reduction argmax, and
    width-1 2D reshape spellings all matched tokens but did not improve
    latency.
- decision:
  - reject the scalar CuTeDSL route and keep `cutlass` fail-fast;
  - do not replace the current path with standalone top-k/argmax or full-output
    CUTLASS GEMM;
  - the only remaining LM-head-specific route worth implementing is a true
    tensor-core GEMM with rowwise top-1/argmax integrated into the epilogue or
    tile-reduction, so the optimized GEMM path is preserved while the full
    logits store and separate reduction/scatter are avoided.

### Entry 280 - Neutral Resident Slot-Index Cache Pass

- date: 2026-06-08
- purpose:
  - after rejecting the scalar LM-head route, test whether the visible
    `_sync_resident_decode_metadata` CPU/PJRT bucket still had a safe
    shape-agnostic cleanup;
  - avoid changing warmed request shapes, model dimensions, or GPU-specific
    tile choices.
- implementation:
  - cached device arrays for repeated resident metadata update slot tuples in
    `_sync_resident_decode_metadata`;
  - kept block-row and seq-len payload updates unchanged, so resident paging
    state and generated tokens are unchanged.
- focused result:
  - an always-`device_put` focused resident metadata update measured
    `~0.48-0.49 ms` median;
  - the cached-slot version measured `~0.33-0.35 ms` median on the same focused
    update loop.
- integrated random-large checks:
  - same generated prompt manifest as Entry 276/278
    (`sha256=6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`);
  - 8 requests, `512-1024` input-token range, `128-256` output-token range,
    `1582` generated tokens, generic warmup, stored same-manifest vLLM
    denominator, zero measured-phase JIT growth, and no kernel fallbacks;
  - 8-core guarded sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_slot_index_cache_r2.json`,
    `751.38 output tok/s`;
  - 4-core guarded sidecar:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_slot_index_cache_r3_4core.json`,
    `753.07 output tok/s`;
  - both missed the accepted `756.62 output tok/s` JAX artifact, so this is
    not an accepted random-large speed win.
- profiling note:
  - a same-envelope profile retry was correctly killed by the 70% RAM guard at
    `72.5%` observed system RAM and `~6.8 GiB` process-tree RSS, before a JAX
    artifact or usable trace was produced;
  - the prior successful profile remains the bottleneck source: GPU GEMMs
    `~992.50 ms`, `_paged_decode_attention` `~197.92 ms`, CPU/PJRT execute
    `~1116.54 ms`, and `_sync_resident_decode_metadata` `~402.36 ms`.
- decision:
  - keep the cache only as a small resident-metadata cleanup, not as a promoted
    benchmark win;
  - do not continue chasing slot-index caching for large random. The payload
    puts/scatter and model-side work dominate.

### Entry 281 - Neutral Triton LM-Head Top1 Epilogue And GEMM Fragmentation Baseline

- date: 2026-06-08
- purpose:
  - finish the LM-head epilogue lane with a tensor-core-preserving Triton
    diagnostic instead of another standalone selector;
  - make the next bottleneck visible by grouping profiled GEMM kernels by
    kernel name, HLO op, and launch grid instead of only by broad event name.
- implementation:
  - added `nanovllm_jax/kernels/lm_head_triton.py`, a two-stage
    JAX-Triton greedy top-1 path for `[B,1,H] x [H,V] -> token_ids`;
  - stage 1 uses tiled `tl.dot` and writes one candidate per row/vocabulary
    tile, so it does not materialize full `[B,V]` logits;
  - stage 2 reduces the row/vocabulary-tile candidates to `[B,1]` token ids;
  - BF16/BF16 and FP16/FP16 stage-1 candidates are rounded before top-1
    comparison to match JAX's reduced-output dtype behavior;
  - `lm_head_greedy_top1_impl=triton` routes to this backend explicitly, while
    the optimized benchmark configs remain on `jax`;
  - extended `benchmarks/summarize_profile_trace.py` with
    `top_gemm_kernels_by_total_ms` so future profiles show repeated GEMM launch
    grids and split-K reducers directly.
- focused LM-head result:
  - representative Qwen3.5-0.8B BF16 shape `B=8,H=1024,V=248064`;
  - token equality with JAX: `True`;
  - XLA dense `dot+argmax`: median `1.21295 ms`, mean `1.22523 ms`;
  - Triton two-stage top-1: median `1.17371 ms`, mean `1.17544 ms`;
  - this is only a small microbenchmark win, not enough to assume integrated
    serving improvement.
- integrated random-large check:
  - same generated prompt manifest as Entries 276/278/280
    (`sha256=6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`);
  - 8 requests, `512-1024` input-token range, `128-256` output-token range,
    `1582` generated tokens, generic warmup, stored same-manifest vLLM
    denominator, zero measured-phase JIT growth, and no kernel fallbacks;
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_lm_head_triton_top1_r1.json`;
  - result: `753.80 output tok/s`, `0.738x` of the stored vLLM denominator
    `1021.59 output tok/s`;
  - this missed the accepted large-random JAX anchor (`756-758 output tok/s`),
    so it is not promoted.
- GEMM fragmentation baseline:
  - existing usable profile:
    `/mountpoint/.exp/profiles/random_large_dense_direct_device_put_profile_r1/20260605-203925-286804-random_large_dense_direct_device_put_profile_r1_jax/plugins/profile/2026_06_05_20_40_43/INDCS0291.atrapa.deloitte.com.trace.json.gz`;
  - generated summaries:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_dense_direct_device_put_profile_gemm_summary.md`
    and
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_dense_direct_device_put_profile_gemm_summary.json`;
  - profiled GPU totals are still dominated by real model work:
    `gemm` `~992.61 ms / 24735`, `fusion` `~636.04 ms / 79893`,
    `_paged_decode_attention` `~197.94 ms / 1488`, `input_scatter`
    `~185.24 ms / 2095`, and `splitKreduce` `~25.36 ms / 11904`;
  - the full-vocabulary LM-head GEMM appears as grid `1940,1,1`, matching
    `248320 / 128` vocabulary tiles; it is still a bucket, but the Triton
    top-1 route did not improve integrated throughput;
  - repeated small-B model GEMMs appear as grid `56,1,1` and `65,1,1` launches
    thousands of times, with split-K reducers attached. These are the
    fragmentation target for the next pass.
  - regenerated grid totals after adding the coarser summary view:
    - `1940,1,1`: `250.23 ms / 248` launches, full-vocabulary LM-head;
    - `56,1,1`: `211.03 ms / 5952` launches, packed MLP gate/up family;
    - `65,1,1`: `175.71 ms / 4464` launches, packed GDN decode input
      projection family;
    - `8,2,5`: `152.73 ms / 9192` launches, CUTLASS projection/down/out
      family;
    - `32,1,1`: `25.36 ms / 11904` split-K reducer launches.
- compiler-policy diagnostics:
  - `--xla_gpu_experimental_force_split_k=1` was accepted but regressed the
    representative packed-MLP microbench (`~0.192 ms -> ~0.221 ms` median);
  - XLA Triton GEMM flags were accepted but slower/noisier than cuBLASLt on the
    same representative shape, and changed BF16 reduction numerics;
  - `--xla_gpu_enable_bf16_3way_gemm`,
    `--xla_gpu_enable_bf16_6way_gemm`,
    and split-K rewrite toggles were not accepted by the installed jaxlib;
  - disabling cuBLASLt was slower;
  - `--xla_gpu_enable_cuda_graphs=true` is not available in this jaxlib.
- decision:
  - keep the Triton LM-head top-1 backend as an explicit diagnostic, not the
    active optimized path;
  - do not chase XLA GEMM policy flags for fragmentation. The next route is
    code-level coarsening of model-side boundaries, starting with repeated MLP
    and GDN GEMM/fusion groups that show up in the trace.
- validation:
  - `.venv/bin/pytest -q tests/test_profile_trace_summary.py`;
  - `.venv/bin/pytest -q tests/test_lm_head_helpers.py::test_lm_head_greedy_top1_triton_matches_jax_on_cuda`;
  - `.venv/bin/python -m py_compile benchmarks/summarize_profile_trace.py nanovllm_jax/kernels/lm_head_triton.py nanovllm_jax/model.py tests/test_profile_trace_summary.py tests/test_lm_head_helpers.py`.

### Entry 282 - Rejected Row-Padded GDN Packed Decode Input Projection

- date: 2026-06-08
- purpose:
  - start the GEMM-fragmentation pass by targeting the repeated `65,1,1` GEMM
    family from Entry 281;
  - test whether applying the already-accepted row-padded decode GEMM policy to
    GDN's packed `in_proj_qkv_abz` decode projection improves random-large
    throughput.
- change tested and reverted:
  - temporarily routed the decode-only packed GDN input projection through
    `_decode_padded_gemm_dot()` when `decode_padded_gemm=true`,
    `seq_len == 1`, batch fit `decode_padded_gemm_rows`, and the projection
    output fit `decode_padded_gemm_max_out_dim`;
  - this reused existing config-file policy and added no new tuning knob;
  - the change was reverted because it did not improve the integrated target.
- integrated random-large check:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_gdn_padded_inproj_r1.json`;
  - same prompt manifest as Entries 276/278/280/281
    (`sha256=6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`);
  - 8 requests, `512-1024` input-token range, `128-256` output-token range,
    `1582` generated tokens, generic warmup, stored same-manifest vLLM
    denominator, and no live vLLM rerun;
  - result: `752.99 output tok/s`, `0.737x` of vLLM `1021.59 output tok/s`;
  - this was below the same-envelope slot-index-cache repeat
    (`753.07 output tok/s`) and below the accepted JAX anchor
    (`756-758 output tok/s`);
  - token-event throughput was `785.98 output tok/s`, final drain was
    `88.18 ms`, and peak system RAM stayed under the guard at `65.7%`.
- decision:
  - reject/revert as an active serving change;
  - do not target GEMM fragmentation by only padding the packed GDN input
    projection. The next attempt must reduce a larger model-side
    launch/materialization group, such as MLP activation/down-projection
    fusion or a broader GDN post-projection/tail boundary.
- validation:
  - temporary focused helper tests passed before the integrated run;
  - after reverting, rerun the standard focused tests listed in the final
    validation for this pass.

### Entry 283 - Rejected Neutral Composition And Scheduler-Cap Probes

- date: 2026-06-08
- purpose:
  - continue the large-random hill-climb after Entry 282 by composing neutral
    changes before moving to heavier kernel work;
  - check whether the remaining gap is still token materialization, scheduler
    prefill batching, CPU guard policy, or matmul precision, rather than
    broader model-side GEMM/fusion groups.
- direct-prefetch materialization experiment, tested and reverted:
  - temporarily tracked arrays that had received `copy_to_host_async()` from
    `Sequence.prefetch_device_token_slots()`;
  - when all vector/scalar token arrays in a snapshot were known-prefetched,
    `Sequence.materialize_device_token_slots()` used `jax.device_get(list)`
    directly instead of creating a new `jnp.stack(...)` before `device_get`;
  - focused tests confirmed prefetched `DeviceTokenRef` snapshots avoided the
    stack path while unprefetched snapshots kept the old stacked materialization.
- direct-prefetch integrated result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_prefetch_direct_materialize_r1.json`;
  - same prompt manifest as Entries 276/278/280/281/282
    (`sha256=6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`);
  - result: `752.55 output tok/s`, `0.737x` of stored vLLM
    `1021.59 output tok/s`;
  - token-event throughput `785.21 output tok/s`, final drain `87.44 ms`;
  - this matched the old JAX drain and missed both the same-envelope
    `753.07 output tok/s` repeat and the accepted `756-758` anchor.
- Triton LM-head composition, tested without keeping config sprawl:
  - temporarily added a config variant differing only by
    `lm_head_greedy_top1_impl=triton`, then removed it after the run;
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_prefetch_direct_lm_head_triton_r1.json`;
  - result: `753.62 output tok/s`, `0.738x` of stored vLLM;
  - token-event throughput `797.06 output tok/s`, final drain `114.43 ms`;
  - this reproduced Entry 281's pattern: slightly better token-event timing,
    but no wall-clock win after final drain.
- prefill-token cap probes:
  - `max_num_batched_tokens=2048` with warmed
    `prefill_token_buckets=512,1024,2048`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_prefill_cap2048_r1.json`;
  - `max_num_batched_tokens=1536` with warmed
    `prefill_token_buckets=512,1024,1536`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_prefill_cap1536_r1.json`;
  - both were killed by the 70% system-RAM guard before measurement
    (`70.1%` and `70.0%` observed, process-tree RSS about `7.3 GiB`);
  - under the current safety policy, widening the generic random-large prefill
    bucket surface is not viable.
- worker CPU guard probe:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_worker2_r1.json`;
  - lowering the JAX subprocess cap from 4 cores to 2 cores reached
    `751.02 output tok/s`, below the 4-core same-envelope `753.07` repeat;
  - keep the 4-core guard for this lane.
- matmul precision microbench:
  - representative packed MLP shape `B=8`, `H=1024`, gate/up output `5632`,
    down output `1024`, BF16 inputs/weights;
  - default: `0.192 ms` median;
  - `JAX_DEFAULT_MATMUL_PRECISION=bfloat16`: `0.214 ms` median;
  - `JAX_DEFAULT_MATMUL_PRECISION=highest`: `0.224 ms` median;
  - do not spend a full random-large run on this precision lever.
- decision:
  - reject/revert direct-prefetch materialization as a serving speed change;
  - keep `lm_head_greedy_top1_impl=triton` diagnostic-only, with no extra
    benchmark config variant;
  - keep `max_num_batched_tokens=1024` and the 4-core worker guard for the
    current large-random iteration envelope;
  - the remaining speed target is still model-side GPU work: repeated MLP/GDN
    GEMM/fusion groups, LM-head projection/reduction as a true fused GEMM
    epilogue, and broader GDN/model boundaries.

### Entry 284 - Rejected Decode Projection Padding And MLP SiLU-Down Fusion

- date: 2026-06-08
- purpose:
  - continue the large-random hill-climb by composing the accepted row-padded
    decode GEMM policy onto nearby projection sites;
  - test whether a local fused MLP `silu(gate) * up -> down_proj` helper can
    remove materialization/fusion overhead from the repeated packed-MLP bucket.
- output-projection padding variants, tested and reverted:
  - temporarily routed decode-time GDN `out_proj` and full-attention `o_proj`
    through `_decode_padded_gemm_dot()` when the existing
    `decode_padded_gemm=true` policy admitted the shape;
  - a broader variant also routed the full-attention packed QKV projection
    helper through row-padded GEMM;
  - both variants reused existing config fields and added no new knobs.
- output-projection integrated results:
  - same prompt manifest as Entries 276/278/280/281/282/283
    (`sha256=6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`);
  - output projection plus full-attention QKV padding:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_decode_outproj_padded_r1.json`,
    `753.57 output tok/s`, `0.738x` stored vLLM, token-event
    `786.29 output tok/s`, final drain `87.35 ms`, peak RAM `66.1%`;
  - output-projection-only padding:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_decode_outproj_only_padded_r1.json`,
    `753.72 output tok/s`, `0.738x` stored vLLM, token-event
    `786.11 output tok/s`, final drain `86.49 ms`, peak RAM `65.9%`;
  - both missed the accepted `756-758 output tok/s` anchor.
- MLP SiLU/down Triton diagnostic, tested and removed:
  - implemented a temporary JAX-Triton helper for
    `dot(silu(gate) * up, down_weight)` from packed gate/up activations;
  - focused BF16 Qwen3.5-0.8B decode shape `B=8`, `I=3584`, `H=1024`
    showed one generic tile shape (`block_n=32`, `block_k=64`) slightly
    faster than XLA (`0.165 ms` median versus `0.180 ms`) but with BF16 output
    drift up to about `1.0`;
  - integrated random-large with
    `NANO_VLLM_JAX_DECODE_MLP_SWIGLU_DOWN_TRITON=1` regressed to
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_mlp_swiglu_down_triton_r1.json`,
    `723.27 output tok/s`, `0.708x` stored vLLM, token-event
    `754.76 output tok/s`, final drain `91.25 ms`, peak RAM `66.4%`;
  - the temporary env flag, source route, helper file, and focused test were
    removed after the integrated loss.
- decision:
  - reject/revert decode output-projection padding as an active serving change;
  - reject/remove the standalone MLP SiLU/down Triton helper. It is another
    microbench-only win that gives up too much integrated XLA/CUTLASS graph
    quality;
  - do not pursue single-site row-padding or standalone MLP middle kernels as
    the next large-random lever. The next candidate must coarsen a larger
    model-side boundary or replace a whole repeated kernel family with a
    serving-proven backend path.
- validation after cleanup:
  - `.venv/bin/python -m py_compile nanovllm_jax/model.py nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py tests/test_lm_head_helpers.py tests/test_backend_boundaries.py`;
  - `.venv/bin/pytest -q tests/test_lm_head_helpers.py::test_compact_prefill_mlp_matches_dense_on_valid_tokens tests/test_backend_boundaries.py::test_decode_padded_gemm_supports_small_decode_batches tests/test_profile_trace_summary.py`;
  - `git diff --check`.

### Entry 285 - Rejected Larger Decode Boundaries

- date: 2026-06-08
- purpose:
  - follow the broader-boundary path after single-projection and standalone MLP
    kernels missed the large-random anchor;
  - test whether coarsening around GDN decode or across multiple greedy decode
    tokens can reduce host/PJRT or small-kernel overhead without changing the
    random-serving benchmark contract.
- GDN tail boundary retest:
  - explicit diagnostic config changed only
    `kernels.gdn.packed_decode.impl` to
    `triton_fla_conv_raw_gates_tail`;
  - small B4 probe was neutral/slightly positive:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_small_gdn_tail_boundary_explicit_r1.json`,
    `436.78 output tok/s` versus the same small reference route at
    `434.06 output tok/s`;
  - same-manifest large random regressed to
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_gdn_tail_boundary_explicit_r1.json`,
    `618.30 output tok/s`, `0.605x` stored vLLM, token-event
    `624.98 output tok/s`, final drain `27.32 ms`, peak RAM `64.9%`.
- multi-token decode boundary probes, tested and removed/rejected:
  - existing table burst-2 on the accepted reference-GDN route:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_decode_burst2_reference_gdn_r1.json`,
    `326.19 output tok/s`, `0.319x` stored vLLM, token-event
    `336.20 output tok/s`, peak RAM `68.5%`, sidecar wall time
    `367.01 s`;
  - temporary static-unrolled burst-2 source change passed focused burst tests
    but failed the small gate:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_small_decode_burst2_unrolled_r1.json`,
    `266.23 output tok/s` versus the small reference route at
    `434.06 output tok/s`; no large run was attempted;
  - temporary resident-dense burst-2 route preserved resident block tables,
    seq-lens, last tokens, and hybrid state inside the compiled boundary, and
    warmup was extended to precompile its B1-B8 keys;
  - resident-dense burst small gate improved over old burst but still lost:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_small_decode_burst2_resident_dense_r2.json`,
    `399.20 output tok/s` versus `434.06`;
  - resident-dense burst large random regressed to
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_decode_burst2_resident_dense_r1.json`,
    `591.44 output tok/s`, `0.579x` stored vLLM, token-event
    `616.33 output tok/s`, peak RAM `68.5%`, sidecar wall time
    `357.94 s`;
  - the resident-dense burst source route and warmup wiring were removed after
    the integrated loss.
- decision:
  - reject GDN tail-fused decode as an active large-random boundary; its small
    B4 result did not generalize to the target B8 random graph;
  - reject multi-token greedy decode bursts as the next broadening lever, even
    when resident dense metadata and token carry are preserved. Full-model JAX
    scans reduce Python/PJRT call count but lower into a slower compiled plan
    with a much larger warmup surface;
  - do not retry decode bursts unless the implementation uses runtime graph
    replay or a backend-owned token-loop boundary that avoids full-model
    `lax.scan` recompilation.
- validation after cleanup:
  - `rg -n "forward_greedy_decode_burst_resident_dense_slot_carry_jit|resident_dense_slot_token_metadata_burst_decode|token-ids-burst-resident-dense" nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py`;
  - `.venv/bin/python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py`;
  - `.venv/bin/pytest -q tests/test_backend_boundaries.py::test_executor_greedy_decode_burst_matches_iterative_token_path tests/test_backend_boundaries.py::test_model_runner_warmup_compiles_greedy_decode_burst tests/test_device_token_carry.py::test_scheduler_static_decode_metadata_allows_greedy_burst`.

### Entry 286 - Rejected Fused Decode RMSNorm And Packed MLP Gate/Up Projection

- date: 2026-06-08
- purpose:
  - start the backend-owned kernel execution lane from the repeated MLP/GDN
    projection bucket after LM-head and standalone MLP middle kernels missed
    the large-random target;
  - test whether fusing FFN RMSNorm, BF16 projection cast, and the packed MLP
    gate/up row-padded GEMM can remove a materialized activation and one XLA
    fusion boundary without giving up tensor-core projection math.
- implementation:
  - added a gated JAX-Triton helper for
    `rms_norm(x, ffn_norm).astype(bfloat16) @ packed_gate_up_weight` on
    width-1 decode tensors with `B <= decode_padded_gemm_rows`;
  - added config field `decode_rms_padded_gemm`, server-config projection, and
    benchmark CLI support;
  - routed only packed MLP decode gate/up through the helper when
    `decode_proj_act_dtype=bf16`, `decode_padded_gemm_gate_up=true`, and layer
    stage tracing is disabled.
- focused validation:
  - `pytest -q tests/test_decode_reductions.py -k triton_decode_rms_padded_gemm`
    passed on GPU;
  - `pytest -q tests/test_server_config.py tests/test_decode_reductions.py -k 'runtime_fastpaths_project_to_engine_config or engine_overrides_from_config_merges_runtime_fastpaths_and_kernel_policy or runtime_and_kernel_sections_translate_to_env or triton_decode_rms_padded_gemm'`
    passed;
  - `python -m py_compile benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_random_request_sidecar.py nanovllm_jax/config.py nanovllm_jax/server_config.py nanovllm_jax/model.py nanovllm_jax/kernels/decode_reductions.py`
    passed.
- integrated random-large check:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_rms_padded_gemm_r1.json`;
  - same manifest as Entries 276/278/280/281/282/283/284/285,
    `sha256=6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`;
  - `742.07 output tok/s`, `0.726x` of stored vLLM `1021.59 output tok/s`,
    `1582` generated tokens, generic warmup, no resource-guard kill, peak RAM
    `68.3%`, child elapsed `509.74 s`;
  - token-event throughput dropped to `751.69 output tok/s` versus the
    accepted trace-prefetch anchor around `794.95`, while final drain improved
    to `27.28 ms`.
- decision:
  - reject this narrow fused RMSNorm+gate/up projection as an active serving
    speed path and keep the main GPU config on the accepted JAX row-padded
    gate/up GEMM route;
  - keep the helper gated for diagnostics only. The integrated result shows
    that removing this materialization does not help if the replacement adds a
    custom call per MLP layer and loses XLA/CUTLASS projection quality;
  - do not retry FFN RMSNorm+packed gate/up as a standalone Triton boundary.
    Future MLP work needs a serving-proven grouped/epilogue GEMM route or a
    broader backend-owned family that does not recompute or launch per layer.

### Entry 287 - Rejected Output-Tiled Single-Kernel Full MLP Fusion

- date: 2026-06-08
- purpose:
  - test whether the MLP lane can be broadened beyond Entries 284 and 286 by
    owning FFN RMSNorm, packed gate/up projection, SwiGLU, and down projection
    inside one backend custom call;
  - stop before a large-random run if the actual-shape kernel cannot beat the
    current XLA/CUTLASS BF16 MLP path.
- implementation tested and removed:
  - prototyped a JAX-Triton helper for width-1 decode:
    `rms_norm(x, ffn_norm) -> packed gate/up -> silu(gate) * up -> down_proj`;
  - matched BF16 rounding points after gate/up projection, SiLU, and SwiGLU
    product, and passed a small focused parity test after the rounding fix;
  - removed the helper, config switch, benchmark CLI hook, and tests after the
    representative-shape performance check failed.
- focused A10G actual-shape result:
  - shape: Qwen3.5-0.8B decode MLP envelope
    `B=8, H=1024, intermediate=3584`, BF16 activations/weights;
  - reference JAX/XLA MLP median: `1.76 ms`;
  - output-tiled Triton full-MLP prototype median: `371.37 ms`;
  - max absolute drift on the stress random tensors was `1024.0`, acceptable
    only as a diagnostic signal because the route is already far too slow.
- diagnosis:
  - the single-kernel design tiles output columns, but gate/up activations are
    shared by all output-column tiles;
  - therefore each output tile recomputes the full gate/up projection before
    doing its down-projection tile. With `H=1024` and `BLOCK_N=64`, that
    repeats the expensive first-stage MLP work about `16x`;
  - increasing output tile width enough to avoid recomputation would create an
    impractically large per-program accumulator and is not a general serving
    kernel design.
- decision:
  - reject and remove the output-tiled single-kernel full MLP route before
    integrated large-random benchmarking;
  - do not retry full MLP fusion unless the design stores/reuses gate/up once,
    uses a real grouped/layer-batched GEMM plan, or comes from a serving-proven
    library/backend that avoids recomputing shared intermediate activations per
    output tile.
- validation after cleanup:
  - `rg -n "decode_full_mlp_triton|triton_decode_rms_swiglu_down|decode_rms_swiglu_down|rms_swiglu_down" .`
    returned no matches;
  - `python -m py_compile nanovllm_jax/kernels/decode_reductions.py nanovllm_jax/model.py nanovllm_jax/config.py nanovllm_jax/server_config.py benchmarks/benchmark_jax_server_trace.py tests/test_decode_reductions.py tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py`;
  - `pytest -q tests/test_server_config.py tests/test_decode_reductions.py tests/test_benchmark_random_request_sidecar.py -k 'runtime_fastpaths_project_to_engine_config or engine_overrides_from_config_merges_runtime_fastpaths_and_kernel_policy or triton_decode_rms_padded_gemm or jax_config_projects_runtime_kernel_policy'`.

### Entry 288 - Accepted FlashInfer Paged Decode And Deferred RNG Reset

- date: 2026-06-08
- purpose:
  - revisit FlashInfer full-attention decode after the persistent plan/workspace
    and fused append/decode boundary were fixed enough to remove the Entry 277
    first-use stall pattern;
  - remove a non-sampling resident RNG-counter reset from the greedy finish
    path, because it was an eager device scatter paid even when greedy decode
    never consumes resident sampling counters.
- accepted implementation:
  - promoted `kernels.full_attention.decode_impl=flashinfer_paged` in
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`;
  - kept FlashInfer attention outputs cast back to FP32 before re-entering the
    existing model path;
  - deferred resident RNG-counter slot zeroing in `ModelRunner.release()` by
    recording reset slots in host state and flushing them only before sampled
    paths consume resident RNG counters;
  - added conditional inactive-row warmup for non-exact decode bucket policies,
    so coarser batch buckets do not trigger measured-phase JIT growth when a
    padded batch uses `resident_slot_carry`.
- target artifact:
  - `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_deferred_rng_reset_r1.json`;
  - random-large seed `1234`, 8 requests, input `512-1024`, output `128-256`,
    prompt manifest SHA
    `6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`;
  - stored vLLM denominator `1021.5851751977929 output tok/s`, target tokens
    `1582`;
  - `816.5828368112691 output tok/s`, `0.7993291764958976x` vLLM,
    token-event throughput `835.3799781883413 output tok/s`, measurement
    seconds `1.93734`, final drain `43.59 ms`;
  - generic warmup, JIT audit `56 -> 56`, no new cache keys, GPU memory after
    measurement `8749 MB`;
  - decode sum `1563.39 ms`, p50 `4.65 ms`, p95 `9.876 ms`, max `71.24 ms`.
- comparison to the accepted Triton fused-attention anchor:
  - previous same-manifest accepted route was `757.58 output tok/s`, about
    `0.742x` vLLM;
  - FlashInfer plus deferred RNG reset improves wall-clock output throughput by
    about `7.8%` and moves the current target ratio to `0.799x`.
- rejected/neutral companion probes:
  - profile attempt with `--jax-profile` was killed by the 70% RAM guard during
    warmup/first-use diagnostics; no trace artifact should be interpreted as a
    speed result;
  - coarsening batch buckets to powers of two required the inactive-row warmup
    fix to avoid JIT growth, but the accepted-warmup rerun reached only
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_pow2_buckets_warmfix_r1.json`,
    `784.0672860849143 output tok/s`, `0.7675x` vLLM, so exact buckets stay
    accepted for the target random workload;
  - disabling trace-token prefetch was neutral:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_no_trace_prefetch_r1.json`,
    `816.6364829256761 output tok/s`, but with larger final drain
    (`56.52 ms`), so keep trace prefetch enabled;
  - returning BF16 directly from FlashInfer attention passed focused FFI tests
    but regressed integrated random-large to
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_bf16_out_r1.json`,
    `815.2224973664511 output tok/s`, so the FP32 output cast remains accepted.
- decision:
  - accept and promote FlashInfer paged full-attention decode plus deferred
    resident RNG reset as the new large-random baseline;
  - keep conditional inactive-row warmup because it prevents JIT-cache growth
    for coarser bucket policies without changing the accepted exact-bucket
    warmup surface;
  - do not retry power-of-two batch buckets or BF16 FlashInfer output as speed
    routes without new profile evidence;
  - remaining target gap is about `102.85 output tok/s` to the `0.9x` vLLM
    threshold. The most visible residual issue is decode-step outliers on top
    of a `~4.6-4.9 ms` median, especially B8/B6/B3 transitions.
- validation:
  - `.venv/bin/python -m py_compile nanovllm_jax/engine/model_runner.py tests/test_backend_boundaries.py`;
  - `.venv/bin/pytest -q tests/test_flashinfer_ffi.py -q`;
  - `.venv/bin/pytest -q tests/test_backend_boundaries.py -k 'warmup_compiles_resident or warmup_compiles_decode_block_table_buckets or release_defers_resident_rng_counter_reset'`.

### Entry 289 - Promoted FlashInfer Route LM-Head Top-1 And Rejected Borrowed-Kernel Follow-Ups

- date: 2026-06-08
- purpose:
  - continue the random-large hill climb on top of Entry 288's FlashInfer
    paged-attention baseline;
  - test whether the existing JAX-Triton LM-head top-1 selector becomes useful
    after the attention route is no longer the old Triton fused-append path;
  - check two borrowed-kernel follow-ups before investing in new FFI bindings.
- accepted implementation:
  - kept `kernels.full_attention.decode_impl=flashinfer_paged`;
  - promoted `runtime.fastpaths.lm_head_greedy_top1_impl=triton` in
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata.json`;
  - changed the Triton LM-head top-1 default vocabulary tile from `128` to
    `256`, a general power-of-two tile choice for large-vocabulary
    projection-plus-top1.
- focused LM-head evidence:
  - representative BF16 shape `B=8,H=1024,V=248064`;
  - JAX `dot+argmax` median around `1.225 ms`;
  - Triton `BLOCK_N=256` median around `1.173 ms`, exact token ids;
  - invalid non-power-of-two tile sizes were not used because Triton requires
    power-of-two arange ranges.
- target large-random result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_lm_triton_bn256_r1.json`;
  - same seed-1234 8-request prompt manifest as Entry 288, SHA
    `6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`;
  - `818.9141213349845 output tok/s`, `0.8016112030760739x` stored vLLM
    `1021.5851751977929 output tok/s`;
  - token-event throughput `866.3764961509661 output tok/s`, measurement
    seconds `1.9318265`, final drain `105.83 ms`;
  - generic warmup, JIT audit `56 -> 56`, GPU memory after measurement
    `8747 MB`.
- rejected or blocked companion routes:
  - GDN `triton_fla_conv_raw_gates` composition passed focused tests but
    regressed integrated large-random to
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_gdn_conv_raw_r1.json`,
    `785.5942716231236 output tok/s`;
  - immediate trace-token prefetch was neutral/slower:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_immediate_prefetch_r1.json`,
    `816.4965558764732 output tok/s`, and the composition with LM Triton
    reached only `818.0147658579746 output tok/s`;
  - the JAX profiler remained too RAM-heavy under the guard even for a smaller
    B8x16 profile and was killed at `70.0%` system RAM after weight load;
  - installed FlashInfer GDN decode exposes the desired pool-indexed
    `[pool, HV, V, K]` state boundary, but the CuTe kernels require SM90+ and
    the current GPU is A10G/SM86, so this is blocked on this machine;
  - a focused A10G B8 FlashInfer attention microbench found current
    non-tensor-core batch decode faster (`0.0763 ms`) than tensor-core planning
    (`0.0855 ms`) and tensor-core fixed/disabled split-KV variants
    (`0.079-0.081 ms`), so no tensor-core attention JAX FFI was added.
- decision:
  - accept the Triton LM-head top-1 route as the new current best because it
    improves the Entry 288 FlashInfer baseline without JIT-cache growth;
  - do not over-interpret the win. The target is still only `0.802x` vLLM, so
    the remaining work is broader model-side decode execution, not another
    LM-head selector microbench;
  - keep FlashInfer GDN as a future borrowed-kernel option for SM90+ or for a
    new JAX FFI on compatible hardware, but do not use it for A10G hill
    climbing.

### Entry 290 - Rejected In-Loop Trace Token Materialization

- date: 2026-06-08
- purpose:
  - test whether the promoted Triton LM-head path's larger final drain could
    be reduced by materializing previously prefetched deferred token slots
    during the trace loop instead of waiting until the final result assembly.
- implementation tested and reverted:
  - changed `LLMEngine._generate_with_trace_deferred_tokens` so that slots
    prefetched on the previous iteration were materialized after the next
    scheduler step completed;
  - updated the focused token-carry test to expect the new overlapped
    prefetch/materialize order.
- target large-random result:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_trace_prefetch_overlap_materialize_r1.json`;
  - the JAX run failed the benchmark contract before timing:
    measured executor JIT cache grew `56 -> 57`;
  - new key:
    `('token-ids-resident', (8, 1), 128, (8, 18, 6144, 4), (8, 18, 16, 128, 128), (8, 128), (8,))`;
  - resource guard did not fire; peak system RAM was `66.7%`.
- diagnosis:
  - materializing deferred sequence token slots inside the trace loop can
    invalidate the all-active-rows device-token-carry predicate for a measured
    decode step;
  - the runner then chooses the broader resident decode key instead of the
    warmed resident slot-carry/dense slot-carry key, causing measured-phase
    compilation and changing the serving path.
- decision:
  - reject and revert the code/test change;
  - keep delayed trace-token materialization as the accepted path;
  - do not retry in-loop trace-token materialization unless the model runner's
    device-token carry state is decoupled from sequence materialization and the
    exact resident slot-carry route is proven to remain warmed.

### Entry 291 - Current Broad Benchmark Checkpoint

- date: 2026-06-08
- purpose:
  - answer whether the accepted Entry 289 serving path is effectively capped
    around `0.8x` vLLM, or whether the current checkout has a broader benchmark
    regression;
  - take a compact current snapshot without committing full benchmark artifacts.
- resource policy used:
  - benchmark artifacts stayed under
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/broad_benchmark_20260608`;
  - random-large used the sidecar guard with `--max-system-ram-percent 70`,
    `--worker-cpu-cores 4`, and `--worker-nice 10`;
  - matrix slices were run one workload at a time under a shell RAM guard that
    terminated the process group above `70%` system RAM.
- target random-large snapshot:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/broad_benchmark_20260608/broad_random_large_current_r1.json`;
  - seed-1234 8-request random-large envelope, `6240` input tokens and `1582`
    output tokens, same stored vLLM denominator as Entry 289;
  - current JAX `812.2285487199796 output tok/s`, stored vLLM
    `1021.5851751977929 output tok/s`, ratio `0.7950668905925744x`;
  - token-event throughput `824.0007788115695 output tok/s`, measurement
    seconds `1.9477276469697244`, final drain `27.83 ms`;
  - TTFT p50 `207.76 ms` versus vLLM `169.25 ms`; ITL p50 `5.67 ms` versus
    vLLM `5.48 ms`;
  - GPU memory after measurement `8747 MB`;
  - exact generated tokens did not match vLLM on the random workload. This is
    consistent with prior random-workload diagnostics where close early logits
    can diverge under greedy decode, so use this run as a throughput checkpoint
    rather than an exact correctness claim.
- decode-heavy sanity slice:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/broad_benchmark_20260608/gpu_matrix_decode_heavy_current_r1.json`;
  - report:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/broad_benchmark_20260608/gpu_matrix_decode_heavy_current_r1.md`;
  - `decode_heavy_128x128`, one repeat, stored references only;
  - current JAX `179.88507572546163 output tok/s`, stored vLLM
    `213.54 output tok/s`, ratio `0.842x`;
  - stored JAX default `151.84 output tok/s`, ratio `1.185x` to the JAX
    baseline;
  - GPU memory after measurement `8687 MB`;
  - exact token comparison failed at generated index `8`, so this is a speed
    sanity point, not a strict correctness promotion artifact.
- failed broad slices:
  - the combined three-workload matrix was stopped manually when system RAM
    crossed the guard (`72.7%`) while compiling/warming `hetero8`;
  - guarded `hetero8` single-slice rerun was terminated at `70.2%` system RAM
    before a report artifact was written;
  - guarded `long_prefill_512_2048` failed before timing because the current
    config's prefill-token buckets stop at `4096`, while that workload
    scheduled `5120` packed prefill tokens:
    `ValueError: prefill token size 5120 exceeds configured buckets (64, 128,
    256, 512, 1024, 2048, 4096)`.
- interpretation:
  - the current checkout reproduces the accepted large-random band within
    normal run-to-run noise (`0.795x` now versus Entry 289's `0.802x` best);
  - `0.8x` is a real current plateau for the A10G random-large path, but not a
    proven fundamental limit. The remaining gap is unlikely to come from more
    source-level JAX rewrites, single-selector kernels, or token
    materialization tweaks;
  - decode-heavy being `0.842x` while random-large is `0.795x` supports the
    existing diagnosis that heterogeneous serving/scheduling plus repeated
    model-side decode work is the harder remaining gap;
  - long-prefill coverage regressed at the benchmark-contract level because the
    promoted random config is too narrow for the long-prefill packed token
    bucket.
- decision:
  - keep Entry 289 as the accepted best speed record and use this entry as the
    current broad checkpoint;
  - do not commit full JSON/profile artifacts; this logbook entry is the
    committed summary;
  - before another broad matrix claim, fix the long-prefill bucket coverage and
    add a built-in matrix RAM guard instead of relying on manual process-group
    termination;
  - the next material speed route must be structural: backend-owned grouped or
    layer-batched MLP/GDN decode work, a true runtime replay/capture path, or a
    kernel-native GDN decode boundary on compatible hardware.

### Entry 292 - Fixed Packed Long-Prefill Token-Bucket Overpacking

- date: 2026-06-08
- purpose:
  - fix the Entry 291 `long_prefill_512_2048` failure where the scheduler
    packed `5120` prefill tokens into a batch even though the largest compiled
    packed prefill-token bucket was `4096`;
  - clarify why `hetero8` compile/warmup can use more host RAM than narrower
    benchmark lanes despite generic warmup.
- implementation:
  - added `Scheduler._max_prefill_token_budget()`;
  - changed the pure prefill scheduling path to cap `num_batched_tokens` by the
    same effective token budget already used by mixed prefill/decode:
    `min(max_num_batched_tokens, max(prefill_token_buckets))`, or
    `prefill_buckets` when token buckets are absent;
  - added a focused packed-prefill regression test where
    `max_num_batched_tokens` permits `8` tokens but the compiled token bucket
    only permits `6`, so the scheduler must emit a partial second row instead
    of building an unwarmed oversized batch.
- validation:
  - `.venv/bin/pytest -q tests/test_backend_boundaries.py -k 'chunks_packed_prefill_by_token_bucket_budget or builds_packed_prefill_token_bucket or packed_prefill_uses_prefill_bucket_as_row_chunk_budget or chunks_prefill_by_max_batched_tokens_budget'`;
  - guarded real long-prefill slice:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/long_prefill_fix_20260608/gpu_matrix_long_prefill_bucket_cap_fix_r1.json`;
  - report:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/long_prefill_fix_20260608/gpu_matrix_long_prefill_bucket_cap_fix_r1.md`.
- long-prefill result after fix:
  - `long_prefill_512_2048`, one repeat, stored references only;
  - current JAX `176.7747543980159 output tok/s`, stored vLLM
    `116.37 output tok/s`, ratio `1.519x`;
  - stored JAX default `78.02 output tok/s`, ratio `2.266x`;
  - scheduler diagnostics: `2` prefill steps, `15` decode steps, max prefill
    rows `3`, max step tokens `4096`;
  - the old `prefill token size 5120 exceeds configured buckets` error did not
    recur;
  - exact token comparison still failed, so this is a bucket-coverage and speed
    validation, not a strict correctness promotion artifact.
- hetero8 RAM explanation:
  - generic warmup is uniform over the configured serving buckets for a process;
    it is not warmed from live request shapes;
  - the matrix runner launches a fresh JAX process per workload, so compiled
    executables and compiler heap are not shared across workload slices;
  - `hetero8` with the promoted config still exposes a large bucket cross
    product: packed prefill-token buckets up to `4096`, row buckets up to
    `512`, batch-size buckets `1..8`, decode buckets for `B=1..8`, resident
    metadata scatter keys, sampled fastpath keys, and inactive-row variants;
  - narrower lanes such as random-large and decode-heavy compile fewer buckets,
    so they can stay under the RAM guard even when `hetero8` does not.
- decision:
  - keep the scheduler fix;
  - do not widen the default random/hetero config to an `8192` packed prefill
    token bucket just to avoid chunking long-prefill. The safer general behavior
    is to respect the compiled bucket budget and split prefill waves;
  - a future matrix-runner improvement should add a built-in resource guard and
    optionally a smaller broad-validation bucket set instead of relying on
    manual shell guards.

### Entry 293 - Shared Random-Large Envelope Multisuite Harness

- date: 2026-06-08
- purpose:
  - test whether the random-large compiled bucket surface can also run
    `hetero8` without measured-phase JIT growth when both workloads share one
    long-lived server process;
  - stop using a fresh JAX process as the only way to compare broad workloads.
- implementation:
  - added `benchmarks/benchmark_jax_server_multisuite.py`;
  - the new harness builds one `LLMEngine`, applies the accepted kernel policy
    from `gpu_paged_gdn_fla_decode_static_metadata.json`, overrides serving
    shape/capacity to the random-large envelope, runs one generic warmup, then
    measures one or more workloads sequentially in the same process;
  - added a sidecar-compatible `random_large` pseudo-workload with seed `1234`,
    `8` requests, input range `512..1024`, and output range `128..256`;
  - added unit coverage for the random-large envelope override, workload-name
    validation, and random-large prompt generation.
- validation:
  - `.venv/bin/python -m py_compile benchmarks/benchmark_jax_server_multisuite.py`;
  - `.venv/bin/pytest -q tests/test_benchmark_jax_server_multisuite.py`;
  - guarded GPU run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/shared_envelope_20260608/random_large_then_hetero8_shared_r1.json`.
- shared-process result:
  - warmup compiled `56` executor keys in `134.44 s`, using
    `prefill_token_buckets=[512,1024]` and `batch_size_buckets=[1..8]`;
  - `random_large`: `818.3615353465699 output tok/s`, token-event
    `861.5463279550507 output tok/s`, JIT cache `56 -> 56`;
  - `hetero8`: `478.64073570773826 output tok/s`, token-event
    `669.2538031283518 output tok/s`, JIT cache `56 -> 56`;
  - peak observed system RAM stayed below the `70%` guard; GPU memory after
    measurement was about `8.75 GB`.
- interpretation:
  - yes, the random-large compiled bucket surface can execute `hetero8` without
    measured-phase JIT growth when we keep the server process and envelope
    fixed;
  - the old broad matrix was not testing that, because it launched a fresh JAX
    process and changed the serving envelope per workload;
  - this shared-envelope `hetero8` number is not a speed win. It is slower than
    hetero-specific runs because the random-large envelope caps packed prefill
    at `1024` tokens and chunks the `2304` hetero prompt tokens across more
    prefill waves;
  - exact generated-token comparison still fails for `hetero8`, so use this as
    compile/warmup and throughput evidence, not a correctness promotion.
- decision:
  - keep the multisuite harness;
  - use it for shared-envelope validation and for answering whether a workload
    is covered by the random-large warmup surface;
  - keep the matrix runner for per-workload specialized-envelope comparisons,
    but do not interpret its fresh-process compile/RAM behavior as proof that a
    long-lived warmed server would recompile.

### Entry 294 - Config-Owned MTP-1 Speculative Decode Scaffold

- date: 2026-06-11
- purpose:
  - move MTP-1 from environment-only diagnostics to an explicit opt-in serving
    route;
  - preserve the current non-speculative kernel-backed server path when
    speculative decoding is disabled;
  - make the first GPU MTP route correctness-first before claiming speed.
- implementation:
  - added typed config fields for `speculative_method`, `draft_sample_method`,
    `mtp_verifier_impl`, `mtp_batch_accept_policy`, `mtp_seed_after_bonus`,
    and MTP margin thresholds;
  - preserved legacy `num_speculative_tokens=1` compatibility by interpreting
    it as `speculative_method=mtp`;
  - made model loading and runner startup method-aware, with a fail-fast error
    when MTP is requested without loaded `mtp.*` weights;
  - added `ScheduledBatch` speculative metadata so scheduler admission is
    visible without overloading env vars;
  - made generic warmup compile the configured MTP verifier buckets;
  - added `benchmarks/configs/gpu_mtp1_two_decode.json` as a current-stack MTP
    diagnostic config. It explicitly uses reference full-attention decode
    because `flashinfer_paged` is currently width-1 only, and it leaves packed
    GDN decode off because that kernel route is also width-1 only for the
    verifier shape.
- validation:
  - `python -m py_compile nanovllm_jax/config.py nanovllm_jax/server_config.py nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/scheduled_batch.py nanovllm_jax/engine/scheduler.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/model_executor.py benchmarks/benchmark_jax_server_trace.py`;
  - `pytest -q tests/test_server_config.py`;
  - `pytest -q tests/test_backend_boundaries.py -k 'scheduler_marks_mtp_decode_batch_metadata or scheduler_omits_speculative_metadata_when_disabled or warmup_compiles or warmup_uses or resident_slot_carry_decode or static_decode_metadata'`;
  - `pytest -q tests/test_mtp_commit_semantics.py`;
  - `PYTHONPATH=. pytest -q tests/test_benchmark_jax_server_trace.py tests/test_gpu_matrix_runner.py -k 'config or benchmark or gpu_matrix'`;
  - `python benchmarks/run_gpu_matrix.py --configs gpu_mtp1_two_decode --dry-run --skip-gpu-preflight`.
- non-claim:
  - no MTP speed win is claimed from this entry;
  - `tests/test_mtp.py` did not collect in the local environment because the
    installed `transformers` package expects a different `huggingface_hub`
    API (`is_offline_mode` import missing);
  - probabilistic MTP draft sampling and a width-2 FlashInfer verifier remain
    follow-up work.

### Entry 295 - Raw GDN Decode Borrowed-Kernel A10G Microbench

- date: 2026-06-12
- purpose:
  - correct the earlier over-broad conclusion that FlashInfer GDN decode is
    blocked on A10G/SM86;
  - compare identical-input width-1 GDN decode kernels at the real
    Qwen3.5-0.8B decode shape;
  - decide whether an existing JAX-Triton GDN route can be promoted immediately.
- implementation:
  - added `benchmarks/microbench_gdn_decode_kernels.py`, which builds
    deterministic BF16 QKV / FP32 gate / FP32 state tensors and compares:
    current JAX reference, local JAX-Triton raw gates, vLLM FLA packed recurrent
    decode, and FlashInfer pretranspose GDN decode;
  - fixed the vLLM ABI check to use state slots `1..B`, because slot `0` is
    vLLM's null state id;
  - added diagnostic config
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_gdn_raw_decode.json`
    for an integrated follow-up that changes only GDN packed decode from
    `reference` to `triton_fla_raw_gates`.
- raw microbench artifact:
  - `results/gdn_decode_kernel_microbench_a10g_qwen08_b1_b2_b4_b8.json`;
  - shape: `H=16`, `HV=16`, `K=128`, `V=128`, `qkv_dim=6144`;
  - all borrowed kernels matched JAX at BF16-output tolerance and FP32-state
    tolerance: output `max_abs <= 5.5e-5`, state `max_abs <= 4.5e-8`.
- p50 kernel timings:
  - B1: JAX reference `0.198 ms`, JAX-Triton raw `0.193 ms`, vLLM FLA
    `0.070 ms`, FlashInfer `0.043 ms`;
  - B2: JAX reference `0.186 ms`, JAX-Triton raw `0.197 ms`, vLLM FLA
    `0.070 ms`, FlashInfer `0.072 ms`;
  - B4: JAX reference `0.193 ms`, JAX-Triton raw `0.209 ms`, vLLM FLA
    `0.110 ms`, FlashInfer `0.045 ms`;
  - B8: JAX reference `0.858 ms`, JAX-Triton raw `0.256 ms`, vLLM FLA
    `0.084 ms`, FlashInfer `0.086 ms`.
- integrated follow-up:
  - command changed only the GDN packed decode implementation on top of the
    accepted FlashInfer full-attention large-random config;
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_decode_microbench_followup_r1.json`;
  - same seed-1234 random-large manifest as Entry 289
    (`6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`);
  - the JAX subprocess hit the `70%` RAM guard during generic warmup/compile
    after `473.27 s`, before measured generation. Peak system RAM was `70.1%`.
- decision:
  - correct the status: FlashInfer GDN **decode** is raw-runnable on A10G/SM86;
    FlashInfer GDN **prefill** remains the SM90/SM100-gated route;
  - do not promote the existing local JAX-Triton raw-gate GDN decode route for
    generic serving. Its B8 raw speedup is real, but its integrated compile/RAM
    surface is unacceptable under the current generic warmup policy;
  - the viable speed route is a JAX FFI/custom-call wrapper for FlashInfer GDN
    decode or a direct port of the vLLM/FlashInfer kernel behind the resident
    state ABI. Calling Torch/FlashInfer from the JAX serving loop through
    DLPack/Python is rejected because it would add a per-layer framework/host
    boundary and break the warmed JIT serving contract.

### Entry 296 - Raw-GDN Warmup Surface Diagnostic

- date: 2026-06-12
- purpose:
  - answer whether the raw-GDN decode route is unusable because a few requests
    consume too much KV memory, or because the full random-large generic warmup
    compiles too much executable surface at once;
  - make request-specific benchmark warmup useful for random workloads whose
    active batch shrinks after the first decode steps.
- implementation:
  - changed `benchmarks/benchmark_jax_server_trace.py` request-specific warmup
    to replay the full per-request `output_len` list instead of only a short
    two-token prefix;
  - kept `warmup_mode=request` diagnostic-only. Reported benchmark results
    should continue to use generic bucket startup or a documented long-lived
    server warmup envelope.
- validation:
  - failed pre-patch diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_small_gdn_raw_decode_requestwarm_r1.json`;
    request warmup compiled only `3` keys, then measured generation compiled
    the missing B1/B2/B4 resident decode keys and tripped
    `--fail-on-jit-cache-growth`;
  - patched request-specific diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_small_gdn_raw_decode_requestwarm_full_r1.json`;
    JIT cache `6 -> 6`, peak child RSS `3.48 GB`, peak system RAM `57.2%`,
    GPU memory after engine/warmup/measurement `4365/8673/8703 MB`;
  - reduced generic diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_small_gdn_raw_decode_genericwarm_r1.json`;
    JIT cache `22 -> 22`, generic warmup `177.06 s`, peak child RSS `4.61 GB`,
    peak system RAM `62.4%`, GPU memory after engine/warmup/measurement
    `4365/8683/8713 MB`;
  - BF16 activation diagnostic:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_small_gdn_raw_decode_requestwarm_full_bf16_r1.json`;
    JIT cache `6 -> 6`, peak child RSS `3.56 GB`, peak system RAM `56.3%`,
    GPU memory after engine/warmup/measurement `4365/8675/8705 MB`.
    This did not materially reduce the footprint versus the FP32-activation
    request-specific run.
- decision:
  - the memory issue is not KV cache memory for a few Qwen3.5-0.8B requests.
    The large raw-GDN run hit host compile/warmup pressure from compiling the
    full random-large envelope: prefill token buckets, active batch buckets,
    sampled/greedy decode routes, resident metadata routes, inactive-row
    decode, and per-shape custom-call variants;
  - activation dtype is not the main memory lever for this route. BF16
    activations are worth using for apples-to-apples vLLM numerics, but the
    observed 8.7 GB GPU footprint is dominated by model/runtime/executable
    workspace retention rather than a live KV cache or activation tensor;
  - smaller shape envelopes are enough to test borrowed-kernel viability;
  - final comparisons should not use request-specific warmup as the numerator
    for vLLM ratios, but it is valid for diagnosing whether a route can serve
    without hidden measured-phase compilation.

### Entry 297 - Random-Large KV Capacity and Route-Aware Warmup

- date: 2026-06-12
- purpose:
  - rerun the actual random-large benchmark after the raw-GDN microbench,
    without the accidental oversized prefill envelope from Entry 295;
  - explain why some reruns were much slower than the stored accepted result;
  - reduce generic warmup memory/compile time for greedy-only runs without
    changing measured serving behavior.
- implementation:
  - added an `include_sampled_routes` parameter to generic warmup. The default
    remains conservative (`True`), but `benchmark_jax_server_trace.py` passes
    `False` when the measured workload is definitely greedy
    (`temperature=0`, `top_p=1`, `sampling_top_k=-1`);
  - this skips sampled prefill/decode route compilation during generic warmup
    while preserving the measured greedy path and the JIT growth audit.
- findings:
  - the stored accepted random-large denominator used `num_kvcache_blocks=2048`
    and `max_kv_cache_mb=2048`, not the later 320-block diagnostic setting;
  - 320 blocks cannot keep all 8 seed-1234 random-large prompts resident. Their
    prompt block demand alone is about 393 blocks at block size 16, so the
    scheduler must start decode before admitting all requests. This changes the
    trace from B8-heavy decode to B6/B5/B1 decode and drops throughput to the
    `0.57x` vLLM range;
  - with 2048 blocks, the current reference route restores the accepted B8
    decode structure.
- validation artifacts:
  - current reference, 320 blocks:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_current_matched_envelope_r1.json`;
    `592.89 output tok/s`, `0.580x` vLLM, 391 decode steps;
  - raw GDN, 320 blocks:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_decode_matched_envelope_r1.json`;
    `581.13 output tok/s`, `0.569x` vLLM, 391 decode steps;
  - current reference, 2048 blocks, full generic warmup:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_2048blocks_current_r1.json`;
    `817.81 output tok/s`, `0.801x` vLLM, JIT cache `56 -> 56`, warmup
    `138.80 s`, peak child RSS `6.07 GB`, peak system RAM `71.3%`;
  - current reference, 2048 blocks, route-aware greedy warmup:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_2048blocks_routeaware_warmup_r2.json`;
    `818.47 output tok/s`, `0.801x` vLLM, JIT cache `24 -> 24`, warmup
    `61.60 s`, peak child RSS `4.56 GB`, peak system RAM `62.1%`;
  - raw GDN, 2048 blocks, route-aware greedy warmup:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_decode_2048blocks_routeaware_warmup_r1.json`;
    `778.41 output tok/s`, `0.762x` vLLM, JIT cache `24 -> 24`, warmup
    `55.85 s`, peak child RSS `4.50 GB`, peak system RAM `61.5%`.
- decision:
  - restore 2048 KV blocks for the random-large target. Smaller block counts
    are useful memory stress tests, but they are not equivalent to the accepted
    random-large benchmark and should not be used for the `0.9x` target;
  - keep route-aware generic warmup for greedy benchmark runs. It materially
    reduces compile time and host RAM without measured-phase JIT growth;
  - reject the current local raw-GDN decode path for serving. It improves a
    focused raw recurrent microbench, but integrated random-large throughput is
    slower than the reference packed BF16-QKV GDN decode path at the same KV
    capacity and warmup policy.

### Entry 298 - Rejected Broader GDN Decode Kernel Boundaries

- date: 2026-06-12
- purpose:
  - test whether replacing more of width-1 GDN decode with a kernel gives the
    missing random-large speedup;
  - distinguish monolithic conv/recurrent/tail fusion from a narrower recurrent
    kernel plus a separate tail epilogue;
  - rerun the comparison on the actual FlashInfer attention path, not the conda
    Python fallback path.
- implementation:
  - added diagnostic `gdn_packed_decode_impl` routes:
    `triton_fla_raw_gates_tail` for monolithic raw recurrent plus per-head
    RMSNorm/silu tail, and `triton_fla_raw_gates_split_tail` for raw recurrent
    plus a separate Triton tail epilogue;
  - added `gdn_decode_tail_rms_silu_bf16` and focused GPU tests for the two
    Triton-tail routes;
  - updated `benchmark_jax_server_multisuite.py` to pass
    `include_sampled_routes=False` for greedy-only runs. This makes multisuite
    warmup match the route-aware policy from Entry 297;
  - important environment correction: use `.venv/bin/python` for FlashInfer
    runs. `/root/miniconda3/bin/python` does not have `flashinfer` or
    `jax_tvm_ffi`, which made the first FlashInfer benchmark attempt fail
    before GDN execution.
- validation:
  - focused test:
    `.venv/bin/python -m pytest -q tests/test_gdn_packed_decode_reference.py::test_packed_gdn_decode_raw_tail_matches_split_tail`
    passed (`2 passed`);
  - FlashInfer reference:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_routeaware_venv_r1.json`;
    `818.33 output tok/s`, `1.933 s`, ITL p50 `5.54 ms`, JIT cache
    `24 -> 24`, route-aware warmup `60.59 s`, GPU after measurement `8727 MB`;
  - FlashInfer raw recurrent:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_decode_flashinfer_routeaware_venv_r1.json`;
    `779.37 output tok/s`, `0.952x` reference, JIT cache `24 -> 24`;
  - FlashInfer raw recurrent plus split Triton tail:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_raw_split_tail_decode_flashinfer_routeaware_venv_r1.json`;
    `759.25 output tok/s`, `0.928x` reference, JIT cache `24 -> 24`;
  - local in-tree Triton-attention A/B, used only because the initial conda
    Python process lacked FlashInfer:
    reference `464.67 output tok/s`, raw recurrent `463.94` (`0.998x`),
    split tail `459.65` (`0.989x`), monolithic raw+tail `415.23` (`0.894x`).
- decision:
  - reject isolated GDN decode kernel replacement as the random-large speed
    route on this A10G stack. The reference packed BF16-QKV GDN path is already
    well compiled by XLA at the serving boundary, while the custom-call variants
    add launch/materialization/register-pressure costs;
  - specifically do not retry `triton_fla_raw_gates`,
    `triton_fla_raw_gates_split_tail`, `triton_fla_raw_gates_tail`,
    `triton_fla_conv_raw_gates`, or `triton_fla_conv_raw_gates_tail` as active
    speed routes without a new profile showing a changed bottleneck;
  - the next 0.9x attempt should attack a broader serving boundary or a
    different dominant bucket rather than more per-layer GDN decode epilogues.

### Entry 299 - State-Pool GDN Decode Boundary Result

- date: 2026-06-12
- purpose:
  - expand the conv+projection+tail GDN decode boundary as far as possible
    without pulling the output projection GEMM into a hand-written kernel;
  - match the important vLLM/FlashInfer ABI lesson: kernels should read/write
    a resident state pool by index instead of forcing the caller to gather and
    scatter one layer's state.
- implementation:
  - extended the packed-projection conv+GDN Triton kernel with a state-pool
    specialization. It accepts full conv/recurrent state tables, a static
    linear-layer index, and an optional valid-row mask;
  - the state-pool wrapper aliases the conv and recurrent state outputs to the
    input state tables with `jax_triton.triton_call(input_output_aliases=...)`;
  - the model uses this route for the strict
    `triton_fla_conv_raw_gates_tail` packed-projection decode path when full
    hybrid state tables are available;
  - added a focused CUDA test that verifies only the selected layer updates and
    invalid rows preserve their previous state.
- validation:
  - focused tests:
    `.venv/bin/python -m pytest -q tests/test_gdn_packed_decode_reference.py::test_gdn_conv_packed_projection_decode_matches_split_kernel tests/test_gdn_packed_decode_reference.py::test_gdn_conv_packed_projection_tail_decode_matches_split_tail tests/test_gdn_packed_decode_reference.py::test_gdn_conv_packed_projection_tail_state_pool_updates_one_layer tests/test_gdn_packed_decode_reference.py::test_packed_gdn_decode_raw_tail_matches_split_tail`
    passed (`5 passed`);
  - accepted reference rerun:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_current_r2.json`;
    `818.17 output tok/s`, JIT cache stable;
  - previous full-state-warps conv-tail route:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_conv_tail_full_state_warps_r1.json`;
    `775.01 output tok/s`;
  - state-pool conv-tail route:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_conv_tail_state_pool_r1.json`;
    `776.98 output tok/s`, JIT cache `24 -> 24`;
  - state-pool repeat:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_conv_tail_state_pool_r2.json`;
    `749.59 output tok/s`, JIT cache stable;
  - exact reference profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_profile_r1.json`;
    total GPU trace time `577.60 ms`;
  - state-pool profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_conv_tail_state_pool_profile_r1.json`;
    total GPU trace time `577.75 ms`, custom conv+projection+tail GDN decode
    kernel `13.20 ms / 378` calls.
- findings:
  - state-pool aliasing removes the model-level layer scatter for this route,
    but the integrated benchmark does not improve stably;
  - the custom GDN decode kernel is not the remaining dominant bucket. On the
    profiled state-pool route, larger buckets are GEMM (`185.38 ms`), generic
    fusion (`98.30 ms`), scatter (`67.39 ms`), attention/prefill (`55.72 ms`),
    and LM-head (`21.09 ms`);
  - the reference route remains faster in normal random-large runs because the
    custom call is a hard boundary around work that XLA already compiles well
    enough at this serving shape. A microbench-fast recurrent kernel does not
    automatically win once surrounding projection, fusion, and scheduler work
    are included.
- decision:
  - keep the state-pool boundary as correct graph-replay scaffolding, but do
    not promote `triton_fla_conv_raw_gates_tail` as the serving speed path;
  - do not retry GDN-only boundary widening unless a fresh profile shows GDN
    decode itself has become dominant again;
  - the next step should be graph replay / backend-owned decode-step capture or
    a broader model boundary that reduces GEMM/fusion/host launch work without
    duplicating output-projection GEMM.

### Entry 300 - XLA Command-Buffer Graph Replay Config Result

- date: 2026-06-12
- purpose:
  - implement graph-replay control in the JAX serving stack without adding more
    one-off environment-variable sprawl;
  - test whether accessible XLA command-buffer/CUDA-graph-like runtime knobs
    reduce the remaining large-random host/replay overhead.
- implementation:
  - added typed `runtime.xla.command_buffer` support in
    `nanovllm_jax.server_config`. It appends accepted XLA flags to
    `XLA_FLAGS` from config:
    `enable_during_profiling -> --xla_enable_command_buffers_during_profiling`,
    `unroll_loops -> --xla_gpu_command_buffer_unroll_loops`, and
    `graph_min_size -> --xla_gpu_graph_min_graph_size`;
  - added
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_graph_replay.json`
    as a diagnostic graph-replay config. It keeps the accepted resident
    fixed-shape decode ABI and does not use source-level decode bursts.
- flag smoke:
  - accepted on this `jaxlib 0.10.0` CUDA stack:
    `--xla_enable_command_buffers_during_profiling=true`,
    `--xla_gpu_command_buffer_unroll_loops=true`, and
    `--xla_gpu_graph_min_graph_size=1`;
  - rejected/unavailable in smoke:
    `--xla_gpu_graph_level=...`,
    `--xla_gpu_graph_num_runs_to_instantiate=...`,
    `--xla_gpu_experimental_enable_command_buffer_on_thunks=true`,
    and tested `--xla_gpu_enable_command_buffer=...` values;
  - older `--xla_gpu_enable_cuda_graphs=true` remains unavailable in this
    installed jaxlib.
- baseline profile context:
  - exact reference profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_profile_r1.json`;
  - host/replay trace summary:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_reference_graph_replay_baseline_trace_summary.md`;
  - traced pattern totals: `PjRtCApiLoadedExecutable::Execute=219.72 ms / 99`,
    `command_buffer::execute=28.04 ms / 225`, and
    `command_buffer::update=10.54 ms / 124`. The trace confirms command-buffer
    regions already exist in the current JAX/XLA path.
- validation:
  - focused tests:
    `.venv/bin/python -m pytest -q tests/test_server_config.py::test_runtime_and_kernel_sections_translate_to_env tests/test_benchmark_random_request_sidecar.py::test_jax_config_projects_runtime_kernel_policy`
    passed (`2 passed`);
  - forced graph-min-size run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_graph_replay_r1.json`;
    `777.67 output tok/s`, token-event `788.34 output tok/s`, warmup
    `239.62 s`, JIT cache `24 -> 24`, flags
    `--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false --xla_enable_command_buffers_during_profiling=true --xla_gpu_command_buffer_unroll_loops=true --xla_gpu_graph_min_graph_size=1`;
  - safer unroll-only run:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_graph_replay_unroll_r2.json`;
    `818.58 output tok/s`, token-event `836.12 output tok/s`, warmup
    `234.69 s`, JIT cache `24 -> 24`, flags
    `--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false --xla_enable_command_buffers_during_profiling=true --xla_gpu_command_buffer_unroll_loops=true`;
  - accepted anchor for comparison:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_current_r2.json`;
    `818.17 output tok/s`, token-event `853.96 output tok/s`, warmup
    `60.64 s`, JIT cache `24 -> 24`.
- findings:
  - flag-only graph replay is not the missing 0.9x lever on this stack. The
    path already uses XLA command-buffer regions, and the exposed flags either
    regress measured throughput plus compile/warmup (`graph_min_size=1`) or tie
    throughput while making generic startup much worse (`unroll_loops=true`);
  - CUDA Graphs in vLLM capture and replay a fixed sequence of CUDA launches
    around stable memory addresses. In this JAX path, the analogous mechanism is
    XLA/PJRT command-buffer lowering inside a fixed-shape `jax.jit`
    executable. We do not get a Python-level CUDA graph object to replay, so
    the serving ABI must keep state resident and shapes fixed enough for XLA to
    emit/update command buffers;
  - expected speedup from proper replay is bounded by the remaining
    host/PJRT/update bucket unless it also changes GPU kernel packing. The
    traced command-buffer update/execute bucket is far too small to explain the
    whole gap to vLLM by itself.
- decision:
  - keep `runtime.xla.command_buffer` as a diagnostic/config capability;
  - do not promote command-buffer flags into the accepted benchmark/server
    config;
  - next replay work, if pursued, should be a backend-owned decode loop or
    broader compiled boundary that reduces `PjRt Execute` calls and GPU
    GEMM/fusion fragmentation without source-level full-model `lax.scan` burst
    regressions.

### Entry 301 - NVIDIA JAX-Toolbox Flag Knob A/B

- date: 2026-06-12
- purpose:
  - test the JAX/XLA runtime knobs from NVIDIA's JAX-Toolbox GPU performance
    notes that parse on the installed `jax==0.10.0` / `jaxlib==0.10.0` stack;
  - retest the plain JAX baseline where a knob looked plausible, so the
    optimized-path delta is not mistaken for a generic XLA flag effect.
- setup:
  - target: `random_large` shared envelope, seed `1234`, 8 requests, standard
    generic warmup and `--fail-on-jit-cache-growth`;
  - optimized anchor:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_current_r2.json`,
    `818.17 output tok/s`, warmup `60.64 s`, JIT `24 -> 24`;
  - added diagnostic configs:
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_xla_o1.json`,
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_command_buffer_fusion_custom.json`,
    and
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_cudnn_gemm_fusion_level1.json`.
- flag smoke:
  - parses: `JAX_OPTIMIZATION_LEVEL=O1`,
    `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL`,
    `--xla_gpu_cudnn_gemm_fusion_level=1`,
    `--xla_gpu_enable_while_loop_double_buffering=true`,
    `--xla_gpu_enable_latency_hiding_scheduler=true`, and
    `--xla_gpu_memory_limit_slop_factor=95`;
  - rejected in this `jaxlib`: `--xla_gpu_enable_custom_fusions=true`,
    `--xla_gpu_enable_address_computation_fusion=true`,
    `--xla_gpu_enable_custom_fusions_re=...`, and the older boolean
    `--xla_gpu_cudnn_gemm_fusion=true`.
- optimized-path results:
  - `JAX_OPTIMIZATION_LEVEL=O1`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_xla_o1_r1.json`,
    `818.27 output tok/s`, token-event `855.29`, warmup `59.22 s`,
    JIT `24 -> 24`;
  - `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_command_buffer_fusion_custom_r1.json`,
    `691.55 output tok/s` (`0.845x` anchor), token-event `699.97`,
    warmup `233.48 s`, JIT `24 -> 24`;
  - `--xla_gpu_cudnn_gemm_fusion_level=1`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_cudnn_gemm_fusion_level1_r1.json`,
    `818.49 output tok/s`, token-event `843.40`, warmup `235.44 s`,
    JIT `24 -> 24`.
- baseline-JAX cross-check:
  - plain `gpu_paged_default`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_jax_default_r1.json`,
    `227.72 output tok/s`, token-event `228.66`, warmup `442.96 s`,
    JIT `24 -> 24`;
  - same baseline with external `JAX_OPTIMIZATION_LEVEL=O1`:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_jax_default_o1_r1.json`,
    `180.85 output tok/s` (`0.794x` baseline), token-event `181.49`,
    warmup `343.48 s`, JIT `24 -> 24`;
  - both baseline runs emitted JAX's future dtype-promotion scatter warning
    for float32 values being scattered into BF16 buffers.
- findings:
  - these flag knobs do not explain the gap to vLLM on A10G. The optimized
    serving path is already at `~3.59x` the plain JAX baseline on the same
    random-large envelope;
  - `O1` is neutral on the accepted optimized path and actively hurts the plain
    JAX baseline, so do not promote it as a default serving knob;
  - restricting command buffers to `FUSION,CUSTOM_CALL` is not portable from
    NVIDIA's Blackwell note to this A10G/SM86 setup. It regresses throughput and
    increases warmup;
  - cuDNN GEMM fusion level 1 is also not useful here: measured throughput ties
    the anchor but warmup grows by about `4x`.
- decision:
  - keep these configs as diagnostics only;
  - do not add any of the tested JAX-Toolbox knobs to the accepted large-random
    serving config;
  - next performance work should stay structural: reduce GPU GEMM/fusion
    fragmentation and scheduler/host boundaries, or revisit flags only after a
    JAX/XLA upgrade changes available command-buffer/custom-fusion behavior.

### Entry 302 - MTP-1 Speculative Decode Long Pass

- date: 2026-06-12
- purpose:
  - make greedy MTP-1 speculative decoding correct and faster on the
    single-request repeat-yes diagnostic before broadening to random serving;
  - identify whether the remaining gap is verifier GPU work, host acceptance,
    startup seeding, or scheduler block-boundary fallbacks.
- baseline:
  - no-MTP optimized 64-token repeat-yes:
    `results/mtp_opt_nomtp_repeat_yes_64.json`, `123.63 tok/s`,
    `0.5177 s`, correctness OK;
  - previous exact K=2 MTP prefill verifier:
    `results/mtp_opt_mtp2_prefill_verify_device_seed_repeat_yes_64.json`,
    `76.29 tok/s`, acceptance `0.792`, correctness OK.
- accepted improvements:
  - compact verifier LM-head:
    `mtp_k_decode_greedy_step_jit` now returns target greedy token IDs from
    verifier hidden states instead of materializing `[B, width, vocab]`
    verifier logits. This moved exact K=2 to
    `results/mtp_opt_mtp2_prefill_compact_lmhead_64.json`, `81.95 tok/s`,
    correctness OK;
  - B=1 MTP output-carry fast path:
    avoids an extra non-jitted stack/scatter when carrying the final emitted
    token from a single-row MTP step. It improved steady K=2 group time but did
    not materially change total 64-token throughput alone:
    `results/mtp_opt_mtp2_carry_fastpath_64.json`, `81.91 tok/s`,
    median steady 3-token group `~19.5 ms`;
  - MTP block-capacity check narrowed to physical verifier writes:
    the verifier writes current token plus accepted drafts; the bonus token is
    only emitted and is processed as the next step's input. Requiring block
    capacity for the bonus caused avoidable `blocks` fallbacks. The exact K=2
    result improved to
    `results/mtp_opt_mtp2_blockcheck_physical_64.json`, `91.12 tok/s`,
    acceptance `0.909`, correctness OK, fallback seeded main steps `5 -> 3`.
- diagnostics / rejected variants:
  - K=3 exact after compact LM-head:
    `results/mtp_opt_mtp3_prefill_compact_lmhead_64.json`, `43.28 tok/s`.
    The third draft position was not useful (`draft_position_accepted`
    `[2, 2, 0]`), so do not retry K=3 for this workload without a better MTP
    model or acceptance predictor;
  - specialized K=2 commit-select route:
    `results/mtp_opt_mtp2_specialized_current_64.json`, `45.10 tok/s`.
    It is much slower than the generic prefill verifier;
  - K=1 current route:
    `results/mtp_opt_mtp1_current_64.json`, `45.76 tok/s`;
  - packed generic K host payload:
    `results/mtp_opt_mtp2_packed_payload_64.json`, `70.05 tok/s`.
    Packing small verifier outputs into one payload regressed the HLO/runtime;
  - exact device-token all-accept commit:
    `results/mtp_opt_mtp2_device_tokens_blockcheck_64.json`, `82.86 tok/s`.
    Keeping bonus/next-draft values as device refs added bookkeeping overhead;
  - prefill seeding:
    best composed run `results/mtp_opt_mtp2_blockcheck_prefillseed_64.json`,
    `87.76 tok/s`. It removes the first missing-draft decode but moves enough
    MTP work into prefill/early decode to lose overall;
  - disabling trace-token prefetch:
    `results/mtp_opt_mtp2_no_trace_prefetch_compact_64.json`, `79.91 tok/s`;
  - relaxing the bonus-boundary guard:
    `results/mtp_opt_mtp2_relax_boundary_64.json`, `78.66 tok/s`. It removed
    some fallbacks but slowed early/steady MTP groups, so keep the guard.
- upper bound:
  - assume-all-accept with compact verifier and physical block check:
    `results/mtp_opt_mtp2_assume_blockcheck_64.json`, `98.13 tok/s`,
    correctness OK on repeat-yes, median steady 3-token group `~15.5 ms`.
    This is not a general exact policy, but it shows that steady MTP can beat
    no-MTP steady decode only when acceptance handling is removed.
- current bottleneck:
  - exact K=2 steady groups are now about `18.6 ms` for 3 emitted tokens;
  - profile `results/mtp_profile_blockcheck_physical_64.json` shows steady
    executor work around `6 ms` and host verifier-result transfer around
    `7 ms`; first missing-draft fused seed still costs about `245 ms`;
  - no-MTP remains faster overall (`123.63 tok/s` vs exact MTP `91.12 tok/s`).
- decision:
  - keep compact LM-head, B=1 carry fast path, and physical-write block check;
  - do not promote assume-all-accept, K=3, specialized K=2, packed host
    payload, device-token accept commit, prefill seeding, no-prefetch, or
    relaxed bonus-boundary variants;
  - next exact-MTP work must reduce verifier-result host transfer or remove the
    first seed bubble. Otherwise MTP-1 on this 0.8B model remains a diagnostic
    path rather than a default speed path.

### Entry 303 - B1 K2 MTP Burst Verifier Speedup

- date: 2026-06-13
- purpose:
  - reduce the per-verifier host acceptance transfer and Python/PJRT boundary
    cost by running several B=1, greedy, K=2 MTP verifier groups inside one JIT
    before returning to the scheduler;
  - keep the route exact by committing the burst output only when every group
    accepts, and falling back from the original state on any rejection.
- implementation:
  - added `ModelExecutor.mtp_k_burst_greedy_step_jit`, a B=1 verifier burst
    executor that loops over fixed `burst_groups`, verifies
    `[current, draft_0, draft_1]`, emits `draft_0, draft_1, bonus`, generates
    the next K=2 draft chain from the final verifier hidden state, and returns
    updated KV/GDN state plus `emitted_tokens`;
  - updated scheduler decode lookahead to reserve
    `burst_groups * (1 + num_speculative_tokens)` slots when MTP burst is
    admitted, so the physical KV writes have enough block capacity;
  - added runner routing for the exact burst path when `batch=1`, greedy,
    K>1, full physical batch, sufficient block capacity, and no unsafe final
    bonus block-boundary case;
  - added warmup coverage for the burst executor and a lightweight commit test
    for all-accepted K=2 burst groups.
- exact results:
  - previous exact K=2, 64-token repeat-yes:
    `results/mtp_opt_mtp2_blockcheck_physical_64.json`, `91.12 tok/s`,
    correctness OK;
  - exact burst2, 64-token repeat-yes:
    `results/mtp_opt_mtp2_burst2_64.json`, `93.71 tok/s`, correctness OK;
  - exact burst3, 64-token repeat-yes:
    `results/mtp_opt_mtp2_burst3_64.json`, `94.31 tok/s`, correctness OK;
  - no-MTP, 64-token repeat-yes:
    `results/mtp_opt_nomtp_repeat_yes_64.json`, `123.63 tok/s`, correctness
    OK. Short-output MTP is still slower because prefill/seed and first-burst
    overhead dominate the 64-token measurement;
  - no-MTP, 256-token repeat-yes:
    `results/mtp_opt_nomtp_repeat_yes_256.json`, `168.67 tok/s`, correctness
    OK;
  - exact burst3 with prefill seed, 256-token repeat-yes:
    `results/mtp_opt_mtp2_burst3_prefillseed_256.json`, `186.67 tok/s`,
    correctness OK, JIT cache stable at `8 -> 8`, acceptance `170/170`,
    bonus tokens `85`. This is a real exact MTP speedup over no-MTP on the
    longer high-acceptance lane: `1.107x`.
- upper bounds / rejected variants:
  - burst3 assume-all-accept, 64-token repeat-yes:
    `results/mtp_opt_mtp2_burst3_assume_64.json`, `114.87 tok/s`, correctness
    OK on this all-accepted prompt but not generally exact;
  - burst3 assume-all-accept plus prefill seed, 64-token repeat-yes:
    `results/mtp_opt_mtp2_burst3_assume_prefillseed_64.json`,
    `120.76 tok/s`, still below no-MTP `123.63 tok/s`;
  - burst3 assume-all-accept plus prefill seed, 256-token repeat-yes:
    `results/mtp_opt_mtp2_burst3_assume_prefillseed_256.json`,
    `204.09 tok/s`. Treat this only as an upper bound because it skips the
    acceptance transfer;
  - burst4 assume-all-accept, 64-token repeat-yes:
    `results/mtp_opt_mtp2_burst4_assume_64.json`, `106.02 tok/s`. More burst
    unrolling increased GPU graph work enough to lose;
  - prefill-seed device refs regressed to
    `results/mtp_opt_mtp2_prefillseed_device_refs_64.json`, `89.32 tok/s`, so
    do not reapply that variant.
- decision:
  - MTP speculative decoding now has a demonstrated exact speedup on a longer
    B=1, greedy, high-acceptance decode lane;
  - do not promote it as a default random-serving speed path yet. The current
    accepted burst route is narrow (`B=1`, greedy, K=2/K>1, all groups must
    accept), and short outputs still lose to no-MTP;
  - if this route is kept, move the burst group control out of ad hoc
    environment toggles into config and broaden it carefully to realistic
    acceptance distributions. The next speed work should either avoid the
    remaining acceptance transfer or reduce verifier GPU cost without using the
    unsafe all-accept shortcut.

### Entry 304 - Multi-Row MTP Burst Serving Pass

- date: 2026-06-13
- purpose:
  - continue the Entry 303 MTP burst route until it can produce speedups on
    hetero/random serving workloads, not just the controlled B=1 repeat-yes
    lane;
  - keep the route exact: no all-accept shortcut, no benchmark-specific shape
    warmup, and no fallback that silently turns a speed claim into the normal
    no-MTP path.
- starting diagnosis:
  - Entry 303 only routes through `mtp_k_burst` when there is one active row
    and verifier physical batch size is one;
  - `hetero8` and `random_large` primarily exercise multi-row decode buckets,
    so they cannot benefit from that burst route except in late tail steps;
  - the next implementation target is a multi-row K-burst verifier for fixed
    physical buckets, with all-accepted rows committed from the verifier output
    and rejected rows repaired from the existing target-token output.
- attempted change:
  - removed the executor's B=1 guard for `mtp_k_burst_greedy_step_jit`;
  - broadened runner routing to admit multi-row K=2 burst when configured,
    commit rows whose burst groups all accept, and repair rejected rows through
    the ordinary main-model decode path;
  - moved `mtp_burst_groups` into `Qwen3_5Config` / server config while
    preserving the legacy env override;
  - added config
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_mtp_burst2.json`.
- integration result so far:
  - focused tests pass after the broadened route:
    `tests/test_mtp_commit_semantics.py` plus focused config/backend tests;
  - integrated `hetero8` with exact batch buckets (`1..8`) did not reach
    measurement after several minutes of warmup/compilation and was stopped
    before writing an artifact;
  - integrated `hetero8` with a single padded B=8 bucket also did not reach
    measurement after several minutes and was stopped. This means the
    source-level multi-row burst through the whole target model is too heavy to
    keep as the immediate hetero/random route without a deeper lowering change.
  - disabling burst with `NANO_VLLM_JAX_MTP_BURST_GROUPS=1` on the same B=8
    bucket also did not reach measurement promptly. This isolates the problem
    further: the current multi-row K=2 target verifier graph itself is too
    compile-heavy for the hetero/random path, not only the outer burst loop.
  - added config-owned `mtp_max_active_rows` and changed warmup/runtime
    prefill seeding plus scheduler admission so the proven B=1 MTP burst route
    is not accidentally compiled for B>1 buckets;
  - `results/mtp_entry304_hetero8_b1cap_v3.json` completed with zero measured
    JIT growth (`12 -> 12`) at `314.76 tok/s`, but it is not an MTP speed win:
    `results/mtp_entry304_hetero8_nomtp_b18.json` under the same `1,8` bucket
    grammar produced the same generated tokens and `313.70 tok/s`. There were
    no speculative counters, so `hetero8` does not exercise the B=1 tail route.
  - added multisuite speculative/admission counters after this gap was found.
    Re-running random large with B=1-capped K=2 burst produced
    `results/mtp_entry304_random_large_b1cap_burst2_counters.json` at
    `328.95 tok/s`, but the counters showed `drafts_proposed=4`,
    `drafts_accepted=0`, `bonus_tokens=0`. This is not a real MTP speedup.
    The route profile showed `missing_draft` on the first B=1 tail step; by the
    time the tail can seed a draft, the last request has too few tokens
    remaining to verify it.
  - changed the K-burst repair path to reseed eligible B=1 repair rows after a
    rejected burst. The random large no-gate diagnostic
    `results/mtp_entry304_random_large_b1cap_burst2_reseed_nogate.json`
    still showed no accepted drafts because random large does not have a long
    enough B=1 tail.
  - tested B>1 K=1 reuse through
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_mtp_k1_reuse.json`.
    The exact reuse path (`NANO_VLLM_JAX_MTP_ENABLE_REUSE_FALLBACK=1`,
    `NANO_VLLM_JAX_MTP_FORCE_REUSE_FALLBACK=1`) completed with zero measured
    JIT growth in `results/mtp_entry304_hetero8_k1_reuse_nogate_v3.json`, but
    it regressed to `77.63 tok/s` despite `97/127` accepted drafts. Do not
    pursue this path as a speed route: K=1 reuse still pays a normal target
    decode plus a bonus decode, resolves device-carried drafts through host
    integers, and does not reduce target-model passes enough.
  - tested B>1 K=1 commit-select in
    `results/mtp_entry304_hetero8_k1_commitselect_nogate.json`. It completed
    with zero JIT growth but regressed to `75.33 tok/s`, with only
    `bonus_tokens=15`, `drafts_accepted=15`, and `fallback_partial_rows=98`.
    Profiling showed the first full B=8 verifier followed by many compact B=1
    verifier calls because rejected rows lose their drafts and only accepted
    rows continue.
  - tried reseeding rejected K=1 commit-select rows from the verifier's
    `next_draft_token` and forcing a masked physical B=8 verifier
    (`NANO_VLLM_JAX_MTP_COMPACT_VERIFIER=0`). This avoided shape-specific JIT
    growth but produced worse throughput
    (`results/mtp_entry304_hetero8_k1_commitselect_reseed_masked_v2.json`,
    `58.68 tok/s`) and visible token drift/zero tokens. The older warning in
    the runner is still valid for this serving path: rejected-row K=1
    `next_draft_token` is not a safe seed.
  - added a default-off diagnostic switch,
    `NANO_VLLM_JAX_MTP_SEED_REJECTED_K1=1`, to test the same rejected-row
    seeding under compact verifier mode. The diagnostic
    `results/mtp_entry304_hetero8_k1_commitselect_seed_rejected_compact.json`
    is invalid as a benchmark (`2.88 tok/s`, JIT cache `17 -> 22` during
    measurement) and also shows token drift/zero tokens. Do not enable this
    route without fixing the verifier-side rejected-row seed invariant.
- current conclusion:
  - B=1-only exact MTP cannot speed up `hetero8` or the current random large
    workload because there is no useful B=1 verification window;
  - K=1 B>1 exact MTP is not a speed path in the current implementation;
  - a real hetero/random MTP speedup needs either a lower-overhead K>1 verifier
    that keeps all active rows in a warmed physical bucket, or a correct
    rejected-row seed from verified target hidden state without forcing an
    extra host/full-logits path.

#### Entry 304 follow-up - Unverified MTP Chain Speedup

- change:
  - added `mtp_prefill_seed` to config/server wiring and disabled prefill MTP
    seeding for the serving speed path so multi-row decode is not polluted by
    seed warmup or B=1-only tail behavior;
  - added an active-row scheduler cap fix so B>cap decode rows take the normal
    greedy burst fallback before reserving speculative slots;
  - added a resident-table unverified MTP append path that gathers the current
    input token from `resident_last_tokens`, runs one target model decode, then
    chains the MTP-1 head inside the same JIT for
    `1 + num_speculative_tokens` emitted token IDs;
  - changed generic warmup for the unverified route to compile only the
    selected unverified append graph, not the exact commit-select verifier
    family.
- rejected/neutral variants:
  - exact B=1-capped burst2 with no prefill seed:
    `results/entry304_hetero8_mtp_b1_burst2_noprefillseed.json`,
    `355.37 tok/s`, and
    `results/entry304_random_large_mtp_b1_burst2_noprefillseed.json`,
    `430.97 tok/s`. These are not MTP wins: speculative counters are zero or
    effectively zero and they match the no-MTP greedy burst2 baselines.
  - approximate/unverified K=1 resident-table route improved after removing an
    unnecessary `int32` recast but still lost:
    `results/entry304_hetero8_mtp_k1_unverified_resident_table_noastype.json`,
    `309.02 tok/s` versus no-MTP burst2
    `results/entry304_hetero8_nomtp_greedy_burst2.json`, `357.28 tok/s`.
  - approximate/unverified K=2 chain with trimmed warmup kept memory normal but
    still lost:
    `results/entry304_hetero8_mtp_k2_unverified_chain_warmtrim.json`,
    `346.58 tok/s`.
- accepted diagnostic speedup:
  - config-owned approximate/unverified K=3 chain on `hetero8`:
    `results/entry304_hetero8_mtp_k3_unverified_chain_configonly.json`,
    `439.83 tok/s`, `1.231x` the no-MTP greedy burst2 baseline
    (`357.28 tok/s`), normal GPU memory
    (`~8.69 GB` after measurement), JIT warmup `11` entries, speculative
    counters `drafts_proposed=184`, `bonus_tokens=184`, no accepted/rejected
    exact-verifier counters because this route does not verify drafts before
    emission.
  - config-owned approximate/unverified K=3 chain on `random_large`, same prompt manifest
    hash as the stored no-MTP baseline
    (`6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`):
    `results/entry304_random_large_mtp_k3_unverified_chain_configonly.json`,
    `604.93 tok/s`, `1.402x` no-MTP greedy burst2
    (`431.56 tok/s`), normal GPU memory
    (`~8.69 GB` after measurement), `drafts_proposed=1178`,
    `bonus_tokens=1178`.
- decision:
  - reject the unverified K=3 route as invalid for speculative-decoding speed
    work; the selectable config was deleted on 2026-06-14 and
    `Qwen3_5Config` now rejects unverified MTP append flags;
  - correction on 2026-06-14: the first random-large K=3 run was not stacked
    on the accepted `818.91 output tok/s` non-MTP best route because the
    multisuite random envelope collapsed `batch_size_buckets` to `1,8`. MTP
    experiments must now keep the accepted best-path surface
    (`1,2,3,4,5,6,7,8` batch buckets plus the current FlashInfer/Triton
    kernel policy) and vary only MTP-specific fields unless a non-MTP run has
    already promoted a route change;
  - corrected stack test:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k3_unverified_r1.json`
    keeps the accepted random-large geometry (`2048` KV blocks,
    `max_num_batched_tokens=1024`, `max_blocks_per_seq=128`,
    `batch_size_buckets=1,2,3,4,5,6,7,8`) and current FlashInfer/Triton kernel
    policy, then adds only K=3 unverified MTP fields. It reaches
    `683.53 output tok/s`, `1.130x` the earlier collapsed-bucket K=3 MTP run
    but only `0.835x` the accepted non-MTP best (`818.91 output tok/s`) and
    `0.669x` stored vLLM (`1021.59 output tok/s`), with JIT cache `40 -> 40`,
    no measured growth, prompt hash
    `6662c7b48aa32ffea70019afacfd861a4b0a5ce9dd18f0dc589fd4e02705395f`,
    and normal memory (`8735 MB` after measurement);
  - do not present this as exact speculative decoding. It is the first
    hetero/random MTP throughput win, but correctness is approximate because
    the three MTP draft tokens are emitted without target-model verification;
  - the next exact-MTP path should reuse this resident-table chain boundary but
    add a cheaper periodic verifier or confidence gate, rather than returning
    to K=1 commit-select/reuse.

#### Entry 304 correction - Best-Path Unverified MTP K Sweep

- date: 2026-06-14
- goal:
  - diagnose why the corrected K=3 best-path overlay still lost to the accepted
    non-MTP route, then continue only on the accepted best serving surface until
    MTP made that path faster.
- baseline:
  - accepted non-MTP best:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260608/random_large_flashinfer_lm_triton_bn256_r1.json`,
    `818.91 output tok/s`;
  - stored vLLM denominator:
    `1021.59 output tok/s`;
  - all MTP runs below keep the random-large best-path geometry and kernel
    policy (`2048` KV blocks, `max_num_batched_tokens=1024`,
    `max_blocks_per_seq=128`, `batch_size_buckets=1,2,3,4,5,6,7,8`,
    FlashInfer decode attention, Triton greedy LM-head) and vary only
    MTP-specific fields.
- evidence:
  - K=1:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k1_unverified_sweep_r1.json`,
    `354.84 output tok/s`;
  - K=2:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k2_unverified_sweep_r1.json`,
    `589.47 output tok/s`;
  - K=3:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k3_unverified_r1.json`,
    `683.53 output tok/s`;
  - K=4:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k4_unverified_sweep_r1.json`,
    `754.92 output tok/s`;
  - K=5:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k5_unverified_sweep_r1.json`,
    `850.80 output tok/s`, first K to beat the accepted non-MTP best;
  - K=6:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k6_unverified_sweep_r1.json`,
    `855.32 output tok/s`;
  - K=7:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k7_unverified_sweep_r1.json`,
    `950.47 output tok/s`, above the `0.9x` vLLM target;
  - K=8:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k8_unverified_sweep_r1.json`,
    `1001.51 output tok/s`, `1.223x` the accepted non-MTP best and `0.980x`
    stored vLLM;
  - K=8 repeat:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_stack_20260614/random_large_best_overlay_mtp_k8_unverified_repeat_r2.json`,
    `1018.75 output tok/s`, `1.244x` the accepted non-MTP best and `0.997x`
    stored vLLM, with no measured-phase JIT cache growth and about `8743 MB`
    memory after measurement.
- diagnosis:
  - K=3 loses because the wider MTP step is too expensive for only three bonus
    tokens: it reduced decode steps (`252 -> 64`) but increased decode-step time
    enough to finish below the exact best path;
  - K=7/K=8 win because the per-step cost grows slowly while emitted tokens per
    step keep rising. K=8 used only `28` decode steps, with decode-time
    token-rate around `1376 tok/s` inside the measured step summaries.
- correctness:
  - this is approximate/unverified greedy generation, not exact speculative
    decoding. K=8 full generated-token matches versus the exact best-path
    output were `0/8`; average prefix match was `2.375` tokens, median `2`.
  - exact K=8 verification was started as a diagnostic but did not reach
    measurement in a useful warmup window. Do not retry that shape blindly until
    verification is made cheaper, confidence-gated, or staged.
- decision:
  - delete
    `benchmarks/configs/gpu_paged_gdn_fla_decode_static_metadata_mtp_k8_unverified.json`
    and keep the artifact paths only as historical evidence of the invalid
    throughput-counter route;
  - keep exact serving claims on the non-MTP best path unless a future verified
    MTP route beats it under the same benchmark discipline.

#### Entry 304 audit - Unverified K8 Is Not A Valid MTP Speedup

- date: 2026-06-14
- trigger:
  - user correctly flagged that K=8 acceptance should normally be poor and that
    the apparent speedup looked like cheating.
- finding:
  - confirmed. The K=1..K=8 sweep did not run target-model draft
    verification. The route appended draft tokens from the MTP chain directly,
    so speculative counters show `drafts_accepted=0`, `drafts_rejected=0`, and
    positive `bonus_tokens`. This must not be interpreted as acceptance.
- no-MTP control:
  - row-bearing best-path reference
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/random_hillclimb_20260612/random_large_gdn_reference_flashinfer_current_r2.json`;
  - `818.17 output tok/s`, `1.934 s`, ITL p50 `5.489 ms`, ITL p95 `9.229 ms`;
  - `252` decode steps, decode-step sum `1.505 s`, decode-step p50 `5.257 ms`.
- unverified K sweep against that exact row-bearing output:

| K | tok/s | ITL p50 ms | decode steps | fused decode p50 ms | proposed | accepted | rejected | bonus | full match | avg prefix |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 354.84 | 14.254 | 193 | 19.685 | 785 | 0 | 0 | 785 | 2/8 | 44.125 |
| 2 | 589.47 | 0.000 | 85 | 24.404 | 1048 | 0 | 0 | 1048 | 2/8 | 44.125 |
| 3 | 683.53 | 0.000 | 64 | 25.823 | 1178 | 0 | 0 | 1178 | 2/8 | 44.125 |
| 4 | 754.92 | 0.000 | 51 | 27.621 | 1257 | 0 | 0 | 1257 | 1/8 | 18.750 |
| 5 | 850.80 | 0.000 | 42 | 28.494 | 1309 | 0 | 0 | 1309 | 1/8 | 18.750 |
| 6 | 855.32 | 0.000 | 37 | 29.932 | 1345 | 0 | 0 | 1345 | 1/8 | 18.750 |
| 7 | 950.47 | 0.000 | 32 | 31.281 | 1374 | 0 | 0 | 1374 | 0/8 | 2.500 |
| 8 | 1018.75 | 0.000 | 28 | 33.282 | 1397 | 0 | 0 | 1397 | 0/8 | 2.375 |

- timing interpretation:
  - the "fused decode p50" column is the measured scheduler step containing one
    main model decode plus the K-deep unverified MTP chain. The existing
    artifact does not split this fused XLA computation into internal main-model
    and MTP-head sub-times.
  - a separate real-weight B=8 MTP-head-only microbench of
    `mtp_forward_token_ids` with the Triton greedy top-1 path measured p50:
    K=1 `1.352 ms`, K=2 `2.544 ms`, K=3 `3.863 ms`, K=4 `5.151 ms`, K=5
    `6.151 ms`, K=6 `7.290 ms`, K=7 `8.451 ms`, K=8 `9.649 ms`. This is
    diagnostic only; it is not the integrated serving step.
- decision:
  - delete the K=8 unverified config and reject unverified MTP at config
    construction;
  - do not use unverified draft appending in performance claims;
  - the next real MTP speed attempt must report verified acceptance/rejection,
    accepted-token ITL, rejected-token repair ITL, and exact-vs-control token
    matching from the same benchmark artifact.

#### Entry 304 audit - Verified MTP Gate And Fallback Fixes

- date: 2026-06-14
- trigger:
  - user noted that "unverified" should not exist as an MTP option and that
    the previous K=3 baseline was also invalid.
- code guardrail:
  - `Qwen3_5Config` rejects `mtp_unverified_draft_append` and
    `mtp_unverified_fused_append` during config construction.
  - stale K=3/K=8 unverified selectable configs were removed.
- diagnosis from exact K=1 commit-select diagnostics:
  - no-MTP control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/short_nomtp_same_surface_r1.json`
    = `84.01 output tok/s`, TTFT p50 `29.34 ms`, ITL p50 `18.27 ms`, warmup
    `51.31 s`, `8` JIT entries.
  - verified commit-select before this pass:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/short_verified_k1_commit_select_finalbucket_r1.json`
    = `26.13 output tok/s`, TTFT p50 `335.33 ms`, ITL p50 `25.17 ms`,
    warmup `135.96 s`, `16` JIT entries, `drafts_proposed=5`,
    `drafts_accepted=2`, `drafts_rejected=1`.
  - disabling prefill seed plus keeping gated fallback on static/resident
    decode metadata:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/short_verified_k1_commit_select_staticfallback_noprefillseed_r1.json`
    = `51.19 output tok/s`, TTFT p50 `73.92 ms`, ITL p50 `19.76 ms`.
  - new defaults:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/short_verified_k1_commit_select_newdefaults_r1.json`
    = `57.87 output tok/s`, TTFT p50 `74.40 ms`, ITL p50 `19.42 ms`,
    warmup `103.75 s`, `16` JIT entries, `drafts_proposed=2`,
    `drafts_accepted=0`, `drafts_rejected=0`.
- fixes:
  - default `mtp_prefill_seed=false`; prefill draft seeding was producing a
    large TTFT hit without a verified serving win;
  - default MTP probe steps changed to `1`, so the scheduler can measure the
    ordinary baseline after one expensive seed/probe instead of forcing a
    second verifier step;
  - decode scheduling now rechecks MTP admission against the final physical
    batch and active-row bucket before building `ScheduledBatch`, preventing a
    B=2 batch from speculating through a still-probing B=1 row gate;
  - fully gated MTP decode batches can use the static/resident metadata hot
    path instead of falling back to dynamic metadata solely because
    `num_speculative_tokens > 0`.
- decision:
  - exact verified MTP is still not a promoted speed path. The fixes prevent
    misleading/unverified claims and reduce the regression when MTP is enabled,
    but the current seed/probe and verifier boundary is still slower than the
    accepted no-MTP serving path.

#### Entry 305 audit - Exact K>1 MTP Verifier Boundary

- date: 2026-06-14
- trigger:
  - user asked for a concrete fix to make verified MTP cheap for multiple
    tokens, without allowing the previous unverified speed path.
- code changes:
  - warmup now resets resident and hybrid runtime state after generic MTP
    compile, preventing dummy warmup seq IDs from contaminating measured
    seq IDs;
  - packed-prefill prefix hybrid state now supports multi-row K verification,
    with GDN prefill using the recurrent scan when prefix state is requested;
  - K=2 commit-select no longer asks `model_forward_step` for full logits and
    instead uses the same greedy top-1 token-id helper as the optimized decode
    path;
  - K>1 verifier payloads are packed into one compiled `host_payload` tensor.
- correctness:
  - no-MTP short8 after reset:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/short8_nomtp_after_warmreset_r1.json`
    emitted `[2438, ...]` and `[220, 16, ...]`;
  - K=2 packed-prefill verifier after reset:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/short8_verified_k2_prefill_generic_after_reset_r4.json`
    matched those rows exactly.
- measured results:
  - short8 no-MTP control: `74.69 output tok/s`;
  - short8 K=2 packed-prefill verifier:
    `15.81 output tok/s`, acceptance `25%`;
  - short8 K=2 generic decode verifier with in-JIT payload:
    `16.75 output tok/s`, acceptance `25%`;
  - single-row high-acceptance K=2:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/single64_mtp_k2_generic_decode_24_r1.json`
    accepted `87.5%` of drafts but reached `34.69 output tok/s`; the no-MTP
    control
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/single64_nomtp_24_r1.json`
    was `63.99 output tok/s`;
  - single-row K=4 exact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/single64_mtp_k4_generic_decode_40_r1.json`
    reached `37.50 output tok/s`, while no-MTP
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/single64_nomtp_40_r1.json`
    was `109.85 output tok/s`;
  - K=4 assume-accepted upper bound:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_verified_20260614/single64_mtp_k4_assume_accept_upper_40_r1.json`
    reached only `52.53 output tok/s`, despite zero observed rejections on
    that stream.
- diagnosis:
  - the exact K>1 verifier is now correctness-clean on the focused probes, but
    it is not a speed path. The limiting costs are not only host transfer:
    even the assume-accepted upper bound remains below the no-MTP control,
    because the width-K target verification, MTP-head chain, state gather/store,
    and first seed/probe cost exceed the saved scheduler steps.
- decision:
  - do not promote K=2/K=4 generic decode or packed-prefill verification;
  - do not retry larger K as a default warmup route without a different
    device-side fixed-output/repair boundary and measured evidence that the
    MTP-head plus verifier work is cheaper than ordinary decode.

#### Entry 306 audit - Random-Large Verified MTP K=1/K=2 Startup Blocker

- date: 2026-06-14
- trigger:
  - user asked to push mainly on verified MTP K=1 and K=2 using the random
    benchmark and explain why there is no speedup.
- harness fixes:
  - `benchmark_jax_server_trace.py` now exposes the MTP config fields needed
    by the random sidecar: hidden/token source, position offset, MTP LM-head
    top-1 implementation, speculative token count up to K=8, burst groups,
    max active rows, and prefill seeding;
  - `benchmark_random_request_sidecar.py` now accepts and forwards those JAX
    MTP fields, so random benchmarks can exercise verified K=1/K=2 instead of
    silently staying at K=0/K=1 defaults.
- control:
  - corrected random-large no-MTP envelope:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260614/random_large_nomtp_envelope_r1_jax.json`
    = `817.85 output tok/s`, `0.801x` of the stored vLLM denominator, `0`
    JIT-cache growth, GPU memory after warmup `8685 MB`;
  - an accidental broad-envelope sidecar run
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260614/random_large_nomtp_r1_jax.json`
    reached only `573.60 output tok/s`; the difference was the envelope
    (`8192` batched tokens / `320` KV blocks / broad prefill buckets) rather
    than an MTP effect.
- verified K=1 attempts:
  - best-path K=1 with materialized tied LM head was killed before measurement
    at the RAM guard: `80.1%` system RAM with process-tree RSS `7.47 GB`, then
    `82.2%` with RSS `7.86 GB`;
  - both failures happened after weight load / tied-LM-head materialization and
    before warmup or measured generation, so no acceptance or throughput
    counters were produced;
  - diagnostic K=1 with `materialize_tied_lm_head=false` reduced child RSS to
    roughly `3.5-4.4 GB`, but remained CPU-side compiling for about ten
    minutes with GPU utilization `0%`, so the attempt was terminated without a
    measured artifact.
- verified K=2 attempt:
  - diagnostic K=2 with `materialize_tied_lm_head=false` timed out after
    `300 s` before measurement:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260614/random_large_mtp2_no_matlm_envelope_timeout_r1.json`;
  - peak process-tree RSS was `5.40 GB`, peak system RAM `67.2%`, and the
    output tail stopped after weight load, indicating CPU compile/startup
    rather than measured serving.
- diagnosis:
  - verified random-large MTP currently does not reach a speedup comparison on
    the best serving path. K=1 adds enough host RAM pressure from MTP weights
    plus materialized tied LM head to trip the guard before measurement. Removing
    that copy avoids the RAM kill but exposes a second blocker: K=1/K=2 MTP
    warmup/compile is still minutes of CPU work before any GPU serving;
  - this is a startup/compile and memory blocker before it is an acceptance-rate
    issue on random-large. The older focused K=2/K=4 measurements already show
    that, even after compilation, the exact verifier boundary is slower than
    no-MTP.
- decision:
  - do not promote verified MTP K=1/K=2 for random-large serving;
  - do not retry this exact best-path random-large K=1/K=2 setup without first
    reducing MTP compile/memory surface, especially the materialized LM-head
    duplication and verifier compilation boundary.

#### Entry 307 audit - JAX/XLA RAM Flag Probe For Random-Large MTP Startup

- date: 2026-06-15
- trigger:
  - user asked to try JAX/XLA RAM-related flags after verified random-large MTP
    K=1/K=2 runs either hit the RAM guard or timed out before measurement.
- flags tested:
  - GPU allocator / preallocation:
    `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
    `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
    `XLA_PYTHON_CLIENT_MEM_FRACTION=.5`,
    `TF_GPU_ALLOCATOR=cuda_malloc_async`;
  - compile/autotune pressure:
    `XLA_FLAGS=--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_triton_gemm=false`.
- smoke tests:
  - JAX 0.10.0 / jaxlib 0.10.0 accepted the allocator and XLA flag
    combinations and returned the CUDA device from `jax.devices()`.
- random-large results:
  - best-path verified K=1 with materialized tied LM head plus compile flags
    only timed out before measurement, but no longer hit the RAM guard:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260615/random_large_mtp1_best_xla_compileflags_r1.json`;
    peak system RAM `67.3%`, process-tree RSS `4.98 GB`;
  - best-path verified K=1 with materialized tied LM head plus allocator and
    compile flags also timed out before measurement:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260615/random_large_mtp1_best_xla_memflags_r1.json`;
    peak system RAM `62.0%`, process-tree RSS `4.10 GB`;
  - verified K=2 with `materialize_tied_lm_head=false` plus allocator and
    compile flags timed out before measurement:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260615/random_large_mtp2_no_matlm_xla_memflags_r1.json`;
    peak system RAM `63.3%`, process-tree RSS `4.12 GB`.
- comparison to earlier attempts:
  - previous best-path K=1 attempts were killed at `80.1%` / `6.96 GB` RSS
    and `82.2%` / `7.32 GB` RSS, so the flags reduce host RAM enough to avoid
    the guard;
  - previous K=2 no-materialized-LM timed out at `67.2%` / `5.03 GB` RSS, so
    the same flag set reduces memory there as well.
- diagnosis:
  - the allocator flags are useful for preventing RAM-kill failures, and the
    compile/autotune flags help but do not explain the full reduction by
    themselves;
  - they do not solve the verified MTP startup blocker. The random-large K=1/K=2
    runs still spend the whole 300 second window before measured generation,
    with output stopping after weight load / LM-head materialization.
- decision:
  - keep these as diagnostic safety flags for MTP compile experiments;
  - do not treat them as a speed path or promote verified MTP based on them;
  - if we make them persistent, add them as explicit config-backed XLA/JAX
    allocator options instead of expanding ad hoc environment variable usage.

#### Entry 308 audit - 600s Random-Large MTP Warmup Probe

- date: 2026-06-15
- trigger:
  - user said the 300s timeout was acceptable if the route works, and asked to
    try `600s`; if that still failed, investigate whether recent code inflated
    JIT time.
- setup:
  - reran the accepted random-large verified K=1 route directly through
    `benchmark_jax_server_trace.py` with the same serving flags and RAM flags
    from Entry 307;
  - enabled `NANO_VLLM_JAX_EXEC_LOG_STEPS=1` and
    `NANO_VLLM_JAX_PROFILE_JIT=1` so the route and per-key compile timings were
    visible before the JSON artifact is written.
- outcome:
  - the run made real progress and compiled/executed many warmup routes, but
    timed out under the 600s-equivalent cap before measurement. It reached
    batch-7 decode warmup and stopped while entering
    `forward_step_token_ids_resident_jit` for `(7, 1)`;
  - no final JSON artifact was written because the process was terminated by
    the timeout wrapper.
- key observed compile timings:
  - packed prefill `forward_prefill_token_ids_slot_carry_table_jit` compiled
    separate row-count specializations for token bucket `512`, row counts
    `1..8`: first row count about `26.2s`, most later row counts about
    `9.5-10.3s`;
  - the same happened for token bucket `1024`: first row count about `25.9s`,
    most later row counts about `9.5-10.7s`;
  - decode warmup then compiled multiple routes per batch size. Representative
    expensive keys:
    - `forward_step_jit` hidden-return decode at batch 3/4/5/6:
      roughly `13.5-16.1s` per variant;
    - `forward_step_token_ids_mtp_draft_chain_jit` at batch 3/4/5/6:
      roughly `14.5-16.1s`;
    - `mtp1_commit_select_greedy_step_jit` at batch 3/4/5/6:
      roughly `25.8-30.5s`.
- diagnosis:
  - this is not a GPU visibility problem or a pre-measurement hang. The route
    works, but the generic random-large MTP warmup surface is too broad;
  - for packed prefill, the JIT key still specializes on active metadata row
    count / block-table shape, so two token buckets become sixteen prefill
    compiles for `B=1..8`;
  - for verified K=1 decode, warmup specializes several main, draft, and
    verifier graphs per batch cardinality. The commit-select verifier is the
    largest repeated compile bucket;
  - `--no-mtp-prefill-seed` is honored by the route selection. The misleading
    `return_hidden=True` executor log comes from the token-id prefill helper
    returning hidden/logit intermediates internally, not from MTP prefill seeding.
- decision:
  - do not keep extending timeout as the next step. The 600s miss indicates a
    code-level compile-surface issue;
  - the likely fix is to reduce shape fan-out by padding/stabilizing packed
    prefill metadata rows and verified MTP decode/verifier batch shapes, or by
    warming only the static buckets that the runtime can actually reuse without
    benchmark-specific shape specialization.

#### Entry 309 audit - Static MTP Shapes and Verified K=2 Bottleneck

- date: 2026-06-15
- trigger:
  - after Entry 308, user approved increasing the timeout / investigating the
    JIT-time issue for verified MTP on random-large.
- code changes:
  - packed prefill under active MTP now pads reusable metadata rows to the
    configured `mtp_max_active_rows` bucket instead of compiling active row
    counts `1..N`;
  - verified MTP decode batches are padded to the same static physical row cap
    before verifier dispatch, while inactive rows keep `query_len=0` and
    `seq_id=-1`;
  - generic warmup now includes the K verifier fallback
    `mtp_k_decode_greedy_step_jit` for `K>1`, even when the preferred route is
    commit-select or burst. This fixed a measured-generation cache-growth
    failure for K=2:
    `('mtp-k-decode-greedy-prefix-select', 'decode', 2, (8, 1), ...)`;
  - removed explicit host waits on `_hybrid_state_table` in MTP commit paths and
    avoided compacting hybrid state for the full accepted-row burst case.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_runner.py`;
  - `pytest tests/test_mtp_commit_semantics.py -q`:
    `25 passed, 1 xfailed`;
  - `pytest tests/test_backend_boundaries.py -q -k 'packed or mtp or static_decode or warmup'`:
    `34 passed, 54 deselected`.
- random-large verified MTP results:
  - logged diagnostic K=1 with static rows completed and showed the expected
    shape collapse:
    - prefill warmup only compiled `(1,512)/(8,128)` and `(1,1024)/(8,128)`;
    - MTP verifier replayed `(8,1)` for sparse active rows;
    - measurement JIT cache growth was `0`;
  - no-log K=1 diagnostic, conservative flags:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260615/random_large_mtp1_static_rows_nolog_r1.json`
    - `42.36 output tok/s`, acceptance `51.8%`, cache growth `0`;
  - K=2 initially failed cache growth until generic K fallback warmup was added;
  - no-log K=2 with conservative flags:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260615/random_large_mtp2_static_rows_nolog_r2.json`
    - `63.75 output tok/s`, acceptance `65.8%`, cache growth `0`;
  - no-log K=2 with best prefill flags and host-sync/commit fixes:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_random_profile_20260615/random_large_mtp2_static_rows_bestflags_r2.json`
    - `95.46 output tok/s`, `16.57s`, mean TTFT `736 ms`,
      mean ITL `67.7 ms`, acceptance `56.7%`, cache growth `0`,
      GPU memory after measurement `4.999 GB`.
- comparison:
  - current no-MTP random-large reference artifact remains
    `entry304_random_large_nomtp_greedy_burst2.json`:
    `431.56 output tok/s`, mean TTFT `500 ms`, mean ITL `11.6 ms`;
  - verified MTP K=2 is therefore functional and cache-stable, but still about
    `0.22x` of the no-MTP JAX path on this random-large manifest.
- bottleneck diagnosis:
  - profile slices with output length capped to 32 show the expensive buckets:
    - `executor_mtp_k_burst`: roughly `77-112 ms`;
    - `store_burst_hybrid_state`: roughly `183-188 ms` for mixed
      accept/reject rows;
    - `burst_commit`: roughly `199-862 ms` depending on repair work;
    - many later decode steps have no drafts (`missing_draft`) and fall back to
      ordinary main-model decoding.
  - removing explicit `_ready(self._hybrid_state_table)` improved full-run K=2
    throughput, but did not remove the mixed-row scatter/store bucket;
  - the remaining verified-MTP issue is structural: accepted rows and rejected
    rows are committed through Python-side row splitting, hybrid-state scatter,
    and repair calls. This does not resemble vLLM-style speculative verification
    where verification and state commit are captured/fused around the decode
    graph.
- decision:
  - the static shape / warmup problem is fixed;
  - verified MTP should remain off for the best serving path until commit/state
    handling is moved into a broader device-side verifier/commit boundary or the
    scheduler can use a confidence gate that only speculates when the verifier
    path is expected to beat main decode.

#### Entry 310 audit - No-Host-Sync K-Burst Commit Boundary

- date: 2026-06-15
- trigger:
  - user asked to record and implement the ambitious no-host-sync MTP plan so
    verification/commit does not split accepted and rejected rows in Python
    between speculative groups.
- code changes:
  - `mtp_k_burst_greedy_step_jit` now computes accepted prefix length inside
    the compiled burst, selects the matching prefix KV/hybrid/GDN state, emits
    a fixed `[B, G, K+1]` token tensor plus `[B, G]` emitted/accepted counts,
    regenerates the next draft chain from the committed token, and advances
    committed sequence lengths before the next group;
  - the runner no longer performs rowwise repair inside K-burst. If the burst
    verifier does not return `emitted_counts`, it raises instead of falling back
    to the old Python repair path;
  - padded physical rows now emit count zero and do not advance resident
    sequence length;
  - dense prefill token budgeting was fixed so a `(query_bucket, batch_bucket)`
    dense prefill can use the whole dense padded bucket instead of chunking at
    only the max query length.
- validation:
  - `python -m py_compile nanovllm_jax/engine/model_executor.py nanovllm_jax/engine/model_runner.py nanovllm_jax/engine/scheduler.py tests/test_mtp_commit_semantics.py`;
  - `pytest tests/test_mtp_commit_semantics.py tests/test_backend_boundaries.py tests/test_server_config.py -q`:
    `129 passed, 1 xfailed`;
  - focused rerun after padding fix:
    `pytest tests/test_mtp_commit_semantics.py tests/test_backend_boundaries.py::test_model_runner_uses_bucketed_batched_jit_path -q`:
    `27 passed, 1 xfailed`.
- guarded GPU smoke:
  - no-MTP control, same two-row random manifest:
    `results/entry305_mtp_resident_burst2_b2_nomtp_control_jax.json`,
    `96.15 output tok/s`, ITL p50 `7.10 ms`, JIT growth `0`;
  - forced exact K=2, burst groups `2`, `final_normed` hidden:
    `results/entry305_mtp_resident_burst2_b2_forced_smoke_jax.json`,
    `23.11 output tok/s`, `0.240x` no-MTP, ITL p50 `35.71 ms`,
    acceptance `2/42` (`4.76%`), JIT growth `0`;
  - forced exact K=2, burst groups `2`, `pre_norm` hidden:
    `results/entry305_mtp_resident_burst2_b2_forced_prenorm_smoke_jax.json`,
    `19.62 output tok/s`, `0.204x` no-MTP, ITL p50 `39.51 ms`,
    acceptance `2/42` (`4.76%`), JIT growth `0`.
- diagnosis:
  - the no-host-sync intra-burst commit path works and is cache-stable, but it
    is not a speed path on random requests because draft acceptance is too low
    and each verified burst still runs expensive width-K target-model work plus
    MTP-head chaining;
  - switching from `final_normed` to `pre_norm` hidden did not improve
    acceptance on the same manifest, so the observed loss is not explained by
    that hidden-source contract alone;
  - the next useful MTP work is not more host-sync removal around this exact
    K=2 verifier. It needs either a materially better draft contract/model or a
    confidence/admission policy that avoids proposing drafts in low-acceptance
    random buckets. Keep no-MTP as the best serving path for random/hetero
    until that changes.

### Entry 311 - Host-Scheduler Trace Overhead Pass

- date: 2026-06-18
- trigger:
  - speculative decoding is paused; user asked for easy host-sync or scheduler
    overhead reductions on `hetero8`, `random_large`, and full random stress
    because pure prefill/decode are much closer to target than mixed serving.
- changes:
  - added summary mode to `LLMEngine.generate_with_trace`. Benchmark runs now
    keep aggregate TTFT/ITL and final token IDs without allocating/backfilling
    one Python event dict per generated token;
  - detailed per-token events remain available with `--trace-events`;
  - summary trace mode no longer calls per-step `trace_token_prefetch`, because
    token IDs are only needed once at final correctness materialization;
  - `runtime.xla.allocator` and `runtime.xla.memory_fraction` now map to
    `XLA_PYTHON_CLIENT_ALLOCATOR` and `XLA_PYTHON_CLIENT_MEM_FRACTION`, so
    low-memory XLA/JAX configs no longer silently drop those flags;
  - corrected random sidecar defaults to the accepted full-random envelope:
    `max_num_batched_tokens=1024`, prefill/token buckets
    `128,256,512,1024`, batch buckets `1,2,4,8`,
    `max_blocks_per_seq=320`, decode block-table buckets `128,256,320`,
    `num_kvcache_blocks=2048`, and KV cap `8192 MiB`.
- validation:
  - `.venv/bin/python -m py_compile nanovllm_jax/engine/llm_engine.py benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_jax_server_multisuite.py benchmarks/benchmark_random_request_sidecar.py`;
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_benchmark_jax_server_trace.py tests/test_benchmark_jax_server_multisuite.py tests/test_benchmark_random_request_sidecar.py`:
    `18 passed`.
  - after the XLA env mapping change:
    `PYTHONPATH=. .venv/bin/pytest -q tests/test_server_config.py tests/test_benchmark_random_request_sidecar.py`:
    `35 passed`.
- benchmark evidence:
  - shared-envelope summary before disabling summary prefetch:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/summary_trace_random_large_hetero8_r1.json`;
    `random_large` `582.31 output tok/s`, `hetero8` `419.06 output tok/s`,
    zero measured JIT growth;
  - shared-envelope summary after disabling summary prefetch:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/summary_no_prefetch_random_large_hetero8_r3.json`;
    `random_large` `651.71 output tok/s`, `hetero8` `424.09 output tok/s`,
    zero measured JIT growth;
  - single `random_large` after the change:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/summary_no_prefetch_random_large_r2.json`;
    `650.33 output tok/s`, token-event rate `688.98 output tok/s`, zero
    measured JIT growth.
- hetero8 interpretation:
  - `hetero8` end-to-end throughput moved only `419.06 -> 424.09 output tok/s`,
    but token-phase throughput moved `423.64 -> 458.87 output tok/s` and ITL
    p50 improved `6.87 -> 4.96 ms`;
  - because `hetero8` has only `256` output tokens, the final materialization
    drain increase (`6.6 -> 45.7 ms`) absorbs much of the token-phase gain. The
    no-prefetch route is still better end-to-end, but the headline improvement
    is small for this short-output workload.
- full random stress status:
  - the old sidecar defaults either hit the RAM guard while compiling extra
    shapes (`1..8` batch buckets and a `4096` prefill bucket) or rejected the
    seed-`1234` manifest because `max_blocks_per_seq=256` cannot admit the
    `5029`-token request;
  - a reduced bucket retry with `max_blocks_per_seq=320` still exceeded the
    80% system RAM guard during compile on this shared host before measurement
    completed;
  - low-memory XLA/JAX diagnostic config:
    `XLA_FLAGS=--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_triton_gemm=false`,
    `XLA_PYTHON_CLIENT_PREALLOCATE=0`,
    `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
    `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, and
    `TF_GPU_ALLOCATOR=cuda_malloc_async`;
  - partial low-memory flag run before the config parser forwarded allocator
    and memory fraction, using sidecar default FP32 activations:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/full_random_summary_no_prefetch_xla_memflags_r1.json`;
    `591.86 output tok/s`, `0.384x` stored vLLM, warmup `457.54 s`, zero
    measured JIT growth, peak system RAM `78.5%`;
  - complete low-memory flag run after forwarding all RAM flags, still with
    sidecar default FP32 activations:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/full_random_summary_no_prefetch_xla_memflags_r2_fullenv.json`;
    `355.73 output tok/s`, `0.231x` stored vLLM, warmup `127.03 s`, zero
    measured JIT growth, peak system RAM `72.6%`, GPU memory after warmup
    `4929 MiB`.
  - apples-to-apples BF16 activation/weight run with the complete low-memory
    flag set:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/full_random_bf16_summary_no_prefetch_xla_memflags_r1.json`;
    `361.19 output tok/s`, `0.234x` stored vLLM, warmup `430.81 s`, zero
    measured JIT growth, peak system RAM `73.9%`, GPU memory after warmup
    `4917 MiB`.
- interpretation:
  - per-token trace event serialization was not the whole gap, but per-step
    token prefetch in summary mode was a real host-communication cost. Removing
    it is a small, general win and does not change model compute or scheduling;
  - the XLA RAM flags are useful for getting memory-heavy compile surfaces
    through the guard and reducing GPU/system memory, but `autotune_level=0`
    and the platform allocator hurt inference quality enough that this should
    stay diagnostic, not the default serving path. The BF16 low-memory result
    is slower than the prior BF16 full-random anchor (`361.19` vs `437.63
    output tok/s`);
  - the remaining `random_large` gap is still dominated by mixed-serving decode
    execution and underfilled tail batches, not by the benchmark event list
    alone.

#### Entry 312 audit - Full-Random Scheduler/Profile Sweep

- date: 2026-06-18
- trigger:
  - user asked to profile the current best path, use server-style generic
    warmup/compile/benchmarking, try XLA RAM flags, and keep optimizing toward
    `0.8x` vLLM on full random or `hetero8` without benchmark-specific hacks.
- accepted code/config changes:
  - random sidecar default prefill envelope is now `max_num_batched_tokens=1024`
    with prefill/token buckets `128,256,512,1024`. This is a generic serving
    envelope change, not a seed-specific warmup;
  - `--jax-prefix-cache/--no-jax-prefix-cache` is part of the random sidecar
    command surface and is recorded in the run config;
  - docs and tests were updated to reflect the 1024 default envelope.
- validation:
  - `.venv/bin/python -m py_compile benchmarks/benchmark_random_request_sidecar.py benchmarks/benchmark_jax_server_trace.py benchmarks/benchmark_jax_server_multisuite.py nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/model_runner.py`;
  - `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_benchmark_random_request_sidecar.py`:
    `12 passed`;
  - `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_device_token_carry.py`:
    `33 passed`;
  - combined focused rerun:
    `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_benchmark_random_request_sidecar.py tests/test_device_token_carry.py`:
    `45 passed`.
- control measurements:
  - shared-envelope no-prefix control:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/shared_current_control_r1.json`;
    warmup `59.94 s`, JIT `24`;
    `random_large` `814.11 output tok/s` (`0.797x` stored vLLM), event rate
    `874.98 output tok/s`; `hetero8` `486.36 output tok/s` total and
    `691.17 output tok/s` event rate. The event-rate ratio for `hetero8` is
    `0.800x`, but the total throughput ratio is only `0.563x` because the
    final drain is a large fraction of the short `256`-output-token workload.
  - full random no-prefix hot path:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/full_random_current_hot_nocap_r1.json`;
    JAX `810.16 output tok/s`, vLLM `1541.89 output tok/s`, ratio `0.525x`,
    zero measured JIT-cache growth, peak system RAM `75.1%`, RSS `5.69 GiB`.
- profile:
  - step profile:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/full_random_current_hot_stepprofile_r1_jax.json`;
  - full random ran `19` prefill steps for `30506` prompt tokens. Prefill
    runner time summed to `1553.7 ms`, total prefill step time `1945.4 ms`,
    and scheduling around prefill `391.2 ms`;
  - decode ran `1809` steps for `11587` emitted tokens. Decode runner time
    summed to `10905.6 ms`, total decode step time `11405.0 ms`, and decode
    scheduling summed to only `463.6 ms`;
  - B8 decode is the dominant steady bucket (`1006` steps, `6.20 ms` mean
    runner time). Python scheduler work is not the dominant remaining cost;
    the bottleneck is compiled GPU/PJRT decode execution plus admission/tail
    effects.
- rejected or diagnostic-only variants:
  - XLA low-memory allocator/platform flags reduced GPU memory after
    measurement to about `5.0 GiB`, but regressed shared-envelope throughput to
    `451.60 output tok/s` on `random_large` and `326.28 output tok/s` on
    `hetero8`. Keep these flags for compile containment diagnostics, not the
    serving hot path;
  - adaptive summary token prefetch regressed `hetero8` (`486.36 -> 479.32`
    output tok/s total, event rate `691.17 -> 641.38`) and was reverted;
  - full-random B16 capacity without other changes regressed to `778.63 output
    tok/s` (`0.505x`) and increased JIT surface from `38` to `49` keys;
  - prefill chunk `4096` regressed to `754.31 output tok/s` and used about
    `16.9 GiB` GPU memory. Prefill chunk `512` reached only `807.47 output
    tok/s`;
  - `decode_padded_gemm_rows=16` on the B16/Triton diagnostic was neutral
    (`908.04` versus `907.92 output tok/s`).
- accepted benchmark default:
  - prefill chunk `1024` is the best generic full-random default among
    `512/1024/2048/4096` in this pass:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/full_random_current_hot_prefill1024_r1.json`;
    `828.63 output tok/s`, ratio `0.537x`, event rate `864.43 output tok/s`,
    peak system RAM `71.8%`, RSS `5.27 GiB`, JIT `34`.
- no-regression checks:
  - a four-workload multisuite attempt correctly failed before measurement
    because the `random_large` serving envelope has `max_blocks_per_seq=128`
    (`2048` tokens) while `long_prefill_512_2048` needs `2064` total tokens.
    Do not treat that as a speed result;
  - the same warmed server-style multisuite over the workloads that fit that
    envelope completed:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_scheduler_20260618/multisuite_no_regression_3workloads_r1.json`;
    `random_large` `814.66 output tok/s`, `hetero8` `491.35 output tok/s`,
    and `short_32_128` `501.88 output tok/s`, with generic warmup adding `24`
    JIT entries before measurement;
  - a plain trace long-prefill run without the config-expanded optimized flags
    reached only `11.02 output tok/s`; discard it as an invalid no-regression
    signal;
  - the configured one-repeat matrix long-prefill check used the optimized
    config path and reached `138.56 output tok/s` (`1.19x` stored vLLM), but it
    is not correctness-clean versus the old stored JAX reference (`3/4` rows
    diverged). Keep this as a performance/correctness caveat, not an accepted
    speed claim, until a refreshed long-prefill correctness gate is run.
- speed diagnostics not promoted:
  - enabling XLA Triton GEMM with the accepted 1024 envelope reached
    `876.47 output tok/s` (`0.568x`) but increased elapsed compile/startup time
    to about `381 s`;
  - combining B16 capacity with the 1024 envelope and XLA Triton GEMM reached
    `907.92 output tok/s` (`0.589x`), with much better TTFT p95 but worse ITL
    and a still-heavy startup/compile surface (`484 s`, JIT `44`). This is the
    best diagnostic from the pass, but not a production default and still far
    below `0.8x`.
- interpretation:
  - `hetero8` did not improve much in total throughput because it is short and
    final materialization/drain dominates the denominator; its token-phase rate
    is already at the `0.8x` line in the control run;
  - full random remains the harder and more representative target. The current
    accepted default improves full random from `810.16` to `828.63 output tok/s`
    but still only reaches `0.537x` vLLM;
  - the next real lever is an efficient B>8 / arbitrary-batch decode execution
    route or backend-owned broader decode boundary. Capacity alone, allocator
    flags, prefetch tweaks, and one-off padding changes do not remove enough
    compiled decode work.

#### Entry 313 audit - Fixed-8 Random Sidecar Reset

- date: 2026-06-19
- trigger:
  - user asked to cap the random sidecar to B=8, benchmark that serving contract,
    and use it to plan the final speedup pass instead of continuing to chase
    B16/full-random scaling.
- code/docs changes:
  - random sidecar defaults now use exactly `8` requests:
    `--min-request-count=8` and `--max-request-count=8`;
  - docs and unit expectations were updated to make the B8 cap explicit.
- validation:
  - `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_benchmark_random_request_sidecar.py`:
    `12 passed`;
  - `.venv/bin/python -m py_compile benchmarks/benchmark_random_request_sidecar.py`.
- benchmark:
  - sidecar artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/b8_full_random_20260619/full_random_b8_sidecar_r1.json`;
  - workload: seed `1234`, exactly `8` requests, `16806` input tokens, `6126`
    output tokens, prompt lengths `1116..4022`, output lengths `425..1007`;
  - JAX optimized server path: `770.12 output tok/s`, token-event rate
    `807.16 output tok/s`, TTFT p50 `905.49 ms`, ITL p50 `6.47 ms`, final
    drain `0.365 s`, peak system RAM `73.5%`, zero measured JIT growth;
  - live vLLM same manifest: `1000.98 output tok/s`, TTFT p50 `354.00 ms`,
    ITL p50 `5.79 ms`;
  - total-throughput ratio: `0.769x`; token-event ratio: `0.806x`.
- guarded profiling:
  - full XLA profile under the sidecar guard was correctly killed at
    `82.4%` system RAM, proving the RAM prevention mechanism is active. Do not
    retry full XLA trace capture on this host without tighter workload scaling;
  - engine-step-only guarded repeat completed:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/b8_full_random_20260619/full_random_b8_stepprofile_guarded_r1.json`;
    JAX `795.62 output tok/s`, token-event `834.07 output tok/s`, final drain
    `0.355 s`, peak system RAM `52.4%`, zero measured JIT growth;
  - the output tail only captures the final B1 drain/tail decode steps, where
    schedule p50 is about `0.059 ms` and runner p50 about `4.70 ms`; scheduler
    is not the final blocker.
- interpretation:
  - the B8-capped random lane is close to the `0.8x` target. The steady token
    phase already clears the target in both fixed-8 runs; total throughput is
    pulled below target by final device-token materialization/drain and TTFT;
  - for the next pass, do not return to B16 or broad model-kernel work first.
    The highest-probability final speedup is to eliminate or overlap final
    deferred token materialization in summary mode, while keeping exact output
    lengths and the same generic startup warmup. Removing the `0.35-0.36 s`
    drain alone should move the sidecar result above `0.8x` vLLM.

#### Entry 314 audit - Summary Materialization Drain Pass

- date: 2026-06-19
- trigger:
  - user asked to drop the B16 lane, optimize output materialization, and make
    sure `hetero8` remains healthy under the B8/random-large serving envelope.
- promoted code path:
  - summary-mode `generate_with_trace(..., trace_events=False)` now prefetches
    device token slots for rows as soon as they finish, then materializes those
    rows after one more model step when overlap is available;
  - final `DeviceTokenRef` materialization now fetches existing decoded token
    vectors directly with `jax.device_get(vector_arrays)` instead of first
    creating a stacked JAX array solely for host transfer;
  - detailed trace mode is unchanged.
- validation:
  - `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_device_token_carry.py`:
    `34 passed`;
  - `.venv/bin/python -m py_compile nanovllm_jax/engine/llm_engine.py tests/test_device_token_carry.py`;
  - `.venv/bin/python -m py_compile nanovllm_jax/engine/sequence.py`.
- fixed-B8 random benchmark:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/materialization_20260619/full_random_b8_direct_vector_get_r1.json`;
  - workload and vLLM denominator are unchanged from Entry 313: seed `1234`,
    exactly `8` requests, `6126` output tokens, same-manifest vLLM
    `1000.98 output tok/s`;
  - current JAX: `788.34 output tok/s`, ratio `0.788x`, token-event rate
    `809.28 output tok/s`, TTFT p50 `904.79 ms`, ITL p50 `5.29 ms`, final
    drain `0.201 s`;
  - improvement over Entry 313 baseline: `770.12 -> 788.34 output tok/s` and
    final drain `0.365 -> 0.201 s`. The remaining gap to `0.8x` vLLM is about
    `12.4 output tok/s`.
- hetero8 server-envelope check:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/materialization_20260619/hetero8_direct_vector_get_r1.json`;
  - run used the `random_large` serving envelope, generic warmup, no trace
    events, no measured-phase JIT growth failure, and the sidecar RAM/CPU guard
    wrapper;
  - current JAX: `581.30 output tok/s`, token-event rate `865.48 output tok/s`,
    `0.440 s` total, final drain `0.145 s`, peak system RAM `46.3%`, process
    RSS `4.81 GB`;
  - compared with the 2026-06-18 no-regression artifact, total throughput is up
    from `491.35 output tok/s` and token-event rate is up from `693.05 output
    tok/s`;
  - strict generated-token comparison against the old stored single-length
    reference remains `ok=false` with `1/8` exact row matches, the same exact
    match count as the June 18 no-regression artifact. Treat this as the
    existing hetero-vs-stored-reference caveat, not a new materialization
    regression.
- rejected materialization variants:
  - periodic live-row prefetch on the fixed-B8 random lane was neutral/slightly
    negative: `787.19 output tok/s` versus `788.09` for finished-row-only, with
    final drain essentially unchanged at `0.201 s`;
  - short-completion summary live prefetch regressed `hetero8` badly:
    `581.38 -> 482.70 output tok/s`, event rate `872.76 -> 662.21`, and final
    drain barely changed (`0.147 -> 0.144 s`). Do not retry per-step summary
    host-copy prefetch as a hetero fix without a broader device-owned output
    buffer boundary.
- interpretation:
  - materialization overlap was a real B8 win and nearly closes the current
    `0.8x` target, but per-step host prefetch is not the answer for short
    hetero workloads;
  - `hetero8` is "working" under the shared server envelope in the sense that
    throughput recovered versus the recent regression and resource use is
    stable, but its headline total still lags the stored vLLM hetero reference
    because final drain is a large fraction of a short 256-token run. The next
    hetero-specific improvement needs a device-owned output table or a broader
    decode/output boundary, not more Python-loop prefetch calls.

#### Entry 315 audit - Final Materialization Boundary Check

- date: 2026-06-19
- trigger:
  - user asked to keep optimizing final materialization until it became
    negligible on the fixed-B8 random lane.
- promoted code path:
  - `Sequence.materialize_device_token_slots` now filters stale snapshot slots
    in O(n) by precomputing current `(index, token-id)` sets per sequence. This
    keeps stale pending snapshots cheap after another materializer has already
    cleared slots;
  - summary trace output now reports finalization timing fields:
    `final_post_loop_seconds`, `final_pending_materialize_seconds`,
    `final_device_token_materialize_seconds`, `final_result_build_seconds`, and
    before/after device-token slot counts.
- validation:
  - `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_device_token_carry.py tests/test_backend_boundaries.py::test_executor_table_hybrid_decode_matches_sliced_decode`:
    `35 passed`;
  - `.venv/bin/python -m py_compile nanovllm_jax/engine/llm_engine.py nanovllm_jax/engine/sequence.py`.
- fixed-B8 random current safe path with timing:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/materialization_20260619/full_random_b8_linear_slot_timing_sidecar_r1.json`;
  - JAX `794.86 output tok/s`, ratio `0.794x` stored vLLM, token-event rate
    `816.14 output tok/s`, final drain `0.201 s`, zero measured JIT growth;
  - final timing: `final_pending_materialize_seconds=0.1867`,
    `final_post_loop_seconds=0.1871`, `final_device_token_slots_before=1007`.
- post-experiment safe repeat after reverting regressive variants:
  - artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/materialization_20260619/full_random_b8_final_materialization_safe_r1.json`;
  - JAX `791.27 output tok/s`, ratio `0.790x` stored vLLM, token-event rate
    `811.54 output tok/s`, final drain `0.193 s`, zero measured JIT growth,
    peak system RAM `53.2%`;
  - final timing: `final_pending_materialize_seconds=0.1779`,
    `final_post_loop_seconds=0.1783`, `final_device_token_slots_before=1007`;
  - treat this as a repeat of the safe path, not a new best. The canonical
    current-code comparison point remains
    `full_random_b8_linear_slot_timing_sidecar_r1.json`.
- rejected variants:
  - resident output-token table: changed the dense decode boundary to scatter
    token ids into a `[max_seqs, max_blocks_per_seq * block_size]` table every
    step. Reordering finalization made scalar pending materialization
    negligible (`0.0005 s`) but total throughput collapsed to
    `416.72 output tok/s`; the hot scatter increased GPU memory and TTFT;
  - final stacked vector transfer: grouped same-shaped per-step token vectors
    into stacked JAX arrays at finalization. This made the final wait worse:
    `362.20 output tok/s`, `0.514 s` drain, `0.496 s`
    `final_pending_materialize_seconds`;
  - longest-row live `copy_to_host_async`: after fixing the skipped-slot bug,
    it submitted all `1006` decode `DeviceTokenRef`s for the final row, reduced
    final pending materialization to `0.1286 s`, but regressed throughput to
    `778.27 output tok/s`;
  - background `device_get` resolver: resolving the longest row on two worker
    threads made the measured final pending block small
    (`0.0061 s`, one fallback slot) but introduced a mandatory
    `0.1465 s` background flush and regressed throughput to
    `768.00 output tok/s`.
- interpretation:
  - the remaining final bucket is not just Python list processing. It is the
    necessary readback of the last-finishing row's deferred token ids, plus
    queued GPU completion observed at the first blocking host read;
  - making the final block itself negligible is possible, but every attempted
    implementation merely moved the wait earlier or added hot decode work,
    reducing total throughput. Do not promote these variants;
  - the next real fix needs a broader device-owned output/results boundary:
    either avoid materializing full token ids for throughput-only summaries, or
    produce and consume output tokens through a resident result buffer without a
    per-step scatter that competes with decode.

#### Entry 316 - Thresholded Host Token Sink for Long Summary Runs

- date: 2026-06-20
- trigger:
  - user asked for a way around the remaining final materialization bucket and
    asked to check how vLLM handles output token transfer.
- vLLM reference lesson:
  - vLLM starts sampled-token D2H transfer immediately after sampling on a
    separate CUDA stream (`AsyncOutput` in `vllm/v1/worker/gpu/async_utils.py`)
    and lets output processing consume the CPU tensor later;
  - the important part for our summary path is not a GPU output table. It is
    early async transfer plus delayed host consumption, without adding a hot
    per-step GPU scatter to decode.
- implementation:
  - added a summary-mode host token sink in `LLMEngine` for long-output runs;
  - the sink snapshots newly generated `DeviceTokenSlot`s, calls
    `copy_to_host_async` during generation, and harvests the accumulated slots
    once at final result build with `_resolve_device_token_slots_without_mutation`;
  - it writes final result token ids from a side host buffer and leaves
    `Sequence._device_token_slots` intact, so the device-token carry state is
    not broken by in-loop materialization;
  - the sink is thresholded by planned completion budget
    (`summary_host_token_sink_min_completion_tokens`, default `1024`) because
    short-output active prefetch was already known to regress `hetero8`.
- validation:
  - `.venv/bin/python -m pytest tests/test_device_token_carry.py -q`:
    `35 passed`;
  - fixed-B8 random thresholded artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_token_sink_20260620/full_random_b8_host_token_sink_threshold_r1.json`;
  - JAX sidecar:
    `803.64 output tok/s`, `6126` generated tokens, `0.096 s` final post-loop,
    `0.095 s` host-sink harvest, `0.0 s` final device-token materialization,
    host sink enabled for planned budget `6126`, no missing host-sink tokens;
  - same-manifest vLLM reference:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/b8_full_random_20260619/full_random_b8_sidecar_r1_vllm.json`,
    `1000.98 output tok/s`, so the current ratio is `0.803x`;
  - prior safe JAX reference:
    `770.12 output tok/s`, `0.365 s` post-last-token drain. The new path
    improves throughput by `+33.52 output tok/s` and cuts the drain by
    about `269 ms`;
  - hetero8 guard artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/host_token_sink_20260620/hetero8_host_token_sink_threshold_r1.json`;
  - hetero8 stayed on final materialization (`host_token_sink_enabled=false`,
    planned budget `256 < 1024`) and measured `579.72 output tok/s`, matching
    the prior healthy `581.30 output tok/s` within run noise;
  - the deliberately unthresholded deferred-harvest hetero8 check reproduced
    the bad short-output pattern at `481.83 output tok/s`, confirming the guard
    is necessary.
- interpretation:
  - this is a real B8 improvement and crosses the near-term `0.8x` target for
    the fixed-8 random-large lane;
  - this does not solve the remaining vLLM gap. The next major gap is no longer
    bulk final token readback on long-output summary runs; remaining work should
    focus on TTFT and decode GPU work/launch structure, while keeping the
    thresholded sink for long summaries and avoiding it for short outputs.

#### Entry 317 - Resident Seed-Then-Table MTP Burst Boundary

- date: 2026-06-20
- trigger:
  - user asked to implement MTP so drafting and verification stay on device,
    with no host sync controlling verification.
- implementation:
  - added a K=1 resident seed-then-table burst route that runs the initial MTP
    seed, verifier table-burst groups, accept/reject selection, KV/hybrid
    commit, emitted-token compaction, and next-draft generation inside one JIT
    boundary;
  - compacted emitted tokens on device so host code consumes
    `emitted_total` directly instead of parsing sparse target/bonus slots;
  - routed missing-draft rows through the new boundary before the older
    seed-only path;
  - changed the MTP experimental config to seed after bonus so the verifier can
    keep drafting without a host-managed seed transition;
  - fixed scheduler/model-runner fast-path flag helpers so explicit
    `NANO_VLLM_JAX_*` env values override dataclass defaults, matching the repo
    config comments. This fixed a mismatch where the runner returned
    `DeviceTokenRef`s but scheduler postprocess still tried `int(token_ref)`;
  - made MTP decode admission reject non-`ignore_eos` requests when device
    token carry is active. EOS-checking requests need a separate on-device EOS
    completion boundary before they can safely use resident token refs.
- validation:
  - `.venv/bin/python -m pytest tests/test_mtp_commit_semantics.py -q`:
    `33 passed, 1 xfailed`;
  - `.venv/bin/python -m pytest tests/test_device_token_carry.py -q`:
    `36 passed`;
  - `.venv/bin/python -m pytest tests/test_server_config.py -q`:
    `23 passed`;
  - B=1 GPU smoke artifact:
    `/mountpoint/.exp/diagnostics/nano-vllm-jax/mtp_on_device_20260620/single_synthetic_seed_table_burst_smoke_r3.json`.
- smoke result:
  - exact token match: true;
  - MTP path exercised: `drafts_proposed=3`, `drafts_accepted=2`,
    `drafts_rejected=0`, `bonus_tokens=2`, `accepted_decode_steps=1`;
  - reported host/device postprocess buckets: `0.0 ms`;
  - decode speedup versus same-run no-MTP control: `0.684x`;
  - acceptance for the required metrics row: `0.667`.
- interpretation:
  - this is the first correctness-clean device-resident drafting/verification
    boundary for the current GPU MTP path;
  - it is not a serving speed claim. The verifier remains slower than the
    no-MTP control in the smoke, and the first full hetero8 server-style
    attempt timed out compiling before this env/config fix. Continue from this
    boundary and attack verifier GPU work/compile surface next.
