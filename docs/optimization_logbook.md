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
