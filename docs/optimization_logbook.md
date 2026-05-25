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
