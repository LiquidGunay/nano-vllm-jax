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
