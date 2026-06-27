# Cleanup Audit Resolution

This note tracks independent cleanup-plan reviews from subagent audits.

Resolved in the structural cleanup:
- Removed benchmark/result/config/tool/archive directories from cleaned main.
- Kept `server.yaml` as the only committed serving YAML.
- Added `fastpath.py` as the promoted CUDA/JAX policy manifest.
- Added `service.py` with queued cross-request engine stepping.
- Added RAM-guarded validation instructions and helper.
- Removed the old backend-selection surface and made operation policy direct
  through `ServingOps` plus config-projected fast-path fields.
- Removed kernel registry/fallback status helpers and local CUDA diagnostic
  probe code.
- Removed public `backend` serving config and stale environment-driven
  operation switches.
- Removed trace/profiling engine paths.
- Removed facade-only split modules and the runner compatibility subclass.
- Removed public sampling options that the hot path rejected.
- Cleaned stale `.gitignore` patterns and stale environment-switch wording.
- Removed stale top-level policy/config exports. The public package API is now
  `LLM`, `EngineConfig`, and `SamplingParams`.
- Tightened `LLMEngine` construction so Python callers can pass only public
  workload/capacity config; kernel policy is projected internally from
  `fastpath.py`.
- Removed the no-op `fastpath` YAML key from `server.yaml`.
- Removed the YAML runtime/XLA section so `server.yaml` carries only transport
  and workload/capacity settings.
- Moved request lifecycle from `llm.py` to the planned `engine.py` module.
- Moved persistent JAX compile-cache configuration out of `executor.py` and
  into `server.py` runtime setup.
- Removed server API and warmup-summary exposure of executor JIT-cache internals.
- Made low-level Triton GDN fallback guards fail fast instead of silently
  returning to a reference route.
- Updated stale docs and tests for `cache.py`, guarded test commands, and
  fail-fast fallback expectations.
- Split the old model monolith into non-facade modules: `model.py` keeps
  parameters, the layer loop, and forward entrypoints; `projection.py`,
  `attention.py`, `gdn.py`, and `lm_head.py` own the corresponding math.
- Pruned stale operation variants so the runtime keeps one reference path and
  one promoted fast path for each accepted speedup.

Validation completed under `tests/ram_guard.py`:
- `python -m compileall -q server.py nanovllm_jax tests`.
- `python -m ruff check server.py nanovllm_jax tests`.
- `pytest --collect-only -q`: 167 tests collected after removing obsolete
  runtime variants.
- `pytest -q tests/test_fastpath_config.py tests/test_public_imports.py
  tests/test_service.py tests/test_server_config.py
  tests/test_causal_conv1d_update.py tests/test_decode_reductions.py
  tests/test_paged_attention_abi.py tests/test_nhd_kv_cache.py`: 34 passed.
- `pytest -q tests/test_device_token_carry.py tests/test_kv_cache.py
  tests/test_flashinfer_ffi.py tests/test_lm_head_helpers.py`: 55 passed.
- `pytest -q tests/test_gdn_packed_decode_reference.py
  tests/test_gdn_packed_prefix_state.py
  tests/test_gdn_post_conv_prefill_reference.py`: 26 passed.
- `pytest -q tests/test_gdn_segmented_reference.py`: 39 passed.
- `pytest -q tests/test_layer_parity.py`: 6 passed.
- `pytest -q tests/test_real_weight_layerwise_parity.py`: 5 passed.
- `pytest -q tests/test_e2e_parity.py::test_logits_parity`: 1 passed.
- `pytest -q tests/test_e2e_parity.py::test_generation_parity`: 1 passed.
- Combined heavyweight parity runs were intentionally split because the RAM
  guard terminated a larger batch at the configured RSS limit.
