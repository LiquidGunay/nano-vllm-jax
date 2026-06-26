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

Explicit structural residual:
- The cleanup plan's ideal tree includes a larger split of `model.py` into
  `model.py`, `attention.py`, and `gdn.py`. This branch keeps the high-level
  layer helpers in `model.py` after removing experimental/MTP/compatibility
  paths. Creating non-facade split modules remains a larger pedagogical
  refactor, not a fallback or public-policy leak.

Validation completed under `tests/ram_guard.py`:
- `pytest -q` was attempted as a single run; RAM guard stopped it before server
  memory pressure could crash the machine.
- Split guarded pytest runs covered all 186 collected tests:
  118 targeted kernel/cache/GDN tests, 27 lightweight API/config/service tests,
  28 device-token tests, 2 E2E parity tests, 6 layer parity tests, and 5
  real-weight parity tests.
- `python -m compileall -q server.py nanovllm_jax tests`.

Second audit follow-up validation under `tests/ram_guard.py`:
- `pytest -q tests/test_fastpath_config.py tests/test_service.py
  tests/test_server_config.py tests/test_public_imports.py`: 14 passed.
- `pytest -q tests/test_gdn_post_conv_prefill_reference.py
  tests/test_gdn_segmented_reference.py tests/test_gdn_packed_decode_reference.py
  tests/test_gdn_packed_prefix_state.py tests/test_kv_cache.py
  tests/test_nhd_kv_cache.py tests/test_paged_attention_abi.py
  tests/test_flashinfer_ffi.py tests/test_decode_reductions.py
  tests/test_lm_head_helpers.py`: 128 passed.
- `pytest --collect-only -q`: 187 tests collected.
- After the `engine.py` move, `pytest -q tests/test_device_token_carry.py`: 28
  passed.
- After removing YAML runtime policy and JIT-cache exposure, reran:
  `pytest -q tests/test_fastpath_config.py tests/test_service.py
  tests/test_server_config.py tests/test_public_imports.py` with 14 passed,
  `pytest -q tests/test_device_token_carry.py` with 28 passed, and
  `pytest --collect-only -q` with 187 collected.
- `python -m compileall -q server.py nanovllm_jax tests`.
