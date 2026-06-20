# Server Configs

Use these YAML files with `python server.py --config <path>`.

- `gpu_optimal.yaml`: promoted non-MTP CUDA serving path. This is the default
  new-user path and mirrors the root `server_config.yaml`. Prefix caching is
  enabled by default; set `engine.prefix_cache: false`, `--no-prefix-cache`, or
  `NANO_VLLM_JAX_PREFIX_CACHE=0` for no-cache baselines. The serving bucket set
  remains broad, while `engine.startup_warmup_*` bounds the one-command startup
  compile profile to common shapes.
- `gpu_minimal_pure_jax.yaml`: small pure-JAX CUDA smoke path for installation
  checks and low compile cost.
- `mtp_experimental.yaml`: exact target-verified true-K MTP diagnostic path
  (`k_decode`, currently K=2) with strict GDN no-fallback settings. It is
  intentionally separate from the default path and should not be used for speed
  claims until MTP beats the same non-MTP config.

Benchmark JSON files under `benchmarks/configs/` are benchmark contracts, not
server setup recipes. Full result/profile artifacts should stay under
`/mountpoint/.exp/diagnostics` or `/mountpoint/.exp/profiles`, not in git.
