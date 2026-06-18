# Benchmark Configs

These JSON files are benchmark contracts for `benchmarks/run_gpu_matrix.py` and
related harnesses. They are not the default way to start a server.

Use server YAML files under `configs/server/` for user-facing server setup:

- `configs/server/gpu_optimal.yaml` for the promoted non-MTP path;
- `configs/server/gpu_minimal_pure_jax.yaml` for small smoke checks;
- `configs/server/mtp_experimental.yaml` for exact MTP diagnostics.

Only promoted benchmark configs should be committed here. One-off kernel probes,
route-specific warmup experiments, random sidecar outputs, prompt manifests, and
full benchmark artifacts should stay under `/mountpoint/.exp/diagnostics` or be
kept as ignored local files.
