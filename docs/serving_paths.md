# Serving Paths

The repo has one promoted server path and several explicitly separated
diagnostic paths. New users should not need to reconstruct the right command
from benchmark JSONs or old optimization notes.

## Promoted Path

Use:

```bash
python server.py --config configs/server/gpu_optimal.yaml
```

The root `server_config.yaml` mirrors this file, so `python server.py` is the
same promoted non-MTP path. It uses:

- Qwen/Qwen3.5-0.8B with BF16 model compute and BF16 weights;
- paged KV cache with chunked/ragged packed prefill;
- prefix caching for reusable full-block prompt prefixes;
- resident/static decode metadata and device token carry;
- FlashInfer paged full-attention decode;
- Triton packed full-attention prefill;
- Triton FLA padded GDN prefill;
- reference BF16-QKV packed GDN decode;
- Triton greedy LM-head top-1;
- local CUDA probe kernels disabled.

MTP is off in this config. Speed claims and new serving work should use this
path as the no-MTP control unless a newer config is explicitly promoted.

The config can serve the full bucket set, but one-command startup warms only
the configured `engine.startup_warmup_*` profile by default. The current
promoted profile warms prefill token buckets `64,128`, batch buckets `1,4`, and
decode block-table buckets `128,320`. Larger serving buckets remain available
and compile on first use unless the warmup profile is expanded.

Prefix caching is enabled by default with `engine.prefix_cache: true` and can be
disabled with `--no-prefix-cache` or `NANO_VLLM_JAX_PREFIX_CACHE=0`. For
Qwen3.5 hybrid GDN layers, the cache only skips a full-block prefix after both
the paged full-attention KV blocks and the matching GDN hybrid state for that
exact prefix hash have been recorded.

## Small Smoke Path

Use:

```bash
python server.py --config configs/server/gpu_minimal_pure_jax.yaml
```

This keeps shapes small and uses reference JAX kernels. It is for installation,
config, and endpoint smoke tests, not performance claims.

## Local Chat UI

After starting any server config, run:

```bash
python tools/chat_ui_server.py --port 6789 --backend-url http://127.0.0.1:6791
```

Open `http://127.0.0.1:6789`. The UI proxies to `/v1/generate_trace` and shows
TTFT plus output tokens/sec after each message completes. It sends structured
chat `messages`; the server applies the loaded tokenizer chat template with an
assistant generation prompt. By default it uses the EOS-compatible trace path.
Requests that set `ignore_eos=true` use the faster no-readback trace path and
run to `max_tokens`.

For a one-line local launch, run:

```bash
tools/start_local_chat.sh
```

By default this starts the promoted model server on `127.0.0.1:6791`, the chat
UI on `127.0.0.1:6789`, and writes logs under `/mountpoint/.exp/run_logs`.
Override `NANO_VLLM_JAX_SERVER_CONFIG`, `NANO_VLLM_JAX_PORT`,
`NANO_VLLM_JAX_CHAT_UI_PORT`, or `NANO_VLLM_JAX_SKIP_COMPILE_STARTUP=1` when
needed.

## Experimental MTP Path

Use:

```bash
python server.py --config configs/server/mtp_experimental.yaml
```

This path keeps target-model verification enabled and unverified append
disabled. It is a diagnostic path only until it beats the same promoted no-MTP
config on the same benchmark envelope.

## Benchmark And Probe Paths

Benchmark JSON files under `benchmarks/configs/` define reproducible benchmark
contracts. They are not the primary user-facing server setup surface.

Local CUDA probe code, old unverified MTP routes, raw GDN decode kernel probes,
and route-specific warmup experiments must stay opt-in diagnostics. Store full
artifacts under `/mountpoint/.exp/diagnostics` or `/mountpoint/.exp/profiles`;
commit only summaries, promoted configs, docs, and tests.
