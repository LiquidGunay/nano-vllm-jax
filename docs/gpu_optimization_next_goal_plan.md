# GPU Optimization Next Goal Plan

This file is the source-of-truth planning artifact for the next goal. It captures
the proposed cleanup, benchmark discipline, and kernel roadmap before further
optimization work starts.

## Resolved Decisions

1. FlashInfer and `jax-tvm-ffi` may be used as optional dependencies for opt-in
   kernel backends. They must not become mandatory for the pure-JAX fallback.
2. `gpu_paged_default` means the fastest accepted non-MTP serving configuration,
   not the most conservative historical path. Experimental FFI/custom kernels
   are still excluded until they pass the gates and are promoted.
3. This file may be committed and revised as the next-goal contract evolves.
4. The near-term implementation order should favor easier kernel wins first:
   paged KV append, paged decode attention, and integration scaffolding. GDN
   remains critical, but the next GDN implementation should use the Qwen 3 Next
   vLLM/Flash Linear Attention kernels as references rather than repeating
   source-level JAX or compile-heavy Pallas attempts.
5. If kernel integration is difficult, add a minimal optional kernel-backend
   strategy and ABI validation path before forcing it through the repo's final
   paging layout.
6. Phase 0 should create the baseline docs and benchmark config JSONs in the
   same commit. Treat docs plus config as the first deliverable; the benchmark
   matrix runner can follow in the next commit.
7. The matrix runner should use stored local vLLM/JAX reference artifacts when
   available. If no stored local reference exists for a requested comparison,
   it should run live where possible and store the resulting artifact: live
   vLLM for missing vLLM references and live `gpu_paged_default` for missing
   JAX default references. `--require-stored-references` remains an explicit
   opt-in gate when live fallback is not acceptable.

## Context

The current validated GPU baseline is the JAX paged serving path for
`Qwen/Qwen3.5-0.8B`. The optimization log shows the accepted Entry 045 baseline
at `367.80 tok/s`, `TTFT p50 289.98 ms`, and `ITL p50 13.14 ms`, compared with
the tracked vLLM async baseline of `864.18 tok/s`. JAX is currently about
`0.426x` vLLM on the tracked hetero8 workload.

The target model is structurally unusual: `Qwen3.5-0.8B` has 24 layers arranged
as `6 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`, so most
layers are Gated DeltaNet / linear-attention layers rather than ordinary
full-attention layers. The model card lists GDN with 16 QK/V heads, head dim
128, full/gated attention with 8 Q heads, 2 KV heads, head dim 256, tied LM
output, and MTP training.

The current lesson from the logbook is that source-level JAX rewrites are mostly
exhausted. Several changes preserved exact generated-token correctness but
regressed integrated performance. The next serious wins should come from a
cleaner benchmark/docs surface and backend-owned kernels, not more ad hoc JAX
reshaping.

## Phase 0 - Repository And Docs Cleanup

### Goal

Make the current state unambiguous before adding new kernels. The agent should
not start by changing kernel code.

### Tasks

Create these files:

- ~~`docs/current_gpu_baseline.md`~~
- ~~`docs/rejected_optimization_index.md`~~
- ~~`docs/kernel_roadmap.md`~~
- ~~`benchmarks/configs/gpu_paged_default.json`~~
- ~~`benchmarks/configs/gpu_paged_fast_optin.json`~~
- ~~`benchmarks/configs/gpu_mtp_diagnostics.json`~~

### `docs/current_gpu_baseline.md`

Include:

```text
Current accepted default:
- model: Qwen/Qwen3.5-0.8B
- hardware: exact GPU name from benchmark artifact
- dtype contract: BF16 checkpoint weights, FP32 activation math
- platform: JAX CUDA
- baseline entry: Entry 045
- benchmark: hetero8 input lengths [64,128,192,256,320,384,448,512], output length 32
- metrics:
  - throughput: 367.80 tok/s
  - TTFT p50: 289.98 ms
  - ITL p50: 13.14 ms
  - ITL p95: 13.59 ms
- vLLM comparison:
  - vLLM async: 864.18 tok/s
  - JAX/vLLM: 0.426x
```

Split configs into two profiles:

```text
gpu_paged_default:
- fastest accepted default flags
- no experimental compact-prefill flags unless already default
- no MTP
- no FFI/custom kernels

gpu_paged_fast_optin:
- accepted opt-in compact prefill/layout flags
- still no MTP
- still correctness-gated
```

### `docs/rejected_optimization_index.md`

Summarize every rejected experiment in a compact table:

```text
| Entry | Experiment | Correct? | Perf result | Decision | Lesson |
|---|---|---:|---:|---|---|
```

Must include at least:

```text
- width-1 GDN scan bypass
- decode slot-mapping reuse
- Pallas paged decode attention
- Pallas LM-head argmax
- GDN chunk-size 16
- head-major paged attention prefill
- flat KV cache allocation
- packed full-attention K/V projection
- skip-unused hidden return norm
- static chunk-major GDN prefill
- default MTP1 serving
```

For MTP, explicitly record that the log rejects default MTP1 serving because the
run was unusably slower and output tokens remained exact only because all
speculative drafts were rejected.

### `docs/kernel_roadmap.md`

Use the kernel priority list from Phase 2 below. Each kernel entry must include:

```text
- motivation
- proposed source/reference implementation
- JAX-facing ABI
- fallback path
- correctness gate
- performance gate
- do-not-merge conditions
```

## Phase 1 - Benchmark And Correctness Discipline

### Goal

Before implementing kernels, make benchmark comparisons reproducible and hard to
game.

### Tasks

Add a benchmark matrix runner:

- ~~`benchmarks/run_gpu_matrix.py`~~

It should run:

```text
1. gpu_paged_default
2. gpu_paged_fast_optin
3. gpu_mtp_diagnostics
```

Across at least:

```text
- hetero8: [64,128,192,256,320,384,448,512] x 32 output
- short_32_128: [32,64,96,128] x 32 output
- long_prefill_512_2048: [512,1024,1536,2048] x 16 output
- decode_heavy_128x128: [128] x 128 output
```

Output one summary file:

```text
results/gpu_matrix_<timestamp>.json
```

Required metrics:

```text
- total tok/s
- TTFT p50/p95
- ITL p50/p95
- exact generated-token match vs baseline
- PjRt Execute total/count
- command_buffer::execute total/count
- command_buffer::update total/count
- forward_step_token_ids_jit total/count
- first forward_step_token_ids_jit
- gather total
- transpose total
- MemcpyD2D total
- tolist / np.asarray sync attribution
```

### Acceptance Rule For Any Performance PR

A PR can only claim a speedup if:

```text
- exact generated-token match holds
- the same benchmark config is used
- at least 2 repeat runs are reported
- median result improves
- no major TTFT/ITL regression is hidden by throughput
- profile bucket movement is explained
```

This matters because the log already shows experiments where a named trace bucket
improved but integrated throughput regressed. For example, smaller GDN chunking
improved one reduce bucket but worsened first prefill, GEMM/module execution, and
end-to-end throughput.

### Status - 2026-05-26

- Matrix configs and `benchmarks/run_gpu_matrix.py` are in place. The runner
  uses stored local references when available. When no workload-specific local
  artifact exists, it can run live vLLM for the vLLM reference and can use the
  live `gpu_paged_default` artifact as the JAX default reference for subsequent
  comparisons.
- The first `long_prefill_512_2048` slice had no stored local vLLM reference, so
  it ran live vLLM and stored
  `results/gpu_matrix_runs/20260526_104818/references/vllm_long_prefill_512_2048.json`.
- One-repeat results: vLLM async `116.37 tok/s`, JAX `gpu_paged_default`
  `78.02 tok/s` (`0.670x` vLLM), and JAX `gpu_paged_fast_optin`
  `78.27 tok/s` (`0.673x` vLLM). Fast opt-in matched the live JAX default
  exactly for all four rows; default was the baseline capture for this slice.
- This is not a speed-claim artifact because it is one repeat and the default
  row is not checked against a same-workload token reference. It is useful as
  current target evidence: long-prefill is closer to vLLM than hetero8, but
  still below the `0.75x` next-goal target.
- The JAX gap remains visible in both TTFT and ITL. The fast-optin profile shows
  first `forward_step_token_ids_jit=237.75 ms`,
  `PjRtCApiLoadedExecutable::Execute=290.66 ms / 140`,
  `command_buffer::execute=228.13 ms / 1936`, and `np.asarray(jax.Array)=
  429.63 ms / 16`.
- A two-repeat `hetero8,long_prefill_512_2048` matrix attempt could not produce
  benchmark evidence because the current session had no visible NVIDIA device
  nodes and `nvidia-smi` could not communicate with the driver. The runner now
  has a CUDA preflight so future real matrix runs fail before loading weights
  when GPU access is absent. The two-repeat performance artifact remains
  pending until GPU access is restored.
- The matrix configs now carry workload-specific stored JAX and vLLM reference
  paths for `hetero8` and `long_prefill_512_2048`. A dry run verified that both
  repeats of `gpu_paged_default` and `gpu_paged_fast_optin` will compare against
  stored JAX references from repeat one instead of leaving the default
  long-prefill baseline unchecked. Stored references are preferred over
  generated same-run defaults so exact-token gates use stable baselines.
- Matrix summaries now include an `acceptance` section that makes the plan's
  speed-claim evidence check explicit: successful subprocesses, minimum
  repeats, exact correctness, JAX/vLLM performance presence, TTFT/ITL p50/p95
  latency, first `forward_step_token_ids_jit`, profile counters, and the
  `0.75x` vLLM target. All configured profile-counter buckets must be present
  for every repeat. The runner validates the summary shape before writing it.
  Human explanation of profile bucket movement still belongs in the logbook.
- `benchmarks/run_gpu_matrix.py --require-speed-claim-ready` can be used for
  the final benchmark command. It still writes the summary, then exits nonzero
  if any selected workload/config is not speed-claim-ready or misses the `0.75x`
  vLLM target.
- `--require-stored-references` can be used before benchmark launch to fail
  fast when selected workloads/configs lack stored JAX or vLLM references.
- Focused tests also verify that all GPU matrix configs have valid stored JAX
  and vLLM references for `hetero8` and `long_prefill_512_2048`, so the next
  GPU-visible two-repeat run should not silently fall back to unchecked
  baselines for the tracked workloads.
- The same focused suite verifies command construction and runtime environment
  defaults for matrix runs: workload overrides, `--reference-json`, warmup,
  profile, `JAX_PLATFORMS=cuda`, and cache/temp roots under the configured
  `/mountpoint` runtime root.

## Phase 2 - Kernel Roadmap

### Kernel Priority Table

```text
P0:
1. kv_append_paged_nhd
2. paged_decode_attention_gqa_nhd

P1:
3. gdn_recurrent_decode_step
4. gdn_segmented_prefill_chunk32

P2:
5. paged_prefill_attention_gqa_nhd
6. qk_norm_rope_kv_append_fused

P3:
7. topk_logits / sampling / logprob
8. silu_and_mul / RMSNorm smoke-test kernels only if needed for FFI validation
```

The important ordering is: first own KV layout and decode attention, then attack
GDN decode/prefill using external kernel references. Do not start with MTP or
top-k.

## P0.1 - `kv_append_paged_nhd`

### Motivation

This defines the full-attention KV-cache physical layout. Without this, every
later attention kernel risks layout conversion overhead.

### Reference

Use FlashInfer's `append_paged_kv_cache` as the first target. It appends ragged
K/V tensors into a paged KV cache and supports `NHD` layout with cache shape
`[max_num_pages, page_size, num_kv_heads, head_dim]`, or a combined 5-D cache
`[max_num_pages, 2, page_size, num_kv_heads, head_dim]`.

Also study vLLM's `reshape_and_cache_flash_kernel`; it writes token K/V into
paged cache using `slot_mapping`, supports NHD/HND-style layout, and is close to
the desired CUDA contract.

### Proposed ABI

```python
kv_append_paged_nhd(
    append_key,        # [nnz_tokens, num_kv_heads, head_dim]
    append_value,      # [nnz_tokens, num_kv_heads, head_dim]
    batch_indices,     # [nnz_tokens]
    positions,         # [nnz_tokens]
    k_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    kv_indices,        # [total_pages]
    kv_indptr,         # [batch + 1]
    kv_last_page_len,  # [batch]
) -> updated_cache
```

For this model's full-attention layers:

```text
num_q_heads = 8
num_kv_heads = 2
head_dim = 256
initial page_size = 16
secondary page_size sweep = 32
```

### Implementation Path

```text
1. Add full-attention KV cache allocation in NHD layout.
2. Keep existing JAX cache path as fallback.
3. Add FlashInfer/JAX FFI wrapper.
4. Route only full-attention layers through the new cache append.
5. Run exact-token parity.
```

FlashInfer documents calling its GPU kernels from JAX through `jax-tvm-ffi`;
that should be the first integration route rather than writing CUDA from
scratch.

### Acceptance Gate

```text
- exact generated-token match vs Entry 045 reference
- no TTFT regression
- no ITL p50/p95 regression
- lower or equal cache write/gather/transpose overhead
- no new host sync around FFI workspace setup
```

### Status

- Focused CUDA FFI proof passed on 2026-05-26 for separate NHD K/V caches,
  BF16 cache tensors, `head_dim=128` and `head_dim=256`, and a non-contiguous
  page table.
- The working JAX FFI registration uses `arg_spec=["args", "attrs.layout"]`
  and `input_output_aliases={4: 0, 5: 1}` so FlashInfer can mutate cache inputs
  while JAX receives functional cache outputs with unwritten entries preserved.
- The opt-in route `NANO_VLLM_JAX_FLASHINFER_KV_APPEND=1` updates canonical
  per-layer cache slices through FlashInfer and then returns the existing cache
  layout to the pure-JAX attention path.
- A full hetero8 server run with the opt-in route failed during warmup before
  producing a benchmark artifact because FlashInfer's append dispatcher does not
  support FP32 cache tensors. The current accepted contract keeps BF16 weights
  and FP32 activation/KV-cache tensors, so this route is rejected for default
  serving unless the dtype policy changes or a FP32 append kernel is added.
- A local CUDA/JAX FFI FP32 append smoke kernel now builds under
  `/mountpoint/.exp/.cache`, uses the same NHD append ABI and cache-output
  aliasing contract, and passes a focused CUDA parity test against the pure-JAX
  NHD append reference. This proves the FP32 custom-call toolchain but is not
  routed into serving or accepted as a speed path yet.
- Routing that local FP32 append kernel through `write_kv` behind
  `NANO_VLLM_JAX_CUDA_FP32_KV_APPEND=1` preserved exact generated-token
  correctness on hetero8, but regressed throughput to `193.62 tok/s` and ITL
  p50 to `31.43 ms`. Keep the kernel as a toolchain smoke proof; do not promote
  standalone append routing without a paired attention/layout consumer or
  integrated profile evidence that removes the added overhead.
- This is not exact-token/integrated-benchmark accepted and is not a default
  backend.

## P0.2 - `paged_decode_attention_gqa_nhd`

### Motivation

This is the closest equivalent to vLLM's core paged decode attention path.

### Reference

vLLM's PagedAttention design relies on a specialized memory layout and access
method for reading paged KV cache efficiently.

FlashInfer provides `BatchDecodeWithPagedKVCacheWrapper`, which is specifically
for batched decode attention with paged KV cache and supports `NHD` layout.

### Proposed ABI

```python
paged_decode_attention_gqa_nhd(
    q,                 # [batch, num_q_heads, head_dim]
    k_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache,           # [num_pages, page_size, num_kv_heads, head_dim]
    kv_indptr,         # [batch + 1]
    kv_indices,        # [total_pages]
    kv_last_page_len,  # [batch]
    seq_lens,          # [batch]
    softmax_scale,     # float
) -> out               # [batch, num_q_heads, head_dim]
```

### Acceptance Gate

```text
- exact generated-token match
- top-5 parity on focused decode tests
- ITL p50 improves
- ITL p95 does not regress
- PjRt Execute improves or stays flat
- command_buffer update/execute does not regress
```

### Do Not Merge If

```text
- FFI planning/setup happens inside the decode loop
- layout conversion is needed every layer
- benchmark only wins in a microbenchmark but loses integrated server throughput
```

### Status

- FlashInfer's batch decode/prefill JIT dtype maps do not include FP32 for the
  Q/KV/O attention path, and the KV dtype map is BF16/FP16/low-precision only.
- Under the current BF16-weights/FP32-activation contract, do not start a
  FlashInfer `paged_decode_attention_gqa_nhd` wrapper as the next serving path.
  The viable options are a FP32-capable custom kernel, or an explicit separate
  decision to change the KV-cache/attention dtype policy.
- A pure-JAX FP32 NHD ABI reference now exists for
  `paged_decode_attention_gqa_nhd`, with focused parity tests against the
  current decode attention path. This is an ABI/correctness target for a future
  CUDA custom-call, not an accepted performance kernel.
- A local FP32 CUDA/JAX FFI implementation now exists for the same ABI. It
  passes focused CUDA parity against the pure-JAX reference, including the
  model's full-attention `8q/2kv/head_dim=256` shape, under the same
  `jax_default_matmul_precision=highest` correctness mode used by the long
  decode top-5 harness.
- The local FP32 CUDA decode attention route is available behind
  `NANO_VLLM_JAX_CUDA_FP32_DECODE_ATTN=1`, but the first integrated hetero8 run
  was slower than Entry 045 despite exact generated-token parity. Keep it
  default-off as a diagnostic route; do not promote it to default or fast
  opt-in.

## P1.1 - `gdn_recurrent_decode_step`

### Motivation

This is probably the highest-leverage model-specific kernel. The model has 18
GDN layers and only 6 full-attention layers, so full-attention kernels alone will
not close the vLLM gap.

vLLM's Qwen3-Next support explicitly calls out hybrid attention with Gated
DeltaNet plus full attention, and says the implementation integrates Triton
kernels from Flash Linear Attention for Gated DeltaNet.

### Reference

Study:

```text
- vLLM Qwen3-Next implementation path
- Flash Linear Attention Gated DeltaNet kernels
- current JAX GDN recurrence implementation
```

Flash Linear Attention provides hardware-efficient building blocks for linear
attention, sparse attention, state-space models, and hybrid LLM architectures.

### Proposed ABI

```python
gdn_recurrent_decode_step(
    q,          # [batch, gdn_heads, head_dim], fp32 activation
    k,          # [batch, gdn_heads, head_dim]
    v,          # [batch, gdn_heads, head_dim]
    beta,       # [batch, gdn_heads] or [batch, gdn_heads, 1]
    gate,       # [batch, gdn_heads] or [batch, gdn_heads, 1]
    state,      # [batch, gdn_heads, head_dim, head_dim], fp32 state
) -> (
    out,        # [batch, gdn_heads, head_dim]
    new_state,  # same shape as state
)
```

For Qwen3.5-0.8B GDN:

```text
gdn_heads = 16
head_dim = 128
state dtype = fp32 unless proven safe otherwise
```

### Implementation Path

```text
1. Write isolated reference test comparing current JAX recurrence vs proposed kernel.
2. Study Qwen 3 Next vLLM and Flash Linear Attention Gated DeltaNet kernels.
3. Start with one CUDA custom-call/FFI kernel, not Pallas.
4. Keep pure-JAX recurrence as default fallback.
5. Integrate behind NANO_VLLM_JAX_KERNEL_BACKEND=gdn_cuda.
```

Pallas is still worth studying, but JAX documents Pallas as a custom-kernel
language for GPU/TPU and marks it with experimental caveats. Given the repo's
previous Pallas regressions, it should not be the first production path for GDN.

### Status

- Current serving GDN decode/prefill is still pure JAX/XLA at the backend
  boundary. `PureJAXBackend.gated_delta_decode` calls
  `jax_recurrent_gated_delta_rule`; `PureJAXBackend.gated_delta_prefill` calls
  `jax_chunk_gated_delta_rule`.
- A local FP32 CUDA/JAX FFI width-1 `gdn_recurrent_decode_step` prototype now
  passes focused parity against `jax_recurrent_gated_delta_rule`, including the
  model's `batch=2`, `gdn_heads=16`, `head_dim=128` shape.
- The local FP32 CUDA GDN decode route is available behind
  `NANO_VLLM_JAX_CUDA_FP32_GDN_DECODE=1`, but the first integrated hetero8 run
  was slower than Entry 045 despite exact generated-token parity. Keep it
  default-off as a diagnostic route; do not promote it to default or fast
  opt-in.

### Acceptance Gate

```text
- recurrent unit test matches current JAX output within strict tolerance
- full 500-step generated-token parity remains exact
- ITL p50 improves on hetero8
- ITL p95 does not regress
- forward_step_token_ids_jit decreases
- no activation/state dtype downgrade
```

## P1.2 - `gdn_segmented_prefill_chunk32`

### Motivation

Entry 045 establishes chunk size 32 as the best verified point for the tracked
hetero8 workload. The rejected static chunk-major GDN prefill experiment shows
that source-level ragged scans explode into many small dynamic-slice/update
kernels and badly regress first prefill.

Therefore, the next GDN prefill attempt must be backend-owned: one coarse op
that internally handles segmentation and chunking.

### Proposed ABI

```python
gdn_segmented_prefill_chunk32(
    q,              # [nnz_tokens, gdn_heads, head_dim]
    k,              # [nnz_tokens, gdn_heads, head_dim]
    v,              # [nnz_tokens, gdn_heads, head_dim]
    beta,           # [nnz_tokens, gdn_heads]
    gate,           # [nnz_tokens, gdn_heads]
    cu_seqlens,     # [batch + 1]
    initial_state,  # [batch, gdn_heads, head_dim, head_dim]
    chunk_size=32,
) -> (
    y,              # [nnz_tokens, gdn_heads, head_dim]
    final_state,    # [batch, gdn_heads, head_dim, head_dim]
)
```

### Acceptance Gate

```text
- exact generated-token match
- first forward_step_token_ids_jit improves
- total TTFT p50 improves
- PjRt Execute does not regress
- command_buffer execute/update does not regress
- no explosion in dynamic_slice/dynamic_update_slice or tiny command buffers
```

### Do Not Merge If

```text
- kernel only improves a microbenchmark
- first prefill gets slower
- chunking creates many more command-buffer executions
```

### Status - 2026-05-26

- GDN prefill is a profile-backed target, not just an architecture guess. Entry
  045's chunk-size change moved the intended `input_reduce_fusion` bucket from
  `59.30 ms / 2512` to `28.65 ms / 1936` and improved the accepted integrated
  server baseline, while later row/chunk source rewrites showed matching
  regressions in first prefill, `PjRt Execute`, and command-buffer work.
- First local CUDA/JAX FFI prototype added as
  `gdn_prefill_chunk32_normalized_fp32` and benchmark variant
  `cuda_fp32_one_piece_chunk32`.
- Focused CUDA FFI tests pass, and the reduced `B=2,H=2,T=64,K=32,V=32`
  benchmark is faster with small drift.
- Full hetero8 model-shape microbenchmark rejects this first prototype:
  `11.50 ms` p50 versus `5.43 ms` p50 for current JAX chunk32, with
  `state_max_abs=2.441e-04`.
- A follow-up V64 value-block variant reduced full-shape p50 from `11.56 ms`
  to `8.60 ms`, confirming value-block/grid overhead is real, but it still lost
  to current JAX `5.44 ms` p50 and kept the same `state_max_abs=2.441e-04`.
- A pure-JAX packed segmented/nnz ABI gate was added before CUDA math. It passes
  reduced shape, but full hetero8 true-token packing fails the strict standalone
  gate versus current padded chunk32: output max `1.431e-05`, state max
  `1.678e-04`. Do not implement CUDA math for this packed ABI until the
  correctness contract is resolved.
- A row-padded diagnostic then padded each packed row back to rectangular
  `T=512` before applying the chunk rule. It still failed (`state_max_abs=
  1.831e-04`), so the drift is not just shorter per-row sequence length; the
  row-wise decomposition itself changes enough FP32 accumulation to miss the
  current gate.
- Keep it default-off and benchmark-only. Do not route into serving. The next
  attempt should move closer to the true segmented/nnz ABI and preserve FP32
  accumulation more closely before a server run.

## P2.1 - `paged_prefill_attention_gqa_nhd`

### Motivation

Useful for the 6 full-attention layers, but lower priority than GDN because this
architecture is GDN-heavy.

### Reference

FlashInfer provides `BatchPrefillWithPagedKVCacheWrapper` for batched
prefill/append attention over paged KV cache.

MaxText is useful as a design reference: its docs say serving attention uses
paged/ragged kernels to fetch non-contiguous KV cache pages in high-throughput
inference.

### Proposed ABI

```python
paged_prefill_attention_gqa_nhd(
    q,                 # [nnz_q, num_q_heads, head_dim]
    k_cache,
    v_cache,
    qo_indptr,          # [batch + 1]
    kv_indptr,          # [batch + 1]
    kv_indices,         # [total_pages]
    kv_last_page_len,   # [batch]
    causal=True,
) -> out                # [nnz_q, num_q_heads, head_dim]
```

### Acceptance Gate

```text
- exact-token match
- TTFT p50 improves
- no ITL regression
- no per-layer layout conversion
```

## P2.2 - `qk_norm_rope_kv_append_fused`

### Motivation

Only attempt this after P0 kernels are stable. The useful fusion is not just
RoPE; it is:

```text
QKV projection output
-> split Q/K/V
-> Q/K norm if present
-> RoPE on Q/K
-> append K/V into paged cache
-> return Q for attention
```

### Acceptance Gate

```text
- exact-token match
- lower decode-layer overhead
- no extra layout conversions
- no host-side metadata changes
```

### Do Not Start Before

```text
- kv_append_paged_nhd is stable
- paged_decode_attention_gqa_nhd is stable
- profile still shows norm/RoPE/cache-write overhead
```

## P3 - Sampling/Top-k/Logprob Kernels

### Motivation

These are useful for features and diagnostics, but not the next throughput lever
for greedy serving.

FlashInfer exposes top-k, sampling, logits processor, norm, RoPE, and activation
APIs, so these can be used later as FFI smoke tests or feature work.

### Possible APIs

```python
topk_logits(
    logits,  # [batch, vocab]
    k: int,
) -> (values, indices)
```

```python
sample_topk_topp(
    logits,
    temperature,
    top_k,
    top_p,
    rng_state,
) -> token_ids
```

### Do Not Prioritize Until

```text
- decode ITL is closer to vLLM
- non-greedy sampling/logprobs are product requirements
- MTP diagnostics need draft-rank/top-k efficiently
```

## Phase 3 - MTP Policy

### Goal

Keep MTP as diagnostics/correctness work, not speed work.

### Tasks

Add MTP diagnostics, but do not optimize MTP kernels yet:

```text
- draft token rank under target model
- accepted/rejected draft count
- acceptance rate by prompt bucket
- verifier overhead
- commit-select JIT/cache-miss overhead
- warmed vs cold commit-select timing
```

The current logbook says MTP1 repeat was unusably slower, dominated by
verifier/commit-select/tracing overhead, and exact output was preserved only
because drafts were rejected.

### MTP Can Be Revisited Only When

```text
- acceptance rate is meaningfully nonzero on at least one realistic prompt suite
- commit-select shapes are warmed before timing
- MTP beats paged baseline on a controlled benchmark
```

## Phase 4 - What To Borrow

### From vLLM

Borrow:

```text
- paged KV layout discipline
- slot/page table contract
- decode attention ABI
- cache append kernel contract
- CUDA graph/static-shape mindset
- hybrid Qwen attention/GDN implementation references
```

Do not borrow:

```text
- full scheduler architecture
- PyTorch-specific graph assumptions
- MTP speed path before acceptance quality is proven
```

vLLM's paged attention docs are the right reference for the core decode attention
memory-access pattern.

### From FlashInfer

Borrow first, because it is the most practical CUDA kernel route from JAX:

```text
- append_paged_kv_cache
- BatchDecodeWithPagedKVCacheWrapper
- BatchPrefillWithPagedKVCacheWrapper
- later: top-k/sampling/RoPE/RMSNorm helpers
```

FlashInfer's docs describe using FlashInfer kernels from JAX through
`jax-tvm-ffi`, with CUDA/JAX/FlashInfer setup requirements.

### From MaxText

Borrow design patterns, not GPU drop-ins:

```text
- profile first
- use custom kernels for bandwidth-bound or irregular/ragged work
- use auxiliary metadata
- validate integrated model performance, not only microbenchmarks
- think in decode/prefill/mixed specialized kernels
```

MaxText's Pallas guide says custom kernels are appropriate for irregular compute
and memory-access-bound attention, but also warns to profile first and prefer
XLA when standard dense ops already perform well.

The Ragged Paged Attention paper is useful conceptually: it uses fine-grained
ragged tiling, fuses KV cache update with attention computation, and specializes
decode/prefill/mixed kernels for TPU serving. Treat it as a design reference for
kernel boundaries, not a CUDA implementation to copy.

## Implementation Order

Commit 1:

- ~~Add `docs/current_gpu_baseline.md`~~
- ~~Add `docs/rejected_optimization_index.md`~~
- ~~Add `docs/kernel_roadmap.md`~~
- ~~Add benchmark config JSONs~~

Commit 2:

- ~~Add benchmark matrix runner~~
- ~~Add summary JSON output schema~~

Commit 3:

- ~~Add kernel backend registry:~~

- ~~`nanovllm_jax/kernels/__init__.py`~~
- ~~`nanovllm_jax/kernels/registry.py`~~
- ~~`nanovllm_jax/kernels/flashinfer_ffi.py`~~
- ~~`nanovllm_jax/kernels/cuda_gdn.py`~~

Commit 4:

- ~~Add NHD full-attention KV cache allocation behind a flag~~
- ~~Keep existing pure-JAX cache fallback~~

Commit 5:

- ~~Add `kv_append_paged_nhd` prototype via FlashInfer/JAX FFI~~
- ~~Add focused CUDA parity test against the pure-JAX NHD append reference~~
- ~~Route full-attention layers through NHD append behind an opt-in flag~~
- ~~Run live integrated attempt and record rejection under the current FP32 KV-cache contract~~

Interim ABI validation:

- ~~Add pure-JAX `kv_append_paged_nhd` ABI reference for the exact NHD append contract~~
- ~~Add focused parity test against the canonical `update_kv_cache` path~~
- ~~Add local CUDA/JAX FFI FP32 `kv_append_paged_nhd` smoke kernel~~
- ~~Add focused CUDA parity test against the pure-JAX NHD append reference~~
- ~~Route local FP32 append through `write_kv` behind an opt-in flag~~
- ~~Run live integrated attempt and record rejection of standalone append routing~~

Commit 6:

- Do not add `paged_decode_attention_gqa_nhd` via FlashInfer/JAX FFI under the
  current FP32 activation/KV-cache contract.
- ~~Add pure-JAX FP32 `paged_decode_attention_gqa_nhd` ABI reference~~
- ~~Add focused parity tests against the current decode path~~
- ~~Add FP32-capable CUDA/custom-call decode implementation with focused CUDA
  parity tests~~
- ~~Route FP32-capable CUDA/custom-call decode implementation behind an opt-in
  backend~~
- ~~Run live integrated attempt and record rejection of standalone decode
  routing~~
- Route only full-attention decode layers as an accepted path after integrated
  gates pass

Commit 7:

- ~~Add gdn_recurrent_decode_step prototype~~
- ~~First isolated tests~~
- ~~Route through `gated_delta_decode` behind an opt-in flag~~
- ~~Run integrated decode benchmark and record rejection of standalone GDN
  decode routing~~

Commit 8:

- ~~Add first gdn_segmented_prefill_chunk32 prototype~~
- Keep first CUDA one-piece chunk32 prototype default-off and benchmark-only;
  do not route it into serving
- ~~Run value-block-width follow-up and record that V64 improves V32 but still
  misses the full-shape microbenchmark gate~~
- ~~Add packed segmented/nnz ABI correctness gate before CUDA math~~
- ~~Run row-padded segmented reference diagnostic and record that row-wise
  decomposition still misses the full-shape gate~~
- Resolve packed-ABI correctness policy after full hetero8 gate failure before
  implementing segmented CUDA math
- Compare a revised segmented prefill candidate against Entry 045 chunk-32
  baseline after it beats the full-shape GDN microbenchmark gate

Commit 9:

- Add paged_prefill_attention_gqa_nhd only if traces justify it

Commit 10:

- Add MTP diagnostics; no MTP speed optimization yet

If FlashInfer/JAX FFI integration is harder than expected, insert a small ABI
validation commit before Commit 4. That commit should prove an optional external
kernel can be discovered, gated, called, profiled, and bypassed without touching
the pure-JAX correctness path.

## Hard Rules For The Agent

```text
1. Do not merge any speed change without exact generated-token parity.
2. Do not compare against stale baselines; compare against Entry 045 or the latest accepted baseline.
3. Do not optimize MTP for speed yet.
4. Do not implement more source-level JAX rewrites unless HLO/profile evidence says they target a real bottleneck.
5. Do not accept microbenchmark-only wins.
6. Do not add per-layer layout conversions to use an external kernel.
7. Keep every external kernel behind a backend flag and fallback path.
8. Record rejected experiments. Rejected experiments are useful evidence, not failure.
```

## Expected Strategic Outcome

The path is:

```text
Clean baseline
-> stable benchmark matrix
-> FlashInfer paged KV append
-> FlashInfer paged decode attention
-> custom/ported GDN decode recurrence
-> custom/ported segmented GDN prefill
-> only then revisit MTP or finer fusions
```

The core bet is: keep the JAX model and correctness harness, but replace the few
serving kernels where vLLM has structural advantage: paged KV append, paged
decode attention, and GDN recurrence/prefill.

## Reference Links From The Proposal

- Optimization logbook:
  <https://raw.githubusercontent.com/LiquidGunay/nano-vllm-jax/main/docs/optimization_logbook.md>
- Qwen/Qwen3.5-0.8B model card:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B>
- FlashInfer `append_paged_kv_cache`:
  <https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html>
- vLLM cache kernels:
  <https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/cache_kernels.cu>
- FlashInfer on JAX with TVM FFI:
  <https://docs.flashinfer.ai/tutorials/generated/jax_tvm_ffi/index.html>
- vLLM paged attention:
  <https://docs.vllm.ai/en/latest/design/paged_attention/>
- FlashInfer attention kernels:
  <https://docs.flashinfer.ai/api/attention.html>
- vLLM Qwen3-Next support note:
  <https://vllm.ai/blog/2025-09-11-qwen3-next>
- Flash Linear Attention:
  <https://github.com/fla-org/flash-linear-attention>
- JAX Pallas docs:
  <https://docs.jax.dev/en/latest/pallas/index.html>
- MaxText Pallas performance guide:
  <https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html>
- Ragged Paged Attention paper:
  <https://arxiv.org/abs/2604.15464>
