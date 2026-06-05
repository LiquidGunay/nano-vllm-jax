# GDN FLA Kernel Flow

This note explains the current block-dot Gated DeltaNet / Flash Linear Attention
prefill route. It is meant to keep benchmark interpretation and future kernel
work from depending on hidden context in the optimization logbook.

## What The Route Optimizes

The current kernel route targets GDN prefill after the model has prepared packed
varlen tensors:

```text
packed q, k, v, gate, beta, cu_seqlens, initial_state
    -> per-chunk gate cumsum
    -> KKT attention matrix A
    -> triangular solve inverse
    -> recompute W and U
    -> recurrent delta-H/state update
    -> forward output O
    -> unpack back to the model layout
```

The block-dot work replaces scalar or row-wise Triton/JAX work in four stages:

```text
KKT:
  old: row-wise dot accumulation
  new: one [BT, BT] tl.dot tile per chunk/head

recompute W/U:
  old: per-row scalar recurrence-style reconstruction
  new: A @ (beta * V) and A @ (beta * exp(gate) * K) block dots

delta-H:
  old: value/key element loops with large materialized intermediates
  new: value-tiled W @ H^T and K @ delta block dots

forward output:
  old: scalar Q/K/V accumulation
  new: Q @ H^T plus triangular (Q @ K^T) @ V_new
```

The route is controlled in typed configs with:

```json
"kernels": {
  "gdn": {
    "prefill_post_conv_impl": "triton_fla_padded",
    "disable_fallbacks": true,
    "prefill_block_dot": true,
    "packed_decode": {
      "impl": "reference",
      "qkv_dtype": "bf16"
    }
  }
}
```

The older per-stage keys remain available for narrow ablations, but benchmark
configs should prefer the single `prefill_block_dot` knob to avoid config/env
sprawl.

With `disable_fallbacks=true`, a requested packed GDN decode kernel must run or
the model raises before entering the recurrent JAX path. Do not use
`packed_decode.max_batch` in promoted benchmark configs; it can make wider
random/heterogeneous decode silently select reference recurrence unless strict
fallback checks catch it.

The promoted random-serving config currently uses explicit packed `reference`
GDN decode with BF16 QKV because integrated large-random repeats beat the
standalone Triton GDN decode route. The widest current diagnostic decode kernel
boundary is
`triton_fla_conv_raw_gates`: packed projection output `[qkv, a, b, z]` feeds a
single Triton call for causal-conv update, SiLU on Q/K/V, Q/K L2 normalization,
raw gate/softplus/sigmoid math, recurrent-state read/update, and per-head
linear-attention output. The projection matmul before it and the output
normalization/projection after it are still separate JAX/XLA operations.

`triton_fla_conv_raw_gates_tail` is an opt-in diagnostic route that additionally
loads `z` and `norm_weight`, computes per-head RMSNorm plus `silu(z)` inside the
same Triton call, and returns `[batch, 1, value_heads * value_dim]` ready for the
existing output projection. It is not the promoted serving route: medium random
A/B on 2026-06-05 regressed from `407.54` to `362.07` output tok/s, likely
because full-head value tiles and the extra custom-call contract cost more than
the removed XLA tail ops. The stored-vLLM denominator in that diagnostic is only
a rough reference because the regenerated JAX envelope had a different output
token count from the earlier vLLM artifact; the raw-vs-tail comparison is the
decision signal.

## Model-Specific Assumptions

The current evidence is for `Qwen/Qwen3.5-0.8B` on an NVIDIA A10G. Treat these as
specific to that model/hardware until revalidated:

- GDN layer share: the 0.8B model has enough GDN prefill work for this route to
  matter on long-prefill shapes. Larger Qwen3.5 variants may have different
  layer mixes and full-attention/GEMM balance.
- Head and state sizes: the kernels have been exercised around 64-wide key/value
  tiles and value tiling such as `BV=32`. Different hidden sizes, head counts,
  or state widths can change occupancy, shared-memory pressure, and the best
  tile shape.
- Hardware tuning: `num_warps`, `num_stages`, and value tile sizes are A10G
  observations, not portable constants. Other GPUs need profiling, not assumed
  reuse.
- Workload mix: long-prefill speedups do not imply decode-heavy speedups. The
  block-dot route mostly attacks prefill GDN stages; decode-heavy workloads can
  still be dominated by decode GEMMs, launch/update overhead, transposes, and
  host/device transfers.

What should generalize:

- the packed varlen ABI using `cu_seqlens` and chunk metadata;
- the FLA stage ordering;
- replacing scalar per-row KKT/recompute/output work with block dot products;
- preserving FP32 gate/beta/state accumulation while allowing BF16 QKV where the
  correctness contract permits it.

What must be remeasured for 4B or 27B Qwen3.5:

- generated-token exactness against that model's accepted baseline;
- long-prefill throughput versus a same-workload JAX reference and vLLM;
- decode-heavy throughput, because the bottleneck mix can move away from GDN;
- top profile buckets after warmup, especially GEMM, transpose, D2D copy, and
  command-buffer update density.

## Benchmark Interpretation

A kernel is not globally usable just because it wins one benchmark, and it is
not globally rejected just because it loses another. Use the benchmark that
matches the intended claim:

- `long_prefill_512_2048`: primary prefill-kernel hill-climb target. A win here
  means the GDN prefill route is doing useful work.
- `decode_heavy_128x128`: decode-path target. A prefill kernel can be neutral or
  irrelevant here; losses point to decode/host/GEMM work, not necessarily GDN
  prefill failure.
- `hetero8`: mixed serving smoke. It is useful for broad correctness and route
  safety, but its throughput combines one prefill step with many decode steps.

The 2026-06-02 mixed-length smoke for the block-dot route passed generated-token
correctness on `hetero8`, but reported about `0.838x` of the stored Entry 045
JAX reference and `0.357x` of the stored vLLM reference in a one-repeat run.
That does not invalidate the long-prefill kernel result; it says the current
route is not a serving-wide win on this mixed workload and should not be
promoted as the default without further decode/host work.

## Promotion Rule

Promote this route only for the claim it has proved:

- keep it as the current long-prefill best when it is speed-claim-ready, exact,
  and above the target ratio on `long_prefill_512_2048`;
- do not claim general serving speedup until `hetero8` and decode-heavy
  workloads are exact and non-regressing against the accepted JAX reference;
- always report the workload, reference artifact/source, repeat count, and
  whether the run is speed-claim-ready.
