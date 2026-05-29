# FLA GDN Kernel Adapter Findings (`2026-05-28`)

Scope: search only local environment/repo for a prefill Gated DeltaNet kernel that can
replace the current staged Triton implementation directly from JAX.

## Modules / files found

- Found FlashInfer package in local venv:
  - `.venv/lib/python3.11/site-packages/flashinfer/gdn_prefill.py`
  - `.venv/lib/python3.11/site-packages/flashinfer/data/csrc/prefill_kernel_delta_rule_sm90.cu`
  - `.venv/lib/python3.11/site-packages/flashinfer/data/csrc/gdn_prefill_launcher.cu`
  - `.venv/lib/python3.11/site-packages/flashinfer/gdn_kernels/blackwell/gdn_prefill.py`
  - `.venv/lib/python3.11/site-packages/flashinfer/jit/gdn.py` (SM90 JIT generation helper)
- Search for `vllm`, `fla`, `fused_recurrent_gated_delta_rule`, and `chunk_gated_delta_rule` in repo env found **no importable vLLM/FLA Python package** in this tree (`importlib.util.find_spec('vllm') == None`).

## Candidate callable

- `flashinfer.gdn_prefill.chunk_gated_delta_rule` exists with signature:
  - `chunk_gated_delta_rule(q, k, v, g: Optional[Tensor]=None, beta: Optional[Tensor]=None, scale: Optional[float]=None, initial_state: Optional[Tensor]=None, output_final_state: bool=False, cu_seqlens: Optional[Tensor]=None, use_qk_l2norm_in_kernel: bool=False, output: Optional[Tensor]=None, output_state: Optional[Tensor]=None, state_checkpoints: Optional[Tensor]=None, checkpoint_cu_starts: Optional[Tensor]=None, checkpoint_every_n_tokens: int=0)`
- Kernel contract documented in FlashInfer source:
  - `q, k, v`: `[total_tokens, H, D]` contiguous CUDA tensors
  - `g, beta`: `[total_tokens, O]`, float32
  - `initial_state` / `output_state`: `[num_rows, O, D, D]`, float32
  - output: `[total_tokens, O, D]`, same dtype as q/k/v
  - requires `cu_seqlens` length `num_rows + 1`

## Chunking and shape constraints

- Blackwell path (`chunk_gated_delta_rule_sm100`) hard-codes chunk size `64` in
  `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py` (`self.b_t = 64`).
- Public docs for `chunk_gated_delta_rule` require chunked processing but do not expose
  a configurable chunk size argument.
- Our staged JAX/Fused path target in this task is `chunk_size=32`; direct FlashInfer
  prefill path is therefore **ABI-incompatible** unless we add remapping/aggregation logic.

## JAX integration feasibility

- Call path is Torch/Triton/C++ based and returns torch tensors.
- It is not callable from JAX without an FFI/custom-call bridge.
- This is effectively a **framework boundary issue**: to use directly in JAX serving we
  would need at least one of:
  1. Torch ↔ JAX bridge (costly and likely unsuitable for this stack), or
  2. New JAX `CustomCall` binding wrapping FlashInfer CUDA entrypoints.

## Probe script added

- Added `benchmarks/adapter_flashinfer_gdn_prefill.py` (isolated, optional).
- It prints importability/signature/contract and supports optional `--compare-jax` parity
  against `nanovllm_jax.kernels.gdn_fla_chunk_gated_delta_rule_packed_reference`.
- It intentionally does not change serving code.

## Recommended next step

Given fixed 64-token chunking and no direct JAX bridge, the smallest useful next step
is to continue with the staged Triton ports and use this information as a target contract
for later full external-kernel integration (if/when a direct JAX binding is added).

## Follow-up probe (2026-05-28, A10G): vLLM/FLA Triton path

### Search results

- In the project venv, there is still no top-level `fla` / `flash-linear-attention` / `flash_linear_attention` package.
- There is a separate `vllm` venv with vendored FLA ops at:
  - `/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages/vllm/model_executor/layers/fla/ops/chunk.py`
  - `/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py`
  - `/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages/vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py`
  - `/mountpoint/.exp/vllm-venv/lib/python3.11/site-packages/vllm/model_executor/layers/fla/ops/chunk.py` also defines:
    - `chunk_gated_delta_rule_fwd`
    - `chunk_gated_delta_rule`
  - `FLA_CHUNK_SIZE = 64` from `fla/ops/utils.py`.
- `fla` module itself is not importable directly (`find_spec('fla') == False`); calls are available through `vllm.model_executor.layers.fla...`.

### Practicality on A10G

- The vLLM/FLA Triton path is importable/runnable with Torch on this host.
- A tiny A10G smoke run with `heads=16`, `D=128`, `V=128`, `BF16` inputs succeeded to execute
  `vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule` (no runtime import/JIT errors).
- A direct parity check against `gdn_fla_chunk_gated_delta_rule_packed_reference` on tiny packed inputs produced large numerical mismatch
  (`out max diff ~10–300` depending on random seed in this smoke), indicating the two pipelines are not
  currently drop-in equivalent in this environment (state/layout and/or intermediate math ordering differences).

### Recommended next step

- Treat vLLM/FLA as the **reference implementation source** only unless a direct JAX binding is added.
- If re-trying direct integration, first normalize exactly the packed layout + gating/state semantics for
  `query/key/value/gate/beta/state` ordering and initial-state update order before enabling in production serving.
