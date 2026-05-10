# vLLM TPU comparison note

Date: 2026-05-10

Worker: D2

Scope: Qwen 0.8B vLLM TPU baseline/speculative comparison on `nano-vllm-jax-spot-v6e2-1527`, `europe-west4-a`, `v6e-1`. No core engine code was changed.

## Feasibility

The TPU VM is reachable and `READY`:

```bash
gcloud compute tpus tpu-vm describe nano-vllm-jax-spot-v6e2-1527 \
  --zone=europe-west4-a \
  --format='yaml(name,state,acceleratorType,apiVersion,networkEndpoints,queuedResource)'
```

Observed:

```text
state: READY
acceleratorType: v6e-1
```

The base VM had Python 3.10.12, JAX 0.6.2, one TPU device, and no vLLM. I installed vLLM TPU in an isolated remote directory, not in this repo:

```bash
mkdir -p "$HOME/work-dir/vllm-tpu-qwen08"
cd "$HOME/work-dir/vllm-tpu-qwen08"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv vllm_env --python 3.12
source vllm_env/bin/activate
uv pip install vllm-tpu
python - <<'PY'
import importlib.metadata, jax, vllm
from vllm.platforms import current_platform
print("vllm", vllm.__version__)
print("tpu_inference", importlib.metadata.version("tpu_inference"))
print("platform", current_platform.get_device_name())
print("jax", jax.__version__)
print("devices", jax.devices())
PY
```

Observed:

```text
vllm 0.19.0
tpu_inference 0.19.0
platform TPU V6E
jax 0.9.2
devices [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

This matches the official vLLM TPU install path: https://docs.vllm.ai/projects/tpu/en/stable/getting_started/installation/

## Baseline attempts for Qwen/Qwen3.5-0.8B

I used `Qwen/Qwen3.5-0.8B` as the intended "Qwen 0.8B" target.

First benchmark command:

```bash
cd "$HOME/work-dir/vllm-tpu-qwen08"
source "$HOME/.local/bin/env"
source vllm_env/bin/activate
mkdir -p hf_cache xla_cache results
export HF_HOME="$HOME/work-dir/vllm-tpu-qwen08/hf_cache"
export VLLM_XLA_CACHE_PATH="$HOME/work-dir/vllm-tpu-qwen08/xla_cache"
timeout 2400 vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen3.5-0.8B \
  --dataset-name random \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 8 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --seed 0 \
  --output-json results/qwen35_08b_baseline_random128_64_n8.json
```

Result: no throughput number. vLLM ignored `--input-len/--output-len` in favor of random defaults and failed before execution:

```text
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger than max_num_batched_tokens (8192). Please increase max_num_batched_tokens.
```

Corrected random-dataset command:

```bash
cd "$HOME/work-dir/vllm-tpu-qwen08"
source "$HOME/.local/bin/env"
source vllm_env/bin/activate
mkdir -p hf_cache xla_cache results
export HF_HOME="$HOME/work-dir/vllm-tpu-qwen08/hf_cache"
export VLLM_XLA_CACHE_PATH="$HOME/work-dir/vllm-tpu-qwen08/xla_cache"
timeout 2400 vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen3.5-0.8B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-prefix-len 0 \
  --num-prompts 8 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-batched-tokens 32768 \
  --seed 0 \
  --output-json results/qwen35_08b_baseline_random128_64_n8_v2.json
```

Result: no throughput number. vLLM loaded the model, but TPU compilation failed:

```text
Model architectures ['Qwen3_5ForConditionalGeneration'] not registered in tpu-inference.
Falling back to vLLM-native Pytorch definition.
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: E1000: CompileTimeHbmOom:
Ran out of memory in memory space hbm. Used 32.08G of 31.25G hbm.
```

I then tried constraining vLLM's compile range:

```bash
cd "$HOME/work-dir/vllm-tpu-qwen08"
source "$HOME/.local/bin/env"
source vllm_env/bin/activate
mkdir -p hf_cache xla_cache results
export HF_HOME="$HOME/work-dir/vllm-tpu-qwen08/hf_cache"
export VLLM_XLA_CACHE_PATH="$HOME/work-dir/vllm-tpu-qwen08/xla_cache"
timeout 2400 vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen3.5-0.8B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-prefix-len 0 \
  --num-prompts 8 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-batched-tokens 32768 \
  --compilation-config '{"compile_ranges_endpoints":[2048]}' \
  --seed 0 \
  --output-json results/qwen35_08b_baseline_random128_64_n8_compile2048.json
```

Result: no throughput number. This attempt was blocked because another process was using the TPU:

```text
ABORTED: The TPU is already in use by process with pid 161620.
```

The owner process was not from vLLM and was not killed:

```text
python3 benchmark_mtp1_engine.py --model Qwen/Qwen3.5-4B ... --output-json /tmp/worker_b_mixed_smoke.json
```

## Speculative decode status

No speculative throughput number was obtained because the baseline did not reach a successful run and a concurrent repo benchmark later occupied the TPU.

vLLM's generic speculative CLI uses `--speculative-config` JSON. The official docs list methods including `ngram`, `mtp`, and `eagle3`: https://docs.vllm.ai/en/latest/features/speculative_decoding/

The vLLM TPU supported-models page lists Qwen TPU support for `Qwen3ForCausalLM`, `Qwen2ForCausalLM`, and `Qwen2.5` variants, but not `Qwen3_5ForConditionalGeneration`: https://docs.vllm.ai/en/latest/models/hardware_supported_models/tpu.html

Observed local evidence agrees: `Qwen3_5ForConditionalGeneration` is not registered in `tpu-inference` 0.19.0 and falls back to a PyTorch definition on TPU. Therefore an exact vLLM TPU Qwen3.5-0.8B MTP comparison is not currently a clean supported baseline on this VM.

Closest speculative command to try only after a baseline works and the TPU is idle:

```bash
cd "$HOME/work-dir/vllm-tpu-qwen08"
source "$HOME/.local/bin/env"
source vllm_env/bin/activate
mkdir -p hf_cache xla_cache results
export HF_HOME="$HOME/work-dir/vllm-tpu-qwen08/hf_cache"
export VLLM_XLA_CACHE_PATH="$HOME/work-dir/vllm-tpu-qwen08/xla_cache"
timeout 2400 vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen3.5-0.8B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-prefix-len 0 \
  --num-prompts 8 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-batched-tokens 32768 \
  --compilation-config '{"compile_ranges_endpoints":[2048]}' \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
  --seed 0 \
  --output-json results/qwen35_08b_ngram_random128_64_n8_compile2048.json
```

Do not treat this as a Qwen MTP comparison. It is only a generic prompt n-gram speculative decode comparison.

## Closest valid comparison plan

1. Wait until no other TPU process owns libtpu.
2. Re-run the constrained baseline command above.
3. If it still OOMs or fails because Qwen3.5-0.8B is not JAX-native in `tpu-inference`, do not report vLLM TPU numbers for exact Qwen3.5-0.8B.
4. For a vLLM TPU-supported Qwen-family sanity comparison, use a supported text-only model from the vLLM TPU supported-model page, for example `Qwen/Qwen3-8B` or `Qwen/Qwen2.5-1.5B-Instruct`, and keep the same `vllm bench throughput` shape.
5. For speculative decode, first use `ngram` because it does not require a Qwen MTP draft head. Only claim MTP if vLLM TPU explicitly supports the target model's MTP path.

## Impact on main optimization work

This does not block the main K=1 mixed/heterogeneous optimization work. The blocker is external comparison availability: current vLLM TPU does not provide a clean exact Qwen3.5-0.8B baseline/speculative number on this v6e-1 VM, and the TPU became occupied by another benchmark.

## Narrowed supported-model candidate

Follow-up date: 2026-05-10

Chosen candidate: `Qwen/Qwen2.5-1.5B-Instruct`.

Reason:

```text
Resolved architecture: Qwen2ForCausalLM
```

This is a Qwen-family small dense model and is closer than switching families. The vLLM TPU supported-model table lists `Qwen/Qwen2.5-1.5B-Instruct` as runnable, and `tpu-inference` 0.19.0 registers `Qwen2ForCausalLM`. This is not the same architecture as the repo's current `Qwen/Qwen3.5-*` path: it does not exercise Qwen3.5 hybrid/MTP behavior and should be treated as a vLLM TPU sanity baseline, not as an exact Qwen3.5 MTP comparator.

Baseline command attempted:

```bash
cd "$HOME/work-dir/vllm-tpu-qwen08"
source "$HOME/.local/bin/env"
source vllm_env/bin/activate
mkdir -p hf_cache xla_cache results
export HF_HOME="$HOME/work-dir/vllm-tpu-qwen08/hf_cache"
export VLLM_XLA_CACHE_PATH="$HOME/work-dir/vllm-tpu-qwen08/xla_cache"
timeout 2400 vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-prefix-len 0 \
  --num-prompts 8 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-batched-tokens 2048 \
  --compilation-config '{"compile_ranges_endpoints":[2048]}' \
  --seed 0 \
  --output-json results/qwen25_15b_baseline_random128_64_n8.json
```

Result: no number obtained. The model resolved natively as `Qwen2ForCausalLM`, but vLLM could not acquire libtpu because another active repo benchmark owned the TPU:

```text
ABORTED: The TPU is already in use by process with pid 168923.
```

The owning process was:

```text
python3 benchmark_mtp1_engine.py --model Qwen/Qwen3.5-4B ... --output-json /tmp/worker_b_serving_runs/interleaved_prefill_decode_b4__baseline.json
```

I did not kill or interrupt that benchmark.

Exact baseline command to run when the TPU is idle is the command above.

Closest speculative command to run when the TPU is idle:

```bash
cd "$HOME/work-dir/vllm-tpu-qwen08"
source "$HOME/.local/bin/env"
source vllm_env/bin/activate
mkdir -p hf_cache xla_cache results
export HF_HOME="$HOME/work-dir/vllm-tpu-qwen08/hf_cache"
export VLLM_XLA_CACHE_PATH="$HOME/work-dir/vllm-tpu-qwen08/xla_cache"
timeout 2400 vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --random-prefix-len 0 \
  --num-prompts 8 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-batched-tokens 2048 \
  --compilation-config '{"compile_ranges_endpoints":[2048]}' \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
  --seed 0 \
  --output-json results/qwen25_15b_ngram_random128_64_n8.json
```

Comparison caveat: the speculative command uses vLLM generic prompt n-gram speculation. It is useful as a baseline-vs-speculative vLLM TPU measurement, but it is not comparable to this repo's Qwen3.5 MTP implementation except at a high level as "some speculative decoding on TPU".
