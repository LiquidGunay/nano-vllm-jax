#!/usr/bin/env python3
"""Run the GPU benchmark matrix from JSON configs.

The runner intentionally does not import JAX. It launches benchmark subprocesses
with `JAX_PLATFORMS=cuda` and all cache/temp paths rooted under
`/mountpoint/.exp` unless explicitly overridden by the environment.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from benchmarks.summarize_gpu_matrix import render_markdown

CONFIG_DIR = REPO_ROOT / "benchmarks" / "configs"
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CONFIGS = ("gpu_paged_default", "gpu_paged_fast_optin", "gpu_mtp_diagnostics")
DEFAULT_WORKLOADS = ("hetero8", "short_32_128", "long_prefill_512_2048", "decode_heavy_128x128")
DEFAULT_JAX_PYTHON = Path(
    os.environ.get(
        "NANO_VLLM_JAX_PYTHON",
        "/mountpoint/.exp/.venv/bin/python"
        if Path("/mountpoint/.exp/.venv/bin/python").exists()
        else sys.executable,
    )
)
DEFAULT_VLLM_PYTHON = Path("/mountpoint/.exp/vllm-venv/bin/python")
MIN_ACCEPTANCE_REPEATS = 2
TARGET_VLLM_RATIO = 0.9
FINAL_TARGET_WORKLOAD = "long_prefill_512_2048"
FINAL_TARGET_CONFIG = "gpu_paged_default"
FINAL_TARGET_DESCRIPTION = (
    "non-speculative Qwen/Qwen3.5-0.8B server on long heterogeneous mixed-shape "
    "requests must be speed-claim-ready and reach at least 0.9x vLLM throughput"
)
PROFILE_NEEDLES = (
    "PjRtCApiLoadedExecutable::Execute",
    "command_buffer::execute",
    "command_buffer::update",
    "forward_step_token_ids_jit",
    "gather",
    "transpose",
    "MemcpyD2D",
    "array.py:325 tolist",
    "np.asarray(jax.Array)",
)


@dataclass(frozen=True)
class Workload:
    name: str
    input_lens: str
    output_len: int
    prompt_suite: str
    arg_overrides: dict[str, Any]
    vllm_overrides: dict[str, Any]
    prompt_source: str = "tokenized_seed_repeat"
    dataset_name: str | None = None
    num_prompts: int | None = None
    seed: int = 0
    random_input_len: int | None = None
    random_output_len: int | None = None
    random_range_ratio: str | None = None
    acceptance_scope: str = "speed_claim"


WORKLOADS: dict[str, Workload] = {
    "hetero8": Workload(
        name="hetero8",
        input_lens="64,128,192,256,320,384,448,512",
        output_len=32,
        prompt_suite="mixed",
        arg_overrides={},
        vllm_overrides={"max_model_len": 1024, "gpu_memory_utilization": 0.65},
    ),
    "short_32_128": Workload(
        name="short_32_128",
        input_lens="32,64,96,128",
        output_len=32,
        prompt_suite="mixed",
        arg_overrides={
            "max_kv_cache_mb": 1024,
            "num_kvcache_blocks": 64,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 512,
            "prefill_buckets": "32,64,128",
            "batch_size_buckets": "1,2,4",
            "max_blocks_per_seq": 16,
        },
        vllm_overrides={"max_model_len": 256, "gpu_memory_utilization": 0.55},
    ),
    "long_prefill_512_2048": Workload(
        name="long_prefill_512_2048",
        input_lens="512,1024,1536,2048",
        output_len=16,
        prompt_suite="mixed",
        arg_overrides={
            "max_kv_cache_mb": 3072,
            "num_kvcache_blocks": 384,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "prefill_buckets": "512,1024,2048",
            "batch_size_buckets": "1,2,4",
            "max_blocks_per_seq": 129,
        },
        vllm_overrides={"max_model_len": 4096, "gpu_memory_utilization": 0.65},
    ),
    "decode_heavy_128x128": Workload(
        name="decode_heavy_128x128",
        input_lens="128",
        output_len=128,
        prompt_suite="mixed",
        arg_overrides={
            "max_kv_cache_mb": 1024,
            "num_kvcache_blocks": 128,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 512,
            "prefill_buckets": "128",
            "batch_size_buckets": "1",
            "max_blocks_per_seq": 24,
        },
        vllm_overrides={"max_model_len": 512, "gpu_memory_utilization": 0.55},
    ),
    "vllm_random_longprefill": Workload(
        name="vllm_random_longprefill",
        input_lens="1280",
        output_len=16,
        prompt_suite="mixed",
        arg_overrides={
            "max_kv_cache_mb": 3072,
            "num_kvcache_blocks": 768,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "prefill_buckets": "512,1024,2048",
            "batch_size_buckets": "1,2,4",
            "max_blocks_per_seq": 160,
        },
        vllm_overrides={"max_model_len": 4096, "gpu_memory_utilization": 0.65},
        prompt_source="vllm_random",
        dataset_name="random",
        num_prompts=128,
        seed=0,
        random_input_len=1280,
        random_output_len=16,
        random_range_ratio='{"input":0.6,"output":0.0}',
        acceptance_scope="sidecar_only",
    ),
    "vllm_random_longprefill_smoke": Workload(
        name="vllm_random_longprefill_smoke",
        input_lens="1280",
        output_len=16,
        prompt_suite="mixed",
        arg_overrides={
            "max_kv_cache_mb": 3072,
            "num_kvcache_blocks": 768,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "prefill_buckets": "512,1024,2048",
            "batch_size_buckets": "1,2,4",
            "max_blocks_per_seq": 160,
        },
        vllm_overrides={"max_model_len": 4096, "gpu_memory_utilization": 0.65},
        prompt_source="vllm_random",
        dataset_name="random",
        num_prompts=16,
        seed=0,
        random_input_len=1280,
        random_output_len=16,
        random_range_ratio='{"input":0.6,"output":0.0}',
        acceptance_scope="sidecar_only",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--report-md",
        default="",
        help=(
            "Markdown report path. Defaults to the output JSON path with .md suffix; "
            "use --no-report-md to skip writing it."
        ),
    )
    parser.add_argument("--no-report-md", dest="write_report_md", action="store_false", default=True)
    parser.add_argument("--report-top-profile-deltas", type=int, default=8)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-live-vllm", dest="live_vllm", action="store_false", default=True)
    parser.add_argument(
        "--goal-target-only",
        action="store_true",
        help=(
            "Run only the final non-speculative goal target "
            f"{FINAL_TARGET_WORKLOAD}/{FINAL_TARGET_CONFIG}."
        ),
    )
    parser.add_argument(
        "--require-speed-claim-ready",
        action="store_true",
        help=(
            "After writing the summary, exit nonzero unless every selected "
            "workload/config is speed-claim-ready and meets the target vLLM ratio."
        ),
    )
    parser.add_argument(
        "--require-goal-target-ready",
        action="store_true",
        help=(
            "After writing the summary, exit nonzero unless the final goal target "
            f"{FINAL_TARGET_WORKLOAD}/{FINAL_TARGET_CONFIG} is present, "
            "speed-claim-ready, and reaches the target vLLM ratio."
        ),
    )
    parser.add_argument(
        "--require-stored-references",
        action="store_true",
        help=(
            "Exit before launching benchmarks unless every selected workload/config "
            "has a stored JAX reference and every selected workload has a stored vLLM reference."
        ),
    )
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip the CUDA device availability check before launching real benchmark subprocesses.",
    )
    parser.add_argument(
        "--vllm-python",
        default=str(DEFAULT_VLLM_PYTHON if DEFAULT_VLLM_PYTHON.exists() else Path(sys.executable)),
    )
    parser.add_argument(
        "--jax-python",
        default=str(DEFAULT_JAX_PYTHON),
        help=(
            "Python interpreter for JAX benchmark subprocesses. Defaults to "
            "NANO_VLLM_JAX_PYTHON, then /mountpoint/.exp/.venv/bin/python if present, "
            "then the current interpreter."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _parse_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _selected_matrix_names(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.goal_target_only:
        return [FINAL_TARGET_CONFIG], [FINAL_TARGET_WORKLOAD]
    return _parse_names(args.configs), _parse_names(args.workloads)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validate_required_keys(row: dict[str, Any], required: list[str], path: str) -> None:
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{path} missing required keys: {', '.join(missing)}")


def _validate_summary_shape(summary: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate the matrix summary shape without depending on jsonschema."""

    _validate_required_keys(summary, list(schema.get("required") or []), "summary")
    if summary.get("schema_version") != schema.get("schema_version"):
        raise ValueError(
            f"summary schema_version={summary.get('schema_version')} does not match "
            f"schema={schema.get('schema_version')}"
        )
    workloads = summary.get("workloads")
    configs = summary.get("configs")
    if not isinstance(workloads, list) or not all(isinstance(item, str) for item in workloads):
        raise ValueError("summary.workloads must be a list of strings")
    if not isinstance(configs, list) or not all(isinstance(item, str) for item in configs):
        raise ValueError("summary.configs must be a list of strings")
    goal_target = summary.get("goal_target")
    if not isinstance(goal_target, dict):
        raise ValueError("summary.goal_target must be an object")
    _validate_required_keys(
        goal_target,
        ["workload", "config", "target_vllm_ratio", "description"],
        "summary.goal_target",
    )
    if not isinstance(summary.get("jax_default_references"), dict):
        raise ValueError("summary.jax_default_references must be an object")
    jax_python = summary.get("jax_python")
    if not isinstance(jax_python, dict):
        raise ValueError("summary.jax_python must be an object")
    _validate_required_keys(jax_python, ["path", "available"], "summary.jax_python")

    matrix_required = ["config", "repeats", "aggregate"]
    repeat_required = ["repeat", "artifact", "reference_json", "reference_source", "run", "metrics"]
    aggregate_required = [
        "repeat_count",
        "tokens_per_second_median",
        "ttft_ms_p50_median",
        "ttft_ms_p95_median",
        "itl_ms_p50_median",
        "itl_ms_p95_median",
        "first_forward_step_token_ids_jit_ms_median",
        "profile_medians",
        "all_correct",
        "all_exact_generated_token_match",
        "all_correctness_checked",
    ]
    acceptance_required = [
        "checks",
        "speed_claim_ready",
        "target_vllm_ratio",
        "target_vllm_ratio_met",
        "missing_profile_counters",
        "notes",
    ]
    acceptance_check_required = [
        "minimum_repeats",
        "runs_succeeded",
        "correctness_checked",
        "exact_generated_token_match",
        "jax_performance_present",
        "jax_latency_present",
        "first_forward_step_present",
        "vllm_reference_present",
        "profile_counters_present",
    ]
    for workload_name in workloads:
        if workload_name not in summary["matrix"]:
            raise ValueError(f"summary.matrix missing workload {workload_name}")
        if workload_name not in summary["comparisons"]:
            raise ValueError(f"summary.comparisons missing workload {workload_name}")
        if workload_name not in summary["acceptance"]:
            raise ValueError(f"summary.acceptance missing workload {workload_name}")
        for config_name in configs:
            path = f"summary.matrix.{workload_name}.{config_name}"
            matrix_row = summary["matrix"][workload_name].get(config_name)
            if not isinstance(matrix_row, dict):
                raise ValueError(f"{path} must be an object")
            _validate_required_keys(matrix_row, matrix_required, path)
            for index, repeat in enumerate(matrix_row["repeats"], start=1):
                if not isinstance(repeat, dict):
                    raise ValueError(f"{path}.repeats[{index}] must be an object")
                _validate_required_keys(repeat, repeat_required, f"{path}.repeats[{index}]")
            _validate_required_keys(matrix_row["aggregate"], aggregate_required, f"{path}.aggregate")

            acceptance = summary["acceptance"][workload_name].get(config_name)
            if not isinstance(acceptance, dict):
                raise ValueError(f"summary.acceptance.{workload_name}.{config_name} must be an object")
            _validate_required_keys(
                acceptance,
                acceptance_required,
                f"summary.acceptance.{workload_name}.{config_name}",
            )
            _validate_required_keys(
                acceptance["checks"],
                acceptance_check_required,
                f"summary.acceptance.{workload_name}.{config_name}.checks",
            )


def _runtime_env(config_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    root = Path(env.get("NANO_VLLM_JAX_CACHE_ROOT", "/mountpoint/.exp"))
    defaults = {
        "NANO_VLLM_JAX_CACHE_ROOT": str(root),
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / ".cache"),
        "XDG_DATA_HOME": str(root / ".local" / "share"),
        "UV_CACHE_DIR": str(root / ".cache" / "uv"),
        "PIP_CACHE_DIR": str(root / ".cache" / "pip"),
        "HF_HOME": str(root / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(root / ".cache" / "huggingface" / "hub"),
        "JAX_COMPILATION_CACHE_DIR": str(root / ".cache" / "jax"),
        "NANO_VLLM_JAX_COMPILE_CACHE_DIR": str(root / ".cache" / "jax"),
        "FLASHINFER_WORKSPACE_BASE": str(root),
        "FLASHINFER_CUBIN_DIR": str(root / ".cache" / "flashinfer" / "cubins"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "XLA_FLAGS": "--xla_gpu_autotune_level=4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        env[key] = value
    env.update({str(key): str(value) for key, value in config_env.items()})
    for key in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "UV_CACHE_DIR",
        "PIP_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "NANO_VLLM_JAX_COMPILE_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "FLASHINFER_CUBIN_DIR",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def _cuda_device_preflight(
    *,
    dev_dir: Path = Path("/dev"),
    runner: Any = subprocess.run,
) -> tuple[bool, str]:
    """Return whether this process can see an NVIDIA GPU before JAX import.

    The matrix runner intentionally avoids importing JAX. This preflight catches
    the common no-device/container case early, before each subprocess starts
    loading model weights and then fails during the first JAX array transfer.
    """

    nvidia_devices = sorted(dev_dir.glob("nvidia[0-9]*")) if dev_dir.exists() else []
    if (dev_dir / "nvidiactl").exists() and nvidia_devices:
        names = ", ".join(path.name for path in nvidia_devices[:4])
        return True, f"visible device nodes: {names}"
    try:
        completed: CompletedProcess[str] = runner(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia-smi is not installed and /dev/nvidia* device nodes are missing"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi -L timed out and /dev/nvidia* device nodes are missing"
    output = (completed.stdout or "").strip()
    if completed.returncode == 0 and output:
        return True, output.splitlines()[0]
    detail = output or f"nvidia-smi -L exited with {completed.returncode}"
    return False, f"{detail}; /dev/nvidia* device nodes are missing"


def _append_cli_arg(command: list[str], key: str, value: Any) -> None:
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if key == "warmup":
            command.append("--warmup" if value else "--no-warmup")
        elif key == "profile":
            command.append("--profile" if value else "--no-profile")
        elif value:
            command.append(flag)
        return
    if value is None:
        return
    command.extend([flag, str(value)])


def _effective_jax_args(config: dict[str, Any], workload: Workload) -> dict[str, Any]:
    args = dict(config.get("args", {}))
    args.update(workload.arg_overrides)
    args["input_lens"] = workload.input_lens
    args["output_len"] = workload.output_len
    args["prompt_suite"] = workload.prompt_suite
    args["prompt_source"] = workload.prompt_source
    if workload.dataset_name is not None:
        args["dataset_name"] = workload.dataset_name
    if workload.num_prompts is not None:
        args["num_prompts"] = workload.num_prompts
    args["seed"] = workload.seed
    if workload.random_input_len is not None:
        args["random_input_len"] = workload.random_input_len
    if workload.random_output_len is not None:
        args["random_output_len"] = workload.random_output_len
    if workload.random_range_ratio is not None:
        args["random_range_ratio"] = workload.random_range_ratio
    return args


def _jax_command(
    config: dict[str, Any],
    workload: Workload,
    output_json: Path,
    reference_json: Path | None,
    run_label: str,
    jax_python: Path,
) -> list[str]:
    args = _effective_jax_args(config, workload)
    args["output_json"] = str(output_json)
    args["run_label"] = run_label
    if reference_json is not None:
        args["reference_json"] = str(reference_json)
    command = [str(jax_python), str(REPO_ROOT / config["benchmark_script"])]
    for key, value in args.items():
        _append_cli_arg(command, key, value)
    return command


def _vllm_command(
    config: dict[str, Any],
    workload: Workload,
    output_json: Path,
    run_label: str,
    vllm_python: Path,
) -> list[str]:
    base_args = dict(config.get("args", {}))
    args: dict[str, Any] = {
        "model": base_args.get("model", "Qwen/Qwen3.5-0.8B"),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "execution": "async",
        "mode": "baseline",
        "input_lens": workload.input_lens,
        "output_len": workload.output_len,
        "prompt_suite": workload.prompt_suite,
        "prompt_source": workload.prompt_source,
        "top_k": base_args.get("top_k", 5),
        "output_json": str(output_json),
        "run_label": run_label,
        "trust_remote_code": True,
    }
    if workload.dataset_name is not None:
        args["dataset_name"] = workload.dataset_name
    if workload.num_prompts is not None:
        args["num_prompts"] = workload.num_prompts
    args["seed"] = workload.seed
    if workload.random_input_len is not None:
        args["random_input_len"] = workload.random_input_len
    if workload.random_output_len is not None:
        args["random_output_len"] = workload.random_output_len
    if workload.random_range_ratio is not None:
        args["random_range_ratio"] = workload.random_range_ratio
    args.update(workload.vllm_overrides)
    command = [str(vllm_python), str(REPO_ROOT / "benchmarks" / "benchmark_vllm_qwen35.py")]
    for key, value in args.items():
        _append_cli_arg(command, key, value)
    return command


def _run_command(command: list[str], env: dict[str, str], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "returncode": None, "command": command}
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "command": command,
        "output_tail": completed.stdout[-12000:],
    }


def _counter_ranges(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counters = artifact.get("profile_counters") or {}
    return counters.get("ranges") or {}


def _first_profile_event_ms(artifact: dict[str, Any], needle: str) -> float | None:
    counters = artifact.get("profile_counters") or {}
    trace_json = counters.get("trace_json_gz")
    if not trace_json:
        return None
    path = Path(trace_json)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            events = json.load(handle).get("traceEvents", [])
    except Exception:
        return None
    candidates = [
        event
        for event in events
        if event.get("dur") is not None and needle in str(event.get("name", ""))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda event: float(event.get("ts", 0.0)))
    return float(candidates[0]["dur"]) / 1000.0


def _metric_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact": str(path), "exists": False}
    artifact = _load_json(path)
    performance = artifact.get("performance") or {}
    correctness = artifact.get("correctness") or {}
    ranges = _counter_ranges(artifact)
    profile = {
        key: {
            "total_ms": (ranges.get(key) or {}).get("total_ms"),
            "count": (ranges.get(key) or {}).get("count"),
        }
        for key in PROFILE_NEEDLES
    }
    return {
        "artifact": str(path),
        "exists": True,
        "performance": {
            "tokens_per_second": performance.get("tokens_per_second"),
            "request_throughput": performance.get("request_throughput"),
            "output_token_throughput": performance.get("output_token_throughput"),
            "total_token_throughput": performance.get("total_token_throughput"),
            "total_input_tokens": performance.get("total_input_tokens"),
            "total_output_tokens": performance.get("total_output_tokens"),
            "ttft_ms_p50": performance.get("ttft_ms_p50"),
            "ttft_ms_p95": performance.get("ttft_ms_p95"),
            "itl_ms_p50": performance.get("itl_ms_p50"),
            "itl_ms_p95": performance.get("itl_ms_p95"),
            "generated_tokens": performance.get("generated_tokens"),
            "seconds": performance.get("seconds"),
        },
        "correctness": {
            "checked": correctness.get("checked"),
            "ok": correctness.get("ok"),
            "full_length_ok": correctness.get("full_length_ok"),
            "exact_generated_token_match": bool(
                correctness.get("checked") and correctness.get("full_length_ok")
            ),
            "rows": correctness.get("rows"),
        },
        "profile": profile,
        "first_forward_step_token_ids_jit_ms": _first_profile_event_ms(
            artifact,
            "forward_step_token_ids_jit",
        ),
        "profile_trace_json_gz": (artifact.get("profile_counters") or {}).get("trace_json_gz"),
        "run": artifact.get("run"),
        "speculative": artifact.get("speculative"),
        "mtp_admission": artifact.get("mtp_admission"),
    }


def _median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _aggregate_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [(row.get("metrics") or {}) for row in repeats]
    perf_rows = [row.get("performance", {}) for row in metric_rows]
    correctness_rows = [row.get("correctness", {}) for row in metric_rows]
    profile_medians = {}
    for needle in PROFILE_NEEDLES:
        total_values = []
        count_values = []
        for row in metric_rows:
            bucket = ((row.get("profile") or {}).get(needle) or {})
            if bucket.get("total_ms") is not None:
                total_values.append(bucket.get("total_ms"))
            if bucket.get("count") is not None:
                count_values.append(bucket.get("count"))
        profile_medians[needle] = {
            "total_ms_median": _median(total_values),
            "count_median": _median(count_values),
        }
    return {
        "repeat_count": len(repeats),
        "tokens_per_second_median": _median([row.get("tokens_per_second") for row in perf_rows]),
        "request_throughput_median": _median([row.get("request_throughput") for row in perf_rows]),
        "output_token_throughput_median": _median([row.get("output_token_throughput") for row in perf_rows]),
        "total_token_throughput_median": _median([row.get("total_token_throughput") for row in perf_rows]),
        "total_input_tokens_median": _median([row.get("total_input_tokens") for row in perf_rows]),
        "total_output_tokens_median": _median([row.get("total_output_tokens") for row in perf_rows]),
        "ttft_ms_p50_median": _median([row.get("ttft_ms_p50") for row in perf_rows]),
        "ttft_ms_p95_median": _median([row.get("ttft_ms_p95") for row in perf_rows]),
        "itl_ms_p50_median": _median([row.get("itl_ms_p50") for row in perf_rows]),
        "itl_ms_p95_median": _median([row.get("itl_ms_p95") for row in perf_rows]),
        "first_forward_step_token_ids_jit_ms_median": _median(
            [row.get("first_forward_step_token_ids_jit_ms") for row in metric_rows]
        ),
        "profile_medians": profile_medians,
        "all_correct": all(bool(row.get("ok")) for row in correctness_rows),
        "all_exact_generated_token_match": all(
            bool(row.get("exact_generated_token_match")) for row in correctness_rows
        ),
        "all_correctness_checked": all(bool(row.get("checked")) for row in correctness_rows),
    }


def _parse_input_lens(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _artifact_matches_workload(path: Path, workload: Workload) -> bool:
    if not path.exists():
        return False
    try:
        artifact = _load_json(path)
    except Exception:
        return False
    run_config = artifact.get("run_config") or {}
    prompt_source = run_config.get("prompt_source", "tokenized_seed_repeat")
    if prompt_source != workload.prompt_source:
        return False
    if workload.prompt_source == "vllm_random":
        if run_config.get("dataset_name") != (workload.dataset_name or "random"):
            return False
        if int(run_config.get("num_prompts") or 0) != int(workload.num_prompts or 0):
            return False
        if int(run_config.get("seed") or 0) != int(workload.seed):
            return False
        if int(run_config.get("random_input_len") or 0) != int(workload.random_input_len or 0):
            return False
        if int(run_config.get("random_output_len") or 0) != int(workload.random_output_len or workload.output_len):
            return False
        configured_ratio = json.loads(workload.random_range_ratio or '{"input":0.0,"output":0.0}')
        artifact_ratio = run_config.get("random_range_ratio") or {}
        return {
            "input": float(artifact_ratio.get("input", 0.0)),
            "output": float(artifact_ratio.get("output", 0.0)),
        } == {
            "input": float(configured_ratio.get("input", 0.0)),
            "output": float(configured_ratio.get("output", 0.0)),
        }
    input_lens = run_config.get("input_lens")
    output_len = run_config.get("output_len")
    prompt_suite = run_config.get("prompt_suite")
    if input_lens != _parse_input_lens(workload.input_lens):
        return False
    if output_len != workload.output_len:
        return False
    return prompt_suite in (None, workload.prompt_suite)


def _configured_workload_reference(
    config: dict[str, Any],
    workload: Workload,
    *,
    mapping_key: str,
    legacy_key: str,
) -> Path | None:
    mapping = config.get(mapping_key) or {}
    candidate_value = mapping.get(workload.name) if isinstance(mapping, dict) else None
    if candidate_value:
        candidate = Path(candidate_value)
        if _artifact_matches_workload(candidate, workload):
            return candidate
    legacy = Path(config.get(legacy_key, ""))
    if _artifact_matches_workload(legacy, workload):
        return legacy
    return None


def _stored_jax_reference_source(workload: Workload) -> str:
    if workload.name == "hetero8":
        return "stored_entry045"
    return "stored_jax_default"


def _reference_for(
    config_name: str,
    workload: Workload,
    repeat_index: int,
    stored_workload_reference: Path | None,
    generated_default_reference: Path | None,
    *,
    allow_unverified_generated_default: bool = False,
) -> tuple[Path | None, str]:
    if stored_workload_reference and _artifact_matches_workload(stored_workload_reference, workload):
        return stored_workload_reference, _stored_jax_reference_source(workload)
    if generated_default_reference and _artifact_matches_workload(generated_default_reference, workload):
        return generated_default_reference, "live_jax_default"
    if generated_default_reference and allow_unverified_generated_default:
        return generated_default_reference, "live_jax_default"
    if config_name == "gpu_paged_default" and repeat_index == 0:
        return None, "none"
    return None, "none"


def _find_local_vllm_reference(config: dict[str, Any], workload: Workload, reference_dir: Path) -> Path | None:
    configured = _configured_workload_reference(
        config,
        workload,
        mapping_key="workload_vllm_reference_jsons",
        legacy_key="vllm_reference_json",
    )
    if configured:
        return configured
    candidate = reference_dir / f"vllm_{workload.name}.json"
    return candidate if _artifact_matches_workload(candidate, workload) else None


def _stored_reference_gaps(
    configs: dict[str, dict[str, Any]],
    workloads: dict[str, Workload],
    reference_dir: Path,
) -> list[str]:
    gaps: list[str] = []
    vllm_config = configs.get("gpu_paged_default") or next(iter(configs.values()))
    for workload_name, workload in workloads.items():
        if _find_local_vllm_reference(vllm_config, workload, reference_dir) is None:
            gaps.append(f"{workload_name}: missing stored vLLM reference")
        for config_name, config in configs.items():
            if (
                _configured_workload_reference(
                    config,
                    workload,
                    mapping_key="workload_reference_jsons",
                    legacy_key="reference_json",
                )
                is None
            ):
                gaps.append(f"{workload_name}/{config_name}: missing stored JAX reference")
    return gaps


def _configs_missing_stored_jax_reference(
    configs: dict[str, dict[str, Any]],
    workload: Workload,
) -> list[str]:
    missing: list[str] = []
    for config_name, config in configs.items():
        if (
            _configured_workload_reference(
                config,
                workload,
                mapping_key="workload_reference_jsons",
                legacy_key="reference_json",
            )
            is None
        ):
            missing.append(config_name)
    return missing


def _should_capture_live_jax_default_reference(
    *,
    selected_configs: list[str],
    configs: dict[str, dict[str, Any]],
    workload: Workload,
    default_config: dict[str, Any],
) -> tuple[bool, list[str]]:
    missing = _configs_missing_stored_jax_reference(configs, workload)
    if not missing:
        return False, []
    if not bool(default_config.get("allow_live_jax_default_if_reference_missing", True)):
        return False, missing
    if workload.acceptance_scope == "sidecar_only":
        return True, missing
    if selected_configs and selected_configs[0] == FINAL_TARGET_CONFIG:
        return False, missing
    return True, missing


def _vllm_available(vllm_python: Path) -> bool:
    if vllm_python == Path(sys.executable):
        return importlib.util.find_spec("vllm") is not None
    if not vllm_python.exists():
        return False
    command = [
        str(vllm_python),
        "-c",
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _reference_metrics_for_comparison(
    repeats: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, str | None]:
    for row in repeats:
        reference_json = row.get("reference_json")
        if not reference_json:
            continue
        metrics = _metric_summary(Path(reference_json))
        if metrics.get("exists"):
            return metrics, row.get("reference_source") or "unknown", reference_json
    return None, "none", None


def _jax_available(jax_python: Path, *, runner: Any = subprocess.run) -> bool:
    """Check for the JAX package without importing JAX or selecting a backend."""

    if jax_python == Path(sys.executable):
        return importlib.util.find_spec("jax") is not None
    if not jax_python.exists():
        return False
    command = [
        str(jax_python),
        "-c",
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('jax') else 1)",
    ]
    completed = runner(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _ratio_or_none(current: float | int | None, reference: float | int | None) -> float | None:
    if current is None or reference in (None, 0):
        return None
    return float(current) / float(reference)


def _delta_or_none(current: float | int | None, reference: float | int | None) -> float | None:
    if current is None or reference is None:
        return None
    return float(current) - float(reference)


def _profile_delta_vs_reference(
    aggregate: dict[str, Any],
    jax_reference_metrics: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    current_profile = aggregate.get("profile_medians") or {}
    reference_profile = (jax_reference_metrics or {}).get("profile") or {}
    result: dict[str, dict[str, float | None]] = {}
    for needle in PROFILE_NEEDLES:
        current_bucket = current_profile.get(needle) or {}
        reference_bucket = reference_profile.get(needle) or {}
        current_total = current_bucket.get("total_ms_median")
        reference_total = reference_bucket.get("total_ms")
        current_count = current_bucket.get("count_median")
        reference_count = reference_bucket.get("count")
        result[needle] = {
            "current_total_ms_median": current_total,
            "reference_total_ms": reference_total,
            "total_ms_delta": _delta_or_none(current_total, reference_total),
            "total_ms_ratio": _ratio_or_none(current_total, reference_total),
            "current_count_median": current_count,
            "reference_count": reference_count,
            "count_delta": _delta_or_none(current_count, reference_count),
        }
    return result


def _comparison_summary(
    aggregate: dict[str, Any],
    vllm_metrics: dict[str, Any] | None,
    vllm_source: str,
    jax_reference_metrics: dict[str, Any] | None = None,
    jax_reference_source: str = "none",
    jax_reference_artifact: str | None = None,
) -> dict[str, Any]:
    jax_tps = aggregate.get("tokens_per_second_median")
    jax_ttft = aggregate.get("ttft_ms_p50_median")
    jax_itl = aggregate.get("itl_ms_p50_median")
    vllm_performance = (vllm_metrics or {}).get("performance") or {}
    vllm_tps = vllm_performance.get("tokens_per_second")
    vllm_ttft = vllm_performance.get("ttft_ms_p50")
    vllm_itl = vllm_performance.get("itl_ms_p50")
    reference_performance = (jax_reference_metrics or {}).get("performance") or {}
    reference_tps = reference_performance.get("tokens_per_second")
    reference_ttft = reference_performance.get("ttft_ms_p50")
    reference_itl = reference_performance.get("itl_ms_p50")
    target_tps = (float(vllm_tps) * TARGET_VLLM_RATIO) if vllm_tps else None
    gap_tps = (
        max(0.0, float(target_tps) - float(jax_tps))
        if target_tps is not None and jax_tps is not None
        else None
    )
    required_speedup = (
        max(1.0, float(target_tps) / float(jax_tps))
        if target_tps is not None and jax_tps
        else None
    )
    return {
        "jax_tokens_per_second_median": jax_tps,
        "vllm_tokens_per_second": vllm_tps,
        "jax_over_vllm_throughput": (jax_tps / vllm_tps) if jax_tps and vllm_tps else None,
        "jax_request_throughput_median": aggregate.get("request_throughput_median"),
        "vllm_request_throughput": vllm_performance.get("request_throughput"),
        "jax_output_token_throughput_median": aggregate.get("output_token_throughput_median"),
        "vllm_output_token_throughput": vllm_performance.get("output_token_throughput"),
        "jax_total_token_throughput_median": aggregate.get("total_token_throughput_median"),
        "vllm_total_token_throughput": vllm_performance.get("total_token_throughput"),
        "jax_over_vllm_total_token_throughput": _ratio_or_none(
            aggregate.get("total_token_throughput_median"),
            vllm_performance.get("total_token_throughput"),
        ),
        "target_vllm_ratio": TARGET_VLLM_RATIO,
        "target_tokens_per_second": target_tps,
        "tokens_per_second_gap_to_target": gap_tps,
        "required_jax_speedup_to_target": required_speedup,
        "ttft_ms_p50_delta_vs_vllm": (jax_ttft - vllm_ttft) if jax_ttft is not None and vllm_ttft is not None else None,
        "itl_ms_p50_delta_vs_vllm": (jax_itl - vllm_itl) if jax_itl is not None and vllm_itl is not None else None,
        "vllm_reference_source": vllm_source,
        "jax_reference_source": jax_reference_source,
        "jax_reference_artifact": jax_reference_artifact,
        "jax_reference_tokens_per_second": reference_tps,
        "jax_over_jax_reference_throughput": _ratio_or_none(jax_tps, reference_tps),
        "tokens_per_second_delta_vs_jax_reference": _delta_or_none(jax_tps, reference_tps),
        "ttft_ms_p50_delta_vs_jax_reference": _delta_or_none(jax_ttft, reference_ttft),
        "itl_ms_p50_delta_vs_jax_reference": _delta_or_none(jax_itl, reference_itl),
        "profile_delta_vs_jax_reference": _profile_delta_vs_reference(
            aggregate,
            jax_reference_metrics,
        ),
    }


def _missing_profile_counters(repeats: list[dict[str, Any]]) -> list[str]:
    missing: set[str] = set()
    for repeat_index, row in enumerate(repeats, start=1):
        metrics = row.get("metrics") or {}
        profile = metrics.get("profile") or {}
        for needle in PROFILE_NEEDLES:
            bucket = profile.get(needle) or {}
            if bucket.get("total_ms") is None or bucket.get("count") is None:
                missing.add(f"repeat{repeat_index}:{needle}")
    return sorted(missing)


def _benchmark_acceptance_summary(
    repeats: list[dict[str, Any]],
    aggregate: dict[str, Any],
    comparison: dict[str, Any],
    vllm_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    vllm_performance = (vllm_metrics or {}).get("performance") or {}
    jax_ratio = comparison.get("jax_over_vllm_throughput")
    missing_profile_counters = _missing_profile_counters(repeats)
    checks = {
        "runs_succeeded": all((row.get("run") or {}).get("status") == "ok" for row in repeats),
        "minimum_repeats": int(aggregate.get("repeat_count") or 0) >= MIN_ACCEPTANCE_REPEATS,
        "correctness_checked": bool(aggregate.get("all_correctness_checked")),
        "exact_generated_token_match": bool(aggregate.get("all_exact_generated_token_match")),
        "jax_performance_present": aggregate.get("tokens_per_second_median") is not None,
        "jax_latency_present": all(
            aggregate.get(key) is not None
            for key in (
                "ttft_ms_p50_median",
                "ttft_ms_p95_median",
                "itl_ms_p50_median",
                "itl_ms_p95_median",
            )
        ),
        "first_forward_step_present": aggregate.get("first_forward_step_token_ids_jit_ms_median") is not None,
        "vllm_reference_present": vllm_performance.get("tokens_per_second") is not None,
        "profile_counters_present": not missing_profile_counters,
    }
    speed_claim_ready = all(checks.values())
    return {
        "checks": checks,
        "speed_claim_ready": speed_claim_ready,
        "target_vllm_ratio": TARGET_VLLM_RATIO,
        "target_vllm_ratio_met": bool(jax_ratio is not None and jax_ratio >= TARGET_VLLM_RATIO),
        "missing_profile_counters": missing_profile_counters,
        "notes": (
            "profile bucket movement still needs human explanation in the logbook"
            if speed_claim_ready
            else "not enough evidence for a performance claim"
        ),
    }


def _acceptance_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for workload_name in summary.get("workloads") or []:
        workload_acceptance = (summary.get("acceptance") or {}).get(workload_name) or {}
        for config_name in summary.get("configs") or []:
            acceptance = workload_acceptance.get(config_name) or {}
            checks = acceptance.get("checks") or {}
            failed_checks = sorted(key for key, value in checks.items() if not value)
            parts: list[str] = []
            if failed_checks:
                parts.append("failed checks: " + ",".join(failed_checks))
            if not acceptance.get("speed_claim_ready"):
                parts.append("speed_claim_ready=false")
            if not acceptance.get("target_vllm_ratio_met"):
                parts.append(
                    f"target_vllm_ratio_met=false target={acceptance.get('target_vllm_ratio')}"
                )
            missing_profile_counters = acceptance.get("missing_profile_counters") or []
            if missing_profile_counters:
                parts.append(f"missing_profile_counters={len(missing_profile_counters)}")
            if parts:
                failures.append(f"{workload_name}/{config_name}: " + "; ".join(parts))
    return failures


def _format_acceptance_failure(workload_name: str, config_name: str, acceptance: dict[str, Any]) -> str:
    checks = acceptance.get("checks") or {}
    failed_checks = sorted(key for key, value in checks.items() if not value)
    parts: list[str] = []
    if failed_checks:
        parts.append("failed checks: " + ",".join(failed_checks))
    if not acceptance.get("speed_claim_ready"):
        parts.append("speed_claim_ready=false")
    if not acceptance.get("target_vllm_ratio_met"):
        parts.append(
            f"target_vllm_ratio_met=false target={acceptance.get('target_vllm_ratio')}"
        )
    missing_profile_counters = acceptance.get("missing_profile_counters") or []
    if missing_profile_counters:
        parts.append(f"missing_profile_counters={len(missing_profile_counters)}")
    return f"{workload_name}/{config_name}: " + "; ".join(parts)


def _goal_target_failure(summary: dict[str, Any]) -> str | None:
    target = summary.get("goal_target") or {}
    workload_name = str(target.get("workload") or FINAL_TARGET_WORKLOAD)
    config_name = str(target.get("config") or FINAL_TARGET_CONFIG)
    if workload_name not in (summary.get("workloads") or []):
        return (
            f"{workload_name}/{config_name}: final goal target workload is not in "
            "this matrix summary"
        )
    if config_name not in (summary.get("configs") or []):
        return (
            f"{workload_name}/{config_name}: final goal target config is not in "
            "this matrix summary"
        )
    acceptance = ((summary.get("acceptance") or {}).get(workload_name) or {}).get(config_name)
    if not isinstance(acceptance, dict):
        return f"{workload_name}/{config_name}: final goal target acceptance is missing"
    if acceptance.get("speed_claim_ready") and acceptance.get("target_vllm_ratio_met"):
        return None
    return _format_acceptance_failure(workload_name, config_name, acceptance)


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = Path(args.run_dir) if args.run_dir else RESULTS_DIR / "gpu_matrix_runs" / timestamp
    output_json = Path(args.output_json) if args.output_json else RESULTS_DIR / f"gpu_matrix_{timestamp}.json"
    report_md = Path(args.report_md) if args.report_md else output_json.with_suffix(".md")
    if not args.dry_run and not args.skip_gpu_preflight:
        gpu_ok, gpu_detail = _cuda_device_preflight()
        if not gpu_ok:
            raise SystemExit(
                "CUDA GPU preflight failed: "
                f"{gpu_detail}. Matrix runs are GPU-only; restore NVIDIA "
                "device visibility before rerunning, or pass "
                "--skip-gpu-preflight only for controlled failure diagnostics."
            )
    jax_python = Path(args.jax_python)
    jax_is_available = _jax_available(jax_python)
    if not args.dry_run and not jax_is_available:
        raise SystemExit(
            "JAX Python preflight failed: "
            f"{jax_python} does not have the jax package visible. Pass "
            "--jax-python /path/to/python or set NANO_VLLM_JAX_PYTHON. "
            "The check uses importlib.util.find_spec and does not import JAX."
        )
    vllm_python = Path(args.vllm_python)
    vllm_is_available = _vllm_available(vllm_python)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    selected_configs, selected_workloads = _selected_matrix_names(args)
    configs = {name: _load_json(CONFIG_DIR / f"{name}.json") for name in selected_configs}
    workloads = {name: WORKLOADS[name] for name in selected_workloads}
    reference_dir = run_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    if args.require_stored_references:
        reference_gaps = _stored_reference_gaps(configs, workloads, reference_dir)
        if reference_gaps:
            raise SystemExit("Stored reference coverage failed:\n- " + "\n- ".join(reference_gaps))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": timestamp,
        "dry_run": bool(args.dry_run),
        "repeats": int(args.repeats),
        "run_dir": str(run_dir),
        "output_json": str(output_json),
        "report_md": str(report_md) if args.write_report_md else None,
        "configs": selected_configs,
        "workloads": selected_workloads,
        "required_metrics": [
            "total tok/s",
            "request throughput",
            "output-token throughput",
            "total-token throughput",
            "TTFT p50/p95",
            "ITL p50/p95",
            "exact generated-token match vs baseline",
            "PjRt Execute total/count",
            "command_buffer::execute total/count",
            "command_buffer::update total/count",
            "forward_step_token_ids_jit total/count",
            "first forward_step_token_ids_jit",
            "gather total",
            "transpose total",
            "MemcpyD2D total",
            "tolist / np.asarray sync attribution",
        ],
        "goal_target": {
            "workload": FINAL_TARGET_WORKLOAD,
            "config": FINAL_TARGET_CONFIG,
            "target_vllm_ratio": TARGET_VLLM_RATIO,
            "description": FINAL_TARGET_DESCRIPTION,
        },
        "jax_python": {
            "path": str(jax_python),
            "available": bool(jax_is_available),
        },
        "jax_default_references": {},
        "matrix": {},
        "vllm_references": {},
        "comparisons": {},
        "acceptance": {},
    }

    for workload_name, workload in workloads.items():
        generated_default_reference: Path | None = None
        summary["matrix"][workload_name] = {}
        summary["comparisons"][workload_name] = {}
        summary["acceptance"][workload_name] = {}
        default_config = configs.get(FINAL_TARGET_CONFIG) or _load_json(
            CONFIG_DIR / f"{FINAL_TARGET_CONFIG}.json"
        )
        capture_live_default, missing_jax_refs = _should_capture_live_jax_default_reference(
            selected_configs=selected_configs,
            configs=configs,
            workload=workload,
            default_config=default_config,
        )
        if capture_live_default:
            generated_default_reference = reference_dir / f"jax_default_{workload_name}.json"
            default_command = _jax_command(
                default_config,
                workload,
                generated_default_reference,
                None,
                f"gpu_matrix_jax_default_reference_{workload_name}_{timestamp}",
                jax_python,
            )
            default_result = _run_command(
                default_command,
                _runtime_env(default_config.get("env", {})),
                dry_run=args.dry_run,
            )
            if not args.dry_run and default_result["status"] != "ok" and not args.continue_on_error:
                raise SystemExit(
                    f"JAX default reference failed for {workload_name}: "
                    f"{default_result.get('output_tail', '')}"
                )
            summary["jax_default_references"][workload_name] = {
                "source": "dry_run" if args.dry_run else "live_jax_default",
                "artifact": str(generated_default_reference),
                "missing_selected_config_references": missing_jax_refs,
                "run": default_result,
                "metrics": _metric_summary(generated_default_reference)
                if generated_default_reference.exists()
                else None,
            }
        else:
            summary["jax_default_references"][workload_name] = {
                "source": "stored" if not missing_jax_refs else "none",
                "artifact": None,
                "missing_selected_config_references": missing_jax_refs,
                "run": None,
                "metrics": None,
            }
        vllm_config = configs.get("gpu_paged_default") or next(iter(configs.values()))
        vllm_reference = _find_local_vllm_reference(vllm_config, workload, reference_dir)
        if vllm_reference is None and args.live_vllm and bool(vllm_config.get("allow_live_vllm_if_reference_missing", True)):
            vllm_reference = reference_dir / f"vllm_{workload_name}.json"
            vllm_command = _vllm_command(
                vllm_config,
                workload,
                vllm_reference,
                f"gpu_matrix_vllm_{workload_name}_{timestamp}",
                vllm_python,
            )
            vllm_result = _run_command(vllm_command, _runtime_env({}), dry_run=args.dry_run or not vllm_is_available)
            if not args.dry_run and vllm_result["status"] != "ok" and not args.continue_on_error:
                raise SystemExit(f"vLLM reference failed for {workload_name}: {vllm_result.get('output_tail', '')}")
            summary["vllm_references"][workload_name] = {
                "source": "live" if vllm_is_available else "missing_vllm",
                "python": str(vllm_python),
                "run": vllm_result,
                "metrics": _metric_summary(vllm_reference) if vllm_reference.exists() else None,
            }
        else:
            summary["vllm_references"][workload_name] = {
                "source": "stored" if vllm_reference is not None else "none",
                "artifact": str(vllm_reference) if vllm_reference is not None else None,
                "metrics": _metric_summary(vllm_reference) if vllm_reference is not None else None,
            }

        for config_name, config in configs.items():
            config_repeats = []
            stored_workload_reference = _configured_workload_reference(
                config,
                workload,
                mapping_key="workload_reference_jsons",
                legacy_key="reference_json",
            )
            for repeat_index in range(int(args.repeats)):
                output_path = run_dir / f"{workload_name}_{config_name}_repeat{repeat_index + 1}.json"
                reference_path, reference_source = _reference_for(
                    config_name,
                    workload,
                    repeat_index,
                    stored_workload_reference,
                    generated_default_reference,
                    allow_unverified_generated_default=bool(args.dry_run),
                )
                run_label = f"gpu_matrix_{workload_name}_{config_name}_r{repeat_index + 1}_{timestamp}"
                command = _jax_command(
                    config,
                    workload,
                    output_path,
                    reference_path,
                    run_label,
                    jax_python,
                )
                env = _runtime_env(config.get("env", {}))
                result = _run_command(command, env, dry_run=args.dry_run)
                metrics = _metric_summary(output_path) if output_path.exists() else None
                config_repeats.append(
                    {
                        "repeat": repeat_index + 1,
                        "artifact": str(output_path),
                        "reference_json": str(reference_path) if reference_path else None,
                        "reference_source": reference_source,
                        "run": result,
                        "metrics": metrics,
                    }
                )
                if not args.dry_run and result["status"] != "ok" and not args.continue_on_error:
                    raise SystemExit(f"JAX run failed for {workload_name}/{config_name}: {result.get('output_tail', '')}")
                if config_name == "gpu_paged_default" and generated_default_reference is None and output_path.exists():
                    generated_default_reference = output_path
            summary["matrix"][workload_name][config_name] = {
                "config": {
                    "description": config.get("description"),
                    "env": config.get("env", {}),
                    "args": _effective_jax_args(config, workload),
                },
                "workload": {
                    "prompt_source": workload.prompt_source,
                    "dataset_name": workload.dataset_name,
                    "num_prompts": workload.num_prompts,
                    "seed": workload.seed,
                    "random_input_len": workload.random_input_len,
                    "random_output_len": workload.random_output_len,
                    "random_range_ratio": workload.random_range_ratio,
                    "acceptance_scope": workload.acceptance_scope,
                },
                "repeats": config_repeats,
                "aggregate": _aggregate_repeats(config_repeats),
            }
            vllm_metrics = (summary["vllm_references"].get(workload_name) or {}).get("metrics")
            (
                jax_reference_metrics,
                jax_reference_source,
                jax_reference_artifact,
            ) = _reference_metrics_for_comparison(config_repeats)
            comparison = _comparison_summary(
                summary["matrix"][workload_name][config_name]["aggregate"],
                vllm_metrics,
                (summary["vllm_references"].get(workload_name) or {}).get("source", "none"),
                jax_reference_metrics,
                jax_reference_source,
                jax_reference_artifact,
            )
            summary["comparisons"][workload_name][config_name] = comparison
            summary["acceptance"][workload_name][config_name] = _benchmark_acceptance_summary(
                config_repeats,
                summary["matrix"][workload_name][config_name]["aggregate"],
                comparison,
                vllm_metrics,
            )

    _validate_summary_shape(summary, _load_json(CONFIG_DIR / "gpu_matrix_summary_schema.json"))
    output_json.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_json)
    if args.write_report_md:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(
            render_markdown(summary, top_profile_deltas=args.report_top_profile_deltas),
            encoding="utf-8",
        )
        print(report_md)
    if args.require_speed_claim_ready:
        failures = _acceptance_failures(summary)
        if failures:
            raise SystemExit("GPU matrix acceptance failed:\n- " + "\n- ".join(failures))
    if args.require_goal_target_ready:
        failure = _goal_target_failure(summary)
        if failure:
            raise SystemExit("GPU matrix final goal target failed:\n- " + failure)


if __name__ == "__main__":
    main()
