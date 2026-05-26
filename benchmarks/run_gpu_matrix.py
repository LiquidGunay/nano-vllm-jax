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
CONFIG_DIR = REPO_ROOT / "benchmarks" / "configs"
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CONFIGS = ("gpu_paged_default", "gpu_paged_fast_optin", "gpu_mtp_diagnostics")
DEFAULT_VLLM_PYTHON = Path("/mountpoint/.exp/vllm-venv/bin/python")
MIN_ACCEPTANCE_REPEATS = 2
TARGET_VLLM_RATIO = 0.75
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
            "num_kvcache_blocks": 768,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "prefill_buckets": "512,1024,2048",
            "batch_size_buckets": "1,2,4",
            "max_blocks_per_seq": 160,
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-live-vllm", dest="live_vllm", action="store_false", default=True)
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip the CUDA device availability check before launching real benchmark subprocesses.",
    )
    parser.add_argument(
        "--vllm-python",
        default=str(DEFAULT_VLLM_PYTHON if DEFAULT_VLLM_PYTHON.exists() else Path(sys.executable)),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _parse_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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
        env.setdefault(key, value)
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
    return args


def _jax_command(
    config: dict[str, Any],
    workload: Workload,
    output_json: Path,
    reference_json: Path | None,
    run_label: str,
) -> list[str]:
    args = _effective_jax_args(config, workload)
    args["output_json"] = str(output_json)
    args["run_label"] = run_label
    if reference_json is not None:
        args["reference_json"] = str(reference_json)
    command = [sys.executable, str(REPO_ROOT / config["benchmark_script"])]
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
        "top_k": base_args.get("top_k", 5),
        "output_json": str(output_json),
        "run_label": run_label,
        "trust_remote_code": True,
    }
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
    return {
        "repeat_count": len(repeats),
        "tokens_per_second_median": _median([row.get("tokens_per_second") for row in perf_rows]),
        "ttft_ms_p50_median": _median([row.get("ttft_ms_p50") for row in perf_rows]),
        "itl_ms_p50_median": _median([row.get("itl_ms_p50") for row in perf_rows]),
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
) -> tuple[Path | None, str]:
    if stored_workload_reference and _artifact_matches_workload(stored_workload_reference, workload):
        return stored_workload_reference, _stored_jax_reference_source(workload)
    if generated_default_reference and _artifact_matches_workload(generated_default_reference, workload):
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


def _comparison_summary(
    aggregate: dict[str, Any],
    vllm_metrics: dict[str, Any] | None,
    vllm_source: str,
) -> dict[str, Any]:
    jax_tps = aggregate.get("tokens_per_second_median")
    jax_ttft = aggregate.get("ttft_ms_p50_median")
    jax_itl = aggregate.get("itl_ms_p50_median")
    vllm_performance = (vllm_metrics or {}).get("performance") or {}
    vllm_tps = vllm_performance.get("tokens_per_second")
    vllm_ttft = vllm_performance.get("ttft_ms_p50")
    vllm_itl = vllm_performance.get("itl_ms_p50")
    return {
        "jax_tokens_per_second_median": jax_tps,
        "vllm_tokens_per_second": vllm_tps,
        "jax_over_vllm_throughput": (jax_tps / vllm_tps) if jax_tps and vllm_tps else None,
        "ttft_ms_p50_delta_vs_vllm": (jax_ttft - vllm_ttft) if jax_ttft is not None and vllm_ttft is not None else None,
        "itl_ms_p50_delta_vs_vllm": (jax_itl - vllm_itl) if jax_itl is not None and vllm_itl is not None else None,
        "vllm_reference_source": vllm_source,
    }


def _has_profile_counters(repeats: list[dict[str, Any]]) -> bool:
    for row in repeats:
        metrics = row.get("metrics") or {}
        profile = metrics.get("profile") or {}
        for bucket in profile.values():
            if bucket.get("total_ms") is not None and bucket.get("count") is not None:
                return True
    return False


def _benchmark_acceptance_summary(
    repeats: list[dict[str, Any]],
    aggregate: dict[str, Any],
    comparison: dict[str, Any],
    vllm_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    vllm_performance = (vllm_metrics or {}).get("performance") or {}
    jax_ratio = comparison.get("jax_over_vllm_throughput")
    checks = {
        "minimum_repeats": int(aggregate.get("repeat_count") or 0) >= MIN_ACCEPTANCE_REPEATS,
        "correctness_checked": bool(aggregate.get("all_correctness_checked")),
        "exact_generated_token_match": bool(aggregate.get("all_exact_generated_token_match")),
        "jax_performance_present": aggregate.get("tokens_per_second_median") is not None,
        "vllm_reference_present": vllm_performance.get("tokens_per_second") is not None,
        "profile_counters_present": _has_profile_counters(repeats),
    }
    speed_claim_ready = all(checks.values())
    return {
        "checks": checks,
        "speed_claim_ready": speed_claim_ready,
        "target_vllm_ratio": TARGET_VLLM_RATIO,
        "target_vllm_ratio_met": bool(jax_ratio is not None and jax_ratio >= TARGET_VLLM_RATIO),
        "notes": (
            "profile bucket movement still needs human explanation in the logbook"
            if speed_claim_ready
            else "not enough evidence for a performance claim"
        ),
    }


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = Path(args.run_dir) if args.run_dir else RESULTS_DIR / "gpu_matrix_runs" / timestamp
    output_json = Path(args.output_json) if args.output_json else RESULTS_DIR / f"gpu_matrix_{timestamp}.json"
    if not args.dry_run and not args.skip_gpu_preflight:
        gpu_ok, gpu_detail = _cuda_device_preflight()
        if not gpu_ok:
            raise SystemExit(
                "CUDA GPU preflight failed: "
                f"{gpu_detail}. Matrix runs are GPU-only; restore NVIDIA "
                "device visibility before rerunning, or pass "
                "--skip-gpu-preflight only for controlled failure diagnostics."
            )
    vllm_python = Path(args.vllm_python)
    vllm_is_available = _vllm_available(vllm_python)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    selected_configs = _parse_names(args.configs)
    selected_workloads = _parse_names(args.workloads)
    configs = {name: _load_json(CONFIG_DIR / f"{name}.json") for name in selected_configs}
    workloads = {name: WORKLOADS[name] for name in selected_workloads}
    reference_dir = run_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": timestamp,
        "dry_run": bool(args.dry_run),
        "repeats": int(args.repeats),
        "run_dir": str(run_dir),
        "output_json": str(output_json),
        "configs": selected_configs,
        "workloads": selected_workloads,
        "required_metrics": [
            "total tok/s",
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
                )
                run_label = f"gpu_matrix_{workload_name}_{config_name}_r{repeat_index + 1}_{timestamp}"
                command = _jax_command(config, workload, output_path, reference_path, run_label)
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
                "repeats": config_repeats,
                "aggregate": _aggregate_repeats(config_repeats),
            }
            vllm_metrics = (summary["vllm_references"].get(workload_name) or {}).get("metrics")
            comparison = _comparison_summary(
                summary["matrix"][workload_name][config_name]["aggregate"],
                vllm_metrics,
                (summary["vllm_references"].get(workload_name) or {}).get("source", "none"),
            )
            summary["comparisons"][workload_name][config_name] = comparison
            summary["acceptance"][workload_name][config_name] = _benchmark_acceptance_summary(
                config_repeats,
                summary["matrix"][workload_name][config_name]["aggregate"],
                comparison,
                vllm_metrics,
            )

    output_json.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
