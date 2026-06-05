#!/usr/bin/env python3
"""Random-request sidecar comparing vLLM and the nano-vllm-jax server path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))
from nanovllm_jax.server_config import engine_overrides_from_config, runtime_env_from_config

DEFAULT_OUTPUT_JSON = str(
    REPO_ROOT.parent
    / "diagnostics"
    / "nano-vllm-jax"
    / "random_request_sidecar"
    / "qwen08_random_request_sidecar.json"
)
DEFAULT_JAX_PYTHON = (
    os.environ.get(
        "NANO_VLLM_JAX_PYTHON",
        str(REPO_ROOT / ".venv" / "bin" / "python"),
    )
    if (REPO_ROOT / ".venv" / "bin" / "python").exists()
    else sys.executable
)
DEFAULT_VLLM_PYTHON = (
    str(REPO_ROOT / ".venv" / "bin" / "python") if (REPO_ROOT / ".venv" / "bin" / "python").exists() else sys.executable
)
if (REPO_ROOT.parent / "vllm-venv" / "bin" / "python").exists():
    DEFAULT_VLLM_PYTHON = str(REPO_ROOT.parent / "vllm-venv" / "bin" / "python")
DEFAULT_WORKER_CPU_CORES = int(
    os.environ.get(
        "NANO_VLLM_JAX_RANDOM_WORKER_CPU_CORES",
        str(min(8, max(1, (os.cpu_count() or 2) // 2))),
    )
)
DEFAULT_WORKER_NICE = int(os.environ.get("NANO_VLLM_JAX_RANDOM_WORKER_NICE", "10"))
DEFAULT_MAX_SYSTEM_RAM_PERCENT = float(
    os.environ.get("NANO_VLLM_JAX_RANDOM_MAX_SYSTEM_RAM_PERCENT", "70")
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--weight-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--jax-execution", default="jit", choices=["eager", "decode-jit", "jit"])
    parser.add_argument("--vllm-execution", default="async", choices=["async", "offline"])
    parser.add_argument(
        "--vllm-dtype",
        default="",
        choices=["", "float16", "bfloat16", "float32", "auto"],
        help="vLLM dtype override. Defaults to bfloat16 when JAX dtype is float32.",
    )
    parser.add_argument("--prompt-suite", default="mixed")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--run-label", default="random_request_sidecar")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-json", default="", help="Optional generated-token reference for both JAX and vLLM.")
    parser.add_argument("--output-json-jax", default="")
    parser.add_argument("--output-json-vllm", default="")
    parser.add_argument("--manifest-jsonl", default="")
    parser.add_argument("--prompt-manifest-output-jsonl", default="")
    parser.add_argument("--jax-python", default=DEFAULT_JAX_PYTHON)
    parser.add_argument(
        "--jax-config",
        default="",
        help=(
            "Optional JSON/YAML config whose runtime/kernels sections supply "
            "JAX subprocess env plus accepted engine fastpath/kernel args."
        ),
    )
    parser.add_argument("--vllm-python", default=DEFAULT_VLLM_PYTHON)
    parser.add_argument("--run-log", default="")

    parser.add_argument("--min-input-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--min-output-tokens", type=int, default=256)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--min-request-count", type=int, default=5)
    parser.add_argument("--max-request-count", type=int, default=15)

    parser.add_argument("--jax-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--jax-warmup-mode",
        choices=["generic", "request"],
        default="generic",
        help="generic compiles configured server buckets; request replays the measured manifest for diagnostics only.",
    )
    parser.add_argument(
        "--jax-fail-on-jit-cache-growth",
        action="store_true",
        help="Fail the JAX run if new executor JIT keys are created during measured generation.",
    )
    parser.add_argument("--jax-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jax-trace-profile-dir", default="")
    parser.add_argument("--jax-max-kv-cache-mb", type=int, default=3072)
    parser.add_argument("--jax-num-kvcache-blocks", type=int, default=320)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--max-num-resident-seqs",
        type=int,
        default=0,
        help="Resident JAX request capacity; 0 keeps it equal to --max-num-seqs.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--prefill-buckets", default="128,256,512,1024,2048,4096")
    parser.add_argument("--prefill-token-buckets", default="")
    parser.add_argument("--prefill-layout", choices=["packed", "dense"], default="packed")
    parser.add_argument("--batch-size-buckets", default="1,2,4,8")
    parser.add_argument("--max-blocks-per-seq", type=int, default=256)
    parser.add_argument("--decode-block-table-buckets", default="")
    parser.add_argument("--resident-decode-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-attention-kv-cache-dtype", default="default")
    parser.add_argument("--full-attention-kv-append-impl", default="reference")
    parser.add_argument("--full-attention-decode-impl", default="reference")
    parser.add_argument("--full-attention-prefill-impl", default="reference")
    parser.add_argument("--jax-num-speculative-tokens", type=int, choices=[0, 1], default=0)

    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-top-k", type=int, default=5)
    parser.add_argument("--vllm-mode", default="baseline", choices=["baseline", "mtp"])
    parser.add_argument("--vllm-speculative-method", default="mtp")
    parser.add_argument("--vllm-num-speculative-tokens", type=int, choices=[0, 1], default=0)

    parser.add_argument("--dry-run", action="store_true", help="Generate manifest and commands without launching benchmarks.")
    parser.add_argument("--skip-vllm", action="store_true", help="Generate suite and run JAX only.")
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=0,
        help="0 disables timeout.",
    )
    parser.add_argument(
        "--max-system-ram-percent",
        type=float,
        default=DEFAULT_MAX_SYSTEM_RAM_PERCENT,
        help="Kill benchmark subprocesses when system RAM usage exceeds this percentage; 0 disables.",
    )
    parser.add_argument(
        "--worker-cpu-cores",
        type=int,
        default=DEFAULT_WORKER_CPU_CORES,
        help="CPU affinity core cap for JAX/vLLM subprocesses; 0 disables affinity limiting.",
    )
    parser.add_argument(
        "--worker-cpu-core-offset",
        type=int,
        default=0,
        help="Offset into the current affinity mask when choosing worker CPU cores.",
    )
    parser.add_argument(
        "--worker-nice",
        type=int,
        default=DEFAULT_WORKER_NICE,
        help="Positive niceness applied to benchmark subprocesses; 0 leaves priority unchanged.",
    )
    parser.add_argument(
        "--resource-poll-seconds",
        type=float,
        default=1.0,
        help="Watchdog polling interval for timeout/RAM checks.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.min_input_tokens < 1 or args.max_input_tokens < args.min_input_tokens:
        parser.error("--max-input-tokens must be >= --min-input-tokens and both > 0")
    if args.min_output_tokens < 1 or args.max_output_tokens < args.min_output_tokens:
        parser.error("--max-output-tokens must be >= --min-output-tokens and both > 0")
    if args.min_request_count < 1 or args.max_request_count < args.min_request_count:
        parser.error("--max-request-count must be >= --min-request-count and both > 0")
    if not (0.0 < args.vllm_gpu_memory_utilization < 1.0):
        parser.error("--vllm-gpu-memory-utilization must be in (0.0, 1.0)")
    if args.max_system_ram_percent < 0 or args.max_system_ram_percent >= 100:
        parser.error("--max-system-ram-percent must be in [0, 100)")
    if args.worker_cpu_cores < 0:
        parser.error("--worker-cpu-cores must be >= 0")
    if args.worker_nice < -20 or args.worker_nice > 19:
        parser.error("--worker-nice must be between -20 and 19")
    if args.resource_poll_seconds <= 0:
        parser.error("--resource-poll-seconds must be > 0")
    return args


def _load_jax_config(path: str) -> dict[str, Any]:
    if not path:
        return {"source": "", "raw": {}, "engine_overrides": {}, "env": {}}
    import yaml

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {
        "source": str(config_path),
        "raw": raw,
        "engine_overrides": engine_overrides_from_config(raw),
        "env": runtime_env_from_config(raw),
    }


def _effective_vllm_dtype(args: argparse.Namespace) -> str:
    if args.vllm_dtype:
        return str(args.vllm_dtype)
    if args.dtype == "float32":
        return "bfloat16"
    return str(args.dtype)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest_path(output_json: str, override: str = "") -> Path:
    if override:
        return Path(override)
    output = Path(output_json)
    if output.suffix:
        return output.with_suffix(".prompts.jsonl")
    return output.with_name(f"{output.name}.prompts.jsonl")


def _build_artifact_path(output_json: str, suffix: str) -> Path:
    output = Path(output_json)
    stem = output.with_suffix("") if output.suffix else output
    return Path(f"{stem}_{suffix}.json")


def generate_random_request_rows(
    *,
    seed: int,
    num_requests: int,
    min_input_tokens: int,
    max_input_tokens: int,
    min_output_tokens: int,
    max_output_tokens: int,
    token_vocab_size: int,
    eos_token_id: int | None,
) -> list[dict[str, Any]]:
    if num_requests < 0:
        raise ValueError("num_requests must be >= 0")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for request_id in range(num_requests):
        prompt_length = int(rng.integers(min_input_tokens, max_input_tokens + 1))
        output_len = int(rng.integers(min_output_tokens, max_output_tokens + 1))
        token_ids = [
            int(item)
            for item in rng.integers(0, token_vocab_size, size=prompt_length, dtype=np.int64).tolist()
        ]
        if eos_token_id is not None and token_ids:
            replacement = eos_token_id + 1 if eos_token_id + 1 < token_vocab_size else 0
            token_ids = [replacement if token_id == eos_token_id else token_id for token_id in token_ids]
        rows.append(
            {
                "request_id": str(request_id),
                "name": f"random_{request_id}",
                "prompt_len": prompt_length,
                "prompt_token_ids": token_ids,
                "output_len": output_len,
            }
        )
    return rows


def build_random_request_suite(args: argparse.Namespace, token_vocab_size: int, eos_token_id: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    num_requests = int(rng.integers(args.min_request_count, args.max_request_count + 1))
    rows = generate_random_request_rows(
        seed=args.seed,
        num_requests=num_requests,
        min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        token_vocab_size=token_vocab_size,
        eos_token_id=eos_token_id,
    )

    prompt_lens = [int(row["prompt_len"]) for row in rows]
    output_lens = [int(row["output_len"]) for row in rows]
    suite_metadata = {
        "seed": int(args.seed),
        "dataset_name": args.dataset_name or "random",
        "request_count": len(rows),
        "input_range": {"min": args.min_input_tokens, "max": args.max_input_tokens},
        "output_range": {"min": args.min_output_tokens, "max": args.max_output_tokens},
        "prompt_len_min": min(prompt_lens),
        "prompt_len_max": max(prompt_lens),
        "prompt_len_mean": float(sum(prompt_lens) / len(prompt_lens)),
        "output_len_min": min(output_lens),
        "output_len_max": max(output_lens),
        "output_len_mean": float(sum(output_lens) / len(output_lens)),
    }
    return rows, suite_metadata


def write_prompt_manifest(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "request_id": row["request_id"],
                        "prompt_token_ids": row["prompt_token_ids"],
                        "prompt_len": row["prompt_len"],
                        "output_len": row["output_len"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return _sha256_file(path)


def _append_cli_arg(command: list[str], key: str, value: Any) -> None:
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            command.append(flag)
        else:
            command.append(f"--no-{flag[2:]}")
        return
    if value is None:
        return
    if value == "":
        return
    command.extend([flag, str(value)])


def _system_ram_percent() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().percent)
    except Exception:
        pass

    meminfo: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    meminfo[parts[0][:-1]] = int(parts[1])
    except OSError:
        return None
    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", 0)
    if total <= 0:
        return None
    return 100.0 * float(total - available) / float(total)


def _current_affinity() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(int(core) for core in os.sched_getaffinity(0))
        except OSError:
            pass
    return list(range(max(1, os.cpu_count() or 1)))


def _selected_worker_cores(worker_cpu_cores: int, worker_cpu_core_offset: int) -> list[int]:
    available = _current_affinity()
    if worker_cpu_cores <= 0 or worker_cpu_cores >= len(available):
        return []
    offset = int(worker_cpu_core_offset) % len(available)
    rotated = available[offset:] + available[:offset]
    return rotated[:worker_cpu_cores]


def _thread_env_for_cpu_cap(worker_cpu_cores: int) -> dict[str, str]:
    if worker_cpu_cores <= 0:
        return {}
    value = str(max(1, worker_cpu_cores))
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "BLIS_NUM_THREADS": value,
    }


def _process_tree_rss_bytes(pid: int) -> int | None:
    try:
        import psutil  # type: ignore

        root = psutil.Process(pid)
        processes = [root] + root.children(recursive=True)
        return int(sum(proc.memory_info().rss for proc in processes if proc.is_running()))
    except Exception:
        return None


def _apply_affinity_to_tree(pid: int, cores: list[int]) -> None:
    if not cores:
        return
    try:
        import psutil  # type: ignore

        root = psutil.Process(pid)
        for proc in [root] + root.children(recursive=True):
            try:
                proc.cpu_affinity(cores)
            except Exception:
                continue
    except Exception:
        return


def _terminate_process_group(process: subprocess.Popen[Any], *, grace_seconds: float = 5.0) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    except OSError:
        process.terminate()
    deadline = time.perf_counter() + grace_seconds
    while process.poll() is None and time.perf_counter() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        process.wait()
    return process.returncode


def _run_command(
    command: list[str],
    *,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
    max_system_ram_percent: float = 0.0,
    worker_cpu_cores: int = 0,
    worker_cpu_core_offset: int = 0,
    worker_nice: int = 0,
    resource_poll_seconds: float = 1.0,
) -> dict[str, Any]:
    env = os.environ.copy()
    selected_cores = _selected_worker_cores(worker_cpu_cores, worker_cpu_core_offset)
    thread_env_cores = (
        len(selected_cores)
        if selected_cores
        else (0 if worker_cpu_cores <= 0 else min(worker_cpu_cores, len(_current_affinity())))
    )
    for key, value in _thread_env_for_cpu_cap(thread_env_cores).items():
        env.setdefault(key, value)
    for key, value in (env_overrides or {}).items():
        env.setdefault(key, value)

    def _limit_child_process() -> None:
        if selected_cores and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, selected_cores)
            except OSError:
                pass
        if worker_nice:
            try:
                os.nice(worker_nice)
            except OSError:
                pass

    started = time.perf_counter()
    limit_reason: dict[str, Any] | None = None
    peak_system_ram_percent: float | None = None
    peak_process_tree_rss_bytes: int | None = None
    with tempfile.TemporaryFile(mode="w+b") as output_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_limit_child_process if (selected_cores or worker_nice) else None,
        )
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if timeout_seconds > 0 and elapsed > timeout_seconds:
                limit_reason = {
                    "kind": "timeout",
                    "elapsed_seconds": elapsed,
                    "limit_seconds": int(timeout_seconds),
                }
                _terminate_process_group(process)
                break
            ram_percent = _system_ram_percent()
            if ram_percent is not None:
                peak_system_ram_percent = max(peak_system_ram_percent or 0.0, ram_percent)
                if max_system_ram_percent > 0 and ram_percent >= max_system_ram_percent:
                    limit_reason = {
                        "kind": "system_ram_percent",
                        "observed_percent": ram_percent,
                        "limit_percent": float(max_system_ram_percent),
                    }
                    _terminate_process_group(process)
                    break
            rss_bytes = _process_tree_rss_bytes(process.pid)
            if rss_bytes is not None:
                peak_process_tree_rss_bytes = max(peak_process_tree_rss_bytes or 0, rss_bytes)
            _apply_affinity_to_tree(process.pid, selected_cores)
            time.sleep(float(resource_poll_seconds))
        returncode = process.wait()
        output_file.flush()
        output_file.seek(0, os.SEEK_END)
        output_size = output_file.tell()
        output_file.seek(max(0, output_size - 24_000), os.SEEK_SET)
        output_tail = output_file.read().decode("utf-8", errors="replace")[-12_000:]

    status = "ok" if returncode == 0 else "failed"
    if limit_reason:
        status = "timeout" if limit_reason["kind"] == "timeout" else "killed_resource_limit"
    return {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "output_tail": output_tail,
        "resource_limits": {
            "max_system_ram_percent": float(max_system_ram_percent),
            "worker_cpu_cores": int(worker_cpu_cores),
            "worker_cpu_core_offset": int(worker_cpu_core_offset),
            "worker_cpu_affinity": selected_cores,
            "worker_nice": int(worker_nice),
            "resource_poll_seconds": float(resource_poll_seconds),
        },
        "resource_limit_reason": limit_reason,
        "peak_system_ram_percent": peak_system_ram_percent,
        "peak_process_tree_rss_bytes": peak_process_tree_rss_bytes,
    }


def _run_benchmark(
    command: list[str],
    artifact_path: Path,
    *,
    dry_run: bool,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
    max_system_ram_percent: float = 0.0,
    worker_cpu_cores: int = 0,
    worker_cpu_core_offset: int = 0,
    worker_nice: int = 0,
    resource_poll_seconds: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resource_limits = {
        "max_system_ram_percent": float(max_system_ram_percent),
        "worker_cpu_cores": int(worker_cpu_cores),
        "worker_cpu_core_offset": int(worker_cpu_core_offset),
        "worker_nice": int(worker_nice),
        "resource_poll_seconds": float(resource_poll_seconds),
    }
    if dry_run:
        return {
            "status": "dry_run",
            "returncode": None,
            "command": command,
            "env_overrides": dict(env_overrides or {}),
            "elapsed_seconds": 0.0,
            "output_tail": "",
            "resource_limits": resource_limits,
        }, {}

    run_result = _run_command(
        command,
        timeout_seconds=timeout_seconds,
        env_overrides=env_overrides,
        max_system_ram_percent=max_system_ram_percent,
        worker_cpu_cores=worker_cpu_cores,
        worker_cpu_core_offset=worker_cpu_core_offset,
        worker_nice=worker_nice,
        resource_poll_seconds=resource_poll_seconds,
    )
    if run_result["status"] != "ok":
        return {**run_result, "env_overrides": dict(env_overrides or {})}, {}
    if not artifact_path.exists():
        return {
            "status": "failed",
            "returncode": -1,
            "command": command,
            "env_overrides": dict(env_overrides or {}),
            "elapsed_seconds": run_result["elapsed_seconds"],
            "output_tail": f"{artifact_path} was not generated despite successful return code.\n" + run_result["output_tail"],
        }, {}
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return {
        **run_result,
        "command": command,
        "env_overrides": dict(env_overrides or {}),
    }, artifact


def _first_diff(left: list[int], right: list[int]) -> dict[str, Any] | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if int(lhs) != int(rhs):
            return {"index": index, "left": int(lhs), "right": int(rhs)}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "left": int(left[index]) if index < len(left) else None,
            "right": int(right[index]) if index < len(right) else None,
            "length_mismatch": True,
        }
    return None


def compare_generated_tokens(jax_rows: list[dict[str, Any]], vllm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    jax_rows_by_name = {str(row.get("name")): row for row in jax_rows}
    vllm_rows_by_name = {str(row.get("name")): row for row in vllm_rows}
    names = sorted(set(jax_rows_by_name) | set(vllm_rows_by_name))
    comparison_rows: list[dict[str, Any]] = []
    for name in names:
        jax_row = jax_rows_by_name.get(name)
        vllm_row = vllm_rows_by_name.get(name)
        if jax_row is None or vllm_row is None:
            comparison_rows.append({"name": name, "checked": False, "reason": "missing_row"})
            continue
        jax_tokens = [int(token) for token in jax_row.get("generated_token_ids", [])]
        vllm_tokens = [int(token) for token in vllm_row.get("generated_token_ids", [])]
        common_len = min(len(jax_tokens), len(vllm_tokens))
        diff = _first_diff(jax_tokens[:common_len], vllm_tokens[:common_len])
        comparison_rows.append(
            {
                "name": name,
                "checked": True,
                "comparison_length": common_len,
                "jax_length": len(jax_tokens),
                "vllm_length": len(vllm_tokens),
                "generated_prefix_match": diff is None,
                "generated_full_match": diff is None and len(jax_tokens) == len(vllm_tokens),
                "first_diff": diff,
                "prompt_len": int(jax_row.get("prompt_length", 0)),
            }
        )
    checked_rows = [row for row in comparison_rows if row.get("checked")]
    return {
        "checked": bool(checked_rows),
        "matches": all(bool(row.get("generated_full_match")) for row in checked_rows),
        "request_count": len(names),
        "checked_request_count": len(checked_rows),
        "rows": comparison_rows,
    }


def _build_jax_command(
    args: argparse.Namespace,
    manifest_jsonl: Path,
    output_json: Path,
    *,
    config_engine_overrides: dict[str, Any] | None = None,
) -> list[str]:
    command = [args.jax_python, str(SCRIPT_DIR / "benchmark_jax_server_trace.py")]
    command_args = {
        "model": args.model,
        "backend": args.backend,
        "dtype": args.dtype,
        "weight_dtype": args.weight_dtype,
        "jax_execution": args.jax_execution,
        "prompt_source": "manifest",
        "prompt_manifest_jsonl": str(manifest_jsonl),
        "seed": args.seed,
        "run_label": f"{args.run_label}_jax",
        "prompt_suite": args.prompt_suite,
        "warmup": args.jax_warmup,
        "warmup_mode": args.jax_warmup_mode,
        "fail_on_jit_cache_growth": args.jax_fail_on_jit_cache_growth,
        "profile": args.jax_profile,
        "profile_dir": args.jax_trace_profile_dir,
        "run_log": args.run_log,
        "max_kv_cache_mb": args.jax_max_kv_cache_mb,
        "num_kvcache_blocks": args.jax_num_kvcache_blocks,
        "max_num_seqs": args.max_num_seqs,
        "max_num_resident_seqs": args.max_num_resident_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prefill_buckets": args.prefill_buckets,
        "prefill_token_buckets": args.prefill_token_buckets or args.prefill_buckets,
        "prefill_layout": args.prefill_layout,
        "batch_size_buckets": args.batch_size_buckets,
        "max_blocks_per_seq": args.max_blocks_per_seq,
        "num_speculative_tokens": args.jax_num_speculative_tokens,
        "dataset_name": args.dataset_name or "random",
        "output_json": str(output_json),
    }
    command_args.update(config_engine_overrides or {})
    if args.decode_block_table_buckets:
        command_args["decode_block_table_buckets"] = args.decode_block_table_buckets
    if args.resident_decode_metadata:
        command_args["resident_decode_metadata"] = True
    if args.full_attention_kv_cache_dtype != "default":
        command_args["full_attention_kv_cache_dtype"] = args.full_attention_kv_cache_dtype
    if args.full_attention_kv_append_impl != "reference":
        command_args["full_attention_kv_append_impl"] = args.full_attention_kv_append_impl
    if args.full_attention_decode_impl != "reference":
        command_args["full_attention_decode_impl"] = args.full_attention_decode_impl
    if args.full_attention_prefill_impl != "reference":
        command_args["full_attention_prefill_impl"] = args.full_attention_prefill_impl
    if args.reference_json:
        command_args["reference_json"] = args.reference_json
    for key, value in command_args.items():
        _append_cli_arg(command, key, value)
    return command


def _effective_jax_kernel_policy(
    args: argparse.Namespace,
    config_engine_overrides: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "full_attention_kv_cache_dtype",
        "full_attention_kv_append_impl",
        "full_attention_decode_impl",
        "full_attention_prefill_impl",
        "gdn_disable_fallbacks",
        "gdn_prefill_post_conv_impl",
        "gdn_prefill_qkv_dtype",
        "gdn_prefill_post_conv_output_dtype",
        "gdn_packed_decode_impl",
        "gdn_packed_decode_qkv_dtype",
        "gdn_packed_decode_pre_normalize_qk",
        "gdn_packed_decode_max_batch",
    )
    policy: dict[str, Any] = {}
    for key in keys:
        if key in config_engine_overrides:
            policy[key] = config_engine_overrides[key]
        elif hasattr(args, key):
            policy[key] = getattr(args, key)
    return policy


def _build_vllm_command(args: argparse.Namespace, manifest_jsonl: Path, output_json: Path) -> list[str]:
    command = [args.vllm_python, str(SCRIPT_DIR / "benchmark_vllm_qwen35.py")]
    command_args = {
        "model": args.model,
        "dtype": _effective_vllm_dtype(args),
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "execution": args.vllm_execution,
        "max_model_len": args.vllm_max_model_len,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "trust_remote_code": True,
        "prompt_source": "manifest",
        "prompt_manifest_jsonl": str(manifest_jsonl),
        "dataset_name": args.dataset_name or "random",
        "mode": args.vllm_mode,
        "top_k": args.vllm_top_k,
        "seed": args.seed,
        "run_label": f"{args.run_label}_vllm",
        "output_json": str(output_json),
        "run_log": args.run_log,
    }
    if args.reference_json:
        command_args["reference_json"] = args.reference_json
    if args.vllm_mode == "mtp":
        command_args["speculative_method"] = args.vllm_speculative_method
        command_args["num_speculative_tokens"] = args.vllm_num_speculative_tokens
    for key, value in command_args.items():
        _append_cli_arg(command, key, value)
    return command


def _extract_rows(artifact: dict[str, Any], field: str) -> list[dict[str, Any]]:
    return artifact.get(field, [])


def _run() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))

    if args.dry_run:
        token_vocab_size = 32_000
        eos_token_id = 0
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        token_vocab_size = int(getattr(tokenizer, "vocab_size", 32_000))
        eos_token_id = getattr(tokenizer, "eos_token_id", 0)

    rows, suite_metadata = build_random_request_suite(args, token_vocab_size, eos_token_id)
    manifest_path = _build_manifest_path(args.output_json, args.manifest_jsonl or args.prompt_manifest_output_jsonl)
    manifest_sha = write_prompt_manifest(rows, manifest_path)
    jax_config = _load_jax_config(args.jax_config)

    jax_output_json = _build_artifact_path(args.output_json, "jax")
    if args.output_json_jax:
        jax_output_json = Path(args.output_json_jax)
    vllm_output_json = _build_artifact_path(args.output_json, "vllm")
    if args.output_json_vllm:
        vllm_output_json = Path(args.output_json_vllm)

    jax_command = _build_jax_command(
        args,
        manifest_jsonl=manifest_path,
        output_json=jax_output_json,
        config_engine_overrides=jax_config["engine_overrides"],
    )
    vllm_command = _build_vllm_command(args, manifest_jsonl=manifest_path, output_json=vllm_output_json)
    effective_jax_kernel_policy = _effective_jax_kernel_policy(
        args,
        jax_config["engine_overrides"],
    )

    jax_run, jax_artifact = _run_benchmark(
        jax_command,
        artifact_path=jax_output_json,
        dry_run=args.dry_run,
        timeout_seconds=args.command_timeout_seconds,
        env_overrides=jax_config["env"],
        max_system_ram_percent=args.max_system_ram_percent,
        worker_cpu_cores=args.worker_cpu_cores,
        worker_cpu_core_offset=args.worker_cpu_core_offset,
        worker_nice=args.worker_nice,
        resource_poll_seconds=args.resource_poll_seconds,
    )

    if args.skip_vllm:
        vllm_run = {
            "status": "skipped",
            "returncode": None,
            "command": vllm_command,
            "elapsed_seconds": 0.0,
            "output_tail": "",
            "resource_limits": {
                "max_system_ram_percent": args.max_system_ram_percent,
                "worker_cpu_cores": args.worker_cpu_cores,
                "worker_cpu_core_offset": args.worker_cpu_core_offset,
                "worker_nice": args.worker_nice,
                "resource_poll_seconds": args.resource_poll_seconds,
            },
        }
        vllm_artifact = {}
    else:
        vllm_run, vllm_artifact = _run_benchmark(
            vllm_command,
            artifact_path=vllm_output_json,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_seconds,
            max_system_ram_percent=args.max_system_ram_percent,
            worker_cpu_cores=args.worker_cpu_cores,
            worker_cpu_core_offset=args.worker_cpu_core_offset,
            worker_nice=args.worker_nice,
            resource_poll_seconds=args.resource_poll_seconds,
        )

    comparison = (
        compare_generated_tokens(_extract_rows(jax_artifact, "rows"), _extract_rows(vllm_artifact, "rows"))
        if jax_artifact and vllm_artifact
        else {
            "checked": False,
            "matches": False,
            "request_count": len(rows),
            "checked_request_count": 0,
            "rows": [],
        }
    )

    output = {
        "schema_version": "1.1",
        "run_config": {
            "run_label": args.run_label,
            "script": Path(__file__).name,
            "model": args.model,
            "seed": args.seed,
            "random_request_generation": {
                "min_input_tokens": args.min_input_tokens,
                "max_input_tokens": args.max_input_tokens,
                "min_output_tokens": args.min_output_tokens,
                "max_output_tokens": args.max_output_tokens,
                "min_request_count": args.min_request_count,
                "max_request_count": args.max_request_count,
            },
            "suite_metadata": suite_metadata,
            "resource_limits": {
                "max_system_ram_percent": args.max_system_ram_percent,
                "worker_cpu_cores": args.worker_cpu_cores,
                "worker_cpu_core_offset": args.worker_cpu_core_offset,
                "worker_nice": args.worker_nice,
                "resource_poll_seconds": args.resource_poll_seconds,
            },
            "manifest": {
                "prompt_manifest_jsonl": str(manifest_path),
                "prompt_manifest_sha256": manifest_sha,
                "prompt_rows": len(rows),
                "prompt_row_sample": rows[:2],
            },
            "jax_args": {
                "backend": args.backend,
                "dtype": args.dtype,
                "weight_dtype": args.weight_dtype,
                "jax_execution": args.jax_execution,
                "max_num_seqs": args.max_num_seqs,
                "max_num_resident_seqs": args.max_num_resident_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "warmup": args.jax_warmup,
                "warmup_mode": args.jax_warmup_mode,
                "fail_on_jit_cache_growth": args.jax_fail_on_jit_cache_growth,
                "profile": args.jax_profile,
                "max_kv_cache_mb": args.jax_max_kv_cache_mb,
                "num_kvcache_blocks": args.jax_num_kvcache_blocks,
                "prefill_buckets": args.prefill_buckets,
                "prefill_token_buckets": args.prefill_token_buckets or args.prefill_buckets,
                "prefill_layout": args.prefill_layout,
                "batch_size_buckets": args.batch_size_buckets,
                "max_blocks_per_seq": args.max_blocks_per_seq,
                "decode_block_table_buckets": args.decode_block_table_buckets,
                "resident_decode_metadata": args.resident_decode_metadata,
                "full_attention_kv_cache_dtype": args.full_attention_kv_cache_dtype,
                "full_attention_kv_append_impl": args.full_attention_kv_append_impl,
                "full_attention_decode_impl": args.full_attention_decode_impl,
                "full_attention_prefill_impl": args.full_attention_prefill_impl,
                "num_speculative_tokens": args.jax_num_speculative_tokens,
                "config": {
                    "source": jax_config["source"],
                    "engine_overrides": jax_config["engine_overrides"],
                    "env": jax_config["env"],
                },
                "effective_kernel_policy": effective_jax_kernel_policy,
            },
            "vllm_args": {
                "dtype": _effective_vllm_dtype(args),
                "dtype_override": args.vllm_dtype,
                "execution": args.vllm_execution,
                "mode": args.vllm_mode,
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "tensor_parallel_size": args.vllm_tensor_parallel_size,
                "top_k": args.vllm_top_k,
                "num_speculative_tokens": args.vllm_num_speculative_tokens,
            },
            "reference_json": args.reference_json,
        },
        "runs": {
            "jax": {
                "artifact": str(jax_output_json),
                "command": jax_command,
                "run": jax_run,
            },
            "vllm": {
                "artifact": str(vllm_output_json),
                "command": vllm_command,
                "run": vllm_run,
            },
        },
        "performance": {
            "jax": jax_artifact.get("performance", {}) if jax_artifact else {},
            "vllm": vllm_artifact.get("performance", {}) if vllm_artifact else {},
            "request_throughput": {
                "jax": jax_artifact.get("performance", {}).get("request_throughput") if jax_artifact else None,
                "vllm": vllm_artifact.get("performance", {}).get("request_throughput") if vllm_artifact else None,
            },
            "ttft_ms_p50": {
                "jax": jax_artifact.get("performance", {}).get("ttft_ms_p50") if jax_artifact else None,
                "vllm": vllm_artifact.get("performance", {}).get("ttft_ms_p50") if vllm_artifact else None,
            },
            "itl_ms_p50": {
                "jax": jax_artifact.get("performance", {}).get("itl_ms_p50") if jax_artifact else None,
                "vllm": vllm_artifact.get("performance", {}).get("itl_ms_p50") if vllm_artifact else None,
            },
        },
        "correctness": {
            "reference_json": args.reference_json,
            "jax_vs_reference": jax_artifact.get("correctness", {}) if jax_artifact else {},
            "vllm_vs_reference": vllm_artifact.get("correctness", {}) if vllm_artifact else {},
            "jax_vs_vllm": comparison,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.skip_vllm and jax_run.get("status") != "ok" and not args.dry_run:
        raise RuntimeError(f"JAX benchmark failed: {jax_run}")
    if args.dry_run:
        return
    if jax_run.get("status") != "ok":
        raise RuntimeError(f"JAX benchmark failed: {jax_run}")
    if not args.skip_vllm and vllm_run.get("status") != "ok":
        raise RuntimeError(f"vLLM benchmark failed: {vllm_run}")


if __name__ == "__main__":
    _run()
