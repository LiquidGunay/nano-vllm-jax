"""Run profiling and journal helpers for local benchmark entry points."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from runtime_paths import default_runtime_root


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value in {"1", "true", "yes", "on", "True"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


class RunRecorder:
    """Append-only run journal plus optional JAX trace directory.

    The journal is intentionally plain JSONL so benchmark failures still leave a
    durable trail under the runtime root.
    """

    def __init__(
        self,
        *,
        script: str,
        args: dict[str, Any],
        run_label: str = "",
        profile_dir: str | Path | None = None,
        run_log: str | Path | None = None,
    ):
        root = default_runtime_root()
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        safe_label = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in (run_label or Path(script).stem)
        ).strip("_")
        self.run_id = f"{timestamp}-{os.getpid()}-{safe_label}"
        self.script = script
        self.args = _json_safe(args)
        self.run_label = run_label or safe_label
        self.started_at = time.time()
        self.profile_root = Path(profile_dir) if profile_dir else root / "profiles"
        self.profile_path = self.profile_root / self.run_id
        self.run_log_path = Path(run_log) if run_log else root / "run_logs" / "run_journal.jsonl"
        self.manifest_path = self.profile_path / "run_manifest.json"
        self.issues: list[dict[str, Any]] = []
        self.profile_started = False
        self.profile_enabled = False
        self.profile_error: str | None = None
        self.profile_path.mkdir(parents=True, exist_ok=True)
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_event(
            "start",
            {
                "script": self.script,
                "args": self.args,
                "profile_path": str(self.profile_path),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "cwd": str(Path.cwd()),
            },
        )

    @classmethod
    def create(
        cls,
        *,
        script: str,
        args: dict[str, Any],
        run_label: str = "",
        profile_dir: str | Path | None = None,
        run_log: str | Path | None = None,
    ) -> "RunRecorder":
        return cls(
            script=script,
            args=args,
            run_label=run_label,
            profile_dir=profile_dir,
            run_log=run_log,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "run_log_path": str(self.run_log_path),
            "profile_path": str(self.profile_path),
            "profile_enabled": bool(self.profile_enabled),
            "profile_started": bool(self.profile_started),
            "profile_error": self.profile_error,
            "issues": list(self.issues),
        }

    def _write_event(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "time": time.time(),
            "event": event,
            "run_id": self.run_id,
            "script": self.script,
            **_json_safe(payload),
        }
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def start_jax_profile(self, *, enabled: bool | None = None) -> None:
        should_profile = _truthy(os.getenv("NANO_VLLM_JAX_PROFILE"), default=True)
        if enabled is not None:
            should_profile = bool(enabled)
        self.profile_enabled = should_profile
        if not should_profile:
            self._write_event("profile_skipped", {"profile_path": str(self.profile_path)})
            return
        try:
            import jax

            jax.profiler.start_trace(str(self.profile_path))
            self.profile_started = True
            self._write_event("profile_start", {"profile_path": str(self.profile_path)})
        except Exception as exc:
            self.profile_error = f"{type(exc).__name__}: {exc}"
            self.record_issue(
                summary="failed to start JAX profiler",
                severity="warning",
                status="open",
                details={"error": self.profile_error, "profile_path": str(self.profile_path)},
                resolution="pending",
            )

    def stop_jax_profile(self) -> None:
        if not self.profile_started:
            return
        try:
            import jax

            jax.profiler.stop_trace()
            self._write_event("profile_stop", {"profile_path": str(self.profile_path)})
        except Exception as exc:
            self.profile_error = f"{type(exc).__name__}: {exc}"
            self.record_issue(
                summary="failed to stop JAX profiler",
                severity="warning",
                status="open",
                details={"error": self.profile_error, "profile_path": str(self.profile_path)},
                resolution="pending",
            )
        finally:
            self.profile_started = False

    def record_issue(
        self,
        *,
        summary: str,
        severity: str = "warning",
        status: str = "open",
        details: dict[str, Any] | None = None,
        learnings: list[str] | None = None,
        resolution: str = "pending",
    ) -> dict[str, Any]:
        issue = {
            "summary": summary,
            "severity": severity,
            "status": status,
            "details": _json_safe(details or {}),
            "learnings": list(learnings or []),
            "resolution": resolution,
        }
        self.issues.append(issue)
        self._write_event("issue", issue)
        return issue

    def finish(
        self,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        learnings: list[str] | None = None,
        resolution: str = "",
    ) -> None:
        elapsed = time.time() - self.started_at
        payload = {
            "status": status,
            "elapsed_seconds": elapsed,
            "summary": _json_safe(summary or {}),
            "issues": list(self.issues),
            "learnings": list(learnings or []),
            "resolution": resolution,
            "profile_path": str(self.profile_path),
            "run_log_path": str(self.run_log_path),
        }
        self.manifest_path.write_text(
            json.dumps({"run": self.metadata(), **payload}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_event("finish", payload)

    def finish_exception(self, exc: BaseException) -> None:
        self.record_issue(
            summary=f"{self.script} failed",
            severity="error",
            status="open",
            details={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=12),
            },
            resolution="pending",
        )
        self.finish(
            status="failed",
            summary={"exception": f"{type(exc).__name__}: {exc}"},
            learnings=["Run failed before producing a complete benchmark summary."],
            resolution="Inspect the recorded exception and rerun after the fix.",
        )
