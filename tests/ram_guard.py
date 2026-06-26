"""Run a command while bounding process-tree resident memory.

This helper is intentionally dependency-free so long GPU/JAX test runs can be
kept from exhausting a shared machine.  It watches the command and all known
children, terminates the process group if resident memory crosses the limit,
and also stops if system available memory falls below the configured floor.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque


GIB = 1024**3


def _children(pid: int) -> list[int]:
    try:
        text = open(f"/proc/{pid}/task/{pid}/children", encoding="utf-8").read()
    except (FileNotFoundError, ProcessLookupError):
        return []
    return [int(value) for value in text.split()]


def _process_tree(root_pid: int) -> set[int]:
    seen: set[int] = set()
    queue: deque[int] = deque([root_pid])
    while queue:
        pid = queue.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(child for child in _children(pid) if child not in seen)
    return seen


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


def _tree_rss_bytes(root_pid: int) -> int:
    return sum(_rss_bytes(pid) for pid in _process_tree(root_pid))


def _mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None
    return None


def _terminate(proc: subprocess.Popen[object], reason: str, grace_seconds: float) -> int:
    print(f"ram_guard: terminating command: {reason}", file=sys.stderr, flush=True)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.poll() or 1
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            return code
        time.sleep(0.25)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return 137


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rss-gib",
        type=float,
        default=float(os.environ.get("RAM_GUARD_RSS_GIB", "12")),
        help="Maximum resident memory for the command process tree.",
    )
    parser.add_argument(
        "--min-available-gib",
        type=float,
        default=float(os.environ.get("RAM_GUARD_MIN_AVAILABLE_GIB", "2")),
        help="Minimum system MemAvailable before the command is stopped.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--report-seconds", type=float, default=60.0)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")

    rss_limit = int(args.rss_gib * GIB)
    available_floor = int(args.min_available_gib * GIB)
    print(
        "ram_guard: rss_limit="
        f"{args.rss_gib:.1f}GiB min_available={args.min_available_gib:.1f}GiB "
        f"command={' '.join(command)}",
        file=sys.stderr,
        flush=True,
    )

    try:
        proc = subprocess.Popen(command, start_new_session=True)
    except FileNotFoundError as exc:
        print(f"ram_guard: {exc}", file=sys.stderr)
        return 127

    last_report = 0.0
    while True:
        code = proc.poll()
        if code is not None:
            return code

        rss = _tree_rss_bytes(proc.pid)
        available = _mem_available_bytes()
        now = time.monotonic()
        if now - last_report >= args.report_seconds:
            available_text = "unknown" if available is None else f"{available / GIB:.1f}GiB"
            print(
                f"ram_guard: rss={rss / GIB:.1f}GiB available={available_text}",
                file=sys.stderr,
                flush=True,
            )
            last_report = now

        if rss > rss_limit:
            return _terminate(
                proc,
                f"RSS {rss / GIB:.1f}GiB exceeded {args.rss_gib:.1f}GiB",
                args.grace_seconds,
            )
        if available is not None and available < available_floor:
            return _terminate(
                proc,
                f"MemAvailable {available / GIB:.1f}GiB below {args.min_available_gib:.1f}GiB",
                args.grace_seconds,
            )

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
