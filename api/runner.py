"""Script execution backend for the /run HTTP endpoint.

Wraps subprocess.run with safety guards:
- 30s timeout (matches the Tkinter /run command behavior).
- Captures stdout + stderr separately.
- Truncates output to MAX_OUTPUT_BYTES to protect against runaway scripts.
- Runs the script in a caller-provided working directory.

This module does NOT import FastAPI — it's a pure async function library
that the HTTP layer calls. This keeps the runner testable in isolation.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Hard caps. Exceed → truncate, don't fail.
MAX_OUTPUT_BYTES = 16 * 1024  # 16 KiB per stream
TIMEOUT_SECONDS = 30


def _truncate(text: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated at {limit} bytes ...]"


async def run_python_script(
    source_path: str,
    args: Optional[List[str]] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
    cwd: Optional[str] = None,
) -> Dict:
    """Execute a Python script in a subprocess and return its output.

    Args:
        source_path: Path to a .py file on the local filesystem.
        args: Optional CLI arguments to pass after the script path.
        timeout_seconds: Hard timeout (default 30s).
        cwd: Working directory.

    Returns:
        Dict with keys: exit_code, stdout, stderr, timed_out,
        duration_seconds, truncated_stdout, truncated_stderr.

    Raises:
        FileNotFoundError: if source_path doesn't exist.
        ValueError: if source_path is not a .py file.
    """
    if not source_path or not os.path.isfile(source_path):
        raise FileNotFoundError(f"Script not found: {source_path}")
    if not source_path.endswith(".py"):
        raise ValueError(f"Only .py files are supported (got: {source_path})")

    cmd = [sys.executable, source_path] + (args or [])

    def _run() -> tuple:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    start = time.monotonic()
    timed_out = False
    try:
        exit_code, stdout, stderr = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -1
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        if not stderr:
            stderr = f"[Script killed after {timeout_seconds}s timeout]"
    duration = time.monotonic() - start

    truncated_stdout = len(stdout) > MAX_OUTPUT_BYTES
    truncated_stderr = len(stderr) > MAX_OUTPUT_BYTES
    return {
        "exit_code": exit_code,
        "stdout": _truncate(stdout),
        "stderr": _truncate(stderr),
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "truncated_stdout": truncated_stdout,
        "truncated_stderr": truncated_stderr,
    }


def format_run_output(meta: Dict, script_name: str) -> str:
    """Format a run result dict into a human-readable string."""
    if meta.get("timed_out"):
        return f"[Script timed out after {int(round(meta.get('duration_seconds', 0)))}s: {script_name}]"
    parts = [f"Script output ({script_name}):"]
    if meta.get("stdout"):
        parts.append(meta["stdout"])
    if meta.get("stderr"):
        parts.append("[stderr]:\n" + meta["stderr"])
    if meta.get("exit_code", 0) != 0:
        parts.append(f"[exit code {meta['exit_code']}]")
    if not (meta.get("stdout") or meta.get("stderr")):
        parts.append("(script produced no output)")
    return "\n".join(parts)
