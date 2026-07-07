"""Process execution action primitives.

Pure-Python subprocess wrapper. The action layer owns the actual shell
invocation; the tool layer owns the human-approval gate that runs before
this code is reached.

Module rules:
  - No upward imports into the tool or state layers, or into the langchain or data modules.
  - Only primitive argument types (str, int).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def execute_python_action(code: str, *, timeout_seconds: int) -> dict:
    """Execute `code` as a Python script in a subprocess.

    The code is written to a temporary file and run with the current
    Python interpreter. stdout and stderr are captured (merged). If the
    script does not complete within `timeout_seconds`, it is killed and
    `timed_out` is set to True in the result.

    Args:
        code: Source code to execute.
        timeout_seconds: Hard wall-clock limit for the subprocess.

    Returns:
        A dict with keys:
          - exit_code (int): process exit status, or -1 if killed by timeout.
          - stdout (str): captured standard output (UTF-8, errors replaced).
          - stderr (str): captured standard error (merged into stdout by
            design - we pipe stderr to STDOUT for simpler consumption).
          - timed_out (bool): True if the timeout fired.
          - duration_seconds (float): wall-clock duration of the run.
    """
    started = time.monotonic()
    timed_out = False
    exit_code = 0
    output_text = ""

    temp_dir = tempfile.gettempdir()
    temp_script = os.path.join(temp_dir, "ved_runtime_exec.py")
    try:
        Path(temp_script).write_text(code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-u", temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = result.returncode
        output_text = result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -1
        captured = exc.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        output_text = captured + (
            f"\nERROR: Terminal process terminated - hardware timeout gate "
            f"({timeout_seconds}s) exceeded."
        )
    except Exception as exc:
        exit_code = -1
        output_text = f"ERROR: {exc}"
    finally:
        try:
            if os.path.exists(temp_script):
                os.remove(temp_script)
        except Exception:
            pass

    duration = time.monotonic() - started
    return {
        "exit_code": exit_code,
        "stdout": output_text,
        "stderr": "",  # merged into stdout above; kept for schema stability
        "timed_out": timed_out,
        "duration_seconds": duration,
    }
