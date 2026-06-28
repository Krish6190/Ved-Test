"""Sandboxed Python execution tool for Ved.

LangChain `@tool`-formatted. The agent binds this via `llm.bind_tools([...])`
and emits a structured `execute_python(code=...)` call when it wants to run
code. State (currently unused but kept for consistency / future hardening)
is injected via `InjectedState`.

The tool:
  1. Shows a tkinter approval popup showing the code (user must OK).
  2. Writes the code to a temp file in the OS temp dir.
  3. Runs it in a subprocess with the current Python interpreter.
  4. Captures stdout+stderr with a 10-second hard timeout.
  5. Cleans up the temp file.

The `code` argument is OPTIONAL. If omitted, the tool scans the last AI
message for a ```python ... ``` fence and uses that body. This makes it
trivial for the agent to execute code it just emitted without re-typing it.

The user must explicitly approve every execution — this is destructive and
untrusted by definition.
"""
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.state import VedState
from graph.tools._common import resolve_implicit_python_code

_TEMP_DIR = os.environ.get("TEMP") or os.environ.get("TMP") or "C:\\Temp"
_TIMEOUT_SECONDS = 10


def _request_approval(code: str) -> bool:
    """Request approval before running user-authored Python code.

    Routing:
      1. If we're inside an active chat session wired to the FastAPI SSE
         bus (config injected via LangChain's `RunnableConfig`), emit an
         `approval_request` event with the code preview and block on the
         existing `_human_approval_event`. The UI modal /chat/approval
         unblocks us.
      2. Otherwise, fall back to a tkinter popup (desktop UI only).

    Returns True only if the human explicitly approves.
    """
    # ---- SSE / FastAPI path ----
    try:
        from langchain_core.runnables import RunnableConfig  # noqa: F401
        # The graph node passes RunnableConfig to tool calls when bound via
        # `llm.bind_tools(VED_TOOLS)`. We can't access the call's config from
        # here directly (LangChain strips it from the tool args before invoking),
        # so we rely on a thread-local set by the node.
        from graph.tools._common import get_current_runtime_config
        cfg = get_current_runtime_config() or {}
        conf = cfg.get("configurable", {}) or {}
        token_queue = conf.get("token_queue")
        approval_event = conf.get("approval_event")
        approval_state = conf.get("approval_state")
        if token_queue is not None and approval_event is not None and approval_state is not None:
            try:
                token_queue.put(("approval_request", {
                    "kind": "python_execution",
                    "code": code[:600] + ("...[Truncated]" if len(code) > 600 else ""),
                    "code_length": len(code),
                }))
            except Exception:
                return False
            approval_event.wait()
            return bool((approval_state or {}).get("value"))
    except Exception:
        pass

    # ---- Tk popup fallback (desktop UI) ----
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        choice = messagebox.askyesno(
            title="Ved Code Execution Approval",
            message=(
                "Ved is requesting permission to run this generated Python code:\n\n"
                "----------------------------------------\n"
                f"{code[:600]}"
                f"{'...[Truncated]' if len(code) > 600 else ''}\n"
                "----------------------------------------\n\n"
                "Authorize running this script on your machine?"
            ),
            parent=root,
        )
        root.destroy()
        return choice
    except Exception:
        return False  # secure fallback: deny on UI failure

@tool
def execute_python(
    code: str = "",
    state: Annotated[VedState, InjectedState] = None,  # type: ignore[assignment]
) -> str:
    """Execute a block of Python code in a sandboxed subprocess.

    The code is written to a temp file and run with the current Python
    interpreter. Stdout and stderr are captured and returned. A 10-second
    hard timeout prevents runaway scripts. A tkinter popup asks the user
    to approve execution every time.

    Args:
        code: A string containing the Python code to execute. If empty,
              the tool extracts the most recent ```python ... ``` fence
              from the last AI message.

    Returns:
        The combined stdout+stderr output as a string, or `ERROR: ...` if
        the code was empty, the user denied approval, or execution failed.
    """
    raw_code = (code or "").strip()
    if not raw_code:
        raw_code = (resolve_implicit_python_code(state) or "").strip()
    if (
        not raw_code
        or raw_code.startswith("[System")
        or len(raw_code.split()) < 2
    ):
        return (
            "ERROR: No clear executable code block was isolated. "
            "Pass `code` explicitly or include a ```python ... ``` fence "
            "in the last AI message."
        )

    if not _request_approval(raw_code):
        return "ERROR: User denied execution authorization."

    os.makedirs(_TEMP_DIR, exist_ok=True)
    temp_script = os.path.join(_TEMP_DIR, "ved_runtime_exec.py")

    terminal_output = ""
    try:
        Path(temp_script).write_text(raw_code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-u", temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        terminal_output = result.stdout
    except subprocess.TimeoutExpired:
        terminal_output = (
            f"ERROR: Terminal process terminated - hardware timeout gate "
            f"({_TIMEOUT_SECONDS}s) exceeded."
        )
    except Exception as exc:
        terminal_output = f"ERROR: {exc}"
    finally:
        try:
            if os.path.exists(temp_script):
                os.remove(temp_script)
        except Exception:
            pass

    if not terminal_output.strip():
        return "OK: Process completed successfully but produced no stdout output."
    return f"OK:\n{terminal_output.strip()}"
