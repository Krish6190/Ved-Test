"""Sandboxed Python execution tool for Ved.

LangChain `@tool`-formatted. The agent binds this via `llm.bind_tools([...])`
and emits a structured `execute_python(code=...)` call when it wants to run
code. State (currently unused but kept for consistency / future hardening)
is injected via `InjectedState`.

The tool:
  1. Shows a tkinter approval popup showing the code (user must OK).
  2. Delegates the actual subprocess run to `execute_python_action` in
     graph/actions/, which writes the code to a temp file, runs it with
     the current Python interpreter, captures stdout+stderr, and enforces
     a 10-second hard timeout.
  3. Formats the action's structured result back into the string format
     the LLM expects.

The `code` argument is OPTIONAL. If omitted, the tool scans the last AI
message for a ```python ... ``` fence and uses that body. This makes it
trivial for the agent to execute code it just emitted without re-typing it.

The user must explicitly approve every execution - this is destructive and
untrusted by definition.
"""
import tkinter as tk
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.actions.process import execute_python_action
from graph.state import VedState
from graph.tools._common import resolve_implicit_python_code

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


def _format_action_result(result: dict) -> str:
    """Convert the action's dict result into the legacy string format.

    This keeps the existing tool contracts intact (test_tools.py asserts on
    specific prefixes like "OK:", "ERROR:", and "timeout").
    """
    if result.get("timed_out"):
        return (
            f"ERROR: Terminal process terminated - hardware timeout gate "
            f"({_TIMEOUT_SECONDS}s) exceeded."
        )
    stdout = (result.get("stdout") or "").strip()
    exit_code = result.get("exit_code", 0)
    if exit_code != 0:
        # Surface whatever the process produced plus the failure signal.
        if stdout:
            return f"ERROR: {stdout}"
        return f"ERROR: Process exited with code {exit_code}"
    if not stdout:
        return "OK: Process completed successfully but produced no stdout output."
    return f"OK:\n{stdout}"


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

    result = execute_python_action(raw_code, timeout_seconds=_TIMEOUT_SECONDS)
    return _format_action_result(result)
