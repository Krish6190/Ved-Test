"""ExecutorRuntime — short-lived context manager for executor chunks.

Wraps the executor_node's agent loop in a try/finally that guarantees:
- Per-chunk scratch dict is cleared (no bleed across chunks)
- Tool instances are shallow-copied and released on exit
- Message buffer is bounded (no unbounded growth if the LLM loops)
- Per-chunk wall-clock timing is tracked for timeout enforcement
- A structured audit record (tool calls, final status, timing) is kept

The class itself is intentionally thin — it holds state and releases it.
The actual LLM/agent loop lives in executor_node and reads rt.* attributes.

Usage in executor_node (planned for Phase 3.2 refactor):
    with ExecutorRuntime(state, chunk, tools, wall_timeout=90.0) as rt:
        for iteration in range(_MAX_AGENT_ITERATIONS):
            if rt.is_over_timeout():
                rt.mark_failed("wall_timeout")
                break
            content, tool_calls = _stream_one_iteration(llm_with_tools, rt.messages, ...)
            ...
            for tc in tool_calls:
                result, ok = _invoke_tool_sync(tool, tc["args"])
                rt.log_tool_call(tc["name"], tc["args"], result, ok)
            rt.trim_messages()
        if not rt._final_status_was_set():
            rt.mark_done()
"""
from __future__ import annotations
import copy
import time
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
_MAX_MESSAGES = 50
_DEFAULT_WALL_TIMEOUT = 90.0

class ExecutorRuntime:
    """Short-lived context manager for one executor chunk.

    Holds per-chunk state (scratch dict, message buffer, tool copies) and
    releases everything cleanly on exit. Designed to make each chunk
    execution self-contained — no state leaks between chunks.

    Attributes (read-only contract for callers):
        state:          The VedState passed in at construction.
        chunk:          The chunk dict from the plan file.
        scratch:        Per-chunk mutable scratch space (cleared on exit).
        messages:       Per-chunk message buffer (cleared on exit).
        tools:          Shallow copies of the bound tools (cleared on exit).
        started_at:     Wall-clock time when __enter__ ran.
        ended_at:       Wall-clock time when __exit__ ran.
        wall_timeout:   Per-chunk timeout in seconds (default 90).
        tool_calls:     Structured log of every tool invocation this chunk.
        final_status:   "done" | "failed" | "unknown" — set via mark_done/mark_failed.
        final_error:    Error string if mark_failed was called.
    """

    def __init__(
        self,
        state: Any,
        chunk: Dict[str, Any],
        tools: List[BaseTool],
        wall_timeout: Optional[float] = None,
    ):
        self.state = state
        self.chunk = chunk
        self._tools_source = list(tools)
        # Mutable per-chunk state
        self.scratch: Dict[str, Any] = {}
        self.messages: List[Any] = []
        self.tools: List[BaseTool] = []
        self.tool_calls: List[Dict[str, Any]] = []
        # Timing
        self.started_at: float = 0.0
        self.ended_at: float = 0.0
        self.wall_timeout: float = (
            float(wall_timeout) if wall_timeout is not None else _DEFAULT_WALL_TIMEOUT
        )
        # Final status
        self.final_status: str = "unknown"
        self.final_error: str = ""

    # ---- Context manager protocol ----

    def __enter__(self) -> "ExecutorRuntime":
        self.started_at = time.time()
        self.tools = [_copy_tool(t) for t in self._tools_source]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.ended_at = time.time()
        for t in self.tools:
            try:
                closer = getattr(t, "close", None)
                if callable(closer):
                    closer()
            except Exception:
                pass
            try:
                resetter = getattr(t, "reset", None)
                if callable(resetter):
                    resetter()
            except Exception:
                pass
        self.scratch.clear()
        self.messages.clear()
        self.tools.clear()
        # Don't suppress exceptions.
        return None

    # ---- Audit helpers ----

    def log_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: str,
        ok: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record one tool invocation for the plan-file audit log."""
        self.tool_calls.append({
            "name": name,
            "args": args,
            "result": result,
            "ok": ok,
            "error": error,
            "at": time.time(),
        })

    def mark_done(self) -> None:
        """Mark the chunk as successfully completed."""
        self.final_status = "done"
        self.final_error = ""

    def mark_failed(self, error: str) -> None:
        """Mark the chunk as failed with the given error message."""
        self.final_status = "failed"
        self.final_error = error

    # ---- Timing helpers ----

    def elapsed(self) -> float:
        """Seconds since __enter__. Returns total elapsed even after exit."""
        if self.ended_at:
            return self.ended_at - self.started_at
        if self.started_at:
            return time.time() - self.started_at
        return 0.0

    def is_over_timeout(self) -> bool:
        """True if the chunk has exceeded its wall-clock budget."""
        return self.elapsed() > self.wall_timeout

    # ---- Buffer management ----

    def trim_messages(self) -> None:
        """Bound the message buffer to _MAX_MESSAGES. Keeps first + recent."""
        if len(self.messages) <= _MAX_MESSAGES:
            return
        head = self.messages[:1]
        tail = self.messages[-(self._MAX_MESSAGES - 1):]
        self.messages = head + tail

    def to_audit_record(self) -> Dict[str, Any]:
        """Serialize the runtime state for the plan-file chunk audit log."""
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": self.elapsed(),
            "wall_timeout": self.wall_timeout,
            "timed_out": self.is_over_timeout(),
            "final_status": self.final_status,
            "final_error": self.final_error,
            "tools_used": [tc["name"] for tc in self.tool_calls],
            "tool_call_count": len(self.tool_calls),
        }


def _copy_tool(tool: BaseTool) -> BaseTool:
    """Return a shallow copy of a LangChain tool.

    Most tools are thin wrappers around a function or a remote API; a
    shallow copy is enough to prevent accidental cross-chunk state
    mutation. If copy.copy fails (e.g., tool uses __slots__), fall back
    to the original — this is safe because most tools are stateless.
    """
    try:
        return copy.copy(tool)
    except Exception:
        return tool


__all__ = ["ExecutorRuntime", "_DEFAULT_WALL_TIMEOUT", "_MAX_MESSAGES"]
