"""Tool creation: let the LLM propose a new tool at runtime.

The `propose_tool` @tool pauses for human approval via the
`_tool_creation_event` on the chatbot, then on approval writes the new
tool to `graph/tools/user_tools/<name>.py` and dynamically imports it
into `VED_TOOLS` so the same conversation can use it immediately.

Companion to: data/thread_files.py, graph/tools/_common.py.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import threading
from pathlib import Path
from typing import Annotated, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from graph.tools._common import PROJECT_ROOT

USER_TOOLS_DIR = PROJECT_ROOT / "graph" / "tools" / "user_tools"

# ---- safety: blocked imports ----
_BLOCKED_IMPORT_PATTERNS = (
    "subprocess", "ctypes", "socket", "multiprocessing",
    "os.system", "shutil.rmtree", "pty", "fcntl",
    "win32api", "win32com",
)


def _scan_blocked_imports(code: str) -> List[str]:
    """Parse code and return a list of blocked import module names found.
    Returns [] if code is clean. Returns ['SYNTAX_ERROR: ...'] on parse error."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SYNTAX_ERROR: {exc}"]
    blocked = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = (alias.name or "").split(".")[0]
                if top in _BLOCKED_IMPORT_PATTERNS or alias.name in _BLOCKED_IMPORT_PATTERNS:
                    blocked.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _BLOCKED_IMPORT_PATTERNS or module in _BLOCKED_IMPORT_PATTERNS:
                blocked.append(module)
            for alias in (node.names or []):
                full = f"{module}.{alias.name}" if module else alias.name
                if full in _BLOCKED_IMPORT_PATTERNS:
                    blocked.append(full)
    return sorted(set(blocked))


# ---- filename safety ----
_SAFE_NAME_RE = re.compile(r"[^a-z0-9_]+")


def _safe_filename(name: str) -> str:
    """snake_case the name, max 60 chars, must start with a letter."""
    s = (name or "").strip().lower()
    s = _SAFE_NAME_RE.sub("_", s).strip("_")
    s = re.sub(r"^[^a-z]+", "", s)
    s = s[:60] or "unnamed_tool"
    return s


# ---- registration helpers ----
_TOOLS_LOCK = threading.Lock()


def _find_tool_decorated(module) -> list:
    """Return every LangChain @tool object defined in `module`."""
    from langchain_core.tools import BaseTool
    tools = []
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, BaseTool):
            tools.append(obj)
    return tools


def _register_user_tool(module) -> List[str]:
    """Append every @tool in `module` to VED_TOOLS. Returns names registered.
    Thread-safe."""
    new_tools = _find_tool_decorated(module)
    if not new_tools:
        return []
    with _TOOLS_LOCK:
        # Local import to avoid circular import at module load.
        from graph.tools import VED_TOOLS
        existing = {getattr(t, "name", None) for t in VED_TOOLS}
        for t in new_tools:
            tname = getattr(t, "name", None)
            if tname not in existing:
                VED_TOOLS.append(t)
                existing.add(tname)
    return [getattr(t, "name", repr(t)) for t in new_tools]


@tool
def propose_tool(
    name: str,
    description: str,
    code: str,
    config: Annotated[RunnableConfig, "injected"],
) -> str:
    """Propose a new tool to be created and added to Ved's tool registry.

    Use this tool when:
      (a) the user explicitly asks "create a tool that does X", OR
      (b) you realize you lack a tool to satisfy the user's request.

    The human will be shown the proposed code in a modal and asked to
    approve. On approval, the file is saved to
    `graph/tools/user_tools/<safe_name>.py` and dynamically registered
    in VED_TOOLS so you can call it immediately.

    Args:
      name: snake_case function name (e.g. "fetch_weather").
      description: one-line description of when to use this tool.
      code: full Python source for the @tool function, including the
            `@tool` decorator and a docstring. Must parse as valid
            Python. Imports are restricted (no subprocess, socket, etc).
    """
    safe_name = _safe_filename(name)
    if not safe_name or safe_name == "unnamed_tool":
        return "ERROR: Tool name is empty after sanitization. Provide a valid snake_case name like 'fetch_weather'."

    # Check the existing code for syntax errors and blocked imports BEFORE
    # bothering the human with a proposal.
    blocked = _scan_blocked_imports(code)
    if blocked:
        return (
            f"ERROR: Tool code contains blocked imports: {blocked}. "
            "User must edit and retry. Allowed: stdlib (excluding the "
            "blocked list) and packages already in requirements.txt."
        )

    target_path = USER_TOOLS_DIR / f"{safe_name}.py"
    if target_path.exists():
        return f"ERROR: A tool named '{safe_name}' already exists at {target_path}. Pick a different name."

    try:
        cfg = (config or {}).get("configurable", {}) or {}
    except Exception:
        cfg = {}
    token_queue = cfg.get("token_queue")
    tool_event = cfg.get("tool_creation_event")
    tool_state = cfg.get("tool_creation_state")

    if token_queue is None or tool_event is None or tool_state is None:
        return (
            "ERROR: propose_tool invoked outside of an active chat session. "
            "The tool requires the chatbot's approval wiring to be present."
        )

    proposal_payload = {
        "tool_name": safe_name,
        "description": description or "(no description provided)",
        "code": code,
        "sample_invocation": f"{safe_name}()",
    }

    # Emit the proposal so the SSE pump can relay it to the UI.
    try:
        token_queue.put(("tool_creation_proposal", proposal_payload))
    except Exception:
        pass

    # Block until the human (or test) resolves the approval.
    tool_event.wait()

    approved = bool((tool_state or {}).get("value"))
    if not approved:
        return f"User rejected tool creation for '{safe_name}'. Proceed without it."

    # Human approved: write, import, register.
    try:
        USER_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        target_path.write_text(code, encoding="utf-8")
    except Exception as exc:
        return f"ERROR: Failed to write {target_path}: {exc}"

    try:
        module = importlib.import_module(f"graph.tools.user_tools.{safe_name}")
    except Exception as exc:
        # Roll back the file write so we don't leave a broken module.
        try:
            target_path.unlink()
        except Exception:
            pass
        return f"ERROR: Failed to import the new tool module: {exc}"

    try:
        registered = _register_user_tool(module)
    except Exception as exc:
        return f"ERROR: Failed to register the new tool: {exc}"

    if not registered:
        # File written but no @tool found inside it. Roll back.
        try:
            target_path.unlink()
        except Exception:
            pass
        return (
            f"ERROR: The provided code does not define any @tool-decorated "
            f"function. Add a `@tool` decorator to the function you want "
            f"registered (e.g. `{safe_name}`)."
        )

    return (
        f"OK: Tool '{safe_name}' registered and ready. "
        f"You may now invoke it as {safe_name}(...). "
        f"Module path: {target_path}"
    )
