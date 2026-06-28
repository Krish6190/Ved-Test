"""User-created tools directory.

Auto-loaded at import time. Existing user tools are added to VED_TOOLS
so they're available immediately on startup. New tools added during a
running session are dynamically imported via `propose_tool`.
"""
from __future__ import annotations
import importlib
from pathlib import Path
from graph.tools._common import PROJECT_ROOT
USER_TOOLS_DIR = PROJECT_ROOT / "graph" / "tools" / "user_tools"
USER_TOOLS = []  # populated below; exposed for tests/introspection.

def _load_all() -> None:
    """Import every *.py file in this directory (excluding __init__.py and
    files starting with '_') and register any @tool functions found.
    Safe to call multiple times — duplicates are skipped by _register_user_tool.
    """
    from graph.tools.tool_creator import _register_user_tool
    if not USER_TOOLS_DIR.exists():
        return
    for path in sorted(USER_TOOLS_DIR.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        stem = path.stem
        try:
            module = importlib.import_module(f"graph.tools.user_tools.{stem}")
            registered = _register_user_tool(module)
            USER_TOOLS.extend(registered)
        except Exception:
            continue

_load_all()
