"""Tool-creation action primitives.

Pure-Python file write + dynamic import + registry registration. The
tool layer owns the safety scanning (blocked imports, name sanitization)
and the human approval gate; the action performs the destructive
filesystem side effects.

Module rules (enforced by chunk-1 acceptance grep):
  - No upward imports into the tool or state layers, or into the langchain or data modules.
  - Only primitive argument types (dict, str).
"""
from __future__ import annotations

import importlib
import inspect
import threading
from pathlib import Path
from typing import List


# Thread lock shared with the rest of the tool layer (declared at action
# layer to keep registration thread-safe regardless of caller).
_REGISTRATION_LOCK = threading.Lock()


def _find_tool_decorated(module) -> list:
    """Return every LangChain @tool object defined in `module`.

    Uses duck-typing instead of `isinstance(..., BaseTool)` so this action
    module has zero langchain_core imports (the chunk-1 acceptance grep
    forbids them). A LangChain @tool exposes `.name`, `.description`, and
    `.invoke`; any object in `module` with all three is treated as one.
    """
    tools: list = []
    for _name, obj in inspect.getmembers(module):
        if (
            callable(obj)
            and hasattr(obj, "name")
            and hasattr(obj, "invoke")
            and hasattr(obj, "description")
        ):
            tools.append(obj)
    return tools


def _register_into(module, target_list: list) -> List[str]:
    """Append every @tool in `module` to `target_list`. Returns names added.

    Thread-safe. The caller passes `target_list` explicitly so this action
    has no upward dependency on graph.tools.VED_TOOLS.
    """
    new_tools = _find_tool_decorated(module)
    if not new_tools:
        return []
    with _REGISTRATION_LOCK:
        existing = {getattr(t, "name", None) for t in target_list}
        for t in new_tools:
            tname = getattr(t, "name", None)
            if tname not in existing:
                target_list.append(t)
                existing.add(tname)
    return [getattr(t, "name", repr(t)) for t in new_tools]


def propose_tool_action(spec: dict, *, user_tools_dir: str) -> dict:
    """Write a user-proposed tool to disk and register it.

    The action assumes the caller has already:
      - sanitized the tool name
      - AST-scanned the code for blocked imports
      - requested (and received) human approval

    Args:
        spec: Dict with the keys:
              - "name" (str): sanitized snake_case tool name.
              - "code" (str): full Python source for the tool module.
              - "register_into" (list, optional): the VED_TOOLS list to
                append the discovered @tool-decorated functions into.
                If omitted, no registration is attempted (the action
                just writes + imports).
        user_tools_dir: Directory where the new tool file will be saved.

    Returns:
        A dict with keys:
          - "ok" (bool): True on full success.
          - "saved_to" (str | None): absolute path of the written file, or
            None if no file was written.
          - "error" (str | None): error description on failure, else None.
    """
    name = (spec or {}).get("name", "")
    code = (spec or {}).get("code", "")
    register_into = (spec or {}).get("register_into")
    if not name:
        return {"ok": False, "saved_to": None, "error": "spec.name is required"}
    if not code:
        return {"ok": False, "saved_to": None, "error": "spec.code is required"}

    target_path = Path(user_tools_dir) / f"{name}.py"

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(code, encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "saved_to": None, "error": f"Failed to write {target_path}: {exc}"}

    try:
        module = importlib.import_module(f"graph.tools.user_tools.{name}")
    except Exception as exc:
        # Roll back the file write so we don't leave a broken module.
        try:
            target_path.unlink()
        except Exception:
            pass
        return {
            "ok": False,
            "saved_to": None,
            "error": f"Failed to import the new tool module: {exc}",
        }

    if register_into is not None:
        try:
            registered = _register_into(module, register_into)
        except Exception as exc:
            return {
                "ok": False,
                "saved_to": str(target_path),
                "error": f"Failed to register the new tool: {exc}",
            }
        if not registered:
            # File written but no @tool found inside it. Roll back.
            try:
                target_path.unlink()
            except Exception:
                pass
            return {
                "ok": False,
                "saved_to": None,
                "error": (
                    "The provided code does not define any @tool-decorated "
                    "function. Add a `@tool` decorator to the function you "
                    f"want registered (e.g. `{name}`)."
                ),
            }

    return {"ok": True, "saved_to": str(target_path), "error": None}
