"""Launch any application on the system by name.

The `open_app(query)` @tool:
  1. Resolves the query against the action's search helpers (Start Menu,
     install dirs, PATH, .desktop entries) to compute the best launch
     candidate.
  2. Shows a human approval request with the resolved path.
  3. Delegates the actual launch to `open_app_action` in graph/actions/.

Like other tools, it gates every launch behind a human approval request.
On an active FastAPI chat session the approval is routed through the SSE
bus (`event: approval_request {kind: "app_launch"}`) so the web UI can
show a modal; in the Tkinter desktop UI it falls back to a yes/no popup.
"""
from __future__ import annotations
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from graph.actions.apps import (
    open_app_action,
    _resolve_candidates,
    _search_windows,
    _search_linux_macos,
    _windows_install_dirs,
    _windows_start_menu_dirs,
)

# Re-exported for back-compat with tests that monkeypatch these names on
# this module (e.g. tests/test_app_launcher.py).
__all__ = [
    "open_app",
    "_resolve_candidates",
    "_search_windows",
    "_search_linux_macos",
    "_windows_install_dirs",
    "_windows_start_menu_dirs",
]

_APPROVAL_LOCK = threading.Lock()


def _request_approval(query: str, resolved_path: str, config: Optional[RunnableConfig]) -> bool:
    """Request approval before launching a user-named application.

    Routing:
      1. If an active FastAPI chat session is wired via `config`, emit an
         `approval_request` SSE event with `kind: "app_launch"` and block
         on the existing `_human_approval_event`.
      2. Otherwise, fall back to a tkinter popup (desktop UI).
    """
    try:
        conf = ((config or {}).get("configurable", {}) or {}) if config else {}
        token_queue = conf.get("token_queue")
        approval_event = conf.get("approval_event")
        approval_state = conf.get("approval_state")
        if token_queue is not None and approval_event is not None and approval_state is not None:
            try:
                token_queue.put(("approval_request", {
                    "kind": "app_launch",
                    "query": query,
                    "resolved_path": resolved_path,
                }))
            except Exception:
                return False
            approval_event.wait()
            return bool((approval_state or {}).get("value"))
    except Exception:
        pass
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        choice = messagebox.askyesno(
            title="Ved App Launch Approval",
            message=(
                f"Ved is requesting permission to launch this application:\n\n"
                f"Name: {query}\n"
                f"Path: {resolved_path}\n\n"
                "Authorize launching this app?"
            ),
            parent=root,
        )
        root.destroy()
        return choice
    except Exception:
        return False


@tool
def open_app(
    query: str,
    config: Annotated[RunnableConfig, "injected"] = None,
) -> str:
    """Open any application on the system by name (e.g. "discord", "steam", "firefox").
    Searches the Windows Start Menu, common install directories, and the
    system PATH to find a match, then launches it via the OS shell
    (non-blocking). The launch is gated behind a human approval request
    every time - opens are destructive in that they start persistent
    processes that the user must close manually.
    Args:
      query: free-form app name. Nicknames accepted ("vscode", "chrome").
    Returns:
      OK message with the resolved path on success, ERROR otherwise.
    """
    if not query or not query.strip():
        return "ERROR: open_app requires a non-empty app name."

    # Resolve locally so the approval popup can show the resolved path.
    # The action re-resolves internally before launch; this preview is
    # purely informational for the human.
    matches = _resolve_candidates(query.strip())
    preview_path = matches[0][0] if matches else "(no match)"

    with _APPROVAL_LOCK:
        if not _request_approval(query.strip(), preview_path, config):
            return "ERROR: User denied app launch."

    return open_app_action(query.strip())
