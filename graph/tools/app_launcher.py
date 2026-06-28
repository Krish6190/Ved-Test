"""Launch any application on the system by name.

The `open_app(query)` @tool searches for the app across:
  - Windows Start Menu shortcuts (.lnk files)
  - System PATH
  - Common install directories (Program Files, AppData\\Local\\Programs)

Then launches the best match via the OS shell (non-blocking subprocess).

Like other tools, it gates every launch behind a human approval request.
On an active FastAPI chat session the approval is routed through the SSE
bus (`event: approval_request {kind: "app_launch"}`) so the web UI can
show a modal; in the Tkinter desktop UI it falls back to a yes/no popup.
"""
from __future__ import annotations
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, List, Optional, Tuple
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

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
        import tkinter as tk
        from tkinter import messagebox
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
def _windows_start_menu_dirs() -> List[Path]:
    """Return the user + system Start Menu directories on Windows."""
    if sys.platform != "win32":
        return []
    candidates = []
    program_data = os.environ.get("ProgramData", "C:\\ProgramData")
    candidates.append(Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    appdata = os.environ.get("AppData", "")
    if appdata:
        candidates.append(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return [d for d in candidates if d.exists()]

def _windows_install_dirs() -> List[Path]:
    """Common install dirs on Windows. Scanned shallowly for speed."""
    if sys.platform != "win32":
        return []
    candidates = [
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]
    local = os.environ.get("LocalAppData", "")
    if local:
        candidates.append(Path(local) / "Programs")
    return [d for d in candidates if d.exists()]

def _search_windows(query: str) -> List[Tuple[str, int]]:
    """Return ranked Windows candidate launches for `query`.
    Each entry is (path_to_launch, score). Higher score = better match.
    """
    q = query.lower().strip()
    if not q:
        return []
    _NICKNAMES = {
        "vscode": "visual studio code",
        "vs code": "visual studio code",
        "chrome": "google chrome",
        "edge": "microsoft edge",
        "ff": "firefox",
    }
    canonical = _NICKNAMES.get(q, q)
    candidates: dict[str, int] = {}
    for d in _windows_start_menu_dirs():
        try:
            for lnk in d.rglob("*.lnk"):
                stem = lnk.stem.lower()
                if not stem:
                    continue
                if canonical == stem or q == stem:
                    candidates[str(lnk)] = max(candidates.get(str(lnk), 0), 100)
                elif canonical in stem or q in stem:
                    candidates[str(lnk)] = max(candidates.get(str(lnk), 0), 50)
                elif stem in canonical:
                    candidates[str(lnk)] = max(candidates.get(str(lnk), 0), 20)
        except Exception:
            continue
    for d in _windows_install_dirs():
        try:
            for entry in os.listdir(d):
                entry_lower = entry.lower()
                if not entry_lower:
                    continue
                full = d / entry
                if not full.is_dir():
                    continue
                score = 0
                if canonical == entry_lower or q == entry_lower:
                    score = 80
                elif canonical in entry_lower or q in entry_lower:
                    score = 30
                if score > 0:
                    exes = sorted(full.glob("*.exe"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                    if exes:
                        candidates[str(exes[0])] = max(candidates.get(str(exes[0]), 0), score)
        except Exception:
            continue
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        if not dir_path or not os.path.isdir(dir_path):
            continue
        try:
            for entry in os.listdir(dir_path):
                name_lower = entry.lower()
                if not (name_lower.endswith(".exe") or sys.platform != "win32"):
                    continue
                stem = name_lower.rsplit(".", 1)[0]
                if canonical == stem or q == stem:
                    full = os.path.join(dir_path, entry)
                    candidates[full] = max(candidates.get(full, 0), 60)
                elif (canonical in stem or q in stem) and len(stem) <= len(canonical) + 6:
                    full = os.path.join(dir_path, entry)
                    candidates[full] = max(candidates.get(full, 0), 25)
        except Exception:
            continue

    return sorted(candidates.items(), key=lambda kv: -kv[1])

def _search_linux_macos(query: str) -> List[Tuple[str, int]]:
    """Linux/macOS search — looks for .desktop files and PATH executables.

    Much shallower than the Windows path because the .desktop convention
    is the canonical user-launch entry point on Linux.
    """
    q = query.lower().strip()
    if not q:
        return []
    candidates: dict[str, int] = {}
    desktop_dirs = [
        Path("/usr/share/applications"),
        Path("/usr/local/share/applications"),
        Path.home() / ".local" / "share" / "applications",
        Path("/Applications"),  # macOS
    ]
    for d in desktop_dirs:
        if not d.exists():
            continue
        try:
            for desktop in d.rglob("*.desktop"):
                stem = desktop.stem.lower()
                if q == stem or q in stem:
                    candidates[str(desktop)] = max(candidates.get(str(desktop), 0), 50)
        except Exception:
            continue
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        if not dir_path or not os.path.isdir(dir_path):
            continue
        try:
            for entry in os.listdir(dir_path):
                full = os.path.join(dir_path, entry)
                if os.path.isfile(full) and os.access(full, os.X_OK):
                    stem = entry.lower()
                    if q == stem or q in stem:
                        candidates[full] = max(candidates.get(full, 0), 30)
        except Exception:
            continue
    return sorted(candidates.items(), key=lambda kv: -kv[1])

def _resolve_candidates(query: str) -> List[Tuple[str, int]]:
    """Platform-appropriate app search."""
    if sys.platform == "win32":
        return _search_windows(query)
    return _search_linux_macos(query)

@tool
def open_app(
    query: str,
    config: Annotated[RunnableConfig, "injected"] = None,
) -> str:
    """Open any application on the system by name (e.g. "discord", "steam", "firefox").
    Searches the Windows Start Menu, common install directories, and the
    system PATH to find a match, then launches it via the OS shell
    (non-blocking). The launch is gated behind a human approval request
    every time — opens are destructive in that they start persistent
    processes that the user must close manually.
    Args:
      query: free-form app name. Nicknames accepted ("vscode", "chrome").
    Returns:
      OK message with the resolved path on success, ERROR otherwise.
    """
    if not query or not query.strip():
        return "ERROR: open_app requires a non-empty app name."

    matches = _resolve_candidates(query.strip())
    if not matches:
        return (
            f"ERROR: No application found matching '{query}'. "
            "Try a different name, or make sure the app is installed and "
            "has a Start Menu shortcut (Windows) / .desktop entry (Linux)."
        )

    best_path, best_score = matches[0]
    if best_score < 10:
        return (
            f"ERROR: No confident match for '{query}'. "
            f"Closest: {Path(best_path).name}. "
            "Try the app's exact name."
        )

    if not _request_approval(query.strip(), best_path, config):
        return "ERROR: User denied app launch."
    try:
        if sys.platform == "win32":
            os.startfile(best_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["open", best_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            subprocess.Popen(
                [best_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as exc:
        return f"ERROR: Failed to launch {best_path}: {exc}"

    return f"OK: Launched '{query}' from {best_path}"
