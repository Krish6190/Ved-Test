"""Application-launch action primitives.

Pure-Python search-and-launch for desktop apps. The tool layer owns the
human-approval gate; the action performs the actual filesystem search
across Start Menu / install dirs / PATH and launches the best match.

Module rules:
  - No upward imports into the tool or state layers, or into the langchain or data modules.
  - Only primitive argument types (str).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# Minimum score for a candidate to be considered a confident match.
_MIN_CONFIDENCE_SCORE = 10

# Common nicknames -> canonical app names. Used to rank fuzzy matches.
_NICKNAMES = {
    "vscode": "visual studio code",
    "vs code": "visual studio code",
    "code": "visual studio code",
    "chrome": "google chrome",
    "edge": "microsoft edge",
    "ff": "firefox",
    "firefox": "mozilla firefox",
    "word": "microsoft word",
    "excel": "microsoft excel",
    "ppt": "microsoft powerpoint",
    "powerpoint": "microsoft powerpoint",
    "teams": "microsoft teams",
}


# ---------------------------------------------------------------------------
# Platform-specific search helpers
# ---------------------------------------------------------------------------

def _windows_start_menu_dirs() -> List[Path]:
    """Return the user + system Start Menu directories on Windows."""
    if sys.platform != "win32":
        return []
    candidates: list[Path] = []
    program_data = os.environ.get("ProgramData", "C:\\ProgramData")
    candidates.append(
        Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    )
    appdata = os.environ.get("AppData", "")
    if appdata:
        candidates.append(
            Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
    return [d for d in candidates if d.exists()]


def _windows_install_dirs() -> List[Path]:
    """Common install dirs on Windows. Scanned shallowly for speed."""
    if sys.platform != "win32":
        return []
    candidates: list[Path] = [
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
    Search order: Start Menu shortcuts > install-dir .exe > PATH .exe.
    """
    q = query.lower().strip()
    if not q:
        return []
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
                    exes = sorted(
                        full.glob("*.exe"),
                        key=lambda p: p.stat().st_size if p.exists() else 0,
                        reverse=True,
                    )
                    if exes:
                        candidates[str(exes[0])] = max(
                            candidates.get(str(exes[0]), 0), score
                        )
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
    """Linux/macOS search - looks for .desktop files and PATH executables."""
    q = query.lower().strip()
    if not q:
        return []
    candidates: dict[str, int] = {}
    desktop_dirs = [
        Path("/usr/share/applications"),
        Path("/usr/local/share/applications"),
        Path.home() / ".local" / "share" / "applications",
        Path("/Applications"),
    ]
    for d in desktop_dirs:
        if not d.exists():
            continue
        try:
            for desktop in d.rglob("*.desktop"):
                stem = desktop.stem.lower()
                if q == stem or q in stem:
                    candidates[str(desktop)] = max(
                        candidates.get(str(desktop), 0), 50
                    )
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


# ---------------------------------------------------------------------------
# Public action function
# ---------------------------------------------------------------------------

def open_app_action(app_name: str) -> str:
    """Resolve `app_name` to a launchable path and launch it.

    Performs the full search (Start Menu / install dirs / PATH /
    .desktop files) and launches the highest-scoring candidate on the
    current platform. This action performs NO human approval gate -
    the tool layer is responsible for confirming with the user before
    calling this function.

    Args:
        app_name: Free-form application name. Nicknames are accepted
                  ("vscode", "chrome", "ff", etc.).

    Returns:
        Status string: "OK: Launched '<query>' from <path>" on success,
        or "ERROR: ..." with a descriptive reason on failure.
    """
    if not app_name or not app_name.strip():
        return "ERROR: open_app requires a non-empty app name."

    matches = _resolve_candidates(app_name.strip())
    if not matches:
        return (
            f"ERROR: No application found matching '{app_name}'. "
            "Try a different name, or make sure the app is installed and "
            "has a Start Menu shortcut (Windows) / .desktop entry (Linux)."
        )

    best_path, best_score = matches[0]
    if best_score < _MIN_CONFIDENCE_SCORE:
        return (
            f"ERROR: No confident match for '{app_name}'. "
            f"Closest: {Path(best_path).name}. "
            "Try the app's exact name."
        )

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

    return f"OK: Launched '{app_name}' from {best_path}"
