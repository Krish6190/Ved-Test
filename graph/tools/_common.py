"""Shared utilities for Ved's tool layer.

Centralizes:
  - Safety policy constants (system path prefixes, cache dirs to skip).
  - Path validation helpers (`is_system_path`, `is_safe_default`,
    `is_safe_self_healing`).
  - The implicit-target fallback (`resolve_implicit_target`) that lets
    the LLM omit a primary arg and have the tool discover the file from
    the conversation context.

All four tools (`read_file`, `edit_file`, `overwrite_file`, `search_files`,
`execute_python`) import from here to avoid duplication.
"""
import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_RESULTS = 100
ALWAYS_SKIP_DIRS = {
    # Python / venv
    ".venv", "venv", "__pycache__",
    # Version control
    ".git",
    # Node / JS
    "node_modules",
    # Build / dist outputs
    "dist", "build", ".tox", ".nox",
    # Pytest / test artifacts (avoid noise from leftover tmp dirs)
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage",
    "tests",  # test files shouldn't be search/edit targets by default
    # Caches
    ".cache", ".kimchi",
}
BLOCKED_ABSOLUTE_PREFIXES_WIN = (
    os.environ.get("SystemRoot", r"C:\Windows").rstrip("\\") + "\\",
    os.environ.get("ProgramFiles", r"C:\Program Files").rstrip("\\") + "\\",
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)").rstrip("\\") + "\\",
    os.environ.get("ProgramData", r"C:\ProgramData").rstrip("\\") + "\\",
)
BLOCKED_ABSOLUTE_PREFIXES_POSIX = ("/etc/", "/sys/", "/proc/", "/boot/", "/dev/")


def is_system_path(path: Path) -> bool:
    """True if `path` lives under a system directory the agent must never touch.

    Uses `Path.is_relative_to()` instead of raw `str.startswith` so the check
    is case-insensitive on Windows and handles env-var capitalization
    correctly (the old `startswith` form could be bypassed when `SystemRoot`
    returned a different case than the resolved path).
    """
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        for env_var, default in (
            ("SystemRoot", r"C:\Windows"),
            ("ProgramFiles", r"C:\Program Files"),
            ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            ("ProgramData", r"C:\ProgramData"),
        ):
            base = Path(os.environ.get(env_var, default))
            try:
                if resolved.is_relative_to(base):
                    return True
            except (OSError, ValueError):
                continue
        return False
    # POSIX: substring match is fine since these paths are lowercase by convention
    return any(str(resolved).startswith(p) for p in BLOCKED_ABSOLUTE_PREFIXES_POSIX)


def is_safe_default(path: Path) -> bool:
    """Default-mode safety: block system paths and other users' profiles."""
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    if is_system_path(resolved):
        return False
    if os.name == "nt":
        users = Path(os.environ.get("SystemDrive", "C:")) / "Users"
        try:
            if resolved.is_relative_to(users):
                current_user = os.environ.get("USERNAME", "").lower()
                if (
                    len(resolved.parts) >= 3
                    and current_user
                    and resolved.parts[2].lower() != current_user
                ):
                    return False
        except ValueError:
            pass
    return True


def is_safe_self_healing(path: Path) -> bool:
    """Self-healing-mode safety: anchor strictly inside the project root."""
    try:
        return path.resolve().is_relative_to(PROJECT_ROOT)
    except (OSError, ValueError):
        return False


def last_user_message_text(state) -> str:
    """Return the content of the most recent HumanMessage, or '' if none."""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def last_ai_message_text(state) -> str:
    """Return the content of the most recent AIMessage, or '' if none."""
    from langchain_core.messages import AIMessage

    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def extract_search_pattern(text: str) -> str | None:
    """Pull a likely filename pattern out of a user message.

    Looks for word.ext (e.g., "config.py") first; falls back to the last
    3+ char word. Returns None if nothing useful is found.
    """
    matches = re.findall(r"\b[\w\-]+\.[a-zA-Z0-9]{1,5}\b", text)
    if matches:
        return matches[-1]
    words = re.findall(r"\b\w{3,}\b", text)
    if words:
        return words[-1]
    return None


def _iter_case_insensitive(base: Path, needle: str):
    """Case-insensitive rglob: walk `base` and yield paths whose name contains
    `needle` (case-insensitive). Returns a generator to avoid loading everything
    into memory."""
    needle_lower = needle.lower()
    try:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if needle_lower in p.name.lower():
                yield p
    except Exception:
        return


def _iter_substring(base: Path, needle: str):
    """Substring fallback: yield files whose full path contains `needle` (case-insensitive)."""
    needle_lower = needle.lower()
    try:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if needle_lower in str(p).lower():
                yield p
    except Exception:
        return


def find_file(base: Path, pattern: str) -> Path | None:
    """Find a single file matching `pattern` under `base` using 3 strategies:

      1. Exact `rglob(pattern)` (handles wildcards and exact filenames).
      2. Case-insensitive name-substring match (handles `readme` -> README.md).
      3. Case-insensitive full-path substring match (last-resort fuzzy).

    Cache dirs (`.venv`, `.git`, etc.) are always skipped. Returns the first
    match, or None if nothing is found.
    """
    def _clean(p: Path) -> bool:
        return not any(part in ALWAYS_SKIP_DIRS for part in p.parts)

    # Strategy 1: exact glob
    try:
        for p in base.rglob(pattern):
            if _clean(p):
                return p
    except Exception:
        pass

    # Strategy 2: case-insensitive name contains needle
    try:
        for p in _iter_case_insensitive(base, pattern):
            if _clean(p):
                return p
    except Exception:
        pass

    # Strategy 3: full-path substring (last resort)
    try:
        for p in _iter_substring(base, pattern):
            if _clean(p):
                return p
    except Exception:
        pass

    return None


def find_files(base: Path, pattern: str, limit: int = MAX_RESULTS) -> list[str]:
    """Find all files matching `pattern` under `base` (all 3 strategies).

    Used by `search_files` so the fallback path benefits from case-insensitive
    matching just like single-file resolution. Capped at `limit` results.
    """
    seen: set[str] = set()
    matches: list[str] = []

    def _add(p: Path) -> None:
        if any(part in ALWAYS_SKIP_DIRS for part in p.parts):
            return
        s = str(p)
        if s in seen:
            return
        seen.add(s)
        matches.append(s)

    # Strategy 1: exact glob
    try:
        for p in base.rglob(pattern):
            _add(p)
            if len(matches) >= limit:
                return matches
    except Exception:
        pass

    # Strategy 2: case-insensitive name contains needle
    try:
        for p in _iter_case_insensitive(base, pattern):
            _add(p)
            if len(matches) >= limit:
                return matches
    except Exception:
        pass

    # Strategy 3: full-path substring (only if still under limit)
    if len(matches) < limit:
        try:
            for p in _iter_substring(base, pattern):
                _add(p)
                if len(matches) >= limit:
                    break
        except Exception:
            pass

    return matches


def resolve_implicit_target(state, hint: str = "") -> str | None:
    """Discover the file the user means when the LLM omits a primary arg.

    Strategy:
      1. If `hint` is provided, return it directly.
      2. Extract a filename pattern from the last user message.
      3. Use `find_file()` (3-strategy) to locate the file under the
         project root (self-healing mode) or cwd (default mode).
      4. Return the first match as a string, or None.

    The LLM no longer has to spell out the path - "show me the readme"
    finds README.md, "open requirements" finds requirements.txt, etc.
    """
    if hint:
        return hint

    text = last_user_message_text(state)
    if not text:
        return None

    pattern = extract_search_pattern(text)
    if not pattern:
        return None

    self_healing = bool(getattr(state, "self_healing", False))
    base = PROJECT_ROOT if self_healing else Path.cwd()

    found = find_file(base, pattern)
    return str(found) if found else None


def resolve_implicit_python_code(state) -> str | None:
    """Pull a Python code block out of the last AI message if any.

    Used by execute_python when the LLM omits `code` — most commonly
    when the AI just wrote ```python ... ``` in its response.
    """
    text = last_ai_message_text(state)
    if not text:
        return None
    fence = re.search(r"```python\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    fence = re.search(r"```\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    return None
