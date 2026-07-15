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
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_RESULTS = 100

# Backup/tmp extensions that must NEVER appear in search, glob, or RAG
# results. Chunk 1 of the structural-repair plan: backup artifacts
# (`*.bak`, `*.tmp`) are filtered at every boundary so the agent cannot
# read, edit, or re-index stale backup copies.
ALWAYS_SKIP_SUFFIXES = {".bak", ".tmp"}


def is_backup_artifact(path: Path) -> bool:
    """True if the file name ends with a backup/tmp extension.

    Matching is case-insensitive on the suffix (`.BAK` == `.bak`) so we
    catch editor-generated uppercase variants as well. Returns False for
    paths that are not file-like or have no extension at all.
    """
    try:
        return path.suffix.lower() in ALWAYS_SKIP_SUFFIXES
    except Exception:
        return False
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
        if any(part in ALWAYS_SKIP_DIRS for part in p.parts):
            return False
        if is_backup_artifact(p):
            return False
        return True

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
        if is_backup_artifact(p):
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


_RETRIEVE_RAG_PATH_KEYS = ("file_path", "path", "directory", "dir", "folder")


def _normalize_retrieve_rag_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Map malformed retrieve_rag payloads onto the required ``query`` field."""
    out = dict(args or {})
    out.pop("config", None)

    query = out.get("query")
    if isinstance(query, str) and query.strip():
        paths = out.get("paths")
        if isinstance(paths, str) and paths.strip():
            out["paths"] = [paths.strip()]
        return out

    derived = None
    paths = out.get("paths")
    if paths is not None:
        if isinstance(paths, list):
            derived = " ".join(str(p).strip() for p in paths if p)
            out["paths"] = [str(p) for p in paths if p]
        elif isinstance(paths, str) and paths.strip():
            derived = paths.strip()
            out["paths"] = [paths.strip()]

    if not derived:
        for key in _RETRIEVE_RAG_PATH_KEYS:
            val = out.get(key)
            if isinstance(val, str) and val.strip():
                derived = val.strip()
                break
            if isinstance(val, list) and val:
                derived = " ".join(str(v).strip() for v in val if v)
                break

    if not derived:
        raw = out.get("_raw")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return _normalize_retrieve_rag_args(parsed)
            except Exception:
                derived = raw.strip()

    if not derived:
        for key, val in out.items():
            if key in ("k", "scope", "layer"):
                continue
            if isinstance(val, str) and val.strip():
                derived = val.strip()
                break

    out["query"] = derived or "project files"
    return out


def normalize_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM tool-call args before Pydantic schema validation."""
    if not isinstance(args, dict):
        return {}
    if tool_name == "retrieve_rag":
        return _normalize_retrieve_rag_args(args)
    return dict(args)


def coerce_retrieve_rag_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Second-chance retrieve_rag normalization after a ValidationError."""
    return _normalize_retrieve_rag_args(args or {})


def ingest_path_to_thread_rag(path: str, thread_id: str) -> None:
    """Trigger background ingestion of `path` into the thread-scoped RAG index.

    Returns immediately; a daemon thread performs the actual embedding and
    commit so the tool call is not blocked on the vector store. Failures
    are swallowed — RAG ingestion is best-effort and must never break a
    file read/search tool.
    """
    if not path or not thread_id:
        return

    def _run() -> None:
        try:
            from graph.rag.vector_engine import LocalVectorDB

            db = LocalVectorDB()
            db.ingest_local_file(
                path,
                scope=thread_id,
                chunker="ast",
                source=os.path.basename(path),
            )
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def ingest_path_to_thread_rag_sync(
    path: str,
    thread_id: str,
    content: str | None = None,
    *,
    chunker: str = "ast",
) -> None:
    """Synchronously re-embed a file (or virtual text) into the thread RAG index.

    Used by rollback so the thread index reflects the reverted virtual state
    immediately instead of waiting on a background daemon thread.
    """
    if not thread_id:
        return
    try:
        from graph.rag.vector_engine import LocalVectorDB

        db = LocalVectorDB()
        source = os.path.basename(path) if path else "staged_file"
        if content is not None:
            db.ingest_text(content, scope=thread_id, source=source, chunker=chunker)
        elif path:
            db.ingest_local_file(path, scope=thread_id, chunker=chunker, source=source)
    except Exception:
        pass


def _resolve_fuzzy_path(path: str, base: Path | None = None) -> str | None:
    """If `path` does not exist on disk, walk its components from the
    closest existing ancestor and try to fix each one by substring
    containment. Returns the corrected absolute path, or None when no
    candidate qualifies.

    Matching rule: a directory or file is a candidate when the provided
    component appears as a contiguous substring of its name (case
    insensitive). The shortest-name candidate wins (closest exact match).

    Args:
        path: A filesystem path string that may or may not exist.
        base: Optional base directory to anchor from (defaults to cwd).

    Returns:
        The resolved absolute path of the closest matching entry, or None.
    """
    if not path:
        return None
    candidate = Path(path)
    try:
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        return None

    # Find the closest existing ancestor of the candidate path
    existing_ancestor = None
    ancestors = list(candidate.parents)[::-1] + [candidate]
    for anc in reversed(ancestors):
        try:
            if anc.exists():
                existing_ancestor = anc
                break
        except OSError:
            continue
    if existing_ancestor is None:
        return None

    # Walk from existing_ancestor down, fuzzy-matching each component
    # until we reach the candidate's level
    try:
        rel_to_ancestor = candidate.relative_to(existing_ancestor).parts
    except ValueError:
        rel_to_ancestor = candidate.parts

    if not rel_to_ancestor:
        return None

    current = existing_ancestor
    for part in rel_to_ancestor:
        needle = part.lower()
        if not needle:
            return None
        matches: list[Path] = []
        try:
            for entry in current.iterdir():
                if needle in entry.name.lower():
                    matches.append(entry)
        except OSError:
            return None
        if not matches:
            return None
        best = min(matches, key=lambda e: len(e.name))
        current = current / best.name

    try:
        return str(current.resolve()) if current.exists() else None
    except OSError:
        return None
