"""Filesystem action primitives.

Pure-Python, stateless functions that the tool layer calls into. The action
takes already-validated primitive arguments and returns a string result. It
is the responsibility of the caller (the @tool in graph/tools/) to:

  - resolve implicit-target paths from the conversation state
  - enforce self_healing vs default safety policy
  - request human approval for destructive operations

Every action in this module validates that the resolved path lives inside
ONE of the `allowed_roots` passed by the caller. This is the last line of
defense before touching the filesystem: if the tool layer makes a mistake,
the action still refuses to escape the approved roots.

Module rules (enforced by the chunk-1 acceptance grep):
  - No upward imports into the tool or state layers, or into the langchain or data modules.
  - Only primitive argument types (str, int, tuple).
  - No global mutable state.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import TYPE_CHECKING

from graph.tools._common import is_backup_artifact

if TYPE_CHECKING:
    from graph.rag.diff_history import DiffHistoryStore

_MAX_READ_CHARS = 8000


# ---------------------------------------------------------------------------
# LLM text normalization (defensive against cloud-model tool-arg drift)
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r'^\s*```\w*\n|\n```\s*$', re.MULTILINE)


def _clean_llm_text(text: str) -> str:
    """Normalize LLM-provided text for matching/splicing against file content.

    Cloud LLMs (Qwen via OpenRouter especially) frequently:
      - Wrap code in ```python ... ``` markdown fences in tool args
      - Use CRLF line endings when the file has LF (or vice versa)
      - Add trailing whitespace from chat formatting

    This strips those artifacts so old_text/new_text reliably match what
    `read_file` would return. Idempotent on already-clean text — calling
    on a normal string is a no-op.
    """
    if not text:
        return text
    text = re.sub(r'^\s*```\w*\n?', '', text)   # strip leading fence
    text = re.sub(r'\n?```\s*$', '', text)      # strip trailing fence
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # normalize EOL
    return text


# ---------------------------------------------------------------------------
# Internal helpers (no LangChain / tool-layer deps)
# ---------------------------------------------------------------------------

def _is_under_any(path: Path, allowed_roots: tuple[str, ...]) -> bool:
    """True if `path` resolves inside one of `allowed_roots`.

    An empty `allowed_roots` tuple is treated as 'nothing allowed' - any
    concrete check fails. Comparison is done with `Path.is_relative_to`
    so case-insensitivity on Windows is handled correctly.
    """
    if not allowed_roots:
        return False
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    for root in allowed_roots:
        try:
            root_path = Path(root).resolve()
            if resolved == root_path or resolved.is_relative_to(root_path):
                return True
        except (OSError, ValueError):
            continue
    return False


def _iter_case_insensitive(base: Path, needle: str, skip_dirs: tuple[str, ...]):
    """Yield files under `base` whose name contains `needle` (case-insensitive).

    Mirrors the legacy tool-layer Strategy 2 behavior. Skipped directories
    (matching any name in `skip_dirs`) are pruned at every level.
    """
    needle_lower = needle.lower()
    skip_set = set(skip_dirs)
    try:
        for p in base.rglob("*"):
            if any(part in skip_set for part in p.parts):
                continue
            if p.is_file() and needle_lower in p.name.lower():
                yield p
    except Exception:
        return


def _iter_substring(base: Path, needle: str, skip_dirs: tuple[str, ...]):
    """Yield files whose full path contains `needle` (case-insensitive)."""
    needle_lower = needle.lower()
    skip_set = set(skip_dirs)
    try:
        for p in base.rglob("*"):
            if any(part in skip_set for part in p.parts):
                continue
            if p.is_file() and needle_lower in str(p).lower():
                yield p
    except Exception:
        return


def _search(base: Path, pattern: str, skip_dirs: tuple[str, ...], limit: int) -> list[str]:
    """3-strategy search: exact glob -> case-insensitive name -> full-path substring.

    Returns up to `limit` unique file paths. Skips directories whose name
    appears in `skip_dirs` at any depth.
    """
    seen: set[str] = set()
    matches: list[str] = []

    def _add(p: Path) -> None:
        if any(part in set(skip_dirs) for part in p.parts):
            return
        if is_backup_artifact(p):
            return
        s = str(p)
        if s in seen:
            return
        seen.add(s)
        matches.append(s)

    # Strategy 1: exact glob (handles wildcards and exact filenames)
    try:
        for p in base.rglob(pattern):
            _add(p)
            if len(matches) >= limit:
                return matches
    except Exception:
        pass

    # Strategy 2: case-insensitive name substring
    try:
        for p in _iter_case_insensitive(base, pattern, skip_dirs):
            _add(p)
            if len(matches) >= limit:
                return matches
    except Exception:
        pass

    # Strategy 3: full-path substring (last resort)
    if len(matches) < limit:
        try:
            for p in _iter_substring(base, pattern, skip_dirs):
                _add(p)
                if len(matches) >= limit:
                    break
        except Exception:
            pass

    return matches


# ---------------------------------------------------------------------------
# Public action functions
# ---------------------------------------------------------------------------

def read_file_action(path: str, *, allowed_roots: tuple[str, ...]) -> str:
    """Read the contents of a UTF-8 text file.

    Returns a formatted string with a FILE / SIZE header and the file body
    wrapped in a fenced block. Output is truncated at 8000 chars. The
    caller is expected to have already passed safety checks; this action
    additionally verifies that the resolved path lives under one of
    `allowed_roots` as a final guardrail.

    Args:
        path: Absolute or already-anchored path to the file. (Relative
              paths are anchored against the current working directory,
              but the resolved location must still fall inside one of
              `allowed_roots`.)
        allowed_roots: Tuple of root directory paths the action is
                       permitted to read from. An empty tuple denies all.

    Returns:
        The formatted file contents, or an `ERROR: ...` string.
    """
    candidate = Path(path)
    try:
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    except Exception as exc:
        return f"ERROR: Could not resolve path '{path}': {exc}"

    if not _is_under_any(candidate, allowed_roots):
        return f"ERROR: Refused to read '{path}' - outside allowed roots."

    if not candidate.exists():
        return f"ERROR: File not found: `{candidate}`"

    try:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: Failed to read `{candidate}`: {exc}"

    total = len(content)
    truncated = total > _MAX_READ_CHARS
    body = content[:_MAX_READ_CHARS] if truncated else content
    header = (
        f"FILE: {candidate}\n"
        f"SIZE: {total} chars\n"
        + (f"NOTE: Truncated at {_MAX_READ_CHARS} chars.\n" if truncated else "")
    )
    return f"{header}\n```\n{body}\n```"


def edit_file_action(
    path: str,
    old_text: str,
    new_text: str,
    *,
    allowed_roots: tuple[str, ...],
    backup_dir: str | None = None,
    diff_history_store: "DiffHistoryStore | None" = None,
) -> str:
    """Replace `old_text` with `new_text` inside the file at `path`.

    The first occurrence of `old_text` is replaced. Chunk 4 of the
    structural-repair plan: no physical ``.bak`` file is written.
    Instead, when ``diff_history_store`` is supplied, the unified diff
    between the previous and new file contents is handed to the store
    so it can be embedded under the hidden ``DIFF_HISTORY_SCOPE``.
    ``backup_dir`` is retained as an accepted keyword for backward
    compatibility with existing callers but is now ignored.

    Args:
        path: Absolute path to the file to edit.
        old_text: Exact substring to find in the existing file.
        new_text: Replacement text.
        allowed_roots: Tuple of permitted root directories.
        backup_dir: Legacy argument, ignored. Retained so existing call
                    sites keep working without modification.
        diff_history_store: Optional ``DiffHistoryStore`` that captures
                    the unified diff into the RAG diff-delta cache.

    Returns:
        Status string describing the result, or `ERROR: ...` on failure.
    """
    # Normalize LLM-provided text first — strips markdown fences, CRLF, etc.
    old_text = _clean_llm_text(old_text)
    new_text = _clean_llm_text(new_text)

    candidate = Path(path)
    try:
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    except Exception as exc:
        return f"ERROR: Could not resolve path '{path}': {exc}"

    if not _is_under_any(candidate, allowed_roots):
        return f"ERROR: Refused to edit '{path}' - outside allowed roots."

    if not old_text:
        return (
            "ERROR: `old_text` is empty. For a full file replacement, call "
            "`overwrite_file_action` instead."
        )

    try:
        current = (
            candidate.read_text(encoding="utf-8", errors="replace")
            if candidate.exists()
            else ""
        )
    except Exception as exc:
        return f"ERROR: Failed to read existing file {candidate}: {exc}"

    idx = current.find(old_text)
    if idx == -1:
        # Provide a diff-friendly hint so the LLM can self-correct on retry.
        head_preview = current[:200].replace('\n', '\\n')
        old_preview = old_text[:200].replace('\n', '\\n')
        return (
            f"ERROR: Could not locate the original text in {candidate}. "
            f"File is {len(current)} chars; first 200 chars: {head_preview!r}. "
            f"Your old_text starts with: {old_preview!r}. "
            f"Re-read the file with read_file to get exact content, then retry."
        )

    new_contents = current[:idx] + new_text + current[idx + len(old_text):]

    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(new_contents, encoding="utf-8")
    except Exception as exc:
        return f"ERROR: Failed to write {candidate}: {exc}"

    # Chunk 4: capture the unified diff in RAG instead of writing a .bak.
    if diff_history_store is not None:
        try:
            diff_history_store.add_diff(str(candidate), current, new_contents)
        except Exception as exc:
            # Diff capture must never undo a successful write; surface a
            # warning but keep the OK result so the user still sees the
            # edit applied.
            print(f"[edit_file_action] diff capture failed: {exc}")

    bytes_written = len(new_contents.encode("utf-8"))
    return f"OK: Edited {candidate} ({bytes_written} bytes written)"


def overwrite_file_action(
    path: str,
    content: str,
    *,
    allowed_roots: tuple[str, ...],
    backup_dir: str | None = None,
    diff_history_store: "DiffHistoryStore | None" = None,
) -> str:
    """Replace the entire contents of `path` with `content`.

    Chunk 4 of the structural-repair plan: no physical ``.bak`` file is
    written. When ``diff_history_store`` is supplied, the unified diff
    between the previous and new contents is embedded into RAG under
    ``DIFF_HISTORY_SCOPE``. ``backup_dir`` is retained for backward
    compatibility but is now ignored.

    Args:
        path: Absolute path to the file to overwrite.
        content: The new file contents.
        allowed_roots: Tuple of permitted root directories.
        backup_dir: Legacy argument, ignored.
        diff_history_store: Optional ``DiffHistoryStore`` that captures
                    the unified diff into the RAG diff-delta cache.

    Returns:
        Status string describing the result, or `ERROR: ...` on failure.
    """
    # Normalize LLM-provided text first — strips markdown fences, CRLF, etc.
    content = _clean_llm_text(content)

    candidate = Path(path)
    try:
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    except Exception as exc:
        return f"ERROR: Could not resolve path '{path}': {exc}"

    if not _is_under_any(candidate, allowed_roots):
        return f"ERROR: Refused to overwrite '{path}' - outside allowed roots."

    # Chunk 4: read existing contents (if any) so we can capture a diff
    # AFTER the write. We do NOT copy the file anywhere -- the diff
    # history store is the single source of recovery.
    previous_contents = ""
    if candidate.exists() and candidate.is_file():
        try:
            previous_contents = candidate.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR: Failed to read existing file {candidate}: {exc}"

    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"ERROR: Failed to write {candidate}: {exc}"

    if diff_history_store is not None:
        try:
            diff_history_store.add_diff(str(candidate), previous_contents, content)
        except Exception as exc:
            print(f"[overwrite_file_action] diff capture failed: {exc}")

    bytes_written = len(content.encode("utf-8"))
    return f"OK: Overwrote {candidate} ({bytes_written} bytes written)"


def search_files_action(
    pattern: str,
    *,
    directory: str,
    skip_dirs: tuple[str, ...],
    max_results: int,
) -> str:
    """Search for files matching `pattern` under `directory`.

    Uses the 3-strategy matcher (exact glob, case-insensitive name
    substring, full-path substring) so patterns like "readme" still
    locate "README.md". Returns a newline-joined list of paths. Returns
    the empty string when no matches are found (the caller decides how
    to format that for the LLM).

    Args:
        pattern: Glob pattern or substring to search for.
        directory: Where to search. Anchored at cwd if relative.
        skip_dirs: Tuple of directory names to prune at every depth
                   (e.g. (".venv", ".git", "__pycache__")).
        max_results: Cap on the number of returned paths.

    Returns:
        Newline-joined list of matching paths, or "" when nothing matched.
    """
    base = Path(directory)
    try:
        if not base.is_absolute():
            base = Path.cwd() / base
    except Exception as exc:
        return f"ERROR: Could not resolve directory '{directory}': {exc}"

    if not base.exists() or not base.is_dir():
        return f"ERROR: Directory not found: `{base}`"

    try:
        matches = _search(base, pattern, skip_dirs, max_results)
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"

    return "\n".join(matches)
