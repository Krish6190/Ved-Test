"""Project indexer — walk cwd respecting .gitignore, ingest into RAG.

Called by chatbot.on_session_start() on the first chat in a session.
Runs in a background thread to avoid blocking the UI.

Per-file SHA-256 hashes are stored in data/rag_index.json so unchanged
files are skipped on re-run. Binary files, lockfiles, .venv/,
__pycache__/, .git/, node_modules/ are excluded by default.

Uses LocalVectorDB.ingest_local_file for chunk storage (shared with
thread-scoped uploads). The 2-layer AST chunking for code files is
applied separately in Phase 2.2 (graph.rag.code_chunker).
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Default ignore patterns (in addition to .gitignore).
# Uses gitignore-style wildcards: * matches anything except /, ** matches
# anything including /, trailing / means directory-only.
_DEFAULT_IGNORES = (
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "node_modules/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "*.pyc", "*.pyo", "*.pyd",
    "*.so", "*.dll", "*.dylib",
    "*.egg-info/",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.svg",
    "*.mp3", "*.mp4", "*.wav", "*.ogg", "*.webm",
    "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.rar", "*.7z",
    "*.pdf", "*.docx", "*.doc",
)

# File extensions to consider for indexing.
_CODE_EXTENSIONS = frozenset({
    ".py", ".pyx", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
    ".go", ".rs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".html", ".css", ".scss", ".sass", ".less",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg",
    ".md", ".markdown", ".rst", ".txt",
    ".sql", ".graphql", ".proto",
    ".lua", ".vim", ".el",
})

# Max file size to index (bytes) — skip huge files.
_MAX_FILE_SIZE = 1024 * 1024  # 1 MiB

# Scope name used for project-indexed chunks in LocalVectorDB.
_PROJECT_SCOPE = "project"


def _parse_gitignore(gitignore_path: Path) -> List[str]:
    """Read a .gitignore file, return non-comment, non-blank patterns."""
    if not gitignore_path.exists():
        return []
    patterns: List[str] = []
    try:
        text = gitignore_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            patterns.append(stripped)
    except Exception:
        pass
    return patterns


def _glob_to_regex(pattern: str) -> str:
    """Convert a gitignore glob to a regex string.

    Supports: * (any non-/ chars), ** (any chars including /), ? (single char),
    [abc] (char class — passed through). Escapes literal dots.
    """
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                # ** matches anything including /
                out.append(".*")
                i += 2
                # Skip following / if present (e.g., **/)
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            else:
                # * matches anything except /
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c in r".^$+()|{}[]\\":
            out.append(re.escape(c))
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _matches_pattern(rel_path: str, pattern: str) -> bool:
    """Check if a relative POSIX path matches a gitignore-style pattern.

    Handles leading / (root-anchored), trailing / (directory marker — also
    matches the directory's contents), and bare basenames.
    """
    is_dir_only = pattern.endswith("/")
    if is_dir_only:
        pattern = pattern[:-1]

    is_anchored = pattern.startswith("/")
    if is_anchored:
        pattern = pattern[1:]

    regex = _glob_to_regex(pattern)

    # Root-anchored: match against full relative path.
    if is_anchored:
        return bool(re.match(f"^{regex}$", rel_path))

    # Non-anchored: match against full path OR any path component OR basename.
    if re.match(f"^{regex}$", rel_path):
        return True
    # Match against each path component (handles patterns like "*.pyc")
    parts = rel_path.split("/")
    for part in parts:
        if re.match(f"^{regex}$", part):
            return True
    # Match against full path with ** semantics (anywhere)
    if "**" in pattern:
        if re.search(regex, rel_path):
            return True

    return False


def _should_ignore(rel_path: str, gitignore_patterns: List[str]) -> bool:
    """Check if a relative path is ignored by default patterns or .gitignore."""
    # Normalize: gitignore uses forward slashes
    rel = rel_path.replace("\\", "/")
    # Ensure leading ./
    if not rel.startswith("./"):
        rel_for_check = "./" + rel
    else:
        rel_for_check = rel

    for pat in _DEFAULT_IGNORES:
        if _matches_pattern(rel_for_check, pat) or _matches_pattern(rel, pat):
            return True
    for pat in gitignore_patterns:
        if _matches_pattern(rel_for_check, pat) or _matches_pattern(rel, pat):
            return True
    return False


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents. Returns '' on error."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _load_index(index_path: Path) -> Dict[str, str]:
    """Load the {rel_path: sha256} index from disk."""
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_index(index_path: Path, index: Dict[str, str]) -> None:
    """Persist the {rel_path: sha256} index to disk."""
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[project_indexer] Failed to save index: {e}")


def index_workspace(
    root: str | Path,
    db: Any,
    force: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, int]:
    """Walk `root`, respect .gitignore, and index changed files into the RAG store.

    Args:
        root: Project root directory to walk.
        db: LocalVectorDB instance to receive chunks (calls db.ingest_local_file).
        force: If True, re-index all files regardless of stored hash.
        progress_callback: Optional callable(stage: str, current: int, total: int)
            invoked periodically for UI progress reporting.

    Returns:
        Dict with counts: files_scanned, files_indexed, files_skipped,
        chunks_added (best-effort), errors.
    """
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        return {
            "files_scanned": 0, "files_indexed": 0, "files_skipped": 0,
            "chunks_added": 0, "errors": 1,
        }

    # Load .gitignore patterns from root only (not parent dirs — keeps scope local).
    gitignore_patterns = _parse_gitignore(root / ".gitignore")

    # Load existing hash index.
    index_path = root / "data" / "rag_index.json"
    index: Dict[str, str] = {} if force else _load_index(index_path)

    stats = {
        "files_scanned": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "chunks_added": 0,
        "errors": 0,
    }

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place to prevent descent.
        kept: List[str] = []
        for d in dirnames:
            rel = os.path.relpath(os.path.join(dirpath, d), root).replace("\\", "/")
            if _should_ignore(rel + "/", gitignore_patterns):
                continue
            kept.append(d)
        dirnames[:] = kept

        for fn in filenames:
            stats["files_scanned"] += 1
            full_path = Path(dirpath) / fn

            try:
                rel_path = os.path.relpath(full_path, root).replace("\\", "/")
            except Exception:
                stats["files_skipped"] += 1
                continue

            # Ignore patterns
            if _should_ignore(rel_path, gitignore_patterns):
                stats["files_skipped"] += 1
                continue

            # Extension filter
            ext = full_path.suffix.lower()
            if ext not in _CODE_EXTENSIONS:
                stats["files_skipped"] += 1
                continue

            # Size filter
            try:
                size = full_path.stat().st_size
                if size == 0 or size > _MAX_FILE_SIZE:
                    stats["files_skipped"] += 1
                    continue
            except Exception:
                stats["files_skipped"] += 1
                continue

            # Hash check (skip if unchanged)
            file_hash = _sha256_file(full_path)
            if not file_hash:
                stats["files_skipped"] += 1
                continue
            if not force and index.get(rel_path) == file_hash:
                stats["files_skipped"] += 1
                continue
            try:
                committed = db.ingest_local_file(
                    str(full_path),
                    scope=_PROJECT_SCOPE,
                    chunker="ast",
                    source=os.path.basename(full_path),
                )
            except Exception as e:
                print(f"[project_indexer] Failed to index {rel_path}: {e}")
                stats["errors"] += 1
                committed = False
            if committed:
                index[rel_path] = file_hash
                stats["files_indexed"] += 1
            else:
                # No chunks committed -- do NOT add to the hash index so
                # the next session retries. Count as a transient error
                # so the stats surface the issue.
                print(f"[project_indexer] No chunks committed for {rel_path}; will retry next session")
                stats["errors"] += 1

            if progress_callback is not None:
                try:
                    progress_callback("indexing", stats["files_scanned"], 0)
                except Exception:
                    pass

    _save_index(index_path, index)
    return stats


__all__ = ["index_workspace", "_PROJECT_SCOPE"]
