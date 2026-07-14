"""Per-file RAG diff-delta cache with FIFO eviction.

When Ved edits a file, the action layer no longer writes a physical
``.bak`` backup. Instead it computes a unified diff and hands it to
``DiffHistoryStore``, which embeds the diff under a hidden RAG scope
(``DIFF_HISTORY_SCOPE``) with a per-path FIFO cap of
``DIFF_CYCLE_LIMIT``. This keeps the most recent five edit cycles for
each file available to historic-review prompts while bounding disk and
RAG growth.

Metadata for the cycle ledger lives in ``data/diff_history.json`` and is
persisted atomically via a ``.tmp`` + ``rename`` (same pattern as
``data/plans.py``). Cycle labels embed the resolved file path plus a
monotonic per-path counter so the label is unique even after evictions.

Public surface:
    DIFF_HISTORY_SCOPE  -- RAG scope tag used for diff chunks.
    DIFF_CYCLE_LIMIT    -- Max cycles retained per file path.
    DiffHistoryStore    -- Class with ``add_diff`` and ``query_history``.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Hidden RAG scope used for diff-delta chunks. Kept distinct from
# per-thread scopes and GLOBAL_SCOPE so generic queries never surface
# historic diffs unless the caller explicitly asks for them.
DIFF_HISTORY_SCOPE = "__DIFF_HISTORY__"

# Maximum number of diff cycles retained per file path. When a path
# already has this many cycles stored, the next add evicts the oldest
# label before ingesting the new one.
DIFF_CYCLE_LIMIT = 5


class DiffHistoryStore:
    """Embed per-file unified diffs under a hidden RAG scope.

    The store is intentionally lightweight: it does not own the RAG
    database, it merely translates filesystem edits into ``ingest_text``
    / ``delete_by_source`` calls and tracks cycle labels on disk so the
    cache can survive process restarts.
    """

    def __init__(
        self,
        rag_db: Any,
        meta_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.rag_db = rag_db
        if meta_path is None:
            meta_path = Path("data") / "diff_history.json"
        self.meta_path = Path(meta_path)
        self._meta: Dict[str, Dict[str, Any]] = self._load_meta()

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------
    def _load_meta(self) -> Dict[str, Dict[str, Any]]:
        """Load the cycle ledger from disk. Returns an empty dict on miss/error."""
        if not self.meta_path.exists():
            return {}
        try:
            raw = self.meta_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        # Defensive: drop any malformed entries.
        cleaned: Dict[str, Dict[str, Any]] = {}
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            counter = val.get("counter", 0)
            cycles = val.get("cycles", [])
            if not isinstance(cycles, list):
                cycles = []
            try:
                counter = int(counter)
            except (TypeError, ValueError):
                counter = 0
            cleaned[key] = {"counter": counter, "cycles": [str(c) for c in cycles]}
        return cleaned

    def _save_meta(self) -> None:
        """Persist the cycle ledger atomically via a .tmp + rename."""
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.meta_path)
        except Exception as e:
            # Failing to persist metadata should not crash the calling
            # action; the RAG ingest has already happened. Surface a
            # warning so operators can investigate if needed.
            print(f"[DiffHistoryStore] Failed to save metadata: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(file_path: Union[str, Path]) -> str:
        """Return a stable absolute string key for the file path."""
        return str(Path(file_path).resolve())

    @staticmethod
    def _compute_diff_text(old_text: str, new_text: str) -> str:
        """Compute a unified diff and normalise whitespace.

        ``lineterm=""`` prevents difflib from injecting extra trailing
        newlines on each hunk header. We additionally strip trailing
        whitespace from every line so the embedded text stays clean.
        """
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                lineterm="",
            )
        )
        cleaned = [line.rstrip() for line in diff_lines]
        return "\n".join(cleaned)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_diff(self, file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Compute and store a unified diff for ``file_path``.

        Returns:
            ``{"stored": bool, "cycle_label": str | None, "evicted": list[str]}``.

            * ``stored`` is False when the diff is empty (no changes).
            * ``cycle_label`` is the new label under which the diff was
              ingested, or None when nothing was stored.
            * ``evicted`` lists any cycle labels removed from RAG to
              honour the FIFO cap (in eviction order).
        """
        diff_text = self._compute_diff_text(old_text, new_text)
        if not diff_text.strip():
            return {"stored": False, "cycle_label": None, "evicted": []}

        resolved = self._resolve_path(file_path)
        entry = self._meta.setdefault(resolved, {"counter": 0, "cycles": []})

        evicted: List[str] = []
        # FIFO eviction: if we're at (or somehow above) the cap, drop the
        # oldest labels until there is room for the new one.
        while len(entry["cycles"]) >= DIFF_CYCLE_LIMIT:
            old_label = entry["cycles"].pop(0)
            try:
                self.rag_db.delete_by_source(
                    scope=DIFF_HISTORY_SCOPE, source=old_label
                )
            except Exception as e:
                print(f"[DiffHistoryStore] Failed to evict {old_label}: {e}")
            evicted.append(old_label)

        entry["counter"] += 1
        new_label = f"{resolved}::diff::{entry['counter']}"
        entry["cycles"].append(new_label)

        try:
            self.rag_db.ingest_text(
                diff_text, scope=DIFF_HISTORY_SCOPE, source=new_label
            )
        except Exception as e:
            # Roll the bookkeeping back so a failed ingest does not leave
            # a dangling label in the registry.
            entry["cycles"].pop()
            print(f"[DiffHistoryStore] ingest_text failed for {new_label}: {e}")
            return {"stored": False, "cycle_label": None, "evicted": evicted}

        self._save_meta()
        return {"stored": True, "cycle_label": new_label, "evicted": evicted}

    def query_history(self, file_path: str, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Return RAG chunks for ``file_path``'s diff history only.

        Calls ``rag_db.query_similarity`` scoped to ``DIFF_HISTORY_SCOPE``
        and filters the returned records down to those whose ``source``
        starts with the resolved ``file_path`` prefix. This keeps
        diffs of other files out of historic-review results.
        """
        resolved = self._resolve_path(file_path)
        try:
            results = self.rag_db.query_similarity(
                query, k=k, scope=DIFF_HISTORY_SCOPE
            )
        except Exception as e:
            print(f"[DiffHistoryStore] query_similarity failed: {e}")
            return []
        if not results:
            return []
        prefix = resolved
        return [r for r in results if str(r.get("source", "")).startswith(prefix)]


__all__ = ["DIFF_HISTORY_SCOPE", "DIFF_CYCLE_LIMIT", "DiffHistoryStore"]
