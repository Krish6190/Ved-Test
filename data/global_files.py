"""Global file storage — separate quota from per-thread storage.

Used by the /upload-global slash command. Accessible only via that command,
so only the local user can populate it (the input bar is the only command surface).

Layout on disk (metadata only — no file bytes are persisted):
    data/global_files.json
        [{filename, uploaded_at, chunk_count}, ...]

The actual chunks live in the shared vector DB (graph.rag.rag_db) with scope="__GLOBAL__".

Quota is independent of per-thread quota. When exceeded, oldest uploads are
evicted from the vector DB via delete_by_source.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict

# Global store quota. Larger than per-thread (since global is shared across all threads).
# 13600 chunks * ~3 KB per embedding ≈ 40 MB embeddings.
GLOBAL_FILE_QUOTA = int(os.getenv("VED_GLOBAL_CHUNK_QUOTA", "13600"))

DATA_ROOT = Path(__file__).resolve().parent


class GlobalFileStore:
    """Global-scope file storage. Tracks chunk counts; enforces quota via LRU eviction."""

    def __init__(self, rag_db):
        self.rag_db = rag_db
        self._meta_path = DATA_ROOT / "global_files.json"

    def _load_meta(self) -> List[Dict]:
        if not self._meta_path.exists():
            return []
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_meta(self, entries: List[Dict]) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def total_chunks(self) -> int:
        return sum(int(e.get("chunk_count", 0)) for e in self._load_meta())

    def list_uploads(self) -> List[Dict]:
        return sorted(self._load_meta(), key=lambda e: e.get("uploaded_at", 0))

    def add(self, source_path: str) -> Dict:
        """Read file, embed chunks into vector DB (scope=__GLOBAL__), track + enforce quota.

        No file bytes are persisted. Raises FileNotFoundError if source missing.
        """
        from graph.rag.mixer import GLOBAL_SCOPE

        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        filename = os.path.basename(source_path)
        chunks_before = self._count_chunks()

        try:
            self.rag_db.ingest_local_file(source_path, scope=GLOBAL_SCOPE)
        except Exception:
            chunks_after_fail = self._count_chunks()
            if chunks_after_fail > chunks_before and hasattr(self.rag_db, "delete_by_source"):
                try:
                    self.rag_db.delete_by_source(scope=GLOBAL_SCOPE, source=filename)
                except Exception:
                    pass
            raise

        chunks_after = self._count_chunks()
        new_chunk_count = max(0, chunks_after - chunks_before)

        entry = {
            "filename": filename,
            "uploaded_at": time.time(),
            "chunk_count": new_chunk_count,
        }
        meta = self._load_meta()
        meta.append(entry)
        self._save_meta(meta)

        evicted = self._enforce_quota()
        entry["evicted"] = [e["filename"] for e in evicted]
        return entry

    def _count_chunks(self) -> int:
        from graph.rag.mixer import GLOBAL_SCOPE
        return sum(1 for r in self.rag_db.registry if r.get("scope") == GLOBAL_SCOPE)

    def _enforce_quota(self) -> List[Dict]:
        meta = self.list_uploads()
        evicted: List[Dict] = []
        total = sum(int(e.get("chunk_count", 0)) for e in meta)

        while total > GLOBAL_FILE_QUOTA and meta:
            oldest = meta.pop(0)
            self._delete_upload(oldest)
            evicted.append(oldest)
            total = sum(int(e.get("chunk_count", 0)) for e in meta)

        if evicted:
            self._save_meta(meta)
        return evicted

    def _delete_upload(self, entry: Dict) -> None:
        from graph.rag.mixer import GLOBAL_SCOPE
        basename = entry.get("filename")
        if basename and hasattr(self.rag_db, "delete_by_source"):
            try:
                self.rag_db.delete_by_source(scope=GLOBAL_SCOPE, source=basename)
            except Exception:
                pass

    def remove_upload(self, filename: str) -> bool:
        meta = self._load_meta()
        for entry in meta:
            if entry.get("filename") == filename:
                meta.remove(entry)
                self._delete_upload(entry)
                self._save_meta(meta)
                return True
        return False

    def clear(self) -> None:
        """Delete all global chunks + metadata."""
        for entry in self._load_meta():
            self._delete_upload(entry)
        if self._meta_path.exists():
            try:
                self._meta_path.unlink()
            except OSError:
                pass
