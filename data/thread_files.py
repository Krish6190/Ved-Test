"""Per-thread chunk-count tracking + LRU eviction.

Layout on disk (metadata only — no file bytes are persisted):
    data/threads/<thread_id>/uploads.json
        [{filename, uploaded_at, chunk_count}, ...]

The actual chunks live in the shared vector DB (graph.rag.rag_db) with scope=thread_id.
Raw file bytes are read once at upload time, embedded, then discarded.

When a thread's total chunk count exceeds THREAD_CHUNK_QUOTA, the oldest uploads
(by uploaded_at ASC) are evicted: their chunks are deleted from the vector DB and
their entries removed from uploads.json.

Why this is the right shape:
  - Embeddings are already in the vector DB pickle. No duplicate storage.
  - SSD usage stays minimal: just one tiny JSON per thread.
  - Eviction is O(1) per chunk (filter + reindex in numpy).
  - RAM is bounded by THREAD_CHUNK_QUOTA * embedding_dim per thread.
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict

# Per-thread chunk quota. Tunable via env or this constant.
# 6800 chunks * ~768-dim float32 * 4 bytes ≈ 20 MB embeddings per thread.
# At ~500 chars/chunk, ~3.4 MB of raw text per thread.
THREAD_CHUNK_QUOTA = int(os.getenv("VED_THREAD_CHUNK_QUOTA", "6800"))

DATA_ROOT = Path(__file__).resolve().parent
THREADS_ROOT = DATA_ROOT / "threads"


class ThreadFileStore:
    """Per-thread chunk-count tracker. Enforces quota via LRU eviction from the vector DB."""

    def __init__(self, rag_db):
        self.rag_db = rag_db
        THREADS_ROOT.mkdir(parents=True, exist_ok=True)

    # ---------- paths / metadata ----------
    def _thread_dir(self, thread_id: str) -> Path:
        return THREADS_ROOT / thread_id

    def _meta_path(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "uploads.json"

    def _load_meta(self, thread_id: str) -> List[Dict]:
        path = self._meta_path(thread_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_meta(self, thread_id: str, entries: List[Dict]) -> None:
        path = self._meta_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- public API ----------
    def total_chunks(self, thread_id: str) -> int:
        return sum(int(e.get("chunk_count", 0)) for e in self._load_meta(thread_id))

    def list_uploads(self, thread_id: str) -> List[Dict]:
        """Return upload metadata sorted by uploaded_at ASC (oldest first)."""
        return sorted(self._load_meta(thread_id), key=lambda e: e.get("uploaded_at", 0))

    def add(self, thread_id: str, source_path: str, filename: str | None = None, chunker: str = "text") -> Dict:
        """Read file, embed chunks into vector DB (scope=thread_id), track chunk count.

        No file bytes are persisted. Raises FileNotFoundError if source_path missing;
        bubbles embedding errors after attempting to roll back the registry addition
        (caller may also catch and surface).

        `chunker` selects the chunking strategy: "text" (default) for the
        document-parser pipeline, "ast" for graph.rag.code_chunker.
        """
        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        filename = filename or os.path.basename(source_path)
        chunks_before = self._count_chunks_in_registry(thread_id)

        try:
            self.rag_db.ingest_local_file(source_path, scope=thread_id, chunker=chunker, source=filename)
        except Exception:
            # Try to roll back any partial registry additions.
            chunks_after_fail = self._count_chunks_in_registry(thread_id)
            if chunks_after_fail > chunks_before and hasattr(self.rag_db, "delete_by_source"):
                try:
                    self.rag_db.delete_by_source(scope=thread_id, source=filename)
                except Exception:
                    pass
            raise

        chunks_after = self._count_chunks_in_registry(thread_id)
        new_chunk_count = max(0, chunks_after - chunks_before)

        entry = {
            "filename": filename,
            "uploaded_at": time.time(),
            "chunk_count": new_chunk_count,
        }
        meta = self._load_meta(thread_id)
        meta.append(entry)
        self._save_meta(thread_id, meta)

        evicted = self._enforce_quota(thread_id)
        entry["evicted"] = [e["filename"] for e in evicted]
        return entry

    def add_text(self, thread_id: str, text: str, source_label: str = "", chunker: str = "text") -> Dict:
        """Embed raw text into the vector DB (scope=thread_id), track chunk count, enforce quota.

        No file bytes are persisted. source_label is used to attribute the chunks so
        future evictions can drop them via delete_by_source. Designed for large
        paste-from-clipboard flows: the pasted text goes here, and a small
        placeholder (with the extracted question) goes into state.messages instead.

        Returns the new upload metadata entry (with "evicted" list of filenames
        dropped to make room).
        """
        if not text or not text.strip():
            raise ValueError("text is empty")
        if not source_label or not source_label.strip():
            raise ValueError("source_label is empty")

        chunks_before = self._count_chunks_in_registry(thread_id)

        try:
            if hasattr(self.rag_db, "ingest_text"):
                self.rag_db.ingest_text(text, scope=thread_id, source=source_label, chunker=chunker)
            else:
                # Legacy fallback: chunk + embed manually using the rag_network helper.
                from graph.rag.rag_network import fetch_ollama_vector
                chunks = self.rag_db.file_parser.text_splitter.split_raw_text(text)
                existing_entries = {r["content"] for r in self.rag_db.registry}
                for chunk in chunks:
                    clean = chunk.strip()
                    if not clean or clean in existing_entries:
                        continue
                    vec = fetch_ollama_vector(clean)
                    if vec:
                        self.rag_db.registry.append({
                            "content": clean,
                            "source": source_label,
                            "scope": thread_id,
                            "embedding": vec,
                        })
                self.rag_db._save_database()
        except Exception:
            # Rollback partial additions
            if hasattr(self.rag_db, "delete_by_source"):
                try:
                    self.rag_db.delete_by_source(scope=thread_id, source=source_label)
                except Exception:
                    pass
            raise

        chunks_after = self._count_chunks_in_registry(thread_id)
        new_chunk_count = max(0, chunks_after - chunks_before)

        entry = {
            "filename": source_label,
            "uploaded_at": time.time(),
            "chunk_count": new_chunk_count,
        }
        meta = self._load_meta(thread_id)
        meta.append(entry)
        self._save_meta(thread_id, meta)

        evicted = self._enforce_quota(thread_id)
        entry["evicted"] = [e["filename"] for e in evicted]
        return entry

    def _count_chunks_in_registry(self, thread_id: str) -> int:
        return sum(1 for r in self.rag_db.registry if r.get("scope") == thread_id)

    def _enforce_quota(self, thread_id: str) -> List[Dict]:
        """Evict oldest uploads (FIFO by uploaded_at) until total chunks <= quota.

        Returns list of evicted metadata entries (caller may surface to the user).
        """
        meta = self.list_uploads(thread_id)  # already ASC
        evicted: List[Dict] = []
        total = sum(int(e.get("chunk_count", 0)) for e in meta)

        while total > THREAD_CHUNK_QUOTA and meta:
            oldest = meta.pop(0)
            self._delete_upload(thread_id, oldest)
            evicted.append(oldest)
            total = sum(int(e.get("chunk_count", 0)) for e in meta)

        if evicted:
            self._save_meta(thread_id, meta)
        return evicted

    def _delete_upload(self, thread_id: str, entry: Dict) -> None:
        basename = entry.get("filename")
        if basename and hasattr(self.rag_db, "delete_by_source"):
            try:
                self.rag_db.delete_by_source(scope=thread_id, source=basename)
            except Exception:
                pass

    def clear_thread(self, thread_id: str) -> None:
        """Delete all chunks + metadata for a thread. Called when the thread is deleted."""
        for entry in self._load_meta(thread_id):
            self._delete_upload(thread_id, entry)
        thread_dir = self._thread_dir(thread_id)
        if thread_dir.exists():
            shutil.rmtree(thread_dir, ignore_errors=True)
