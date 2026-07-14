"""Tests for the per-file RAG diff-delta cache (graph.rag.diff_history).

These tests cover four behaviours:
  1. add_diff computes a unified diff, stores it under the hidden RAG scope,
     and returns a stable cycle label.
  2. FIFO eviction: once a path has DIFF_CYCLE_LIMIT (5) cycles stored,
     the next add evicts the oldest label and removes it from the registry.
  3. Every diff chunk is tagged with scope == DIFF_HISTORY_SCOPE.
  4. query_history only returns chunks whose source belongs to the resolved
     file path.

The tests use a FakeVectorDB that mimics the LocalVectorDB surface used
by the DiffHistoryStore (ingest_text / query_similarity / delete_by_source)
without needing a real Ollama embedder. The metadata file is written to
a tmp_path so the real `data/diff_history.json` is never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Ensure the project root is importable for the `graph.*` imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------
class FakeVectorDB:
    """Minimal in-memory replacement for LocalVectorDB.

    Mirrors the surface DiffHistoryStore relies on:
      - ingest_text(text, scope, source, chunker=...)
      - query_similarity(query_text, k=..., scope=...)
      - delete_by_source(scope, source) -> int

    Embedding is stubbed with a zero vector so no Ollama calls are made.
    Records are stored in a flat list keyed by (scope, source).
    """

    DIM = 8

    def __init__(self) -> None:
        self.registry: List[Dict[str, Any]] = []

    @staticmethod
    def _vec() -> List[float]:
        return [0.0] * FakeVectorDB.DIM

    # ---- API used by DiffHistoryStore ----
    def ingest_text(
        self,
        text: str,
        scope: str = "__GLOBAL__",
        source: str = "raw_text",
        chunker: str = "text",
    ) -> None:
        if not text or not text.strip():
            return
        # Each ingest_text may produce multiple sub-chunks (mirrors splitter
        # behaviour) — keep one record per chunk for the test surface.
        self.registry.append(
            {
                "content": text,
                "source": source,
                "scope": scope,
                "chunker": chunker,
                "embedding": self._vec(),
            }
        )

    def query_similarity(
        self,
        query_text: str,
        k: int = 2,
        lambda_mult: float = 0.5,
        scope: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.registry:
            return []
        if scope is None:
            return list(self.registry[:k])
        filtered = [r for r in self.registry if r.get("scope") == scope]
        return list(filtered[:k])

    def delete_by_source(self, scope: str, source: str) -> int:
        before = len(self.registry)
        self.registry = [
            r
            for r in self.registry
            if not (r.get("scope") == scope and r.get("source") == source)
        ]
        return before - len(self.registry)

    # ---- helpers used by tests ----
    def records_with_source(self, source: str) -> List[Dict[str, Any]]:
        return [r for r in self.registry if r.get("source") == source]


@pytest.fixture
def fake_rag() -> FakeVectorDB:
    return FakeVectorDB()


@pytest.fixture
def meta_path(tmp_path: Path) -> Path:
    return tmp_path / "diff_history.json"


@pytest.fixture
def store(fake_rag: FakeVectorDB, meta_path: Path):
    # Import lazily so the fixture name resolves even before the module exists.
    from graph.rag.diff_history import DiffHistoryStore

    return DiffHistoryStore(rag_db=fake_rag, meta_path=meta_path)


# ---------------------------------------------------------------------------
# Test 1: add_diff returns a cycle label and stores the unified diff
# ---------------------------------------------------------------------------
def test_add_diff_stores_unified_diff(store, fake_rag, meta_path, tmp_path):
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a = 1\n", encoding="utf-8")

    old_text = "a = 1\nb = 2\n"
    new_text = "a = 1\nb = 3\nc = 4\n"

    result = store.add_diff(str(target), old_text, new_text)

    # Result shape
    assert isinstance(result, dict)
    assert result.get("stored") is True
    assert result.get("cycle_label")
    assert result.get("evicted") == []

    # Metadata persisted
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = str(target.resolve())
    assert resolved in meta
    assert meta[resolved]["counter"] == 1
    assert meta[resolved]["cycles"] == [result["cycle_label"]]

    # At least one chunk was written to RAG under the cycle label.
    cycle_label = result["cycle_label"]
    records = fake_rag.records_with_source(cycle_label)
    assert len(records) >= 1
    body = "\n".join(r["content"] for r in records)
    # Unified-diff markers must be present.
    assert "---" in body
    assert "+++" in body
    assert "-b = 2" in body
    assert "+b = 3" in body


# ---------------------------------------------------------------------------
# Test 2: FIFO eviction kicks in once DIFF_CYCLE_LIMIT is exceeded
# ---------------------------------------------------------------------------
def test_diff_history_fifo_evicts_oldest(store, fake_rag, meta_path, tmp_path):
    from graph.rag.diff_history import DIFF_CYCLE_LIMIT

    target = tmp_path / "evict.py"
    target.write_text("placeholder\n", encoding="utf-8")

    cycle_labels: List[str] = []
    for i in range(DIFF_CYCLE_LIMIT):
        old = f"line_old_{i}\n"
        new = f"line_new_{i}\n"
        out = store.add_diff(str(target), old, new)
        assert out["stored"] is True
        assert out["evicted"] == []
        cycle_labels.append(out["cycle_label"])

    # All five labels should currently be present in the RAG registry.
    for label in cycle_labels:
        assert fake_rag.records_with_source(label), f"{label} missing before overflow"

    # The 6th add should evict the oldest (first) label.
    out6 = store.add_diff(str(target), "line_old_5\n", "line_new_5\n")
    assert out6["stored"] is True
    assert out6["evicted"] == [cycle_labels[0]]

    # Oldest is gone from RAG registry.
    assert fake_rag.records_with_source(cycle_labels[0]) == []

    # Newest label exists.
    new_label = out6["cycle_label"]
    assert fake_rag.records_with_source(new_label)

    # Cycles list trimmed back to DIFF_CYCLE_LIMIT entries.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = str(target.resolve())
    assert len(meta[resolved]["cycles"]) == DIFF_CYCLE_LIMIT
    assert meta[resolved]["cycles"][0] == cycle_labels[1]
    assert meta[resolved]["cycles"][-1] == new_label


# ---------------------------------------------------------------------------
# Test 3: all diff chunks carry the hidden DIFF_HISTORY_SCOPE
# ---------------------------------------------------------------------------
def test_diff_chunks_use_hidden_scope(store, fake_rag, tmp_path):
    from graph.rag.diff_history import DIFF_HISTORY_SCOPE

    target = tmp_path / "scoped.py"
    target.write_text("placeholder\n", encoding="utf-8")

    for i in range(3):
        store.add_diff(str(target), f"old_{i}\n", f"new_{i}\n")

    assert fake_rag.registry, "Expected diff chunks to be stored"
    for record in fake_rag.registry:
        assert record["scope"] == DIFF_HISTORY_SCOPE
        # No chunk should leak under any other scope (e.g., GLOBAL).
        assert record["scope"] != "__GLOBAL__"


# ---------------------------------------------------------------------------
# Test 4: query_history only returns chunks belonging to the requested path
# ---------------------------------------------------------------------------
def test_query_history_filters_by_path(store, fake_rag, tmp_path):
    target_a = tmp_path / "a.py"
    target_b = tmp_path / "b.py"
    target_a.write_text("placeholder\n", encoding="utf-8")
    target_b.write_text("placeholder\n", encoding="utf-8")

    store.add_diff(str(target_a), "alpha_old\n", "alpha_new\n")
    store.add_diff(str(target_b), "beta_old\n", "beta_new\n")
    store.add_diff(str(target_a), "alpha_old_2\n", "alpha_new_2\n")

    # Querying for a returns only a's chunks.
    results_a = store.query_history(str(target_a), "anything", k=10)
    sources_a = {r["source"] for r in results_a}
    assert sources_a, "Expected results for path A"
    for src in sources_a:
        assert src.startswith(str(target_a.resolve()))

    # Querying for b returns only b's chunks and excludes a's.
    results_b = store.query_history(str(target_b), "anything", k=10)
    sources_b = {r["source"] for r in results_b}
    assert sources_b, "Expected results for path B"
    for src in sources_b:
        assert src.startswith(str(target_b.resolve()))
    assert sources_a.isdisjoint(sources_b)


# ---------------------------------------------------------------------------
# Test 5 (bonus): empty diffs are not stored and report stored=False
# ---------------------------------------------------------------------------
def test_add_diff_empty_change_is_not_stored(store, fake_rag, meta_path, tmp_path):
    target = tmp_path / "noop.py"
    target.write_text("placeholder\n", encoding="utf-8")

    out = store.add_diff(str(target), "same\n", "same\n")
    assert out["stored"] is False
    assert out["cycle_label"] is None
    assert out["evicted"] == []
    assert fake_rag.registry == []

    # Metadata file should NOT be created when nothing changed.
    assert not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) == {}
