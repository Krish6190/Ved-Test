"""Tests for the retrieve_rag tool.

The tool fetches chunks from the thread's RAG store on demand. We mock
the underlying rag_db / mixer so no real embeddings are needed.

Note: imports bypass `graph.tools` package __init__ to avoid pulling in
file_editor (which imports tkinter) — the test environment may not have
tkinter available (e.g., WSL without Tk bindings).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow direct module import without triggering graph.tools.__init__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import directly, not via `graph.tools` package.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rag_retrieve_under_test",
    Path(__file__).resolve().parent.parent / "graph" / "tools" / "rag_retrieve.py",
)
rag_retrieve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_retrieve)
retrieve_rag = rag_retrieve.retrieve_rag


def _invoke_retrieve(query, cfg=None, **kwargs):
    """retrieve_rag.invoke requires config as a separate arg, not in the input dict."""
    return retrieve_rag.invoke({"query": query, **kwargs}, config=cfg)


def test_retrieve_rag_empty_query():
    result = _invoke_retrieve("")
    assert "ERROR" in result
    assert "non-empty" in result.lower()


def test_retrieve_rag_uses_active_thread_id_from_config():
    """When no scope is passed, fall back to active_thread_id from config."""
    cfg = {"configurable": {"active_thread_id": "thr_active"}}
    captured_kwargs = {}

    def fake_retrieve_context(query, scope, k):
        captured_kwargs["query"] = query
        captured_kwargs["scope"] = scope
        captured_kwargs["k"] = k
        return []

    with patch.object(rag_retrieve, "retrieve_context", fake_retrieve_context), \
         patch.object(rag_retrieve, "_format_rag_block", return_value=""):
        result = _invoke_retrieve("test query", cfg=cfg)
    assert captured_kwargs["scope"] == "thr_active"
    assert captured_kwargs["query"] == "test query"


def test_retrieve_rag_explicit_scope_overrides_config():
    """If scope is passed explicitly, it wins over active_thread_id."""
    captured_scope = {}

    def fake_retrieve_context(query, scope, k):
        captured_scope["scope"] = scope
        return []

    with patch.object(rag_retrieve, "retrieve_context", fake_retrieve_context), \
         patch.object(rag_retrieve, "_format_rag_block", return_value=""):
        _invoke_retrieve("test", cfg={"configurable": {"active_thread_id": "thr_a"}}, scope="thr_b")
    assert captured_scope["scope"] == "thr_b"


def test_retrieve_rag_returns_no_match_message_when_empty():
    cfg = {"configurable": {"active_thread_id": "thr_x"}}
    with patch.object(rag_retrieve, "retrieve_context", return_value=[]), \
         patch.object(rag_retrieve, "_format_rag_block", return_value=""):
        result = _invoke_retrieve("nothing", cfg=cfg, k=5)
    assert "No RAG chunks found" in result
    assert "nothing" in result
    assert "thr_x" in result


def test_retrieve_rag_returns_formatted_chunks():
    """When chunks are returned, they're formatted and included in the response."""
    cfg = {"configurable": {"active_thread_id": "thr_x"}}
    chunks = [
        {"source": "ai_response_abc", "content": "first chunk content here"},
        {"source": "upload.py", "content": "second chunk content here"},
    ]
    with patch.object(rag_retrieve, "retrieve_context", return_value=chunks), \
         patch.object(rag_retrieve, "_format_rag_block", return_value="[chunk1]\n[chunk2]"):
        result = _invoke_retrieve("anything", cfg=cfg, k=5)
    assert "Retrieved 2 chunk(s)" in result
    assert "ai_response_abc" in result or "[chunk1]" in result


def test_retrieve_rag_k_clamped_to_20():
    """k is clamped to [1, 20] regardless of caller input."""
    captured_k = {}

    def fake_retrieve_context(query, scope, k):
        captured_k["k"] = k
        return []

    with patch.object(rag_retrieve, "retrieve_context", fake_retrieve_context), \
         patch.object(rag_retrieve, "_format_rag_block", return_value=""):
        # Too small
        _invoke_retrieve("q", cfg=None, k=0)
        assert captured_k["k"] == 1
        # Too large
        _invoke_retrieve("q", cfg=None, k=1000)
        assert captured_k["k"] == 20
        # In range
        _invoke_retrieve("q", cfg=None, k=7)
        assert captured_k["k"] == 7


def test_retrieve_rag_handles_retrieve_failure():
    """If retrieve_context raises, returns ERROR without crashing."""
    cfg = {"configurable": {"active_thread_id": "thr_x"}}
    with patch.object(rag_retrieve, "retrieve_context", side_effect=RuntimeError("db down")):
        result = _invoke_retrieve("q", cfg=cfg)
    assert "ERROR" in result
    assert "db down" in result


def test_retrieve_rag_handles_format_failure_with_fallback():
    """If _format_rag_block raises, falls back to inline formatting."""
    cfg = {"configurable": {"active_thread_id": "thr_x"}}
    chunks = [{"source": "x.py", "content": "hello world"}]
    with patch.object(rag_retrieve, "retrieve_context", return_value=chunks), \
         patch.object(rag_retrieve, "_format_rag_block", side_effect=AttributeError("nope")):
        result = _invoke_retrieve("q", cfg=cfg)
    assert "Retrieved 1 chunk" in result
    assert "hello world" in result
    assert "x.py" in result


def test_retrieve_rag_no_active_thread_falls_back_to_global():
    """If no scope is given and no active_thread_id is in config, search globally."""
    captured_scope = {}

    def fake_retrieve_context(query, scope, k):
        captured_scope["scope"] = scope
        return []

    with patch.object(rag_retrieve, "retrieve_context", fake_retrieve_context), \
         patch.object(rag_retrieve, "_format_rag_block", return_value=""):
        _invoke_retrieve("q", cfg=None)
    # scope will be None — the underlying mixer treats None as global search.
    assert captured_scope["scope"] is None


def test_retrieve_rag_handles_no_rag_stack(monkeypatch):
    """When retrieve_context is None (RAG imports failed), return ERROR gracefully."""
    monkeypatch.setattr(rag_retrieve, "retrieve_context", None)
    result = _invoke_retrieve("q")
    assert "ERROR" in result
    assert "RAG stack unavailable" in result


def test_truncate_helper():
    """Long strings get truncated with a marker; short ones pass through."""
    short = "x" * 100
    assert rag_retrieve._truncate(short) == short
    long = "x" * 2000
    out = rag_retrieve._truncate(long, limit=500)
    assert len(out) < 600  # 500 + marker
    assert "truncated" in out
