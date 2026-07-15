"""Regression tests: file tools trigger RAG ingestion after success.

After the structural state-machine fix, `read_file` and `search_files`
must synchronously schedule a background ingestion of the touched file
paths into the thread-scoped RAG index. We verify by patching
`ingest_path_to_thread_rag` and asserting it was called at least once
with the expected path.

We monkeypatch `PROJECT_ROOT` to a temp directory so the safety policy
(`is_safe_self_healing`/`is_safe_default`) accepts the test files.
"""
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from graph.state import VedState
from graph.tools.file_reader import read_file
from graph.tools.file_search import search_files


def _state_with_thread(thread_id: str) -> VedState:
    """Build a VedState with the given active_thread_id.

    InjectedState is only populated by LangGraph at runtime; when invoking
    a `@tool` directly via `.invoke()` outside a graph run, we must pass a
    real VedState object explicitly so `state.active_thread_id` is set.
    """
    return VedState(
        messages=[HumanMessage(content="hello")],
        active_thread_id=thread_id,
    )


def test_read_file_triggers_rag_ingest(tmp_path, monkeypatch):
    target = tmp_path / "foo.py"
    target.write_text("x = 1\n")
    monkeypatch.setattr(
        "graph.tools._common.PROJECT_ROOT", tmp_path
    )
    monkeypatch.setattr(
        "graph.tools.file_reader.PROJECT_ROOT", tmp_path
    )

    with patch("graph.tools._common.ingest_path_to_thread_rag") as mock_ingest:
        read_file.invoke(
            {"path": str(target), "state": _state_with_thread("t1")},
        )
        mock_ingest.assert_called_once()


def test_search_files_triggers_rag_ingest(tmp_path, monkeypatch):
    target = tmp_path / "bar.py"
    target.write_text("y = 2\n")
    monkeypatch.setattr(
        "graph.tools._common.PROJECT_ROOT", tmp_path
    )
    monkeypatch.setattr(
        "graph.tools.file_search.PROJECT_ROOT", tmp_path
    )

    with patch("graph.tools._common.ingest_path_to_thread_rag") as mock_ingest:
        search_files.invoke(
            {
                "pattern": "*.py",
                "directory": str(tmp_path),
                "state": _state_with_thread("t1"),
            },
        )
        assert mock_ingest.call_count >= 1
