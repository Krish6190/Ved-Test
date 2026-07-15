"""Tests for retrieve_rag argument normalization and tool-loop resilience."""
from unittest.mock import MagicMock, patch

import pytest

from graph.tools._common import coerce_retrieve_rag_args, normalize_tool_args


def test_normalize_retrieve_rag_paths_only():
    """A paths-only payload must be mapped onto the required query field."""
    out = normalize_tool_args("retrieve_rag", {"paths": ["sandbox_dir/"]})
    assert out["query"] == "sandbox_dir/"
    assert out["paths"] == ["sandbox_dir/"]


def test_normalize_retrieve_rag_file_path_alias():
    """Common path aliases should synthesize a valid query string."""
    out = normalize_tool_args("retrieve_rag", {"file_path": "foo.py"})
    assert out["query"] == "foo.py"


def test_planner_tool_loop_survives_bad_rag_args():
    """Planner tool dispatch must not crash when the model omits query."""
    from graph.nodes.planner import _execute_planner_tool_call

    captured = {}

    def _fake_invoke(args, config=None):
        captured["args"] = args
        return "Retrieved 1 chunk(s)"

    with patch("graph.nodes.planner.retrieve_rag") as mock_tool:
        mock_tool.invoke.side_effect = _fake_invoke
        result = _execute_planner_tool_call(
            {"name": "retrieve_rag", "args": {"paths": ["sandbox_dir/"]}},
            config={"configurable": {"active_thread_id": "thr_norm"}},
        )

    assert "ERROR" not in result
    assert captured["args"]["query"] == "sandbox_dir/"
    assert captured["args"]["paths"] == ["sandbox_dir/"]


def test_executor_tool_loop_survives_bad_rag_args():
    """Executor inline invoke must normalize retrieve_rag args before validation."""
    from graph.nodes import executor as executor_mod
    from graph.tools.rag_retrieve import retrieve_rag

    result, ok = executor_mod._invoke_tool_sync(
        retrieve_rag,
        {"paths": ["sandbox_test_dir/"]},
        config={"configurable": {"active_thread_id": "thr_exec_norm"}},
    )

    assert ok is True
    assert isinstance(result, str)
    assert not result.startswith("ERROR: ValidationError")


def test_coerce_retrieve_rag_args_after_partial_normalization():
    """Second-chance coercion still produces a non-empty query."""
    out = coerce_retrieve_rag_args({"paths": "voice/"})
    assert out["query"] == "voice/"
    assert out["paths"] == ["voice/"]
