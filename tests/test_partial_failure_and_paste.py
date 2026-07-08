"""Tests for executor failure handling and incremental plan updates.

Updated for the new agent-loop design: failures are detected at the
tool level (when _invoke_tool_sync returns ok=False), not by scanning
the LLM's prose output for ERROR keywords.
"""
import queue
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.plans as plan_store
import graph.nodes.executor as executor_mod


# ---- helpers ----

class _FakeLLM:
    def bind_tools(self, tools):
        return self


class _FakeState:
    def __init__(self, plan_id, chunk_id):
        self.active_plan_id = plan_id
        self.current_chunk_id = chunk_id
        self.mode = "standard"
        self.route_intent = "A"


def _make_config():
    q = queue.Queue()
    return {"configurable": {
        "token_queue": q, "approval_event": threading.Event(),
        "approval_state": {"value": True, "session_id": "test"},
        "session_id": "test",
    }}, q


def _setup_plan(tmp_path, monkeypatch, chunks=None):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", chunks or ["do thing 1", "do thing 2"])
    plan_store.save_plan(plan)
    return plan


# ---- tool-level failure detection ----

def test_executor_marks_chunk_failed_when_tool_returns_error(tmp_path, monkeypatch):
    """When _invoke_tool_sync returns ok=False, mark_failed + reset to pending."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    def _stream_with_tool_call(*a, **kw):
        # LLM emits a tool_call. Executor will execute it inline.
        return ("trying execute_python", [{"id": "c1", "name": "execute_python", "args": {"code": "import x"}}])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream_with_tool_call)

    # The tool fails:
    def _failing_tool(_tool, _args):
        return ("ERROR: NameError: x is not defined", False)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync", _failing_tool)

    # Use coder mode so execute_python is in the available tool set.
    state = _FakeState(plan["plan_id"], 1)
    state.mode = "coder"
    executor_mod.executor_node(
        state,
        get_llm=lambda: _FakeLLM(),
        config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "pending", "should reset to pending for retry"
    assert "NameError" in chunk1["output"]
    # The structured tool_calls log is preserved.
    assert len(chunk1["tool_calls"]) == 1
    assert chunk1["tool_calls"][0]["name"] == "execute_python"
    assert chunk1["tool_calls"][0]["ok"] is False


def test_executor_marks_done_when_all_tools_succeed(tmp_path, monkeypatch):
    """When all tools succeed and the LLM stops emitting tool calls, mark_done."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    def _stream_with_tool_call(*a, **kw):
        return ("read foo.py successfully", [{"id": "c1", "name": "read_file", "args": {"path": "foo.py"}}])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream_with_tool_call)

    def _ok_tool(_tool, _args):
        return ("file contents here", True)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync", _ok_tool)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(),
        config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "done"
    assert chunk1["tool_calls"][0]["ok"] is True
    # Auto-queue the next chunk.
    chunk2 = next(c for c in updated["chunks"] if c["id"] == 2)
    assert chunk2["status"] == "executing"


def test_executor_chunk_failed_event_emitted_with_tool_context(tmp_path, monkeypatch):
    """The chunk_failed SSE event includes the failing tool name."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, q = _make_config()

    def _stream_with_tool_call(*a, **kw):
        return ("trying", [{"id": "c1", "name": "execute_python", "args": {}}])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream_with_tool_call)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync",
                        lambda *a: ("ERROR: boom", False))

    state = _FakeState(plan["plan_id"], 1)
    state.mode = "coder"
    executor_mod.executor_node(
        state,
        get_llm=lambda: _FakeLLM(),
        config=cfg,
    )

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    failed = [
        e for e in events
        if isinstance(e, tuple) and e[0] == "plan_update" and e[1].get("event") == "chunk_failed"
    ]
    assert failed, f"expected chunk_failed event, got: {events}"
    assert failed[0][1]["failed_tool"] == "execute_python"
    assert "boom" in failed[0][1]["error"]


def test_executor_partial_failure_does_not_double_mark(monkeypatch, tmp_path):
    """If the executor's persist step fails, the executor returns gracefully."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    def _stream(*a, **kw):
        return ("trying", [{"id": "c1", "name": "execute_python", "args": {}}])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync",
                        lambda *a: ("ERROR: nope", False))
    monkeypatch.setattr(plan_store, "save_plan",
                        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))

    result = executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(),
        config=cfg,
    )
    assert result["messages"] == []  # No crash; empty messages preserved.


def test_executor_stops_at_first_tool_error(tmp_path, monkeypatch):
    """When tool 2 fails after tool 1 succeeds, tool 3 is NOT executed."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    # First iteration: LLM emits 3 tool calls.
    # Second iteration: LLM emits 0 tool calls (would normally finish).
    iter_count = [0]
    def _stream_iter(*a, **kw):
        iter_count[0] += 1
        if iter_count[0] == 1:
            return ("first", [
                {"id": "c1", "name": "read_file", "args": {"path": "a"}},
                {"id": "c2", "name": "execute_python", "args": {"code": "bad"}},
                {"id": "c3", "name": "edit_file", "args": {"path": "b"}},
            ])
        # Should never reach here (we STOP at the failing tool).
        return ("should not see this", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream_iter)

    call_count = [0]
    def _track_tool(_tool, args):
        call_count[0] += 1
        # read_file succeeds, execute_python fails -> we should stop here.
        if args.get("code") == "bad":
            return ("ERROR: code is bad", False)
        return ("ok", True)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync", _track_tool)

    # Coder mode so execute_python and edit_file are available.
    state = _FakeState(plan["plan_id"], 1)
    state.mode = "coder"
    executor_mod.executor_node(
        state,
        get_llm=lambda: _FakeLLM(),
        config=cfg,
    )

    # Only 2 tools called (read_file succeeded, execute_python failed,
    # edit_file was NEVER called).
    assert call_count[0] == 2, f"expected 2 tool calls, got {call_count[0]}"

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    # The structured log has exactly 2 entries (the third was never tried).
    assert len(chunk1["tool_calls"]) == 2


# ---- /chat large-paste handling ----

def test_chat_large_prompt_saves_to_rag_and_shortens():
    """Prompts over the threshold get their full text saved to RAG and replaced
    with a short reference. The bot's save_user_input_to_thread_rag is called."""
    from fastapi.testclient import TestClient
    from api import lifecycle
    from api.server import app

    lifecycle.reset_for_tests()
    fake_bot = MagicMock()
    fake_bot.save_user_input_to_thread_rag = MagicMock(return_value=True)

    def _respond(prompt):
        assert len(prompt) < 200, f"prompt was not shortened: len={len(prompt)}"
        return "OK: shortened prompt received"
    fake_bot.respond = _respond
    lifecycle._chatbot = fake_bot

    client = TestClient(app)
    big_prompt = "x" * 5000
    r = client.post("/chat", json={"prompt": big_prompt})

    assert r.status_code == 200
    assert fake_bot.save_user_input_to_thread_rag.called
    saved_args = fake_bot.save_user_input_to_thread_rag.call_args
    assert saved_args.args[0] == big_prompt
    assert "UserPaste_" in saved_args.args[1]


def test_chat_short_prompt_skips_rag_save():
    from fastapi.testclient import TestClient
    from api import lifecycle
    from api.server import app

    lifecycle.reset_for_tests()
    fake_bot = MagicMock()
    fake_bot.save_user_input_to_thread_rag = MagicMock(return_value=True)
    fake_bot.respond = MagicMock(return_value="OK: short prompt")
    lifecycle._chatbot = fake_bot

    client = TestClient(app)
    short_prompt = "hi" * 100
    r = client.post("/chat", json={"prompt": short_prompt})

    assert r.status_code == 200
    assert not fake_bot.save_user_input_to_thread_rag.called


def test_chat_large_prompt_rag_save_failure_falls_through_gracefully():
    from fastapi.testclient import TestClient
    from api import lifecycle
    from api.server import app

    lifecycle.reset_for_tests()
    fake_bot = MagicMock()
    fake_bot.save_user_input_to_thread_rag = MagicMock(side_effect=RuntimeError("rag down"))
    fake_bot.respond = MagicMock(return_value="OK: original prompt")
    lifecycle._chatbot = fake_bot

    client = TestClient(app)
    r = client.post("/chat", json={"prompt": "x" * 5000})

    assert r.status_code == 200
    fake_bot.respond.assert_called_once()
    called_prompt = fake_bot.respond.call_args.args[0]
    assert len(called_prompt) >= 5000
