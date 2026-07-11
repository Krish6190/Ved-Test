"""Tests for executor failure handling and incremental plan updates.

Verifies:
  - When the LLM call fails, the chunk is marked_failed in the plan file
  - The failed chunk is reset to pending so the planner can retry
  - The plan file is updated incrementally (mark_done/mark_failed only
    mutate one chunk's entry; the file stays small)
"""
import json
import queue
import threading
import data.plans as plan_store
import graph.nodes.executor as executor_mod
from langchain_core.messages import AIMessage


def _make_fake_llm():
    """Fake LLM that has bind_tools (returns self) so executor_node reaches
    the streaming call. _stream_one_iteration is mocked per-test to either
    return a tuple or raise."""
    class _FakeLLM:
        def bind_tools(self, tools):
            return self
    return _FakeLLM()


def _make_plan(tmp_path, monkeypatch, chunks=None):
    """Create a real plan in tmp_path with the given chunk instructions."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan(
        "test task",
        chunks or ["first chunk", "second chunk"],
    )
    plan_store.save_plan(plan)
    return plan


def _make_config(approval_value=True):
    q = queue.Queue()
    event = threading.Event()
    state = {"value": approval_value, "session_id": "test"}
    return {
        "configurable": {
            "token_queue": q,
            "approval_event": event,
            "approval_state": state,
            "session_id": "test",
        }
    }, q, event, state


def test_executor_marks_chunk_failed_on_llm_exception(tmp_path, monkeypatch):
    """If _stream_one_iteration raises, mark_failed + reset to pending."""
    plan = _make_plan(tmp_path, monkeypatch)
    cfg, q, _, _ = _make_config()
    def _raise(*a, **kw):
        raise RuntimeError("llm down")
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _raise)

    # Build a fake VedState-ish object the executor can read.
    class _FakeState:
        active_plan_id = plan["plan_id"]
        current_chunk_id = 1
        mode = "standard"
        route_intent = "A"

    result = executor_mod.executor_node(_FakeState(), get_llm=lambda: _make_fake_llm(), config=cfg)
    # No message stored (executor returns empty messages on failure).
    assert result["messages"] == []

    # Plan was reloaded and the failed chunk is now pending (retry-ready).
    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "pending"  # reset for retry
    assert "RuntimeError" in chunk1["output"]
    assert "llm down" in chunk1["output"]


def test_executor_emits_chunk_failed_event_on_exception(tmp_path, monkeypatch):
    """The token_queue receives a 'chunk_failed' plan_update event."""
    plan = _make_plan(tmp_path, monkeypatch)
    cfg, q, _, _ = _make_config()
    def _raise2(*a, **kw):
        raise ValueError("boom")
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _raise2)

    class _FakeState:
        active_plan_id = plan["plan_id"]
        current_chunk_id = 1
        mode = "standard"
        route_intent = "A"

    executor_mod.executor_node(_FakeState(), get_llm=lambda: _make_fake_llm(), config=cfg)

    # Drain the queue and find the chunk_failed event.
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    failed_events = [e for e in events if isinstance(e, tuple) and e[0] == "plan_update" and e[1].get("event") == "chunk_failed"]
    assert failed_events, f"no chunk_failed event found in {events}"
    assert failed_events[0][1]["chunk_id"] == 1
    assert "boom" in failed_events[0][1]["error"]


def test_executor_handles_missing_plan_gracefully(tmp_path, monkeypatch):
    """If plan_id points to a non-existent file, executor returns empty messages, no crash."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    cfg, q, _, _ = _make_config()

    class _FakeState:
        active_plan_id = "deadbeef"  # doesn't exist
        current_chunk_id = 1
        mode = "standard"
        route_intent = "A"

    result = executor_mod.executor_node(_FakeState(), get_llm=lambda: _make_fake_llm(), config=cfg)
    assert result["messages"] == []
    # No plan_update events emitted either.
    assert q.empty()


def test_executor_handles_missing_active_plan_id(tmp_path, monkeypatch):
    """When active_plan_id is None, executor returns empty messages."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    cfg, q, _, _ = _make_config()

    class _FakeState:
        active_plan_id = None
        current_chunk_id = None
        mode = "standard"
        route_intent = "A"

    result = executor_mod.executor_node(_FakeState(), get_llm=lambda: _make_fake_llm(), config=cfg)
    assert result["messages"] == []


def test_executor_successful_run_marks_done_incrementally(tmp_path, monkeypatch):
    """On success: mark_done writes only the chunk entry; file size stays small."""
    plan = _make_plan(tmp_path, monkeypatch, chunks=["read foo", "edit foo"])
    cfg, q, _, _ = _make_config()

    # Mock a successful LLM call that returns content + no tool calls.
    def _ok(*a, **kw):
        return ("executor finished chunk", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _ok)

    class _FakeState:
        active_plan_id = plan["plan_id"]
        current_chunk_id = 1
        mode = "standard"
        route_intent = "A"

    result = executor_mod.executor_node(_FakeState(), get_llm=lambda: _make_fake_llm(), config=cfg)
    # Bug #6 fix: executor now surfaces chunk output as an AIMessage so the
    # user sees the chunk result in chat. Verify exactly one AIMessage with
    # the LLM's output_text is emitted on success.
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "executor finished chunk"

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    chunk2 = next(c for c in updated["chunks"] if c["id"] == 2)
    assert chunk1["status"] == "done"
    assert chunk1["output"] == "executor finished chunk"
    assert chunk2["status"] == "executing"  # next chunk auto-queued


def test_plan_file_stays_small_with_incremental_updates(tmp_path, monkeypatch):
    """Adding a 4th chunk shouldn't rewrite the previous 3 chunks' content."""
    plan = _make_plan(tmp_path, monkeypatch, chunks=["a", "b", "c"])
    # Mutate only chunk 2's status.
    plan_store.mark_done(plan, 2, "result of b")
    plan_store.save_plan(plan)

    raw = json.loads((tmp_path / f"{plan['plan_id']}.json").read_text())
    # chunk 1 untouched, chunk 2 updated, chunk 3 untouched.
    assert raw["chunks"][0]["output"] is None
    assert raw["chunks"][0]["status"] == "pending"
    assert raw["chunks"][1]["output"] == "result of b"
    assert raw["chunks"][1]["status"] == "done"
    assert raw["chunks"][2]["output"] is None
    assert raw["chunks"][2]["status"] == "pending"
