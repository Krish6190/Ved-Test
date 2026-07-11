"""Tests for the non-blocking cumulative multi-file file-edit approval gate.

Covers the new Cursor-style design:
  - edit_file / overwrite_file calls queue pending tasks in
    `file_edit_pending_tasks` keyed by absolute path and return immediately.
  - Multiple distinct files accumulate in the queue; repeat edits to the
    same path overwrite that entry.
  - read_file returns the virtual (post-edit) content when a pending edit
    exists, neutralizing the stale-read warning.
  - chatbot._file_edit_approval_worker applies approved edits and drops
    rejected ones, supporting approve_all / reject_all / per-file decisions.
"""
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.plans as plan_store
import graph.actions.filesystem as fs_actions
import graph.nodes.executor as executor_mod
import graph.tools.file_reader as file_reader_mod
import graph.tools.file_search as file_search_mod


# ---- helpers ----


def _bypass_safety(monkeypatch):
    """Make filesystem safety checks permit the temp test paths."""
    monkeypatch.setattr(fs_actions, "_is_under_any", lambda path, roots: True)
    monkeypatch.setattr(file_reader_mod, "is_safe_self_healing", lambda path: True)
    monkeypatch.setattr(file_reader_mod, "is_safe_default", lambda path: True)
    monkeypatch.setattr(file_search_mod, "is_safe_self_healing", lambda path: True)
    monkeypatch.setattr(file_search_mod, "is_safe_default", lambda path: True)

class _FakeLLM:
    def bind_tools(self, tools):
        return self


class _FakeState:
    """Minimal VedState-like object."""
    def __init__(self, plan_id, chunk_id, *, mode="coder", self_healing=True):
        self.active_plan_id = plan_id
        self.current_chunk_id = chunk_id
        self.mode = mode
        self.route_intent = "A"
        self.self_healing = self_healing


def _setup_plan(tmp_path, monkeypatch, chunks=None):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", chunks or ["edit thing"])
    plan_store.save_plan(plan)
    return plan


def _make_config(tmp_path, *, file_edit_event=None, file_edit_state=None,
                 file_edit_tasks=None, file_edit_lock=None):
    """Build a RunnableConfig dict with the multi-file approval gate wired up."""
    q = queue.Queue()
    configurable = {"token_queue": q}
    if file_edit_event is not None:
        configurable["file_edit_approval_event"] = file_edit_event
        configurable["file_edit_approval_state"] = file_edit_state or {"value": None}
        configurable["file_edit_pending_tasks"] = file_edit_tasks if file_edit_tasks is not None else {}
        configurable["file_edit_pending_lock"] = file_edit_lock or threading.Lock()
    return {"configurable": configurable}, q


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def _find_file_edit_requests(events):
    return [e for e in events
            if isinstance(e, tuple) and e[0] == "file_edit_approval_request"]


def _run_worker_once(chatbot, decision):
    """Spawn the file-edit worker long enough to process one decision."""
    stop = threading.Event()
    chatbot._file_edit_worker_stop = stop
    t = threading.Thread(target=chatbot._file_edit_approval_worker, daemon=True)
    t.start()
    chatbot.submit_file_edit_approval(decision)
    # Give the worker a moment to process the event.
    time.sleep(0.2)
    stop.set()
    t.join(timeout=1.0)


# ---- 1. Distinct files accumulate in the pending queue ----

def test_multiple_file_edits_accumulate_in_queue(tmp_path, monkeypatch):
    """Edits to file_A.py and file_B.py should both exist in the queue."""
    plan = _setup_plan(tmp_path, monkeypatch)
    file_a = tmp_path / "file_A.py"
    file_b = tmp_path / "file_B.py"
    file_a.write_text("a\n", encoding="utf-8")
    file_b.write_text("b\n", encoding="utf-8")

    def fake_resolve(path_str, self_healing):
        if "file_A" in path_str:
            return file_a, None
        if "file_B" in path_str:
            return file_b, None
        return Path(path_str), None
    monkeypatch.setattr(executor_mod, "_resolve_and_check", fake_resolve)

    pending_tasks = {}
    lock = threading.Lock()
    event = threading.Event()
    state = {"value": None}
    cfg, q = _make_config(
        tmp_path,
        file_edit_event=event, file_edit_state=state,
        file_edit_tasks=pending_tasks, file_edit_lock=lock,
    )

    calls = []
    def _stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return ("editing", [
                {"id": "c1", "name": "edit_file",
                 "args": {"path": str(file_a), "old_text": "a", "new_text": "A"}},
                {"id": "c2", "name": "edit_file",
                 "args": {"path": str(file_b), "old_text": "b", "new_text": "B"}},
            ])
        return ("done", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    with lock:
        assert str(file_a) in pending_tasks
        assert str(file_b) in pending_tasks
        assert pending_tasks[str(file_a)]["args"]["new_text"] == "A"
        assert pending_tasks[str(file_b)]["args"]["new_text"] == "B"

    # Disk was NOT touched yet.
    assert file_a.read_text(encoding="utf-8") == "a\n"
    assert file_b.read_text(encoding="utf-8") == "b\n"

    # UI payload includes the full task map.
    events = _drain(q)
    reqs = _find_file_edit_requests(events)
    assert len(reqs) >= 1
    payload = reqs[-1][1]
    assert "tasks" in payload
    assert str(file_a) in payload["tasks"]
    assert str(file_b) in payload["tasks"]


# ---- 2. Repeat edits to the same path overwrite the queue entry ----

def test_repeat_edit_to_same_path_overwrites_queue_entry(tmp_path, monkeypatch):
    """Two edit_file calls on the same file should leave only the latest task."""
    plan = _setup_plan(tmp_path, monkeypatch)
    target = tmp_path / "single.py"
    target.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(executor_mod, "_resolve_and_check", lambda p, sh: (target, None))

    pending_tasks = {}
    lock = threading.Lock()
    cfg, q = _make_config(
        tmp_path,
        file_edit_event=threading.Event(), file_edit_state={"value": None},
        file_edit_tasks=pending_tasks, file_edit_lock=lock,
    )

    calls = []
    def _stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return ("editing", [
                {"id": "c1", "name": "edit_file",
                 "args": {"path": str(target), "old_text": "original", "new_text": "first"}},
                {"id": "c2", "name": "edit_file",
                 "args": {"path": str(target), "old_text": "first", "new_text": "second"}},
            ])
        return ("done", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    with lock:
        assert len(pending_tasks) == 1
        assert pending_tasks[str(target)]["args"]["new_text"] == "second"


# ---- 3. read_file returns virtual content when a pending edit exists ----

def test_read_file_returns_virtual_content_with_pending_edit(tmp_path, monkeypatch):
    """The LLM should see its own proposed change, not the on-disk content."""
    _bypass_safety(monkeypatch)
    plan = _setup_plan(tmp_path, monkeypatch)
    target = tmp_path / "readme.txt"
    target.write_text("hello world\n", encoding="utf-8")
    monkeypatch.setattr(executor_mod, "_resolve_and_check", lambda p, sh: (target, None))

    pending_tasks = {
        str(target): {
            "tool_name": "edit_file",
            "path": str(target),
            "args": {
                "path": str(target),
                "old_text": "hello world",
                "new_text": "hello virtual world",
            },
        },
    }
    cfg, q = _make_config(
        tmp_path,
        file_edit_event=threading.Event(), file_edit_state={"value": None},
        file_edit_tasks=pending_tasks, file_edit_lock=threading.Lock(),
    )

    calls = []
    def _stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return ("reading", [
                {"id": "c1", "name": "read_file", "args": {"path": str(target)}},
            ])
        return ("done", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    tc = chunk1["tool_calls"][0]
    assert tc["name"] == "read_file"
    assert tc["ok"] is True
    assert "hello virtual world" in tc["result"]
    assert "hello world" not in tc["result"]
    # Disk is still the original content.
    assert target.read_text(encoding="utf-8") == "hello world\n"


# ---- 4. overwrite_file is queued and returns pending result ----

def test_overwrite_file_queued_without_touching_disk(tmp_path, monkeypatch):
    """overwrite_file should land in the queue and leave disk untouched."""
    plan = _setup_plan(tmp_path, monkeypatch)
    target = tmp_path / "wipe.py"
    target.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(executor_mod, "_resolve_and_check", lambda p, sh: (target, None))

    pending_tasks = {}
    cfg, q = _make_config(
        tmp_path,
        file_edit_event=threading.Event(), file_edit_state={"value": None},
        file_edit_tasks=pending_tasks, file_edit_lock=threading.Lock(),
    )

    calls = []
    def _stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return ("editing", [
                {"id": "c1", "name": "overwrite_file",
                 "args": {"path": str(target), "content": "new"}},
            ])
        return ("done", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    assert str(target) in pending_tasks
    assert pending_tasks[str(target)]["tool_name"] == "overwrite_file"
    assert target.read_text(encoding="utf-8") == "old\n"

    events = _drain(q)
    reqs = _find_file_edit_requests(events)
    assert len(reqs) == 1
    assert reqs[0][1]["operation"] == "overwrite"


# ---- 5. Worker apply-all applies every queued edit ----

def test_worker_approve_all_applies_all_edits(tmp_path, monkeypatch):
    """approve_all should apply every pending task and clear the queue."""
    _bypass_safety(monkeypatch)
    from chatbot import Chatbot

    target_a = tmp_path / "a.py"
    target_b = tmp_path / "b.py"
    target_a.write_text("a\n", encoding="utf-8")
    target_b.write_text("b\n", encoding="utf-8")

    chatbot = Chatbot.__new__(Chatbot)
    chatbot._file_edit_pending_tasks = {
        str(target_a): {
            "tool_name": "edit_file",
            "path": str(target_a),
            "args": {"path": str(target_a), "old_text": "a", "new_text": "A"},
            "self_healing": True,
        },
        str(target_b): {
            "tool_name": "edit_file",
            "path": str(target_b),
            "args": {"path": str(target_b), "old_text": "b", "new_text": "B"},
            "self_healing": True,
        },
    }
    chatbot._file_edit_pending_lock = threading.Lock()
    chatbot._file_edit_approval_event = threading.Event()
    chatbot._file_edit_approval_state = {"value": None}

    _run_worker_once(chatbot, {"action": "approve_all", "paths": []})

    assert target_a.read_text(encoding="utf-8") == "A\n"
    assert target_b.read_text(encoding="utf-8") == "B\n"
    with chatbot._file_edit_pending_lock:
        assert chatbot._file_edit_pending_tasks == {}


# ---- 6. Worker reject-all drops every queued edit ----

def test_worker_reject_all_drops_all_edits(tmp_path, monkeypatch):
    """reject_all should clear the queue without touching disk."""
    _bypass_safety(monkeypatch)
    from chatbot import Chatbot

    target = tmp_path / "keep.py"
    target.write_text("keep me\n", encoding="utf-8")

    chatbot = Chatbot.__new__(Chatbot)
    chatbot._file_edit_pending_tasks = {
        str(target): {
            "tool_name": "edit_file",
            "path": str(target),
            "args": {"path": str(target), "old_text": "keep me", "new_text": "replaced"},
            "self_healing": True,
        },
    }
    chatbot._file_edit_pending_lock = threading.Lock()
    chatbot._file_edit_approval_event = threading.Event()
    chatbot._file_edit_approval_state = {"value": None}

    _run_worker_once(chatbot, {"action": "reject_all", "paths": []})

    assert target.read_text(encoding="utf-8") == "keep me\n"
    with chatbot._file_edit_pending_lock:
        assert chatbot._file_edit_pending_tasks == {}


# ---- 7. Worker approve individual applies only selected files ----

def test_worker_approve_individual_applies_only_selected_files(tmp_path, monkeypatch):
    """approve with paths=[a] should apply a and leave b pending."""
    _bypass_safety(monkeypatch)
    from chatbot import Chatbot

    target_a = tmp_path / "a.py"
    target_b = tmp_path / "b.py"
    target_a.write_text("a\n", encoding="utf-8")
    target_b.write_text("b\n", encoding="utf-8")

    chatbot = Chatbot.__new__(Chatbot)
    chatbot._file_edit_pending_tasks = {
        str(target_a): {
            "tool_name": "edit_file",
            "path": str(target_a),
            "args": {"path": str(target_a), "old_text": "a", "new_text": "A"},
            "self_healing": True,
        },
        str(target_b): {
            "tool_name": "edit_file",
            "path": str(target_b),
            "args": {"path": str(target_b), "old_text": "b", "new_text": "B"},
            "self_healing": True,
        },
    }
    chatbot._file_edit_pending_lock = threading.Lock()
    chatbot._file_edit_approval_event = threading.Event()
    chatbot._file_edit_approval_state = {"value": None}

    _run_worker_once(chatbot, {"action": "approve", "paths": [str(target_a)]})

    assert target_a.read_text(encoding="utf-8") == "A\n"
    assert target_b.read_text(encoding="utf-8") == "b\n"
    with chatbot._file_edit_pending_lock:
        assert str(target_a) not in chatbot._file_edit_pending_tasks
        assert str(target_b) in chatbot._file_edit_pending_tasks


# ---- 8. Worker reject individual drops only selected files ----

def test_worker_reject_individual_drops_only_selected_files(tmp_path, monkeypatch):
    """reject with paths=[a] should drop a and leave b pending."""
    _bypass_safety(monkeypatch)
    from chatbot import Chatbot

    target_a = tmp_path / "a.py"
    target_b = tmp_path / "b.py"
    target_a.write_text("a\n", encoding="utf-8")
    target_b.write_text("b\n", encoding="utf-8")

    chatbot = Chatbot.__new__(Chatbot)
    chatbot._file_edit_pending_tasks = {
        str(target_a): {
            "tool_name": "edit_file",
            "path": str(target_a),
            "args": {"path": str(target_a), "old_text": "a", "new_text": "A"},
            "self_healing": True,
        },
        str(target_b): {
            "tool_name": "edit_file",
            "path": str(target_b),
            "args": {"path": str(target_b), "old_text": "b", "new_text": "B"},
            "self_healing": True,
        },
    }
    chatbot._file_edit_pending_lock = threading.Lock()
    chatbot._file_edit_approval_event = threading.Event()
    chatbot._file_edit_approval_state = {"value": None}

    _run_worker_once(chatbot, {"action": "reject", "paths": [str(target_a)]})

    # Neither file should be changed.
    assert target_a.read_text(encoding="utf-8") == "a\n"
    assert target_b.read_text(encoding="utf-8") == "b\n"
    with chatbot._file_edit_pending_lock:
        assert str(target_a) not in chatbot._file_edit_pending_tasks
        assert str(target_b) in chatbot._file_edit_pending_tasks


# ---- 9. Missing approval infrastructure falls back to direct tool invoke ----

def test_missing_file_edit_infrastructure_falls_back_to_tool_invoke(tmp_path, monkeypatch):
    """No event/state in config -> the original _invoke_tool_sync path is used."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, q = _make_config(tmp_path)  # no file_edit_event/state

    def _explode(*a, **kw):
        raise AssertionError("action should not be called directly in fallback mode")
    monkeypatch.setattr(executor_mod, "edit_file_action", _explode)
    monkeypatch.setattr(executor_mod, "overwrite_file_action", _explode)

    invoke_calls = []
    def fake_invoke(tool, args):
        invoke_calls.append((tool.name, args))
        return ("tool-layer-result", True)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync", fake_invoke)

    calls = []
    def _stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return ("editing", [{
                "id": "c1", "name": "edit_file",
                "args": {"path": "x.py", "old_text": "a", "new_text": "b"},
            }])
        return ("done", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    assert len(invoke_calls) >= 1
    assert invoke_calls[0] == ("edit_file", {"path": "x.py", "old_text": "a", "new_text": "b"})
    events = _drain(q)
    assert _find_file_edit_requests(events) == []


# ---- 10. Preview payload is truncated ----

def test_preview_text_is_truncated_in_payload(tmp_path, monkeypatch):
    """The preview in the file_edit_approval_request payload must be truncated."""
    plan = _setup_plan(tmp_path, monkeypatch)
    target = tmp_path / "big.txt"
    target.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(executor_mod, "_resolve_and_check", lambda p, sh: (target, None))

    huge_old = "OLD-" * 500
    huge_new = "NEW-" * 500

    cfg, q = _make_config(
        tmp_path,
        file_edit_event=threading.Event(), file_edit_state={"value": None},
        file_edit_tasks={}, file_edit_lock=threading.Lock(),
    )

    calls = []
    def _stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return ("go", [{
                "id": "c1", "name": "edit_file",
                "args": {"path": str(target), "old_text": huge_old, "new_text": huge_new},
            }])
        return ("done", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    reqs = _find_file_edit_requests(_drain(q))
    assert len(reqs) == 1
    payload = reqs[0][1]
    assert payload["operation"] == "edit"
    assert len(payload["preview"]["old"]) == executor_mod._PREVIEW_MAX_CHARS
    assert len(payload["preview"]["new"]) == executor_mod._PREVIEW_MAX_CHARS


# ---- 11. submit_file_edit_approval accepts legacy bool ----

def test_submit_file_edit_approval_accepts_bool():
    """Backward-compat: True -> approve_all, False -> reject_all."""
    from chatbot import Chatbot

    chatbot = Chatbot.__new__(Chatbot)
    chatbot._file_edit_approval_state = {"value": None}
    chatbot._file_edit_approval_event = threading.Event()

    chatbot.submit_file_edit_approval(True)
    assert chatbot._file_edit_approval_state["value"]["action"] == "approve_all"
    assert chatbot._file_edit_approval_event.is_set()

    chatbot._file_edit_approval_event.clear()
    chatbot._file_edit_approval_state["value"] = None
    chatbot.submit_file_edit_approval(False)
    assert chatbot._file_edit_approval_state["value"]["action"] == "reject_all"
    assert chatbot._file_edit_approval_event.is_set()
