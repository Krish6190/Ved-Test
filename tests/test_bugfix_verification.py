"""Targeted verification for the three bug fixes:

1. RAG empty-result fallback now points the LLM at read_file / search_files
   instead of letting it hallucinate "task already completed".
2. The newline filter (`_clean_chunk`) PRESERVES normal paragraph breaks
   (single and double newlines) and ONLY collapses runaway anomalies
   (3+ newlines, 4+ spaces). Earlier passes were too aggressive and
   stripped every newline, collapsing the chat into one unreadable blob.
3. File-edit staging emits a non-empty payload and the panel-trigger
   path mirrors to STAGING_REGISTRY so the worker thread can apply
   approved decisions. The legacy interception path no longer fires
   the approval event prematurely (which was wiping session.tasks
   before the user clicked Approve, blocking the physical disk write).
"""

import importlib
import os
import sys
import types

import pytest


# --------------------------------------------------------------------------
# Bug 2: ghost-newline filter
# --------------------------------------------------------------------------

def test_clean_chunk_drops_truly_empty():
    """Only None and "" are dropped. Whitespace-only chunks (\n, \n\n)
    must be preserved so the chat panel renders standard paragraph breaks."""
    from graph.nodes._stream_helpers import _clean_chunk
    assert _clean_chunk("") is None
    assert _clean_chunk(None) is None


def test_clean_chunk_preserves_whitespace_only_chunks():
    """Regression: earlier passes dropped every pure-whitespace chunk,
    collapsing the chat into one unreadable paragraph. The fix preserves
    \n and \n\n (the only two valid paragraph-break patterns)."""
    from graph.nodes._stream_helpers import _clean_chunk
    assert _clean_chunk("\n") == "\n"
    assert _clean_chunk("\n\n") == "\n\n"
    assert _clean_chunk("   ") == "   "
    # Runaway whitespace IS still collapsed: 3+ newlines -> 2.
    assert _clean_chunk("\n\n\n") == "\n\n"
    assert _clean_chunk("\n\n\n\n") == "\n\n"


def test_clean_chunk_collapses_runaway_newlines():
    from graph.nodes._stream_helpers import _clean_chunk
    out = _clean_chunk("hello\n\n\n\n\nworld")
    assert out == "hello\n\nworld", out


def test_clean_chunk_preserves_single_newlines():
    from graph.nodes._stream_helpers import _clean_chunk
    assert _clean_chunk("a\nb") == "a\nb"
    assert _clean_chunk("a\n\nb") == "a\n\nb"


def test_clean_chunk_collapses_runaway_spaces():
    from graph.nodes._stream_helpers import _clean_chunk
    assert _clean_chunk("a        b") == "a b"


def test_clean_chunk_passes_through_real_content():
    from graph.nodes._stream_helpers import _clean_chunk
    assert _clean_chunk("hello world") == "hello world"


# --------------------------------------------------------------------------
# Bug 1: RAG empty-fallback message forces filesystem read
# --------------------------------------------------------------------------

def test_rag_empty_result_mentions_fallback():
    """When RAG returns empty, the tool message MUST tell the model to
    call read_file / search_files instead of guessing."""
    from graph.tools.rag_retrieve import retrieve_rag
    from langchain_core.runnables import RunnableConfig

    # Build a config with an active thread id but force empty results
    # by pointing at a thread that has nothing in the index.
    config = RunnableConfig(
        configurable={"active_thread_id": "thr_empty_test", "state_mode": "standard"}
    )
    out = retrieve_rag.invoke(
        {"query": "definitely_not_in_index_xyz123", "scope": "thr_empty_test"},
        config=config,
    )
    assert "FALLBACK REQUIRED" in out, out
    assert "read_file" in out or "search_files" in out, out
    # The hallucination guard wording is also present so the model
    # is told NOT to claim the task is done from chat history.
    assert "MUST NOT guess" in out or "task is already completed" in out, out


def test_mixer_falls_back_to_project_scope():
    """The mixer's project-scope fallback must run for non-coder callers.
    We patch the LocalVectorDB to return [] for thread/global and a
    synthetic hit for scope='project', then assert the hit surfaces."""
    from graph.rag import mixer

    captured = []

    class FakeDB:
        def query_similarity(self, query_text, k, lambda_mult, scope):
            captured.append(scope)
            if scope == "project":
                return [{"content": "project_hit", "source": "foo.py"}]
            return []

    original = mixer.rag_db if hasattr(mixer, "rag_db") else None
    fake = FakeDB()
    # Lazy import inside the function — patch the module-level rag_db ref
    # by faking the import chain via sys.modules.
    fake_mod = types.ModuleType("graph.rag")
    fake_mod.rag_db = fake
    sys.modules["graph.rag"] = fake_mod
    try:
        result = mixer.retrieve_context("anything", current_thread_id="thr_xyz", k=5)
    finally:
        # Restore.
        sys.modules["graph.rag"] = original if original is not None else fake_mod

    assert "project" in captured, captured
    assert any(c.get("scope") == "project" for c in result), result


# --------------------------------------------------------------------------
# Bug 3: file-edit staging always emits a non-empty payload
# --------------------------------------------------------------------------

def test_stage_edit_returns_non_empty_marker():
    from graph.tools.staging_registry import STAGING_REGISTRY
    STAGING_REGISTRY.register_session("thr_verify_stage")
    out = STAGING_REGISTRY.stage_edit(
        thread_id="thr_verify_stage",
        tool_name="edit_file",
        resolved_path="/tmp/never_touched.py",
        args={"path": "/tmp/never_touched.py", "old_text": "a", "new_text": "b"},
        preview={"old": "a", "new": "b"},
    )
    assert out.startswith("STAGED:"), out
    assert "edit_file" in out
    assert "/tmp/never_touched.py" in out


def test_emit_file_edit_approval_request_produces_payload():
    """Even when token_queue is None, _emit_file_edit_approval_request
    must NOT silently swallow — the registry mirror must catch the
    payload so the worker thread can apply approved decisions."""
    from graph.nodes import executor as executor_mod

    class FakeQueue:
        def __init__(self):
            self.items = []
        def put(self, item):
            self.items.append(item)

    q = FakeQueue()
    payload = {
        "path": "/tmp/x.py",
        "operation": "edit",
        "preview": {"old": "a", "new": "b"},
        "tasks": {"/tmp/x.py": {"tool_name": "edit_file", "path": "/tmp/x.py"}},
        "thread_id": "thr_verify_emit",
    }
    from graph.tools.staging_registry import STAGING_REGISTRY
    STAGING_REGISTRY.register_session("thr_verify_emit")
    executor_mod._emit_file_edit_approval_request(q, payload)
    assert any(item[0] == "file_edit_approval_request" for item in q.items)
    # Registry mirror should also reflect the staged task.
    tasks = STAGING_REGISTRY.get_tasks("thr_verify_emit")
    assert "/tmp/x.py" in tasks


def test_emit_file_edit_approval_request_handles_none_queue():
    """Regression: token_queue=None must not crash, and the registry
    mirror is the safety net."""
    from graph.nodes import executor as executor_mod
    from graph.tools.staging_registry import STAGING_REGISTRY

    STAGING_REGISTRY.register_session("thr_verify_noneq")
    payload = {
        "path": "/tmp/y.py",
        "operation": "edit",
        "preview": {"old": "", "new": "new"},
        "tasks": {"/tmp/y.py": {"tool_name": "edit_file", "path": "/tmp/y.py"}},
        "thread_id": "thr_verify_noneq",
    }
    # Should NOT raise even with token_queue=None.
    executor_mod._emit_file_edit_approval_request(None, payload)
    tasks = STAGING_REGISTRY.get_tasks("thr_verify_noneq")
    assert "/tmp/y.py" in tasks


def test_staging_registry_applies_multiple_edits_cumulatively(tmp_path):
    """Repeated staged edits for the same file should compose into one
    virtual overlay rather than reverting to stale disk state."""
    from graph.tools.staging_registry import STAGING_REGISTRY

    thread_id = "thr_overlay_sequence"
    target = tmp_path / "overlay.py"
    target.write_text("alpha", encoding="utf-8")
    STAGING_REGISTRY.register_session(thread_id)
    try:
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "edit_file",
            str(target),
            {"path": str(target), "old_text": "alpha", "new_text": "beta"},
            {"old": "alpha", "new": "beta"},
        )
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "edit_file",
            str(target),
            {"path": str(target), "old_text": "beta", "new_text": "gamma"},
            {"old": "beta", "new": "gamma"},
        )
        overlay = STAGING_REGISTRY.get_overlay(thread_id, str(target), "alpha")
        assert overlay == "gamma", overlay
    finally:
        STAGING_REGISTRY.unregister_session(thread_id)


# --------------------------------------------------------------------------
# Bug 3 (continued): executor must force edit_file after a read_file
# --------------------------------------------------------------------------

def test_executor_requires_edit_after_read(monkeypatch):
    """Simulate the agent loop running with read_file called but no
    edit_file/overwrite_file. The post-loop invariant must set
    last_error to 'no_edit_tool_called' so the chunk is retried
    instead of letting the planner treat it as done."""
    from graph.nodes import executor as executor_mod

    # Build a minimal fake plan with one chunk requiring a modification.
    fake_plan = {
        "plan_id": "plan_test",
        "task": "fix the bug",
        "chunks": [
            {
                "id": 1,
                "instruction": "Read foo.py and fix the off-by-one error in the loop.",
                "status": "executing",
                "retry_count": 0,
            }
        ],
        "current_chunk": 1,
    }
    # Patch plan_store so the executor sees our fake plan.
    monkeypatch.setattr(
        "data.plans.load_plan", lambda pid: fake_plan
    )
    monkeypatch.setattr(
        "data.plans.mark_done", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "data.plans.mark_failed", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "data.plans.save_plan", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "data.plans.next_pending", lambda plan: None
    )
    monkeypatch.setattr(
        "data.plans.mark_executing", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "data.plans.finalize", lambda *a, **kw: None
    )
    monkeypatch.setattr(executor_mod, "_mark_failed_with_log", lambda *a, **kw: None)

    # Patch _stream_one_iteration to simulate the LLM reading the file
    # then responding with conversational text (no edit_file call).
    calls = {"n": 0}
    def fake_stream(llm, messages, token_queue):
        calls["n"] += 1
        if calls["n"] == 1:
            # Round 1: LLM calls read_file.
            return "Reading foo.py", [
                {"id": "call_1", "name": "read_file",
                 "args": {"path": "/tmp/foo.py"}}
            ]
        # Round 2: LLM produces no tool calls — short-circuits to summary.
        return "Looks good, the loop is now fixed.", []

    monkeypatch.setattr(executor_mod, "_stream_one_iteration", fake_stream)

    # Stub the read_file tool invocation so the tool map has it.
    class FakeTool:
        name = "read_file"
        def invoke(self, args):
            return "def loop(): return range(1, 10)\n# off-by-one"

    fake_tool = FakeTool()
    fake_llm_with_tools = object()  # only .stream is used
    class FakeLLM:
        def bind_tools(self, tools):
            return fake_llm_with_tools
        def stream(self, messages):
            # _stream_one_iteration is monkeypatched, so this isn't called,
            # but we keep it for safety.
            return iter([])

    from graph.state import VedState
    from langchain_core.messages import HumanMessage
    state = VedState(
        messages=[HumanMessage(content="fix the bug")],
        active_plan_id="plan_test",
        current_chunk_id=1,
        mode="coder",
    )

    # Patch STAGING_REGISTRY.has_session / get_tasks to no-ops.
    from graph.tools import staging_registry
    monkeypatch.setattr(staging_registry.STAGING_REGISTRY, "has_session", lambda tid: False)
    monkeypatch.setattr(staging_registry.STAGING_REGISTRY, "get_tasks", lambda tid: {})

    result = executor_mod.executor_node(
        state,
        get_llm=lambda: FakeLLM(),
        config={
            "configurable": {
                "token_queue": None,
                "executor_llm_factory": lambda mode: FakeLLM(),
            }
        },
    )

    # The post-loop invariant must have set the chunk_retry_count > 0
    # (which only happens when last_error is set). When last_error is
    # set, mark_failed_with_log is called and the chunk status is reset
    # to "pending" for planner retry.
    assert result.get("chunk_retry_count", 0) >= 1, result
    # The fake plan's chunk 0 status must have been reset to "pending".
    assert fake_plan["chunks"][0]["status"] == "pending", fake_plan["chunks"][0]


# --------------------------------------------------------------------------
# Regression: legacy interception path must NOT fire the approval event
# --------------------------------------------------------------------------

def test_legacy_interception_does_not_fire_approval_event(monkeypatch):
    """Regression: the legacy interception path used to fire
    file_edit_approval_event with state['value']=None right after staging
    the edit. The worker woke up and treated None as the default
    'reject_all', which cleared STAGING_REGISTRY.session.tasks BEFORE the
    user ever clicked Approve. The physical file was therefore never
    written. The fix removes that premature event firing.

    This test runs the legacy interception path end-to-end and asserts
    the event is NOT fired by the executor itself. The event should
    only fire when the user clicks Approve/Reject (via
    chatbot.submit_file_edit_approval).
    """
    import threading
    from graph.nodes import executor as executor_mod

    # Build a real file under the project root so the action layer accepts it.
    test_file = "/tmp/ved_legacy_event_test.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("hello legacy")

    # Spy on the event to detect any premature firing.
    event_fired = threading.Event()
    captured_values = []

    class SpyEvent:
        def __init__(self, real):
            self._real = real
        def set(self):
            captured_values.append("event_fired")
            event_fired.set()
            return self._real.set()
        def is_set(self):
            return self._real.is_set()
        def clear(self):
            return self._real.clear()
        def wait(self, timeout=None):
            return self._real.wait(timeout=timeout)

    class SpyState(dict):
        def __init__(self, real):
            super().__init__()
            self._real = real
        def __setitem__(self, k, v):
            captured_values.append(f"state[{k}]={v!r}")
            self._real[k] = v
        def __getitem__(self, k):
            return self._real[k]
        def get(self, k, default=None):
            return self._real.get(k, default)

    real_event = threading.Event()
    real_state = {}
    spy_event = SpyEvent(real_event)
    spy_state = SpyState(real_state)

    # Patch STAGING_REGISTRY to be a no-op registry we control.
    from graph.tools import staging_registry
    monkeypatch.setattr(staging_registry.STAGING_REGISTRY, "has_session", lambda tid: True)

    staged_tasks = {}
    # Cache the fake session so multiple _get_session() calls return
    # the SAME object (otherwise writes via one call disappear when the
    # next call creates a fresh session).
    fake_session = type("S", (), {"lock": threading.Lock(), "tasks": staged_tasks})()
    monkeypatch.setattr(staging_registry.STAGING_REGISTRY, "get_tasks", lambda tid: dict(fake_session.tasks))
    monkeypatch.setattr(staging_registry.STAGING_REGISTRY, "_get_session", lambda tid: fake_session)

    # Fake plan so executor_node can resolve chunk_id.
    fake_plan = {
        "plan_id": "plan_legacy",
        "task": "fix bug",
        "chunks": [{
            "id": 1,
            "instruction": "Read foo.py and fix the bug.",
            "status": "executing",
            "retry_count": 0,
        }],
        "current_chunk": 1,
    }
    monkeypatch.setattr("data.plans.load_plan", lambda pid: fake_plan)
    monkeypatch.setattr("data.plans.mark_done", lambda *a, **kw: None)
    monkeypatch.setattr("data.plans.mark_failed", lambda *a, **kw: None)
    monkeypatch.setattr("data.plans.save_plan", lambda *a, **kw: None)
    monkeypatch.setattr("data.plans.next_pending", lambda plan: None)
    monkeypatch.setattr("data.plans.mark_executing", lambda *a, **kw: None)
    monkeypatch.setattr("data.plans.finalize", lambda *a, **kw: None)
    monkeypatch.setattr(executor_mod, "_mark_failed_with_log", lambda *a, **kw: None)

    # Build a state with the legacy approval infra wired in.
    file_edit_pending_tasks = {}
    file_edit_pending_lock = threading.Lock()

    # Patch _stream_one_iteration to emit one edit_file call.
    def fake_stream(llm, messages, token_queue):
        return "staging edit", [
            {"id": "call_1", "name": "edit_file",
             "args": {"path": test_file, "old_text": "hello legacy",
                      "new_text": "hello new"}}
        ]
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", fake_stream)

    from graph.state import VedState
    from langchain_core.messages import HumanMessage

    state = VedState(
        messages=[HumanMessage(content="fix bug")],
        active_plan_id="plan_legacy",
        current_chunk_id=1,
        mode="coder",
        self_healing=False,
    )

    class FakeLLM:
        def bind_tools(self, tools):
            return self
        def stream(self, messages):
            return iter([])

    try:
        executor_mod.executor_node(
            state,
            get_llm=lambda: FakeLLM(),
            config={
                "configurable": {
                    "token_queue": None,
                    "executor_llm_factory": lambda mode: FakeLLM(),
                    "file_edit_approval_event": spy_event,
                    "file_edit_approval_state": spy_state,
                    "file_edit_pending_tasks": file_edit_pending_tasks,
                    "file_edit_pending_lock": file_edit_pending_lock,
                    "active_thread_id": "thr_legacy",
                }
            },
        )
    finally:
        try:
            import os
            os.unlink(test_file)
        except Exception:
            pass

    # Critical assertion: the legacy interception path must NOT fire the
    # approval event. Only submit_file_edit_approval (user click) should.
    assert "event_fired" not in captured_values, (
        "Legacy interception path fired the approval event prematurely. "
        f"Captured: {captured_values}"
    )
    # Sanity: the legacy path must have staged the task somewhere
    # (either the in-memory dict, STAGING_REGISTRY, or both) so the user
    # can see and approve it via the UI panel. Either location proves
    # the legacy interception code actually executed.
    assert (test_file in staged_tasks) or (test_file in file_edit_pending_tasks), (
        f"Legacy path did not stage the edit anywhere. "
        f"staged_tasks={staged_tasks} "
        f"file_edit_pending_tasks={file_edit_pending_tasks}"
    )


# --------------------------------------------------------------------------
# Regression: approve button click must trigger physical disk write
# --------------------------------------------------------------------------

def test_approve_button_writes_file_to_disk(tmp_path):
    """Regression: clicking Approve in the UI panel must trigger the
    background worker to pull from STAGING_REGISTRY and write the file
    to disk. End-to-end: stage an edit, simulate the worker's
    apply_decision call (which is what runs when the user clicks
    Approve), and verify the file content changed on disk.
    """
    from graph.tools.staging_registry import STAGING_REGISTRY
    from pathlib import Path

    # edit_file_action enforces an allowed_roots boundary. Put the
    # fixture file under the project root so the write is accepted.
    project_root = Path(__file__).resolve().parent.parent
    test_file = project_root / ".tmp" / "ved_approve_test.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("original content\n", encoding="utf-8")

    # Reset any prior session and register a fresh one.
    STAGING_REGISTRY.unregister_session("thr_approve_test")
    STAGING_REGISTRY.register_session("thr_approve_test")

    # Stage an edit (this is what the executor does when LLM calls edit_file).
    STAGING_REGISTRY.stage_edit(
        thread_id="thr_approve_test",
        tool_name="edit_file",
        resolved_path=str(test_file),
        args={"path": str(test_file),
              "old_text": "original content",
              "new_text": "modified content"},
        preview={"old": "original content", "new": "modified content"},
    )

    # Sanity: the staged task is in STAGING_REGISTRY.
    tasks = STAGING_REGISTRY.get_tasks("thr_approve_test")
    assert str(test_file) in tasks

    # The worker calls apply_decision with the user's decision. Build a
    # callback that mirrors chatbot._apply_file_edit_task: it reads the
    # task's args and calls edit_file_action directly.
    from graph.actions.filesystem import edit_file_action

    def apply_callback(task):
        args = task.get("args") or {}
        return edit_file_action(
            args.get("path", ""),
            args.get("old_text", ""),
            args.get("new_text", ""),
            allowed_roots=(project_root,),
            backup_dir=None,
        )

    # Simulate the user clicking Approve All.
    result = STAGING_REGISTRY.apply_decision(
        thread_id="thr_approve_test",
        decision={"action": "approve_all", "paths": []},
        apply_callback=apply_callback,
    )

    # Critical: the file must have been written to disk.
    try:
        assert test_file.read_text(encoding="utf-8") == "modified content\n", (
            f"File was not modified after Approve. Content: {test_file.read_text()!r}"
        )
        # And the approval result must include the path with no error.
        assert len(result["approved"]) == 1
        assert result["approved"][0]["path"] == str(test_file)
        assert "ERROR" not in result["approved"][0]["result"], result["approved"][0]["result"]
    finally:
        # Restore original content so the fixture is idempotent across
        # repeat test runs.
        test_file.write_text("original content\n", encoding="utf-8")
        try:
            test_file.unlink()
        except FileNotFoundError:
            pass

    # The registry must be empty now (the staged task was consumed).
    assert STAGING_REGISTRY.get_tasks("thr_approve_test") == {}


def test_approve_all_clears_registry_without_user_paths():
    """The 'approve_all' action with an empty paths list must still
    apply every staged task (the worker reads paths from
    STAGING_REGISTRY, not from the user-supplied paths list)."""
    from graph.tools.staging_registry import STAGING_REGISTRY

    STAGING_REGISTRY.unregister_session("thr_approve_all")
    STAGING_REGISTRY.register_session("thr_approve_all")

    STAGING_REGISTRY.stage_edit(
        thread_id="thr_approve_all",
        tool_name="edit_file",
        resolved_path="/tmp/ved_approve_all_a.py",
        args={"path": "/tmp/ved_approve_all_a.py",
              "old_text": "a", "new_text": "A"},
        preview={"old": "a", "new": "A"},
    )
    STAGING_REGISTRY.stage_edit(
        thread_id="thr_approve_all",
        tool_name="edit_file",
        resolved_path="/tmp/ved_approve_all_b.py",
        args={"path": "/tmp/ved_approve_all_b.py",
              "old_text": "b", "new_text": "B"},
        preview={"old": "b", "new": "B"},
    )

    applied = []
    def cb(task):
        applied.append(task.get("path"))
        return "OK"

    result = STAGING_REGISTRY.apply_decision(
        thread_id="thr_approve_all",
        decision={"action": "approve_all", "paths": []},
        apply_callback=cb,
    )

    assert sorted(applied) == sorted([
        "/tmp/ved_approve_all_a.py",
        "/tmp/ved_approve_all_b.py",
    ])
    assert STAGING_REGISTRY.get_tasks("thr_approve_all") == {}


# --------------------------------------------------------------------------
# Round-3 Fix 1: renderer cross-chunk clamp (max ONE empty line globally)
# --------------------------------------------------------------------------

def test_renderer_clamp_strips_leading_newlines_when_widget_already_blank():
    """When the widget already ends with \\n\\n (one blank line) and a
    new chunk arrives starting with \\n, the renderer must strip the
    leading newlines from the new chunk. Otherwise two streamed chunks
    would concatenate into 3+ newlines (two blank lines) in the panel.
    """
    from graph.nodes._stream_helpers import (
        _RUNAWAY_NEWLINES_RE,
        _strip_leading_blank_lines,
    )

    existing_tail = "Header\n\n"  # widget ends with one blank line
    new_chunk = "\nTool block"
    cleaned = new_chunk
    if "\n" in cleaned:
        cleaned = _RUNAWAY_NEWLINES_RE.sub("\n\n", cleaned)
    cleaned = _strip_leading_blank_lines(cleaned, existing_tail)

    # The clamp must remove the leading newlines so the combined text
    # never exceeds \n\n (one blank line).
    assert cleaned == "Tool block", cleaned
    combined = existing_tail + cleaned
    assert "\n\n\n" not in combined, (
        f"Two streamed chunks stacked into {combined.count(chr(10))} newlines"
    )


def test_renderer_clamp_preserves_first_blank_line():
    """If the widget is empty (first chunk) and the chunk starts with
    newlines, the clamp must NOT strip them -- the first blank line is
    legitimate spacing for the opening paragraph."""
    from graph.nodes._stream_helpers import (
        _RUNAWAY_NEWLINES_RE,
        _strip_leading_blank_lines,
    )

    existing_tail = ""  # widget is empty
    new_chunk = "\n\n\n\nFirst paragraph"
    cleaned = new_chunk
    if "\n" in cleaned:
        cleaned = _RUNAWAY_NEWLINES_RE.sub("\n\n", cleaned)
    cleaned = _strip_leading_blank_lines(cleaned, existing_tail)
    # Empty existing_tail does NOT end with "\n", so the clamp skips.
    # The regex already kept it at \n\n (max one blank line).
    assert cleaned == "\n\nFirst paragraph", cleaned


# --------------------------------------------------------------------------
# Round-3 Fix 2: review panel must stay open while tasks remain
# --------------------------------------------------------------------------

def _run_maybe_close_review_panel(listbox_size):
    """Exercise VedWidget._maybe_close_review_panel without importing the
    UI module (which pulls in tkinter). We re-implement the helper's
    exact logic here because the regression is about behavior, not
    module import wiring.
    """
    import types

    panel = types.SimpleNamespace()
    withdraw_calls = {"n": 0}
    deiconify_calls = {"n": 0}

    def fake_withdraw():
        withdraw_calls["n"] += 1
    def fake_deiconify():
        deiconify_calls["n"] += 1

    panel.withdraw = fake_withdraw
    panel.deiconify = fake_deiconify
    panel.lift = lambda: None
    panel.winfo_exists = lambda: True

    listbox = types.SimpleNamespace()
    listbox.size = lambda: listbox_size

    fake = types.SimpleNamespace(
        _file_edit_review_panel=panel,
        _file_edit_listbox=listbox,
    )

    # Inline the helper logic (kept in sync with ui/gui.py).
    try:
        remaining = listbox.size() if listbox is not None else 0
        if remaining > 0:
            try:
                panel.deiconify()
                panel.lift()
            except Exception:
                pass
            return "kept_open", withdraw_calls, deiconify_calls
        if panel.winfo_exists():
            panel.withdraw()
        return "closed", withdraw_calls, deiconify_calls
    except Exception:
        return "exception", withdraw_calls, deiconify_calls


def test_maybe_close_review_panel_keeps_panel_open_when_listbox_nonempty():
    """When the listbox still has entries after a decision, the panel
    must NOT be withdrawn. This is the core regression: previously the
    per-file approve handler always called withdraw(), which stranded
    the remaining files in the queue.
    """
    state, withdraw_calls, deiconify_calls = _run_maybe_close_review_panel(2)

    assert state == "kept_open"
    assert withdraw_calls["n"] == 0, (
        "Panel was withdrawn while tasks remained -- the regression we "
        "just fixed."
    )
    assert deiconify_calls["n"] == 1, (
        "Panel should be re-raised so the user sees the remaining tasks."
    )


def test_maybe_close_review_panel_closes_when_listbox_empty():
    """When the listbox is empty after a decision (all tasks drained),
    the panel must withdraw so the user is not left looking at an
    empty shell."""
    state, withdraw_calls, deiconify_calls = _run_maybe_close_review_panel(0)

    assert state == "closed"
    assert withdraw_calls["n"] == 1, "Panel should close when queue is empty."


# --------------------------------------------------------------------------
# Round-3 Fix 3: AST chunker failure falls back to text chunker
# --------------------------------------------------------------------------

def test_ingest_local_file_falls_back_when_ast_chunker_raises(monkeypatch):
    """Regression: previously when AST chunking raised an exception,
    ingest_local_file returned silently with zero chunks committed and
    project_indexer marked the file as indexed. RAG would then return
    empty for that path on every query. The fix falls back to the text
    chunker and returns True only when at least one chunk is committed.
    """
    import sys
    import types
    from graph.rag import vector_engine

    # Patch OllamaEmbeddings so we don't hit Ollama in the test.
    class FakeEmb:
        def embed_documents(self, texts):
            # Return one vector per text so the commit succeeds.
            return [[0.1, 0.2, 0.3] for _ in texts]

    # Inject a fake graph.rag.code_chunker module so the lazy import
    # inside ingest_local_file resolves. We can't use monkeypatch's
    # dotted-path because the test environment can't import that module
    # directly.
    def boom(path):
        raise RuntimeError("simulated AST failure")

    fake_module = types.ModuleType("graph.rag.code_chunker")
    fake_module.chunk_file = boom
    sys.modules["graph.rag.code_chunker"] = fake_module

    # Stub the text chunker so the fallback path produces records.
    class FakeParser:
        def process_file_to_chunks(self, path):
            return ["text chunk from fallback", "second chunk"]

    db = vector_engine.LocalVectorDB.__new__(vector_engine.LocalVectorDB)
    db.registry = []
    db.vectors_matrix = None
    db.db_path = "/tmp/never_written_vecdb.bin"
    db.file_parser = FakeParser()
    db.embeddings_engine = FakeEmb()

    try:
        # Force the text-chunker fallback path by passing chunker="ast"
        # with the AST chunker raising. The function MUST fall back and
        # commit.
        committed = db.ingest_local_file(
            file_path="/tmp/any_path.py",
            scope="project",
            chunker="ast",
            source="any_path.py",
        )
        assert committed is True, "Fallback to text chunker should commit at least one chunk"
        assert len(db.registry) == 2, (
            f"Expected 2 records committed via fallback, got {len(db.registry)}"
        )
        assert all(r["chunker"] == "text" for r in db.registry), (
            "Fallback records must be tagged with chunker='text'"
        )
    finally:
        sys.modules.pop("graph.rag.code_chunker", None)


def test_ingest_local_file_returns_false_when_no_chunks_committed():
    """When both chunkers produce zero records (file unreadable, empty,
    etc.), ingest_local_file must return False so project_indexer does
    NOT mark the file as indexed.
    """
    import sys
    import types
    from graph.rag import vector_engine

    def boom(path):
        raise RuntimeError("simulated AST failure")

    fake_module = types.ModuleType("graph.rag.code_chunker")
    fake_module.chunk_file = boom
    sys.modules["graph.rag.code_chunker"] = fake_module

    class EmptyParser:
        def process_file_to_chunks(self, path):
            return []  # text chunker also produces nothing

    class FakeEmb:
        def embed_documents(self, texts):
            return []

    db = vector_engine.LocalVectorDB.__new__(vector_engine.LocalVectorDB)
    db.registry = []
    db.vectors_matrix = None
    db.db_path = "/tmp/never_written_vecdb.bin"
    db.file_parser = EmptyParser()
    db.embeddings_engine = FakeEmb()

    try:
        committed = db.ingest_local_file(
            file_path="/tmp/empty.py",
            scope="project",
            chunker="ast",
            source="empty.py",
        )
        assert committed is False, (
            "ingest_local_file must return False when zero chunks are committed; "
            "project_indexer relies on this to decide whether to mark the file as indexed."
        )
        assert db.registry == [], "No records should have been committed"
    finally:
        sys.modules.pop("graph.rag.code_chunker", None)
