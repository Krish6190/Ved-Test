"""Tests for the executor's agent-loop behavior.

Covers:
  - Tool calls are recorded with structured {name, args, result, ok} format
  - Agent loop terminates when LLM stops emitting tool_calls
  - Agent loop terminates at MAX_ITERATIONS even if LLM keeps emitting tool_calls
  - Tool result truncation prevents chunk bloat
  - Standard mode correctly restricts to PATH_A_EXECUTOR_TOOLS
"""
import queue
import threading
import data.plans as plan_store
import graph.nodes.executor as executor_mod


# ---- helpers ----

class _FakeLLM:
    def bind_tools(self, tools):
        return self


class _FakeState:
    def __init__(self, plan_id, chunk_id, mode="standard"):
        self.active_plan_id = plan_id
        self.current_chunk_id = chunk_id
        self.mode = mode
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
    plan = plan_store.make_blank_plan("task", chunks or ["do thing"])
    plan_store.save_plan(plan)
    return plan


# ---- structured tool_calls recording ----

def test_successful_tool_call_is_recorded_with_structured_fields(tmp_path, monkeypatch):
    """When a tool succeeds, its {name, args, result, ok} are in chunk.tool_calls."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    iter_count = [0]

    def _stream(*a, **kw):
        iter_count[0] += 1
        # Iteration 1: emit one tool call. Iteration 2: LLM is done.
        if iter_count[0] == 1:
            return ("reading", [{
                "id": "c1", "name": "read_file", "args": {"path": "foo.py"},
            }])
        return ("all done", [])

    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)
    monkeypatch.setattr(
        executor_mod, "_invoke_tool_sync",
        lambda tool, args: ("def foo(): pass", True),
    )

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(),
        config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "done"
    assert len(chunk1["tool_calls"]) == 1
    tc = chunk1["tool_calls"][0]
    assert tc["name"] == "read_file"
    assert tc["args"] == {"path": "foo.py"}
    assert tc["ok"] is True
    assert "def foo(): pass" in tc["result"]


def test_failed_tool_call_records_error_in_tool_calls(tmp_path, monkeypatch):
    """When a tool fails, ok=False and error contains the message."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    state = _FakeState(plan["plan_id"], 1)
    state.mode = "coder"
    monkeypatch.setattr(
        executor_mod, "_stream_one_iteration",
        lambda *a, **kw: ("trying", [{"id": "c1", "name": "execute_python", "args": {}}]),
    )
    monkeypatch.setattr(
        executor_mod, "_invoke_tool_sync",
        lambda *a, **kw: ("ERROR: NameError: x", False),
    )

    executor_mod.executor_node(state, get_llm=lambda: _FakeLLM(), config=cfg)

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "pending"
    assert chunk1["tool_calls"][0]["ok"] is False
    assert "NameError" in chunk1["tool_calls"][0]["error"]


# ---- agent loop termination ----

def test_agent_loop_terminates_when_no_tool_calls(tmp_path, monkeypatch):
    """If LLM emits no tool_calls, the loop ends immediately and mark_done runs."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    iter_count = [0]

    def _stream_no_tools(*a, **kw):
        iter_count[0] += 1
        return ("all done, no tools needed", [])

    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream_no_tools)
    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    assert iter_count[0] == 1, "should have streamed exactly once"
    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "done"


def test_agent_loop_terminates_at_max_iterations(tmp_path, monkeypatch):
    """If LLM keeps emitting tool_calls forever, the loop stops at MAX_ITERATIONS."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    iter_count = [0]

    def _stream_forever(*a, **kw):
        iter_count[0] += 1
        return ("looping", [{
            "id": f"c{iter_count[0]}", "name": "read_file", "args": {"path": "x"},
        }])

    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream_forever)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync",
                        lambda *a, **kw: ("content", True))

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    # MAX_AGENT_ITERATIONS is 8. Each iteration may call stream + tools once.
    assert iter_count[0] <= executor_mod._MAX_AGENT_ITERATIONS
    assert iter_count[0] >= 1
    # Chunk should be marked done (the last iteration had no tool_calls error).
    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "done"


def test_agent_loop_continues_through_multiple_successful_tool_calls(tmp_path, monkeypatch):
    """LLM emits 2 tool_calls in iteration 1, then no tools in iteration 2."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    iter_count = [0]
    def _stream(*a, **kw):
        iter_count[0] += 1
        if iter_count[0] == 1:
            return ("two tools", [
                {"id": "c1", "name": "read_file", "args": {"path": "a"}},
                {"id": "c2", "name": "read_file", "args": {"path": "b"}},
            ])
        return ("done", [])

    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)
    monkeypatch.setattr(executor_mod, "_invoke_tool_sync",
                        lambda *a, **kw: ("contents", True))

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "done"
    # Both tools were executed and logged.
    assert len(chunk1["tool_calls"]) == 2
    assert chunk1["tool_calls"][0]["args"] == {"path": "a"}
    assert chunk1["tool_calls"][1]["args"] == {"path": "b"}


# ---- tool set per mode ----

def test_executor_uses_path_a_tools_in_standard_mode(tmp_path, monkeypatch):
    """In standard mode, the executor binds PATH_A_EXECUTOR_TOOLS (no edit/overwrite/execute_python)."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    captured_tools = []

    class _CapturingLLM:
        def bind_tools(self, tools):
            captured_tools.extend(t.name for t in tools)
            return self

    def _stream(*a, **kw):
        return ("ok", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1, mode="standard"),
        get_llm=lambda: _CapturingLLM(), config=cfg,
    )

    # Standard mode: read-only + scripts/apps/thread RAG. Coding tools
    # (edit_file, overwrite_file, propose_tool) are still forbidden.
    assert "read_file" in captured_tools
    assert "execute_python" in captured_tools   # Path A can run scripts
    assert "edit_file" not in captured_tools
    assert "overwrite_file" not in captured_tools


def test_executor_uses_full_tools_in_coder_mode(tmp_path, monkeypatch):
    """In coder mode, the executor binds full VED_TOOLS."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    captured_tools = []

    class _CapturingLLM:
        def bind_tools(self, tools):
            captured_tools.extend(t.name for t in tools)
            return self

    def _stream(*a, **kw):
        return ("ok", [])
    monkeypatch.setattr(executor_mod, "_stream_one_iteration", _stream)

    state = _FakeState(plan["plan_id"], 1, mode="coder")
    executor_mod.executor_node(state, get_llm=lambda: _CapturingLLM(), config=cfg)

    # Coder mode: full tool set including the coding tools.
    assert "read_file" in captured_tools
    assert "execute_python" in captured_tools
    assert "edit_file" in captured_tools
    assert "propose_tool" in captured_tools


# ---- tool result truncation ----

def test_tool_result_is_truncated_when_very_long(tmp_path, monkeypatch):
    """A tool result longer than _TOOL_RESULT_MAX_CHARS gets truncated."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    long_result = "x" * (executor_mod._TOOL_RESULT_MAX_CHARS + 500)

    monkeypatch.setattr(
        executor_mod, "_stream_one_iteration",
        lambda *a, **kw: ("reading", [{
            "id": "c1", "name": "read_file", "args": {"path": "huge.txt"},
        }]),
    )
    monkeypatch.setattr(
        executor_mod, "_invoke_tool_sync",
        lambda *a, **kw: (long_result, True),
    )

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    tc = chunk1["tool_calls"][0]
    # Result is truncated with marker.
    assert "truncated" in tc["result"]
    assert len(tc["result"]) <= executor_mod._TOOL_RESULT_MAX_CHARS + 100  # marker room


# ---- empty chunks / missing plan ----

def test_executor_handles_unknown_tool_gracefully(tmp_path, monkeypatch):
    """If the LLM emits a tool that doesn't exist in tool_map, mark_failed."""
    plan = _setup_plan(tmp_path, monkeypatch)
    cfg, _ = _make_config()

    monkeypatch.setattr(
        executor_mod, "_stream_one_iteration",
        lambda *a, **kw: ("trying", [{
            "id": "c1", "name": "nonexistent_tool", "args": {},
        }]),
    )

    executor_mod.executor_node(
        _FakeState(plan["plan_id"], 1),
        get_llm=lambda: _FakeLLM(), config=cfg,
    )

    updated = plan_store.load_plan(plan["plan_id"])
    chunk1 = next(c for c in updated["chunks"] if c["id"] == 1)
    assert chunk1["status"] == "pending"
    assert "unknown tool" in chunk1["output"]
    assert chunk1["tool_calls"][0]["name"] == "nonexistent_tool"
    assert chunk1["tool_calls"][0]["ok"] is False
