"""Tests for the planner/executor transition lock (one-planner guard)."""
from langchain_core.messages import HumanMessage
from graph.state import VedState
from graph.nodes import planner as planner_mod
from graph.nodes.planner import planner_node, _build_planner_system_prompt
from graph.nodes.executor import executor_node
import data.plans as plan_store


class _FakeChunk:
    def __init__(self, content=""):
        self.content = content
        self.tool_call_chunks = []


class _FakeBoundLLM:
    def __init__(self, content=""):
        self._content = content

    def stream(self, _messages):
        yield _FakeChunk(self._content)


class _FakeLLM:
    def bind_tools(self, _tools):
        return _FakeBoundLLM("done")


def _make_state(**kwargs) -> VedState:
    defaults = {
        "messages": [],
        "route_intent": "P",
        "mode": "standard",
        "active_thread_id": "thr_test",
    }
    defaults.update(kwargs)
    return VedState(**defaults)


def test_state_has_lock_fields():
    s = _make_state(current_step=1, last_step_status="dispatched")
    assert s.current_step == 1
    assert s.last_step_status == "dispatched"


def test_planner_passes_through_when_chunk_dispatched():
    """If current_step is dispatched, planner must not invoke the LLM."""
    s = _make_state(
        active_plan_id="abc123",
        current_chunk_id=1,
        current_step=1,
        last_step_status="dispatched",
    )
    result = planner_node(s, lambda: None, None)
    assert result["route_intent"] == "P"
    assert result["current_step"] == 1
    assert result["last_step_status"] == "dispatched"


def test_planner_clears_lock_when_summary_emitted():
    s = _make_state(
        active_plan_id="abc123",
        current_chunk_id=1,
        current_step=1,
        last_step_status="done",
        summary_emitted=True,
    )
    result = planner_node(s, lambda: None, None)
    assert result["route_intent"] == "A"
    assert result.get("current_step") is None
    assert result.get("last_step_status") == ""


def test_executor_idempotent_when_chunk_not_executing(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b"])
    plan_store.mark_executing(plan, 1)
    plan_store.mark_done(plan, 1, "done output")
    plan_store.save_plan(plan)

    s = _make_state(
        active_plan_id=plan["plan_id"],
        current_chunk_id=1,
        current_step=1,
        last_step_status="dispatched",  # stale dispatch; chunk is actually done
    )
    result = executor_node(s, lambda: None, None)
    assert result["route_intent"] == "P"
    assert result["messages"] == []


def test_executor_returns_terminal_when_summary_emitted():
    s = _make_state(
        active_plan_id="abc123",
        current_chunk_id=1,
        current_step=1,
        summary_emitted=True,
    )
    result = executor_node(s, lambda: None, None)
    assert result["route_intent"] == "A"
    assert result["active_plan_id"] is None
    assert result["current_step"] is None
    assert result["last_step_status"] == ""


def test_planner_system_prompt_omits_recommend_in_coder_mode():
    prompt = _build_planner_system_prompt("coder")
    assert "RECOMMEND_CODER_MODE" not in prompt


def test_planner_system_prompt_includes_recommend_in_standard_mode():
    prompt = _build_planner_system_prompt("standard")
    assert "RECOMMEND_CODER_MODE" in prompt


def test_planner_system_prompt_includes_recommend_in_turbo_mode():
    prompt = _build_planner_system_prompt("turbo")
    assert "RECOMMEND_CODER_MODE" in prompt


def test_lock_progression_across_two_chunks(tmp_path, monkeypatch):
    """Executor advances current_step to the next chunk after marking the first done."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)

    text = 'CREATE_PLAN: ["chunk one", "chunk two"]'
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "create_plan"

    plan = plan_store.make_blank_plan("task", payload)
    first = plan["chunks"][0]
    second = plan["chunks"][1]
    plan_store.mark_executing(plan, first["id"])
    plan_store.save_plan(plan)

    state = _make_state(
        active_plan_id=plan["plan_id"],
        current_chunk_id=first["id"],
        current_step=first["id"],
        last_step_status="dispatched",
    )

    def _fake_get(mode=None):
        return _FakeLLM()
    result = executor_node(state, _fake_get, None)
    assert result["route_intent"] == "P"
    assert result["current_step"] == second["id"]
    assert result["last_step_status"] == "dispatched"

    reloaded = plan_store.load_plan(plan["plan_id"])
    assert reloaded["chunks"][0]["status"] == "done"
    assert reloaded["chunks"][1]["status"] == "executing"
