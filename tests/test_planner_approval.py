"""Tests for the planner's plan-approval human-in-the-loop gate (Chunk 1).

Covers:
  - Plan approved: persists plan file, sets active_plan_id, routes to "P".
  - Plan rejected: no plan file written, returns user-facing message,
    route_intent == "A", active_plan_id is None.
  - Missing approval infrastructure: behaves as today (non-blocking).

These mirror the helper signature in `planner._wait_for_plan_approval`
and the `plan_approval_event` / `plan_approval_state` config keys.
"""
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

import data.plans as plan_store
from graph.nodes import planner as planner_mod
from graph.state import VedState


# ---- LLM + state mocks ----

class _FakeLLM:
    """A no-op planner LLM. `_stream_with_tool_loop` is monkey-patched away."""

    def bind_tools(self, tools):
        return self


class _FakeChunk:
    def __init__(self, content="", tool_call_chunks=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []


class _FakeState(VedState):
    """Minimal VedState stub — only the fields the planner actually reads."""

    def __init__(self, messages=None, mode="standard", route_intent="A",
                 active_plan_id=None, current_chunk_id=None):
        super().__init__(
            messages=list(messages or []),
            route_intent=route_intent,
            mode=mode,
            active_plan_id=active_plan_id,
            current_chunk_id=current_chunk_id,
        )


# ---- helpers ----

def _make_config(tmp_path, monkeypatch, *, plan_approval_event=None,
                 plan_approval_state=None, pre_set_event=False):
    """Build a planner config dict for tests.

    If `pre_set_event` is True and an event was supplied, we set the event
    before calling the node so `_wait_for_plan_approval` does not actually
    block. This keeps the tests synchronous.
    """
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    q = queue.Queue()
    configurable = {
        "token_queue": q,
        "planner_llm_factory": lambda mode: _FakeLLM(),
    }
    if plan_approval_event is not None:
        configurable["plan_approval_event"] = plan_approval_event
        configurable["plan_approval_state"] = plan_approval_state
        if pre_set_event and plan_approval_state is not None:
            plan_approval_event.set()
    return {"configurable": configurable}, q


def _planner_create_plan(monkeypatch):
    """Stub `_stream_with_tool_loop` so the planner emits CREATE_PLAN."""
    def fake_stream_with_tool_loop(llm, msgs, config, token_queue, max_rounds=3):
        ai_msg = AIMessage(content='CREATE_PLAN: ["step 1", "step 2"]')
        return ai_msg, [], []
    monkeypatch.setattr(planner_mod, "_stream_with_tool_loop", fake_stream_with_tool_loop)


def _planner_direct_answer(monkeypatch):
    """Stub the planner to emit DIRECT_ANSWER (no approval gate should fire)."""
    def fake_stream_with_tool_loop(llm, msgs, config, token_queue, max_rounds=3):
        ai_msg = AIMessage(content="DIRECT_ANSWER: no plan needed.")
        return ai_msg, [], []
    monkeypatch.setattr(planner_mod, "_stream_with_tool_loop", fake_stream_with_tool_loop)


def _drain_queue(q):
    """Drain token_queue so tests can assert on the events emitted."""
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


# ---- 1. Approved plan persists and routes to executor ----

def test_plan_approved_creates_plan_and_routes_to_executor(tmp_path, monkeypatch):
    event = threading.Event()
    state = {"value": True}
    cfg, q = _make_config(
        tmp_path, monkeypatch,
        plan_approval_event=event,
        plan_approval_state=state,
        pre_set_event=True,
    )
    _planner_create_plan(monkeypatch)

    state_obj = _FakeState(
        messages=[HumanMessage(content="refactor foo.py and run tests")],
        mode="standard",
    )

    result = planner_mod.planner_node(state_obj, lambda: _FakeLLM(), cfg)

    # State dict should be drained of the True so the next round starts clean.
    assert state["value"] is None, "approval_state['value'] must be reset to None"
    assert not event.is_set(), "plan_approval_event must be cleared after consumption"

    assert result["route_intent"] == "P", (
        f"Expected route_intent=='P' after approval; got {result.get('route_intent')!r}"
    )
    assert result.get("active_plan_id"), (
        "Expected active_plan_id to be set after approved plan; got None"
    )
    assert result.get("current_chunk_id") == 1

    # Plan file should exist on disk.
    plan_id = result["active_plan_id"]
    assert plan_store.load_plan(plan_id) is not None
    plan_file = tmp_path / f"{plan_id}.json"
    assert plan_file.exists(), f"Expected plan file at {plan_file}"

    # plan_approval_request event must have been emitted to the UI.
    events = _drain_queue(q)
    approval_events = [e for e in events
                       if isinstance(e, tuple) and e[0] == "plan_approval_request"]
    assert approval_events, "Expected plan_approval_request event on token_queue"
    payload = approval_events[0][1]
    assert payload["chunks"] == ["step 1", "step 2"]


# ---- 2. Rejected plan does not create a plan file ----

def test_plan_rejected_does_not_create_plan(tmp_path, monkeypatch):
    event = threading.Event()
    state = {"value": False}
    cfg, q = _make_config(
        tmp_path, monkeypatch,
        plan_approval_event=event,
        plan_approval_state=state,
        pre_set_event=True,
    )
    _planner_create_plan(monkeypatch)

    state_obj = _FakeState(
        messages=[HumanMessage(content="refactor foo.py and run tests")],
        mode="standard",
    )

    result = planner_mod.planner_node(state_obj, lambda: _FakeLLM(), cfg)

    # State dict should be drained of the False so the next round starts clean.
    assert state["value"] is None
    assert not event.is_set()

    # No plan file side effect.
    assert list(tmp_path.iterdir()) == [], (
        f"Expected no plan file written on rejection; found: "
        f"{[p.name for p in tmp_path.iterdir()]}"
    )

    # User-facing rejection message, routed back to the chat node.
    assert result["route_intent"] == "A", (
        f"Expected route_intent=='A' on rejection; got {result.get('route_intent')!r}"
    )
    assert result.get("active_plan_id") is None
    msgs = result.get("messages") or []
    assert msgs, "Expected a user-facing AIMessage on rejection"
    msg = msgs[0]
    assert isinstance(msg, AIMessage)
    msg_text = (msg.content or "").lower()
    assert "reject" in msg_text, f"Expected message to mention rejection; got: {msg.content!r}"
    assert "refine" in msg_text, f"Expected message to ask user to refine; got: {msg.content!r}"

    # plan_approval_request event still fired before blocking — UI must see it.
    events = _drain_queue(q)
    approval_events = [e for e in events
                       if isinstance(e, tuple) and e[0] == "plan_approval_request"]
    assert approval_events, "Expected plan_approval_request event before blocking"


# ---- 3. Missing approval infrastructure is non-blocking ----

def test_plan_approval_missing_infrastructure_is_non_blocking(tmp_path, monkeypatch):
    # No plan_approval_event / plan_approval_state keys in config.
    cfg, q = _make_config(tmp_path, monkeypatch)
    _planner_create_plan(monkeypatch)

    state_obj = _FakeState(
        messages=[HumanMessage(content="refactor foo.py and run tests")],
        mode="standard",
    )

    # Should NOT block — no event present.
    result = planner_mod.planner_node(state_obj, lambda: _FakeLLM(), cfg)

    # Existing behavior: CREATE_PLAN path persists a plan and routes to executor.
    assert result["route_intent"] == "P"
    assert result.get("active_plan_id")
    plan_id = result["active_plan_id"]
    assert plan_store.load_plan(plan_id) is not None

    # No plan_approval_request event was emitted (gate was skipped).
    events = _drain_queue(q)
    approval_events = [e for e in events
                       if isinstance(e, tuple) and e[0] == "plan_approval_request"]
    assert not approval_events, (
        "Did not expect plan_approval_request when approval infrastructure is absent"
    )


# ---- 4. Approval gate only fires for CREATE_PLAN, not DIRECT_ANSWER ----

def test_direct_answer_bypasses_plan_approval_gate(tmp_path, monkeypatch):
    """Approval infrastructure present but LLM emits DIRECT_ANSWER -> gate is
    skipped, plan file is not created, message is the direct answer."""
    event = threading.Event()
    state = {"value": None}
    cfg, q = _make_config(
        tmp_path, monkeypatch,
        plan_approval_event=event,
        plan_approval_state=state,
        pre_set_event=False,  # do NOT set — the gate must not block here
    )
    _planner_direct_answer(monkeypatch)

    state_obj = _FakeState(
        messages=[HumanMessage(content="what is 2+2?")],
        mode="standard",
    )

    result = planner_mod.planner_node(state_obj, lambda: _FakeLLM(), cfg)

    assert result["route_intent"] == "A"
    assert result.get("active_plan_id") is None
    msgs = result.get("messages") or []
    assert msgs and isinstance(msgs[0], AIMessage)
    assert "no plan needed" in (msgs[0].content or "").lower()

    # Event must remain unset because the gate was never invoked.
    assert not event.is_set()
    assert list(tmp_path.iterdir()) == [], "No plan file should be written"


# ---- 5. Helper resets event+state cleanly between rounds ----

def test_wait_for_plan_approval_clears_event_and_state():
    """Direct unit test of the helper: returns bool, clears event, resets state."""
    q = queue.Queue()
    event = threading.Event()
    state = {"value": True}
    event.set()

    approved = planner_mod._wait_for_plan_approval(
        q, ["chunk A"], event, state
    )
    assert approved is True
    assert state["value"] is None
    assert not event.is_set()

    # And the queued event carries the proposed chunks.
    kind, payload = q.get_nowait()
    assert kind == "plan_approval_request"
    assert payload == {"chunks": ["chunk A"]}
