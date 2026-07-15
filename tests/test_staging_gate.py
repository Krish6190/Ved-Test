"""Regression tests for the staging gate node that intercepts graph entry."""

from graph import staging_gate_node, _start_router, _route_after_staging_gate
from graph.state import VedState
from graph.tools.staging_registry import STAGING_REGISTRY
from langgraph.graph import END


def _make_state(**kwargs):
    defaults = {
        "messages": [], "route_intent": "", "mode": "coder",
        "active_thread_id": "thr_test",
        "dual_role_phase": "", "plan_executed": False,
    }
    defaults.update(kwargs)
    return VedState(**defaults)


def test_staging_gate_sets_terminal_flags_when_pending(tmp_path):
    tid = "thr_pending_test"
    STAGING_REGISTRY.register_session(tid)
    STAGING_REGISTRY.stage_edit(tid, "edit_file", str(tmp_path / "a.py"), {"old_text": "x", "new_text": "y"}, {"old": "", "new": ""})
    try:
        state = _make_state(active_thread_id=tid)
        result = staging_gate_node(state, get_llm=None, config=None)
        assert result["plan_executed"] is True
        assert result["dual_role_phase"] == "awaiting_user_approval"
    finally:
        STAGING_REGISTRY.unregister_session(tid)


def test_staging_gate_resets_when_no_pending():
    state = _make_state(active_thread_id="thr_empty")
    result = staging_gate_node(state, get_llm=None, config=None)
    assert result["plan_executed"] is False
    assert result["dual_role_phase"] == ""


def test_start_router_routes_through_gate_when_pending(tmp_path):
    tid = "thr_router_pending"
    STAGING_REGISTRY.register_session(tid)
    STAGING_REGISTRY.stage_edit(tid, "edit_file", str(tmp_path / "a.py"), {"old_text": "x", "new_text": "y"}, {"old": "", "new": ""})
    try:
        state = _make_state(active_thread_id=tid, mode="coder")
        assert _start_router(state) == "staging_gate_node"
    finally:
        STAGING_REGISTRY.unregister_session(tid)


def test_route_after_staging_gate_returns_end_when_terminal():
    state = _make_state(dual_role_phase="awaiting_user_approval")
    assert _route_after_staging_gate(state) == END


def test_route_after_staging_gate_returns_planner_when_clean():
    state = _make_state(mode="coder", dual_role_phase="")
    assert _route_after_staging_gate(state) == "planner_node"