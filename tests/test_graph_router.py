"""Regression tests for the conditional router after the executor node.

The structural state-machine fix introduces two terminal conditions:
  - `plan_executed` is True (executor staged edits and is waiting).
  - `dual_role_phase == "awaiting_user_approval"`.

When either is true, the graph must terminate (`END`) instead of looping
back to `planner_node`. Otherwise the planner-executor graph ping-pongs
forever after a file edit is staged.
"""
from graph import _route_after_executor
from graph.state import VedState


def _make_state(**kwargs):
    defaults = {
        "messages": [], "route_intent": "P", "mode": "coder",
        "active_plan_id": "abc123", "dual_role_phase": "",
        "plan_executed": False,
    }
    defaults.update(kwargs)
    return VedState(**defaults)


def test_route_after_executor_returns_end_when_plan_executed():
    state = _make_state(plan_executed=True)
    assert _route_after_executor(state) == "__end__"


def test_route_after_executor_returns_end_when_awaiting_approval():
    state = _make_state(dual_role_phase="awaiting_user_approval")
    assert _route_after_executor(state) == "__end__"


def test_route_after_executor_returns_planner_when_plan_active():
    state = _make_state(route_intent="P", active_plan_id="abc123")
    assert _route_after_executor(state) == "planner_node"


def test_route_after_executor_returns_end_when_no_active_plan():
    state = _make_state(route_intent="P", active_plan_id=None)
    assert _route_after_executor(state) == "__end__"
