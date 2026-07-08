"""Tests for the new `needs_planning` routing logic in the intent router.

Chunk 5 of the Ved plan introduces a `needs_planning` flag on the router's
output and changes the `_route_after_intent` conditional edge so that:

  - Path A in `standard` mode + needs_planning=False -> standalone_chat_node
  - Path A in `standard` mode + needs_planning=True  -> planner_node
  - Path A in `coder` mode                          -> planner_node (always)
  - Path B                                           -> content_pipeline_node
"""
from langchain_core.messages import HumanMessage, SystemMessage

from graph.nodes.intent_router import intent_router_node
from graph.state import VedState
from graph import _route_after_intent


# ----- intent_router_node: needs_planning flag -----

def test_simple_greeting_sets_no_planning():
    state = VedState(
        messages=[
            SystemMessage(content="sys"),
            HumanMessage(content="hello"),
        ],
        mode="standard",
    )
    result = intent_router_node(state, lambda: None)
    assert result["route_intent"] == "A"
    assert result["needs_planning"] is False


def test_complex_request_sets_planning():
    state = VedState(
        messages=[HumanMessage(content="implement a todo app")],
        mode="standard",
    )
    result = intent_router_node(state, lambda: None)
    assert result["route_intent"] == "A"
    assert result["needs_planning"] is True


def test_coder_ignores_planning_flag():
    state = VedState(
        messages=[HumanMessage(content="hello")],
        mode="coder",
    )
    result = intent_router_node(state, lambda: None)
    assert result["route_intent"] == "A"
    # In coder mode the flag is intentionally not populated / not consulted;
    # the value is allowed to be either True or False, so don't assert it.


# ----- _route_after_intent conditional edge -----

def test_route_after_intent_simple_standard_goes_standalone():
    state = VedState(
        messages=[HumanMessage(content="hello")],
        mode="standard",
        route_intent="A",
        needs_planning=False,
    )
    assert _route_after_intent(state) == "standalone_chat_node"


def test_route_after_intent_complex_standard_goes_planner():
    state = VedState(
        messages=[HumanMessage(content="implement a todo app")],
        mode="standard",
        route_intent="A",
        needs_planning=True,
    )
    assert _route_after_intent(state) == "planner_node"


def test_route_after_intent_coder_goes_planner():
    state = VedState(
        messages=[HumanMessage(content="hello")],
        mode="coder",
        route_intent="A",
        needs_planning=False,
    )
    assert _route_after_intent(state) == "planner_node"


def test_route_after_intent_content_gen_path_B():
    state = VedState(
        messages=[HumanMessage(content="write me an essay about cats")],
        mode="standard",
        route_intent="B",
    )
    assert _route_after_intent(state) == "content_pipeline_node"
