from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

from .state import VedState
from .nodes import intent_router_node
from .nodes.standalone_chat import standalone_chat_node
from .nodes.simple_chat import simple_chat_node
from .nodes.planner import planner_node
from .nodes.executor import executor_node
from .content_generation.pipeline_node import content_pipeline_node
from .tools.staging_registry import STAGING_REGISTRY


def hibernate_node(state: VedState, get_llm, config) -> dict:
    return {
        "messages": [AIMessage(content="Ved is hibernating. Type /wake to resume.")],
        "route_intent": state.route_intent, "mode": state.mode,
    }


def staging_gate_node(state: VedState, get_llm, config) -> dict:
    """First node after START. Inspects STAGING_REGISTRY and decides:
      - If pending edits exist for the active thread: set terminal flags
        (plan_executed=True, dual_role_phase="awaiting_user_approval") so the
        graph terminates and the UI re-renders the review window. The
        Planner is never invoked.
      - If no pending edits: reset the flags so the Planner (if reached)
        behaves normally.
    """
    thread_id = getattr(state, "active_thread_id", "") or ""
    pending = False
    if thread_id and STAGING_REGISTRY.has_session(thread_id):
        pending = bool(STAGING_REGISTRY.get_tasks(thread_id))

    if pending:
        return {
            "plan_executed": True,
            "dual_role_phase": "awaiting_user_approval",
            "route_intent": "A",
        }
    return {
        "plan_executed": False,
        "dual_role_phase": "",
    }


def _route_after_planner(state: VedState) -> str:
    """Conditional edge after `planner_node`. Reads state.route_intent
    set by the planner:
      - "P"        -> executor (plan just created or chunk ready)
      - "A"        -> END (direct answer / plan complete with summary emitted)
      - "B"        -> END (planner emitted a fallback that we route elsewhere)
    """
    intent = getattr(state, "route_intent", "")
    if intent == "P":
        return "executor_node"
    return END


def _start_router(state: VedState) -> str:
    """Route from START. If pending staged edits exist, route through
    staging_gate_node to force terminal pause; otherwise route normally
    based on mode."""
    thread_id = getattr(state, "active_thread_id", "") or ""
    if thread_id and STAGING_REGISTRY.has_session(thread_id):
        if STAGING_REGISTRY.get_tasks(thread_id):
            return "staging_gate_node"
    return {
        "hibernate": "hibernate_node",
        "standard": "simple_chat_node",
        "turbo": "intent_router_node",
        "coder": "planner_node",
    }.get(state.mode, "simple_chat_node")


def _route_after_staging_gate(state: VedState) -> str:
    """Route after staging_gate_node. If terminal flags were set, go to END;
    otherwise route to the normal mode-based destination."""
    if getattr(state, "dual_role_phase", "") == "awaiting_user_approval":
        return END
    mode = getattr(state, "mode", "standard")
    return {
        "hibernate": "hibernate_node",
        "standard": "simple_chat_node",
        "turbo": "intent_router_node",
        "coder": "planner_node",
    }.get(mode, "simple_chat_node")


def _route_after_executor(state: VedState) -> str:
    """Conditional edge after `executor_node`.

    Routes:
      - END when the executor has staged edits awaiting user approval
        (`plan_executed` True or `dual_role_phase` == "awaiting_user_approval").
      - planner_node when a plan is still active and no terminal approval
        state is set.
      - END otherwise.
    """
    if getattr(state, "plan_executed", False):
        return END
    if getattr(state, "dual_role_phase", "") == "awaiting_user_approval":
        return END
    if getattr(state, "route_intent", "") == "P" and getattr(state, "active_plan_id", None):
        return "planner_node"
    return END


def _route_after_intent(state: VedState) -> str:
    """Conditional edge after `intent_router_node`.

    Routes Path A differently based on mode and complexity:
      - coder Path A                 → planner_node (always; the coder
        pipeline is built around planner+executor).
      - non-coder Path A, complex    → planner_node (set when
        `state.needs_planning` is True; complex multi-step / long
        tool-triggering requests get the full planner-executor pipeline
        even on the simpler 8B/14B models).
      - non-coder Path A, simple     → standalone_chat_node (chatbot with
        bound tools; no planner-executor split). Used for short, casual
        requests that don't need a plan.

    Path B (content generation) is unchanged.
    """
    intent = getattr(state, "route_intent", "")
    if intent == "A":
        if state.mode == "coder":
            return "planner_node"
        if getattr(state, "needs_planning", False):
            return "planner_node"
        return "standalone_chat_node"
    if intent == "B":
        return "content_pipeline_node"
    return END


def build_graph(get_llm):
    g = StateGraph(VedState)
    g.add_node("staging_gate_node", lambda state, config: staging_gate_node(state, get_llm, config))
    g.add_node("intent_router_node", lambda state, config: intent_router_node(state, get_llm))
    g.add_node("standalone_chat_node", lambda state, config: standalone_chat_node(state, get_llm, config))
    g.add_node("simple_chat_node", lambda state, config: simple_chat_node(state, get_llm, config))
    g.add_node("planner_node", lambda state, config: planner_node(state, get_llm, config))
    g.add_node("executor_node", lambda state, config: executor_node(state, get_llm, config))
    g.add_node("content_pipeline_node", lambda state, config: content_pipeline_node(state, get_llm, config))
    g.add_node("hibernate_node", lambda state, config: hibernate_node(state, get_llm, config))

    g.add_conditional_edges(
        START,
        _start_router,
        {
            "staging_gate_node": "staging_gate_node",
            "hibernate_node": "hibernate_node",
            "simple_chat_node": "simple_chat_node",
            "intent_router_node": "intent_router_node",
            "planner_node": "planner_node",
        },
    )
    g.add_conditional_edges(
        "staging_gate_node",
        _route_after_staging_gate,
        {
            "hibernate_node": "hibernate_node",
            "simple_chat_node": "simple_chat_node",
            "intent_router_node": "intent_router_node",
            "planner_node": "planner_node",
            END: END,
        },
    )
    # Route Path A based on mode: non-coder skips the planner, coder keeps it.
    g.add_conditional_edges(
        "intent_router_node",
        _route_after_intent,
        {
            "standalone_chat_node": "standalone_chat_node",
            "planner_node": "planner_node",
            "content_pipeline_node": "content_pipeline_node",
            END: END,
        },
    )
    # standalone_chat_node is terminal — it returns the final response directly.
    g.add_edge("standalone_chat_node", END)
    # simple_chat_node (standard mode) is terminal — no tools, no planner.
    g.add_edge("simple_chat_node", END)
    # hibernate_node is terminal — no model, no work.
    g.add_edge("hibernate_node", END)

    # Planner decides: send to executor (plan), or end (direct/finalize).
    g.add_conditional_edges(
        "planner_node",
        _route_after_planner,
        {"executor_node": "executor_node", END: END},
    )
    # Executor runs one chunk as a self-contained agent loop (tools
    # executed inline). It returns to the planner only when a plan is still
    # active and awaiting the next chunk. Otherwise the session ends.
    g.add_conditional_edges(
        "executor_node",
        _route_after_executor,
        {"planner_node": "planner_node", END: END},
    )

    g.add_edge("content_pipeline_node", END)
    return g.compile()
