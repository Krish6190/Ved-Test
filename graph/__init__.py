from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from .state import VedState
from .nodes import intent_router_node, chat_node, coder_chat_node
from .nodes.standalone_chat import standalone_chat_node
from .nodes.planner import planner_node
from .nodes.executor import executor_node
from .content_generation.pipeline_node import content_pipeline_node
from graph.tools import VED_TOOLS


def _route_after_llm(state: VedState) -> str:
    """Conditional edge after `chat_node` / `coder_chat_node` / `executor_node`.

    If the LLM's last message contains `tool_calls`, hand off to the
    `tools` node (ToolNode) which executes them and returns a ToolMessage.
    Otherwise end the turn (or loop back to planner for the executor).
    """
    last = state.messages[-1] if state.messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def _route_after_tools(state: VedState) -> str:
    """After ToolNode executes, loop back to the originating LLM node."""
    return "coder_chat_node" if state.mode == "coder" else "chat_node"


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


def _route_after_intent(state: VedState) -> str:
    """Conditional edge after `intent_router_node`.

    Routes Path A differently based on mode:
      - non-coder Path A → standalone_chat_node (simple chatbot with bound
        tools; no planner-executor split). Used for 8B/14B models that can
        handle tool use directly.
      - coder Path A → planner_node (full planner+executor flow for the
        structured 7B+3B split).

    Path B (content generation) is unchanged.
    """
    intent = getattr(state, "route_intent", "")
    if intent == "A":
        return "standalone_chat_node" if state.mode != "coder" else "planner_node"
    if intent == "B":
        return "content_pipeline_node"
    return END


def build_graph(get_llm):
    g = StateGraph(VedState)
    g.add_node("intent_router_node", lambda state, config: intent_router_node(state, get_llm))
    g.add_node("standalone_chat_node", lambda state, config: standalone_chat_node(state, get_llm, config))
    g.add_node("planner_node", lambda state, config: planner_node(state, get_llm, config))
    g.add_node("executor_node", lambda state, config: executor_node(state, get_llm, config))
    g.add_node("chat_node", lambda state, config: chat_node(state, get_llm, config))
    g.add_node("content_pipeline_node", lambda state, config: content_pipeline_node(state, get_llm, config))
    g.add_node("coder_chat_node", lambda state, config: coder_chat_node(state, get_llm, config))
    # ToolNode executes any tool_calls emitted by the bound LLM.
    g.add_node("tools", ToolNode(VED_TOOLS))

    g.add_conditional_edges(
        START,
        lambda state: "planner_node" if state.mode == "coder" else "intent_router_node"
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
    # coder_chat_node is kept for backwards compatibility but is no longer
    # in the active routing path (all modes now go through intent_router,
    # which sends coder Path A → planner → executor loop). Left in the
    # graph in case future code wants to invoke it directly.

    # Planner decides: send to executor (plan), or end (direct/finalize).
    g.add_conditional_edges(
        "planner_node",
        _route_after_planner,
        {"executor_node": "executor_node", END: END},
    )
    # Executor runs one chunk as a self-contained agent loop (tools
    # executed inline). It always returns to the planner — the planner
    # reads the chunk output + structured tool_calls from the plan file.
    g.add_edge("executor_node", "planner_node")

    # coder_chat_node: LLM emits tool_calls or ends. (No more /run -> C; that
    # path is gone. /run and "execute ..." now flow through Path A.)
    g.add_conditional_edges(
        "coder_chat_node",
        _route_after_llm,
        {"tools": "tools", END: END},
    )

    # chat_node: route to ToolNode if the LLM emitted tool_calls, else END.
    g.add_conditional_edges(
        "chat_node",
        _route_after_llm,
        {"tools": "tools", END: END}
    )
    # ToolNode loops back to the originating LLM.
    g.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"chat_node": "chat_node", "coder_chat_node": "coder_chat_node"}
    )
    g.add_edge("content_pipeline_node", END)
    return g.compile()
