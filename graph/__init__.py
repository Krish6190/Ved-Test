from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from .state import VedState
from .nodes import intent_router_node, chat_node, python_tool_node, coder_chat_node
from .content_generation.pipeline_node import content_pipeline_node
from graph.tools import VED_TOOLS


def _route_after_llm(state: VedState) -> str:
    """Conditional edge after `chat_node` / `coder_chat_node`.

    If the LLM's last message contains `tool_calls`, hand off to the
    `tools` node (ToolNode) which executes them and returns a ToolMessage.
    Otherwise end the turn.
    """
    last = state.messages[-1] if state.messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def _route_after_tools(state: VedState) -> str:
    """After ToolNode executes, loop back to the same LLM node that called
    the tools (chat_node for Path A, coder_chat_node for the coder mode)."""
    return "coder_chat_node" if state.mode == "coder" else "chat_node"


def build_graph(get_llm):
    g = StateGraph(VedState)
    g.add_node("intent_router_node", lambda state, config: intent_router_node(state, get_llm))
    g.add_node("chat_node", lambda state, config: chat_node(state, get_llm, config))
    g.add_node("content_pipeline_node", lambda state, config: content_pipeline_node(state, get_llm, config))
    g.add_node("python_tool_node", python_tool_node)
    g.add_node("coder_chat_node", lambda state, config: coder_chat_node(state, get_llm, config))
    # ToolNode executes any tool_calls emitted by the bound LLM.
    g.add_node("tools", ToolNode(VED_TOOLS))

    g.add_conditional_edges(
        START,
        lambda state: "coder_chat_node" if state.mode == "coder" else "intent_router_node"
    )
    g.add_conditional_edges(
        "intent_router_node",
        lambda state: state.route_intent,
        {"A": "chat_node", "B": "content_pipeline_node", "C": "python_tool_node"}
    )
    # coder_chat_node: explicit "/run ..." still goes to python_tool_node;
    # otherwise the LLM decides what to do (including calling tools).
    g.add_conditional_edges(
        "coder_chat_node",
        lambda state: (
            "python_tool_node"
            if state.route_intent == "C"
            else _route_after_llm(state)
        ),
        {"python_tool_node": "python_tool_node", "tools": "tools", END: END}
    )
    g.add_conditional_edges(
        "python_tool_node",
        lambda state: "coder_chat_node" if state.mode == "coder" else "chat_node"
    )
    # chat_node: route to ToolNode if the LLM emitted tool_calls, else END.
    g.add_conditional_edges(
        "chat_node",
        _route_after_llm,
        {"tools": "tools", END: END}
    )
    # ToolNode loops back to the LLM that called it (so it sees the result).
    g.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"chat_node": "chat_node", "coder_chat_node": "coder_chat_node"}
    )
    g.add_edge("content_pipeline_node", END)
    return g.compile()
