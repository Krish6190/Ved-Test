from langgraph.graph import StateGraph, START, END
from .state import VedState
from .nodes import intent_router_node, chat_node, python_tool_node, coder_chat_node
from .content_generation.pipeline_node import content_pipeline_node

def build_graph(get_llm):
    g = StateGraph(VedState)
    g.add_node("intent_router_node", lambda state, config: intent_router_node(state, get_llm))
    g.add_node("chat_node", lambda state, config: chat_node(state, get_llm, config))
    g.add_node("content_pipeline_node", lambda state, config: content_pipeline_node(state, get_llm, config))
    g.add_node("python_tool_node", python_tool_node)
    g.add_node("coder_chat_node", lambda state, config: coder_chat_node(state, get_llm, config))
    g.add_conditional_edges(
        START,
        lambda state: "coder_chat_node" if state.mode == "coder" else "intent_router_node"
    )
    g.add_conditional_edges(
        "intent_router_node",
        lambda state: state.route_intent,
        {"A": "chat_node", "B": "content_pipeline_node", "C": "python_tool_node"}
    )
    g.add_conditional_edges(
        "coder_chat_node",
        lambda state: "python_tool_node" if state.route_intent == "C" else END
    )
    g.add_edge("chat_node", END)
    g.add_edge("content_pipeline_node", END)
    g.add_edge("python_tool_node", END)
    return g.compile()