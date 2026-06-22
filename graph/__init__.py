from langgraph.graph import StateGraph, START, END
from .state import VedState
from .nodes import intent_router_node, chat_node, essay_generator_node, python_tool_node

def build_graph(get_llm):
    """
    Build the Ved LangGraph with full 3-way routing support
    and secure hyperparameter-shifted placeholder connections.
    """
    g = StateGraph(VedState)
    g.add_node("intent_router_node", lambda state: intent_router_node(state, get_llm))
    g.add_node("chat_node", lambda state, config: chat_node(state, get_llm, config))
    g.add_node("essay_generator_node", essay_generator_node)
    g.add_node("python_tool_node", python_tool_node)
    
    # 2. Wire core entry point and conditional routing pathways
    g.add_edge(START, "intent_router_node")
    
    g.add_conditional_edges(
        "intent_router_node",
        lambda state: state.route_intent,  # Pydantic dot notation lookup
        {
            "A": "chat_node",
            "B": "essay_generator_node",
            "C": "python_tool_node"
        }
    )
    g.add_edge("chat_node", END)
    g.add_edge("essay_generator_node", END)
    g.add_edge("python_tool_node", END)
    
    return g.compile()
