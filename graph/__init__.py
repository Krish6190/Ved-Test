from langgraph.graph import StateGraph, START, END

from .state import VedState
from .nodes import intent_router_node, chat_node


def build_graph(get_llm):
    """Build the Ved LangGraph with intent routing and chat node.
    
    Args:
        get_llm: callable that returns the current LLM instance
    
    Returns:
        Compiled LangGraph ready to invoke
    """
    # Create a wrapper for chat_node that binds get_llm
    def chat_node_wrapper(state: VedState) -> dict:
        return chat_node(state, get_llm)

    g = StateGraph(VedState)
    
    # Add nodes
    g.add_node("intent_router_node", intent_router_node)
    g.add_node("chat_node", chat_node_wrapper)
    
    # Wire edges
    g.add_edge(START, "intent_router_node")
    g.add_edge("intent_router_node", "chat_node")
    g.add_edge("chat_node", END)
    
    return g.compile()
