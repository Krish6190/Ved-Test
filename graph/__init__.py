from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import VedState
from .nodes import intent_router_node, chat_node
from .tools import all_ved_tools

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
    
    g.add_node("intent_router_node", intent_router_node)
    g.add_node("chat_node", chat_node_wrapper)
    g.add_node("tools", ToolNode(all_ved_tools)) # Built-in tool runner
    
    g.add_edge(START, "intent_router_node")
    g.add_edge("intent_router_node", "chat_node")
    
    # Directs to tools node if Llama asks for it, otherwise goes to END
    g.add_conditional_edges("chat_node", tools_condition)
    g.add_edge("tools", "chat_node")
    
    return g.compile()
