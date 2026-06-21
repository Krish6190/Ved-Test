from langchain_core.messages import AIMessage

from .state import VedState


def intent_router_node(state: VedState) -> dict:
    """Route user intent to appropriate handler.
    
    For now, always routes to chat.
    Later: can detect os, whatsapp, tool requests.
    """
    return {
        "route_intent": "chat",
        "messages": state["messages"],
        "mode": state["mode"],
    }


def chat_node(state: VedState, get_llm) -> dict:
    """Conversational chat node.
    
    Receives state and LLM getter, runs inference, returns response.
    """
    if state["mode"] == "hibernate":
        return {
            "messages": [AIMessage(content="Ved is hibernating. Switch to turbo or standard mode first.")],
            "route_intent": state["route_intent"],
            "mode": state["mode"],
        }

    llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="No local model is available. Start Ollama and set OLLAMA_BASE_URL.")],
            "route_intent": state["route_intent"],
            "mode": state["mode"],
        }

    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "route_intent": state["route_intent"],
        "mode": state["mode"],
    }
