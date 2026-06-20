from langchain_core.messages import AIMessage, SystemMessage
from .tools import all_ved_tools
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
            "messages": [AIMessage(content="Ved is hibernating. Switch to turbo or standard mode first.\nTo switch modes, use the command: /mode turbo or /mode standard.\n")],
            "route_intent": state["route_intent"],
            "mode": state["mode"],
        }

    llm = get_llm()
    if llm is None:
        return {"messages": [AIMessage(content="No local model is available.")]}

    # Bind the combined tools list directly to the model
    llm_with_tools = llm.bind_tools(all_ved_tools)
    
    # Enforce categorization rules to stop hallucinated tool usage
    guardrail = SystemMessage(content=(
        "You are an AI agent named Ved. You have access to tools to open apps and websites. "
        "Categorize the user's intent internally before making a decision:\n\n"
        "CATEGORY 1: Conversational Chat\n"
        "- Definition: The user is greeting you, asking a question, or talking.\n"
        "- Action: DO NOT CALL ANY TOOLS. Respond with a direct text response.\n\n"
        "CATEGORY 2: OS Action\n"
        "- Definition: Direct command to execute, open, or run a local app or game.\n"
        "- Action: Call the 'run_os_app' tool immediately.\n\n"
        "CATEGORY 3: Browser Action\n"
        "- Definition: Direct command to visit, browse, or launch a live website link.\n"
        "- Action: Call the 'open_browser_url' tool immediately.\n\n"
        "If you decide to use a tool, output a single clean tool call. "
        "Do not invent new tool names outside your list.\n"
        "CRITICAL: Be conservative. If you are unsure, default to Category 1."
    ))
    
    all_messages = state["messages"]
    if len(all_messages) > 16:
        all_messages = all_messages[-10:]
    full_prompt = [guardrail] + all_messages
    response = llm_with_tools.invoke(full_prompt)
    
    return {
        "messages": [response],
        "route_intent": state["route_intent"],
        "mode": state["mode"],
    }