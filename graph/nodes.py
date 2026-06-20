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
            "saved_memories": state.get("saved_memories", []),
        }

    llm = get_llm()
    if llm is None:
        return {"messages": [AIMessage(content="No local model is available.")]}

    # Bind the combined tools list directly to the model
    llm_with_tools = llm.bind_tools(all_ved_tools)
    saved_items = state.get("saved_memories", [])
    
    # 2. Bundle those permanent facts into an un-erasable Core Memory text block
    long_term_prompt = ""
    if saved_items:
        formatted_saves = "\n".join([f"* {item}" for item in saved_items])
        long_term_prompt = f"\nCRITICAL CORE MEMORY (Never forget these facts):\n{formatted_saves}\n"
    # Enforce categorization rules to stop hallucinated tool usage
    guardrail = SystemMessage(content=(
        "You are an AI agent named Ved. You have access to tools to open apps and websites. "
        f"{long_term_prompt}"
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
    
    MAX_CONTEXT_MESSAGES = 16

    all_messages = state["messages"]
    if len(all_messages) > MAX_CONTEXT_MESSAGES:
        all_messages = all_messages[-MAX_CONTEXT_MESSAGES:]
    full_prompt = [guardrail] + all_messages
    response = llm_with_tools.invoke(full_prompt)
    print("DEBUG response:", repr(response.content), "tool_calls:", response.tool_calls)
    return {
        "messages": [response],
        "route_intent": state["route_intent"],
        "mode": state["mode"],
        "saved_memories": state["saved_memories"],
    }