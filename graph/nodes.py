import re
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .state import VedState
from langchain_core.runnables import RunnableConfig

def intent_router_node(state: VedState, get_llm) -> dict:
    """Analyzes the user message to determine the optimal workflow path with fallback safety."""
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""
    lower_text = last_user_text.lower()
    
    # 1. Fast Keyword/Regex Intercept Layers (Instant response, zero LLM latency)
    content_triggers = ["generate", "write me", "draft", "compose", "summarize", "summary of"]
    if any(trigger in lower_text for trigger in content_triggers):
        return {"route_intent": "B"}
    if lower_text.startswith("/run") or lower_text.startswith("execute"):
        return {"route_intent": "C"}
        
    llm = get_llm()
    if llm is None: 
        return {"route_intent": "A"}

    # 2. Complete, Unabridged Prompt System to Guard Decision Quality
    router_prompt = (
        "You are a strict request classifier. Your sole job is to classify the user message into exactly one route.\n\n"
        "THE PROTOCOLS:\n"
        "Route A: ANY question asking 'what is', 'how does', 'explain', meanings, definitions, terminology, conceptual breakdowns, "
        "informational queries, status updates, greetings, small talk, or standard back-and-forth conversation. When in doubt, choose A.\n"
        "Route B: Explicit requests to generate, draft, compile, or write long-form assets (e.g., 'write an essay', 'compose a letter', "
        "'draft a multi-paragraph blog article', 'generate a full report') of any size.\n"
        "Route C: Explicit standalone requests to run commands, execute local files, launch sandboxed scripts, or compile code lines "
        "inside the workspace terminal boundaries.\n\n"
        "EXAMPLES:\n"
        "- 'hey' -> A\n"
        "- 'my name is John' -> A\n"
        "- 'what is Python' -> A\n"
        "- 'write me a 1000 word essay on climate change' -> B\n"
        "- 'draft a formal letter to my landlord' -> B\n"
        "- 'generate a sales report' -> B\n"
        "- 'run this script' -> C\n"
        "- 'execute the command ls -la' -> C\n"
        "- 'open browser then go to https://whatsappweb.com' -> C\n"
        "- 'Text shivam to say I will be late' -> A\n\n"
        f"User message: '{last_user_text}'\n\n"
        "OUTPUT FORMAT RULE:\n"
        "Reply with ONLY a single character: A, B, or C. Do not output any markdown, explanations, or surrounding text."
    )
    
    try:
        # Direct raw token invocation avoids structured tool loops entirely
        raw_res = llm.invoke([SystemMessage(content=router_prompt)]).content.strip().upper()
        
        # Robust regex fallback wrapper: looks for the first standalone A, B, or C in the response
        match = re.search(r"\b([A-C])\b", raw_res)
        chosen_route = match.group(1) if match else "A"
    except Exception:
        chosen_route = "A"
        
    return {"route_intent": chosen_route}

def chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Conversational chat node handling Path A with real-time streaming hooks."""
    if state.mode == "hibernate":
        return {
            "messages": [AIMessage(content="Ved is hibernating. Switch to turbo or standard mode first.")],
            "route_intent": state.route_intent, "mode": state.mode
        }
    llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="No local model is available. Start Ollama.")],
            "route_intent": state.route_intent, "mode": state.mode
        }
    if hasattr(llm, "temperature"):
        llm.temperature = 0.1
    full_content = ""
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    
    for chunk in llm.stream(state.messages):
        if chunk.content:
            full_content += chunk.content
            if token_queue:
                token_queue.put(chunk.content)

    return {"messages": [AIMessage(content=full_content)], "route_intent": state.route_intent, "mode": state.mode}

def python_tool_node(state: VedState) -> dict:
    """Temporary placeholder for Path C: Python script executor."""
    return {
        "messages": [AIMessage(content="[System Placeholder] Python script path triggered. Flow building in next step.")]
    }

def coder_chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Isolated coding assistant node using Qwen 2.5 Coder 7B with streaming support."""
    llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="No coding model available. Verify Ollama initialization.")],
            "route_intent": "", 
            "mode": state.mode
        }    
    if hasattr(llm, "temperature"):
        llm.temperature = 0.1
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content.strip().lower() if user_messages else ""
    if last_user_text.startswith("/run") or "execute script" in last_user_text:
        return {
            "messages": state.messages,
            "route_intent": "C",
            "mode": state.mode
        }
    full_content = ""
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    for chunk in llm.stream(state.messages):
        if chunk.content:
            full_content += chunk.content
            if token_queue:
                token_queue.put(chunk.content)
    return {"messages": [AIMessage(content=full_content)], "route_intent": state.route_intent, "mode": state.mode}