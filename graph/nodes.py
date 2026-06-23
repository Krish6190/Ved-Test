from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .state import VedState
from langchain_core.runnables import RunnableConfig

class RouterSchema(BaseModel):
    intent: Literal["A", "B", "C"] = Field(
        description=(
            "Route the user request to the single best matching execution pathway:\n"
            "A = Standard interactive Q&A, general conversation, casual chat, definitions, greetings, or short code/math concept explanations.\n"
            "B = Structured long-form text synthesis, multi-paragraph essays, deep formal articles, or research stories. Use this path EVEN IF the request explicitly mentions needing real-time web facts, live statistics, or web-scraped data to complete the essay text.\n"
            "C = Explicit standalone requests to execute sandboxed Python scripts, process local files, or run system-level utility tools directly without generating a long creative text document."
        )
    )
def intent_router_node(state: VedState, get_llm) -> dict:
    """Analyzes the user message to determine the workflow path."""
    llm = get_llm()
    if llm is None:
        return {"route_intent": "A", "messages": state.messages, "mode": state.mode}

    if hasattr(llm, "temperature"):
        llm.temperature = 0.0
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content if user_messages else ""

    router_prompt = (
        "Classify this user message into one of the explicit route paths.\n"
        "CRITICAL RULE: If the user is asking a direct, short question about your identity, what model you are running, "
        "your current operational mode, or basic setup status, you MUST choose route 'A'. Do not route brief conversational "
        f"system questions to 'B' or 'C'.\n\nUser Message: {last_user_text}"
    )
    try:
        structured_llm = llm.with_structured_output(RouterSchema)
        response = structured_llm.invoke([SystemMessage(content=router_prompt)])
        chosen_route = response.intent
    except Exception:
        chosen_route = "A"

    return {
        "route_intent": chosen_route,
        "messages": state.messages,
        "mode": state.mode
    }

def chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Conversational chat node handling Path A."""
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
    response = llm.invoke(state.messages)
    return {
        "messages": [response],
        "route_intent": state.route_intent,
        "mode": state.mode
    }
def content_pipeline_node(state: VedState) -> dict:
    """Temporary placeholder for Path B: Multi-format document synthesis pipeline."""
    return {
        "messages": [AIMessage(content="[System Placeholder] Content pipeline path triggered. Flow building in next step.")]
    }

def python_tool_node(state: VedState) -> dict:
    """Temporary placeholder for Path C: Python script executor."""
    return {
        "messages": [AIMessage(content="[System Placeholder] Python script path triggered. Flow building in next step.")]
    }

def coder_chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Isolated coding assistant node using Qwen 2.5 Coder 7B."""
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
    response = llm.invoke(state.messages)
    return {
        "messages": [response],
        "route_intent": "",
        "mode": state.mode
    }