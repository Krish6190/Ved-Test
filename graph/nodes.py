from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .state import VedState
from langchain_core.runnables import RunnableConfig

class RouterSchema(BaseModel):
    intent: Literal["A", "B", "C"] = Field(
        description=(
            "Route the user request to the single best matching execution pathway:\n"
            "A = ANY question asking 'what is', 'how does', 'explain', meanings, definitions, terminology, conceptual breakdowns, "
            "informational queries, status updates, or standard back-and-forth conversation. If the user expects a direct, "
            "conversational response under 3 paragraphs, you MUST choose 'A'.\n"
            "B = Explicit requests to generate, draft, compile, or write long-form assets (e.g., 'write an essay', 'compose a letter', "
            "'draft a multi-paragraph blog article', 'generate a full report'). Do NOT use 'B' unless the user is explicitly "
            "asking you to perform a creative or formal document generation task.\n"
            "C = Explicit standalone requests to run commands, execute local files, launch sandboxed scripts, or compile code lines "
            "inside the workspace terminal boundaries."
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