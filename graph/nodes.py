from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .state import VedState
from langchain_core.runnables import RunnableConfig

# Create a dynamic runtime schema straight from our VedState property definition
# This completely eliminates copy-pasted validation text or redundant model schemas
class RouterSchema(BaseModel):
    intent: Literal["A", "B", "C"] = Field(
        description=VedState.model_fields["route_intent"].description
    )

def intent_router_node(state: VedState, get_llm) -> dict:
    """
    Analyzes the user message with a cold model footprint to determine 
    the workflow path (A, B, or C). Preserves mode status.
    """
    llm = get_llm()
    if llm is None:
        return {
            "route_intent": "A",
            "messages": state.messages,
            "mode": state.mode
        }

    if hasattr(llm, "temperature"):
        llm.temperature = 0.0
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content if user_messages else ""

    router_prompt = (
        "Classify this user message into one of the explicit route paths:\n"
        f"User Message: {last_user_text}"
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
            "route_intent": state.route_intent,
            "mode": state.mode
        }
    llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="No local model is available. Start Ollama.")],
            "route_intent": state.route_intent,
            "mode": state.mode
        }
    if hasattr(llm, "temperature"):
        llm.temperature = 0.1
    system_prompt = config.get("configurable", {}).get("system_prompt", "")
    llm_inputs = []
    if system_prompt:
        llm_inputs.append(SystemMessage(content=system_prompt))
    llm_inputs.extend(state.messages)
    response = llm.invoke(llm_inputs)
    return {
        "messages": [response],
        "route_intent": state.route_intent,
        "mode": state.mode
    }
def essay_generator_node(state: VedState) -> dict:
    """Temporary placeholder for Path B: Creative essay loop."""
    return {
        "messages": [AIMessage(content="[System Placeholder] Essay path triggered. Flow building in next step.")]
    }

def python_tool_node(state: VedState) -> dict:
    """Temporary placeholder for Path C: Python script executor."""
    return {
        "messages": [AIMessage(content="[System Placeholder] Python script path triggered. Flow building in next step.")]
    }
