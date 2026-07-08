"""Simple chat node for standard mode — no tools, just conversation."""
from __future__ import annotations
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from graph.state import VedState

SYSTEM_PROMPT = (
    "You are Ved, a helpful conversational assistant. "
    "Answer questions, chat, and explain things concisely. "
    "You do not have access to file tools, code execution, or apps in this mode."
)

def simple_chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="Ved is hibernating. Switch to turbo or standard mode first.")],
            "route_intent": state.route_intent, "mode": state.mode,
        }
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.messages)
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    content = ""
    for chunk in llm.stream(messages):
        c = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        content += c
        if token_queue is not None and c:
            try: token_queue.put(c)
            except Exception: pass
    return {
        "messages": [AIMessage(content=content)],
        "route_intent": state.route_intent, "mode": state.mode,
    }
