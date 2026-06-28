"""Path C — isolated coding assistant node.

Uses Qwen 2.5 Coder 7B with streaming. Bound with VED_TOOLS so the coder
can emit structured tool calls (including `propose_tool` for in-conversation
tool creation). The coder is prompted to plan first, then call tools one
at a time. Each tool call is surfaced to the UI as a `tool_call` SSE event.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from graph.nodes._helpers import (
    _emit_tool_call_event,
    _maybe_trigger_tool_creation,
    _stream_llm_with_tools,
    _trim_history_for_model,
)
from graph.nodes._hints import DEFAULT_HINTS
from graph.state import VedState
from graph.tools import VED_TOOLS


def coder_chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Isolated coding assistant node using Qwen 2.5 Coder 7B with streaming support.

    Bound with VED_TOOLS so the coder LLM can emit structured tool calls
    (including `propose_tool` for in-conversation tool creation). The coder
    is prompted to plan first, then call tools one at a time. Each tool
    call is surfaced to the UI as a `tool_call` SSE event so the user can
    see what the coder is about to do.
    """
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
    llm_with_tools = llm.bind_tools(VED_TOOLS)
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    messages_to_stream = list(DEFAULT_HINTS) + _trim_history_for_model(state.messages, llm)
    full_content, tool_calls_list = _stream_llm_with_tools(
        llm_with_tools, messages_to_stream, token_queue,
    )
    _emit_tool_call_event(tool_calls_list, token_queue)
    ai_msg = AIMessage(content=full_content, tool_calls=tool_calls_list)
    if not tool_calls_list:
        triggered = _maybe_trigger_tool_creation(state, llm_with_tools, config, ai_msg)
        if triggered is not ai_msg:
            new_ai, new_mode = triggered
            return {"messages": [new_ai], "route_intent": state.route_intent, "mode": new_mode}
    return {"messages": [ai_msg], "route_intent": state.route_intent, "mode": state.mode}