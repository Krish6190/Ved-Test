"""Path A — conversational chat node.

Handles plain chat with RAG retrieval and tool use. Bound with VED_TOOLS
so the LLM can emit structured tool calls; streaming is preserved for text
content; tool_call_chunks are accumulated and merged into the final
AIMessage so the graph can route to ToolNode next.

Also includes the cross-mode tool-creation trigger: if the LLM admits it
lacks a tool, flushes the current model, loads coder, retries with a nudge.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from graph.nodes._helpers import (
    _build_rag_block,
    _emit_tool_call_event,
    _maybe_trigger_tool_creation,
    _message_requires_tools,
    _stream_llm_with_tools,
    _trim_history_for_model,
)
from graph.nodes._hints import DEFAULT_HINTS
from graph.state import VedState
from graph.tools import VED_TOOLS

def chat_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Conversational chat node handling Path A with real-time streaming hooks.

    Tool binding is conditional: VED_TOOLS are only bound when the user's
    message looks tool-requiring (verbs/nouns like read, open, search, run).
    For casual greetings ("hello", "what can you do") we bind an empty
    tool list so smaller models don't hallucinate fake tool calls.

    The streaming helper also filters out tool calls with empty required
    args (a separate defense layer for models that ignore the hint).
    """
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
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    active_thread_id = None
    if config and isinstance(config.get("configurable"), dict):
        active_thread_id = config["configurable"].get("active_thread_id")
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""

    messages_to_stream = _trim_history_for_model(state.messages, llm)
    context_block = _build_rag_block(last_user_text, active_thread_id)
    if context_block:
        messages_to_stream = [SystemMessage(content=context_block)] + messages_to_stream
    messages_to_stream = list(DEFAULT_HINTS) + messages_to_stream
    tools_to_bind = VED_TOOLS if _message_requires_tools(last_user_text) else []
    llm_to_use = llm.bind_tools(tools_to_bind)

    full_content, tool_calls_list = _stream_llm_with_tools(
        llm_to_use, messages_to_stream, token_queue,
    )

    ai_msg = AIMessage(content=full_content, tool_calls=tool_calls_list)
    _emit_tool_call_event(tool_calls_list, token_queue)
    if not tool_calls_list:
        triggered = _maybe_trigger_tool_creation(state, llm_to_use, config, ai_msg)
        if triggered is not ai_msg:
            new_ai, new_mode = triggered
            return {"messages": [new_ai], "route_intent": state.route_intent, "mode": new_mode}
    return {"messages": [ai_msg], "route_intent": state.route_intent, "mode": state.mode}