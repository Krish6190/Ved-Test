import re
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .state import VedState
from langchain_core.runnables import RunnableConfig
import sys
import subprocess

def intent_router_node(state: VedState, get_llm) -> dict:
    """Route the user message to Path A (chat + RAG + web), B (content generation),
    or C (tool execution). Pure heuristic router — no LLM call, so routing is
    deterministic, fast, and easy to debug.

    Rules (evaluated top-down, first match wins):
      1. Explicit override:  "... path A|B|C"  anywhere in the message
      2. Slash commands: any "/..."           → A (commands handled separately)
      3. Tool execution: "/run ..." or "execute ..."  → C
      4. File/tool verbs: "read this file", "open this file", "perform this"  → C
      5. Word-count spec:  "<N> word(s)/paragraph(s)/page(s)..."  → B
      6. Generation verb at start: "write/draft/compose/generate/create/make/build ..."  → B
      7. Generation phrases anywhere: "write me", "essay on", "blog post", etc.  → B
      8. Default: A
    """
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""
    lower_text = last_user_text.lower()

    # 1. Explicit override ("use path A" / "path B" etc.).
    override = re.search(r"\b(?:use\s+)?path\s+([abc])\b", lower_text)
    if override:
        return {"route_intent": override.group(1).upper()}

    # 2. Slash commands are handled separately by command_processor; never run through graph nodes.
    if lower_text.startswith("/"):
        return {"route_intent": "A"}

    # 3. Tool execution — explicit command forms.
    if lower_text.startswith("/run") or lower_text.startswith("execute"):
        return {"route_intent": "C"}

    # 4. File/tool operation verbs — only if it doesn't look like a question.
    if not re.search(r"\b(what|how|why|when|where|who|which)\b", lower_text):
        if re.match(r"^(read|open|run|execute|perform|launch|start|invoke)\s+", lower_text):
            return {"route_intent": "C"}

    # 5. Specific length specifiers — almost always generation.
    if re.search(r"\b\d+\s+(word|words|paragraph|paragraphs|page|pages|line|lines|char|chars|character|characters)\b", lower_text):
        return {"route_intent": "B"}

    # 6. Generation verbs at the start of the message.
    if re.match(r"^(write|draft|compose|generate|create|make|build|craft|author|produce)\s+", lower_text):
        return {"route_intent": "B"}

    # 7. Generation phrases anywhere in the message.
    generation_phrases = (
        "write me", "write a", "write an", "draft a", "draft an",
        "compose a", "essay on", "essay about", "blog post",
        "story about", "letter to", "report on", "summary of", "summarize",
    )
    if any(p in lower_text for p in generation_phrases):
        return {"route_intent": "B"}

    # 8. Default: Path A (chat + RAG + web).
    return {"route_intent": "A"}

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

    # Dual-source RAG: thread + global, with priority fallback. Silent on empty.
    active_thread_id = None
    if config and isinstance(config.get("configurable"), dict):
        active_thread_id = config["configurable"].get("active_thread_id")
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""
    messages_to_stream = list(state.messages)
    if last_user_text:
        context_block = ""
        try:
            from graph.rag.mixer import retrieve_context, _format_rag_block
            rag_chunks = retrieve_context(last_user_text, active_thread_id, k=5)
            if rag_chunks:
                context_block = _format_rag_block(rag_chunks)
        except Exception:
            pass  # RAG is optional; never block chat on a retrieval failure
        if not context_block:
            # RAG had nothing useful — try DuckDuckGo before falling back to bare prompt.
            try:
                from graph.tools.web_search import web_search, format_web_results_block
                web_results = web_search(last_user_text, max_results=5)
                if web_results:
                    context_block = format_web_results_block(web_results)
            except Exception:
                pass
        if context_block:
            messages_to_stream = [SystemMessage(content=context_block)] + messages_to_stream

    for chunk in llm.stream(messages_to_stream):
        if chunk.content:
            full_content += chunk.content
            if token_queue:
                token_queue.put(chunk.content)

    return {"messages": [AIMessage(content=full_content)], "route_intent": state.route_intent, "mode": state.mode}

def python_tool_node(state: VedState, config: RunnableConfig) -> dict:
    """Path C Execution Engine: Delegates shell tool execution tasks to our clean tool folder module."""
    # Append the project root dynamically to avoid module path mismatch faults on startup
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    from tools.python_runner import execute_safe_python_block
    return execute_safe_python_block(state, config)

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