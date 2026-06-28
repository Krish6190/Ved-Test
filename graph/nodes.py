import re
import json
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .state import VedState
from langchain_core.runnables import RunnableConfig
import sys
import subprocess
from graph.tools import VED_TOOLS
_PLANNING_HINT = SystemMessage(content=(
    "You have access to tools (read_file, edit_file, overwrite_file, "
    "search_files, execute_python). Use them when the task needs file I/O "
    "or code execution instead of guessing.\n"
    "\n"
    "For multi-step tasks: state a short plan first, then execute step by "
    "step. If a tool returns an error or no result, replan and try a "
    "different approach - do not give up after one failure."
))
_TOOL_RESTRAINT_HINT = SystemMessage(content=(
    "Tool invocation policy: only call a tool (read_file, edit_file, "
    "overwrite_file, search_files, execute_python, web_search) when the "
    "user explicitly asks for file I/O, code execution, web search, or "
    "file search. For greetings, chitchat, knowledge questions, opinions, "
    "and conversational replies — respond directly with text. Do not call "
    "a tool speculatively or to 'be helpful' when the user did not request "
    "it. If unsure whether the user wants a tool, prefer a direct reply."
))
_TOOL_OUTPUT_HINT = SystemMessage(content=(
    "When a tool returns content, REPRODUCE IT VERBATIM in your reply. "
    "Do not summarize, paraphrase, or describe what a file contains - "
    "show the actual text the user asked for. The user wants to see exactly "
    "what the tool returned, not your interpretation of it.\n"
    "\n"
    "Specifically for read_file: paste the full FILE/SIZE block plus the "
    "complete file contents inside the ``` ... ``` block. Truncation is OK "
    "only if the file is huge (over 8000 chars) - in that case mention the "
    "truncation but still paste what was returned.\n"
    "\n"
    "For search_files: list every match the tool returned, one per line, "
    "with a brief count. Do not pick a subset.\n"
    "\n"
    "For edit_file/overwrite_file: state the planned change in plain "
    "English BEFORE invoking the tool, so the user knows what the approval "
    "popup is about."
))

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

    # 0. Self-healing intent (highest priority). When the user asks Ved to repair
    # itself, restrict file_read/file_search/file_edit to the project root by
    # flipping state.self_healing=True and routing to Path C.
    self_heal_phrases = (
        "self heal", "self-heal", "self-healing",
        "heal yourself", "heal itself",
        "fix yourself", "fix itself", "fix your code", "fix your own code",
        "repair yourself", "repair your code", "repair itself",
        "self repair", "self-repair",
    )
    if any(p in lower_text for p in self_heal_phrases):
        return {"route_intent": "C", "self_healing": True}

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

    # 4. File/search/edit operations route to chat_node (Path A), which has
    # the LangChain @tools bound with implicit-target fallback. Bare "/run"
    # and "execute ..." still go to Path C (rule 3). This is what makes
    # "read this file" / "open the config" / "search for *.py" trigger
    # the LLM's tool calls instead of the old fence-tag dispatcher.
    if not re.search(r"\b(what|how|why|when|where|who|which)\b", lower_text):
        if re.match(r"^(read|open|edit|search|find|modify|update)\s+", lower_text):
            return {"route_intent": "A"}

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
    """Conversational chat node handling Path A with real-time streaming hooks.

    Bound with VED_TOOLS so the LLM can emit structured tool calls. Streaming
    is preserved for text content; tool_call_chunks are accumulated and merged
    into the final AIMessage so the graph can route to ToolNode next.
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
    # Bind the tools so the LLM can emit structured tool calls.
    llm_with_tools = llm.bind_tools(VED_TOOLS)
    full_content = ""
    tool_calls_acc: dict[int, dict] = {}
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

    # Always inject the planning + tool-output hints so the LLM knows it has
    # tools, should plan for multi-step work, and must reproduce file/search
    # results verbatim instead of summarizing.
    messages_to_stream = [_PLANNING_HINT, _TOOL_RESTRAINT_HINT, _TOOL_OUTPUT_HINT] + messages_to_stream

    for chunk in llm_with_tools.stream(messages_to_stream):
        # Text content - stream to UI token_queue
        if hasattr(chunk, "content") and chunk.content:
            c = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            full_content += c
            if token_queue:
                token_queue.put(c)
        # Tool call accumulation - merge partial chunks into final tool_calls
        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                idx = tc.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"name": "", "args": "", "id": ""}
                if tc.get("id"):
                    tool_calls_acc[idx]["id"] = tc["id"]
                if tc.get("name"):
                    tool_calls_acc[idx]["name"] = tc["name"]
                if tc.get("args"):
                    tool_calls_acc[idx]["args"] += tc["args"]

    # Reconstruct the final AIMessage with merged tool_calls
    tool_calls_list: list[dict] = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        try:
            args = json.loads(tc["args"]) if tc["args"] else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls_list.append({"id": tc["id"], "name": tc["name"], "args": args})

    ai_msg = AIMessage(content=full_content, tool_calls=tool_calls_list)
    return {"messages": [ai_msg], "route_intent": state.route_intent, "mode": state.mode}

def python_tool_node(state: VedState, config: RunnableConfig) -> dict:
    """Legacy `/run ...` entrypoint.

    With LangChain tool calling wired into `chat_node`/`coder_chat_node`, the
    LLM normally invokes `execute_python` as a LangChain tool. This node
    remains for the explicit `/run ...` slash command: it extracts the
    Python code from the last message (any type) and runs it via the
    `execute_python` @tool, which still enforces the approval popup + 10s
    timeout + cleanup.
    """
    import re
    from langchain_core.messages import HumanMessage
    from graph.tools.python_runner import execute_python

    # Find the most recent message with content (mirrors the original logic)
    last_text = ""
    for msg in reversed(state.messages):
        if hasattr(msg, "content") and msg.content:
            last_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Prefer code from a python fence; fall back to the whole message
    fence = re.search(r"```python\s*([\s\S]*?)```", last_text)
    raw_code = fence.group(1).strip() if fence else last_text.strip()

    # Delegate to the @tool (it handles approval, timeout, temp file, cleanup)
    result = execute_python.invoke({"code": raw_code, "state": state})
    return {"messages": [HumanMessage(content=result)], "route_intent": "", "mode": state.mode}

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