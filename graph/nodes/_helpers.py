"""Shared utilities for graph nodes.

- LLM streaming with tool_calls accumulation
- Tool-call SSE event emission (UI visibility)
- Cross-mode tool-creation trigger
- RAG/web context block construction (used by chat_node)

These were extracted from the original monolithic `nodes.py` so chat_node
and coder_chat_node can share the same streaming + tool-call accounting
without duplication.
"""
from __future__ import annotations
import json
import re
from typing import Any, List, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_NEEDS_TOOL_RE = re.compile(
    r"\b(I can'?t|I cannot|not able to|don'?t have (a )?tool|no tool (found|exists|available))\b",
    re.IGNORECASE,
)

_TOOL_TRIGGER_RE = re.compile(
    r"\b("
    r"read|open|edit|write|modify|update|save|delete|remove|"
    r"search|find|locate|where|grep|"
    r"run|execute|script|code|"
    r"launch|start|spawn|"
    r"create|make|build|generate|compose|draft|"
    r"show|list|display|cat|head|tail|"
    r"file|folder|directory|path"
    r")\b",
    re.IGNORECASE,
)


def _message_requires_tools(text: str) -> bool:
    """Heuristic: does the user's message look tool-requiring?

    Returns True if the message contains tool-trigger verbs/nouns. Used by
    chat_node to conditionally bind VED_TOOLS — for casual greetings we
    bind an empty tool list so the LLM can't hallucinate tool calls.

    This is deliberately conservative. False positives (binding tools when
    not needed) only cost a few extra output tokens; false negatives (not
    binding when needed) make the LLM helpless for valid requests.
    """
    if not text or not text.strip():
        return False
    return bool(_TOOL_TRIGGER_RE.search(text))

_SMALL_MODEL_HISTORY_CAP = 10
_DEFAULT_HISTORY_CAP = 40


def _is_small_model(llm) -> bool:
    """Heuristic: is this a small model that needs tighter context trimming?

    Inspects attributes Ollama / LangChain set on the chat instance.
    Falls back to "not small" when nothing matches.
    """
    candidates = []
    for attr in ("model", "model_name"):
        v = getattr(llm, attr, None)
        if isinstance(v, str):
            candidates.append(v.lower())
    base = getattr(llm, "bound_llm", None) or llm
    for attr in ("model", "model_name"):
        v = getattr(base, attr, None)
        if isinstance(v, str):
            candidates.append(v.lower())
    name = " ".join(candidates)
    if "coder:7b" in name or "coder:14b" in name:
        return False
    # Parse the size tag at the end of the model name. Handles:
    # "llama3.2:3b", "qwen2.5:1.5b", "gemma:2b", "phi:2.7b", "qwen:0.5b".
    last_seg = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    size_m = re.match(r"^(\d+(?:\.\d+)?)(b|m)(?:[-:]|$)", last_seg)
    if size_m:
        size = float(size_m.group(1))
        unit = size_m.group(2)
        billions = size / 1000.0 if unit == "m" else size
        if billions <= 3.5:
            return True
        return False
    if any(tag in name for tag in ("tiny", "mini", "nano")):
        return True
    return False


def _trim_history_for_model(state_messages: list, llm) -> list:
    """Trim conversation history based on model size.

    Small models (llama3.2:3b, etc.) get the last 10 messages only.
    Larger models keep the full conversation (up to THREAD_MESSAGE_CAP).
    Always preserves the SystemMessage at index 0 if present, plus the
    final HumanMessage (the current request).
    """
    cap = _SMALL_MODEL_HISTORY_CAP if _is_small_model(llm) else _DEFAULT_HISTORY_CAP
    if len(state_messages) <= cap:
        return list(state_messages)
    # Always keep the last `cap` messages; drop older ones.
    return list(state_messages[-cap:])


def _filter_empty_tool_calls(tool_calls_list: List[dict]) -> List[dict]:
    """Drop hallucinated tool calls where any string arg is empty/None.

    Small models (llama3.2:3b) hallucinate tool calls with empty args
    (e.g. code="", pattern="") instead of responding with text. Filtering
    falls back to a normal text response. A call is dropped when any of
    its string args is empty/whitespace, or when it has no meaningful
    (non-empty) content at all.
    """
    if not tool_calls_list:
        return tool_calls_list
    filtered = []
    for tc in tool_calls_list:
        args = tc.get("args") or {}
        if not args:
            continue
        any_bad_string = False
        any_meaningful = False
        for v in args.values():
            if isinstance(v, str):
                if not v.strip():
                    any_bad_string = True
                else:
                    any_meaningful = True
            elif v is not None:
                any_meaningful = True
        if not any_meaningful or any_bad_string:
            continue
        filtered.append(tc)
    return filtered

def _summarize_args(args: dict) -> str:
    """Compact preview of tool args for UI display in tool_call SSE events."""
    if not args:
        return ""
    parts = [f"{k}={repr(v)[:80]}" for k, v in args.items()]
    return ", ".join(parts)

def _stream_llm_with_tools(
    llm_with_tools,
    messages_to_stream,
    token_queue,
) -> Tuple[str, list]:
    """Stream an LLM (with bound tools) and accumulate content + tool_calls.

    Args:
        llm_with_tools: an `llm.bind_tools(VED_TOOLS)` instance.
        messages_to_stream: the full list of messages to send (including
            any prepended SystemMessage hints).
        token_queue: optional queue.Queue; text tokens are pushed as raw
            strings, tool_call_chunks are accumulated silently.

    Returns:
        (full_content, tool_calls_list) where tool_calls_list is a list of
        dicts with keys {id, name, args}.
    """
    full_content = ""
    tool_calls_acc: dict[int, dict] = {}
    for chunk in llm_with_tools.stream(messages_to_stream):
        # Text content - stream to UI token_queue
        if hasattr(chunk, "content") and chunk.content:
            c = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            full_content += c
            if token_queue:
                try:
                    token_queue.put(c)
                except Exception:
                    pass
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

    tool_calls_list: list[dict] = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        try:
            args = json.loads(tc["args"]) if tc["args"] else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls_list.append({"id": tc["id"], "name": tc["name"], "args": args})

    tool_calls_list = _filter_empty_tool_calls(tool_calls_list)
    return full_content, tool_calls_list

def _emit_tool_call_event(tool_calls_list, token_queue) -> None:
    """Surface planned tool calls to the UI as a `tool_call` SSE event.

    Lets the UI render "→ calling read_file(path=foo.py)" before any
    approval gate fires. No-op if there are no tool calls or no queue.
    """
    if not tool_calls_list or token_queue is None:
        return
    try:
        token_queue.put(("tool_call", {
            "calls": [
                {"name": tc.get("name"), "args_preview": _summarize_args(tc.get("args", {}))}
                for tc in tool_calls_list
            ]
        }))
    except Exception:
        pass

def _build_rag_block(last_user_text: str, active_thread_id: Optional[str]) -> str:
    """Dual-source RAG: thread + global, with web-search fallback.

    Returns a formatted context block (as a plain string) for injection as
    a SystemMessage at the head of the LLM stream. Returns "" if nothing
    useful is found (caller should not inject anything).
    """
    if not last_user_text:
        return ""
    context_block = ""
    try:
        from graph.rag.mixer import retrieve_context, _format_rag_block
        rag_chunks = retrieve_context(last_user_text, active_thread_id, k=5)
        if rag_chunks:
            context_block = _format_rag_block(rag_chunks)
    except Exception:
        pass  
    if not context_block:
        try:
            from graph.tools.web_search import web_search, format_web_results_block
            web_results = web_search(last_user_text, max_results=5)
            if web_results:
                context_block = format_web_results_block(web_results)
        except Exception:
            pass
    return context_block

def _maybe_trigger_tool_creation(state, llm_with_tools, config, first_ai_msg):
    """If the LLM's first response admits it lacks a tool, trigger cross-mode
    tool creation: switch to coder if needed, append a nudge HumanMessage,
    and re-invoke the LLM once. Returns the (possibly replaced) AIMessage.

    Safe to call from any mode. If the response already includes tool calls
    or doesn't match the trigger, returns first_ai_msg unchanged.

    Returns either:
      - first_ai_msg (no trigger fired)
      - (new_ai_msg, new_mode) tuple (trigger fired)
    """
    # Only fire when the LLM admits it can't do it AND didn't already emit a tool call.
    content = (first_ai_msg.content or "").strip() if hasattr(first_ai_msg, "content") else ""
    tool_calls = getattr(first_ai_msg, "tool_calls", None) or []
    if not content or tool_calls:
        return first_ai_msg
    if not _NEEDS_TOOL_RE.search(content):
        return first_ai_msg

    cfg = (config or {}).get("configurable", {}) or {}
    token_queue = cfg.get("token_queue")
    set_mode = cfg.get("set_mode")
    rebuild_graph = cfg.get("rebuild_graph")

    original_mode = state.mode
    if original_mode != "coder":
        if token_queue is not None:
            try:
                token_queue.put(("mode_switch", {
                    "from": original_mode,
                    "to": "coder",
                    "reason": "tool_creation",
                }))
            except Exception:
                pass
        if callable(set_mode):
            try:
                set_mode("coder")
            except Exception:
                pass
        if callable(rebuild_graph):
            try:
                rebuild_graph()
            except Exception:
                pass
    nudge = HumanMessage(content=(
        "SYSTEM: You have a tool called `propose_tool`. Call it now to "
        "design the tool the user asked for. Pass the function name, a "
        "one-line description, and the full Python source for the @tool "
        "function. The human will be shown the code for approval."
    ))
    messages_to_retry = list(state.messages) + [first_ai_msg, nudge]
    retry_content, retry_tool_calls_list = _stream_llm_with_tools(
        llm_with_tools, messages_to_retry, token_queue,
    )

    new_mode = state.mode if original_mode == "coder" else "coder"
    return AIMessage(content=retry_content, tool_calls=retry_tool_calls_list), new_mode