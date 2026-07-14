"""Standalone chat node — simple single-agent chatbot with bound tools.

ARCHITECTURE PIVOT PREP (Phase 2 architecture review):

For non-coder mode with a capable model (8B/14B 4q), the planner-executor
split is unnecessary overhead. A single agent loop with bound tools does
the same job with less latency and indirection. This module is the sketch
of that alternative — a drop-in replacement for the planner_node +
executor_node pair when the model is big enough to handle tool use
directly.

Design:
  - Single node: standalone_chat_node(state, get_llm, config)
  - Binds tools directly: PATH_A_EXECUTOR_TOOLS (or a subset for safety)
  - Agent loop: stream LLM -> if tool_calls, execute -> repeat
  - Multi-step tasks: lightweight in-memory todo list (not persisted)
  - No plan file, no markers, no executor
  - Bounded iterations (8) + wall-clock timeout (90s) for safety

When to enable: when mode != "coder" AND the active model is >= 8B.
For 3B models, keep the planner-executor split (the 3B can't reliably
hold a tool-using agent loop in its head).

Wiring (future):
  - chatbot.py routes non-coder chats to this node instead of planner_node
  - Flag: self._use_standalone_chat = True if mode != "coder" and model_size >= 8B
  - Graph builder conditionally adds this node
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from graph.state import VedState
from graph.tools import PATH_A_EXECUTOR_TOOLS, _assert_tool_isolation
from graph.nodes._stream_helpers import _clean_chunk


# ---- Tunables ----
_MAX_AGENT_ITERATIONS = 8
_TOOL_RESULT_MAX_CHARS = 1500
_MAX_WALL_SECONDS = 90.0
_MAX_TODO_ITEMS = 12  # in-memory todo list cap


# ---- Todo list helpers ----
# Lightweight in-memory task tracking for multi-step requests. NOT persisted
# to disk — if the user closes the session mid-task, the todo is lost.
# That's an acceptable tradeoff for the simplicity gain vs. the plan file.

def _todo_add(todos: List[str], item: str) -> Tuple[List[str], Optional[str]]:
    """Append a todo item. Returns (new_list, error_or_None)."""
    if not item or not item.strip():
        return todos, "empty todo"
    item = item.strip()
    if item in todos:
        return todos, None  # idempotent
    if len(todos) >= _MAX_TODO_ITEMS:
        return todos, f"todo cap reached ({_MAX_TODO_ITEMS})"
    return todos + [item], None


def _todo_done(todos: List[str], item: str) -> Tuple[List[str], Optional[str]]:
    """Mark a todo as done (remove from list). Returns (new_list, error)."""
    if item in todos:
        return [t for t in todos if t != item], None
    return todos, f"todo not found: {item}"


def _todo_render(todos: List[str]) -> str:
    """Format the todo list for the LLM's context."""
    if not todos:
        return "(no pending todos)"
    return "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(todos))


# ---- Node ----

def standalone_chat_node(
    state: VedState,
    get_llm,
    config: RunnableConfig,
) -> dict:
    """Single-agent chatbot with bound tools. Replaces planner+executor for non-coder mode.

    Flow:
      1. Build system prompt with tools + todo list
      2. Stream LLM turn
      3. If tool_calls: execute inline, append ToolMessages, loop
      4. If no tool_calls: return the AI text as the final response
      5. Bound by max iterations + wall-clock timeout

    State writes:
      - messages: [AIMessage(final_text)]
      - route_intent: "A" (treated as final answer)
      - active_plan_id: None
    """
    started_at = time.time()

    # Tool set: read-only for non-coder (same as Path A executor).
    # Defensive assertion catches any future leak.
    tools_to_bind = PATH_A_EXECUTOR_TOOLS
    _assert_tool_isolation(state.mode, tools_to_bind)
    tool_map = {t.name: t for t in tools_to_bind}

    # Resolve LLM (mode-aware factory if injected)
    llm = None
    if config and isinstance(config.get("configurable"), dict):
        factory = config["configurable"].get("planner_llm_factory")
        if factory is not None:
            llm = factory(state.mode)
    if llm is None:
        llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="No local model is available. Start Ollama.")],
            "route_intent": state.route_intent, "mode": state.mode,
        }

    try:
        llm_with_tools = llm.bind_tools(tools_to_bind)
    except Exception as exc:
        return {
            "messages": [AIMessage(content=f"Setup failed: {type(exc).__name__}: {exc}")],
            "route_intent": state.route_intent, "mode": state.mode,
        }

    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None

    # Build system prompt
    user_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
    last_user = user_msgs[-1].content if user_msgs else ""

    tools_note = (
        "Available tools: read_file, search_files, retrieve_rag, open_app. "
        "Use them directly to answer the user's request."
    )
    system_prompt = (
        "You are a capable AI assistant with access to local files and the "
        "thread's RAG store. You can read files, search the project, retrieve "
        "prior chat context, and open apps. There is NO planner or executor — "
        "you decide what to do and do it directly.\n\n"
        f"{tools_note}\n\n"
        "For multi-step tasks, keep a lightweight mental todo list of the "
        "remaining work. You don't need to announce it unless the user asks.\n\n"
        "Be concise. When done, give a direct answer summarizing what you did."
    )

    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user),
    ]

    # ---- Agent loop ----
    last_content = ""
    for iteration in range(_MAX_AGENT_ITERATIONS):
        # Wall-clock guard
        if time.time() - started_at > _MAX_WALL_SECONDS:
            last_content = (last_content or "") + (
                "\n\n(Stopped: per-turn wall timeout reached.)"
            )
            break

        # Stream one turn
        full_content = ""
        tool_calls_acc: Dict[int, Dict] = {}
        try:
            for chunk in llm_with_tools.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    raw = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    full_content += raw
                    c = _clean_chunk(raw)
                    if c is None:
                        continue
                    if token_queue is not None:
                        try:
                            token_queue.put(c)
                        except Exception:
                            pass
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
        except Exception as exc:
            last_content = f"Stream failed: {type(exc).__name__}: {exc}"
            break

        last_content = full_content

        # Parse tool calls
        tool_calls_list: List[Dict] = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except Exception:
                args = {"_raw": tc["args"]}
            tool_calls_list.append({"id": tc["id"], "name": tc["name"], "args": args})

        if not tool_calls_list:
            # LLM is done — no more tool calls.
            break

        # Append AI message and execute each tool
        ai_msg = AIMessage(content=full_content, tool_calls=tool_calls_list)
        messages.append(ai_msg)

        any_error = False
        for tc in tool_calls_list:
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"ERROR: unknown tool '{tc['name']}'"
                any_error = True
            else:
                try:
                    result = tool.invoke(tc["args"])
                    result = result if isinstance(result, str) else str(result)
                except Exception as exc:
                    result = f"ERROR: {type(exc).__name__}: {exc}"
                    any_error = True
            if len(result) > _TOOL_RESULT_MAX_CHARS:
                result = result[:_TOOL_RESULT_MAX_CHARS] + f"\n...[truncated at {_TOOL_RESULT_MAX_CHARS} chars]"
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        if any_error:
            # One error is enough to stop — surface it to the user
            last_content = full_content + (
                "\n\nA tool call failed. I'll stop here and report what happened."
            )
            break

    return {
        "messages": [AIMessage(content=last_content or "(no response)")],
        "route_intent": "A",
        "mode": state.mode,
        "active_plan_id": None,
    }


__all__ = ["standalone_chat_node", "_todo_add", "_todo_done", "_todo_render"]
