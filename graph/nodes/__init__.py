"""Graph node implementations, split by responsibility.

Public surface (re-exported here so `from graph.nodes import X` still works):

  - intent_router_node    — heuristic routing (Path A/B/C)
  - chat_node             — Path A: chat + RAG + tools
  - coder_chat_node       — Path C: coding assistant + tools
  - python_tool_node      — legacy /run entrypoint

Helpers (used internally; not part of the public surface but re-exported
for test introspection):

  - _PLANNING_HINT, _TOOL_RESTRAINT_HINT, _TOOL_OUTPUT_HINT, _CODER_PLAN_DIRECTIVE
  - _summarize_args, _stream_llm_with_tools, _emit_tool_call_event
  - _build_rag_block, _maybe_trigger_tool_creation

Module layout:

  _hints.py      — SystemMessage prompts
  _helpers.py    — shared streaming / tool-call accounting / cross-mode trigger
  intent_router.py
  chat.py
  coder.py
  python_tool.py
"""
from __future__ import annotations
from graph.nodes.intent_router import intent_router_node
from graph.nodes.chat import chat_node
from graph.nodes.coder import coder_chat_node
from graph.nodes.python_tool import python_tool_node
from graph.nodes._hints import (
    CORE_HINTS,
    _CODER_PLAN_DIRECTIVE,
    _PLANNING_HINT,
    _TOOL_OUTPUT_HINT,
    _TOOL_RESTRAINT_HINT,
)
from graph.nodes._helpers import (
    _NEEDS_TOOL_RE,
    _build_rag_block,
    _emit_tool_call_event,
    _filter_empty_tool_calls,
    _is_small_model,
    _maybe_trigger_tool_creation,
    _message_requires_tools,
    _stream_llm_with_tools,
    _summarize_args,
    _trim_history_for_model,
)

__all__ = [
    "intent_router_node",
    "chat_node",
    "coder_chat_node",
    "python_tool_node",
    # Hints (re-exported for backward compat)
    "CORE_HINTS",
    "_CODER_PLAN_DIRECTIVE",
    "_PLANNING_HINT",
    "_TOOL_OUTPUT_HINT",
    "_TOOL_RESTRAINT_HINT",
    # Helpers (re-exported for tests)
    "_NEEDS_TOOL_RE",
    "_build_rag_block",
    "_emit_tool_call_event",
    "_filter_empty_tool_calls",
    "_is_small_model",
    "_maybe_trigger_tool_creation",
    "_message_requires_tools",
    "_stream_llm_with_tools",
    "_summarize_args",
    "_trim_history_for_model",
]
