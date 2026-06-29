"""Graph node implementations, split by responsibility.

Public surface (re-exported here so `from graph.nodes import X` still works):

  - intent_router_node    - heuristic routing (Path A or B)
  - chat_node             - Path A: chat + RAG + tools
  - coder_chat_node       - coder mode: coding assistant + tools
  - planner_node          - planner role: NO tools, parses CREATE_PLAN / DIRECT_ANSWER / etc.
  - executor_node         - executor role: runs ONE plan chunk with full VED_TOOLS

(Path C was removed - /run now flows through Path A's planner.)

Helpers (re-exported for test introspection):

  - _PLANNING_HINT, _TOOL_RESTRAINT_HINT, _TOOL_OUTPUT_HINT, _CODER_PLAN_DIRECTIVE,
    _FRESH_QUESTION_HINT
  - _summarize_args, _stream_llm_with_tools, _emit_tool_call_event
  - _build_rag_block, _maybe_trigger_tool_creation

Module layout:

  _hints.py      - SystemMessage prompts
  _helpers.py    - shared streaming / tool-call accounting / cross-mode trigger
  intent_router.py
  chat.py
  coder.py
  planner.py
  executor.py
"""
from __future__ import annotations

# Public node functions (consumed by graph/__init__.py for build_graph).
from graph.nodes.intent_router import intent_router_node
from graph.nodes.chat import chat_node
from graph.nodes.coder import coder_chat_node
from graph.nodes.planner import planner_node
from graph.nodes.executor import executor_node

# Re-export hints for backward compat with existing tests / external code.
from graph.nodes._hints import (
    _CODER_PLAN_DIRECTIVE,
    _FRESH_QUESTION_HINT,
    _PLANNING_HINT,
    _TOOL_OUTPUT_HINT,
    _TOOL_RESTRAINT_HINT,
)

# Re-export helpers for unit tests.
from graph.nodes._helpers import (
    _NEEDS_TOOL_RE,
    _build_rag_block,
    _emit_tool_call_event,
    _maybe_trigger_tool_creation,
    _stream_llm_with_tools,
    _summarize_args,
)

__all__ = [
    # Public nodes
    "intent_router_node",
    "chat_node",
    "coder_chat_node",
    "planner_node",
    "executor_node",
    # Hints
    "_FRESH_QUESTION_HINT",
    "_CODER_PLAN_DIRECTIVE",
    "_PLANNING_HINT",
    "_TOOL_OUTPUT_HINT",
    "_TOOL_RESTRAINT_HINT",
    # Helpers
    "_NEEDS_TOOL_RE",
    "_build_rag_block",
    "_emit_tool_call_event",
    "_maybe_trigger_tool_creation",
    "_stream_llm_with_tools",
    "_summarize_args",
]
