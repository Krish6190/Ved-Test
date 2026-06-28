"""Per-turn SystemMessage hints injected into the LLM stream.

NOTE: Most tool-use discipline (restraint, output-verbatim, anti-hallucination)
is now baked into the Modelfile.system block at model load time, so it
doesn't need to be re-sent every turn. The hints here are intentionally
minimal — just the per-turn fresh-question reminder that fights
in-context history contamination.

  - _FRESH_QUESTION_HINT: small per-turn nudge that resets the model's
    attention away from prior tool calls in the conversation history.
    Particularly important for small models (llama3.2:3b) that
    pattern-match off the immediate context.

  - _PLANNING_HINT: optional, only injected by nodes that explicitly
    need the planning workflow.

  - _TOOL_RESTRAINT_HINT / _TOOL_OUTPUT_HINT: kept as no-op stubs for
    backward compatibility with tests / external imports. The actual
    content now lives in Modelfile.standard / Modelfile.turbo /
    Modelfile.coder.
"""
from langchain_core.messages import SystemMessage


_FRESH_QUESTION_HINT = SystemMessage(content=(
    "Treat this user message as a FRESH, INDEPENDENT request. "
    "Tool calls in earlier turns do NOT imply tool calls now — only call "
    "a tool if THIS message explicitly requires it. If unsure, respond with text."
))

_PLANNING_HINT = SystemMessage(content=(
    "For multi-step tasks: state a short plan first, then execute step by step."
))

# Backward-compat stubs. Real content lives in the Modelfile.
_TOOL_RESTRAINT_HINT = SystemMessage(content="(see Modelfile)")
_TOOL_OUTPUT_HINT = SystemMessage(content="(see Modelfile)")
_CODER_PLAN_DIRECTIVE = SystemMessage(content="(see Modelfile.coder)")
DEFAULT_HINTS = (_FRESH_QUESTION_HINT,)
CORE_HINTS = DEFAULT_HINTS