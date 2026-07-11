"""Heuristic intent router — picks Path A (chat + planner + tools) or B (content gen).

Pure regex router; no LLM call. Deterministic, fast, easy to debug.

Path C (legacy /run -> python_tool_node) has been removed. All tool use,
including /run and "execute ...", now flows through Path A's planner:
the planner decides whether to DIRECT_ANSWER (simple) or CREATE_PLAN
(one or more chunks executed by the executor node with full VED_TOOLS).
"""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage
from graph.state import VedState
from graph.nodes._helpers import _TOOL_TRIGGER_RE

_SELF_HEAL_PHRASES = (
    "self heal", "self-heal", "self-healing",
    "heal yourself", "heal itself",
    "fix yourself", "fix itself", "fix your code", "fix your own code",
    "repair yourself", "repair your code", "repair itself",
    "self repair", "self-repair",
)
_PLANNING_SIGNAL_RE = re.compile(
    r"\b("
    r"plan|planning|"
    r"step[- ]by[- ]step|"
    r"break\s+down|breakdown|"
    r"implement|"
    r"build|"
    r"create\s+a|"
    r"multiple|"
    r"project|"
    r"complex|"
    r"edit|"
    r"modify|"
    r"write\s+code|"
    r"refactor|"
    r"several|"
    r"and\s+then|"
    r"first\b.*\bthen\b"
    r")\b",
    re.IGNORECASE,
)

_PLANNING_LONG_MESSAGE_LEN = 250

# Content-generation triggers. These produce multi-pass drafts (Path B).
_LENGTH_SPEC_RE = re.compile(
    r"\b\d+\s+(word|words|paragraph|paragraphs|page|pages|line|lines|"
    r"char|chars|character|characters)\b"
)
# Narrowed: only clear prose verbs. create/make/build/generate were removed
# because they commonly mean code/tool work, not prose drafting.
_GENERATION_VERB_RE = re.compile(
    r"^(write|draft|compose|author)\s+"
)
# dropped summary/summarize (those need file-read tools — Path A handles),
# added poem/speech as clearly-prose signals.
_GENERATION_PHRASES = (
    "essay on", "essay about", "blog post",
    "story about", "letter to", "report on",
    "poem about", "speech about",
)

# Explicit user override. Only A and B are valid now; C falls back to A.
_EXPLICIT_PATH_RE = re.compile(r"\b(?:use\s+)?path\s+([ab])\b", re.IGNORECASE)


def _compute_needs_planning(last_user_text: str) -> bool:
    """Return True when a Path-A message looks complex enough to warrant
    the planner-executor pipeline.

    Triggers:
      - The message text matches any planning / complex-task signal in
        `_PLANNING_SIGNAL_RE` (e.g. "implement", "step by step",
        "build", "refactor", "and then", "first...then").
      - OR the message is longer than `_PLANNING_LONG_MESSAGE_LEN` chars
        AND contains a tool-trigger verb (from `_TOOL_TRIGGER_RE`). Long
        tool-touching requests are nearly always multi-step.

    The planner itself still decides DIRECT_ANSWER vs CREATE_PLAN; this
    flag only steers Path A toward the planner node vs the simpler
    standalone_chat node.
    """
    lower = last_user_text.lower()
    if _PLANNING_SIGNAL_RE.search(lower):
        return True
    if (
        len(last_user_text) > _PLANNING_LONG_MESSAGE_LEN
        and bool(_TOOL_TRIGGER_RE.search(lower))
    ):
        return True
    return False


def intent_router_node(state: VedState, get_llm) -> dict:
    """Route to Path A (chat + planner + tools) or Path B (content generation).

    Rules (evaluated top-down, first match wins):

      1. Self-healing phrases     -> A + self_healing=True (flag only, not a route)
      2. Explicit "use path A|B"   -> that path (C overrides fall back to A)
      3. Word-count spec + prose signal -> B ("write 5 paragraphs", "200 word essay")
      4. Generation verb at start  -> B ("write a poem", "draft an email")
      5. Generation phrase         -> B ("essay on...", "blog post about...")
      (Length spec alone, or create/make/build/summarize, fall through to A.)
      6. Slash command "/..."      -> A (commands handled by command_processor
                                         before reaching the router in practice;
                                         this is a defensive fallback)
      7. Default                   -> A (chat / planner / executor handles it)

    For Path-A outcomes in non-coder mode, the returned dict also carries
    `needs_planning`: True when the message matches a planning signal or
    is a long tool-triggering request (see `_compute_needs_planning`).
    `_route_after_intent` uses this to send complex Path-A requests to
    `planner_node` instead of `standalone_chat_node`.
    """
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""
    lower_text = last_user_text.lower()

    # 1. Self-healing flag (does not change route).
    self_healing = any(p in lower_text for p in _SELF_HEAL_PHRASES)

    # 2. Explicit override. Only A and B valid now; C falls back to A.
    override = _EXPLICIT_PATH_RE.search(lower_text)
    if override:
        path = override.group(1).upper()
        result = {"route_intent": path, "self_healing": self_healing}
        if path == "A" and state.mode != "coder":
            result["needs_planning"] = _compute_needs_planning(last_user_text)
        return result

    has_length_spec = bool(_LENGTH_SPEC_RE.search(lower_text))
    has_prose_verb = bool(_GENERATION_VERB_RE.match(lower_text))
    has_content_phrase = any(p in lower_text for p in _GENERATION_PHRASES)
    if has_length_spec and (has_prose_verb or has_content_phrase):
        return {"route_intent": "B", "self_healing": self_healing}
    if has_prose_verb:
        return {"route_intent": "B", "self_healing": self_healing}
    if has_content_phrase:
        return {"route_intent": "B", "self_healing": self_healing}

    if lower_text.startswith("/"):
        result = {"route_intent": "A", "self_healing": self_healing}
        if state.mode != "coder":
            result["needs_planning"] = _compute_needs_planning(last_user_text)
        return result
    result = {"route_intent": "A", "self_healing": self_healing}
    if state.mode != "coder":
        result["needs_planning"] = _compute_needs_planning(last_user_text)
    return result