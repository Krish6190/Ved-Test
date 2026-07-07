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


# ---- Pattern tables ----

# Self-healing phrases still set a flag so tools can restrict their scope,
# but they no longer change the route (everything is Path A now).
_SELF_HEAL_PHRASES = (
    "self heal", "self-heal", "self-healing",
    "heal yourself", "heal itself",
    "fix yourself", "fix itself", "fix your code", "fix your own code",
    "repair yourself", "repair your code", "repair itself",
    "self repair", "self-repair",
)

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
# Narrowed: dropped write*/draft*/compose* generic prefixes (too broad),
# dropped summary/summarize (those need file-read tools — Path A handles),
# added poem/speech as clearly-prose signals.
_GENERATION_PHRASES = (
    "essay on", "essay about", "blog post",
    "story about", "letter to", "report on",
    "poem about", "speech about",
)

# Explicit user override. Only A and B are valid now; C falls back to A.
_EXPLICIT_PATH_RE = re.compile(r"\b(?:use\s+)?path\s+([ab])\b", re.IGNORECASE)


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
    """
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""
    lower_text = last_user_text.lower()

    # 1. Self-healing flag (does not change route).
    self_healing = any(p in lower_text for p in _SELF_HEAL_PHRASES)

    # 2. Explicit override. Only A and B valid now; C falls back to A.
    override = _EXPLICIT_PATH_RE.search(lower_text)
    if override:
        return {
            "route_intent": override.group(1).upper(),
            "self_healing": self_healing,
        }

    # 3-5. Content-generation signals -> Path B (the multi-pass draft pipeline).
    # Length spec alone is too broad (any "5 paragraphs" hit B even with
    # non-prose verbs). Require the length spec to be paired with a prose
    # verb OR a content-gen phrase; otherwise fall through to Path A.
    has_length_spec = bool(_LENGTH_SPEC_RE.search(lower_text))
    has_prose_verb = bool(_GENERATION_VERB_RE.match(lower_text))
    has_content_phrase = any(p in lower_text for p in _GENERATION_PHRASES)
    if has_length_spec and (has_prose_verb or has_content_phrase):
        return {"route_intent": "B", "self_healing": self_healing}
    if has_prose_verb:
        return {"route_intent": "B", "self_healing": self_healing}
    if has_content_phrase:
        return {"route_intent": "B", "self_healing": self_healing}

    # 6. Slash commands route to A as a fallback. (command_processor.py
    # actually handles most slash commands before they reach the graph,
    # so this branch is rarely hit, but it's defensive.)
    if lower_text.startswith("/"):
        return {"route_intent": "A", "self_healing": self_healing}

    # 7. Default -> A. The planner will decide plan vs direct answer;
    #    the executor will call tools if needed.
    return {"route_intent": "A", "self_healing": self_healing}