"""Heuristic intent router — picks Path A / B / C from the user's text.

Pure regex router; no LLM call. Deterministic, fast, easy to debug.
"""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage
from graph.state import VedState

def intent_router_node(state: VedState, get_llm) -> dict:
    """Route the user message to Path A (chat + RAG + web), B (content generation),
    or C (tool execution).

    Rules (evaluated top-down, first match wins):
      1. Self-healing phrases ("fix yourself" etc.) → C + self_healing=True
      2. Explicit override:  "... path A|B|C"  anywhere in the message
      3. Slash commands: any "/..."           → A (commands handled separately)
      4. Tool execution: "/run ..." or "execute ..."  → C
      5. File/tool verbs: "read this file", "open this file", "perform this"  → A
      6. Word-count spec:  "<N> word(s)/paragraph(s)/page(s)..."  → B
      7. Generation verb at start: "write/draft/compose/generate/create/make/build ..."  → B
      8. Generation phrases anywhere: "write me", "essay on", "blog post", etc.  → B
      9. Default: A
    """
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    last_user_text = user_messages[-1].content.strip() if user_messages else ""
    lower_text = last_user_text.lower()
    self_heal_phrases = (
        "self heal", "self-heal", "self-healing",
        "heal yourself", "heal itself",
        "fix yourself", "fix itself", "fix your code", "fix your own code",
        "repair yourself", "repair your code", "repair itself",
        "self repair", "self-repair",
    )
    if any(p in lower_text for p in self_heal_phrases):
        return {"route_intent": "C", "self_healing": True}
    override = re.search(r"\b(?:use\s+)?path\s+([abc])\b", lower_text)
    if override:
        return {"route_intent": override.group(1).upper()}
    if lower_text.startswith("/"):
        return {"route_intent": "A"}
    if lower_text.startswith("/run") or lower_text.startswith("execute"):
        return {"route_intent": "C"}
    if not re.search(r"\b(what|how|why|when|where|who|which)\b", lower_text):
        if re.match(r"^(read|open|edit|search|find|modify|update)\s+", lower_text):
            return {"route_intent": "A"}
    if re.search(r"\b\d+\s+(word|words|paragraph|paragraphs|page|pages|line|lines|char|chars|character|characters)\b", lower_text):
        return {"route_intent": "B"}
    if re.match(r"^(write|draft|compose|generate|create|make|build|craft|author|produce)\s+", lower_text):
        return {"route_intent": "B"}
    generation_phrases = (
        "write me", "write a", "write an", "draft a", "draft an",
        "compose a", "essay on", "essay about", "blog post",
        "story about", "letter to", "report on", "summary of", "summarize",
    )
    if any(p in lower_text for p in generation_phrases):
        return {"route_intent": "B"}
    return {"route_intent": "A"}
