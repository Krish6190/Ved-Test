"""Shared streaming helpers for graph nodes.

These helpers consolidate text-only LLM streaming that the planner and
executor nodes previously duplicated. Tool-call streaming (which also
parses `tool_call_chunks`) is intentionally left in the individual nodes
because the nodes diverge on how they handle tool execution.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

# Collapse runs of 3+ newlines to exactly two newlines. Anything shorter
# (1 or 2 newlines) is preserved verbatim because it's a valid paragraph
# or list break. Also collapse runs of 3+ spaces. Without this filter,
# small models stream back tokens like "\n\n\n\n\n" which renders as a
# visible flood of blank lines in the UI.
#
# These regexes live in this module (not in the UI renderer) so they are
# importable from headless test environments without pulling in tkinter.
# The renderer imports them from here.
_RUNAWAY_NEWLINES_RE = re.compile(r"\n{3,}")
_RUNAWAY_SPACES_RE = re.compile(r" {4,}")


def _strip_leading_blank_lines(chunk: str, existing_tail: str) -> str:
    """Cross-chunk clamp: strip leading newlines from `chunk` when the
    already-rendered text (`existing_tail`) ends with one or more
    newlines, so the combined output never exceeds a single empty line.

    Used by the UI renderer to prevent two streamed chunks from stacking
    into two blank lines in the chat panel. Returns the (possibly
    trimmed) chunk; never returns an empty string when the input had
    non-whitespace content.
    """
    if not chunk:
        return chunk
    if not existing_tail:
        return chunk
    if not existing_tail.endswith("\n"):
        return chunk
    if not chunk.startswith("\n"):
        return chunk
    stripped = chunk.lstrip("\n")
    return stripped if stripped else ""


def _clean_chunk(chunk: str) -> Optional[str]:
    """Normalize a streamed text token. Returns None if the chunk should
    be dropped entirely.

    Rules (conservative \u2014 readability is the priority):
      - Truly empty chunks (None / "") are dropped (nothing to show).
      - Whitespace-only chunks (\n, \n\n, "  ", etc.) are PRESERVED so
        the UI renders normal paragraph breaks. Without this, output
        collapses into a single unreadable paragraph because the model
        streams "\n" as its own chunk between text segments.
      - Runs of 3+ newlines collapse to 2 (the ONLY whitespace anomaly
        we collapse). Standard \n and \n\n pass through unchanged.
      - Runs of 4+ spaces collapse to 1 (kills accidental indent floods).

    Applied at every text-streaming call site so the UI never has to do
    post-processing.
    """
    if chunk is None:
        return None
    if not isinstance(chunk, str):
        chunk = str(chunk)
    if not chunk:
        return None
    cleaned = _RUNAWAY_NEWLINES_RE.sub("\n\n", chunk)
    cleaned = _RUNAWAY_SPACES_RE.sub(" ", cleaned)
    return cleaned


def _stream_text(
    llm: Any,
    messages: List[Any],
    token_queue: Optional[Any] = None,
) -> str:
    """Stream an LLM response as text tokens.

    Accumulates `chunk.content` into the returned string and pushes each
    non-empty token to `token_queue` (if provided). Designed for nodes
    that do not need tool-call parsing — the dual-role Planner (Thinker)
    and Executor (Typist) phases both fit this pattern.

    Args:
        llm: A LangChain chat model with a `.stream(messages)` method.
        messages: The message list to send to the model.
        token_queue: Optional queue-like object; tokens are pushed via
            `token_queue.put(c)`. Failures are swallowed silently.

    Returns:
        The full accumulated text content (empty string on stream failure).
    """
    full_content = ""
    try:
        for chunk in llm.stream(messages):
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
    except Exception:
        return full_content
    return full_content
