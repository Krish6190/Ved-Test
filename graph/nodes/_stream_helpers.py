"""Shared streaming helpers for graph nodes.

These helpers consolidate text-only LLM streaming that the planner and
executor nodes previously duplicated. Tool-call streaming (which also
parses `tool_call_chunks`) is intentionally left in the individual nodes
because the nodes diverge on how they handle tool execution.
"""
from __future__ import annotations

from typing import Any, List, Optional


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
                c = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_content += c
                if token_queue is not None:
                    try:
                        token_queue.put(c)
                    except Exception:
                        pass
    except Exception:
        return full_content
    return full_content
