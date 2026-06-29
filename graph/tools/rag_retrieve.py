"""Explicit RAG retrieval tool.

Lets the LLM pull full content from the thread's RAG store on demand,
rather than relying solely on the auto-injected context block.

Use cases:
  - User says "show me the full analysis you just did" → LLM calls
    `retrieve_rag("the analysis")` to recover the compressed content
  - User references something from earlier in the conversation that's
    no longer in the visible context → LLM calls retrieve_rag to fetch it

The auto-injection in `chat_node._build_rag_block` still runs on every
turn (so the LLM usually sees top-k relevant chunks without asking).
This tool is the LLM's escape hatch when it needs to pull more.
"""
from __future__ import annotations

from typing import Annotated, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


_MAX_CHARS_PER_CHUNK = 1500  # truncate retrieved chunks for prompt size


# Module-level references to the lazy-loaded RAG functions. Tests can patch
# these directly. Set to None if the imports below fail (e.g., embedding
# pipeline unavailable in this environment).
retrieve_context = None  # type: ignore[assignment]
_format_rag_block = None  # type: ignore[assignment]
_rag_import_error: Optional[str] = None

try:
    from graph.rag.mixer import retrieve_context as _retrieve_context_imported
    from graph.rag.mixer import _format_rag_block as _format_rag_block_imported
    retrieve_context = _retrieve_context_imported
    _format_rag_block = _format_rag_block_imported
except Exception as _exc:
    _rag_import_error = str(_exc)


def _truncate(text: str, limit: int = _MAX_CHARS_PER_CHUNK) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated at {limit} chars]"


@tool
def retrieve_rag(
    query: str,
    scope: Annotated[Optional[str], "Thread id to search. None = active thread."] = None,
    k: Annotated[int, "Number of chunks to retrieve (default 5)."] = 5,
    config: Annotated[RunnableConfig, "injected"] = None,
) -> str:
    """Pull relevant chunks from the thread's RAG store.

    Args:
      query: free-form search string (semantic match against stored chunks).
      scope: thread id; if None, uses the active thread from config.
      k: how many chunks to return (default 5).

    Returns:
      Formatted string of retrieved chunks (truncated per chunk). If no
      chunks match, returns an explanatory message.
    """
    if not query or not query.strip():
        return "ERROR: retrieve_rag requires a non-empty query."

    if retrieve_context is None:
        return f"ERROR: RAG stack unavailable: {_rag_import_error or 'unknown'}"

    # Resolve scope: prefer explicit, fall back to active_thread_id from config.
    thread_id = scope
    if not thread_id and config:
        try:
            conf = (config.get("configurable") or {}) if isinstance(config, dict) else {}
            thread_id = conf.get("active_thread_id")
        except Exception:
            pass

    try:
        chunks = retrieve_context(query.strip(), thread_id, k=max(1, min(k, 20)))
    except Exception as exc:
        return f"ERROR: retrieve_rag failed: {exc}"

    if not chunks:
        return (
            f"No RAG chunks found matching '{query}'"
            + (f" in thread {thread_id[:8]}" if thread_id else ".")
        )

    # Format the chunks. Reuse the project standard formatter so the LLM sees
    # consistent citations. Then truncate per-chunk to keep prompt manageable.
    if _format_rag_block is not None:
        try:
            formatted = _format_rag_block(chunks)
        except Exception:
            formatted = _fallback_format(chunks)
    else:
        formatted = _fallback_format(chunks)

    return (
        f"Retrieved {len(chunks)} chunk(s) from thread RAG"
        + (f" for '{query}'" if thread_id else f" matching '{query}'")
        + ":\n\n" + formatted
    )


def _fallback_format(chunks: list) -> str:
    """Inline chunk formatter used when _format_rag_block is unavailable."""
    return "\n\n---\n\n".join(
        f"[source: {c.get('source', '?')}]\n{_truncate(c.get('content', ''))}"
        for c in chunks
    )
