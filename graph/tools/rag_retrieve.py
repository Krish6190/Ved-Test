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
    layer: Annotated[Optional[str], "Filter by chunk layer: 'sig' (signatures only), 'body' (bodies only), or None (both)."] = None,
    paths: Annotated[Optional[List[str]], "Restrict to chunks whose source path starts with one of these prefixes (e.g. ['voice/', 'src/']). None = no path filter."] = None,
    config: Annotated[RunnableConfig, "injected"] = None,
) -> str:
    """Pull relevant chunks from the thread's RAG store.

    Args:
      query: free-form search string (semantic match against stored chunks).
      scope: thread id; if None, uses the active thread from config.
      k: how many chunks to return (default 5).
      layer: optional layer filter — 'sig' for function/class signatures only
        (cheap navigation), 'body' for implementations only, None for both.
      paths: optional list of path prefixes to restrict results to (e.g.
        ['voice/'] for "only stuff in the voice folder"). Backward-compat:
        old entries have just a filename as source and won't match any
        directory prefix, so they're effectively excluded when paths is set.

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

    # Resolve the caller's mode from config (same place active_thread_id comes
    # from). Falls back to empty string if config doesn't carry it, which
    # disables the project fallback — Path A's chat calls never have mode
    # in their config so they correctly lose the project scope here.
    caller_mode = ""
    if config:
        try:
            conf = (config.get("configurable") or {}) if isinstance(config, dict) else {}
            caller_mode = (conf.get("state_mode") or conf.get("mode") or "").lower()
        except Exception:
            pass

    # Project-scope fallback: ONLY for coder mode. Path A (standard/turbo)
    # keeps retrieve_rag for thread-scoped past-chat memory but does NOT
    # fall back to the project indexer — codebase discovery in Path A goes
    # through search_files / read_file instead.
    if not chunks and thread_id and caller_mode == "coder":
        try:
            from graph.rag.rag_db import rag_db  # the LocalVectorDB instance
            project_results = rag_db.query_similarity(
                query.strip(), k=max(1, min(k, 20)), scope="project",
            ) or []
        except Exception:
            project_results = []
        if paths:
            prefixes = [p.replace("\\", "/").rstrip("/") + "/" for p in paths if p]
            def _matches(source: str) -> bool:
                src = (source or "").replace("\\", "/")
                return any(src.startswith(pref) or src == pref.rstrip("/")
                           for pref in prefixes)
            project_results = [c for c in project_results
                               if _matches(c.get("source", ""))]
        if project_results:
            for c in project_results:
                c["scope"] = "project"
            chunks = project_results

    # Client-side layer filtering. No-op until code_chunker is wired into
    # LocalVectorDB (Phase 2.2+2.4 wiring — chunks currently have no 'layer'
    # field, so this filter passes everything when layer=None and drops
    # everything when layer is set). Safe to keep in place.
    if layer is not None and chunks:
        filtered = [c for c in chunks if c.get("layer") == layer]
        chunks = filtered

    # Client-side path filtering. Restricts results to chunks whose source
    # path starts with one of the provided prefixes. Used for scoped queries
    # like "fix voice in voice folder" — only chunks from that folder match.
    if paths and chunks:
        prefixes = [p.replace("\\", "/").rstrip("/") + "/" for p in paths if p]
        if prefixes:
            def _matches_prefix(source: str) -> bool:
                src = (source or "").replace("\\", "/")
                return any(src.startswith(pref) or src == pref.rstrip("/") for pref in prefixes)
            chunks = [c for c in chunks if _matches_prefix(c.get("source", ""))]

    if not chunks:
        suffix = ""
        if layer:
            suffix += f" (layer={layer})"
        if paths:
            suffix += f" (paths={paths})"
        return (
            f"No RAG chunks found matching '{query}'"
            + (f" in thread {thread_id[:8]}" if thread_id else ".")
            + suffix
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
