"""Dual-source RAG mixer: thread-local + global, with priority fallback.

Exposes:
  - GLOBAL_SCOPE               sentinel for chunks not tied to a thread
  - THREAD_RAG_PRIORITY_RATIO  fraction of slots from thread-local when both return
  - DEFAULT_RAG_K              default top-k for Path A
  - PATH_B_DIVERSITY           per-pass (k, lambda_mult) for Path B
  - retrieve_context(query, current_thread_id, k, ratio, lambda_mult)
  - _format_rag_block(chunks)  human-readable block for prompt injection
"""
from typing import Optional

# Sentinel scope for chunks that are not tied to any specific thread.
GLOBAL_SCOPE = "__GLOBAL__"

# 75% thread-local, 25% global — the user's specified mix ratio.
THREAD_RAG_PRIORITY_RATIO = 0.75

DEFAULT_RAG_K = 5

# Per-pass diversity for Path B: more candidates + lower lambda each retry.
PATH_B_DIVERSITY = [
    {"k": 3, "lambda_mult": 0.7},   # pass 0 — relevant
    {"k": 5, "lambda_mult": 0.3},   # pass 1 — more diverse
    {"k": 8, "lambda_mult": 0.0},   # pass 2 — max diversity
]


def _format_rag_block(chunks):
    """Format a list of retrieved chunks into a single labeled block for prompt injection.

    Each chunk must have keys: "content", "source", and "scope" (thread | global).
    Returns "" if chunks is empty.
    """
    if not chunks:
        return ""
    lines = ["[RAG Context]"]
    for i, c in enumerate(chunks, 1):
        scope = c.get("scope", "unknown")
        source = c.get("source", "unknown")
        content = c.get("content", "")
        lines.append(f"({i}) [{scope}] {source}\n{content}")
    return "\n\n".join(lines)


def retrieve_context(
    query_text: str,
    current_thread_id: Optional[str] = None,
    k: int = DEFAULT_RAG_K,
    ratio: float = THREAD_RAG_PRIORITY_RATIO,
    lambda_mult: float = 0.5,
):
    """Retrieve context from both thread-local and global sources.

    Allocation rules:
      - thread_slots = round(k * ratio)
      - global_slots = k - thread_slots

    Behavior:
      - If both return results: return thread + global (deduped by content; thread wins).
      - If only thread: re-query thread with full k.
      - If only global: re-query global with full k.
      - If neither: return [].
    """
    from graph.rag import rag_db  # lazy import: avoids loading Ollama at module-import time

    if not query_text:
        return []

    thread_slots = round(k * ratio)
    global_slots = k - thread_slots

    thread_results = []
    if current_thread_id and thread_slots > 0:
        try:
            thread_results = rag_db.query_similarity(
                query_text,
                k=thread_slots,
                lambda_mult=lambda_mult,
                scope=current_thread_id,
            ) or []
        except Exception:
            thread_results = []

    global_results = []
    if global_slots > 0:
        try:
            global_results = rag_db.query_similarity(
                query_text,
                k=global_slots,
                lambda_mult=lambda_mult,
                scope=GLOBAL_SCOPE,
            ) or []
        except Exception:
            global_results = []

    # Tag with display scope for the prompt formatter.
    for c in thread_results:
        c["scope"] = "thread"
    for c in global_results:
        c["scope"] = "global"

    if thread_results and global_results:
        thread_contents = {c.get("content") for c in thread_results}
        deduped_global = [c for c in global_results if c.get("content") not in thread_contents]
        combined = thread_results + deduped_global
        return combined[:k]

    if thread_results and thread_slots < k:
        try:
            more = rag_db.query_similarity(
                query_text, k=k, lambda_mult=lambda_mult, scope=current_thread_id,
            ) or []
            for c in more:
                c["scope"] = "thread"
            return more
        except Exception:
            return thread_results
    if thread_results:
        return thread_results

    if global_results and global_slots < k:
        try:
            more = rag_db.query_similarity(
                query_text, k=k, lambda_mult=lambda_mult, scope=GLOBAL_SCOPE,
            ) or []
            for c in more:
                c["scope"] = "global"
            return more
        except Exception:
            return global_results
    if global_results:
        return global_results

    return []
