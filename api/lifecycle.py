"""Singleton lifecycle for the FastAPI server.

Holds the single Chatbot instance (lazy) and the registry of in-flight
approval sessions used by /chat (SSE) and /chat/approval.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

# Module-level singletons. Populated on first call to get_chatbot().
_chatbot = None
_chatbot_lock = threading.Lock()

# Per-session approval events. SSE handlers register an event here when
# the content pipeline emits an approval_request; the /chat/approval
# route sets the event to unblock the generator.
_pending_approvals: Dict[str, threading.Event] = {}
_pending_lock = threading.Lock()


def get_chatbot():
    """Return the singleton Chatbot, constructing it on first call.

    Lazy so that `uvicorn api.server:app` boots instantly even when Ollama
    is not running. Thread-safe via a lock around the assignment.
    """
    global _chatbot
    if _chatbot is None:
        with _chatbot_lock:
            if _chatbot is None:
                from chatbot import Chatbot  # local import: avoids loading Ollama at import time
                _chatbot = Chatbot()
    return _chatbot


def register_approval(session_id: str) -> threading.Event:
    """Register a new approval session and return its Event.

    Called by the SSE handler when the pipeline emits approval_request.
    The caller must store the returned Event where the approval route
    can find it (we already store it in _pending_approvals — this
    function returns it for the caller's convenience).
    """
    event = threading.Event()
    with _pending_lock:
        _pending_approvals[session_id] = event
    return event


def resolve_approval(session_id: str, approved: bool) -> bool:
    """Resolve a pending approval. Returns True if found, False otherwise.

    Sets the event so the SSE generator resumes. Also calls
    `chatbot.submit_human_approval(approved)` if a chatbot instance exists
    so the chatbot's own _human_approval_state is updated.
    """
    with _pending_lock:
        event = _pending_approvals.pop(session_id, None)
    if event is None:
        return False
    event.set()
    try:
        get_chatbot().submit_human_approval(approved)
    except Exception:
        # Don't let a chatbot error mask the fact that we resolved the event.
        pass
    return True


def discard_approval(session_id: str) -> None:
    """Remove a session from the registry without resolving it.

    Called when the SSE stream ends (cleanly or by error) so the registry
    doesn't leak entries.
    """
    with _pending_lock:
        _pending_approvals.pop(session_id, None)


def reset_for_tests() -> None:
    """Reset module state. Only for use in tests."""
    global _chatbot
    with _chatbot_lock:
        _chatbot = None
    with _pending_lock:
        _pending_approvals.clear()
