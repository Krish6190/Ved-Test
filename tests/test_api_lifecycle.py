"""Unit tests for api.lifecycle. Patches Chatbot so no Ollama calls happen."""
import threading
from unittest.mock import patch, MagicMock

from api import lifecycle


def test_get_chatbot_is_lazy_and_singleton():
    """First call constructs, second call returns same instance."""
    lifecycle.reset_for_tests()
    fake = MagicMock(name="ChatbotInstance")
    with patch("chatbot.Chatbot", return_value=fake) as mock_ctor:
        # First call → constructs
        a = lifecycle.get_chatbot()
        # Second call → returns cached instance, no new construction
        b = lifecycle.get_chatbot()
    assert a is b is fake
    assert mock_ctor.call_count == 1


def test_register_and_resolve_approval():
    """register_approval returns an Event; resolve_approval sets it."""
    lifecycle.reset_for_tests()
    event = lifecycle.register_approval("sess-1")
    assert isinstance(event, threading.Event)
    assert not event.is_set()
    # Resolve from a different thread to mimic the HTTP route handler.
    result = []
    def waiter():
        event.wait(timeout=1.0)
        result.append(event.is_set())
    t = threading.Thread(target=waiter)
    t.start()
    found = lifecycle.resolve_approval("sess-1", approved=True)
    t.join(timeout=1.0)
    assert found is True
    assert result == [True]
    # Resolving again returns False (already removed).
    assert lifecycle.resolve_approval("sess-1", approved=True) is False


def test_resolve_unknown_session_returns_false():
    lifecycle.reset_for_tests()
    assert lifecycle.resolve_approval("does-not-exist", approved=False) is False
    # And discard_approval on unknown id is a no-op (does not raise).
    lifecycle.discard_approval("does-not-exist")


# ---- Tool-creation proposal registry ----

def test_register_and_resolve_tool_proposal():
    lifecycle.reset_for_tests()
    event = lifecycle.register_tool_proposal("tool-1")
    assert isinstance(event, threading.Event)
    assert not event.is_set()
    # Resolve from a different thread to mimic the HTTP route handler.
    result = []
    def waiter():
        event.wait(timeout=1.0)
        result.append(event.is_set())
    t = threading.Thread(target=waiter)
    t.start()
    found = lifecycle.resolve_tool_proposal("tool-1", approved=True)
    t.join(timeout=1.0)
    assert found is True
    assert result == [True]
    # Resolving again returns False (already removed).
    assert lifecycle.resolve_tool_proposal("tool-1", approved=True) is False


def test_resolve_unknown_tool_proposal_returns_false():
    lifecycle.reset_for_tests()
    assert lifecycle.resolve_tool_proposal("does-not-exist", approved=False) is False
    lifecycle.discard_tool_proposal("does-not-exist")  # no-op


def test_tool_proposal_registry_isolated_from_approval_registry():
    """Tool proposals and content approvals use separate registries."""
    lifecycle.reset_for_tests()
    lifecycle.register_approval("content-1")
    lifecycle.register_tool_proposal("tool-1")
    assert lifecycle.resolve_approval("content-1", approved=True) is True
    # Resolving the content approval doesn't affect the tool proposal.
    assert lifecycle.resolve_tool_proposal("tool-1", approved=False) is True
