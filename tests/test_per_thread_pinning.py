"""Tests for per-thread pinning (pinned = metadata flag, no cross-thread injection).

Verifies:
  - /pin marks the last AI message + preceding Human with pinned=True
  - /unpin <n> clears the Nth pinned message
  - /unpin_all clears all pins in the current thread
  - /list shows pinned messages from the CURRENT thread only
  - Creating a new thread does NOT carry over pins from other threads
  - The limit_messages reducer preserves pinned messages
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from chatbot import Chatbot
from graph.state import VedState, limit_messages


def _make_bot_with_thread(messages=None, thread_id="thr_a"):
    """Build a Chatbot-shaped object with just the bits pinning touches."""
    bot = Chatbot.__new__(Chatbot)
    # Attributes accessed by ChatbotCommandProcessor.handle_command.
    bot.mode = "standard"
    bot._hibernating = False
    bot._threads = {thread_id: {
        "id": thread_id,
        "title": "Test Thread",
        "created_at": 0.0,
        "messages": list(messages or []),
    }}
    bot._active_thread_id = thread_id
    bot.project_root = Path("/tmp")
    bot.threads_db_path = Path("/tmp/test_threads.json")
    return bot, bot._threads[thread_id]


# ---- pin_last_turn_in_active_thread ----

def test_pin_marks_last_ai_and_preceding_human():
    bot, thread = _make_bot_with_thread([
        HumanMessage(content="h1"),
        AIMessage(content="a1"),
        HumanMessage(content="h2"),
        AIMessage(content="a2"),
    ])
    count = bot.pin_last_turn_in_active_thread()
    assert count == 2, "should pin the last AI + preceding Human"
    # Verify the flag is set on the right messages.
    assert thread["messages"][2].additional_kwargs.get("pinned") is True   # h2
    assert thread["messages"][3].additional_kwargs.get("pinned") is True   # a2
    # Earlier messages are untouched.
    assert "pinned" not in thread["messages"][0].additional_kwargs
    assert "pinned" not in thread["messages"][1].additional_kwargs


def test_pin_returns_zero_when_no_ai_message():
    bot, _ = _make_bot_with_thread([HumanMessage(content="only human")])
    assert bot.pin_last_turn_in_active_thread() == 0


def test_pin_returns_zero_for_empty_thread():
    bot, _ = _make_bot_with_thread([])
    assert bot.pin_last_turn_in_active_thread() == 0


# ---- unpin_in_active_thread ----

def test_unpin_clears_specific_pinned_message():
    bot, thread = _make_bot_with_thread([
        HumanMessage(content="h1"), AIMessage(content="a1"),
        HumanMessage(content="h2"), AIMessage(content="a2"),
    ])
    bot.pin_last_turn_in_active_thread()
    assert bot.get_pinned_messages_in_active_thread().__len__() == 2

    # Unpin the first one (the human).
    removed = bot.unpin_in_active_thread(1)
    assert removed == 1
    assert thread["messages"][2].additional_kwargs.get("pinned") is False
    # The AI is still pinned.
    assert thread["messages"][3].additional_kwargs.get("pinned") is True


def test_unpin_returns_zero_for_out_of_range_index():
    bot, _ = _make_bot_with_thread([HumanMessage(content="h"), AIMessage(content="a")])
    bot.pin_last_turn_in_active_thread()
    assert bot.unpin_in_active_thread(99) == 0
    assert bot.unpin_in_active_thread(0) == 0


# ---- unpin_all_in_active_thread ----

def test_unpin_all_clears_all_pinned_messages_in_thread():
    bot, thread = _make_bot_with_thread([
        HumanMessage(content="h1"), AIMessage(content="a1"),
        HumanMessage(content="h2"), AIMessage(content="a2"),
    ])
    # Pin both turns.
    bot.pin_last_turn_in_active_thread()
    # Reset and pin the first turn too — easiest: just call again after
    # mutating messages. Since the last is now pinned, call should be a
    # no-op. Instead, manually pin h1/a1:
    thread["messages"][0].additional_kwargs["pinned"] = True
    thread["messages"][1].additional_kwargs["pinned"] = True

    assert len(bot.get_pinned_messages_in_active_thread()) == 4
    cleared = bot.unpin_all_in_active_thread()
    assert cleared == 4
    assert bot.get_pinned_messages_in_active_thread() == []


def test_unpin_all_returns_zero_when_nothing_pinned():
    bot, _ = _make_bot_with_thread([HumanMessage(content="h"), AIMessage(content="a")])
    assert bot.unpin_all_in_active_thread() == 0


# ---- get_pinned_messages_in_active_thread ----

def test_get_pinned_returns_only_pinned_messages_in_order():
    bot, _ = _make_bot_with_thread([
        HumanMessage(content="h1"), AIMessage(content="a1"),
        HumanMessage(content="h2"), AIMessage(content="a2"),
    ])
    bot.pin_last_turn_in_active_thread()
    pinned = bot.get_pinned_messages_in_active_thread()
    assert len(pinned) == 2
    assert pinned[0].content == "h2"
    assert pinned[1].content == "a2"


# ---- cross-thread isolation ----

def test_pins_in_thread_a_do_not_appear_in_thread_b():
    """The whole point: new threads don't inherit pins from other threads."""
    bot, _ = _make_bot_with_thread([
        HumanMessage(content="thread A user"), AIMessage(content="thread A ai"),
    ], thread_id="thr_a")
    bot.pin_last_turn_in_active_thread()

    # Create a new thread (simulated by switching active_thread_id).
    bot._threads["thr_b"] = {
        "id": "thr_b", "title": "New Thread", "created_at": 0.0,
        "messages": [],
    }
    bot._active_thread_id = "thr_b"

    # The new thread's pin list is empty.
    assert bot.get_pinned_messages_in_active_thread() == []


# ---- /list, /unpin, /unpin_all command flow ----

def test_command_processor_list_returns_pinned():
    """Verify the /list slash command works end-to-end on the chatbot."""
    bot, _ = _make_bot_with_thread([
        HumanMessage(content="user q"), AIMessage(content="ai a"),
    ])
    bot.pin_last_turn_in_active_thread()
    # Chatbot already inherits from ChatbotCommandProcessor, so handle_command works directly.
    result = bot.handle_command("/list")
    assert "Pinned" in result
    assert "ai a" in result or "user q" in result


def test_command_processor_unpin_all_clears():
    bot, _ = _make_bot_with_thread([
        HumanMessage(content="q1"), AIMessage(content="a1"),
        HumanMessage(content="q2"), AIMessage(content="a2"),
    ])
    bot.pin_last_turn_in_active_thread()
    assert len(bot.get_pinned_messages_in_active_thread()) == 2
    result = bot.handle_command("/unpin_all")
    assert "Cleared" in result
    assert bot.get_pinned_messages_in_active_thread() == []


def test_command_processor_pin_returns_count():
    bot, _ = _make_bot_with_thread([
        HumanMessage(content="q"), AIMessage(content="a"),
    ])
    result = bot.handle_command("/pin")
    assert "Pinned" in result
    assert "2 message(s)" in result


# ---- limit_messages reducer preserves pinned ----

def test_limit_messages_keeps_pinned_messages():
    """The reducer already preserved pinned — verify it still does."""
    pinned = AIMessage(content="pinned", additional_kwargs={"pinned": True})
    normal = [HumanMessage(content=f"h{i}") for i in range(50)]
    msgs = [pinned] + normal
    out = limit_messages(left=[], right=msgs)
    # The pinned message must be in the output.
    contents = [m.content for m in out]
    assert "pinned" in contents
