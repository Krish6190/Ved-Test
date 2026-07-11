"""Unit tests for the CONFIRMATION keyword matching helper (Bug #2c).

The helper `_matches_any` in `voice/audio_processors.py` is a pure module-
level function (no I/O, no audio device, no tkinter-bound state) that
tokenizes a transcribed utterance and decides whether any keyword is
present. We import it directly from `voice.audio_processors` and exercise
it against the expanded `confirm_keywords` list (re-declared locally here
to avoid coupling the test to a specific list value, and to keep the
keyword vocabulary documented next to the assertions).

Run with: `python -m pytest tests/test_confirm_keywords.py -v`
"""
import sys
from pathlib import Path
from voice.audio_processors import _matches_any


# Local copy of the production keyword list. Kept in sync with the
# expansion in `voice/audio_processors.py:_process_confirmation_logic`.
# If the prod list changes, update this constant and re-run.
CONFIRM_KEYWORDS = [
    "yes", "yeah", "yep", "yup", "ok", "okay", "sure",
    "send", "submit", "execute", "run", "go", "do it", "enter", "correct",
]

CANCEL_KEYWORDS = ["no", "wrong", "stop", "cancel", "don't"]


# ---------------------------------------------------------------------------
# Positive matches: each keyword should confirm on its own.
# ---------------------------------------------------------------------------

def test_yes_confirms():
    assert _matches_any("yes", CONFIRM_KEYWORDS) is True


def test_yeah_confirms():
    assert _matches_any("yeah", CONFIRM_KEYWORDS) is True


def test_yep_confirms():
    assert _matches_any("yep", CONFIRM_KEYWORDS) is True


def test_ok_confirms():
    assert _matches_any("ok", CONFIRM_KEYWORDS) is True


def test_okay_confirms():
    assert _matches_any("okay", CONFIRM_KEYWORDS) is True


def test_sure_confirms():
    assert _matches_any("sure", CONFIRM_KEYWORDS) is True


# ---------------------------------------------------------------------------
# Multi-word: a token from the list appears alongside other tokens.
# ---------------------------------------------------------------------------

def test_send_it_confirms():
    # `send` is in the list; `it` is filler.
    assert _matches_any("send it", CONFIRM_KEYWORDS) is True


def test_yeah_send_confirms():
    # Both tokens appear in the list; either should match.
    assert _matches_any("yeah send", CONFIRM_KEYWORDS) is True


# ---------------------------------------------------------------------------
# Negative matches: words that must NOT trigger confirmation.
# ---------------------------------------------------------------------------

def test_no_does_not_confirm():
    # `no` is a cancel keyword, not a confirm keyword.
    assert _matches_any("no", CONFIRM_KEYWORDS) is False


def test_cancel_does_not_confirm():
    # `cancel` is a cancel keyword, not a confirm keyword.
    assert _matches_any("cancel", CONFIRM_KEYWORDS) is False


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------

def test_empty_text_does_not_confirm():
    assert _matches_any("", CONFIRM_KEYWORDS) is False


def test_trailing_punctuation_stripped_then_confirms():
    # Production strips ',' and '.' before splitting, so `yeah.` -> `yeah`.
    assert _matches_any("yeah.", CONFIRM_KEYWORDS) is True


# ---------------------------------------------------------------------------
# Cross-check: the cancel-keyword helper rejects the same strings above.
# ---------------------------------------------------------------------------

def test_cancel_keywords_match_cancel_inputs():
    assert _matches_any("no", CANCEL_KEYWORDS) is True
    assert _matches_any("cancel", CANCEL_KEYWORDS) is True


def test_cancel_keywords_reject_confirm_inputs():
    assert _matches_any("yes", CANCEL_KEYWORDS) is False
    assert _matches_any("send", CANCEL_KEYWORDS) is False


# ---------------------------------------------------------------------------
# Regression: Fix #1 (Bug #1) — command responses must persist into the
# active thread so they survive GUI re-renders. Typing `/mode turbo` while
# in coder mode produces a rejection string from the command processor.
# Before the fix, `respond()` returned the string but never appended an
# AIMessage, so the message vanished on the next keystroke.
# ---------------------------------------------------------------------------

# Make sure the project root is on sys.path so `chatbot` imports cleanly
# even when pytest is invoked from a different cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_minimal_coder_bot(thread_id="thr_reject"):
    """Build a Chatbot instance that bypasses the heavy __init__.

    Only the attributes that `respond()` and the command processor's
    `handle_command()` actually touch are populated. No Modelfile parsing,
    no Ollama wiring, no graph build, no disk I/O beyond a tmp stub.
    """
    from chatbot import Chatbot
    from langchain_core.messages import AIMessage  # noqa: F401  (imported for assertions below)

    bot = Chatbot.__new__(Chatbot)
    # Attributes read by ChatbotCommandProcessor.handle_command.
    bot.mode = "coder"
    bot._hibernating = False
    # Attributes read by Chatbot.get_active_thread / _save_threads.
    bot._threads = {
        thread_id: {
            "id": thread_id,
            "title": "Test Thread",
            "created_at": 0.0,
            "messages": [],
        }
    }
    bot._active_thread_id = thread_id
    bot.project_root = Path("/tmp")
    bot.threads_db_path = Path("/tmp/_test_reject_threads.json")
    return bot


def test_respond_persists_mode_rejection_to_active_thread():
    """`/mode turbo` while in coder mode must append the rejection as an
    AIMessage to the active thread AND persist via _save_threads().
    """
    from langchain_core.messages import AIMessage

    bot = _build_minimal_coder_bot()

    # Spy on _save_threads so the test doesn't touch the real filesystem.
    save_calls = {"count": 0}

    def _spy_save():
        save_calls["count"] += 1

    bot._save_threads = _spy_save

    rejection_text = (
        "Command Rejected: Hardware configuration adjustments are "
        "blocked while coder mode is active."
    )

    result = bot.respond("/mode turbo")

    # 1. The function still returns the rejection string (existing contract).
    assert result == rejection_text

    # 2. The active thread now contains exactly one AIMessage whose
    #    content matches the rejection text — this is the persistence the
    #    GUI re-render path relies on.
    active = bot.get_active_thread()
    ai_contents = [m.content for m in active["messages"] if isinstance(m, AIMessage)]
    assert rejection_text in ai_contents, (
        f"Active thread did not persist the command response. "
        f"AIMessage contents: {ai_contents!r}"
    )

    # 3. _save_threads() was invoked so the new message survives a restart.
    assert save_calls["count"] >= 1, "_save_threads() was not called"


def test_respond_does_not_persist_when_no_command_response():
    """If `handle_command` returns None (not a command), no AIMessage is
    appended and no save happens on the command path. We stub handle_command
    to return None and verify the active thread is untouched.
    """
    from langchain_core.messages import AIMessage

    bot = _build_minimal_coder_bot()

    def _stub_handle_command(_msg):
        return None

    bot.handle_command = _stub_handle_command
    save_calls = {"count": 0}
    bot._save_threads = lambda: save_calls.__setitem__("count", save_calls["count"] + 1)

    # respond() will proceed past the command branch; we don't care about
    # the downstream return value here, only that the command branch did
    # NOT touch the thread. We catch any downstream exceptions so the test
    # is purely about the command-response path.
    try:
        bot.respond("hello there")
    except Exception:
        pass

    active = bot.get_active_thread()
    assert active["messages"] == [], (
        f"Thread was mutated on the command-response path even though "
        f"handle_command returned None. Messages: {active['messages']!r}"
    )
    assert save_calls["count"] == 0, (
        "_save_threads() was called on the command-response path even "
        "though handle_command returned None."
    )
