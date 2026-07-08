"""Tests for the tool-binding gating + empty-call filter.

Covers:
  - _message_requires_tools: casual greetings return False, tool-y messages return True
  - _filter_empty_tool_calls: drops empty-args hallucinated calls, keeps real ones
"""
from graph.nodes._helpers import (
    _filter_empty_tool_calls,
    _message_requires_tools,
)


# ---- _message_requires_tools ----

def test_casual_greetings_do_not_require_tools():
    """Small models fabricate tool calls when tools are bound on greetings.
    The gate must return False so we don't bind VED_TOOLS for these."""
    assert _message_requires_tools("hello") is False
    assert _message_requires_tools("hi") is False
    assert _message_requires_tools("how are you?") is False
    assert _message_requires_tools("what can you do?") is False
    assert _message_requires_tools("thanks!") is False
    assert _message_requires_tools("good morning") is False
    assert _message_requires_tools("") is False


def test_tool_requiring_messages_return_true():
    """Messages with file/code/app/search verbs return True."""
    assert _message_requires_tools("read foo.py") is True
    assert _message_requires_tools("open discord") is True
    assert _message_requires_tools("launch steam") is True
    assert _message_requires_tools("start firefox") is True
    assert _message_requires_tools("run this script") is True
    assert _message_requires_tools("execute the test") is True
    assert _message_requires_tools("find all .py files") is True
    assert _message_requires_tools("search for TODO") is True
    assert _message_requires_tools("edit the config") is True
    assert _message_requires_tools("delete the cache") is True
    assert _message_requires_tools("show me the file") is True
    assert _message_requires_tools("list the directory") is True


def test_knowledge_questions_do_not_require_tools():
    """Pure knowledge/opinion questions should NOT trigger tool binding."""
    assert _message_requires_tools("what is python?") is False
    assert _message_requires_tools("explain recursion") is False
    assert _message_requires_tools("why is the sky blue?") is False
    assert _message_requires_tools("tell me a joke") is False


def test_tool_trigger_is_case_insensitive():
    assert _message_requires_tools("READ foo.py") is True
    assert _message_requires_tools("Open Discord") is True
    assert _message_requires_tools("Hello") is False


# ---- _filter_empty_tool_calls ----

def test_filter_drops_empty_args_dict():
    """A tool call with no args at all is dropped (clearly hallucinated)."""
    calls = [{"id": "1", "name": "read_file", "args": {}}]
    assert _filter_empty_tool_calls(calls) == []


def test_filter_drops_empty_string_args():
    """read_file(path=''), execute_python(code=''), etc. are dropped."""
    calls = [
        {"id": "1", "name": "read_file", "args": {"path": ""}},
        {"id": "2", "name": "search_files", "args": {"pattern": "", "directory": "."}},
        {"id": "3", "name": "execute_python", "args": {"code": ""}},
    ]
    assert _filter_empty_tool_calls(calls) == []


def test_filter_drops_whitespace_only_args():
    calls = [
        {"id": "1", "name": "read_file", "args": {"path": "   "}},
        {"id": "2", "name": "execute_python", "args": {"code": "\n\t"}},
    ]
    assert _filter_empty_tool_calls(calls) == []


def test_filter_keeps_real_calls():
    """Calls with real, non-empty string args survive."""
    calls = [
        {"id": "1", "name": "open_app", "args": {"query": "discord"}},
        {"id": "2", "name": "read_file", "args": {"path": "src/foo.py"}},
        {"id": "3", "name": "execute_python", "args": {"code": "print('hi')"}},
        {"id": "4", "name": "edit_file", "args": {"path": "a.txt", "old_text": "x", "new_text": "y"}},
    ]
    assert _filter_empty_tool_calls(calls) == calls


def test_filter_keeps_calls_with_non_string_args():
    """Bool / int / list / dict args are considered meaningful even if empty-string-free."""
    calls = [
        {"id": "1", "name": "some_tool", "args": {"flag": True}},
        {"id": "2", "name": "another_tool", "args": {"count": 0, "names": []}},
    ]
    assert _filter_empty_tool_calls(calls) == calls


def test_filter_preserves_order():
    """Stable filter — drop-in-place keeps relative order of surviving calls."""
    calls = [
        {"id": "1", "name": "open_app", "args": {"query": "discord"}},
        {"id": "2", "name": "read_file", "args": {"path": ""}},  # drop
        {"id": "3", "name": "read_file", "args": {"path": "src/foo.py"}},
        {"id": "4", "name": "execute_python", "args": {"code": ""}},  # drop
        {"id": "5", "name": "open_app", "args": {"query": "steam"}},
    ]
    result = _filter_empty_tool_calls(calls)
    assert [c["id"] for c in result] == ["1", "3", "5"]


def test_filter_handles_none_args():
    """None-valued args should not crash the filter; we just don't count them as meaningful."""
    # Tool returns args with a None value — rare but possible. Should be dropped
    # because no string arg has content.
    calls = [{"id": "1", "name": "read_file", "args": {"path": None}}]
    assert _filter_empty_tool_calls(calls) == []


def test_filter_empty_input_returns_empty():
    assert _filter_empty_tool_calls([]) == []


# ---- _is_small_model + _trim_history_for_model ----

def test_is_small_model_detects_3b():
    """llama3.2:3b should be detected as small."""
    from graph.nodes._helpers import _is_small_model
    class FakeLLM:
        model = "llama3.2:3b"
    assert _is_small_model(FakeLLM()) is True


def test_is_small_model_detects_1b():
    from graph.nodes._helpers import _is_small_model
    class FakeLLM:
        model = "qwen2.5:1.5b"
    assert _is_small_model(FakeLLM()) is True


def test_is_small_model_coder_is_not_small():
    """qwen2.5-coder:7b should NOT be detected as small (it's our planner)."""
    from graph.nodes._helpers import _is_small_model
    class FakeLLM:
        model = "qwen2.5-coder:7b"
    assert _is_small_model(FakeLLM()) is False


def test_is_small_model_unknown_assumes_not_small():
    from graph.nodes._helpers import _is_small_model
    class FakeLLM:
        model = "some-huge-70b-model"
    assert _is_small_model(FakeLLM()) is False


def test_trim_history_caps_small_model_at_10():
    """Small models should only see the last 10 messages."""
    from graph.nodes._helpers import _trim_history_for_model
    from langchain_core.messages import HumanMessage

    class FakeLLM:
        model = "llama3.2:3b"

    msgs = [HumanMessage(content=f"msg {i}") for i in range(40)]
    trimmed = _trim_history_for_model(msgs, FakeLLM())
    assert len(trimmed) == 10
    assert trimmed[-1].content == "msg 39"


def test_trim_history_keeps_all_for_large_model():
    """Larger models (40+ context) keep the full conversation."""
    from graph.nodes._helpers import _trim_history_for_model
    from langchain_core.messages import HumanMessage

    class FakeLLM:
        model = "llama3.1:70b"

    msgs = [HumanMessage(content=f"msg {i}") for i in range(20)]
    trimmed = _trim_history_for_model(msgs, FakeLLM())
    assert len(trimmed) == 20


def test_trim_history_short_history_unchanged():
    """Conversations already under the cap are returned unchanged."""
    from graph.nodes._helpers import _trim_history_for_model
    from langchain_core.messages import HumanMessage

    class FakeLLM:
        model = "llama3.2:3b"

    msgs = [HumanMessage(content=f"msg {i}") for i in range(5)]
    trimmed = _trim_history_for_model(msgs, FakeLLM())
    assert len(trimmed) == 5
    assert [m.content for m in trimmed] == [m.content for m in msgs]


# ---- _FRESH_QUESTION_HINT replaces the bloated core hints ----

def test_default_hints_is_minimal():
    """DEFAULT_HINTS should be just the fresh-question hint, not the old
    3-hint stack. Verifies we actually slimmed the per-turn payload.
    """
    from graph.nodes._hints import DEFAULT_HINTS, _FRESH_QUESTION_HINT
    assert len(DEFAULT_HINTS) == 1
    assert DEFAULT_HINTS[0] is _FRESH_QUESTION_HINT
