"""Tests for chat-history compression and tool-message persistence.

Covers:
  - ToolMessage serialization roundtrip (preserves tool_call_id)
  - _compress_ai_content: short content passes through, long content becomes
    head+tail summary, exactly-threshold content passes through
  - _save_ai_response_to_thread_rag: best-effort failure mode (returns False
    when RAG embedding pipeline is unavailable)
"""
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from chatbot import (
    _AI_SUMMARY_HEAD_WORDS,
    _AI_SUMMARY_TAIL_WORDS,
    _AI_SUMMARY_THRESHOLD_CHARS,
    _compress_ai_content,
    _save_ai_response_to_thread_rag,
)
from data.threads import _deserialize_message, _serialize_message


# ---- ToolMessage roundtrip ----

def test_tool_message_serialize_has_tool_role():
    msg = ToolMessage(content="file contents here", tool_call_id="call_abc123")
    out = _serialize_message(msg)
    assert out["role"] == "tool"
    assert out["content"] == "file contents here"
    assert out["tool_call_id"] == "call_abc123"


def test_tool_message_deserialize_roundtrip():
    original = ToolMessage(content="result of read_file", tool_call_id="call_xyz")
    serialized = _serialize_message(original)
    restored = _deserialize_message(serialized)
    assert isinstance(restored, ToolMessage)
    assert restored.content == "result of read_file"
    assert restored.tool_call_id == "call_xyz"


def test_tool_message_without_call_id_roundtrips():
    """A legacy save missing tool_call_id gets a placeholder on load."""
    # Simulate a legacy save (no tool_call_id in JSON).
    legacy_payload = {"role": "tool", "content": "some output"}
    restored = _deserialize_message(legacy_payload)
    assert isinstance(restored, ToolMessage)
    # Placeholder for unpaired tool messages (e.g., legacy saves).
    assert restored.tool_call_id == "legacy_unpaired"
    assert restored.content == "some output"


def test_human_ai_system_messages_still_work():
    """Sanity: the existing 3-way serialization still works."""
    h = _serialize_message(HumanMessage(content="hello"))
    assert h["role"] == "human"
    a = _serialize_message(AIMessage(content="hi"))
    assert a["role"] == "ai"
    s = _serialize_message(SystemMessage(content="be helpful"))
    assert s["role"] == "system"

    assert _deserialize_message(h).content == "hello"
    assert _deserialize_message(a).content == "hi"
    assert _deserialize_message(s).content == "be helpful"


def test_ai_message_with_tool_calls_roundtrips():
    """AIMessage that emitted tool_calls preserves them on roundtrip via
    additional_kwargs (LangChain stores tool_calls there on load)."""
    msg = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [{"id": "c1", "name": "read_file", "args": {}}]},
    )
    serialized = _serialize_message(msg)
    restored = _deserialize_message(serialized)
    assert isinstance(restored, AIMessage)
    assert restored.additional_kwargs.get("tool_calls")


def test_tool_message_long_content_is_truncated_in_history():
    """ToolMessage content above _TOOL_HISTORY_TRUNCATE_CHARS gets a marker
    pointing to retrieve_rag for full recovery. Keeps history compact."""
    from chatbot import _TOOL_HISTORY_TRUNCATE_CHARS
    long_output = "x" * (_TOOL_HISTORY_TRUNCATE_CHARS * 3)
    msg = ToolMessage(content=long_output, tool_call_id="call_x")
    out = _serialize_message(msg)
    assert "truncated" in out["content"]
    assert "retrieve_rag" in out["content"]
    # Full output is NOT preserved in history.
    assert long_output not in out["content"]
    # But tool_call_id is preserved.
    assert out["tool_call_id"] == "call_x"


def test_tool_message_short_content_passes_through():
    """Short ToolMessage content is not truncated."""
    msg = ToolMessage(content="small result", tool_call_id="call_y")
    out = _serialize_message(msg)
    assert out["content"] == "small result"
    assert "truncated" not in out["content"]


# ---- _compress_ai_content ----

def test_compress_short_content_unchanged():
    """Content below the threshold passes through verbatim."""
    short = "Hello, this is a short response with only a few words."
    assert _compress_ai_content(short) == short


def test_compress_empty_content_unchanged():
    assert _compress_ai_content("") == ""


def test_compress_exactly_threshold_unchanged():
    """Content at exactly the threshold length is not compressed (boundary)."""
    content = "x" * _AI_SUMMARY_THRESHOLD_CHARS
    assert _compress_ai_content(content) == content


def test_compress_long_content_produces_head_and_tail():
    """Content above the threshold becomes head + tail summary."""
    # Build a long string with distinguishable head and tail markers.
    head = " ".join([f"HEAD{w}" for w in range(_AI_SUMMARY_HEAD_WORDS)])
    filler = " ".join(["filler"] * 200)
    tail = " ".join([f"TAIL{w}" for w in range(_AI_SUMMARY_TAIL_WORDS)])
    content = f"{head} {filler} {tail}"
    assert len(content) > _AI_SUMMARY_THRESHOLD_CHARS

    summary = _compress_ai_content(content)
    assert "HEAD0" in summary  # head words present
    assert f"HEAD{_AI_SUMMARY_HEAD_WORDS - 1}" in summary
    assert "TAIL0" in summary  # tail words present
    assert f"TAIL{_AI_SUMMARY_TAIL_WORDS - 1}" in summary
    # Filler words should be GONE from the summary.
    assert "filler" not in summary
    # Marker phrase explains where the full content went.
    assert "RAG" in summary.upper()


def test_compress_short_word_count_skipped():
    """If the word count is small even though char count is high, skip."""
    # 200 chars but only a handful of words.
    content = "x " * (_AI_SUMMARY_THRESHOLD_CHARS // 2)
    assert _compress_ai_content(content) == content


# ---- _save_ai_response_to_thread_rag ----

def test_save_to_rag_returns_false_for_empty_content():
    assert _save_ai_response_to_thread_rag("thr_x", "", "label") is False
    assert _save_ai_response_to_thread_rag("", "some content", "label") is False


def test_save_to_rag_swallows_embedding_errors(monkeypatch):
    """Best-effort: if the embedding pipeline is unavailable (no Ollama),
    returns False rather than raising — chat persistence must not break."""
    import data.thread_files as tf_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated: no Ollama available")

    monkeypatch.setattr(tf_module, "ThreadFileStore", _boom)
    result = _save_ai_response_to_thread_rag("thr_x", "x" * 5000, "label")
    assert result is False
