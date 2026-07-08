"""Unit tests for the _clean_llm_text helper in graph/actions/filesystem.py.

Locks in the behavior of text normalization for LLM-provided tool args.
These tests guard against regressions in the markdown-fence / CRLF stripping
that cloud LLMs (Qwen via OpenRouter) frequently need.
"""
from graph.actions.filesystem import _clean_llm_text


def test_strips_leading_markdown_fence():
    text = "```python\ndef foo():\n    return 1\n```"
    out = _clean_llm_text(text)
    assert "```" not in out
    assert "def foo" in out
    assert "return 1" in out


def test_strips_trailing_fence_only():
    text = "def bar(): pass\n```"
    out = _clean_llm_text(text)
    assert "```" not in out
    assert "def bar" in out


def test_strips_both_fences():
    text = "```\nx = 1\n```"
    out = _clean_llm_text(text)
    assert out == "x = 1"


def test_normalizes_crlf_to_lf():
    text = "line1\r\nline2\r\nline3"
    out = _clean_llm_text(text)
    assert "\r" not in out
    assert out == "line1\nline2\nline3"


def test_idempotent_on_clean_text():
    text = "def hello():\n    print('hi')\n"
    out = _clean_llm_text(text)
    assert out == text


def test_empty_string_returns_empty():
    assert _clean_llm_text("") == ""


def test_strips_fence_without_language_hint():
    text = "```\nplain code\n```"
    out = _clean_llm_text(text)
    assert out == "plain code"
