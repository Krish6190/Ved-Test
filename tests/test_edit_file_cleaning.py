"""Integration tests: edit_file_action accepts LLM-formatted old_text.

Verifies that markdown-fence wrapping and CRLF line endings in the LLM's
tool arguments don't break the splice. Without _clean_llm_text, the
current.find(old_text) would return -1 and the action would refuse to edit.
"""
from pathlib import Path

from graph.actions.filesystem import edit_file_action


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_edit_succeeds_with_markdown_fenced_old_text(tmp_path):
    """LLM wraps old_text in ```python fence — action should still match."""
    target = tmp_path / "test_auth.py"
    _write(target, "def login(user, password):\n    return True\n")

    fenced_old = "```python\ndef login(user, password):\n    return True\n```"
    new_text = "def login(user, password):\n    return authenticate(user, password)\n"

    result = edit_file_action(
        str(target),
        fenced_old,
        new_text,
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert "OK" in result, f"edit failed: {result}"
    actual = target.read_text(encoding="utf-8")
    assert "authenticate(user, password)" in actual
    assert "return True" not in actual


def test_edit_succeeds_with_crlf_old_text(tmp_path):
    """LLM sends CRLF line endings — action should still match LF file."""
    target = tmp_path / "test_db.py"
    _write(target, "def connect():\n    return conn\n")

    crlf_old = "def connect():\r\n    return conn\r\n"
    new_text = "def connect():\n    return pool.getconn()\n"

    result = edit_file_action(
        str(target),
        crlf_old,
        new_text,
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert "OK" in result, f"edit failed: {result}"
    actual = target.read_text(encoding="utf-8")
    assert "pool.getconn()" in actual


def test_edit_fails_on_genuinely_mismatched_text(tmp_path):
    """If old_text really doesn't exist, action must return ERROR (not silently write)."""
    target = tmp_path / "test_x.py"
    _write(target, "def real_function():\n    pass\n")

    bogus_old = "def hallucinated_function():\n    pass\n"
    result = edit_file_action(
        str(target),
        bogus_old,
        "def replaced():\n    pass\n",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert "ERROR" in result
    assert "Could not locate" in result
    # File must be unchanged
    actual = target.read_text(encoding="utf-8")
    assert "real_function" in actual
    assert "hallucinated_function" not in actual
    assert "replaced" not in actual


def test_edit_error_message_includes_file_hint(tmp_path):
    """The error message must give the LLM enough info to self-correct."""
    target = tmp_path / "test_y.py"
    _write(target, "actual_content_here = 42\n")

    result = edit_file_action(
        str(target),
        "different_content",
        "new",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert "ERROR" in result
    assert "actual_content_here" in result or "first 200 chars" in result
