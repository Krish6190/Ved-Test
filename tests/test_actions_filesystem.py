"""Unit tests for the filesystem action layer.

Pure-Python tests against `tmp_path`. No LLM, no LangChain imports. The
goal is to lock in the action-layer contract (primitive args in, primitive
result out) so the future client-server split can replace the action
implementations without breaking tool-layer behavior.

Run with: `cd /mnt/c/Users/krish/OneDrive/Desktop/ved && .venv/bin/python -m pytest tests/test_actions_filesystem.py -v`
"""
from pathlib import Path

from graph.actions.filesystem import (
    edit_file_action,
    overwrite_file_action,
    read_file_action,
    search_files_action,
)


# ---------------------------------------------------------------------------
# read_file_action
# ---------------------------------------------------------------------------

def test_read_existing_file(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("hello from the test", encoding="utf-8")
    result = read_file_action(str(target), allowed_roots=(str(tmp_path),))
    assert "hello from the test" in result
    assert "FILE:" in result
    assert "ERROR" not in result


def test_read_missing_file(tmp_path: Path):
    """A missing file must return an error string, never raise."""
    target = tmp_path / "does_not_exist.txt"
    result = read_file_action(str(target), allowed_roots=(str(tmp_path),))
    assert isinstance(result, str)
    assert result.startswith("ERROR:")
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# edit_file_action
# ---------------------------------------------------------------------------

def test_edit_and_verify(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    result = edit_file_action(
        str(target),
        "world",
        "VED",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "hello VED"
    # Default backup_dir=None means the .bak lands next to the file.
    backup = target.with_suffix(target.suffix + ".bak")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "hello world"


def test_edit_old_text_not_found(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    result = edit_file_action(
        str(target),
        "missing phrase",
        "x",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert isinstance(result, str)
    assert result.startswith("ERROR:")
    assert "could not locate" in result.lower() or "not found" in result.lower()
    # File must be untouched.
    assert target.read_text(encoding="utf-8") == "hello world"


# ---------------------------------------------------------------------------
# overwrite_file_action
# ---------------------------------------------------------------------------

def test_overwrite_and_verify(tmp_path: Path):
    target = tmp_path / "data.txt"
    target.write_text("original content", encoding="utf-8")
    result = overwrite_file_action(
        str(target),
        "brand new content",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "brand new content"
    backup = target.with_suffix(target.suffix + ".bak")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "original content"


# ---------------------------------------------------------------------------
# search_files_action
# ---------------------------------------------------------------------------

def test_search_with_match(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("a", encoding="utf-8")
    (tmp_path / "beta.py").write_text("b", encoding="utf-8")
    (tmp_path / "gamma.txt").write_text("c", encoding="utf-8")

    result = search_files_action(
        "*.py",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert "alpha.py" in result
    assert "beta.py" in result
    assert "gamma.txt" not in result
    assert "ERROR" not in result


def test_search_no_match(tmp_path: Path):
    """An unmatched pattern must return an empty string (no error, no raise).

    The tool layer formats the empty case as an ERROR for the LLM; the
    action's contract is just 'no matches -> empty string'.
    """
    (tmp_path / "alpha.py").write_text("a", encoding="utf-8")
    result = search_files_action(
        "*.zzz",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert result == ""
