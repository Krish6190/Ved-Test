"""Unit tests for the Ved tool layer.

Covers:
  - Implicit-target fallback (when the LLM omits the primary arg)
  - Dual-mode safety (default vs self-healing)
  - System path blocking (Windows / POSIX)
  - "User denied approval" path for edit / execute tools (tkinter mocked)
  - execute_python happy path and short-code rejection
  - Fallback from AI message fence for execute_python

The approval popup (`messagebox.askyesno`) is mocked via `unittest.mock.patch`
so tests run headless. To control the user's choice, set
`graph.tools._common.APPROVAL_RETURN` or use the mock's `.return_value`.

Run with: `cd C:\\Users\\krish\\OneDrive\\Desktop\\ved && .venv\\Scripts\\pytest tests/test_tools.py -v`
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from graph.state import VedState
from graph.tools.file_reader import read_file
from graph.tools.file_editor import edit_file, overwrite_file
from graph.tools.file_search import search_files
from graph.tools.python_runner import execute_python


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """Create an isolated workspace with a few files and patch PROJECT_ROOT
    so self-healing-mode tests resolve against `tmp_path`."""
    (tmp_path / "README.md").write_text("# Test README\n\nHello world.", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest>=8.0\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    # Point all four modules at this workspace
    for mod_name in (
        "graph.tools._common",
        "graph.tools.file_reader",
        "graph.tools.file_editor",
        "graph.tools.file_search",
    ):
        try:
            mod = __import__(mod_name, fromlist=["PROJECT_ROOT"])
            monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path, raising=False)
        except ImportError:
            pass
    # Also patch the name in graph.tools._common itself, since tools re-export
    monkeypatch.setattr("graph.tools._common.PROJECT_ROOT", tmp_path)
    return tmp_path


def make_state(messages=None, self_healing=False, mode="standard"):
    """Build a VedState with the given conversation + flags."""
    return VedState(
        messages=messages or [HumanMessage(content="hello")],
        self_healing=self_healing,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_explicit_path_happy(self, tmp_workspace):
        # Use absolute path so the test resolves against the workspace,
        # not whatever cwd pytest happens to be running from.
        target = str(tmp_workspace / "README.md")
        result = read_file.invoke({"path": target, "state": make_state()})
        assert "FILE:" in result
        assert "Hello world" in result
        assert "ERROR" not in result

    def test_fallback_infers_readme(self, tmp_workspace):
        """LLM omits path; user message says 'show me the readme'. Patch cwd
        so the fallback searches inside the workspace."""
        import os as _os
        old_cwd = _os.getcwd()
        try:
            _os.chdir(str(tmp_workspace))
            state = make_state([HumanMessage(content="show me the readme")])
            result = read_file.invoke({"path": "", "state": state})
            assert "Hello world" in result, f"expected fallback to find README.md, got: {result[:200]}"
            assert "ERROR" not in result
        finally:
            _os.chdir(old_cwd)

    def test_fallback_infers_requirements(self, tmp_workspace):
        import os as _os
        old_cwd = _os.getcwd()
        try:
            _os.chdir(str(tmp_workspace))
            state = make_state([HumanMessage(content="open requirements")])
            result = read_file.invoke({"path": "", "state": state})
            assert "pytest>=8.0" in result, f"expected fallback to find requirements.txt, got: {result[:200]}"
        finally:
            _os.chdir(old_cwd)

    def test_cannot_infer_returns_error(self, tmp_workspace):
        state = make_state([HumanMessage(content="tell me about xyzzy plover")])
        result = read_file.invoke({"path": "", "state": state})
        assert result.startswith("ERROR:")
        assert "could not infer" in result.lower()

    def test_system_path_blocked_default(self, tmp_workspace):
        result = read_file.invoke(
            {"path": r"C:\Windows\System32\drivers\etc\hosts", "state": make_state()}
        )
        assert result.startswith("ERROR:")
        assert "Refused" in result

    def test_system_path_blocked_self_healing(self, tmp_workspace):
        state = make_state([HumanMessage(content="anything")], self_healing=True)
        result = read_file.invoke(
            {"path": r"C:\Windows\System32\drivers\etc\hosts", "state": state}
        )
        assert result.startswith("ERROR:")
        assert "self-healing" in result.lower()

    def test_self_healing_blocks_project_external(self, tmp_workspace, monkeypatch):
        """Even a file outside project_root is blocked in self-healing mode."""
        # Create a file outside the workspace
        outside = tmp_workspace.parent / "outside.txt"
        outside.write_text("secret data", encoding="utf-8")
        try:
            state = make_state([HumanMessage(content="anything")], self_healing=True)
            result = read_file.invoke({"path": str(outside), "state": state})
            assert result.startswith("ERROR:")
            assert "self-healing" in result.lower()
        finally:
            outside.unlink(missing_ok=True)

    def test_nonexistent_file(self, tmp_workspace):
        # Use an absolute path inside tmp_workspace so the action's
        # allowed_roots check (which now enforces PROJECT_ROOT as a
        # safety gate) passes and we hit the file-not-found branch.
        target = str(tmp_workspace / "nope.txt")
        result = read_file.invoke({"path": target, "state": make_state()})
        assert result.startswith("ERROR:")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# edit_file + overwrite_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_happy_path(self, tmp_workspace):
        target = tmp_workspace / "notes.txt"
        target.write_text("hello world", encoding="utf-8")
        with patch("graph.tools.file_editor._request_approval", return_value=True):
            result = edit_file.invoke({
                "path": str(target),
                "old_text": "world",
                "new_text": "VED",
                "state": make_state(),
            })
        assert result.startswith("OK:")
        assert target.read_text(encoding="utf-8") == "hello VED"
        # Backup file exists
        backup = target.with_suffix(target.suffix + ".bak")
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "hello world"

    def test_old_text_not_found(self, tmp_workspace):
        target = tmp_workspace / "notes.txt"
        target.write_text("hello world", encoding="utf-8")
        with patch("graph.tools.file_editor._request_approval", return_value=True):
            result = edit_file.invoke({
                "path": str(target),
                "old_text": "missing",
                "new_text": "x",
                "state": make_state(),
            })
        assert result.startswith("ERROR:")
        assert "could not locate" in result.lower()

    def test_user_denied_approval(self, tmp_workspace):
        target = tmp_workspace / "notes.txt"
        target.write_text("hello world", encoding="utf-8")
        original = target.read_text(encoding="utf-8")
        with patch("graph.tools.file_editor._request_approval", return_value=False):
            result = edit_file.invoke({
                "path": str(target),
                "old_text": "world",
                "new_text": "VED",
                "state": make_state(),
            })
        assert result.startswith("ERROR:")
        assert "denied" in result.lower()
        # File untouched
        assert target.read_text(encoding="utf-8") == original

    def test_self_healing_blocks_project_external(self, tmp_workspace):
        outside = tmp_workspace.parent / "outside.txt"
        outside.write_text("data", encoding="utf-8")
        try:
            state = make_state(self_healing=True)
            result = edit_file.invoke({
                "path": str(outside),
                "old_text": "data",
                "new_text": "x",
                "state": state,
            })
            assert result.startswith("ERROR:")
            assert "self-healing" in result.lower()
        finally:
            outside.unlink(missing_ok=True)

    def test_empty_old_text_returns_error(self, tmp_workspace):
        target = tmp_workspace / "notes.txt"
        target.write_text("hello", encoding="utf-8")
        result = edit_file.invoke({
            "path": str(target),
            "old_text": "",
            "new_text": "x",
            "state": make_state(),
        })
        assert result.startswith("ERROR:")
        assert "overwrite_file" in result  # suggests the right alternative


class TestOverwriteFile:
    def test_happy_path(self, tmp_workspace):
        target = tmp_workspace / "notes.txt"
        target.write_text("old content", encoding="utf-8")
        with patch("graph.tools.file_editor._request_approval", return_value=True):
            result = overwrite_file.invoke({
                "path": str(target),
                "content": "new content",
                "state": make_state(),
            })
        assert result.startswith("OK:")
        assert target.read_text(encoding="utf-8") == "new content"

    def test_user_denied_approval(self, tmp_workspace):
        target = tmp_workspace / "notes.txt"
        target.write_text("original", encoding="utf-8")
        with patch("graph.tools.file_editor._request_approval", return_value=False):
            result = overwrite_file.invoke({
                "path": str(target),
                "content": "X",
                "state": make_state(),
            })
        assert result.startswith("ERROR:")
        assert "denied" in result.lower()
        assert target.read_text(encoding="utf-8") == "original"

    def test_self_healing_blocks_external(self, tmp_workspace):
        outside = tmp_workspace.parent / "outside.txt"
        outside.write_text("data", encoding="utf-8")
        try:
            state = make_state(self_healing=True)
            result = overwrite_file.invoke({
                "path": str(outside),
                "content": "X",
                "state": state,
            })
            assert result.startswith("ERROR:")
            assert "self-healing" in result.lower()
        finally:
            outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    def test_explicit_pattern_happy(self, tmp_workspace):
        state = make_state()
        result = search_files.invoke({"pattern": "*.md", "directory": ".", "state": state})
        assert "Found" in result
        assert "README.md" in result

    def test_fallback_infers_pattern(self, tmp_workspace):
        state = make_state([HumanMessage(content="show me the readme")])
        result = search_files.invoke({"pattern": "", "directory": ".", "state": state})
        assert "Found" in result
        assert "README.md" in result

    def test_no_matches_returns_error(self, tmp_workspace):
        state = make_state()
        result = search_files.invoke({"pattern": "*.xyzzy", "directory": ".", "state": state})
        assert result.startswith("ERROR:")
        assert "no files matched" in result.lower()

    def test_self_healing_search_works_inside_project(self, tmp_workspace):
        state = make_state(self_healing=True)
        result = search_files.invoke({"pattern": "*.py", "directory": ".", "state": state})
        assert "Found" in result
        assert "main.py" in result

    def test_self_healing_blocks_external_directory(self, tmp_workspace):
        outside_dir = tmp_workspace.parent / "outside_dir"
        outside_dir.mkdir(exist_ok=True)
        try:
            state = make_state(self_healing=True)
            result = search_files.invoke({
                "pattern": "*",
                "directory": str(outside_dir),
                "state": state,
            })
            assert result.startswith("ERROR:")
            assert "self-healing" in result.lower()
        finally:
            outside_dir.rmdir()


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------


class TestExecutePython:
    def test_happy_path(self, tmp_workspace):
        with patch("graph.tools.python_runner._request_approval", return_value=True):
            result = execute_python.invoke({
                "code": "print('hello from test')",
                "state": make_state(),
            })
        assert "OK" in result
        assert "hello from test" in result

    def test_user_denied(self, tmp_workspace):
        # Use a snippet with a space so the `len < 2 tokens` check passes.
        with patch("graph.tools.python_runner._request_approval", return_value=False):
            result = execute_python.invoke({
                "code": "print('blocked')  # test",
                "state": make_state(),
            })
        assert result.startswith("ERROR:")
        assert "denied" in result.lower()

    def test_short_code_rejected(self, tmp_workspace):
        result = execute_python.invoke({
            "code": "x",
            "state": make_state(),
        })
        assert result.startswith("ERROR:")
        assert "no clear executable" in result.lower()

    def test_empty_code_no_ai_message_rejected(self, tmp_workspace):
        """If `code` is empty AND there's no AI message with a python fence,
        the tool must refuse rather than running an empty script."""
        state = VedState(
            messages=[HumanMessage(content="hi")],
            self_healing=False,
            mode="standard",
        )
        result = execute_python.invoke({"code": "", "state": state})
        assert result.startswith("ERROR:")

    def test_fallback_extracts_from_ai_message(self, tmp_workspace):
        """When `code` is omitted, the tool pulls from the last AI message."""
        from langchain_core.messages import AIMessage
        state = VedState(
            messages=[
                HumanMessage(content="run something"),
                AIMessage(content="Here you go:\n```python\nprint('from ai')\n```"),
            ],
            self_healing=False,
            mode="standard",
        )
        with patch("graph.tools.python_runner._request_approval", return_value=True):
            result = execute_python.invoke({"code": "", "state": state})
        assert "from ai" in result

    def test_timeout_handled(self, tmp_workspace):
        """A script that runs longer than the 10s timeout must return a clean
        timeout error (not hang the test)."""
        with patch("graph.tools.python_runner._request_approval", return_value=True):
            with patch("graph.tools.python_runner._TIMEOUT_SECONDS", 1):
                result = execute_python.invoke({
                    "code": "import time; time.sleep(5); print('should not print')",
                    "state": make_state(),
                })
        assert "ERROR" in result
        assert "timeout" in result.lower()
