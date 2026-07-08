"""Unit tests for the propose_tool machinery.

Covers: filename sanitization, AST blocked-import detection, full happy-
path write+import+register (via monkeypatching the user_tools dir), and
the duplicate-name rejection path.

Does NOT load the real Ollama-backed chatbot. The approval event is
driven directly via the runnable config.
"""
import queue
import sys
import threading
from pathlib import Path

import pytest

import graph.tools as tools_module
import graph.tools.tool_creator as tc_module
from graph.tools.tool_creator import (
    _safe_filename,
    _scan_blocked_imports,
    propose_tool,
)


# ---- _safe_filename ----

def test_safe_filename_basic():
    assert _safe_filename("fetch_weather") == "fetch_weather"
    assert _safe_filename("FetchWeather") == "fetchweather"
    assert _safe_filename("hello world!") == "hello_world"


def test_safe_filename_strips_leading_non_letter():
    assert _safe_filename("123_tool") == "tool"
    assert _safe_filename("_underscore") == "underscore"


def test_safe_filename_caps_length():
    long_name = "a" * 100
    out = _safe_filename(long_name)
    assert len(out) <= 60


def test_safe_filename_empty_falls_back():
    assert _safe_filename("") == "unnamed_tool"
    assert _safe_filename("!!!") == "unnamed_tool"


# ---- _scan_blocked_imports ----

def test_scan_clean_code():
    code = "import os\nimport json\nfrom pathlib import Path\n"
    assert _scan_blocked_imports(code) == []


def test_scan_blocks_subprocess():
    code = "import subprocess\ndef run(): pass\n"
    assert "subprocess" in _scan_blocked_imports(code)


def test_scan_blocks_socket():
    code = "from socket import socket\ndef f(): pass\n"
    assert "socket" in _scan_blocked_imports(code)


def test_scan_blocks_os_system_from_import():
    code = "from os import system\ndef f(): pass\n"
    out = _scan_blocked_imports(code)
    assert any("os.system" in x or x == "os.system" for x in out)


def test_scan_handles_syntax_error():
    code = "def f(:\n    pass\n"
    out = _scan_blocked_imports(code)
    assert len(out) == 1
    assert out[0].startswith("SYNTAX_ERROR")


def test_scan_handles_no_imports():
    code = "x = 1\ny = 2\n"
    assert _scan_blocked_imports(code) == []


# ---- propose_tool full happy-path ----

def _make_config(approval_value=True):
    """Build a RunnableConfig with the approval wiring propose_tool expects."""
    q = queue.Queue()
    event = threading.Event()
    state = {"value": approval_value, "session_id": "test"}
    return {
        "configurable": {
            "token_queue": q,
            "tool_creation_event": event,
            "tool_creation_state": state,
            "session_id": "test",
        }
    }, q, event, state


def _invoke_propose_tool(name, description, code, cfg):
    """Call propose_tool.invoke() with the config as a separate parameter.

    LangChain's @tool ignores 'config' keys in the input dict (they are
    handled via the second positional arg). Use this helper to avoid that
    footgun.
    """
    return propose_tool.invoke(
        {"name": name, "description": description, "code": code},
        config=cfg,
    )


def _good_tool_code(name="dummy_test_tool"):
    return (
        "from langchain_core.tools import tool\n\n"
        f"@tool\n"
        f"def {name}(query: str) -> str:\n"
        f"    \"\"\"Echo the query.\"\"\"\n"
        f"    return f'echo: {{query}}'\n"
    )


@pytest.fixture
def real_user_tools_dir():
    """Use the real graph/tools/user_tools/ for one test. Snapshot the
    directory contents at setup; remove anything added during the test at
    teardown, and drop the corresponding entries from VED_TOOLS + sys.modules."""
    from graph.tools import VED_TOOLS

    real_dir = tc_module.USER_TOOLS_DIR
    real_dir.mkdir(parents=True, exist_ok=True)
    pre_existing = {p.name for p in real_dir.glob("*.py")}
    yield real_dir
    # Teardown: anything that wasn't here before is ours to delete.
    for path in real_dir.glob("*.py"):
        if path.name not in pre_existing:
            stem = path.stem
            path.unlink()
            VED_TOOLS[:] = [
                t for t in VED_TOOLS if getattr(t, "name", None) != stem
            ]
            sys.modules.pop(f"graph.tools.user_tools.{stem}", None)


def test_propose_tool_happy_path(real_user_tools_dir):
    """Approve flow: file written, module imported, tool registered."""
    cfg, q, event, state = _make_config(approval_value=True)

    def resolve():
        item = q.get(timeout=5)
        assert item[0] == "tool_creation_proposal"
        state["value"] = True
        event.set()

    t = threading.Thread(target=resolve, daemon=True)
    t.start()

    result = _invoke_propose_tool("dummy_test_tool", "echo the query", _good_tool_code(), cfg)
    t.join(timeout=5)
    assert "OK: Tool 'dummy_test_tool' registered and ready" in result, result
    assert (real_user_tools_dir / "dummy_test_tool.py").exists()
    names = {getattr(t, "name", None) for t in tools_module.VED_TOOLS}
    assert "dummy_test_tool" in names


def test_propose_tool_rejects_duplicate(real_user_tools_dir):
    """If the file already exists, return ERROR without writing or asking."""
    (real_user_tools_dir / "exists_already.py").write_text("# pre-existing\n")

    cfg, q, _event, _state = _make_config()
    result = _invoke_propose_tool("exists_already", "x", _good_tool_code("exists_already"), cfg)
    assert result.startswith("ERROR:"), result
    assert "already exists" in result
    assert q.empty()


def test_propose_tool_rejected_returns_rejection_message(real_user_tools_dir):
    """If human rejects, return rejection message; no file written."""
    cfg, q, event, state = _make_config(approval_value=False)

    def resolve():
        q.get(timeout=5)
        state["value"] = False
        event.set()

    t = threading.Thread(target=resolve, daemon=True)
    t.start()

    result = _invoke_propose_tool("rejected_tool", "x", _good_tool_code("rejected_tool"), cfg)
    t.join(timeout=5)
    assert "rejected" in result.lower(), result
    assert not (real_user_tools_dir / "rejected_tool.py").exists()


def test_propose_tool_blocks_dangerous_imports(real_user_tools_dir):
    """Code with subprocess import is rejected without bothering the human."""
    cfg, q, _, _ = _make_config()
    result = _invoke_propose_tool("danger_tool", "x", "import subprocess\ndef f(): pass\n", cfg)
    assert result.startswith("ERROR:"), result
    assert "blocked imports" in result
    assert "subprocess" in result
    assert q.empty()
    assert not (real_user_tools_dir / "danger_tool.py").exists()


def test_propose_tool_requires_config_wiring(real_user_tools_dir):
    """If no token_queue / event in config, return ERROR without approval."""
    result = _invoke_propose_tool("no_config_tool", "x", _good_tool_code(), {"configurable": {}})
    assert result.startswith("ERROR:"), result


def test_propose_tool_sanitizes_name(real_user_tools_dir):
    """Punny name 'My Tool!' becomes 'my_tool'."""
    cfg, q, event, state = _make_config(approval_value=True)

    def resolve():
        item = q.get(timeout=5)
        assert item[1]["tool_name"] == "my_tool"
        state["value"] = True
        event.set()

    t = threading.Thread(target=resolve, daemon=True)
    t.start()

    result = _invoke_propose_tool("My Tool!", "echo", _good_tool_code("my_tool"), cfg)
    t.join(timeout=5)
    assert "OK: Tool 'my_tool' registered and ready" in result, result
    assert (real_user_tools_dir / "my_tool.py").exists()
