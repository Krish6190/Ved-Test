"""Unit tests for the open_app tool.

Tests cover:
  - Approval gate (both SSE-config and Tk fallback)
  - Empty / unknown query error paths
  - Search ranking (Start Menu > install dirs > PATH)
  - Launch path (os.startfile on Windows, subprocess.Popen elsewhere)
  - Nickname normalization (vscode -> Visual Studio Code)
"""
import queue
import threading
from unittest.mock import MagicMock, patch
import graph.tools.app_launcher as al


# ---- Helpers ----

def _make_config(approval_value=True):
    """Build a RunnableConfig that approves via the SSE bus."""
    q = queue.Queue()
    event = threading.Event()
    state = {"value": approval_value, "session_id": "test"}
    return {
        "configurable": {
            "token_queue": q,
            "approval_event": event,
            "approval_state": state,
            "session_id": "test",
        }
    }, q, event, state


def _invoke_open_app(query, cfg):
    """open_app.invoke requires config as a separate arg, not in the input dict."""
    return al.open_app.invoke({"query": query}, config=cfg)


# ---- Approval flow ----

def test_open_app_requires_approval_via_sse(tmp_path):
    """When SSE wiring is present, the SSE approval gate is used."""
    fake_exe = tmp_path / "discord.exe"
    fake_exe.write_text("dummy")  # exists for Path checks but never executed

    cfg, _q, _event, _state = _make_config(approval_value=True)

    def resolve():
        item = _q.get(timeout=2)
        assert item[0] == "approval_request"
        assert item[1]["kind"] == "app_launch"
        assert item[1]["query"] == "discord"
        assert item[1]["resolved_path"] == str(fake_exe)
        _state["value"] = True
        _event.set()

    t = threading.Thread(target=resolve, daemon=True)
    t.start()

    with patch.object(al, "_resolve_candidates", return_value=[(str(fake_exe), 100)]), \
         patch.object(al.os, "startfile") as mock_startfile:
        result = _invoke_open_app("discord", cfg)
    t.join(timeout=2)

    assert result.startswith("OK:"), result
    assert "discord" in result
    assert str(fake_exe) in result
    mock_startfile.assert_called_once_with(str(fake_exe))


def test_open_app_denied_returns_error(tmp_path):
    """When the human denies, no launch happens."""
    fake_exe = tmp_path / "discord.exe"
    fake_exe.write_text("dummy")
    cfg, _q, _event, state = _make_config(approval_value=False)

    def resolve():
        _q.get(timeout=2)
        state["value"] = False
        _event.set()

    t = threading.Thread(target=resolve, daemon=True)
    t.start()

    with patch.object(al, "_resolve_candidates", return_value=[(str(fake_exe), 100)]), \
         patch.object(al.os, "startfile") as mock_startfile:
        result = _invoke_open_app("discord", cfg)
    t.join(timeout=2)

    assert "denied" in result.lower(), result
    mock_startfile.assert_not_called()


def test_open_app_no_sse_wiring_denies_safely(tmp_path, monkeypatch):
    """Without SSE wiring AND with Tk patched to deny, the tool returns ERROR
    safely and does not launch.
    """
    fake_exe = tmp_path / "app.exe"
    fake_exe.write_text("dummy")
    cfg = {"configurable": {}}  # no SSE wiring

    # Block the Tk popup so it can never silently approve; force the
    # import + askyesno path to raise (caught by the secure fallback).
    import builtins
    real_import = builtins.__import__

    def _block_tk(name, *args, **kwargs):
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError("tkinter blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_tk)

    with patch.object(al, "_resolve_candidates", return_value=[(str(fake_exe), 100)]), \
         patch.object(al.os, "startfile") as mock_startfile:
        result = _invoke_open_app("anything", cfg)
    assert "ERROR" in result, f"expected denial error, got: {result!r}"
    mock_startfile.assert_not_called()


# ---- Error paths ----

def test_open_app_empty_query_returns_error():
    result = _invoke_open_app("", {"configurable": {}})
    assert "ERROR" in result
    assert "non-empty" in result.lower() or "required" in result.lower()


def test_open_app_no_candidates_returns_error():
    with patch.object(al, "_resolve_candidates", return_value=[]):
        result = _invoke_open_app("nonexistent", {"configurable": {}})
    assert "ERROR" in result
    assert "No application found" in result


def test_open_app_weak_match_returns_error():
    """All matches below score threshold -> treated as not found."""
    with patch.object(al, "_resolve_candidates", return_value=[("/some/path.exe", 5)]):
        result = _invoke_open_app("xyz", {"configurable": {}})
    assert "ERROR" in result


# ---- Search ranking (Windows) ----

def test_search_windows_prefers_start_menu(tmp_path, monkeypatch):
    """A Start Menu shortcut beats a same-name install dir entry."""
    start_menu = tmp_path / "StartMenu"
    start_menu.mkdir()
    shortcut = start_menu / "Discord.lnk"
    shortcut.write_text("shortcut")  # .lnk existence is what matters
    install = tmp_path / "Install"
    install.mkdir()
    (install / "Discord").mkdir()
    discord_exe = install / "Discord" / "Discord.exe"
    discord_exe.write_text("dummy")
    # Also create a fake PATH executable
    path_dir = tmp_path / "Path"
    path_dir.mkdir()
    (path_dir / "discord.exe").write_text("dummy")

    monkeypatch.setenv("PATH", str(path_dir))
    monkeypatch.setattr(al.sys, "platform", "win32")
    monkeypatch.setattr(al, "_windows_start_menu_dirs", lambda: [start_menu])
    monkeypatch.setattr(al, "_windows_install_dirs", lambda: [install])

    results = al._search_windows("discord")
    assert results, "expected at least one match"
    # Highest score should be the Start Menu shortcut.
    assert results[0][0] == str(shortcut), f"expected shortcut first, got {results[0]}"
    assert results[0][1] >= 50, f"shortcut score too low: {results[0][1]}"


def test_search_windows_nickname_normalization(tmp_path):
    """vscode nickname resolves to Visual Studio Code shortcut."""
    start_menu = tmp_path / "StartMenu"
    start_menu.mkdir()
    (start_menu / "Visual Studio Code.lnk").write_text("shortcut")
    with patch.object(al, "_windows_start_menu_dirs", return_value=[start_menu]), \
         patch.object(al, "_windows_install_dirs", return_value=[]):
        results = al._search_windows("vscode")
    assert results, "vscode nickname should match 'Visual Studio Code'"
    assert "Visual Studio Code.lnk" in results[0][0]


def test_search_windows_no_match_returns_empty(tmp_path):
    start_menu = tmp_path / "StartMenu"
    start_menu.mkdir()
    (start_menu / "Firefox.lnk").write_text("x")
    with patch.object(al, "_windows_start_menu_dirs", return_value=[start_menu]), \
         patch.object(al, "_windows_install_dirs", return_value=[]):
        results = al._search_windows("totally-unknown-app-xyz")
    assert results == []


# ---- Platform dispatch ----

def test_open_app_uses_subprocess_popen_on_non_windows(tmp_path):
    """On Linux/macOS, subprocess.Popen is used (os.startfile doesn't exist)."""
    fake_exe = tmp_path / "firefox"
    fake_exe.write_text("dummy")
    cfg, _q, _event, _state = _make_config(approval_value=True)

    def resolve():
        _q.get(timeout=2)
        _state["value"] = True
        _event.set()

    t = threading.Thread(target=resolve, daemon=True)
    t.start()

    with patch.object(al, "_resolve_candidates", return_value=[(str(fake_exe), 100)]), \
         patch.object(al, "sys") as msys, \
         patch("graph.tools.app_launcher.subprocess") as mock_sp:
        msys.platform = "linux"
        # os.startfile doesn't exist on Linux — make AttributeError if accessed.
        del al.os.startfile  # would raise AttributeError if used
        result = _invoke_open_app("firefox", cfg)
    t.join(timeout=2)

    assert result.startswith("OK:"), result
    mock_sp.Popen.assert_called_once()
    args = mock_sp.Popen.call_args.args[0]
    assert args == [str(fake_exe)]
