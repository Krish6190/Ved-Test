"""Tests for rolling cap-5 staging version history and rollback."""
from unittest.mock import patch

from graph.tools.staging_registry import STAGING_REGISTRY


def _register(thread_id: str):
    STAGING_REGISTRY.register_session(thread_id)


def _cleanup(thread_id: str):
    STAGING_REGISTRY.unregister_session(thread_id)


def test_stage_edit_caps_at_five_versions(tmp_path):
    thread_id = "thr_cap5"
    target = tmp_path / "cap.py"
    target.write_text("v0\n", encoding="utf-8")
    _register(thread_id)
    try:
        for i in range(1, 7):
            STAGING_REGISTRY.stage_edit(
                thread_id,
                "overwrite_file",
                str(target),
                {"path": str(target), "content": f"v{i}\n"},
                {"old": "", "new": f"v{i}"},
            )
        assert STAGING_REGISTRY.get_version_count(thread_id, str(target)) == 5
        overlay = STAGING_REGISTRY.get_overlay(thread_id, str(target), target.read_text())
        assert overlay == "v6\n"
    finally:
        _cleanup(thread_id)


def test_get_overlay_returns_latest_only(tmp_path):
    thread_id = "thr_latest"
    target = tmp_path / "latest.py"
    target.write_text("alpha", encoding="utf-8")
    _register(thread_id)
    try:
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "edit_file",
            str(target),
            {"path": str(target), "old_text": "alpha", "new_text": "beta"},
            {"old": "alpha", "new": "beta"},
        )
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "edit_file",
            str(target),
            {"path": str(target), "old_text": "beta", "new_text": "gamma"},
            {"old": "beta", "new": "gamma"},
        )
        overlay = STAGING_REGISTRY.get_overlay(thread_id, str(target), "alpha")
        assert overlay == "gamma"
    finally:
        _cleanup(thread_id)


def test_rollback_pops_newest(tmp_path):
    thread_id = "thr_rollback"
    target = tmp_path / "rollback.py"
    target.write_text("one", encoding="utf-8")
    _register(thread_id)
    try:
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "overwrite_file",
            str(target),
            {"path": str(target), "content": "two"},
            {"old": "one", "new": "two"},
        )
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "overwrite_file",
            str(target),
            {"path": str(target), "content": "three"},
            {"old": "two", "new": "three"},
        )
        result = STAGING_REGISTRY.rollback(thread_id, str(target))
        assert result["ok"] is True
        assert result["remaining_versions"] == 1
        assert result["current_text"] == "two"
        overlay = STAGING_REGISTRY.get_overlay(thread_id, str(target), target.read_text())
        assert overlay == "two"
    finally:
        _cleanup(thread_id)


def test_rollback_triggers_sync_rag_ingest(tmp_path):
    thread_id = "thr_rag_rollback"
    target = tmp_path / "rag.py"
    target.write_text("base", encoding="utf-8")
    _register(thread_id)
    try:
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "overwrite_file",
            str(target),
            {"path": str(target), "content": "first"},
            {"old": "base", "new": "first"},
        )
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "overwrite_file",
            str(target),
            {"path": str(target), "content": "second"},
            {"old": "first", "new": "second"},
        )

        import chatbot

        class _Bot:
            _active_thread_id = thread_id
            _file_edit_pending_lock = __import__("threading").Lock()
            _file_edit_pending_tasks = {}

            def _rag_chunker(self):
                return "text"

        bot = _Bot()
        with patch("graph.tools._common.ingest_path_to_thread_rag_sync") as mock_ingest:
            ok = chatbot.Chatbot.submit_file_rollback(bot, str(target))
        assert ok is True
        mock_ingest.assert_called_once()
        _, kwargs = mock_ingest.call_args
        assert kwargs.get("content") == "first"
        assert kwargs.get("thread_id") == thread_id
    finally:
        _cleanup(thread_id)


def test_rollback_on_single_version_fails_gracefully(tmp_path):
    thread_id = "thr_rollback_fail"
    target = tmp_path / "single.py"
    target.write_text("only", encoding="utf-8")
    _register(thread_id)
    try:
        STAGING_REGISTRY.stage_edit(
            thread_id,
            "overwrite_file",
            str(target),
            {"path": str(target), "content": "staged"},
            {"old": "only", "new": "staged"},
        )
        result = STAGING_REGISTRY.rollback(thread_id, str(target))
        assert result["ok"] is False
        assert STAGING_REGISTRY.get_version_count(thread_id, str(target)) == 1
        overlay = STAGING_REGISTRY.get_overlay(thread_id, str(target), target.read_text())
        assert overlay == "staged"
    finally:
        _cleanup(thread_id)
