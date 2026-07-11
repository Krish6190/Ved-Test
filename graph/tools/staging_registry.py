"""Thread-safe in-memory staging registry for file edits.

This module provides the backing store for Ved's non-blocking, multi-file
file-edit review flow. File-modification tools (edit_file / overwrite_file)
can stage changes here instead of touching disk. Read tools can overlay
staged changes so models never see stale content. The UI (and the
chatbot.py worker thread) can approve or reject staged changes, at which
point the registry hands the tasks back to a callback that writes to disk.

Registry is keyed by session/thread id (VedState.active_thread_id) so
multiple concurrent conversations do not collide.
"""
from __future__ import annotations

from threading import Lock
from typing import Any, Callable, Dict, List, Optional


class _Session:
    """Per-session staging bucket."""

    def __init__(
        self,
        approval_event=None,
        approval_state: Optional[Dict[str, Any]] = None,
    ):
        self.approval_event = approval_event
        self.approval_state = approval_state
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()


class StagingRegistry:
    """Global, thread-safe registry of pending file edits keyed by session."""

    def __init__(self):
        self._sessions: Dict[str, _Session] = {}
        self._global_lock = Lock()

    def register_session(
        self,
        thread_id: str,
        approval_event=None,
        approval_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a staging bucket for the given session."""
        if not thread_id:
            return
        with self._global_lock:
            self._sessions[thread_id] = _Session(approval_event, approval_state)

    def unregister_session(self, thread_id: str) -> None:
        """Drop the staging bucket for a session (e.g. when response ends)."""
        if not thread_id:
            return
        with self._global_lock:
            self._sessions.pop(thread_id, None)

    def _get_session(self, thread_id: str) -> Optional[_Session]:
        if not thread_id:
            return None
        with self._global_lock:
            return self._sessions.get(thread_id)

    def has_session(self, thread_id: str) -> bool:
        """Return True if a staging bucket exists for the session."""
        if not thread_id:
            return False
        with self._global_lock:
            return thread_id in self._sessions

    def stage_edit(
        self,
        thread_id: str,
        tool_name: str,
        resolved_path: str,
        args: Dict[str, Any],
        preview: Dict[str, str],
    ) -> str:
        """Stage a file edit in memory and return a STAGED marker string.

        The returned marker starts with "STAGED: " so the caller can detect
        it and emit a `file_edit_approval_request` event to the UI.
        """
        session = self._get_session(thread_id)
        if session is None:
            return (
                "ERROR: File-edit staging session is not registered. "
                "The edit was not applied."
            )
        task = {
            "tool_name": tool_name,
            "path": resolved_path,
            "args": dict(args),
            "preview": dict(preview),
        }
        with session.lock:
            session.tasks[resolved_path] = task
        # Signal the worker thread that a decision may be waiting.
        if session.approval_event is not None:
            try:
                session.approval_event.set()
            except Exception:
                pass
        return (
            f"STAGED: {tool_name} on {resolved_path} is staged for review. "
            "Continuing without blocking..."
        )

    def get_overlay(self, thread_id: str, resolved_path: str, raw_content: str) -> str:
        """Return `raw_content` with the most recent staged edit applied.

        If the staged edit cannot be applied (e.g. edit_file old_text no
        longer matches), the raw content is returned with a warning marker.
        """
        session = self._get_session(thread_id)
        if session is None:
            return raw_content
        with session.lock:
            task = session.tasks.get(resolved_path)
        if task is None:
            return raw_content
        args = task.get("args", {}) or {}
        tool_name = task.get("tool_name", "")
        if tool_name == "edit_file":
            old_text = args.get("old_text", "") or ""
            new_text = args.get("new_text", "") or ""
            if old_text and old_text in raw_content:
                return raw_content.replace(old_text, new_text, 1)
            return raw_content + (
                "\n\n[VIRTUAL OVERLAY WARNING] Pending edit_file old_text "
                "no longer matches this file's on-disk content; the virtual "
                "overlay was not applied. Read the file again or re-issue "
                "the edit with a matching old_text."
            )
        if tool_name == "overwrite_file":
            return args.get("content", "") or ""
        return raw_content

    def get_tasks(self, thread_id: str) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of the staged tasks for a session."""
        session = self._get_session(thread_id)
        if session is None:
            return {}
        with session.lock:
            return dict(session.tasks)

    def apply_decision(
        self,
        thread_id: str,
        decision: Dict[str, Any],
        apply_callback: Callable[[Dict[str, Any]], str],
    ) -> Dict[str, Any]:
        """Apply a user decision to the staged tasks for a session.

        `decision` is a dict with keys:
          - action: "approve_all" | "reject_all" | "approve" | "reject"
          - paths: list of absolute paths to approve/reject (for per-file actions)

        `apply_callback` is called once per approved task and should write
        the change to disk. It receives the task dict and returns a result
        string.

        Returns a summary dict {"approved": [...], "rejected": [...]}.
        """
        session = self._get_session(thread_id)
        result = {"approved": [], "rejected": []}
        if session is None:
            return result

        action = (decision or {}).get("action", "reject_all")
        paths = (decision or {}).get("paths") or []

        with session.lock:
            snapshot = dict(session.tasks)
            approved_tasks: List[Dict[str, Any]] = []
            rejected_tasks: List[Dict[str, Any]] = []

            if action == "approve_all":
                approved_tasks = list(snapshot.values())
                session.tasks.clear()
            elif action == "reject_all":
                rejected_tasks = list(snapshot.values())
                session.tasks.clear()
            elif action == "approve":
                for p in paths:
                    t = snapshot.get(p)
                    if t is not None:
                        approved_tasks.append(t)
                        session.tasks.pop(p, None)
            else:  # reject / unknown
                for p in paths:
                    t = snapshot.get(p)
                    if t is not None:
                        rejected_tasks.append(t)
                        session.tasks.pop(p, None)

        for task in approved_tasks:
            try:
                res = apply_callback(task)
                result["approved"].append({"path": task.get("path"), "result": res})
            except Exception as exc:
                result["approved"].append(
                    {"path": task.get("path"), "result": f"ERROR: {type(exc).__name__}: {exc}"}
                )

        for task in rejected_tasks:
            result["rejected"].append({"path": task.get("path"), "result": "rejected"})

        return result


# Global singleton used by file-editing tools and the chatbot worker thread.
STAGING_REGISTRY = StagingRegistry()
