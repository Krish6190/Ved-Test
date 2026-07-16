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

from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

_MAX_VERSIONS = 5


def _read_disk_text(resolved_path: str) -> str:
    try:
        return Path(resolved_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _apply_task_to_text(base_text: str, task: Dict[str, Any]) -> str:
    """Apply one staged edit task onto ``base_text`` and return virtual file text."""
    args = task.get("args", {}) or {}
    tool_name = task.get("tool_name", "")
    if tool_name == "edit_file":
        old_text = args.get("old_text", "") or ""
        new_text = args.get("new_text", "") or ""
        if old_text and old_text in base_text:
            return base_text.replace(old_text, new_text, 1)
        return base_text + (
            "\n\n[VIRTUAL OVERLAY WARNING] Pending edit_file old_text "
            "no longer matches this file's staged content; the virtual "
            "overlay was not applied. Read the file again or re-issue "
            "the edit with a matching old_text."
        )
    if tool_name == "overwrite_file":
        return args.get("content", "") or ""
    return base_text


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
        self.code_versions: Dict[str, List[str]] = {}
        self.lock = Lock()


class StagingRegistry:
    """Global, thread-safe registry of pending file edits keyed by session.

    Tracks announced paths via a per-session set to prevent duplicate
    UI update events. The set is cleared when the session's tasks are
    inspected or flushed.
    """

    def __init__(self):
        self._sessions: Dict[str, _Session] = {}
        self._global_lock = Lock()
        self._announced_paths: Dict[str, set[str]] = {}

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
            versions = session.code_versions.setdefault(resolved_path, [])
            if versions:
                base = versions[-1]
            else:
                base = _read_disk_text(resolved_path)
            new_text = _apply_task_to_text(base, task)
            versions.append(new_text)
            while len(versions) > _MAX_VERSIONS:
                versions.pop(0)
            session.tasks[resolved_path] = task
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
        """Return the latest virtual file text for a staged path.

        Always pivots on ``code_versions[path][-1]`` when versions exist.
        """
        session = self._get_session(thread_id)
        if session is None:
            return raw_content

        with session.lock:
            versions = session.code_versions.get(resolved_path) or []
            if versions:
                return versions[-1]
            return raw_content

    def get_tasks(self, thread_id: str) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of the staged tasks for a session."""
        session = self._get_session(thread_id)
        if session is None:
            return {}
        with session.lock:
            return dict(session.tasks)

    def get_unannounced_paths(self, thread_id: str) -> List[str]:
        """Return paths that have staged tasks but have not yet been
        announced to the UI. The caller is expected to:

          1. Render these paths in the chat log (one announcement per path).
          2. Call ``mark_paths_announced`` with the same list to suppress
             future announcements until a new edit lands on the same path.

        Inner rolling lists are NEVER iterated here — only the keys of the
        tasks dict are exposed, which is cheap and avoids walking the
        5-element version history.
        """
        session = self._get_session(thread_id)
        if session is None:
            return []
        with self._global_lock:
            already = set(self._announced_paths.get(thread_id, set()))
        with session.lock:
            pending_paths = set(session.tasks.keys())
        unseen = sorted(pending_paths - already)
        return unseen

    def mark_paths_announced(self, thread_id: str, paths: List[str]) -> None:
        """Record `paths` as announced so subsequent calls to
        ``get_unannounced_paths`` skip them until a new edit resets them.

        Called by the chatbot.py stream consumer right after the chat
        log renders the announcement so a duplicate render in the same
        turn (e.g., when the executor adds more tasks) does not produce
        a second copy of the same path announcement.
        """
        if not thread_id or not paths:
            return
        with self._global_lock:
            bucket = self._announced_paths.setdefault(thread_id, set())
            for p in paths:
                bucket.add(p)

    def reset_announced_paths(self, thread_id: str) -> None:
        """Clear the announced-paths tracking for a session. Called after
        the user resolves (approve / reject) so a future edit to the same
        path is announced again."""
        if not thread_id:
            return
        with self._global_lock:
            self._announced_paths.pop(thread_id, None)

    def get_version_count(self, thread_id: str, resolved_path: str) -> int:
        """Return how many in-memory versions are stored for a staged path."""
        session = self._get_session(thread_id)
        if session is None:
            return 0
        with session.lock:
            return len(session.code_versions.get(resolved_path) or [])

    def rollback(self, thread_id: str, resolved_path: str) -> Dict[str, Any]:
        """Pop the newest staged version and expose the previous virtual text.

        Returns a dict with keys:
          - ok (bool)
          - reason (str | None)  present when ok is False; either
              "no_session" (session not registered), "no_versions"
              (no edits ever staged for this path), or "single_version"
              (only one staged edit — nothing to roll back further).
          - remaining_versions (int)
          - current_text (str | None)

        Callers should inspect `reason` to distinguish "the path has
        never been edited" from "only one edit exists, so rollback has
        nowhere to go". Previously both cases returned ok=False silently,
        which made UI feedback indistinguishable.
        """
        session = self._get_session(thread_id)
        result: Dict[str, Any] = {
            "ok": False,
            "reason": "no_session",
            "remaining_versions": 0,
            "current_text": None,
        }
        if session is None:
            return result

        with session.lock:
            versions = session.code_versions.get(resolved_path) or []
            if not versions:
                result["reason"] = "no_versions"
                return result
            if len(versions) == 1:
                result["reason"] = "single_version"
                result["remaining_versions"] = 1
                result["current_text"] = versions[0]
                return result

            versions.pop()
            current = versions[-1]
            session.code_versions[resolved_path] = versions
            task = session.tasks.get(resolved_path, {})
            if task:
                task = dict(task)
                task["tool_name"] = "overwrite_file"
                task["args"] = {
                    "path": resolved_path,
                    "content": current,
                }
                preview = dict(task.get("preview", {}) or {})
                preview["new"] = current[:300]
                task["preview"] = preview
                session.tasks[resolved_path] = task

            result["ok"] = True
            result["reason"] = None
            result["remaining_versions"] = len(versions)
            result["current_text"] = current
            return result

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
                session.code_versions.clear()
            elif action == "reject_all":
                rejected_tasks = list(snapshot.values())
                session.tasks.clear()
                session.code_versions.clear()
            elif action == "approve":
                for p in paths:
                    t = snapshot.get(p)
                    if t is not None:
                        approved_tasks.append(t)
                        session.tasks.pop(p, None)
                        session.code_versions.pop(p, None)
            else:  # reject / unknown
                for p in paths:
                    t = snapshot.get(p)
                    if t is not None:
                        rejected_tasks.append(t)
                        session.tasks.pop(p, None)
                        session.code_versions.pop(p, None)

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
