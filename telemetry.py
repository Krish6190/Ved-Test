"""Telemetry — lightweight active-user tracking for Ved.

Tracks how many users currently have an active session, who they are,
when they last interacted, and when they started. Sessions are
identified by a stable session_id and a username (when known). A
session is considered "active" if its last heartbeat is within
``ACTIVE_TIMEOUT_SECONDS``.

Design goals:
- Zero external dependencies (pure stdlib).
- Thread-safe — the GUI uses background threads for TTS, RAG, and
  graph execution, and the FastAPI server may serve requests
  concurrently.
- Crash-safe — state is persisted to ``data/telemetry.json`` so an
  abrupt exit doesn't leave stale "active" entries forever; on load,
  entries older than the timeout are pruned before counting.
- Non-blocking I/O — disk writes happen on a background thread so
  heartbeats don't stall the chat loop.

Usage:
    from telemetry import telemetry

    telemetry.start_session(username="alice")
    telemetry.heartbeat(username="alice")           # on activity
    telemetry.end_session(username="alice")         # on shutdown
    telemetry.get_active_count()                    # int
    telemetry.get_active_users()                    # list[dict]
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# A session is "active" if its last heartbeat was within this many seconds.
# Default: 5 minutes. Tunable via the VED_TELEMETRY_TIMEOUT env var.
ACTIVE_TIMEOUT_SECONDS = float(os.getenv("VED_TELEMETRY_TIMEOUT", "300"))

# Persistence file. Lives next to the other data stores so it survives
# restarts and is easy to find/inspect.
DEFAULT_STATE_PATH = Path("data/telemetry.json")


def _telemetry_disabled() -> bool:
    """Return True if the user opted out via the VED_TELEMETRY_DISABLED env var.

    Accepts the same truthy spellings as USE_CLOUD_API elsewhere in the
    codebase: "1", "true", "yes" (case-insensitive). Read on every call
    so toggling the env var and restarting flips the behavior without
    requiring code changes.
    """
    return os.getenv("VED_TELEMETRY_DISABLED", "").lower() in ("1", "true", "yes")


@dataclass
class Session:
    """One running client of Ved (one GUI window, one API client, etc.)."""
    session_id: str
    username: str
    started_at: float
    last_heartbeat: float
    source: str = "gui"   # "gui", "api", or any custom client tag
    mode: str = "standard"
    # Free-form metadata — version, hostname, anything useful.
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        # Tolerate older payloads missing newer optional fields.
        return cls(
            session_id=data.get("session_id", secrets.token_hex(8)),
            username=data.get("username", "anonymous"),
            started_at=float(data.get("started_at", time.time())),
            last_heartbeat=float(data.get("last_heartbeat", time.time())),
            source=data.get("source", "gui"),
            mode=data.get("mode", "standard"),
            meta=data.get("meta", {}) or {},
        )

    def is_active(self, now: Optional[float] = None, timeout: float = ACTIVE_TIMEOUT_SECONDS) -> bool:
        if now is None:
            now = time.time()
        return (now - self.last_heartbeat) <= timeout


class Telemetry:
    """Thread-safe active-user tracker with disk persistence.

    The singleton instance is exported as ``telemetry`` at the bottom of
    this module — import it directly:

        from telemetry import telemetry
    """

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE_PATH
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        # Dedicated writer thread + condition. Avoids blocking heartbeats
        # on disk I/O while still serializing writes.
        self._write_queue: list[dict] = []
        self._write_event = threading.Event()
        self._writer_stop = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, name="telemetry-writer", daemon=True)
        self._writer.start()
        self._load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start_session(
        self,
        username: str = "anonymous",
        source: str = "gui",
        mode: str = "standard",
        meta: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Register a new active session. Returns the session_id.

        If ``session_id`` is provided and already exists, the existing
        session is refreshed (heartbeat + username updated) rather than
        creating a duplicate.

        No-op (returns an empty string) when ``VED_TELEMETRY_DISABLED``
        is set to a truthy value in the environment.
        """
        if _telemetry_disabled():
            return ""
        sid = session_id or secrets.token_hex(8)
        now = time.time()
        with self._lock:
            existing = self._sessions.get(sid)
            if existing is not None:
                existing.username = username
                existing.source = source
                existing.mode = mode
                existing.meta = meta or {}
                existing.last_heartbeat = now
            else:
                self._sessions[sid] = Session(
                    session_id=sid,
                    username=username,
                    started_at=now,
                    last_heartbeat=now,
                    source=source,
                    mode=mode,
                    meta=meta or {},
                )
            self._schedule_write()
        return sid

    def heartbeat(
        self,
        session_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> None:
        """Update the last-activity timestamp for a session.

        Look-up order: ``session_id`` → ``username`` → no-op. When the
        lookup is by username, all sessions for that user are bumped
        (typical case: one GUI session per user).

        No-op when ``VED_TELEMETRY_DISABLED`` is set to a truthy value.
        """
        if _telemetry_disabled():
            return
        now = time.time()
        with self._lock:
            touched = False
            if session_id and session_id in self._sessions:
                self._sessions[session_id].last_heartbeat = now
                touched = True
            elif username:
                for s in self._sessions.values():
                    if s.username == username:
                        s.last_heartbeat = now
                        touched = True
            if touched:
                self._schedule_write()

    def end_session(
        self,
        session_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> None:
        """Remove a session. Same lookup rules as heartbeat().

        No-op when ``VED_TELEMETRY_DISABLED`` is set to a truthy value.
        """
        if _telemetry_disabled():
            return
        with self._lock:
            removed = False
            if session_id and session_id in self._sessions:
                del self._sessions[session_id]
                removed = True
            elif username:
                to_drop = [sid for sid, s in self._sessions.items() if s.username == username]
                for sid in to_drop:
                    del self._sessions[sid]
                    removed = True
            if removed:
                self._schedule_write()

    def get_active_sessions(self, now: Optional[float] = None) -> list[Session]:
        """Return all currently-active sessions (heartbeat within timeout)."""
        now = now if now is not None else time.time()
        with self._lock:
            return [s for s in self._sessions.values() if s.is_active(now)]

    def get_active_users(self, now: Optional[float] = None) -> list[dict]:
        """Return one entry per distinct active username."""
        sessions = self.get_active_sessions(now)
        # De-duplicate by username, keeping the most-recent session.
        by_user: dict[str, Session] = {}
        for s in sessions:
            cur = by_user.get(s.username)
            if cur is None or s.last_heartbeat > cur.last_heartbeat:
                by_user[s.username] = s
        out = []
        for s in by_user.values():
            out.append({
                "username": s.username,
                "session_id": s.session_id,
                "source": s.source,
                "mode": s.mode,
                "started_at": s.started_at,
                "last_heartbeat": s.last_heartbeat,
                "meta": s.meta,
            })
        out.sort(key=lambda r: r["username"])
        return out

    def get_active_count(self, now: Optional[float] = None) -> int:
        """Number of distinct active usernames right now."""
        return len(self.get_active_users(now))

    def get_total_sessions(self) -> int:
        """All sessions on disk, including expired ones (for debugging)."""
        with self._lock:
            return len(self._sessions)

    def prune_expired(self, now: Optional[float] = None) -> int:
        """Drop sessions whose heartbeat is older than the timeout. Returns count removed."""
        now = now if now is not None else time.time()
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if not s.is_active(now)]
            for sid in stale:
                del self._sessions[sid]
            if stale:
                self._schedule_write()
            return len(stale)

    def snapshot(self) -> dict:
        """Full state for debugging / CLI dump."""
        with self._lock:
            return {
                "active_count": self.get_active_count(),
                "active_users": self.get_active_users(),
                "total_sessions_tracked": len(self._sessions),
                "all_sessions": [s.to_dict() for s in self._sessions.values()],
                "timeout_seconds": ACTIVE_TIMEOUT_SECONDS,
            }

    # ------------------------------------------------------------------ #
    # Internal: persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Load state from disk. Missing/corrupt file → start empty."""
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            sessions = raw.get("sessions", [])
            with self._lock:
                self._sessions = {s["session_id"]: Session.from_dict(s) for s in sessions if s.get("session_id")}
            # Prune anything that already expired while the app was off.
            self.prune_expired()
        except Exception:
            # Corrupt file — ignore and start fresh. The next write will
            # overwrite it.
            with self._lock:
                self._sessions = {}

    def _schedule_write(self) -> None:
        """Snapshot current state and hand it to the writer thread."""
        # Snapshot under the lock so the writer sees a consistent view.
        with self._lock:
            snapshot = {
                "sessions": [s.to_dict() for s in self._sessions.values()],
                "saved_at": time.time(),
            }
        self._write_queue.append(snapshot)
        self._write_event.set()

    def _writer_loop(self) -> None:
        """Drain the write queue. Coalesces — only the latest snapshot is persisted."""
        while not self._writer_stop.is_set():
            # Wait until something is queued.
            self._write_event.wait(timeout=1.0)
            if self._writer_stop.is_set():
                break
            self._write_event.clear()
            # Drain queue, keep only the last snapshot.
            snapshot = None
            while self._write_queue:
                snapshot = self._write_queue.pop(0)
            if snapshot is None:
                continue
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.state_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                tmp.replace(self.state_path)
            except Exception:
                # Persistence is best-effort — never crash the app over telemetry.
                pass

    def shutdown(self) -> None:
        """Flush pending writes and stop the writer thread. Safe to call multiple times."""
        self._writer_stop.set()
        self._write_event.set()
        # Give the writer a moment to drain; it's a daemon thread so it
        # won't block process exit.
        self._writer.join(timeout=2.0)


# Module-level singleton — import this name from app code.
telemetry = Telemetry()
