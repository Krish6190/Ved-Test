"""Unit tests for the telemetry module + its auth and API integration."""
from __future__ import annotations
import json
import time
from pathlib import Path
import pytest
from telemetry import ACTIVE_TIMEOUT_SECONDS, Telemetry, telemetry as global_telemetry


# ----------------------------------------------------------------------- #
# Fixtures
# ----------------------------------------------------------------------- #

@pytest.fixture
def tmp_telemetry(tmp_path: Path) -> Telemetry:
    """Fresh Telemetry instance per test, with persistence in tmp_path."""
    return Telemetry(state_path=tmp_path / "telemetry.json")


@pytest.fixture(autouse=True)
def _isolate_global_telemetry(tmp_path):
    """Tests that touch the module-level singleton save/restore around it."""
    # Save original state-path
    original_path = global_telemetry.state_path
    global_telemetry.state_path = tmp_path / "global_telemetry.json"
    # Drop all sessions so tests don't leak into each other.
    with global_telemetry._lock:
        global_telemetry._sessions.clear()
    yield
    # Restore
    global_telemetry.state_path = original_path
    with global_telemetry._lock:
        global_telemetry._sessions.clear()


# ----------------------------------------------------------------------- #
# Core lifecycle
# ----------------------------------------------------------------------- #

class TestStartSession:
    def test_returns_session_id(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice", source="gui")
        assert isinstance(sid, str) and len(sid) >= 8

    def test_new_user_is_active(self, tmp_telemetry: Telemetry):
        tmp_telemetry.start_session(username="alice")
        assert tmp_telemetry.get_active_count() == 1

    def test_distinct_users_counted_separately(self, tmp_telemetry: Telemetry):
        tmp_telemetry.start_session(username="alice")
        tmp_telemetry.start_session(username="bob")
        assert tmp_telemetry.get_active_count() == 2

    def test_same_username_two_sessions_dedupes(self, tmp_telemetry: Telemetry):
        """A single user with two GUI windows is still one active user."""
        tmp_telemetry.start_session(username="alice")
        tmp_telemetry.start_session(username="alice", source="gui2")
        assert tmp_telemetry.get_active_count() == 1

    def test_explicit_session_id_reused(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice", session_id="fixed")
        sid2 = tmp_telemetry.start_session(username="alice", session_id="fixed")
        assert sid == sid2
        assert tmp_telemetry.get_total_sessions() == 1


class TestHeartbeat:
    def test_heartbeat_by_session_id(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice")
        first_hb = tmp_telemetry._sessions[sid].last_heartbeat
        time.sleep(0.02)
        tmp_telemetry.heartbeat(session_id=sid)
        assert tmp_telemetry._sessions[sid].last_heartbeat > first_hb

    def test_heartbeat_by_username(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice")
        first_hb = tmp_telemetry._sessions[sid].last_heartbeat
        time.sleep(0.02)
        tmp_telemetry.heartbeat(username="alice")
        assert tmp_telemetry._sessions[sid].last_heartbeat > first_hb

    def test_heartbeat_unknown_user_noop(self, tmp_telemetry: Telemetry):
        tmp_telemetry.heartbeat(username="ghost")
        assert tmp_telemetry.get_total_sessions() == 0


class TestEndSession:
    def test_end_by_session_id(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice")
        tmp_telemetry.end_session(session_id=sid)
        assert tmp_telemetry.get_active_count() == 0

    def test_end_by_username_removes_all_user_sessions(self, tmp_telemetry: Telemetry):
        tmp_telemetry.start_session(username="alice")
        tmp_telemetry.start_session(username="alice", source="gui2")
        tmp_telemetry.end_session(username="alice")
        assert tmp_telemetry.get_total_sessions() == 0

    def test_end_unknown_user_noop(self, tmp_telemetry: Telemetry):
        tmp_telemetry.end_session(username="ghost")
        assert tmp_telemetry.get_total_sessions() == 0


# ----------------------------------------------------------------------- #
# Active-user semantics
# ----------------------------------------------------------------------- #

class TestActiveSemantics:
    def test_expired_session_not_active(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice")
        # Pretend 10 minutes have passed.
        tmp_telemetry._sessions[sid].last_heartbeat = time.time() - 600
        assert tmp_telemetry.get_active_count() == 0

    def test_freshly_active_after_heartbeat(self, tmp_telemetry: Telemetry):
        sid = tmp_telemetry.start_session(username="alice")
        tmp_telemetry._sessions[sid].last_heartbeat = time.time() - 600
        assert tmp_telemetry.get_active_count() == 0
        tmp_telemetry.heartbeat(session_id=sid)
        assert tmp_telemetry.get_active_count() == 1

    def test_get_active_users_shape(self, tmp_telemetry: Telemetry):
        tmp_telemetry.start_session(username="alice", source="gui", mode="turbo")
        users = tmp_telemetry.get_active_users()
        assert len(users) == 1
        u = users[0]
        assert u["username"] == "alice"
        assert u["source"] == "gui"
        assert u["mode"] == "turbo"
        for k in ("session_id", "started_at", "last_heartbeat", "meta"):
            assert k in u

    def test_dedup_picks_most_recent_session(self, tmp_telemetry: Telemetry):
        a = tmp_telemetry.start_session(username="alice")
        time.sleep(0.02)
        b = tmp_telemetry.start_session(username="alice", source="gui2")
        users = tmp_telemetry.get_active_users()
        assert len(users) == 1
        assert users[0]["session_id"] == b
        # a is still tracked internally even though the user dedupes it.
        assert a in tmp_telemetry._sessions


# ----------------------------------------------------------------------- #
# Pruning
# ----------------------------------------------------------------------- #

class TestPrune:
    def test_prune_removes_expired(self, tmp_telemetry: Telemetry):
        sid_active = tmp_telemetry.start_session(username="alice")
        sid_stale = tmp_telemetry.start_session(username="bob")
        tmp_telemetry._sessions[sid_stale].last_heartbeat = time.time() - 9999
        removed = tmp_telemetry.prune_expired()
        assert removed == 1
        assert sid_active in tmp_telemetry._sessions
        assert sid_stale not in tmp_telemetry._sessions


# ----------------------------------------------------------------------- #
# Persistence
# ----------------------------------------------------------------------- #

class TestPersistence:
    def test_state_written_to_disk(self, tmp_telemetry: Telemetry):
        tmp_telemetry.start_session(username="alice")
        tmp_telemetry._writer_event_wait(timeout=2.0)
        assert tmp_telemetry.state_path.exists()
        raw = json.loads(tmp_telemetry.state_path.read_text(encoding="utf-8"))
        assert "sessions" in raw
        assert any(s["username"] == "alice" for s in raw["sessions"])

    def test_state_reloaded_on_init(self, tmp_path: Path):
        path = tmp_path / "t.json"
        first = Telemetry(state_path=path)
        first.start_session(username="alice")
        first._writer_event_wait(timeout=2.0)
        first.shutdown()
        second = Telemetry(state_path=path)
        assert second.get_active_count() == 1
        assert second.get_active_users()[0]["username"] == "alice"

    def test_stale_entries_pruned_on_load(self, tmp_path: Path):
        path = tmp_path / "t.json"
        first = Telemetry(state_path=path)
        sid = first.start_session(username="alice")
        first._sessions[sid].last_heartbeat = time.time() - 9999
        first._schedule_write()
        first._writer_event_wait(timeout=2.0)
        first.shutdown()
        second = Telemetry(state_path=path)
        assert second.get_active_count() == 0
        assert second.get_total_sessions() == 0


# ----------------------------------------------------------------------- #
# Session dataclass
# ----------------------------------------------------------------------- #

class TestSessionDataclass:
    def test_roundtrip_dict(self):
        from telemetry import Session
        s = Session(
            session_id="abc",
            username="alice",
            started_at=1.0,
            last_heartbeat=2.0,
            source="gui",
            mode="turbo",
            meta={"k": "v"},
        )
        d = s.to_dict()
        s2 = Session.from_dict(d)
        assert s2.session_id == "abc"
        assert s2.username == "alice"
        assert s2.started_at == 1.0
        assert s2.last_heartbeat == 2.0
        assert s2.source == "gui"
        assert s2.mode == "turbo"
        assert s2.meta == {"k": "v"}

    def test_from_dict_tolerates_missing_optional_fields(self):
        from telemetry import Session
        s = Session.from_dict({"session_id": "x", "username": "y"})
        assert s.session_id == "x"
        assert s.username == "y"
        assert s.source == "gui"   # default
        assert s.mode == "standard"

    def test_is_active_respects_timeout(self):
        from telemetry import Session
        s = Session(
            session_id="x",
            username="y",
            started_at=0.0,
            last_heartbeat=0.0,
        )
        # last_heartbeat is "now" → active by default
        s.last_heartbeat = time.time()
        assert s.is_active() is True
        # 9999s old with default 300s timeout → inactive
        s.last_heartbeat = time.time() - 9999
        assert s.is_active() is False


# ----------------------------------------------------------------------- #
# Auth integration
# ----------------------------------------------------------------------- #

class TestAuthIntegration:
    def test_successful_login_registers_session(self):
        import auth
        # Reset telemetry so this test is isolated from others.
        with global_telemetry._lock:
            global_telemetry._sessions.clear()
        db = {"admin": "admin"[::-1]}  # "nimda"
        assert auth.check_login("admin", "admin", db) is True
        assert global_telemetry.get_active_count() == 1
        users = global_telemetry.get_active_users()
        assert users[0]["username"] == "admin"

    def test_failed_login_does_not_register(self):
        import auth
        with global_telemetry._lock:
            global_telemetry._sessions.clear()
        db = {"admin": "nimda"}
        assert auth.check_login("admin", "wrong", db) is False
        assert global_telemetry.get_active_count() == 0

    def test_logout_ends_session(self):
        import auth
        with global_telemetry._lock:
            global_telemetry._sessions.clear()
        db = {"admin": "nimda"}
        auth.check_login("admin", "admin", db)
        assert global_telemetry.get_active_count() == 1
        auth.logout("admin")
        assert global_telemetry.get_active_count() == 0


# ----------------------------------------------------------------------- #
# FastAPI integration
# ----------------------------------------------------------------------- #

class TestTelemetryApi:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        with TestClient(app) as c:
            yield c

    def test_active_endpoint_empty(self, client, _isolate_global_telemetry):
        resp = client.get("/telemetry/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_count"] == 0
        assert data["active_users"] == []
        assert data["timeout_seconds"] == ACTIVE_TIMEOUT_SECONDS

    def test_active_endpoint_reflects_active_users(self, client):
        # Register a session via the heartbeat endpoint.
        client.post("/telemetry/heartbeat", json={"username": "alice"})
        resp = client.get("/telemetry/active")
        data = resp.json()
        assert data["active_count"] == 1
        assert data["active_users"][0]["username"] == "alice"

    def test_heartbeat_endpoint_returns_count(self, client):
        resp = client.post("/telemetry/heartbeat", json={"username": "alice"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["resolved"] is True
        assert "session_id" in body
        assert body["active_count"] >= 1


# ----------------------------------------------------------------------- #
# Helper: wait for the async writer to drain its queue.
# ----------------------------------------------------------------------- #

def _writer_event_wait(self, timeout: float = 2.0) -> bool:
    """Wait until the writer thread has drained the current queue."""
    # Implementation lives on the Telemetry instance, but pytest fixtures
    # above call it as ``tmp_telemetry._writer_event_wait``. Attach it
    # here so test code reads naturally.
    import threading as _t
    deadline = time.time() + timeout
    while time.time() < deadline:
        # The writer clears the event after processing a batch.
        if not self._write_event.is_set() and not self._write_queue:
            return True
        time.sleep(0.05)
    return False


# Bind the helper onto the class so fixtures can call it.
Telemetry._writer_event_wait = _writer_event_wait
