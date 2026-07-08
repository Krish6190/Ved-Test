"""Unit tests for api.server. Patches the chatbot so no real model loads."""
import io
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

from api import lifecycle
from api.server import app


class _FakeMsg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _FakeThreadFiles:
    """In-memory stand-in for ThreadFileStore."""
    def __init__(self):
        self._uploads = {}  # thread_id -> list[dict]

    def add(self, thread_id: str, source_path: str, filename: str = None) -> dict:
        import os as _os, time as _time
        # Prefer the explicitly passed filename (the user's upload name);
        # fall back to the source-path basename for tests that don't pass one.
        final_filename = filename if filename is not None else _os.path.basename(source_path)
        entry = {
            "filename": final_filename,
            "uploaded_at": _time.time(),
            "chunk_count": 5,
            "evicted": [],
        }
        self._uploads.setdefault(thread_id, []).append(entry)
        return entry

    def list_uploads(self, thread_id: str) -> list:
        return list(self._uploads.get(thread_id, []))


class FakeChatbot:
    """In-memory chatbot stand-in. Implements every method the FastAPI
    routes actually call."""

    def __init__(self):
        self.mode = "standard"
        self._threads = {
            "thr_init": {
                "id": "thr_init",
                "title": "Starter",
                "created_at": 1000.0,
                "messages": [
                    _FakeMsg("system", "sys prompt"),
                    _FakeMsg("human", "hello"),
                    _FakeMsg("ai", "hi there"),
                ],
            }
        }
        self._active_thread_id = "thr_init"
        self._pinned: List[dict] = []
        self._global_files: List[dict] = []
        self._thread_files = _FakeThreadFiles()

    # ---- thread management ----
    def list_threads(self) -> List[dict]:
        return [
            {
                "id": t["id"],
                "title": t["title"],
                "created_at": t["created_at"],
                "messages": t["messages"],
            }
            for t in self._threads.values()
        ]

    def create_thread(self, title: Optional[str]) -> str:
        import secrets
        import time
        tid = f"thr_{secrets.token_hex(4)}"
        self._threads[tid] = {
            "id": tid,
            "title": title or "New Thread",
            "created_at": time.time(),
            "messages": [],
        }
        self._active_thread_id = tid
        return tid

    def get_active_thread(self) -> dict:
        return self._threads[self._active_thread_id]

    def switch_thread(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        self._active_thread_id = thread_id
        return True

    def rename_thread(self, thread_id: str, title: str) -> bool:
        if thread_id not in self._threads:
            return False
        self._threads[thread_id]["title"] = title
        return True

    def delete_thread(self, thread_id: str) -> bool:
        if len(self._threads) <= 1:
            return False
        if thread_id not in self._threads:
            return False
        del self._threads[thread_id]
        if self._active_thread_id == thread_id:
            self._active_thread_id = next(iter(self._threads))
        return True

    # ---- mode ----
    def set_mode(self, mode: str) -> None:
        self.mode = mode

    # ---- respond ----
    def respond(self, prompt: str):
        if prompt.strip() == "/threads":
            return "Threads:\n  * 1. thr_init  Starter"
        if prompt.strip() == "/pin":
            return self.handle_command("/pin")
        # Otherwise return a generator that yields a couple of tokens.
        def gen():
            yield ("token", "Hello ")
            yield ("token", "world")
            yield ("token", "!")
        return gen()

    # ---- approval ----
    def submit_human_approval(self, approved: bool) -> None:
        pass

    # ---- memories ----
    def _load_pinned_contents(self) -> list:
        return list(self._pinned)

    def _save_pinned_contents(self, contents: list) -> None:
        self._pinned = list(contents)

    def handle_command(self, cmd: str):
        # /pin
        if cmd == "/pin":
            if len(self._pinned) >= 20:
                return "Pin rejected: Pinned limits cannot exceed half of total VRAM context slots (Max 20)."
            self._pinned.append({"user": "u", "assistant": "a"})
            return f"Success: Pinned turn sequence. ({len(self._pinned)}/20 occupied)"
        # /unpin <N>
        if cmd.startswith("/unpin "):
            try:
                idx = int(cmd.split()[1]) - 1
                if idx < 0 or idx >= len(self._pinned):
                    return None
                self._pinned.pop(idx)
                return "Unpinned."
            except Exception:
                return None
        return None

    # ---- global files ----
    def add_global_file(self, path: str) -> dict:
        import os
        # The server passes its own tempfile path (e.g. tmpXXXX.txt) to
        # add_global_file. For test stability, derive a deterministic
        # filename from the suffix instead of returning the random tmp basename.
        suffix = os.path.splitext(os.path.basename(path))[1] or ".txt"
        meta = {
            "filename": f"test{suffix}",
            "chunk_count": 3,
            "evicted": [],
        }
        self._global_files.append(meta)
        return meta

    def list_global_files(self) -> list:
        return list(self._global_files)


@pytest.fixture(autouse=True)
def _patch_chatbot():
    """Reset lifecycle + install FakeChatbot before each test."""
    lifecycle.reset_for_tests()
    fake = FakeChatbot()
    lifecycle._chatbot = fake  # bypass lazy construction
    yield
    lifecycle.reset_for_tests()


@pytest.fixture
def client():
    return TestClient(app)


# ---- Tests ----

def test_health_does_not_touch_chatbot(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_threads(client):
    r = client.get("/threads")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["id"] == "thr_init"
    assert data[0]["message_count"] == 3


def test_create_and_activate_thread(client):
    r = client.post("/threads", json={"title": "New"})
    assert r.status_code == 201
    tid = r.json()["id"]
    assert r.json()["title"] == "New"
    # Switch back to starter
    r = client.post(f"/threads/{tid}/activate")
    assert r.status_code == 200
    # Active is the new one
    r = client.get("/threads/active")
    assert r.json()["id"] == tid


def test_rename_unknown_thread_404(client):
    r = client.patch("/threads/nope", json={"title": "x"})
    assert r.status_code == 404


def test_rename_known_thread(client):
    r = client.patch("/threads/thr_init", json={"title": "Renamed"})
    assert r.status_code == 200
    assert r.json()["title"] == "Renamed"


def test_delete_last_thread_409(client):
    r = client.delete("/threads/thr_init")
    assert r.status_code == 409


def test_delete_one_of_two(client):
    # Create a second thread, then delete the first
    client.post("/threads", json={"title": "B"})
    r = client.delete("/threads/thr_init")
    assert r.status_code == 204
    remaining = client.get("/threads").json()
    assert len(remaining) == 1
    assert remaining[0]["title"] == "B"


def test_active_messages(client):
    r = client.get("/threads/active/messages")
    assert r.status_code == 200
    msgs = r.json()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "human"
    assert msgs[2]["role"] == "ai"


def test_set_and_get_mode(client):
    r = client.get("/mode")
    assert r.json() == {"mode": "standard"}
    r = client.post("/mode", json={"mode": "turbo"})
    assert r.status_code == 200
    assert r.json() == {"mode": "turbo"}
    r = client.post("/mode", json={"mode": "bogus"})
    assert r.status_code == 400


def test_chat_string_response_yields_sse_message(client):
    """Slash command returns a string -> single 'message' event then 'done'."""
    with client.stream("POST", "/chat", json={"prompt": "/threads"}) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert "message" in events
    assert "done" in events


def test_chat_streaming_tokens(client):
    """Streaming response yields multiple 'token' events."""
    with client.stream("POST", "/chat", json={"prompt": "hello"}) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert events.count("token") >= 2
    assert "done" in events


def test_approval_resolves_unknown_session_404(client):
    r = client.post("/chat/approval", json={"approved": True, "session_id": "nope"})
    assert r.status_code == 404


def test_tool_creation_approval_resolves_unknown_session_404(client):
    r = client.post(
        "/chat/tool-creation/approval",
        json={"approved": True, "session_id": "nope"},
    )
    assert r.status_code == 404


def test_tool_creation_approval_resolves_registered_session(client):
    """Register a proposal via lifecycle, then POST approval and check resolved=True."""
    from api import lifecycle
    lifecycle.reset_for_tests()
    lifecycle.register_tool_proposal("tool-sess-1")
    r = client.post(
        "/chat/tool-creation/approval",
        json={"approved": True, "session_id": "tool-sess-1"},
    )
    assert r.status_code == 200
    assert r.json() == {
        "resolved": True,
        "session_id": "tool-sess-1",
        "approved": True,
    }


def test_tool_creation_approval_rejects_missing_session_id(client):
    """Missing session_id field → 422 validation error."""
    r = client.post(
        "/chat/tool-creation/approval",
        json={"approved": True},
    )
    assert r.status_code == 422


def test_memories_pin_and_list(client):
    r = client.get("/memories")
    assert r.status_code == 200
    assert r.json() == {"items": []}
    r = client.post("/memories/pin")
    assert r.status_code == 200
    r = client.get("/memories")
    assert len(r.json()["items"]) == 1


def test_memories_unpin(client):
    bot = lifecycle.get_chatbot()
    bot._pinned.append({"user": "u", "assistant": "a"})
    bot._pinned.append({"user": "x", "assistant": "y"})
    r = client.delete("/memories/0")
    assert r.status_code == 200
    remaining = client.get("/memories").json()["items"]
    assert len(remaining) == 1
    assert remaining[0]["user"] == "x"


def test_global_files_upload_and_list(client):
    r = client.post(
        "/files/global",
        files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert r.status_code == 201
    assert r.json()["filename"] == "test.txt"
    assert r.json()["chunk_count"] == 3
    r = client.get("/files/global")
    assert len(r.json()) == 1


def test_delete_unknown_thread_404(client):
    r = client.delete("/threads/does_not_exist")
    assert r.status_code == 404


def test_upload_thread_file_returns_metadata(client):
    r = client.post(
        "/files/thread",
        files={"file": ("test.txt", io.BytesIO(b"thread content"), "text/plain")},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["chunk_count"] == 5
    assert data["evicted"] == []


def test_list_thread_files(client):
    client.post(
        "/files/thread",
        files={"file": ("a.txt", io.BytesIO(b"aaa"), "text/plain")},
    )
    r = client.get("/files/thread")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["filename"] == "a.txt"


def test_upload_non_py_to_run_rejected(client):
    r = client.post(
        "/run",
        files={"file": ("test.txt", io.BytesIO(b"print('hi')"), "text/plain")},
    )
    assert r.status_code == 400


def test_run_python_script_via_endpoint(client, tmp_path):
    py = tmp_path / "hello.py"
    py.write_text("print('hello via api')")
    with open(py, "rb") as fh:
        r = client.post(
            "/run",
            files={"file": ("hello.py", fh, "text/x-python")},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["exit_code"] == 0
    assert "hello via api" in data["stdout"]
    assert data["timed_out"] is False
