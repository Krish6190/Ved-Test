"""Manual smoke test for the Ved FastAPI server.

Run with:
    uvicorn api.server:app --port 8000   # in one terminal
    python api/smoke_test.py             # in another

Exits 0 if all checks pass, 1 otherwise. Requires httpx (already in
requirements.txt) and a running FastAPI server (does NOT start one).
"""
from __future__ import annotations
import asyncio
import io
import os
import sys
from typing import Any, Awaitable, Callable, List, Tuple
import httpx

API_URL = os.getenv("VED_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Each check is (name, async-callable-that-returns-None-on-pass-or-raises)
Check = Tuple[str, Callable[[httpx.AsyncClient], Awaitable[None]]]

_results: List[Tuple[str, bool, str]] = []


async def _run(name: str, fn: Callable[[httpx.AsyncClient], Awaitable[None]], client: httpx.AsyncClient) -> None:
    try:
        await fn(client)
        _results.append((name, True, ""))
        print(f"  [PASS] {name}", flush=True)
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  [FAIL] {name}: {e}", flush=True)


# ---- Individual checks ----

async def check_health(c: httpx.AsyncClient) -> None:
    r = await c.get("/health")
    r.raise_for_status()
    assert r.json() == {"status": "ok"}, f"unexpected: {r.text}"


async def check_list_threads(c: httpx.AsyncClient) -> None:
    r = await c.get("/threads")
    r.raise_for_status()
    assert isinstance(r.json(), list), "expected list"


async def check_create_and_delete_thread(c: httpx.AsyncClient) -> None:
    r = await c.post("/threads", json={"title": "Smoke"})
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    r = await c.post(f"/threads/{tid}/activate")
    r.raise_for_status()
    r = await c.delete(f"/threads/{tid}")
    assert r.status_code == 204, r.text


async def check_active_messages(c: httpx.AsyncClient) -> None:
    r = await c.get("/threads/active/messages")
    r.raise_for_status()
    assert isinstance(r.json(), list)


async def check_get_and_set_mode(c: httpx.AsyncClient) -> None:
    r = await c.get("/mode")
    r.raise_for_status()
    original = r.json()["mode"]
    r = await c.post("/mode", json={"mode": "standard"})
    r.raise_for_status()
    assert r.json()["mode"] == "standard"
    r = await c.post("/mode", json={"mode": "bogus"})
    assert r.status_code == 400, f"expected 400 for bogus mode, got {r.status_code}"
    # Restore original.
    await c.post("/mode", json={"mode": original})


async def check_chat_sse_with_slash_command(c: httpx.AsyncClient) -> None:
    """Slash commands return a single 'message' SSE event."""
    events: List[str] = []
    async with c.stream("POST", "/chat", json={"prompt": "/threads"}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert "message" in events, f"no 'message' event in {events}"
    assert "done" in events, f"no 'done' event in {events}"


async def check_memories_round_trip(c: httpx.AsyncClient) -> None:
    # List first.
    r = await c.get("/memories")
    r.raise_for_status()
    initial = len(r.json()["items"])
    # Try to pin (may fail if no conversation exchange — that's OK, just check status).
    r = await c.post("/memories/pin")
    assert r.status_code in (200, 400), r.text
    # List again — should be initial or initial+1.
    r = await c.get("/memories")
    r.raise_for_status()
    after = len(r.json()["items"])
    assert after in (initial, initial + 1), f"items grew unexpectedly: {initial} -> {after}"


async def check_global_files_upload_and_list(c: httpx.AsyncClient) -> None:
    files = {"file": ("smoke.txt", io.BytesIO(b"hello smoke"), "text/plain")}
    r = await c.post("/files/global", files=files)
    assert r.status_code == 201, r.text
    assert r.json()["filename"] == "smoke.txt"
    r = await c.get("/files/global")
    r.raise_for_status()
    assert isinstance(r.json(), list)


async def check_thread_file_upload_and_list(c: httpx.AsyncClient) -> None:
    files = {"file": ("smoke_thread.txt", io.BytesIO(b"thread smoke"), "text/plain")}
    r = await c.post("/files/thread", files=files)
    assert r.status_code == 201, r.text
    assert r.json()["filename"] == "smoke_thread.txt"
    r = await c.get("/files/thread")
    r.raise_for_status()
    assert isinstance(r.json(), list)


async def check_run_endpoint(c: httpx.AsyncClient) -> None:
    code = b"print('SMOKE_RUN_OK')\n"
    files = {"file": ("smoke_run.py", io.BytesIO(code), "text/x-python")}
    r = await c.post("/run", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exit_code"] == 0, body
    assert "SMOKE_RUN_OK" in body["stdout"], body


CHECKS: List[Check] = [
    ("health",            check_health),
    ("list threads",      check_list_threads),
    ("create+delete thr", check_create_and_delete_thread),
    ("active messages",   check_active_messages),
    ("get/set mode",      check_get_and_set_mode),
    ("chat SSE",          check_chat_sse_with_slash_command),
    ("memories",          check_memories_round_trip),
    ("global files",      check_global_files_upload_and_list),
    ("thread files",      check_thread_file_upload_and_list),
    ("run endpoint",      check_run_endpoint),
]


async def main() -> int:
    print(f"[Smoke] Target: {API_URL}")
    async with httpx.AsyncClient(base_url=API_URL, timeout=30.0) as client:
        # Quick reachability check.
        try:
            r = await client.get("/health", timeout=5.0)
            r.raise_for_status()
        except Exception as e:
            print(f"[Smoke] FATAL: cannot reach {API_URL}/health — {e}")
            print("[Smoke] Is the server running? Try:")
            print(f"         uvicorn api.server:app --port 8000")
            return 2

        for name, fn in CHECKS:
            await _run(name, fn, client)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print(f"\n[Smoke] Results: {passed} passed, {failed} failed (of {len(_results)} checks)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
