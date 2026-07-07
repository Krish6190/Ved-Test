"""FastAPI server wrapping the Ved chatbot.

Routes are documented in /mnt/c/Users/krish/OneDrive/Desktop/ved/.kimchi/docs/fastapi-plan.md.
The chatbot instance is obtained lazily via `api.lifecycle.get_chatbot()`.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api import lifecycle, runner
from api.schemas import (
    ApprovalReq,
    ChatReq,
    CreateThreadReq,
    GlobalFileOut,
    HealthOut,
    MemoriesOut,
    MemoryPinItem,
    MessageOut,
    ModeOut,
    RenameThreadReq,
    RunOut,
    ToolCreationApprovalReq,
    SetModeReq,
    TelemetryOut,
    ActiveUserOut,
    ThreadFileOut,
    ThreadOut,
)


app = FastAPI(title="Ved HTTP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # desktop app, same-machine React
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    # Lazy: do not construct chatbot here. First /chat or /mode call will.
    pass


@app.on_event("shutdown")
async def _shutdown() -> None:
    pass


# ---- Health ----

@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    """Does NOT touch the chatbot; safe to call while Ollama is down."""
    return HealthOut(status="ok")


# ---- Telemetry ----

@app.get("/telemetry/active", response_model=TelemetryOut)
async def telemetry_active() -> TelemetryOut:
    """Return active-user count and the list of currently-active sessions.

    A session counts as active if its last heartbeat was within the
    telemetry timeout (default 5 minutes, tunable via the
    ``VED_TELEMETRY_TIMEOUT`` env var). Heartbeats are emitted by the GUI
    on every prompt and on every successful ``/chat`` call to this
    server, so an idle client naturally drops off the count after the
    timeout.

    This endpoint never touches the chatbot, so it's safe to call while
    Ollama is down or the chatbot hasn't been constructed yet.
    """
    from telemetry import telemetry as _telemetry
    users = _telemetry.get_active_users()
    from telemetry import ACTIVE_TIMEOUT_SECONDS
    return TelemetryOut(
        active_count=len(users),
        active_users=[ActiveUserOut(**u) for u in users],
        timeout_seconds=ACTIVE_TIMEOUT_SECONDS,
    )


@app.post("/telemetry/heartbeat")
async def telemetry_heartbeat(req: Request) -> dict:
    """Record an API-client heartbeat. The caller may pass a username via
    JSON body (``{"username": "alice"}``) or ``X-Ved-Username`` header.

    If neither is provided, the session is recorded as ``"anonymous"``.
    Returns the new active-user count.
    """
    from telemetry import telemetry as _telemetry
    username = "anonymous"
    try:
        body = await req.json()
        if isinstance(body, dict) and body.get("username"):
            username = str(body["username"])
    except Exception:
        pass
    if username == "anonymous":
        username = req.headers.get("X-Ved-Username", "anonymous")
    sid = _telemetry.start_session(username=username, source="api")
    _telemetry.heartbeat(session_id=sid)
    return {"resolved": True, "session_id": sid, "active_count": _telemetry.get_active_count()}


# ---- Helpers ----

def _thread_to_out(t: dict) -> ThreadOut:
    return ThreadOut(
        id=t["id"],
        title=t.get("title", "New Thread"),
        created_at=t.get("created_at", 0.0),
        message_count=len(t.get("messages", []) or []),
    )


# ---- Threads ----

@app.get("/threads", response_model=List[ThreadOut])
async def list_threads() -> List[ThreadOut]:
    bot = lifecycle.get_chatbot()
    out: List[ThreadOut] = []
    for t in bot.list_threads():
        full = bot._threads.get(t["id"], {})  # type: ignore[attr-defined]
        out.append(ThreadOut(
            id=t["id"],
            title=t.get("title", "New Thread"),
            created_at=t.get("created_at", 0.0),
            message_count=len(full.get("messages", []) or []),
        ))
    return out


@app.post("/threads", response_model=ThreadOut, status_code=status.HTTP_201_CREATED)
async def create_thread(req: CreateThreadReq) -> ThreadOut:
    bot = lifecycle.get_chatbot()
    bot.create_thread(req.title)
    return _thread_to_out(bot.get_active_thread())


@app.get("/threads/active", response_model=ThreadOut)
async def get_active_thread() -> ThreadOut:
    bot = lifecycle.get_chatbot()
    return _thread_to_out(bot.get_active_thread())


@app.get("/threads/active/messages", response_model=List[MessageOut])
async def get_active_thread_messages() -> List[MessageOut]:
    bot = lifecycle.get_chatbot()
    msgs = bot.get_active_thread().get("messages", []) or []
    out: List[MessageOut] = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "human"
        elif isinstance(m, AIMessage):
            role = "ai"
        elif hasattr(m, "role"):
            # Test fixture compat: objects with explicit .role attribute.
            role = str(getattr(m, "role"))
        else:
            role = type(m).__name__.lower()
        content = m.content if isinstance(m.content, str) else str(m.content)
        out.append(MessageOut(role=role, content=content))
    return out


@app.post("/threads/{thread_id}/activate", response_model=ThreadOut)
async def activate_thread(thread_id: str) -> ThreadOut:
    bot = lifecycle.get_chatbot()
    if not bot.switch_thread(thread_id):
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    return _thread_to_out(bot.get_active_thread())


@app.patch("/threads/{thread_id}", response_model=ThreadOut)
async def rename_thread(thread_id: str, req: RenameThreadReq) -> ThreadOut:
    bot = lifecycle.get_chatbot()
    if thread_id not in bot._threads:  # type: ignore[attr-defined]
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    if not bot.rename_thread(thread_id, req.title):
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    # Active thread must remain unchanged — fetch the renamed record by id.
    full = bot._threads.get(thread_id, {})  # type: ignore[attr-defined]
    return ThreadOut(
        id=thread_id,
        title=req.title,
        created_at=full.get("created_at", 0.0),
        message_count=len(full.get("messages", []) or []),
    )


@app.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: str) -> None:
    bot = lifecycle.get_chatbot()
    if thread_id not in bot._threads:  # type: ignore[attr-defined]
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    if not bot.delete_thread(thread_id):
        # delete_thread returns False only when len(_threads) <= 1 now.
        raise HTTPException(status_code=409, detail="Cannot delete the last remaining thread.")
    return None


# ---- Mode ----

@app.get("/mode", response_model=ModeOut)
async def get_mode() -> ModeOut:
    bot = lifecycle.get_chatbot()
    return ModeOut(mode=bot.mode)


@app.post("/mode", response_model=ModeOut)
async def set_mode(req: SetModeReq) -> ModeOut:
    from __init__ import MODES
    if req.mode not in MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {MODES}",
        )
    bot = lifecycle.get_chatbot()
    try:
        bot.set_mode(req.mode)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ModeOut(mode=bot.mode)


# ---- Chat (SSE streaming) ----

def _sse(event: str, data: dict) -> str:
    """Format one SSE event. Each event ends with two newlines per spec."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(req: ChatReq) -> StreamingResponse:
    bot = lifecycle.get_chatbot()
    session_id = uuid.uuid4().hex

    # Large-paste handling (mirrors the Tkinter UI's behavior in
    # gui_rag_worker._ingest_payload): if the user pastes a big block,
    # save the full text into the thread's RAG store and replace the
    # prompt with a short reference + a tail excerpt. The LLM can call
    # retrieve_rag if it needs the full version.
    LARGE_PASTE_THRESHOLD = 1700
    if len(req.prompt) > LARGE_PASTE_THRESHOLD:
        try:
            import time as _t, secrets as _s
            source_label = f"UserPaste_{int(_t.time())}_{_s.token_hex(3)}"
            bot.save_user_input_to_thread_rag(req.prompt, source_label)
            tail = req.prompt[-200:].replace("\n", " ")
            req = ChatReq(
                prompt=(
                    f"[The user pasted {len(req.prompt)} characters. "
                    f"Full text saved to thread RAG under '{source_label}'. "
                    "Use retrieve_rag if you need the full version.]\n\n"
                    f"...(tail of paste): ...{tail}"
                ),
                attachments=req.attachments,
            )
        except Exception:
            # If RAG save fails, fall through with the original prompt.
            pass

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            response = bot.respond(req.prompt)
        except Exception as e:
            yield _sse("error", {"message": str(e), "session_id": session_id}).encode("utf-8")
            yield _sse("done", {"session_id": session_id}).encode("utf-8")
            return

        # Case 1: slash command returned a string immediately.
        if isinstance(response, str):
            yield _sse("message", {"text": response, "session_id": session_id}).encode("utf-8")
            yield _sse("done", {"session_id": session_id}).encode("utf-8")
            return

        # Case 2: streaming generator. Bridge sync generator → async SSE via a queue.
        loop = asyncio.get_running_loop()
        bridge: asyncio.Queue = asyncio.Queue()

        def pump() -> None:
            try:
                for item in response:
                    loop.call_soon_threadsafe(bridge.put_nowait, item)
            except Exception as e:
                loop.call_soon_threadsafe(bridge.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(bridge.put_nowait, None)  # sentinel

        threading.Thread(target=pump, daemon=True).start()

        approval_registered = False
        tool_proposal_registered = False
        try:
            while True:
                item = await bridge.get()
                if item is None:
                    break
                if not isinstance(item, tuple):
                    # Plain token string.
                    yield _sse("token", {"text": str(item), "session_id": session_id}).encode("utf-8")
                    continue
                event_type = item[0]
                payload = item[1] if len(item) > 1 else None
                if event_type == "token":
                    yield _sse("token", {"text": str(payload), "session_id": session_id}).encode("utf-8")
                elif event_type == "approval_request":
                    if not approval_registered:
                        lifecycle.register_approval(session_id)
                        approval_registered = True
                    try:
                        pass_num = int((payload or {}).get("pass", 0))
                    except Exception:
                        pass_num = 0
                    yield _sse(
                        "approval_request",
                        {"pass": pass_num, "session_id": session_id},
                    ).encode("utf-8")
                elif event_type == "tool_creation_proposal":
                    if not tool_proposal_registered:
                        lifecycle.register_tool_proposal(session_id)
                        # Also stamp the chatbot's session_id so propose_tool's
                        # wait loop can be resolved by the right caller.
                        try:
                            if hasattr(bot, "_tool_creation_state"):
                                bot._tool_creation_state["session_id"] = session_id
                        except Exception:
                            pass
                        tool_proposal_registered = True
                    yield _sse(
                        "tool_creation_proposal",
                        {"session_id": session_id, **(payload or {})},
                    ).encode("utf-8")
                elif event_type == "tool_call":
                    yield _sse(
                        "tool_call",
                        {"session_id": session_id, **(payload or {})},
                    ).encode("utf-8")
                elif event_type == "mode_switch":
                    # Informational — no approval gate.
                    yield _sse(
                        "mode_switch",
                        {"session_id": session_id, **(payload or {})},
                    ).encode("utf-8")
                elif event_type == "error":
                    yield _sse("error", {"message": str(payload), "session_id": session_id}).encode("utf-8")
        finally:
            if approval_registered:
                lifecycle.discard_approval(session_id)
            if tool_proposal_registered:
                lifecycle.discard_tool_proposal(session_id)
            yield _sse("done", {"session_id": session_id}).encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---- Approval ----

@app.post("/chat/approval")
async def submit_approval(req: ApprovalReq) -> dict:
    found = lifecycle.resolve_approval(req.session_id, req.approved)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"No pending approval for session: {req.session_id}",
        )
    return {"resolved": True, "session_id": req.session_id, "approved": req.approved}


@app.post("/chat/tool-creation/approval")
async def submit_tool_creation_approval(req: ToolCreationApprovalReq) -> dict:
    """Resolve a pending tool-creation proposal emitted by `propose_tool`.

    The chat SSE handler dispatches `event: tool_creation_proposal` with
    `session_id` + the proposed code; the UI shows a modal and POSTs here
    with the human's decision. On approval, `propose_tool` writes the file,
    imports the new module, and registers the new tool in VED_TOOLS.
    """
    found = lifecycle.resolve_tool_proposal(req.session_id, req.approved)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"No pending tool-creation proposal for session: {req.session_id}",
        )
    return {"resolved": True, "session_id": req.session_id, "approved": req.approved}


# ---- Memories (pinned) ----

@app.get("/memories", response_model=MemoriesOut)
async def list_memories() -> MemoriesOut:
    bot = lifecycle.get_chatbot()
    saved = bot._load_pinned_contents()
    items: List[MemoryPinItem] = []
    for entry in saved:
        if isinstance(entry, dict):
            items.append(
                MemoryPinItem(
                    user=entry.get("user", ""),
                    assistant=entry.get("assistant", ""),
                )
            )
        else:
            items.append(MemoryPinItem(user=str(entry), assistant=""))
    return MemoriesOut(items=items)


@app.post("/memories/pin")
async def pin_last_turn() -> dict:
    bot = lifecycle.get_chatbot()
    result = bot.handle_command("/pin")
    if result is None:
        raise HTTPException(status_code=400, detail="No conversation exchange found to pin.")
    return {"result": result}


@app.delete("/memories/{index}")
async def unpin_memory(index: int) -> dict:
    bot = lifecycle.get_chatbot()
    cmd = f"/unpin {index + 1}"  # /unpin is 1-indexed
    result = bot.handle_command(cmd)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Index error. Range 1 to {len(bot._load_pinned_contents())}.",
        )
    return {"result": result}


# ---- Global files ----

@app.get("/files/global", response_model=List[GlobalFileOut])
async def list_global_files() -> List[GlobalFileOut]:
    bot = lifecycle.get_chatbot()
    items: List[GlobalFileOut] = []
    for entry in bot.list_global_files():
        items.append(
            GlobalFileOut(
                filename=entry.get("filename", "unknown"),
                chunk_count=int(entry.get("chunk_count", 0)),
                evicted=[],  # list_uploads doesn't return evicted; that's only on add()
            )
        )
    return items


@app.post("/files/global", response_model=GlobalFileOut, status_code=status.HTTP_201_CREATED)
async def upload_global_file(file: UploadFile = File(...)) -> GlobalFileOut:
    bot = lifecycle.get_chatbot()
    # Stream upload to a temp file (don't load whole file into memory).
    suffix = Path(file.filename or "").suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        meta = bot.add_global_file(tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return GlobalFileOut(
        filename=file.filename or meta.get("filename", "unknown"),
        chunk_count=int(meta.get("chunk_count", 0)),
        evicted=list(meta.get("evicted", []) or []),
    )


# ---- Thread-scoped files ----

@app.get("/files/thread", response_model=List[ThreadFileOut])
async def list_thread_files() -> List[ThreadFileOut]:
    """List uploads attached to the active thread (oldest first)."""
    bot = lifecycle.get_chatbot()
    active = bot.get_active_thread()
    thread_id = active["id"]
    if not hasattr(bot, "_thread_files") or bot._thread_files is None:
        return []
    raw = bot._thread_files.list_uploads(thread_id)
    out: List[ThreadFileOut] = []
    for entry in raw:
        out.append(ThreadFileOut(
            filename=entry.get("filename", "unknown"),
            chunk_count=int(entry.get("chunk_count", 0)),
            evicted=[],
            uploaded_at=float(entry.get("uploaded_at", 0.0)),
        ))
    return out


@app.post("/files/thread", response_model=ThreadFileOut, status_code=status.HTTP_201_CREATED)
async def upload_thread_file(file: UploadFile = File(...)) -> ThreadFileOut:
    """Upload a file to the ACTIVE thread's RAG index.

    The file is parsed, chunked, embedded, and stored under the active
    thread's scope. FIFO eviction applies if the thread's chunk quota is
    exceeded; evicted filenames are returned so the UI can warn the user.
    """
    bot = lifecycle.get_chatbot()
    active = bot.get_active_thread()
    thread_id = active["id"]
    suffix = Path(file.filename or "").suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if not hasattr(bot, "_thread_files") or bot._thread_files is None:
            raise HTTPException(status_code=500, detail="Thread file store not initialized.")
        try:
            entry = bot._thread_files.add(thread_id, tmp_path, filename=file.filename or "")
        except TypeError:
            entry = bot._thread_files.add(thread_id, tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return ThreadFileOut(
        filename=file.filename or entry.get("filename", "unknown"),
        chunk_count=int(entry.get("chunk_count", 0)),
        evicted=list(entry.get("evicted", []) or []),
        uploaded_at=float(entry.get("uploaded_at", time.time())),
    )


# ---- /run: script execution ----

@app.post("/run", response_model=RunOut)
async def run_script(
    file: UploadFile = File(...),
    args: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> RunOut:
    """Upload a .py file and execute it via subprocess.

    Args:    - args: optional CLI args (whitespace-split)
             - timeout_seconds: max 120 (default 30)
    Output:  - JSON with exit_code, stdout, stderr, timed_out, duration_seconds,
               truncated_stdout, truncated_stderr
    Working directory is a fresh tempdir so scripts cannot pollute the project root.
    """
    if not file.filename or not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py files are supported.")
    suffix = Path(file.filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        with tempfile.TemporaryDirectory() as workdir:
            cli_args = args.split() if args else []
            timeout = min(timeout_seconds, 120) if timeout_seconds else 30
            result = await runner.run_python_script(
                source_path=tmp_path,
                args=cli_args,
                timeout_seconds=timeout,
                cwd=workdir,
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Run failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return RunOut(**result)


# ---- Global exception handler ----

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
