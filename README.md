# Ved — Local Multi-Format Offline AI Agent

A highly decoupled, resource-constrained AI Agent architecture running 100% offline within a strict 4GB VRAM / 16GB RAM hardware footprint. Features deterministic model switching loops, an isolated graphics-accelerated coding lane, and sticky-viewport desktop interface layers.

## Core Project Architecture Directory
- `__main__.py` — Application bootstrapper. Run with: `python __main__.py`
- `__init__.py` — Global configurations and hardware execution mode registers.
- `chatbot.py` — Core orchestration engine handling graph input/output loops.
- `command_processor.py` — Isolated non-LLM text command mixin matrix.
- `model_adapter.py` — Low-level ChatOllama integration configuration parser.
- `Modelfile.*` — Per-mode Ollama engine configuration settings profiles.
- `data/` — Local text cache stores (`memories.json` and `long_term_memory.json`).
- `graph/` — LangGraph orchestration workspace state machine pipelines.
  - `nodes.py` — Path A & Coder node workflow routing conditions logic.
  - `state.py` — Pydantic models and context window memory limit reducers.
- `ui/` — Modular window construction and layout interface components.
  - `window_base.py` — Screen capture shields and desktop window positioning rules.
  - `components.py` — Stationary headers and visual widget packing loops.
  - `gui.py` — Asynchronous thread event triggers and screen updates.
- `voice/` — Offline fast whisper speech transcription and Piper synthesis pipelines.
  - `assets/` — Local sound files, voice files, and audio assets.

## HTTP API

The chatbot is exposed over HTTP via FastAPI so that a React/Node UI can
talk to the backend without importing any Python. Start the server:

```bash
.venv/Scripts/python.exe -m uvicorn api.server:app --port 8000
```

Then point your JS client at `http://127.0.0.1:8000`. The full endpoint
map and SSE event shape are documented in
[`.kimchi/docs/fastapi-plan.md`](.kimchi/docs/fastapi-plan.md).

Quick sanity check from another terminal:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"/threads"}'
# event: message
# data: {"text": "Threads:\n  * 1. thr_xxx  New Thread", "session_id": "..."}
#
# event: done
# data: {"session_id": "..."}
```

A manual end-to-end smoke test against a running server:

```bash
python api/smoke_test.py
```

The Tkinter desktop UI (`ui/gui.py`) is unchanged — it still talks to the
backend directly. The HTTP layer is for the future React frontend and
any other non-Python client (curl, scripts, mobile).

**Thread-scoped file upload.** `POST /files/thread` ingests a file into
the *active* thread's RAG store with FIFO eviction when the per-thread
chunk quota is exceeded; the response lists any evicted filenames.
List current thread attachments with `GET /files/thread`. The Tkinter
UI's drag-and-drop chips map to this endpoint — files are persisted,
not transient.

**Script execution.** `POST /run` accepts a `.py` file upload and
executes it via subprocess (30s default, override up to 120s via
`?timeout_seconds=N`). Returns JSON with `exit_code`, `stdout`, `stderr`,
`timed_out`, `duration_seconds`, and `truncated_*` flags. Output capped
at 16 KiB per stream. Working directory is a fresh tempdir.
