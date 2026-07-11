# Ved — Local Multi-Format Offline AI Agent

A resource-constrained AI agent for Windows/Linux/macOS. Runs fully offline with local models on **12 GB VRAM / 32 GB RAM**; falls back to **4 GB VRAM / 16 GB RAM** when using OpenRouter/cloud API. Includes a Tkinter UI, FastAPI server, offline voice pipeline, per-thread RAG, and a planner-executor coding lane.

> ⚠️ **Telemetry.** Ved records active-user sessions in `data/telemetry.json` (local only, nothing leaves your network). Sessions are registered on GUI open or `/auth/login`, refreshed by heartbeats, and expire after 5 minutes of inactivity. To disable it entirely, add `VED_TELEMETRY_DISABLED=true` to `.env`. See [Telemetry](#telemetry) below.

---

## Table of contents

1. [What is Ved?](#what-is-ved)
2. [Features](#features)
3. [Project layout](#project-layout)
4. [Quick start](#quick-start)
5. [First-time setup](#first-time-setup)
6. [Environment setup](#environment-setup)
7. [Local model setup (Ollama)](#local-model-setup-ollama)
8. [Cloud model setup (OpenRouter)](#cloud-model-setup-openrouter)
9. [Audio files](#audio-files)
10. [RAG vector-DB index](#rag-vector-db-index)
11. [Running Ved](#running-ved)
12. [HTTP API](#http-api)
13. [Voice pipeline](#voice-pipeline)
14. [Threads, RAG, and file uploads](#threads-rag-and-file-uploads)
15. [Script execution](#script-execution)
16. [Modes](#modes)
17. [Telemetry](#telemetry)
18. [Slash commands](#slash-commands)
19. [Testing](#testing)
20. [Troubleshooting](#troubleshooting)

---

## What is Ved?

Ved is a single-user (or small-team) AI agent that runs entirely on your machine.

- **LangGraph orchestrator** (`chatbot.py`) with three lanes:
  - `standard` — 3B CPU casual chat, no tools.
  - `turbo` / `coder` — planner + executor with tools (8B / 7B planner + 3B executor).
- **Tkinter desktop UI** (`ui/`) with sticky mode chips, thread tabs, RAG attachment chips, and human-in-the-loop approval.
- **FastAPI HTTP server** (`api/`) exposing the same chatbot via Server-Sent Events.
- **Offline voice** (`voice/`) — wake-word → faster-whisper → LLM → Piper TTS, with mid-sentence barge-in.
- **Per-thread RAG** with FIFO eviction.
- **Script executor** (`api/runner.py`) for uploaded `.py` files.
- **Telemetry** (`telemetry.py`) — local-only active-user counter.

Everything is offline by default. Flip `USE_CLOUD_API=true` to route calls through OpenRouter.

---

## Features

- 100% offline by default; optional OpenRouter fallback.
- Deterministic model switching: `standard`, `turbo`, `coder`, `hibernate`.
- Per-thread RAG file uploads with FIFO eviction.
- Human-in-the-loop approval for filesystem/code/app tool calls.
- Interruptible offline voice pipeline.
- HTTP API + Tkinter GUI.
- Local telemetry (opt-out via `.env`).

---

## Project layout

```
ved/
├── __main__.py              # Bootstrapper: python __main__.py
├── chatbot.py               # Core LangGraph orchestrator
├── model_adapter.py         # ChatOllama / ChatOpenAI factory
├── command_processor.py     # Slash-command dispatcher
├── telemetry.py             # Active-user tracking
├── Modelfile.standard       # 3B casual-chat model
├── Modelfile.turbo          # 8B planner model
├── Modelfile.coder          # 7B coder planner model
├── Modelfile.executor       # 3B executor model
├── Modelfile.hibernate      # Empty profile (sleep mode)
├── data/                    # Persistent state
├── graph/                   # LangGraph nodes + state
├── ui/                      # Tkinter UI
├── voice/                   # Offline voice pipeline
├── api/                     # FastAPI HTTP layer
└── tests/                   # Pytest suite
```

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\pip install -r requirements.txt
# Linux/macOS: .venv/bin/pip install -r requirements.txt

cp .env.example .env
# Edit .env with your model paths / API keys.

.venv/bin/python __main__.py        # Launch GUI
# or
.venv/bin/python -m uvicorn api.server:app --port 8000   # Launch API
```

For full first-time setup, see below.

---

## First-time setup

1. **Install dependencies.** See [Quick start](#quick-start).
2. **Copy `.env.example` → `.env`** and fill in anything you need.
3. **Download the Piper voice** (~60 MB). Pick a voice at [piper-voices](https://rhasspy.github.io/piper-voices/), download both `.onnx` + `.onnx.json`, and place them in `voice/assets/`. Update `voice_file` / `voice_json` in `.env`.
4. **Pull Ollama models** (see [Local model setup](#local-model-setup-ollama)).
5. **(Optional)** Set `DB_PATH` for the RAG index (defaults to `data/vectordb/index.bin`).
6. **(Optional)** Set `VED_TELEMETRY_DISABLED=true` to disable telemetry.
7. **(Optional)** Set `USE_CLOUD_API=true` + `API_KEY` for OpenRouter.
8. **Launch.** `python __main__.py`.

---

## Environment setup

Copy `.env.example` → `.env` and uncomment what you need.

| Key | Purpose |
|---|---|
| `USE_CLOUD_API` | Route LLM calls to OpenRouter (`true` / unset). |
| `OPENROUTER_MODEL` | OpenRouter model slug (default: `poolside/laguna-m.1:free`). |
| `API_KEY` | OpenRouter API key. |
| `voice_file` / `voice_json` | Piper TTS model + config paths. |
| `wake_sound` | `.wav` played when switching to `turbo`. |
| `DB_PATH` | RAG vector-DB index file. |
| `VED_TELEMETRY_DISABLED` | Set `true` to disable telemetry. |
| `VED_TELEMETRY_TIMEOUT` | Idle seconds before a session drops (default `300`). |

---

## Local model setup (Ollama)

Install Ollama from <https://ollama.com/download>, then pull the models referenced by the Modelfiles:

```bash
# Executor (used by turbo + coder)
ollama pull qwen2.5:3b-instruct

# Standard / turbo planner
ollama pull qwen2.5:7b-instruct-q4_K_M

# Coder planner
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

To swap a model, edit the `FROM` line in the corresponding `Modelfile.{mode}` and re-pull.

---

## Cloud model setup (OpenRouter)

For machines that can't host local models:

```env
USE_CLOUD_API=true
OPENROUTER_MODEL=poolside/laguna-m.1:free
API_KEY=sk-or-v1-your-key-here
```

Get a free key at <https://openrouter.ai/keys>.

---

## Audio files

Ved needs three audio assets (all configured in `.env`):

| Asset | Default path | Provided? |
|---|---|---|
| Piper voice model | `voice/assets/*.onnx` | No — download from [piper-voices](https://rhasspy.github.io/piper-voices/). |
| Piper voice config | `voice/assets/*.onnx.json` | No — must match the `.onnx`. |
| Wake sound | `voice/assets/turbo_engine_short.wav` | Yes. |

The wake-word model (`alexa`) is downloaded automatically by `openwakeword` on first launch.

---

## RAG vector-DB index

Ved stores embeddings in a single binary pickle file. Set `DB_PATH` in `.env` (default: `data/vectordb/index.bin`). The file is created automatically on the first upload. To reset it, delete the file and restart.

---

## Running Ved

**GUI:**

```bash
.venv/bin/python __main__.py
```

**HTTP API:**

```bash
.venv/bin/python -m uvicorn api.server:app --port 8000
```

API health check: `curl http://127.0.0.1:8000/health`

---

## HTTP API

Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check. |
| GET/POST | `/threads` | List / create threads. |
| POST | `/chat` | Send a prompt, returns SSE. |
| POST | `/chat/approval` | Resolve a pending tool approval. |
| POST | `/mode` | Set mode (`standard`, `turbo`, `coder`, `hibernate`). |
| GET/POST | `/files/thread` | List / upload to active thread. |
| GET/POST | `/files/global` | List / upload to global store. |
| POST | `/run` | Upload and execute a `.py` file. |
| GET/POST | `/telemetry/active` | Active users. |

Full schema is in `api/schemas.py`.

---

## Voice pipeline

States: `WAKE_WORD` → `RECORDING` → `CONFIRMATION` → `PLAYING` → reset.

- Wake word: `alexa` (configurable in `voice/voice_module.py`).
- STT: `faster-whisper` `base` on CPU.
- TTS: Piper, interruptible.
- Ambient calibration on first launch.

---

## Threads, RAG, and file uploads

- Each thread has its own message history and RAG store.
- Upload `.txt`, `.md`, `.pdf`, `.docx`, code files, etc. to the active thread or global store.
- Oldest uploads are evicted FIFO when the per-thread chunk quota is exceeded.
- Use slash commands like `/new`, `/threads`, `/switch`, `/list`, `/upload-global`.

---

## Script execution

`POST /run` accepts a `.py` file and executes it in a fresh tempdir:

- Default timeout: 30 s (max 120 s via `?timeout_seconds=`).
- Output capped at 16 KiB per stream.
- CLI args via `args=foo bar`.

---

## Modes

| Mode | Planner/Chat | Executor | Tools | Use for |
|---|---|---|---|---|
| `standard` | 3B CPU | — | none | Casual chat, CPU-only. |
| `turbo` | 8B GPU | 3B | read/search/run/open-app | Quick Q&A, summaries, light automation. |
| `coder` | 7B GPU | 3B | full incl. edit/overwrite/propose_tool | Coding, refactoring, file edits. |
| `hibernate` | none | none | none | Sleep; flushes all models from VRAM. |

Switch via the UI chips or `POST /mode`.

---

## Telemetry

Ved records active sessions in `data/telemetry.json` (local file only — nothing is sent over the network).

- A session is registered when the GUI opens or `/auth/login` succeeds.
- Heartbeats refresh activity while the user is active.
- Sessions expire after `VED_TELEMETRY_TIMEOUT` seconds of inactivity (default 300).
- Distinct users are de-duplicated by username.

**To disable:** add `VED_TELEMETRY_DISABLED=true` to `.env`.

Query from Python (`telemetry` singleton) or HTTP:

```bash
curl http://127.0.0.1:8000/telemetry/active
curl -X POST http://127.0.0.1:8000/telemetry/heartbeat \
  -H "Content-Type: application/json" \
  -d '{"username": "api-user"}'
```

---

## Slash commands

| Command | Action |
|---|---|
| `/new [title]` | Create thread. |
| `/threads` | List threads. |
| `/switch <id>` | Switch thread. |
| `/rename <id> <title>` | Rename thread. |
| `/delete <id>` | Delete thread. |
| `/clear` | Clear visible chat log. |
| `/mode <mode>` | Switch mode. |
| `/sleep` / `/hibernate` | Hibernate. |
| `/wake` / `/resume` | Wake from hibernate. |
| `/upload-global` | Upload to global RAG. |
| `/run` | Run a `.py` script. |
| `/pin` / `/unpin <n>` / `/unpin_all` | Pin/unpin memories. |
| `/list` | List thread attachments. |
| `/memories` | Show pinned memories. |

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

The Linux test venv can be created with `requirements-dev.txt` (lighter deps, no TensorFlow/Piper voice stack).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: langchain_ollama` | Run `pip install -r requirements.txt`. |
| Can't reach Ollama | Start `ollama serve` and pull the Modelfile models, or enable `USE_CLOUD_API`. |
| Voice doesn't hear wake word | Delete `data/telemetry.json` and `voice/__pycache__/` and relaunch. |
| `/run` times out | Increase `timeout_seconds` (max 120). |
| Telemetry shows 0 users | Heartbeats may have expired; lower `VED_TELEMETRY_TIMEOUT` or check `data/telemetry.json`. |

---

## License

See repository root for license details.
