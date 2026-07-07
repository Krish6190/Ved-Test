# Ved — Local Multi-Format Offline AI Agent

A highly decoupled, resource-constrained AI Agent architecture running 100% offline within a strict 4 GB VRAM / 16 GB RAM hardware footprint. Features deterministic model-switching loops, an isolated graphics-accelerated coding lane, sticky-viewport desktop UI layers, a fully offline voice pipeline (wake-word → speech-to-text → LLM → text-to-speech), per-thread RAG file uploads, a thread-scoped script executor, an HTTP API for non-Python clients, and a lightweight **telemetry system** for tracking active users.

> **Heads up — Telemetry.** Ved records active-user sessions in `data/telemetry.json` so you can see how many users are currently using your deployment. Sessions are registered when a GUI window opens (or on a successful `/auth/login`), refreshed by heartbeats while the user is active, and dropped automatically after 5 minutes of inactivity. See [Telemetry](#telemetry--active-user-tracking) below for full details, including how to disable it.

---

## Table of contents

1. [What is Ved?](#what-is-ved)
2. [Features](#features)
3. [Project layout](#project-layout)
4. [Quick start](#quick-start)
5. [Environment setup](#environment-setup)
6. [Local model setup (Ollama)](#local-model-setup-ollama)
7. [Cloud model setup (OpenRouter)](#cloud-model-setup-openrouter)
8. [Audio files — what they are and how to get them](#audio-files--what-they-are-and-how-to-get-them)
9. [Running Ved](#running-ved)
10. [HTTP API](#http-api)
11. [Voice pipeline](#voice-pipeline)
12. [Threads, RAG, and file uploads](#threads-rag-and-file-uploads)
13. [Script execution (`/run` and `POST /run`)](#script-execution-run-and-post-run)
14. [Modes (`standard` / `turbo` / `coder` / `hibernate`)](#modes-standard--turbo--coder--hibernate)
15. [Telemetry — active-user tracking](#telemetry--active-user-tracking)
16. [Slash commands](#slash-commands)
17. [Testing](#testing)
18. [Troubleshooting](#troubleshooting)

---

## What is Ved?

Ved is a single-user (or small-team) AI agent you run on your own machine. It bundles:

- a **chat orchestrator** (`chatbot.py`) that drives a LangGraph state machine with two execution lanes — a general conversational lane and a coder lane that can read, edit, search, and execute code;
- a **Tkinter desktop UI** (`ui/`) with a sticky-header chip strip, thread tabs, attachment chips, a human-in-the-loop approval bar, and a hidden-from-screen-capture window;
- a **FastAPI HTTP server** (`api/`) exposing the same chatbot to React/Node/curl/mobile clients via Server-Sent Events;
- a **fully offline voice pipeline** (`voice/`) using `openwakeword` + `faster-whisper` + Piper TTS, interruptible mid-sentence;
- **per-thread RAG** with FIFO eviction so each conversation can ingest documents without bloating global context;
- a **script executor** (`api/runner.py` + `POST /run`) that runs uploaded `.py` files in a fresh tempdir with a 30–120 s timeout;
- a **telemetry layer** (`telemetry.py`) that counts active users in real time, persists state across restarts, and exposes both a Python API and an HTTP endpoint.

Everything runs locally by default. If you don't have a GPU or can't host Ollama locally, you can flip one env var and route every LLM call through **OpenRouter** instead — see [Cloud model setup](#cloud-model-setup-openrouter).

---

## Features

- **100% offline by default.** No telemetry, no cloud calls, no phone-home unless you opt in.
- **Deterministic model switching** between four modes: `standard`, `turbo`, `coder`, `hibernate`. Each mode reads its own `Modelfile.{mode}` so swapping models is a one-line change.
- **Cloud fallback.** Set `USE_CLOUD_API=true` to route every call to OpenRouter — useful when your hardware can't host a local LLM.
- **Isolated graphics-accelerated coding lane.** The `coder` mode loads a 7B reasoning model on the GPU while keeping the small executor model on CPU, so editing and chatting don't fight for VRAM.
- **Sticky-viewport desktop UI.** Window is hidden from screen capture and stays anchored above the taskbar; mode chips and thread tabs survive a session.
- **Threaded conversations.** Each thread has its own message history, RAG store, and pinned-memory bucket. Switch between them via tabs.
- **Per-thread RAG with FIFO eviction.** Drop `.pdf`, `.docx`, `.txt`, code files, etc. into the active thread; oldest uploads are evicted when the per-thread chunk quota is exceeded.
- **Human-in-the-loop approval.** Every tool call that touches the filesystem, executes code, or launches an app waits for an explicit Yes/No click from the user.
- **Interruptible voice.** Mid-sentence barge-in on TTS; ambient-noise calibration on first launch.
- **HTTP API.** `/chat` (SSE streaming), `/threads`, `/mode`, `/files/thread`, `/files/global`, `/memories`, `/run`, `/telemetry/active`, `/telemetry/heartbeat`.
- **Telemetry.** Real-time active-user count, persisted to disk, thread-safe, exposed via Python API and HTTP.

---

## Project layout

```
ved/
├── __main__.py              # Application bootstrapper. Run with `python __main__.py`.
├── __init__.py              # Version + global mode registers (standard/turbo/coder/hibernate).
├── chatbot.py               # Core orchestration engine (graph input/output loops, threads, RAG).
├── command_processor.py     # Isolated non-LLM slash-command mixin matrix.
├── model_adapter.py         # Low-level ChatOllama / ChatOpenAI factory; reads Modelfile.{mode}.
├── auth.py                  # Password hashing + login helper (also records telemetry on success).
├── telemetry.py             # Active-user tracking: start/heartbeat/end, persistence, thread-safe.
├── Modelfile.standard       # 3B general-conversation model profile (CPU-friendly).
├── Modelfile.turbo          # 3B fast-executor profile used by the coder pipeline.
├── Modelfile.coder          # 7B reasoning-coder profile (GPU-accelerated).
├── Modelfile.hibernate      # Profile used when the assistant is asleep (no model load).
├── database.json            # Local user → password DB (rot13-style reversed strings).
├── requirements.txt         # All Python dependencies.
├── data/                    # Persistent state — threads, memories, plans, telemetry, RAG indexes.
│   ├── threads.json
│   ├── memories.json
│   ├── long_term_memory.json
│   ├── telemetry.json       # ← telemetry state lives here.
│   └── ...
├── graph/                   # LangGraph orchestration workspace.
│   ├── __init__.py          #   build_graph() entry point.
│   ├── state.py             #   Pydantic models + message-limit reducers.
│   └── nodes/               #   Path A (chat) + Path C (coder) node implementations.
├── ui/                      # Modular Tkinter window construction.
│   ├── window_base.py       #   Screen-capture shields + desktop positioning rules.
│   ├── components.py        #   Sticky headers, mode chips, thread tabs, attachment chips.
│   ├── gui_rag_worker.py    #   RAG ingest + render engine used by the GUI.
│   └── gui.py               #   Async event triggers + screen updates; main Tk loop.
├── voice/                   # Offline audio pipeline.
│   ├── voice_module.py      #   VoiceSystem orchestrator (wake + STT + TTS).
│   ├── audio_loop.py        #   Single unified input-stream processing loop.
│   ├── audio_processors.py  #   State-machine logic (WAKE_WORD / RECORDING / CONFIRMATION / PLAYING).
│   ├── audio_utils.py       #   Interruptible TTS worker + helpers.
│   └── assets/              #   The 3 audio files (see below).
├── api/                     # FastAPI HTTP layer.
│   ├── server.py            #   Route definitions + SSE bridge.
│   ├── lifecycle.py         #   Lazy chatbot instantiation + approval-gate registry.
│   ├── runner.py            #   Subprocess executor for `/run` and `POST /run`.
│   ├── schemas.py           #   Pydantic request/response models.
│   └── smoke_test.py        #   End-to-end smoke test against a running server.
└── tests/                   # Pytest suite.
```

---

## First-time setup (recommended walkthrough)

The repo is set up so a fresh clone can install and run, but a few assets are intentionally **not** shipped with Git (they're too heavy or user-specific). Walk through this list on a new machine before you launch Ved for the first time:

### 1. Clone and install

```bash
git clone https://github.com/Krish6190/Ved-Test.git
cd ved
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux / macOS:
.venv/bin/pip install -r requirements.txt
```

### 2. Copy `.env.example` → `.env`

```bash
# Windows:
copy .env.example .env
# Linux / macOS:
cp .env.example .env
```

Open `.env` and uncomment / edit the keys you need (cloud API key, telemetry toggle, etc.). Everything is commented inline — see [Environment setup](#environment-setup) below for the full reference.

### 3. Download the Piper TTS voice (~60 MB)

The Piper voice model and its config (`voice_file` + `voice_json` in `.env`) are **gitignored** because each `.onnx` is around 60 MB and they don't belong in version control. Download them once into `voice/assets/`:

1. Go to <https://rhasspy.github.io/piper-voices/> and pick a voice. The project defaults to **en_GB-southern_english_female-low**.
2. Download **both** files for that voice — the `.onnx` and the matching `.onnx.json`.
3. Drop them into `voice/assets/`.
4. Make sure `.env` points at them:
   ```env
   voice_file=voice/assets/en_GB-southern_english_female-low.onnx
   voice_json=voice/assets/en_GB-southern_english_female-low.onnx.json
   ```
5. If you picked a different voice, rename the keys in `.env` to match your filenames.

Without these two files Ved will fail to start the voice pipeline (the GUI still works — it just won't be able to speak).

The wake-sound file (`turbo_engine_short.wav`) **is** shipped with the repo — you don't need to download it.

### 4. (Optional) Pick where the RAG vector-DB index lives

Ved creates the vector-DB index file automatically the first time you upload a document — but the **path** is yours to choose. See [RAG vector-DB index](#rag-vector-db-index) below for the full guide. The default in `.env.example` points at `data/vectordb/index.bin` inside the project.

### 5. (Optional) Enable or disable telemetry

Telemetry is **on by default** and writes only to `data/telemetry.json` on your machine — nothing leaves your network. To turn it off, add this to `.env`:

```env
VED_TELEMETRY_DISABLED=true
```

See [Telemetry](#telemetry--active-user-tracking) for the full details.

### 6. (Optional) Add a cloud API key

If your hardware can't host Ollama, uncomment these lines in `.env` and paste your OpenRouter key:

```env
USE_CLOUD_API=true
OPENROUTER_MODEL=qwen/qwen-2.5-coder-7b-instruct
API_KEY=sk-or-v1-your-key-here
```

Get a free key at <https://openrouter.ai/keys>. See [Cloud model setup](#cloud-model-setup-openrouter).

### 7. Launch

```bash
.venv/Scripts/python.exe __main__.py            # Windows
.venv/bin/python __main__.py                    # Linux / macOS
```

---

## Quick start (TL;DR)

If you've already done the first-time setup once, future runs are just:

```bash
.venv/Scripts/python.exe __main__.py            # Windows
.venv/bin/python __main__.py                    # Linux / macOS
```

That's it — Ved opens a Tkinter window with the chat input, mode chips, thread tabs, and a mic button. Type or click the mic to start talking.

---

## Environment setup

Ved reads configuration from a `.env` file at the project root (it loads automatically via `python-dotenv` on startup, with a built-in fallback parser if `python-dotenv` isn't installed). **Start from `.env.example`** — it documents every key. Copy it to `.env` and uncomment what you need:

| Key | Default | Purpose |
|---|---|---|
| `USE_CLOUD_API` | `false` | When `true`, skip local Ollama and route every LLM call to OpenRouter. See [Cloud model setup](#cloud-model-setup-openrouter). |
| `OPENROUTER_MODEL` | `qwen/qwen-2.5-coder-7b-instruct` | Which OpenRouter model to call when `USE_CLOUD_API=true`. Use any free or paid model slug. |
| `API_KEY` | _none_ | Your OpenRouter API key. Get one at <https://openrouter.ai/keys>. |
| `voice_file` | `voice/assets/en_GB-southern_english_female-low.onnx` | Piper TTS voice model. **Not shipped with the repo** — download separately. |
| `voice_json` | `voice/assets/en_GB-southern_english_female-low.onnx.json` | Piper voice config (must match `voice_file`). **Not shipped with the repo**. |
| `wake_sound` | `voice/assets/turbo_engine_short.wav` | Sound played when switching into `turbo` mode. Shipped with the repo. |
| `DB_PATH` | `data/vectordb/index.bin` | Path for the RAG vector-DB index file. Ved creates the parent directory on first use. See [RAG vector-DB index](#rag-vector-db-index). |
| `VED_TELEMETRY_DISABLED` | _unset (telemetry ON)_ | Set to `true` / `1` / `yes` to disable telemetry entirely. See [Telemetry](#telemetry--active-user-tracking). |
| `VED_TELEMETRY_TIMEOUT` | `300` | Seconds before an idle session drops off the active-user count. |
| `VED_USERNAME` | _none_ | Optional. If set, Ved uses this as the GUI telemetry username (falls back to OS env `USERNAME`, then `anonymous`). |

A minimal `.env` for a cloud-only install:

```env
USE_CLOUD_API=true
OPENROUTER_MODEL=qwen/qwen-2.5-coder-7b-instruct
API_KEY=sk-or-v1-your-key-here
```

A minimal `.env` for a fully-local install (no cloud):

```env
# Leave USE_CLOUD_API unset / false — Ved will talk to Ollama at http://localhost:11434.
```

A minimal `.env` for users who don't want telemetry:

```env
VED_TELEMETRY_DISABLED=true
```

---

## Local model setup (Ollama)

Ved expects a local Ollama server with the models named in `Modelfile.{mode}`. Install Ollama from <https://ollama.com/download>, then pull the models the Modelfiles reference:

```bash
# Standard + hibernate + turbo + executor share this 3B model
ollama pull qwen2.5:3b-instruct

# Coder lane uses this 7B reasoning model (q4_K_M fits in ~5 GB VRAM)
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

If you want to use different models, edit the `FROM` line in each `Modelfile.{mode}` and re-pull that model — no code changes required.

Then start Ollama (`ollama serve` runs by default after install) and launch Ved:

```bash
.venv/Scripts/python.exe __main__.py
```

---

## Cloud model setup (OpenRouter)

> **Use this if your hardware can't host a local LLM** (no GPU, too little RAM, or you're on a machine where installing Ollama isn't an option). The project's own `.env` ships with an OpenRouter key as a working example.

1. Create a free account at <https://openrouter.ai/> and grab an API key from <https://openrouter.ai/keys>.
2. Pick a model. The project defaults to `qwen/qwen-2.5-coder-7b-instruct` (free tier available), but any OpenRouter chat model works. Some popular free choices:
   - `qwen/qwen-2.5-coder-7b-instruct`
   - `meta-llama/llama-3.3-70b-instruct:free`
   - `google/gemini-2.0-flash-exp:free`
3. Add the following to your `.env` at the project root:

   ```env
   USE_CLOUD_API=true
   OPENROUTER_MODEL=qwen/qwen-2.5-coder-7b-instruct
   API_KEY=sk-or-v1-your-key-here
   ```

4. Run Ved normally. The `model_adapter.py` factory detects `USE_CLOUD_API=true`, builds a `langchain_openai.ChatOpenAI` pointed at `https://openrouter.ai/api/v1`, and routes every LLM call through it. No Ollama required.

5. To switch back to local: delete (or comment out) `USE_CLOUD_API=true` and restart.

The cloud path uses the same tool-calling contract as the local path, so all features (file ops, code execution, RAG retrieval) work identically. You only pay for tokens; no data leaves the cloud unless you explicitly uploaded something.

---

## Audio files — what they are and how to get them

Ved's voice pipeline uses **3 audio files**, all referenced from `.env` and resolved relative to the `voice/` directory. **Two of them are not shipped with the repo** — they're gitignored because each Piper voice model is ~60 MB. You need to download them yourself on first install:

| Key in `.env` | Default path | What it is | Where to get it | Shipped with repo? |
|---|---|---|---|---|
| `voice_file` | `voice/assets/en_GB-southern_english_female-low.onnx` | Piper TTS voice model (the synthesized speech voice). ~60 MB. | <https://rhasspy.github.io/piper-voices/> — pick any `.onnx` + matching `.onnx.json` pair (e.g. `en_US-lessac-medium`). | **No — gitignored.** Download on first install. |
| `voice_json` | `voice/assets/en_GB-southern_english_female-low.onnx.json` | Piper voice config (phoneme map, sample rate, etc.) — **must match** the `.onnx` filename. ~4 KB. | Same Piper voices page — always comes paired with the `.onnx`. | **No — gitignored.** Download with the `.onnx`. |
| `wake_sound` | `voice/assets/turbo_engine_short.wav` | The "turbo engine" sound effect played when you switch into `turbo` mode. ~550 KB. | Already in the repo. You can swap any `.wav` you like and update the `wake_sound` path in `.env`. | **Yes — tracked on GitHub.** |

### Step-by-step: download the Piper voice on first install

1. Go to <https://rhasspy.github.io/piper-voices/> and pick a voice. The project defaults to **en_GB-southern_english_female-low**.
2. Download **both** files for that voice — the `.onnx` model and the matching `.onnx.json` config. They always come as a pair.
3. Drop them into `voice/assets/` so Ved can find them.
4. Make sure `.env` points at them:
   ```env
   voice_file=voice/assets/en_GB-southern_english_female-low.onnx
   voice_json=voice/assets/en_GB-southern_english_female-low.onnx.json
   ```
5. If you picked a different voice, rename the keys in `.env` to match your filenames.

> **Without these two files Ved's voice pipeline will fail to start.** The GUI still works — it just won't be able to speak. If you don't need voice, you can ignore this section and launch normally; the voice module is opt-in via the 🎙 button.

### Optional: swap the wake sound

- Drop any short `.wav` you like into `voice/assets/` (a chime, a click, a startup sound — whatever you want to hear when turbo mode kicks in).
- Update `wake_sound` in `.env` to point at it.
- Set `wake_sound=` (empty) in `.env` if you want to disable the sound entirely.

### Wake-word model

- The wake-word model (`alexa`) is downloaded automatically by `openwakeword` on first launch — no manual setup needed.
- If you want a custom wake word, edit `self.wake_phrase = "alexa"` in `voice/voice_module.py` and add the matching `.tflite`/`.onnx` model.

---

## RAG vector-DB index

Ved stores document embeddings for RAG retrieval in a single binary index file (a Python `pickle` containing the registry + NumPy matrix). The path is controlled by the `DB_PATH` key in `.env`.

**You don't need to create this file yourself** — Ved creates it automatically the first time you upload a document to a thread or to the global store. The parent directory is also auto-created (`os.makedirs(..., exist_ok=True)` in `graph/rag/vector_engine.py`), so you only need to set the path.

### Pick a location that makes sense for your machine

| OS | Recommended `DB_PATH` |
|---|---|
| Windows | `DB_PATH=C:\Users\<you>\VectorDB\index.bin` |
| Linux | `DB_PATH=/home/<you>/.local/share/ved/vectordb/index.bin` |
| macOS | `DB_PATH=/Users/<you>/.local/share/ved/vectordb/index.bin` |
| Portable / per-project (default) | `DB_PATH=data/vectordb/index.bin` |

### How to set it

Open `.env` and uncomment / edit the `DB_PATH` line:

```env
# Windows:
DB_PATH=C:\Users\you\VectorDB\index.bin

# Linux / macOS:
DB_PATH=/home/you/.local/share/ved/vectordb/index.bin
```

Save `.env` and restart Ved. On first document upload, Ved creates the file at the new path. To migrate an existing index, just copy `index.bin` from the old location to the new one.

### Resetting the index

Delete the file at `DB_PATH` and restart Ved. The next upload will recreate it empty.

---

## Running Ved

### Desktop GUI

```bash
.venv/Scripts/python.exe __main__.py            # Windows
.venv/bin/python __main__.py                    # Linux / macOS
```

The GUI window is hidden from screen capture (`ui/window_base.py`) and stays sticky above the taskbar. On launch:

1. A telemetry session is registered for `VED_USERNAME` (or `USERNAME`, or `anonymous`).
2. The chat input, mode chips (`standard` / `turbo` / `coder` / `hibernate`), and thread tabs are rendered.
3. The voice pipeline starts in the background — press the 🎙 button or say the wake word (`alexa`) to talk.

### HTTP API

```bash
.venv/Scripts/python.exe -m uvicorn api.server:app --port 8000
```

The server uses the **same** chatbot instance pattern as the GUI. It does **not** require the GUI to be running — you can run either, or both, side by side.

Quick health check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

End-to-end smoke test:

```bash
.venv/Scripts/python.exe api/smoke_test.py
```

---

## HTTP API

Full endpoint map (also in `.kimchi/docs/fastapi-plan.md`):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. Never touches the chatbot. |
| `GET` | `/threads` | List all conversation threads. |
| `POST` | `/threads` | Create a new thread (`{"title": "..."}`). |
| `GET` | `/threads/active` | Get the active thread. |
| `GET` | `/threads/active/messages` | Get all messages in the active thread. |
| `POST` | `/threads/{id}/activate` | Switch to a thread. |
| `PATCH` | `/threads/{id}` | Rename a thread. |
| `DELETE` | `/threads/{id}` | Delete a thread (refuses if it's the last one). |
| `GET` | `/mode` | Get current mode. |
| `POST` | `/mode` | Set mode (`{"mode": "coder"}`). |
| `POST` | `/chat` | Send a prompt. Returns SSE stream: `message` / `token` / `approval_request` / `done`. |
| `POST` | `/chat/approval` | Resolve a pending human approval (`{"session_id": "...", "approved": true}`). |
| `POST` | `/chat/tool-creation/approval` | Approve/reject a tool-creation proposal. |
| `GET` | `/memories` | List pinned long-term memories. |
| `POST` | `/memories/pin` | Pin the last user/assistant exchange. |
| `DELETE` | `/memories/{index}` | Unpin a memory. |
| `GET` | `/files/global` | List globally-scoped RAG uploads. |
| `POST` | `/files/global` | Upload a file to the global RAG store. |
| `GET` | `/files/thread` | List uploads attached to the active thread. |
| `POST` | `/files/thread` | Upload a file to the **active** thread's RAG store (FIFO eviction on quota). |
| `POST` | `/run` | Upload a `.py` file and execute it via subprocess (30 s default, max 120 s). |
| **`GET`** | **`/telemetry/active`** | **Active-user count + list of active sessions.** |
| **`POST`** | **`/telemetry/heartbeat`** | **Record an API-client heartbeat.** |

Quick chat example:

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"/threads"}'
# event: message
# data: {"text": "Threads:\n  * 1. thr_xxx  New Thread", "session_id": "..."}
#
# event: done
# data: {"session_id": "..."}
```

---

## Voice pipeline

The voice module (`voice/voice_module.py`) runs a single unified audio loop with four states:

```
WAKE_WORD  ──(wake phrase detected)──▶  RECORDING  ──(silence)──▶  CONFIRMATION
   ▲                                                                          │
   │                                                                          ▼
   └────────────────────(reset)───  PLAYING  ◀────(TTS done)──────────────  │
```

- **Wake word:** `openwakeword` listens for the configured phrase (`alexa` by default) with `vad_threshold=0.25`. Configurable via `voice/voice_module.py`.
- **Speech-to-text:** `faster-whisper` `base` model on CPU, int8 quant, 4 threads. ~real-time on a modern laptop.
- **Ambient calibration:** on first launch, the loop reads 1 second of mic input and derives a silence threshold from the 75th-percentile RMS × 1.8 (with a floor of 80).
- **Confirmation:** the captured audio is re-listened to; if it passes the energy gate, the transcribed text is shown in the input box for one tick before being sent.
- **TTS:** Piper, interruptible — mid-sentence barge-in cuts off the playback thread and starts the next reply.

To use voice, click the 🎙 button in the input row. To use it hands-free, just say the wake phrase.

---

## Threads, RAG, and file uploads

Ved supports multiple conversation threads, each with its own message history and RAG store.

- **Create:** `/new` or `/new <title>`.
- **List:** `/threads`.
- **Switch:** `/switch <id>` (or click a tab).
- **Rename:** `/rename <id> <title>`.
- **Delete:** `/delete <id>` (refuses if it's the last thread).
- **Clear log:** `/clear` (clears the visible log; messages stay in the thread).

**RAG uploads** are thread-scoped: each thread has its own chunk quota, and the oldest uploads are evicted FIFO when the quota is exceeded. The Tk GUI's drag-and-drop chips map to `POST /files/thread`. Files are persisted, not transient.

Supported extensions include `.txt`, `.md`, `.pdf`, `.docx`, `.doc`, all common source-code extensions, `.html`, `.css`, `.json`, `.yaml`, `.csv`, `.sql`, `.log`, and `.zip`.

---

## Script execution (`/run` and `POST /run`)

`POST /run` accepts a `.py` file upload and executes it via subprocess:

- **Default timeout:** 30 s. Override up to 120 s via `?timeout_seconds=N`.
- **Working directory:** a fresh tempdir, so scripts cannot pollute the project root.
- **Output:** JSON with `exit_code`, `stdout`, `stderr`, `timed_out`, `duration_seconds`, and `truncated_stdout` / `truncated_stderr` flags. Each stream is capped at 16 KiB.
- **CLI args:** pass via `args=foo bar` (whitespace-split).

The same path is reachable from the GUI via the `/run` slash command.

---

## Modes (`standard` / `turbo` / `coder` / `hibernate`)

Each mode reads its own `Modelfile.{mode}` from the project root. To change the model for any mode, edit the `FROM` line and re-pull via Ollama (or update `OPENROUTER_MODEL` for the cloud path).

| Mode | Modelfile | Profile | When to use it |
|---|---|---|---|
| `standard` | `Modelfile.standard` | `qwen2.5:3b-instruct`, 4 K context, CPU-friendly. | Default. Conversations, explanations, planning. |
| `turbo` | `Modelfile.turbo` | Same 3B model, tuned for fast single-shot replies. | Quick Q&A, summaries, routing. |
| `coder` | `Modelfile.coder` | `qwen2.5-coder:7b-instruct-q4_K_M`, 8 K context, GPU-accelerated. | File ops, multi-step coding, refactors. The 7B does the thinking; the 3B does the tool execution. |
| `hibernate` | `Modelfile.hibernate` | (No model loaded.) | Sleeping; cuts VRAM to zero. Wake with `/wake`. |

Switch with the mode chips in the UI, or via `POST /mode`.

---

## Telemetry — active-user tracking

> **Ved records active-user sessions in `data/telemetry.json`.** This is a **local** file on the machine running Ved — nothing is sent over the network. It exists so you can answer the question *"how many people are using my Ved right now?"* without instrumenting an external service.

### What it tracks

- **Active sessions** — each GUI window open, each successful `auth.check_login(...)` call, and each `POST /telemetry/heartbeat` request.
- **Per session:** `session_id`, `username`, `source` (`gui` / `login` / `api`), `mode`, `started_at`, `last_heartbeat`, free-form `meta`.
- **Distinct active users** — multiple sessions for the same username count as one user. (One user with two GUI windows = one active user.)
- **Idle expiry** — sessions whose last heartbeat is older than `VED_TELEMETRY_TIMEOUT` seconds (default 300 s = 5 min) drop off the count automatically.

### Where it hooks in

| Hook | Behavior |
|---|---|
| `__init__.py` (Tk GUI) | Registers a session on window open, ends it on `WM_DELETE_WINDOW`. |
| `_send_command` (Tk GUI) | Refreshes the heartbeat on every prompt the user actually sends. |
| `auth.check_login(...)` | Registers a session for the logged-in user on success. |
| `auth.logout(...)` | Ends the session for a user. |
| `POST /telemetry/heartbeat` | API clients register/refresh their own session. |

### How to query it

**From Python:**

```python
from telemetry import telemetry

telemetry.get_active_count()               # int — number of distinct active users
telemetry.get_active_users()               # list of dicts (username, mode, source, last_heartbeat, ...)
telemetry.snapshot()                       # full debug dump (all sessions, including expired)
```

**From HTTP:**

```bash
curl http://127.0.0.1:8000/telemetry/active
# {
#   "active_count": 2,
#   "active_users": [
#     {"username": "alice", "source": "gui", "mode": "standard", ...},
#     {"username": "bob",   "source": "api", "mode": "coder",   ...}
#   ],
#   "timeout_seconds": 300.0
# }

# API clients register themselves with:
curl -X POST http://127.0.0.1:8000/telemetry/heartbeat \
  -H "Content-Type: application/json" \
  -d '{"username": "mobile-app-user-42"}'
```

**From disk:**

```bash
cat data/telemetry.json   # inspect raw session records
```

### How to disable it

Telemetry is **opt-out** rather than opt-in. The recommended way — no code changes, takes effect on next launch:

```env
# .env
VED_TELEMETRY_DISABLED=true
```

Accepted truthy values: `true`, `1`, `yes` (case-insensitive). When set, every public call on the `telemetry` singleton (`start_session`, `heartbeat`, `end_session`) becomes a no-op. The `/telemetry/active` and `/telemetry/heartbeat` HTTP endpoints still respond but always report `active_count: 0`.

For more granular control, edit the code instead:

1. **Stop the GUI from registering:** comment out the `_telemetry.start_session(...)` call in `ui/gui.py` (search for `VED_USERNAME`).
2. **Stop `auth.check_login` from registering:** pass `record_telemetry=False` to `check_login(...)` everywhere it's called.
3. **Stop the API endpoints:** remove the `/telemetry/*` routes from `api/server.py`.
4. **Disable persistence:** delete `data/telemetry.json` (or set its parent directory read-only).

Or, if you only want a different idle window, set `VED_TELEMETRY_TIMEOUT` in `.env` (e.g. `VED_TELEMETRY_TIMEOUT=900` for 15 minutes).

### Privacy notes

- Telemetry state lives **only** in `data/telemetry.json` on the host running Ved. Nothing is sent to any remote server.
- The `meta` field can carry whatever you put in it (e.g. `pid`, `hostname`). Avoid storing PII unless you intend to.
- Sessions are de-duplicated by username — running two windows as the same user shows one active user, not two.

---

## Slash commands

| Command | What it does |
|---|---|
| `/new` / `/new <title>` | Create a new thread. |
| `/threads` | List all threads. |
| `/switch <id>` | Switch to a thread. |
| `/rename <id> <title>` | Rename a thread. |
| `/delete <id>` | Delete a thread. |
| `/clear` | Clear the visible chat log. |
| `/activate coder` / `/deactivate coder` | Toggle the coder lane. |
| `/mode standard` / `turbo` / `coder` / `hibernate` | Switch mode. |
| `/sleep` / `/hibernate` / `/wake` / `/resume` | Sleep / hibernate / wake. |
| `/upload-global` | Upload a file to the global RAG store. |
| `/run` | Run a `.py` script (opens file picker). |
| `/pin` | Pin the last exchange to long-term memory. |
| `/unpin <n>` | Unpin memory index `n` (1-indexed). |
| `/unpin_all` | Clear all pinned memories. |
| `/list` | List thread attachments. |
| `/memories` | Show pinned memories. |

Start typing `/` in the input box for an autocomplete popup.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

The suite covers the API layer (`test_api_server.py`, `test_api_runner.py`, `test_api_lifecycle.py`), the graph nodes and routing (`test_intent_router.py`, `test_model_router.py`, `test_planner_*.py`, `test_executor_*.py`), the thread + RAG stores (`test_threads.py`, `test_rag_retrieve.py`, `test_per_thread_pinning.py`), the auth + telemetry layer (`test_auth.py`, `test_telemetry.py`), and the voice/audio calibration (`test_wake_hysteresis.py`, `test_ambient_calibration.py`, `test_confirm_keywords.py`).

The telemetry suite (`test_telemetry.py`, 28 tests) covers session lifecycle, expiry, persistence, the auth integration, and the HTTP endpoints.

---

## Troubleshooting

**`ModuleNotFoundError: langchain_ollama`** — Run `pip install -r requirements.txt`. If you're on the cloud path, this import isn't required at runtime, but the package is still imported lazily by the local fallback in `model_adapter.py`.

**`openwakeword` import error on Python 3.12+** — Install `tensorflow>=2.16` (already in `requirements.txt`). The standalone `tflite-runtime` package only ships wheels up to 3.11; `tensorflow` ships `tf.lite.Interpreter` which `openwakeword` detects automatically.

**Ved can't reach Ollama** — Verify Ollama is running (`ollama serve`) and that the models in your `Modelfile.{mode}` files are pulled (`ollama list`). Or switch to the cloud path with `USE_CLOUD_API=true`.

**Voice doesn't hear the wake word** — Run the ambient calibration: the first launch should auto-calibrate. If it didn't, delete `data/telemetry.json` and `voice/__pycache__/` and relaunch. You can also lower `vad_threshold` (default `0.25`) in `voice/voice_module.py`.

**`/run` times out** — Bump `timeout_seconds` (max 120). Long-running scripts should be broken into smaller pieces or run outside Ved.

**Telemetry shows 0 users but people are chatting** — Their heartbeats may have expired. Lower `VED_TELEMETRY_TIMEOUT` in `.env`, or check `data/telemetry.json` to see raw session records (including expired ones).

---

## License

See repository root for license details.
