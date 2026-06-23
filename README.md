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