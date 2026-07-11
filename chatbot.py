import json
import os
import secrets
import time
from pathlib import Path
import requests
from __init__ import DEFAULT_MODE, MODES
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from graph import build_graph
from model_adapter import ModelAdapter, parse_modelfile
from command_processor import ChatbotCommandProcessor
from graph.tools.staging_registry import STAGING_REGISTRY
import queue
import threading

THREADS_DB_PATH = "data/threads.json"
THREAD_MESSAGE_CAP = 40  # 1 system prompt + up to 39 other messages; matches graph/state.py limit_messages.

# Appended to every Modelfile system prompt at runtime. Acts as a hallucination
# guard: when no RAG or web context is available, the model is told to admit it
# rather than fabricate. Once a web-search tool is added in a future pass, this
# instruction can be softened to "use web search instead of fabricating".
HALLUCINATION_GUARD = (
    "\n\nIMPORTANT: If you cannot answer from your training data or any "
    "provided context, respond with 'I don't know' rather than fabricating "
    "information."
)

def _trim_thread_messages(messages: list) -> list:
    """Keep at most THREAD_MESSAGE_CAP messages: first SystemMessage (if any) + most recent (CAP-1) others."""
    if len(messages) <= THREAD_MESSAGE_CAP:
        return messages
    system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    others = [m for m in messages if not isinstance(m, SystemMessage)]
    if system is not None:
        return [system] + others[-(THREAD_MESSAGE_CAP - 1):]
    return others[-THREAD_MESSAGE_CAP:]

def _save_ai_response_to_thread_rag(thread_id: str, content: str, source_label: str) -> bool:
    """Inject a long AI response into the thread's RAG store with FIFO eviction.

    Best-effort: if the embedding pipeline or RAG store is unavailable,
    this returns False silently. The summary remains in history either way,
    so the conversation history is never blocked on RAG availability.

    Returns True on success, False if RAG was unavailable.
    """
    if not content or not thread_id:
        return False
    try:
        # Lazy import to avoid loading the embedding model at import time.
        from data.thread_files import ThreadFileStore
        from graph.rag import rag_db as default_rag_db
        store = ThreadFileStore(default_rag_db)
        store.add_text(thread_id, content, source_label)
        return True
    except Exception as e:
        # Embedding pipeline may be unavailable (no Ollama, no model loaded).
        # Log but don't raise — persistence must never break the chat loop.
        print(f"[RAG Save] Skipped for thread {thread_id[:8]}: {e}", flush=True)
        return False



# history. The full text is injected into the thread's RAG store with FIFO
# eviction; only the head + tail summary stays in the message list. This
# keeps thread history compact while preserving the full content for RAG
# retrieval.
_AI_SUMMARY_THRESHOLD_CHARS = 1200  # ~300 tokens
_AI_SUMMARY_HEAD_WORDS = 30
_AI_SUMMARY_TAIL_WORDS = 30

# Tool messages persisted to history get truncated to this many chars.
# The LLM usually only needs the first chunk to know what the tool returned.
# Use retrieve_rag to recover the full output if needed.
_TOOL_HISTORY_TRUNCATE_CHARS = 800  # ~200 tokens


def _compress_ai_content(content: str) -> str:
    """Compress long AI content to head + tail summary for history persistence.

    Short content is returned unchanged. Long content is reduced to the
    first 30 words, a marker, and the last 30 words. The full content is
    intended to be saved separately into RAG.
    """
    if not content:
        return content
    if len(content) <= _AI_SUMMARY_THRESHOLD_CHARS:
        return content
    words = content.split()
    if len(words) <= _AI_SUMMARY_HEAD_WORDS + _AI_SUMMARY_TAIL_WORDS + 4:
        return content
    head = " ".join(words[:_AI_SUMMARY_HEAD_WORDS])
    tail = " ".join(words[-_AI_SUMMARY_TAIL_WORDS:])
    return f"{head}\n\n... [full response stored in thread RAG; retrieval key: see message metadata] ...\n\n{tail}"


def _serialize_message(msg) -> dict:
    cls_name = type(msg).__name__
    if cls_name == "HumanMessage":
        role = "human"
    elif cls_name == "AIMessage":
        role = "ai"
    elif cls_name == "SystemMessage":
        role = "system"
    elif cls_name == "ToolMessage":
        role = "tool"
    else:
        role = cls_name.lower()
    content = msg.content
    # Compact long tool outputs in history. The LLM usually only needs the
    # first chunk to know what the tool returned; full output can be
    # recovered later via retrieve_rag if needed.
    if cls_name == "ToolMessage" and isinstance(content, str) and len(content) > _TOOL_HISTORY_TRUNCATE_CHARS:
        content = (
            content[:_TOOL_HISTORY_TRUNCATE_CHARS]
            + f"\n\n...[truncated; full output recoverable via retrieve_rag]"
        )
    out = {"role": role, "content": content}
    # ToolMessage carries tool_call_id which the LLM uses to associate the
    # result with the originating tool_call. Preserve it across save/load.
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    # Preserve additional_kwargs (e.g., "pinned": True so FIFO doesn't drop pinned turns).
    extra = getattr(msg, "additional_kwargs", None)
    if extra:
        out["additional_kwargs"] = dict(extra)
    return out


def _deserialize_message(data: dict) -> BaseMessage:
    role = data.get("role", "")
    content = data.get("content", "")
    extra = data.get("additional_kwargs") or {}
    tool_call_id = data.get("tool_call_id")
    if role == "human":
        msg = HumanMessage(content=content)
    elif role == "ai":
        msg = AIMessage(content=content)
    elif role == "system":
        msg = SystemMessage(content=content)
    elif role == "tool":
        # ToolMessage requires tool_call_id; use a placeholder if missing.
        # Older saves (pre-tool-persistence) may not have stored one.
        msg = ToolMessage(content=content, tool_call_id=tool_call_id or "legacy_unpaired")
    else:
        msg = HumanMessage(content=content)
    if extra:
        # Restore pinned flag (and any future additional_kwargs) on reload.
        try:
            msg.additional_kwargs.update(extra)
        except Exception:
            pass
    return msg

class Chatbot(ChatbotCommandProcessor):
    def __init__(self, mode=None):
        self.mode = mode or DEFAULT_MODE
        self._hibernating = (self.mode == "hibernate")
        self.project_root = Path(__file__).resolve().parent
        self.memory_db_path = self.project_root / "data" / "memories.json"
        self.threads_db_path = self.project_root / "data" / "threads.json"
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_memories = self._load_persistent_memories()
        self.adapters = {}
        for m in MODES:
            self.adapters[m] = None if m == "hibernate" else self._load_adapter_for_mode(m)
        self._llm_cache = {}
        self._threads = {}
        self._active_thread_id = None
        # UI components reference — set by gui.py after the UI is built.
        # Used by command_processor._handle_cd() to push cwd updates to the
        # title-bar chip, and by on_session_start() to push index status.
        self._ui_components = None
        self._load_threads()

        # Per-thread file quota tracker. Lazy-imported to avoid loading
        # the Ollama embeddings engine on every Chatbot instantiation.
        from data.thread_files import ThreadFileStore
        from data.global_files import GlobalFileStore
        from graph.rag import rag_db
        self._thread_files = ThreadFileStore(rag_db)
        self._global_files = GlobalFileStore(rag_db)
        if not self._threads:
            self._create_starter_thread()
        self._graph = build_graph(self._get_llm)
        print(f"[DEBUG] graph after init: {self._graph}")

    def _create_starter_thread(self):
        tid = f"thr_{secrets.token_hex(4)}"
        self._threads[tid] = {
            "id": tid,
            "title": "New Thread",
            "created_at": time.time(),
            "messages": [],
        }
        self._active_thread_id = tid
        self._save_threads()

    def _load_threads(self) -> None:
        path = self.threads_db_path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict):
            entries = list(raw.values())
        else:
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tid = entry.get("id")
            if not tid or not isinstance(tid, str):
                continue
            msgs_raw = entry.get("messages", [])
            if not isinstance(msgs_raw, list):
                msgs_raw = []
            messages = [_deserialize_message(m) for m in msgs_raw if isinstance(m, dict)]
            messages = _trim_thread_messages(messages)
            self._threads[tid] = {
                "id": tid,
                "title": entry.get("title", "New Thread"),
                "created_at": entry.get("created_at", time.time()),
                "messages": messages,
            }
        if self._threads and (self._active_thread_id is None or self._active_thread_id not in self._threads):
            self._active_thread_id = next(iter(sorted(self._threads.keys(), key=lambda k: self._threads[k]["created_at"])))

    def _save_threads(self) -> None:
        path = self.threads_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        # Iterate threads in REVERSE chronological order so the JSON file
        # has newest threads first. Dict iteration order is preserved in
        # Python 3.7+, so this puts the most recent thread at the top of
        # the file for easy human inspection and predictable load order.
        for tid, thread in sorted(
            self._threads.items(),
            key=lambda kv: kv[1].get("created_at", 0.0),
            reverse=True,
        ):
            thread["messages"] = _trim_thread_messages(thread["messages"])
            payload[tid] = {
                "id": thread["id"],
                "title": thread["title"],
                "created_at": thread["created_at"],
                "messages": [_serialize_message(m) for m in thread["messages"]],
            }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_threads(self) -> list:
        """Return threads sorted newest-first by created_at."""
        return sorted(
            ({"id": t["id"], "title": t["title"], "created_at": t["created_at"]} for t in self._threads.values()),
            key=lambda d: d["created_at"],
            reverse=True,
        )

    def create_thread(self, title: str | None = None) -> str:
        tid = f"thr_{secrets.token_hex(4)}"
        self._threads[tid] = {
            "id": tid,
            "title": title if title else "New Thread",
            "created_at": time.time(),
            "messages": [],
        }
        self._active_thread_id = tid
        self._save_threads()
        return tid

    def switch_thread(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        self._active_thread_id = thread_id
        return True

    def rename_thread(self, thread_id: str, title: str) -> bool:
        if thread_id not in self._threads:
            return False
        self._threads[thread_id]["title"] = title
        self._save_threads()
        return True

    def delete_thread(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        if len(self._threads) <= 1:
            return False
        was_active = (self._active_thread_id == thread_id)
        del self._threads[thread_id]
        # Drop the thread's chunks and metadata from the vector DB + SSD.
        if hasattr(self, "_thread_files") and self._thread_files is not None:
            try:
                self._thread_files.clear_thread(thread_id)
            except Exception:
                pass
        if was_active:
            if self._threads:
                oldest = min(self._threads.values(), key=lambda t: t["created_at"])
                self._active_thread_id = oldest["id"]
            else:
                self._create_starter_thread()
                return True
        self._save_threads()
        return True

    def get_active_thread(self) -> dict:
        if self._active_thread_id is None or self._active_thread_id not in self._threads:
            if not self._threads:
                self._create_starter_thread()
            else:
                self._active_thread_id = next(iter(self._threads))
        return self._threads[self._active_thread_id]

    def _autotitle_from_message(self, text: str) -> str:
        stripped = (text or "").strip()
        if len(stripped) <= 40:
            return stripped
        return stripped[:40]

    @property
    def _conversation_history(self):
        return self.get_active_thread()["messages"]

    @_conversation_history.setter
    def _conversation_history(self, value):
        if self._active_thread_id and self._active_thread_id in self._threads:
            self._threads[self._active_thread_id]["messages"] = value
        else:
            self.get_active_thread()["messages"] = value

    def _load_persistent_memories(self) -> list:
        if self.memory_db_path.exists():
            try: return json.loads(self.memory_db_path.read_text(encoding="utf-8"))
            except Exception: return []
        return []

    def _load_adapter_for_mode(self, mode: str) -> ModelAdapter:
        info = parse_modelfile(self.project_root / f"Modelfile.{mode}")
        model_name = info.get("from") or f"{mode}-stub"
        params = info.get("params", {})
        device = "gpu" if mode in ["turbo", "coder"] else "cpu"
        if "num_gpu" in params:
            try: device = "gpu" if int(params["num_gpu"]) > 0 else "cpu"
            except Exception: pass
        return ModelAdapter(model_name=model_name, device=device, params=params, system_prompt=info.get("system", ""))

    def _get_llm(self):
        if self.mode == "hibernate": return None
        if self.mode in self._llm_cache: return self._llm_cache[self.mode]
        adapter = self.adapters.get(self.mode)
        if adapter is None: return None
        llm = adapter.create_llm(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self._llm_cache[self.mode] = llm
        return llm

    def _flush_all_models(self) -> None:
        """Send `keep_alive: 0` to Ollama for every adapter + embeddings.

        Called when entering hibernate mode so cached models are evicted
        from Ollama's RAM/VRAM. Best-effort: any single failure is
        swallowed so one unreachable model doesn't block the rest.
        """
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        models_to_flush: set[str] = set()

        for adapter in (self.adapters or {}).values():
            if adapter is None:
                continue
            model_name = getattr(adapter, "model_name", None)
            if model_name:
                models_to_flush.add(model_name)

        # The RAG embedding model is loaded independently of the chat
        # adapters. Flush it too so hibernate truly frees all VRAM.
        models_to_flush.add("nomic-embed-text:latest")

        for model_name in models_to_flush:
            try:
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=5,
                )
            except Exception:
                pass

    def _make_planner_llm_factory(self):
        """Return a closure `f(mode) -> ChatOllama` that the planner node
        will call with `state.mode` to get the right model. Re-resolved
        on every call so a `/set_mode` switch (which calls
        `_rebuild_graph`) automatically picks up the new mode's adapter.
        """
        from model_adapter import get_planner_llm

        def _factory(mode: str):
            adapter = self.adapters.get(mode)
            if adapter is None:
                return None
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return get_planner_llm(
                mode,
                base_url=base_url,
                device=adapter.device,
                params=adapter.params or {},
            )

        return _factory

    def _make_executor_llm_factory(self):
        """Return a closure `f(mode) -> ChatOllama` that the executor node
        will call with `state.mode` to get the right model. Re-resolved
        on every call so a `/set_mode` switch automatically picks up the
        new mode's adapter.
        """
        from model_adapter import get_executor_llm

        def _factory(mode: str):
            adapter = self.adapters.get(mode)
            if adapter is None:
                return None
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return get_executor_llm(
                mode,
                base_url=base_url,
                device=adapter.device,
                params=adapter.params or {},
            )

        return _factory

    def set_mode(self, mode: str):
        if mode not in MODES: raise ValueError(f"Unknown mode: {mode}")
        if self.mode == "coder" and mode in ["standard", "turbo"]:
            raise RuntimeError("Hardware Interlock Triggered: Standard mode switching is blocked while coder mode is active.")
        if mode == self.mode: return
        old_mode = self.mode
        self.mode = mode
        self._hibernating = (mode == "hibernate")
        self._llm_cache.clear()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if mode == "hibernate":
            # Entering hibernate: flush every cached model from Ollama.
            # The existing per-transition flush below is skipped because
            # _flush_all_models already covers the previous mode.
            self._flush_all_models()
        else:
            # If we are waking up from hibernate, reset the RAG embedding
            # engine so the next upload/query starts a fresh Ollama request.
            if old_mode == "hibernate":
                from graph.rag import rag_db
                rag_db.reset_embeddings_engine()
            # Normal mode switch: flush only the previous mode's model,
            # then keep the new model alive and rebuild the graph.
            prev_adapter = self.adapters.get(old_mode)
            if prev_adapter and prev_adapter.model_name:
                try: requests.post(f"{base_url}/api/generate", json={"model": prev_adapter.model_name, "keep_alive": 0}, timeout=5)
                except Exception: pass
            self._graph = build_graph(self._get_llm)
            active_adapter = self.adapters.get(mode)
            if active_adapter and active_adapter.model_name:
                try: requests.post(f"{base_url}/api/generate", json={"model": active_adapter.model_name, "prompt": "", "keep_alive": "20m"}, timeout=15)
                except Exception: pass

    def save_user_input_to_thread_rag(self, prompt: str, source_label: str) -> bool:
        """Persist a long user paste into the active thread's RAG store.

        Used by the web UI when /chat receives a prompt above the threshold.
        The full text is chunked + embedded + stored with FIFO eviction;
        a short reference is sent to the LLM and the full version is
        recoverable via retrieve_rag(query).
        """
        thread = self.get_active_thread()
        thread_id = thread.get("id") if thread else None
        if not thread_id:
            return False
        return _save_ai_response_to_thread_rag(thread_id, prompt, source_label)

    def _rebuild_graph(self) -> None:
        """Rebuild the graph for the current mode. Called by nodes after a
        cross-mode tool-creation trigger so the next LLM invocation uses
        the newly-loaded model's bound tools."""
        if self.mode == "hibernate":
            return
        from graph import build_graph
        self._graph = build_graph(self._get_llm)

    def set_ui_components(self, ui_components) -> None:
        """Wire the UI components reference and seed initial status chips.

        Called by gui.py after the UI is built. Stores the reference so
        command_processor._handle_cd() can push cwd updates to the title-
        bar chip, and seeds both chips (cwd + index status) with their
        initial values so the UI doesn't look stale before the first chat.
        """
        self._ui_components = ui_components
        # Seed cwd chip with current working directory.
        try:
            if hasattr(ui_components, "set_current_directory"):
                ui_components.set_current_directory(os.getcwd())
        except Exception:
            pass
        # Seed index status chip with Idle — on_session_start will update it
        # when the first chat triggers indexing.
        try:
            if hasattr(ui_components, "set_index_status"):
                ui_components.set_index_status("Idle", "#6c7086")
        except Exception:
            pass

    def on_session_start(self) -> None:
        """Kick off project RAG indexing in a background thread.

        Called once per session (on first chat, or from app entrypoint).
        Returns immediately. Subsequent calls are no-ops (guarded by
        `_project_index_started`). The indexing runs in a daemon thread
        so it doesn't block the UI or the chat loop.

        Indexing is incremental: unchanged files (by SHA-256) are skipped
        via the hash index persisted at data/rag_index.json.
        """
        if getattr(self, "_project_index_started", False):
            return
        self._project_index_started = True

        def _run():
            try:
                from graph.rag.vector_engine import LocalVectorDB
                from graph.rag.project_indexer import index_workspace
                db = LocalVectorDB()
                # Warm up the embedding model so the first RAG query doesn't
                # pay the model-load cost (Ollama auto-loads on first use,
                # which can take 5-10s on cold start). Best-effort.
                try:
                    db.embeddings_engine.embed_query("warmup")
                except Exception as warmup_exc:
                    if os.getenv("VED_DEBUG"):
                        print(f"[on_session_start] Embedding warmup skipped: {warmup_exc}", flush=True)
                root = os.getcwd()
                stats = index_workspace(root, db)
                print(f"[on_session_start] Indexed project: {stats}", flush=True)
            except Exception as e:
                print(f"[on_session_start] Indexing failed: {e}", flush=True)

        t = threading.Thread(target=_run, daemon=True, name="project-indexer")
        t.start()

    def submit_tool_creation_approval(self, session_id: str, approved: bool) -> bool:
        """Resolve a pending tool-creation proposal. Returns True if a
        matching session was found, False otherwise."""
        state = getattr(self, "_tool_creation_state", None)
        event = getattr(self, "_tool_creation_event", None)
        if state is None or event is None:
            return False
        if state.get("session_id") != session_id:
            return False
        state["value"] = bool(approved)
        event.set()
        return True

    def respond(self, message: str):
        cmd_resp = self.handle_command(message)
        if cmd_resp is not None:
            if isinstance(cmd_resp, str):
                active = self.get_active_thread()
                active["messages"].append(AIMessage(content=cmd_resp))
                self._save_threads()
            return cmd_resp
        if self._hibernating:
            return "(hibernate) Bot is currently hibernating. Use /wake to wake."
        adapter = self.adapters.get(self.mode)
        if adapter is None:
            return "No model available for current mode."
        def _stream_generator():
            print("[DEBUG] stream generator started", flush=True)
            active = self.get_active_thread()
            was_empty = len(active["messages"]) == 0 and active["title"] == "New Thread"
            history = active["messages"]
            initial_messages = list(history) + [HumanMessage(content=message)]
            if was_empty:
                active["title"] = self._autotitle_from_message(message)
            active = self.get_active_thread()
            thread_id = active.get("id", "")
            input_state = {
                "messages": initial_messages,
                "route_intent": "",
                "mode": self.mode,
                "saved_memories": getattr(self, "saved_memories", []),
                "current_draft": "",
                "critique_notes": "",
                "content_score": 0,
                "loop_count": 0,
                "active_thread_id": thread_id,
            }
            token_queue = queue.Queue()
            self._human_approval_event = threading.Event()
            self._human_approval_state = {"value": None}
            self._tool_creation_event = threading.Event()
            self._tool_creation_state = {"value": None, "session_id": None}
            self._plan_approval_event = threading.Event()
            self._plan_approval_state = {"value": None}
            self._file_edit_approval_event = threading.Event()
            self._file_edit_approval_state = {"value": None}
            self._file_edit_pending_tasks = {}
            self._file_edit_pending_lock = threading.Lock()
            self._file_edit_worker_stop = threading.Event()
            self._file_edit_thread_id = thread_id
            STAGING_REGISTRY.register_session(
                thread_id,
                approval_event=self._file_edit_approval_event,
                approval_state=self._file_edit_approval_state,
            )
            config = {"configurable": {"system_prompt": adapter.system_prompt + HALLUCINATION_GUARD, "token_queue": token_queue, "approval_event": self._human_approval_event, "approval_state": self._human_approval_state, "plan_approval_event": self._plan_approval_event, "plan_approval_state": self._plan_approval_state, "tool_creation_event": self._tool_creation_event, "tool_creation_state": self._tool_creation_state, "file_edit_approval_event": self._file_edit_approval_event, "file_edit_approval_state": self._file_edit_approval_state, "file_edit_pending_tasks": self._file_edit_pending_tasks, "file_edit_pending_lock": self._file_edit_pending_lock, "tool_approved": True, "active_thread_id": self._active_thread_id, "session_id": "", "set_mode": self.set_mode, "rebuild_graph": self._rebuild_graph, "planner_llm_factory": self._make_planner_llm_factory(), "executor_llm_factory": self._make_executor_llm_factory()}}
            worker_thread = threading.Thread(
                target=self._file_edit_approval_worker,
                daemon=True,
                name="file-edit-approval-worker",
            )
            worker_thread.start()
            last_node_seen = "Unknown"
            accumulated_state = dict(input_state)
            def run_graph():
                nonlocal last_node_seen, accumulated_state
                try:
                    for chunk in self._graph.stream(input_state, config=config, stream_mode="updates"):
                        for node_name, node_output in chunk.items():
                            last_node_seen = node_name
                            for key, val in node_output.items():
                                accumulated_state[key] = val
                except Exception as exc:
                    token_queue.put(("error", str(exc)))
                finally:
                    token_queue.put(None)
            threading.Thread(target=run_graph, daemon=True).start()
            while True:
                item = token_queue.get()
                if item is None:
                    break
                if isinstance(item, tuple):
                    event_type = item[0]
                    payload = item[1] if len(item) > 1 else None
                    if event_type == "error":
                        yield ("error", payload)
                    else:
                        yield (event_type, payload)
                else:
                    yield ("token", item)
            if accumulated_state and "messages" in accumulated_state:
                final_msgs = accumulated_state["messages"]
                active = self.get_active_thread()
                if len(final_msgs) > len(initial_messages):
                    active["messages"] = list(final_msgs)
                else:
                    new_ai = [m for m in final_msgs if isinstance(m, AIMessage) and m not in initial_messages]
                    if new_ai:
                        active["messages"] = list(initial_messages) + new_ai
                    else:
                        active["messages"] = list(final_msgs)
                active_thread_id = active.get("id")
                def _compress_and_save(messages_snapshot, thread_id):
                    for msg in messages_snapshot:
                        if isinstance(msg, AIMessage):
                            full_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if len(full_content) > _AI_SUMMARY_THRESHOLD_CHARS and thread_id:
                                source_label = f"ai_response_{int(time.time())}_{secrets.token_hex(3)}"
                                _save_ai_response_to_thread_rag(
                                    thread_id, full_content, source_label
                                )
                                msg.content = _compress_ai_content(full_content)
                    self._save_threads()
                threading.Thread(
                    target=_compress_and_save,
                    args=(list(active["messages"]), active_thread_id),
                    daemon=True,
                ).start()
            if "saved_memories" in accumulated_state:
                self.saved_memories = accumulated_state["saved_memories"]
            ollama_active = ["None"]
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                r = requests.get(f"{base_url}/api/ps", timeout=2)
                if r.status_code == 200:
                    ollama_active = [m.get("name") for m in r.json().get("models", [])]
            except Exception:
                ollama_active = ["Error"]
            # Hardware debug header — prints to stdout (terminal). Safe in
            # this codebase because tkinter doesn't capture stdout.
            print(f"\n==== [VED HARDWARE DEBUG] ====\n  -> Request Mode: {self.mode.upper()}\n  -> Route completed: Node {last_node_seen}\n  -> RAM Active Models: {', '.join(ollama_active)}\n  -> Context size: {len(self._conversation_history)}\n==============================\n", flush=True)
            self._file_edit_worker_stop.set()
            worker_thread.join(timeout=2.0)
            STAGING_REGISTRY.unregister_session(thread_id)
        return _stream_generator()

    def _apply_file_edit_task(self, task: dict):
        """Apply a single approved file-edit task by calling the underlying
        filesystem action directly.

        Returns the action's result string. Errors raised inside the
        action are caught and returned as an "ERROR: ..." string so the
        worker thread never crashes.
        """
        from graph.actions.filesystem import edit_file_action, overwrite_file_action
        from pathlib import Path as _Path
        tool_name = task.get("tool_name")
        args = task.get("args") or {}
        # _resolve_and_check uses PROJECT_ROOT as the sole allowed root
        # in non-self-healing mode. Mirror the same boundary here so a
        # post-approval apply cannot escape the project tree.
        project_root = str(_Path(__file__).resolve().parent)
        try:
            if tool_name == "edit_file":
                return edit_file_action(
                    args.get("path", ""),
                    args.get("old_text", ""),
                    args.get("new_text", ""),
                    allowed_roots=(project_root,),
                    backup_dir=None,
                )
            elif tool_name == "overwrite_file":
                return overwrite_file_action(
                    args.get("path", ""),
                    args.get("content", ""),
                    allowed_roots=(project_root,),
                    backup_dir=None,
                )
            return f"ERROR: unknown file-edit tool '{tool_name}'"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def _file_edit_approval_worker(self):
        """Daemon worker: waits for file-edit approval events and applies
        the approved subset of pending tasks.

        Loop semantics:
          - Exits when `self._file_edit_worker_stop` is set.
          - Otherwise blocks on `self._file_edit_approval_event`.
          - On wake, reads the decision dict from
            `self._file_edit_approval_state["value"]` and applies it.
          - Uses the staging registry when a session is registered; falls
            back to the legacy in-memory dict for callers that bypass
            `respond()` (e.g. unit tests).
          - Then resets state and clears the event so the next decision
            can be processed.
        """
        event = self._file_edit_approval_event
        stop = self._file_edit_worker_stop
        state = self._file_edit_approval_state
        thread_id = getattr(self, "_file_edit_thread_id", "")
        while not stop.is_set():
            event.wait(timeout=0.2)
            if stop.is_set():
                break
            if not event.is_set():
                continue
            decision = None
            try:
                decision = state.get("value") if state else None
            except Exception:
                decision = None
            if thread_id and STAGING_REGISTRY.has_session(thread_id):
                STAGING_REGISTRY.apply_decision(
                    thread_id,
                    decision or {},
                    apply_callback=self._apply_file_edit_task,
                )
                with self._file_edit_pending_lock:
                    self._file_edit_pending_tasks.clear()
                    self._file_edit_pending_tasks.update(
                        STAGING_REGISTRY.get_tasks(thread_id)
                    )
            else:
                approved: list = []
                with self._file_edit_pending_lock:
                    snapshot = dict(self._file_edit_pending_tasks)
                    action = (decision or {}).get("action", "reject")
                    paths = (decision or {}).get("paths") or []
                    if action == "approve_all":
                        approved = list(snapshot.values())
                        self._file_edit_pending_tasks.clear()
                    elif action == "approve":
                        for p in paths:
                            t = snapshot.get(p)
                            if t is not None:
                                approved.append(t)
                                self._file_edit_pending_tasks.pop(p, None)
                    elif action == "reject_all":
                        self._file_edit_pending_tasks.clear()
                    else:
                        for p in paths:
                            self._file_edit_pending_tasks.pop(p, None)
                for task in approved:
                    self._apply_file_edit_task(task)
            try:
                state["value"] = None
            except Exception:
                pass
            event.clear()
    
    def submit_human_approval(self, approved: bool) -> None:
        """Unblocks the content pipeline after it emits an approval_request event.
        Safe to call when no approval is pending (no-op)."""
        state = getattr(self, "_human_approval_state", None)
        if state is not None:
            state["value"] = bool(approved)
        event = getattr(self, "_human_approval_event", None)
        if event is not None:
            event.set()

    def submit_plan_approval(self, approved: bool) -> None:
        """Unblocks the planner after it emits a plan_approval_request event.

        Called by the UI when the user accepts or rejects the proposed plan.
        Safe to call when no approval is pending (no-op).
        """
        state = getattr(self, "_plan_approval_state", None)
        if state is not None:
            state["value"] = bool(approved)
        event = getattr(self, "_plan_approval_event", None)
        if event is not None:
            event.set()

    def submit_file_edit_approval(self, decision) -> None:
        """Unblocks the executor after it emits a file_edit_approval_request event.

        Called by the UI when the user accepts or rejects a proposed file
        edit (edit_file / overwrite_file). Safe to call when no approval
        is pending (no-op).

        `decision` may be:
          - a bool (backward-compat): True -> approve_all, False -> reject_all.
          - a dict {"action": "approve_all"|"reject_all"|"approve"|"reject",
                    "paths": [..]} for per-file control.
        """
        if isinstance(decision, bool):
            decision = {"action": "approve_all" if decision else "reject_all", "paths": []}
        elif not isinstance(decision, dict):
            decision = {"action": "reject_all", "paths": []}
        state = getattr(self, "_file_edit_approval_state", None)
        if state is not None:
            state["value"] = decision
        event = getattr(self, "_file_edit_approval_event", None)
        if event is not None:
            event.set()

    def add_global_file(self, source_path: str) -> dict:
        """Add a file to the global store (accessible only via /upload-global).

        The chunker is selected from the active mode: "ast" in coder mode,
        "text" elsewhere.
        """
        import os as _os
        return self._global_files.add(source_path, filename=_os.path.basename(source_path), chunker=self._rag_chunker())

    def _rag_chunker(self) -> str:
        """Return the chunker name to use for RAG ingests based on the active mode."""
        return "ast" if self.mode == "coder" else "text"

    def list_global_files(self) -> list:
        return self._global_files.list_uploads()

    def get_pinned_messages_in_active_thread(self) -> list:
        """Return the pinned messages in the current thread, oldest first."""
        thread = self.get_active_thread()
        out = []
        for m in thread.get("messages", []):
            if getattr(m, "additional_kwargs", {}).get("pinned", False):
                out.append(m)
        return out

    def pin_last_turn_in_active_thread(self) -> int:
        """Mark the last AI message (and its preceding Human) as pinned.

        Returns the number of messages newly pinned. 0 if nothing to pin.
        """
        thread = self.get_active_thread()
        msgs = thread.get("messages", [])
        # Find the last AIMessage in the thread.
        last_ai_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], AIMessage):
                last_ai_idx = i
                break
        if last_ai_idx < 0:
            return 0
        pinned_count = 0
        msgs[last_ai_idx].additional_kwargs["pinned"] = True
        pinned_count += 1
        # Also pin the immediately preceding HumanMessage if present.
        if last_ai_idx > 0 and isinstance(msgs[last_ai_idx - 1], HumanMessage):
            msgs[last_ai_idx - 1].additional_kwargs["pinned"] = True
            pinned_count += 1
        self._save_threads()
        return pinned_count

    def unpin_in_active_thread(self, index_1based: int) -> int:
        """Unpin the Nth pinned message in the current thread (1-based)."""
        thread = self.get_active_thread()
        pinned = self.get_pinned_messages_in_active_thread()
        if index_1based < 1 or index_1based > len(pinned):
            return 0
        target = pinned[index_1based - 1]
        target.additional_kwargs["pinned"] = False
        self._save_threads()
        return 1

    def unpin_all_in_active_thread(self) -> int:
        """Clear the pinned flag on all messages in the current thread."""
        thread = self.get_active_thread()
        cleared = 0
        for m in thread.get("messages", []):
            if getattr(m, "additional_kwargs", {}).get("pinned", False):
                m.additional_kwargs["pinned"] = False
                cleared += 1
        if cleared:
            self._save_threads()
        return cleared