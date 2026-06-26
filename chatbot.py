import json
import os
import secrets
import time
from pathlib import Path
import requests
from __init__ import DEFAULT_MODE, MODES
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from graph import build_graph
from model_adapter import ModelAdapter, parse_modelfile
from command_processor import ChatbotCommandProcessor
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

def _serialize_message(msg) -> dict:
    cls_name = type(msg).__name__
    if cls_name == "HumanMessage":
        role = "human"
    elif cls_name == "AIMessage":
        role = "ai"
    elif cls_name == "SystemMessage":
        role = "system"
    else:
        role = cls_name.lower()
    return {"role": role, "content": msg.content}

def _deserialize_message(data: dict) -> BaseMessage:
    role = data.get("role", "")
    content = data.get("content", "")
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)

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
        for tid, thread in self._threads.items():
            thread["messages"] = _trim_thread_messages(thread["messages"])
            payload[tid] = {
                "id": thread["id"],
                "title": thread["title"],
                "created_at": thread["created_at"],
                "messages": [_serialize_message(m) for m in thread["messages"]],
            }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_threads(self) -> list:
        return sorted(
            ({"id": t["id"], "title": t["title"], "created_at": t["created_at"]} for t in self._threads.values()),
            key=lambda d: d["created_at"],
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
        prev_adapter = self.adapters.get(old_mode)
        if prev_adapter and prev_adapter.model_name:
            try: requests.post(f"{base_url}/api/generate", json={"model": prev_adapter.model_name, "keep_alive": 0}, timeout=5)
            except Exception: pass
        if mode != "hibernate":
            self._graph = build_graph(self._get_llm)
            active_adapter = self.adapters.get(mode)
            if active_adapter and active_adapter.model_name:
                try: requests.post(f"{base_url}/api/generate", json={"model": active_adapter.model_name, "prompt": "", "keep_alive": "20m"}, timeout=15)
                except Exception: pass

    def respond(self, message: str):
        cmd_resp = self.handle_command(message)
        if cmd_resp is not None:
            return cmd_resp
        if self._hibernating:
            return "(hibernate) Bot is currently hibernating. Use /wake to wake."
        adapter = self.adapters.get(self.mode)
        if adapter is None:
            return "No model available for current mode."
        def _stream_generator():
            print("[DEBUG] stream generator started")
            active = self.get_active_thread()
            was_empty = len(active["messages"]) == 0 and active["title"] == "New Thread"
            pinned_memories = self._load_pinned_contents()
            history = active["messages"]
            if not history:
                initial_messages = [SystemMessage(content=adapter.system_prompt)]
                for pair in pinned_memories:
                    if isinstance(pair, dict):
                        initial_messages.append(HumanMessage(content=pair.get("user", "")))
                        initial_messages.append(AIMessage(content=pair.get("assistant", "")))
                initial_messages.append(HumanMessage(content=message))
            else:
                initial_messages = list(history) + [HumanMessage(content=message)]
            if was_empty:
                active["title"] = self._autotitle_from_message(message)
            input_state = {
                "messages": initial_messages,
                "route_intent": "",
                "mode": self.mode,
                "saved_memories": getattr(self, "saved_memories", []),
                "current_draft": "",
                "critique_notes": "",
                "content_score": 0,
                "loop_count": 0
            }
            token_queue = queue.Queue()
            self._human_approval_event = threading.Event()
            self._human_approval_state = {"value": None}
            config = {"configurable": {"system_prompt": adapter.system_prompt + HALLUCINATION_GUARD, "token_queue": token_queue, "approval_event": self._human_approval_event, "approval_state": self._human_approval_state, "tool_approved": True, "active_thread_id": self._active_thread_id}}
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
                self._save_threads()
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
            print(f"\n==== [VED HARDWARE DEBUG] ====\n  -> Request Mode: {self.mode.upper()}\n  -> Route completed: Node {last_node_seen}\n  -> RAM Active Models: {', '.join(ollama_active)}\n  -> Context size: {len(self._conversation_history)}\n==============================\n")
        return _stream_generator()
    
    def submit_human_approval(self, approved: bool) -> None:
        """Unblocks the content pipeline after it emits an approval_request event.
        Safe to call when no approval is pending (no-op)."""
        state = getattr(self, "_human_approval_state", None)
        if state is not None:
            state["value"] = bool(approved)
        event = getattr(self, "_human_approval_event", None)
        if event is not None:
            event.set()

    def add_global_file(self, source_path: str) -> dict:
        """Add a file to the global store (accessible only via /upload-global)."""
        return self._global_files.add(source_path)

    def list_global_files(self) -> list:
        return self._global_files.list_uploads()

    def _get_memory_filepath(self) -> Path: return self.project_root / "data" / "long_term_memory.json"
    def _load_pinned_contents(self) -> list:
        path = self._get_memory_filepath()
        if not path.exists(): return []
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: return []
    def _save_pinned_contents(self, contents: list): self._get_memory_filepath().write_text(json.dumps(contents), encoding="utf-8")