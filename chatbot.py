import json
import os
from pathlib import Path
import requests
from __init__ import DEFAULT_MODE, MODES
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from graph import build_graph
from model_adapter import ModelAdapter, parse_modelfile
from command_processor import ChatbotCommandProcessor
import queue
import threading

class Chatbot(ChatbotCommandProcessor):
    def __init__(self, mode=None):
        self.mode = mode or DEFAULT_MODE
        self._hibernating = (self.mode == "hibernate")
        self.project_root = Path(__file__).resolve().parent
        self.memory_db_path = self.project_root / "data" / "memories.json"
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_memories = self._load_persistent_memories()
        self.adapters = {}
        for m in MODES:
            self.adapters[m] = None if m == "hibernate" else self._load_adapter_for_mode(m)
        self._llm_cache = {}
        self._conversation_history = []
        self._graph = build_graph(self._get_llm)
        print(f"[DEBUG] graph after init: {self._graph}")

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
            pinned_memories = self._load_pinned_contents()
            if not self._conversation_history:
                initial_messages = [SystemMessage(content=adapter.system_prompt)]
                for pair in pinned_memories:
                    if isinstance(pair, dict):
                        initial_messages.append(HumanMessage(content=pair.get("user", "")))
                        initial_messages.append(AIMessage(content=pair.get("assistant", "")))
                initial_messages.append(HumanMessage(content=message))
            else:
                initial_messages = list(self._conversation_history) + [HumanMessage(content=message)]
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
            config = {"configurable": {"system_prompt": adapter.system_prompt, "token_queue": token_queue, "tool_approved": True}}
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
                if isinstance(item, tuple) and item[0] == "error":
                    yield ("error", item[1])
                else:
                    yield ("token", item)
            if accumulated_state and "messages" in accumulated_state:
                final_msgs = accumulated_state["messages"]
                if len(final_msgs) > len(initial_messages):
                    self._conversation_history = list(final_msgs)
                else:
                    new_ai = [m for m in final_msgs if isinstance(m, AIMessage) and m not in initial_messages]
                    if new_ai:
                        self._conversation_history = list(initial_messages) + new_ai
                    else:
                        self._conversation_history = list(final_msgs)
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
    
    def _get_memory_filepath(self) -> Path: return self.project_root / "data" / "long_term_memory.json"
    def _load_pinned_contents(self) -> list:
        path = self._get_memory_filepath()
        if not path.exists(): return []
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: return []
    def _save_pinned_contents(self, contents: list): self._get_memory_filepath().write_text(json.dumps(contents), encoding="utf-8")