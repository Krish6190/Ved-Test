import json
import os
import re
from pathlib import Path
from __init__ import DEFAULT_MODE, MODES
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from graph import build_graph
import requests

class ModelAdapter:
    def __init__(self, model_name: str = "local-stub", device: str = "cpu", params=None, system_prompt: str = ""):
        self.model_name = model_name
        self.device = device
        self.params = params or {}
        self.system_prompt = system_prompt

    def create_llm(self, base_url: str):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("langchain-ollama is required for the ved graph flow.") from exc

        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
            "temperature": float(self.params.get("temperature", 0.1)),
            "keep_alive": "20m",
        }
        for key, value in self.params.items():
            kwargs[key] = value
        kwargs.setdefault("temperature", 0.1)
        if "num_gpu" not in kwargs:
            kwargs["num_gpu"] = 1 if self.device == "gpu" else 0
        else:
            try:
                kwargs["num_gpu"] = int(kwargs["num_gpu"])
            except Exception:
                kwargs["num_gpu"] = 0
        return ChatOllama(**kwargs)

def _parse_modelfile(path: Path) -> dict:
    data = {"from": None, "params": {}, "system": ""}
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8")
    m = re.search(r"FROM\s+(.+)", text)
    if m:
        data["from"] = m.group(1).strip()
    for pm in re.finditer(r"PARAMETER\s+(\S+)\s+(.+)", text):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except Exception:
                pass
        data["params"][key] = val
    sys_m = re.search(r"SYSTEM\s+\"\"\"([\s\S]*?)\"\"\"", text)
    if sys_m:
        data["system"] = sys_m.group(1).strip()
    return data

class Chatbot:
    def __init__(self, mode=None):
        self.mode = mode or DEFAULT_MODE
        self._hibernating = (self.mode == "hibernate")
        self.project_root = Path(__file__).resolve().parent
        self.memory_db_path = self.project_root / "memories.json"
        self.saved_memories = self._load_persistent_memories()
        self.adapters = {}
        for m in MODES:
            self.adapters[m] = None if m == "hibernate" else self._load_adapter_for_mode(m)
        self._llm_cache = {}
        self._conversation_history = []
        self._graph = build_graph(self._get_llm)

    def _load_persistent_memories(self) -> list[str]:
        if self.memory_db_path.exists():
            try:
                return json.loads(self.memory_db_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_persistent_memories(self):
        try:
            self.memory_db_path.write_text(json.dumps(self.saved_memories, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_adapter_for_mode(self, mode: str) -> ModelAdapter:
        fname = f"Modelfile.{mode}"
        path = self.project_root / fname
        info = _parse_modelfile(path)
        model_name = info.get("from") or f"{mode}-stub"
        params = info.get("params", {})
        device = "cpu"
        if "num_gpu" in params:
            try:
                if int(params["num_gpu"]) > 0:
                    device = "gpu"
            except Exception:
                device = "cpu"
        else:
            device = "gpu" if mode in ["turbo", "coder"] else "cpu"
        return ModelAdapter(model_name=model_name, device=device, params=params, system_prompt=info.get("system", ""))

    def _get_llm(self):
        if self.mode == "hibernate":
            return None
        if self.mode in self._llm_cache:
            return self._llm_cache[self.mode]
        adapter = self.adapters.get(self.mode)
        if adapter is None:
            return None
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = adapter.create_llm(base_url=base_url)
        self._llm_cache[self.mode] = llm
        return llm

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if self.mode == "coder" and mode in ["standard", "turbo"]:
            raise RuntimeError("Hardware Interlock Triggered: Standard mode switching is blocked while coder mode is active.")
        if mode == self.mode:
            return
        old_mode = self.mode
        self.mode = mode
        self._hibernating = (mode == "hibernate")
        self._llm_cache.clear()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        previous_adapter = self.adapters.get(old_mode)
        if previous_adapter and previous_adapter.model_name:
            try:
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": previous_adapter.model_name, "keep_alive": 0},
                    timeout=5
                )
            except Exception:
                pass
        if mode != "hibernate":
            self._graph = build_graph(self._get_llm)
            active_adapter = self.adapters.get(mode)
            if active_adapter and active_adapter.model_name:
                try:
                    requests.post(
                        f"{base_url}/api/generate",
                        json={
                            "model": active_adapter.model_name, 
                            "prompt": "", 
                            "keep_alive": "20m"
                        },
                        timeout=15
                    )
                except Exception:
                    pass
    def _get_memory_filepath(self) -> Path:
        return self.project_root / "long_term_memory.json"

    def _load_pinned_contents(self) -> list:
        import json
        path = self._get_memory_filepath()
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_pinned_contents(self, contents: list):
        import json
        self._get_memory_filepath().write_text(json.dumps(contents, indent=2), encoding="utf-8")

    def handle_command(self, message: str):
        cmd = message.strip().lower()
        if cmd == "/activate coder":
            try:
                self.set_mode("coder")
                return "Coder Mode Active. Specialized Qwen 2.5 Coder 7B preloaded on GPU."
            except Exception as e:
                return f"Failed to activate coder mode: {e}"
        if cmd.startswith("/deactivate coder"):
            if self.mode != "coder":
                return "Coder mode is not currently active."
            parts = cmd.split()
            target_mode = parts[2] if len(parts) >= 3 and parts[2] in MODES else "standard"
            self.mode = target_mode
            self._hibernating = (target_mode == "hibernate")
            self._llm_cache.clear()
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                requests.post(f"{base_url}/api/generate", json={"model": "qwen2.5-coder:7b", "keep_alive": 0}, timeout=5)
            except Exception:
                pass
            if target_mode != "hibernate":
                self.set_mode(target_mode)
                return f"Coder Mode Deactivated. System redirected to {target_mode.upper()} configuration."
            else:
                self.set_mode("hibernate")
                return "Coder Mode Deactivated. Entering hibernate mode."
        if self.mode == "coder" and not self._hibernating and (cmd in ("/wake", "/resume") or cmd.startswith("/mode")):
            return "Command Rejected: Hardware configuration adjustments are blocked while coder mode is active."
        if cmd in ("/hibernate", "/sleep"):
            self.set_mode("hibernate")
            return "Entering hibernate mode. Use /wake to wake me."
        if cmd in ("/wake", "/resume"):
            self.set_mode("standard")
            return "Waking up. Mode set to standard."
        if cmd.startswith("/mode"):
            parts = cmd.split()
            if len(parts) >= 2 and parts[1] in MODES:
                self.set_mode(parts[1])
                return f"Mode set to {parts[1]}."
            return f"Usage: /mode [{'|'.join(MODES)}]"
        if cmd == "pin":
            user_msgs = [m for m in self._conversation_history if isinstance(m, HumanMessage)]
            if not user_msgs:
                return "Error: No conversation exchange found to pin."
            
            saved = self._load_pinned_contents()
            if len(saved) >= 20:
                return "Pin rejected: Pinned limits cannot exceed half of total VRAM context slots (Max 20)."
                
            last_text = user_msgs[-1].content
            if last_text in saved:
                return "Message is already pinned."
                
            saved.append(last_text)
            self._save_pinned_contents(saved)
            for m in reversed(self._conversation_history):
                if m.content == last_text:
                    m.additional_kwargs["pinned"] = True
                    break
            return f"Success: Pinned conversation history segment. ({len(saved)}/20 occupied)"

        if cmd == "unpin_all":
            self._save_pinned_contents([])
            for m in self._conversation_history:
                if "pinned" in m.additional_kwargs:
                    m.additional_kwargs["pinned"] = False
            return "Cleared all long-term memory pins."

        if cmd.startswith("unpin "):
            try:
                idx = int(cmd.split()[1]) - 1
                saved = self._load_pinned_contents()
                if idx < 0 or idx >= len(saved):
                    return f"Index error. Use an integer between 1 and {len(saved)}."
                removed_text = saved.pop(idx)
                self._save_pinned_contents(saved)
                # Unmark matching history frames instantly
                for m in self._conversation_history:
                    if m.content == removed_text:
                        m.additional_kwargs["pinned"] = False
                return f"Unpinned memory element {idx + 1}."
            except (ValueError, IndexError):
                return "Usage error. Format: unpin <integer_index>"
        if cmd == "list":
            saved = self._load_pinned_contents()
            if not saved:
                return "Long term context storage is empty."
            lines = [f"{i+1}. {text[:60]}..." for i, text in enumerate(saved)]
            return "Pinned Memories:\n" + "\n".join(lines)
        return None

    def respond(self, message: str) -> str:
        cmd_resp = self.handle_command(message)
        if cmd_resp is not None:
            return cmd_resp
        if self._hibernating:
            return "(hibernate) Bot is currently hibernating. Use /wake to wake."
        adapter = self.adapters.get(self.mode)
        if adapter is None:
            return "No model available for current mode."
        pinned_texts = self._load_pinned_contents()
        if not self._conversation_history:
            initial_messages = [SystemMessage(content=adapter.system_prompt)]
            for p_text in pinned_texts:
                initial_messages.append(HumanMessage(content=p_text, additional_kwargs={"pinned": True}))
            initial_messages.append(HumanMessage(content=message))
        else:
            initial_messages = [HumanMessage(content=message)]
        try:
            result = self._graph.invoke({
                "messages": initial_messages,
                "route_intent": "",  # Decided dynamically inside graph nodes
                "mode": self.mode,
                "saved_memories": getattr(self, "saved_memories", []),
                "current_draft": "",
                "critique_notes": "",
                "essay_score": 0,
                "loop_count": 0
            }, config={"configurable": {"system_prompt": adapter.system_prompt}})
            output_messages = result.get("messages", [])
            if not output_messages:
                return "[ved] No response from graph."
            assistant_message = output_messages[-1]
            content = getattr(assistant_message, "content", str(assistant_message))
            self._conversation_history = list(output_messages)
            executed_route = result.get("route_intent", "Unknown/Bypassed")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_active_models = ["None (Loaded from script cache)"]
            try:
                r = requests.get(f"{base_url}/api/ps", timeout=2)
                if r.status_code == 200:
                    models_data = r.json().get("models", [])
                    if models_data:
                        ollama_active_models = [m.get("name") for m in models_data]
            except Exception:
                ollama_active_models = ["Error reading Ollama status"]
            print("\n" + "="*60)
            print("[VED HARDWARE DEBUG]")
            print(f"  -> Requested Python Mode: {self.mode.upper()}")
            print(f"  -> Active Graph Node Route: Path {executed_route}")
            print(f"  -> User Input Prompt: '{message}'")
            print(f"  -> Models Active in RAM/VRAM: {', '.join(ollama_active_models)}")
            print(f"  -> Total Tracked Context: {len(self._conversation_history)} messages")
            print(f"  -> Long-Term Memories Saved: {len(self.saved_memories)}")
            print("="*60 + "\n")
            return content
        except Exception as exc:
            return f"[ved] Graph error: {exc}"