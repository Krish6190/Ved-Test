import os
import re
from pathlib import Path
from __init__ import DEFAULT_MODE, MODES
from langchain_core.messages import HumanMessage, SystemMessage
from graph import build_graph
import importlib


class ModelAdapter:
    """Lightweight adapter that reads a Modelfile and configures a local LLM."""

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

        if "num_gpu" in self.params:
            try:
                kwargs["num_gpu"] = int(self.params["num_gpu"])
            except Exception:
                kwargs["num_gpu"] = 0
        else:
            kwargs["num_gpu"] = 1 if self.device == "gpu" else 0

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
        self.adapters = {}
        for m in MODES:
            self.adapters[m] = None if m == "hibernate" else self._load_adapter_for_mode(m)

        self._llm_cache = {}
        self._conversation_history = []
        self._graph = build_graph(self._get_llm)

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
            device = "gpu" if mode == "turbo" else "cpu"
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
        self.mode = mode
        self._hibernating = (mode == "hibernate")

    def handle_command(self, message: str):
        cmd = message.strip().lower()
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
        if cmd == "/reload":
            try:
                # 1. Force Python to re-read your graph files from your disk
                import graph
                importlib.reload(graph)                
                from graph import state, nodes
                importlib.reload(state)
                importlib.reload(nodes)
                               # If you have a tools file or module, reload it too
                try:
                    from graph import tools
                    importlib.reload(tools)
                except ImportError:
                    pass
                # 2. Re-import the fresh build function and rebuild your graph layout
                from graph import build_graph
                self._graph = build_graph(self._get_llm)
                return "[System] Hot-reload successful! All updated files are now live."
            except Exception as e:
                return f"[System] Hot-reload failed: {e}"
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
        
        #Append the user message to history once and build the messages list for the graph input
        #Keep only the last 10 messages in memory!
        user_message = HumanMessage(content=message)
        self._conversation_history.append(user_message)
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]
        messages = []
        if adapter.system_prompt:
            messages.append(SystemMessage(content=adapter.system_prompt))
        messages.extend(self._conversation_history)

        try:
            result = self._graph.invoke({
                "messages": messages,
                "route_intent": "chat",
                "mode": self.mode,
            })
            output_messages = result.get("messages", [])
            if not output_messages:
                return "[ved] No response from graph."
            assistant_message = output_messages[-1]
            content = getattr(assistant_message, "content", str(assistant_message))
            self._conversation_history.append(assistant_message)
            return content
        except Exception as exc:
            return f"[ved] Graph error: {exc}"
