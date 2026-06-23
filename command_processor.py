import os
import requests
from langchain_core.messages import HumanMessage, AIMessage
from __init__ import MODES

class ChatbotCommandProcessor:
    def handle_command(self, message: str) -> str or None:
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
            try:
                requests.post(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate", json={"model": "qwen2.5-coder:7b", "keep_alive": 0}, timeout=5)
            except Exception:
                pass
            self.set_mode(target_mode)
            return "Coder Mode Deactivated. Returning to target Llama defaults." if target_mode != "hibernate" else "Coder Mode Deactivated. Entering hibernate mode."

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

        return self._handle_memory_commands(cmd)

    def _handle_memory_commands(self, cmd: str) -> str or None:
        if cmd == "pin":
            user_msgs = [m for m in self._conversation_history if isinstance(m, HumanMessage)]
            ai_msgs = [m for m in self._conversation_history if isinstance(m, AIMessage)]
            if not user_msgs or not ai_msgs:
                return "Error: No conversation exchange found to pin."
            saved = self._load_pinned_contents()
            if len(saved) >= 20:
                return "Pin rejected: Pinned limits cannot exceed half of total VRAM context slots (Max 20)."
            last_user_text, last_ai_text = user_msgs[-1].content, ai_msgs[-1].content
            if any(isinstance(item, dict) and item.get("user") == last_user_text for item in saved):
                return "Message turn is already pinned."
            saved.append({"user": last_user_text, "assistant": last_ai_text})
            self._save_pinned_contents(saved)
            for m in reversed(self._conversation_history):
                if m.content in [last_user_text, last_ai_text]:
                    m.additional_kwargs["pinned"] = True
            return f"Success: Pinned turn sequence. ({len(saved)}/20 occupied)"

        if cmd == "unpin_all":
            self._save_pinned_contents([])
            for m in self._conversation_history:
                if "pinned" in m.additional_kwargs: m.additional_kwargs["pinned"] = False
            return "Cleared all long-term memory pins."

        if cmd.startswith("unpin "):
            try:
                idx = int(cmd.split()[1]) - 1
                saved = self._load_pinned_contents()
                if idx < 0 or idx >= len(saved): return f"Index error. Range 1 to {len(saved)}."
                removed_pair = saved.pop(idx)
                self._save_pinned_contents(saved)
                if isinstance(removed_pair, dict):
                    rem_user, rem_ai = removed_pair.get("user", ""), removed_pair.get("assistant", "")
                    for m in self._conversation_history:
                        if m.content in [rem_user, rem_ai]: m.additional_kwargs["pinned"] = False
                return f"Unpinned memory element {idx + 1}."
            except (ValueError, IndexError): return "Usage error. Format: unpin <integer_index>"

        if cmd == "list":
            saved = self._load_pinned_contents()
            if not saved: return "Long term context storage is empty."
            lines = [f"{i+1}. {item.get('user', '')[:60]}..." if isinstance(item, dict) else f"{i+1}. {str(item)[:60]}..." for i, item in enumerate(saved)]
            return "Pinned Memories (Prompts):\n" + "\n".join(lines)

        return None
