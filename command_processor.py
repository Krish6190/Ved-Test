import os
import sys
import subprocess
import requests
from langchain_core.messages import HumanMessage, AIMessage
from __init__ import MODES

class ChatbotCommandProcessor:
    def handle_command(self, message: str) -> str | None:
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

        mem_resp = self._handle_memory_commands(cmd)
        if mem_resp is not None:
            return mem_resp
        return self._handle_thread_commands(message)

    def _handle_memory_commands(self, cmd: str) -> str | None:
        if cmd == "/pin":
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
            self._save_threads()
            return f"Success: Pinned turn sequence. ({len(saved)}/20 occupied)"

        if cmd == "/unpin_all":
            self._save_pinned_contents([])
            for m in self._conversation_history:
                if "pinned" in m.additional_kwargs:
                    m.additional_kwargs["pinned"] = False
            self._save_threads()
            return "Cleared all long-term memory pins."

        if cmd.startswith("/unpin "):
            try:
                idx = int(cmd.split()[1]) - 1
                saved = self._load_pinned_contents()
                if idx < 0 or idx >= len(saved):
                    return f"Index error. Range 1 to {len(saved)}."
                removed_pair = saved.pop(idx)
                self._save_pinned_contents(saved)
                if isinstance(removed_pair, dict):
                    rem_user, rem_ai = removed_pair.get("user", ""), removed_pair.get("assistant", "")
                    for m in self._conversation_history:
                        if m.content in [rem_user, rem_ai]:
                            m.additional_kwargs["pinned"] = False
                self._save_threads()
                return f"Unpinned memory element {idx + 1}."
            except (ValueError, IndexError):
                return "Usage error. Format: /unpin <integer_index>"

        if cmd in ("/list", "/memories"):
            saved = self._load_pinned_contents()
            if not saved:
                return "Long term context storage is empty."
            lines = [
                f"{i+1}. {item.get('user', '')[:60]}..."
                if isinstance(item, dict)
                else f"{i+1}. {str(item)[:60]}..."
                for i, item in enumerate(saved)
            ]
            return "Pinned Memories (Prompts):\n" + "\n".join(lines)

        if cmd == "/run":
            # Open file dialog → confirm → execute script via subprocess.
            from tkinter import filedialog, messagebox
            chosen = filedialog.askopenfilename(
                title="Select Python Script to Run",
                filetypes=[("Python Scripts", "*.py"), ("All Files", "*.*")],
            )
            if not chosen:
                return "(run cancelled)"
            proceed = messagebox.askyesno(
                title="⚠️ Confirm Script Execution",
                message=(
                    f"Run this script?\n\n{chosen}\n\n"
                    "Script execution can modify files or make network calls. Continue?"
                ),
            )
            if not proceed:
                return "(run cancelled by user)"
            try:
                result = subprocess.run(
                    [sys.executable, chosen],
                    capture_output=True, text=True, timeout=30,
                )
                output = result.stdout or ""
                if result.stderr:
                    output += "\n[stderr]:\n" + result.stderr
                if result.returncode != 0:
                    output += f"\n[exit code {result.returncode}]"
                if not output:
                    output = "(script produced no output)"
                return f"Script output ({os.path.basename(chosen)}):\n{output[:2000]}"
            except subprocess.TimeoutExpired:
                return f"Script timed out after 30 seconds: {chosen}"
            except Exception as e:
                return f"Script failed: {e}"

        return None

    def _resolve_thread_ref(self, ref: str):
        threads = self.list_threads()
        if not threads:
            return None
        if ref is None or ref == "":
            return self._active_thread_id
        if ref.isdigit():
            idx = int(ref) - 1
            if 0 <= idx < len(threads):
                return threads[idx]["id"]
            return None
        for t in threads:
            if t["id"] == ref or t["id"].startswith(ref):
                return t["id"]
        return None

    def _handle_thread_commands(self, message: str) -> str | None:
        raw = message.strip()
        lower = raw.lower()
        if lower == "/new":
            new_id = self.create_thread()
            return f"Created thread {new_id}."
        if lower.startswith("/new "):
            title = raw[5:].strip()
            if not title:
                title = None
            new_id = self.create_thread(title)
            return f"Created thread {new_id} ({title})." if title else f"Created thread {new_id}."
        if lower in ("/threads", "/list_threads"):
            threads = self.list_threads()
            if not threads:
                return "No threads."
            lines = []
            for i, t in enumerate(threads, start=1):
                marker = "* " if t["id"] == self._active_thread_id else "  "
                title_display = t["title"][:30] if t["title"] else "(untitled)"
                lines.append(f"{marker}{i}. {t['id']}  {title_display}")
            return "Threads:\n" + "\n".join(lines)
        if lower.startswith("/switch"):
            parts = raw.split(maxsplit=1)
            if len(parts) < 2:
                return "Usage: /switch <id|number>"
            target = self._resolve_thread_ref(parts[1].strip())
            if target is None:
                return f"Unknown thread: {parts[1].strip()}"
            self.switch_thread(target)
            return f"Switched to thread {target}."
        if lower.startswith("/rename"):
            parts = raw.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                return "Usage: /rename <new_title>"
            if not self._active_thread_id:
                return "No active thread."
            self.rename_thread(self._active_thread_id, parts[1].strip())
            return f"Renamed thread to: {parts[1].strip()}"
        if lower.startswith("/delete"):
            parts = raw.split(maxsplit=1)
            if len(parts) < 2:
                target = self._active_thread_id
            else:
                target = self._resolve_thread_ref(parts[1].strip())
            if target is None:
                return f"Unknown thread: {parts[1].strip()}"
            if len(self._threads) <= 1:
                return "Cannot delete the last remaining thread."
            if not self.delete_thread(target):
                return f"Failed to delete thread {target}."
            return f"Deleted thread {target}."
        if lower == "/clear":
            if not self._active_thread_id or self._active_thread_id not in self._threads:
                return "No active thread."
            self._threads[self._active_thread_id]["messages"] = []
            self._save_threads()
            return "Cleared active thread messages."

        # /upload-global — add a file to the private global store (no thread scope).
        # Only reachable via the input bar, so only the local user can populate it.
        if lower == "/upload-global":
            from tkinter import filedialog
            supported = [(
                "Parsable Assets",
                "*.txt *.md *.pdf *.docx *.doc "
                "*.py *.js *.jsx *.ts *.tsx "
                "*.java *.go *.rs *.rb *.php *.cs "
                "*.cpp *.c *.h *.hpp *.swift *.kt *.scala "
                "*.sh *.bash *.zsh *.ps1 "
                "*.html *.css *.scss *.xml *.svg "
                "*.json *.yaml *.yml *.toml *.csv *.sql "
                "*.log *.zip"
            )]
            chosen = filedialog.askopenfilename(
                title="Attach File to Global Store",
                filetypes=supported,
            )
            if not chosen:
                return "(global upload cancelled)"
            try:
                meta = self.add_global_file(chosen)
            except FileNotFoundError as e:
                return f"[System Upload Rejected]: {e}"
            except Exception as e:
                return f"[System Upload Failure]: {e}"
            evicted = meta.get("evicted", [])
            msg = f"[Attached to global]: {meta['filename']} ({meta['chunk_count']} chunks)"
            if evicted:
                msg += f" — evicted {len(evicted)} oldest upload(s) to make room"
            return msg

        return None
