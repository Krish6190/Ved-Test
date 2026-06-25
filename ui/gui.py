import os
import sys
import time
import threading
import tkinter as tk
import winsound
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from voice.voice_module import VoiceSystem
from chatbot import Chatbot
from .gui_rag_worker import VedRagWorker
from .components import MODE_COLORS

MODE_COMMANDS = {"/activate coder", "/deactivate coder", "/sleep", "/hibernate", "/wake", "/resume"}

class VedWidget(VedRagWorker):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.chatbot = Chatbot()
        self.current_mode = self.chatbot.mode
        self.chat_history = []
        self.is_generating = False

        input_frame = self._build_ui_layout(self._on_mode_click)
        self.voice  = VoiceSystem(self.root, input_frame, self.input_entry, self._on_enter)
        self.input_entry.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        if hasattr(self, "upload_btn") and self.upload_btn:
            self.upload_btn.config(command=self._trigger_file_attachment)

        self.input_entry.bind(
            "<Return>",
            lambda ev: [
                threading.Thread(target=self._send_command, daemon=True).start() if not self.is_generating else None,
                "break",
            ],
        )

        self._hide_from_screen_capture()
        self._update_mode_ui(self.current_mode)
        def focus_guard(event):
            if event.widget == self.root or str(event.widget).endswith('frame'):
                self.input_entry.focus_set()
        self.root.bind("<FocusIn>", focus_guard)

    def _on_mode_click(self, event, mode: str):
        self.root.event_generate("<ButtonRelease-1>")
        if mode == self.current_mode: return
        msg = f"[System] Initializing CODER pipeline...\n" if mode == "coder" else (
            "[System] Deactivating CODER...\n" if self.current_mode == "coder" else f"[System] Switching to {mode.upper()}...\n"
        )
        self._append_text(msg, "#f9e2af")
        threading.Thread(target=self._do_switch_mode, args=(mode,), daemon=True).start()

    def _do_switch_mode(self, mode: str):
        try:
            self.chatbot.set_mode(mode)
            self.current_mode = mode
            self.root.after(0, lambda: self._update_mode_ui(mode))
            wake_sound = os.getenv("wake_sound", "")
            if mode == "turbo" and os.path.exists(wake_sound):
                winsound.PlaySound(wake_sound, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.root.after(0, self._render_chat_history)
        except Exception as e:
            self._append_text(f"[System] Mode switch failed: {e}\n", MODE_COLORS["error"])

    def _update_mode_ui(self, mode: str):
        self.sticky_header_label.config(text=f"Ved ready — {mode.upper()} mode.", fg=MODE_COLORS.get(mode, "#a6e3a1"))
        for m, btn in self.mode_buttons.items():
            active = m == mode
            bg     = "#2e3440" if active else "#161b26"
            btn.config(bg=bg, bd=(1 if active else 0), relief=("solid" if active else "flat"))
            for child in btn.winfo_children(): child.config(bg=bg)

    def _on_enter(self, event):
        threading.Thread(target=self._send_command, daemon=True).start()

    def _send_command(self):
        prompt = self.input_entry.get("1.0", tk.END).strip()
        if not prompt: return
        self.user_scrolled_up = False
        self.is_bold_active = False
        self.token_buffer = ""
        self.is_snapped_to_max = False
        self.resize_throttle_counter = 0
        self.root.after(0, lambda: self.input_entry.delete("1.0", tk.END))
        self.is_generating = True
        self._append_text("You: ", "#89b4fa")
        if len(prompt) > 300:
            self._append_text(f"{prompt[:300]}... [Long context truncated in UI history]\n", "#e5e9f0")
        else:
            self._append_text(f"{prompt}\n", "#e5e9f0")

        if len(prompt.split()) < 6 and not prompt.startswith("/") and not os.path.exists(prompt):
            pass
        elif len(prompt) > 1700:
            self._process_rag_ingest_pipeline(prompt, is_raw_file=False)
            self._append_text("[System: Isolating human instruction from raw data dump...]\n", "#89b4fa")
            prompt = self._extract_real_human_prompt(prompt)
            self._append_text(f"[System Extracted Command]: {prompt}\n", "#a6e3a1")
        elif os.path.exists(prompt) and os.path.isfile(prompt):
            ext = os.path.splitext(prompt).lower()
            if ext in {".txt", ".py", ".md", ".json", ".js", ".cpp", ".h"}:
                self._process_rag_ingest_pipeline(prompt, is_raw_file=True)
                self.is_generating = False
                return
            self._append_text("[System Warning: Extension type rejected.]\n", "#f38ba8")
            self.is_generating = False
            return

        try:
            from graph.nodes import intent_router_node
            from tkinter import messagebox
            predicted_intent = "A"
            if prompt.startswith("/") or prompt.startswith("execute"):
                predicted_intent = "C"
            if predicted_intent == "C":
                user_choice = messagebox.askyesno(
                    title="⚠️ Critical Tool Execution Authorization Requested",
                    message=f"Ved is requesting administrative permission to run this action:\n\n'{prompt}'\n\nDo you authorize this terminal execution?",
                    parent=self.root
                )
                if not user_choice:
                    self._append_text("\n[System Notice: Tool execution blocked by human supervisor parameter.]\n", "#f38ba8")
                    self.is_generating = False
                    return
            response_obj = self.chatbot.respond(prompt)
            full_response = self._consume_response(response_obj)
            cleaned = prompt.strip().lower()
            if cleaned in MODE_COMMANDS or cleaned.startswith(("/deactivate coder", "/mode")):
                self.current_mode = self.chatbot.mode
                self.root.after(0, lambda: self._update_mode_ui(self.current_mode))
            if full_response.strip():
                self.chat_history.append({"user": prompt, "assistant": full_response.strip()})
                self.chat_history = self.chat_history[-10:]
        except Exception as e:
            self._append_text(f"\nChatbot error: {e}\n", MODE_COLORS["error"])
        finally:
            self.is_generating = False
            self.root.after(0, lambda: self.input_entry.focus())

    def _consume_response(self, response_obj) -> str:
        if isinstance(response_obj, str):
            self._append_text(response_obj, color="")
            return response_obj
            
        full_response, printed_newline = "", False
        for event_type, chunk in response_obj:
            if event_type == "token":
                full_response += chunk
                self._append_stream_chunk(chunk)
            elif event_type == "error": 
                self._append_text(f"[System Error]: {chunk}\n", MODE_COLORS["error"])
                
        if full_response.strip():
            self._append_text("\n", color="")
            
        return full_response

def main():
    root = tk.Tk()
    VedWidget(root)
    root.mainloop()
