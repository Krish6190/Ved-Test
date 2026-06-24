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
from .components import VedComponentLayout

MODE_COLORS = {
    "turbo":     "#a6e3a1",
    "standard":  "#89b4fa",
    "hibernate": "#6c7086",
    "coder":     "#cba6f7",
    "error":     "#f38ba8",
}
MODE_COMMANDS = {
    "/activate coder", "/deactivate coder",
    "/sleep", "/hibernate", "/wake", "/resume",
}

class VedWidget(VedComponentLayout):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.chatbot      = Chatbot()
        self.current_mode = self.chatbot.mode
        self.chat_history = []
        self.is_generating = False

        input_frame = self._build_ui_layout(self._on_mode_click)
        self.voice  = VoiceSystem(self.root, input_frame, self.input_entry, self._on_enter)
        self.input_entry.pack(side="left", fill="both", expand=True, padx=5, pady=10)

        self.input_entry.bind(
            "<Return>",
            lambda ev: [
                threading.Thread(target=self._send_command, daemon=True).start()
                if not self.is_generating else None,
                "break",
            ],
        )

        self._hide_from_screen_capture()
        self._update_mode_ui(self.current_mode)
        self.root.bind("<FocusIn>", lambda ev: self.input_entry.focus_set())
    # ------------------------------------------------------------------ #
    # Mode switching
    # ------------------------------------------------------------------ #
    def _on_mode_click(self, event, mode: str):
        self.root.event_generate("<ButtonRelease-1>")
        if mode == self.current_mode:
            return
        if mode == "coder":
            msg = "[System] Initializing CODER pipeline...\n"
        elif self.current_mode == "coder":
            msg = "[System] Deactivating CODER...\n"
        else:
            msg = f"[System] Switching to {mode.upper()}...\n"

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
        self.sticky_header_label.config(
            text=f"Ved ready — {mode.upper()} mode.",
            fg=MODE_COLORS.get(mode, "#a6e3a1"),
        )
        for m, btn in self.mode_buttons.items():
            active = m == mode
            bg     = "#2e3440" if active else "#161b26"
            btn.config(bg=bg, bd=(1 if active else 0), relief=("solid" if active else "flat"))
            for child in btn.winfo_children():
                child.config(bg=bg)
    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _on_enter(self, event):
        threading.Thread(target=self._send_command, daemon=True).start()

    def _send_command(self):
        prompt = self.input_entry.get("1.0", tk.END).strip()
        if not prompt:
            return

        self.root.after(0, lambda: self.input_entry.delete("1.0", tk.END))
        self.is_generating = True
        self._append_text("You: ", "#89b4fa")
        self._append_text(f"{prompt}\n", "#e5e9f0")

        try:
            response_obj  = self.chatbot.respond(prompt)
            full_response = self._consume_response(response_obj)
            # Sync mode if a slash command changed it on the chatbot side
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
        """Handle both plain-string and streaming generator responses."""
        if isinstance(response_obj, str):
            self._append_text("Ved: ", "#a6e3a1")
            self._append_text(f"{response_obj}\n", "#e5e9f0")
            return response_obj

        full_response    = ""
        printed_header   = False

        for event_type, chunk in response_obj:
            if event_type == "token":
                if not printed_header:
                    self._append_text("Ved: ", "#a6e3a1")
                    printed_header = True
                full_response += chunk
                self._append_stream_chunk(chunk, "#e5e9f0")
                time.sleep(0.01)
            elif event_type == "error":
                self._append_text(f"\n{chunk}\n", MODE_COLORS["error"])

        if printed_header:
            self._append_text("\n", "#e5e9f0")
        elif full_response.strip():
            self._append_text("Ved: ", "#a6e3a1")
            self._append_text(f"{full_response.strip()}\n", "#e5e9f0")

        return full_response
    # ------------------------------------------------------------------ #
    # Output rendering
    # ------------------------------------------------------------------ #
    def _append_text(self, text: str, color: str = "#e5e9f0"):
        def action():
            self.output_text.configure(state="normal")
            idx = self.output_text.index("end-1c")
            self.output_text.insert("end", text)
            tag = f"col_{idx.replace('.', '_')}_{len(text)}"
            self.output_text.tag_configure(tag, foreground=color)
            self.output_text.tag_add(tag, idx, "end-1c")
            self.output_text.configure(state="disabled")
            self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, action)

    def _append_stream_chunk(self, text: str, color: str = "#e5e9f0"):
        # Separate from _append_text so stream tags don't bleed into prior text regions
        def action():
            self.output_text.configure(state="normal")
            idx = self.output_text.index("end-1c")
            self.output_text.insert("end", text)
            tag = f"stream_{idx.replace('.', '_')}_{len(text)}"
            self.output_text.tag_configure(tag, foreground=color)
            self.output_text.tag_add(tag, idx, "end-1c")
            self.output_text.configure(state="disabled")
            self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, action)

    def _render_chat_history(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")
        for turn in self.chat_history:
            self._append_text("You: ", "#89b4fa")
            self._append_text(f"{turn['user']}\n", "#e5e9f0")
            self._append_text("Ved: ", "#a6e3a1")
            self._append_text(f"{turn['assistant']}\n", "#e5e9f0")
        self.output_text.see("end")
        self._resize_to_fit_content()

    def _resize_to_fit_content(self):
        self.root.update_idletasks()
        count    = self.output_text.count("1.0", "end-1c", "displaylines")
        num_lines = (
            int(count[0]) if isinstance(count, (list, tuple))
            else (int(count) if count is not None
                  else int(self.output_text.index("end-1c").split(".")[0]))
        )
        target_h = max(
            self.default_content_h,
            min((num_lines * self.line_height) + 16, self.max_content_h)
        )
        new_h = self.TITLE_BAR_H + target_h + self.INPUT_BAR_H
        self.root.geometry(
            f"{self.default_width}x{new_h}"
            f"+{self.root.winfo_x()}+{self.root.winfo_y() - (new_h - self.root.winfo_height())}"
        )

def main():
    root = tk.Tk()
    VedWidget(root)
    root.mainloop()