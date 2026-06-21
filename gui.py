import tkinter as tk
import tkinter.font as tkfont
import ctypes
import threading

from voice_module import VoiceSystem
from chatbot import Chatbot

# Windows API constant for excluding a window from screen capture
WDA_EXCLUDEFROMCAPTURE = 0x00000011

MODE_COLORS = {
    "turbo":     "#a6e3a1",
    "standard":  "#89b4fa",
    "hibernate": "#6c7086",
    "error":     "#f38ba8",
}

class VedWidget:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.chatbot = Chatbot()
        self.current_mode = self.chatbot.mode

        self.root.title("Ved")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#090a0f")

        self.TITLE_BAR_H = 32
        self.INPUT_BAR_H = 55
        self.default_width = 480
        self.default_content_h = 235 - self.TITLE_BAR_H - self.INPUT_BAR_H
        self.max_content_h = self.default_content_h * 2

        self.default_height = self.TITLE_BAR_H + self.default_content_h + self.INPUT_BAR_H
        self.max_height = self.TITLE_BAR_H + self.max_content_h + self.INPUT_BAR_H

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(
            f"{self.default_width}x{self.default_height}"
            f"+{sw - self.default_width - 20}+{sh - self.default_height - 60}"
        )
        self.root.minsize(self.default_width, self.default_height)
        self.root.maxsize(self.default_width, self.max_height)
        self.root.resizable(False, True)

        self.chat_history: list[dict] = []
        self._build_ui()
        self._hide_from_screen_capture()
        self._update_mode_ui(self.current_mode)
        self._set_output(f"Ved ready — {self.current_mode.upper()} mode.", "#a6e3a1")

    def _build_ui(self):
        title_bar = tk.Frame(self.root, bg="#12131b", height=self.TITLE_BAR_H)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        self._bind_title_drag(title_bar)

        title_label = tk.Label(
            title_bar, text="  ● VED", bg="#12131b", fg="#e5e9f0",
            font=("Segoe UI Semibold", 10, "bold"), anchor="w"
        )
        title_label.pack(side="left", padx=6)
        self._bind_title_drag(title_label)

        drag_area = tk.Frame(title_bar, bg="#12131b")
        drag_area.pack(fill="x", side="left", expand=True)
        self._bind_title_drag(drag_area)

        right_group = tk.Frame(title_bar, bg="#12131b")
        right_group.pack(side="right", padx=8)

        self.mode_buttons = {}
        self._drag_block_widgets = []
        for icon, label, mode, color in [
            ("⚡", "Turbo",    "turbo",     "#a6e3a1"),
            ("🧠", "Standard", "standard",  "#89b4fa"),
            ("😴", "Hibernate","hibernate", "#6c7086"),
        ]:
            button = tk.Frame(right_group, bg="#161b26", bd=1, relief="flat")
            button.pack(side="left", padx=4)
            button.configure(cursor="hand2")

            emoji_label = tk.Label(
                button, text=icon, bg="#161b26", fg="#e5e9f0",
                font=("Segoe UI Emoji", 10)
            )
            emoji_label.pack(side="left", padx=(8, 2), pady=5)

            text_label = tk.Label(
                button, text=label, bg="#161b26", fg=color,
                font=("Segoe UI", 9)
            )
            text_label.pack(side="left", padx=(0, 8), pady=5)

            for widget in (button, emoji_label, text_label):
                widget.bind("<Button-1>", lambda event, m=mode: self._on_mode_click(event, m))
                widget.bind("<B1-Motion>", lambda event: "break")
                self._drag_block_widgets.append(widget)

            self.mode_buttons[mode] = button

        close_button = tk.Button(
            right_group, text="✕", bg="#12131b", fg="#f38ba8",
            bd=0, activebackground="#12131b", activeforeground="#f38ba8",
            font=("Segoe UI", 11), command=self.root.destroy
        )
        close_button.pack(side="left", padx=6)
        close_button.bind("<B1-Motion>", lambda event: "break")
        self._drag_block_widgets.append(close_button)

        input_frame = tk.Frame(self.root, bg="#090a0f", height=self.INPUT_BAR_H)
        input_frame.pack(side="bottom", fill="x")
        input_frame.pack_propagate(False)

        self.input_entry = tk.Text(
            input_frame, 
            bg="#313244", 
            fg="#cdd6f4",
            insertbackground="white", 
            bd=0, 
            font=("Consolas", 11),
            wrap="word", # Keeps words from getting cut off at edges
        )
        self.voice = VoiceSystem(self.root, input_frame, self.input_entry, self._on_enter)

        self.input_entry.pack(side="left", fill="both", expand=True,padx=5, pady=10)
        self.input_entry.bind("<Return>", lambda event: [self._on_enter(event), "break"])
        self.input_entry.bind("<Shift-Return>", lambda event: None)

        self.content_frame = tk.Frame(self.root, bg="#090a0f")
        self.content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(8, 4))

        self.line_height = tkfont.Font(font=("Segoe UI", 10)).metrics("linespace")

        self.output_text = tk.Text(
            self.content_frame,
            bg="#090a0f", fg="#e5e9f0", font=("Segoe UI", 10),
            wrap="word", bd=0, highlightthickness=0, padx=2, pady=2,
            state="disabled", spacing3=4, cursor="arrow",
        )
        self.output_text.pack(side="left", fill="both", expand=True)

        self.output_scroll = tk.Scrollbar(self.content_frame, command=self.output_text.yview)
        self.output_scroll.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=self.output_scroll.set)

    def _bind_title_drag(self, widget):
        widget.bind("<Button-1>", self._start_drag)
        widget.bind("<B1-Motion>", self._drag_window)

    def _start_drag(self, event):
        target = event.widget
        if getattr(self, '_drag_block_widgets', None) and target in self._drag_block_widgets:
            return
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        self._orig_x = self.root.winfo_x()
        self._orig_y = self.root.winfo_y()

    def _drag_window(self, event):
        dx = event.x_root - self._drag_x
        dy = event.y_root - self._drag_y
        self.root.geometry(f"+{self._orig_x + dx}+{self._orig_y + dy}")

    def _switch_mode(self, mode: str):
        self._append_text(f"[System] Switching to {mode.upper()}...\n\n", "#f9e2af")
        threading.Thread(target=self._do_switch_mode, args=(mode,), daemon=True).start()

    def _on_mode_click(self, event, mode: str):
        self.root.event_generate("<ButtonRelease-1>")
        self._switch_mode(mode)

    def _do_switch_mode(self, mode: str):
        try:
            self.chatbot.set_mode(mode)
            self.current_mode = mode
            self._update_mode_ui(mode)
            self._append_text(f"[System] Ved switched to {mode.upper()} mode.\n\n", "#cdd6f4")
        except Exception as e:
            self._append_text(f"[System] Mode switch failed: {e}\n\n", MODE_COLORS["error"])

    def _refresh_mode_status(self):
        self.current_mode = self.chatbot.mode
        self._update_mode_ui(self.current_mode)

    def _update_mode_ui(self, mode: str):
        for m, button in self.mode_buttons.items():
            if m == mode:
                button.config(bg="#2e3440", bd=1, relief="solid")
                for child in button.winfo_children():
                    child.config(bg="#2e3440")
            else:
                button.config(bg="#161b26", bd=0, relief="flat")
                for child in button.winfo_children():
                    child.config(bg="#161b26")

    def _on_enter(self, event):
        threading.Thread(target=self._send_command, daemon=True).start()

    def _insert_colored(self, text: str, color: str):
        end_index = self.output_text.index("end-1c")
        self.output_text.insert("end", text)
        start_index = f"{end_index} + 0 chars"
        end_index = self.output_text.index("end-1c")
        tag_name = f"color_{start_index.replace('.', '_').replace(' ', '')}"
        self.output_text.tag_configure(tag_name, foreground=color)
        self.output_text.tag_add(tag_name, start_index, end_index)

    def _append_text(self, text: str, color: str = "#cdd6f4"):
        def action():
            self.output_text.configure(state="normal")
            self._insert_colored(text, color)
            self.output_text.configure(state="disabled")
            self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, action)

    def _render_chat_history(self):
        def action():
            self.output_text.configure(state="normal")
            self.output_text.delete("1.0", "end")
            for turn in self.chat_history:
                self._insert_colored("You: ", "#89b4fa")
                self._insert_colored(f"{turn['user']}\n\n", "#e5e9f0")
                self._insert_colored("Ved: ", "#a6e3a1")
                self._insert_colored(f"{turn['assistant']}\n\n", "#e5e9f0")
            self.output_text.configure(state="disabled")
            self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, action)

    def _send_command(self):
        prompt = self.input_entry.get("1.0", tk.END).strip()
        if not prompt:
            return

        self.root.after(0, lambda: self.input_entry.delete("1.0", tk.END))
        self.root.after(0, lambda: self.input_entry.config(state="disabled"))
        self.root.after(0, self._render_chat_history)

        self._append_text("You: ", "#89b4fa")
        self._append_text(f"{prompt}\n\n")

        full_response = ""
        try:
            full_response = self.chatbot.respond(prompt)
            self._append_text("Ved: ", "#a6e3a1")
            self._append_text(f"{full_response}\n\n")
        except Exception as e:
            full_response = f"Chatbot error: {e}"
            self._append_text(f"\n{full_response}", MODE_COLORS["error"])
        finally:
            if full_response.strip():
                self.chat_history.append({
                    "user": prompt,
                    "assistant": full_response.strip(),
                })
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]
                self.root.after(0, self._render_chat_history)

            self._refresh_mode_status()
            self.root.after(0, lambda: self.input_entry.config(state="normal"))
            self.root.after(0, lambda: self.input_entry.focus())

    def _hide_from_screen_capture(self):
        try:
            self.root.update()
            raw_id = self.root.winfo_id()
            result = ctypes.windll.user32.SetWindowDisplayAffinity(
                raw_id, WDA_EXCLUDEFROMCAPTURE
            )
            if not result:
                hwnd_parent = ctypes.windll.user32.GetParent(raw_id)
                if hwnd_parent:
                    result = ctypes.windll.user32.SetWindowDisplayAffinity(
                        hwnd_parent, WDA_EXCLUDEFROMCAPTURE
                    )
            if not result:
                print("[ved] Warning: could not set display affinity (older Windows build?)")
        except Exception as e:
            print(f"[ved] Display affinity not available: {e}")

    def _set_output(self, text: str, color: str = "#cdd6f4"):
        def update():
            self.output_text.configure(state="normal")
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", text)
            self.output_text.tag_configure("color", foreground=color)
            self.output_text.tag_add("color", "1.0", "end")
            self.output_text.configure(state="disabled")
            self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, update)

    def _resize_to_fit_content(self):
        self.root.update_idletasks()
        count_result = self.output_text.count("1.0", "end-1c", "displaylines")
        if count_result is not None:
            if isinstance(count_result, (list, tuple)):
                num_lines = int(count_result[0] or 1)
            else:
                num_lines = int(count_result or 1)
        else:
            num_lines = 1

        needed_content_h = num_lines * self.line_height + 16
        target_content_h = max(self.default_content_h, min(needed_content_h, self.max_content_h))
        target_window_h = self.TITLE_BAR_H + target_content_h + self.INPUT_BAR_H

        current_h = self.root.winfo_height()
        if abs(target_window_h - current_h) > 2:
            current_x = self.root.winfo_x()
            current_y = self.root.winfo_y()
            height_difference = target_window_h - current_h
            new_y = current_y - height_difference
            self.root.geometry(f"{self.default_width}x{int(target_window_h)}+{current_x}+{int(new_y)}")


def main():
    root = tk.Tk()
    app = VedWidget(root)
    root.mainloop()

if __name__ == "__main__":
    main()
