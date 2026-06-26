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
        # Modern file-input: staged attachments shown as chips above the input row.
        # Populated by _trigger_file_attachment, drained by _send_command.
        self.pending_attachments = []
        self.chip_frame = tk.Frame(self.root, bg="#090a0f")

        self.thread_callbacks = {
            "on_new": self._create_new_thread,
            "on_switch": self._switch_thread,
        }
        input_frame = self._build_ui_layout(self._on_mode_click, thread_callbacks=self.thread_callbacks)
        self.input_frame = input_frame
        self._build_approval_bar()
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
        # Slash autocomplete: when the user types "/" at the start of the input,
        # show a scrollable popup listing all slash commands.
        self.cmd_popup = None
        self.cmd_listbox = None
        self.input_entry.bind("<KeyRelease>", self._on_input_keyrelease)

        self._hide_from_screen_capture()
        self._update_mode_ui(self.current_mode)

        # Populate the freshly-built tab strip from the active chatbot state.
        self._refresh_thread_tabs()
        self._render_active_thread()

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

    # ------------------------------------------------------------------ #
    # Slash autocomplete popup
    # ------------------------------------------------------------------ #
    def _on_input_keyrelease(self, event):
        """Show / hide slash command popup based on whether input starts with '/'."""
        if event.keysym in ("Return", "Escape", "Up", "Down"):
            if event.keysym == "Escape":
                self._hide_cmd_popup()
            return
        text = self.input_entry.get("1.0", tk.END).lstrip()
        if text.startswith("/"):
            self._show_cmd_popup(text)
        else:
            self._hide_cmd_popup()

    def _show_cmd_popup(self, current_text):
        if self.cmd_popup is None:
            self.cmd_popup = tk.Toplevel(self.root)
            self.cmd_popup.wm_overrideredirect(True)
            self.cmd_popup.configure(bg="#1e2030")

            frame = tk.Frame(self.cmd_popup, bg="#1e2030")
            frame.pack(fill="both", expand=True)

            scrollbar = tk.Scrollbar(frame, bg="#1e2030")
            scrollbar.pack(side="right", fill="y")

            self.cmd_listbox = tk.Listbox(
                frame,
                bg="#1e2030", fg="#cdd6f4",
                font=("Segoe UI", 10),
                selectbackground="#313244", selectforeground="#cdd6f4",
                highlightthickness=0, bd=0,
                yscrollcommand=scrollbar.set,
                height=8,
                activestyle="none",
            )
            self.cmd_listbox.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=self.cmd_listbox.yview)

            self.cmd_listbox.bind("<ButtonRelease-1>", self._on_cmd_select)
            self.cmd_listbox.bind("<Return>", self._on_cmd_select)
            self.cmd_listbox.bind("<Double-Button-1>", self._on_cmd_select)
            self.cmd_listbox.bind("<Escape>", lambda e: self._hide_cmd_popup())

        # Populate (filtered by current input prefix)
        all_cmds = self._list_all_commands()
        prefix = current_text.lower()
        filtered = [c for c in all_cmds if c.lower().startswith(prefix)] or all_cmds
        self.cmd_listbox.delete(0, tk.END)
        for cmd in filtered:
            self.cmd_listbox.insert(tk.END, cmd)
        if filtered:
            self.cmd_listbox.selection_clear(0, tk.END)

        # Position directly below input_entry. If we can't get coords yet, defer.
        try:
            x = self.input_entry.winfo_rootx()
            y = self.input_entry.winfo_rooty() + self.input_entry.winfo_height()
            width = max(self.input_entry.winfo_width(), 360)
            self.cmd_popup.geometry(f"{width}x180+{x}+{y}")
            self.cmd_popup.deiconify()
            self.cmd_popup.lift()
        except Exception:
            pass

    def _hide_cmd_popup(self):
        if self.cmd_popup is not None:
            try:
                self.cmd_popup.withdraw()
            except Exception:
                pass

    def _on_cmd_select(self, event=None):
        if self.cmd_listbox is None:
            return
        sel = self.cmd_listbox.curselection()
        if not sel:
            # If Return was pressed without a selection, pick the first item.
            if self.cmd_listbox.size() > 0:
                sel = (0,)
            else:
                return
        cmd = self.cmd_listbox.get(sel[0])
        self.input_entry.delete("1.0", tk.END)
        self.input_entry.insert("1.0", cmd + " ")
        self._hide_cmd_popup()
        self.input_entry.focus_set()

    def _list_all_commands(self):
        """Canonical list of slash commands shown in the autocomplete popup."""
        return [
            "/new",
            "/new <title>",
            "/threads",
            "/switch <id>",
            "/rename <id> <title>",
            "/delete <id>",
            "/clear",
            "/activate coder",
            "/deactivate coder",
            "/mode turbo",
            "/mode standard",
            "/mode coder",
            "/mode hibernate",
            "/sleep",
            "/hibernate",
            "/wake",
            "/resume",
            "/upload-global",
            "/run <script>",
            "/pin <user|assistant|list|unpin|unpin_all>",
            "/memories",
        ]

    # ------------------------------------------------------------------ #
    # Attachment chips (modern file input)
    # ------------------------------------------------------------------ #
    def _render_attachment_chips(self):
        """Rebuild the chip row from self.pending_attachments. No-ops if empty."""
        for w in self.chip_frame.winfo_children():
            w.destroy()
        if not self.pending_attachments:
            self.chip_frame.pack_forget()
            return
        self.chip_frame.pack(side="bottom", fill="x", before=self.input_frame)
        for path in self.pending_attachments:
            filename = os.path.basename(path)
            chip = tk.Frame(self.chip_frame, bg="#1e2030", bd=0)
            chip.pack(side="left", padx=(6, 0), pady=(6, 0))
            tk.Label(
                chip, text=f"📎 {filename}", bg="#1e2030", fg="#cdd6f4",
                font=("Segoe UI", 9), padx=8, pady=3,
            ).pack(side="left")
            x_btn = tk.Label(
                chip, text="×", bg="#1e2030", fg="#f38ba8",
                font=("Arial", 10, "bold"), padx=6, pady=3, cursor="hand2",
            )
            x_btn.pack(side="left")
            x_btn.bind("<Button-1>", lambda ev, p=path: self._remove_attachment(p))
            x_btn.bind("<B1-Motion>", lambda ev: "break")

    def _remove_attachment(self, path: str):
        if path in self.pending_attachments:
            self.pending_attachments.remove(path)
        self._render_attachment_chips()

    # ------------------------------------------------------------------ #
    # Human-in-the-loop approval bar (Yes / No for content pipeline)
    # ------------------------------------------------------------------ #
    def _build_approval_bar(self):
        """Creates a slim bar with Yes/No buttons above the input frame.
        Hidden by default; shown when the content pipeline emits an approval_request."""
        bar = tk.Frame(self.root, bg="#161b26", height=32)
        self._approval_bar = bar
        self.approval_label = tk.Label(
            bar, text="Is this draft good enough?", bg="#161b26", fg="#f9e2af",
            font=("Segoe UI", 10, "bold"),
        )
        self.approval_label.pack(side="left", padx=(10, 6), pady=4)
        self.approval_yes_btn = tk.Button(
            bar, text="\u2713 Yes", command=lambda: self._on_approval_decision(True),
            bg="#a6e3a1", fg="#1e1e2e", activebackground="#b6f3c1",
            bd=0, font=("Segoe UI", 10, "bold"), padx=10, cursor="hand2",
        )
        self.approval_yes_btn.pack(side="left", padx=2, pady=4)
        self.approval_no_btn = tk.Button(
            bar, text="\u2717 No (regenerate)", command=lambda: self._on_approval_decision(False),
            bg="#f38ba8", fg="#1e1e2e", activebackground="#f5a3c3",
            bd=0, font=("Segoe UI", 10, "bold"), padx=10, cursor="hand2",
        )
        self.approval_no_btn.pack(side="left", padx=2, pady=4)
        # Hidden until approval_request arrives.
        self._approval_bar_visible = False

    def _show_approval_bar(self, pass_num: int):
        if getattr(self, "_approval_bar_visible", False):
            return
        self.approval_label.config(text=f"Pass {pass_num}/3 — is this draft good enough?")
        # Pack just above the input frame so it stays anchored to the bottom.
        self._approval_bar.pack(side="bottom", fill="x", before=self.input_frame)
        self._approval_bar_visible = True

    def _hide_approval_bar(self):
        if not getattr(self, "_approval_bar_visible", False):
            return
        try:
            self._approval_bar.pack_forget()
        except Exception:
            pass
        self._approval_bar_visible = False

    def _on_approval_decision(self, approved: bool):
        # Runs on the main thread (Tk button callback).
        self._hide_approval_bar()
        if approved:
            self._append_text("\n[Human] Approved — finalizing draft.\n", "#a6e3a1")
        else:
            self._append_text("\n[Human] Rejected — regenerating next pass.\n", "#f9e2af")
        try:
            self.chatbot.submit_human_approval(approved)
        except Exception as e:
            self._append_text(f"[System Error] Failed to submit approval: {e}\n", MODE_COLORS["error"])

    def _send_command(self):
        prompt = self.input_entry.get("1.0", tk.END).strip()
        # Drain pending attachments up front so the UI clears even on early returns.
        pending = list(self.pending_attachments)
        self.pending_attachments.clear()
        self._render_attachment_chips()

        if not prompt and not pending:
            return

        # Ingest staged files synchronously so the LLM's RAG query sees them.
        if pending:
            print(f"[Send] Ingesting {len(pending)} attachment(s) before sending prompt...", flush=True)
            for path in pending:
                self._ingest_payload(path, is_raw_file=True)

        # No text but files attached → placeholder prompt so the model acknowledges.
        if not prompt:
            prompt = (
                f"I've attached {len(pending)} file(s) for context. "
                "Acknowledge and tell me what you see."
            )

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
            # Large paste: store as RAG chunks (quota-tracked per thread) and replace
            # the prompt with a small placeholder that points the LLM at the DB.
            # One LLM call extracts the actual question from the paste.
            import time as _time
            import uuid as _uuid
            source_label = f"PastedText_{int(_time.time())}_{_uuid.uuid4().hex[:6]}"
            thread_id = (
                self.chatbot._active_thread_id
                if self.chatbot and getattr(self.chatbot, "_active_thread_id", None)
                else None
            )
            if thread_id and getattr(self.chatbot, "_thread_files", None):
                try:
                    entry = self.chatbot._thread_files.add_text(thread_id, prompt, source_label)
                    evicted = entry.get("evicted", [])
                    print(
                        f"[Paste] {len(prompt)} chars → {entry['chunk_count']} chunks "
                        f"as '{source_label}' (thread {thread_id[:8]})",
                        flush=True,
                    )
                    if evicted:
                        print(f"[Quota] Evicted {len(evicted)} oldest upload(s): {evicted}", flush=True)
                except Exception as e:
                    print(f"[Paste Failure]: {e}", flush=True)
            else:
                # No active thread — fall back to raw RAG ingest with a global scope.
                self._ingest_payload(prompt, is_raw_file=False)
                print(f"[Paste] {len(prompt)} chars saved to global scope (no thread bound)", flush=True)

            print("[System] Isolating human instruction from raw data dump...", flush=True)
            extracted = self._extract_real_human_prompt(prompt)
            print(f"[System Extracted Command]: {extracted}", flush=True)
            prompt = (
                f"[The user pasted a large block of text ({len(prompt)} characters). "
                f"It has been stored in this thread's RAG index under the source label "
                f"'{source_label}'. Relevant context will be retrieved automatically "
                f"via RAG. Below is the extracted question / instruction.]\n\n"
                f"Extracted question / instruction:\n{extracted}"
            )
        elif os.path.exists(prompt) and os.path.isfile(prompt):
            supported_exts = {
                ".txt", ".md", ".pdf", ".docx", ".doc",
                ".py", ".js", ".jsx", ".ts", ".tsx",
                ".java", ".go", ".rs", ".rb", ".php", ".cs",
                ".cpp", ".c", ".h", ".hpp", ".swift", ".kt", ".scala",
                ".sh", ".bash", ".zsh", ".ps1",
                ".html", ".css", ".scss", ".xml", ".svg",
                ".json", ".yaml", ".yml", ".toml", ".csv", ".sql",
                ".log", ".zip",
            }
            ext = os.path.splitext(prompt)[1].lower()
            if ext in supported_exts:
                self._ingest_payload(prompt, is_raw_file=True)
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
            # If a thread command (/new, /switch, /rename, /delete, /clear, /threads)
            # was handled inside respond(), the active thread may have changed.
            # Refresh the tab strip + re-render so the visible log matches state.
            if isinstance(response_obj, str):
                self.root.after(0, self._refresh_thread_tabs)
                self.root.after(0, self._render_active_thread)
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
            elif event_type == "approval_request":
                # Pipeline is now blocked waiting on the approval event.
                # Show the Yes/No bar; the generator will block on its next
                # iteration because no further items will be queued until the
                # user clicks a button.
                try:
                    pass_num = int((chunk or {}).get("pass", 0))
                except Exception:
                    pass_num = 0
                self.root.after(0, lambda p=pass_num: self._show_approval_bar(p))
            elif event_type == "error":
                self._append_text(f"[System Error]: {chunk}\n", MODE_COLORS["error"])

        # Stream finished — make sure the bar is hidden.
        self.root.after(0, lambda: self._hide_approval_bar())

        if full_response.strip():
            self._append_text("\n", color="")

        return full_response

    # ------------------------------------------------------------------ #
    # Thread management — tab strip + active-thread switching
    # ------------------------------------------------------------------ #
    def _refresh_thread_tabs(self):
        """Destroys existing tab buttons and rebuilds from
        ``self.chatbot.list_threads()``. The active thread's tab is highlighted.
        Also keeps a per-thread close '×' button wired to deletion.
        """
        parent = getattr(self, "thread_tabs_frame", None)
        if parent is None:
            return

        # Destroy existing per-thread tab buttons (but keep the trailing '+').
        for tab in self.thread_tab_buttons.values():
            try:
                tab.destroy()
            except Exception:
                pass
        self.thread_tab_buttons = {}

        try:
            threads = self.chatbot.list_threads()
            active_id = self.chatbot.get_active_thread().get("id")
        except Exception as e:
            self._append_text(f"[System] Failed to list threads: {e}\n", MODE_COLORS["error"])
            return

        for thread in threads:
            tid = thread["id"]
            tab = self._build_single_thread_tab(
                parent, tid, thread.get("title", "New Thread"),
                is_active=(tid == active_id),
                on_switch=self._switch_thread,
                on_delete=self._delete_thread,
            )
            self.thread_tab_buttons[tid] = tab

    def _create_new_thread(self, title=None):
        """Creates a fresh thread via the chatbot, clears the chat log,
        refreshes the tab strip, and refocuses the input entry."""
        try:
            self.chatbot.create_thread(title)
        except Exception as e:
            self._append_text(f"[System] Failed to create thread: {e}\n", MODE_COLORS["error"])
            return
        self._clear_output()
        self._refresh_thread_tabs()
        self.input_entry.focus_set()

    def _switch_thread(self, thread_id):
        """Switches the active thread in the chatbot, clears the chat log,
        and re-renders the newly-active thread's history."""
        if not self.chatbot.switch_thread(thread_id):
            return
        self._clear_output()
        self._render_active_thread()
        self._refresh_thread_tabs()
        self.input_entry.focus_set()

    def _delete_thread(self, thread_id):
        """Deletes the chosen thread after a yes/no confirmation prompt.
        Falls back to whatever the chatbot now treats as active."""
        from tkinter import messagebox
        threads = self.chatbot.list_threads()
        if len(threads) <= 1:
            messagebox.showinfo(
                "Cannot delete",
                "At least one thread must remain open.",
                parent=self.root,
            )
            return
        target_title = next(
            (t.get("title", "New Thread") for t in threads if t["id"] == thread_id),
            "this thread",
        )
        ok = messagebox.askyesno(
            "Delete thread",
            f"Delete thread '{target_title}'? This cannot be undone.",
            parent=self.root,
        )
        if not ok:
            return
        if not self.chatbot.delete_thread(thread_id):
            return
        self._clear_output()
        self._refresh_thread_tabs()
        self._render_active_thread()
        self.input_entry.focus_set()

    def _render_active_thread(self):
        """Clears the chat log and replays the active thread's message history
        using the existing _append_text rendering helpers."""
        self._clear_output()
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        try:
            messages = self.chatbot.get_active_thread().get("messages", []) or []
        except Exception as e:
            self._append_text(f"[System] Failed to read active thread: {e}\n", MODE_COLORS["error"])
            return
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage):
                self._append_text(msg.content, "#e5e9f0")
            elif isinstance(msg, AIMessage):
                self._append_text(msg.content + "\n", "")
        if not self.user_scrolled_up:
            self.output_text.see("end")
        self._resize_to_fit_content()

    def _render_chat_history(self):
        """Backwards-compat alias used by _do_switch_mode.
        Re-renders from the active thread (the new source of truth)."""
        self._render_active_thread()

def main():
    root = tk.Tk()
    VedWidget(root)
    root.mainloop()
