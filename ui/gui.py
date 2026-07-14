import os
import threading
import tkinter as tk
import winsound
from voice.voice_module import VoiceSystem
from chatbot import Chatbot
from graph.tools.staging_registry import STAGING_REGISTRY
from .gui_rag_worker import VedRagWorker
from .components import MODE_COLORS

# Telemetry — registers an active-user session on GUI startup, sends
try:
    from telemetry import telemetry as _telemetry
except Exception:  # pragma: no cover - import-failure fallback
    _telemetry = None

MODE_COMMANDS = {"/activate coder", "/deactivate coder", "/sleep", "/hibernate", "/wake", "/resume"}

class VedWidget(VedRagWorker):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.chatbot = Chatbot()
        # Wire the UI components into the chatbot so command_processor._handle_cd()
        try:
            self.chatbot.set_ui_components(self)
        except Exception:
            pass
        # Telemetry: register this GUI window as an active session.
        self._telemetry_session_id: str | None = None
        try:
            if _telemetry is not None:
                self._telemetry_session_id = _telemetry.start_session(
                    username=os.getenv("VED_USERNAME", os.getenv("USERNAME", "anonymous")),
                    source="gui",
                    mode=self.chatbot.mode,
                    meta={"pid": os.getpid()},
                )
        except Exception:
            pass
        self.current_mode = self.chatbot.mode
        self.chat_history = []
        self.is_generating = False
        # Modern file-input: staged attachments shown as chips above the input row.
        self.pending_attachments = []
        self.chip_frame = tk.Frame(self.root, bg="#090a0f")

        self.thread_callbacks = {
            "on_new": self._create_new_thread,
            "on_switch": self._switch_thread,
        }
        input_frame = self._build_ui_layout(self._on_mode_click, thread_callbacks=self.thread_callbacks)
        self.input_frame = input_frame
        self._build_approval_bar()
        self.voice  = VoiceSystem(self.root, input_frame, self.input_entry, self._send_command)
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
            "/run",
            "/pin",
            "/unpin <n>",
            "/unpin_all",
            "/list",
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
        self._approval_bar_visible = False
        self._pending_approval_kind = None

    def _show_approval_bar(self, pass_num: int):
        if getattr(self, "_approval_bar_visible", False):
            return
        self._pending_approval_kind = "content"
        self.approval_label.config(text=f"Pass {pass_num}/3 — is this draft good enough?")
        self.approval_no_btn.config(text="\u2717 No (regenerate)")
        # Pack just above the input frame so it stays anchored to the bottom.
        self._approval_bar.pack(side="bottom", fill="x", before=self.input_frame)
        self._approval_bar_visible = True

    def _show_plan_approval_bar(self, num_steps: int):
        """Variant of _show_approval_bar used by the planner-executor
        pipeline (Path A). Reuses the same Yes/No bar but updates the
        prompt text and the No button so it reads naturally for a plan
        decision rather than a content regeneration."""
        if getattr(self, "_approval_bar_visible", False):
            return
        self._pending_approval_kind = "plan"
        self.approval_label.config(
            text=f"Planner proposes {num_steps} step(s) — approve this plan?"
        )
        self.approval_no_btn.config(text="\u2717 No (reject)")
        self._approval_bar.pack(side="bottom", fill="x", before=self.input_frame)
        self._approval_bar_visible = True

    # ------------------------------------------------------------------ #
    # Cumulative multi-file file-edit review panel (Cursor-style)
    # ------------------------------------------------------------------ #
    def _build_file_edit_review_panel(self):
        """Create a persistent review panel for pending file edits.

        The panel is a Toplevel window that accumulates edits across
        multiple files. It displays a file list on the left and a diff
        preview on the right. The user can approve/reject all files at
        once or act on individual files.
        """
        panel = tk.Toplevel(self.root)
        panel.title("Review file edits")
        panel.configure(bg="#161b26")
        panel.geometry("700x450")
        panel.protocol("WM_DELETE_WINDOW", lambda: None)  # disable close
        panel.transient(self.root)
        panel.attributes("-topmost", True)

        header = tk.Frame(panel, bg="#161b26", height=36)
        header.pack(side="top", fill="x")
        tk.Label(
            header, text="Pending file edits", bg="#161b26", fg="#cdd6f4",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=10, pady=6)

        btn_frame = tk.Frame(header, bg="#161b26")
        btn_frame.pack(side="right", padx=10, pady=4)
        tk.Button(
            btn_frame, text="Approve all", command=self._on_file_edit_approve_all,
            bg="#a6e3a1", fg="#1e1e2e", activebackground="#b6f3c1",
            bd=0, font=("Segoe UI", 9, "bold"), padx=10, cursor="hand2",
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame, text="Reject all", command=self._on_file_edit_reject_all,
            bg="#f38ba8", fg="#1e1e2e", activebackground="#f5a3c3",
            bd=0, font=("Segoe UI", 9, "bold"), padx=10, cursor="hand2",
        ).pack(side="left", padx=2)

        body = tk.Frame(panel, bg="#161b26")
        body.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        # Left: scrollable file list + per-file actions
        list_frame = tk.Frame(body, bg="#161b26", width=220)
        list_frame.pack(side="left", fill="y")
        list_frame.pack_propagate(False)
        self._file_edit_listbox = tk.Listbox(
            list_frame, bg="#1e2030", fg="#cdd6f4", selectbackground="#313244",
            selectforeground="#cdd6f4", highlightthickness=0, bd=0,
            font=("Segoe UI", 10), activestyle="none",
        )
        self._file_edit_listbox.pack(side="top", fill="both", expand=True)
        list_scroll = tk.Scrollbar(list_frame, command=self._file_edit_listbox.yview, bg="#1e2030")
        list_scroll.pack(side="right", fill="y")
        self._file_edit_listbox.config(yscrollcommand=list_scroll.set)
        self._file_edit_listbox.bind("<<ListboxSelect>>", self._on_file_edit_selected)
        per_file_btns = tk.Frame(list_frame, bg="#161b26")
        per_file_btns.pack(side="bottom", fill="x", pady=(6, 0))
        tk.Button(
            per_file_btns, text="Approve", command=self._on_file_edit_approve_selected,
            bg="#a6e3a1", fg="#1e1e2e", activebackground="#b6f3c1",
            bd=0, font=("Segoe UI", 8, "bold"), padx=6, cursor="hand2",
        ).pack(side="left", padx=(0, 2), expand=True, fill="x")
        tk.Button(
            per_file_btns, text="Reject", command=self._on_file_edit_reject_selected,
            bg="#f38ba8", fg="#1e1e2e", activebackground="#f5a3c3",
            bd=0, font=("Segoe UI", 8, "bold"), padx=6, cursor="hand2",
        ).pack(side="left", padx=(2, 0), expand=True, fill="x")

        # Right: diff preview
        diff_frame = tk.Frame(body, bg="#161b26")
        diff_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._file_edit_diff_label = tk.Label(
            diff_frame, text="Select a file to review", bg="#1e2030", fg="#6c7086",
            font=("Segoe UI", 10), anchor="nw", justify="left",
        )
        self._file_edit_diff_label.pack(side="top", fill="x", pady=(0, 4))
        self._file_edit_diff_text = tk.Text(
            diff_frame, bg="#0b0c15", fg="#cdd6f4", wrap="word",
            font=("Consolas", 9), highlightthickness=0, bd=0, padx=8, pady=8,
            state="disabled",
        )
        self._file_edit_diff_text.pack(side="top", fill="both", expand=True)
        diff_scroll = tk.Scrollbar(diff_frame, command=self._file_edit_diff_text.yview, bg="#1e2030")
        diff_scroll.pack(side="right", fill="y")
        self._file_edit_diff_text.config(yscrollcommand=diff_scroll.set)

        self._file_edit_review_panel = panel
        self._file_edit_pending_tasks: dict = {}

    def _update_file_edit_review_panel(self, tasks: dict):
        """Refresh the file list from the latest pending tasks snapshot."""
        if not getattr(self, "_file_edit_review_panel", None) or not self._file_edit_review_panel.winfo_exists():
            self._build_file_edit_review_panel()
        self._file_edit_pending_tasks = dict(tasks)
        selected_idx = self._file_edit_listbox.curselection()
        selected_path = ""
        if selected_idx:
            selected_path = self._file_edit_listbox.get(selected_idx[0])
        self._file_edit_listbox.delete(0, tk.END)
        for path in sorted(tasks.keys()):
            self._file_edit_listbox.insert(tk.END, path)
        # Try to restore selection.
        if selected_path:
            for i in range(self._file_edit_listbox.size()):
                if self._file_edit_listbox.get(i) == selected_path:
                    self._file_edit_listbox.selection_set(i)
                    break
        if not self._file_edit_listbox.curselection() and self._file_edit_listbox.size() > 0:
            self._file_edit_listbox.selection_set(0)
        self._on_file_edit_selected()
        self._file_edit_review_panel.deiconify()
        self._file_edit_review_panel.lift()

    def _on_file_edit_selected(self, event=None):
        """Render the diff for the currently selected pending file."""
        if not getattr(self, "_file_edit_diff_text", None):
            return
        sel = self._file_edit_listbox.curselection()
        self._file_edit_diff_text.config(state="normal")
        self._file_edit_diff_text.delete("1.0", tk.END)
        self._file_edit_diff_label.config(text="Select a file to review")
        if not sel:
            self._file_edit_diff_text.config(state="disabled")
            return
        path = self._file_edit_listbox.get(sel[0])
        task = self._file_edit_pending_tasks.get(path, {})
        preview = task.get("preview", {}) or {}
        operation = task.get("tool_name", "edit")
        old_snippet = str(preview.get("old", "") or "")
        new_snippet = str(preview.get("new", "") or "")
        self._file_edit_diff_label.config(text=f"{path} ({operation})")
        lines = [f"Path: {path}", f"Operation: {operation}", "", "--- old ---", old_snippet, "", "--- new ---", new_snippet]
        self._file_edit_diff_text.insert("1.0", "\n".join(lines))
        self._file_edit_diff_text.config(state="disabled")

    def _get_current_staged_paths(self) -> list:
        """Return the authoritative set of paths currently staged for approval."""
        paths = []
        thread_id = getattr(self.chatbot, "_file_edit_thread_id", None)
        if thread_id:
            try:
                paths = list(STAGING_REGISTRY.get_tasks(thread_id).keys())
            except Exception:
                paths = []
        if not paths:
            paths = list(self._file_edit_pending_tasks.keys())
        if not paths and getattr(self, "_file_edit_listbox", None):
            paths = [self._file_edit_listbox.get(i) for i in range(self._file_edit_listbox.size())]
        return paths
 
    def _submit_file_edit_decision(self, action: str, paths: list | None = None):
        """Send a file-edit approval decision to the chatbot and keep the review panel open until the backend is notified."""
        if paths is None and action in ("approve_all", "reject_all"):
            paths = self._get_current_staged_paths()
        decision = {"action": action, "paths": list(paths) if paths else []}
        try:
            self.chatbot.submit_file_edit_approval(decision)
        except Exception as e:
            self._append_text(f"[System Error] Failed to submit file edit approval: {e}\n", MODE_COLORS["error"])
        # The worker will remove approved/rejected entries; refresh on next event.

    def _on_file_edit_approve_all(self):
        # Try to eagerly drain the staging registry for the active session
        # so the worker doesn't hang waiting for the UI. Do this in a
        # background thread to avoid blocking the Tk mainloop.
        decision = {"action": "approve_all", "paths": []}
        try:
            thread_id = getattr(self.chatbot, "_file_edit_thread_id", None)
            if thread_id and STAGING_REGISTRY.has_session(thread_id):
                def _drain_apply():
                    try:
                        STAGING_REGISTRY.apply_decision(
                            thread_id,
                            decision,
                            apply_callback=self.chatbot._apply_file_edit_task,
                        )
                    except Exception:
                        # Fallback to the existing submit path if direct apply fails
                        try:
                            self.chatbot.submit_file_edit_approval(decision)
                        except Exception:
                            pass
                    finally:
                        # Ensure the review panel closes on the main thread.
                        try:
                            self.root.after(0, self._force_close_review_panel)
                        except Exception:
                            pass
                threading.Thread(target=_drain_apply, daemon=True).start()
            else:
                self._submit_file_edit_decision("approve_all")
                self._maybe_close_review_panel()
        except Exception:
            # Best-effort: fallback to original flow.
            self._submit_file_edit_decision("approve_all")
            self._maybe_close_review_panel()

    def _on_file_edit_reject_all(self):
        # Mirror approve_all: ensure the registry is drained and the
        # review panel is closed even if the worker thread hasn't yet
        # processed the decision.
        decision = {"action": "reject_all", "paths": []}
        try:
            thread_id = getattr(self.chatbot, "_file_edit_thread_id", None)
            if thread_id and STAGING_REGISTRY.has_session(thread_id):
                def _drain_reject():
                    try:
                        STAGING_REGISTRY.apply_decision(
                            thread_id,
                            decision,
                            apply_callback=lambda task: "rejected",
                        )
                    except Exception:
                        try:
                            self.chatbot.submit_file_edit_approval(decision)
                        except Exception:
                            pass
                    finally:
                        try:
                            self.root.after(0, self._force_close_review_panel)
                        except Exception:
                            pass
                threading.Thread(target=_drain_reject, daemon=True).start()
            else:
                self._submit_file_edit_decision("reject_all")
                self._maybe_close_review_panel()
        except Exception:
            self._submit_file_edit_decision("reject_all")
            self._maybe_close_review_panel()

    def _on_file_edit_approve_selected(self):
        sel = self._file_edit_listbox.curselection()
        if not sel:
            return
        paths = [self._file_edit_listbox.get(i) for i in sel]
        self._submit_file_edit_decision("approve", paths)
        self._remove_listbox_paths(paths)
        self._maybe_close_review_panel()

    def _on_file_edit_reject_selected(self):
        sel = self._file_edit_listbox.curselection()
        if not sel:
            return
        paths = [self._file_edit_listbox.get(i) for i in sel]
        self._submit_file_edit_decision("reject", paths)
        self._remove_listbox_paths(paths)
        self._maybe_close_review_panel()

    def _remove_listbox_paths(self, paths: list):
        """Delete specific paths from the review listbox.

        Iterates in reverse index order so deleting earlier items does
        not shift the indices of later items. Any path not currently in
        the listbox is ignored (idempotent).
        """
        if not paths or self._file_edit_listbox is None:
            return
        # Build a map of path -> index so we can delete in reverse order.
        size = self._file_edit_listbox.size()
        path_to_idx = {}
        for i in range(size):
            path_to_idx[self._file_edit_listbox.get(i)] = i
        indices_to_delete = sorted(
            (path_to_idx[p] for p in paths if p in path_to_idx),
            reverse=True,
        )
        for idx in indices_to_delete:
            try:
                self._file_edit_listbox.delete(idx)
            except Exception:
                pass
        # Also drop the acted-on items from the widget's pending dict so
        # the UI state stays consistent with the listbox.
        for p in paths:
            self._file_edit_pending_tasks.pop(p, None)

    def _maybe_close_review_panel(self):
        """Close the review panel ONLY when the pending queue is empty.

        Mirrors the worker thread's drain: after apply_decision returns,
        session.tasks is cleared (or shrunk). The GUI listbox is refreshed
        by the next file_edit_approval_request payload, but in the
        per-file flow we have to check the current listbox size here.

        The withdraw() is called via root.after(0, ...) so it runs on the
        Tk main thread -- the button click handler itself is already on
        the main thread, but the worker thread may also call into this
        path via _update_file_edit_review_panel, and that runs on the
        main thread anyway because it is scheduled through root.after.
        """
        panel = getattr(self, "_file_edit_review_panel", None)
        if panel is None:
            return
        try:
            listbox = getattr(self, "_file_edit_listbox", None)
            remaining = listbox.size() if listbox is not None else 0
            if remaining > 0:
                try:
                    panel.deiconify()
                    panel.lift()
                except Exception:
                    pass
                return
            if panel.winfo_exists():
                panel.withdraw()
        except Exception:
            pass

    def _force_close_review_panel(self):
        """Forcefully close and destroy the review panel (main-thread only).

        This is used after an approve/reject-all drain to ensure the UI
        does not remain frozen awaiting worker callbacks.
        """
        panel = getattr(self, "_file_edit_review_panel", None)
        try:
            if panel is None:
                return
            if panel.winfo_exists():
                try:
                    panel.withdraw()
                except Exception:
                    pass
                try:
                    panel.destroy()
                except Exception:
                    pass
        except Exception:
            pass

    def _show_file_edit_approval_bar(self, tasks: dict):
        """Show or refresh the cumulative multi-file file-edit review panel.

        The old single-file bar has been replaced by a Cursor-style review
        panel that accumulates edits across files. This method is kept as a
        thin wrapper for backwards compatibility with _consume_response.
        """
        self._update_file_edit_review_panel(tasks)

    def _hide_approval_bar(self):
        if not getattr(self, "_approval_bar_visible", False):
            return
        try:
            self._approval_bar.pack_forget()
        except Exception:
            pass
        self._approval_bar_visible = False
        self._pending_approval_kind = None

    def _on_approval_decision(self, approved: bool):
        # Runs on the main thread (Tk button callback).
        kind = getattr(self, "_pending_approval_kind", None) or "content"
        self._hide_approval_bar()
        if kind == "plan":
            if approved:
                self._append_text("\n[Human] Approved plan\n", "#a6e3a1")
            else:
                self._append_text("\n[Human] Rejected plan\n", "#f9e2af")
            try:
                self.chatbot.submit_plan_approval(approved)
            except Exception as e:
                self._append_text(f"[System Error] Failed to submit plan approval: {e}\n", MODE_COLORS["error"])
            return
        if kind == "file_edit":
            if approved:
                self._append_text("\n[Human] Approved file edit\n", "#a6e3a1")
            else:
                self._append_text("\n[Human] Rejected file edit\n", "#f9e2af")
            try:
                self.chatbot.submit_file_edit_approval(approved)
            except Exception as e:
                self._append_text(f"[System Error] Failed to submit file edit approval: {e}\n", MODE_COLORS["error"])
            return
        # Default: content-pipeline approval flow (preserves prior behavior).
        if approved:
            self._append_text("\n[Human] Approved — finalizing draft.\n", "#a6e3a1")
        else:
            self._append_text("\n[Human] Rejected — regenerating next pass.\n", "#f9e2af")
        try:
            self.chatbot.submit_human_approval(approved)
        except Exception as e:
            self._append_text(f"[System Error] Failed to submit approval: {e}\n", MODE_COLORS["error"])

    def _send_command(self, prompt=None):
        # Chunk C: an optional explicit prompt arg lets the voice processor
        if prompt is None:
            prompt = self.input_entry.get("1.0", tk.END).strip()
        # Chunk C: interrupt any in-flight bot-response speech so the new
        voice = getattr(self, "voice", None)
        if voice is not None and hasattr(voice, "stop_tts"):
            try:
                voice.stop_tts()
            except Exception:
                pass

        # Drain pending attachments up front so the UI clears even on early returns.
        pending = list(self.pending_attachments)
        self.pending_attachments.clear()
        self._render_attachment_chips()

        if not prompt and not pending:
            return
        try:
            if _telemetry is not None and getattr(self, "_telemetry_session_id", None):
                _telemetry.heartbeat(session_id=self._telemetry_session_id)
        except Exception:
            pass

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
                    entry = self.chatbot._thread_files.add_text(thread_id, prompt, source_label, chunker=self.chatbot._rag_chunker())
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
            if isinstance(response_obj, str):
                self.root.after(0, self._refresh_thread_tabs)
                self.root.after(0, self._render_active_thread)
            return full_response
        except Exception as e:
            self._append_text(f"\nChatbot error: {e}\n", MODE_COLORS["error"])
            return ""
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
                # Only drop completely empty tokens. Whitespace-only
                # tokens ("\n", "\n\n", "  ") MUST pass through so the
                # chat panel renders standard paragraph breaks. Runaway
                # whitespace (>3 newlines, >4 spaces) is handled by the
                # producer-side _clean_chunk and the UI renderer's
                # defence-in-depth check.
                if not isinstance(chunk, str) or not chunk:
                    continue
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
            elif event_type == "plan_approval_request":
                # Planner proposed a plan (Path A). Render the proposed steps
                # in the chat area so the user can read them before deciding,
                # then show the approval bar wired to submit_plan_approval.
                chunks = []
                if isinstance(chunk, dict):
                    raw_chunks = chunk.get("chunks", [])
                    if isinstance(raw_chunks, (list, tuple)):
                        chunks = [str(c) for c in raw_chunks]
                # Render the proposed plan as a simple formatted block.
                rendered_lines = ["", "[Planner] Proposed plan:"]
                if chunks:
                    for idx, step in enumerate(chunks, start=1):
                        rendered_lines.append(f"  {idx}. {step}")
                else:
                    rendered_lines.append("  (no steps provided)")
                rendered_lines.append("")
                block = "\n".join(rendered_lines) + "\n"
                self._append_text(block, "#cdd6f4")
                self.root.after(
                    0,
                    lambda n=len(chunks): self._show_plan_approval_bar(n),
                )
            elif event_type == "file_edit_approval_request":
                payload = chunk if isinstance(chunk, dict) else {}
                tasks = payload.get("tasks") or {}
                if not tasks and payload.get("path"):
                    # Backwards-compat for single-file payloads.
                    tasks = {payload["path"]: payload}
                operation = str(payload.get("operation", "edit"))
                path = str(payload.get("path", ""))
                filename = os.path.basename(path) if path else "<unknown>"
                self._append_text(
                    f"\n[Planner] Requesting approval to {operation} {filename} "
                    f"({len(tasks)} pending file edit(s))\n",
                    "#cdd6f4",
                )
                self.root.after(
                    0,
                    lambda t=tasks: self._show_file_edit_approval_bar(t),
                )
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
    widget = VedWidget(root)

    # Telemetry: end the active-user session when the window is closed
    # (either via WM_DELETE_WINDOW or a normal mainloop exit). Bound
    # before mainloop so the close handler always fires.
    def _on_close():
        try:
            if _telemetry is not None:
                sid = getattr(widget, "_telemetry_session_id", None)
                if sid:
                    _telemetry.end_session(session_id=sid)
                _telemetry.shutdown()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", _on_close)
    try:
        root.mainloop()
    finally:
        # Belt-and-braces: if mainloop exits via something other than
        # WM_DELETE_WINDOW (e.g. root.quit()), still close the session.
        try:
            if _telemetry is not None:
                sid = getattr(widget, "_telemetry_session_id", None)
                if sid:
                    _telemetry.end_session(session_id=sid)
                _telemetry.shutdown()
        except Exception:
            pass
