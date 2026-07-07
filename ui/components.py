import os
import tkinter as tk
import tkinter.font as tkfont
from .window_base import VedWindowBase

MODE_COLORS = {
    "turbo":     "#a6e3a1",
    "standard":  "#89b4fa",
    "hibernate": "#6c7086",
    "coder":     "#cba6f7",
    "error":     "#f38ba8",
}

MODE_BUTTONS = [
    ("💻", "Coder",     "coder",     "#cba6f7"),
    ("⚡", "Turbo",     "turbo",     "#a6e3a1"),
    ("🧠", "Standard",  "standard",  "#89b4fa"),
    ("😴", "Hibernate", "hibernate", "#6c7086"),
]

class VedComponentLayout(VedWindowBase):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.mode_buttons      = {}
        self._drag_block_widgets = []
        self.thread_tab_buttons = {}
        self.index_status_label = None  # "Project indexed: N files" chip
        self.cwd_label = None  # "current directory" chip in title bar

    def _build_ui_layout(self, mode_click_cb, thread_callbacks=None):
        self._build_title_bar(mode_click_cb)
        self._build_cwd_bar()
        input_frame = self._build_input_bar()
        self._build_content_area(thread_callbacks=thread_callbacks)
        return input_frame
    # ------------------------------------------------------------------ #
    # Title bar
    # ------------------------------------------------------------------ #
    def _build_title_bar(self, mode_click_cb):
        title_bar = tk.Frame(self.root, bg="#12131b", height=self.TITLE_BAR_H)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        self._bind_title_drag(title_bar)

        title_label = tk.Label(
            title_bar, text="  ● VED", bg="#12131b", fg="#e5e9f0",
            font=("Arial", 10, "bold"), anchor="w"
        )
        title_label.pack(side="left", padx=6)
        self._bind_title_drag(title_label)

        drag_area = tk.Frame(title_bar, bg="#12131b")
        drag_area.pack(fill="x", side="left", expand=True)
        self._bind_title_drag(drag_area)

        right_group = tk.Frame(title_bar, bg="#12131b")
        right_group.pack(side="right", padx=8)

        self._build_mode_buttons(right_group, mode_click_cb)
        self._build_index_status(right_group)
        self._build_window_controls(right_group)

    def _build_mode_buttons(self, parent, mode_click_cb):
        for icon, label, mode, color in MODE_BUTTONS:
            btn = tk.Frame(parent, bg="#161b26", bd=1, relief="flat", cursor="hand2")
            btn.pack(side="left", padx=4)

            em_lbl = tk.Label(btn, text=icon, bg="#161b26", fg="#e5e9f0",
                              font=("Segoe UI Emoji", 10))
            em_lbl.pack(side="left", padx=(8, 2), pady=5)

            tx_lbl = tk.Label(btn, text=label, bg="#161b26", fg=color,
                              font=("Segoe UI", 10, "bold"))
            tx_lbl.pack(side="left", padx=(0, 8), pady=5)

            for w in (btn, em_lbl, tx_lbl):
                w.bind("<Button-1>", lambda ev, m=mode: mode_click_cb(ev, m))
                w.bind("<B1-Motion>", lambda ev: "break")
                self._drag_block_widgets.append(w)

            self.mode_buttons[mode] = btn

    def _build_index_status(self, parent):
        """Small status chip showing project-RAG indexing state.

        States (text + color):
          - "Indexing..." (yellow) while background thread runs
          - "Indexed: N files" (green) when complete
          - "Idle" (gray) initially or when index is fresh
          - "Index error" (red) on failure

        Wired from chatbot.on_session_start via set_index_status().
        """
        frame = tk.Frame(parent, bg="#12131b")
        frame.pack(side="left", padx=4)
        self.index_status_label = tk.Label(
            frame, text="📂 Idle", bg="#12131b", fg="#6c7086",
            font=("Segoe UI", 9), padx=4,
        )
        self.index_status_label.pack(side="left")

    def _build_cwd_bar(self):
        """Thin bar between the title bar and the content/threads area.

        Houses the cwd chip ("📁 <path>") so it doesn't crowd the title
        bar's mode buttons and has room for longer paths. Sits between
        the title bar (above) and the content area (below).

        Pack order: _build_ui_layout calls _build_title_bar first (packs
        to top), then _build_cwd_bar (packs to top, appears below the
        title bar naturally), then _build_content_area (expands below).
        """
        bar = tk.Frame(self.root, bg="#0f1018", height=22)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        self._build_cwd_display(bar)

    def _build_cwd_display(self, parent):
        """Current-working-directory chip in its own bar (below the title).

        Shows "📁 <cwd>" with a muted foreground. Updates when the
        chatbot changes directory via the /cd command. The chip sits in
        its own thin bar so long paths don't push the title bar's mode
        buttons around.
        """
        try:
            initial = os.getcwd()
        except Exception:
            initial = "(unknown)"
        self.cwd_label = tk.Label(
            parent, text=f"📁 {initial}", bg="#0f1018", fg="#a6adc8",
            font=("Segoe UI", 9), padx=8, anchor="w",
        )
        self.cwd_label.pack(side="left", padx=4, fill="x", expand=True)
        self._drag_block_widgets.append(self.cwd_label)

    def set_current_directory(self, path: str) -> None:
        """Update the cwd chip. Safe to call before UI build."""
        if self.cwd_label is None:
            return
        try:
            self.cwd_label.config(text=f"📁 {path}")
        except Exception:
            pass

    def set_index_status(self, text: str, color: str = "#a6e3a1") -> None:
        """Update the project-index status chip. Safe to call before UI build."""
        if self.index_status_label is None:
            return
        try:
            self.index_status_label.config(text=text, fg=color)
        except Exception:
            pass

    def _build_window_controls(self, parent):
        for text, fg, cmd in [
            ("—", "#f9e2af", self._safe_minimize),
            ("✕", "#f38ba8", self.root.destroy),
        ]:
            btn = tk.Button(
                parent, text=text, bg="#12131b", fg=fg, bd=0,
                activebackground="#12131b", activeforeground=fg,
                font=("Arial", 11), command=cmd
            )
            btn.pack(side="left", padx=(2 if text == "—" else 6))
            btn.bind("<B1-Motion>", lambda ev: "break")
            self._drag_block_widgets.append(btn)
    # ------------------------------------------------------------------ #
    # Input bar
    # ------------------------------------------------------------------ #
    def _build_input_bar(self):
        input_frame = tk.Frame(self.root, bg="#090a0f", height=self.INPUT_BAR_H)
        input_frame.pack(side="bottom", fill="x")
        input_frame.pack_propagate(False)
        self.upload_btn = tk.Button(
            input_frame, text="＋", bg="#161b26", fg="#cba6f7",
            activebackground="#313244", activeforeground="#cba6f7",
            bd=0, font=("Arial", 14, "bold"), width=3, cursor="hand2"
        )
        self.upload_btn.pack(side="left", fill="y", padx=(6, 2), pady=6)
        self.upload_btn.bind("<B1-Motion>", lambda ev: "break")
        self._drag_block_widgets.append(self.upload_btn)
        self.input_entry = tk.Text(
            input_frame, bg="#313244", fg="#cdd6f4",
            insertbackground="white", bd=0,
            font=("Arial", 11), wrap="word", padx=8, pady=6
        )
        return input_frame
    # ------------------------------------------------------------------ #
    # Content area (header + output text + scrollbar)
    # ------------------------------------------------------------------ #
    def _build_content_area(self, thread_callbacks=None):
        self.content_frame = tk.Frame(self.root, bg="#090a0f")
        self.content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(8, 4))

        self.line_height = tkfont.Font(font=("Arial", 12)).metrics("linespace")

        if thread_callbacks is not None:
            self._build_thread_tabs(
                self.content_frame,
                thread_callbacks["on_switch"],
                thread_callbacks["on_new"],
            )

        header_frame = tk.Frame(self.content_frame, bg="#090a0f")
        header_frame.pack(side="top", fill="x", pady=(0, 4))

        self.sticky_header_label = tk.Label(
            header_frame, text="", bg="#090a0f", fg="#a6e3a1",
            font=("Times", 12, "bold"), anchor="w"
        )
        self.sticky_header_label.pack(side="top", fill="x")

        tk.Label(
            header_frame,
            text="=============================================",
            bg="#090a0f", fg="#313244", anchor="w"
        ).pack(side="top", fill="x")

        self.output_text = tk.Text(
            self.content_frame, bg="#090a0f", fg="#e5e9f0",
            font=("Times", 12), wrap="word", bd=0, highlightthickness=0,
            padx=4, pady=4, state="normal",
            spacing1=3, spacing3=4, cursor="xterm"
        )
        self.output_text.pack(side="left", fill="both", expand=True)

        self.output_scroll = tk.Scrollbar(self.content_frame, command=self.output_text.yview)
        self.output_scroll.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=self.output_scroll.set)
    # ------------------------------------------------------------------ #
    # Thread tab strip (populated by GUI's _refresh_thread_tabs)
    # ------------------------------------------------------------------ #
    def _build_thread_tabs(self, tabs_parent, on_thread_click, on_new_thread_click):
        """Builds the thread-tab strip container plus the trailing '+' button.
        The per-thread pill buttons are added later by the GUI via
        _refresh_thread_tabs, which also stores them in self.thread_tab_buttons.
        """
        self.thread_tabs_frame = tk.Frame(tabs_parent, bg="#090a0f")
        self.thread_tabs_frame.pack(side="top", fill="x", pady=(0, 4))
        self.thread_tab_buttons = {}

        new_btn = tk.Label(
            self.thread_tabs_frame, text="＋", bg="#1e2030", fg="#a6e3a1",
            font=("Arial", 12, "bold"), padx=8, pady=2, cursor="hand2", bd=0
        )
        new_btn.pack(side="right", padx=(4, 0))
        new_btn.bind("<Button-1>", lambda ev, cb=on_new_thread_click: cb())
        new_btn.bind("<B1-Motion>", lambda ev: "break")
        self._drag_block_widgets.append(new_btn)

    def _build_single_thread_tab(self, parent, thread_id, title, is_active,
                                 on_switch, on_delete):
        """Creates one pill-style thread tab (title + close '×') inside `parent`.
        Returns the tab Frame; caller stores it in self.thread_tab_buttons."""
        bg = "#313244" if is_active else "#161b26"
        fg = "#e5e9f0" if is_active else "#a6adc8"
        tab = tk.Frame(parent, bg=bg, bd=0, cursor="hand2")
        tab.pack(side="left", padx=(0, 4))

        display_title = (title or "New Thread")[:18]
        title_lbl = tk.Label(
            tab, text=display_title, bg=bg, fg=fg,
            font=("Segoe UI", 9, "bold"), padx=8, pady=2
        )
        title_lbl.pack(side="left")

        close_lbl = tk.Label(
            tab, text="×", bg=bg, fg=("#f38ba8" if is_active else "#6c7086"),
            font=("Arial", 10, "bold"), padx=4, pady=2, cursor="hand2"
        )
        close_lbl.pack(side="left")

        for w in (tab, title_lbl, close_lbl):
            w.bind("<B1-Motion>", lambda ev: "break")
            self._drag_block_widgets.append(w)

        def _switch(ev, tid=thread_id):
            if on_switch:
                on_switch(tid)
        tab.bind("<Button-1>", _switch)
        title_lbl.bind("<Button-1>", _switch)

        def _delete(ev, tid=thread_id):
            if on_delete:
                on_delete(tid)
        close_lbl.bind("<Button-1>", _delete)

        return tab