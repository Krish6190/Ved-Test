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

    def _build_ui_layout(self, mode_click_cb):
        self._build_title_bar(mode_click_cb)
        input_frame = self._build_input_bar()
        self._build_content_area()
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
            font=("ONE DAY", 10, "bold"), anchor="w"
        )
        title_label.pack(side="left", padx=6)
        self._bind_title_drag(title_label)

        drag_area = tk.Frame(title_bar, bg="#12131b")
        drag_area.pack(fill="x", side="left", expand=True)
        self._bind_title_drag(drag_area)

        right_group = tk.Frame(title_bar, bg="#12131b")
        right_group.pack(side="right", padx=8)

        self._build_mode_buttons(right_group, mode_click_cb)
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

    def _build_window_controls(self, parent):
        for text, fg, cmd in [
            ("—", "#f9e2af", self._safe_minimize),
            ("✕", "#f38ba8", self.root.destroy),
        ]:
            btn = tk.Button(
                parent, text=text, bg="#12131b", fg=fg, bd=0,
                activebackground="#12131b", activeforeground=fg,
                font=("ONE DAY", 11), command=cmd
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
    def _build_content_area(self):
        self.content_frame = tk.Frame(self.root, bg="#090a0f")
        self.content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(8, 4))

        self.line_height = tkfont.Font(font=("Arial", 12)).metrics("linespace")

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