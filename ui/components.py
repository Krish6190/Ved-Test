import tkinter as tk
import tkinter.font as tkfont
from .window_base import VedWindowBase

class VedComponentLayout(VedWindowBase):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.mode_buttons = {}
        self._drag_block_widgets = []

    def _build_ui_layout(self, mode_click_cb):
        # 1. Title Bar Component
        title_bar = tk.Frame(self.root, bg="#12131b", height=self.TITLE_BAR_H)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        self._bind_title_drag(title_bar)

        title_label = tk.Label(title_bar, text="  ● VED", bg="#12131b", fg="#e5e9f0", font=("ONE DAY", 10, "bold"), anchor="w")
        title_label.pack(side="left", padx=6)
        self._bind_title_drag(title_label)

        drag_area = tk.Frame(title_bar, bg="#12131b")
        drag_area.pack(fill="x", side="left", expand=True)
        self._bind_title_drag(drag_area)

        right_group = tk.Frame(title_bar, bg="#12131b")
        right_group.pack(side="right", padx=8)

        # 2. Render Hardware Buttons
        for icon, label, mode, color in [
            ("💻", "Coder",    "coder",     "#cba6f7"),
            ("⚡", "Turbo",    "turbo",     "#a6e3a1"),
            ("🧠", "Standard", "standard",  "#89b4fa"),
            ("😴", "Hibernate","hibernate", "#6c7086"),
        ]:
            btn = tk.Frame(right_group, bg="#161b26", bd=1, relief="flat", cursor="hand2")
            btn.pack(side="left", padx=4)

            em_lbl = tk.Label(btn, text=icon, bg="#161b26", fg="#e5e9f0", font=("Segoe UI Emoji", 10))
            em_lbl.pack(side="left", padx=(8, 2), pady=5)

            tx_lbl = tk.Label(btn, text=label, bg="#161b26", fg=color, font=("Segoe UI", 10, "bold"))
            tx_lbl.pack(side="left", padx=(0, 8), pady=5)

            for w in (btn, em_lbl, tx_lbl):
                w.bind("<Button-1>", lambda ev, m=mode: mode_click_cb(ev, m))
                w.bind("<B1-Motion>", lambda ev: "break")
                self._drag_block_widgets.append(w)
            self.mode_buttons[mode] = btn

        # 3. Window Close Utility
        cls_btn = tk.Button(right_group, text="✕", bg="#12131b", fg="#f38ba8", bd=0, activebackground="#12131b", activeforeground="#f38ba8", font=("ONE DAY", 11), command=self.root.destroy)
        cls_btn.pack(side="left", padx=6)
        cls_btn.bind("<B1-Motion>", lambda ev: "break")
        self._drag_block_widgets.append(cls_btn)

        # 4. Input Panel Component
        input_frame = tk.Frame(self.root, bg="#090a0f", height=self.INPUT_BAR_H)
        input_frame.pack(side="bottom", fill="x")
        input_frame.pack_propagate(False)

        self.input_entry = tk.Text(input_frame, bg="#313244", fg="#cdd6f4", insertbackground="white", bd=0, font=("Arial", 11), wrap="word", padx=8, pady=2)
        # 5. Output Canvas and Sticky Headers
        self.content_frame = tk.Frame(self.root, bg="#090a0f")
        self.content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(8, 4))
        self.line_height = tkfont.Font(font=("Arial", 12)).metrics("linespace")

        self.sticky_header_frame = tk.Frame(self.content_frame, bg="#090a0f")
        self.sticky_header_frame.pack(side="top", fill="x", pady=(0, 4))
        
        self.sticky_header_label = tk.Label(self.sticky_header_frame, text="", bg="#090a0f", fg="#a6e3a1", font=("Times", 12, "bold"), anchor="w")
        self.sticky_header_label.pack(side="top", fill="x")
        
        self.sticky_divider_label = tk.Label(self.sticky_header_frame, text="=============================================", bg="#090a0f", fg="#313244", anchor="w")
        self.sticky_divider_label.pack(side="top", fill="x")

        self.output_text = tk.Text(self.content_frame, bg="#090a0f", fg="#e5e9f0", font=("Times", 12), wrap="word", bd=0, highlightthickness=0, padx=4, pady=4, state="disabled", spacing1=3, spacing3=4, cursor="arrow")
        self.output_text.pack(side="left", fill="both", expand=True)

        self.output_scroll = tk.Scrollbar(self.content_frame, command=self.output_text.yview)
        self.output_scroll.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=self.output_scroll.set)
        
        return input_frame
