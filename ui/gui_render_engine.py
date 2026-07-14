import re
import tkinter as tk
from .components import VedComponentLayout
from graph.nodes._stream_helpers import (
    _RUNAWAY_NEWLINES_RE,
    _RUNAWAY_SPACES_RE,
    _strip_leading_blank_lines,
)

class VedGuiRenderEngine(VedComponentLayout):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.user_scrolled_up = False
        self.token_buffer = ""
        self.root.after(15, self._setup_render_engine_styles)

    def _setup_render_engine_styles(self):
        """Maps layout formatting styles cleanly onto the main text view log panel."""
        try:
            self.output_text.configure(state="normal", cursor="xterm")
            self.output_text.configure(selectbackground="#313244", selectforeground="#cdd6f4")
            
            # --- FIXED VED MESSAGES: REMOVED ALL THE EXTREME PADDING ---
            self.output_text.tag_configure("ved_msg", 
                foreground="#a6e3a1", 
                justify="left", 
                lmargin1=10, lmargin2=10, 
                rmargin=20, # Allowed text to expand across the full window width safely
                spacing1=6, spacing3=6
            )

            # Standard Markdown sub-tags
            self.output_text.tag_configure("md_bold", font=("Times", 12, "bold"))
            self.output_text.tag_configure("md_italic", font=("Times", 12, "italic"))
            self.output_text.tag_configure("md_underline", font=("Times", 12, "underline"))
            self.output_text.tag_configure("md_h2", font=("Times", 15, "bold"))
            self.output_text.tag_configure("md_h3", font=("Times", 13, "bold"))
            self.output_text.tag_configure("md_code", font=("Consolas", 11), background="#1e1e2e", foreground="#f5e0dc")
            
            self.output_text.bind("<Button-1>", lambda e: self.output_text.focus_set())
            
            # Read-only block but explicitly allow selections and Ctrl+C copying
            def block_typing(e):
                if (e.state & 0x4) and e.keysym.lower() == 'c':
                    return None
                if e.keysym in ("Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"):
                    return None
                return "break"
            self.output_text.bind("<Key>", block_typing)
            
            self.output_text.bind("<MouseWheel>", self._evaluate_scroll_position)
            self.output_text.bind("<Button-4>", self._evaluate_scroll_position)
            self.output_text.bind("<Button-5>", self._evaluate_scroll_position)
        except Exception as e:
            print(f"[Render Engine Error] Style binding sequence failed: {e}")

    def _evaluate_scroll_position(self, event=None):
        try:
            visible_fraction = self.output_text.yview()
            if visible_fraction and len(visible_fraction) > 1 and visible_fraction[1] < 0.99:
                self.user_scrolled_up = True
            else:
                self.user_scrolled_up = False
        except Exception:
            pass

    def _append_text(self, text: str, color: str = ""):
        """Processes and dynamically appends user bubbles or plain Ved text to the screen."""
        if text.startswith("You: ") or text.startswith("Dev: ") or text.startswith("Ved: "):
            return

        def action():
            is_user = (color == "#e5e9f0" or "You:" in text)
            clean_text = text.strip()
            if not clean_text:
                return

            if is_user:
                # 1. Row calculation matching full screen limits
                row_width = self.output_text.winfo_width() - 25 if self.output_text.winfo_width() > 100 else 560
                row_frame = tk.Frame(self.output_text, bg="#090a0f", width=row_width)
                row_frame.pack_propagate(False)

                # 2. FIXED: Removed padding values completely so the box tightly hugs your text characters
                bubble_frame = tk.Frame(row_frame, bg="#313244", padx=0, pady=0)
                bubble_frame.pack(side="right", padx=(40, 5)) 

                # 3. Fit word limits inside character bounds cleanly
                char_count = len(clean_text)
                if char_count < 10:
                    calculated_width = char_count + 1
                elif char_count < 50:
                    calculated_width = char_count
                else:
                    calculated_width = 50 # Let user text bubble take more width room if needed

                # 4. Create the core text widget container inside the bubble
                lbl = tk.Text(
                    bubble_frame, bg="#313244", fg="#89b4fa", font=("Arial", 11),
                    wrap="word", bd=0, highlightthickness=0, width=calculated_width, height=1
                )
                lbl.insert("1.0", clean_text)
                lbl.configure(state="disabled")
                lbl.pack(fill="both", expand=True)

                # 5. Measure text wrapping requirements
                self.root.update_idletasks()
                needed_lines = int(float(lbl.index("end-1c").split('.')[0]))
                lbl.configure(height=needed_lines)

                # 6. Adjust parent container bounds safely
                needed_h = lbl.winfo_reqheight()
                row_frame.configure(height=needed_h)

                # 7. Inject our structural layout row cleanly straight into the text line log
                self.output_text.window_create("end", window=row_frame)
                self.output_text.insert("end", "\n")
            else:
                # --- FIXED VED RENDERING: SITS IMMEDIATELY BELOW YOUR ROW ---
                idx_start = self.output_text.index("end-1c")
                self.output_text.insert("end", clean_text + "\n")
                self.output_text.tag_add("ved_msg", idx_start, "end-1c")
            
            if not self.user_scrolled_up:
                self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, action)

    def _append_stream_chunk(self, text: str, color: str = ""):
        """Appends live AI token streams cleanly pinned onto the left chat margin.

        Normalises the chunk so RUNAWAY whitespace (3+ newlines, 4+ spaces)
        does not render as a flood of blank lines or accidental indents.
        Standard "\n" and "\n\n" paragraph breaks are preserved verbatim so
        the chat panel remains readable. Without this preservation the model
        streams "\n" as its own chunk between text segments and the entire
        conversation collapses into one unreadable paragraph.
        """
        if not text:
            return
        cleaned = text
        if "\n" in cleaned:
            cleaned = _RUNAWAY_NEWLINES_RE.sub("\n\n", cleaned)
        if "    " in cleaned:
            cleaned = _RUNAWAY_SPACES_RE.sub(" ", cleaned)

        def action():
            try:
                existing_tail = self.output_text.get("end-2c", "end-1c")
            except Exception:
                existing_tail = ""
            local_cleaned = _strip_leading_blank_lines(cleaned, existing_tail)
            if not local_cleaned:
                return
            idx_start = self.output_text.index("end-1c")
            self.output_text.insert("end", local_cleaned)
            self.output_text.tag_add("ved_msg", idx_start, "end-1c")
            if not self.user_scrolled_up:
                self.output_text.see("end")
            self._resize_to_fit_content()
        self.root.after(0, action)

    def _clear_output(self):
        """Wipes the chat log output cleanly so a thread switch starts blank."""
        self.output_text.delete("1.0", "end")

    def _render_chat_history(self):
        """Flushes the log viewer screen cleanly and processes memory history frames."""
        self.output_text.delete("1.0", "end")
        for turn in self.chat_history:
            self._append_text(turn['user'], color="#e5e9f0")
            self._append_text(turn['assistant'], color="")
        
        if not self.user_scrolled_up:
            self.output_text.see("end")
        self._resize_to_fit_content()

    def _resize_to_fit_content(self):
        if not hasattr(self, 'is_snapped_to_max'):
            self.is_snapped_to_max = False
        if self.is_snapped_to_max:
            return
        raw_content_length = len(self.output_text.get("1.0", "end-1c"))
        if raw_content_length > 400:
            target_h = self.max_height
            target_y = self.root.winfo_screenheight() - self.max_height - 49
            self.is_snapped_to_max = True
        else:
            target_h = self.default_height
            target_y = self.root.winfo_screenheight() - self.default_height - 49
        self.root.geometry(f"{self.default_width}x{target_h}+{self.root.winfo_x()}+{target_y}")
