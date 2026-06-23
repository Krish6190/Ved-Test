import tkinter as tk
import ctypes

WDA_EXCLUDEFROMCAPTURE = 0x00000011

class VedWindowBase:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ved")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#090a0f")
        self.TITLE_BAR_H = 32
        self.INPUT_BAR_H = 55
        self.default_width = 600
        self.default_content_h = 250  # Capped base boundary height
        self.max_content_h = 550

        self.default_height = self.TITLE_BAR_H + self.default_content_h + self.INPUT_BAR_H
        self.max_height = self.TITLE_BAR_H + self.max_content_h + self.INPUT_BAR_H

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(
            f"{self.default_width}x{self.default_height}"
            f"+{sw - self.default_width - 2}+{sh - self.default_height - 49}"
        )
        self.root.minsize(self.default_width, self.default_height)
        self.root.maxsize(self.default_width, self.max_height)
        self.root.resizable(False, True)
        
        self._drag_x = 0
        self._drag_y = 0
        self._orig_x = 0
        self._orig_y = 0

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

    def _hide_from_screen_capture(self):
        try:
            self.root.update()
            raw_id = self.root.winfo_id()
            result = ctypes.windll.user32.SetWindowDisplayAffinity(raw_id, WDA_EXCLUDEFROMCAPTURE)
            if not result and ctypes.windll.user32.GetParent(raw_id):
                ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.windll.user32.GetParent(raw_id), WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:
            print(f"[ved] Display affinity bypass error: {e}")
