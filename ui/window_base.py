import tkinter as tk
import ctypes
import ctypes.wintypes

WDA_EXCLUDEFROMCAPTURE = 0x00000011
WS_EX_APPWINDOW        = 0x00040000
WS_EX_TOOLWINDOW       = 0x00000080
GWL_EXSTYLE            = -20
GWL_HWNDPARENT         = -8
GA_ROOT                = 2
SWP_NOMOVE             = 0x0002
SWP_NOSIZE             = 0x0001
SWP_FRAMECHANGED       = 0x0020
HWND_NOTOPMOST         = -2
HWND_MESSAGE           = -3
SW_HIDE                = 0
SW_SHOWNA              = 8

MODE_COLORS = {
    "turbo":     "#a6e3a1",
    "standard":  "#89b4fa",
    "hibernate": "#6c7086",
    "coder":     "#cba6f7",
    "error":     "#f38ba8",
}

class VedWindowBase:
    def __init__(self, root: tk.Tk):
        self.root = root
        # Withdraw before the shell ever sees the window — prevents taskbar registration
        self.root.withdraw()
        self.root.title("Ved")
        self.root.overrideredirect(True)
        self.root.configure(bg="#090a0f")

        self.TITLE_BAR_H       = 32
        self.INPUT_BAR_H       = 55
        self.default_width     = 600
        self.default_content_h = 220
        self.max_content_h     = 530
        self.default_height    = self.TITLE_BAR_H + self.default_content_h + self.INPUT_BAR_H
        self.max_height        = self.TITLE_BAR_H + self.max_content_h + self.INPUT_BAR_H

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
        # Parent to a message-only window before first paint so the shell never
        # registers a taskbar entry, then apply the TOOLWINDOW style as a backstop.
        self._detach_from_shell()
        self.root.after(150, self._apply_window_style)
        self.root.after(500, self._apply_window_style)
    # ------------------------------------------------------------------ #
    # HWND helpers
    # ------------------------------------------------------------------ #
    def _get_root_hwnd(self):
        child_hwnd = self.root.winfo_id()
        root_hwnd  = ctypes.windll.user32.GetAncestor(child_hwnd, GA_ROOT)
        return root_hwnd if root_hwnd else child_hwnd

    def _detach_from_shell(self):
        """Parent the HWND to a hidden message-only window while the root is
        still withdrawn. The shell only creates taskbar buttons for top-level
        windows with no owner, so this suppresses the button permanently."""
        try:
            self.root.update_idletasks()
            hwnd = self._get_root_hwnd()

            self._msg_hwnd = ctypes.windll.user32.CreateWindowExW(
                0, "Static", None, 0, 0, 0, 0, 0,
                HWND_MESSAGE, None, None, None
            )
            ctypes.windll.user32.SetWindowLongPtrW(hwnd, GWL_HWNDPARENT, self._msg_hwnd)
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED
            )
            self.root.deiconify()
        except Exception as e:
            print(f"[ved] Shell detach error: {e}")

    def _apply_window_style(self):
        """Set WS_EX_TOOLWINDOW and clear WS_EX_APPWINDOW.
        TOOLWINDOW is the unconditional shell signal to suppress taskbar buttons."""
        try:
            self.root.update_idletasks()
            hwnd = self._get_root_hwnd()
            cur  = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new  = (cur & ~WS_EX_APPWINDOW) | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new)
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED
            )
        except Exception as e:
            print(f"[ved] Style error: {e}")
    # ------------------------------------------------------------------ #
    # Screen capture exclusion
    # ------------------------------------------------------------------ #
    def _hide_from_screen_capture(self):
        try:
            self.root.update()
            hwnd   = self.root.winfo_id()
            result = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            if not result:
                parent = ctypes.windll.user32.GetParent(hwnd)
                if parent:
                    ctypes.windll.user32.SetWindowDisplayAffinity(parent, WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:
            print(f"[ved] Display affinity error: {e}")
    # ------------------------------------------------------------------ #
    # Minimize / tray / restore
    # ------------------------------------------------------------------ #
    def _safe_minimize(self):
        self._show_tray_icon()
        ctypes.windll.user32.ShowWindow(self._get_root_hwnd(), SW_HIDE)

    def _show_tray_icon(self):
        self._tray = tk.Toplevel(self.root)
        self._tray.overrideredirect(True)
        self._tray.attributes("-topmost", True)
        self._tray.configure(bg="#12131b")

        icon_size = 42
        sw        = self.root.winfo_screenwidth()
        self._tray.geometry(f"{icon_size}x{icon_size}+{sw - icon_size}+100")

        canvas = tk.Canvas(
            self._tray, width=icon_size, height=icon_size,
            bg="#12131b", highlightthickness=1, highlightbackground="#313244"
        )
        canvas.pack()

        mode  = getattr(self, "current_mode", "standard")
        color = MODE_COLORS.get(mode, "#a6e3a1")
        canvas.create_text(icon_size // 2, icon_size // 2 - 4, text="V",
                           fill=color, font=("ONE DAY", 10, "bold"))
        canvas.create_text(icon_size // 2, icon_size // 2 + 7, text=mode[:3].upper(),
                           fill=color, font=("Segoe UI", 5, "bold"))

        canvas.bind("<Button-1>", self._restore_from_tray)
        self._tray.bind("<Button-1>", self._restore_from_tray)
        canvas.bind("<Enter>", lambda e: canvas.configure(bg="#1e2030"))
        canvas.bind("<Leave>", lambda e: canvas.configure(bg="#12131b"))

    def _restore_from_tray(self, event=None):
        if hasattr(self, "_tray") and self._tray.winfo_exists():
            self._tray.destroy()
        hwnd = self._get_root_hwnd()
        # Re-assert style before the shell sees the window become visible
        self._apply_window_style()
        ctypes.windll.user32.ShowWindow(hwnd, SW_SHOWNA)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        self.root.after(50, lambda: self.input_entry.focus_set() if hasattr(self, "input_entry") else None)
    # ------------------------------------------------------------------ #
    # Drag
    # ------------------------------------------------------------------ #
    def _bind_title_drag(self, widget):
        widget.bind("<Button-1>", self._start_drag)
        widget.bind("<B1-Motion>", self._drag_window)

    def _start_drag(self, event):
        if getattr(self, "_drag_block_widgets", None) and event.widget in self._drag_block_widgets:
            return
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        self._orig_x = self.root.winfo_x()
        self._orig_y = self.root.winfo_y()

    def _drag_window(self, event):
        dx = event.x_root - self._drag_x
        dy = event.y_root - self._drag_y
        self.root.geometry(f"+{self._orig_x + dx}+{self._orig_y + dy}")