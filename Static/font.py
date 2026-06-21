import os
import ctypes

# 1. Get the exact path to this folder (the Static folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Link it directly to your .otf filename inside this folder
font_path = os.path.join(current_dir, "ONEDAY.otf")

# 3. Register it with Windows
if os.path.exists(font_path):
    ctypes.windll.gdi32.AddFontResourceW(font_path)
    print(f"[Font] Registered custom font from {font_path}")
else:
    print(f"[Font Error] Could not find file at: {font_path}")
