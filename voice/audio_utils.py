import time
import tkinter as tk
import numpy as np
import sounddevice as sd

def _say_reply(self, text):
    """Plays speech using Piper, silently forcing master hardware level modifications without overlays."""
    previous_state = self.current_state
    self.current_state = "PLAYING"
    import ctypes
    try:
        Ole32 = ctypes.windll.ole32
        Ole32.CoInitialize(None)
        winmm = ctypes.WinDLL('winmm.dll')
        orig_vol = ctypes.c_uint32()
        winmm.waveOutGetVolume(0, ctypes.byref(orig_vol))
        winmm.waveOutSetVolume(0, 0xFFFFFFFF)
    except Exception:
        orig_vol = None
    if self.piper_model:
        try:
            slowed_text = text.replace(".", "... ").replace(",", ",... ").replace("?", "?... ")
            raw_bytes = b""
            for chunk in self.piper_model.synthesize(slowed_text):
                raw_bytes += chunk.audio_int16_bytes
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
            silence_padding = np.zeros(8000, dtype=np.int16)
            silence_trailing = np.zeros(24000, dtype=np.int16)
            full_audio = np.concatenate([silence_padding, audio_data, silence_trailing])
            sd.play(full_audio, 16000)
            sd.wait() 
        except Exception as e:
            print(f"[Audio Error] {e}")
    if orig_vol is not None:
        try:
            winmm.waveOutSetVolume(0, orig_vol)
        except Exception:
            pass
    try:
        Ole32.CoUninitialize()
    except Exception:
        pass
    self.current_state = previous_state

def trigger_reread(self):
    """Resets text fields and instantly drops back into prompt recording."""
    self.root.after(0, self._clear_input_box)
    self._say_reply("Try again.")
    time.sleep(0.2)
    self.current_state = "RECORDING"

def toggle_listening(self):
    """Handles manual button clicks cleanly."""
    if self.current_state in ["RECORDING", "CONFIRMATION"]:
        self.mic_button.config(text="🎙️", fg="#a6adc8")
        self.current_state = "WAKE_WORD"
        print("[Voice] Listening stopped manually.")
    else:
        self.mic_button.config(text="🛑", fg="#f38ba8")
        self.root.update_idletasks()
        self._say_reply("Yes? What's up?")
        self.current_state = "RECORDING"

def _clear_input_box(self):
    if hasattr(self.input_entry, "delete"):
        if isinstance(self.input_entry, tk.Text):
            self.input_entry.delete("1.0", tk.END)
        else:
            self.input_entry.delete(0, tk.END)

def _reset_to_wake_word(self, stream):
    """Helper to cleanly clear caches and reset state back to listening."""
    time.sleep(0.2)
    if stream.read_available > 0:
        stream.read(stream.read_available)
    if hasattr(self.oww_model, "reset"):
        self.oww_model.reset()
    self.current_state = "WAKE_WORD"
    print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")
