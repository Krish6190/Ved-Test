import time
import threading
import tkinter as tk
import numpy as np
import sounddevice as sd


def _say_reply(self, text):
    """Plays speech using Piper, silently forcing master hardware level modifications without overlays.

    Chunk C: this call is blocking (used for the confirmation echo). It calls
    ``sd.stop()`` at the very top so any in-flight interruptible bot-response
    speech is cancelled before we begin. The volume duck + restore are wrapped
    in a try/finally so volume ALWAYS restores, even if Piper or sd raises.
    """
    # Clear any in-flight interruptible playback so this blocking call gets clean audio.
    try:
        sd.stop()
    except Exception:
        pass

    previous_state = self.current_state
    self.current_state = "PLAYING"
    import ctypes
    orig_vol = None
    winmm = None
    Ole32 = None
    try:
        try:
            Ole32 = ctypes.windll.ole32
            Ole32.CoInitialize(None)
            winmm = ctypes.WinDLL('winmm.dll')
            orig_vol = ctypes.c_uint32()
            winmm.waveOutGetVolume(0, ctypes.byref(orig_vol))
            winmm.waveOutSetVolume(0, 0xFFFFFFFF)
        except Exception:
            orig_vol = None
            winmm = None

        if self.piper_model:
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
    finally:
        # ALWAYS restore volume, even on exception or interruption.
        if winmm is not None and orig_vol is not None:
            try:
                winmm.waveOutSetVolume(0, orig_vol)
            except Exception:
                pass
        if Ole32 is not None:
            try:
                Ole32.CoUninitialize()
            except Exception:
                pass
    self.current_state = previous_state


def _say_reply_interruptible(self, text):
    """Spawn a daemon thread that speaks ``text`` and can be interrupted.

    Lifecycle (Chunk C):
      1. Acquire ``self._tts_lock`` (serializes concurrent spawn attempts).
      2. If a previous TTS thread is still alive, set the stop event,
         call ``sd.stop()`` to abort audio playback, and join with a 0.5s
         timeout.
      3. Clear the stop event so the new playback can proceed.
      4. Spawn a daemon thread running ``self._tts_worker(text)``.
      5. Release the lock and return immediately (non-blocking).
    """
    with self._tts_lock:
        if self._tts_thread is not None and self._tts_thread.is_alive():
            self._tts_stop_event.set()
            try:
                sd.stop()
            except Exception:
                pass
            self._tts_thread.join(timeout=0.5)
        self._tts_stop_event.clear()
        self._tts_thread = threading.Thread(
            target=self._tts_worker, args=(text,), daemon=True
        )
        self._tts_thread.start()


def _tts_worker(self, text):
    """Daemon worker: synthesize ``text`` with Piper and play it.

    Checks ``self._tts_stop_event`` between Piper chunks so an interrupt
    can break the synthesis loop early. After the loop, only plays audio
    if the stop event was NOT set. Volume duck + restore are wrapped in
    a try/finally so volume is ALWAYS restored, even on interrupt or
    exception (a critical safety property — interruption while ducked
    would otherwise leave the OS volume at max).
    """
    import ctypes
    orig_vol = None
    winmm = None
    Ole32 = None
    try:
        try:
            Ole32 = ctypes.windll.ole32
            Ole32.CoInitialize(None)
            winmm = ctypes.WinDLL('winmm.dll')
            orig_vol = ctypes.c_uint32()
            winmm.waveOutGetVolume(0, ctypes.byref(orig_vol))
            winmm.waveOutSetVolume(0, 0xFFFFFFFF)
        except Exception:
            orig_vol = None
            winmm = None

        slowed_text = text.replace(".", "... ").replace(",", ",... ").replace("?", "?... ")
        raw_bytes = b""
        for chunk in self.piper_model.synthesize(slowed_text):
            if self._tts_stop_event.is_set():
                break
            raw_bytes += chunk.audio_int16_bytes

        if raw_bytes and not self._tts_stop_event.is_set():
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
            silence_padding = np.zeros(8000, dtype=np.int16)
            silence_trailing = np.zeros(24000, dtype=np.int16)
            full_audio = np.concatenate([silence_padding, audio_data, silence_trailing])
            sd.play(full_audio, 16000)
            sd.wait()
    except Exception as e:
        print(f"[TTS Worker Error] {e}")
    finally:
        # ALWAYS restore volume, even on interrupt or exception.
        if winmm is not None and orig_vol is not None:
            try:
                winmm.waveOutSetVolume(0, orig_vol)
            except Exception:
                pass
        if Ole32 is not None:
            try:
                Ole32.CoUninitialize()
            except Exception:
                pass


def stop_tts(self):
    """Idempotently stop any in-flight interruptible TTS playback.

    Safe to call when nothing is playing (no thread, no exception).
    After this returns, ``self._tts_stop_event`` is cleared so the next
    ``_say_reply_interruptible`` call can start fresh.
    """
    self._tts_stop_event.set()
    try:
        sd.stop()
    except Exception:
        pass
    if self._tts_thread is not None and self._tts_thread.is_alive():
        self._tts_thread.join(timeout=0.5)
    self._tts_stop_event.clear()


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
