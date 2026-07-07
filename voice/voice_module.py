import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import threading
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from .audio_loop import _unified_audio_loop
from .audio_processors import _process_captured_audio, _process_initial_prompt_logic, _process_confirmation_logic
from .audio_utils import _say_reply, _say_reply_interruptible, _tts_worker, stop_tts, trigger_reread, toggle_listening, _clear_input_box, _reset_to_wake_word


def _compute_silence_threshold(rms_samples):
    """Pure helper: derive silence_threshold from per-chunk RMS values (Chunk B).

    Uses the 75th percentile of RMS samples (robust against a single transient
    pop), then applies a 1.8x multiplier for ~80% headroom above the noise
    floor, with a floor of 80 to keep the threshold sensible in absurdly
    quiet rooms. Empty input returns the floor.

    Args:
        rms_samples: iterable of RMS floats (one per audio chunk).

    Returns:
        int: silence_threshold value to assign to the VoiceSystem instance.
    """
    samples = list(rms_samples)
    if not samples:
        return 80
    noise_rms = float(np.percentile(samples, 75))
    return max(int(noise_rms * 1.8), 80)


class VoiceSystem:
    def __init__(self, root, input_frame, input_entry, send_command):
        # tkinter and the heavy voice backends (faster_whisper, openwakeword,
        # piper) are imported lazily so this module can be imported (e.g.
        # during `pytest --collect-only` on headless Linux) without
        # requiring the tk runtime or any of the heavy voice deps to be
        # installed. The UI mic button and the audio pipeline are the only
        # consumers; deferring the imports keeps the module surface clean.
        import tkinter as tk
        from faster_whisper import WhisperModel
        from openwakeword.model import Model as OWWModel
        from piper import PiperVoice

        self.root = root
        self.input_frame = input_frame
        self.input_entry = input_entry
        self.send_command = send_command
        
        # Core State System
        self.is_running = True
        self.current_state = "WAKE_WORD"  # States: WAKE_WORD, RECORDING, CONFIRMATION, PLAYING
        self.wake_phrase = "alexa"
        self.pending_text = ""
        self._wake_hits = 0
        self._wake_cooldown_until = 0.0

        # Ambient-noise calibration (Chunk B).
        # self.silence_threshold is consumed by the audio loop's energy gate.
        # It starts at a safe default (slightly above the legacy 150) so the
        # system still works if calibration is skipped. calibrate_ambient_noise
        # overwrites this in-place once it runs.
        # Production wiring landed in Chunk C: _unified_audio_loop now checks
        # self._needs_calibration on its first iteration and invokes
        # self.calibrate_ambient_noise(stream=stream, duration_s=1.0) using
        # the live InputStream, then clears the flag in a finally block.
        self.silence_threshold = 360
        self._needs_calibration = True

        # Interruptible TTS (Chunk C).
        # _tts_lock: serializes concurrent _say_reply_interruptible spawns so
        #   two callers can't race to set self._tts_thread.
        # _tts_thread: the currently-playing TTS worker thread (or None).
        # _tts_stop_event: set to request the worker exit early / abandon sd.play.
        self._tts_lock = threading.Lock()
        self._tts_thread: threading.Thread | None = None
        self._tts_stop_event = threading.Event()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, os.getenv("voice_file"))
        json_path = os.path.join(base_dir, os.getenv("voice_json"))
        
        self.piper_model = PiperVoice.load(onnx_path, json_path)
        self.oww_model = OWWModel(
            wakeword_models=[self.wake_phrase],
            vad_threshold=0.25,
            inference_framework="onnx",
        )
        self.model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)

        # UI Button Setup
        self.mic_button = tk.Button(
            self.input_frame, text="🎙", bg="#12131b", fg="#a6adc8", bd=0,
            activebackground="#1e1e2e", activeforeground="#b4befe",
            font=("Arial", 12), cursor="hand2", width=3, justify="center"
        )
        self.mic_button.pack(side="right", padx=(0,5), pady=5)
        self.mic_button.config(command=self.toggle_listening)
        
        # Single background processing thread
        threading.Thread(target=self._unified_audio_loop, daemon=True).start()

    def calibrate_ambient_noise(self, stream=None, duration_s=1.0, rms_samples=None):
        """Calibrate self.silence_threshold from ambient noise (Chunk B).

        Two call modes:

        - Production (pass ``stream``): reads ``int(duration_s * 16000 / 1280)``
          chunks of 1280 samples at 16 kHz from the live sounddevice stream,
          computes per-chunk RMS as ``np.abs(audio_chunk.flatten()).mean()``
          (matches the pattern in ``audio/audio_loop.py``), and derives the
          threshold via :func:`_compute_silence_threshold`.

        - Test / seed (pass ``rms_samples``): derives the threshold from a known
          list of RMS values; skips all audio I/O. Used by unit tests and by
          any future caller that has RMS pre-computed from another source.

        The method assigns the resulting value to ``self.silence_threshold``
        and returns it.

        Production wiring (Chunk C): ``_unified_audio_loop`` reads
        ``self._needs_calibration`` on its first iteration and invokes
        ``calibrate_ambient_noise(stream=stream, duration_s=1.0)`` against the
        live InputStream. On any exception the loop logs and falls back to the
        default threshold; on success the flag is cleared.
        """
        if rms_samples is None:
            # Production path. ``stream`` is expected to be a sounddevice
            # InputStream with 1280-sample blocks at 16 kHz.
            chunk_size = 1280
            n_chunks = int(duration_s * 16000 / chunk_size)
            collected = []
            for _ in range(n_chunks):
                audio_chunk, _ = stream.read(chunk_size)
                collected.append(np.abs(audio_chunk.flatten()).mean())
            samples = collected
        else:
            samples = list(rms_samples)

        self.silence_threshold = _compute_silence_threshold(samples)
        return self.silence_threshold

    # Bind imported core behaviors cleanly onto the class architecture
    _say_reply = _say_reply
    _say_reply_interruptible = _say_reply_interruptible
    _tts_worker = _tts_worker
    stop_tts = stop_tts
    trigger_reread = trigger_reread
    toggle_listening = toggle_listening
    _clear_input_box = _clear_input_box
    _unified_audio_loop = _unified_audio_loop
    _process_captured_audio = _process_captured_audio
    _process_initial_prompt_logic = _process_initial_prompt_logic
    _process_confirmation_logic = _process_confirmation_logic
    _reset_to_wake_word = _reset_to_wake_word
