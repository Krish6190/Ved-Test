"""
Ved Voice Tuner
================
Standalone tkinter tool for live-tuning the voice pipeline's parameters
without touching the real GUI / chatbot wiring. Mirrors the logic in
audio_loop.py + audio_processors.py (wake-word hysteresis, silence-gated
recording, faster-whisper VAD) but every knob is a slider you can drag
while the mic is live, plus a log + meters so you can SEE what each knob
does in real time.

Run directly:
    python voice_tuner.py

Dependencies (same as the real voice module):
    pip install sounddevice numpy faster-whisper openwakeword

Notes:
- No Piper / TTS, no .env, no wake_sound asset needed. This tool only
  exercises the *listening* side of the pipeline (wake detection, VAD,
  silence-gated recording, transcription), which is what you said you
  want to fine-tune. Tell me if you also want TTS/Piper tuning wired in
  and I'll extend this with that (would need voice_file/voice_json paths
  from your .env).
- openwakeword's "alexa" model ships built-in, no extra model file needed.
  If you've trained/are using a custom wake model, tell me the path and
  I'll wire it in (currently hardcoded to self.wake_phrase = "alexa" to
  match voice_module.py).
"""

import time
import threading
import queue
import collections

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk

from faster_whisper import WhisperModel
from openwakeword.model import Model as OWWModel


# --------------------------------------------------------------------------- #
# Pure helpers — copied verbatim from audio_loop.py / voice_module.py so the
# tuner's behavior matches production exactly.
# --------------------------------------------------------------------------- #

def evaluate_wake_hit(score, current_hits, now, cooldown_until, threshold, required_hits):
    if now < cooldown_until:
        return current_hits, False
    if score > threshold:
        new_hits = current_hits + 1
        if new_hits >= required_hits:
            return new_hits, True
        return new_hits, False
    return 0, False


def compute_silence_threshold(rms_samples):
    samples = list(rms_samples)
    if not samples:
        return 80
    noise_rms = float(np.percentile(samples, 75))
    return max(int(noise_rms * 1.8), 80)


# --------------------------------------------------------------------------- #
# Tuner
# --------------------------------------------------------------------------- #

FS = 16000
CHUNK_SIZE = 1280  # 80ms @ 16kHz, matches production


class VoiceTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Ved Voice Tuner")
        self.root.geometry("760x720")
        self.root.configure(bg="#11121a")

        self.is_running = False
        self.current_state = "IDLE"  # IDLE, WAKE_WORD, RECORDING, COOLDOWN
        self._wake_hits = 0
        self._wake_cooldown_until = 0.0
        self._stream_thread = None
        self._stop_flag = threading.Event()
        self.silence_threshold_live = 200

        self.log_queue = queue.Queue()
        self.level_history = collections.deque(maxlen=120)  # for the volume meter trail

        # Loaded lazily so the window opens instantly.
        self.oww_model = None
        self.whisper_model = None

        self._build_ui()
        self._poll_log_queue()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        FG = "#cdd6f4"
        BG = "#11121a"
        PANEL = "#1b1c27"

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TScale", background=PANEL)
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=PANEL, foreground="#b4befe", font=("Segoe UI", 10, "bold"))

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=10)

        self.start_btn = tk.Button(top, text="▶ Start Listening", command=self.toggle_engine,
                                    bg="#a6e3a1", fg="#11121a", font=("Segoe UI", 10, "bold"),
                                    relief="flat", padx=10, pady=6)
        self.start_btn.pack(side="left")

        self.calib_btn = tk.Button(top, text="🎚 Calibrate Ambient Noise (1s)", command=self.calibrate,
                                    bg="#89b4fa", fg="#11121a", font=("Segoe UI", 10, "bold"),
                                    relief="flat", padx=10, pady=6)
        self.calib_btn.pack(side="left", padx=8)

        self.state_label = tk.Label(top, text="STATE: IDLE", bg=BG, fg="#f9e2af", font=("Consolas", 11, "bold"))
        self.state_label.pack(side="right")

        # --- Live meters -----------------------------------------------
        meters = tk.Frame(self.root, bg=PANEL)
        meters.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(meters, text="Mic Volume (vs. silence threshold)", bg=PANEL, fg=FG, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
        self.vol_canvas = tk.Canvas(meters, height=22, bg="#0c0d13", highlightthickness=0)
        self.vol_canvas.pack(fill="x", padx=10, pady=(2, 8))

        tk.Label(meters, text="Wake Word Confidence (vs. wake threshold)", bg=PANEL, fg=FG, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10)
        self.wake_canvas = tk.Canvas(meters, height=22, bg="#0c0d13", highlightthickness=0)
        self.wake_canvas.pack(fill="x", padx=10, pady=(2, 10))

        # --- Sliders ------------------------------------------------------
        sliders = tk.Frame(self.root, bg=PANEL)
        sliders.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(sliders, text="Live-tunable parameters", bg=PANEL, fg="#b4befe", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        self.wake_threshold = self._add_slider(sliders, "Wake threshold (0-1)", 0.05, 0.95, 0.45, 0.01)
        self.wake_required_hits = self._add_slider(sliders, "Wake required consecutive hits", 1, 10, 3, 1)
        self.wake_cooldown = self._add_slider(sliders, "Wake cooldown (seconds)", 0.0, 5.0, 2.0, 0.1)
        self.silence_threshold = self._add_slider(sliders, "Silence threshold (manual override, RMS)", 20, 1500, 200, 5)
        self.speech_pause_limit = self._add_slider(sliders, "Speech pause limit (chunks of 80ms before cutting)", 1, 60, 12, 1)
        self.initial_wait_limit = self._add_slider(sliders, "Initial wait limit (chunks before giving up on silence)", 5, 150, 50, 1)
        self.max_chunks = self._add_slider(sliders, "Max recording length (chunks, hard cap)", 50, 600, 250, 10)
        self.vad_min_speech_ms = self._add_slider(sliders, "Whisper VAD min speech duration (ms)", 0, 1000, 250, 10)
        self.beam_size = self._add_slider(sliders, "Whisper beam size", 1, 10, 3, 1)

        self.vad_enabled = tk.BooleanVar(value=True)
        vad_row = tk.Frame(sliders, bg=PANEL)
        vad_row.pack(fill="x", padx=10, pady=4)
        tk.Checkbutton(vad_row, text="Enable Whisper VAD filter", variable=self.vad_enabled,
                        bg=PANEL, fg=FG, selectcolor="#313244", activebackground=PANEL,
                        activeforeground=FG).pack(anchor="w")

        # --- Log ------------------------------------------------------
        tk.Label(self.root, text="Event Log", bg=BG, fg="#b4befe", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14)
        log_frame = tk.Frame(self.root, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(2, 6))
        self.log_box = tk.Text(log_frame, bg="#0c0d13", fg="#a6e3a1", font=("Consolas", 9), wrap="word")
        scrollbar = tk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=scrollbar.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(bottom, text="📋 Print current values as Python constants", command=self.dump_constants,
                  bg="#cba6f7", fg="#11121a", font=("Segoe UI", 9, "bold"), relief="flat", padx=8, pady=4).pack(side="left")
        tk.Button(bottom, text="Clear Log", command=lambda: self.log_box.delete("1.0", tk.END),
                  bg="#45475a", fg=FG, font=("Segoe UI", 9), relief="flat", padx=8, pady=4).pack(side="left", padx=8)

    def _add_slider(self, parent, label, lo, hi, default, resolution):
        row = tk.Frame(parent, bg="#1b1c27")
        row.pack(fill="x", padx=10, pady=3)
        lbl = tk.Label(row, text=label, bg="#1b1c27", fg="#cdd6f4", font=("Segoe UI", 9), width=42, anchor="w")
        lbl.pack(side="left")
        var = tk.DoubleVar(value=default)
        val_lbl = tk.Label(row, text=str(default), bg="#1b1c27", fg="#f9e2af", font=("Consolas", 9), width=6)
        val_lbl.pack(side="right")

        def on_move(v):
            val_lbl.config(text=f"{float(v):.2f}" if resolution < 1 else str(int(float(v))))

        scale = tk.Scale(row, from_=lo, to=hi, resolution=resolution, orient="horizontal",
                          variable=var, showvalue=False, command=on_move,
                          bg="#1b1c27", fg="#cdd6f4", troughcolor="#313244",
                          highlightthickness=0, sliderrelief="flat", length=300)
        scale.pack(side="left", fill="x", expand=True, padx=8)
        on_move(default)
        return var

    # ------------------------------------------------------------------ #
    # Logging / meters (main-thread safe via queue)
    # ------------------------------------------------------------------ #
    def log(self, msg):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.insert(tk.END, msg + "\n")
                self.log_box.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log_queue)

    def _draw_bar(self, canvas, frac, marker_frac, color):
        canvas.delete("all")
        w = canvas.winfo_width() or 700
        h = int(canvas["height"])
        frac = max(0.0, min(1.0, frac))
        canvas.create_rectangle(0, 0, w, h, fill="#0c0d13", outline="")
        canvas.create_rectangle(0, 0, int(w * frac), h, fill=color, outline="")
        mx = int(w * max(0.0, min(1.0, marker_frac)))
        canvas.create_line(mx, 0, mx, h, fill="#f38ba8", width=2)

    def set_state(self, s):
        self.current_state = s
        self.state_label.config(text=f"STATE: {s}")

    # ------------------------------------------------------------------ #
    # Engine
    # ------------------------------------------------------------------ #
    def toggle_engine(self):
        if self.is_running:
            self._stop_flag.set()
            self.is_running = False
            self.start_btn.config(text="▶ Start Listening", bg="#a6e3a1")
            self.set_state("IDLE")
            return

        if self.oww_model is None:
            self.log("[init] Loading openWakeWord model (alexa)...")
            self.oww_model = OWWModel(wakeword_models=["alexa"], vad_threshold=0.25, inference_framework="onnx")
        if self.whisper_model is None:
            self.log("[init] Loading faster-whisper (base, cpu/int8)...")
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)

        self._stop_flag.clear()
        self.is_running = True
        self.start_btn.config(text="■ Stop Listening", bg="#f38ba8")
        self.set_state("WAKE_WORD")
        self._stream_thread = threading.Thread(target=self._loop, daemon=True)
        self._stream_thread.start()

    def calibrate(self):
        if self.is_running:
            self.log("[calibrate] Stop listening first, then calibrate.")
            return

        def run():
            self.log("[calibrate] Listening to ambient noise for 1s...")
            with sd.InputStream(samplerate=FS, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
                n_chunks = int(1.0 * FS / CHUNK_SIZE)
                collected = []
                for _ in range(n_chunks):
                    audio_chunk, _ = stream.read(CHUNK_SIZE)
                    collected.append(np.abs(audio_chunk.flatten()).mean())
            threshold = compute_silence_threshold(collected)
            self.silence_threshold.set(threshold)
            self.log(f"[calibrate] Done. silence_threshold = {threshold}")

        threading.Thread(target=run, daemon=True).start()

    def _loop(self):
        recorded_chunks = []
        silent_chunk_counter = 0
        initial_wait_counter = 0
        has_started_talking = False

        self.log(f"[engine] Listening for wake word 'alexa'...")
        try:
            with sd.InputStream(samplerate=FS, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
                while not self._stop_flag.is_set():
                    try:
                        audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                    except Exception as e:
                        time.sleep(0.05)
                        continue

                    audio_data = audio_chunk.flatten()
                    volume_score = float(np.abs(audio_data).mean())

                    silence_thr = self.silence_threshold.get()
                    self.root.after(0, self._draw_bar, self.vol_canvas, min(volume_score / max(silence_thr * 3, 1), 1.0), silence_thr / max(silence_thr * 3, 1), "#89b4fa")

                    if self.current_state == "WAKE_WORD":
                        prediction = self.oww_model.predict(audio_data)
                        wake_score = float(prediction.get("alexa", 0.0))
                        thr = self.wake_threshold.get()
                        req_hits = int(self.wake_required_hits.get())
                        cooldown_s = self.wake_cooldown.get()

                        self.root.after(0, self._draw_bar, self.wake_canvas, wake_score, thr, "#cba6f7")

                        self._wake_hits, triggered = evaluate_wake_hit(
                            wake_score, self._wake_hits, time.monotonic(),
                            self._wake_cooldown_until, thr, req_hits,
                        )
                        if triggered:
                            self._wake_hits = 0
                            self._wake_cooldown_until = time.monotonic() + cooldown_s
                            self.log(f"[wake] TRIGGERED (confidence={wake_score:.3f}, threshold={thr:.2f})")
                            self.root.after(0, self.set_state, "RECORDING")
                            self.current_state = "RECORDING"
                            if hasattr(self.oww_model, "reset"):
                                self.oww_model.reset()
                            recorded_chunks = []
                            silent_chunk_counter = 0
                            initial_wait_counter = 0
                            has_started_talking = False
                        continue

                    if self.current_state == "RECORDING":
                        recorded_chunks.append(audio_chunk)
                        pause_limit = int(self.speech_pause_limit.get())
                        wait_limit = int(self.initial_wait_limit.get())
                        max_chunks = int(self.max_chunks.get())

                        if not has_started_talking:
                            if volume_score > silence_thr:
                                has_started_talking = True
                                initial_wait_counter = 0
                            else:
                                initial_wait_counter += 1
                            if initial_wait_counter >= wait_limit:
                                silent_chunk_counter = 999
                        else:
                            if volume_score < silence_thr:
                                silent_chunk_counter += 1
                            else:
                                silent_chunk_counter = 0

                        cutoff = (1 if silent_chunk_counter == 999 else pause_limit)
                        if silent_chunk_counter >= cutoff or len(recorded_chunks) >= max_chunks:
                            current_chunks = list(recorded_chunks)
                            current_talked = has_started_talking
                            recorded_chunks = []
                            silent_chunk_counter = 0
                            has_started_talking = False
                            self.root.after(0, self.set_state, "TRANSCRIBING")
                            self.current_state = "TRANSCRIBING"
                            self._handle_capture(current_chunks, current_talked, stream)

        except Exception as e:
            self.log(f"[engine] FATAL: {e}")
        finally:
            self.log("[engine] Stopped.")

    def _handle_capture(self, chunks, talked, stream):
        if not chunks or not talked:
            self.log("[capture] No speech detected (false start) — back to wake word.")
            self.current_state = "WAKE_WORD"
            self.root.after(0, self.set_state, "WAKE_WORD")
            return

        duration_s = len(chunks) * CHUNK_SIZE / FS
        try:
            audio_data = np.concatenate(chunks, axis=0).flatten().astype(np.float32) / 32768.0
            vad_kwargs = dict(min_speech_duration_ms=int(self.vad_min_speech_ms.get())) if self.vad_enabled.get() else {}
            t0 = time.monotonic()
            segments, info = self.whisper_model.transcribe(
                audio_data,
                beam_size=int(self.beam_size.get()),
                language="en",
                vad_filter=self.vad_enabled.get(),
                vad_parameters=vad_kwargs if self.vad_enabled.get() else None,
            )
            text = "".join(s.text for s in segments).strip()
            elapsed = time.monotonic() - t0
            self.log(f"[capture] {duration_s:.2f}s audio -> transcribed in {elapsed:.2f}s: \"{text}\"")
        except Exception as e:
            self.log(f"[capture] Transcription error: {e}")
        finally:
            if stream.read_available > 0:
                stream.read(stream.read_available)
            self.current_state = "WAKE_WORD"
            self.root.after(0, self.set_state, "WAKE_WORD")

    # ------------------------------------------------------------------ #
    def dump_constants(self):
        lines = [
            "# --- Copy these into audio_loop.py / voice_module.py ---",
            f"WAKE_THRESHOLD = {self.wake_threshold.get():.2f}",
            f"WAKE_REQUIRED_HITS = {int(self.wake_required_hits.get())}",
            f"WAKE_COOLDOWN_SECONDS = {self.wake_cooldown.get():.1f}",
            f"# silence_threshold (use as self.silence_threshold default, or via calibrate_ambient_noise)",
            f"silence_threshold = {int(self.silence_threshold.get())}",
            f"speech_pause_limit = {int(self.speech_pause_limit.get())}",
            f"initial_wait_limit = {int(self.initial_wait_limit.get())}",
            f"# recorded_chunks hard cap (len(recorded_chunks) >= N)",
            f"max_recording_chunks = {int(self.max_chunks.get())}",
            f"# faster_whisper transcribe() args",
            f"vad_filter = {self.vad_enabled.get()}",
            f"vad_parameters = dict(min_speech_duration_ms={int(self.vad_min_speech_ms.get())})",
            f"beam_size = {int(self.beam_size.get())}",
        ]
        self.log("\n".join(lines))


def main():
    root = tk.Tk()
    app = VoiceTuner(root)

    def on_close():
        app._stop_flag.set()
        root.after(150, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
