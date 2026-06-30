import os
import time
import threading
import numpy as np
import sounddevice as sd
import winsound
WAKE_THRESHOLD = 0.45           
WAKE_REQUIRED_HITS = 2     
WAKE_COOLDOWN_SECONDS = 2.0     

def evaluate_wake_hit(score, current_hits, now, cooldown_until):
    """Pure helper for wake-word hysteresis + cooldown.
    Returns (new_hit_count, triggered):
      - new_hit_count: the updated consecutive-hit counter (reset on below-threshold).
      - triggered: True exactly once when the threshold + cooldown + required-hits
        conditions all line up on this frame.

    Args:
        score:         per-frame wake-word confidence (float, 0..1).
        current_hits:  running consecutive-above-threshold counter (int).
        now:           monotonic timestamp for this frame (float).
        cooldown_until: monotonic timestamp until which wake is suppressed (float).
    """
    if now < cooldown_until:
        return current_hits, False

    if score > WAKE_THRESHOLD:
        new_hits = current_hits + 1
        if new_hits >= WAKE_REQUIRED_HITS:
            return new_hits, True
        return new_hits, False
    return 0, False

def _unified_audio_loop(self):
    """Single microphone loop to avoid hardware streaming clashes."""
    fs = 16000
    chunk_size = 1280

    recorded_chunks = []
    # Chunk B/C: silence_threshold is now owned by the instance. Calibration
    # (below, inside the with-block, before the while-loop) overwrites it on
    # the live InputStream. Until calibration runs we use the default set in
    # __init__ (200) or whatever calibrate_ambient_noise(rms_samples=...) was
    # called with earlier. The local variable is snapshotted after calibration
    # so the hot path avoids per-chunk attribute lookups.
    silence_threshold = self.silence_threshold
    speech_pause_limit = 12  # unified; CONFIRMATION was 6 which was too aggressive
    initial_wait_limit = 50
    silent_chunk_counter = 0
    has_started_talking = False

    print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=chunk_size) as stream:
        # Chunk C: deferred calibration wiring. Runs once, on the live stream,
        # before the wake loop starts listening. If it fails the loop still
        # runs with the default silence_threshold; the flag is cleared either
        # way so we don't retry on every wake cycle.
        if getattr(self, "_needs_calibration", False):
            try:
                self.calibrate_ambient_noise(stream=stream, duration_s=1.0)
            except Exception as e:
                print(f"[Calibration] Failed, using default threshold: {e}")
            finally:
                self._needs_calibration = False
        # Snapshot the (possibly calibrated) threshold for the hot path.
        silence_threshold = self.silence_threshold
        while self.is_running:
            if self.current_state == "PLAYING":
                if stream.read_available > 0:
                    stream.read(stream.read_available)
                time.sleep(0.05)
                continue

            try:
                audio_chunk, overflowed = stream.read(chunk_size)
                audio_data = audio_chunk.flatten()
                volume_score = np.abs(audio_data).mean()
                if self.current_state == "WAKE_WORD":
                    prediction = self.oww_model.predict(audio_data)
                    wake_score = prediction[self.wake_phrase]
                    self._wake_hits, triggered = evaluate_wake_hit(
                        wake_score,
                        self._wake_hits,
                        time.monotonic(),
                        self._wake_cooldown_until,
                    )
                    if triggered:
                        self._wake_hits = 0
                        self._wake_cooldown_until = time.monotonic() + WAKE_COOLDOWN_SECONDS
                        print(f"[Wake Engine] Wake word triggered! (Confidence: {wake_score:.2f})")
                        self.root.after(0, lambda: self.mic_button.config(text="🛑", fg="#f38ba8"))
                        self.current_state = "PLAYING"
                        def play_engine_wake_worker():
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            audio_asset = os.path.join(base_dir, os.getenv("wake_sound"))
                            if os.path.exists(audio_asset):
                                try:
                                    winsound.PlaySound(audio_asset, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                                except Exception as sound_err:
                                    print(f"[Voice Audio Error] Native driver failed to play wake sound: {sound_err}")
                            else:
                                print(f"[Voice Audio Warning] Could not locate wake asset file: {audio_asset}")

                            print("[Voice] Actively recording your prompt...")
                            self.current_state = "RECORDING"
                        threading.Thread(target=play_engine_wake_worker, daemon=True).start()
                        if stream.read_available > 0:
                            stream.read(stream.read_available)
                        if hasattr(self.oww_model, "reset"):
                            self.oww_model.reset()
                        recorded_chunks = []
                        silent_chunk_counter = 0
                        has_started_talking = False
                    continue

                if self.current_state in ["RECORDING", "CONFIRMATION"]:
                    recorded_chunks.append(audio_chunk)
                    if not has_started_talking:
                        if volume_score > silence_threshold:
                            has_started_talking = True
                            initial_wait_counter = 0
                        else:
                            initial_wait_counter += 1
                        if initial_wait_counter >= initial_wait_limit:
                            silent_chunk_counter = 999
                    else:
                        if volume_score < silence_threshold:
                            silent_chunk_counter += 1
                        else:
                            silent_chunk_counter = 0
                    if silent_chunk_counter >= (1 if silent_chunk_counter == 999 else speech_pause_limit) or len(recorded_chunks) >= 250:
                            current_chunks = list(recorded_chunks)
                            current_talked = has_started_talking
                            recorded_chunks = []
                            silent_chunk_counter = 0
                            has_started_talking = False

                            self._process_captured_audio(current_chunks, current_talked, fs, stream)

            except Exception as e:
                time.sleep(0.05)
