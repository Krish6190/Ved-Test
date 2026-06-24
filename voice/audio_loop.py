import os
import time
import threading
import numpy as np
import sounddevice as sd
import winsound

def _unified_audio_loop(self):
    """Single microphone loop to avoid hardware streaming clashes."""
    fs = 16000
    chunk_size = 1280  
    
    recorded_chunks = []
    silence_threshold = 150      
    speech_pause_limit = 6 if self.current_state == "CONFIRMATION" else 12  # Fast trailing cut-off (0.5s - 1s)
    initial_wait_limit = 50  
    silent_chunk_counter = 0
    has_started_talking = False

    print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=chunk_size) as stream:
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
                    if wake_score > 0.45:
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
